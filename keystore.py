"""
Versioned Keystore Module for Cheating Surveillance System.

Manages:
- Per-installation Data Encryption Key (DEK) generation and persistence.
- Dual-wrapped key protection: Windows DPAPI User Scope and offline Master Recovery Key (MRK).
- Persistent per-key encryption usage accounting and cross-process reservation locks (resolves SEC-28-1-01).
- Strict, fail-closed filesystem access controls (NTFS ACLs / POSIX permissions) (resolves SEC-28-1-02).
- Canonical path safety, symlink/junction rejection, and atomic replacement.
- Key rotation while preserving retired wrapped keys for historical evidence decryption.
- Fail-closed validation for corrupted or invalid keystores.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from filelock import FileLock

from crypto_core import (
    AES_KEY_SIZE,
    MAX_ENCRYPTIONS_PER_KEY,
    DPAPIError,
    DPAPIKeyWrapper,
    InvalidKeyError,
    KeyLimitReachedError,
    KeyManagementError,
    RecoveryError,
    SecurityConfigurationError,
    UnsupportedPlatformError,
    generate_dek,
    generate_mrk,
    parse_and_verify_mrk,
    unwrap_dek_with_mrk,
    wrap_dek_with_mrk,
)
from path_security import is_symlink_or_junction

KEYSTORE_SCHEMA_VERSION: int = 1
DEFAULT_KEYSTORE_FILENAME: str = "keystore.enc"


def apply_restrictive_permissions(filepath: Path, fail_closed: bool = True) -> bool:
    """
    Applies restrictive access controls to a file, verifying enforcement.

    Resolves SEC-28-1-02:
    - On Windows: Uses icacls to strip inherited permissions and grant Full Control
      exclusively to the current user and SYSTEM. Verifies exit code and execution output.
    - On POSIX: Uses chmod 0600 (owner read/write only).

    Args:
        filepath: Target file path to secure.
        fail_closed: If True, raises SecurityConfigurationError on permission failure.
                     If False, returns False on failure.

    Returns:
        bool: True if permissions were verified applied.

    Raises:
        SecurityConfigurationError: If fail_closed=True and permission enforcement fails.
    """
    if sys.platform == "win32":
        try:
            import subprocess
            username = os.environ.get("USERNAME")
            if not username:
                if fail_closed:
                    raise SecurityConfigurationError(
                        f"Cannot apply Windows ACLs to '{filepath}': USERNAME environment variable is missing."
                    )
                return False

            cmd = [
                "icacls",
                str(filepath),
                "/inheritance:r",
                "/grant:r",
                f"{username}:(F)",
                "/grant:r",
                "*S-1-5-18:(F)",  # Local System account
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            success = (res.returncode == 0) and (
                "Successfully processed 1 files" in res.stdout or "processed file" in res.stdout
            )
            if not success:
                err_msg = (res.stderr or res.stdout or "icacls non-zero exit").strip()
                if fail_closed:
                    raise SecurityConfigurationError(
                        f"Failed to enforce restrictive ACLs on '{filepath}' via icacls (exit code {res.returncode}): {err_msg}"
                    )
                return False
            return True
        except Exception as exc:
            if fail_closed:
                if isinstance(exc, SecurityConfigurationError):
                    raise
                raise SecurityConfigurationError(f"Failed to enforce restrictive ACLs on '{filepath}': {exc}") from exc
            return False
    else:
        try:
            os.chmod(filepath, 0o600)
            return True
        except Exception as exc:
            if fail_closed:
                raise SecurityConfigurationError(f"Failed to apply chmod 0600 permissions on '{filepath}': {exc}") from exc
            return False


@dataclass
class KeyEntry:
    """Represents a single versioned DEK entry wrapped by DPAPI and MRK."""
    key_id: str
    created_at: str
    status: str  # "active" or "retired"
    dpapi_wrapped_dek: bytes
    mrk_wrapped_dek: bytes
    mrk_salt: bytes
    encryption_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes key entry to dictionary with base64 encoded binary fields."""
        return {
            "key_id": self.key_id,
            "created_at": self.created_at,
            "status": self.status,
            "dpapi_wrapped_dek": base64.b64encode(self.dpapi_wrapped_dek).decode("ascii"),
            "mrk_wrapped_dek": base64.b64encode(self.mrk_wrapped_dek).decode("ascii"),
            "mrk_salt": base64.b64encode(self.mrk_salt).decode("ascii"),
            "encryption_count": self.encryption_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyEntry:
        """Deserializes and validates a key entry from a dictionary."""
        required = {"key_id", "created_at", "status", "dpapi_wrapped_dek", "mrk_wrapped_dek", "mrk_salt"}
        missing = required - set(data.keys())
        if missing:
            raise KeyManagementError(f"Malformed key entry in keystore: missing fields {missing}.")

        try:
            dpapi_bytes = base64.b64decode(data["dpapi_wrapped_dek"])
            mrk_bytes = base64.b64decode(data["mrk_wrapped_dek"])
            salt_bytes = base64.b64decode(data["mrk_salt"])
        except Exception as exc:
            raise KeyManagementError(f"Corrupted base64 payload in key entry: {exc}") from exc

        return cls(
            key_id=str(data["key_id"]),
            created_at=str(data["created_at"]),
            status=str(data["status"]),
            dpapi_wrapped_dek=dpapi_bytes,
            mrk_wrapped_dek=mrk_bytes,
            mrk_salt=salt_bytes,
            encryption_count=int(data.get("encryption_count", 0)),
        )


class Keystore:
    """
    Thread-safe and cross-process safe versioned keystore holding DPAPI and MRK dual-wrapped DEKs.

    Guarantees:
    - Never stores plaintext DEK or MRK on disk.
    - All disk writes are atomic using temporary file replacement and fsync.
    - Rejects path traversal, symlinks, and directory junctions.
    - Fails closed on any corrupted or unreadable keystore format.
    - Preserves historical wrapped keys during key rotation.
    - Durable cross-process encryption usage accounting (SEC-28-1-01).
    - Strict fail-closed ACL verification on file persistence (SEC-28-1-02).
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        active_key_id: str,
        keys: Dict[str, KeyEntry],
        version: int = KEYSTORE_SCHEMA_VERSION,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.filepath = Path(filepath).resolve()
        self.version = version
        self.active_key_id = active_key_id
        self.keys = keys
        now_iso = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now_iso
        self.updated_at = updated_at or now_iso

        if self.active_key_id not in self.keys:
            raise KeyManagementError(f"Active key ID '{self.active_key_id}' not found in keystore keys.")

    @property
    def key_count(self) -> int:
        """Returns total number of keys in the keystore."""
        return len(self.keys)

    def list_key_ids(self) -> List[str]:
        """Returns a list of all key IDs present in the keystore."""
        return list(self.keys.keys())

    def get_key_entry(self, key_id: Optional[str] = None) -> KeyEntry:
        """Returns key entry for specified key_id or the active key if None."""
        target_id = key_id or self.active_key_id
        if target_id not in self.keys:
            raise KeyManagementError(f"Key ID '{target_id}' does not exist in keystore.")
        return self.keys[target_id]

    @classmethod
    def initialize(
        cls,
        filepath: Union[str, Path],
        mrk_input: Optional[Union[str, bytes]] = None,
        dpapi_wrapper: Optional[DPAPIKeyWrapper] = None,
        fail_closed_acls: bool = True,
    ) -> Tuple[Keystore, bytes, str]:
        """
        Initializes a brand-new keystore file with a freshly generated DEK.

        Security Safeguards:
        - Fails closed if the keystore file already exists (never overwrites).
        - Verifies directory safety and rejects symlinks or junctions.
        - Generates 256-bit DEK, wraps via DPAPI User Scope.
        - Generates 256-bit MRK (or verifies provided MRK) and wraps DEK via RFC 3394 AES Key Wrap.
        - Writes atomically and sets restrictive file permissions fail-closed.

        Returns:
            Tuple[Keystore, bytes, str]: (initialized_keystore, raw_mrk [32 bytes], formatted_mrk [string])

        Raises:
            KeyManagementError: If file exists, directory is insecure, or ACL enforcement fails.
        """
        target = Path(filepath).resolve()
        if target.exists():
            raise KeyManagementError(
                f"Keystore file already exists at '{target}'. Initialization cannot overwrite an existing keystore."
            )

        if is_symlink_or_junction(target):
            raise KeyManagementError(f"Keystore path '{target}' cannot be a symlink or junction.")

        parent_dir = target.parent
        if is_symlink_or_junction(parent_dir):
            raise KeyManagementError(f"Keystore parent directory '{parent_dir}' cannot be a symlink or junction.")

        parent_dir.mkdir(parents=True, exist_ok=True)

        # Generate new 256-bit Data Encryption Key
        dek = generate_dek()
        key_id = str(uuid.uuid4())

        # Wrap with DPAPI User Scope
        if dpapi_wrapper is None and sys.platform == "win32":
            dpapi_wrapper = DPAPIKeyWrapper()

        if dpapi_wrapper is not None:
            dpapi_wrapped_dek = dpapi_wrapper.protect(dek, description=f"CSS_DEK_{key_id}")
        else:
            raise UnsupportedPlatformError("Cannot initialize keystore without a valid DPAPI wrapper on Windows.")

        # Generate or parse MRK
        if mrk_input is None:
            raw_mrk, formatted_mrk = generate_mrk()
        else:
            raw_mrk = parse_and_verify_mrk(mrk_input)
            checksum = hashlib.sha256(b"CSS_MRK_CHECKSUM_v1:" + raw_mrk).digest()[:8]
            b32 = base64.b32encode(raw_mrk + checksum).decode("ascii")
            groups = [b32[i:i + 4] for i in range(0, len(b32), 4)]
            formatted_mrk = "CSS-MRK-" + "-".join(groups)

        # Wrap with MRK
        mrk_wrapped_dek, mrk_salt = wrap_dek_with_mrk(dek, raw_mrk)

        now_iso = datetime.now(timezone.utc).isoformat()
        entry = KeyEntry(
            key_id=key_id,
            created_at=now_iso,
            status="active",
            dpapi_wrapped_dek=dpapi_wrapped_dek,
            mrk_wrapped_dek=mrk_wrapped_dek,
            mrk_salt=mrk_salt,
            encryption_count=0,
        )

        keystore = cls(
            filepath=target,
            active_key_id=key_id,
            keys={key_id: entry},
            version=KEYSTORE_SCHEMA_VERSION,
            created_at=now_iso,
            updated_at=now_iso,
        )

        keystore.save(fail_closed_acls=fail_closed_acls)
        return keystore, raw_mrk, formatted_mrk

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Keystore:
        """
        Loads and validates an existing keystore from disk.

        Security Safeguards:
        - Rejects symlinks or junctions.
        - Fails closed on malformed JSON, unsupported schema versions, or missing fields.

        Raises:
            FileNotFoundError: If keystore file does not exist.
            KeyManagementError: If keystore is corrupted or invalid.
        """
        target = Path(filepath).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Keystore file not found at '{target}'.")

        if is_symlink_or_junction(target):
            raise KeyManagementError(f"Keystore path '{target}' cannot be a symlink or directory junction.")

        try:
            content = target.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as exc:
            raise KeyManagementError(f"Failed to read or parse keystore at '{target}': {exc}") from exc

        version = data.get("version")
        if version != KEYSTORE_SCHEMA_VERSION:
            raise KeyManagementError(
                f"Unsupported keystore schema version: {version}. Expected version {KEYSTORE_SCHEMA_VERSION}."
            )

        active_key_id = data.get("active_key_id")
        if not active_key_id:
            raise KeyManagementError("Keystore is missing 'active_key_id'.")

        raw_keys = data.get("keys", {})
        if not isinstance(raw_keys, dict) or not raw_keys:
            raise KeyManagementError("Keystore contains no valid keys dictionary.")

        keys: Dict[str, KeyEntry] = {}
        for kid, entry_dict in raw_keys.items():
            keys[kid] = KeyEntry.from_dict(entry_dict)

        return cls(
            filepath=target,
            active_key_id=str(active_key_id),
            keys=keys,
            version=int(version),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def save(self, fail_closed_acls: bool = True) -> None:
        """
        Atomically persists the keystore to disk using a temporary file and os.replace.

        Resolves SEC-28-1-02:
        - Ensures data is flushed and synced to disk before renaming.
        - Enforces restrictive file permissions on temporary file before replacement.
        - If ACL application fails and fail_closed_acls=True, cleans up temp file,
          preserves any existing keystore, and raises SecurityConfigurationError.
        """
        if is_symlink_or_junction(self.filepath):
            raise KeyManagementError(f"Cannot save: target path '{self.filepath}' is a symlink or junction.")

        parent_dir = self.filepath.parent
        if is_symlink_or_junction(parent_dir):
            raise KeyManagementError(f"Cannot save: parent directory '{parent_dir}' is a symlink or junction.")

        parent_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self.version,
            "active_key_id": self.active_key_id,
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "keys": {k: v.to_dict() for k, v in self.keys.items()},
        }

        serialized = json.dumps(data, indent=2)
        tmp_target = self.filepath.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")

        try:
            with open(tmp_target, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())

            # Apply restrictive permissions prior to atomic replace (SEC-28-1-02)
            apply_restrictive_permissions(tmp_target, fail_closed=fail_closed_acls)

            # Atomic rename replaces existing target safely
            os.replace(tmp_target, self.filepath)
            apply_restrictive_permissions(self.filepath, fail_closed=fail_closed_acls)
        except Exception as exc:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except OSError:
                    pass
            if isinstance(exc, (KeyManagementError, SecurityConfigurationError)):
                raise
            raise KeyManagementError(f"Failed to atomically persist keystore to '{self.filepath}': {exc}") from exc

    def reserve_encryption(self, key_id: Optional[str] = None, max_limit: Optional[int] = None) -> int:
        """
        Durable, cross-process reservation of an encryption operation under the specified DEK.

        Resolves SEC-28-1-01:
        - Acquires a cross-process file lock (<keystore_path>.lock).
        - Reloads latest persisted state from disk under the lock to prevent lost updates.
        - Checks whether encryption_count has reached max_limit (default MAX_ENCRYPTIONS_PER_KEY).
        - If limit reached, raises KeyLimitReachedError.
        - Increments encryption_count and atomically saves to disk before encryption occurs.
        - Releases lock and returns the reserved count.
        - If encryption subsequently fails, the reserved count remains consumed (never decremented).

        Returns:
            int: The reserved encryption sequence number.

        Raises:
            KeyLimitReachedError: If the per-key encryption limit has been reached.
            KeyManagementError: On locking, reload, or persistence errors.
        """
        target_id = key_id or self.active_key_id
        limit = max_limit if max_limit is not None else MAX_ENCRYPTIONS_PER_KEY
        lock_path = self.filepath.with_suffix(".lock")

        try:
            with FileLock(str(lock_path), timeout=15.0):
                if self.filepath.exists():
                    reloaded = Keystore.load(self.filepath)
                    self.keys = reloaded.keys
                    self.active_key_id = reloaded.active_key_id
                    self.version = reloaded.version
                    self.updated_at = reloaded.updated_at

                entry = self.get_key_entry(target_id)
                if entry.encryption_count >= limit:
                    raise KeyLimitReachedError(
                        f"DEK Key ID '{target_id}' has reached the maximum permitted encryption limit "
                        f"({limit} operations). Key rotation is required before further encryption."
                    )

                entry.encryption_count += 1
                reserved_seq = entry.encryption_count
                self.save(fail_closed_acls=True)
                return reserved_seq
        except FileNotFoundError:
            raise KeyManagementError(f"Cannot reserve encryption: keystore file '{self.filepath}' not found.")
        except (KeyLimitReachedError, KeyManagementError, SecurityConfigurationError):
            raise
        except Exception as exc:
            raise KeyManagementError(f"Cross-process lock reservation failure: {exc}") from exc

    def unwrap_dek_via_dpapi(
        self,
        key_id: Optional[str] = None,
        dpapi_wrapper: Optional[DPAPIKeyWrapper] = None,
    ) -> bytes:
        """
        Unwraps the DEK using Windows DPAPI in User Scope.

        Args:
            key_id: Key ID to unwrap (defaults to active_key_id).
            dpapi_wrapper: Optional DPAPIKeyWrapper instance (auto-instantiated if None).

        Returns:
            bytes: Unwrapped 32-byte DEK.

        Raises:
            DPAPIError: If unprotection fails (e.g. wrong user profile or corrupted DPAPI blob).
            KeyManagementError: If key_id is not found.
        """
        entry = self.get_key_entry(key_id)
        if dpapi_wrapper is None:
            dpapi_wrapper = DPAPIKeyWrapper()

        dek = dpapi_wrapper.unprotect(entry.dpapi_wrapped_dek)
        if len(dek) != AES_KEY_SIZE:
            raise KeyManagementError(
                f"DPAPI unwrapped DEK has invalid length: expected {AES_KEY_SIZE} bytes, got {len(dek)} bytes."
            )
        return dek

    def recover_dek_via_mrk(
        self,
        mrk_input: Union[str, bytes],
        key_id: Optional[str] = None,
    ) -> bytes:
        """
        Recovers the DEK using the offline Master Recovery Key (MRK).

        Args:
            mrk_input: Formatted string or 32-byte raw MRK key.
            key_id: Key ID to recover (defaults to active_key_id).

        Returns:
            bytes: Recovered 32-byte DEK.

        Raises:
            RecoveryError: If MRK checksum is invalid or key unwrapping fails.
        """
        raw_mrk = parse_and_verify_mrk(mrk_input)
        entry = self.get_key_entry(key_id)

        dek = unwrap_dek_with_mrk(
            wrapped_dek=entry.mrk_wrapped_dek,
            raw_mrk=raw_mrk,
            salt=entry.mrk_salt,
        )
        return dek

    def rotate_key(
        self,
        mrk_input: Optional[Union[str, bytes]] = None,
        dpapi_wrapper: Optional[DPAPIKeyWrapper] = None,
        fail_closed_acls: bool = True,
    ) -> Tuple[str, bytes, str]:
        """
        Rotates the active encryption key by generating a new DEK and retiring the current key.

        Security Controls:
        - Marks current active key as 'retired' (preserving it for decrypting existing evidence).
        - Generates new 256-bit DEK, wraps with DPAPI User Scope.
        - Generates new 256-bit MRK (or accepts new MRK input) and wraps new DEK.
        - Initializes encryption_count to 0 for the new key.
        - Sets new key as 'active' and atomically persists the updated keystore.

        Returns:
            Tuple[str, bytes, str]: (new_key_id, new_raw_mrk, new_formatted_mrk)
        """
        # Retire current active key
        current_entry = self.get_key_entry(self.active_key_id)
        current_entry.status = "retired"

        new_key_id = str(uuid.uuid4())
        new_dek = generate_dek()

        if dpapi_wrapper is None and sys.platform == "win32":
            dpapi_wrapper = DPAPIKeyWrapper()

        if dpapi_wrapper is not None:
            new_dpapi_wrapped_dek = dpapi_wrapper.protect(new_dek, description=f"CSS_DEK_{new_key_id}")
        else:
            raise UnsupportedPlatformError("Cannot rotate key without a valid DPAPI wrapper on Windows.")

        if mrk_input is None:
            new_raw_mrk, new_formatted_mrk = generate_mrk()
        else:
            new_raw_mrk = parse_and_verify_mrk(mrk_input)
            checksum = hashlib.sha256(b"CSS_MRK_CHECKSUM_v1:" + new_raw_mrk).digest()[:8]
            b32 = base64.b32encode(new_raw_mrk + checksum).decode("ascii")
            groups = [b32[i:i + 4] for i in range(0, len(b32), 4)]
            new_formatted_mrk = "CSS-MRK-" + "-".join(groups)

        new_mrk_wrapped_dek, new_mrk_salt = wrap_dek_with_mrk(new_dek, new_raw_mrk)

        now_iso = datetime.now(timezone.utc).isoformat()
        new_entry = KeyEntry(
            key_id=new_key_id,
            created_at=now_iso,
            status="active",
            dpapi_wrapped_dek=new_dpapi_wrapped_dek,
            mrk_wrapped_dek=new_mrk_wrapped_dek,
            mrk_salt=new_mrk_salt,
            encryption_count=0,
        )

        self.keys[new_key_id] = new_entry
        self.active_key_id = new_key_id
        self.save(fail_closed_acls=fail_closed_acls)

        return new_key_id, new_raw_mrk, new_formatted_mrk
