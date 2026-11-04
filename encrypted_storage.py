"""
Encrypted Evidence Storage Core & Binary Envelope I/O Module.

Provides:
- Deterministic, versioned binary evidence envelope serializer and parser.
- AES-256-GCM authenticated file encryption and decryption.
- Canonical metadata binding and original filename authentication (AAD).
- Unencrypted authenticated metadata extraction for retention lifecycle auditing.
- Persistent per-key encryption usage accounting integration (resolves SEC-28-1-01).
- Fail-closed filesystem access controls (NTFS ACLs / POSIX permissions) (resolves SEC-28-1-02).
- Atomic encrypted file persistence (.tmp -> flush -> fsync -> verify -> os.replace).
- Historical DEK lookup via envelope-embedded Key ID.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from crypto_core import (
    AES_KEY_SIZE,
    GCM_NONCE_SIZE,
    GCM_TAG_SIZE,
    AESGCMCipher,
    AuthenticationError,
    CryptoError,
    DecryptionError,
    EncryptionError,
    InvalidKeyError,
    InvalidNonceError,
    KeyLimitReachedError,
    KeyManagementError,
    RecoveryError,
    SecurityConfigurationError,
    generate_nonce,
)
from keystore import Keystore, apply_restrictive_permissions
from path_security import is_symlink_or_junction

# Binary Envelope Constants
ENVELOPE_MAGIC: bytes = b"CSS_ENC\x01"  # 8 bytes magic identifier
ENVELOPE_VERSION: int = 1               # uint16 big-endian
FIXED_HEADER_FORMAT: str = ">8sHH16s12sIQ"
FIXED_HEADER_SIZE: int = struct.calcsize(FIXED_HEADER_FORMAT)  # exactly 52 bytes

# Strict Envelope Security Limits
MAX_HEADER_SIZE: int = 1024
MAX_METADATA_SIZE: int = 64 * 1024       # 64 KB
MAX_FILENAME_LENGTH: int = 255           # 255 characters
MAX_SESSION_ID_LENGTH: int = 128         # 128 characters
MAX_CIPHERTEXT_SIZE: int = 100 * 1024 * 1024  # 100 MB


# ==============================================================================
# Storage Exceptions
# ==============================================================================

class EncryptedStorageError(CryptoError):
    """Base exception for all encrypted storage and envelope errors."""
    pass


class InvalidEnvelopeError(EncryptedStorageError):
    """Raised when an envelope has invalid magic bytes, truncated size, or corrupt format."""
    pass


class UnsupportedVersionError(EncryptedStorageError):
    """Raised when an envelope has an unsupported version number."""
    pass


class FilenameMismatchError(EncryptedStorageError):
    """Raised when an encrypted file's authenticated filename does not match expected identity."""
    pass


class AtomicWriteError(EncryptedStorageError):
    """Raised when atomic encrypted file persistence or replacement fails."""
    pass


# ==============================================================================
# Authenticated Evidence Metadata
# ==============================================================================

@dataclass(frozen=True)
class EvidenceMetadata:
    """
    Canonical metadata bound to an evidence envelope via Associated Authenticated Data.
    """
    evidence_id: str
    original_filename: str
    evidence_type: str
    created_at: str
    created_timestamp: float
    session_id: Optional[str] = None
    format_version: int = 1
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validates bounds and format of metadata fields."""
        if not self.evidence_id or not isinstance(self.evidence_id, str):
            raise InvalidEnvelopeError("Metadata 'evidence_id' must be a non-empty string.")
        if not self.original_filename or not isinstance(self.original_filename, str):
            raise InvalidEnvelopeError("Metadata 'original_filename' must be a non-empty string.")
        if len(self.original_filename) > MAX_FILENAME_LENGTH:
            raise InvalidEnvelopeError(
                f"Metadata 'original_filename' exceeds limit of {MAX_FILENAME_LENGTH} characters."
            )
        if self.session_id is not None and len(self.session_id) > MAX_SESSION_ID_LENGTH:
            raise InvalidEnvelopeError(
                f"Metadata 'session_id' exceeds limit of {MAX_SESSION_ID_LENGTH} characters."
            )
        if not self.created_at or not isinstance(self.created_at, str):
            raise InvalidEnvelopeError("Metadata 'created_at' must be an ISO 8601 string.")
        if not isinstance(self.created_timestamp, (int, float)):
            raise InvalidEnvelopeError("Metadata 'created_timestamp' must be a numeric timestamp.")

    def to_canonical_bytes(self) -> bytes:
        """
        Serializes metadata into deterministic canonical UTF-8 JSON.
        Guarantees sorted keys and standard delimiters for bit-exact AAD hashing.
        """
        self.validate()
        data = {
            "created_at": self.created_at,
            "created_timestamp": float(self.created_timestamp),
            "custom_attributes": self.custom_attributes,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "format_version": int(self.format_version),
            "original_filename": self.original_filename,
            "session_id": self.session_id,
        }
        raw_json = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        encoded = raw_json.encode("utf-8")
        if len(encoded) > MAX_METADATA_SIZE:
            raise InvalidEnvelopeError(
                f"Canonical metadata size ({len(encoded)} bytes) exceeds limit of {MAX_METADATA_SIZE} bytes."
            )
        return encoded

    @classmethod
    def from_canonical_bytes(cls, raw_bytes: bytes) -> EvidenceMetadata:
        """Deserializes metadata from canonical JSON bytes."""
        if len(raw_bytes) > MAX_METADATA_SIZE:
            raise InvalidEnvelopeError(
                f"Metadata payload size ({len(raw_bytes)} bytes) exceeds limit of {MAX_METADATA_SIZE} bytes."
            )
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise InvalidEnvelopeError(f"Malformed metadata JSON payload: {exc}") from exc

        meta = cls(
            evidence_id=str(data.get("evidence_id", "")),
            original_filename=str(data.get("original_filename", "")),
            evidence_type=str(data.get("evidence_type", "")),
            created_at=str(data.get("created_at", "")),
            created_timestamp=float(data.get("created_timestamp", 0.0)),
            session_id=data.get("session_id"),
            format_version=int(data.get("format_version", 1)),
            custom_attributes=data.get("custom_attributes", {}),
        )
        meta.validate()
        return meta


@dataclass(frozen=True)
class EnvelopeHeader:
    """Fixed-size binary envelope header fields."""
    magic: bytes
    version: int
    header_length: int
    key_id_bytes: bytes
    nonce: bytes
    metadata_length: int
    ciphertext_length: int

    @property
    def key_id_str(self) -> str:
        """Returns standard hyphenated UUID string for Key ID."""
        return str(uuid.UUID(bytes=self.key_id_bytes))


# ==============================================================================
# Binary Envelope Packing & Parsing
# ==============================================================================

def serialize_envelope(
    key_id_bytes: bytes,
    nonce: bytes,
    canonical_metadata_bytes: bytes,
    ciphertext_bytes: bytes,
    tag_bytes: bytes,
) -> bytes:
    """
    Serializes a complete binary evidence envelope according to specification:
    [52 bytes Fixed Header] + [M bytes Canonical Metadata] + [C bytes Ciphertext] + [16 bytes Tag]
    """
    if len(key_id_bytes) != 16:
        raise InvalidEnvelopeError(f"Key ID must be 16 bytes, got {len(key_id_bytes)}.")
    if len(nonce) != GCM_NONCE_SIZE:
        raise InvalidNonceError(f"Nonce must be {GCM_NONCE_SIZE} bytes, got {len(nonce)}.")
    if len(tag_bytes) != GCM_TAG_SIZE:
        raise InvalidEnvelopeError(f"Tag must be {GCM_TAG_SIZE} bytes, got {len(tag_bytes)}.")

    meta_len = len(canonical_metadata_bytes)
    if meta_len > MAX_METADATA_SIZE:
        raise InvalidEnvelopeError(f"Metadata length {meta_len} exceeds limit of {MAX_METADATA_SIZE}.")

    cipher_len = len(ciphertext_bytes)
    if cipher_len > MAX_CIPHERTEXT_SIZE:
        raise InvalidEnvelopeError(f"Ciphertext length {cipher_len} exceeds limit of {MAX_CIPHERTEXT_SIZE}.")

    fixed_header = struct.pack(
        FIXED_HEADER_FORMAT,
        ENVELOPE_MAGIC,
        ENVELOPE_VERSION,
        FIXED_HEADER_SIZE,
        key_id_bytes,
        nonce,
        meta_len,
        cipher_len,
    )

    return fixed_header + canonical_metadata_bytes + ciphertext_bytes + tag_bytes


def parse_envelope(
    envelope_bytes: bytes,
) -> Tuple[EnvelopeHeader, bytes, bytes, bytes, bytes]:
    """
    Parses and validates binary envelope structure.

    Returns:
        Tuple[EnvelopeHeader, canonical_metadata_bytes, ciphertext_bytes, tag_bytes, fixed_header_bytes]

    Raises:
        InvalidEnvelopeError: If magic is wrong, envelope is truncated, or lengths mismatch.
        UnsupportedVersionError: If envelope version is not supported.
    """
    if not isinstance(envelope_bytes, bytes):
        raise InvalidEnvelopeError("Envelope payload must be bytes.")

    if len(envelope_bytes) < (FIXED_HEADER_SIZE + GCM_TAG_SIZE):
        raise InvalidEnvelopeError(
            f"Envelope truncated: length {len(envelope_bytes)} is less than minimum possible envelope size."
        )

    fixed_header_bytes = envelope_bytes[:FIXED_HEADER_SIZE]
    magic, version, header_len, key_id_bytes, nonce, meta_len, cipher_len = struct.unpack(
        FIXED_HEADER_FORMAT,
        fixed_header_bytes,
    )

    if magic != ENVELOPE_MAGIC:
        raise InvalidEnvelopeError(f"Invalid envelope magic identifier: {magic!r}.")

    if version != ENVELOPE_VERSION:
        raise UnsupportedVersionError(
            f"Unsupported envelope version {version}. Current supported version is {ENVELOPE_VERSION}."
        )

    if header_len != FIXED_HEADER_SIZE:
        raise InvalidEnvelopeError(
            f"Header length mismatch: expected {FIXED_HEADER_SIZE}, got {header_len}."
        )

    if meta_len > MAX_METADATA_SIZE:
        raise InvalidEnvelopeError(f"Metadata length {meta_len} exceeds maximum allowed {MAX_METADATA_SIZE}.")

    if cipher_len > MAX_CIPHERTEXT_SIZE:
        raise InvalidEnvelopeError(f"Ciphertext length {cipher_len} exceeds maximum allowed {MAX_CIPHERTEXT_SIZE}.")

    expected_total_len = FIXED_HEADER_SIZE + meta_len + cipher_len + GCM_TAG_SIZE
    if len(envelope_bytes) != expected_total_len:
        raise InvalidEnvelopeError(
            f"Envelope length inconsistency: expected {expected_total_len} bytes from header lengths, "
            f"got {len(envelope_bytes)} bytes."
        )

    meta_start = FIXED_HEADER_SIZE
    meta_end = meta_start + meta_len
    cipher_start = meta_end
    cipher_end = cipher_start + cipher_len
    tag_start = cipher_end
    tag_end = tag_start + GCM_TAG_SIZE

    metadata_bytes = envelope_bytes[meta_start:meta_end]
    ciphertext_bytes = envelope_bytes[cipher_start:cipher_end]
    tag_bytes = envelope_bytes[tag_start:tag_end]

    header = EnvelopeHeader(
        magic=magic,
        version=version,
        header_length=header_len,
        key_id_bytes=key_id_bytes,
        nonce=nonce,
        metadata_length=meta_len,
        ciphertext_length=cipher_len,
    )

    return header, metadata_bytes, ciphertext_bytes, tag_bytes, fixed_header_bytes


def read_authenticated_metadata(envelope_bytes: bytes) -> EvidenceMetadata:
    """
    Extracts validated evidence metadata from an envelope without decrypting the payload.

    Permits retention management to inspect creation timestamps and evidence types
    without unprotecting encryption keys.
    """
    _, metadata_bytes, _, _, _ = parse_envelope(envelope_bytes)
    return EvidenceMetadata.from_canonical_bytes(metadata_bytes)


# ==============================================================================
# Encrypted Storage Manager
# ==============================================================================

class EncryptedStorageManager:
    """
    High-level manager for encrypted evidence persistence, authenticated metadata,
    and historical key resolution.
    """

    def __init__(
        self,
        keystore: Keystore,
        base_dir: Optional[Union[str, Path]] = None,
        fail_closed_acls: bool = True,
    ):
        self.keystore = keystore
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else self.keystore.filepath.parent
        self.fail_closed_acls = fail_closed_acls

    def encrypt_bytes(
        self,
        plaintext: bytes,
        metadata: EvidenceMetadata,
        key_id: Optional[str] = None,
        max_limit: Optional[int] = None,
    ) -> bytes:
        """
        Encrypts plaintext payload into a versioned binary evidence envelope.

        Security Safeguards:
        - Atomically reserves an encryption operation under cross-process locking (SEC-28-1-01).
        - Enforces per-key usage limit (raising KeyLimitReachedError if reached).
        - Embeds 128-bit Key ID UUID and 96-bit CSPRNG nonce into envelope.
        - Cryptographically binds fixed header and canonical metadata into AAD.
        - Appends 128-bit GCM authentication tag.

        Returns:
            bytes: Complete binary evidence envelope bytes.
        """
        if not isinstance(plaintext, bytes):
            raise EncryptionError("Plaintext payload must be bytes.")

        target_key_id = key_id or self.keystore.active_key_id

        # 1. Durable cross-process reservation of encryption usage (SEC-28-1-01)
        self.keystore.reserve_encryption(key_id=target_key_id, max_limit=max_limit)

        # 2. Resolve DEK via DPAPI
        dek = self.keystore.unwrap_dek_via_dpapi(key_id=target_key_id)
        if len(dek) != AES_KEY_SIZE:
            raise KeyManagementError(f"Resolved DEK has invalid length: {len(dek)} bytes.")

        # 3. Canonical metadata serialization
        canonical_meta = metadata.to_canonical_bytes()

        # 4. Generate 96-bit CSPRNG nonce
        nonce = generate_nonce()

        # 5. Build fixed header bytes
        key_id_bytes = uuid.UUID(target_key_id).bytes
        cipher_len = len(plaintext)
        fixed_header = struct.pack(
            FIXED_HEADER_FORMAT,
            ENVELOPE_MAGIC,
            ENVELOPE_VERSION,
            FIXED_HEADER_SIZE,
            key_id_bytes,
            nonce,
            len(canonical_meta),
            cipher_len,
        )

        # 6. AAD binds the entire fixed header and canonical metadata bytes
        aad = fixed_header + canonical_meta

        # 7. Encrypt payload with AES-256-GCM
        cipher = AESGCMCipher(dek)
        _, ciphertext_with_tag = cipher.encrypt(plaintext, nonce=nonce, associated_data=aad)

        # Split ciphertext and tag
        ciphertext_body = ciphertext_with_tag[:-GCM_TAG_SIZE]
        tag_body = ciphertext_with_tag[-GCM_TAG_SIZE:]

        # 8. Assemble complete binary envelope
        return serialize_envelope(
            key_id_bytes=key_id_bytes,
            nonce=nonce,
            canonical_metadata_bytes=canonical_meta,
            ciphertext_bytes=ciphertext_body,
            tag_bytes=tag_body,
        )

    def decrypt_bytes(
        self,
        envelope_bytes: bytes,
        expected_filename: Optional[str] = None,
        mrk_input: Optional[Union[str, bytes]] = None,
    ) -> Tuple[bytes, EvidenceMetadata]:
        """
        Decrypts a binary evidence envelope, verifying AEAD authentication tag, AAD, and metadata.

        Security Safeguards:
        - Resolves exact historical DEK via embedded Key ID.
        - Reconstructs bit-exact AAD from fixed header and metadata bytes.
        - Enforces original filename validation where expected_filename is provided.
        - Fails closed on any corruption, tampering, or key mismatch.
        - Releases plaintext strictly in memory; zero disk writes.

        Returns:
            Tuple[bytes, EvidenceMetadata]: (decrypted_plaintext, verified_metadata)
        """
        header, meta_bytes, cipher_bytes, tag_bytes, fixed_header = parse_envelope(envelope_bytes)
        metadata = EvidenceMetadata.from_canonical_bytes(meta_bytes)

        # Validate expected filename binding
        if expected_filename is not None:
            clean_expected = str(expected_filename).strip()
            if clean_expected != metadata.original_filename:
                raise FilenameMismatchError(
                    f"Authenticated filename mismatch: expected '{clean_expected}', "
                    f"envelope authenticated identity is '{metadata.original_filename}'."
                )

        key_id_str = header.key_id_str

        # Resolve DEK: via MRK if supplied (disaster recovery), else via DPAPI
        if mrk_input is not None:
            dek = self.keystore.recover_dek_via_mrk(mrk_input=mrk_input, key_id=key_id_str)
        else:
            dek = self.keystore.unwrap_dek_via_dpapi(key_id=key_id_str)

        # Reconstruct AAD
        aad = fixed_header + meta_bytes
        ciphertext_with_tag = cipher_bytes + tag_bytes

        cipher = AESGCMCipher(dek)
        plaintext = cipher.decrypt(header.nonce, ciphertext_with_tag, associated_data=aad)

        return plaintext, metadata

    def write_encrypted_file(
        self,
        destination_path: Union[str, Path],
        plaintext: bytes,
        metadata: EvidenceMetadata,
        allow_overwrite: bool = False,
        key_id: Optional[str] = None,
        max_limit: Optional[int] = None,
    ) -> Path:
        """
        Safely writes an encrypted evidence file to disk atomically.

        Security Controls:
        - Canonical containment and reparse-point rejection.
        - Refuses silent overwrites unless allow_overwrite=True.
        - Writes to temporary file (.tmp_<uuid>), flushes, and fsyncs.
        - Applies restrictive ACLs to temporary file before replacement (SEC-28-1-02).
        - Structurally verifies written temporary file before replacement.
        - Replaces destination atomically via os.replace.
        - Preserves existing files on failure and cleans up temporary artifacts.

        Returns:
            Path: Confirmed destination path of encrypted file.
        """
        target = Path(destination_path).resolve()

        if is_symlink_or_junction(target):
            raise AtomicWriteError(f"Cannot write evidence: destination '{target}' is a symlink or junction.")

        parent_dir = target.parent
        if is_symlink_or_junction(parent_dir):
            raise AtomicWriteError(f"Cannot write evidence: parent directory '{parent_dir}' is a symlink or junction.")

        parent_dir.mkdir(parents=True, exist_ok=True)

        if target.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Destination evidence file already exists at '{target}' and allow_overwrite is False."
            )

        envelope_data = self.encrypt_bytes(
            plaintext=plaintext,
            metadata=metadata,
            key_id=key_id,
            max_limit=max_limit,
        )

        tmp_file = parent_dir / f"{target.name}.tmp_{uuid.uuid4().hex[:8]}"

        try:
            with open(tmp_file, "wb") as f:
                f.write(envelope_data)
                f.flush()
                os.fsync(f.fileno())

            # Apply restrictive permissions prior to atomic replacement (SEC-28-1-02)
            apply_restrictive_permissions(tmp_file, fail_closed=self.fail_closed_acls)

            # Structural sanity check of temporary file
            with open(tmp_file, "rb") as f:
                verified_head = f.read(FIXED_HEADER_SIZE)
                if len(verified_head) != FIXED_HEADER_SIZE or not verified_head.startswith(ENVELOPE_MAGIC):
                    raise AtomicWriteError("Temporary encrypted evidence envelope failed structural verification.")

            # Atomic rename replaces existing target safely
            os.replace(tmp_file, target)
            apply_restrictive_permissions(target, fail_closed=self.fail_closed_acls)
            return target
        except Exception as exc:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            if isinstance(exc, (AtomicWriteError, KeyLimitReachedError, SecurityConfigurationError)):
                raise
            raise AtomicWriteError(f"Failed to atomically write encrypted evidence to '{target}': {exc}") from exc

    def read_encrypted_file(
        self,
        filepath: Union[str, Path],
        expected_filename: Optional[str] = None,
        mrk_input: Optional[Union[str, bytes]] = None,
    ) -> Tuple[bytes, EvidenceMetadata]:
        """
        Reads and decrypts an encrypted evidence file from disk.

        Returns:
            Tuple[bytes, EvidenceMetadata]: (decrypted_plaintext, verified_metadata)
        """
        target = Path(filepath).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Encrypted evidence file not found at '{target}'.")

        if is_symlink_or_junction(target):
            raise EncryptedStorageError(f"Evidence file '{target}' cannot be a symlink or directory junction.")

        try:
            envelope_bytes = target.read_bytes()
        except Exception as exc:
            raise EncryptedStorageError(f"Failed to read encrypted evidence file at '{target}': {exc}") from exc

        if expected_filename is None:
            expected_name = target.name[:-4] if target.name.endswith(".enc") else target.name
        else:
            expected_name = expected_filename

        return self.decrypt_bytes(
            envelope_bytes=envelope_bytes,
            expected_filename=expected_name,
            mrk_input=mrk_input,
        )

    def read_metadata_from_file(self, filepath: Union[str, Path]) -> EvidenceMetadata:
        """
        Reads and returns unencrypted authenticated metadata from an evidence file without decrypting payload.
        """
        target = Path(filepath).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Evidence file not found at '{target}'.")

        if is_symlink_or_junction(target):
            raise EncryptedStorageError(f"Evidence file '{target}' cannot be a symlink or directory junction.")

        try:
            envelope_bytes = target.read_bytes()
        except Exception as exc:
            raise EncryptedStorageError(f"Failed to read file at '{target}': {exc}") from exc

        return read_authenticated_metadata(envelope_bytes)


class EncryptedCSVWriter:
    """
    Appender and writer for authenticated encrypted CSV segments.
    Resolves Section 3.3 of Phase 10.13.28.3:
    - Buffers records in memory.
    - Serializes to canonical CSV without writing plaintext to disk.
    - Flushes segments atomically into binary envelopes via EncryptedStorageManager.
    - Preserves sequence ordering via segment indexing: <stem>_<session_id>_seg<idx:05d><suffix>.enc.
    - Independently authenticates each segment with AES-256-GCM AEAD.
    - Records segment_index, record_count, and is_final in authenticated metadata.
    """

    def __init__(
        self,
        storage_manager: EncryptedStorageManager,
        log_dir: Union[str, Path],
        base_filename: str,
        session_id: str,
        fieldnames: List[str],
        chunk_size: int = 50,
        evidence_type: str = "EVENT_LOG",
    ):
        self.storage_manager = storage_manager
        self.log_dir = Path(log_dir).resolve()
        self.base_filename = base_filename
        self.session_id = session_id
        self.fieldnames = list(fieldnames)
        self.chunk_size = max(1, int(chunk_size))
        self.evidence_type = evidence_type

        p = Path(base_filename)
        self.stem = p.stem
        self.suffix = p.suffix or ".csv"

        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._segment_index: int = 0
        self._is_closed: bool = False

        if self.log_dir.exists():
            pattern = f"{self.stem}_{self.session_id}_seg*.enc"
            existing = list(self.log_dir.glob(pattern))
            if existing:
                indices = []
                for f in existing:
                    m = re.search(r"_seg(\d+)\.", f.name)
                    if m:
                        try:
                            indices.append(int(m.group(1)))
                        except ValueError:
                            pass
                if indices:
                    self._segment_index = max(indices) + 1

    def write_record(self, record: Dict[str, Any]) -> bool:
        """Buffers a single CSV record and flushes if chunk_size is reached."""
        with self._lock:
            if self._is_closed:
                raise RuntimeError("Cannot write to closed EncryptedCSVWriter.")
            self._buffer.append(dict(record))
            if len(self._buffer) >= self.chunk_size:
                self._flush_internal(is_final=False)
            return True

    def flush(self, is_final: bool = False) -> Optional[Path]:
        """Manually flushes buffered records to an encrypted segment file."""
        with self._lock:
            return self._flush_internal(is_final=is_final)

    def _flush_internal(self, is_final: bool = False) -> Optional[Path]:
        if not self._buffer and not is_final:
            return None

        # Build in-memory CSV
        sio = io.StringIO()
        writer = csv.DictWriter(sio, fieldnames=self.fieldnames)
        writer.writeheader()
        for rec in self._buffer:
            writer.writerow(rec)

        csv_text = sio.getvalue()
        sio.close()
        raw_bytes = csv_text.encode("utf-8")

        seg_str = f"{self._segment_index:05d}"
        orig_filename = f"{self.stem}_{self.session_id}_seg{seg_str}{self.suffix}"
        enc_filename = f"{orig_filename}.enc"
        dest_path = self.log_dir / enc_filename

        now_dt = datetime.now(timezone.utc)
        metadata = EvidenceMetadata(
            evidence_id=str(uuid.uuid4()),
            original_filename=orig_filename,
            evidence_type=self.evidence_type,
            created_at=now_dt.isoformat(),
            created_timestamp=now_dt.timestamp(),
            session_id=self.session_id,
            custom_attributes={
                "segment_index": self._segment_index,
                "record_count": len(self._buffer),
                "is_final": is_final,
                "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            },
        )

        self.storage_manager.write_encrypted_file(
            destination_path=dest_path,
            plaintext=raw_bytes,
            metadata=metadata,
            allow_overwrite=False,
        )

        self._segment_index += 1
        self._buffer.clear()
        return dest_path

    def close(self) -> None:
        """Flushes remaining records with is_final=True and closes the writer."""
        with self._lock:
            if self._is_closed:
                return
            self._flush_internal(is_final=True)
            self._is_closed = True


def read_encrypted_csv_segments(
    storage_manager: EncryptedStorageManager,
    log_dir: Union[str, Path],
    session_id: str,
    base_stem: str = "events",
) -> List[Dict[str, Any]]:
    """
    Finds, verifies, and reads all encrypted CSV segments for a session in order,
    returning the reconstructed records.
    """
    directory = Path(log_dir).resolve()
    pattern = f"{base_stem}_{session_id}_seg*.enc"
    segment_files = list(directory.glob(pattern))

    def get_seg_idx(p: Path) -> int:
        m = re.search(r"_seg(\d+)\.", p.name)
        return int(m.group(1)) if m else -1

    segment_files.sort(key=get_seg_idx)

    all_records: List[Dict[str, Any]] = []
    for seg_file in segment_files:
        plaintext, meta = storage_manager.read_encrypted_file(seg_file)
        text = plaintext.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            all_records.append(dict(row))
    return all_records


_default_storage_manager: Optional[EncryptedStorageManager] = None


def get_encrypted_storage_manager(
    keystore_path: Optional[Union[str, Path]] = None,
    fail_closed: bool = True,
) -> EncryptedStorageManager:
    """
    Returns or lazily initializes the singleton EncryptedStorageManager.
    Fails closed if the keystore is absent or invalid.
    """
    global _default_storage_manager
    if _default_storage_manager is None or keystore_path is not None:
        if keystore_path is None:
            import config
            keystore_path = getattr(config, "KEYSTORE_PATH", config.BASE_DIR / "keys" / "keystore.enc")

        target = Path(keystore_path).resolve()
        if not target.is_file():
            if fail_closed:
                raise SecurityConfigurationError(
                    f"Encrypted storage cannot be initialized: keystore file does not exist at '{target}'. "
                    f"A valid keystore must be initialized before enabling encryption."
                )
        keystore = Keystore.load(target)
        _default_storage_manager = EncryptedStorageManager(keystore)
    return _default_storage_manager


def reset_default_storage_manager() -> None:
    """Resets the singleton EncryptedStorageManager instance (for testing)."""
    global _default_storage_manager
    _default_storage_manager = None
