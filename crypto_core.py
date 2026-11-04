"""
Cryptographic Core Module for Cheating Surveillance System.

Provides:
- AES-256-GCM authenticated encryption with Associated Authenticated Data (AEAD).
- CSPRNG nonce generation and nonce-safety policy enforcement.
- Native Windows DPAPI User Scope key wrapping via ctypes (crypt32.dll).
- Master Recovery Key (MRK) generation, checksummed formatting, HKDF key derivation,
  and RFC 3394 AES Key Wrap / Unwrap.
- Strongly-typed, descriptive cryptographic exceptions with zero plaintext/secret leakage.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import hashlib
import hmac
import os
import secrets
import sys
from typing import Optional, Tuple, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.keywrap import (
    InvalidUnwrap,
    aes_key_unwrap,
    aes_key_wrap,
)

# Cryptographic Constants
AES_KEY_SIZE: int = 32       # 256 bits
GCM_NONCE_SIZE: int = 12     # 96 bits (standard NIST SP 800-38D)
GCM_TAG_SIZE: int = 16       # 128 bits
MRK_KEY_SIZE: int = 32       # 256 bits
MRK_CHECKSUM_SIZE: int = 8   # 64 bits domain-separated checksum
MRK_SALT_SIZE: int = 16      # 128 bits
MRK_HKDF_INFO: bytes = b"CSS_MRK_KEY_WRAP_v1"
MRK_CHECKSUM_DOMAIN: bytes = b"CSS_MRK_CHECKSUM_v1:"

# Recommended NIST SP 800-38D limit for 96-bit random nonces under a single key
MAX_ENCRYPTIONS_PER_KEY: int = 2**32


# ==============================================================================
# Project-Specific Cryptographic Exceptions
# ==============================================================================

class CryptoError(Exception):
    """Base exception for all cryptographic and key-management errors."""
    pass


class InvalidKeyError(CryptoError):
    """Raised when an encryption key is malformed, invalid, or wrong length."""
    pass


class InvalidNonceError(CryptoError):
    """Raised when an initialization vector / nonce is invalid or wrong length."""
    pass


class EncryptionError(CryptoError):
    """Raised when an encryption operation fails."""
    pass


class DecryptionError(CryptoError):
    """Raised when a decryption operation fails."""
    pass


class AuthenticationError(DecryptionError):
    """Raised when AEAD authentication verification fails (tampering detected)."""
    pass


class KeyManagementError(CryptoError):
    """Raised when keystore or key lifecycle operations fail."""
    pass


class KeyLimitReachedError(KeyManagementError):
    """Raised when a DEK has reached MAX_ENCRYPTIONS_PER_KEY and must be rotated."""
    pass


class SecurityConfigurationError(KeyManagementError):
    """Raised when security controls (such as strict filesystem ACLs) cannot be enforced."""
    pass


class DPAPIError(KeyManagementError):
    """Raised when Windows DPAPI protection or unprotection fails."""
    pass


class RecoveryError(KeyManagementError):
    """Raised when Master Recovery Key (MRK) validation or unwrapping fails."""
    pass


class UnsupportedPlatformError(CryptoError):
    """Raised when a platform-specific feature (e.g. Windows DPAPI) is executed on an unsupported OS."""
    pass


# ==============================================================================
# AES-256-GCM Primitives & Nonce Safety
# ==============================================================================

def generate_dek() -> bytes:
    """
    Generates a cryptographically secure 256-bit (32-byte) Data Encryption Key (DEK).

    Uses secrets.token_bytes backed by the operating system CSPRNG
    (CryptGenRandom / BCryptGenRandom on Windows, /dev/urandom on Unix).
    """
    return secrets.token_bytes(AES_KEY_SIZE)


def generate_nonce() -> bytes:
    """
    Generates a cryptographically secure 96-bit (12-byte) nonce for AES-256-GCM.

    Nonce Safety Policy:
    - Exactly 96 bits generated via CSPRNG (secrets.token_bytes(12)).
    - Never derived from timestamps, volatile counters, or user-controlled inputs.
    - Under NIST SP 800-38D, a 96-bit random nonce bound guarantees collision
      probability < 6.3e-18 for up to 1,000,000 invocations under the same key.
    - If total encryption operations exceed MAX_ENCRYPTIONS_PER_KEY (2^32),
      key rotation must be triggered.
    """
    return secrets.token_bytes(GCM_NONCE_SIZE)


class AESGCMCipher:
    """
    Authenticated Encryption with Associated Data (AEAD) cipher using AES-256-GCM.

    Guarantees:
    - Confidentiality via AES-256 in Counter (CTR) mode.
    - Authenticity and Integrity via Galois MAC (GMAC) with 128-bit authentication tags.
    - Cryptographic binding of optional Associated Authenticated Data (AAD).
    - Constant-time MAC comparison; zero plaintext leakage on authentication failure.
    """

    def __init__(self, key: bytes):
        if not isinstance(key, bytes):
            raise InvalidKeyError("Encryption key must be bytes.")
        if len(key) != AES_KEY_SIZE:
            raise InvalidKeyError(
                f"Invalid key length: expected {AES_KEY_SIZE} bytes (256 bits), got {len(key)} bytes."
            )
        self._key = key
        self._cipher = AESGCM(self._key)

    @property
    def key_size(self) -> int:
        """Returns key size in bytes (32)."""
        return AES_KEY_SIZE

    def encrypt(
        self,
        plaintext: bytes,
        nonce: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Encrypts plaintext with AES-256-GCM and appends a 16-byte authentication tag.

        Args:
            plaintext: Plaintext bytes to encrypt (can be empty b"").
            nonce: Optional 12-byte nonce. If None, a secure 96-bit nonce is generated.
            associated_data: Optional bytes authenticated alongside the ciphertext.

        Returns:
            Tuple[bytes, bytes]: (nonce [12 bytes], ciphertext_with_tag [len(plaintext) + 16 bytes])

        Raises:
            InvalidNonceError: If nonce is not exactly 12 bytes.
            EncryptionError: If encryption fails.
        """
        if not isinstance(plaintext, bytes):
            raise EncryptionError("Plaintext must be bytes.")

        if nonce is None:
            nonce = generate_nonce()
        elif not isinstance(nonce, bytes):
            raise InvalidNonceError("Nonce must be bytes.")
        elif len(nonce) != GCM_NONCE_SIZE:
            raise InvalidNonceError(
                f"Invalid nonce length: expected {GCM_NONCE_SIZE} bytes (96 bits), got {len(nonce)} bytes."
            )

        if associated_data is not None and not isinstance(associated_data, bytes):
            raise EncryptionError("Associated authenticated data must be bytes or None.")

        try:
            ciphertext_with_tag = self._cipher.encrypt(nonce, plaintext, associated_data)
            return nonce, ciphertext_with_tag
        except Exception as exc:
            raise EncryptionError(f"AES-GCM encryption failed: {exc}") from exc

    def decrypt(
        self,
        nonce: bytes,
        ciphertext_with_tag: bytes,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypts ciphertext and verifies the 16-byte authentication tag.

        Args:
            nonce: 12-byte nonce used during encryption.
            ciphertext_with_tag: Ciphertext payload with 16-byte authentication tag appended.
            associated_data: Optional associated data that must match the data supplied at encryption.

        Returns:
            bytes: Authenticated and decrypted plaintext.

        Raises:
            InvalidNonceError: If nonce length is not exactly 12 bytes.
            DecryptionError: If ciphertext is shorter than the 16-byte tag.
            AuthenticationError: If tag validation fails or data has been altered.
        """
        if not isinstance(nonce, bytes):
            raise InvalidNonceError("Nonce must be bytes.")
        if len(nonce) != GCM_NONCE_SIZE:
            raise InvalidNonceError(
                f"Invalid nonce length: expected {GCM_NONCE_SIZE} bytes (96 bits), got {len(nonce)} bytes."
            )
        if not isinstance(ciphertext_with_tag, bytes):
            raise DecryptionError("Ciphertext must be bytes.")
        if len(ciphertext_with_tag) < GCM_TAG_SIZE:
            raise DecryptionError(
                f"Ciphertext is truncated: expected at least {GCM_TAG_SIZE} bytes for authentication tag, "
                f"got {len(ciphertext_with_tag)} bytes."
            )

        if associated_data is not None and not isinstance(associated_data, bytes):
            raise DecryptionError("Associated authenticated data must be bytes or None.")

        try:
            plaintext = self._cipher.decrypt(nonce, ciphertext_with_tag, associated_data)
            return plaintext
        except InvalidTag as exc:
            # Re-raise specifically as AuthenticationError to prevent plaintext leakage
            raise AuthenticationError(
                "AEAD authentication verification failed: data has been altered, corrupted, "
                "or key/nonce/associated data does not match."
            ) from exc
        except Exception as exc:
            raise DecryptionError(f"AES-GCM decryption failed: {exc}") from exc


# ==============================================================================
# Windows DPAPI User Scope Implementation
# ==============================================================================

if sys.platform == "win32":
    class DATA_BLOB(ctypes.Structure):
        """Windows API DATA_BLOB structure for DPAPI calls."""
        _fields_ = [
            ("cbData", ctypes.wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_byte)),
        ]


class DPAPIKeyWrapper:
    """
    Key Encryption Key (KEK) wrapper utilizing Windows Data Protection API (DPAPI).

    Security Controls:
    - Enforces DPAPI User Scope strictly (flags=0).
    - NEVER sets CRYPTPROTECT_LOCAL_MACHINE (flag 0x4), preventing other local users
      from decrypting the DEK on shared workstations.
    - Automatically frees native memory buffers allocated by crypt32.dll via kernel32.LocalFree.
    - Raises UnsupportedPlatformError on non-Windows platforms.
    """

    def __init__(self):
        if sys.platform != "win32":
            raise UnsupportedPlatformError(
                f"Windows DPAPI is only supported on Windows operating systems (detected: {sys.platform})."
            )
        self._crypt32 = ctypes.windll.crypt32
        self._kernel32 = ctypes.windll.kernel32

        # Configure CryptProtectData signatures
        self._crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),          # pDataIn
            ctypes.c_wchar_p,                   # szDataDescr
            ctypes.POINTER(DATA_BLOB),          # pOptionalEntropy
            ctypes.c_void_p,                    # pvReserved
            ctypes.c_void_p,                    # pPromptStruct
            ctypes.wintypes.DWORD,              # dwFlags
            ctypes.POINTER(DATA_BLOB),          # pDataOut
        ]
        self._crypt32.CryptProtectData.restype = ctypes.wintypes.BOOL

        # Configure CryptUnprotectData signatures
        self._crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),          # pDataIn
            ctypes.POINTER(ctypes.c_wchar_p),   # ppszDataDescr
            ctypes.POINTER(DATA_BLOB),          # pOptionalEntropy
            ctypes.c_void_p,                    # pvReserved
            ctypes.c_void_p,                    # pPromptStruct
            ctypes.wintypes.DWORD,              # dwFlags
            ctypes.POINTER(DATA_BLOB),          # pDataOut
        ]
        self._crypt32.CryptUnprotectData.restype = ctypes.wintypes.BOOL

        # Configure LocalFree signature
        self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel32.LocalFree.restype = ctypes.c_void_p

    def protect(self, plaintext_key: bytes, description: str = "CSS_DEK_v1") -> bytes:
        """
        Wraps a plaintext key using Windows DPAPI in User Scope.

        Args:
            plaintext_key: Raw key bytes to wrap (e.g. 32-byte DEK).
            description: Optional textual description embedded into the DPAPI blob.

        Returns:
            bytes: DPAPI encrypted blob.

        Raises:
            DPAPIError: If CryptProtectData fails.
        """
        if not isinstance(plaintext_key, bytes) or len(plaintext_key) == 0:
            raise InvalidKeyError("Data to protect via DPAPI must be non-empty bytes.")

        in_blob = DATA_BLOB(
            len(plaintext_key),
            ctypes.cast(ctypes.c_char_p(plaintext_key), ctypes.POINTER(ctypes.c_byte)),
        )
        out_blob = DATA_BLOB()

        # dwFlags = 0 strictly enforces USER SCOPE.
        # CRYPTPROTECT_LOCAL_MACHINE (0x4) is NEVER set.
        dw_flags = 0

        res = self._crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            description,
            None,  # No optional entropy
            None,  # pvReserved
            None,  # pPromptStruct
            dw_flags,
            ctypes.byref(out_blob),
        )

        if not res:
            err_code = ctypes.GetLastError()
            raise DPAPIError(f"Windows DPAPI CryptProtectData failed with error code: {err_code}.")

        try:
            protected_bytes = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return protected_bytes
        finally:
            if out_blob.pbData:
                self._kernel32.LocalFree(out_blob.pbData)

    def unprotect(self, protected_blob: bytes) -> bytes:
        """
        Unwraps a DPAPI protected key blob using Windows DPAPI in User Scope.

        Args:
            protected_blob: DPAPI encrypted blob previously generated via protect().

        Returns:
            bytes: Unwrapped plaintext key bytes.

        Raises:
            DPAPIError: If CryptUnprotectData fails (e.g. wrong user profile or corrupted blob).
        """
        if not isinstance(protected_blob, bytes) or len(protected_blob) == 0:
            raise DPAPIError("Protected DPAPI blob must be non-empty bytes.")

        in_blob = DATA_BLOB(
            len(protected_blob),
            ctypes.cast(ctypes.c_char_p(protected_blob), ctypes.POINTER(ctypes.c_byte)),
        )
        out_blob = DATA_BLOB()
        dw_flags = 0

        res = self._crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            dw_flags,
            ctypes.byref(out_blob),
        )

        if not res:
            err_code = ctypes.GetLastError()
            raise DPAPIError(
                f"Windows DPAPI CryptUnprotectData failed with error code: {err_code}. "
                "Ensure execution occurs under the original Windows user account that initialized the keystore."
            )

        try:
            plaintext = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return plaintext
        finally:
            if out_blob.pbData:
                self._kernel32.LocalFree(out_blob.pbData)


# ==============================================================================
# Master Recovery Key (MRK) Architecture
# ==============================================================================

def generate_mrk() -> Tuple[bytes, str]:
    """
    Generates a 256-bit Master Recovery Key (MRK) with an 8-byte domain-separated checksum.

    Specification:
    - 256 bits (32 bytes) CSPRNG entropy.
    - 64 bits (8 bytes) domain-separated checksum: SHA-256("CSS_MRK_CHECKSUM_v1:" + raw_key)[:8].
    - Combined payload: 40 bytes = exactly 320 bits.
    - Base32 encoded (RFC 4648 without padding): exactly 64 characters (A-Z, 2-7).
    - Formatted into 16 hyphen-delimited groups of 4:
      CSS-MRK-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX

    Returns:
        Tuple[bytes, str]: (raw_mrk [32 bytes], formatted_human_readable_mrk [string])
    """
    raw_key = secrets.token_bytes(MRK_KEY_SIZE)
    checksum = hashlib.sha256(MRK_CHECKSUM_DOMAIN + raw_key).digest()[:MRK_CHECKSUM_SIZE]
    payload = raw_key + checksum
    b32 = base64.b32encode(payload).decode("ascii")

    groups = [b32[i:i + 4] for i in range(0, len(b32), 4)]
    formatted = "CSS-MRK-" + "-".join(groups)
    return raw_key, formatted


def parse_and_verify_mrk(mrk_input: Union[str, bytes]) -> bytes:
    """
    Parses and verifies a Master Recovery Key (MRK) from raw bytes or formatted string.

    Validation Rules:
    - If raw bytes: must be exactly 32 bytes.
    - If string:
      - Strips optional 'CSS-MRK-' or 'CSS-REC-' prefixes and hyphens/spaces.
      - Case-insensitive (uppercased).
      - Must decode to exactly 40 bytes (32 bytes key + 8 bytes checksum).
      - Verifies domain-separated SHA-256 checksum in constant time.

    Returns:
        bytes: Verified 32-byte raw MRK key.

    Raises:
        RecoveryError: If format is invalid, length is wrong, or checksum does not match.
    """
    if isinstance(mrk_input, bytes):
        if len(mrk_input) != MRK_KEY_SIZE:
            raise RecoveryError(
                f"Invalid raw MRK byte length: expected {MRK_KEY_SIZE} bytes, got {len(mrk_input)} bytes."
            )
        return mrk_input

    if not isinstance(mrk_input, str):
        raise RecoveryError("MRK input must be a formatted string or 32-byte raw bytes.")

    clean = mrk_input.strip().upper()
    if clean.startswith("CSS-MRK-"):
        clean = clean[8:]
    elif clean.startswith("CSS-REC-"):
        clean = clean[8:]

    clean = clean.replace("-", "").replace(" ", "")

    if len(clean) != 64:
        raise RecoveryError(
            f"Invalid MRK string length: expected 64 Base32 characters (16 groups of 4), got {len(clean)} characters."
        )

    try:
        decoded = base64.b32decode(clean.encode("ascii"), casefold=True)
    except Exception as exc:
        raise RecoveryError(f"Malformed Base32 encoding in MRK string: {exc}") from exc

    if len(decoded) != (MRK_KEY_SIZE + MRK_CHECKSUM_SIZE):
        raise RecoveryError(
            f"Decoded MRK payload invalid length: expected {MRK_KEY_SIZE + MRK_CHECKSUM_SIZE} bytes, got {len(decoded)} bytes."
        )

    raw_key = decoded[:MRK_KEY_SIZE]
    received_checksum = decoded[MRK_KEY_SIZE:]
    expected_checksum = hashlib.sha256(MRK_CHECKSUM_DOMAIN + raw_key).digest()[:MRK_CHECKSUM_SIZE]

    if not hmac.compare_digest(received_checksum, expected_checksum):
        raise RecoveryError(
            "Master Recovery Key checksum verification failed: string contains transcription or typing errors."
        )

    return raw_key


def wrap_dek_with_mrk(
    dek: bytes,
    raw_mrk: bytes,
    salt: Optional[bytes] = None,
) -> Tuple[bytes, bytes]:
    """
    Wraps a 32-byte DEK using an MRK-derived KEK via RFC 3394 AES Key Wrap.

    Derivation:
    - KEK = HKDF-SHA256(ikm=raw_mrk, length=32, salt=salt, info=MRK_HKDF_INFO).
    - Wrapped DEK = aes_key_wrap(KEK, DEK). Produces 40 bytes.

    Args:
        dek: 32-byte Data Encryption Key to wrap.
        raw_mrk: 32-byte verified raw Master Recovery Key.
        salt: Optional 16-byte random salt. If None, generated via CSPRNG.

    Returns:
        Tuple[bytes, bytes]: (wrapped_dek [40 bytes], salt [16 bytes])
    """
    if len(dek) != AES_KEY_SIZE:
        raise InvalidKeyError(f"DEK must be {AES_KEY_SIZE} bytes, got {len(dek)}.")
    if len(raw_mrk) != MRK_KEY_SIZE:
        raise RecoveryError(f"MRK must be {MRK_KEY_SIZE} bytes, got {len(raw_mrk)}.")

    if salt is None:
        salt = secrets.token_bytes(MRK_SALT_SIZE)
    elif len(salt) != MRK_SALT_SIZE:
        raise KeyManagementError(f"Salt must be {MRK_SALT_SIZE} bytes, got {len(salt)}.")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        info=MRK_HKDF_INFO,
    )
    kek = hkdf.derive(raw_mrk)

    try:
        wrapped_dek = aes_key_wrap(kek, dek)
        return wrapped_dek, salt
    except Exception as exc:
        raise KeyManagementError(f"AES Key Wrap failed: {exc}") from exc


def unwrap_dek_with_mrk(
    wrapped_dek: bytes,
    raw_mrk: bytes,
    salt: bytes,
) -> bytes:
    """
    Unwraps a 32-byte DEK using an MRK-derived KEK via RFC 3394 AES Key Unwrap.

    Args:
        wrapped_dek: 40-byte wrapped DEK payload.
        raw_mrk: 32-byte verified raw Master Recovery Key.
        salt: 16-byte salt used during wrapping.

    Returns:
        bytes: Unwrapped 32-byte Data Encryption Key.

    Raises:
        RecoveryError: If unwrapping fails (wrong MRK, wrong salt, or tampered key wrap).
    """
    if len(raw_mrk) != MRK_KEY_SIZE:
        raise RecoveryError(f"MRK must be {MRK_KEY_SIZE} bytes, got {len(raw_mrk)}.")
    if len(salt) != MRK_SALT_SIZE:
        raise KeyManagementError(f"Salt must be {MRK_SALT_SIZE} bytes, got {len(salt)}.")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        info=MRK_HKDF_INFO,
    )
    kek = hkdf.derive(raw_mrk)

    try:
        unwrapped_dek = aes_key_unwrap(kek, wrapped_dek)
        if len(unwrapped_dek) != AES_KEY_SIZE:
            raise RecoveryError(f"Unwrapped DEK length invalid: expected {AES_KEY_SIZE}, got {len(unwrapped_dek)}.")
        return unwrapped_dek
    except InvalidUnwrap as exc:
        raise RecoveryError(
            "Failed to unwrap DEK: Invalid Master Recovery Key, wrong salt, or corrupted key wrap payload."
        ) from exc
    except Exception as exc:
        raise RecoveryError(f"Key unwrapping failure: {exc}") from exc
