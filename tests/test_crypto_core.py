"""
Unit Tests for Phase 10.13.28.1 — Cryptographic Core Module.

Tests:
- AES-256-GCM authenticated encryption and decryption.
- Nonce generation and validation.
- Authentication tag and AAD tamper detection.
- Malformed inputs and exception handling.
- Windows DPAPI User Scope protection and memory safety.
- Master Recovery Key (MRK) generation, checksum validation, and key wrap/unwrap.
"""

import sys
import unittest.mock as mock
import pytest

from crypto_core import (
    AES_KEY_SIZE,
    GCM_NONCE_SIZE,
    GCM_TAG_SIZE,
    MRK_KEY_SIZE,
    AESGCMCipher,
    AuthenticationError,
    CryptoError,
    DecryptionError,
    DPAPIError,
    DPAPIKeyWrapper,
    EncryptionError,
    InvalidKeyError,
    InvalidNonceError,
    KeyManagementError,
    RecoveryError,
    UnsupportedPlatformError,
    generate_dek,
    generate_mrk,
    generate_nonce,
    parse_and_verify_mrk,
    unwrap_dek_with_mrk,
    wrap_dek_with_mrk,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# AES-256-GCM Tests
# ==============================================================================

def test_generate_dek_length_and_randomness():
    """Verify generated DEK is exactly 32 bytes and distinct across calls."""
    key1 = generate_dek()
    key2 = generate_dek()
    assert isinstance(key1, bytes)
    assert len(key1) == AES_KEY_SIZE
    assert len(key2) == AES_KEY_SIZE
    assert key1 != key2


def test_generate_nonce_length_and_randomness():
    """Verify generated nonce is exactly 12 bytes and distinct across calls."""
    nonce1 = generate_nonce()
    nonce2 = generate_nonce()
    assert isinstance(nonce1, bytes)
    assert len(nonce1) == GCM_NONCE_SIZE
    assert len(nonce2) == GCM_NONCE_SIZE
    assert nonce1 != nonce2


def test_aes_gcm_successful_roundtrip():
    """Verify AES-GCM encryption and decryption roundtrip matches byte-for-byte."""
    key = generate_dek()
    cipher = AESGCMCipher(key)
    plaintext = b"Examination incident evidence: candidate looking left for 3.5 seconds."

    nonce, ciphertext = cipher.encrypt(plaintext)
    assert len(nonce) == GCM_NONCE_SIZE
    assert len(ciphertext) == len(plaintext) + GCM_TAG_SIZE
    assert ciphertext != plaintext

    decrypted = cipher.decrypt(nonce, ciphertext)
    assert decrypted == plaintext


def test_aes_gcm_empty_payload():
    """Verify handling of empty plaintext bytes."""
    key = generate_dek()
    cipher = AESGCMCipher(key)
    plaintext = b""

    nonce, ciphertext = cipher.encrypt(plaintext)
    assert len(ciphertext) == GCM_TAG_SIZE
    assert cipher.decrypt(nonce, ciphertext) == b""


def test_aes_gcm_with_associated_authenticated_data():
    """Verify AEAD Associated Authenticated Data (AAD) roundtrip and validation."""
    key = generate_dek()
    cipher = AESGCMCipher(key)
    plaintext = b"Sensitive frame payload"
    aad = b"session_20261101_user_001_eye_looking_down"

    nonce, ciphertext = cipher.encrypt(plaintext, associated_data=aad)
    decrypted = cipher.decrypt(nonce, ciphertext, associated_data=aad)
    assert decrypted == plaintext

    # Tampered AAD must cause AuthenticationError
    tampered_aad = b"session_20261101_user_002_eye_looking_down"
    with pytest.raises(AuthenticationError):
        cipher.decrypt(nonce, ciphertext, associated_data=tampered_aad)

    # Missing AAD when encrypted with AAD must fail
    with pytest.raises(AuthenticationError):
        cipher.decrypt(nonce, ciphertext, associated_data=None)


def test_aes_gcm_wrong_key_rejection():
    """Verify decryption fails when attempted with a different DEK."""
    key1 = generate_dek()
    key2 = generate_dek()
    cipher1 = AESGCMCipher(key1)
    cipher2 = AESGCMCipher(key2)

    plaintext = b"Top secret exam telemetry"
    nonce, ciphertext = cipher1.encrypt(plaintext)

    with pytest.raises(AuthenticationError):
        cipher2.decrypt(nonce, ciphertext)


def test_aes_gcm_ciphertext_tampering_detected():
    """Verify bit-flipping in ciphertext is caught by GCM tag verification."""
    key = generate_dek()
    cipher = AESGCMCipher(key)
    plaintext = b"Normal candidate head pose angles: pitch=0.0 yaw=0.0"

    nonce, ciphertext = cipher.encrypt(plaintext)

    # Flip a single bit in the ciphertext payload (prior to the tag)
    tampered = bytearray(ciphertext)
    tampered[5] ^= 0x01
    tampered_bytes = bytes(tampered)

    with pytest.raises(AuthenticationError):
        cipher.decrypt(nonce, tampered_bytes)


def test_aes_gcm_tag_tampering_detected():
    """Verify modifying the authentication tag causes authentication failure."""
    key = generate_dek()
    cipher = AESGCMCipher(key)
    plaintext = b"Telemetry record"

    nonce, ciphertext = cipher.encrypt(plaintext)

    # Flip a single bit in the 16-byte authentication tag at the end
    tampered = bytearray(ciphertext)
    tampered[-1] ^= 0x01
    tampered_bytes = bytes(tampered)

    with pytest.raises(AuthenticationError):
        cipher.decrypt(nonce, tampered_bytes)


def test_aes_gcm_truncated_ciphertext_rejected():
    """Verify truncated ciphertexts raise DecryptionError."""
    key = generate_dek()
    cipher = AESGCMCipher(key)
    nonce = generate_nonce()

    # Fewer than 16 bytes (tag size)
    with pytest.raises(DecryptionError):
        cipher.decrypt(nonce, b"short")


def test_aes_gcm_invalid_key_length():
    """Verify AESGCMCipher rejects keys that are not 32 bytes."""
    with pytest.raises(InvalidKeyError):
        AESGCMCipher(b"short_key")
    with pytest.raises(InvalidKeyError):
        AESGCMCipher(b"a" * 16)  # 128-bit key rejected (256-bit required)
    with pytest.raises(InvalidKeyError):
        AESGCMCipher("string_key")  # type: ignore


def test_aes_gcm_invalid_nonce_length():
    """Verify encrypt and decrypt reject nonces not equal to 12 bytes."""
    key = generate_dek()
    cipher = AESGCMCipher(key)

    with pytest.raises(InvalidNonceError):
        cipher.encrypt(b"data", nonce=b"short_nonce")
    with pytest.raises(InvalidNonceError):
        cipher.encrypt(b"data", nonce=b"a" * 16)

    with pytest.raises(InvalidNonceError):
        cipher.decrypt(b"short_nonce", b"a" * 20)


# ==============================================================================
# Windows DPAPI Tests
# ==============================================================================

@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI requires win32 platform")
def test_dpapi_user_scope_roundtrip():
    """Verify native Windows DPAPI User Scope protect and unprotect roundtrip."""
    wrapper = DPAPIKeyWrapper()
    secret = b"Ephemeral 256-bit synthetic DEK bytes for test"

    protected = wrapper.protect(secret, description="Test_DEK")
    assert isinstance(protected, bytes)
    assert len(protected) > len(secret)
    assert protected != secret

    unprotected = wrapper.unprotect(protected)
    assert unprotected == secret


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI requires win32 platform")
def test_dpapi_machine_scope_never_used():
    """Verify that DPAPIKeyWrapper strictly uses dwFlags=0 (User Scope) and never Machine Scope."""
    wrapper = DPAPIKeyWrapper()
    secret = b"Synthetic secret"

    # Inspect protect implementation: calls CryptProtectData with dwFlags = 0
    with mock.patch.object(wrapper._crypt32, "CryptProtectData", wraps=wrapper._crypt32.CryptProtectData) as mock_protect:
        wrapper.protect(secret)
        assert mock_protect.called
        # Arg 6 is dwFlags (index 5)
        flags = mock_protect.call_args[0][5]
        assert flags == 0, f"Expected dwFlags=0 (User Scope), got {flags}"
        # Ensure CRYPTPROTECT_LOCAL_MACHINE (0x4) is NOT set
        assert (flags & 0x4) == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI requires win32 platform")
def test_dpapi_empty_input_rejected():
    """Verify DPAPI protect rejects empty input."""
    wrapper = DPAPIKeyWrapper()
    with pytest.raises(InvalidKeyError):
        wrapper.protect(b"")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI requires win32 platform")
def test_dpapi_corrupted_blob_fails():
    """Verify DPAPI unprotect raises DPAPIError on corrupted blob."""
    wrapper = DPAPIKeyWrapper()
    with pytest.raises(DPAPIError):
        wrapper.unprotect(b"corrupted_dpapi_blob_bytes_1234567890")


def test_dpapi_unsupported_platform():
    """Verify DPAPIKeyWrapper raises UnsupportedPlatformError on non-Windows platforms."""
    with mock.patch("sys.platform", "linux"):
        with pytest.raises(UnsupportedPlatformError):
            DPAPIKeyWrapper()


# ==============================================================================
# Master Recovery Key (MRK) Tests
# ==============================================================================

def test_mrk_generation_and_formatting():
    """Verify MRK generation produces a 32-byte raw key and 16-group Base32 string."""
    raw_key, formatted = generate_mrk()
    assert isinstance(raw_key, bytes)
    assert len(raw_key) == MRK_KEY_SIZE
    assert formatted.startswith("CSS-MRK-")

    # 16 groups of 4 characters separated by 15 hyphens = 64 + 15 = 79 chars (+ 8 for prefix = 87)
    assert len(formatted) == 87
    parts = formatted[8:].split("-")
    assert len(parts) == 16
    for p in parts:
        assert len(p) == 4


def test_mrk_parse_and_verify_valid():
    """Verify parsing and checksum validation of valid MRK string."""
    raw_key, formatted = generate_mrk()
    parsed_key = parse_and_verify_mrk(formatted)
    assert parsed_key == raw_key

    # Also test case-insensitivity and whitespace stripping
    lower_formatted = formatted.lower()
    assert parse_and_verify_mrk(lower_formatted) == raw_key
    spaced_formatted = f"  {formatted}  \n"
    assert parse_and_verify_mrk(spaced_formatted) == raw_key


def test_mrk_checksum_tamper_detection():
    """Verify that any modified character in MRK string fails checksum validation."""
    raw_key, formatted = generate_mrk()
    parts = formatted.split("-")

    # Tamper with character in 5th group
    group = list(parts[5])
    group[0] = "A" if group[0] != "A" else "B"
    parts[5] = "".join(group)
    tampered_mrk = "-".join(parts)

    with pytest.raises(RecoveryError, match="checksum verification failed"):
        parse_and_verify_mrk(tampered_mrk)


def test_mrk_invalid_length_rejected():
    """Verify MRK parsing rejects strings with missing or extra characters."""
    raw_key, formatted = generate_mrk()
    truncated = formatted[:-4]
    with pytest.raises(RecoveryError):
        parse_and_verify_mrk(truncated)


def test_wrap_and_unwrap_dek_with_mrk():
    """Verify wrapping and unwrapping DEK using MRK-derived KEK via RFC 3394."""
    dek = generate_dek()
    raw_mrk, formatted_mrk = generate_mrk()

    wrapped_dek, salt = wrap_dek_with_mrk(dek, raw_mrk)
    assert len(wrapped_dek) == 40  # 32-byte key wrapped produces 40 bytes
    assert len(salt) == 16
    assert wrapped_dek != dek

    # Successful unwrap
    recovered_dek = unwrap_dek_with_mrk(wrapped_dek, raw_mrk, salt)
    assert recovered_dek == dek


def test_unwrap_dek_with_wrong_mrk_fails():
    """Verify unwrapping DEK with wrong MRK raises RecoveryError."""
    dek = generate_dek()
    raw_mrk1, _ = generate_mrk()
    raw_mrk2, _ = generate_mrk()

    wrapped_dek, salt = wrap_dek_with_mrk(dek, raw_mrk1)

    with pytest.raises(RecoveryError, match="Failed to unwrap DEK"):
        unwrap_dek_with_mrk(wrapped_dek, raw_mrk2, salt)


def test_unwrap_dek_with_tampered_wrapped_bytes_fails():
    """Verify unwrapping tampered wrapped DEK bytes raises RecoveryError."""
    dek = generate_dek()
    raw_mrk, _ = generate_mrk()
    wrapped_dek, salt = wrap_dek_with_mrk(dek, raw_mrk)

    tampered_wrapped = bytearray(wrapped_dek)
    tampered_wrapped[10] ^= 0x01

    with pytest.raises(RecoveryError, match="Failed to unwrap DEK"):
        unwrap_dek_with_mrk(bytes(tampered_wrapped), raw_mrk, salt)
