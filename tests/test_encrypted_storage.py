"""
Unit and Integration Tests for Phase 10.13.28.2 — Encrypted Evidence Storage Core.

Tests:
- Binary envelope layout, packing, parsing, and strict bounds validation.
- AES-256-GCM AEAD with full canonical metadata and filename binding.
- Anti-tampering: ciphertext, tag, nonce, Key ID, and metadata alterations.
- Persistent per-key encryption usage accounting (resolves SEC-28-1-01).
- Strict, fail-closed filesystem ACL enforcement (resolves SEC-28-1-02).
- Atomic encrypted file writes, temporary artifact cleanup, and overwrite protection.
- Historical DEK lookup and disaster recovery via MRK.
"""

import os
from pathlib import Path
import struct
import sys
import unittest.mock as mock
import uuid
import pytest

from crypto_core import (
    AES_KEY_SIZE,
    GCM_NONCE_SIZE,
    GCM_TAG_SIZE,
    AuthenticationError,
    KeyLimitReachedError,
    KeyManagementError,
    RecoveryError,
    SecurityConfigurationError,
)
from encrypted_storage import (
    ENVELOPE_MAGIC,
    ENVELOPE_VERSION,
    FIXED_HEADER_SIZE,
    MAX_METADATA_SIZE,
    AtomicWriteError,
    EncryptedStorageManager,
    EnvelopeHeader,
    EvidenceMetadata,
    FilenameMismatchError,
    InvalidEnvelopeError,
    UnsupportedVersionError,
    parse_envelope,
    read_authenticated_metadata,
    serialize_envelope,
)
from keystore import Keystore, apply_restrictive_permissions

pytestmark = pytest.mark.unit


def create_sample_metadata(
    filename: str = "eye_Looking_Down_1793079704.png",
    evidence_id: str = "ev-test-uuid-001",
    evidence_type: str = "violation_screenshot",
    timestamp: float = 1793079704.0,
    session_id: str = "sess_20261101_abc123",
) -> EvidenceMetadata:
    """Helper to generate valid synthetic evidence metadata."""
    return EvidenceMetadata(
        evidence_id=evidence_id,
        original_filename=filename,
        evidence_type=evidence_type,
        created_at="2026-11-01T14:30:00Z",
        created_timestamp=timestamp,
        session_id=session_id,
    )


# ==============================================================================
# Category 1: Envelope Packing, Parsing, and Bounds Tests
# ==============================================================================

def test_envelope_serialization_and_parsing_roundtrip():
    """Verify envelope serialization and parsing matches byte-for-byte."""
    key_id_bytes = uuid.uuid4().bytes
    nonce = b"N" * GCM_NONCE_SIZE
    meta_bytes = b'{"original_filename":"test.png"}'
    cipher_bytes = b"ciphertext_data_12345"
    tag_bytes = b"T" * GCM_TAG_SIZE

    envelope = serialize_envelope(key_id_bytes, nonce, meta_bytes, cipher_bytes, tag_bytes)
    assert len(envelope) == FIXED_HEADER_SIZE + len(meta_bytes) + len(cipher_bytes) + GCM_TAG_SIZE

    header, meta, cipher, tag, fixed_head = parse_envelope(envelope)
    assert header.magic == ENVELOPE_MAGIC
    assert header.version == ENVELOPE_VERSION
    assert header.header_length == FIXED_HEADER_SIZE
    assert header.key_id_bytes == key_id_bytes
    assert header.nonce == nonce
    assert header.metadata_length == len(meta_bytes)
    assert header.ciphertext_length == len(cipher_bytes)
    assert meta == meta_bytes
    assert cipher == cipher_bytes
    assert tag == tag_bytes


def test_envelope_invalid_magic_rejected():
    """Verify envelope with corrupt magic bytes is rejected immediately."""
    key_id_bytes = uuid.uuid4().bytes
    nonce = b"N" * GCM_NONCE_SIZE
    envelope = serialize_envelope(key_id_bytes, nonce, b"{}", b"c", b"T" * GCM_TAG_SIZE)

    corrupt_magic = b"BADMAGIC" + envelope[8:]
    with pytest.raises(InvalidEnvelopeError, match="Invalid envelope magic"):
        parse_envelope(corrupt_magic)


def test_envelope_unsupported_version_rejected():
    """Verify envelope with unsupported version is rejected."""
    key_id_bytes = uuid.uuid4().bytes
    nonce = b"N" * GCM_NONCE_SIZE
    # Patch version to 99
    bad_header = struct.pack(
        ">8sHH16s12sIQ",
        ENVELOPE_MAGIC,
        99,  # version 99
        FIXED_HEADER_SIZE,
        key_id_bytes,
        nonce,
        2,
        4,
    )
    bad_envelope = bad_header + b"{}" + b"data" + b"T" * GCM_TAG_SIZE
    with pytest.raises(UnsupportedVersionError, match="Unsupported envelope version 99"):
        parse_envelope(bad_envelope)


def test_envelope_truncated_rejected():
    """Verify truncated envelope is rejected cleanly without index errors."""
    key_id_bytes = uuid.uuid4().bytes
    nonce = b"N" * GCM_NONCE_SIZE
    envelope = serialize_envelope(key_id_bytes, nonce, b"{}", b"some_cipher_bytes", b"T" * GCM_TAG_SIZE)

    # Truncate before fixed header end
    with pytest.raises(InvalidEnvelopeError, match="Envelope truncated"):
        parse_envelope(envelope[:30])

    # Truncate mid-ciphertext
    with pytest.raises(InvalidEnvelopeError, match="Envelope length inconsistency"):
        parse_envelope(envelope[:-5])


def test_envelope_oversized_metadata_rejected():
    """Verify metadata exceeding MAX_METADATA_SIZE is rejected."""
    key_id_bytes = uuid.uuid4().bytes
    nonce = b"N" * GCM_NONCE_SIZE
    oversized_meta = b"A" * (MAX_METADATA_SIZE + 1)
    with pytest.raises(InvalidEnvelopeError, match="exceeds limit"):
        serialize_envelope(key_id_bytes, nonce, oversized_meta, b"c", b"T" * GCM_TAG_SIZE)


# ==============================================================================
# Category 2: AES-GCM and AAD Anti-Tampering Tests
# ==============================================================================

def test_storage_encryption_and_decryption_roundtrip(tmp_path):
    """Verify end-to-end encryption and decryption of payload and metadata."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    plaintext = b"High-resolution facial violation screenshot bytes..."
    metadata = create_sample_metadata(filename="eye_violation_001.png")

    envelope_bytes = mgr.encrypt_bytes(plaintext, metadata)
    decrypted_data, verified_meta = mgr.decrypt_bytes(envelope_bytes, expected_filename="eye_violation_001.png")

    assert decrypted_data == plaintext
    assert verified_meta.original_filename == "eye_violation_001.png"
    assert verified_meta.evidence_id == metadata.evidence_id
    assert verified_meta.created_timestamp == metadata.created_timestamp


def test_tampered_ciphertext_rejected(tmp_path):
    """Verify bit-flipping in ciphertext payload triggers AuthenticationError."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    envelope_bytes = mgr.encrypt_bytes(b"Secret surveillance payload", create_sample_metadata())
    header, meta, cipher, tag, fixed_head = parse_envelope(envelope_bytes)

    # Flip one bit in ciphertext
    tampered_cipher = bytearray(cipher)
    tampered_cipher[0] ^= 0x01
    tampered_envelope = fixed_head + meta + bytes(tampered_cipher) + tag

    with pytest.raises(AuthenticationError):
        mgr.decrypt_bytes(tampered_envelope)


def test_tampered_tag_rejected(tmp_path):
    """Verify modifying the authentication tag causes immediate failure."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    envelope_bytes = mgr.encrypt_bytes(b"Secret surveillance payload", create_sample_metadata())
    header, meta, cipher, tag, fixed_head = parse_envelope(envelope_bytes)

    tampered_tag = bytearray(tag)
    tampered_tag[-1] ^= 0x01
    tampered_envelope = fixed_head + meta + cipher + bytes(tampered_tag)

    with pytest.raises(AuthenticationError):
        mgr.decrypt_bytes(tampered_envelope)


def test_tampered_nonce_rejected(tmp_path):
    """Verify altering the nonce in fixed header breaks AAD / tag validation."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    envelope_bytes = mgr.encrypt_bytes(b"Secret surveillance payload", create_sample_metadata())
    header, meta, cipher, tag, fixed_head = parse_envelope(envelope_bytes)

    # Tamper with nonce inside fixed header
    tampered_head = bytearray(fixed_head)
    tampered_head[28] ^= 0x01  # nonce offset in fixed header
    tampered_envelope = bytes(tampered_head) + meta + cipher + tag

    with pytest.raises(AuthenticationError):
        mgr.decrypt_bytes(tampered_envelope)


def test_tampered_metadata_rejected(tmp_path):
    """Verify modifying metadata bytes causes GCM AAD verification failure."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    envelope_bytes = mgr.encrypt_bytes(b"Payload", create_sample_metadata(filename="legit.png"))
    header, meta, cipher, tag, fixed_head = parse_envelope(envelope_bytes)

    # Modify metadata byte
    tampered_meta = bytearray(meta)
    tampered_meta[-2] = ord("X") if tampered_meta[-2] != ord("X") else ord("Y")
    tampered_envelope = fixed_head + bytes(tampered_meta) + cipher + tag

    with pytest.raises((AuthenticationError, InvalidEnvelopeError)):
        mgr.decrypt_bytes(tampered_envelope)


def test_filename_mismatch_rejected(tmp_path):
    """Verify providing an expected_filename different from authenticated metadata raises FilenameMismatchError."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    envelope_bytes = mgr.encrypt_bytes(b"Payload", create_sample_metadata(filename="original.png"))

    with pytest.raises(FilenameMismatchError, match="Authenticated filename mismatch"):
        mgr.decrypt_bytes(envelope_bytes, expected_filename="renamed_target.png")


# ==============================================================================
# Category 3: Nonce Accounting & Per-Key Limits (SEC-28-1-01)
# ==============================================================================

def test_encryption_usage_reservation_persists(tmp_path):
    """Verify reserve_encryption increments and persists count across reloads."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    assert ks.get_key_entry().encryption_count == 0

    # First reservation
    seq1 = ks.reserve_encryption()
    assert seq1 == 1

    # Second reservation via manager encryption
    mgr.encrypt_bytes(b"sample", create_sample_metadata())
    assert ks.get_key_entry().encryption_count == 2

    # Reload keystore from disk: count must be 2
    ks_reloaded = Keystore.load(keystore_path)
    assert ks_reloaded.get_key_entry().encryption_count == 2


def test_encryption_limit_enforced_strictly(tmp_path):
    """Verify reaching MAX_ENCRYPTIONS_PER_KEY rejects further encryption with KeyLimitReachedError."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    # Set synthetic limit to 3 for testing
    mgr.encrypt_bytes(b"sample 1", create_sample_metadata(), max_limit=3)
    mgr.encrypt_bytes(b"sample 2", create_sample_metadata(), max_limit=3)
    mgr.encrypt_bytes(b"sample 3", create_sample_metadata(), max_limit=3)

    assert ks.get_key_entry().encryption_count == 3

    # 4th encryption must be rejected
    with pytest.raises(KeyLimitReachedError, match="reached the maximum permitted encryption limit"):
        mgr.encrypt_bytes(b"sample 4", create_sample_metadata(), max_limit=3)


def test_reservation_retained_on_encryption_failure(tmp_path):
    """Verify that if an encryption operation fails after reservation, the reserved count remains consumed."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    assert ks.get_key_entry().encryption_count == 0

    # Force failure in cipher after reservation
    with mock.patch("encrypted_storage.AESGCMCipher.encrypt", side_effect=Exception("Simulated hardware crash")):
        with pytest.raises(Exception, match="Simulated hardware crash"):
            mgr.encrypt_bytes(b"sample", create_sample_metadata())

    # Count must remain 1 (not decremented)
    assert ks.get_key_entry().encryption_count == 1
    ks_reloaded = Keystore.load(keystore_path)
    assert ks_reloaded.get_key_entry().encryption_count == 1


# ==============================================================================
# Category 4: Strict ACL Enforcement Tests (SEC-28-1-02)
# ==============================================================================

def test_acl_failure_fails_closed_on_keystore_save(tmp_path):
    """Verify that when icacls fails and fail_closed=True, Keystore.save raises SecurityConfigurationError."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)

    with mock.patch("keystore.apply_restrictive_permissions", side_effect=SecurityConfigurationError("ACL permission denied")):
        with pytest.raises(SecurityConfigurationError, match="ACL permission denied"):
            ks.save(fail_closed_acls=True)


def test_existing_keystore_preserved_on_acl_save_failure(tmp_path):
    """Verify that an existing valid keystore is untouched if save fails ACL verification."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    original_bytes = keystore_path.read_bytes()

    with mock.patch("keystore.apply_restrictive_permissions", side_effect=SecurityConfigurationError("ACL mock fail")):
        with pytest.raises(SecurityConfigurationError):
            ks.save(fail_closed_acls=True)

    # Existing file on disk must be byte-for-byte identical to original
    assert keystore_path.read_bytes() == original_bytes


# ==============================================================================
# Category 5: Historical Key Lookup and MRK Recovery Tests
# ==============================================================================

def test_historical_key_lookup_and_decryption(tmp_path):
    """Verify evidence encrypted with retired DEK remains decryptable after key rotation."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    old_key_id = ks.active_key_id
    old_data = b"Historical evidence captured under Key 1"
    envelope_old = mgr.encrypt_bytes(old_data, create_sample_metadata(filename="old_file.png"))

    # Rotate key
    new_key_id, _, _ = ks.rotate_key()
    assert new_key_id != old_key_id
    assert ks.active_key_id == new_key_id

    # New evidence under Key 2
    new_data = b"Recent evidence captured under Key 2"
    envelope_new = mgr.encrypt_bytes(new_data, create_sample_metadata(filename="new_file.png"))

    # Decrypt both
    decrypted_old, _ = mgr.decrypt_bytes(envelope_old, expected_filename="old_file.png")
    assert decrypted_old == old_data

    decrypted_new, _ = mgr.decrypt_bytes(envelope_new, expected_filename="new_file.png")
    assert decrypted_new == new_data


def test_historical_key_recovery_via_mrk(tmp_path):
    """Verify evidence encrypted under any key can be decrypted via Master Recovery Key."""
    keystore_path = tmp_path / "keystore.enc"
    ks, orig_raw_mrk, orig_fmt_mrk = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    data = b"Evidence requiring disaster recovery"
    envelope = mgr.encrypt_bytes(data, create_sample_metadata(filename="disaster.png"))

    # Simulate DPAPI unavailability: provide MRK input
    with mock.patch.object(ks, "unwrap_dek_via_dpapi", side_effect=Exception("DPAPI unavailable")):
        decrypted, _ = mgr.decrypt_bytes(
            envelope,
            expected_filename="disaster.png",
            mrk_input=orig_fmt_mrk,
        )
        assert decrypted == data


# ==============================================================================
# Category 6: Atomic Encrypted File Writes & Path Security Tests
# ==============================================================================

def test_atomic_write_creates_encrypted_file(tmp_path):
    """Verify write_encrypted_file writes complete envelope and cleans up .tmp files."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    dest = tmp_path / "evidence_output.png.enc"
    data = b"Image payload"
    meta = create_sample_metadata(filename="evidence_output.png")

    confirmed_path = mgr.write_encrypted_file(dest, data, meta)
    assert confirmed_path == dest
    assert dest.is_file()

    # Zero leftover .tmp files
    assert len(list(tmp_path.glob("*.tmp_*"))) == 0

    # Read back and decrypt
    decrypted, meta_read = mgr.read_encrypted_file(dest)
    assert decrypted == data
    assert meta_read.original_filename == "evidence_output.png"


def test_write_refuses_overwrite_without_flag(tmp_path):
    """Verify write_encrypted_file raises FileExistsError if file exists and allow_overwrite is False."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    dest = tmp_path / "existing.png.enc"
    dest.write_bytes(b"existing content")

    with pytest.raises(FileExistsError, match="already exists"):
        mgr.write_encrypted_file(dest, b"new data", create_sample_metadata(), allow_overwrite=False)


def test_write_replaces_atomically_when_allowed(tmp_path):
    """Verify allow_overwrite=True replaces target atomically."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    dest = tmp_path / "overwrite_me.png.enc"
    dest.write_bytes(b"old content")

    mgr.write_encrypted_file(dest, b"new content", create_sample_metadata(filename="overwrite_me.png"), allow_overwrite=True)
    decrypted, _ = mgr.read_encrypted_file(dest)
    assert decrypted == b"new content"


def test_write_cleans_up_temp_on_failure(tmp_path):
    """Verify simulated failure during write removes .tmp file and preserves target."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    dest = tmp_path / "target.png.enc"

    real_replace = os.replace

    def mock_replace(src, dst):
        if "target.png.enc" in str(dst):
            raise OSError("Disk I/O error during rename")
        return real_replace(src, dst)

    with mock.patch("os.replace", side_effect=mock_replace):
        with pytest.raises(AtomicWriteError, match="Disk I/O error during rename"):
            mgr.write_encrypted_file(dest, b"payload", create_sample_metadata())

    # Temp file must be cleaned up
    assert len(list(tmp_path.glob("*.tmp_*"))) == 0
    assert not dest.exists()


def test_write_rejects_symlink_destination(tmp_path):
    """Verify symlink or directory junction destination is rejected."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    dest = tmp_path / "symlink.png.enc"

    with mock.patch("encrypted_storage.is_symlink_or_junction", return_value=True):
        with pytest.raises(AtomicWriteError, match="symlink or junction"):
            mgr.write_encrypted_file(dest, b"payload", create_sample_metadata())


def test_read_metadata_without_decrypting_payload(tmp_path):
    """Verify read_metadata_from_file inspects timestamps without DEK unwrap."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    dest = tmp_path / "unopened.png.enc"
    meta = create_sample_metadata(filename="unopened.png", timestamp=1793080000.0)
    mgr.write_encrypted_file(dest, b"Encrypted image", meta)

    # Read metadata with DPAPI mocked to fail
    with mock.patch.object(ks, "unwrap_dek_via_dpapi", side_effect=Exception("Should not be called!")):
        extracted_meta = mgr.read_metadata_from_file(dest)
        assert extracted_meta.created_timestamp == 1793080000.0
        assert extracted_meta.original_filename == "unopened.png"
        assert extracted_meta.evidence_id == meta.evidence_id
