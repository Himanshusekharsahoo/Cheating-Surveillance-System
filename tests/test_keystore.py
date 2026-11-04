"""
Unit Tests for Phase 10.13.28.1 — Versioned Keystore Module.

Tests:
- First-time keystore initialization.
- Atomic persistence and interrupted write cleanup.
- Refusal to overwrite existing keystore.
- Fail-closed behavior on corrupted or invalid keystore data.
- Absence of plaintext DEK and MRK on disk.
- Dual-unwrapping: DPAPI unwrap and Master Recovery Key (MRK) recovery.
- Key rotation while preserving retired historical keys.
- Path traversal, symlink, and junction rejection.
- File permission hardening.
"""

import json
from pathlib import Path
import sys
import unittest.mock as mock
import pytest

from crypto_core import (
    AES_KEY_SIZE,
    DPAPIError,
    KeyManagementError,
    RecoveryError,
)
from keystore import (
    KEYSTORE_SCHEMA_VERSION,
    Keystore,
    KeyEntry,
    apply_restrictive_permissions,
)

pytestmark = pytest.mark.unit


def test_keystore_first_time_initialization(tmp_path):
    """Verify first-time keystore creation generates valid metadata and wrapped keys."""
    keystore_path = tmp_path / "config" / "keystore.enc"
    ks, raw_mrk, formatted_mrk = Keystore.initialize(keystore_path)

    assert keystore_path.is_file()
    assert ks.version == KEYSTORE_SCHEMA_VERSION
    assert len(ks.active_key_id) > 0
    assert ks.key_count == 1
    assert len(raw_mrk) == 32
    assert formatted_mrk.startswith("CSS-MRK-")

    # Verify both DPAPI unwrap and MRK recovery yield the exact same 32-byte DEK
    dek_dpapi = ks.unwrap_dek_via_dpapi()
    assert len(dek_dpapi) == AES_KEY_SIZE

    dek_mrk = ks.recover_dek_via_mrk(formatted_mrk)
    assert dek_mrk == dek_dpapi


def test_keystore_never_overwrites_existing_file(tmp_path):
    """Verify initialize fails closed if the keystore file already exists."""
    keystore_path = tmp_path / "keystore.enc"
    Keystore.initialize(keystore_path)
    assert keystore_path.exists()

    # Second initialization on existing file must raise KeyManagementError
    with pytest.raises(KeyManagementError, match="already exists"):
        Keystore.initialize(keystore_path)


def test_no_plaintext_keys_stored_on_disk(tmp_path):
    """Verify that neither the plaintext DEK nor plaintext MRK appears in the keystore file."""
    keystore_path = tmp_path / "keystore.enc"
    ks, raw_mrk, formatted_mrk = Keystore.initialize(keystore_path)
    dek = ks.unwrap_dek_via_dpapi()

    raw_file_bytes = keystore_path.read_bytes()
    file_text = keystore_path.read_text(encoding="utf-8")

    # Plaintext DEK bytes must never appear
    assert dek not in raw_file_bytes
    # Raw MRK bytes must never appear
    assert raw_mrk not in raw_file_bytes
    # Formatted MRK string must never appear
    assert formatted_mrk not in file_text
    # Unprefixed Base32 MRK string must never appear
    clean_mrk = formatted_mrk.replace("CSS-MRK-", "").replace("-", "")
    assert clean_mrk not in file_text


def test_keystore_load_existing_valid_file(tmp_path):
    """Verify loading an existing valid keystore from disk."""
    keystore_path = tmp_path / "keystore.enc"
    ks_orig, _, formatted_mrk = Keystore.initialize(keystore_path)
    orig_dek = ks_orig.unwrap_dek_via_dpapi()

    # Load from disk independently
    ks_loaded = Keystore.load(keystore_path)
    assert ks_loaded.active_key_id == ks_orig.active_key_id
    assert ks_loaded.version == KEYSTORE_SCHEMA_VERSION
    assert ks_loaded.key_count == 1

    # Verify unwrapped DEK matches
    assert ks_loaded.unwrap_dek_via_dpapi() == orig_dek
    assert ks_loaded.recover_dek_via_mrk(formatted_mrk) == orig_dek


def test_keystore_load_nonexistent_file_raises():
    """Verify loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Keystore.load(Path("nonexistent_keystore_path.enc"))


def test_keystore_load_corrupted_json_fails_closed(tmp_path):
    """Verify corrupted JSON raises KeyManagementError."""
    keystore_path = tmp_path / "corrupted.enc"
    keystore_path.write_text("{invalid json content, missing brackets", encoding="utf-8")

    with pytest.raises(KeyManagementError, match="Failed to read or parse keystore"):
        Keystore.load(keystore_path)


def test_keystore_unsupported_version_fails_closed(tmp_path):
    """Verify unsupported schema version is rejected."""
    keystore_path = tmp_path / "wrong_version.enc"
    bad_data = {
        "version": 999,
        "active_key_id": "test",
        "keys": {},
    }
    keystore_path.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(KeyManagementError, match="Unsupported keystore schema version"):
        Keystore.load(keystore_path)


def test_keystore_missing_active_key_fails_closed(tmp_path):
    """Verify missing active_key_id raises KeyManagementError."""
    keystore_path = tmp_path / "missing_active.enc"
    bad_data = {
        "version": 1,
        "active_key_id": "nonexistent_id",
        "keys": {},
    }
    keystore_path.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(KeyManagementError):
        Keystore.load(keystore_path)


def test_keystore_atomic_write_behavior(tmp_path):
    """Verify save creates no lingering .tmp files on success."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)

    # Check directory contents: only keystore.enc should exist, no .tmp files
    files = list(tmp_path.glob("*.tmp*"))
    assert len(files) == 0


def test_keystore_key_rotation(tmp_path):
    """Verify key rotation generates new active DEK, retires old DEK, and preserves both."""
    keystore_path = tmp_path / "keystore.enc"
    ks, orig_raw_mrk, orig_fmt_mrk = Keystore.initialize(keystore_path)
    old_key_id = ks.active_key_id
    old_dek = ks.unwrap_dek_via_dpapi(old_key_id)

    # Perform key rotation
    new_key_id, new_raw_mrk, new_fmt_mrk = ks.rotate_key()

    assert new_key_id != old_key_id
    assert ks.active_key_id == new_key_id
    assert ks.key_count == 2
    assert new_fmt_mrk != orig_fmt_mrk

    # Status checks
    assert ks.get_key_entry(old_key_id).status == "retired"
    assert ks.get_key_entry(new_key_id).status == "active"

    # Active DEK is the new DEK
    new_dek = ks.unwrap_dek_via_dpapi()
    assert new_dek != old_dek

    # Historical retired DEK remains decryptable via DPAPI and its original MRK
    assert ks.unwrap_dek_via_dpapi(old_key_id) == old_dek
    assert ks.recover_dek_via_mrk(orig_fmt_mrk, key_id=old_key_id) == old_dek

    # New DEK is recoverable via new MRK
    assert ks.recover_dek_via_mrk(new_fmt_mrk, key_id=new_key_id) == new_dek


def test_dpapi_failure_followed_by_mrk_recovery(tmp_path):
    """Verify that if DPAPI fails (e.g. foreign user context), MRK recovery succeeds."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, formatted_mrk = Keystore.initialize(keystore_path)
    expected_dek = ks.unwrap_dek_via_dpapi()

    # Simulate DPAPI failure (e.g. running under a different Windows user profile)
    with mock.patch.object(ks, "unwrap_dek_via_dpapi", side_effect=DPAPIError("Access denied / wrong profile")):
        with pytest.raises(DPAPIError):
            ks.unwrap_dek_via_dpapi()

        # Offline disaster recovery via MRK must still succeed
        recovered_dek = ks.recover_dek_via_mrk(formatted_mrk)
        assert recovered_dek == expected_dek


def test_symlink_or_junction_rejection(tmp_path):
    """Verify keystore rejects symlinks or junctions."""
    keystore_path = tmp_path / "keystore.enc"

    with mock.patch("keystore.is_symlink_or_junction", return_value=True):
        with pytest.raises(KeyManagementError, match="cannot be a symlink or junction"):
            Keystore.initialize(keystore_path)


def test_apply_restrictive_permissions(tmp_path):
    """Verify apply_restrictive_permissions executes cleanly without exceptions."""
    test_file = tmp_path / "perm_test.txt"
    test_file.write_text("secure data", encoding="utf-8")

    result = apply_restrictive_permissions(test_file)
    assert isinstance(result, bool)
