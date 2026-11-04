"""
Unit and Integration Tests for Phase 10.13.28.4 — Authorized Evidence Decryption and Export.

Tests:
- Valid export of individual encrypted screenshots.
- Valid session export consolidating ordered CSV segments.
- Metadata listing without decrypting payloads.
- Rejection of tampered or corrupted envelopes (zero plaintext release).
- Master Recovery Key (MRK) disaster recovery export.
- Overwrite prevention and collision detection.
- Symlink and directory junction destination rejection.
- Temporary file cleanup on export failure.
- Audit logging verification (timestamps, accounts, statuses, sanitization).
- Immutability of source encrypted files.
"""

import csv
import hashlib
import os
from pathlib import Path
import unittest.mock as mock
import uuid
import pytest

from crypto_core import (
    AuthenticationError,
    DecryptionError,
)
from keystore import Keystore
from encrypted_storage import (
    EncryptedStorageManager,
    EncryptedCSVWriter,
    EvidenceMetadata,
)
from evidence_export import (
    EvidenceExporter,
    ExportAuditLogger,
    ExportSecurityError,
    DestinationCollisionError,
    EvidenceCorruptedError,
)

pytestmark = pytest.mark.unit


def create_sample_encrypted_screenshot(
    mgr: EncryptedStorageManager,
    dest_path: Path,
    session_id: str = "sess_export_001",
    payload: bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest_image_bytes",
) -> Path:
    """Helper creating a valid encrypted screenshot file."""
    meta = EvidenceMetadata(
        evidence_id=str(uuid.uuid4()),
        original_filename=dest_path.name[:-4] if dest_path.name.endswith(".enc") else dest_path.name,
        evidence_type="SCREENSHOT",
        created_at="2026-11-01T15:00:00Z",
        created_timestamp=1793545200.0,
        session_id=session_id,
        custom_attributes={"file_sha256": hashlib.sha256(payload).hexdigest()},
    )
    return mgr.write_encrypted_file(
        destination_path=dest_path,
        plaintext=payload,
        metadata=meta,
    )


def test_valid_single_screenshot_export(tmp_path: Path):
    """Verify authorized decryption and atomic export of a valid encrypted screenshot."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enc_path = log_dir / "violation_1001.png.enc"
    test_bytes = b"fake_png_binary_data_123456"
    create_sample_encrypted_screenshot(mgr, enc_path, payload=test_bytes)

    out_dir = tmp_path / "exports"
    audit_path = tmp_path / "export_audit.csv"
    audit_logger = ExportAuditLogger(audit_path)
    exporter = EvidenceExporter(mgr, audit_logger=audit_logger)

    exported_file = exporter.export_single_evidence(
        source_path=enc_path,
        output_dir=out_dir,
    )

    assert exported_file.is_file()
    assert exported_file.name == "violation_1001.png"
    assert exported_file.read_bytes() == test_bytes

    # Source file must remain completely untouched
    assert enc_path.is_file()

    # Verify audit log record
    assert audit_path.is_file()
    with open(audit_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["operation"] == "EXPORT_SINGLE"
        assert rows[0]["status"] == "SUCCESS"
        assert rows[0]["source_file"] == str(enc_path)
        assert rows[0]["destination_file"] == str(exported_file)
        assert rows[0]["operator_account"] != ""


def test_list_encrypted_evidence_metadata(tmp_path: Path):
    """Verify cataloging encrypted evidence without decrypting payloads."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    create_sample_encrypted_screenshot(mgr, log_dir / "shot1.png.enc", session_id="sess_a")
    create_sample_encrypted_screenshot(mgr, log_dir / "shot2.png.enc", session_id="sess_b")

    exporter = EvidenceExporter(mgr, audit_log_path=tmp_path / "test_audit.csv")
    items = exporter.list_encrypted_evidence(log_dir)
    assert len(items) == 2

    # Verify filtering by session
    sess_a_items = exporter.list_encrypted_evidence(log_dir, session_filter="sess_a")
    assert len(sess_a_items) == 1
    assert sess_a_items[0].session_id == "sess_a"
    assert sess_a_items[0].original_filename == "shot1.png"


def test_export_session_consolidates_csv_segments(tmp_path: Path):
    """Verify exporting a session consolidates ordered CSV segments and decrypts screenshots."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_id = "sess_multi_export"

    # Write multiple CSV segments
    writer = EncryptedCSVWriter(
        storage_manager=mgr,
        log_dir=log_dir,
        base_filename="events.csv",
        session_id=session_id,
        fieldnames=["event_id", "type", "val"],
        chunk_size=2,
    )
    for i in range(5):
        writer.write_record({"event_id": str(i), "type": f"type_{i}", "val": f"v_{i}"})
    writer.close()

    # Add a screenshot for the session
    create_sample_encrypted_screenshot(mgr, log_dir / "shot_sess.png.enc", session_id=session_id)

    out_dir = tmp_path / "session_export"
    exporter = EvidenceExporter(mgr, audit_log_path=tmp_path / "test_audit.csv")

    exported = exporter.export_session_evidence(
        session_id=session_id,
        log_dir=log_dir,
        output_dir=out_dir,
    )

    # Must contain consolidated events CSV + screenshot
    assert len(exported) == 2
    csv_export = out_dir / f"events_{session_id}.csv"
    assert csv_export.is_file()

    with open(csv_export, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert rows[0]["event_id"] == "0"
        assert rows[4]["event_id"] == "4"


def test_tampered_ciphertext_rejected_without_plaintext_release(tmp_path: Path):
    """Verify tampering with ciphertext prevents plaintext release and logs failure."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enc_path = log_dir / "tampered.png.enc"
    create_sample_encrypted_screenshot(mgr, enc_path)

    # Tamper with the last byte of ciphertext/tag
    raw = bytearray(enc_path.read_bytes())
    raw[-1] ^= 0xFF
    enc_path.write_bytes(bytes(raw))

    out_dir = tmp_path / "exports"
    audit_path = tmp_path / "audit.csv"
    exporter = EvidenceExporter(mgr, audit_logger=ExportAuditLogger(audit_path))

    with pytest.raises(EvidenceCorruptedError):
        exporter.export_single_evidence(enc_path, out_dir)

    # Verify no plaintext file released
    assert not (out_dir / "tampered.png").exists()

    # Verify audit failure record
    with open(audit_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert rows[-1]["status"] == "FAILED"
        assert rows[-1]["failure_category"] == "AUTHENTICATION_FAILED"


def test_export_destination_collision_refuses_overwrite(tmp_path: Path):
    """Verify export refuses to overwrite existing destination files unless allow_overwrite=True."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enc_path = log_dir / "collision.png.enc"
    create_sample_encrypted_screenshot(mgr, enc_path, payload=b"new_data")

    out_dir = tmp_path / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_file = out_dir / "collision.png"
    dest_file.write_bytes(b"existing_data")

    exporter = EvidenceExporter(mgr, audit_log_path=tmp_path / "test_audit.csv")

    # Default: raises DestinationCollisionError
    with pytest.raises(DestinationCollisionError):
        exporter.export_single_evidence(enc_path, out_dir, allow_overwrite=False)

    # Existing file unmodified
    assert dest_file.read_bytes() == b"existing_data"

    # Permitted overwrite
    exporter.export_single_evidence(enc_path, out_dir, allow_overwrite=True)
    assert dest_file.read_bytes() == b"new_data"


def test_mrk_recovery_export(tmp_path: Path):
    """Verify export succeeds when DPAPI fails if valid MRK is provided."""
    ks_path = tmp_path / "keystore.enc"
    ks, raw_mrk, formatted_mrk = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enc_path = log_dir / "mrk_recovered.png.enc"
    payload = b"mrk_test_secret_payload"
    create_sample_encrypted_screenshot(mgr, enc_path, payload=payload)

    out_dir = tmp_path / "exports"
    exporter = EvidenceExporter(mgr, audit_log_path=tmp_path / "test_audit.csv")

    # Simulate DPAPI unwrap failure by patching unwrap_dek_via_dpapi on keystore
    from crypto_core import DPAPIError
    with mock.patch.object(ks, "unwrap_dek_via_dpapi", side_effect=DPAPIError("DPAPI unavailable")):
        # Without MRK -> fails
        with pytest.raises(EvidenceCorruptedError):
            exporter.export_single_evidence(enc_path, out_dir)

        # With valid MRK -> succeeds
        exported = exporter.export_single_evidence(
            enc_path,
            out_dir,
            mrk_input=formatted_mrk,
            allow_overwrite=True,
        )
        assert exported.read_bytes() == payload


def test_symlink_destination_rejected(tmp_path: Path):
    """Verify export rejects symlink or directory junction output destinations."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enc_path = log_dir / "safe.png.enc"
    create_sample_encrypted_screenshot(mgr, enc_path)

    out_dir = tmp_path / "symlink_dir"

    exporter = EvidenceExporter(mgr, audit_log_path=tmp_path / "test_audit.csv")
    with mock.patch("evidence_export.is_symlink_or_junction", return_value=True):
        with pytest.raises(ExportSecurityError, match="cannot be a symlink"):
            exporter.export_single_evidence(enc_path, out_dir)


def test_temporary_file_cleanup_on_write_failure(tmp_path: Path):
    """Verify temporary files are cleaned up if an error occurs during plaintext persist."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enc_path = log_dir / "cleanup_test.png.enc"
    create_sample_encrypted_screenshot(mgr, enc_path)

    out_dir = tmp_path / "exports"
    exporter = EvidenceExporter(mgr, audit_log_path=tmp_path / "test_audit.csv")

    # Simulate error during os.replace
    real_replace = os.replace

    def mock_replace(src, dst):
        if "cleanup_test.png" in str(dst):
            raise OSError("Disk write failed during rename")
        return real_replace(src, dst)

    with mock.patch("os.replace", side_effect=mock_replace):
        with pytest.raises(ExportSecurityError, match="Disk write failed during rename"):
            exporter.export_single_evidence(enc_path, out_dir)

    # Verify temp files cleaned up
    assert len(list(out_dir.glob("*.tmp_*"))) == 0
    assert not (out_dir / "cleanup_test.png").exists()
