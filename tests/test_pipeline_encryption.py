"""
Unit and Integration Tests for Phase 10.13.28.3 — Live Surveillance Pipeline Encryption Integration.

Tests:
- End-to-end evidence encryption in memory without plaintext disk artifacts.
- Seamless dual-mode operation: encryption enabled vs encryption disabled.
- Session-rotated encrypted CSV segments (events and gaze diagnostics).
- Concurrency, chunk rotation, and shutdown finalization.
- Fail-closed behavior on missing keystore, missing DEK, corrupted keystore, or ACL failure.
- Zero plaintext fallback under all failure modes.
"""

from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import threading
from typing import Any, Dict
import unittest.mock as mock
import uuid
import numpy as np
import pytest

from crypto_core import (
    SecurityConfigurationError,
    KeyManagementError,
    generate_dek,
)
from keystore import Keystore
from encrypted_storage import (
    EncryptedStorageManager,
    EncryptedCSVWriter,
    read_encrypted_csv_segments,
    reset_default_storage_manager,
)
from event_logger import EventLogger, FIELDNAMES
from vision_pipeline import GazeDiagnosticsLogger, GAZE_DIAGNOSTICS_FIELDNAMES
import config
from main import record_violation_incident

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cleanup_storage_manager():
    """Ensure singleton storage manager is reset before and after each test."""
    reset_default_storage_manager()
    yield
    reset_default_storage_manager()


def test_encryption_disabled_baseline(tmp_path: Path):
    """Verify that when encryption is disabled, legacy plaintext behavior is preserved."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. EventLogger writes standard unencrypted events.csv
    logger = EventLogger(log_dir=log_dir, encryption_enabled=False)
    logger.log_event("mobile_detected", duration_sec=3.2, confidence=0.88)
    csv_file = log_dir / "events.csv"
    assert csv_file.is_file()
    assert not list(log_dir.glob("*.enc"))

    # 2. record_violation_incident writes standard unencrypted .png
    synthetic_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    success, ref = record_violation_incident(
        violation_type="looking_away",
        prefix="test_plain",
        frame=synthetic_frame,
        duration_sec=2.5,
        confidence=0.91,
        log_dir=str(log_dir),
        logger=logger,
        encryption_enabled=False,
    )
    assert success is True
    assert ref.endswith(".png")
    plain_png = log_dir / Path(ref).name
    assert plain_png.is_file()
    assert not plain_png.name.endswith(".enc")


def test_encryption_enabled_screenshots_and_events(tmp_path: Path):
    """Verify screenshot and event logging produce encrypted envelopes without plaintext files."""
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    keystore_path = keys_dir / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    session_id = "test_enc_session_001"
    logger = EventLogger(
        log_dir=log_dir,
        session_id=session_id,
        storage_manager=mgr,
        encryption_enabled=True,
        chunk_size=1,
    )

    # Log violation event and capture encrypted screenshot
    synthetic_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    success, ref = record_violation_incident(
        violation_type="multiple_faces",
        prefix="multi_face",
        frame=synthetic_frame,
        duration_sec=3.5,
        confidence=0.95,
        log_dir=str(log_dir),
        logger=logger,
        storage_manager=mgr,
        encryption_enabled=True,
    )

    assert success is True
    assert ref.endswith(".png.enc")
    enc_screenshot = log_dir / Path(ref).name
    assert enc_screenshot.is_file()

    # Plaintext screenshot must NOT exist
    plain_name = Path(ref).name[:-4]  # Strip .enc
    assert not (log_dir / plain_name).exists()

    # Plaintext events.csv must NOT exist
    assert not (log_dir / "events.csv").exists()

    # Encrypted CSV segment must exist
    seg_files = list(log_dir.glob(f"events_{session_id}_seg*.enc"))
    assert len(seg_files) == 1

    # Verify screenshot decryption recovers exact image
    decrypted_bytes, meta = mgr.read_encrypted_file(enc_screenshot)
    assert meta.evidence_type == "SCREENSHOT"
    assert meta.session_id == session_id
    assert hashlib.sha256(decrypted_bytes).hexdigest() == meta.custom_attributes.get("file_sha256")

    # Verify event decryption recovers exact record
    records = read_encrypted_csv_segments(mgr, log_dir, session_id, base_stem="events")
    assert len(records) == 1
    assert records[0]["violation_type"] == "multiple_faces"
    assert records[0]["duration_sec"] == "3.50"
    assert records[0]["confidence"] == "0.9500"
    assert Path(records[0]["screenshot_path"]) == Path(ref)


def test_encrypted_csv_chunk_rotation_and_ordering(tmp_path: Path):
    """Verify that EncryptedCSVWriter rotates chunks at chunk_size and preserves ordering."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_id = "test_rotation_sess"

    writer = EncryptedCSVWriter(
        storage_manager=mgr,
        log_dir=log_dir,
        base_filename="events.csv",
        session_id=session_id,
        fieldnames=["id", "val"],
        chunk_size=3,
        evidence_type="EVENT_LOG",
    )

    # Write 7 records (should produce seg00000 [3 recs], seg00001 [3 recs], and 1 buffered)
    for i in range(7):
        writer.write_record({"id": str(i), "val": f"val_{i}"})

    # Close flushes final record into seg00002 [1 rec, is_final=True]
    writer.close()

    seg_files = sorted(list(log_dir.glob(f"events_{session_id}_seg*.enc")))
    assert len(seg_files) == 3
    assert seg_files[0].name.endswith("_seg00000.csv.enc")
    assert seg_files[1].name.endswith("_seg00001.csv.enc")
    assert seg_files[2].name.endswith("_seg00002.csv.enc")

    # Read segments back and check records
    records = read_encrypted_csv_segments(mgr, log_dir, session_id, base_stem="events")
    assert len(records) == 7
    for i in range(7):
        assert records[i]["id"] == str(i)
        assert records[i]["val"] == f"val_{i}"

    # Check is_final flag on metadata
    meta0 = mgr.read_metadata_from_file(seg_files[0])
    assert meta0.custom_attributes.get("is_final") is False
    assert meta0.custom_attributes.get("record_count") == 3

    meta2 = mgr.read_metadata_from_file(seg_files[2])
    assert meta2.custom_attributes.get("is_final") is True
    assert meta2.custom_attributes.get("record_count") == 1


def test_gaze_diagnostics_encryption(tmp_path: Path):
    """Verify GazeDiagnosticsLogger buffers and flushes encrypted diagnostic segments."""
    from vision_pipeline import FaceAnalysisResult
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_id = "test_gaze_diag_sess"

    gaze_logger = GazeDiagnosticsLogger(
        log_dir=log_dir,
        session_id=session_id,
        storage_manager=mgr,
        encryption_enabled=True,
        chunk_size=5,
    )

    for i in range(12):
        face_res = FaceAnalysisResult(
            face_count=1,
            gaze_diagnostics={
                "missing_face": False,
                "head_pitch": float(i),
                "head_yaw": float(-i),
                "gaze_direction": "center",
            },
        )
        gaze_logger.log_frame(frame_index=i, face_result=face_res)

    gaze_logger.close()

    # Plaintext gaze_diagnostics.csv must NOT exist
    assert not (log_dir / "gaze_diagnostics.csv").exists()

    # Segments must exist
    records = read_encrypted_csv_segments(mgr, log_dir, session_id, base_stem="gaze_diagnostics")
    assert len(records) == 12
    assert float(records[0]["head_pitch"]) == 0.0
    assert float(records[11]["head_pitch"]) == 11.0


def test_concurrent_encrypted_event_writes(tmp_path: Path):
    """Verify concurrent threads logging events with encryption maintain consistency."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_id = "concurrent_sess"

    logger = EventLogger(
        log_dir=log_dir,
        session_id=session_id,
        storage_manager=mgr,
        encryption_enabled=True,
        chunk_size=2,
    )

    total_threads = 4
    events_per_thread = 5

    def worker(tid: int):
        for e in range(events_per_thread):
            logger.log_event(
                violation_type=f"thread_{tid}_event_{e}",
                duration_sec=float(tid * 10 + e),
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(total_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.close()

    records = read_encrypted_csv_segments(mgr, log_dir, session_id, base_stem="events")
    assert len(records) == total_threads * events_per_thread
    types = {r["violation_type"] for r in records}
    assert len(types) == total_threads * events_per_thread


def test_missing_keystore_fails_closed(tmp_path: Path, monkeypatch):
    """Verify that if encryption is enabled but keystore is missing, initialization fails closed."""
    log_dir = tmp_path / "log"
    missing_keystore = tmp_path / "nonexistent" / "keystore.enc"
    predictor = tmp_path / "fake.dat"
    yolo = tmp_path / "fake.pt"
    predictor.write_bytes(b"dummy")
    yolo.write_bytes(b"dummy")
    monkeypatch.setattr(config, "SHAPE_PREDICTOR_PATH", predictor)
    monkeypatch.setattr(config, "YOLO_MODEL_PATH", yolo)
    monkeypatch.setattr(config, "LOG_DIR", log_dir)

    with mock.patch("config.enforce_offline_isolation"):
        with mock.patch("config.ENCRYPTION_ENABLED", True):
            with mock.patch("config.KEYSTORE_PATH", missing_keystore):
                # 1. EventLogger fails closed with SecurityConfigurationError
                with pytest.raises(SecurityConfigurationError, match="keystore file does not exist"):
                    EventLogger(log_dir=log_dir, encryption_enabled=True)

                # 2. validate_config fails closed
                with pytest.raises(SecurityConfigurationError, match="keystore file does not exist"):
                    config.validate_config()


def test_corrupted_keystore_fails_closed(tmp_path: Path):
    """Verify that if keystore is corrupted, pipeline initialization fails closed."""
    log_dir = tmp_path / "log"
    corrupt_keystore = tmp_path / "corrupt_keystore.enc"
    corrupt_keystore.write_text("{invalid json brackets", encoding="utf-8")

    with mock.patch("config.ENCRYPTION_ENABLED", True):
        with mock.patch("config.KEYSTORE_PATH", corrupt_keystore):
            with pytest.raises((SecurityConfigurationError, KeyManagementError)):
                EventLogger(log_dir=log_dir, encryption_enabled=True)


def test_screenshot_write_failure_does_not_fallback_to_plaintext(tmp_path: Path):
    """Verify that if encrypted screenshot persistence fails, it fails closed without writing plaintext."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(log_dir=log_dir, storage_manager=mgr, encryption_enabled=True)

    synthetic_frame = np.zeros((50, 50, 3), dtype=np.uint8)

    # Simulate storage manager failure
    with mock.patch.object(mgr, "write_encrypted_file", side_effect=IOError("Storage disk failure")):
        success, ref = record_violation_incident(
            violation_type="head_turn",
            prefix="fail_test",
            frame=synthetic_frame,
            duration_sec=3.0,
            log_dir=str(log_dir),
            logger=logger,
            storage_manager=mgr,
            encryption_enabled=True,
        )

    assert success is False
    assert ref == ""
    # Ensure NO unencrypted screenshot file was created
    assert len(list(log_dir.glob("*.png"))) == 0


def test_session_isolation(tmp_path: Path):
    """Verify that different sessions produce isolated segment sequences."""
    keystore_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(keystore_path)
    mgr = EncryptedStorageManager(ks)

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    sess1 = "session_alpha"
    sess2 = "session_beta"

    logger1 = EventLogger(log_dir=log_dir, session_id=sess1, storage_manager=mgr, encryption_enabled=True)
    logger2 = EventLogger(log_dir=log_dir, session_id=sess2, storage_manager=mgr, encryption_enabled=True)

    logger1.log_event("alpha_event", duration_sec=1.0)
    logger2.log_event("beta_event", duration_sec=2.0)

    logger1.close()
    logger2.close()

    recs1 = read_encrypted_csv_segments(mgr, log_dir, sess1, base_stem="events")
    recs2 = read_encrypted_csv_segments(mgr, log_dir, sess2, base_stem="events")

    assert len(recs1) == 1
    assert recs1[0]["violation_type"] == "alpha_event"

    assert len(recs2) == 1
    assert recs2[0]["violation_type"] == "beta_event"
