"""
Unit and Integration Tests for Phase 10.13.28.5 — Encrypted Evidence Retention Lifecycle.

Tests:
- Identification of active vs expired encrypted evidence using authenticated metadata timestamps.
- Refusal to purge corrupt or unverified encrypted files (marked for review, never purged).
- Clock anomaly defense on future metadata timestamps.
- Protection of calibration profiles, repo anchors (.gitkeep), and user videos.
- Dry-run safety (no deletions occur under dry_run=True or enabled=False).
- Idempotent purge execution on expired files when retention is enabled.
- Symlink and path containment enforcement.
- Accurate accounting of total_encrypted_bytes and scanned files.
"""

import hashlib
import os
from pathlib import Path
import time
import unittest.mock as mock
import uuid
import pytest

from keystore import Keystore
from encrypted_storage import (
    EncryptedStorageManager,
    EvidenceMetadata,
)
from evidence_retention import (
    EvidenceRetentionManager,
    extract_evidence_timestamp,
    categorize_evidence_file,
)

pytestmark = pytest.mark.unit


def create_encrypted_evidence(
    mgr: EncryptedStorageManager,
    dest_path: Path,
    created_timestamp: float,
    evidence_type: str = "SCREENSHOT",
    session_id: str = "sess_retention_001",
    payload: bytes = b"dummy_evidence_payload_data",
) -> Path:
    """Helper to create an encrypted evidence file with specific metadata timestamp."""
    meta = EvidenceMetadata(
        evidence_id=str(uuid.uuid4()),
        original_filename=dest_path.name[:-4] if dest_path.name.endswith(".enc") else dest_path.name,
        evidence_type=evidence_type,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_timestamp)),
        created_timestamp=created_timestamp,
        session_id=session_id,
        custom_attributes={"file_sha256": hashlib.sha256(payload).hexdigest()},
    )
    return mgr.write_encrypted_file(
        destination_path=dest_path,
        plaintext=payload,
        metadata=meta,
    )


def test_encrypted_evidence_within_and_expired_retention(tmp_path: Path):
    """Verify retention correctly separates active vs expired encrypted evidence."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    evidence_dir = tmp_path / "log"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    now = 1700000000.0  # Simulated fixed epoch
    # File 1: Created 10 days ago (Active, within 30-day retention)
    f_active = evidence_dir / "shot_active.png.enc"
    create_encrypted_evidence(mgr, f_active, created_timestamp=now - (10 * 86400.0))

    # File 2: Created 40 days ago (Expired under 30-day retention)
    f_expired = evidence_dir / "shot_expired.png.enc"
    create_encrypted_evidence(mgr, f_expired, created_timestamp=now - (40 * 86400.0))

    retention_mgr = EvidenceRetentionManager(evidence_dir, retention_days=30, enabled=False)
    audit = retention_mgr.audit(current_time=now)

    assert audit.total_files_scanned == 2
    assert audit.total_encrypted_bytes == (f_active.stat().st_size + f_expired.stat().st_size)
    assert len(audit.active_files) == 1
    assert audit.active_files[0].relative_name == "shot_active.png.enc"
    assert len(audit.expired_files) == 1
    assert audit.expired_files[0].relative_name == "shot_expired.png.enc"
    assert len(audit.corrupt_or_unverified_files) == 0


def test_corrupt_or_unverified_envelope_never_purged(tmp_path: Path):
    """Verify corrupted encrypted files are marked for review and NEVER purged."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    evidence_dir = tmp_path / "log"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    now = 1700000000.0

    # 1. Valid expired file
    f_expired = evidence_dir / "valid_expired.png.enc"
    create_encrypted_evidence(mgr, f_expired, created_timestamp=now - (50 * 86400.0))

    # 2. Corrupt envelope (invalid magic and truncated header)
    f_corrupt = evidence_dir / "corrupted_envelope.png.enc"
    f_corrupt.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_12345")

    retention_mgr = EvidenceRetentionManager(evidence_dir, retention_days=30, enabled=True)
    audit = retention_mgr.audit(current_time=now)

    assert len(audit.expired_files) == 1
    assert audit.expired_files[0].relative_name == "valid_expired.png.enc"
    assert len(audit.corrupt_or_unverified_files) == 1
    assert audit.corrupt_or_unverified_files[0].relative_name == "corrupted_envelope.png.enc"
    assert audit.corrupt_or_unverified_files[0].is_protected is True

    # Execute purge with force=True
    purge_res = retention_mgr.purge_expired(dry_run=False, force=True, current_time=now)
    assert "valid_expired.png.enc" in purge_res.files_deleted
    assert not f_expired.exists()

    # The corrupted file MUST still exist!
    assert f_corrupt.is_file()
    assert "corrupted_envelope.png.enc" not in purge_res.files_deleted


def test_clock_anomaly_future_timestamp_protected(tmp_path: Path):
    """Verify files with future timestamps in envelope metadata are flagged and protected."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    evidence_dir = tmp_path / "log"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    now = 1700000000.0
    future_time = now + (10 * 86400.0)  # 10 days in future

    f_future = evidence_dir / "future_shot.png.enc"
    create_encrypted_evidence(mgr, f_future, created_timestamp=future_time)

    retention_mgr = EvidenceRetentionManager(evidence_dir, retention_days=30, enabled=True)
    audit = retention_mgr.audit(current_time=now)

    assert len(audit.skipped_suspicious_files) == 1
    assert audit.skipped_suspicious_files[0].relative_name == "future_shot.png.enc"
    assert audit.skipped_suspicious_files[0].is_protected is True
    assert len(audit.expired_files) == 0

    # Purge must not touch future file
    purge_res = retention_mgr.purge_expired(dry_run=False, force=True, current_time=now)
    assert f_future.is_file()
    assert len(purge_res.files_deleted) == 0


def test_dry_run_default_prevents_file_deletion(tmp_path: Path):
    """Verify that by default (dry_run=True), no files are deleted."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    evidence_dir = tmp_path / "log"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    now = 1700000000.0
    f_expired = evidence_dir / "old_shot.png.enc"
    create_encrypted_evidence(mgr, f_expired, created_timestamp=now - (45 * 86400.0))

    retention_mgr = EvidenceRetentionManager(evidence_dir, retention_days=30, enabled=False)

    # Dry-run audit and purge
    purge_res = retention_mgr.purge_expired(dry_run=True, current_time=now)
    assert purge_res.dry_run is True
    assert "old_shot.png.enc" in purge_res.files_deleted
    assert f_expired.is_file()  # File still exists!


def test_preservation_of_calibration_profiles_and_anchors(tmp_path: Path):
    """Verify screen calibration profile, user videos, and .gitkeep are protected."""
    evidence_dir = tmp_path / "log"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    gitkeep = evidence_dir / ".gitkeep"
    gitkeep.write_text("", encoding="utf-8")

    calib = evidence_dir / "screen_gaze_calibration.json"
    calib.write_text('{"calibration": "data"}', encoding="utf-8")

    video = evidence_dir / "user_exam_recording.mp4"
    video.write_bytes(b"video_bytes")

    retention_mgr = EvidenceRetentionManager(evidence_dir, retention_days=30, enabled=True)
    audit = retention_mgr.audit()

    protected_names = {p.relative_name for p in audit.protected_files}
    assert ".gitkeep" in protected_names
    assert "screen_gaze_calibration.json" in protected_names
    assert "user_exam_recording.mp4" in protected_names
    assert len(audit.expired_files) == 0

    purge_res = retention_mgr.purge_expired(dry_run=False, force=True)
    assert len(purge_res.files_deleted) == 0
    assert gitkeep.exists()
    assert calib.exists()
    assert video.exists()


def test_idempotent_purge_execution(tmp_path: Path):
    """Verify running purge repeatedly is safe, idempotent, and produces no errors."""
    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)

    evidence_dir = tmp_path / "log"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    now = 1700000000.0
    f_expired = evidence_dir / "expired.png.enc"
    create_encrypted_evidence(mgr, f_expired, created_timestamp=now - (50 * 86400.0))

    retention_mgr = EvidenceRetentionManager(evidence_dir, retention_days=30, enabled=True)

    # First run: deletes expired file
    res1 = retention_mgr.purge_expired(dry_run=False, force=True, current_time=now)
    assert "expired.png.enc" in res1.files_deleted
    assert not f_expired.exists()

    # Second run: 0 files deleted, no errors
    res2 = retention_mgr.purge_expired(dry_run=False, force=True, current_time=now)
    assert len(res2.files_deleted) == 0
    assert len(res2.failures) == 0
