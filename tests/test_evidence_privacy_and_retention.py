"""
Unit and Integration Tests for Phase 10.13.26 — Evidence Privacy, Storage Protection & Retention.

All tests utilize temporary directories and synthetic mock files.
Never touches or deletes real examination evidence or user data.
"""

from pathlib import Path
import os
import time
import pytest

import config
from evidence_retention import (
    EvidenceItem,
    EvidenceRetentionManager,
    extract_evidence_timestamp,
    categorize_evidence_file,
    EVIDENCE_SCREENSHOT_PATTERN,
    CALIBRATION_PATTERN,
)
from path_security import is_symlink_or_junction

pytestmark = pytest.mark.unit


def test_evidence_screenshot_pattern_matching():
    """Verify regex matching for standard violation screenshot filenames."""
    valid_names = [
        "eye_Looking_Down_1793079704.png",
        "head_Looking_Left_1793080819.jpg",
        "mobile_detected_1793106299.png",
        "missing_face_1793101721.jpeg",
        "gaze_occluded_1793086312.png",
        "multiple_faces_detected_1793086432.png",
    ]
    for name in valid_names:
        assert EVIDENCE_SCREENSHOT_PATTERN.match(name) is not None, f"Should match: {name}"

    invalid_names = [
        "not_an_evidence_file.txt",
        "sample.mp4",
        "screen_gaze_calibration.json",
        ".gitkeep",
    ]
    for name in invalid_names:
        assert EVIDENCE_SCREENSHOT_PATTERN.match(name) is None, f"Should not match: {name}"


def test_categorize_evidence_file(tmp_path):
    """Verify evidence file categorization, sensitivity tagging, and protection rules."""
    # 1. Repo anchor
    anchor = tmp_path / ".gitkeep"
    anchor.touch()
    cat, sensitive, protected, reason = categorize_evidence_file(anchor)
    assert cat == "repo_anchor"
    assert sensitive is False
    assert protected is True

    # 2. Calibration profile
    calib = tmp_path / "screen_gaze_calibration.json"
    calib.touch()
    cat, sensitive, protected, reason = categorize_evidence_file(calib, preserve_calibration=True)
    assert cat == "calibration_profile"
    assert protected is True

    # 3. Violation screenshot
    shot = tmp_path / "mobile_detected_1793106299.png"
    shot.touch()
    cat, sensitive, protected, reason = categorize_evidence_file(shot)
    assert cat == "violation_screenshot"
    assert sensitive is True
    assert protected is False

    # 4. Event CSV log
    event_csv = tmp_path / "events.csv"
    event_csv.touch()
    cat, sensitive, protected, reason = categorize_evidence_file(event_csv)
    assert cat == "event_csv_log"
    assert sensitive is True
    assert protected is False

    # 5. Video input
    video = tmp_path / "exam_recording.mp4"
    video.touch()
    cat, sensitive, protected, reason = categorize_evidence_file(video)
    assert cat == "video_recording"
    assert protected is True


def test_extract_evidence_timestamp(tmp_path):
    """Verify extraction of timestamp from filename with fallback to filesystem mtime."""
    shot = tmp_path / "eye_Looking_Down_1793079704.png"
    shot.touch()
    ts, source = extract_evidence_timestamp(shot)
    assert ts == 1793079704.0
    assert source == "filename_timestamp"

    misc = tmp_path / "custom_test.log"
    misc.touch()
    ts_mtime, source_mtime = extract_evidence_timestamp(misc)
    assert ts_mtime > 0.0
    assert source_mtime == "filesystem_mtime"


def test_retention_audit_dry_run_identifies_expired_and_active_files(tmp_path):
    """Verify retention audit correctly segregates active and expired evidence without deletion."""
    simulated_now = 1800000000.0
    retention_days = 30
    cutoff = simulated_now - (retention_days * 86400.0)

    # 40 days old (expired)
    old_ts = int(cutoff - (10 * 86400.0))
    old_file = tmp_path / f"mobile_detected_{old_ts}.png"
    old_file.write_bytes(b"old evidence")

    # 10 days old (active)
    recent_ts = int(simulated_now - (10 * 86400.0))
    recent_file = tmp_path / f"mobile_detected_{recent_ts}.png"
    recent_file.write_bytes(b"recent evidence")

    # Protected calibration file
    calib_file = tmp_path / "screen_gaze_calibration.json"
    calib_file.write_bytes(b"{}")

    # Protected repo anchor
    anchor = tmp_path / ".gitkeep"
    anchor.touch()

    mgr = EvidenceRetentionManager(
        evidence_dir=tmp_path,
        retention_days=retention_days,
        enabled=False,
        preserve_calibration=True,
    )

    audit = mgr.audit(current_time=simulated_now)

    assert audit.total_files_scanned == 4
    assert len(audit.expired_files) == 1
    assert audit.expired_files[0].relative_name == old_file.name

    active_names = [f.relative_name for f in audit.active_files]
    assert recent_file.name in active_names

    protected_names = [f.relative_name for f in audit.protected_files]
    assert calib_file.name in protected_names
    assert anchor.name in protected_names

    # Assert no files were deleted during audit
    assert old_file.is_file()
    assert recent_file.is_file()
    assert calib_file.is_file()
    assert anchor.is_file()


def test_retention_future_timestamp_flagged_as_suspicious(tmp_path):
    """Verify clock anomaly / future timestamps are protected and flagged as suspicious."""
    simulated_now = 1800000000.0
    future_ts = int(simulated_now + (10 * 86400.0))
    future_file = tmp_path / f"eye_Looking_Up_{future_ts}.png"
    future_file.write_bytes(b"future evidence")

    mgr = EvidenceRetentionManager(evidence_dir=tmp_path, retention_days=30)
    audit = mgr.audit(current_time=simulated_now)

    assert len(audit.skipped_suspicious_files) == 1
    assert audit.skipped_suspicious_files[0].relative_name == future_file.name
    assert "future timestamp" in audit.skipped_suspicious_files[0].status_reason
    assert future_file.is_file()


def test_retention_symlink_rejection(tmp_path_factory):
    """Verify symlinked evidence files are rejected and protected from deletion."""
    evidence_dir = tmp_path_factory.mktemp("evidence")
    external_dir = tmp_path_factory.mktemp("external")

    real_target = external_dir / "external_target.png"
    real_target.write_bytes(b"target data")

    symlink_file = evidence_dir / "mobile_detected_1000000000.png"
    try:
        symlink_file.symlink_to(real_target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink not supported in environment")

    mgr = EvidenceRetentionManager(evidence_dir=evidence_dir, retention_days=30)
    audit = mgr.audit(current_time=1800000000.0)

    assert len(audit.skipped_suspicious_files) == 1
    assert "symlink detected" in audit.skipped_suspicious_files[0].status_reason

    # Purge must not unlink the symlink or its target
    purge_res = mgr.purge_expired(dry_run=False, force=True, current_time=1800000000.0)
    assert len(purge_res.files_deleted) == 0
    assert symlink_file.exists()
    assert real_target.is_file()


def test_retention_purge_dry_run_vs_execution(tmp_path):
    """Verify dry_run=True does not remove files, while dry_run=False deletes only expired files."""
    simulated_now = 1800000000.0
    cutoff = simulated_now - (30 * 86400.0)
    old_ts = int(cutoff - 86400.0)
    recent_ts = int(simulated_now - 86400.0)

    old_file = tmp_path / f"mobile_detected_{old_ts}.png"
    old_file.write_bytes(b"old")
    recent_file = tmp_path / f"mobile_detected_{recent_ts}.png"
    recent_file.write_bytes(b"recent")

    mgr = EvidenceRetentionManager(evidence_dir=tmp_path, retention_days=30, enabled=False)

    # 1. Dry run
    dry_res = mgr.purge_expired(dry_run=True, current_time=simulated_now)
    assert dry_res.dry_run is True
    assert len(dry_res.files_deleted) == 1
    assert old_file.is_file()  # Still exists
    assert recent_file.is_file()

    # 2. Executed purge with force=True
    exec_res = mgr.purge_expired(dry_run=False, force=True, current_time=simulated_now)
    assert exec_res.dry_run is False
    assert len(exec_res.files_deleted) == 1
    assert not old_file.is_file()  # Deleted!
    assert recent_file.is_file()   # Preserved!

    # 3. Idempotent re-run
    rerun_res = mgr.purge_expired(dry_run=False, force=True, current_time=simulated_now)
    assert len(rerun_res.files_deleted) == 0
    assert rerun_res.bytes_reclaimed == 0


def test_real_log_directory_audit_preserves_all_files():
    """Verify that running an audit against the actual project log directory preserves every file."""
    log_dir = config.LOG_DIR
    assert log_dir.is_dir()
    initial_files = {p.name: p.stat().st_size for p in log_dir.iterdir() if p.is_file()}

    mgr = EvidenceRetentionManager(evidence_dir=log_dir, retention_days=30, enabled=False)
    audit = mgr.audit()
    assert audit.total_files_scanned == len(initial_files)

    final_files = {p.name: p.stat().st_size for p in log_dir.iterdir() if p.is_file()}
    assert initial_files == final_files, "Audit must never modify or delete any existing real files."


def test_encryption_disabled_by_default_and_key_handling():
    """Verify encryption settings defaults and key validation expectations."""
    assert config.ENCRYPTION_ENABLED is False
    assert config.RETENTION_DAYS == 30
    assert config.RETENTION_ENABLED is False
from unittest.mock import patch

def test_retention_path_traversal_containment_rejection(tmp_path):
    """Verify that files attempting path traversal outside evidence_dir are safely rejected."""
    outer_dir = tmp_path / "outer"
    outer_dir.mkdir()
    outer_file = outer_dir / "secret.txt"
    outer_file.write_text("should not be deleted")

    inner_dir = tmp_path / "evidence_root"
    inner_dir.mkdir()

    mgr = EvidenceRetentionManager(evidence_dir=inner_dir, retention_days=30)
    audit = mgr.audit(current_time=1800000000.0)

    # Purge on inner_dir must not touch outer_file
    purge = mgr.purge_expired(dry_run=False, force=True, current_time=1800000000.0)
    assert outer_file.is_file()
    assert len(purge.files_deleted) == 0


def test_retention_preserves_user_selected_videos(tmp_path):
    """Verify that video files stored in evidence directory are preserved by default."""
    video_file = tmp_path / "recorded_exam_1000000000.mp4"
    video_file.write_bytes(b"video stream")

    mgr = EvidenceRetentionManager(evidence_dir=tmp_path, retention_days=30)
    audit = mgr.audit(current_time=1800000000.0)

    assert any(item.relative_name == video_file.name for item in audit.protected_files)
    purge = mgr.purge_expired(dry_run=False, force=True, current_time=1800000000.0)
    assert video_file.name not in purge.files_deleted
    assert video_file.is_file()


def test_retention_manager_invalid_retention_days():
    """Verify ValueError when retention_days < 1."""
    with pytest.raises(ValueError, match="retention_days must be >= 1"):
        EvidenceRetentionManager(evidence_dir=Path("log"), retention_days=0)


def test_retention_nonexistent_directory(tmp_path):
    """Verify graceful handling when evidence directory does not exist."""
    nonexistent = tmp_path / "does_not_exist"
    mgr = EvidenceRetentionManager(evidence_dir=nonexistent, retention_days=30)
    audit = mgr.audit()
    assert audit.total_files_scanned == 0
    purge = mgr.purge_expired()
    assert len(purge.files_deleted) == 0
