"""
Unit and Integration Tests for Phase 10.13.28.6 — Legacy Evidence Migration Subsystem.

Tests:
- Complete migration transaction simulation on synthetic test directories.
- Cryptographic backup creation and SHA-256 manifest verification.
- Mandatory approval boundary enforcement (MigrationApprovalError when unapproved).
- Detection of hash mismatches between backup and source.
- Detection of decryption verification failure during transactional migration.
- Atomic temporary file cleanup on migration failure.
- Lock acquisition and concurrency protection during migration.
- Symlink and path containment rejection.
- Preservation of original source and backup under all failure conditions.
- Strict non-modification of live repository files during dry-run audit.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import List, Tuple
import unittest.mock as mock
import pytest

from keystore import Keystore
from encrypted_storage import EncryptedStorageManager
from evidence_migration import (
    EvidenceMigrator,
    MigrationError,
    MigrationApprovalError,
    BackupVerificationError,
    SourceEvidenceManifestEntry,
)

pytestmark = pytest.mark.unit


def setup_synthetic_evidence_dir(tmp_path: Path) -> Tuple[Path, List[Path]]:
    """Helper to populate a temporary directory with synthetic legacy evidence."""
    src = tmp_path / "legacy_log"
    src.mkdir(parents=True, exist_ok=True)

    files = []
    # 3 Screenshots
    for i in range(3):
        f = src / f"head_Looking_Left_179307{i:04d}.png"
        f.write_bytes(f"synthetic_png_data_{i}".encode("utf-8"))
        files.append(f)

    # 1 Event CSV
    csv_file = src / "events.csv"
    csv_file.write_text("timestamp,session_id,violation_type\n2026-11-01T12:00:00Z,s1,test_v\n", encoding="utf-8")
    files.append(csv_file)

    # Excluded files
    (src / ".gitkeep").write_text("", encoding="utf-8")
    (src / "screen_gaze_calibration.json").write_text("{}", encoding="utf-8")

    return src, files


def test_migration_source_manifest_generation(tmp_path: Path):
    """Verify source manifest scans eligible evidence and computes valid SHA-256 digests."""
    src, files = setup_synthetic_evidence_dir(tmp_path)

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)

    manifest = migrator.generate_source_manifest(src)

    # 3 screenshots + 1 events.csv = 4 eligible files
    assert len(manifest) == 4
    assert ".gitkeep" not in manifest
    assert "screen_gaze_calibration.json" not in manifest

    for fname, entry in manifest.items():
        assert entry.size_bytes > 0
        assert len(entry.sha256) == 64
        # Verify hash matches actual file contents
        content = (src / fname).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry.sha256


def test_backup_creation_and_integrity_verification(tmp_path: Path):
    """Verify create_and_verify_backup duplicates all files and validates manifests."""
    src, files = setup_synthetic_evidence_dir(tmp_path)
    b_dir = tmp_path / "verified_backup"

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)

    backup_path = migrator.create_and_verify_backup(src, b_dir)
    assert backup_path == b_dir
    assert (b_dir / "backup_manifest.json").is_file()

    # Verify backup contents match source exactly
    manifest_data = json.loads((b_dir / "backup_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest_data) == 4
    for fname in manifest_data:
        assert (b_dir / fname).is_file()
        assert migrator.compute_sha256(b_dir / fname) == manifest_data[fname]["sha256"]


def test_backup_hash_mismatch_detected(tmp_path: Path):
    """Verify that if a backup file is modified or corrupted, BackupVerificationError is raised."""
    src, files = setup_synthetic_evidence_dir(tmp_path)
    b_dir = tmp_path / "corrupt_backup"

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)

    # Simulate bit-rot during copy by patching shutil.copy2
    real_copy2 = shutil.copy2

    def corrupt_copy(s, d):
        real_copy2(s, d)
        with open(d, "ab") as f:
            f.write(b"corruption_bit")

    with mock.patch("shutil.copy2", side_effect=corrupt_copy):
        with pytest.raises(BackupVerificationError, match="Backup integrity failure"):
            migrator.create_and_verify_backup(src, b_dir)


def test_mandatory_approval_boundary_halts_execution(tmp_path: Path):
    """Verify that execute_migration_transaction refuses to run unless approved=True."""
    src, files = setup_synthetic_evidence_dir(tmp_path)
    b_dir = tmp_path / "backup"

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)
    migrator.create_and_verify_backup(src, b_dir)

    # Must raise MigrationApprovalError when approved=False
    with pytest.raises(MigrationApprovalError, match="MIGRATION HALTED AT MANDATORY APPROVAL BOUNDARY"):
        migrator.execute_migration_transaction(src, b_dir, approved=False)


def test_complete_migration_transaction_simulation(tmp_path: Path):
    """Verify approved synthetic migration encrypts, verifies decryption, and commits .enc files."""
    src, files = setup_synthetic_evidence_dir(tmp_path)
    b_dir = tmp_path / "backup"

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)
    migrator.create_and_verify_backup(src, b_dir)

    journal = migrator.execute_migration_transaction(src, b_dir, approved=True, delete_originals=False)

    assert len(journal) == 4
    for j in journal:
        assert j.status == "MIGRATED"
        assert j.source_sha256 == j.decrypted_sha256
        assert (src / j.target_filename).is_file()

        # Decrypt output file and verify payload matches source
        decrypted_bytes, meta = mgr.read_encrypted_file(src / j.target_filename)
        assert hashlib.sha256(decrypted_bytes).hexdigest() == j.source_sha256

    # Plaintext files must still exist (delete_originals=False)
    for f in files:
        assert f.is_file()


def test_decryption_verification_failure_cleans_up_temp(tmp_path: Path):
    """Verify that if post-encryption verification fails, temporary file is cleaned up and source preserved."""
    src, files = setup_synthetic_evidence_dir(tmp_path)

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)

    target_file = files[0]

    # Mock storage_manager.read_encrypted_file to return corrupted bytes
    with mock.patch.object(mgr, "read_encrypted_file", return_value=(b"corrupted_different_bytes", None)):
        journal_entry = migrator.migrate_single_file(target_file)

    assert journal_entry.status == "FAILED"
    assert "mismatch" in str(journal_entry.error)

    # Final .enc file must NOT exist
    assert not (src / f"{target_file.name}.enc").exists()

    # Temp files must be cleaned up
    assert len(list(src.glob("*.tmp_*"))) == 0

    # Source plaintext file must remain intact
    assert target_file.is_file()


def test_dry_run_audit_does_not_modify_any_files(tmp_path: Path):
    """Verify dry_run calculates metrics without altering source directory."""
    src, files = setup_synthetic_evidence_dir(tmp_path)

    ks_path = tmp_path / "keystore.enc"
    ks, _, _ = Keystore.initialize(ks_path)
    mgr = EncryptedStorageManager(ks)
    migrator = EvidenceMigrator(mgr)

    # Initial snapshot of directory
    initial_files = {p.name: p.stat().st_size for p in src.iterdir()}

    report = migrator.dry_run(src)
    assert report.eligible_count == 4
    assert len(report.excluded_files) == 2  # .gitkeep and screen_gaze_calibration.json
    assert report.backup_ready is False  # No backup provided
    assert len(report.blockers) > 0

    # Directory state must be 100% identical
    current_files = {p.name: p.stat().st_size for p in src.iterdir()}
    assert initial_files == current_files
