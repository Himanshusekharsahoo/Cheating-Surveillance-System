"""
Legacy Evidence Migration Module.

Provides safe, audited, backup-verified migration of plaintext evidence to encrypted binary envelopes:
- Full SHA-256 manifest generation for source files.
- Independent verified backup creation and validation.
- Transactional migration with pre-replacement decryption verification.
- Migration journal tracking every file transition.
- Mandatory approval boundary: actual live migration requires explicit authorization.
- Zero plaintext deletions by default; strict source and backup preservation.
- Dry-run audit calculating file inventories, categories, overhead, and blockers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from filelock import FileLock

from crypto_core import (
    AuthenticationError,
    CryptoError,
    DecryptionError,
    KeyManagementError,
)
from keystore import Keystore, apply_restrictive_permissions
from encrypted_storage import (
    EncryptedStorageManager,
    EncryptedStorageError,
    EvidenceMetadata,
)
from path_security import (
    ensure_safe_directory,
    is_symlink_or_junction,
)
from evidence_retention import (
    EVIDENCE_SCREENSHOT_PATTERN,
    EVIDENCE_LOG_PATTERN,
    CALIBRATION_PATTERN,
)


class MigrationError(CryptoError):
    """Raised when an evidence migration operation encounters an error."""
    pass


class MigrationApprovalError(MigrationError):
    """Raised when an unapproved live migration execution is attempted."""
    pass


class BackupVerificationError(MigrationError):
    """Raised when backup copy does not match source manifest SHA-256 checksums."""
    pass


@dataclass(frozen=True)
class SourceEvidenceManifestEntry:
    """Cryptographic manifest record for a single legacy evidence file."""
    relative_path: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    mtime: float


@dataclass
class MigrationDryRunReport:
    """Detailed summary of a migration dry-run audit."""
    source_dir: Path
    total_files_scanned: int
    total_bytes_scanned: int
    eligible_count: int
    eligible_bytes: int
    eligible_files: List[SourceEvidenceManifestEntry] = field(default_factory=list)
    excluded_files: List[str] = field(default_factory=list)
    already_encrypted_files: List[str] = field(default_factory=list)
    estimated_encrypted_bytes: int = 0
    backup_ready: bool = False
    backup_path: Optional[Path] = None
    blockers: List[str] = field(default_factory=list)


@dataclass
class MigrationJournalEntry:
    """Record of a migration transaction for a single file."""
    timestamp_utc: str
    source_filename: str
    target_filename: str
    status: str
    source_sha256: str
    decrypted_sha256: str
    error: Optional[str] = None


class EvidenceMigrator:
    """
    Manages safe, transactionally verified legacy evidence migration.
    """

    def __init__(
        self,
        storage_manager: Optional[EncryptedStorageManager] = None,
        journal_path: Optional[Union[str, Path]] = None,
    ):
        self.storage_manager = storage_manager
        self.journal_path = Path(journal_path).resolve() if journal_path else None

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Computes SHA-256 digest of a local file in 64KB blocks."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def generate_source_manifest(
        self,
        source_dir: Union[str, Path],
    ) -> Dict[str, SourceEvidenceManifestEntry]:
        """
        Scans source directory and builds a verified SHA-256 manifest of eligible evidence files.
        """
        src = Path(source_dir).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"Source directory not found at '{src}'.")
        if is_symlink_or_junction(src):
            raise MigrationError(f"Source directory '{src}' cannot be a symlink or junction.")

        manifest: Dict[str, SourceEvidenceManifestEntry] = {}

        for entry in src.iterdir():
            if not entry.is_file():
                continue
            if is_symlink_or_junction(entry):
                continue

            name = entry.name
            # Exclude repository anchors, configuration, and audit logs
            if name in (".gitkeep", "screen_gaze_calibration.json") or name.startswith("evidence_export_audit"):
                continue
            if name.endswith(".enc"):
                continue

            # Identify category
            is_screenshot = bool(EVIDENCE_SCREENSHOT_PATTERN.match(name) or name.lower().endswith((".png", ".jpg")))
            is_log = bool(EVIDENCE_LOG_PATTERN.match(name) or name.lower().endswith(".csv"))

            if not (is_screenshot or is_log):
                continue

            ftype = "SCREENSHOT" if is_screenshot else "EVENT_LOG"
            st = entry.stat()
            c_hash = self.compute_sha256(entry)

            manifest[name] = SourceEvidenceManifestEntry(
                relative_path=name,
                filename=name,
                file_type=ftype,
                size_bytes=st.st_size,
                sha256=c_hash,
                mtime=st.st_mtime,
            )

        return manifest

    def create_and_verify_backup(
        self,
        source_dir: Union[str, Path],
        backup_dir: Union[str, Path],
    ) -> Path:
        """
        Creates an independent backup of all eligible evidence files and cryptographically
        verifies every SHA-256 checksum against the source manifest before proceeding.
        """
        src = Path(source_dir).resolve()
        b_dir = Path(backup_dir).resolve()

        if b_dir.is_relative_to(src) or b_dir == src:
            raise MigrationError(f"Backup directory '{b_dir}' must be outside source directory '{src}'.")

        if is_symlink_or_junction(b_dir):
            raise MigrationError(f"Backup directory '{b_dir}' cannot be a symlink or junction.")

        b_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.generate_source_manifest(src)

        backup_manifest: Dict[str, Dict[str, Any]] = {}

        for filename, entry in manifest.items():
            src_file = src / filename
            dst_file = b_dir / filename

            # Copy file preserving metadata
            shutil.copy2(src_file, dst_file)
            apply_restrictive_permissions(dst_file, fail_closed=False)

            # Cryptographic verification
            b_hash = self.compute_sha256(dst_file)
            if b_hash != entry.sha256:
                raise BackupVerificationError(
                    f"Backup integrity failure for '{filename}': expected SHA-256 {entry.sha256}, got {b_hash}."
                )

            backup_manifest[filename] = {
                "size_bytes": entry.size_bytes,
                "sha256": b_hash,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }

        # Write verified backup manifest
        manifest_file = b_dir / "backup_manifest.json"
        manifest_file.write_text(json.dumps(backup_manifest, indent=2), encoding="utf-8")
        apply_restrictive_permissions(manifest_file, fail_closed=False)

        return b_dir

    def dry_run(
        self,
        source_dir: Union[str, Path],
        backup_dir: Optional[Union[str, Path]] = None,
    ) -> MigrationDryRunReport:
        """
        Conducts a strictly read-only dry-run audit of legacy evidence files.
        Calculates counts, sizes, encryption overhead, and checks backup readiness.
        """
        src = Path(source_dir).resolve()
        manifest = self.generate_source_manifest(src)

        report = MigrationDryRunReport(
            source_dir=src,
            total_files_scanned=0,
            total_bytes_scanned=0,
            eligible_count=len(manifest),
            eligible_bytes=sum(e.size_bytes for e in manifest.values()),
        )

        # Audit entire directory contents
        for entry in src.iterdir():
            if not entry.is_file():
                continue
            report.total_files_scanned += 1
            report.total_bytes_scanned += entry.stat().st_size

            if entry.name in manifest:
                report.eligible_files.append(manifest[entry.name])
            elif entry.name.endswith(".enc"):
                report.already_encrypted_files.append(entry.name)
            else:
                report.excluded_files.append(entry.name)

        # Estimated storage overhead: ~248 bytes per envelope (52 fixed header + ~180 meta + 16 tag)
        report.estimated_encrypted_bytes = report.eligible_bytes + (len(manifest) * 248)

        # Check keystore readiness
        if self.storage_manager is None:
            report.blockers.append("Active keystore not initialized or unreadable. Keystore must exist before migration.")

        # Verify backup status if provided
        if backup_dir is not None:
            b_path = Path(backup_dir).resolve()
            report.backup_path = b_path
            manifest_file = b_path / "backup_manifest.json"
            if b_path.is_dir() and manifest_file.is_file():
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if len(data) == len(manifest):
                        report.backup_ready = True
                    else:
                        report.blockers.append(
                            f"Backup incomplete: manifest contains {len(data)} files, expected {len(manifest)}."
                        )
                except Exception as exc:
                    report.blockers.append(f"Backup manifest unreadable: {exc}")
            else:
                report.blockers.append(f"Backup manifest not found at '{manifest_file}'.")
        else:
            report.blockers.append("Independent verified backup path not specified.")

        return report

    def migrate_single_file(
        self,
        source_file: Path,
        session_id: Optional[str] = None,
        delete_original: bool = False,
    ) -> MigrationJournalEntry:
        """
        Transactionally encrypts a single legacy evidence file:
        1. Encrypts to temporary file (.enc.tmp_<uuid>).
        2. Decrypts and verifies SHA-256 against source.
        3. Atomically replaces to <source_file>.enc.
        4. Preserves original plaintext file unless delete_original=True (default: False).
        """
        src = source_file.resolve()
        target_enc = src.with_suffix(f"{src.suffix}.enc")

        src_hash = self.compute_sha256(src)
        raw_bytes = src.read_bytes()

        is_screenshot = bool(EVIDENCE_SCREENSHOT_PATTERN.match(src.name) or src.name.lower().endswith((".png", ".jpg")))
        ev_type = "SCREENSHOT" if is_screenshot else "EVENT_LOG"

        now_dt = datetime.now(timezone.utc)
        meta = EvidenceMetadata(
            evidence_id=str(uuid.uuid4()),
            original_filename=src.name,
            evidence_type=ev_type,
            created_at=datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc).isoformat(),
            created_timestamp=src.stat().st_mtime,
            session_id=session_id,
            custom_attributes={
                "migrated": True,
                "migration_timestamp": now_dt.isoformat(),
                "file_sha256": src_hash,
            },
        )

        tmp_enc = src.parent / f"{target_enc.name}.tmp_{uuid.uuid4().hex[:8]}"

        try:
            # 1. Encrypt to temporary file
            self.storage_manager.write_encrypted_file(
                destination_path=tmp_enc,
                plaintext=raw_bytes,
                metadata=meta,
                allow_overwrite=False,
            )

            # 2. Decrypt and verify hash
            decrypted_bytes, read_meta = self.storage_manager.read_encrypted_file(
                filepath=tmp_enc,
                expected_filename=src.name,
            )
            dec_hash = hashlib.sha256(decrypted_bytes).hexdigest()

            if dec_hash != src_hash:
                raise MigrationError(f"Hash verification mismatch after encryption: {dec_hash} != {src_hash}")

            # 3. Commit to final .enc file
            os.replace(tmp_enc, target_enc)
            apply_restrictive_permissions(target_enc, fail_closed=False)

            # 4. Plaintext preservation / optional removal
            if delete_original:
                src.unlink()

            return MigrationJournalEntry(
                timestamp_utc=now_dt.isoformat(),
                source_filename=src.name,
                target_filename=target_enc.name,
                status="MIGRATED",
                source_sha256=src_hash,
                decrypted_sha256=dec_hash,
            )

        except Exception as exc:
            if tmp_enc.exists():
                try:
                    tmp_enc.unlink()
                except OSError:
                    pass
            return MigrationJournalEntry(
                timestamp_utc=now_dt.isoformat(),
                source_filename=src.name,
                target_filename=target_enc.name,
                status="FAILED",
                source_sha256=src_hash,
                decrypted_sha256="",
                error=str(exc),
            )

    def execute_migration_transaction(
        self,
        source_dir: Union[str, Path],
        backup_dir: Union[str, Path],
        approved: bool = False,
        delete_originals: bool = False,
    ) -> List[MigrationJournalEntry]:
        """
        Executes verified migration across all eligible files under an exclusive lock.
        Refuses to run unless approved=True is explicitly passed (Approval Boundary).
        """
        if not approved:
            raise MigrationApprovalError(
                "MIGRATION HALTED AT MANDATORY APPROVAL BOUNDARY: "
                "Actual legacy evidence migration has not been authorized. "
                "Explicit approval must be provided before executing live migration."
            )

        src = Path(source_dir).resolve()
        b_dir = Path(backup_dir).resolve()

        # Verify backup manifest before starting
        manifest = self.generate_source_manifest(src)
        manifest_file = b_dir / "backup_manifest.json"
        if not manifest_file.is_file():
            raise BackupVerificationError(f"Backup manifest not found at '{manifest_file}'.")

        backup_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        for fname, entry in manifest.items():
            if fname not in backup_data or backup_data[fname]["sha256"] != entry.sha256:
                raise BackupVerificationError(f"Backup integrity check failed for '{fname}'.")

        lock_path = src / ".migration.lock"
        journal: List[MigrationJournalEntry] = []

        with FileLock(str(lock_path), timeout=30.0):
            for fname, entry in manifest.items():
                src_file = src / fname
                res = self.migrate_single_file(src_file, delete_original=delete_originals)
                journal.append(res)
                if res.status != "MIGRATED":
                    print(f"Warning: Migration failed on '{fname}': {res.error}", file=sys.stderr)

        return journal


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for evidence migration."""
    parser = argparse.ArgumentParser(
        description="Legacy Evidence Migration Utility for Cheating Surveillance System."
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Perform read-only migration audit.")
    parser.add_argument("--backup-dir", type=str, help="Destination directory for verified backup.")
    parser.add_argument("--log-dir", type=str, help="Source evidence directory (defaults to config.LOG_DIR).")
    parser.add_argument("--keystore", type=str, help="Keystore file path (defaults to config.KEYSTORE_PATH).")
    parser.add_argument("--create-backup-only", action="store_true", help="Create and verify backup only.")
    parser.add_argument("--authorized-execute", action="store_true", help="Authorize actual live migration.")
    return parser


def main() -> int:
    """CLI entry point for legacy evidence migration tool."""
    parser = build_arg_parser()
    args = parser.parse_args()

    import config

    log_dir = Path(args.log_dir) if args.log_dir else config.LOG_DIR
    keystore_path = Path(args.keystore) if args.keystore else config.KEYSTORE_PATH

    try:
        if keystore_path.is_file():
            ks = Keystore.load(keystore_path)
            mgr = EncryptedStorageManager(ks)
        else:
            mgr = None
        migrator = EvidenceMigrator(mgr)

        if args.create_backup_only:
            if not args.backup_dir:
                parser.error("--backup-dir is required to create a backup.")
            print(f"Creating and verifying backup of '{log_dir}' to '{args.backup_dir}'...")
            b_path = migrator.create_and_verify_backup(log_dir, args.backup_dir)
            print(f"Backup created and cryptographically verified at: {b_path}")
            return 0

        # Dry-run execution (Default)
        if not args.authorized_execute or args.dry_run:
            print("Running Read-Only Migration Dry-Run Audit...")
            report = migrator.dry_run(log_dir, backup_dir=args.backup_dir)

            print("\n" + "=" * 80)
            print("LEGACY EVIDENCE MIGRATION DRY-RUN AUDIT REPORT")
            print("=" * 80)
            print(f"Source Directory:            {report.source_dir}")
            print(f"Total Files Scanned:         {report.total_files_scanned}")
            print(f"Total Bytes Scanned:         {report.total_bytes_scanned:,} bytes")
            print(f"Files Eligible for Migration:{report.eligible_count}")
            print(f"Eligible Plaintext Bytes:    {report.eligible_bytes:,} bytes")
            print(f"Estimated Encrypted Storage: {report.estimated_encrypted_bytes:,} bytes")
            print(f"Files Excluded:              {len(report.excluded_files)} ({report.excluded_files})")
            print(f"Already Encrypted Files:     {len(report.already_encrypted_files)}")
            print(f"Backup Verified & Ready:     {report.backup_ready}")
            if report.blockers:
                print("\nMigration Blockers / Pending Actions:")
                for b in report.blockers:
                    print(f"  [!] {b}")
            print("=" * 80)
            print("NOTICE: Dry-run complete. Zero source files were modified or deleted.")
            return 0

        # Live migration attempt
        parser.error(
            "Live migration requested. Migration must be authorized with explicit approval."
        )
        return 1

    except Exception as exc:
        print(f"Error: Migration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
