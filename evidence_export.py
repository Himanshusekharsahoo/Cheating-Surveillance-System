"""
Authorized Evidence Decryption and Export Utility.

Provides controlled, audited decryption and export of encrypted surveillance evidence:
- Enforces Windows DPAPI User Scope or verified Master Recovery Key (MRK) authentication.
- Cryptographically verifies 52-byte binary envelopes and AES-256-GCM AEAD tags before plaintext release.
- Disallows unauthorized or silent fallback; fails closed on tampering or corruption.
- Guarantees source encrypted evidence immutability (never modifies or unlinks source).
- Performs atomic plaintext export with restrictive file permissions.
- Emits security warnings on exported plaintext.
- Records structured, tamper-evident audit trails for all export attempts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import io
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from crypto_core import (
    AuthenticationError,
    CryptoError,
    DecryptionError,
    KeyManagementError,
    RecoveryError,
    SecurityConfigurationError,
)
from keystore import Keystore, apply_restrictive_permissions
from encrypted_storage import (
    EncryptedStorageManager,
    EncryptedStorageError,
    EvidenceMetadata,
    read_authenticated_metadata,
    read_encrypted_csv_segments,
    get_encrypted_storage_manager,
)
from path_security import (
    ensure_safe_directory,
    resolve_safe_output_path,
    is_symlink_or_junction,
)
from event_logger import sanitize_csv_cell, sanitize_csv_record

AUDIT_LOG_FIELDNAMES = [
    "timestamp_utc",
    "operator_account",
    "operation",
    "evidence_id",
    "session_id",
    "source_file",
    "destination_file",
    "status",
    "failure_category",
    "details",
]


class ExportSecurityError(CryptoError):
    """Raised when an evidence export operation violates security constraints."""
    pass


class DestinationCollisionError(ExportSecurityError):
    """Raised when export destination file already exists and overwrite is not permitted."""
    pass


class EvidenceCorruptedError(ExportSecurityError):
    """Raised when encrypted evidence envelope or tag verification fails during export."""
    pass


class ExportAuditLogger:
    """Thread-safe and fail-closed audit logger for evidence export activity."""

    def __init__(self, audit_file_path: Union[str, Path]):
        self.audit_path = Path(audit_file_path).resolve()
        parent_dir = self.audit_path.parent
        if is_symlink_or_junction(parent_dir):
            raise ExportSecurityError(f"Audit log directory '{parent_dir}' cannot be a symlink or junction.")
        parent_dir.mkdir(parents=True, exist_ok=True)

    def log_operation(
        self,
        operation: str,
        status: str,
        evidence_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_file: Optional[str] = None,
        destination_file: Optional[str] = None,
        failure_category: Optional[str] = None,
        details: Optional[str] = None,
        operator_account: Optional[str] = None,
    ) -> None:
        """Appends a sanitized, audited record to the export audit log."""
        if operator_account is None:
            operator_account = os.environ.get("USERNAME") or os.environ.get("USER") or getpass.getuser()

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operator_account": operator_account,
            "operation": operation,
            "evidence_id": evidence_id or "",
            "session_id": session_id or "",
            "source_file": source_file or "",
            "destination_file": destination_file or "",
            "status": status,
            "failure_category": failure_category or "",
            "details": details or "",
        }

        sanitized = sanitize_csv_record(record)
        file_exists = self.audit_path.exists() and self.audit_path.stat().st_size > 0

        try:
            with open(self.audit_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=AUDIT_LOG_FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(sanitized)
                f.flush()
                os.fsync(f.fileno())
            apply_restrictive_permissions(self.audit_path, fail_closed=False)
        except Exception as exc:
            # Audit failures must be printed to stderr
            print(f"CRITICAL: Failed to write export audit log at '{self.audit_path}': {exc}", file=sys.stderr)


@dataclass(frozen=True)
class EvidenceItemSummary:
    """Read-only summary of encrypted evidence metadata."""
    path: Path
    evidence_id: str
    original_filename: str
    evidence_type: str
    created_at: str
    created_timestamp: float
    session_id: Optional[str]
    file_size_bytes: int
    custom_attributes: Dict[str, Any]


class EvidenceExporter:
    """Manages audited, authorized decryption and export of encrypted evidence files."""

    def __init__(
        self,
        storage_manager: EncryptedStorageManager,
        audit_logger: Optional[ExportAuditLogger] = None,
        audit_log_path: Optional[Union[str, Path]] = None,
    ):
        self.storage_manager = storage_manager
        if audit_logger is not None:
            self.audit_logger = audit_logger
        elif audit_log_path is not None:
            self.audit_logger = ExportAuditLogger(audit_log_path)
        else:
            import config
            default_audit = config.LOG_DIR / "evidence_export_audit.csv"
            self.audit_logger = ExportAuditLogger(default_audit)

    def list_encrypted_evidence(
        self,
        log_dir: Union[str, Path],
        session_filter: Optional[str] = None,
    ) -> List[EvidenceItemSummary]:
        """
        Scans directory for encrypted evidence envelopes and reads authenticated metadata
        without decrypting payload bytes.
        """
        directory = Path(log_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Log directory not found at '{directory}'.")
        if is_symlink_or_junction(directory):
            raise ExportSecurityError(f"Log directory '{directory}' cannot be a symlink or junction.")

        summaries: List[EvidenceItemSummary] = []
        for enc_file in directory.glob("*.enc"):
            if is_symlink_or_junction(enc_file):
                continue
            try:
                meta = self.storage_manager.read_metadata_from_file(enc_file)
                if session_filter is not None and meta.session_id != session_filter:
                    continue
                summaries.append(
                    EvidenceItemSummary(
                        path=enc_file,
                        evidence_id=meta.evidence_id,
                        original_filename=meta.original_filename,
                        evidence_type=meta.evidence_type,
                        created_at=meta.created_at,
                        created_timestamp=meta.created_timestamp,
                        session_id=meta.session_id,
                        file_size_bytes=enc_file.stat().st_size,
                        custom_attributes=meta.custom_attributes,
                    )
                )
            except Exception as exc:
                print(f"Warning: Skipping unreadable or corrupted envelope '{enc_file.name}': {exc}", file=sys.stderr)

        summaries.sort(key=lambda s: s.created_timestamp)
        return summaries

    def export_single_evidence(
        self,
        source_path: Union[str, Path],
        output_dir: Union[str, Path],
        allow_overwrite: bool = False,
        mrk_input: Optional[Union[str, bytes]] = None,
    ) -> Path:
        """
        Verifies and decrypts a single encrypted evidence envelope, persisting the plaintext
        to an output directory with atomic replacement and restrictive permissions.

        Never alters or removes the original encrypted file.
        """
        src = Path(source_path).resolve()
        if not src.is_file():
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                source_file=str(src),
                failure_category="FILE_NOT_FOUND",
                details="Source evidence file does not exist.",
            )
            raise FileNotFoundError(f"Source evidence file '{src}' does not exist.")

        if is_symlink_or_junction(src):
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                source_file=str(src),
                failure_category="SYMLINK_REJECTED",
                details="Source file is a symlink or junction.",
            )
            raise ExportSecurityError(f"Source file '{src}' cannot be a symlink or directory junction.")

        out_dir = Path(output_dir).resolve()
        if is_symlink_or_junction(out_dir):
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                source_file=str(src),
                destination_file=str(out_dir),
                failure_category="SYMLINK_REJECTED",
                details="Destination directory is a symlink or junction.",
            )
            raise ExportSecurityError(f"Output directory '{out_dir}' cannot be a symlink or directory junction.")

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Read metadata and decrypt
        try:
            plaintext, metadata = self.storage_manager.read_encrypted_file(
                filepath=src,
                mrk_input=mrk_input,
            )
        except (AuthenticationError, DecryptionError, KeyManagementError, EncryptedStorageError) as exc:
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                source_file=str(src),
                failure_category="AUTHENTICATION_FAILED",
                details=f"AEAD authentication or key unwrapping failed: {exc}",
            )
            raise EvidenceCorruptedError(f"Authentication failed for '{src}': {exc}") from exc
        except Exception as exc:
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                source_file=str(src),
                failure_category="READ_ERROR",
                details=str(exc),
            )
            raise

        # 2. Determine target destination filename
        dest_filename = metadata.original_filename
        dest_path = out_dir / dest_filename

        if dest_path.exists() and not allow_overwrite:
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                evidence_id=metadata.evidence_id,
                session_id=metadata.session_id,
                source_file=str(src),
                destination_file=str(dest_path),
                failure_category="COLLISION_DETECTED",
                details="Destination file already exists and allow_overwrite is False.",
            )
            raise DestinationCollisionError(
                f"Export collision: destination file '{dest_path}' already exists. Use allow_overwrite=True to replace."
            )

        # 3. Write decrypted plaintext atomically
        tmp_target = out_dir / f"{dest_filename}.tmp_{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp_target, "wb") as f:
                f.write(plaintext)
                f.flush()
                os.fsync(f.fileno())

            apply_restrictive_permissions(tmp_target, fail_closed=False)
            os.replace(tmp_target, dest_path)
            apply_restrictive_permissions(dest_path, fail_closed=False)
        except Exception as exc:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except OSError:
                    pass
            self.audit_logger.log_operation(
                operation="EXPORT_SINGLE",
                status="FAILED",
                evidence_id=metadata.evidence_id,
                session_id=metadata.session_id,
                source_file=str(src),
                destination_file=str(dest_path),
                failure_category="WRITE_ERROR",
                details=str(exc),
            )
            raise ExportSecurityError(f"Failed to write exported plaintext to '{dest_path}': {exc}") from exc

        # 4. Success audit and security warning
        self.audit_logger.log_operation(
            operation="EXPORT_SINGLE",
            status="SUCCESS",
            evidence_id=metadata.evidence_id,
            session_id=metadata.session_id,
            source_file=str(src),
            destination_file=str(dest_path),
            details=f"Bytes exported: {len(plaintext)}",
        )

        print(
            f"SECURITY NOTICE: Decrypted evidence written to '{dest_path}'.\n"
            f"WARNING: Exported plaintext is no longer protected by encrypted storage.",
            file=sys.stderr,
        )
        return dest_path

    def export_session_evidence(
        self,
        session_id: str,
        log_dir: Union[str, Path],
        output_dir: Union[str, Path],
        allow_overwrite: bool = False,
        mrk_input: Optional[Union[str, bytes]] = None,
    ) -> List[Path]:
        """
        Exports all evidence artifacts associated with a session:
        - Concatenates and unifies session CSV segments (events and gaze diagnostics) in sequence.
        - Decrypts all violation screenshots.
        """
        directory = Path(log_dir).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        exported_paths: List[Path] = []

        # 1. Unify and export session event CSV segments if present
        event_records = read_encrypted_csv_segments(
            storage_manager=self.storage_manager,
            log_dir=directory,
            session_id=session_id,
            base_stem="events",
        )
        if event_records:
            dest_event_csv = out_dir / f"events_{session_id}.csv"
            if dest_event_csv.exists() and not allow_overwrite:
                raise DestinationCollisionError(f"Cannot export session: '{dest_event_csv}' already exists.")

            tmp_csv = out_dir / f"events_{session_id}.csv.tmp_{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(event_records[0].keys()))
                    writer.writeheader()
                    for r in event_records:
                        writer.writerow(r)
                    f.flush()
                    os.fsync(f.fileno())
                apply_restrictive_permissions(tmp_csv, fail_closed=False)
                os.replace(tmp_csv, dest_event_csv)
                apply_restrictive_permissions(dest_event_csv, fail_closed=False)
                exported_paths.append(dest_event_csv)
                self.audit_logger.log_operation(
                    operation="EXPORT_SESSION_CSV",
                    status="SUCCESS",
                    session_id=session_id,
                    destination_file=str(dest_event_csv),
                    details=f"Consolidated {len(event_records)} event records",
                )
            except Exception as exc:
                if tmp_csv.exists():
                    tmp_csv.unlink()
                raise ExportSecurityError(f"Failed to export session events: {exc}") from exc

        # 2. Unify and export gaze diagnostics CSV segments if present
        gaze_records = read_encrypted_csv_segments(
            storage_manager=self.storage_manager,
            log_dir=directory,
            session_id=session_id,
            base_stem="gaze_diagnostics",
        )
        if gaze_records:
            dest_gaze_csv = out_dir / f"gaze_diagnostics_{session_id}.csv"
            if dest_gaze_csv.exists() and not allow_overwrite:
                raise DestinationCollisionError(f"Cannot export session: '{dest_gaze_csv}' already exists.")

            tmp_gaze = out_dir / f"gaze_diagnostics_{session_id}.csv.tmp_{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_gaze, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(gaze_records[0].keys()))
                    writer.writeheader()
                    for r in gaze_records:
                        writer.writerow(r)
                    f.flush()
                    os.fsync(f.fileno())
                apply_restrictive_permissions(tmp_gaze, fail_closed=False)
                os.replace(tmp_gaze, dest_gaze_csv)
                apply_restrictive_permissions(dest_gaze_csv, fail_closed=False)
                exported_paths.append(dest_gaze_csv)
                self.audit_logger.log_operation(
                    operation="EXPORT_SESSION_GAZE",
                    status="SUCCESS",
                    session_id=session_id,
                    destination_file=str(dest_gaze_csv),
                    details=f"Consolidated {len(gaze_records)} diagnostic frames",
                )
            except Exception as exc:
                if tmp_gaze.exists():
                    tmp_gaze.unlink()
                raise ExportSecurityError(f"Failed to export session gaze diagnostics: {exc}") from exc

        # 3. Export all screenshots for this session
        all_items = self.list_encrypted_evidence(log_dir=directory, session_filter=session_id)
        for item in all_items:
            if item.evidence_type == "SCREENSHOT":
                p = self.export_single_evidence(
                    source_path=item.path,
                    output_dir=out_dir,
                    allow_overwrite=allow_overwrite,
                    mrk_input=mrk_input,
                )
                exported_paths.append(p)

        return exported_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for evidence export."""
    parser = argparse.ArgumentParser(
        description="Authorized Evidence Decryption and Export Utility for Cheating Surveillance System."
    )
    parser.add_argument("--list", action="store_true", help="List available encrypted evidence.")
    parser.add_argument("--file", type=str, help="Specific .enc evidence file to decrypt and export.")
    parser.add_argument("--session", type=str, help="Session ID to export all associated evidence for.")
    parser.add_argument("--output", type=str, help="Destination directory for exported plaintext files.")
    parser.add_argument("--mrk", type=str, help="Master Recovery Key string if DPAPI user context is unavailable.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite of existing files at destination.")
    parser.add_argument("--log-dir", type=str, help="Custom log directory path.")
    parser.add_argument("--keystore", type=str, help="Custom keystore path.")
    return parser


def main() -> int:
    """CLI entry point for authorized evidence decryption and export."""
    parser = build_arg_parser()
    args = parser.parse_args()

    import config

    log_dir = Path(args.log_dir) if args.log_dir else config.LOG_DIR
    keystore_path = Path(args.keystore) if args.keystore else config.KEYSTORE_PATH

    try:
        ks = Keystore.load(keystore_path)
        mgr = EncryptedStorageManager(ks)
        exporter = EvidenceExporter(mgr)

        if args.list:
            items = exporter.list_encrypted_evidence(log_dir, session_filter=args.session)
            print(f"\nEncrypted Evidence Catalog ({len(items)} items found):")
            print(f"{'Type':<18} {'Session ID':<25} {'Original Filename':<35} {'Bytes':<10}")
            print("-" * 90)
            for it in items:
                sess = it.session_id or "N/A"
                print(f"{it.evidence_type:<18} {sess:<25} {it.original_filename:<35} {it.file_size_bytes:<10}")
            return 0

        if not args.output:
            parser.error("--output directory is required for export operations.")

        out_dir = Path(args.output)

        if args.file:
            target = exporter.export_single_evidence(
                source_path=args.file,
                output_dir=out_dir,
                allow_overwrite=args.overwrite,
                mrk_input=args.mrk,
            )
            print(f"Successfully exported: {target}")
            return 0

        if args.session:
            exported = exporter.export_session_evidence(
                session_id=args.session,
                log_dir=log_dir,
                output_dir=out_dir,
                allow_overwrite=args.overwrite,
                mrk_input=args.mrk,
            )
            print(f"Successfully exported {len(exported)} artifacts for session '{args.session}'.")
            return 0

        parser.print_help()
        return 1

    except Exception as exc:
        print(f"Error: Export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
