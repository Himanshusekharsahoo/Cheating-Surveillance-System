"""
Evidence Privacy, Storage Protection, and Retention Management Module.

Provides:
- Inventory and categorization of surveillance evidence and diagnostic logs.
- Safe containment and symlink/junction rejection for evidence management.
- Configurable, conservative evidence retention policy (default: 30 days).
- Dry-run audit and idempotent purge operations.
- Protection for calibration profiles, user input videos, and repository anchors.
- Zero-exposure logging that prevents candidate data or secret leakage.
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

from path_security import is_symlink_or_junction, ensure_safe_directory

# Recognized filename patterns for evidence screenshots
EVIDENCE_SCREENSHOT_PATTERN = re.compile(
    r"^(eye|head|mobile|missing_face|gaze_occluded|multiple_faces)(?:_.*)?_(\d{9,12})\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)

# Recognized filename patterns for diagnostic and event logs
EVIDENCE_LOG_PATTERN = re.compile(
    r"^(events|gaze_diagnostics|surveillance)(_v\d+)?.*\.csv$",
    re.IGNORECASE,
)

CALIBRATION_PATTERN = re.compile(
    r"^screen_gaze_calibration.*\.json$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceItem:
    """Represents a discovered evidence or log artifact."""
    path: Path
    relative_name: str
    category: str
    timestamp: float
    size_bytes: int
    is_sensitive: bool
    is_expired: bool
    is_protected: bool
    status_reason: str


@dataclass
class RetentionAuditResult:
    """Detailed summary of an evidence retention audit."""
    evidence_dir: Path
    total_files_scanned: int
    total_bytes_scanned: int
    total_encrypted_bytes: int = 0
    active_files: List[EvidenceItem] = field(default_factory=list)
    expired_files: List[EvidenceItem] = field(default_factory=list)
    protected_files: List[EvidenceItem] = field(default_factory=list)
    corrupt_or_unverified_files: List[EvidenceItem] = field(default_factory=list)
    skipped_suspicious_files: List[EvidenceItem] = field(default_factory=list)
    cutoff_timestamp: float = 0.0
    retention_days: int = 30


@dataclass
class RetentionPurgeResult:
    """Results of an evidence retention purge operation."""
    evidence_dir: Path
    dry_run: bool
    enabled: bool
    files_deleted: List[str] = field(default_factory=list)
    bytes_reclaimed: int = 0
    failures: List[str] = field(default_factory=list)
    audit_summary: Optional[RetentionAuditResult] = None


def extract_evidence_timestamp(file_path: Path) -> Tuple[float, str]:
    """
    Extracts timestamp from authenticated envelope metadata, filename, or filesystem mtime.

    Returns:
        (timestamp_float, source_description)
    """
    filename = file_path.name

    # Encrypted evidence: extract timestamp directly from authenticated envelope metadata
    if filename.endswith(".enc"):
        try:
            from encrypted_storage import read_authenticated_metadata
            envelope_bytes = file_path.read_bytes()
            meta = read_authenticated_metadata(envelope_bytes)
            if hasattr(meta, "created_timestamp") and math.isfinite(meta.created_timestamp) and meta.created_timestamp > 0:
                return float(meta.created_timestamp), "envelope_authenticated_metadata"
        except Exception:
            return 0.0, "corrupt_metadata"

    match = EVIDENCE_SCREENSHOT_PATTERN.match(filename)
    if match:
        try:
            ts = float(match.group(2))
            if 1000000000.0 <= ts <= 3000000000.0:
                return ts, "filename_timestamp"
        except (ValueError, OverflowError):
            pass

    try:
        st = file_path.stat()
        return float(st.st_mtime), "filesystem_mtime"
    except (OSError, ValueError):
        return 0.0, "unknown"


def categorize_evidence_file(
    file_path: Path,
    preserve_calibration: bool = True,
) -> Tuple[str, bool, bool, str]:
    """
    Categorizes an evidence file by type, sensitivity, and protection status.

    Returns:
        (category, is_sensitive, is_protected, reason)
    """
    name = file_path.name

    # Git keep files must never be removed
    if name == ".gitkeep":
        return "repo_anchor", False, True, "Preserved git repository anchor"

    # Encrypted evidence envelopes (Phase 10.13.28.5)
    if name.endswith(".enc"):
        try:
            from encrypted_storage import read_authenticated_metadata
            envelope_bytes = file_path.read_bytes()
            meta = read_authenticated_metadata(envelope_bytes)
            if meta.evidence_type == "SCREENSHOT":
                return "encrypted_screenshot", True, False, "Encrypted violation screenshot"
            elif meta.evidence_type in ("EVENT_LOG", "GAZE_DIAGNOSTICS"):
                return "encrypted_csv_log", True, False, f"Encrypted {meta.evidence_type.lower()} segment"
            else:
                return "encrypted_evidence", True, False, f"Encrypted evidence ({meta.evidence_type})"
        except Exception as exc:
            # Metadata cannot be authenticated or envelope is corrupt
            # Section 5.2: "If metadata cannot be authenticated: Do not purge the file. Mark it for review."
            return "corrupt_or_unverified", True, True, f"Unverified envelope metadata: {exc}"

    # Calibration profiles
    if CALIBRATION_PATTERN.match(name):
        return (
            "calibration_profile",
            False,
            preserve_calibration,
            "Active screen gaze calibration profile" if preserve_calibration else "Candidate for cleanup",
        )

    # Violation evidence screenshots
    if EVIDENCE_SCREENSHOT_PATTERN.match(name):
        return "violation_screenshot", True, False, "Candidate violation image evidence"

    # Event and diagnostic CSV logs
    if EVIDENCE_LOG_PATTERN.match(name):
        return "event_csv_log", True, False, "Surveillance incident or diagnostics log"

    # Generic log or image files in log directory
    suffix = file_path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg"):
        return "unclassified_screenshot", True, False, "Unclassified image in log directory"
    if suffix in (".csv", ".log"):
        return "unclassified_log", True, False, "Unclassified log in log directory"

    # Protect video inputs if stored locally
    if suffix in (".mp4", ".avi", ".mov", ".mkv"):
        return "video_recording", False, True, "Protected video media file"

    return "miscellaneous", False, True, "Unrecognized file type protected by default"


class EvidenceRetentionManager:
    """
    Manages audit and controlled lifecycle retention for surveillance evidence.
    """

    def __init__(
        self,
        evidence_dir: Union[str, Path],
        retention_days: int = 30,
        enabled: bool = False,
        preserve_calibration: bool = True,
    ):
        self.evidence_dir = Path(evidence_dir).resolve()
        if retention_days < 1:
            raise ValueError(f"retention_days must be >= 1, got {retention_days}")
        self.retention_days = retention_days
        self.enabled = enabled
        self.preserve_calibration = preserve_calibration

    def audit(
        self,
        retention_days: Optional[int] = None,
        current_time: Optional[float] = None,
    ) -> RetentionAuditResult:
        """
        Conducts a strictly read-only audit of evidence files in evidence_dir.
        """
        days = retention_days if retention_days is not None else self.retention_days
        now = current_time if current_time is not None else time.time()
        cutoff = now - (days * 86400.0)

        result = RetentionAuditResult(
            evidence_dir=self.evidence_dir,
            total_files_scanned=0,
            total_bytes_scanned=0,
            cutoff_timestamp=cutoff,
            retention_days=days,
        )

        if not self.evidence_dir.is_dir():
            return result

        # Check evidence root safety
        if is_symlink_or_junction(self.evidence_dir):
            return result

        try:
            entries = list(self.evidence_dir.iterdir())
        except OSError:
            return result

        for entry in entries:
            # Ignore subdirectories during standard top-level scan
            if not entry.is_file():
                continue

            # Security check 1: Reject symlinks and Windows junctions
            if is_symlink_or_junction(entry):
                item = EvidenceItem(
                    path=entry,
                    relative_name=entry.name,
                    category="reparse_point",
                    timestamp=0.0,
                    size_bytes=0,
                    is_sensitive=False,
                    is_expired=False,
                    is_protected=True,
                    status_reason="Skipped: reparse point or symlink detected",
                )
                result.skipped_suspicious_files.append(item)
                continue

            # Security check 2: Strict path containment verification
            try:
                resolved_entry = entry.resolve()
                if not resolved_entry.is_relative_to(self.evidence_dir):
                    item = EvidenceItem(
                        path=entry,
                        relative_name=entry.name,
                        category="traversal_anomaly",
                        timestamp=0.0,
                        size_bytes=0,
                        is_sensitive=False,
                        is_expired=False,
                        is_protected=True,
                        status_reason="Skipped: resolves outside evidence directory",
                    )
                    result.skipped_suspicious_files.append(item)
                    continue
            except (OSError, ValueError):
                continue

            # Size and timestamp
            try:
                size = entry.stat().st_size
            except (OSError, ValueError):
                size = 0

            file_ts, ts_source = extract_evidence_timestamp(entry)
            category, is_sensitive, is_protected, reason = categorize_evidence_file(
                entry,
                preserve_calibration=self.preserve_calibration,
            )

            result.total_files_scanned += 1
            result.total_bytes_scanned += size
            if entry.name.endswith(".enc"):
                result.total_encrypted_bytes += size

            # Clock anomaly / future timestamp check
            if file_ts > (now + 86400.0):
                item = EvidenceItem(
                    path=entry,
                    relative_name=entry.name,
                    category=category,
                    timestamp=file_ts,
                    size_bytes=size,
                    is_sensitive=is_sensitive,
                    is_expired=False,
                    is_protected=True,
                    status_reason="Suspicious: future timestamp detected",
                )
                result.skipped_suspicious_files.append(item)
                continue

            # Check expiration
            is_expired = (file_ts < cutoff) and not is_protected

            item = EvidenceItem(
                path=entry,
                relative_name=entry.name,
                category=category,
                timestamp=file_ts,
                size_bytes=size,
                is_sensitive=is_sensitive,
                is_expired=is_expired,
                is_protected=is_protected,
                status_reason=reason,
            )

            if category == "corrupt_or_unverified":
                result.corrupt_or_unverified_files.append(item)
                result.protected_files.append(item)
            elif is_protected:
                result.protected_files.append(item)
            elif is_expired:
                result.expired_files.append(item)
            else:
                result.active_files.append(item)

        return result

    def purge_expired(
        self,
        retention_days: Optional[int] = None,
        dry_run: bool = True,
        force: bool = False,
        current_time: Optional[float] = None,
    ) -> RetentionPurgeResult:
        """
        Executes a retention purge of expired evidence.

        Safety controls:
        - If dry_run is True or self.enabled is False (unless force=True):
          reports candidate deletions without removing any files.
        - Deletes strictly expired, unprotected files within self.evidence_dir.
        - Verifies containment and symlink safety immediately prior to os.unlink.
        """
        effective_dry_run = dry_run or (not self.enabled and not force)
        audit_res = self.audit(retention_days=retention_days, current_time=current_time)

        purge_result = RetentionPurgeResult(
            evidence_dir=self.evidence_dir,
            dry_run=effective_dry_run,
            enabled=self.enabled or force,
            audit_summary=audit_res,
        )

        for item in audit_res.expired_files:
            file_path = item.path
            filename = item.relative_name

            if effective_dry_run:
                purge_result.files_deleted.append(filename)
                purge_result.bytes_reclaimed += item.size_bytes
                continue

            # Pre-unlink safety verification
            try:
                if is_symlink_or_junction(file_path):
                    purge_result.failures.append(f"{filename}: symlink detected prior to deletion")
                    continue
                if not file_path.resolve().is_relative_to(self.evidence_dir):
                    purge_result.failures.append(f"{filename}: path traversal anomaly")
                    continue

                os.unlink(file_path)
                purge_result.files_deleted.append(filename)
                purge_result.bytes_reclaimed += item.size_bytes
            except OSError as err:
                purge_result.failures.append(f"{filename}: deletion error ({err.strerror})")

        return purge_result
