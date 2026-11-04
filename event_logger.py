"""
Structured Event Logger for Cheating Surveillance System.

Logs potential examination rule violations to a local CSV file with ISO 8601 timestamps,
session identifiers, violation durations, optional confidence scores, and evidence references.
Hardened against CSV formula injection and filesystem path traversal attacks.
"""

import csv
from datetime import datetime
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, Optional, Union
import uuid

from path_security import resolve_safe_output_path, ensure_safe_directory

FIELDNAMES = [
    "timestamp",
    "session_id",
    "violation_type",
    "duration_sec",
    "confidence",
    "screenshot_path",
]


def is_legitimate_number(text: str) -> bool:
    """
    Checks whether a stripped string represents a legitimate numeric value (integer or float).
    Allows optional sign, decimal point, and scientific notation (e.g. '-0.5', '-12.50', '3.14').
    Rejects formulas and expressions such as '-1+1', '-cmd|...', or '-@SUM(A1)'.
    """
    s = text.strip()
    if re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", s):
        try:
            val = float(s)
            return math.isfinite(val)
        except (ValueError, OverflowError):
            return False
    return False


def sanitize_csv_cell(value: Any) -> Any:
    """
    Sanitizes a single CSV cell value against spreadsheet formula injection (CSV Injection).

    If the value is an untrusted string that begins with a dangerous formula trigger
    ('=', '+', '@', or leading control characters like tab, CR, LF), or starts with '-'
    and is NOT a legitimate pure number, it is prefixed with a single quote (') to force
    spreadsheet applications (Excel, Calc, Sheets) to treat it strictly as text.

    Legitimate numeric values (int, float, and strings representing pure negative/positive
    numbers like '-0.5' or '-12.50') are preserved without modification.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value
    if isinstance(value, bool):
        return value

    s = str(value)
    if not s:
        return ""

    stripped = s.lstrip()
    if not stripped:
        if any(c in s for c in ("\t", "\r", "\n")):
            return f"'{s}"
        return s

    first_char = stripped[0]
    has_leading_control = bool(s and s[0] in ("\t", "\r", "\n"))

    if first_char in ("=", "+", "@") or has_leading_control:
        return f"'{s}"

    if first_char == "-":
        if is_legitimate_number(stripped):
            return s
        return f"'{s}"

    return s


def sanitize_csv_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Applies CSV cell sanitization to all values in a record dictionary."""
    return {k: sanitize_csv_cell(v) for k, v in record.items()}


class EventLogger:
    """Thread-safe CSV event logger for surveillance violation incidents."""

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        csv_filename: str = "events.csv",
        storage_manager: Optional[Any] = None,
        encryption_enabled: Optional[bool] = None,
        chunk_size: int = 1,
    ):
        if log_dir is None:
            from config import LOG_DIR
            raw_dir = Path(LOG_DIR)
        else:
            raw_dir = Path(log_dir)

        # Validate directory and resolve target CSV path safely without link redirection
        self.log_dir = ensure_safe_directory(raw_dir)
        self.csv_path = resolve_safe_output_path(
            self.log_dir,
            csv_filename,
            allow_extensions={".csv"},
        )
        self.session_id = session_id or self._generate_session_id()
        self._lock = threading.Lock()

        # Determine encryption state
        if encryption_enabled is None:
            import config
            self.encryption_enabled = bool(getattr(config, "ENCRYPTION_ENABLED", False))
        else:
            self.encryption_enabled = bool(encryption_enabled)

        self.storage_manager = storage_manager
        self._encrypted_writer = None

        if self.encryption_enabled:
            if self.storage_manager is None:
                from encrypted_storage import get_encrypted_storage_manager
                self.storage_manager = get_encrypted_storage_manager()
            from encrypted_storage import EncryptedCSVWriter
            self._encrypted_writer = EncryptedCSVWriter(
                storage_manager=self.storage_manager,
                log_dir=self.log_dir,
                base_filename=self.csv_path.name,
                session_id=self.session_id,
                fieldnames=FIELDNAMES,
                chunk_size=chunk_size,
                evidence_type="EVENT_LOG",
            )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a compact unique session identifier."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        return f"sess_{ts}_{uid}"

    def log_event(
        self,
        violation_type: str,
        duration_sec: float,
        confidence: Optional[float] = None,
        screenshot_path: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Appends a violation incident to the CSV log.

        Args:
            violation_type: Identifier of the violation (e.g. 'multiple_faces', 'mobile_detected').
            duration_sec: Duration in seconds the condition persisted.
            confidence: Optional model detection confidence score.
            screenshot_path: Optional path to the evidence screenshot.
            timestamp: Optional ISO 8601 timestamp string (auto-generated if None).

        Returns:
            bool: True if write succeeded, False otherwise.
        """
        if not violation_type or not isinstance(violation_type, str):
            violation_type = "unknown_violation"

        # Format ISO 8601 timestamp with local timezone
        if timestamp is None:
            ts_str = datetime.now().astimezone().isoformat()
        else:
            ts_str = str(timestamp)

        # Format duration safely
        try:
            dur_val = f"{float(duration_sec):.2f}"
        except (ValueError, TypeError):
            dur_val = "0.00"

        # Format confidence safely (leave empty string if None or invalid)
        if confidence is not None:
            try:
                conf_val = f"{float(confidence):.4f}"
            except (ValueError, TypeError):
                conf_val = ""
        else:
            conf_val = ""

        # Format screenshot path safely
        shot_val = str(screenshot_path).replace("\\", "/") if screenshot_path else ""

        raw_record = {
            "timestamp": ts_str,
            "session_id": self.session_id,
            "violation_type": violation_type,
            "duration_sec": dur_val,
            "confidence": conf_val,
            "screenshot_path": shot_val,
        }

        # Apply strict CSV formula injection sanitization to all fields
        record = sanitize_csv_record(raw_record)

        if self.encryption_enabled and self._encrypted_writer is not None:
            with self._lock:
                try:
                    return self._encrypted_writer.write_record(record)
                except Exception as e:
                    print(f"Error: Failed to write encrypted event: {e}")
                    return False

        with self._lock:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

                with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(record)
                return True
            except (OSError, IOError, PermissionError) as e:
                # Log to stderr without crashing
                print(f"Error: Failed to write event to {self.csv_path}: {e}")
                return False

    def flush(self) -> None:
        """Flush any pending buffered writes."""
        if self.encryption_enabled and self._encrypted_writer is not None:
            with self._lock:
                self._encrypted_writer.flush()

    def close(self) -> None:
        """Close logger resources and finalize encrypted segments."""
        if self.encryption_enabled and self._encrypted_writer is not None:
            with self._lock:
                self._encrypted_writer.close()


_default_logger: Optional[EventLogger] = None


def get_event_logger(
    log_dir: Optional[Union[str, Path]] = None,
    session_id: Optional[str] = None,
    storage_manager: Optional[Any] = None,
    encryption_enabled: Optional[bool] = None,
    chunk_size: int = 1,
) -> EventLogger:
    """Returns or lazily creates the default shared EventLogger singleton."""
    global _default_logger
    if _default_logger is None or log_dir is not None:
        _default_logger = EventLogger(
            log_dir=log_dir,
            session_id=session_id,
            storage_manager=storage_manager,
            encryption_enabled=encryption_enabled,
            chunk_size=chunk_size,
        )
    return _default_logger
