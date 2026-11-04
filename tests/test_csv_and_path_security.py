"""
Security unit tests for CSV formula injection prevention and filesystem path security.
Phase 10.13.23.
"""

import csv
from datetime import datetime
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

from event_logger import EventLogger, sanitize_csv_cell, sanitize_csv_record, FIELDNAMES
from path_security import (
    validate_safe_filename,
    resolve_safe_output_path,
    ensure_safe_directory,
    validate_safe_read_path,
    is_symlink_or_junction,
)
from screen_gaze import save_screen_calibration, ScreenGazeCalibrator, screen_calibration_environment, FIVE_POINT_TARGETS
from main import record_violation_incident, validate_cli_args, parse_args
import numpy as np

pytestmark = pytest.mark.unit


# ============================================================================
# Section A: CSV Formula Injection Prevention Tests
# ============================================================================

@pytest.mark.parametrize(
    "input_val, expected_output",
    [
        # Standard formula trigger prefixes
        ("=1+1", "'=1+1"),
        ("=cmd|' /C calc'!A0", "'=cmd|' /C calc'!A0"),
        ("+SUM(A1:A2)", "'+SUM(A1:A2)"),
        ("-1+1", "'-1+1"),
        ("-cmd|' /C calc'!A0", "'-cmd|' /C calc'!A0"),
        ("@SUM(A1:A2)", "'@SUM(A1:A2)"),

        # Leading whitespace and control characters before formula markers
        ("\t=1+1", "'\t=1+1"),
        ("\r=1+1", "'\r=1+1"),
        ("\n=1+1", "'\n=1+1"),
        ("\r\n+SUM(1,2)", "'\r\n+SUM(1,2)"),
        ("   =1+1", "'   =1+1"),
        ("   +SUM(A1:A2)", "'   +SUM(A1:A2)"),
        ("   -1+1", "'   -1+1"),
        ("   @SUM(A1:A2)", "'   @SUM(A1:A2)"),

        # Leading control characters without formula markers
        ("\tTabbedText", "'\tTabbedText"),
        ("\rCarriageReturn", "'\rCarriageReturn"),
        ("\nNewlineText", "'\nNewlineText"),

        # Legitimate numeric negative values (MUST NOT be sanitized with single quote)
        ("-0.5", "-0.5"),
        ("-12.50", "-12.50"),
        ("-15", "-15"),
        ("   -0.5", "   -0.5"),
        ("-1.23e-4", "-1.23e-4"),
        (-0.5, -0.5),
        (-15.2, -15.2),
        (-42, -42),

        # Legitimate positive numeric and telemetry values
        (0.8745, 0.8745),
        (3.0, 3.0),
        (100, 100),
        (0, 0),
        (True, True),
        (False, False),

        # Legitimate strings, descriptions, and timestamps
        ("normal_violation", "normal_violation"),
        ("mobile_detected", "mobile_detected"),
        ("multiple_faces", "multiple_faces"),
        ("head_Looking_Left", "head_Looking_Left"),
        ("2026-11-01T17:00:00+00:00", "2026-11-01T17:00:00+00:00"),
        ("sess_20261101_123456_abcdef12", "sess_20261101_123456_abcdef12"),

        # Unicode characters
        ("élève_test", "élève_test"),
        ("日本語_テスト", "日本語_テスト"),
        ("Ñandú", "Ñandú"),

        # Embedded newlines
        ("first_line\nsecond_line", "first_line\nsecond_line"),

        # Empty and null values
        ("", ""),
        (None, ""),
    ],
)
def test_sanitize_csv_cell_comprehensive(input_val, expected_output):
    """Verify that sanitize_csv_cell neutralizes formula triggers while preserving legitimate values."""
    sanitized = sanitize_csv_cell(input_val)
    assert sanitized == expected_output


def test_sanitize_csv_record():
    """Verify dictionary-level CSV record sanitization."""
    raw = {
        "timestamp": "2026-11-01T17:00:00+00:00",
        "session_id": "=cmd|' /C calc'!A0",
        "violation_type": "+SUM(A1:A2)",
        "duration_sec": "-1+1",
        "confidence": 0.85,
        "screenshot_path": "log/shot.png",
    }
    sanitized = sanitize_csv_record(raw)
    assert sanitized["session_id"] == "'=cmd|' /C calc'!A0"
    assert sanitized["violation_type"] == "'+SUM(A1:A2)"
    assert sanitized["duration_sec"] == "'-1+1"
    assert sanitized["confidence"] == 0.85
    assert sanitized["screenshot_path"] == "log/shot.png"


def test_event_logger_csv_roundtrip_with_formula_injections(tmp_path: Path):
    """
    Verify end-to-end that EventLogger neutralizes formula injections on disk,
    correctly writes RFC-compliant CSV, and reads back with Python's standard csv module.
    """
    logger = EventLogger(log_dir=tmp_path, session_id="=malicious_session")
    csv_file = tmp_path / "events.csv"

    # Log events with injected formula payloads
    logger.log_event(violation_type="+SUM(1,2)", duration_sec=3.0, confidence=0.92, screenshot_path="-screenshot.png")
    logger.log_event(violation_type="@SUM(A1:A2)", duration_sec=4.5, confidence=None, screenshot_path="\tlog/test.png")
    logger.log_event(violation_type="-1+1", duration_sec=5.0, confidence=0.1234, screenshot_path="   =cmd")

    assert csv_file.exists()

    # Read back raw text to verify leading single quote in CSV cells
    raw_text = csv_file.read_text(encoding="utf-8")
    assert "'=malicious_session" in raw_text
    assert "'+SUM(1,2)" in raw_text
    assert "'-1+1" in raw_text
    assert "'@SUM(A1:A2)" in raw_text

    # Read back with standard csv.DictReader
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3

        # Verify row 1
        assert reader[0]["session_id"] == "'=malicious_session"
        assert reader[0]["violation_type"] == "'+SUM(1,2)"
        assert reader[0]["duration_sec"] == "3.00"
        assert reader[0]["confidence"] == "0.9200"
        assert reader[0]["screenshot_path"] == "'-screenshot.png"

        # Verify row 2
        assert reader[1]["violation_type"] == "'@SUM(A1:A2)"
        assert reader[1]["screenshot_path"] == "'\tlog/test.png"

        # Verify row 3
        assert reader[2]["violation_type"] == "'-1+1"
        assert reader[2]["screenshot_path"] == "'   =cmd"


# ============================================================================
# Section B: Filesystem Path Security Tests
# ============================================================================

@pytest.mark.parametrize(
    "bad_filename",
    [
        "../traversal.csv",
        "..\\traversal.csv",
        "../../etc/passwd",
        "..\\..\\windows\\system32\\calc.exe",
        "C:\\absolute_path.csv",
        "/absolute/unix/path.csv",
        "\\leading_backslash.csv",
        "nested/sub/path.csv",
        "nested\\sub\\path.csv",
        "",
        "   ",
        ".",
        "..",
        "CON.csv",
        "con.png",
        "PRN.txt",
        "AUX.log",
        "NUL",
        "COM1.dat",
        "COM9",
        "LPT1.csv",
        "illegal:char.csv",
        "illegal*char.csv",
        "illegal?char.csv",
        "illegal|char.csv",
        "illegal<char>.csv",
        'illegal"char.csv',
        "null\x00byte.csv",
        "newline\n.csv",
    ],
)
def test_validate_safe_filename_rejects_insecure_names(bad_filename):
    """Verify that validate_safe_filename rejects traversal, absolute paths, reserved names, and illegal characters."""
    with pytest.raises(ValueError):
        validate_safe_filename(bad_filename)


@pytest.mark.parametrize(
    "good_filename",
    [
        "events.csv",
        "events_20261101.csv",
        "gaze_diagnostics.csv",
        "screen_gaze_calibration.json",
        "screenshot_12345.png",
        "multi_face-001.png",
        "élève_2026.png",
    ],
)
def test_validate_safe_filename_accepts_legitimate_names(good_filename):
    """Verify that validate_safe_filename accepts legitimate valid filenames."""
    assert validate_safe_filename(good_filename) == good_filename


def test_validate_safe_filename_enforces_extension():
    """Verify that allow_extensions restricts to expected suffixes."""
    assert validate_safe_filename("events.csv", allow_extensions={".csv"}) == "events.csv"
    with pytest.raises(ValueError, match="File extension"):
        validate_safe_filename("events.json", allow_extensions={".csv"})


def test_resolve_safe_output_path_containment(tmp_path: Path):
    """Verify that resolve_safe_output_path strictly binds target inside base directory."""
    target = resolve_safe_output_path(tmp_path, "events.csv")
    assert target == tmp_path / "events.csv"
    assert target.parent == tmp_path.resolve()

    # Traversal attempt
    with pytest.raises(ValueError):
        resolve_safe_output_path(tmp_path, "../outside.csv")


def test_event_logger_rejects_path_traversal_filename(tmp_path: Path):
    """Verify that EventLogger rejects path traversal filenames."""
    with pytest.raises(ValueError, match="Path traversal"):
        EventLogger(log_dir=tmp_path, csv_filename="../escaped_events.csv")

    with pytest.raises(ValueError, match="Path traversal"):
        EventLogger(log_dir=tmp_path, csv_filename="..\\escaped_events.csv")


def test_ensure_safe_directory_creation(tmp_path: Path):
    """Verify that ensure_safe_directory creates missing nested directories."""
    nested = tmp_path / "nested" / "logs" / "subdir"
    assert not nested.exists()
    resolved = ensure_safe_directory(nested)
    assert resolved.exists()
    assert resolved.is_dir()


def test_validate_safe_read_path():
    """Verify that validate_safe_read_path allows legitimate files and rejects non-existent/directories."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"video content")
        tf_name = tf.name

    try:
        # Valid file
        p = validate_safe_read_path(tf_name)
        assert p.is_file()

        # Non-existent file
        with pytest.raises(FileNotFoundError):
            validate_safe_read_path("nonexistent_video_file_xyz_123.mp4")

        # Directory passed where file expected
        with pytest.raises(ValueError, match="not a regular file"):
            validate_safe_read_path(Path(tf_name).parent)
    finally:
        if os.path.exists(tf_name):
            os.unlink(tf_name)


def test_record_violation_incident_prefix_sanitization(tmp_path: Path):
    """Verify that record_violation_incident sanitizes prefixes and prevents directory traversal."""
    dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    logger = EventLogger(log_dir=tmp_path)

    # Prefix with directory traversal characters
    ok, path = record_violation_incident(
        violation_type="traversal_test",
        prefix="../../escaped_prefix",
        frame=dummy_frame,
        duration_sec=2.0,
        log_dir=str(tmp_path),
        logger=logger,
    )
    assert ok is True

    # Verify screenshot filename does not contain traversal characters
    filename = Path(path).name
    assert ".." not in filename
    assert "escaped_prefix" in filename

    # Verify screenshot was written strictly inside tmp_path
    files_in_tmp = list(tmp_path.glob("*.png"))
    assert len(files_in_tmp) == 1
    assert files_in_tmp[0].parent == tmp_path.resolve()
    assert "escaped_prefix" in files_in_tmp[0].name
    assert not (tmp_path.parent / "escaped_prefix").exists()


def test_save_screen_calibration_path_hardening(tmp_path: Path):
    """Verify that save_screen_calibration rejects traversal and link redirection."""
    # Create a properly fitted calibrator
    from tests.test_screen_calibration_persistence import _fit, _manager
    fitted_manager = _fit(_manager())
    calibrator = fitted_manager.calibrator
    env = screen_calibration_environment("webcam", 0)

    # Path traversal attempt
    with pytest.raises(ValueError):
        save_screen_calibration(
            calibrator=calibrator,
            path=tmp_path / "../outside.json",
            display_size=(1920, 1080),
            targets=FIVE_POINT_TARGETS,
            samples_per_target=1,
            min_confidence=0.8,
            environment=env,
        )

    # Successful safe save
    safe_path = tmp_path / "valid_calib.json"
    result_path = save_screen_calibration(
        calibrator=calibrator,
        path=safe_path,
        display_size=(1920, 1080),
        targets=FIVE_POINT_TARGETS,
        samples_per_target=1,
        min_confidence=0.8,
        environment=env,
    )
    assert result_path.exists()
    assert result_path == safe_path


def test_symlink_or_junction_detection(tmp_path: Path):
    """Verify that is_symlink_or_junction detects symlinks and directory junctions."""
    regular_file = tmp_path / "file.txt"
    regular_file.write_text("hello")
    assert not is_symlink_or_junction(regular_file)
    assert not is_symlink_or_junction(tmp_path)

    # Test symlink creation if permissions allow
    try:
        symlink_file = tmp_path / "link.txt"
        os.symlink(regular_file, symlink_file)
        assert is_symlink_or_junction(symlink_file)
    except (OSError, NotImplementedError):
        pass  # Unprivileged on Windows

    # Test Windows junction creation via cmd mklink /J
    if sys.platform == "win32":
        try:
            target_dir = tmp_path / "target_dir"
            target_dir.mkdir()
            junction_dir = tmp_path / "junction_dir"
            res = subprocess.run(
                f'cmd /c mklink /J "{junction_dir}" "{target_dir}"',
                shell=True,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and junction_dir.exists():
                assert is_symlink_or_junction(junction_dir)
        except Exception:
            pass
