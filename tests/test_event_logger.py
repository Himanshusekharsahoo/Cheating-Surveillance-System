import csv
from datetime import datetime
from pathlib import Path
import threading
from unittest.mock import patch
import pytest

from event_logger import EventLogger, FIELDNAMES

pytestmark = pytest.mark.unit


def test_csv_creation_and_header(tmp_path: Path):
    """Verify that EventLogger creates events.csv with the exact required header."""
    logger = EventLogger(log_dir=tmp_path)
    csv_file = tmp_path / "events.csv"
    assert not csv_file.exists()

    success = logger.log_event("test_violation", duration_sec=3.0)
    assert success is True
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == FIELDNAMES
        row = next(reader)
        assert row[2] == "test_violation"
        assert row[3] == "3.00"


def test_append_multiple_events_without_overwriting(tmp_path: Path):
    """Verify that multiple logged events append sequentially without overwriting."""
    logger = EventLogger(log_dir=tmp_path)
    csv_file = tmp_path / "events.csv"

    logger.log_event("violation_1", duration_sec=3.1)
    logger.log_event("violation_2", duration_sec=4.2)
    logger.log_event("violation_3", duration_sec=5.3)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        assert reader[0]["violation_type"] == "violation_1"
        assert reader[0]["duration_sec"] == "3.10"
        assert reader[1]["violation_type"] == "violation_2"
        assert reader[1]["duration_sec"] == "4.20"
        assert reader[2]["violation_type"] == "violation_3"
        assert reader[2]["duration_sec"] == "5.30"


def test_timestamp_and_session_id_formatting(tmp_path: Path):
    """Verify ISO 8601 timezone-aware timestamps and consistent session ID reuse."""
    session_id = "test_custom_session_123"
    logger = EventLogger(log_dir=tmp_path, session_id=session_id)
    csv_file = tmp_path / "events.csv"

    logger.log_event("head_Looking_Left", duration_sec=3.0)
    logger.log_event("mobile_detected", duration_sec=3.5)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            assert row["session_id"] == session_id
            # Verify timestamp parses as valid ISO 8601 with tzinfo
            dt = datetime.fromisoformat(row["timestamp"])
            assert dt.tzinfo is not None


def test_empty_confidence_when_unavailable(tmp_path: Path):
    """Verify that confidence is formatted as float when given and empty string when None."""
    logger = EventLogger(log_dir=tmp_path)
    csv_file = tmp_path / "events.csv"

    # Event without confidence
    logger.log_event("missing_face", duration_sec=3.0, confidence=None)
    # Event with confidence
    logger.log_event("mobile_detected", duration_sec=3.0, confidence=0.87452)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert reader[0]["confidence"] == ""
        assert reader[1]["confidence"] == "0.8745"


def test_screenshot_path_serialization_and_normalization(tmp_path: Path):
    """Verify screenshot paths normalize backslashes and accept empty values."""
    logger = EventLogger(log_dir=tmp_path)
    csv_file = tmp_path / "events.csv"

    logger.log_event("head_Looking_Right", duration_sec=3.0, screenshot_path=r"log\head_right_123.png")
    logger.log_event("eye_Looking_Left", duration_sec=3.0, screenshot_path="")

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert reader[0]["screenshot_path"] == "log/head_right_123.png"
        assert reader[1]["screenshot_path"] == ""


def test_filesystem_write_failure_handling(tmp_path: Path):
    """Verify that write/permission errors return False and do not crash the caller."""
    logger = EventLogger(log_dir=tmp_path)

    with patch("builtins.open", side_effect=PermissionError("Mock Permission Denied")):
        result = logger.log_event("violation_err", duration_sec=3.0)
        assert result is False


def test_handling_existing_csv_file(tmp_path: Path):
    """Verify that an existing CSV file is not wiped and header is not duplicated."""
    csv_file = tmp_path / "events.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "timestamp": "2026-10-19T10:00:00+00:00",
            "session_id": "initial_session",
            "violation_type": "initial_violation",
            "duration_sec": "3.00",
            "confidence": "",
            "screenshot_path": "log/init.png"
        })

    logger = EventLogger(log_dir=tmp_path, session_id="new_session")
    logger.log_event("second_violation", duration_sec=4.0)

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Should have 1 header line + 2 data rows = 3 lines
        assert len(lines) == 3
        assert lines[0].strip() == ",".join(FIELDNAMES)
        assert "initial_violation" in lines[1]
        assert "second_violation" in lines[2]


def test_thread_safe_concurrent_logging(tmp_path: Path):
    """Verify thread safety when multiple worker threads log events simultaneously."""
    logger = EventLogger(log_dir=tmp_path)
    csv_file = tmp_path / "events.csv"

    def worker(idx):
        logger.log_event(f"worker_violation_{idx}", duration_sec=float(idx))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 20
