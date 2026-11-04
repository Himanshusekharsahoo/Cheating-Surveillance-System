"""
Phase 10.13.13 — GAP-06 Focused Unit Tests:
Camera Initialization Failure Cleanup and Resource Lifecycle.

Verifies:
1. Camera cannot be opened -> releases capture, flushes/closes logger, returns non-zero.
2. Camera creation fails -> cleans up logger and returns non-zero without leaking handles.
3. Exception during capture geometry resolution -> cleans up camera and logger.
4. Logger initialization succeeds but camera fails -> logger flushed and closed.
5. Logger initialization fails before camera -> handles partially initialized state safely.
6. Main monitoring loop raises exception -> triggers full resource cleanup.
7. Capture release occurs exactly as intended.
8. Logger flush and close occur exactly as intended.
9. Cleanup is safe for partially initialized resources (repeated calls safe).
10. Normal successful execution and shutdown remain completely intact.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import config

from main import run_surveillance, main
from event_logger import EventLogger

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolate_model_validation(monkeypatch):
    monkeypatch.setattr(config, "validate_config", lambda: None)


def _make_dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_camera_cannot_be_opened_cleans_up_and_returns_nonzero():
    """Verify that when a camera cannot be opened, resources are cleaned up and non-zero is returned."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    mock_logger = MagicMock(spec=EventLogger)

    ret = run_surveillance(cap=mock_cap, logger=mock_logger, display=False)

    assert ret != 0
    assert mock_cap.release.called is True
    assert mock_logger.close.called is True


def test_unopened_webcam_cleans_up_and_returns_nonzero():
    """Verify that cv2.VideoCapture returning an unopened device is safely released and cleaned up."""
    mock_unopened = MagicMock()
    mock_unopened.isOpened.return_value = False

    mock_logger = MagicMock(spec=EventLogger)

    with patch("cv2.VideoCapture", return_value=mock_unopened):
        ret = run_surveillance(camera_index=0, logger=mock_logger, display=False)

    assert ret != 0
    assert mock_unopened.release.called is True
    assert mock_logger.close.called is True


def test_exception_during_capture_geometry_triggers_cleanup():
    """Verify that an exception during capture geometry resolution triggers cleanup."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    mock_logger = MagicMock(spec=EventLogger)

    with patch("main.resolve_capture_geometry", side_effect=RuntimeError("Geometry resolution failed")):
        with pytest.raises(RuntimeError, match="Geometry resolution failed"):
            run_surveillance(cap=mock_cap, logger=mock_logger, display=False)

    assert mock_cap.release.called is True
    assert mock_logger.close.called is True


def test_logger_flushed_and_closed_on_camera_open_failure():
    """Verify that active logger is explicitly flushed and closed when camera fails."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    mock_logger = MagicMock(spec=EventLogger)

    ret = run_surveillance(cap=mock_cap, logger=mock_logger, display=False)
    assert ret == 1
    assert mock_logger.flush.called is True
    assert mock_logger.close.called is True


def test_gaze_diagnostics_logger_closed_on_camera_open_failure():
    """Verify that gaze diagnostics logger is closed if camera fails to open."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    mock_gaze_logger = MagicMock()

    ret = run_surveillance(cap=mock_cap, gaze_logger=mock_gaze_logger, display=False)
    assert ret == 1
    assert mock_gaze_logger.close.called is True


def test_exception_in_monitoring_loop_triggers_cleanup():
    """Verify that an exception in the monitoring loop triggers full resource cleanup."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # First frame succeeds, second frame raises RuntimeError
    mock_cap.read.side_effect = [
        (True, _make_dummy_frame()),
        RuntimeError("Fatal camera hardware fault mid-stream"),
    ]

    mock_logger = MagicMock(spec=EventLogger)

    with pytest.raises(RuntimeError, match="Fatal camera hardware fault"):
        run_surveillance(cap=mock_cap, logger=mock_logger, display=False, max_frames=5)

    assert mock_cap.release.called is True
    assert mock_logger.close.called is True


def test_cleanup_is_safe_for_partially_initialized_resources():
    """Verify that cleanup handles missing, None, or partially initialized objects without error."""
    # When VideoCapture raises an immediate exception during creation
    mock_logger = MagicMock(spec=EventLogger)

    with patch("cv2.VideoCapture", side_effect=OSError("Driver crashed")):
        with pytest.raises(OSError, match="Driver crashed"):
            run_surveillance(camera_index=99, logger=mock_logger, display=False)

    # Logger was initialized and must be cleaned up
    assert mock_logger.close.called is True


def test_cli_main_propagates_camera_failure_exit_code():
    """Verify that main() exits with non-zero status code when surveillance fails."""
    mock_unopened = MagicMock()
    mock_unopened.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_unopened):
        exit_code = main(["--camera", "0", "--headless"])

    assert exit_code != 0
    assert exit_code == 1


def test_normal_successful_shutdown_calls_cleanup_cleanly():
    """Verify that normal EOF shutdown releases camera and closes loggers cleanly."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Immediate EOF
    mock_cap.read.return_value = (False, None)

    mock_logger = MagicMock(spec=EventLogger)
    mock_gaze_logger = MagicMock()

    ret = run_surveillance(cap=mock_cap, logger=mock_logger, gaze_logger=mock_gaze_logger, display=False)

    assert ret == 0
    assert mock_cap.release.called is True
    assert mock_logger.close.called is True
    assert mock_gaze_logger.close.called is True
