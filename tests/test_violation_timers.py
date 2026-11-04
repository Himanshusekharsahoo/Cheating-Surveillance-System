import time
import pytest
from eye_movement import GAZE_CENTER, GAZE_LEFT, GAZE_NO_FACE, COMPLIANT_GAZE_STATES
from head_pose import HEAD_SCREEN, HEAD_LEFT, HEAD_NO_FACE, COMPLIANT_HEAD_STATES

pytestmark = pytest.mark.unit

def evaluate_gaze_violation(gaze_direction, start_time, current_time, threshold=3.0):
    """
    Simulation of the gaze violation timer logic from main.py.
    Returns: (new_start_time, alert_triggered)
    """
    if gaze_direction in (GAZE_LEFT, "Looking Right", "Looking Up", "Looking Down"):
        if start_time is None:
            return current_time, False
        elif (current_time - start_time) >= threshold:
            return None, True
        else:
            return start_time, False
    else:
        return None, False

def test_neutral_gaze_does_not_trigger_violation():
    """Verify that compliant center gaze never triggers a violation timer."""
    start_time = None
    # 10 continuous seconds of compliant center gaze
    for t in range(10):
        start_time, triggered = evaluate_gaze_violation(GAZE_CENTER, start_time, float(t))
        assert triggered is False
        assert start_time is None

def test_sustained_gaze_violation_triggers_after_threshold():
    """Verify that looking away triggers an alert only when sustained for >= 3 seconds."""
    start_time = None

    # Time = 0s: Look left
    start_time, triggered = evaluate_gaze_violation(GAZE_LEFT, start_time, 0.0)
    assert triggered is False
    assert start_time == 0.0

    # Time = 2.0s: Still looking left (< 3.0s)
    start_time, triggered = evaluate_gaze_violation(GAZE_LEFT, start_time, 2.0)
    assert triggered is False
    assert start_time == 0.0

    # Time = 3.0s: Reached 3.0s threshold -> should trigger alert and reset timer
    start_time, triggered = evaluate_gaze_violation(GAZE_LEFT, start_time, 3.0)
    assert triggered is True
    assert start_time is None

def test_intermittent_gaze_resets_timer():
    """Verify that returning to screen resets timer and avoids false alerts."""
    start_time = None

    # Time = 0s: Look left
    start_time, triggered = evaluate_gaze_violation(GAZE_LEFT, start_time, 0.0)
    assert start_time == 0.0

    # Time = 2.0s: Glance back to center screen
    start_time, triggered = evaluate_gaze_violation(GAZE_CENTER, start_time, 2.0)
    assert triggered is False
    assert start_time is None  # Timer was reset!

    # Time = 3.0s: Look left again -> starts new timer
    start_time, triggered = evaluate_gaze_violation(GAZE_LEFT, start_time, 3.0)
    assert triggered is False
    assert start_time == 3.0

def test_missing_face_not_treated_as_compliant():
    """Verify that 'No Face Detected' is not in compliant states and does not reset missing face timer."""
    assert GAZE_NO_FACE not in COMPLIANT_GAZE_STATES
    assert HEAD_NO_FACE not in COMPLIANT_HEAD_STATES

    # Missing face timer simulation
    face_missing_start = None
    threshold = 3.0

    def evaluate_face_absence(is_missing, start_time, current_time):
        if is_missing:
            if start_time is None:
                return current_time, False
            elif (current_time - start_time) >= threshold:
                return None, True
            else:
                return start_time, False
        else:
            return None, False

    # Face absent at t=0
    face_missing_start, triggered = evaluate_face_absence(True, face_missing_start, 0.0)
    assert triggered is False
    assert face_missing_start == 0.0

    # Face absent at t=3.1s -> trigger alert
    face_missing_start, triggered = evaluate_face_absence(True, face_missing_start, 3.1)
    assert triggered is True
    assert face_missing_start is None


def evaluate_multi_face_incident(is_multi_face, start_time, incident_logged, current_time, threshold=3.0):
    """
    Simulates the multiple-face violation incident tracking logic from main.py.
    Returns: (new_start_time, new_incident_logged, event_triggered)
    """
    if is_multi_face:
        if start_time is None:
            return current_time, False, False
        elif not incident_logged and (current_time - start_time >= threshold):
            return start_time, True, True
        else:
            return start_time, incident_logged, False
    else:
        return None, False, False


def test_multi_face_under_threshold_does_not_trigger():
    """Verify that multiple faces lasting < 3.0s do not trigger an event."""
    start_time, logged, triggered = evaluate_multi_face_incident(True, None, False, 0.0)
    assert triggered is False
    assert start_time == 0.0
    assert logged is False

    # 2.5s elapsed (< 3.0s)
    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 2.5)
    assert triggered is False
    assert start_time == 0.0
    assert logged is False


def test_multi_face_sustained_triggers_single_event_without_per_frame_spam():
    """Verify that multiple faces for >= 3.0s trigger exactly one event and do not spam on subsequent frames."""
    start_time, logged, triggered = evaluate_multi_face_incident(True, None, False, 0.0)
    # At 3.0s -> trigger!
    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 3.0)
    assert triggered is True
    assert logged is True

    # At 3.5s and 4.0s (condition still sustained) -> no duplicate trigger
    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 3.5)
    assert triggered is False
    assert logged is True

    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 4.0)
    assert triggered is False
    assert logged is True


def test_multi_face_clears_and_subsequent_incident_triggers_new_event():
    """Verify timer resets when condition clears, and a subsequent incident triggers a new event."""
    start_time, logged, triggered = evaluate_multi_face_incident(True, None, False, 0.0)
    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 3.0)
    assert triggered is True
    assert logged is True

    # Second face disappears at t=4.0s -> resets timer and logged flag
    start_time, logged, triggered = evaluate_multi_face_incident(False, start_time, logged, 4.0)
    assert triggered is False
    assert start_time is None
    assert logged is False

    # Second face reappears at t=5.0s (new separate incident begins)
    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 5.0)
    assert triggered is False
    assert start_time == 5.0
    assert logged is False

    # Reaches 3.0s threshold for second incident at t=8.0s -> triggers new event
    start_time, logged, triggered = evaluate_multi_face_incident(True, start_time, logged, 8.0)
    assert triggered is True
    assert logged is True


def test_screenshot_write_failure_handled_safely(tmp_path):
    """Verify that cv2.imwrite failure logs event with empty path and does not raise exception."""
    from unittest.mock import patch
    import numpy as np
    from main import record_violation_incident
    from event_logger import EventLogger

    logger = EventLogger(log_dir=tmp_path)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("cv2.imwrite", return_value=False):
        write_ok, path = record_violation_incident(
            violation_type="multiple_faces",
            prefix="multi_face",
            frame=dummy_frame,
            duration_sec=3.0,
            confidence=None,
            log_dir=str(tmp_path),
            logger=logger,
        )
        assert write_ok is False
        assert path == ""

    # Verify event was still logged in CSV with empty screenshot_path
    csv_file = tmp_path / "events.csv"
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "multiple_faces" in content
        assert "3.00" in content


def test_camera_cleanup_after_simulated_processing_exception(tmp_path):
    """Verify that camera handle is released in finally block when an exception occurs in frame loop."""
    from unittest.mock import MagicMock, patch
    from main import run_surveillance

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Simulate an unhandled exception on cap.read()
    mock_cap.read.side_effect = RuntimeError("Simulated camera frame read error")

    # Run in headless mode so it doesn't open GUI
    fake_pred = tmp_path / "fake.dat"
    fake_yolo = tmp_path / "fake.pt"
    fake_pred.write_bytes(b"dummy")
    fake_yolo.write_bytes(b"dummy")

    import hashlib
    import config
    config.register_test_model("fake.dat", hashlib.sha256(b"dummy").hexdigest(), 5, "dlib_shape_predictor")
    config.register_test_model("fake.pt", hashlib.sha256(b"dummy").hexdigest(), 5, "yolo")
    try:
        with pytest.raises(RuntimeError, match="Simulated camera frame read error"):
            with patch("config.SHAPE_PREDICTOR_PATH", fake_pred), \
                 patch("config.YOLO_MODEL_PATH", fake_yolo):
                run_surveillance(cap=mock_cap, display=False)
    finally:
        config.reset_model_registry()

    # Assert camera release was called in finally block
    assert mock_cap.release.called is True


def test_momentary_gaze_uncertainty_resets_deviation_timer():
    """Verify that a momentary uncertain frame (blink) resets gaze deviation timer, avoiding false alerts."""
    from eye_movement import GAZE_LEFT, GAZE_UNCERTAIN, GAZE_CENTER

    def evaluate_gaze_with_uncertainty(gaze_dir, dev_start, dev_logged, current_time, threshold=3.0):
        if gaze_dir in ("Looking Left", "Looking Right", "Looking Up", "Looking Down"):
            if dev_start is None:
                return current_time, False, False
            elif not dev_logged and (current_time - dev_start >= threshold):
                return dev_start, True, True
            else:
                return dev_start, dev_logged, False
        elif gaze_dir == GAZE_UNCERTAIN:
            # Uncertain frame resets deviation timer
            return None, False, False
        else:
            return None, False, False

    # Look left at t=0
    dev_start, logged, triggered = evaluate_gaze_with_uncertainty(GAZE_LEFT, None, False, 0.0)
    assert dev_start == 0.0
    assert triggered is False

    # Still looking left at t=2.0s
    dev_start, logged, triggered = evaluate_gaze_with_uncertainty(GAZE_LEFT, dev_start, logged, 2.0)
    assert dev_start == 0.0
    assert triggered is False

    # Blink at t=2.5s (GAZE_UNCERTAIN) -> resets deviation timer!
    dev_start, logged, triggered = evaluate_gaze_with_uncertainty(GAZE_UNCERTAIN, dev_start, logged, 2.5)
    assert dev_start is None
    assert triggered is False

    # Look left again at t=3.0s -> starts new timer from t=3.0
    dev_start, logged, triggered = evaluate_gaze_with_uncertainty(GAZE_LEFT, dev_start, logged, 3.0)
    assert dev_start == 3.0
    assert triggered is False


def test_sustained_uncertain_gaze_triggers_occluded_alert():
    """Verify that sustained eye occlusion (GAZE_UNCERTAIN >= 3.0s) triggers gaze_occluded alert."""
    from eye_movement import GAZE_UNCERTAIN, GAZE_CENTER

    def evaluate_occlusion(gaze_dir, occ_start, occ_logged, current_time, threshold=3.0):
        if gaze_dir == GAZE_UNCERTAIN:
            if occ_start is None:
                return current_time, False, False
            elif not occ_logged and (current_time - occ_start >= threshold):
                return occ_start, True, True
            else:
                return occ_start, occ_logged, False
        else:
            return None, False, False

    # Occluded at t=0
    occ_start, logged, triggered = evaluate_occlusion(GAZE_UNCERTAIN, None, False, 0.0)
    assert occ_start == 0.0
    assert triggered is False

    # Still occluded at t=2.5s (< 3.0s)
    occ_start, logged, triggered = evaluate_occlusion(GAZE_UNCERTAIN, occ_start, logged, 2.5)
    assert triggered is False

    # At t=3.0s -> trigger alert!
    occ_start, logged, triggered = evaluate_occlusion(GAZE_UNCERTAIN, occ_start, logged, 3.0)
    assert triggered is True
    assert logged is True

    # At t=3.5s -> no duplicate trigger
    occ_start, logged, triggered = evaluate_occlusion(GAZE_UNCERTAIN, occ_start, logged, 3.5)
    assert triggered is False

    # Eyes open at t=4.0s (GAZE_CENTER) -> resets occlusion timer
    occ_start, logged, triggered = evaluate_occlusion(GAZE_CENTER, occ_start, logged, 4.0)
    assert occ_start is None
    assert logged is False


def test_gaze_violation_micro_blink_tolerance():
    """Verify that brief micro-blinks (<= 3 frames) preserve sustained deviation timer, while prolonged resets."""
    from eye_movement import GAZE_LEFT, GAZE_UNCERTAIN, GAZE_CENTER

    def evaluate_gaze_with_blink_tolerance(
        gaze_dir, dev_start, dev_logged, uncert_frames, current_time, threshold=3.0, max_blink=3
    ):
        if gaze_dir in ("Looking Left", "Looking Right", "Looking Up", "Looking Down"):
            uncert_frames = 0
            if dev_start is None:
                return current_time, False, False, uncert_frames
            elif not dev_logged and (current_time - dev_start >= threshold):
                return dev_start, True, True, uncert_frames
            else:
                return dev_start, dev_logged, False, uncert_frames
        elif gaze_dir == GAZE_UNCERTAIN:
            uncert_frames += 1
            if uncert_frames > max_blink:
                return None, False, False, uncert_frames
            return dev_start, dev_logged, False, uncert_frames
        else:
            return None, False, False, 0

    # Look left at t=0
    dev_start, logged, trig, u_frames = evaluate_gaze_with_blink_tolerance(GAZE_LEFT, None, False, 0, 0.0)
    assert dev_start == 0.0
    assert trig is False

    # 1 frame micro-blink at t=1.0 (frame 1)
    dev_start, logged, trig, u_frames = evaluate_gaze_with_blink_tolerance(GAZE_UNCERTAIN, dev_start, logged, u_frames, 1.0)
    assert dev_start == 0.0  # Preserved across micro-blink!
    assert trig is False

    # Look left again at t=1.1
    dev_start, logged, trig, u_frames = evaluate_gaze_with_blink_tolerance(GAZE_LEFT, dev_start, logged, u_frames, 1.1)
    assert dev_start == 0.0
    assert u_frames == 0

    # Sustained past 3.0s (e.g. t=3.1) -> triggers alert!
    dev_start, logged, trig, u_frames = evaluate_gaze_with_blink_tolerance(GAZE_LEFT, dev_start, logged, u_frames, 3.1)
    assert trig is True

    # Prolonged uncertainty (4 consecutive frames) resets
    dev_start, logged, trig, u_frames = evaluate_gaze_with_blink_tolerance(GAZE_LEFT, None, False, 0, 5.0)
    for _ in range(4):
        dev_start, logged, trig, u_frames = evaluate_gaze_with_blink_tolerance(GAZE_UNCERTAIN, dev_start, logged, u_frames, 5.5)
    assert dev_start is None


def test_single_eye_failure_is_uncertain_but_not_sustained_occlusion():
    from eye_movement import calculate_gaze_direction, GAZE_UNCERTAIN
    from main import gaze_occlusion_is_confirmed

    state = calculate_gaze_direction(
        (10, 40), 100, 100, None, 100, 100,
        calibrated_gaze=((0.50, 0.40), (0.50, 0.40)),
    )
    assert state == GAZE_UNCERTAIN
    assert not gaze_occlusion_is_confirmed({"missing_left_pupil": False, "missing_right_pupil": True})
    assert gaze_occlusion_is_confirmed({"missing_left_pupil": True, "missing_right_pupil": True})
