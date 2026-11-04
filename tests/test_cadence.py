from unittest.mock import MagicMock
import numpy as np
import pytest

from mobile_detection import (
    PhoneDetectionTracker,
    DETECTION_STATE_FRESH_POSITIVE,
    DETECTION_STATE_FRESH_NEGATIVE,
    DETECTION_STATE_CACHED_POSITIVE,
    DETECTION_STATE_CACHED_NEGATIVE,
    DETECTION_STATE_EXPIRED,
    DETECTION_STATE_INFERENCE_ERROR,
)

pytestmark = pytest.mark.unit


def create_mock_yolo_result(detected=True, confidence=0.85):
    """Helper to generate a mock YOLO result object."""
    mock_res = MagicMock()
    if detected:
        mock_box = MagicMock()
        mock_box.conf = [MagicMock(item=lambda: confidence)]
        mock_box.cls = [MagicMock(item=lambda: 0)]  # class 0 = mobile
        mock_box.xyxy = [np.array([50, 50, 200, 300])]
        mock_res.boxes = [mock_box]
    else:
        mock_res.boxes = []
    return [mock_res]


def test_yolo_cadence_inference_interval():
    """Verify that YOLO inference executes only at the configured interval."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=False)

    tracker = PhoneDetectionTracker(interval_frames=3, cache_expiry_frames=6, model_instance=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Inference frame
    _, detected, res = tracker.process_frame(frame.copy())
    assert mock_model.call_count == 1
    assert res.state == DETECTION_STATE_FRESH_NEGATIVE

    # Frame 2: Skipped (cached)
    _, detected, res = tracker.process_frame(frame.copy())
    assert mock_model.call_count == 1  # Not called!
    assert res.state == DETECTION_STATE_CACHED_NEGATIVE

    # Frame 3: Skipped (cached)
    _, detected, res = tracker.process_frame(frame.copy())
    assert mock_model.call_count == 1  # Not called!
    assert res.state == DETECTION_STATE_CACHED_NEGATIVE

    # Frame 4: Inference frame (1 + 3)
    _, detected, res = tracker.process_frame(frame.copy())
    assert mock_model.call_count == 2
    assert res.state == DETECTION_STATE_FRESH_NEGATIVE


def test_cached_detection_reuse_and_confidence():
    """Verify positive phone detection is cached and reused with matching confidence."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.92)

    tracker = PhoneDetectionTracker(interval_frames=3, cache_expiry_frames=6, model_instance=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Fresh positive
    _, detected, res1 = tracker.process_frame(frame.copy())
    assert detected is True
    assert res1.state == DETECTION_STATE_FRESH_POSITIVE
    assert res1.confidence == 0.92
    assert len(res1.boxes) == 1

    # Frame 2: Cached positive
    _, detected, res2 = tracker.process_frame(frame.copy())
    assert detected is True
    assert res2.state == DETECTION_STATE_CACHED_POSITIVE
    assert res2.confidence == 0.92
    assert len(res2.boxes) == 1
    assert res2.age_frames == 1

    # Frame 3: Cached positive
    _, detected, res3 = tracker.process_frame(frame.copy())
    assert detected is True
    assert res3.state == DETECTION_STATE_CACHED_POSITIVE
    assert res3.confidence == 0.92
    assert res3.age_frames == 2


def test_cache_expiry_removes_stale_detection():
    """Verify that cached detections expire after cache_expiry_frames and clear stale boxes."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.88)

    # Interval = 10, Expiry = 4
    tracker = PhoneDetectionTracker(interval_frames=10, cache_expiry_frames=4, model_instance=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Fresh positive (age = 0)
    _, det, res = tracker.process_frame(frame.copy())
    assert det is True
    assert res.state == DETECTION_STATE_FRESH_POSITIVE

    # Frames 2, 3, 4, 5: Age reaches 1, 2, 3, 4 (within expiry)
    for expected_age in range(1, 5):
        _, det, res = tracker.process_frame(frame.copy())
        assert det is True
        assert res.state == DETECTION_STATE_CACHED_POSITIVE
        assert res.age_frames == expected_age
        assert len(res.boxes) == 1

    # Frame 6: Age becomes 5 > expiry (4) -> EXPIRED
    _, det, res = tracker.process_frame(frame.copy())
    assert det is False
    assert res.state == DETECTION_STATE_EXPIRED
    assert res.confidence is None
    assert len(res.boxes) == 0  # Stale boxes removed!


def test_model_inference_failure_handled_gracefully():
    """Verify that YOLO model inference exception does not crash the loop and transitions gracefully."""
    mock_model = MagicMock()
    # Frame 1 succeeds
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.85)

    tracker = PhoneDetectionTracker(interval_frames=2, cache_expiry_frames=3, model_instance=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Fresh positive
    _, det, res = tracker.process_frame(frame.copy())
    assert det is True
    assert res.state == DETECTION_STATE_FRESH_POSITIVE

    # Frame 2: Skipped (age 1)
    _, det, res = tracker.process_frame(frame.copy())
    assert res.state == DETECTION_STATE_CACHED_POSITIVE

    # Frame 3: Model fails on inference frame
    mock_model.side_effect = RuntimeError("CUDA out of memory simulation")
    _, det, res = tracker.process_frame(frame.copy())
    # Should catch gracefully, not crash
    assert res.state == DETECTION_STATE_INFERENCE_ERROR
    assert len(res.boxes) == 0  # Do not draw stale boxes on error

    # Subsequent skipped frame
    _, det, res = tracker.process_frame(frame.copy())
    # Age increases to 3, which is <= 3
    # Next frame age becomes 4 > 3 -> EXPIRED
    _, det, res = tracker.process_frame(frame.copy())
    assert res.state == DETECTION_STATE_EXPIRED


def test_violation_timer_cadence_behavior():
    """
    Simulation of violation timer evaluation with fresh, cached, expired, and error states.
    Verifies that:
    1. Fresh and cached positive sustain the timer.
    2. Fresh negative resets the timer.
    3. Expired resets the timer (stale presence not sustained).
    4. Inference error does not wipe timer immediately, but cannot trigger violation.
    """
    def evaluate_phone_timer(state, start_time, current_time, threshold=3.0):
        if state in (DETECTION_STATE_FRESH_POSITIVE, DETECTION_STATE_CACHED_POSITIVE):
            if start_time is None:
                return current_time, False
            elif (current_time - start_time) >= threshold:
                return start_time, True
            else:
                return start_time, False
        elif state in (DETECTION_STATE_FRESH_NEGATIVE, DETECTION_STATE_CACHED_NEGATIVE):
            return None, False
        elif state == DETECTION_STATE_EXPIRED:
            return None, False
        elif state == DETECTION_STATE_INFERENCE_ERROR:
            # Retains start_time without triggering
            return start_time, False
        return None, False

    # 1. Fresh positive at t=0
    timer, triggered = evaluate_phone_timer(DETECTION_STATE_FRESH_POSITIVE, None, 0.0)
    assert timer == 0.0
    assert triggered is False

    # 2. Cached positive at t=1.0 and t=2.0
    timer, triggered = evaluate_phone_timer(DETECTION_STATE_CACHED_POSITIVE, timer, 1.0)
    assert timer == 0.0
    assert triggered is False

    timer, triggered = evaluate_phone_timer(DETECTION_STATE_CACHED_POSITIVE, timer, 2.0)
    assert timer == 0.0
    assert triggered is False

    # 3. Inference error at t=2.5: does not wipe timer immediately!
    timer, triggered = evaluate_phone_timer(DETECTION_STATE_INFERENCE_ERROR, timer, 2.5)
    assert timer == 0.0  # Still 0.0, not None
    assert triggered is False

    # 4. Fresh positive at t=3.0 -> triggers violation!
    timer, triggered = evaluate_phone_timer(DETECTION_STATE_FRESH_POSITIVE, timer, 3.0)
    assert triggered is True

    # 5. Clean negative resets timer
    timer, triggered = evaluate_phone_timer(DETECTION_STATE_FRESH_NEGATIVE, timer, 3.5)
    assert timer is None
    assert triggered is False

    # 6. Expired state resets timer (cannot count stale detections as presence)
    timer, _ = evaluate_phone_timer(DETECTION_STATE_FRESH_POSITIVE, None, 10.0)
    assert timer == 10.0
    timer, triggered = evaluate_phone_timer(DETECTION_STATE_EXPIRED, timer, 12.0)
    assert timer is None  # Reset!
    assert triggered is False
