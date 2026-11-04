"""
Tests for Phase 10.13.12 GAP-01: Screen Calibration Fixation Dwell & Stability.

Verifies:
1. Three adjacent valid frames cannot immediately complete a target.
2. Insufficient dwell does not advance calibration.
3. Stable observations over the required duration can complete a target.
4. Unstable observations reset the stability window.
5. Invalid observations are rejected.
6. Non-finite observations are rejected.
7. Insufficient-confidence observations are rejected.
8. Face loss during dwell resets collection according to documented policy.
9. Cancellation preserves the previous valid calibration.
10. Failed recalibration preserves the previous valid calibration.
11. Successful calibration still passes existing fitting and quality checks.
"""

from types import SimpleNamespace
import numpy as np
import pytest

import config
from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult
from screen_gaze import (
    ScreenCalibrationManager,
    ScreenGazeCalibrator,
    FIVE_POINT_TARGETS,
    CALIB_STATE_NOT_CALIBRATED,
    CALIB_STATE_CALIBRATING,
    CALIB_STATE_VALID,
    CALIB_STATE_FAILED,
    CALIB_STATE_CANCELLED,
    screen_calibration_environment,
)
from time_source import MockTimeSource

pytestmark = pytest.mark.unit


def _make_valid_sample(yaw=0.0, pitch=0.0, confidence=0.90):
    return CombinedGazeResult(
        yaw=float(yaw),
        pitch=float(pitch),
        confidence=float(confidence),
        status="VALID",
    )


def test_three_adjacent_valid_frames_cannot_immediately_complete_target():
    """
    Test 1: Three consecutive adjacent valid frames (e.g. 33ms cadence) cannot
    advance past the first target when min_dwell_sec is enforced.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(
        min_dwell_sec=1.0,
        stability_window_sec=0.4,
        samples_per_target=3,
        time_source=time_src,
    )
    mgr.start()
    assert mgr.state == CALIB_STATE_CALIBRATING
    assert mgr.current_target_idx == 0

    # Deliver 3 frames across 66ms (standard 30 FPS video cadence)
    for frame_idx in range(3):
        gaze = _make_valid_sample(yaw=0.0, pitch=0.0)
        accepted = mgr.process_sample(gaze, timestamp=time_src.now())
        assert accepted is True
        time_src.advance(0.033)

    # Dwell elapsed is 0.066s < 1.0s: target MUST NOT advance
    assert mgr.current_target_idx == 0
    assert mgr.state == CALIB_STATE_CALIBRATING
    assert mgr.target_dwell_progress < 0.20


def test_insufficient_dwell_does_not_advance_calibration():
    """
    Test 2: Observations with insufficient dwell duration (e.g. 0.8s < 1.0s) do not
    advance calibration even if measurements are stable.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(
        min_dwell_sec=1.0,
        stability_window_sec=0.4,
        samples_per_target=3,
        time_source=time_src,
    )
    mgr.start()

    # Feed stable samples every 100ms up to 0.8s (dwell elapsed 0.8s < 1.0s)
    for _ in range(8):
        gaze = _make_valid_sample(yaw=0.0, pitch=0.0)
        mgr.process_sample(gaze, timestamp=time_src.now())
        time_src.advance(0.10)

    # 8 samples collected, stable for 0.8s, but total dwell < 1.0s
    assert mgr.current_target_idx == 0
    assert mgr.state == CALIB_STATE_CALIBRATING
    assert mgr.target_dwell_progress == pytest.approx(0.70, abs=0.05)


def test_stable_observations_over_required_duration_completes_target():
    """
    Test 3: Stable observations over the required dwell and stability window
    satisfy target acceptance criteria and advance to the next target.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(
        min_dwell_sec=1.0,
        stability_window_sec=0.4,
        samples_per_target=3,
        time_source=time_src,
    )
    mgr.start()

    # Feed stable samples every 100ms past 1.0s dwell
    for _ in range(12):  # 1.1s total elapsed
        gaze = _make_valid_sample(yaw=0.0, pitch=0.0)
        mgr.process_sample(gaze, timestamp=time_src.now())
        time_src.advance(0.10)

    # Now target 0 ("Center") must have completed and advanced to target 1 ("Top-Left")
    assert mgr.current_target_idx == 1
    assert mgr.current_target[2] == "Top-Left"


def test_unstable_observations_reset_stability_window():
    """
    Test 4: Transient gaze movement exceeding stability threshold resets the
    stability window and prevents target advancement even if total dwell exceeds threshold.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(
        min_dwell_sec=1.0,
        stability_window_sec=0.4,
        stability_threshold_deg=2.0,
        samples_per_target=3,
        time_source=time_src,
    )
    mgr.start()

    # Dwell for 0.8s stably
    for _ in range(8):
        gaze = _make_valid_sample(yaw=0.0, pitch=0.0)
        mgr.process_sample(gaze, timestamp=time_src.now())
        time_src.advance(0.10)

    assert mgr.stability_status == "STABLE"

    # Sudden saccade / instability at 0.8s: yaw jumps to 15 degrees (> 2.0 deg threshold)
    unstable_gaze = _make_valid_sample(yaw=15.0, pitch=0.0)
    mgr.process_sample(unstable_gaze, timestamp=time_src.now())
    time_src.advance(0.10)

    assert mgr.stability_status == "UNSTABLE"

    # Advance time to 1.2s total dwell with only 0.2s of new stability (< 0.4s window)
    for _ in range(2):
        gaze = _make_valid_sample(yaw=15.0, pitch=0.0)
        mgr.process_sample(gaze, timestamp=time_src.now())
        time_src.advance(0.10)

    # Even though total dwell is 1.1s (> 1.0s), stability window was reset and is only 0.2s (< 0.4s)
    assert mgr.current_target_idx == 0


def test_invalid_observations_rejected():
    """
    Test 5: Observations with non-VALID status or None are rejected and reset stability.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(min_dwell_sec=1.0, time_source=time_src)
    mgr.start()

    assert mgr.process_sample(None, timestamp=time_src.now()) is False
    assert mgr.stability_status == "INVALID_GAZE"

    uncert = CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="UNCERTAIN")
    assert mgr.process_sample(uncert, timestamp=time_src.now()) is False
    assert mgr.stability_status == "INVALID_GAZE"


def test_non_finite_observations_rejected():
    """
    Test 6: Observations with NaN or infinite angles/confidence are rejected.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(min_dwell_sec=1.0, time_source=time_src)
    mgr.start()

    for val in (float("nan"), float("inf"), float("-inf")):
        nan_yaw = CombinedGazeResult(yaw=val, pitch=0.0, confidence=0.9, status="VALID")
        assert mgr.process_sample(nan_yaw, timestamp=time_src.now()) is False

        nan_conf = CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=val, status="VALID")
        assert mgr.process_sample(nan_conf, timestamp=time_src.now()) is False


def test_insufficient_confidence_observations_rejected():
    """
    Test 7: Observations with confidence below the calibrated floor (0.20) are rejected.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(min_dwell_sec=1.0, min_confidence=0.20, time_source=time_src)
    mgr.start()

    low_conf = CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.19, status="VALID")
    assert mgr.process_sample(low_conf, timestamp=time_src.now()) is False
    assert mgr.stability_status == "LOW_CONFIDENCE"


def test_face_loss_during_dwell_resets_collection_policy():
    """
    Test 8: Documented face loss policy: loss of facial detection resets the
    stability window and in-progress dwell state.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(
        min_dwell_sec=1.0,
        stability_window_sec=0.4,
        samples_per_target=3,
        time_source=time_src,
    )
    mgr.start()

    # Fixate stably for 0.7s
    for _ in range(7):
        gaze = _make_valid_sample(yaw=0.0, pitch=0.0)
        mgr.process_sample(gaze, timestamp=time_src.now())
        time_src.advance(0.10)

    # Face is lost (face_count = 0)
    missing_face = SimpleNamespace(face_count=0, gaze_direction="No Face Detected", head_direction="No Face Detected")
    assert mgr.process_sample(None, face_result=missing_face, timestamp=time_src.now()) is False
    assert mgr.stability_status == "FACE_LOST"

    # Gaze reappears: stability window starts afresh
    time_src.advance(0.10)
    gaze = _make_valid_sample(yaw=0.0, pitch=0.0)
    mgr.process_sample(gaze, timestamp=time_src.now())

    # After 1 more sample, total stable time is only 0.0s since restart (< 0.4s)
    assert mgr.current_target_idx == 0


def test_cancellation_preserves_previous_valid_calibration():
    """
    Test 9: Cancelling an interactive calibration restores the previous valid model.
    """
    calibrator = ScreenGazeCalibrator()
    calibrator._is_calibrated = True
    calibrator._calibration_matrix = np.eye(3, 2, dtype=np.float64)
    calibrator._quality_metrics = {"success": True, "quality_score": 0.95}

    mgr = ScreenCalibrationManager(calibrator=calibrator)
    assert mgr.is_calibrated is True

    mgr.start()
    assert mgr.state == CALIB_STATE_CALIBRATING
    assert mgr.is_calibrated is False

    mgr.cancel()
    assert mgr.state == CALIB_STATE_CANCELLED
    assert mgr.is_calibrated is True
    assert mgr.calibrator._calibration_matrix is not None


def test_failed_recalibration_preserves_previous_valid_calibration():
    """
    Test 10: Calibration failure restores the previous valid model.
    """
    calibrator = ScreenGazeCalibrator()
    calibrator._is_calibrated = True
    calibrator._calibration_matrix = np.eye(3, 2, dtype=np.float64)
    calibrator._quality_metrics = {"success": True, "quality_score": 0.95}

    mgr = ScreenCalibrationManager(calibrator=calibrator)
    mgr.start()
    mgr.fail("simulated_error")

    assert mgr.state == CALIB_STATE_FAILED
    assert mgr.is_calibrated is True
    assert mgr.calibrator._calibration_matrix is not None


def test_successful_calibration_passes_existing_fitting_and_quality_checks():
    """
    Test 11: Deliberate dwell-based calibration completes all 5 targets and passes
    all existing fitting and quality checks.
    """
    time_src = MockTimeSource(initial_time=0.0)
    mgr = ScreenCalibrationManager(
        min_dwell_sec=0.5,
        stability_window_sec=0.2,
        samples_per_target=3,
        min_sample_interval_sec=0.05,
        time_source=time_src,
    )
    mgr.start()

    target_angles = {
        "Center": (0.0, 0.0),
        "Top-Left": (-18.0, 18.0),
        "Top-Right": (18.0, 18.0),
        "Bottom-Left": (-18.0, -18.0),
        "Bottom-Right": (18.0, -18.0),
    }

    for idx, (tx, ty, name) in enumerate(FIVE_POINT_TARGETS):
        assert mgr.current_target_idx == idx
        base_yaw, base_pitch = target_angles[name]

        # Feed 10 stable frames at 0.06s intervals (0.6s total dwell > 0.5s)
        for s in range(10):
            # Slight physiological micro-tremor well within 2.0 deg stability threshold
            yaw = base_yaw + (0.05 * (s % 3))
            pitch = base_pitch + (0.05 * (s % 3))
            gaze = _make_valid_sample(yaw=yaw, pitch=pitch, confidence=0.90)
            mgr.process_sample(gaze, timestamp=time_src.now())
            time_src.advance(0.06)

    assert mgr.state == CALIB_STATE_VALID
    assert mgr.is_calibrated is True
    assert mgr.calibrator.quality_metrics["success"] is True
    assert mgr.calibrator.quality_metrics["rmse"] <= config.SCREEN_CALIBRATION_MAX_RMSE
