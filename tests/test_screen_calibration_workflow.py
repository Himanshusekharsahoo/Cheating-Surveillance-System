"""
Tests for Phase 10.13.9: Interactive Screen Calibration Workflow & Boundary Tracking.

Verifies:
- 5-point calibration target sequencing, ordering, and sample validation.
- ScreenCalibrationManager lifecycle (start, sample collection, advance, cancel, fit, fail).
- Runtime integration into main.py and VisionPipeline.
- HUD rendering for valid, invalid, and uncalibrated states.
- Boundary classification (ON_SCREEN, NEAR_BOUNDARY, OUTSIDE_SCREEN).
- Dedicated sustained outside-screen violation (gaze_outside_screen) with timing and false-positive protections.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest

import config
import main
from eye_movement import GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UNCERTAIN
from head_pose import HEAD_SCREEN
from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult
from screen_gaze import (
    ScreenGazeResult,
    ScreenGazeCalibrator,
    ScreenCalibrationSample,
    ScreenCalibrationManager,
    FIVE_POINT_TARGETS,
    BOUNDARY_ON_SCREEN,
    BOUNDARY_NEAR_BOUNDARY,
    BOUNDARY_OUTSIDE_SCREEN,
    BOUNDARY_UNCERTAIN,
    BOUNDARY_CALIBRATION_REQUIRED,
    CALIB_STATE_NOT_CALIBRATED,
    CALIB_STATE_CALIBRATING,
    CALIB_STATE_VALID,
    CALIB_STATE_FAILED,
    CALIB_STATE_CANCELLED,
)
from mobile_detection import PhoneDetectionResult
from time_source import MockTimeSource

pytestmark = pytest.mark.unit


# =========================================================================
# Part 1: ScreenCalibrationManager Unit Tests
# =========================================================================

def test_five_point_target_coordinates_and_ordering():
    """Verify 5-point targets are in normalized coordinates in the specified order."""
    assert len(FIVE_POINT_TARGETS) == 5
    expected_order = ["Center", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    actual_order = [t[2] for t in FIVE_POINT_TARGETS]
    assert actual_order == expected_order

    # Verify coordinates are in valid normalized range [0.0, 1.0]
    for tx, ty, name in FIVE_POINT_TARGETS:
        assert 0.0 <= tx <= 1.0
        assert 0.0 <= ty <= 1.0

    # Specifically Center is (0.5, 0.5)
    assert FIVE_POINT_TARGETS[0] == (0.5, 0.5, "Center")


def test_manager_lifecycle_start_and_cancel():
    """Verify start resets calibrator and cancel halts process."""
    mgr = ScreenCalibrationManager(samples_per_target=3)
    assert mgr.state == CALIB_STATE_NOT_CALIBRATED
    assert not mgr.is_calibrated

    mgr.start()
    assert mgr.state == CALIB_STATE_CALIBRATING
    assert mgr.current_target_idx == 0
    assert mgr.current_target == (0.5, 0.5, "Center")

    mgr.cancel()
    assert mgr.state == CALIB_STATE_CANCELLED
    assert not mgr.is_calibrated
    assert mgr.current_target is None


def test_manager_rejects_invalid_missing_and_low_confidence_samples():
    """Verify process_sample rejects invalid, low-confidence, or non-finite observations."""
    mgr = ScreenCalibrationManager(samples_per_target=3, min_confidence=0.20)
    mgr.start()

    # 1. None observation
    assert mgr.process_sample(None) is False
    assert mgr.current_target_samples == 0

    # 2. Uncertain status
    uncert = CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.8, status="UNCERTAIN")
    assert mgr.process_sample(uncert) is False

    # 3. Low confidence below threshold
    low_conf = CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.15, status="VALID")
    assert mgr.process_sample(low_conf) is False

    # 4. Non-finite values
    nan_gaze = CombinedGazeResult(yaw=float("nan"), pitch=0.0, confidence=0.8, status="VALID")
    assert mgr.process_sample(nan_gaze) is False

    # 5. Missing yaw or pitch
    none_yaw = CombinedGazeResult(yaw=None, pitch=0.0, confidence=0.8, status="VALID")
    assert mgr.process_sample(none_yaw) is False

    assert mgr.current_target_samples == 0


def test_manager_advances_targets_and_fits_successfully():
    """Verify manager collects required samples per target and completes valid calibration."""
    # Ideal mapping: yaw [-20, 20] -> x [0.1, 0.9], pitch [-20, 20] -> y [0.9, 0.1]
    # For each target, provide realistic gaze angles
    target_angles = {
        "Center": (0.0, 0.0),
        "Top-Left": (-18.0, 18.0),
        "Top-Right": (18.0, 18.0),
        "Bottom-Left": (-18.0, -18.0),
        "Bottom-Right": (18.0, -18.0),
    }

    mgr = ScreenCalibrationManager(samples_per_target=3)
    mgr.start()

    for idx, (tx, ty, name) in enumerate(FIVE_POINT_TARGETS):
        assert mgr.current_target_idx == idx
        assert mgr.current_target[2] == name
        yaw, pitch = target_angles[name]

        for s in range(3):
            # Slightly vary angles for realistic variance
            sample_yaw = yaw + (s * 0.2)
            sample_pitch = pitch + (s * 0.2)
            gaze = CombinedGazeResult(
                yaw=sample_yaw, pitch=sample_pitch, confidence=0.90, status="VALID"
            )
            accepted = mgr.process_sample(gaze)
            assert accepted is True

    # After 5 targets * 3 samples = 15 samples, manager should have fitted
    assert mgr.state == CALIB_STATE_VALID
    assert mgr.is_calibrated
    assert mgr.calibrator.is_calibrated

    # Predict test point: neutral should map close to center (0.5, 0.5)
    pred = mgr.calibrator.estimate_screen_gaze(gaze_yaw=0.0, gaze_pitch=0.0, gaze_confidence=0.9)
    assert pred.status == "VALID"
    assert pred.screen_x == pytest.approx(0.5, abs=0.10)
    assert pred.screen_y == pytest.approx(0.5, abs=0.10)
    assert pred.boundary_state == BOUNDARY_ON_SCREEN


def test_manager_fitting_failure_on_zero_variance():
    """Verify manager transitions to CALIB_STATE_FAILED if gaze features have zero variance."""
    mgr = ScreenCalibrationManager(samples_per_target=3)
    mgr.start()

    # Pass identical (0.0, 0.0) gaze angles for all targets
    for _ in FIVE_POINT_TARGETS:
        for _ in range(3):
            gaze = CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.90, status="VALID")
            mgr.process_sample(gaze)

    assert mgr.state == CALIB_STATE_FAILED
    assert not mgr.is_calibrated
    assert "variance" in mgr.failure_reason.lower()


def test_manager_draw_overlay_renders_safely():
    """Verify draw_overlay draws calibration target without crashing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mgr = ScreenCalibrationManager(samples_per_target=3)
    mgr.start()

    out = mgr.draw_overlay(frame.copy())
    assert out.shape == frame.shape
    # Ensure drawing modified pixels (not all black anymore)
    assert np.any(out > 0)


# =========================================================================
# Part 2: HUD Formatting Tests
# =========================================================================

def test_screen_gaze_hud_uncalibrated_and_missing():
    """Verify uncalibrated and None results display NOT_CALIBRATED and no false zeroes."""
    line1, line2 = main.screen_gaze_hud(None)
    assert "NOT_CALIBRATED" in line1
    assert "(—, —)" in line2
    assert "0.0" not in line2

    uncalib = ScreenGazeResult(is_calibrated=False, status="NOT_CALIBRATED")
    line1, line2 = main.screen_gaze_hud(uncalib)
    assert "NOT_CALIBRATED" in line1
    assert "(—, —)" in line2


def test_screen_gaze_hud_valid_states():
    """Verify valid calibrated screen gaze displays normalized coordinates and confidence."""
    valid_res = ScreenGazeResult(
        screen_x=0.5234,
        screen_y=0.4812,
        boundary_state=BOUNDARY_ON_SCREEN,
        confidence=0.88,
        status="VALID",
        is_calibrated=True,
    )
    line1, line2 = main.screen_gaze_hud(valid_res)
    assert "ON_SCREEN" in line1
    assert "(0.52, 0.48)" in line1
    assert "88%" in line1
    assert "VALID" in line2

    outside_res = ScreenGazeResult(
        screen_x=1.25,
        screen_y=-0.10,
        boundary_state=BOUNDARY_OUTSIDE_SCREEN,
        confidence=0.65,
        status="VALID",
        is_calibrated=True,
    )
    line1, line2 = main.screen_gaze_hud(outside_res)
    assert "OUTSIDE_SCREEN" in line1
    assert "(1.25, -0.10)" in line1


# =========================================================================
# Part 3: Runtime Integration & Outside-Screen Violation Tests
# =========================================================================

def _run_screen_surveillance(monkeypatch, results, frame_count=5, step_sec=1.0, screen_calibrator=None, calibrate_screen=False, display=False):
    """Helper to run surveillance loop with mocked screen gaze results."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.side_effect = [(True, frame.copy()) for _ in range(frame_count)] + [(False, None)]
    cap.isOpened.return_value = True

    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(main, "get_primary_display_resolution", lambda strict=False: (1920, 1080))
    monkeypatch.setattr(main.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *args: 0)
    monkeypatch.setattr(main.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(main.cv2, "setWindowProperty", lambda *args: None)
    monkeypatch.setattr(main.cv2, "destroyWindow", lambda *args: None)
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(main, "calibrate_gaze_baseline", lambda *a, **k: (((0.5, 0.5), (0.5, 0.5)), {}))

    res_queue = list(results)
    def mock_process_frame_face(display_frame, calibrated_angles=None, calibrated_gaze=None, **kwargs):
        if calibrated_angles is None:
            # Baseline calibration frame
            calib_res = SimpleNamespace(
                face_count=1, is_multi_face=False, gaze_direction=GAZE_CENTER,
                head_direction="Looking at Screen", head_pose_sample=(0.0, 0.0, 0.0),
                gaze_sample=((0.5, 0.5), (0.5, 0.5)),
                gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
                combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
                head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
                eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
                screen_gaze=ScreenGazeResult(is_calibrated=False, status="NOT_CALIBRATED"),
            )
            return calib_res, display_frame
        item = res_queue.pop(0) if res_queue else results[-1]
        return item, display_frame

    monkeypatch.setattr(main, "process_frame_face", mock_process_frame_face)
    monkeypatch.setattr(main, "PhoneDetectionTracker", lambda **k: SimpleNamespace(
        process_frame=lambda clean, annotate, display_frame: (
            display_frame, False, PhoneDetectionResult()
        )
    ))

    recorded_incidents = []
    def mock_record_violation(**kwargs):
        recorded_incidents.append(kwargs)
        return True, "test_shot.png"

    monkeypatch.setattr(main, "record_violation_incident", mock_record_violation)

    time_source = MockTimeSource()
    time_source.update = lambda **kwargs: time_source.advance(step_sec)

    main.run_surveillance(
        cap=cap,
        logger=MagicMock(session_id="test_sess"),
        gaze_logger=False,
        display=display,
        max_frames=frame_count,
        time_source=time_source,
        screen_calibrator=screen_calibrator,
        calibrate_screen=calibrate_screen,
    )
    return recorded_incidents


def test_explicit_recalibration_starts_even_when_existing_profile_is_valid(monkeypatch):
    calib = ScreenGazeCalibrator()
    calib._is_calibrated = True
    calib._calibration_matrix = np.zeros((3, 2), dtype=np.float64)
    calib._quality_metrics = {"success": True, "quality_score": 1.0}
    calib._calibrated_bounding_box = (0.1, 0.1, 0.9, 0.9)
    started = []
    original_start = ScreenCalibrationManager.start

    def track_start(manager):
        started.append(manager)
        return original_start(manager)

    monkeypatch.setattr(ScreenCalibrationManager, "start", track_start)
    _run_screen_surveillance(
        monkeypatch,
        [SimpleNamespace(
            face_count=1, is_multi_face=False, gaze_direction=GAZE_CENTER,
            head_direction=HEAD_SCREEN,
            gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
            combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
            head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
            eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
            screen_gaze=ScreenGazeResult(is_calibrated=True, status="VALID"),
        )] * 3,
        frame_count=3,
        screen_calibrator=calib,
        calibrate_screen=True,
        display=True,
    )
    assert len(started) == 1
    assert started[0].state == CALIB_STATE_CANCELLED


@pytest.mark.parametrize(
    "result",
    [
        None,
        ScreenGazeResult(boundary_state=BOUNDARY_OUTSIDE_SCREEN, status="UNCERTAIN", is_calibrated=True),
        ScreenGazeResult(boundary_state=BOUNDARY_OUTSIDE_SCREEN, status="VALID", confidence=0.19, is_calibrated=True),
        ScreenGazeResult(boundary_state=BOUNDARY_OUTSIDE_SCREEN, status="VALID", confidence=float("nan"), is_calibrated=True),
        ScreenGazeResult(boundary_state=BOUNDARY_OUTSIDE_SCREEN, status="VALID", confidence=0.9, is_calibrated=False),
    ],
)
def test_uncertain_or_low_confidence_screen_gaze_is_not_confirmed(result):
    assert not main.is_confirmed_outside_screen(result)


def test_sustained_outside_screen_gaze_triggers_violation(monkeypatch):
    """Sustained OUTSIDE_SCREEN gaze (>= 3.0s) triggers gaze_outside_screen event."""
    calib = ScreenGazeCalibrator()
    calib._is_calibrated = True

    outside_res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_RIGHT,
        head_direction=HEAD_SCREEN,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
        combined_gaze=CombinedGazeResult(yaw=25.0, pitch=0.0, confidence=0.85, status="VALID"),
        screen_gaze=ScreenGazeResult(
            screen_x=1.35,
            screen_y=0.50,
            boundary_state=BOUNDARY_OUTSIDE_SCREEN,
            confidence=0.75,
            status="VALID",
            is_calibrated=True,
        ),
    )

    # 5 frames with 1s step -> 0s, 1s, 2s, 3s, 4s (sustained 4s >= 3s)
    incidents = _run_screen_surveillance(
        monkeypatch, [outside_res] * 5, frame_count=5, step_sec=1.0, screen_calibrator=calib
    )

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "gaze_outside_screen" in violation_types
    # Verify confidence was logged
    outside_inc = next(inc for inc in incidents if inc["violation_type"] == "gaze_outside_screen")
    assert outside_inc["confidence"] == 0.75


def test_near_boundary_gaze_does_not_trigger_outside_violation(monkeypatch):
    """NEAR_BOUNDARY gaze does NOT trigger gaze_outside_screen violation."""
    calib = ScreenGazeCalibrator()
    calib._is_calibrated = True

    near_res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_RIGHT,
        head_direction=HEAD_SCREEN,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
        combined_gaze=CombinedGazeResult(yaw=15.0, pitch=0.0, confidence=0.85, status="VALID"),
        screen_gaze=ScreenGazeResult(
            screen_x=0.98,
            screen_y=0.50,
            boundary_state=BOUNDARY_NEAR_BOUNDARY,
            confidence=0.80,
            status="VALID",
            is_calibrated=True,
        ),
    )

    incidents = _run_screen_surveillance(
        monkeypatch, [near_res] * 5, frame_count=5, step_sec=1.0, screen_calibrator=calib
    )

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "gaze_outside_screen" not in violation_types


def test_uncalibrated_pipeline_never_triggers_outside_screen_violation(monkeypatch):
    """When screen is NOT calibrated, no outside-screen violation can ever trigger."""
    uncalib_res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_RIGHT,
        head_direction=HEAD_SCREEN,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
        combined_gaze=CombinedGazeResult(yaw=25.0, pitch=0.0, confidence=0.85, status="VALID"),
        screen_gaze=ScreenGazeResult(
            screen_x=None,
            screen_y=None,
            boundary_state=BOUNDARY_CALIBRATION_REQUIRED,
            confidence=0.0,
            status="NOT_CALIBRATED",
            is_calibrated=False,
        ),
    )

    # screen_calibrator=None
    incidents = _run_screen_surveillance(
        monkeypatch, [uncalib_res] * 5, frame_count=5, step_sec=1.0, screen_calibrator=None
    )

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "gaze_outside_screen" not in violation_types


def test_brief_outside_jitter_resets_violation_timer(monkeypatch):
    """Brief outside gaze (< 3.0s) returning to ON_SCREEN resets timer without incident."""
    calib = ScreenGazeCalibrator()
    calib._is_calibrated = True

    def make_res(boundary, sx=0.5):
        return SimpleNamespace(
            face_count=1,
            is_multi_face=False,
            gaze_direction=GAZE_CENTER,
            head_direction=HEAD_SCREEN,
            gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
            head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
            eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
            combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
            screen_gaze=ScreenGazeResult(
                screen_x=sx,
                screen_y=0.50,
                boundary_state=boundary,
                confidence=0.80,
                status="VALID",
                is_calibrated=True,
            ),
        )

    # Sequence: 2s outside, 1s on-screen (resets), 2s outside (2s < 3s threshold)
    seq = [
        make_res(BOUNDARY_OUTSIDE_SCREEN, 1.2),
        make_res(BOUNDARY_OUTSIDE_SCREEN, 1.2),
        make_res(BOUNDARY_ON_SCREEN, 0.5),
        make_res(BOUNDARY_OUTSIDE_SCREEN, 1.2),
        make_res(BOUNDARY_OUTSIDE_SCREEN, 1.2),
    ]

    incidents = _run_screen_surveillance(
        monkeypatch, seq, frame_count=5, step_sec=1.0, screen_calibrator=calib
    )

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "gaze_outside_screen" not in violation_types
