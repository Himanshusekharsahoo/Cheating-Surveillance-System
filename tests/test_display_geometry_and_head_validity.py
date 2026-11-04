"""
Tests for Phase 10.13.9.2: Display Geometry Correction & Head-Pose Validity Hardening.

Verifies:
1. Head-Pose Validity Hardening:
   - Missing head pose (head_pose is None) strictly rejected (no violations).
   - head_pose.status == "INVALID" strictly rejected (no violations).
   - head_pose.status == "UNCERTAIN" strictly rejected (no violations).
   - Valid head pose with HEAD_LEFT, HEAD_RIGHT, HEAD_UP, HEAD_DOWN, HEAD_TILTED
     generates independent head violations after sustained duration.
   - Fused overall-gaze monitoring operates independently of head-pose validity.

2. Display Geometry & Physical Calibration Canvas:
   - Target normalized positions on physical display.
   - Exact coordinate transformation from normalized canvas to display pixels.
   - Aspect-ratio handling (standard, widescreen, ultrawide).
   - Letterboxing and pillarboxing coordinate transformations.
   - Unsupported display geometry rejection.
   - Fullscreen window lifecycle, transition, and cleanup mocks.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call
import numpy as np
import pytest

import config
import main
from eye_movement import GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UNCERTAIN, GAZE_NO_FACE
from head_pose import (
    HEAD_SCREEN,
    HEAD_LEFT,
    HEAD_RIGHT,
    HEAD_UP,
    HEAD_DOWN,
    HEAD_TILTED,
    HEAD_NO_FACE,
)
from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult
from mobile_detection import PhoneDetectionResult
from time_source import MockTimeSource
from screen_gaze import (
    DisplayGeometry,
    FIVE_POINT_TARGETS,
    WINDOW_CALIBRATION_NAME,
    CALIB_STATE_NOT_CALIBRATED,
    CALIB_STATE_CALIBRATING,
    CALIB_STATE_VALID,
    CALIB_STATE_FAILED,
    CALIB_STATE_CANCELLED,
    ScreenCalibrationManager,
    get_primary_display_resolution,
    validate_display_geometry,
    map_normalized_to_display_pixel,
)

pytestmark = pytest.mark.unit


# =========================================================================
# Part 1: Head-Pose Validity Hardening Tests
# =========================================================================

def _run_mock_surveillance_with_head_pose(
    monkeypatch,
    face_results,
    frame_count=5,
    step_sec=1.0,
):
    """Executes run_surveillance with deterministic time steps and collects violation incidents."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.side_effect = [(True, frame.copy()) for _ in range(frame_count)] + [(False, None)]
    cap.isOpened.return_value = True

    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(main, "calibrate_gaze_baseline", lambda *a, **k: (((0.5, 0.5), (0.5, 0.5)), {}))

    res_queue = list(face_results)

    def mock_process_frame_face(display_frame, calibrated_angles=None, calibrated_gaze=None, **kwargs):
        if calibrated_angles is None:
            # Baseline calibration frame
            calib_res = SimpleNamespace(
                face_count=1,
                is_multi_face=False,
                gaze_direction=GAZE_CENTER,
                head_direction=HEAD_SCREEN,
                head_pose_sample=(0.0, 0.0, 0.0),
                gaze_sample=((0.5, 0.5), (0.5, 0.5)),
                gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
                combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
                head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
                eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
                screen_gaze=None,
            )
            return calib_res, display_frame
        item = res_queue.pop(0) if res_queue else face_results[-1]
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
        display=False,
        max_frames=frame_count,
        time_source=time_source,
        calibrate_screen=False,
    )

    return recorded_incidents


def test_head_pose_none_rejects_head_violation(monkeypatch):
    """
    CRITICAL REGRESSION TEST:
    When face_result.head_pose is None, head violations must NOT be triggered,
    even if head_direction is set to HEAD_LEFT.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_LEFT,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=None,  # Missing head pose estimation
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    # Sustained for 5 seconds: must NOT trigger any head violation
    head_incidents = [inc for inc in incidents if inc.get("violation_type", "").startswith("head_")]
    assert len(head_incidents) == 0, f"Unexpected head violation triggered when head_pose is None: {head_incidents}"


def test_head_pose_status_invalid_rejects_head_violation(monkeypatch):
    """
    When head_pose.status == "INVALID", head violations must NOT be triggered.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_RIGHT,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=25.0, pitch=0.0, roll=0.0, confidence=0.0, status="INVALID"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type", "").startswith("head_")]
    assert len(head_incidents) == 0


def test_head_pose_status_uncertain_rejects_head_violation(monkeypatch):
    """
    When head_pose.status == "UNCERTAIN", head violations must NOT be triggered.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_UP,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=0.0, pitch=25.0, roll=0.0, confidence=0.15, status="UNCERTAIN"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type", "").startswith("head_")]
    assert len(head_incidents) == 0


def test_valid_head_pose_triggers_head_left(monkeypatch):
    """
    Valid head pose with HEAD_LEFT sustained >= 3.0s triggers head_Looking_Left.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_LEFT,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=-25.0, pitch=0.0, roll=0.0, confidence=0.92, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type") == "head_Looking_Left"]
    assert len(head_incidents) == 1
    assert head_incidents[0]["confidence"] == pytest.approx(0.92, abs=1e-4)


def test_valid_head_pose_triggers_head_right(monkeypatch):
    """
    Valid head pose with HEAD_RIGHT sustained >= 3.0s triggers head_Looking_Right.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_RIGHT,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=25.0, pitch=0.0, roll=0.0, confidence=0.88, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type") == "head_Looking_Right"]
    assert len(head_incidents) == 1
    assert head_incidents[0]["confidence"] == pytest.approx(0.88, abs=1e-4)


def test_valid_head_pose_triggers_head_up_and_down(monkeypatch):
    """
    Valid head pose with HEAD_UP and HEAD_DOWN sustained triggers appropriate violations.
    """
    res_up = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_UP,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=0.0, pitch=25.0, roll=0.0, confidence=0.90, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res_up] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type") == "head_Looking_Up"]
    assert len(head_incidents) == 1

    res_down = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_DOWN,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=0.0, pitch=-25.0, roll=0.0, confidence=0.87, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents_down = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res_down] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_down_incidents = [inc for inc in incidents_down if inc.get("violation_type") == "head_Looking_Down"]
    assert len(head_down_incidents) == 1


def test_valid_head_pose_triggers_head_tilted(monkeypatch):
    """
    Valid head pose with HEAD_TILTED sustained >= 3.0s triggers head_Tilted.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_TILTED,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.85, status="VALID"),
        head_pose=HeadPoseResult(yaw=0.0, pitch=0.0, roll=25.0, confidence=0.91, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.85, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type") == "head_Tilted"]
    assert len(head_incidents) == 1


def test_fused_gaze_events_remain_independent_when_head_pose_invalid(monkeypatch):
    """
    When head_pose is INVALID (so head violation is rejected), fused visual attention
    violation (eye_Looking_Left) still triggers independently if fused gaze is sustained left.
    """
    res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_LEFT,
        head_direction=HEAD_LEFT,
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(yaw=-22.0, pitch=0.0, confidence=0.88, status="VALID"),
        head_pose=HeadPoseResult(yaw=-25.0, pitch=0.0, roll=0.0, confidence=0.0, status="INVALID"),
        eye_gaze=EyeGazeResult(confidence=0.88, status="VALID"),
        screen_gaze=None,
    )

    incidents = _run_mock_surveillance_with_head_pose(
        monkeypatch,
        face_results=[res] * 5,
        frame_count=5,
        step_sec=1.0,
    )

    head_incidents = [inc for inc in incidents if inc.get("violation_type", "").startswith("head_")]
    eye_incidents = [inc for inc in incidents if inc.get("violation_type") == "eye_Looking_Left"]

    # Head violation must be rejected due to invalid head pose
    assert len(head_incidents) == 0
    # Fused eye/visual attention violation must trigger independently
    assert len(eye_incidents) == 1


# =========================================================================
# Part 2: Display Geometry & Physical Calibration Canvas Tests
# =========================================================================

def test_five_point_target_normalized_coordinates():
    """Verify standard 5-point normalized target coordinates and bounds."""
    expected_targets = [
        (0.50, 0.50, "Center"),
        (0.10, 0.10, "Top-Left"),
        (0.90, 0.10, "Top-Right"),
        (0.10, 0.90, "Bottom-Left"),
        (0.90, 0.90, "Bottom-Right"),
    ]
    assert len(FIVE_POINT_TARGETS) == 5
    for (exp_x, exp_y, exp_name), (act_x, act_y, act_name) in zip(expected_targets, FIVE_POINT_TARGETS):
        assert exp_name == act_name
        assert act_x == pytest.approx(exp_x, abs=1e-5)
        assert act_y == pytest.approx(exp_y, abs=1e-5)
        assert 0.0 <= act_x <= 1.0
        assert 0.0 <= act_y <= 1.0


def test_coordinate_transformation_direct_display():
    """
    Verify exact mathematical transformation from normalized coordinates to display canvas pixels.
    For a 1920x1080 monitor:
      X_px = clip(round(u * (W - 1)), 0, W - 1)
      Y_px = clip(round(v * (H - 1)), 0, H - 1)
    """
    disp = (1920, 1080)

    # 1. Center (0.50, 0.50)
    cx, cy = map_normalized_to_display_pixel(0.50, 0.50, disp)
    assert cx == 960
    assert cy == 540

    # 2. Top-Left (0.10, 0.10)
    tlx, tly = map_normalized_to_display_pixel(0.10, 0.10, disp)
    assert tlx == 192
    assert tly == 108

    # 3. Top-Right (0.90, 0.10)
    trx, try_ = map_normalized_to_display_pixel(0.90, 0.10, disp)
    assert trx == 1727
    assert try_ == 108

    # 4. Bottom-Left (0.10, 0.90)
    blx, bly = map_normalized_to_display_pixel(0.10, 0.90, disp)
    assert blx == 192
    assert bly == 971

    # 5. Bottom-Right (0.90, 0.90)
    brx, bry = map_normalized_to_display_pixel(0.90, 0.90, disp)
    assert brx == 1727
    assert bry == 971

    # Bounds extremes
    min_x, min_y = map_normalized_to_display_pixel(0.0, 0.0, disp)
    assert min_x == 0 and min_y == 0

    max_x, max_y = map_normalized_to_display_pixel(1.0, 1.0, disp)
    assert max_x == 1919 and max_y == 1079


def test_aspect_ratio_and_letterboxing_coordinate_mapping():
    """
    Verify letterboxing and pillarboxing coordinate transformations
    when target aspect ratio differs from display aspect ratio.
    """
    # Pillarboxing: Mapping 4:3 target aspect (1.3333) into 16:9 display (1920x1080, aspect 1.7778)
    disp = (1920, 1080)
    target_aspect = 4.0 / 3.0

    # Effective width = 1080 * (4/3) = 1440. Offset X = (1920 - 1440) / 2 = 240.
    cx, cy = map_normalized_to_display_pixel(0.50, 0.50, disp, letterbox=True, source_aspect=target_aspect)
    assert cx == 960  # 240 + round(0.50 * 1439) = 240 + 720 = 960
    assert cy == 540

    tlx, tly = map_normalized_to_display_pixel(0.10, 0.10, disp, letterbox=True, source_aspect=target_aspect)
    # X = 240 + round(0.10 * 1439) = 240 + 144 = 384
    assert tlx == 384
    assert tly == 108

    # Letterboxing: Mapping 16:9 target aspect (1.7778) into 4:3 display (1024x768, aspect 1.3333)
    disp_43 = (1024, 768)
    source_aspect_169 = 16.0 / 9.0

    # Effective height = 1024 / (16/9) = 576. Offset Y = (768 - 576) / 2 = 96.
    cx43, cy43 = map_normalized_to_display_pixel(0.50, 0.50, disp_43, letterbox=True, source_aspect=source_aspect_169)
    assert cx43 == 512
    assert cy43 == 384  # 96 + round(0.50 * 575) = 96 + 288 = 384

    tlx43, tly43 = map_normalized_to_display_pixel(0.10, 0.10, disp_43, letterbox=True, source_aspect=source_aspect_169)
    assert tlx43 == 102
    assert tly43 == 154  # 96 + round(0.10 * 575) = 96 + 58 = 154


def test_validate_display_geometry_supported_and_unsupported():
    """Verify validation of standard, ultrawide, and unsupported display configurations."""
    # Standard 16:9
    geo_1080p = validate_display_geometry(1920, 1080)
    assert geo_1080p.is_supported is True
    assert geo_1080p.aspect_ratio == pytest.approx(1.7778, abs=1e-3)

    # Standard 16:10
    geo_1610 = validate_display_geometry(1920, 1200)
    assert geo_1610.is_supported is True
    assert geo_1610.aspect_ratio == pytest.approx(1.6000, abs=1e-3)

    # Ultrawide 21:9
    geo_ultrawide = validate_display_geometry(2560, 1080)
    assert geo_ultrawide.is_supported is True
    assert geo_ultrawide.aspect_ratio == pytest.approx(2.3704, abs=1e-3)

    # Super-ultrawide 32:9
    geo_super_ultra = validate_display_geometry(3840, 1080)
    assert geo_super_ultra.is_supported is True
    assert geo_super_ultra.aspect_ratio == pytest.approx(3.5556, abs=1e-3)

    # Unsupported: Non-positive dimensions
    geo_zero_w = validate_display_geometry(0, 1080)
    assert geo_zero_w.is_supported is False
    assert "Invalid" in geo_zero_w.status_message

    geo_neg_h = validate_display_geometry(1920, -500)
    assert geo_neg_h.is_supported is False

    # Unsupported: Extreme aspect ratios
    geo_extreme_wide = validate_display_geometry(5000, 500)  # 10:1
    assert geo_extreme_wide.is_supported is False
    assert "extreme aspect ratio" in geo_extreme_wide.status_message.lower()

    geo_extreme_tall = validate_display_geometry(500, 2500)  # 0.2:1
    assert geo_extreme_tall.is_supported is False


def test_screen_calibration_manager_display_canvas_rendering():
    """
    Verify ScreenCalibrationManager renders high-resolution physical display canvas
    with bullseye target, progress indicators, and non-overlapping webcam inset.
    """
    mgr = ScreenCalibrationManager(samples_per_target=3, display_size=(1920, 1080))
    mgr.start()

    # Target 0 is Center
    coords = mgr.get_target_screen_coordinates()
    assert coords == (960, 540)

    webcam_preview = np.ones((480, 640, 3), dtype=np.uint8) * 120
    canvas = mgr.render_calibration_canvas(webcam_preview=webcam_preview)

    assert canvas.shape == (1080, 1920, 3)
    # Ensure canvas is not blank
    assert np.any(canvas > 30)

    # Verify target bullseye pixel is drawn around (960, 540)
    # Center pixel is white (255, 255, 255)
    assert np.array_equal(canvas[540, 960], [255, 255, 255])

    # Advance to target 1 (Top-Left)
    mgr.process_sample(CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"))
    mgr.process_sample(CombinedGazeResult(yaw=0.1, pitch=0.1, confidence=0.9, status="VALID"))
    mgr.process_sample(CombinedGazeResult(yaw=0.2, pitch=0.2, confidence=0.9, status="VALID"))

    assert mgr.current_target_idx == 1
    coords_tl = mgr.get_target_screen_coordinates()
    assert coords_tl == (192, 108)

    canvas_tl = mgr.render_calibration_canvas(webcam_preview=webcam_preview)
    assert np.array_equal(canvas_tl[108, 192], [255, 255, 255])


def test_fullscreen_window_lifecycle_and_cleanup_mocks(monkeypatch):
    """
    Verify dedicated calibration window lifecycle:
    - Creates fullscreen window on calibration start.
    - Updates window with calibration canvas during calibration.
    - Cleanly destroys window on cancellation, failure, or completion.
    """
    created_windows = []
    window_properties = []
    destroyed_windows = []
    imshow_windows = []

    import cv2 as cv2_real

    def mock_named_window(win_name, flags):
        created_windows.append((win_name, flags))

    def mock_set_window_property(win_name, prop_id, prop_val):
        window_properties.append((win_name, prop_id, prop_val))

    def mock_destroy_window(win_name):
        destroyed_windows.append(win_name)

    def mock_imshow(win_name, frame):
        imshow_windows.append(win_name)

    keys = [-1, -1, ord('q'), ord('q')]
    def mock_wait_key(t):
        return keys.pop(0) if keys else ord('q')

    monkeypatch.setattr("cv2.namedWindow", mock_named_window)
    monkeypatch.setattr("cv2.setWindowProperty", mock_set_window_property)
    monkeypatch.setattr("cv2.destroyWindow", mock_destroy_window)
    monkeypatch.setattr("cv2.imshow", mock_imshow)
    monkeypatch.setattr("cv2.waitKey", mock_wait_key)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.return_value = (True, frame.copy())
    cap.isOpened.return_value = True

    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(main, "calibrate_gaze_baseline", lambda *a, **k: (((0.5, 0.5), (0.5, 0.5)), {}))

    calib_res = SimpleNamespace(
        face_count=1,
        is_multi_face=False,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_SCREEN,
        head_pose_sample=(0.0, 0.0, 0.0),
        gaze_sample=((0.5, 0.5), (0.5, 0.5)),
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
        head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
        eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
        screen_gaze=None,
    )
    monkeypatch.setattr(main, "process_frame_face", lambda *a, **k: (calib_res, frame))
    monkeypatch.setattr(main, "PhoneDetectionTracker", lambda **k: SimpleNamespace(
        process_frame=lambda clean, annotate, display_frame: (
            display_frame, False, PhoneDetectionResult()
        )
    ))

    # Run surveillance with display=True and calibrate_screen=True
    main.run_surveillance(
        display=True,
        video_source=None,
        camera_index=0,
        cap=cap,
        max_frames=4,
        calibrate_screen=True,
        logger=MagicMock(session_id="test_sess"),
        gaze_logger=False,
    )

    # Verify dedicated fullscreen calibration window was created
    assert any(win[0] == WINDOW_CALIBRATION_NAME for win in created_windows)
    assert any(
        prop[0] == WINDOW_CALIBRATION_NAME
        and prop[1] == cv2_real.WND_PROP_FULLSCREEN
        and prop[2] == cv2_real.WINDOW_FULLSCREEN
        for prop in window_properties
    )
    assert WINDOW_CALIBRATION_NAME in imshow_windows

    # Verify dedicated calibration window was cleanly destroyed
    assert WINDOW_CALIBRATION_NAME in destroyed_windows
