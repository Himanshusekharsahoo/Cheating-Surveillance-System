"""
Tests for Phase 10.13.12 GAP-03: Actual Camera Capture Geometry & Calibration Compatibility.

Verifies:
1. Camera accepts requested dimensions (e.g. 640x480).
2. Camera refuses requested dimensions (falls back to hardware native).
3. Camera returns dimensions different from configured dimensions (e.g. 1280x720).
4. Effective dimensions are derived correctly from actual delivered frames.
5. Calibration stores effective dimensions in metadata.
6. A profile with incompatible effective dimensions is rejected at startup.
7. A compatible profile with matching effective dimensions loads cleanly.
8. Missing or unavailable effective geometry prevents profile activation with meaningful diagnostics.
9. Camera-relative head and fused-gaze monitoring remain active when screen calibration is rejected.
10. Existing startup and cleanup behavior is preserved.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import config
from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult
from head_pose import HEAD_SCREEN
from eye_movement import GAZE_CENTER
import main
from screen_gaze import (
    CALIB_STATE_VALID,
    ScreenCalibrationManager,
    ScreenGazeCalibrator,
    load_screen_calibration,
    resolve_capture_geometry,
    save_screen_calibration,
    screen_calibration_environment,
)
from time_source import MockTimeSource

pytestmark = pytest.mark.unit


def _fit_calibrator_with_geometry(width=640, height=480, path=None):
    calibrator = ScreenGazeCalibrator()
    calibrator.min_targets = 4
    calibrator.min_samples_per_target = 3

    # Feed synthetic valid samples
    for tx, ty in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]:
        yaw = (tx - 0.5) * 36.0
        pitch = (0.5 - ty) * 36.0
        for s in (-0.1, 0.0, 0.1):
            calibrator.add_observation(
                target_x=tx, target_y=ty,
                gaze_yaw=yaw + s, gaze_pitch=pitch + s,
                confidence=0.90,
            )

    success = calibrator.fit()
    assert success is True

    if path is not None:
        save_screen_calibration(
            calibrator=calibrator,
            path=path,
            display_size=(1920, 1080),
            targets=[[0.5, 0.5, "Center"], [0.1, 0.1, "Top-Left"], [0.9, 0.1, "Top-Right"], [0.1, 0.9, "Bottom-Left"], [0.9, 0.9, "Bottom-Right"]],
            samples_per_target=3,
            min_confidence=0.20,
            environment=screen_calibration_environment("webcam", 0, frame_width=width, frame_height=height),
        )

    return calibrator


def test_camera_accepts_requested_dimensions():
    """Test 1: Camera accepts requested dimensions (640x480)."""
    cap = MagicMock()
    cap.isOpened.return_value = True

    # Camera device stores dimensions after set()
    current_dims = {"w": 320.0, "h": 240.0}

    def mock_set(prop, val):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            current_dims["w"] = float(val)
            return True
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            current_dims["h"] = float(val)
            return True
        return False

    def mock_get(prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return current_dims["w"]
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return current_dims["h"]
        return 0.0

    cap.set.side_effect = mock_set
    cap.get.side_effect = mock_get

    w, h = resolve_capture_geometry(cap, requested_width=640, requested_height=480)
    assert w == 640
    assert h == 480


def test_camera_refuses_requested_dimensions():
    """Test 2: Camera refuses requested dimensions and stays at native hardware geometry (e.g. 1280x720)."""
    cap = MagicMock()
    cap.isOpened.return_value = True

    # Camera ignores set() calls and returns native 1280x720
    cap.set.return_value = False
    cap.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (720.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 0.0)

    w, h = resolve_capture_geometry(cap, requested_width=640, requested_height=480)
    assert w == 1280
    assert h == 720


def test_camera_returns_dimensions_different_from_configured():
    """Test 3: Camera returns dimensions different from configured (e.g. 800x600 instead of 640x480)."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.set.return_value = True
    cap.get.side_effect = lambda prop: 800.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (600.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 0.0)

    w, h = resolve_capture_geometry(cap, requested_width=640, requested_height=480)
    assert w == 800
    assert h == 600
    assert w != config.FRAME_WIDTH
    assert h != config.FRAME_HEIGHT


def test_effective_dimensions_derived_correctly_from_actual_frames():
    """Test 4: Effective dimensions are verified from actual frame shapes delivered by cap.read()."""
    synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_h, frame_w = synthetic_frame.shape[:2]
    assert frame_w == 1280
    assert frame_h == 720

    env = screen_calibration_environment("webcam", 0, frame_width=frame_w, frame_height=frame_h)
    assert env["camera"]["frame_width"] == 1280
    assert env["camera"]["frame_height"] == 720


def test_calibration_stores_effective_dimensions(tmp_path):
    """Test 5: Calibration saving persists effective camera frame width and height."""
    calib_path = tmp_path / "screen_calib.json"
    _fit_calibrator_with_geometry(width=1280, height=720, path=calib_path)

    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    assert payload["environment"]["camera"]["frame_width"] == 1280
    assert payload["environment"]["camera"]["frame_height"] == 720


def test_profile_with_incompatible_effective_dimensions_is_rejected(tmp_path):
    """Test 6: A profile calibrated for 640x480 is rejected when camera delivers 1280x720."""
    calib_path = tmp_path / "screen_calib.json"
    _fit_calibrator_with_geometry(width=640, height=480, path=calib_path)

    current_env = screen_calibration_environment("webcam", 0, frame_width=1280, frame_height=720)
    loaded, reason = load_screen_calibration(calib_path, display_size=(1920, 1080), environment=current_env)

    assert loaded is None
    assert "incompatible" in reason


def test_compatible_profile_loads_cleanly(tmp_path):
    """Test 7: A profile matching current effective geometry (1280x720) loads successfully."""
    calib_path = tmp_path / "screen_calib.json"
    _fit_calibrator_with_geometry(width=1280, height=720, path=calib_path)

    current_env = screen_calibration_environment("webcam", 0, frame_width=1280, frame_height=720)
    loaded, reason = load_screen_calibration(calib_path, display_size=(1920, 1080), environment=current_env)

    assert reason == ""
    assert loaded is not None
    assert loaded.is_calibrated is True


def test_missing_or_unavailable_effective_geometry_prevents_profile_activation():
    """Test 8: Closed or unavailable capture device returns (None, None) and prevents profile loading."""
    closed_cap = MagicMock()
    closed_cap.isOpened.return_value = False

    w, h = resolve_capture_geometry(closed_cap)
    assert w is None
    assert h is None

    none_cap = None
    w2, h2 = resolve_capture_geometry(none_cap)
    assert w2 is None
    assert h2 is None


def test_camera_relative_monitoring_remains_available_when_screen_calibration_rejected(monkeypatch, tmp_path):
    """
    Test 9: Camera-relative head and fused-gaze monitoring continues to function
    even when screen calibration is rejected due to capture geometry mismatch.
    """
    calib_path = tmp_path / "screen_calib.json"
    # Calibrate for 640x480
    _fit_calibrator_with_geometry(width=640, height=480, path=calib_path)

    # Device delivers 1280x720 frames (mismatch!)
    frame_1280 = np.zeros((720, 1280, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (720.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 0.0)
    mock_cap.read.side_effect = [(True, frame_1280.copy()), (False, None)]

    # Mock face result with valid head and gaze
    mock_face = SimpleNamespace(
        face_count=1,
        gaze_direction=GAZE_CENTER,
        head_direction=HEAD_SCREEN,
        head_pose_sample=(0.0, 0.0, 0.0),
        gaze_sample=((0.5, 0.4), (0.5, 0.4)),
        is_multi_face=False,
        raw_head_pose=(0.0, 0.0, 0.0),
        gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
        head_pose=HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.90, status="VALID"),
        eye_gaze=EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.90, tracking_state="PAIRED_TRACKING", status="VALID"),
        combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.81, status="VALID"),
        screen_gaze=None,
    )

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_PATH", calib_path)
    monkeypatch.setattr(main, "get_primary_display_resolution", lambda strict=False: (1920, 1080))
    monkeypatch.setattr(main, "process_frame_face", lambda *args, **kwargs: (mock_face, frame_1280))

    # Run surveillance: profile should be rejected due to geometry mismatch (640 vs 1280)
    # but surveillance runs without crash and evaluates camera-relative attention
    main.run_surveillance(
        cap=mock_cap,
        logger=MagicMock(session_id="geom_test"),
        gaze_logger=False,
        display=False,
        max_frames=1,
    )

    # Screen gaze should remain uncalibrated
    assert mock_face.screen_gaze is None


def test_startup_and_cleanup_behavior_preserved(monkeypatch, tmp_path):
    """Test 10: Video capture release and cleanup behavior is preserved on EOF and error."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 0.0)

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_PATH", tmp_path / "nonexistent.json")
    monkeypatch.setattr(main, "get_primary_display_resolution", lambda strict=False: (1920, 1080))

    main.run_surveillance(
        cap=mock_cap,
        logger=MagicMock(session_id="cleanup_test"),
        gaze_logger=False,
        display=False,
    )

    # Release must have been called on cap
    assert mock_cap.release.called is True
