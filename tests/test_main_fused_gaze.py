from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import config
import main
from eye_movement import GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UNCERTAIN
from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult
from mobile_detection import PhoneDetectionResult
from time_source import MockTimeSource

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("yaw", "pitch", "expected"),
    [
        (20.0, 0.0, GAZE_RIGHT),
        (-20.0, 0.0, GAZE_LEFT),
        (0.0, 0.0, GAZE_CENTER),
        (0.0, 15.0, "Looking Up"),
        (0.0, -15.0, "Looking Down"),
    ],
)
def test_camera_relative_fused_direction_mapping(yaw, pitch, expected):
    result = CombinedGazeResult(yaw=yaw, pitch=pitch, confidence=0.8, status="VALID")
    assert main.classify_combined_attention(result) == expected


@pytest.mark.parametrize(
    "result",
    [
        None,
        CombinedGazeResult(yaw=20.0, pitch=0.0, confidence=0.8, status="UNCERTAIN"),
        CombinedGazeResult(yaw=20.0, pitch=0.0, confidence=0.19, status="VALID"),
        CombinedGazeResult(yaw=None, pitch=0.0, confidence=0.8, status="VALID"),
        CombinedGazeResult(yaw=20.0, pitch=0.0, confidence=0.8, status="INVALID_ROTATION"),
    ],
)
def test_invalid_missing_or_low_confidence_fusion_is_uncertain(result):
    assert main.classify_combined_attention(result) == GAZE_UNCERTAIN


def test_fused_hud_shows_angles_confidence_and_unavailable_values():
    valid_text = main.combined_gaze_hud(
        CombinedGazeResult(yaw=12.3, pitch=-4.5, confidence=0.876, status="VALID")
    )
    assert "yaw +12.3° pitch -4.5°" in valid_text[0]
    assert "88%" in valid_text[1]

    missing_text = main.combined_gaze_hud(
        CombinedGazeResult(yaw=None, pitch=None, confidence=0.0, status="UNCERTAIN")
    )
    assert "yaw — pitch —" in missing_text[0]
    assert "0%" in missing_text[1]
    assert "UNAVAILABLE" in main.combined_gaze_hud(None)[0]


def test_run_loop_consumes_fused_attention_without_duplicate_opposing_events(monkeypatch):
    """A neutral fused vector vetoes opposing legacy head/eye directions in runtime."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.side_effect = [(True, frame.copy()) for _ in range(5)] + [(False, None)]
    cap.isOpened.return_value = True

    def face_result(*, calibrated, direction):
        return SimpleNamespace(
            face_count=1,
            is_multi_face=False,
            gaze_direction=direction,
            head_direction="Looking Right",
            gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
            head_pose_sample=(0.0, 0.0, 0.0) if not calibrated else None,
            gaze_sample=((0.5, 0.5), (0.5, 0.5)) if not calibrated else None,
            combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
            head_pose=HeadPoseResult(yaw=18.0, pitch=0.0, roll=0.0, confidence=0.9, status="VALID"),
            eye_gaze=EyeGazeResult(
                selected_eye_direction=direction, confidence=0.9,
                tracking_state="PAIRED_TRACKING", status="VALID",
            ),
        )

    results = [face_result(calibrated=False, direction=GAZE_LEFT)] + [
        face_result(calibrated=True, direction=GAZE_LEFT) for _ in range(4)
    ]
    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(main, "calibrate_gaze_baseline", lambda *a, **k: (((0.5, 0.5), (0.5, 0.5)), {}))
    monkeypatch.setattr(main, "process_frame_face", lambda *a, **k: (results.pop(0), frame.copy()))
    monkeypatch.setattr(main, "PhoneDetectionTracker", lambda **k: SimpleNamespace(
        process_frame=lambda clean, annotate, display_frame: (
            display_frame, False, PhoneDetectionResult()
        )
    ))
    record = MagicMock()
    monkeypatch.setattr(main, "record_violation_incident", record)
    time_source = MockTimeSource()
    time_source.update = lambda **kwargs: time_source.advance(1.0)

    main.run_surveillance(
        cap=cap, logger=MagicMock(session_id="test"), gaze_logger=False,
        display=False, max_frames=5, time_source=time_source,
    )

    # In Option A architecture, independent head-movement monitoring is restored:
    # A sustained non-neutral head pose triggers an independent head event, while the
    # neutral fused gaze prevents any fused directional violation.
    assert record.call_count == 1
    assert record.call_args.kwargs["violation_type"] == "head_Looking_Right"


def test_run_loop_logs_sustained_fused_direction(monkeypatch):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.side_effect = [(True, frame.copy()) for _ in range(5)] + [(False, None)]
    cap.isOpened.return_value = True
    def result_factory(*args, **kwargs):
        if kwargs.get("calibrated_angles") is None:
            res = SimpleNamespace(
                face_count=1, is_multi_face=False, gaze_direction=GAZE_CENTER,
                head_direction="Looking at Screen", head_pose_sample=(0.0, 0.0, 0.0),
                gaze_sample=((0.5, 0.5), (0.5, 0.5)),
                gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
                combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
                head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
                eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
            )
            return res, frame.copy()
        return SimpleNamespace(
            face_count=1, is_multi_face=False, gaze_direction=GAZE_CENTER,
            head_direction="Looking at Screen", gaze_diagnostics={},
            head_pose_sample=None, gaze_sample=None,
            combined_gaze=CombinedGazeResult(yaw=20.0, pitch=0.0, confidence=0.9, status="VALID"),
            head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
            eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
        ), frame.copy()

    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(main, "calibrate_gaze_baseline", lambda *a, **k: (((0.5, 0.5), (0.5, 0.5)), {}))
    monkeypatch.setattr(main, "process_frame_face", result_factory)
    monkeypatch.setattr(main, "PhoneDetectionTracker", lambda **k: SimpleNamespace(
        process_frame=lambda clean, annotate, display_frame: (
            display_frame, False, PhoneDetectionResult()
        )
    ))
    record = MagicMock()
    monkeypatch.setattr(main, "record_violation_incident", record)
    time_source = MockTimeSource()
    time_source.update = lambda **kwargs: time_source.advance(1.0)

    main.run_surveillance(
        cap=cap, logger=MagicMock(session_id="test"), gaze_logger=False,
        display=False, max_frames=5, time_source=time_source,
    )

    assert record.call_count == 1
    assert record.call_args.kwargs["violation_type"] == "eye_Looking_Right"
