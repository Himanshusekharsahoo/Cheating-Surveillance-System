"""
Tests for Phase 10.13.8.1: Independent Head Monitoring Restoration & Fused Overall-Gaze Separation.

Verifies Option A architecture:
- Head movement is monitored independently and generates its own violation events.
- Fused head-eye gaze independently determines overall camera-relative visual attention.
- Independent eye diagnostics remain intact.
- Separate timers, latching, and false-positive protections operate correctly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest

import config
import main
from eye_movement import GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UNCERTAIN, GAZE_NO_FACE
from head_pose import HEAD_SCREEN, HEAD_LEFT, HEAD_RIGHT, HEAD_UP, HEAD_DOWN, HEAD_TILTED, HEAD_NO_FACE
from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult
from mobile_detection import PhoneDetectionResult
from time_source import MockTimeSource

pytestmark = pytest.mark.unit


def _build_test_frame_result(
    head_direction=HEAD_SCREEN,
    gaze_direction=GAZE_CENTER,
    combined_yaw=0.0,
    combined_pitch=0.0,
    combined_status="VALID",
    combined_confidence=0.85,
    head_status="VALID",
    head_confidence=0.90,
    eye_status="VALID",
    eye_confidence=0.85,
    face_count=1,
    left_pupil_valid=True,
    right_pupil_valid=True,
):
    """Constructs a deterministic mock FaceAnalysisResult for runtime testing."""
    if face_count == 0:
        return SimpleNamespace(
            face_count=0,
            is_multi_face=False,
            gaze_direction=GAZE_NO_FACE,
            head_direction=HEAD_NO_FACE,
            gaze_diagnostics={"missing_face": True, "missing_pupils": True},
            head_pose_sample=None,
            gaze_sample=None,
            combined_gaze=CombinedGazeResult(status="NO_FACE", confidence=0.0),
            head_pose=HeadPoseResult(status="NO_FACE", confidence=0.0),
            eye_gaze=EyeGazeResult(status="NO_FACE", confidence=0.0),
        )

    return SimpleNamespace(
        face_count=face_count,
        is_multi_face=(face_count > 1),
        gaze_direction=gaze_direction,
        head_direction=head_direction,
        gaze_diagnostics={
            "left_pupil_valid": left_pupil_valid,
            "right_pupil_valid": right_pupil_valid,
            "missing_left_pupil": not left_pupil_valid,
            "missing_right_pupil": not right_pupil_valid,
        },
        head_pose_sample=None,
        gaze_sample=None,
        combined_gaze=CombinedGazeResult(
            yaw=combined_yaw,
            pitch=combined_pitch,
            confidence=combined_confidence,
            status=combined_status,
        ),
        head_pose=HeadPoseResult(
            yaw=0.0 if head_direction == HEAD_SCREEN else -20.0,
            pitch=0.0,
            roll=0.0,
            confidence=head_confidence,
            status=head_status,
        ),
        eye_gaze=EyeGazeResult(
            selected_eye_direction=gaze_direction,
            confidence=eye_confidence,
            tracking_state="PAIRED_TRACKING" if eye_status == "VALID" else "UNCERTAIN",
            status=eye_status,
        ),
    )


def _run_mock_surveillance(monkeypatch, results, frame_count=5, step_sec=1.0):
    """Executes run_surveillance with mocked environment and returns recorded violation calls."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.side_effect = [(True, frame.copy()) for _ in range(frame_count)] + [(False, None)]
    cap.isOpened.return_value = True

    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 1)
    monkeypatch.setattr(main, "calibrate_gaze_baseline", lambda *a, **k: (((0.5, 0.5), (0.5, 0.5)), {}))

    # Pre-calibrated results queue
    res_queue = list(results)
    def mock_process_frame_face(display_frame, calibrated_angles=None, calibrated_gaze=None, **kwargs):
        if calibrated_angles is None:
            # Calibration frame
            calib_res = SimpleNamespace(
                face_count=1, is_multi_face=False, gaze_direction=GAZE_CENTER,
                head_direction="Looking at Screen", head_pose_sample=(0.0, 0.0, 0.0),
                gaze_sample=((0.5, 0.5), (0.5, 0.5)),
                gaze_diagnostics={"left_pupil_valid": True, "right_pupil_valid": True},
                combined_gaze=CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="VALID"),
                head_pose=HeadPoseResult(confidence=0.9, status="VALID"),
                eye_gaze=EyeGazeResult(confidence=0.9, status="VALID", tracking_state="PAIRED_TRACKING"),
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
        display=False,
        max_frames=frame_count,
        time_source=time_source,
    )
    return recorded_incidents


# =========================================================================
# Scenario 1: Head Left, Eye Right, Combined Forward
# Expected: Independent head event triggers; no fused directional event.
# =========================================================================
def test_scenario_1_head_left_eye_right_combined_forward_sustained(monkeypatch):
    """Sustained head-turn with counter-rotated eye gaze triggers head event, not fused event."""
    res = _build_test_frame_result(
        head_direction=HEAD_LEFT,
        gaze_direction=GAZE_RIGHT,
        combined_yaw=0.0,  # Fused gaze is straight ahead
        combined_pitch=0.0,
        combined_status="VALID",
    )
    # 5 frames with 1s step -> 0s, 1s, 2s, 3s, 4s (sustained >= 3s)
    incidents = _run_mock_surveillance(monkeypatch, [res] * 5, frame_count=5, step_sec=1.0)

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "head_Looking_Left" in violation_types
    assert not any(v.startswith("eye_Looking_") for v in violation_types)


def test_scenario_1_head_left_eye_right_combined_forward_transient(monkeypatch):
    """Transient head-turn (< 3s) triggers no violation."""
    res = _build_test_frame_result(
        head_direction=HEAD_LEFT,
        gaze_direction=GAZE_RIGHT,
        combined_yaw=0.0,
        combined_pitch=0.0,
    )
    # 2 frames with 1s step -> elapsed 1.0s < 3.0s threshold
    incidents = _run_mock_surveillance(monkeypatch, [res] * 2, frame_count=2, step_sec=1.0)
    assert len(incidents) == 0


# =========================================================================
# Scenario 2: Head Left, Eye Left, Combined Left
# Expected: Head and fused directional monitoring both active.
# =========================================================================
def test_scenario_2_head_left_eye_left_combined_left_sustained(monkeypatch):
    """Both head and fused gaze sustained left trigger both independent violations."""
    res = _build_test_frame_result(
        head_direction=HEAD_LEFT,
        gaze_direction=GAZE_LEFT,
        combined_yaw=-25.0,  # Fused camera yaw left
        combined_pitch=0.0,
    )
    incidents = _run_mock_surveillance(monkeypatch, [res] * 5, frame_count=5, step_sec=1.0)

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "head_Looking_Left" in violation_types
    assert "eye_Looking_Left" in violation_types
    assert len(incidents) == 2


# =========================================================================
# Scenario 3: Head Forward, Eye Right, Combined Right
# Expected: Fused right-direction monitoring active.
# =========================================================================
def test_scenario_3_head_forward_eye_right_combined_right_sustained(monkeypatch):
    """Head forward with eye glancing right triggers fused right event, no head event."""
    res = _build_test_frame_result(
        head_direction=HEAD_SCREEN,
        gaze_direction=GAZE_RIGHT,
        combined_yaw=20.0,  # Fused camera yaw right
        combined_pitch=0.0,
    )
    incidents = _run_mock_surveillance(monkeypatch, [res] * 5, frame_count=5, step_sec=1.0)

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "eye_Looking_Right" in violation_types
    assert not any(v.startswith("head_") for v in violation_types)


# =========================================================================
# Scenario 4: Head Left, Eye Missing, Combined Uncertain
# Expected: Head monitored independently; fused event not confirmed.
# =========================================================================
def test_scenario_4_head_left_eye_missing_combined_uncertain(monkeypatch):
    """Head looking left with uncertain eye gaze generates head event, but no fused event."""
    res = _build_test_frame_result(
        head_direction=HEAD_LEFT,
        gaze_direction=GAZE_UNCERTAIN,
        combined_yaw=None,
        combined_pitch=None,
        combined_status="UNCERTAIN",
        eye_status="UNCERTAIN",
        left_pupil_valid=False,
        right_pupil_valid=False,
    )
    incidents = _run_mock_surveillance(monkeypatch, [res] * 5, frame_count=5, step_sec=1.0)

    violation_types = [inc["violation_type"] for inc in incidents]
    assert "head_Looking_Left" in violation_types
    # Fused directional event must NOT be confirmed
    assert not any(v in ("eye_Looking_Left", "eye_Looking_Right", "eye_Looking_Up", "eye_Looking_Down") for v in violation_types)


# =========================================================================
# Scenario 5: Head Missing, Eye Right, Combined Uncertain
# Expected: Fused event not confirmed; eye diagnostic retained.
# =========================================================================
def test_scenario_5_head_missing_eye_right_combined_uncertain(monkeypatch):
    """Missing head pose (UNCERTAIN status) does not produce false head or fused events."""
    res = _build_test_frame_result(
        head_direction=HEAD_LEFT,  # Raw label might exist, but status is UNCERTAIN
        gaze_direction=GAZE_RIGHT,
        combined_yaw=None,
        combined_pitch=None,
        combined_status="UNCERTAIN",
        head_status="UNCERTAIN",
        head_confidence=0.0,
    )
    incidents = _run_mock_surveillance(monkeypatch, [res] * 5, frame_count=5, step_sec=1.0)

    # Neither head nor fused directional event should be confirmed due to UNCERTAIN status
    assert len(incidents) == 0


# =========================================================================
# Scenario 6: Head Forward, Eye Forward, Combined Forward
# Expected: No directional violation.
# =========================================================================
def test_scenario_6_all_forward_compliant(monkeypatch):
    """All signals compliant forward produce zero violations."""
    res = _build_test_frame_result(
        head_direction=HEAD_SCREEN,
        gaze_direction=GAZE_CENTER,
        combined_yaw=0.0,
        combined_pitch=0.0,
        combined_status="VALID",
    )
    incidents = _run_mock_surveillance(monkeypatch, [res] * 5, frame_count=5, step_sec=1.0)
    assert len(incidents) == 0


# =========================================================================
# Additional Stage 1 Requirements: Latching, Face Loss, Low Confidence
# =========================================================================
def test_incident_latching_prevents_duplicate_spam(monkeypatch):
    """Sustained non-compliant condition produces exactly 1 incident without per-frame spam."""
    res = _build_test_frame_result(
        head_direction=HEAD_LEFT,
        gaze_direction=GAZE_CENTER,
        combined_yaw=0.0,
        combined_pitch=0.0,
    )
    # Run 10 frames (10 seconds)
    incidents = _run_mock_surveillance(monkeypatch, [res] * 10, frame_count=10, step_sec=1.0)
    head_events = [inc for inc in incidents if inc["violation_type"] == "head_Looking_Left"]
    assert len(head_events) == 1


def test_face_loss_resets_independent_head_and_eye_timers(monkeypatch):
    """Face loss resets head and fused timers; returning requires new 3s accumulation."""
    res_bad = _build_test_frame_result(
        head_direction=HEAD_LEFT,
        gaze_direction=GAZE_LEFT,
        combined_yaw=-25.0,
    )
    res_noface = _build_test_frame_result(face_count=0)

    # Sequence: 2s bad, 1s no face (resets), 2s bad (only 2s accumulated, threshold is 3s)
    sequence = [res_bad, res_bad, res_noface, res_bad, res_bad]
    incidents = _run_mock_surveillance(monkeypatch, sequence, frame_count=5, step_sec=1.0)

    directional_events = [
        inc["violation_type"] for inc in incidents
        if inc["violation_type"] in ("head_Looking_Left", "eye_Looking_Left")
    ]
    assert len(directional_events) == 0
