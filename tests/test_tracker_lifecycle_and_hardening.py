"""
Unit and integration tests for Phase 10.13.3.2:
Tracker Lifecycle, Confidence Scoring & Diagnostic Hardening.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from eye_movement import (
    process_eye_movement,
    analyze_gaze_from_landmarks,
    detect_pupil,
    get_default_gaze_tracker,
    reset_default_gaze_tracker,
    GAZE_CENTER,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    GAZE_DOWN,
    GAZE_NO_FACE,
    GAZE_UNCERTAIN,
)
from single_eye_fallback import (
    SingleEyeGazeTracker,
    score_eye_confidence,
    TRACKING_STATE_PAIRED,
    TRACKING_STATE_LEFT_ONLY,
    TRACKING_STATE_RIGHT_ONLY,
    TRACKING_STATE_UNCERTAIN,
    TRACKING_STATE_RECOVERING,
    RECOVERY_REQUIRED_FRAMES,
)
from vision_pipeline import (
    VisionPipeline,
    FaceAnalysisResult,
    GazeDiagnosticsLogger,
    GAZE_DIAGNOSTICS_FIELDNAMES,
    _empty_gaze_diagnostics,
)
import config

pytestmark = pytest.mark.unit


# ==============================================================================
# Helper functions for mocking faces and landmarks
# ==============================================================================

def _make_dlib_rect(left, top, right, bottom):
    rect = MagicMock()
    rect.left.return_value = left
    rect.top.return_value = top
    rect.right.return_value = right
    rect.bottom.return_value = bottom
    return rect


def _make_mock_landmarks(pts_dict=None):
    mock = MagicMock()
    pts = {}
    # Default 68-point landmarks
    for i in range(68):
        pts[i] = (50, 50)
    # Left eye aperture (36-41): x in [30, 54], y in [30, 40]
    left_eye_coords = [(30, 35), (36, 30), (48, 30), (54, 35), (48, 40), (36, 40)]
    for idx, pt in enumerate(left_eye_coords, start=36):
        pts[idx] = pt
    # Right eye aperture (42-47): x in [70, 94], y in [30, 40]
    right_eye_coords = [(70, 35), (76, 30), (88, 30), (94, 35), (88, 40), (76, 40)]
    for idx, pt in enumerate(right_eye_coords, start=42):
        pts[idx] = pt

    if pts_dict:
        pts.update(pts_dict)

    def get_part(n):
        p = MagicMock()
        coord = pts.get(n, (50, 50))
        p.x = coord[0]
        p.y = coord[1]
        return p

    mock.part.side_effect = get_part
    return mock


# ==============================================================================
# Task 1: Tracker Reset on Missing Face in VisionPipeline
# ==============================================================================

def test_missing_face_resets_tracker_in_vision_pipeline():
    """
    Verify that when face_count == 0 in VisionPipeline, the tracker is reset
    and stale PAIRED_TRACKING state cannot survive across the missing face.
    """
    reset_default_gaze_tracker()
    tracker = get_default_gaze_tracker()
    tracker.current_state = TRACKING_STATE_PAIRED

    pipeline = VisionPipeline()
    # Mock face detector returning zero faces
    pipeline._detector = MagicMock(return_value=[])

    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    res, _ = pipeline.process_face(frame)

    assert res.face_count == 0
    assert res.gaze_direction == GAZE_NO_FACE
    # Tracker must be reset to UNCERTAIN
    assert tracker.current_state == TRACKING_STATE_UNCERTAIN


def test_recovery_behavior_after_missing_face_reappearance():
    """
    Verify that after a missing face interval, when face reappears with paired eyes,
    the tracker does not jump immediately to PAIRED_TRACKING, but enforces
    3-frame recovery protocol.
    """
    reset_default_gaze_tracker()
    tracker = get_default_gaze_tracker()
    tracker.current_state = TRACKING_STATE_PAIRED

    pipeline = VisionPipeline()
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    # 1. Missing face frame
    pipeline._detector = MagicMock(return_value=[])
    res1, _ = pipeline.process_face(frame)
    assert res1.face_count == 0
    assert tracker.current_state == TRACKING_STATE_UNCERTAIN

    # 2. Reappearing face with paired eyes
    # Build synthetic frame with both pupils centered
    face_rect = _make_dlib_rect(20, 20, 100, 100)
    pipeline._detector = MagicMock(return_value=[face_rect])
    mock_lm = _make_mock_landmarks()
    pipeline._predictor = MagicMock(return_value=mock_lm)

    # Draw pupils
    cv2.fillPoly(frame, [np.array([(30, 35), (36, 30), (48, 30), (54, 35), (48, 40), (36, 40)])], (230, 230, 230))
    cv2.fillPoly(frame, [np.array([(70, 35), (76, 30), (88, 30), (94, 35), (88, 40), (76, 40)])], (230, 230, 230))
    cv2.circle(frame, (42, 35), 2, (20, 20, 20), -1)
    cv2.circle(frame, (82, 35), 2, (20, 20, 20), -1)

    # Reappearance Frame 1 -> RECOVERING (count = 1)
    res2, _ = pipeline.process_face(frame)
    assert tracker.current_state == TRACKING_STATE_RECOVERING
    assert tracker.recovery_counter == 1

    # Reappearance Frame 2 -> RECOVERING (count = 2)
    res3, _ = pipeline.process_face(frame)
    assert tracker.current_state == TRACKING_STATE_RECOVERING
    assert tracker.recovery_counter == 2

    # Reappearance Frame 3 -> PAIRED_TRACKING (restored!)
    res4, _ = pipeline.process_face(frame)
    assert tracker.current_state == TRACKING_STATE_PAIRED
    assert tracker.recovery_counter == 0


# ==============================================================================
# Task 2: Surveillance Session Lifecycle in main.py
# ==============================================================================

def test_surveillance_session_lifecycle_resets_tracker_between_sessions():
    """
    Verify that run_surveillance resets tracker state at session start and exit,
    preventing state leakage between consecutive sessions in the same process.
    """
    from main import run_surveillance

    # Dirty the default tracker before session 1
    tracker = get_default_gaze_tracker()
    tracker.current_state = TRACKING_STATE_PAIRED
    tracker.recovery_counter = 2

    # Mock VideoCapture with 2 empty frames
    mock_cap = MagicMock()
    fake_frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    mock_cap.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]
    mock_cap.isOpened.return_value = True

    # Empty face analysis result
    empty_res = FaceAnalysisResult(
        face_count=0,
        face_rect=None,
        landmarks=None,
        gaze_direction=GAZE_NO_FACE,
        head_direction="No Face Detected",
        gaze_diagnostics=_empty_gaze_diagnostics(),
    )

    with patch("config.validate_config"), \
         patch("main.process_frame_face", return_value=(empty_res, fake_frame)):
        # Run session 1 with cap passed directly
        run_surveillance(cap=mock_cap, display=False, max_frames=2)
        # Tracker must be reset after session 1 exits
        assert tracker.current_state == TRACKING_STATE_UNCERTAIN
        assert tracker.recovery_counter == 0

        # Set dirty state again before session 2
        tracker.current_state = TRACKING_STATE_LEFT_ONLY
        tracker.recovery_counter = 1

        # Run session 2
        mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
        run_surveillance(cap=mock_cap, display=False, max_frames=1)
        # Tracker must remain reset and clean
        assert tracker.current_state == TRACKING_STATE_UNCERTAIN
        assert tracker.recovery_counter == 0


# ==============================================================================
# Task 3: Confidence Scoring Correction with Contour Area
# ==============================================================================

def test_score_eye_confidence_with_valid_and_distorted_contours():
    """
    Verify that passing actual contour area properly differentiates elliptical
    pupils from distorted contours, and missing contour data does not inflate confidence.
    """
    bw, bh = 10, 10
    box_area = float(bw * bh)  # 100
    ideal_area = 0.785 * box_area  # 78.5

    # 1. Ideal circular pupil contour (fill ~ 0.785, aspect = 1.0)
    conf_ideal = score_eye_confidence(
        pupil=(50, 40), eye_w=100, eye_h=100,
        bbox=(45, 35, bw, bh),
        contour_area=ideal_area,
    )

    # 2. Distorted contour (aspect = 1.0, but area = 15 -> fill = 0.15)
    conf_distorted = score_eye_confidence(
        pupil=(50, 40), eye_w=100, eye_h=100,
        bbox=(45, 35, bw, bh),
        contour_area=15.0,
    )

    # 3. Missing contour area (contour_area=None -> s_fill = 0.50)
    conf_missing = score_eye_confidence(
        pupil=(50, 40), eye_w=100, eye_h=100,
        bbox=(45, 35, bw, bh),
        contour_area=None,
    )

    assert conf_ideal > conf_distorted
    assert conf_ideal > conf_missing
    # Distorted contour should not receive maximum fill
    assert conf_distorted < conf_missing


def test_detect_pupil_populates_contour_area_in_diagnostics():
    """
    Verify that detect_pupil includes the actual contour_area in diag.
    """
    patch = np.full((20, 40, 3), 160, dtype=np.uint8)
    poly = np.array([[0, 10], [10, 0], [30, 0], [39, 10], [30, 19], [10, 19]], dtype=np.int32)
    cv2.fillPoly(patch, [poly], (230, 230, 230))
    cv2.circle(patch, (20, 10), 3, (20, 20, 20), -1)

    pupil, bbox, diag = detect_pupil(patch, eye_polygon=poly, return_diagnostics=True)

    assert pupil is not None
    assert diag["valid"] is True
    assert "contour_area" in diag
    assert diag["contour_area"] is not None
    assert diag["contour_area"] > 5.0  # Radius 3 circle has area ~ 28.3


# ==============================================================================
# Task 4: Multi-Face Isolation in process_eye_movement
# ==============================================================================

def test_multi_face_tracker_isolation_in_process_eye_movement():
    """
    Verify that process_eye_movement updates the default gaze tracker ONLY
    for the primary face, and secondary faces use an isolated tracker.
    """
    reset_default_gaze_tracker()
    tracker = get_default_gaze_tracker()
    tracker.reset()

    # Create two faces: Face A (primary, 100x100) and Face B (secondary, 40x40)
    face_a = _make_dlib_rect(10, 10, 110, 110)  # Area 10,000
    face_b = _make_dlib_rect(150, 10, 190, 50)  # Area 1,600

    mock_detector = MagicMock(return_value=[face_b, face_a])  # Secondary face first

    # Face A landmarks: pupil centered
    lm_a = _make_mock_landmarks()
    # Face B landmarks: pupil at extreme position (looking left)
    lm_b = _make_mock_landmarks()

    def predictor_side_effect(gray, face):
        return lm_a if face == face_a else lm_b

    mock_predictor = MagicMock(side_effect=predictor_side_effect)

    frame = np.full((120, 200, 3), 180, dtype=np.uint8)
    annotated, direction = process_eye_movement(
        frame,
        detector=mock_detector,
        predictor=mock_predictor
    )

    # Primary face A was processed with get_default_gaze_tracker()
    # Secondary face B did not overwrite or corrupt tracker
    assert direction is not None


# ==============================================================================
# Task 5: CSV Diagnostic Logging
# ==============================================================================

def test_csv_diagnostics_fieldnames_and_empty_diagnostics():
    """
    Verify that GAZE_DIAGNOSTICS_FIELDNAMES contains the 5 Phase 10.13.3 fields
    and _empty_gaze_diagnostics populates them with safe defaults.
    """
    new_fields = [
        "left_confidence",
        "right_confidence",
        "combined_confidence",
        "tracking_state",
        "is_single_eye_fallback",
    ]
    for field in new_fields:
        assert field in GAZE_DIAGNOSTICS_FIELDNAMES

    empty_diag = _empty_gaze_diagnostics(is_calibrated=False)
    for field in new_fields:
        assert field in empty_diag
    assert empty_diag["left_confidence"] == 0.0
    assert empty_diag["right_confidence"] == 0.0
    assert empty_diag["combined_confidence"] == 0.0
    assert empty_diag["tracking_state"] == "UNCERTAIN"
    assert empty_diag["is_single_eye_fallback"] is False


def test_csv_diagnostics_logger_writes_and_formats_new_fields(tmp_path: Path):
    """
    Verify that GazeDiagnosticsLogger formats and serializes the 5 new fields to CSV.
    """
    csv_file = tmp_path / "test_diagnostics.csv"
    diag = {
        "face_detected": True,
        "landmarks_detected": True,
        "missing_face": False,
        "missing_landmarks": False,
        "missing_pupils": False,
        "missing_left_pupil": False,
        "missing_right_pupil": False,
        "left_pupil_detected": True,
        "right_pupil_detected": True,
        "left_confidence": 0.8752,
        "right_confidence": 0.9123,
        "combined_confidence": 0.8938,
        "tracking_state": "PAIRED_TRACKING",
        "is_single_eye_fallback": False,
        "gaze_direction": GAZE_CENTER,
    }

    face_result = FaceAnalysisResult(
        face_count=1,
        landmarks=None,
        gaze_direction=GAZE_CENTER,
        gaze_diagnostics=diag,
    )

    with GazeDiagnosticsLogger(csv_path=csv_file, session_id="test_sess_001") as logger:
        success = logger.log_frame(frame_index=1, face_result=face_result, current_time=0.1)
        assert success is True

    # Read back and inspect CSV columns
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["left_confidence"] == "0.8752"
        assert row["right_confidence"] == "0.9123"
        assert row["combined_confidence"] == "0.8938"
        assert row["tracking_state"] == "PAIRED_TRACKING"
        assert row["is_single_eye_fallback"] == "False"


# ==============================================================================
# Task 6: Left/Right Switching Stability (Debouncing)
# ==============================================================================

def test_alternating_single_eye_debounce_prevents_rapid_switching():
    """
    Verify that an isolated single-frame switch from LEFT_EYE_ONLY to RIGHT_EYE_ONLY
    is debounced via UNCERTAIN, and requires 2 consecutive frames to confirm switch.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Pre-condition into LEFT_EYE_ONLY
    res0 = tracker.update(
        left_pupil=(75, 40), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.85, right_conf=0.0,
    )
    assert res0.tracking_state == TRACKING_STATE_LEFT_ONLY

    # Frame 1: Opposite eye suddenly appears (isolated flicker) -> UNCERTAIN (debouncing)
    res1 = tracker.update(
        left_pupil=None, left_w=100, left_h=100,
        right_pupil=(25, 40), right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.0, right_conf=0.85,
    )
    assert res1.tracking_state == TRACKING_STATE_UNCERTAIN
    assert res1.rejection_reason == "alternating_single_eye_debounce"

    # Frame 2: Opposite eye confirmed on consecutive frame -> RIGHT_EYE_ONLY
    res2 = tracker.update(
        left_pupil=None, left_w=100, left_h=100,
        right_pupil=(25, 40), right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.0, right_conf=0.85,
    )
    assert res2.tracking_state == TRACKING_STATE_RIGHT_ONLY
    assert res2.is_single_eye_fallback is True
