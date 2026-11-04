"""
Phase 10.13.13 — GAP-04 Focused Unit Tests:
Face Detector and Landmark Predictor Exception Handling.

Verifies:
1. Face detector raises exception -> does not crash, returns structured DETECTION_ERROR result.
2. Landmark predictor raises exception -> does not crash, returns structured PREDICTION_ERROR result.
3. Detector fails on frame 1 and succeeds on frame 2 (normal transient recovery).
4. Predictor fails on frame 1 and succeeds on frame 2 (normal transient recovery).
5. No stale face, landmark, eye, or head measurements leak into failed frames.
6. Distinction preserved between NO_FACE, DETECTION_ERROR, and PREDICTION_ERROR.
7. Downstream attention classification returns GAZE_UNCERTAIN (never GAZE_CENTER or directional).
8. Downstream gaze occlusion does not trigger false occlusion on detector/predictor error.
9. Diagnostic log messages are rate-limited / throttled against per-frame spam.
10. Existing valid face processing remains completely unchanged.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import config
from vision_pipeline import VisionPipeline, FaceAnalysisResult, process_frame_face
from gaze_fusion import HeadPoseResult, EyeGazeResult, CombinedGazeResult
from eye_movement import GAZE_NO_FACE, GAZE_UNCERTAIN, GAZE_CENTER
from head_pose import HEAD_NO_FACE, HEAD_SCREEN
from main import classify_combined_attention, gaze_occlusion_is_confirmed

pytestmark = pytest.mark.unit


def _make_dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeRect:
    def left(self): return 100
    def top(self): return 100
    def right(self): return 200
    def bottom(self): return 200


def test_detector_exception_returns_structured_detection_error():
    """Verify that a face detector exception produces a structured error result without terminating."""
    failing_detector = MagicMock(side_effect=RuntimeError("Hardware detector failure!"))
    mock_predictor = MagicMock()
    pipe = VisionPipeline(detector=failing_detector, predictor=mock_predictor)

    frame = _make_dummy_frame()
    result, returned_frame = pipe.process_face(frame)

    assert isinstance(result, FaceAnalysisResult)
    assert result.face_count == 0
    assert result.landmarks is None
    assert result.processing_status == "DETECTION_ERROR"
    assert "Hardware detector failure!" in result.error
    assert result.head_direction == "DETECTION_ERROR"
    assert result.gaze_direction == GAZE_UNCERTAIN
    assert result.head_pose.status == "DETECTION_ERROR"
    assert result.eye_gaze.status == "DETECTION_ERROR"
    assert result.combined_gaze.status == "DETECTION_ERROR"
    assert result.combined_gaze.confidence == 0.0
    assert result.gaze_diagnostics["detection_error"] == "Hardware detector failure!"
    assert mock_predictor.called is False


def test_predictor_exception_returns_structured_prediction_error():
    """Verify that a landmark predictor exception produces a structured error result without terminating."""
    mock_detector = MagicMock(return_value=[_FakeRect()])
    failing_predictor = MagicMock(side_effect=ValueError("Corrupted face landmarks model!"))
    pipe = VisionPipeline(detector=mock_detector, predictor=failing_predictor)

    frame = _make_dummy_frame()
    result, returned_frame = pipe.process_face(frame)

    assert isinstance(result, FaceAnalysisResult)
    assert result.face_count == 1
    assert result.face_rect is not None
    assert result.landmarks is None
    assert result.processing_status == "PREDICTION_ERROR"
    assert "Corrupted face landmarks model!" in result.error
    assert result.head_direction == "PREDICTION_ERROR"
    assert result.gaze_direction == GAZE_UNCERTAIN
    assert result.head_pose.status == "PREDICTION_ERROR"
    assert result.eye_gaze.status == "PREDICTION_ERROR"
    assert result.combined_gaze.status == "PREDICTION_ERROR"
    assert result.combined_gaze.confidence == 0.0
    assert result.gaze_diagnostics["prediction_error"] == "Corrupted face landmarks model!"
    assert result.gaze_diagnostics["face_detected"] is True


def test_transient_detector_exception_recovery():
    """Verify that detector recovers normally on frame 2 after a transient exception on frame 1."""
    detector = MagicMock(side_effect=[
        RuntimeError("Transient detector glitch"),
        [_FakeRect()],
    ])
    mock_predictor = MagicMock()
    # Mock landmark part points
    mock_parts = MagicMock()
    mock_parts.part.return_value = MagicMock(x=150, y=150)
    mock_predictor.return_value = mock_parts

    pipe = VisionPipeline(detector=detector, predictor=mock_predictor)
    frame = _make_dummy_frame()

    # Frame 1: detector failure
    res1, _ = pipe.process_face(frame)
    assert res1.processing_status == "DETECTION_ERROR"

    # Frame 2: detector succeeds
    res2, _ = pipe.process_face(frame)
    assert res2.processing_status != "DETECTION_ERROR"
    assert res2.face_count == 1
    assert res2.landmarks is not None


def test_transient_predictor_exception_recovery():
    """Verify that predictor recovers normally on frame 2 after a transient exception on frame 1."""
    mock_detector = MagicMock(return_value=[_FakeRect()])
    mock_parts = MagicMock()
    mock_parts.part.return_value = MagicMock(x=150, y=150)

    predictor = MagicMock(side_effect=[
        RuntimeError("Transient predictor glitch"),
        mock_parts,
    ])

    pipe = VisionPipeline(detector=mock_detector, predictor=predictor)
    frame = _make_dummy_frame()

    # Frame 1: predictor failure
    res1, _ = pipe.process_face(frame)
    assert res1.processing_status == "PREDICTION_ERROR"
    assert res1.landmarks is None

    # Frame 2: predictor succeeds
    res2, _ = pipe.process_face(frame)
    assert res2.processing_status != "PREDICTION_ERROR"
    assert res2.landmarks is not None


def test_stale_data_isolation_between_frames():
    """Verify that measurements from a successful frame never leak into a subsequent failed frame."""
    mock_detector = MagicMock(side_effect=[
        [_FakeRect()],  # Frame 1 succeeds
        RuntimeError("Detector crash on frame 2"),  # Frame 2 fails
    ])
    mock_parts = MagicMock()
    mock_parts.part.return_value = MagicMock(x=150, y=150)
    mock_predictor = MagicMock(return_value=mock_parts)

    pipe = VisionPipeline(detector=mock_detector, predictor=mock_predictor)
    frame = _make_dummy_frame()

    # Frame 1
    res1, _ = pipe.process_face(frame)
    assert res1.face_count == 1
    assert res1.landmarks is not None

    # Frame 2: fails
    res2, _ = pipe.process_face(frame)
    assert res2.processing_status == "DETECTION_ERROR"
    assert res2.face_count == 0
    assert res2.landmarks is None
    assert res2.head_pose_sample is None
    assert res2.gaze_sample is None
    assert res2.raw_head_pose is None
    assert res2.head_pose.rotation_matrix is None
    assert res2.head_pose.translation is None
    assert res2.combined_gaze.confidence == 0.0


def test_distinction_between_no_face_and_processing_errors():
    """Verify clear distinction between NO_FACE, DETECTION_ERROR, and PREDICTION_ERROR."""
    # 1. NO_FACE
    empty_detector = MagicMock(return_value=[])
    pipe_no_face = VisionPipeline(detector=empty_detector, predictor=MagicMock())
    res_no_face, _ = pipe_no_face.process_face(_make_dummy_frame())
    assert res_no_face.processing_status == "NO_FACE"
    assert res_no_face.head_pose.status == "NO_FACE"
    assert res_no_face.eye_gaze.status == "NO_FACE"
    assert res_no_face.head_direction == HEAD_NO_FACE
    assert res_no_face.gaze_direction == GAZE_NO_FACE

    # 2. DETECTION_ERROR
    err_detector = MagicMock(side_effect=RuntimeError("det_err"))
    pipe_det_err = VisionPipeline(detector=err_detector, predictor=MagicMock())
    res_det_err, _ = pipe_det_err.process_face(_make_dummy_frame())
    assert res_det_err.processing_status == "DETECTION_ERROR"
    assert res_det_err.head_pose.status == "DETECTION_ERROR"
    assert res_det_err.eye_gaze.status == "DETECTION_ERROR"
    assert res_det_err.head_direction != HEAD_NO_FACE
    assert res_det_err.gaze_direction != GAZE_NO_FACE

    # 3. PREDICTION_ERROR
    err_predictor = MagicMock(side_effect=RuntimeError("pred_err"))
    pipe_pred_err = VisionPipeline(detector=MagicMock(return_value=[_FakeRect()]), predictor=err_predictor)
    res_pred_err, _ = pipe_pred_err.process_face(_make_dummy_frame())
    assert res_pred_err.processing_status == "PREDICTION_ERROR"
    assert res_pred_err.head_pose.status == "PREDICTION_ERROR"
    assert res_pred_err.eye_gaze.status == "PREDICTION_ERROR"
    assert res_pred_err.face_count == 1  # Face was found!
    assert res_pred_err.head_direction != HEAD_NO_FACE


def test_downstream_fusion_never_produces_compliant_or_directional_gaze():
    """Verify that DETECTION_ERROR and PREDICTION_ERROR produce GAZE_UNCERTAIN in classify_combined_attention."""
    det_err_comb = CombinedGazeResult(status="DETECTION_ERROR", confidence=0.0, uncertainty=1.0)
    pred_err_comb = CombinedGazeResult(status="PREDICTION_ERROR", confidence=0.0, uncertainty=1.0)

    assert classify_combined_attention(det_err_comb) == GAZE_UNCERTAIN
    assert classify_combined_attention(pred_err_comb) == GAZE_UNCERTAIN
    assert classify_combined_attention(det_err_comb) != GAZE_CENTER
    assert classify_combined_attention(pred_err_comb) != GAZE_CENTER


def test_gaze_occlusion_not_falsely_triggered_by_processing_errors():
    """Verify that gaze occlusion is never confirmed when processing errors occur."""
    diag_det_err = {
        "missing_left_pupil": True,
        "missing_right_pupil": True,
        "detection_error": "Some detector exception",
        "processing_status": "DETECTION_ERROR",
    }
    diag_pred_err = {
        "missing_left_pupil": True,
        "missing_right_pupil": True,
        "prediction_error": "Some predictor exception",
        "processing_status": "PREDICTION_ERROR",
    }
    assert gaze_occlusion_is_confirmed(diag_det_err) is False
    assert gaze_occlusion_is_confirmed(diag_pred_err) is False


def test_throttled_logging_avoids_per_frame_spam(capsys):
    """Verify that consecutive frame exceptions are throttled and do not print every frame."""
    failing_detector = MagicMock(side_effect=RuntimeError("Recurring detector fault"))
    pipe = VisionPipeline(detector=failing_detector, predictor=MagicMock())
    frame = _make_dummy_frame()

    # Process 50 consecutive failing frames
    for _ in range(50):
        pipe.process_face(frame)

    captured = capsys.readouterr().out
    # Must only print once initially due to throttling window (min_interval_sec=3.0)
    occurrences = captured.count("Warning: face_detector exception during frame processing")
    assert occurrences == 1, f"Expected 1 throttled log message, got {occurrences}"
