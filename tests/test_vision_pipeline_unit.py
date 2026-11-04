from unittest.mock import MagicMock, call
import numpy as np
import pytest

from vision_pipeline import VisionPipeline, FaceAnalysisResult, process_frame_face
from eye_movement import GAZE_NO_FACE, GAZE_CENTER
from head_pose import HEAD_NO_FACE, HEAD_SCREEN

pytestmark = pytest.mark.unit

class FakeDlibRect:
    def __init__(self, left, top, right, bottom):
        self._l = left
        self._t = top
        self._r = right
        self._b = bottom

    def left(self):
        return self._l

    def top(self):
        return self._t

    def right(self):
        return self._r

    def bottom(self):
        return self._b

class FakePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class FakeLandmarks:
    def __init__(self):
        # 68 landmark points
        self._points = [FakePoint(100 + i, 100 + i) for i in range(68)]
        # Set distinctive eye points
        for i in range(36, 42):
            self._points[i] = FakePoint(100 + (i - 36) * 5, 120)
        for i in range(42, 48):
            self._points[i] = FakePoint(150 + (i - 42) * 5, 120)
        # Head pose points
        self._points[30] = FakePoint(130, 140)  # Nose tip
        self._points[8] = FakePoint(130, 180)   # Chin
        self._points[48] = FakePoint(115, 160)  # Left mouth
        self._points[54] = FakePoint(145, 160)  # Right mouth

    def part(self, idx):
        return self._points[idx]

def test_pipeline_zero_faces_handling():
    """Verify that zero detected faces cleanly yields GAZE_NO_FACE and HEAD_NO_FACE."""
    mock_detector = MagicMock(return_value=[])
    mock_predictor = MagicMock()

    pipeline = VisionPipeline(detector=mock_detector, predictor=mock_predictor)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result, annotated_frame = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

    assert mock_detector.call_count == 1
    assert mock_predictor.call_count == 0
    assert result.face_count == 0
    assert result.is_multi_face is False
    assert result.gaze_direction == GAZE_NO_FACE
    assert result.head_direction == HEAD_NO_FACE
    assert result.landmarks is None

def test_pipeline_single_face_detection_call_and_landmark_reuse():
    """Verify that detector is called exactly ONCE and predictor is called ONCE for a face."""
    rect = FakeDlibRect(100, 100, 200, 200)
    mock_detector = MagicMock(return_value=[rect])
    landmarks_obj = FakeLandmarks()
    mock_predictor = MagicMock(return_value=landmarks_obj)

    pipeline = VisionPipeline(detector=mock_detector, predictor=mock_predictor)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result, annotated_frame = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

    # Face detector must be called exactly once per frame
    assert mock_detector.call_count == 1
    # Landmark predictor must be called exactly once for the face
    assert mock_predictor.call_count == 1
    assert mock_predictor.call_args[0][1] == rect

    # Result contains reused landmarks and computed states
    assert result.face_count == 1
    assert result.is_multi_face is False
    assert result.landmarks is landmarks_obj
    assert result.gaze_direction != GAZE_NO_FACE
    assert result.head_direction != HEAD_NO_FACE

def test_pipeline_multiple_faces_handling_and_primary_selection():
    """Verify that multiple faces are flagged as is_multi_face=True and largest face is analyzed."""
    small_rect = FakeDlibRect(10, 10, 30, 30)       # area = 20 * 20 = 400
    large_rect = FakeDlibRect(100, 100, 250, 250)   # area = 150 * 150 = 22500
    medium_rect = FakeDlibRect(300, 300, 350, 350)  # area = 50 * 50 = 2500

    mock_detector = MagicMock(return_value=[small_rect, large_rect, medium_rect])
    landmarks_obj = FakeLandmarks()
    mock_predictor = MagicMock(return_value=landmarks_obj)

    pipeline = VisionPipeline(detector=mock_detector, predictor=mock_predictor)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result, annotated_frame = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

    assert mock_detector.call_count == 1
    assert mock_predictor.call_count == 1
    # Ensure predictor was called with the largest face (large_rect)
    assert mock_predictor.call_args[0][1] == large_rect

    assert result.face_count == 3
    assert result.is_multi_face is True
    assert result.face_rect == large_rect

def test_pipeline_calibration_sample_collection():
    """Verify that during calibration (calibrated_angles=None), head_pose_sample is populated."""
    rect = FakeDlibRect(100, 100, 200, 200)
    mock_detector = MagicMock(return_value=[rect])
    landmarks_obj = FakeLandmarks()
    mock_predictor = MagicMock(return_value=landmarks_obj)

    pipeline = VisionPipeline(detector=mock_detector, predictor=mock_predictor)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result, _ = pipeline.process_face(frame, calibrated_angles=None)
    assert result.head_direction == "Calibrating"
    assert result.head_pose_sample is not None
    assert isinstance(result.head_pose_sample, tuple)
    assert len(result.head_pose_sample) == 3

def test_pipeline_empty_or_none_frame_safe():
    """Verify that empty or None frames do not raise an unhandled exception."""
    pipeline = VisionPipeline(detector=MagicMock(), predictor=MagicMock())
    result, returned_frame = pipeline.process_face(None)
    assert result.face_count == 0
    assert result.gaze_direction == GAZE_NO_FACE

    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    result, returned_frame = pipeline.process_face(empty_frame)
    assert result.face_count == 0
