from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from mobile_detection import (
    PhoneDetectionTracker,
    process_mobile_detection,
    DETECTION_STATE_FRESH_POSITIVE,
    DETECTION_STATE_CACHED_POSITIVE,
    DETECTION_STATE_FRESH_NEGATIVE,
)
from main import run_surveillance

pytestmark = pytest.mark.unit


def create_mock_yolo_result(detected=True, confidence=0.88):
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


def test_tracker_process_frame_clean_frame_unmutated():
    """Verify that PhoneDetectionTracker draws on display_frame while clean frame remains untouched."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.90)

    tracker = PhoneDetectionTracker(interval_frames=1, model_instance=mock_model)

    # Create distinct clean and display frames
    clean_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    clean_snapshot = clean_frame.copy()
    display_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    annotated, detected, res = tracker.process_frame(
        clean_frame, annotate=True, display_frame=display_frame
    )

    # 1. Clean frame must be completely identical to its original state (zero pixels modified)
    assert np.array_equal(clean_frame, clean_snapshot), "Clean frame was mutated during phone detection!"

    # 2. display_frame must receive the annotations
    assert not np.array_equal(display_frame, clean_snapshot), "display_frame was not annotated!"
    assert annotated is display_frame

    # 3. Model must have been called with clean_frame
    called_frame = mock_model.call_args[0][0]
    assert np.array_equal(called_frame, clean_snapshot), "YOLO model received mutated pixels!"


def test_tracker_process_frame_fallback_without_display_frame():
    """Verify backwards-compatibility: if display_frame is None, annotations are drawn on frame."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.85)

    tracker = PhoneDetectionTracker(interval_frames=1, model_instance=mock_model)
    single_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    original_snapshot = single_frame.copy()

    annotated, detected, res = tracker.process_frame(single_frame, annotate=True)

    # Without display_frame, single_frame receives annotations
    assert not np.array_equal(single_frame, original_snapshot)
    assert annotated is single_frame


def test_process_mobile_detection_clean_frame_isolation():
    """Verify process_mobile_detection isolates clean inference frame from display_frame."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.95)

    clean_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clean_snapshot = clean_frame.copy()
    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated, detected = process_mobile_detection(
        clean_frame, model_instance=mock_model, display_frame=display_frame
    )

    assert detected is True
    assert np.array_equal(clean_frame, clean_snapshot), "Clean frame was mutated in process_mobile_detection!"
    assert not np.array_equal(display_frame, clean_snapshot), "display_frame was not annotated!"
    assert annotated is display_frame


def test_clean_frame_isolation_with_cadence_caching():
    """Verify clean-frame isolation works across cadence skip/cached frames (interval=3)."""
    mock_model = MagicMock()
    mock_model.return_value = create_mock_yolo_result(detected=True, confidence=0.88)

    tracker = PhoneDetectionTracker(interval_frames=3, model_instance=mock_model)

    for frame_idx in range(1, 7):
        clean_frame = np.full((100, 100, 3), frame_idx * 10, dtype=np.uint8)
        clean_snapshot = clean_frame.copy()
        display_frame = clean_frame.copy()

        annotated, detected, res = tracker.process_frame(
            clean_frame, annotate=True, display_frame=display_frame
        )

        assert detected is True
        assert np.array_equal(clean_frame, clean_snapshot), f"Frame {frame_idx} clean frame was mutated!"
        assert not np.array_equal(display_frame, clean_snapshot), f"Frame {frame_idx} display_frame was not annotated!"

        if frame_idx in (1, 4):
            assert res.state == DETECTION_STATE_FRESH_POSITIVE
        else:
            assert res.state == DETECTION_STATE_CACHED_POSITIVE

    assert mock_model.call_count == 2  # Called only on frames 1 and 4


def test_main_surveillance_yolo_receives_unannotated_pixels(tmp_path):
    """Verify that in run_surveillance, YOLO receives clean pixels rather than face-annotated pixels."""
    # Synthetic frame: grey canvas
    raw_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, raw_frame.copy()), (False, None)]

    captured_yolo_inputs = []

    def mock_process_frame(frame, annotate=True, display_frame=None):
        captured_yolo_inputs.append(frame.copy())
        from mobile_detection import PhoneDetectionResult, DETECTION_STATE_FRESH_NEGATIVE
        return display_frame if display_frame is not None else frame, False, PhoneDetectionResult(
            detected=False, state=DETECTION_STATE_FRESH_NEGATIVE
        )

    # Create dummy model files so validate_config passes
    dummy_pred = tmp_path / "fake.dat"
    dummy_yolo = tmp_path / "fake.pt"
    dummy_pred.write_text("dummy")
    dummy_yolo.write_text("dummy")

    with patch("config.SHAPE_PREDICTOR_PATH", dummy_pred), \
         patch("config.YOLO_MODEL_PATH", dummy_yolo), \
         patch("main.process_frame_face") as mock_face, \
         patch("main.PhoneDetectionTracker") as MockTrackerClass:

        mock_tracker = MagicMock()
        mock_tracker.process_frame.side_effect = mock_process_frame
        MockTrackerClass.return_value = mock_tracker

        # Make process_frame_face draw heavy alterations on its input frame
        def fake_process_face(frame, **kwargs):
            from vision_pipeline import FaceAnalysisResult
            # Mutate frame heavily (draw bright white box)
            frame[50:150, 50:150] = 255
            res = FaceAnalysisResult(face_count=1, gaze_direction="Center", head_direction="Screen")
            return res, frame

        mock_face.side_effect = fake_process_face

        run_surveillance(cap=mock_cap, display=False, max_frames=1)

    assert len(captured_yolo_inputs) == 1
    passed_to_yolo = captured_yolo_inputs[0]

    # The array passed to YOLO must NOT have the white box [50:150, 50:150] = 255 drawn by process_frame_face!
    assert np.all(passed_to_yolo == 100), "YOLO received a frame that had been mutated by face processing!"
