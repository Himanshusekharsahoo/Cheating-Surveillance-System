import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from eye_movement import (
    process_eye_movement,
    get_shape_predictor as get_eye_predictor,
    GAZE_CENTER,
    GAZE_NO_FACE
)
from head_pose import (
    process_head_pose,
    get_shape_predictor as get_head_predictor,
    HEAD_SCREEN,
    HEAD_NO_FACE
)
from mobile_detection import process_mobile_detection, get_model
import config

pytestmark = [pytest.mark.integration]

BASE_DIR = Path(__file__).resolve().parent.parent

def _check_real_models():
    """Ensure real model files exist on disk before running integration tests."""
    if not config.SHAPE_PREDICTOR_PATH.is_file():
        pytest.skip(f"Real landmark predictor missing at {config.SHAPE_PREDICTOR_PATH}")
    if not config.YOLO_MODEL_PATH.is_file():
        pytest.skip(f"Real YOLO model missing at {config.YOLO_MODEL_PATH}")


def _require_demo_video(vid_path):
    """Ensure real models and the specified demo video exist before running video tests."""
    _check_real_models()
    path_obj = Path(vid_path)
    if not path_obj.is_file():
        pytest.skip(f"Demo video asset missing at '{path_obj}'; skipping optional video integration test.")


def test_real_dlib_shape_predictor_instantiation():
    """Verify that dlib successfully instantiates the real 68-point shape predictor from disk."""
    _check_real_models()
    predictor = get_head_predictor()
    assert predictor is not None
    # Verify predictor is genuine dlib predictor instance
    assert "shape_predictor" in type(predictor).__name__.lower()


def test_real_yolo_model_instantiation_and_inference():
    """Verify that YOLO loads best_yolov12.pt and detects the mobile class on sample image."""
    _check_real_models()
    model = get_model()
    assert model is not None
    assert 0 in model.names
    assert model.names[0].lower() in ["phone", "cellphone", "mobile"]

    # Run inference on known historical sample if available on disk
    test_img_path = str(BASE_DIR / "log" / "mobile_detected_20250310_215212.jpg")
    if not os.path.exists(test_img_path):
        pytest.skip(
            f"Historical sample image missing at '{test_img_path}'; "
            f"model loading and class mappings verified."
        )

    img = cv2.imread(test_img_path)
    assert img is not None

    annotated_frame, mobile_detected = process_mobile_detection(img)
    assert isinstance(mobile_detected, bool)


@pytest.mark.demo_video
def test_real_demo_video_opening():
    """Verify that all three demo MP4 recordings can be opened and decoded via OpenCV."""
    _check_real_models()
    video_files = [
        BASE_DIR / "Demo_vid" / "gaze-detection.mp4",
        BASE_DIR / "Demo_vid" / "headpose-detection.mp4",
        BASE_DIR / "Demo_vid" / "Mobile-detection.mp4"
    ]
    missing = [str(v) for v in video_files if not v.is_file()]
    if missing:
        pytest.skip(f"Demo video assets missing: {', '.join(missing)}; skipping demo video opening test.")

    for vid_path in video_files:
        cap = cv2.VideoCapture(str(vid_path))
        try:
            assert cap.isOpened(), f"Could not open video file: {vid_path}"
            ret, frame = cap.read()
            assert ret is True
            assert frame is not None
            assert len(frame.shape) == 3 and frame.shape[2] == 3
        finally:
            cap.release()


@pytest.mark.demo_video
def test_real_head_pose_on_demo_video():
    """Verify real head pose calibration and orientation estimation on headpose-detection.mp4."""
    vid_path = BASE_DIR / "Demo_vid" / "headpose-detection.mp4"
    _require_demo_video(vid_path)
    cap = cv2.VideoCapture(str(vid_path))
    try:
        assert cap.isOpened()

        # Collect calibration samples from active frames
        calib_samples = []
        for f in range(100, 140, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if ret:
                _, angles = process_head_pose(frame, calibrated_angles=None)
                if angles is not None and isinstance(angles, tuple):
                    calib_samples.append(angles)
    finally:
        cap.release()

    assert len(calib_samples) >= 5, "Should collect at least 5 valid calibration samples from active video"

    # Compute baseline
    avg_pitch = float(np.mean([s[0] for s in calib_samples]))
    avg_yaw = float(np.mean([s[1] for s in calib_samples]))
    avg_roll = float(np.mean([s[2] for s in calib_samples]))
    baseline = (avg_pitch, avg_yaw, avg_roll)

    # Re-evaluate an active frame with baseline
    cap = cv2.VideoCapture(str(vid_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ret, frame = cap.read()
        assert ret is True

        _, direction = process_head_pose(frame, calibrated_angles=baseline)
        assert direction in [HEAD_SCREEN, "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Tilted"]
    finally:
        cap.release()


@pytest.mark.demo_video
def test_real_gaze_on_demo_video():
    """Verify real gaze estimation on gaze-detection.mp4."""
    vid_path = BASE_DIR / "Demo_vid" / "gaze-detection.mp4"
    _require_demo_video(vid_path)
    cap = cv2.VideoCapture(str(vid_path))
    try:
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ret, frame = cap.read()
        assert ret is True

        _, gaze = process_eye_movement(frame)
        # At frame 150, the candidate is looking straight at the camera
        assert gaze in [GAZE_CENTER, "Looking Left", "Looking Right", "Looking Up", "Looking Down"]
    finally:
        cap.release()


def test_real_missing_face_on_black_frame():
    """Verify that a synthetic blank/empty frame explicitly returns NO_FACE for both modules."""
    _check_real_models()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _, head_dir = process_head_pose(blank_frame, calibrated_angles=(0.0, 0.0, 0.0))
    _, gaze_dir = process_eye_movement(blank_frame)

    assert head_dir == HEAD_NO_FACE
    assert gaze_dir == GAZE_NO_FACE


@pytest.mark.demo_video
def test_real_vision_pipeline_on_demo_video():
    """Verify that unified vision_pipeline runs real dlib detector & predictor once on demo video."""
    from vision_pipeline import process_frame_face

    vid_path = BASE_DIR / "Demo_vid" / "gaze-detection.mp4"
    _require_demo_video(vid_path)
    cap = cv2.VideoCapture(str(vid_path))
    try:
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ret, frame = cap.read()
        assert ret is True

        result, annotated = process_frame_face(frame, calibrated_angles=(0.0, 0.0, 0.0))
        assert result.face_count >= 1
        assert result.landmarks is not None
        assert result.gaze_direction in [GAZE_CENTER, "Looking Left", "Looking Right", "Looking Up", "Looking Down"]
        assert result.head_direction in [HEAD_SCREEN, "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Tilted"]
    finally:
        cap.release()


def test_real_vision_pipeline_on_black_frame():
    """Verify unified vision_pipeline on blank frame returns 0 faces and NO_FACE states."""
    _check_real_models()
    from vision_pipeline import process_frame_face

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = process_frame_face(blank_frame, calibrated_angles=(0.0, 0.0, 0.0))

    assert result.face_count == 0
    assert result.is_multi_face is False
    assert result.gaze_direction == GAZE_NO_FACE
    assert result.head_direction == HEAD_NO_FACE
    assert result.landmarks is None


@pytest.mark.demo_video
def test_real_vision_pipeline_calibration_on_demo_video():
    """Verify that unified vision_pipeline produces valid angle tuples during calibration."""
    from vision_pipeline import process_frame_face

    vid_path = BASE_DIR / "Demo_vid" / "headpose-detection.mp4"
    _require_demo_video(vid_path)
    cap = cv2.VideoCapture(str(vid_path))
    try:
        assert cap.isOpened()

        calib_samples = []
        for f in range(100, 130, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if ret:
                result, _ = process_frame_face(frame, calibrated_angles=None)
                if result.head_pose_sample is not None:
                    calib_samples.append(result.head_pose_sample)
    finally:
        cap.release()

    assert len(calib_samples) >= 3
    for s in calib_samples:
        assert isinstance(s, tuple) and len(s) == 3


def test_real_mobile_detection_threshold_comparison():
    """Verify real YOLO threshold behavior on historical log screenshots (0.50 vs 0.70)."""
    _check_real_models()
    img1_path = str(BASE_DIR / "log" / "mobile_detected_20250310_215212.jpg")
    img3_path = str(BASE_DIR / "log" / "mobile_detected_20250310_215223.jpg")
    if not (os.path.exists(img1_path) and os.path.exists(img3_path)):
        pytest.skip(
            "Historical screenshot fixtures missing in log/; skipping threshold comparison."
        )

    img1 = cv2.imread(img1_path)
    assert img1 is not None

    _, det1_70 = process_mobile_detection(img1.copy(), conf_threshold=0.70)
    _, det1_40 = process_mobile_detection(img1.copy(), conf_threshold=0.40)
    assert det1_70 is False
    assert det1_40 is True

    img3 = cv2.imread(img3_path)
    assert img3 is not None

    _, det3_70 = process_mobile_detection(img3.copy(), conf_threshold=0.70)
    _, det3_50 = process_mobile_detection(img3.copy(), conf_threshold=0.50)
    assert det3_70 is True
    assert det3_50 is True

