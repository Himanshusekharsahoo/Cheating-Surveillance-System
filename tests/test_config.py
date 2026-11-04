import hashlib
import os
from pathlib import Path
import pytest

import config

pytestmark = pytest.mark.unit

def test_valid_config_validates_successfully(monkeypatch, tmp_path):
    """Verify that valid configuration validates successfully when model files exist and match integrity."""
    fake_predictor = tmp_path / "shape_predictor_68_face_landmarks.dat"
    fake_yolo = tmp_path / "best_yolov12.pt"
    pred_bytes = b"placeholder landmark model"
    yolo_bytes = b"placeholder yolo model"
    fake_predictor.write_bytes(pred_bytes)
    fake_yolo.write_bytes(yolo_bytes)

    config.register_test_model(
        fake_predictor.name,
        hashlib.sha256(pred_bytes).hexdigest(),
        len(pred_bytes),
        model_type="dlib_shape_predictor",
    )
    config.register_test_model(
        fake_yolo.name,
        hashlib.sha256(yolo_bytes).hexdigest(),
        len(yolo_bytes),
        model_type="yolo",
    )

    try:
        monkeypatch.setattr(config, "SHAPE_PREDICTOR_PATH", fake_predictor)
        monkeypatch.setattr(config, "YOLO_MODEL_PATH", fake_yolo)
        monkeypatch.setattr(config, "LOG_DIR", tmp_path / "log")

        # Should not raise any exception
        config.validate_config()
        assert config.SHAPE_PREDICTOR_PATH.is_file()
        assert config.YOLO_MODEL_PATH.is_file()
        assert config.PHONE_CONFIDENCE_THRESHOLD == 0.70
        assert config.VIOLATION_THRESHOLD_SEC == 3.0
        assert config.CALIBRATION_DURATION_SEC == 5.0
        assert config.MIN_CALIBRATION_SAMPLES == 10
    finally:
        config.reset_model_registry()

def test_config_paths_relative_to_project(monkeypatch, tmp_path):
    """Verify that model paths resolve relative to project directory even if cwd changes."""
    monkeypatch.chdir(tmp_path)
    assert config.BASE_DIR.is_dir()
    assert config.SHAPE_PREDICTOR_PATH == config.BASE_DIR / "model" / "shape_predictor_68_face_landmarks.dat"
    assert config.YOLO_MODEL_PATH == config.BASE_DIR / "model" / "best_yolov12.pt"

def test_missing_shape_predictor_raises_filenotfound(monkeypatch, tmp_path):
    """Verify that missing shape predictor raises FileNotFoundError with actionable message."""
    fake_yolo = tmp_path / "placeholder_yolo.pt"
    fake_yolo.write_text("placeholder")
    monkeypatch.setattr(config, "YOLO_MODEL_PATH", fake_yolo)

    fake_path = tmp_path / "nonexistent_shape_predictor.dat"
    monkeypatch.setattr(config, "SHAPE_PREDICTOR_PATH", fake_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_config()
    assert "Missing required facial landmark model at" in str(exc_info.value)

def test_missing_yolo_model_raises_filenotfound(monkeypatch, tmp_path):
    """Verify that missing YOLO checkpoint raises FileNotFoundError with actionable message."""
    fake_predictor = tmp_path / "placeholder_predictor.dat"
    fake_predictor.write_text("placeholder")
    monkeypatch.setattr(config, "SHAPE_PREDICTOR_PATH", fake_predictor)

    fake_path = tmp_path / "nonexistent_yolo.pt"
    monkeypatch.setattr(config, "YOLO_MODEL_PATH", fake_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_config()
    assert "Missing required YOLO model at" in str(exc_info.value)

def test_invalid_phone_confidence_threshold_raises_valueerror(monkeypatch):
    """Verify that out-of-bounds phone confidence threshold raises ValueError."""
    monkeypatch.setattr(config, "PHONE_CONFIDENCE_THRESHOLD", 0.0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_config()
    assert "Invalid PHONE_CONFIDENCE_THRESHOLD" in str(exc_info.value)

    monkeypatch.setattr(config, "PHONE_CONFIDENCE_THRESHOLD", 1.5)
    with pytest.raises(ValueError) as exc_info:
        config.validate_config()
    assert "Invalid PHONE_CONFIDENCE_THRESHOLD" in str(exc_info.value)

def test_invalid_violation_threshold_raises_valueerror(monkeypatch):
    """Verify that negative or zero violation threshold raises ValueError."""
    monkeypatch.setattr(config, "VIOLATION_THRESHOLD_SEC", -1.0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_config()
    assert "Invalid VIOLATION_THRESHOLD_SEC" in str(exc_info.value)

def test_invalid_calibration_duration_raises_valueerror(monkeypatch):
    """Verify that negative or zero calibration duration raises ValueError."""
    monkeypatch.setattr(config, "CALIBRATION_DURATION_SEC", 0.0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_config()
    assert "Invalid CALIBRATION_DURATION_SEC" in str(exc_info.value)

def test_invalid_min_calibration_samples_raises_valueerror(monkeypatch):
    """Verify that min calibration samples < 1 raises ValueError."""
    monkeypatch.setattr(config, "MIN_CALIBRATION_SAMPLES", 0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_config()
    assert "Invalid MIN_CALIBRATION_SAMPLES" in str(exc_info.value)


def test_invalid_head_pose_thresholds_raise_valueerror(monkeypatch):
    """Verify that non-positive head thresholds or deviation < neutral raises ValueError."""
    # Non-positive neutral
    monkeypatch.setattr(config, "HEAD_NEUTRAL_YAW_THRESHOLD", 0.0)
    with pytest.raises(ValueError, match="Head neutral thresholds must be strictly positive"):
        config.validate_config()

    monkeypatch.setattr(config, "HEAD_NEUTRAL_YAW_THRESHOLD", 12.0)
    # Deviation < neutral
    monkeypatch.setattr(config, "HEAD_DEVIATION_YAW_THRESHOLD", 10.0)  # less than neutral 12.0
    with pytest.raises(ValueError, match="HEAD_DEVIATION_YAW_THRESHOLD"):
        config.validate_config()


def test_invalid_pupil_parameters_raise_valueerror(monkeypatch):
    """Verify that invalid pupil thresholding method or fixed threshold bounds raise ValueError."""
    monkeypatch.setattr(config, "PUPIL_DETECTION_METHOD", "invalid_method")
    with pytest.raises(ValueError, match="Invalid PUPIL_DETECTION_METHOD"):
        config.validate_config()

    monkeypatch.setattr(config, "PUPIL_DETECTION_METHOD", "adaptive")
    monkeypatch.setattr(config, "PUPIL_FIXED_THRESHOLD", 300)
    with pytest.raises(ValueError, match="Invalid PUPIL_FIXED_THRESHOLD"):
        config.validate_config()


def test_invalid_gaze_calibration_parameters_raise_valueerror(monkeypatch):
    """Verify that invalid min gaze calibration samples or deflection bounds raise ValueError."""
    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 0)
    with pytest.raises(ValueError, match="Invalid MIN_GAZE_CALIBRATION_SAMPLES"):
        config.validate_config()

    monkeypatch.setattr(config, "MIN_GAZE_CALIBRATION_SAMPLES", 10)
    monkeypatch.setattr(config, "GAZE_HORIZONTAL_THRESHOLD", 0.6)
    with pytest.raises(ValueError, match="Invalid GAZE_HORIZONTAL_THRESHOLD"):
        config.validate_config()

    monkeypatch.setattr(config, "GAZE_HORIZONTAL_THRESHOLD", 0.16)
    monkeypatch.setattr(config, "GAZE_UPWARD_THRESHOLD", 0.6)
    with pytest.raises(ValueError, match="Invalid GAZE_UPWARD_THRESHOLD"):
        config.validate_config()


def test_invalid_fused_gaze_confidence_floor_raises_valueerror(monkeypatch):
    monkeypatch.setattr(config, "FUSED_GAZE_MIN_CONFIDENCE", 1.1)
    with pytest.raises(ValueError, match="Invalid FUSED_GAZE_MIN_CONFIDENCE"):
        config.validate_config()


def test_invalid_cadence_parameters_raise_valueerror(monkeypatch):
    """Verify that invalid YOLO interval or cache expiry parameters raise ValueError."""
    monkeypatch.setattr(config, "PHONE_DETECTION_INTERVAL_FRAMES", 0)
    with pytest.raises(ValueError, match="Invalid PHONE_DETECTION_INTERVAL_FRAMES"):
        config.validate_config()

    monkeypatch.setattr(config, "PHONE_DETECTION_INTERVAL_FRAMES", 3)
    # Expiry less than interval
    monkeypatch.setattr(config, "PHONE_DETECTION_CACHE_EXPIRY_FRAMES", 2)
    with pytest.raises(ValueError, match="Invalid PHONE_DETECTION_CACHE_EXPIRY_FRAMES"):
        config.validate_config()


def test_invalid_video_fps_raises_valueerror(monkeypatch):
    """Verify that non-positive DEFAULT_VIDEO_FPS raises ValueError."""
    monkeypatch.setattr(config, "DEFAULT_VIDEO_FPS", 0.0)
    with pytest.raises(ValueError, match="Invalid DEFAULT_VIDEO_FPS"):
        config.validate_config()


def test_invalid_screen_calibration_parameters_raise_valueerror(monkeypatch):
    """Verify that invalid screen calibration parameters raise ValueError."""
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_TARGETS", 2)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_MIN_TARGETS"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_TARGETS", 4)
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET", 0)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET", 3)
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MAX_RMSE", -0.1)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_MAX_RMSE"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MAX_RMSE", 0.18)
    monkeypatch.setattr(config, "SCREEN_BOUNDARY_MARGIN", 0.6)
    with pytest.raises(ValueError, match="SCREEN_BOUNDARY_MARGIN"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_BOUNDARY_MARGIN", 0.05)
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_DWELL_SEC", 0.0)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_MIN_DWELL_SEC"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_DWELL_SEC", 1.0)
    # Stability window > dwell duration
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_STABILITY_WINDOW_SEC", 1.5)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_STABILITY_WINDOW_SEC"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_STABILITY_WINDOW_SEC", 0.4)
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG", -1.0)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG"):
        config.validate_config()


@pytest.mark.parametrize("name", ["FRAME_WIDTH", "FRAME_HEIGHT", "DEFAULT_DISPLAY_WIDTH", "DEFAULT_DISPLAY_HEIGHT"])
def test_screen_calibration_geometry_requires_positive_integer_dimensions(monkeypatch, name):
    monkeypatch.setattr(config, name, 0)
    with pytest.raises(ValueError, match=name):
        config.validate_config()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_screen_calibration_confidence_must_be_finite_and_bounded(monkeypatch, value):
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MIN_CONFIDENCE", value)
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_MIN_CONFIDENCE"):
        config.validate_config()
