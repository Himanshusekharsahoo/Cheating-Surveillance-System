import os
from pathlib import Path
import pytest

from mobile_detection import get_model_path, BASE_DIR as MOBILE_BASE_DIR
from head_pose import get_predictor_path, BASE_DIR as HEAD_BASE_DIR
from eye_movement import get_predictor_path as get_eye_predictor_path

pytestmark = pytest.mark.unit

def test_yolo_model_path_resolution(monkeypatch, tmp_path):
    """Verify that YOLO model path resolves to best_yolov12.pt within the project directory."""
    fake_model = tmp_path / "model" / "best_yolov12.pt"
    fake_model.parent.mkdir(parents=True, exist_ok=True)
    fake_model.write_text("placeholder yolo model")
    monkeypatch.setattr("mobile_detection.DEFAULT_MODEL_PATH", fake_model)

    resolved = get_model_path()
    assert resolved == fake_model
    assert resolved.exists()
    assert resolved.is_file()

    # Verify default constant filename and directory structure
    assert (MOBILE_BASE_DIR / "model" / "best_yolov12.pt").name == "best_yolov12.pt"

def test_yolo_model_path_from_different_cwd(monkeypatch, tmp_path):
    """Verify that model path resolution is robust to changes in the current working directory."""
    fake_model = tmp_path / "model" / "best_yolov12.pt"
    fake_model.parent.mkdir(parents=True, exist_ok=True)
    fake_model.write_text("placeholder yolo model")
    monkeypatch.setattr("mobile_detection.DEFAULT_MODEL_PATH", fake_model)

    sub_dir = tmp_path / "other_cwd"
    sub_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(sub_dir)

    resolved = get_model_path()
    assert resolved == fake_model
    assert resolved.exists()

def test_yolo_model_missing_raises_filenotfound(tmp_path):
    """Verify that a nonexistent model path raises FileNotFoundError with a clear message."""
    fake_path = tmp_path / "nonexistent_model.pt"
    with pytest.raises(FileNotFoundError) as exc_info:
        get_model_path(fake_path)
    assert "YOLO model file not found at" in str(exc_info.value)

def test_dlib_shape_predictor_path_resolution(monkeypatch, tmp_path):
    """Verify that dlib predictor resolves to shape_predictor_68_face_landmarks.dat."""
    fake_predictor = tmp_path / "model" / "shape_predictor_68_face_landmarks.dat"
    fake_predictor.parent.mkdir(parents=True, exist_ok=True)
    fake_predictor.write_text("placeholder landmark model")

    monkeypatch.setattr("head_pose.DEFAULT_PREDICTOR_PATH", fake_predictor)
    monkeypatch.setattr("eye_movement.DEFAULT_PREDICTOR_PATH", fake_predictor)

    resolved_head = get_predictor_path()
    resolved_eye = get_eye_predictor_path()
    assert resolved_head == fake_predictor
    assert resolved_eye == fake_predictor
    assert resolved_head.exists()
    assert resolved_head.is_file()

    # Verify default constant filename and directory structure
    assert (HEAD_BASE_DIR / "model" / "shape_predictor_68_face_landmarks.dat").name == "shape_predictor_68_face_landmarks.dat"

def test_dlib_shape_predictor_missing_raises_filenotfound(tmp_path):
    """Verify that missing shape predictor raises FileNotFoundError."""
    fake_path = tmp_path / "missing_predictor.dat"
    with pytest.raises(FileNotFoundError) as exc_info:
        get_predictor_path(fake_path)
    assert "dlib shape predictor not found at" in str(exc_info.value)
