import json
from unittest.mock import MagicMock, patch

import pytest

import config
import main
from gaze_fusion import CombinedGazeResult
from screen_gaze import (
    BOUNDARY_ON_SCREEN,
    CALIB_STATE_CANCELLED,
    CALIB_STATE_FAILED,
    CALIB_STATE_VALID,
    FIVE_POINT_TARGETS,
    ScreenCalibrationManager,
    load_screen_calibration,
    screen_calibration_environment,
)

pytestmark = pytest.mark.unit


def _fit(manager, scale=18.0, flat=False):
    manager.start()
    for tx, ty, _ in manager.targets:
        yaw = 0.0 if flat else (tx - 0.5) * scale * 2.0
        pitch = 0.0 if flat else (0.5 - ty) * scale * 2.0
        for offset in (-0.1, 0.0, 0.1):
            manager.process_sample(
                CombinedGazeResult(
                    yaw=yaw + offset, pitch=pitch + offset,
                    confidence=0.9, status="VALID",
                )
            )
    return manager


def _manager(path=None, calibrator=None, environment=None):
    return ScreenCalibrationManager(
        calibrator=calibrator,
        display_size=(1920, 1080),
        calibration_path=path,
        environment=environment or screen_calibration_environment("webcam", 0),
    )


def test_successful_calibration_saves_and_loads(tmp_path):
    path = tmp_path / "screen.json"
    manager = _fit(_manager(path))
    assert manager.state == CALIB_STATE_VALID
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["display"] == {"width": 1920, "height": 1080, "aspect_ratio": 1.7778}
    assert payload["target_configuration"] == [list(target) for target in FIVE_POINT_TARGETS]
    assert len(payload["model"]["coefficients"]) == 3
    assert payload["quality_metrics"]["success"] is True
    assert payload["calibrated_at"]

    loaded, reason = load_screen_calibration(
        path, (1920, 1080), screen_calibration_environment("webcam", 0)
    )
    assert reason == ""
    assert loaded is not None and loaded.is_calibrated
    result = loaded.estimate_screen_gaze(0.0, 0.0, 0.9)
    assert result.status == "VALID"
    assert result.boundary_state == BOUNDARY_ON_SCREEN


def test_missing_file_is_graceful(tmp_path):
    loaded, reason = load_screen_calibration(
        tmp_path / "missing.json", (1920, 1080), screen_calibration_environment("webcam", 0)
    )
    assert loaded is None
    assert reason == "calibration_file_missing"


def test_main_activates_compatible_saved_profile_at_startup(monkeypatch, tmp_path):
    calibrator = _fit(_manager()).calibrator
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (False, None)
    created = []
    original_manager = ScreenCalibrationManager

    def capture_manager(*args, **kwargs):
        manager = original_manager(*args, **kwargs)
        created.append(manager)
        return manager

    monkeypatch.setattr(config, "validate_config", lambda: None)
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_PATH", tmp_path / "screen.json")
    monkeypatch.setattr(main, "get_primary_display_resolution", lambda strict=False: (1920, 1080))
    monkeypatch.setattr(main, "load_screen_calibration", lambda *args, **kwargs: (calibrator, ""))
    monkeypatch.setattr(main, "ScreenCalibrationManager", capture_manager)

    main.run_surveillance(
        cap=cap,
        logger=MagicMock(session_id="test"),
        gaze_logger=False,
        display=False,
        max_frames=0,
    )

    assert len(created) == 1
    assert created[0].calibrator is calibrator
    assert created[0].state == CALIB_STATE_VALID


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("corrupt", "invalid_calibration_file"),
        ("missing_quality", "invalid_calibration_file"),
        ("bad_coefficients", "invalid_calibration_file"),
        ("nan", "invalid_calibration_file"),
        ("infinity", "invalid_calibration_file"),
        ("schema", "unsupported_schema_version"),
        ("resolution", "display_geometry_incompatible"),
        ("aspect", "display_geometry_incompatible"),
    ],
)
def test_invalid_or_incompatible_saved_calibration_is_rejected(tmp_path, mutation, reason):
    path = tmp_path / "screen.json"
    _fit(_manager(path))
    if mutation == "corrupt":
        path.write_text("{ broken", encoding="utf-8")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "missing_quality":
            del payload["quality_metrics"]
        elif mutation == "bad_coefficients":
            payload["model"]["coefficients"] = [[0.0, 0.0]]
        elif mutation == "nan":
            payload["model"]["coefficients"][0][0] = float("nan")
        elif mutation == "infinity":
            payload["quality_metrics"]["rmse"] = float("inf")
        elif mutation == "schema":
            payload["schema_version"] = 999
        elif mutation == "resolution":
            payload["display"]["width"] = 1600
        elif mutation == "aspect":
            payload["display"]["aspect_ratio"] = 1.6
        path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, failure = load_screen_calibration(
        path, (1920, 1080), screen_calibration_environment("webcam", 0)
    )
    assert loaded is None
    assert failure.startswith(reason)


def test_camera_gaze_configuration_mismatch_rejects_profile(tmp_path):
    path = tmp_path / "screen.json"
    _fit(_manager(path))
    loaded, reason = load_screen_calibration(
        path, (1920, 1080), screen_calibration_environment("webcam", 1)
    )
    assert loaded is None
    assert reason == "camera_or_gaze_configuration_incompatible"


def test_cancelled_recalibration_preserves_active_model_and_saved_file(tmp_path):
    path = tmp_path / "screen.json"
    original = _fit(_manager(path)).calibrator
    before = path.read_bytes()
    old_matrix = original._calibration_matrix.copy()

    manager = _manager(path, calibrator=original)
    manager.start()
    manager.cancel()

    assert manager.state == CALIB_STATE_CANCELLED
    assert manager.is_calibrated
    assert (manager.calibrator._calibration_matrix == old_matrix).all()
    assert path.read_bytes() == before


def test_failed_recalibration_preserves_active_model_and_saved_file(tmp_path):
    path = tmp_path / "screen.json"
    original = _fit(_manager(path)).calibrator
    before = path.read_bytes()
    old_matrix = original._calibration_matrix.copy()

    manager = _fit(_manager(path, calibrator=original), flat=True)
    assert manager.state == CALIB_STATE_FAILED
    assert manager.is_calibrated
    assert (manager.calibrator._calibration_matrix == old_matrix).all()
    assert path.read_bytes() == before


def test_successful_recalibration_replaces_old_profile(tmp_path):
    path = tmp_path / "screen.json"
    original = _fit(_manager(path)).calibrator
    old_matrix = original._calibration_matrix.copy()
    manager = _fit(_manager(path, calibrator=original), scale=23.0)

    assert manager.state == CALIB_STATE_VALID
    assert not (manager.calibrator._calibration_matrix == old_matrix).all()
    loaded, reason = load_screen_calibration(
        path, (1920, 1080), screen_calibration_environment("webcam", 0)
    )
    assert reason == "" and loaded is not None
    assert (loaded._calibration_matrix == manager.calibrator._calibration_matrix).all()


def test_failed_atomic_replacement_preserves_existing_file_and_calibration(tmp_path):
    path = tmp_path / "screen.json"
    original = _fit(_manager(path)).calibrator
    previous_payload = path.read_bytes()
    previous_matrix = original._calibration_matrix.copy()

    with patch("screen_gaze.os.replace", side_effect=OSError("simulated replace failure")):
        manager = _fit(_manager(path, calibrator=original), scale=23.0)

    assert manager.state == CALIB_STATE_FAILED
    assert manager.is_calibrated
    assert (manager.calibrator._calibration_matrix == previous_matrix).all()
    assert path.read_bytes() == previous_payload


def test_insufficient_samples_and_face_loss_do_not_complete_calibration(tmp_path):
    manager = _manager(tmp_path / "screen.json")
    manager.start()
    assert not manager.process_sample(None)
    assert not manager.process_sample(
        CombinedGazeResult(yaw=0.0, pitch=0.0, confidence=0.9, status="UNCERTAIN")
    )
    assert manager.current_target_samples == 0
    assert manager.state != CALIB_STATE_VALID
    assert not (tmp_path / "screen.json").exists()


def test_loaded_profile_requires_known_display_geometry(tmp_path):
    path = tmp_path / "screen.json"
    _fit(_manager(path))
    loaded, reason = load_screen_calibration(
        path, None, screen_calibration_environment("webcam", 0)
    )
    assert loaded is None
    assert reason == "display_geometry_unavailable"


def test_configuration_rejects_non_finite_calibration_and_duration_values(monkeypatch):
    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MAX_RMSE", float("nan"))
    with pytest.raises(ValueError, match="SCREEN_CALIBRATION_MAX_RMSE.*finite"):
        config.validate_config()

    monkeypatch.setattr(config, "SCREEN_CALIBRATION_MAX_RMSE", 0.18)
    monkeypatch.setattr(config, "VIOLATION_THRESHOLD_SEC", float("inf"))
    with pytest.raises(ValueError, match="VIOLATION_THRESHOLD_SEC.*finite"):
        config.validate_config()
