from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from main import parse_args, validate_cli_args, run_surveillance
import config

pytestmark = pytest.mark.unit


def test_cli_defaults():
    """Verify default CLI arguments when none are provided."""
    args = parse_args([])
    assert args.camera is None
    assert args.video is None
    assert args.headless is False
    assert args.detection_interval is None

    # Default args should validate without error
    validate_cli_args(args)


def test_cli_camera_index():
    """Verify --camera argument parses correctly."""
    args = parse_args(["--camera", "2"])
    assert args.camera == 2
    assert args.video is None
    validate_cli_args(args)


def test_cli_negative_camera_index():
    """Verify negative camera index raises ValueError."""
    args = parse_args(["--camera", "-1"])
    with pytest.raises(ValueError, match="Camera index must be a non-negative integer"):
        validate_cli_args(args)


def test_cli_video_source(tmp_path):
    """Verify --video argument with existing file validates successfully."""
    dummy_video = tmp_path / "test_vid.mp4"
    dummy_video.write_bytes(b"dummy video data")

    args = parse_args(["--video", str(dummy_video)])
    assert args.video == str(dummy_video)
    assert args.camera is None
    validate_cli_args(args)


def test_cli_invalid_video_path():
    """Verify non-existent video path raises FileNotFoundError."""
    args = parse_args(["--video", "nonexistent_video_path_xyz.mp4"])
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        validate_cli_args(args)


def test_cli_conflicting_camera_and_video(tmp_path):
    """Verify specifying both --camera and --video is rejected as mutually exclusive."""
    dummy_video = tmp_path / "test_vid.mp4"
    dummy_video.write_bytes(b"dummy")

    with pytest.raises(SystemExit):
        parse_args(["--camera", "0", "--video", str(dummy_video)])


def test_cli_headless_flag():
    """Verify --headless flag sets headless mode to True."""
    args = parse_args(["--headless"])
    assert args.headless is True

    args_custom = parse_args(["--camera", "1", "--headless", "--detection-interval", "5"])
    assert args_custom.camera == 1
    assert args_custom.headless is True
    assert args_custom.detection_interval == 5
    validate_cli_args(args_custom)


def test_cli_invalid_detection_interval():
    """Verify detection interval < 1 raises ValueError."""
    args = parse_args(["--detection-interval", "0"])
    with pytest.raises(ValueError, match="Detection interval must be a positive integer"):
        validate_cli_args(args)


def test_headless_mode_does_not_open_gui_windows(tmp_path):
    """Verify that display=False never invokes cv2.imshow or cv2.destroyAllWindows."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Frame reading returns empty frame for 2 iterations, then EOF
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, dummy_frame), (True, dummy_frame), (False, None)]

    with patch("cv2.imshow") as mock_imshow, \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("config.SHAPE_PREDICTOR_PATH", tmp_path / "fake.dat"), \
         patch("config.YOLO_MODEL_PATH", tmp_path / "fake.pt"), \
         patch("config.enforce_offline_isolation"):
        (tmp_path / "fake.dat").write_text("dummy")
        (tmp_path / "fake.pt").write_text("dummy")

        run_surveillance(cap=mock_cap, display=False, max_frames=2)

    assert mock_imshow.called is False
    assert mock_destroy.called is False


def test_video_eof_clean_exit(tmp_path):
    """Verify that video end-of-file terminates surveillance loop cleanly without errors."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Immediate EOF
    mock_cap.read.return_value = (False, None)

    with patch("config.SHAPE_PREDICTOR_PATH", tmp_path / "fake.dat"), \
         patch("config.YOLO_MODEL_PATH", tmp_path / "fake.pt"), \
         patch("config.enforce_offline_isolation"):
        (tmp_path / "fake.dat").write_text("dummy")
        (tmp_path / "fake.pt").write_text("dummy")

        # Must exit cleanly without exception
        run_surveillance(cap=mock_cap, display=False)

    assert mock_cap.release.called is True
