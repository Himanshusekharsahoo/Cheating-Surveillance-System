import csv
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from gaze_fusion import CombinedGazeResult, EyeGazeResult, HeadPoseResult

from eye_movement import (
    detect_pupil,
    analyze_gaze_from_landmarks,
    get_last_gaze_diagnostics,
    GAZE_CENTER,
    GAZE_NO_FACE,
    GAZE_UNCERTAIN,
)
from vision_pipeline import (
    VisionPipeline,
    FaceAnalysisResult,
    GazeDiagnosticsLogger,
    GAZE_DIAGNOSTICS_FIELDNAMES,
)
from main import run_surveillance

pytestmark = pytest.mark.unit


def _create_mock_landmarks(left_eye_pts=None, right_eye_pts=None, extra_pts=None):
    """Helper creating a mock dlib full_object_detection / landmarks object."""
    mock = MagicMock()
    # Default coordinates
    left_default = [(30, 20), (35, 17), (45, 17), (50, 20), (45, 23), (35, 23)]
    right_default = [(70, 20), (75, 17), (85, 17), (90, 20), (85, 23), (75, 23)]

    pts_map = {}
    pts_left = left_eye_pts if left_eye_pts is not None else left_default
    for idx, (x, y) in enumerate(pts_left, start=36):
        pts_map[idx] = (x, y)

    pts_right = right_eye_pts if right_eye_pts is not None else right_default
    for idx, (x, y) in enumerate(pts_right, start=42):
        pts_map[idx] = (x, y)

    # 3D head pose landmark indices: 30 (nose tip), 8 (chin), 36, 45, 48 (mouth left), 54 (mouth right)
    defaults_extra = {
        30: (60, 40),
        8: (60, 80),
        48: (45, 65),
        54: (75, 65),
    }
    if extra_pts:
        defaults_extra.update(extra_pts)
    for idx, (x, y) in defaults_extra.items():
        if idx not in pts_map:
            pts_map[idx] = (x, y)

    def get_part(n):
        p = MagicMock()
        coord = pts_map.get(n, (50, 50))
        p.x = coord[0]
        p.y = coord[1]
        return p

    mock.part.side_effect = get_part
    return mock


def _create_synthetic_eye_frame():
    """Create a 120x160 synthetic BGR image with two distinct dark circular pupils on bright sclera."""
    frame = np.full((120, 160, 3), 220, dtype=np.uint8)  # Bright background/sclera
    import cv2
    # Left eye: box [30, 17, 20, 7] -> center around (40, 20)
    cv2.circle(frame, (40, 20), 2, (15, 15, 15), -1)
    # Right eye: box [70, 17, 20, 7] -> center around (80, 20)
    cv2.circle(frame, (80, 20), 2, (15, 15, 15), -1)
    return frame


def test_valid_diagnostic_record_generation(tmp_path: Path):
    """
    Verify that a frame with valid face, landmarks, pupils, and calibrated baseline
    generates all required diagnostic measurements and writes a complete CSV row.
    """
    frame = _create_synthetic_eye_frame()
    landmarks = _create_mock_landmarks()

    calibrated_gaze = ((0.50, 0.45), (0.50, 0.45))
    annotated, gaze_dir = analyze_gaze_from_landmarks(frame, landmarks, calibrated_gaze=calibrated_gaze)
    diag = get_last_gaze_diagnostics()

    assert diag is not None
    # 1. Verification of measurement integrity
    assert diag["face_detected"] is True
    assert diag["landmarks_detected"] is True
    assert diag["missing_face"] is False
    assert diag["missing_landmarks"] is False
    assert diag["missing_pupils"] is False
    assert diag["left_pupil_detected"] is True
    assert diag["right_pupil_detected"] is True
    assert diag["left_pupil_valid"] is True
    assert diag["right_pupil_valid"] is True
    assert diag["left_rejection_reason"] == ""
    assert diag["right_rejection_reason"] == ""

    # Raw pupil coordinates and crop dimensions
    assert isinstance(diag["left_pupil_x"], int)
    assert isinstance(diag["left_pupil_y"], int)
    assert isinstance(diag["right_pupil_x"], int)
    assert isinstance(diag["right_pupil_y"], int)
    assert diag["left_eye_w"] > 0
    assert diag["left_eye_h"] > 0
    assert diag["right_eye_w"] > 0
    assert diag["right_eye_h"] > 0

    # Normalized pupil coordinates in [0, 1]
    assert 0.0 <= diag["norm_left_x"] <= 1.0
    assert 0.0 <= diag["norm_left_y"] <= 1.0
    assert 0.0 <= diag["norm_right_x"] <= 1.0
    assert 0.0 <= diag["norm_right_y"] <= 1.0

    # Calibration baseline and deviations
    assert diag["is_calibrated"] is True
    assert diag["baseline_left_x"] == 0.50
    assert diag["baseline_left_y"] == 0.45
    assert diag["baseline_right_x"] == 0.50
    assert diag["baseline_right_y"] == 0.45
    assert diag["diff_left_x"] == pytest.approx(0.50 - diag["norm_left_x"], abs=1e-4)
    assert diag["diff_left_y"] == pytest.approx(diag["norm_left_y"] - 0.45, abs=1e-4)
    assert diag["mean_diff_x"] is not None
    assert diag["mean_diff_y"] is not None

    # CSV Logging test
    csv_file = tmp_path / "valid_gaze.csv"
    with GazeDiagnosticsLogger(csv_path=csv_file, session_id="sess_valid_123") as logger:
        # Mock FaceAnalysisResult wrapping diag
        face_result = FaceAnalysisResult(
            face_count=1,
            landmarks=landmarks,
            gaze_direction=gaze_dir,
            raw_head_pose=(2.5, -1.2, 0.4),
            gaze_diagnostics=diag,
            head_pose=HeadPoseResult(yaw=-1.2, pitch=2.5, roll=0.4, confidence=0.91, status="VALID"),
            eye_gaze=EyeGazeResult(
                left_eye_direction="Looking Center", right_eye_direction="Looking Center",
                selected_eye_direction="Looking Center", confidence=0.88,
                tracking_state="PAIRED_TRACKING", status="VALID",
            ),
            combined_gaze=CombinedGazeResult(
                yaw=1.25, pitch=-2.5, confidence=0.80, status="VALID"
            ),
        )
        success = logger.log_frame(
            frame_index=1,
            face_result=face_result,
            is_calibrated=True,
            calibrated_gaze=calibrated_gaze,
            current_time=0.033,
        )
        assert success is True

    # Read back CSV
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["session_id"] == "sess_valid_123"
        assert row["frame_index"] == "1"
        assert row["face_detected"] == "True"
        assert row["missing_face"] == "False"
        assert row["left_pupil_valid"] == "True"
        assert row["baseline_left_x"] == "0.5000"
        assert row["head_pitch"] == "2.50"
        assert row["head_yaw"] == "-1.20"
        assert row["head_roll"] == "0.40"
        assert row["head_confidence"] == "0.9100"
        assert row["head_status"] == "VALID"
        assert row["eye_gaze_direction"] == "Looking Center"
        assert row["left_eye_direction"] == "Looking Center"
        assert row["right_eye_direction"] == "Looking Center"
        assert row["eye_status"] == "VALID"
        assert row["combined_gaze_yaw"] == "1.25"
        assert row["combined_gaze_pitch"] == "-2.50"
        assert row["combined_gaze_confidence"] == "0.8000"
        assert row["combined_gaze_status"] == "VALID"


def test_expanded_diagnostics_preserve_existing_legacy_csv(tmp_path: Path):
    csv_file = tmp_path / "gaze_diagnostics.csv"
    legacy_header = GAZE_DIAGNOSTICS_FIELDNAMES[:-10]
    csv_file.write_text(",".join(legacy_header) + "\nlegacy-row\n", encoding="utf-8")
    original_contents = csv_file.read_bytes()

    with GazeDiagnosticsLogger(csv_path=csv_file, session_id="sess_new_schema") as logger:
        assert logger.log_frame(1, FaceAnalysisResult()) is True
        assert logger.csv_path == tmp_path / "gaze_diagnostics_v2.csv"

    assert csv_file.read_bytes() == original_contents
    with csv_file.with_name("gaze_diagnostics_v2.csv").open(encoding="utf-8") as new_file:
        assert next(csv.reader(new_file)) == GAZE_DIAGNOSTICS_FIELDNAMES


def test_missing_pupil_coordinates(tmp_path: Path):
    """
    Verify that when eye crops contain no pupil (e.g. uniform blank or shadow),
    missing pupil flags are set, validity is False with rejection reason, and CSV logs empty coords.
    """
    # Uniform blank frame: no pupil contour will pass area/contrast thresholds
    blank_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    landmarks = _create_mock_landmarks()

    analyze_gaze_from_landmarks(blank_frame, landmarks)
    diag = get_last_gaze_diagnostics()

    assert diag is not None
    assert diag["face_detected"] is True
    assert diag["missing_pupils"] is True
    assert diag["missing_left_pupil"] is True
    assert diag["missing_right_pupil"] is True
    assert diag["left_pupil_detected"] is False
    assert diag["right_pupil_detected"] is False
    assert diag["left_pupil_valid"] is False
    assert diag["right_pupil_valid"] is False
    assert len(diag["left_rejection_reason"]) > 0
    assert diag["left_pupil_x"] is None
    assert diag["norm_left_x"] is None
    assert diag["gaze_direction"] == GAZE_UNCERTAIN

    # Verify CSV serialization
    csv_file = tmp_path / "missing_pupils.csv"
    with GazeDiagnosticsLogger(csv_path=csv_file, session_id="sess_missing_pupil") as logger:
        face_result = FaceAnalysisResult(
            face_count=1,
            landmarks=landmarks,
            gaze_direction=GAZE_UNCERTAIN,
            raw_head_pose=(0.0, 0.0, 0.0),
            gaze_diagnostics=diag,
        )
        logger.log_frame(frame_index=5, face_result=face_result, is_calibrated=False, current_time=0.166)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["missing_pupils"] == "True"
        assert row["missing_left_pupil"] == "True"
        assert row["left_pupil_valid"] == "False"
        assert row["left_pupil_x"] == ""
        assert row["norm_left_x"] == ""
        assert row["left_rejection_reason"] != ""


def test_missing_face_or_landmarks_diagnostic(tmp_path: Path):
    """
    Verify that when no face is detected in the frame, VisionPipeline safely flags
    missing_face, missing_landmarks, missing_pupils, and sets gaze state to GAZE_NO_FACE.
    """
    pipeline = VisionPipeline()
    # Mock detector returning no faces
    pipeline._detector = MagicMock(return_value=[])

    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = pipeline.process_face(black_frame)

    assert result.face_count == 0
    assert result.gaze_direction == GAZE_NO_FACE
    assert result.gaze_diagnostics is not None

    diag = result.gaze_diagnostics
    assert diag["face_detected"] is False
    assert diag["landmarks_detected"] is False
    assert diag["missing_face"] is True
    assert diag["missing_landmarks"] is True
    assert diag["missing_pupils"] is True
    assert diag["missing_left_pupil"] is True
    assert diag["missing_right_pupil"] is True
    assert diag["left_rejection_reason"] == "no_face_detected"
    assert diag["right_rejection_reason"] == "no_face_detected"
    assert diag["head_pitch"] is None
    assert diag["left_pupil_x"] is None

    # CSV Logging check
    csv_file = tmp_path / "missing_face.csv"
    with GazeDiagnosticsLogger(csv_path=csv_file) as logger:
        logger.log_frame(frame_index=1, face_result=result, is_calibrated=False)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["face_detected"] == "False"
        assert row["missing_face"] == "True"
        assert row["missing_landmarks"] == "True"
        assert row["missing_pupils"] == "True"
        assert row["gaze_direction"] == GAZE_NO_FACE
        assert row["head_pitch"] == ""


def test_csv_output_structure_and_header(tmp_path: Path):
    """
    Verify that GazeDiagnosticsLogger creates a CSV file with the exact required header,
    preserves sequential rows across writes, and handles reopening without header duplication.
    """
    csv_file = tmp_path / "structure_test.csv"
    logger1 = GazeDiagnosticsLogger(csv_path=csv_file, session_id="sess_struct_1")

    # Verify file is not created until first record
    assert not csv_file.exists()

    record1 = {
        "frame_index": 0,
        "face_detected": True,
        "missing_face": False,
        "missing_landmarks": False,
        "missing_pupils": False,
        "left_pupil_x": 42,
        "left_pupil_y": 18,
        "right_pupil_x": 82,
        "right_pupil_y": 19,
        "norm_left_x": 0.4852,
        "norm_left_y": 0.4011,
        "gaze_direction": GAZE_CENTER,
    }
    logger1.log_record(record1, timestamp="2026-10-27T12:00:00+00:00")
    logger1.close()

    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == GAZE_DIAGNOSTICS_FIELDNAMES
        rows = list(reader)
        assert len(rows) == 1

    # Reopen same CSV with second logger instance and append another record
    logger2 = GazeDiagnosticsLogger(csv_path=csv_file, session_id="sess_struct_1")
    record2 = {
        "frame_index": 1,
        "face_detected": True,
        "missing_face": False,
        "gaze_direction": "Looking Left",
    }
    logger2.log_record(record2, timestamp="2026-10-27T12:00:01+00:00")
    logger2.close()

    with open(csv_file, "r", encoding="utf-8") as f:
        dict_reader = list(csv.DictReader(f))
        assert len(dict_reader) == 2
        assert dict_reader[0]["frame_index"] == "0"
        assert dict_reader[0]["gaze_direction"] == GAZE_CENTER
        assert dict_reader[1]["frame_index"] == "1"
        assert dict_reader[1]["gaze_direction"] == "Looking Left"


def test_session_id_and_timestamp_handling(tmp_path: Path):
    """
    Verify ISO 8601 timestamp parsing and session ID constancy across frames.
    """
    # 1. Custom session ID
    custom_id = "test_custom_gaze_sess_999"
    csv_file = tmp_path / "session_test.csv"
    logger = GazeDiagnosticsLogger(csv_path=csv_file, session_id=custom_id)
    assert logger.session_id == custom_id

    logger.log_record({"frame_index": 1})
    logger.log_record({"frame_index": 2})
    logger.close()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        for r in reader:
            assert r["session_id"] == custom_id
            # Verify timestamp is valid ISO 8601
            dt = datetime.fromisoformat(r["timestamp"])
            assert dt is not None

    # 2. Auto-generated session ID format
    auto_logger = GazeDiagnosticsLogger(log_dir=tmp_path)
    assert auto_logger.session_id.startswith("sess_")
    auto_logger.close()


def test_resource_cleanup_safe_close(tmp_path: Path):
    """
    Verify that close() cleans up the file descriptor safely and idempotent calls don't raise.
    """
    csv_file = tmp_path / "cleanup.csv"
    logger = GazeDiagnosticsLogger(csv_path=csv_file)
    logger.log_record({"frame_index": 0})
    assert logger._file is not None
    assert not logger._file.closed

    logger.close()
    assert logger._file is None
    assert logger._is_closed is True

    # Repeated close call is safe and idempotent
    logger.close()

    # Logging after close returns False and does not crash
    assert logger.log_record({"frame_index": 1}) is False


def test_run_surveillance_integrates_gaze_diagnostics(tmp_path: Path):
    """
    Verify run_surveillance records raw gaze measurements per frame to gaze_diagnostics.csv
    and cleanly closes resources upon completion.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, dummy_frame), (True, dummy_frame), (False, None)]

    csv_dest = tmp_path / "gaze_diagnostics.csv"

    with patch("config.LOG_DIR", tmp_path), \
         patch("config.SHAPE_PREDICTOR_PATH", tmp_path / "fake.dat"), \
         patch("config.YOLO_MODEL_PATH", tmp_path / "fake.pt"):
        (tmp_path / "fake.dat").write_text("dummy")
        (tmp_path / "fake.pt").write_text("dummy")

        gaze_logger = GazeDiagnosticsLogger(csv_path=csv_dest, session_id="test_surv_sess")
        run_surveillance(cap=mock_cap, display=False, max_frames=2, gaze_logger=gaze_logger)

    assert csv_dest.exists()
    with open(csv_dest, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["frame_index"] == "1"
        assert reader[1]["frame_index"] == "2"
        assert reader[0]["session_id"] == "test_surv_sess"
