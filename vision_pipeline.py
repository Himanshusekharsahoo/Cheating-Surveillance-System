"""
Unified Vision Pipeline for Face Analysis.

Eliminates duplicate face detection and landmark extraction by running
dlib face detection once per frame and reusing landmarks across gaze
and head-pose analysis modules.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import threading
from typing import Optional, Tuple, Any, Dict, Union
import uuid
import cv2
import math
import numpy as np

import config
from event_logger import sanitize_csv_cell
from path_security import resolve_safe_output_path, ensure_safe_directory

from eye_movement import (
    analyze_gaze_from_landmarks,
    get_face_detector,
    get_shape_predictor,
    get_last_gaze_sample,
    get_last_gaze_diagnostics,
    get_last_gaze_tracking_result,
    reset_default_gaze_tracker,
    GAZE_CENTER,
    GAZE_NO_FACE,
    GAZE_UNCERTAIN,
    COMPLIANT_GAZE_STATES,
)
from head_pose import (
    analyze_head_pose_from_landmarks,
    get_camera_matrix,
    get_head_pose_angles,
    get_head_pose_estimation,
    reset_head_pose_state,
    HEAD_SCREEN,
    HEAD_NO_FACE,
    COMPLIANT_HEAD_STATES,
)
from gaze_fusion import (
    HeadPoseResult,
    EyeGazeResult,
    CombinedGazeResult,
    fuse_head_and_eye_gaze,
    eye_direction_to_head_vector,
)
from screen_gaze import (
    ScreenGazeResult,
    ScreenGazeCalibrator,
    BOUNDARY_CALIBRATION_REQUIRED,
    BOUNDARY_UNCERTAIN,
    CALIB_STATUS_NOT_CALIBRATED,
)


GAZE_DIAGNOSTICS_FIELDNAMES = [
    "timestamp",
    "session_id",
    "frame_index",
    "stream_time_sec",
    "face_detected",
    "landmarks_detected",
    "missing_face",
    "missing_landmarks",
    "missing_pupils",
    "missing_left_pupil",
    "missing_right_pupil",
    "left_pupil_detected",
    "right_pupil_detected",
    "left_pupil_x",
    "left_pupil_y",
    "right_pupil_x",
    "right_pupil_y",
    "left_eye_w",
    "left_eye_h",
    "right_eye_w",
    "right_eye_h",
    "norm_left_x",
    "norm_left_y",
    "norm_right_x",
    "norm_right_y",
    "left_pupil_valid",
    "left_rejection_reason",
    "right_pupil_valid",
    "right_rejection_reason",
    "is_calibrated",
    "baseline_left_x",
    "baseline_left_y",
    "baseline_right_x",
    "baseline_right_y",
    "diff_left_x",
    "diff_left_y",
    "diff_right_x",
    "diff_right_y",
    "mean_diff_x",
    "mean_diff_y",
    "gaze_direction",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "head_confidence",
    "head_status",
    "eye_gaze_direction",
    "left_eye_direction",
    "right_eye_direction",
    "eye_status",
    "combined_gaze_yaw",
    "combined_gaze_pitch",
    "combined_gaze_confidence",
    "combined_gaze_status",
    "left_confidence",
    "right_confidence",
    "combined_confidence",
    "tracking_state",
    "is_single_eye_fallback",
]


def _format_float(val: Optional[Any], precision: int = 4) -> str:
    """Format float with fixed precision, or empty string if None/invalid."""
    if val is None:
        return ""
    try:
        return f"{float(val):.{precision}f}"
    except (ValueError, TypeError):
        return ""


def _format_int(val: Optional[Any]) -> str:
    """Format integer, or empty string if None/invalid."""
    if val is None:
        return ""
    try:
        return str(int(val))
    except (ValueError, TypeError):
        return ""


def _format_bool(val: Optional[Any]) -> str:
    """Format boolean as True/False string."""
    return "True" if bool(val) else "False"


def _empty_gaze_diagnostics(is_calibrated: bool = False, calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> Dict[str, Any]:
    """Construct empty diagnostic dictionary for frames where face is not detected."""
    base_lx = calibrated_gaze[0][0] if calibrated_gaze else None
    base_ly = calibrated_gaze[0][1] if calibrated_gaze else None
    base_rx = calibrated_gaze[1][0] if calibrated_gaze else None
    base_ry = calibrated_gaze[1][1] if calibrated_gaze else None
    return {
        "face_detected": False,
        "landmarks_detected": False,
        "missing_face": True,
        "missing_landmarks": True,
        "missing_pupils": True,
        "missing_left_pupil": True,
        "missing_right_pupil": True,
        "left_pupil_detected": False,
        "right_pupil_detected": False,
        "left_pupil_x": None,
        "left_pupil_y": None,
        "right_pupil_x": None,
        "right_pupil_y": None,
        "left_eye_w": None,
        "left_eye_h": None,
        "right_eye_w": None,
        "right_eye_h": None,
        "norm_left_x": None,
        "norm_left_y": None,
        "norm_right_x": None,
        "norm_right_y": None,
        "left_pupil_valid": False,
        "left_rejection_reason": "no_face_detected",
        "right_pupil_valid": False,
        "right_rejection_reason": "no_face_detected",
        "is_calibrated": is_calibrated,
        "baseline_left_x": base_lx,
        "baseline_left_y": base_ly,
        "baseline_right_x": base_rx,
        "baseline_right_y": base_ry,
        "diff_left_x": None,
        "diff_left_y": None,
        "diff_right_x": None,
        "diff_right_y": None,
        "mean_diff_x": None,
        "mean_diff_y": None,
        "gaze_direction": GAZE_NO_FACE,
        "head_pitch": None,
        "head_yaw": None,
        "head_roll": None,
        "left_confidence": 0.0,
        "right_confidence": 0.0,
        "combined_confidence": 0.0,
        "tracking_state": "UNCERTAIN",
        "is_single_eye_fallback": False,
    }


class GazeDiagnosticsLogger:
    """
    Temporary thread-safe diagnostic CSV logger capturing raw gaze measurements.
    Writes one row per processed frame during controlled webcam validation.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        csv_filename: str = "gaze_diagnostics.csv",
        csv_path: Optional[Union[str, Path]] = None,
        storage_manager: Optional[Any] = None,
        encryption_enabled: Optional[bool] = None,
        chunk_size: int = 100,
    ):
        if csv_path is not None:
            raw_path = Path(csv_path)
            self.log_dir = ensure_safe_directory(raw_path.parent)
            self.csv_path = resolve_safe_output_path(
                self.log_dir,
                raw_path.name,
                allow_extensions={".csv"},
            )
        else:
            if log_dir is None:
                from config import LOG_DIR
                raw_dir = Path(LOG_DIR)
            else:
                raw_dir = Path(log_dir)
            self.log_dir = ensure_safe_directory(raw_dir)
            self.csv_path = resolve_safe_output_path(
                self.log_dir,
                csv_filename,
                allow_extensions={".csv"},
            )

        self.session_id = session_id or self._generate_session_id()
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._is_closed = False

        if encryption_enabled is None:
            import config
            self.encryption_enabled = bool(getattr(config, "ENCRYPTION_ENABLED", False))
        else:
            self.encryption_enabled = bool(encryption_enabled)

        self.storage_manager = storage_manager
        self._encrypted_writer = None

        if self.encryption_enabled:
            if self.storage_manager is None:
                from encrypted_storage import get_encrypted_storage_manager
                self.storage_manager = get_encrypted_storage_manager()
            from encrypted_storage import EncryptedCSVWriter
            self._encrypted_writer = EncryptedCSVWriter(
                storage_manager=self.storage_manager,
                log_dir=self.log_dir,
                base_filename=self.csv_path.name,
                session_id=self.session_id,
                fieldnames=GAZE_DIAGNOSTICS_FIELDNAMES,
                chunk_size=chunk_size,
                evidence_type="GAZE_DIAGNOSTICS",
            )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a compact unique session identifier."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        return f"sess_{ts}_{uid}"

    def _ensure_file_open(self):
        """Ensures the CSV file handle is open and header written if needed."""
        if self._file is None or self._file.closed:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            if file_exists:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as existing_file:
                    existing_fields = next(csv.reader(existing_file), [])
                if existing_fields != GAZE_DIAGNOSTICS_FIELDNAMES:
                    original_path = self.csv_path
                    version = 2
                    while True:
                        candidate = original_path.with_name(
                            f"{original_path.stem}_v{version}{original_path.suffix}"
                        )
                        if not candidate.exists():
                            self.csv_path = candidate
                            file_exists = False
                            break
                        with open(candidate, "r", newline="", encoding="utf-8") as version_file:
                            if next(csv.reader(version_file), []) == GAZE_DIAGNOSTICS_FIELDNAMES:
                                self.csv_path = candidate
                                file_exists = candidate.stat().st_size > 0
                                break
                        version += 1
            self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=GAZE_DIAGNOSTICS_FIELDNAMES)
            if not file_exists:
                self._writer.writeheader()
                self._file.flush()

    def log_frame(
        self,
        frame_index: int,
        face_result: "FaceAnalysisResult",
        is_calibrated: bool = False,
        calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        current_time: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Extracts diagnostic values from face_result and logs a single frame row to CSV.
        """
        diag = face_result.gaze_diagnostics or {}
        record = dict(diag)
        record["frame_index"] = frame_index
        record["stream_time_sec"] = current_time
        record["is_calibrated"] = is_calibrated

        if calibrated_gaze is not None:
            record["baseline_left_x"] = calibrated_gaze[0][0]
            record["baseline_left_y"] = calibrated_gaze[0][1]
            record["baseline_right_x"] = calibrated_gaze[1][0]
            record["baseline_right_y"] = calibrated_gaze[1][1]
            norm_lx = record.get("norm_left_x")
            norm_ly = record.get("norm_left_y")
            norm_rx = record.get("norm_right_x")
            norm_ry = record.get("norm_right_y")
            if norm_lx is not None:
                record["diff_left_x"] = calibrated_gaze[0][0] - norm_lx
            if norm_ly is not None:
                record["diff_left_y"] = norm_ly - calibrated_gaze[0][1]
            if norm_rx is not None:
                record["diff_right_x"] = calibrated_gaze[1][0] - norm_rx
            if norm_ry is not None:
                record["diff_right_y"] = norm_ry - calibrated_gaze[1][1]
            if record.get("diff_left_x") is not None and record.get("diff_right_x") is not None:
                record["mean_diff_x"] = (record["diff_left_x"] + record["diff_right_x"]) / 2.0
            if record.get("diff_left_y") is not None and record.get("diff_right_y") is not None:
                record["mean_diff_y"] = (record["diff_left_y"] + record["diff_right_y"]) / 2.0

        if face_result.raw_head_pose is not None:
            record["head_pitch"] = face_result.raw_head_pose[0]
            record["head_yaw"] = face_result.raw_head_pose[1]
            record["head_roll"] = face_result.raw_head_pose[2]
        if face_result.head_pose is not None:
            record["head_confidence"] = face_result.head_pose.confidence
            record["head_status"] = face_result.head_pose.status
        if face_result.eye_gaze is not None:
            record["eye_gaze_direction"] = face_result.eye_gaze.selected_eye_direction
            record["left_eye_direction"] = face_result.eye_gaze.left_eye_direction
            record["right_eye_direction"] = face_result.eye_gaze.right_eye_direction
            record["eye_status"] = face_result.eye_gaze.status
        if face_result.combined_gaze is not None:
            record["combined_gaze_yaw"] = face_result.combined_gaze.yaw
            record["combined_gaze_pitch"] = face_result.combined_gaze.pitch
            record["combined_gaze_confidence"] = face_result.combined_gaze.confidence
            record["combined_gaze_status"] = face_result.combined_gaze.status

        record["gaze_direction"] = face_result.gaze_direction

        return self.log_record(record, timestamp=timestamp)

    def log_record(
        self,
        record: dict,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Formats and appends a single diagnostic record to the CSV file.
        """
        if self._is_closed:
            return False

        if timestamp is None:
            ts_str = datetime.now().astimezone().isoformat()
        else:
            ts_str = str(timestamp)

        row = {
            "timestamp": ts_str,
            "session_id": self.session_id,
            "frame_index": _format_int(record.get("frame_index")),
            "stream_time_sec": _format_float(record.get("stream_time_sec"), 3),
            "face_detected": _format_bool(record.get("face_detected", False)),
            "landmarks_detected": _format_bool(record.get("landmarks_detected", False)),
            "missing_face": _format_bool(record.get("missing_face", True)),
            "missing_landmarks": _format_bool(record.get("missing_landmarks", True)),
            "missing_pupils": _format_bool(record.get("missing_pupils", True)),
            "missing_left_pupil": _format_bool(record.get("missing_left_pupil", True)),
            "missing_right_pupil": _format_bool(record.get("missing_right_pupil", True)),
            "left_pupil_detected": _format_bool(record.get("left_pupil_detected", False)),
            "right_pupil_detected": _format_bool(record.get("right_pupil_detected", False)),
            "left_pupil_x": _format_int(record.get("left_pupil_x")),
            "left_pupil_y": _format_int(record.get("left_pupil_y")),
            "right_pupil_x": _format_int(record.get("right_pupil_x")),
            "right_pupil_y": _format_int(record.get("right_pupil_y")),
            "left_eye_w": _format_int(record.get("left_eye_w")),
            "left_eye_h": _format_int(record.get("left_eye_h")),
            "right_eye_w": _format_int(record.get("right_eye_w")),
            "right_eye_h": _format_int(record.get("right_eye_h")),
            "norm_left_x": _format_float(record.get("norm_left_x"), 4),
            "norm_left_y": _format_float(record.get("norm_left_y"), 4),
            "norm_right_x": _format_float(record.get("norm_right_x"), 4),
            "norm_right_y": _format_float(record.get("norm_right_y"), 4),
            "left_pupil_valid": _format_bool(record.get("left_pupil_valid", False)),
            "left_rejection_reason": str(record.get("left_rejection_reason", "")),
            "right_pupil_valid": _format_bool(record.get("right_pupil_valid", False)),
            "right_rejection_reason": str(record.get("right_rejection_reason", "")),
            "is_calibrated": _format_bool(record.get("is_calibrated", False)),
            "baseline_left_x": _format_float(record.get("baseline_left_x"), 4),
            "baseline_left_y": _format_float(record.get("baseline_left_y"), 4),
            "baseline_right_x": _format_float(record.get("baseline_right_x"), 4),
            "baseline_right_y": _format_float(record.get("baseline_right_y"), 4),
            "diff_left_x": _format_float(record.get("diff_left_x"), 4),
            "diff_left_y": _format_float(record.get("diff_left_y"), 4),
            "diff_right_x": _format_float(record.get("diff_right_x"), 4),
            "diff_right_y": _format_float(record.get("diff_right_y"), 4),
            "mean_diff_x": _format_float(record.get("mean_diff_x"), 4),
            "mean_diff_y": _format_float(record.get("mean_diff_y"), 4),
            "gaze_direction": str(record.get("gaze_direction", "")),
            "head_pitch": _format_float(record.get("head_pitch"), 2),
            "head_yaw": _format_float(record.get("head_yaw"), 2),
            "head_roll": _format_float(record.get("head_roll"), 2),
            "head_confidence": _format_float(record.get("head_confidence"), 4),
            "head_status": str(record.get("head_status", "")),
            "eye_gaze_direction": str(record.get("eye_gaze_direction", "")),
            "left_eye_direction": str(record.get("left_eye_direction") or ""),
            "right_eye_direction": str(record.get("right_eye_direction") or ""),
            "eye_status": str(record.get("eye_status", "")),
            "combined_gaze_yaw": _format_float(record.get("combined_gaze_yaw"), 2),
            "combined_gaze_pitch": _format_float(record.get("combined_gaze_pitch"), 2),
            "combined_gaze_confidence": _format_float(record.get("combined_gaze_confidence"), 4),
            "combined_gaze_status": str(record.get("combined_gaze_status", "")),
            "left_confidence": _format_float(record.get("left_confidence"), 4),
            "right_confidence": _format_float(record.get("right_confidence"), 4),
            "combined_confidence": _format_float(record.get("combined_confidence"), 4),
            "tracking_state": str(record.get("tracking_state", "")),
            "is_single_eye_fallback": _format_bool(record.get("is_single_eye_fallback", False)),
        }

        if self.encryption_enabled and self._encrypted_writer is not None:
            with self._lock:
                try:
                    sanitized_row = {k: sanitize_csv_cell(v) for k, v in row.items()}
                    return self._encrypted_writer.write_record(sanitized_row)
                except Exception as e:
                    print(f"Warning: Exception logging encrypted gaze diagnostic record: {e}")
                    return False

        with self._lock:
            try:
                self._ensure_file_open()
                sanitized_row = {k: sanitize_csv_cell(v) for k, v in row.items()}
                self._writer.writerow(sanitized_row)
                self._file.flush()
                return True
            except Exception as e:
                print(f"Warning: Exception logging gaze diagnostic record: {e}")
                return False

    def flush(self):
        """Flushes written CSV records to disk."""
        with self._lock:
            if self.encryption_enabled and self._encrypted_writer is not None:
                self._encrypted_writer.flush()
                return
            if self._file and not self._file.closed:
                try:
                    self._file.flush()
                except Exception:
                    pass

    def close(self):
        """Closes the CSV file handle cleanly."""
        with self._lock:
            if not self._is_closed:
                self._is_closed = True
                if self.encryption_enabled and self._encrypted_writer is not None:
                    self._encrypted_writer.close()
                    return
                if self._file and not self._file.closed:
                    try:
                        self._file.flush()
                        self._file.close()
                    except Exception:
                        pass
                    self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass
class FaceAnalysisResult:
    """Holds the result of a single frame's face analysis."""
    face_count: int = 0
    face_rect: Optional[Any] = None
    landmarks: Optional[Any] = None
    gaze_direction: str = GAZE_NO_FACE
    head_direction: str = HEAD_NO_FACE
    head_pose_sample: Optional[Tuple[float, float, float]] = None
    gaze_sample: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    is_multi_face: bool = False
    raw_head_pose: Optional[Tuple[float, float, float]] = None
    gaze_diagnostics: Optional[Dict[str, Any]] = None
    # Phase 10.13.4 Structured Output Models
    head_pose: Optional[HeadPoseResult] = None
    eye_gaze: Optional[EyeGazeResult] = None
    combined_gaze: Optional[CombinedGazeResult] = None
    # Phase 10.13.5 Screen Gaze Output Model
    screen_gaze: Optional[ScreenGazeResult] = None
    # Phase 10.13.13 Runtime Resilience & Lifecycle (GAP-04)
    processing_status: str = "SUCCESS"
    error: Optional[str] = None


class VisionPipeline:
    """
    Unified face analysis pipeline that coordinates face detection, landmark extraction,
    gaze tracking, and head pose estimation in a single pass per frame.
    """

    def __init__(self, detector=None, predictor=None, screen_calibrator=None):
        self._detector = detector
        self._predictor = predictor
        self.screen_calibrator = screen_calibrator
        self._last_error_log: Dict[str, float] = {}
        self._error_counts: Dict[str, int] = {}

    def reset(self) -> None:
        """Reset all tracking and temporal smoothing states across eye and head subsystems (GAP-05)."""
        reset_default_gaze_tracker()
        reset_head_pose_state()

    def _log_exception_throttled(self, source: str, error: Exception, min_interval_sec: float = 3.0) -> None:
        """Log diagnostic error message with throttling to prevent per-frame console spam (GAP-04)."""
        import time
        now = time.monotonic()
        last_time = self._last_error_log.get(source, 0.0)
        self._error_counts[source] = self._error_counts.get(source, 0) + 1
        count = self._error_counts[source]

        if (now - last_time) >= min_interval_sec or count == 1:
            self._last_error_log[source] = now
            suffix = f" (repeated {count} times)" if count > 1 else ""
            print(f"Warning: {source} exception during frame processing: {error}{suffix}")

    def _reset_error_counts(self) -> None:
        """Reset error suppression counts upon successful frame processing."""
        self._error_counts.clear()

    @property
    def detector(self):
        if self._detector is None:
            self._detector = get_face_detector()
        return self._detector

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = get_shape_predictor()
        return self._predictor

    def process_face(
        self,
        frame: np.ndarray,
        calibrated_angles: Optional[Tuple[float, float, float]] = None,
        calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        camera_matrix: Optional[np.ndarray] = None,
        screen_calibrator: Optional[ScreenGazeCalibrator] = None,
    ) -> Tuple[FaceAnalysisResult, np.ndarray]:
        """
        Executes face detection once on the frame, extracts landmarks once for the primary face,
        and delegates to gaze and head-pose analysis.

        Returns:
            Tuple[FaceAnalysisResult, np.ndarray]: (analysis_result, annotated_frame)
        """
        active_calib = screen_calibrator or self.screen_calibrator
        if frame is None or frame.size == 0:
            self.reset()
            diag = _empty_gaze_diagnostics(is_calibrated=(calibrated_gaze is not None), calibrated_gaze=calibrated_gaze)
            empty_head = HeadPoseResult(status="NO_FACE", coordinate_frame="camera_opencv")
            empty_eye = EyeGazeResult(selected_eye_direction=GAZE_NO_FACE, status="NO_FACE", coordinate_frame="head_local")
            empty_comb = CombinedGazeResult(status="NO_FACE", confidence=0.0, uncertainty=1.0)
            empty_screen = ScreenGazeResult(
                status="UNCERTAIN" if (active_calib and active_calib.is_calibrated) else CALIB_STATUS_NOT_CALIBRATED,
                boundary_state=BOUNDARY_UNCERTAIN if (active_calib and active_calib.is_calibrated) else BOUNDARY_CALIBRATION_REQUIRED,
                confidence=0.0,
                is_calibrated=bool(active_calib and active_calib.is_calibrated),
            )
            return FaceAnalysisResult(
                face_count=0,
                face_rect=None,
                landmarks=None,
                gaze_direction=GAZE_NO_FACE,
                head_direction=HEAD_NO_FACE,
                head_pose_sample=None,
                gaze_sample=None,
                is_multi_face=False,
                raw_head_pose=None,
                gaze_diagnostics=diag,
                head_pose=empty_head,
                eye_gaze=empty_eye,
                combined_gaze=empty_comb,
                screen_gaze=empty_screen,
                processing_status="NO_FACE",
            ), frame

        # Convert to grayscale once per frame with exception safety
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            gray = None

        if gray is None:
            self.reset()
            empty_head = HeadPoseResult(status="DETECTION_ERROR", coordinate_frame="camera_opencv")
            empty_eye = EyeGazeResult(
                selected_eye_direction=GAZE_UNCERTAIN,
                status="DETECTION_ERROR",
                tracking_state="ERROR",
                rejection_reason="Grayscale conversion failed",
                coordinate_frame="head_local",
            )
            empty_comb = CombinedGazeResult(status="DETECTION_ERROR", confidence=0.0, uncertainty=1.0)
            empty_screen = ScreenGazeResult(
                status="UNCERTAIN",
                boundary_state=BOUNDARY_UNCERTAIN,
                confidence=0.0,
                is_calibrated=bool(active_calib and active_calib.is_calibrated),
            )
            diag = _empty_gaze_diagnostics(is_calibrated=(calibrated_gaze is not None), calibrated_gaze=calibrated_gaze)
            diag["detection_error"] = "Grayscale conversion failed"
            diag["processing_status"] = "DETECTION_ERROR"
            return FaceAnalysisResult(
                face_count=0,
                face_rect=None,
                landmarks=None,
                gaze_direction=GAZE_UNCERTAIN,
                head_direction="DETECTION_ERROR",
                head_pose_sample=None,
                gaze_sample=None,
                is_multi_face=False,
                raw_head_pose=None,
                gaze_diagnostics=diag,
                head_pose=empty_head,
                eye_gaze=empty_eye,
                combined_gaze=empty_comb,
                screen_gaze=empty_screen,
                processing_status="DETECTION_ERROR",
                error="Grayscale conversion failed",
            ), frame

        # Execute face detection ONCE with scoped exception handling (GAP-04)
        detector_error = None
        faces = None
        try:
            faces = self.detector(gray)
        except Exception as e:
            detector_error = e
            self._log_exception_throttled("face_detector", e)

        if detector_error is not None:
            self.reset()
            empty_head = HeadPoseResult(status="DETECTION_ERROR", coordinate_frame="camera_opencv")
            empty_eye = EyeGazeResult(
                selected_eye_direction=GAZE_UNCERTAIN,
                status="DETECTION_ERROR",
                tracking_state="ERROR",
                rejection_reason=f"Face detector failure: {detector_error}",
                coordinate_frame="head_local",
            )
            empty_comb = CombinedGazeResult(
                status="DETECTION_ERROR",
                confidence=0.0,
                uncertainty=1.0,
                source_components={"error": str(detector_error), "error_type": "face_detector_error"},
            )
            empty_screen = ScreenGazeResult(
                status="UNCERTAIN",
                boundary_state=BOUNDARY_UNCERTAIN,
                confidence=0.0,
                is_calibrated=bool(active_calib and active_calib.is_calibrated),
            )
            diag = _empty_gaze_diagnostics(is_calibrated=(calibrated_gaze is not None), calibrated_gaze=calibrated_gaze)
            diag["detection_error"] = str(detector_error)
            diag["processing_status"] = "DETECTION_ERROR"
            return FaceAnalysisResult(
                face_count=0,
                face_rect=None,
                landmarks=None,
                gaze_direction=GAZE_UNCERTAIN,
                head_direction="DETECTION_ERROR",
                head_pose_sample=None,
                gaze_sample=None,
                is_multi_face=False,
                raw_head_pose=None,
                gaze_diagnostics=diag,
                head_pose=empty_head,
                eye_gaze=empty_eye,
                combined_gaze=empty_comb,
                screen_gaze=empty_screen,
                processing_status="DETECTION_ERROR",
                error=str(detector_error),
            ), frame

        face_count = len(faces) if faces is not None else 0

        if face_count == 0:
            self.reset()
            diag = _empty_gaze_diagnostics(is_calibrated=(calibrated_gaze is not None), calibrated_gaze=calibrated_gaze)
            empty_head = HeadPoseResult(status="NO_FACE", coordinate_frame="camera_opencv")
            empty_eye = EyeGazeResult(selected_eye_direction=GAZE_NO_FACE, status="NO_FACE", coordinate_frame="head_local")
            empty_comb = CombinedGazeResult(status="NO_FACE", confidence=0.0, uncertainty=1.0)
            empty_screen = ScreenGazeResult(
                status="UNCERTAIN" if (active_calib and active_calib.is_calibrated) else CALIB_STATUS_NOT_CALIBRATED,
                boundary_state=BOUNDARY_UNCERTAIN if (active_calib and active_calib.is_calibrated) else BOUNDARY_CALIBRATION_REQUIRED,
                confidence=0.0,
                is_calibrated=bool(active_calib and active_calib.is_calibrated),
            )
            return FaceAnalysisResult(
                face_count=0,
                face_rect=None,
                landmarks=None,
                gaze_direction=GAZE_NO_FACE,
                head_direction=HEAD_NO_FACE,
                head_pose_sample=None,
                gaze_sample=None,
                is_multi_face=False,
                raw_head_pose=None,
                gaze_diagnostics=diag,
                head_pose=empty_head,
                eye_gaze=empty_eye,
                combined_gaze=empty_comb,
                screen_gaze=empty_screen,
                processing_status="NO_FACE",
            ), frame

        # Handle multiple faces explicitly
        is_multi_face = (face_count > 1)

        # Select primary face: choose the largest face bounding box by area
        primary_face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))

        # Annotate detected faces (draw rectangle for all detected faces)
        for f in faces:
            color = (0, 255, 0) if f == primary_face else (0, 165, 255)
            cv2.rectangle(frame, (f.left(), f.top()), (f.right(), f.bottom()), color, 2)

        # Extract landmarks ONCE for the primary face with scoped exception handling (GAP-04)
        predictor_error = None
        landmarks = None
        try:
            landmarks = self.predictor(gray, primary_face)
        except Exception as e:
            predictor_error = e
            self._log_exception_throttled("landmark_predictor", e)

        if predictor_error is not None or landmarks is None:
            self.reset()
            empty_head = HeadPoseResult(status="PREDICTION_ERROR", coordinate_frame="camera_opencv")
            empty_eye = EyeGazeResult(
                selected_eye_direction=GAZE_UNCERTAIN,
                status="PREDICTION_ERROR",
                tracking_state="ERROR",
                rejection_reason=f"Landmark predictor failure: {predictor_error}",
                coordinate_frame="head_local",
            )
            empty_comb = CombinedGazeResult(
                status="PREDICTION_ERROR",
                confidence=0.0,
                uncertainty=1.0,
                source_components={"error": str(predictor_error), "error_type": "landmark_predictor_error"},
            )
            empty_screen = ScreenGazeResult(
                status="UNCERTAIN",
                boundary_state=BOUNDARY_UNCERTAIN,
                confidence=0.0,
                is_calibrated=bool(active_calib and active_calib.is_calibrated),
            )
            diag = _empty_gaze_diagnostics(is_calibrated=(calibrated_gaze is not None), calibrated_gaze=calibrated_gaze)
            diag["prediction_error"] = str(predictor_error)
            diag["processing_status"] = "PREDICTION_ERROR"
            diag["face_detected"] = True
            return FaceAnalysisResult(
                face_count=face_count,
                face_rect=primary_face,
                landmarks=None,
                gaze_direction=GAZE_UNCERTAIN,
                head_direction="PREDICTION_ERROR",
                head_pose_sample=None,
                gaze_sample=None,
                is_multi_face=is_multi_face,
                raw_head_pose=None,
                gaze_diagnostics=diag,
                head_pose=empty_head,
                eye_gaze=empty_eye,
                combined_gaze=empty_comb,
                screen_gaze=empty_screen,
                processing_status="PREDICTION_ERROR",
                error=str(predictor_error),
            ), frame

        # Extract raw head-pose angles and 3D estimation
        raw_head_pose = None
        head_estimation = None
        try:
            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),    # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
                (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
            ], dtype=np.float64)
            cam_mat = camera_matrix if camera_matrix is not None else get_camera_matrix(frame.shape)
            head_estimation = get_head_pose_estimation(image_points, camera_matrix=cam_mat, frame_shape=frame.shape)
            if head_estimation is not None:
                raw_head_pose = head_estimation[0]
        except Exception:
            raw_head_pose = None
            head_estimation = None

        # Analyze gaze from landmarks with optional calibrated neutral gaze baseline
        frame, gaze_direction = analyze_gaze_from_landmarks(
            frame,
            landmarks,
            calibrated_gaze=calibrated_gaze
        )
        gaze_sample = get_last_gaze_sample()
        gaze_diag = get_last_gaze_diagnostics()
        if gaze_diag is not None:
            gaze_diag = dict(gaze_diag)
            if raw_head_pose is not None:
                gaze_diag["head_pitch"] = raw_head_pose[0]
                gaze_diag["head_yaw"] = raw_head_pose[1]
                gaze_diag["head_roll"] = raw_head_pose[2]
            else:
                gaze_diag["head_pitch"] = None
                gaze_diag["head_yaw"] = None
                gaze_diag["head_roll"] = None

        # Analyze head pose from landmarks
        frame, head_result = analyze_head_pose_from_landmarks(
            frame,
            landmarks,
            calibrated_angles=calibrated_angles,
            camera_matrix=camera_matrix
        )

        head_sample = None
        head_direction = HEAD_NO_FACE
        if calibrated_angles is None:
            # During calibration, head_result is (pitch, yaw, roll) tuple or None
            head_sample = head_result
            head_direction = "Calibrating"
        else:
            head_direction = head_result if isinstance(head_result, str) else HEAD_SCREEN

        # Construct Output A - HeadPoseResult
        if head_estimation is not None:
            hp_angles, hp_R_cam_head, hp_tvec, hp_conf = head_estimation
            try:
                raw_hp_conf = float(hp_conf)
                if not math.isfinite(raw_hp_conf) or raw_hp_conf < 0.0:
                    raw_hp_conf = 0.0
            except (TypeError, ValueError):
                raw_hp_conf = 0.0

            head_threshold = float(config.FUSED_GAZE_MIN_CONFIDENCE)
            # Classification must use the unrounded source measurement
            head_status = "VALID" if raw_hp_conf >= head_threshold else "UNCERTAIN"

            # Rounding must never promote an uncertain measurement into a valid one
            rounded_hp_conf = round(raw_hp_conf, 4)
            if raw_hp_conf < head_threshold and rounded_hp_conf >= head_threshold:
                rounded_hp_conf = round(head_threshold - 1e-4, 4)

            head_pose_res = HeadPoseResult(
                yaw=round(float(hp_angles[1]), 2),
                pitch=round(float(hp_angles[0]), 2),
                roll=round(float(hp_angles[2]), 2),
                rotation_matrix=hp_R_cam_head,
                translation=hp_tvec,
                confidence=rounded_hp_conf,
                status=head_status,
                coordinate_frame="camera_opencv",
            )
        else:
            head_pose_res = HeadPoseResult(
                yaw=round(float(raw_head_pose[1]), 2) if raw_head_pose else None,
                pitch=round(float(raw_head_pose[0]), 2) if raw_head_pose else None,
                roll=round(float(raw_head_pose[2]), 2) if raw_head_pose else None,
                rotation_matrix=None,
                translation=None,
                confidence=0.0,
                status="UNCERTAIN",
                coordinate_frame="camera_opencv",
            )

        # Construct Output B - EyeGazeResult
        tracking_res = get_last_gaze_tracking_result()
        diff_x = gaze_diag.get("mean_diff_x") if gaze_diag else None
        diff_y = gaze_diag.get("mean_diff_y") if gaze_diag else None
        v_head = eye_direction_to_head_vector(gaze_direction, diff_x, diff_y)

        # Defense-in-depth against stale tracker state leakage (ISSUE-02):
        # Observation is invalid if gaze direction is uncertain/calibrating, or if diagnostics report missing pupils
        # without an active validated single-eye fallback
        is_observation_invalid = (
            gaze_direction in (GAZE_UNCERTAIN, GAZE_NO_FACE, "Calibrating", "") or
            (gaze_diag is not None and gaze_diag.get("missing_pupils", False) and (tracking_res is None or not tracking_res.is_single_eye_fallback)) or
            (gaze_diag is not None and (gaze_diag.get("missing_face", False) or gaze_diag.get("missing_landmarks", False)))
        )

        if tracking_res is not None and not is_observation_invalid:
            eye_status = "VALID" if tracking_res.tracking_state in ("PAIRED_TRACKING", "LEFT_EYE_ONLY", "RIGHT_EYE_ONLY") else "UNCERTAIN"
            eye_gaze_res = EyeGazeResult(
                left_eye_direction=tracking_res.left_gaze_dir,
                right_eye_direction=tracking_res.right_gaze_dir,
                selected_eye_direction=gaze_direction,
                confidence=round(float(tracking_res.combined_confidence), 4),
                tracking_state=tracking_res.tracking_state,
                status=eye_status,
                coordinate_frame="head_local",
                gaze_vector_head=v_head,
                is_single_eye_fallback=tracking_res.is_single_eye_fallback,
                left_confidence=round(float(tracking_res.left_confidence), 4),
                right_confidence=round(float(tracking_res.right_confidence), 4),
                rejection_reason=tracking_res.rejection_reason,
            )
        else:
            rejection = tracking_res.rejection_reason if tracking_res else (gaze_diag.get("left_rejection_reason") if gaze_diag else "missing_observation")
            eye_gaze_res = EyeGazeResult(
                left_eye_direction=None,
                right_eye_direction=None,
                selected_eye_direction=gaze_direction,
                confidence=0.0,
                tracking_state="UNCERTAIN",
                status="UNCERTAIN",
                coordinate_frame="head_local",
                gaze_vector_head=None,
                is_single_eye_fallback=False,
                left_confidence=0.0,
                right_confidence=0.0,
                rejection_reason=str(rejection or "uncertain_or_missing_observation"),
            )

        # Construct Output C - CombinedGazeResult
        combined_gaze_res = fuse_head_and_eye_gaze(head_pose_res, eye_gaze_res)

        # Construct Output D - ScreenGazeResult (Phase 10.13.5)
        if active_calib is not None and active_calib.is_calibrated:
            screen_gaze_res = active_calib.estimate_screen_gaze(
                gaze_yaw=combined_gaze_res.yaw,
                gaze_pitch=combined_gaze_res.pitch,
                gaze_confidence=combined_gaze_res.confidence,
                gaze_status=combined_gaze_res.status,
            )
        else:
            screen_gaze_res = ScreenGazeResult(
                screen_x=None,
                screen_y=None,
                boundary_state=BOUNDARY_CALIBRATION_REQUIRED,
                confidence=0.0,
                uncertainty=1.0,
                status=CALIB_STATUS_NOT_CALIBRATED,
                is_calibrated=False,
                source_components={"error": "not_calibrated"},
            )

        result = FaceAnalysisResult(
            face_count=face_count,
            face_rect=primary_face,
            landmarks=landmarks,
            gaze_direction=gaze_direction,
            head_direction=head_direction,
            head_pose_sample=head_sample,
            gaze_sample=gaze_sample,
            is_multi_face=is_multi_face,
            raw_head_pose=raw_head_pose,
            gaze_diagnostics=gaze_diag,
            head_pose=head_pose_res,
            eye_gaze=eye_gaze_res,
            combined_gaze=combined_gaze_res,
            screen_gaze=screen_gaze_res,
        )

        self._reset_error_counts()
        return result, frame


_default_pipeline: Optional[VisionPipeline] = None


def get_vision_pipeline() -> VisionPipeline:
    """Lazily loads and returns the shared VisionPipeline singleton."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = VisionPipeline()
    return _default_pipeline


def process_frame_face(
    frame: np.ndarray,
    calibrated_angles: Optional[Tuple[float, float, float]] = None,
    calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    pipeline: Optional[VisionPipeline] = None,
    screen_calibrator: Optional[ScreenGazeCalibrator] = None,
) -> Tuple[FaceAnalysisResult, np.ndarray]:
    """
    Convenience function for processing a single frame through the shared face pipeline.
    """
    active_pipeline = pipeline or get_vision_pipeline()
    return active_pipeline.process_face(
        frame,
        calibrated_angles=calibrated_angles,
        calibrated_gaze=calibrated_gaze,
        screen_calibrator=screen_calibrator,
    )
