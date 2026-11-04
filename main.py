import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Any
import cv2
import numpy as np

import config
import re
from path_security import resolve_safe_output_path, validate_safe_read_path
from vision_pipeline import process_frame_face, GazeDiagnosticsLogger
from eye_movement import (
    GAZE_CENTER,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    GAZE_DOWN,
    GAZE_NO_FACE,
    GAZE_UNCERTAIN,
    COMPLIANT_GAZE_STATES,
    calibrate_gaze_baseline,
    reset_default_gaze_tracker,
)
from head_pose import (
    HEAD_SCREEN,
    HEAD_LEFT,
    HEAD_RIGHT,
    HEAD_UP,
    HEAD_DOWN,
    HEAD_TILTED,
    HEAD_NO_FACE,
    COMPLIANT_HEAD_STATES,
    reset_head_pose_state,
)
from mobile_detection import (
    PhoneDetectionTracker,
    PhoneDetectionResult,
    DETECTION_STATE_FRESH_POSITIVE,
    DETECTION_STATE_FRESH_NEGATIVE,
    DETECTION_STATE_CACHED_POSITIVE,
    DETECTION_STATE_CACHED_NEGATIVE,
    DETECTION_STATE_EXPIRED,
    DETECTION_STATE_INFERENCE_ERROR,
    process_mobile_detection,
    get_last_detected_confidence,
)
from event_logger import EventLogger, get_event_logger
from telemetry import PerformanceTelemetry
from time_source import TimeSource, LiveTimeSource, VideoTimeSource, MockTimeSource
from screen_gaze import (
    ScreenGazeResult,
    ScreenGazeCalibrator,
    ScreenCalibrationManager,
    WINDOW_CALIBRATION_NAME,
    BOUNDARY_ON_SCREEN,
    BOUNDARY_NEAR_BOUNDARY,
    BOUNDARY_OUTSIDE_SCREEN,
    BOUNDARY_UNCERTAIN,
    BOUNDARY_CALIBRATION_REQUIRED,
    CALIB_STATE_NOT_CALIBRATED,
    CALIB_STATE_CALIBRATING,
    CALIB_STATE_VALID,
    CALIB_STATE_FAILED,
    CALIB_STATE_CANCELLED,
    get_primary_display_resolution,
    load_screen_calibration,
    resolve_capture_geometry,
    screen_calibration_environment,
)


def parse_args(args=None):
    """
    Parses command-line arguments for Cheating Surveillance System.
    """
    parser = argparse.ArgumentParser(
        description="Cheating Surveillance System - AI proctoring for online assessments."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Index of the webcam video device to capture (e.g. 0).",
    )
    source_group.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a recorded video file to process.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without displaying GUI windows.",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=None,
        help="YOLO mobile detection inference interval in frames (overrides config).",
    )
    parser.add_argument(
        "--calibrate-screen",
        action="store_true",
        default=False,
        help="Perform interactive 5-point screen gaze calibration after baseline setup.",
    )
    parser.add_argument(
        "--retention-audit",
        action="store_true",
        default=False,
        help="Run a dry-run privacy and retention audit of stored evidence.",
    )
    parser.add_argument(
        "--retention-purge",
        action="store_true",
        default=False,
        help="Purge expired evidence files exceeding retention period (requires --no-dry-run).",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        default=False,
        help="Authorize actual deletion during retention purge (default is dry-run).",
    )
    return parser.parse_args(args)


def validate_cli_args(parsed_args):
    """
    Validates CLI arguments. Raises ValueError or FileNotFoundError if invalid.
    """
    if parsed_args.camera is not None and parsed_args.camera < 0:
        raise ValueError(f"Camera index must be a non-negative integer, got {parsed_args.camera}")

    if parsed_args.video is not None:
        video_path = Path(parsed_args.video)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found at '{parsed_args.video}'")

    if parsed_args.detection_interval is not None and parsed_args.detection_interval < 1:
        raise ValueError(
            f"Detection interval must be a positive integer >= 1, got {parsed_args.detection_interval}"
        )
    if parsed_args.calibrate_screen and parsed_args.headless:
        raise ValueError("--calibrate-screen requires display mode; remove --headless.")
    if parsed_args.calibrate_screen and parsed_args.video is not None:
        raise ValueError("Interactive screen calibration requires a webcam source, not --video.")


def gaze_occlusion_is_confirmed(diagnostics):
    """Require both pupil detections to fail before logging sustained gaze occlusion."""
    if not diagnostics:
        return False
    if diagnostics.get("detection_error") or diagnostics.get("prediction_error") or diagnostics.get("missing_face"):
        return False
    return bool(
        diagnostics.get("missing_left_pupil")
        and diagnostics.get("missing_right_pupil")
    )


def classify_combined_attention(combined_gaze):
    """Classify camera-relative fused gaze (+yaw right, +pitch up); uncertain inputs never count."""
    if combined_gaze is None:
        return GAZE_UNCERTAIN
    if combined_gaze.status == "NO_FACE":
        return GAZE_NO_FACE
    values = (combined_gaze.yaw, combined_gaze.pitch, combined_gaze.confidence)
    try:
        usable_values = all(value is not None and math.isfinite(float(value)) for value in values)
        confidence = float(combined_gaze.confidence)
    except (TypeError, ValueError):
        usable_values = False
        confidence = 0.0
    if combined_gaze.status != "VALID" or not usable_values or not (
        config.FUSED_GAZE_MIN_CONFIDENCE <= confidence <= 1.0
    ):
        return GAZE_UNCERTAIN

    if combined_gaze.yaw > config.HEAD_DEVIATION_YAW_THRESHOLD:
        return GAZE_RIGHT
    if combined_gaze.yaw < -config.HEAD_DEVIATION_YAW_THRESHOLD:
        return GAZE_LEFT
    if combined_gaze.pitch > config.HEAD_DEVIATION_PITCH_THRESHOLD:
        return GAZE_UP
    if combined_gaze.pitch < -config.HEAD_DEVIATION_PITCH_THRESHOLD:
        return GAZE_DOWN
    return GAZE_CENTER


def _format_angle(value):
    try:
        return f"{float(value):+.1f}°" if value is not None and math.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return "—"


def combined_gaze_hud(combined_gaze):
    """Build HUD text without presenting missing angles as valid zeroes."""
    if combined_gaze is None:
        return "Fused Gaze: UNAVAILABLE | yaw — pitch —", "Fused Confidence: —"
    state = classify_combined_attention(combined_gaze)
    if state == GAZE_UNCERTAIN and combined_gaze.status == "VALID":
        state = "UNCERTAIN"
    confidence = combined_gaze.confidence
    try:
        confidence_text = f"{float(confidence) * 100:.0f}%" if math.isfinite(float(confidence)) else "—"
    except (TypeError, ValueError):
        confidence_text = "—"
    return (
        f"Fused Gaze: {state} | yaw {_format_angle(combined_gaze.yaw)} pitch {_format_angle(combined_gaze.pitch)}",
        f"Fused Confidence: {confidence_text}",
    )


def screen_gaze_hud(screen_gaze: Optional[ScreenGazeResult]) -> Tuple[str, str]:
    """Format screen gaze coordinates, boundary state, and confidence for HUD display."""
    if screen_gaze is None or not getattr(screen_gaze, "is_calibrated", False):
        return "Screen Gaze: NOT_CALIBRATED", "Screen Pos: (—, —) | Boundary: CALIBRATION_REQUIRED"

    status = getattr(screen_gaze, "status", "NOT_CALIBRATED")
    boundary = getattr(screen_gaze, "boundary_state", BOUNDARY_CALIBRATION_REQUIRED)
    confidence = getattr(screen_gaze, "confidence", 0.0)

    try:
        conf_pct = f"{float(confidence) * 100:.0f}%" if math.isfinite(float(confidence)) else "—"
    except (TypeError, ValueError):
        conf_pct = "—"

    sx = getattr(screen_gaze, "screen_x", None)
    sy = getattr(screen_gaze, "screen_y", None)

    if status == "VALID" and sx is not None and sy is not None:
        try:
            pos_str = f"({float(sx):.2f}, {float(sy):.2f})"
        except (TypeError, ValueError):
            pos_str = "(—, —)"
        return (
            f"Screen Gaze: {boundary} | Pos: {pos_str} ({conf_pct})",
            f"Screen Status: {status} | Boundary: {boundary}",
        )
    elif status == "UNCERTAIN":
        return "Screen Gaze: UNCERTAIN", "Screen Pos: (—, —) | Boundary: UNCERTAIN"
    else:
        return f"Screen Gaze: {status}", f"Screen Pos: (—, —) | Boundary: {boundary}"


def is_confirmed_outside_screen(screen_gaze: Optional[ScreenGazeResult]) -> bool:
    """Require a calibrated, finite, sufficiently confident outside-screen result."""
    if (
        screen_gaze is None
        or not getattr(screen_gaze, "is_calibrated", False)
        or getattr(screen_gaze, "status", None) != "VALID"
        or getattr(screen_gaze, "boundary_state", None) != BOUNDARY_OUTSIDE_SCREEN
    ):
        return False
    try:
        x = float(screen_gaze.screen_x)
        y = float(screen_gaze.screen_y)
        confidence = float(screen_gaze.confidence)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence)
        and config.SCREEN_CALIBRATION_MIN_CONFIDENCE <= confidence <= 1.0
    )


def record_violation_incident(
    violation_type: str,
    prefix: str,
    frame: np.ndarray,
    duration_sec: float,
    confidence: Optional[float] = None,
    log_dir: str = str(config.LOG_DIR),
    logger: Optional[EventLogger] = None,
    storage_manager: Optional[Any] = None,
    encryption_enabled: Optional[bool] = None,
) -> Tuple[bool, str]:
    """
    Saves screenshot evidence and appends a structured incident to the event log.
    If encryption is enabled, encrypts directly in memory and persists an envelope (.png.enc).
    Never writes plaintext to disk when encryption is enabled (fails closed).

    Returns:
        Tuple[bool, str]: (screenshot_written_successfully, evidence_reference_path)
    """
    if encryption_enabled is None:
        encryption_enabled = bool(getattr(config, "ENCRYPTION_ENABLED", False))

    clean_prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(prefix).strip()).strip("_")
    if not clean_prefix:
        clean_prefix = "violation"
    screenshot_filename = f"{clean_prefix}_{int(time.time())}.png"

    write_success = False
    saved_ref = ""

    if encryption_enabled:
        # Encrypted Evidence Workflow (Phase 10.13.28.3)
        try:
            enc_filename = f"{screenshot_filename}.enc"
            target_path = resolve_safe_output_path(
                log_dir,
                enc_filename,
                allow_extensions={".enc"},
            )
            # Retrieve or reuse storage manager
            if storage_manager is None:
                if logger is not None and getattr(logger, "storage_manager", None) is not None:
                    storage_manager = logger.storage_manager
                else:
                    from encrypted_storage import get_encrypted_storage_manager
                    storage_manager = get_encrypted_storage_manager()

            # Encode frame in memory directly to PNG bytes (no plaintext disk writes)
            encode_ok, buf = cv2.imencode(".png", frame)
            if not encode_ok:
                raise ValueError(f"cv2.imencode failed for {violation_type}")
            plaintext_bytes = buf.tobytes()

            from datetime import datetime, timezone
            import hashlib
            import uuid
            from encrypted_storage import EvidenceMetadata
            now_dt = datetime.now(timezone.utc)
            meta = EvidenceMetadata(
                evidence_id=str(uuid.uuid4()),
                original_filename=screenshot_filename,
                evidence_type="SCREENSHOT",
                created_at=now_dt.isoformat(),
                created_timestamp=now_dt.timestamp(),
                session_id=logger.session_id if logger else None,
                custom_attributes={
                    "violation_type": violation_type,
                    "duration_sec": float(duration_sec),
                    "file_sha256": hashlib.sha256(plaintext_bytes).hexdigest(),
                },
            )
            storage_manager.write_encrypted_file(
                destination_path=target_path,
                plaintext=plaintext_bytes,
                metadata=meta,
                allow_overwrite=False,
            )
            try:
                saved_ref = os.path.relpath(target_path, config.BASE_DIR).replace("\\", "/")
            except ValueError:
                saved_ref = str(target_path)
            write_success = True
            print(f"Encrypted screenshot saved ({violation_type}): {saved_ref}")
        except Exception as e:
            print(f"Error: Failed to write encrypted screenshot for {violation_type}: {e}")
            write_success = False
            saved_ref = ""
    else:
        # Unencrypted Legacy Workflow
        try:
            target_path = resolve_safe_output_path(
                log_dir,
                screenshot_filename,
                allow_extensions={".png"},
            )
            filename = str(target_path)
        except (ValueError, TypeError) as exc:
            print(f"Warning: Rejected invalid screenshot path for {violation_type}: {exc}")
            filename = ""

        try:
            if filename:
                write_success = bool(cv2.imwrite(filename, frame))
        except Exception as e:
            print(f"Warning: Exception while writing screenshot to {filename}: {e}")
            write_success = False

        if write_success:
            try:
                saved_ref = os.path.relpath(filename, config.BASE_DIR).replace("\\", "/")
            except ValueError:
                saved_ref = filename
            print(f"Screenshot saved ({violation_type}): {saved_ref}")
        else:
            print(f"Warning: Failed to save screenshot for {violation_type} to {filename}")

    # Append to structured event log
    if logger is not None:
        try:
            logger.log_event(
                violation_type=violation_type,
                duration_sec=duration_sec,
                confidence=confidence,
                screenshot_path=saved_ref if write_success else "",
            )
        except Exception as e:
            print(f"Warning: Exception while logging event for {violation_type}: {e}")

    return write_success, saved_ref


def run_surveillance(
    cap=None,
    logger: Optional[EventLogger] = None,
    max_frames: Optional[int] = None,
    display: bool = True,
    video_source: Optional[str] = None,
    camera_index: Optional[int] = None,
    detection_interval: Optional[int] = None,
    cache_expiry: Optional[int] = None,
    telemetry: Optional[PerformanceTelemetry] = None,
    time_source: Optional[TimeSource] = None,
    gaze_logger: Optional[Any] = None,
    calibrate_screen: bool = False,
    screen_calibrator: Optional[ScreenGazeCalibrator] = None,
):
    """
    Main application runtime loop protected by resource cleanup.
    """
    active_logger = None
    active_gaze_logger = None
    active_telemetry = None
    screen_calibration_mgr = None
    calibration_window_active = False
    cleaned_up = False
    frame_count = 0
    current_time = 0.0

    def _cleanup_resources():
        nonlocal cleaned_up, calibration_window_active
        if cleaned_up:
            return
        cleaned_up = True

        if screen_calibration_mgr is not None and getattr(screen_calibration_mgr, "state", None) == CALIB_STATE_CALIBRATING:
            try:
                screen_calibration_mgr.cancel()
                print("Screen calibration cancelled during shutdown; any previous calibration was preserved.")
            except Exception as e:
                print(f"Warning: Exception while cancelling screen calibration: {e}")

        if active_gaze_logger is not None:
            try:
                active_gaze_logger.close()
            except Exception as e:
                print(f"Warning: Exception while closing gaze diagnostics logger: {e}")

        if active_logger is not None:
            try:
                if hasattr(active_logger, "flush"):
                    active_logger.flush()
                if hasattr(active_logger, "close"):
                    active_logger.close()
            except Exception as e:
                print(f"Warning: Exception while closing event logger: {e}")

        reset_default_gaze_tracker()
        reset_head_pose_state()

        if cap is not None:
            try:
                if hasattr(cap, "release"):
                    cap.release()
            except Exception as e:
                print(f"Warning: Exception while releasing camera: {e}")

        if display:
            if calibration_window_active:
                try:
                    cv2.destroyWindow(WINDOW_CALIBRATION_NAME)
                except Exception as cleanup_exc:
                    print(f"Warning: calibration window cleanup failed during shutdown: {cleanup_exc}")
                calibration_window_active = False
            try:
                cv2.destroyAllWindows()
            except Exception as cleanup_exc:
                print(f"Warning: display window cleanup failed during shutdown: {cleanup_exc}")

    try:
        # Validate configuration and model paths at startup
        config.validate_config()

        # Reset tracking history for new session (GAP-05)
        reset_default_gaze_tracker()
        reset_head_pose_state()

        # Directory for violation screenshots
        log_dir = str(config.LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize event logger
        active_logger = logger or get_event_logger(log_dir=log_dir)

        # Initialize gaze diagnostics logger (temporary instrumentation)
        active_gaze_logger = gaze_logger
        if active_gaze_logger is None:
            try:
                active_gaze_logger = GazeDiagnosticsLogger(
                    log_dir=log_dir,
                    session_id=active_logger.session_id if active_logger else None,
                )
            except Exception as e:
                print(f"Warning: Could not initialize gaze diagnostics logger: {e}")
                active_gaze_logger = None
        elif active_gaze_logger is False:
            active_gaze_logger = None

        # Initialize video capture if not provided
        own_cap = False
        if cap is None:
            if video_source is not None:
                vpath = Path(video_source)
                if not vpath.is_file():
                    raise FileNotFoundError(f"Video file not found at: '{video_source}'")
                cap = cv2.VideoCapture(str(vpath))
                own_cap = True
                if not cap.isOpened():
                    print(f"Error: Could not open video file at: '{video_source}'.")
                    _cleanup_resources()
                    return 1
            else:
                cam_idx = camera_index if camera_index is not None else config.CAMERA_INDEX
                if cam_idx < 0:
                    raise ValueError(f"Camera index must be non-negative, got {cam_idx}")
                cap = cv2.VideoCapture(cam_idx)
                own_cap = True
                if not cap.isOpened():
                    print(f"Warning: Could not open webcam at index {cam_idx}. Check video device connection.")
                    _cleanup_resources()
                    return 1
        elif hasattr(cap, "isOpened") and not cap.isOpened():
            print("Warning: Provided video capture device is not open.")
            _cleanup_resources()
            return 1

        # Setup time source (determines elapsed duration for calibration and violations)
        active_time_source = time_source
        if active_time_source is None:
            if video_source is not None:
                active_time_source = VideoTimeSource(cap=cap, fallback_fps=config.DEFAULT_VIDEO_FPS)
            elif cap is not None and hasattr(cap, "get"):
                try:
                    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if isinstance(fc, (int, float)) and fc > 0:
                        active_time_source = VideoTimeSource(cap=cap, fallback_fps=config.DEFAULT_VIDEO_FPS)
                    else:
                        active_time_source = LiveTimeSource()
                except Exception:
                    active_time_source = LiveTimeSource()
            else:
                active_time_source = LiveTimeSource()

        # Authoritatively establish effective camera capture geometry (GAP-03)
        effective_w, effective_h = resolve_capture_geometry(
            cap,
            requested_width=config.FRAME_WIDTH,
            requested_height=config.FRAME_HEIGHT,
        )

        # Active screen calibrator and interactive manager setup
        active_screen_calibrator = screen_calibrator
        screen_camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        screen_display_size = get_primary_display_resolution(strict=True)
        screen_calibration_mgr = None
        screen_recalibration_requested = calibrate_screen

        if effective_w is None or effective_h is None:
            # Effective capture geometry could not be established; reject profile activation (GAP-03)
            if active_screen_calibrator is not None and active_screen_calibrator.is_calibrated:
                active_screen_calibrator = None
            print("Screen calibration unavailable (effective_capture_geometry_unavailable); screen-relative gaze requires recalibration.")
        else:
            screen_environment = screen_calibration_environment(
                source_type="video" if video_source is not None else "webcam",
                camera_index=screen_camera_index if video_source is None else None,
                frame_width=effective_w,
                frame_height=effective_h,
            )
            if active_screen_calibrator is None and video_source is None:
                active_screen_calibrator, load_reason = load_screen_calibration(
                    config.SCREEN_CALIBRATION_PATH,
                    display_size=screen_display_size,
                    environment=screen_environment,
                )
                if load_reason not in ("", "calibration_file_missing"):
                    print(f"Screen calibration unavailable ({load_reason}); screen-relative gaze requires recalibration.")

            if calibrate_screen and (not display or video_source is not None):
                print("Screen calibration not started: interactive calibration requires a displayed webcam session.")
                calibrate_screen = False
                screen_recalibration_requested = False

            if calibrate_screen or active_screen_calibrator is not None:
                screen_manager_options = {
                    "calibration_path": config.SCREEN_CALIBRATION_PATH,
                    "environment": screen_environment,
                    "min_dwell_sec": config.SCREEN_CALIBRATION_MIN_DWELL_SEC,
                    "stability_window_sec": config.SCREEN_CALIBRATION_STABILITY_WINDOW_SEC,
                    "stability_threshold_deg": config.SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG,
                    "min_sample_interval_sec": config.SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC,
                    "time_source": active_time_source,
                }
                if screen_display_size is not None:
                    screen_manager_options["display_size"] = screen_display_size
                screen_calibration_mgr = ScreenCalibrationManager(
                    calibrator=active_screen_calibrator,
                    **screen_manager_options,
                )
                if active_screen_calibrator is not None and active_screen_calibrator.is_calibrated:
                    screen_calibration_mgr.state = CALIB_STATE_VALID

        # Setup telemetry and mobile detection tracker
        reset_default_gaze_tracker()
        active_telemetry = telemetry or PerformanceTelemetry()

        det_interval = (
            detection_interval if detection_interval is not None else config.PHONE_DETECTION_INTERVAL_FRAMES
        )
        det_expiry = (
            cache_expiry if cache_expiry is not None else config.PHONE_DETECTION_CACHE_EXPIRY_FRAMES
        )
        phone_tracker = PhoneDetectionTracker(
            interval_frames=det_interval,
            cache_expiry_frames=det_expiry,
            conf_threshold=config.PHONE_CONFIDENCE_THRESHOLD,
        )

        # Calibration parameters for head pose and gaze
        CALIBRATION_DURATION = config.CALIBRATION_DURATION_SEC
        MIN_CALIBRATION_SAMPLES = config.MIN_CALIBRATION_SAMPLES
        MIN_GAZE_CALIBRATION_SAMPLES = config.MIN_GAZE_CALIBRATION_SAMPLES
        MAX_CALIBRATION_TIMEOUT_SEC = 15.0

        calibration_samples = []
        gaze_samples = []
        calibrated_angles = None
        calibrated_gaze = None
        calibration_start_time = active_time_source.now()
        is_calibrated = False

        # Violation duration threshold
        VIOLATION_THRESHOLD_SEC = config.VIOLATION_THRESHOLD_SEC

        # Timers and incident-logged state for sustained violations
        head_misalignment_start_time = None
        head_misalignment_logged = False

        eye_misalignment_start_time = None
        eye_misalignment_logged = False

        uncertain_gaze_start_time = None
        uncertain_gaze_logged = False
        uncertain_gaze_frames = 0

        mobile_detection_start_time = None
        mobile_detection_logged = False

        face_missing_start_time = None
        face_missing_logged = False

        multi_face_start_time = None
        multi_face_logged = False

        outside_screen_start_time = None
        outside_screen_logged = False

        # Initialize default states
        head_direction = "Calibrating"
        gaze_direction = GAZE_CENTER

        frame_count = 0
        current_time = active_time_source.now()

        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            t_start = active_telemetry.start_frame()
            ret, frame = cap.read()
            if not ret:
                print("Camera feed unavailable or ended. Exiting loop.")
                break

            # Preserve pristine, unannotated source frame for object detection
            clean_frame = frame.copy()
            display_frame = frame

            # Authoritatively verify effective capture geometry against delivered frames (GAP-03)
            if frame is not None and frame.size > 0:
                frame_h, frame_w = frame.shape[:2]
                if effective_w is None or effective_h is None:
                    effective_w, effective_h = frame_w, frame_h
                elif (frame_w != effective_w or frame_h != effective_h):
                    if active_screen_calibrator is not None:
                        print(f"Warning: Delivered frame dimensions ({frame_w}x{frame_h}) differ from calibrated capture geometry ({effective_w}x{effective_h}); deactivating screen calibration.")
                        active_screen_calibrator = None

            # Synchronize and advance elapsed stream/camera time for this frame
            current_time = active_time_source.update(frame=clean_frame, cap=cap)
            frame_count += 1

            # 1. Process unified face analysis (single face detection + single landmark extraction)
            face_result, display_frame = process_frame_face(
                display_frame,
                calibrated_angles=calibrated_angles,
                calibrated_gaze=calibrated_gaze,
                screen_calibrator=active_screen_calibrator,
            )
            gaze_direction = face_result.gaze_direction
            combined_gaze = face_result.combined_gaze
            attention_direction = (
                classify_combined_attention(combined_gaze) if is_calibrated else GAZE_UNCERTAIN
            )

            # 1b. Interactive screen calibration step (runs after head/eye baseline calibration)
            if is_calibrated and screen_calibration_mgr is not None:
                if (
                    screen_calibration_mgr.state == CALIB_STATE_NOT_CALIBRATED
                    or screen_recalibration_requested
                ):
                    screen_calibration_mgr.start()
                    screen_recalibration_requested = False
                    if screen_calibration_mgr.state == CALIB_STATE_FAILED:
                        print(f"Screen calibration unavailable: {screen_calibration_mgr.failure_reason}")
                elif screen_calibration_mgr.state == CALIB_STATE_CALIBRATING:
                    try:
                        screen_calibration_mgr.process_sample(
                            face_result.combined_gaze,
                            face_result=face_result,
                            timestamp=current_time,
                        )
                    except Exception as exc:
                        screen_calibration_mgr.fail(f"calibration_processing_failed: {exc}")
                    if screen_calibration_mgr.state == CALIB_STATE_VALID:
                        active_screen_calibrator = screen_calibration_mgr.calibrator
                        q = active_screen_calibrator.quality_metrics
                        print(
                            f"Screen calibration successful: targets={q.get('target_count')}, "
                            f"RMSE={q.get('rmse', 0):.4f}, score={q.get('quality_score', 0):.2f}"
                        )
                        # Re-estimate screen gaze for the current frame with newly fitted calibrator
                        if face_result.combined_gaze is not None:
                            face_result.screen_gaze = active_screen_calibrator.estimate_screen_gaze(
                                gaze_yaw=face_result.combined_gaze.yaw,
                                gaze_pitch=face_result.combined_gaze.pitch,
                                gaze_confidence=face_result.combined_gaze.confidence,
                                gaze_status=face_result.combined_gaze.status,
                            )
                    elif screen_calibration_mgr.state == CALIB_STATE_FAILED:
                        print(f"Screen calibration failed: {screen_calibration_mgr.failure_reason}")

            # Log raw gaze diagnostic measurement for this frame
            if active_gaze_logger is not None:
                try:
                    active_gaze_logger.log_frame(
                        frame_index=frame_count,
                        face_result=face_result,
                        is_calibrated=is_calibrated,
                        calibrated_gaze=calibrated_gaze,
                        current_time=current_time,
                    )
                except Exception as e:
                    print(f"Warning: Exception logging gaze diagnostics: {e}")

            if display:
                if gaze_direction in COMPLIANT_GAZE_STATES:
                    gaze_color = (0, 255, 0)
                elif gaze_direction == GAZE_UNCERTAIN:
                    gaze_color = (0, 255, 255)
                else:
                    gaze_color = (0, 0, 255)

                cv2.putText(
                    display_frame,
                    f"Gaze Direction: {gaze_direction}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    gaze_color,
                    2,
                )
                fused_text, fused_confidence_text = combined_gaze_hud(combined_gaze)
                cv2.putText(
                    display_frame, fused_text, (350, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 0), 1,
                )
                cv2.putText(
                    display_frame, fused_confidence_text, (350, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 0), 1,
                )
                head_pose = face_result.head_pose
                head_detail = (
                    f"Head {head_pose.status}: y {_format_angle(head_pose.yaw)} "
                    f"p {_format_angle(head_pose.pitch)} r {_format_angle(head_pose.roll)} "
                    f"{head_pose.confidence * 100:.0f}%"
                    if head_pose is not None else "Head: UNAVAILABLE"
                )
                eye_gaze = face_result.eye_gaze
                eye_detail = (
                    f"Eye {eye_gaze.status}: {eye_gaze.selected_eye_direction} "
                    f"{eye_gaze.confidence * 100:.0f}% ({eye_gaze.tracking_state})"
                    if eye_gaze is not None else "Eye: UNAVAILABLE"
                )
                cv2.putText(
                    display_frame, head_detail, (350, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1,
                )
                cv2.putText(
                    display_frame, eye_detail, (350, 103), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1,
                )
                if screen_calibration_mgr is not None and screen_calibration_mgr.state == CALIB_STATE_CALIBRATING:
                    display_frame = screen_calibration_mgr.draw_overlay(display_frame)
                else:
                    sg_line, sg_sub = screen_gaze_hud(face_result.screen_gaze)
                    sg_color = (0, 255, 0)
                    if face_result.screen_gaze:
                        if face_result.screen_gaze.boundary_state == BOUNDARY_OUTSIDE_SCREEN:
                            sg_color = (0, 0, 255)
                        elif face_result.screen_gaze.boundary_state == BOUNDARY_NEAR_BOUNDARY:
                            sg_color = (0, 255, 255)
                        elif face_result.screen_gaze.boundary_state in (BOUNDARY_UNCERTAIN, BOUNDARY_CALIBRATION_REQUIRED):
                            sg_color = (180, 180, 180)
                    cv2.putText(
                        display_frame, sg_line, (350, 126), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, sg_color, 1,
                    )
                    cv2.putText(
                        display_frame, sg_sub, (350, 149), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (200, 200, 200), 1,
                    )

            # 2. Process head pose & calibration
            if not is_calibrated:
                elapsed = current_time - calibration_start_time
                if face_result.head_pose_sample is not None:
                    calibration_samples.append(face_result.head_pose_sample)
                gaze_quality = face_result.gaze_diagnostics or {}
                if (
                    face_result.gaze_sample is not None
                    and gaze_quality.get("left_pupil_valid") is True
                    and gaze_quality.get("right_pupil_valid") is True
                ):
                    gaze_samples.append(face_result.gaze_sample)

                has_min_samples = (
                    len(calibration_samples) >= MIN_CALIBRATION_SAMPLES
                    and len(gaze_samples) >= MIN_GAZE_CALIBRATION_SAMPLES
                )

                if elapsed >= CALIBRATION_DURATION and has_min_samples:
                    avg_pitch = float(np.mean([s[0] for s in calibration_samples]))
                    avg_yaw = float(np.mean([s[1] for s in calibration_samples]))
                    avg_roll = float(np.mean([s[2] for s in calibration_samples]))
                    candidate_angles = (avg_pitch, avg_yaw, avg_roll)
                    candidate_gaze, quality = calibrate_gaze_baseline(
                        gaze_samples,
                        min_samples=MIN_GAZE_CALIBRATION_SAMPLES,
                        return_diagnostics=True,
                    )
                    if candidate_gaze is not None:
                        calibrated_angles = candidate_angles
                        calibrated_gaze = candidate_gaze
                        is_calibrated = True
                        print(
                            f"Calibration successful: head={len(calibration_samples)} samples, "
                            f"gaze={len(gaze_samples)} samples (pitch={avg_pitch:.1f}, yaw={avg_yaw:.1f})"
                        )
                    else:
                        reason = quality["failure_reason"]
                        print(f"Calibration rejected: {reason}")
                        calibration_start_time = current_time
                        calibration_samples.clear()
                        gaze_samples.clear()
                        if display:
                            cv2.putText(
                                display_frame,
                                f"Calibration rejected: {reason[:72]}",
                                (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )
                elif elapsed >= MAX_CALIBRATION_TIMEOUT_SEC:
                    # Maximum timeout expired without sufficient samples; reset accumulation window
                    reason = (
                        f"calibration timed out after {MAX_CALIBRATION_TIMEOUT_SEC:.1f}s: "
                        f"insufficient valid samples (head {len(calibration_samples)}/"
                        f"{MIN_CALIBRATION_SAMPLES}, gaze {len(gaze_samples)}/"
                        f"{MIN_GAZE_CALIBRATION_SAMPLES})"
                    )
                    print(f"Calibration rejected: {reason}")
                    calibration_start_time = current_time
                    calibration_samples.clear()
                    gaze_samples.clear()
                    if display:
                        cv2.putText(
                            display_frame,
                            f"Calibration rejected: {reason[:72]}",
                            (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                else:
                    if display:
                        if elapsed < CALIBRATION_DURATION:
                            remaining = max(1, int(CALIBRATION_DURATION - elapsed))
                            msg = f"Calibrating... Keep head and gaze centered ({remaining}s, head:{len(calibration_samples)}, gaze:{len(gaze_samples)})"
                        else:
                            remaining = max(1, int(MAX_CALIBRATION_TIMEOUT_SEC - elapsed))
                            msg = f"Calibrating... Accumulating samples ({remaining}s left, head:{len(calibration_samples)}, gaze:{len(gaze_samples)})"
                        cv2.putText(
                            display_frame,
                            msg,
                            (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2,
                        )

                head_direction = "Calibrating"
            else:
                head_direction = face_result.head_direction
                if display:
                    head_color = (
                        (0, 255, 0) if head_direction in COMPLIANT_HEAD_STATES else (0, 0, 255)
                    )
                    cv2.putText(
                        display_frame,
                        f"Head Direction: {head_direction}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        head_color,
                        2,
                    )

            # 3. Process mobile detection with inference cadence & caching on clean frame
            display_frame, mobile_detected, phone_result = phone_tracker.process_frame(
                clean_frame, annotate=display, display_frame=display_frame
            )

            # Telemetry measurement (compute duration per frame)
            frame_compute_sec = time.perf_counter() - t_start
            is_yolo_run = phone_result.state in (
                DETECTION_STATE_FRESH_POSITIVE,
                DETECTION_STATE_FRESH_NEGATIVE,
            )
            active_telemetry.record_frame(
                processing_duration_sec=frame_compute_sec,
                is_yolo_inference=is_yolo_run,
                is_phone_positive=phone_result.detected,
            )

            if display:
                if phone_result.state in (
                    DETECTION_STATE_FRESH_POSITIVE,
                    DETECTION_STATE_CACHED_POSITIVE,
                ):
                    tag = "cached" if phone_result.state == DETECTION_STATE_CACHED_POSITIVE else "fresh"
                    mobile_text = f"Mobile Detected: True ({tag})"
                    mobile_color = (0, 0, 255)
                elif phone_result.state == DETECTION_STATE_INFERENCE_ERROR:
                    mobile_text = "Mobile Detected: Error"
                    mobile_color = (0, 255, 255)
                elif phone_result.state == DETECTION_STATE_EXPIRED:
                    mobile_text = "Mobile Detected: Expired"
                    mobile_color = (0, 255, 255)
                else:
                    mobile_text = "Mobile Detected: False"
                    mobile_color = (0, 255, 0)

                cv2.putText(
                    display_frame,
                    mobile_text,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    mobile_color,
                    2,
                )

            # 4. Handle multiple faces explicitly (timed violation)
            if face_result.is_multi_face:
                if display:
                    cv2.putText(
                        display_frame,
                        f"WARNING: Multiple Faces Detected ({face_result.face_count})!",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                if multi_face_start_time is None:
                    multi_face_start_time = current_time
                    multi_face_logged = False
                elif not multi_face_logged and (
                    current_time - multi_face_start_time >= VIOLATION_THRESHOLD_SEC
                ):
                    duration = current_time - multi_face_start_time
                    record_violation_incident(
                        violation_type="multiple_faces",
                        prefix="multi_face",
                        frame=display_frame,
                        duration_sec=duration,
                        confidence=None,
                        log_dir=log_dir,
                        logger=active_logger,
                    )
                    multi_face_logged = True
            else:
                multi_face_start_time = None
                multi_face_logged = False

            # 5. Check for face absence safely
            is_face_missing = (
                head_direction == HEAD_NO_FACE or gaze_direction == GAZE_NO_FACE
            )
            if is_calibrated and is_face_missing:
                if display:
                    cv2.putText(
                        display_frame,
                        "WARNING: Candidate Not Detected!",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                if face_missing_start_time is None:
                    face_missing_start_time = current_time
                    face_missing_logged = False
                elif not face_missing_logged and (
                    current_time - face_missing_start_time >= VIOLATION_THRESHOLD_SEC
                ):
                    duration = current_time - face_missing_start_time
                    record_violation_incident(
                        violation_type="missing_face",
                        prefix="missing_face",
                        frame=display_frame,
                        duration_sec=duration,
                        confidence=None,
                        log_dir=log_dir,
                        logger=active_logger,
                    )
                    face_missing_logged = True
            else:
                face_missing_start_time = None
                face_missing_logged = False

            # 6. Check for independent head misalignment (only if face is present and calibrated)
            # Under approved Option A architecture, head movement is monitored independently
            # and generates its own violation events (head_Looking_Left, head_Looking_Right,
            # head_Looking_Up, head_Looking_Down, head_Tilted).
            # Hardened head validity (Phase 10.13.9.2): missing head pose is strictly rejected.
            is_head_valid = (
                face_result.head_pose is not None
                and getattr(face_result.head_pose, "status", None) == "VALID"
            )
            if (
                is_calibrated
                and is_head_valid
                and head_direction
                in (
                    HEAD_LEFT,
                    HEAD_RIGHT,
                    HEAD_UP,
                    HEAD_DOWN,
                    HEAD_TILTED,
                )
            ):
                if head_misalignment_start_time is None:
                    head_misalignment_start_time = current_time
                    head_misalignment_logged = False
                elif not head_misalignment_logged and (
                    current_time - head_misalignment_start_time >= VIOLATION_THRESHOLD_SEC
                ):
                    duration = current_time - head_misalignment_start_time
                    clean_name = head_direction.replace(" ", "_")
                    head_conf = (
                        face_result.head_pose.confidence
                        if (face_result.head_pose and face_result.head_pose.confidence is not None)
                        else None
                    )
                    record_violation_incident(
                        violation_type=f"head_{clean_name}",
                        prefix=f"head_{clean_name}",
                        frame=display_frame,
                        duration_sec=duration,
                        confidence=head_conf,
                        log_dir=log_dir,
                        logger=active_logger,
                    )
                    head_misalignment_logged = True
            else:
                head_misalignment_start_time = None
                head_misalignment_logged = False

            # 7. Check for fused directional visual attention or occluded gaze (only if face is present and calibrated)
            # Fused head-eye gaze separately determines overall camera-relative visual attention.
            # Directional events (eye_Looking_Left, eye_Looking_Right, eye_Looking_Up, eye_Looking_Down)
            # are generated from fused overall camera-relative gaze rather than pupil-only heuristics.
            if is_calibrated and attention_direction in (
                GAZE_LEFT,
                GAZE_RIGHT,
                GAZE_UP,
                GAZE_DOWN,
            ):
                uncertain_gaze_start_time = None
                uncertain_gaze_logged = False
                uncertain_gaze_frames = 0
                if eye_misalignment_start_time is None:
                    eye_misalignment_start_time = current_time
                    eye_misalignment_logged = False
                elif not eye_misalignment_logged and (
                    current_time - eye_misalignment_start_time >= VIOLATION_THRESHOLD_SEC
                ):
                    duration = current_time - eye_misalignment_start_time
                    clean_name = attention_direction.replace(" ", "_")
                    record_violation_incident(
                        violation_type=f"eye_{clean_name}",
                        prefix=f"eye_{clean_name}",
                        frame=display_frame,
                        duration_sec=duration,
                        confidence=None,
                        log_dir=log_dir,
                        logger=active_logger,
                    )
                    eye_misalignment_logged = True
            elif is_calibrated and attention_direction == GAZE_UNCERTAIN:
                # Tolerate brief micro-blinks (up to 3 consecutive frames) without wiping deviation timer
                uncertain_gaze_frames += 1
                if uncertain_gaze_frames > 3:
                    eye_misalignment_start_time = None
                    eye_misalignment_logged = False

                # A single-eye failure is insufficient evidence of sustained occlusion.
                if gaze_occlusion_is_confirmed(face_result.gaze_diagnostics):
                    if uncertain_gaze_start_time is None:
                        uncertain_gaze_start_time = current_time
                        uncertain_gaze_logged = False
                    elif not uncertain_gaze_logged and (
                        current_time - uncertain_gaze_start_time >= VIOLATION_THRESHOLD_SEC
                    ):
                        duration = current_time - uncertain_gaze_start_time
                        record_violation_incident(
                            violation_type="gaze_occluded",
                            prefix="gaze_occluded",
                            frame=display_frame,
                            duration_sec=duration,
                            confidence=None,
                            log_dir=log_dir,
                            logger=active_logger,
                        )
                        uncertain_gaze_logged = True
                else:
                    uncertain_gaze_start_time = None
                    uncertain_gaze_logged = False
            else:
                eye_misalignment_start_time = None
                eye_misalignment_logged = False
                uncertain_gaze_start_time = None
                uncertain_gaze_logged = False
                uncertain_gaze_frames = 0

            # 8. Check for sustained outside-screen gaze violation (only if screen is calibrated)
            screen_gaze = getattr(face_result, "screen_gaze", None)
            if is_confirmed_outside_screen(screen_gaze):
                if outside_screen_start_time is None:
                    outside_screen_start_time = current_time
                    outside_screen_logged = False
                elif not outside_screen_logged and (
                    current_time - outside_screen_start_time >= VIOLATION_THRESHOLD_SEC
                ):
                    duration = current_time - outside_screen_start_time
                    record_violation_incident(
                        violation_type="gaze_outside_screen",
                        prefix="gaze_outside_screen",
                        frame=display_frame,
                        duration_sec=duration,
                        confidence=getattr(screen_gaze, "confidence", None),
                        log_dir=log_dir,
                        logger=active_logger,
                    )
                    outside_screen_logged = True
            else:
                outside_screen_start_time = None
                outside_screen_logged = False

            # 9. Check for mobile detection with cadence and expiry
            if phone_result.state in (
                DETECTION_STATE_FRESH_POSITIVE,
                DETECTION_STATE_CACHED_POSITIVE,
            ):
                if mobile_detection_start_time is None:
                    mobile_detection_start_time = current_time
                    mobile_detection_logged = False
                elif not mobile_detection_logged and (
                    current_time - mobile_detection_start_time >= VIOLATION_THRESHOLD_SEC
                ):
                    duration = current_time - mobile_detection_start_time
                    phone_conf = phone_result.confidence or get_last_detected_confidence()
                    record_violation_incident(
                        violation_type="mobile_detected",
                        prefix="mobile_detected",
                        frame=display_frame,
                        duration_sec=duration,
                        confidence=phone_conf,
                        log_dir=log_dir,
                        logger=active_logger,
                    )
                    mobile_detection_logged = True
            elif phone_result.state in (
                DETECTION_STATE_FRESH_NEGATIVE,
                DETECTION_STATE_CACHED_NEGATIVE,
            ):
                mobile_detection_start_time = None
                mobile_detection_logged = False
            elif phone_result.state == DETECTION_STATE_EXPIRED:
                # Expired/stale results cannot count as confirmed presence
                mobile_detection_start_time = None
                mobile_detection_logged = False
            elif phone_result.state == DETECTION_STATE_INFERENCE_ERROR:
                # Transient inference failure: do not reset immediately (not proof of absence),
                # but do not trigger violation
                pass

            # 9. Frame presentation or headless telemetry
            if display:
                # Screen calibration dedicated fullscreen window management (Phase 10.13.9.2)
                if (
                    screen_calibration_mgr is not None
                    and screen_calibration_mgr.state == CALIB_STATE_CALIBRATING
                ):
                    if not calibration_window_active:
                        try:
                            cv2.namedWindow(WINDOW_CALIBRATION_NAME, cv2.WINDOW_NORMAL)
                            cv2.setWindowProperty(
                                WINDOW_CALIBRATION_NAME,
                                cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN,
                            )
                            calibration_window_active = True
                        except Exception as exc:
                            screen_calibration_mgr.fail(f"calibration_window_initialization_failed: {exc}")
                            print(f"Screen calibration failed: {screen_calibration_mgr.failure_reason}")
                            try:
                                cv2.destroyWindow(WINDOW_CALIBRATION_NAME)
                            except Exception as cleanup_exc:
                                print(f"Warning: calibration window cleanup failed: {cleanup_exc}")
                            calibration_window_active = False
                    if screen_calibration_mgr.state == CALIB_STATE_CALIBRATING:
                        try:
                            calib_canvas = screen_calibration_mgr.render_calibration_canvas(
                                webcam_preview=display_frame
                            )
                            cv2.imshow(WINDOW_CALIBRATION_NAME, calib_canvas)
                        except Exception as exc:
                            screen_calibration_mgr.fail(f"calibration_display_failed: {exc}")
                            print(f"Screen calibration failed: {screen_calibration_mgr.failure_reason}")
                            if calibration_window_active:
                                try:
                                    cv2.destroyWindow(WINDOW_CALIBRATION_NAME)
                                except Exception as cleanup_exc:
                                    print(f"Warning: calibration window cleanup failed: {cleanup_exc}")
                                calibration_window_active = False
                else:
                    if calibration_window_active:
                        try:
                            cv2.destroyWindow(WINDOW_CALIBRATION_NAME)
                        except Exception as cleanup_exc:
                            print(f"Warning: calibration window cleanup failed: {cleanup_exc}")
                        calibration_window_active = False

                # Telemetry HUD overlay in display window
                h = display_frame.shape[0] if display_frame is not None else 480
                overlay_text = f"Time: {current_time:.1f}s | {active_telemetry.format_overlay()}"
                cv2.putText(
                    display_frame,
                    overlay_text,
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Combined Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if (
                        screen_calibration_mgr is not None
                        and screen_calibration_mgr.state == CALIB_STATE_CALIBRATING
                    ):
                        screen_calibration_mgr.cancel()
                        if calibration_window_active:
                            try:
                                cv2.destroyWindow(WINDOW_CALIBRATION_NAME)
                            except Exception as cleanup_exc:
                                print(f"Warning: calibration window cleanup failed: {cleanup_exc}")
                            calibration_window_active = False
                    else:
                        break
                elif key == ord('c'):
                    if video_source is not None:
                        print("Screen calibration is unavailable for prerecorded video sources.")
                    elif screen_calibration_mgr is None:
                        screen_calibration_mgr = ScreenCalibrationManager(
                            calibrator=active_screen_calibrator,
                            **screen_manager_options,
                        )
                    if video_source is None:
                        screen_calibration_mgr.start()
            else:
                # Periodic console telemetry in headless mode
                if active_telemetry.should_log_console():
                    print(f"[Stream Time: {current_time:.1f}s] " + active_telemetry.format_console_log())

        return 0
    finally:
        _cleanup_resources()
        if active_telemetry is not None and frame_count > 0:
            summary = active_telemetry.get_summary()
            print(
                f"Surveillance session completed: {summary['total_frames']} frames processed "
                f"({current_time:.1f}s elapsed stream time), "
                f"{summary['yolo_inferences']} YOLO inferences, "
                f"avg compute latency: {summary['avg_latency_ms']:.1f}ms, "
                f"effective FPS: {summary['fps']:.1f}"
            )


def main(argv=None):
    """Entry point for CLI execution."""
    args = parse_args(argv)
    try:
        validate_cli_args(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if getattr(args, "retention_audit", False):
        from evidence_retention import EvidenceRetentionManager
        mgr = EvidenceRetentionManager(
            evidence_dir=config.LOG_DIR,
            retention_days=config.RETENTION_DAYS,
            enabled=config.RETENTION_ENABLED,
        )
        audit_res = mgr.audit()
        print(f"[Retention Audit] Evidence directory: {audit_res.evidence_dir}")
        print(f"[Retention Audit] Retention threshold: {audit_res.retention_days} days")
        print(f"[Retention Audit] Total files scanned: {audit_res.total_files_scanned} ({audit_res.total_bytes_scanned / 1024 / 1024:.2f} MB)")
        print(f"[Retention Audit] Active evidence files: {len(audit_res.active_files)}")
        print(f"[Retention Audit] Expired evidence files: {len(audit_res.expired_files)}")
        print(f"[Retention Audit] Protected files: {len(audit_res.protected_files)}")
        if audit_res.skipped_suspicious_files:
            print(f"[Retention Audit] Skipped / suspicious files: {len(audit_res.skipped_suspicious_files)}")
        return 0

    if getattr(args, "retention_purge", False):
        from evidence_retention import EvidenceRetentionManager
        dry_run = not getattr(args, "no_dry_run", False)
        mgr = EvidenceRetentionManager(
            evidence_dir=config.LOG_DIR,
            retention_days=config.RETENTION_DAYS,
            enabled=config.RETENTION_ENABLED or not dry_run,
        )
        purge_res = mgr.purge_expired(dry_run=dry_run, force=not dry_run)
        action_str = "DRY-RUN (no files deleted)" if purge_res.dry_run else "EXECUTED"
        print(f"[Retention Purge] Mode: {action_str}")
        print(f"[Retention Purge] Files expired/deleted: {len(purge_res.files_deleted)}")
        print(f"[Retention Purge] Storage reclaimed: {purge_res.bytes_reclaimed / 1024 / 1024:.2f} MB")
        if purge_res.failures:
            print(f"[Retention Purge] Failures encountered: {len(purge_res.failures)}")
        return 0

    status = run_surveillance(
        display=not args.headless,
        video_source=args.video,
        camera_index=args.camera,
        detection_interval=args.detection_interval,
        calibrate_screen=getattr(args, "calibrate_screen", False),
    )
    return status if isinstance(status, int) else 0


if __name__ == "__main__":
    sys.exit(main())
