"""
Centralized Configuration Module for Cheating Surveillance System.

Resolves project-relative model paths, camera settings, calibration parameters,
violation thresholds, phone detection parameters, head-pose classification limits,
pupil thresholding settings, environment configuration, and evidence retention.
"""

from pathlib import Path
import math
import os
import re
from typing import Collection, Optional, Tuple, Union

from model_integrity import (
    DEFAULT_MODEL_INTEGRITY_REGISTRY as MODEL_INTEGRITY_REGISTRY,
    ModelIntegrityError,
    ModelMissingError,
    ModelCorruptedError,
    ModelUnregisteredError,
    verify_model_asset,
    compute_file_sha256,
    temporary_model_registry,
    register_test_model,
    reset_model_registry,
)
from network_isolation import configure_offline_environment, enforce_offline_isolation
from path_security import is_symlink_or_junction, ensure_safe_directory

# Configure environment before model operations
configure_offline_environment()

# Resolve base project directory
BASE_DIR: Path = Path(__file__).resolve().parent


def load_dotenv_file(filepath: Optional[Union[str, Path]] = None, override: bool = False) -> bool:
    """
    Safely loads environment variables from a .env file into os.environ.

    Security controls:
    - Verifies file exists and is not a symlink or Windows directory junction.
    - Strips comments and parses standard KEY=VALUE format.
    - Validates key identifiers (alphanumeric + underscore).
    - Avoids overwriting existing process environment variables unless override=True.
    - Never logs or exposes secret values in exceptions or logs.

    Returns:
        bool: True if a .env file was successfully located and parsed, False otherwise.
    """
    target = Path(filepath) if filepath is not None else (BASE_DIR / ".env")
    if not target.is_file():
        return False
    if is_symlink_or_junction(target):
        return False

    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            if len(val) >= 2:
                val = val[1:-1]
        else:
            val = val.split("#", 1)[0].strip()

        if override or key not in os.environ:
            os.environ[key] = val

    return True


def get_env_int(
    name: str,
    default: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> int:
    """Safely parses an integer environment variable with bounds validation."""
    val_str = os.environ.get(name)
    if val_str is None:
        return default
    val_str = val_str.strip()
    if not val_str:
        return default
    try:
        val = int(val_str)
    except ValueError:
        raise ValueError(f"Invalid integer value for environment variable '{name}'.")
    if min_val is not None and val < min_val:
        raise ValueError(f"Environment variable '{name}' must be >= {min_val}.")
    if max_val is not None and val > max_val:
        raise ValueError(f"Environment variable '{name}' must be <= {max_val}.")
    return val


def get_env_float(
    name: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Safely parses a float environment variable with bounds and finite validation."""
    val_str = os.environ.get(name)
    if val_str is None:
        return default
    val_str = val_str.strip()
    if not val_str:
        return default
    try:
        val = float(val_str)
    except ValueError:
        raise ValueError(f"Invalid float value for environment variable '{name}'.")
    if not math.isfinite(val):
        raise ValueError(f"Environment variable '{name}' must be a finite float.")
    if min_val is not None and val < min_val:
        raise ValueError(f"Environment variable '{name}' must be >= {min_val}.")
    if max_val is not None and val > max_val:
        raise ValueError(f"Environment variable '{name}' must be <= {max_val}.")
    return val


def get_env_bool(name: str, default: bool) -> bool:
    """Safely parses a boolean environment variable."""
    val_str = os.environ.get(name)
    if val_str is None:
        return default
    val_str = val_str.strip().lower()
    if val_str in ("1", "true", "yes", "on"):
        return True
    if val_str in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value for environment variable '{name}'.")


def get_env_str(
    name: str,
    default: str,
    allowed_values: Optional[Collection[str]] = None,
) -> str:
    """Safely parses a string environment variable with membership validation."""
    val_str = os.environ.get(name)
    if val_str is None:
        return default
    val_str = val_str.strip()
    if not val_str:
        return default
    if allowed_values is not None and val_str not in allowed_values:
        raise ValueError(
            f"Invalid value for environment variable '{name}'. "
            f"Must be one of: {sorted(list(allowed_values))}."
        )
    return val_str


def get_env_path(name: str, default: Path) -> Path:
    """Safely resolves an environment directory path relative to BASE_DIR."""
    val_str = os.environ.get(name)
    if val_str is None:
        return default
    val_str = val_str.strip()
    if not val_str:
        return default
    p = Path(val_str)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


# Required model filenames
SHAPE_PREDICTOR_FILENAME: str = "shape_predictor_68_face_landmarks.dat"
YOLO_MODEL_FILENAME: str = "best_yolov12.pt"

# Demo video directory
DEMO_VID_DIR: Path = BASE_DIR / "Demo_vid"

# Screen gaze calibration default geometry
DEFAULT_DISPLAY_WIDTH: int = 1920
DEFAULT_DISPLAY_HEIGHT: int = 1080
SCREEN_CALIBRATION_MIN_TARGETS: int = 4
SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET: int = 3
SCREEN_CALIBRATION_MAX_RMSE: float = 0.18
SCREEN_CALIBRATION_MIN_CONFIDENCE: float = 0.20
SCREEN_BOUNDARY_MARGIN: float = 0.05
SCREEN_CALIBRATION_MIN_DWELL_SEC: float = 1.0
SCREEN_CALIBRATION_STABILITY_WINDOW_SEC: float = 0.4
SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG: float = 2.0
SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC: float = 0.15

# Pupil and gaze defaults
PUPIL_MIN_CONTOUR_AREA: float = 3.0
PUPIL_MAX_CONTOUR_RATIO: float = 0.80
GAZE_HORIZONTAL_THRESHOLD: float = 0.16
GAZE_VERTICAL_THRESHOLD: float = 0.10
GAZE_UPWARD_THRESHOLD: float = 0.08
FUSED_GAZE_MIN_CONFIDENCE: float = 0.20


def reload_config() -> None:
    """
    Re-evaluates configurable settings from process environment variables.
    Enforces validation and ensures offline isolation remains active.
    """
    global CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_VIDEO_FPS
    global CALIBRATION_DURATION_SEC, MIN_CALIBRATION_SAMPLES, MIN_GAZE_CALIBRATION_SAMPLES
    global VIOLATION_THRESHOLD_SEC, PHONE_CONFIDENCE_THRESHOLD
    global PHONE_DETECTION_INTERVAL_FRAMES, PHONE_DETECTION_CACHE_EXPIRY_FRAMES
    global ANGLE_HISTORY_SIZE, HEAD_NEUTRAL_PITCH_THRESHOLD, HEAD_NEUTRAL_YAW_THRESHOLD
    global HEAD_NEUTRAL_ROLL_THRESHOLD, HEAD_DEVIATION_PITCH_THRESHOLD, HEAD_DEVIATION_YAW_THRESHOLD
    global HEAD_DEVIATION_ROLL_THRESHOLD, PUPIL_DETECTION_METHOD, PUPIL_FIXED_THRESHOLD
    global LOG_DIR, MODEL_DIR, SHAPE_PREDICTOR_PATH, YOLO_MODEL_PATH, SCREEN_CALIBRATION_PATH
    global RETENTION_DAYS, RETENTION_ENABLED, ENCRYPTION_ENABLED, KEYSTORE_DIR, KEYSTORE_PATH

    MODEL_DIR = get_env_path("SURVEILLANCE_MODEL_DIR", BASE_DIR / "model")
    LOG_DIR = get_env_path("SURVEILLANCE_LOG_DIR", BASE_DIR / "log")
    SCREEN_CALIBRATION_PATH = LOG_DIR / "screen_gaze_calibration.json"
    SHAPE_PREDICTOR_PATH = MODEL_DIR / SHAPE_PREDICTOR_FILENAME
    YOLO_MODEL_PATH = MODEL_DIR / YOLO_MODEL_FILENAME

    KEYSTORE_DIR = get_env_path("SURVEILLANCE_KEYSTORE_DIR", BASE_DIR / "keys")
    KEYSTORE_PATH = get_env_path("SURVEILLANCE_KEYSTORE_PATH", KEYSTORE_DIR / "keystore.enc")

    CAMERA_INDEX = get_env_int("SURVEILLANCE_CAMERA_INDEX", 0, min_val=0)
    FRAME_WIDTH = get_env_int("SURVEILLANCE_FRAME_WIDTH", 640, min_val=1)
    FRAME_HEIGHT = get_env_int("SURVEILLANCE_FRAME_HEIGHT", 480, min_val=1)
    DEFAULT_VIDEO_FPS = get_env_float("SURVEILLANCE_DEFAULT_VIDEO_FPS", 30.0, min_val=0.1)

    CALIBRATION_DURATION_SEC = get_env_float("SURVEILLANCE_CALIBRATION_DURATION_SEC", 5.0, min_val=0.1)
    MIN_CALIBRATION_SAMPLES = get_env_int("SURVEILLANCE_MIN_CALIBRATION_SAMPLES", 10, min_val=1)
    MIN_GAZE_CALIBRATION_SAMPLES = get_env_int("SURVEILLANCE_MIN_GAZE_CALIBRATION_SAMPLES", 10, min_val=1)

    VIOLATION_THRESHOLD_SEC = get_env_float("SURVEILLANCE_VIOLATION_THRESHOLD_SEC", 3.0, min_val=0.1)
    PHONE_CONFIDENCE_THRESHOLD = get_env_float("SURVEILLANCE_PHONE_CONFIDENCE_THRESHOLD", 0.70, min_val=0.01, max_val=1.0)
    PHONE_DETECTION_INTERVAL_FRAMES = get_env_int("SURVEILLANCE_PHONE_DETECTION_INTERVAL_FRAMES", 3, min_val=1)
    PHONE_DETECTION_CACHE_EXPIRY_FRAMES = get_env_int("SURVEILLANCE_PHONE_DETECTION_CACHE_EXPIRY_FRAMES", 6, min_val=1)

    ANGLE_HISTORY_SIZE = get_env_int("SURVEILLANCE_ANGLE_HISTORY_SIZE", 10, min_val=1)
    HEAD_NEUTRAL_PITCH_THRESHOLD = get_env_float("SURVEILLANCE_HEAD_NEUTRAL_PITCH_THRESHOLD", 8.0, min_val=0.1)
    HEAD_NEUTRAL_YAW_THRESHOLD = get_env_float("SURVEILLANCE_HEAD_NEUTRAL_YAW_THRESHOLD", 12.0, min_val=0.1)
    HEAD_NEUTRAL_ROLL_THRESHOLD = get_env_float("SURVEILLANCE_HEAD_NEUTRAL_ROLL_THRESHOLD", 5.0, min_val=0.1)
    HEAD_DEVIATION_PITCH_THRESHOLD = get_env_float("SURVEILLANCE_HEAD_DEVIATION_PITCH_THRESHOLD", 10.0, min_val=0.1)
    HEAD_DEVIATION_YAW_THRESHOLD = get_env_float("SURVEILLANCE_HEAD_DEVIATION_YAW_THRESHOLD", 15.0, min_val=0.1)
    HEAD_DEVIATION_ROLL_THRESHOLD = get_env_float("SURVEILLANCE_HEAD_DEVIATION_ROLL_THRESHOLD", 7.0, min_val=0.1)

    PUPIL_DETECTION_METHOD = get_env_str("SURVEILLANCE_PUPIL_DETECTION_METHOD", "adaptive", allowed_values=("adaptive", "fixed", "otsu"))
    PUPIL_FIXED_THRESHOLD = get_env_int("SURVEILLANCE_PUPIL_FIXED_THRESHOLD", 50, min_val=0, max_val=255)

    RETENTION_DAYS = get_env_int("SURVEILLANCE_RETENTION_DAYS", 30, min_val=1)
    RETENTION_ENABLED = get_env_bool("SURVEILLANCE_RETENTION_ENABLED", False)
    ENCRYPTION_ENABLED = get_env_bool("SURVEILLANCE_ENCRYPTION_ENABLED", False)


# Attempt loading .env file if present in project root
load_dotenv_file()

# Initialize configuration variables
reload_config()


def validate_config(registry: Optional[dict] = None) -> None:
    """
    Validates required configuration values, directories, and model files at startup.
    Raises ValueError if numeric parameters are outside valid operational bounds.
    Raises ModelIntegrityError (or FileNotFoundError) if a required model file is missing,
    corrupted, unapproved, or fails cryptographic integrity checks.
    """
    finite_settings = {
        "CALIBRATION_DURATION_SEC": CALIBRATION_DURATION_SEC,
        "VIOLATION_THRESHOLD_SEC": VIOLATION_THRESHOLD_SEC,
        "SCREEN_CALIBRATION_MAX_RMSE": SCREEN_CALIBRATION_MAX_RMSE,
        "SCREEN_CALIBRATION_MIN_CONFIDENCE": SCREEN_CALIBRATION_MIN_CONFIDENCE,
        "SCREEN_BOUNDARY_MARGIN": SCREEN_BOUNDARY_MARGIN,
        "SCREEN_CALIBRATION_MIN_DWELL_SEC": SCREEN_CALIBRATION_MIN_DWELL_SEC,
        "SCREEN_CALIBRATION_STABILITY_WINDOW_SEC": SCREEN_CALIBRATION_STABILITY_WINDOW_SEC,
        "SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG": SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG,
        "SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC": SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC,
    }
    for name, value in finite_settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Invalid {name}: {value}. Value must be finite.")

    for name, value in {
        "SCREEN_CALIBRATION_MIN_TARGETS": SCREEN_CALIBRATION_MIN_TARGETS,
        "SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET": SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET,
        "FRAME_WIDTH": FRAME_WIDTH,
        "FRAME_HEIGHT": FRAME_HEIGHT,
        "DEFAULT_DISPLAY_WIDTH": DEFAULT_DISPLAY_WIDTH,
        "DEFAULT_DISPLAY_HEIGHT": DEFAULT_DISPLAY_HEIGHT,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid {name}: {value}. Value must be a positive integer.")

    # 1. Validate numeric thresholds
    if not (0.0 < PHONE_CONFIDENCE_THRESHOLD <= 1.0):
        raise ValueError(
            f"Invalid PHONE_CONFIDENCE_THRESHOLD: {PHONE_CONFIDENCE_THRESHOLD}. "
            f"Threshold must be a float in range (0.0, 1.0]."
        )

    if not isinstance(PHONE_DETECTION_INTERVAL_FRAMES, int) or PHONE_DETECTION_INTERVAL_FRAMES < 1:
        raise ValueError(
            f"Invalid PHONE_DETECTION_INTERVAL_FRAMES: {PHONE_DETECTION_INTERVAL_FRAMES}. "
            f"Must be a positive integer >= 1."
        )

    if not isinstance(PHONE_DETECTION_CACHE_EXPIRY_FRAMES, int) or PHONE_DETECTION_CACHE_EXPIRY_FRAMES < PHONE_DETECTION_INTERVAL_FRAMES:
        raise ValueError(
            f"Invalid PHONE_DETECTION_CACHE_EXPIRY_FRAMES: {PHONE_DETECTION_CACHE_EXPIRY_FRAMES}. "
            f"Must be an integer >= PHONE_DETECTION_INTERVAL_FRAMES ({PHONE_DETECTION_INTERVAL_FRAMES})."
        )

    if VIOLATION_THRESHOLD_SEC <= 0.0:
        raise ValueError(
            f"Invalid VIOLATION_THRESHOLD_SEC: {VIOLATION_THRESHOLD_SEC}. "
            f"Threshold must be strictly positive."
        )

    if DEFAULT_VIDEO_FPS <= 0.0:
        raise ValueError(
            f"Invalid DEFAULT_VIDEO_FPS: {DEFAULT_VIDEO_FPS}. "
            f"FPS must be strictly positive."
        )

    if CALIBRATION_DURATION_SEC <= 0.0:
        raise ValueError(
            f"Invalid CALIBRATION_DURATION_SEC: {CALIBRATION_DURATION_SEC}. "
            f"Duration must be strictly positive."
        )

    if MIN_CALIBRATION_SAMPLES < 1:
        raise ValueError(
            f"Invalid MIN_CALIBRATION_SAMPLES: {MIN_CALIBRATION_SAMPLES}. "
            f"Must require at least 1 calibration sample."
        )

    if MIN_GAZE_CALIBRATION_SAMPLES < 1:
        raise ValueError(
            f"Invalid MIN_GAZE_CALIBRATION_SAMPLES: {MIN_GAZE_CALIBRATION_SAMPLES}. "
            f"Must require at least 1 gaze calibration sample."
        )

    # Validate screen gaze calibration parameters (Phase 10.13.5)
    if SCREEN_CALIBRATION_MIN_TARGETS < 3:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_MIN_TARGETS: {SCREEN_CALIBRATION_MIN_TARGETS}. Must be >= 3."
        )
    if SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET < 1:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET: {SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET}. Must be >= 1."
        )
    if SCREEN_CALIBRATION_MAX_RMSE <= 0.0:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_MAX_RMSE: {SCREEN_CALIBRATION_MAX_RMSE}. Must be > 0.0."
        )
    if not (0.0 < SCREEN_CALIBRATION_MIN_CONFIDENCE <= 1.0):
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_MIN_CONFIDENCE: {SCREEN_CALIBRATION_MIN_CONFIDENCE}. "
            "Must be in range (0.0, 1.0]."
        )
    if not (0.0 <= SCREEN_BOUNDARY_MARGIN < 0.5):
        raise ValueError(
            f"Invalid SCREEN_BOUNDARY_MARGIN: {SCREEN_BOUNDARY_MARGIN}. Must be in [0.0, 0.5)."
        )
    if SCREEN_CALIBRATION_MIN_DWELL_SEC <= 0.0:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_MIN_DWELL_SEC: {SCREEN_CALIBRATION_MIN_DWELL_SEC}. Must be > 0.0."
        )
    if SCREEN_CALIBRATION_STABILITY_WINDOW_SEC <= 0.0:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_STABILITY_WINDOW_SEC: {SCREEN_CALIBRATION_STABILITY_WINDOW_SEC}. Must be > 0.0."
        )
    if SCREEN_CALIBRATION_STABILITY_WINDOW_SEC > SCREEN_CALIBRATION_MIN_DWELL_SEC:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_STABILITY_WINDOW_SEC: {SCREEN_CALIBRATION_STABILITY_WINDOW_SEC}. "
            f"Must be <= SCREEN_CALIBRATION_MIN_DWELL_SEC ({SCREEN_CALIBRATION_MIN_DWELL_SEC})."
        )
    if SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG <= 0.0:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG: {SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG}. Must be > 0.0."
        )
    if SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC <= 0.0:
        raise ValueError(
            f"Invalid SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC: {SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC}. Must be > 0.0."
        )
    if DEFAULT_DISPLAY_WIDTH <= 0:
        raise ValueError(
            f"Invalid DEFAULT_DISPLAY_WIDTH: {DEFAULT_DISPLAY_WIDTH}. Must be > 0."
        )
    if DEFAULT_DISPLAY_HEIGHT <= 0:
        raise ValueError(
            f"Invalid DEFAULT_DISPLAY_HEIGHT: {DEFAULT_DISPLAY_HEIGHT}. Must be > 0."
        )

    # Validate head pose thresholds
    if HEAD_NEUTRAL_PITCH_THRESHOLD <= 0.0 or HEAD_NEUTRAL_YAW_THRESHOLD <= 0.0 or HEAD_NEUTRAL_ROLL_THRESHOLD <= 0.0:
        raise ValueError("Head neutral thresholds must be strictly positive.")

    if HEAD_DEVIATION_PITCH_THRESHOLD < HEAD_NEUTRAL_PITCH_THRESHOLD:
        raise ValueError(
            f"HEAD_DEVIATION_PITCH_THRESHOLD ({HEAD_DEVIATION_PITCH_THRESHOLD}) must be >= "
            f"HEAD_NEUTRAL_PITCH_THRESHOLD ({HEAD_NEUTRAL_PITCH_THRESHOLD})."
        )

    if HEAD_DEVIATION_YAW_THRESHOLD < HEAD_NEUTRAL_YAW_THRESHOLD:
        raise ValueError(
            f"HEAD_DEVIATION_YAW_THRESHOLD ({HEAD_DEVIATION_YAW_THRESHOLD}) must be >= "
            f"HEAD_NEUTRAL_YAW_THRESHOLD ({HEAD_NEUTRAL_YAW_THRESHOLD})."
        )

    if HEAD_DEVIATION_ROLL_THRESHOLD < HEAD_NEUTRAL_ROLL_THRESHOLD:
        raise ValueError(
            f"HEAD_DEVIATION_ROLL_THRESHOLD ({HEAD_DEVIATION_ROLL_THRESHOLD}) must be >= "
            f"HEAD_NEUTRAL_ROLL_THRESHOLD ({HEAD_NEUTRAL_ROLL_THRESHOLD})."
        )

    # Validate pupil detection parameters
    if PUPIL_DETECTION_METHOD not in ("adaptive", "fixed", "otsu"):
        raise ValueError(
            f"Invalid PUPIL_DETECTION_METHOD: '{PUPIL_DETECTION_METHOD}'. "
            f"Must be one of ('adaptive', 'fixed', 'otsu')."
        )

    if not (0 <= PUPIL_FIXED_THRESHOLD <= 255):
        raise ValueError(
            f"Invalid PUPIL_FIXED_THRESHOLD: {PUPIL_FIXED_THRESHOLD}. Must be between 0 and 255."
        )

    if PUPIL_MIN_CONTOUR_AREA <= 0.0:
        raise ValueError(
            f"Invalid PUPIL_MIN_CONTOUR_AREA: {PUPIL_MIN_CONTOUR_AREA}. Must be strictly positive."
        )

    if not (0.0 < PUPIL_MAX_CONTOUR_RATIO <= 1.0):
        raise ValueError(
            f"Invalid PUPIL_MAX_CONTOUR_RATIO: {PUPIL_MAX_CONTOUR_RATIO}. Must be in range (0.0, 1.0]."
        )

    # Validate gaze deflection thresholds
    if not (0.0 < GAZE_HORIZONTAL_THRESHOLD < 0.5):
        raise ValueError(
            f"Invalid GAZE_HORIZONTAL_THRESHOLD: {GAZE_HORIZONTAL_THRESHOLD}. Must be in range (0.0, 0.5)."
        )

    if not (0.0 < GAZE_VERTICAL_THRESHOLD < 0.5):
        raise ValueError(
            f"Invalid GAZE_VERTICAL_THRESHOLD: {GAZE_VERTICAL_THRESHOLD}. Must be in range (0.0, 0.5)."
        )

    if not (0.0 < GAZE_UPWARD_THRESHOLD < 0.5):
        raise ValueError(
            f"Invalid GAZE_UPWARD_THRESHOLD: {GAZE_UPWARD_THRESHOLD}. Must be in range (0.0, 0.5)."
        )

    if not (0.0 < FUSED_GAZE_MIN_CONFIDENCE <= 1.0):
        raise ValueError(
            f"Invalid FUSED_GAZE_MIN_CONFIDENCE: {FUSED_GAZE_MIN_CONFIDENCE}. Must be in range (0.0, 1.0]."
        )

    # Validate retention settings
    if not isinstance(RETENTION_DAYS, int) or RETENTION_DAYS < 1:
        raise ValueError(
            f"Invalid RETENTION_DAYS: {RETENTION_DAYS}. Must be an integer >= 1."
        )

    # 2. Validate required model files and cryptographic integrity
    if not SHAPE_PREDICTOR_PATH.is_file():
        raise FileNotFoundError(
            f"Missing required facial landmark model at '{SHAPE_PREDICTOR_PATH}'. "
            f"Please ensure '{SHAPE_PREDICTOR_FILENAME}' exists in '{MODEL_DIR}'."
        )

    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing required YOLO model at '{YOLO_MODEL_PATH}'. "
            f"Please ensure '{YOLO_MODEL_FILENAME}' exists in '{MODEL_DIR}'."
        )

    verify_model_asset(SHAPE_PREDICTOR_PATH, registry=registry)
    verify_model_asset(YOLO_MODEL_PATH, registry=registry)

    # 3. Enforce offline isolation for ML backend
    enforce_offline_isolation()

    # 4. Ensure log directory exists safely
    ensure_safe_directory(LOG_DIR)

    # 5. Validate encryption subsystem if enabled (Phase 10.13.28.3)
    if ENCRYPTION_ENABLED:
        from crypto_core import SecurityConfigurationError, KeyManagementError
        from keystore import Keystore, apply_restrictive_permissions
        from encrypted_storage import EncryptedStorageManager

        if not KEYSTORE_PATH.is_file():
            raise SecurityConfigurationError(
                f"SURVEILLANCE_ENCRYPTION_ENABLED is True, but keystore file does not exist at '{KEYSTORE_PATH}'."
            )
        try:
            ks = Keystore.load(KEYSTORE_PATH)
            # Verify active DEK availability
            active_dek = ks.unwrap_dek(ks.active_key_id)
            if not active_dek or len(active_dek) != 32:
                raise SecurityConfigurationError("Active DEK is missing or invalid in keystore.")
            # Verify ACL enforcement
            apply_restrictive_permissions(KEYSTORE_PATH, fail_closed=True)
            # Verify EncryptedStorageManager initialization
            EncryptedStorageManager(ks)
        except (SecurityConfigurationError, KeyManagementError):
            raise
        except Exception as exc:
            raise SecurityConfigurationError(
                f"Failed to initialize encrypted storage subsystem: {exc}"
            ) from exc
