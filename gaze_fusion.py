"""
3D Head-Eye Gaze Fusion and Structured Result Models Module.
Phase 10.13.4 Implementation.

Coordinate Frame Definitions:
=============================
1. Camera Coordinate Frame (OpenCV Standard):
   - Origin: Camera optical center (pinhole).
   - +X: Points rightward across the image frame.
   - +Y: Points downward across the image frame.
   - +Z: Points away from camera into the scene (optical axis depth).
   - Handedness: Right-handed (X x Y = Z).

2. Head-Local Frame (Anthropometric Facial Model):
   - Defined relative to facial landmarks (nose tip as origin [0, 0, 0]).
   - +X: Points towards subject's anatomical left (image right).
   - +Y: Points downwards towards the chin.
   - +Z: Points forward along the line of sight when facing neutral.
     Note on Model Points: In solvePnP landmark definition (head_pose.py), landmarks
     are specified with +Z into the skull. The rotation matrix R_pnp maps model coordinates
     to camera coordinates (P_cam = R_pnp @ P_model + T).
     The similarity reflection R_camera_head = Sz @ R_pnp @ Sz (with Sz = diag(1, 1, -1))
     transforms head-local vectors (where forward neutral gaze is [0, 0, 1]) directly into
     camera-frame vectors, preserving proper rotation order (Z-Y-X) and roll direction (+roll):
     g_camera = R_camera_head @ g_head

3. Angle Sign Conventions (Degrees):
   - Pitch: Nose UP is positive (+), Nose DOWN is negative (-).
   - Yaw: Head turned RIGHT is positive (+), Head turned LEFT is negative (-).
   - Roll: Head tilted RIGHT (clockwise) is positive (+), LEFT is negative (-).

4. Camera-Relative Gaze Angle Conventions (from normalized g = [x, y, z]):
   - yaw = atan2(x, z) in degrees (positive = right, negative = left).
   - pitch = atan2(-y, sqrt(x^2 + z^2)) in degrees (positive = up, negative = down).

Limitations & Disclaimers:
==========================
This module implements a geometrically consistent 3D coordinate transformation.
However, current eye tracking extracts 2D normalized pupil positions from palpebral
apertures without physical corneal reflection, eyeball radius, or optical axis calibration.
Therefore, head-frame eye gaze vectors represent an estimated geometric deflection heuristic
rather than physical millimeter-accurate screen-point intersection.
Screen-point gaze prediction is strictly deferred until proper 3D eye calibration is performed.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import config
from screen_gaze import ScreenGazeResult


# ---------------------------------------------------------------------------
# Structured Result Models
# ---------------------------------------------------------------------------

@dataclass
class HeadPoseResult:
    """
    Independent 3D Head Orientation Result (Output A).
    Represents head orientation estimated via 2D-3D solvePnP.
    """
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    rotation_matrix: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    confidence: float = 0.0
    status: str = "NO_FACE"
    coordinate_frame: str = "camera_opencv"

    @property
    def head_yaw(self) -> Optional[float]:
        return self.yaw

    @property
    def head_pitch(self) -> Optional[float]:
        return self.pitch

    @property
    def head_roll(self) -> Optional[float]:
        return self.roll

    @property
    def head_rotation_matrix(self) -> Optional[np.ndarray]:
        return self.rotation_matrix

    @property
    def head_translation(self) -> Optional[np.ndarray]:
        return self.translation

    @property
    def head_confidence(self) -> float:
        return self.confidence

    @property
    def head_tracking_status(self) -> str:
        return self.status


@dataclass
class EyeGazeResult:
    """
    Independent Eye Gaze Direction Result (Output B).
    Represents eye gaze estimated independently of head pose, supporting single-eye fallback.
    """
    left_eye_direction: Optional[str] = None
    right_eye_direction: Optional[str] = None
    selected_eye_direction: str = "Uncertain Gaze"
    confidence: float = 0.0
    tracking_state: str = "UNCERTAIN"
    status: str = "UNCERTAIN"
    coordinate_frame: str = "head_local"
    gaze_vector_head: Optional[np.ndarray] = None
    is_single_eye_fallback: bool = False
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    rejection_reason: str = ""

    @property
    def eye_direction(self) -> str:
        return self.selected_eye_direction

    @property
    def eye_confidence(self) -> float:
        return self.confidence

    @property
    def eye_tracking_state(self) -> str:
        return self.tracking_state

    @property
    def eye_tracking_status(self) -> str:
        return self.status


@dataclass
class CombinedGazeResult:
    """
    Combined Camera-Relative 3D Gaze Result (Output C).
    Transforms head-frame eye gaze into the camera coordinate system:
        g_camera = R_camera_head @ g_head
    """
    gaze_vector_camera: Optional[Tuple[float, float, float]] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    confidence: float = 0.0
    status: str = "UNCERTAIN"
    uncertainty: float = 1.0
    source_components: Dict[str, Any] = field(default_factory=dict)

    @property
    def camera_gaze_vector(self) -> Optional[Tuple[float, float, float]]:
        return self.gaze_vector_camera

    @property
    def camera_gaze_yaw(self) -> Optional[float]:
        return self.yaw

    @property
    def camera_gaze_pitch(self) -> Optional[float]:
        return self.pitch

    @property
    def combined_gaze_confidence(self) -> float:
        return self.confidence

    @property
    def combined_gaze_status(self) -> str:
        return self.status

    @property
    def combined_gaze_uncertainty(self) -> float:
        return self.uncertainty


# ---------------------------------------------------------------------------
# Reusable Mathematical Utilities
# ---------------------------------------------------------------------------

def validate_vector3d(v: Any) -> bool:
    """
    Validates that input is a 3-element numeric vector containing finite numbers.
    """
    if v is None:
        return False
    try:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.shape != (3,):
            return False
        return bool(np.all(np.isfinite(arr)))
    except (ValueError, TypeError):
        return False


def is_zero_vector(v: Any, tol: float = 1e-6) -> bool:
    """
    Checks if a 3D vector has Euclidean length approximately zero.
    """
    if not validate_vector3d(v):
        return True
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return bool(np.linalg.norm(arr) < tol)


def normalize_vector3d(v: Any, tol: float = 1e-6) -> Optional[np.ndarray]:
    """
    Validates and normalizes a 3D vector to unit Euclidean norm.
    Returns None if vector is invalid, non-finite, or zero-length.
    """
    if not validate_vector3d(v):
        return None
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < tol or not np.isfinite(norm):
        return None
    return arr / norm


def validate_rotation_matrix(R: Any, tol: float = 1e-3) -> bool:
    """
    Validates a 3x3 rotation matrix:
      1. Correct shape (3, 3) and finite real values.
      2. Determinant approximately +1: |det(R) - 1.0| < tol.
      3. Orthogonality: ||R^T R - I||_inf < tol and ||R R^T - I||_inf < tol.
    """
    if R is None:
        return False
    try:
        arr = np.asarray(R, dtype=np.float64)
        if arr.shape != (3, 3):
            return False
        if not np.all(np.isfinite(arr)):
            return False
        det = float(np.linalg.det(arr))
        if abs(det - 1.0) >= tol:
            return False
        eye = np.eye(3, dtype=np.float64)
        ortho_check1 = np.max(np.abs(arr.T @ arr - eye))
        ortho_check2 = np.max(np.abs(arr @ arr.T - eye))
        return bool(ortho_check1 < tol and ortho_check2 < tol)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return False


def camera_gaze_to_angles(g: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates yaw and pitch angles in degrees from a normalized camera gaze vector g = (x, y, z).
    Conventions:
      yaw = atan2(x, z): positive = right, negative = left.
      pitch = atan2(-y, sqrt(x^2 + z^2)): positive = up, negative = down (-Y is up in camera coords).
    Returns:
      (yaw, pitch) in degrees rounded to 2 decimal places, or (None, None) if invalid.
    """
    if not validate_vector3d(g) or is_zero_vector(g):
        return None, None

    arr = np.asarray(g, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > 1e-3:
        arr = arr / norm

    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
    # Protect against exact backward vectors (z <= 0) where atan2(x, z) wraps to +-180
    yaw_rad = math.atan2(x, z)
    pitch_rad = math.atan2(-y, math.sqrt(x * x + z * z))

    return round(math.degrees(yaw_rad), 2), round(math.degrees(pitch_rad), 2)


def angles_to_camera_gaze(yaw_deg: float, pitch_deg: float) -> Optional[np.ndarray]:
    """
    Constructs a normalized 3D camera-frame gaze vector from yaw and pitch angles in degrees.
    Inverse of camera_gaze_to_angles for synthetic testing and round-trip verification.
    """
    try:
        y_rad = math.radians(yaw_deg)
        p_rad = math.radians(pitch_deg)
        # In camera convention:
        # y = -sin(pitch)
        # x = sin(yaw) * cos(pitch)
        # z = cos(yaw) * cos(pitch)
        vec = np.array([
            math.sin(y_rad) * math.cos(p_rad),
            -math.sin(p_rad),
            math.cos(y_rad) * math.cos(p_rad)
        ], dtype=np.float64)
        return normalize_vector3d(vec)
    except (ValueError, TypeError):
        return None


def compute_camera_head_rotation_matrix(rvec_or_R_pnp: np.ndarray) -> Optional[np.ndarray]:
    """
    Derives R_camera_head from cv2.solvePnP output.
    In solvePnP, R_pnp maps model coordinates (where +Z points into the head) to camera coordinates.
    The similarity transformation R_camera_head = Sz @ R_pnp @ Sz (where Sz = diag(1, 1, -1))
    maps head-local vectors (where forward neutral gaze is [0, 0, 1])
    to camera-frame gaze vectors, maintaining correct rotation order and roll sign.
    """
    if rvec_or_R_pnp is None:
        return None
    try:
        import cv2
        arr = np.asarray(rvec_or_R_pnp, dtype=np.float64)
        if arr.shape in ((3, 1), (1, 3), (3,)):
            R_pnp, _ = cv2.Rodrigues(arr.reshape(3, 1))
        elif arr.shape == (3, 3):
            R_pnp = arr
        else:
            return None

        Sz = np.diag([1.0, 1.0, -1.0])
        R_camera_head = Sz @ R_pnp @ Sz
        if validate_rotation_matrix(R_camera_head):
            return R_camera_head
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Eye Gaze Direction to Head-Local Vector
# ---------------------------------------------------------------------------

def eye_direction_to_head_vector(
    gaze_dir: Optional[str],
    diff_x: Optional[float] = None,
    diff_y: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Maps eye gaze observations into a unit 3D direction vector in the head coordinate frame.

    Mathematical Projection Model:
      In head coordinates:
        +X: image-right (subject's anatomical left).
        +Y: image-down (towards chin).
        +Z: forward along the neutral optical axis.

      If normalized differentials (diff_x, diff_y) are provided:
        diff_x = base_x - norm_x: positive for rightward gaze (+X), negative for leftward (-X).
        diff_y = norm_y - base_y: positive for downward gaze (+Y), negative for upward (-Y).
        Sensitivity scale factor: kappa = 1.5 (maps +-0.20 pupil ratio excursion to ~+-18 deg).
        v_x = clamp(diff_x * 1.5, -0.6, 0.6)
        v_y = clamp(diff_y * 1.5, -0.6, 0.6)
        v_z = sqrt(max(0.01, 1.0 - v_x^2 - v_y^2))
        g_head = normalize(v_x, v_y, v_z)

      If differentials are not available, canonical angles are used:
        'Looking Center': [0.0, 0.0, 1.0]
        'Looking Right':  [sin(15 deg), 0.0, cos(15 deg)]
        'Looking Left':   [-sin(15 deg), 0.0, cos(15 deg)]
        'Looking Up':     [0.0, -sin(12 deg), cos(12 deg)]
        'Looking Down':   [0.0, sin(15 deg), cos(15 deg)]
        'Uncertain Gaze' or invalid: None
    """
    if gaze_dir is None or gaze_dir in ("Uncertain Gaze", "No Face Detected", "Calibrating", ""):
        return None

    # If continuous differentials are provided and finite
    if (diff_x is not None and diff_y is not None and
            np.isfinite(diff_x) and np.isfinite(diff_y)):
        vx = float(np.clip(diff_x * 1.5, -0.6, 0.6))
        vy = float(np.clip(diff_y * 1.5, -0.6, 0.6))
        vz_sq = max(0.01, 1.0 - vx * vx - vy * vy)
        vz = math.sqrt(vz_sq)
        return normalize_vector3d(np.array([vx, vy, vz], dtype=np.float64))

    # Canonical directional fallbacks
    if gaze_dir == "Looking Center":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif gaze_dir == "Looking Right":
        rad = math.radians(15.0)
        return np.array([math.sin(rad), 0.0, math.cos(rad)], dtype=np.float64)
    elif gaze_dir == "Looking Left":
        rad = math.radians(15.0)
        return np.array([-math.sin(rad), 0.0, math.cos(rad)], dtype=np.float64)
    elif gaze_dir == "Looking Up":
        rad = math.radians(12.0)
        return np.array([0.0, -math.sin(rad), math.cos(rad)], dtype=np.float64)
    elif gaze_dir == "Looking Down":
        rad = math.radians(15.0)
        return np.array([0.0, math.sin(rad), math.cos(rad)], dtype=np.float64)

    return None


def _safe_joint_confidence(head_conf: float, eye_conf: float, threshold: float = config.FUSED_GAZE_MIN_CONFIDENCE) -> float:
    """
    Computes joint confidence product ensuring rounding never promotes
    an uncertain / below-threshold measurement into a valid one.
    """
    try:
        h = float(head_conf)
        e = float(eye_conf)
        if not (math.isfinite(h) and math.isfinite(e)) or h <= 0.0 or e <= 0.0:
            return 0.0
        raw_joint = h * e
        joint = round(raw_joint, 4)
        if (h < threshold or e < threshold or raw_joint < threshold) and joint >= threshold:
            joint = round(threshold - 1e-4, 4)
        return float(np.clip(joint, 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Confidence-Aware 3D Gaze Fusion
# ---------------------------------------------------------------------------

def fuse_head_and_eye_gaze(
    head_pose: Optional[HeadPoseResult],
    eye_gaze: Optional[EyeGazeResult],
) -> CombinedGazeResult:
    """
    Fuses independent 3D head pose and eye gaze into a combined camera-relative gaze vector:
        g_camera = R_camera_head @ g_head

    Confidence Aggregation Model:
      - Independent component confidences are preserved in source_components.
      - Combined confidence is computed as the product of independent confidences:
            confidence_combined = round(head_confidence * eye_confidence, 4)
        Mathematical Rationale: Combined gaze requires BOTH the head pose estimate and
        the eye gaze estimate to be accurate. The joint probability / reliability of two
        independent modalities degrades if either modality becomes uncertain.
      - Combined uncertainty is:
            uncertainty = round(1.0 - confidence_combined, 4)
      - If either component is invalid, missing, or uncertain:
        Combined result returns status 'UNCERTAIN', vector None, angles None.
      - Rounding must never promote an uncertain or below-threshold measurement into a valid one.
      - Explicit UNCERTAIN states are never classified as misconduct.
    """
    components: Dict[str, Any] = {}
    min_conf_threshold = float(config.FUSED_GAZE_MIN_CONFIDENCE)

    # Handle missing head pose
    if head_pose is None or getattr(head_pose, "status", None) == "NO_FACE":
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=0.0,
            status="NO_FACE",
            uncertainty=1.0,
            source_components={"error": "no_face"},
        )

    components["head_yaw"] = head_pose.yaw
    components["head_pitch"] = head_pose.pitch
    components["head_roll"] = head_pose.roll
    components["head_confidence"] = head_pose.confidence
    components["head_status"] = head_pose.status

    try:
        raw_head_conf = float(head_pose.confidence) if (head_pose.confidence is not None and math.isfinite(float(head_pose.confidence))) else 0.0
    except (TypeError, ValueError):
        raw_head_conf = 0.0

    try:
        raw_eye_conf = float(eye_gaze.confidence) if (eye_gaze is not None and eye_gaze.confidence is not None and math.isfinite(float(eye_gaze.confidence))) else 0.0
    except (TypeError, ValueError):
        raw_eye_conf = 0.0

    # Explicitly validate head-pose status before using head rotation (GAP-02 Rule 3)
    if head_pose.status != "VALID":
        joint_conf = _safe_joint_confidence(raw_head_conf, raw_eye_conf, min_conf_threshold)
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=joint_conf,
            status="UNCERTAIN",
            uncertainty=round(1.0 - joint_conf, 4),
            source_components={**components, "error": f"non_valid_head_status_{head_pose.status}"},
        )

    # Validate head confidence against minimum threshold (GAP-02 Rule 4)
    if raw_head_conf < min_conf_threshold or not math.isfinite(raw_head_conf):
        joint_conf = _safe_joint_confidence(raw_head_conf, raw_eye_conf, min_conf_threshold)
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=joint_conf,
            status="UNCERTAIN",
            uncertainty=round(1.0 - joint_conf, 4),
            source_components={**components, "error": "low_head_confidence"},
        )

    # Validate head pose angles finiteness
    if (
        head_pose.yaw is None or head_pose.pitch is None
        or not math.isfinite(float(head_pose.yaw)) or not math.isfinite(float(head_pose.pitch))
    ):
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=0.0,
            status="UNCERTAIN",
            uncertainty=1.0,
            source_components={**components, "error": "non_finite_head_angles"},
        )

    # Validate head rotation matrix
    if head_pose.rotation_matrix is None or not validate_rotation_matrix(head_pose.rotation_matrix):
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=0.0,
            status="INVALID_ROTATION",
            uncertainty=1.0,
            source_components={**components, "error": "invalid_rotation_matrix"},
        )

    # Handle missing or uncertain eye gaze
    if eye_gaze is None:
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=0.0,
            status="UNCERTAIN_EYE",
            uncertainty=1.0,
            source_components={**components, "error": "missing_eye_gaze"},
        )

    components["eye_direction"] = eye_gaze.selected_eye_direction
    components["eye_confidence"] = eye_gaze.confidence
    components["eye_tracking_state"] = eye_gaze.tracking_state
    components["is_single_eye_fallback"] = eye_gaze.is_single_eye_fallback

    # Eye gaze must have valid vector and minimum usability confidence
    if (
        eye_gaze.tracking_state == "UNCERTAIN"
        or eye_gaze.status != "VALID"
        or eye_gaze.gaze_vector_head is None
        or not validate_vector3d(eye_gaze.gaze_vector_head)
        or raw_eye_conf < min_conf_threshold
    ):
        # Degraded/uncertain eye tracking
        joint_conf = _safe_joint_confidence(raw_head_conf, raw_eye_conf, min_conf_threshold)
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=joint_conf,
            status="UNCERTAIN",
            uncertainty=round(1.0 - joint_conf, 4),
            source_components={**components, "error": "uncertain_eye_gaze"},
        )

    # 3D Vector Transformation: g_camera = R_camera_head @ g_head
    R_cam_head = head_pose.rotation_matrix
    v_head = eye_gaze.gaze_vector_head
    v_cam_raw = R_cam_head @ v_head

    g_cam = normalize_vector3d(v_cam_raw)
    if g_cam is None:
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=0.0,
            status="UNCERTAIN",
            uncertainty=1.0,
            source_components={**components, "error": "zero_length_gaze_vector"},
        )

    yaw, pitch = camera_gaze_to_angles(g_cam)
    joint_confidence = _safe_joint_confidence(raw_head_conf, raw_eye_conf, min_conf_threshold)

    # Fused confidence must not fall below minimum floor to be marked VALID
    if joint_confidence < min_conf_threshold:
        return CombinedGazeResult(
            gaze_vector_camera=None,
            yaw=None,
            pitch=None,
            confidence=joint_confidence,
            status="UNCERTAIN",
            uncertainty=round(1.0 - joint_confidence, 4),
            source_components={**components, "error": "insufficient_joint_confidence"},
        )

    uncertainty = round(1.0 - joint_confidence, 4)

    components["gaze_vector_head"] = tuple(float(x) for x in v_head)
    components["gaze_vector_camera"] = tuple(float(x) for x in g_cam)

    return CombinedGazeResult(
        gaze_vector_camera=tuple(float(x) for x in g_cam),
        yaw=yaw,
        pitch=pitch,
        confidence=joint_confidence,
        status="VALID",
        uncertainty=uncertainty,
        source_components=components,
    )
