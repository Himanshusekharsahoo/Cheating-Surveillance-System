import math
import os
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, Any
import cv2
import numpy as np

from model_integrity import verify_model_asset, ModelIntegrityError

# Resolve base project directory and default predictor path
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTOR_PATH = BASE_DIR / "model" / "shape_predictor_68_face_landmarks.dat"

# Head pose state constants
HEAD_SCREEN = "Looking at Screen"
HEAD_LEFT = "Looking Left"
HEAD_RIGHT = "Looking Right"
HEAD_UP = "Looking Up"
HEAD_DOWN = "Looking Down"
HEAD_TILTED = "Tilted"
HEAD_NO_FACE = "No Face Detected"

COMPLIANT_HEAD_STATES = {HEAD_SCREEN}

def get_predictor_path(predictor_path=None):
    """
    Resolve and validate the dlib shape predictor path.
    Raises FileNotFoundError if the file does not exist.
    """
    target = Path(predictor_path) if predictor_path else DEFAULT_PREDICTOR_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"dlib shape predictor not found at: '{target}'. "
            f"Please ensure 'shape_predictor_68_face_landmarks.dat' exists in '{BASE_DIR / 'model'}'."
        )
    return target

_detector = None
_predictor = None

def get_face_detector():
    """Lazily load dlib frontal face detector."""
    global _detector
    if _detector is None:
        import dlib
        _detector = dlib.get_frontal_face_detector()
    return _detector

def get_shape_predictor(predictor_path=None, registry=None):
    """Lazily load dlib shape predictor model after verifying cryptographic integrity."""
    global _predictor
    if _predictor is None:
        target_path = get_predictor_path(predictor_path)
        # Strict fail-closed verification before dlib binary deserialization
        verify_model_asset(target_path, registry=registry)
        import dlib
        _predictor = dlib.shape_predictor(str(target_path))
    return _predictor

# 3D Model Points in Camera-Aligned Coordinates:
# +X: Image-right (subject's anatomical left)
# +Y: Image-down (chin direction)
# +Z: Optical axis depth away from camera (recessed behind nose tip into scene)
# Anthropometrically proportioned for 6 facial landmarks (dlib 30, 8, 36, 45, 48, 54):
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip (part 30)
    (0.0, 65.0, 20.0),      # Chin (part 8: down +65, back +20)
    (-33.0, -20.0, 25.0),   # Left eye (part 36: image-left -33, up -20, back +25)
    (33.0, -20.0, 25.0),    # Right eye (part 45: image-right +33, up -20, back +25)
    (-24.0, 35.0, 20.0),    # Left mouth corner (part 48: image-left -24, down +35, back +20)
    (24.0, 35.0, 20.0)      # Right mouth corner (part 54: image-right +24, down +35, back +20)
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion

def get_camera_matrix(frame_shape):
    """Compute camera matrix dynamically based on frame dimensions."""
    h, w = frame_shape[:2]
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    return np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

# Smoothing filter for stable head pose estimation
ANGLE_HISTORY_SIZE = 10
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)

# Global variables for state management
previous_state = HEAD_SCREEN


def reset_head_pose_state() -> None:
    """
    Explicitly reset head-pose smoothing history and tracking state (GAP-05).
    Safe to call repeatedly (idempotent).
    """
    global previous_state
    yaw_history.clear()
    pitch_history.clear()
    roll_history.clear()
    previous_state = HEAD_SCREEN


def get_head_pose_history_counts() -> Tuple[int, int, int]:
    """Return lengths of (yaw_history, pitch_history, roll_history) for state lifecycle testing."""
    return len(yaw_history), len(pitch_history), len(roll_history)


def get_head_pose_previous_state() -> str:
    """Return the current previous_state."""
    return previous_state

def get_head_pose_estimation(image_points, camera_matrix=None, frame_shape=(480, 640)):
    """
    Estimate pitch, yaw, roll, camera-head rotation matrix, translation vector,
    and fit confidence from 2D facial landmarks using solvePnP.

    Returns:
        tuple: (angles, rotation_matrix_camera_head, translation_vector, confidence)
        where:
          - angles is (pitch, yaw, roll) in degrees
          - rotation_matrix_camera_head is 3x3 ndarray mapping head vectors to camera coords
          - translation_vector is 3x1 ndarray from camera to head
          - confidence is float in [0.0, 1.0] derived from reprojection error
        or None if solvePnP fails.
    """
    if image_points is None or len(image_points) < 6 or not np.all(np.isfinite(image_points)):
        return None

    image_points = np.ascontiguousarray(image_points, dtype=np.float64)

    # Validate landmark variance to prevent unhandled C++ cv2.error in SQPNP (ISSUE-03)
    var_2d = np.var(image_points[:6], axis=0)
    if np.any(var_2d < 1e-4) or np.any(np.isnan(var_2d)):
        return None

    if camera_matrix is None:
        camera_matrix = get_camera_matrix(frame_shape)

    try:
        # Use SOLVEPNP_SQPNP for globally optimal non-iterative convergence; fallback to ITERATIVE
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not success:
            return None
    except (cv2.error, Exception):
        return None

    if rotation_vector is None or translation_vector is None:
        return None
    if not np.all(np.isfinite(rotation_vector)) or not np.all(np.isfinite(translation_vector)):
        return None

    # Translation Z must be positive (subject in front of camera)
    if translation_vector[2, 0] <= 0:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        cam_pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        cam_yaw = math.atan2(-rotation_matrix[2, 0], sy)
        cam_roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        cam_pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        cam_yaw = math.atan2(-rotation_matrix[2, 0], sy)
        cam_roll = 0.0

    # Convert camera frame rotation to anatomical physical angles:
    # Pitch: in camera coordinates (+Y down), nose up rotates negatively about X -> pitch = -cam_pitch
    # Yaw: turning head left moves nose to image-right (+X) -> yaw = -cam_yaw
    # Roll: tilting right rotates clockwise (+Z into scene) -> roll = cam_roll
    pitch = -float(np.degrees(cam_pitch))
    yaw = -float(np.degrees(cam_yaw))
    roll = float(np.degrees(cam_roll))

    # Reprojection error to assess confidence (ISSUE-05: fallback to 0.0 on error)
    try:
        projected_points, _ = cv2.projectPoints(
            model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        reproj_error = float(np.mean(np.linalg.norm(image_points[:6] - projected_points[:6], axis=1)))
        if not np.isfinite(reproj_error):
            confidence = 0.0
        else:
            confidence = float(np.clip(1.0 - (reproj_error / 10.0), 0.0, 1.0))
    except Exception:
        confidence = 0.0

    # R_camera_head maps head-local vectors ([0,0,1] forward) to camera frame
    # via similarity reflection Sz @ R_pnp @ Sz (ISSUE-01 fix)
    Sz = np.diag([1.0, 1.0, -1.0])
    rotation_matrix_camera_head = Sz @ rotation_matrix @ Sz

    return (pitch, yaw, roll), rotation_matrix_camera_head, translation_vector, confidence


def get_head_pose_angles(image_points, camera_matrix=None, frame_shape=(480, 640)):
    """
    Estimate pitch, yaw, and roll from 2D facial landmarks using solvePnP.
    Returns anatomical / physical angles in degrees:
      - Neutral frontal face: (0.0, 0.0, 0.0)
      - Pitch: Nose UP is positive (+), Nose DOWN is negative (-)
      - Yaw: Head turned LEFT is negative (-), Head turned RIGHT is positive (+)
      - Roll: Head tilted RIGHT is positive (+), Head tilted LEFT is negative (-)
    """
    estimation = get_head_pose_estimation(image_points, camera_matrix=camera_matrix, frame_shape=frame_shape)
    if estimation is None:
        return None
    return estimation[0]

def smooth_angle(angle_history, new_angle):
    """Append new angle to sliding window history and return mean."""
    angle_history.append(new_angle)
    return float(np.mean(angle_history))

from config import (
    HEAD_NEUTRAL_PITCH_THRESHOLD,
    HEAD_NEUTRAL_YAW_THRESHOLD,
    HEAD_NEUTRAL_ROLL_THRESHOLD,
    HEAD_DEVIATION_PITCH_THRESHOLD,
    HEAD_DEVIATION_YAW_THRESHOLD,
    HEAD_DEVIATION_ROLL_THRESHOLD,
)

def classify_head_pose(
    pitch,
    yaw,
    roll,
    calibrated_angles,
    prev_state=HEAD_SCREEN,
    neutral_pitch=None,
    neutral_yaw=None,
    neutral_roll=None,
    deviation_pitch=None,
    deviation_yaw=None,
    deviation_roll=None,
):
    """
    Classify head direction given smoothed angles and calibrated neutral baseline.
    Thresholds default to centralized config parameters when not explicitly passed.
    """
    pitch_offset, yaw_offset, roll_offset = calibrated_angles

    npitch = neutral_pitch if neutral_pitch is not None else HEAD_NEUTRAL_PITCH_THRESHOLD
    nyaw = neutral_yaw if neutral_yaw is not None else HEAD_NEUTRAL_YAW_THRESHOLD
    nroll = neutral_roll if neutral_roll is not None else HEAD_NEUTRAL_ROLL_THRESHOLD

    dpitch = deviation_pitch if deviation_pitch is not None else HEAD_DEVIATION_PITCH_THRESHOLD
    dyaw = deviation_yaw if deviation_yaw is not None else HEAD_DEVIATION_YAW_THRESHOLD
    droll = deviation_roll if deviation_roll is not None else HEAD_DEVIATION_ROLL_THRESHOLD

    if (abs(yaw - yaw_offset) <= nyaw and
        abs(pitch - pitch_offset) <= npitch and
        abs(roll - roll_offset) <= nroll):
        return HEAD_SCREEN
    elif yaw < yaw_offset - dyaw:
        return HEAD_LEFT
    elif yaw > yaw_offset + dyaw:
        return HEAD_RIGHT
    elif pitch > pitch_offset + dpitch:
        return HEAD_UP
    elif pitch < pitch_offset - dpitch:
        return HEAD_DOWN
    elif abs(roll - roll_offset) > droll:
        return HEAD_TILTED
    else:
        return prev_state

def analyze_head_pose_from_landmarks(frame, landmarks, calibrated_angles=None, camera_matrix=None):
    """
    Estimates head pose orientation from pre-extracted facial landmarks.
    During calibration (calibrated_angles is None):
        Returns: (frame, (pitch, yaw, roll))
    After calibration:
        Returns: (frame, head_direction_str)
    """
    global previous_state

    if camera_matrix is None:
        camera_matrix = get_camera_matrix(frame.shape)

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
        (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
    ], dtype=np.float64)

    angles = get_head_pose_angles(image_points, camera_matrix=camera_matrix, frame_shape=frame.shape)
    if angles is None:
        if calibrated_angles is None:
            return frame, None
        return frame, previous_state

    pitch = smooth_angle(pitch_history, angles[0])
    yaw = smooth_angle(yaw_history, angles[1])
    roll = smooth_angle(roll_history, angles[2])

    if calibrated_angles is None:
        return frame, (pitch, yaw, roll)

    current_state = classify_head_pose(pitch, yaw, roll, calibrated_angles, previous_state)
    previous_state = current_state
    return frame, current_state

def process_head_pose(frame, calibrated_angles=None, detector=None, predictor=None):
    """
    Processes a frame to estimate head pose direction.
    Backward-compatibility wrapper around face detection and landmark head pose analysis.
    During calibration (calibrated_angles is None):
        Returns: (frame, (pitch, yaw, roll)) if a face is detected, or (frame, None) if no face.
    After calibration:
        Returns: (frame, head_direction_str) where head_direction_str can be:
                 HEAD_SCREEN, HEAD_LEFT, HEAD_RIGHT, HEAD_UP, HEAD_DOWN, HEAD_TILTED, or HEAD_NO_FACE.
    """
    face_detector = detector or get_face_detector()
    landmark_predictor = predictor or get_shape_predictor()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_detector(gray)
    except Exception:
        reset_head_pose_state()
        if calibrated_angles is None:
            return frame, None
        return frame, "ERROR"

    # Handle missing face
    if not faces or len(faces) == 0:
        reset_head_pose_state()
        if calibrated_angles is None:
            return frame, None
        return frame, HEAD_NO_FACE

    head_direction = HEAD_SCREEN
    camera_matrix = get_camera_matrix(frame.shape)

    for face in faces:
        try:
            landmarks = landmark_predictor(gray, face)
        except Exception:
            reset_head_pose_state()
            if calibrated_angles is None:
                return frame, None
            return frame, "ERROR"
        frame, result = analyze_head_pose_from_landmarks(
            frame, landmarks, calibrated_angles=calibrated_angles, camera_matrix=camera_matrix
        )
        head_direction = result

    return frame, head_direction

