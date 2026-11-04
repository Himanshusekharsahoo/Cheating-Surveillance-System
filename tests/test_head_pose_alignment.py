"""
Unit tests for head pose coordinate alignment and solvePnP stability.

Validates that:
1. Neutral frontal head pose is represented consistently around zero degrees (0.0, 0.0, 0.0).
2. Yaw, pitch, and roll remain stable without branch-cut wrapping or false cross-coupling.
3. Physical movements (left, right, up, down, tilt) map to expected anatomical signs.
4. solvePnP yields valid SO(3) rotation matrices (orthogonal, det = +1.0) and positive depth (T_z > 0).
5. Degenerate and invalid landmarks are safely rejected without unhandled exceptions.
"""

import math
import cv2
import numpy as np
import pytest

from head_pose import (
    model_points,
    dist_coeffs,
    get_camera_matrix,
    get_head_pose_angles,
    classify_head_pose,
    HEAD_SCREEN,
    HEAD_LEFT,
    HEAD_RIGHT,
    HEAD_UP,
    HEAD_DOWN,
    HEAD_TILTED,
)

pytestmark = pytest.mark.unit

DEFAULT_CAM_MATRIX = np.array([
    [640.0, 0.0, 320.0],
    [0.0, 640.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


def _project_ground_truth(pitch_deg: float, yaw_deg: float, roll_deg: float, depth: float = 500.0) -> np.ndarray:
    """
    Synthesize ground-truth 2D landmark projections from known 3D Euler angles.
    Camera-frame rotations:
      - Physical Pitch UP is negative rotation about camera X -> cp = -pitch
      - Physical Yaw LEFT is positive rotation about camera Y -> cy = -yaw
      - Physical Roll RIGHT is clockwise rotation about camera Z -> cr = roll
    """
    cp_rad = np.radians(-pitch_deg)
    cy_rad = np.radians(-yaw_deg)
    cr_rad = np.radians(roll_deg)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(cp_rad), -math.sin(cp_rad)], [0.0, math.sin(cp_rad), math.cos(cp_rad)]])
    Ry = np.array([[math.cos(cy_rad), 0.0, math.sin(cy_rad)], [0.0, 1.0, 0.0], [-math.sin(cy_rad), 0.0, math.cos(cy_rad)]])
    Rz = np.array([[math.cos(cr_rad), -math.sin(cr_rad), 0.0], [math.sin(cr_rad), math.cos(cr_rad), 0.0], [0.0, 0.0, 1.0]])
    R_cam = Rz @ Ry @ Rx

    rvec, _ = cv2.Rodrigues(R_cam)
    tvec = np.array([0.0, 0.0, float(depth)], dtype=np.float64)

    pts2d, _ = cv2.projectPoints(model_points, rvec, tvec, DEFAULT_CAM_MATRIX, dist_coeffs)
    return pts2d.reshape(-1, 2)


def test_neutral_frontal_face_angles_and_depth():
    """Verify that a synthetic neutral frontal face yields (0, 0, 0) and positive depth."""
    pts2d = _project_ground_truth(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0, depth=500.0)

    angles = get_head_pose_angles(pts2d, camera_matrix=DEFAULT_CAM_MATRIX)
    assert angles is not None
    pitch, yaw, roll = angles

    assert abs(pitch) < 0.1, f"Expected pitch ~0.0, got {pitch}"
    assert abs(yaw) < 0.1, f"Expected yaw ~0.0, got {yaw}"
    assert abs(roll) < 0.1, f"Expected roll ~0.0, got {roll}"

    # Verify positive depth and rotation matrix validity directly
    ret, rvec, tvec = cv2.solvePnP(model_points, pts2d, DEFAULT_CAM_MATRIX, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    assert ret is True
    assert tvec[2][0] > 0.0, f"Expected positive depth (T_z > 0), got {tvec[2][0]}"
    assert abs(tvec[2][0] - 500.0) < 1.0


def test_rotation_matrix_orthogonality_and_determinant():
    """Verify that solvePnP produces a mathematically valid SO(3) rotation matrix."""
    for p, y, r in [(0.0, 0.0, 0.0), (15.0, -20.0, 5.0), (-10.0, 25.0, -8.0)]:
        pts2d = _project_ground_truth(pitch_deg=p, yaw_deg=y, roll_deg=r)
        ret, rvec, tvec = cv2.solvePnP(model_points, pts2d, DEFAULT_CAM_MATRIX, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        assert ret is True

        R, _ = cv2.Rodrigues(rvec)
        det = np.linalg.det(R)
        orth_err = np.linalg.norm(R.T @ R - np.eye(3))

        assert abs(det - 1.0) < 1e-5, f"Determinant must be ~1.0, got {det}"
        assert orth_err < 1e-5, f"Orthogonality residual must be < 1e-5, got {orth_err}"


def test_head_yaw_left_and_right():
    """Verify that physical head-left yields negative yaw and head-right yields positive yaw."""
    baseline = (0.0, 0.0, 0.0)

    # 1. Turn Left (20 deg)
    pts_left = _project_ground_truth(pitch_deg=0.0, yaw_deg=-20.0, roll_deg=0.0)
    pitch_l, yaw_l, roll_l = get_head_pose_angles(pts_left, camera_matrix=DEFAULT_CAM_MATRIX)
    assert abs(yaw_l - (-20.0)) < 0.5, f"Expected yaw ~ -20.0, got {yaw_l}"
    assert abs(pitch_l) < 0.5, f"Expected pitch decoupling ~ 0.0, got {pitch_l}"
    assert abs(roll_l) < 0.5, f"Expected roll decoupling ~ 0.0, got {roll_l}"
    assert classify_head_pose(pitch_l, yaw_l, roll_l, baseline) == HEAD_LEFT

    # 2. Turn Right (20 deg)
    pts_right = _project_ground_truth(pitch_deg=0.0, yaw_deg=20.0, roll_deg=0.0)
    pitch_r, yaw_r, roll_r = get_head_pose_angles(pts_right, camera_matrix=DEFAULT_CAM_MATRIX)
    assert abs(yaw_r - 20.0) < 0.5, f"Expected yaw ~ +20.0, got {yaw_r}"
    assert abs(pitch_r) < 0.5, f"Expected pitch decoupling ~ 0.0, got {pitch_r}"
    assert abs(roll_r) < 0.5, f"Expected roll decoupling ~ 0.0, got {roll_r}"
    assert classify_head_pose(pitch_r, yaw_r, roll_r, baseline) == HEAD_RIGHT


def test_head_pitch_up_and_down():
    """Verify that physical pitch-up yields positive pitch and pitch-down yields negative pitch."""
    baseline = (0.0, 0.0, 0.0)

    # 1. Pitch Up (15 deg)
    pts_up = _project_ground_truth(pitch_deg=15.0, yaw_deg=0.0, roll_deg=0.0)
    pitch_u, yaw_u, roll_u = get_head_pose_angles(pts_up, camera_matrix=DEFAULT_CAM_MATRIX)
    assert abs(pitch_u - 15.0) < 0.5, f"Expected pitch ~ +15.0, got {pitch_u}"
    assert abs(yaw_u) < 0.5, f"Expected yaw decoupling ~ 0.0, got {yaw_u}"
    assert abs(roll_u) < 0.5, f"Expected roll decoupling ~ 0.0, got {roll_u}"
    assert classify_head_pose(pitch_u, yaw_u, roll_u, baseline) == HEAD_UP

    # 2. Pitch Down (15 deg)
    pts_down = _project_ground_truth(pitch_deg=-15.0, yaw_deg=0.0, roll_deg=0.0)
    pitch_d, yaw_d, roll_d = get_head_pose_angles(pts_down, camera_matrix=DEFAULT_CAM_MATRIX)
    assert abs(pitch_d - (-15.0)) < 0.5, f"Expected pitch ~ -15.0, got {pitch_d}"
    assert abs(yaw_d) < 0.5, f"Expected yaw decoupling ~ 0.0, got {yaw_d}"
    assert abs(roll_d) < 0.5, f"Expected roll decoupling ~ 0.0, got {roll_d}"
    assert classify_head_pose(pitch_d, yaw_d, roll_d, baseline) == HEAD_DOWN


def test_head_roll_left_and_right():
    """Verify that lateral head tilt triggers HEAD_TILTED."""
    baseline = (0.0, 0.0, 0.0)

    # 1. Tilt Right (+12 deg)
    pts_roll_r = _project_ground_truth(pitch_deg=0.0, yaw_deg=0.0, roll_deg=12.0)
    pitch_r, yaw_r, roll_r = get_head_pose_angles(pts_roll_r, camera_matrix=DEFAULT_CAM_MATRIX)
    assert abs(roll_r - 12.0) < 0.5, f"Expected roll ~ +12.0, got {roll_r}"
    assert classify_head_pose(pitch_r, yaw_r, roll_r, baseline) == HEAD_TILTED

    # 2. Tilt Left (-12 deg)
    pts_roll_l = _project_ground_truth(pitch_deg=0.0, yaw_deg=0.0, roll_deg=-12.0)
    pitch_l, yaw_l, roll_l = get_head_pose_angles(pts_roll_l, camera_matrix=DEFAULT_CAM_MATRIX)
    assert abs(roll_l - (-12.0)) < 0.5, f"Expected roll ~ -12.0, got {roll_l}"
    assert classify_head_pose(pitch_l, yaw_l, roll_l, baseline) == HEAD_TILTED


def test_combined_yaw_and_pitch():
    """Verify that combined compound yaw and pitch motions are resolved accurately without cross-talk."""
    pts_combined = _project_ground_truth(pitch_deg=12.0, yaw_deg=-18.0, roll_deg=3.0)
    pitch, yaw, roll = get_head_pose_angles(pts_combined, camera_matrix=DEFAULT_CAM_MATRIX)

    assert abs(pitch - 12.0) < 1.0, f"Expected pitch ~ 12.0, got {pitch}"
    assert abs(yaw - (-18.0)) < 1.0, f"Expected yaw ~ -18.0, got {yaw}"
    assert abs(roll - 3.0) < 1.0, f"Expected roll ~ 3.0, got {roll}"


def test_small_movement_around_neutral_remains_compliant():
    """Verify that small natural jitter around neutral stays classified as HEAD_SCREEN."""
    baseline = (0.0, 0.0, 0.0)

    # Small perturbations: pitch +4, yaw -5, roll +2
    pts_small = _project_ground_truth(pitch_deg=4.0, yaw_deg=-5.0, roll_deg=2.0)
    pitch, yaw, roll = get_head_pose_angles(pts_small, camera_matrix=DEFAULT_CAM_MATRIX)
    assert classify_head_pose(pitch, yaw, roll, baseline) == HEAD_SCREEN


def test_stability_near_branch_boundaries():
    """Verify that angles do not blow up or wrap erroneously near operational bounds."""
    for p in [-35.0, 35.0]:
        for y in [-45.0, 45.0]:
            pts = _project_ground_truth(pitch_deg=p, yaw_deg=y, roll_deg=0.0)
            angles = get_head_pose_angles(pts, camera_matrix=DEFAULT_CAM_MATRIX)
            assert angles is not None
            p_est, y_est, r_est = angles
            assert abs(p_est - p) < 1.5
            assert abs(y_est - y) < 1.5
            assert abs(r_est) < 2.0


def test_missing_or_invalid_facial_landmarks_safe():
    """Verify that None, empty, NaN, or insufficient landmark arrays return None safely."""
    # None
    assert get_head_pose_angles(None) is None

    # Empty
    assert get_head_pose_angles(np.zeros((0, 2))) is None

    # Insufficient points (< 6)
    assert get_head_pose_angles(np.zeros((5, 2))) is None

    # NaN coordinates
    nan_points = np.full((6, 2), np.nan)
    assert get_head_pose_angles(nan_points) is None

    # Inf coordinates
    inf_points = np.full((6, 2), np.inf)
    assert get_head_pose_angles(inf_points) is None


def test_head_forward_vector_direction():
    """Verify that the 3D forward vector points toward the camera lens (-Z in camera space)."""
    pts2d = _project_ground_truth(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)
    ret, rvec, tvec = cv2.solvePnP(model_points, pts2d, DEFAULT_CAM_MATRIX, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    assert ret is True

    R, _ = cv2.Rodrigues(rvec)
    # Model forward vector is [0, 0, -1]^T (pointing forward out of the face)
    forward_cam = R @ np.array([0.0, 0.0, -1.0])

    assert abs(forward_cam[0]) < 1e-4
    assert abs(forward_cam[1]) < 1e-4
    assert abs(forward_cam[2] - (-1.0)) < 1e-4, f"Forward vector should point along -Z, got {forward_cam}"
