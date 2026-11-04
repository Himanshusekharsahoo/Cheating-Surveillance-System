"""
Comprehensive Unit and Integration Tests for Phase 10.13.4.2:
Corrective Mathematical Transformation, solvePnP Hardening, and State-Isolation Hardening.
"""

import math
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from gaze_fusion import (
    HeadPoseResult,
    EyeGazeResult,
    CombinedGazeResult,
    validate_vector3d,
    is_zero_vector,
    normalize_vector3d,
    validate_rotation_matrix,
    camera_gaze_to_angles,
    angles_to_camera_gaze,
    compute_camera_head_rotation_matrix,
    eye_direction_to_head_vector,
    fuse_head_and_eye_gaze,
)
from head_pose import (
    get_head_pose_estimation,
    get_head_pose_angles,
    get_camera_matrix,
    model_points,
)
import eye_movement
from vision_pipeline import VisionPipeline, FaceAnalysisResult
from single_eye_fallback import GazeTrackingResult

pytestmark = pytest.mark.unit


# ===========================================================================
# 1. Rotation Matrix Tests
# ===========================================================================

class TestRotationMatrix:
    """Mathematical validation for 3D rotation matrices."""

    def test_identity_matrix(self):
        """Identity matrix is valid, orthogonal, with det = 1.0."""
        I = np.eye(3, dtype=np.float64)
        assert validate_rotation_matrix(I) is True

    def test_valid_yaw_rotations(self):
        """Valid yaw rotations around camera Y axis."""
        for angle_deg in [10.0, 30.0, 45.0, -25.0, -60.0]:
            rad = math.radians(angle_deg)
            Ry = np.array([
                [math.cos(rad), 0.0, math.sin(rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(rad), 0.0, math.cos(rad)],
            ], dtype=np.float64)
            assert validate_rotation_matrix(Ry) is True
            assert abs(np.linalg.det(Ry) - 1.0) < 1e-6
            assert np.allclose(Ry.T @ Ry, np.eye(3), atol=1e-6)

    def test_valid_pitch_rotations(self):
        """Valid pitch rotations around camera X axis."""
        for angle_deg in [15.0, 35.0, -10.0, -45.0]:
            rad = math.radians(angle_deg)
            Rx = np.array([
                [1.0, 0.0, 0.0],
                [0.0, math.cos(rad), -math.sin(rad)],
                [0.0, math.sin(rad), math.cos(rad)],
            ], dtype=np.float64)
            assert validate_rotation_matrix(Rx) is True
            assert abs(np.linalg.det(Rx) - 1.0) < 1e-6
            assert np.allclose(Rx.T @ Rx, np.eye(3), atol=1e-6)

    def test_valid_roll_rotations(self):
        """Valid roll rotations around camera Z axis."""
        for angle_deg in [5.0, 20.0, -15.0, -40.0]:
            rad = math.radians(angle_deg)
            Rz = np.array([
                [math.cos(rad), -math.sin(rad), 0.0],
                [math.sin(rad), math.cos(rad), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            assert validate_rotation_matrix(Rz) is True
            assert abs(np.linalg.det(Rz) - 1.0) < 1e-6

    def test_compound_rotations_orthogonality_and_determinant(self):
        """Compound Euler rotations preserve orthogonality and det = 1."""
        y_rad = math.radians(25.0)
        p_rad = math.radians(-15.0)
        r_rad = math.radians(10.0)

        Ry = np.array([
            [math.cos(y_rad), 0, math.sin(y_rad)],
            [0, 1, 0],
            [-math.sin(y_rad), 0, math.cos(y_rad)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(p_rad), -math.sin(p_rad)],
            [0, math.sin(p_rad), math.cos(p_rad)],
        ])
        Rz = np.array([
            [math.cos(r_rad), -math.sin(r_rad), 0],
            [math.sin(r_rad), math.cos(r_rad), 0],
            [0, 0, 1],
        ])

        R_compound = Ry @ Rx @ Rz
        assert validate_rotation_matrix(R_compound) is True
        assert abs(np.linalg.det(R_compound) - 1.0) < 1e-5
        assert np.allclose(R_compound.T @ R_compound, np.eye(3), atol=1e-5)
        assert np.allclose(R_compound @ R_compound.T, np.eye(3), atol=1e-5)

    def test_invalid_determinant_rejected(self):
        """Matrices with det != 1 (e.g. scaling or reflection) are rejected."""
        R_refl = np.diag([1.0, 1.0, -1.0])
        assert validate_rotation_matrix(R_refl) is False
        R_scaled = 2.0 * np.eye(3)
        assert validate_rotation_matrix(R_scaled) is False

    def test_non_orthogonal_matrix_rejected(self):
        """Sheared or non-orthogonal matrices are rejected."""
        R_sheared = np.array([
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert validate_rotation_matrix(R_sheared) is False

    def test_nan_and_inf_matrix_rejected(self):
        """Matrices containing NaN or Inf values are safely rejected."""
        R_nan = np.eye(3)
        R_nan[0, 1] = np.nan
        assert validate_rotation_matrix(R_nan) is False

        R_inf = np.eye(3)
        R_inf[2, 2] = np.inf
        assert validate_rotation_matrix(R_inf) is False

    def test_invalid_shape_and_types_rejected(self):
        """Malformed dimensions or non-matrix inputs are safely rejected."""
        assert validate_rotation_matrix(None) is False
        assert validate_rotation_matrix(np.zeros((2, 2))) is False
        assert validate_rotation_matrix(np.zeros((3, 4))) is False
        assert validate_rotation_matrix("not_a_matrix") is False


# ===========================================================================
# 2. Vector Transformation and Normalization Tests
# ===========================================================================

class TestVectorTransformation:
    """Mathematical validation of 3D vector algebra and camera gaze angles."""

    def test_neutral_forward_vector_angles(self):
        """Forward neutral vector [0, 0, 1] maps to yaw = 0.0, pitch = 0.0."""
        v_forward = np.array([0.0, 0.0, 1.0])
        yaw, pitch = camera_gaze_to_angles(v_forward)
        assert yaw == 0.0
        assert pitch == 0.0

    def test_horizontal_yaw_angles_independent(self):
        """Left and right unit vectors produce correct yaw signs using trigonometric ground truth."""
        # Right: 30 degrees right -> x = sin(30) = 0.5, z = cos(30) = sqrt(3)/2
        v_right = np.array([0.5, 0.0, math.sqrt(3.0) / 2.0])
        yaw, pitch = camera_gaze_to_angles(v_right)
        assert abs(yaw - 30.0) < 0.01
        assert abs(pitch - 0.0) < 0.01

        # Left: 45 degrees left -> x = -sin(45) = -sqrt(2)/2, z = cos(45) = sqrt(2)/2
        v_left = np.array([-math.sqrt(2.0) / 2.0, 0.0, math.sqrt(2.0) / 2.0])
        yaw, pitch = camera_gaze_to_angles(v_left)
        assert abs(yaw - (-45.0)) < 0.01
        assert abs(pitch - 0.0) < 0.01

    def test_vertical_pitch_angles_independent(self):
        """Up and down unit vectors produce correct pitch signs (OpenCV -Y is up)."""
        # Up: 30 degrees up -> y = -sin(30) = -0.5, z = cos(30) = sqrt(3)/2
        v_up = np.array([0.0, -0.5, math.sqrt(3.0) / 2.0])
        yaw, pitch = camera_gaze_to_angles(v_up)
        assert abs(yaw - 0.0) < 0.01
        assert abs(pitch - 30.0) < 0.01

        # Down: 30 degrees down -> y = +sin(30) = +0.5, z = cos(30) = sqrt(3)/2
        v_down = np.array([0.0, 0.5, math.sqrt(3.0) / 2.0])
        yaw, pitch = camera_gaze_to_angles(v_down)
        assert abs(yaw - 0.0) < 0.01
        assert abs(pitch - (-30.0)) < 0.01

    def test_vector_normalization(self):
        """Unnormalized valid vectors normalize accurately to unit norm."""
        v = np.array([3.0, 0.0, 4.0])
        normed = normalize_vector3d(v)
        assert normed is not None
        assert np.allclose(normed, np.array([0.6, 0.0, 0.8]), atol=1e-6)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6

    def test_zero_vector_handling(self):
        """Zero-length vectors are recognized and normalize to None."""
        v_zero = np.array([0.0, 0.0, 0.0])
        assert is_zero_vector(v_zero) is True
        assert normalize_vector3d(v_zero) is None
        yaw, pitch = camera_gaze_to_angles(v_zero)
        assert yaw is None and pitch is None

    def test_invalid_and_nonfinite_vectors(self):
        """Vectors with NaN, Inf, or wrong dimensionality return None."""
        assert validate_vector3d([np.nan, 0.0, 1.0]) is False
        assert normalize_vector3d([np.nan, 0.0, 1.0]) is None
        assert validate_vector3d([0.0, np.inf, 1.0]) is False
        assert normalize_vector3d([0.0, np.inf, 1.0]) is None
        assert validate_vector3d([1.0, 2.0]) is False
        assert validate_vector3d(None) is False

    def test_hand_calculated_representative_rotations(self):
        """
        Independent verification against hand-calculated geometric values:
        1. 90-degree yaw around Y:
           g_head = [0, 0, 1] -> [1, 0, 0].
           atan2(1, 0) = +90 degrees.
        2. Compound 45 deg yaw, 30 deg pitch:
           x = sin(45)*cos(30) = sqrt(2)/2 * sqrt(3)/2 = sqrt(6)/4 = 0.6123724
           y = -sin(30) = -0.5
           z = cos(45)*cos(30) = 0.6123724
        """
        v_90 = np.array([1.0, 0.0, 0.0])
        y90, p90 = camera_gaze_to_angles(v_90)
        assert y90 == 90.0
        assert p90 == 0.0

        x_hand = math.sqrt(6.0) / 4.0
        y_hand = -0.5
        z_hand = math.sqrt(6.0) / 4.0
        v_compound = np.array([x_hand, y_hand, z_hand])
        assert abs(np.linalg.norm(v_compound) - 1.0) < 1e-6
        y_calc, p_calc = camera_gaze_to_angles(v_compound)
        assert abs(y_calc - 45.0) < 0.01
        assert abs(p_calc - 30.0) < 0.01


# ===========================================================================
# 3. Eye Gaze Head Vector Mapping Tests
# ===========================================================================

class TestEyeDirectionToHeadVector:
    """Verifies mapping of pupil displacement and canonical directions to head-local vectors."""

    def test_canonical_directions_head_vectors(self):
        """Canonical gaze strings map to documented head-local 3D unit vectors."""
        v_center = eye_direction_to_head_vector("Looking Center")
        assert np.allclose(v_center, [0.0, 0.0, 1.0], atol=1e-5)

        v_right = eye_direction_to_head_vector("Looking Right")
        assert v_right[0] > 0.0 and v_right[1] == 0.0 and v_right[2] > 0.0

        v_left = eye_direction_to_head_vector("Looking Left")
        assert v_left[0] < 0.0 and v_left[1] == 0.0 and v_left[2] > 0.0

        v_up = eye_direction_to_head_vector("Looking Up")
        assert v_up[0] == 0.0 and v_up[1] < 0.0 and v_up[2] > 0.0

        v_down = eye_direction_to_head_vector("Looking Down")
        assert v_down[0] == 0.0 and v_down[1] > 0.0 and v_down[2] > 0.0

    def test_continuous_differential_head_vectors(self):
        """Continuous pupil differentials project smoothly into 3D."""
        v = eye_direction_to_head_vector("Looking Right", diff_x=0.10, diff_y=0.0)
        assert v is not None
        assert v[0] > 0.0
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

        v_up = eye_direction_to_head_vector("Looking Up", diff_x=0.0, diff_y=-0.08)
        assert v_up is not None
        assert v_up[1] < 0.0

    def test_uncertain_and_invalid_returns_none(self):
        """Uncertain or missing gaze strings return None for 3D vector."""
        assert eye_direction_to_head_vector("Uncertain Gaze") is None
        assert eye_direction_to_head_vector("No Face Detected") is None
        assert eye_direction_to_head_vector("Calibrating") is None
        assert eye_direction_to_head_vector(None) is None


# ===========================================================================
# 4. Correct 3D Head-Eye Fusion Tests (ISSUE-01 Verification)
# ===========================================================================

class TestHeadEyeFusion:
    """Verifies mathematically correct 3D fusion using similarity reflection Sz @ R_pnp @ Sz."""

    @staticmethod
    def _make_pnp_matrix(pitch_deg, yaw_deg, roll_deg):
        """Helper constructing R_pnp from Tait-Bryan Euler angles in solvePnP convention."""
        cp = math.radians(-pitch_deg)
        cy = math.radians(-yaw_deg)
        cr = math.radians(roll_deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(cp), -math.sin(cp)], [0, math.sin(cp), math.cos(cp)]])
        Ry = np.array([[math.cos(cy), 0, math.sin(cy)], [0, 1, 0], [-math.sin(cy), 0, math.cos(cy)]])
        Rz = np.array([[math.cos(cr), -math.sin(cr), 0], [math.sin(cr), math.cos(cr), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def test_neutral_head_and_neutral_eye(self):
        """Neutral frontal head + neutral eyes produces straight-ahead camera gaze."""
        R_pnp = self._make_pnp_matrix(0, 0, 0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.95, status="VALID")
        eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.90, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
        comb = fuse_head_and_eye_gaze(head, eye)

        assert comb.status == "VALID"
        assert comb.gaze_vector_camera == (0.0, 0.0, 1.0)
        assert comb.yaw == 0.0
        assert comb.pitch == 0.0

    def test_pure_yaw_positive_and_negative(self):
        """Pure yaw turns produce correct positive (right) and negative (left) camera gaze angles."""
        for y_deg in [20.0, -25.0]:
            R_pnp = self._make_pnp_matrix(0, y_deg, 0)
            R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
            head = HeadPoseResult(yaw=y_deg, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")
            eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
            comb = fuse_head_and_eye_gaze(head, eye)
            assert abs(comb.yaw - y_deg) < 0.05
            assert abs(comb.pitch - 0.0) < 0.05

    def test_pure_pitch_positive_and_negative(self):
        """Pure pitch turns produce correct positive (up) and negative (down) camera gaze angles."""
        for p_deg in [18.0, -15.0]:
            R_pnp = self._make_pnp_matrix(p_deg, 0, 0)
            R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
            head = HeadPoseResult(yaw=0.0, pitch=p_deg, roll=0.0, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")
            eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
            comb = fuse_head_and_eye_gaze(head, eye)
            assert abs(comb.yaw - 0.0) < 0.05
            assert abs(comb.pitch - p_deg) < 0.05

    def test_pure_roll_leaves_forward_gaze_invariant(self):
        """Pure roll around optical axis leaves forward line of sight invariant."""
        for r_deg in [15.0, -30.0, 45.0]:
            R_pnp = self._make_pnp_matrix(0, 0, r_deg)
            R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
            head = HeadPoseResult(yaw=0.0, pitch=0.0, roll=r_deg, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")
            eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
            comb = fuse_head_and_eye_gaze(head, eye)
            assert abs(comb.yaw - 0.0) < 0.05
            assert abs(comb.pitch - 0.0) < 0.05

    def test_compound_yaw_pitch_roll_hand_calculated_audit_case8(self):
        """
        Independent verification of Audit Case 8:
        Head Pose: yaw = +25 deg, pitch = +15 deg, roll = +10 deg, eye neutral [0, 0, 1].
        Hand-calculated expected camera vector: [0.4470, -0.1840, 0.8754]
        Expected angles: yaw = 27.05 deg, pitch = 10.60 deg.
        """
        R_pnp = self._make_pnp_matrix(15.0, 25.0, 10.0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head = HeadPoseResult(yaw=25.0, pitch=15.0, roll=10.0, rotation_matrix=R_cam_head, confidence=0.92, status="VALID")
        eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.88, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
        comb = fuse_head_and_eye_gaze(head, eye)

        assert abs(comb.yaw - 27.05) < 0.05
        assert abs(comb.pitch - 10.60) < 0.05
        assert np.allclose(comb.gaze_vector_camera, [0.446960, -0.184001, 0.875426], atol=1e-4)

    def test_deflected_eye_under_rolled_head_audit_case9(self):
        """
        Independent verification of Audit Case 9 (Critical Defect in R_pnp.T resolved by Sz R_pnp Sz):
        Head Pose: yaw = +30 deg, pitch = 0 deg, roll = +45 deg.
        Eye deflected UP by 20 deg: v_eye = [0, -sin(20), cos(20)].
        Hand-calculated expected camera vector: [0.5741, 0.0904, 0.8138]
        Expected angles: yaw = 35.20 deg, pitch = -5.19 deg.
        """
        R_pnp = self._make_pnp_matrix(0.0, 30.0, 45.0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head = HeadPoseResult(yaw=30.0, pitch=0.0, roll=45.0, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")

        rad20 = math.radians(20.0)
        v_eye_up = np.array([0.0, -math.sin(rad20), math.cos(rad20)])
        eye = EyeGazeResult(selected_eye_direction="Looking Up", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=v_eye_up)
        comb = fuse_head_and_eye_gaze(head, eye)

        # Under the correct similarity reflection, roll and pitch couple correctly:
        assert abs(comb.yaw - 35.20) < 0.05
        assert abs(comb.pitch - (-5.19)) < 0.05
        assert np.allclose(comb.gaze_vector_camera, [0.574076, 0.090387, 0.813798], atol=1e-4)

    def test_compensatory_opposing_head_and_eye_movement(self):
        """Head turned +15 deg right and eyes looking -15 deg left cancel to 0 deg camera gaze."""
        R_pnp = self._make_pnp_matrix(0, 15.0, 0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head = HeadPoseResult(yaw=15.0, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")
        v_eye_left = eye_direction_to_head_vector("Looking Left")
        eye = EyeGazeResult(selected_eye_direction="Looking Left", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=v_eye_left)
        comb = fuse_head_and_eye_gaze(head, eye)

        assert abs(comb.yaw - 0.0) < 0.1
        assert abs(comb.pitch - 0.0) < 0.1

    def test_single_eye_fallback_fusion(self):
        """Single-eye fallback state produces valid fused gaze with discounted confidence."""
        R_pnp = self._make_pnp_matrix(0, 0, 0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")
        eye_fallback = EyeGazeResult(
            left_eye_direction=None,
            right_eye_direction="Looking Center",
            selected_eye_direction="Looking Center",
            confidence=0.72,
            tracking_state="RIGHT_EYE_ONLY",
            status="VALID",
            is_single_eye_fallback=True,
            gaze_vector_head=np.array([0.0, 0.0, 1.0]),
        )
        comb = fuse_head_and_eye_gaze(head, eye_fallback)

        assert comb.status == "VALID"
        assert comb.yaw == 0.0
        assert comb.confidence == round(0.90 * 0.72, 4)
        assert comb.source_components["is_single_eye_fallback"] is True

    def test_missing_head_pose_returns_uncertain(self):
        """Missing head pose produces explicit NO_FACE/UNCERTAIN."""
        eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.80, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
        comb_none = fuse_head_and_eye_gaze(None, eye)
        assert comb_none.status == "NO_FACE"
        assert comb_none.confidence == 0.0

    def test_uncertain_eye_gaze_returns_uncertain(self):
        """Uncertain eye gaze tracking produces UNCERTAIN combined gaze."""
        R_pnp = self._make_pnp_matrix(0, 0, 0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.90, status="VALID")
        eye_uncertain = EyeGazeResult(selected_eye_direction="Uncertain Gaze", confidence=0.0, tracking_state="UNCERTAIN", status="UNCERTAIN", gaze_vector_head=None)
        comb = fuse_head_and_eye_gaze(head, eye_uncertain)
        assert comb.status == "UNCERTAIN"
        assert comb.gaze_vector_camera is None

    def test_invalid_rotation_matrix_returns_invalid_status(self):
        """Corrupted or non-orthogonal rotation matrix produces INVALID_ROTATION."""
        head_corrupt = HeadPoseResult(
            yaw=10.0, pitch=0.0, roll=0.0,
            rotation_matrix=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            confidence=0.80, status="VALID"
        )
        eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
        comb = fuse_head_and_eye_gaze(head_corrupt, eye)
        assert comb.status == "INVALID_ROTATION"

    def test_low_confidence_components_do_not_produce_high_confidence_fusion(self):
        """Unreliable components cannot produce confident fusion."""
        R_pnp = self._make_pnp_matrix(0, 0, 0)
        R_cam_head = compute_camera_head_rotation_matrix(R_pnp)
        head_low = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.30, status="VALID")
        eye_low = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.30, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))
        comb = fuse_head_and_eye_gaze(head_low, eye_low)
        assert comb.confidence == round(0.30 * 0.30, 4)
        assert comb.confidence < 0.10


# ===========================================================================
# 5. solvePnP Hardening Tests (ISSUE-03 and ISSUE-05 Verification)
# ===========================================================================

class TestSolvePnPResilience:
    """Verifies exception handling, input validation, and confidence fallbacks in get_head_pose_estimation."""

    def test_collinear_or_zero_variance_landmarks_safe(self):
        """Zero-variance or identical landmark coordinates return None safely without cv2.error (ISSUE-03)."""
        degenerate_pts = np.zeros((6, 2), dtype=np.float64)
        cam = get_camera_matrix((480, 640))
        result = get_head_pose_estimation(degenerate_pts, camera_matrix=cam)
        assert result is None

    def test_malformed_and_nonfinite_landmarks_safe(self):
        """Non-finite or wrong-shape landmarks return None safely."""
        cam = get_camera_matrix((480, 640))
        nan_pts = np.ones((6, 2), dtype=np.float64)
        nan_pts[0, 0] = np.nan
        assert get_head_pose_estimation(nan_pts, camera_matrix=cam) is None

        short_pts = np.ones((4, 2), dtype=np.float64)
        assert get_head_pose_estimation(short_pts, camera_matrix=cam) is None

        none_pts = None
        assert get_head_pose_estimation(none_pts, camera_matrix=cam) is None

    def test_reprojection_exception_fallback_zero_confidence(self, monkeypatch):
        """When cv2.projectPoints raises an exception, confidence falls back to 0.0, not 0.85 (ISSUE-05)."""
        cam = get_camera_matrix((480, 640))
        # Valid synthetic projection
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([[0.0], [0.0], [600.0]])
        import cv2
        img_pts, _ = cv2.projectPoints(model_points, rvec, tvec, cam, np.zeros((4, 1)))
        img_pts = img_pts.reshape(-1, 2)

        # Monkeypatch projectPoints to raise an exception during reprojection evaluation
        def mock_project_points(*args, **kwargs):
            raise cv2.error("Simulated projection failure")

        monkeypatch.setattr(cv2, "projectPoints", mock_project_points)
        est = get_head_pose_estimation(img_pts, camera_matrix=cam)
        assert est is not None
        angles, R_cam, t, conf = est
        assert conf == 0.0  # Must be 0.0, never 0.85!


# ===========================================================================
# 6. Eye-Tracking State-Isolation Regression Tests (ISSUE-02 Verification)
# ===========================================================================

class TestTrackerStateIsolation:
    """Verifies that invalid or degenerate frames never inherit stale tracking state or confidence."""

    def test_valid_frame_followed_by_degenerate_landmarks_clears_state(self):
        """
        Critical Regression Test for ISSUE-02:
        Frame 1: Valid paired tracking (90% confidence).
        Frame 2: Face detected, but degenerate eye landmarks (width/height <= 0).
        Asserts: Frame 2 eye_gaze receives UNCERTAIN status and 0.0 confidence, never inheriting stale PAIRED state.
        """
        pipeline = VisionPipeline(detector=MagicMock(), predictor=MagicMock())

        # Frame 1: Valid tracking result cached
        eye_movement._last_gaze_tracking_result = GazeTrackingResult(
            gaze_direction="Looking Center",
            tracking_state="PAIRED_TRACKING",
            left_confidence=0.90,
            right_confidence=0.90,
            combined_confidence=0.90,
            is_single_eye_fallback=False
        )

        # Frame 2: Face detected
        mock_rect = MagicMock()
        mock_rect.left.return_value = 100
        mock_rect.top.return_value = 100
        mock_rect.right.return_value = 300
        mock_rect.bottom.return_value = 300
        pipeline._detector = MagicMock(return_value=[mock_rect])

        # Valid head pose points for solvePnP, but collapsed eyes (x=100, y=100)
        pts_dict = {
            30: (200, 200), 8: (200, 265),
            36: (100, 100), 45: (100, 100),  # collapsed
            48: (170, 240), 54: (230, 240)
        }
        mock_lm = MagicMock()
        mock_lm.part.side_effect = lambda idx: MagicMock(x=pts_dict.get(idx, (100, 100))[0], y=pts_dict.get(idx, (100, 100))[1])
        pipeline._predictor = MagicMock(return_value=mock_lm)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _ = pipeline.process_face(frame)

        # Assert no stale tracking leaked:
        assert result.gaze_direction == "Uncertain Gaze"
        assert result.eye_gaze.status == "UNCERTAIN"
        assert result.eye_gaze.confidence == 0.0
        assert result.eye_gaze.tracking_state == "UNCERTAIN"
        assert result.combined_gaze.status == "UNCERTAIN"
        assert result.combined_gaze.confidence == 0.0

    def test_reset_default_gaze_tracker_clears_cached_tracking_result(self):
        """reset_default_gaze_tracker explicitly clears _last_gaze_tracking_result."""
        eye_movement._last_gaze_tracking_result = "STALE_DATA"
        eye_movement._last_gaze_sample = "STALE_SAMPLE"
        eye_movement._last_gaze_diagnostics = {"stale": True}

        eye_movement.reset_default_gaze_tracker()

        assert eye_movement.get_last_gaze_tracking_result() is None
        assert eye_movement.get_last_gaze_sample() is None
        assert eye_movement.get_last_gaze_diagnostics() is None

    def test_process_eye_movement_zero_faces_clears_tracking_result(self):
        """process_eye_movement with 0 faces resets tracker and clears all module globals."""
        eye_movement._last_gaze_tracking_result = "STALE_DATA"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector_mock = MagicMock(return_value=[])

        frame_out, direction = eye_movement.process_eye_movement(
            frame, detector=detector_mock, predictor=MagicMock()
        )
        assert direction == "No Face Detected"
        assert eye_movement.get_last_gaze_tracking_result() is None

    def test_independent_pipeline_instances_do_not_leak_stale_tracking(self):
        """Two independent VisionPipeline instances do not inherit stale tracking results across calls."""
        pipe1 = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock())
        pipe2 = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock())

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res1, _ = pipe1.process_face(frame)
        res2, _ = pipe2.process_face(frame)

        assert res1.eye_gaze.status == "NO_FACE"
        assert res2.eye_gaze.status == "NO_FACE"


# ===========================================================================
# 7. Structured Result Models & Backward Compatibility Tests
# ===========================================================================

class TestStructuredModelsAndCompatibility:
    """Verifies dataclass fields, property aliases, and backward compatibility."""

    def test_head_pose_result_property_aliases(self):
        """HeadPoseResult provides both primary and legacy property aliases."""
        R = np.eye(3)
        t = np.array([[0.0], [0.0], [600.0]])
        hp = HeadPoseResult(
            yaw=12.5, pitch=-8.0, roll=3.2,
            rotation_matrix=R, translation=t,
            confidence=0.91, status="VALID"
        )
        assert hp.head_yaw == 12.5
        assert hp.head_pitch == -8.0
        assert hp.head_roll == 3.2
        assert hp.head_rotation_matrix is R
        assert hp.head_translation is t
        assert hp.head_confidence == 0.91
        assert hp.head_tracking_status == "VALID"
        assert hp.coordinate_frame == "camera_opencv"

    def test_eye_gaze_result_property_aliases(self):
        """EyeGazeResult provides both primary and legacy property aliases."""
        eg = EyeGazeResult(
            left_eye_direction="Looking Left",
            right_eye_direction="Looking Center",
            selected_eye_direction="Looking Left",
            confidence=0.74,
            tracking_state="LEFT_EYE_ONLY",
            status="VALID"
        )
        assert eg.eye_direction == "Looking Left"
        assert eg.eye_confidence == 0.74
        assert eg.eye_tracking_state == "LEFT_EYE_ONLY"
        assert eg.eye_tracking_status == "VALID"
        assert eg.coordinate_frame == "head_local"

    def test_combined_gaze_result_property_aliases(self):
        """CombinedGazeResult provides both primary and legacy property aliases."""
        cg = CombinedGazeResult(
            gaze_vector_camera=(0.25, 0.0, 0.96),
            yaw=14.5, pitch=-2.1,
            confidence=0.78, status="VALID", uncertainty=0.22,
            source_components={"sample": 1}
        )
        assert cg.camera_gaze_vector == (0.25, 0.0, 0.96)
        assert cg.camera_gaze_yaw == 14.5
        assert cg.camera_gaze_pitch == -2.1
        assert cg.combined_gaze_confidence == 0.78
        assert cg.combined_gaze_status == "VALID"
        assert cg.combined_gaze_uncertainty == 0.22

    def test_face_analysis_result_backward_compatibility(self):
        """Existing FaceAnalysisResult consumers can instantiate without new fields."""
        res = FaceAnalysisResult(
            face_count=1,
            gaze_direction="Looking Center",
            head_direction="Looking Screen",
            raw_head_pose=(-2.0, 5.0, 1.0)
        )
        assert res.face_count == 1
        assert res.gaze_direction == "Looking Center"
        assert res.head_direction == "Looking Screen"
        assert res.raw_head_pose == (-2.0, 5.0, 1.0)
        assert res.head_pose is None
        assert res.eye_gaze is None
        assert res.combined_gaze is None
        assert res.screen_gaze is None


# ===========================================================================
# 8. Pipeline Integration Tests
# ===========================================================================

class TestPipelineIntegration:
    """Verifies that VisionPipeline populates all three structured outputs."""

    def test_pipeline_zero_faces_populates_empty_structured_models(self):
        """Empty frame or 0 faces produces FaceAnalysisResult with NO_FACE structured models."""
        pipeline = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _ = pipeline.process_face(frame)

        assert result.face_count == 0
        assert result.head_pose.status == "NO_FACE"
        assert result.head_pose.confidence == 0.0
        assert result.eye_gaze.status == "NO_FACE"
        assert result.eye_gaze.confidence == 0.0
        assert result.combined_gaze.status == "NO_FACE"
        assert result.combined_gaze.confidence == 0.0

    def test_pipeline_face_detected_populates_three_outputs(self, monkeypatch):
        """When face is detected, pipeline constructs HeadPoseResult, EyeGazeResult, CombinedGazeResult."""
        mock_rect = MagicMock()
        mock_rect.left.return_value = 100
        mock_rect.top.return_value = 100
        mock_rect.right.return_value = 300
        mock_rect.bottom.return_value = 300

        mock_landmarks = MagicMock()
        pts_dict = {
            30: (200, 200), 8: (200, 265),
            36: (150, 150), 45: (250, 150),
            48: (170, 240), 54: (230, 240)
        }
        for pt_idx in range(68):
            mock_pt = MagicMock()
            mock_pt.x, mock_pt.y = pts_dict.get(pt_idx, (150 + pt_idx, 150 + pt_idx))
            mock_landmarks.part.return_value = mock_pt

        pipeline = VisionPipeline(
            detector=MagicMock(return_value=[mock_rect]),
            predictor=MagicMock(return_value=mock_landmarks)
        )

        R_dummy = np.eye(3)
        t_dummy = np.array([[0.0], [0.0], [600.0]])
        monkeypatch.setattr(
            "vision_pipeline.get_head_pose_estimation",
            lambda *args, **kwargs: ((0.0, 5.0, 0.0), R_dummy, t_dummy, 0.90)
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_head_pose_from_landmarks",
            lambda frame, lm, **kwargs: (frame, "Looking Screen")
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_gaze_from_landmarks",
            lambda frame, lm, **kwargs: (frame, "Looking Center")
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert result.face_count == 1
        assert result.head_pose is not None
        assert result.head_pose.status == "VALID"
        assert result.head_pose.yaw == 5.0

        assert result.eye_gaze is not None
        assert result.eye_gaze.selected_eye_direction == "Looking Center"

        assert result.combined_gaze is not None
        assert result.combined_gaze.status in ("VALID", "UNCERTAIN")
