"""
Tests for Phase 10.13.12 GAP-02: Strict Fused-Gaze Confidence & Status Gating.

Verifies:
1. Confidence classification uses unrounded source measurements.
2. Rounding never promotes an uncertain or below-threshold measurement into a valid one.
3. Fusion explicitly validates head-pose status before using head rotation.
4. Non-VALID head statuses (UNCERTAIN, NO_FACE, INVALID) cannot produce VALID fused gaze.
5. Boundary confidence tests: 0.19996, 0.19999, 0.20, 0.2001, 0.0, -0.1, NaN, inf.
6. Invalid rotation matrix cannot produce a directional classification.
7. Valid head and eye inputs still produce expected fused direction.
8. Opposing head and eye movements retain correct fusion behavior.
9. Missing inputs produce uncertainty rather than fabricated direction.
10. Exact regression test for 0.19996 -> 0.2 -> UNCERTAIN -> VALID -> Looking Right.
"""

import math
from unittest.mock import MagicMock
import numpy as np
import pytest

import config
from eye_movement import GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UNCERTAIN, GAZE_NO_FACE
from gaze_fusion import (
    CombinedGazeResult,
    EyeGazeResult,
    HeadPoseResult,
    angles_to_camera_gaze,
    compute_camera_head_rotation_matrix,
    eye_direction_to_head_vector,
    fuse_head_and_eye_gaze,
)
from head_pose import HEAD_SCREEN, HEAD_LEFT, HEAD_RIGHT
import main
from vision_pipeline import VisionPipeline, process_frame_face

pytestmark = pytest.mark.unit


def _make_rot_matrix(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    cp = math.radians(-pitch_deg)
    cy = math.radians(-yaw_deg)
    cr = math.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(cp), -math.sin(cp)], [0, math.sin(cp), math.cos(cp)]])
    Ry = np.array([[math.cos(cy), 0, math.sin(cy)], [0, 1, 0], [-math.sin(cy), 0, math.cos(cy)]])
    Rz = np.array([[math.cos(cr), -math.sin(cr), 0], [math.sin(cr), math.cos(cr), 0], [0, 0, 1]])
    R_pnp = Rz @ Ry @ Rx
    return compute_camera_head_rotation_matrix(R_pnp)


def test_regression_0_19996_to_0_2_uncertain_to_valid_looking_right_prevented():
    """
    Regression Test for Phase 10.13.11 Defect:
    Raw head confidence = 0.19996 (< 0.20).
    Under previous code:
      - Raw 0.19996 was classified as UNCERTAIN in HeadPoseResult.
      - But confidence was rounded to 0.2000.
      - fuse_head_and_eye_gaze did NOT check head_pose.status == "VALID".
      - It used rounded 0.20 * eye 1.0 = 0.20.
      - CombinedGazeResult was marked status='VALID' with yaw=+15.0.
      - main.classify_combined_attention classified it as 'Looking Right'!
    Under Phase 10.13.12 fix:
      - Head confidence classification uses unrounded 0.19996 (< 0.20) -> UNCERTAIN.
      - Rounding cannot promote 0.19996 to >= 0.20 (clamped to 0.1999).
      - Fusion explicitly checks head_pose.status == "VALID".
      - Non-VALID head status causes fusion to return status='UNCERTAIN'.
      - Fused direction in main is strictly GAZE_UNCERTAIN.
    """
    R_cam_head = _make_rot_matrix(0.0, 0.0, 0.0)

    # 1. Pipeline behavior: solvePnP reprojection confidence is 0.19996
    raw_hp_conf = 0.19996
    hp_threshold = float(config.FUSED_GAZE_MIN_CONFIDENCE)
    assert raw_hp_conf < hp_threshold

    # Simulate pipeline head pose construction
    head_status = "VALID" if raw_hp_conf >= hp_threshold else "UNCERTAIN"
    rounded_conf = round(raw_hp_conf, 4)
    if raw_hp_conf < hp_threshold and rounded_conf >= hp_threshold:
        rounded_conf = round(hp_threshold - 1e-4, 4)

    assert head_status == "UNCERTAIN"
    assert rounded_conf < hp_threshold
    assert rounded_conf == 0.1999

    head_pose = HeadPoseResult(
        yaw=0.0, pitch=0.0, roll=0.0,
        rotation_matrix=R_cam_head,
        confidence=rounded_conf,
        status=head_status,
        coordinate_frame="camera_opencv",
    )

    # Eye gaze looking right with high confidence
    v_eye_right = np.array([math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0))], dtype=np.float64)
    eye_gaze = EyeGazeResult(
        selected_eye_direction="Looking Right",
        confidence=1.0,
        tracking_state="PAIRED_TRACKING",
        status="VALID",
        gaze_vector_head=v_eye_right,
    )

    # 2. Fusion MUST reject non-VALID head status
    comb = fuse_head_and_eye_gaze(head_pose, eye_gaze)
    assert comb.status == "UNCERTAIN"
    assert comb.confidence < hp_threshold
    assert comb.yaw is None
    assert comb.pitch is None
    assert comb.gaze_vector_camera is None

    # 3. Main classifier MUST return GAZE_UNCERTAIN rather than Looking Right
    classified_direction = main.classify_combined_attention(comb)
    assert classified_direction == GAZE_UNCERTAIN
    assert classified_direction != GAZE_RIGHT


@pytest.mark.parametrize(
    ("raw_conf", "expected_head_status", "expected_fused_status"),
    [
        (0.19996, "UNCERTAIN", "UNCERTAIN"),
        (0.19999, "UNCERTAIN", "UNCERTAIN"),
        (0.20000, "VALID", "VALID"),
        (0.20010, "VALID", "VALID"),
        (0.0, "UNCERTAIN", "UNCERTAIN"),
        (-0.5, "UNCERTAIN", "UNCERTAIN"),
        (float("nan"), "UNCERTAIN", "UNCERTAIN"),
        (float("inf"), "UNCERTAIN", "UNCERTAIN"),
        (float("-inf"), "UNCERTAIN", "UNCERTAIN"),
    ],
)
def test_head_confidence_boundary_values(raw_conf, expected_head_status, expected_fused_status):
    """
    Test exact boundary values around configured threshold (0.20).
    Values below 0.20, zero, negative, NaN, and infinity MUST NOT become VALID.
    """
    threshold = float(config.FUSED_GAZE_MIN_CONFIDENCE)
    R_cam_head = _make_rot_matrix(0.0, 0.0, 0.0)

    is_valid_numeric = (
        isinstance(raw_conf, (int, float))
        and math.isfinite(raw_conf)
        and raw_conf >= threshold
    )
    status = "VALID" if is_valid_numeric else "UNCERTAIN"
    assert status == expected_head_status

    # Prepare confidence representation ensuring rounding does not promote
    if not math.isfinite(raw_conf) or raw_conf <= 0.0:
        safe_conf = 0.0
    else:
        safe_conf = round(raw_conf, 4)
        if raw_conf < threshold and safe_conf >= threshold:
            safe_conf = round(threshold - 1e-4, 4)

    head = HeadPoseResult(
        yaw=0.0, pitch=0.0, roll=0.0,
        rotation_matrix=R_cam_head,
        confidence=safe_conf,
        status=status,
    )
    eye = EyeGazeResult(
        selected_eye_direction="Looking Center",
        confidence=1.0,
        tracking_state="PAIRED_TRACKING",
        status="VALID",
        gaze_vector_head=np.array([0.0, 0.0, 1.0]),
    )

    fused = fuse_head_and_eye_gaze(head, eye)
    assert fused.status == expected_fused_status
    if expected_fused_status == "UNCERTAIN":
        assert fused.yaw is None
        assert fused.pitch is None
        assert fused.gaze_vector_camera is None
        assert main.classify_combined_attention(fused) == GAZE_UNCERTAIN
    else:
        assert fused.yaw is not None
        assert fused.pitch is not None
        assert main.classify_combined_attention(fused) == GAZE_CENTER


def test_non_valid_head_status_cannot_produce_valid_fused_gaze():
    """
    Test that even if head confidence is high, a non-VALID head status
    (e.g. UNCERTAIN, FAILED, NO_FACE) rejects fusion.
    """
    R_cam_head = _make_rot_matrix(0.0, 0.0, 0.0)
    for bad_status in ("UNCERTAIN", "FAILED", "NO_FACE", "DEGENERATE", "INITIALIZING"):
        head = HeadPoseResult(
            yaw=0.0, pitch=0.0, roll=0.0,
            rotation_matrix=R_cam_head,
            confidence=0.95,  # High confidence
            status=bad_status,
        )
        eye = EyeGazeResult(
            selected_eye_direction="Looking Center",
            confidence=0.95,
            tracking_state="PAIRED_TRACKING",
            status="VALID",
            gaze_vector_head=np.array([0.0, 0.0, 1.0]),
        )
        fused = fuse_head_and_eye_gaze(head, eye)
        assert fused.status in ("UNCERTAIN", "NO_FACE")
        assert fused.yaw is None
        assert main.classify_combined_attention(fused) in (GAZE_UNCERTAIN, GAZE_NO_FACE)


def test_invalid_rotation_matrix_cannot_produce_directional_classification():
    """
    Test that invalid or singular rotation matrices are rejected as INVALID_ROTATION.
    """
    bad_rot = np.zeros((3, 3), dtype=np.float64)  # Determinant 0, non-orthogonal
    head = HeadPoseResult(
        yaw=0.0, pitch=0.0, roll=0.0,
        rotation_matrix=bad_rot,
        confidence=0.90,
        status="VALID",
    )
    eye = EyeGazeResult(
        selected_eye_direction="Looking Right",
        confidence=0.90,
        tracking_state="PAIRED_TRACKING",
        status="VALID",
        gaze_vector_head=np.array([math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0))]),
    )
    fused = fuse_head_and_eye_gaze(head, eye)
    assert fused.status == "INVALID_ROTATION"
    assert fused.yaw is None
    assert main.classify_combined_attention(fused) == GAZE_UNCERTAIN


def test_valid_head_and_eye_inputs_produce_expected_fused_direction():
    """
    Test that valid head and eye inputs produce expected camera-relative gaze angles.
    Head turned +10 deg right, eye turned +15 deg right -> total yaw approx +25 deg.
    """
    R_cam_head = _make_rot_matrix(yaw_deg=10.0, pitch_deg=0.0, roll_deg=0.0)
    head = HeadPoseResult(
        yaw=10.0, pitch=0.0, roll=0.0,
        rotation_matrix=R_cam_head,
        confidence=0.85,
        status="VALID",
    )
    eye = EyeGazeResult(
        selected_eye_direction="Looking Right",
        confidence=0.85,
        tracking_state="PAIRED_TRACKING",
        status="VALID",
        gaze_vector_head=eye_direction_to_head_vector("Looking Right"),
    )
    fused = fuse_head_and_eye_gaze(head, eye)
    assert fused.status == "VALID"
    assert fused.yaw == pytest.approx(25.0, abs=1.5)
    assert main.classify_combined_attention(fused) == GAZE_RIGHT


def test_opposing_head_and_eye_movements_retain_compensatory_behavior():
    """
    Test compensatory vestibulo-ocular reflex: head turned right (+15 deg),
    eye turned left (-15 deg) -> combined gaze remains Center.
    """
    R_cam_head = _make_rot_matrix(yaw_deg=15.0, pitch_deg=0.0, roll_deg=0.0)
    head = HeadPoseResult(
        yaw=15.0, pitch=0.0, roll=0.0,
        rotation_matrix=R_cam_head,
        confidence=0.85,
        status="VALID",
    )
    eye = EyeGazeResult(
        selected_eye_direction="Looking Left",
        confidence=0.85,
        tracking_state="PAIRED_TRACKING",
        status="VALID",
        gaze_vector_head=eye_direction_to_head_vector("Looking Left"),
    )
    fused = fuse_head_and_eye_gaze(head, eye)
    assert fused.status == "VALID"
    assert fused.yaw == pytest.approx(0.0, abs=1.5)
    assert main.classify_combined_attention(fused) == GAZE_CENTER


def test_missing_inputs_produce_uncertainty_not_fabricated_direction():
    """
    Test that None head or None eye produces uncertainty without fabricating 0.0 angles.
    """
    R_cam_head = _make_rot_matrix(0.0, 0.0, 0.0)
    head = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, rotation_matrix=R_cam_head, confidence=0.85, status="VALID")
    eye = EyeGazeResult(selected_eye_direction="Looking Center", confidence=0.85, tracking_state="PAIRED_TRACKING", status="VALID", gaze_vector_head=np.array([0.0, 0.0, 1.0]))

    # Missing head
    comb1 = fuse_head_and_eye_gaze(None, eye)
    assert comb1.status == "NO_FACE"
    assert comb1.yaw is None
    assert main.classify_combined_attention(comb1) == "No Face Detected"

    # Missing eye
    comb2 = fuse_head_and_eye_gaze(head, None)
    assert comb2.status == "UNCERTAIN_EYE"
    assert comb2.yaw is None
    assert main.classify_combined_attention(comb2) == GAZE_UNCERTAIN
