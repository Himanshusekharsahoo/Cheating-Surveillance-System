"""
Unit and integration tests for upward gaze detection and eye geometry reliability.
Phase 10.13.2 verification suite.
"""

from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from eye_movement import (
    detect_pupil,
    calculate_gaze_direction,
    calibrate_gaze_baseline,
    analyze_gaze_from_landmarks,
    process_eye_movement,
    get_last_gaze_diagnostics,
    get_last_gaze_sample,
    GAZE_CENTER,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    GAZE_DOWN,
    GAZE_NO_FACE,
    GAZE_UNCERTAIN,
)
import config

pytestmark = pytest.mark.unit


# ==============================================================================
# Helper functions for synthetic eye test generation
# ==============================================================================

def _make_synthetic_eye_crop(h: int, w: int, pupil_y: int, pupil_x: int, pupil_r: int = 2):
    """
    Constructs a synthetic eye crop with a realistic palpebral aperture polygon
    and a dark pupil disc at (pupil_x, pupil_y).
    """
    crop = np.full((h, w, 3), 160, dtype=np.uint8)  # Surrounding skin
    mid_y = h // 2
    poly = np.array([
        [0, mid_y],
        [w // 4, 0],
        [3 * w // 4, 0],
        [w - 1, mid_y],
        [3 * w // 4, h - 1],
        [w // 4, h - 1],
    ], dtype=np.int32)
    # Bright sclera inside the palpebral fissure
    cv2.fillPoly(crop, [poly], (230, 230, 230))
    # Dark pupil
    cv2.circle(crop, (pupil_x, pupil_y), pupil_r, (20, 20, 20), -1)
    return crop, poly


# ==============================================================================
# 1. Eye Crop Heights: 8, 10, 12, 14, and 20 Pixels
# ==============================================================================

@pytest.mark.parametrize("h,w", [
    (8, 20),
    (10, 24),
    (12, 28),
    (14, 32),
    (20, 44),
])
def test_pupil_near_upper_eyelid_detected_across_crop_heights(h, w):
    """
    Verify that pupils positioned near the upper eyelid margin are successfully
    detected without false rejection across short to medium eye heights (8, 10, 12, 14, 20 px).
    """
    pupil_r = max(2, h // 4)
    # Position pupil near the upper eyelid (y is small, close to row 1 or 2)
    pupil_y = max(1, pupil_r - 1)
    pupil_x = w // 2

    crop, poly = _make_synthetic_eye_crop(h, w, pupil_y=pupil_y, pupil_x=pupil_x, pupil_r=pupil_r)
    center, bbox, diag = detect_pupil(crop, eye_polygon=poly, return_diagnostics=True)

    assert center is not None, f"Failed for h={h}, diag={diag}"
    assert diag["valid"] is True
    assert diag["rejection_reason"] == ""
    assert abs(center[0] - pupil_x) <= 2
    assert abs(center[1] - pupil_y) <= 2


# ==============================================================================
# 2. Pupil Touching the Upper Eyelid Without Automatic Rejection
# ==============================================================================

def test_pupil_touching_upper_eyelid_without_automatic_rejection():
    """
    Verify that when pupil disc reaches the superior palpebral seam (touching y=0/eyelid edge),
    it is accepted and not falsely rejected as an eyelash or eyelid border artifact.
    """
    h, w = 12, 28
    crop, poly = _make_synthetic_eye_crop(h, w, pupil_y=2, pupil_x=14, pupil_r=3)
    # Pupil radius 3 centered at y=2 means it directly reaches the top boundary y=0
    center, bbox, diag = detect_pupil(crop, eye_polygon=poly, return_diagnostics=True)

    assert center is not None, f"Expected pupil detection, got diag={diag}"
    assert diag["valid"] is True
    assert diag["rejection_reason"] == ""
    assert abs(center[0] - 14) <= 2
    assert abs(center[1] - 2) <= 2


# ==============================================================================
# 3. Genuine Eyelash Streak Rejection
# ==============================================================================

@pytest.mark.parametrize("h,w", [(10, 24), (14, 32), (30, 60)])
def test_genuine_eyelash_streak_rejected(h, w):
    """
    Verify that an eyelash-only horizontal streak located along the eyelid boundary
    remains strictly rejected (not falsely classified as a pupil).
    """
    crop = np.full((h, w, 3), 160, dtype=np.uint8)
    mid_y = h // 2
    poly = np.array([
        [0, mid_y],
        [w // 4, 0],
        [3 * w // 4, 0],
        [w - 1, mid_y],
        [3 * w // 4, h - 1],
        [w // 4, h - 1],
    ], dtype=np.int32)
    cv2.fillPoly(crop, [poly], (230, 230, 230))
    # Draw horizontal dark streak right on the upper seam
    cv2.line(crop, (w // 4, 0), (3 * w // 4, 0), (15, 15, 15), 1)

    center, bbox, diag = detect_pupil(crop, eye_polygon=poly, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["valid"] is False
    assert diag["rejection_reason"] in ("edge_attached_eyelid_or_eyelash", "invalid_aspect_ratio", "horizontal_span_artifact")


# ==============================================================================
# 4. Canonical Gaze Directions: Center, Left, Right, Up, Down
# ==============================================================================

def test_five_canonical_gaze_directions_classification():
    """
    Verify clean, deterministic classification of all five canonical gaze states:
    Center, Left, Right, Up, and Down relative to calibrated baseline.
    """
    base = ((0.50, 0.40), (0.50, 0.40))
    lw, lh = 100, 100
    rw, rh = 100, 100

    # Center: pupil at (50, 40)
    assert calculate_gaze_direction((50, 40), lw, lh, (50, 40), rw, rh, calibrated_gaze=base) == GAZE_CENTER

    # Left: physical user-left is image-right (x > 50 + 16 = 66)
    assert calculate_gaze_direction((70, 40), lw, lh, (70, 40), rw, rh, calibrated_gaze=base) == GAZE_LEFT

    # Right: physical user-right is image-left (x < 50 - 16 = 34)
    assert calculate_gaze_direction((30, 40), lw, lh, (30, 40), rw, rh, calibrated_gaze=base) == GAZE_RIGHT

    # Up: physical up is negative image-y delta (y <= 40 - 8 = 32)
    assert calculate_gaze_direction((50, 31), lw, lh, (50, 31), rw, rh, calibrated_gaze=base) == GAZE_UP

    # Down: physical down is positive image-y delta (y >= 40 + 10 = 50)
    assert calculate_gaze_direction((50, 52), lw, lh, (50, 52), rw, rh, calibrated_gaze=base) == GAZE_DOWN


# ==============================================================================
# 5. Upward Deflection -0.08
# ==============================================================================

def test_asymmetric_upward_deflection_threshold_negative_08():
    """
    Verify that an upward deflection reaching exactly -0.08 evaluates to GAZE_UP,
    respecting the asymmetric upward threshold GAZE_UPWARD_THRESHOLD = 0.08.
    """
    base = ((0.50, 0.40), (0.50, 0.40))
    lw, lh = 100, 100
    # norm_ly = 32 / 100 = 0.32 -> diff_ly = 0.32 - 0.40 = -0.08
    pupil_at_neg_08 = (50, 32)

    direction = calculate_gaze_direction(
        pupil_at_neg_08, lw, lh,
        pupil_at_neg_08, lw, lh,
        calibrated_gaze=base
    )
    assert direction == GAZE_UP


# ==============================================================================
# 6. Downward Deflection +0.12
# ==============================================================================

def test_downward_deflection_threshold_positive_12():
    """
    Verify that downward deflection of +0.12 evaluates to GAZE_DOWN,
    respecting GAZE_VERTICAL_THRESHOLD = 0.10.
    """
    base = ((0.50, 0.40), (0.50, 0.40))
    lw, lh = 100, 100
    # norm_ly = 52 / 100 = 0.52 -> diff_ly = 0.52 - 0.40 = +0.12
    pupil_at_pos_12 = (50, 52)

    direction = calculate_gaze_direction(
        pupil_at_pos_12, lw, lh,
        pupil_at_pos_12, lw, lh,
        calibrated_gaze=base
    )
    assert direction == GAZE_DOWN


# ==============================================================================
# 7. Neutral Jitter -0.03
# ==============================================================================

def test_neutral_vertical_jitter_negative_03_remains_center():
    """
    Verify that subtle upward vertical jitter (-0.03 delta) does NOT trigger GAZE_UP
    and remains safely classified as GAZE_CENTER.
    """
    base = ((0.50, 0.40), (0.50, 0.40))
    lw, lh = 100, 100
    # norm_ly = 37 / 100 = 0.37 -> diff_ly = 0.37 - 0.40 = -0.03
    pupil_at_neg_03 = (50, 37)

    direction = calculate_gaze_direction(
        pupil_at_neg_03, lw, lh,
        pupil_at_neg_03, lw, lh,
        calibrated_gaze=base
    )
    assert direction == GAZE_CENTER


# ==============================================================================
# 8. Partial Eyelid Occlusion and Low Contrast
# ==============================================================================

def test_partial_eyelid_occlusion_localized_successfully():
    """
    Verify that when the eyelid partially occludes the upper pupil (simulating a drooping eyelid
    or steep upward gaze), the visible pupil segment is localized without error.
    """
    h, w = 14, 30
    crop, poly = _make_synthetic_eye_crop(h, w, pupil_y=3, pupil_x=15, pupil_r=3)
    # Droop upper eyelid down over row 0..2
    crop[0:2, :] = (160, 160, 160)

    center, bbox, diag = detect_pupil(crop, eye_polygon=poly, return_diagnostics=True)
    assert center is not None
    assert diag["valid"] is True


def test_low_contrast_eye_crop_rejected_with_reason():
    """
    Verify that low-contrast crops (uniform illumination / darkness) are safely rejected.
    """
    crop = np.full((12, 28, 3), 100, dtype=np.uint8)
    crop[5:7, 13:15] = 95  # Contrast delta = 5 (< 12)
    poly = np.array([[0, 6], [7, 0], [21, 0], [27, 6], [21, 11], [7, 11]], dtype=np.int32)

    center, bbox, diag = detect_pupil(crop, eye_polygon=poly, return_diagnostics=True)
    assert center is None
    assert diag["valid"] is False
    assert diag["rejection_reason"] == "low_contrast_or_uniform_shadow"


# ==============================================================================
# 9. Full Blink Rejection
# ==============================================================================

def test_full_blink_rejected_as_closed_eye():
    """
    Verify that a collapsed eye polygon (eyelids closed during a blink) returns None
    with 'closed_eye_or_blink'.
    """
    crop = np.full((20, 40, 3), 120, dtype=np.uint8)
    # Collapsed polygon with height = 1
    blink_poly = np.array([[2, 10], [20, 10], [38, 10], [20, 11]], dtype=np.int32)

    center, bbox, diag = detect_pupil(crop, eye_polygon=blink_poly, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["valid"] is False
    assert diag["rejection_reason"] == "closed_eye_or_blink"


# ==============================================================================
# 10. Invalid and Out-of-Bounds Crops
# ==============================================================================

def test_invalid_and_degenerate_crops_safe():
    """
    Verify that degenerate, empty, or None crops return safe failure without exceptions.
    """
    # None crop
    c, b, diag = detect_pupil(None, return_diagnostics=True)
    assert c is None and diag["valid"] is False

    # Empty 0x0 crop
    c, b, diag = detect_pupil(np.zeros((0, 0, 3), dtype=np.uint8), return_diagnostics=True)
    assert c is None and diag["valid"] is False

    # 1x1 crop
    c, b, diag = detect_pupil(np.full((1, 1, 3), 100, dtype=np.uint8), return_diagnostics=True)
    assert c is None and diag["valid"] is False


# ==============================================================================
# 11. Both-Eye and Single-Eye Policy
# ==============================================================================

def test_both_eye_and_single_eye_policy():
    """
    Verify that under the existing Phase 10.13.2 policy, missing either pupil
    strictly yields GAZE_UNCERTAIN to prevent unverified single-eye assumptions.
    """
    base = ((0.50, 0.40), (0.50, 0.40))
    pupil = (50, 30)

    # Both eyes present -> Upward direction confirmed
    assert calculate_gaze_direction(pupil, 100, 100, pupil, 100, 100, calibrated_gaze=base) == GAZE_UP

    # Left eye missing -> GAZE_UNCERTAIN
    assert calculate_gaze_direction(None, 100, 100, pupil, 100, 100, calibrated_gaze=base) == GAZE_UNCERTAIN

    # Right eye missing -> GAZE_UNCERTAIN
    assert calculate_gaze_direction(pupil, 100, 100, None, 100, 100, calibrated_gaze=base) == GAZE_UNCERTAIN

    # Both eyes missing -> GAZE_UNCERTAIN
    assert calculate_gaze_direction(None, 100, 100, None, 100, 100, calibrated_gaze=base) == GAZE_UNCERTAIN


# ==============================================================================
# 12. Backward Compatibility of Existing Public APIs
# ==============================================================================

def test_backward_compatibility_public_apis():
    """
    Verify that existing function signatures and return types continue to operate seamlessly.
    """
    # 1. detect_pupil without return_diagnostics
    crop, poly = _make_synthetic_eye_crop(14, 30, pupil_y=7, pupil_x=15, pupil_r=3)
    res = detect_pupil(crop, eye_polygon=poly)
    assert isinstance(res, tuple) and len(res) == 2
    assert res[0] is not None and res[1] is not None

    # 2. calculate_gaze_direction with legacy arguments (omitting v_thresh_up)
    legacy_dir = calculate_gaze_direction((50, 40), 100, 100, (50, 40), 100, 100)
    assert legacy_dir == GAZE_CENTER

    # 3. calibrate_gaze_baseline
    valid_samples = [((0.50, 0.40), (0.50, 0.40)) for _ in range(12)]
    baseline = calibrate_gaze_baseline(valid_samples, min_samples=10)
    assert baseline is not None
    assert abs(baseline[0][0] - 0.50) < 0.01

    # 4. config thresholds
    assert config.GAZE_UPWARD_THRESHOLD == 0.08
    assert config.GAZE_VERTICAL_THRESHOLD == 0.10


# ==============================================================================
# 13. End-to-End Landmark Gaze Analysis with Padded Crop Geometry
# ==============================================================================

def _create_mock_landmarks_with_coords(left_eye_pts, right_eye_pts):
    mock = MagicMock()
    pts_map = {}
    for idx, (x, y) in enumerate(left_eye_pts, start=36):
        pts_map[idx] = (x, y)
    for idx, (x, y) in enumerate(right_eye_pts, start=42):
        pts_map[idx] = (x, y)

    def get_part(n):
        p = MagicMock()
        coord = pts_map.get(n, (50, 50))
        p.x = coord[0]
        p.y = coord[1]
        return p

    mock.part.side_effect = get_part
    return mock


def test_end_to_end_upward_gaze_from_landmarks_with_padded_geometry():
    """
    Verify that analyze_gaze_from_landmarks processes padded crops, maps coordinates
    back to original aperture frame, and accurately evaluates GAZE_UP.
    """
    # 120x160 synthetic face frame
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    # Left eye aperture: x in [30, 54], y in [30, 40] -> lw=24, lh=10
    left_pts = [(30, 35), (36, 30), (48, 30), (54, 35), (48, 40), (36, 40)]
    # Right eye aperture: x in [80, 104], y in [30, 40] -> rw=24, rh=10
    right_pts = [(80, 35), (86, 30), (98, 30), (104, 35), (98, 40), (86, 40)]

    cv2.fillPoly(frame, [np.array(left_pts)], (230, 230, 230))
    cv2.fillPoly(frame, [np.array(right_pts)], (230, 230, 230))

    # Pupils looking up: placed at y=31 in frame (which is y=1 in eye aperture coords, radius 2)
    cv2.circle(frame, (42, 31), 2, (20, 20, 20), -1)
    cv2.circle(frame, (92, 31), 2, (20, 20, 20), -1)

    mock_landmarks = _create_mock_landmarks_with_coords(left_pts, right_pts)
    calibrated_gaze = ((0.50, 0.40), (0.50, 0.40))

    annotated, direction = analyze_gaze_from_landmarks(
        frame,
        mock_landmarks,
        calibrated_gaze=calibrated_gaze
    )

    assert direction == GAZE_UP

    diag = get_last_gaze_diagnostics()
    assert diag is not None
    assert diag["left_pupil_detected"] is True
    assert diag["right_pupil_detected"] is True
    assert diag["left_pupil_valid"] is True
    assert diag["right_pupil_valid"] is True
    assert diag["diff_left_y"] <= -0.08
    assert diag["diff_right_y"] <= -0.08
    assert diag["gaze_direction"] == GAZE_UP
