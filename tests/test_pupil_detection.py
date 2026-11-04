import cv2
import numpy as np
import pytest
from eye_movement import (
    calculate_gaze_direction,
    detect_pupil,
    calibrate_gaze_baseline,
    GAZE_CENTER,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    GAZE_DOWN,
    GAZE_UNCERTAIN,
)

pytestmark = pytest.mark.unit

def test_gaze_direction_center():
    """Verify that centered pupils in both eyes evaluate to GAZE_CENTER."""
    # Pupil at (15, 10) in a 30x20 bounding box is normalized to (0.50, 0.50)
    direction = calculate_gaze_direction(
        left_pupil=(15, 10), left_eye_w=30, left_eye_h=20,
        right_pupil=(15, 10), right_eye_w=30, right_eye_h=20
    )
    assert direction == GAZE_CENTER

def test_gaze_direction_left():
    """Physical user-left is image-right for unmirrored webcam input."""
    direction = calculate_gaze_direction(
        left_pupil=(24, 8), left_eye_w=30, left_eye_h=20,
        right_pupil=(20, 8), right_eye_w=25, right_eye_h=20
    )
    assert direction == GAZE_LEFT

def test_gaze_direction_right():
    """Physical user-right is image-left for unmirrored webcam input."""
    direction = calculate_gaze_direction(
        left_pupil=(6, 8), left_eye_w=30, left_eye_h=20,
        right_pupil=(5, 8), right_eye_w=25, right_eye_h=20
    )
    assert direction == GAZE_RIGHT

def test_gaze_direction_up():
    """Verify that pupils in the upper third evaluate to GAZE_UP."""
    # Both pupils at 20% height (< 0.30)
    direction = calculate_gaze_direction(
        left_pupil=(15, 4), left_eye_w=30, left_eye_h=20,
        right_pupil=(15, 4), right_eye_w=30, right_eye_h=20
    )
    assert direction == GAZE_UP

def test_gaze_direction_down():
    """Verify that pupils in the lower half evaluate to GAZE_DOWN."""
    # Both pupils at 70% height (> 0.50)
    direction = calculate_gaze_direction(
        left_pupil=(15, 14), left_eye_w=30, left_eye_h=20,
        right_pupil=(15, 14), right_eye_w=30, right_eye_h=20
    )
    assert direction == GAZE_DOWN

def test_independent_eye_normalization_asymmetry():
    """Verify that eyes with significantly different bounding box sizes are normalized independently."""
    # Independent normalization preserves image-left motion as physical user-right.
    # If right pupil was incorrectly compared against left eye width (60), 7/60 would be 0.11
    # If left pupil was compared against right width (30), 15/30 would be 0.5 (center)
    direction = calculate_gaze_direction(
        left_pupil=(15, 10), left_eye_w=60, left_eye_h=20,
        right_pupil=(7, 10), right_eye_w=30, right_eye_h=20
    )
    assert direction == GAZE_RIGHT

def test_missing_or_none_pupil_yields_uncertain_gaze():
    """Verify that if one or both pupils are not found, state is explicitly GAZE_UNCERTAIN."""
    assert calculate_gaze_direction(None, 30, 20, (15, 10), 30, 20) == GAZE_UNCERTAIN
    assert calculate_gaze_direction((15, 10), 30, 20, None, 30, 20) == GAZE_UNCERTAIN
    assert calculate_gaze_direction(None, 30, 20, None, 30, 20) == GAZE_UNCERTAIN


def test_detect_pupil_empty_crop_safe():
    """Verify that detect_pupil handles empty, None, or degenerate regions safely."""
    center, bbox = detect_pupil(None)
    assert center is None and bbox is None

    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    center, bbox = detect_pupil(empty)
    assert center is None and bbox is None

    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    center, bbox = detect_pupil(tiny)
    assert center is None and bbox is None


def test_adaptive_pupil_detection_on_synthetic_eye_patch():
    """Verify that detect_pupil accurately locates pupil on bright and low-light synthetic eye crops."""
    # 1. Normal/Bright eye patch: white background (200), dark pupil (30) at center
    eye_bright = np.full((30, 50, 3), 200, dtype=np.uint8)
    cv2.circle(eye_bright, (25, 15), 5, (30, 30, 30), -1)

    center, bbox = detect_pupil(eye_bright, method="adaptive")
    assert center is not None
    assert abs(center[0] - 25) <= 2
    assert abs(center[1] - 15) <= 2

    # 2. Low-light eye patch: gray background (100), dark pupil (15) at center
    eye_dark = np.full((30, 50, 3), 100, dtype=np.uint8)
    cv2.circle(eye_dark, (25, 15), 5, (15, 15, 15), -1)

    center, bbox = detect_pupil(eye_dark, method="adaptive")
    assert center is not None
    assert abs(center[0] - 25) <= 2
    assert abs(center[1] - 15) <= 2


def test_pupil_detection_rejects_noise_and_extreme_shadow():
    """Verify that detect_pupil rejects uniform blank patches, noise below min_area, and extreme shadows."""
    # 1. Blank uniform eye patch without pupil
    eye_blank = np.full((30, 50, 3), 200, dtype=np.uint8)
    center, bbox = detect_pupil(eye_blank)
    assert center is None and bbox is None

    # Speckle rejected by min_area=15.0
    eye_speckle = np.full((30, 50, 3), 220, dtype=np.uint8)
    eye_speckle[15, 25] = (0, 0, 0)
    center, bbox = detect_pupil(eye_speckle, min_area=15.0)
    assert center is None and bbox is None

    # Complete dark shadow (> 80% area)
    eye_shadow = np.full((30, 50, 3), 20, dtype=np.uint8)
    center, bbox = detect_pupil(eye_shadow, max_ratio=0.80)
    assert center is None and bbox is None


def test_low_contrast_eye_crop_is_rejected_with_reason():
    crop = np.full((12, 28, 3), 100, dtype=np.uint8)
    crop[6, 14] = (92, 92, 92)
    center, bbox, diag = detect_pupil(crop, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["rejection_reason"] == "low_contrast_or_uniform_shadow"


def test_eye_boundary_eyelash_artifact_is_rejected():
    crop = np.full((30, 60, 3), 150, dtype=np.uint8)
    polygon = np.array([[5, 15], [18, 6], [42, 6], [55, 15], [42, 24], [18, 24]])
    cv2.fillPoly(crop, [polygon], (210, 210, 210))
    cv2.line(crop, (20, 6), (40, 6), (10, 10, 10), 2)

    center, bbox, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["rejection_reason"] == "edge_attached_eyelid_or_eyelash"


def test_left_and_right_eye_synthetic_detection_is_symmetric():
    cases = (
        ((20, 50), np.array([[3, 10], [13, 4], [37, 4], [47, 10], [37, 16], [13, 16]]), (25, 10), 3),
        ((18, 44), np.array([[3, 9], [12, 4], [32, 4], [41, 9], [32, 14], [12, 14]]), (22, 9), 2),
    )
    normalized = []
    for (height, width), polygon, pupil_xy, radius in cases:
        crop = np.full((height, width, 3), 200, dtype=np.uint8)
        cv2.fillPoly(crop, [polygon], (220, 220, 220))
        cv2.circle(crop, pupil_xy, radius, (20, 20, 20), -1)
        center, _, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
        assert diag["valid"] is True
        normalized.append((center[0] / width, center[1] / height))
    assert normalized[0] == pytest.approx(normalized[1], abs=0.03)


def test_calibrate_gaze_baseline_median_and_threshold():
    """Verify that calibrate_gaze_baseline computes robust median and enforces minimum samples."""
    # Stable samples with a blink.
    samples = [
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.51, 0.41), (0.49, 0.39)),
        None,  # blink
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.49, 0.39), (0.51, 0.41)),
        ((0.51, 0.39), (0.49, 0.41)),
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.52, 0.40), (0.48, 0.40)),
        ((0.50, 0.39), (0.50, 0.41)),
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.51, 0.41), (0.49, 0.39)),
    ]
    baseline = calibrate_gaze_baseline(samples, min_samples=10)
    assert baseline is not None
    (base_lx, base_ly), (base_rx, base_ry) = baseline
    assert abs(base_lx - 0.50) < 0.03
    assert abs(base_ly - 0.40) < 0.03

    # Insufficient samples returns None
    insufficient = [((0.5, 0.4), (0.5, 0.4))] * 5
    assert calibrate_gaze_baseline(insufficient, min_samples=10) is None


def test_calculate_gaze_direction_with_calibrated_baseline():
    """Verify that gaze direction is classified relative to non-default calibrated baseline."""
    # Candidate with natural gaze offset at 0.40 horizontal instead of 0.50
    custom_baseline = ((0.40, 0.40), (0.40, 0.40))

    # Pupil at 0.40 in 100px eye -> Center relative to custom baseline
    res_center = calculate_gaze_direction(
        left_pupil=(40, 40), left_eye_w=100, left_eye_h=100,
        right_pupil=(40, 40), right_eye_w=100, right_eye_h=100,
        calibrated_gaze=custom_baseline
    )
    assert res_center == GAZE_CENTER

    # Image-right movement is physical user-left for unmirrored input.
    res_left = calculate_gaze_direction(
        left_pupil=(60, 40), left_eye_w=100, left_eye_h=100,
        right_pupil=(60, 40), right_eye_w=100, right_eye_h=100,
        calibrated_gaze=custom_baseline
    )
    assert res_left == GAZE_LEFT


def test_detect_pupil_with_eye_polygon_masking():
    """Verify that detect_pupil isolates pupil when constrained by eye landmark polygon."""
    # Create 30x60 eye crop with white sclera inside polygon (180) and skin outside (120)
    crop = np.full((30, 60, 3), 120, dtype=np.uint8)
    poly = np.array([
        [0, 15], [15, 5], [45, 5], [60, 15], [45, 25], [15, 25]
    ], dtype=np.int32)
    # Fill aperture with sclera
    cv2.fillPoly(crop, [poly], (180, 180, 180))
    # Draw dark pupil at (30, 15) inside polygon
    cv2.circle(crop, (30, 15), 5, (25, 25, 25), -1)

    center, bbox = detect_pupil(crop, eye_polygon=poly)
    assert center is not None
    assert abs(center[0] - 30) <= 2
    assert abs(center[1] - 15) <= 2


def test_detect_pupil_rejects_full_crop_contour():
    """Verify that a dark contour spanning the entire crop is rejected rather than returning midpoint."""
    # Crop where dark region spans the full width and height
    crop = np.full((20, 40, 3), 200, dtype=np.uint8)
    crop[0:20, 0:40] = (30, 30, 30)  # Full crop dark
    center, bbox = detect_pupil(crop)
    assert center is None
    assert bbox is None


def test_detect_pupil_rejects_eyelash_horizontal_streak():
    """Verify that a dark eyelash line spanning the top width is rejected."""
    crop = np.full((20, 40, 3), 180, dtype=np.uint8)
    # 2-pixel tall horizontal streak across full 40px width
    crop[0:3, 0:40] = (20, 20, 20)
    center, bbox = detect_pupil(crop)
    assert center is None
    assert bbox is None


def test_detect_pupil_rejects_low_contrast_uniform_crop():
    """Verify that crops with insufficient contrast (delta < 12) return None."""
    crop = np.full((25, 35, 3), 100, dtype=np.uint8)
    # Minor noise with delta = 4 (< 12)
    crop[10:15, 10:15] = 104
    center, bbox = detect_pupil(crop)
    assert center is None
    assert bbox is None


def test_detect_pupil_left_and_right_positions():
    """Verify that pupil shifted left or right produces accurately shifted coordinates."""
    # Left-shifted pupil at x=15 in 60-wide crop
    crop_left = np.full((30, 60, 3), 200, dtype=np.uint8)
    cv2.circle(crop_left, (15, 15), 5, (30, 30, 30), -1)
    center_l, _ = detect_pupil(crop_left)
    assert center_l is not None
    assert abs(center_l[0] - 15) <= 2

    # Right-shifted pupil at x=45 in 60-wide crop
    crop_right = np.full((30, 60, 3), 200, dtype=np.uint8)
    cv2.circle(crop_right, (45, 15), 5, (30, 30, 30), -1)
    center_r, _ = detect_pupil(crop_right)
    assert center_r is not None
    assert abs(center_r[0] - 45) <= 2


def test_detect_pupil_closed_eye_polygon_rejected():
    """Verify that a collapsed/closed eye polygon (blink) returns None safely."""
    crop = np.full((30, 60, 3), 120, dtype=np.uint8)
    # Collapsed polygon with height = 1
    closed_poly = np.array([[5, 15], [30, 15], [55, 15], [30, 16]], dtype=np.int32)
    center, bbox = detect_pupil(crop, eye_polygon=closed_poly)
    assert center is None
    assert bbox is None


def test_gaze_direction_asymmetric_left_movement():
    """Verify asymmetrical image-right motion maps to physical user-left."""
    direction = calculate_gaze_direction(
        left_pupil=(70, 40), left_eye_w=100, left_eye_h=100,
        right_pupil=(63, 40), right_eye_w=100, right_eye_h=100,
        calibrated_gaze=((0.50, 0.40), (0.50, 0.40))
    )
    assert direction == GAZE_LEFT


def test_gaze_direction_asymmetric_right_movement():
    """Verify asymmetrical image-left motion maps to physical user-right."""
    direction = calculate_gaze_direction(
        left_pupil=(30, 40), left_eye_w=100, left_eye_h=100,
        right_pupil=(36, 40), right_eye_w=100, right_eye_h=100,
        calibrated_gaze=((0.50, 0.40), (0.50, 0.40))
    )
    assert direction == GAZE_RIGHT


def test_gaze_direction_contradictory_movement_rejected():
    """Verify that opposing eye movements (one left, one right) are rejected as center/invalid."""
    # Left eye delta = -0.25 (left), Right eye delta = +0.15 (right) -> Contradiction!
    direction = calculate_gaze_direction(
        left_pupil=(25, 40), left_eye_w=100, left_eye_h=100,
        right_pupil=(65, 40), right_eye_w=100, right_eye_h=100,
        calibrated_gaze=((0.50, 0.40), (0.50, 0.40))
    )
    assert direction == GAZE_CENTER
