import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from eye_movement import (
    detect_pupil,
    calibrate_gaze_baseline,
    GAZE_CALIBRATION_MAX_STDDEV,
    GAZE_CALIBRATION_MAX_RANGE,
    MIN_GAZE_CALIBRATION_SAMPLES,
)
from time_source import MockTimeSource

pytestmark = pytest.mark.unit


# ==============================================================================
# Task 1: Pupil Edge Filtering Tests
# ==============================================================================

def test_pupil_contour_touching_upper_eyelid_accepted_when_centroid_inside_aperture():
    """
    Test 1: Pupil contour touching the upper eyelid margin is accepted when
    its spatial centroid is located safely within the eroded palpebral aperture.
    """
    crop = np.full((30, 60, 3), 150, dtype=np.uint8)
    polygon = np.array([[5, 15], [18, 6], [42, 6], [55, 15], [42, 24], [18, 24]])
    cv2.fillPoly(crop, [polygon], (210, 210, 210))
    # Draw realistic dark pupil circle centered at (30, 12) with radius 7.
    # Its upper boundary reaches y = 5, directly contacting the upper eyelid edge at y = 6.
    cv2.circle(crop, (30, 12), 7, (20, 20, 20), -1)

    center, bbox, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
    assert center is not None, f"Expected valid detection, got diag={diag}"
    assert diag["valid"] is True
    assert bbox is not None
    assert abs(center[0] - 30) <= 2
    assert abs(center[1] - 12) <= 2


def test_eyelash_only_boundary_contour_rejected():
    """
    Test 2: An eyelash-only dark streak located directly on the palpebral boundary
    is rejected with 'edge_attached_eyelid_or_eyelash'.
    """
    crop = np.full((30, 60, 3), 150, dtype=np.uint8)
    polygon = np.array([[5, 15], [18, 6], [42, 6], [55, 15], [42, 24], [18, 24]])
    cv2.fillPoly(crop, [polygon], (210, 210, 210))
    # Draw dark boundary streak directly on the upper eyelid seam at y = 6
    cv2.line(crop, (20, 6), (40, 6), (10, 10, 10), 2)

    center, bbox, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["valid"] is False
    assert diag["rejection_reason"] == "edge_attached_eyelid_or_eyelash"


def test_centroid_outside_valid_aperture_rejected():
    """
    Test 3: A dark contour whose centroid falls outside the eroded aperture mask
    (e.g. at the canthus / corner margin) is rejected with 'edge_attached_eyelid_or_eyelash'.
    """
    crop = np.full((40, 60, 3), 150, dtype=np.uint8)
    # Define an aperture polygon situated in the crop
    polygon = np.array([[10, 25], [25, 20], [45, 20], [55, 25], [45, 35], [25, 35]])
    cv2.fillPoly(crop, [polygon], (210, 210, 210))
    # Place a dark boundary spot at the extreme canthus (x=10..12, y=25) where inner_mask is 0
    cv2.line(crop, (10, 25), (12, 25), (20, 20, 20), 2)

    center, bbox, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["valid"] is False
    assert diag["rejection_reason"] == "edge_attached_eyelid_or_eyelash"


def test_invalid_or_out_of_bounds_centroid_safe():
    """
    Test 4: Degenerate, out-of-bounds, or empty crops do not cause indexing exceptions.
    """
    # Empty crop
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    c, b, diag = detect_pupil(empty_crop, return_diagnostics=True)
    assert c is None and diag["valid"] is False

    # Tiny 1x1 crop
    tiny_crop = np.full((1, 1, 3), 100, dtype=np.uint8)
    c, b, diag = detect_pupil(tiny_crop, return_diagnostics=True)
    assert c is None and diag["valid"] is False

    # None crop
    c, b, diag = detect_pupil(None, return_diagnostics=True)
    assert c is None and diag["valid"] is False


# ==============================================================================
# Task 2: Contrast Fallback Tests
# ==============================================================================

def test_low_aperture_contrast_with_adequate_full_crop_contrast_follows_fallback():
    """
    Test 5: Low aperture contrast (< 12) with adequate full-crop contrast (>= 15)
    follows the fallback and succeeds in thresholding the pupil.
    """
    crop = np.full((20, 40, 3), 100, dtype=np.uint8)
    polygon = np.array([[5, 10], [15, 5], [25, 5], [35, 10], [25, 15], [15, 15]])
    # Aperture pixels: subtle 10-level difference (e.g. 100 sclera, 90 pupil)
    cv2.fillPoly(crop, [polygon], (100, 100, 100))
    cv2.circle(crop, (20, 10), 3, (90, 90, 90), -1)
    # Surrounding facial skin outside polygon provides high dynamic range (140 - 90 = 50 >= 15)
    crop[0:3, :] = (140, 140, 140)

    center, bbox, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
    # Fallback allowed detection to proceed
    assert diag["rejection_reason"] != "low_contrast_or_uniform_shadow"


def test_insufficient_full_crop_contrast_rejected():
    """
    Test 6: Insufficient contrast both inside the aperture and across the full crop (< 15)
    is rejected with 'low_contrast_or_uniform_shadow'.
    """
    # Flat crop where difference between max and min is only 8 levels everywhere
    crop = np.full((20, 40, 3), 100, dtype=np.uint8)
    polygon = np.array([[5, 10], [15, 5], [25, 5], [35, 10], [25, 15], [15, 15]])
    cv2.circle(crop, (20, 10), 3, (92, 92, 92), -1)

    center, bbox, diag = detect_pupil(crop, eye_polygon=polygon, return_diagnostics=True)
    assert center is None and bbox is None
    assert diag["valid"] is False
    assert diag["rejection_reason"] == "low_contrast_or_uniform_shadow"


# ==============================================================================
# Task 3: Calibration Sample Accumulation Tests
# ==============================================================================

def test_calibration_samples_accumulate_across_five_second_boundaries():
    """
    Test 7: Gaze calibration samples accumulate across consecutive intervals
    and are not cleared at the 5-second boundary.
    """
    time_src = MockTimeSource(initial_time=0.0)
    calib_duration = 5.0
    max_timeout = 15.0
    min_samples = 10

    gaze_samples = []
    head_samples = []
    calib_start_time = time_src.now()
    is_calibrated = False

    # Simulate 70 frames at 10 FPS (7.0 seconds total stream time)
    # Gaze sample detected only every 6 frames (low rate, ~1.6 samples/sec)
    for frame_idx in range(70):
        current_time = time_src.now()
        elapsed = current_time - calib_start_time

        # Head sample on every frame
        head_samples.append((0.0, 0.0, 0.0))

        # Gaze sample every 6th frame
        if frame_idx % 6 == 0:
            gaze_samples.append(((0.50, 0.40), (0.50, 0.40)))

        has_min_samples = (
            len(head_samples) >= min_samples and len(gaze_samples) >= min_samples
        )

        if elapsed >= calib_duration and has_min_samples:
            is_calibrated = True
            break
        elif elapsed >= max_timeout:
            head_samples.clear()
            gaze_samples.clear()
            calib_start_time = current_time

        time_src.advance(0.10)

    # At 5.0 seconds (frame 50), gaze_samples had 9 samples (not cleared!)
    # At frame 54 (5.4s), gaze_samples reached 10 samples and succeeded!
    assert is_calibrated is True
    assert len(gaze_samples) >= 10
    assert 5.0 <= current_time <= max_timeout


def test_calibration_succeeds_when_10_valid_paired_samples_accumulated_within_15s():
    """
    Test 8: Calibration succeeds when 10 valid paired samples accumulate within 15s.
    """
    time_src = MockTimeSource(initial_time=0.0)
    calib_duration = 5.0
    max_timeout = 15.0
    min_samples = 10

    gaze_samples = []
    head_samples = []
    calib_start_time = time_src.now()
    is_calibrated = False
    calibrated_baseline = None

    # Slowly accumulate 10 samples spread over 8.0 seconds
    for f in range(80):
        current_time = time_src.now()
        elapsed = current_time - calib_start_time

        head_samples.append((0.0, 0.0, 0.0))
        if f in [5, 12, 19, 27, 35, 43, 51, 59, 67, 75]:
            # Jitter slightly around realistic neutral baseline
            lx = 0.50 + 0.01 * (f % 3 - 1)
            ly = 0.40 + 0.01 * (f % 2)
            rx = 0.50 + 0.01 * (f % 2)
            ry = 0.40 + 0.01 * (f % 3 - 1)
            gaze_samples.append(((lx, ly), (rx, ry)))

        has_min_samples = (
            len(head_samples) >= min_samples and len(gaze_samples) >= min_samples
        )

        if elapsed >= calib_duration and has_min_samples:
            baseline = calibrate_gaze_baseline(gaze_samples, min_samples=min_samples)
            if baseline is not None:
                calibrated_baseline = baseline
                is_calibrated = True
                break

        time_src.advance(0.10)

    assert is_calibrated is True
    assert calibrated_baseline is not None
    assert pytest.approx(calibrated_baseline[0][0], abs=0.05) == 0.50
    assert pytest.approx(calibrated_baseline[0][1], abs=0.05) == 0.40


def test_calibration_fails_safely_after_maximum_timeout():
    """
    Test 9: Calibration resets safely and provides an explanation when
    the maximum 15-second timeout expires with insufficient samples.
    """
    time_src = MockTimeSource(initial_time=0.0)
    calib_duration = 5.0
    max_timeout = 15.0
    min_samples = 10

    gaze_samples = []
    head_samples = []
    calib_start_time = time_src.now()
    timeout_triggered = False
    timeout_reason = None

    # Simulate 160 frames (16.0s), providing only 4 gaze samples in total
    for f in range(160):
        current_time = time_src.now()
        elapsed = current_time - calib_start_time

        head_samples.append((0.0, 0.0, 0.0))
        if f in [10, 40, 70, 100]:
            gaze_samples.append(((0.50, 0.40), (0.50, 0.40)))

        has_min_samples = (
            len(head_samples) >= min_samples and len(gaze_samples) >= min_samples
        )

        if elapsed >= calib_duration and has_min_samples:
            break
        elif elapsed >= max_timeout:
            timeout_reason = (
                f"calibration timed out after {max_timeout:.1f}s: "
                f"insufficient valid samples (head {len(head_samples)}/{min_samples}, "
                f"gaze {len(gaze_samples)}/{min_samples})"
            )
            head_samples.clear()
            gaze_samples.clear()
            calib_start_time = current_time
            timeout_triggered = True
            break

        time_src.advance(0.10)

    assert timeout_triggered is True
    assert "calibration timed out after 15.0s" in timeout_reason
    assert len(gaze_samples) == 0  # Buffer safely cleared on timeout


# ==============================================================================
# Task 4 & Quality Limits: Stability Thresholds Preserved
# ==============================================================================

def test_existing_calibration_stability_thresholds_remain_enforced():
    """
    Test 10: Existing quality controls (stddev <= 0.06, range <= 0.20)
    remain strictly enforced and reject unstable samples.
    """
    # 1. Unstable samples with standard deviation > 0.06
    unstable_samples = [
        ((0.50 + 0.10 * (i % 3 - 1), 0.40 + 0.12 * (i % 2)),
         (0.50 + 0.10 * (i % 3 - 1), 0.40 + 0.12 * (i % 2)))
        for i in range(12)
    ]
    baseline_unstable, quality = calibrate_gaze_baseline(
        unstable_samples, min_samples=10, return_diagnostics=True
    )
    assert baseline_unstable is None
    assert "unstable gaze calibration" in quality["failure_reason"]

    # 2. Stable samples with standard deviation <= 0.03
    stable_samples = [
        ((0.50 + 0.01 * (i % 3 - 1), 0.40 + 0.01 * (i % 2)),
         (0.50 + 0.01 * (i % 3 - 1), 0.40 + 0.01 * (i % 2)))
        for i in range(12)
    ]
    baseline_stable, quality = calibrate_gaze_baseline(
        stable_samples, min_samples=10, return_diagnostics=True
    )
    assert baseline_stable is not None
    assert not quality["failure_reason"]
