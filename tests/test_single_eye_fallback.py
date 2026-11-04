"""
Unit and integration tests for Single-Eye Fallback & Confidence-Aware Gaze Tracking.
Phase 10.13.3 verification suite.
"""

from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from single_eye_fallback import (
    SingleEyeGazeTracker,
    score_eye_confidence,
    evaluate_single_eye_direction,
    GazeTrackingResult,
    TRACKING_STATE_PAIRED,
    TRACKING_STATE_LEFT_ONLY,
    TRACKING_STATE_RIGHT_ONLY,
    TRACKING_STATE_UNCERTAIN,
    TRACKING_STATE_RECOVERING,
    MIN_EYE_CONFIDENCE,
    SINGLE_EYE_CONFIDENCE_THRESHOLD,
    MAX_SINGLE_EYE_COMBINED_CONF,
    RECOVERY_REQUIRED_FRAMES,
)
from eye_movement import (
    calculate_gaze_direction,
    analyze_gaze_from_landmarks,
    process_eye_movement,
    get_last_gaze_diagnostics,
    get_last_gaze_sample,
    get_default_gaze_tracker,
    reset_default_gaze_tracker,
    GAZE_CENTER,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    GAZE_DOWN,
    GAZE_UNCERTAIN,
    GAZE_NO_FACE,
)
import config

pytestmark = pytest.mark.unit


# ==============================================================================
# Helper functions for synthetic test generation
# ==============================================================================

def _make_mock_landmarks(left_eye_pts, right_eye_pts):
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


def _make_eye_patch(h=14, w=30, pupil_pos=(15, 7), contrast_delta=40):
    """Creates a synthetic eye patch with controlled dimensions and contrast."""
    patch = np.full((h, w, 3), 160, dtype=np.uint8)
    poly = np.array([
        [0, h // 2],
        [w // 4, 0],
        [3 * w // 4, 0],
        [w - 1, h // 2],
        [3 * w // 4, h - 1],
        [w // 4, h - 1],
    ], dtype=np.int32)
    sclera_val = int(min(255, 40 + contrast_delta))
    cv2.fillPoly(patch, [poly], (sclera_val, sclera_val, sclera_val))
    px, py = pupil_pos
    cv2.circle(patch, (px, py), 2, (20, 20, 20), -1)
    return patch, poly


# ==============================================================================
# Scenario 1: Paired-Eye Mode When Both Eyes Are Reliable
# ==============================================================================

def test_paired_eye_mode_when_both_eyes_reliable():
    """
    Verify that when both eyes provide reliable detections (confidence >= 0.50),
    the tracker enters PAIRED_TRACKING, achieves high combined confidence (> 0.75),
    and correctly classifies paired directional gaze.
    """
    tracker = SingleEyeGazeTracker()
    # Looking center: pupil at 50% width, 40% height
    base = ((0.50, 0.40), (0.50, 0.40))
    # Pre-condition tracker into paired tracking
    for _ in range(RECOVERY_REQUIRED_FRAMES):
        res = tracker.update(
            left_pupil=(50, 40), left_w=100, left_h=100,
            right_pupil=(50, 40), right_w=100, right_h=100,
            calibrated_gaze=base,
            left_conf=0.90, right_conf=0.92,
        )

    assert res.tracking_state == TRACKING_STATE_PAIRED
    assert res.gaze_direction == GAZE_CENTER
    assert res.combined_confidence >= 0.85
    assert res.is_single_eye_fallback is False

    # Paired upward gaze test
    res_up = tracker.update(
        left_pupil=(50, 31), left_w=100, left_h=100,
        right_pupil=(50, 31), right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.88, right_conf=0.89,
    )
    assert res_up.tracking_state == TRACKING_STATE_PAIRED
    assert res_up.gaze_direction == GAZE_UP
    assert res_up.combined_confidence >= 0.85


# ==============================================================================
# Scenario 2: Single-Eye Fallback (Left Eye Reliable, Right Unavailable)
# ==============================================================================

def test_single_eye_fallback_left_eye_reliable_right_unavailable():
    """
    Verify that when the right eye is unavailable (e.g. occluded or lost)
    and the left eye is reliable (confidence >= 0.70), the system safely falls
    back to LEFT_EYE_ONLY, computes direction, and bounds combined confidence.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Left eye looking right: pupil x = 30px (norm=0.30, diff = 0.50 - 0.30 = +0.20 > ht*1.15)
    res = tracker.update(
        left_pupil=(30, 40), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.85, right_conf=0.0,
    )

    assert res.tracking_state == TRACKING_STATE_LEFT_ONLY
    assert res.is_single_eye_fallback is True
    assert res.gaze_direction == GAZE_RIGHT
    assert res.left_confidence == 0.85
    assert res.right_confidence == 0.0
    # Single-eye combined confidence must be discounted below MAX_SINGLE_EYE_COMBINED_CONF (0.75)
    assert res.combined_confidence <= MAX_SINGLE_EYE_COMBINED_CONF
    assert res.combined_confidence > 0.60


# ==============================================================================
# Scenario 3: Single-Eye Fallback (Right Eye Reliable, Left Unavailable)
# ==============================================================================

def test_single_eye_fallback_right_eye_reliable_left_unavailable():
    """
    Verify that when the left eye is unavailable and the right eye is reliable
    (confidence >= 0.70), the system safely falls back to RIGHT_EYE_ONLY.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Right eye looking left: pupil x = 70px (norm=0.70, diff = 0.50 - 0.70 = -0.20 < -ht*1.15)
    res = tracker.update(
        left_pupil=None, left_w=100, left_h=100,
        right_pupil=(70, 40), right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.0, right_conf=0.88,
    )

    assert res.tracking_state == TRACKING_STATE_RIGHT_ONLY
    assert res.is_single_eye_fallback is True
    assert res.gaze_direction == GAZE_LEFT
    assert res.right_confidence == 0.88
    assert res.combined_confidence <= MAX_SINGLE_EYE_COMBINED_CONF
    assert res.combined_confidence > 0.60


# ==============================================================================
# Scenario 4: Fallback Rejection When Both Eyes Are Unavailable
# ==============================================================================

def test_fallback_rejection_when_both_eyes_unavailable():
    """
    Verify that when neither eye is detected or usable, state is UNCERTAIN,
    direction is GAZE_UNCERTAIN, and confidence is zero.
    """
    tracker = SingleEyeGazeTracker()
    res = tracker.update(
        left_pupil=None, left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        left_conf=0.0, right_conf=0.0,
    )

    assert res.tracking_state == TRACKING_STATE_UNCERTAIN
    assert res.gaze_direction == GAZE_UNCERTAIN
    assert res.combined_confidence == 0.0
    assert res.is_single_eye_fallback is False
    assert res.rejection_reason == "no_usable_pupil_observations"


# ==============================================================================
# Scenario 5: Fallback Rejection When Single Eye Has Low Confidence
# ==============================================================================

def test_fallback_rejection_when_single_eye_below_confidence_threshold():
    """
    Verify that a single surviving eye with confidence below SINGLE_EYE_CONFIDENCE_THRESHOLD (0.70)
    is rejected from single-eye fallback and outputs GAZE_UNCERTAIN.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Left eye confidence 0.55 meets paired threshold (0.50) but fails single-eye threshold (0.70)
    res = tracker.update(
        left_pupil=(30, 40), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.55, right_conf=0.0,
    )

    assert res.tracking_state == TRACKING_STATE_UNCERTAIN
    assert res.gaze_direction == GAZE_UNCERTAIN
    assert res.is_single_eye_fallback is False
    assert res.combined_confidence <= 0.35
    assert res.rejection_reason == "single_eye_confidence_below_threshold"


# ==============================================================================
# Scenario 6: Contradictory Eye Directions Penalty and Uncertain State
# ==============================================================================

def test_contradictory_eye_directions_penalty_and_uncertain():
    """
    Verify that when the two eyes contradict each other horizontally or vertically
    (e.g. left eye looks Left while right eye looks Right), direction is GAZE_UNCERTAIN
    and combined confidence receives a severe penalty.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Left eye looks LEFT (pupil x = 75, diff_lx = -0.25)
    # Right eye looks RIGHT (pupil x = 25, diff_rx = +0.25)
    res = tracker.update(
        left_pupil=(75, 40), left_w=100, left_h=100,
        right_pupil=(25, 40), right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.85, right_conf=0.85,
    )

    assert res.gaze_direction == GAZE_UNCERTAIN
    assert res.rejection_reason == "contradictory_eye_deflections"
    # Combined confidence penalized by 0.20x
    assert res.combined_confidence < 0.25


# ==============================================================================
# Scenario 7: Partial Eyelid Occlusion Handling
# ==============================================================================

def test_partial_eyelid_occlusion_scoring():
    """
    Verify that narrow aperture opening (e.g. heavy eyelid droop or ptosis)
    penalizes confidence via S_aperture, while a wide open eye gets full credit.
    """
    # Occluded aperture (vertical opening only 4px)
    poly_narrow = np.array([[0, 2], [10, 0], [20, 0], [30, 2], [20, 4], [10, 4]])
    conf_narrow = score_eye_confidence(
        pupil=(15, 2), eye_w=30, eye_h=4,
        eye_polygon=poly_narrow,
    )

    # Wide aperture (vertical opening 12px)
    poly_wide = np.array([[0, 6], [10, 0], [20, 0], [30, 6], [20, 12], [10, 12]])
    conf_wide = score_eye_confidence(
        pupil=(15, 6), eye_w=30, eye_h=12,
        eye_polygon=poly_wide,
    )

    assert conf_narrow < conf_wide
    assert conf_narrow >= 0.0
    assert conf_wide <= 1.0


# ==============================================================================
# Scenario 8: Blink and Temporary Pupil Loss Handling
# ==============================================================================

def test_blink_and_temporary_pupil_loss_transition():
    """
    Verify that an instantaneous blink (both pupils None) safely transitions
    to UNCERTAIN without crashing or throwing exceptions.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Tracking paired
    tracker.current_state = TRACKING_STATE_PAIRED
    # Blink frame
    res_blink = tracker.update(
        left_pupil=None, left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base,
    )
    assert res_blink.tracking_state == TRACKING_STATE_UNCERTAIN
    assert res_blink.gaze_direction == GAZE_UNCERTAIN


# ==============================================================================
# Scenario 9: Head-Turn-Related Eye Occlusion (Surviving Eye Tracks)
# ==============================================================================

def test_head_turn_related_eye_occlusion_maintains_tracking():
    """
    Simulate extreme head yaw where one eye is occluded/lost by perspective distortion
    or self-occlusion, but the other eye remains clearly visible.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # User turns head, right eye is lost (None), left eye maintains gaze Looking Left
    # (pupil x = 70px -> norm 0.70 -> diff -0.20 -> Left)
    res = tracker.update(
        left_pupil=(70, 40), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.82, right_conf=0.0,
    )

    assert res.tracking_state == TRACKING_STATE_LEFT_ONLY
    assert res.gaze_direction == GAZE_LEFT
    assert res.is_single_eye_fallback is True


# ==============================================================================
# Scenario 10: Single-Eye Fallback Recovery Transitions to RECOVERING
# ==============================================================================

def test_single_eye_to_recovering_transition():
    """
    Verify that when operating in single-eye mode, the first reappearance of
    paired eyes transitions immediately to RECOVERING, not straight to PAIRED_TRACKING.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Set initial state to LEFT_EYE_ONLY
    tracker.current_state = TRACKING_STATE_LEFT_ONLY

    res = tracker.update(
        left_pupil=(50, 40), left_w=100, left_h=100,
        right_pupil=(50, 40), right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.85, right_conf=0.85,
    )

    assert res.tracking_state == TRACKING_STATE_RECOVERING
    assert tracker.recovery_counter == 1


# ==============================================================================
# Scenario 11: Paired-Eye Recovery Requires 3 Consecutive Frames
# ==============================================================================

def test_paired_recovery_requires_three_consecutive_frames():
    """
    Verify that transitioning from single-eye or uncertain to PAIRED_TRACKING
    strictly requires 3 consecutive paired frames.
    """
    tracker = SingleEyeGazeTracker(recovery_frames_required=3)
    base = ((0.50, 0.40), (0.50, 0.40))
    tracker.current_state = TRACKING_STATE_UNCERTAIN

    # Frame 1: Reappearance of paired observations -> RECOVERING (count=1)
    f1 = tracker.update(
        left_pupil=(50, 40), left_w=100, left_h=100,
        right_pupil=(50, 40), right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.85, right_conf=0.85,
    )
    assert f1.tracking_state == TRACKING_STATE_RECOVERING
    assert tracker.recovery_counter == 1

    # Frame 2: Second consecutive paired observation -> RECOVERING (count=2)
    f2 = tracker.update(
        left_pupil=(50, 40), left_w=100, left_h=100,
        right_pupil=(50, 40), right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.85, right_conf=0.85,
    )
    assert f2.tracking_state == TRACKING_STATE_RECOVERING
    assert tracker.recovery_counter == 2

    # Frame 3: Third consecutive paired observation -> PAIRED_TRACKING (restored!)
    f3 = tracker.update(
        left_pupil=(50, 40), left_w=100, left_h=100,
        right_pupil=(50, 40), right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.85, right_conf=0.85,
    )
    assert f3.tracking_state == TRACKING_STATE_PAIRED
    assert tracker.recovery_counter == 0


# ==============================================================================
# Scenario 12: Rapid Switching Prevention (Debouncing / Hysteresis)
# ==============================================================================

def test_rapid_switching_prevention_aborts_recovery_on_intermittent_loss():
    """
    Verify that an isolated, fleeting paired frame does not prematurely switch to
    PAIRED_TRACKING if the next frame immediately loses an eye.
    """
    tracker = SingleEyeGazeTracker(recovery_frames_required=3)
    base = ((0.50, 0.40), (0.50, 0.40))
    tracker.current_state = TRACKING_STATE_LEFT_ONLY

    # Frame 1: Transient paired frame -> RECOVERING (count=1)
    tracker.update(
        left_pupil=(50, 40), left_w=100, left_h=100,
        right_pupil=(50, 40), right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.85, right_conf=0.85,
    )
    assert tracker.current_state == TRACKING_STATE_RECOVERING
    assert tracker.recovery_counter == 1

    # Frame 2: Right eye lost again -> Recovery immediately aborted back to LEFT_EYE_ONLY
    res = tracker.update(
        left_pupil=(50, 40), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base, left_conf=0.85, right_conf=0.0,
    )
    assert res.tracking_state == TRACKING_STATE_LEFT_ONLY
    assert tracker.recovery_counter == 0  # Counter reset to zero


# ==============================================================================
# Scenario 13: Invalid or Missing Calibration Baseline Handling
# ==============================================================================

def test_missing_or_partial_calibration_baseline_handling():
    """
    Verify that if calibrated_gaze contains None for one eye or is entirely None,
    the tracker defaults safely to (0.50, 0.40) and neither eye baseline is corrupted.
    """
    tracker = SingleEyeGazeTracker()

    # calibrated_gaze has valid left baseline, but None for right
    partial_calib = ((0.46, 0.38), None)
    res = tracker.update(
        left_pupil=(46, 38), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=partial_calib,
        left_conf=0.85, right_conf=0.0,
    )
    assert res.tracking_state == TRACKING_STATE_LEFT_ONLY
    assert res.gaze_direction == GAZE_CENTER

    # evaluate_single_eye_direction with baseline=None
    s_dir = evaluate_single_eye_direction((50, 40), 100, 100, baseline=None)
    assert s_dir == GAZE_CENTER


# ==============================================================================
# Scenario 14: Confidence Boundary Conditions
# ==============================================================================

@pytest.mark.parametrize("pupil,w,h,expected", [
    (None, 100, 100, 0.0),
    ((50, 40), 0, 100, 0.0),
    ((50, 40), 100, -5, 0.0),
    ((-10, 40), 100, 100, 0.0),       # Negative x
    ((150, 40), 100, 100, 0.0),       # Out-of-bounds x
    ((float('nan'), 40), 100, 100, 0.0),  # NaN coordinate
    ((50, float('inf')), 100, 100, 0.0),  # Inf coordinate
])
def test_confidence_boundary_conditions(pupil, w, h, expected):
    """
    Verify that boundary conditions, NaNs, infinities, and invalid dimensions
    produce exact 0.0 confidence without exceptions.
    """
    conf = score_eye_confidence(pupil=pupil, eye_w=w, eye_h=h)
    assert conf == expected


# ==============================================================================
# Scenario 15: Backward Compatibility of Existing Public APIs
# ==============================================================================

def test_backward_compatibility_calculate_gaze_direction():
    """
    Verify that calculate_gaze_direction retains its default behavior:
    single-eye inputs return GAZE_UNCERTAIN unless allow_single_eye=True is explicitly set.
    """
    base = ((0.50, 0.40), (0.50, 0.40))
    pupil = (50, 40)

    # Legacy default call with one eye missing -> strictly GAZE_UNCERTAIN
    assert calculate_gaze_direction(pupil, 100, 100, None, 100, 100, calibrated_gaze=base) == GAZE_UNCERTAIN
    assert calculate_gaze_direction(None, 100, 100, pupil, 100, 100, calibrated_gaze=base) == GAZE_UNCERTAIN

    # Explicit allow_single_eye=True with high confidence -> evaluates single eye
    res_l = calculate_gaze_direction(
        (75, 40), 100, 100, None, 100, 100,
        calibrated_gaze=base, allow_single_eye=True, left_conf=0.85
    )
    assert res_l == GAZE_LEFT


# ==============================================================================
# Scenario 16: Low-Confidence Observations Cannot Produce High-Confidence Direction
# ==============================================================================

def test_low_confidence_cannot_produce_high_confidence_classification():
    """
    Verify that even if pupil displacement is large (e.g. at extreme edge),
    if confidence is low (e.g. poor contrast or blurry contour), fallback is forbidden
    and direction is GAZE_UNCERTAIN.
    """
    tracker = SingleEyeGazeTracker()
    base = ((0.50, 0.40), (0.50, 0.40))

    # Extreme displacement (norm_x = 0.90 -> diff = -0.40) but confidence = 0.40
    res = tracker.update(
        left_pupil=(90, 40), left_w=100, left_h=100,
        right_pupil=None, right_w=100, right_h=100,
        calibrated_gaze=base,
        left_conf=0.40, right_conf=0.0,
    )

    assert res.tracking_state == TRACKING_STATE_UNCERTAIN
    assert res.gaze_direction == GAZE_UNCERTAIN
    assert res.combined_confidence < 0.30
    assert res.is_single_eye_fallback is False


# ==============================================================================
# Scenario 17: End-to-End Landmark Gaze Analysis with Single-Eye Fallback
# ==============================================================================

def test_end_to_end_landmarks_single_eye_fallback():
    """
    Verify that analyze_gaze_from_landmarks successfully performs single-eye fallback
    when one eye is occluded, annotates frame, and populates diagnostics dictionary.
    """
    reset_default_gaze_tracker()
    tracker = get_default_gaze_tracker()
    tracker.reset()

    frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    # Left eye aperture: x in [30, 54], y in [30, 40]
    left_pts = [(30, 35), (36, 30), (48, 30), (54, 35), (48, 40), (36, 40)]
    # Right eye aperture: completely degenerate or blank
    right_pts = [(80, 35), (86, 30), (98, 30), (104, 35), (98, 40), (86, 40)]

    # Draw clear left eye sclera and dark pupil looking left
    cv2.fillPoly(frame, [np.array(left_pts)], (230, 230, 230))
    # Place pupil at x=48 in frame -> x in eye crop = 48 - 30 = 18. lw=24 -> norm = 18/24 = 0.75 -> Looking Left!
    cv2.circle(frame, (48, 35), 2, (20, 20, 20), -1)

    # Right eye is flat skin (no sclera or pupil disc -> low contrast / invalid)
    mock_landmarks = _make_mock_landmarks(left_pts, right_pts)
    calibrated_gaze = ((0.50, 0.40), (0.50, 0.40))

    annotated, direction = analyze_gaze_from_landmarks(
        frame,
        mock_landmarks,
        calibrated_gaze=calibrated_gaze
    )

    diag = get_last_gaze_diagnostics()
    assert diag is not None
    assert diag["face_detected"] is True
    assert diag["left_pupil_detected"] is True
    assert diag["right_pupil_detected"] is False
    assert diag["left_confidence"] >= 0.70
    assert diag["tracking_state"] == TRACKING_STATE_LEFT_ONLY
    assert diag["is_single_eye_fallback"] is True
    assert direction == GAZE_LEFT
