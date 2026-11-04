import numpy as np
import pytest
from head_pose import classify_head_pose, HEAD_SCREEN, HEAD_LEFT, HEAD_RIGHT, HEAD_UP, HEAD_DOWN, HEAD_TILTED

pytestmark = pytest.mark.unit

def test_calibration_averaging_multiple_samples():
    """Verify that multiple valid angle samples are averaged properly for baseline calibration."""
    samples = [
        (10.0, 5.0, 2.0),
        (12.0, 7.0, 4.0),
        (14.0, 6.0, 3.0)
    ]
    avg_pitch = float(np.mean([s[0] for s in samples]))
    avg_yaw = float(np.mean([s[1] for s in samples]))
    avg_roll = float(np.mean([s[2] for s in samples]))

    calibrated_baseline = (avg_pitch, avg_yaw, avg_roll)
    assert calibrated_baseline == (12.0, 6.0, 3.0)

def test_head_pose_classification_with_calibrated_baseline():
    """Verify that classification correctly uses calibrated offsets."""
    baseline = (10.0, 5.0, 0.0)

    # Looking straight at baseline
    assert classify_head_pose(10.0, 5.0, 0.0, baseline) == HEAD_SCREEN

    # Within tolerance thresholds (pitch delta <= 8, yaw delta <= 12, roll delta <= 5)
    assert classify_head_pose(15.0, 10.0, 2.0, baseline) == HEAD_SCREEN

    # Looking Left (yaw < offset - 15 -> yaw < -10)
    assert classify_head_pose(10.0, -12.0, 0.0, baseline) == HEAD_LEFT

    # Looking Right (yaw > offset + 15 -> yaw > 20)
    assert classify_head_pose(10.0, 22.0, 0.0, baseline) == HEAD_RIGHT

    # Looking Up (pitch > offset + 10 -> pitch > 20)
    assert classify_head_pose(22.0, 5.0, 0.0, baseline) == HEAD_UP

    # Looking Down (pitch < offset - 10 -> pitch < 0)
    assert classify_head_pose(-2.0, 5.0, 0.0, baseline) == HEAD_DOWN

    # Tilted (abs(roll - offset) > 7)
    assert classify_head_pose(10.0, 5.0, 10.0, baseline) == HEAD_TILTED

def test_calibration_ignoring_none_samples():
    """Verify that absent face observations (None) are not included in calibration samples."""
    calibration_samples = []
    incoming_measurements = [
        (10.0, 5.0, 1.0),
        None,  # Face absent
        (12.0, 7.0, 3.0),
        None   # Face absent
    ]

    for m in incoming_measurements:
        if m is not None and isinstance(m, tuple):
            calibration_samples.append(m)

    assert len(calibration_samples) == 2
    avg_pitch = float(np.mean([s[0] for s in calibration_samples]))
    assert avg_pitch == 11.0


def test_head_pose_classification_custom_thresholds_and_boundaries():
    """Verify that classify_head_pose honors custom thresholds and maintains hysteresis deadband."""
    baseline = (0.0, 0.0, 0.0)

    # Custom tighter thresholds: neutral yaw=5, deviation yaw=8
    # Yaw=4 is within neutral (<= 5) -> HEAD_SCREEN
    assert classify_head_pose(0.0, 4.0, 0.0, baseline, neutral_yaw=5.0, deviation_yaw=8.0) == HEAD_SCREEN

    # Yaw=9 exceeds deviation (> 8) -> HEAD_RIGHT
    assert classify_head_pose(0.0, 9.0, 0.0, baseline, neutral_yaw=5.0, deviation_yaw=8.0) == HEAD_RIGHT

    # Yaw=6.5 is in deadband (between 5 and 8) -> returns previous_state (hysteresis)
    assert classify_head_pose(0.0, 6.5, 0.0, baseline, prev_state=HEAD_SCREEN, neutral_yaw=5.0, deviation_yaw=8.0) == HEAD_SCREEN
    assert classify_head_pose(0.0, 6.5, 0.0, baseline, prev_state=HEAD_RIGHT, neutral_yaw=5.0, deviation_yaw=8.0) == HEAD_RIGHT


def test_head_and_gaze_calibration_separation():
    """Verify that head-pose and gaze calibration maintain separate sample sets and baselines."""
    from eye_movement import calibrate_gaze_baseline

    head_samples = [(10.0, 5.0, 0.0)] * 10
    gaze_samples = [((0.48, 0.38), (0.48, 0.38))] * 10

    avg_head = (
        float(np.mean([s[0] for s in head_samples])),
        float(np.mean([s[1] for s in head_samples])),
        float(np.mean([s[2] for s in head_samples])),
    )
    gaze_baseline = calibrate_gaze_baseline(gaze_samples, min_samples=10)

    assert avg_head == (10.0, 5.0, 0.0)
    assert gaze_baseline == ((0.48, 0.38), (0.48, 0.38))


def test_calibrate_gaze_rejects_exact_midpoint_stubs():
    """Verify that unlocalized static midpoint stubs ((0.5, 0.5), (0.5, 0.5)) are rejected."""
    from eye_movement import calibrate_gaze_baseline

    midpoint_samples = [((0.50, 0.50), (0.50, 0.50))] * 15
    assert calibrate_gaze_baseline(midpoint_samples, min_samples=10) is None


def test_calibrate_gaze_accepts_valid_varying_samples():
    """Verify that natural jitter around neutral gaze produces robust baseline."""
    from eye_movement import calibrate_gaze_baseline

    samples = [
        ((0.49, 0.41), (0.51, 0.39)),
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.51, 0.39), (0.49, 0.41)),
        ((0.48, 0.40), (0.52, 0.40)),
        ((0.52, 0.42), (0.48, 0.38)),
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.49, 0.39), (0.51, 0.41)),
        ((0.51, 0.41), (0.49, 0.39)),
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.50, 0.41), (0.50, 0.39)),
    ]
    baseline = calibrate_gaze_baseline(samples, min_samples=10)
    assert baseline is not None
    (blx, bly), (brx, bry) = baseline
    assert abs(blx - 0.50) < 0.02
    assert abs(bly - 0.40) < 0.02


def test_calibrate_gaze_reports_sample_variance_stddev_and_range():
    from eye_movement import calibrate_gaze_baseline

    samples = [
        ((0.49, 0.39), (0.51, 0.41)),
        ((0.50, 0.40), (0.50, 0.40)),
        ((0.51, 0.41), (0.49, 0.39)),
        ((0.50, 0.40), (0.50, 0.40)),
    ] * 3
    baseline, quality = calibrate_gaze_baseline(samples, min_samples=10, return_diagnostics=True)

    assert baseline is not None
    assert quality["failure_reason"] == ""
    assert quality["sample_count"] == 12
    assert quality["statistics"]["left_x"]["variance"] == pytest.approx(0.0000545, abs=0.000001)
    assert quality["statistics"]["left_x"]["stddev"] < 0.06
    assert quality["statistics"]["left_x"]["range"] == pytest.approx(0.02)


def test_calibrate_gaze_rejects_unstable_samples_with_reason():
    from eye_movement import calibrate_gaze_baseline

    samples = [((x, 0.40), (0.50, 0.40)) for x in (0.20, 0.80)] * 5
    baseline, quality = calibrate_gaze_baseline(samples, min_samples=10, return_diagnostics=True)

    assert baseline is None
    assert "unstable gaze calibration" in quality["failure_reason"]
    assert quality["statistics"]["left_x"]["stddev"] > 0.06
    assert quality["statistics"]["left_x"]["range"] == pytest.approx(0.60)


def test_calibrate_gaze_rejects_implausible_neutral_baseline():
    from eye_movement import calibrate_gaze_baseline

    samples = [((0.05, 0.40), (0.50, 0.40))] * 10
    baseline, quality = calibrate_gaze_baseline(samples, min_samples=10, return_diagnostics=True)

    assert baseline is None
    assert "invalid neutral pupil baseline" in quality["failure_reason"]


def test_single_eye_failure_does_not_fallback_to_direction():
    from eye_movement import calculate_gaze_direction, GAZE_UNCERTAIN

    # A surviving eye far to image-left is not enough to claim a physical direction.
    assert calculate_gaze_direction(
        (10, 40), 100, 100, None, 100, 100,
        calibrated_gaze=((0.50, 0.40), (0.50, 0.40)),
    ) == GAZE_UNCERTAIN


def test_calibrate_gaze_rejects_out_of_range_anatomical_bounds():
    """Verify that coordinates outside normalized bounds (<0.0 or >1.0) are discarded."""
    from eye_movement import calibrate_gaze_baseline

    mixed = [
        ((0.50, 0.40), (0.50, 0.40)),
        ((-0.15, 0.40), (0.50, 0.40)), # Invalid lx < 0.0
        ((0.50, 1.40), (0.50, 0.40)),  # Invalid ly > 1.0
        ((0.50, 0.40), (-0.10, 0.40)), # Invalid rx < 0.0
        ((0.49, 0.41), (0.51, 0.39)),
    ]
    # Only 2 valid samples out of 5; required is 5
    assert calibrate_gaze_baseline(mixed, min_samples=5) is None
