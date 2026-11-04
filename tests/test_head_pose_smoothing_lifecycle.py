"""
Phase 10.13.13 — GAP-05 Focused Unit Tests:
Head-Pose Smoothing State Lifecycle and Reset Behavior.

Verifies:
1. Valid head-pose samples populate smoothing history.
2. Face loss clears relevant history.
3. Reacquisition begins with fresh state.
4. New monitoring session starts with clean state.
5. Processing failure invalidates relevant continuity.
6. Repeated reset calls do not cause errors (idempotence).
7. Existing smoothing behavior remains unchanged for uninterrupted valid tracking.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from head_pose import (
    reset_head_pose_state,
    get_head_pose_history_counts,
    get_head_pose_previous_state,
    smooth_angle,
    yaw_history,
    pitch_history,
    roll_history,
    process_head_pose,
    HEAD_SCREEN,
    HEAD_LEFT,
    HEAD_RIGHT,
)
from vision_pipeline import VisionPipeline

pytestmark = pytest.mark.unit


class _FakeRect:
    def left(self): return 100
    def top(self): return 100
    def right(self): return 200
    def bottom(self): return 200


def _make_dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_valid_samples_populate_smoothing_history():
    """Verify that consecutive valid head pose angles populate the history deques."""
    reset_head_pose_state()
    assert get_head_pose_history_counts() == (0, 0, 0)

    # Add 3 samples to yaw_history
    smooth_angle(yaw_history, 10.0)
    smooth_angle(yaw_history, 12.0)
    smooth_angle(yaw_history, 14.0)

    # Add 2 samples to pitch_history
    smooth_angle(pitch_history, 5.0)
    smooth_angle(pitch_history, 6.0)

    assert get_head_pose_history_counts() == (3, 2, 0)


def test_face_loss_clears_smoothing_history():
    """Verify that face loss (face_count == 0) resets smoothing history and previous_state."""
    reset_head_pose_state()
    # Populate state
    smooth_angle(yaw_history, 25.0)
    smooth_angle(pitch_history, 15.0)
    smooth_angle(roll_history, 5.0)
    assert get_head_pose_history_counts() == (1, 1, 1)

    # Process empty/no-face frame through pipeline
    pipe = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock())
    pipe.process_face(_make_dummy_frame())

    # History must be cleared
    assert get_head_pose_history_counts() == (0, 0, 0)
    assert get_head_pose_previous_state() == HEAD_SCREEN


def test_face_loss_in_process_head_pose_clears_history():
    """Verify that process_head_pose clears history on missing face."""
    reset_head_pose_state()
    smooth_angle(yaw_history, 30.0)
    assert get_head_pose_history_counts()[0] == 1

    # Call with detector returning no faces
    empty_det = MagicMock(return_value=[])
    process_head_pose(
        _make_dummy_frame(), calibrated_angles=(0, 0, 0), detector=empty_det, predictor=MagicMock()
    )

    assert get_head_pose_history_counts() == (0, 0, 0)


def test_reacquisition_begins_with_fresh_state():
    """Verify that a reacquired face does not inherit smoothed angles from previous face."""
    reset_head_pose_state()

    # Face 1: looking far right (+30.0 deg)
    for _ in range(10):
        val = smooth_angle(yaw_history, 30.0)
    assert val == 30.0
    assert len(yaw_history) == 10

    # Face is lost
    reset_head_pose_state()
    assert len(yaw_history) == 0

    # Face 2 (reacquired): looking straight center (0.0 deg)
    new_val = smooth_angle(yaw_history, 0.0)
    # If stale history had persisted, new_val would be (9*30 + 0)/10 = 27.0 deg.
    # With clean lifecycle, new_val is exactly 0.0 deg!
    assert len(yaw_history) == 1
    assert new_val == 0.0


def test_new_session_starts_with_clean_state():
    """Verify that VisionPipeline.reset() restores completely clean state."""
    smooth_angle(yaw_history, 45.0)
    smooth_angle(pitch_history, -20.0)
    smooth_angle(roll_history, 10.0)

    pipe = VisionPipeline()
    pipe.reset()

    assert get_head_pose_history_counts() == (0, 0, 0)
    assert get_head_pose_previous_state() == HEAD_SCREEN


def test_processing_failure_invalidates_smoothing_continuity():
    """Verify that a detector or predictor exception clears smoothing history."""
    reset_head_pose_state()
    smooth_angle(yaw_history, 15.0)
    smooth_angle(pitch_history, 10.0)
    assert get_head_pose_history_counts() == (1, 1, 0)

    # Detector fails
    failing_det = MagicMock(side_effect=RuntimeError("Hardware fault"))
    pipe = VisionPipeline(detector=failing_det, predictor=MagicMock())
    pipe.process_face(_make_dummy_frame())

    assert get_head_pose_history_counts() == (0, 0, 0)


def test_repeated_reset_calls_safe_and_idempotent():
    """Verify that calling reset_head_pose_state repeatedly does not cause any errors."""
    for _ in range(10):
        reset_head_pose_state()

    assert get_head_pose_history_counts() == (0, 0, 0)
    assert get_head_pose_previous_state() == HEAD_SCREEN


def test_uninterrupted_tracking_preserves_sliding_window_smoothing():
    """Verify that existing smoothing mathematics are preserved for valid tracking."""
    reset_head_pose_state()

    # Step input from 0 to 10
    v1 = smooth_angle(yaw_history, 0.0)
    assert v1 == 0.0

    v2 = smooth_angle(yaw_history, 10.0)
    assert v2 == 5.0  # mean(0, 10) = 5.0

    v3 = smooth_angle(yaw_history, 20.0)
    assert v3 == 10.0  # mean(0, 10, 20) = 10.0
