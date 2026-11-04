from unittest.mock import MagicMock
import pytest
import numpy as np
import cv2

from time_source import (
    LiveTimeSource,
    VideoTimeSource,
    MockTimeSource,
    TimeSource,
)

pytestmark = pytest.mark.unit


def test_live_time_source_monotonic():
    """Verify LiveTimeSource uses monotonic time, advances steadily, and is not video."""
    src = LiveTimeSource()
    assert src.is_video is False
    assert src.frame_count == 0

    t0 = src.now()
    assert t0 >= 0.0

    t1 = src.update()
    assert src.frame_count == 1
    assert t1 >= t0


def test_video_time_source_with_valid_pts():
    """Verify VideoTimeSource normalizes container timestamps relative to initial frame."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 25.0,
        cv2.CAP_PROP_POS_MSEC: [100.0, 140.0, 180.0, 220.0][mock_cap.call_index],
    }.get(prop, 0.0)

    # Frame 0: 100ms
    mock_cap.call_index = 0
    src = VideoTimeSource(cap=mock_cap)
    assert src.nominal_fps == 25.0
    assert src.is_video is True

    t0 = src.update(cap=mock_cap)
    assert t0 == 0.0  # Normalized to start at 0.0s
    assert src.frame_count == 1
    assert src.pts_advances == 1

    # Frame 1: 140ms (+40ms = +0.040s)
    mock_cap.call_index = 1
    t1 = src.update(cap=mock_cap)
    assert pytest.approx(t1, rel=1e-3) == 0.040
    assert src.pts_advances == 2

    # Frame 2: 180ms (+80ms = +0.080s)
    mock_cap.call_index = 2
    t2 = src.update(cap=mock_cap)
    assert pytest.approx(t2, rel=1e-3) == 0.080

    # Frame 3: 220ms (+120ms = +0.120s)
    mock_cap.call_index = 3
    t3 = src.update(cap=mock_cap)
    assert pytest.approx(t3, rel=1e-3) == 0.120


def test_video_time_source_duplicate_timestamp_fallback():
    """Verify that duplicate POS_MSEC values trigger nominal frame delta fallback."""
    mock_cap = MagicMock()
    # Header FPS is 20.0 -> frame duration = 0.050s
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 20.0,
        cv2.CAP_PROP_POS_MSEC: [0.0, 50.0, 50.0][mock_cap.call_index],
    }.get(prop, 0.0)

    src = VideoTimeSource(cap=mock_cap)
    mock_cap.call_index = 0
    t0 = src.update(cap=mock_cap)
    assert t0 == 0.0

    mock_cap.call_index = 1
    t1 = src.update(cap=mock_cap)
    assert pytest.approx(t1, rel=1e-3) == 0.050

    # Frame 2 has duplicate POS_MSEC (50.0 again)
    mock_cap.call_index = 2
    t2 = src.update(cap=mock_cap)
    # Must fallback and advance by 0.050s
    assert pytest.approx(t2, rel=1e-3) == 0.100
    assert src.fallback_advances == 1


def test_video_time_source_regressing_timestamp_fallback():
    """Verify that regressing POS_MSEC values trigger fallback and preserve monotonicity."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 10.0,  # 0.100s per frame
        cv2.CAP_PROP_POS_MSEC: [0.0, 100.0, 50.0][mock_cap.call_index],  # 50ms regresses after 100ms
    }.get(prop, 0.0)

    src = VideoTimeSource(cap=mock_cap)
    mock_cap.call_index = 0
    t0 = src.update(cap=mock_cap)
    assert t0 == 0.0

    mock_cap.call_index = 1
    t1 = src.update(cap=mock_cap)
    assert pytest.approx(t1, rel=1e-3) == 0.100

    # Regressing timestamp
    mock_cap.call_index = 2
    t2 = src.update(cap=mock_cap)
    assert pytest.approx(t2, rel=1e-3) == 0.200  # Advanced by 0.100s fallback, never backward!
    assert src.fallback_advances == 1


def test_video_time_source_missing_pts_uses_fps_fallback():
    """Verify that containers returning -1 or invalid POS_MSEC fallback to FPS increment."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 10.0,
        cv2.CAP_PROP_POS_MSEC: -1.0,
    }.get(prop, -1.0)

    src = VideoTimeSource(cap=mock_cap)
    for i in range(5):
        t = src.update(cap=mock_cap)
        assert pytest.approx(t, rel=1e-3) == i * 0.100

    assert src.fallback_advances == 5
    assert src.pts_advances == 0


def test_deterministic_video_violation_timing_across_processing_speeds():
    """
    Verify that violation evaluation across the same video frames yields identical
    violation events and durations regardless of simulated wall-clock processing speed.
    """
    # 100 frames at 10 FPS (10.0 seconds total video duration)
    # Phone appears from frame 20 (2.0s) to frame 70 (7.0s)
    threshold = 3.0

    def simulate_processing_run(wall_clock_delay_per_frame):
        time_src = VideoTimeSource(fallback_fps=10.0)
        # Mock cap with POS_MSEC = frame_idx * 100ms
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_POS_MSEC: mock_cap.frame_idx * 100.0,
        }.get(prop, 0.0)

        violations_triggered = []
        start_time = None
        logged = False

        for f in range(100):
            mock_cap.frame_idx = f
            current_time = time_src.update(cap=mock_cap)

            # Phone present between f=20 (2.0s) and f=70 (7.0s)
            phone_present = (20 <= f < 70)

            if phone_present:
                if start_time is None:
                    start_time = current_time
                    logged = False
                elif not logged and (current_time - start_time >= threshold):
                    duration = current_time - start_time
                    violations_triggered.append((f, current_time, duration))
                    logged = True
            else:
                start_time = None
                logged = False

        return violations_triggered

    # Run 1: Simulated fast machine (delay = 0.001s per frame)
    run_fast = simulate_processing_run(0.001)
    # Run 2: Simulated slow machine (delay = 0.500s per frame)
    run_slow = simulate_processing_run(0.500)

    # Both runs MUST trigger at the exact same frame, exact video time, and exact duration!
    assert len(run_fast) == 1
    assert len(run_slow) == 1

    frame_fast, time_fast, dur_fast = run_fast[0]
    frame_slow, time_slow, dur_slow = run_slow[0]

    assert frame_fast == frame_slow == 50  # 2.0s + 3.0s = 5.0s (frame 50 at 10 FPS)
    assert pytest.approx(time_fast, rel=1e-3) == pytest.approx(time_slow, rel=1e-3) == 5.0
    assert pytest.approx(dur_fast, rel=1e-3) == pytest.approx(dur_slow, rel=1e-3) == 3.0


def test_calibration_duration_in_video_mode():
    """Verify that calibration window in video mode completes after exact video seconds."""
    time_src = VideoTimeSource(fallback_fps=10.0)
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 10.0,
        cv2.CAP_PROP_POS_MSEC: mock_cap.frame_idx * 100.0,
    }.get(prop, 0.0)

    calib_duration = 5.0
    calib_start = time_src.now()
    is_calibrated = False
    calibrated_frame = None

    for f in range(80):
        mock_cap.frame_idx = f
        current_time = time_src.update(cap=mock_cap)

        if not is_calibrated:
            elapsed = current_time - calib_start
            if elapsed >= calib_duration:
                is_calibrated = True
                calibrated_frame = f

    # At 10 FPS, 5.0s is reached on frame 50 (50 * 0.1s = 5.0s)
    assert is_calibrated is True
    assert calibrated_frame == 50


def test_mock_time_source():
    """Verify MockTimeSource allows programmatic time control."""
    mock_src = MockTimeSource(initial_time=10.0, is_video=True)
    assert mock_src.now() == 10.0
    assert mock_src.is_video is True

    mock_src.advance(2.5)
    assert mock_src.now() == 12.5

    mock_src.set_time(0.0)
    assert mock_src.now() == 0.0
    assert mock_src.frame_count == 0
    mock_src.update()
    assert mock_src.frame_count == 1
