import time
import pytest
from telemetry import PerformanceTelemetry

pytestmark = pytest.mark.unit


def test_telemetry_latency_and_fps_calculation():
    """Verify latency recording and rolling FPS computation."""
    telem = PerformanceTelemetry(window_size=10, console_interval=20)
    assert telem.total_frames == 0
    assert telem.fps == 0.0

    # Simulate 5 frames with 20ms duration (0.020s)
    for _ in range(5):
        telem.record_frame(0.020, is_yolo_inference=True, is_phone_positive=False)

    assert telem.total_frames == 5
    assert telem.yolo_inferences == 5
    assert telem.phone_positive_frames == 0
    assert pytest.approx(telem.avg_latency_ms, rel=1e-2) == 20.0
    assert pytest.approx(telem.latency_ms, rel=1e-2) == 20.0


def test_telemetry_summary_statistics():
    """Verify summary dictionary calculation including p50 and p95."""
    telem = PerformanceTelemetry(window_size=50)

    # Record 100 frames with varying durations
    for i in range(1, 101):
        # Durations from 1ms to 100ms
        telem.record_frame(i / 1000.0, is_yolo_inference=(i % 3 == 0), is_phone_positive=(i > 90))

    summary = telem.get_summary()
    assert summary["total_frames"] == 100
    assert summary["yolo_inferences"] == 33
    assert summary["phone_positive_frames"] == 10
    assert pytest.approx(summary["avg_latency_ms"], rel=1e-1) == 50.5
    assert pytest.approx(summary["p50_latency_ms"], rel=1e-1) == 51.0
    assert pytest.approx(summary["p95_latency_ms"], rel=1e-1) == 96.0


def test_telemetry_formatting_and_console_trigger():
    """Verify overlay string and headless console logging triggers."""
    telem = PerformanceTelemetry(window_size=5, console_interval=10)

    telem.record_frame(0.015, is_yolo_inference=True)
    overlay = telem.format_overlay()
    assert "Proc FPS:" in overlay
    assert "Latency: 15.0ms" in overlay

    # Console trigger should be false until console_interval is hit
    assert telem.should_log_console() is False

    for _ in range(9):
        telem.record_frame(0.015)

    assert telem.total_frames == 10
    assert telem.should_log_console() is True
    console_log = telem.format_console_log()
    assert "[Telemetry] Frame 10" in console_log
    assert "Avg Latency: 15.0ms" in console_log
