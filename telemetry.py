"""
Performance Telemetry Module for Cheating Surveillance System.

Measures frame processing latency and effective processed FPS using a monotonic clock
over a rolling window. Telemetry tracks frame-processing compute time only and does
not claim to represent camera capture or hardware I/O latency.
"""

from collections import deque
import time
from typing import Dict, Any, List


class PerformanceTelemetry:
    """
    Lightweight telemetry tracker for frame processing latency and FPS.
    """

    def __init__(self, window_size: int = 30, console_interval: int = 60):
        self.window_size: int = max(1, window_size)
        self.console_interval: int = max(1, console_interval)
        self._latencies: deque = deque(maxlen=self.window_size)
        self._frame_times: deque = deque(maxlen=self.window_size)

        self.total_frames: int = 0
        self.yolo_inferences: int = 0
        self.phone_positive_frames: int = 0
        self._all_latencies: List[float] = []

        self._current_fps: float = 0.0
        self._avg_latency_ms: float = 0.0
        self._last_latency_ms: float = 0.0

    def start_frame(self) -> float:
        """Records the monotonic start timestamp of frame processing."""
        return time.perf_counter()

    def record_frame(
        self,
        processing_duration_sec: float,
        is_yolo_inference: bool = False,
        is_phone_positive: bool = False,
    ) -> None:
        """
        Records the completion of a processed frame.
        """
        now = time.perf_counter()
        latency_ms = processing_duration_sec * 1000.0
        self._last_latency_ms = latency_ms
        self._latencies.append(latency_ms)
        self._frame_times.append(now)
        self._all_latencies.append(latency_ms)

        self.total_frames += 1
        if is_yolo_inference:
            self.yolo_inferences += 1
        if is_phone_positive:
            self.phone_positive_frames += 1

        # Calculate metrics over the rolling window
        if len(self._latencies) > 0:
            self._avg_latency_ms = sum(self._latencies) / len(self._latencies)

        if len(self._frame_times) >= 2:
            time_span = self._frame_times[-1] - self._frame_times[0]
            if time_span > 0:
                self._current_fps = (len(self._frame_times) - 1) / time_span
            else:
                self._current_fps = 0.0
        else:
            self._current_fps = 0.0

    @property
    def fps(self) -> float:
        """Effective processed FPS over the rolling window."""
        return self._current_fps

    @property
    def latency_ms(self) -> float:
        """Processing latency of the most recently completed frame in milliseconds."""
        return self._last_latency_ms

    @property
    def avg_latency_ms(self) -> float:
        """Average processing latency over the rolling window in milliseconds."""
        return self._avg_latency_ms

    def get_summary(self) -> Dict[str, Any]:
        """Returns overall session telemetry metrics."""
        if not self._all_latencies:
            return {
                "total_frames": 0,
                "yolo_inferences": 0,
                "phone_positive_frames": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "fps": 0.0,
            }

        sorted_latencies = sorted(self._all_latencies)
        p50_idx = int(0.50 * len(sorted_latencies))
        p95_idx = min(int(0.95 * len(sorted_latencies)), len(sorted_latencies) - 1)

        return {
            "total_frames": self.total_frames,
            "yolo_inferences": self.yolo_inferences,
            "phone_positive_frames": self.phone_positive_frames,
            "avg_latency_ms": sum(self._all_latencies) / len(self._all_latencies),
            "p50_latency_ms": sorted_latencies[p50_idx],
            "p95_latency_ms": sorted_latencies[p95_idx],
            "fps": self._current_fps,
        }

    def format_overlay(self) -> str:
        """Returns formatted string suitable for GUI display."""
        return f"Proc FPS: {self._current_fps:.1f} | Latency: {self._last_latency_ms:.1f}ms"

    def should_log_console(self) -> bool:
        """Returns True if the current frame count hits the console logging cadence."""
        return self.total_frames > 0 and (self.total_frames % self.console_interval == 0)

    def format_console_log(self) -> str:
        """Returns formatted console log message for headless operation."""
        return (
            f"[Telemetry] Frame {self.total_frames} | "
            f"Proc FPS: {self._current_fps:.1f} | "
            f"Avg Latency: {self._avg_latency_ms:.1f}ms | "
            f"YOLO Inferences: {self.yolo_inferences}"
        )
