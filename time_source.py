"""
Time Source Module for Cheating Surveillance System.

Provides deterministic time management for live webcam capture and offline video processing:
- LiveTimeSource: Real-time monotonic clock provider for live camera feeds.
- VideoTimeSource: Deterministic presentation timestamp provider for prerecorded video files,
  with automatic fallback to nominal frame-duration increments when container timestamps
  are missing, negative, duplicate, or regressing.
- MockTimeSource: Fully controlled time provider for unit testing.
"""

from abc import ABC, abstractmethod
import math
import time
from typing import Optional, Any
import cv2


class TimeSource(ABC):
    """Abstract base class for surveillance time providers."""

    @abstractmethod
    def now(self) -> float:
        """Returns the current elapsed time in seconds."""
        raise NotImplementedError

    @abstractmethod
    def update(self, frame=None, cap=None) -> float:
        """
        Advances or synchronizes time upon reading a new frame.
        Returns the updated time in seconds.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_video(self) -> bool:
        """True if the time source represents prerecorded video time."""
        raise NotImplementedError


class LiveTimeSource(TimeSource):
    """
    Monotonic clock provider for live webcam surveillance.
    Ensures steady, drift-free, non-regressing elapsed timing using time.monotonic().
    """

    def __init__(self, start_offset: float = 0.0):
        self._start_time = time.monotonic()
        self._offset = start_offset
        self._frame_count = 0

    def now(self) -> float:
        """Returns elapsed seconds since initialization."""
        return (time.monotonic() - self._start_time) + self._offset

    def update(self, frame=None, cap=None) -> float:
        """Increments frame count and returns current monotonic elapsed time."""
        self._frame_count += 1
        return self.now()

    @property
    def is_video(self) -> bool:
        return False

    @property
    def frame_count(self) -> int:
        return self._frame_count


class VideoTimeSource(TimeSource):
    """
    Deterministic time provider for prerecorded video files.

    Primary timing uses container presentation timestamps (CAP_PROP_POS_MSEC).
    If container timestamps are missing, invalid, duplicate, or regressing, the source
    falls back safely to nominal frame duration intervals (1.0 / fps) based on the
    video header FPS (or a documented default fallback of 30.0 FPS).
    """

    DEFAULT_FALLBACK_FPS: float = 30.0

    def __init__(self, cap: Optional[Any] = None, fallback_fps: Optional[float] = None):
        self.fallback_fps = (
            fallback_fps if (fallback_fps is not None and fallback_fps > 0) else self.DEFAULT_FALLBACK_FPS
        )

        # Resolve video header FPS if cap provided
        detected_fps = None
        if cap is not None and hasattr(cap, "get"):
            try:
                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                if raw_fps is not None and math.isfinite(raw_fps) and raw_fps > 0.1:
                    detected_fps = float(raw_fps)
            except Exception:
                detected_fps = None

        self.nominal_fps: float = detected_fps if detected_fps is not None else self.fallback_fps
        self.frame_duration: float = 1.0 / self.nominal_fps

        self._current_time: float = 0.0
        self._frame_count: int = 0
        self._initial_msec: Optional[float] = None
        self._last_valid_msec: Optional[float] = None
        self._pts_advances: int = 0
        self._fallback_advances: int = 0

    def now(self) -> float:
        """Returns the current elapsed video time in seconds."""
        return self._current_time

    def update(self, frame=None, cap=None) -> float:
        """
        Advances video presentation time upon reading a frame.
        Evaluates container POS_MSEC and applies safe monotonic fallback if needed.
        """
        raw_msec = None
        if cap is not None and hasattr(cap, "get"):
            try:
                msec_val = cap.get(cv2.CAP_PROP_POS_MSEC)
                if msec_val is not None and math.isfinite(msec_val) and msec_val >= 0.0:
                    raw_msec = float(msec_val)
            except Exception:
                raw_msec = None

        # First frame initialization
        if self._frame_count == 0:
            self._frame_count = 1
            if raw_msec is not None:
                self._initial_msec = raw_msec
                self._last_valid_msec = raw_msec
                self._current_time = 0.0
                self._pts_advances += 1
            else:
                self._initial_msec = 0.0
                self._last_valid_msec = 0.0
                self._current_time = 0.0
                self._fallback_advances += 1
            return self._current_time

        # Subsequent frames
        self._frame_count += 1

        # Check if container timestamp is strictly advancing monotonically
        if (
            raw_msec is not None
            and self._last_valid_msec is not None
            and raw_msec > self._last_valid_msec
        ):
            pts_time = (raw_msec - self._initial_msec) / 1000.0
            if pts_time > self._current_time:
                self._current_time = pts_time
                self._last_valid_msec = raw_msec
                self._pts_advances += 1
                return self._current_time

        # Fallback: advance by nominal frame duration
        self._current_time += self.frame_duration
        self._fallback_advances += 1
        return self._current_time

    @property
    def is_video(self) -> bool:
        return True

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def pts_advances(self) -> int:
        """Number of frames advanced via container presentation timestamps."""
        return self._pts_advances

    @property
    def fallback_advances(self) -> int:
        """Number of frames advanced via nominal frame-duration fallback."""
        return self._fallback_advances


class MockTimeSource(TimeSource):
    """
    Deterministic time provider for unit and regression testing.
    Allows exact programmatic time setting and increments.
    """

    def __init__(self, initial_time: float = 0.0, is_video: bool = False):
        self._current_time = initial_time
        self._is_video = is_video
        self._frame_count = 0

    def now(self) -> float:
        return self._current_time

    def set_time(self, t: float) -> None:
        self._current_time = float(t)

    def advance(self, delta_sec: float) -> float:
        self._current_time += float(delta_sec)
        return self._current_time

    def update(self, frame=None, cap=None) -> float:
        self._frame_count += 1
        return self._current_time

    @property
    def is_video(self) -> bool:
        return self._is_video

    @property
    def frame_count(self) -> int:
        return self._frame_count
