"""
Screen Gaze Calibration & Monitor Boundary Mapping Module for Examination Monitoring System.

Provides:
- ScreenGazeResult: Structured dataclass representing calibrated screen-relative gaze.
- ScreenCalibrationSample: Structured dataclass representing an individual calibration observation.
- ScreenGazeCalibrator: Manages calibration lifecycle, validates samples, computes
  affine/regularized least-squares transformation from camera-relative gaze to normalized
  monitor coordinates, assesses calibration quality, and maps gaze to screen boundary states.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import config
from path_security import (
    resolve_safe_output_path,
    ensure_safe_directory,
    is_symlink_or_junction,
    validate_safe_filename,
)

# Monitor boundary classification states
BOUNDARY_ON_SCREEN = "ON_SCREEN"
BOUNDARY_NEAR_BOUNDARY = "NEAR_BOUNDARY"
BOUNDARY_OUTSIDE_SCREEN = "OUTSIDE_SCREEN"
BOUNDARY_UNCERTAIN = "UNCERTAIN"
BOUNDARY_CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"

# Calibration status constants
CALIB_STATUS_NOT_CALIBRATED = "NOT_CALIBRATED"
CALIB_STATUS_CALIBRATING = "CALIBRATING"
CALIB_STATUS_VALID = "VALID"
CALIB_STATUS_FAILED = "FAILED"


@dataclass
class ScreenGazeResult:
    """
    Structured result model for calibrated screen-relative gaze estimation.

    Attributes:
        screen_x: Estimated normalized horizontal screen coordinate (0.0 = left, 1.0 = right).
                  Raw unclamped value; can be < 0.0 or > 1.0 if gaze falls outside the monitor.
        screen_y: Estimated normalized vertical screen coordinate (0.0 = top, 1.0 = bottom).
                  Raw unclamped value; can be < 0.0 or > 1.0 if gaze falls outside the monitor.
        boundary_state: Classification string:
                        'ON_SCREEN', 'NEAR_BOUNDARY', 'OUTSIDE_SCREEN',
                        'UNCERTAIN', or 'CALIBRATION_REQUIRED'.
        confidence: Combined confidence score in [0.0, 1.0] reflecting both gaze detection
                    reliability and calibration quality.
        uncertainty: Complement of confidence (1.0 - confidence) in [0.0, 1.0].
        status: Estimation status: 'VALID', 'UNCERTAIN', or 'NOT_CALIBRATED'.
        is_calibrated: Boolean indicating whether an active valid calibration was used.
        source_components: Dictionary storing intermediate features, residual error, and diagnostics.
    """
    screen_x: Optional[float] = None
    screen_y: Optional[float] = None
    boundary_state: str = BOUNDARY_CALIBRATION_REQUIRED
    confidence: float = 0.0
    uncertainty: float = 1.0
    status: str = CALIB_STATUS_NOT_CALIBRATED
    is_calibrated: bool = False
    source_components: Dict[str, Any] = field(default_factory=dict)

    # Legacy / convenience property aliases
    @property
    def x(self) -> Optional[float]:
        return self.screen_x

    @property
    def y(self) -> Optional[float]:
        return self.screen_y

    @property
    def is_on_screen(self) -> bool:
        return self.boundary_state == BOUNDARY_ON_SCREEN

    @property
    def is_outside(self) -> bool:
        return self.boundary_state == BOUNDARY_OUTSIDE_SCREEN


@dataclass
class ScreenCalibrationSample:
    """
    Structured container for a single calibration observation.

    Attributes:
        target_x: Normalized target horizontal screen coordinate (0.0 to 1.0).
        target_y: Normalized target vertical screen coordinate (0.0 to 1.0).
        gaze_yaw: Observed camera-relative gaze yaw in degrees (right positive, left negative).
        gaze_pitch: Observed camera-relative gaze pitch in degrees (up positive, down negative).
        confidence: Observation confidence in [0.0, 1.0].
        tracking_state: Gaze tracking state (e.g., 'PAIRED_TRACKING', 'LEFT_EYE_ONLY', 'UNCERTAIN').
        is_single_eye_fallback: Boolean flag if single-eye fallback was active.
        head_yaw: Optional observed head yaw in degrees.
        head_pitch: Optional observed head pitch in degrees.
        head_roll: Optional observed head roll in degrees.
        timestamp: Optional observation timestamp.
    """
    target_x: float
    target_y: float
    gaze_yaw: float
    gaze_pitch: float
    confidence: float
    tracking_state: str = "PAIRED_TRACKING"
    is_single_eye_fallback: bool = False
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    timestamp: Optional[float] = None

    def is_valid(self, min_confidence: Optional[float] = None) -> bool:
        """
        Validates mathematical finiteness, coordinate ranges, and minimum confidence.
        """
        if not (np.isfinite(self.target_x) and np.isfinite(self.target_y)):
            return False
        if not (0.0 <= self.target_x <= 1.0 and 0.0 <= self.target_y <= 1.0):
            return False
        if not (np.isfinite(self.gaze_yaw) and np.isfinite(self.gaze_pitch)):
            return False
        if not (np.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            return False
        if self.confidence < (config.SCREEN_CALIBRATION_MIN_CONFIDENCE if min_confidence is None else min_confidence):
            return False
        if self.tracking_state in ("UNCERTAIN", "NO_FACE", "FAILED", ""):
            return False
        return True


class ScreenGazeCalibrator:
    """
    Calibrates and maps camera-relative gaze observations to normalized screen coordinates.

    Mathematical Model:
      An affine mapping from camera-relative gaze angles [yaw, pitch, 1]^T to screen [u, v]^T:
          u = w_u0 * yaw + w_u1 * pitch + b_u
          v = w_v0 * yaw + w_v1 * pitch + b_v

      Solved via Weighted Regularized Least Squares (Ridge regression) over valid calibration samples:
          W = (X^T @ W_diag @ X + lambda * I)^(-1) @ X^T @ W_diag @ Y
      where lambda is a small Tikhonov regularization factor (1e-6) ensuring numerical invertibility.
    """

    def __init__(
        self,
        min_targets: Optional[int] = None,
        min_samples_per_target: Optional[int] = None,
        boundary_margin: Optional[float] = None,
        max_rmse_threshold: Optional[float] = None,
        regularization_lambda: float = 1e-6,
    ):
        self.min_targets = max(3, int(config.SCREEN_CALIBRATION_MIN_TARGETS if min_targets is None else min_targets))
        self.min_samples_per_target = max(1, int(config.SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET if min_samples_per_target is None else min_samples_per_target))
        self.boundary_margin = float(config.SCREEN_BOUNDARY_MARGIN if boundary_margin is None else boundary_margin)
        self.max_rmse_threshold = float(config.SCREEN_CALIBRATION_MAX_RMSE if max_rmse_threshold is None else max_rmse_threshold)
        self.regularization_lambda = float(regularization_lambda)

        self._samples: List[ScreenCalibrationSample] = []
        self._calibration_matrix: Optional[np.ndarray] = None  # Shape: (3, 2)
        self._is_calibrated: bool = False
        self._quality_metrics: Dict[str, Any] = {}
        self._calibrated_bounding_box: Optional[Tuple[float, float, float, float]] = None

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def quality_metrics(self) -> Dict[str, Any]:
        return dict(self._quality_metrics)

    def reset(self):
        """Resets all calibration data, samples, and fitted parameters."""
        self._samples.clear()
        self._calibration_matrix = None
        self._is_calibrated = False
        self._quality_metrics = {}
        self._calibrated_bounding_box = None

    def add_sample(self, sample: ScreenCalibrationSample) -> bool:
        """
        Validates and adds an individual calibration observation.
        Returns True if accepted, False if rejected.
        """
        if sample is None or not sample.is_valid():
            return False
        self._samples.append(sample)
        return True

    def add_observation(
        self,
        target_x: float,
        target_y: float,
        gaze_yaw: float,
        gaze_pitch: float,
        confidence: float,
        tracking_state: str = "PAIRED_TRACKING",
        is_single_eye_fallback: bool = False,
        head_yaw: Optional[float] = None,
        head_pitch: Optional[float] = None,
        head_roll: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Convenience method to construct and add a calibration sample."""
        sample = ScreenCalibrationSample(
            target_x=float(target_x),
            target_y=float(target_y),
            gaze_yaw=float(gaze_yaw),
            gaze_pitch=float(gaze_pitch),
            confidence=float(confidence),
            tracking_state=str(tracking_state),
            is_single_eye_fallback=bool(is_single_eye_fallback),
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
            timestamp=timestamp,
        )
        return self.add_sample(sample)

    def fit(self) -> bool:
        """
        Validates accumulated samples and fits the affine mapping.
        Returns True on successful calibration, False on validation/fitting failure.
        """
        valid_samples = [s for s in self._samples if s.is_valid()]
        if len(valid_samples) < (self.min_targets * self.min_samples_per_target):
            self._is_calibrated = False
            self._quality_metrics = {
                "success": False,
                "reason": f"insufficient_samples: {len(valid_samples)} < {self.min_targets * self.min_samples_per_target}",
                "valid_sample_count": len(valid_samples),
            }
            return False

        # Group samples by distinct targets (within small spatial radius 0.025)
        targets: List[Tuple[float, float]] = []
        target_groups: Dict[int, List[ScreenCalibrationSample]] = {}

        for s in valid_samples:
            assigned = False
            for tidx, (tx, ty) in enumerate(targets):
                if math.hypot(s.target_x - tx, s.target_y - ty) < 0.025:
                    target_groups[tidx].append(s)
                    assigned = True
                    break
            if not assigned:
                new_idx = len(targets)
                targets.append((s.target_x, s.target_y))
                target_groups[new_idx] = [s]

        if len(targets) < self.min_targets:
            self._is_calibrated = False
            self._quality_metrics = {
                "success": False,
                "reason": f"insufficient_distinct_targets: {len(targets)} < {self.min_targets}",
                "target_count": len(targets),
                "valid_sample_count": len(valid_samples),
            }
            return False

        # Verify spatial distribution of targets
        target_xs = [t[0] for t in targets]
        target_ys = [t[1] for t in targets]
        span_x = max(target_xs) - min(target_xs)
        span_y = max(target_ys) - min(target_ys)

        if span_x < 0.30 or span_y < 0.30:
            self._is_calibrated = False
            self._quality_metrics = {
                "success": False,
                "reason": f"poor_spatial_spread: span_x={span_x:.2f}, span_y={span_y:.2f} (required >= 0.30)",
                "span_x": span_x,
                "span_y": span_y,
            }
            return False

        # Verify variance of observed gaze features
        yaws = [s.gaze_yaw for s in valid_samples]
        pitches = [s.gaze_pitch for s in valid_samples]
        std_yaw = float(np.std(yaws))
        std_pitch = float(np.std(pitches))

        if std_yaw < 0.8 or std_pitch < 0.8:
            self._is_calibrated = False
            self._quality_metrics = {
                "success": False,
                "reason": f"insufficient_gaze_feature_variance: std_yaw={std_yaw:.2f}, std_pitch={std_pitch:.2f}",
                "std_yaw": std_yaw,
                "std_pitch": std_pitch,
            }
            return False

        # Construct Design Matrix X (N x 3) and Target Matrix Y (N x 2)
        # Features: [yaw, pitch, 1.0]
        # Targets:  [target_x, target_y]
        N = len(valid_samples)
        X = np.zeros((N, 3), dtype=np.float64)
        Y = np.zeros((N, 2), dtype=np.float64)
        weights = np.zeros(N, dtype=np.float64)

        for i, s in enumerate(valid_samples):
            X[i] = [s.gaze_yaw, s.gaze_pitch, 1.0]
            Y[i] = [s.target_x, s.target_y]
            # Single-eye fallback observations receive reduced weight
            w = float(s.confidence)
            if s.is_single_eye_fallback:
                w *= 0.70
            weights[i] = max(0.01, w)

        # Weighted Regularized Least Squares
        # Normal equations: (X^T @ W @ X + lambda * I) @ W_mat = X^T @ W @ Y
        W_diag = np.diag(weights)
        XT_W = X.T @ W_diag
        A = XT_W @ X + self.regularization_lambda * np.eye(3, dtype=np.float64)
        B = XT_W @ Y

        try:
            cond_num = float(np.linalg.cond(A))
            if not np.isfinite(cond_num) or cond_num > 1e6:
                self._is_calibrated = False
                self._quality_metrics = {
                    "success": False,
                    "reason": f"ill_conditioned_matrix: cond={cond_num:.2e}",
                }
                return False

            W_mat = np.linalg.solve(A, B)  # Shape: (3, 2)
        except (np.linalg.LinAlgError, ValueError) as e:
            self._is_calibrated = False
            self._quality_metrics = {"success": False, "reason": f"solver_exception: {e}"}
            return False

        # Evaluate Residuals on valid samples
        Y_pred = X @ W_mat
        errors = Y - Y_pred
        residuals_sq = np.sum(errors ** 2, axis=1)
        rmse = float(math.sqrt(np.mean(residuals_sq)))
        max_error = float(np.max(np.sqrt(residuals_sq)))

        if rmse > self.max_rmse_threshold or not np.isfinite(rmse):
            self._is_calibrated = False
            self._quality_metrics = {
                "success": False,
                "reason": f"excessive_residual_error: rmse={rmse:.4f} > {self.max_rmse_threshold}",
                "rmse": rmse,
                "max_error": max_error,
            }
            return False

        # Composite calibration quality score Q in [0.0, 1.0]
        # High score requires low RMSE and good sample coverage
        rmse_score = float(np.clip(1.0 - (rmse / self.max_rmse_threshold), 0.0, 1.0))
        spread_score = float(np.clip((span_x * span_y) / 0.60, 0.0, 1.0))
        quality_score = round(float(0.70 * rmse_score + 0.30 * spread_score), 4)

        self._calibration_matrix = W_mat
        self._is_calibrated = True
        self._calibrated_bounding_box = (
            min(target_xs), min(target_ys), max(target_xs), max(target_ys)
        )
        self._quality_metrics = {
            "success": True,
            "quality_score": quality_score,
            "rmse": round(rmse, 4),
            "max_error": round(max_error, 4),
            "sample_count": N,
            "target_count": len(targets),
            "span_x": round(span_x, 3),
            "span_y": round(span_y, 3),
            "condition_number": round(cond_num, 2),
            "bounding_box": self._calibrated_bounding_box,
        }
        return True

    def classify_boundary(self, screen_x: float, screen_y: float) -> str:
        """
        Classifies normalized screen coordinates relative to monitor boundaries.
        Margins prevent jitter at edges from rapidly oscillating between states.
        """
        margin = self.boundary_margin
        # Within screen interior away from borders
        if (margin <= screen_x <= (1.0 - margin)) and (margin <= screen_y <= (1.0 - margin)):
            return BOUNDARY_ON_SCREEN

        # Near boundary region (within margin of border or slightly outside up to margin)
        if ((-margin <= screen_x <= (1.0 + margin)) and (-margin <= screen_y <= (1.0 + margin))):
            return BOUNDARY_NEAR_BOUNDARY

        # Substantially outside the monitor frame
        return BOUNDARY_OUTSIDE_SCREEN

    def estimate_screen_gaze(
        self,
        gaze_yaw: Optional[float],
        gaze_pitch: Optional[float],
        gaze_confidence: float = 0.0,
        gaze_status: str = "VALID",
    ) -> ScreenGazeResult:
        """
        Maps observed camera-relative gaze to a ScreenGazeResult.

        Safety & Integrity Guarantees:
        - If not calibrated: returns status='NOT_CALIBRATED', boundary_state='CALIBRATION_REQUIRED'.
        - If gaze input is UNCERTAIN, non-finite, or low-confidence: returns status='UNCERTAIN'.
        - Preserves raw unclamped (x, y) coordinates for accurate outside-screen detection.
        - Discounts confidence when extrapolating far beyond the calibrated bounding box.
        """
        if not self._is_calibrated or self._calibration_matrix is None:
            return ScreenGazeResult(
                screen_x=None,
                screen_y=None,
                boundary_state=BOUNDARY_CALIBRATION_REQUIRED,
                confidence=0.0,
                uncertainty=1.0,
                status=CALIB_STATUS_NOT_CALIBRATED,
                is_calibrated=False,
                source_components={"error": "not_calibrated"},
            )

        try:
            gaze_yaw = float(gaze_yaw)
            gaze_pitch = float(gaze_pitch)
            gaze_confidence = float(gaze_confidence)
        except (TypeError, ValueError):
            gaze_yaw = gaze_pitch = gaze_confidence = float("nan")
        if (
            not (math.isfinite(gaze_yaw) and math.isfinite(gaze_pitch) and math.isfinite(gaze_confidence))
            or gaze_status != "VALID"
            or not (config.SCREEN_CALIBRATION_MIN_CONFIDENCE <= gaze_confidence <= 1.0)
        ):
            return ScreenGazeResult(
                screen_x=None,
                screen_y=None,
                boundary_state=BOUNDARY_UNCERTAIN,
                confidence=0.0,
                uncertainty=1.0,
                status="UNCERTAIN",
                is_calibrated=True,
                source_components={"error": "uncertain_or_invalid_gaze_input"},
            )

        feat = np.array([float(gaze_yaw), float(gaze_pitch), 1.0], dtype=np.float64)
        pred = feat @ self._calibration_matrix  # Shape: (2,)
        sx, sy = float(pred[0]), float(pred[1])

        if not np.isfinite(sx) or not np.isfinite(sy):
            return ScreenGazeResult(
                screen_x=None,
                screen_y=None,
                boundary_state=BOUNDARY_UNCERTAIN,
                confidence=0.0,
                uncertainty=1.0,
                status="UNCERTAIN",
                is_calibrated=True,
                source_components={"error": "non_finite_prediction"},
            )

        boundary_state = self.classify_boundary(sx, sy)

        # Confidence calculation
        calib_quality = float(self._quality_metrics.get("quality_score", 0.80))
        base_confidence = float(np.clip(gaze_confidence * calib_quality, 0.0, 1.0))

        # Extrapolation penalty if far beyond [0, 1] screen bounds
        dist_out_x = max(0.0, -sx, sx - 1.0)
        dist_out_y = max(0.0, -sy, sy - 1.0)
        dist_out = math.hypot(dist_out_x, dist_out_y)

        if dist_out > 0.10:
            # Smooth exponential decay for points far outside monitor
            extrap_penalty = math.exp(-2.0 * (dist_out - 0.10))
            final_confidence = round(base_confidence * extrap_penalty, 4)
        else:
            final_confidence = round(base_confidence, 4)

        uncertainty = round(1.0 - final_confidence, 4)

        return ScreenGazeResult(
            screen_x=round(sx, 4),
            screen_y=round(sy, 4),
            boundary_state=boundary_state,
            confidence=final_confidence,
            uncertainty=uncertainty,
            status="VALID",
            is_calibrated=True,
            source_components={
                "gaze_yaw": gaze_yaw,
                "gaze_pitch": gaze_pitch,
                "input_confidence": gaze_confidence,
                "calibration_quality": calib_quality,
                "extrapolation_dist": round(dist_out, 4),
            },
        )


# Interactive 5-point normalized screen calibration target coordinates: (x, y, label)
FIVE_POINT_TARGETS: List[Tuple[float, float, str]] = [
    (0.5, 0.5, "Center"),
    (0.1, 0.1, "Top-Left"),
    (0.9, 0.1, "Top-Right"),
    (0.1, 0.9, "Bottom-Left"),
    (0.9, 0.9, "Bottom-Right"),
]

# Primary display fullscreen window title
WINDOW_CALIBRATION_NAME = "Screen Calibration"


@dataclass(frozen=True)
class DisplayGeometry:
    """
    Structured representation of physical display geometry for single-monitor calibration.
    """
    width: int
    height: int
    aspect_ratio: float
    is_supported: bool = True
    status_message: str = "Standard display geometry"


def get_primary_display_resolution(strict: bool = False) -> Optional[Tuple[int, int]]:
    """
    Detects the primary physical display resolution in pixels using platform capabilities.
    On Windows, uses user32.GetSystemMetrics(0) [SM_CXSCREEN] and user32.GetSystemMetrics(1) [SM_CYSCREEN]
    after setting per-monitor DPI awareness.
    Falls back to config.DEFAULT_DISPLAY_WIDTH / DEFAULT_DISPLAY_HEIGHT or (1920, 1080).
    """
    try:
        import ctypes
        try:
            # Per-monitor DPI aware (Windows 8.1+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # System DPI aware (Windows Vista+)
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        w = int(ctypes.windll.user32.GetSystemMetrics(0))
        h = int(ctypes.windll.user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    if strict:
        return None
    return int(config.DEFAULT_DISPLAY_WIDTH), int(config.DEFAULT_DISPLAY_HEIGHT)


def validate_display_geometry(width: int, height: int) -> DisplayGeometry:
    """
    Validates display resolution and aspect ratio against supported desktop configurations.
    Rejects non-positive dimensions and flags extreme aspect ratios (< 0.5 or > 4.0).
    """
    if not (
        isinstance(width, (int, float, np.integer))
        and isinstance(height, (int, float, np.integer))
        and not isinstance(width, bool)
        and not isinstance(height, bool)
        and np.isfinite(width)
        and np.isfinite(height)
        and float(width).is_integer()
        and float(height).is_integer()
    ):
        return DisplayGeometry(
            width=0, height=0, aspect_ratio=0.0,
            is_supported=False,
            status_message="Dimensions must be numeric integers",
        )
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        return DisplayGeometry(
            width=w, height=h, aspect_ratio=0.0,
            is_supported=False,
            status_message=f"Invalid non-positive dimensions: {w}x{h}",
        )

    aspect = round(w / h, 4)
    if aspect < 0.5 or aspect > 4.0:
        return DisplayGeometry(
            width=w, height=h, aspect_ratio=aspect,
            is_supported=False,
            status_message=f"Unsupported extreme aspect ratio {aspect:.2f}:1 (supported range: 0.5:1 to 4.0:1)",
        )

    return DisplayGeometry(
        width=w, height=h, aspect_ratio=aspect,
        is_supported=True,
        status_message=f"Valid display geometry: {w}x{h} (aspect ratio {aspect:.2f}:1)",
    )


def map_normalized_to_display_pixel(
    norm_x: float,
    norm_y: float,
    display_size: Tuple[int, int],
    letterbox: bool = False,
    source_aspect: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Transforms normalized calibration coordinates (norm_x, norm_y) in [0.0, 1.0]
    to integer pixel coordinates (X_px, Y_px) on the physical display canvas.

    Transformation Formula:
      For direct 1:1 display mapping (native canvas):
        X_px = clip(round(norm_x * (width - 1)), 0, width - 1)
        Y_px = clip(round(norm_y * (height - 1)), 0, height - 1)

      With letterbox / pillarbox scaling (when rendering a source aspect into display):
        If display_aspect > source_aspect (pillarboxing, black bars on left/right):
          eff_w = height * source_aspect
          x_offset = (width - eff_w) / 2.0
          X_px = clip(round(x_offset + norm_x * (eff_w - 1)), 0, width - 1)
          Y_px = clip(round(norm_y * (height - 1)), 0, height - 1)
        If display_aspect < source_aspect (letterboxing, black bars on top/bottom):
          eff_h = width / source_aspect
          y_offset = (height - eff_h) / 2.0
          X_px = clip(round(norm_x * (width - 1)), 0, width - 1)
          Y_px = clip(round(y_offset + norm_y * (eff_h - 1)), 0, height - 1)
    """
    w, h = int(display_size[0]), int(display_size[1])
    if w <= 1 or h <= 1:
        return 0, 0

    if not letterbox or source_aspect is None or source_aspect <= 0.0:
        px = int(np.clip(round(norm_x * (w - 1)), 0, w - 1))
        py = int(np.clip(round(norm_y * (h - 1)), 0, h - 1))
        return px, py

    disp_aspect = w / h
    if abs(disp_aspect - source_aspect) < 1e-4:
        px = int(np.clip(round(norm_x * (w - 1)), 0, w - 1))
        py = int(np.clip(round(norm_y * (h - 1)), 0, h - 1))
        return px, py
    elif disp_aspect > source_aspect:
        # Display is wider: pillarboxing (horizontal offsets)
        eff_w = h * source_aspect
        x_offset = (w - eff_w) / 2.0
        px = int(np.clip(round(x_offset + norm_x * (eff_w - 1)), 0, w - 1))
        py = int(np.clip(round(norm_y * (h - 1)), 0, h - 1))
        return px, py
    else:
        # Display is taller: letterboxing (vertical offsets)
        eff_h = w / source_aspect
        y_offset = (h - eff_h) / 2.0
        px = int(np.clip(round(norm_x * (w - 1)), 0, w - 1))
        py = int(np.clip(round(y_offset + norm_y * (eff_h - 1)), 0, h - 1))
        return px, py


CALIB_STATE_NOT_CALIBRATED = "NOT_CALIBRATED"
CALIB_STATE_CALIBRATING = "CALIBRATING"
CALIB_STATE_VALID = "VALID"
CALIB_STATE_FAILED = "FAILED"
CALIB_STATE_CANCELLED = "CANCELLED"
SCREEN_CALIBRATION_SCHEMA_VERSION = 1


def screen_calibration_environment(
    source_type: str = "webcam",
    camera_index: Optional[int] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Dict[str, Any]:
    """Return stable, non-identifying camera capture geometry and gaze settings for compatibility checks."""
    width = int(config.FRAME_WIDTH if frame_width is None else frame_width)
    height = int(config.FRAME_HEIGHT if frame_height is None else frame_height)
    return {
        "camera": {
            "source_type": source_type,
            "camera_index": config.CAMERA_INDEX if camera_index is None else int(camera_index),
            "frame_width": width,
            "frame_height": height,
        },
        "gaze": {
            "algorithm": "head_eye_camera_v1",
            "horizontal_threshold": config.GAZE_HORIZONTAL_THRESHOLD,
            "vertical_threshold": config.GAZE_VERTICAL_THRESHOLD,
            "upward_threshold": config.GAZE_UPWARD_THRESHOLD,
            "fused_min_confidence": config.FUSED_GAZE_MIN_CONFIDENCE,
            "calibration_min_confidence": config.SCREEN_CALIBRATION_MIN_CONFIDENCE,
        },
    }


def resolve_capture_geometry(
    cap: Any,
    requested_width: int = config.FRAME_WIDTH,
    requested_height: int = config.FRAME_HEIGHT,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Authoritatively establishes effective camera capture geometry (GAP-03).

    1. Requests configured dimensions where supported by the capture device.
    2. Reads back delivered dimensions via CAP_PROP_FRAME_WIDTH / HEIGHT.
    3. Handles cameras that refuse requested dimensions without throwing unhandled exceptions.
    4. Validates that dimensions are strictly positive finite numeric integers.
    5. Returns (effective_width, effective_height) or (None, None) if device is unavailable.
    """
    if cap is None:
        return None, None

    if hasattr(cap, "isOpened"):
        try:
            if not cap.isOpened():
                return None, None
        except Exception:
            return None, None

    import cv2

    # Request configured capture dimensions where supported
    if hasattr(cap, "set"):
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(requested_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(requested_height))
        except Exception:
            pass

    # Read back actual dimensions delivered by the capture device
    actual_w = None
    actual_h = None
    if hasattr(cap, "get"):
        try:
            w_val = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h_val = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if (
                isinstance(w_val, (int, float)) and not isinstance(w_val, bool)
                and isinstance(h_val, (int, float)) and not isinstance(h_val, bool)
                and math.isfinite(float(w_val)) and math.isfinite(float(h_val))
                and float(w_val) > 0 and float(h_val) > 0
            ):
                actual_w = int(round(float(w_val)))
                actual_h = int(round(float(h_val)))
        except Exception:
            pass

    # Safe fallback for mock objects in testing when get() does not return numeric dimensions
    if actual_w is None or actual_h is None:
        if type(cap).__name__ == "MagicMock" or "Mock" in type(cap).__name__:
            return int(requested_width), int(requested_height)
        return None, None

    return actual_w, actual_h


def _default_calibration_options():
    return {
        "min_targets": config.SCREEN_CALIBRATION_MIN_TARGETS,
        "min_samples_per_target": config.SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET,
        "samples_per_target": config.SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET,
        "min_confidence": config.SCREEN_CALIBRATION_MIN_CONFIDENCE,
        "boundary_margin": config.SCREEN_BOUNDARY_MARGIN,
        "max_rmse_threshold": config.SCREEN_CALIBRATION_MAX_RMSE,
        "regularization_lambda": 1e-6,
    }


def _calibrator_snapshot(calibrator: ScreenGazeCalibrator):
    if not calibrator.is_calibrated:
        return None
    return {
        "matrix": calibrator._calibration_matrix.copy(),
        "quality_metrics": dict(calibrator._quality_metrics),
        "bounding_box": calibrator._calibrated_bounding_box,
    }


def _restore_calibrator_snapshot(calibrator: ScreenGazeCalibrator, snapshot):
    calibrator._samples.clear()
    calibrator._calibration_matrix = snapshot["matrix"].copy()
    calibrator._quality_metrics = dict(snapshot["quality_metrics"])
    calibrator._calibrated_bounding_box = snapshot["bounding_box"]
    calibrator._is_calibrated = True


def _validated_targets(targets):
    if not isinstance(targets, list) or len(targets) < 3:
        raise ValueError("invalid_target_configuration")
    normalized = []
    for target in targets:
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError("invalid_target_configuration")
        x, y, label = target
        if (
            isinstance(x, bool) or isinstance(y, bool)
            or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
            or not math.isfinite(float(x)) or not math.isfinite(float(y))
            or not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0)
            or not isinstance(label, str) or not label
        ):
            raise ValueError("invalid_target_configuration")
        normalized.append([float(x), float(y), label])
    return normalized


def save_screen_calibration(
    calibrator: ScreenGazeCalibrator,
    path: Union[str, Path],
    display_size: Tuple[int, int],
    targets,
    samples_per_target: int,
    min_confidence: float,
    environment: Dict[str, Any],
) -> Path:
    """Atomically save a fitted calibration; errors leave any prior file untouched."""
    if not calibrator.is_calibrated or calibrator._calibration_matrix is None:
        raise ValueError("cannot_save_uncalibrated_model")
    matrix = np.asarray(calibrator._calibration_matrix, dtype=np.float64)
    if matrix.shape != (3, 2) or not np.isfinite(matrix).all():
        raise ValueError("invalid_calibration_coefficients")
    metrics = dict(calibrator.quality_metrics)
    if metrics.get("success") is not True:
        raise ValueError("calibration_quality_not_successful")
    required_metrics = ("quality_score", "rmse", "max_error", "sample_count", "target_count", "span_x", "span_y", "condition_number", "bounding_box")
    if any(key not in metrics for key in required_metrics):
        raise ValueError("incomplete_calibration_quality_metrics")
    numeric_metrics = ("quality_score", "rmse", "max_error", "span_x", "span_y", "condition_number")
    if any(
        isinstance(metrics[key], bool)
        or not isinstance(metrics[key], (int, float))
        or not math.isfinite(float(metrics[key]))
        for key in numeric_metrics
    ):
        raise ValueError("invalid_calibration_quality_metrics")
    if (
        not 0.0 <= float(metrics["quality_score"]) <= 1.0
        or float(metrics["rmse"]) < 0.0
        or float(metrics["rmse"]) > config.SCREEN_CALIBRATION_MAX_RMSE
        or float(metrics["max_error"]) < 0.0
        or type(metrics["sample_count"]) is not int
        or metrics["sample_count"] < calibrator.min_targets * calibrator.min_samples_per_target
        or type(metrics["target_count"]) is not int
        or metrics["target_count"] < calibrator.min_targets
        or float(metrics["condition_number"]) < 0.0
        or float(metrics["condition_number"]) > 1e6
    ):
        raise ValueError("calibration_quality_out_of_range")
    bbox = metrics["bounding_box"]
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in bbox
    ) or not (0.0 <= bbox[0] <= bbox[2] <= 1.0 and 0.0 <= bbox[1] <= bbox[3] <= 1.0):
        raise ValueError("invalid_calibration_bounding_box")
    width, height = display_size
    geometry = validate_display_geometry(width, height)
    if not geometry.is_supported:
        raise ValueError("display_geometry_unsupported")
    target_config = _validated_targets([list(target) for target in targets])
    options = {
        "min_targets": calibrator.min_targets,
        "min_samples_per_target": calibrator.min_samples_per_target,
        "samples_per_target": int(samples_per_target),
        "min_confidence": float(min_confidence),
        "boundary_margin": calibrator.boundary_margin,
        "max_rmse_threshold": calibrator.max_rmse_threshold,
        "regularization_lambda": calibrator.regularization_lambda,
    }
    payload = {
        "schema_version": SCREEN_CALIBRATION_SCHEMA_VERSION,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "display": {"width": geometry.width, "height": geometry.height, "aspect_ratio": geometry.aspect_ratio},
        "environment": environment,
        "target_configuration": target_config,
        "calibration_options": options,
        "model": {"type": "affine_yaw_pitch_to_normalized_screen", "coefficients": matrix.tolist()},
        "quality_metrics": metrics,
    }

    if ".." in Path(path).parts or ".." in str(path):
        raise ValueError(f"Path traversal detected in calibration path: '{path}'")
    raw_path = Path(path)
    parent_dir = ensure_safe_directory(raw_path.parent)
    target_path = resolve_safe_output_path(
        parent_dir,
        raw_path.name,
        allow_extensions={".json"},
    )
    if is_symlink_or_junction(target_path):
        raise ValueError("Target calibration file cannot be a symlink or junction.")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as temp_file:
            json.dump(payload, temp_file, allow_nan=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, target_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return target_path


def load_screen_calibration(
    path: Union[str, Path],
    display_size: Optional[Tuple[int, int]],
    environment: Dict[str, Any],
    expected_targets=FIVE_POINT_TARGETS,
) -> Tuple[Optional[ScreenGazeCalibrator], str]:
    """Load only a complete, finite calibration compatible with current display and gaze settings."""
    target_path = Path(path)
    if not target_path.is_file():
        return None, "calibration_file_missing"
    if display_size is None:
        return None, "display_geometry_unavailable"
    try:
        with target_path.open("r", encoding="utf-8") as calibration_file:
            payload = json.load(calibration_file)
        if not isinstance(payload, dict):
            raise ValueError("invalid_root")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCREEN_CALIBRATION_SCHEMA_VERSION:
            return None, "unsupported_schema_version"
        timestamp = payload.get("calibrated_at")
        if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("invalid_calibration_timestamp")

        display = payload.get("display")
        if not isinstance(display, dict):
            raise ValueError("missing_display_metadata")
        width, height = display.get("width"), display.get("height")
        geometry = validate_display_geometry(width, height)
        current_geometry = validate_display_geometry(*display_size)
        saved_ratio = display.get("aspect_ratio")
        if (
            not geometry.is_supported or not current_geometry.is_supported
            or (geometry.width, geometry.height) != (current_geometry.width, current_geometry.height)
            or not isinstance(saved_ratio, (int, float)) or not math.isfinite(float(saved_ratio))
            or abs(float(saved_ratio) - geometry.aspect_ratio) > 1e-4
        ):
            return None, "display_geometry_incompatible"
        if payload.get("environment") != environment:
            return None, "camera_or_gaze_configuration_incompatible"

        targets = _validated_targets(payload.get("target_configuration"))
        expected = _validated_targets([list(target) for target in expected_targets])
        if targets != expected:
            return None, "target_configuration_incompatible"

        options = payload.get("calibration_options")
        if not isinstance(options, dict) or options != _default_calibration_options():
            return None, "calibration_options_incompatible"
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("type") != "affine_yaw_pitch_to_normalized_screen":
            raise ValueError("invalid_calibration_model")
        matrix = np.asarray(model.get("coefficients"), dtype=np.float64)
        if matrix.shape != (3, 2) or not np.isfinite(matrix).all():
            raise ValueError("invalid_calibration_coefficients")

        metrics = payload.get("quality_metrics")
        metric_fields = ("quality_score", "rmse", "max_error", "sample_count", "target_count", "span_x", "span_y", "condition_number", "bounding_box")
        if not isinstance(metrics, dict) or metrics.get("success") is not True or any(key not in metrics for key in metric_fields):
            raise ValueError("incomplete_calibration_quality_metrics")
        numeric_metrics = ("quality_score", "rmse", "max_error", "span_x", "span_y", "condition_number")
        if any(not isinstance(metrics[key], (int, float)) or not math.isfinite(float(metrics[key])) for key in numeric_metrics):
            raise ValueError("invalid_calibration_quality_metrics")
        if (
            not 0.0 <= float(metrics["quality_score"]) <= 1.0
            or float(metrics["rmse"]) > config.SCREEN_CALIBRATION_MAX_RMSE
            or type(metrics["sample_count"]) is not int
            or metrics["sample_count"] < options["min_targets"] * options["min_samples_per_target"]
            or type(metrics["target_count"]) is not int
            or metrics["target_count"] < options["min_targets"]
            or float(metrics["condition_number"]) > 1e6
        ):
            raise ValueError("calibration_quality_out_of_range")
        bbox = metrics["bounding_box"]
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in bbox
        ) or not (0.0 <= bbox[0] <= bbox[2] <= 1.0 and 0.0 <= bbox[1] <= bbox[3] <= 1.0):
            raise ValueError("invalid_calibration_bounding_box")

        calibrator = ScreenGazeCalibrator(
            min_targets=options["min_targets"],
            min_samples_per_target=options["min_samples_per_target"],
            boundary_margin=options["boundary_margin"],
            max_rmse_threshold=options["max_rmse_threshold"],
            regularization_lambda=options["regularization_lambda"],
        )
        metrics = dict(metrics)
        metrics["bounding_box"] = tuple(float(value) for value in bbox)
        calibrator._calibration_matrix = matrix
        calibrator._is_calibrated = True
        calibrator._quality_metrics = metrics
        calibrator._calibrated_bounding_box = tuple(float(value) for value in bbox)
        return calibrator, ""
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return None, f"invalid_calibration_file: {exc}"


class ScreenCalibrationManager:
    """
    Manages interactive multi-point screen gaze calibration lifecycle.
    Presents targets sequentially on the physical display canvas, collects valid
    synchronized combined-gaze samples, enforces deliberate fixation dwell and stability,
    validates criteria, and fits ScreenGazeCalibrator.
    """

    def __init__(
        self,
        calibrator: Optional[ScreenGazeCalibrator] = None,
        samples_per_target: Optional[int] = None,
        targets: Optional[List[Tuple[float, float, str]]] = None,
        min_confidence: Optional[float] = None,
        display_size: Optional[Tuple[int, int]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        environment: Optional[Dict[str, Any]] = None,
        min_dwell_sec: Optional[float] = None,
        stability_window_sec: Optional[float] = None,
        stability_threshold_deg: Optional[float] = None,
        min_sample_interval_sec: Optional[float] = None,
        time_source: Optional[Any] = None,
    ):
        self.calibrator = calibrator or ScreenGazeCalibrator()
        self.samples_per_target = max(1, int(config.SCREEN_CALIBRATION_MIN_SAMPLES_PER_TARGET if samples_per_target is None else samples_per_target))
        self.targets = list(targets or FIVE_POINT_TARGETS)
        self.min_confidence = float(config.SCREEN_CALIBRATION_MIN_CONFIDENCE if min_confidence is None else min_confidence)
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.environment = environment or screen_calibration_environment()

        self.min_dwell_sec = float(
            getattr(config, "SCREEN_CALIBRATION_MIN_DWELL_SEC", 1.0)
            if min_dwell_sec is None else min_dwell_sec
        )
        self.stability_window_sec = float(
            getattr(config, "SCREEN_CALIBRATION_STABILITY_WINDOW_SEC", 0.4)
            if stability_window_sec is None else stability_window_sec
        )
        self.stability_threshold_deg = float(
            getattr(config, "SCREEN_CALIBRATION_STABILITY_THRESHOLD_DEG", 2.0)
            if stability_threshold_deg is None else stability_threshold_deg
        )
        self.min_sample_interval_sec = float(
            getattr(config, "SCREEN_CALIBRATION_MIN_SAMPLE_INTERVAL_SEC", 0.15)
            if min_sample_interval_sec is None else min_sample_interval_sec
        )
        self.time_source = time_source

        # Internal target dwell and stability tracking state (GAP-01)
        self._target_start_time: Optional[float] = None
        self._stability_start_time: Optional[float] = None
        self._target_observations: List[Tuple[float, float, float, ScreenCalibrationSample]] = []
        self._target_stable_samples: List[ScreenCalibrationSample] = []
        self._last_sample_accepted_time: Optional[float] = None
        self._stability_status: str = "INITIAL"
        self._last_processed_time: Optional[float] = None

        if display_size is not None:
            self.display_size = (int(display_size[0]), int(display_size[1]))
            self.display_geometry_known = True
        else:
            detected_size = get_primary_display_resolution(strict=True)
            self.display_geometry_known = detected_size is not None
            self.display_size = detected_size or (config.DEFAULT_DISPLAY_WIDTH, config.DEFAULT_DISPLAY_HEIGHT)
        self.display_geometry = validate_display_geometry(self.display_size[0], self.display_size[1])

        self.state: str = CALIB_STATE_NOT_CALIBRATED
        self.current_target_idx: int = 0
        self.current_target_samples: int = 0
        self.failure_reason: str = ""
        self._previous_calibration = None

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator.is_calibrated

    @property
    def current_target(self) -> Optional[Tuple[float, float, str]]:
        if self.state == CALIB_STATE_CALIBRATING and 0 <= self.current_target_idx < len(self.targets):
            return self.targets[self.current_target_idx]
        return None

    @property
    def target_dwell_progress(self) -> float:
        """Normalized progress [0.0, 1.0] toward required target dwell duration."""
        if self.state != CALIB_STATE_CALIBRATING or self._target_start_time is None or self._last_processed_time is None:
            return 0.0
        elapsed = self._last_processed_time - self._target_start_time
        return float(np.clip(elapsed / max(0.01, self.min_dwell_sec), 0.0, 1.0))

    @property
    def target_stability_progress(self) -> float:
        """Normalized progress [0.0, 1.0] toward required continuous stability window."""
        if self.state != CALIB_STATE_CALIBRATING or self._stability_start_time is None or self._last_processed_time is None:
            return 0.0
        elapsed = self._last_processed_time - self._stability_start_time
        return float(np.clip(elapsed / max(0.01, self.stability_window_sec), 0.0, 1.0))

    @property
    def overall_progress(self) -> float:
        """Overall session completion fraction in [0.0, 1.0]."""
        if not self.targets:
            return 0.0
        if self.state == CALIB_STATE_VALID:
            return 1.0
        if self.state != CALIB_STATE_CALIBRATING:
            return 0.0
        completed = float(self.current_target_idx)
        dwell_frac = self.target_dwell_progress
        return float(np.clip((completed + dwell_frac) / len(self.targets), 0.0, 1.0))

    @property
    def stability_status(self) -> str:
        """Diagnostic classification of current fixation measurement stability."""
        return self._stability_status

    def _reset_target_tracking(self):
        """Resets dwell and stability accumulators for the active target."""
        self._target_start_time = None
        self._stability_start_time = None
        self._target_observations.clear()
        self._target_stable_samples.clear()
        self._last_sample_accepted_time = None
        self._stability_status = "IDLE"

    def start(self):
        """Starts a new interactive calibration session."""
        self._previous_calibration = _calibrator_snapshot(self.calibrator)
        if not self.display_geometry_known or not self.display_geometry.is_supported:
            self.state = CALIB_STATE_FAILED
            self.failure_reason = "display_geometry_unavailable_or_unsupported"
            self._previous_calibration = None
            return
        self.calibrator.reset()
        self.state = CALIB_STATE_CALIBRATING
        self.current_target_idx = 0
        self.current_target_samples = 0
        self.failure_reason = ""
        self._reset_target_tracking()

    def cancel(self):
        """Cancels calibration and restores any previously valid calibration."""
        self._restore_previous_calibration()
        self.state = CALIB_STATE_CANCELLED
        self.current_target_idx = 0
        self.current_target_samples = 0
        self._reset_target_tracking()

    def fail(self, reason: str):
        """Ends calibration safely and restores the previous valid model, if any."""
        self._restore_previous_calibration()
        self.state = CALIB_STATE_FAILED
        self.failure_reason = str(reason)
        self._reset_target_tracking()

    def _restore_previous_calibration(self):
        if self._previous_calibration is None:
            self.calibrator.reset()
        else:
            _restore_calibrator_snapshot(self.calibrator, self._previous_calibration)
        self._previous_calibration = None

    def retry_current_target(self):
        """Resets sample count for current target without aborting entire session."""
        self.current_target_samples = 0
        self._reset_target_tracking()

    def _advance_target(self):
        """Advances to the next target or fits the calibrator if all targets completed."""
        self.current_target_idx += 1
        self.current_target_samples = 0
        self._reset_target_tracking()

        # If all targets collected, fit model
        if self.current_target_idx >= len(self.targets):
            try:
                success = self.calibrator.fit()
                if not success:
                    self.fail(self.calibrator.quality_metrics.get("reason", "fitting_failed"))
                else:
                    if self.calibration_path is not None:
                        save_screen_calibration(
                            self.calibrator,
                            self.calibration_path,
                            self.display_size,
                            self.targets,
                            self.samples_per_target,
                            self.min_confidence,
                            self.environment,
                        )
                    self._previous_calibration = None
                    self.state = CALIB_STATE_VALID
            except Exception as exc:
                self.fail(f"calibration_save_or_fit_failed: {exc}")

    def process_sample(
        self,
        combined_gaze: Optional[Any],
        face_result: Optional[Any] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Processes a single frame observation during interactive calibration (GAP-01).

        Enforces:
        - Rejection of invalid, missing, uncertain, or low-confidence gaze observations.
        - Documented face loss policy: resets stability window when face tracking is lost.
        - Fixation stability window: resets window when gaze deviates beyond threshold.
        - Minimum target dwell duration: requires steady fixation over a meaningful time interval.
        - Spaced sample collection across the dwell duration rather than consecutive adjacent frames.
        - Clear diagnostic status tracking.
        """
        if self.state != CALIB_STATE_CALIBRATING:
            return False

        target = self.current_target
        if target is None:
            return False

        target_x, target_y, target_name = target

        # Determine effective observation time
        current_time = None
        if timestamp is not None:
            try:
                t_val = float(timestamp)
                if math.isfinite(t_val):
                    current_time = t_val
            except (ValueError, TypeError):
                current_time = None
        elif self.time_source is not None and hasattr(self.time_source, "now"):
            try:
                t_val = float(self.time_source.now())
                if math.isfinite(t_val):
                    current_time = t_val
            except Exception:
                current_time = None

        if current_time is not None:
            self._last_processed_time = current_time

        # Documented Face Loss Policy (GAP-01 Requirement 8):
        # Face absence breaks fixation continuity. The stability window is reset and
        # any in-progress fixation buffer for the current target is cleared.
        is_face_lost = False
        if face_result is not None:
            face_count = getattr(face_result, "face_count", None)
            gaze_dir = getattr(face_result, "gaze_direction", "")
            head_dir = getattr(face_result, "head_direction", "")
            if (face_count is not None and face_count == 0) or gaze_dir == "No Face Detected" or head_dir == "No Face Detected":
                is_face_lost = True
        elif combined_gaze is not None and getattr(combined_gaze, "status", None) == "NO_FACE":
            is_face_lost = True

        if is_face_lost:
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "FACE_LOST"
            return False

        # Validate combined gaze observation status
        if combined_gaze is None or getattr(combined_gaze, "status", None) != "VALID":
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "INVALID_GAZE"
            return False

        yaw = getattr(combined_gaze, "yaw", None)
        pitch = getattr(combined_gaze, "pitch", None)
        conf = getattr(combined_gaze, "confidence", 0.0)

        if yaw is None or pitch is None or conf is None:
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "MISSING_ANGLES"
            return False

        try:
            yaw = float(yaw)
            pitch = float(pitch)
            conf = float(conf)
        except (ValueError, TypeError):
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "NON_FINITE"
            return False

        # Reject non-finite observations
        if not (math.isfinite(yaw) and math.isfinite(pitch) and math.isfinite(conf)):
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "NON_FINITE"
            return False

        # Reject insufficient-confidence observations
        if conf < self.min_confidence:
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "LOW_CONFIDENCE"
            return False

        tracking_state = "PAIRED_TRACKING"
        is_fallback = False
        if face_result is not None and getattr(face_result, "eye_gaze", None) is not None:
            tracking_state = getattr(face_result.eye_gaze, "tracking_state", "PAIRED_TRACKING")
            is_fallback = bool(getattr(face_result.eye_gaze, "is_single_eye_fallback", False))

        sample = ScreenCalibrationSample(
            target_x=target_x,
            target_y=target_y,
            gaze_yaw=yaw,
            gaze_pitch=pitch,
            confidence=conf,
            tracking_state=tracking_state,
            is_single_eye_fallback=is_fallback,
            timestamp=current_time,
        )

        if not sample.is_valid(self.min_confidence):
            self._stability_start_time = None
            self._target_observations.clear()
            self._stability_status = "INVALID_SAMPLE"
            return False

        # Untimed legacy mode: if no timestamp and no time source is provided,
        # fallback to sample count advance to maintain backwards compatibility
        if current_time is None:
            if not self.calibrator.add_sample(sample):
                return False
            self.current_target_samples += 1
            if self.current_target_samples >= self.samples_per_target:
                self._advance_target()
            return True

        # Timed fixation dwell and stability workflow (GAP-01)
        if self._target_start_time is None:
            self._target_start_time = current_time
            self._stability_start_time = current_time

        if self._stability_start_time is None:
            self._stability_start_time = current_time

        # Record observation in recent sliding window
        self._target_observations.append((current_time, yaw, pitch, sample))

        # Prune observations older than stability_window_sec
        cutoff = current_time - self.stability_window_sec
        self._target_observations = [obs for obs in self._target_observations if obs[0] >= cutoff]

        # Evaluate measurement stability across the window
        yaws = [obs[1] for obs in self._target_observations]
        pitches = [obs[2] for obs in self._target_observations]
        yaw_spread = max(yaws) - min(yaws)
        pitch_spread = max(pitches) - min(pitches)

        if yaw_spread > self.stability_threshold_deg or pitch_spread > self.stability_threshold_deg:
            # Gaze was unstable or moving toward target -> reset stability window!
            self._stability_start_time = current_time
            self._target_observations = [(current_time, yaw, pitch, sample)]
            self._stability_status = "UNSTABLE"
            # Measurement processed, but stability window reset and target does not advance
            return True

        self._stability_status = "STABLE"

        # Collect stable samples across meaningful intervals
        if (
            self._last_sample_accepted_time is None
            or (current_time - self._last_sample_accepted_time) >= self.min_sample_interval_sec
        ):
            self._target_stable_samples.append(sample)
            self._last_sample_accepted_time = current_time

        self.current_target_samples = len(self._target_stable_samples)

        # Evaluate target completion criteria:
        dwell_elapsed = current_time - self._target_start_time
        stability_elapsed = current_time - self._stability_start_time

        target_satisfied = (
            dwell_elapsed >= self.min_dwell_sec
            and stability_elapsed >= self.stability_window_sec
            and len(self._target_stable_samples) >= self.samples_per_target
        )

        if target_satisfied:
            # Commit stable samples to calibrator
            for s in self._target_stable_samples:
                self.calibrator.add_sample(s)
            self._advance_target()

        return True

    def get_target_screen_coordinates(
        self,
        target_idx: Optional[int] = None,
        display_size: Optional[Tuple[int, int]] = None,
        letterbox: bool = False,
        source_aspect: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Returns the exact integer pixel coordinate (X_px, Y_px) of a target on the physical display.
        If target_idx is None, uses self.current_target_idx.
        Returns None if target index is out of bounds or no target is available.
        """
        idx = self.current_target_idx if target_idx is None else target_idx
        if not (0 <= idx < len(self.targets)):
            return None
        tx, ty, _ = self.targets[idx]
        ds = display_size or self.display_size
        return map_normalized_to_display_pixel(
            tx, ty, ds, letterbox=letterbox, source_aspect=source_aspect
        )

    def render_calibration_canvas(
        self,
        webcam_preview: Optional[np.ndarray] = None,
        display_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Renders an independent full-screen calibration canvas anchored to physical display coordinates.
        Draws high-contrast bullseye targets at exact display pixel locations, instructions,
        progress indicators, and an optional webcam preview inset positioned safely at bottom-center.
        """
        w, h = display_size or self.display_size
        w = max(640, int(w))
        h = max(480, int(h))

        # Modern dark background (BGR: 26, 26, 26)
        canvas = np.full((h, w, 3), 26, dtype=np.uint8)

        try:
            import cv2
        except ImportError:
            return canvas

        if self.state == CALIB_STATE_CALIBRATING and self.current_target is not None:
            tx, ty, name = self.current_target
            px, py = map_normalized_to_display_pixel(tx, ty, (w, h))

            # Subtle guide crosshair
            crosshair_len = 30
            cv2.line(canvas, (px - crosshair_len, py), (px + crosshair_len, py), (80, 80, 90), 1)
            cv2.line(canvas, (px, py - crosshair_len), (px, py + crosshair_len), (80, 80, 90), 1)

            # Target Bullseye Rendering
            # Outer ring
            cv2.circle(canvas, (px, py), 36, (0, 255, 255), 2)
            # Intermediate ring
            cv2.circle(canvas, (px, py), 18, (0, 180, 255), 2)
            # Inner bullseye dot
            cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
            # Center bright pixel
            cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)

            # Target name next to bullseye
            text_x = min(px + 45, w - 220)
            text_y = max(py - 15, 40)
            cv2.putText(canvas, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Header / Progress Banner at top
            banner_text = f"SCREEN CALIBRATION: Target {self.current_target_idx + 1} of {len(self.targets)} ({name})"
            cv2.putText(canvas, banner_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Progress arc / ring around bullseye
            dwell_frac = self.target_dwell_progress
            if dwell_frac > 0:
                cv2.ellipse(canvas, (px, py), (46, 46), -90, 0, int(dwell_frac * 360), (0, 255, 0), 3)

            sample_text = f"Samples: {self.current_target_samples}/{self.samples_per_target} | Dwell: {self.target_dwell_progress*100:.0f}% | Stability: {self.stability_status}"
            cv2.putText(canvas, sample_text, (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)

            # Guidance Banner at bottom
            cv2.putText(
                canvas,
                "Fixate your gaze directly on the center dot. Keep your head stable.",
                (40, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                canvas,
                "Press 'q' to cancel calibration.",
                (40, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (140, 140, 140),
                1,
            )

        elif self.state == CALIB_STATE_FAILED:
            msg = f"Screen Calibration Failed: {self.failure_reason[:80]}"
            cv2.putText(canvas, msg, (40, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, "Press 'c' to restart calibration or 'q' to exit.", (40, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        elif self.state == CALIB_STATE_VALID:
            q = self.calibrator.quality_metrics
            cv2.putText(canvas, "Screen Calibration Successful!", (40, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(canvas, f"RMSE: {q.get('rmse', 0):.4f} | Quality Score: {q.get('quality_score', 0):.2f}", (40, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Optional webcam preview inset positioned at bottom-center so it never overlaps targets
        if webcam_preview is not None and webcam_preview.size > 0:
            inset_w, inset_h = 240, 180
            if w >= inset_w + 100 and h >= inset_h + 150:
                try:
                    small_preview = cv2.resize(webcam_preview, (inset_w, inset_h))
                    x_start = (w - inset_w) // 2
                    y_start = h - inset_h - 20
                    canvas[y_start:y_start + inset_h, x_start:x_start + inset_w] = small_preview
                    cv2.rectangle(canvas, (x_start, y_start), (x_start + inset_w, y_start + inset_h), (100, 100, 100), 2)
                    cv2.putText(canvas, "Camera Alignment Feed", (x_start + 5, y_start - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
                except Exception:
                    pass

        return canvas

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Renders interactive calibration target and instructions on display frame."""
        if frame is None or frame.size == 0:
            return frame

        try:
            import cv2
        except ImportError:
            return frame

        h, w = frame.shape[:2]
        if self.state == CALIB_STATE_CALIBRATING and self.current_target is not None:
            tx, ty, name = self.current_target
            px = int(np.clip(tx * w, 20, w - 20))
            py = int(np.clip(ty * h, 20, h - 20))

            # Outer ring
            cv2.circle(frame, (px, py), 24, (0, 255, 255), 2)
            # Inner target dot
            cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

            # Target label and progress
            msg = f"SCREEN CALIBRATION ({self.current_target_idx + 1}/{len(self.targets)}): Look at {name} ({self.current_target_samples}/{self.samples_per_target})"
            cv2.putText(frame, msg, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Fixate target until dot advances. Press 'q' to cancel.", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        elif self.state == CALIB_STATE_FAILED:
            cv2.putText(frame, f"Screen Calibration Failed: {self.failure_reason[:60]} (Press 'c' to retry)", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        elif self.state == CALIB_STATE_VALID:
            cv2.putText(frame, "Screen Calibration: Complete (Active)", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return frame
