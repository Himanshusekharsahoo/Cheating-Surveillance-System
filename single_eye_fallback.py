"""
Single-Eye Fallback and Confidence-Aware Gaze Tracking Module.
Phase 10.13.3 Implementation.

Provides independent per-eye confidence scoring, combined reliability calculation,
and a deterministic state machine for managing transitions between paired tracking,
single-eye fallback, and recovery states.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from config import (
    GAZE_HORIZONTAL_THRESHOLD,
    GAZE_VERTICAL_THRESHOLD,
    GAZE_UPWARD_THRESHOLD,
)

# Canonical gaze direction constants
GAZE_CENTER = "Looking Center"
GAZE_LEFT = "Looking Left"
GAZE_RIGHT = "Looking Right"
GAZE_UP = "Looking Up"
GAZE_DOWN = "Looking Down"
GAZE_UNCERTAIN = "Uncertain Gaze"

# Explicit Tracking States
TRACKING_STATE_PAIRED = "PAIRED_TRACKING"
TRACKING_STATE_LEFT_ONLY = "LEFT_EYE_ONLY"
TRACKING_STATE_RIGHT_ONLY = "RIGHT_EYE_ONLY"
TRACKING_STATE_UNCERTAIN = "UNCERTAIN"
TRACKING_STATE_RECOVERING = "RECOVERING"

# Confidence and Recovery Limits
MIN_EYE_CONFIDENCE = 0.50               # Minimum per-eye confidence for paired tracking
SINGLE_EYE_CONFIDENCE_THRESHOLD = 0.70  # Higher bar required for single-eye fallback
MAX_SINGLE_EYE_COMBINED_CONF = 0.75     # Redundancy discount cap in single-eye mode
RECOVERY_REQUIRED_FRAMES = 3            # Consecutive paired frames before confirming recovery


@dataclass
class GazeTrackingResult:
    """
    Structured outcome of confidence-aware gaze evaluation.
    """
    gaze_direction: str
    tracking_state: str
    left_confidence: float
    right_confidence: float
    combined_confidence: float
    is_single_eye_fallback: bool
    left_gaze_dir: Optional[str] = None
    right_gaze_dir: Optional[str] = None
    rejection_reason: str = ""


def score_eye_confidence(
    pupil: Optional[Tuple[int, int]],
    eye_w: int,
    eye_h: int,
    eye_crop: Optional[np.ndarray] = None,
    eye_polygon: Optional[np.ndarray] = None,
    diag_info: Optional[Dict[str, Any]] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    contour_area: Optional[float] = None,
) -> float:
    """
    Calculates an independent per-eye confidence score C_eye in range [0.0, 1.0].

    Documented Mathematical Scoring Model:
      C_eye = 0.35 * S_contrast + 0.30 * S_geometry + 0.20 * S_aperture + 0.15 * S_interior

    Sub-score Models:
      1. S_contrast: Contrast dynamic range within eye aperture.
         Linear ramp from delta=12 (0.0) to delta=50 (1.0).
      2. S_geometry: Aspect ratio and fill factor (0.6 * S_aspect + 0.4 * S_fill).
         Ideal circular/elliptical pupil has aspect near 1.0 and fill factor near 0.785.
      3. S_aperture: Vertical palpebral aperture opening.
         Penalizes drooping eyelids or heavy partial occlusion (height <= 3px -> 0.2, >= 8px -> 1.0).
      4. S_interior: Normalized horizontal pupil position within aperture.
         Central zone (0.20 <= norm_x <= 0.80) scores 1.0; extreme canthus positions decay to 0.5.

    Returns:
      float: Confidence score in range [0.0, 1.0]. Returns 0.0 if pupil is missing or invalid.
    """
    if pupil is None or eye_w <= 0 or eye_h <= 0:
        return 0.0

    try:
        px, py = pupil
        if np.isnan(px) or np.isnan(py) or np.isinf(px) or np.isinf(py):
            return 0.0
    except Exception:
        return 0.0

    if diag_info is not None and not diag_info.get("valid", True):
        return 0.0

    norm_x = px / float(eye_w)
    if not (0.0 <= norm_x <= 1.0):
        return 0.0

    norm_y = py / float(eye_h)
    if not (-0.50 <= norm_y <= 1.50):
        return 0.0

    # 1. Contrast Quality Score (0.35 weight)
    if eye_crop is not None and eye_crop.size > 0:
        try:
            gray = eye_crop if len(eye_crop.shape) == 2 else np.mean(eye_crop, axis=2).astype(np.uint8)
            if eye_polygon is not None and len(eye_polygon) >= 3:
                import cv2
                mask = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [np.int32(eye_polygon)], 255)
                vals = gray[mask > 0]
                contrast = (float(np.max(vals)) - float(np.min(vals))) if len(vals) > 0 else 0.0
            else:
                contrast = float(np.max(gray)) - float(np.min(gray))
            s_contrast = np.clip((contrast - 12.0) / 38.0, 0.0, 1.0)
        except Exception:
            s_contrast = 0.50
    else:
        s_contrast = 0.85

    # 2. Contour Geometry Score (0.30 weight)
    if bbox is not None:
        bx, by, bw, bh = bbox
        box_area = float(max(1, bw * bh))
        aspect = bw / float(max(1, bh))
        s_aspect = float(np.clip(1.0 - abs(aspect - 1.0) / 2.0, 0.0, 1.0))

        if contour_area is not None:
            try:
                c_area = float(contour_area)
                if np.isnan(c_area) or np.isinf(c_area) or c_area <= 0:
                    s_fill = 0.0
                else:
                    fill = c_area / box_area
                    # Ideal ellipse has fill = pi / 4 ~= 0.785
                    s_fill = float(np.clip(1.0 - 2.0 * abs(fill - 0.785), 0.0, 1.0))
            except (ValueError, TypeError):
                s_fill = 0.0
        else:
            # Unavailable contour measurement is not awarded maximum fill confidence
            s_fill = 0.50

        s_geom = 0.6 * s_aspect + 0.4 * s_fill
    else:
        s_geom = 0.50

    # 3. Aperture Opening Score (0.20 weight)
    ap_h = eye_h
    if eye_polygon is not None and len(eye_polygon) >= 3:
        ap_h = int(np.max(eye_polygon[:, 1]) - np.min(eye_polygon[:, 1]))
    s_aperture = np.clip((ap_h - 3.0) / 5.0, 0.2, 1.0)

    # 4. Interior Centroid Position Score (0.15 weight)
    s_interior = np.clip(1.0 - 2.5 * max(0.0, 0.20 - norm_x, norm_x - 0.80), 0.5, 1.0)

    total = 0.35 * s_contrast + 0.30 * s_geom + 0.20 * s_aperture + 0.15 * s_interior
    return round(float(np.clip(total, 0.0, 1.0)), 4)


def evaluate_single_eye_direction(
    pupil: Optional[Tuple[int, int]],
    eye_w: int,
    eye_h: int,
    baseline: Optional[Tuple[float, float]] = None,
    h_thresh: float = GAZE_HORIZONTAL_THRESHOLD,
    v_thresh_down: float = GAZE_VERTICAL_THRESHOLD,
    v_thresh_up: float = GAZE_UPWARD_THRESHOLD,
    conservative_multiplier: float = 1.15,
) -> str:
    """
    Evaluates gaze direction from a single verified eye using conservative thresholds.
    """
    if pupil is None or eye_w <= 0 or eye_h <= 0:
        return GAZE_UNCERTAIN

    norm_x = pupil[0] / float(eye_w)
    norm_y = pupil[1] / float(eye_h)
    base_x, base_y = baseline if baseline is not None else (0.50, 0.40)

    diff_x = base_x - norm_x
    diff_y = norm_y - base_y

    ht = h_thresh * conservative_multiplier
    vt_down = v_thresh_down * conservative_multiplier
    vt_up = v_thresh_up * conservative_multiplier

    if diff_x < -ht:
        return GAZE_LEFT
    elif diff_x > ht:
        return GAZE_RIGHT
    elif diff_y <= -vt_up:
        return GAZE_UP
    elif diff_y >= vt_down:
        return GAZE_DOWN
    else:
        return GAZE_CENTER


class SingleEyeGazeTracker:
    """
    Deterministic tracking state machine for managing confidence-aware paired gaze
    and single-eye fallback transitions.
    """

    def __init__(
        self,
        min_eye_conf: float = MIN_EYE_CONFIDENCE,
        single_eye_conf_thresh: float = SINGLE_EYE_CONFIDENCE_THRESHOLD,
        recovery_frames_required: int = RECOVERY_REQUIRED_FRAMES,
    ):
        self.min_eye_conf = min_eye_conf
        self.single_eye_conf_thresh = single_eye_conf_thresh
        self.recovery_frames_required = recovery_frames_required

        self.current_state = TRACKING_STATE_UNCERTAIN
        self.recovery_counter = 0
        self.switch_counter = 0
        self.last_valid_direction = GAZE_UNCERTAIN

    def reset(self):
        """Resets tracker state to UNCERTAIN."""
        self.current_state = TRACKING_STATE_UNCERTAIN
        self.recovery_counter = 0
        self.switch_counter = 0
        self.last_valid_direction = GAZE_UNCERTAIN

    def update(
        self,
        left_pupil: Optional[Tuple[int, int]],
        left_w: int,
        left_h: int,
        right_pupil: Optional[Tuple[int, int]],
        right_w: int,
        right_h: int,
        calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        left_conf: Optional[float] = None,
        right_conf: Optional[float] = None,
        h_thresh: Optional[float] = None,
        v_thresh: Optional[float] = None,
        v_thresh_up: Optional[float] = None,
    ) -> GazeTrackingResult:
        """
        Processes single-frame eye observations, evaluates per-eye usability,
        updates the state machine, and outputs a GazeTrackingResult.
        """
        # 1. Resolve baselines
        if calibrated_gaze is not None:
            base_l = calibrated_gaze[0] if (len(calibrated_gaze) > 0 and calibrated_gaze[0] is not None) else (0.50, 0.40)
            base_r = calibrated_gaze[1] if (len(calibrated_gaze) > 1 and calibrated_gaze[1] is not None) else (0.50, 0.40)
        else:
            base_l = (0.50, 0.40)
            base_r = (0.50, 0.40)

        # 2. Resolve thresholds
        ht = h_thresh if h_thresh is not None else GAZE_HORIZONTAL_THRESHOLD
        vt_down = v_thresh if v_thresh is not None else GAZE_VERTICAL_THRESHOLD
        vt_up = v_thresh_up if v_thresh_up is not None else GAZE_UPWARD_THRESHOLD

        # 3. Resolve per-eye confidences
        c_left = left_conf if left_conf is not None else (0.85 if left_pupil else 0.0)
        c_right = right_conf if right_conf is not None else (0.85 if right_pupil else 0.0)

        # 4. Usability assessment
        left_usable = (left_pupil is not None and left_w > 0 and left_h > 0 and c_left >= self.min_eye_conf)
        right_usable = (right_pupil is not None and right_w > 0 and right_h > 0 and c_right >= self.min_eye_conf)

        # Single-eye directional hypotheses
        dir_l = evaluate_single_eye_direction(
            left_pupil, left_w, left_h, base_l,
            h_thresh=ht, v_thresh_down=vt_down, v_thresh_up=vt_up
        ) if left_usable else None

        dir_r = evaluate_single_eye_direction(
            right_pupil, right_w, right_h, base_r,
            h_thresh=ht, v_thresh_down=vt_down, v_thresh_up=vt_up
        ) if right_usable else None

        # 5. State Machine & Direction Determination
        if left_usable and right_usable:
            # === PAIRED EYE MODE ===
            norm_lx = left_pupil[0] / float(left_w)
            norm_ly = left_pupil[1] / float(left_h)
            norm_rx = right_pupil[0] / float(right_w)
            norm_ry = right_pupil[1] / float(right_h)

            diff_lx = base_l[0] - norm_lx
            diff_rx = base_r[0] - norm_rx
            diff_ly = norm_ly - base_l[1]
            diff_ry = norm_ry - base_r[1]

            # Contradiction check: opposing horizontal or vertical deflections
            h_contradictory = (diff_lx < -0.10 and diff_rx > 0.10) or (diff_lx > 0.10 and diff_rx < -0.10)
            v_contradictory = (diff_ly <= -0.08 and diff_ry >= 0.10) or (diff_ly >= 0.10 and diff_ry <= -0.08)
            is_contradictory = h_contradictory or v_contradictory

            if is_contradictory:
                combined_conf = round(float(np.mean([c_left, c_right]) * 0.20), 4)
                gaze_dir = GAZE_UNCERTAIN
                rejection = "contradictory_eye_deflections"
            else:
                total_w = c_left + c_right
                if total_w > 0:
                    mean_diff_x = (diff_lx * c_left + diff_rx * c_right) / total_w
                    mean_diff_y = (diff_ly * c_left + diff_ry * c_right) / total_w
                else:
                    mean_diff_x = (diff_lx + diff_rx) / 2.0
                    mean_diff_y = (diff_ly + diff_ry) / 2.0

                if (diff_lx < -ht and diff_rx < -ht) or (mean_diff_x < -ht and diff_lx <= 0.05 and diff_rx <= 0.05):
                    paired_dir = GAZE_LEFT
                elif (diff_lx > ht and diff_rx > ht) or (mean_diff_x > ht and diff_lx >= -0.05 and diff_rx >= -0.05):
                    paired_dir = GAZE_RIGHT
                elif (diff_ly <= -vt_up and diff_ry <= -vt_up) or (mean_diff_y <= -vt_up and diff_ly <= 0.03 and diff_ry <= 0.03):
                    paired_dir = GAZE_UP
                elif (diff_ly >= vt_down and diff_ry >= vt_down) or (mean_diff_y >= vt_down and diff_ly >= -0.03 and diff_ry >= -0.03):
                    paired_dir = GAZE_DOWN
                else:
                    paired_dir = GAZE_CENTER

                gaze_dir = paired_dir
                conf_penalty = max(0.0, abs(c_left - c_right) - 0.25) * 0.20
                combined_conf = round(float(np.clip(min(1.0, np.mean([c_left, c_right]) * 1.05) - conf_penalty, 0.0, 1.0)), 4)
                rejection = ""

            # State transition with recovery confirmation
            if self.current_state == TRACKING_STATE_PAIRED:
                tracking_state = TRACKING_STATE_PAIRED
                self.recovery_counter = 0
            elif self.current_state == TRACKING_STATE_RECOVERING:
                self.recovery_counter += 1
                if self.recovery_counter >= self.recovery_frames_required:
                    tracking_state = TRACKING_STATE_PAIRED
                    self.current_state = TRACKING_STATE_PAIRED
                    self.recovery_counter = 0
                else:
                    tracking_state = TRACKING_STATE_RECOVERING
            else:
                # Transitioning from single-eye or uncertain to recovering
                tracking_state = TRACKING_STATE_RECOVERING
                self.current_state = TRACKING_STATE_RECOVERING
                self.recovery_counter = 1

            if tracking_state == TRACKING_STATE_RECOVERING:
                combined_conf = round(float(combined_conf * 0.85), 4)

            self.switch_counter = 0
            self.last_valid_direction = gaze_dir
            return GazeTrackingResult(
                gaze_direction=gaze_dir,
                tracking_state=tracking_state,
                left_confidence=c_left,
                right_confidence=c_right,
                combined_confidence=combined_conf,
                is_single_eye_fallback=False,
                left_gaze_dir=dir_l,
                right_gaze_dir=dir_r,
                rejection_reason=rejection,
            )

        elif left_usable and not right_usable:
            # === LEFT EYE ONLY ===
            self.recovery_counter = 0
            if c_left >= self.single_eye_conf_thresh:
                # Debounce opposite-eye direct switch if previously tracking opposite eye
                if self.current_state == TRACKING_STATE_RIGHT_ONLY:
                    self.switch_counter += 1
                    if self.switch_counter < 2:
                        return GazeTrackingResult(
                            gaze_direction=GAZE_UNCERTAIN,
                            tracking_state=TRACKING_STATE_UNCERTAIN,
                            left_confidence=c_left,
                            right_confidence=c_right,
                            combined_confidence=round(float(min(0.35, c_left * 0.40)), 4),
                            is_single_eye_fallback=False,
                            left_gaze_dir=dir_l,
                            right_gaze_dir=None,
                            rejection_reason="alternating_single_eye_debounce",
                        )

                self.switch_counter = 0
                tracking_state = TRACKING_STATE_LEFT_ONLY
                self.current_state = TRACKING_STATE_LEFT_ONLY
                gaze_dir = dir_l or GAZE_CENTER
                combined_conf = round(float(min(MAX_SINGLE_EYE_COMBINED_CONF, c_left * 0.85)), 4)
                self.last_valid_direction = gaze_dir
                return GazeTrackingResult(
                    gaze_direction=gaze_dir,
                    tracking_state=tracking_state,
                    left_confidence=c_left,
                    right_confidence=c_right,
                    combined_confidence=combined_conf,
                    is_single_eye_fallback=True,
                    left_gaze_dir=dir_l,
                    right_gaze_dir=None,
                    rejection_reason="",
                )
            else:
                self.switch_counter = 0
                # Left eye usable for paired mode, but insufficient for single-eye fallback
                tracking_state = TRACKING_STATE_UNCERTAIN
                self.current_state = TRACKING_STATE_UNCERTAIN
                return GazeTrackingResult(
                    gaze_direction=GAZE_UNCERTAIN,
                    tracking_state=tracking_state,
                    left_confidence=c_left,
                    right_confidence=c_right,
                    combined_confidence=round(float(min(0.35, c_left * 0.50)), 4),
                    is_single_eye_fallback=False,
                    left_gaze_dir=dir_l,
                    right_gaze_dir=None,
                    rejection_reason="single_eye_confidence_below_threshold",
                )

        elif right_usable and not left_usable:
            # === RIGHT EYE ONLY ===
            self.recovery_counter = 0
            if c_right >= self.single_eye_conf_thresh:
                # Debounce opposite-eye direct switch if previously tracking opposite eye
                if self.current_state == TRACKING_STATE_LEFT_ONLY:
                    self.switch_counter += 1
                    if self.switch_counter < 2:
                        return GazeTrackingResult(
                            gaze_direction=GAZE_UNCERTAIN,
                            tracking_state=TRACKING_STATE_UNCERTAIN,
                            left_confidence=c_left,
                            right_confidence=c_right,
                            combined_confidence=round(float(min(0.35, c_right * 0.40)), 4),
                            is_single_eye_fallback=False,
                            left_gaze_dir=None,
                            right_gaze_dir=dir_r,
                            rejection_reason="alternating_single_eye_debounce",
                        )

                self.switch_counter = 0
                tracking_state = TRACKING_STATE_RIGHT_ONLY
                self.current_state = TRACKING_STATE_RIGHT_ONLY
                gaze_dir = dir_r or GAZE_CENTER
                combined_conf = round(float(min(MAX_SINGLE_EYE_COMBINED_CONF, c_right * 0.85)), 4)
                self.last_valid_direction = gaze_dir
                return GazeTrackingResult(
                    gaze_direction=gaze_dir,
                    tracking_state=tracking_state,
                    left_confidence=c_left,
                    right_confidence=c_right,
                    combined_confidence=combined_conf,
                    is_single_eye_fallback=True,
                    left_gaze_dir=None,
                    right_gaze_dir=dir_r,
                    rejection_reason="",
                )
            else:
                self.switch_counter = 0
                tracking_state = TRACKING_STATE_UNCERTAIN
                self.current_state = TRACKING_STATE_UNCERTAIN
                return GazeTrackingResult(
                    gaze_direction=GAZE_UNCERTAIN,
                    tracking_state=tracking_state,
                    left_confidence=c_left,
                    right_confidence=c_right,
                    combined_confidence=round(float(min(0.35, c_right * 0.50)), 4),
                    is_single_eye_fallback=False,
                    left_gaze_dir=None,
                    right_gaze_dir=dir_r,
                    rejection_reason="single_eye_confidence_below_threshold",
                )

        else:
            # === NEITHER EYE USABLE ===
            self.recovery_counter = 0
            self.switch_counter = 0
            self.current_state = TRACKING_STATE_UNCERTAIN
            return GazeTrackingResult(
                gaze_direction=GAZE_UNCERTAIN,
                tracking_state=TRACKING_STATE_UNCERTAIN,
                left_confidence=c_left,
                right_confidence=c_right,
                combined_confidence=0.0,
                is_single_eye_fallback=False,
                left_gaze_dir=None,
                right_gaze_dir=None,
                rejection_reason="no_usable_pupil_observations",
            )
