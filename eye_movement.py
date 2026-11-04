import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

from config import (
    PUPIL_DETECTION_METHOD,
    PUPIL_FIXED_THRESHOLD,
    PUPIL_MIN_CONTOUR_AREA,
    PUPIL_MAX_CONTOUR_RATIO,
    MIN_GAZE_CALIBRATION_SAMPLES,
    GAZE_HORIZONTAL_THRESHOLD,
    GAZE_VERTICAL_THRESHOLD,
    GAZE_UPWARD_THRESHOLD,
)
from model_integrity import verify_model_asset, ModelIntegrityError
from single_eye_fallback import (
    TRACKING_STATE_PAIRED,
    TRACKING_STATE_LEFT_ONLY,
    TRACKING_STATE_RIGHT_ONLY,
    TRACKING_STATE_UNCERTAIN,
    TRACKING_STATE_RECOVERING,
    MIN_EYE_CONFIDENCE,
    SINGLE_EYE_CONFIDENCE_THRESHOLD,
    MAX_SINGLE_EYE_COMBINED_CONF,
    score_eye_confidence,
    SingleEyeGazeTracker,
    GazeTrackingResult,
    evaluate_single_eye_direction,
)

# Resolve base project directory and default predictor path
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTOR_PATH = BASE_DIR / "model" / "shape_predictor_68_face_landmarks.dat"

# Gaze state constants
GAZE_CENTER = "Looking Center"
GAZE_LEFT = "Looking Left"
GAZE_RIGHT = "Looking Right"
GAZE_UP = "Looking Up"
GAZE_DOWN = "Looking Down"
GAZE_NO_FACE = "No Face Detected"
GAZE_UNCERTAIN = "Uncertain Gaze"

COMPLIANT_GAZE_STATES = {GAZE_CENTER}

# Engineering quality limits for neutral calibration; tune only with clean capture data.
GAZE_CALIBRATION_MAX_STDDEV = 0.06
GAZE_CALIBRATION_MAX_RANGE = 0.20
GAZE_BASELINE_MIN_RATIO = 0.10
GAZE_BASELINE_MAX_RATIO = 0.90


def get_predictor_path(predictor_path=None):
    """
    Resolve and validate the dlib shape predictor path.
    Raises FileNotFoundError if the file does not exist.
    """
    target = Path(predictor_path) if predictor_path else DEFAULT_PREDICTOR_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"dlib shape predictor not found at: '{target}'. "
            f"Please ensure 'shape_predictor_68_face_landmarks.dat' exists in '{BASE_DIR / 'model'}'."
        )
    return target


_detector = None
_predictor = None
_default_gaze_tracker = None


def get_default_gaze_tracker() -> SingleEyeGazeTracker:
    """Lazily load shared SingleEyeGazeTracker instance."""
    global _default_gaze_tracker
    if _default_gaze_tracker is None:
        _default_gaze_tracker = SingleEyeGazeTracker()
    return _default_gaze_tracker


def reset_default_gaze_tracker():
    """Resets the state machine of the default tracker and clears cached results."""
    global _default_gaze_tracker, _last_gaze_sample, _last_gaze_diagnostics, _last_gaze_tracking_result
    if _default_gaze_tracker is not None:
        _default_gaze_tracker.reset()
    _last_gaze_sample = None
    _last_gaze_diagnostics = None
    _last_gaze_tracking_result = None


def get_face_detector():
    """Lazily load dlib frontal face detector."""
    global _detector
    if _detector is None:
        import dlib
        _detector = dlib.get_frontal_face_detector()
    return _detector


def get_shape_predictor(predictor_path=None, registry=None):
    """Lazily load dlib shape predictor model after verifying cryptographic integrity."""
    global _predictor
    if _predictor is None:
        target_path = get_predictor_path(predictor_path)
        # Strict fail-closed verification before dlib binary deserialization
        verify_model_asset(target_path, registry=registry)
        import dlib
        _predictor = dlib.shape_predictor(str(target_path))
    return _predictor


def detect_pupil(
    eye_region: Optional[np.ndarray],
    method: Optional[str] = None,
    fixed_threshold: Optional[int] = None,
    min_area: Optional[float] = None,
    max_ratio: Optional[float] = None,
    eye_polygon: Optional[np.ndarray] = None,
    return_diagnostics: bool = False,
) -> Tuple[Any, ...]:
    """
    Detects pupil location within a cropped eye region using adaptive or fixed thresholding,
    optionally constrained by an eye aperture polygon.

    Args:
        eye_region: BGR image crop containing a single eye.
        method: Thresholding method ("adaptive", "otsu", or "fixed"). Defaults to config.
        fixed_threshold: Grayscale cutoff (0-255) for fixed thresholding or fallback.
        min_area: Minimum contour area constraint in pixels.
        max_ratio: Maximum allowable ratio of contour area relative to eye crop area.
        eye_polygon: Optional (N, 2) array of landmark coordinates relative to the crop.
        return_diagnostics: If True, returns a 3-tuple ((center_x, center_y), bbox, diag_info).

    Returns:
        tuple: ((center_x, center_y), (bbox_x, bbox_y, bbox_w, bbox_h)) or (None, None)
               If return_diagnostics is True, appends a dict with validity and rejection reason.
    """
    if eye_region is None or eye_region.size == 0 or eye_region.shape[0] < 3 or eye_region.shape[1] < 3:
        diag = {"valid": False, "rejection_reason": "invalid_or_empty_crop"}
        return (None, None, diag) if return_diagnostics else (None, None)

    h, w = eye_region.shape[:2]
    crop_area = float(w * h)

    min_contour_area = min_area if min_area is not None else PUPIL_MIN_CONTOUR_AREA
    max_contour_area = crop_area * (max_ratio if max_ratio is not None else PUPIL_MAX_CONTOUR_RATIO)
    detection_method = method if method is not None else PUPIL_DETECTION_METHOD
    fallback_thresh = fixed_threshold if fixed_threshold is not None else PUPIL_FIXED_THRESHOLD

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # Preserve detail in the short eye crops produced by 68-point landmarks.
    kernel_size = 3 if min(w, h) <= 14 else 7
    kw = min(kernel_size, w if w % 2 != 0 else w - 1)
    kh = min(kernel_size, h if h % 2 != 0 else h - 1)
    kw = max(3, kw)
    kh = max(3, kh)
    blurred_eye = cv2.GaussianBlur(gray_eye, (kw, kh), 0)

    # 1. Polygon masking if eye landmarks are provided
    mask = None
    if eye_polygon is not None and len(eye_polygon) >= 3:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(eye_polygon)], 255)
        # Closed eye / blink rejection (insufficient pixel area or collapsed aperture height <= 1)
        aperture_h = int(np.max(eye_polygon[:, 1]) - np.min(eye_polygon[:, 1]))
        if cv2.countNonZero(mask) < min_contour_area or aperture_h <= 1:
            diag = {"valid": False, "rejection_reason": "closed_eye_or_blink"}
            return (None, None, diag) if return_diagnostics else (None, None)
        valid_pixels = blurred_eye[mask > 0]
    else:
        valid_pixels = blurred_eye.reshape(-1)

    # Low-contrast / uniform shadow rejection with crop-level fallback
    if len(valid_pixels) == 0:
        diag = {"valid": False, "rejection_reason": "low_contrast_or_uniform_shadow"}
        return (None, None, diag) if return_diagnostics else (None, None)

    aperture_contrast = int(np.max(valid_pixels)) - int(np.min(valid_pixels))
    if aperture_contrast < 12:
        full_crop_contrast = int(np.max(blurred_eye)) - int(np.min(blurred_eye))
        if full_crop_contrast < 15:
            diag = {"valid": False, "rejection_reason": "low_contrast_or_uniform_shadow"}
            return (None, None, diag) if return_diagnostics else (None, None)

    # 2. Determine threshold
    if detection_method == "fixed":
        thresh_val = fallback_thresh
    elif mask is not None:
        p25 = float(np.percentile(valid_pixels, 25))
        otsu_val, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = min(p25, float(otsu_val))
    else:
        thresh_val, _ = cv2.threshold(blurred_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Apply thresholding and mask
    _, threshold_eye = cv2.threshold(blurred_eye, thresh_val, 255, cv2.THRESH_BINARY_INV)
    if mask is not None:
        threshold_eye[mask == 0] = 0

    # Fallback to fixed threshold if primary threshold produced no pixels
    if cv2.countNonZero(threshold_eye) == 0:
        _, threshold_eye = cv2.threshold(blurred_eye, fallback_thresh, 255, cv2.THRESH_BINARY_INV)
        if mask is not None:
            threshold_eye[mask == 0] = 0

    # 4. Extract and filter contour candidates
    found_contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found_contours:
        diag = {"valid": False, "rejection_reason": "no_contours_found"}
        return (None, None, diag) if return_diagnostics else (None, None)

    valid_candidates = []
    rejection_reasons = []

    # Use anisotropic horizontal erosion on small crops (h <= 16) to avoid prematurely
    # eroding the narrow vertical aperture when pupils approach the upper/lower eyelids,
    # while continuing to reject eyelash/shadow artifacts at lateral and medial canthi.
    if mask is not None:
        if h <= 16:
            kw_inner = min(3, w if w % 2 != 0 else w - 1)
            kw_inner = max(1, kw_inner)
            inner_mask = cv2.erode(mask, np.ones((1, kw_inner), dtype=np.uint8), iterations=1)
        else:
            inner_mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    else:
        inner_mask = None

    for c in found_contours:
        c_area = cv2.contourArea(c)
        if c_area < min_contour_area:
            rejection_reasons.append("contour_too_small")
            continue
        if c_area > max_contour_area:
            rejection_reasons.append("contour_too_large")
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Reject full-crop or span artifacts (eyelashes, crop boundaries)
        if bw >= 0.85 * w and w > 10:
            rejection_reasons.append("horizontal_span_artifact")
            continue
        if bh >= 0.85 * h and h > 10:
            rejection_reasons.append("vertical_span_artifact")
            continue
        # Compute spatial centroid using OpenCV moments
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_c = int(M["m10"] / M["m00"])
            cy_c = int(M["m01"] / M["m00"])
        else:
            cx_c = bx + bw // 2
            cy_c = by + bh // 2

        # Validate coordinates before indexing the mask
        if not (0 <= cx_c < w and 0 <= cy_c < h):
            rejection_reasons.append("centroid_out_of_bounds")
            continue

        # Spatial centroid containment inside the eroded inner aperture
        if inner_mask is not None and inner_mask[cy_c, cx_c] == 0:
            rejection_reasons.append("edge_attached_eyelid_or_eyelash")
            continue

        # Height-adaptive vertical centroid bounds to reject boundary margin streaks
        # while permitting realistic upward/downward pupil excursion in short crops.
        if h <= 16:
            min_cy = 1
            max_cy = h - 2
        else:
            v_margin = max(1, int(0.08 * h))
            min_cy = v_margin
            max_cy = h - 1 - v_margin

        if cy_c < min_cy or cy_c > max_cy:
            rejection_reasons.append("edge_attached_eyelid_or_eyelash")
            continue
        aspect = bw / max(1, bh)
        if not (0.15 <= aspect <= 5.5):
            rejection_reasons.append("invalid_aspect_ratio")
            continue
        valid_candidates.append((c_area, c, (bx, by, bw, bh)))

    if not valid_candidates:
        primary_reason = rejection_reasons[0] if rejection_reasons else "no_valid_candidates"
        diag = {"valid": False, "rejection_reason": primary_reason, "contour_area": None}
        return (None, None, diag) if return_diagnostics else (None, None)

    # 5. Select candidate with largest valid contour area
    best = max(valid_candidates, key=lambda x: x[0])
    pupil_contour = best[1]
    px, py, pw, ph = best[2]

    # Compute intensity-weighted center of mass using spatial moments
    M = cv2.moments(pupil_contour)
    if M["m00"] > 0:
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))
    else:
        cx = px + pw // 2
        cy = py + ph // 2

    # Coordinate boundary validation
    if 0 <= cx < w and 0 <= cy < h:
        diag = {"valid": True, "rejection_reason": "", "contour_area": float(best[0])}
        return ((cx, cy), (px, py, pw, ph), diag) if return_diagnostics else ((cx, cy), (px, py, pw, ph))

    diag = {"valid": False, "rejection_reason": "centroid_out_of_bounds", "contour_area": float(best[0])}
    return (None, None, diag) if return_diagnostics else (None, None)


def calculate_gaze_direction(
    left_pupil: Optional[Tuple[int, int]],
    left_eye_w: int,
    left_eye_h: int,
    right_pupil: Optional[Tuple[int, int]],
    right_eye_w: int,
    right_eye_h: int,
    calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    h_thresh: Optional[float] = None,
    v_thresh: Optional[float] = None,
    v_thresh_up: Optional[float] = None,
    allow_single_eye: bool = False,
    left_conf: Optional[float] = None,
    right_conf: Optional[float] = None,
) -> str:
    """
    Determines anatomical gaze direction from unmirrored image coordinates.
    Image x grows rightward, so physical user-right is a negative image-x delta;
    image y grows downward, so physical up is a negative image-y delta.

    Args:
        left_pupil: (x, y) center coordinates of left pupil, or None.
        left_eye_w, left_eye_h: Width and height of left eye bounding rect.
        right_pupil: (x, y) center coordinates of right pupil, or None.
        right_eye_w, right_eye_h: Width and height of right eye bounding rect.
        calibrated_gaze: Optional ((left_x_ratio, left_y_ratio), (right_x_ratio, right_y_ratio)).
        h_thresh: Horizontal deviation threshold from neutral (default: config.GAZE_HORIZONTAL_THRESHOLD).
        v_thresh: Downward vertical deviation threshold from neutral (default: config.GAZE_VERTICAL_THRESHOLD).
        v_thresh_up: Upward vertical deviation threshold from neutral (default: config.GAZE_UPWARD_THRESHOLD).
        allow_single_eye: If True, permits single-eye fallback when the surviving eye satisfies confidence.
        left_conf: Optional precomputed left eye confidence.
        right_conf: Optional precomputed right eye confidence.

    Returns:
        str: GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UP, GAZE_DOWN, or GAZE_UNCERTAIN.
    """
    if left_pupil is None and right_pupil is None:
        return GAZE_UNCERTAIN

    ht = h_thresh if h_thresh is not None else GAZE_HORIZONTAL_THRESHOLD
    vt_down = v_thresh if v_thresh is not None else GAZE_VERTICAL_THRESHOLD
    vt_up = v_thresh_up if v_thresh_up is not None else (v_thresh if v_thresh is not None else GAZE_UPWARD_THRESHOLD)

    # Neutral baseline offsets
    if calibrated_gaze is not None:
        base_lx, base_ly = calibrated_gaze[0] if (len(calibrated_gaze) > 0 and calibrated_gaze[0] is not None) else (0.50, 0.40)
        base_rx, base_ry = calibrated_gaze[1] if (len(calibrated_gaze) > 1 and calibrated_gaze[1] is not None) else (0.50, 0.40)
    else:
        base_lx, base_ly = 0.50, 0.40
        base_rx, base_ry = 0.50, 0.40

    if left_pupil is None or right_pupil is None:
        if not allow_single_eye:
            return GAZE_UNCERTAIN
        # Conservative single-eye fallback
        if left_pupil is not None:
            c_l = left_conf if left_conf is not None else 1.0
            if c_l < SINGLE_EYE_CONFIDENCE_THRESHOLD:
                return GAZE_UNCERTAIN
            return evaluate_single_eye_direction(
                left_pupil, left_eye_w, left_eye_h, (base_lx, base_ly),
                h_thresh=ht, v_thresh_down=vt_down, v_thresh_up=vt_up,
            )
        else:
            c_r = right_conf if right_conf is not None else 1.0
            if c_r < SINGLE_EYE_CONFIDENCE_THRESHOLD:
                return GAZE_UNCERTAIN
            return evaluate_single_eye_direction(
                right_pupil, right_eye_w, right_eye_h, (base_rx, base_ry),
                h_thresh=ht, v_thresh_down=vt_down, v_thresh_up=vt_up,
            )

    lx, ly = left_pupil
    rx, ry = right_pupil

    lw = max(1, left_eye_w)
    lh = max(1, left_eye_h)
    rw = max(1, right_eye_w)
    rh = max(1, right_eye_h)

    # Normalize pupil coordinates independently for each eye
    norm_lx = lx / lw
    norm_rx = rx / rw
    norm_ly = ly / lh
    norm_ry = ry / rh

    # Convert horizontal image-space motion to anatomical user-left/right semantics.
    diff_lx = base_lx - norm_lx
    diff_rx = base_rx - norm_rx
    diff_ly = norm_ly - base_ly
    diff_ry = norm_ry - base_ry

    mean_diff_x = (diff_lx + diff_rx) / 2.0
    mean_diff_y = (diff_ly + diff_ry) / 2.0

    # Evaluate horizontal deflection (allowing physiological asymmetry without contradiction)
    if (diff_lx < -ht and diff_rx < -ht) or (mean_diff_x < -ht and diff_lx <= 0.05 and diff_rx <= 0.05):
        return GAZE_LEFT
    elif (diff_lx > ht and diff_rx > ht) or (mean_diff_x > ht and diff_lx >= -0.05 and diff_rx >= -0.05):
        return GAZE_RIGHT
    # Evaluate vertical deflection with asymmetric upward / downward thresholds
    elif (diff_ly <= -vt_up and diff_ry <= -vt_up) or (mean_diff_y <= -vt_up and diff_ly <= 0.03 and diff_ry <= 0.03):
        return GAZE_UP
    elif (diff_ly >= vt_down and diff_ry >= vt_down) or (mean_diff_y >= vt_down and diff_ly >= -0.03 and diff_ry >= -0.03):
        return GAZE_DOWN
    else:
        return GAZE_CENTER


def calibrate_gaze_baseline(
    samples: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]],
    min_samples: Optional[int] = None,
    max_stddev: Optional[float] = None,
    max_range: Optional[float] = None,
    return_diagnostics: bool = False,
) -> Any:
    """
    Computes a median neutral baseline and rejects unstable or implausible samples.
    Ratios remain image-space values; directional classification maps x deltas to
    anatomical user-left/right semantics for an unmirrored input frame.

    Args:
        samples: List of ((norm_lx, norm_ly), (norm_rx, norm_ry)) sample tuples (or None for failed frames).
        min_samples: Required sample threshold (defaults to config.MIN_GAZE_CALIBRATION_SAMPLES).

    Returns:
        Baseline or None. With return_diagnostics=True, returns (baseline, diagnostics).
    """
    threshold = min_samples if min_samples is not None else MIN_GAZE_CALIBRATION_SAMPLES
    std_limit = max_stddev if max_stddev is not None else GAZE_CALIBRATION_MAX_STDDEV
    range_limit = max_range if max_range is not None else GAZE_CALIBRATION_MAX_RANGE
    rejected = 0

    def result(baseline, reason="", statistics=None):
        info = {
            "failure_reason": reason,
            "sample_count": len(valid),
            "rejected_sample_count": rejected,
            "statistics": statistics or {},
        }
        return (baseline, info) if return_diagnostics else baseline

    valid = []
    for sample in samples or []:
        try:
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise ValueError
            left, right = sample
            if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)) or len(left) != 2 or len(right) != 2:
                raise ValueError
            values = tuple(float(value) for eye in (left, right) for value in eye)
            if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise ValueError
            valid.append(((values[0], values[1]), (values[2], values[3])))
        except (TypeError, ValueError, OverflowError):
            rejected += 1

    if len(valid) < threshold:
        return result(None, f"insufficient valid pupil samples: need {threshold}, got {len(valid)}")

    if len(valid) < 2:
        return result(None, "at least two valid pupil samples are required to measure stability")

    # Reject static degenerate midpoint stubs (e.g. repeated unlocalized full-crop center artifacts at exact 0.50)
    all_exact_midpoints = all(
        abs(s[0][0] - 0.50) < 1e-4 and abs(s[0][1] - 0.50) < 1e-4 and
        abs(s[1][0] - 0.50) < 1e-4 and abs(s[1][1] - 0.50) < 1e-4
        for s in valid
    )
    if all_exact_midpoints:
        return result(None, "all pupil samples are static midpoint values")

    coordinates = {
        "left_x": [s[0][0] for s in valid],
        "left_y": [s[0][1] for s in valid],
        "right_x": [s[1][0] for s in valid],
        "right_y": [s[1][1] for s in valid],
    }
    statistics = {
        name: {
            "variance": float(np.var(values, ddof=1)),
            "stddev": float(np.std(values, ddof=1)),
            "range": float(max(values) - min(values)),
        }
        for name, values in coordinates.items()
    }
    unstable = [
        f"{name} stddev {values['stddev']:.3f} exceeds {std_limit:.3f}"
        for name, values in statistics.items()
        if values["stddev"] > std_limit
    ]
    unstable.extend(
        f"{name} range {values['range']:.3f} exceeds {range_limit:.3f}"
        for name, values in statistics.items()
        if values["range"] > range_limit
    )
    if unstable:
        return result(None, "unstable gaze calibration: " + "; ".join(unstable), statistics)

    med_lx = float(np.median([s[0][0] for s in valid]))
    med_ly = float(np.median([s[0][1] for s in valid]))
    med_rx = float(np.median([s[1][0] for s in valid]))
    med_ry = float(np.median([s[1][1] for s in valid]))

    baseline_values = (med_lx, med_ly, med_rx, med_ry)
    if any(value < GAZE_BASELINE_MIN_RATIO or value > GAZE_BASELINE_MAX_RATIO for value in baseline_values):
        return result(
            None,
            f"invalid neutral pupil baseline: each ratio must be between {GAZE_BASELINE_MIN_RATIO:.2f} and {GAZE_BASELINE_MAX_RATIO:.2f}",
            statistics,
        )

    return result(((med_lx, med_ly), (med_rx, med_ry)), statistics=statistics)


_last_gaze_sample = None
_last_gaze_diagnostics = None
_last_gaze_tracking_result = None


def get_last_gaze_tracking_result() -> Optional[Any]:
    """Returns the most recent GazeTrackingResult from SingleEyeGazeTracker, or None."""
    global _last_gaze_tracking_result
    return _last_gaze_tracking_result


def get_last_gaze_sample() -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Returns the most recent normalized gaze ratio sample tuple, or None."""
    global _last_gaze_sample
    return _last_gaze_sample


def get_last_gaze_diagnostics() -> Optional[Dict[str, Any]]:
    """Returns the most recent detailed gaze diagnostic dictionary, or None."""
    global _last_gaze_diagnostics
    return _last_gaze_diagnostics


def analyze_gaze_from_landmarks(
    frame: np.ndarray,
    landmarks,
    calibrated_gaze: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    tracker: Optional[SingleEyeGazeTracker] = None,
) -> Tuple[np.ndarray, str]:
    """
    Analyzes gaze direction for a single face using pre-extracted facial landmarks.
    Annotates eye bounding boxes and pupil centers on the frame.

    Returns:
        tuple: (annotated_frame, gaze_direction_str)
    """
    global _last_gaze_sample, _last_gaze_diagnostics, _last_gaze_tracking_result
    _last_gaze_tracking_result = None
    h_frame, w_frame = frame.shape[:2]

    # Extract left and right eye landmarks (dlib 68-point model: 36-41 and 42-47)
    left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
    right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

    # Get bounding rectangles for each eye
    left_eye_rect = cv2.boundingRect(left_eye_points)
    right_eye_rect = cv2.boundingRect(right_eye_points)

    lx, ly, lw, lh = left_eye_rect
    rx, ry, rw, rh = right_eye_rect

    lx1, ly1 = max(0, lx), max(0, ly)
    lx2, ly2 = min(w_frame, lx + lw), min(h_frame, ly + lh)

    rx1, ry1 = max(0, rx), max(0, ry)
    rx2, ry2 = min(w_frame, rx + rw), min(h_frame, ry + rh)

    if lx2 <= lx1 or ly2 <= ly1 or rx2 <= rx1 or ry2 <= ry1:
        _last_gaze_sample = None
        _last_gaze_tracking_result = None
        _last_gaze_diagnostics = {
            "face_detected": True,
            "landmarks_detected": True,
            "missing_face": False,
            "missing_landmarks": False,
            "missing_pupils": True,
            "missing_left_pupil": True,
            "missing_right_pupil": True,
            "left_pupil_detected": False,
            "right_pupil_detected": False,
            "left_pupil_x": None,
            "left_pupil_y": None,
            "right_pupil_x": None,
            "right_pupil_y": None,
            "left_eye_w": lw,
            "left_eye_h": lh,
            "right_eye_w": rw,
            "right_eye_h": rh,
            "norm_left_x": None,
            "norm_left_y": None,
            "norm_right_x": None,
            "norm_right_y": None,
            "left_pupil_valid": False,
            "left_rejection_reason": "degenerate_eye_crop",
            "right_pupil_valid": False,
            "right_rejection_reason": "degenerate_eye_crop",
            "is_calibrated": (calibrated_gaze is not None),
            "baseline_left_x": calibrated_gaze[0][0] if (calibrated_gaze and calibrated_gaze[0]) else None,
            "baseline_left_y": calibrated_gaze[0][1] if (calibrated_gaze and calibrated_gaze[0]) else None,
            "baseline_right_x": calibrated_gaze[1][0] if (calibrated_gaze and calibrated_gaze[1]) else None,
            "baseline_right_y": calibrated_gaze[1][1] if (calibrated_gaze and calibrated_gaze[1]) else None,
            "diff_left_x": None,
            "diff_left_y": None,
            "diff_right_x": None,
            "diff_right_y": None,
            "mean_diff_x": None,
            "mean_diff_y": None,
            "gaze_direction": GAZE_UNCERTAIN,
            "left_confidence": 0.0,
            "right_confidence": 0.0,
            "combined_confidence": 0.0,
            "tracking_state": TRACKING_STATE_UNCERTAIN,
            "is_single_eye_fallback": False,
        }
        return frame, GAZE_UNCERTAIN

    # Apply bounded vertical padding (min 2px, 20% of eye height) to provide headroom
    # for upward pupil excursion without clipping against the crop border or eyelid margin.
    v_pad_l = max(2, int(round(0.20 * lh)))
    crop_ly1 = max(0, ly - v_pad_l)
    crop_ly2 = min(h_frame, ly + lh + v_pad_l)
    crop_lx1 = max(0, lx)
    crop_lx2 = min(w_frame, lx + lw)
    pad_top_l = ly - crop_ly1

    v_pad_r = max(2, int(round(0.20 * rh)))
    crop_ry1 = max(0, ry - v_pad_r)
    crop_ry2 = min(h_frame, ry + rh + v_pad_r)
    crop_rx1 = max(0, rx)
    crop_rx2 = min(w_frame, rx + rw)
    pad_top_r = ry - crop_ry1

    left_eye = frame[crop_ly1:crop_ly2, crop_lx1:crop_lx2]
    right_eye = frame[crop_ry1:crop_ry2, crop_rx1:crop_rx2]

    # Convert landmarks to eye crop local coordinates for aperture masking
    left_poly = left_eye_points - np.array([crop_lx1, crop_ly1])
    right_poly = right_eye_points - np.array([crop_rx1, crop_ry1])

    # Detect pupils using adaptive thresholding with polygon masking and fallback
    left_pupil, left_bbox, left_diag = detect_pupil(left_eye, eye_polygon=left_poly, return_diagnostics=True)
    right_pupil, right_bbox, right_diag = detect_pupil(right_eye, eye_polygon=right_poly, return_diagnostics=True)

    # Draw bounding boxes (unpadded anatomical eyelid aperture rects)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

    # Map detected pupil positions from padded crop space back to original aperture space
    if left_pupil is not None and left_bbox is not None:
        left_pupil_orig = (left_pupil[0] - (lx - crop_lx1), left_pupil[1] - pad_top_l)
        cv2.circle(frame, (crop_lx1 + left_pupil[0], crop_ly1 + left_pupil[1]), 5, (0, 0, 255), -1)
    else:
        left_pupil_orig = None

    if right_pupil is not None and right_bbox is not None:
        right_pupil_orig = (right_pupil[0] - (rx - crop_rx1), right_pupil[1] - pad_top_r)
        cv2.circle(frame, (crop_rx1 + right_pupil[0], crop_ry1 + right_pupil[1]), 5, (0, 0, 255), -1)
    else:
        right_pupil_orig = None

    # Calculate gaze tracking result using SingleEyeGazeTracker and per-eye confidence scoring
    left_conf = score_eye_confidence(
        pupil=left_pupil_orig,
        eye_w=lw,
        eye_h=lh,
        eye_crop=left_eye,
        eye_polygon=left_poly,
        diag_info=left_diag,
        bbox=left_bbox,
        contour_area=left_diag.get("contour_area"),
    )
    right_conf = score_eye_confidence(
        pupil=right_pupil_orig,
        eye_w=rw,
        eye_h=rh,
        eye_crop=right_eye,
        eye_polygon=right_poly,
        diag_info=right_diag,
        bbox=right_bbox,
        contour_area=right_diag.get("contour_area"),
    )

    active_tracker = tracker or get_default_gaze_tracker()
    tracking_res = active_tracker.update(
        left_pupil=left_pupil_orig,
        left_w=lw,
        left_h=lh,
        right_pupil=right_pupil_orig,
        right_w=rw,
        right_h=rh,
        calibrated_gaze=calibrated_gaze,
        left_conf=left_conf,
        right_conf=right_conf,
    )
    _last_gaze_tracking_result = tracking_res
    gaze_direction = tracking_res.gaze_direction

    if left_pupil_orig and right_pupil_orig and lw > 0 and lh > 0 and rw > 0 and rh > 0:
        _last_gaze_sample = ((left_pupil_orig[0] / lw, left_pupil_orig[1] / lh), (right_pupil_orig[0] / rw, right_pupil_orig[1] / rh))
    else:
        _last_gaze_sample = None

    norm_lx = (left_pupil_orig[0] / lw) if (left_pupil_orig and lw > 0) else None
    norm_ly = (left_pupil_orig[1] / lh) if (left_pupil_orig and lh > 0) else None
    norm_rx = (right_pupil_orig[0] / rw) if (right_pupil_orig and rw > 0) else None
    norm_ry = (right_pupil_orig[1] / rh) if (right_pupil_orig and rh > 0) else None

    is_calib = (calibrated_gaze is not None)
    base_lx = calibrated_gaze[0][0] if is_calib else None
    base_ly = calibrated_gaze[0][1] if is_calib else None
    base_rx = calibrated_gaze[1][0] if is_calib else None
    base_ry = calibrated_gaze[1][1] if is_calib else None

    diff_lx = (base_lx - norm_lx) if (norm_lx is not None and base_lx is not None) else None
    diff_ly = (norm_ly - base_ly) if (norm_ly is not None and base_ly is not None) else None
    diff_rx = (base_rx - norm_rx) if (norm_rx is not None and base_rx is not None) else None
    diff_ry = (norm_ry - base_ry) if (norm_ry is not None and base_ry is not None) else None

    mean_diff_x = ((diff_lx + diff_rx) / 2.0) if (diff_lx is not None and diff_rx is not None) else None
    mean_diff_y = ((diff_ly + diff_ry) / 2.0) if (diff_ly is not None and diff_ry is not None) else None

    _last_gaze_diagnostics = {
        "face_detected": True,
        "landmarks_detected": True,
        "missing_face": False,
        "missing_landmarks": False,
        "missing_pupils": (left_pupil_orig is None or right_pupil_orig is None),
        "missing_left_pupil": (left_pupil_orig is None),
        "missing_right_pupil": (right_pupil_orig is None),
        "left_pupil_detected": (left_pupil_orig is not None),
        "right_pupil_detected": (right_pupil_orig is not None),
        "left_pupil_x": left_pupil_orig[0] if left_pupil_orig else None,
        "left_pupil_y": left_pupil_orig[1] if left_pupil_orig else None,
        "right_pupil_x": right_pupil_orig[0] if right_pupil_orig else None,
        "right_pupil_y": right_pupil_orig[1] if right_pupil_orig else None,
        "left_eye_w": lw,
        "left_eye_h": lh,
        "right_eye_w": rw,
        "right_eye_h": rh,
        "norm_left_x": norm_lx,
        "norm_left_y": norm_ly,
        "norm_right_x": norm_rx,
        "norm_right_y": norm_ry,
        "left_pupil_valid": bool(left_diag.get("valid", False)),
        "left_rejection_reason": left_diag.get("rejection_reason", ""),
        "right_pupil_valid": bool(right_diag.get("valid", False)),
        "right_rejection_reason": right_diag.get("rejection_reason", ""),
        "is_calibrated": is_calib,
        "baseline_left_x": base_lx,
        "baseline_left_y": base_ly,
        "baseline_right_x": base_rx,
        "baseline_right_y": base_ry,
        "diff_left_x": diff_lx,
        "diff_left_y": diff_ly,
        "diff_right_x": diff_rx,
        "diff_right_y": diff_ry,
        "mean_diff_x": mean_diff_x,
        "mean_diff_y": mean_diff_y,
        "gaze_direction": gaze_direction,
        "left_confidence": tracking_res.left_confidence,
        "right_confidence": tracking_res.right_confidence,
        "combined_confidence": tracking_res.combined_confidence,
        "tracking_state": tracking_res.tracking_state,
        "is_single_eye_fallback": tracking_res.is_single_eye_fallback,
    }

    return frame, gaze_direction


def process_eye_movement(frame: np.ndarray, detector=None, predictor=None, calibrated_gaze=None) -> Tuple[np.ndarray, str]:
    """
    Tracks eye movement and gaze direction for faces in the frame.
    Backward-compatibility wrapper around face detection and landmark gaze analysis.
    """
    face_detector = detector or get_face_detector()
    landmark_predictor = predictor or get_shape_predictor()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_detector(gray)
    except Exception:
        reset_default_gaze_tracker()
        return frame, GAZE_UNCERTAIN

    if not faces or len(faces) == 0:
        global _last_gaze_sample, _last_gaze_diagnostics, _last_gaze_tracking_result
        get_default_gaze_tracker().reset()
        _last_gaze_sample = None
        _last_gaze_tracking_result = None
        _last_gaze_diagnostics = {
            "face_detected": False,
            "landmarks_detected": False,
            "missing_face": True,
            "missing_landmarks": True,
            "missing_pupils": True,
            "missing_left_pupil": True,
            "missing_right_pupil": True,
            "left_pupil_detected": False,
            "right_pupil_detected": False,
            "left_pupil_x": None,
            "left_pupil_y": None,
            "right_pupil_x": None,
            "right_pupil_y": None,
            "left_eye_w": None,
            "left_eye_h": None,
            "right_eye_w": None,
            "right_eye_h": None,
            "norm_left_x": None,
            "norm_left_y": None,
            "norm_right_x": None,
            "norm_right_y": None,
            "left_pupil_valid": False,
            "left_rejection_reason": "no_face_detected",
            "right_pupil_valid": False,
            "right_rejection_reason": "no_face_detected",
            "is_calibrated": (calibrated_gaze is not None),
            "baseline_left_x": calibrated_gaze[0][0] if (calibrated_gaze and calibrated_gaze[0]) else None,
            "baseline_left_y": calibrated_gaze[0][1] if (calibrated_gaze and calibrated_gaze[0]) else None,
            "baseline_right_x": calibrated_gaze[1][0] if (calibrated_gaze and calibrated_gaze[1]) else None,
            "baseline_right_y": calibrated_gaze[1][1] if (calibrated_gaze and calibrated_gaze[1]) else None,
            "diff_left_x": None,
            "diff_left_y": None,
            "diff_right_x": None,
            "diff_right_y": None,
            "mean_diff_x": None,
            "mean_diff_y": None,
            "gaze_direction": GAZE_NO_FACE,
            "left_confidence": 0.0,
            "right_confidence": 0.0,
            "combined_confidence": 0.0,
            "tracking_state": TRACKING_STATE_UNCERTAIN,
            "is_single_eye_fallback": False,
        }
        return frame, GAZE_NO_FACE

    primary_face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
    gaze_direction = GAZE_UNCERTAIN
    for face in faces:
        try:
            landmarks = landmark_predictor(gray, face)
        except Exception:
            reset_default_gaze_tracker()
            return frame, GAZE_UNCERTAIN
        if face == primary_face:
            frame, gaze_direction = analyze_gaze_from_landmarks(
                frame, landmarks, calibrated_gaze=calibrated_gaze, tracker=get_default_gaze_tracker()
            )
        else:
            frame, _ = analyze_gaze_from_landmarks(
                frame, landmarks, calibrated_gaze=calibrated_gaze, tracker=SingleEyeGazeTracker()
            )

    return frame, gaze_direction
