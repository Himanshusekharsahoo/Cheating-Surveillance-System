import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, List
import cv2
import numpy as np

from network_isolation import configure_offline_environment, enforce_offline_isolation

# Establish verified offline environment variables before third-party model packages load
configure_offline_environment()

from config import (
    YOLO_MODEL_PATH,
    PHONE_CONFIDENCE_THRESHOLD,
    PHONE_DETECTION_INTERVAL_FRAMES,
    PHONE_DETECTION_CACHE_EXPIRY_FRAMES,
)
from model_integrity import verify_model_asset, ModelIntegrityError, ModelMissingError

# Resolve base project directory and default model path
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = YOLO_MODEL_PATH

_model = None

# Detection state constants
DETECTION_STATE_FRESH_POSITIVE = "FRESH_POSITIVE"
DETECTION_STATE_FRESH_NEGATIVE = "FRESH_NEGATIVE"
DETECTION_STATE_CACHED_POSITIVE = "CACHED_POSITIVE"
DETECTION_STATE_CACHED_NEGATIVE = "CACHED_NEGATIVE"
DETECTION_STATE_EXPIRED = "EXPIRED"
DETECTION_STATE_INFERENCE_ERROR = "INFERENCE_ERROR"


@dataclass
class PhoneDetectionResult:
    """Holds metadata for a single frame's phone detection."""
    detected: bool = False
    confidence: Optional[float] = None
    boxes: list = field(default_factory=list)
    state: str = DETECTION_STATE_FRESH_NEGATIVE
    age_frames: int = 0


def get_model_path(model_path=None):
    """
    Resolve and validate the YOLO model path.
    Raises ModelMissingError (FileNotFoundError) if the model file does not exist.
    """
    target = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target.is_file():
        raise ModelMissingError(
            f"YOLO model file not found at: '{target}'. "
            f"Please ensure 'best_yolov12.pt' exists in '{BASE_DIR / 'model'}'."
        )
    return target


def get_model(model_path=None, registry=None):
    """
    Lazily loads and returns the YOLO model singleton after verifying cryptographic integrity
    and enforcing offline telemetry suppression.
    Raises ModelIntegrityError if the model is unapproved, missing, or corrupted.
    """
    global _model
    if _model is None:
        target_path = get_model_path(model_path)
        # Enforce telemetry suppression and offline isolation before YOLO is initialized
        enforce_offline_isolation()
        # Strict fail-closed verification before YOLO/PyTorch deserialization
        verify_model_asset(target_path, registry=registry)
        import torch
        from ultralytics import YOLO
        _model = YOLO(str(target_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
    return _model


_last_detected_confidence = None


def get_last_detected_confidence():
    """Returns the highest detection confidence from the most recent mobile detection, or None."""
    global _last_detected_confidence
    return _last_detected_confidence


class PhoneDetectionTracker:
    """
    Manages YOLO inference cadence, cached detection reuse, result expiry, and failure resilience.
    """

    def __init__(
        self,
        interval_frames: Optional[int] = None,
        cache_expiry_frames: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        model_instance: Optional[Any] = None,
    ):
        self.interval_frames: int = (
            interval_frames if interval_frames is not None else PHONE_DETECTION_INTERVAL_FRAMES
        )
        self.cache_expiry_frames: int = (
            cache_expiry_frames if cache_expiry_frames is not None else PHONE_DETECTION_CACHE_EXPIRY_FRAMES
        )
        self.conf_threshold: float = (
            conf_threshold if conf_threshold is not None else PHONE_CONFIDENCE_THRESHOLD
        )
        self.model_instance = model_instance

        self.frame_count: int = 0
        self.inference_count: int = 0
        self.last_age: int = 0
        self._last_boxes: List[Tuple[int, int, int, int, float, str]] = []
        self._last_detected: bool = False
        self._last_conf: Optional[float] = None
        self._last_state: str = DETECTION_STATE_FRESH_NEGATIVE

    def reset(self) -> None:
        """Resets tracker state."""
        self.frame_count = 0
        self.inference_count = 0
        self.last_age = 0
        self._last_boxes = []
        self._last_detected = False
        self._last_conf = None
        self._last_state = DETECTION_STATE_FRESH_NEGATIVE

    def process_frame(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        display_frame: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, PhoneDetectionResult]:
        """
        Processes a frame with configured YOLO inference cadence.
        Reuses cached detections between inference frames; expires stale detections after cache_expiry_frames.
        Catches inference errors gracefully.

        Args:
            frame: Clean, unannotated BGR image frame used for YOLO inference.
            annotate: Whether to draw bounding boxes and confidence labels.
            display_frame: Optional separate frame buffer where visual annotations
                are drawn. If provided, 'frame' remains strictly unmutated.
                If None, annotations are drawn directly on 'frame'.

        Returns:
            Tuple[np.ndarray, bool, PhoneDetectionResult]: (annotated_frame, detected, result)
        """
        global _last_detected_confidence
        self.frame_count += 1

        # Run inference if interval is 1, on first frame, or on modulo boundary
        run_inference = (
            (self.interval_frames <= 1)
            or (self.frame_count == 1)
            or ((self.frame_count - 1) % self.interval_frames == 0)
        )

        if run_inference:
            try:
                model = self.model_instance or get_model()
                results = model(frame, verbose=False)
                self.inference_count += 1

                detected = False
                max_conf = None
                boxes = []

                for result in results:
                    if not hasattr(result, "boxes") or result.boxes is None:
                        continue
                    for box in result.boxes:
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())

                        if conf < self.conf_threshold or cls != 0:
                            continue

                        max_conf = conf if max_conf is None else max(max_conf, conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"Mobile ({conf:.2f})"
                        boxes.append((x1, y1, x2, y2, conf, label))
                        detected = True

                self._last_boxes = boxes
                self._last_detected = detected
                self._last_conf = max_conf
                self.last_age = 0
                state = DETECTION_STATE_FRESH_POSITIVE if detected else DETECTION_STATE_FRESH_NEGATIVE
                self._last_state = state

            except Exception as e:
                # Catch inference failure cleanly without crashing the processing loop
                print(f"Warning: YOLO inference error on frame {self.frame_count}: {e}")
                self.last_age += 1
                if self.last_age > self.cache_expiry_frames:
                    state = DETECTION_STATE_EXPIRED
                    detected = False
                    max_conf = None
                    boxes = []
                    self._last_boxes = []
                    self._last_detected = False
                    self._last_conf = None
                else:
                    state = DETECTION_STATE_INFERENCE_ERROR
                    # Retain last detection values for grace period, but clear bounding boxes so stale boxes are not drawn
                    detected = self._last_detected
                    max_conf = self._last_conf
                    boxes = []
                self._last_state = state
        else:
            # Cadence skip frame: reuse cached detection if not expired
            self.last_age += 1
            if self.last_age > self.cache_expiry_frames:
                state = DETECTION_STATE_EXPIRED
                detected = False
                max_conf = None
                boxes = []
                self._last_boxes = []
                self._last_detected = False
                self._last_conf = None
            else:
                detected = self._last_detected
                max_conf = self._last_conf
                boxes = list(self._last_boxes)
                state = DETECTION_STATE_CACHED_POSITIVE if detected else DETECTION_STATE_CACHED_NEGATIVE
            self._last_state = state

        # Draw annotations on display_frame if provided, else on frame
        target_frame = display_frame if display_frame is not None else frame
        if annotate and target_frame is not None and len(boxes) > 0:
            for x1, y1, x2, y2, conf, label in boxes:
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                tag = f"{label} (cached)" if state == DETECTION_STATE_CACHED_POSITIVE else label
                cv2.rectangle(target_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(target_frame, tag, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _last_detected_confidence = max_conf
        res = PhoneDetectionResult(
            detected=detected,
            confidence=max_conf,
            boxes=boxes,
            state=state,
            age_frames=self.last_age,
        )
        return target_frame, detected, res


def process_mobile_detection(
    frame: np.ndarray,
    model_instance: Optional[Any] = None,
    conf_threshold: Optional[float] = None,
    tracker: Optional[PhoneDetectionTracker] = None,
    display_frame: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Detects mobile phones in a frame using YOLO.
    Args:
        frame: The BGR image frame from camera or video (clean unannotated pixels).
        model_instance: Optional loaded YOLO model instance.
        conf_threshold: Optional float threshold in (0, 1]. Defaults to config.PHONE_CONFIDENCE_THRESHOLD.
        tracker: Optional PhoneDetectionTracker instance for cadence and caching.
        display_frame: Optional separate frame buffer for drawing annotations.
    Returns:
        tuple: (annotated_frame, mobile_detected_boolean)
    """
    if tracker is not None:
        annotated, detected, _ = tracker.process_frame(
            frame, annotate=True, display_frame=display_frame
        )
        return annotated, detected

    target_frame = display_frame if display_frame is not None else frame
    global _last_detected_confidence
    threshold = conf_threshold if conf_threshold is not None else PHONE_CONFIDENCE_THRESHOLD
    model = model_instance or get_model()
    results = model(frame, verbose=False)
    mobile_detected = False
    max_conf = None

    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue
        for box in result.boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if conf < threshold or cls != 0:  # Mobile class index is 0
                continue

            max_conf = conf if max_conf is None else max(max_conf, conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"Mobile ({conf:.2f})"

            # Prevent text label from drawing off-screen
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.rectangle(target_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(target_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mobile_detected = True

    _last_detected_confidence = max_conf
    return target_frame, mobile_detected
