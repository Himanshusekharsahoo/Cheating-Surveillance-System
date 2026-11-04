from unittest.mock import MagicMock
import numpy as np
import pytest

from mobile_detection import process_mobile_detection
import config

pytestmark = pytest.mark.unit

class FakeBox:
    def __init__(self, conf: float, cls: int, xyxy=(10, 10, 100, 100)):
        self.conf = [MagicMock(item=lambda: conf)]
        self.cls = [MagicMock(item=lambda: cls)]
        self.xyxy = [xyxy]

class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

def test_mobile_detection_uses_default_config_threshold():
    """Verify that process_mobile_detection defaults to config.PHONE_CONFIDENCE_THRESHOLD."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Box with conf 0.65 is below default 0.70
    mock_model = MagicMock(return_value=[FakeResult([FakeBox(conf=0.65, cls=0)])])
    annotated, detected = process_mobile_detection(frame, model_instance=mock_model)
    assert detected is False

    # Box with conf 0.75 is above default 0.70
    mock_model = MagicMock(return_value=[FakeResult([FakeBox(conf=0.75, cls=0)])])
    annotated, detected = process_mobile_detection(frame, model_instance=mock_model)
    assert detected is True

def test_mobile_detection_respects_custom_threshold():
    """Verify that explicit conf_threshold parameter overrides the default config threshold."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Box with conf 0.55
    mock_model = MagicMock(return_value=[FakeResult([FakeBox(conf=0.55, cls=0)])])

    # With threshold 0.70 -> False
    _, detected_70 = process_mobile_detection(frame, model_instance=mock_model, conf_threshold=0.70)
    assert detected_70 is False

    # With threshold 0.50 -> True
    _, detected_50 = process_mobile_detection(frame, model_instance=mock_model, conf_threshold=0.50)
    assert detected_50 is True

def test_mobile_detection_ignores_non_mobile_class():
    """Verify that non-mobile classes (cls != 0) are ignored even with high confidence."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_model = MagicMock(return_value=[FakeResult([FakeBox(conf=0.99, cls=1)])])
    _, detected = process_mobile_detection(frame, model_instance=mock_model, conf_threshold=0.50)
    assert detected is False
