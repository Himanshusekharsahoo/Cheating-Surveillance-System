import cv2
import pytest
import config

pytestmark = pytest.mark.hardware

def test_webcam_hardware_stream_available():
    """
    Optional hardware test to verify that a physical webcam device can be opened
    at CAMERA_INDEX and delivers non-empty video frames.
    Skipped by default in automated runs unless invoked with '-m hardware'.
    """
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        pytest.skip(f"No physical webcam detected at index {config.CAMERA_INDEX}; skipping hardware test.")
    ret, frame = cap.read()
    cap.release()
    assert ret is True, "Failed to capture frame from opened camera device."
    assert frame is not None and frame.size > 0, "Captured frame is empty or corrupt."
