"""
Unit and Integration Tests for Phase 10.13.25 — Dependency Hardening & Reproducibility.

Covers:
- Verifying opencv-contrib-python conflict resolution in requirements.txt.
- Verifying pruning of unused packages (jax, jaxlib, mediapipe, facenet-pytorch, seaborn).
- Verifying preservation of the unverified pinned ultralytics version.
- Verifying all required core runtime OpenCV APIs operate correctly without contrib modules.
- Verifying importability and offline isolation compliance of core ML dependencies.
"""

from pathlib import Path
import numpy as np
import cv2
import pytest

import config
from network_isolation import configure_offline_environment, enforce_offline_isolation

pytestmark = pytest.mark.unit


def test_requirements_txt_removes_opencv_contrib():
    """Verify that opencv-contrib-python has been pruned to prevent wheel conflicts."""
    req_file = config.BASE_DIR / "requirements.txt"
    assert req_file.is_file()
    lines = [line.strip().lower() for line in req_file.read_text(encoding="utf-8").splitlines()]
    non_comment_lines = [l for l in lines if l and not l.startswith("#")]

    assert not any("opencv-contrib-python" in l for l in non_comment_lines), (
        "opencv-contrib-python must not be listed in requirements.txt"
    )


def test_requirements_txt_retains_opencv_python():
    """Verify that standard opencv-python is retained."""
    req_file = config.BASE_DIR / "requirements.txt"
    lines = [line.strip().lower() for line in req_file.read_text(encoding="utf-8").splitlines()]
    non_comment_lines = [l for l in lines if l and not l.startswith("#")]

    assert any(l.startswith("opencv-python") for l in non_comment_lines), (
        "opencv-python must be retained in requirements.txt"
    )


def test_requirements_txt_prunes_unused_packages():
    """Verify that unused exploratory packages (jax, jaxlib, mediapipe, facenet-pytorch, seaborn) are pruned."""
    req_file = config.BASE_DIR / "requirements.txt"
    lines = [line.strip().lower() for line in req_file.read_text(encoding="utf-8").splitlines()]
    non_comment_lines = [l for l in lines if l and not l.startswith("#")]

    pruned = ["jax==", "jaxlib==", "mediapipe==", "facenet-pytorch==", "seaborn=="]
    for pkg in pruned:
        assert not any(l.startswith(pkg) for l in non_comment_lines), (
            f"Unused package '{pkg}' must not be listed in active requirements"
        )


def test_requirements_txt_preserves_ultralytics_pin():
    """Verify that the unverified pinned ultralytics version is preserved per safety rules."""
    req_file = config.BASE_DIR / "requirements.txt"
    lines = [line.strip().lower() for line in req_file.read_text(encoding="utf-8").splitlines()]
    non_comment_lines = [l for l in lines if l and not l.startswith("#")]

    assert any(l.startswith("ultralytics==") for l in non_comment_lines), (
        "ultralytics pinned requirement must be present"
    )


def test_opencv_core_apis_functional():
    """Verify that core OpenCV routines used in vision_pipeline, head_pose, and eye_movement work."""
    # Test color conversion
    img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    assert img_gray.shape == (100, 100)

    # Test threshold and contours
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert isinstance(contours, (list, tuple))

    # Test solvePnP geometry with 6 points
    object_pts = np.array([
        (0.0, 0.0, 0.0),
        (0.0, 65.0, 20.0),
        (-33.0, -20.0, 25.0),
        (33.0, -20.0, 25.0),
        (-24.0, 35.0, 20.0),
        (24.0, 35.0, 20.0)
    ], dtype=np.float64)
    image_pts = np.array([
        [320.0, 240.0],
        [320.0, 340.0],
        [260.0, 200.0],
        [380.0, 200.0],
        [280.0, 300.0],
        [360.0, 300.0],
    ], dtype=np.float64)
    cam_matrix = np.array([
        [640.0, 0.0, 320.0],
        [0.0, 640.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    assert success is True
    assert rvec.shape == (3, 1)
    assert tvec.shape == (3, 1)


@pytest.mark.integration
def test_core_runtime_packages_importable():
    """Verify that all core runtime dependencies import cleanly."""
    import numpy
    import cv2
    import dlib
    import torch
    import torchvision
    import ultralytics
    import pytest

    assert numpy.__version__ is not None
    assert cv2.__version__ is not None
    assert dlib.__version__ is not None
    assert torch.__version__ is not None
    assert torchvision.__version__ is not None
    assert ultralytics.__version__ is not None
    assert pytest.__version__ is not None


@pytest.mark.integration
def test_ultralytics_runtime_version_and_offline_isolation():
    """Verify installed Ultralytics runtime complies with offline isolation controls."""
    configure_offline_environment()
    enforce_offline_isolation()

    import ultralytics
    # Verify installed version is 8.4.x
    assert ultralytics.__version__.startswith("8.4")
    # Verify in-memory telemetry suppression
    from ultralytics import settings as ultralytics_settings
    assert ultralytics_settings.get("sync") is False
