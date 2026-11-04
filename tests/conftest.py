import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_integrity import register_test_model, reset_model_registry
from network_isolation import configure_offline_environment

# Keep the runtime protections active from collection onward, then restore the
# caller's original process environment when pytest exits.
_OFFLINE_ENV = ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE", "YOLO_CONFIG_DIR")
_ORIGINAL_OFFLINE_ENV = {key: os.environ.get(key) for key in _OFFLINE_ENV}
configure_offline_environment()


def pytest_unconfigure(config):
    for key, value in _ORIGINAL_OFFLINE_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def test_model_registry_fixture(monkeypatch):
    """
    Isolated test fixture providing temporary model registry entries for
    mock-based functional tests that use dummy model files ('fake.dat', 'fake.pt').
    Preserves strict production defaults without an environment variable bypass.
    """
    monkeypatch.setenv("YOLO_OFFLINE", "True")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "False")
    monkeypatch.setenv("YOLO_VERBOSE", "False")
    dummy_hash = hashlib.sha256(b"dummy").hexdigest()
    register_test_model("fake.dat", dummy_hash, 5, "dlib_shape_predictor")
    register_test_model("fake.pt", dummy_hash, 5, "yolo")
    yield
    reset_model_registry()
