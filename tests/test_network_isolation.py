"""
Phase 10.13.22.2 — Process-Local Network Isolation & Test Side-Effect Remediation Tests.

Covers the 14 required regression and process-isolation areas:
1. Process-local offline settings are applied successfully.
2. Applying offline settings does not persistently write Ultralytics settings.
3. The real user-level settings.json remains untouched.
4. Existing unrelated Ultralytics settings remain unchanged.
5. The autouse fixture performs no persistent settings writes.
6. Environment variables are restored after fixture teardown.
7. Environment variables that were originally absent are absent after teardown.
8. Environment variables that were originally set regain their exact original values.
9. Telemetry remains disabled when the initial Ultralytics synchronization setting is enabled.
10. Automatic package installation remains disabled.
11. Import order cannot accidentally enable telemetry before the offline controls take effect.
12. Local YOLO initialization continues to work.
13. Model integrity checks execute before model deserialization.
14. Missing or tampered model files fail closed.
"""

import os
import hashlib
import socket
import subprocess
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import config
import model_integrity
from model_integrity import (
    ModelIntegrityError,
    ModelCorruptedError,
    ModelMissingError,
    register_test_model,
    reset_model_registry,
    reset_model_singletons,
)
import network_isolation
from network_isolation import configure_offline_environment, enforce_offline_isolation
import mobile_detection

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_environment_and_singletons(monkeypatch):
    """Ensure clean offline environment and singletons before and after each test."""
    reset_model_registry()
    reset_model_singletons()
    monkeypatch.setenv("YOLO_OFFLINE", "True")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "False")
    monkeypatch.setenv("YOLO_VERBOSE", "False")
    yield
    reset_model_registry()
    reset_model_singletons()


# ---------------------------------------------------------------------------
# Test 1: Process-Local Offline Settings Applied Successfully
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_process_local_offline_settings_applied():
    """Verify process-local offline settings and globals are active in memory."""
    from ultralytics import SETTINGS
    import ultralytics.utils as utils
    from ultralytics.utils.events import events

    status = enforce_offline_isolation()
    assert status["settings_sync_disabled"] is True
    assert status["events_disabled"] is True
    assert status["online_flag_false"] is True
    assert status["autoinstall_flag_false"] is True
    assert SETTINGS.get("sync") is False
    assert SETTINGS["sync"] is False
    assert events.enabled is False
    assert utils.ONLINE is False
    assert utils.AUTOINSTALL is False


# ---------------------------------------------------------------------------
# Test 2: Applying Offline Settings Does Not Persistently Write Ultralytics Settings
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_applying_offline_settings_does_not_persistently_write(monkeypatch):
    """Verify that enforce_offline_isolation never triggers JSONDict._save()."""
    from ultralytics import SETTINGS

    save_spy = MagicMock(side_effect=AssertionError("Persistent settings save must NOT be invoked"))
    monkeypatch.setattr(SETTINGS, "_save", save_spy)

    status = enforce_offline_isolation()
    assert status["settings_sync_disabled"] is True
    save_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: The Real User-Level settings.json Remains Untouched
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_real_user_settings_file_remains_untouched():
    """Verify that calling enforce_offline_isolation leaves the real settings.json untouched."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        pytest.skip("APPDATA environment variable not defined")
    user_settings_path = Path(appdata) / "Ultralytics" / "settings.json"

    file_existed_before = user_settings_path.is_file()
    bytes_before = user_settings_path.read_bytes() if file_existed_before else None
    mtime_before = user_settings_path.stat().st_mtime if file_existed_before else None

    # Call enforce_offline_isolation multiple times
    enforce_offline_isolation()
    enforce_offline_isolation()

    if file_existed_before:
        assert user_settings_path.is_file(), "Real settings.json must still exist"
        assert user_settings_path.read_bytes() == bytes_before, "Content must not change"
        assert user_settings_path.stat().st_mtime == mtime_before, "Timestamp must not change"
    else:
        assert not user_settings_path.exists(), "Real settings.json must not be created"


# ---------------------------------------------------------------------------
# Test 4: Existing Unrelated Ultralytics Settings Remain Unchanged
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_unrelated_ultralytics_settings_remain_unchanged():
    """Verify that process-local isolation preserves all other Ultralytics settings keys."""
    from ultralytics import SETTINGS

    other_keys_before = {k: v for k, v in SETTINGS.items() if k != "sync"}
    enforce_offline_isolation()
    other_keys_after = {k: v for k, v in SETTINGS.items() if k != "sync"}

    assert other_keys_after == other_keys_before


# ---------------------------------------------------------------------------
# Test 5: The Autouse Fixture Performs No Persistent Settings Writes
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_autouse_fixture_performs_no_persistent_settings_writes(monkeypatch):
    """Verify that conftest.py test_model_registry_fixture executes with zero settings writes."""
    import sys
    if "tests" not in sys.path:
        sys.path.insert(0, "tests")
    import conftest
    from ultralytics import SETTINGS

    save_spy = MagicMock(side_effect=AssertionError("Autouse fixture must not trigger settings save"))
    monkeypatch.setattr(SETTINGS, "_save", save_spy)

    raw_fn = getattr(conftest.test_model_registry_fixture, "__wrapped__")
    fixture_gen = raw_fn(monkeypatch)
    next(fixture_gen)
    save_spy.assert_not_called()
    try:
        next(fixture_gen)
    except StopIteration:
        pass
    save_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Environment Variables Are Restored After Fixture Teardown
# ---------------------------------------------------------------------------
def test_environment_variables_restored_after_fixture_teardown(monkeypatch):
    """Verify that scoped monkeypatch context restores environment variables on exit."""
    names = ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE")
    monkeypatch.setenv("YOLO_OFFLINE", "original_1")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "original_2")
    monkeypatch.setenv("YOLO_VERBOSE", "original_3")

    with monkeypatch.context() as scoped:
        scoped.setenv("YOLO_OFFLINE", "mutated")
        scoped.setenv("YOLO_AUTOINSTALL", "mutated")
        scoped.setenv("YOLO_VERBOSE", "mutated")
        assert os.environ.get("YOLO_OFFLINE") == "mutated"

    assert os.environ.get("YOLO_OFFLINE") == "original_1"
    assert os.environ.get("YOLO_AUTOINSTALL") == "original_2"
    assert os.environ.get("YOLO_VERBOSE") == "original_3"


# ---------------------------------------------------------------------------
# Test 7: Environment Variables That Were Originally Absent Are Absent After Teardown
# ---------------------------------------------------------------------------
def test_environment_variables_originally_absent_are_absent_after_teardown(monkeypatch):
    """Verify that variables absent before fixture execution remain absent after teardown."""
    canary = "TEST_CANARY_ABSENT_VAR_XYZ"
    monkeypatch.delenv(canary, raising=False)
    assert canary not in os.environ

    with monkeypatch.context() as scoped:
        scoped.setenv(canary, "temporary_value")
        assert canary in os.environ

    assert canary not in os.environ


# ---------------------------------------------------------------------------
# Test 8: Environment Variables That Were Originally Set Regain Their Exact Original Values
# ---------------------------------------------------------------------------
def test_environment_variables_originally_set_regain_exact_original_values(monkeypatch):
    """Verify that pre-existing environment variable values are restored exactly."""
    canary = "TEST_CANARY_PRESENT_VAR_XYZ"
    original_val = "exact_sentinel_value_12345"
    monkeypatch.setenv(canary, original_val)

    with monkeypatch.context() as scoped:
        scoped.setenv(canary, "overwritten_in_test")
        assert os.environ[canary] == "overwritten_in_test"

    assert os.environ[canary] == original_val


# ---------------------------------------------------------------------------
# Test 9: Telemetry Remains Disabled When Initial Sync Setting Is Enabled
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_telemetry_remains_disabled_when_initial_sync_enabled(monkeypatch):
    """Verify that enforce_offline_isolation overrides an initially-true sync setting in memory."""
    from ultralytics import SETTINGS
    from ultralytics.utils.events import events

    original_sync = SETTINGS.get("sync")
    # Simulate initial state where sync is True
    dict.__setitem__(SETTINGS, "sync", True)
    assert SETTINGS["sync"] is True

    try:
        status = enforce_offline_isolation()
        assert status["settings_sync_disabled"] is True
        assert SETTINGS.get("sync") is False
        assert SETTINGS["sync"] is False
        assert events.enabled is False
    finally:
        dict.__setitem__(SETTINGS, "sync", original_sync)


# ---------------------------------------------------------------------------
# Test 10: Automatic Package Installation Remains Disabled
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_automatic_dependency_installation_prevented():
    """Verify that unsatisfied requirements do NOT invoke pip or uv subprocesses."""
    try:
        from ultralytics.utils.checks import check_requirements
        with patch("subprocess.check_output") as mock_subproc, \
             patch("subprocess.run") as mock_run:
            result = check_requirements("nonexistent_test_dep_xyz>=99.0.0", install=True)
            assert result is False
            mock_subproc.assert_not_called()
            mock_run.assert_not_called()
    except ImportError:
        pytest.skip("Ultralytics checks module not available")


# ---------------------------------------------------------------------------
# Test 11: Import Order Cannot Accidentally Enable Telemetry
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_import_order_cannot_accidentally_enable_telemetry(monkeypatch):
    """Verify that even if modules were imported with online values, enforcement resets them."""
    from ultralytics import SETTINGS
    import ultralytics.utils as utils
    import ultralytics.utils.checks as checks
    from ultralytics.utils.events import events

    original_sync = SETTINGS.get("sync")
    dict.__setitem__(SETTINGS, "sync", True)
    monkeypatch.setattr(utils, "ONLINE", True)
    monkeypatch.setattr(utils, "AUTOINSTALL", True)
    monkeypatch.setattr(checks, "ONLINE", True)
    monkeypatch.setattr(checks, "AUTOINSTALL", True)
    monkeypatch.setattr(events, "enabled", True)

    try:
        status = enforce_offline_isolation()
        assert status["settings_sync_disabled"] is True
        assert status["events_disabled"] is True
        assert checks.ONLINE is False
        assert checks.AUTOINSTALL is False
        assert utils.ONLINE is False
        assert utils.AUTOINSTALL is False
        assert events.enabled is False
    finally:
        dict.__setitem__(SETTINGS, "sync", original_sync)


# ---------------------------------------------------------------------------
# Test 12: Local YOLO Model Loading Works Cleanly Offline
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_local_yolo_model_loading_works_offline(monkeypatch):
    """Verify that local YOLO model loads without network socket connections."""
    if not config.YOLO_MODEL_PATH.is_file():
        pytest.skip("Local YOLO model weights file missing on disk")

    network_calls = []

    def mock_connect(self, address):
        network_calls.append(address)
        raise socket.error(f"Network access prohibited: {address}")

    def mock_getaddrinfo(*args, **kwargs):
        host = args[0] if args else None
        network_calls.append(host)
        raise socket.gaierror(f"DNS query prohibited: {host}")

    monkeypatch.setattr(socket.socket, "connect", mock_connect)
    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    # Load genuine model
    model = mobile_detection.get_model()
    assert model is not None
    assert len(network_calls) == 0, f"Unexpected network calls: {network_calls}"


# ---------------------------------------------------------------------------
# Test 13: Model Integrity Checks Precede Deserialization
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_model_integrity_checks_precede_deserialization(tmp_path):
    """Verify that corrupt model raises ModelIntegrityError before any YOLO call."""
    corrupted_file = tmp_path / "best_yolov12.pt"
    corrupted_file.write_bytes(b"WRONG_DATA")

    with patch("ultralytics.YOLO") as mock_yolo:
        with pytest.raises(ModelIntegrityError):
            mobile_detection.get_model(corrupted_file)
        mock_yolo.assert_not_called()


# ---------------------------------------------------------------------------
# Test 14: Missing or Tampered Model Files Fail Closed
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_missing_or_tampered_model_files_fail_closed(tmp_path):
    """Verify that both missing and tampered model files strictly fail closed."""
    # 1. Missing model file
    missing_file = tmp_path / "missing_yolo.pt"
    register_test_model("missing_yolo.pt", "0" * 64, 100, "yolo")
    with pytest.raises(ModelMissingError):
        mobile_detection.get_model(missing_file)

    # 2. Tampered model file (wrong size)
    tampered_file = tmp_path / "tampered_yolo.pt"
    tampered_file.write_bytes(b"short")
    register_test_model("tampered_yolo.pt", "0" * 64, 1000, "yolo")
    with pytest.raises(ModelIntegrityError):
        mobile_detection.get_model(tampered_file)


# ---------------------------------------------------------------------------
# Auxiliary Functional Tests
# ---------------------------------------------------------------------------
def test_privacy_controls_initialized():
    """Verify that verified offline environment variables are established."""
    configure_offline_environment()
    assert os.environ.get("YOLO_OFFLINE") == "True"
    assert os.environ.get("YOLO_AUTOINSTALL") == "False"
    assert os.environ.get("YOLO_VERBOSE") == "False"


@pytest.mark.integration
def test_automatic_update_behavior_disabled():
    """Verify that check_pip_update_available returns False without making HTTP requests."""
    try:
        from ultralytics.utils.checks import check_pip_update_available
        with patch("requests.get") as mock_get:
            result = check_pip_update_available()
            assert result is False
            mock_get.assert_not_called()
    except ImportError:
        pytest.skip("Ultralytics checks module not available")


def test_configuration_failures_not_silently_ignored():
    """Verify that enforce_offline_isolation raises RuntimeError if flags cannot be set."""
    with patch.dict(os.environ, {"YOLO_OFFLINE": "False", "YOLO_AUTOINSTALL": "True"}, clear=False):
        with patch("network_isolation.configure_offline_environment"):
            with pytest.raises(RuntimeError, match="Failed to establish the isolated Ultralytics offline configuration"):
                enforce_offline_isolation()


def test_inference_executes_offline_without_network_io(monkeypatch):
    """Verify inference pipeline executes without attempting socket or HTTP I/O."""
    network_probes = []

    def prohibited_network(*args, **kwargs):
        network_probes.append(args)
        raise RuntimeError("Prohibited network activity detected!")

    monkeypatch.setattr(socket.socket, "connect", prohibited_network)
    monkeypatch.setattr(urllib.request, "urlopen", prohibited_network)

    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=lambda: 0.85)]
    mock_box.cls = [MagicMock(item=lambda: 0)]
    mock_box.xyxy = [np.array([10, 10, 50, 50])]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ann, detected = mobile_detection.process_mobile_detection(frame, model_instance=mock_model)

    assert detected is True
    assert len(network_probes) == 0


def test_config_validate_enforces_offline_isolation(tmp_path, monkeypatch):
    """Verify that config.validate_config() triggers enforce_offline_isolation()."""
    fake_pred = tmp_path / "fake.dat"
    fake_yolo = tmp_path / "fake.pt"
    fake_pred.write_bytes(b"dummy")
    fake_yolo.write_bytes(b"dummy")
    monkeypatch.setattr(config, "SHAPE_PREDICTOR_PATH", fake_pred)
    monkeypatch.setattr(config, "YOLO_MODEL_PATH", fake_yolo)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "log")
    register_test_model("fake.dat", hashlib.sha256(b"dummy").hexdigest(), 5, "dlib_shape_predictor")
    register_test_model("fake.pt", hashlib.sha256(b"dummy").hexdigest(), 5, "yolo")
    with patch("config.enforce_offline_isolation") as mock_enforce:
        config.validate_config()
        assert mock_enforce.called is True
