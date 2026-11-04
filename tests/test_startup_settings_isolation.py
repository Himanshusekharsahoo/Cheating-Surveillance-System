"""Startup tests proving Ultralytics settings stay inside an app-controlled temp profile."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_startup(tmp_path, profile, settings_payload=None):
    settings_path = profile / "AppData" / "Roaming" / "Ultralytics" / "settings.json"
    if settings_payload is not None:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_bytes(settings_payload)
    before = settings_path.read_bytes() if settings_path.is_file() else None
    before_mtime = settings_path.stat().st_mtime_ns if settings_path.is_file() else None

    env = os.environ.copy()
    for name in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE"):
        env.pop(name, None)
    env.update({
        "AUDIT_FAKE_HOME": str(profile),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "YOLO_OFFLINE": "False",
        "YOLO_AUTOINSTALL": "True",
        "YOLO_VERBOSE": "True",
    })
    script = textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        Path.home = classmethod(lambda cls: Path(os.environ["AUDIT_FAKE_HOME"]))
        assert not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules)
        import main
        assert not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules)
        from network_isolation import enforce_offline_isolation
        from ultralytics import SETTINGS
        import ultralytics.utils as utils
        import ultralytics.utils.checks as checks
        from ultralytics.utils.events import events
        first = enforce_offline_isolation()
        second = enforce_offline_isolation()
        settings_path = Path(SETTINGS.file).resolve()
        isolated_root = Path(os.environ["YOLO_CONFIG_DIR"]).resolve()
        assert settings_path == isolated_root / "Ultralytics" / "settings.json"
        assert settings_path.is_file()
        assert all(first[key] for key in (
            "env_yolo_offline", "env_yolo_autoinstall", "env_yolo_verbose",
            "config_dir_isolated", "settings_sync_disabled", "events_disabled",
            "online_flag_false", "autoinstall_flag_false"
        ))
        assert all(second[key] for key in first)
        assert os.environ["YOLO_OFFLINE"] == "True"
        assert os.environ["YOLO_AUTOINSTALL"] == "False"
        assert os.environ["YOLO_VERBOSE"] == "False"
        assert utils.ONLINE is False and checks.ONLINE is False
        assert utils.AUTOINSTALL is False and checks.AUTOINSTALL is False
        assert SETTINGS["sync"] is False and events.enabled is False
        print(json.dumps({"isolated_root": str(isolated_root), "settings_written": settings_path.is_file()}))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    after = settings_path.read_bytes() if settings_path.is_file() else None
    after_mtime = settings_path.stat().st_mtime_ns if settings_path.is_file() else None

    if settings_payload is None:
        assert not settings_path.exists(), "Global settings file must not be created"
        assert after is None
    else:
        assert settings_path.is_file(), "Existing settings file must remain in place"
        assert before == after == settings_payload, "Settings file contents must not be modified"
        assert before_mtime == after_mtime, "Settings file mtime must not be changed"

    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["settings_written"] is True
    assert not Path(data["isolated_root"]).exists(), "Temporary directory must be cleaned up on process exit"


@pytest.mark.parametrize(
    "label, payload",
    [
        ("missing", None),
        ("empty", b""),
        ("malformed", b"{invalid json"),
        ("incompatible", b'{"version": "999.0.0"}'),
    ],
)
def test_startup_does_not_create_or_modify_global_settings(tmp_path, label, payload):
    profile = tmp_path / f"profile-{label}"
    _run_isolated_startup(tmp_path, profile, payload)


def test_isolation_setup_failure_fails_closed_before_ultralytics_import(tmp_path):
    env = os.environ.copy()
    for name in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE"):
        env.pop(name, None)
    env.update({"TEMP": str(tmp_path), "TMP": str(tmp_path), "TMPDIR": str(tmp_path)})
    script = textwrap.dedent("""
        import sys, tempfile
        from unittest.mock import patch
        with patch.object(tempfile, "TemporaryDirectory", side_effect=OSError("test denial")):
            try:
                import config
            except RuntimeError as exc:
                assert "isolated Ultralytics configuration directory" in str(exc)
            else:
                raise AssertionError("isolation failure must stop startup")
        assert not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_pytest_session_hook_restores_present_and_absent_environment(tmp_path):
    env = os.environ.copy()
    env.update({
        "YOLO_OFFLINE": "caller-offline-value",
        "YOLO_VERBOSE": "caller-verbose-value",
        "YOLO_CONFIG_DIR": "caller-config-value",
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
    })
    env.pop("YOLO_AUTOINSTALL", None)
    script = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
        import conftest
        assert os.environ["YOLO_OFFLINE"] == "True"
        assert os.environ["YOLO_AUTOINSTALL"] == "False"
        assert os.environ["YOLO_VERBOSE"] == "False"
        assert os.environ["YOLO_CONFIG_DIR"] != "caller-config-value"
        conftest.pytest_unconfigure(None)
        assert os.environ["YOLO_OFFLINE"] == "caller-offline-value"
        assert "YOLO_AUTOINSTALL" not in os.environ
        assert os.environ["YOLO_VERBOSE"] == "caller-verbose-value"
        assert os.environ["YOLO_CONFIG_DIR"] == "caller-config-value"
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _run_preflight_failure_test(tmp_path, failure_setup_code, expected_error_substr):
    """
    Helper to run preflight failure injection tests in an isolated child process
    with simulated fake home and verify that:
    1. A controlled RuntimeError is raised containing expected_error_substr.
    2. Ultralytics is NEVER imported (not in sys.modules).
    3. Global settings file in simulated profile is not created or modified.
    4. CWD / 'Ultralytics' fallback directory is not created.
    """
    profile = tmp_path / "simulated_home"
    global_settings = profile / "AppData" / "Roaming" / "Ultralytics" / "settings.json"
    before_exists = global_settings.exists()
    before_bytes = global_settings.read_bytes() if before_exists else None

    env = os.environ.copy()
    for name in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE"):
        env.pop(name, None)
    env.update({
        "AUDIT_FAKE_HOME": str(profile),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
    })

    dedented_failure = textwrap.dedent(failure_setup_code).strip()

    script = f"""import os, sys, shutil
from pathlib import Path
Path.home = classmethod(lambda cls: Path(os.environ["AUDIT_FAKE_HOME"]))

import network_isolation
network_isolation.configure_offline_environment()

# Inject failure:
{dedented_failure}

# Attempt isolation enforcement:
try:
    network_isolation.enforce_offline_isolation()
    sys.exit(101)  # Expected failure did not occur
except RuntimeError as exc:
    assert "{expected_error_substr}" in str(exc), f"Unexpected error: {{exc}}"

# Verify Ultralytics was NEVER imported:
assert not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules), \\
    "Ultralytics must not be in sys.modules after preflight failure"

# Verify CWD fallback directory was not created:
assert not (Path.cwd() / "Ultralytics").exists(), \\
    "CWD fallback directory 'Ultralytics' must not be created"

print("PREFLIGHT_FAILURE_INJECTION_VERIFIED")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"Child failed (rc={result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "PREFLIGHT_FAILURE_INJECTION_VERIFIED" in result.stdout

    # Verify global settings file was not created or modified
    if not before_exists:
        assert not global_settings.exists(), "Global settings file must not be created during failure injection"
    else:
        assert global_settings.read_bytes() == before_bytes, "Global settings file must not be modified"


def test_preflight_missing_isolated_directory_fails_closed(tmp_path):
    """Verify that a missing isolated directory aborts before Ultralytics import."""
    setup_code = """
    shutil.rmtree(network_isolation._ultralytics_config_dir)
    """
    _run_preflight_failure_test(tmp_path, setup_code, "does not exist")


def test_preflight_isolated_path_replaced_by_file_fails_closed(tmp_path):
    """Verify that an isolated directory replaced by a file aborts before Ultralytics import."""
    setup_code = """
    shutil.rmtree(network_isolation._ultralytics_config_dir)
    network_isolation._ultralytics_config_dir.write_text("not_a_directory")
    """
    _run_preflight_failure_test(tmp_path, setup_code, "not a directory")


def test_preflight_isolated_directory_not_writable_fails_closed(tmp_path):
    """Verify that an un-writable isolated directory aborts before Ultralytics import."""
    setup_code = """
    from unittest.mock import patch
    patch("builtins.open", side_effect=PermissionError("Simulated permission denied on probe file")).start()
    """
    _run_preflight_failure_test(tmp_path, setup_code, "not writable or failed probe")


def test_preflight_isolated_directory_removed_before_import_fails_closed(tmp_path):
    """Verify that removing the entire isolated temp root aborts before Ultralytics import."""
    setup_code = """
    shutil.rmtree(network_isolation._ultralytics_config_owner.name)
    """
    _run_preflight_failure_test(tmp_path, setup_code, "does not exist")


def test_preflight_probe_file_verification_failure_fails_closed(tmp_path):
    """Verify that probe write/sync failure aborts before Ultralytics import."""
    setup_code = """
    from unittest.mock import patch
    patch("os.fsync", side_effect=OSError("Simulated fsync failure on probe")).start()
    """
    _run_preflight_failure_test(tmp_path, setup_code, "not writable or failed probe")


def test_preflight_yolo_config_dir_mismatch_fails_closed(tmp_path):
    """Verify that tampering with YOLO_CONFIG_DIR aborts before Ultralytics import."""
    setup_code = """
    os.environ["YOLO_CONFIG_DIR"] = str(Path(os.environ["AUDIT_FAKE_HOME"]) / "unexpected_dir")
    """
    _run_preflight_failure_test(tmp_path, setup_code, "does not match expected isolated root")
