"""
Ultralytics Telemetry Suppression & Network Isolation Module.

Enforces offline isolation for the Ultralytics YOLO runtime in the Cheating Surveillance System.
Verified against Ultralytics version 8.4.182 on Python 3.13.5 Windows AMD64.

Mechanisms verified from installed source code:
1. YOLO_OFFLINE="True": Forces ultralytics.utils.is_online() to return False without
   performing DNS lookups to Cloudflare (one.one.one.one) or Google (dns.google).
   Sets global ONLINE=False at module import time, which disables check_pip_update_available(),
   events.py Google Analytics telemetry, and set_sentry() crash reporting.
2. YOLO_AUTOINSTALL="False": Forces ultralytics.utils.AUTOINSTALL=False, preventing
   check_requirements() from silently running pip/uv subprocesses to install dependencies.
3. YOLO_VERBOSE="False": Quiets verbose initialization banners.
4. Pre-initialization validation: An app-controlled, process-lifetime TemporaryDirectory
   is created and preflight-probed (write/flush/fsync/unlink) before Ultralytics is imported.
   If preflight fails, execution aborts with a controlled error and Ultralytics is NEVER imported,
   preventing silent fallback to CWD or global settings.
5. In-memory process-local sync suppression: SETTINGS["sync"] is set via dict.__setitem__
   without invoking JSONDict persistence.
6. events.enabled=False: Explicitly disables the Events analytics dispatcher singleton.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

_ultralytics_config_owner: Optional[tempfile.TemporaryDirectory] = None
_ultralytics_config_dir: Optional[Path] = None
_isolation_established: bool = False


def validate_isolated_config_dir_preflight() -> None:
    """
    Validates that the isolated Ultralytics configuration directory exists,
    is a real directory (not symlink or file), is writable, and is probed
    before Ultralytics is imported.

    Raises RuntimeError if any precondition fails, ensuring Ultralytics is
    never imported into sys.modules when isolation cannot be guaranteed.
    """
    global _ultralytics_config_owner, _ultralytics_config_dir

    # 1. Reject if Ultralytics was already imported before configuration
    if _ultralytics_config_owner is None:
        if any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules):
            raise RuntimeError(
                "Cannot isolate Ultralytics settings: Ultralytics was imported before application configuration."
            )
        raise RuntimeError("Isolated Ultralytics configuration directory has not been established.")

    try:
        owner_path = Path(_ultralytics_config_owner.name)
        config_dir = Path(_ultralytics_config_dir)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid isolated config directory reference: {exc}") from exc

    # 2. Verify owner path exists, is a real directory, and is not a symlink
    if not owner_path.exists():
        raise RuntimeError(f"Isolated config root '{owner_path}' does not exist.")
    if not owner_path.is_dir():
        raise RuntimeError(f"Isolated config root '{owner_path}' is not a directory.")
    if owner_path.is_symlink():
        raise RuntimeError(f"Isolated config root '{owner_path}' is an unexpected symlink.")

    # 3. Verify config_dir exists, is a real directory, and is not a symlink
    if not config_dir.exists():
        raise RuntimeError(f"Isolated config directory '{config_dir}' does not exist.")
    if not config_dir.is_dir():
        raise RuntimeError(f"Isolated config directory '{config_dir}' is not a directory.")
    if config_dir.is_symlink():
        raise RuntimeError(f"Isolated config directory '{config_dir}' is an unexpected symlink.")

    # 4. Verify YOLO_CONFIG_DIR environment variable exactly matches owner path
    env_dir = os.environ.get("YOLO_CONFIG_DIR")
    if not env_dir:
        raise RuntimeError("YOLO_CONFIG_DIR environment variable is not set.")
    try:
        env_resolved = Path(env_dir).resolve()
        owner_resolved = owner_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to resolve YOLO_CONFIG_DIR path: {exc}") from exc

    if env_resolved != owner_resolved:
        raise RuntimeError(
            f"YOLO_CONFIG_DIR '{env_dir}' does not match expected isolated root '{owner_path}'."
        )

    # 5. Execute live probe: write, flush, fsync, verify, and remove a harmless unique probe file
    probe_id = uuid.uuid4().hex
    probe_file = config_dir / f".probe_{probe_id}.tmp"
    try:
        with open(probe_file, "wb") as f:
            f.write(b"preflight_probe_ok\n")
            f.flush()
            os.fsync(f.fileno())
        if not probe_file.is_file() or probe_file.stat().st_size == 0:
            raise RuntimeError(f"Probe file '{probe_file}' could not be verified on disk.")
    except (OSError, PermissionError, ValueError) as exc:
        raise RuntimeError(
            f"Isolated config directory '{config_dir}' is not writable or failed probe: {exc}"
        ) from exc
    finally:
        try:
            if probe_file.exists():
                probe_file.unlink()
        except OSError:
            pass


def _ensure_application_config_dir() -> None:
    """Create one private, process-lifetime config directory before Ultralytics loads."""
    global _ultralytics_config_owner, _ultralytics_config_dir

    if _ultralytics_config_owner is None:
        if any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules):
            raise RuntimeError(
                "Cannot isolate Ultralytics settings: Ultralytics was imported before application configuration."
            )
        owner = None
        try:
            owner = tempfile.TemporaryDirectory(prefix="cheating-surveillance-ultralytics-")
            config_dir = Path(owner.name) / "Ultralytics"
            config_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            if owner is not None:
                owner.cleanup()
            raise RuntimeError("Could not create an isolated Ultralytics configuration directory.") from exc
        _ultralytics_config_owner = owner
        _ultralytics_config_dir = config_dir
        os.environ["YOLO_CONFIG_DIR"] = str(_ultralytics_config_owner.name)
    else:
        current_yolo_dir = os.environ.get("YOLO_CONFIG_DIR")
        if current_yolo_dir is not None:
            try:
                if Path(current_yolo_dir).resolve() != Path(_ultralytics_config_owner.name).resolve():
                    raise RuntimeError(
                        f"YOLO_CONFIG_DIR '{current_yolo_dir}' does not match expected isolated root '{_ultralytics_config_owner.name}'."
                    )
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(f"Failed to resolve YOLO_CONFIG_DIR path: {exc}") from exc
        else:
            os.environ["YOLO_CONFIG_DIR"] = str(_ultralytics_config_owner.name)
    if not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules):
        validate_isolated_config_dir_preflight()


def configure_offline_environment() -> None:
    """
    Sets verified environment variables that must be present before Ultralytics
    is imported to prevent DNS queries, auto-installs, and update checks.
    """
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_VERBOSE"] = "False"
    _ensure_application_config_dir()


def enforce_offline_isolation() -> Dict[str, Any]:
    """
    Applies comprehensive telemetry suppression and offline isolation controls.
    Can be called repeatedly (idempotent).

    Returns:
        Dict[str, Any]: Verification status of all applied controls.

    Raises:
        RuntimeError: If critical offline controls could not be established.
    """
    global _isolation_established

    # 1. Apply environment controls before or during runtime
    configure_offline_environment()

    status: Dict[str, Any] = {
        "env_yolo_offline": os.environ.get("YOLO_OFFLINE") == "True",
        "env_yolo_autoinstall": os.environ.get("YOLO_AUTOINSTALL") == "False",
        "env_yolo_verbose": os.environ.get("YOLO_VERBOSE") == "False",
        "config_dir_isolated": False,
        "settings_sync_disabled": False,
        "events_disabled": False,
        "online_flag_false": False,
        "autoinstall_flag_false": False,
    }

    # 2. Strict Pre-Import Preflight Validation:
    # Must succeed BEFORE importing ultralytics. If it fails, raises RuntimeError
    # and aborts before ultralytics ever enters sys.modules.
    if not any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules):
        validate_isolated_config_dir_preflight()

    # 3. Configure Ultralytics settings and runtime singletons if installed.
    # Ultralytics 8.4.182 SETTINGS is a JSONDict subclass whose setters persist to disk.
    try:
        import ultralytics
        from ultralytics import SETTINGS
        import ultralytics.utils as ultralytics_utils

        if not isinstance(SETTINGS, dict):
            raise RuntimeError("Installed Ultralytics settings do not support safe process-local updates.")

        expected_settings = _ultralytics_config_dir / "settings.json"
        status["config_dir_isolated"] = (
            Path(os.environ.get("YOLO_CONFIG_DIR", "")).resolve()
            == Path(_ultralytics_config_owner.name).resolve()
            and Path(SETTINGS.file).resolve() == expected_settings.resolve()
            and Path(ultralytics_utils.USER_CONFIG_DIR).resolve() == expected_settings.parent.resolve()
        )
        if not status["config_dir_isolated"]:
            raise RuntimeError("Ultralytics selected an unexpected settings directory; refusing to continue.")

        # Bypass JSONDict.__setitem__ so this process does not rewrite user settings.
        dict.__setitem__(SETTINGS, "sync", False)
        status["settings_sync_disabled"] = (SETTINGS.get("sync") is False)

        # These modules import constants by value, so update references made earlier.
        try:
            import ultralytics.utils.checks as checks
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Could not configure Ultralytics update and auto-install controls.") from exc
        checks.ONLINE = False
        checks.AUTOINSTALL = False
        ultralytics_utils.ONLINE = False
        ultralytics_utils.AUTOINSTALL = False
        status["online_flag_false"] = (ultralytics_utils.ONLINE is False)
        status["autoinstall_flag_false"] = (ultralytics_utils.AUTOINSTALL is False)

        # Disable Events analytics dispatcher
        try:
            from ultralytics.utils.events import events
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Could not disable Ultralytics events telemetry.") from exc
        events.enabled = False
        status["events_disabled"] = (events.enabled is False)

        _isolation_established = True

    except ImportError as exc:
        raise RuntimeError("Ultralytics is required to verify offline isolation controls.") from exc

    # Validate that critical controls are active
    if not all((
        status["env_yolo_offline"],
        status["env_yolo_autoinstall"],
        status["env_yolo_verbose"],
        status["config_dir_isolated"],
        status["settings_sync_disabled"],
        status["events_disabled"],
        status["online_flag_false"],
        status["autoinstall_flag_false"],
    )):
        raise RuntimeError(
            "Failed to establish the isolated Ultralytics offline configuration."
        )

    return status


def reset_offline_configuration() -> None:
    """Reset the isolated configuration state (primarily for testing and process cleanup)."""
    global _ultralytics_config_owner, _ultralytics_config_dir, _isolation_established
    if _ultralytics_config_owner is not None:
        try:
            _ultralytics_config_owner.cleanup()
        except OSError:
            pass
    _ultralytics_config_owner = None
    _ultralytics_config_dir = None
    _isolation_established = False
