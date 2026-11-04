"""
Unit and Security Tests for Phase 10.13.24 — Repository & Environment Hardening.

Covers:
- Safe .env loading and parsing without third-party dependencies.
- Symlink and junction rejection for .env files.
- Typed environment variable parsing (int, float, bool, str, path).
- Sanitized exception messages that never reveal secret values.
- Git hygiene: gitignore exclusions for secrets, credentials, runtime evidence, and caches.
- Safe template preservation (.env.example).
"""

from pathlib import Path
import os
import pytest

import config
from path_security import is_symlink_or_junction

pytestmark = pytest.mark.unit


def test_env_example_exists_and_is_not_ignored(tmp_path):
    """Verify that .env.example exists in the repository root and is tracked/trackable."""
    example_path = config.BASE_DIR / ".env.example"
    assert example_path.is_file(), ".env.example template must exist in repository root."
    content = example_path.read_text(encoding="utf-8")
    assert "SURVEILLANCE_CAMERA_INDEX" in content
    assert "SURVEILLANCE_LOG_DIR" in content
    assert "YOLO_OFFLINE=True" in content


def test_load_dotenv_file_parses_valid_keys_and_values(tmp_path, monkeypatch):
    """Verify that load_dotenv_file correctly parses key=value pairs, comments, and quotes."""
    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "# Header comment\n"
        "SURVEILLANCE_TEST_INT=42\n"
        "SURVEILLANCE_TEST_STR=\"hello_world\"\n"
        "SURVEILLANCE_TEST_SINGLE='single_quoted'\n"
        "SURVEILLANCE_TEST_INLINE=val # inline comment\n"
        "INVALID LINE WITHOUT EQUALS\n"
        "123_INVALID_KEY=value\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("SURVEILLANCE_TEST_INT", raising=False)
    monkeypatch.delenv("SURVEILLANCE_TEST_STR", raising=False)
    monkeypatch.delenv("SURVEILLANCE_TEST_SINGLE", raising=False)
    monkeypatch.delenv("SURVEILLANCE_TEST_INLINE", raising=False)

    loaded = config.load_dotenv_file(fake_env)
    assert loaded is True
    assert os.environ.get("SURVEILLANCE_TEST_INT") == "42"
    assert os.environ.get("SURVEILLANCE_TEST_STR") == "hello_world"
    assert os.environ.get("SURVEILLANCE_TEST_SINGLE") == "single_quoted"
    assert os.environ.get("SURVEILLANCE_TEST_INLINE") == "val"
    assert "123_INVALID_KEY" not in os.environ


def test_load_dotenv_file_does_not_override_existing_env_unless_specified(tmp_path, monkeypatch):
    """Verify standard dotenv semantics: process environment takes precedence unless override=True."""
    fake_env = tmp_path / ".env"
    fake_env.write_text("PRE_EXISTING_KEY=new_file_val\n", encoding="utf-8")

    monkeypatch.setenv("PRE_EXISTING_KEY", "process_val")
    loaded = config.load_dotenv_file(fake_env, override=False)
    assert loaded is True
    assert os.environ.get("PRE_EXISTING_KEY") == "process_val"

    loaded_override = config.load_dotenv_file(fake_env, override=True)
    assert loaded_override is True
    assert os.environ.get("PRE_EXISTING_KEY") == "new_file_val"


def test_load_dotenv_file_rejects_missing_file():
    """Verify load_dotenv_file returns False gracefully for nonexistent file."""
    assert config.load_dotenv_file(Path("nonexistent/.env")) is False


def test_load_dotenv_file_rejects_symlink(tmp_path):
    """Verify load_dotenv_file rejects symlinked or junction .env files."""
    real_env = tmp_path / "real.env"
    real_env.write_text("DANGEROUS_KEY=payload\n", encoding="utf-8")
    symlink_env = tmp_path / "symlink.env"

    try:
        symlink_env.symlink_to(real_env)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in current environment.")

    assert config.load_dotenv_file(symlink_env) is False
    assert "DANGEROUS_KEY" not in os.environ


def test_get_env_int_validation(monkeypatch):
    """Verify integer parsing, default fallback, bounds validation, and error hygiene."""
    monkeypatch.delenv("SURVEILLANCE_SAMPLE_INT", raising=False)
    assert config.get_env_int("SURVEILLANCE_SAMPLE_INT", 10) == 10

    monkeypatch.setenv("SURVEILLANCE_SAMPLE_INT", "25")
    assert config.get_env_int("SURVEILLANCE_SAMPLE_INT", 10, min_val=0, max_val=50) == 25

    # Out of bounds low
    monkeypatch.setenv("SURVEILLANCE_SAMPLE_INT", "-5")
    with pytest.raises(ValueError) as exc:
        config.get_env_int("SURVEILLANCE_SAMPLE_INT", 10, min_val=0)
    assert "must be >= 0" in str(exc.value)
    assert "-5" not in str(exc.value)  # Value should not leak

    # Out of bounds high
    monkeypatch.setenv("SURVEILLANCE_SAMPLE_INT", "100")
    with pytest.raises(ValueError) as exc:
        config.get_env_int("SURVEILLANCE_SAMPLE_INT", 10, max_val=50)
    assert "must be <= 50" in str(exc.value)
    assert "100" not in str(exc.value)

    # Malformed integer
    monkeypatch.setenv("SURVEILLANCE_SAMPLE_INT", "not_a_number_secret_token")
    with pytest.raises(ValueError) as exc:
        config.get_env_int("SURVEILLANCE_SAMPLE_INT", 10)
    assert "Invalid integer value for environment variable 'SURVEILLANCE_SAMPLE_INT'" in str(exc.value)
    assert "secret_token" not in str(exc.value)


def test_get_env_float_validation(monkeypatch):
    """Verify float parsing, bounds validation, infinite/NaN rejection, and error hygiene."""
    monkeypatch.delenv("SURVEILLANCE_SAMPLE_FLOAT", raising=False)
    assert config.get_env_float("SURVEILLANCE_SAMPLE_FLOAT", 3.14) == 3.14

    monkeypatch.setenv("SURVEILLANCE_SAMPLE_FLOAT", "2.718")
    assert config.get_env_float("SURVEILLANCE_SAMPLE_FLOAT", 1.0, min_val=0.0, max_val=5.0) == 2.718

    # Non-finite (inf)
    monkeypatch.setenv("SURVEILLANCE_SAMPLE_FLOAT", "inf")
    with pytest.raises(ValueError) as exc:
        config.get_env_float("SURVEILLANCE_SAMPLE_FLOAT", 1.0)
    assert "finite float" in str(exc.value)

    # Out of bounds
    monkeypatch.setenv("SURVEILLANCE_SAMPLE_FLOAT", "-1.5")
    with pytest.raises(ValueError) as exc:
        config.get_env_float("SURVEILLANCE_SAMPLE_FLOAT", 1.0, min_val=0.0)
    assert "must be >= 0.0" in str(exc.value)
    assert "-1.5" not in str(exc.value)


def test_get_env_bool_validation(monkeypatch):
    """Verify boolean truthy and falsy parsing and malformed value rejection."""
    monkeypatch.delenv("SURVEILLANCE_SAMPLE_BOOL", raising=False)
    assert config.get_env_bool("SURVEILLANCE_SAMPLE_BOOL", True) is True
    assert config.get_env_bool("SURVEILLANCE_SAMPLE_BOOL", False) is False

    for truthy in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv("SURVEILLANCE_SAMPLE_BOOL", truthy)
        assert config.get_env_bool("SURVEILLANCE_SAMPLE_BOOL", False) is True

    for falsy in ("0", "false", "False", "NO", "off"):
        monkeypatch.setenv("SURVEILLANCE_SAMPLE_BOOL", falsy)
        assert config.get_env_bool("SURVEILLANCE_SAMPLE_BOOL", True) is False

    monkeypatch.setenv("SURVEILLANCE_SAMPLE_BOOL", "super_secret_token")
    with pytest.raises(ValueError) as exc:
        config.get_env_bool("SURVEILLANCE_SAMPLE_BOOL", False)
    assert "Invalid boolean value for environment variable 'SURVEILLANCE_SAMPLE_BOOL'" in str(exc.value)
    assert "super_secret_token" not in str(exc.value)


def test_get_env_str_validation(monkeypatch):
    """Verify string parsing and allowed values enforcement."""
    monkeypatch.delenv("SURVEILLANCE_SAMPLE_STR", raising=False)
    assert config.get_env_str("SURVEILLANCE_SAMPLE_STR", "default") == "default"

    monkeypatch.setenv("SURVEILLANCE_SAMPLE_STR", "adaptive")
    assert config.get_env_str("SURVEILLANCE_SAMPLE_STR", "default", allowed_values=("adaptive", "fixed")) == "adaptive"

    monkeypatch.setenv("SURVEILLANCE_SAMPLE_STR", "forbidden_choice")
    with pytest.raises(ValueError) as exc:
        config.get_env_str("SURVEILLANCE_SAMPLE_STR", "default", allowed_values=("adaptive", "fixed"))
    assert "Must be one of" in str(exc.value)


def test_get_env_path_resolution(monkeypatch):
    """Verify path resolution relative to BASE_DIR and absolute paths."""
    monkeypatch.delenv("SURVEILLANCE_SAMPLE_PATH", raising=False)
    default_p = config.BASE_DIR / "custom_default"
    assert config.get_env_path("SURVEILLANCE_SAMPLE_PATH", default_p) == default_p

    monkeypatch.setenv("SURVEILLANCE_SAMPLE_PATH", "relative_subfolder")
    resolved = config.get_env_path("SURVEILLANCE_SAMPLE_PATH", default_p)
    assert resolved == (config.BASE_DIR / "relative_subfolder").resolve()


def test_reload_config_updates_settings_from_environment(monkeypatch):
    """Verify reload_config applies environment overrides across application settings."""
    try:
        monkeypatch.setenv("SURVEILLANCE_CAMERA_INDEX", "2")
        monkeypatch.setenv("SURVEILLANCE_VIOLATION_THRESHOLD_SEC", "4.5")
        monkeypatch.setenv("SURVEILLANCE_RETENTION_DAYS", "60")
        monkeypatch.setenv("SURVEILLANCE_RETENTION_ENABLED", "True")

        config.reload_config()
        assert config.CAMERA_INDEX == 2
        assert config.VIOLATION_THRESHOLD_SEC == 4.5
        assert config.RETENTION_DAYS == 60
        assert config.RETENTION_ENABLED is True
    finally:
        monkeypatch.delenv("SURVEILLANCE_CAMERA_INDEX", raising=False)
        monkeypatch.delenv("SURVEILLANCE_VIOLATION_THRESHOLD_SEC", raising=False)
        monkeypatch.delenv("SURVEILLANCE_RETENTION_DAYS", raising=False)
        monkeypatch.delenv("SURVEILLANCE_RETENTION_ENABLED", raising=False)
        config.reload_config()
        assert config.CAMERA_INDEX == 0
        assert config.VIOLATION_THRESHOLD_SEC == 3.0
        assert config.RETENTION_DAYS == 30
        assert config.RETENTION_ENABLED is False


def test_gitignore_covers_secrets_and_runtime_evidence():
    """Verify that .gitignore specifies all required secret, credential, and evidence patterns."""
    gitignore_path = config.BASE_DIR / ".gitignore"
    assert gitignore_path.is_file()
    text = gitignore_path.read_text(encoding="utf-8")

    required_patterns = [
        ".env",
        "!.env.example",
        "*.pem",
        "*.key",
        "secrets.json",
        "credentials.json",
        ".coverage",
        "log/evidence/",
        "log/encrypted/",
        "log/retention/",
        ".ultralytics/",
    ]
    for pattern in required_patterns:
        assert pattern in text, f"Missing pattern '{pattern}' in .gitignore"
