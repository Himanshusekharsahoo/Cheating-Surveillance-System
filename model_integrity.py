"""
Cryptographic Model Integrity and Supply-Chain Protection Module.

Provides cryptographic verification (SHA-256 and byte-size checking) for all
machine learning model assets loaded by the Cheating Surveillance System.

Trust Limitations:
    Cryptographic hashes and expected sizes are embedded directly in this application
    source code. This mechanism protects against supply-chain tampering, download
    corruption, accidental file substitution, and unauthorized local modifications by
    unprivileged users.
    However, in an adversarial environment where an attacker possesses write permissions
    to the Python source files or the Python runtime environment itself, the attacker
    could modify this registry alongside the model files. Full host-level non-repudiation
    requires external code signing, OS-level read-only access controls, or hardware TPM
    enclave validation.
"""

from contextlib import contextmanager
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union


class ModelIntegrityError(ValueError):
    """Base exception for model integrity, verification, or supply-chain defects."""
    pass


class ModelMissingError(ModelIntegrityError, FileNotFoundError):
    """Raised when a required model asset does not exist on disk."""
    pass


class ModelCorruptedError(ModelIntegrityError):
    """Raised when a model asset has an unexpected size, corrupted bytes, or hash mismatch."""
    pass


class ModelUnregisteredError(ModelIntegrityError):
    """Raised when an unapproved model asset not present in the integrity registry is requested."""
    pass


# Canonical production registry of independently verified model assets
DEFAULT_MODEL_INTEGRITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "best_yolov12.pt": {
        "sha256": "c9edee36f87fb8d450c1608f9e2bd88a7c3f16e5351c1cbdc7079de5ff470d2a",
        "size_bytes": 18930963,
        "model_type": "yolo",
    },
    "shape_predictor_68_face_landmarks.dat": {
        "sha256": "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f",
        "size_bytes": 99693937,
        "model_type": "dlib_shape_predictor",
    },
}

# Active registry for runtime queries (can be scoped via dependency injection or temporary_model_registry in tests)
_active_registry: Dict[str, Dict[str, Any]] = dict(DEFAULT_MODEL_INTEGRITY_REGISTRY)


def get_active_model_registry() -> Dict[str, Dict[str, Any]]:
    """Returns a copy of the active model integrity registry."""
    return dict(_active_registry)


def register_test_model(
    filename: str,
    sha256: str,
    size_bytes: int,
    model_type: str = "test",
) -> None:
    """
    Registers a temporary model entry in the active registry.
    Intended for unit testing and dependency injection.
    """
    _active_registry[filename] = {
        "sha256": sha256.lower(),
        "size_bytes": int(size_bytes),
        "model_type": model_type,
    }


def unregister_test_model(filename: str) -> None:
    """Removes a model entry from the active registry if present."""
    _active_registry.pop(filename, None)


def reset_model_registry() -> None:
    """Resets the active registry back to the default production registry."""
    global _active_registry
    _active_registry = dict(DEFAULT_MODEL_INTEGRITY_REGISTRY)


@contextmanager
def temporary_model_registry(
    temp_registry: Optional[Dict[str, Dict[str, Any]]] = None
) -> Iterator[Dict[str, Dict[str, Any]]]:
    """
    Context manager to safely inject a temporary model registry for tests.
    Restores the previous registry state upon exit.
    """
    global _active_registry
    previous_registry = dict(_active_registry)
    try:
        if temp_registry is not None:
            _active_registry = dict(temp_registry)
        yield _active_registry
    finally:
        _active_registry = previous_registry


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Streams a file from disk in bounded chunks and computes its SHA-256 hexadecimal digest.

    Args:
        file_path: Path to the target file.
        chunk_size: Streaming chunk size in bytes (default: 64 KB).

    Returns:
        str: Lowercase 64-character SHA-256 hex digest.

    Raises:
        ModelMissingError: If file does not exist.
        ModelIntegrityError: If file cannot be read due to permission or I/O error.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ModelMissingError(f"Model file not found: '{path.name}'")

    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
    except (PermissionError, OSError) as e:
        raise ModelIntegrityError(f"Failed to read model file '{path.name}': {e}")

    return hasher.hexdigest().lower()


def verify_model_asset(
    model_path: Union[str, Path],
    registry: Optional[Dict[str, Dict[str, Any]]] = None,
    chunk_size: int = 65536,
) -> Path:
    """
    Cryptographically verifies a model file against the integrity registry.
    Checks existence, readability, file size, and SHA-256 checksum.

    Args:
        model_path: Path to the candidate model file.
        registry: Optional custom registry dictionary for dependency injection.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        Path: Resolved Path object to the verified model.

    Raises:
        ModelMissingError: If model file does not exist.
        ModelUnregisteredError: If model filename is not in the trusted registry.
        ModelCorruptedError: If file size or SHA-256 hash does not match registry.
        ModelIntegrityError: If file cannot be read due to filesystem/permission errors.
    """
    path = Path(model_path)
    filename = path.name

    active_reg = registry if registry is not None else _active_registry

    # 1. Validate file existence first
    if not path.is_file():
        if "landmark" in filename or "shape_predictor" in filename:
            desc = "facial landmark model"
        elif "yolo" in filename:
            desc = "YOLO model"
        else:
            desc = "model asset"

        raise ModelMissingError(
            f"Missing required {desc} at '{path}'. "
            f"Please ensure '{filename}' exists."
        )

    # 2. Validate registration in trusted registry
    if filename not in active_reg:
        raise ModelUnregisteredError(
            f"Model asset '{filename}' is not registered in the trusted model integrity registry. "
            f"Loading unverified model assets is prohibited."
        )

    expected_spec = active_reg[filename]
    expected_sha256 = expected_spec["sha256"].lower()
    expected_size = expected_spec["size_bytes"]

    # Validate file size
    try:
        actual_size = path.stat().st_size
    except (PermissionError, OSError) as e:
        raise ModelIntegrityError(f"Cannot inspect model asset '{filename}': {e}")

    if actual_size != expected_size:
        raise ModelCorruptedError(
            f"Model asset '{filename}' size mismatch. Expected {expected_size} bytes, "
            f"got {actual_size} bytes. The model file may be truncated or modified."
        )

    # Compute and verify SHA-256 hash using streaming and constant-time comparison
    actual_sha256 = compute_file_sha256(path, chunk_size=chunk_size)

    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ModelCorruptedError(
            f"Model asset '{filename}' integrity check failed: SHA-256 mismatch. "
            f"Expected: {expected_sha256}, got: {actual_sha256}. "
            f"The model file has been modified, corrupted, or replaced."
        )

    return path


def reset_model_singletons() -> None:
    """
    Safely resets cached model singletons across all detection modules.
    Used during test isolation and error recovery.
    """
    try:
        import mobile_detection
        mobile_detection._model = None
    except ImportError:
        pass

    try:
        import eye_movement
        eye_movement._predictor = None
        eye_movement._detector = None
    except ImportError:
        pass

    try:
        import head_pose
        head_pose._predictor = None
        head_pose._detector = None
    except ImportError:
        pass
