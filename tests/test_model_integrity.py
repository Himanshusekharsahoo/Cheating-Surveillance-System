"""
Phase 10.13.21 — Cryptographic Model Integrity and Supply-Chain Protection Tests.

Covers:
1. Valid model hash verification.
2. Modified model (hash mismatch) rejection.
3. Incorrect file size rejection.
4. Missing model file rejection (fails closed).
5. Unreadable model file (permission/IO error) handling.
6. Corrupted model bytes rejection.
7. Model-loading prevention after failed verification (no execution/deserialization).
8. Both YOLO and Dlib verification paths fail closed.
9. Temporary test models through dependency injection (registry scoping).
10. Existing mock-based functional test compatibility and singleton resource reset.
"""

import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import config
import model_integrity
from model_integrity import (
    DEFAULT_MODEL_INTEGRITY_REGISTRY,
    ModelIntegrityError,
    ModelMissingError,
    ModelCorruptedError,
    ModelUnregisteredError,
    compute_file_sha256,
    verify_model_asset,
    register_test_model,
    reset_model_registry,
    temporary_model_registry,
    reset_model_singletons,
)
import mobile_detection
import eye_movement
import head_pose

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_registry_and_singletons():
    """Ensure registry and model singletons are pristine before and after each test."""
    reset_model_registry()
    reset_model_singletons()
    yield
    reset_model_registry()
    reset_model_singletons()


def _make_dummy_model(tmp_path: Path, filename: str, content: bytes) -> Path:
    """Helper to create a dummy model file with specific content."""
    model_file = tmp_path / filename
    model_file.write_bytes(content)
    return model_file


# ---------------------------------------------------------------------------
# Test 1: Valid Model Hash Verification
# ---------------------------------------------------------------------------
def test_valid_model_hash_verification(tmp_path):
    """Verify that a valid model file with matching size and SHA-256 passes verification."""
    data = b"VALID_MODEL_WEIGHTS_FOR_TESTING"
    model_file = _make_dummy_model(tmp_path, "test_model.pt", data)
    expected_hash = hashlib.sha256(data).hexdigest()

    register_test_model("test_model.pt", expected_hash, len(data), "yolo")
    verified_path = verify_model_asset(model_file)
    assert verified_path == model_file


def test_production_models_match_registry():
    """Verify that genuine production models on disk strictly match the integrity registry."""
    # Verify shape predictor
    if config.SHAPE_PREDICTOR_PATH.is_file():
        verified_pred = verify_model_asset(config.SHAPE_PREDICTOR_PATH)
        assert verified_pred == config.SHAPE_PREDICTOR_PATH
        spec = DEFAULT_MODEL_INTEGRITY_REGISTRY[config.SHAPE_PREDICTOR_FILENAME]
        assert config.SHAPE_PREDICTOR_PATH.stat().st_size == spec["size_bytes"]

    # Verify YOLO model
    if config.YOLO_MODEL_PATH.is_file():
        verified_yolo = verify_model_asset(config.YOLO_MODEL_PATH)
        assert verified_yolo == config.YOLO_MODEL_PATH
        spec = DEFAULT_MODEL_INTEGRITY_REGISTRY[config.YOLO_MODEL_FILENAME]
        assert config.YOLO_MODEL_PATH.stat().st_size == spec["size_bytes"]


# ---------------------------------------------------------------------------
# Test 2: Modified Model (Hash Mismatch)
# ---------------------------------------------------------------------------
def test_modified_model_rejected_with_hash_mismatch(tmp_path):
    """Verify that a model with identical byte-size but altered content is rejected."""
    original_data = b"AUTHENTIC_MODEL_WEIGHTS_12345"
    tampered_data = b"MALICIOUS_MODEL_WEIGHTS_12345"
    assert len(original_data) == len(tampered_data)

    model_file = _make_dummy_model(tmp_path, "tampered.pt", tampered_data)
    expected_hash = hashlib.sha256(original_data).hexdigest()

    register_test_model("tampered.pt", expected_hash, len(original_data), "yolo")

    with pytest.raises(ModelCorruptedError, match="SHA-256 mismatch"):
        verify_model_asset(model_file)


# ---------------------------------------------------------------------------
# Test 3: Incorrect File Size
# ---------------------------------------------------------------------------
def test_incorrect_file_size_rejected_immediately(tmp_path):
    """Verify that a model with mismatched file size is rejected fail-closed without hashing."""
    data = b"SHORT"
    model_file = _make_dummy_model(tmp_path, "wrong_size.pt", data)

    # Register expecting 100 bytes, but actual is 5 bytes
    register_test_model("wrong_size.pt", "0" * 64, 100, "yolo")

    with pytest.raises(ModelCorruptedError, match="size mismatch"):
        verify_model_asset(model_file)


# ---------------------------------------------------------------------------
# Test 4: Missing Model File (Fail-Closed)
# ---------------------------------------------------------------------------
def test_missing_model_file_fails_closed(tmp_path):
    """Verify that a missing model file raises ModelMissingError (subclass of FileNotFoundError)."""
    missing_file = tmp_path / "missing_model.pt"
    register_test_model("missing_model.pt", "a" * 64, 1024, "yolo")

    with pytest.raises(ModelMissingError) as exc_info:
        verify_model_asset(missing_file)

    # Verify both inheritance branches
    assert isinstance(exc_info.value, ModelIntegrityError)
    assert isinstance(exc_info.value, FileNotFoundError)
    assert "missing" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 5: Unreadable Model File
# ---------------------------------------------------------------------------
def test_unreadable_model_file_handled_safely(tmp_path):
    """Verify that filesystem permission or I/O errors are safely wrapped in ModelIntegrityError."""
    model_file = _make_dummy_model(tmp_path, "unreadable.pt", b"SOME_DATA")
    register_test_model("unreadable.pt", "b" * 64, 9, "yolo")

    with patch("builtins.open", side_effect=PermissionError("Access Denied")):
        with pytest.raises(ModelIntegrityError, match="Failed to read model file"):
            compute_file_sha256(model_file)


# ---------------------------------------------------------------------------
# Test 6: Corrupted Model Bytes
# ---------------------------------------------------------------------------
def test_corrupted_model_bytes_rejected(tmp_path):
    """Verify that corrupted model bytes trigger ModelCorruptedError."""
    corrupted_file = _make_dummy_model(tmp_path, "corrupted.pt", b"\x00\x00\x00\x00\x00")
    register_test_model("corrupted.pt", "c" * 64, 5, "yolo")

    with pytest.raises(ModelCorruptedError, match="SHA-256 mismatch"):
        verify_model_asset(corrupted_file)


# ---------------------------------------------------------------------------
# Test 7: Model-Loading Prevention After Failed Verification
# ---------------------------------------------------------------------------
def test_yolo_loading_prevented_on_failed_verification(tmp_path):
    """Verify that YOLO() is never invoked if cryptographic verification fails."""
    tampered_file = _make_dummy_model(tmp_path, "weaponized.pt", b"PAYLOAD")
    register_test_model("weaponized.pt", "f" * 64, 7, "yolo")

    mock_yolo = MagicMock()
    with patch("mobile_detection.enforce_offline_isolation"), \
         patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=mock_yolo)}):
        with pytest.raises(ModelIntegrityError):
            mobile_detection.get_model(tampered_file)

        # Assert ultralytics YOLO was NEVER called with the unverified file
        mock_yolo.assert_not_called()
        assert mobile_detection._model is None


def test_dlib_loading_prevented_on_failed_verification(tmp_path):
    """Verify that dlib.shape_predictor() is never invoked if verification fails."""
    tampered_file = _make_dummy_model(tmp_path, "tampered_landmarks.dat", b"CORRUPT")
    register_test_model("tampered_landmarks.dat", "e" * 64, 7, "dlib_shape_predictor")

    mock_dlib = MagicMock()
    with patch.dict(sys.modules, {"dlib": MagicMock(shape_predictor=mock_dlib)}):
        with pytest.raises(ModelIntegrityError):
            eye_movement.get_shape_predictor(tampered_file)

        mock_dlib.assert_not_called()
        assert eye_movement._predictor is None


# ---------------------------------------------------------------------------
# Test 8: Both YOLO and Dlib Verification Paths
# ---------------------------------------------------------------------------
def test_both_yolo_and_dlib_verification_paths_fail_closed(tmp_path):
    """Verify that both eye_movement and head_pose shape predictor loaders fail closed."""
    tampered_file = _make_dummy_model(tmp_path, "bad_head_pred.dat", b"BAD_DATA")
    register_test_model("bad_head_pred.dat", "d" * 64, 8, "dlib_shape_predictor")

    # eye_movement loader
    with pytest.raises(ModelCorruptedError):
        eye_movement.get_shape_predictor(tampered_file)

    # head_pose loader
    with pytest.raises(ModelCorruptedError):
        head_pose.get_shape_predictor(tampered_file)


# ---------------------------------------------------------------------------
# Test 9: Temporary Test Models Through Dependency Injection
# ---------------------------------------------------------------------------
def test_temporary_model_registry_context_manager(tmp_path):
    """Verify that temporary_model_registry scopes test models and restores production state."""
    test_content = b"INJECTED_TEST_MODEL_DATA"
    test_file = _make_dummy_model(tmp_path, "injected.pt", test_content)
    test_hash = hashlib.sha256(test_content).hexdigest()

    temp_reg = {
        "injected.pt": {
            "sha256": test_hash,
            "size_bytes": len(test_content),
            "model_type": "yolo",
        }
    }

    # Within context manager: injected model verifies cleanly
    with temporary_model_registry(temp_reg):
        verified = verify_model_asset(test_file)
        assert verified == test_file
        # Production model should not be in temp_reg
        with pytest.raises(ModelUnregisteredError):
            verify_model_asset(config.YOLO_MODEL_PATH)

    # Outside context manager: production registry is restored
    with pytest.raises(ModelUnregisteredError):
        verify_model_asset(test_file)


# ---------------------------------------------------------------------------
# Test 10: Existing Mock-Based Functional Test Compatibility & Resource Reset
# ---------------------------------------------------------------------------
def test_reset_model_singletons_clears_all_caches():
    """Verify that reset_model_singletons safely clears singletons across modules."""
    mobile_detection._model = "MOCK_YOLO"
    eye_movement._predictor = "MOCK_PRED"
    eye_movement._detector = "MOCK_DET"
    head_pose._predictor = "MOCK_HEAD_PRED"
    head_pose._detector = "MOCK_HEAD_DET"

    reset_model_singletons()

    assert mobile_detection._model is None
    assert eye_movement._predictor is None
    assert eye_movement._detector is None
    assert head_pose._predictor is None
    assert head_pose._detector is None


def test_unregistered_model_file_rejected_without_bypass(tmp_path):
    """Verify that any model asset not in registry is unconditionally rejected."""
    arbitrary_file = _make_dummy_model(tmp_path, "unregistered_custom_model.onnx", b"WEIGHTS")
    with pytest.raises(ModelUnregisteredError, match="is not registered"):
        verify_model_asset(arbitrary_file)
