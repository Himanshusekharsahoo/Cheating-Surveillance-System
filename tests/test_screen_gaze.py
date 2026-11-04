"""
Comprehensive Unit and Integration Tests for Phase 10.13.5:
Screen Gaze Calibration & Monitor Boundary Mapping.
"""

import math
from unittest.mock import MagicMock
import numpy as np
import pytest

from screen_gaze import (
    ScreenGazeResult,
    ScreenCalibrationSample,
    ScreenGazeCalibrator,
    BOUNDARY_ON_SCREEN,
    BOUNDARY_NEAR_BOUNDARY,
    BOUNDARY_OUTSIDE_SCREEN,
    BOUNDARY_UNCERTAIN,
    BOUNDARY_CALIBRATION_REQUIRED,
    CALIB_STATUS_NOT_CALIBRATED,
    CALIB_STATUS_VALID,
)
from gaze_fusion import CombinedGazeResult, HeadPoseResult, EyeGazeResult
from vision_pipeline import VisionPipeline, FaceAnalysisResult

pytestmark = pytest.mark.unit


# ===========================================================================
# 1. ScreenCalibrationSample Validation Tests
# ===========================================================================

class TestScreenCalibrationSample:
    """Verifies input validation and integrity of calibration observations."""

    def test_valid_sample_construction(self):
        sample = ScreenCalibrationSample(
            target_x=0.5,
            target_y=0.5,
            gaze_yaw=0.0,
            gaze_pitch=0.0,
            confidence=0.92,
            tracking_state="PAIRED_TRACKING",
            is_single_eye_fallback=False
        )
        assert sample.is_valid() is True
        assert sample.target_x == 0.5
        assert sample.target_y == 0.5
        assert sample.confidence == 0.92

    def test_reject_out_of_bounds_target_coordinates(self):
        """Target coordinates outside [0.0, 1.0] are rejected."""
        s_neg_x = ScreenCalibrationSample(-0.1, 0.5, 0.0, 0.0, 0.9)
        assert s_neg_x.is_valid() is False

        s_high_x = ScreenCalibrationSample(1.1, 0.5, 0.0, 0.0, 0.9)
        assert s_high_x.is_valid() is False

        s_neg_y = ScreenCalibrationSample(0.5, -0.05, 0.0, 0.0, 0.9)
        assert s_neg_y.is_valid() is False

        s_high_y = ScreenCalibrationSample(0.5, 1.05, 0.0, 0.0, 0.9)
        assert s_high_y.is_valid() is False

    def test_reject_non_finite_values(self):
        """NaN or Inf values in coordinates, gaze angles, or confidence are rejected."""
        assert ScreenCalibrationSample(np.nan, 0.5, 0.0, 0.0, 0.9).is_valid() is False
        assert ScreenCalibrationSample(0.5, np.inf, 0.0, 0.0, 0.9).is_valid() is False
        assert ScreenCalibrationSample(0.5, 0.5, np.nan, 0.0, 0.9).is_valid() is False
        assert ScreenCalibrationSample(0.5, 0.5, 0.0, np.inf, 0.9).is_valid() is False
        assert ScreenCalibrationSample(0.5, 0.5, 0.0, 0.0, np.nan).is_valid() is False

    def test_reject_low_confidence_or_uncertain_tracking(self):
        """Observations with confidence below threshold or UNCERTAIN tracking state are rejected."""
        s_low_conf = ScreenCalibrationSample(0.5, 0.5, 0.0, 0.0, 0.15)
        assert s_low_conf.is_valid(min_confidence=0.20) is False

        s_uncertain = ScreenCalibrationSample(0.5, 0.5, 0.0, 0.0, 0.85, tracking_state="UNCERTAIN")
        assert s_uncertain.is_valid() is False

        s_no_face = ScreenCalibrationSample(0.5, 0.5, 0.0, 0.0, 0.85, tracking_state="NO_FACE")
        assert s_no_face.is_valid() is False


# ===========================================================================
# 2. ScreenGazeCalibrator Mathematical Fitting Tests
# ===========================================================================

class TestScreenGazeCalibratorFitting:
    """Verifies mathematical calibration fitting on synthetic ground-truth data."""

    @staticmethod
    def _generate_synthetic_calibration_data(
        targets=None,
        samples_per_target=4,
        scale_x=0.035,   # degrees per normalized screen unit
        scale_y=-0.040,  # degrees per normalized screen unit (pitch up is positive)
        bias_x=0.50,
        bias_y=0.50,
        noise_std=0.005,
    ):
        """
        Synthesizes realistic gaze observations for given monitor targets:
          yaw = (target_x - bias_x) / scale_x
          pitch = (target_y - bias_y) / scale_y
        """
        if targets is None:
            # 5-point calibration: center + 4 corners
            targets = [
                (0.50, 0.50),  # Center
                (0.10, 0.10),  # Top-Left
                (0.90, 0.10),  # Top-Right
                (0.10, 0.90),  # Bottom-Left
                (0.90, 0.90),  # Bottom-Right
            ]

        samples = []
        for tx, ty in targets:
            true_yaw = (tx - bias_x) / scale_x
            true_pitch = (ty - bias_y) / scale_y

            for k in range(samples_per_target):
                jitter_y = np.random.normal(0, noise_std / abs(scale_x)) if noise_std > 0 else 0.0
                jitter_p = np.random.normal(0, noise_std / abs(scale_y)) if noise_std > 0 else 0.0
                samples.append(ScreenCalibrationSample(
                    target_x=tx,
                    target_y=ty,
                    gaze_yaw=true_yaw + jitter_y,
                    gaze_pitch=true_pitch + jitter_p,
                    confidence=0.90,
                    tracking_state="PAIRED_TRACKING"
                ))
        return samples

    def test_ideal_5_point_calibration_recovers_affine_mapping(self):
        """Ideal zero-noise 5-point calibration recovers exact target coordinates (< 0.001 RMSE)."""
        np.random.seed(42)
        samples = self._generate_synthetic_calibration_data(samples_per_target=5, noise_std=0.0)

        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=3)
        for s in samples:
            assert calibrator.add_sample(s) is True

        assert calibrator.fit() is True
        assert calibrator.is_calibrated is True

        metrics = calibrator.quality_metrics
        assert metrics["success"] is True
        assert metrics["rmse"] < 0.005
        assert metrics["quality_score"] > 0.90
        assert metrics["target_count"] == 5

        # Test recovery of center screen: yaw = 0, pitch = 0 -> (0.50, 0.50)
        res_center = calibrator.estimate_screen_gaze(gaze_yaw=0.0, gaze_pitch=0.0, gaze_confidence=0.95)
        assert res_center.status == "VALID"
        assert res_center.boundary_state == BOUNDARY_ON_SCREEN
        assert abs(res_center.screen_x - 0.50) < 0.01
        assert abs(res_center.screen_y - 0.50) < 0.01

    def test_9_point_grid_calibration(self):
        """9-point calibration grid fits successfully with low residual error."""
        targets_9 = [
            (x, y) for x in (0.15, 0.50, 0.85) for y in (0.15, 0.50, 0.85)
        ]
        samples = self._generate_synthetic_calibration_data(targets=targets_9, samples_per_target=3, noise_std=0.002)

        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=2)
        for s in samples:
            calibrator.add_sample(s)

        assert calibrator.fit() is True
        assert calibrator.quality_metrics["target_count"] == 9
        assert calibrator.quality_metrics["rmse"] < 0.02

    def test_single_eye_fallback_samples_receive_reduced_weight(self):
        """Single-eye fallback samples are accepted with reduced weight during fitting."""
        samples = self._generate_synthetic_calibration_data(samples_per_target=4, noise_std=0.0)
        # Mark half the samples as single-eye fallback
        for i in range(0, len(samples), 2):
            samples[i].is_single_eye_fallback = True
            samples[i].confidence = 0.70

        calibrator = ScreenGazeCalibrator()
        for s in samples:
            calibrator.add_sample(s)

        assert calibrator.fit() is True
        assert calibrator.is_calibrated is True


# ===========================================================================
# 3. Degenerate Inputs & Edge Cases Rejection Tests
# ===========================================================================

class TestScreenCalibrationDegenerateInputs:
    """Verifies that uncalibrated, degenerate, or ill-conditioned data safely fails calibration."""

    def test_insufficient_sample_count_fails(self):
        """Fewer samples than (min_targets * min_samples_per_target) fails calibration."""
        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=3)  # requires 12
        for _ in range(5):
            calibrator.add_observation(0.5, 0.5, 0.0, 0.0, 0.90)

        assert calibrator.fit() is False
        assert calibrator.is_calibrated is False
        assert "insufficient_samples" in calibrator.quality_metrics["reason"]

    def test_repeated_single_target_fails(self):
        """All samples at the same target point fails for lack of distinct targets."""
        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=3)
        for _ in range(20):
            # All observations at (0.5, 0.5)
            calibrator.add_observation(0.5, 0.5, 0.0, 0.0, 0.90)

        assert calibrator.fit() is False
        assert "insufficient_distinct_targets" in calibrator.quality_metrics["reason"]

    def test_poor_spatial_spread_fails(self):
        """Targets clustered tightly within small screen region (< 0.30 span) are rejected."""
        clustered_targets = [
            (0.44, 0.44), (0.56, 0.44), (0.44, 0.56), (0.56, 0.56)
        ]
        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=3)
        for tx, ty in clustered_targets:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 20, (ty - 0.5) * 20, 0.90)

        assert calibrator.fit() is False
        assert "poor_spatial_spread" in calibrator.quality_metrics["reason"]

    def test_zero_variance_gaze_features_fails(self):
        """Targets spread across screen but identical gaze measurements fails for zero variance."""
        targets = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=3)
        for tx, ty in targets:
            for _ in range(3):
                # Gaze angles completely static (e.g. frozen sensor or stub)
                calibrator.add_observation(tx, ty, 0.0, 0.0, 0.90)

        assert calibrator.fit() is False
        assert "insufficient_gaze_feature_variance" in calibrator.quality_metrics["reason"]

    def test_excessive_residual_error_fails(self):
        """Completely random uncorrelated gaze observations fail max RMSE check."""
        np.random.seed(99)
        targets = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.5)]
        calibrator = ScreenGazeCalibrator(min_targets=4, min_samples_per_target=3, max_rmse_threshold=0.15)
        for tx, ty in targets:
            for _ in range(4):
                # Random random uncorrelated angles
                calibrator.add_observation(tx, ty, np.random.uniform(-30, 30), np.random.uniform(-30, 30), 0.90)

        assert calibrator.fit() is False
        assert "excessive_residual_error" in calibrator.quality_metrics["reason"]


# ===========================================================================
# 4. Monitor Boundary Mapping Tests
# ===========================================================================

class TestMonitorBoundaryMapping:
    """Verifies screen coordinate prediction and boundary state classification."""

    @pytest.fixture
    def calibrated_instance(self):
        """Provides a calibrated ScreenGazeCalibrator instance."""
        calibrator = ScreenGazeCalibrator(boundary_margin=0.05)
        # Yaw: -20 to +20 maps to x: 0.1 to 0.9  (slope = 0.8/40 = 0.02, offset = 0.5)
        # Pitch: -15 to +15 maps to y: 0.9 to 0.1 (slope = -0.8/30 = -0.0267, offset = 0.5)
        targets = [
            (0.50, 0.50, 0.0, 0.0),
            (0.10, 0.10, -20.0, 15.0),
            (0.90, 0.10, 20.0, 15.0),
            (0.10, 0.90, -20.0, -15.0),
            (0.90, 0.90, 20.0, -15.0),
        ]
        for tx, ty, y, p in targets:
            for _ in range(3):
                calibrator.add_observation(tx, ty, y, p, 0.95)
        assert calibrator.fit() is True
        return calibrator

    def test_on_screen_center_mapping(self, calibrated_instance):
        """Looking straight ahead maps to center ON_SCREEN."""
        res = calibrated_instance.estimate_screen_gaze(gaze_yaw=0.0, gaze_pitch=0.0, gaze_confidence=0.90)
        assert res.status == "VALID"
        assert res.boundary_state == BOUNDARY_ON_SCREEN
        assert abs(res.screen_x - 0.50) < 0.02
        assert abs(res.screen_y - 0.50) < 0.02
        assert res.confidence > 0.70

    def test_near_boundary_classification(self, calibrated_instance):
        """Coordinates near the 5% margin boundary are classified as NEAR_BOUNDARY."""
        # Gaze at x ≈ 0.03 (near left border, margin = 0.05)
        # x = 0.02 * yaw + 0.5 -> yaw = (0.03 - 0.5) / 0.02 = -23.5
        res_near_left = calibrated_instance.estimate_screen_gaze(gaze_yaw=-23.5, gaze_pitch=0.0, gaze_confidence=0.90)
        assert res_near_left.boundary_state == BOUNDARY_NEAR_BOUNDARY
        assert res_near_left.screen_x < 0.05
        assert res_near_left.screen_x >= -0.05

        # Gaze at x ≈ 0.97 (near right border)
        res_near_right = calibrated_instance.estimate_screen_gaze(gaze_yaw=23.5, gaze_pitch=0.0, gaze_confidence=0.90)
        assert res_near_right.boundary_state == BOUNDARY_NEAR_BOUNDARY

    def test_outside_screen_classification_and_unclamped_coordinates(self, calibrated_instance):
        """Coordinates outside the screen margin are classified as OUTSIDE_SCREEN and NOT silently clamped."""
        # Yaw = 35.0 -> x = 0.02 * 35 + 0.5 = 1.20 (> 1.05)
        res_out_right = calibrated_instance.estimate_screen_gaze(gaze_yaw=35.0, gaze_pitch=0.0, gaze_confidence=0.90)
        assert res_out_right.boundary_state == BOUNDARY_OUTSIDE_SCREEN
        assert res_out_right.screen_x > 1.05  # Raw unclamped value preserved!
        assert res_out_right.status == "VALID"

        # Yaw = -35.0 -> x = -0.20 (< -0.05)
        res_out_left = calibrated_instance.estimate_screen_gaze(gaze_yaw=-35.0, gaze_pitch=0.0, gaze_confidence=0.90)
        assert res_out_left.boundary_state == BOUNDARY_OUTSIDE_SCREEN
        assert res_out_left.screen_x < -0.05  # Raw unclamped value preserved!

    def test_extrapolation_confidence_penalty(self, calibrated_instance):
        """Points far outside the monitor receive smooth confidence downscaling."""
        res_near = calibrated_instance.estimate_screen_gaze(gaze_yaw=0.0, gaze_pitch=0.0, gaze_confidence=0.90)
        # Far outside: yaw = 60.0 (x ~ 1.70)
        res_far = calibrated_instance.estimate_screen_gaze(gaze_yaw=60.0, gaze_pitch=0.0, gaze_confidence=0.90)

        assert res_far.boundary_state == BOUNDARY_OUTSIDE_SCREEN
        assert res_far.confidence < (res_near.confidence * 0.5)

    def test_uncalibrated_calibrator_returns_calibration_required(self):
        """Uncalibrated instance returns status NOT_CALIBRATED and CALIBRATION_REQUIRED."""
        calibrator = ScreenGazeCalibrator()
        res = calibrator.estimate_screen_gaze(gaze_yaw=0.0, gaze_pitch=0.0, gaze_confidence=0.90)

        assert res.status == CALIB_STATUS_NOT_CALIBRATED
        assert res.boundary_state == BOUNDARY_CALIBRATION_REQUIRED
        assert res.screen_x is None
        assert res.confidence == 0.0
        assert res.is_calibrated is False

    def test_uncertain_and_invalid_inputs_return_uncertain(self, calibrated_instance):
        """UNCERTAIN or non-finite inputs return status UNCERTAIN with zero confidence."""
        res_none = calibrated_instance.estimate_screen_gaze(None, 0.0, 0.90)
        assert res_none.status == "UNCERTAIN"
        assert res_none.screen_x is None

        res_nan = calibrated_instance.estimate_screen_gaze(np.nan, 0.0, 0.90)
        assert res_nan.status == "UNCERTAIN"

        res_low_conf = calibrated_instance.estimate_screen_gaze(0.0, 0.0, 0.10)
        assert res_low_conf.status == "UNCERTAIN"

        res_uncertain_status = calibrated_instance.estimate_screen_gaze(0.0, 0.0, 0.90, gaze_status="UNCERTAIN")
        assert res_uncertain_status.status == "UNCERTAIN"


# ===========================================================================
# 5. Lifecycle and Recalibration Tests
# ===========================================================================

class TestScreenGazeLifecycle:
    """Verifies reset, session boundary clearing, and recalibration behavior."""

    def test_reset_clears_all_calibrated_state(self):
        calibrator = ScreenGazeCalibrator()
        for tx, ty in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 20, (ty - 0.5) * 20, 0.90)
        calibrator.fit()
        assert calibrator.is_calibrated is True

        calibrator.reset()
        assert calibrator.is_calibrated is False
        assert calibrator.sample_count == 0
        assert calibrator.quality_metrics == {}

        res = calibrator.estimate_screen_gaze(0.0, 0.0, 0.90)
        assert res.boundary_state == BOUNDARY_CALIBRATION_REQUIRED

    def test_recalibration_after_reset(self):
        """Calibrator can be re-calibrated successfully after a reset."""
        calibrator = ScreenGazeCalibrator()
        # Initial fit
        for tx, ty in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 20, (ty - 0.5) * 20, 0.90)
        assert calibrator.fit() is True

        # Reset
        calibrator.reset()
        assert calibrator.is_calibrated is False

        # Second fit with new samples
        for tx, ty in [(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8)]:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 25, (ty - 0.5) * 25, 0.92)
        assert calibrator.fit() is True
        assert calibrator.is_calibrated is True


# ===========================================================================
# 6. VisionPipeline Integration Tests
# ===========================================================================

class TestVisionPipelineScreenGazeIntegration:
    """Verifies end-to-end integration of screen gaze mapping in VisionPipeline."""

    def test_pipeline_without_screen_calibrator_emits_calibration_required(self):
        """Standard pipeline run without screen calibration leaves camera-gaze working and emits CALIBRATION_REQUIRED."""
        pipeline = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result, _ = pipeline.process_face(frame)
        assert result.screen_gaze is not None
        assert result.screen_gaze.status == CALIB_STATUS_NOT_CALIBRATED
        assert result.screen_gaze.boundary_state == BOUNDARY_CALIBRATION_REQUIRED
        assert result.screen_gaze.is_calibrated is False

    def test_pipeline_with_calibrated_calibrator_populates_screen_gaze(self, monkeypatch):
        """When ScreenGazeCalibrator is calibrated, pipeline populates valid screen_gaze."""
        calibrator = ScreenGazeCalibrator()
        for tx, ty in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 20, (ty - 0.5) * 20, 0.90)
        assert calibrator.fit() is True

        mock_rect = MagicMock()
        mock_rect.left.return_value = 100
        mock_rect.top.return_value = 100
        mock_rect.right.return_value = 300
        mock_rect.bottom.return_value = 300

        mock_landmarks = MagicMock()
        pts_dict = {
            30: (200, 200), 8: (200, 265),
            36: (150, 150), 45: (250, 150),
            48: (170, 240), 54: (230, 240)
        }
        for pt_idx in range(68):
            mock_pt = MagicMock()
            mock_pt.x, mock_pt.y = pts_dict.get(pt_idx, (150 + pt_idx, 150 + pt_idx))
            mock_landmarks.part.return_value = mock_pt

        pipeline = VisionPipeline(
            detector=MagicMock(return_value=[mock_rect]),
            predictor=MagicMock(return_value=mock_landmarks),
            screen_calibrator=calibrator
        )

        R_dummy = np.eye(3)
        t_dummy = np.array([[0.0], [0.0], [600.0]])
        mock_tracking = MagicMock()
        mock_tracking.gaze_direction = "Looking Center"
        mock_tracking.tracking_state = "PAIRED_TRACKING"
        mock_tracking.combined_confidence = 0.90
        mock_tracking.left_confidence = 0.90
        mock_tracking.right_confidence = 0.90
        mock_tracking.left_gaze_dir = "Looking Center"
        mock_tracking.right_gaze_dir = "Looking Center"
        mock_tracking.is_single_eye_fallback = False
        mock_tracking.rejection_reason = ""
        monkeypatch.setattr("vision_pipeline.get_last_gaze_tracking_result", lambda: mock_tracking)
        monkeypatch.setattr("vision_pipeline.get_last_gaze_diagnostics", lambda: {"missing_pupils": False, "missing_face": False, "missing_landmarks": False})

        monkeypatch.setattr(
            "vision_pipeline.get_head_pose_estimation",
            lambda *args, **kwargs: ((0.0, 0.0, 0.0), R_dummy, t_dummy, 0.90)
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_head_pose_from_landmarks",
            lambda frame, lm, **kwargs: (frame, "Looking Screen")
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_gaze_from_landmarks",
            lambda frame, lm, **kwargs: (frame, "Looking Center")
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert result.screen_gaze is not None
        assert result.screen_gaze.status == "VALID"
        assert result.screen_gaze.is_calibrated is True
        assert result.screen_gaze.boundary_state in (BOUNDARY_ON_SCREEN, BOUNDARY_NEAR_BOUNDARY)
        assert abs(result.screen_gaze.screen_x - 0.50) < 0.10

    def test_pipeline_zero_faces_emits_uncertain_when_calibrated(self):
        """Zero faces detected emits UNCERTAIN screen gaze without crashing or claiming VALID."""
        calibrator = ScreenGazeCalibrator()
        for tx, ty in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 20, (ty - 0.5) * 20, 0.90)
        assert calibrator.fit() is True

        pipeline = VisionPipeline(
            detector=MagicMock(return_value=[]),
            predictor=MagicMock(),
            screen_calibrator=calibrator
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _ = pipeline.process_face(frame)

        assert result.screen_gaze.status == "UNCERTAIN"
        assert result.screen_gaze.confidence == 0.0
        assert result.screen_gaze.screen_x is None

    def test_independent_pipeline_instances_screen_calibration_isolation(self):
        """Two pipeline instances do not share screen calibrator state."""
        calibrator = ScreenGazeCalibrator()
        for tx, ty in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]:
            for _ in range(3):
                calibrator.add_observation(tx, ty, (tx - 0.5) * 20, (ty - 0.5) * 20, 0.90)
        calibrator.fit()

        pipe_calibrated = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock(), screen_calibrator=calibrator)
        pipe_uncalibrated = VisionPipeline(detector=MagicMock(return_value=[]), predictor=MagicMock(), screen_calibrator=None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res_calib, _ = pipe_calibrated.process_face(frame)
        res_uncalib, _ = pipe_uncalibrated.process_face(frame)

        assert res_calib.screen_gaze.is_calibrated is True
        assert res_uncalib.screen_gaze.is_calibrated is False
        assert res_uncalib.screen_gaze.boundary_state == BOUNDARY_CALIBRATION_REQUIRED
