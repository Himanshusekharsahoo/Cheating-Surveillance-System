"""
Full System Integration Tests for Phase 10.13.6:
Validates the end-to-end processing chain across Phase 10.13.1 through Phase 10.13.5.

Chain:
Face Detection
-> Facial Landmark Extraction
-> 3D Head Pose Estimation (PnP + Similarity Reflection)
-> Eye Detection & Pupil Localization
-> Eye Gaze Differential to Head-Local Vector
-> 3D Head-Eye Gaze Fusion (Camera-Relative Vector & Angles)
-> Screen Gaze Calibration & Monitor Boundary Mapping
-> Diagnostic Integrity & Temporal Tracker State Lifecycle
"""

import math
from unittest.mock import MagicMock
import numpy as np
import pytest

from screen_gaze import (
    ScreenGazeCalibrator,
    BOUNDARY_ON_SCREEN,
    BOUNDARY_NEAR_BOUNDARY,
    BOUNDARY_OUTSIDE_SCREEN,
    BOUNDARY_UNCERTAIN,
    BOUNDARY_CALIBRATION_REQUIRED,
)
from vision_pipeline import VisionPipeline, FaceAnalysisResult


class TestFullSystemIntegrationChain:
    """Verifies that the entire perception and calibration pipeline operates seamlessly as one system."""

    @pytest.fixture
    def calibrated_pipeline_fixture(self):
        """Constructs a VisionPipeline instance equipped with a calibrated ScreenGazeCalibrator."""
        calibrator = ScreenGazeCalibrator(boundary_margin=0.05)
        # 5-point calibration grid:
        # yaw: -20 to +20 maps to x: 0.1 to 0.9 (scale = 0.02, offset = 0.50)
        # pitch: -15 to +15 maps to y: 0.9 to 0.1 (scale = -0.0267, offset = 0.50)
        calib_points = [
            (0.50, 0.50, 0.0, 0.0),
            (0.10, 0.10, -20.0, 15.0),
            (0.90, 0.10, 20.0, 15.0),
            (0.10, 0.90, -20.0, -15.0),
            (0.90, 0.90, 20.0, -15.0),
        ]
        for tx, ty, yaw, pitch in calib_points:
            for _ in range(3):
                calibrator.add_observation(tx, ty, yaw, pitch, confidence=0.95)

        assert calibrator.fit() is True
        assert calibrator.is_calibrated is True

        pipeline = VisionPipeline(
            detector=MagicMock(),
            predictor=MagicMock(),
            screen_calibrator=calibrator,
        )
        return pipeline, calibrator

    def _create_synthetic_landmarks(self, face_rect, nose_pos=(200, 200), eye_offset_x=50, eye_offset_y=-30):
        """Creates realistic dlib 68-point mock landmarks with valid geometry."""
        pts = {}
        nx, ny = nose_pos
        pts[30] = (nx, ny)                 # Nose tip
        pts[8] = (nx, ny + 65)             # Chin
        pts[36] = (nx - eye_offset_x, ny + eye_offset_y)  # Left eye outer corner
        pts[39] = (nx - eye_offset_x + 20, ny + eye_offset_y) # Left eye inner corner
        pts[42] = (nx + eye_offset_x - 20, ny + eye_offset_y) # Right eye inner corner
        pts[45] = (nx + eye_offset_x, ny + eye_offset_y)  # Right eye outer corner
        pts[48] = (nx - 24, ny + 35)       # Left mouth corner
        pts[54] = (nx + 24, ny + 35)       # Right mouth corner

        # Fill remaining parts with non-zero coordinates
        for idx in range(68):
            if idx not in pts:
                pts[idx] = (nx + (idx - 34), ny + (idx - 34))

        mock_lm = MagicMock()
        mock_lm.part.side_effect = lambda idx: MagicMock(x=pts.get(idx, (100, 100))[0], y=pts.get(idx, (100, 100))[1])
        return mock_lm

    def test_end_to_end_multiframe_perception_lifecycle(self, calibrated_pipeline_fixture, monkeypatch):
        """
        Executes a 6-frame sequential lifecycle verifying component coordination:
        Frame 1: Frontal compliant head & gaze -> ON_SCREEN, VALID.
        Frame 2: Head turned right 25 deg -> OUTSIDE_SCREEN, raw x > 1.05, VALID.
        Frame 3: Head returns center, single-eye fallback active -> ON_SCREEN, discounted confidence.
        Frame 4: Micro-blink (both eyes occluded) -> UNCERTAIN, confidence = 0.0.
        Frame 5: Full paired-eye recovery -> ON_SCREEN, PAIRED_TRACKING, VALID.
        Frame 6: Candidate leaves frame (0 faces) -> NO_FACE, UNCERTAIN, tracker cleared.
        """
        pipeline, calibrator = calibrated_pipeline_fixture
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_rect = MagicMock()
        mock_rect.left.return_value = 100
        mock_rect.top.return_value = 100
        mock_rect.right.return_value = 300
        mock_rect.bottom.return_value = 300
        pipeline._detector.return_value = [mock_rect]

        # -------------------------------------------------------------
        # FRAME 1: Frontal Neutral Face Looking at Screen
        # -------------------------------------------------------------
        lm_f1 = self._create_synthetic_landmarks(mock_rect)
        pipeline._predictor.return_value = lm_f1

        R_cam_f1 = np.eye(3)
        t_f1 = np.array([[0.0], [0.0], [600.0]])

        monkeypatch.setattr(
            "vision_pipeline.get_head_pose_estimation",
            lambda *args, **kwargs: ((0.0, 0.0, 0.0), R_cam_f1, t_f1, 0.95)
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_head_pose_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Looking at Screen")
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_gaze_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Looking Center")
        )

        mock_tracking_f1 = MagicMock()
        mock_tracking_f1.gaze_direction = "Looking Center"
        mock_tracking_f1.tracking_state = "PAIRED_TRACKING"
        mock_tracking_f1.combined_confidence = 0.92
        mock_tracking_f1.left_confidence = 0.92
        mock_tracking_f1.right_confidence = 0.92
        mock_tracking_f1.left_gaze_dir = "Looking Center"
        mock_tracking_f1.right_gaze_dir = "Looking Center"
        mock_tracking_f1.is_single_eye_fallback = False
        mock_tracking_f1.rejection_reason = ""
        monkeypatch.setattr("vision_pipeline.get_last_gaze_tracking_result", lambda: mock_tracking_f1)
        monkeypatch.setattr("vision_pipeline.get_last_gaze_diagnostics", lambda: {
            "missing_pupils": False, "missing_face": False, "missing_landmarks": False,
            "mean_diff_x": 0.0, "mean_diff_y": 0.0
        })

        res1, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert res1.face_count == 1
        assert res1.head_pose.status == "VALID"
        assert res1.head_pose.yaw == 0.0
        assert res1.eye_gaze.status == "VALID"
        assert res1.eye_gaze.tracking_state == "PAIRED_TRACKING"
        assert res1.combined_gaze.status == "VALID"
        assert res1.combined_gaze.yaw == 0.0
        assert res1.screen_gaze.status == "VALID"
        assert res1.screen_gaze.boundary_state == BOUNDARY_ON_SCREEN
        assert abs(res1.screen_gaze.screen_x - 0.50) < 0.05
        assert abs(res1.screen_gaze.screen_y - 0.50) < 0.05

        # -------------------------------------------------------------
        # FRAME 2: Candidate turns head Right 30 degrees (eyes follow head)
        # -------------------------------------------------------------
        # Yaw +30 deg -> screen_x = 0.02 * 30 + 0.5 = 1.10 (> 1.05 = OUTSIDE_SCREEN)
        rad30 = math.radians(30.0)
        R_cam_f2 = np.array([
            [math.cos(rad30), 0.0, math.sin(rad30)],
            [0.0, 1.0, 0.0],
            [-math.sin(rad30), 0.0, math.cos(rad30)],
        ])
        monkeypatch.setattr(
            "vision_pipeline.get_head_pose_estimation",
            lambda *args, **kwargs: ((0.0, 30.0, 0.0), R_cam_f2, t_f1, 0.90)
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_head_pose_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Looking Right")
        )

        res2, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert res2.head_pose.yaw == 30.0
        assert res2.combined_gaze.status == "VALID"
        assert abs(res2.combined_gaze.yaw - 30.0) < 0.1
        assert res2.screen_gaze.status == "VALID"
        assert res2.screen_gaze.boundary_state == BOUNDARY_OUTSIDE_SCREEN
        assert res2.screen_gaze.screen_x > 1.05  # Raw unclamped coordinate preserved!

        # -------------------------------------------------------------
        # FRAME 3: Head returns center, right eye occluded -> Single-Eye Fallback
        # -------------------------------------------------------------
        monkeypatch.setattr(
            "vision_pipeline.get_head_pose_estimation",
            lambda *args, **kwargs: ((0.0, 0.0, 0.0), R_cam_f1, t_f1, 0.95)
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_head_pose_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Looking at Screen")
        )
        monkeypatch.setattr(
            "vision_pipeline.analyze_gaze_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Looking Center")
        )

        mock_tracking_f3 = MagicMock()
        mock_tracking_f3.gaze_direction = "Looking Center"
        mock_tracking_f3.tracking_state = "LEFT_EYE_ONLY"
        mock_tracking_f3.combined_confidence = 0.72  # Discounted single-eye
        mock_tracking_f3.left_confidence = 0.90
        mock_tracking_f3.right_confidence = 0.0
        mock_tracking_f3.left_gaze_dir = "Looking Center"
        mock_tracking_f3.right_gaze_dir = None
        mock_tracking_f3.is_single_eye_fallback = True
        mock_tracking_f3.rejection_reason = "right_eye_occluded"
        monkeypatch.setattr("vision_pipeline.get_last_gaze_tracking_result", lambda: mock_tracking_f3)
        monkeypatch.setattr("vision_pipeline.get_last_gaze_diagnostics", lambda: {
            "missing_pupils": True, "missing_right_pupil": True, "missing_left_pupil": False,
            "missing_face": False, "missing_landmarks": False,
            "mean_diff_x": 0.0, "mean_diff_y": 0.0
        })

        res3, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert res3.eye_gaze.is_single_eye_fallback is True
        assert res3.eye_gaze.tracking_state == "LEFT_EYE_ONLY"
        assert res3.combined_gaze.status == "VALID"
        assert res3.combined_gaze.confidence == round(0.95 * 0.72, 4)
        assert res3.screen_gaze.status == "VALID"
        assert res3.screen_gaze.boundary_state == BOUNDARY_ON_SCREEN
        assert res3.screen_gaze.confidence < res1.screen_gaze.confidence

        # -------------------------------------------------------------
        # FRAME 4: Both Eyes Closed / Micro-Blink -> UNCERTAIN
        # -------------------------------------------------------------
        mock_tracking_f4 = MagicMock()
        mock_tracking_f4.gaze_direction = "Uncertain Gaze"
        mock_tracking_f4.tracking_state = "UNCERTAIN"
        mock_tracking_f4.combined_confidence = 0.0
        mock_tracking_f4.is_single_eye_fallback = False
        mock_tracking_f4.rejection_reason = "full_blink"
        monkeypatch.setattr("vision_pipeline.get_last_gaze_tracking_result", lambda: mock_tracking_f4)
        monkeypatch.setattr("vision_pipeline.get_last_gaze_diagnostics", lambda: {
            "missing_pupils": True, "missing_right_pupil": True, "missing_left_pupil": True,
            "missing_face": False, "missing_landmarks": False
        })
        monkeypatch.setattr(
            "vision_pipeline.analyze_gaze_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Uncertain Gaze")
        )

        res4, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert res4.eye_gaze.status == "UNCERTAIN"
        assert res4.eye_gaze.confidence == 0.0
        assert res4.combined_gaze.status == "UNCERTAIN"
        assert res4.combined_gaze.confidence == 0.0
        assert res4.screen_gaze.status == "UNCERTAIN"
        assert res4.screen_gaze.confidence == 0.0
        assert res4.screen_gaze.screen_x is None

        # -------------------------------------------------------------
        # FRAME 5: Eyes Reopen -> Recovery to PAIRED_TRACKING
        # -------------------------------------------------------------
        monkeypatch.setattr("vision_pipeline.get_last_gaze_tracking_result", lambda: mock_tracking_f1)
        monkeypatch.setattr("vision_pipeline.get_last_gaze_diagnostics", lambda: {
            "missing_pupils": False, "missing_face": False, "missing_landmarks": False,
            "mean_diff_x": 0.0, "mean_diff_y": 0.0
        })
        monkeypatch.setattr(
            "vision_pipeline.analyze_gaze_from_landmarks",
            lambda fr, lm, **kwargs: (fr, "Looking Center")
        )

        res5, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert res5.eye_gaze.status == "VALID"
        assert res5.eye_gaze.tracking_state == "PAIRED_TRACKING"
        assert res5.screen_gaze.status == "VALID"
        assert res5.screen_gaze.boundary_state == BOUNDARY_ON_SCREEN

        # -------------------------------------------------------------
        # FRAME 6: Candidate leaves desk (0 faces detected)
        # -------------------------------------------------------------
        pipeline._detector.return_value = []

        res6, _ = pipeline.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert res6.face_count == 0
        assert res6.head_pose.status == "NO_FACE"
        assert res6.eye_gaze.status == "NO_FACE"
        assert res6.combined_gaze.status == "NO_FACE"
        assert res6.screen_gaze.status == "UNCERTAIN"
        assert res6.screen_gaze.confidence == 0.0

    def test_uncalibrated_pipeline_preserves_full_camera_relative_tracking(self, monkeypatch):
        """When ScreenGazeCalibrator is omitted, camera-relative gaze operates 100% normally."""
        pipeline_uncalib = VisionPipeline(detector=MagicMock(), predictor=MagicMock(), screen_calibrator=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_rect = MagicMock()
        mock_rect.left.return_value = 100
        mock_rect.top.return_value = 100
        mock_rect.right.return_value = 300
        mock_rect.bottom.return_value = 300
        pipeline_uncalib._detector.return_value = [mock_rect]
        pipeline_uncalib._predictor.return_value = self._create_synthetic_landmarks(mock_rect)

        R_cam = np.eye(3)
        t_cam = np.array([[0.0], [0.0], [600.0]])
        monkeypatch.setattr("vision_pipeline.get_head_pose_estimation", lambda *args, **kwargs: ((0.0, 10.0, 0.0), R_cam, t_cam, 0.90))
        monkeypatch.setattr("vision_pipeline.analyze_head_pose_from_landmarks", lambda fr, lm, **kwargs: (fr, "Looking Screen"))
        monkeypatch.setattr("vision_pipeline.analyze_gaze_from_landmarks", lambda fr, lm, **kwargs: (fr, "Looking Center"))

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
        monkeypatch.setattr("vision_pipeline.get_last_gaze_diagnostics", lambda: {
            "missing_pupils": False, "missing_face": False, "missing_landmarks": False,
            "mean_diff_x": 0.0, "mean_diff_y": 0.0
        })

        result, _ = pipeline_uncalib.process_face(frame, calibrated_angles=(0.0, 0.0, 0.0))

        assert result.head_pose.status == "VALID"
        assert result.eye_gaze.status == "VALID"
        assert result.combined_gaze.status == "VALID"
        assert result.screen_gaze.status == "NOT_CALIBRATED"
        assert result.screen_gaze.boundary_state == BOUNDARY_CALIBRATION_REQUIRED
        assert result.screen_gaze.confidence == 0.0
