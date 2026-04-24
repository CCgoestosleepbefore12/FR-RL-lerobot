"""Real TCP tracking via ArUco marker on the gripper + RealSense D455.

Provides `noisy_real_tcp` signal that matches the simulation semantics
(ground-truth TCP + 5mm Gaussian noise, independent of encoder bias).

Features:
  - EMA smoothing to reduce frame-to-frame jitter
  - Hold last-known position during short-term occlusion
  - Optional fallback to FK TCP after occlusion timeout
  - 5mm Gaussian noise (matches simulation's DR_TCP_POS_STD)

Usage:
    tracker = TCPTracker(T_cam_to_robot=np.load(".../T_cam_to_robot.npy"))
    # Each step (share D455 frames with HandDetector):
    real_tcp = tracker.update(color_img, depth_img, fk_tcp=current_fk_tcp)
    if real_tcp is None:
        # Lost and no fallback — handle appropriately
        ...
"""
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TCPTrackerState:
    """Diagnostic state for external inspection/logging."""
    valid: bool = False
    source: str = "none"          # "aruco", "hold", "fk", "none"
    raw_pos_cam: Optional[np.ndarray] = None
    smoothed_pos_robot: Optional[np.ndarray] = None
    time_since_last_aruco: float = float("inf")
    aruco_detected_this_step: bool = False


class TCPTracker:
    """Track TCP via ArUco marker on the gripper, output noisy_real_tcp."""

    def __init__(
        self,
        T_cam_to_robot: np.ndarray,
        cam_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        marker_id: int = 55,
        marker_size: float = 0.15,
        marker_dict: int = cv2.aruco.DICT_5X5_100,
        ema_alpha: float = 0.3,
        hold_timeout_s: float = 0.5,
        enable_fk_fallback: bool = True,
        add_noise: bool = True,
        noise_std: float = 0.005,  # 5mm, matches simulation DR_TCP_POS_STD
        seed: Optional[int] = None,
    ):
        self.T_cam_to_robot = T_cam_to_robot
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.ema_alpha = ema_alpha
        self.hold_timeout_s = hold_timeout_s
        self.enable_fk_fallback = enable_fk_fallback
        self.add_noise = add_noise
        self.noise_std = noise_std

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, cv2.aruco.DetectorParameters())

        half = marker_size / 2.0
        self._obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        self._smoothed_robot: Optional[np.ndarray] = None
        self._last_aruco_time: float = 0.0
        self._random = np.random.default_rng(seed)

        self.state = TCPTrackerState()

    def set_camera_intrinsics(self, cam_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set or update camera intrinsics (e.g., after D455 is started)."""
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs

    def _detect_aruco_pos_cam(self, color_img: np.ndarray) -> Optional[np.ndarray]:
        """Run ArUco detection + solvePnP. Returns marker center in camera frame or None."""
        if self.cam_matrix is None:
            return None
        corners, ids, _ = self._detector.detectMarkers(color_img)
        if ids is None:
            return None
        for i, mid in enumerate(ids.flatten()):
            if mid == self.marker_id:
                ok, _, tvec = cv2.solvePnP(
                    self._obj_points, corners[i].reshape(4, 2),
                    self.cam_matrix, self.dist_coeffs,
                )
                if ok:
                    return tvec.flatten()
        return None

    def update(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray = None,  # unused (reserved for depth-refined modes)
        fk_tcp: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Produce noisy_real_tcp for this step. Returns None if no valid source."""
        now = time.time()
        self.state = TCPTrackerState()

        raw_cam = self._detect_aruco_pos_cam(color_img)
        self.state.aruco_detected_this_step = raw_cam is not None

        if raw_cam is not None:
            # Aruco detected: transform + EMA smooth
            raw_robot = (self.T_cam_to_robot @ np.append(raw_cam, 1.0))[:3]
            if self._smoothed_robot is None:
                self._smoothed_robot = raw_robot
            else:
                self._smoothed_robot = (
                    (1 - self.ema_alpha) * self._smoothed_robot + self.ema_alpha * raw_robot
                )
            self._last_aruco_time = now
            self.state.valid = True
            self.state.source = "aruco"
            self.state.raw_pos_cam = raw_cam
            self.state.smoothed_pos_robot = self._smoothed_robot.copy()
            self.state.time_since_last_aruco = 0.0
            return self._apply_noise(self._smoothed_robot)

        # Aruco lost: decide fallback
        time_since = now - self._last_aruco_time if self._last_aruco_time > 0 else float("inf")
        self.state.time_since_last_aruco = time_since

        if self._smoothed_robot is not None and time_since < self.hold_timeout_s:
            # Short-term occlusion: hold last smoothed value
            self.state.valid = True
            self.state.source = "hold"
            self.state.smoothed_pos_robot = self._smoothed_robot.copy()
            return self._apply_noise(self._smoothed_robot)

        if self.enable_fk_fallback and fk_tcp is not None:
            self.state.valid = True
            self.state.source = "fk"
            return self._apply_noise(np.asarray(fk_tcp, dtype=np.float64))

        self.state.valid = False
        self.state.source = "none"
        return None

    def _apply_noise(self, pos: np.ndarray) -> np.ndarray:
        if self.add_noise and self.noise_std > 0:
            return pos + self._random.normal(0, self.noise_std, 3)
        return pos.copy()

    def reset(self):
        self._smoothed_robot = None
        self._last_aruco_time = 0.0
        self.state = TCPTrackerState()
