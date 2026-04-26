"""Hand detection using WiLoR's YOLOv8 hand detector + RealSense D455 depth.

Detects hands (any pose/orientation) via WiLoR's pretrained detector,
queries aligned depth at bbox center for 3D position,
transforms to robot base frame using T_cam_to_robot.

Usage:
    detector = HandDetector(T_cam_to_robot=np.load("calibration_data/T_cam_to_robot.npy"))
    detector.start()
    ...
    color_img, depth_img = detector.get_frames()
    hand = detector.detect(color_img, depth_img)
    ...
    detector.stop()
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs

# PyTorch 2.6+ compatibility: ultralytics 8.1.34 needs weights_only=False
import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': False})

from ultralytics import YOLO

from frrl.envs.real_config import FRONT_CAMERA_SERIAL

WILOR_DETECTOR_PATH = Path("/home/lab1/WiLoR/pretrained_models/detector.pt")


@dataclass
class HandDetection:
    active: bool = False
    pos_robot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pixel: tuple = (0, 0)
    bbox: tuple = (0, 0, 0, 0)  # x1, y1, x2, y2 (pixels)
    bbox_size_3d: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (w, h) meters, image plane at center depth
    confidence: float = 0.0
    hand_side: str = ""  # "left" or "right"


class HandDetector:
    """WiLoR YOLOv8 hand detector + D455 depth → 3D hand position in robot base frame."""

    def __init__(
        self,
        T_cam_to_robot: np.ndarray,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        depth_min: float = 0.2,
        depth_max: float = 1.5,
        detection_conf: float = 0.3,
        detector_path: Optional[str] = None,
        serial: Optional[str] = None,
    ):
        self.T_cam_to_robot = T_cam_to_robot
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.detection_conf = detection_conf
        # 默认锁定 front 相机 serial：双相机插上时 pyrealsense 枚举顺序不可控，
        # 必须 enable_device 显式选 front，否则会随机抓到 wrist。
        self.serial = serial or FRONT_CAMERA_SERIAL

        path = detector_path or str(WILOR_DETECTOR_PATH)
        self._model = YOLO(path)

        self._pipeline = None
        self._align = None
        self._depth_scale = None
        self._intrinsics = None

    def start(self):
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self._pipeline.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsics = intr
        self._cam_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ])

        self._align = rs.align(rs.stream.color)

        time.sleep(3)
        for _ in range(10):
            self._pipeline.wait_for_frames(timeout_ms=10000)

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

    def get_frames(self):
        frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        aligned = self._align.process(frames)
        color_img = np.asanyarray(aligned.get_color_frame().get_data())
        depth_img = np.asanyarray(aligned.get_depth_frame().get_data())
        return color_img, depth_img

    def detect(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        exclude_near_flange: Optional[np.ndarray] = None,
        flange_radius: float = 0.10,
        exclude_near_tcp: Optional[np.ndarray] = None,
        tcp_radius: float = 0.06,
    ) -> HandDetection:
        """Detect a hand and return its 3D position in robot base frame.

        Self-detection 双参考点几何过滤：
          - ``exclude_near_flange`` (= ``compute_hand_body_equiv``)：franka hand
            主体中心，半径 ``flange_radius`` (默认 10cm) 覆盖手掌+腕部。
          - ``exclude_near_tcp`` (= TCP/pinch_site 位置)：finger 末端，半径
            ``tcp_radius`` (默认 6cm) 覆盖夹爪指尖区域。
        任一球内的 detection 视为 self-detection 丢弃。两个独立球比单一 12cm
        球更紧贴 franka hand 几何（手掌侧短、指尖侧长），既不漏过 finger 自检测，
        也不把贴 gripper 旁的人手误杀。

        ⚠️ 操作员安全约定：人手不能伸入 flange 10cm 球内 —— 进入后 detection
        被滤，supervisor 看不到障碍 (sim 训练分布外盲区)。BACKUP 触发距 D_SAFE
        通常 ≥ 30cm，正常人手作业不会撞此盲区。
        """
        results = self._model.predict(color_img, verbose=False, conf=self.detection_conf)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return HandDetection(active=False)

        # 先把每个 box 转成 3D 候选，方便统一过滤 self-detection。
        candidates = []  # list of dicts with bbox/3d/conf/cls
        cx_img, cy_img = self.width / 2, self.height / 2
        for i in range(len(boxes)):
            x1f, y1f, x2f, y2f = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            depth_val = self._get_depth_at(depth_img, u, v, radius=5)
            if depth_val <= self.depth_min or depth_val > self.depth_max:
                continue

            x_cam = (u - self._intrinsics.ppx) * depth_val / self._intrinsics.fx
            y_cam = (v - self._intrinsics.ppy) * depth_val / self._intrinsics.fy
            p_cam = np.array([x_cam, y_cam, depth_val])
            p_robot = (self.T_cam_to_robot @ np.append(p_cam, 1.0))[:3]

            # 几何过滤：双参考点。flange 球覆盖手掌+腕部 (10cm)，TCP 球覆盖
            # finger 末端 (6cm)。两个球任一命中 → self-detection 丢弃。
            if exclude_near_flange is not None:
                d_flange = float(np.linalg.norm(
                    p_robot - np.asarray(exclude_near_flange).reshape(3)
                ))
                if d_flange < flange_radius:
                    continue
            if exclude_near_tcp is not None:
                d_tcp = float(np.linalg.norm(
                    p_robot - np.asarray(exclude_near_tcp).reshape(3)
                ))
                if d_tcp < tcp_radius:
                    continue

            cx_box = (x1 + x2) / 2.0
            cy_box = (y1 + y2) / 2.0
            img_dist2 = (cx_box - cx_img) ** 2 + (cy_box - cy_img) ** 2

            candidates.append({
                "i": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "u": u, "v": v, "depth": depth_val,
                "p_cam": p_cam, "p_robot": p_robot,
                "img_dist2": img_dist2,
            })

        if not candidates:
            return HandDetection(active=False)

        # 在剩余候选里挑最靠图像中心的（保留之前的优先级）。
        best = min(candidates, key=lambda c: c["img_dist2"])
        i = best["i"]
        box = boxes[i]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        hand_side = self._model.names[cls_id]

        x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
        u, v = best["u"], best["v"]
        depth_val = best["depth"]
        p_cam = best["p_cam"]
        p_robot = best["p_robot"]

        bbox_w_3d = (x2 - x1) * depth_val / self._intrinsics.fx
        bbox_h_3d = (y2 - y1) * depth_val / self._intrinsics.fy
        bbox_size_3d = np.array([bbox_w_3d, bbox_h_3d], dtype=np.float32)

        return HandDetection(
            active=True,
            pos_robot=p_robot,
            pos_cam=p_cam,
            pixel=(u, v),
            bbox=(x1, y1, x2, y2),
            bbox_size_3d=bbox_size_3d,
            confidence=conf,
            hand_side=hand_side,
        )

    def detect_aruco(self, color_img: np.ndarray, depth_img: np.ndarray,
                     marker_id: int = 55, marker_size: float = 0.15) -> Optional[np.ndarray]:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(color_img)

        if ids is None:
            return None

        dist_coeffs = np.array(self._intrinsics.coeffs[:5])
        half = marker_size / 2.0
        obj_points = np.array([
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        for i, mid in enumerate(ids.flatten()):
            if mid == marker_id:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i].reshape(4, 2),
                    self._cam_matrix, dist_coeffs,
                )
                if not ok:
                    continue
                p_cam = tvec.flatten()
                p_robot = (self.T_cam_to_robot @ np.append(p_cam, 1.0))[:3]
                return p_robot

        return None

    def _get_depth_at(self, depth_img, u, v, radius=5):
        u_min = max(0, u - radius)
        u_max = min(self.width, u + radius + 1)
        v_min = max(0, v - radius)
        v_max = min(self.height, v + radius + 1)
        patch = depth_img[v_min:v_max, u_min:u_max].astype(np.float32) * self._depth_scale
        valid = patch[patch > 0.1]
        if len(valid) == 0:
            return 0.0
        # 25th percentile（lower quartile）而非 median：人手部分遮挡 robot hand
        # 时 median 倾向距离更近的物体（robot hand 金属夹具），lower-quartile
        # 更接近"前景"语义，对人手 - robot hand 重叠场景更鲁棒。
        return float(np.percentile(valid, 25))

    def draw_detection(self, color_img: np.ndarray, detection: HandDetection,
                       aruco_pos: Optional[np.ndarray] = None) -> np.ndarray:
        vis = color_img.copy()

        if detection.active:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            u, v = detection.pixel
            cv2.circle(vis, (u, v), 5, (0, 255, 0), -1)
            label = (f"{detection.hand_side} {detection.confidence:.2f} "
                     f"({detection.pos_robot[0]:.2f},{detection.pos_robot[1]:.2f},{detection.pos_robot[2]:.2f})")
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 右上角，避开 caller 在左上角写的 MODE / dist 标签；
            # max(10, ...) 防 width<130 时跑负坐标
            x = max(10, vis.shape[1] - 130)
            cv2.putText(vis, "No hand", (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if aruco_pos is not None:
            cv2.putText(vis, f"ArUco TCP: ({aruco_pos[0]:.3f},{aruco_pos[1]:.3f},{aruco_pos[2]:.3f})",
                        (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return vis
