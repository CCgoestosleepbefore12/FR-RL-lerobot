"""
RealSense D455 相机采集模块

移植自 hil-serl/serl_robot_infra/franka_env/camera/，提供：
- RSCapture: 单相机 RealSense pipeline 封装
- VideoCapture: 线程化异步采集（丢弃旧帧，保持最新）
- RealSenseCameraManager: 多相机统一管理接口
"""

import copy
import logging
import queue
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# pyrealsense2 延迟导入（GPU 工作站上才有）
try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False


class RSCapture:
    """单个 RealSense 相机的 pipeline 封装。"""

    @staticmethod
    def get_device_serial_numbers():
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name: str, serial_number: str,
                 dim: Tuple[int, int] = (640, 480), fps: int = 15,
                 depth: bool = False, exposure: int = 40000):
        assert _HAS_REALSENSE, "pyrealsense2 未安装，请 pip install pyrealsense2"

        self.name = name
        available = self.get_device_serial_numbers()
        if serial_number not in available:
            raise RuntimeError(
                f"RealSense {serial_number} 未找到。可用设备: {available}"
            )

        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)

        self.profile = self.pipe.start(self.cfg)
        sensor = self.profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.exposure, exposure)

        # 深度对齐到 RGB
        self.align = rs.align(rs.stream.color)
        logging.info(f"RealSense '{name}' ({serial_number}) 初始化成功")

    def read(self):
        """读取一帧。返回 (success, image)。image 为 BGR uint8。"""
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame.is_video_frame():
            return False, None

        image = np.asarray(color_frame.get_data())

        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)

        return True, image

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()


class VideoCapture:
    """线程化视频采集，丢弃旧帧保持最新。"""

    def __init__(self, cap, name: Optional[str] = None):
        self.name = name or cap.name
        self.cap = cap
        self.q = queue.Queue()
        self.enable = True
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        """获取最新帧，阻塞最多 5 秒。"""
        return self.q.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join(timeout=3)
        self.cap.close()


class RealSenseCameraManager:
    """多 RealSense 相机统一管理。

    Args:
        camera_configs: {相机名: {serial_number, dim, fps, exposure, ...}}
        image_crop: {相机名: crop函数}，可选
        image_size: 输出图像尺寸 (width, height)
    """

    def __init__(
        self,
        camera_configs: Dict[str, dict],
        image_crop: Optional[Dict[str, callable]] = None,
        image_size: Tuple[int, int] = (128, 128),
    ):
        self.image_crop = image_crop or {}
        self.image_size = image_size
        self.caps: OrderedDict[str, VideoCapture] = OrderedDict()

        for cam_name, kwargs in camera_configs.items():
            rs_cap = RSCapture(name=cam_name, **kwargs)
            self.caps[cam_name] = VideoCapture(rs_cap)

        logging.info(f"RealSenseCameraManager 初始化: {list(self.caps.keys())}")

    def get_images(self) -> Dict[str, np.ndarray]:
        """获取所有相机图像，裁剪并缩放到目标尺寸。

        Returns:
            {相机名: RGB uint8 (H, W, 3)}
        """
        images = {}
        for cam_name, cap in self.caps.items():
            try:
                bgr = cap.read()
            except queue.Empty:
                logging.error(f"相机 {cam_name} 超时，无法读取帧")
                raise

            # 裁剪（可选）
            if cam_name in self.image_crop:
                bgr = self.image_crop[cam_name](bgr)

            # 缩放到目标尺寸
            resized = cv2.resize(bgr, self.image_size)
            # BGR → RGB
            images[cam_name] = resized[..., ::-1].copy()

        return images

    def close(self):
        for cap in self.caps.values():
            try:
                cap.close()
            except Exception as e:
                logging.warning(f"关闭相机失败: {e}")
