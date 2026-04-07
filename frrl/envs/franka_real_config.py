"""
Franka Panda 真机环境配置

参考 hil-serl 的 DefaultEnvConfig，适配 FR-RL 的观测格式。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from frrl.fault_injection import EncoderBiasConfig


@dataclass
class FrankaRealConfig:
    """Franka Panda 真机环境配置。"""

    # Flask Server 连接（RT PC）
    server_url: str = "http://192.168.1.1:5000/"

    # RealSense D455 相机配置
    cameras: Dict[str, dict] = field(default_factory=lambda: {
        "front": {"serial_number": "000000000000", "dim": (640, 480), "fps": 15, "exposure": 40000},
        "wrist": {"serial_number": "000000000000", "dim": (640, 480), "fps": 15, "exposure": 40000},
    })
    image_crop: Dict[str, callable] = field(default_factory=dict)
    image_size: Tuple[int, int] = (128, 128)

    # 动作缩放: (xyz_m/step, rotation_rad/step, gripper_scale)
    action_scale: Tuple[float, float, float] = (0.04, 0.2, 1.0)

    # 笛卡尔安全边界 (6D: x,y,z,rx,ry,rz)
    abs_pose_limit_low: np.ndarray = field(
        default_factory=lambda: np.array([0.2, -0.4, 0.0, -np.pi, -0.2, -0.2])
    )
    abs_pose_limit_high: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.4, 0.5, np.pi, 0.2, 0.2])
    )

    # 重置位姿 (6D: x,y,z,rx,ry,rz euler)
    reset_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.0, 0.3, np.pi, 0.0, 0.0])
    )
    random_reset: bool = False
    random_xy_range: float = 0.02
    random_rz_range: float = 0.0

    # 阻抗控制参数（通过 HTTP POST /update_param 设置）
    compliance_param: Dict[str, float] = field(default_factory=lambda: {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0,
    })
    precision_param: Dict[str, float] = field(default_factory=lambda: {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    })

    # Episode 控制
    max_episode_length: int = 200
    hz: int = 10
    gripper_sleep: float = 0.6
    joint_reset_period: int = 0  # N episode 做一次关节重置，0=不做

    # 奖励阈值 (pick-and-place)
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.08, 0.08, 0.05, 0.1, 0.1, 0.1])
    )
    target_pose: Optional[np.ndarray] = None  # 6D 目标位姿（可选，用于位姿阈值 reward）

    # 物块检测
    plate_position: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.02])
    )
    T_cam_to_robot: Optional[np.ndarray] = None  # 4x4 相机标定矩阵

    # 编码器偏差注入
    encoder_bias_config: Optional[EncoderBiasConfig] = None

    # 显示
    display_image: bool = True
