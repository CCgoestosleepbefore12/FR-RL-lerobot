"""
Franka Panda 真机环境配置

参考 hil-serl 的 DefaultEnvConfig，适配 FR-RL 的观测格式。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from frrl.fault_injection import EncoderBiasConfig

# RealSense D455 物理 serial（hpc-old 工作站 2 台 D455，2026-04-23 物理确认）。
# 单相机脚本（calibration / workspace / aruco / tcp tracker / hand detector）必须
# 走 enable_device(FRONT_CAMERA_SERIAL)，否则双机插线后 pyrealsense 枚举顺序不可控，
# 会随机抓到 wrist。换相机时改这两个常量 + FrankaRealConfig.cameras 即可。
FRONT_CAMERA_SERIAL = "234222303420"
WRIST_CAMERA_SERIAL = "318122303303"


def center_square_crop(img: np.ndarray) -> np.ndarray:
    """对任意 H×W×C 图像做中心正方形裁剪：边长 = min(H, W)。"""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0 : y0 + s, x0 : x0 + s]


def workspace_roi_crop_placeholder(img: np.ndarray) -> np.ndarray:
    """workspace 标定完成前的 front ROI 占位：暂用中心正方形。

    跑完 scripts/real/define_workspace.py 后，把 cfg.image_crop["front"] 换成
    make_workspace_roi_crop(u0, v0, u1, v1) 即可。
    """
    return center_square_crop(img)


def make_workspace_roi_crop(u0: int, v0: int, u1: int, v1: int):
    """构造 front 相机的 workspace ROI crop 函数。

    参数为 define_workspace.py 采样+投影得到的像素 bbox（inclusive u0/v0，
    exclusive u1/v1，遵循 numpy slice 语义）。返回的闭包对 640×480 BGR 原
    始帧做 `img[v0:v1, u0:u1]`，输出交给下游 cv2.resize 到 image_size。

    bbox 非正方形也没关系：resize 会把它拉到目标尺寸；SAC encoder 只要求各
    相机输出 image_size 一致，不要求 crop 前等比。
    """
    def crop(img: np.ndarray) -> np.ndarray:
        return img[v0:v1, u0:u1]
    return crop


@dataclass
class FrankaRealConfig:
    """Franka Panda 真机环境配置。"""

    # Flask Server 连接（RT PC）
    # IP 必须匹配 RT PC 上 USB 网卡的静态地址。见 docs/rt_pc_runbook.md。
    server_url: str = "http://192.168.100.1:5000/"

    # RealSense D455 相机配置
    # front: 正前方俯视，承担全局定位 + 小目标（吸管）远距离识别 → 分辨率更高
    # wrist: 腕部末端视角，近距离细节 → 分辨率较低即可
    # Serial numbers 对应 hpc-old 工作站上 2 台 D455（用户 2026-04-23 物理确认）。
    # 换机器时修改这里或传自定义 config。
    cameras: Dict[str, dict] = field(default_factory=lambda: {
        "front": {"serial_number": FRONT_CAMERA_SERIAL, "dim": (640, 480), "fps": 15, "exposure": 40000},
        "wrist": {"serial_number": WRIST_CAMERA_SERIAL, "dim": (640, 480), "fps": 15, "exposure": 40000},
    })
    # front ROI 来自 scripts/hw_check/select_workspace_roi.py 鼠标手动框选
    # (2026-04-26)，bbox=(200,171,408,379) = 208×208 正方形，覆盖 pick-place
    # 工作区域且裁掉下方无关背景。换 workspace 时重跑 select 脚本 → 改这里。
    image_crop: Dict[str, callable] = field(default_factory=lambda: {
        "front": make_workspace_roi_crop(200, 171, 408, 379),
        "wrist": center_square_crop,
    })
    # SAC 架构 (frrl/policies/sac/modeling_sac.py:586) 要求所有相机同尺寸
    # (单一 image_encoder 批 forward 所有相机)。都用 128² 符合 HIL-SERL paper。
    # 如需高分辨率识别小目标，需先 refactor encoder 支持 per-camera shape。
    image_size: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "front": (128, 128),
        "wrist": (128, 128),
    })

    # 动作缩放: (xyz_m/step, rotation_rad/step, gripper_scale)
    # P0-3: 0.03 × 10Hz = 0.30 m/s 正好等于 max_cart_speed，避免 0.04 时极限动作
    # 被 clip_safety 0.75x 非线性压缩、policy 学错动作幅度信念。
    action_scale: Tuple[float, float, float] = (0.03, 0.2, 1.0)

    # 笛卡尔安全边界 (6D: x,y,z,rx,ry,rz)
    # 来自 scripts/real/sample_workspace_bounds.py 16 点采样 (2026-04-25)，
    # 用 tour_workspace_corners.py 实地巡过 8 角确认机械臂可达。
    # z_low=0.175：桌面上方安全距离（实测 0.174 略往上+1mm 防夹爪撞桌面）。
    abs_pose_limit_low: np.ndarray = field(
        default_factory=lambda: np.array([0.386, -0.213, 0.175, -np.pi, -0.2, -0.2])
    )
    abs_pose_limit_high: np.ndarray = field(
        default_factory=lambda: np.array([0.709, 0.198, 0.315, np.pi, 0.2, 0.2])
    )

    # 重置位姿 (6D: x,y,z,rx,ry,rz euler)
    # z=0.21：abs_pose_limit_high[2]=0.315 - 0.21 = 0.105m headroom >= real.py:108
    # 要求的 0.1m 抬升预算（reset 三段路径 lift→transit→descend 需要这个余量）。
    reset_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.0, 0.21, np.pi, 0.0, 0.0])
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
    max_episode_length: int = 300   # 30s @ 10Hz; 20s (200) 不够做 pick-place
    hz: int = 10
    # 末端直线速度安全上限 (m/s)。配合 action_scale[0]=0.03 × 10Hz = 0.30 m/s，
    # 极限动作刚好不会被 clip → policy 学到的动作幅度与执行幅度一致。
    # 若 action_scale 调大，runtime clip 会非线性压缩，建议同步调 max_cart_speed。
    max_cart_speed: float = 0.30
    gripper_sleep: float = 0.6
    joint_reset_period: int = 0  # N episode 做一次关节重置，0=不做

    # Reward 后端选择
    #   "pose":     依据 target_pose + reward_threshold 的位姿阈值（默认，兼容老脚本）
    #   "keyboard": HIL-SERL 风格人工按键（S/Enter/Space/Backspace），需要 pynput
    reward_backend: str = "pose"

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

    # Live view：每 step 在本地 imshow 两路相机的原始 640×480 BGR 帧（crop 前）。
    # Demo 采集 / online HIL 训练时都默认开，便于肉眼监控相机视角和手/物体位置。
    # 开销 ~3-5% 单核 CPU；headless（无 X11）会首次 cv2.error 后自动降级。
    display_image: bool = True

    # BiasMonitor：matplotlib 弹窗，2Hz 绘制每个带 bias joint 的 q_true vs
    # q_biased 波形（含 episode 边界竖线 + ep 号 + bias 数值标注）。env 仅在
    # encoder_bias_config 非空 & 本开关 True 时创建 BiasMonitor。默认关：
    # headless 训练不要 GUI，本地 demo / online HIL 时脚本侧显式置 True 即可。
    enable_bias_monitor: bool = False
