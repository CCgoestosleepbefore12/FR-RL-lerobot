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
    # (2026-04-27)，bbox=(187,211,383,407) = 196×196 正方形，覆盖 pick-place
    # 工作区域且裁掉下方无关背景。换 workspace 时重跑 select 脚本 → 改这里。
    image_crop: Dict[str, callable] = field(default_factory=lambda: {
        "front": make_workspace_roi_crop(187, 211, 383, 407),
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
    # 对齐 hil-serl franka_env 经验值 (0.05, 0.1)：rot/xyz 比例 2× 而非 6.7×，
    # 旋转维不会主导视觉动作。P0-3 不变性：action_scale[0] × hz == max_cart_speed
    # （0.05 × 10 == 0.50）→ 极限动作不被 clip 非线性压缩，policy 学到 = 实际执行。
    # ⚠️ gripper_scale 必须 ≥ 0.5 / 1.0 = 0.5：_send_gripper_command 用 ±0.5 阈值
    # 判 close/open，policy 输出 ±1（discrete head 解码后）× scale 必须能穿越阈值。
    # 调小到 < 0.5 会让 close/open 永远不触发硬件命令。
    action_scale: Tuple[float, float, float] = (0.05, 0.1, 1.0)

    # 笛卡尔安全边界 (6D: x,y,z,rx,ry,rz)
    # 来自 scripts/real/sample_workspace_bounds.py 16 点采样 (2026-04-25)，
    # 用 tour_workspace_corners.py 实地巡过 8 角确认机械臂可达。
    # z_low=0.175：桌面上方安全距离（实测 0.174 略往上+1mm 防夹爪撞桌面）。
    abs_pose_limit_low: np.ndarray = field(
        default_factory=lambda: np.array([0.386, -0.213, 0.175, -np.pi, -0.2, -0.2])
    )
    # z 上限 0.8m：现场工作台上方完全无遮挡（无相机/灯架/天花板障碍），
    # 远高于 Franka 最大 reach (~0.855m)，等价于 z 方向不约束。所有 task 共享这个
    # 宽松上限，每个 factory 不再 override z 维。lift headroom 检查 (real.py:108
    # `abs_pose_limit_high[2] - reset_pose[2] ≥ 0.10`) 自动满足。
    abs_pose_limit_high: np.ndarray = field(
        default_factory=lambda: np.array([0.709, 0.198, 0.8, np.pi, 0.2, 0.2])
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
    # 末端直线速度安全上限 (m/s)。配合 action_scale[0]=0.05 × 10Hz = 0.50 m/s，
    # 极限动作刚好不会被 clip → policy 学到的动作幅度与执行幅度一致。
    # ⚠️ 0.30 → 0.50 是真实安全侧改动：动量 +67%，撞击响应距离变长。建议放工件
    # 之前先空跑 reset 路径 + 几集随机 episode 确认轨迹仍在 abs_pose_limit_low/high
    # 内、夹爪不会撞桌。若 action_scale 调小回 0.03，记得同步把这个值降回 0.30。
    max_cart_speed: float = 0.50
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

    # BiasMonitor 数据 npz + 同名 png 输出路径前缀（None = 不存）。脚本侧
    # 通常设到 charts/bias_<timestamp>，BiasMonitor.close() 时一次性写入。
    bias_monitor_save_path: Optional[str] = None

    # Gripper 锁定模式：
    #   "none"   — 默认，env.step 用 action[6] 自由控制夹爪，reset 时张开
    #   "closed" — 夹爪永远闭合（如 wipe 任务夹住海绵），reset 时也闭合，
    #              env.step 忽略 action[6]，policy / SpaceMouse 的夹爪指令无效
    #   "open"   — 夹爪永远张开（如 push 任务），reset 时张开，step 忽略 action[6]
    # 锁定模式下 dataset_stats 的 action[6] 应固定为该模式的 const，避免
    # std=0 触发归一化 NaN。collect_demo 也对应屏蔽 SpaceMouse 夹爪按键。
    gripper_locked: str = "none"


# ============================================================
# Per-task config factories
# ------------------------------------------------------------
# 每个新任务在这里加一个 make_<task>_config()。collect_demo /
# 训练 env / 测试都通过 task name 拿配置，避免散在脚本里改字段。
# wipe 显式重传 random_reset=False / max_episode_length=300 与 dataclass
# 默认重复 — intentional：作为 task contract 锚点，dataclass 默认日后改了
# 也不影响 wipe 行为。
# ============================================================


def _make_j1_bias_cfg(bias_range: Tuple[float, float] = (-0.2, 0.2)) -> EncoderBiasConfig:
    """每次新建一份 J1 random_uniform bias config。

    不用 module-level singleton 是因为 EncoderBiasConfig 是普通 dataclass
    （非 frozen，target_joints 是可变 list），多个 cfg 共享同一引用时若有
    脚本 mutate（例如 ablation 里 cfg.encoder_bias_config.bias_range = ...）
    会跨 cfg 污染。每次 new 实例 cheap 而且消除 footgun。

    bias_range 默认 (-0.2, 0.2) 对齐 wipe baseline；pickup 用更窄的 (-0.1, 0.1)
    （pickup 是接触+抓取任务，bias 太大会让 demo 都失败）。
    """
    return EncoderBiasConfig(
        enable=True,
        error_probability=1.0,
        target_joints=[0],
        bias_mode="random_uniform",
        bias_range=bias_range,
    )


def make_wipe_config(
    use_bias: bool = True,
    reward_backend: str = "keyboard",
    enable_bias_monitor: bool = False,
    bias_monitor_save_path: Optional[str] = None,
) -> FrankaRealConfig:
    """Wipe sponge across plate — gripper 锁闭夹海绵，长 episode (300 step)。

    与 pickup 关键差异：
      - gripper_locked="closed"：海绵已被夹住，整局不开合 → action[6] 锁 -1
      - max_episode_length=300：30s @ 10Hz，覆盖完整擦拭来回轨迹
      - reset_pose=[0.4608, -0.0935, 0.2575, -3.078, -0.020, 0.018]：现场用 SpaceMouse
        摆到擦拭起点上方 hover 位置后从 /getstate_true 读出（2026-04-30 标定）
    abs_pose_limit 用 FrankaRealConfig 默认（z 上限 0.8m 不约束）。
    """
    cfg = FrankaRealConfig(
        reward_backend=reward_backend,
        gripper_locked="closed",
        max_episode_length=300,
        random_reset=False,
    )
    cfg.reset_pose = np.array([0.4608, -0.0935, 0.2575, -3.07817, -0.01998, 0.01770])
    # front 相机 ROI：select_workspace_roi.py 框选写回 (2026-04-30)
    # roi_front_final = (192, 136, 394, 338) — 202×202 正方形 wipe 工作面
    cfg.image_crop = {
        "front": make_workspace_roi_crop(192, 136, 394, 338),
        "wrist": center_square_crop,
    }
    if use_bias:
        cfg.encoder_bias_config = _make_j1_bias_cfg()
        cfg.enable_bias_monitor = enable_bias_monitor
        cfg.bias_monitor_save_path = bias_monitor_save_path
    return cfg


def make_pickup_config(
    use_bias: bool = True,
    reward_backend: str = "keyboard",
    enable_bias_monitor: bool = False,
    bias_monitor_save_path: Optional[str] = None,
) -> FrankaRealConfig:
    """Pickup sponge block — 夹爪自由控制，短 episode (80 step)，物块 reset xy 随机化。

    与 wipe 的关键差异：
      - gripper_locked="none"：抓取需要夹爪开合 → action[6] 自由
      - max_episode_length=100：10s @ 10Hz，对齐 hil-serl ram_insertion 的 100 步 horizon
      - random_reset=True + random_xy_range=0.05：物块位置随机的话，机械臂
        reset 也随机化能学 generalization
      - reset_pose=[0.5334, 0.0149, 0.2586, -3.098, 0.016, 0.014]：现场用 SpaceMouse
        摆到 pre-grasp hover 位置后从 /getstate_true 读出（quat→euler）。
        z=0.2586 给 pickup 抓取留抓 + 抬升余量。
      - rz 范围扩到 ±π/2（默认 ±0.2 约 ±11.5°）：让操作员能转 yaw 抓横放物块。
        pitch (ry) 保持默认 ±0.2 防夹爪侧倾撞桌；roll (rx) 保持默认 ±π 全自由。
    """
    cfg = FrankaRealConfig(
        reward_backend=reward_backend,
        gripper_locked="none",
        max_episode_length=100,
        # baseline pipeline 验证阶段：关 random_reset 让机械臂每次 reset 到完全相同
        # 位姿，排除 ±5cm 随机化 + settle 残差叠加导致的"位置抖动"。物块的位置
        # variation 由操作员手动放置提供。pipeline 跑通后恢复 True/0.05 引入自动
        # generalization。
        random_reset=False,
        random_xy_range=0.0,
    )
    cfg.reset_pose = np.array([0.5334, 0.0149, 0.2586, -3.09848, 0.01591, 0.01375])
    # rz 范围扩到 ±π/2 (±90°)：让操作员能转 yaw 抓横放物块。其他维（xyz / rx / ry）
    # 都用 FrankaRealConfig 默认（z 上限 0.8m 由默认覆盖，不再 task 级 override）。
    cfg.abs_pose_limit_low[5] = -np.pi / 2
    cfg.abs_pose_limit_high[5] = np.pi / 2
    # front 相机 ROI：calibration_data/workspace.json 里 select_workspace_roi.py
    # 框选写回的 roi_front_final = [180, 94, 400, 314] (220×220 正方形)，覆盖
    # pickup 工作面。runtime 不读 JSON，需在这里显式接上。
    cfg.image_crop = {
        "front": make_workspace_roi_crop(180, 94, 400, 314),
        "wrist": center_square_crop,
    }
    if use_bias:
        cfg.encoder_bias_config = _make_j1_bias_cfg(bias_range=(-0.1, 0.1))
        cfg.enable_bias_monitor = enable_bias_monitor
        cfg.bias_monitor_save_path = bias_monitor_save_path
    return cfg


def make_assembly_config(
    use_bias: bool = True,
    reward_backend: str = "keyboard",
    enable_bias_monitor: bool = False,
    bias_monitor_save_path: Optional[str] = None,
) -> FrankaRealConfig:
    """Assembly task — gripper 锁闭夹零件，从 hover 高位下降插入/对齐。

    与 pickup/wipe 关键差异：
      - gripper_locked="closed"：reset 时已预夹零件（gripper_pos≈0.33 ~26mm 持物开度）
      - reset_pose=[0.5067, -0.0197, 0.3600, -3.075, -0.011, -0.035]：现场用 SpaceMouse
        摆到装配面正上方 hover 位置后从 /getstate_true 读出（2026-04-30 标定）。
        z=0.36 比 pickup/wipe 高 ~10cm，给下降插入留余量
      - max_episode_length=200：20s @ 10Hz，多阶段对齐+插入
      - bias_range=(-0.05, 0.05)：装配是接触类任务，对 J1 bias 极其敏感，
        比 pickup 的 ±0.1 还窄一半
      - rz 范围 ±π/2：装配通常需要精确 yaw 对齐；如果你的零件只能小范围旋转
        对齐（比如花键、键槽），可以收窄
    abs_pose_limit 其他维用 FrankaRealConfig 默认（z 上限 0.8m 不约束）。
    """
    cfg = FrankaRealConfig(
        reward_backend=reward_backend,
        gripper_locked="closed",
        max_episode_length=200,
        random_reset=False,
    )
    cfg.reset_pose = np.array([0.5067, -0.0197, 0.3600, -3.07533, -0.01079, -0.03457])
    cfg.abs_pose_limit_low[5] = -np.pi / 2
    cfg.abs_pose_limit_high[5] = np.pi / 2
    # ⚠️ image_crop / 工作面 ROI 待 select_workspace_roi.py 框选后接入；当前用默认值
    if use_bias:
        cfg.encoder_bias_config = _make_j1_bias_cfg(bias_range=(-0.05, 0.05))
        cfg.enable_bias_monitor = enable_bias_monitor
        cfg.bias_monitor_save_path = bias_monitor_save_path
    return cfg


TASK_CONFIG_FACTORIES = {
    "wipe": make_wipe_config,
    "pickup": make_pickup_config,
    "assembly": make_assembly_config,
}


def make_task_config(task: str, **kwargs) -> FrankaRealConfig:
    """Dispatch by task name. 新任务在 TASK_CONFIG_FACTORIES 里注册即可。"""
    if task not in TASK_CONFIG_FACTORIES:
        raise ValueError(
            f"unknown task '{task}'; registered: {list(TASK_CONFIG_FACTORIES.keys())}"
        )
    return TASK_CONFIG_FACTORIES[task](**kwargs)
