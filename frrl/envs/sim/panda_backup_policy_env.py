"""
Backup Policy 训练环境（TRACKING-only）。

场景：人手进入共享工作空间 → Backup Policy 激活 → 小幅闪避持续退让；
手不会主动撤走（"人拿手逼近末端观察反应"的对抗场景）。
"手撤走"由外部 Supervisor 按时间或实际检测触发，属于真机部署/集成阶段的事情。

运动模式：TRACKING（单一）
  - 每步方向重算为 normalize(current_TCP - obstacle_pos)，手会追当前末端而不是 spawn 时末端
  - 贴到 D_TIGHT(8cm) 内后停顿 3-8 步模拟"怼脸"，然后继续追
  - TCP 退让时手立刻跟进，policy 必须学会在持续压迫下保持安全距离

场景变体：
  S1: 1 个移动障碍物（主用）
  S2: 2 个移动障碍物（保留用于多手对抗实验）

观测：(robot_state(18) + num_obstacles × obstacle_info(10)) × 3帧堆叠
  S1: 84D (28×3), S2: 114D (38×3)

奖励（参考 Kiemel et al. 2024，每步奖励非负）：
  终止: collision / block_drop / zone_c / displacement>15cm → -10.0
  存活每步: +0.5 - 0.5*位移 - 0.01*动作幅度 - 0.01*动作变化
  存活完整 episode: +5.0 bonus

Domain Randomization（仅观测层）：
  障碍物位置噪声 N(0, 0.03) / 速度噪声 N(0, 0.01) / 观测延迟 U(0, 2) 步
  障碍物速度 U(0.005, 0.015) m/step / TCP 位置噪声 N(0, 0.005)

参考: Kiemel et al. "Safe RL of Robot Trajectories in the Presence of Moving Obstacles", IEEE RAL 2024
"""

from collections import deque
from enum import Enum, auto
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.sim.panda_pick_place_safe_env import (
    PandaPickPlaceSafeEnv,
    HAND_COLLISION_DIST,
    ZONE_C_MIN_Y,
)
from frrl.envs.sim.base import GymRenderingSpec
from frrl.fault_injection import EncoderBiasConfig

# ── 常量 ──────────────────────────────────────────────────
BLOCK_DROP_Z = -0.01
MAX_EPISODE_STEPS = 20           # TRACKING 持续压迫场景，episode 延长（原 10）
ACTION_SCALE = 0.03              # 位置增量缩放 m/step
ROT_ACTION_SCALE = 0.1           # 姿态增量缩放 rad/step（≈5.7°/step）

# 障碍物生成
HAND_SPAWN_DIST = (0.15, 0.30)   # V1 相对 TCP；V2 使用 ARM_SPAWN_DIST
ARM_SPAWN_DIST = (0.21, 0.30)    # V2 相对 arm_center；下限 >1.5×ARM_COLLISION_DIST (0.2025m) 防 1-step death
HAND_SPEED_RANGE = (0.005, 0.015)
HIDDEN_POS = np.array([0.4, 0.5, 0.05])

# TRACKING 参数
D_TIGHT = 0.08                   # 贴近 TCP 的停顿距离（"怼脸"）
STALL_STEPS_RANGE = (3, 8)       # 每次贴近后停顿的步数范围

HAND_BODY_NAMES = ["human_hand", "human_hand_2", "human_hand_3"]
HAND_GEOM_NAMES = ["human_hand", "human_hand_2", "human_hand_3"]

# 机械臂初始随机位置范围（自然操作区，不绑定工作空间边距）
# V2 训练时通常配合 enforce_cartesian_bounds=False，允许 policy 沿 -hand_dir 直线退让
TCP_INIT_X = (0.30, 0.55)
TCP_INIT_Y = (-0.30, 0.10)
TCP_INIT_Z = (0.15, 0.40)

# 奖励参数
SURVIVAL_REWARD = 0.5
DISPLACEMENT_COEFF = 0.5
ACTION_NORM_COEFF = 0.01
ACTION_SMOOTH_COEFF = 0.01
TERMINATION_PENALTY = -10.0
SURVIVAL_BONUS = 5.0
MAX_DISPLACEMENT = 0.15

# V2 防作弊：腕+手单球避障 + 旋转软惩罚
# arm_sphere 球心 = mocap weld 点，对绕该点旋转 rigid → 转腕无法改变球位置 →
# 旋转不可能用于"躲避作弊"，硬上限只会挡住几何必要的换 IK 构型（如手从上方来、
# 机械臂限位无法下退时必须转腕切解）。因此旋转预算改为纯软惩罚：
# - MAX_ROTATION=π 等效关闭硬终止
# - ROTATION_COEFF 从 0.5 降到 0.2，仍保留 "能不转就不转" 的偏置
ARM_SPHERE_RADIUS = 0.10              # m，包住 wrist+hand 的球半径
ARM_COLLISION_DIST = ARM_SPHERE_RADIUS + 0.035   # + obstacle 球半径 = 0.135m
ROTATION_COEFF = 0.2                  # 旋转软惩罚系数
MAX_ROTATION = np.pi                  # ≈180°，等效关闭硬上限

# V3 全臂多球避障：解决 "policy 只学 EE 避让，肘/前臂仍会撞" 的问题
# 沿 Franka 链 5 个 link body 各放一个球，min over 算最近距离；
# obstacle 半径放大到 0.10 对齐真机 hand bbox 等效半径 (~10cm 外接球)，
# 让 sim 训练边界 = 真机最坏 case (开掌伸出) 接触边界。
ARM_SPHERES_V3 = [
    ("link3", 0.07),                  # 上臂
    ("link4", 0.07),                  # 肘 (最容易被撞)
    ("link5", 0.065),                 # 前臂
    ("link6", 0.065),                 # 腕前
    ("hand",  0.10),                  # 腕+手 (= ARM_SPHERE_RADIUS, 沿用)
]
OBSTACLE_RADIUS_V3 = 0.10             # m，对齐真机 WiLoR bbox 等效半径
ARM_COLLISION_DIST_V3 = 0.10 + OBSTACLE_RADIUS_V3   # = 0.20m（panda_hand 球的最严阈值，
                                                     # 5 球中半径最大者 0.10 + obstacle 0.10）
ARM_SPAWN_DIST_V3 = (0.30, 0.40)      # 相对 panda_hand body；下限 = 1.5×ARM_COLLISION_DIST_V3
                                       #   = 1.5 × 0.20 = 0.30，配合 _spawn_obstacle_pos 的
                                       #   1.5× 拒绝采样系数刚好不留 1-step death 风险
HAND_SPEED_RANGE_V3 = (0.015, 0.030)  # m/step，对齐真机人手 15-30cm/s 区间
MAX_DISPLACEMENT_V3 = 0.50            # 给 policy 更大退让预算 (V2 是 0.30; V3 首训 71.5% 后
                                       # 0.40 → 0.50，针对 14.5% excessive_displacement 失效模式)

# V3 proximity reward：诱导 "维持 ~10cm 表面间隙"
# - clear=0: 0 (碰撞，紧接 -10 终止)
# - clear<10cm: 线性增长，梯度 PROXIMITY_REWARD_MAX/PROXIMITY_SAFE_DIST = 2.0/m
# - clear>=10cm: 饱和 +0.20 (退到此处即可，再退只剩 disp penalty)
PROXIMITY_REWARD_MAX = 0.20
PROXIMITY_SAFE_DIST  = 0.10

# DR 噪声参数
DR_OBS_POS_STD = 0.03
DR_OBS_VEL_STD = 0.01
DR_TCP_POS_STD = 0.005
DR_DELAY_MAX = 2
DR_POS_HISTORY_LEN = 5
OBS_STACK_SIZE = 3


class MotionMode(Enum):
    """障碍物运动模式。当前仅实现 TRACKING（追踪当前 TCP）。"""
    TRACKING = auto()


class PandaBackupPolicyEnv(PandaPickPlaceSafeEnv):
    """Backup Policy 训练环境（TRACKING-only）。

    Args:
        num_obstacles: 障碍物数量（1=S1, 2=S2），所有障碍均为 TRACKING 模式
        enable_dr: 是否启用 Domain Randomization
        max_displacement: TCP 相对 episode 起点位移硬上限 (m)，超过即 terminate。
            0.15（默认）为严格版本；0.20 为放宽版本（给 policy 更大退让预算）。
        survival_bonus: 存活完整 episode（truncated）时追加的终局 bonus。
        use_arm_sphere_collision: True → 用腕+手单球（半径 10cm，球心在 mocap weld 点）
            做唯一避障碰撞体；False（默认）→ 保留 TCP+左右指尖三检测点。V2 防作弊开关。
        use_full_arm_collision: True → V3 模式，沿臂 5 球 (link3/4/5/6/hand) 全臂避障，
            obstacle 半径放大到 0.10 对齐真机 hand bbox，spawn 距离 (0.30, 0.40)，
            hand 速度 (0.015, 0.030) m/step，引入 proximity reward 诱导维持 ~10cm
            表面间隙。要求 use_arm_sphere_collision=True（V3 在 V2 panda_hand 球基础上扩展）。
        max_rotation: 姿态旋转预算硬上限 (rad)，当 use_arm_sphere_collision=True 时生效。
            超过即 terminate（防"转手腕躲避"作弊）。
        rotation_coeff: 旋转幅度负奖励系数（与 DISPLACEMENT_COEFF 对称），仅在
            use_arm_sphere_collision=True 时启用。
        enforce_cartesian_bounds: 是否启用工作空间硬 clamp（默认 True，与真机一致）。
            V2 sim 训练可设 False，移除工作空间束缚让 policy 学沿 -hand_dir 直线退让；
            真机部署时由 Homing/Supervisor 层另行提供 clamp。
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        num_obstacles: int = 1,
        enable_dr: bool = True,
        obs_stack: int = OBS_STACK_SIZE,
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
        max_displacement: float = MAX_DISPLACEMENT,
        survival_bonus: float = SURVIVAL_BONUS,
        use_arm_sphere_collision: bool = False,
        use_full_arm_collision: bool = False,
        max_rotation: float = MAX_ROTATION,
        rotation_coeff: float = ROTATION_COEFF,
        enforce_cartesian_bounds: bool = True,
    ):
        self._num_obstacles = min(num_obstacles, 2)
        self._enable_dr = enable_dr
        self._obs_stack = max(obs_stack, 1)
        self._max_displacement = float(max_displacement)
        self._survival_bonus = float(survival_bonus)
        self._use_arm_sphere_collision = bool(use_arm_sphere_collision)
        self._use_full_arm_collision = bool(use_full_arm_collision)
        if self._use_full_arm_collision and not self._use_arm_sphere_collision:
            raise ValueError(
                "use_full_arm_collision=True requires use_arm_sphere_collision=True "
                "(V3 builds on V2 panda_hand sphere)."
            )
        self._max_rotation = float(max_rotation)
        self._rotation_coeff = float(rotation_coeff)
        self._enforce_cartesian_bounds = bool(enforce_cartesian_bounds)

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            hand_appear_prob=1.0,
            hand_duration_range=(999, 1000),
            hand_appear_step_range=(0, 1),
            hand_reappear_interval=(999, 1000),
            safety_layer_enabled=False,
            encoder_bias_config=encoder_bias_config,
        )

        # sim 训练时可关闭工作空间硬 clamp，让 policy 学习沿 -hand_dir 直线退让
        # （真机部署层另行提供 clamp，训练 env 无此负担）
        if not self._enforce_cartesian_bounds:
            self._cartesian_bounds = np.asarray(
                [[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]]
            )

        # MuJoCo body/geom ID
        self._obstacle_body_ids = [self._model.body(n).id for n in HAND_BODY_NAMES]
        self._obstacle_geom_ids = [self._model.geom(n).id for n in HAND_GEOM_NAMES]
        # V2 腕+手避障的球心：mocap weld 点 (panda hand body)
        self._panda_hand_body_id = self._model.body("hand").id
        # V3 全臂多球：解析 link3/4/5/6 + hand 各球的 body_id 和半径
        self._arm_sphere_specs: list[Tuple[int, float]] = []
        if self._use_full_arm_collision:
            self._arm_sphere_specs = [
                (self._model.body(name).id, float(r)) for name, r in ARM_SPHERES_V3
            ]

        # 障碍物状态
        self._obstacle_pos = [HIDDEN_POS.copy() for _ in range(self._num_obstacles)]
        self._obstacle_prev_pos = [HIDDEN_POS.copy() for _ in range(self._num_obstacles)]
        self._obstacle_speed = 0.015  # 当前 episode 的追踪速度
        self._stall_remaining = [0] * self._num_obstacles  # 每个障碍剩余停顿步数

        # 运动模式（当前仅 TRACKING，保留枚举以便未来扩展）
        self._motion_mode = MotionMode.TRACKING

        # DR: 位置历史缓冲区（用于模拟观测延迟）
        self._pos_histories: list[deque] = [
            deque(maxlen=DR_POS_HISTORY_LEN) for _ in range(self._num_obstacles)
        ]

        # 奖励辅助状态
        self._tcp_start = np.zeros(3)
        self._tcp_start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # shape: (4,) [w,x,y,z]
        self._action_prev = np.zeros(6, dtype=np.float32)
        # V3 当步表面间隙（min over arm_spheres of (center_dist − sphere_r − obstacle_r)）
        # 仅 V3 使用；step() 内由 _check_multi_safety 写入，再被 proximity reward 读取
        self._step_surface_clearance = float("inf")

        # 帧堆叠
        self._single_obs_dim = 18 + self._num_obstacles * 10
        self._obs_history: deque = deque(maxlen=self._obs_stack)

        # 观测空间
        agent_dim = self._single_obs_dim * self._obs_stack
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"pixels": spaces.Dict({
                "front": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                "wrist": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
            }), "agent_pos": agent_box} if self.image_obs
            else {"agent_pos": agent_box}
        )

        # 动作 6D（夹爪在 step 内强制为 0）
        self.action_space = spaces.Box(-np.ones(6, dtype=np.float32), np.ones(6, dtype=np.float32))

    # ================================================================
    # Gym API
    # ================================================================

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 先初始化障碍物状态（父类 reset 会调用 _compute_observation）
        self._obstacle_speed = 0.015
        for i in range(self._num_obstacles):
            self._obstacle_pos[i] = HIDDEN_POS.copy()
            self._obstacle_prev_pos[i] = HIDDEN_POS.copy()
            self._stall_remaining[i] = 0
            self._pos_histories[i].clear()
            self._pos_histories[i].append(HIDDEN_POS.copy())
        self._tcp_start = np.zeros(3)
        self._tcp_start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._action_prev = np.zeros(6, dtype=np.float32)
        self._step_surface_clearance = float("inf")
        self._obs_history.clear()

        super().reset(seed=seed)

        # 机械臂随机位置 + 方块稳定夹持
        self._init_robot_with_block()

        tcp = self._data.site_xpos[self._pinch_site_id].copy()
        self._tcp_start = tcp.copy()
        # 记录 episode 起点末端姿态（V2 旋转预算基准）
        self._tcp_start_quat = np.asarray(self._data.mocap_quat[0], dtype=np.float64).copy()

        # 追踪速度 per-episode 随机；V3 用更快的速度区间匹配真机
        speed_range = HAND_SPEED_RANGE_V3 if self._use_full_arm_collision else HAND_SPEED_RANGE
        self._obstacle_speed = self._random.uniform(*speed_range)

        # 所有障碍都采用 TRACKING 模式；spawn 基点 + 距离范围 + 安全检查按 V3/V2/V1 三档
        safety_spheres: Optional[list] = None
        if self._use_full_arm_collision:
            # V3: spawn 基点为 panda_hand body，但安全距离对所有 5 个臂球都验证
            spawn_center = self._data.xpos[self._panda_hand_body_id].copy()
            spawn_range = ARM_SPAWN_DIST_V3
            spawn_collision_dist = 0.0    # safety_spheres 路径走多球检查，此值不被使用
            safety_spheres = [
                (self._data.xpos[bid].copy(), r + OBSTACLE_RADIUS_V3)
                for bid, r in self._arm_sphere_specs
            ]
        elif self._use_arm_sphere_collision:
            spawn_center = self._data.xpos[self._panda_hand_body_id].copy()
            spawn_range = ARM_SPAWN_DIST
            spawn_collision_dist = ARM_COLLISION_DIST
        else:
            spawn_center = tcp
            spawn_range = HAND_SPAWN_DIST
            spawn_collision_dist = HAND_COLLISION_DIST
        for i in range(self._num_obstacles):
            self._obstacle_pos[i] = self._spawn_obstacle_pos(
                spawn_center, spawn_range, spawn_collision_dist,
                safety_spheres=safety_spheres,
            )
            self._obstacle_prev_pos[i] = self._obstacle_pos[i].copy()

        # 重置 DR 位置历史
        for i in range(self._num_obstacles):
            self._pos_histories[i].clear()
            self._pos_histories[i].append(self._obstacle_pos[i].copy())

        self._hand_active = True
        self._sync_all_visuals()

        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        action_6d = np.asarray(action, dtype=np.float32).flat[:6]

        # 每个障碍都追踪当前 TCP
        for i in range(self._num_obstacles):
            self._obstacle_prev_pos[i] = self._obstacle_pos[i].copy()
            self._move_obstacle(i)

        # 更新 DR 位置历史
        for i in range(self._num_obstacles):
            self._pos_histories[i].append(self._obstacle_pos[i].copy())

        self._sync_all_visuals()

        # 执行动作：位置增量由父类 apply_action 处理；
        # 姿态增量父类忽略，这里先手动应用到 mocap_quat 再调父类。
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:3] = action_6d[:3] * ACTION_SCALE
        rot_delta = action_6d[3:6] * ROT_ACTION_SCALE                # shape: (3,) axis-angle, rad
        self._apply_rotation_to_mocap(rot_delta)
        # rx/ry/rz 父类本不处理，显式清零保险；gripper 保持 0（锁定）
        self.apply_action(full_action)

        # 安全检测
        terminated, safety_info = self._check_multi_safety()

        # 方块掉落
        if not terminated and self._data.sensor("block_pos").data[2] < BLOCK_DROP_Z:
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "block_dropped"}

        # 位移硬上限检测
        tcp = self._data.site_xpos[self._pinch_site_id].copy()
        displacement = float(np.linalg.norm(tcp - self._tcp_start))
        if not terminated and displacement > self._max_displacement:
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "excessive_displacement"}

        # V2 旋转预算检测（仅启用时）
        rotation = self._compute_rotation_angle() if self._use_arm_sphere_collision else 0.0
        if (
            not terminated
            and self._use_arm_sphere_collision
            and rotation > self._max_rotation
        ):
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "excessive_rotation"}

        truncated = self._step_count >= MAX_EPISODE_STEPS

        if terminated:
            reward = TERMINATION_PENALTY
        else:
            action_norm = float(np.linalg.norm(action_6d))
            action_diff = float(np.linalg.norm(action_6d - self._action_prev))

            reward = (
                SURVIVAL_REWARD
                - DISPLACEMENT_COEFF * displacement
                - ACTION_NORM_COEFF * action_norm
                - ACTION_SMOOTH_COEFF * action_diff
            )
            if self._use_arm_sphere_collision:
                reward -= self._rotation_coeff * rotation
            if self._use_full_arm_collision:
                # V3 proximity reward：诱导维持 ~10cm 表面间隙；超过即饱和
                clear = max(0.0, self._step_surface_clearance)
                proximity = PROXIMITY_REWARD_MAX * min(1.0, clear / PROXIMITY_SAFE_DIST)
                reward += proximity
            if truncated:
                reward += self._survival_bonus

        self._action_prev = action_6d.copy()

        obs = self._compute_observation()
        info = {
            "succeed": False,
            "hand_collisions": self._hand_collisions,
            "min_hand_dist": self._min_hand_dist,
            "displacement": displacement,
            "rotation": rotation,
            "num_obstacles": self._num_obstacles,
            "motion_mode": self._motion_mode.name,
            **safety_info,
        }
        # V3 才有意义的字段；V1/V2 路径下 _step_surface_clearance 永远 inf，避免下游
        # logger/jsonl/BiasMonitor 序列化脏值
        if self._use_full_arm_collision:
            info["step_surface_clearance"] = self._step_surface_clearance
        return obs, reward, terminated, truncated, info

    # ================================================================
    # 障碍物运动（TRACKING）
    # ================================================================

    def _move_obstacle(self, idx: int):
        """TRACKING：追踪当前 TCP，贴近后停顿。

        逻辑：
          1. 若仍在停顿计数内，本步不动
          2. 每步重算方向 = normalize(current_TCP - obstacle_pos)
          3. 若距离 < D_TIGHT，触发新一轮停顿（3-8 步），本步不动
          4. 否则沿方向前进 obstacle_speed（受工作空间 clip）
        """
        if self._stall_remaining[idx] > 0:
            self._stall_remaining[idx] -= 1
            return

        tcp = self._data.site_xpos[self._pinch_site_id].copy()  # shape: (3,)
        to_tcp = tcp - self._obstacle_pos[idx]                  # shape: (3,)
        dist = float(np.linalg.norm(to_tcp))

        if dist < D_TIGHT:
            self._stall_remaining[idx] = int(self._random.randint(*STALL_STEPS_RANGE))
            return

        direction = to_tcp / (dist + 1e-8)                       # shape: (3,)
        self._obstacle_pos[idx] += direction * self._obstacle_speed
        self._obstacle_pos[idx] = np.clip(self._obstacle_pos[idx], *self._cartesian_bounds)

    # ================================================================
    # 内部方法
    # ================================================================

    def _init_robot_with_block(self):
        """机械臂随机位置 + 方块稳定夹持。"""
        random_tcp = np.array([
            self._random.uniform(*TCP_INIT_X),
            self._random.uniform(*TCP_INIT_Y),
            self._random.uniform(*TCP_INIT_Z),
        ])
        self._data.mocap_pos[0] = np.clip(random_tcp, *self._cartesian_bounds)

        zero_action = np.zeros(7, dtype=np.float32)
        for _ in range(5):
            self.apply_action(zero_action)

        self._data.ctrl[self._gripper_ctrl_id] = 0  # Franka Hand: 0=关闭
        for _ in range(5):
            self.apply_action(zero_action)

        tcp_pos = self._data.site_xpos[self._pinch_site_id].copy()
        self._data.jnt("block").qpos[:3] = tcp_pos
        self._data.jnt("block").qpos[3:] = [1, 0, 0, 0]
        self._data.jnt("block").qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        for _ in range(10):
            self.apply_action(zero_action)

    def _spawn_obstacle_pos(
        self,
        center: np.ndarray,
        dist_range: Tuple[float, float],
        collision_dist: float,
        safety_spheres: Optional[list] = None,
    ) -> np.ndarray:
        """在指定中心附近球面随机方向生成障碍物位置。

        Args:
            center: 采样基点（V1=TCP，V2/V3=panda_hand body xpos）
            dist_range: 采样距离范围 (m)
            collision_dist: 单球碰撞阈值（V1/V2 用 1.5× 该值做拒绝采样）
            safety_spheres: V3 用，list of (sphere_pos, threshold)；
                若提供，则要求采样位置离**所有**臂球都 > 1.5× 阈值（防 1-step death）。

        失败回退：20 次拒绝采样仍不满足时，沿最远方向 fallback 到 spawn 上限处
        （`center + dir × dist_range[1]`），保证起码不在 spawn 中心球内；warn 一次。
        """
        last_pos = center.copy()
        last_dir = np.array([1.0, 0.0, 0.0])
        success = False
        for _ in range(20):
            raw_dir = self._random.normal(size=3)
            raw_dir /= np.linalg.norm(raw_dir) + 1e-8
            dist = self._random.uniform(*dist_range)
            pos = center + raw_dir * dist
            pos = np.clip(pos, *self._cartesian_bounds)
            last_pos = pos
            last_dir = raw_dir
            if safety_spheres is not None:
                # V3: 对所有臂球都做安全距离检查
                if all(
                    np.linalg.norm(pos - sp_pos) > thr * 1.5
                    for sp_pos, thr in safety_spheres
                ):
                    success = True
                    break
            else:
                if np.linalg.norm(pos - center) > collision_dist * 1.5:
                    success = True
                    break
        if not success:
            # Fallback：用最后一次方向 + spawn 上限，至少远离 spawn 中心
            import warnings
            warnings.warn(
                f"_spawn_obstacle_pos: 20 retries failed (safety_spheres={safety_spheres is not None}); "
                f"fallback to far end of dist_range. May indicate workspace too tight or arm pose extreme.",
                stacklevel=2,
            )
            fallback_pos = center + last_dir * dist_range[1]
            last_pos = np.clip(fallback_pos, *self._cartesian_bounds)
        return last_pos

    def _check_multi_safety(self) -> Tuple[bool, Dict[str, Any]]:
        """碰撞检测：

        V1（默认）：TCP + 左右指尖三检测点，阈值 HAND_COLLISION_DIST。
        V2（use_arm_sphere_collision=True）：腕+手单球，球心在 panda hand body
            （= mocap weld 点，对旋转 rigid），阈值 ARM_COLLISION_DIST。
        V3（use_full_arm_collision=True）：5 球沿臂 (link3/4/5/6/hand)，逐球
            阈值 = sphere_r + OBSTACLE_RADIUS_V3，min over 5 球做最近距离 +
            碰撞判定。同时写 _step_surface_clearance 供 proximity reward 读取。
        """
        tcp = self._data.site_xpos[self._pinch_site_id].copy()

        terminated = False
        safety_info = {"safety_violation": False, "violation_type": None}

        # C 区入侵
        if tcp[1] > ZONE_C_MIN_Y:
            self._zone_c_violations += 1
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "zone_c_intrusion"}
            return terminated, safety_info

        if self._use_full_arm_collision:
            # V3：5 球沿臂迭代，min over center distance + min over surface clearance
            step_surface_clear = float("inf")
            for bid, sphere_r in self._arm_sphere_specs:
                sphere_pos = self._data.xpos[bid].copy()
                threshold = sphere_r + OBSTACLE_RADIUS_V3
                for i in range(self._num_obstacles):
                    d = float(np.linalg.norm(sphere_pos - self._obstacle_pos[i]))
                    self._min_hand_dist = min(self._min_hand_dist, d)
                    surface_clear = d - threshold
                    step_surface_clear = min(step_surface_clear, surface_clear)
                    if not terminated and d < threshold:
                        self._hand_collisions += 1
                        terminated = True
                        safety_info = {"safety_violation": True, "violation_type": "hand_collision"}
            self._step_surface_clearance = step_surface_clear
        elif self._use_arm_sphere_collision:
            # V2：单球避障，球心 = panda hand body xpos（rotation invariant）
            arm_sphere_center = self._data.xpos[self._panda_hand_body_id].copy()
            collision_dist = ARM_COLLISION_DIST
            for i in range(self._num_obstacles):
                d = float(np.linalg.norm(arm_sphere_center - self._obstacle_pos[i]))
                self._min_hand_dist = min(self._min_hand_dist, d)
                if not terminated and d < collision_dist:
                    self._hand_collisions += 1
                    terminated = True
                    safety_info = {"safety_violation": True, "violation_type": "hand_collision"}
        else:
            # V1：TCP + 左右指尖三检测点
            check_points = [tcp, self._data.xpos[self._left_pad_body_id], self._data.xpos[self._right_pad_body_id]]
            for i in range(self._num_obstacles):
                min_d = min(float(np.linalg.norm(pt - self._obstacle_pos[i])) for pt in check_points)
                self._min_hand_dist = min(self._min_hand_dist, min_d)
                if not terminated and min_d < HAND_COLLISION_DIST:
                    self._hand_collisions += 1
                    terminated = True
                    safety_info = {"safety_violation": True, "violation_type": "hand_collision"}

        return terminated, safety_info

    def _sync_all_visuals(self):
        """同步障碍物位置到 MuJoCo 渲染。"""
        for i in range(len(HAND_BODY_NAMES)):
            if i < self._num_obstacles:
                self._model.body_pos[self._obstacle_body_ids[i]] = self._obstacle_pos[i]
                self._model.geom_rgba[self._obstacle_geom_ids[i]][3] = 0.8
            else:
                self._model.body_pos[self._obstacle_body_ids[i]] = HIDDEN_POS
                self._model.geom_rgba[self._obstacle_geom_ids[i]][3] = 0.0

    def _update_human_hand(self):
        pass

    # ================================================================
    # 姿态增量（父类 apply_action 忽略 rx/ry/rz，这里补齐）
    # ================================================================

    def _apply_rotation_to_mocap(self, axis_angle: np.ndarray) -> None:
        """把 axis-angle 增量（已乘过 ROT_ACTION_SCALE）叠加到 mocap_quat。

        Args:
            axis_angle: shape (3,)，单位 rad。向量方向 = 旋转轴，模长 = 旋转角。
        """
        angle = float(np.linalg.norm(axis_angle))
        if angle < 1e-8:
            return
        axis = axis_angle / angle                                     # shape: (3,) 单位向量
        half = 0.5 * angle
        sin_h = float(np.sin(half))
        dquat = np.array(
            [np.cos(half), axis[0] * sin_h, axis[1] * sin_h, axis[2] * sin_h],
            dtype=np.float64,
        )                                                              # shape: (4,) [w,x,y,z]
        q = np.asarray(self._data.mocap_quat[0], dtype=np.float64)    # shape: (4,)
        new_q = self._quat_multiply(q, dquat)
        new_q /= np.linalg.norm(new_q) + 1e-12
        self._data.mocap_quat[0] = new_q

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton 四元数乘法（输入/输出均 [w, x, y, z]）。shape: (4,) × (4,) → (4,)。"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        """单位四元数的共轭 = 其逆。shape: (4,) → (4,)，[w,x,y,z]。"""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    def _compute_rotation_angle(self) -> float:
        """当前末端姿态相对 episode 起点姿态的旋转角 (rad)。

        q_err = q_start⁻¹ · q_cur；rotation = 2·arccos(|q_err.w|)，
        取 |w| 消除双覆盖歧义。"""
        q_cur = np.asarray(self._data.mocap_quat[0], dtype=np.float64)       # shape: (4,)
        q_err = self._quat_multiply(self._quat_conjugate(self._tcp_start_quat), q_cur)
        w_clamped = float(np.clip(abs(q_err[0]), 0.0, 1.0))
        return 2.0 * float(np.arccos(w_clamped))

    # ================================================================
    # 观测
    # ================================================================

    def _compute_observation(self) -> Dict[str, Any]:
        # 使用基类方法获取 robot_state，确保编码器偏差正确注入
        robot_state = self.get_robot_state()                                     # shape: (18,)
        # TCP 位置取 robot_state 的最后 3 维（已含编码器偏差），加 DR 噪声
        noisy_tcp = robot_state[15:18] + self._random.normal(0, DR_TCP_POS_STD, 3).astype(np.float32)
        robot_state[15:18] = noisy_tcp

        obstacle_obs = []
        for i in range(self._num_obstacles):
            if self._enable_dr:
                delay = self._random.randint(0, min(DR_DELAY_MAX, len(self._pos_histories[i]) - 1) + 1)
                delayed_pos = self._pos_histories[i][-(1 + delay)]
                obs_pos = (delayed_pos + self._random.normal(0, DR_OBS_POS_STD, 3)).astype(np.float32)

                if len(self._pos_histories[i]) >= 2 + delay:
                    prev_delayed = self._pos_histories[i][-(2 + delay)]
                else:
                    prev_delayed = delayed_pos
                obs_vel = ((delayed_pos - prev_delayed) / self._control_dt).astype(np.float32)
                obs_vel += self._random.normal(0, DR_OBS_VEL_STD, 3).astype(np.float32)
            else:
                obs_pos = self._obstacle_pos[i].astype(np.float32)
                obs_vel = ((self._obstacle_pos[i] - self._obstacle_prev_pos[i]) / self._control_dt).astype(np.float32)

            obs_i = np.concatenate([
                [1.0],                                          # active
                obs_pos,                                        # pos (3,)
                obs_vel,                                        # vel (3,)
                (obs_pos - noisy_tcp).astype(np.float32),       # rel_pos (3,)
            ]).astype(np.float32)                                # shape: (10,)
            obstacle_obs.append(obs_i)

        single_obs = np.concatenate([robot_state, *obstacle_obs])  # shape: (28,) or (38,)

        # 帧堆叠
        self._obs_history.append(single_obs)
        if len(self._obs_history) < self._obs_stack:
            padding = [np.zeros_like(single_obs)] * (self._obs_stack - len(self._obs_history))
            agent_pos = np.concatenate([*padding, *self._obs_history])
        else:
            agent_pos = np.concatenate(list(self._obs_history))

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": agent_pos}
        return {"agent_pos": agent_pos}

    def _is_success(self) -> bool:
        return False
