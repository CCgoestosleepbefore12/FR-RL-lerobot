"""
Backup Policy 训练环境

场景：人手进入共享工作空间 → Backup Policy 激活 → 小幅闪避 → 人手离开后交还 Task Policy。

场景变体：
  S1: 1 个移动障碍物，4 种运动模式随机
  S2: 1 个移动障碍物 + 1 个静止障碍物

观测：robot_state(18) + num_obstacles × obstacle_info(10)
  S1: 28D, S2: 38D

运动模式：
  LINEAR   — 匀速直线朝 TCP
  ARC      — 带曲率弧线绕向 TCP
  STOP_GO  — 朝 TCP 但随机停顿
  PASSING  — 横穿工作空间，不朝 TCP

奖励（参考 Kiemel et al. 2024，每步奖励非负）：
  终止: collision / block_drop / zone_c / displacement>15cm → -10.0
  存活每步: +0.5 - 0.5*位移 - 0.01*动作幅度 - 0.01*动作变化
  存活完整 episode: +5.0 bonus
  discount = 1.0

Domain Randomization（仅观测层，不影响碰撞检测）：
  障碍物位置噪声 N(0, 0.03)   ← MediaPipe + D455 精度
  障碍物速度噪声 N(0, 0.01)   ← 帧间差分抖动
  观测延迟 U(0, 2) 步          ← 检测延迟
  障碍物速度 U(0.008, 0.025)  ← 不同手速
  TCP 位置噪声 N(0, 0.005)    ← 已有

参考: Kiemel et al. "Safe RL of Robot Trajectories in the Presence of Moving Obstacles", IEEE RAL 2024
"""

from collections import deque
from enum import Enum, auto
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.panda_pick_place_safe_env import (
    PandaPickPlaceSafeEnv,
    HAND_COLLISION_DIST,
    ZONE_C_MIN_Y,
    _CARTESIAN_BOUNDS,
)
from frrl.envs.base import GymRenderingSpec
from frrl.fault_injection import EncoderBiasConfig

# ── 常量 ──────────────────────────────────────────────────
BLOCK_DROP_Z = -0.01
MAX_EPISODE_STEPS = 10
ACTION_SCALE = 0.03

# 障碍物生成
HAND_SPAWN_DIST = (0.12, 0.25)       # 距 TCP 生成距离
HAND_SPEED_RANGE = (0.005, 0.015)     # 速度范围 m/step（降速给策略更多反应时间）
HIDDEN_POS = np.array([0.4, 0.5, 0.05])

HAND_BODY_NAMES = ["human_hand", "human_hand_2", "human_hand_3"]
HAND_GEOM_NAMES = ["human_hand", "human_hand_2", "human_hand_3"]

# 机械臂初始随机位置范围
TCP_INIT_X = (0.20, 0.55)
TCP_INIT_Y = (-0.30, 0.15)
TCP_INIT_Z = (0.05, 0.30)

# 奖励参数（参考 Kiemel et al. 2024: 每步奖励非负，碰撞大惩罚）
SURVIVAL_REWARD = 0.5            # 每步存活基础正奖励
DISPLACEMENT_COEFF = 0.5         # 位移软惩罚系数
ACTION_NORM_COEFF = 0.01         # 动作幅度惩罚
ACTION_SMOOTH_COEFF = 0.01       # 动作平滑惩罚
TERMINATION_PENALTY = -10.0      # 碰撞/掉块/越界/跑远 终止惩罚
SURVIVAL_BONUS = 5.0             # 存活完整 episode 的 bonus
MAX_DISPLACEMENT = 0.15          # 位移硬上限 15cm，超过视为任务失败

# 路过模式速度倍率（路过比朝TCP运动快，模拟人手快速经过）
PASSING_SPEED_MULT = 1.3

# DR 噪声参数
DR_OBS_POS_STD = 0.03       # 障碍物位置噪声 σ=3cm
DR_OBS_VEL_STD = 0.01       # 障碍物速度噪声 σ=1cm/s
DR_TCP_POS_STD = 0.005      # TCP 位置噪声 σ=5mm
DR_DELAY_MAX = 2            # 最大观测延迟步数
DR_POS_HISTORY_LEN = 5      # 位置历史缓冲区长度


class MotionMode(Enum):
    """障碍物运动模式。"""
    LINEAR = auto()     # 匀速直线朝 TCP
    ARC = auto()        # 带曲率弧线接近 TCP
    STOP_GO = auto()    # 朝 TCP 但随机停顿
    PASSING = auto()    # 横穿工作空间


class PandaBackupPolicyEnv(PandaPickPlaceSafeEnv):
    """Backup Policy 训练环境。

    Args:
        num_obstacles: 障碍物数量（1=S1, 2=S2: 1移动+1静止）
        enable_dr: 是否启用 Domain Randomization
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
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
    ):
        self._num_obstacles = min(num_obstacles, 2)
        self._enable_dr = enable_dr

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

        # MuJoCo body/geom ID
        self._obstacle_body_ids = [self._model.body(n).id for n in HAND_BODY_NAMES]
        self._obstacle_geom_ids = [self._model.geom(n).id for n in HAND_GEOM_NAMES]

        # 障碍物状态
        self._obstacle_pos = [HIDDEN_POS.copy() for _ in range(self._num_obstacles)]
        self._obstacle_prev_pos = [HIDDEN_POS.copy() for _ in range(self._num_obstacles)]
        self._obstacle_dir = [np.zeros(3) for _ in range(self._num_obstacles)]
        self._obstacle_speed = 0.015  # 当前 episode 的移动速度

        # 运动模式
        self._motion_mode = MotionMode.LINEAR
        self._stop_go_prob = 0.3  # 停走式每步停顿概率

        # DR: 位置历史缓冲区（用于模拟观测延迟）
        self._pos_histories: list[deque] = [
            deque(maxlen=DR_POS_HISTORY_LEN) for _ in range(self._num_obstacles)
        ]

        # 奖励辅助状态
        self._tcp_start = np.zeros(3)
        self._action_prev = np.zeros(6, dtype=np.float32)

        # 观测空间: robot(18) + num_obstacles × obstacle(10)
        agent_dim = 18 + self._num_obstacles * 10
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"pixels": spaces.Dict({
                "front": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                "wrist": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
            }), "agent_pos": agent_box} if self.image_obs
            else {"agent_pos": agent_box}
        )

        # 动作 6D
        self.action_space = spaces.Box(-np.ones(6, dtype=np.float32), np.ones(6, dtype=np.float32))

    # ================================================================
    # Gym API
    # ================================================================

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 先初始化障碍物状态（父类 reset 会调用 _compute_observation）
        self._motion_mode = MotionMode.LINEAR
        self._obstacle_speed = 0.015
        for i in range(self._num_obstacles):
            self._obstacle_pos[i] = HIDDEN_POS.copy()
            self._obstacle_prev_pos[i] = HIDDEN_POS.copy()
            self._obstacle_dir[i] = np.zeros(3)
            self._pos_histories[i].clear()
            self._pos_histories[i].append(HIDDEN_POS.copy())
        self._tcp_start = np.zeros(3)
        self._action_prev = np.zeros(6, dtype=np.float32)

        super().reset(seed=seed)

        # 机械臂随机位置 + 方块稳定夹持
        self._init_robot_with_block()

        tcp = self._data.site_xpos[self._pinch_site_id].copy()
        self._tcp_start = tcp.copy()

        # 随机运动模式和速度
        self._motion_mode = self._random.choice(list(MotionMode))
        self._obstacle_speed = self._random.uniform(*HAND_SPEED_RANGE)

        # 障碍物 0: 移动障碍物
        pos_0 = self._spawn_obstacle_pos(tcp)
        self._obstacle_pos[0] = pos_0
        self._obstacle_prev_pos[0] = pos_0.copy()
        self._obstacle_dir[0] = self._compute_moving_direction(pos_0, tcp)

        # 障碍物 1（S2）: 静止障碍物
        if self._num_obstacles >= 2:
            pos_1 = self._spawn_obstacle_pos(tcp)
            self._obstacle_pos[1] = pos_1
            self._obstacle_prev_pos[1] = pos_1.copy()
            self._obstacle_dir[1] = np.zeros(3)  # 静止

        # 重置 DR 位置历史（用真实初始位置）
        for i in range(self._num_obstacles):
            self._pos_histories[i].clear()
            self._pos_histories[i].append(self._obstacle_pos[i].copy())

        # 隐藏未使用的障碍物
        self._hand_active = True
        self._sync_all_visuals()

        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        action_6d = np.asarray(action, dtype=np.float32).flat[:6]

        # 移动障碍物 0（根据运动模式）
        self._obstacle_prev_pos[0] = self._obstacle_pos[0].copy()
        self._move_obstacle(0)

        # 静止障碍物 1 不动（仅更新 prev）
        if self._num_obstacles >= 2:
            self._obstacle_prev_pos[1] = self._obstacle_pos[1].copy()

        # 更新 DR 位置历史
        for i in range(self._num_obstacles):
            self._pos_histories[i].append(self._obstacle_pos[i].copy())

        self._sync_all_visuals()

        # 执行动作（6D 缩放 + 夹爪锁定）
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:6] = action_6d * ACTION_SCALE
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
        if not terminated and displacement > MAX_DISPLACEMENT:
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "excessive_displacement"}

        truncated = self._step_count >= MAX_EPISODE_STEPS

        # 奖励（参考 Kiemel et al. 2024: 每步奖励非负，终止大惩罚）
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
            if truncated:
                reward += SURVIVAL_BONUS

        self._action_prev = action_6d.copy()

        obs = self._compute_observation()
        info = {
            "succeed": False,
            "hand_collisions": self._hand_collisions,
            "min_hand_dist": self._min_hand_dist,
            "displacement": displacement,
            "num_obstacles": self._num_obstacles,
            "motion_mode": self._motion_mode.name,
            **safety_info,
        }
        return obs, reward, terminated, truncated, info

    # ================================================================
    # 障碍物运动
    # ================================================================

    def _move_obstacle(self, idx: int):
        """根据当前运动模式移动障碍物。

        运动模式区别：
          ARC     — 每步旋转方向向量（弧线轨迹）
          STOP_GO — 每步有概率跳过移动（停顿）
          PASSING — 速度 × PASSING_SPEED_MULT（横穿更快）
          LINEAR  — 默认直线匀速
        """
        # ARC: 旋转方向向量
        if self._motion_mode == MotionMode.ARC:
            omega = self._random.uniform(-0.1, 0.1)
            cos_w, sin_w = np.cos(omega), np.sin(omega)
            d = self._obstacle_dir[idx].copy()
            self._obstacle_dir[idx][0] = cos_w * d[0] - sin_w * d[1]
            self._obstacle_dir[idx][1] = sin_w * d[0] + cos_w * d[1]
            self._obstacle_dir[idx][2] += self._random.uniform(-0.02, 0.02)
            norm = np.linalg.norm(self._obstacle_dir[idx])
            if norm > 1e-8:
                self._obstacle_dir[idx] /= norm

        # STOP_GO: 概率跳过
        if self._motion_mode == MotionMode.STOP_GO and self._random.random() < self._stop_go_prob:
            return

        # 计算速度
        speed = self._obstacle_speed
        if self._motion_mode == MotionMode.PASSING:
            speed *= PASSING_SPEED_MULT

        self._obstacle_pos[idx] += self._obstacle_dir[idx] * speed
        self._obstacle_pos[idx] = np.clip(self._obstacle_pos[idx], *_CARTESIAN_BOUNDS)

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
        self._data.mocap_pos[0] = np.clip(random_tcp, *_CARTESIAN_BOUNDS)

        zero_action = np.zeros(7, dtype=np.float32)
        for _ in range(5):
            self.apply_action(zero_action)

        self._data.ctrl[self._gripper_ctrl_id] = 0  # Franka Hand: 0=关闭, MAX=全开
        for _ in range(5):
            self.apply_action(zero_action)

        tcp_pos = self._data.site_xpos[self._pinch_site_id].copy()
        self._data.jnt("block").qpos[:3] = tcp_pos
        self._data.jnt("block").qpos[3:] = [1, 0, 0, 0]
        self._data.jnt("block").qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        for _ in range(10):
            self.apply_action(zero_action)

    def _spawn_obstacle_pos(self, tcp: np.ndarray) -> np.ndarray:
        """在 TCP 附近生成障碍物位置（保证不在碰撞距离内）。"""
        for _ in range(20):
            raw_dir = self._random.normal(size=3)
            raw_dir /= np.linalg.norm(raw_dir) + 1e-8
            dist = self._random.uniform(*HAND_SPAWN_DIST)
            pos = tcp + raw_dir * dist
            pos = np.clip(pos, *_CARTESIAN_BOUNDS)
            if np.linalg.norm(pos - tcp) > HAND_COLLISION_DIST * 1.5:
                break
        return pos

    def _compute_moving_direction(self, pos: np.ndarray, tcp: np.ndarray) -> np.ndarray:
        """根据运动模式计算移动方向。"""
        toward_tcp = tcp - pos
        toward_tcp /= np.linalg.norm(toward_tcp) + 1e-8

        if self._motion_mode == MotionMode.PASSING:
            # 垂直于 toward_tcp 的方向（横穿）
            up = np.array([0.0, 0.0, 1.0])
            cross = np.cross(toward_tcp, up)
            cross_norm = np.linalg.norm(cross)
            if cross_norm > 1e-8:
                cross /= cross_norm
            else:
                cross = np.array([1.0, 0.0, 0.0])
            direction = cross + self._random.normal(0, 0.2, size=3)
            if self._random.random() < 0.5:
                direction = -direction
        else:
            direction = toward_tcp + self._random.normal(0, 0.2, size=3)

        direction /= np.linalg.norm(direction) + 1e-8
        return direction

    def _check_multi_safety(self) -> Tuple[bool, Dict[str, Any]]:
        """碰撞检测：所有障碍物 × 三检测点（TCP + 左右指尖）。"""
        tcp = self._data.site_xpos[self._pinch_site_id].copy()
        check_points = [tcp, self._data.xpos[self._left_pad_body_id], self._data.xpos[self._right_pad_body_id]]

        terminated = False
        safety_info = {"safety_violation": False, "violation_type": None}

        # C 区入侵
        if tcp[1] > ZONE_C_MIN_Y:
            self._zone_c_violations += 1
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "zone_c_intrusion"}
            return terminated, safety_info

        # 障碍物碰撞（用真实位置，不用 DR 噪声位置）
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
    # 观测
    # ================================================================

    def _compute_observation(self) -> Dict[str, Any]:
        # 使用基类方法获取 robot_state，确保编码器偏差正确注入
        robot_state = self.get_robot_state()                                     # shape: (18,)
        # TCP 位置取 robot_state 的最后 3 维（已含编码器偏差），加 DR 噪声模拟外部定位误差
        noisy_tcp = robot_state[15:18] + self._random.normal(0, DR_TCP_POS_STD, 3).astype(np.float32)
        robot_state[15:18] = noisy_tcp  # robot_state 中的 TCP 也使用带噪声版本

        obstacle_obs = []
        for i in range(self._num_obstacles):
            if self._enable_dr:
                # DR: 观测延迟 — 使用历史位置
                delay = self._random.randint(0, min(DR_DELAY_MAX, len(self._pos_histories[i]) - 1) + 1)
                delayed_pos = self._pos_histories[i][-(1 + delay)]

                # DR: 位置噪声
                obs_pos = (delayed_pos + self._random.normal(0, DR_OBS_POS_STD, 3)).astype(np.float32)

                # DR: 速度噪声（基于延迟位置计算速度）
                if len(self._pos_histories[i]) >= 2 + delay:
                    prev_delayed = self._pos_histories[i][-(2 + delay)]
                else:
                    prev_delayed = delayed_pos
                obs_vel = ((delayed_pos - prev_delayed) / self._control_dt).astype(np.float32)
                obs_vel += self._random.normal(0, DR_OBS_VEL_STD, 3).astype(np.float32)
            else:
                # 无 DR: 精确观测
                obs_pos = self._obstacle_pos[i].astype(np.float32)
                obs_vel = ((self._obstacle_pos[i] - self._obstacle_prev_pos[i]) / self._control_dt).astype(np.float32)

            obs_i = np.concatenate([
                [1.0],                                         # active
                obs_pos,                                       # pos (3,)
                obs_vel,                                       # vel (3,)
                (obs_pos - noisy_tcp).astype(np.float32),      # rel_pos (3,)
            ]).astype(np.float32)                               # shape: (10,)
            obstacle_obs.append(obs_i)

        agent_pos = np.concatenate([robot_state, *obstacle_obs])  # shape: (28,) or (38,)

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": agent_pos}
        return {"agent_pos": agent_pos}

    def _is_success(self) -> bool:
        return False
