"""
Safe Policy 训练环境

场景：人手进入共享工作空间 → Safe Policy 激活 → 小幅闪避 → 人手离开后交还 Task Policy。
每个 episode 随机 1~max_obstacles 个球体，其中 1 个移动威胁 + 其余静止障碍，策略需要躲开移动球且不碰静止球。

观测（48D）: 机器人状态(18) + 3 × 球体信息(10)
Reward: 碰撞任意球 -1，其余 -0.01 * ||action||
终止条件: 碰撞任意球 / 方块掉落 / 进入C区 / 10步到期

参考: Kiemel et al. "Safe RL of Robot Trajectories in the Presence of Moving Obstacles", IEEE RAL 2024
"""

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
HAND_SPEED = 0.015
HAND_SPAWN_DIST = (0.15, 0.25)
MAX_OBSTACLES = 3              # 观测槽位固定 3 个
DANGER_ZONE = 0.12             # proximity reward 生效范围
HIDDEN_POS = np.array([0.4, 0.5, 0.05])

HAND_BODY_NAMES = ["human_hand", "human_hand_2", "human_hand_3"]
HAND_GEOM_NAMES = ["human_hand", "human_hand_2", "human_hand_3"]

TCP_INIT_X = (0.20, 0.55)
TCP_INIT_Y = (-0.30, 0.15)
TCP_INIT_Z = (0.05, 0.30)


class PandaBackupPolicyEnv(PandaPickPlaceSafeEnv):
    """Safe Policy 训练环境：每 episode 随机 1~N 个球体，观测包含全部 3 个槽位（48D）。"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        max_obstacles: int = 1,
        fixed_obstacles: bool = False,
        use_proximity_reward: bool = False,
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
    ):
        self._max_obstacles = min(max_obstacles, MAX_OBSTACLES)
        self._fixed_obstacles = fixed_obstacles
        self._use_proximity_reward = use_proximity_reward

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

        # 球体状态（固定 MAX_OBSTACLES 个槽位）
        self._obstacle_active = [False] * MAX_OBSTACLES
        self._obstacle_pos = [HIDDEN_POS.copy() for _ in range(MAX_OBSTACLES)]
        self._obstacle_prev_pos = [HIDDEN_POS.copy() for _ in range(MAX_OBSTACLES)]
        self._obstacle_dir = [np.zeros(3) for _ in range(MAX_OBSTACLES)]
        self._num_active = 1

        # 观测 48D = 机器人(18) + 3 × 球体(10)
        agent_dim = 18 + MAX_OBSTACLES * 10
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
        super().reset(seed=seed)

        # 机械臂随机位置 + 方块稳定夹持
        self._init_robot_with_block()

        tcp = self._data.site_xpos[self._pinch_site_id].copy()
        if self._fixed_obstacles:
            self._num_active = self._max_obstacles
        elif self._max_obstacles == 1:
            self._num_active = 1
        else:
            weights = np.array([0.2, 0.3, 0.5][:self._max_obstacles])
            weights /= weights.sum()
            self._num_active = self._random.choice(range(1, self._max_obstacles + 1), p=weights)

        # 随机选 1 个球作为移动威胁，其余静止
        self._moving_idx = self._random.randint(0, self._num_active) if self._num_active > 0 else 0

        for i in range(MAX_OBSTACLES):
            if i < self._num_active:
                pos, direction = self._spawn_obstacle_near_tcp(tcp)
                self._obstacle_active[i] = True
                self._obstacle_pos[i] = pos
                self._obstacle_prev_pos[i] = pos.copy()
                self._obstacle_dir[i] = direction if i == self._moving_idx else np.zeros(3)
            else:
                self._obstacle_active[i] = False
                self._obstacle_pos[i] = HIDDEN_POS.copy()
                self._obstacle_prev_pos[i] = HIDDEN_POS.copy()
                self._obstacle_dir[i] = np.zeros(3)

        self._hand_active = True
        self._sync_all_visuals()
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # 移动所有活跃球体
        for i in range(self._num_active):
            self._obstacle_prev_pos[i] = self._obstacle_pos[i].copy()
            self._obstacle_pos[i] += self._obstacle_dir[i] * HAND_SPEED
            self._obstacle_pos[i] = np.clip(self._obstacle_pos[i], *_CARTESIAN_BOUNDS)
        self._sync_all_visuals()

        # 执行动作（缩放 + 夹爪锁定）
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:6] = np.asarray(action, dtype=np.float32).flat[:6] * ACTION_SCALE
        self.apply_action(full_action)

        # 安全检测
        terminated, safety_info = self._check_multi_safety()

        # 方块掉落
        if not terminated and self._data.sensor("block_pos").data[2] < BLOCK_DROP_Z:
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "block_dropped"}

        truncated = self._step_count >= MAX_EPISODE_STEPS

        # Reward（参考 Kiemel et al. 2024: r = α·R_d(移动) + β·R_d(静态) - action_penalty）
        if terminated:
            reward = -1.0
        elif self._use_proximity_reward:
            tcp = self._data.site_xpos[self._pinch_site_id]
            r_moving = 0.0
            r_static = 0.0
            n_static = 0
            for i in range(self._num_active):
                d = float(np.linalg.norm(tcp - self._obstacle_pos[i]))
                r_d = min(1.0, (d / DANGER_ZONE) ** 2)
                if i == self._moving_idx:
                    r_moving = r_d
                else:
                    r_static += r_d
                    n_static += 1
            if n_static > 0:
                r_static /= n_static  # 静止球平均
            else:
                r_static = 1.0  # 没有静止球，满分
            # α=0.6（移动球权重高），β=0.4（静止球权重低）
            reward = 0.6 * r_moving + 0.4 * r_static - 0.01 * float(np.linalg.norm(np.asarray(action).flat[:6]))
        else:
            reward = -0.01 * float(np.linalg.norm(np.asarray(action).flat[:6]))

        obs = self._compute_observation()
        info = {
            "succeed": False,
            "hand_collisions": self._hand_collisions,
            "min_hand_dist": self._min_hand_dist,
            "num_obstacles": self._num_active,
            **safety_info,
        }
        return obs, reward, terminated, truncated, info

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

        self._data.ctrl[self._gripper_ctrl_id] = 255.0
        for _ in range(5):
            self.apply_action(zero_action)

        tcp_pos = self._data.site_xpos[self._pinch_site_id].copy()
        self._data.jnt("block").qpos[:3] = tcp_pos
        self._data.jnt("block").qpos[3:] = [1, 0, 0, 0]
        self._data.jnt("block").qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        for _ in range(10):
            self.apply_action(zero_action)

    def _spawn_obstacle_near_tcp(self, tcp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成球体（保证不在碰撞距离内），方向大致朝 TCP。"""
        for _ in range(20):
            raw_dir = self._random.normal(size=3)
            raw_dir /= np.linalg.norm(raw_dir) + 1e-8
            dist = self._random.uniform(*HAND_SPAWN_DIST)
            pos = tcp + raw_dir * dist
            pos = np.clip(pos, *_CARTESIAN_BOUNDS)
            if np.linalg.norm(pos - tcp) > HAND_COLLISION_DIST * 1.5:
                break

        toward_tcp = tcp - pos
        toward_tcp /= np.linalg.norm(toward_tcp) + 1e-8
        noise = self._random.normal(0, 0.3, size=3)
        direction = toward_tcp + noise
        direction /= np.linalg.norm(direction) + 1e-8
        return pos, direction

    def _check_multi_safety(self) -> Tuple[bool, Dict[str, Any]]:
        """碰撞检测：任意活跃球 × 三检测点（TCP + 左右指尖）。"""
        tcp = self._data.site_xpos[self._pinch_site_id].copy()
        check_points = [tcp, self._data.xpos[self._left_pad_body_id], self._data.xpos[self._right_pad_body_id]]

        terminated = False
        safety_info = {"safety_violation": False, "violation_type": None}

        # C区入侵
        if tcp[1] > ZONE_C_MIN_Y:
            self._zone_c_violations += 1
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "zone_c_intrusion"}
            return terminated, safety_info

        # 球体碰撞
        for i in range(self._num_active):
            min_d = min(float(np.linalg.norm(pt - self._obstacle_pos[i])) for pt in check_points)
            self._min_hand_dist = min(self._min_hand_dist, min_d)
            if not terminated and min_d < HAND_COLLISION_DIST:
                self._hand_collisions += 1
                terminated = True
                safety_info = {"safety_violation": True, "violation_type": "hand_collision"}

        return terminated, safety_info

    def _sync_all_visuals(self):
        """同步球体位置到 MuJoCo 渲染。"""
        for i in range(MAX_OBSTACLES):
            active = i < self._num_active
            self._model.body_pos[self._obstacle_body_ids[i]] = self._obstacle_pos[i] if active else HIDDEN_POS
            self._model.geom_rgba[self._obstacle_geom_ids[i]][3] = 0.8 if active else 0.0

    def _update_human_hand(self):
        pass

    # ================================================================
    # 观测（48D）
    # ================================================================

    def _compute_observation(self) -> Dict[str, Any]:
        qpos = self._data.qpos[self._panda_dof_ids].astype(np.float32)
        qvel = self._data.qvel[self._panda_dof_ids].astype(np.float32)
        gripper = self.get_gripper_pose()  # [0, 1] 归一化
        real_tcp = self._data.site_xpos[self._pinch_site_id].copy().astype(np.float32)
        noisy_real_tcp = real_tcp + self._random.normal(0, 0.005, 3).astype(np.float32)

        robot_state = np.concatenate([qpos, qvel, gripper, noisy_real_tcp])  # 18D

        obstacle_obs = []
        for i in range(MAX_OBSTACLES):
            if self._obstacle_active[i]:
                pos = self._obstacle_pos[i].astype(np.float32)
                vel = ((self._obstacle_pos[i] - self._obstacle_prev_pos[i]) / self._control_dt).astype(np.float32)
                obs_i = np.concatenate([
                    [1.0], pos, vel, (pos - noisy_real_tcp).astype(np.float32),
                ])
            else:
                obs_i = np.concatenate([[0.0], np.full(3, -1.0), np.zeros(3), np.zeros(3)])
            obstacle_obs.append(obs_i.astype(np.float32))

        agent_pos = np.concatenate([robot_state, *obstacle_obs])  # 48D

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": agent_pos}
        return {"agent_pos": agent_pos}

    def _is_success(self) -> bool:
        return False
