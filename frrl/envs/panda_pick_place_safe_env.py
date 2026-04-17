"""
人机协作安全 Pick-and-Place 环境

场景布局（俯视图，+y向左）：
    区域A (y<-0.15)   区域B (-0.15<y<0.20)   区域C (y>0.20)
    机械臂抓取区        共享交接区               人员专属区（禁区）
       ■ 方块             ● 放置目标               👤 人员

任务：从区域A抓取方块 → 搬到区域B放置目标上
安全：
  - 硬约束：末端不能进入区域C（直接终止）
  - 硬约束：碰到人手（<8cm）则终止（TCP+左右指尖三点检测）
  - 安全层：人手在B区时机械臂完全停止，人手离开后全速（互斥模式，不可绕过）
人手行为：循环出现/消失，模拟真实人类反复操作
偏差：Joint 1（底座旋转）→ 横向偏移 → 可能偏向人员侧
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.base import FrankaGymEnv, GymRenderingSpec
from frrl.fault_injection import EncoderBiasConfig

# 机器人home位置
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))

# 笛卡尔工作空间（y_max限制在C区边界内，防止mocap目标进入禁区）
_CARTESIAN_BOUNDS = np.asarray([[0.15, -0.4, 0], [0.65, 0.17, 0.5]])

# 区域边界（y坐标）
ZONE_A_MAX_Y = -0.15   # 区域A上界
ZONE_C_MIN_Y = 0.20    # 区域C下界（禁区）— B区宽度0.35m

# 人手安全距离
HAND_COLLISION_DIST = 0.08   # 8cm，碰撞判定（覆盖夹爪展开尺寸）

# 方块和目标位置
BLOCK_INIT_POS = np.array([0.4, -0.2, 0.02])
PLATE_POS = np.array([0.4, 0.05, 0.0])

# 人手出现范围（区域B内，偏人员侧）
HAND_X_RANGE = (0.30, 0.50)
HAND_Y_RANGE = (0.0, 0.17)
HAND_Z_RANGE = (0.02, 0.08)

# 人手不可见时的存放位置（远离工作空间）
HAND_HIDDEN_POS = np.array([0.4, 0.5, 0.05])


class PandaPickPlaceSafeEnv(FrankaGymEnv):
    """人机协作安全 pick-and-place 环境。"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        hand_appear_prob: float = 0.9,
        hand_duration_range: Tuple[int, int] = (40, 120),
        hand_appear_step_range: Tuple[int, int] = (0, 3),
        hand_reappear_interval: Tuple[int, int] = (20, 50),
        safety_layer_enabled: bool = True,
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
        obs_mode: Literal["obs31", "obs28", "obs25", "obs27", "obs24", "obs21"] = "obs31",
    ):
        self._hand_appear_prob = hand_appear_prob
        self._hand_duration_range = hand_duration_range
        self._hand_appear_step_range = hand_appear_step_range
        self._hand_reappear_interval = hand_reappear_interval
        self._safety_layer_enabled = safety_layer_enabled
        # 观测消融模式
        #   带 hand（Safe 策略）:
        #     obs31: 完整 = robot(18) + block(3) + plate(3) + real_tcp(3) + hand_active(1) + hand_pos(3)
        #     obs28: 去 real_tcp = robot(18) + block(3) + plate(3) + hand_active(1) + hand_pos(3)
        #     obs25: 去 block/plate = robot(18) + real_tcp(3) + hand_active(1) + hand_pos(3)
        #   不带 hand（Task 策略，与带 hand 的逐一对应，去掉 hand_active+hand_pos=4 维）:
        #     obs27 = robot(18) + block(3) + plate(3) + real_tcp(3)
        #     obs24 = robot(18) + block(3) + plate(3)
        #     obs21 = robot(18) + real_tcp(3)
        if obs_mode not in ("obs31", "obs28", "obs25", "obs27", "obs24", "obs21"):
            raise ValueError(f"Unknown obs_mode: {obs_mode}")
        self._obs_mode = obs_mode

        project_root = Path(__file__).parent.parent.parent
        super().__init__(
            xml_path=project_root / "assets" / "pick_place_safe_scene.xml",
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            encoder_bias_config=encoder_bias_config,
        )

        # 物体ID（验证场景XML中所有必需的body/geom存在）
        self._block_body_id = self._model.body("block").id
        self._plate_body_id = self._model.body("plate").id
        self._hand_body_id = self._model.body("human_hand").id
        self._hand_geom_id = self._model.geom("human_hand").id
        self._block_z = self._model.geom("block").size[2]

        # 夹爪指尖body ID，用于多点碰撞检测
        self._left_pad_body_id = self._model.body("left_pad").id
        self._right_pad_body_id = self._model.body("right_pad").id

        # 人手状态机: HIDDEN → ENTERING → STAYING → LEAVING → HIDDEN
        self._hand_state = "HIDDEN"
        self._hand_pos = HAND_HIDDEN_POS.copy()
        self._hand_start_pos = HAND_HIDDEN_POS.copy()  # C区起点
        self._hand_target_pos = HAND_HIDDEN_POS.copy()  # B区目标
        self._hand_active = False  # 是否在B区附近（需要避让）
        self._hand_move_speed = 0.008  # 每步移动距离(m)
        self._hand_appear_step = 0
        self._hand_stay_duration = 0
        self._hand_stay_counter = 0
        self._step_count = 0

        # 安全统计（每episode）
        self._zone_c_violations = 0
        self._hand_collisions = 0
        self._min_hand_dist = float('inf')

        # 观测空间维度由 obs_mode 决定
        _obs_dim_map = {"obs31": 31, "obs28": 28, "obs25": 25, "obs27": 27, "obs24": 24, "obs21": 21}
        agent_dim = _obs_dim_map[self._obs_mode]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)

        if self.image_obs:
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({
                    "front": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                    "wrist": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                }),
                "agent_pos": agent_box,
            })
        else:
            self.observation_space = spaces.Dict({
                "agent_pos": agent_box,
                "environment_state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
            })

    # ================================================================
    # Gym API
    # ================================================================

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        # 方块放到区域A
        self._data.jnt("block").qpos[:3] = BLOCK_INIT_POS.copy()
        mujoco.mj_forward(self._model, self._data)

        # 重置步数和安全统计
        self._step_count = 0
        self._zone_c_violations = 0
        self._hand_collisions = 0
        self._min_hand_dist = float('inf')

        # 人手状态机重置
        self._hand_state = "HIDDEN"
        self._hand_active = False
        self._hand_stay_counter = 0

        if self._random.random() < self._hand_appear_prob:
            self._randomize_hand_params()

            # 50%概率人手一开始就在B区（跳过ENTERING），50%正常从C区进入
            if self._random.random() < 0.5:
                # 直接在B区，跳过ENTERING
                self._hand_pos = self._hand_target_pos.copy()
                self._hand_state = "STAYING"
                self._hand_active = True
                self._sync_hand_visual()
            else:
                # 正常从C区进入
                self._hand_appear_step = self._random.randint(*self._hand_appear_step_range)
                self._hand_pos = HAND_HIDDEN_POS.copy()
                self._sync_hand_visual()
        else:
            self._hand_appear_step = 9999
            self._hand_pos = HAND_HIDDEN_POS.copy()
            self._sync_hand_visual()

        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # 先更新人手位置，再算安全层（确保用最新的人手位置）
        self._update_human_hand()

        # 安全层：根据TCP与人手距离缩放动作速度（不可绕过）
        safe_action, speed_scale = self._apply_safety_layer(action)

        self.apply_action(safe_action)

        # 安全检测（仅硬约束：碰撞终止）
        terminated, safety_info = self._check_safety()

        # 任务判定（纯任务奖励，安全由安全层保证）
        obs = self._compute_observation()
        success = self._is_success()
        reward = 1.0 if success else 0.0

        if success:
            terminated = True

        info = {
            "succeed": success,
            "hand_active": self._hand_active,
            "zone_c_violations": self._zone_c_violations,
            "hand_collisions": self._hand_collisions,
            "min_hand_dist": self._min_hand_dist,
            "speed_scale": speed_scale,
            **safety_info,
        }

        return obs, reward, terminated, False, info

    # ================================================================
    # 人手逻辑
    # ================================================================

    def _randomize_hand_params(self):
        """随机生成人手的B区目标位置、C区起始位置和停留时长。"""
        self._hand_stay_duration = self._random.randint(*self._hand_duration_range)
        self._hand_target_pos = np.array([
            self._random.uniform(*HAND_X_RANGE),
            self._random.uniform(*HAND_Y_RANGE),
            self._random.uniform(*HAND_Z_RANGE),
        ])
        self._hand_start_pos = self._hand_target_pos.copy()
        self._hand_start_pos[1] = ZONE_C_MIN_Y + 0.1

    def _move_toward(self, target: np.ndarray) -> bool:
        """将人手向目标移动一步，返回是否已到达。"""
        direction = target - self._hand_pos
        dist = np.linalg.norm(direction)
        if dist < self._hand_move_speed:
            self._hand_pos = target.copy()
            return True
        self._hand_pos += direction / dist * self._hand_move_speed
        return False

    def _update_human_hand(self):
        """
        人手状态机: HIDDEN → ENTERING → STAYING → LEAVING → HIDDEN（循环）

        HIDDEN:   人手不可见，等待出现时机
        ENTERING: 人手从C区向B区移动
        STAYING:  人手在B区停留
        LEAVING:  人手从B区缩回C区
        """
        if self._hand_state == "HIDDEN":
            if self._step_count >= self._hand_appear_step:
                self._hand_state = "ENTERING"
                self._hand_pos = self._hand_start_pos.copy()
                self._hand_active = True

        elif self._hand_state == "ENTERING":
            if self._move_toward(self._hand_target_pos):
                self._hand_state = "STAYING"
                self._hand_stay_counter = 0

        elif self._hand_state == "STAYING":
            self._hand_stay_counter += 1
            if self._hand_stay_counter >= self._hand_stay_duration:
                self._hand_state = "LEAVING"

        elif self._hand_state == "LEAVING":
            if self._move_toward(self._hand_start_pos):
                self._hand_pos = HAND_HIDDEN_POS.copy()
                self._hand_state = "HIDDEN"
                self._hand_active = False
                # 循环出现：间隔随机步数后再次进入
                self._hand_appear_step = self._step_count + self._random.randint(*self._hand_reappear_interval)
                self._randomize_hand_params()

        self._sync_hand_visual()

    def _sync_hand_visual(self):
        """同步人手模型位置和可见性到MuJoCo渲染。"""
        if self._hand_state == "HIDDEN":
            self._model.body_pos[self._hand_body_id] = HAND_HIDDEN_POS
            self._model.geom_rgba[self._hand_geom_id][3] = 0.0
        else:
            self._model.body_pos[self._hand_body_id] = self._hand_pos
            self._model.geom_rgba[self._hand_geom_id][3] = 0.8

    # ================================================================
    # 安全检测
    # ================================================================

    def _min_dist_to_hand(self) -> float:
        """计算机械臂关键点（TCP + 左右指尖）到人手的最小距离。"""
        tcp = self._data.site_xpos[self._pinch_site_id]
        left_pad = self._data.xpos[self._left_pad_body_id]
        right_pad = self._data.xpos[self._right_pad_body_id]

        return min(
            np.linalg.norm(tcp - self._hand_pos),
            np.linalg.norm(left_pad - self._hand_pos),
            np.linalg.norm(right_pad - self._hand_pos),
        )

    def _check_safety(self) -> Tuple[bool, Dict[str, Any]]:
        """检查硬安全约束，返回(是否终止, 安全信息)。"""
        real_tcp = self._data.site_xpos[self._pinch_site_id].copy()
        terminated = False
        safety_info = {"safety_violation": False, "violation_type": None}

        # 硬约束：末端进入区域C → 直接终止
        if real_tcp[1] > ZONE_C_MIN_Y:
            self._zone_c_violations += 1
            terminated = True
            safety_info = {"safety_violation": True, "violation_type": "zone_c_intrusion"}

        # 硬约束：机械臂任意部分碰到人手 → 终止（TCP + 左右指尖三点检测）
        if self._hand_active:
            dist_to_hand = self._min_dist_to_hand()
            self._min_hand_dist = min(self._min_hand_dist, dist_to_hand)

            if dist_to_hand < HAND_COLLISION_DIST:
                self._hand_collisions += 1
                terminated = True
                safety_info = {"safety_violation": True, "violation_type": "hand_collision"}

        return terminated, safety_info

    # ================================================================
    # 安全层
    # ================================================================

    def _hand_in_b_zone(self) -> bool:
        """判断人手是否在B区（共享交接区）内。"""
        return self._hand_active and ZONE_A_MAX_Y < self._hand_pos[1] < ZONE_C_MIN_Y

    def _apply_safety_layer(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        硬编码安全层：人手在B区时机械臂完全停止，人手不在B区时全速。

        互斥逻辑（ISO/TS 15066 安全停止监控模式）：
          人手在B区   → scale = 0.0（停止，等人手离开）
          人手不在B区 → scale = 1.0（全速，抓住窗口完成任务）

        全部动作归零（包括夹爪），机械臂完全冻结。

        返回：(缩放后的动作, 速度缩放系数)
        """
        if not self._safety_layer_enabled or not self._hand_in_b_zone():
            return action, 1.0

        return np.zeros_like(action), 0.0

    # ================================================================
    # 观测
    # ================================================================

    def _compute_observation(self) -> Dict[str, Any]:
        robot_state = self.get_robot_state().astype(np.float32)
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        plate_pos = self._data.sensor("plate_pos").data.astype(np.float32)

        # 真实末端位置 + 5mm噪声（外部定位系统）
        real_tcp = self._data.site_xpos[self._pinch_site_id].copy().astype(np.float32)
        noisy_real_tcp = real_tcp + self._random.normal(0, 0.005, 3).astype(np.float32)

        # 人手信息
        hand_active = np.array([1.0 if self._hand_active else 0.0], dtype=np.float32)
        hand_pos = self._hand_pos.astype(np.float32) if self._hand_active else np.full(3, -1.0, dtype=np.float32)

        # 按 obs_mode 拼接观测
        if self._obs_mode == "obs31":
            # 完整 Safe
            agent_pos = np.concatenate([robot_state, block_pos, plate_pos, noisy_real_tcp, hand_active, hand_pos])
        elif self._obs_mode == "obs28":
            # Safe 去外部 TCP oracle
            agent_pos = np.concatenate([robot_state, block_pos, plate_pos, hand_active, hand_pos])
        elif self._obs_mode == "obs25":
            # Safe 去 block/plate，保留外部 TCP 和人手
            agent_pos = np.concatenate([robot_state, noisy_real_tcp, hand_active, hand_pos])
        elif self._obs_mode == "obs27":
            # Task 完整（去 hand 4 维）
            agent_pos = np.concatenate([robot_state, block_pos, plate_pos, noisy_real_tcp])
        elif self._obs_mode == "obs24":
            # Task 去外部 TCP oracle
            agent_pos = np.concatenate([robot_state, block_pos, plate_pos])
        else:  # obs21
            # Task 去 block/plate
            agent_pos = np.concatenate([robot_state, noisy_real_tcp])

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": agent_pos}

        return {
            "agent_pos": agent_pos,
            "environment_state": np.concatenate([block_pos, plate_pos]),
        }

    # ================================================================
    # 任务判定
    # ================================================================

    def _is_success(self) -> bool:
        """方块在放置目标附近（距离<5cm）且高度低（已放下）。"""
        block_pos = self._data.sensor("block_pos").data
        plate_pos = self._data.sensor("plate_pos").data
        dist = np.linalg.norm(block_pos[:2] - plate_pos[:2])
        return dist < 0.05 and block_pos[2] < 0.05
