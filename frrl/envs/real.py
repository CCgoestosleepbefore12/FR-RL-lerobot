"""
Franka Panda 真机 Gym 环境

通过 HTTP 与 RT PC 上的 Flask Server (franka_server.py) 通信控制真实 Franka Panda。
观测格式与仿真 FrankaGymEnv 对齐：18D agent_pos + 6D environment_state + pixels。

移植自 hil-serl/serl_robot_infra/franka_env/envs/franka_env.py，
适配 FR-RL 的观测/动作接口。
"""

import copy
import logging
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from frrl.envs.real_config import FrankaRealConfig
from frrl.fault_injection import EncoderBiasConfig, EncoderBiasInjector

logging.basicConfig(level=logging.INFO)

# HTTP request default timeout (seconds). Flask server on RT PC responds in
# milliseconds; anything >100ms indicates a stuck C++ controller or USB issue.
# 2 second ceiling prevents the training loop from hanging forever.
_HTTP_TIMEOUT_S: float = 2.0


def euler_2_quat(euler: np.ndarray) -> np.ndarray:
    """欧拉角 (xyz) → 四元数 (xyzw)。"""
    return Rotation.from_euler("xyz", euler).as_quat()


def quat_2_euler(quat: np.ndarray) -> np.ndarray:
    """四元数 (xyzw) → 欧拉角 (xyz)。"""
    return Rotation.from_quat(quat).as_euler("xyz")


class FrankaRealEnv(gym.Env):
    """Franka Panda 真机 Gym 环境。

    观测格式（与仿真 FrankaGymEnv 对齐）：
    - agent_pos: 18D = q(7) + dq(7) + gripper(1, [0,1]) + tcp_pos(3)
    - environment_state: 6D = block_pos(3) + plate_pos(3)
    - pixels: {"front": (128,128,3), "wrist": (128,128,3)}

    动作: 7D = [dx, dy, dz, rx, ry, rz, gripper]
    """

    @staticmethod
    def _image_size_for(config: FrankaRealConfig, cam_name: str) -> Tuple[int, int]:
        """读取 per-camera image_size；支持旧版 Tuple 全局值。"""
        s = config.image_size
        if isinstance(s, dict):
            if cam_name not in s:
                raise KeyError(f"image_size dict 缺少 '{cam_name}'")
            return s[cam_name]
        return s

    def __init__(self, config: FrankaRealConfig):
        super().__init__()
        self.config = config
        self.url = config.server_url

        # 动作/观测空间
        self.action_space = spaces.Box(
            low=np.full(7, -1.0, dtype=np.float32),
            high=np.full(7, 1.0, dtype=np.float32),
        )

        # 29D agent_pos:
        #   [0:7]   q_biased           从 /getstate（含 J1 bias）
        #   [7:14]  dq                 (bias 固定偏移不影响 Δq/dt)
        #   [14]    gripper            [0, 1]
        #   [15:18] tcp_pos_biased     从 /getstate（biased FK）
        #   [18:22] tcp_quat_biased    scipy xyzw
        #   [22:25] tcp_pos_true       从 /getstate_true （privileged 作弊通道）
        #   [25:29] tcp_quat_true      scipy xyzw        （privileged 作弊通道）
        agent_dim = 29
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32),
            "environment_state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
            "pixels": spaces.Dict({
                name: spaces.Box(
                    0, 255,
                    (*self._image_size_for(config, name), 3),
                    dtype=np.uint8,
                )
                for name in config.cameras
            }),
        })

        # 重置位姿（euler → quat）
        self.resetpos = np.concatenate([
            config.reset_pose[:3],
            euler_2_quat(config.reset_pose[3:]),
        ])

        # 安全边界
        self.xyz_bounding_box = spaces.Box(
            config.abs_pose_limit_low[:3],
            config.abs_pose_limit_high[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = spaces.Box(
            config.abs_pose_limit_low[3:],
            config.abs_pose_limit_high[3:],
            dtype=np.float64,
        )

        # 编码器偏差注入
        if config.encoder_bias_config is not None:
            self.bias_injector = EncoderBiasInjector(config.encoder_bias_config)
        else:
            self.bias_injector = None

        # BiasMonitor（可选）：matplotlib 波形图，q_true vs q_biased + ep 边界竖线。
        # 仅在有 bias 注入且显式启用时创建，避免 headless 训练/离线场景弹窗。
        self._bias_monitor = None
        if self.bias_injector is not None and config.enable_bias_monitor:
            from frrl.fault_injection import BiasMonitor
            self._bias_monitor = BiasMonitor(update_hz=2.0)

        # 运行时状态 —— biased（来自 /getstate）
        self.currpos = np.zeros(7)     # xyz + quat  (7D) biased
        self.currvel = np.zeros(6)     # 末端速度    (6D)
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        self.q = np.zeros(7)           # 关节角      (7D) biased
        self.dq = np.zeros(7)          # 关节角速度  (7D) bias 不影响
        self.curr_gripper_pos = 0.0
        # 运行时状态 —— true（来自 /getstate_true，privileged 作弊通道）
        self.currpos_true = np.zeros(7)  # xyz + quat (7D) true
        self.q_true = np.zeros(7)        # 关节角     (7D) true
        self.curr_path_length = 0
        self.cycle_count = 0
        self.last_gripper_act = 0.0

        # 相机（延迟初始化，允许无相机的 mock 测试）
        self._camera_manager = None
        self._camera_init_failed = False  # 一次失败就标记，避免每步重试刷屏
        # Live view（config.display_image=True 时两路相机原始 BGR imshow）：
        # headless 或 cv2 GUI 不可用时自动降级为 False，避免每步刷警告。
        self._live_view_failed = False

        # 键盘 reward listener（仅当 reward_backend == "keyboard" 时启动）
        self._keyboard_reward = None
        if config.reward_backend == "keyboard":
            from frrl.rewards.keyboard_reward import KeyboardRewardListener
            # required=True: pynput ImportError 直接抛，不静默降级为"永不 reward"
            self._keyboard_reward = KeyboardRewardListener(required=True)

        # ESC 紧急停止
        self.terminate = False

        # Intervention 标记：下一次 step() 写入的 info["is_intervention"]。
        # Demo collection / InterventionWrapper 在 step 前调 set_intervention(True)；
        # 默认 False 表示 action 来自 policy。
        self._is_intervention_pending: bool = False

        logging.info(
            f"FrankaRealEnv 初始化: server={self.url}, "
            f"reward_backend={config.reward_backend}"
        )

    # ================================================================
    # Gym API
    # ================================================================

    def reset(self, *, seed=None, options=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.last_gripper_act = time.time()

        # 周期性关节重置
        self.cycle_count += 1
        joint_reset = (
            self.config.joint_reset_period != 0
            and self.cycle_count % self.config.joint_reset_period == 0
        )

        self._recover()
        self._go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        # 编码器偏差注入
        if self.bias_injector is not None:
            self.bias_injector.on_episode_start(num_joints=7)
            bias = self.bias_injector.current_bias
            if bias is not None:
                self._set_encoder_bias(bias.tolist())
                logging.info(f"真机 Episode: bias = {np.round(bias, 4)}")
                if self._bias_monitor is not None:
                    self._bias_monitor.mark_episode_boundary(
                        ep_num=self.cycle_count, bias=bias,
                    )
            else:
                self._set_encoder_bias([0.0] * 7)

        self._update_currpos()
        self._update_currpos_true()
        self.terminate = False

        # Keyboard backend: clear any stale outcome and block until human presses S.
        # This is after all robot reset so the operator has time to arrange the scene.
        if self._keyboard_reward is not None:
            self._keyboard_reward.mark_episode_ended()
            logging.info("等待操作者按 S 开始 episode（Enter=成功 / Space=失败 / Backspace=作废）")
            self._keyboard_reward.wait_for_start()

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 解析 7D 动作: [dx, dy, dz, rx, ry, rz, gripper]
        xyz_delta = action[:3]
        rot_delta = action[3:6]
        gripper_action = action[6] * self.config.action_scale[2]

        # 计算目标位姿
        nextpos = self.currpos.copy()
        raw_xyz_delta = xyz_delta * self.config.action_scale[0]
        # Safety net: cap xyz step magnitude so末端 max speed ≤ config.max_cart_speed.
        # At 10Hz the per-step ceiling is max_cart_speed / hz (meters).
        max_step_xyz = self.config.max_cart_speed / max(self.config.hz, 1)
        step_norm = float(np.linalg.norm(raw_xyz_delta))
        if step_norm > max_step_xyz and step_norm > 0.0:
            raw_xyz_delta = raw_xyz_delta * (max_step_xyz / step_norm)
        nextpos[:3] += raw_xyz_delta
        nextpos[3:] = (
            Rotation.from_rotvec(rot_delta * self.config.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        # 夹爪
        self._send_gripper_command(gripper_action)
        # 位姿命令
        self._send_pos_command(self.clip_safety_box(nextpos))

        self.curr_path_length += 1

        # 控制频率
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.hz) - dt))

        self._update_currpos()
        self._update_currpos_true()
        obs = self._compute_observation()
        self._render_live_view()
        # BiasMonitor：推 (q_true, q_biased, bias) 一个样本；绘制被 update_hz 限流
        # （默认 2Hz）。bias_injector 在无故障 episode 下 current_bias 为 None，
        # 此时用零向量占位以保持时间轴连续（plot init 仍需首个非零 bias）。
        if self._bias_monitor is not None:
            bias = self.bias_injector.current_bias
            if bias is None:
                bias = np.zeros(7, dtype=np.float32)
            self._bias_monitor.update(self.q_true, self.q, bias)
        rinfo = self.compute_reward(obs)
        # Separate terminated (task success / user stop) from truncated (time limit):
        # gymnasium's bootstrap logic relies on this split (terminated disables bootstrap,
        # truncated keeps it). Conflating both into `done` silently biases Q estimates.
        terminated = bool(rinfo["terminal"] or self.terminate)
        truncated = self.curr_path_length >= self.config.max_episode_length
        # When truncation happens before the human presses a key, clear the
        # listener so the next reset() blocks on S as expected.
        if truncated and not terminated and self._keyboard_reward is not None:
            self._keyboard_reward.mark_episode_ended()
        # Consume intervention flag set by upstream before step().
        is_intervention = self._is_intervention_pending
        self._is_intervention_pending = False
        return obs, float(rinfo["reward"]), terminated, truncated, {
            "succeed": rinfo["reward"] == 1,
            "discard": rinfo["discard"],
            # HIL-SERL schema for RLPD: downstream actor / pickle loader uses
            # these to route transitions between online and offline/prior buffers.
            "teleop_action": np.asarray(action, dtype=np.float32),
            "is_intervention": is_intervention,
            "intervene_action": np.asarray(action, dtype=np.float32) if is_intervention else None,
        }

    def go_home(self) -> None:
        """Drive the robot back to reset pose without the keyboard wait.

        Useful for collection / training scripts that finish early and want the
        arm in a safe position before shutting down. Unlike `reset()`, this
        does NOT resample bias, touch episode counters, or block on the S key.
        Errors are logged and swallowed so close-out always reaches `env.close()`.
        """
        try:
            self._recover()
            self._go_to_reset(joint_reset=False)
            self._recover()
            logging.info("FrankaRealEnv: robot returned to home pose")
        except Exception as e:
            logging.warning(f"go_home() failed (robot may be left in place): {e}")

    def set_intervention(self, is_intervention: bool) -> None:
        """Mark the next step()'s action as human-intervened.

        Upstream callers (demo collection scripts, online intervention wrappers)
        call this **before** env.step(action) so the written
        ``info["is_intervention"]`` tag correctly routes the transition
        into the prior/offline replay buffer (RLPD convention). Default is
        False (action is from the policy).
        """
        self._is_intervention_pending = bool(is_intervention)

    def _render_live_view(self) -> None:
        """本地 imshow 两路相机的原始 BGR 帧（640×480，crop 前）。

        仅当 `config.display_image=True` 且相机已初始化且没有历史失败时才调用
        `cv2.imshow`。headless（无 X11）首次调用会抛 cv2.error → 吞掉 + 打一次
        warning + 置 `_live_view_failed=True`，后续 step 不再尝试。
        """
        if (
            not self.config.display_image
            or self._live_view_failed
            or self._camera_manager is None
        ):
            return
        try:
            for name, bgr in self._camera_manager.get_images_raw().items():
                cv2.imshow(f"cam/{name}", bgr)
            cv2.waitKey(1)
        except cv2.error as e:
            logging.warning(
                f"live view 失败（无 GUI？）: {e} — 后续步骤跳过 imshow"
            )
            self._live_view_failed = True

    def close(self):
        if self._camera_manager is not None:
            try:
                self._camera_manager.close()
            except Exception as e:
                logging.warning(f"camera close failed: {e}")
            self._camera_manager = None
        if self._keyboard_reward is not None:
            try:
                self._keyboard_reward.stop()
            except Exception as e:
                logging.warning(f"keyboard listener stop failed: {e}")
            self._keyboard_reward = None
        if self._bias_monitor is not None:
            try:
                self._bias_monitor.close()
            except Exception as e:
                logging.warning(f"bias monitor close failed: {e}")
            self._bias_monitor = None
        # 关闭所有 live view 窗口（若曾打开过）；headless 下无窗口也安全。
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # ================================================================
    # 观测
    # ================================================================

    def get_robot_state(self) -> np.ndarray:
        """构建 29D 机器人状态。

        布局（索引 → 含义 → 来源）：
          [0:7]   q_biased         /getstate（可能含 J1 bias）
          [7:14]  dq               /getstate（bias 固定偏移不影响 Δq/dt）
          [14]    gripper          [0, 1]
          [15:18] tcp_pos_biased   /getstate（biased FK 的 xyz）
          [18:22] tcp_quat_biased  /getstate（biased FK 的 quat，scipy xyzw）
          [22:25] tcp_pos_true     /getstate_true（privileged 作弊通道）
          [25:29] tcp_quat_true    /getstate_true（privileged 作弊通道，scipy xyzw）
        """
        return np.concatenate([
            self.q.astype(np.float32),                            # [0:7]
            self.dq.astype(np.float32),                           # [7:14]
            np.array([self.curr_gripper_pos], dtype=np.float32),  # [14]
            self.currpos[:3].astype(np.float32),                  # [15:18]
            self.currpos[3:].astype(np.float32),                  # [18:22]
            self.currpos_true[:3].astype(np.float32),             # [22:25]
            self.currpos_true[3:].astype(np.float32),             # [25:29]
        ])

    def get_environment_state(self) -> np.ndarray:
        """6D 环境状态: block_pos(3) + plate_pos(3)。

        block_pos 需要通过相机+CV检测获取（暂时用固定值占位）。
        plate_pos 固定写死。
        """
        # TODO: 接入相机物块检测（AprilTag/YOLO + D455 深度 + T_cam_to_robot）
        block_pos = self.config.plate_position.copy()  # 占位：与 plate 相同
        plate_pos = self.config.plate_position.copy()
        return np.concatenate([block_pos, plate_pos]).astype(np.float32)

    def _compute_observation(self) -> Dict[str, Any]:
        """构建完整观测（与仿真 FrankaGymEnv 对齐）。"""
        obs = {
            "agent_pos": self.get_robot_state(),
            "environment_state": self.get_environment_state(),
        }

        # 图像
        images = self._get_images()
        if images is not None:
            obs["pixels"] = images

        return obs

    def _get_images(self) -> Optional[Dict[str, np.ndarray]]:
        """从 RealSense 获取图像。无相机时返回 None。

        初始化失败会打印一次 warning 并置位 _camera_init_failed，之后每步直接
        返回 None —— 否则每个 step 都重试 init 会刷屏。
        """
        if self._camera_init_failed:
            return None

        if self._camera_manager is None:
            try:
                from frrl.robots.franka_real.cameras.realsense import RealSenseCameraManager
                self._camera_manager = RealSenseCameraManager(
                    camera_configs=self._resolve_camera_serials(self.config.cameras),
                    image_crop=self.config.image_crop,
                    image_size=self.config.image_size,
                )
            except (ImportError, RuntimeError) as e:
                logging.warning(f"相机初始化失败（后续步骤跳过图像观测）: {e}")
                self._camera_init_failed = True
                return None

        try:
            return self._camera_manager.get_images()
        except Exception as e:
            logging.warning(f"相机读取失败: {e}")
            return None

    @staticmethod
    def _resolve_camera_serials(cameras_cfg: Dict[str, dict]) -> Dict[str, dict]:
        """把 `"000000000000"` 占位符替换成真实检测到的 RealSense serial。

        映射规则：按 cameras_cfg 的 key 顺序，绑到已检测设备的相应顺序上。例如
        cameras = {"front": ..., "wrist": ...} + 检测到 [A, B]
           → front=A, wrist=B
        如果某个 serial 已经是真实 serial（非占位符），原样保留。
        """
        from frrl.robots.franka_real.cameras.realsense import RSCapture

        PLACEHOLDER = "000000000000"
        has_placeholder = any(
            v.get("serial_number") == PLACEHOLDER for v in cameras_cfg.values()
        )
        if not has_placeholder:
            return cameras_cfg

        available = RSCapture.get_device_serial_numbers()
        resolved: Dict[str, dict] = {}
        used = set()
        # 先处理已显式配置的 serial
        for name, cfg in cameras_cfg.items():
            serial = cfg.get("serial_number", "")
            if serial and serial != PLACEHOLDER:
                resolved[name] = dict(cfg)
                used.add(serial)
        # 剩余的占位符按 available 顺序分配
        remaining = [s for s in available if s not in used]
        for name, cfg in cameras_cfg.items():
            if name in resolved:
                continue
            if not remaining:
                raise RuntimeError(
                    f"相机 '{name}' 的 serial 是占位符，但可用设备已分配完。"
                    f"available={available}, already_used={used}"
                )
            picked = remaining.pop(0)
            new_cfg = dict(cfg)
            new_cfg["serial_number"] = picked
            resolved[name] = new_cfg
            logging.info(f"相机 auto-resolve: {name} -> {picked}")
        return resolved

    # ================================================================
    # 奖励
    # ================================================================

    def compute_reward(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to the configured reward backend.

        Returns a dict {reward:int, terminal:bool, discard:bool}. ``discard``
        signals the actor/learner to drop this episode from the replay buffer
        (e.g. operator messed up the scene setup). Only meaningful for the
        keyboard backend; pose backend always returns discard=False.
        """
        if self.config.reward_backend == "keyboard":
            return self._compute_reward_keyboard()
        return self._compute_reward_pose(obs)

    def _compute_reward_pose(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.target_pose is None:
            return {"reward": 0, "terminal": False, "discard": False}
        current_pose = np.concatenate([
            self.currpos[:3],
            quat_2_euler(self.currpos[3:]),
        ])
        delta = np.abs(current_pose - self.config.target_pose)
        success = bool(np.all(delta < self.config.reward_threshold))
        return {"reward": int(success), "terminal": success, "discard": False}

    def _compute_reward_keyboard(self) -> Dict[str, Any]:
        if self._keyboard_reward is None:
            return {"reward": 0, "terminal": False, "discard": False}
        outcome = self._keyboard_reward.poll()
        if outcome is None:
            return {"reward": 0, "terminal": False, "discard": False}
        return outcome

    # ================================================================
    # 机器人控制
    # ================================================================

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """裁剪位姿到安全边界（xyz + rpy）。"""
        pose = pose.copy()
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        # 第一个欧拉角处理 ±π 不连续
        sign = np.sign(euler[0])
        euler[0] = sign * np.clip(
            np.abs(euler[0]),
            self.rpy_bounding_box.low[0],
            self.rpy_bounding_box.high[0],
        )
        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
        return pose

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """线性插值平滑移动到目标位姿。"""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.config.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1.0 / self.config.hz)
        self._update_currpos()

    def _go_to_reset(self, joint_reset: bool = False):
        """执行环境重置：precision 模式 → 移动到 reset 位姿 → compliance 模式。"""
        import requests

        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.precision_param, timeout=_HTTP_TIMEOUT_S)
        time.sleep(0.5)

        if joint_reset:
            logging.info("JOINT RESET")
            # joint reset can take >1s on hardware, use longer timeout
            requests.post(self.url + "jointreset", timeout=15.0)
            time.sleep(0.5)

        # 笛卡尔重置
        reset_pose = self.resetpos.copy()
        if self.config.random_reset:
            reset_pose[:2] += np.random.uniform(
                -self.config.random_xy_range,
                self.config.random_xy_range,
                (2,),
            )
            euler = self.config.reset_pose[3:].copy()
            euler[-1] += np.random.uniform(
                -self.config.random_rz_range,
                self.config.random_rz_range,
            )
            reset_pose[3:] = euler_2_quat(euler)

        self.interpolate_move(reset_pose, timeout=1.0)

        # 切换到 compliance 模式
        requests.post(self.url + "update_param", json=self.config.compliance_param, timeout=_HTTP_TIMEOUT_S)

    # ================================================================
    # HTTP 通信
    # ================================================================

    def _update_currpos(self):
        """通过 HTTP 获取机器人完整状态（biased：含 J1 bias）。"""
        import requests

        ps = requests.post(self.url + "getstate", timeout=_HTTP_TIMEOUT_S).json()
        self.currpos = np.array(ps["pose"])         # 7D: xyz + quat (biased FK)
        self.currvel = np.array(ps["vel"])           # 6D
        self.currforce = np.array(ps["force"])       # 3D
        self.currtorque = np.array(ps["torque"])     # 3D
        self.q = np.array(ps["q"])                   # 7D（biased）
        self.dq = np.array(ps["dq"])                 # 7D（bias 偏移不影响）
        self.curr_gripper_pos = float(ps["gripper_pos"])  # [0, 1]

    def _update_currpos_true(self):
        """拉 /getstate_true 拿 unbiased 关节 + TCP（privileged 作弊通道）。

        Fails loud: 如果 server 层出错，宁可崩溃也不要静默给出 biased 值伪装成 true。
        """
        import requests

        ps = requests.post(self.url + "getstate_true", timeout=_HTTP_TIMEOUT_S).json()
        self.currpos_true = np.array(ps["pose"])     # 7D: xyz + quat (true FK)
        self.q_true = np.array(ps["q"])              # 7D (unbiased)

    def _send_pos_command(self, pos: np.ndarray):
        """发送笛卡尔位姿命令 (7D: xyz + quat)。"""
        import requests

        self._recover()
        data = {"arr": np.array(pos, dtype=np.float32).tolist()}
        requests.post(self.url + "pose", json=data, timeout=_HTTP_TIMEOUT_S)

    def _send_gripper_command(self, pos: float, mode: str = "binary"):
        """发送夹爪命令。二值模式：>0.5 开, <-0.5 关。"""
        import requests

        if mode != "binary":
            raise NotImplementedError("仅支持 binary 夹爪控制")

        now = time.time()
        if (
            pos <= -0.5
            and self.curr_gripper_pos > 0.85
            and now - self.last_gripper_act > self.config.gripper_sleep
        ):
            requests.post(self.url + "close_gripper", timeout=_HTTP_TIMEOUT_S)
            self.last_gripper_act = now
            time.sleep(self.config.gripper_sleep)
        elif (
            pos >= 0.5
            and self.curr_gripper_pos < 0.85
            and now - self.last_gripper_act > self.config.gripper_sleep
        ):
            requests.post(self.url + "open_gripper", timeout=_HTTP_TIMEOUT_S)
            self.last_gripper_act = now
            time.sleep(self.config.gripper_sleep)

    def _recover(self):
        """清除机器人错误状态。"""
        import requests
        requests.post(self.url + "clearerr", timeout=_HTTP_TIMEOUT_S)

    def _set_encoder_bias(self, bias: list):
        """设置编码器偏差（通过 Flask Server → /encoder_bias topic → C++ 控制器）。

        Fails loud: swallowing network/server errors here would mean silently
        training with zero bias while believing the fault is injected, which
        is an experimental-result landmine.
        """
        import requests
        r = requests.post(self.url + "set_encoder_bias", json={"bias": bias}, timeout=2.0)
        r.raise_for_status()
