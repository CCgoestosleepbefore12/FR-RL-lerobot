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

from frrl.envs.franka_real_config import FrankaRealConfig
from frrl.fault_injection import EncoderBiasConfig, EncoderBiasInjector

logging.basicConfig(level=logging.INFO)


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

    def __init__(self, config: FrankaRealConfig):
        super().__init__()
        self.config = config
        self.url = config.server_url

        # 动作/观测空间
        self.action_space = spaces.Box(
            low=np.full(7, -1.0, dtype=np.float32),
            high=np.full(7, 1.0, dtype=np.float32),
        )

        agent_dim = 18  # q(7) + dq(7) + gripper(1) + tcp(3)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32),
            "environment_state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
            "pixels": spaces.Dict({
                name: spaces.Box(0, 255, (*config.image_size, 3), dtype=np.uint8)
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

        # 运行时状态
        self.currpos = np.zeros(7)     # xyz + quat  (7D)
        self.currvel = np.zeros(6)     # 末端速度    (6D)
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        self.q = np.zeros(7)           # 关节角      (7D)
        self.dq = np.zeros(7)          # 关节角速度  (7D)
        self.curr_gripper_pos = 0.0
        self.curr_path_length = 0
        self.cycle_count = 0
        self.last_gripper_act = 0.0

        # 相机（延迟初始化，允许无相机的 mock 测试）
        self._camera_manager = None

        # ESC 紧急停止
        self.terminate = False

        logging.info(f"FrankaRealEnv 初始化: server={self.url}")

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
            else:
                self._set_encoder_bias([0.0] * 7)

        self._update_currpos()
        obs = self._compute_observation()
        self.terminate = False
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
        nextpos[:3] += xyz_delta * self.config.action_scale[0]
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
        obs = self._compute_observation()
        reward = self.compute_reward(obs)
        done = (
            self.curr_path_length >= self.config.max_episode_length
            or reward
            or self.terminate
        )
        return obs, int(reward), done, False, {"succeed": reward}

    def close(self):
        if self._camera_manager is not None:
            self._camera_manager.close()

    # ================================================================
    # 观测
    # ================================================================

    def get_robot_state(self) -> np.ndarray:
        """构建 18D 机器人状态: q(7) + dq(7) + gripper(1) + tcp(3)。

        gripper 归一化到 [0, 1]。
        如果有 bias 注入（通过 set_encoder_bias），getstate 返回的 q 已包含 bias。
        """
        # shape: (18,)
        return np.concatenate([
            self.q.astype(np.float32),                       # 7D 关节角（可能含bias）
            self.dq.astype(np.float32),                      # 7D 关节角速度
            np.array([self.curr_gripper_pos], dtype=np.float32),  # 1D 夹爪 [0,1]
            self.currpos[:3].astype(np.float32),             # 3D TCP 位置
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
        """从 RealSense 获取图像。无相机时返回 None。"""
        if self._camera_manager is None:
            try:
                from frrl.cameras.realsense import RealSenseCameraManager
                self._camera_manager = RealSenseCameraManager(
                    camera_configs=self.config.cameras,
                    image_crop=self.config.image_crop,
                    image_size=self.config.image_size,
                )
            except (ImportError, RuntimeError) as e:
                logging.warning(f"相机初始化失败（mock模式无相机正常）: {e}")
                return None

        try:
            return self._camera_manager.get_images()
        except Exception as e:
            logging.warning(f"相机读取失败: {e}")
            return None

    # ================================================================
    # 奖励
    # ================================================================

    def compute_reward(self, obs: Dict[str, Any]) -> bool:
        """基于位姿阈值的 sparse reward（pick-and-place）。"""
        if self.config.target_pose is None:
            return False

        current_pose = np.concatenate([
            self.currpos[:3],
            quat_2_euler(self.currpos[3:]),
        ])
        delta = np.abs(current_pose - self.config.target_pose)
        return bool(np.all(delta < self.config.reward_threshold))

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
        requests.post(self.url + "update_param", json=self.config.precision_param)
        time.sleep(0.5)

        if joint_reset:
            logging.info("JOINT RESET")
            requests.post(self.url + "jointreset")
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
        requests.post(self.url + "update_param", json=self.config.compliance_param)

    # ================================================================
    # HTTP 通信
    # ================================================================

    def _update_currpos(self):
        """通过 HTTP 获取机器人完整状态。"""
        import requests

        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])         # 7D: xyz + quat
        self.currvel = np.array(ps["vel"])           # 6D
        self.currforce = np.array(ps["force"])       # 3D
        self.currtorque = np.array(ps["torque"])     # 3D
        self.q = np.array(ps["q"])                   # 7D（可能含 bias）
        self.dq = np.array(ps["dq"])                 # 7D
        self.curr_gripper_pos = float(ps["gripper_pos"])  # [0, 1]

    def _send_pos_command(self, pos: np.ndarray):
        """发送笛卡尔位姿命令 (7D: xyz + quat)。"""
        import requests

        self._recover()
        data = {"arr": np.array(pos, dtype=np.float32).tolist()}
        requests.post(self.url + "pose", json=data)

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
            requests.post(self.url + "close_gripper")
            self.last_gripper_act = now
            time.sleep(self.config.gripper_sleep)
        elif (
            pos >= 0.5
            and self.curr_gripper_pos < 0.85
            and now - self.last_gripper_act > self.config.gripper_sleep
        ):
            requests.post(self.url + "open_gripper")
            self.last_gripper_act = now
            time.sleep(self.config.gripper_sleep)

    def _recover(self):
        """清除机器人错误状态。"""
        import requests
        requests.post(self.url + "clearerr")

    def _set_encoder_bias(self, bias: list):
        """设置编码器偏差（通过 Flask Server → ROS param → C++ 控制器）。"""
        import requests
        try:
            requests.post(self.url + "set_encoder_bias", json={"bias": bias})
        except Exception as e:
            logging.warning(f"设置 encoder bias 失败（Flask Server 可能未支持此路由）: {e}")
