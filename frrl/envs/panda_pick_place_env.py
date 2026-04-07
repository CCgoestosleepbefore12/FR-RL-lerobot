"""
FR-RL Panda Pick-and-Place 环境

任务：抓取物块放到指定板上。
编码器偏差同时影响OSC控制和观测（执行偏差+感知偏差）。
奖励基于真实物理状态，不受偏差影响。
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.base import FrankaGymEnv, GymRenderingSpec
from frrl.fault_injection import EncoderBiasConfig

_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0.0], [0.6, 0.3, 0.5]])
# 物块采样范围（x, y）
_SAMPLING_BOUNDS = np.asarray([[0.35, -0.2], [0.55, 0.2]])


class PandaPickPlaceEnv(FrankaGymEnv):
    """
    Panda pick-and-place 环境（支持编码器偏差注入）。

    任务：抓取物块(block)放到目标板(plate)上。
    成功条件：物块xy距离板<8cm且高度<5cm。

    观测:
      - agent_pos: 18D = qpos(7) + qvel(7) + gripper(1) + tcp_pos(3)
      - environment_state: 6D = block_pos(3) + plate_pos(3)（state-only模式）
      - pixels: front + wrist 相机（image模式）
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        action_scale: float = 1.0,
        reward_type: str = "sparse",
        random_block_position: bool = False,
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
    ):
        self.reward_type = reward_type
        self._random_block_position = random_block_position

        project_root = Path(__file__).parent.parent.parent
        super().__init__(
            xml_path=project_root / "assets" / "scene.xml",
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            action_scale=action_scale,
            encoder_bias_config=encoder_bias_config,
        )

        # 任务物体ID
        self._block_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self._plate_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self._block_z = self._model.geom("block").size[2]

        # 覆盖observation_space: environment_state是6D（block3 + plate3）
        if not self.image_obs:
            agent_dim = self.get_robot_state().shape[0]
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32),
                "environment_state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
            })

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        # 物块初始位置
        if self._random_block_position:
            block_xy = self._random.uniform(*_SAMPLING_BOUNDS)
        else:
            block_xy = np.asarray([0.5, 0.15])
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)

        obs = self._compute_observation()
        reward = self._compute_reward()
        success = self._is_success()
        info = {"succeed": success}

        # 物块出界检测 — 物块掉出工作空间则终止
        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = (
            np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.1))
            or np.any(block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.1))
            or block_pos[2] > 0.4  # 物块飞太高
        )

        terminated = bool(success or exceeded_bounds)

        if self.bias_injector and self.bias_injector.is_active:
            info["encoder_bias"] = {
                "active": True,
                "bias": self.bias_injector.current_bias.tolist(),
            }

        return obs, reward, terminated, False, info

    def _compute_observation(self) -> Dict[str, Any]:
        robot_state = self.get_robot_state()

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {
                "agent_pos": robot_state,
                "pixels": {"front": front_view, "wrist": wrist_view},
            }

        block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        plate_pos = self._data.sensor("plate_pos").data.astype(np.float32)
        return {
            "agent_pos": robot_state,
            "environment_state": np.concatenate([block_pos, plate_pos]),
        }

    def _compute_reward(self) -> float:
        """基于真实物理状态计算奖励（不受编码器偏差影响）。"""
        block_pos = self._data.sensor("block_pos").data
        plate_pos = self._data.sensor("plate_pos").data
        tcp_pos = self._data.sensor("panda_hand/pinch_pos").data

        if self.reward_type == "dense":
            # 接近物块 + 物块接近目标板
            dist_to_block = np.linalg.norm(block_pos - tcp_pos)
            dist_to_plate = np.linalg.norm(block_pos[:2] - plate_pos[:2])
            r_reach = np.exp(-20 * dist_to_block)
            r_place = np.exp(-20 * dist_to_plate)
            return 0.3 * r_reach + 0.7 * r_place

        # sparse
        block_to_plate = np.linalg.norm(block_pos[:2] - plate_pos[:2])
        return float(block_to_plate < 0.08 and block_pos[2] < 0.05)

    def _is_success(self) -> bool:
        """判断任务是否成功完成。"""
        block_pos = self._data.sensor("block_pos").data
        plate_pos = self._data.sensor("plate_pos").data
        block_to_plate = np.linalg.norm(block_pos[:2] - plate_pos[:2])
        return block_to_plate < 0.08 and block_pos[2] < 0.05
