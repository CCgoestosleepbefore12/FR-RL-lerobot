"""
Panda Arrange Boxes 环境（Forked from gym_hil）

任务：将多个彩色方块整理到对应目标位置。
支持编码器偏差注入。
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.base import FrankaGymEnv, GymRenderingSpec
from frrl.fault_injection import EncoderBiasConfig

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.5, 0], [0.6, 0.5, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaArrangeBoxesGymEnv(FrankaGymEnv):
    """整理多个盒子到目标位置的环境。"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
    ):
        self.reward_type = reward_type
        project_root = Path(__file__).parent.parent.parent
        super().__init__(
            xml_path=project_root / "assets" / "arrange_boxes_scene.xml",
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

        self._block_z = self._model.geom("block1").size[2]
        self.no_blocks = self._get_no_boxes()
        self.block_range = 0.3

        # 覆盖观测空间
        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

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
                "environment_state": env_box,
            })

    def _get_no_boxes(self):
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)
        ]
        return len([j for j in joint_names if j and "block" in j])

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        positions_coords = np.linspace(-self.block_range, self.block_range, self.no_blocks)
        self._random.shuffle(positions_coords)

        blocks = [f"block{i}" for i in range(1, self.no_blocks + 1)]
        self._random.shuffle(blocks)

        for block, pos in zip(blocks, positions_coords, strict=False):
            block_x_coord = self._data.joint(block).qpos[0]
            self._data.joint(block).qpos[:3] = (block_x_coord, pos, self._block_z)

        mujoco.mj_forward(self._model, self._data)
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)

        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        block_pos = self._data.sensor("block1_pos").data
        exceeded_bounds = (
            np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - self.block_range - 0.05))
            or np.any(block_pos[:2] > (_SAMPLING_BOUNDS[1] + self.block_range + 0.05))
        )
        terminated = bool(success or exceeded_bounds)
        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        robot_state = self.get_robot_state().astype(np.float32)
        block_pos = self._data.sensor("block1_pos").data.astype(np.float32)

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": robot_state}
        return {"agent_pos": robot_state, "environment_state": block_pos}

    def _get_sensors(self) -> Tuple[list, list]:
        return (
            [self._data.sensor(f"block{i}_pos") for i in range(1, self.no_blocks + 1)],
            [self._data.sensor(f"target{i}_pos") for i in range(1, self.no_blocks + 1)],
        )

    def _compute_reward(self) -> float:
        block_sensors, target_sensors = self._get_sensors()
        distance = np.linalg.norm(block_sensors[1].data - target_sensors[1].data)
        if self.reward_type == "dense":
            return np.exp(-20 * distance)
        return float(distance < 0.05)

    def _is_success(self) -> bool:
        block_sensors, target_sensors = self._get_sensors()
        return np.linalg.norm(block_sensors[1].data - target_sensors[1].data) < 0.05
