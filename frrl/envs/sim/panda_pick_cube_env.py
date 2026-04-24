"""
Panda Pick Cube 环境（Forked from gym_hil）

任务：抓取方块并举高0.2m。
支持编码器偏差注入。
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.sim.base import FrankaGymEnv, GymRenderingSpec
from frrl.fault_injection import EncoderBiasConfig

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.15, -0.4, 0], [0.8, 0.4, 0.6]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickCubeGymEnv(FrankaGymEnv):
    """抓取方块并举高的环境。"""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
    ):
        self.reward_type = reward_type
        project_root = Path(__file__).parent.parent.parent.parent
        super().__init__(
            xml_path=project_root / "assets" / "pick_cube_scene.xml",
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

        self._block_z = self._model.geom("block").size[2]
        self._random_block_position = random_block_position

        # 覆盖观测空间: robot_state(18D) + block_pos(3D) + noisy_real_tcp(3D) = 24D
        agent_dim = self.get_robot_state().shape[0] + 6  # +3 block_pos +3 noisy_real_tcp
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

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        if self._random_block_position:
            block_xy = self._random.uniform(*_SAMPLING_BOUNDS)
        else:
            block_xy = np.asarray([0.5, 0.0])
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2
        return self._compute_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)

        obs = self._compute_observation()
        success = self._is_success()
        rew = self._compute_reward()

        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = (
            np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05))
            or np.any(block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05))
        )
        terminated = bool(success or exceeded_bounds)
        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        robot_state = self.get_robot_state().astype(np.float32)
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)

        # 真实末端位置 + 5mm噪声（模拟外部定位系统：overhead RGBD相机 + marker追踪）
        real_tcp = self._data.site_xpos[self._pinch_site_id].copy().astype(np.float32)
        noisy_real_tcp = real_tcp + self._random.normal(0, 0.005, 3).astype(np.float32)

        # agent_pos = robot_state(18D) + block_pos(3D) + noisy_real_tcp(3D) = 24D
        agent_pos = np.concatenate([robot_state, block_pos, noisy_real_tcp])

        if self.image_obs:
            front_view, wrist_view = self.render()
            return {"pixels": {"front": front_view, "wrist": wrist_view}, "agent_pos": agent_pos}
        return {"agent_pos": agent_pos, "environment_state": block_pos}

    def _compute_reward(self) -> float:
        if self.reward_type == "dense":
            block_pos = self._data.sensor("block_pos").data
            tcp_pos = self._data.sensor("panda_hand/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = np.clip((block_pos[2] - self._z_init) / (self._z_success - self._z_init), 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        # sparse: 必须真正抓住（距离<5cm）且举高>20cm
        return float(self._is_success())

    def _is_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("panda_hand/pinch_pos").data
        return np.linalg.norm(block_pos - tcp_pos) < 0.05 and (block_pos[2] - self._z_init) > 0.2
