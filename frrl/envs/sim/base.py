"""
Franka Panda 机器人环境基类

统一的MuJoCo环境基础设施：
- MujocoGymEnv: 低层MuJoCo + Gym接口
- FrankaGymEnv: Panda机器人控制（OSC）+ 编码器偏差注入

所有任务环境（PickPlace、PickCube、Safe、Backup）都继承FrankaGymEnv，
只需实现 reset()、step()、_compute_observation()、_compute_reward()。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from frrl.envs.sim.opspace import mat_to_quat, opspace
from frrl.fault_injection import EncoderBiasConfig, EncoderBiasInjector

MAX_GRIPPER_COMMAND = 255


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """低层MuJoCo环境，提供模型加载、渲染、基础属性。"""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        if hasattr(viewer, "close") and callable(viewer.close):
            try:
                viewer.close()
            except Exception:
                pass
        self._viewer = None

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random


class FrankaGymEnv(MujocoGymEnv):
    """
    Franka Panda 机器人环境基类。

    提供：
    - OSC操作空间控制（opspace）
    - 编码器偏差注入（encoder bias injection）
    - 双相机渲染（front + wrist）
    - 标准18D机器人状态（qpos7 + qvel7 + gripper1 + tcp3）

    编码器偏差的因果链：
        bias → q_measured = q_true + bias
          ├→ OSC用biased q计算Jacobian → 力矩不准（执行偏差）
          └→ FK(biased q) → 错误的TCP位置（感知偏差）
        物理仿真用真实q，不受影响。
    """

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)),  # noqa: B008
        cartesian_bounds: np.ndarray = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]]),  # noqa: B008
        action_scale: float = 1.0,
        encoder_bias_config: Optional[EncoderBiasConfig] = None,
    ):
        if xml_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            xml_path = project_root / "assets" / "pick_cube_scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds
        self._action_scale = action_scale

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }
        self.render_mode = render_mode
        self.image_obs = image_obs

        # 相机
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)

        # 机器人ID缓存
        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id

        # 编码器偏差注入器
        if encoder_bias_config is not None:
            self.bias_injector = EncoderBiasInjector(encoder_bias_config)
        else:
            self.bias_injector = None

        # 观测/动作空间（子类可覆盖）
        self._setup_observation_space()
        self._setup_action_space()

        # 离屏渲染器（图像观测用）
        self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
        self._viewer.render()

    def _setup_observation_space(self):
        agent_dim = 18  # qpos7 + qvel7 + gripper1 + tcp3
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)

        if self.image_obs:
            self.observation_space = spaces.Dict({
                "agent_pos": agent_box,
                "pixels": spaces.Dict({
                    "front": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                    "wrist": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                }),
            })
        else:
            self.observation_space = spaces.Dict({
                "agent_pos": agent_box,
                "environment_state": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            })

    def _setup_action_space(self):
        self.action_space = spaces.Box(
            low=np.full(7, -1.0, dtype=np.float32),
            high=np.full(7, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    # ================================================================
    # 机器人控制
    # ================================================================

    def reset_robot(self):
        """
        重置机器人到home位置。

        模拟真实场景：控制器发送home指令，但编码器有bias时，
        关节PD控制器用biased反馈，以为到了home，实际停在 home - bias。
        所以真实关节位置、末端位置都因bias而偏移。
        """
        # Episode偏差注入（必须在设置关节位置之前）
        if self.bias_injector:
            active = self.bias_injector.on_episode_start(num_joints=7)
            if active:
                bias = self.bias_injector.current_bias
                logging.info(f"Episode {self.bias_injector.episode_count}: "
                             f"编码器偏差注入 | bias = {np.round(bias, 4)}")

        # 真实场景：控制器命令关节到home，但编码器反馈 q_measured = q_true + bias
        # 关节PD控制器稳态条件：q_measured = q_home → q_true = q_home - bias
        bias = self._get_current_bias()
        if bias is not None:
            self._data.qpos[self._panda_dof_ids] = self._home_position - bias[:7]
        else:
            self._data.qpos[self._panda_dof_ids] = self._home_position

        self._data.ctrl[self._panda_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # 初始化mocap目标（控制器用biased FK算出的"当前位置"作为起点）
        # 无bias时：mocap = FK(home) = 真实位置
        # 有bias时：mocap = biased_FK(home - bias) ，控制器以为自己在这里
        self._data.mocap_pos[0] = self._get_tcp_pos()
        self._data.mocap_quat[0] = self._get_tcp_quat()

    def apply_action(self, action):
        """
        执行7D动作: [dx, dy, dz, rx, ry, rz, gripper]。

        编码器偏差注入点：在每个substep的OSC计算前，
        临时将qpos替换为biased值 → Jacobian和FK基于错误状态 → 力矩偏差。
        物理仿真始终用真实qpos。
        """
        x, y, z, rx, ry, rz, grasp_command = action

        # 更新mocap目标位置
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale
        npos = np.clip(pos + dpos, *self._cartesian_bounds)
        self._data.mocap_pos[0] = npos

        # 更新夹爪
        g = self._data.ctrl[self._gripper_ctrl_id] / MAX_GRIPPER_COMMAND
        ng = np.clip(g + grasp_command, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * MAX_GRIPPER_COMMAND

        bias = self._get_current_bias()

        for _ in range(self._n_substeps):
            if bias is not None:
                # 编码器偏差注入：临时替换qpos → OSC基于错误状态计算力矩
                true_qpos = self._data.qpos[self._panda_dof_ids].copy()
                self._data.qpos[self._panda_dof_ids] = true_qpos + bias[:7]
                mujoco.mj_forward(self._model, self._data)

            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=self._home_position,
                gravity_comp=True,
            )

            if bias is not None:
                # 恢复真实qpos再执行物理仿真
                self._data.qpos[self._panda_dof_ids] = true_qpos
                mujoco.mj_forward(self._model, self._data)

            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

    # ================================================================
    # 状态观测
    # ================================================================

    def get_robot_state(self) -> np.ndarray:
        """
        18D机器人状态: qpos(7) + qvel(7) + gripper(1) + tcp_pos(3)。

        有编码器偏差时：
          - qpos → q_true + bias（编码器读数）
          - qvel → 不受影响（固定bias导数为0）
          - tcp_pos → FK(biased qpos)（感知偏差）
        """
        bias = self._get_current_bias()

        qpos_true = self._data.qpos[self._panda_dof_ids].astype(np.float32)
        qvel = self._data.qvel[self._panda_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()

        if bias is not None:
            qpos_measured = qpos_true + bias[:7].astype(np.float32)
            tcp_pos = self._get_biased_tcp_pos(bias)
        else:
            qpos_measured = qpos_true
            tcp_pos = self._data.sensor("panda_hand/pinch_pos").data.astype(np.float32)

        return np.concatenate([qpos_measured, qvel, gripper_pose, tcp_pos])

    def get_gripper_pose(self) -> np.ndarray:
        """夹爪开度，归一化到 [0, 1]。Franka Hand: 0=关闭，1=全开。"""
        return np.array([self._data.ctrl[self._gripper_ctrl_id] / MAX_GRIPPER_COMMAND], dtype=np.float32)

    def _get_tcp_pos(self) -> np.ndarray:
        """获取TCP位置（有bias时用biased FK）。"""
        bias = self._get_current_bias()
        if bias is not None:
            return self._get_biased_tcp_pos(bias)
        return self._data.sensor("panda_hand/pinch_pos").data.copy()

    def _get_biased_tcp_pos(self, bias: np.ndarray) -> np.ndarray:
        """通过biased FK计算TCP位置（临时改qpos → mj_forward → 读site_xpos → 恢复）。"""
        true_qpos = self._data.qpos[self._panda_dof_ids].copy()
        self._data.qpos[self._panda_dof_ids] = true_qpos + bias[:7]
        mujoco.mj_forward(self._model, self._data)
        biased_pos = self._data.site_xpos[self._pinch_site_id].copy().astype(np.float32)
        self._data.qpos[self._panda_dof_ids] = true_qpos
        mujoco.mj_forward(self._model, self._data)
        return biased_pos

    def _get_tcp_quat(self) -> np.ndarray:
        """获取TCP姿态四元数（有bias时用biased FK）。"""
        bias = self._get_current_bias()
        if bias is not None:
            true_qpos = self._data.qpos[self._panda_dof_ids].copy()
            self._data.qpos[self._panda_dof_ids] = true_qpos + bias[:7]
            mujoco.mj_forward(self._model, self._data)
            xmat = self._data.site_xmat[self._pinch_site_id].reshape(3, 3)
            quat = mat_to_quat(xmat)
            self._data.qpos[self._panda_dof_ids] = true_qpos
            mujoco.mj_forward(self._model, self._data)
            return quat
        xmat = self._data.site_xmat[self._pinch_site_id].reshape(3, 3)
        return mat_to_quat(xmat)

    def _get_current_bias(self) -> Optional[np.ndarray]:
        """获取当前episode的编码器偏差向量。"""
        if self.bias_injector is not None and self.bias_injector.is_active:
            return self.bias_injector.current_bias
        return None

    # ================================================================
    # 渲染
    # ================================================================

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames

    def close(self) -> None:
        super().close()
