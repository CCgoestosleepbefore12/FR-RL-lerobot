"""
环境包装工厂 — 创建带有完整wrapper栈的环境
"""

import gymnasium as gym

from frrl.envs.sim.panda_pick_cube_env import PandaPickCubeGymEnv
from frrl.envs.sim.panda_pick_place_env import PandaPickPlaceEnv
from frrl.envs.sim.panda_backup_policy_env import PandaBackupPolicyEnv
from frrl.envs.sim.panda_pick_place_safe_env import PandaPickPlaceSafeEnv
from frrl.envs.wrappers.hil_wrappers import (
    GripperPenaltyWrapper,
    InputsControlWrapper,
    ResetDelayWrapper,
)
from frrl.envs.wrappers.viewer_wrapper import PassiveViewerWrapper


def wrap_env(
    env: gym.Env,
    use_viewer: bool = False,
    use_gamepad: bool = False,
    use_gripper: bool = True,
    use_inputs_control: bool = False,
    auto_reset: bool = False,
    show_ui: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
    controller_config_path: str = None,
) -> gym.Env:
    """
    按顺序包装环境（7D 动作空间直通，不经过 EEActionWrapper）：
    GripperPenaltyWrapper → InputsControlWrapper → PassiveViewerWrapper → ResetDelayWrapper
    """
    if use_gripper:
        env = GripperPenaltyWrapper(env, penalty=gripper_penalty)

    if use_inputs_control:
        env = InputsControlWrapper(
            env,
            x_step_size=1.0,
            y_step_size=1.0,
            z_step_size=1.0,
            use_gripper=use_gripper,
            auto_reset=auto_reset,
            use_gamepad=use_gamepad,
            controller_config_path=controller_config_path,
        )

    if use_viewer:
        env = PassiveViewerWrapper(env, show_left_ui=show_ui, show_right_ui=show_ui)

    env = ResetDelayWrapper(env, delay_seconds=reset_delay_seconds)
    return env


# 环境类映射（substring 匹配）
_ENV_CLASSES = {
    "PandaPickCubeBase": PandaPickCubeGymEnv,
    "FRRLPandaPickPlace": PandaPickPlaceEnv,
    "PandaPickPlaceSafe": PandaPickPlaceSafeEnv,
    "PandaPickPlaceTask": PandaPickPlaceSafeEnv,  # Task = Safe class + hand_appear_prob=0（由 base kwargs 透传）
    "PandaBackupPolicy": PandaBackupPolicyEnv,
}


def make_env(
    env_id: str,
    use_viewer: bool = False,
    use_gamepad: bool = False,
    use_gripper: bool = True,
    use_inputs_control: bool = False,
    auto_reset: bool = False,
    show_ui: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
    controller_config_path: str | None = None,
    **kwargs,
) -> gym.Env:
    """创建base环境 + 包装wrapper栈。"""
    # 从env_id提取base名（去掉命名空间和版本号）
    # 例如 "gym_frrl/PandaPickCubeBase-v0" → "PandaPickCubeBase"
    task_name = env_id.split("/")[-1].rsplit("-", 1)[0]

    # 查找对应的环境类（substring 匹配，Task 变体复用 Safe class）
    env_cls = None
    for key, cls in _ENV_CLASSES.items():
        if key in task_name:
            env_cls = cls
            break

    if env_cls is None:
        raise ValueError(f"Unknown environment: {env_id}. Known bases: {list(_ENV_CLASSES.keys())}")

    # 从 base env 注册透传 kwargs（encoder_bias_config / obs_mode / hand_appear_prob 等）
    # Keyboard 注册里显式传的 kwargs 优先覆盖 base
    try:
        base_kwargs = dict(gym.spec(env_id).kwargs or {})
    except gym.error.NameNotFound:
        base_kwargs = {}
    merged_kwargs = {**base_kwargs, **kwargs}
    env = env_cls(**merged_kwargs)

    return wrap_env(
        env,
        use_viewer=use_viewer,
        use_gamepad=use_gamepad,
        use_gripper=use_gripper,
        use_inputs_control=use_inputs_control,
        auto_reset=auto_reset,
        show_ui=show_ui,
        gripper_penalty=gripper_penalty,
        reset_delay_seconds=reset_delay_seconds,
        controller_config_path=controller_config_path,
    )
