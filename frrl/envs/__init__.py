"""
frrl.envs - 统一环境注册（gym_frrl/ 命名空间）

所有环境共享同一个FrankaGymEnv基类和OSC控制器，
都支持 encoder_bias_config 参数进行编码器偏差注入。

环境变体:
  FR-RL 故障容错:
    - gym_frrl/FRRLPandaPickPlace-v0: pick-and-place 无故障baseline
    - gym_frrl/FRRLPandaPickPlaceBias*-v0: 各种偏差配置

  通用操作任务:
    - gym_frrl/PandaPickCubeBase-v0: 抓取方块并举高
    - gym_frrl/PandaArrangeBoxesBase-v0: 整理多个盒子
"""

from gymnasium.envs.registration import register

from frrl.fault_injection import EncoderBiasConfig
from frrl.envs.base import FrankaGymEnv, GymRenderingSpec  # noqa: F401
from frrl.envs.configs import EnvConfig, HILSerlRobotEnvConfig  # noqa: F401

# ============================================================
# FR-RL: 无故障 baseline
# ============================================================
register(
    id="gym_frrl/FRRLPandaPickPlace-v0",
    entry_point="frrl.envs.panda_pick_place_env:PandaPickPlaceEnv",
    max_episode_steps=200,
)

# ============================================================
# FR-RL: Joint 4 固定偏差 0.3 rad
# ============================================================
register(
    id="gym_frrl/FRRLPandaPickPlaceBiasJ4Fixed03-v0",
    entry_point="frrl.envs.panda_pick_place_env:PandaPickPlaceEnv",
    max_episode_steps=200,
    kwargs={
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[3],
            bias_mode="fixed",
            fixed_bias_value=0.3,
        ),
    },
)

# ============================================================
# FR-RL: Joint 4 随机偏差 [0, 1] rad
# ============================================================
register(
    id="gym_frrl/FRRLPandaPickPlaceBiasJ4Random-v0",
    entry_point="frrl.envs.panda_pick_place_env:PandaPickPlaceEnv",
    max_episode_steps=200,
    kwargs={
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[3],
            bias_mode="random_uniform",
            bias_range=[0.0, 1.0],
        ),
    },
)

# ============================================================
# FR-RL: 全关节随机偏差
# ============================================================
register(
    id="gym_frrl/FRRLPandaPickPlaceBiasAllJoints-v0",
    entry_point="frrl.envs.panda_pick_place_env:PandaPickPlaceEnv",
    max_episode_steps=200,
    kwargs={
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=0.3,
            target_joints=None,
            per_joint_probability=0.7,
            bias_mode="random_uniform",
            bias_range=[0.0, 0.5],
        ),
    },
)

# ============================================================
# PandaPickCube: 抓取方块并举高
# ============================================================
register(
    id="gym_frrl/PandaPickCubeBase-v0",
    entry_point="frrl.envs.panda_pick_cube_env:PandaPickCubeGymEnv",
    max_episode_steps=200,
)

# ============================================================
# PandaArrangeBoxes: 整理多个盒子
# ============================================================
register(
    id="gym_frrl/PandaArrangeBoxesBase-v0",
    entry_point="frrl.envs.panda_arrange_boxes_env:PandaArrangeBoxesGymEnv",
    max_episode_steps=200,
)

# ============================================================
# Keyboard-v0 变体（带完整wrapper栈：EEAction+键盘干预+夹爪惩罚+Viewer+重置延迟）
# 策略输出3D+离散夹爪，wrapper自动扩展为7D给OSC
# ============================================================
register(
    id="gym_frrl/FRRLPandaPickPlaceKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/FRRLPandaPickPlaceBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

register(
    id="gym_frrl/PandaPickCubeKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickCubeBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

register(
    id="gym_frrl/PandaArrangeBoxesKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaArrangeBoxesBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

# ============================================================
# PickCube + 编码器偏差 变体
# ============================================================
register(
    id="gym_frrl/PandaPickCubeBiasJ4Fixed02-v0",
    entry_point="frrl.envs.panda_pick_cube_env:PandaPickCubeGymEnv",
    kwargs={
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[3],
            bias_mode="fixed",
            fixed_bias_value=0.2,
        ),
    },
)

register(
    id="gym_frrl/PandaPickCubeBiasJ4Random-v0",
    entry_point="frrl.envs.panda_pick_cube_env:PandaPickCubeGymEnv",
    kwargs={
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[3],
            bias_mode="random_uniform",
            bias_range=[0.0, 0.25],
        ),
    },
)

# PickCube + 偏差 + Keyboard（env_id用Base，bias通过kwargs传给构造函数）
register(
    id="gym_frrl/PandaPickCubeBiasJ4Fixed02Keyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickCubeBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[3],
            bias_mode="fixed",
            fixed_bias_value=0.2,
        ),
    },
)

register(
    id="gym_frrl/PandaPickCubeBiasJ4RandomKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickCubeBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[3],
            bias_mode="random_uniform",
            bias_range=[0.0, 0.25],
        ),
    },
)

# ============================================================
# 人机协作安全场景: PickPlaceSafe
# ============================================================

register(
    id="gym_frrl/PandaPickPlaceSafeBase-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
)

register(
    id="gym_frrl/PandaPickPlaceSafeNoFreeze-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
    kwargs={"safety_layer_enabled": False},
)

register(
    id="gym_frrl/PandaPickPlaceSafeKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickPlaceSafeBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

register(
    id="gym_frrl/PandaPickPlaceSafeBiasJ1Random-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
    kwargs={
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[0],
            bias_mode="random_uniform",
            bias_range=[-0.15, 0.15],
        ),
    },
)

register(
    id="gym_frrl/PandaPickPlaceSafeBiasJ1RandomKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickPlaceSafeBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
        "encoder_bias_config": EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[0],
            bias_mode="random_uniform",
            bias_range=[-0.15, 0.15],
        ),
    },
)

# ============================================================
# Backup Policy 训练环境
# ============================================================

register(
    id="gym_frrl/PandaBackupPolicyBase-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
)

_BIAS_J1_CONFIG = EncoderBiasConfig(
    enable=True,
    error_probability=1.0,
    target_joints=[0],
    bias_mode="random_uniform",
    bias_range=[-0.15, 0.15],
)

register(
    id="gym_frrl/PandaBackupPolicyBiasJ1-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"encoder_bias_config": _BIAS_J1_CONFIG},
)

register(
    id="gym_frrl/PandaBackupPolicyFixed2BiasJ1-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 2, "fixed_obstacles": True, "encoder_bias_config": _BIAS_J1_CONFIG},
)

register(
    id="gym_frrl/PandaBackupPolicyFixed3BiasJ1-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 3, "fixed_obstacles": True, "encoder_bias_config": _BIAS_J1_CONFIG},
)

register(
    id="gym_frrl/PandaBackupPolicyMulti2-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 2},
)

register(
    id="gym_frrl/PandaBackupPolicyMulti3-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 3},
)

register(
    id="gym_frrl/PandaBackupPolicyFixed2-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 2, "fixed_obstacles": True},
)

register(
    id="gym_frrl/PandaBackupPolicyMulti3Prox-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 3, "use_proximity_reward": True},
)

register(
    id="gym_frrl/PandaBackupPolicyFixed3Prox-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 3, "fixed_obstacles": True, "use_proximity_reward": True},
)

register(
    id="gym_frrl/PandaBackupPolicyFixed3-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"max_obstacles": 3, "fixed_obstacles": True},
)

register(
    id="gym_frrl/PandaBackupPolicyKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaBackupPolicyBase-v0",
        "use_viewer": True,
        "gripper_penalty": 0.0,
        "use_inputs_control": True,
    },
)
