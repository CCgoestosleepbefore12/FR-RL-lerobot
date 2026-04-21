"""
frrl.envs - 统一环境注册（gym_frrl/ 命名空间）

所有环境共享同一个FrankaGymEnv基类和OSC控制器，
都支持 encoder_bias_config 参数进行编码器偏差注入。
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
# PandaPickCube: 抓取方块并举高
# ============================================================
register(
    id="gym_frrl/PandaPickCubeBase-v0",
    entry_point="frrl.envs.panda_pick_cube_env:PandaPickCubeGymEnv",
    max_episode_steps=200,
)

# ============================================================
# Keyboard-v0 变体（带完整wrapper栈：键盘干预 + 夹爪惩罚 + Viewer + 重置延迟）
# 动作空间 7D 直通 OSC（不经过 EEActionWrapper），orn 维度 base 层忽略
# ============================================================
register(
    id="gym_frrl/FRRLPandaPickPlaceKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/FRRLPandaPickPlace-v0",
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

# ============================================================
# 人机协作安全场景: PickPlaceSafe
# ============================================================

register(
    id="gym_frrl/PandaPickPlaceSafeBase-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
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

# ------------------------------------------------------------
# Exp 1 观测消融：31D / 28D / 25D，全部 BiasJ1Random
# ------------------------------------------------------------
_SAFE_BIAS_J1_RANDOM_KW = dict(
    encoder_bias_config=EncoderBiasConfig(
        enable=True,
        error_probability=1.0,
        target_joints=[0],
        bias_mode="random_uniform",
        bias_range=[-0.15, 0.15],
    ),
)

# 31D Full：完整观测（含 real_tcp + block + plate + hand）
register(
    id="gym_frrl/PandaPickPlaceSafeObs31BiasJ1Random-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
    kwargs={**_SAFE_BIAS_J1_RANDOM_KW, "obs_mode": "obs31"},
)
register(
    id="gym_frrl/PandaPickPlaceSafeObs31BiasJ1RandomKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickPlaceSafeObs31BiasJ1Random-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

# 28D：去 real_tcp，测试外部 TCP oracle 是否必要
register(
    id="gym_frrl/PandaPickPlaceSafeObs28BiasJ1Random-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
    kwargs={**_SAFE_BIAS_J1_RANDOM_KW, "obs_mode": "obs28"},
)
register(
    id="gym_frrl/PandaPickPlaceSafeObs28BiasJ1RandomKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickPlaceSafeObs28BiasJ1Random-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

# 25D：去 block/plate，保留 real_tcp + hand，测试显式物体坐标是否必要
register(
    id="gym_frrl/PandaPickPlaceSafeObs25BiasJ1Random-v0",
    entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
    max_episode_steps=200,
    kwargs={**_SAFE_BIAS_J1_RANDOM_KW, "obs_mode": "obs25"},
)
register(
    id="gym_frrl/PandaPickPlaceSafeObs25BiasJ1RandomKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaPickPlaceSafeObs25BiasJ1Random-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

# ------------------------------------------------------------
# Exp 1 观测消融（Task Policy 变体）: 27D / 24D / 21D
#   - 与 Safe obs31/28/25 逐一对应，去掉 hand 4 维
#   - 复用 Safe 场景 XML，但不激活人手、关闭安全层
#   - 作为 Modular/Shielded 架构中的 Task Policy
# ------------------------------------------------------------
_TASK_BIAS_J1_RANDOM_KW = dict(
    encoder_bias_config=EncoderBiasConfig(
        enable=True,
        error_probability=1.0,
        target_joints=[0],
        bias_mode="random_uniform",
        bias_range=[-0.15, 0.15],
    ),
    hand_appear_prob=0.0,
    safety_layer_enabled=False,
    action_scale=0.025,
)

for _dim in (27, 24, 21):
    _base_id = f"gym_frrl/PandaPickPlaceTaskObs{_dim}BiasJ1Random-v0"
    register(
        id=_base_id,
        entry_point="frrl.envs.panda_pick_place_safe_env:PandaPickPlaceSafeEnv",
        max_episode_steps=200,
        kwargs={**_TASK_BIAS_J1_RANDOM_KW, "obs_mode": f"obs{_dim}"},
    )
    register(
        id=f"gym_frrl/PandaPickPlaceTaskObs{_dim}BiasJ1RandomKeyboard-v0",
        entry_point="frrl.envs.wrappers.factory:make_env",
        max_episode_steps=200,
        kwargs={
            "env_id": _base_id,
            "use_viewer": True,
            "gripper_penalty": -0.05,
            "use_inputs_control": True,
        },
    )

# ============================================================
# Backup Policy 训练环境
# ============================================================

# S1: 单个移动障碍物（28D 观测），TRACKING 模式，20 步 episode
register(
    id="gym_frrl/PandaBackupPolicyS1-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=20,
    kwargs={"num_obstacles": 1},
)

# S1 无 DR（用于对比实验）
register(
    id="gym_frrl/PandaBackupPolicyS1NoDR-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=20,
    kwargs={"num_obstacles": 1, "enable_dr": False},
)

# S2: 2 移动障碍物（38D 观测）；旧 checkpoint 兼容保留 10 步 episode
register(
    id="gym_frrl/PandaBackupPolicyS2-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"num_obstacles": 2},
)

_BIAS_J1_CONFIG = EncoderBiasConfig(
    enable=True,
    error_probability=1.0,
    target_joints=[0],
    bias_mode="random_uniform",
    bias_range=[-0.15, 0.15],
)

# S1 + 编码器偏差（TRACKING 模式，20 步 episode）
register(
    id="gym_frrl/PandaBackupPolicyS1BiasJ1-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=20,
    kwargs={"num_obstacles": 1, "encoder_bias_config": _BIAS_J1_CONFIG},
)

# S2 + 编码器偏差
register(
    id="gym_frrl/PandaBackupPolicyS2BiasJ1-v0",
    entry_point="frrl.envs.panda_backup_policy_env:PandaBackupPolicyEnv",
    max_episode_steps=10,
    kwargs={"num_obstacles": 2, "encoder_bias_config": _BIAS_J1_CONFIG},
)

# 键盘可视化调试（基于 S1）
register(
    id="gym_frrl/PandaBackupPolicyKeyboard-v0",
    entry_point="frrl.envs.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_frrl/PandaBackupPolicyS1-v0",
        "use_viewer": True,
        "gripper_penalty": 0.0,
        "use_inputs_control": True,
    },
)
