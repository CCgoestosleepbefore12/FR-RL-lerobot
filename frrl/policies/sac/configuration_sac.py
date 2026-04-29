# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from frrl.configs.policies import PreTrainedConfig
from frrl.configs.types import NormalizationMode
from frrl.optim.optimizers import MultiAdamConfig
from frrl.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature.

    Args:
        key: The feature key to check

    Returns:
        True if the key represents an image feature, False otherwise
    """
    return key.startswith(OBS_IMAGE)


@dataclass
class ConcurrencyConfig:
    """Configuration for the concurrency of the actor and learner.
    Possible values are:
    - "threads": Use threads for the actor and learner.
    - "processes": Use processes for the actor and learner.
    """

    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # MLP 内部用 getattr(nn, activations)() 实例化，所以必须是 nn 类名（"SiLU" / "Tanh" / "ReLU"）。
    # hil-serl/RLPD baseline 用 "Tanh"；保持默认 "SiLU" 不影响现有 config。
    activations: str = "SiLU"
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activations: str = "SiLU"
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("sac")
@dataclass
class SACConfig(PreTrainedConfig):
    """Soft Actor-Critic (SAC) configuration.

    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It learns a policy and a Q-function simultaneously
    using experience collected from the environment.

    This configuration class contains all the parameters needed to define a SAC agent,
    including network architectures, optimization settings, and algorithm-specific
    hyperparameters.
    """

    # Mapping of feature types to normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Statistics for normalizing different types of inputs
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # Architecture specifics
    # Device to run the model on (e.g., "cuda", "cpu")
    device: str = "cpu"
    # Device to store the model on
    storage_device: str = "cpu"
    # Name of the vision encoder model. Auto-detects CNN vs ViT in PretrainedImageEncoder.
    # Default in 11 train configs (2026-04-26 起): "facebook/dinov3-vits16-pretrain-lvd1689m"
    #   (ViT-S/16, 22M frozen, LVD-1689M SSL pretrained; gated repo → huggingface-cli login)
    # Legacy CNN: "helper2424/resnet10" (5M frozen, ImageNet supervised, hil-serl baseline)
    vision_encoder_name: str | None = None
    # Whether to freeze the vision encoder during training
    freeze_vision_encoder: bool = True
    # Hidden dimension size for the image encoder
    image_encoder_hidden_dim: int = 32
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = True
    # Number of discrete actions, eg for gripper actions
    num_discrete_actions: int | None = None
    # Dimension of the image embedding pooling
    image_embedding_pooling_dim: int = 8

    # Training parameter
    # Number of steps for online training
    online_steps: int = 1000000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100
    # HIL-SERL offline→online 过渡：进 online 主循环后，在真机 interaction step
    # 数 < critic_only_online_steps 期间，只训 critic，冻结 actor + temperature。
    # 目的：pretrain 过的 policy 先采数据，让 critic 适配真机分布，等 critic
    # 稳定后再解冻 actor，避免冷启动时 critic/actor 同时被稀疏 online 数据冲击。
    # 设 0 关闭（等价于一进主循环 actor+critic 同步更新的原行为）。
    critic_only_online_steps: int = 0
    # Offline-demo warmup: number of gradient steps on offline buffer before
    # switching to online interaction. Set to 0 to disable warmup.
    offline_warmup_steps: int = 500
    # Pretrain-only mode: when True, learner skips gRPC actor server, runs
    # `offline_pretrain_steps` gradient updates on offline_replay_buffer
    # (populated from demo pickles), saves a checkpoint, and exits. Online
    # training can then resume from that checkpoint. Use for HIL-SERL style
    # pretraining from SpaceMouse demos before wiring up the real-robot actor.
    offline_only_mode: bool = False
    # Number of gradient steps to run in offline_only_mode. Ignored unless
    # offline_only_mode == True.
    offline_pretrain_steps: int = 5000
    # Pickle paths for demo transitions (hil-serl schema). 非空时会在 learner
    # 启动时由 `initialize_offline_replay_buffer` 加载成 offline_replay_buffer。
    # 对两种模式都适用：
    #   * offline_only_mode=True  → pretrain-only，跑完 offline_pretrain_steps 退出
    #   * offline_only_mode=False → HIL-SERL 混合模式，先 offline_warmup_steps
    #     warmup，再进 online 主循环，RLPD 50/50 mix offline/online buffer
    demo_pickle_paths: list[str] = field(default_factory=list)
    # Pickle → buffer state-key rename map. Used by from_pickle_transitions to
    # turn collect-time schema (e.g. "pixels.front", "agent_pos") into the
    # "observation.*" convention SAC.validate_features requires.
    demo_key_map: dict[str, str] = field(default_factory=dict)
    # List of buffer-side state keys whose tensors should be HWC→CHW permuted
    # at load time. Image encoders expect CHW; our pickles store HWC uint8.
    demo_transpose_hwc_to_chw: list[str] = field(default_factory=list)
    # Buffer-side image key → target (H, W) for bilinear resize at load time.
    # Needed when collected demos have a different resolution than the policy
    # expects (e.g. old 224² front pickles but config now uses 128²). Runs
    # AFTER the transpose, so expects CHW layout.
    demo_resize_images: dict[str, list[int]] = field(default_factory=dict)
    # Buffer-side image keys to divide by 255 (uint8 [0,255] → float32 [0,1])
    # at load time. Must match the per-key normalization the online path does
    # in VanillaObservationProcessorStep — otherwise the encoder sees a 255×
    # offset between pretrain and online phases.
    demo_normalize_to_unit: list[str] = field(default_factory=list)
    # Frequency of policy updates
    policy_update_freq: int = 1

    # SAC algorithm parameters
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Initial temperature value
    temperature_init: float = 1.0
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the actor network
    actor_lr: float = 3e-4
    # Learning rate for the temperature parameter
    temperature_lr: float = 3e-4
    # Weight for the critic target update
    critic_target_update_weight: float = 0.005
    # Update-to-data ratio for the UTD algorithm (If you want enable utd_ratio, you need to set it to >1)
    utd_ratio: int = 1
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256
    # Target entropy for the SAC algorithm
    target_entropy: float | None = None
    # Whether to use backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # P0-2: pretrain 阶段冻结 log_alpha，防止 demo 高似然把 α 拉到 1e-3 以下导致
    # entropy bonus 消失、policy 在真机零探索。只在 offline warmup/pretrain 期间冻，
    # 进入 online phase 后正常学。
    freeze_temperature_in_pretrain: bool = True
    # 同上保护 actor：BC resume 后的 warmup 在 demo offline buffer 上训 critic，
    # 但此时 critic 是 random init，actor + critic 联合更新会让 actor gradient
    # = -∇(α·log_prob) - ∇min_q 里 min_q 项乱推 actor 偏离 BC ckpt，导致 online
    # phase 第二条 episode 起 actor 行为崩坏。symmetric 于 critic_only_online_steps
    # 在 online 阶段冻 actor 等 critic calibrate。建议 BC resume + warmup 流程开。
    freeze_actor_in_warmup: bool = True
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0

    # Network configuration
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the policy parameters
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    # torch.compile can break on some GPU / Python / CUDA combos; all current
    # production configs (configs/train_hil_sac_*.json) disable it explicitly.
    # Default is False so the repo works out of the box; enable in a config when
    # you've confirmed it compiles cleanly on your setup.
    use_torch_compile: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Phase 2 契约：critic_only_online_steps 允许负数但语义等同 0（关闭），
        # 这里规范化一次避免下游每个使用点都防御判断。
        if self.critic_only_online_steps < 0:
            self.critic_only_online_steps = 0
        self._validate_phase2()

    def _validate_phase2(self) -> None:
        """Phase 2 critic-only warmup 窗口合法性校验。

        抽出成独立方法是为了让 resume 路径（handle_resume_logic 白名单 setattr
        覆盖后）也能复用同一套校验，避免 setattr 绕过 __post_init__ 导致契约
        drift（"前端显示 Phase 2，实际窗口被 cold-start 吃光"）。
        """
        # online_step_before_learning 是 cold-start 冷启动阈值（buffer 不够就不训）；
        # critic_only_online_steps 是 Phase 2 窗口结束点。要 Phase 2 真正生效，
        # 必须 critic_only_online_steps > online_step_before_learning，否则 Phase 2
        # 窗口被 cold-start 完全吃掉，actor 一解冻就立刻被允许训——等于没启用。
        if (
            self.critic_only_online_steps > 0
            and self.critic_only_online_steps <= self.online_step_before_learning
        ):
            raise ValueError(
                f"critic_only_online_steps ({self.critic_only_online_steps}) must be > "
                f"online_step_before_learning ({self.online_step_before_learning}) "
                "to enable Phase 2 critic-only warmup on real env."
            )

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> list | None:
        return None
