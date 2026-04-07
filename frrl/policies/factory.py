"""
策略工厂 — 精简版，仅支持SAC策略。
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import torch
from typing_extensions import Unpack

from frrl.configs.policies import PreTrainedConfig
from frrl.configs.types import FeatureType
from frrl.envs.configs import EnvConfig
from frrl.policies.pretrained import PreTrainedPolicy
from frrl.policies.sac.configuration_sac import SACConfig
from frrl.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from frrl.policies.utils import validate_visual_features_consistency
from frrl.processor import PolicyAction, PolicyProcessorPipeline
from frrl.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from frrl.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# 延迟导入：仅在需要数据集时使用
try:
    from frrl.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from frrl.datasets.utils import dataset_to_policy_features
except ImportError:
    LeRobotDatasetMetadata = None
    dataset_to_policy_features = None

try:
    from frrl.envs.utils import env_to_policy_features
except ImportError:
    env_to_policy_features = None


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """根据名称获取策略类。仅支持 sac 和 reward_classifier。"""
    if name == "sac":
        from frrl.policies.sac.modeling_sac import SACPolicy
        return SACPolicy
    elif name == "reward_classifier":
        from frrl.policies.sac.reward_model.modeling_classifier import Classifier
        return Classifier
    else:
        raise ValueError(f"不支持的策略类型 '{name}'。可用: sac, reward_classifier")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """根据类型创建策略配置。"""
    if policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    else:
        raise ValueError(f"不支持的策略类型 '{policy_type}'。可用: sac, reward_classifier")


class ProcessorConfigKwargs(TypedDict, total=False):
    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建策略的前处理和后处理pipeline。"""
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    if isinstance(policy_cfg, SACConfig):
        from frrl.policies.sac.processor_sac import make_sac_pre_post_processors
        return make_sac_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    elif isinstance(policy_cfg, RewardClassifierConfig):
        from frrl.policies.sac.reward_model.processor_classifier import make_classifier_processor
        return make_classifier_processor(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    else:
        raise ValueError(f"不支持的策略类型 '{policy_cfg.type}'")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: Any = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """创建策略模型实例。"""
    if ds_meta is None and env_cfg is None:
        raise ValueError("必须提供 dataset metadata 或 env config 之一。")

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}

    # 从数据集或环境推断features（仅在配置中未指定时）
    if not cfg.input_features or not cfg.output_features:
        features = None
        if ds_meta is not None and dataset_to_policy_features is not None:
            features = dataset_to_policy_features(ds_meta.features)
            if hasattr(ds_meta, 'stats') and ds_meta.stats:
                cfg.dataset_stats = ds_meta.stats
        if env_cfg is not None and env_to_policy_features is not None:
            env_features = env_to_policy_features(env_cfg)
            if env_features:
                features = env_features

        if features:
            if not cfg.output_features:
                cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
            if not cfg.input_features:
                cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    kwargs["config"] = cfg

    if cfg.pretrained_path:
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    if rename_map is None:
        validate_visual_features_consistency(cfg, cfg.input_features)

    return policy
