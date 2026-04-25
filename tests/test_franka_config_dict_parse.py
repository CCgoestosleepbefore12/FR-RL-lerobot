"""env_factory.py 的 franka_config dict → FrankaRealConfig 转换测试。

env_factory.make_robot_env 进入 franka 分支时，如果 cfg.franka_config 是 dict
（JSON 反序列化后的形态），需要构造 FrankaRealConfig。本测试只针对 dict→config
的转换逻辑，不触发真机 env 构造（会连 RT PC HTTP）。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_franka_branch(cfg_dict):
    """调用 env_factory 内部 helper，避免真正构造 FrankaRealEnv（会连 RT PC HTTP）。"""
    from frrl.rl.core.env_factory import _build_franka_config_from_dict
    return _build_franka_config_from_dict(cfg_dict)


class TestDictParse:
    def test_empty_dict_uses_defaults(self):
        """空 dict → 用全部默认值。"""
        from frrl.envs.real_config import FrankaRealConfig
        cfg = _get_franka_branch({})
        default = FrankaRealConfig()
        assert cfg.reward_backend == default.reward_backend  # "pose"
        assert cfg.max_episode_length == default.max_episode_length  # 300
        assert cfg.encoder_bias_config is None

    def test_scalar_override(self):
        """scalar 字段（reward_backend, enable_bias_monitor）可通过 dict 覆盖。"""
        cfg = _get_franka_branch({
            "reward_backend": "keyboard",
            "enable_bias_monitor": True,
        })
        assert cfg.reward_backend == "keyboard"
        assert cfg.enable_bias_monitor is True

    def test_encoder_bias_nested_dict(self):
        """encoder_bias_config 作为嵌套 dict 被 EncoderBiasConfig 解析。"""
        from frrl.fault_injection import EncoderBiasConfig
        cfg = _get_franka_branch({
            "reward_backend": "keyboard",
            "encoder_bias_config": {
                "enable": True,
                "error_probability": 1.0,
                "target_joints": [0],
                "bias_mode": "random_uniform",
                "bias_range": [-0.2, 0.2],
            },
        })
        assert isinstance(cfg.encoder_bias_config, EncoderBiasConfig)
        assert cfg.encoder_bias_config.enable is True
        assert cfg.encoder_bias_config.target_joints == [0]
        assert cfg.encoder_bias_config.bias_mode == "random_uniform"
        # bias_range 必须是 tuple（dataclass 类型声明为 Tuple[float, float]）
        assert isinstance(cfg.encoder_bias_config.bias_range, tuple)
        assert cfg.encoder_bias_config.bias_range == (-0.2, 0.2)

    def test_numpy_fields_retain_defaults(self):
        """dict 不覆盖时，numpy / callable 字段保留 Python 默认。"""
        import numpy as np
        from frrl.envs.real_config import FrankaRealConfig
        cfg = _get_franka_branch({"reward_backend": "keyboard"})
        default = FrankaRealConfig()
        # reset_pose 是 np.ndarray，不能用 == 直接比较布尔
        assert np.array_equal(cfg.reset_pose, default.reset_pose)
        assert np.array_equal(cfg.abs_pose_limit_low, default.abs_pose_limit_low)
        # image_crop 是 callable dict：保留默认 key
        assert set(cfg.image_crop.keys()) == set(default.image_crop.keys())


class TestFriendlyErrors:
    """P1 清理：解析失败时把底层 TypeError 包成 ValueError + 上下文，避免晦涩堆栈。"""

    def test_unknown_top_level_field(self):
        try:
            _get_franka_branch({"nonexistent_field": 123})
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "franka_config 解析失败" in str(e)
            assert "nonexistent_field" in str(e)

    def test_unknown_bias_field(self):
        try:
            _get_franka_branch({
                "encoder_bias_config": {
                    "enable": True,
                    "nonexistent_bias_field": 42,
                },
            })
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "encoder_bias_config 解析失败" in str(e)
            assert "nonexistent_bias_field" in str(e)


class TestConfigFileIntegrity:
    def test_task_real_config_parses(self):
        """scripts/configs/train_hil_sac_task_real.json 的 franka_config 段落可解析。"""
        import json
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.fault_injection import EncoderBiasConfig

        root = Path(__file__).resolve().parent.parent
        cfg_path = root / "scripts/configs/train_hil_sac_task_real.json"
        with open(cfg_path) as f:
            full_cfg = json.load(f)

        franka_dict = full_cfg["env"]["franka_config"]
        assert isinstance(franka_dict, dict)
        cfg = _get_franka_branch(franka_dict)

        # Sanity：S6 定下来的关键参数都活下来了
        assert isinstance(cfg, FrankaRealConfig)
        assert cfg.reward_backend == "keyboard"
        assert cfg.enable_bias_monitor is True
        assert isinstance(cfg.encoder_bias_config, EncoderBiasConfig)
        assert cfg.encoder_bias_config.target_joints == [0]

    def test_task_real_config_hil_params(self):
        """S6 锁定的 HIL 超参都落在 JSON 里。"""
        import json
        root = Path(__file__).resolve().parent.parent
        with open(root / "scripts/configs/train_hil_sac_task_real.json") as f:
            cfg = json.load(f)

        policy = cfg["policy"]
        assert policy["offline_only_mode"] is False
        assert policy["offline_pretrain_steps"] == 5000
        assert policy["offline_warmup_steps"] == 500
        assert policy["online_steps"] == 50000
        # Phase 2 配置（真机风险审查 P0-6 后调整）：
        #   online_step_before_learning=500：等 online buffer 够 1 个独立 batch
        #     再开训，避免 batch shrink 把同一 transition 反复梯度踩。
        #   critic_only_online_steps=2000：真机 ~3.5min / 6+ episode，让 critic
        #     在 RLPD 50/50 下充分适配 online 分布后再解冻 actor。
        # 必须 critic_only > online_before（SACConfig._validate_phase2 校验）。
        assert policy["online_step_before_learning"] == 500
        assert policy["critic_only_online_steps"] == 2000
        # gripper penalty 对齐 sim 的 -0.05（S6 review 决定激活）
        assert cfg["env"]["processor"]["gripper"]["gripper_penalty"] == -0.05
        # P0-2 真机风险审查：α 防坍塌 + 探索温度抬升
        assert policy["temperature_init"] == 0.1
        assert policy["target_entropy"] == -3.5
        # P0-1: 两路相机都必须配 demo_resize 防 cat shape 静默不一致
        assert "observation.images.front" in policy["demo_resize_images"]
        assert "observation.images.wrist" in policy["demo_resize_images"]
