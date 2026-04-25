"""handle_resume_logic 的 Phase 2 白名单覆盖测试。

场景：resume 时 checkpoint 里 train_config.json 的 critic_only_online_steps /
online_step_before_learning 是旧值，用户当前 JSON 用新值启动——期望新值生效。
"""
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_cfg(critic_only: int, online_before: int, resume: bool, out_dir: str):
    """构造一个最小 cfg stub，只需要带 resume / output_dir / policy 两个字段。"""
    policy = types.SimpleNamespace(
        critic_only_online_steps=critic_only,
        online_step_before_learning=online_before,
    )
    return types.SimpleNamespace(
        resume=resume,
        output_dir=out_dir,
        policy=policy,
    )


class TestResumeWhitelistOverride:
    def test_not_resuming_returns_original(self, tmp_path):
        """resume=False → 原 cfg 原样返回，不触发 from_pretrained。"""
        from frrl.rl.core.learner import handle_resume_logic

        cfg = _make_cfg(500, 100, resume=False, out_dir=str(tmp_path))
        out = handle_resume_logic(cfg)
        assert out is cfg

    def test_resume_overrides_phase2_fields(self, tmp_path):
        """resume=True：新 JSON 的 Phase 2 超参覆盖 checkpoint 里的旧值。"""
        from frrl.rl.core import learner as learner_mod

        ckpt_dir = tmp_path / "checkpoints" / "last"
        (ckpt_dir / "pretrained_model").mkdir(parents=True)

        # checkpoint 里存的旧配置
        ckpt_cfg = _make_cfg(500, 500, resume=True, out_dir=str(tmp_path))
        # 当前命令行 / JSON 提供的新值
        new_cfg = _make_cfg(1000, 100, resume=True, out_dir=str(tmp_path))

        with patch.object(
            learner_mod.TrainRLServerPipelineConfig,
            "from_pretrained",
            return_value=ckpt_cfg,
        ):
            out = learner_mod.handle_resume_logic(new_cfg)

        assert out.policy.critic_only_online_steps == 1000
        assert out.policy.online_step_before_learning == 100
        assert out.resume is True

    def test_resume_unchanged_fields_stay_ckpt(self, tmp_path):
        """新旧一致时不改动，log 应该不出 override 行（行为测试通过不崩即 OK）。"""
        from frrl.rl.core import learner as learner_mod

        ckpt_dir = tmp_path / "checkpoints" / "last"
        (ckpt_dir / "pretrained_model").mkdir(parents=True)

        ckpt_cfg = _make_cfg(500, 100, resume=True, out_dir=str(tmp_path))
        new_cfg = _make_cfg(500, 100, resume=True, out_dir=str(tmp_path))

        with patch.object(
            learner_mod.TrainRLServerPipelineConfig,
            "from_pretrained",
            return_value=ckpt_cfg,
        ):
            out = learner_mod.handle_resume_logic(new_cfg)

        assert out.policy.critic_only_online_steps == 500
        assert out.policy.online_step_before_learning == 100

    def test_resume_missing_checkpoint_raises(self, tmp_path):
        """resume=True 但 checkpoint 目录缺失 → RuntimeError。"""
        from frrl.rl.core.learner import handle_resume_logic

        cfg = _make_cfg(500, 100, resume=True, out_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="No model checkpoint found"):
            handle_resume_logic(cfg)

    def test_not_resume_existing_dir_raises(self, tmp_path):
        """resume=False 但 output_dir/checkpoints/last 已存在 → RuntimeError 防覆盖。"""
        from frrl.rl.core.learner import handle_resume_logic

        ckpt_dir = tmp_path / "checkpoints" / "last"
        ckpt_dir.mkdir(parents=True)

        cfg = _make_cfg(500, 100, resume=False, out_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="already exists"):
            handle_resume_logic(cfg)


class TestResumableOverridesList:
    def test_whitelist_contains_phase2_fields(self):
        """保险：白名单字段就是 Phase 2 的两个关键超参，不能被误加/误删。"""
        from frrl.rl.core.learner import RESUMABLE_POLICY_OVERRIDES

        assert "critic_only_online_steps" in RESUMABLE_POLICY_OVERRIDES
        assert "online_step_before_learning" in RESUMABLE_POLICY_OVERRIDES


class TestResumeContractRevalidation:
    """Resume 路径契约校验：setattr 白名单覆盖后必须重跑 _validate_phase2。

    当前白名单（critic_only_online_steps + online_step_before_learning）整组
    一起覆盖时不会产生违反契约的中间态，因为两者总是从同一个 new_cfg 拿值；
    但这套防御针对的是**未来场景**：白名单扩展或局部 setattr 引入了非法组合。
    所以测试用 object.__setattr__ 绕过 __post_init__ 构造非法 new_cfg，验证
    revalidation 会捕获，证明 handle_resume_logic 内联了 _validate_phase2 调用。
    """

    def test_override_revalidates_phase2_contract(self, tmp_path):
        """new_cfg 状态非法（绕过 __post_init__）→ handle_resume_logic 必须抛。"""
        from frrl.rl.core import learner as learner_mod
        from frrl.policies.sac.configuration_sac import SACConfig

        ckpt_dir = tmp_path / "checkpoints" / "last"
        (ckpt_dir / "pretrained_model").mkdir(parents=True)

        # 构造非法状态的 new_cfg：(500, 500) 违反 critic_only > online_before
        illegal_policy = SACConfig()
        object.__setattr__(illegal_policy, "critic_only_online_steps", 500)
        object.__setattr__(illegal_policy, "online_step_before_learning", 500)
        new_cfg = types.SimpleNamespace(
            resume=True, output_dir=str(tmp_path), policy=illegal_policy
        )
        # ckpt 同样非法（确保 setattr 实际改写 + revalidation 触发）
        ckpt_policy = SACConfig()
        object.__setattr__(ckpt_policy, "critic_only_online_steps", 100)
        object.__setattr__(ckpt_policy, "online_step_before_learning", 100)
        ckpt_cfg = types.SimpleNamespace(
            resume=True, output_dir=str(tmp_path), policy=ckpt_policy
        )

        with patch.object(
            learner_mod.TrainRLServerPipelineConfig,
            "from_pretrained",
            return_value=ckpt_cfg,
        ):
            with pytest.raises(
                ValueError,
                match=r"critic_only_online_steps.*online_step_before_learning",
            ):
                learner_mod.handle_resume_logic(new_cfg)

    def test_legacy_ckpt_missing_phase2_fields_handled(self, tmp_path):
        """旧 ckpt 可能根本没有 critic_only_online_steps 这俩字段（Phase 2 引入前
        的版本）。getattr default sentinel 必须把它们落成新 cfg 提供的值，不抛
        AttributeError。
        """
        from frrl.rl.core import learner as learner_mod

        ckpt_dir = tmp_path / "checkpoints" / "last"
        (ckpt_dir / "pretrained_model").mkdir(parents=True)

        # 旧 ckpt：policy 上**没有** critic_only_online_steps / online_step_before_learning
        legacy_policy = types.SimpleNamespace()  # 空 namespace
        ckpt_cfg = types.SimpleNamespace(
            resume=True, output_dir=str(tmp_path), policy=legacy_policy
        )
        new_cfg = _make_cfg(1000, 100, resume=True, out_dir=str(tmp_path))

        with patch.object(
            learner_mod.TrainRLServerPipelineConfig,
            "from_pretrained",
            return_value=ckpt_cfg,
        ):
            out = learner_mod.handle_resume_logic(new_cfg)

        # 缺字段 → 用新 cfg 的值填上
        assert out.policy.critic_only_online_steps == 1000
        assert out.policy.online_step_before_learning == 100
