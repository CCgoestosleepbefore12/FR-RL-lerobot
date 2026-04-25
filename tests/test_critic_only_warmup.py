"""HIL-SERL Phase 2 (critic-only on real env) 相关测试。

覆盖两件事：
  1. SACConfig.critic_only_online_steps 字段默认值 / 可赋值 / 实例独立
  2. learner._should_freeze_actor 谓词的边界行为

主循环 actor-skip 的端到端验证依赖 gRPC + 真机，留到集成测试阶段。
这里只锁死"在什么情况下应该冻结 actor"这个契约。
"""
from unittest.mock import MagicMock

from frrl.policies.sac.configuration_sac import SACConfig
from frrl.rl.core.learner import _should_freeze_actor, _should_run_actor_optimization


class TestSACConfigFieldWiring:
    def test_default_is_zero(self):
        """默认 0 → 不改变现有行为。"""
        cfg = SACConfig()
        assert cfg.critic_only_online_steps == 0

    def test_is_assignable(self):
        cfg = SACConfig()
        cfg.critic_only_online_steps = 500
        assert cfg.critic_only_online_steps == 500


class TestSACConfigPhase2Contract:
    """__post_init__ 的 Phase 2 契约：负数规范化 + 窗口闭合错配检测。"""

    def test_negative_normalized_to_zero(self):
        """负数在 __post_init__ 里被规范化到 0（语义等同关闭）。"""
        cfg = SACConfig(critic_only_online_steps=-5)
        assert cfg.critic_only_online_steps == 0

    def test_zero_passes_regardless_of_before_learning(self):
        """0 表示关闭 Phase 2，不应该触发窗口大小校验。"""
        cfg = SACConfig(
            critic_only_online_steps=0,
            online_step_before_learning=1000,
        )
        assert cfg.critic_only_online_steps == 0

    def test_window_must_exceed_cold_start(self):
        """critic_only_online_steps <= online_step_before_learning → 窗口被 cold-start 吃掉，报错。"""
        try:
            SACConfig(
                critic_only_online_steps=500,
                online_step_before_learning=500,
            )
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "critic_only_online_steps" in str(e)
            assert "online_step_before_learning" in str(e)

    def test_valid_window_passes(self):
        """Phase 2 正确配置：窗口严格大于 cold-start 阈值。"""
        cfg = SACConfig(
            critic_only_online_steps=500,
            online_step_before_learning=100,
        )
        assert cfg.critic_only_online_steps == 500
        assert cfg.online_step_before_learning == 100


class TestShouldFreezeActor:
    """_should_freeze_actor 的语义：
    - critic_only_online_steps <= 0 → 永远 False（关闭）
    - 0 < buffer_size < critic_only_online_steps → True（Phase 2）
    - buffer_size >= critic_only_online_steps → False（Phase 3 解冻）
    """

    def _cfg(self, critic_only_online_steps: int):
        cfg = MagicMock()
        cfg.policy = MagicMock()
        cfg.policy.critic_only_online_steps = critic_only_online_steps
        return cfg

    def test_disabled_when_zero(self):
        """0（默认）→ 任何 buffer 大小都返回 False。"""
        cfg = self._cfg(0)
        assert _should_freeze_actor(cfg, 0) is False
        assert _should_freeze_actor(cfg, 100) is False
        assert _should_freeze_actor(cfg, 99999) is False

    def test_frozen_when_buffer_below_threshold(self):
        """Phase 2：真机 transition 数不够阈值 → 冻结。"""
        cfg = self._cfg(500)
        assert _should_freeze_actor(cfg, 0) is True
        assert _should_freeze_actor(cfg, 1) is True
        assert _should_freeze_actor(cfg, 499) is True

    def test_unfrozen_at_exact_threshold(self):
        """边界：buffer_size == 阈值时就解冻（和 online_step_before_learning 语义一致）。"""
        cfg = self._cfg(500)
        assert _should_freeze_actor(cfg, 500) is False

    def test_unfrozen_above_threshold(self):
        """Phase 3：buffer 充足 → 解冻，走正常 SAC。"""
        cfg = self._cfg(500)
        assert _should_freeze_actor(cfg, 501) is False
        assert _should_freeze_actor(cfg, 10000) is False

    def test_negative_treated_as_disabled(self):
        """防御性：负数和 0 行为一致。"""
        cfg = self._cfg(-1)
        assert _should_freeze_actor(cfg, 0) is False
        assert _should_freeze_actor(cfg, 100) is False


class TestShouldRunActorOptimization:
    """主循环 actor/temperature 守卫：把 Phase 2 冻结和 policy_update_freq 周期
    合并成单一 helper，测试直接锁死两者的组合语义，防止主循环重构时漏掉。
    """

    def _cfg(self, critic_only_online_steps: int):
        cfg = MagicMock()
        cfg.policy = MagicMock()
        cfg.policy.critic_only_online_steps = critic_only_online_steps
        return cfg

    def test_frozen_skips_even_on_freq_boundary(self):
        """Phase 2 期间：哪怕 optimization_step % freq == 0，也不跑 actor。"""
        cfg = self._cfg(500)
        # buffer < 阈值 → freeze_actor=True
        assert _should_run_actor_optimization(cfg, replay_buffer_size=100,
                                              optimization_step=0,
                                              policy_update_freq=2) is False
        assert _should_run_actor_optimization(cfg, replay_buffer_size=499,
                                              optimization_step=10,
                                              policy_update_freq=2) is False

    def test_unfrozen_respects_freq(self):
        """Phase 3：解冻后依然要遵守 policy_update_freq 周期。"""
        cfg = self._cfg(500)
        # buffer >= 阈值 → freeze_actor=False，但 step % freq != 0 → 不跑
        assert _should_run_actor_optimization(cfg, replay_buffer_size=500,
                                              optimization_step=1,
                                              policy_update_freq=2) is False
        # 到周期 → 跑
        assert _should_run_actor_optimization(cfg, replay_buffer_size=500,
                                              optimization_step=2,
                                              policy_update_freq=2) is True

    def test_disabled_phase2_only_freq_matters(self):
        """critic_only_online_steps=0（关闭 Phase 2）→ 只看 freq。"""
        cfg = self._cfg(0)
        assert _should_run_actor_optimization(cfg, replay_buffer_size=0,
                                              optimization_step=0,
                                              policy_update_freq=2) is True
        assert _should_run_actor_optimization(cfg, replay_buffer_size=0,
                                              optimization_step=1,
                                              policy_update_freq=2) is False

    def test_freq_one_always_runs_when_unfrozen(self):
        """policy_update_freq=1 + 解冻 → 每步都跑。"""
        cfg = self._cfg(500)
        for step in range(5):
            assert _should_run_actor_optimization(cfg, replay_buffer_size=1000,
                                                  optimization_step=step,
                                                  policy_update_freq=1) is True


class TestMainLoopActorGuardStaticCheck:
    """静态检查：主循环必须通过 _should_run_actor_optimization 守卫 actor 优化。

    用 AST 而不是字符串 grep —— 字符串 grep 会被 helper 自身定义和 docstring
    引用假通过：哪怕主循环裸跑 actor optimizer，文件里仍有 `def
    _should_run_actor_optimization(` 让 grep 命中。AST 直接定位 train() 函数
    体里的 Call 节点，确保真有调用。
    """

    @staticmethod
    def _find_function(tree: "ast.Module", name: str) -> "ast.FunctionDef | None":
        import ast as _ast
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == name:
                return node
        return None

    @staticmethod
    def _has_call_to(node: "ast.AST", func_name: str) -> bool:
        import ast as _ast
        for sub in _ast.walk(node):
            if isinstance(sub, _ast.Call) and isinstance(sub.func, _ast.Name) and sub.func.id == func_name:
                return True
        return False

    def test_main_loop_uses_helper(self):
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "frrl/rl/core/learner.py"
        tree = ast.parse(src.read_text())

        # 主循环在 add_actor_information_and_train 里（gRPC 端口接 transition 后
        # 跑训练 step 的循环）。这里 AST 定位它的体，确认 helper 真被调用。
        loop_fn = self._find_function(tree, "add_actor_information_and_train")
        assert loop_fn is not None, (
            "learner.py 必须有 add_actor_information_and_train(...)（主循环入口）"
        )
        assert self._has_call_to(loop_fn, "_should_run_actor_optimization"), (
            "add_actor_information_and_train() 函数体里必须出现 "
            "_should_run_actor_optimization(...) 调用 —— Phase 2 actor 冻结守卫"
            "不能被绕过。如果这条断言失败，说明主循环重构误删了 helper 调用。"
        )
