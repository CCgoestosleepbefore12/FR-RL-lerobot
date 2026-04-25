"""P0-2 后续：`freeze_temperature_in_pretrain` 字段 + warmup loop 的 α 冻结契约。

跑端到端的 warmup 需要 offline_replay_buffer + dataset + 真 SACPolicy，太重；
本测试只锁两件事：
  1. `SACConfig.freeze_temperature_in_pretrain` 默认 True、可赋值、实例独立
  2. learner.py 的 warmup 分支里**确实**用这个 flag 把 temperature 优化包起来了
     （静态 AST 检查，防止主循环重构时漏掉守卫）

端到端"flag=True 时 log_alpha 真不动"留到集成测试。
"""

import ast
from pathlib import Path

from frrl.policies.sac.configuration_sac import SACConfig


class TestSACConfigFieldWiring:
    def test_default_is_true(self):
        """默认 True：pretrain 期 α 冻结，避免 demo 高似然把 α 推向坍塌。"""
        cfg = SACConfig()
        assert cfg.freeze_temperature_in_pretrain is True

    def test_is_assignable(self):
        cfg = SACConfig()
        cfg.freeze_temperature_in_pretrain = False
        assert cfg.freeze_temperature_in_pretrain is False

    def test_instance_independent(self):
        a = SACConfig(freeze_temperature_in_pretrain=False)
        b = SACConfig()  # 默认 True
        assert a.freeze_temperature_in_pretrain is False
        assert b.freeze_temperature_in_pretrain is True


class TestWarmupAlphaFreezeGuard:
    """静态 AST 检查：learner.py 的 warmup 分支必须用 freeze_temperature_in_pretrain
    守卫 temperature 优化。grep 不够（docstring/comment 也算命中），AST 定位真 If 节点。
    """

    def _read_learner_ast(self) -> ast.Module:
        src = Path(__file__).resolve().parent.parent / "frrl/rl/core/learner.py"
        return ast.parse(src.read_text())

    def test_flag_referenced_with_negation_around_temperature_step(self):
        """验证存在 `if not <...>.freeze_temperature_in_pretrain: <temperature step>` 结构。"""
        tree = self._read_learner_ast()

        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # test 必须形如 `not X.freeze_temperature_in_pretrain`
            if not isinstance(node.test, ast.UnaryOp) or not isinstance(node.test.op, ast.Not):
                continue
            inner = node.test.operand
            if not isinstance(inner, ast.Attribute) or inner.attr != "freeze_temperature_in_pretrain":
                continue

            # body 里必须出现 temperature 相关 step（compute_loss_temperature 或
            # optimizers["temperature"]） —— 防止 if 分支只剩注释/log
            body_src = ast.unparse(node)
            if "compute_loss_temperature" in body_src and "temperature" in body_src:
                found = True
                break

        assert found, (
            "learner.py warmup 分支必须有 `if not <cfg>.freeze_temperature_in_pretrain:` 守卫，"
            "且分支体内调用 compute_loss_temperature / optimizers['temperature']。"
            "如果断言失败，说明 P0-2 的 α 冻结守卫被重构误删，pretrain 期 α 会再次坍塌。"
        )
