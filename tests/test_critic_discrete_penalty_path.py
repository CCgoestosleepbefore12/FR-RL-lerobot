"""P0-2 后续：`compute_loss_critic` 的 `complementary_info["discrete_penalty"]` 路径。

当 `num_discrete_actions is None`（纯连续策略），gripper 误触惩罚没有 discrete
critic 可走，必须由 continuous critic 的 td_target 把负反馈传给 actor。本测试
锁这个加法行为：
  - 没传 complementary_info / penalty=None → td_target = rewards + γ·min_q
  - 传了 penalty → td_target = (rewards + penalty) + γ·min_q
  - num_discrete_actions != None → penalty 不进 continuous critic（discrete critic 处理）

直接 instantiate SACPolicy 太重（需要真 ResNet10 + buffer + dataset_stats）。这里
用 unittest.mock 替换 actor/critic_forward，仅验证 td_target 的算术行为。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from frrl.policies.sac.modeling_sac import SACPolicy


def _build_mock_policy(num_discrete_actions=None, q_value: float = 0.5):
    """构造一个最小可调用 compute_loss_critic 的 mock：mock actor + critic_forward。

    critic_forward 返回 shape [num_critics=2, B] 的常量 q_value。actor 返回
    next_action_preds + next_log_probs（不影响 td_target，因为 use_backup_entropy=False）。
    """
    mock = MagicMock(spec=SACPolicy)
    mock.config = SimpleNamespace(
        num_discrete_actions=num_discrete_actions,
        num_subsample_critics=None,
        use_backup_entropy=False,  # 关掉 entropy bonus，让 td_target 只依赖 reward + γ·min_q
        discount=0.99,
        num_critics=2,
    )

    # actor 返回 (next_action_preds, next_log_probs, _)；shape 不重要，
    # use_backup_entropy=False 时 next_log_probs 不会被读
    def fake_actor(_obs, _feat):
        return torch.zeros(4, 7), torch.zeros(4), None
    mock.actor = fake_actor

    # critic_forward 返回 [num_critics, B] 常量。两次调用都返回同样的常量，
    # 让 q_targets.min(dim=0) = q_preds = q_value，td_target 容易手算
    def fake_critic_forward(observations, actions, use_target, observation_features=None):
        B = observations.shape[0]
        return torch.full((2, B), q_value)
    mock.critic_forward = fake_critic_forward

    # 真正调用 SACPolicy.compute_loss_critic（unbound method）
    mock.compute_loss_critic = SACPolicy.compute_loss_critic.__get__(mock)
    return mock


class TestDiscretePenaltyMergeIntoContinuousReward:
    """连续策略路径（num_discrete_actions=None），penalty 必须进 td_target reward 项。"""

    def _run(self, complementary_info, rewards, q_value=0.5, num_discrete=None):
        policy = _build_mock_policy(num_discrete_actions=num_discrete, q_value=q_value)
        B = rewards.shape[0]
        observations = torch.zeros(B, 3)
        actions = torch.zeros(B, 7)
        next_observations = torch.zeros(B, 3)
        done = torch.zeros(B)
        return policy.compute_loss_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            complementary_info=complementary_info,
        )

    def test_no_complementary_info_baseline(self):
        """complementary_info=None：纯 continuous 路径，rewards 不加 penalty。"""
        rewards = torch.full((4,), 1.0)
        loss = self._run(None, rewards)
        # td_target = rewards + γ·min_q = 1.0 + 0.99*0.5 = 1.495；q_pred=0.5
        # mse = (0.5-1.495)^2 = 0.99002...，2 个 critic sum = 1.9800...
        expected = 2 * (0.5 - (1.0 + 0.99 * 0.5)) ** 2
        assert abs(loss.item() - expected) < 1e-4

    def test_penalty_present_shifts_td_target(self):
        """penalty=-0.05：td_target 整体下移，loss 应增大对应量。"""
        rewards = torch.full((4,), 1.0)
        penalty = torch.full((4,), -0.05)
        loss_with = self._run({"discrete_penalty": penalty}, rewards)
        # td_target = (1.0 - 0.05) + 0.99*0.5 = 1.445
        expected_with = 2 * (0.5 - (0.95 + 0.99 * 0.5)) ** 2
        assert abs(loss_with.item() - expected_with) < 1e-4

        # 与无 penalty 版本严格不等
        loss_without = self._run(None, rewards)
        assert loss_with.item() != loss_without.item()

    def test_penalty_none_value_is_noop(self):
        """complementary_info 存在但 discrete_penalty=None：等价 baseline，不报错。"""
        rewards = torch.full((4,), 1.0)
        loss = self._run({"discrete_penalty": None}, rewards)
        expected = 2 * (0.5 - (1.0 + 0.99 * 0.5)) ** 2
        assert abs(loss.item() - expected) < 1e-4

    def test_discrete_critic_path_skips_continuous_penalty(self):
        """num_discrete_actions=2：penalty 由 discrete critic 消费，continuous td_target 不动。"""
        rewards = torch.full((4,), 1.0)
        penalty = torch.full((4,), -0.05)
        loss_with = self._run({"discrete_penalty": penalty}, rewards, num_discrete=2)
        # 期望：penalty **没**进 td_target，与无 penalty baseline 一致
        expected = 2 * (0.5 - (1.0 + 0.99 * 0.5)) ** 2
        assert abs(loss_with.item() - expected) < 1e-4
