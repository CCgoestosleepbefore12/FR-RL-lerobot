"""锁 hil-serl discrete gripper 约定：action[-1] ∈ {-1, 0, +1} 三方共享。

历史 bug：modeling_sac 之前的实现 select_action 输出 argmax idx ∈ {0, 1, 2} 直接当
action[-1]，compute_loss_discrete_critic 用 round(action[-1]) 当 gather idx。这套：
  - 真机 env._send_gripper_command 阈值 ±0.5：{0, 1, 2} → {no-op, open, open}（close 永远不发）
  - demo schema 写 {-1, 0, +1}：round(-1)=-1，gather(Q, dim=1, index=-1) Python 负索引
    取 Q[:, 2]，把 close demo 静默标成 open 训练

修复（对齐 hil-serl serl_launcher sac_hybrid_dual.py:443-454, 237）：
  - select_action: argmax-1 → action[-1] ∈ {-1, 0, +1}
  - compute_loss: round(action[-1])+1 → idx ∈ {0, 1, 2}，clamp 防越界

本测试两个层面：
  1) **算术 invariant**：纯 tensor 算术验证 select_action / compute_loss 的转换互逆
  2) **mock SACPolicy.select_action**：传入受控 discrete_critic Q 值，断言输出 action[-1]
     对应 hil-serl 三态而不是 raw idx
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from frrl.policies.sac.modeling_sac import SACPolicy


class TestDiscreteActionArithmetic:
    """compute_loss_discrete_critic 里的 (round(a)+1).clamp 应正确把 {-1, 0, +1} 映射到 {0, 1, 2}。"""

    def _action_to_idx(self, action_tail: torch.Tensor, n_discrete: int = 3) -> torch.Tensor:
        """复刻 modeling_sac.compute_loss_discrete_critic 的偏移 + clamp 算术。"""
        return (torch.round(action_tail) + 1).long().clamp(0, n_discrete - 1)

    def test_close_demo_maps_to_idx0(self):
        """demo 写 -1（close）→ idx 0。修复前 round(-1)=-1 → gather wrap 到 idx=2（错）。"""
        action_tail = torch.tensor([[-1.0]])
        idx = self._action_to_idx(action_tail)
        assert idx.item() == 0

    def test_noop_demo_maps_to_idx1(self):
        """demo 写 0（no-op）→ idx 1。"""
        action_tail = torch.tensor([[0.0]])
        idx = self._action_to_idx(action_tail)
        assert idx.item() == 1

    def test_open_demo_maps_to_idx2(self):
        """demo 写 +1（open）→ idx 2。"""
        action_tail = torch.tensor([[1.0]])
        idx = self._action_to_idx(action_tail)
        assert idx.item() == 2

    def test_clamp_below_minus_one(self):
        """demo / teleop 偶发噪声 action[-1] = -2 不应炸 gather；clamp 到 idx=0。"""
        action_tail = torch.tensor([[-2.0]])
        idx = self._action_to_idx(action_tail)
        assert idx.item() == 0

    def test_clamp_above_plus_one(self):
        """同理 +2 → clamp 到 idx=2。"""
        action_tail = torch.tensor([[2.5]])
        idx = self._action_to_idx(action_tail)
        assert idx.item() == 2

    def test_round_handles_fractional_demos(self):
        """连续噪声（intervention 或 SpaceMouse）action[-1] = -0.6 → round 到 -1 → idx 0。"""
        action_tail = torch.tensor([[-0.6], [0.4], [0.7]])
        idx = self._action_to_idx(action_tail)
        assert idx.flatten().tolist() == [0, 1, 2]

    def test_idx_dtype_is_long(self):
        """gather 要求 index dtype=long。"""
        action_tail = torch.tensor([[0.0]])
        idx = self._action_to_idx(action_tail)
        assert idx.dtype == torch.long

    def test_round_trip_argmax_to_action_to_idx(self):
        """select_action(argmax) 输出 action[-1]，喂回 compute_loss_critic 应还原同一 idx。

        模拟 policy → buffer → loss 路径：argmax=0/1/2 → action[-1]=-1/0/+1 → idx=0/1/2。
        修复前断裂在 buffer 阶段（policy 写 idx，demo 写 continuous，schema 不一致）。
        """
        for argmax_idx in [0, 1, 2]:
            policy_argmax = torch.tensor([[argmax_idx]])
            # select_action 段：argmax-1
            action_tail = (policy_argmax - 1).float()
            # compute_loss 段：round+1
            recovered_idx = self._action_to_idx(action_tail)
            assert recovered_idx.item() == argmax_idx


class TestSelectActionDiscreteOutput:
    """SACPolicy.select_action 在 num_discrete_actions=3 时 action[-1] ∈ {-1, 0, +1}。"""

    def _build_mock_policy(self, argmax_idx: int):
        """最小 mock：actor 返回 6D 连续，discrete_critic 返回让 argmax=argmax_idx 的 Q 值。"""
        mock = MagicMock(spec=SACPolicy)
        mock.config = SimpleNamespace(num_discrete_actions=3)
        mock.shared_encoder = False  # 跳过 cached image features 路径

        # actor 返回 6D 连续 (B=2, 6)
        def fake_actor(_batch, _feat):
            return torch.zeros(2, 6), None, None
        mock.actor = SimpleNamespace(encoder=SimpleNamespace(has_images=False))
        # 真正调用时 actor 是 callable；MagicMock 已支持
        mock.actor = lambda batch, feat: fake_actor(batch, feat)
        mock.actor.encoder = SimpleNamespace(has_images=False)

        # discrete_critic 返回 (B=2, 3) Q 值，让 argmax 落在 argmax_idx
        q = torch.full((2, 3), -1.0)
        q[:, argmax_idx] = 1.0

        def fake_discrete_critic(_batch, _feat):
            return q
        mock.discrete_critic = fake_discrete_critic

        # bind 真正的 select_action（unbound method）
        mock.select_action = SACPolicy.select_action.__get__(mock)
        return mock

    def test_argmax_close_outputs_minus_one(self):
        policy = self._build_mock_policy(argmax_idx=0)
        action = policy.select_action({})
        assert action.shape == (2, 7)
        assert torch.allclose(action[:, -1], torch.tensor([-1.0, -1.0]))

    def test_argmax_noop_outputs_zero(self):
        policy = self._build_mock_policy(argmax_idx=1)
        action = policy.select_action({})
        assert torch.allclose(action[:, -1], torch.tensor([0.0, 0.0]))

    def test_argmax_open_outputs_plus_one(self):
        policy = self._build_mock_policy(argmax_idx=2)
        action = policy.select_action({})
        assert torch.allclose(action[:, -1], torch.tensor([1.0, 1.0]))

    def test_action_dtype_is_floating_point(self):
        """action[-1] 必须是 float（不能是 long），否则 cat 类型不一致或 env clip 出错。

        用 is_floating_point 而非 == float32：未来 amp / bf16 训练 dtype 不同也兼容。
        """
        policy = self._build_mock_policy(argmax_idx=2)
        action = policy.select_action({})
        assert action.dtype.is_floating_point
