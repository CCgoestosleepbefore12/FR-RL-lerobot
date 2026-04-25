"""Tests for the offline-buffer pickle path — config + learner dispatch.

SACConfig-level contract：
  * offline_only_mode / offline_pretrain_steps / demo_pickle_paths 默认值合理
  * 每个 cfg 实例有自己的 demo_pickle_paths list（不共享可变状态）

Learner-level dispatch（S6 方案 A：混合模式也能吃 pickle demo）：
  * `initialize_offline_replay_buffer` 在 `demo_pickle_paths` 非空时，
    不论 `offline_only_mode` True/False，都走 `from_pickle_transitions`
    分支（不再回落到 make_dataset → LeRobotDataset）。
"""
from unittest.mock import MagicMock, patch

from frrl.policies.sac.configuration_sac import SACConfig


class TestSACConfigFields:
    def test_offline_only_mode_default_false(self):
        cfg = SACConfig()
        assert cfg.offline_only_mode is False

    def test_offline_pretrain_steps_default_5000(self):
        cfg = SACConfig()
        assert cfg.offline_pretrain_steps == 5000

    def test_demo_pickle_paths_default_empty_list(self):
        cfg = SACConfig()
        assert cfg.demo_pickle_paths == []

    def test_demo_pickle_paths_is_assignable(self):
        cfg = SACConfig()
        cfg.demo_pickle_paths = ["data/a.pkl", "data/b.pkl"]
        assert len(cfg.demo_pickle_paths) == 2

    def test_each_config_instance_gets_fresh_list(self):
        """Dataclass field(default_factory=list) must not share mutable state."""
        a = SACConfig()
        b = SACConfig()
        a.demo_pickle_paths.append("x.pkl")
        assert b.demo_pickle_paths == []


class TestInitializeOfflineReplayBufferDispatch:
    """方案 A 的核心契约：demo_pickle_paths 非空时总走 pickle 分支。"""

    def _make_cfg(self, *, offline_only_mode: bool, demo_paths: list[str]):
        """构造 initialize_offline_replay_buffer 只需要的最小 cfg。"""
        policy = MagicMock()
        policy.offline_only_mode = offline_only_mode
        policy.demo_pickle_paths = demo_paths
        policy.offline_buffer_capacity = 100
        policy.input_features = None
        policy.demo_key_map = None
        policy.demo_transpose_hwc_to_chw = None
        policy.demo_resize_images = None
        policy.demo_normalize_to_unit = None

        cfg = MagicMock()
        cfg.policy = policy
        cfg.dataset = None
        cfg.resume = False
        return cfg

    def test_mixed_mode_with_demos_hits_pickle_branch(self):
        """offline_only_mode=False + demo_pickle_paths 非空 → 走 pickle loader。"""
        from frrl.rl.core.learner import initialize_offline_replay_buffer

        cfg = self._make_cfg(offline_only_mode=False, demo_paths=["fake.pkl"])
        with patch(
            "frrl.rl.core.learner.ReplayBuffer.from_pickle_transitions"
        ) as mock_loader, patch("frrl.rl.core.learner.make_dataset") as mock_make_dataset:
            mock_loader.return_value = MagicMock()
            initialize_offline_replay_buffer(cfg, device="cpu", storage_device="cpu")
            # 验证 dispatch 把 demo_paths 透传给 loader（防 args 错位 / 漏传）
            mock_loader.assert_called_once()
            args, kwargs = mock_loader.call_args
            paths_arg = args[0] if args else kwargs.get("pickle_paths") or kwargs.get("paths")
            assert paths_arg == ["fake.pkl"]
            mock_make_dataset.assert_not_called()

    def test_pretrain_only_mode_with_demos_hits_pickle_branch(self):
        """offline_only_mode=True + demo_pickle_paths 非空 → 仍走 pickle loader。"""
        from frrl.rl.core.learner import initialize_offline_replay_buffer

        cfg = self._make_cfg(offline_only_mode=True, demo_paths=["fake.pkl"])
        with patch(
            "frrl.rl.core.learner.ReplayBuffer.from_pickle_transitions"
        ) as mock_loader, patch("frrl.rl.core.learner.make_dataset") as mock_make_dataset:
            mock_loader.return_value = MagicMock()
            initialize_offline_replay_buffer(cfg, device="cpu", storage_device="cpu")
            mock_loader.assert_called_once()
            args, kwargs = mock_loader.call_args
            paths_arg = args[0] if args else kwargs.get("pickle_paths") or kwargs.get("paths")
            assert paths_arg == ["fake.pkl"]
            mock_make_dataset.assert_not_called()

    def test_empty_demos_falls_back_to_dataset_path(self):
        """demo_pickle_paths 空时走 LeRobotDataset 路径（cfg.dataset 必须提供）。"""
        from frrl.rl.core.learner import initialize_offline_replay_buffer

        cfg = self._make_cfg(offline_only_mode=False, demo_paths=[])
        cfg.dataset = MagicMock(repo_id="fake/ds")
        cfg.policy.input_features = {"observation.state": MagicMock()}
        with patch(
            "frrl.rl.core.learner.ReplayBuffer.from_pickle_transitions"
        ) as mock_pickle_loader, patch(
            "frrl.rl.core.learner.make_dataset"
        ) as mock_make_dataset, patch(
            "frrl.rl.core.learner.ReplayBuffer.from_lerobot_dataset"
        ) as mock_ds_loader:
            mock_ds_loader.return_value = MagicMock()
            initialize_offline_replay_buffer(cfg, device="cpu", storage_device="cpu")
            mock_pickle_loader.assert_not_called()
            mock_make_dataset.assert_called_once()
            mock_ds_loader.assert_called_once()


class TestMixedModePickleRoundTrip:
    """Resume 场景下 offline buffer 的一致性：pickle 路径本身是幂等重载（每次
    从 pickle 文件重新读），所以只要同一批 pickle 加载结果一致，就等价于
    resume 后 offline buffer 与初次 init 一致。
    """

    def _write_pickle(self, path, transitions):
        import pickle as pkl
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pkl.dump(transitions, f)
        return path

    def _fake_transition(self):
        import numpy as np
        obs = {
            "agent_pos": np.zeros(29, dtype=np.float32),
            "pixels": {
                "front": np.zeros((128, 128, 3), dtype=np.uint8),
                "wrist": np.zeros((128, 128, 3), dtype=np.uint8),
            },
        }
        return {
            "observations": obs,
            "actions": np.zeros(7, dtype=np.float32),
            "next_observations": obs,
            "rewards": 0.0,
            "masks": 1.0,
            "dones": False,
            "infos": {},
        }

    def _assert_buffers_content_equal(self, buf1, buf2, n: int):
        """逐 transition 对比 state dict + actions + rewards 张量内容；长度等于
        n 而且每条记录字段+值都对齐才算 round-trip 幂等。
        """
        import torch
        assert len(buf1) == len(buf2) == n
        # 状态 key 必须一致（防 round-trip 漏字段）
        assert set(buf1.states.keys()) == set(buf2.states.keys())
        for i in range(n):
            for key in buf1.states:
                t1 = buf1.states[key][i]
                t2 = buf2.states[key][i]
                assert torch.equal(t1, t2), f"state key {key!r} 第 {i} 条不一致"
            assert torch.equal(buf1.actions[i], buf2.actions[i])
            assert torch.equal(buf1.rewards[i], buf2.rewards[i])

    def test_reload_yields_same_content(self, tmp_path):
        """混合模式下连续两次 load 同一 pickle → buffer 大小、字段、张量内容全一致。

        长度一致是必要不充分条件——如果 round-trip 静默丢字段（比如 obs.pixels.front
        被漏 normalize/transpose），长度仍然 6 但 buffer 实际不可用。所以要测张量内容。
        """
        from frrl.rl.core.buffer import ReplayBuffer

        path = self._write_pickle(
            tmp_path / "demo.pkl", [self._fake_transition() for _ in range(6)]
        )

        buf1 = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        buf2 = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        self._assert_buffers_content_equal(buf1, buf2, n=6)

    def test_mixed_mode_initialize_is_idempotent(self, tmp_path):
        """initialize_offline_replay_buffer 连续两次调用（模拟 first-init 与 resume-reload）
        产出的 buffer 长度一致。"""
        from frrl.rl.core.learner import initialize_offline_replay_buffer

        path = self._write_pickle(
            tmp_path / "demo.pkl", [self._fake_transition() for _ in range(4)]
        )

        policy = MagicMock()
        policy.offline_only_mode = False  # 混合模式
        policy.demo_pickle_paths = [str(path)]
        policy.offline_buffer_capacity = 100
        policy.input_features = None
        policy.demo_key_map = None
        policy.demo_transpose_hwc_to_chw = None
        policy.demo_resize_images = None
        policy.demo_normalize_to_unit = None

        cfg = MagicMock()
        cfg.policy = policy
        cfg.dataset = None
        cfg.resume = False

        buf_a = initialize_offline_replay_buffer(cfg, device="cpu", storage_device="cpu")
        buf_b = initialize_offline_replay_buffer(cfg, device="cpu", storage_device="cpu")
        self._assert_buffers_content_equal(buf_a, buf_b, n=4)
