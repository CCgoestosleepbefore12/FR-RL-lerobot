"""Tests for frrl.rl.core.actor.should_discard_episode.

This validates the discard hook used by the training actor loop. The hook decides
whether an episode's transitions should be dropped from BOTH replay buffers
(online + prior) — used when the human operator hits Backspace on a bad scene.

Integration with the full actor loop isn't unit-testable (distributed gRPC,
torch multiprocessing); we test the pure decision function plus the contract
that FrankaRealEnv feeds into it via info["discard"].
"""
import pytest

from frrl.rl.infra.actor_utils import DISCARD_INFO_KEY, should_discard_episode


class TestShouldDiscardEpisode:
    def test_empty_info_is_not_discarded(self):
        assert should_discard_episode({}) is False

    def test_none_info_is_not_discarded(self):
        assert should_discard_episode(None) is False

    def test_missing_key_is_not_discarded(self):
        assert should_discard_episode({"succeed": True, "hand_collisions": 0}) is False

    def test_discard_true_returns_true(self):
        assert should_discard_episode({"discard": True}) is True

    def test_discard_false_returns_false(self):
        assert should_discard_episode({"discard": False}) is False

    def test_truthy_nonbool_discard(self):
        """Defensive: a truthy value like 1 or 'yes' also counts."""
        assert should_discard_episode({"discard": 1}) is True
        assert should_discard_episode({"discard": "yes"}) is True

    def test_falsy_nonbool_discard(self):
        assert should_discard_episode({"discard": 0}) is False
        assert should_discard_episode({"discard": ""}) is False
        assert should_discard_episode({"discard": None}) is False

    def test_constant_matches_contract_key(self):
        """DISCARD_INFO_KEY must equal the key FrankaRealEnv emits."""
        assert DISCARD_INFO_KEY == "discard"


class TestEnvActorContract:
    """Verifies FrankaRealEnv's step info actually carries the discard signal
    the actor looks for. This is the integration glue: if either side changes
    the key name, this test catches it."""

    def test_keyboard_discard_flows_to_info(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv
        from frrl.rewards.keyboard_reward import KeyboardRewardListener

        cfg = FrankaRealConfig(reward_backend="keyboard")
        env = FrankaRealEnv(cfg)
        env._keyboard_reward = KeyboardRewardListener(pynput_backend=False)
        # Simulate: S pressed → running → Backspace pressed → discard outcome pending
        env._keyboard_reward._on_start_key()
        env._keyboard_reward._on_discard_key()

        rinfo = env.compute_reward(obs={})
        # rinfo is what step() consumes; we mirror what step() puts in info.
        step_info = {
            "succeed": rinfo["reward"] == 1,
            "discard": rinfo["discard"],
        }
        assert should_discard_episode(step_info) is True
        env.close()

    def test_keyboard_success_does_not_discard(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv
        from frrl.rewards.keyboard_reward import KeyboardRewardListener

        cfg = FrankaRealConfig(reward_backend="keyboard")
        env = FrankaRealEnv(cfg)
        env._keyboard_reward = KeyboardRewardListener(pynput_backend=False)
        env._keyboard_reward._on_start_key()
        env._keyboard_reward._on_success_key()

        rinfo = env.compute_reward(obs={})
        step_info = {
            "succeed": rinfo["reward"] == 1,
            "discard": rinfo["discard"],
        }
        assert should_discard_episode(step_info) is False
        env.close()

    def test_pose_backend_never_discards(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv

        cfg = FrankaRealConfig()  # pose backend
        env = FrankaRealEnv(cfg)
        rinfo = env.compute_reward(obs={})
        step_info = {"succeed": False, "discard": rinfo["discard"]}
        assert should_discard_episode(step_info) is False
        env.close()
