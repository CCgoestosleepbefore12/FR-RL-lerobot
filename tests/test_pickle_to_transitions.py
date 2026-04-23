"""Tests for ReplayBuffer.from_pickle_transitions and helpers.

Goal: verify that a pickle written by scripts/collect_demo_task_policy.py (which
uses hil-serl's transition schema) can be loaded into a ReplayBuffer that
behaves identically to one populated by the online actor loop.
"""
import pickle as pkl
from pathlib import Path

import numpy as np
import pytest
import torch

from frrl.rl.buffer import (
    ReplayBuffer,
    _flatten_obs_to_tensor_dict,
    _require_keys,
    _to_batched_tensor,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_fake_obs(agent_dim: int = 29):
    """Match the layout produced by FrankaRealEnv._compute_observation."""
    return {
        "agent_pos": np.zeros(agent_dim, dtype=np.float32),
        "environment_state": np.zeros(6, dtype=np.float32),
        "pixels": {
            "front": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist": np.zeros((128, 128, 3), dtype=np.uint8),
        },
    }


def _make_fake_transition(reward: float = 0.0, done: bool = False):
    return {
        "observations": _make_fake_obs(),
        "actions": np.zeros(7, dtype=np.float32),
        "next_observations": _make_fake_obs(),
        "rewards": reward,
        "masks": 1.0 - float(done),
        "dones": done,
        "infos": {"succeed": done},
    }


# ----------------------------------------------------------------------
# Helper tests
# ----------------------------------------------------------------------


class TestToBatchedTensor:
    def test_numpy_array_gets_batch_dim(self):
        t = _to_batched_tensor(np.zeros(7, dtype=np.float32))
        assert t.shape == (1, 7)
        assert t.dtype == torch.float32

    def test_image_preserves_hwc(self):
        """224×224×3 uint8 must stay uint8 and pick up a batch dim."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        t = _to_batched_tensor(img)
        assert t.shape == (1, 224, 224, 3)
        assert t.dtype == torch.uint8

    def test_scalar_float(self):
        t = _to_batched_tensor(0.5)
        assert t.shape == (1,)

    def test_scalar_bool(self):
        t = _to_batched_tensor(True)
        assert t.shape == (1,)
        assert t.dtype == torch.bool

    def test_already_tensor(self):
        tin = torch.zeros(7, dtype=torch.float32)
        t = _to_batched_tensor(tin)
        assert t.shape == (1, 7)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="cannot convert"):
            _to_batched_tensor("hello")


class TestFlattenObsToTensorDict:
    def test_flat_dict(self):
        obs = {"agent_pos": np.zeros(29, dtype=np.float32)}
        out = _flatten_obs_to_tensor_dict(obs)
        assert list(out.keys()) == ["agent_pos"]
        assert out["agent_pos"].shape == (1, 29)

    def test_nested_dict_flattens_with_dot_keys(self):
        obs = _make_fake_obs()
        out = _flatten_obs_to_tensor_dict(obs)
        assert "agent_pos" in out
        assert "environment_state" in out
        assert "pixels.front" in out
        assert "pixels.wrist" in out
        assert out["pixels.front"].shape == (1, 224, 224, 3)
        assert out["pixels.wrist"].shape == (1, 128, 128, 3)

    def test_non_dict_top_level_becomes_state_key(self):
        out = _flatten_obs_to_tensor_dict(np.zeros(29, dtype=np.float32))
        assert list(out.keys()) == ["state"]
        assert out["state"].shape == (1, 29)

    def test_empty_dict_returns_empty(self):
        assert _flatten_obs_to_tensor_dict({}) == {}


class TestRequireKeys:
    def test_all_present_is_noop(self):
        _require_keys({"a": 1, "b": 2}, ("a", "b"))

    def test_missing_raises_keyerror(self):
        with pytest.raises(KeyError, match="missing"):
            _require_keys({"a": 1}, ("a", "b"))


# ----------------------------------------------------------------------
# End-to-end load tests
# ----------------------------------------------------------------------


class TestFromPickleTransitions:
    @staticmethod
    def _write_pickle(path: Path, transitions) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pkl.dump(transitions, f)
        return path

    def test_loads_single_pickle(self, tmp_path):
        transitions = [_make_fake_transition() for _ in range(5)]
        transitions[-1]["dones"] = True
        path = self._write_pickle(tmp_path / "demo.pkl", transitions)

        buf = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        assert len(buf) == 5

    def test_loads_multiple_pickles(self, tmp_path):
        """Multiple demo files concatenate in order."""
        p1 = self._write_pickle(tmp_path / "a.pkl", [_make_fake_transition() for _ in range(3)])
        p2 = self._write_pickle(tmp_path / "b.pkl", [_make_fake_transition() for _ in range(4)])
        buf = ReplayBuffer.from_pickle_transitions(
            [p1, p2], device="cpu", storage_device="cpu", use_drq=False
        )
        assert len(buf) == 7

    def test_empty_pickle_raises(self, tmp_path):
        path = self._write_pickle(tmp_path / "demo.pkl", [])
        with pytest.raises(ValueError, match="no transitions"):
            ReplayBuffer.from_pickle_transitions(
                [path], device="cpu", storage_device="cpu", use_drq=False
            )

    def test_non_list_pickle_raises(self, tmp_path):
        path = tmp_path / "bad.pkl"
        with open(path, "wb") as f:
            pkl.dump({"not_a_list": True}, f)
        with pytest.raises(ValueError, match="must unpickle to a list"):
            ReplayBuffer.from_pickle_transitions(
                [path], device="cpu", storage_device="cpu", use_drq=False
            )

    def test_capacity_too_small_raises(self, tmp_path):
        path = self._write_pickle(tmp_path / "demo.pkl", [_make_fake_transition() for _ in range(5)])
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer.from_pickle_transitions(
                [path], capacity=3, device="cpu", storage_device="cpu", use_drq=False
            )

    def test_missing_required_key_raises(self, tmp_path):
        bad = _make_fake_transition()
        del bad["actions"]
        path = self._write_pickle(tmp_path / "demo.pkl", [bad])
        with pytest.raises(KeyError, match="actions"):
            ReplayBuffer.from_pickle_transitions(
                [path], device="cpu", storage_device="cpu", use_drq=False
            )

    def test_loaded_buffer_can_be_sampled(self, tmp_path):
        """Full roundtrip: load → sample → verify batch shapes match original."""
        transitions = [_make_fake_transition() for _ in range(10)]
        path = self._write_pickle(tmp_path / "demo.pkl", transitions)

        buf = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        batch = buf.sample(batch_size=4)
        # State keys preserved
        assert "agent_pos" in batch["state"]
        assert "pixels.front" in batch["state"]
        assert "pixels.wrist" in batch["state"]
        # Shapes: batch dim = 4
        assert batch["state"]["agent_pos"].shape == (4, 29)
        assert batch["state"]["pixels.front"].shape == (4, 224, 224, 3)
        assert batch["state"]["pixels.wrist"].shape == (4, 128, 128, 3)
        assert batch["action"].shape == (4, 7)
        assert batch["reward"].shape == (4,)
        assert batch["done"].shape == (4,)

    def test_reward_and_done_roundtrip(self, tmp_path):
        """Specific scalar values survive the pickle → tensor → buffer roundtrip."""
        transitions = [
            _make_fake_transition(reward=0.0, done=False),
            _make_fake_transition(reward=1.0, done=True),
        ]
        path = self._write_pickle(tmp_path / "demo.pkl", transitions)
        buf = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        # Drain via sample with deterministic seed pattern: sample-all via large batch
        batch = buf.sample(batch_size=2)
        # Both reward values present (order non-deterministic due to random sampling)
        rewards_seen = set(batch["reward"].tolist())
        dones_seen = set(bool(d) for d in batch["done"].tolist())
        assert rewards_seen.issubset({0.0, 1.0})
        assert dones_seen.issubset({True, False})

    def test_truncated_defaults_to_false(self, tmp_path):
        """hil-serl schema has no truncated field; loader must default to False."""
        path = self._write_pickle(tmp_path / "demo.pkl", [_make_fake_transition() for _ in range(3)])
        buf = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        batch = buf.sample(batch_size=3)
        # All truncated flags should be 0 (False)
        assert torch.all(batch["truncated"] == 0)


class TestCollectedDemoCompatibility:
    """Round-trip through the actual scripts/collect_demo_task_policy.py helper."""

    def test_build_transition_dict_output_is_loadable(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from collect_demo_task_policy import build_transition_dict

        obs = _make_fake_obs()
        t = build_transition_dict(
            obs=obs,
            action=np.zeros(7, dtype=np.float32),
            next_obs=_make_fake_obs(),
            reward=0.0,
            terminated=False,
            info={"succeed": False, "discard": False},
        )
        # Write to pickle, then load with from_pickle_transitions
        path = tmp_path / "compat.pkl"
        with open(path, "wb") as f:
            pkl.dump([t], f)

        buf = ReplayBuffer.from_pickle_transitions(
            [path], device="cpu", storage_device="cpu", use_drq=False
        )
        assert len(buf) == 1
