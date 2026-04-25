"""Tests for pure helpers in scripts/real/collect_demo_task_policy.py.

Heavy I/O (SpaceMouse HID, Franka HTTP, pickle writes) is not exercised —
only the stateless conversion functions.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ isn't a Python package; add it to path to import the module.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "real"
sys.path.insert(0, str(SCRIPTS))

from collect_demo_task_policy import (  # noqa: E402
    build_action_from_spacemouse,
    build_transition_dict,
    classify_episode_end,
    make_config,
    output_filename,
)


class TestBuildActionFromSpacemouse:
    def test_zero_sm_state_produces_zero_action(self):
        action = build_action_from_spacemouse(np.zeros(6), [0, 0, 0, 0])
        assert action.shape == (7,)
        assert action.dtype == np.float32
        assert np.all(action == 0.0)

    def test_xyz_rpy_propagate(self):
        sm = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)
        action = build_action_from_spacemouse(sm, [0, 0, 0, 0])
        np.testing.assert_allclose(action[:6], sm)
        assert action[6] == 0.0

    def test_button_0_closes_gripper(self):
        action = build_action_from_spacemouse(np.zeros(6), [1, 0, 0, 0])
        assert action[6] == -1.0

    def test_button_1_opens_gripper(self):
        action = build_action_from_spacemouse(np.zeros(6), [0, 1, 0, 0])
        assert action[6] == +1.0

    def test_both_buttons_close_wins(self):
        """button[0] (close) is checked first per FrankaRealEnv convention."""
        action = build_action_from_spacemouse(np.zeros(6), [1, 1, 0, 0])
        assert action[6] == -1.0

    def test_scaling_applied(self):
        sm = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        action = build_action_from_spacemouse(sm, [0, 0, 0, 0], scale_xyz=0.5, scale_rpy=0.25)
        np.testing.assert_allclose(action[:3], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(action[3:6], [0.25, 0.25, 0.25])


class TestMakeConfig:
    def test_bias_disabled_has_no_injector(self):
        cfg = make_config(use_bias=False)
        assert cfg.encoder_bias_config is None
        assert cfg.reward_backend == "keyboard"

    def test_bias_enabled_targets_j1(self):
        cfg = make_config(use_bias=True)
        assert cfg.encoder_bias_config is not None
        assert cfg.encoder_bias_config.enable is True
        assert cfg.encoder_bias_config.target_joints == [0]
        assert cfg.encoder_bias_config.bias_mode == "random_uniform"

    def test_bias_range_is_symmetric_02(self):
        cfg = make_config(use_bias=True)
        assert cfg.encoder_bias_config.bias_range == (-0.2, 0.2)

    def test_error_probability_is_1(self):
        """Collection wants bias every episode, not sampled per-episode probability."""
        cfg = make_config(use_bias=True)
        assert cfg.encoder_bias_config.error_probability == 1.0


class TestBuildTransitionDict:
    def test_keys_match_hil_serl_schema(self):
        t = build_transition_dict(
            obs={"agent_pos": np.zeros(29)},
            action=np.zeros(7),
            next_obs={"agent_pos": np.ones(29)},
            reward=0.0,
            terminated=False,
            info={"succeed": False},
        )
        assert set(t.keys()) == {
            "observations", "actions", "next_observations",
            "rewards", "masks", "dones", "infos",
        }

    def test_masks_inverts_terminated(self):
        t = build_transition_dict({}, np.zeros(7), {}, 0.0, terminated=False, info={})
        assert t["masks"] == 1.0
        t = build_transition_dict({}, np.zeros(7), {}, 1.0, terminated=True, info={})
        assert t["masks"] == 0.0

    def test_dones_is_bool(self):
        t = build_transition_dict({}, np.zeros(7), {}, 0.0, terminated=1, info={})
        assert isinstance(t["dones"], bool)
        assert t["dones"] is True

    def test_reward_is_float(self):
        t = build_transition_dict({}, np.zeros(7), {}, np.int64(1), terminated=False, info={})
        assert isinstance(t["rewards"], float)
        assert t["rewards"] == 1.0


class TestClassifyEpisodeEnd:
    def test_discard_takes_priority_over_succeed(self):
        """Operator pressed Backspace — drop regardless of other flags."""
        assert classify_episode_end({"discard": True, "succeed": True}) == "discard"

    def test_succeed_without_discard(self):
        assert classify_episode_end({"succeed": True, "discard": False}) == "success"

    def test_neither_means_fail(self):
        assert classify_episode_end({"succeed": False, "discard": False}) == "fail"

    def test_missing_keys_means_fail(self):
        assert classify_episode_end({}) == "fail"


class TestOutputFilename:
    def test_filename_includes_count_and_suffix(self):
        name = output_filename("data/task_policy_demos", 42, aborted=False)
        assert "task_policy_demos_42_complete_" in name
        assert name.endswith(".pkl")

    def test_aborted_flag_changes_suffix(self):
        name = output_filename("data/task_policy_demos", 7, aborted=True)
        assert "task_policy_demos_7_aborted_" in name

    def test_respects_output_dir(self):
        name = output_filename("/tmp/foo", 1, aborted=False)
        assert name.startswith("/tmp/foo/")
