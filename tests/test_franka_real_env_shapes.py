"""Tests for FrankaRealEnv observation/action space shapes (no HTTP, no camera).

Verifies the 29D observation layout decided for task policy training:
  [0:7]   q_biased        — /getstate (may contain J1 bias)
  [7:14]  dq              — /getstate (bias is constant offset → Δq/dt unaffected)
  [14]    gripper         — [0, 1]
  [15:18] tcp_pos_biased  — /getstate
  [18:22] tcp_quat_biased — /getstate, scipy [xyzw]
  [22:25] tcp_pos_true    — /getstate_true (privileged "cheat" channel)
  [25:29] tcp_quat_true   — /getstate_true, scipy [xyzw]
"""
import numpy as np
import pytest

from frrl.envs.franka_real_config import FrankaRealConfig
from frrl.envs.franka_real_env import FrankaRealEnv


@pytest.fixture
def env():
    """Build env with default config. No HTTP/camera calls in __init__."""
    return FrankaRealEnv(FrankaRealConfig())


class TestObservationSpace:
    def test_agent_pos_is_29d(self, env):
        assert env.observation_space["agent_pos"].shape == (29,)

    def test_agent_pos_dtype_is_float32(self, env):
        assert env.observation_space["agent_pos"].dtype == np.float32

    def test_environment_state_is_6d(self, env):
        assert env.observation_space["environment_state"].shape == (6,)

    def test_pixels_has_front_and_wrist_keys(self, env):
        pixels = env.observation_space["pixels"].spaces
        assert set(pixels.keys()) == {"front", "wrist"}

    def test_front_image_is_128x128x3_uint8(self, env):
        # Both cameras unified at 128² — see modeling_sac.py:586 rationale.
        front = env.observation_space["pixels"].spaces["front"]
        assert front.shape == (128, 128, 3)
        assert front.dtype == np.uint8

    def test_wrist_image_is_128x128x3_uint8(self, env):
        wrist = env.observation_space["pixels"].spaces["wrist"]
        assert wrist.shape == (128, 128, 3)
        assert wrist.dtype == np.uint8


class TestActionSpace:
    def test_action_is_7d_box(self, env):
        assert env.action_space.shape == (7,)

    def test_action_range_is_neg1_to_1(self, env):
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)


class TestGetRobotStateLayout:
    def test_returns_29d(self, env):
        state = env.get_robot_state()
        assert state.shape == (29,)

    def test_dtype_is_float32(self, env):
        assert env.get_robot_state().dtype == np.float32

    def test_layout_slices_after_manual_assignment(self, env):
        """Assign known values to each underlying field and check they land at the right indices."""
        env.q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        env.dq = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
        env.curr_gripper_pos = 0.5
        env.currpos = np.array([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0])        # pos + quat
        env.currpos_true = np.array([11.0, 21.0, 31.0, 0.1, 0.2, 0.3, 0.9])

        s = env.get_robot_state()

        np.testing.assert_allclose(s[0:7], env.q)
        np.testing.assert_allclose(s[7:14], env.dq)
        assert s[14] == pytest.approx(0.5)
        np.testing.assert_allclose(s[15:18], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(s[18:22], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(s[22:25], [11.0, 21.0, 31.0])
        np.testing.assert_allclose(s[25:29], [0.1, 0.2, 0.3, 0.9])


class TestInitialState:
    def test_biased_and_true_state_are_independent_arrays(self, env):
        """Mutating biased state must not leak into true state (and vice versa)."""
        env.currpos[0] = 99.0
        assert env.currpos_true[0] == 0.0
        env.q_true[0] = 88.0
        assert env.q[0] == 0.0

    def test_true_state_fields_exist(self, env):
        """The privileged channel fields must be initialized (not None)."""
        assert env.currpos_true.shape == (7,)
        assert env.q_true.shape == (7,)
