"""Tests for FrankaRealEnv observation/action space shapes (no HTTP, no camera).

Verifies the 14D observation layout (对齐 hil-serl ram_insertion proprio 子集):
  [0:7]   tcp_pose_true  — /getstate_true xyz + quat (xyzw)（绕过 J1 bias 的真 TCP）
  [7:13]  tcp_vel        — /getstate 末端 6D twist（bias 是关节角偏移，不影响 dpose/dt）
  [13]    gripper_pose   — /getstate [0, 1]

历史：之前是 29D 含 q_biased / dq / tcp_pos_biased / tcp_quat_biased，2026-04-28
切换到 14D 对齐 hil-serl，去掉 bias-aware proprio 让网络只看 bias-鲁棒的 TCP 信号。
"""
import numpy as np
import pytest

from frrl.envs.real_config import FrankaRealConfig
from frrl.envs.real import FrankaRealEnv


@pytest.fixture
def env():
    """Build env with default config. No HTTP/camera calls in __init__."""
    return FrankaRealEnv(FrankaRealConfig())


class TestObservationSpace:
    def test_agent_pos_is_14d(self, env):
        assert env.observation_space["agent_pos"].shape == (14,)

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
    def test_returns_14d(self, env):
        state = env.get_robot_state()
        assert state.shape == (14,)

    def test_dtype_is_float32(self, env):
        assert env.get_robot_state().dtype == np.float32

    def test_layout_slices_after_manual_assignment(self, env):
        """Assign known values to each underlying field and check they land at the right indices."""
        env.currpos_true = np.array([11.0, 21.0, 31.0, 0.1, 0.2, 0.3, 0.9])  # pos + quat (xyzw)
        env.currvel = np.array([0.5, 0.6, 0.7, 0.05, 0.06, 0.07])             # 6D twist
        env.curr_gripper_pos = 0.5

        s = env.get_robot_state()

        np.testing.assert_allclose(s[0:3], [11.0, 21.0, 31.0])              # tcp_pos_true
        np.testing.assert_allclose(s[3:7], [0.1, 0.2, 0.3, 0.9])            # tcp_quat_true
        np.testing.assert_allclose(s[7:13], [0.5, 0.6, 0.7, 0.05, 0.06, 0.07])  # tcp_vel
        assert s[13] == pytest.approx(0.5)                                  # gripper


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
