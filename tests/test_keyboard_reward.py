"""Tests for frrl.rewards.keyboard_reward.KeyboardRewardListener state machine.

pynput backend is disabled in all tests — we drive the state machine directly
via the `_on_*_key` hooks so the tests don't need an X server.
"""
import threading
import time

import pytest

from frrl.rewards.keyboard_reward import KeyboardRewardListener


@pytest.fixture
def listener():
    """Listener with pynput backend disabled, so tests are pure state-machine."""
    lst = KeyboardRewardListener(pynput_backend=False)
    yield lst
    lst.stop()


class TestInitialState:
    def test_starts_idle(self, listener):
        assert listener.state == "idle"

    def test_poll_returns_none_when_idle(self, listener):
        assert listener.poll() is None

    def test_no_pynput_listener_when_backend_disabled(self, listener):
        assert listener._listener is None


class TestStartTransition:
    def test_s_moves_idle_to_running(self, listener):
        listener._on_start_key()
        assert listener.state == "running"

    def test_s_while_running_is_noop(self, listener):
        listener._on_start_key()
        listener._on_start_key()
        assert listener.state == "running"
        assert listener._outcome is None

    def test_success_while_idle_is_ignored(self, listener):
        """Reward keys must not fire before S has been pressed."""
        listener._on_success_key()
        assert listener.state == "idle"
        assert listener.poll() is None

    def test_fail_while_idle_is_ignored(self, listener):
        listener._on_fail_key()
        assert listener.state == "idle"
        assert listener.poll() is None

    def test_discard_while_idle_is_ignored(self, listener):
        listener._on_discard_key()
        assert listener.state == "idle"
        assert listener.poll() is None


class TestOutcomes:
    def _setup_running(self, listener):
        listener._on_start_key()

    def test_success_outcome(self, listener):
        self._setup_running(listener)
        listener._on_success_key()
        result = listener.poll()
        assert result == {"reward": 1, "terminal": True, "discard": False}

    def test_fail_outcome(self, listener):
        self._setup_running(listener)
        listener._on_fail_key()
        result = listener.poll()
        assert result == {"reward": 0, "terminal": True, "discard": False}

    def test_discard_outcome(self, listener):
        self._setup_running(listener)
        listener._on_discard_key()
        result = listener.poll()
        assert result == {"reward": 0, "terminal": True, "discard": True}

    def test_first_outcome_wins(self, listener):
        """If operator double-presses, the first key decides."""
        self._setup_running(listener)
        listener._on_success_key()
        listener._on_fail_key()
        result = listener.poll()
        assert result["reward"] == 1
        assert result["discard"] is False

    def test_poll_is_consumable(self, listener):
        """After poll() returns an outcome, state resets and second poll returns None."""
        self._setup_running(listener)
        listener._on_success_key()
        assert listener.poll() is not None
        assert listener.poll() is None
        assert listener.state == "idle"

    def test_poll_while_running_returns_none(self, listener):
        """No outcome yet — poll should be non-blocking and return None."""
        self._setup_running(listener)
        assert listener.poll() is None
        assert listener.state == "running"


class TestEpisodeCycle:
    def test_full_cycle_success_then_fail(self, listener):
        # Episode 1: success
        listener._on_start_key()
        listener._on_success_key()
        assert listener.poll() == {"reward": 1, "terminal": True, "discard": False}
        assert listener.state == "idle"

        # Episode 2: fail
        listener._on_start_key()
        listener._on_fail_key()
        assert listener.poll() == {"reward": 0, "terminal": True, "discard": False}
        assert listener.state == "idle"

    def test_mark_episode_ended_clears_pending_outcome(self, listener):
        """If an episode was decided but not yet polled, external end should drop it."""
        listener._on_start_key()
        listener._on_success_key()
        listener.mark_episode_ended()
        assert listener.state == "idle"
        assert listener.poll() is None

    def test_mark_episode_ended_from_running_state(self, listener):
        """Timeout-style end: running state with no outcome → clean idle."""
        listener._on_start_key()
        listener.mark_episode_ended()
        assert listener.state == "idle"
        assert listener._outcome is None


class TestWaitForStart:
    def test_returns_true_immediately_if_already_running(self, listener):
        listener._on_start_key()
        assert listener.wait_for_start(timeout=0.1) is True

    def test_blocks_and_returns_true_when_s_pressed(self, listener):
        def delayed_start():
            time.sleep(0.05)
            listener._on_start_key()

        t = threading.Thread(target=delayed_start, daemon=True)
        t.start()
        assert listener.wait_for_start(timeout=1.0) is True
        t.join()

    def test_returns_false_on_timeout(self, listener):
        assert listener.wait_for_start(timeout=0.05) is False


class TestThreadSafety:
    def test_concurrent_key_hooks_do_not_crash(self, listener):
        """Hammer all four hooks from multiple threads; end state must be coherent."""
        listener._on_start_key()

        def hammer(fn, n=100):
            for _ in range(n):
                fn()

        threads = [
            threading.Thread(target=hammer, args=(listener._on_success_key,)),
            threading.Thread(target=hammer, args=(listener._on_fail_key,)),
            threading.Thread(target=hammer, args=(listener._on_discard_key,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one outcome should have been recorded (first writer wins).
        result = listener.poll()
        assert result is not None
        assert result["reward"] in (0, 1)
        assert result["terminal"] is True
        # After poll state is idle, no pending outcome.
        assert listener.state == "idle"
        assert listener.poll() is None


class TestEnvIntegration:
    """Minimal smoke test: FrankaRealEnv with reward_backend='keyboard' starts up."""

    def test_env_init_keyboard_backend(self, monkeypatch):
        """FrankaRealEnv with keyboard backend instantiates a listener.

        Swap the listener immediately for a pynput-free one so the test doesn't
        start a real X11 keyboard hook on dev machines.
        """
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv

        cfg = FrankaRealConfig(reward_backend="keyboard")
        env = FrankaRealEnv(cfg)
        assert env._keyboard_reward is not None
        env._keyboard_reward.stop()
        env._keyboard_reward = KeyboardRewardListener(pynput_backend=False)
        env.close()

    def test_env_init_pose_backend_has_no_listener(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv

        cfg = FrankaRealConfig()  # default is "pose"
        env = FrankaRealEnv(cfg)
        assert env._keyboard_reward is None
        env.close()

    def test_compute_reward_keyboard_no_outcome(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv

        cfg = FrankaRealConfig(reward_backend="keyboard")
        env = FrankaRealEnv(cfg)
        # Replace listener with a stub so pynput isn't involved
        env._keyboard_reward = KeyboardRewardListener(pynput_backend=False)
        result = env.compute_reward(obs={})
        assert result == {"reward": 0, "terminal": False, "discard": False}
        env.close()

    def test_compute_reward_keyboard_success(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv

        cfg = FrankaRealConfig(reward_backend="keyboard")
        env = FrankaRealEnv(cfg)
        env._keyboard_reward = KeyboardRewardListener(pynput_backend=False)
        env._keyboard_reward._on_start_key()
        env._keyboard_reward._on_success_key()
        result = env.compute_reward(obs={})
        assert result == {"reward": 1, "terminal": True, "discard": False}
        env.close()

    def test_compute_reward_pose_without_target(self):
        from frrl.envs.real_config import FrankaRealConfig
        from frrl.envs.real import FrankaRealEnv

        cfg = FrankaRealConfig()  # target_pose default None
        env = FrankaRealEnv(cfg)
        result = env.compute_reward(obs={})
        assert result == {"reward": 0, "terminal": False, "discard": False}
        env.close()
