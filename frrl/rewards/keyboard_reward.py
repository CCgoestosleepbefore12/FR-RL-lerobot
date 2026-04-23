"""Keyboard-driven reward signal for HIL-SERL-style real-robot training.

Human watches the rollout and presses one of four keys to decide the outcome:

    S          start an episode (sent from the idle state before rollout begins)
    Enter      success → reward=1, terminal
    Space      failure → reward=0, terminal
    Backspace  discard → reward=0, terminal, episode NOT written to replay buffer

Esc is intentionally *not* bound here — it remains owned by the upstream
`frrl.utils.control_utils.init_keyboard_listener()` for global stop-recording.

Design notes:

* The listener runs in a daemon pynput thread. pynput is only imported when the
  pynput backend is enabled, so tests can run without an X server.
* Internal methods (`_on_start_key`, `_on_success_key`, etc.) encapsulate the
  state machine and are called from both the pynput callback and unit tests.
* Outcomes are consumed by a single `poll()` call; after poll returns a decided
  outcome the state machine resets to IDLE, so the next episode again waits for
  S. If the episode ends for a non-keyboard reason (timeout, env.terminate),
  call `mark_episode_ended()` before the next reset() to clear state.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional


_STATE_IDLE = "idle"
_STATE_RUNNING = "running"


class KeyboardRewardListener:
    """Thread-safe human-in-the-loop reward listener.

    Args:
        pynput_backend: when True (default) attaches a pynput keyboard listener.
            Set False in unit tests to drive the state machine via the _on_*_key
            helpers directly.
        poll_interval: seconds to sleep between state checks in blocking waits.
    """

    def __init__(
        self,
        pynput_backend: bool = True,
        poll_interval: float = 0.01,
        required: bool = False,
    ):
        """
        Args:
            pynput_backend: when True (default) attaches a pynput keyboard listener.
                Set False in unit tests to drive the state machine via the _on_*_key
                helpers directly.
            poll_interval: seconds to sleep between state checks in blocking waits.
            required: when True and pynput is unavailable, raise ImportError instead
                of degrading silently. Callers that depend on the keyboard (e.g.
                FrankaRealEnv with ``reward_backend="keyboard"``) should pass True.
        """
        self._state = _STATE_IDLE
        self._outcome: Optional[str] = None
        self._lock = threading.Lock()
        self._poll_interval = poll_interval
        self._listener = None
        self._required = required

        if pynput_backend:
            self._start_pynput_listener()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def wait_for_start(
        self,
        timeout: Optional[float] = None,
        shutdown_event: Optional[threading.Event] = None,
    ) -> bool:
        """Block until user presses S (moves IDLE → RUNNING).

        Returns True once RUNNING, False if `timeout` expired or `shutdown_event`
        was set. Accepting a shutdown_event lets actor/training loops abort the
        wait cleanly when a Ctrl+C / grpc-disconnect signal arrives instead of
        hanging until the operator physically presses S.
        """
        start = time.time()
        while True:
            with self._lock:
                if self._state == _STATE_RUNNING:
                    return True
            if shutdown_event is not None and shutdown_event.is_set():
                return False
            if timeout is not None and time.time() - start > timeout:
                return False
            time.sleep(self._poll_interval)

    def poll(self) -> Optional[Dict[str, Any]]:
        """Non-blocking: if an outcome is decided, consume it and return its dict.

        Returns None while still RUNNING (no outcome yet) or if already IDLE.
        Consuming an outcome resets state to IDLE.
        """
        with self._lock:
            if self._outcome is None:
                return None
            outcome = self._outcome
            self._outcome = None
            self._state = _STATE_IDLE
        return _outcome_to_dict(outcome)

    def mark_episode_ended(self) -> None:
        """Force state → IDLE, clear any pending outcome.

        Call this when an episode ends for a non-keyboard reason (step-limit
        truncation, env.terminate) so the next reset() sees a clean state.
        """
        with self._lock:
            self._state = _STATE_IDLE
            self._outcome = None

    def stop(self) -> None:
        """Stop the pynput listener thread, if any."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as e:
                logging.warning(f"KeyboardRewardListener stop failed: {e}")
            self._listener = None

    # ------------------------------------------------------------------
    # State machine hooks — called from both pynput callback and tests
    # ------------------------------------------------------------------

    def _on_start_key(self) -> None:
        with self._lock:
            if self._state == _STATE_IDLE:
                self._state = _STATE_RUNNING
                self._outcome = None

    def _on_success_key(self) -> None:
        self._set_outcome_if_running("success")

    def _on_fail_key(self) -> None:
        self._set_outcome_if_running("fail")

    def _on_discard_key(self) -> None:
        self._set_outcome_if_running("discard")

    def _set_outcome_if_running(self, outcome: str) -> None:
        with self._lock:
            if self._state == _STATE_RUNNING and self._outcome is None:
                self._outcome = outcome

    # ------------------------------------------------------------------
    # pynput plumbing
    # ------------------------------------------------------------------

    def _start_pynput_listener(self) -> None:
        try:
            from pynput import keyboard as _kb
        except ImportError as e:
            msg = (
                "pynput unavailable — KeyboardRewardListener disabled "
                "(rewards would stay at 0 forever). Install pynput or switch backend."
            )
            if self._required:
                raise ImportError(msg) from e
            logging.warning(msg)
            return

        def on_press(key):
            try:
                if key == _kb.Key.enter:
                    self._on_success_key()
                elif key == _kb.Key.space:
                    self._on_fail_key()
                elif key == _kb.Key.backspace:
                    self._on_discard_key()
                elif hasattr(key, "char") and key.char is not None and key.char.lower() == "s":
                    self._on_start_key()
            except Exception as e:
                logging.warning(f"KeyboardRewardListener on_press error: {e}")

        self._listener = _kb.Listener(on_press=on_press, daemon=True)
        self._listener.start()
        logging.info(
            "KeyboardRewardListener started: S=start, Enter=success, "
            "Space=fail, Backspace=discard"
        )


def _outcome_to_dict(outcome: str) -> Dict[str, Any]:
    """Canonical mapping from outcome name to env.step() reward semantics."""
    table = {
        "success": {"reward": 1, "terminal": True, "discard": False},
        "fail":    {"reward": 0, "terminal": True, "discard": False},
        "discard": {"reward": 0, "terminal": True, "discard": True},
    }
    if outcome not in table:
        raise ValueError(f"unknown outcome: {outcome!r}")
    return table[outcome]
