"""Collect SpaceMouse demos for HIL-SERL task policy pretraining.

Adapted from rail-berkeley/hil-serl examples/record_demos.py — same transition-
dict pickle format so demos can later be consumed by RLPD offline buffer / BC
pretraining routines expecting hil-serl schema.

Differences vs hil-serl's script:
  * env      — our FrankaRealEnv with keyboard-backed reward (阶段 2)
  * teleop   — SpaceMouseExpert direct (no InputsControlWrapper), matching
               deploy_backup_policy.py proven pattern
  * protocol — keyboard-driven episode boundaries (阶段 2 keys: S/Enter/Space/Backspace)
               instead of controller trigger buttons
  * bias     — optional J1 U(-0.2, 0.2) encoder bias injected per-episode, so
               demo distribution matches training-time distribution

Operator protocol during collection:
    S          start a new episode (env.reset blocks until S is pressed)
    SpaceMouse full teleop — every step's action is your input
    Enter      success → trajectory appended to buffer, success_count++
    Space      failure → trajectory discarded
    Backspace  discard → trajectory discarded (scene error)
    timeout    (max_episode_length steps) → trajectory discarded
    Ctrl+C     abort — already-collected successes are saved

Output: demos/task_policy/task_policy_demos_{N}_{complete|aborted}_{timestamp}.pkl

Usage:
    python scripts/collect_demo_task_policy.py                 # default 50 successes, bias on
    python scripts/collect_demo_task_policy.py -n 20           # 20 successes
    python scripts/collect_demo_task_policy.py --no-bias       # no bias (clean demos)
"""

import argparse
import copy
import datetime
import logging
import os
import pickle as pkl
import sys
from typing import Tuple

import numpy as np

from frrl.envs.franka_real_config import FrankaRealConfig
from frrl.envs.franka_real_env import FrankaRealEnv
from frrl.fault_injection import EncoderBiasConfig
from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert


def _flush_stdin_buffer() -> None:
    """Discard unread stdin — prevents S/Enter keypresses accumulated during
    collection (pynput on Linux does NOT suppress keys from reaching the terminal)
    from being executed by the shell after the script exits.
    """
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        # Not a tty (redirected stdin, Windows, etc.) → nothing to flush.
        pass


# ----------------------------------------------------------------------
# Pure helpers (tested separately)
# ----------------------------------------------------------------------


def build_action_from_spacemouse(
    sm_state: np.ndarray,
    sm_buttons,
    scale_xyz: float = 1.0,
    scale_rpy: float = 1.0,
) -> np.ndarray:
    """Convert a SpaceMouse sample into the env's 7D action layout.

    SpaceMouse output:  (6D xyz+rpy in [-1, 1], list[int] buttons)
    Env action:         [dx, dy, dz, rx, ry, rz, gripper] in [-1, 1]

    Gripper encoding matches FrankaRealEnv._send_gripper_command:
        button[0] held → -1.0 (close)
        button[1] held → +1.0 (open)
        else          →  0.0 (no change — env rate-limits via gripper_sleep)
    """
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.asarray(sm_state[0:3], dtype=np.float32) * scale_xyz
    action[3:6] = np.asarray(sm_state[3:6], dtype=np.float32) * scale_rpy
    if len(sm_buttons) >= 1 and sm_buttons[0]:
        action[6] = -1.0
    elif len(sm_buttons) >= 2 and sm_buttons[1]:
        action[6] = +1.0
    return action


def make_config(use_bias: bool) -> FrankaRealConfig:
    """FrankaRealConfig for demo collection (keyboard reward + optional J1 bias)."""
    cfg = FrankaRealConfig(reward_backend="keyboard")
    if use_bias:
        cfg.encoder_bias_config = EncoderBiasConfig(
            enable=True,
            error_probability=1.0,
            target_joints=[0],              # J1
            bias_mode="random_uniform",
            bias_range=(-0.2, 0.2),
        )
    return cfg


def build_transition_dict(
    obs, action, next_obs, reward, terminated, info
) -> dict:
    """hil-serl-compatible transition dict (record_demos.py:36-46)."""
    return dict(
        observations=obs,
        actions=action,
        next_observations=next_obs,
        rewards=float(reward),
        masks=1.0 - float(terminated),
        dones=bool(terminated),
        infos=info,
    )


def classify_episode_end(info: dict) -> str:
    """Return 'success' | 'fail' | 'discard' | 'timeout'."""
    if info.get("discard", False):
        return "discard"
    if info.get("succeed", False):
        return "success"
    return "fail"  # fail key pressed OR timeout — both are just "not saved"


def output_filename(output_dir: str, success_count: int, aborted: bool) -> str:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "aborted" if aborted else "complete"
    return os.path.join(
        output_dir,
        f"task_policy_demos_{success_count}_{suffix}_{stamp}.pkl",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", "--successes-needed", type=int, default=50,
                    help="target number of successful demos (default 50)")
    ap.add_argument("--no-bias", action="store_true",
                    help="disable J1 bias injection (default: bias enabled)")
    ap.add_argument("--output-dir", type=str, default="demos/task_policy")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = make_config(use_bias=not args.no_bias)
    env = FrankaRealEnv(cfg)
    spacemouse = SpaceMouseExpert()
    logging.info(
        f"collection ready — reward=keyboard, "
        f"bias={'J1 U(-0.2, 0.2) per episode' if not args.no_bias else 'DISABLED'}, "
        f"target {args.successes_needed} successes"
    )

    transitions = []      # all flat transitions from successful demos
    trajectory = []       # transitions from the CURRENT episode
    success_count = 0
    aborted = False

    try:
        obs, _ = env.reset()  # blocks on S key
        ep_return = 0.0

        while success_count < args.successes_needed:
            sm_state, sm_buttons = spacemouse.get_action()
            action = build_action_from_spacemouse(sm_state, sm_buttons)

            # Every step in demo collection is 100% human teleop → mark
            # is_intervention=True so RLPD routes the transition into the
            # prior/offline buffer (hil-serl record_demos.py convention).
            env.set_intervention(True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

            trajectory.append(build_transition_dict(
                obs, action, next_obs, reward, terminated, info
            ))

            obs = next_obs

            if terminated or truncated:
                outcome = classify_episode_end(info)
                if outcome == "success":
                    # trajectory items are already deep-copied via build_transition_dict
                    # upstream — do not deep-copy again (saves ~3GB per 50 demos).
                    transitions.extend(trajectory)
                    success_count += 1
                    logging.info(
                        f"[DEMO {success_count:3d}/{args.successes_needed}] "
                        f"SAVED (len={len(trajectory)}, return={ep_return:.1f})"
                    )
                else:
                    logging.info(
                        f"[DEMO ---/{args.successes_needed}] "
                        f"DROPPED ({outcome}, len={len(trajectory)}, return={ep_return:.1f})"
                    )
                trajectory = []
                ep_return = 0.0
                if success_count < args.successes_needed:
                    obs, _ = env.reset()

    except KeyboardInterrupt:
        logging.info("Ctrl+C received — saving partial results")
        aborted = True
    finally:
        # Always send the robot home before closing — avoids leaving the arm
        # stopped in an awkward / unsafe end-of-task pose after the last demo.
        try:
            env.go_home()
        except Exception as e:
            logging.warning(f"go_home on shutdown failed: {e}")
        try:
            spacemouse.close()
        finally:
            env.close()
        # Discard S/Enter keystrokes buffered by the terminal during collection
        # so they don't dump into bash as spurious commands after script exit.
        _flush_stdin_buffer()

    if success_count == 0:
        logging.warning("no successful demos collected — not saving anything")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    filename = output_filename(args.output_dir, success_count, aborted)
    with open(filename, "wb") as f:
        pkl.dump(transitions, f)
    logging.info(
        f"saved {success_count} demos ({len(transitions)} transitions) → {filename}"
    )


if __name__ == "__main__":
    main()
