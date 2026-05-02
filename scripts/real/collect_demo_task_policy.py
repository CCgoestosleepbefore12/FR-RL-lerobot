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

Output: data/{task}_demos/{task}_demos_{N}_{complete|aborted}_{timestamp}.pkl

Usage:
    python scripts/real/collect_demo_task_policy.py --task wipe                # 50 wipe demos, bias on, gripper closed
    python scripts/real/collect_demo_task_policy.py --task pickup -n 30        # 30 pickup demos, bias on, gripper free
    python scripts/real/collect_demo_task_policy.py --task pickup --no-bias    # baseline (no bias)
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

from frrl.envs.real_config import FrankaRealConfig, TASK_CONFIG_FACTORIES, make_task_config
from frrl.envs.real import FrankaRealEnv
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


def build_action_from_spacemouse_locked_gripper(
    sm_state, sm_buttons,
    locked_value: float,
    scale_xyz: float = 1.0,
    scale_rpy: float = 1.0,
) -> np.ndarray:
    """夹爪锁定模式：忽略 sm_buttons，action[6] 强制 const。"""
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.asarray(sm_state[0:3], dtype=np.float32) * scale_xyz
    action[3:6] = np.asarray(sm_state[3:6], dtype=np.float32) * scale_rpy
    action[6] = locked_value
    return action


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


def make_config(task: str, use_bias: bool) -> FrankaRealConfig:
    """Dispatch to per-task factory in real_config.py。

    采集模式默认开 BiasMonitor 弹窗（仅本地 demo 时；远程 headless 训练用默认 False）。
    bias_monitor_save_path 默认存到 charts/bias_<timestamp>.{npz,png}。
    """
    bias_monitor_save_path = None
    if use_bias:
        from datetime import datetime
        from pathlib import Path
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bias_monitor_save_path = str(charts_dir / f"bias_{ts}")
    return make_task_config(
        task,
        use_bias=use_bias,
        reward_backend="keyboard",
        enable_bias_monitor=use_bias,
        bias_monitor_save_path=bias_monitor_save_path,
    )


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


def output_filename(output_dir: str, success_count: int, aborted: bool, task: str = "task_policy") -> str:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "aborted" if aborted else "complete"
    return os.path.join(
        output_dir,
        f"{task}_demos_{success_count}_{suffix}_{stamp}.pkl",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIG_FACTORIES.keys()),
                    help="task name → 决定 gripper / episode 长度 / xy 随机化等。"
                         f" 已注册: {list(TASK_CONFIG_FACTORIES.keys())}")
    ap.add_argument("-n", "--successes-needed", type=int, default=50,
                    help="target number of successful demos (default 50)")
    ap.add_argument("--no-bias", action="store_true",
                    help="disable J1 bias injection (default: bias enabled)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="default: data/{task}_demos/")
    ap.add_argument("--save-every", type=int, default=10,
                    help="每 N 条 success 写一次 partial pkl 防 crash 丢数据。0=禁用。"
                         "partial 文件名 {task}_demos_inprogress.pkl，主 pkl 写成功后自动删。")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # 默认目录按 bias 状态分流（2026-04-30 数据重组）：
    #   --no-bias    → data/no_bias/{task}/
    #   default ON   → data/with_bias/{task}/
    # --output-dir 显式指定时覆盖此默认。
    bias_dir = "no_bias" if args.no_bias else "with_bias"
    output_dir = args.output_dir or f"data/{bias_dir}/{args.task}"

    cfg = make_config(task=args.task, use_bias=not args.no_bias)
    env = FrankaRealEnv(cfg)
    spacemouse = SpaceMouseExpert()
    logging.info(
        f"collection ready — task={args.task}, reward=keyboard, "
        f"bias={'J1 U(-0.2, 0.2) per episode' if not args.no_bias else 'DISABLED'}, "
        f"gripper_locked={cfg.gripper_locked}, max_ep_len={cfg.max_episode_length}, "
        f"random_reset={cfg.random_reset} (xy_range={cfg.random_xy_range}), "
        f"output_dir={output_dir}, target {args.successes_needed} successes"
    )

    # 锁定夹爪时 SpaceMouse 按键无效，action[6] 强制 const（-1=closed, +1=open）。
    if cfg.gripper_locked == "closed":
        _locked_val = -1.0
    elif cfg.gripper_locked == "open":
        _locked_val = +1.0
    else:
        _locked_val = None  # 自由模式

    transitions = []      # all flat transitions from successful demos
    trajectory = []       # transitions from the CURRENT episode
    success_count = 0
    aborted = False

    # partial save 路径 —— 每 --save-every 条 success 原子写一次，防 RuntimeError /
    # 段错误等异常丢数据。主 pkl 写成功后会被删掉。
    inprogress_path = os.path.join(output_dir, f"{args.task}_demos_inprogress.pkl")

    def _save_partial():
        """原子写 inprogress.pkl：先写 .tmp 再 rename，避免写一半 crash 留半截文件。"""
        os.makedirs(output_dir, exist_ok=True)
        tmp = inprogress_path + ".tmp"
        with open(tmp, "wb") as f:
            pkl.dump(transitions, f)
        os.replace(tmp, inprogress_path)
        logging.info(f"[partial save] {success_count} demos / {len(transitions)} transitions → {inprogress_path}")

    try:
        obs, _ = env.reset()  # blocks on S key
        ep_return = 0.0

        while success_count < args.successes_needed:
            sm_state, sm_buttons = spacemouse.get_action()
            if _locked_val is None:
                action = build_action_from_spacemouse(sm_state, sm_buttons)
            else:
                action = build_action_from_spacemouse_locked_gripper(
                    sm_state, sm_buttons, locked_value=_locked_val,
                )

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
                    # 增量保存防 crash 丢数据（每 save_every 条成功写一次）
                    if args.save_every > 0 and success_count % args.save_every == 0:
                        _save_partial()
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
    except Exception as e:
        # ⚠️ 关键：env.reset() 抛 RuntimeError、SpaceMouse / RT PC 突发故障等都
        # 走这条路。之前只 catch KeyboardInterrupt，RuntimeError 直接逃出
        # try-except → 跳过下面 pkl.dump → 几十条 success 内存数据全丢（实测
        # 2026-04-30 在 demo 62 时丢了前 61 条）。这里 catch 后让 finally 里
        # 写 inprogress 兜底，再 re-raise 让 traceback 还能看到根因。
        import traceback
        logging.error(f"采集中遇到异常 — 尝试保存 partial 数据: {type(e).__name__}: {e}")
        traceback.print_exc()
        aborted = True
    finally:
        # Always send the robot home before closing — avoids leaving the arm
        # stopped in an awkward / unsafe end-of-task pose after the last demo.
        try:
            env.go_home()
        except Exception as e:
            logging.warning(f"go_home on shutdown failed: {e}")
        # 清 RT PC 残留 bias，防下次跑别的脚本读到上 episode 的 bias。go_home
        # 不触发 on_episode_start，所以这里显式清。
        try:
            import requests
            requests.post("http://192.168.100.1:5000/clear_encoder_bias", timeout=2.0)
            logging.info("encoder_bias cleared on shutdown")
        except Exception as e:
            logging.warning(f"clear_encoder_bias on shutdown failed: {e}")
        try:
            spacemouse.close()
        finally:
            env.close()
        # Discard S/Enter keystrokes buffered by the terminal during collection
        # so they don't dump into bash as spurious commands after script exit.
        _flush_stdin_buffer()

    if success_count == 0:
        logging.warning("no successful demos collected — not saving anything")
        # 即便 0 success 也清掉残留 inprogress（重跑别误读）
        if os.path.exists(inprogress_path):
            os.remove(inprogress_path)
        return

    os.makedirs(output_dir, exist_ok=True)
    filename = output_filename(output_dir, success_count, aborted, task=args.task)
    with open(filename, "wb") as f:
        pkl.dump(transitions, f)
    logging.info(
        f"saved {success_count} demos ({len(transitions)} transitions) → {filename}"
    )
    # 主文件写成功 → 删 inprogress（不留残留，避免下次运行被认为是上次 crash 的恢复文件）
    if os.path.exists(inprogress_path):
        os.remove(inprogress_path)


if __name__ == "__main__":
    main()
