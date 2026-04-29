"""HG-DAgger 介入数据采集 —— 部署 BC 策略 + 录制人类介入帧。

为什么独立脚本：deploy_bc_inference.py 用于 *诊断* policy 的部署行为（不存盘）；
collect_demo_task_policy.py 用于 *从零* 采 demo（每帧都是介入）。本脚本介于两者之间——
让 BC 自主跑，仅在策略卡死/OOD 时人介入纠错，落盘时每帧带 `is_intervention` 标签，
后续 BC trainer 可只取介入帧（HG-DAgger 标准做法），与原始 demo 合并重训。

操作协议（与 collect_demo_task_policy.py 一致）：
  S          start episode
  Enter      mark success → 整条 episode 全帧落盘（无论介入/自主）
  Space      mark fail → discard
  Backspace  discard
  Ctrl+C     退出（已存的 success 仍 dump）

介入检测（hysteresis 状态机）：
  IDLE      sm 静止 → policy action，is_intervention=False
  IDLE → ACTIVE
            sm_mag > enter_thresh 或任意按键 → 立即切换
  ACTIVE    sm 动作覆盖 policy，is_intervention=True
  ACTIVE → IDLE
            sm 持续静止 tail_k 帧后退回 IDLE。
            tail 期间 action 仍取 sm（near-zero），标 is_intervention=True，
            捕捉操作员"释放后 follow-through" 信号。

输出：data/{task}_dagger/{task}_dagger_iter{ITER}_{N}_{stamp}.pkl
schema 与 hil-serl record_demos.py 同款，infos 含 is_intervention bool。

用法：
  python scripts/real/deploy_bc_with_dagger.py \\
      --ckpt checkpoints/pickup_bc_*/checkpoints/020000/pretrained_model \\
      --task pickup --iter 1 -n 30
"""
import argparse
import datetime
import logging
import os
import pickle as pkl
import sys

import numpy as np
import torch

import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
from frrl.configs.policies import PreTrainedConfig
from frrl.envs.real import FrankaRealEnv
from frrl.envs.real_config import TASK_CONFIG_FACTORIES, make_task_config
from frrl.policies.sac.modeling_sac import SACPolicy


def _flush_stdin_buffer() -> None:
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _obs_to_batch(obs: dict, device: str) -> dict:
    """将 env.reset/step 返回的原始 obs 字典转成 policy.select_action 期望的 batched tensor。

    与 deploy_bc_inference.py 保持一致：图像 uint8(H,W,3) → float [0,1] (1,3,H,W)，
    state float32 → (1, D)。
    """
    batch = {}
    for k, v in obs.items():
        if k == "agent_pos":
            batch["observation.state"] = (
                torch.from_numpy(np.asarray(v, dtype=np.float32))
                .unsqueeze(0).to(device)
            )
        elif k == "pixels":
            for cam_name, img in v.items():
                t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()
                batch[f"observation.images.{cam_name}"] = (
                    t.unsqueeze(0).to(device) / 255.0
                )
        elif k == "environment_state":
            batch["observation.environment_state"] = (
                torch.from_numpy(np.asarray(v, dtype=np.float32))
                .unsqueeze(0).to(device)
            )
    return batch


def _policy_action(policy: SACPolicy, batch: dict, deterministic: bool) -> np.ndarray:
    """走与 deploy_bc_inference.py 同款的 policy forward，返回 7D action np。"""
    with torch.no_grad():
        if deterministic:
            if policy.shared_encoder and policy.actor.encoder.has_images:
                obs_feat = policy.actor.encoder.get_cached_image_features(batch)
            else:
                obs_feat = None
            obs_enc = policy.actor.encoder(batch, cache=obs_feat, detach=True)
            outputs = policy.actor.network(obs_enc)
            means = policy.actor.mean_layer(outputs)
            action = torch.tanh(means)
            if policy.config.num_discrete_actions is not None:
                disc_q = policy.discrete_critic(batch, obs_feat)
                disc_idx = torch.argmax(disc_q, dim=-1, keepdim=True)
                disc_action = (disc_idx - 1).to(action.dtype)
                action = torch.cat([action, disc_action], dim=-1)
        else:
            action = policy.select_action(batch)
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _build_sm_action(sm_state: np.ndarray, sm_buttons) -> np.ndarray:
    """与 collect_demo_task_policy.py.build_action_from_spacemouse 同款约定。

    button[0] held → -1.0 close, button[1] held → +1.0 open, else → 0.0 no-op。
    """
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.asarray(sm_state[0:3], dtype=np.float32)
    action[3:6] = np.asarray(sm_state[3:6], dtype=np.float32)
    if len(sm_buttons) >= 1 and sm_buttons[0]:
        action[6] = -1.0
    elif len(sm_buttons) >= 2 and sm_buttons[1]:
        action[6] = +1.0
    return action


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True,
                    help="path to .../pretrained_model 目录")
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIG_FACTORIES.keys()))
    ap.add_argument("--no-bias", action="store_true",
                    help="禁用 J1 encoder bias 注入（默认开，让 DAgger 介入数据与 deploy 分布一致）")
    ap.add_argument("-n", "--successes-needed", type=int, default=30,
                    help="目标 success episode 数")
    ap.add_argument("--iter", type=int, default=1,
                    help="DAgger 迭代轮次编号；写入文件名方便后续合并管理")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="default: data/{task}_dagger/")
    ap.add_argument("--enter-threshold", type=float, default=0.05,
                    help="sm magnitude > 此值或按键 → 立刻进 ACTIVE 状态（默认 0.05，"
                         "对齐 frrl/teleoperators/spacemouse/configuration_spacemouse.py）")
    ap.add_argument("--tail-k", type=int, default=10,
                    help="sm 静止后保持 ACTIVE 的帧数（默认 10 ≈ 1s @ 10fps）。"
                         "捕捉操作员释放后的 follow-through 信号；过短会丢恢复尾巴，"
                         "过长会把'已经回到正常'帧也标成介入污染 BC")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true",
                    help="select_action 走 mean 不 sample（与 deploy_bc_inference 同款）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    output_dir = args.output_dir or f"data/{args.task}_dagger"

    # ---------- Load policy ----------
    logging.info(f"[dagger] loading SACPolicy from {args.ckpt}")
    cfg_p = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg_p.pretrained_path = args.ckpt
    cfg_p.device = args.device
    policy = SACPolicy.from_pretrained(args.ckpt, config=cfg_p)
    policy.eval().to(args.device)

    # ---------- Build env ----------
    cfg_env = make_task_config(
        task=args.task,
        use_bias=not args.no_bias,
        reward_backend="keyboard",  # 操作员按 Enter/Space 决定 success/discard
        enable_bias_monitor=False,
        bias_monitor_save_path=None,
    )
    env = FrankaRealEnv(cfg_env)
    logging.info(
        f"[dagger] env up: task={args.task}, gripper_locked={cfg_env.gripper_locked}, "
        f"max_ep_len={cfg_env.max_episode_length}, bias={'ON' if not args.no_bias else 'OFF'}, "
        f"target {args.successes_needed} successes"
    )

    # ---------- SpaceMouse ----------
    from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
    spacemouse = SpaceMouseExpert()
    logging.info(
        f"[dagger] hysteresis state machine: enter_thresh={args.enter_threshold}, "
        f"tail_k={args.tail_k} frames"
    )

    transitions = []      # 全部 success episode 拼起来的 flat transitions
    trajectory = []       # 当前 episode 的 transitions
    success_count = 0
    aborted = False

    # 介入状态机
    INTV_IDLE = 0
    INTV_ACTIVE = 1
    intv_state = INTV_IDLE
    intv_idle_count = 0   # ACTIVE 状态下 sm 已经连续静止的帧数

    try:
        obs, _ = env.reset()  # blocks on S
        ep_return = 0.0
        ep_intvn = 0
        ep_len = 0

        while success_count < args.successes_needed:
            batch = _obs_to_batch(obs, args.device)
            policy_action = _policy_action(policy, batch, args.deterministic)

            sm_state, sm_buttons = spacemouse.get_action()
            sm_mag = float(np.linalg.norm(sm_state[:6]))
            sm_active = (sm_mag > args.enter_threshold) or any(sm_buttons)

            # 状态机更新（与 hil-serl teleop_spacemouse.py 同款 hysteresis 逻辑，
            # 但用 tail_k 替代默认 persist_steps=3，给操作员更宽容的释放窗口）
            if sm_active:
                intv_state = INTV_ACTIVE
                intv_idle_count = 0
            elif intv_state == INTV_ACTIVE:
                intv_idle_count += 1
                if intv_idle_count >= args.tail_k:
                    intv_state = INTV_IDLE
                    intv_idle_count = 0

            # 选 action：ACTIVE 期间（含 tail）一律用 sm；IDLE 用 policy
            if intv_state == INTV_ACTIVE:
                action = _build_sm_action(sm_state, sm_buttons)
                env.set_intervention(True)
                is_intervention = True
                ep_intvn += 1
            else:
                action = policy_action
                is_intervention = False

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            # 关键：每个 transition 标 is_intervention，让 buffer.from_pickle_transitions
            # 在 BC 训练时按 --intervention-only 过滤
            info["is_intervention"] = is_intervention

            trajectory.append({
                "observations": obs,
                "actions": action,
                "next_observations": next_obs,
                "rewards": float(reward),
                "masks": 1.0 - float(terminated),
                "dones": bool(terminated),
                "infos": info,
            })

            obs = next_obs

            if terminated or truncated:
                outcome = (
                    "discard" if info.get("discard", False) else
                    "success" if info.get("succeed", False) else
                    "fail"
                )
                if outcome == "success":
                    transitions.extend(trajectory)
                    success_count += 1
                    pct = 100.0 * ep_intvn / max(1, ep_len)
                    logging.info(
                        f"[DAGGER {success_count:3d}/{args.successes_needed}] "
                        f"SAVED len={ep_len}, intvn={ep_intvn} ({pct:.1f}%), "
                        f"return={ep_return:.1f}"
                    )
                else:
                    logging.info(
                        f"[DAGGER ---/{args.successes_needed}] "
                        f"DROPPED ({outcome}, len={ep_len}, intvn={ep_intvn})"
                    )
                trajectory = []
                ep_return = 0.0
                ep_intvn = 0
                ep_len = 0
                intv_state = INTV_IDLE
                intv_idle_count = 0
                if success_count < args.successes_needed:
                    obs, _ = env.reset()

    except KeyboardInterrupt:
        logging.info("[dagger] Ctrl+C — saving partial results")
        aborted = True
    finally:
        try:
            env.go_home()
        except Exception as e:
            logging.warning(f"go_home on shutdown failed: {e}")
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
        _flush_stdin_buffer()

    if success_count == 0:
        logging.warning("[dagger] no successful episodes — nothing saved")
        return

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "aborted" if aborted else "complete"
    filename = os.path.join(
        output_dir,
        f"{args.task}_dagger_iter{args.iter}_{success_count}_{suffix}_{stamp}.pkl",
    )
    with open(filename, "wb") as f:
        pkl.dump(transitions, f)

    n_total = len(transitions)
    n_intvn = sum(1 for t in transitions if t["infos"].get("is_intervention", False))
    pct = 100.0 * n_intvn / max(1, n_total)
    logging.info(
        f"[dagger] saved {success_count} success episodes "
        f"({n_total} transitions, intvn {n_intvn}/{n_total}={pct:.1f}%) → {filename}"
    )


if __name__ == "__main__":
    main()
