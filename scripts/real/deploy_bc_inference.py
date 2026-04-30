"""纯 inference 部署脚本 —— 不启动 learner / 不训练 / 不更新策略权重。

用途：
  1. 隔离调试 BC ckpt 的真实部署行为，剔除 learner、parameter push、warmup 等
     干扰因素。policy 权重整局不变，看 episode 1/2/3... 之间是否仍有 chaos。
  2. 如果观察到第一条 episode 行为正常但第二条乱动 → 排除 actor weight drift，
     把矛头指向 obs 漂移 / actor 过拟合 / 物块位置变化 等环境侧因素。
  3. 不需要 RT PC 之外的额外服务，不开 gRPC 端口。

操作（与 collect_demo_task_policy.py 一致）：
  S        启动 / 重启 episode
  Enter    标记成功（env.reset 后等下次 S）
  Space    标记失败 / discard
  Backspace 同 Space
  Ctrl+C   退出
  SpaceMouse 干预（可选）—— 同 hil-serl 风格 hysteresis 触发：晃动 stick 或按钮。
            干预动作仅 override 当前 step，policy 权重不动。

用法：
  python scripts/real/deploy_bc_inference.py \\
      --ckpt checkpoints/pickup_bc_20260429_171959/checkpoints/020000/pretrained_model \\
      --task pickup

  # 不要干预（pure deterministic policy 测试）：
  python scripts/real/deploy_bc_inference.py \\
      --ckpt <PATH>/pretrained_model --task pickup --no-teleop
"""
import argparse
import logging
import sys
import time
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True,
                    help="path to .../pretrained_model 目录（与 from_pretrained 同语义）")
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIG_FACTORIES.keys()),
                    help="走哪个 task factory 起 env")
    ap.add_argument("--no-bias", action="store_true",
                    help="env 上不注入 J1 encoder bias（部署测试默认关，避免再加扰动）")
    ap.add_argument("--no-teleop", action="store_true",
                    help="不开 SpaceMouse；纯 policy deterministic 测试，最干净的隔离")
    ap.add_argument("--n-episodes", type=int, default=10,
                    help="跑多少条 episode 后退出（不影响 Ctrl+C）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true",
                    help="select_action 用 mean 代替 dist.rsample()，消除 stochastic 噪声"
                         "（结合 --no-teleop 看是否纯 deterministic 也 episode 间不同）")
    # 自动成功判定：复刻 hil-serl pickup wrapper.compute_reward (line 71-106)
    #   (a) tcp_z >= reset_z + lift_threshold（绝对高度 ≥ 桌面以上）
    #   (b) gripper_pos ∈ (held_min, held_max)（真夹住物块，非空夹也非张开）
    # 满足两条就 terminated=True，env.reset 自动起下一条。配 reward_backend="pose"
    # （+ target_pose=None）让 env 不走键盘 wait_for_start，真 auto-reset。
    ap.add_argument("--auto-success", action="store_true", default=True,
                    help="开启 hil-serl pickup 同款 (z + gripper) 自动成功判定，"
                         "默认开。--no-auto-success 关闭走纯键盘 reward")
    ap.add_argument("--no-auto-success", dest="auto_success", action="store_false")
    ap.add_argument("--lift-threshold", type=float, default=0.06,
                    help="tcp_z 抬升触发 success 的阈值 m（默认 0.06m，对齐 hil-serl）")
    ap.add_argument("--gripper-held-min", type=float, default=0.02,
                    help="gripper_pos 下界（< 此值视作空夹）。2026-04-30 从 0.05 降到 0.02："
                         "海绵软材质被 130N 压扁到 ~0.038，原 0.05 阈值会把抓住海绵误判为空夹"
                         "OUT range。0.02 既容纳压扁海绵 (~0.038)，又能过滤空抓 (~0.005)。"
                         "硬物（USB / RAM 等）抓住通常 > 0.05，原默认 0.05 仍可覆盖。")
    ap.add_argument("--gripper-held-max", type=float, default=0.6,
                    help="gripper_pos 上界（> 此值视作张开未夹，hil-serl 默认 0.6）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # ---------- Load policy ----------
    logging.info(f"[deploy] loading SACPolicy from {args.ckpt}")
    cfg_p = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg_p.pretrained_path = args.ckpt
    cfg_p.device = args.device
    policy = SACPolicy.from_pretrained(args.ckpt, config=cfg_p)
    policy.eval().to(args.device)
    logging.info(f"[deploy] num_discrete_actions={policy.config.num_discrete_actions}, "
                 f"continuous_action_dim={policy.actor.action_dim}")

    # ---------- Build env via task factory ----------
    # auto_success=True：用 pose backend + target_pose=None 让 env._compute_reward
    # 走 no-op 分支（real.py:578），跳过键盘 wait_for_start 实现真自动复位。
    # 实际 success 判定在本脚本主循环里做，对齐 hil-serl pickup wrapper。
    backend = "pose" if args.auto_success else "keyboard"
    cfg_env = make_task_config(
        task=args.task,
        use_bias=not args.no_bias,
        reward_backend=backend,
        enable_bias_monitor=False,
        bias_monitor_save_path=None,
    )
    env = FrankaRealEnv(cfg_env)
    reset_z = float(cfg_env.reset_pose[2])
    target_z = reset_z + args.lift_threshold
    logging.info(f"[deploy] env up: gripper_locked={cfg_env.gripper_locked}, "
                 f"max_ep={cfg_env.max_episode_length}, reward_backend={backend}")
    if args.auto_success:
        logging.info(
            f"[deploy] auto-success: tcp_z >= {target_z:.3f}m "
            f"(reset_z {reset_z:.3f} + lift {args.lift_threshold:.2f}) "
            f"AND gripper_pos ∈ ({args.gripper_held_min}, {args.gripper_held_max})"
        )

    # ---------- Optional SpaceMouse for intervention ----------
    spacemouse = None
    if not args.no_teleop:
        from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
        spacemouse = SpaceMouseExpert()
        logging.info("[deploy] SpaceMouse 干预开启：晃动 stick 或按按钮覆盖 policy 输出")
    else:
        logging.info("[deploy] --no-teleop：纯 policy 输出，无任何人工干预")

    # ---------- Inference loop ----------
    success_cnt = 0
    abort = False
    try:
        for ep_idx in range(args.n_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
            ep_intvn = 0
            policy_actions_log = []     # 每条 episode 收集 action 时间序列做后续诊断

            logging.info(f"[deploy] ===== episode {ep_idx + 1}/{args.n_episodes} 开始 =====")

            done = False
            while not done:
                # ---------- batch obs ----------
                batch = {}
                for k, v in obs.items():
                    if k == "agent_pos":
                        batch["observation.state"] = (
                            torch.from_numpy(np.asarray(v, dtype=np.float32))
                            .unsqueeze(0).to(args.device)
                        )
                    elif k == "pixels":
                        for cam_name, img in v.items():
                            # uint8 (H,W,3) → float [0,1] (1,3,H,W)
                            t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()
                            batch[f"observation.images.{cam_name}"] = (
                                t.unsqueeze(0).to(args.device) / 255.0
                            )
                    elif k == "environment_state":
                        batch["observation.environment_state"] = (
                            torch.from_numpy(np.asarray(v, dtype=np.float32))
                            .unsqueeze(0).to(args.device)
                        )

                # ---------- policy forward ----------
                with torch.no_grad():
                    if args.deterministic:
                        # 走 mean（不 sample）：消除 std 噪声看 episode 间确定性
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
                action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

                # ---------- intervention override ----------
                is_intvn = False
                if spacemouse is not None:
                    sm_state, sm_buttons = spacemouse.get_action()
                    sm_mag = float(np.linalg.norm(sm_state[:6]))
                    any_button = any(sm_buttons)
                    if sm_mag > 0.05 or any_button:
                        # 直接覆盖；hysteresis 这里不做（部署测试关心当下 step）
                        action_np[0:3] = sm_state[0:3].astype(np.float32)
                        action_np[3:6] = sm_state[3:6].astype(np.float32)
                        if len(sm_buttons) >= 1 and sm_buttons[0]:
                            action_np[6] = -1.0
                        elif len(sm_buttons) >= 2 and sm_buttons[1]:
                            action_np[6] = +1.0
                        is_intvn = True
                        env.set_intervention(True)

                # ---------- step env ----------
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                policy_actions_log.append(action_np.copy())
                ep_return += float(reward)
                ep_len += 1
                if is_intvn:
                    ep_intvn += 1

                obs = next_obs
                done = terminated or truncated

                # ---------- auto-success 判定（hil-serl pickup wrapper:71-106 同款）----------
                # tcp_pose_true layout: agent_pos[0:3]=xyz, agent_pos[13]=gripper_pos
                if args.auto_success and not done:
                    tcp_z = float(np.asarray(obs["agent_pos"], dtype=np.float32)[2])
                    grip = float(np.asarray(obs["agent_pos"], dtype=np.float32)[13])
                    z_high = tcp_z >= target_z
                    held = args.gripper_held_min < grip < args.gripper_held_max
                    if z_high and held:
                        logging.info(
                            f"[auto-success] ✓ tcp_z={tcp_z:.3f}m (≥{target_z:.3f}), "
                            f"gripper={grip:.3f} ∈ ({args.gripper_held_min}, {args.gripper_held_max})"
                        )
                        info["succeed"] = True
                        ep_return = max(ep_return, 1.0)
                        done = True
                    elif tcp_z >= target_z - 0.03:
                        # 接近阈值时打 debug，方便排查"已抬起但 gripper 不在范围"等情况
                        logging.info(
                            f"[auto-success] near miss: tcp_z={tcp_z:.3f} "
                            f"(target {target_z:.3f}), gripper={grip:.3f} "
                            f"({'OK' if held else 'OUT'} range)"
                        )

            # ---------- episode end ----------
            outcome = "success" if info.get("succeed") else (
                "discard" if info.get("discard") else "fail/timeout"
            )
            if outcome == "success":
                success_cnt += 1

            # 诊断：打印 episode 内 action 统计，方便对比 episode 1 vs episode N
            actions_arr = np.stack(policy_actions_log)
            xyz_mean = actions_arr[:, :3].mean(axis=0)
            xyz_std = actions_arr[:, :3].std(axis=0)
            grip_dist = {
                -1: int((actions_arr[:, 6] < -0.5).sum()),
                 0: int(((actions_arr[:, 6] >= -0.5) & (actions_arr[:, 6] <= 0.5)).sum()),
                +1: int((actions_arr[:, 6] > 0.5).sum()),
            }
            logging.info(
                f"[deploy] ep {ep_idx + 1} {outcome.upper()} (len={ep_len}, "
                f"intvn={ep_intvn}, return={ep_return:.2f}, success_so_far={success_cnt})"
            )
            logging.info(
                f"          action[:3] mean={np.round(xyz_mean, 4).tolist()}, "
                f"std={np.round(xyz_std, 4).tolist()}, gripper={grip_dist}"
            )

    except KeyboardInterrupt:
        logging.info("[deploy] Ctrl+C — exiting")
        abort = True
    finally:
        try:
            env.go_home()
        except Exception as e:
            logging.warning(f"go_home on shutdown failed: {e}")
        try:
            import requests
            requests.post("http://192.168.100.1:5000/clear_encoder_bias", timeout=2.0)
        except Exception:
            pass
        if spacemouse is not None:
            try: spacemouse.close()
            except Exception: pass
        env.close()
        _flush_stdin_buffer()

    logging.info(f"[deploy] DONE. success {success_cnt}/{args.n_episodes}{' (aborted)' if abort else ''}")


if __name__ == "__main__":
    main()
