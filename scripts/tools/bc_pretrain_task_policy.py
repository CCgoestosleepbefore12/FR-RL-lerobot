"""BC pretrain task policy actor — pure imitation learning, no critic / no SAC.

为什么独立脚本：SAC pretrain 在 sparse reward + 长 episode (γ^130≈0.013)
的 wipe 任务上 actor 解冻后必崩（critic Q surface 在 demo 中段平坦，actor
沿任意方向 OOD 漂飞）。BC 不依赖 critic gradient，actor 直接 supervised
学 demo 的 (state, action) → 不会崩飞，但 generalization 差（OOD 状态行为
未定义）。

流程：
  1. 加载 SACPolicy + demo pickles 进 ReplayBuffer
  2. Loop: sample batch → actor forward 得 mean → MSE(tanh(mean), demo_action)
  3. 仅训 actor + actor encoder 部分（critic / temperature 不动）
  4. 保存 ckpt 兼容 SAC resume schema（含 random init critic state）

后续 online HIL：用 helper 加载这个 ckpt，进 online 主循环。critic 头几分钟
critic_only_online_steps=2000 自己 calibrate（actor 已是 imitation policy，
不会沿错误 critic gradient 漂走）。

用法：
  python scripts/tools/bc_pretrain_task_policy.py \\
      --config scripts/configs/train_task_policy_franka.json \\
      --demo-paths "data/wipe_demos/*.pkl" \\
      --steps 5000 \\
      --output-dir checkpoints/wipe_bc_pretrain_$(date +%Y%m%d_%H%M%S)

⚠️ 真机后续 resume 时 train_config.json 里的 offline_only_mode 会被覆盖成 false
（resume_online_from_pretrain.sh 会 paste online config），不需要手动改。
"""
import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import draccus
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored

import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
from frrl.configs.train import TrainRLServerPipelineConfig
from frrl.utils.constants import ACTION, TRAINING_STATE_DIR
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.rl.core.buffer import ReplayBuffer
from frrl.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from frrl.utils.utils import init_logging, get_safe_torch_device


def _expand_globs(patterns):
    expanded = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        expanded.extend(matches if matches else [p])
    return expanded


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--demo-paths", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    init_logging()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    demos = _expand_globs(args.demo_paths)
    if not demos:
        print(f"ERROR: no demo files matched {args.demo_paths}", file=sys.stderr)
        sys.exit(1)
    logging.info(f"BC pretrain: {len(demos)} demo file(s), {args.steps} steps")
    for d in demos:
        logging.info(f"  · {d}")

    cfg = draccus.parse(
        config_class=TrainRLServerPipelineConfig,
        config_path=args.config,
        args=[],
    )
    cfg.policy.demo_pickle_paths = demos
    cfg.policy.offline_only_mode = True   # 让 ckpt train_config.json 标记为 pretrain mode
    cfg.policy.offline_pretrain_steps = args.steps
    cfg.output_dir = Path(args.output_dir)
    cfg.batch_size = args.batch_size

    device = get_safe_torch_device(cfg.policy.device, log=True)

    # Load policy
    logging.info("[BC] initializing SACPolicy")
    policy = SACPolicy(config=cfg.policy)
    policy.train()
    policy.to(device)

    # Load demos
    logging.info(f"[BC] loading demos from {len(demos)} pickle(s)")
    state_keys = list(cfg.policy.input_features.keys()) if cfg.policy.input_features else None
    buffer = ReplayBuffer.from_pickle_transitions(
        pickle_paths=demos,
        capacity=cfg.policy.offline_buffer_capacity,
        device=str(device),
        storage_device=cfg.policy.storage_device,
        state_keys=state_keys,
        optimize_memory=False,
        key_map=cfg.policy.demo_key_map,
        transpose_hwc_to_chw=cfg.policy.demo_transpose_hwc_to_chw,
        resize_images=cfg.policy.demo_resize_images,
        normalize_to_unit=cfg.policy.demo_normalize_to_unit,
    )
    logging.info(f"[BC] loaded {buffer.size} transitions into buffer")

    # Optimizer: only actor (含 actor encoder 中可学部分；frozen image encoder 自动跳过)
    actor_params = policy.get_optim_params()["actor"]
    optimizer = torch.optim.Adam(actor_params, lr=args.lr)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[BC] output_dir = {output_dir}")

    # ============================================================
    # BC training loop
    # ============================================================
    logging.info(colored(f"[BC] starting BC pretrain for {args.steps} steps", "cyan", attrs=["bold"]))
    iterator = buffer.get_iterator(batch_size=args.batch_size, async_prefetch=False, queue_size=2)

    losses_window = []
    for step in range(1, args.steps + 1):
        batch = next(iterator)
        observations = batch["state"]
        demo_actions = batch[ACTION]  # shape: (B, 7) tanh-squashed in [-1, 1]

        # actor forward — get mean (deterministic)
        if policy.shared_encoder and policy.actor.encoder.has_images:
            obs_features = policy.actor.encoder.get_cached_image_features(observations)
        else:
            obs_features = None

        _, _, means = policy.actor(observations, obs_features)
        pred_actions = torch.tanh(means)  # in [-1, 1]

        # MSE BC loss
        loss = F.mse_loss(pred_actions, demo_actions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=cfg.policy.grad_clip_norm)
        optimizer.step()

        losses_window.append(loss.item())
        if step % 100 == 0:
            avg = sum(losses_window) / len(losses_window)
            logging.info(f"[BC] step {step}/{args.steps}, mse_loss(window)={avg:.5f}")
            losses_window.clear()

    # ============================================================
    # Save checkpoint (SAC-resumable schema)
    # ============================================================
    logging.info("[BC] saving checkpoint…")
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, args.steps, args.steps)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=args.steps,
        cfg=cfg,
        policy=policy,
        optimizer={"actor": optimizer},  # 占位；resume 时不一定 load actor optim state
        scheduler=None,
    )
    # 占位 training_state 让 lerobot resume load 不挂
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    torch.save({"step": args.steps, "interaction_step": 0},
               os.path.join(training_state_dir, "training_state.pt"))
    update_last_checkpoint(checkpoint_dir)

    logging.info(colored(
        f"[BC] DONE. checkpoint saved to {checkpoint_dir}\n"
        f"     resume online via:\n"
        f"     bash scripts/real/resume_online_from_pretrain.sh {output_dir} learner",
        "green", attrs=["bold"]))


if __name__ == "__main__":
    main()
