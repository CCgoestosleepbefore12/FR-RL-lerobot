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
    ap.add_argument("--steps", type=int, default=20000,
                    help="对齐 hil-serl train_bc.py 默认 20k。50 demo × 100 step = 5k transitions, "
                         "20k step batch=128 ≈ 64 epoch，足够小数据 BC 收敛")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--action-norm-min", type=float, default=1e-6,
                    help="丢弃 ||action|| ≤ 此阈值的静止帧（hil-serl train_bc.py:183 同款），"
                         "0=不过滤；1e-6 通常足够剔除全零 action，含 gripper close 帧（norm≥1）保留")
    ap.add_argument("--image-aug-pad", type=int, default=4,
                    help="图像 random shift 像素数（hil-serl batched_random_crop padding=4 同款），"
                         "对 128² 图实测让小数据 BC 泛化显著提升；0=禁用")
    ap.add_argument("--discrete-bc-weight", type=float, default=1.0,
                    help="discrete head BC loss 权重（CE on gripper {-1,0,+1}→idx{0,1,2}）。"
                         "0=禁用（仅训 continuous actor）。仅当 cfg.policy.num_discrete_actions != None 时生效")
    ap.add_argument("--save-tail-every", type=int, default=0,
                    help="对齐 hil-serl 末尾密集保存：>0 时训练最后 100 步内每 N 步存一次 ckpt（默认仅末步存）")
    ap.add_argument("--intervention-only", action="store_true",
                    help="HG-DAgger 模式：只载入 infos.is_intervention=True 的 transition。"
                         "老 demo（无该 key）默认全是 intervention（全程人 teleop），向后兼容；"
                         "新 dagger pkl 在此 flag 下只取人类纠错帧，自主帧（policy 当时是对的）丢弃。"
                         "典型用法：--demo-paths 'data/pickup_demos/*.pkl' 'data/pickup_dagger/*.pkl' "
                         "--intervention-only → 原 demo 全用 + dagger 介入帧补强。")
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
        action_norm_min=args.action_norm_min,  # 丢静止帧，hil-serl 同款
        intervention_only=args.intervention_only,
    )
    logging.info(f"[BC] loaded {buffer.size} transitions into buffer "
                 f"(action_norm_min={args.action_norm_min}, "
                 f"intervention_only={args.intervention_only})")

    # ============================================================
    # BC-specific 修正：默认 SAC 设计下 shared encoder 由 critic 更新、actor 走 detach。
    # 但 BC 没有 critic，必须让 actor loss 直接训 encoder 才能学到 spatial→action 映射。
    # 否则 spatial_embeddings + post_encoders + state_encoder 全停在 random init，
    # actor MLP 头只能在 random feature 上做 lookup-style 拟合，完全没泛化（典型表现：
    # 真机 deployment 时 actor 输出近常数，连"下移/上抬"都做不出来）。
    #
    # 修法：(1) 临时关掉 actor 的 encoder_is_shared → forward 不 detach；
    #       (2) optimizer 收所有 requires_grad 参数（DINOv3 backbone 已 frozen 不会被 update）。
    # 保存 ckpt 前恢复 encoder_is_shared=True，让 SAC online resume 行为正确。
    # ============================================================
    _original_encoder_is_shared = policy.actor.encoder_is_shared
    policy.actor.encoder_is_shared = False  # BC 期间 forward 不 detach

    # 收集 actor 路径上所有可训参数（含 encoder 的 spatial/post/state 但跳过 frozen DINOv3 backbone）。
    seen = set()
    bc_params = []
    # actor.encoder（含 image_encoder, spatial_embeddings, post_encoders, state_encoder）
    for p in policy.actor.encoder.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); bc_params.append(p)
    # actor.network MLP + mean_layer + std_layer
    for n, p in policy.actor.named_parameters():
        if n.startswith("encoder"): continue   # 上面已收过
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); bc_params.append(p)
    n_train = sum(p.numel() for p in bc_params)
    logging.info(f"[BC] trainable params for BC = {n_train:,} (encoder + actor MLP, frozen backbone excluded)")

    optimizer = torch.optim.Adam(bc_params, lr=args.lr)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[BC] output_dir = {output_dir}")

    # ============================================================
    # BC training loop
    # ============================================================
    logging.info(colored(f"[BC] starting BC pretrain for {args.steps} steps", "cyan", attrs=["bold"]))
    iterator = buffer.get_iterator(batch_size=args.batch_size, async_prefetch=False, queue_size=2)

    # Hybrid SAC：actor 只输出 continuous_action_dim（如 6 = ee delta），剩下的
    # 维度（如 gripper）由 discrete grasp head 学。
    continuous_action_dim = policy.actor.action_dim
    has_discrete = (cfg.policy.num_discrete_actions is not None and args.discrete_bc_weight > 0)
    if has_discrete:
        n_discrete = cfg.policy.num_discrete_actions
        # 把 discrete_critic 加入 optimizer。BC 期间 critic 不动（critic_ensemble），
        # 仅 discrete_critic 用 demo 上 gripper 三态 CE 学 close/no-op/open 判定。
        # 配 critic_lr 不是 actor_lr：discrete head 是 critic 类，对齐 SAC 主 loop。
        for p in policy.discrete_critic.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); bc_params.append(p)
        # 重建 optimizer 把 discrete head 也吸进来
        optimizer = torch.optim.Adam(bc_params, lr=args.lr)
        logging.info(f"[BC] discrete head BC enabled: weight={args.discrete_bc_weight}, "
                     f"n_classes={n_discrete}, params now {sum(p.numel() for p in bc_params):,}")

    # NLL loss（tanh-Gaussian negative log-likelihood）：同时训练 mean_layer + std_layer。
    # 纯 MSE on tanh(mean) 只训 mean_layer，std_layer 停在 random init，部署 select_action()
    # 走 dist.rsample() 时 std≈[0.1, 5] 撒向 [-1,1] 全域，policy 行为完全乱。
    # NLL 推 std 自然下降到 demo 给定 state 下的真实 action 不确定度（typically 0.05~0.2）。
    from frrl.policies.sac.modeling_sac import TanhMultivariateNormalDiag, DISCRETE_DIMENSION_INDEX
    from frrl.rl.core.buffer import random_shift

    image_keys = [k for k in (state_keys or []) if k.startswith("observation.images.")]
    if args.image_aug_pad > 0 and image_keys:
        logging.info(f"[BC] image augmentation: random_shift(pad={args.image_aug_pad}) on {image_keys}")

    losses_window = []
    losses_disc_window = []
    for step in range(1, args.steps + 1):
        batch = next(iterator)
        observations = batch["state"]

        # P0-1: 图像 random shift 增强（hil-serl batched_random_crop padding=4 同款）
        # 对 NCHW 图像在每个 batch 独立采样 (offset_h, offset_w)，pad-replicate 后 crop 回原尺寸。
        # 让 actor / encoder 学到 spatial-translation 不变性，小数据 BC 关键 trick。
        if args.image_aug_pad > 0:
            observations = dict(observations)  # 浅拷贝避免污染 buffer 内部缓存
            for k in image_keys:
                if k in observations:
                    observations[k] = random_shift(observations[k], pad=args.image_aug_pad)

        demo_actions_full = batch[ACTION]  # shape: (B, full_action_dim)
        demo_actions = demo_actions_full[:, :continuous_action_dim]
        # clamp 远离 ±1：tanh-Gaussian log_prob 内部 atanh，输入到 ±1 时 log(0) → -inf
        demo_actions = demo_actions.clamp(-0.999, 0.999)

        # actor forward — 拿到 mean + std；走 detach=False 让 encoder 也吃 BC 梯度
        if policy.shared_encoder and policy.actor.encoder.has_images:
            obs_features = policy.actor.encoder.get_cached_image_features(observations)
        else:
            obs_features = None

        # 手动重做 actor.forward 以拿到 std（actor.forward 只返回 sampled+log_prob+mean，没暴露 std）
        obs_enc = policy.actor.encoder(observations, cache=obs_features, detach=False)
        outputs = policy.actor.network(obs_enc)
        means = policy.actor.mean_layer(outputs)
        log_std = policy.actor.std_layer(outputs)
        std = torch.exp(log_std).clamp(policy.actor.std_min, policy.actor.std_max)

        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
        log_prob = dist.log_prob(demo_actions)              # (B,)
        loss_actor = -log_prob.mean()

        # P1: discrete head BC loss（CE on gripper 三态）
        # demo 里 action[-1] ∈ {-1, 0, +1}，与 modeling_sac.compute_loss_discrete_critic 同样
        # 转换：round(a)+1 → idx ∈ {0, 1, 2}，clamp 防 demo 噪声越界。discrete_critic
        # forward 不需要 action 输入（只看 obs），输出 (B, n_classes) Q logits，CE 直接喂。
        loss_disc = torch.tensor(0.0, device=device)
        if has_discrete:
            gripper_action = demo_actions_full[:, DISCRETE_DIMENSION_INDEX:]   # (B, 1)
            gripper_idx = (torch.round(gripper_action) + 1).long().clamp(0, n_discrete - 1).squeeze(-1)
            disc_logits = policy.discrete_critic(observations, obs_features)   # (B, n_classes)
            loss_disc = F.cross_entropy(disc_logits, gripper_idx)

        loss = loss_actor + args.discrete_bc_weight * loss_disc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bc_params, max_norm=cfg.policy.grad_clip_norm)
        optimizer.step()

        losses_window.append(loss_actor.item())
        if has_discrete:
            losses_disc_window.append(loss_disc.item())
        if step % 100 == 0:
            avg = sum(losses_window) / len(losses_window)
            mean_std = std.mean().item()
            disc_str = ""
            if has_discrete and losses_disc_window:
                avg_disc = sum(losses_disc_window) / len(losses_disc_window)
                disc_str = f", disc_ce={avg_disc:.4f}"
                losses_disc_window.clear()
            logging.info(f"[BC] step {step}/{args.steps}, nll_loss={avg:.4f}, "
                         f"mean(std)={mean_std:.4f}{disc_str}")
            losses_window.clear()

        # P1.5: 末尾密集保存（hil-serl train_bc.py:119-122 同款）
        if (
            args.save_tail_every > 0
            and step > args.steps - 100
            and step % args.save_tail_every == 0
            and step != args.steps  # 末步另外保存
        ):
            tail_dir = get_step_checkpoint_dir(cfg.output_dir, args.steps, step)
            policy.actor.encoder_is_shared = _original_encoder_is_shared
            save_checkpoint(checkpoint_dir=tail_dir, step=step, cfg=cfg, policy=policy,
                           optimizer={"actor": optimizer}, scheduler=None)
            tail_state = os.path.join(tail_dir, TRAINING_STATE_DIR)
            os.makedirs(tail_state, exist_ok=True)
            torch.save({"step": step, "interaction_step": 0},
                       os.path.join(tail_state, "training_state.pt"))
            policy.actor.encoder_is_shared = False  # 恢复 BC 模式
            logging.info(f"[BC] tail ckpt saved at step {step} → {tail_dir}")

    # ============================================================
    # Save checkpoint (SAC-resumable schema)
    # ============================================================
    # 恢复 encoder_is_shared=True：SAC online resume 时 actor 必须走 detach 路径
    # 否则 actor loss 会回传到 encoder 与 critic loss 在同 encoder 参数上互相拉扯。
    policy.actor.encoder_is_shared = _original_encoder_is_shared

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
