"""5 分钟诊断：actor 输出的 action 是否真的随 image 变化（视觉条件反射建立否）。

测试方法：
  - 固定 proprio (agent_pos)，仅替换 pixels.front + pixels.wrist 为 demo 不同时刻的图像
  - 观察 actor select_action 输出 action 的方差
  - 对照：固定 image，仅改 proprio，看 action 方差

判读：
  Δaction (image-induced) >> Δaction (proprio-induced) → vision 主导（不太可能）
  Δaction (image-induced) ~ Δaction (proprio-induced) → vision 接通正常
  Δaction (image-induced) ≈ 0 → **vision 没接通**，actor 仅依赖 proprio
  Δaction (image-induced) ≈ 0 且 Δaction (proprio-induced) ≈ 0 → actor 学到 const action

用法：
  python scripts/tools/diagnose_vision_conditioning.py \
      --ckpt checkpoints/wipe_pretrain_2k_20260427_161127/checkpoints/last/pretrained_model \
      --demos data/task_policy_demos/*.pkl
"""
import argparse
import glob
import pickle as pkl
from pathlib import Path

import numpy as np
import torch

import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
from frrl.configs.policies import PreTrainedConfig
from frrl.policies.sac.modeling_sac import SACPolicy


def make_batch(obs_dict, device):
    """把 hil-serl pickle 格式 obs 转成 SAC 期望的 batch dict。"""
    agent_pos = np.asarray(obs_dict["agent_pos"], dtype=np.float32)
    front = np.asarray(obs_dict["pixels"]["front"])  # (H, W, 3) uint8
    wrist = np.asarray(obs_dict["pixels"]["wrist"])
    return {
        "observation.state": torch.from_numpy(agent_pos).unsqueeze(0).to(device),
        "observation.images.front": (
            torch.from_numpy(front).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        ),
        "observation.images.wrist": (
            torch.from_numpy(wrist).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .../pretrained_model")
    ap.add_argument("--demos", required=True, help="pickle glob")
    ap.add_argument("--n-samples", type=int, default=12, help="number of test obs to sample")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Load policy
    print(f"[load] {args.ckpt}")
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = args.device
    policy = SACPolicy.from_pretrained(args.ckpt, config=cfg)
    policy.eval().to(args.device)

    # Load demos
    paths = sorted(glob.glob(args.demos))
    if not paths:
        raise SystemExit(f"no demos matched {args.demos}")
    transitions = []
    for p in paths:
        with open(p, "rb") as f:
            transitions.extend(pkl.load(f))
    print(f"[load] {len(transitions)} transitions from {len(paths)} pickle(s)")

    # 均匀采样 n 个 transition 作为不同时刻 / episode 的代表
    n = min(args.n_samples, len(transitions))
    idx_grid = np.linspace(0, len(transitions) - 1, n).astype(int)
    samples = [transitions[i] for i in idx_grid]
    print(f"[sample] {n} obs at indices {idx_grid.tolist()[:6]}...")

    # ============================================================
    # Test 1: 固定 proprio (用 sample 0 的 agent_pos)，换不同 image
    # ============================================================
    print("\n=== Test 1: 固定 proprio，换 image ===")
    base_proprio = samples[0]["observations"]["agent_pos"]
    actions_image_swap = []
    for i, s in enumerate(samples):
        test_obs = {
            "agent_pos": base_proprio,
            "pixels": s["observations"]["pixels"],
        }
        batch = make_batch(test_obs, args.device)
        action = policy.select_action(batch).squeeze().cpu().numpy()
        actions_image_swap.append(action)
        if i < 5:
            print(f"  [img idx={idx_grid[i]:>5d}] action[:6] = {np.round(action[:6], 4)}")
    actions_image_swap = np.stack(actions_image_swap)
    std_per_dim_img = actions_image_swap.std(axis=0)
    print(f"\n  Δaction(image-only) per-dim std: {np.round(std_per_dim_img[:6], 4)}")
    print(f"  Δaction(image-only) overall std: {std_per_dim_img[:6].mean():.5f}")

    # ============================================================
    # Test 2: 固定 image (用 sample 0 的)，换不同 proprio
    # ============================================================
    print("\n=== Test 2: 固定 image，换 proprio ===")
    base_pixels = samples[0]["observations"]["pixels"]
    actions_proprio_swap = []
    for i, s in enumerate(samples):
        test_obs = {
            "agent_pos": s["observations"]["agent_pos"],
            "pixels": base_pixels,
        }
        batch = make_batch(test_obs, args.device)
        action = policy.select_action(batch).squeeze().cpu().numpy()
        actions_proprio_swap.append(action)
        if i < 5:
            print(f"  [pro idx={idx_grid[i]:>5d}] action[:6] = {np.round(action[:6], 4)}")
    actions_proprio_swap = np.stack(actions_proprio_swap)
    std_per_dim_pro = actions_proprio_swap.std(axis=0)
    print(f"\n  Δaction(proprio-only) per-dim std: {np.round(std_per_dim_pro[:6], 4)}")
    print(f"  Δaction(proprio-only) overall std: {std_per_dim_pro[:6].mean():.5f}")

    # ============================================================
    # Test 3: 都用 demo 原 obs（image + proprio 都对应同一时刻）
    # ============================================================
    print("\n=== Test 3: 完整 demo obs（image + proprio 同时刻）===")
    actions_full = []
    for i, s in enumerate(samples):
        batch = make_batch(s["observations"], args.device)
        action = policy.select_action(batch).squeeze().cpu().numpy()
        actions_full.append(action)
        if i < 5:
            print(f"  [demo idx={idx_grid[i]:>5d}] action[:6] = {np.round(action[:6], 4)} "
                  f"vs demo_action[:6] = {np.round(s['actions'][:6], 4)}")
    actions_full = np.stack(actions_full)
    std_per_dim_full = actions_full.std(axis=0)
    print(f"\n  Δaction(full obs) per-dim std: {np.round(std_per_dim_full[:6], 4)}")
    print(f"  Δaction(full obs) overall std: {std_per_dim_full[:6].mean():.5f}")

    # ============================================================
    # Test 0 (baseline): 同一 obs 跑 N 次 sample，看 stochastic policy 噪声
    # ============================================================
    print("\n=== Test 0 (baseline): 同 obs N 次 sample，stochastic 噪声 ===")
    base_batch = make_batch(samples[0]["observations"], args.device)
    n_sample = 30
    sample_actions = []
    for _ in range(n_sample):
        action = policy.select_action(base_batch).squeeze().cpu().numpy()
        sample_actions.append(action)
    sample_actions = np.stack(sample_actions)
    std_per_dim_noise = sample_actions.std(axis=0)
    print(f"  per-dim std (固定 obs, {n_sample} samples): {np.round(std_per_dim_noise[:6], 4)}")
    print(f"  overall: {std_per_dim_noise[:6].mean():.5f}")
    print(f"  mean: {np.round(sample_actions.mean(axis=0)[:6], 4)}")

    # ============================================================
    # 判读
    # ============================================================
    print("\n" + "=" * 60)
    print("判读")
    print("=" * 60)

    img_std = std_per_dim_img[:6].mean()
    pro_std = std_per_dim_pro[:6].mean()
    full_std = std_per_dim_full[:6].mean()
    noise_std = std_per_dim_noise[:6].mean()

    print(f"  Δa(stochastic noise) = {noise_std:.5f}  ← 噪声基准")
    print(f"  Δa(image-only)       = {img_std:.5f}  (vs noise: {img_std/max(noise_std,1e-6):.2f}x)")
    print(f"  Δa(proprio-only)     = {pro_std:.5f}  (vs noise: {pro_std/max(noise_std,1e-6):.2f}x)")
    print(f"  Δa(full obs)         = {full_std:.5f}  (vs noise: {full_std/max(noise_std,1e-6):.2f}x)")
    print()

    # 判读规则：image / proprio swap 的 std 必须显著超过 noise std 才算"真信号"
    img_signal = img_std > 1.5 * noise_std
    pro_signal = pro_std > 1.5 * noise_std

    if not img_signal and not pro_signal:
        print("  ❌ Image 和 proprio 都没真信号 — actor 输出基本是 stochastic 噪声")
        print("     说明 actor head 没学到任何 obs-conditioned policy")
    elif not img_signal:
        print("  ❌ Vision 没接通（image-swap 跟 noise 差不多）")
        print("     proprio 有信号 → critic 走 proprio (含 tcp_true) 捷径，vision 死")
    elif not pro_signal:
        print("  ⚠️ Proprio 没信号但 vision 有 — 不寻常，可能 proprio encoder 有问题")
    elif img_std < 0.005 and pro_std < 0.005 and full_std < 0.005:
        print("  ❌ Actor 输出几乎是 const → critic 梯度无效，actor 学到 demo mean")
        print("     (action[0] mean ≈ +0.126 in demo per agent B)")
    elif img_std < 0.005:
        print("  ❌ Vision 完全没接通：actor 不响应 image 变化")
        print("     可能原因：DINOv3 patch 区分度差 / pretrain 时间不够 / vision encoder 梯度路径堵")
    elif img_std < 0.2 * pro_std:
        print("  ⚠️ Vision 接通但弱：actor 主要靠 proprio (含 tcp_true cheat channel)")
        print("     online HIL 阶段会 vision 学得很慢")
    elif img_std > pro_std:
        print("  ✅ Vision 主导 action：reasonable for visual task")
    else:
        print("  ✅ Vision + proprio 大致平衡：训练健康")


if __name__ == "__main__":
    main()
