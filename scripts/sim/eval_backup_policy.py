"""
Backup Policy 评估脚本

评估指标：
  - 存活率：连续存活步数超过阈值的 episode 比例
  - 碰撞率：各终止原因的分布
  - 最近距离：episode 中 TCP 与障碍物的最小距离
  - 累计奖励：反映位移 + 动作平滑的综合表现
  - 运动模式分布：4 种运动模式下的存活率对比

用法:
  # S1 评估
  python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model

  # S2 评估
  python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model --env_task PandaBackupPolicyS2-v0

  # 可视化
  python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model --render --n_episodes 10

  # 无 DR 对比（测试策略在精确观测下的表现）
  python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model --env_task PandaBackupPolicyS1NoDR-v0

  # V2 防作弊几何评估
  python scripts/sim/eval_backup_policy.py --checkpoint checkpoints/backup_policy_s1_v2_newgeom_145k --env_task PandaBackupPolicyS1V2-v0

  # V3 全臂避障评估（5 球 + obstacle r=0.10 + proximity reward）
  python scripts/sim/eval_backup_policy.py --checkpoint outputs/.../pretrained_model --env_task PandaBackupPolicyS1V3-v0
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import gymnasium as gym
import mujoco.viewer

import frrl.envs  # noqa: F401 — 触发 gym 注册
import frrl.policies.sac.configuration_sac  # noqa: F401 — 触发策略注册
from frrl.configs.policies import PreTrainedConfig
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.processor import (
    Numpy2TorchActionProcessorStep,
    VanillaObservationProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    DataProcessorPipeline,
    create_transition,
)
from frrl.processor.converters import identity_transition
from frrl.processor.core import TransitionKey


def evaluate(checkpoint_path: str, n_episodes: int, render: bool,
             survival_threshold: int, max_steps: int, env_task: str):
    # ── 加载策略 ──
    print(f"加载策略: {checkpoint_path}", flush=True)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.pretrained_path = checkpoint_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    policy = SACPolicy.from_pretrained(checkpoint_path, config=cfg)
    policy.eval()
    policy.to(device)
    print(f"策略加载完成, device={device}", flush=True)

    # ── 创建环境 ──
    env = gym.make(f"gym_frrl/{env_task}", image_obs=False)
    print(f"环境: {env_task}", flush=True)

    # ── 可视化 ──
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(env.unwrapped._model, env.unwrapped._data)

    env_processor = DataProcessorPipeline(
        steps=[
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ],
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    # ── 评估 ──
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        env_processor.reset()
        transition = env_processor(create_transition(obs))

        step = 0
        episode_reward = 0.0
        min_hand_dist = float("inf")
        motion_mode = "unknown"
        done = False

        while not done and step < max_steps:
            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.input_features
            }
            with torch.no_grad():
                action = policy.select_action(batch=observation)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step += 1
            episode_reward += reward

            if "min_hand_dist" in info:
                min_hand_dist = min(min_hand_dist, info["min_hand_dist"])
            if "motion_mode" in info:
                motion_mode = info["motion_mode"]

            if render and viewer is not None:
                viewer.sync()
                time.sleep(0.05)

            if not done:
                transition = env_processor(create_transition(obs))

        violation = info.get("violation_type", None)
        if violation is None:
            violation = "survived" if step >= max_steps else "truncated"

        survived = step >= survival_threshold
        final_displacement = info.get("displacement", 0.0)
        results.append({
            "steps": step,
            "survived": survived,
            "min_hand_dist": min_hand_dist,
            "displacement": final_displacement,
            "violation": violation,
            "reward": episode_reward,
            "motion_mode": motion_mode,
        })

        status = "ALIVE" if step >= max_steps else (
            f"TRUNC" if violation == "truncated" else f"DEAD({violation})"
        )
        dist_str = f"{min_hand_dist:.3f}" if min_hand_dist < float("inf") else "N/A"
        print(f"  EP {ep+1:3d}/{n_episodes}: steps={step:3d} {status:20s} "
              f"reward={episode_reward:+.3f} min_dist={dist_str} "
              f"disp={final_displacement:.3f} mode={motion_mode}",
              flush=True)

    if render and viewer is not None:
        viewer.close()
    env.close()

    # ── 汇总统计 ──
    steps_arr = np.array([r["steps"] for r in results])
    reward_arr = np.array([r["reward"] for r in results])
    survival_count = sum(1 for r in results if r["survived"])
    full_survival = sum(1 for r in results if r["steps"] >= max_steps)
    dist_list = [r["min_hand_dist"] for r in results if r["min_hand_dist"] < float("inf")]
    disp_list = [r["displacement"] for r in results]

    # 终止原因
    violations = defaultdict(int)
    for r in results:
        violations[r["violation"]] += 1

    # 运动模式统计
    mode_stats = defaultdict(lambda: {"total": 0, "survived": 0, "reward": []})
    for r in results:
        m = r["motion_mode"]
        mode_stats[m]["total"] += 1
        if r["survived"]:
            mode_stats[m]["survived"] += 1
        mode_stats[m]["reward"].append(r["reward"])

    print(f"\n{'='*65}", flush=True)
    print(f" Backup Policy 评估结果", flush=True)
    print(f"{'='*65}", flush=True)
    print(f" 环境: {env_task}", flush=True)
    print(f" 总 episodes: {n_episodes}", flush=True)
    print(f" 存活率 (>={survival_threshold}步): "
          f"{survival_count}/{n_episodes} ({survival_count/n_episodes*100:.1f}%)", flush=True)
    print(f" 满存活 ({max_steps}步): "
          f"{full_survival}/{n_episodes} ({full_survival/n_episodes*100:.1f}%)", flush=True)
    print(f" 平均存活步数: {np.mean(steps_arr):.1f} ± {np.std(steps_arr):.1f}", flush=True)
    print(f" 平均累计奖励: {np.mean(reward_arr):.3f} ± {np.std(reward_arr):.3f}", flush=True)
    if dist_list:
        print(f" 平均最近距离: {np.mean(dist_list):.3f}m", flush=True)
    if disp_list:
        print(f" 平均终止位移: {np.mean(disp_list):.3f}m ± {np.std(disp_list):.3f}", flush=True)

    print(f"\n 终止原因:", flush=True)
    for v, count in sorted(violations.items(), key=lambda x: -x[1]):
        print(f"   {v:20s}: {count:3d} ({count/n_episodes*100:.1f}%)", flush=True)

    print(f"\n 运动模式分析:", flush=True)
    print(f"   {'模式':<12s} {'数量':>4s} {'存活率':>8s} {'平均奖励':>10s}", flush=True)
    print(f"   {'-'*38}", flush=True)
    for mode in sorted(mode_stats.keys()):
        s = mode_stats[mode]
        rate = s["survived"] / s["total"] * 100 if s["total"] > 0 else 0
        avg_r = np.mean(s["reward"]) if s["reward"] else 0
        print(f"   {mode:<12s} {s['total']:4d} {rate:7.1f}% {avg_r:+10.3f}", flush=True)

    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup Policy 评估")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--n_episodes", type=int, default=50, help="评估 episode 数")
    parser.add_argument("--render", action="store_true", help="MuJoCo 可视化")
    parser.add_argument("--survival_threshold", type=int, default=20,
                        help="存活判定阈值（步数，默认 20 = 完整 episode，与训练 max_episode_steps 对齐）")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="每 episode 最大步数（默认 20，与训练对齐）")
    parser.add_argument("--env_task", default="PandaBackupPolicyS1-v0",
                        choices=[
                            "PandaBackupPolicyS1-v0",
                            "PandaBackupPolicyS1Relaxed-v0",
                            "PandaBackupPolicyS1Combo-v0",
                            "PandaBackupPolicyS1V2-v0",
                            "PandaBackupPolicyS1V3-v0",
                            "PandaBackupPolicyS2-v0",
                            "PandaBackupPolicyS1NoDR-v0",
                            "PandaBackupPolicyS1BiasJ1-v0",
                            "PandaBackupPolicyS2BiasJ1-v0",
                        ],
                        help="评估环境 ID")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.n_episodes, args.render,
             args.survival_threshold, args.max_steps, args.env_task)
