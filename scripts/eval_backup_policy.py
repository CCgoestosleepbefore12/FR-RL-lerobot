"""
Backup Policy 评估脚本

统计指标：存活率（存活>100步的比例）、平均存活步数、平均最近距离

用法:
  python scripts/eval_backup_policy.py --checkpoint outputs/.../pretrained_model --n_episodes 50
  python scripts/eval_backup_policy.py --checkpoint outputs/.../pretrained_model --n_episodes 10 --render
"""

import argparse
import numpy as np
import torch
import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
import gymnasium as gym
from frrl.configs.policies import PreTrainedConfig


def evaluate(checkpoint_path: str, n_episodes: int, render: bool, survival_threshold: int = 100, env_task: str = "PandaBackupPolicyBase-v0"):
    # 加载策略
    print(f"加载策略: {checkpoint_path}", flush=True)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.pretrained_path = checkpoint_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    from frrl.policies.sac.modeling_sac import SACPolicy
    policy = SACPolicy.from_pretrained(checkpoint_path, config=cfg)
    policy.eval()
    policy.to(device)
    print(f"策略加载完成, device={device}", flush=True)

    # 创建环境
    env = gym.make(f"gym_frrl/{env_task}", image_obs=False)
    print(f"环境: {env_task}", flush=True)

    # 可视化
    viewer = None
    if render:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.unwrapped._model, env.unwrapped._data)

    # 观测处理
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

    # 评估
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        env_processor.reset()
        transition = env_processor(create_transition(obs))

        done = False
        step = 0
        max_steps = 200
        min_hand_dist = float('inf')

        while not done and step < max_steps:
            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.input_features
            }
            with torch.no_grad():
                action = policy.select_action(batch=observation)

            action_np = action.squeeze(0).cpu().numpy()

            # Backup 环境期望 6D 动作，但经过 processor 后可能维度不同
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step += 1

            if 'min_hand_dist' in info:
                min_hand_dist = min(min_hand_dist, info['min_hand_dist'])

            if render and viewer is not None:
                viewer.sync()
                import time
                time.sleep(0.05)

            if not done:
                transition = env_processor(create_transition(obs))

        violation = info.get('violation_type', 'survived' if step >= max_steps else 'unknown')
        survived = step >= survival_threshold
        results.append({
            'steps': step,
            'survived': survived,
            'min_hand_dist': min_hand_dist,
            'violation': violation,
            'displacement': info.get('displacement', 0),
        })

        status = "ALIVE" if step >= max_steps else f"DEAD({violation})"
        print(f"  EP {ep+1:3d}/{n_episodes}: steps={step:3d} {status} "
              f"min_dist={min_hand_dist:.3f} disp={info.get('displacement',0):.3f}", flush=True)

    if render and viewer is not None:
        viewer.close()
    env.close()

    # 统计
    steps_list = [r['steps'] for r in results]
    survival_count = sum(1 for r in results if r['survived'])
    full_survival = sum(1 for r in results if r['steps'] >= max_steps)
    dist_list = [r['min_hand_dist'] for r in results if r['min_hand_dist'] < float('inf')]

    # 终止原因统计
    violations = {}
    for r in results:
        v = r['violation']
        violations[v] = violations.get(v, 0) + 1

    print(f"\n{'='*60}", flush=True)
    print(f"Backup Policy 评估结果", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"总 episodes: {n_episodes}", flush=True)
    print(f"存活率 (>={survival_threshold}步): {survival_count}/{n_episodes} ({survival_count/n_episodes*100:.1f}%)", flush=True)
    print(f"满存活 (200步): {full_survival}/{n_episodes} ({full_survival/n_episodes*100:.1f}%)", flush=True)
    print(f"平均存活步数: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}", flush=True)
    if dist_list:
        print(f"平均最近距离: {np.mean(dist_list):.3f}m", flush=True)
    print(f"\n终止原因:", flush=True)
    for v, count in sorted(violations.items(), key=lambda x: -x[1]):
        print(f"  {v}: {count} ({count/n_episodes*100:.1f}%)", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--survival_threshold", type=int, default=100)
    parser.add_argument("--env_task", default="PandaBackupPolicyBase-v0")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.n_episodes, args.render, args.survival_threshold, args.env_task)
