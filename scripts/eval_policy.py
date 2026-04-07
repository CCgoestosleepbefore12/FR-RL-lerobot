"""
策略评估脚本 — 加载checkpoint，在指定环境中跑N个episode，统计成功率。

用法:
  # 无偏差评估
  python scripts/eval_policy.py --checkpoint outputs/train/.../pretrained_model --env_task PandaPickCubeBase-v0 --n_episodes 50

  # 有偏差评估
  python scripts/eval_policy.py --checkpoint outputs/train/.../pretrained_model --env_task PandaPickCubeBiasJ4Fixed02-v0 --n_episodes 50

  # 带可视化
  python scripts/eval_policy.py --checkpoint outputs/train/.../pretrained_model --env_task PandaPickCubeBase-v0 --n_episodes 10 --render
"""

import argparse
import numpy as np
import torch
import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
import gymnasium as gym
from frrl.configs.policies import PreTrainedConfig
from frrl.envs.wrappers import wrap_env


def evaluate(checkpoint_path: str, env_task: str, n_episodes: int, render: bool):
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
    # 用Base-v0创建裸环境，然后手动加EEActionWrapper（不加键盘干预）
    # 这样策略输出4D，EEActionWrapper转成7D给env
    render_mode = "human" if render else "rgb_array"
    base_env = gym.make(f"gym_frrl/{env_task}", image_obs=True, render_mode=render_mode)

    env = wrap_env(
        base_env,
        use_viewer=render,
        use_gripper=True,
        use_inputs_control=False,  # 不启用键盘干预
        gripper_penalty=-0.05,
        reset_delay_seconds=0.5,
    )

    print(f"环境: gym_frrl/{env_task}", flush=True)
    print(f"Action space: {env.action_space}", flush=True)
    print(f"评估 {n_episodes} episodes...", flush=True)

    # 观测处理pipeline（和训练时一致）
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

    # 评估循环
    successes = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        env_processor.reset()
        transition = env_processor(create_transition(obs))

        done = False
        step = 0
        ep_reward = 0.0
        max_steps = 200
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
            ep_reward += reward
            step += 1

            if not done:
                transition = env_processor(create_transition(obs))

        success = info.get("succeed", ep_reward > 0.5)
        successes.append(success)
        episode_lengths.append(step)

        status = "✓" if success else "✗"
        print(f"  Episode {ep+1:3d}/{n_episodes}: {status} (steps={step}, reward={ep_reward:.1f})", flush=True)

    env.close()

    # 统计
    success_rate = np.mean(successes) * 100
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"\n{'='*50}", flush=True)
    print(f"环境: {env_task}", flush=True)
    print(f"总episodes: {n_episodes}", flush=True)
    print(f"成功率: {success_rate:.1f}% ({sum(successes)}/{n_episodes})", flush=True)
    print(f"平均步数: {avg_length:.1f} ± {std_length:.1f}", flush=True)
    print(f"{'='*50}", flush=True)

    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint路径 (pretrained_model目录)")
    parser.add_argument("--env_task", default="PandaPickCubeBase-v0", help="环境ID（用Base-v0）")
    parser.add_argument("--n_episodes", type=int, default=50, help="评估episode数")
    parser.add_argument("--render", action="store_true", help="是否显示MuJoCo窗口")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.env_task, args.n_episodes, args.render)
