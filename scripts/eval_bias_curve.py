"""
偏差-成功率曲线评估脚本

测试策略在不同固定偏差值下的成功率，输出完整的对比表。

用法:
  python scripts/eval_bias_curve.py --checkpoint outputs/train/.../pretrained_model --n_episodes 50
"""

import argparse
import numpy as np
import torch
import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
import gymnasium as gym
from frrl.configs.policies import PreTrainedConfig
from frrl.envs.wrappers import wrap_env
from frrl.fault_injection import EncoderBiasConfig
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


def eval_single_bias(policy, cfg, bias_value, n_episodes, device):
    """评估策略在某个固定偏差下的表现"""
    # 创建带偏差的环境
    bias_cfg = None
    if bias_value > 0:
        bias_cfg = EncoderBiasConfig(
            enable=True, error_probability=1.0,
            target_joints=[3], bias_mode="fixed",
            fixed_bias_value=bias_value,
        )

    base_env = gym.make("gym_frrl/PandaPickCubeBase-v0",
                        image_obs=True, render_mode="rgb_array",
                        encoder_bias_config=bias_cfg)
    env = wrap_env(base_env, use_viewer=False, use_gripper=True,
                   use_inputs_control=False, reset_delay_seconds=0.1)

    env_processor = DataProcessorPipeline(
        steps=[Numpy2TorchActionProcessorStep(), VanillaObservationProcessorStep(),
               AddBatchDimensionProcessorStep(), DeviceProcessorStep(device=device)],
        to_transition=identity_transition, to_output=identity_transition,
    )

    successes = []
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        env_processor.reset()
        transition = env_processor(create_transition(obs))

        done = False
        step = 0
        ep_reward = 0.0
        max_steps = 200
        while not done and step < max_steps:
            observation = {k: v for k, v in transition[TransitionKey.OBSERVATION].items()
                          if k in cfg.input_features}
            with torch.no_grad():
                action = policy.select_action(batch=observation)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            if not done:
                transition = env_processor(create_transition(obs))

        successes.append(info.get("succeed", ep_reward > 0.5))
        lengths.append(step)

    env.close()
    return np.mean(successes) * 100, np.mean(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--bias_values", type=str, default="0,0.05,0.1,0.15,0.2,0.25,0.3",
                        help="逗号分隔的偏差值列表")
    args = parser.parse_args()

    # 加载策略
    print(f"加载策略: {args.checkpoint}", flush=True)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    from frrl.policies.sac.modeling_sac import SACPolicy
    policy = SACPolicy.from_pretrained(args.checkpoint, config=cfg)
    policy.eval().to(device)
    print(f"策略加载完成, device={device}\n", flush=True)

    bias_values = [float(v) for v in args.bias_values.split(",")]

    # 逐个评估
    results = []
    for bias in bias_values:
        print(f"评估 bias={bias:.2f}rad ({args.n_episodes} episodes)...", end=" ", flush=True)
        success_rate, avg_steps = eval_single_bias(policy, cfg, bias, args.n_episodes, device)
        results.append((bias, success_rate, avg_steps))
        print(f"成功率={success_rate:.1f}%  平均步数={avg_steps:.1f}", flush=True)

    # 输出汇总表
    print(f"\n{'='*55}", flush=True)
    print(f"{'偏差(rad)':>10} | {'成功率':>8} | {'平均步数':>8}", flush=True)
    print(f"{'-'*55}", flush=True)
    for bias, sr, steps in results:
        bar = "█" * int(sr / 5)
        print(f"{bias:>10.2f} | {sr:>7.1f}% | {steps:>8.1f}  {bar}", flush=True)
    print(f"{'='*55}", flush=True)


if __name__ == "__main__":
    main()
