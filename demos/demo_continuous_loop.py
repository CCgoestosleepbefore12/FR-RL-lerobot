#!/usr/bin/env python3
"""
连续循环执行轨迹演示

在同一个仿真环境中连续执行多次pick-and-place任务：
- 任务完成后自动重置
- 立即开始下一次执行
- Viewer保持打开
- 按Ctrl+C停止
"""

import sys
from pathlib import Path
# 添加 mujoco_sim/ 到 sys.path（当前文件在 demos/ 目录）
# 使用frrl包

import signal
import numpy as np
import time
import yaml
from pathlib import Path
from frrl.envs.panda_pick_place_env import PandaPickPlaceEnv
from frrl.trajectory_executor import Waypoint, TrajectoryInterpolator, TrajectoryExecutor
from frrl.fault_injection import EncoderBiasConfig

def _sigint_handler(signum, frame):
    """处理 SIGINT (Ctrl+C)，确保即使在 C 扩展内部阻塞也能响应。"""
    raise KeyboardInterrupt


def create_trajectory(block_pos: np.ndarray, plate_pos: np.ndarray) -> list:
    """创建pick-and-place轨迹。"""
    standard_quat = np.array([1.0, 0.0, 0.0, 0.0])

    waypoints = []

    # 起点
    waypoints.append(Waypoint(
        position=np.array([0.49, 0.0, 0.08]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=0.0,
        name="起点"
    ))

    # Block上方
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], 0.15]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=2.0,
        name="Block上方"
    ))

    # 下降
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.003]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.5,
        name="下降"
    ))

    # 预抓取
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.003]),
        orientation=standard_quat.copy(),
        gripper=0.3,
        duration=1.0,
        name="预抓取"
    ))

    # 完全抓取
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.003]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.5,
        name="完全抓取"
    ))

    # 抬升
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], 0.18]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.0,
        name="抬升"
    ))

    # 移动到Plate上方
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.18]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.5,
        name="Plate上方"
    ))

    # 下降到Plate
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.035]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.0,
        name="下降到Plate"
    ))

    # 释放
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.035]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.5,
        name="释放"
    ))

    # 等待稳定
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.035]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.0,
        name="稳定"
    ))

    # 远离
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.15]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.5,
        name="远离"
    ))

    return waypoints


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="连续循环执行轨迹演示")
    parser.add_argument('--episodes', type=int, default=None,
                       help='执行次数（默认无限循环）')
    parser.add_argument('--pause', type=float, default=2.0,
                       help='每次执行之间的暂停时间（秒，默认2秒）')
    parser.add_argument('--no-render', action='store_true',
                       help='不显示viewer')

    # 编码器偏差参数
    parser.add_argument('--enable-encoder-bias', action='store_true',
                       help='启用编码器偏差注入')
    parser.add_argument('--bias-config', type=str,
                       default='configs/encoder_bias_joint4_random.yaml',
                       help='编码器偏差配置文件路径')

    args = parser.parse_args()

    # 注册信号处理器（必须在 viewer 创建之前注册，防止被 GLFW 覆盖）
    signal.signal(signal.SIGINT, _sigint_handler)

    print()
    print("=" * 70)
    print("连续循环执行轨迹演示")
    print("=" * 70)
    print()

    if args.episodes:
        print(f"将连续执行 {args.episodes} 次")
    else:
        print("将无限循环执行（按 Ctrl+C 停止）")

    print(f"每次执行之间暂停: {args.pause}秒")
    print()
    print("每次执行包括:")
    print("  1. 执行完整轨迹")
    print("  2. 显示结果")
    print("  3. 暂停观察")
    print("  4. 重置环境")
    print("  5. 开始下一次")
    print()
    print("=" * 70)
    print()

    # 加载编码器偏差配置
    bias_config = None
    if args.enable_encoder_bias:
        config_path = Path(__file__).parent.parent / args.bias_config
        print(f"加载编码器偏差配置: {config_path}")
        try:
            bias_config = EncoderBiasConfig.from_yaml(str(config_path))
            print("配置加载成功")
        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            print("使用默认配置")
            bias_config = EncoderBiasConfig()
        print()

    # 创建环境
    env = PandaPickPlaceEnv(
        control_dt=0.1,
        physics_dt=0.002,
        render_mode="human" if not args.no_render else None,
        encoder_bias_config=bias_config,
    )

    # 统计变量
    episode = 0
    success_count = 0

    try:
        while args.episodes is None or episode < args.episodes:
            episode += 1

            print()
            print(f"=== Episode {episode} ===")

            # 重置环境
            obs, info = env.reset()

            # 让物理稳定
            for _ in range(50):
                action = np.zeros(4, dtype=np.float32)
                obs, _, _, _, _ = env.step(action)
                if not args.no_render:
                    env.render()

            block_pos = obs['state'][4:7]
            plate_pos = obs['state'][7:10]

            # 创建并执行轨迹
            waypoints = create_trajectory(block_pos, plate_pos)
            interpolator = TrajectoryInterpolator(waypoints, control_dt=env.control_dt)
            executor = TrajectoryExecutor(env, interpolator)
            result = executor.execute(render=not args.no_render, verbose=False)

            if result['success']:
                success_count += 1

            print(f"  成功: {'YES' if result['success'] else 'NO'} | "
                  f"累计: {success_count}/{episode} ({success_count/episode*100:.0f}%)")

            # 暂停观察
            if not args.no_render:
                pause_steps = int(args.pause / env.control_dt)
                for _ in range(pause_steps):
                    env.step(np.zeros(4, dtype=np.float32))
                    env.render()

    except KeyboardInterrupt:
        print("\n用户中断")

    # 最终统计
    print(f"\n总计: {success_count}/{episode} 成功 ({success_count/episode*100:.0f}%)" if episode > 0 else "")

    if env.bias_injector:
        env.bias_injector.print_stats()

    env.close()

    # 强制退出，防止GLFW残留线程阻塞
    sys.exit(0)


if __name__ == "__main__":
    main()
