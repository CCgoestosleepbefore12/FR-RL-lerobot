#!/usr/bin/env python3
"""
重复执行轨迹演示

可以连续执行多次pick-and-place任务，观察成功率和稳定性。
"""

import sys
from pathlib import Path
# 使用frrl包

import numpy as np
from frrl.envs.panda_pick_place_env import PandaPickPlaceEnv
from frrl.trajectory_executor import Waypoint, TrajectoryInterpolator, TrajectoryExecutor


def create_stable_trajectory(
    block_pos: np.ndarray,
    plate_pos: np.ndarray,
) -> list:
    """创建稳定的pick-and-place轨迹。"""
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
        duration=2.5,
        name="Block上方"
    ))

    # 下降到Block
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.003]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=2.0,
        name="下降到Block"
    ))

    # 预抓取
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.003]),
        orientation=standard_quat.copy(),
        gripper=0.3,
        duration=1.5,
        name="预抓取"
    ))

    # 完全抓取
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.003]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=3.0,
        name="完全抓取"
    ))

    # 抬升
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], 0.20]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.5,
        name="抬升"
    ))

    # 移动到Plate上方
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.20]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=3.0,
        name="Plate上方"
    ))

    # 下降到Plate
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.035]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.5,
        name="下降到Plate"
    ))

    # 稳定后释放
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.035]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=2.0,
        name="释放"
    ))

    # 等待稳定
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.035]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.5,
        name="等待稳定"
    ))

    # 上移
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.15]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=2.0,
        name="远离"
    ))

    return waypoints


def run_single_episode(env, waypoints, episode_num, render=True, verbose=False):
    """执行一次完整的pick-and-place任务。"""

    if verbose:
        print()
        print("=" * 70)
        print(f"第 {episode_num} 次执行")
        print("=" * 70)
        print()

    # 重置环境
    obs, info = env.reset()
    block_pos = obs['state'][4:7]
    plate_pos = obs['state'][7:10]

    # 更新轨迹（Block位置可能随机）
    waypoints_updated = create_stable_trajectory(block_pos, plate_pos)

    # 创建插值器和执行器
    interpolator = TrajectoryInterpolator(waypoints_updated, control_dt=env.control_dt)
    executor = TrajectoryExecutor(env, interpolator)

    # 执行
    result = executor.execute(render=render, verbose=verbose)

    return result


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="重复执行轨迹演示")
    parser.add_argument('--episodes', type=int, default=3,
                       help='执行次数（默认3次）')
    parser.add_argument('--no-render', action='store_true',
                       help='不显示viewer')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("重复执行轨迹演示")
    print("=" * 70)
    print()
    print(f"将执行 {args.episodes} 次完整的pick-and-place任务")
    print()
    print("每次任务包括:")
    print("  1. 重置环境")
    print("  2. 执行完整轨迹")
    print("  3. 统计结果")
    print()
    print("=" * 70)
    print()

    # 创建环境
    env = PandaPickPlaceEnv(
        control_dt=0.1,
        physics_dt=0.002,
        max_episode_steps=800,
        delta_action_scale=0.015,  # 1.5cm per action
        render_mode="human" if not args.no_render else None,
    )

    # 初始化统计
    results = []
    success_count = 0
    grasped_count = 0
    total_rewards = []

    # 获取初始轨迹模板
    obs, _ = env.reset()
    block_pos = obs['state'][4:7]
    plate_pos = obs['state'][7:10]
    waypoints = create_stable_trajectory(block_pos, plate_pos)

    # 执行多次
    for episode in range(1, args.episodes + 1):
        result = run_single_episode(
            env, waypoints, episode,
            render=not args.no_render,
            verbose=args.verbose
        )

        results.append(result)
        if result['success']:
            success_count += 1
        if result['grasped']:
            grasped_count += 1
        total_rewards.append(result['total_reward'])

        # 显示简要结果
        print()
        print(f"第 {episode} 次结果:")
        print(f"  成功: {'✅' if result['success'] else '❌'}")
        print(f"  抓取: {'✅' if result['grasped'] else '❌'}")
        print(f"  奖励: {result['total_reward']:.2f}")
        print(f"  Block高度: {result['final_block_pos'][2]*100:.1f}cm")
        print()

        # 每次之间暂停一下
        if episode < args.episodes:
            print(f"准备第 {episode + 1} 次执行...")
            print()

    # 保持viewer打开
    if not args.no_render:
        print()
        print("=" * 70)
        print("所有任务执行完成，Viewer保持打开")
        print("=" * 70)
        print()
        print("提示:")
        print("  - 观察最终状态")
        print("  - 按 Ctrl+C 退出")
        print()

        try:
            while True:
                action = np.zeros(4, dtype=np.float32)
                env.step(action)
                env.render()
        except KeyboardInterrupt:
            print()
            print("用户退出")

    env.close()

    # 最终统计
    print()
    print("=" * 70)
    print("执行统计")
    print("=" * 70)
    print()
    print(f"总执行次数: {args.episodes}")
    print(f"成功次数: {success_count}")
    print(f"成功率: {success_count/args.episodes*100:.1f}%")
    print()
    print(f"抓取成功次数: {grasped_count}")
    print(f"抓取成功率: {grasped_count/args.episodes*100:.1f}%")
    print()
    if total_rewards:
        print(f"平均奖励: {np.mean(total_rewards):.2f}")
        print(f"最高奖励: {np.max(total_rewards):.2f}")
        print(f"最低奖励: {np.min(total_rewards):.2f}")
    else:
        print("未执行任何episode，无法计算奖励统计")
    print()

    # 详细结果
    if args.verbose:
        print("详细结果:")
        for i, result in enumerate(results, 1):
            print(f"  第{i}次: 成功={result['success']}, "
                  f"抓取={result['grasped']}, "
                  f"奖励={result['total_reward']:.2f}")
        print()

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
