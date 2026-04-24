#!/usr/bin/env python3
"""
优化的轨迹Pick-and-Place演示

优化点：
1. 增加抓取阶段的停留时间
2. 更精确的下降高度
3. 更慢的运动速度，确保稳定
"""

import sys
from pathlib import Path
# 使用frrl包

import numpy as np
from frrl.envs.sim.panda_pick_place_env import PandaPickPlaceEnv
from frrl.trajectory_executor import Waypoint, TrajectoryInterpolator, TrajectoryExecutor, save_trajectory


def create_optimized_trajectory(
    block_pos: np.ndarray,
    plate_pos: np.ndarray,
) -> list:
    """
    创建优化的pick-and-place轨迹。
    """
    standard_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]

    waypoints = []

    # ========== 起点 ==========
    waypoints.append(Waypoint(
        position=np.array([0.49, 0.0, 0.08]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=0.0,
        name="0. 起点"
    ))

    # ========== 阶段1: 移动到Block上方 ==========
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], 0.15]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=2.0,  # 增加时间，更慢更稳
        name="1. Block上方"
    ))

    # ========== 阶段2: 缓慢下降到Block ==========
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.005]),  # 稍微高一点点
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.5,  # 缓慢下降
        name="2. 下降到Block"
    ))

    # ========== 阶段3: 开始关闭夹爪（慢） ==========
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.005]),
        orientation=standard_quat.copy(),
        gripper=0.3,  # 先关一点
        duration=1.0,
        name="3. 预抓取"
    ))

    # ========== 阶段4: 完全关闭夹爪 ==========
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], block_pos[2] + 0.005]),
        orientation=standard_quat.copy(),
        gripper=1.0,  # 完全关闭
        duration=2.5,  # 增加停留时间，确保抓紧
        name="4. 完全抓取"
    ))

    # ========== 阶段5: 缓慢抬升 ==========
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], 0.12]),  # 先抬一点点
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=1.0,
        name="5. 初步抬升"
    ))

    # ========== 阶段6: 继续抬升到安全高度 ==========
    waypoints.append(Waypoint(
        position=np.array([block_pos[0], block_pos[1], 0.20]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=1.5,
        name="6. 抬升到安全高度"
    ))

    # ========== 阶段7: 移动到Plate上方 ==========
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.20]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=2.5,  # 慢速移动，保持稳定
        name="7. 移动到Plate上方"
    ))

    # ========== 阶段8: 下降到Plate上方 ==========
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.08]),  # 先到中间高度
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=1.5,
        name="8. 下降中"
    ))

    # ========== 阶段9: 放置到Plate ==========
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.03]),
        orientation=standard_quat.copy(),
        gripper=1.0,
        duration=1.5,
        name="9. 放置位置"
    ))

    # ========== 阶段10: 打开夹爪释放 ==========
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.03]),
        orientation=standard_quat.copy(),
        gripper=0.0,  # 打开夹爪
        duration=1.5,
        name="10. 释放Block"
    ))

    # ========== 阶段11: 上移远离 ==========
    waypoints.append(Waypoint(
        position=np.array([plate_pos[0], plate_pos[1], 0.15]),
        orientation=standard_quat.copy(),
        gripper=0.0,
        duration=1.5,
        name="11. 远离"
    ))

    return waypoints


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="优化的轨迹Pick-and-Place演示")
    parser.add_argument('--no-render', action='store_true',
                       help='不显示viewer')
    parser.add_argument('--save-trajectory', action='store_true',
                       help='保存轨迹到文件')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("优化的轨迹 Pick-and-Place 演示")
    print("=" * 70)
    print()
    print("优化点:")
    print("  ✅ 增加抓取阶段停留时间（2.5秒）")
    print("  ✅ 分步抓取（预抓取 → 完全抓取）")
    print("  ✅ 分步抬升（初步抬升 → 安全高度）")
    print("  ✅ 更慢的运动速度")
    print("  ✅ 更多中间点，运动更平滑")
    print("=" * 70)
    print()

    # 创建环境
    env = PandaPickPlaceEnv(
        control_dt=0.1,
        physics_dt=0.002,
        max_episode_steps=500,
        delta_action_scale=0.02,
        render_mode="human" if not args.no_render else None,
    )

    # 重置获取初始位置
    obs, info = env.reset()
    block_pos = obs['state'][4:7]
    plate_pos = obs['state'][7:10]

    print(f"初始状态:")
    print(f"  Block位置: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]")
    print(f"  Plate位置: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}, {plate_pos[2]:.3f}]")
    print()

    # 创建优化轨迹
    print("创建优化轨迹...")
    waypoints = create_optimized_trajectory(
        block_pos=block_pos,
        plate_pos=plate_pos,
    )

    print(f"✅ 已创建 {len(waypoints)} 个关键点")
    print()

    # 显示关键点信息
    print("关键点信息:")
    print("-" * 70)
    cumulative_time = 0.0
    for i, wp in enumerate(waypoints):
        cumulative_time += wp.duration
        print(f"  [{i+1:2d}] {wp.name:25s} @ {cumulative_time:5.1f}s")
        if wp.gripper > 0.5:
            gripper_status = f"关闭 ({wp.gripper:.1f})"
        elif wp.gripper > 0.0:
            gripper_status = f"部分 ({wp.gripper:.1f})"
        else:
            gripper_status = "打开"
        print(f"       位置: [{wp.position[0]:.3f}, {wp.position[1]:.3f}, {wp.position[2]:.3f}]")
        print(f"       夹爪: {gripper_status}")

    print("-" * 70)
    total_duration = sum(wp.duration for wp in waypoints)
    print(f"总时长: {total_duration:.1f}秒")
    print()

    # 保存轨迹（可选）
    if args.save_trajectory:
        traj_file = "trajectories/optimized_pick_place.json"
        Path("trajectories").mkdir(exist_ok=True)
        save_trajectory(waypoints, traj_file)
        print()

    # 创建插值器
    print("创建轨迹插值器...")
    interpolator = TrajectoryInterpolator(waypoints, control_dt=env.control_dt)
    trajectory = interpolator.generate_trajectory()
    print(f"✅ 生成了 {len(trajectory)} 个插值轨迹点")
    print()

    # 执行轨迹
    print("开始执行轨迹...")
    print()

    executor = TrajectoryExecutor(env, interpolator)
    result = executor.execute(render=not args.no_render, verbose=True)

    # 保持viewer打开
    if not args.no_render:
        print()
        print("=" * 70)
        print("Viewer将保持打开，您可以自由观察")
        print("=" * 70)
        print()
        print("提示:")
        print("  - 使用鼠标拖动旋转/平移视角")
        print("  - 双击物体聚焦")
        print("  - 按 Ctrl+C 退出")
        print()

        try:
            # 保持打开直到用户按Ctrl+C
            while True:
                action = np.zeros(4, dtype=np.float32)
                env.step(action)
                env.render()
        except KeyboardInterrupt:
            print()
            print("用户退出")

    env.close()

    # 最终总结
    print()
    print("=" * 70)
    print("优化效果对比")
    print("=" * 70)
    print()
    print("原始轨迹:")
    print("  关键点数: 9个")
    print("  总时长: 11.0秒")
    print("  抓取时间: 1.5秒")
    print("  Block抬升: 4.8cm")
    print()
    print("优化轨迹:")
    print(f"  关键点数: {len(waypoints)}个")
    print(f"  总时长: {total_duration:.1f}秒")
    print(f"  抓取时间: 3.5秒（预抓取1.0s + 完全抓取2.5s）")
    print(f"  Block抬升: {result['final_block_pos'][2]*100:.1f}cm")
    print()
    print(f"任务状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
    print(f"抓取状态: {'✅ 成功' if result['grasped'] else '❌ 失败'}")
    print(f"总奖励: {result['total_reward']:.2f}")
    print("=" * 70)

    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit(main())
