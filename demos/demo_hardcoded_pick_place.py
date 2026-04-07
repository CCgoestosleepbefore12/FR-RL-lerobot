#!/usr/bin/env python3
"""
硬编码的Pick-and-Place演示

使用gym环境接口，展示完整的抓取-放置任务。
Agent只输出Delta EE动作，OSC控制器封装在环境内部。
"""

import sys
from pathlib import Path
# 使用frrl包

import numpy as np
from frrl.envs.panda_pick_place_env import PandaPickPlaceEnv


def hardcoded_pick_and_place(render=True):
    """
    硬编码的pick-and-place策略。

    Args:
        render: 是否显示viewer
    """
    print("=" * 70)
    print("硬编码 Pick-and-Place 演示")
    print("=" * 70)
    print()
    print("架构说明:")
    print("  - Agent输出: [dx, dy, dz, gripper] (Delta EE)")
    print("  - OSC控制: 完全封装在env.step()内部")
    print("  - Agent不知道关节角度、力矩等底层细节")
    print("=" * 70)
    print()

    # 创建环境
    env = PandaPickPlaceEnv(
        control_dt=0.1,
        physics_dt=0.002,
        max_episode_steps=500,
        delta_action_scale=0.02,  # 2cm per action
        render_mode="human" if render else None,
    )

    # 重置环境
    obs, info = env.reset()

    # 获取初始位置
    ee_pos = obs['state'][:3]
    block_pos = obs['state'][4:7]
    plate_pos = obs['state'][7:10]

    print(f"初始状态:")
    print(f"  末端位置: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Block位置: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]")
    print(f"  Plate位置: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}, {plate_pos[2]:.3f}]")
    print()

    total_reward = 0
    step_count = 0

    # ==================== 阶段1: 移动到Block上方 ====================
    print("[阶段 1/7] 移动到Block上方...")
    print("-" * 70)

    target_pos = np.array([block_pos[0], block_pos[1], 0.15])

    while True:
        ee_pos = obs['state'][:3]
        delta = target_pos - ee_pos
        distance = np.linalg.norm(delta)

        # 计算动作 (Delta EE)
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(delta / 0.02, -1, 1)  # 归一化到[-1,1]
        action[3] = 0.0  # 夹爪开

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

        # 检查是否到达
        if distance < 0.01 or step_count > 50:
            break

    final_ee = obs['state'][:3]
    error = np.linalg.norm(final_ee - target_pos)
    print(f"  步数: {step_count}")
    print(f"  最终误差: {error*100:.1f}cm")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    # ==================== 阶段2: 下降到Block ====================
    print("[阶段 2/7] 下降到Block位置...")
    print("-" * 70)

    step_count = 0
    # 更新block位置（可能移动了）
    block_pos = obs['state'][4:7]
    target_pos = np.array([block_pos[0], block_pos[1], block_pos[2]])

    while True:
        ee_pos = obs['state'][:3]
        # 实时更新block位置
        block_pos = obs['state'][4:7]
        target_pos = np.array([block_pos[0], block_pos[1], block_pos[2]])

        delta = target_pos - ee_pos
        distance = np.linalg.norm(delta)

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(delta / 0.02, -1, 1)
        action[3] = 0.0  # 夹爪开

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

        if distance < 0.003 or step_count > 50:  # 更严格的阈值
            break

    final_ee = obs['state'][:3]
    error = np.linalg.norm(final_ee - target_pos)
    print(f"  步数: {step_count}")
    print(f"  最终误差: {error*1000:.1f}mm")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    # ==================== 阶段3: 关闭夹爪 ====================
    print("[阶段 3/7] 关闭夹爪抓取...")
    print("-" * 70)

    step_count = 0
    for _ in range(50):  # 5秒 @ 10Hz
        action = np.zeros(4, dtype=np.float32)
        action[3] = 1.0  # 夹爪关闭

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

    gripper_pos = obs['state'][3]

    # 检查接触点（更可靠的方式）
    n_contacts = env.robot.data.ncon
    block_geom_id = env.robot.model.geom('block').id
    contacts = 0

    for i in range(n_contacts):
        contact = env.robot.data.contact[i]
        if contact.geom1 == block_geom_id or contact.geom2 == block_geom_id:
            g1 = env.robot.model.geom(contact.geom1).name
            g2 = env.robot.model.geom(contact.geom2).name
            if 'pad' in g1 or 'pad' in g2:
                contacts += 1

    has_contact = contacts > 10
    print(f"  步数: {step_count}")
    print(f"  夹爪位置: {gripper_pos:.4f}")
    print(f"  接触点数: {contacts}")
    print(f"  接触状态: {'✅ 良好' if has_contact else '❌ 不足'}")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    if not has_contact:
        print("⚠️  接触不足，但继续尝试...")
        # 不立即终止，给一次机会

    # ==================== 阶段4: 抬升Block ====================
    print("[阶段 4/7] 抬升Block...")
    print("-" * 70)

    step_count = 0
    target_height = 0.20  # 20cm

    while True:
        ee_pos = obs['state'][:3]
        target_pos = ee_pos.copy()
        target_pos[2] = target_height
        delta = target_pos - ee_pos

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(delta / 0.02, -1, 1)
        action[3] = 1.0  # 夹爪保持关闭

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

        if abs(ee_pos[2] - target_height) < 0.01 or step_count > 40:
            break

    block_height = obs['state'][6]  # block z position
    grasped = info.get('grasped', False)
    print(f"  步数: {step_count}")
    print(f"  Block高度: {block_height*100:.1f}cm")
    print(f"  抓取状态: {'✅ 保持' if grasped else '❌ 掉落'}")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    if not grasped:
        print("⚠️  Block掉落，终止演示")
        env.close()
        return False

    # ==================== 阶段5: 移动到Plate上方 ====================
    print("[阶段 5/7] 移动到Plate上方...")
    print("-" * 70)

    step_count = 0
    target_pos = np.array([plate_pos[0], plate_pos[1], 0.15])

    while True:
        ee_pos = obs['state'][:3]
        delta = target_pos - ee_pos
        distance = np.linalg.norm(delta)

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(delta / 0.02, -1, 1)
        action[3] = 1.0  # 夹爪保持关闭

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

        if distance < 0.02 or step_count > 60:
            break

    block_height = obs['state'][6]
    grasped = info.get('grasped', False)
    print(f"  步数: {step_count}")
    print(f"  Block高度: {block_height*100:.1f}cm")
    print(f"  抓取状态: {'✅ 保持' if grasped else '❌ 掉落'}")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    # ==================== 阶段6: 下降到Plate ====================
    print("[阶段 6/7] 下降到Plate...")
    print("-" * 70)

    step_count = 0
    target_pos = np.array([plate_pos[0], plate_pos[1], plate_pos[2] + 0.03])

    while True:
        ee_pos = obs['state'][:3]
        delta = target_pos - ee_pos
        distance = np.linalg.norm(delta)

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(delta / 0.02, -1, 1)
        action[3] = 1.0  # 夹爪保持关闭

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

        if distance < 0.005 or step_count > 30:
            break

    print(f"  步数: {step_count}")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    # ==================== 阶段7: 打开夹爪释放 ====================
    print("[阶段 7/7] 打开夹爪释放Block...")
    print("-" * 70)

    step_count = 0
    for _ in range(15):  # 1.5秒
        action = np.zeros(4, dtype=np.float32)
        action[3] = 0.0  # 夹爪打开

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染viewer
        if render:
            env.render()

    print(f"  步数: {step_count}")
    print(f"  累计奖励: {total_reward:.2f}")
    print()

    # 稍微上移，远离Block
    for _ in range(20):
        action = np.zeros(4, dtype=np.float32)
        action[2] = 0.5  # 向上移动
        action[3] = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 渲染viewer
        if render:
            env.render()

    # ==================== 最终结果 ====================
    print()
    print("=" * 70)
    print("最终结果")
    print("=" * 70)

    final_block_pos = obs['state'][4:7]
    block_to_plate = info['block_to_plate']
    success = info['success']

    print(f"Block最终位置: [{final_block_pos[0]:.3f}, {final_block_pos[1]:.3f}, {final_block_pos[2]:.3f}]")
    print(f"Plate目标位置: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}, {plate_pos[2]:.3f}]")
    print(f"与Plate距离: {block_to_plate*100:.1f}cm")
    print(f"Block高度: {final_block_pos[2]*100:.1f}cm")
    print(f"总奖励: {total_reward:.2f}")
    print()

    if success:
        print("🎉🎉🎉 任务成功完成！ 🎉🎉🎉")
        print("✅ Block已成功放置到Plate上")
    elif info.get('grasped', False):
        print("⚠️  部分成功：Block被抓起但未正确放置")
    else:
        print("❌ 任务失败")

    print("=" * 70)
    print()

    # 保持viewer打开
    if render:
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
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
        except KeyboardInterrupt:
            print()
            print("用户退出")

    env.close()

    return success


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="硬编码Pick-and-Place演示")
    parser.add_argument('--no-render', action='store_true',
                       help='不显示viewer（加快速度）')
    args = parser.parse_args()

    print()
    print("🤖 启动硬编码Pick-and-Place演示...")
    print()

    success = hardcoded_pick_and_place(render=not args.no_render)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
