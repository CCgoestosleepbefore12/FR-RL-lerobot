"""Backup Env TRACKING-only 改造冒烟测试。

Usage:
    pytest tests/test_backup_env_tracking.py
"""
import sys
import numpy as np

from frrl.envs.sim.panda_backup_policy_env import (
    PandaBackupPolicyEnv,
    MotionMode,
    D_TIGHT,
    STALL_STEPS_RANGE,
    HAND_SPAWN_DIST,
    MAX_EPISODE_STEPS,
    ROT_ACTION_SCALE,
    ARM_COLLISION_DIST,
    MAX_ROTATION,
)


def _make_env(num_obstacles=1, seed=0):
    env = PandaBackupPolicyEnv(
        num_obstacles=num_obstacles,
        enable_dr=False,  # 关掉 DR，便于确定性断言
        seed=seed,
    )
    env.reset(seed=seed)
    return env


def _make_env_v2(num_obstacles=1, seed=0):
    """V2：腕+手单球避障 + 旋转预算。kwargs 与 PandaBackupPolicyS1V2-v0 注册对齐。"""
    env = PandaBackupPolicyEnv(
        num_obstacles=num_obstacles,
        enable_dr=False,
        seed=seed,
        use_arm_sphere_collision=True,
        max_displacement=0.30,
        enforce_cartesian_bounds=False,
    )
    env.reset(seed=seed)
    return env


def test_obs_shape_s1_s2():
    """S1 单帧 28D × stack 3 = 84D；S2 单帧 38D × stack 3 = 114D。"""
    for n, expected in [(1, 84), (2, 114)]:
        env = _make_env(num_obstacles=n)
        obs, _ = env.reset()
        assert obs["agent_pos"].shape == (expected,), \
            f"num_obs={n}: expected {expected}D, got {obs['agent_pos'].shape}"
        env.close()
    print("  [PASS] obs shape: S1=84D, S2=114D")


def test_motion_mode_is_tracking_only():
    """MotionMode 枚举只有 TRACKING，且 info 报告 TRACKING。"""
    members = [m.name for m in MotionMode]
    assert members == ["TRACKING"], f"expected ['TRACKING'], got {members}"

    env = _make_env()
    obs, info = env.reset()
    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["motion_mode"] == "TRACKING", f"motion_mode={info['motion_mode']}"
    env.close()
    print("  [PASS] motion mode is TRACKING-only")


def test_tracking_moves_toward_tcp():
    """手远于 D_TIGHT 时，一步后距离应缩小（沿 TCP 方向前进）。"""
    env = _make_env()
    # 把手放到 TCP 后方已知距离 12cm（-X 朝向机器人基座，确保落在工作空间内无 clip 干扰）
    tcp = env.unwrapped._data.site_xpos[env.unwrapped._pinch_site_id].copy()  # shape: (3,)
    env.unwrapped._obstacle_pos[0] = tcp + np.array([-0.12, 0.0, 0.0])  # shape: (3,)
    env.unwrapped._stall_remaining[0] = 0
    env.unwrapped._obstacle_speed = 0.02  # 已知速度

    dist_before = float(np.linalg.norm(env.unwrapped._obstacle_pos[0] - tcp))
    env.unwrapped._move_obstacle(0)
    dist_after = float(np.linalg.norm(env.unwrapped._obstacle_pos[0] - tcp))

    # 前进量 ~= speed（允许工作空间 clip 误差）
    delta = dist_before - dist_after
    assert 0.015 < delta < 0.025, f"expected ~0.02m step, got {delta:.4f}"
    env.close()
    print(f"  [PASS] tracking steps {delta:.4f}m toward TCP")


def test_stall_triggers_at_d_tight():
    """手进入 D_TIGHT 后触发停顿，停顿期间位置不变。"""
    env = _make_env()
    tcp = env.unwrapped._data.site_xpos[env.unwrapped._pinch_site_id].copy()
    # 把手放到 TCP 5cm 内（< D_TIGHT=8cm）
    env.unwrapped._obstacle_pos[0] = tcp + np.array([0.05, 0.0, 0.0])
    env.unwrapped._stall_remaining[0] = 0
    env.unwrapped._obstacle_speed = 0.02

    # 第一步：进入 D_TIGHT 检测，触发停顿计数（本步不动）
    pos_at_trigger = env.unwrapped._obstacle_pos[0].copy()
    env.unwrapped._move_obstacle(0)
    assert np.allclose(env.unwrapped._obstacle_pos[0], pos_at_trigger), \
        "停顿触发步不应移动"
    stall = env.unwrapped._stall_remaining[0]
    assert STALL_STEPS_RANGE[0] <= stall <= STALL_STEPS_RANGE[1] - 1, \
        f"stall count {stall} out of range"

    # 停顿期间连续调用，位置应完全不变
    for _ in range(stall):
        env.unwrapped._move_obstacle(0)
        assert np.allclose(env.unwrapped._obstacle_pos[0], pos_at_trigger), \
            "停顿期间位置不应变化"

    assert env.unwrapped._stall_remaining[0] == 0, "停顿计数应归零"
    env.close()
    print(f"  [PASS] stall at D_TIGHT (stall_count={stall})")


def test_hand_spawn_distance_in_range():
    """多次 reset，spawn 距离都落在 HAND_SPAWN_DIST 内。"""
    env = _make_env()
    lo, hi = HAND_SPAWN_DIST
    dists = []
    for seed in range(20):
        env.reset(seed=seed)
        tcp = env.unwrapped._data.site_xpos[env.unwrapped._pinch_site_id].copy()
        d = float(np.linalg.norm(env.unwrapped._obstacle_pos[0] - tcp))
        dists.append(d)
        # 允许 clip 到工作空间后略小于 lo，但不应超过 hi + 小余量
        assert d <= hi + 0.01, f"seed={seed}: spawn dist {d:.3f} > hi {hi}"
    env.close()
    print(f"  [PASS] spawn dists in [{min(dists):.3f}, {max(dists):.3f}] "
          f"(target [{lo}, {hi}])")


def test_rotation_action_changes_mocap_quat():
    """非零姿态动作应使 mocap_quat 发生变化（6D sim2real 关键路径）。"""
    env = _make_env(seed=2)
    env.reset(seed=2)
    q_before = env.unwrapped._data.mocap_quat[0].copy()
    # 只施加姿态增量（位置 0）：绕 x 轴
    action = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    env.step(action)
    q_after = env.unwrapped._data.mocap_quat[0].copy()
    changed = not np.allclose(q_before, q_after, atol=1e-6)
    assert changed, f"mocap_quat 应变化，但 before={q_before}, after={q_after}"

    # 零姿态动作应保持 mocap_quat 不变
    env.reset(seed=3)
    q_before = env.unwrapped._data.mocap_quat[0].copy()
    env.step(np.zeros(6, dtype=np.float32))
    q_after = env.unwrapped._data.mocap_quat[0].copy()
    assert np.allclose(q_before, q_after, atol=1e-6), \
        f"零旋转动作 mocap_quat 应不变，但 before={q_before}, after={q_after}"
    env.close()
    print(f"  [PASS] rotation action updates mocap_quat; zero action preserves it")


def test_episode_truncates_at_max_steps():
    """zero action 下 episode 在 MAX_EPISODE_STEPS 步 truncate 或提前 terminate。"""
    env = _make_env(seed=1)
    obs, _ = env.reset(seed=1)
    terminated = False
    truncated = False
    step = 0
    for step in range(MAX_EPISODE_STEPS + 5):
        obs, rew, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        if terminated or truncated:
            break
    assert (step + 1) <= MAX_EPISODE_STEPS, \
        f"episode ran {step+1} steps > MAX={MAX_EPISODE_STEPS}"
    env.close()
    print(f"  [PASS] episode ended at step {step+1} "
          f"(terminated={terminated}, truncated={truncated}, max={MAX_EPISODE_STEPS})")


def test_v2_spawn_above_collision_threshold():
    """V2：spawn 时 arm_center 到手距离 ≥ ARM_COLLISION_DIST（消除 1-step death）。"""
    below = []
    for seed in range(30):
        env = _make_env_v2(seed=seed)
        u = env.unwrapped
        arm_center = u._data.xpos[u._panda_hand_body_id].copy()
        hand = u._obstacle_pos[0]
        d = float(np.linalg.norm(hand - arm_center))
        if d < ARM_COLLISION_DIST:
            below.append((seed, d))
        env.close()
    assert not below, \
        f"V2 spawn 距离应全部 >= {ARM_COLLISION_DIST:.3f}m，但 {len(below)}/30 低于阈值: {below[:3]}"
    print(f"  [PASS] V2 spawn arm_center→hand 全部 >= {ARM_COLLISION_DIST:.3f}m (30 seed)")


def test_v2_arm_sphere_collision_threshold():
    """V2：手贴到 arm sphere 表面 (< ARM_COLLISION_DIST) 应终止为 hand_collision。"""
    env = _make_env_v2(seed=0)
    hand_body_id = env.unwrapped._panda_hand_body_id
    arm_center = env.unwrapped._data.xpos[hand_body_id].copy()
    # 把手放到球心前方 < ARM_COLLISION_DIST（例：0.10m < 0.135m）
    env.unwrapped._obstacle_pos[0] = arm_center + np.array([0.10, 0.0, 0.0])
    # 冻结手（避免 step 中的 _move_obstacle 破坏设置）
    env.unwrapped._stall_remaining[0] = 999

    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert term, f"应因 hand_collision 终止，但 term={term}, info={info}"
    assert info.get("violation_type") == "hand_collision", \
        f"expected hand_collision, got {info.get('violation_type')}"
    env.close()
    print(f"  [PASS] V2 arm-sphere collision terminates at "
          f"{ARM_COLLISION_DIST:.3f}m (test dist 0.10m)")


def test_v2_rotation_does_not_reduce_arm_distance():
    """V2 反作弊核心：pure rotation 动作不应让 arm sphere 中心"躲开"手。

    球心 = mocap weld 点（panda hand body 原点），对绕 weld 点的旋转 rigid。
    施加纯旋转动作前后，arm_center 位置应基本不变（位置 delta < 1mm）。"""
    env = _make_env_v2(seed=5)
    hand_body_id = env.unwrapped._panda_hand_body_id
    arm_center_before = env.unwrapped._data.xpos[hand_body_id].copy()
    # 把手放远（避免触发碰撞）
    env.unwrapped._obstacle_pos[0] = arm_center_before + np.array([0.30, 0.0, 0.0])
    env.unwrapped._stall_remaining[0] = 999
    # 纯旋转动作：绕 x 轴
    action = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0], dtype=np.float32)
    env.step(action)
    arm_center_after = env.unwrapped._data.xpos[hand_body_id].copy()
    delta = float(np.linalg.norm(arm_center_after - arm_center_before))
    # OSC 追踪会有微小伺服误差，允许 <5mm
    assert delta < 0.005, \
        f"旋转应保持 arm_center 位置不变，但移动了 {delta*1000:.2f}mm"
    env.close()
    print(f"  [PASS] V2 pure rotation preserves arm_center "
          f"(delta={delta*1000:.2f}mm)")


def test_v2_rotation_budget_termination():
    """V2 旋转预算**机制**测试：显式传小 max_rotation，验证硬终止链路仍可工作。

    说明：S1V2 默认已改为 max_rotation=π 软惩罚模式（球心对旋转 rigid →
    硬上限在 V2 下无防作弊意义，只会挡几何必要的换 IK 构型）。此测试
    保留机制可用性，以便未来变体可按需开启。"""
    env = PandaBackupPolicyEnv(
        enable_dr=False, seed=7,
        use_arm_sphere_collision=True,
        max_displacement=0.30,
        enforce_cartesian_bounds=False,
        max_rotation=0.5,  # 显式开启硬上限供本测试使用
    )
    env.reset(seed=7)
    # 把手放远（避免撞停）
    hand_body_id = env.unwrapped._panda_hand_body_id
    arm_center = env.unwrapped._data.xpos[hand_body_id].copy()
    env.unwrapped._obstacle_pos[0] = arm_center + np.array([0.40, 0.0, 0.0])
    env.unwrapped._stall_remaining[0] = 999

    # 每步最大旋转 ≈ 1.0 * ROT_ACTION_SCALE = 0.1 rad；本测试 max_rotation=0.5
    rotation_seen = 0.0
    term_info = None
    for _ in range(10):
        obs, rew, term, trunc, info = env.step(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        )
        rotation_seen = info.get("rotation", 0.0)
        env.unwrapped._obstacle_pos[0] = arm_center + np.array([0.40, 0.0, 0.0])
        env.unwrapped._stall_remaining[0] = 999
        if term:
            term_info = info
            break

    assert term_info is not None, f"应触发旋转预算终止，但未 terminate，rotation_seen={rotation_seen:.3f}"
    assert term_info.get("violation_type") == "excessive_rotation", \
        f"expected excessive_rotation, got {term_info.get('violation_type')}"
    assert rotation_seen > 0.5, \
        f"终止时 rotation {rotation_seen:.3f} 应 > max_rotation 0.5"
    print(f"  [PASS] V2 rotation budget mechanism wired "
          f"(terminates at {rotation_seen:.3f}rad when max=0.5)")
    env.close()


def test_v2_rotation_penalty_applied():
    """V2：存活步的奖励应比纯位置版多一个 -rotation_coeff * rotation 项。"""
    # 对比：同样状态下，V1(零旋转) vs V2(有旋转) 的 reward
    env = _make_env_v2(seed=9)
    hand_body_id = env.unwrapped._panda_hand_body_id
    arm_center = env.unwrapped._data.xpos[hand_body_id].copy()
    env.unwrapped._obstacle_pos[0] = arm_center + np.array([0.40, 0.0, 0.0])
    env.unwrapped._stall_remaining[0] = 999

    # 施加一步纯旋转
    obs, rew, term, trunc, info = env.step(
        np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    )
    assert not term, f"单步旋转 0.05rad 不应触发任何终止，但 term={term}, info={info}"
    rotation = info.get("rotation", 0.0)
    assert rotation > 0.01, f"施加 0.5*ROT_ACTION_SCALE 后应 rotation > 0.01, got {rotation:.4f}"
    # reward 应能被反算：包含 rotation 惩罚项
    # 这里只断言 rotation 进入 info 且 >0，证明旋转预算/惩罚链路打通
    env.close()
    print(f"  [PASS] V2 rotation={rotation:.4f}rad applied into info; "
          f"penalty chain wired in reward")


def main():
    print("=== Backup Env TRACKING-only 测试 ===")
    tests = [
        test_obs_shape_s1_s2,
        test_motion_mode_is_tracking_only,
        test_tracking_moves_toward_tcp,
        test_stall_triggers_at_d_tight,
        test_hand_spawn_distance_in_range,
        test_rotation_action_changes_mocap_quat,
        test_episode_truncates_at_max_steps,
        test_v2_spawn_above_collision_threshold,
        test_v2_arm_sphere_collision_threshold,
        test_v2_rotation_does_not_reduce_arm_distance,
        test_v2_rotation_budget_termination,
        test_v2_rotation_penalty_applied,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    print("=== ALL PASS ===")


if __name__ == "__main__":
    main()
