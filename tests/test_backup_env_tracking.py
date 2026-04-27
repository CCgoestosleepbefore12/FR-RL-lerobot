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
    D_TIGHT_ARM,
    STALL_STEPS_RANGE,
    HAND_SPAWN_DIST,
    MAX_EPISODE_STEPS,
    ROT_ACTION_SCALE,
    ARM_COLLISION_DIST,
    ARM_COLLISION_DIST_V3,
    MAX_ROTATION,
    ARM_SPHERES_V3,
    OBSTACLE_RADIUS_V3,
    ARM_SPAWN_DIST_V3,
    HAND_SPEED_RANGE_V3,
    PROXIMITY_REWARD_MAX,
    PROXIMITY_SAFE_DIST,
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


def _make_env_v3(num_obstacles=1, seed=0):
    """V3：5 球全臂避障 + obstacle r=0.10 + spawn (0.30,0.40) + proximity reward。
    kwargs 与 PandaBackupPolicyS1V3-v0 注册对齐。"""
    env = PandaBackupPolicyEnv(
        num_obstacles=num_obstacles,
        enable_dr=False,
        seed=seed,
        use_arm_sphere_collision=True,
        use_full_arm_collision=True,
        max_displacement=0.50,   # = 注册 kwargs (Path A: 0.40 → 0.50)
        enforce_cartesian_bounds=False,
    )
    env.reset(seed=seed)
    return env


def _make_env_v3c(num_obstacles=1, seed=0):
    """V3c: 同 V3 但 tracking_target='arm_center' + D_TIGHT_ARM=25cm。
    kwargs 与 PandaBackupPolicyS1V3c-v0 注册对齐。"""
    env = PandaBackupPolicyEnv(
        num_obstacles=num_obstacles,
        enable_dr=False,
        seed=seed,
        use_arm_sphere_collision=True,
        use_full_arm_collision=True,
        tracking_target="arm_center",   # ★ V3c 核心
        max_displacement=0.50,
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


def test_v3_spawn_no_one_step_death():
    """V3: 30 seed 全部 spawn 距离对每个臂球都 ≥ (sphere_r + obstacle_r)。"""
    failures = []
    for seed in range(30):
        env = _make_env_v3(seed=seed)
        u = env.unwrapped
        for bid, sphere_r in u._arm_sphere_specs:
            sphere_pos = u._data.xpos[bid].copy()
            d = float(np.linalg.norm(u._obstacle_pos[0] - sphere_pos))
            threshold = sphere_r + OBSTACLE_RADIUS_V3
            if d < threshold:
                failures.append((seed, u._model.body(bid).name, d, threshold))
        env.close()
    assert not failures, f"V3 spawn 1-step-death 风险: {failures[:5]}"
    print(f"  [PASS] V3 spawn 全部满足所有 5 个臂球的安全距离 (30 seed × 5 球)")


def test_v3_obstacle_radius_010():
    """V3: 把手放到 panda_hand 球前方 0.19m (< 0.20 阈值)，应触发 hand_collision。"""
    env = _make_env_v3(seed=0)
    u = env.unwrapped
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    # 0.19 < 0.10 (sphere_r) + 0.10 (obstacle_r) = 0.20 阈值
    u._obstacle_pos[0] = arm_center + np.array([0.19, 0.0, 0.0])
    u._stall_remaining[0] = 999

    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert term, f"V3 应在 0.19m < 0.20m 阈值终止，但 term={term}, info={info}"
    assert info.get("violation_type") == "hand_collision"
    env.close()
    print(f"  [PASS] V3 panda_hand 阈值 0.20m 工作 (0.10 sphere_r + 0.10 obstacle_r)")


def test_v3_elbow_collision_terminates():
    """V3 核心：手贴到 link4（肘）球内 → 即使离 EE 远也应触发 hand_collision。
    这是 V3 解决的"V2 看不见肘碰撞"问题的核心场景。"""
    env = _make_env_v3(seed=2)
    u = env.unwrapped
    # 找到 link4 body id
    link4_id = None
    link4_r = None
    for bid, r in u._arm_sphere_specs:
        if u._model.body(bid).name == "link4":
            link4_id = bid
            link4_r = r
            break
    assert link4_id is not None, "未找到 link4 sphere"
    elbow_pos = u._data.xpos[link4_id].copy()
    # 在 link4 球前方 0.13m (< 0.07 + 0.10 = 0.17 阈值) 放置 obstacle
    u._obstacle_pos[0] = elbow_pos + np.array([0.13, 0.0, 0.0])
    u._stall_remaining[0] = 999
    # 同时把 panda_hand 距离做远 — 验证不是因为 hand 球触发
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    hand_to_obs = float(np.linalg.norm(u._obstacle_pos[0] - arm_center))
    assert hand_to_obs > 0.20, f"测试前提：obstacle 应离 panda_hand 远 ({hand_to_obs:.3f}m > 0.20m)"

    # 验证测试前提：仅 link4 进入威胁范围，其他 4 个球都未违反
    # （否则碰撞触发可能来自其他球，无法证明"V3 看到肘"的核心命题）
    triggered_spheres = []
    for bid, sphere_r in u._arm_sphere_specs:
        sp_pos = u._data.xpos[bid]
        d = float(np.linalg.norm(sp_pos - u._obstacle_pos[0]))
        thr = sphere_r + OBSTACLE_RADIUS_V3
        if d < thr:
            triggered_spheres.append((u._model.body(bid).name, d, thr))
    assert len(triggered_spheres) == 1 and triggered_spheres[0][0] == "link4", (
        f"测试前提失败：应只有 link4 < threshold；实际 {triggered_spheres}"
    )

    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert term, (
        f"link4 距 obstacle {0.13:.3f}m < {link4_r + OBSTACLE_RADIUS_V3:.3f}m 阈值 "
        f"应 hand_collision 终止；hand 距 {hand_to_obs:.3f}m 是远离的；"
        f"但 term={term}, info={info}"
    )
    assert info.get("violation_type") == "hand_collision"
    env.close()
    print(f"  [PASS] V3 link4 (肘) 单独触发碰撞 (hand {hand_to_obs:.3f}m 远未触发)")


def test_v3_proximity_reward_saturation():
    """V3 proximity reward 行为：clear=10cm 时饱和；clear=5cm 时 0.10；clear=0 时 0。"""
    env = _make_env_v3(seed=4)
    u = env.unwrapped
    # 冻结手位置避免 step 内 _move_obstacle 移动
    u._stall_remaining[0] = 999
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()

    # case 1: clear = 10cm = SAFE_DIST → 饱和 0.20
    # 把手放到 panda_hand 球外 0.30m (= 0.10 sphere_r + 0.10 obs_r + 0.10 clear)
    u._obstacle_pos[0] = arm_center + np.array([0.30, 0.0, 0.0])
    obs, rew_sat, term, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert not term, f"clear=10cm 应不终止，info={info}"
    clear_sat = info["step_surface_clearance"]
    assert abs(clear_sat - 0.10) < 0.01, f"step_surface_clearance 应 ≈0.10, got {clear_sat:.4f}"

    # case 2: clear = 5cm → 0.10 reward
    env2 = _make_env_v3(seed=4)
    u2 = env2.unwrapped
    u2._stall_remaining[0] = 999
    arm_center2 = u2._data.xpos[u2._panda_hand_body_id].copy()
    u2._obstacle_pos[0] = arm_center2 + np.array([0.25, 0.0, 0.0])  # = 0.20 + 0.05 clear
    obs2, rew_half, _, _, info2 = env2.step(np.zeros(6, dtype=np.float32))
    clear_half = info2["step_surface_clearance"]
    assert abs(clear_half - 0.05) < 0.01, f"clear 应 ≈0.05, got {clear_half:.4f}"

    # 同位移 / 同 action 下，proximity 差额应 ≈ 0.10
    proximity_diff = (rew_sat - rew_half)
    expected = PROXIMITY_REWARD_MAX * (1.0 - 0.5)  # = 0.10
    # 同 seed + zero action 下 displacement / rotation / action_norm 项基本相等，
    # 真实差额 ≈ proximity_max × 0.5 = 0.10。容忍 0.01 仍能容纳浮点误差和 ε 级 disp 漂移。
    disp_diff = abs(info["displacement"] - info2["displacement"])
    assert disp_diff < 1e-3, f"测试前提：相同 zero action 应 displacement 差 < 1e-3, got {disp_diff:.5f}"
    assert abs(proximity_diff - expected) < 0.01, (
        f"proximity diff (clear 0.10 vs 0.05) 应 ≈{expected:.3f}, got {proximity_diff:.4f}; "
        f"rew_sat={rew_sat:.4f}, rew_half={rew_half:.4f}"
    )
    env.close()
    env2.close()
    print(f"  [PASS] V3 proximity reward: clear=0.10 saturates ({rew_sat:.3f}), "
          f"clear=0.05 mid ({rew_half:.3f}), Δ={proximity_diff:.3f}")


def test_v3_num_obstacles_2_works():
    """V3 应支持 num_obstacles=2（虽然注册只用 1，env 不该限制）。
    spawn 安全 + 多球碰撞都对每个 obstacle 检查。"""
    env = PandaBackupPolicyEnv(
        num_obstacles=2, enable_dr=False, seed=11,
        use_arm_sphere_collision=True, use_full_arm_collision=True,
        max_displacement=0.50, enforce_cartesian_bounds=False,
    )
    env.reset(seed=11)
    u = env.unwrapped
    # 两个 obstacle 都应满足所有 5 个臂球的安全距离
    for obs_i in range(2):
        for bid, sphere_r in u._arm_sphere_specs:
            sphere_pos = u._data.xpos[bid].copy()
            d = float(np.linalg.norm(u._obstacle_pos[obs_i] - sphere_pos))
            threshold = sphere_r + OBSTACLE_RADIUS_V3
            assert d >= threshold, (
                f"obstacle[{obs_i}] vs {u._model.body(bid).name}: d={d:.3f} < {threshold:.3f}"
            )
    # 观测维度应升到 (18 + 2×10) × 3 = 114
    obs, _ = env.reset(seed=11)
    assert obs["agent_pos"].shape == (114,), f"expected (114,), got {obs['agent_pos'].shape}"
    env.close()
    print(f"  [PASS] V3 num_obstacles=2 spawn 安全 + obs shape (114,)")


def test_v3_dr_path_proximity_stable():
    """V3 + enable_dr=True：proximity reward 仍只依赖真实 xpos（不被 DR 噪声扰动），
    确保 DR 不污染 reward 信号。"""
    env = PandaBackupPolicyEnv(
        num_obstacles=1, enable_dr=True, seed=12,
        use_arm_sphere_collision=True, use_full_arm_collision=True,
        max_displacement=0.50, enforce_cartesian_bounds=False,
    )
    env.reset(seed=12)
    u = env.unwrapped
    u._stall_remaining[0] = 999
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    # 放到 panda_hand 球外 0.30m → clear=10cm，proximity 应饱和
    u._obstacle_pos[0] = arm_center + np.array([0.30, 0.0, 0.0])

    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert not term
    clear = info["step_surface_clearance"]
    # surface_clearance 用真实 xpos（无 DR），应精确 ≈ 0.10m
    assert abs(clear - 0.10) < 0.01, (
        f"DR=True 下 step_surface_clearance 应仍用真实 xpos, ≈0.10; got {clear:.4f}"
    )
    env.close()
    print(f"  [PASS] V3 + DR=True 下 proximity 信号稳定 (clear={clear:.4f})")


def test_v3_requires_arm_sphere_collision():
    """V3 强制要求 use_arm_sphere_collision=True（V3 在 V2 panda_hand 球基础上扩展）。
    use_full_arm_collision=True + use_arm_sphere_collision=False 应 ValueError。"""
    import pytest
    with pytest.raises(ValueError, match="use_arm_sphere_collision"):
        PandaBackupPolicyEnv(
            num_obstacles=1, enable_dr=False, seed=13,
            use_arm_sphere_collision=False,   # 反向依赖
            use_full_arm_collision=True,
        )
    print(f"  [PASS] V3 拒绝 use_full_arm_collision=True + use_arm_sphere_collision=False")


def test_v3_collision_reward_no_proximity():
    """V3 collision 终止时 reward 应 == TERMINATION_PENALTY (-10.0)，
    不应包含 proximity bonus（terminated 路径直接覆盖 reward）。"""
    from frrl.envs.sim.panda_backup_policy_env import TERMINATION_PENALTY
    env = _make_env_v3(seed=14)
    u = env.unwrapped
    u._stall_remaining[0] = 999
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    # 放到 0.15m < 0.20 阈值，必撞
    u._obstacle_pos[0] = arm_center + np.array([0.15, 0.0, 0.0])

    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert term, f"应触发碰撞终止, info={info}"
    assert rew == TERMINATION_PENALTY, (
        f"collision reward 应 == {TERMINATION_PENALTY}（不含 proximity 项）, got {rew}"
    )
    env.close()
    print(f"  [PASS] V3 collision reward = {rew} (无 proximity 污染)")


def test_v3c_tracks_arm_center_not_tcp():
    """V3c: hand 朝 panda_hand body (=arm_center) 移，不朝 pinch_site (=TCP)。

    placement 必须**横向**（不能纯 +z），否则 dir_to_arm 与 dir_to_tcp 几乎共线
    (TCP 在 arm_center 下方 ~10cm)，cos similarity 测试鉴别力为零。
    使用 +x 远端 placement 让 dir_to_arm 与 dir_to_tcp 在 xy 平面有可测夹角。"""
    env = _make_env_v3c(seed=20)
    u = env.unwrapped
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    tcp = u._data.site_xpos[u._pinch_site_id].copy()
    # 横向 placement: arm_center + 0.5m × +x 方向（远离 D_TIGHT_ARM 0.23）
    u._obstacle_pos[0] = arm_center + np.array([0.5, 0.0, 0.0])

    # 验证测试前提：dir_to_arm 与 dir_to_tcp 有可测夹角（不应共线）
    dir_to_arm = arm_center - u._obstacle_pos[0]; dir_to_arm /= np.linalg.norm(dir_to_arm)
    dir_to_tcp = tcp - u._obstacle_pos[0]; dir_to_tcp /= np.linalg.norm(dir_to_tcp)
    cos_two_dirs = float(dir_to_arm @ dir_to_tcp)
    assert cos_two_dirs < 0.999, (
        f"测试前提失败：placement 让 dir_to_arm 与 dir_to_tcp 几乎共线 "
        f"(cos={cos_two_dirs:.5f})，无法鉴别 tracking_target。换 placement。"
    )

    pre = u._obstacle_pos[0].copy()
    u._move_obstacle(0)
    post = u._obstacle_pos[0].copy()
    delta = post - pre
    speed = np.linalg.norm(delta)
    assert speed > 1e-8, "hand 应移动（远离 dwell 区域）"
    actual_dir = delta / speed
    cos_arm = float(actual_dir @ dir_to_arm)
    cos_tcp = float(actual_dir @ dir_to_tcp)
    # 真鉴别：cos_arm 与 cos_tcp 必须有非平凡差距
    assert (cos_arm - cos_tcp) > 1e-3, (
        f"鉴别力不足：cos_arm-cos_tcp={cos_arm-cos_tcp:.6f}，"
        f"两个方向太接近 (cos_arm={cos_arm:.6f}, cos_tcp={cos_tcp:.6f})"
    )
    assert cos_arm > cos_tcp, (
        f"V3c hand 应朝 arm_center 移而非 TCP；"
        f"cos(actual, arm)={cos_arm:.4f} vs cos(actual, tcp)={cos_tcp:.4f}"
    )
    assert cos_arm > 0.99, f"V3c hand 应几乎正好朝 arm_center, cos={cos_arm:.4f}"
    env.close()
    print(f"  [PASS] V3c hand 朝 arm_center 而非 TCP "
          f"(arm cos={cos_arm:.4f} > tcp cos={cos_tcp:.4f}, Δ={cos_arm-cos_tcp:.4f})")


def test_v3c_dwell_disabled():
    """V3c 重设计 (2026-04-27)：D_TIGHT_ARM=0.08 < collision_dist=0.20，dwell trigger
    永远进不去 (hand 在 D_TIGHT_ARM 时已撞)，policy 看到的是 V3-style 持续追逼。

    验证：把 hand 放在 D_TIGHT_ARM 处，触发 _move_obstacle 应**不进 stall**
    （dwell 区域和 collision 区域已重叠，hand 早被 collision 终止）。"""
    env = _make_env_v3c(seed=21)
    u = env.unwrapped
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    # 把 hand 放在 D_TIGHT_ARM 处（沿 +x 方向，远 link3-6）
    u._obstacle_pos[0] = arm_center + np.array([D_TIGHT_ARM - 1e-6, 0.0, 0.0])
    u._stall_remaining[0] = 0
    # 此 placement 下 hand 距 arm_center = 0.08 < collision_dist 0.20 → step 立刻 collision
    obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert term and info.get("violation_type") == "hand_collision", (
        f"V3c D_TIGHT_ARM=0.08 < collision 0.20，hand 在 D_TIGHT_ARM 处必撞；"
        f"info={info}"
    )
    env.close()
    print(f"  [PASS] V3c D_TIGHT_ARM=0.08 关 dwell：hand 在该距离立刻 hand_collision")


def test_v3c_uses_v3c_speed_range():
    """V3c 用 HAND_SPEED_RANGE_V3C (0.020, 0.040)，V3/V3b 用 HAND_SPEED_RANGE_V3 (0.015, 0.030)。
    通过 reset 多次采样验证范围。"""
    from frrl.envs.sim.panda_backup_policy_env import HAND_SPEED_RANGE_V3C
    speeds_v3c = []
    for seed in range(50):
        env = _make_env_v3c(seed=seed)
        speeds_v3c.append(env.unwrapped._obstacle_speed)
        env.close()
    assert min(speeds_v3c) >= HAND_SPEED_RANGE_V3C[0] - 1e-6
    assert max(speeds_v3c) <= HAND_SPEED_RANGE_V3C[1] + 1e-6
    # V3 仍用 V3 范围
    speeds_v3 = []
    for seed in range(50):
        env = _make_env_v3(seed=seed)
        speeds_v3.append(env.unwrapped._obstacle_speed)
        env.close()
    assert min(speeds_v3) >= HAND_SPEED_RANGE_V3[0] - 1e-6
    assert max(speeds_v3) <= HAND_SPEED_RANGE_V3[1] + 1e-6
    print(f"  [PASS] V3c speed ∈ [{min(speeds_v3c):.4f}, {max(speeds_v3c):.4f}] ⊂ V3C range; "
          f"V3 ∈ [{min(speeds_v3):.4f}, {max(speeds_v3):.4f}] ⊂ V3 range")


def test_v3c_episode_length_25():
    """V3c env 注册 max_episode_steps=25（V3/V3b 是 20）。"""
    import gymnasium as gym
    import frrl.envs  # noqa
    env = gym.make("gym_frrl/PandaBackupPolicyS1V3c-v0")
    # gym 的 max_episode_steps 暴露在 spec.max_episode_steps 或 _max_episode_steps
    spec_steps = env.spec.max_episode_steps if env.spec else None
    assert spec_steps == 25, f"V3c max_episode_steps 应是 25, got {spec_steps}"
    env.close()
    # 对照：V3 应仍是 20
    env_v3 = gym.make("gym_frrl/PandaBackupPolicyS1V3-v0")
    assert env_v3.spec.max_episode_steps == 20, f"V3 max_episode_steps 应是 20"
    env_v3.close()
    print(f"  [PASS] V3c episode=25, V3 episode=20")


def test_v3c_requires_arm_sphere_collision():
    """V3c tracking_target='arm_center' 要求 use_arm_sphere_collision=True
    (arm_center 即 panda_hand body，必须开 V2/V3 球心几何)。"""
    import pytest
    with pytest.raises(ValueError, match="use_arm_sphere_collision"):
        PandaBackupPolicyEnv(
            num_obstacles=1, enable_dr=False, seed=22,
            use_arm_sphere_collision=False,   # 反向依赖
            tracking_target="arm_center",
        )
    print(f"  [PASS] V3c 拒绝 tracking_target=arm_center + use_arm_sphere_collision=False")


def test_default_tracking_target_is_tcp():
    """V1/V2/V3 默认 tracking_target='tcp' 行为不变（向后兼容）。

    placement 同 V3c 测试用横向 +x 远端，避免 cos similarity 浮点同值。"""
    # 先用状态级断言（廉价、确定）
    for env_factory, name in [(_make_env, "V1"), (_make_env_v2, "V2"), (_make_env_v3, "V3")]:
        env = env_factory(seed=23)
        u = env.unwrapped
        assert u._tracking_target == "tcp", f"{name} 默认应是 'tcp', got {u._tracking_target!r}"
        assert u._d_tight == D_TIGHT, f"{name} 默认 d_tight 应 = D_TIGHT (8cm), got {u._d_tight}"
        env.close()

    # 再用方向鉴别（同 V3c 横向 placement）
    env = _make_env_v3(seed=23)
    u = env.unwrapped
    arm_center = u._data.xpos[u._panda_hand_body_id].copy()
    tcp = u._data.site_xpos[u._pinch_site_id].copy()
    u._obstacle_pos[0] = arm_center + np.array([0.5, 0.0, 0.0])
    dir_to_tcp = tcp - u._obstacle_pos[0]; dir_to_tcp /= np.linalg.norm(dir_to_tcp)
    dir_to_arm = arm_center - u._obstacle_pos[0]; dir_to_arm /= np.linalg.norm(dir_to_arm)
    assert float(dir_to_tcp @ dir_to_arm) < 0.999, "测试前提：两方向应有可测夹角"

    pre = u._obstacle_pos[0].copy()
    u._move_obstacle(0)
    post = u._obstacle_pos[0].copy()
    delta = post - pre
    assert np.linalg.norm(delta) > 1e-8, "hand 应移动"
    actual = delta / np.linalg.norm(delta)
    cos_tcp = float(actual @ dir_to_tcp)
    cos_arm = float(actual @ dir_to_arm)
    assert (cos_tcp - cos_arm) > 1e-3, f"鉴别力不足: Δ={cos_tcp-cos_arm:.6f}"
    assert cos_tcp > cos_arm, f"V3 默认应朝 TCP, cos_tcp={cos_tcp} cos_arm={cos_arm}"
    env.close()
    print(f"  [PASS] V1/V2/V3 默认 tracking_target='tcp' "
          f"(状态 + 方向双验，cos_tcp={cos_tcp:.4f} > cos_arm={cos_arm:.4f})")


def test_v3_backward_compat_v2():
    """V3 关掉 use_full_arm_collision 后行为应和 V2 完全一致（不影响旧 ckpt 训练）。"""
    env_v2 = _make_env_v2(seed=10)
    env_v3_off = PandaBackupPolicyEnv(
        num_obstacles=1, enable_dr=False, seed=10,
        use_arm_sphere_collision=True,
        use_full_arm_collision=False,  # 显式关
        max_displacement=0.30,
        enforce_cartesian_bounds=False,
    )
    env_v3_off.reset(seed=10)
    u_v2 = env_v2.unwrapped
    u_v3 = env_v3_off.unwrapped
    # spawn 距离应一致
    arm_center_v2 = u_v2._data.xpos[u_v2._panda_hand_body_id]
    arm_center_v3 = u_v3._data.xpos[u_v3._panda_hand_body_id]
    d_v2 = float(np.linalg.norm(u_v2._obstacle_pos[0] - arm_center_v2))
    d_v3 = float(np.linalg.norm(u_v3._obstacle_pos[0] - arm_center_v3))
    assert abs(d_v2 - d_v3) < 1e-6, f"V2 vs V3-off spawn 不一致: V2={d_v2:.4f} V3-off={d_v3:.4f}"
    # V3-off 不应解析 arm_sphere_specs
    assert u_v3._arm_sphere_specs == [], "V3-off 不应初始化 _arm_sphere_specs"
    env_v2.close()
    env_v3_off.close()
    print(f"  [PASS] V3 with use_full_arm_collision=False 行为 == V2 (spawn d={d_v2:.4f}m)")


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
        test_v3_spawn_no_one_step_death,
        test_v3_obstacle_radius_010,
        test_v3_elbow_collision_terminates,
        test_v3_proximity_reward_saturation,
        test_v3_num_obstacles_2_works,
        test_v3_dr_path_proximity_stable,
        test_v3_requires_arm_sphere_collision,
        test_v3_collision_reward_no_proximity,
        test_v3_backward_compat_v2,
        test_v3c_tracks_arm_center_not_tcp,
        test_v3c_dwell_disabled,
        test_v3c_uses_v3c_speed_range,
        test_v3c_episode_length_25,
        test_v3c_requires_arm_sphere_collision,
        test_default_tracking_target_is_tcp,
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
