"""Backup Env TRACKING-only 改造冒烟测试。

Usage:
    python scripts/test_backup_env_tracking.py
"""
import sys
import numpy as np

from frrl.envs.panda_backup_policy_env import (
    PandaBackupPolicyEnv,
    MotionMode,
    D_TIGHT,
    STALL_STEPS_RANGE,
    HAND_SPAWN_DIST,
    MAX_EPISODE_STEPS,
    ROT_ACTION_SCALE,
)


def _make_env(num_obstacles=1, seed=0):
    env = PandaBackupPolicyEnv(
        num_obstacles=num_obstacles,
        enable_dr=False,  # 关掉 DR，便于确定性断言
        seed=seed,
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
    # 把手放到 TCP 前方已知距离 25cm
    tcp = env.unwrapped._data.site_xpos[env.unwrapped._pinch_site_id].copy()  # shape: (3,)
    env.unwrapped._obstacle_pos[0] = tcp + np.array([0.25, 0.0, 0.0])  # shape: (3,)
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
