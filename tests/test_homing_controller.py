"""HomingController 6D 单测。

Usage:
    pytest tests/test_homing_controller.py
"""
import sys
import numpy as np

from frrl.rl.supervisor import (
    HomingController,
    quat_conjugate,
    quat_multiply,
    quat_to_axis_angle,
)


Q_I = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 单位四元数 [w,x,y,z]


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """辅助：轴角 → 四元数 [w,x,y,z]。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.array([
        np.cos(angle / 2.0),
        *(axis * np.sin(angle / 2.0)),
    ], dtype=np.float64)


# ============================================================
# 模块级 quat 工具函数
# ============================================================


def test_quat_conjugate():
    q = np.array([0.7, 0.1, 0.2, 0.3])
    assert np.allclose(quat_conjugate(q), [0.7, -0.1, -0.2, -0.3])
    print("  [PASS] quat_conjugate")


def test_quat_multiply_identity():
    q = np.array([0.7071, 0.7071, 0.0, 0.0])
    assert np.allclose(quat_multiply(Q_I, q), q)
    assert np.allclose(quat_multiply(q, Q_I), q)
    print("  [PASS] quat_multiply identity")


def test_quat_to_axis_angle_identity():
    assert np.allclose(quat_to_axis_angle(Q_I), 0.0)
    print("  [PASS] quat_to_axis_angle(I) = 0")


def test_quat_to_axis_angle_roundtrip():
    """随机轴角 → 四元数 → axis-angle，应与原值一致（短路径范围内）。"""
    rng = np.random.RandomState(0)
    for _ in range(20):
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi + 0.1, np.pi - 0.1)
        q = _quat_from_axis_angle(axis, angle)
        aa = quat_to_axis_angle(q)
        # 轴可能翻转符号（短路径），但 axis*angle 应一致
        expected = axis * angle
        # 允许 (axis,angle) ↔ (-axis,-angle) 等价
        assert np.allclose(aa, expected, atol=1e-5) or np.allclose(aa, -expected, atol=1e-5), \
            f"aa={aa}, expected={expected}"
    print("  [PASS] axis-angle roundtrip")


# ============================================================
# HomingController 6D
# ============================================================


def test_reset_required():
    """reset 未调用时 get_action 应 raise。"""
    h = HomingController()
    try:
        h.get_action(np.zeros(3), Q_I)
    except RuntimeError:
        print("  [PASS] get_action requires reset()")
        return
    raise AssertionError("expected RuntimeError")


def test_large_pos_delta_saturates():
    """位置偏差远大于 action_scale 时输出饱和到 ±1。"""
    h = HomingController(kp_pos=1.0, action_scale=0.03)
    h.reset(np.array([1.0, 0.0, 0.0]), Q_I)
    action = h.get_action(np.zeros(3), Q_I)  # 偏差 1m
    assert action.shape == (7,)
    assert np.allclose(action[:3], [1.0, 0.0, 0.0]), f"action[:3]={action[:3]}"
    assert np.allclose(action[3:6], 0.0), "姿态对齐时 rx/ry/rz 应为 0"
    assert action[6] == 0.0, "夹爪应恒 0"
    print(f"  [PASS] large pos delta saturates → action[:3]={action[:3]}")


def test_small_pos_delta_proportional():
    """小位置偏差按比例输出。"""
    h = HomingController(kp_pos=1.0, action_scale=0.03)
    h.reset(np.array([0.01, 0.0, 0.0]), Q_I)
    action = h.get_action(np.zeros(3), Q_I)
    expected = 0.01 / 0.03
    assert abs(action[0] - expected) < 1e-6, f"action[0]={action[0]}, expected={expected}"
    assert np.allclose(action[1:6], 0.0)
    print(f"  [PASS] proportional pos: 1cm delta → {action[0]:.4f}")


def test_negative_pos_delta():
    h = HomingController(kp_pos=1.0, action_scale=0.03)
    h.reset(np.array([-0.05, 0.0, 0.0]), Q_I)
    action = h.get_action(np.zeros(3), Q_I)
    assert action[0] < 0 and abs(action[0] + 1.0) < 1e-6
    print(f"  [PASS] negative pos delta → action[0]={action[0]:.4f}")


def test_rotation_delta():
    """姿态偏差 → rx/ry/rz 有分量。"""
    # target：绕 x 轴转 0.3 rad；current：identity
    target_q = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.3)
    h = HomingController(kp_rot=1.0, rot_action_scale=0.1)
    h.reset(np.zeros(3), target_q)
    action = h.get_action(np.zeros(3), Q_I)
    # 需要正向绕 x 转 0.3 rad → action[3] = 0.3/0.1 = 3.0 → clip 到 1.0
    assert action[3] > 0 and abs(action[3] - 1.0) < 1e-6
    assert abs(action[4]) < 1e-6 and abs(action[5]) < 1e-6
    print(f"  [PASS] rotation delta → action[3:6]={action[3:6]}")


def test_rotation_negative():
    """target 绕 x 轴 -0.03 rad，action[3] 应为 -0.3（不饱和）。"""
    target_q = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -0.03)
    h = HomingController(kp_rot=1.0, rot_action_scale=0.1)
    h.reset(np.zeros(3), target_q)
    action = h.get_action(np.zeros(3), Q_I)
    assert abs(action[3] + 0.3) < 1e-5, f"action[3]={action[3]}"
    print(f"  [PASS] rotation negative → action[3]={action[3]:.4f}")


def test_is_done_both_conditions():
    """位置 OR 姿态有一个超限都 not done。"""
    h = HomingController(pos_tol=0.02, rot_tol=0.05)
    h.reset(np.zeros(3), Q_I)

    # 两者都在容差内
    assert h.is_done(np.array([0.01, 0.0, 0.0]), Q_I) is True
    # 位置超限
    assert h.is_done(np.array([0.03, 0.0, 0.0]), Q_I) is False
    # 姿态超限（0.1 rad > 0.05 rad）
    q_off = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.1)
    assert h.is_done(np.zeros(3), q_off) is False
    print("  [PASS] is_done requires pos AND rot within tol")


def test_is_done_consecutive_n_requires_streak():
    """N=3：连续 N-1 帧 in_tol 仍 False，第 N 帧才 True。"""
    h = HomingController(pos_tol=0.02, rot_tol=0.05, done_consecutive_n=3)
    h.reset(np.zeros(3), Q_I)
    in_tol_pos = np.array([0.01, 0.0, 0.0])

    # Frame 1, 2: in tol but streak < 3
    assert h.is_done(in_tol_pos, Q_I) is False, "frame 1 should not be done"
    assert h.done_streak == 1
    assert h.is_done(in_tol_pos, Q_I) is False, "frame 2 should not be done"
    assert h.done_streak == 2
    # Frame 3: streak hits 3 → True
    assert h.is_done(in_tol_pos, Q_I) is True, "frame 3 should be done"
    assert h.done_streak == 3
    print("  [PASS] N=3 streak gate works")


def test_is_done_streak_resets_on_out_of_tol():
    """中途 out-of-tol 重置 streak，必须重新积满 N 帧。"""
    h = HomingController(pos_tol=0.02, rot_tol=0.05, done_consecutive_n=3)
    h.reset(np.zeros(3), Q_I)
    in_tol = np.array([0.01, 0.0, 0.0])
    out_of_tol = np.array([0.05, 0.0, 0.0])  # 0.05 > 0.02

    h.is_done(in_tol, Q_I)
    h.is_done(in_tol, Q_I)
    assert h.done_streak == 2
    # 超限：streak 清零
    assert h.is_done(out_of_tol, Q_I) is False
    assert h.done_streak == 0
    # 重新积满
    assert h.is_done(in_tol, Q_I) is False  # 1
    assert h.is_done(in_tol, Q_I) is False  # 2
    assert h.is_done(in_tol, Q_I) is True   # 3
    print("  [PASS] streak resets on out-of-tol and rebuilds")


def test_is_done_streak_resets_on_reset():
    """reset() 清零 streak，新 episode 从 0 重新积。"""
    h = HomingController(pos_tol=0.02, rot_tol=0.05, done_consecutive_n=3)
    h.reset(np.zeros(3), Q_I)
    in_tol = np.array([0.01, 0.0, 0.0])

    h.is_done(in_tol, Q_I)
    h.is_done(in_tol, Q_I)
    assert h.done_streak == 2
    # 重置 — streak 应该清零
    h.reset(np.zeros(3), Q_I)
    assert h.done_streak == 0
    assert h.is_done(in_tol, Q_I) is False  # streak=1 not 3
    print("  [PASS] reset() clears streak")


def test_is_done_default_n1_backward_compat():
    """N=1 (默认) 等价旧行为：单帧 in_tol 立即返回 True。"""
    h = HomingController(pos_tol=0.02, rot_tol=0.05)  # default n=1
    h.reset(np.zeros(3), Q_I)
    assert h.is_done(np.array([0.01, 0.0, 0.0]), Q_I) is True
    print("  [PASS] N=1 default preserves old behavior")


def test_pos_convergence_simulation():
    """纯位置收敛（姿态恒对齐）。"""
    h = HomingController(kp_pos=1.0, action_scale=0.03, pos_tol=0.02)
    target_pos = np.array([0.10, 0.0, 0.0])
    h.reset(target_pos, Q_I)
    cur = np.zeros(3, dtype=np.float32)
    for i in range(20):
        if h.is_done(cur, Q_I):
            print(f"  [PASS] pos converged in {i} steps, final={cur}")
            return
        action = h.get_action(cur, Q_I)
        cur = cur + action[:3] * 0.03
    raise AssertionError(f"pos did not converge in 20 steps, final={cur}")


def test_rot_convergence_simulation():
    """纯姿态收敛（位置恒对齐）。仿真 env 右乘局部系 dquat（与 backup env 一致）。"""
    h = HomingController(kp_rot=1.0, rot_action_scale=0.1, rot_tol=0.05)
    target_q = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.5)
    h.reset(np.zeros(3), target_q)
    cur_q = Q_I.copy()
    for i in range(30):
        if h.is_done(np.zeros(3), cur_q):
            print(f"  [PASS] rot converged in {i} steps")
            return
        action = h.get_action(np.zeros(3), cur_q)
        rot_vec = action[3:6].astype(np.float64) * 0.1
        ang = float(np.linalg.norm(rot_vec))
        if ang > 1e-8:
            axis = rot_vec / ang
            dq = np.array([np.cos(ang / 2.0), *(axis * np.sin(ang / 2.0))])
            cur_q = quat_multiply(cur_q, dq)  # env 右乘：局部系增量
            cur_q /= np.linalg.norm(cur_q)
    raise AssertionError(f"rot did not converge in 30 steps")


def test_rot_convergence_nonidentity_start():
    """q_cur ≠ I 且 q_target ≠ I 且异轴 — 唯一能暴露局部/世界系 bug 的场景。"""
    h = HomingController(kp_rot=1.0, rot_action_scale=0.1, rot_tol=0.05)
    target_q = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.4)  # 绕 x
    cur_q = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.6)     # 绕 z
    h.reset(np.zeros(3), target_q)
    for i in range(50):
        if h.is_done(np.zeros(3), cur_q):
            print(f"  [PASS] nonidentity rot converged in {i} steps")
            return
        action = h.get_action(np.zeros(3), cur_q)
        rot_vec = action[3:6].astype(np.float64) * 0.1
        ang = float(np.linalg.norm(rot_vec))
        if ang > 1e-8:
            axis = rot_vec / ang
            dq = np.array([np.cos(ang / 2.0), *(axis * np.sin(ang / 2.0))])
            cur_q = quat_multiply(cur_q, dq)  # env 局部系右乘
            cur_q /= np.linalg.norm(cur_q)
    raise AssertionError(f"nonidentity rot did not converge in 50 steps, cur={cur_q}")


def test_no_overshoot_kp1():
    """kp=1 时不应超调（位置一维测试）。"""
    h = HomingController(kp_pos=1.0, action_scale=0.03)
    target = np.array([0.05, 0.0, 0.0])
    h.reset(target, Q_I)
    cur = np.zeros(3, dtype=np.float32)
    max_overshoot = 0.0
    for _ in range(30):
        action = h.get_action(cur, Q_I)
        cur = cur + action[:3] * 0.03
        if cur[0] > target[0]:
            max_overshoot = max(max_overshoot, cur[0] - target[0])
    assert max_overshoot < 1e-6, f"kp=1 shouldn't overshoot, max={max_overshoot}"
    print(f"  [PASS] kp=1 no pos overshoot (max={max_overshoot:.6f})")


def test_tcp_start_properties():
    """tcp_start_pos / tcp_start_quat 返回 copy 不让外部污染内部状态。"""
    h = HomingController()
    assert h.tcp_start_pos is None and h.tcp_start_quat is None
    h.reset(np.array([0.1, 0.2, 0.3]), Q_I)
    p = h.tcp_start_pos
    q = h.tcp_start_quat
    assert p is not None and np.allclose(p, [0.1, 0.2, 0.3])
    assert q is not None and np.allclose(q, Q_I)
    p[0] = 99.0
    q[1] = 99.0
    assert h.tcp_start_pos[0] == 0.1 and h.tcp_start_quat[1] == 0.0, \
        "内部状态被外部修改污染"
    print("  [PASS] tcp_start_pos/quat return safe copies")


def main():
    print("=== HomingController 6D 测试 ===")
    tests = [
        test_quat_conjugate,
        test_quat_multiply_identity,
        test_quat_to_axis_angle_identity,
        test_quat_to_axis_angle_roundtrip,
        test_reset_required,
        test_large_pos_delta_saturates,
        test_small_pos_delta_proportional,
        test_negative_pos_delta,
        test_rotation_delta,
        test_rotation_negative,
        test_is_done_both_conditions,
        test_pos_convergence_simulation,
        test_rot_convergence_simulation,
        test_rot_convergence_nonidentity_start,
        test_no_overshoot_kp1,
        test_tcp_start_properties,
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
