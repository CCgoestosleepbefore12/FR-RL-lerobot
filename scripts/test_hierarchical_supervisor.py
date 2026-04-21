"""HierarchicalSupervisor 三态 FSM 单测（6D）。

Usage:
    python scripts/test_hierarchical_supervisor.py
"""
import sys
import numpy as np

from frrl.rl.hierarchical_supervisor import HierarchicalSupervisor, Mode


Q_I = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 单位四元数


def _make(d_safe=0.10, d_clear=0.20, clear_n_steps=3, homing_pos_tol=0.02):
    return HierarchicalSupervisor(
        d_safe=d_safe, d_clear=d_clear,
        clear_n_steps=clear_n_steps, homing_pos_tol=homing_pos_tol,
    )


def test_init_state():
    s = _make()
    assert s.mode == Mode.TASK
    assert s.tcp_start is None
    assert s.tcp_start_quat is None
    assert s.clear_count == 0
    print("  [PASS] init state = TASK, tcp_start=None")


def test_task_to_backup_records_tcp_start():
    s = _make()
    tcp = np.array([0.3, -0.1, 0.2], dtype=np.float32)
    s.step(min_hand_dist=0.25, tcp_current_pos=tcp, tcp_current_quat=Q_I)  # 手远
    assert s.mode == Mode.TASK

    s.step(min_hand_dist=0.08, tcp_current_pos=tcp, tcp_current_quat=Q_I)
    assert s.mode == Mode.BACKUP
    assert np.allclose(s.tcp_start_pos, tcp), f"tcp_start_pos={s.tcp_start_pos}"
    assert np.allclose(s.tcp_start_quat, Q_I), f"tcp_start_quat={s.tcp_start_quat}"
    print(f"  [PASS] TASK→BACKUP records tcp_start={s.tcp_start_pos}, quat={s.tcp_start_quat}")


def test_backup_stays_when_hand_near():
    s = _make()
    tcp = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    s.step(0.08, tcp, Q_I)  # TASK → BACKUP
    for _ in range(10):
        s.step(0.08, tcp, Q_I)
        assert s.mode == Mode.BACKUP
    print("  [PASS] BACKUP holds while hand near")


def test_backup_to_homing_needs_n_clear_steps():
    s = _make(clear_n_steps=3)
    tcp0 = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    s.step(0.08, tcp0, Q_I)  # TASK → BACKUP
    assert s.mode == Mode.BACKUP

    # 2 步 clear 不够
    s.step(0.25, tcp0, Q_I); assert s.mode == Mode.BACKUP
    s.step(0.25, tcp0, Q_I); assert s.mode == Mode.BACKUP
    # 第 3 步触发
    s.step(0.25, tcp0, Q_I); assert s.mode == Mode.HOMING
    print("  [PASS] BACKUP→HOMING after 3 clear steps")


def test_clear_count_resets_on_near():
    """clear 中途手回来 clear_count 清零，需要重新计数。"""
    s = _make(clear_n_steps=3)
    tcp = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    s.step(0.05, tcp, Q_I)  # → BACKUP
    s.step(0.25, tcp, Q_I); assert s.clear_count == 1
    s.step(0.25, tcp, Q_I); assert s.clear_count == 2
    s.step(0.15, tcp, Q_I); assert s.clear_count == 0 and s.mode == Mode.BACKUP  # 滞回区
    s.step(0.25, tcp, Q_I); assert s.clear_count == 1 and s.mode == Mode.BACKUP
    s.step(0.25, tcp, Q_I); assert s.clear_count == 2 and s.mode == Mode.BACKUP
    s.step(0.25, tcp, Q_I); assert s.mode == Mode.HOMING
    print("  [PASS] clear_count resets on d dropping back")


def test_homing_to_task_on_done():
    s = _make(clear_n_steps=1, homing_pos_tol=0.02)
    tcp_start = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    tcp_far = np.array([0.25, 0.0, 0.2], dtype=np.float32)  # 5cm 偏差

    s.step(0.05, tcp_start, Q_I)   # TASK → BACKUP，记 tcp_start=0.3
    s.step(0.25, tcp_far, Q_I)     # BACKUP → HOMING
    assert s.mode == Mode.HOMING
    # 5cm > 2cm，HOMING 保持
    s.step(0.25, tcp_far, Q_I); assert s.mode == Mode.HOMING
    # 回到 tcp_start 附近，姿态也对齐
    s.step(0.25, tcp_start + np.array([0.01, 0, 0]), Q_I)
    assert s.mode == Mode.TASK
    assert s.tcp_start is None and s.tcp_start_quat is None
    print("  [PASS] HOMING→TASK on done, tcp_start cleared")


def test_homing_interrupted_by_hand_return():
    """HOMING 期间手回来 → 切回 BACKUP，tcp_start 保持不变。"""
    s = _make(clear_n_steps=1)
    tcp_start = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    tcp_far = np.array([0.25, 0.0, 0.2], dtype=np.float32)

    s.step(0.05, tcp_start, Q_I)   # → BACKUP
    original_tcp_start = s.tcp_start_pos.copy()
    s.step(0.25, tcp_far, Q_I)     # → HOMING
    assert s.mode == Mode.HOMING

    s.step(0.05, tcp_far, Q_I)     # 手回来 → BACKUP
    assert s.mode == Mode.BACKUP
    assert np.allclose(s.tcp_start_pos, original_tcp_start), \
        "tcp_start 不应在 HOMING→BACKUP 时被覆盖"

    # 再次等待 clear → HOMING，目标仍然是原 tcp_start
    s.step(0.25, tcp_far, Q_I); assert s.mode == Mode.HOMING
    # 回到原 tcp_start 附近，HOMING→TASK
    s.step(0.25, tcp_start, Q_I)
    assert s.mode == Mode.TASK
    print(f"  [PASS] HOMING→BACKUP preserves tcp_start={original_tcp_start}")


def test_tcp_start_updates_only_on_task_to_backup():
    """第二次 TASK→BACKUP 才更新 tcp_start。"""
    s = _make(clear_n_steps=1)
    tcp_1 = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    tcp_2 = np.array([0.4, 0.1, 0.25], dtype=np.float32)

    # 第一轮
    s.step(0.05, tcp_1, Q_I); assert np.allclose(s.tcp_start_pos, tcp_1)
    s.step(0.25, tcp_1, Q_I); assert s.mode == Mode.HOMING
    s.step(0.25, tcp_1, Q_I); assert s.mode == Mode.TASK
    assert s.tcp_start is None

    # 第二轮：从 TCP=tcp_2 处再次被打断
    s.step(0.05, tcp_2, Q_I)
    assert np.allclose(s.tcp_start_pos, tcp_2), f"tcp_start={s.tcp_start_pos}, expected {tcp_2}"
    print("  [PASS] tcp_start updates on each TASK→BACKUP")


def test_quat_recorded_and_used_for_homing_done():
    """tcp_start_quat 记录后，姿态不一致时 HOMING 不完成。"""
    q_off = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0], dtype=np.float64)  # 绕x转0.2rad
    q_off /= np.linalg.norm(q_off)
    s = _make(clear_n_steps=1)
    tcp = np.array([0.3, 0.0, 0.2], dtype=np.float32)

    # TASK→BACKUP 时记录 quat=Q_I
    s.step(0.05, tcp, Q_I)
    assert np.allclose(s.tcp_start_quat, Q_I)

    # → HOMING
    s.step(0.25, tcp, Q_I); assert s.mode == Mode.HOMING

    # 位置匹配但姿态偏 0.2 rad > 默认 rot_tol 0.05 → HOMING 保持
    s.step(0.25, tcp, q_off); assert s.mode == Mode.HOMING
    # 姿态也对齐 → HOMING→TASK
    s.step(0.25, tcp, Q_I); assert s.mode == Mode.TASK
    print("  [PASS] HOMING done 要求 pos AND quat 都到位")


def test_select_action_routing():
    """action 路由：TASK 调 task_fn，BACKUP 调 backup_fn，HOMING 用 homing_controller。"""
    s = _make(clear_n_steps=1)
    task_called = [0]
    backup_called = [0]
    def task_fn():
        task_called[0] += 1
        return np.array([0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    def backup_fn():
        backup_called[0] += 1
        return np.array([-0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    tcp = np.array([0.3, 0.0, 0.2], dtype=np.float32)

    # TASK 模式
    a = s.select_action(task_fn, backup_fn, tcp, Q_I)
    assert task_called == [1] and backup_called == [0]
    assert np.allclose(a, [0.1, 0, 0, 0, 0, 0, 0])

    # → BACKUP
    s.step(0.05, tcp, Q_I)
    a = s.select_action(task_fn, backup_fn, tcp, Q_I)
    assert task_called == [1] and backup_called == [1]
    assert np.allclose(a, [-0.5, 0, 0, 0, 0, 0, 0])

    # → HOMING（tcp_start=tcp，当前 tcp 偏移 5cm）
    s.step(0.25, tcp + np.array([0.05, 0, 0]), Q_I)  # → HOMING
    a = s.select_action(task_fn, backup_fn, tcp + np.array([0.05, 0, 0]), Q_I)
    # HOMING 控制器：偏差 (-0.05, 0, 0) → action[:3] = -0.05/0.03 clip 到 -1
    assert task_called == [1] and backup_called == [1], "HOMING 不应调 task/backup fn"
    assert a[0] < 0 and a[0] == -1.0
    assert np.allclose(a[3:], 0.0)
    print("  [PASS] action routing + lazy evaluation")


def test_reset():
    s = _make(clear_n_steps=1)
    tcp = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    s.step(0.05, tcp, Q_I)  # → BACKUP
    assert s.tcp_start is not None
    s.reset()
    assert s.mode == Mode.TASK
    assert s.tcp_start is None and s.tcp_start_quat is None
    assert s.clear_count == 0
    print("  [PASS] reset() restores initial state")


def test_d_safe_must_lt_d_clear():
    try:
        HierarchicalSupervisor(d_safe=0.20, d_clear=0.10)
    except AssertionError:
        print("  [PASS] d_safe >= d_clear rejected")
        return
    raise AssertionError("expected AssertionError")


def main():
    print("=== HierarchicalSupervisor 6D 测试 ===")
    tests = [
        test_init_state,
        test_task_to_backup_records_tcp_start,
        test_backup_stays_when_hand_near,
        test_backup_to_homing_needs_n_clear_steps,
        test_clear_count_resets_on_near,
        test_homing_to_task_on_done,
        test_homing_interrupted_by_hand_return,
        test_tcp_start_updates_only_on_task_to_backup,
        test_quat_recorded_and_used_for_homing_done,
        test_select_action_routing,
        test_reset,
        test_d_safe_must_lt_d_clear,
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
