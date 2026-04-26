"""HierarchicalSupervisor：Task ↔ Backup ↔ Homing 三态状态机。

设计要点：
  1. 三态：TASK / BACKUP / HOMING
  2. 滞回阈值 d_safe < d_clear，避免阈值附近模式抖动
  3. BACKUP → HOMING 需要连续 N 步手距 > d_clear（防短暂离开误触发）
  4. 任何时候手距 < d_safe 都立即切回 BACKUP（HOMING 期间手回来 → BACKUP）
  5. HOMING → TASK 条件：位置偏差 < pos_tol AND 姿态偏差角 < rot_tol
  6. `tcp_start_pos` / `tcp_start_quat` 在 TASK→BACKUP 转移时记录**一次**，
     BACKUP↔HOMING 来回切时不更新，回到 TASK 后清空。保证 homing 目标永远是
     task 被打断瞬间的 6D 位姿。

使用流程（外部调用者每步）：
    supervisor.step(min_hand_dist, tcp_current_pos, tcp_current_quat)
    action = supervisor.select_action(
        task_action_fn, backup_action_fn, tcp_current_pos, tcp_current_quat
    )

Task/Backup policy 返回的是 env 能接受的 7D action，supervisor 仅做路由。
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from frrl.rl.supervisor.homing import HomingController


class Mode(Enum):
    TASK = auto()
    BACKUP = auto()
    HOMING = auto()


class HierarchicalSupervisor:
    """三态滞回 supervisor（6D 位姿恢复）。

    Args:
        d_safe: 手距 < 该值 → 切入 BACKUP，m。默认 0.10。
        d_clear: 手距 > 该值 → 允许 BACKUP 退出，m。默认 0.20。
        clear_n_steps: BACKUP → HOMING 需要连续多少步 d > d_clear。默认 3。
        homing_pos_tol: HOMING → TASK 的位置容差，m。默认 0.02。
        homing_rot_tol: HOMING → TASK 的姿态容差，rad。默认 0.05 ≈ 2.9°。
        homing_kp_pos / homing_kp_rot: Homing P 控制器增益。
        homing_action_scale / homing_rot_action_scale: 与 env 一致。
    """

    def __init__(
        self,
        d_safe: float = 0.10,
        d_clear: float = 0.20,
        clear_n_steps: int = 3,
        homing_pos_tol: float = 0.02,
        homing_rot_tol: float = 0.05,
        homing_kp_pos: float = 1.0,
        homing_kp_rot: float = 1.0,
        homing_action_scale: float = 0.03,
        homing_rot_action_scale: float = 0.1,
        homing_done_consecutive_n: int = 1,
    ):
        assert d_safe < d_clear, f"需要 d_safe ({d_safe}) < d_clear ({d_clear})"
        self._d_safe = float(d_safe)
        self._d_clear = float(d_clear)
        self._clear_n_steps = int(clear_n_steps)

        self._homing = HomingController(
            kp_pos=homing_kp_pos,
            kp_rot=homing_kp_rot,
            pos_tol=homing_pos_tol,
            rot_tol=homing_rot_tol,
            action_scale=homing_action_scale,
            rot_action_scale=homing_rot_action_scale,
            done_consecutive_n=homing_done_consecutive_n,
        )

        self._mode: Mode = Mode.TASK
        self._tcp_start_pos: Optional[np.ndarray] = None   # shape: (3,)
        self._tcp_start_quat: Optional[np.ndarray] = None  # shape: (4,) [w,x,y,z]
        self._clear_count: int = 0

    # ================================================================
    # 状态查询
    # ================================================================

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def tcp_start(self) -> Optional[np.ndarray]:
        """task 被打断时记录的 TCP 位置；TASK 期间为 None。向后兼容别名。"""
        return None if self._tcp_start_pos is None else self._tcp_start_pos.copy()

    @property
    def tcp_start_pos(self) -> Optional[np.ndarray]:
        return None if self._tcp_start_pos is None else self._tcp_start_pos.copy()

    @property
    def tcp_start_quat(self) -> Optional[np.ndarray]:
        return None if self._tcp_start_quat is None else self._tcp_start_quat.copy()

    @property
    def clear_count(self) -> int:
        """当前 BACKUP 下连续 d>d_clear 的步数计数（仅调试用）。"""
        return self._clear_count

    @property
    def homing_done_streak(self) -> int:
        """HOMING 下当前连续 within-tol 的帧数（调试用）。"""
        return self._homing.done_streak

    @property
    def homing_done_consecutive_n(self) -> int:
        """HOMING 完成需要的连续 within-tol 帧数（vis 显示用）。"""
        return self._homing._done_consecutive_n

    # ================================================================
    # FSM
    # ================================================================

    def reset(self) -> None:
        """Episode 开始时调用，回到 TASK 态。"""
        self._mode = Mode.TASK
        self._tcp_start_pos = None
        self._tcp_start_quat = None
        self._clear_count = 0

    def step(
        self,
        min_hand_dist: float,
        tcp_current_pos: np.ndarray,
        tcp_current_quat: np.ndarray,
    ) -> Mode:
        """更新 FSM 状态；返回新状态。

        Args:
            min_hand_dist: 当前步手到 TCP 的最近距离，m。
            tcp_current_pos: 当前 TCP 位置，shape (3,) m。
            tcp_current_quat: 当前 TCP 四元数，shape (4,) [w, x, y, z]。
        """
        tcp_pos = np.asarray(tcp_current_pos, dtype=np.float32).reshape(3)
        tcp_quat = np.asarray(tcp_current_quat, dtype=np.float64).reshape(4)

        if self._mode == Mode.TASK:
            if min_hand_dist < self._d_safe:
                # TASK → BACKUP：记录 tcp_start 一次
                self._tcp_start_pos = tcp_pos.copy()
                q = tcp_quat.copy()
                q /= np.linalg.norm(q) + 1e-12
                self._tcp_start_quat = q
                self._clear_count = 0
                self._mode = Mode.BACKUP
        elif self._mode == Mode.BACKUP:
            if min_hand_dist > self._d_clear:
                self._clear_count += 1
                if self._clear_count >= self._clear_n_steps:
                    # BACKUP → HOMING：加载 6D 目标到 homing controller
                    assert self._tcp_start_pos is not None
                    assert self._tcp_start_quat is not None
                    self._homing.reset(self._tcp_start_pos, self._tcp_start_quat)
                    self._mode = Mode.HOMING
            else:
                # 手又靠近（或仍在滞回区），清零计数
                self._clear_count = 0
        elif self._mode == Mode.HOMING:
            if min_hand_dist < self._d_safe:
                # HOMING 期间手回来 → 切回 BACKUP；tcp_start 保持不变
                self._clear_count = 0
                self._mode = Mode.BACKUP
            elif self._homing.is_done(tcp_pos, tcp_quat):
                # HOMING 完成（位置 + 姿态都到位）→ 回到 TASK，清空 tcp_start
                self._tcp_start_pos = None
                self._tcp_start_quat = None
                self._clear_count = 0
                self._mode = Mode.TASK

        return self._mode

    # ================================================================
    # 动作路由
    # ================================================================

    def select_action(
        self,
        task_action_fn: Callable[[], np.ndarray],
        backup_action_fn: Callable[[], np.ndarray],
        tcp_current_pos: np.ndarray,
        tcp_current_quat: np.ndarray,
    ) -> np.ndarray:
        """按当前 mode 路由动作来源。

        Args:
            task_action_fn: 返回 task policy 的 7D action（仅 TASK 模式调用）。
            backup_action_fn: 返回 backup policy 的 7D action（仅 BACKUP 模式调用）。
            tcp_current_pos: 当前 TCP 位置 (3,)，HOMING 用。
            tcp_current_quat: 当前 TCP 四元数 (4,)，HOMING 用。

        使用 callable 而不是直接传 action：延迟计算，避免不需要的 policy forward。
        """
        if self._mode == Mode.TASK:
            return np.asarray(task_action_fn(), dtype=np.float32)
        if self._mode == Mode.BACKUP:
            return np.asarray(backup_action_fn(), dtype=np.float32)
        # HOMING
        return self._homing.get_action(tcp_current_pos, tcp_current_quat)
