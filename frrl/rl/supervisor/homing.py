"""HomingController：TCP 笛卡尔 6D Pose 比例控制器。

用于 HierarchicalSupervisor 的 HOMING 阶段：Supervisor 从 BACKUP 切换到 HOMING
之后，用这个控制器把 TCP 从被推离的位置/姿态拉回 task 被打断时记录的
`(tcp_start_pos, tcp_start_quat)`，再交还 task policy 续跑。

为什么 6D？
  仿真 env (`PandaBackupPolicyEnv`) 已 override 让姿态增量 rx/ry/rz 生效；
  backup policy 在 6D 动作空间训练并可能学到姿态避让。Homing 必须把姿态也
  恢复到 task 被打断瞬间，否则 task policy 续跑时姿态分布外。

控制律（每维度独立 clipped-proportional）：
    action_pos = clip(kp_pos · (pos_start - pos_cur) / ACTION_SCALE, ±1)
    action_rot = clip(kp_rot · quat_err_axis_angle / ROT_ACTION_SCALE, ±1)
  kp=1 为 clipped-deadbeat（不超调）；>1 会在容差附近震荡。

完成条件：位置偏差 < pos_tol AND 姿态偏差角 < rot_tol。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ── Quaternion 工具（约定 [w, x, y, z]）────────────────────


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """共轭四元数。shape: (4,) → (4,)。"""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 乘法。shape: (4,) × (4,) → (4,)。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """四元数 → axis-angle 向量（方向=轴，模长=角度 rad），取短路径。

    shape: (4,) → (3,)。
    """
    q = np.asarray(q, dtype=np.float64)
    # 短路径：w<0 等价 -q，保证 |angle| <= π
    sign = 1.0 if q[0] >= 0 else -1.0
    xyz = sign * q[1:]                                          # shape: (3,)
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    sin_half = min(sin_half, 1.0)                               # 数值安全
    angle = 2.0 * float(np.arcsin(sin_half))
    axis = xyz / sin_half
    return axis * angle                                          # shape: (3,) rad


class HomingController:
    """TCP 6D Pose P 控制器。

    Args:
        kp_pos: 位置比例增益；1.0 = clipped-deadbeat 不超调。
        kp_rot: 姿态比例增益。
        pos_tol: 位置完成阈值 (m)。
        rot_tol: 姿态完成阈值 (rad)，默认 0.05 rad ≈ 2.9°。
        action_scale: 位置动作 scale，与 env ACTION_SCALE 一致 (m/step)。
        rot_action_scale: 姿态动作 scale，与 env ROT_ACTION_SCALE 一致 (rad/step)。
        max_action: action 归一化上限。
    """

    def __init__(
        self,
        kp_pos: float = 1.0,
        kp_rot: float = 1.0,
        pos_tol: float = 0.02,
        rot_tol: float = 0.05,
        action_scale: float = 0.03,
        rot_action_scale: float = 0.1,
        max_action: float = 1.0,
        done_consecutive_n: int = 1,
    ):
        self._kp_pos = float(kp_pos)
        self._kp_rot = float(kp_rot)
        self._pos_tol = float(pos_tol)
        self._rot_tol = float(rot_tol)
        self._action_scale = float(action_scale)
        self._rot_action_scale = float(rot_action_scale)
        self._max_action = float(max_action)
        # 连续 N 帧 within tol 才判 done。阻抗收尾抖动 ±2-5mm 容易瞬时触发
        # 单帧 done 然后被噪声打回，产生定格误差；N≥3 显著缓解。
        n = int(done_consecutive_n)
        assert n >= 1, f"done_consecutive_n must be >= 1, got {done_consecutive_n}"
        self._done_consecutive_n = n
        self._done_streak = 0
        self._tcp_start_pos: Optional[np.ndarray] = None       # shape: (3,)
        self._tcp_start_quat: Optional[np.ndarray] = None      # shape: (4,) [w,x,y,z]

    def reset(self, tcp_start_pos: np.ndarray, tcp_start_quat: np.ndarray) -> None:
        """设定 homing 目标位姿。由 Supervisor 在 BACKUP→HOMING 转移时调用。"""
        self._tcp_start_pos = (
            np.asarray(tcp_start_pos, dtype=np.float32).reshape(3).copy()
        )
        q = np.asarray(tcp_start_quat, dtype=np.float64).reshape(4).copy()
        q /= np.linalg.norm(q) + 1e-12
        self._tcp_start_quat = q
        self._done_streak = 0

    def _pos_error(self, cur_pos: np.ndarray) -> np.ndarray:
        """shape: (3,) m。"""
        return self._tcp_start_pos - np.asarray(cur_pos, dtype=np.float32).reshape(3)

    def _rot_error_axis_angle(self, cur_quat: np.ndarray) -> np.ndarray:
        """姿态误差 axis-angle，**末端局部系**增量。shape: (3,) rad。

        约定与 backup env (`_apply_rotation_to_mocap`) 一致：env 执行
        `q_new = q_cur · dquat`，即 axis-angle 被解释为局部系增量。
            q_target = q_cur · dq_local   →   dq_local = q_cur⁻¹ · q_target
        """
        q_cur = np.asarray(cur_quat, dtype=np.float64).reshape(4)
        q_cur /= np.linalg.norm(q_cur) + 1e-12
        q_err_local = quat_multiply(quat_conjugate(q_cur), self._tcp_start_quat)
        return quat_to_axis_angle(q_err_local)

    def get_action(
        self,
        tcp_current_pos: np.ndarray,
        tcp_current_quat: np.ndarray,
    ) -> np.ndarray:
        """计算一步 homing 动作。

        Returns:
            action: shape (7,) 归一化到 [-1, 1]；[dx, dy, dz, drx, dry, drz, 0]。
                夹爪维度恒 0。
        """
        if self._tcp_start_pos is None or self._tcp_start_quat is None:
            raise RuntimeError("HomingController.reset() 必须在 get_action 之前调用")

        delta_pos = self._pos_error(tcp_current_pos)                         # (3,) m
        delta_rot = self._rot_error_axis_angle(tcp_current_quat)             # (3,) rad

        action_pos = np.clip(
            self._kp_pos * delta_pos / self._action_scale,
            -self._max_action, self._max_action,
        ).astype(np.float32)
        action_rot = np.clip(
            self._kp_rot * delta_rot / self._rot_action_scale,
            -self._max_action, self._max_action,
        ).astype(np.float32)

        action = np.zeros(7, dtype=np.float32)
        action[:3] = action_pos
        action[3:6] = action_rot
        return action

    def is_done(
        self,
        tcp_current_pos: np.ndarray,
        tcp_current_quat: np.ndarray,
    ) -> bool:
        """位置 < pos_tol AND 姿态 < rot_tol 连续 N 帧才算完成。"""
        if self._tcp_start_pos is None or self._tcp_start_quat is None:
            return False
        pos_err = float(np.linalg.norm(self._pos_error(tcp_current_pos)))
        rot_err = float(np.linalg.norm(self._rot_error_axis_angle(tcp_current_quat)))
        in_tol = pos_err < self._pos_tol and rot_err < self._rot_tol
        if in_tol:
            self._done_streak += 1
        else:
            self._done_streak = 0
        return self._done_streak >= self._done_consecutive_n

    @property
    def done_streak(self) -> int:
        """当前连续 within-tol 的帧数（调试用）。N=1 时该值无诊断意义
        （任意 in_tol 即触发 done，streak 立即被消费）；N>1 才有意义。"""
        return self._done_streak

    @property
    def tcp_start_pos(self) -> Optional[np.ndarray]:
        return None if self._tcp_start_pos is None else self._tcp_start_pos.copy()

    @property
    def tcp_start_quat(self) -> Optional[np.ndarray]:
        return None if self._tcp_start_quat is None else self._tcp_start_quat.copy()
