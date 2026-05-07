"""Franka kinematic geometry helpers."""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Franka Hand kinematic flange→pinch offset (m)。TCP 沿 gripper +z 方向距 flange
# 中心 ~10.34cm。HierarchicalSupervisor 的 hand 距离判定参考点是 flange 中心
# （sim 里 backup 训练的几何中心），所以 deploy 时要把真机 TCP 退回 flange。
TCP_OFFSET = 0.1034


def compute_hand_body_equiv(tcp_pos: np.ndarray, quat_xyzw) -> np.ndarray:
    """Flange 中心 = TCP 沿 gripper +z 退 ``TCP_OFFSET``，对齐 sim 碰撞参考点。

    Args:
        tcp_pos: 末端 TCP 位置 (3,) m。
        quat_xyzw: 末端姿态 quaternion [x,y,z,w]。

    Returns:
        flange (hand body) 中心位置 (3,) m，喂给 HierarchicalSupervisor 做 hand-arm
        距离判定。
    """
    gripper_z_base = R.from_quat(quat_xyzw).apply([0, 0, 1])
    return tcp_pos - TCP_OFFSET * gripper_z_base
