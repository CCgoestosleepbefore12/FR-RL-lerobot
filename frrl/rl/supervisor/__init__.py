"""Supervisor 子包：部署态 FSM + 6D 归位控制器。

对外统一从此包 import：
    from frrl.rl.supervisor import HierarchicalSupervisor, Mode, HomingController
"""
from frrl.rl.supervisor.hierarchical import HierarchicalSupervisor, Mode
from frrl.rl.supervisor.homing import (
    HomingController,
    quat_conjugate,
    quat_multiply,
    quat_to_axis_angle,
)

__all__ = [
    "HierarchicalSupervisor",
    "Mode",
    "HomingController",
    "quat_conjugate",
    "quat_multiply",
    "quat_to_axis_angle",
]
