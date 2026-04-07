"""
环境 Wrappers — 在核心环境外层叠加功能

① InputsControlWrapper — 键盘/手柄人类介入（HIL-SERL核心机制）
② GripperPenaltyWrapper — 频繁开关夹爪扣分
③ ResetDelayWrapper     — episode重置后等几秒
④ PassiveViewerWrapper  — 弹出MuJoCo交互窗口

注意：EEActionWrapper 已废弃（7D 动作空间统一后不再需要 3D→7D 扩展）
"""

from frrl.envs.wrappers.hil_wrappers import (
    GripperPenaltyWrapper,
    InputsControlWrapper,
    ResetDelayWrapper,
)
from frrl.envs.wrappers.viewer_wrapper import PassiveViewerWrapper
from frrl.envs.wrappers.factory import wrap_env

__all__ = [
    "GripperPenaltyWrapper",
    "InputsControlWrapper",
    "ResetDelayWrapper",
    "PassiveViewerWrapper",
    "wrap_env",
]
