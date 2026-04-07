"""
遥操作工具 — 精简版，仅支持键盘遥操。
"""

from enum import Enum

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


class TeleopEvents(Enum):
    """遥操作事件常量。"""

    SUCCESS = "success"
    FAILURE = "failure"
    RERECORD_EPISODE = "rerecord_episode"
    IS_INTERVENTION = "is_intervention"
    TERMINATE_EPISODE = "terminate_episode"


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    """根据配置创建遥操作设备。"""
    if config.type == "keyboard":
        from .keyboard import KeyboardTeleop
        return KeyboardTeleop(config)
    elif config.type == "keyboard_ee":
        from .keyboard.teleop_keyboard import KeyboardEndEffectorTeleop
        return KeyboardEndEffectorTeleop(config)
    elif config.type == "spacemouse":
        from .spacemouse import SpaceMouseTeleop
        return SpaceMouseTeleop(config)
    else:
        raise ValueError(
            f"不支持的遥操作类型 '{config.type}'。可用: keyboard, keyboard_ee, spacemouse"
        )
