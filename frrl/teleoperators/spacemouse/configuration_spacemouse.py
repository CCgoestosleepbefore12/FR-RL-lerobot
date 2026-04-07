"""SpaceMouse 遥操作配置"""

from dataclasses import dataclass

from frrl.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig):
    """SpaceMouse 遥操作配置。

    SpaceMouse 通过 USB HID 自动发现，不需要额外的设备配置字段。
    """

    # 动作灵敏度（用于干预时的死区判断）
    action_threshold: float = 0.001
