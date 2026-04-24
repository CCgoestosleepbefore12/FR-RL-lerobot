"""SpaceMouse 遥操作配置"""

from dataclasses import dataclass

from frrl.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig):
    """SpaceMouse 遥操作配置。

    SpaceMouse 通过 USB HID 自动发现，不需要额外的设备配置字段。
    """

    # 干预检测（hil-serl 方案 A + 迟滞）
    # SpaceMouse 原始输出是 [-1, 1] 的 6D，L2 范数大致：静止 0~0.02 / 搭手 0.02~0.05 / 推动 >0.08
    # enter: 信号越过即进入接管；exit: 降到该值以下持续 persist_steps 步才退出接管。
    # enter > exit 形成迟滞带，防止边界抖动引起 is_intervention 反复翻转。
    enter_threshold: float = 0.05
    exit_threshold: float = 0.02
    persist_steps: int = 3
