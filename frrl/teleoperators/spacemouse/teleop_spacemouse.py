"""
SpaceMouse 遥操作器

输出 7D 动作 [dx, dy, dz, rx, ry, rz, gripper]，与 FR-RL 的动作空间对齐。
按钮映射：button[0] = close gripper (-1), button[1] = open gripper (+1)。
"""

import logging
from typing import Any

import numpy as np

from frrl.teleoperators.teleoperator import Teleoperator
from frrl.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseConfig


class SpaceMouseTeleop(Teleoperator):
    """SpaceMouse 遥操作，实现 Teleoperator 抽象接口。

    get_action() 返回 dict：
        - 各关节 key: 7D 动作值（与 action_features 对齐）
        - 额外 key: is_intervention, buttons
    """

    config_class = SpaceMouseConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseConfig):
        super().__init__(config)
        self.config = config
        self._expert = None
        self._connected = False

    # ================================================================
    # 抽象属性实现
    # ================================================================

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (7,),  # dx, dy, dz, rx, ry, rz, gripper
            "names": None,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # SpaceMouse 不需要校准

    # ================================================================
    # 抽象方法实现
    # ================================================================

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert

        self._expert = SpaceMouseExpert()
        self._connected = True
        logging.info("SpaceMouse 已连接")

    def calibrate(self) -> None:
        pass  # SpaceMouse 不需要校准

    def configure(self) -> None:
        pass  # 无需额外配置

    def get_action(self) -> dict[str, Any]:
        """获取 SpaceMouse 当前动作。

        Returns:
            dict 包含：
                - 7 个 action key（与 action_features 对齐）
                - "is_intervention": bool，SpaceMouse 是否有有效输入
                - "buttons": list[int]，原始按钮状态
        """
        if not self._connected or self._expert is None:
            raise RuntimeError("SpaceMouse 未连接，请先调用 connect()")

        raw_action, buttons = self._expert.get_action()
        # raw_action: 6D [dx, dy, dz, rx, ry, rz]

        # 按钮 → gripper 命令: button[0]=close(-1), button[1]=open(+1), 都没按=0
        if len(buttons) >= 2:
            if buttons[0]:
                gripper = -1.0
            elif buttons[1]:
                gripper = 1.0
            else:
                gripper = 0.0
        else:
            gripper = 0.0

        # 7D 动作: [dx, dy, dz, rx, ry, rz, gripper]
        action_7d = np.concatenate([raw_action[:6], [gripper]]).astype(np.float32)

        # 干预判断：SpaceMouse 有有效输入（超过死区）或按了按钮
        is_intervention = (
            np.linalg.norm(raw_action[:6]) > self.config.action_threshold
            or any(buttons)
        )

        # 构建返回 dict
        result = {
            f"action_{i}": float(action_7d[i]) for i in range(7)
        }
        result["is_intervention"] = is_intervention
        result["buttons"] = buttons

        return result

    def get_teleop_action(self) -> tuple[np.ndarray, bool]:
        """便捷方法：直接返回 7D 动作数组和干预标志。

        用于 Actor 的 action_processor pipeline。
        """
        result = self.get_action()
        action = np.array([result[f"action_{i}"] for i in range(7)], dtype=np.float32)
        return action, result["is_intervention"]

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # SpaceMouse 无反馈通道

    def disconnect(self) -> None:
        if self._expert is not None:
            self._expert.close()
            self._expert = None
        self._connected = False
        logging.info("SpaceMouse 已断开")
