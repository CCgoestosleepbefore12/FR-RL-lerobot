"""
SpaceMouse 底层接口

多进程架构：后台进程持续轮询 USB HID 设备，主进程通过共享内存读取最新状态。
移植自 hil-serl/serl_robot_infra/franka_env/spacemouse/spacemouse_expert.py。
"""

import multiprocessing
from typing import Tuple

import numpy as np

from frrl.teleoperators.spacemouse import pyspacemouse


class SpaceMouseExpert:
    """SpaceMouse USB HID 接口，多进程持续读取。

    get_action() 返回:
        action: 6D np.ndarray [dx, dy, dz, rx, ry, rz]
        buttons: list[int] 按钮状态
    """

    def __init__(self):
        pyspacemouse.open()

        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0, 0, 0]

        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()
        self._closed = False

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                # 双 SpaceMouse（12D，不常用）
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw,
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                # 单 SpaceMouse（6D）
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                ]
                buttons = state[0].buttons

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """返回最新的 6D 动作和按钮状态。"""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self):
        """Release the reader process, manager, and HID handle.

        Idempotent — calling twice is safe. Call from teleop scripts'
        `finally:` so repeated sessions don't leak Manager subprocesses
        or keep the SpaceMouse HID device locked.
        """
        if self._closed:
            return
        self._closed = True

        # 1. Tear down the background reader.
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)

        # 2. Shut down the multiprocessing.Manager server (it spawns its own
        #    process that otherwise sticks around).
        try:
            self.manager.shutdown()
        except Exception:
            pass

        # 3. Release the HID device held by the bundled pyspacemouse module's
        #    module-level _active_device so the next SpaceMouseExpert() can open it.
        try:
            pyspacemouse.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
