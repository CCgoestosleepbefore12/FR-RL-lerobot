"""GripperCommander — hil-serl 同款 gripper command 节流 + 状态去重."""
import time

import requests

from frrl.robots.franka_real.http_client import URL


class GripperCommander:
    """跟 ``frrl/envs/real.py:_send_gripper_command`` 同语义：基于 BC action[6]
    阈值 ±0.5 触发 ``/open_gripper`` or ``/close_gripper``，加 hysteresis 避免抖动。

    内部状态 ``commanded_state`` 记录上次发送的命令（"open" / "closed"），
    避免重复 HTTP 调用。``sleep_after`` rate limit 防 BC 短时间反复切换 gripper。
    """

    def __init__(self, sleep_after: float = 0.6):
        self.sleep_after = sleep_after
        self.last_act_time = 0.0
        self.commanded_state = "open"  # 假设 reset 后是张开

    def step(self, action_gripper: float, current_gripper_pos: float) -> None:
        """根据 BC 输出的 action[6] ∈ [-1, 1] 决定是否发开/关命令。"""
        now = time.time()
        if (now - self.last_act_time) < self.sleep_after:
            return  # rate limit
        # hil-serl thresholding：> 0.85 视作"已张开"，< 0.85 视作"未张开/已合"
        if action_gripper <= -0.5 and current_gripper_pos > 0.85 and self.commanded_state != "closed":
            try:
                requests.post(URL + "close_gripper", timeout=1.0)
                self.commanded_state = "closed"
                self.last_act_time = now
            except Exception as e:
                print(f"[gripper] close failed: {e}")
        elif action_gripper >= 0.5 and current_gripper_pos < 0.85 and self.commanded_state != "open":
            try:
                requests.post(URL + "open_gripper", timeout=1.0)
                self.commanded_state = "open"
                self.last_act_time = now
            except Exception as e:
                print(f"[gripper] open failed: {e}")

    def force_open(self) -> None:
        """同步释放 gripper 并把内部 commanded_state 标 "open"。

        用于 episode 边界 / recovery 之后 go_home 内部已调过 ``/open_gripper`` 的场景，
        保 caller 一致性（不发新 HTTP 也不重置 last_act_time 的 rate limit）。
        """
        try:
            requests.post(URL + "open_gripper", timeout=1.0)
            self.commanded_state = "open"
            self.last_act_time = time.time()
        except Exception:
            pass
