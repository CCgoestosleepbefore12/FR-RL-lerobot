"""SpaceMouseTeleop 干预检测状态机测试（方案 A + 迟滞）。

不触发真实 HID 硬件：用 mock 替换 SpaceMouseExpert.get_action。
重点覆盖：
  * 进入/退出阈值
  * 迟滞带（enter~exit 之间不翻转）
  * persist_steps 持续计数
  * 按钮直接进入/保持
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frrl.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseConfig
from frrl.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop


def _make_teleop(enter=0.05, exit_=0.02, persist=3) -> SpaceMouseTeleop:
    """构造一个 teleop，把底层 HID 替换成 MagicMock（避免连真实设备）。"""
    cfg = SpaceMouseConfig(
        enter_threshold=enter,
        exit_threshold=exit_,
        persist_steps=persist,
    )
    teleop = SpaceMouseTeleop(cfg)
    teleop._expert = MagicMock()
    teleop._connected = True
    return teleop


def _feed(teleop: SpaceMouseTeleop, sm6: np.ndarray, buttons=(0, 0, 0, 0)):
    """模拟一次 get_action 调用：让 mock expert 返回指定样本。"""
    teleop._expert.get_action.return_value = (np.asarray(sm6, dtype=np.float32), list(buttons))
    teleop.get_action()
    return teleop._is_intervention


class TestEnterThreshold:
    def test_below_enter_not_intervention(self):
        t = _make_teleop()
        assert _feed(t, [0.03, 0, 0, 0, 0, 0]) is False  # |sm| = 0.03 < 0.05

    def test_above_enter_is_intervention(self):
        t = _make_teleop()
        assert _feed(t, [0.08, 0, 0, 0, 0, 0]) is True   # |sm| = 0.08 > 0.05


class TestHysteresisBand:
    def test_signal_in_band_stays_false_when_started_false(self):
        """信号一直在 (exit, enter) 之间，未进入过 → 保持 False。"""
        t = _make_teleop()
        for _ in range(10):
            assert _feed(t, [0.03, 0, 0, 0, 0, 0]) is False  # 0.02 < 0.03 < 0.05

    def test_signal_in_band_stays_true_when_started_true(self):
        """进入接管后，信号落入迟滞带内（仍 > exit）→ 保持 True。"""
        t = _make_teleop()
        assert _feed(t, [0.1, 0, 0, 0, 0, 0]) is True     # 进入
        for _ in range(10):
            assert _feed(t, [0.03, 0, 0, 0, 0, 0]) is True  # 在带内 → 不退出


class TestExitWithPersistence:
    def test_exit_requires_persist_steps_below(self):
        """从 True 退出需要连续 persist_steps 步低于 exit。"""
        t = _make_teleop(persist=3)
        _feed(t, [0.1, 0, 0, 0, 0, 0])   # 进入
        assert _feed(t, [0.01, 0, 0, 0, 0, 0]) is True   # 第1步 < exit，还不退出
        assert _feed(t, [0.01, 0, 0, 0, 0, 0]) is True   # 第2步
        assert _feed(t, [0.01, 0, 0, 0, 0, 0]) is False  # 第3步 → 退出

    def test_counter_resets_on_rebound(self):
        """<exit 计数中途反弹到带内 → 计数清零，要重新数 persist_steps。"""
        t = _make_teleop(persist=3)
        _feed(t, [0.1, 0, 0, 0, 0, 0])
        _feed(t, [0.01, 0, 0, 0, 0, 0])   # count=1
        _feed(t, [0.01, 0, 0, 0, 0, 0])   # count=2
        _feed(t, [0.03, 0, 0, 0, 0, 0])   # 带内 → count 清零，仍 True
        assert t._is_intervention is True
        assert _feed(t, [0.01, 0, 0, 0, 0, 0]) is True   # 重新从 count=1 数
        assert _feed(t, [0.01, 0, 0, 0, 0, 0]) is True   # count=2
        assert _feed(t, [0.01, 0, 0, 0, 0, 0]) is False  # count=3 → 退出


class TestButtons:
    def test_button_forces_intervention_even_below_enter(self):
        t = _make_teleop()
        # 信号远小于 enter，但按钮按下 → 强制进入
        assert _feed(t, [0.001, 0, 0, 0, 0, 0], buttons=(1, 0, 0, 0)) is True

    def test_button_held_blocks_exit(self):
        """按钮按着 → 即便幅值一直 < exit 也不退出。"""
        t = _make_teleop(persist=3)
        _feed(t, [0.1, 0, 0, 0, 0, 0])
        for _ in range(10):
            # 幅值低但按钮按着 → counter 不递增
            assert _feed(t, [0.001, 0, 0, 0, 0, 0], buttons=(1, 0, 0, 0)) is True

    def test_button_release_allows_exit(self):
        """按钮松开后，幅值 < exit 连续 persist 步才退出。"""
        t = _make_teleop(persist=3)
        _feed(t, [0.001, 0, 0, 0, 0, 0], buttons=(1, 0, 0, 0))   # 按钮进入
        # 松开按钮，幅值一直低
        assert _feed(t, [0.001, 0, 0, 0, 0, 0]) is True   # count=1
        assert _feed(t, [0.001, 0, 0, 0, 0, 0]) is True   # count=2
        assert _feed(t, [0.001, 0, 0, 0, 0, 0]) is False  # count=3


class TestEdgeCases:
    def test_multi_axis_l2_norm(self):
        """L2 是 6D 范数，不是单轴：三个轴各 0.03 → L2 ≈ 0.052 > enter。"""
        t = _make_teleop()
        assert _feed(t, [0.03, 0.03, 0.03, 0, 0, 0]) is True

    def test_rotation_axes_also_count(self):
        """rpy 轴也进入 L2（前 6 维全部纳入）。"""
        t = _make_teleop()
        assert _feed(t, [0, 0, 0, 0.1, 0, 0]) is True
