#!/usr/bin/env python3
"""BiasDeployController — deploy 脚本里 bias 注入 + monitor 的 lifecycle 封装.

替代 deploy 脚本里散落的 ``set_encoder_bias / clear_encoder_bias / anchor pose
/ sample / monitor.update`` 模板代码。Caller 不需要懂 FSM——只需在 episode
生命周期 4 个钩子上调对应方法即可。
"""
import time
from typing import Optional

import numpy as np

from frrl.fault_injection.config import EncoderBiasConfig
from frrl.fault_injection.injector import EncoderBiasInjector
from frrl.fault_injection.monitor import BiasMonitor


class BiasDeployController:
    """Bias 注入 + monitor 在手写 deploy 脚本里的统一封装。

    替代 deploy 脚本里散落的 ``set_encoder_bias / clear_encoder_bias / anchor pose
    / sample / monitor.update`` 模板代码。Caller 不需要懂 FSM——只需在 episode
    生命周期 4 个钩子上调本对象的对应方法即可：

      启动:    begin_transition() → caller 自己 go_home → finish_transition(0,
               resample=True)
      Recovery: begin_transition() → caller 自己 go_home → finish_transition(ep,
               resample=False)   # 同一 episode，沿用旧 bias
      新集 (success/timeout): begin_transition() → go_home → finish_transition(ep,
               resample=True)    # 采新 bias
      每 iter: on_step(state_true, state_biased)
      shutdown: close()

    可视化 trick：``begin_transition`` 推一个 NaN 样本进 BiasMonitor 的 deque，
    matplotlib `set_data` 遇 NaN 自动断线 → episode 边界曲线不再被错误连接产生
    交叉，recovery 期间的 home（bias=0）也跟前后段断开不会拖一道斜线。

    精神上等价于 ``frrl/envs/real.py:125-138, 210-250`` 在 FrankaRealEnv 里做的
    bias 生命周期管理，但适配手写 deploy 脚本（不走 env.reset / env.step）。
    """

    def __init__(
        self,
        encoder_bias_config: EncoderBiasConfig,
        *,
        url: str,
        monitor_save_path: Optional[str] = None,
        enable_monitor: bool = False,
        sleep_after_inject: float = 0.3,
        sleep_after_clear: float = 0.5,
        request_timeout: float = 2.0,
    ):
        """
        Args:
            encoder_bias_config: 来自 task_cfg.encoder_bias_config（None 时 caller
                应该不构造本对象，直接 None 即可）。
            url: Flask server base URL，例如 ``http://192.168.100.1:5000/``。
            monitor_save_path: BiasMonitor npz/png 输出前缀，None 时不存盘。
            enable_monitor: 是否构造 BiasMonitor（即使 bias 注入开着也可以不画图）。
            sleep_after_inject: set_encoder_bias 后的 settle 时间，对齐 real.py:250。
            sleep_after_clear: clear_encoder_bias + anchor 后的 settle 时间，对齐
                real.py:217。
        """
        # 延迟导入 requests，避免本模块在仅离线分析（不连机）的场景拉 requests 依赖
        import requests
        self._requests = requests

        self._injector = EncoderBiasInjector(encoder_bias_config)
        self._url = url.rstrip("/") + "/"
        self._sleep_after_inject = float(sleep_after_inject)
        self._sleep_after_clear = float(sleep_after_clear)
        self._request_timeout = float(request_timeout)

        self._monitor: Optional[BiasMonitor] = None
        self._monitor_save_path = monitor_save_path
        if enable_monitor:
            self._monitor = BiasMonitor(
                update_hz=2.0,
                save_path=monitor_save_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """True iff this controller is doing anything（caller 用来 short-circuit
        生命周期调用，避免外面 4 处 ``if bias_ctrl is not None``）。"""
        return True

    @property
    def current_bias(self) -> Optional[np.ndarray]:
        """当前 episode 的 bias，None 表示本集没注入。"""
        return self._injector.current_bias

    def begin_transition(self) -> None:
        """Episode 边界开始（go_home 之前调）：

        1. 推 NaN 样本进 monitor → matplotlib 折线断点（recovery 的 home 期间
           bias=0，前后两段不会被一条斜线连接产生视觉混乱；episode 间也不会被
           连接产生交叉）。
        2. HTTP clear_encoder_bias = 0 → controller 切到 unbiased frame，让接下来
           的 setpoint（reset_pose）能物理到达。
        3. Anchor: 把当前 unbiased pose 立刻发给 /pose，避免 controller 用 stale
           biased setpoint 朝错误位置走（real.py:212-217 的同款保护）。
        """
        if self._monitor is not None:
            nan7 = np.full(7, np.nan, dtype=np.float32)
            self._monitor.update(nan7, nan7, nan7)
        try:
            self._post("set_encoder_bias", json={"bias": [0.0] * 7})
            state_true = self._post("getstate_true").json()
            self._post("pose", json={"arr": state_true["pose"]}, timeout=0.5)
            time.sleep(self._sleep_after_clear)
        except Exception as e:
            print(f"[BiasDeployController] begin_transition failed: {e}")

    def finish_transition(self, ep_num: int, *, resample: bool) -> None:
        """Episode 边界结束（go_home 之后调）：

        Args:
            ep_num: 用于 BiasMonitor 边界标签和日志。Caller 自己算（一般是
                ``success_count + failure_count``）。
            resample: True = 调 ``on_episode_start`` 采新 bias（新 episode）；
                False = 沿用 ``current_bias`` 重新注入（recovery，同一 episode 的
                bias 应该恒定）。
        """
        if resample:
            self._injector.on_episode_start(num_joints=7)
        bias = self._injector.current_bias
        if bias is None:
            # 概率 < 1 时本 episode 不注入故障，仍要确保 controller 在 unbiased
            # frame（begin_transition 已经清过了）。
            return
        try:
            self._post("set_encoder_bias", json={"bias": bias.tolist()})
            tag = "resampled" if resample else "reinjected"
            print(f"[bias] ep#{ep_num} {tag}: {np.round(bias, 4).tolist()} rad")
            if self._monitor is not None:
                self._monitor.mark_episode_boundary(ep_num=ep_num, bias=bias)
            time.sleep(self._sleep_after_inject)
        except Exception as e:
            print(f"[BiasDeployController] finish_transition failed: {e}")

    def on_step(self, state_true: dict, state_biased: dict) -> None:
        """主循环每 iter 调一次。喂 BiasMonitor 一个样本。

        Args:
            state_true: ``/getstate_true`` 返回，含 ``q``。
            state_biased: ``/getstate`` 返回，含 ``q`` 和（可选）``bias``。
        """
        if self._monitor is None:
            return
        active_bias = state_biased.get("bias")
        if active_bias is None:
            # Fallback：q_biased - q_true（若 server 没返回 bias 字段时）
            active_bias = (
                np.array(state_biased["q"]) - np.array(state_true["q"])
            ).tolist()
        self._monitor.update(state_true["q"], state_biased["q"], active_bias)

    def close(self) -> None:
        """Shutdown：保 npz/png + 关图 + HTTP clear_encoder_bias 防 bias 残留下次启动。"""
        if self._monitor is not None:
            try:
                self._monitor.close()
                if self._monitor_save_path is not None:
                    print(f"[BiasDeployController] monitor saved → "
                          f"{self._monitor_save_path}.{{npz,png}}")
            except Exception as e:
                print(f"[BiasDeployController] monitor.close failed: {e}")
        try:
            self._post("clear_encoder_bias")
        except Exception as e:
            print(f"[BiasDeployController] final clear_encoder_bias failed: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _post(self, path: str, *, json=None, timeout: Optional[float] = None):
        r = self._requests.post(
            self._url + path,
            json=json,
            timeout=timeout if timeout is not None else self._request_timeout,
        )
        r.raise_for_status()
        return r
