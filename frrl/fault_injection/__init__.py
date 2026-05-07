#!/usr/bin/env python3
"""编码器校准偏差注入模块（package）

历史：本模块原本是单文件 ``frrl/fault_injection.py``。2026-05-07 拆成 package：

  config.py            — EncoderBiasConfig (dataclass + YAML loader)
  injector.py          — EncoderBiasInjector (per-episode bias 采样)
  monitor.py           — BiasMonitor (matplotlib 实时曲线 + npz/png 存盘)
  deploy_controller.py — BiasDeployController (lifecycle 4 钩子封装上面 3 个)

对外 API 完全不变。所有调用点（`from frrl.fault_injection import X`）零迁移成本。

模拟工业机器人编码器校准误差（encoder calibration bias）：
- 一个故障源（编码器bias）→ 同时影响控制和观测
- 控制器基于错误的关节角度计算Jacobian → 力矩不准 → 执行偏差
- 观测基于错误的FK计算 → EE位姿报告不准 → 感知偏差
- 物理仿真本身不受影响（关节真实位置正确）

这精确模拟了真实机器人编码器校准偏差的因果链：
    编码器偏差 → q_measured = q_true + bias
        ├→ 控制器用 q_measured 计算（控制受影响）
        └→ FK(q_measured) 得到错误的 EE 位姿（观测受影响）
"""
from frrl.fault_injection.config import EncoderBiasConfig
from frrl.fault_injection.injector import EncoderBiasInjector
from frrl.fault_injection.monitor import BiasMonitor
from frrl.fault_injection.deploy_controller import BiasDeployController

__all__ = [
    "EncoderBiasConfig",
    "EncoderBiasInjector",
    "BiasMonitor",
    "BiasDeployController",
]
