"""真机机器人子包（按机器人型号分子目录）。

目前只有 `franka_real/`（Franka Panda + RealSense D455 + 手检测/TCP 视觉）。

注意：
- `frrl/envs/configs.py` 等处用 `try: from frrl.robots import RobotConfig` 做 hil-serl
  风格的 robot factory 抽象占位，当前仍未实装。import 失败会回退到 `Any`，不阻塞 sim。
"""
