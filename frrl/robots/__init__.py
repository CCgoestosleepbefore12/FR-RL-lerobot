"""真机机器人子包（按机器人型号分子目录）。

目前只有 `franka_real/`（Franka Panda + RealSense D455 + 手检测/TCP 视觉）。

**本包不 export 任何符号**：`frrl/envs/configs.py`、`frrl/utils/control_utils.py`、
`frrl/processor/joint_observations_processor.py`、`frrl/rl/core/env_factory.py` 里形如
`try: from frrl.robots import RobotConfig / Robot / make_robot_from_config` 的 import
均为 hil-serl 风格的 robot factory 占位抽象，**预期会抛 ImportError 并走 except 分支
回退到 `Any`**，不阻塞 sim。未来若实装真机 factory，再在此 re-export 对应类。
"""
