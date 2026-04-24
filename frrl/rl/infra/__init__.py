"""RL 基础设施：进程/队列管理 + wandb 日志 + gRPC transport。

子模块:
  process.py          进程信号处理（SIGTERM/SIGINT 优雅退出）
  queue.py            跨进程队列工具
  wandb_utils.py      WandBLogger 封装
  actor_utils.py      actor 侧共用（episode discard 钩子等）
  transport/          actor↔learner gRPC proto + bytes 分块 helpers
"""
