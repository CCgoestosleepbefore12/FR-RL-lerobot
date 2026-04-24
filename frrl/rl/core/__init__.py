"""RL 核心训练组件：分布式 actor/learner + replay buffer + env factory。

启动入口:
  python -m frrl.rl.core.learner      # learner 进程（gRPC server + optimizer）
  python -m frrl.rl.core.actor        # actor 进程（环境交互 + 经验上报）
  python -m frrl.rl.core.env_factory  # 录制 demo / 独立环境跑单机
"""
