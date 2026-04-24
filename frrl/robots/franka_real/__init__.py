"""Franka Panda 真机外围：HTTP server + 相机 + 视觉 + 轨迹执行器。

子模块:
  servers/              FR3/Panda HTTP server（启动: python -m frrl.robots.franka_real.servers.franka_server）
  cameras/              RealSense D455 采集管理
  vision/               hand_detector + tcp_tracker（ArUco + 皮肤色手检测）
  trajectory_executor   离线轨迹回放 / 录制工具
"""
