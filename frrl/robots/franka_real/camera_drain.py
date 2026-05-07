"""Camera-aware blocking primitives for deploy main loops.

Deploy 主循环里所有 sleep / homing 期间，pyrealsense pipeline 必须持续被 read，
否则 buffer 堵塞 → wait_for_frames 超时 / 拿到 stale 帧 / pipeline 进入坏状态。
hand_detector + cam_mgr 一旦 init 就必须以至少 ~10Hz 被 drain。

本模块抽出 4 个核心 utility，让 deploy 脚本（pickup/wipe/pickandplace
_with_backup）共用同一套 camera-safe 阻塞原语：

  drain_sleep                       — `time.sleep(s)` 的安全替代，期间 ~20Hz drain 双相机
  wait_for_operator_with_camera_drain — 阻塞等操作员按 Enter，期间 drain 双相机
  check_hand_close_during_homing    — homing 期间一帧 hand 检测，发现手返回 True
  interpolate_move_with_drain       — 平滑插值 send_pose 同时 ~20Hz drain + hand check
"""
import time

import numpy as np

from frrl.robots.franka_real.geometry import compute_hand_body_equiv
from frrl.robots.franka_real.http_client import get_state_true, send_pose


def wait_for_operator_with_camera_drain(hand_detector, cam_mgr, prompt: str) -> None:
    """阻塞等操作员按 Enter（stdin），期间 ~20Hz drain 两个相机的 pyrealsense
    pipeline。``--wait-operator`` flag 路径用，让操作员在 episode 结束后能停下
    来摆好物块再开下一集，而 BACKUP 安全网在阻塞期间也活着。
    """
    import select
    import sys
    print(prompt, flush=True)
    while True:
        try:
            hand_detector.get_frames()
        except Exception:
            pass
        try:
            cam_mgr.get_images()
        except Exception:
            pass
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return
        time.sleep(0.05)


def drain_sleep(seconds: float, hand_detector, cam_mgr) -> None:
    """阻塞 ``seconds`` 秒，期间 ~20Hz drain 两个相机的 pyrealsense pipeline。

    所有 deploy 主循环里的 ``time.sleep`` 在 hand_detector 和 cam_mgr 已经 init
    之后都应该改用这个，让 BACKUP 安全网在阻塞期间也活着。
    """
    end = time.time() + seconds
    while time.time() < end:
        try:
            hand_detector.get_frames()
        except Exception:
            pass
        try:
            cam_mgr.get_images()
        except Exception:
            pass
        time.sleep(0.05)  # ~20 Hz drain rate


def check_hand_close_during_homing(hand_detector, d_safe: float) -> bool:
    """读相机帧 + 检测 hand，如果 hand 距离机械臂 < d_safe 返回 True。

    复刻主循环的 hand 检测逻辑，给 ``interpolate_move_with_drain`` 用。这一步
    同时也 drain 掉 hand_detector pipeline 一帧（detect 内部调 wait_for_frames）。
    """
    try:
        color, depth = hand_detector.get_frames()
        state = get_state_true()
        fk_tcp = np.array(state["pose"][:3], dtype=np.float64)
        hand_body = compute_hand_body_equiv(fk_tcp, state["pose"][3:])
        hand = hand_detector.detect(
            color, depth,
            exclude_near_flange=hand_body, flange_radius=0.10,
            exclude_near_tcp=fk_tcp, tcp_radius=0.06,
        )
        if hand.active:
            dist = float(np.linalg.norm(hand.pos_robot - hand_body))
            return dist < d_safe
    except Exception:
        pass
    return False


def interpolate_move_with_drain(
    start_xyz, start_quat_xyzw, goal_xyz, goal_quat_xyzw,
    timeout: float, hand_detector, cam_mgr,
    d_safe: float, hz: float = 10.0,
) -> bool:
    """从 start 线性插值到 goal，每 0.1s 发一帧 pose；每 tick 内 ~20Hz 检查
    hand 距离，hand < d_safe 立即返回 False（caller 中断 homing）。

    Returns:
        True = 完成全部插值；False = hand 检测中断。
    """
    steps = max(int(timeout * hz), 2)
    dt = 1.0 / hz
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    goal_xyz = np.asarray(goal_xyz, dtype=np.float64)
    start_quat = np.asarray(start_quat_xyzw, dtype=np.float64)
    goal_quat = np.asarray(goal_quat_xyzw, dtype=np.float64)
    if float(np.dot(start_quat, goal_quat)) < 0:
        goal_quat = -goal_quat

    for i in range(1, steps + 1):
        t = i / steps
        xyz = start_xyz + t * (goal_xyz - start_xyz)
        quat = (1.0 - t) * start_quat + t * goal_quat
        quat = quat / (np.linalg.norm(quat) + 1e-12)
        send_pose(xyz, list(quat))

        # tick 内 ~20 Hz 检查 hand 距离 + drain wrist cam pipeline
        end_t = time.time() + dt
        while time.time() < end_t:
            if check_hand_close_during_homing(hand_detector, d_safe):
                print(f"[safety] hand intrusion @ homing step {i}/{steps} → abort")
                return False
            try:
                cam_mgr.get_images()
            except Exception:
                pass
            time.sleep(0.05)
    return True
