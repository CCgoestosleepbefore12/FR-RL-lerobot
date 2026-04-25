"""定义 Franka 工作空间边界 + front 相机 workspace ROI crop。

操作流程（参考 calibrate_cam_to_robot.py 的交互模式）：
  1. SpaceMouse 牵引 Franka 在工作空间内游走
  2. 到工作空间边界的极端点（4 底角 / 顶角 / 任意最远点）按 Enter 记录 TCP
  3. 采够 ≥4 个样本后按 Q 退出
  4. 脚本取 axis-aligned xyz bbox → abs_pose_limit[:3]
  5. 用已有 T_cam_to_robot 把 workspace xy 底面 4 角投影到 front 像素空间 → ROI bbox
  6. ROI 默认扩成正方形（最小外接），SAC encoder 要求同尺寸输入
  7. 保存 JSON + 打印 real_config.py 的修改片段

前置条件：
  - calibration_data/T_cam_to_robot.npy 已生成（见 calibrate_cam_to_robot.py）
  - RT PC Flask server 运行
  - SpaceMouse 连好
  - front D455 已接（脚本独占使用；跑完记得关）

用法：
  python scripts/real/define_workspace.py
  python scripts/real/define_workspace.py --roi-shape rect      # 保留矩形 ROI
  python scripts/real/define_workspace.py --z-plane 0.0         # 强制投影到桌面 z=0
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
from scipy.spatial.transform import Rotation as R

from frrl.envs.real_config import FRONT_CAMERA_SERIAL

URL = "http://192.168.100.1:5000/"
SAVE_DIR = Path("calibration_data")

# SpaceMouse 灵敏度（与 calibrate_cam_to_robot.py 一致，缓慢便于精确停位）
ACTION_SCALE = 0.015
ROTATION_SCALE = 0.035
DEADZONE = 0.05


# ----------------------------------------------------------------------
# 硬件 & 控制辅助（复制自 calibrate_cam_to_robot.py，保持两脚本独立）
# ----------------------------------------------------------------------


def start_d455(serial: str, width=640, height=480, fps=15):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ])

    print(f"D455 [{serial}]: {width}x{height}@{fps}fps")
    print(f"  fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
          f"cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}")

    time.sleep(3)
    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=10000)

    return pipeline, cam_matrix


def start_spacemouse():
    from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
    expert = SpaceMouseExpert()
    print("SpaceMouse connected")
    return expert


def get_tcp_pose():
    r = requests.post(URL + "getstate", timeout=2.0)
    r.raise_for_status()
    state = r.json()
    return np.array(state["pose"][:3]), state["pose"]


def send_pose(pose7):
    try:
        requests.post(URL + "pose", json={"arr": list(pose7)}, timeout=0.5)
    except requests.exceptions.Timeout:
        pass


def apply_spacemouse(target_xyz, target_quat, action6):
    dz = lambda v: 0.0 if abs(v) < DEADZONE else v
    dxyz = [dz(v) * ACTION_SCALE for v in action6[:3]]
    drpy = [dz(v) * ROTATION_SCALE for v in action6[3:]]

    target_xyz[0] += dxyz[0]
    target_xyz[1] += dxyz[1]
    target_xyz[2] += dxyz[2]

    if abs(drpy[0]) + abs(drpy[1]) + abs(drpy[2]) > 1e-6:
        # Match FrankaRealEnv:249-252: left-mult + rotvec = world-frame, axis-angle.
        # Right-mult + Euler is body-frame and diverges from env behavior.
        new = R.from_rotvec(np.array(drpy)) * R.from_quat(target_quat)
        target_quat[:] = list(new.as_quat())

    return target_xyz, target_quat


# ----------------------------------------------------------------------
# Workspace ROI 计算
# ----------------------------------------------------------------------


def project_base_to_pixel(points_base, T_cam_to_robot, cam_matrix):
    """base 坐标 (N,3) → front 像素 (N,2) via pinhole model，无畸变近似。"""
    pts_h = np.hstack([points_base, np.ones((len(points_base), 1))])   # (N, 4)
    T_base_to_cam = np.linalg.inv(T_cam_to_robot)
    pts_cam = (T_base_to_cam @ pts_h.T).T[:, :3]                        # (N, 3)
    if np.any(pts_cam[:, 2] <= 0):
        raise ValueError("workspace 角点投影到相机后方 (z≤0)；检查 T_cam_to_robot")
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy
    return np.stack([u, v], axis=1)


def compute_roi_bbox(xyz_low, xyz_high, T_cam_to_robot, cam_matrix, z_plane, img_wh):
    """xy 底面 4 角 → front 像素 bbox (u0, v0, u1, v1)，裁到图像内。"""
    x_lo, y_lo = xyz_low[0], xyz_low[1]
    x_hi, y_hi = xyz_high[0], xyz_high[1]
    corners = np.array([
        [x_lo, y_lo, z_plane],
        [x_lo, y_hi, z_plane],
        [x_hi, y_lo, z_plane],
        [x_hi, y_hi, z_plane],
    ])
    uv = project_base_to_pixel(corners, T_cam_to_robot, cam_matrix)
    W, H = img_wh
    u0 = int(np.clip(np.floor(uv[:, 0].min()), 0, W))
    u1 = int(np.clip(np.ceil(uv[:, 0].max()), 0, W))
    v0 = int(np.clip(np.floor(uv[:, 1].min()), 0, H))
    v1 = int(np.clip(np.ceil(uv[:, 1].max()), 0, H))
    return u0, v0, u1, v1


def bbox_to_square(u0, v0, u1, v1, img_wh, mode):
    """把矩形 bbox 调成正方形（SAC encoder 要求 H=W）。

    mode:
      rect       — 不调整（如果后续 resize 接受非等比，会被拉伸）
      inscribed  — 以 bbox 中心取最大内接正方形（丢失 workspace 边缘）
      bounding   — 以 bbox 中心取最小外接正方形（保全 workspace，可能碰到图像边）

    默认 bounding：宁可 crop 到图像边（边界外为黑边无所谓），也不丢 workspace。
    """
    if mode == "rect":
        return u0, v0, u1, v1
    W, H = img_wh
    w, h = u1 - u0, v1 - v0
    cu, cv_ = (u0 + u1) // 2, (v0 + v1) // 2
    s = min(w, h) if mode == "inscribed" else max(w, h)
    half = s // 2
    nu0 = max(0, cu - half)
    nu1 = min(W, cu + half)
    nv0 = max(0, cv_ - half)
    nv1 = min(H, cv_ + half)
    return nu0, nv0, nu1, nv1


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, default=SAVE_DIR / "T_cam_to_robot.npy",
                    help="T_cam_to_robot.npy 路径（calibrate_cam_to_robot.py 产物）")
    ap.add_argument("--output", type=Path, default=SAVE_DIR / "workspace.json",
                    help="采样+ROI 结果输出 JSON")
    ap.add_argument("--z-plane", type=float, default=None,
                    help="ROI 投影平面高度 (m)。默认 = 采样 z 最低值（桌面）")
    ap.add_argument("--roi-shape", choices=["rect", "inscribed", "bounding"],
                    default="bounding", help="ROI 形状调整策略（默认 bounding）")
    ap.add_argument("--serial", type=str, default=FRONT_CAMERA_SERIAL,
                    help=f"D455 serial (default: front {FRONT_CAMERA_SERIAL})")
    args = ap.parse_args()

    if not args.calib.exists():
        print(f"[!] 未找到 {args.calib}；请先跑 calibrate_cam_to_robot.py")
        return
    T_cam_to_robot = np.load(args.calib)
    print(f"Loaded T_cam_to_robot from {args.calib}")

    pipeline, cam_matrix = start_d455(args.serial)
    expert = start_spacemouse()

    state = requests.post(URL + "getstate", timeout=2.0).json()
    target_xyz = list(state["pose"][:3])
    # franka_server.py 的 /getstate 返回 pose[3:] = [qx,qy,qz,qw]（scipy xyzw,
    # 见 r.as_quat() at franka_server.py:222）。/pose endpoint 同样吃 xyzw。
    target_quat = list(state["pose"][3:])  # xyzw, 不需要 swap

    sampled = []   # list of np.array([x, y, z])

    print("\n=== Workspace Definition (SpaceMouse) ===")
    print("  SpaceMouse: 移动 TCP 到工作空间边界的极端点")
    print("  ENTER   : 记录当前 TCP 位置")
    print("  Q       : 结束采样并计算 bbox（至少 4 个点）")
    print("  推荐采样: 桌面 4 角 + 1-2 个最高点；也可沿边界自由游走多采几点\n")

    try:
        while True:
            t0 = time.time()

            # SpaceMouse → robot
            action6, _ = expert.get_action()
            target_xyz, target_quat = apply_spacemouse(target_xyz, target_quat, action6)
            send_pose([*target_xyz, *target_quat])  # /pose 吃 xyzw

            # Camera frame
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_img = np.asanyarray(frames.get_color_frame().get_data())
            vis = color_img.copy()
            H, W = vis.shape[:2]

            # 实时 overlay：当前 bbox + ROI 投影
            if len(sampled) >= 2:
                pts = np.array(sampled)
                lo = pts.min(axis=0)
                hi = pts.max(axis=0)
                z_plane = args.z_plane if args.z_plane is not None else float(lo[2])
                try:
                    u0, v0, u1, v1 = compute_roi_bbox(
                        lo, hi, T_cam_to_robot, cam_matrix, z_plane, (W, H)
                    )
                    su0, sv0, su1, sv1 = bbox_to_square(
                        u0, v0, u1, v1, (W, H), mode=args.roi_shape
                    )
                    # 黄色：原 bbox；绿色：最终 ROI（按 roi-shape 调整后）
                    cv2.rectangle(vis, (u0, v0), (u1, v1), (0, 255, 255), 1)
                    cv2.rectangle(vis, (su0, sv0), (su1, sv1), (0, 255, 0), 2)
                    cv2.putText(vis, f"ROI [{args.roi_shape}]",
                                (su0 + 4, max(sv0 - 8, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    cv2.putText(vis, f"ROI error: {e}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 采样点投影到当前画面（红点）
            if sampled:
                try:
                    uv = project_base_to_pixel(
                        np.array(sampled), T_cam_to_robot, cam_matrix,
                    )
                    for (u_px, v_px) in uv:
                        if 0 <= u_px < W and 0 <= v_px < H:
                            cv2.circle(vis, (int(u_px), int(v_px)), 5, (0, 0, 255), -1)
                except Exception:
                    pass

            cv2.putText(vis, f"Samples: {len(sampled)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis, f"TCP: ({target_xyz[0]:.3f}, "
                             f"{target_xyz[1]:.3f}, {target_xyz[2]:.3f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(vis, "ENTER=capture  Q=finish",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Define Workspace", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                try:
                    pos, _ = get_tcp_pose()
                except Exception as e:
                    print(f"  [!] getstate failed: {e}")
                    continue
                sampled.append(pos)
                print(f"  [{len(sampled):2d}] TCP = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            elif key == ord('q'):
                break

            # ~10 Hz loop
            dt_left = 0.1 - (time.time() - t0)
            if dt_left > 0:
                time.sleep(dt_left)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        expert.close()

    if len(sampled) < 4:
        print(f"\n只采了 {len(sampled)} 个点 — 至少需要 4 个。Aborting.")
        return

    pts = np.array(sampled)
    xyz_low = pts.min(axis=0)
    xyz_high = pts.max(axis=0)
    z_plane = args.z_plane if args.z_plane is not None else float(xyz_low[2])

    u0, v0, u1, v1 = compute_roi_bbox(
        xyz_low, xyz_high, T_cam_to_robot, cam_matrix, z_plane, (640, 480),
    )
    su0, sv0, su1, sv1 = bbox_to_square(
        u0, v0, u1, v1, (640, 480), mode=args.roi_shape,
    )

    # ------- 保存 JSON -------
    SAVE_DIR.mkdir(exist_ok=True)
    out = {
        "xyz_low": xyz_low.tolist(),
        "xyz_high": xyz_high.tolist(),
        "z_plane": z_plane,
        "roi_front_bbox": [u0, v0, u1, v1],
        "roi_front_final": [su0, sv0, su1, sv1],
        "roi_shape": args.roi_shape,
        "samples": [p.tolist() for p in sampled],
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\n已保存: {args.output}")

    # ------- 打印诊断 + config 修改片段 -------
    print("\n" + "=" * 60)
    print("采样结果")
    print("=" * 60)
    print(f"  x ∈ [{xyz_low[0]:+.3f}, {xyz_high[0]:+.3f}]  "
          f"(Δ = {xyz_high[0] - xyz_low[0]:.3f} m)")
    print(f"  y ∈ [{xyz_low[1]:+.3f}, {xyz_high[1]:+.3f}]  "
          f"(Δ = {xyz_high[1] - xyz_low[1]:.3f} m)")
    print(f"  z ∈ [{xyz_low[2]:+.3f}, {xyz_high[2]:+.3f}]  "
          f"(Δ = {xyz_high[2] - xyz_low[2]:.3f} m)")
    print(f"  ROI 投影平面 z = {z_plane:.3f} m")
    print(f"  ROI bbox   : [{u0}, {v0}] → [{u1}, {v1}]  "
          f"(w={u1-u0}, h={v1-v0})")
    print(f"  ROI {args.roi_shape:>9s}: [{su0}, {sv0}] → [{su1}, {sv1}]  "
          f"(w={su1-su0}, h={sv1-sv0})")

    print("\n" + "=" * 60)
    print("应用到 FrankaRealConfig（paste 到 real_config.py 或初始化时覆盖）")
    print("=" * 60)
    print(f"""
import numpy as np
from frrl.envs.real_config import FrankaRealConfig, make_workspace_roi_crop

cfg = FrankaRealConfig()
cfg.abs_pose_limit_low[:3]  = np.array([{xyz_low[0]:.3f}, {xyz_low[1]:.3f}, {xyz_low[2]:.3f}])
cfg.abs_pose_limit_high[:3] = np.array([{xyz_high[0]:.3f}, {xyz_high[1]:.3f}, {xyz_high[2]:.3f}])
cfg.image_crop["front"] = make_workspace_roi_crop({su0}, {sv0}, {su1}, {sv1})
""")


if __name__ == "__main__":
    main()
