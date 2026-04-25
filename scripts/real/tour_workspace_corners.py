"""依次访问 workspace AABB 8 个顶点，实地验证 abs_pose_limit_low/high 安全性。

读取 calibration_data/workspace_bounds.json（或用 --bounds 指定）→ 由 xyz min/max
构造 axis-aligned 长方体 8 角 → 用 3 段路径（lift → transit → descend）安全访问。

每段都 polling /getstate 等到位（误差 < arrival_tol），不依赖固定 sleep。访问顺序：
  bottom 4 corners (z_low) → top 4 corners (z_high)
默认每个角点到位后等用户按 Enter 再继续，方便肉眼检查 / 拍照 / 测距。

⚠️ 注意：脚本直接 POST /pose，不走 FrankaRealEnv.clip_safety_box。理论上 AABB 角点
都已在 abs_pose_limit 内（因为这就是这个 AABB），但若 workspace_bounds.json 是别处
来的或机械臂位形特殊，仍可能撞外壳。开始前确认 E-stop 在手。

用法：
  python scripts/real/tour_workspace_corners.py
  python scripts/real/tour_workspace_corners.py --auto --dwell 2.0      # 自动 2 秒/角
  python scripts/real/tour_workspace_corners.py --bounds path/to.json
  python scripts/real/tour_workspace_corners.py --lift-margin 0.10      # transit 抬高 10cm

退出：q + Enter / Ctrl+C 都会触发 go-home（升 + 平移到 reset_pose）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests


def get_pose(url: str):
    r = requests.post(url + "getstate", timeout=2.0)
    r.raise_for_status()
    return np.array(r.json()["pose"], dtype=np.float64)  # 7D xyz + xyzw


def send_pose(url: str, pose7: np.ndarray):
    """POST /pose 一次。/pose 接收 [x,y,z,qx,qy,qz,qw] (scipy xyzw)。"""
    try:
        requests.post(url + "pose", json={"arr": pose7.tolist()}, timeout=1.0)
    except requests.exceptions.RequestException as e:
        print(f"  [!] /pose 失败: {e}")


def wait_for_arrival(url: str, target_xyz: np.ndarray, tol: float = 0.01,
                     timeout: float = 10.0, settle: float = 0.3) -> bool:
    """轮询 /getstate 直到 xyz 误差 < tol 或超时。返回是否到位。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            cur = get_pose(url)[:3]
        except requests.exceptions.RequestException:
            time.sleep(0.1)
            continue
        err = float(np.linalg.norm(cur - target_xyz))
        if err < tol:
            time.sleep(settle)  # 让 impedance 稳定
            return True
        time.sleep(0.1)
    return False


def make_corners(xyz_low: np.ndarray, xyz_high: np.ndarray) -> list:
    """生成 AABB 8 角，按 bottom→top + CCW 顺序。"""
    xl, yl, zl = xyz_low
    xh, yh, zh = xyz_high
    # 底面 4 角（z=zl）：(xl,yl) → (xh,yl) → (xh,yh) → (xl,yh)
    bottom = [
        ("B-low-y-low",   np.array([xl, yl, zl])),
        ("B-high-y-low",  np.array([xh, yl, zl])),
        ("B-high-y-high", np.array([xh, yh, zl])),
        ("B-low-y-high",  np.array([xl, yh, zl])),
    ]
    # 顶面 4 角（z=zh），同样 CCW
    top = [
        ("T-low-y-low",   np.array([xl, yl, zh])),
        ("T-high-y-low",  np.array([xh, yl, zh])),
        ("T-high-y-high", np.array([xh, yh, zh])),
        ("T-low-y-high",  np.array([xl, yh, zh])),
    ]
    return bottom + top


def go_to_corner(url: str, target_xyz: np.ndarray, quat_xyzw: np.ndarray,
                 lift_z: float, arrival_tol: float, segment_timeout: float):
    """3 段安全路径：当前位置 → 抬升到 lift_z → 平移到 target.xy → 下降到 target.z。"""
    cur = get_pose(url)
    # 段 1：抬到 lift_z（保 xy）
    lift = np.array([cur[0], cur[1], max(cur[2], lift_z)])
    if abs(lift[2] - cur[2]) > 1e-3:
        print(f"    [1/3] lift  → z={lift[2]:.3f}")
        send_pose(url, np.concatenate([lift, quat_xyzw]))
        wait_for_arrival(url, lift, tol=arrival_tol, timeout=segment_timeout)

    # 段 2：水平到 target.xy（z 保持 lift_z）
    transit = np.array([target_xyz[0], target_xyz[1], lift[2]])
    print(f"    [2/3] transit → xy=({transit[0]:.3f}, {transit[1]:.3f})")
    send_pose(url, np.concatenate([transit, quat_xyzw]))
    wait_for_arrival(url, transit, tol=arrival_tol, timeout=segment_timeout)

    # 段 3：垂直降到 target.z
    print(f"    [3/3] descend → z={target_xyz[2]:.3f}")
    send_pose(url, np.concatenate([target_xyz, quat_xyzw]))
    wait_for_arrival(url, target_xyz, tol=arrival_tol, timeout=segment_timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default="http://192.168.100.1:5000/")
    ap.add_argument("--bounds", type=Path,
                    default=Path("calibration_data/workspace_bounds.json"),
                    help="sample_workspace_bounds.py 输出 JSON")
    ap.add_argument("--auto", action="store_true",
                    help="不等 Enter，每个角点 dwell 秒后自动下一个")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="--auto 模式下每角点停留秒数（默认 2）")
    ap.add_argument("--lift-margin", type=float, default=0.05,
                    help="transit 时 z 抬高到 z_high + lift_margin（默认 5cm）")
    ap.add_argument("--arrival-tol", type=float, default=0.01,
                    help="到位判定容差 m（默认 1cm）")
    ap.add_argument("--segment-timeout", type=float, default=10.0,
                    help="每段最多等多久（默认 10s）")
    args = ap.parse_args()

    if not args.bounds.exists():
        print(f"[!] 未找到 {args.bounds} — 先跑 scripts/real/sample_workspace_bounds.py")
        sys.exit(1)

    data = json.loads(args.bounds.read_text())
    samples = data.get("samples", [])
    if not samples:
        print(f"[!] {args.bounds} 没有 samples"); sys.exit(1)

    xyz = np.array([s["xyz"] for s in samples])
    xyz_low = xyz.min(axis=0)
    xyz_high = xyz.max(axis=0)
    lift_z = float(xyz_high[2]) + args.lift_margin

    print(f"\n=== AABB 工作空间巡游 ===")
    print(f"  bounds: {args.bounds} ({len(samples)} samples)")
    print(f"  xyz_low  = ({xyz_low[0]:+.3f}, {xyz_low[1]:+.3f}, {xyz_low[2]:+.3f})")
    print(f"  xyz_high = ({xyz_high[0]:+.3f}, {xyz_high[1]:+.3f}, {xyz_high[2]:+.3f})")
    print(f"  transit lift z = {lift_z:.3f}  (= z_high + {args.lift_margin})")
    print(f"  arrival tol = {args.arrival_tol:.3f} m, per-segment timeout = {args.segment_timeout}s")
    if args.auto:
        print(f"  auto mode, dwell = {args.dwell}s/corner")
    else:
        print(f"  manual mode (Enter to advance, q to abort)")
    print()

    # 起始位姿（rotation 沿用，仅平移到角点）
    try:
        start_pose = get_pose(args.url)
    except requests.exceptions.RequestException as e:
        print(f"[!] /getstate 失败: {e}")
        sys.exit(1)
    start_xyz = start_pose[:3].copy()
    start_quat = start_pose[3:].copy()  # xyzw
    print(f"  start xyz=({start_xyz[0]:+.3f}, {start_xyz[1]:+.3f}, {start_xyz[2]:+.3f})")
    print(f"  start quat (xyzw)={np.round(start_quat, 3)}\n")

    if not args.auto:
        if input("Ready to start? (Enter / q to cancel): ").strip().lower() == "q":
            print("Cancelled."); return

    corners = make_corners(xyz_low, xyz_high)
    aborted = False
    try:
        for i, (name, target) in enumerate(corners, start=1):
            print(f"[corner {i}/8] {name:18s} → ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})")
            go_to_corner(args.url, target, start_quat, lift_z,
                         args.arrival_tol, args.segment_timeout)

            # 到位后实测一下
            actual = get_pose(args.url)[:3]
            err = float(np.linalg.norm(actual - target))
            status = "OK" if err < args.arrival_tol * 2 else "WARN"
            print(f"    arrived: actual=({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})  "
                  f"err={err*1000:.1f}mm  [{status}]")

            # 停留
            if args.auto:
                time.sleep(args.dwell)
            else:
                line = input(f"    Enter to next corner (q to abort): ").strip().lower()
                if line == "q":
                    aborted = True
                    break
    except KeyboardInterrupt:
        print("\n[Ctrl+C] aborting tour")
        aborted = True

    # 收尾：回到起始位置（同样 3 段安全路径）
    print(f"\n=== 收尾：回起始位置 ===")
    try:
        go_to_corner(args.url, start_xyz, start_quat, lift_z,
                     args.arrival_tol, args.segment_timeout)
        actual = get_pose(args.url)[:3]
        err = float(np.linalg.norm(actual - start_xyz))
        print(f"home: actual=({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})  "
              f"err={err*1000:.1f}mm")
    except Exception as e:
        print(f"[!] go-home 失败: {e}")

    if aborted:
        print("\n中途退出。")
    else:
        print("\n8 角全部访问完毕。")


if __name__ == "__main__":
    main()
