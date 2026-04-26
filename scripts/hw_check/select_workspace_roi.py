"""鼠标拖拽框选 front camera workspace ROI，并写回 workspace.json。

交互（在 "Select ROI" 窗口内操作）：
  鼠标左键拖   — 拖出一个矩形选区
  S            — 把选区调成正方形（取短边内接，center 不变）
  W            — 写回 calibration_data/workspace.json 的 roi_front_final
  R            — 清空选区，重选
  Q / ESC      — 退出（不写）

旁边窗口：
  "Cropped"      — 当前选区裁出的原始像素
  "Policy view"  — 选区 resize 到 SAC encoder 输入尺寸（默认 128²）

⚠️ 写回时如果选区不是正方形，会先打印一个警告再写——SAC encoder 要求 H==W，
非正方形会被 resize 时拉伸。建议先按 S 调正方形再 W 写回。

用法：
  python scripts/hw_check/select_workspace_roi.py
  python scripts/hw_check/select_workspace_roi.py --policy-size 128
  python scripts/hw_check/select_workspace_roi.py --serial 318122303303    # wrist
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from frrl.envs.real_config import FRONT_CAMERA_SERIAL


def start_d455(serial: str, width=640, height=480, fps=15):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    print(f"D455 [{serial}]: {width}x{height}@{fps}fps")
    time.sleep(2)
    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=10000)
    return pipeline


class ROISelector:
    """鼠标拖拽 + 状态机。"""

    def __init__(self, width: int, height: int):
        self.W, self.H = width, height
        self.dragging = False
        self.drag_start = None       # (u, v) when mouse pressed
        self.drag_current = None     # (u, v) latest while dragging
        self.roi = None              # (u0, v0, u1, v1) finalized (after release)

    def on_mouse(self, event, x, y, flags, param):
        x = int(np.clip(x, 0, self.W - 1))
        y = int(np.clip(y, 0, self.H - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            self.roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_current = (x, y)
            self.roi = self._normalize(self.drag_start, (x, y))

    @staticmethod
    def _normalize(p1, p2):
        u0, v0 = min(p1[0], p2[0]), min(p1[1], p2[1])
        u1, v1 = max(p1[0], p2[0]), max(p1[1], p2[1])
        if u1 - u0 < 4 or v1 - v0 < 4:
            return None
        return (u0, v0, u1, v1)

    def current_box(self):
        """获取当前应该画在屏幕上的 box：拖拽中 → 实时；松开后 → 最终；否则 None。"""
        if self.dragging and self.drag_start is not None and self.drag_current is not None:
            return self._normalize(self.drag_start, self.drag_current)
        return self.roi

    def to_square(self):
        """把已确定的 ROI 调成正方形（取短边，内接，center 不变）。"""
        if self.roi is None:
            return
        u0, v0, u1, v1 = self.roi
        cu, cv_ = (u0 + u1) // 2, (v0 + v1) // 2
        edge = min(u1 - u0, v1 - v0)
        half = edge // 2
        nu0 = max(0, cu - half)
        nu1 = min(self.W, cu + half)
        nv0 = max(0, cv_ - half)
        nv1 = min(self.H, cv_ + half)
        self.roi = (nu0, nv0, nu1, nv1)

    def reset(self):
        self.dragging = False
        self.drag_start = None
        self.drag_current = None
        self.roi = None


def draw_overlay(frame, box, label_color=(0, 255, 0), label_prefix="ROI"):
    vis = frame.copy()
    if box is None:
        cv2.putText(vis, "drag mouse to select ROI", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return vis
    u0, v0, u1, v1 = box
    w, h = u1 - u0, v1 - v0
    cv2.rectangle(vis, (u0, v0), (u1, v1), label_color, 2)
    text = f"{label_prefix} {w}x{h}  ({u0},{v0})->({u1},{v1})"
    cv2.putText(vis, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1)
    if w != h:
        cv2.putText(vis, "NOT SQUARE - press S to make square",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(vis, "S=square  W=write json  R=reset  Q=quit",
                (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return vis


def write_workspace_json(workspace_path: Path, roi):
    """把 roi 写入 workspace.json 的 roi_front_final 字段（保留其他字段）。"""
    if not workspace_path.exists():
        # 如果文件不存在，建一个最小可用的
        data = {"roi_front_final": list(roi), "roi_shape": "manual"}
    else:
        data = json.loads(workspace_path.read_text())
        data["roi_front_final"] = list(roi)
        data["roi_shape"] = "manual"
    tmp = workspace_path.with_suffix(workspace_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(workspace_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path,
                    default=Path("calibration_data/workspace.json"),
                    help="目标 JSON 路径，写回 roi_front_final")
    ap.add_argument("--policy-size", type=int, default=128,
                    help="SAC encoder 输入尺寸预览（默认 128）")
    ap.add_argument("--serial", type=str, default=FRONT_CAMERA_SERIAL,
                    help=f"D455 serial (default: front {FRONT_CAMERA_SERIAL})")
    args = ap.parse_args()

    pipeline = start_d455(args.serial)

    # 提前读现有 ROI 当默认显示
    initial_roi = None
    if args.workspace.exists():
        try:
            data = json.loads(args.workspace.read_text())
            roi = data.get("roi_front_final")
            if roi and len(roi) == 4:
                initial_roi = tuple(int(v) for v in roi)
                print(f"Loaded current ROI from {args.workspace}: {initial_roi}")
        except Exception as e:
            print(f"  [warn] cannot read {args.workspace}: {e}")

    selector = ROISelector(width=640, height=480)
    if initial_roi is not None:
        selector.roi = initial_roi

    win_main = "Select ROI"
    cv2.namedWindow(win_main, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_main, selector.on_mouse)

    print("\n=== ROI Selector ===")
    print("  drag mouse on 'Select ROI' to select")
    print("  S=square  W=write json  R=reset  Q/ESC=quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = np.asanyarray(frames.get_color_frame().get_data())

            box = selector.current_box()
            overlay = draw_overlay(color, box)
            cv2.imshow(win_main, overlay)

            if box is not None:
                u0, v0, u1, v1 = box
                cropped = color[v0:v1, u0:u1]
                if cropped.size > 0:
                    cv2.imshow("Cropped", cropped)
                    policy_view = cv2.resize(
                        cropped, (args.policy_size, args.policy_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow(f"Policy view {args.policy_size}^2", policy_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                print("quit without write")
                break
            elif key == ord('r'):
                selector.reset()
                print("[reset]")
            elif key == ord('s'):
                if selector.roi is None:
                    print("[!] no ROI to square")
                else:
                    before = selector.roi
                    selector.to_square()
                    after = selector.roi
                    print(f"[square] {before} -> {after}")
            elif key == ord('w'):
                if selector.roi is None:
                    print("[!] no ROI to write")
                    continue
                u0, v0, u1, v1 = selector.roi
                w, h = u1 - u0, v1 - v0
                if w != h:
                    print(f"[!] WARN: writing non-square ROI {w}x{h} — SAC encoder will distort on resize")
                write_workspace_json(args.workspace, selector.roi)
                print(f"[write] {args.workspace} <- roi_front_final = {selector.roi}")
                print(f"        贴到 real_config.py:  cfg.image_crop['front'] = "
                      f"make_workspace_roi_crop({u0}, {v0}, {u1}, {v1})")
                break
    except KeyboardInterrupt:
        print("\n[Ctrl+C]")
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
