"""Preview workspace ROI crop — see what the policy will actually receive.

读取 calibration_data/workspace.json 里 define_workspace.py 算出来的
roi_front_final，应用到 front 相机实时帧，并 resize 到 SAC encoder 输入尺寸
（默认 128×128，对齐 FrankaRealConfig.image_size["front"]）。

显示三窗口对比：
  1. Original 640×480 + 绿色 ROI 矩形（裁剪范围标注）
  2. Cropped ROI（裁出来的实际像素，原分辨率）
  3. Policy view 128×128（最终送进 SAC encoder 的尺寸）

判断标准：
  - 第一个窗口看绿框是否覆盖你的工作区域（吸管/海绵/目标盘等）
  - 第三个窗口看 policy 真正能识别的内容（小目标在 128² 下还看得清吗？）

操作：
  Q  退出
  S  保存三个窗口当前帧到 calibration_data/workspace_preview_{TIMESTAMP}_*.png

用法：
  python scripts/hw_check/preview_workspace_crop.py
  python scripts/hw_check/preview_workspace_crop.py --workspace path/to/workspace.json
  python scripts/hw_check/preview_workspace_crop.py --policy-size 224
  python scripts/hw_check/preview_workspace_crop.py --serial 318122303303    # 切到 wrist
"""
import argparse
import json
import time
from datetime import datetime
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


def load_roi(workspace_path: Path):
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"{workspace_path} 不存在 — 先跑 scripts/real/define_workspace.py"
        )
    data = json.loads(workspace_path.read_text())
    roi = data.get("roi_front_final")
    if roi is None or len(roi) != 4:
        raise ValueError(f"{workspace_path} 缺少 roi_front_final 字段")
    u0, v0, u1, v1 = [int(v) for v in roi]
    return u0, v0, u1, v1, data


def draw_overlay(frame, u0, v0, u1, v1, roi_shape="?"):
    """在原始帧上画 ROI 矩形 + 标注尺寸 + 工作区中心。"""
    vis = frame.copy()
    cv2.rectangle(vis, (u0, v0), (u1, v1), (0, 255, 0), 2)

    w, h = u1 - u0, v1 - v0
    cu, cv_ = (u0 + u1) // 2, (v0 + v1) // 2
    cv2.drawMarker(vis, (cu, cv_), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)

    text = f"ROI [{roi_shape}] {w}x{h}px  ({u0},{v0})->({u1},{v1})"
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    if w != h:
        cv2.putText(vis, f"NOT SQUARE (w!=h, will distort on resize)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path,
                    default=Path("calibration_data/workspace.json"),
                    help="define_workspace.py 输出 JSON 路径")
    ap.add_argument("--policy-size", type=int, default=128,
                    help="SAC encoder 输入尺寸（FrankaRealConfig.image_size['front'] 默认 128）")
    ap.add_argument("--serial", type=str, default=FRONT_CAMERA_SERIAL,
                    help=f"D455 serial (default: front {FRONT_CAMERA_SERIAL})")
    ap.add_argument("--override-roi", type=str, default=None,
                    help="覆盖 workspace.json 的 ROI，格式 u0,v0,u1,v1（逗号分隔，"
                         "无空格）。用于不改 json 直接试 crop。建议保持 u1-u0 == v1-v0 "
                         "（正方形），SAC encoder 要求 H==W；非方时 resize 会拉伸。")
    args = ap.parse_args()

    u0, v0, u1, v1, ws_data = load_roi(args.workspace)
    if args.override_roi is not None:
        try:
            parts = [int(x) for x in args.override_roi.split(",")]
            assert len(parts) == 4, f"need 4 ints u0,v0,u1,v1, got {parts}"
        except (ValueError, AssertionError) as e:
            raise SystemExit(f"--override-roi 解析失败: {e}")
        u0, v0, u1, v1 = parts
        ws_data = {**ws_data, "roi_shape": "override"}
        w, h = u1 - u0, v1 - v0
        print(f"OVERRIDE ROI: ({u0},{v0}) -> ({u1},{v1}) = {w}x{h}px"
              f"{'  [WARN: not square]' if w != h else ''}")
    print(f"Loaded ROI from {args.workspace}: ({u0},{v0}) -> ({u1},{v1}) "
          f"= {u1-u0}x{v1-v0}px [{ws_data.get('roi_shape', '?')}]")
    print(f"Policy input size: {args.policy_size}x{args.policy_size}")
    print(f"  (xyz workspace bbox: low={ws_data['xyz_low']}, high={ws_data['xyz_high']})")
    print("\nQ: quit  |  S: save 3-window snapshot to calibration_data/\n")

    pipeline = start_d455(args.serial)
    snapshot_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = np.asanyarray(frames.get_color_frame().get_data())

            overlay = draw_overlay(color, u0, v0, u1, v1, ws_data.get("roi_shape", "?"))
            cropped = color[v0:v1, u0:u1]
            if cropped.size == 0:
                cropped = np.zeros((10, 10, 3), dtype=np.uint8)
            policy_view = cv2.resize(cropped, (args.policy_size, args.policy_size),
                                     interpolation=cv2.INTER_AREA)

            cv2.imshow("1. Original 640x480 + ROI", overlay)
            cv2.imshow("2. Cropped ROI (raw)", cropped)
            cv2.imshow(f"3. Policy view {args.policy_size}^2", policy_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                outdir = args.workspace.parent
                cv2.imwrite(str(outdir / f"workspace_preview_{ts}_1_overlay.png"), overlay)
                cv2.imwrite(str(outdir / f"workspace_preview_{ts}_2_cropped.png"), cropped)
                cv2.imwrite(str(outdir / f"workspace_preview_{ts}_3_policy.png"), policy_view)
                snapshot_count += 1
                print(f"  [snapshot {snapshot_count}] saved to {outdir}/workspace_preview_{ts}_*.png")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] exiting")
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
