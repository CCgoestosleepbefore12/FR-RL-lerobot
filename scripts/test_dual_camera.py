"""Dual-camera RealSense bandwidth & stability test.

Opens all detected D455s in parallel threads, streams for N seconds, reports:
  - actual FPS vs expected
  - frame drops (detected via frame-number gaps)
  - inter-frame delay mean / max
  - timeout / error counts
  - lsusb -t (USB controller topology)

Usage:
    python scripts/test_dual_camera.py                   # 30s, 640x480 @ 15fps, color only
    python scripts/test_dual_camera.py --duration 60     # 60s run
    python scripts/test_dual_camera.py --depth           # also stream depth
    python scripts/test_dual_camera.py --fps 30          # test higher fps
"""
import argparse
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pyrealsense2 as rs


@dataclass
class CamStats:
    serial: str
    frame_numbers: List[int] = field(default_factory=list)
    wall_timestamps: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def list_realsense_serials():
    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]


def open_pipeline(serial, width, height, fps, depth):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    return pipeline, profile


def capture_loop(pipeline, stats: CamStats, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if color is None:
                stats.errors.append("no color frame")
                continue
            stats.frame_numbers.append(color.get_frame_number())
            stats.wall_timestamps.append(time.time())
        except RuntimeError as e:
            stats.errors.append(str(e))


def summarize(stats: CamStats, duration: float, expected_fps: int):
    fns = stats.frame_numbers
    ts = np.asarray(stats.wall_timestamps, dtype=np.float64)
    n = len(fns)
    expected = expected_fps * duration

    if n < 2:
        print(f"  [{stats.serial}] TOO FEW FRAMES: {n} (errors: {len(stats.errors)})")
        if stats.errors:
            print("    error samples:")
            for e in list(dict.fromkeys(stats.errors))[:5]:
                count = stats.errors.count(e)
                print(f"      · [{count}x] {e}")
        return

    span = fns[-1] - fns[0] + 1  # frame numbers are monotonic per camera
    drops = span - n  # gaps in frame numbers

    intervals_ms = np.diff(ts) * 1000.0
    actual_fps = n / duration

    print(f"  [{stats.serial}]")
    print(f"    frames received : {n}  (expected ≈ {int(expected)}, recovery rate {100*n/expected:.1f}%)")
    print(f"    actual FPS      : {actual_fps:.2f}  (target {expected_fps})")
    print(f"    frame drops     : {drops}  (hw frame-number gaps)")
    print(f"    inter-frame ms  : mean {intervals_ms.mean():.1f}  median {np.median(intervals_ms):.1f}  max {intervals_ms.max():.1f}  p95 {np.percentile(intervals_ms, 95):.1f}")
    print(f"    errors          : {len(stats.errors)}")
    if stats.errors:
        unique_errs = list(set(stats.errors))[:3]
        for e in unique_errs:
            print(f"      · {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to stream")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--depth", action="store_true", help="also stream depth (doubles bandwidth)")
    ap.add_argument("--open-delay", type=float, default=1.5,
                    help="seconds to wait between opening successive pipelines (helps avoid SDK race)")
    ap.add_argument("--reverse", action="store_true",
                    help="open cameras in reverse serial order (diagnostic)")
    args = ap.parse_args()

    serials = list_realsense_serials()
    if len(serials) == 0:
        print("ERROR: no RealSense detected")
        return
    if len(serials) == 1:
        print(f"WARN: only 1 camera detected — running single-camera test ({serials[0]})")
    else:
        print(f"Detected {len(serials)} RealSense devices: {serials}")

    mode = "RGB + Depth" if args.depth else "RGB only"
    print(f"Config: {args.width}x{args.height} @ {args.fps}fps  |  {mode}  |  duration {args.duration}s\n")

    if args.reverse:
        serials = list(reversed(serials))
        print(f"  (reverse order: {serials})")

    pipelines = []
    for i, s in enumerate(serials):
        if i > 0 and args.open_delay > 0:
            print(f"  waiting {args.open_delay:.1f}s before next open...")
            time.sleep(args.open_delay)
        try:
            pipe, _ = open_pipeline(s, args.width, args.height, args.fps, args.depth)
            pipelines.append((s, pipe))
            print(f"  opened {s}")
        except Exception as e:
            print(f"  FAILED {s}: {e}")

    if not pipelines:
        print("ERROR: no pipeline opened")
        return

    # warm up — D455 auto-exposure needs time to settle
    print("\n  warming up (3s)...")
    time.sleep(3)
    for _, pipe in pipelines:
        for _ in range(5):
            try:
                pipe.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                pass

    # start capture threads
    stats_list = [CamStats(s) for s, _ in pipelines]
    stop_event = threading.Event()
    threads = []
    print(f"\n  streaming for {args.duration}s ...")
    start_time = time.time()
    for (s, pipe), stats in zip(pipelines, stats_list):
        t = threading.Thread(target=capture_loop, args=(pipe, stats, stop_event), daemon=True)
        t.start()
        threads.append(t)

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("  interrupted early")

    stop_event.set()
    for t in threads:
        t.join(timeout=5.0)
    actual_duration = time.time() - start_time

    # report
    print(f"\n=== Stats (actual duration {actual_duration:.1f}s) ===")
    for stats in stats_list:
        summarize(stats, actual_duration, args.fps)

    for _, pipe in pipelines:
        try:
            pipe.stop()
        except Exception:
            pass

    # USB topology
    print("\n=== lsusb -t ===")
    try:
        out = subprocess.run(["lsusb", "-t"], capture_output=True, text=True, timeout=5)
        print(out.stdout)
    except FileNotFoundError:
        print("  (lsusb not found)")
    except subprocess.TimeoutExpired:
        print("  (lsusb timed out)")

    print("\n=== Interpretation guide ===")
    print("  · FPS near target & 0 drops               → bandwidth OK")
    print("  · drops > 1% or actual FPS 20%+ below target → USB contention / controller overload")
    print("  · max inter-frame ms >> 2000/fps          → occasional stall, may be OK if drops=0")
    print("  · two cameras on same 'Class=Hub' in lsusb -t → consider moving to separate USB controllers")
    print("\nOK")


if __name__ == "__main__":
    main()
