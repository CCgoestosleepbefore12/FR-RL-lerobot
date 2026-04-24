"""D455 RealSense smoke test — verify RGB + Depth streaming.

Auto-detects all connected D455 cameras. Falls back to single camera if only one is found.

Usage:
    python scripts/hw_check/test_d455.py              # print stats only (headless OK)
    python scripts/hw_check/test_d455.py --show       # display live RGB+Depth per camera (needs GUI)
    python scripts/hw_check/test_d455.py --save       # save one RGB + Depth frame per camera
"""
import argparse
import time
import numpy as np
import pyrealsense2 as rs


def list_realsense_serials():
    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]


def open_pipeline(serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    return pipeline, profile


def print_info(serial: str, profile, width: int, height: int, fps: int):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    print(f"=== D455 [{serial}] ===")
    print(f"  resolution: {width}x{height} @ {fps}fps")
    print(f"  depth scale: {depth_scale:.6f} (1 unit = {depth_scale*1000:.2f} mm)")
    print(f"  color intrinsics:")
    print(f"    fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}")
    print(f"    cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")
    print(f"    distortion: {intrinsics.model}")
    return depth_scale, intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="live display (needs GUI)")
    ap.add_argument("--save", action="store_true", help="save one frame pair per camera")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    serials = list_realsense_serials()
    if len(serials) == 0:
        print("ERROR: no RealSense device detected")
        return
    if len(serials) == 1:
        print(f"WARN: only 1 camera detected — running in single-camera mode ({serials[0]})")
    else:
        print(f"Detected {len(serials)} RealSense devices: {serials}\n")

    pipelines = []
    aligns = []
    scales = []
    intrinsics_list = []
    for s in serials:
        try:
            pipe, prof = open_pipeline(s, args.width, args.height, args.fps)
        except Exception as e:
            print(f"ERROR opening {s}: {e}")
            continue
        ds, intr = print_info(s, prof, args.width, args.height, args.fps)
        pipelines.append((s, pipe))
        aligns.append(rs.align(rs.stream.color))
        scales.append(ds)
        intrinsics_list.append(intr)
        print()

    if not pipelines:
        print("ERROR: no pipeline could be opened")
        return

    print("  warming up (3s)...")
    time.sleep(3)
    for (_, pipe) in pipelines:
        for _ in range(10):
            pipe.wait_for_frames(timeout_ms=10000)

    # Single-frame stats per camera
    print("\n=== Frame Stats ===")
    single_frames = []
    for (s, pipe), align, ds, intr in zip(pipelines, aligns, scales, intrinsics_list):
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        color_img = np.asanyarray(aligned.get_color_frame().get_data())
        depth_img = np.asanyarray(aligned.get_depth_frame().get_data())
        depth_m = depth_img.astype(np.float32) * ds
        valid = depth_m[depth_m > 0]
        cx, cy = int(intr.ppx), int(intr.ppy)
        print(f"  [{s}]")
        print(f"    color shape: {color_img.shape}  dtype: {color_img.dtype}")
        print(f"    depth shape: {depth_img.shape}  dtype: {depth_img.dtype}")
        if valid.size > 0:
            print(f"    depth range: {valid.min():.3f} - {valid.max():.3f} m")
        print(f"    center pixel depth: {depth_m[cy, cx]:.3f} m")
        single_frames.append((s, color_img, depth_img))

    if args.save:
        import cv2
        for s, color_img, depth_img in single_frames:
            suffix = s[-6:]
            cv2.imwrite(f"d455_{suffix}_color.png", color_img)
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imwrite(f"d455_{suffix}_depth.png", depth_vis)
            print(f"  saved: d455_{suffix}_color.png, d455_{suffix}_depth.png")

    if args.show:
        import cv2
        print("\n  showing live feed — press 'q' in any window to quit")
        try:
            while True:
                for (s, pipe), align in zip(pipelines, aligns):
                    frames = pipe.wait_for_frames()
                    aligned = align.process(frames)
                    c = np.asanyarray(aligned.get_color_frame().get_data())
                    d = np.asanyarray(aligned.get_depth_frame().get_data())
                    d_vis = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)
                    combined = np.hstack([c, d_vis])
                    cv2.imshow(f"D455 [{s}] RGB | Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()

    for _, pipe in pipelines:
        pipe.stop()
    print("\nOK")


if __name__ == "__main__":
    main()
