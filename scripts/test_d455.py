"""D455 RealSense smoke test — verify RGB + Depth streaming.

Usage:
    python scripts/test_d455.py              # print stats only (headless OK)
    python scripts/test_d455.py --show       # display live RGB+Depth window (needs display)
    python scripts/test_d455.py --save       # save one RGB + Depth frame as PNG
"""
import argparse
import time
import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="live display (needs GUI)")
    ap.add_argument("--save", action="store_true", help="save one frame pair")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    print("=== D455 Info ===")
    print(f"  resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"  depth scale: {depth_scale:.6f} (1 unit = {depth_scale*1000:.2f} mm)")
    print(f"  color intrinsics:")
    print(f"    fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}")
    print(f"    cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")
    print(f"    distortion: {intrinsics.model}")

    # warm up — D455 needs a few seconds after start
    print("\n  warming up (3s)...")
    time.sleep(3)
    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=10000)

    align = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    depth_m = depth_img.astype(np.float32) * depth_scale

    print(f"\n=== Frame Stats ===")
    print(f"  color shape: {color_img.shape}  dtype: {color_img.dtype}")
    print(f"  depth shape: {depth_img.shape}  dtype: {depth_img.dtype}")
    print(f"  depth range: {depth_m[depth_m > 0].min():.3f} - {depth_m[depth_m > 0].max():.3f} m")
    cx, cy = int(intrinsics.ppx), int(intrinsics.ppy)
    print(f"  center pixel depth: {depth_m[cy, cx]:.3f} m")

    if args.save:
        import cv2
        cv2.imwrite("d455_color.png", color_img)
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
        )
        cv2.imwrite("d455_depth.png", depth_vis)
        print("\n  saved: d455_color.png, d455_depth.png")

    if args.show:
        import cv2
        print("\n  showing live feed — press 'q' to quit")
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                c = np.asanyarray(aligned.get_color_frame().get_data())
                d = np.asanyarray(aligned.get_depth_frame().get_data())
                d_vis = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)
                combined = np.hstack([c, d_vis])
                cv2.imshow("D455 RGB | Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()

    pipeline.stop()
    print("\nOK")


if __name__ == "__main__":
    main()
