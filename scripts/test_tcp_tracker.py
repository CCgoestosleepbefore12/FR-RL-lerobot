"""Test TCPTracker: real-time ArUco TCP tracking with smoothing + occlusion handling.

Shows:
  - Yellow: smoothed TCP from ArUco tracker (what the policy will see, minus noise)
  - White: FK TCP from /getstate
  - Green/red/orange status indicator: source (aruco / hold / fk / none)
  - ArUco-FK error: calibration sanity check (when no bias is injected)

Usage:
    python scripts/test_tcp_tracker.py
    python scripts/test_tcp_tracker.py --marker-size 0.07
    python scripts/test_tcp_tracker.py --no-noise   # disable 5mm noise for clean test
"""
import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import requests

from frrl.vision.tcp_tracker import TCPTracker

URL = "http://192.168.100.1:5000/"


def get_fk_tcp():
    try:
        r = requests.post(URL + "getstate", timeout=0.5)
        r.raise_for_status()
        return np.array(r.json()["pose"][:3])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-id", type=int, default=55)
    ap.add_argument("--marker-size", type=float, default=0.15)
    ap.add_argument("--no-noise", action="store_true", help="disable 5mm noise for clean test")
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    args = ap.parse_args()

    T = np.load("calibration_data/T_cam_to_robot.npy")
    print(f"Loaded T_cam_to_robot")

    # Start D455
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist_coeffs = np.array(intr.coeffs[:5])
    align = rs.align(rs.stream.color)

    time.sleep(3)
    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=10000)
    print(f"D455 ready: fx={intr.fx:.1f}")

    tracker = TCPTracker(
        T_cam_to_robot=T,
        cam_matrix=cam_matrix,
        dist_coeffs=dist_coeffs,
        marker_id=args.marker_id,
        marker_size=args.marker_size,
        ema_alpha=args.ema_alpha,
        add_noise=not args.no_noise,
    )

    print(f"TCPTracker started. marker_id={args.marker_id}, size={args.marker_size}m,"
          f" ema_alpha={args.ema_alpha}, noise={not args.no_noise}")
    print("Press 'q' to quit\n")

    src_colors = {
        "aruco": (0, 255, 0),     # green
        "hold": (0, 165, 255),    # orange
        "fk": (0, 0, 255),        # red
        "none": (128, 128, 128),  # gray
    }

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            color_img = np.asanyarray(aligned.get_color_frame().get_data())
            depth_img = np.asanyarray(aligned.get_depth_frame().get_data())

            fk_tcp = get_fk_tcp()
            real_tcp = tracker.update(color_img, depth_img, fk_tcp=fk_tcp)
            s = tracker.state

            vis = color_img.copy()

            # Status banner
            color = src_colors[s.source]
            cv2.putText(vis, f"source: {s.source}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if real_tcp is not None:
                cv2.putText(vis,
                            f"real_tcp (noisy): ({real_tcp[0]:+.3f},{real_tcp[1]:+.3f},{real_tcp[2]:+.3f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if fk_tcp is not None:
                cv2.putText(vis,
                            f"FK tcp:          ({fk_tcp[0]:+.3f},{fk_tcp[1]:+.3f},{fk_tcp[2]:+.3f})",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if s.source == "aruco" and s.smoothed_pos_robot is not None and fk_tcp is not None:
                err = np.linalg.norm(s.smoothed_pos_robot - fk_tcp)
                err_color = (0, 255, 0) if err < 0.03 else (0, 165, 255) if err < 0.06 else (0, 0, 255)
                cv2.putText(vis,
                            f"ArUco-FK error: {err*1000:.0f}mm",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, err_color, 2)

            if s.time_since_last_aruco != float("inf"):
                cv2.putText(vis, f"dt_aruco: {s.time_since_last_aruco*1000:.0f}ms", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("TCP Tracker", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

    print("\nDone")


if __name__ == "__main__":
    main()
