"""Test HandDetector: skin color + depth hand detection + ArUco TCP tracking.

Shows real-time D455 feed with:
  - Green circle: detected hand/skin blob centroid
  - Yellow text: ArUco marker position (TCP reference from camera)
  - White text: TCP from /getstate (robot FK)
  - Debug window: skin+depth mask (shows what the detector "sees")

Usage:
    python scripts/hw_check/test_hand_detector.py
    python scripts/hw_check/test_hand_detector.py --no-aruco     # skip ArUco, hand only
    python scripts/hw_check/test_hand_detector.py --no-robot      # skip /getstate, camera only
    python scripts/hw_check/test_hand_detector.py --debug          # show skin mask window
"""
import argparse
import time

import cv2
import numpy as np
import requests

from frrl.vision.hand_detector import HandDetector

URL = "http://192.168.100.1:5000/"


def get_tcp():
    try:
        r = requests.post(URL + "getstate", timeout=0.5)
        r.raise_for_status()
        return np.array(r.json()["pose"][:3])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-aruco", action="store_true", help="skip ArUco detection")
    ap.add_argument("--no-robot", action="store_true", help="skip /getstate")
    ap.add_argument("--marker-id", type=int, default=55)
    ap.add_argument("--marker-size", type=float, default=0.15)
    ap.add_argument("--debug", action="store_true", help="show skin+depth mask window")
    args = ap.parse_args()

    T = np.load("calibration_data/T_cam_to_robot.npy")
    print(f"Loaded T_cam_to_robot from calibration_data/")

    detector = HandDetector(T_cam_to_robot=T)
    detector.start()
    print("D455 + MediaPipe ready\n")

    print("=== Live Test ===")
    print("  Show your hand to the camera — green circle tracks wrist")
    if not args.no_aruco:
        print("  ArUco marker on gripper — yellow text shows camera-derived TCP")
    if not args.no_robot:
        print("  White text shows /getstate TCP (FK)")
    print("  Press 'q' to quit\n")

    frame_times = []

    try:
        while True:
            t0 = time.time()

            color_img, depth_img = detector.get_frames()

            # Hand detection
            hand = detector.detect(color_img, depth_img)

            # ArUco TCP tracking
            aruco_tcp = None
            if not args.no_aruco:
                aruco_tcp = detector.detect_aruco(
                    color_img, depth_img,
                    marker_id=args.marker_id,
                    marker_size=args.marker_size,
                )

            # FK TCP
            fk_tcp = None
            if not args.no_robot:
                fk_tcp = get_tcp()

            # Draw
            vis = detector.draw_detection(color_img, hand, aruco_tcp)

            if fk_tcp is not None:
                cv2.putText(vis, f"FK TCP: ({fk_tcp[0]:.3f},{fk_tcp[1]:.3f},{fk_tcp[2]:.3f})",
                            (10, vis.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Calibration error (ArUco vs FK)
            if aruco_tcp is not None and fk_tcp is not None:
                err = np.linalg.norm(aruco_tcp - fk_tcp)
                color = (0, 255, 0) if err < 0.03 else (0, 165, 255) if err < 0.06 else (0, 0, 255)
                cv2.putText(vis, f"ArUco-FK error: {err*1000:.0f}mm", (10, vis.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Hand-TCP distance (for backup policy trigger)
            if hand.active and fk_tcp is not None:
                dist = np.linalg.norm(hand.pos_robot - fk_tcp)
                dist_color = (0, 0, 255) if dist < 0.15 else (0, 255, 255) if dist < 0.30 else (0, 255, 0)
                cv2.putText(vis, f"Hand-TCP dist: {dist*1000:.0f}mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)

            # FPS
            dt = time.time() - t0
            frame_times.append(dt)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / np.mean(frame_times)
            cv2.putText(vis, f"{fps:.0f} FPS", (vis.shape[1] - 80, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Hand Detector Test", vis)

            if args.debug:
                mask = detector.get_debug_mask(color_img, depth_img)
                cv2.imshow("Skin+Depth Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        detector.stop()

    print("\nDone")


if __name__ == "__main__":
    main()
