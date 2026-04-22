"""Debug ArUco detection — save frame, try all dictionaries."""
import time
import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(config)
    time.sleep(3)
    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=10000)

    frames = pipeline.wait_for_frames(timeout_ms=10000)
    color = np.asanyarray(frames.get_color_frame().get_data())
    pipeline.stop()

    cv2.imwrite("debug_aruco_frame.png", color)
    print(f"Saved debug_aruco_frame.png ({color.shape})")
    print(f"  brightness: mean={color.mean():.1f}, min={color.min()}, max={color.max()}")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    dicts = [
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
        ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
        ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
        ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ]

    print("\nTrying all dictionaries:")
    for name, dict_id in dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
        n_rejected = len(rejected) if rejected is not None else 0
        if ids is not None:
            print(f"  {name:25s}  FOUND IDs: {ids.flatten().tolist()}  (rejected: {n_rejected})")
        else:
            print(f"  {name:25s}  nothing   (rejected candidates: {n_rejected})")

    print(f"\nCheck debug_aruco_frame.png — is the marker visible and in focus?")


if __name__ == "__main__":
    main()
