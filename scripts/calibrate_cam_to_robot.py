"""Eye-to-hand camera calibration using ArUco marker on TCP + SpaceMouse.

Mount ArUco marker rigidly on the Franka gripper/TCP.
Use SpaceMouse to move robot to 10-15 different poses (stays in impedance mode).
At each pose, press Enter to capture T_target2cam (ArUco) and T_gripper2base (FK).
Solve T_cam2base using cv2.calibrateRobotWorldHandEye().

Usage:
    python scripts/calibrate_cam_to_robot.py
    python scripts/calibrate_cam_to_robot.py --marker-id 55 --marker-size 0.15
    python scripts/calibrate_cam_to_robot.py --load  # re-solve from saved data
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

URL = "http://192.168.100.1:5000/"
SAVE_DIR = Path("calibration_data")

ACTION_SCALE = 0.015
ROTATION_SCALE = 0.035
DEADZONE = 0.05


def start_d455(width=640, height=480, fps=15):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)

    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    cam_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ])
    dist_coeffs = np.array(intrinsics.coeffs[:5])

    print(f"D455: {width}x{height}@{fps}fps")
    print(f"  fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}")

    time.sleep(3)
    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=10000)

    return pipeline, cam_matrix, dist_coeffs


def start_spacemouse():
    from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
    expert = SpaceMouseExpert()
    print("SpaceMouse connected")
    return expert


def detect_marker_pose(color_img, cam_matrix, dist_coeffs, marker_size, marker_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(color_img)

    vis = color_img.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    if ids is None:
        return None, None, vis

    half = marker_size / 2.0
    obj_points = np.array([
        [-half, half, 0], [half, half, 0],
        [half, -half, 0], [-half, -half, 0],
    ], dtype=np.float32)

    for i, mid in enumerate(ids.flatten()):
        if mid == marker_id:
            ok, rvec, tvec = cv2.solvePnP(
                obj_points, corners[i].reshape(4, 2), cam_matrix, dist_coeffs,
            )
            if not ok:
                continue
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            cv2.drawFrameAxes(vis, cam_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
            return rvec, tvec, vis

    return None, None, vis


def get_tcp_pose():
    r = requests.post(URL + "getstate", timeout=2.0)
    r.raise_for_status()
    state = r.json()
    pos = np.array(state["pose"][:3])
    return pos, state["pose"]


def send_pose(pose7):
    try:
        requests.post(URL + "pose", json={"arr": list(pose7)}, timeout=0.5)
    except requests.exceptions.Timeout:
        pass


def apply_spacemouse(target_xyz, target_quat, action6):
    dz_fn = lambda v: 0.0 if abs(v) < DEADZONE else v
    dxyz = [dz_fn(v) * ACTION_SCALE for v in action6[:3]]
    drpy = [dz_fn(v) * ROTATION_SCALE for v in action6[3:]]

    target_xyz[0] += dxyz[0]
    target_xyz[1] += dxyz[1]
    target_xyz[2] += dxyz[2]

    if abs(drpy[0]) + abs(drpy[1]) + abs(drpy[2]) > 1e-6:
        new = R.from_quat(target_quat) * R.from_euler("xyz", drpy)
        target_quat[:] = list(new.as_quat())

    return target_xyz, target_quat


def pose7_to_T(pose7):
    """Convert [x, y, z, qw, qx, qy, qz] to 4x4 matrix."""
    t = np.array(pose7[:3])
    quat_wxyz = pose7[3:]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot = R.from_quat(quat_xyzw)

    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = t
    return T


def rvec_tvec_to_RT(rvec, tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat, tvec.reshape(3, 1)


def solve_hand_eye(data_pairs):
    R_base2gripper_list = []
    t_base2gripper_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    for pair in data_pairs:
        # FK gives T_gripper2base (gripper pose in base frame)
        # OpenCV wants T_base2gripper = inv(T_gripper2base)
        T_g2b = pose7_to_T(pair["pose7"])
        T_b2g = np.linalg.inv(T_g2b)
        R_base2gripper_list.append(T_b2g[:3, :3])
        t_base2gripper_list.append(T_b2g[:3, 3].reshape(3, 1))

        # ArUco gives T_marker2cam (marker pose in camera frame) — correct as-is
        R_t2c, t_t2c = rvec_tvec_to_RT(
            np.array(pair["rvec"]), np.array(pair["tvec"]),
        )
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(t_t2c)

    R_base2world, t_base2world, R_gripper2target, t_gripper2target = \
        cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_target2cam_list,
            t_world2cam=t_target2cam_list,
            R_base2gripper=R_base2gripper_list,
            t_base2gripper=t_base2gripper_list,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        )

    # Output R_base2world = T_base_to_cam (for eye-to-hand, "world" = camera frame)
    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R_base2world
    T_base2cam[:3, 3] = t_base2world.flatten()

    T_cam2base = np.linalg.inv(T_base2cam)

    # T_gripper2marker: transforms from gripper frame to marker frame
    T_gripper2marker = np.eye(4)
    T_gripper2marker[:3, :3] = R_gripper2target
    T_gripper2marker[:3, 3] = t_gripper2target.flatten()

    return T_cam2base, T_gripper2marker


def compute_reprojection_errors(data_pairs, T_cam2base, T_gripper2marker):
    errors = []
    for pair in data_pairs:
        T_g2b = pose7_to_T(pair["pose7"])
        R_t2c, t_t2c = rvec_tvec_to_RT(np.array(pair["rvec"]), np.array(pair["tvec"]))
        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_t2c
        T_t2c[:3, 3] = t_t2c.flatten()

        # Marker center in base frame via camera path
        p_marker_in_cam = T_t2c[:3, 3]
        p_marker_base_via_cam = (T_cam2base @ np.append(p_marker_in_cam, 1.0))[:3]

        # Marker center in base frame via FK path
        # marker origin [0,0,0] in marker frame → gripper frame → base frame
        T_marker2gripper = np.linalg.inv(T_gripper2marker)
        p_marker_base_via_fk = (T_g2b @ T_marker2gripper @ np.array([0, 0, 0, 1.0]))[:3]

        err = np.linalg.norm(p_marker_base_via_cam - p_marker_base_via_fk)
        errors.append(err)

    return errors


def print_result(T_cam2base, T_gripper2marker, errors):
    print("\n=== T_cam_to_robot (4x4) ===")
    for row in T_cam2base:
        print(f"  [{row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f} {row[3]:+.6f}]")

    print("\n=== T_gripper_to_marker (4x4) ===")
    for row in T_gripper2marker:
        print(f"  [{row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f} {row[3]:+.6f}]")

    print(f"\n=== Reprojection Errors ===")
    for i, e in enumerate(errors):
        quality = "OK" if e < 0.01 else "WARN" if e < 0.02 else "BAD"
        print(f"  pose {i+1}: {e*1000:.1f} mm  [{quality}]")
    print(f"  mean: {np.mean(errors)*1000:.1f} mm")
    print(f"  max:  {np.max(errors)*1000:.1f} mm")

    if np.mean(errors) < 0.01:
        print("\n  Result: GOOD (mean < 10mm)")
    elif np.mean(errors) < 0.02:
        print("\n  Result: ACCEPTABLE (mean < 20mm)")
    else:
        print("\n  Result: POOR — consider re-calibrating with better poses")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-id", type=int, default=55)
    ap.add_argument("--marker-size", type=float, default=0.15, help="marker edge length in meters")
    ap.add_argument("--load", action="store_true", help="re-solve from saved data")
    args = ap.parse_args()

    SAVE_DIR.mkdir(exist_ok=True)
    pairs_file = SAVE_DIR / "handeye_pairs.json"

    if args.load:
        data = json.loads(pairs_file.read_text())
        print(f"Loaded {len(data['pairs'])} poses from {pairs_file}")
        T_cam2base, T_g2m = solve_hand_eye(data["pairs"])
        errors = compute_reprojection_errors(data["pairs"], T_cam2base, T_g2m)
        print_result(T_cam2base, T_g2m, errors)
        np.save(SAVE_DIR / "T_cam_to_robot.npy", T_cam2base)
        np.save(SAVE_DIR / "T_gripper_to_marker.npy", T_g2m)
        print(f"\nSaved to {SAVE_DIR}/")
        return

    # Init hardware
    pipeline, cam_matrix, dist_coeffs = start_d455()
    expert = start_spacemouse()

    # Init target pose from current robot state
    state = requests.post(URL + "getstate", timeout=2.0).json()
    target_xyz = list(state["pose"][:3])
    # /getstate returns [w,x,y,z], scipy needs [x,y,z,w]
    qw, qx, qy, qz = state["pose"][3:]
    target_quat = [qx, qy, qz, qw]  # scipy convention

    data_pairs = []

    print(f"\nArUco: DICT_5X5_100, ID={args.marker_id}, size={args.marker_size}m")
    print("\n=== Eye-to-Hand Calibration (SpaceMouse) ===")
    print("  SpaceMouse: move robot to different poses")
    print("  ENTER: capture current pose")
    print("  Q: finish and solve (need >= 5 poses)")
    print("  Tips: vary BOTH position and wrist rotation!\n")

    try:
        while True:
            t0 = time.time()

            # SpaceMouse → move robot
            action6, buttons = expert.get_action()
            target_xyz, target_quat = apply_spacemouse(target_xyz, target_quat, action6)

            # Convert back to [w,x,y,z] for /pose
            qx, qy, qz, qw = target_quat
            pose7 = [*target_xyz, qw, qx, qy, qz]
            send_pose(pose7)

            # Camera
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_img = np.asanyarray(frames.get_color_frame().get_data())

            rvec, tvec, vis = detect_marker_pose(
                color_img, cam_matrix, dist_coeffs, args.marker_size, args.marker_id
            )

            # Overlay
            if rvec is not None:
                cv2.putText(vis, f"MARKER OK  z={tvec[2]:.3f}m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "NO MARKER", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(vis, f"Poses: {len(data_pairs)}  |  SpaceMouse active", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis, f"TCP: ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Hand-Eye Calibration", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter
                if rvec is None:
                    print("  [!] No marker detected — move so D455 can see it")
                    continue

                try:
                    pos, current_pose7 = get_tcp_pose()
                except Exception as e:
                    print(f"  [!] getstate failed: {e}")
                    continue

                pair = {
                    "rvec": rvec.tolist(),
                    "tvec": tvec.tolist(),
                    "pose7": current_pose7,
                }
                data_pairs.append(pair)
                print(f"  [{len(data_pairs):2d}] marker_z={tvec[2]:.3f}m  "
                      f"tcp=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

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

    if len(data_pairs) < 3:
        print(f"\nOnly {len(data_pairs)} poses — need at least 3. Aborting.")
        return

    # Save raw data
    save_data = {
        "marker_id": args.marker_id,
        "marker_size": args.marker_size,
        "cam_matrix": cam_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "pairs": data_pairs,
    }
    pairs_file.write_text(json.dumps(save_data, indent=2))
    print(f"\nSaved {len(data_pairs)} poses to {pairs_file}")

    # Solve
    T_cam2base, T_g2m = solve_hand_eye(data_pairs)
    errors = compute_reprojection_errors(data_pairs, T_cam2base, T_g2m)
    print_result(T_cam2base, T_g2m, errors)

    np.save(SAVE_DIR / "T_cam_to_robot.npy", T_cam2base)
    np.save(SAVE_DIR / "T_gripper_to_marker.npy", T_g2m)
    print(f"\nSaved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
