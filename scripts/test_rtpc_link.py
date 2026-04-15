"""Minimal link test: GPU station <-> RT PC Flask server.

Usage:
    python scripts/test_rtpc_link.py                 # getstate only (safe)
    python scripts/test_rtpc_link.py --dry-pose      # build pose payload, print it
    python scripts/test_rtpc_link.py --send-pose     # POST /pose (no motion: target=current)
    python scripts/test_rtpc_link.py --move-down     # TCP -10cm z, then back to start
    python scripts/test_rtpc_link.py --gripper       # open -> close -> open (safe)
    python scripts/test_rtpc_link.py --reset         # /jointreset — WILL MOVE TO JOINT HOME
    python scripts/test_rtpc_link.py --teleop        # SpaceMouse -> /pose, 10 Hz loop
"""
import argparse
import json
import sys
import time
import requests

URL = "http://192.168.100.1:5000/"


def getstate():
    r = requests.post(URL + "getstate", timeout=2.0)
    r.raise_for_status()
    s = r.json()
    print("=== /getstate ===")
    print(f"  q          = {[round(x, 4) for x in s['q']]}")
    print(f"  dq         = {[round(x, 4) for x in s['dq']]}")
    print(f"  pose(xyz)  = {[round(x, 4) for x in s['pose'][:3]]}")
    print(f"  pose(quat) = {[round(x, 4) for x in s['pose'][3:]]}")
    print(f"  gripper    = {round(s['gripper_pos'], 4)}")
    return s


def pose_payload_from_current(state):
    """Build a pose command payload where target == current pose (no-op)."""
    return {"arr": state["pose"]}


def send_pose(pose7):
    r = requests.post(URL + "pose", json={"arr": list(pose7)}, timeout=10.0)
    print(f"  POST /pose  status={r.status_code}  body={r.text[:120]}")
    return r


def move_down_and_back(state, dz=0.10, settle=1.5):
    start_pose = list(state["pose"])
    target = list(start_pose)
    target[2] -= dz  # z down

    print(f"\n=== move down {dz*100:.0f}cm ===")
    print(f"  start z = {start_pose[2]:.4f}")
    print(f"  target z = {target[2]:.4f}")
    send_pose(target)
    time.sleep(settle)

    after = requests.post(URL + "getstate", timeout=2.0).json()
    print(f"  after z = {after['pose'][2]:.4f}  (delta = {after['pose'][2] - start_pose[2]:+.4f})")

    print(f"\n=== return to start ===")
    send_pose(start_pose)
    time.sleep(settle)

    final = requests.post(URL + "getstate", timeout=2.0).json()
    err = [final["pose"][i] - start_pose[i] for i in range(3)]
    print(f"  final xyz = {[round(x,4) for x in final['pose'][:3]]}")
    print(f"  xyz error = {[round(x,4) for x in err]}  (norm = {sum(e*e for e in err)**0.5:.4f} m)")


def gripper_cycle(settle=1.0):
    print("\n=== gripper open -> close -> open ===")
    for verb in ["open_gripper", "close_gripper", "open_gripper"]:
        r = requests.post(URL + verb, timeout=5.0)
        time.sleep(settle)
        state = requests.post(URL + "getstate", timeout=2.0).json()
        print(f"  {verb:<15s} status={r.status_code}  gripper_pos={state['gripper_pos']:.4f}")


def teleop_loop(action_scale=0.015, rotation_scale=0.035, rate_hz=10.0, deadzone=0.05):
    """SpaceMouse -> /pose using the project's SpaceMouseExpert.

    Axis mapping and button convention follow frrl.teleoperators.spacemouse:
        action = [-s.y, s.x, s.z, -s.roll, -s.pitch, -s.yaw]
        button[0] = close_gripper  (press to close)
        button[1] = open_gripper   (press to open)
    """
    from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
    from scipy.spatial.transform import Rotation as R

    expert = SpaceMouseExpert()

    state = requests.post(URL + "getstate", timeout=2.0).json()
    target_xyz = list(state["pose"][:3])
    target_quat = list(state["pose"][3:])  # xyzw
    gripper_state = "open"  # track current commanded state to avoid spam

    print(f"Teleop @ {rate_hz:.0f} Hz  (action_scale={action_scale} m  rotation_scale={rotation_scale} rad)")
    print("  button[0] -> close gripper    button[1] -> open gripper")
    print(f"  start xyz = {[round(x,4) for x in target_xyz]}\n")

    dt = 1.0 / rate_hz
    try:
        while True:
            t0 = time.time()
            action6, buttons = expert.get_action()  # 6D already with frrl axis conv.

            # deadzone
            action6 = [0.0 if abs(v) < deadzone else v for v in action6]
            dxyz = [v * action_scale for v in action6[:3]]
            drpy = [v * rotation_scale for v in action6[3:]]

            target_xyz[0] += dxyz[0]
            target_xyz[1] += dxyz[1]
            target_xyz[2] += dxyz[2]

            if abs(drpy[0]) + abs(drpy[1]) + abs(drpy[2]) > 1e-6:
                new = R.from_quat(target_quat) * R.from_euler("xyz", drpy)
                target_quat = list(new.as_quat())

            # gripper: button[0]=close, button[1]=open (held = repeat is OK)
            if len(buttons) >= 2:
                if buttons[0] and gripper_state != "closed":
                    requests.post(URL + "close_gripper", timeout=1.0)
                    gripper_state = "closed"
                    print("\n  gripper -> close")
                elif buttons[1] and gripper_state != "open":
                    requests.post(URL + "open_gripper", timeout=1.0)
                    gripper_state = "open"
                    print("\n  gripper -> open")

            pose7 = [*target_xyz, *target_quat]
            try:
                requests.post(URL + "pose", json={"arr": pose7}, timeout=0.5)
            except requests.exceptions.Timeout:
                pass

            print(
                f"  xyz=({target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f})  "
                f"sm=({action6[0]:+.2f},{action6[1]:+.2f},{action6[2]:+.2f},"
                f"{action6[3]:+.2f},{action6[4]:+.2f},{action6[5]:+.2f})  "
                f"btn={list(buttons)}",
                end="\r",
                flush=True,
            )

            dt_left = dt - (time.time() - t0)
            if dt_left > 0:
                time.sleep(dt_left)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        expert.close()


def joint_reset(settle=5.0):
    print("\n=== /jointreset — robot WILL move to joint home ===")
    before = requests.post(URL + "getstate", timeout=2.0).json()
    print(f"  q before = {[round(x,4) for x in before['q']]}")
    print(f"  xyz before = {[round(x,4) for x in before['pose'][:3]]}")
    r = requests.post(URL + "jointreset", timeout=30.0)
    print(f"  POST /jointreset  status={r.status_code}  body={r.text[:120]}")
    time.sleep(settle)
    after = requests.post(URL + "getstate", timeout=2.0).json()
    print(f"  q after  = {[round(x,4) for x in after['q']]}")
    print(f"  xyz after  = {[round(x,4) for x in after['pose'][:3]]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-pose", action="store_true", help="print pose payload, do not send")
    ap.add_argument("--send-pose", action="store_true", help="POST the no-op pose payload")
    ap.add_argument("--move-down", action="store_true", help="TCP -10cm z, then back to start")
    ap.add_argument("--gripper", action="store_true", help="open -> close -> open cycle")
    ap.add_argument("--reset", action="store_true", help="/jointreset — WILL MOVE to joint home")
    ap.add_argument("--teleop", action="store_true", help="SpaceMouse teleop loop")
    ap.add_argument("--action-scale", type=float, default=0.015, help="m per step at full deflection")
    ap.add_argument("--rotation-scale", type=float, default=0.035, help="rad per step at full deflection")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    args = ap.parse_args()

    if args.teleop:
        teleop_loop(
            action_scale=args.action_scale,
            rotation_scale=args.rotation_scale,
            rate_hz=args.rate_hz,
        )
        return

    try:
        state = getstate()
    except Exception as e:
        print(f"[FAIL] getstate: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_pose or args.send_pose:
        payload = pose_payload_from_current(state)
        print("\n=== /pose payload (target = current pose, no-op) ===")
        print(json.dumps(payload, indent=2))

    if args.send_pose:
        print("\n=== POST /pose (no-op) ===")
        send_pose(state["pose"])

    if args.move_down:
        move_down_and_back(state, dz=0.10, settle=1.5)

    if args.gripper:
        gripper_cycle()

    if args.reset:
        joint_reset()

    print("\nOK")


if __name__ == "__main__":
    main()
