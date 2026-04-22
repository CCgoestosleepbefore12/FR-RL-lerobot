"""Real-robot backup policy deployment.

State machine:
  task (SpaceMouse control) <--trigger: hand_dist < 0.15m--> backup (policy control)
                           <--release: hand_dist > 0.20m OR backup_steps >= 10--

Observation (per frame, 28D):
  robot_state (18D): q(7) + dq(7) + gripper(1) + noisy_tcp(3)
    - From /getstate_true (unaffected by encoder bias injection)
    - TCP gets 5mm Gaussian noise added to match training DR
  obstacle (10D): active(1) + hand_pos(3) + hand_vel(3) + hand_rel_pos(3)

Frame stacked x3 -> 84D policy input.

Usage:
    python scripts/deploy_backup_policy.py
    python scripts/deploy_backup_policy.py --checkpoint checkpoints/backup_policy_s1
    python scripts/deploy_backup_policy.py --dry-run   # no /pose commands, test obs/inference only
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from scipy.spatial.transform import Rotation as R

import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
from frrl.configs.policies import PreTrainedConfig
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
from frrl.vision.hand_detector import HandDetector

URL = "http://192.168.100.1:5000/"

# Hand-TCP trigger/release (hysteresis avoids oscillation)
TRIGGER_DIST = 0.15
RELEASE_DIST = 0.20
MAX_BACKUP_STEPS = 10

# Action scales — apply LOOKAHEAD multiplier to both backup and task modes
# to compensate for impedance controller lag under "target = actual + delta" scheme.
BACKUP_ACTION_SCALE = 0.025    # m/step at full deflection
BACKUP_ROTATION_SCALE = 0.020
TASK_ACTION_SCALE = 0.025      # same scale as backup (SpaceMouse feels responsive)
TASK_ROTATION_SCALE = 0.040

# Look-ahead: commands target further than single-step delta, giving impedance
# controller more pull force to overcome its tracking lag.
LOOKAHEAD = 2.0

# Rate limit: max Cartesian speed (m/s). Clips single-step delta.
MAX_CART_SPEED = 0.30          # m/s — Franka nominal max is ~2 m/s

# Franka Hand bounding sphere radius (meters).
# Standard Franka Hand is ~85x90x120mm; bounding sphere ~70mm centered slightly
# behind TCP (pinch point). 0.06m is a conservative approximation.
R_GRIPPER = 0.06

OBS_STACK = 3
CTRL_HZ = 10.0

# Safety workspace (adjust to your setup)
WORKSPACE_MIN = np.array([0.20, -0.30, 0.10])
WORKSPACE_MAX = np.array([0.70,  0.30, 0.60])

TCP_NOISE_STD = 0.005  # matches training DR_TCP_POS_STD


def post(path, **kw):
    return requests.post(URL + path, timeout=2.0, **kw)


def get_state_true():
    """Unbiased state — use for POLICY OBSERVATION (matches training distribution)."""
    r = post("getstate_true")
    r.raise_for_status()
    return r.json()


def get_state_biased():
    """Biased state — use for TARGET construction so controller frame matches
    the frame we send /pose in. Without this, bias creates a phantom error the
    controller keeps trying to 'fix', causing weird drift."""
    r = post("getstate")
    r.raise_for_status()
    return r.json()


def send_pose(target_xyz, target_quat_xyzw):
    # franka_server expects [x, y, z, qx, qy, qz, qw] (scipy convention)
    pose7 = [*target_xyz.tolist(), *target_quat_xyzw]
    try:
        requests.post(URL + "pose", json={"arr": pose7}, timeout=0.5)
    except requests.exceptions.Timeout:
        pass


def compute_virtual_hand_pos(bbox_center, bbox_size_3d, tcp_pos, r_gripper=R_GRIPPER):
    """Minkowski trick: inflate hand sphere by gripper radius, shrink gripper to
    a point (TCP). The resulting point-to-point distance from TCP to the virtual
    hand point equals the surface-to-surface distance between the two bboxes.

    bbox_size_3d: (w, h) in meters, image-plane extent at hand depth.
    """
    # Hand bounding sphere = half of the 2D bbox diagonal
    r_hand = 0.5 * float(np.linalg.norm(bbox_size_3d[:2]))
    r_effective = r_gripper + r_hand

    direction = tcp_pos - bbox_center
    distance = float(np.linalg.norm(direction))

    if distance < 1e-6:
        return bbox_center.copy(), 0.0, r_hand
    if distance <= r_effective:
        # Surfaces overlap — policy sees "obstacle on TCP"
        return tcp_pos.copy(), 0.0, r_hand
    # Move hand point toward TCP by r_effective
    virtual = bbox_center + (direction / distance) * r_effective
    surface_dist = distance - r_effective
    return virtual, surface_dist, r_hand


def build_obs28(state, hand_pos, hand_vel, tcp_noisy):
    """Build 28D observation for backup policy."""
    q = np.asarray(state["q"], dtype=np.float32)
    dq = np.asarray(state["dq"], dtype=np.float32)
    gripper = np.array([state["gripper_pos"]], dtype=np.float32)
    # robot_state layout: q(7) + dq(7) + gripper(1) + tcp(3)
    robot_state = np.concatenate([q, dq, gripper, tcp_noisy.astype(np.float32)])

    # obstacle info (10D): active + pos + vel + rel_pos
    rel = (hand_pos - tcp_noisy).astype(np.float32)
    obstacle = np.concatenate([
        np.array([1.0], dtype=np.float32),
        hand_pos.astype(np.float32),
        hand_vel.astype(np.float32),
        rel,
    ])
    return np.concatenate([robot_state, obstacle])  # 28D


def stack_frames(buf, single_dim=28):
    """Concatenate last OBS_STACK frames, zero-pad if fewer."""
    if len(buf) < OBS_STACK:
        pad = [np.zeros(single_dim, dtype=np.float32)] * (OBS_STACK - len(buf))
        frames = [*pad, *buf]
    else:
        frames = list(buf)
    return np.concatenate(frames)  # OBS_STACK * single_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/backup_policy_s1")
    ap.add_argument("--dry-run", action="store_true", help="don't send /pose, just test obs/policy")
    ap.add_argument("--calibration", default="calibration_data/T_cam_to_robot.npy")
    ap.add_argument("--bias", type=str, default=None,
                    help="inject bias at startup, 7 comma-separated rad values e.g. '0,0,0,0.2,0,0,0'")
    ap.add_argument("--clear-bias", action="store_true", help="clear bias at startup")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load policy ---
    ckpt = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.pretrained_path = str(ckpt)
    cfg.device = device
    policy = SACPolicy.from_pretrained(str(ckpt), config=cfg)
    policy.eval().to(device)
    print(f"[OK] Policy loaded: {ckpt} on {device}")

    # --- Load calibration ---
    T_cam_to_robot = np.load(args.calibration)
    print(f"[OK] Calibration loaded: {args.calibration}")

    # --- Hand detector + D455 ---
    hand_detector = HandDetector(T_cam_to_robot=T_cam_to_robot)
    hand_detector.start()
    print("[OK] HandDetector started (D455 + WiLoR)")

    # --- SpaceMouse ---
    spacemouse = SpaceMouseExpert()
    print("[OK] SpaceMouse connected")

    # --- Enter impedance control mode so /pose commands are accepted ---
    # Stop-then-start to handle the case where a previous run left the
    # controller loaded (/startimp errors with "already running" in that case).
    try:
        # Clear any lingering error state from previous crash
        try:
            requests.post(URL + "clearerr", timeout=5.0)
        except Exception:
            pass
        # Stop any existing impedance controller (OK if not running)
        try:
            r_stop = requests.post(URL + "stopimp", timeout=10.0)
            print(f"[..] /stopimp: status={r_stop.status_code} body={r_stop.text[:60]}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[..] /stopimp (ignored): {e}")
        # Clear again after stop
        try:
            requests.post(URL + "clearerr", timeout=5.0)
        except Exception:
            pass
        time.sleep(0.3)
        # Now start fresh
        r = requests.post(URL + "startimp", timeout=15.0)
        print(f"[OK] /startimp: status={r.status_code} body={r.text[:60]}")
        time.sleep(1.0)
        # Verify
        _ = get_state_true()
        print("[OK] impedance controller active (getstate_true OK)")
    except Exception as e:
        print(f"[FAIL] controller setup failed: {e}")
        print("      Fix: restart franka_server.py on RT PC (kill + relaunch), then retry.")
        return

    # --- Bias injection controls ---
    if args.clear_bias:
        r = requests.post(URL + "clear_encoder_bias", timeout=5.0)
        print(f"[OK] /clear_encoder_bias: {r.status_code} {r.text[:40]}")
    if args.bias is not None:
        bias_vec = [float(v) for v in args.bias.split(",")]
        assert len(bias_vec) == 7, f"--bias needs 7 values, got {len(bias_vec)}"
        r = requests.post(URL + "set_encoder_bias", json={"bias": bias_vec}, timeout=5.0)
        print(f"[OK] /set_encoder_bias: {r.status_code} {r.text[:40]}")

    # Log current bias (whether from flags or left over)
    try:
        r = requests.post(URL + "get_encoder_bias", timeout=2.0)
        active_bias = r.json().get("bias", [0]*7)
        bias_str = ", ".join(f"{v:+.3f}" for v in active_bias)
        print(f"[INFO] encoder_bias active: [{bias_str}] rad")
        if np.any(np.abs(active_bias) > 1e-4):
            print("       *** BIAS ACTIVE — execution will be offset from commanded ***")
    except Exception:
        pass

    # --- Init target pose from current BIASED state (controller's frame) ---
    # Target must be in the same frame as the controller's internal "current"
    # view, which is FK(q_biased). We read /getstate to get that frame.
    state0 = get_state_biased()
    target_xyz = np.array(state0["pose"][:3], dtype=np.float64)
    target_quat_xyzw = list(state0["pose"][3:])
    print(f"[OK] Initial target (biased frame): xyz={np.round(target_xyz, 3)}")
    print(f"     quat(xyzw)={np.round(target_quat_xyzw, 3)}")

    # --- State machine ---
    mode = "task"
    backup_steps = 0
    obs_buf = deque(maxlen=OBS_STACK)

    # Hand velocity tracking
    last_hand_pos = None
    last_hand_time = None

    # For visualization
    rng = np.random.default_rng(0)

    print("\n=== Deployment Active ===")
    print("  SpaceMouse = task policy placeholder")
    print("  Put hand near TCP to trigger backup policy")
    print("  Press 'q' in window to quit\n")

    dt = 1.0 / CTRL_HZ
    try:
        while True:
            t0 = time.time()

            # --- Sensors ---
            # state_true: for POLICY OBSERVATION (matches training distribution)
            # state_biased: for TARGET construction (controller's reference frame)
            try:
                state = get_state_true()
                state_biased = get_state_biased()
            except Exception as e:
                print(f"[!] getstate failed: {e}")
                time.sleep(0.1)
                continue

            color_img, depth_img = hand_detector.get_frames()
            hand = hand_detector.detect(color_img, depth_img)

            # True TCP + 5mm noise (matches training DR)
            fk_tcp = np.array(state["pose"][:3], dtype=np.float64)
            tcp_noisy = fk_tcp + rng.normal(0, TCP_NOISE_STD, 3)

            # Hand position + velocity, with Minkowski inflation for box-vs-box
            # avoidance semantics.
            r_hand_debug = 0.0
            if hand.active:
                bbox_center = hand.pos_robot  # raw YOLO bbox center in robot frame
                bbox_size = hand.bbox_size_3d  # (w, h) in meters

                # Virtual hand point: distance from TCP to this point equals
                # surface-to-surface distance between gripper and hand bboxes.
                hand_pos, hand_dist, r_hand_debug = compute_virtual_hand_pos(
                    bbox_center, bbox_size, fk_tcp, r_gripper=R_GRIPPER,
                )

                # Velocity from raw bbox center (physical motion of the hand)
                now = time.time()
                if last_hand_pos is not None and last_hand_time is not None:
                    dt_h = max(now - last_hand_time, 1e-3)
                    hand_vel = (bbox_center - last_hand_pos) / dt_h
                else:
                    hand_vel = np.zeros(3)
                last_hand_pos, last_hand_time = bbox_center, now
            else:
                hand_pos = None
                hand_vel = np.zeros(3)
                hand_dist = None

            # --- State transitions ---
            if mode == "task":
                if hand_dist is not None and hand_dist < TRIGGER_DIST:
                    mode = "backup"
                    backup_steps = 0
                    obs_buf.clear()
                    print(f"[TRIGGER] hand_dist={hand_dist*1000:.0f}mm -> backup")
            elif mode == "backup":
                release = (
                    hand_dist is None or
                    hand_dist > RELEASE_DIST or
                    backup_steps >= MAX_BACKUP_STEPS
                )
                if release:
                    reason = ("lost" if hand_dist is None else
                              "far" if hand_dist > RELEASE_DIST else "timeout")
                    print(f"[RELEASE] reason={reason}, backup_steps={backup_steps} -> task")
                    mode = "task"
                    backup_steps = 0

            # --- Action ---
            if mode == "backup":
                # Backup policy path
                # Use last known hand if detection dropped
                use_hand_pos = hand_pos if hand_pos is not None else (
                    last_hand_pos if last_hand_pos is not None else fk_tcp
                )
                obs28 = build_obs28(state, use_hand_pos, hand_vel, tcp_noisy)
                obs_buf.append(obs28)
                obs84 = stack_frames(obs_buf, single_dim=28)
                obs_t = torch.from_numpy(obs84).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    action_t = policy.select_action(batch={"observation.state": obs_t})
                action6 = action_t.squeeze(0).cpu().numpy()[:6]

                # Scale + look-ahead
                action_scaled_xyz = action6[:3] * BACKUP_ACTION_SCALE * LOOKAHEAD
                action_scaled_rpy = action6[3:6] * BACKUP_ROTATION_SCALE * LOOKAHEAD
                backup_steps += 1

                # DIAGNOSTIC
                actual_xyz = np.array(state["pose"][:3])
                tgt_err = np.linalg.norm(target_xyz - actual_xyz) * 1000
                print(f"  [backup {backup_steps:2d}] "
                      f"dxyz={np.round(action_scaled_xyz*1000,1)}mm  "
                      f"ACTUAL_TCP=({actual_xyz[0]:.3f},{actual_xyz[1]:.3f},{actual_xyz[2]:.3f})  "
                      f"TGT_ERR={tgt_err:.0f}mm")

            else:  # task mode — SpaceMouse
                sm_action, _ = spacemouse.get_action()
                # Deadzone
                sm_action = [0.0 if abs(v) < 0.05 else v for v in sm_action]
                action_scaled_xyz = np.array(sm_action[:3]) * TASK_ACTION_SCALE * LOOKAHEAD
                action_scaled_rpy = np.array(sm_action[3:6]) * TASK_ROTATION_SCALE * LOOKAHEAD

            # --- Apply delta to CURRENT BIASED pose (controller's frame) ---
            # state_biased["pose"] = what controller considers "current" position.
            # Adding delta to this keeps target in a frame consistent with the
            # controller's internal reference, so no phantom error develops.
            actual_xyz = np.array(state_biased["pose"][:3], dtype=np.float64)
            actual_quat_xyzw = list(state_biased["pose"][3:])

            # Rate limit the xyz delta
            delta_mag = np.linalg.norm(action_scaled_xyz)
            max_step = MAX_CART_SPEED * dt
            if delta_mag > max_step:
                action_scaled_xyz = action_scaled_xyz * (max_step / delta_mag)

            target_xyz = actual_xyz + action_scaled_xyz
            target_xyz = np.clip(target_xyz, WORKSPACE_MIN, WORKSPACE_MAX)

            if np.any(np.abs(action_scaled_rpy) > 1e-6):
                cur_R = R.from_quat(actual_quat_xyzw)
                dR = R.from_euler("xyz", action_scaled_rpy)
                target_quat_xyzw = list((cur_R * dR).as_quat())
            else:
                target_quat_xyzw = actual_quat_xyzw

            if not args.dry_run:
                pose_resp = None
                try:
                    # franka_server expects [x, y, z, qx, qy, qz, qw]
                    pose_resp = requests.post(URL + "pose",
                        json={"arr": [*target_xyz.tolist(), *target_quat_xyzw]},
                        timeout=0.5)
                except requests.exceptions.Timeout:
                    pass
                if mode == "backup" and pose_resp is not None:
                    print(f"    /pose status={pose_resp.status_code}  body={pose_resp.text[:40]}  "
                          f"target=({target_xyz[0]:.3f},{target_xyz[1]:.3f},{target_xyz[2]:.3f})")

            # --- Visualization ---
            vis = hand_detector.draw_detection(color_img, hand)
            mode_color = (0, 0, 255) if mode == "backup" else (0, 255, 0)
            cv2.putText(vis, f"MODE: {mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            if hand_dist is not None:
                dist_color = (0, 0, 255) if hand_dist < TRIGGER_DIST else (
                    (0, 165, 255) if hand_dist < RELEASE_DIST else (0, 255, 0))
                cv2.putText(vis,
                            f"surface-dist: {hand_dist*1000:.0f}mm  (r_hand={r_hand_debug*1000:.0f}mm, r_grip={R_GRIPPER*1000:.0f}mm)",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_color, 2)
            if mode == "backup":
                cv2.putText(vis, f"backup step {backup_steps}/{MAX_BACKUP_STEPS}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis, f"TCP: ({target_xyz[0]:.3f},{target_xyz[1]:.3f},{target_xyz[2]:.3f})",
                        (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            cv2.imshow("Backup Policy Deployment", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # --- Pace ---
            left = dt - (time.time() - t0)
            if left > 0:
                time.sleep(left)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        cv2.destroyAllWindows()
        hand_detector.stop()
        spacemouse.close()
        print("Done")


if __name__ == "__main__":
    main()
