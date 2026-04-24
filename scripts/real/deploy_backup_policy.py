"""Real-robot backup policy deployment via HierarchicalSupervisor.

Three-state FSM (driven by frrl.rl.supervisor):
  TASK     — SpaceMouse controls the arm (task policy placeholder)
  BACKUP   — Backup policy actively dodges a detected hand
  HOMING   — Returns TCP to the 6D pose recorded when TASK was interrupted,
             then hands control back to TASK (seamless resume).

Transitions (center-to-center distance, matches sim training semantics):
  TASK → BACKUP      : min_hand_dist < d_safe (0.30m hand_body_equiv ↔ bbox_center)
  BACKUP → HOMING    : min_hand_dist > d_clear (0.35m) for clear_n_steps (3) steps
  HOMING → BACKUP    : hand returns (dist < d_safe); tcp_start retained
  HOMING → TASK      : pos_err < 2cm AND rot_err < 2.9° (6D converged)

Observation for BACKUP policy (per frame, 28D × 3 stack = 84D):
  robot_state (18D): q(7) + dq(7) + gripper(1) + noisy_tcp(3) — from /getstate_true
  obstacle (10D)  : active(1) + hand_pos(3) + hand_vel(3) + hand_rel_pos(3)
                    hand_pos is the raw bbox geometric center, matching sim
                    training's obstacle_pos semantics (center-to-center dist).

Usage:
    python scripts/real/deploy_backup_policy.py
    python scripts/real/deploy_backup_policy.py --checkpoint checkpoints/backup_policy_s1_v2_newgeom_145k
    python scripts/real/deploy_backup_policy.py --dry-run   # no /pose commands, test obs/inference only
    python scripts/real/deploy_backup_policy.py --bias "0.1,0,0,0,0,0,0"  # inject J1 bias
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
from frrl.fault_injection import BiasMonitor
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.rl.supervisor import HierarchicalSupervisor, Mode
from frrl.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert
from frrl.robots.franka_real.vision.hand_detector import HandDetector

URL = "http://192.168.100.1:5000/"

# HierarchicalSupervisor thresholds (Route A: center-to-center semantics)
# Sim training uses hand_body-to-obstacle center distance with
# ARM_COLLISION_DIST = 0.135m (hard termination) and spawn range 0.21-0.30m.
# D_SAFE=0.30 lands BACKUP trigger right at sim's max spawn distance (fully
# in-distribution); D_CLEAR=0.35 gives a 5cm hysteresis band.
D_SAFE = 0.30          # m — center-to-center dist < d_safe → enter BACKUP
D_CLEAR = 0.35         # m — center-to-center dist > d_clear → allow BACKUP → HOMING
CLEAR_N_STEPS = 3      # consecutive clear steps before BACKUP → HOMING
HOMING_POS_TOL = 0.02  # m
HOMING_ROT_TOL = 0.05  # rad (~2.9°)

# Action scales — apply LOOKAHEAD multiplier to all modes to compensate for
# impedance controller lag under "target = actual + delta" scheme.
BACKUP_ACTION_SCALE = 0.025    # m/step at full deflection
# × LOOKAHEAD=2.0 = 0.1 rad/step，与 sim ROT_ACTION_SCALE 对齐；
# 旋转刚度只有平动的 7.5%，阻抗跟踪滞后会把实际增量削弱到 ~0.05 rad。
BACKUP_ROTATION_SCALE = 0.05
TASK_ACTION_SCALE = 0.025      # same scale as backup (SpaceMouse feels responsive)
TASK_ROTATION_SCALE = 0.040

# Look-ahead: commands target further than single-step delta, giving impedance
# controller more pull force to overcome its tracking lag.
LOOKAHEAD = 2.0

# Rate limit: max Cartesian speed (m/s). Clips single-step delta.
MAX_CART_SPEED = 0.30          # m/s — Franka nominal max is ~2 m/s

# Franka flange → pinch_site (TCP) offset along gripper +z axis.
# Sim's arm bounding sphere is centered at panda_hand body (= flange, mocap weld
# point, rotation-invariant). Real's state["pose"][:3] is at pinch_site (TCP).
# To align, we reconstruct the flange-equivalent point:
#   hand_body_equiv = TCP - TCP_OFFSET * gripper_z_base
# and use it as the center for min_hand_dist (matches sim's collision-check
# reference point). See docs/backup_policy_deployment.md §6.4 for rationale.
TCP_OFFSET = 0.1034  # m, Franka Hand kinematic flange→pinch offset

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


def quat_xyzw_to_wxyz(q_xyzw):
    """Franka server (scipy) → HomingController (sim env) convention."""
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def compute_hand_body_equiv(tcp_pos, quat_xyzw):
    """Sim-equivalent arm sphere center: flange point 10.34cm behind TCP along
    the gripper +z axis. Matches panda_hand body xpos (= mocap weld point) in
    sim, which is the reference center for collision termination at training
    time. Use true (unbiased) pose so real geometry matches bbox_center frame.
    """
    gripper_z_base = R.from_quat(quat_xyzw).apply([0, 0, 1])
    return tcp_pos - TCP_OFFSET * gripper_z_base


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
                    help="inject bias at startup, 7 comma-separated rad values e.g. '0,0,0,0.2,0,0,0'. "
                         "For NEGATIVE values use --bias=-0.2,0,0,0,0,0,0 (equals sign, no space, "
                         "otherwise argparse treats leading '-' as a flag)")
    ap.add_argument("--clear-bias", action="store_true", help="clear bias at startup")
    ap.add_argument("--no-workspace-clamp", action="store_true",
                    help="disable cartesian workspace clip on target pose (E-STOP READY!)")
    ap.add_argument("--plot-joints", action="store_true",
                    help="open a matplotlib window plotting q_true vs q_biased over time "
                         "(only joints with nonzero bias are shown)")
    ap.add_argument("--save-plot-data", type=str, default=None,
                    help="save the joint bias time series to this npz path on exit")
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
    state0 = get_state_biased()
    target_xyz = np.array(state0["pose"][:3], dtype=np.float64)
    target_quat_xyzw = list(state0["pose"][3:])
    print(f"[OK] Initial target (biased frame): xyz={np.round(target_xyz, 3)}")
    print(f"     quat(xyzw)={np.round(target_quat_xyzw, 3)}")

    # --- HierarchicalSupervisor (3-state FSM with 6D homing) ---
    supervisor = HierarchicalSupervisor(
        d_safe=D_SAFE,
        d_clear=D_CLEAR,
        clear_n_steps=CLEAR_N_STEPS,
        homing_pos_tol=HOMING_POS_TOL,
        homing_rot_tol=HOMING_ROT_TOL,
        # Homing controller's internal scale must match what we apply to /pose.
        # We re-use BACKUP scales so the homing response rate matches the same
        # feedback loop the impedance controller is tuned for.
        homing_action_scale=BACKUP_ACTION_SCALE * LOOKAHEAD,
        homing_rot_action_scale=BACKUP_ROTATION_SCALE * LOOKAHEAD,
        homing_kp_pos=1.0,
        homing_kp_rot=1.0,
    )
    supervisor.reset()

    # Observation buffer for backup policy (3-frame stack)
    obs_buf = deque(maxlen=OBS_STACK)

    # Hand velocity tracking
    last_hand_pos = None
    last_hand_time = None

    # SpaceMouse gripper edge-detection (TASK mode only)
    prev_buttons = [0, 0, 0, 0]
    gripper_state = "open"  # track commanded state to avoid redundant HTTP calls

    # Bias monitor (lazy: only draws once a nonzero bias is observed)
    bias_monitor = None
    if args.plot_joints or args.save_plot_data:
        bias_monitor = BiasMonitor(
            history_seconds=30.0,
            sample_hz=CTRL_HZ,
            update_hz=2.0,
            save_path=args.save_plot_data,
            render=args.plot_joints,
        )
        if args.plot_joints:
            print("[OK] BiasMonitor active (draws at 2 Hz; only joints with bias shown)")
        if args.save_plot_data:
            print(f"[OK] BiasMonitor will save timeseries to {args.save_plot_data} on exit")

    # For visualization
    rng = np.random.default_rng(0)

    print("\n=== Deployment Active ===")
    print("  TASK    (green)  — SpaceMouse controls arm + gripper (btn0=close, btn1=open)")
    print(f"  BACKUP  (red)    — hand_body_equiv-to-bbox center-dist < {D_SAFE*100:.0f}cm triggers policy avoidance")
    print("  HOMING  (orange) — after 3 consecutive clear frames, return to tcp_start")
    if args.no_workspace_clamp:
        print("  !! WORKSPACE CLAMP DISABLED — target pose can go anywhere. E-STOP READY.")
    else:
        print(f"  workspace clamp: xyz in [{WORKSPACE_MIN.tolist()}, {WORKSPACE_MAX.tolist()}]")
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

            # Route A alignment: policy obs uses obstacle geometric center
            # (matches sim training's obstacle_pos semantics). FSM uses
            # hand_body_equiv ↔ bbox_center center-to-center distance (matches
            # sim's arm_sphere_center ↔ obstacle_center collision reference).
            r_hand_debug = 0.0
            if hand.active:
                bbox_center = hand.pos_robot  # raw YOLO bbox center in robot frame
                bbox_size = hand.bbox_size_3d  # (w, h) in meters
                r_hand_debug = 0.5 * float(np.linalg.norm(bbox_size[:2]))  # viz only

                # Policy observation: raw bbox center (no Minkowski inflation).
                hand_pos = bbox_center.copy()

                # FSM trigger distance: center-to-center between flange-equiv
                # (sim's arm sphere center) and the raw hand bbox center.
                # Use true (unbiased) pose so distance reflects real geometry.
                hand_body_equiv = compute_hand_body_equiv(fk_tcp, state["pose"][3:])
                hand_dist = float(np.linalg.norm(hand_body_equiv - bbox_center))

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

            # --- BiasMonitor: record per-joint true vs biased values ---
            if bias_monitor is not None:
                # /getstate already includes the current bias vector; fall back
                # to (q_biased - q_true) if for some reason it's missing.
                active_bias = state_biased.get("bias")
                if active_bias is None:
                    active_bias = (np.array(state_biased["q"]) - np.array(state["q"])).tolist()
                bias_monitor.update(state["q"], state_biased["q"], active_bias)

            # --- TCP in controller frame (biased) — input to supervisor ---
            # Supervisor/HomingController use [w, x, y, z] quaternion convention.
            actual_xyz_biased = np.array(state_biased["pose"][:3], dtype=np.float64)
            actual_quat_xyzw = list(state_biased["pose"][3:])
            actual_quat_wxyz = quat_xyzw_to_wxyz(actual_quat_xyzw)

            # --- FSM step ---
            prev_mode = supervisor.mode
            # Supervisor needs a finite hand distance; use a large value when no hand.
            dist_for_fsm = hand_dist if hand_dist is not None else 1e9
            new_mode = supervisor.step(
                min_hand_dist=dist_for_fsm,
                tcp_current_pos=actual_xyz_biased,
                tcp_current_quat=actual_quat_wxyz,
            )
            if new_mode != prev_mode:
                extra = ""
                if new_mode == Mode.BACKUP and supervisor.tcp_start_pos is not None:
                    tsp = supervisor.tcp_start_pos
                    extra = f"  tcp_start=({tsp[0]:.3f},{tsp[1]:.3f},{tsp[2]:.3f})"
                print(f"[FSM] {prev_mode.name} -> {new_mode.name}{extra}")
                if prev_mode != Mode.BACKUP and new_mode == Mode.BACKUP:
                    # Fresh backup episode: reset obs buffer
                    obs_buf.clear()

            # --- SpaceMouse gripper control (TASK mode only, edge-triggered) ---
            # Reading is cheap; doing it here ensures we always have fresh button state.
            sm_latest, sm_buttons = spacemouse.get_action()
            if new_mode == Mode.TASK and len(sm_buttons) >= 2:
                # buttons[0] = close, buttons[1] = open (frrl convention)
                if sm_buttons[0] and not prev_buttons[0] and gripper_state != "closed":
                    try:
                        requests.post(URL + "close_gripper", timeout=1.0)
                        gripper_state = "closed"
                        print("[gripper] close")
                    except Exception as e:
                        print(f"[gripper] close failed: {e}")
                elif sm_buttons[1] and not prev_buttons[1] and gripper_state != "open":
                    try:
                        requests.post(URL + "open_gripper", timeout=1.0)
                        gripper_state = "open"
                        print("[gripper] open")
                    except Exception as e:
                        print(f"[gripper] open failed: {e}")
            prev_buttons = list(sm_buttons) if sm_buttons else [0, 0, 0, 0]

            # --- Action via supervisor.select_action (lazy callbacks) ---
            def task_action_fn():
                sm = [0.0 if abs(v) < 0.05 else v for v in sm_latest]
                return np.array([*sm[:6], 0.0], dtype=np.float32)  # 7D [dx,dy,dz,drx,dry,drz,grip]

            def backup_action_fn():
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
                return np.array([*action6, 0.0], dtype=np.float32)

            action7 = supervisor.select_action(
                task_action_fn, backup_action_fn,
                tcp_current_pos=actual_xyz_biased,
                tcp_current_quat=actual_quat_wxyz,
            )

            # Pick scale per mode (homing uses same as backup — see supervisor init)
            if new_mode == Mode.TASK:
                scale_xyz = TASK_ACTION_SCALE * LOOKAHEAD
                scale_rot = TASK_ROTATION_SCALE * LOOKAHEAD
            else:  # BACKUP or HOMING
                scale_xyz = BACKUP_ACTION_SCALE * LOOKAHEAD
                scale_rot = BACKUP_ROTATION_SCALE * LOOKAHEAD

            action_scaled_xyz = action7[:3] * scale_xyz
            action_scaled_rpy = action7[3:6] * scale_rot

            # --- Apply delta to CURRENT BIASED pose (controller's frame) ---
            # Rate limit the xyz delta
            delta_mag = np.linalg.norm(action_scaled_xyz)
            max_step = MAX_CART_SPEED * dt
            if delta_mag > max_step:
                action_scaled_xyz = action_scaled_xyz * (max_step / delta_mag)

            target_xyz = actual_xyz_biased + action_scaled_xyz
            if not args.no_workspace_clamp:
                target_xyz = np.clip(target_xyz, WORKSPACE_MIN, WORKSPACE_MAX)

            if np.any(np.abs(action_scaled_rpy) > 1e-6):
                # HomingController returns axis-angle vector (|v| = angle, v/|v| = axis).
                # Backup policy's rpy is also naturally a rotation vector at small angles.
                # from_rotvec is the correct decoder for both; from_euler("xyz") would
                # interpret the vector as sequential Euler angles.
                cur_R = R.from_quat(actual_quat_xyzw)
                dR = R.from_rotvec(action_scaled_rpy)
                target_quat_xyzw = list((cur_R * dR).as_quat())
            else:
                target_quat_xyzw = actual_quat_xyzw

            if not args.dry_run:
                send_pose(target_xyz, target_quat_xyzw)

            # --- Visualization ---
            vis = hand_detector.draw_detection(color_img, hand)
            mode_colors = {
                Mode.TASK:   (0, 255, 0),      # green
                Mode.BACKUP: (0, 0, 255),      # red
                Mode.HOMING: (0, 165, 255),    # orange
            }
            mode_color = mode_colors[new_mode]
            cv2.putText(vis, f"MODE: {new_mode.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

            if hand_dist is not None:
                dist_color = (0, 0, 255) if hand_dist < D_SAFE else (
                    (0, 165, 255) if hand_dist < D_CLEAR else (0, 255, 0))
                cv2.putText(vis,
                            f"center-dist: {hand_dist*1000:.0f}mm  "
                            f"(r_hand={r_hand_debug*1000:.0f}mm)",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_color, 2)

            # State-specific annotations
            if new_mode == Mode.BACKUP:
                cv2.putText(vis,
                            f"backup | clear_count {supervisor.clear_count}/{CLEAR_N_STEPS}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif new_mode == Mode.HOMING and supervisor.tcp_start_pos is not None:
                tsp = supervisor.tcp_start_pos
                pos_err = float(np.linalg.norm(actual_xyz_biased - tsp)) * 1000
                cv2.putText(vis,
                            f"homing | pos_err={pos_err:.0f}mm tol={HOMING_POS_TOL*1000:.0f}mm",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            cv2.putText(vis, f"TCP target: ({target_xyz[0]:.3f},{target_xyz[1]:.3f},{target_xyz[2]:.3f})",
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
        # Clear any injected bias so it doesn't persist to the next run.
        try:
            requests.post(URL + "clear_encoder_bias", timeout=2.0)
            print("[cleanup] encoder_bias cleared to zero")
        except Exception as e:
            print(f"[cleanup] WARNING: failed to clear bias: {e}")
            print("          run  curl -X POST http://192.168.100.1:5000/clear_encoder_bias  manually")
        if bias_monitor is not None:
            bias_monitor.close()
        cv2.destroyAllWindows()
        hand_detector.stop()
        spacemouse.close()
        print("Done")


if __name__ == "__main__":
    main()
