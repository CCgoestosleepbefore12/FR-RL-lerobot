"""BC pickup policy + backup safety policy 联合部署 via HierarchicalSupervisor.

基于 scripts/real/deploy_backup_policy.py 的 3-state FSM，把 task_action_fn
从 SpaceMouse 替换成 BC pickup policy 推理。BC 用 front + wrist 双相机 +
14D state（来自 /getstate_true 的 unbiased TCP pose + vel + gripper），跟
deploy_bc_inference.py 训练分布一致。

FSM 状态：
  TASK    — BC pickup policy 自主控制（state + 2 cam images）
  BACKUP  — Backup policy 主动避让检测到的手（84D 堆叠 state，无图像）
  HOMING  — BACKUP 退出后把 TCP 拉回 tcp_start（进入 BACKUP 时记录），
            6D 位姿收敛后切回 TASK（避免 BC 在 OOD pose 上输出异常）

切换条件（中心距，对齐 backup sim training 语义）：
  TASK → BACKUP : hand_body_equiv ↔ bbox_center 距离 < D_SAFE
  BACKUP → HOMING : 距离 > D_CLEAR 持续 CLEAR_N_STEPS 帧
  HOMING → TASK : pos_err < tol AND rot_err < tol

Action scale 按模式切换：
  TASK_*_SCALE   = (0.05, 0.1)  ← 跟 BC 训练时的 frrl pickup ACTION_SCALE 一致
  BACKUP_*_SCALE = (0.025, 0.05) ← 跟 backup sim 训练 scale 一致

相机配置（系统只有 2 个 RealSense，HandDetector 跟 BC 共享 front）：
  front (234222303420) — HandDetector 用（color+depth 检测手），同时 BC 复用其 RGB
                          （crop 工作面 ROI [94:314, 180:400] → resize 128×128）
  wrist (318122303303) — BC 独占，单独开 RealSenseCameraManager 拿 RGB

用法：
  python scripts/real/deploy_pickup_with_backup.py \\
      --bc-ckpt checkpoints/pickup_bc_20260429_171959/checkpoints/020000/pretrained_model \\
      --backup-ckpt checkpoints/backup_policy_s1_v3b_300k_95pct \\
      --ckpt-version v3

  # dry-run 不发 /pose，只测 obs / inference / FSM 切换
  python scripts/real/deploy_pickup_with_backup.py --bc-ckpt ... --dry-run
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
from frrl.rl.supervisor import HierarchicalSupervisor, Mode
from frrl.envs.real_config import center_square_crop, make_task_config
from frrl.envs.real import euler_2_quat
from frrl.robots.franka_real.cameras.realsense import RealSenseCameraManager
from frrl.robots.franka_real.vision.hand_detector import HandDetector

URL = "http://192.168.100.1:5000/"

# ---------- Backup FSM thresholds（沿用 deploy_backup_policy.py）----------
D_SAFE_BY_VERSION = {"v2": 0.30, "v3": 0.40}
D_CLEAR_BY_VERSION = {"v2": 0.35, "v3": 0.45}
DEFAULT_BACKUP_CKPT = {
    "v2": "checkpoints/backup_policy_s1_v2_newgeom_145k",
    "v3": "checkpoints/backup_policy_s1_v3b_300k_95pct",
}
CLEAR_N_STEPS = 3
HOMING_POS_TOL = 0.02
HOMING_ROT_TOL = 0.05

# ---------- Action scales（per mode）----------
# TASK 走 BC 训练时的 scale (frrl pickup action_scale=(0.05, 0.1, 1.0))；
# BACKUP 走 sim 训练 scale (0.025, 0.05)。两者通过 supervisor 切换不同分支。
TASK_ACTION_SCALE = 0.05      # BC training scale
TASK_ROTATION_SCALE = 0.10
BACKUP_ACTION_SCALE = 0.025
BACKUP_ROTATION_SCALE = 0.05

LOOKAHEAD = 2.0
MAX_CART_SPEED = 0.30
TCP_OFFSET = 0.1034
OBS_STACK = 3
CTRL_HZ = 10.0
TCP_NOISE_STD = 0.005

WORKSPACE_MIN = np.array([0.20, -0.30, 0.10])
WORKSPACE_MAX = np.array([0.70,  0.30, 0.60])

# 相机序列号（来自 frrl/envs/real_config.py 默认配置）
WRIST_CAMERA_SERIAL = "318122303303"  # 只 BC 用，单独开 RealSenseCameraManager
# front (234222303420) 由 HandDetector 占用，BC 复用其 color frame

# Image size for BC network input（跟训练对齐）
IMAGE_SIZE = (128, 128)

# front 工作面 ROI（来自 frrl/envs/real_config.py:make_pickup_config:image_crop["front"]
# = make_workspace_roi_crop(180, 94, 400, 314)）—— BC 训练时 front 帧是 [v=94:314, u=180:400]
# 220×220 ROI 然后 resize 到 128×128。
FRONT_CROP_VYUX = (94, 314, 180, 400)


def crop_resize_front(rgb_img):
    """对 HandDetector.get_frames() 的 RGB 帧做 BC 训练时同款 crop + resize。"""
    v0, v1, u0, u1 = FRONT_CROP_VYUX
    cropped = rgb_img[v0:v1, u0:u1]
    return cv2.resize(cropped, IMAGE_SIZE)


# ---------- HTTP helpers ----------
def post(path, **kw):
    return requests.post(URL + path, timeout=2.0, **kw)


def get_state_true():
    r = post("getstate_true")
    r.raise_for_status()
    return r.json()


def get_state_biased():
    r = post("getstate")
    r.raise_for_status()
    return r.json()


def send_pose(target_xyz, target_quat_xyzw):
    pose7 = [*target_xyz.tolist(), *target_quat_xyzw]
    try:
        requests.post(URL + "pose", json={"arr": pose7}, timeout=0.5)
    except requests.exceptions.Timeout:
        pass


def compute_hand_body_equiv(tcp_pos, quat_xyzw):
    """Flange 中心 = TCP 沿 gripper +z 退 10.34cm，对齐 sim 碰撞参考点。"""
    gripper_z_base = R.from_quat(quat_xyzw).apply([0, 0, 1])
    return tcp_pos - TCP_OFFSET * gripper_z_base


# ---------- Backup obs helpers（沿用 deploy_backup_policy.py）----------
def build_obs28(state, hand_pos, hand_vel, tcp_noisy):
    """28D obs for backup policy: q(7) + dq(7) + gripper(1) + tcp(3) + obstacle(10)."""
    q = np.asarray(state["q"], dtype=np.float32)
    dq = np.asarray(state["dq"], dtype=np.float32)
    gripper = np.array([state["gripper_pos"]], dtype=np.float32)
    robot_state = np.concatenate([q, dq, gripper, tcp_noisy.astype(np.float32)])
    rel = (hand_pos - tcp_noisy).astype(np.float32)
    obstacle = np.concatenate([
        np.array([1.0], dtype=np.float32),
        hand_pos.astype(np.float32),
        hand_vel.astype(np.float32),
        rel,
    ])
    return np.concatenate([robot_state, obstacle]).astype(np.float32)


def stack_frames(buf, single_dim=28):
    if len(buf) < OBS_STACK:
        pad = [np.zeros(single_dim, dtype=np.float32)] * (OBS_STACK - len(buf))
        frames = pad + list(buf)
    else:
        frames = list(buf)
    return np.concatenate(frames)


# ---------- BC obs helpers（新增）----------
def build_bc_state14(state):
    """14D state matching BC training (frrl/envs/real.py:get_robot_state):
       [tcp_pose_true(7) + tcp_vel(6) + gripper_pos(1)]
    """
    pose = np.asarray(state["pose"], dtype=np.float32)        # 7D
    vel = np.asarray(state["vel"], dtype=np.float32)          # 6D
    grip = np.array([state["gripper_pos"]], dtype=np.float32) # 1D
    return np.concatenate([pose, vel, grip])


def build_bc_batch(state14, images_dict, device):
    """Build batch dict matching BC's input_features (state + front + wrist images)."""
    batch = {
        "observation.state": torch.from_numpy(state14).unsqueeze(0).to(device),
    }
    for cam_name in ("front", "wrist"):
        img = images_dict[cam_name]  # HWC uint8 RGB, 已 resize 到 128×128
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        batch[f"observation.images.{cam_name}"] = t.to(device)
    return batch


# ---------- Episode reset homing（success 后开新 episode 时复位机械臂）----------
def go_home_to_reset_pose(reset_pose_6d, precision_param, compliance_param,
                          lift_clearance=0.1, settle_s=0.5):
    """跟 frrl/envs/real.py:_go_to_reset 同语义的 inline homing。

    流程：开夹爪 → PRECISION_PARAM → 抬升 lift_z → 横移到 reset.xy（保 lift_z）→
    下降到 reset_pose → COMPLIANCE_PARAM。供 BC episode 边界使用，让下一集
    起点跟 BC 训练时的 reset 分布一致。

    Args:
        reset_pose_6d: 6D [x,y,z,rx,ry,rz]（euler）
        precision_param / compliance_param: 阻抗参数 dict
    """
    print("[homing] return to reset_pose...")
    # 张开夹爪释放物块
    try:
        requests.post(URL + "open_gripper", timeout=1.0)
    except Exception:
        pass
    time.sleep(0.6)

    # 切 precision 模式（高刚度，跟踪快）
    try:
        requests.post(URL + "update_param", json=precision_param, timeout=2.0)
    except Exception:
        pass
    time.sleep(0.3)

    # 读当前位姿
    try:
        state = get_state_biased()
    except Exception as e:
        print(f"[homing] getstate failed: {e}, abort homing")
        return
    curr_xyz = np.array(state["pose"][:3], dtype=np.float64)
    curr_quat_xyzw = list(state["pose"][3:])

    reset_xyz = np.array(reset_pose_6d[:3], dtype=np.float64)
    reset_quat_xyzw = list(euler_2_quat(np.array(reset_pose_6d[3:])))

    # 三段路径：lift → transit → descend
    lift_z = max(curr_xyz[2] + lift_clearance, reset_xyz[2])
    # 段1：保 xy 抬升
    lift_xyz = curr_xyz.copy()
    lift_xyz[2] = lift_z
    send_pose(lift_xyz, curr_quat_xyzw)
    time.sleep(0.6)
    # 段2：高空横移到 reset.xy + 旋转对齐
    transit_xyz = reset_xyz.copy()
    transit_xyz[2] = lift_z
    send_pose(transit_xyz, reset_quat_xyzw)
    time.sleep(1.2)
    # 段3：垂直下降到 reset.xyz
    send_pose(reset_xyz, reset_quat_xyzw)
    time.sleep(1.5)

    # 切回 compliance 模式
    try:
        requests.post(URL + "update_param", json=compliance_param, timeout=2.0)
    except Exception:
        pass
    time.sleep(settle_s)
    print("[homing] reset done")


# ---------- Gripper command（hil-serl convention thresholding）----------
class GripperCommander:
    """跟 frrl/envs/real.py:_send_gripper_command 同语义：基于 BC action[6]
    阈值 ±0.5 触发 /open_gripper or /close_gripper，加 hysteresis 避免抖动。
    """
    def __init__(self, sleep_after=0.6):
        self.sleep_after = sleep_after
        self.last_act_time = 0.0
        self.commanded_state = "open"  # 假设 reset 后是张开

    def step(self, action_gripper, current_gripper_pos):
        now = time.time()
        if (now - self.last_act_time) < self.sleep_after:
            return  # rate limit
        # hil-serl thresholding
        if action_gripper <= -0.5 and current_gripper_pos > 0.85 and self.commanded_state != "closed":
            try:
                requests.post(URL + "close_gripper", timeout=1.0)
                self.commanded_state = "closed"
                self.last_act_time = now
            except Exception as e:
                print(f"[gripper] close failed: {e}")
        elif action_gripper >= 0.5 and current_gripper_pos < 0.85 and self.commanded_state != "open":
            try:
                requests.post(URL + "open_gripper", timeout=1.0)
                self.commanded_state = "open"
                self.last_act_time = now
            except Exception as e:
                print(f"[gripper] open failed: {e}")

    def force_open(self):
        try:
            requests.post(URL + "open_gripper", timeout=1.0)
            self.commanded_state = "open"
            self.last_act_time = time.time()
        except Exception:
            pass


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True,
                    help="path to BC pickup .../pretrained_model")
    ap.add_argument("--backup-ckpt", default=None,
                    help="path to backup ckpt（默认按 --ckpt-version 选）")
    ap.add_argument("--ckpt-version", choices=["v2", "v3"], default="v3",
                    help="决定 D_SAFE/D_CLEAR + 默认 backup ckpt")
    ap.add_argument("--d-safe", type=float, default=None)
    ap.add_argument("--d-clear", type=float, default=None)
    ap.add_argument("--calibration", default="calibration_data/T_cam_to_robot.npy")
    ap.add_argument("--dry-run", action="store_true",
                    help="不发 /pose / 不发 gripper 命令，纯测 obs/inference")
    ap.add_argument("--no-workspace-clamp", action="store_true")
    ap.add_argument("--auto-reset-on-success", action="store_true", default=True,
                    help="BC 触发 success 后调 go_home_to_reset_pose 完整复位")
    ap.add_argument("--lift-threshold", type=float, default=0.06,
                    help="z 抬升触发 success 的阈值 m（reset_z + 此值）")
    ap.add_argument("--wait-operator", action="store_true",
                    help="success 复位后阻塞等待操作员按 Enter 再开下一集（默认 sleep 1.5s 自动继续）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Resolve thresholds ----------
    d_safe = args.d_safe if args.d_safe is not None else D_SAFE_BY_VERSION[args.ckpt_version]
    d_clear = args.d_clear if args.d_clear is not None else D_CLEAR_BY_VERSION[args.ckpt_version]
    backup_ckpt = args.backup_ckpt or DEFAULT_BACKUP_CKPT[args.ckpt_version]
    print(f"[OK] ckpt-version={args.ckpt_version}, D_SAFE={d_safe}m, D_CLEAR={d_clear}m")

    if not Path(backup_ckpt).exists():
        raise SystemExit(f"[ERROR] backup ckpt not found: {backup_ckpt}")
    if not Path(args.bc_ckpt).exists():
        raise SystemExit(f"[ERROR] BC ckpt not found: {args.bc_ckpt}")

    # ---------- Load both policies ----------
    print(f"[..] Loading BC policy from {args.bc_ckpt}")
    bc_cfg = PreTrainedConfig.from_pretrained(args.bc_ckpt)
    bc_cfg.pretrained_path = args.bc_ckpt
    bc_cfg.device = device
    bc_policy = SACPolicy.from_pretrained(args.bc_ckpt, config=bc_cfg)
    bc_policy.eval().to(device)
    print("[OK] BC policy loaded")

    print(f"[..] Loading backup policy from {backup_ckpt}")
    backup_cfg = PreTrainedConfig.from_pretrained(backup_ckpt)
    backup_cfg.pretrained_path = backup_ckpt
    backup_cfg.device = device
    backup_policy = SACPolicy.from_pretrained(backup_ckpt, config=backup_cfg)
    backup_policy.eval().to(device)
    print("[OK] Backup policy loaded")

    # ---------- Sensors ----------
    T_cam_to_robot = np.load(args.calibration)
    hand_detector = HandDetector(T_cam_to_robot=T_cam_to_robot)
    hand_detector.start()
    print("[OK] HandDetector + D455 started")

    # 单 RealSense for BC wrist —— front 跟 HandDetector 共享，避免 device busy。
    # image_crop["wrist"]=center_square_crop 跟 BC 训练时
    # frrl/envs/real_config.py:make_pickup_config 一致：先裁中心 480×480 保长宽比
    # 再 resize 到 128×128（不裁会被拉伸压扁）。
    cam_mgr = RealSenseCameraManager(
        camera_configs={
            "wrist": {"serial_number": WRIST_CAMERA_SERIAL, "dim": (640, 480), "exposure": 10500},
        },
        image_crop={"wrist": center_square_crop},
        image_size=IMAGE_SIZE,
    )
    print(f"[OK] RealSense wrist initialized at {IMAGE_SIZE} "
          f"(front shared with HandDetector, ROI {FRONT_CROP_VYUX})")

    # ---------- Init impedance controller ----------
    try:
        try: post("clearerr")
        except Exception: pass
        try:
            r_stop = post("stopimp")
            time.sleep(0.5)
            print(f"[..] /stopimp: {r_stop.status_code}")
        except Exception: pass
        try: post("clearerr")
        except Exception: pass
        time.sleep(0.3)
        r = requests.post(URL + "startimp", timeout=15.0)
        print(f"[OK] /startimp: {r.status_code}")
        time.sleep(1.0)
        _ = get_state_true()
        print("[OK] impedance active")
    except Exception as e:
        print(f"[FAIL] controller setup: {e}")
        return

    # ---------- Pickup task config（拿 reset_pose / 阻抗参数给 homing 用） ----------
    task_cfg = make_task_config(task="pickup", use_bias=False, reward_backend="pose")
    reset_pose_6d = task_cfg.reset_pose
    precision_param = task_cfg.precision_param
    compliance_param = task_cfg.compliance_param
    success_z_threshold = float(reset_pose_6d[2]) + args.lift_threshold
    print(f"[OK] reset_pose.xyz = {np.round(reset_pose_6d[:3], 3)}, "
          f"success_z_threshold = {success_z_threshold:.3f}m")

    # 起步先 homing 一次，让机械臂到 reset_pose 准备开始
    go_home_to_reset_pose(reset_pose_6d, precision_param, compliance_param)

    # ---------- Supervisor ----------
    supervisor = HierarchicalSupervisor(
        d_safe=d_safe,
        d_clear=d_clear,
        clear_n_steps=CLEAR_N_STEPS,
        homing_pos_tol=HOMING_POS_TOL,
        homing_rot_tol=HOMING_ROT_TOL,
        homing_action_scale=MAX_CART_SPEED / CTRL_HZ,
        homing_rot_action_scale=BACKUP_ROTATION_SCALE * LOOKAHEAD,
        homing_kp_pos=1.0,
        homing_kp_rot=1.0,
        homing_done_consecutive_n=3,
    )
    supervisor.reset()

    backup_obs_buf = deque(maxlen=OBS_STACK)
    last_hand_pos = None
    last_hand_time = None
    gripper_cmdr = GripperCommander()
    rng = np.random.default_rng(0)
    success_count = 0

    print("\n=== Deployment Active ===")
    print(f"  TASK    (green)  — BC pickup policy ({args.bc_ckpt})")
    print(f"  BACKUP  (red)    — backup policy ({backup_ckpt}) when hand_dist < {d_safe}m")
    print(f"  HOMING  (orange) — return to tcp_start after BACKUP clears")
    print(f"  Action scale: TASK=({TASK_ACTION_SCALE},{TASK_ROTATION_SCALE}), "
          f"BACKUP=({BACKUP_ACTION_SCALE},{BACKUP_ROTATION_SCALE})")
    if args.dry_run:
        print("  ⚠️ --dry-run: 不发 /pose 不发 gripper")
    print("  Ctrl+C 退出\n")

    dt = 1.0 / CTRL_HZ
    try:
        while True:
            t0 = time.time()

            # ---------- Sensors ----------
            try:
                state = get_state_true()
                state_biased = get_state_biased()
            except Exception as e:
                print(f"[!] getstate failed: {e}")
                time.sleep(0.1)
                continue

            fk_tcp = np.array(state["pose"][:3], dtype=np.float64)
            tcp_noisy = fk_tcp + rng.normal(0, TCP_NOISE_STD, 3)
            hand_body_equiv = compute_hand_body_equiv(fk_tcp, state["pose"][3:])

            # ---------- Hand detection ----------
            color_img, depth_img = hand_detector.get_frames()
            # Self-detection 双参考点几何过滤：跟 deploy_backup_policy.py 一致
            #   exclude_near_flange = hand_body_equiv（10cm 球覆盖手掌+腕部）
            #   exclude_near_tcp    = fk_tcp（6cm 球覆盖指尖）
            hand = hand_detector.detect(
                color_img, depth_img,
                exclude_near_flange=hand_body_equiv,
                flange_radius=0.10,
                exclude_near_tcp=fk_tcp,
                tcp_radius=0.06,
            )

            # HandDetection 永远返回非 None 对象，靠 .active 判 bool；位置在 .pos_robot
            if hand.active:
                hand_pos = hand.pos_robot.copy()
                now = time.time()
                if last_hand_pos is not None and last_hand_time is not None:
                    hand_vel = (hand_pos - last_hand_pos) / max(now - last_hand_time, 1e-3)
                else:
                    hand_vel = np.zeros(3)
                last_hand_pos = hand_pos
                last_hand_time = now
            else:
                hand_pos = None
                hand_vel = np.zeros(3)

            min_hand_dist = (
                np.linalg.norm(hand_pos - hand_body_equiv) if hand_pos is not None
                else float("inf")
            )

            # ---------- Read cameras for BC ----------
            # wrist 走 RealSenseCameraManager（已 RGB + crop+resize 到 128×128）；
            # front 复用 HandDetector 的 BGR 帧，手动做 BC 训练时同款 ROI crop +
            # resize + BGR→RGB。这样跟 deploy_bc_inference.py 训练分布一致。
            try:
                cam_images = cam_mgr.get_images()  # {"wrist": HWC RGB uint8 128×128}
            except Exception as e:
                print(f"[!] wrist camera read failed: {e}")
                time.sleep(0.1)
                continue
            front_rgb = cv2.cvtColor(crop_resize_front(color_img), cv2.COLOR_BGR2RGB)
            cam_images["front"] = front_rgb

            # ---------- Supervisor: FSM update + 选 mode ----------
            actual_xyz_biased = np.array(state_biased["pose"][:3], dtype=np.float64)
            actual_quat_xyzw = list(state_biased["pose"][3:])
            actual_quat_wxyz = np.array(
                [actual_quat_xyzw[3], actual_quat_xyzw[0], actual_quat_xyzw[1], actual_quat_xyzw[2]]
            )
            new_mode = supervisor.step(
                min_hand_dist=min_hand_dist,
                tcp_current_pos=actual_xyz_biased,
                tcp_current_quat=actual_quat_wxyz,
            )

            # ---------- Action callbacks ----------
            def task_action_fn():
                """BC pickup policy forward。返回 7D action [-1, 1]^7。"""
                state14 = build_bc_state14(state)
                batch = build_bc_batch(state14, cam_images, device)
                with torch.no_grad():
                    a = bc_policy.select_action(batch)
                return a.squeeze(0).cpu().numpy().astype(np.float32)  # 7D

            def backup_action_fn():
                """Backup policy forward。84D 堆叠 state，输出 6D + 0 凑 7D。"""
                use_hand = hand_pos if hand_pos is not None else (
                    last_hand_pos if last_hand_pos is not None else fk_tcp
                )
                obs28 = build_obs28(state, use_hand, hand_vel, tcp_noisy)
                backup_obs_buf.append(obs28)
                obs84 = stack_frames(backup_obs_buf, single_dim=28)
                obs_t = torch.from_numpy(obs84).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    a = backup_policy.select_action(batch={"observation.state": obs_t})
                a6 = a.squeeze(0).cpu().numpy()[:6]
                return np.array([*a6, 0.0], dtype=np.float32)  # gripper 维 0（不动）

            action7 = supervisor.select_action(
                task_action_fn, backup_action_fn,
                tcp_current_pos=actual_xyz_biased,
                tcp_current_quat=actual_quat_wxyz,
            )

            # ---------- Per-mode action scale ----------
            if new_mode == Mode.TASK:
                scale_xyz = TASK_ACTION_SCALE * LOOKAHEAD
                scale_rot = TASK_ROTATION_SCALE * LOOKAHEAD
            else:  # BACKUP / HOMING
                scale_xyz = BACKUP_ACTION_SCALE * LOOKAHEAD
                scale_rot = BACKUP_ROTATION_SCALE * LOOKAHEAD

            action_xyz = action7[:3] * scale_xyz
            action_rpy = action7[3:6] * scale_rot

            # ---------- Rate limit + clamp ----------
            delta_mag = np.linalg.norm(action_xyz)
            max_step = MAX_CART_SPEED * dt
            if delta_mag > max_step:
                action_xyz = action_xyz * (max_step / delta_mag)

            target_xyz = actual_xyz_biased + action_xyz
            if not args.no_workspace_clamp:
                target_xyz = np.clip(target_xyz, WORKSPACE_MIN, WORKSPACE_MAX)

            if np.any(np.abs(action_rpy) > 1e-6):
                cur_R = R.from_quat(actual_quat_xyzw)
                dR = R.from_rotvec(action_rpy)
                target_quat_xyzw = list((cur_R * dR).as_quat())
            else:
                target_quat_xyzw = actual_quat_xyzw

            # ---------- Send pose ----------
            if not args.dry_run:
                send_pose(target_xyz, target_quat_xyzw)

            # ---------- Gripper command (TASK 模式才接管 gripper) ----------
            if new_mode == Mode.TASK and not args.dry_run:
                gripper_cmdr.step(float(action7[6]), float(state["gripper_pos"]))

            # ---------- Auto-success detection (TASK 模式) + episode reset ----------
            if args.auto_reset_on_success and new_mode == Mode.TASK:
                z = float(state["pose"][2])
                grip = float(state["gripper_pos"])
                if z >= success_z_threshold and 0.05 < grip < 0.6:
                    success_count += 1
                    print(f"\n[SUCCESS #{success_count}] z={z:.3f}, gripper={grip:.3f}")
                    if not args.dry_run:
                        # 完整复位到 reset_pose（跟 BC 训练时 episode 边界一致）
                        go_home_to_reset_pose(reset_pose_6d, precision_param, compliance_param)
                        # 等操作员把物块放回（按 Enter 继续；没按按 1.5s 后自动继续）
                        if args.wait_operator:
                            input("    [pause] 把物块放回任意位置，按 Enter 开始下一集 (Ctrl+C 退出)... ")
                        else:
                            time.sleep(1.5)
                    supervisor.reset()
                    backup_obs_buf.clear()
                    last_hand_pos = None
                    last_hand_time = None

            # ---------- Visualization ----------
            vis = hand_detector.draw_detection(color_img, hand)
            mode_colors = {
                Mode.TASK:   (0, 255, 0),
                Mode.BACKUP: (0, 0, 255),
                Mode.HOMING: (0, 165, 255),
            }
            cv2.putText(vis, f"MODE: {new_mode.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_colors[new_mode], 2)
            cv2.putText(vis, f"hand_dist: {min_hand_dist:.3f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"successes: {success_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("deploy_pickup_with_backup", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ---------- Rate ----------
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C, 收尾")
    finally:
        try:
            hand_detector.stop()
        except Exception:
            pass
        try:
            cam_mgr.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"=== Done. Total successes: {success_count} ===")


if __name__ == "__main__":
    main()
