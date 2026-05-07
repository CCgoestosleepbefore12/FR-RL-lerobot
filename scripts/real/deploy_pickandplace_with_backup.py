"""BC pickandplace policy + backup safety policy 联合部署 via HierarchicalSupervisor.

基于 scripts/real/deploy_backup_policy.py 的 3-state FSM，把 task_action_fn
从 SpaceMouse 替换成 BC pickandplace policy 推理。BC 用 front + wrist 双相机 +
14D state（来自 /getstate_true 的 unbiased TCP pose + vel + gripper），跟
deploy_bc_inference.py 训练分布一致。

FSM 状态：
  TASK    — BC pickandplace policy 自主控制（state + 2 cam images）
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
  python scripts/real/deploy_pickandplace_with_backup.py \\
      --bc-ckpt checkpoints/pickup_bc_20260429_171959/checkpoints/020000/pretrained_model \\
      --backup-ckpt checkpoints/backup_policy/backup_policy_s1_v3b_300k_95pct \\
      --ckpt-version v3

  # dry-run 不发 /pose，只测 obs / inference / FSM 切换
  python scripts/real/deploy_pickandplace_with_backup.py --bc-ckpt ... --dry-run
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
from frrl.fault_injection import BiasDeployController
from frrl.robots.franka_real.cameras.realsense import RealSenseCameraManager
from frrl.robots.franka_real.vision.hand_detector import HandDetector
from frrl.robots.franka_real.http_client import (
    URL, post, get_state_true, get_state_biased, send_pose, align_quat_sign,
)
from frrl.robots.franka_real.geometry import compute_hand_body_equiv
from frrl.robots.franka_real.camera_drain import (
    drain_sleep, check_hand_close_during_homing, interpolate_move_with_drain,
)
from frrl.robots.franka_real.gripper_commander import GripperCommander
from frrl.robots.franka_real.abort_signal import install as install_sigint, aborted

# ---------- Backup FSM thresholds（沿用 deploy_backup_policy.py）----------
D_SAFE_BY_VERSION = {"v2": 0.30, "v3": 0.40}
D_CLEAR_BY_VERSION = {"v2": 0.35, "v3": 0.45}
DEFAULT_BACKUP_CKPT = {
    "v2": "checkpoints/backup_policy/backup_policy_s1_v2_newgeom_145k",
    "v3": "checkpoints/backup_policy/backup_policy_s1_v3b_300k_95pct",
}
CLEAR_N_STEPS = 3
HOMING_POS_TOL = 0.02
HOMING_ROT_TOL = 0.05

# ---------- Action scales（per mode）----------
# 跟各自训练时的 env 完全对齐，避免训练-部署 mismatch：
#   TASK:   BC 在 FrankaRealEnv 里训练，target = current + bc_action × 0.05，
#           rate limit max_cart_speed=0.50（无 LOOKAHEAD）。
#   BACKUP: sim 训练时有隐式 lookahead × 2，max_cart_speed=0.30，所以 deploy
#           保留 LOOKAHEAD=2.0 + scale 0.025/0.05（跟 deploy_backup_policy.py 一致）。
TASK_ACTION_SCALE = 0.05            # BC training scale（frrl pickup action_scale[0]）
TASK_ROTATION_SCALE = 0.10
TASK_LOOKAHEAD = 1.0                # BC 训练时无 lookahead
TASK_MAX_CART_SPEED = 0.50          # frrl pickup max_cart_speed

BACKUP_ACTION_SCALE = 0.025
BACKUP_ROTATION_SCALE = 0.05
BACKUP_LOOKAHEAD = 2.0
BACKUP_MAX_CART_SPEED = 0.30

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
FRONT_CROP_VYUX = (122, 350, 177, 405)  # pickandplace ROI: select_workspace_roi.py 框选 (177,122,405,350)


def crop_resize_front(rgb_img):
    """对 HandDetector.get_frames() 的 RGB 帧做 BC 训练时同款 crop + resize。"""
    v0, v1, u0, u1 = FRONT_CROP_VYUX
    cropped = rgb_img[v0:v1, u0:u1]
    return cv2.resize(cropped, IMAGE_SIZE)


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
                          hand_detector, cam_mgr, d_safe: float,
                          lift_clearance=0.08, settle_s=0.5):
    """跟 frrl/envs/real.py:_go_to_reset 同语义的 inline homing。

    流程：开夹爪 → PRECISION_PARAM → 抬升 lift_z → 横移到 reset.xy（保 lift_z）→
    下降到 reset_pose → COMPLIANCE_PARAM。供 BC episode 边界使用，让下一集
    起点跟 BC 训练时的 reset 分布一致。

    所有 sleep 期间通过 drain_sleep 持续读相机；插值移动期间每个 tick 也调
    check_hand_close_during_homing 实时安全检查 —— hand 距离 < d_safe 立即
    return False，让 caller 中断 homing 把控制权交回主 FSM（自动切 BACKUP）。

    返回：True = 正常完成；False = 中途被 hand 检测中断（caller 应跳过 supervisor.reset，
    让下一帧 supervisor.step 接管）。

    Args:
        reset_pose_6d: 6D [x,y,z,rx,ry,rz]（euler）
        precision_param / compliance_param: 阻抗参数 dict
        hand_detector / cam_mgr: 给 drain + 安全检查用
        d_safe: hand 触发 BACKUP 的距离阈值（跟主 FSM 一致）
    """
    print("[homing] return to reset_pose...")
    # 张开夹爪释放物块（open_gripper 失败不致命，只 log，主要操作往后还能跑）
    try:
        requests.post(URL + "open_gripper", timeout=1.0)
    except Exception as e:
        print(f"[homing] open_gripper warning: {e}")
    drain_sleep(0.6, hand_detector, cam_mgr)

    # 先 anchor setpoint 到当前 pose，避免切 precision 时 controller 朝旧 setpoint 突跳
    # （复刻 frrl/envs/real.py:_go_to_reset 第 646-647 行的安全模式）
    try:
        state_anchor = get_state_biased()
        anchor_xyz = np.array(state_anchor["pose"][:3], dtype=np.float64)
        anchor_quat_xyzw = list(state_anchor["pose"][3:])
        send_pose(anchor_xyz, anchor_quat_xyzw)
    except Exception as e:
        print(f"[homing] setpoint anchor failed: {e}, abort homing")
        return False

    # 切 precision 模式（高刚度，跟踪快）—— 失败要响亮，stiffness mismatch 后续运行不安全
    try:
        r = requests.post(URL + "update_param", json=precision_param, timeout=2.0)
        r.raise_for_status()
    except Exception as e:
        print(f"[homing] update_param(precision) failed: {e}, abort homing")
        return False
    drain_sleep(0.3, hand_detector, cam_mgr)

    # 读当前位姿
    try:
        state = get_state_biased()
    except Exception as e:
        print(f"[homing] getstate failed: {e}, abort homing")
        return False
    curr_xyz = np.array(state["pose"][:3], dtype=np.float64)
    curr_quat_xyzw = list(state["pose"][3:])

    reset_xyz = np.array(reset_pose_6d[:3], dtype=np.float64)
    reset_quat_xyzw = list(euler_2_quat(np.array(reset_pose_6d[3:])))

    # 三段路径：lift → transit → descend，每段平滑插值 + 实时 hand 安全检查
    lift_z = max(curr_xyz[2] + lift_clearance, reset_xyz[2])

    # 段1：保 xy 抬升（1.0s，每步 ~1cm/0.1s）
    lift_xyz = curr_xyz.copy()
    lift_xyz[2] = lift_z
    if not interpolate_move_with_drain(curr_xyz, curr_quat_xyzw, lift_xyz, curr_quat_xyzw,
                                        timeout=1.0, hand_detector=hand_detector,
                                        cam_mgr=cam_mgr, d_safe=d_safe):
        return False  # hand 中断

    # 段2：高空横移到 reset.xy + 旋转对齐（2.0s）
    transit_xyz = reset_xyz.copy()
    transit_xyz[2] = lift_z
    if not interpolate_move_with_drain(lift_xyz, curr_quat_xyzw, transit_xyz, reset_quat_xyzw,
                                        timeout=2.0, hand_detector=hand_detector,
                                        cam_mgr=cam_mgr, d_safe=d_safe):
        return False

    # 段3：垂直下降到 reset.xyz（1.5s，~5cm 下降）
    if not interpolate_move_with_drain(transit_xyz, reset_quat_xyzw, reset_xyz, reset_quat_xyzw,
                                        timeout=1.5, hand_detector=hand_detector,
                                        cam_mgr=cam_mgr, d_safe=d_safe):
        return False

    # 切回 compliance 模式（失败要响亮，stiffness 不对会让后续 BC 在错误动力学上跑）
    try:
        r = requests.post(URL + "update_param", json=compliance_param, timeout=2.0)
        r.raise_for_status()
    except Exception as e:
        print(f"[homing] update_param(compliance) failed: {e}; ⚠️ controller 可能仍在 precision")
        return False
    drain_sleep(settle_s, hand_detector, cam_mgr)
    print("[homing] reset done")
    return True


# ---------- Main ----------
def main():
    # SIGINT handler: 把 Ctrl+C 转 flag（matplotlib Tk 吞 KeyboardInterrupt 解决方案，
    # 详见 frrl/robots/franka_real/abort_signal.py）
    install_sigint()

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
    ap.add_argument("--no-reset-on-recovery", action="store_true",
                    help="BACKUP/HOMING → TASK 转换时不强制 go_home_to_reset_pose，让 supervisor "
                         "的 HOMING 自然把 TCP 拉回 tcp_start，BC 接着 episode 半路 resume。"
                         "pickandplace 是多阶段任务（pick → transit → place），mid-episode resume "
                         "理论上比 pickup 更复杂——BACKUP 期间餐具可能位置变化 / 不在夹爪里，"
                         "BC 在 tcp_start resume 后大概率走错。默认仍走 force home。")
    ap.add_argument("--max-episode-steps", type=int, default=400,
                    help="TASK 模式单 episode 最大步数（默认 400 = 40s @ 10Hz；pickandplace 较长）。"
                         "兜底用——主路径仍是操作员 SPACE/Enter；超时计为 fail 强制 go_home，"
                         "避免 BC 卡死时主循环死锁。")
    ap.add_argument("--bias", action="store_true",
                    help="启用 J1 encoder bias 注入（默认 OFF）。每 episode 边界采新 bias，"
                         "recovery 期间沿用同一 bias（episode 内恒定）。详见 BiasDeployController。")
    ap.add_argument("--bias-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="覆盖 bias 采样范围（rad）。默认 None = 用 task factory 内置值。"
                         "仅当 --bias 时生效。")
    ap.add_argument("--bias-monitor", action="store_true",
                    help="启用 BiasMonitor：保存 q_true/q_biased 时间序列 npz + 实时双线图。"
                         "默认输出 charts/bias_deploy_pickandplace_with_backup_*。仅当 --bias 时生效。")
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
            "wrist": {"serial_number": WRIST_CAMERA_SERIAL, "dim": (640, 480), "exposure": 40000},
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

    # ---------- Pickandplace task config（拿 reset_pose / 阻抗参数 / 工作空间边界给 TASK 用） ----------
    bias_monitor_save_path = None
    if args.bias_monitor and args.bias:
        from datetime import datetime
        Path("charts").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bias_monitor_save_path = f"charts/bias_deploy_pickandplace_with_backup_{ts}"
        print(f"[OK] BiasMonitor enabled → {bias_monitor_save_path}.{{npz,png}}")
    task_cfg = make_task_config(
        task="pickandplace", use_bias=args.bias, reward_backend="keyboard",
        enable_bias_monitor=(args.bias_monitor and args.bias),
        bias_monitor_save_path=bias_monitor_save_path,
        bias_range=tuple(args.bias_range) if args.bias_range is not None else None,
    )
    if args.bias and args.bias_range is not None:
        print(f"[OK] bias_range override: {tuple(args.bias_range)} rad")
    reset_pose_6d = task_cfg.reset_pose
    precision_param = task_cfg.precision_param
    compliance_param = task_cfg.compliance_param
    # TASK 模式用 BC 训练时同款 workspace（abs_pose_limit_low/high），避免 impedance
    # controller 在桌面 / 工作面以下卡死。BACKUP / HOMING 仍用宽 WORKSPACE_MIN/MAX。
    task_workspace_min = task_cfg.abs_pose_limit_low[:3].astype(np.float64)
    task_workspace_max = task_cfg.abs_pose_limit_high[:3].astype(np.float64)
    print(f"[OK] reset_pose.xyz = {np.round(reset_pose_6d[:3], 3)}")
    print(f"[OK] TASK workspace: min={task_workspace_min.tolist()}, max={task_workspace_max.tolist()}")

    # Bias controller（封装 EncoderBiasInjector + BiasMonitor + HTTP 调用）
    bias_ctrl = None
    if args.bias and task_cfg.encoder_bias_config is not None:
        bias_ctrl = BiasDeployController(
            task_cfg.encoder_bias_config,
            url=URL,
            monitor_save_path=bias_monitor_save_path,
            enable_monitor=args.bias_monitor,
        )
        print(f"[OK] BiasDeployController active, range={task_cfg.encoder_bias_config.bias_range}")
    elif args.bias_monitor and not args.bias:
        print("[WARN] --bias-monitor 需配 --bias 才有效；当前 --bias 未开，monitor 跳过")

    # 起步先 homing 一次（bias=ON 时先 clear+anchor）
    if bias_ctrl is not None:
        bias_ctrl.begin_transition()
    if not go_home_to_reset_pose(reset_pose_6d, precision_param, compliance_param,
                                  hand_detector, cam_mgr, d_safe=d_safe):
        print("[FATAL] 启动 homing 失败（hand 在工作区？stiffness 切换失败？）→ 不进入主循环")
        try:
            hand_detector.stop()
        except Exception:
            pass
        try:
            cam_mgr.close()
        except Exception:
            pass
        return

    # ---------- Supervisor ----------
    supervisor = HierarchicalSupervisor(
        d_safe=d_safe,
        d_clear=d_clear,
        clear_n_steps=CLEAR_N_STEPS,
        homing_pos_tol=HOMING_POS_TOL,
        homing_rot_tol=HOMING_ROT_TOL,
        # ⚠️ 必须等于 apply 处的 scale_xyz/scale_rot（HomingController 内部 clip(kp·δ/scale)
        # 与外部缩放必须一致），否则等效 kp 偏离 1.0 (deadbeat)。
        homing_action_scale=BACKUP_ACTION_SCALE * BACKUP_LOOKAHEAD,
        homing_rot_action_scale=BACKUP_ROTATION_SCALE * BACKUP_LOOKAHEAD,
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
    fail_count = 0
    task_steps = 0  # TASK 模式累计步数；超过 max_episode_steps 计 timeout 失败
    iter_count = 0
    prev_mode = Mode.TASK  # 跟踪 mode 转换，BACKUP/HOMING→TASK 时 force home

    # KeyboardRewardListener: pynput 独立线程（"双线程"），与 deploy_bc_inference.py
    # 同款键盘协议：S=开始 episode, ENTER=success, SPACE=fail, BACKSPACE=discard。
    # 替代之前的 cv2 ENTER + wait_for_operator + stdin Enter 的混乱（cv2 buffer 残留
    # 会误触发 SUCCESS）。
    from frrl.rewards.keyboard_reward import KeyboardRewardListener
    keyboard_listener = KeyboardRewardListener(required=True)
    print("\n=== Keyboard Protocol (双线程, pynput) ===")
    print("  S         开始 episode（每集开始前都要按）")
    print("  ENTER     标记 success → 当前 episode 结束")
    print("  SPACE     标记 fail → 当前 episode 结束")
    print("  BACKSPACE discard → 当前 episode 结束")
    print()

    print("=== Deployment Active ===")
    print(f"  TASK    (green)  — BC pickandplace policy ({args.bc_ckpt})")
    print(f"  BACKUP  (red)    — backup policy ({backup_ckpt}) when hand_dist < {d_safe}m")
    print(f"  HOMING  (orange) — return to tcp_start after BACKUP clears")
    print(f"  Action scale: TASK=({TASK_ACTION_SCALE},{TASK_ROTATION_SCALE}), "
          f"BACKUP=({BACKUP_ACTION_SCALE},{BACKUP_ROTATION_SCALE})")
    if args.dry_run:
        print("  ⚠️ --dry-run: 不发 /pose 不发 gripper")
    print("  Ctrl+C 退出 / Q 关闭 OpenCV 窗口\n")

    # 第一 episode 之前先等操作员按 S（与 deploy_bc_inference.py 同款流程）
    print("=== 等待操作员按 S 开始第一 episode ===")
    keyboard_listener.wait_for_start()
    if bias_ctrl is not None:
        bias_ctrl.finish_transition(ep_num=0, resample=True)
    print("=== ep 1 开始 ===\n")

    dt = 1.0 / CTRL_HZ
    try:
        while not aborted():
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

            if bias_ctrl is not None:
                bias_ctrl.on_step(state, state_biased)

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
                """BC pickandplace policy forward。返回 7D action [-1, 1]^7。"""
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

            # BACKUP/HOMING 退出回 TASK 时：强制完整 go_home_to_reset_pose，
            # 否则 BC 在 mid-task state 上 resume（物块可能凌空 / 抓在手里）→ OOD。
            # prev_mode 跟踪上次 step 后的 mode；HOMING/BACKUP→TASK 转换触发复位。
            #
            # --no-reset-on-recovery 实验路径：跳过强制复位，让 supervisor HOMING 自然
            # 把 TCP 拉回 tcp_start，BC 在 episode 半路 resume。BACKUP 期间场景可能已变
            # (手碰过物块 / 夹爪状态变化)，BC 看到的 tcp_start 状态可能 OOD。
            if (prev_mode in (Mode.BACKUP, Mode.HOMING)) and (new_mode == Mode.TASK):
                if args.no_reset_on_recovery:
                    print(f"\n[recover] {prev_mode.name}→TASK：no-reset 模式，supervisor HOMING 自然返回 tcp_start，BC 半路 resume")
                    backup_obs_buf.clear()
                    last_hand_pos = None
                    last_hand_time = None
                    prev_mode = new_mode
                    # 不 reset supervisor、不 force_open、不 go_home，直接进下一帧让 BC 接管
                else:
                    print(f"\n[recover] {prev_mode.name}→TASK：force home + 释放餐具 + 等 S 继续")
                    homing_ok = True
                    if not args.dry_run:
                        # bias=ON：clear+anchor → home → resample=False（同集沿用旧 bias）
                        if bias_ctrl is not None:
                            bias_ctrl.begin_transition()
                        homing_ok = go_home_to_reset_pose(
                            reset_pose_6d, precision_param, compliance_param,
                            hand_detector, cam_mgr, d_safe=d_safe,
                        )
                        if not homing_ok:
                            # hand 中断：不 reset supervisor，让下一帧 step 自然切 BACKUP
                            print("[recover] homing 被 hand 检测中断 → 让主 FSM 接管")
                    if homing_ok:
                        gripper_cmdr.force_open()
                        supervisor.reset()
                        backup_obs_buf.clear()
                        last_hand_pos = None
                        last_hand_time = None
                        task_steps = 0
                        # 清掉 BACKUP 期间操作员可能误触的 ENTER/SPACE（pynput thread 一直在跑），
                        # 否则下一帧 TASK poll 会把那个 outcome 当 success/fail 误终结新 episode。
                        # 然后跟 success 路径同样的协议：等 S 才进下一集。
                        keyboard_listener.mark_episode_ended()
                        print("=== [recover] 完成。把餐具放回盘子 + 收手离开 + 按 S 继续 ===")
                        keyboard_listener.wait_for_start()
                        if bias_ctrl is not None:
                            bias_ctrl.finish_transition(
                                ep_num=success_count + fail_count,
                                resample=False,  # 同集延续，不采新 bias
                            )
                    prev_mode = new_mode
                continue  # 重读 state 再进下一帧

            action7 = supervisor.select_action(
                task_action_fn, backup_action_fn,
                tcp_current_pos=actual_xyz_biased,
                tcp_current_quat=actual_quat_wxyz,
            )
            prev_mode = new_mode

            # ---------- Debug print: 每 10 iter (1s) 一行 ----------
            iter_count += 1
            if iter_count % 10 == 0:
                z = float(state["pose"][2])
                grip = float(state["gripper_pos"])
                a3 = action7[:3].round(3).tolist()
                print(f"[t={iter_count*0.1:5.1f}s] mode={new_mode.name} "
                      f"z={z:.3f} grip={grip:.3f} "
                      f"hand_d={min_hand_dist if min_hand_dist != float('inf') else -1:.3f} "
                      f"action.xyz={a3} action[6]={action7[6]:+.2f}")

            # ---------- Per-mode action scale + rate limit ----------
            # TASK：跟 BC 标准部署 (FrankaRealEnv) 完全一致，无 lookahead，0.50 m/s 上限
            # BACKUP/HOMING：跟 sim 训练 + deploy_backup_policy.py 一致，2× lookahead, 0.30 m/s
            if new_mode == Mode.TASK:
                scale_xyz = TASK_ACTION_SCALE * TASK_LOOKAHEAD
                scale_rot = TASK_ROTATION_SCALE * TASK_LOOKAHEAD
                max_step = TASK_MAX_CART_SPEED * dt
            else:  # BACKUP / HOMING
                scale_xyz = BACKUP_ACTION_SCALE * BACKUP_LOOKAHEAD
                scale_rot = BACKUP_ROTATION_SCALE * BACKUP_LOOKAHEAD
                max_step = BACKUP_MAX_CART_SPEED * dt

            action_xyz = action7[:3] * scale_xyz
            action_rpy = action7[3:6] * scale_rot

            # ---------- Rate limit ----------
            delta_mag = np.linalg.norm(action_xyz)
            if delta_mag > max_step:
                action_xyz = action_xyz * (max_step / delta_mag)

            target_xyz = actual_xyz_biased + action_xyz
            if not args.no_workspace_clamp:
                # 按模式选 workspace：TASK 跟 BC 训练 envelope 一致；BACKUP 用宽边界
                if new_mode == Mode.TASK:
                    target_xyz = np.clip(target_xyz, task_workspace_min, task_workspace_max)
                else:  # BACKUP / HOMING
                    target_xyz = np.clip(target_xyz, WORKSPACE_MIN, WORKSPACE_MAX)

            if np.any(np.abs(action_rpy) > 1e-6):
                cur_R = R.from_quat(actual_quat_xyzw)
                dR = R.from_rotvec(action_rpy)
                # 旋转 frame **per-mode**（详见 deploy_pickup_with_backup.py 同款注释）：
                #   TASK   (BC, real.py:278): world frame `dR * cur_R`
                #   BACKUP (sim env): body frame `cur_R * dR`
                #   HOMING (HomingController): body frame `cur_R * dR`
                if new_mode == Mode.TASK:
                    target_quat_xyzw = list((dR * cur_R).as_quat())  # world
                else:  # BACKUP / HOMING
                    target_quat_xyzw = list((cur_R * dR).as_quat())  # body
            else:
                target_quat_xyzw = actual_quat_xyzw
            # 跟上一帧 quat 同半球，避免 sign-flip 让 impedance 走 360° 长路
            target_quat_xyzw = align_quat_sign(target_quat_xyzw, actual_quat_xyzw)

            # ---------- Send pose ----------
            if not args.dry_run:
                send_pose(target_xyz, target_quat_xyzw)

            # ---------- Gripper command (TASK 模式才接管 gripper) ----------
            if new_mode == Mode.TASK and not args.dry_run:
                gripper_cmdr.step(float(action7[6]), float(state["gripper_pos"]))

            # ---------- Success detection (KeyboardRewardListener) + episode reset ----------
            # 用 pynput 独立线程接 S/Enter/Space/Backspace（与 deploy_bc_inference.py 同款），
            # 比 cv2.waitKey 稳：不会有 buffer 误触发，不依赖 OpenCV 窗口聚焦。
            # Timeout 兜底：BC 卡死时 task_steps 超 max_episode_steps 自动计 fail。
            episode_done = False
            outcome = None
            if new_mode == Mode.TASK:
                task_steps += 1
                outcome_dict = keyboard_listener.poll()
                if outcome_dict is not None:
                    outcome = outcome_dict.get("outcome", "unknown")
                    if outcome == "success":
                        success_count += 1
                        print(f"\n[SUCCESS #{success_count}] keyboard ENTER")
                    else:
                        fail_count += 1
                        print(f"\n[{outcome.upper()}] (fail/discard #{fail_count})")
                    episode_done = True
                elif task_steps >= args.max_episode_steps:
                    outcome = "timeout"
                    fail_count += 1
                    print(f"\n[TIMEOUT #{fail_count}] task_steps={task_steps} "
                          f">= {args.max_episode_steps} (fail)")
                    episode_done = True
            else:
                task_steps = 0

            if episode_done:
                homing_ok = True
                if not args.dry_run:
                    # 立刻 force_open 释放餐具（让操作员能拿起放回盘子）
                    gripper_cmdr.force_open()
                    # 新集边界：clear+anchor → home → resample=True
                    if bias_ctrl is not None:
                        bias_ctrl.begin_transition()
                    homing_ok = go_home_to_reset_pose(
                        reset_pose_6d, precision_param, compliance_param,
                        hand_detector, cam_mgr, d_safe=d_safe,
                    )
                    if not homing_ok:
                        print("[recover] homing 被 hand 检测中断 → 主 FSM 接管下一帧")
                if homing_ok:
                    supervisor.reset()
                    backup_obs_buf.clear()
                    last_hand_pos = None
                    last_hand_time = None
                    task_steps = 0
                    keyboard_listener.mark_episode_ended()
                    # 等下次操作员按 S 才进下一集
                    print(f"=== ep done. 把餐具放回盘子 + 收手离开工作区 + 按 S 开始下一集 ===")
                    keyboard_listener.wait_for_start()
                    if bias_ctrl is not None:
                        bias_ctrl.finish_transition(
                            ep_num=success_count + fail_count,
                            resample=True,
                        )
                    print(f"=== 下一 episode 开始 ===\n")

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
            cv2.putText(vis, f"S {success_count}  F {fail_count}  step:{task_steps}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, "S=start  ENTER=success  SPACE=fail  Q=quit",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            cv2.imshow("deploy_pickandplace_with_backup", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ---------- Rate ----------
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        # Backup path：install_sigint 后正常 Ctrl+C 应走 aborted() flag 不抛异常。
        print("\n[!] KeyboardInterrupt fallback, 收尾")
    finally:
        # ---------- 安全收尾 ----------
        # 顺序：强制张爪释放物块 → 把机械臂带回 reset_pose 安全停泊 → 清 bias →
        # 关相机/HandDetector → 关 cv2 窗口。每步独立 try 防一项失败阻塞后续。
        print("[shutdown] 强制 force_open 释放物块...")
        try:
            requests.post(URL + "open_gripper", timeout=1.0)
            time.sleep(0.6)
        except Exception as e:
            print(f"[shutdown] force_open 失败: {e}")

        print("[shutdown] 把机械臂送回 reset_pose...")
        try:
            # finally 路径不依赖 supervisor / hand check，直接用 frrl env.go_home 走
            # 那套 lift→transit→descend 路径。如果 task_cfg 等局部变量未定义（main
            # 在初始化中段失败），就跳过 home 步骤但仍要清 bias + 关资源。
            go_home_to_reset_pose(
                reset_pose_6d, precision_param, compliance_param,
                hand_detector, cam_mgr, d_safe=d_safe,
            )
        except (NameError, UnboundLocalError):
            print("[shutdown] task_cfg 未初始化，跳过 home")
        except Exception as e:
            print(f"[shutdown] go_home 失败: {e}")

        # BiasDeployController 自己负责 clear_encoder_bias + 保 npz/png + 关图。
        # 没开 --bias 也走一次 HTTP clear（兜底，防上次跑完没清干净）。
        try:
            if bias_ctrl is not None:
                bias_ctrl.close()
            else:
                requests.post(URL + "clear_encoder_bias", timeout=2.0)
        except (NameError, UnboundLocalError):
            pass
        except Exception as e:
            print(f"[shutdown] bias 收尾失败: {e}")

        try:
            hand_detector.stop()
        except Exception:
            pass
        try:
            cam_mgr.close()
        except Exception:
            pass
        try:
            keyboard_listener.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"=== Done. Total successes: {success_count}, fails: {fail_count} ===")


if __name__ == "__main__":
    main()
