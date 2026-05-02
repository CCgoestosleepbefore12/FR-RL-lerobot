"""Backup 145k + Homing 集成测试：验证 backup 退让后能否用 HomingController 回到起始位姿。

流程（env 内同一 episode）：
  1. env.reset → 记录 (tcp_start_pos, tcp_start_quat)
  2. N_BACKUP 步由 145k backup policy 驱动 → TCP 被推离起始（记录最大偏差）
  3. 冻结障碍物（HIDDEN_POS + 无限 stall），重置 step_count
  4. N_HOMING 步由 HomingController 驱动 → 测量每步 pos_err / rot_err 收敛

用法:
  python scripts/sim/run_homing_integration.py                 # 默认 20 个 episode
  python scripts/sim/run_homing_integration.py --render        # 可视化
  python scripts/sim/run_homing_integration.py --n_episodes 5 --render
"""
import argparse
import time

import numpy as np
import torch
import gymnasium as gym
import mujoco
import mujoco.viewer

import frrl.envs  # noqa: F401
import frrl.policies.sac.configuration_sac  # noqa: F401
from frrl.configs.policies import PreTrainedConfig
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.processor import (
    Numpy2TorchActionProcessorStep,
    VanillaObservationProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    DataProcessorPipeline,
    create_transition,
)
from frrl.processor.converters import identity_transition
from frrl.processor.core import TransitionKey
from frrl.rl.supervisor import HomingController
from frrl.envs.sim.panda_backup_policy_env import HIDDEN_POS


def run(ckpt: str, n_episodes: int, n_backup: int, n_homing: int,
        render: bool, env_task: str, pos_tol: float, rot_tol: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    policy = SACPolicy.from_pretrained(ckpt, config=cfg)
    policy.eval().to(device)
    print(f"[LOAD] policy {ckpt} on {device}", flush=True)

    env = gym.make(f"gym_frrl/{env_task}", image_obs=False)
    unwrapped = env.unwrapped
    print(f"[ENV]  {env_task}", flush=True)

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(unwrapped._model, unwrapped._data)

    proc = DataProcessorPipeline(
        steps=[
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    homing = HomingController(pos_tol=pos_tol, rot_tol=rot_tol)

    results = []
    print(f"\n{'EP':>3s} {'BackupDisp':>11s} {'BackupRot':>10s} "
          f"{'HomingSteps':>12s} {'FinalPosErr':>12s} {'FinalRotErr':>12s} {'Status':>8s}",
          flush=True)
    print("-" * 80, flush=True)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        proc.reset()
        transition = proc(create_transition(obs))

        tcp_start_pos = unwrapped._tcp_start.copy()
        tcp_start_quat = unwrapped._tcp_start_quat.copy()

        # ---- 阶段 1: backup policy ----
        for _ in range(n_backup):
            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.input_features
            }
            with torch.no_grad():
                action = policy.select_action(batch=observation)
            action_np = action.squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(action_np)
            if render and viewer is not None:
                viewer.sync(); time.sleep(0.05)
            if term or trunc:
                break
            transition = proc(create_transition(obs))

        tcp_cur = unwrapped._data.site_xpos[unwrapped._pinch_site_id].copy()
        backup_disp = float(np.linalg.norm(tcp_cur - tcp_start_pos))
        q_cur = np.asarray(unwrapped._data.mocap_quat[0], dtype=np.float64).copy()
        backup_rot = float(np.linalg.norm(
            homing._rot_error_axis_angle(q_cur) if False else
            _aa_err(q_cur, tcp_start_quat)
        ))

        # ---- 阶段 2: homing ----
        # 冻结障碍物：搬到隐藏位置并拉满 stall
        for i in range(unwrapped._num_obstacles):
            unwrapped._obstacle_pos[i] = HIDDEN_POS.copy()
            unwrapped._stall_remaining[i] = 10**9
        unwrapped._sync_all_visuals()
        unwrapped._step_count = 0  # 避免 truncated 截断 homing

        homing.reset(tcp_start_pos, tcp_start_quat)
        homing_steps = n_homing
        converged = False
        for s in range(n_homing):
            tcp_cur = unwrapped._data.site_xpos[unwrapped._pinch_site_id].copy()
            q_cur = np.asarray(unwrapped._data.mocap_quat[0], dtype=np.float64).copy()
            if homing.is_done(tcp_cur, q_cur):
                homing_steps = s
                converged = True
                break
            action_h = homing.get_action(tcp_cur, q_cur)[:6]
            env.step(action_h)
            if render and viewer is not None:
                viewer.sync(); time.sleep(0.05)

        tcp_final = unwrapped._data.site_xpos[unwrapped._pinch_site_id].copy()
        q_final = np.asarray(unwrapped._data.mocap_quat[0], dtype=np.float64).copy()
        final_pos_err = float(np.linalg.norm(tcp_final - tcp_start_pos))
        final_rot_err = float(np.linalg.norm(_aa_err(q_final, tcp_start_quat)))

        status = "OK" if converged else "TIMEOUT"
        results.append({
            "backup_disp": backup_disp,
            "backup_rot": backup_rot,
            "homing_steps": homing_steps,
            "pos_err": final_pos_err,
            "rot_err": final_rot_err,
            "converged": converged,
        })
        print(f"{ep+1:3d} {backup_disp:11.4f} {backup_rot:10.4f} "
              f"{homing_steps:12d} {final_pos_err:12.5f} {final_rot_err:12.5f} {status:>8s}",
              flush=True)

    if render and viewer is not None:
        viewer.close()
    env.close()

    # ---- 汇总 ----
    n_ok = sum(1 for r in results if r["converged"])
    print("-" * 80, flush=True)
    print(f"收敛率          : {n_ok}/{n_episodes} ({n_ok/n_episodes*100:.1f}%)", flush=True)
    print(f"平均 backup 位移: {np.mean([r['backup_disp'] for r in results]):.4f} m", flush=True)
    print(f"平均 backup 旋转: {np.mean([r['backup_rot'] for r in results]):.4f} rad", flush=True)
    print(f"平均 homing 步数: {np.mean([r['homing_steps'] for r in results]):.1f}", flush=True)
    print(f"平均末态 pos 误差: {np.mean([r['pos_err'] for r in results]):.5f} m "
          f"(tol={pos_tol})", flush=True)
    print(f"平均末态 rot 误差: {np.mean([r['rot_err'] for r in results]):.5f} rad "
          f"(tol={rot_tol})", flush=True)


def _aa_err(q_cur: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """复用 HomingController 的局部系误差公式（避免要 target 实例）。"""
    from frrl.rl.supervisor import quat_conjugate, quat_multiply, quat_to_axis_angle
    q_cur = q_cur / (np.linalg.norm(q_cur) + 1e-12)
    q_target = q_target / (np.linalg.norm(q_target) + 1e-12)
    q_err = quat_multiply(quat_conjugate(q_cur), q_target)
    return quat_to_axis_angle(q_err)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/backup_policy/backup_policy_s1_v2_newgeom_145k")
    p.add_argument("--env_task", default="PandaBackupPolicyS1V2-v0")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--n_backup", type=int, default=10, help="backup policy 步数（≤20）")
    p.add_argument("--n_homing", type=int, default=30, help="homing 最多步数")
    p.add_argument("--pos_tol", type=float, default=0.02)
    p.add_argument("--rot_tol", type=float, default=0.05)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    run(args.checkpoint, args.n_episodes, args.n_backup, args.n_homing,
        args.render, args.env_task, args.pos_tol, args.rot_tol)
