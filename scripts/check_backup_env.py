"""可视化检查 Backup Policy 环境（S1/S2 + TRACKING-only, 20step/ep, 6D 动作）。

Usage:
    python scripts/check_backup_env.py --episodes 10
    python scripts/check_backup_env.py --episodes 10 --num_obstacles 2
    python scripts/check_backup_env.py --episodes 10 --action_mode zero --no_dr
"""
import argparse
import time

import numpy as np
import mujoco.viewer

from frrl.envs.panda_backup_policy_env import PandaBackupPolicyEnv


def main():
    parser = argparse.ArgumentParser(description="可视化 Backup Policy 环境 (TRACKING)")
    parser.add_argument("--num_obstacles", type=int, default=1, choices=[1, 2],
                        help="障碍物数量（1=S1, 2=S2）")
    parser.add_argument("--no_dr", action="store_true", help="关闭 Domain Randomization")
    parser.add_argument("--episodes", type=int, default=10, help="运行 episode 数")
    parser.add_argument("--action_mode", choices=["zero", "random", "retreat"],
                        default="retreat",
                        help="zero=站桩/random=随机探索/retreat=沿手→TCP 反向退让")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step_dt", type=float, default=0.1, help="可视化每步 sleep 秒")
    args = parser.parse_args()

    env = PandaBackupPolicyEnv(
        seed=args.seed,
        render_mode="human",
        image_obs=False,
        num_obstacles=args.num_obstacles,
        enable_dr=not args.no_dr,
    )
    viewer = mujoco.viewer.launch_passive(env._model, env._data)

    scene = "S1" if args.num_obstacles == 1 else "S2"
    dr_status = "关闭" if args.no_dr else "开启"
    print(f"场景: {scene}, DR: {dr_status}, action_mode: {args.action_mode}")

    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        viewer.sync()

        tcp = env._data.site_xpos[env._pinch_site_id].copy()
        print(f"\nEP{ep+1}: tcp={np.round(tcp, 3)}, "
              f"mode={env._motion_mode.name}, "
              f"speed={env._obstacle_speed:.4f}")
        for i in range(env._num_obstacles):
            print(f"  障碍物{i}: pos={np.round(env._obstacle_pos[i], 3)}")

        step = 0
        while True:
            tcp_cur = env._data.site_xpos[env._pinch_site_id].copy()
            if args.action_mode == "zero":
                action = np.zeros(6, dtype=np.float32)
            elif args.action_mode == "random":
                action = rng.uniform(-0.3, 0.3, size=6).astype(np.float32)
            else:  # retreat：沿 (TCP - 最近手) 方向平移，姿态 0
                min_d = np.inf
                dir_vec = np.zeros(3)
                for i in range(env._num_obstacles):
                    v = tcp_cur - env._obstacle_pos[i]
                    d = float(np.linalg.norm(v))
                    if d < min_d:
                        min_d = d
                        dir_vec = v / (d + 1e-8)
                action = np.zeros(6, dtype=np.float32)
                action[:3] = dir_vec * 0.5

            obs, r, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(args.step_dt)

            print(f"  step {step:2d}: min_dist={info['min_hand_dist']:.3f}, "
                  f"reward={r:+.4f}, term={terminated}, trunc={truncated}")

            step += 1
            if terminated or truncated:
                tag = "terminated" if terminated else "truncated"
                reason = info.get("violation_type", "")
                print(f"  [{tag}] at step {step}{f' ({reason})' if reason else ''}")
                time.sleep(0.8)
                break

    viewer.close()
    env.close()


if __name__ == "__main__":
    main()
