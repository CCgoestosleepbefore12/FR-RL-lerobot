"""可视化检查 Backup Policy 环境（S1/S2 + 4 种运动模式）。"""
import argparse
import time

import numpy as np
import mujoco.viewer

from frrl.envs.panda_backup_policy_env import PandaBackupPolicyEnv


def main():
    parser = argparse.ArgumentParser(description="可视化 Backup Policy 环境")
    parser.add_argument("--num_obstacles", type=int, default=1, choices=[1, 2],
                        help="障碍物数量（1=S1, 2=S2）")
    parser.add_argument("--no_dr", action="store_true", help="关闭 Domain Randomization")
    parser.add_argument("--episodes", type=int, default=5, help="运行 episode 数")
    args = parser.parse_args()

    env = PandaBackupPolicyEnv(
        seed=42,
        render_mode="human",
        image_obs=False,
        num_obstacles=args.num_obstacles,
        enable_dr=not args.no_dr,
    )
    viewer = mujoco.viewer.launch_passive(env._model, env._data)

    scene = "S1" if args.num_obstacles == 1 else "S2"
    dr_status = "关闭" if args.no_dr else "开启"
    print(f"场景: {scene}, DR: {dr_status}")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        viewer.sync()

        tcp = env._data.site_xpos[env._pinch_site_id].copy()
        print(f"\nEP{ep+1}: tcp={np.round(tcp, 3)}, "
              f"mode={env._motion_mode.name}, "
              f"speed={env._obstacle_speed:.4f}")
        for i in range(env._num_obstacles):
            label = "移动" if i == 0 else "静止"
            print(f"  障碍物{i}({label}): pos={np.round(env._obstacle_pos[i], 3)}")

        for step in range(200):
            action = np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
            obs, r, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(0.1)

            if step % 10 == 0:
                print(f"  step {step:3d}: min_dist={info['min_hand_dist']:.3f}, "
                      f"reward={r:+.4f}")

            if terminated:
                print(f"  terminated at step {step+1}: {info.get('violation_type', '?')}")
                time.sleep(1.0)
                break

        if not terminated:
            print(f"  survived 200 steps!")

    viewer.close()
    env.close()


if __name__ == "__main__":
    main()
