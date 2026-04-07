"""可视化检查 Backup Policy 环境。"""
import numpy as np
import time
import mujoco
import mujoco.viewer
from frrl.envs.panda_backup_policy_env import PandaBackupPolicyEnv

env = PandaBackupPolicyEnv(seed=42, render_mode='human', image_obs=False)
viewer = mujoco.viewer.launch_passive(env._model, env._data)

for ep in range(5):
    obs, _ = env.reset()
    viewer.sync()

    tcp = env._data.site_xpos[env._pinch_site_id].copy()
    block = env._data.sensor('block_pos').data.copy()
    print(f"\nEP{ep+1}: tcp={np.round(tcp,3)}, block_dist={np.linalg.norm(tcp-block):.4f}")
    print(f"  hand_start={np.round(env._hand_pos, 3)}")

    for step in range(200):
        # 随机小幅动作（模拟策略输出），能看到机械臂在动
        action = np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
        obs, r, t, tr, info = env.step(action)
        viewer.sync()
        time.sleep(0.1)  # 慢一点方便观察

        if step % 10 == 0:
            hand_d = info['min_hand_dist']
            print(f"  step {step:3d}: hand_dist={hand_d:.3f}, hand_pos={np.round(env._hand_pos, 3)}")

        if t:
            print(f"  terminated at step {step+1}: {info.get('violation_type', '?')}")
            time.sleep(1.0)  # 碰撞后暂停1秒看清楚
            break

    if not t:
        print(f"  survived 200 steps!")

viewer.close()
env.close()
