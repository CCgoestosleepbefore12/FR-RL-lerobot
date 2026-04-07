"""检查初始位置下前视和腕部相机的视野，保存图片到 outputs/camera_check/"""
import numpy as np
import mujoco
from pathlib import Path
from frrl.envs.panda_pick_place_safe_env import PandaPickPlaceSafeEnv

env = PandaPickPlaceSafeEnv(
    image_obs=True,
    render_mode="rgb_array",
    hand_appear_prob=1.0,
    hand_appear_step_range=(0, 1),  # 人手立刻出现
)
obs, _ = env.reset()

out_dir = Path("outputs/camera_check")
out_dir.mkdir(parents=True, exist_ok=True)

front = obs["pixels"]["front"]
wrist = obs["pixels"]["wrist"]

# 保存为图片
from PIL import Image
Image.fromarray(front).save(out_dir / "front_view.png")
Image.fromarray(wrist).save(out_dir / "wrist_view.png")

print(f"图片已保存到 {out_dir}/")
print(f"  front_view.png: {front.shape}")
print(f"  wrist_view.png: {wrist.shape}")

# 打印关键位置
tcp_pos = env._data.site_xpos[env._pinch_site_id]
block_pos = env._data.sensor("block_pos").data
plate_pos = env._data.sensor("plate_pos").data
print(f"\nTCP位置: {np.round(tcp_pos, 3)}")
print(f"方块位置: {np.round(block_pos, 3)}")
print(f"Plate位置: {np.round(plate_pos, 3)}")
print(f"人手位置: {np.round(env._hand_pos, 3)}")

env.close()
