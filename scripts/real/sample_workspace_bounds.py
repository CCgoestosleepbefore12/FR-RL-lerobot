"""交互式采样 task policy 笛卡尔工作空间边界。

用法：
  1. RT PC 上 ~/start_franka_server.sh
  2. 机械臂处于 compliance 模式（默认 impedance），用手推 / SpaceMouse 牵到目标点
  3. 终端按 Enter → 记录当前 TCP xyz
  4. 移动到下一点，重复
  5. q + Enter（或 Ctrl+C）结束

输出：
  - 每次按 Enter 立刻打印 + atomic-write 到 calibration_data/workspace_bounds.json
    （崩溃 / Ctrl+C 也不丢已采样点）
  - 退出时打印 xyz min/max + 可直接贴回 frrl/envs/real_config.py 的代码片段

不需要凑 8 个角，多采也行——边界中点 / 边上的点也能加进来给 min/max 提供冗余。
脚本只看 xyz 极值，旋转分量保留默认（real_config 里的 ±0.2 rad 限制）。

CLI:
  python scripts/real/sample_workspace_bounds.py
  python scripts/real/sample_workspace_bounds.py --url http://192.168.100.1:5000/
  python scripts/real/sample_workspace_bounds.py --output calibration_data/my_bounds.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests


def get_pose(url: str):
    r = requests.post(url + "getstate", timeout=2.0)
    r.raise_for_status()
    state = r.json()
    return np.array(state["pose"][:3], dtype=np.float64), np.array(state["pose"][3:7], dtype=np.float64)


def save_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def summarize(samples, output_path: Path):
    """退出时打印 xyz min/max + real_config.py 修改片段。"""
    if not samples:
        print("\n未采集任何点，退出。")
        return

    xyz = np.array([s["xyz"] for s in samples])
    xyz_low = xyz.min(axis=0)
    xyz_high = xyz.max(axis=0)

    print("\n" + "=" * 60)
    print(f"已采集 {len(samples)} 个点 → {output_path}")
    print("=" * 60)
    print(f"  x ∈ [{xyz_low[0]:+.4f}, {xyz_high[0]:+.4f}]   "
          f"(Δ = {xyz_high[0] - xyz_low[0]:.4f} m)")
    print(f"  y ∈ [{xyz_low[1]:+.4f}, {xyz_high[1]:+.4f}]   "
          f"(Δ = {xyz_high[1] - xyz_low[1]:.4f} m)")
    print(f"  z ∈ [{xyz_low[2]:+.4f}, {xyz_high[2]:+.4f}]   "
          f"(Δ = {xyz_high[2] - xyz_low[2]:.4f} m)")
    print("=" * 60)
    print("\n贴回 frrl/envs/real_config.py（保留 rotation 默认 ±0.2 rad）：")
    print(f"""
abs_pose_limit_low: np.ndarray = field(
    default_factory=lambda: np.array([{xyz_low[0]:.3f}, {xyz_low[1]:.3f}, {xyz_low[2]:.3f}, -np.pi, -0.2, -0.2])
)
abs_pose_limit_high: np.ndarray = field(
    default_factory=lambda: np.array([{xyz_high[0]:.3f}, {xyz_high[1]:.3f}, {xyz_high[2]:.3f}, np.pi, 0.2, 0.2])
)
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default="http://192.168.100.1:5000/",
                    help="franka_server URL（GPU→RT PC 默认；RT PC 本机用 http://127.0.0.1:5000/）")
    ap.add_argument("--output", type=Path,
                    default=Path("calibration_data/workspace_bounds.json"),
                    help="增量保存路径（每次 Enter 后立刻原子写）")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 续采：如果 JSON 已存在，问 append/overwrite
    samples = []
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
            n_existing = len(existing.get("samples", []))
        except Exception:
            n_existing = 0
        if n_existing > 0:
            choice = input(
                f"\n{args.output} 已有 {n_existing} 个点。\n"
                f"  [a] append (续采)  [o] overwrite  [c] cancel  [a/o/c, 默认 a]: "
            ).strip().lower() or "a"
            if choice == "c":
                print("Cancelled."); return
            if choice == "a":
                samples = list(existing["samples"])
                print(f"续采，从第 {len(samples)+1} 个点开始")

    print(f"\n=== Workspace bound sampler ===")
    print(f"  URL: {args.url}")
    print(f"  output: {args.output}")
    print("\n用法：")
    print("  把 TCP 移到目标点（手推 / SpaceMouse）→ 按 Enter 记录")
    print("  采够（含边界点）后输入 q + Enter 结束（或 Ctrl+C）")
    print("  脚本不限点数；多采点边界中点 / 边上点也会被纳入 min/max\n")

    try:
        while True:
            line = input(f"[#{len(samples)+1}] Enter to record (q to finish): ").strip().lower()
            if line == "q":
                break

            try:
                xyz, quat_xyzw = get_pose(args.url)
            except requests.exceptions.RequestException as e:
                print(f"  [!] /getstate 失败: {e}")
                continue

            samples.append({
                "xyz": xyz.tolist(),
                "quat_xyzw": quat_xyzw.tolist(),
                "t": time.time(),
            })
            save_atomic(args.output, {"samples": samples})
            print(f"  recorded: xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) "
                  f"[saved → {args.output.name}]")

    except (KeyboardInterrupt, EOFError):
        print("\n[Ctrl+C / EOF] 结束采样")

    summarize(samples, args.output)


if __name__ == "__main__":
    main()
