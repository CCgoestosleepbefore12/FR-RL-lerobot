#!/usr/bin/env python
"""
切片 LeRobotDataset 的 observation.state 列：
  obs31 (Safe demo, 带 hand)     → obs28 / obs25
  obs27 (Task demo, 不带 hand)   → obs24 / obs21

obs31 布局（来自 panda_pick_place_safe_env.py::_compute_observation obs31 分支）:
  [0:18]   robot_state        (18)
  [18:21]  block_pos          (3)
  [21:24]  plate_pos          (3)
  [24:27]  noisy_real_tcp     (3)
  [27:28]  hand_active        (1)
  [28:31]  hand_pos           (3)

obs27 布局（obs_mode="obs27"）:
  [0:18]   robot_state        (18)
  [18:21]  block_pos          (3)
  [21:24]  plate_pos          (3)
  [24:27]  noisy_real_tcp     (3)

切片规则（按 src-dim → mode 索引保留表）:
  src=31, obs28 = 去 real_tcp         → keep [0..23, 27..30]   (28D)
  src=31, obs25 = 去 block/plate      → keep [0..17, 24..30]   (25D)
  src=27, obs24 = 去 real_tcp         → keep [0..23]           (24D)
  src=27, obs21 = 去 block/plate      → keep [0..17, 24..26]   (21D)

用法:
  python scripts/tools/slice_safe_demo.py --src-dim 31 --mode obs28
  python scripts/tools/slice_safe_demo.py --src-dim 27 --mode obs24 --force

依赖: pyarrow, numpy（lerobot 环境自带）。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROOT = Path.home() / ".cache/huggingface/lerobot"

# src_dim → 默认源 repo
DEFAULT_SRC_REPO = {
    31: "frrl/pick_place_safe_bias_demo",
    27: "frrl/pick_place_task_bias_demo",
}

# (src_dim, mode) → 保留的列索引
KEEP_IDX: dict[int, dict[str, list[int]]] = {
    31: {
        "obs28": list(range(24)) + list(range(27, 31)),   # 去 real_tcp，保留 hand
        "obs25": list(range(18)) + list(range(24, 31)),   # 去 block/plate，保留 hand
    },
    27: {
        "obs24": list(range(24)),                          # 去 real_tcp
        "obs21": list(range(18)) + list(range(24, 27)),    # 去 block/plate
    },
}

OBS_COL = "observation.state"

# meta/episodes/*.parquet 中 per-episode 统计里需按 dim 切片的列后缀
PER_DIM_EPISODE_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def slice_data_parquet(src_path: Path, dst_path: Path, keep_idx: list[int], src_dim: int) -> int:
    """切片 data/chunk-*/file-*.parquet 的 observation.state 列。

    observation.state: fixed_size_list<float>[src_dim] → fixed_size_list<float>[D]
    返回行数。
    """
    table = pq.read_table(src_path)
    state_col = table.column(OBS_COL)
    state_np = (
        state_col.combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, src_dim)
    )
    sliced = state_np[:, keep_idx].astype(np.float32)

    flat = pa.array(sliced.flatten(), type=pa.float32())
    new_col = pa.FixedSizeListArray.from_arrays(flat, len(keep_idx))

    col_idx = table.column_names.index(OBS_COL)
    new_table = table.set_column(col_idx, OBS_COL, new_col)
    # 剥离旧的 HF schema metadata（里面嵌了 src_dim 描述，避免与新列冲突）
    new_table = new_table.replace_schema_metadata(None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return new_table.num_rows


def slice_episodes_parquet(src_path: Path, dst_path: Path, keep_idx: list[int]) -> None:
    """切片 meta/episodes/chunk-*/file-*.parquet 里 per-episode 的 obs.state 统计列。

    列形如 `stats/observation.state/min`，类型 list<double>（每行长度 src_dim）。
    count 列是 list<int64> 长度 1，不用切。
    """
    table = pq.read_table(src_path)
    col_names = table.column_names
    target_cols = [f"stats/{OBS_COL}/{k}" for k in PER_DIM_EPISODE_KEYS if f"stats/{OBS_COL}/{k}" in col_names]

    for col_name in target_cols:
        col = table.column(col_name)
        new_rows = [
            (None if row is None else [row[i] for i in keep_idx]) for row in col.to_pylist()
        ]
        new_col = pa.array(new_rows, type=pa.list_(pa.float64()))
        col_idx = col_names.index(col_name)
        table = table.set_column(col_idx, col_name, new_col)

    table = table.replace_schema_metadata(None)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)


def slice_stats_json(stats: dict, keep_idx: list[int], src_dim: int) -> dict:
    """切片 meta/stats.json 里 observation.state 的 per-dim 列表。"""
    entry = stats.get(OBS_COL)
    if entry is None:
        return stats
    for k, v in list(entry.items()):
        if k == "count":  # 标量 [N]，不切
            continue
        if isinstance(v, list) and len(v) == src_dim:
            entry[k] = [v[i] for i in keep_idx]
    return stats


def slice_info_json(info: dict, obs_dim: int) -> dict:
    """更新 meta/info.json 里 observation.state.shape。"""
    info["features"][OBS_COL]["shape"] = [obs_dim]
    return info


def main():
    ap = argparse.ArgumentParser(description="Slice LeRobotDataset obs31/obs27 → 子集")
    ap.add_argument("--src-dim", type=int, required=True, choices=list(KEEP_IDX.keys()),
                    help="源 observation.state 维度（31=Safe demo，27=Task demo）")
    ap.add_argument("--mode", required=True,
                    help="目标维度（31→obs28/obs25，27→obs24/obs21）")
    ap.add_argument("--src-repo", default=None,
                    help="源 repo_id（默认按 --src-dim 选）")
    ap.add_argument("--dst-repo", default=None, help="目标 repo_id（默认 <src>_<mode>）")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="LeRobot cache 根（默认 ~/.cache/huggingface/lerobot）")
    ap.add_argument("--force", action="store_true", help="目标存在时覆盖")
    args = ap.parse_args()

    src_dim: int = args.src_dim
    valid_modes = KEEP_IDX[src_dim]
    if args.mode not in valid_modes:
        raise SystemExit(f"--src-dim {src_dim} 仅支持 mode ∈ {list(valid_modes)}，收到: {args.mode}")

    src_repo = args.src_repo or DEFAULT_SRC_REPO[src_dim]
    root = Path(args.root)
    src = root / src_repo
    dst_repo = args.dst_repo or f"{src_repo}_{args.mode}"
    dst = root / dst_repo
    keep_idx = valid_modes[args.mode]
    obs_dim = len(keep_idx)

    if not src.exists():
        raise SystemExit(f"源数据集不存在: {src}")

    # 校验源 obs 是 src_dim
    with open(src / "meta/info.json") as f:
        info = json.load(f)
    src_shape = info["features"][OBS_COL]["shape"]
    if src_shape != [src_dim]:
        raise SystemExit(f"源 {OBS_COL} shape={src_shape}，预期 [{src_dim}]（--src-dim 与实际不一致）")

    if dst.exists():
        if not args.force:
            raise SystemExit(f"目标已存在: {dst}\n加 --force 以覆盖。")
        shutil.rmtree(dst)

    print(f"[1/5] 复制非数据子树: {src} → {dst}")
    # 忽略 data / meta/episodes（稍后切片重写），videos/images 直接复制
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("data", "episodes"),
    )

    print(f"[2/5] 切片 data/*.parquet  (keep_idx 长度={obs_dim}, 前5={keep_idx[:5]} ...)")
    total_rows = 0
    for pq_file in sorted((src / "data").rglob("*.parquet")):
        rel = pq_file.relative_to(src)
        n = slice_data_parquet(pq_file, dst / rel, keep_idx, src_dim)
        total_rows += n
        print(f"   {rel}  ({n} 行 → {obs_dim}D)")

    print(f"[3/5] 切片 meta/episodes/*.parquet  (per-episode 统计)")
    episodes_src = src / "meta/episodes"
    if episodes_src.exists():
        for pq_file in sorted(episodes_src.rglob("*.parquet")):
            rel = pq_file.relative_to(src)
            slice_episodes_parquet(pq_file, dst / rel, keep_idx)
            print(f"   {rel}  (已切片 {len(PER_DIM_EPISODE_KEYS)} 个统计列)")

    print(f"[4/5] 重写 meta/info.json  (shape [{src_dim}] → [{obs_dim}])")
    info = slice_info_json(info, obs_dim)
    with open(dst / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"[5/5] 切片 meta/stats.json  (per-dim 列表 [{src_dim}] → [{obs_dim}])")
    with open(src / "meta/stats.json") as f:
        stats = json.load(f)
    stats = slice_stats_json(stats, keep_idx, src_dim)
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("\n完成.")
    print(f"  源:   {src}  ({src_dim}D)")
    print(f"  目标: {dst}  ({obs_dim}D, {total_rows} 帧)")
    print(f"  训练 repo_id: {dst_repo}")


if __name__ == "__main__":
    main()
