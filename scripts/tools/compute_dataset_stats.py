"""Compute dataset_stats (min/max) from demo pickles.

Used by NormalizerProcessorStep to map state/action into a standard range
before feeding the SAC networks.

两种用法：

* CLI 独立运行（打印 + 写 dataset_stats_generated.json，给手动场景）：
    python scripts/tools/compute_dataset_stats.py --demos "data/wipe/*.pkl"

* 作为函数被训练脚本调用（自动注入 cfg.policy.dataset_stats）：
    from scripts.tools.compute_dataset_stats import compute_stats_from_paths
    stats = compute_stats_from_paths(["data/x/*.pkl", "data/y/*.pkl"])

bc_pretrain_task_policy / pretrain_task_policy 默认走第二条路径，免去
"compute → 复制 → 粘贴到 config" 的人工步骤。
"""
import argparse
import glob
import json
import logging
import pickle as pkl
from typing import Dict, List, Tuple

import numpy as np


def _expand_globs(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        else:
            paths.append(pattern)
    return paths


def _stats_min_max(arr: np.ndarray, name: str = "") -> Dict[str, list]:
    """Return min/max per-dim as Python lists (JSON-ready, MIN_MAX norm format).

    Const-channel guard: 锁定夹爪等任务下 action[6] 永远 -1 → min=max → MIN_MAX
    归一化 (x-min)/(max-min) = 0/0 = NaN。检测 const 维度并人为把 max 抬到
    min+1，使该维度归一化恒等于 0（无信息但不爆 NaN）。
    """
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    const_mask = (mx - mn) < 1e-6
    if const_mask.any():
        const_idx = np.where(const_mask)[0].tolist()
        if name:
            logging.warning(
                f"[dataset_stats] {name}: const channels at dim(s) {const_idx} — "
                f"setting max=min+1 to avoid normalize NaN"
            )
        mx = mx.copy()
        mx[const_mask] = mn[const_mask] + 1.0
    return {"min": mn.tolist(), "max": mx.tolist()}


def compute_stats_from_paths(
    pickle_patterns: List[str],
    *,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, list]], int]:
    """从 demo pickle 计算 observation.state / action 的 MIN_MAX stats。

    Args:
        pickle_patterns: glob 模式列表（也支持直接给路径）。
        verbose: 是否打印每个数组的 shape / mean / std（人工 sanity check 用）。

    Returns:
        (stats_dict, num_transitions)
        stats_dict 形如 ``{"observation.state": {"min": [...], "max": [...]}, "action": {...}}``，
        可直接 json.dump 或注入 cfg.policy.dataset_stats。
    """
    paths = _expand_globs(pickle_patterns)
    if not paths:
        raise ValueError(f"no demo files matched {pickle_patterns}")

    transitions = []
    for p in paths:
        with open(p, "rb") as f:
            transitions.extend(pkl.load(f))
    if verbose:
        logging.info(
            f"[dataset_stats] loaded {len(transitions)} transitions "
            f"from {len(paths)} file(s)"
        )

    agent_pos = np.stack(
        [t["observations"]["agent_pos"] for t in transitions]
    ).astype(np.float64)
    action = np.stack([t["actions"] for t in transitions]).astype(np.float64)

    if verbose:
        for name, arr in (("observation.state (agent_pos)", agent_pos),
                          ("action", action)):
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            logging.info(
                f"[dataset_stats] {name} shape={arr.shape} "
                f"mean={[round(float(v), 4) for v in mu]} "
                f"std={[round(float(v), 4) for v in sd]}"
            )

    stats = {
        "observation.state": _stats_min_max(agent_pos, "observation.state"),
        "action": _stats_min_max(action, "action"),
    }
    return stats, len(transitions)


def main():
    """CLI 路径：保留 print + 写 dataset_stats_generated.json，给手动 paste 流程。"""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demos", nargs="+", required=True, help="pickle glob(s)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    stats, n = compute_stats_from_paths(args.demos, verbose=True)

    out_path = "dataset_stats_generated.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote compact MIN_MAX stats → {out_path}")
    print(f"({n} transitions)")
    print("Paste this under `policy.dataset_stats` in your training config:\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
