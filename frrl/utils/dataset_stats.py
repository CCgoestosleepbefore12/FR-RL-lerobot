"""Compute & inject dataset_stats (min/max) for training pipelines.

为什么在 frrl/utils 而不是 scripts/tools：scripts/ 不是 Python 包（没有
``__init__.py``），所以 deploy/training entry scripts 直接 ``python
scripts/tools/X.py`` 时无法 import 同目录下的 helper（sys.path 只有
``scripts/tools/``）。把 lib 函数搬到 frrl 后，learner / actor / bc_pretrain /
pretrain 全部走 ``from frrl.utils.dataset_stats import ...``，统一可达。

CLI 路径（``python scripts/tools/compute_dataset_stats.py --demos ...``）继续
用，里面 import 这个模块的函数。
"""
import glob
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
    """Per-dim MIN_MAX stats with const-channel guard。

    锁定夹爪等任务下 action[6] 永远 -1 → min=max → MIN_MAX (x-min)/(max-min)
    = 0/0 = NaN。检测 const 维度并把 max 抬到 min+1，归一化恒为 0（无信息但不爆 NaN）。
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
        (stats_dict, num_transitions). stats_dict 形如
        ``{"observation.state": {"min": [...], "max": [...]}, "action": {...}}``。
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


def auto_inject_dataset_stats(cfg, *, force: bool = False) -> bool:
    """Compute dataset_stats from cfg.policy.demo_pickle_paths and inject into
    cfg.policy.dataset_stats. Idempotent guard for the SAC training stack.

    Args:
        cfg: TrainRLServerPipelineConfig（带 .policy.demo_pickle_paths 与
            .policy.dataset_stats）。
        force: True 时无视所有 skip 条件强制重算。默认 False（学习器场景下
            cfg.dataset 非空 / cfg.resume / 无 demo_paths 都跳过）。

    Returns:
        True = 成功 inject；False = skip（resume / 无 demo / 走 HF 数据集）。

    Skip 条件（非 force）：
      * 走 HuggingFace lerobot 数据集（cfg.dataset 非 None） → ds_meta.stats 走它的路径
      * Resume 训练（cfg.resume=True） → ckpt 内 train_config.json 已有 stats
      * demo_pickle_paths 空 → 没数据可算

    必须在 make_policy 之前调用，否则 normalizer 已经 snapshot 旧 stats，改 cfg 不生效。
    """
    if not force:
        if getattr(cfg, "dataset", None) is not None:
            return False
        if getattr(cfg, "resume", False):
            logging.info("[auto_stats] skip: cfg.resume=True，ckpt 内 train_config.json 已有 stats")
            return False
        if not cfg.policy.demo_pickle_paths:
            return False

    auto_stats, n_trans = compute_stats_from_paths(
        cfg.policy.demo_pickle_paths, verbose=True
    )
    if cfg.policy.dataset_stats is None:
        cfg.policy.dataset_stats = {}
    for key in ("observation.state", "action"):
        cfg.policy.dataset_stats[key] = auto_stats[key]
    logging.info(
        f"[auto_stats] {n_trans} transitions → "
        f"overwrote dataset_stats[{', '.join(repr(k) for k in auto_stats)}]"
    )
    return True
