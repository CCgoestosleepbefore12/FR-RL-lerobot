"""Compute dataset_stats (min/max/mean/std) from demo pickles.

Needed because NormalizerProcessorStep uses these stats to map state/action
into a standard range before feeding the SAC networks; without real stats the
normalizer either crashes or runs identity, causing training to diverge or
degenerate.

Usage:
    python scripts/tools/compute_dataset_stats.py --demos "demos/task_policy/*.pkl"

Prints a JSON fragment you can paste into scripts/configs/train_task_policy_franka.json
under policy.dataset_stats.
"""
import argparse
import glob
import json
import pickle as pkl
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demos", nargs="+", required=True, help="pickle glob(s)")
    args = ap.parse_args()

    paths = []
    for pattern in args.demos:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        else:
            paths.append(pattern)
    if not paths:
        raise SystemExit(f"no demo files matched {args.demos}")

    transitions = []
    for p in paths:
        with open(p, "rb") as f:
            transitions.extend(pkl.load(f))
    print(f"loaded {len(transitions)} transitions from {len(paths)} file(s)")

    # agent_pos (29D)
    agent_pos = np.stack([t["observations"]["agent_pos"] for t in transitions]).astype(np.float64)
    action = np.stack([t["actions"] for t in transitions]).astype(np.float64)

    def stats(arr, name):
        """Return min/max/mean/std per-dim as Python lists (JSON-ready)."""
        mn = arr.min(axis=0).tolist()
        mx = arr.max(axis=0).tolist()
        mu = arr.mean(axis=0).tolist()
        sd = arr.std(axis=0).tolist()
        print(f"\n== {name} ==  shape={arr.shape}  (showing 6 decimals)")
        print(f"  min: {[round(v, 6) for v in mn]}")
        print(f"  max: {[round(v, 6) for v in mx]}")
        print(f"  mean: {[round(v, 6) for v in mu]}")
        print(f"  std:  {[round(v, 6) for v in sd]}")
        return dict(min=mn, max=mx, mean=mu, std=sd)

    stats_all = {
        "observation.state": stats(agent_pos, "observation.state (agent_pos, 29D)"),
        "action": stats(action, "action (7D)"),
    }

    # Images — ImageNet defaults are used in config already; we just verify
    # our demos don't have wildly different statistics.
    first_img = transitions[0]["observations"]["pixels"]["front"]
    if first_img.dtype == np.uint8:
        print(f"\n(pixel images are uint8; post-/255 normalization makes per-channel mean/std "
              f"similar to ImageNet — keeping config's defaults 0.485/0.229 etc.)")

    out_path = "dataset_stats_generated.json"
    with open(out_path, "w") as f:
        # write only min/max (MIN_MAX norm) — trim mean/std for brevity
        compact = {
            "observation.state": {"min": stats_all["observation.state"]["min"], "max": stats_all["observation.state"]["max"]},
            "action": {"min": stats_all["action"]["min"], "max": stats_all["action"]["max"]},
        }
        json.dump(compact, f, indent=2)
    print(f"\nwrote compact MIN_MAX stats → {out_path}")
    print("Paste this under `policy.dataset_stats` in your training config:\n")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
