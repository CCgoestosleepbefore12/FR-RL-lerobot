"""Inspect demo pickle file(s) — verify shape, count, loadability into ReplayBuffer.

Usage:
    python scripts/tools/inspect_demo_pickle.py demos/task_policy/*.pkl
    python scripts/tools/inspect_demo_pickle.py demos/task_policy/task_policy_demos_1_complete_2026-04-23_12-34-56.pkl
"""
import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np


def inspect_raw_pickle(path: str):
    """Peek at raw pickle contents before loading into ReplayBuffer."""
    with open(path, "rb") as f:
        transitions = pkl.load(f)
    if not isinstance(transitions, list):
        raise ValueError(f"{path} is not a list (got {type(transitions)})")
    print(f"  raw pickle: {len(transitions)} transitions")
    if not transitions:
        return
    t0 = transitions[0]
    print(f"  first transition keys: {list(t0.keys())}")
    for k in ("observations", "next_observations"):
        if k not in t0:
            continue
        obs = t0[k]
        if isinstance(obs, dict):
            print(f"    {k}:")
            for ok, ov in obs.items():
                if isinstance(ov, dict):
                    for ok2, ov2 in ov.items():
                        print(f"      {ok}.{ok2}: shape={np.asarray(ov2).shape}, dtype={np.asarray(ov2).dtype}")
                else:
                    print(f"      {ok}: shape={np.asarray(ov).shape}, dtype={np.asarray(ov).dtype}")
        else:
            print(f"    {k}: shape={np.asarray(obs).shape}")
    if "actions" in t0:
        a = np.asarray(t0["actions"])
        print(f"    actions: shape={a.shape}, dtype={a.dtype}, value={a.tolist()}")
    if "rewards" in t0:
        print(f"    rewards (first): {t0['rewards']}")
    if "dones" in t0:
        print(f"    dones (first): {t0['dones']}")
    # Aggregate reward/done stats
    rewards = [t.get("rewards", 0.0) for t in transitions]
    dones = [bool(t.get("dones", False)) for t in transitions]
    print(f"  reward sum: {sum(rewards):.3f}, done count: {sum(dones)}")


def load_into_buffer(paths):
    from frrl.rl.buffer import ReplayBuffer
    buf = ReplayBuffer.from_pickle_transitions(
        pickle_paths=paths,
        device="cpu",
        storage_device="cpu",
        use_drq=False,
    )
    print(f"\n=== Loaded into ReplayBuffer ===")
    print(f"  total transitions: {len(buf)}")
    if len(buf) == 0:
        return
    n = min(4, len(buf))
    batch = buf.sample(batch_size=n)
    print(f"  sampled batch (n={n}):")
    print(f"    state keys: {list(batch['state'].keys())}")
    for k, v in batch["state"].items():
        print(f"      {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    print(f"    action: shape={tuple(batch['action'].shape)}, dtype={batch['action'].dtype}")
    print(f"    reward: shape={tuple(batch['reward'].shape)}")
    print(f"    done:   shape={tuple(batch['done'].shape)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="one or more .pkl paths")
    args = ap.parse_args()

    # Resolve/validate inputs
    resolved = []
    for p in args.paths:
        q = Path(p)
        if not q.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)
        resolved.append(str(q))

    print(f"Inspecting {len(resolved)} file(s):\n")
    for p in resolved:
        print(f"=== {p} ===")
        try:
            inspect_raw_pickle(p)
        except Exception as e:
            print(f"  ERROR reading {p}: {e}")
            sys.exit(1)

    load_into_buffer(resolved)
    print("\nOK")


if __name__ == "__main__":
    main()
