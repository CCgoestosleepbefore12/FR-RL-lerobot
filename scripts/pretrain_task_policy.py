"""Offline RLPD pretraining on demo pickles, no actor.

This launches the learner in `offline_only_mode=True`:

  1. Load demos from `--demo-paths` into offline_replay_buffer (hil-serl pkl schema)
  2. Run `--steps` gradient updates (critic + actor + temperature) on that buffer
  3. Save the final checkpoint to `--output-dir` and exit
  4. (Later) Start the actor + learner for online HIL training with
     `--resume-from` pointing at the saved checkpoint

This matches the spirit of rail-berkeley/hil-serl's `train_rlpd.py` when run
with demos only; the difference is that we reuse the full
`frrl/rl/learner.py` (PyTorch) so the network + optimizer state are byte-for-byte
what the online phase will pick up.

Usage:
    python scripts/pretrain_task_policy.py \\
        --config configs/train_task_policy_franka.json \\
        --demo-paths "demos/task_policy/*.pkl" \\
        --steps 5000 \\
        --output-dir checkpoints/task_policy_pretrain
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

import draccus

from frrl.configs.train import TrainRLServerPipelineConfig
from frrl.rl.learner import train


def _expand_globs(patterns: list[str]) -> list[str]:
    """Expand shell-style globs; pass through literal paths as-is."""
    out: list[str] = []
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            out.extend(sorted(matches))
        else:
            # Treat as a literal path even if it doesn't exist; let the loader error loudly.
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--config", required=True,
        help="Path to TrainRLServerPipelineConfig JSON (same schema as for online training)"
    )
    ap.add_argument(
        "--demo-paths", nargs="+", required=True,
        help="One or more pickle paths or globs (e.g. 'demos/task_policy/*.pkl')"
    )
    ap.add_argument(
        "--steps", type=int, default=5000,
        help="Number of gradient updates (default 5000)"
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Checkpoint output dir (overrides cfg.output_dir)"
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    demos = _expand_globs(args.demo_paths)
    if not demos:
        print(f"ERROR: no demo files matched {args.demo_paths}", file=sys.stderr)
        sys.exit(1)
    logging.info(f"Loading {len(demos)} demo file(s):")
    for d in demos:
        logging.info(f"  · {d}")

    # Load config via draccus (same path frrl/configs/parser.py:232 uses).
    cfg = draccus.parse(
        config_class=TrainRLServerPipelineConfig,
        config_path=args.config,
        args=[],
    )

    cfg.policy.offline_only_mode = True
    cfg.policy.offline_pretrain_steps = args.steps
    cfg.policy.demo_pickle_paths = demos
    # learner uses pathlib's / on cfg.output_dir — keep as Path.
    # Don't pre-create: cfg.validate() refuses to start if the dir already exists
    # (it's train()'s job to mkdir inside the run).
    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)
    elif cfg.output_dir is not None and not isinstance(cfg.output_dir, Path):
        cfg.output_dir = Path(cfg.output_dir)

    logging.info(
        f"pretrain: steps={args.steps}, output_dir={cfg.output_dir}, "
        f"demo_files={len(demos)}"
    )

    # Delegate to the main learner entrypoint. It will detect offline_only_mode
    # and skip the gRPC actor server + main online loop, then save & exit.
    train(cfg)


if __name__ == "__main__":
    main()
