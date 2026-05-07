"""CLI: compute dataset_stats (min/max) from demo pickles 并打印 + 写 JSON。

用于手动场景（CI / 需要单独保存 JSON / paste 到 config）。训练 entry script
（bc_pretrain / pretrain / learner / actor）已默认在启动时自动算并注入 cfg，
不再需要手动跑这一步。

实际逻辑在 frrl.utils.dataset_stats 模块（让 frrl/scripts 都可 import；
scripts/ 不是 package，子模块互相 import 不可达）。
"""
import argparse
import json
import logging

from frrl.utils.dataset_stats import compute_stats_from_paths


def main():
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
