"""Summarize final test metrics of a jepa-repeat output root.

Usage: python scripts/test_run/summarize_jepa_repeat.py jepa-repeat/gwnet-ln
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(root):
    rows = defaultdict(list)
    for log_path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "seed_*", "training_log.json"))):
        arm, dataset, task, seed = log_path.split(os.sep)[-5:-1]
        with open(log_path) as f:
            test = json.load(f).get("epoch_logs", {}).get("test")
        if test is None:
            print(f"(unfinished) {arm}/{dataset}/{task}/{seed}")
            continue
        rows[(dataset, task, arm)].append((seed, test["metrics"]))

    for (dataset, task, arm), runs in sorted(rows.items()):
        keys = list(runs[0][1].keys())
        per_seed = ", ".join(f"{s}: MAE={m.get('MAE', float('nan')):.3f}" for s, m in runs)
        mean = {k: np.mean([m[k] for _, m in runs]) for k in keys}
        std = {k: np.std([m[k] for _, m in runs]) for k in keys}
        summary = "  ".join(f"{k}={mean[k]:.3f}±{std[k]:.3f}" for k in keys)
        print(f"{dataset:8s} {task:10s} {arm:24s} n={len(runs)}  {summary}   [{per_seed}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "jepa-repeat/gwnet-ln")
