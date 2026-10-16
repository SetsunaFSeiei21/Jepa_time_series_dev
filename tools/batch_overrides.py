"""Hydra overrides of the micro-batch and gradient accumulation of one configuration, read from
the table written by tools/probe_batch_size.py, so that batch_size x accumulation = effective,
plus any memory-saving model option the table records. Exits with status 2 when the table has
no micro-batch that fits.

    python tools/batch_overrides.py gwnet pems07 long
    -> train.batch_size=2 train.accumulation_steps=4 model.gradient_checkpointing=true
"""

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("task")
    parser.add_argument("--table", type=Path, default=Path("configs/batch_sizes.yaml"))
    parser.add_argument("--effective", type=int, default=8, help="effective batch size")
    args = parser.parse_args()
    entry = yaml.safe_load(args.table.read_text())[args.model][args.dataset][args.task]
    batch_size = entry["batch_size"]
    if not batch_size:
        print(f"{args.model}/{args.dataset}/{args.task}: no micro-batch fits", file=sys.stderr)
        sys.exit(2)
    overrides = [
        f"train.batch_size={batch_size}",
        f"train.accumulation_steps={args.effective // batch_size}",
        *entry.get("overrides", []),
    ]
    print(" ".join(overrides))


if __name__ == "__main__":
    main()
