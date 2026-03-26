#!/usr/bin/env python3
"""
Experiment 0.5: Mini Pilot 2x2 -- run a single condition.

Supports both standard and contrastive scoring for any method.
Config determines method and scoring mode.
"""

import argparse
import sys
import json
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, get_config_from_args, add_common_args, expand_path
from seed_utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Pilot: Run 2x2 Condition")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    method = config["attribution"]["method"]
    scoring = config["attribution"].get("scoring", "standard")
    task = config["evaluation"]["task"]

    print(f"Pilot 2x2 | Method: {method} | Scoring: {scoring} | Task: {task}")

    # Delegate to main attribution pipeline
    from run_attribution import main as run_main
    sys.argv = [
        "run_attribution.py",
        "--config", args.config,
        *(["--dry-run", "--max-steps", str(args.max_steps)] if args.dry_run else []),
        *(["--seed", str(args.seed)] if args.seed else []),
    ]
    run_main()


if __name__ == "__main__":
    main()
