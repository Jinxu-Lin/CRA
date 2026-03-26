#!/usr/bin/env python3
"""
Probe Experiment 0: Run TRAK attribution via DATE-LM implementation.

Wraps DATE-LM's TRAK scoring and saves results in CRA format.
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
    parser = argparse.ArgumentParser(description="Probe: TRAK Attribution (DATE-LM)")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    task = config["evaluation"]["task"]
    print(f"Probe TRAK | Task: {task} | Dry run: {args.dry_run}")

    # DATE-LM TRAK integration
    date_lm_path = Path(expand_path(config["paths"]["data"])).parent / "date-lm"
    core_date_lm = Path(__file__).parent.parent.parent / "core" / "date-lm"

    # Check DATE-LM availability
    if not core_date_lm.exists():
        print(f"WARNING: DATE-LM codebase not found at {core_date_lm}")
        print("TRAK requires DATE-LM. Please clone it first:")
        print(f"  git clone <DATE-LM-repo> {core_date_lm}")

        if args.dry_run:
            # Generate placeholder scores for dry-run
            print("\n[DRY RUN] Generating placeholder TRAK scores...")
            n_train = 100
            n_test = 10
            scores = torch.randn(n_test, n_train)

            cache_dir = Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task
            cache_dir.mkdir(parents=True, exist_ok=True)
            score_file = cache_dir / f"trak_standard_seed{seed}.pt"
            torch.save(scores, score_file)
            print(f"  Placeholder saved: {score_file}")

            results_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "probe"
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / "trak_scores.json", "w") as f:
                json.dump({"placeholder": True, "shape": list(scores.shape)}, f, indent=2)

            print("[DRY RUN PASSED]")
            return
        else:
            sys.exit(1)

    # TODO: Integrate DATE-LM TRAK implementation
    # from core.date_lm import run_trak
    # scores = run_trak(config)
    print("DATE-LM TRAK integration: implementation pending DATE-LM clone")


if __name__ == "__main__":
    main()
