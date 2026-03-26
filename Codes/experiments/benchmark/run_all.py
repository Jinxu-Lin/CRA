#!/usr/bin/env python3
"""
Experiment 1: Run all methods on a task for systematic benchmark.

Iterates over all methods and seeds, calling run_attribution.py for each.
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, expand_path

METHODS = ["repsim", "rept", "trak", "gradsim", "bm25", "random"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Run All Methods")
    parser.add_argument("--config", required=True, help="Base config")
    parser.add_argument("--task", type=str, default=None, help="Override task")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods")
    args = parser.parse_args()

    config = load_config(args.config)
    task = args.task or config["evaluation"]["task"]
    base_seed = config.get("reproducibility", {}).get("seed", 42)
    methods = args.methods or METHODS

    codes_dir = Path(__file__).parent.parent.parent
    run_script = codes_dir / "run_attribution.py"

    print(f"Benchmark | Task: {task} | Methods: {methods} | Seeds: {args.seeds}")

    results = {}
    for method in methods:
        for seed_idx in range(args.seeds):
            seed = base_seed + seed_idx
            exp_name = f"{method}_{task}_seed{seed}"
            print(f"\n{'='*40}")
            print(f"Running: {exp_name}")

            cmd = [
                sys.executable, str(run_script),
                "--config", args.config,
                "--seed", str(seed),
                "--override", f"attribution.method={method}",
                f"evaluation.task={task}",
            ]
            if args.dry_run:
                cmd.extend(["--dry-run", "--max-steps", str(args.max_steps)])

            result = subprocess.run(cmd, capture_output=False)
            results[exp_name] = {"returncode": result.returncode}

    # Summary
    passed = sum(1 for r in results.values() if r["returncode"] == 0)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Benchmark complete: {passed}/{total} passed")

    if args.dry_run:
        print("[DRY RUN PASSED]" if passed == total else "[DRY RUN FAILED]")


if __name__ == "__main__":
    main()
