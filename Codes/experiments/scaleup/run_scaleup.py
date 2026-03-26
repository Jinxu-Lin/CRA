#!/usr/bin/env python3
"""
Experiment 5: Scale-up to Llama-7B.
"""

import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, expand_path


def main():
    parser = argparse.ArgumentParser(description="Scale-up: Llama-7B")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="llama-7b")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    codes_dir = Path(__file__).parent.parent.parent
    config_path = args.config or str(codes_dir / "configs" / "scaleup.yaml")
    config = load_config(config_path)
    run_script = codes_dir / "run_attribution.py"

    su = config.get("scaleup", {})
    methods = su.get("methods", ["repsim", "trak"])
    tasks = su.get("tasks", ["toxicity", "selection"])
    base_seed = config.get("reproducibility", {}).get("seed", 42)

    total = len(methods) * len(tasks) * args.seeds
    print(f"Scale-up | Model: {args.model} | {total} runs")

    passed = 0
    for method in methods:
        for task in tasks:
            for seed_idx in range(args.seeds):
                seed = base_seed + seed_idx
                name = f"scaleup_{method}_{task}_seed{seed}"
                print(f"\n--- {name} ---")

                cmd = [
                    sys.executable, str(run_script),
                    "--config", config_path,
                    "--seed", str(seed),
                    "--override",
                    f"attribution.method={method}",
                    f"evaluation.task={task}",
                ]
                if args.dry_run:
                    cmd.extend(["--dry-run", "--max-steps", str(args.max_steps)])

                result = subprocess.run(cmd, capture_output=False)
                if result.returncode == 0:
                    passed += 1

    print(f"\nComplete: {passed}/{total} passed")
    if args.dry_run:
        print("[DRY RUN PASSED]" if passed == total else "[DRY RUN FAILED]")


if __name__ == "__main__":
    main()
