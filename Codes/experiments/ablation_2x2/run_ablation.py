#!/usr/bin/env python3
"""
Experiment 2: 2x2(+) Ablation -- run all conditions.

{RepSim, TRAK, Grad-Sim} x {standard, contrastive} x 3 tasks x 3 seeds
"""

import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, expand_path


def main():
    parser = argparse.ArgumentParser(description="Ablation: Run All Conditions")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    codes_dir = Path(__file__).parent.parent.parent
    config_path = args.config or str(codes_dir / "configs" / "ablation_2x2.yaml")
    config = load_config(config_path)
    run_script = codes_dir / "run_attribution.py"

    abl = config.get("ablation", {})
    methods = abl.get("methods", ["repsim", "trak", "gradsim"])
    scorings = abl.get("scorings", ["standard", "contrastive"])
    tasks = abl.get("tasks", ["toxicity", "selection", "factual"])
    base_seed = config.get("reproducibility", {}).get("seed", 42)

    total = len(methods) * len(scorings) * len(tasks) * args.seeds
    print(f"Ablation 2x2 | {len(methods)} methods x {len(scorings)} scorings x {len(tasks)} tasks x {args.seeds} seeds = {total} runs")

    passed = 0
    for method in methods:
        for scoring in scorings:
            for task in tasks:
                for seed_idx in range(args.seeds):
                    seed = base_seed + seed_idx
                    name = f"{method}_{scoring}_{task}_seed{seed}"
                    print(f"\n--- {name} ---")

                    cmd = [
                        sys.executable, str(run_script),
                        "--config", config_path,
                        "--seed", str(seed),
                        "--override",
                        f"attribution.method={method}",
                        f"attribution.scoring={scoring}",
                        f"evaluation.task={task}",
                    ]
                    if args.dry_run:
                        cmd.extend(["--dry-run", "--max-steps", str(args.max_steps)])

                    result = subprocess.run(cmd, capture_output=False)
                    if result.returncode == 0:
                        passed += 1

    print(f"\n{'='*40}")
    print(f"Ablation complete: {passed}/{total} passed")
    if args.dry_run:
        print("[DRY RUN PASSED]" if passed == total else "[DRY RUN FAILED]")


if __name__ == "__main__":
    main()
