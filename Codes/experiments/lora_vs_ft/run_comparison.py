#!/usr/bin/env python3
"""
Experiment 3: LoRA vs Full-FT comparison.

{LoRA, Full-FT} x {RepSim, TRAK} x {toxicity, selection} x 3 seeds.
"""

import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, expand_path


def main():
    parser = argparse.ArgumentParser(description="LoRA vs FT: Run Comparison")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    codes_dir = Path(__file__).parent.parent.parent
    config_path = args.config or str(codes_dir / "configs" / "lora_vs_ft.yaml")
    config = load_config(config_path)
    run_script = codes_dir / "run_attribution.py"

    comp = config.get("comparison", {})
    ft_modes = comp.get("ft_modes", ["lora", "full"])
    methods = comp.get("methods", ["repsim", "trak"])
    tasks = comp.get("tasks", ["toxicity", "selection"])
    base_seed = config.get("reproducibility", {}).get("seed", 42)

    total = len(ft_modes) * len(methods) * len(tasks) * args.seeds
    print(f"LoRA vs FT | {total} total runs")

    passed = 0
    for ft_mode in ft_modes:
        for method in methods:
            for task in tasks:
                for seed_idx in range(args.seeds):
                    seed = base_seed + seed_idx
                    name = f"{ft_mode}_{method}_{task}_seed{seed}"
                    print(f"\n--- {name} ---")

                    cmd = [
                        sys.executable, str(run_script),
                        "--config", config_path,
                        "--seed", str(seed),
                        "--override",
                        f"fine_tuning.mode={ft_mode}",
                        f"attribution.method={method}",
                        f"evaluation.task={task}",
                    ]
                    if args.dry_run:
                        cmd.extend(["--dry-run", "--max-steps", str(args.max_steps)])

                    result = subprocess.run(cmd, capture_output=False)
                    if result.returncode == 0:
                        passed += 1

    print(f"\n{'='*40}")
    print(f"Complete: {passed}/{total} passed")
    if args.dry_run:
        print("[DRY RUN PASSED]" if passed == total else "[DRY RUN FAILED]")


if __name__ == "__main__":
    main()
