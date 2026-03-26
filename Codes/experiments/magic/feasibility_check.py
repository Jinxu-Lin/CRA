#!/usr/bin/env python3
"""
Experiment 4: MAGIC feasibility assessment.
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, get_config_from_args, add_common_args, expand_path


def main():
    parser = argparse.ArgumentParser(description="MAGIC: Feasibility Check")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "magic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Estimate without loading model
    n_params = 1_000_000_000  # Pythia-1B
    model_size_gb = n_params * 2 / (1024**3)  # fp16
    n_train = 10000
    n_test = config.get("magic", {}).get("n_test_samples", 5)
    n_steps = config.get("fine_tuning", {}).get("decay_steps", 200)

    checkpoint_gb = n_steps * model_size_gb * 3
    time_per_test_hours = (n_params / 1e9) * 4.0
    total_time_hours = time_per_test_hours * n_test

    report = {
        "n_params": n_params,
        "model_size_gb_fp16": model_size_gb,
        "n_train": n_train,
        "n_test": n_test,
        "n_steps": n_steps,
        "estimated_checkpoint_storage_gb": checkpoint_gb,
        "estimated_time_per_test_hours": time_per_test_hours,
        "estimated_total_time_hours": total_time_hours,
        "estimated_total_time_gpu_days": total_time_hours / 24,
        "feasible": checkpoint_gb < 500 and total_time_hours < 500,
    }

    md_lines = [
        "# MAGIC Feasibility Report", "",
        f"- Model: Pythia-1B ({n_params/1e9:.1f}B params)",
        f"- Training steps: {n_steps}",
        f"- Test samples: {n_test}",
        "",
        "## Storage",
        f"- Checkpoint storage: **{checkpoint_gb:.0f} GB** (200 steps x {model_size_gb*3:.1f} GB/step)",
        f"- {'FEASIBLE' if checkpoint_gb < 500 else 'INFEASIBLE'} (limit: 500 GB)",
        "",
        "## Compute",
        f"- Time per test sample: ~{time_per_test_hours:.1f} hours",
        f"- Total for {n_test} samples: ~{total_time_hours:.0f} hours ({total_time_hours/24:.1f} GPU-days)",
        f"- {'FEASIBLE' if total_time_hours < 500 else 'INFEASIBLE'} (budget: 5 GPU-days = 120 hours)",
        "",
        f"## Verdict: {'FEASIBLE' if report['feasible'] else 'LIKELY INFEASIBLE'}",
    ]

    with open(output_dir / "feasibility_report.md", "w") as f:
        f.write("\n".join(md_lines))
    with open(output_dir / "feasibility_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n".join(md_lines))

    if args.dry_run:
        print("\n[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
