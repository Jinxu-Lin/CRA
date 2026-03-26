#!/usr/bin/env python3
"""
Experiment 1: Profile efficiency of all methods.

Measures: GPU-hours per 1K test samples, peak GPU memory, throughput.
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

METHODS = ["repsim", "rept", "gradsim"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Efficiency Profiling")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    task = config["evaluation"]["task"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Efficiency Profiling | Task: {task}")

    from run_attribution import (
        load_model, compute_scores_repsim, compute_scores_rept, compute_scores_gradsim
    )
    from core.data.date_lm_loader import create_dataloader

    model = load_model(config, device)
    data_path = expand_path(config["paths"]["data"])
    batch_size = config.get("data", {}).get("batch_size", 32)
    max_samples = args.max_steps * batch_size * 2 if args.dry_run else 100

    train_loader = create_dataloader(data_path, task, "train", batch_size, max_samples)
    test_loader = create_dataloader(data_path, task, "test", batch_size,
                                     min(max_samples, 20) if args.dry_run else 10)

    scoring_fns = {
        "repsim": compute_scores_repsim,
        "rept": compute_scores_rept,
        "gradsim": compute_scores_gradsim,
    }

    profiles = {}
    for method in METHODS:
        if method not in scoring_fns:
            continue

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        n_test = len(test_loader.dataset)
        start = time.time()
        try:
            scores, _ = scoring_fns[method](
                model, config, train_loader, test_loader, device,
                dry_run=args.dry_run, max_steps=args.max_steps,
            )
            elapsed = time.time() - start
            success = True
        except Exception as e:
            elapsed = time.time() - start
            success = False
            print(f"  {method}: FAILED ({e})")
            continue

        peak_mem = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        gpu_hours_per_1k = (elapsed / n_test) * 1000 / 3600 if n_test > 0 else 0
        throughput = n_test / (elapsed / 3600) if elapsed > 0 else 0

        profiles[method] = {
            "gpu_hours_per_1k": gpu_hours_per_1k,
            "peak_memory_gb": peak_mem,
            "throughput_per_gpu_hour": throughput,
            "wall_time_seconds": elapsed,
            "n_test": n_test,
        }
        print(f"  {method}: {gpu_hours_per_1k:.4f} GPU-hr/1K, {peak_mem:.2f} GB peak, {throughput:.0f} samples/GPU-hr")

    # Save
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "efficiency_profile.json", "w") as f:
        json.dump(profiles, f, indent=2)

    md_lines = ["# Efficiency Profile", "", "| Method | GPU-hr/1K | Peak Mem (GB) | Throughput |",
                "|--------|-----------|--------------|------------|"]
    for method, p in sorted(profiles.items()):
        md_lines.append(
            f"| {method} | {p['gpu_hours_per_1k']:.4f} | {p['peak_memory_gb']:.2f} | {p['throughput_per_gpu_hour']:.0f} |"
        )

    with open(output_dir / "efficiency_profile.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nProfile: {output_dir / 'efficiency_profile.md'}")
    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
