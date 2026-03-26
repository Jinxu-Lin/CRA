#!/usr/bin/env python3
"""
Probe Experiment 0: Run Grad-Sim attribution.
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
    parser = argparse.ArgumentParser(description="Probe: Grad-Sim Attribution")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    task = config["evaluation"]["task"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Probe Grad-Sim | Task: {task} | Dry run: {args.dry_run}")

    from run_attribution import load_model, compute_scores_gradsim
    from core.data.date_lm_loader import create_dataloader

    model = load_model(config, device)
    data_path = expand_path(config["paths"]["data"])
    batch_size = config.get("data", {}).get("batch_size", 32)
    max_samples = args.max_steps * batch_size * 2 if args.dry_run else config.get("data", {}).get("max_samples")

    train_loader = create_dataloader(data_path, task, "train", batch_size, max_samples)
    test_loader = create_dataloader(data_path, task, "test", batch_size,
                                     min(int(max_samples or 1e9), 20) if args.dry_run else None)

    start = time.time()
    scores, features = compute_scores_gradsim(
        model, config, train_loader, test_loader, device,
        dry_run=args.dry_run, max_steps=args.max_steps,
    )
    elapsed = time.time() - start

    # Save
    cache_dir = Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_file = cache_dir / f"gradsim_standard_seed{seed}.pt"
    torch.save(scores, score_file)

    results_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "probe"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "gradsim_scores.json", "w") as f:
        json.dump({
            "scores_shape": list(scores.shape),
            "wall_time_seconds": elapsed,
        }, f, indent=2)

    print(f"Scores saved: {score_file}")
    print(f"Time: {elapsed:.1f}s")

    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
