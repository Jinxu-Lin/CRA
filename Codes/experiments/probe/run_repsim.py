#!/usr/bin/env python3
"""
Probe Experiment 0: Run RepSim attribution on DATE-LM toxicity.

Computes RepSim scores at specified layers and saves to _Data/scores/.
Supports --random-init for control experiment (Design-Review Mandatory #4).
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
from logging_utils import ExperimentLogger, get_gpu_info


def main():
    parser = argparse.ArgumentParser(description="Probe: RepSim Attribution")
    add_common_args(parser)
    parser.add_argument("--random-init", action="store_true", help="Use random model weights")
    parser.add_argument("--layers", nargs="+", default=None,
                        help="Layers to evaluate (default: middle + last)")
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = config["evaluation"]["task"]

    print(f"Probe RepSim | Task: {task} | Random init: {args.random_init} | Dry run: {args.dry_run}")

    # Determine layers to evaluate
    n_layers = config["model"]["n_layers"]
    if args.layers:
        layers = args.layers
    else:
        layers = ["middle", "last"]

    # Load model
    from run_attribution import load_model
    model = load_model(config, device, random_init=args.random_init)

    # Load data
    from core.data.date_lm_loader import create_dataloader
    data_path = expand_path(config["paths"]["data"])
    batch_size = config.get("data", {}).get("batch_size", 32)
    max_samples = config.get("data", {}).get("max_samples", None)
    if args.dry_run:
        max_samples = args.max_steps * batch_size * 2

    train_loader = create_dataloader(data_path, task, "train", batch_size, max_samples)
    test_loader = create_dataloader(data_path, task, "test", batch_size,
                                     min(max_samples, 20) if args.dry_run else None)

    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Compute scores for each layer
    from core.data.representation import extract_representations
    from core.attribution.repsim import repsim_score_batched

    aggregation = config["attribution"].get("token_aggregation", "last_token")
    results = {}

    for layer_spec in layers:
        print(f"\n--- Layer: {layer_spec} ---")
        start = time.time()

        h_train = extract_representations(model, train_loader, layer_spec, aggregation, n_layers, device)
        h_test = extract_representations(model, test_loader, layer_spec, aggregation, n_layers, device)

        scores = repsim_score_batched(h_test.float(), h_train.float())
        elapsed = time.time() - start

        print(f"  Representations: train={h_train.shape}, test={h_test.shape}")
        print(f"  Scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Time: {elapsed:.1f}s")

        # Save
        prefix = "random_repsim" if args.random_init else "repsim"
        cache_dir = Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task
        cache_dir.mkdir(parents=True, exist_ok=True)
        score_file = cache_dir / f"{prefix}_{layer_spec}_seed{seed}.pt"
        torch.save(scores, score_file)
        print(f"  Saved: {score_file}")

        results[str(layer_spec)] = {
            "scores_shape": list(scores.shape),
            "score_mean": scores.mean().item(),
            "score_std": scores.std().item(),
            "wall_time_seconds": elapsed,
        }

    # Save results summary
    results_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "probe"
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "random_repsim" if args.random_init else "repsim"
    summary_file = results_dir / f"{prefix}_scores.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_file}")

    if args.dry_run:
        print("\n[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
