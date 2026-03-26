#!/usr/bin/env python3
"""
Experiment 1: RepSim layer sweep -- evaluate all layers.
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
    parser = argparse.ArgumentParser(description="Benchmark: Layer Sweep")
    add_common_args(parser)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    task = args.task or config["evaluation"]["task"]
    n_layers = config["model"]["n_layers"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Layer Sweep | Task: {task} | Layers: 0-{n_layers-1}")

    from run_attribution import load_model
    from core.data.date_lm_loader import create_dataloader
    from core.data.representation import extract_representations
    from core.attribution.repsim import repsim_score_batched

    model = load_model(config, device)
    data_path = expand_path(config["paths"]["data"])
    batch_size = config.get("data", {}).get("batch_size", 32)

    if args.dry_run:
        max_samples = args.max_steps * batch_size * 2
        layers_to_sweep = [0, n_layers // 2, n_layers - 1]
    else:
        max_samples = config.get("data", {}).get("max_samples")
        layers_to_sweep = list(range(n_layers))

    train_loader = create_dataloader(data_path, task, "train", batch_size, max_samples)
    test_loader = create_dataloader(data_path, task, "test", batch_size,
                                     min(int(max_samples or 1e9), 20) if args.dry_run else None)

    aggregation = config["attribution"].get("token_aggregation", "last_token")
    results = {}

    for layer in layers_to_sweep:
        start = time.time()
        h_train = extract_representations(model, train_loader, layer, aggregation, n_layers, device)
        h_test = extract_representations(model, test_loader, layer, aggregation, n_layers, device)
        scores = repsim_score_batched(h_test.float(), h_train.float())
        elapsed = time.time() - start

        results[layer] = {
            "score_mean": scores.mean().item(),
            "score_std": scores.std().item(),
            "wall_time": elapsed,
        }
        print(f"  Layer {layer:2d}: mean={scores.mean():.4f}, std={scores.std():.4f}, time={elapsed:.1f}s")

    # Save
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "layer_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    md_lines = ["# Layer Sweep Results", "", f"> Task: {task}", "", "| Layer | Mean Score | Std | Time (s) |",
                "|-------|-----------|-----|----------|"]
    for l, r in sorted(results.items(), key=lambda x: int(x[0])):
        md_lines.append(f"| {l} | {r['score_mean']:.4f} | {r['score_std']:.4f} | {r['wall_time']:.1f} |")

    with open(output_dir / "layer_sweep.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nResults: {output_dir / 'layer_sweep.md'}")
    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
