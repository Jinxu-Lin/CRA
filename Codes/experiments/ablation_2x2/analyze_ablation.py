#!/usr/bin/env python3
"""
Experiment 2: Analyze 2x2 ablation results.

Computes FM1/FM2 main effects, interaction, CMRR, significance tests.
"""

import argparse
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, get_config_from_args, add_common_args, expand_path
from core.evaluation.ablation_analysis import full_ablation_analysis, compute_cmrr


def main():
    parser = argparse.ArgumentParser(description="Ablation: Analyze Results")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = config.get("ablation", {}).get("tasks", ["toxicity", "selection", "factual"])
    all_results = {}

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        scores_dir = Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task

        if args.dry_run:
            n = 50
            cells = {
                "param_std": torch.randn(n) * 0.1 + 0.15,
                "param_contr": torch.randn(n) * 0.1 + 0.20,
                "repr_std": torch.randn(n) * 0.1 + 0.25,
                "repr_contr": torch.randn(n) * 0.1 + 0.30,
            }
        else:
            def load_cell(method, scoring):
                f = scores_dir / f"{method}_{scoring}_seed{seed}.pt"
                if f.exists():
                    return torch.load(f, weights_only=False).mean(dim=1)
                return None

            cells = {
                "param_std": load_cell("trak", "standard"),
                "param_contr": load_cell("trak", "contrastive"),
                "repr_std": load_cell("repsim", "standard"),
                "repr_contr": load_cell("repsim", "contrastive"),
            }
            if any(v is None for v in cells.values()):
                print(f"  Skipping {task}: missing conditions")
                continue

        n_perm = 100 if args.dry_run else 10000
        n_boot = 100 if args.dry_run else 1000

        result = full_ablation_analysis(
            cells["param_std"], cells["param_contr"],
            cells["repr_std"], cells["repr_contr"],
            n_permutations=n_perm, n_bootstrap=n_boot, seed=seed,
        )

        # CMRR
        if not args.dry_run:
            std_scores = load_cell("repsim", "standard") or cells["repr_std"]
            contr_scores = load_cell("repsim", "contrastive") or cells["repr_contr"]
        else:
            std_scores = cells["repr_std"]
            contr_scores = cells["repr_contr"]

        cmrr_val = compute_cmrr(std_scores.unsqueeze(0), contr_scores.unsqueeze(0))
        result["cmrr"] = cmrr_val

        all_results[task] = result
        me = result["main_effects"]
        print(f"  FM1: {me['fm1_effect']:.4f} (p={result['fm1_test']['p_value']:.4f})")
        print(f"  FM2: {me['fm2_effect']:.4f} (p={result['fm2_test']['p_value']:.4f})")
        print(f"  Interaction: {me['interaction']:.4f} (ratio={me['interaction_ratio']:.4f})")
        print(f"  Independence: {result['independence_assessment']}")
        print(f"  CMRR: {cmrr_val:.4f}")

    # Save
    with open(output_dir / "ablation_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    md_lines = [
        "# 2x2 Ablation Results", "",
        "## Main Effects by Task", "",
        "| Task | FM1 Effect | FM2 Effect | Interaction | Ratio | Independence | CMRR |",
        "|------|-----------|-----------|------------|-------|-------------|------|",
    ]
    for task, r in all_results.items():
        me = r["main_effects"]
        md_lines.append(
            f"| {task} | {me['fm1_effect']:.4f} | {me['fm2_effect']:.4f} | "
            f"{me['interaction']:.4f} | {me['interaction_ratio']:.4f} | "
            f"{r['independence_assessment']} | {r.get('cmrr', 0):.4f} |"
        )

    with open(output_dir / "ablation_results.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nResults: {output_dir / 'ablation_results.md'}")
    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
