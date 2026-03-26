#!/usr/bin/env python3
"""
Experiment 1: Statistical analysis of benchmark results.

Pairwise permutation tests, bootstrap CIs, BH-FDR correction, Cohen's d.
"""

import argparse
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, get_config_from_args, add_common_args, expand_path
from seed_utils import set_seed
from core.evaluation.statistical import pairwise_significance, bootstrap_ci


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Statistical Analysis")
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config_from_args(args)
    task = config["evaluation"]["task"]
    seed = config.get("reproducibility", {}).get("seed", 42)

    scores_dir = Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Statistical Analysis | Task: {task}")

    # Load per-test-sample scores for each method
    method_scores = {}
    if scores_dir.exists():
        for f in sorted(scores_dir.glob("*.pt")):
            scores = torch.load(f, weights_only=False)
            per_test = scores.mean(dim=1)
            method_scores[f.stem] = per_test

    if args.dry_run:
        # Generate dummy data
        for name in ["repsim_standard_seed42", "trak_standard_seed42", "gradsim_standard_seed42"]:
            method_scores[name] = torch.randn(50)

    if len(method_scores) < 2:
        print("Need at least 2 methods for comparison. Run benchmark first.")
        return

    # Truncate to same length
    min_len = min(len(v) for v in method_scores.values())
    method_scores = {k: v[:min_len] for k, v in method_scores.items()}

    print(f"Methods: {list(method_scores.keys())}")
    print(f"Samples per method: {min_len}")

    # Pairwise significance
    n_perm = 100 if args.dry_run else 10000
    results = pairwise_significance(method_scores, n_permutations=n_perm, seed=seed)

    print(f"\nPairwise comparisons ({results['n_significant']} significant):")
    for comp in results["comparisons"]:
        sig = "*" if comp["significant"] else ""
        print(f"  {comp['method_a']} vs {comp['method_b']}: "
              f"diff={comp['mean_diff']:.4f}, p={comp['p_value']:.4f}{sig}, d={comp['cohens_d']:.4f}")

    # Bootstrap CIs per method
    cis = {}
    n_boot = 100 if args.dry_run else 1000
    for name, scores in method_scores.items():
        est, lo, hi = bootstrap_ci(scores, n_bootstrap=n_boot, seed=seed)
        cis[name] = {"mean": est, "ci_lower": lo, "ci_upper": hi}

    # Save
    output = {
        "pairwise": results,
        "bootstrap_cis": cis,
    }
    with open(output_dir / "statistical_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    md_lines = [
        "# Statistical Analysis", "",
        "## Pairwise Comparisons", "",
        "| Method A | Method B | Diff | p-value | Significant | Cohen's d |",
        "|----------|----------|------|---------|-------------|-----------|",
    ]
    for comp in results["comparisons"]:
        sig = "Yes" if comp["significant"] else "No"
        md_lines.append(
            f"| {comp['method_a']} | {comp['method_b']} | {comp['mean_diff']:.4f} | "
            f"{comp['p_value']:.4f} | {sig} | {comp['cohens_d']:.4f} |"
        )
    md_lines.extend(["", "## Bootstrap 95% CIs", "", "| Method | Mean | CI Lower | CI Upper |",
                      "|--------|------|----------|----------|"])
    for name, ci in sorted(cis.items()):
        md_lines.append(f"| {name} | {ci['mean']:.4f} | {ci['ci_lower']:.4f} | {ci['ci_upper']:.4f} |")

    with open(output_dir / "benchmark_results.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nResults: {output_dir / 'benchmark_results.md'}")
    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
