#!/usr/bin/env python3
"""
CRA Evaluation Script.

Loads attribution scores and computes all metrics defined in experiment-design.md Section 5.
Outputs results as Markdown tables to _Results/.

Usage:
    python evaluate.py --config configs/base.yaml
    python evaluate.py --config configs/base.yaml --checkpoint _Data/scores/toxicity/repsim_standard_seed42.pt
    python evaluate.py --config configs/base.yaml --dry-run
"""

import argparse
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config_utils import load_config, add_common_args, get_config_from_args, expand_path
from seed_utils import set_seed
from core.evaluation.metrics import (
    lds, auprc, precision_at_k, recall_at_k, mrr, compute_all_metrics
)
from core.evaluation.statistical import (
    permutation_test, bootstrap_ci, cohens_d, pairwise_significance
)
from core.evaluation.ablation_analysis import (
    compute_main_effects, assess_independence, full_ablation_analysis, compute_cmrr
)
from core.data.date_lm_loader import get_task_labels, get_actual_changes


def load_scores(scores_dir: Path, task: str, method: str, scoring: str, seed: int) -> Optional[torch.Tensor]:
    """Load attribution scores from file."""
    score_file = scores_dir / task / f"{method}_{scoring}_seed{seed}.pt"
    if score_file.exists():
        return torch.load(score_file, weights_only=False)
    return None


def generate_comparison_table(
    results: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str = "Method Comparison",
) -> str:
    """Generate a Markdown comparison table."""
    lines = [f"## {title}", ""]

    # Header
    header = "| Method | " + " | ".join(metric_names) + " |"
    sep = "|--------|" + "|".join(["-------:" for _ in metric_names]) + "|"
    lines.extend([header, sep])

    # Find best per metric
    best = {}
    for metric in metric_names:
        vals = [(m, r.get(metric, float("-inf"))) for m, r in results.items()]
        if vals:
            best[metric] = max(vals, key=lambda x: x[1])[0]

    # Rows
    for method, metrics in sorted(results.items()):
        cells = []
        for metric in metric_names:
            val = metrics.get(metric, None)
            if val is not None:
                cell = f"{val:.4f}"
                if best.get(metric) == method:
                    cell = f"**{cell}**"
                cells.append(cell)
            else:
                cells.append("--")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_result_md(
    results: Dict[str, Dict[str, float]],
    config: Dict,
    task: str,
    extra_info: Optional[Dict] = None,
) -> str:
    """Generate a full result Markdown file."""
    lines = [
        f"# CRA Results: {task.capitalize()}",
        "",
        f"> Config: {config.get('_config_path', 'unknown')}",
        f"> Seed: {config.get('reproducibility', {}).get('seed', 'unknown')}",
        f"> Model: {config.get('model', {}).get('name', 'unknown')}",
        "",
    ]

    # Determine metrics per task
    metric_names = config["evaluation"].get("metrics", ["lds", "auprc", "pk"])
    task_metrics = {
        "toxicity": ["lds", "auprc", "pk"],
        "selection": ["lds", "pk"],
        "factual": ["lds", "recall", "mrr", "pk"],
    }
    metric_names = task_metrics.get(task, metric_names)

    # Main comparison table
    table = generate_comparison_table(results, metric_names, f"Results on {task.capitalize()}")
    lines.append(table)

    # Efficiency metrics if available
    if any("gpu_hours_per_1k" in r for r in results.values()):
        eff_table = generate_comparison_table(
            {m: {k: v for k, v in r.items() if k in ("gpu_hours_per_1k", "peak_memory_gb", "throughput")}
             for m, r in results.items()},
            ["gpu_hours_per_1k", "peak_memory_gb", "throughput"],
            "Efficiency Profile",
        )
        lines.append(eff_table)

    # Extra info
    if extra_info:
        lines.append("## Additional Analysis")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(extra_info, indent=2, default=str))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def dry_run_evaluation(config: Dict, output_dir: Path):
    """Generate a dry-run evaluation with dummy data."""
    task = config["evaluation"]["task"]
    methods = ["repsim", "trak", "gradsim", "rept", "bm25", "random"]

    print(f"[DRY RUN] Generating evaluation with dummy data...")

    n_train = 100
    n_test = 10
    results = {}

    for method in methods:
        # Generate dummy scores
        scores = torch.randn(n_test, n_train)
        labels = (torch.rand(n_train) > 0.9).long()  # ~10% positive
        actual_changes = torch.randn(n_train)

        per_test_scores = scores.mean(dim=1)

        method_results = {}
        method_results["lds"] = lds(per_test_scores[:n_train] if n_test >= n_train
                                     else torch.randn(n_train),
                                     actual_changes).item()
        method_results["auprc"] = auprc(scores[0], labels).item()
        method_results["pk"] = precision_at_k(scores[0], labels, k=10).item()
        method_results["recall"] = recall_at_k(scores[0], labels, k=50).item()
        method_results["mrr"] = mrr(scores[0], labels).item()

        results[method] = method_results
        print(f"  {method}: LDS={method_results['lds']:.4f}, AUPRC={method_results['auprc']:.4f}")

    # Generate Markdown
    md_content = generate_result_md(results, config, task)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task}_results.md"
    with open(output_file, "w") as f:
        f.write(md_content)
    print(f"\n  Output: {output_file}")

    # Verify all metric functions callable
    print(f"\n[DRY RUN] Metric function verification:")
    test_pred = torch.randn(50)
    test_labels = (torch.rand(50) > 0.5).long()
    test_changes = torch.randn(50)

    print(f"  lds(): {lds(test_pred, test_changes).item():.4f} -- OK")
    print(f"  auprc(): {auprc(test_pred, test_labels).item():.4f} -- OK")
    print(f"  precision_at_k(): {precision_at_k(test_pred, test_labels, 10).item():.4f} -- OK")
    print(f"  recall_at_k(): {recall_at_k(test_pred, test_labels, 50).item():.4f} -- OK")
    print(f"  mrr(): {mrr(test_pred, test_labels).item():.4f} -- OK")

    # Verify statistical functions
    a, b = torch.randn(30), torch.randn(30)
    diff, pval = permutation_test(a, b, n_permutations=100, seed=42)
    print(f"  permutation_test(): diff={diff:.4f}, p={pval:.4f} -- OK")
    est, ci_lo, ci_hi = bootstrap_ci(a, n_bootstrap=100, seed=42)
    print(f"  bootstrap_ci(): est={est:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}] -- OK")
    d = cohens_d(a, b)
    print(f"  cohens_d(): d={d:.4f} -- OK")

    # Verify ablation analysis
    c1, c2, c3, c4 = torch.randn(20), torch.randn(20), torch.randn(20), torch.randn(20)
    effects = compute_main_effects(c1, c2, c3, c4)
    print(f"  compute_main_effects(): FM1={effects['fm1_effect']:.4f}, FM2={effects['fm2_effect']:.4f} -- OK")
    assessment = assess_independence(effects['interaction_ratio'])
    print(f"  assess_independence(): {assessment} -- OK")

    print(f"\n[DRY RUN PASSED]")


def full_evaluation(config: Dict, args, output_dir: Path):
    """Run full evaluation on computed scores."""
    task = config["evaluation"]["task"]
    data_path = expand_path(config["paths"]["data"])
    cache_path = expand_path(config["paths"].get("cache", "_Data"))
    scores_dir = Path(cache_path) / "scores"
    n_seeds = config["evaluation"].get("n_seeds", 1)

    labels = get_task_labels(data_path, task, "train")
    actual_changes = get_actual_changes(data_path, task)

    methods = ["repsim", "trak", "gradsim", "rept", "bm25", "random"]
    scorings = ["standard"]
    if config["attribution"].get("scoring") == "contrastive":
        scorings.append("contrastive")

    results = {}
    for method in methods:
        for scoring in scorings:
            all_seed_metrics = []
            for seed in range(n_seeds):
                actual_seed = config.get("reproducibility", {}).get("seed", 42) + seed
                scores = load_scores(scores_dir, task, method, scoring, actual_seed)
                if scores is None:
                    continue

                # Per-test-sample mean score as aggregate
                per_test_scores = scores.mean(dim=1)

                metric_names = config["evaluation"].get("metrics", ["lds", "auprc", "pk"])
                seed_metrics = compute_all_metrics(
                    scores[0] if scores.shape[0] > 0 else torch.tensor([]),
                    labels=labels,
                    actual_changes=actual_changes,
                    metric_names=metric_names,
                )
                all_seed_metrics.append(seed_metrics)

            if all_seed_metrics:
                # Average over seeds
                avg_metrics = {}
                for key in all_seed_metrics[0]:
                    vals = [m[key] for m in all_seed_metrics if key in m]
                    avg_metrics[key] = np.mean(vals)
                    if len(vals) > 1:
                        avg_metrics[f"{key}_std"] = np.std(vals)

                name = f"{method}" + (f"_{scoring}" if scoring != "standard" else "")
                results[name] = avg_metrics

    if results:
        md_content = generate_result_md(results, config, task)
        output_file = output_dir / f"{task}_results.md"
        with open(output_file, "w") as f:
            f.write(md_content)
        print(f"Output: {output_file}")
    else:
        print("No scores found. Run run_attribution.py first.")


def main():
    parser = argparse.ArgumentParser(description="CRA Evaluation Script")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific score file to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    task = config["evaluation"]["task"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        expand_path(config["paths"].get("output", "_Results"))
    )

    print(f"=" * 60)
    print(f"CRA Evaluation")
    print(f"  Task: {task}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Output: {output_dir}")
    print(f"=" * 60)

    if args.dry_run:
        dry_run_evaluation(config, output_dir)
    else:
        full_evaluation(config, args, output_dir)


if __name__ == "__main__":
    main()
