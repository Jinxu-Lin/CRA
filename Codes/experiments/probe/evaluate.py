#!/usr/bin/env python3
"""
Probe Experiment 0: Evaluate all probe scores and generate summary.

Reads scores from _Data/scores/toxicity/ and computes LDS, AUPRC, P@K.
Generates _Results/probe/evaluation_summary.md with gate decision.
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
from core.evaluation.metrics import lds, auprc, precision_at_k, compute_all_metrics
from core.data.date_lm_loader import get_task_labels, get_actual_changes


def main():
    parser = argparse.ArgumentParser(description="Probe: Evaluate All Scores")
    add_common_args(parser)
    parser.add_argument("--scores-dir", type=str, default=None, help="Directory with score files")
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    task = config["evaluation"]["task"]

    scores_dir = Path(args.scores_dir) if args.scores_dir else (
        Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task
    )
    data_path = expand_path(config["paths"]["data"])
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probe Evaluation | Task: {task} | Scores dir: {scores_dir}")

    # Load ground truth
    labels = get_task_labels(data_path, task, "train")
    actual_changes = get_actual_changes(data_path, task)

    # Discover and evaluate all score files
    results = {}
    if scores_dir.exists():
        for score_file in sorted(scores_dir.glob("*.pt")):
            method_name = score_file.stem
            scores = torch.load(score_file, weights_only=False)
            print(f"\n  Evaluating: {method_name} (shape={scores.shape})")

            method_metrics = {}

            # LDS: requires actual_changes
            if actual_changes is not None:
                per_test = scores.mean(dim=1)
                if len(per_test) <= len(actual_changes):
                    lds_val = lds(per_test, actual_changes[:len(per_test)]).item()
                    method_metrics["lds"] = lds_val
                    print(f"    LDS: {lds_val:.4f}")

            # Retrieval metrics: per test sample
            if labels is not None and scores.shape[1] == len(labels):
                auprc_vals = []
                pk_vals = []
                for i in range(scores.shape[0]):
                    auprc_vals.append(auprc(scores[i], labels).item())
                    pk_vals.append(precision_at_k(scores[i], labels, k=10).item())
                method_metrics["auprc_mean"] = sum(auprc_vals) / len(auprc_vals)
                method_metrics["pk_mean"] = sum(pk_vals) / len(pk_vals)
                print(f"    AUPRC: {method_metrics['auprc_mean']:.4f}")
                print(f"    P@10: {method_metrics['pk_mean']:.4f}")

            results[method_name] = method_metrics

    # Generate summary
    lines = [
        "# Probe Evaluation Summary",
        "",
        f"> Task: {task}",
        f"> Seed: {seed}",
        f"> Model: {config['model']['name']}",
        "",
        "## Results",
        "",
    ]

    if results:
        # Table
        metrics_cols = ["lds", "auprc_mean", "pk_mean"]
        header = "| Method | " + " | ".join(metrics_cols) + " |"
        sep = "|--------|" + "|".join(["-------:" for _ in metrics_cols]) + "|"
        lines.extend([header, sep])

        for method, metrics in sorted(results.items()):
            cells = [f"{metrics.get(m, '--'):.4f}" if isinstance(metrics.get(m), float) else "--"
                     for m in metrics_cols]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")

        lines.append("")

        # Gate decision
        repsim_lds = results.get("repsim_middle_seed42", {}).get("lds")
        trak_lds = results.get("trak_standard_seed42", {}).get("lds")

        lines.append("## Gate Decision")
        lines.append("")
        if repsim_lds is not None and trak_lds is not None:
            diff = repsim_lds - trak_lds
            if diff >= 0:
                lines.append(f"**STRONG PASS**: RepSim LDS ({repsim_lds:.4f}) >= TRAK LDS ({trak_lds:.4f})")
            elif diff >= -0.05:
                lines.append(f"**PASS**: RepSim LDS ({repsim_lds:.4f}) >= TRAK LDS - 5pp ({trak_lds:.4f})")
            else:
                lines.append(f"**NEEDS REVIEW**: RepSim LDS ({repsim_lds:.4f}) < TRAK LDS - 5pp ({trak_lds:.4f})")
        else:
            lines.append("Gate decision pending: not all required scores available.")

    else:
        lines.append("No score files found. Run probe experiments first.")

    summary_file = output_dir / "evaluation_summary.md"
    with open(summary_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary written: {summary_file}")

    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
