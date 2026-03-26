#!/usr/bin/env python3
"""
Experiment 0.5: Evaluate 2x2 pilot results.

Computes FM1/FM2 main effects and interaction from pilot scores.
"""

import argparse
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_utils import load_config, get_config_from_args, add_common_args, expand_path
from seed_utils import set_seed
from core.evaluation.ablation_analysis import compute_main_effects, assess_independence


def main():
    parser = argparse.ArgumentParser(description="Pilot: Evaluate 2x2")
    add_common_args(parser)
    parser.add_argument("--scores-dir", type=str, default=None)
    args = parser.parse_args()

    config = get_config_from_args(args)
    task = config["evaluation"]["task"]
    seed = config.get("reproducibility", {}).get("seed", 42)

    scores_dir = Path(args.scores_dir) if args.scores_dir else (
        Path(expand_path(config["paths"].get("cache", "_Data"))) / "scores" / task
    )
    output_dir = Path(expand_path(config["paths"].get("output", "_Results"))) / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pilot 2x2 Evaluation | Task: {task}")

    # Load 4 conditions (or use dummy for dry-run)
    if args.dry_run:
        n = 50
        cell_param_std = torch.randn(n) * 0.1 + 0.15
        cell_param_contr = torch.randn(n) * 0.1 + 0.20
        cell_repr_std = torch.randn(n) * 0.1 + 0.25
        cell_repr_contr = torch.randn(n) * 0.1 + 0.30
    else:
        # Load actual scores
        def load_cell(method, scoring):
            f = scores_dir / f"{method}_{scoring}_seed{seed}.pt"
            if f.exists():
                scores = torch.load(f, weights_only=False)
                return scores.mean(dim=1)  # aggregate per test sample
            return None

        cell_param_std = load_cell("trak", "standard")
        cell_param_contr = load_cell("trak", "contrastive")
        cell_repr_std = load_cell("repsim", "standard")
        cell_repr_contr = load_cell("repsim", "contrastive")

        if any(c is None for c in [cell_param_std, cell_param_contr, cell_repr_std, cell_repr_contr]):
            print("Not all 4 conditions available. Run pilot experiments first.")
            return

    # Compute effects
    effects = compute_main_effects(cell_param_std, cell_param_contr, cell_repr_std, cell_repr_contr)
    assessment = assess_independence(effects["interaction_ratio"])

    print(f"\nFM1 effect (repr - param): {effects['fm1_effect']:.4f}")
    print(f"FM2 effect (contrastive - standard): {effects['fm2_effect']:.4f}")
    print(f"Interaction: {effects['interaction']:.4f}")
    print(f"Interaction ratio: {effects['interaction_ratio']:.4f}")
    print(f"Independence: {assessment}")

    # Gate decision
    fm1_positive = effects["fm1_effect"] > 0
    fm2_positive = effects["fm2_effect"] > 0

    if fm1_positive and fm2_positive:
        gate = "PASS: Both FM1 and FM2 effects positive"
    elif fm1_positive or fm2_positive:
        gate = "ADJUST: Only one effect positive"
    else:
        gate = "FAIL: Both effects non-positive"

    print(f"\nGate: {gate}")

    # Write summary
    summary = {
        "effects": effects,
        "assessment": assessment,
        "gate_decision": gate,
        "cell_means": {
            "param_std": cell_param_std.mean().item(),
            "param_contr": cell_param_contr.mean().item(),
            "repr_std": cell_repr_std.mean().item(),
            "repr_contr": cell_repr_contr.mean().item(),
        },
    }

    md_lines = [
        "# Pilot 2x2 Summary",
        "",
        f"> Task: {task}",
        "",
        "## 2x2 Table (Mean Scores)",
        "",
        "| | Standard | Contrastive |",
        "|--|----------|-------------|",
        f"| Parameter (TRAK) | {summary['cell_means']['param_std']:.4f} | {summary['cell_means']['param_contr']:.4f} |",
        f"| Representation (RepSim) | {summary['cell_means']['repr_std']:.4f} | {summary['cell_means']['repr_contr']:.4f} |",
        "",
        "## Effects",
        "",
        f"- **FM1 main effect**: {effects['fm1_effect']:.4f}",
        f"- **FM2 main effect**: {effects['fm2_effect']:.4f}",
        f"- **Interaction**: {effects['interaction']:.4f}",
        f"- **Interaction ratio**: {effects['interaction_ratio']:.4f}",
        f"- **Independence**: {assessment}",
        "",
        f"## Gate Decision: {gate}",
    ]

    with open(output_dir / "pilot_summary.md", "w") as f:
        f.write("\n".join(md_lines))
    with open(output_dir / "pilot_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary: {output_dir / 'pilot_summary.md'}")

    if args.dry_run:
        print("[DRY RUN PASSED]")


if __name__ == "__main__":
    main()
