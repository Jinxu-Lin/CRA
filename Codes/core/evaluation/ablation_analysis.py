# Component: 2x2 Ablation Analysis
# Source: research/experiment-design.md §3.2 (2x2 Ablation)
# Ablation config key: N/A (analysis utility)

"""
2x2 ablation analysis for FM1/FM2 decomposition.

Design: {parameter-space, representation-space} x {standard scoring, contrastive scoring}

Four cells:
- (param, standard): TRAK           -- FM1 present, FM2 present
- (param, contrastive): TRAK-C      -- FM1 present, FM2 fixed
- (repr, standard): RepSim          -- FM1 fixed, FM2 present
- (repr, contrastive): RepSim-C     -- FM1 fixed, FM2 fixed

Reports FM1 main effect, FM2 main effect, interaction term, and CMRR.
"""

import torch
from typing import Dict, Tuple, Optional

from core.evaluation.statistical import permutation_test, bootstrap_ci, cohens_d


def compute_main_effects(
    cell_param_std: torch.Tensor,
    cell_param_contr: torch.Tensor,
    cell_repr_std: torch.Tensor,
    cell_repr_contr: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute 2x2 main effects and interaction from per-sample scores.

    Args:
        cell_param_std: (N,) per-sample LDS for (parameter, standard).
        cell_param_contr: (N,) per-sample LDS for (parameter, contrastive).
        cell_repr_std: (N,) per-sample LDS for (representation, standard).
        cell_repr_contr: (N,) per-sample LDS for (representation, contrastive).

    Returns:
        Dict with:
        - fm1_effect: mean(repr conditions) - mean(param conditions)
        - fm2_effect: mean(contrastive conditions) - mean(standard conditions)
        - interaction: (repr+contr) - (repr+std) - (param+contr) + (param+std)
        - interaction_ratio: |interaction| / min(|fm1|, |fm2|) -- independence indicator
    """
    assert cell_param_std.shape == cell_param_contr.shape == cell_repr_std.shape == cell_repr_contr.shape

    # FM1 main effect: representation-space improves over parameter-space
    fm1_effect = (
        (cell_repr_std.mean() + cell_repr_contr.mean()) / 2 -
        (cell_param_std.mean() + cell_param_contr.mean()) / 2
    ).item()

    # FM2 main effect: contrastive scoring improves over standard scoring
    fm2_effect = (
        (cell_param_contr.mean() + cell_repr_contr.mean()) / 2 -
        (cell_param_std.mean() + cell_repr_std.mean()) / 2
    ).item()

    # Interaction: non-additive effect
    interaction = (
        cell_repr_contr.mean() - cell_repr_std.mean() -
        cell_param_contr.mean() + cell_param_std.mean()
    ).item()

    # Interaction ratio for independence assessment
    min_main = min(abs(fm1_effect), abs(fm2_effect))
    if min_main > 1e-10:
        interaction_ratio = abs(interaction) / min_main
    else:
        interaction_ratio = float("inf") if abs(interaction) > 1e-10 else 0.0

    return {
        "fm1_effect": fm1_effect,
        "fm2_effect": fm2_effect,
        "interaction": interaction,
        "interaction_ratio": interaction_ratio,
    }


def assess_independence(interaction_ratio: float) -> str:
    """
    Interpret interaction ratio per formalize review thresholds.

    Args:
        interaction_ratio: |interaction| / min(|FM1|, |FM2|)

    Returns:
        Assessment string.
    """
    if interaction_ratio < 0.10:
        return "clean_independence"
    elif interaction_ratio < 0.30:
        return "moderate_interaction"
    else:
        return "strong_interaction"


def compute_cmrr(
    score_standard: torch.Tensor,
    score_contrastive: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute Common-Mode Rejection Ratio (CMRR) as secondary FM2 metric.

    CMRR = mean(|standard - contrastive| / (|standard| + eps))

    Args:
        score_standard: (N,) or (n_test, n_train) standard scores.
        score_contrastive: (N,) or (n_test, n_train) contrastive scores.
        eps: Numerical stability.

    Returns:
        Mean CMRR value.
    """
    diff = (score_standard - score_contrastive).abs()
    denom = score_standard.abs() + eps
    return (diff / denom).mean().item()


def full_ablation_analysis(
    cell_param_std: torch.Tensor,
    cell_param_contr: torch.Tensor,
    cell_repr_std: torch.Tensor,
    cell_repr_contr: torch.Tensor,
    n_permutations: int = 10000,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Dict:
    """
    Complete 2x2 ablation analysis with statistical tests.

    Args:
        cell_param_std: (N,) per-sample scores for (param, standard).
        cell_param_contr: (N,) per-sample scores for (param, contrastive).
        cell_repr_std: (N,) per-sample scores for (repr, standard).
        cell_repr_contr: (N,) per-sample scores for (repr, contrastive).
        n_permutations: Permutation test iterations.
        n_bootstrap: Bootstrap CI iterations.
        seed: Random seed.

    Returns:
        Dict with main effects, interaction, significance tests, CIs, and assessment.
    """
    effects = compute_main_effects(
        cell_param_std, cell_param_contr, cell_repr_std, cell_repr_contr
    )

    # FM1 significance: repr vs param (averaged over scoring)
    repr_avg = (cell_repr_std + cell_repr_contr) / 2
    param_avg = (cell_param_std + cell_param_contr) / 2
    fm1_diff, fm1_pval = permutation_test(repr_avg, param_avg, n_permutations, seed)
    fm1_d = cohens_d(repr_avg, param_avg)

    # FM2 significance: contrastive vs standard (averaged over space)
    contr_avg = (cell_param_contr + cell_repr_contr) / 2
    std_avg = (cell_param_std + cell_repr_std) / 2
    fm2_diff, fm2_pval = permutation_test(contr_avg, std_avg, n_permutations, seed)
    fm2_d = cohens_d(contr_avg, std_avg)

    # Bootstrap CIs for main effects
    fm1_diffs = repr_avg - param_avg
    _, fm1_ci_low, fm1_ci_high = bootstrap_ci(fm1_diffs, n_bootstrap=n_bootstrap, seed=seed)

    fm2_diffs = contr_avg - std_avg
    _, fm2_ci_low, fm2_ci_high = bootstrap_ci(fm2_diffs, n_bootstrap=n_bootstrap, seed=seed)

    assessment = assess_independence(effects["interaction_ratio"])

    return {
        "main_effects": effects,
        "fm1_test": {
            "mean_diff": fm1_diff,
            "p_value": fm1_pval,
            "cohens_d": fm1_d,
            "ci_95": (fm1_ci_low, fm1_ci_high),
        },
        "fm2_test": {
            "mean_diff": fm2_diff,
            "p_value": fm2_pval,
            "cohens_d": fm2_d,
            "ci_95": (fm2_ci_low, fm2_ci_high),
        },
        "independence_assessment": assessment,
        "cell_means": {
            "param_std": cell_param_std.mean().item(),
            "param_contr": cell_param_contr.mean().item(),
            "repr_std": cell_repr_std.mean().item(),
            "repr_contr": cell_repr_contr.mean().item(),
        },
    }
