# Component: Contrastive Scoring Enhancement
# Source: research/method-design.md §5 Component B
# Ablation config key: attribution.scoring = "contrastive"

"""
Contrastive scoring: removes common-mode pre-training influence from attribution scores.

I_contrastive(z_test, z_train) = I(z_test, z_train; M_ft) - I(z_test, z_train; M_base)

Generic wrapper: works with any scoring method (RepSim, RepT, GradSim, TRAK, etc.).
Addresses FM2 (common influence contamination) by subtracting base-model scores.
"""

import torch
from typing import Callable, Optional


def contrastive_score(
    score_ft: torch.Tensor,
    score_base: torch.Tensor,
) -> torch.Tensor:
    """
    Compute contrastive attribution scores.

    I_contrastive = I(M_ft) - I(M_base)

    Args:
        score_ft: (n_test, n_train) attribution scores from fine-tuned model.
        score_base: (n_test, n_train) attribution scores from base (pre-FT) model.

    Returns:
        scores: (n_test, n_train) contrastive scores.
    """
    # score_ft: (n_test, n_train), score_base: (n_test, n_train)
    assert score_ft.shape == score_base.shape, (
        f"Shape mismatch: score_ft {score_ft.shape} vs score_base {score_base.shape}"
    )

    return score_ft - score_base


def contrastive_score_from_representations(
    h_test_ft: torch.Tensor,
    h_train_ft: torch.Tensor,
    h_test_base: torch.Tensor,
    h_train_base: torch.Tensor,
    scoring_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute contrastive scores from representations of both models.

    Convenience function that computes scores from both models and subtracts.

    Args:
        h_test_ft: (n_test, d) representations from fine-tuned model.
        h_train_ft: (n_train, d) representations from fine-tuned model.
        h_test_base: (n_test, d) representations from base model.
        h_train_base: (n_train, d) representations from base model.
        scoring_fn: Function that takes (h_test, h_train) -> (n_test, n_train) scores.
                     E.g., repsim_score or gradsim_score.

    Returns:
        scores: (n_test, n_train) contrastive scores.
    """
    score_ft = scoring_fn(h_test_ft, h_train_ft)
    score_base = scoring_fn(h_test_base, h_train_base)

    return contrastive_score(score_ft, score_base)


def compute_cmrr(
    score_standard: torch.Tensor,
    score_contrastive: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Common-Mode Rejection Ratio (CMRR).

    CMRR = |standard_score - contrastive_score| / (|standard_score| + eps)

    Quantifies how much of the standard attribution is common-mode contamination.
    High CMRR = high FM2 contamination in standard scores.

    Args:
        score_standard: (n_test, n_train) standard attribution scores.
        score_contrastive: (n_test, n_train) contrastive scores (standard - base).
        eps: Numerical stability constant.

    Returns:
        cmrr: Scalar mean CMRR across all test-train pairs.
    """
    diff = (score_standard - score_contrastive).abs()
    denom = score_standard.abs() + eps

    cmrr_per_pair = diff / denom  # (n_test, n_train)

    return cmrr_per_pair.mean()
