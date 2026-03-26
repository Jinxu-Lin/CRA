# Component: Evaluation Metrics
# Source: research/experiment-design.md §5 (Metric Definition)
# Ablation config key: evaluation.metrics

"""
Evaluation metrics for training data attribution.

Primary: LDS (Linear Datamodeling Score) -- Spearman correlation between
predicted and actual model output changes when training subsets are removed.
Secondary: AUPRC, P@K, Recall@50, MRR.

These complement DATE-LM's built-in evaluation. Implemented here for
flexibility and custom analysis (e.g., per-sample LDS for 2x2 ablation).
"""

import torch
from typing import Optional, Dict, List, Tuple


def spearman_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        pred: (N,) predicted scores.
        target: (N,) actual scores.

    Returns:
        rho: Scalar Spearman correlation.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    assert pred.ndim == 1, f"Expected 1D, got {pred.ndim}D"

    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    # Rank transform
    pred_ranks = _rank(pred)
    target_ranks = _rank(target)

    # Pearson correlation of ranks
    return _pearson(pred_ranks, target_ranks)


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Compute ranks (1-indexed, average for ties)."""
    n = len(x)
    sorted_indices = x.argsort()
    sorted_x = x[sorted_indices]

    # Assign base ranks 1..n
    ranks = torch.zeros(n, dtype=torch.float, device=x.device)

    # Handle ties by averaging ranks within each group of equal values
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # All elements from i to j-1 have the same value
        avg_rank = (i + 1 + j) / 2.0  # average of ranks (i+1) through j
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j

    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient."""
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    numer = (x_centered * y_centered).sum()
    denom = (x_centered.pow(2).sum() * y_centered.pow(2).sum()).sqrt()
    if denom < 1e-10:
        return torch.tensor(0.0)
    return numer / denom


def lds(
    attribution_scores: torch.Tensor,
    actual_changes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Linear Datamodeling Score (LDS).

    LDS = Spearman correlation between attribution scores and actual model
    output changes when training subsets are removed.

    Args:
        attribution_scores: (N,) predicted influence scores for N training samples.
        actual_changes: (N,) actual model output changes when each sample is removed.

    Returns:
        lds_score: Scalar LDS value in [-1, 1].
    """
    return spearman_correlation(attribution_scores, actual_changes)


def auprc(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Area Under the Precision-Recall Curve.

    Used for toxicity filtering: detecting unsafe training samples.

    Args:
        scores: (N,) attribution scores (higher = more likely positive).
        labels: (N,) binary labels (1 = positive/unsafe, 0 = negative/safe).

    Returns:
        auprc_score: Scalar AUPRC value in [0, 1].
    """
    assert scores.shape == labels.shape
    assert scores.ndim == 1

    # Sort by score descending
    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices].float()

    n_positive = sorted_labels.sum()
    if n_positive == 0:
        return torch.tensor(0.0)

    # Compute precision and recall at each threshold
    tp_cumsum = sorted_labels.cumsum(dim=0)
    fp_cumsum = (1 - sorted_labels).cumsum(dim=0)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_positive

    # Compute AUPRC using trapezoidal rule
    # Prepend (recall=0, precision=1) for proper AUC computation
    recall_with_zero = torch.cat([torch.tensor([0.0]), recall])
    precision_with_one = torch.cat([torch.tensor([1.0]), precision])

    # Trapezoidal integration
    recall_diff = recall_with_zero[1:] - recall_with_zero[:-1]
    auprc_val = (precision_with_one[:-1] * recall_diff).sum()

    return auprc_val


def precision_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Compute Precision@K.

    Args:
        scores: (N,) attribution scores.
        labels: (N,) binary labels.
        k: Number of top-scored samples to consider.

    Returns:
        p_at_k: Scalar precision in [0, 1].
    """
    assert scores.shape == labels.shape
    assert k > 0

    k = min(k, len(scores))
    top_k_indices = scores.argsort(descending=True)[:k]
    top_k_labels = labels[top_k_indices].float()

    return top_k_labels.mean()


def recall_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 50,
) -> torch.Tensor:
    """
    Compute Recall@K.

    Args:
        scores: (N,) attribution scores.
        labels: (N,) binary labels.
        k: Number of top-scored samples to consider.

    Returns:
        recall: Scalar recall in [0, 1].
    """
    assert scores.shape == labels.shape

    n_positive = labels.float().sum()
    if n_positive == 0:
        return torch.tensor(0.0)

    k = min(k, len(scores))
    top_k_indices = scores.argsort(descending=True)[:k]
    tp = labels[top_k_indices].float().sum()

    return tp / n_positive


def mrr(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mean Reciprocal Rank.

    Used for factual attribution: rank of first relevant training sample.

    Args:
        scores: (N,) attribution scores.
        labels: (N,) binary labels.

    Returns:
        mrr_score: Scalar MRR value in [0, 1].
    """
    assert scores.shape == labels.shape

    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices].float()

    # Find position of first positive label (1-indexed)
    positive_positions = (sorted_labels > 0).nonzero(as_tuple=True)[0]

    if len(positive_positions) == 0:
        return torch.tensor(0.0)

    first_pos = positive_positions[0].float() + 1  # 1-indexed
    return 1.0 / first_pos


def compute_all_metrics(
    scores: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    actual_changes: Optional[torch.Tensor] = None,
    metric_names: Optional[List[str]] = None,
    k_values: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Compute all requested metrics.

    Args:
        scores: (N,) attribution scores.
        labels: (N,) binary labels (for AUPRC, P@K, Recall@K, MRR).
        actual_changes: (N,) actual model output changes (for LDS).
        metric_names: List of metrics to compute. Default: all applicable.
        k_values: Dict of K values for P@K and Recall@K. Default: {"pk": 10, "recall": 50}.

    Returns:
        Dict mapping metric name to value.
    """
    if k_values is None:
        k_values = {"pk": 10, "recall": 50}
    if metric_names is None:
        metric_names = ["lds", "auprc", "pk", "recall", "mrr"]

    results = {}

    if "lds" in metric_names and actual_changes is not None:
        results["lds"] = lds(scores, actual_changes).item()

    if "auprc" in metric_names and labels is not None:
        results["auprc"] = auprc(scores, labels).item()

    if "pk" in metric_names and labels is not None:
        results["pk"] = precision_at_k(scores, labels, k=k_values.get("pk", 10)).item()

    if "recall" in metric_names and labels is not None:
        results["recall"] = recall_at_k(scores, labels, k=k_values.get("recall", 50)).item()

    if "mrr" in metric_names and labels is not None:
        results["mrr"] = mrr(scores, labels).item()

    return results
