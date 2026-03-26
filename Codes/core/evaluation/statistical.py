# Component: Statistical Analysis Suite
# Source: research/experiment-design.md §5.4 (Statistical Significance Plan)
# Ablation config key: N/A (analysis utility)

"""
Statistical analysis tools for CRA experiments.

- Permutation test: per-sample significance testing
- Bootstrap CI: confidence intervals for all metrics
- BH-FDR: multiple comparison correction
- Cohen's d: effect size for pairwise comparisons
"""

import torch
from typing import Tuple, List, Optional


def permutation_test(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Two-sided permutation test for paired differences.

    Tests H0: mean(scores_a) == mean(scores_b).

    Args:
        scores_a: (N,) per-sample scores for method A.
        scores_b: (N,) per-sample scores for method B.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        (observed_diff, p_value): Observed mean difference and two-sided p-value.
    """
    assert scores_a.shape == scores_b.shape, f"Shape mismatch: {scores_a.shape} vs {scores_b.shape}"
    assert scores_a.ndim == 1

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    diffs = scores_a - scores_b  # (N,)
    observed_diff = diffs.mean().item()
    n = len(diffs)

    # Under H0, signs of differences are random
    count_extreme = 0
    for _ in range(n_permutations):
        signs = (torch.randint(0, 2, (n,), generator=generator).float() * 2 - 1)
        perm_diff = (diffs * signs).mean().item()
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 for continuity correction

    return observed_diff, p_value


def bootstrap_ci(
    scores: torch.Tensor,
    statistic_fn=None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        scores: (N,) or (N, K) sample scores. If 2D, statistic_fn must handle it.
        statistic_fn: Function mapping scores -> scalar. Default: mean.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        seed: Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper): Point estimate and CI bounds.
    """
    if statistic_fn is None:
        statistic_fn = lambda x: x.mean().item()

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    n = scores.shape[0]
    point_estimate = statistic_fn(scores)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = torch.randint(0, n, (n,), generator=generator)
        resample = scores[indices]
        stat = statistic_fn(resample)
        bootstrap_stats.append(stat)

    bootstrap_stats = sorted(bootstrap_stats)
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return point_estimate, bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]


def cohens_d(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    d = mean(A - B) / std(A - B)

    Args:
        scores_a: (N,) scores for condition A.
        scores_b: (N,) scores for condition B.

    Returns:
        d: Cohen's d effect size.
    """
    assert scores_a.shape == scores_b.shape
    diffs = scores_a - scores_b
    mean_diff = diffs.mean().item()
    std_diff = diffs.std().item()

    if std_diff < 1e-10:
        return 0.0 if abs(mean_diff) < 1e-10 else float("inf")

    return mean_diff / std_diff


def benjamini_hochberg(
    p_values: List[float],
    q: float = 0.05,
) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests.
        q: Target FDR level.

    Returns:
        List of bools: True if the corresponding test is significant after correction.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values and track original indices
    indexed_pvals = sorted(enumerate(p_values), key=lambda x: x[1])

    significant = [False] * m

    # BH procedure: find largest k such that p_(k) <= k/m * q
    max_k = 0
    for rank, (orig_idx, pval) in enumerate(indexed_pvals, start=1):
        threshold = rank / m * q
        if pval <= threshold:
            max_k = rank

    # All tests with rank <= max_k are significant
    for rank, (orig_idx, pval) in enumerate(indexed_pvals, start=1):
        if rank <= max_k:
            significant[orig_idx] = True

    return significant


def pairwise_significance(
    method_scores: dict,
    n_permutations: int = 10000,
    q: float = 0.05,
    seed: Optional[int] = None,
) -> dict:
    """
    Run pairwise permutation tests with BH-FDR correction.

    Args:
        method_scores: Dict mapping method name -> (N,) per-sample scores.
        n_permutations: Number of permutations per test.
        q: Target FDR level.
        seed: Random seed.

    Returns:
        Dict with keys:
        - "comparisons": List of (method_a, method_b, diff, p_value, significant, cohens_d)
        - "n_significant": Number of significant comparisons after correction.
    """
    methods = list(method_scores.keys())
    comparisons = []
    p_values = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            diff, pval = permutation_test(
                method_scores[a], method_scores[b],
                n_permutations=n_permutations, seed=seed,
            )
            d = cohens_d(method_scores[a], method_scores[b])
            comparisons.append((a, b, diff, pval, d))
            p_values.append(pval)

    significant = benjamini_hochberg(p_values, q=q)

    results = []
    for comp, sig in zip(comparisons, significant):
        a, b, diff, pval, d = comp
        results.append({
            "method_a": a,
            "method_b": b,
            "mean_diff": diff,
            "p_value": pval,
            "significant": sig,
            "cohens_d": d,
        })

    return {
        "comparisons": results,
        "n_significant": sum(significant),
    }
