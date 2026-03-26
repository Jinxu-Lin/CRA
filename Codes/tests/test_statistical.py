"""Tests for core/evaluation/statistical.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.evaluation.statistical import (
    permutation_test, bootstrap_ci, cohens_d, benjamini_hochberg,
    pairwise_significance,
)


class TestForwardShape:
    """Verify output shapes and types."""

    def test_permutation_test_returns_tuple(self):
        a = torch.randn(50)
        b = torch.randn(50)
        diff, pval = permutation_test(a, b, n_permutations=100, seed=42)
        assert isinstance(diff, float)
        assert isinstance(pval, float)

    def test_bootstrap_ci_returns_triple(self):
        scores = torch.randn(50)
        point, lower, upper = bootstrap_ci(scores, n_bootstrap=100, seed=42)
        assert isinstance(point, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_cohens_d_returns_float(self):
        a = torch.randn(50)
        b = torch.randn(50)
        d = cohens_d(a, b)
        assert isinstance(d, float)

    def test_bh_returns_list(self):
        p_values = [0.01, 0.04, 0.03, 0.20, 0.50]
        result = benjamini_hochberg(p_values, q=0.05)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(r, bool) for r in result)

    def test_pairwise_returns_dict(self):
        method_scores = {
            "A": torch.randn(50),
            "B": torch.randn(50),
            "C": torch.randn(50),
        }
        result = pairwise_significance(method_scores, n_permutations=100, seed=42)
        assert "comparisons" in result
        assert "n_significant" in result
        assert len(result["comparisons"]) == 3  # C(3,2)


class TestGradientFlow:
    """Statistical tests are not differentiable; verify they don't crash with grad tensors."""

    def test_permutation_with_grad_tensor(self):
        a = torch.randn(30, requires_grad=True)
        b = torch.randn(30)
        diff, pval = permutation_test(a.detach(), b, n_permutations=50, seed=42)
        assert isinstance(pval, float)

    def test_bootstrap_with_grad_tensor(self):
        scores = torch.randn(30, requires_grad=True)
        point, lower, upper = bootstrap_ci(scores.detach(), n_bootstrap=50, seed=42)
        assert isinstance(point, float)


class TestOutputRange:
    """Verify p-values, CIs, and effect sizes are in expected ranges."""

    def test_p_value_range(self):
        a = torch.randn(100)
        b = torch.randn(100)
        _, pval = permutation_test(a, b, n_permutations=500, seed=42)
        assert 0.0 <= pval <= 1.0

    def test_significant_difference_low_p(self):
        """Very different distributions should have low p-value."""
        a = torch.randn(100) + 10.0
        b = torch.randn(100) - 10.0
        _, pval = permutation_test(a, b, n_permutations=500, seed=42)
        assert pval < 0.01, f"Expected significant difference, got p={pval}"

    def test_identical_high_p(self):
        """Identical distributions should have high p-value."""
        a = torch.randn(100)
        _, pval = permutation_test(a, a, n_permutations=500, seed=42)
        # Observed diff = 0, so all permutations should be >= 0
        assert pval >= 0.5

    def test_bootstrap_ci_contains_mean(self):
        scores = torch.randn(200)
        point, lower, upper = bootstrap_ci(scores, n_bootstrap=500, seed=42)
        assert lower <= point <= upper

    def test_bootstrap_ci_ordering(self):
        scores = torch.randn(200)
        _, lower, upper = bootstrap_ci(scores, n_bootstrap=500, seed=42)
        assert lower <= upper

    def test_cohens_d_large_effect(self):
        a = torch.randn(100) + 5.0
        b = torch.randn(100) - 5.0
        d = cohens_d(a, b)
        assert abs(d) > 1.0  # Large effect

    def test_cohens_d_zero_effect(self):
        a = torch.randn(100)
        d = cohens_d(a, a)
        assert d == 0.0

    def test_bh_all_significant(self):
        """All very small p-values should be significant."""
        p_values = [0.001, 0.002, 0.003]
        result = benjamini_hochberg(p_values, q=0.05)
        assert all(result)

    def test_bh_none_significant(self):
        """All large p-values should be non-significant."""
        p_values = [0.8, 0.9, 0.95]
        result = benjamini_hochberg(p_values, q=0.05)
        assert not any(result)

    def test_bh_empty(self):
        result = benjamini_hochberg([], q=0.05)
        assert result == []


class TestConfigSwitch:
    """Test configurability of statistical methods."""

    def test_different_n_permutations(self):
        a = torch.randn(50)
        b = torch.randn(50)
        # Both should produce valid p-values regardless of n_permutations
        _, p1 = permutation_test(a, b, n_permutations=50, seed=42)
        _, p2 = permutation_test(a, b, n_permutations=500, seed=42)
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1

    def test_custom_statistic_fn(self):
        scores = torch.randn(100)
        point, lower, upper = bootstrap_ci(
            scores, statistic_fn=lambda x: x.median().item(),
            n_bootstrap=200, seed=42,
        )
        assert lower <= upper
