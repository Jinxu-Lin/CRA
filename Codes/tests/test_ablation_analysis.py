"""Tests for core/evaluation/ablation_analysis.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.evaluation.ablation_analysis import (
    compute_main_effects, assess_independence, compute_cmrr,
    full_ablation_analysis,
)


class TestForwardShape:
    """Verify output structure."""

    def test_main_effects_keys(self):
        n = 50
        result = compute_main_effects(
            torch.randn(n), torch.randn(n), torch.randn(n), torch.randn(n)
        )
        assert "fm1_effect" in result
        assert "fm2_effect" in result
        assert "interaction" in result
        assert "interaction_ratio" in result

    def test_full_analysis_keys(self):
        n = 50
        result = full_ablation_analysis(
            torch.randn(n), torch.randn(n), torch.randn(n), torch.randn(n),
            n_permutations=50, n_bootstrap=50, seed=42,
        )
        assert "main_effects" in result
        assert "fm1_test" in result
        assert "fm2_test" in result
        assert "independence_assessment" in result
        assert "cell_means" in result

    def test_cmrr_returns_float(self):
        result = compute_cmrr(torch.randn(100), torch.randn(100))
        assert isinstance(result, float)


class TestGradientFlow:
    """Statistical analysis is not differentiable; verify no crashes."""

    def test_main_effects_no_crash(self):
        n = 30
        result = compute_main_effects(
            torch.randn(n), torch.randn(n), torch.randn(n), torch.randn(n)
        )
        assert isinstance(result["fm1_effect"], float)


class TestOutputRange:
    """Verify effects and assessments are sensible."""

    def test_known_fm1_effect(self):
        """If repr >> param, FM1 effect should be positive."""
        n = 100
        param_std = torch.randn(n)
        param_contr = torch.randn(n)
        repr_std = torch.randn(n) + 5.0  # Much higher
        repr_contr = torch.randn(n) + 5.0
        result = compute_main_effects(param_std, param_contr, repr_std, repr_contr)
        assert result["fm1_effect"] > 0

    def test_known_fm2_effect(self):
        """If contrastive >> standard, FM2 effect should be positive."""
        n = 100
        param_std = torch.randn(n)
        param_contr = torch.randn(n) + 5.0
        repr_std = torch.randn(n)
        repr_contr = torch.randn(n) + 5.0
        result = compute_main_effects(param_std, param_contr, repr_std, repr_contr)
        assert result["fm2_effect"] > 0

    def test_no_interaction(self):
        """Additive effects should produce small interaction."""
        n = 1000
        base = torch.randn(n)
        fm1_boost = 3.0
        fm2_boost = 2.0
        param_std = base
        param_contr = base + fm2_boost
        repr_std = base + fm1_boost
        repr_contr = base + fm1_boost + fm2_boost
        result = compute_main_effects(param_std, param_contr, repr_std, repr_contr)
        assert abs(result["interaction"]) < 0.5  # Near zero

    def test_independence_assessment_categories(self):
        assert assess_independence(0.05) == "clean_independence"
        assert assess_independence(0.15) == "moderate_interaction"
        assert assess_independence(0.50) == "strong_interaction"

    def test_cmrr_identical_zero(self):
        scores = torch.randn(100)
        cmrr = compute_cmrr(scores, scores)
        assert cmrr < 1e-6

    def test_cmrr_non_negative(self):
        cmrr = compute_cmrr(torch.randn(100), torch.randn(100))
        assert cmrr >= 0

    def test_full_analysis_significance(self):
        """Strong effects should produce significant p-values."""
        n = 100
        param_std = torch.randn(n)
        param_contr = torch.randn(n)
        repr_std = torch.randn(n) + 10.0
        repr_contr = torch.randn(n) + 10.0
        result = full_ablation_analysis(
            param_std, param_contr, repr_std, repr_contr,
            n_permutations=200, n_bootstrap=200, seed=42,
        )
        assert result["fm1_test"]["p_value"] < 0.05


class TestConfigSwitch:
    """Test different analysis configurations."""

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_main_effects(
                torch.randn(50), torch.randn(50), torch.randn(30), torch.randn(50)
            )

    def test_different_seeds_stable(self):
        """Results should be deterministic with fixed seed."""
        n = 50
        args = (torch.randn(n), torch.randn(n), torch.randn(n), torch.randn(n))
        r1 = full_ablation_analysis(*args, n_permutations=100, n_bootstrap=100, seed=42)
        r2 = full_ablation_analysis(*args, n_permutations=100, n_bootstrap=100, seed=42)
        assert r1["main_effects"] == r2["main_effects"]
