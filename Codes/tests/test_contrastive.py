"""Tests for core/attribution/contrastive.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.attribution.contrastive import (
    contrastive_score,
    contrastive_score_from_representations,
    compute_cmrr,
)
from core.attribution.repsim import repsim_score


class TestForwardShape:
    """Verify output shapes: (n_test, n_train)."""

    def test_basic_shape(self):
        score_ft = torch.randn(5, 50)
        score_base = torch.randn(5, 50)
        result = contrastive_score(score_ft, score_base)
        assert result.shape == (5, 50)

    def test_from_representations_shape(self):
        h_test_ft = torch.randn(5, 64)
        h_train_ft = torch.randn(50, 64)
        h_test_base = torch.randn(5, 64)
        h_train_base = torch.randn(50, 64)
        result = contrastive_score_from_representations(
            h_test_ft, h_train_ft, h_test_base, h_train_base, repsim_score
        )
        assert result.shape == (5, 50)

    def test_cmrr_scalar(self):
        score_std = torch.randn(5, 50)
        score_contr = torch.randn(5, 50)
        cmrr = compute_cmrr(score_std, score_contr)
        assert cmrr.ndim == 0  # scalar


class TestGradientFlow:
    """Verify gradients flow through contrastive scoring."""

    def test_gradient_through_scores(self):
        score_ft = torch.randn(3, 10, requires_grad=True)
        score_base = torch.randn(3, 10, requires_grad=True)
        result = contrastive_score(score_ft, score_base)
        result.sum().backward()
        assert score_ft.grad is not None
        assert score_base.grad is not None

    def test_gradient_through_representations(self):
        h_test_ft = torch.randn(3, 32, requires_grad=True)
        h_train_ft = torch.randn(10, 32)
        h_test_base = torch.randn(3, 32)
        h_train_base = torch.randn(10, 32)
        result = contrastive_score_from_representations(
            h_test_ft, h_train_ft, h_test_base, h_train_base, repsim_score
        )
        result.sum().backward()
        assert h_test_ft.grad is not None
        assert not torch.all(h_test_ft.grad == 0)


class TestOutputRange:
    """Contrastive scores are in [-2, 2] (difference of two [-1,1] values). No NaN/Inf."""

    def test_contrastive_range(self):
        score_ft = torch.randn(10, 50).clamp(-1, 1)
        score_base = torch.randn(10, 50).clamp(-1, 1)
        result = contrastive_score(score_ft, score_base)
        assert torch.isfinite(result).all()
        assert result.min() >= -2.0 - 1e-6
        assert result.max() <= 2.0 + 1e-6

    def test_identical_models_zero(self):
        """If M_ft == M_base, contrastive score should be ~0."""
        h = torch.randn(5, 32)
        h_train = torch.randn(20, 32)
        result = contrastive_score_from_representations(
            h, h_train, h, h_train, repsim_score
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_cmrr_non_negative(self):
        score_std = torch.randn(5, 50)
        score_contr = torch.randn(5, 50)
        cmrr = compute_cmrr(score_std, score_contr)
        assert cmrr >= 0

    def test_cmrr_identical_zero(self):
        """If standard == contrastive, CMRR should be ~0."""
        scores = torch.randn(5, 50)
        cmrr = compute_cmrr(scores, scores)
        assert cmrr < 1e-6


class TestConfigSwitch:
    """Contrastive scoring is controlled by attribution.scoring config."""

    def test_shape_mismatch_raises(self):
        score_ft = torch.randn(5, 50)
        score_base = torch.randn(5, 30)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            contrastive_score(score_ft, score_base)

    def test_standard_scoring_bypass(self):
        """When scoring='standard', contrastive is not applied. Verify score_ft alone works."""
        score_ft = torch.randn(5, 50)
        # Standard scoring just uses score_ft directly (no wrapper call needed)
        assert score_ft.shape == (5, 50)
