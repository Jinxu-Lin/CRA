"""Tests for core/attribution/repsim.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.attribution.repsim import repsim_score, repsim_score_batched


class TestForwardShape:
    """Verify output shapes match method-design.md spec: (n_test, n_train)."""

    def test_basic_shape(self):
        h_test = torch.randn(10, 64)
        h_train = torch.randn(100, 64)
        scores = repsim_score(h_test, h_train)
        assert scores.shape == (10, 100)

    def test_single_test(self):
        h_test = torch.randn(1, 32)
        h_train = torch.randn(50, 32)
        scores = repsim_score(h_test, h_train)
        assert scores.shape == (1, 50)

    def test_batched_matches_full(self):
        h_test = torch.randn(5, 64)
        h_train = torch.randn(200, 64)
        scores_full = repsim_score(h_test, h_train)
        scores_batched = repsim_score_batched(h_test, h_train, batch_size=50)
        assert scores_batched.shape == (5, 200)
        assert torch.allclose(scores_full, scores_batched, atol=1e-6)


class TestGradientFlow:
    """Verify gradients flow through repsim_score."""

    def test_gradient_through_test(self):
        h_test = torch.randn(5, 32, requires_grad=True)
        h_train = torch.randn(20, 32)
        scores = repsim_score(h_test, h_train)
        loss = scores.sum()
        loss.backward()
        assert h_test.grad is not None
        assert not torch.all(h_test.grad == 0)

    def test_gradient_through_train(self):
        h_test = torch.randn(5, 32)
        h_train = torch.randn(20, 32, requires_grad=True)
        scores = repsim_score(h_test, h_train)
        loss = scores.sum()
        loss.backward()
        assert h_train.grad is not None
        assert not torch.all(h_train.grad == 0)


class TestOutputRange:
    """Cosine similarity must be in [-1, 1], no NaN/Inf."""

    def test_range(self):
        h_test = torch.randn(10, 64)
        h_train = torch.randn(100, 64)
        scores = repsim_score(h_test, h_train)
        assert torch.isfinite(scores).all(), "Scores contain NaN or Inf"
        assert scores.min() >= -1.0 - 1e-6, f"Min score {scores.min()} < -1"
        assert scores.max() <= 1.0 + 1e-6, f"Max score {scores.max()} > 1"

    def test_identical_vectors(self):
        """Identical vectors should have cosine similarity ~1."""
        h = torch.randn(5, 32)
        scores = repsim_score(h, h)
        diag = scores.diag()
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_zero_vector_safety(self):
        """Zero vectors should not produce NaN (eps protection)."""
        h_test = torch.zeros(2, 32)
        h_train = torch.randn(10, 32)
        scores = repsim_score(h_test, h_train)
        assert torch.isfinite(scores).all(), "Zero vector produced NaN/Inf"


class TestConfigSwitch:
    """RepSim is always-on when attribution.method='repsim'. Test dimension mismatch errors."""

    def test_dimension_mismatch_raises(self):
        h_test = torch.randn(5, 32)
        h_train = torch.randn(10, 64)
        with pytest.raises(AssertionError, match="Dimension mismatch"):
            repsim_score(h_test, h_train)

    def test_wrong_ndim_raises(self):
        h_test = torch.randn(5, 3, 32)
        h_train = torch.randn(10, 32)
        with pytest.raises(AssertionError):
            repsim_score(h_test, h_train)

    def test_different_eps(self):
        """Different eps values should not change results for normal vectors."""
        h_test = torch.randn(5, 32)
        h_train = torch.randn(10, 32)
        s1 = repsim_score(h_test, h_train, eps=1e-8)
        s2 = repsim_score(h_test, h_train, eps=1e-12)
        assert torch.allclose(s1, s2, atol=1e-5)
