"""Tests for core/attribution/gradsim.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.attribution.gradsim import gradsim_score, gradsim_score_batched


class TestForwardShape:
    """Verify output shape: (n_test, n_train)."""

    def test_basic_shape(self):
        g_test = torch.randn(5, 128)
        g_train = torch.randn(50, 128)
        scores = gradsim_score(g_test, g_train)
        assert scores.shape == (5, 50)

    def test_single_test(self):
        g_test = torch.randn(1, 256)
        g_train = torch.randn(100, 256)
        scores = gradsim_score(g_test, g_train)
        assert scores.shape == (1, 100)

    def test_batched_matches_full(self):
        g_test = torch.randn(3, 128)
        g_train = torch.randn(100, 128)
        scores_full = gradsim_score(g_test, g_train)
        scores_batched = gradsim_score_batched(g_test, g_train, batch_size=30)
        assert scores_batched.shape == (3, 100)
        assert torch.allclose(scores_full, scores_batched, atol=1e-6)


class TestGradientFlow:
    """Verify gradients flow through gradsim_score."""

    def test_gradient_through_test(self):
        g_test = torch.randn(3, 64, requires_grad=True)
        g_train = torch.randn(10, 64)
        scores = gradsim_score(g_test, g_train)
        scores.sum().backward()
        assert g_test.grad is not None
        assert not torch.all(g_test.grad == 0)

    def test_gradient_through_train(self):
        g_test = torch.randn(3, 64)
        g_train = torch.randn(10, 64, requires_grad=True)
        scores = gradsim_score(g_test, g_train)
        scores.sum().backward()
        assert g_train.grad is not None
        assert not torch.all(g_train.grad == 0)


class TestOutputRange:
    """Cosine similarity must be in [-1, 1], no NaN/Inf."""

    def test_range(self):
        g_test = torch.randn(5, 128)
        g_train = torch.randn(50, 128)
        scores = gradsim_score(g_test, g_train)
        assert torch.isfinite(scores).all()
        assert scores.min() >= -1.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

    def test_identical_gradients(self):
        g = torch.randn(5, 64)
        scores = gradsim_score(g, g)
        assert torch.allclose(scores.diag(), torch.ones(5), atol=1e-5)

    def test_zero_gradient_safety(self):
        g_test = torch.zeros(2, 64)
        g_train = torch.randn(10, 64)
        scores = gradsim_score(g_test, g_train)
        assert torch.isfinite(scores).all()


class TestConfigSwitch:
    """Validate input constraints and dimension mismatch handling."""

    def test_dimension_mismatch_raises(self):
        g_test = torch.randn(3, 64)
        g_train = torch.randn(10, 128)
        with pytest.raises(AssertionError, match="Gradient dimension mismatch"):
            gradsim_score(g_test, g_train)

    def test_wrong_ndim_raises(self):
        g_test = torch.randn(3, 4, 64)
        g_train = torch.randn(10, 64)
        with pytest.raises(AssertionError):
            gradsim_score(g_test, g_train)
