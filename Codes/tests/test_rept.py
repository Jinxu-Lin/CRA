"""Tests for core/attribution/rept.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.attribution.rept import (
    detect_phase_transition_layer,
    extract_rept_features,
    rept_score,
)


class TestForwardShape:
    """Verify output shapes match method-design.md spec."""

    def test_rept_feature_shape(self):
        hidden_rep = torch.randn(10, 64)
        hidden_grad = torch.randn(10, 64)
        phi = extract_rept_features(hidden_rep, hidden_grad)
        assert phi.shape == (10, 128), f"Expected (10, 128), got {phi.shape}"

    def test_rept_score_shape(self):
        phi_test = torch.randn(5, 128)
        phi_train = torch.randn(50, 128)
        scores = rept_score(phi_test, phi_train)
        assert scores.shape == (5, 50)

    def test_phase_transition_detection(self):
        # Create synthetic gradient norms with a clear transition at layer 3
        norms = torch.tensor([1.0, 1.1, 1.0, 0.5, 5.0, 4.8, 4.5])
        # Ratio at index 3->4 = 5.0/0.5 = 10.0 (largest ratio)
        l_star = detect_phase_transition_layer(norms)
        assert l_star == 4, f"Expected phase transition at layer 4, got {l_star}"

    def test_phase_transition_monotonic(self):
        # Monotonically increasing norms: max ratio at first big jump
        norms = torch.tensor([0.1, 0.5, 1.0, 2.0, 10.0])
        l_star = detect_phase_transition_layer(norms)
        # Ratios: 5.0, 2.0, 2.0, 5.0 -> tied, argmax picks first = index 0 -> layer 1
        # Actually: 0.5/0.1=5.0, 1.0/0.5=2.0, 2.0/1.0=2.0, 10.0/2.0=5.0 -> tied
        # argmax picks first occurrence: index 0 -> layer 1
        assert isinstance(l_star, int)
        assert 1 <= l_star <= 4


class TestGradientFlow:
    """Verify gradients flow through rept_score and feature extraction."""

    def test_gradient_through_score(self):
        phi_test = torch.randn(3, 64, requires_grad=True)
        phi_train = torch.randn(10, 64)
        scores = rept_score(phi_test, phi_train)
        scores.sum().backward()
        assert phi_test.grad is not None
        assert not torch.all(phi_test.grad == 0)

    def test_gradient_through_features(self):
        h_rep = torch.randn(5, 32, requires_grad=True)
        h_grad = torch.randn(5, 32, requires_grad=True)
        phi = extract_rept_features(h_rep, h_grad)
        loss = phi.sum()
        loss.backward()
        assert h_rep.grad is not None
        assert h_grad.grad is not None


class TestOutputRange:
    """Scores in [-1,1], no NaN/Inf."""

    def test_score_range(self):
        phi_test = torch.randn(5, 128)
        phi_train = torch.randn(50, 128)
        scores = rept_score(phi_test, phi_train)
        assert torch.isfinite(scores).all()
        assert scores.min() >= -1.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

    def test_identical_features(self):
        phi = torch.randn(5, 64)
        scores = rept_score(phi, phi)
        assert torch.allclose(scores.diag(), torch.ones(5), atol=1e-5)

    def test_feature_finite(self):
        h_rep = torch.randn(10, 32)
        h_grad = torch.randn(10, 32)
        phi = extract_rept_features(h_rep, h_grad)
        assert torch.isfinite(phi).all()

    def test_zero_gradient_safety(self):
        phi_test = torch.zeros(2, 64)
        phi_train = torch.randn(10, 64)
        scores = rept_score(phi_test, phi_train)
        assert torch.isfinite(scores).all()


class TestConfigSwitch:
    """RepT can be bypassed by using RepSim only (dropping gradient component)."""

    def test_feature_without_gradient(self):
        """If gradient is zero, phi degenerates to [h, 0] -> score ~ repsim."""
        h_rep = torch.randn(5, 32)
        h_grad = torch.zeros(5, 32)
        phi = extract_rept_features(h_rep, h_grad)
        # First half should be h_rep, second half zeros
        assert torch.allclose(phi[:, :32], h_rep)
        assert torch.allclose(phi[:, 32:], torch.zeros(5, 32))

    def test_shape_mismatch_raises(self):
        h_rep = torch.randn(5, 32)
        h_grad = torch.randn(5, 64)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            extract_rept_features(h_rep, h_grad)

    def test_phase_transition_min_layers(self):
        with pytest.raises(AssertionError):
            detect_phase_transition_layer(torch.tensor([1.0]))
