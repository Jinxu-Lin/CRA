"""Tests for core/evaluation/metrics.py"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.evaluation.metrics import (
    spearman_correlation, lds, auprc, precision_at_k,
    recall_at_k, mrr, compute_all_metrics,
)


class TestForwardShape:
    """Verify output shapes: all metrics return scalars."""

    def test_lds_scalar(self):
        scores = torch.randn(100)
        changes = torch.randn(100)
        result = lds(scores, changes)
        assert result.ndim == 0

    def test_auprc_scalar(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = auprc(scores, labels)
        assert result.ndim == 0

    def test_pk_scalar(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = precision_at_k(scores, labels, k=10)
        assert result.ndim == 0

    def test_recall_scalar(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = recall_at_k(scores, labels, k=50)
        assert result.ndim == 0

    def test_mrr_scalar(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = mrr(scores, labels)
        assert result.ndim == 0

    def test_compute_all_returns_dict(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        changes = torch.randn(100)
        result = compute_all_metrics(scores, labels=labels, actual_changes=changes)
        assert isinstance(result, dict)
        assert "lds" in result
        assert "auprc" in result


class TestGradientFlow:
    """Verify gradients flow (where applicable)."""

    def test_spearman_gradient(self):
        # Spearman uses ranks which are not differentiable in the standard sense,
        # but the operations should not crash with requires_grad
        scores = torch.randn(50, requires_grad=True)
        changes = torch.randn(50)
        result = lds(scores, changes)
        # LDS is rank-based, gradient may not be meaningful but should not error
        assert torch.isfinite(result)

    def test_auprc_no_crash(self):
        scores = torch.randn(50, requires_grad=True)
        labels = (torch.randn(50) > 0).long()
        result = auprc(scores, labels)
        assert torch.isfinite(result)


class TestOutputRange:
    """Metrics should be in expected ranges, no NaN/Inf."""

    def test_lds_range(self):
        scores = torch.randn(100)
        changes = torch.randn(100)
        result = lds(scores, changes)
        assert torch.isfinite(result)
        assert -1.0 - 1e-6 <= result.item() <= 1.0 + 1e-6

    def test_perfect_lds(self):
        """Identical rankings should give LDS = 1."""
        scores = torch.arange(100, dtype=torch.float)
        changes = torch.arange(100, dtype=torch.float)
        result = lds(scores, changes)
        assert abs(result.item() - 1.0) < 1e-5

    def test_auprc_range(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = auprc(scores, labels)
        assert torch.isfinite(result)
        assert 0.0 <= result.item() <= 1.0 + 1e-6

    def test_perfect_auprc(self):
        """Perfect scorer should have high AUPRC."""
        labels = torch.cat([torch.ones(10), torch.zeros(90)]).long()
        scores = torch.cat([torch.ones(10) * 10, torch.ones(90) * -10])
        result = auprc(scores, labels)
        assert result.item() > 0.99

    def test_pk_range(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = precision_at_k(scores, labels, k=10)
        assert 0.0 <= result.item() <= 1.0

    def test_recall_range(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = recall_at_k(scores, labels, k=50)
        assert 0.0 <= result.item() <= 1.0

    def test_mrr_range(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = mrr(scores, labels)
        assert 0.0 <= result.item() <= 1.0

    def test_mrr_perfect(self):
        """If top-scored sample is positive, MRR = 1."""
        scores = torch.tensor([10.0, 1.0, 2.0, 3.0])
        labels = torch.tensor([1, 0, 0, 0])
        result = mrr(scores, labels)
        assert abs(result.item() - 1.0) < 1e-5

    def test_no_positives(self):
        """No positive labels should return 0 for retrieval metrics."""
        scores = torch.randn(50)
        labels = torch.zeros(50).long()
        assert auprc(scores, labels).item() == 0.0
        assert recall_at_k(scores, labels).item() == 0.0
        assert mrr(scores, labels).item() == 0.0


class TestConfigSwitch:
    """compute_all_metrics should respect metric_names filter."""

    def test_subset_metrics(self):
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        changes = torch.randn(100)
        result = compute_all_metrics(
            scores, labels=labels, actual_changes=changes,
            metric_names=["lds", "auprc"]
        )
        assert "lds" in result
        assert "auprc" in result
        assert "pk" not in result
        assert "mrr" not in result

    def test_no_labels(self):
        """Without labels, only LDS should be computed."""
        scores = torch.randn(100)
        changes = torch.randn(100)
        result = compute_all_metrics(scores, actual_changes=changes)
        assert "lds" in result
        assert "auprc" not in result

    def test_no_changes(self):
        """Without actual_changes, LDS should be skipped."""
        scores = torch.randn(100)
        labels = (torch.randn(100) > 0).long()
        result = compute_all_metrics(scores, labels=labels)
        assert "lds" not in result
        assert "auprc" in result
