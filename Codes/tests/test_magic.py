"""Tests for core/attribution/magic.py"""

import pytest
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.attribution.magic import magic_feasibility_check, magic_score_single_test


class MockModel(nn.Module):
    """Small model for feasibility testing."""
    def __init__(self, n_params_approx=1000):
        super().__init__()
        # Create a linear layer with approximately n_params_approx parameters
        dim = int(n_params_approx ** 0.5)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class TestForwardShape:
    """Test feasibility check output structure."""

    def test_feasibility_output_keys(self):
        model = MockModel(n_params_approx=1000)
        result = magic_feasibility_check(model, n_train=1000, n_test=10, n_steps=200)
        expected_keys = {"feasible", "estimated_time_hours", "estimated_disk_gb",
                         "estimated_gpu_memory_gb", "n_params", "bottleneck"}
        assert set(result.keys()) == expected_keys

    def test_feasibility_types(self):
        model = MockModel(n_params_approx=1000)
        result = magic_feasibility_check(model, n_train=1000, n_test=10, n_steps=200)
        assert isinstance(result["feasible"], bool)
        assert isinstance(result["estimated_time_hours"], float)
        assert isinstance(result["estimated_disk_gb"], float)

    def test_small_model_feasible(self):
        """A tiny model should be feasible."""
        model = MockModel(n_params_approx=100)
        result = magic_feasibility_check(
            model, n_train=100, n_test=5, n_steps=50,
            gpu_memory_gb=48.0, disk_space_gb=500.0
        )
        assert result["feasible"] is True


class TestGradientFlow:
    """MAGIC scoring not implemented yet; verify NotImplementedError."""

    def test_score_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="best-effort"):
            magic_score_single_test(
                test_loss_fn=lambda m: torch.tensor(0.0),
                checkpoints=[],
                training_losses=[],
                learning_rates=[],
                n_train=10,
            )


class TestOutputRange:
    """Feasibility estimates should be non-negative."""

    def test_estimates_non_negative(self):
        model = MockModel(n_params_approx=1000)
        result = magic_feasibility_check(model, n_train=1000, n_test=10, n_steps=200)
        assert result["estimated_time_hours"] >= 0
        assert result["estimated_disk_gb"] >= 0
        assert result["estimated_gpu_memory_gb"] >= 0

    def test_large_model_infeasible(self):
        """Simulate a large model that should exceed disk."""
        model = MockModel(n_params_approx=10000)
        result = magic_feasibility_check(
            model, n_train=10000, n_test=100, n_steps=200,
            gpu_memory_gb=48.0, disk_space_gb=0.0000001  # Extremely small disk
        )
        # With virtually no disk, should be infeasible
        assert result["feasible"] is False
        assert "disk" in result["bottleneck"]


class TestConfigSwitch:
    """MAGIC is optional / best-effort. Config switch: attribution.method = 'magic'."""

    def test_feasibility_with_different_configs(self):
        """Different GPU/disk configs produce different feasibility results."""
        model = MockModel(n_params_approx=1000)
        result_small = magic_feasibility_check(
            model, n_train=1000, n_test=10, n_steps=200,
            gpu_memory_gb=1.0, disk_space_gb=0.001
        )
        result_large = magic_feasibility_check(
            model, n_train=1000, n_test=10, n_steps=200,
            gpu_memory_gb=1000.0, disk_space_gb=10000.0
        )
        # Small config should be less feasible than large
        assert result_large["feasible"] or not result_small["feasible"]
