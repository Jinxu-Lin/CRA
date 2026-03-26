"""Tests for core/data/representation.py"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --- Minimal mock model for testing ---

class MockTransformerConfig:
    num_hidden_layers = 4
    hidden_size = 32


class MockModelOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class MockTransformerModel(nn.Module):
    """Minimal model that produces hidden_states when output_hidden_states=True."""

    def __init__(self, n_layers=4, d_model=32, vocab_size=100):
        super().__init__()
        self.config = MockTransformerConfig()
        self.config.num_hidden_layers = n_layers
        self.config.hidden_size = d_model
        self.n_layers = n_layers
        self.d_model = d_model
        # Simple embedding + linear layers to produce hidden states
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embedding(input_ids)  # (B, seq_len, d_model)
        hidden_states = [x]  # embedding output
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        return MockModelOutput(hidden_states=tuple(hidden_states))


def _make_dataloader(batch_size=4, seq_len=16, n_samples=12, vocab_size=100, with_mask=True):
    """Create a simple dataloader with random token IDs."""
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
    if with_mask:
        # Random attention masks (at least 1 token per sample)
        attention_mask = torch.ones(n_samples, seq_len)
        for i in range(n_samples):
            real_len = torch.randint(1, seq_len + 1, (1,)).item()
            attention_mask[i, real_len:] = 0
        dataset = TensorDataset(input_ids, attention_mask)
        def collate_fn(batch):
            ids, masks = zip(*batch)
            return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks)}
    else:
        dataset = TensorDataset(input_ids)
        def collate_fn(batch):
            ids = [b[0] for b in batch]
            return {"input_ids": torch.stack(ids)}
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


# --- Import module under test ---
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.data.representation import (
    resolve_layer_index,
    aggregate_tokens,
    extract_representations,
    extract_all_layer_representations,
)


class TestForwardShape:
    """test_forward_shape: Verify output shapes match method-design.md spec."""

    def test_extract_shape_last_token(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=12)
        reps = extract_representations(model, dl, layer="middle", aggregation="last_token")
        assert reps.shape == (12, 32), f"Expected (12, 32), got {reps.shape}"

    def test_extract_shape_mean_pool(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=12)
        reps = extract_representations(model, dl, layer="last", aggregation="mean_pool")
        assert reps.shape == (12, 32), f"Expected (12, 32), got {reps.shape}"

    def test_extract_shape_int_layer(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=8)
        reps = extract_representations(model, dl, layer=0, aggregation="last_token")
        assert reps.shape == (8, 32)

    def test_all_layer_extraction_shape(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=8)
        all_reps = extract_all_layer_representations(model, dl, aggregation="last_token")
        assert len(all_reps) == 4
        for reps in all_reps:
            assert reps.shape == (8, 32)

    def test_no_mask(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=8, with_mask=False)
        reps = extract_representations(model, dl, layer="middle", aggregation="last_token")
        assert reps.shape == (8, 32)


class TestGradientFlow:
    """test_gradient_flow: Not directly applicable (extraction is inference-only).
    Instead verify that the embedding gradients propagate if we remove @no_grad."""

    def test_aggregate_tokens_gradient(self):
        h = torch.randn(2, 5, 32, requires_grad=True)
        mask = torch.ones(2, 5)
        mask[0, 3:] = 0
        out = aggregate_tokens(h, mask, "last_token")
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert not torch.all(h.grad == 0)

    def test_aggregate_mean_pool_gradient(self):
        h = torch.randn(2, 5, 32, requires_grad=True)
        mask = torch.ones(2, 5)
        out = aggregate_tokens(h, mask, "mean_pool")
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert not torch.all(h.grad == 0)


class TestOutputRange:
    """test_output_range: Outputs should be finite, no NaN/Inf."""

    def test_no_nan_inf(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=12)
        reps = extract_representations(model, dl, layer="middle", aggregation="last_token")
        assert torch.isfinite(reps).all(), "Output contains NaN or Inf"

    def test_reasonable_range(self):
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=12)
        reps = extract_representations(model, dl, layer="last", aggregation="mean_pool")
        assert torch.isfinite(reps).all()
        # Values should not be astronomically large
        assert reps.abs().max() < 1e6, f"Max absolute value too large: {reps.abs().max()}"


class TestConfigSwitch:
    """test_config_switch: Layer resolution and aggregation switching work correctly."""

    def test_layer_resolution_middle(self):
        assert resolve_layer_index("middle", 24) == 12
        assert resolve_layer_index("middle", 4) == 2

    def test_layer_resolution_last(self):
        assert resolve_layer_index("last", 24) == 23
        assert resolve_layer_index("last", 4) == 3

    def test_layer_resolution_int(self):
        assert resolve_layer_index(0, 24) == 0
        assert resolve_layer_index(10, 24) == 10

    def test_invalid_layer(self):
        with pytest.raises(ValueError):
            resolve_layer_index(25, 24)
        with pytest.raises(ValueError):
            resolve_layer_index("unknown", 24)

    def test_aggregation_switch(self):
        """Both aggregation methods produce valid outputs of same shape."""
        model = MockTransformerModel(n_layers=4, d_model=32)
        dl = _make_dataloader(batch_size=4, n_samples=8)
        reps_last = extract_representations(model, dl, layer=1, aggregation="last_token")
        reps_mean = extract_representations(model, dl, layer=1, aggregation="mean_pool")
        assert reps_last.shape == reps_mean.shape
