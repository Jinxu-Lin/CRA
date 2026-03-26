"""
Integration test: Full CRA pipeline end-to-end.

Tests:
1. Complete forward + backward with all components
2. All attribution methods produce valid scores
3. Contrastive scoring works with all base methods
4. Evaluation metrics work with attribution scores
5. Statistical analysis works with metric results
6. 2x2 ablation analysis produces interpretable output
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data.representation import extract_representations, resolve_layer_index
from core.attribution.repsim import repsim_score
from core.attribution.gradsim import gradsim_score
from core.attribution.rept import extract_rept_features, rept_score, detect_phase_transition_layer
from core.attribution.contrastive import contrastive_score, contrastive_score_from_representations, compute_cmrr
from core.attribution.magic import magic_feasibility_check
from core.evaluation.metrics import lds, auprc, precision_at_k, recall_at_k, mrr, compute_all_metrics
from core.evaluation.statistical import permutation_test, bootstrap_ci, cohens_d, benjamini_hochberg
from core.evaluation.ablation_analysis import compute_main_effects, full_ablation_analysis


# --- Mock model (same as test_representation.py) ---

class MockTransformerConfig:
    num_hidden_layers = 6
    hidden_size = 64


class MockModelOutput:
    def __init__(self, hidden_states, logits=None):
        self.hidden_states = hidden_states
        self.logits = logits


class MockTransformerModel(nn.Module):
    def __init__(self, n_layers=6, d_model=64, vocab_size=100):
        super().__init__()
        self.config = MockTransformerConfig()
        self.config.num_hidden_layers = n_layers
        self.config.hidden_size = d_model
        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embedding(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        logits = self.head(x)
        return MockModelOutput(
            hidden_states=tuple(hidden_states),
            logits=logits,
        )


def _make_dataloader(batch_size=4, seq_len=16, n_samples=20, vocab_size=100):
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
    attention_mask = torch.ones(n_samples, seq_len)
    dataset = TensorDataset(input_ids, attention_mask)
    def collate_fn(batch):
        ids, masks = zip(*batch)
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks)}
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


# --- Integration Tests ---

class TestFullPipeline:
    """4a: Complete forward + score + evaluate pipeline."""

    def test_repsim_full_pipeline(self):
        """RepSim: extract reps -> score -> evaluate."""
        model = MockTransformerModel()
        dl_train = _make_dataloader(n_samples=50)
        dl_test = _make_dataloader(n_samples=10)

        h_train = extract_representations(model, dl_train, layer="middle")
        h_test = extract_representations(model, dl_test, layer="middle")

        assert h_train.shape == (50, 64)
        assert h_test.shape == (10, 64)

        scores = repsim_score(h_test, h_train)
        assert scores.shape == (10, 50)
        assert torch.isfinite(scores).all()
        assert scores.min() >= -1.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

        # Evaluate per test sample (sum scores as proxy for attribution)
        per_train_scores = scores.mean(dim=0)  # (50,) average influence of each train sample
        labels = (torch.randn(50) > 0.5).long()
        actual_changes = torch.randn(50)

        results = compute_all_metrics(per_train_scores, labels=labels, actual_changes=actual_changes)
        assert "lds" in results
        assert "auprc" in results
        assert -1.0 <= results["lds"] <= 1.0
        assert 0.0 <= results["auprc"] <= 1.0

    def test_contrastive_repsim_pipeline(self):
        """Contrastive RepSim: two models -> contrastive scores -> evaluate."""
        model_ft = MockTransformerModel()
        model_base = MockTransformerModel()
        dl_train = _make_dataloader(n_samples=30)
        dl_test = _make_dataloader(n_samples=5)

        h_train_ft = extract_representations(model_ft, dl_train, layer="middle")
        h_test_ft = extract_representations(model_ft, dl_test, layer="middle")
        h_train_base = extract_representations(model_base, dl_train, layer="middle")
        h_test_base = extract_representations(model_base, dl_test, layer="middle")

        scores = contrastive_score_from_representations(
            h_test_ft, h_train_ft, h_test_base, h_train_base, repsim_score
        )
        assert scores.shape == (5, 30)
        assert torch.isfinite(scores).all()

    def test_rept_feature_pipeline(self):
        """RepT: features -> score -> finite output."""
        d = 64
        h_rep_test = torch.randn(5, d)
        h_grad_test = torch.randn(5, d)
        h_rep_train = torch.randn(30, d)
        h_grad_train = torch.randn(30, d)

        phi_test = extract_rept_features(h_rep_test, h_grad_test)
        phi_train = extract_rept_features(h_rep_train, h_grad_train)

        scores = rept_score(phi_test, phi_train)
        assert scores.shape == (5, 30)
        assert torch.isfinite(scores).all()

    def test_gradsim_pipeline(self):
        """GradSim: gradients -> score -> finite output."""
        g_test = torch.randn(5, 256)
        g_train = torch.randn(30, 256)
        scores = gradsim_score(g_test, g_train)
        assert scores.shape == (5, 30)
        assert torch.isfinite(scores).all()


class TestConfigAllOff:
    """4b: Verify each method works independently and can be swapped."""

    def test_method_switching(self):
        """Simulate config-driven method selection."""
        n_test, n_train, d = 5, 30, 64

        methods = {
            "repsim": lambda: repsim_score(torch.randn(n_test, d), torch.randn(n_train, d)),
            "gradsim": lambda: gradsim_score(torch.randn(n_test, d), torch.randn(n_train, d)),
            "rept": lambda: rept_score(torch.randn(n_test, 2*d), torch.randn(n_train, 2*d)),
        }

        for name, fn in methods.items():
            scores = fn()
            assert scores.shape == (n_test, n_train), f"{name} failed shape check"
            assert torch.isfinite(scores).all(), f"{name} produced NaN/Inf"

    def test_scoring_switching(self):
        """Simulate config-driven scoring mode switch (standard vs contrastive)."""
        n_test, n_train, d = 5, 30, 64
        h_test = torch.randn(n_test, d)
        h_train = torch.randn(n_train, d)

        # Standard
        scores_std = repsim_score(h_test, h_train)
        assert scores_std.shape == (n_test, n_train)

        # Contrastive
        h_test_base = torch.randn(n_test, d)
        h_train_base = torch.randn(n_train, d)
        scores_contr = contrastive_score_from_representations(
            h_test, h_train, h_test_base, h_train_base, repsim_score
        )
        assert scores_contr.shape == (n_test, n_train)

    def test_layer_switching(self):
        """All layer specs produce valid representations."""
        model = MockTransformerModel(n_layers=6, d_model=64)
        dl = _make_dataloader(n_samples=8)

        for layer_spec in [0, 2, "middle", "last"]:
            reps = extract_representations(model, dl, layer=layer_spec)
            assert reps.shape == (8, 64), f"Layer {layer_spec} failed"
            assert torch.isfinite(reps).all()


class TestStatisticalPipeline:
    """End-to-end statistical analysis on synthetic 2x2 data."""

    def test_full_2x2_analysis(self):
        """Simulate 2x2 ablation with known effects."""
        n = 100
        torch.manual_seed(42)

        # Simulate known FM1 and FM2 effects
        base = torch.randn(n)
        fm1_boost = 2.0
        fm2_boost = 1.0

        cell_param_std = base
        cell_param_contr = base + fm2_boost
        cell_repr_std = base + fm1_boost
        cell_repr_contr = base + fm1_boost + fm2_boost

        result = full_ablation_analysis(
            cell_param_std, cell_param_contr, cell_repr_std, cell_repr_contr,
            n_permutations=500, n_bootstrap=500, seed=42,
        )

        # FM1 effect should be ~2.0
        assert abs(result["main_effects"]["fm1_effect"] - fm1_boost) < 0.5
        # FM2 effect should be ~1.0
        assert abs(result["main_effects"]["fm2_effect"] - fm2_boost) < 0.5
        # Interaction should be near zero (additive design)
        assert abs(result["main_effects"]["interaction"]) < 0.5
        # Both should be significant
        assert result["fm1_test"]["p_value"] < 0.05
        assert result["fm2_test"]["p_value"] < 0.05
        # Independence assessment
        assert result["independence_assessment"] in [
            "clean_independence", "moderate_interaction", "strong_interaction"
        ]

    def test_pairwise_with_metrics(self):
        """Combine metric computation with statistical testing."""
        torch.manual_seed(42)
        n_train = 100

        # Simulate per-sample LDS contributions for two methods
        method_a_lds = torch.randn(n_train) * 0.1 + 0.5  # Mean ~0.5
        method_b_lds = torch.randn(n_train) * 0.1 + 0.3  # Mean ~0.3

        diff, pval = permutation_test(method_a_lds, method_b_lds, n_permutations=500, seed=42)
        d = cohens_d(method_a_lds, method_b_lds)

        assert diff > 0  # A should be better
        assert pval < 0.05  # Significant difference
        assert d > 1.0  # Large effect

    def test_magic_feasibility_in_pipeline(self):
        """MAGIC feasibility check integrates with pipeline."""
        model = MockTransformerModel()
        result = magic_feasibility_check(
            model, n_train=10000, n_test=100, n_steps=200,
            gpu_memory_gb=48.0, disk_space_gb=500.0,
        )
        assert isinstance(result["feasible"], bool)
        assert result["n_params"] > 0


class TestCMRRPipeline:
    """CMRR computation integrates with contrastive scores."""

    def test_cmrr_from_repsim_scores(self):
        n_test, n_train, d = 5, 30, 64
        h_test_ft = torch.randn(n_test, d)
        h_train_ft = torch.randn(n_train, d)
        h_test_base = torch.randn(n_test, d)
        h_train_base = torch.randn(n_train, d)

        score_std = repsim_score(h_test_ft, h_train_ft)
        score_contr = contrastive_score_from_representations(
            h_test_ft, h_train_ft, h_test_base, h_train_base, repsim_score
        )

        cmrr_val = compute_cmrr(score_std, score_contr)
        # compute_cmrr from contrastive.py returns a tensor; from ablation_analysis.py returns float
        cmrr_float = cmrr_val.item() if hasattr(cmrr_val, 'item') else cmrr_val
        assert isinstance(cmrr_float, float)
        assert cmrr_float >= 0
