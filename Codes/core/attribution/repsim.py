# Component: RepSim (Representation Similarity Attribution)
# Source: research/method-design.md §5 Component A1
# Ablation config key: attribution.method = "repsim"

"""
RepSim: Training data attribution via cosine similarity of hidden representations.

I_RepSim(z_test, z_train) = cos(h^(l)(z_test), h^(l)(z_train))

Operates in R^d (d ~ 2048-4096) instead of R^B (B ~ 10^9), addressing FM1 (signal dilution).
Uses cosine similarity (not inner product), per DATE-LM finding: cosine > inner product.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def repsim_score(
    h_test: torch.Tensor,
    h_train: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute RepSim attribution scores: cosine similarity matrix.

    I_RepSim(z_test, z_train) = cos(h^(l)(z_test), h^(l)(z_train))

    Args:
        h_test: (n_test, d_model) -- hidden representations of test samples.
        h_train: (n_train, d_model) -- hidden representations of training samples.
        eps: Small constant for numerical stability in normalization.

    Returns:
        scores: (n_test, n_train) cosine similarity matrix.
    """
    # h_test: (n_test, d_model)
    # h_train: (n_train, d_model)
    assert h_test.ndim == 2, f"h_test must be 2D, got {h_test.ndim}D"
    assert h_train.ndim == 2, f"h_train must be 2D, got {h_train.ndim}D"
    assert h_test.shape[1] == h_train.shape[1], (
        f"Dimension mismatch: h_test has d={h_test.shape[1]}, h_train has d={h_train.shape[1]}"
    )

    # Normalize to unit vectors (with eps for zero-vector safety)
    h_test_norm = F.normalize(h_test, p=2, dim=1, eps=eps)   # (n_test, d_model)
    h_train_norm = F.normalize(h_train, p=2, dim=1, eps=eps)  # (n_train, d_model)

    # Cosine similarity via matrix multiply of normalized vectors
    scores = h_test_norm @ h_train_norm.T  # (n_test, n_train)

    return scores


def repsim_score_batched(
    h_test: torch.Tensor,
    h_train: torch.Tensor,
    batch_size: int = 1024,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Memory-efficient batched RepSim scoring for large training sets.

    Args:
        h_test: (n_test, d_model)
        h_train: (n_train, d_model)
        batch_size: Number of training samples per batch.
        eps: Numerical stability constant.

    Returns:
        scores: (n_test, n_train)
    """
    assert h_test.ndim == 2 and h_train.ndim == 2
    assert h_test.shape[1] == h_train.shape[1]

    h_test_norm = F.normalize(h_test, p=2, dim=1, eps=eps)
    h_train_norm = F.normalize(h_train, p=2, dim=1, eps=eps)

    n_train = h_train_norm.shape[0]
    score_chunks = []

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        chunk = h_test_norm @ h_train_norm[start:end].T  # (n_test, chunk_size)
        score_chunks.append(chunk)

    return torch.cat(score_chunks, dim=1)  # (n_test, n_train)
