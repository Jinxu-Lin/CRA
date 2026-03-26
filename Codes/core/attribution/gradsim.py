# Component: Grad-Sim (Gradient Cosine Similarity)
# Source: research/method-design.md §5 (baseline, parameter-space)
# Ablation config key: attribution.method = "gradsim"

"""
Grad-Sim: Training data attribution via cosine similarity of per-sample gradients.

I_GradSim(z_test, z_train) = cos(g(z_test), g(z_train))

where g(z) = nabla_theta L(z; theta) is the per-sample gradient at the fine-tuned model.

This is a parameter-space baseline. If DATE-LM provides Grad-Sim, use theirs.
This implementation exists as fallback and for controlled experiments.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def gradsim_score(
    g_test: torch.Tensor,
    g_train: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Grad-Sim attribution scores: cosine similarity of gradients.

    I_GradSim(z_test, z_train) = cos(g(z_test), g(z_train))

    Args:
        g_test: (n_test, B) -- flattened per-sample gradients of test samples.
        g_train: (n_train, B) -- flattened per-sample gradients of training samples.
        eps: Numerical stability constant.

    Returns:
        scores: (n_test, n_train) cosine similarity matrix.
    """
    # g_test: (n_test, B), g_train: (n_train, B)
    assert g_test.ndim == 2, f"g_test must be 2D, got {g_test.ndim}D"
    assert g_train.ndim == 2, f"g_train must be 2D, got {g_train.ndim}D"
    assert g_test.shape[1] == g_train.shape[1], (
        f"Gradient dimension mismatch: g_test has B={g_test.shape[1]}, g_train has B={g_train.shape[1]}"
    )

    g_test_norm = F.normalize(g_test, p=2, dim=1, eps=eps)    # (n_test, B)
    g_train_norm = F.normalize(g_train, p=2, dim=1, eps=eps)   # (n_train, B)

    scores = g_test_norm @ g_train_norm.T  # (n_test, n_train)

    return scores


def gradsim_score_batched(
    g_test: torch.Tensor,
    g_train: torch.Tensor,
    batch_size: int = 256,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Memory-efficient batched Grad-Sim scoring.

    For Pythia-1B, B ~ 10^9, so per-sample gradients are huge.
    This batches over training samples to control memory.

    Args:
        g_test: (n_test, B)
        g_train: (n_train, B)
        batch_size: Number of training samples per batch.
        eps: Numerical stability constant.

    Returns:
        scores: (n_test, n_train)
    """
    assert g_test.ndim == 2 and g_train.ndim == 2
    assert g_test.shape[1] == g_train.shape[1]

    g_test_norm = F.normalize(g_test, p=2, dim=1, eps=eps)
    g_train_norm = F.normalize(g_train, p=2, dim=1, eps=eps)

    n_train = g_train_norm.shape[0]
    score_chunks = []

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        chunk = g_test_norm @ g_train_norm[start:end].T
        score_chunks.append(chunk)

    return torch.cat(score_chunks, dim=1)


def extract_per_sample_gradients(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: Optional[torch.device] = None,
    max_params: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract per-sample gradients for all samples.

    WARNING: This creates (N, B) tensors where B can be ~10^9 for Pythia-1B.
    Only use for small models or subsets of parameters.

    Args:
        model: Model to compute gradients for.
        dataloader: DataLoader yielding dicts with 'input_ids', 'labels', optionally 'attention_mask'.
        loss_fn: Loss function (e.g., CrossEntropyLoss with reduction='none').
        device: Device for computation.
        max_params: If set, only use first max_params parameters (for testing).

    Returns:
        gradients: (N, B) tensor of flattened per-sample gradients.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Collect parameter references
    params = [p for p in model.parameters() if p.requires_grad]
    if max_params is not None:
        params = params[:max_params]

    all_grads = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            model.zero_grad()
            # Single-sample forward
            inp = input_ids[i:i+1]
            lab = labels[i:i+1]
            mask = attention_mask[i:i+1] if attention_mask is not None else None

            outputs = model(input_ids=inp, attention_mask=mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = loss_fn(logits.view(-1, logits.size(-1)), lab.view(-1))
            loss = loss.mean()
            loss.backward()

            # Flatten all parameter gradients
            grad_vec = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
            all_grads.append(grad_vec.detach().cpu())

    return torch.stack(all_grads, dim=0)  # (N, B)
