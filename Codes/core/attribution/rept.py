# Component: RepT (Representation Gradient Tracing)
# Source: research/method-design.md §5 Component A2
# Ablation config key: attribution.method = "rept"

"""
RepT: Training data attribution via representation + gradient features.

phi(z) = concat[h^(l*)(z), nabla_h L(z)]
I_RepT(z_test, z_train) = cos(phi(z_test), phi(z_train))

l* = phase-transition layer (layer where gradient norm ratio is maximized).
Gradient component nabla_h L adds causal signal to correlational representation.

WARNING: Most likely component to have bugs (design review prediction).
"""

import warnings

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
from torch.utils.data import DataLoader


def detect_phase_transition_layer(
    gradient_norms: torch.Tensor,
) -> int:
    """
    Detect the phase-transition layer l* where gradient norm exhibits sharp change.

    l* = argmax_l (||nabla_h^(l) L|| / ||nabla_h^(l-1) L||)

    Args:
        gradient_norms: (n_layers,) tensor of per-layer gradient norms.

    Returns:
        l_star: Index of the phase-transition layer.
    """
    # gradient_norms: (n_layers,)
    assert gradient_norms.ndim == 1, f"Expected 1D tensor, got {gradient_norms.ndim}D"
    assert len(gradient_norms) >= 2, "Need at least 2 layers for transition detection"

    # If all gradient norms are near-zero (degenerate model), fall back to L/2
    eps = 1e-10
    if gradient_norms.max() < eps:
        fallback = len(gradient_norms) // 2
        warnings.warn(
            f"All gradient norms are near-zero (max={gradient_norms.max():.2e}). "
            f"Falling back to middle layer {fallback}."
        )
        return fallback

    # Compute ratio of consecutive layers
    ratios = gradient_norms[1:] / (gradient_norms[:-1] + eps)  # (n_layers - 1,)

    # l* is the layer WITH the max ratio (index in ratios corresponds to layer index + 1)
    l_star = ratios.argmax().item() + 1  # +1 because ratios[i] = norm[i+1] / norm[i]

    return l_star


def compute_layer_gradient_norms(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    n_layers: Optional[int] = None,
    device: Optional[torch.device] = None,
    n_samples: int = 100,
) -> torch.Tensor:
    """
    Compute average gradient norm ||nabla_h^(l) L|| at each layer.

    Uses a subset of samples to estimate the gradient norm profile.

    Args:
        model: HuggingFace model.
        dataloader: DataLoader yielding dicts with 'input_ids', 'labels'.
        loss_fn: Per-sample loss function.
        n_layers: Number of layers. If None, inferred from model config.
        device: Computation device.
        n_samples: Number of samples to use for estimation.

    Returns:
        gradient_norms: (n_layers,) average gradient norms per layer.
    """
    if device is None:
        device = next(model.parameters()).device
    if n_layers is None:
        n_layers = model.config.num_hidden_layers

    model.eval()
    accumulated_norms = torch.zeros(n_layers, device="cpu")
    count = 0

    for batch in dataloader:
        if count >= n_samples:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            if count >= n_samples:
                break

            # Forward with hidden states, enabling gradient computation
            with torch.enable_grad():
                outputs = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
                    output_hidden_states=True,
                )

                # Compute loss
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels[i:i+1].view(-1))
                loss = loss.mean()

                # Get hidden states (tuple of n_layers+1 tensors)
                hidden_states = outputs.hidden_states

                # Compute gradient norm at each layer
                for l in range(n_layers):
                    h_l = hidden_states[l + 1]  # (1, seq_len, d_model)
                    if h_l.requires_grad:
                        grad = torch.autograd.grad(
                            loss, h_l, retain_graph=True, create_graph=False
                        )[0]
                        accumulated_norms[l] += grad.norm().item()
                    # If h_l doesn't require grad, norm stays 0

            count += 1

    # Average
    if count > 0:
        accumulated_norms /= count

    return accumulated_norms


def extract_rept_features(
    hidden_rep: torch.Tensor,
    hidden_grad: torch.Tensor,
) -> torch.Tensor:
    """
    Construct RepT feature vector: phi(z) = concat[h^(l*)(z), nabla_h L(z)].

    Args:
        hidden_rep: (N, d_model) -- hidden representations at layer l*.
        hidden_grad: (N, d_model) -- gradients of loss w.r.t. hidden at layer l*.

    Returns:
        phi: (N, 2 * d_model) -- concatenated feature vector.
    """
    # hidden_rep: (N, d_model), hidden_grad: (N, d_model)
    assert hidden_rep.ndim == 2 and hidden_grad.ndim == 2
    assert hidden_rep.shape == hidden_grad.shape, (
        f"Shape mismatch: rep {hidden_rep.shape} vs grad {hidden_grad.shape}"
    )

    return torch.cat([hidden_rep, hidden_grad], dim=1)  # (N, 2 * d_model)


def rept_score(
    phi_test: torch.Tensor,
    phi_train: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute RepT attribution scores: cosine similarity of RepT features.

    I_RepT(z_test, z_train) = cos(phi(z_test), phi(z_train))

    Args:
        phi_test: (n_test, 2*d_model) -- RepT features for test samples.
        phi_train: (n_train, 2*d_model) -- RepT features for training samples.
        eps: Numerical stability constant.

    Returns:
        scores: (n_test, n_train) cosine similarity matrix.
    """
    # phi_test: (n_test, 2*d_model), phi_train: (n_train, 2*d_model)
    assert phi_test.ndim == 2, f"phi_test must be 2D, got {phi_test.ndim}D"
    assert phi_train.ndim == 2, f"phi_train must be 2D, got {phi_train.ndim}D"
    assert phi_test.shape[1] == phi_train.shape[1], (
        f"Feature dim mismatch: {phi_test.shape[1]} vs {phi_train.shape[1]}"
    )

    phi_test_norm = F.normalize(phi_test, p=2, dim=1, eps=eps)
    phi_train_norm = F.normalize(phi_train, p=2, dim=1, eps=eps)

    return phi_test_norm @ phi_train_norm.T  # (n_test, n_train)


def extract_hidden_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    layer: int,
    aggregation: str = "last_token",
    n_layers: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract both hidden representations and their gradients at specified layer.

    Args:
        model: HuggingFace model.
        dataloader: DataLoader yielding dicts with 'input_ids', 'labels', optionally 'attention_mask'.
        loss_fn: Loss function.
        layer: Layer index (0-indexed).
        aggregation: "last_token" or "mean_pool".
        n_layers: Number of layers.
        device: Computation device.

    Returns:
        (representations, gradients): Both (N, d_model) tensors.
    """
    from core.data.representation import aggregate_tokens

    if device is None:
        device = next(model.parameters()).device
    if n_layers is None:
        n_layers = model.config.num_hidden_layers

    model.eval()
    all_reps = []
    all_grads = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            with torch.enable_grad():
                outputs = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
                    output_hidden_states=True,
                )

                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels[i:i+1].view(-1))
                loss = loss.mean()

                h_l = outputs.hidden_states[layer + 1]  # (1, seq_len, d_model)

                if h_l.requires_grad:
                    grad = torch.autograd.grad(loss, h_l, create_graph=False)[0]
                else:
                    warnings.warn(
                        f"Layer {layer} hidden state has requires_grad=False; "
                        "gradient will be zero. Check model forward pass."
                    )
                    grad = torch.zeros_like(h_l)

            # Aggregate token dimension
            mask_i = attention_mask[i:i+1] if attention_mask is not None else None
            rep = aggregate_tokens(h_l.detach(), mask_i, aggregation)  # (1, d_model)
            grad_agg = aggregate_tokens(grad.detach(), mask_i, aggregation)  # (1, d_model)

            all_reps.append(rep.cpu())
            all_grads.append(grad_agg.cpu())

    return torch.cat(all_reps, dim=0), torch.cat(all_grads, dim=0)
