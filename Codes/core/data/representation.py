# Component: Representation Extraction
# Source: research/method-design.md §5 Component A (shared infrastructure)
# Ablation config key: attribution.layer, attribution.token_aggregation

"""
Hidden representation extraction from HuggingFace transformer models.

Extracts h^(l)(z) at a specified layer for all samples, with configurable
token aggregation strategy (last_token or mean_pool).
"""

import torch
import torch.nn.functional as F
from typing import Union, Optional, Dict, Any, List
from torch.utils.data import DataLoader


def resolve_layer_index(layer: Union[int, str], n_layers: int) -> int:
    """
    Resolve layer specification to a 0-indexed layer number.

    Args:
        layer: int (0-indexed), "middle" (L/2), or "last" (L-1).
        n_layers: Total number of layers in the model.

    Returns:
        int: 0-indexed layer number.
    """
    if isinstance(layer, int):
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers})")
        return layer
    elif layer == "middle":
        return n_layers // 2
    elif layer == "last":
        return n_layers - 1
    else:
        raise ValueError(f"Unknown layer spec: {layer}. Use int, 'middle', or 'last'.")


def aggregate_tokens(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    aggregation: str = "last_token",
) -> torch.Tensor:
    """
    Aggregate token-level hidden states to sample-level representations.

    Args:
        hidden_states: (B, seq_len, d_model)
        attention_mask: (B, seq_len) with 1 for real tokens, 0 for padding. None if no padding.
        aggregation: "last_token" or "mean_pool"

    Returns:
        representations: (B, d_model)
    """
    # hidden_states: (B, seq_len, d_model)
    assert hidden_states.ndim == 3, f"Expected 3D tensor, got {hidden_states.ndim}D"

    if aggregation == "last_token":
        if attention_mask is not None:
            # Find the index of the last real token per sample
            # attention_mask: (B, seq_len), sum gives sequence lengths
            seq_lengths = attention_mask.sum(dim=1).long()  # (B,)
            # Clamp to at least 1 to avoid indexing -1
            last_indices = (seq_lengths - 1).clamp(min=0)  # (B,)
            # Gather last token representations
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, last_indices, :]  # (B, d_model)
        else:
            # No padding: last position
            return hidden_states[:, -1, :]  # (B, d_model)

    elif aggregation == "mean_pool":
        if attention_mask is not None:
            # Mask out padding tokens before averaging
            mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            summed = (hidden_states * mask).sum(dim=1)  # (B, d_model)
            counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            return summed / counts  # (B, d_model)
        else:
            return hidden_states.mean(dim=1)  # (B, d_model)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'last_token' or 'mean_pool'.")


@torch.no_grad()
def extract_representations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer: Union[int, str],
    aggregation: str = "last_token",
    n_layers: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract hidden representations at specified layer for all samples in dataloader.

    Args:
        model: HuggingFace model (must support output_hidden_states=True).
        dataloader: DataLoader yielding dicts with 'input_ids' and optionally 'attention_mask'.
        layer: int (0-indexed), "middle", or "last".
        aggregation: "last_token" or "mean_pool".
        n_layers: Total number of layers. If None, inferred from model config.
        device: Device to run inference on. If None, uses model's device.

    Returns:
        representations: (N, d_model) tensor of aggregated hidden representations.
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Infer n_layers from model config if not provided
    if n_layers is None:
        if hasattr(model, "config"):
            n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is None:
            raise ValueError("Cannot infer n_layers from model. Please provide n_layers explicitly.")

    layer_idx = resolve_layer_index(layer, n_layers)

    all_representations = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states is a tuple of (n_layers + 1) tensors: (embedding, layer_0, ..., layer_{n-1})
        # layer_idx=0 corresponds to hidden_states[1] (output of first transformer layer)
        hidden = outputs.hidden_states[layer_idx + 1]  # (B, seq_len, d_model)

        reps = aggregate_tokens(hidden, attention_mask, aggregation)  # (B, d_model)
        all_representations.append(reps.cpu())

    return torch.cat(all_representations, dim=0)  # (N, d_model)


@torch.no_grad()
def extract_all_layer_representations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    aggregation: str = "last_token",
    n_layers: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Extract hidden representations at ALL layers (for layer sweep analysis).

    Args:
        model: HuggingFace model.
        dataloader: DataLoader yielding dicts with 'input_ids' and optionally 'attention_mask'.
        aggregation: "last_token" or "mean_pool".
        n_layers: Total number of layers. If None, inferred from model config.
        device: Device to run inference on.

    Returns:
        List of (N, d_model) tensors, one per layer (L tensors total).
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    if n_layers is None:
        if hasattr(model, "config"):
            n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is None:
            raise ValueError("Cannot infer n_layers from model. Please provide n_layers explicitly.")

    # Collect per-layer representations
    layer_reps = [[] for _ in range(n_layers)]

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        for l in range(n_layers):
            hidden = outputs.hidden_states[l + 1]  # (B, seq_len, d_model)
            reps = aggregate_tokens(hidden, attention_mask, aggregation)  # (B, d_model)
            layer_reps[l].append(reps.cpu())

    return [torch.cat(reps, dim=0) for reps in layer_reps]
