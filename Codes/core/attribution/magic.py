# Component: MAGIC (Metagradient-based Attribution for Genuine Influence Computation)
# Source: research/method-design.md §5 Component C
# Ablation config key: attribution.method = "magic"

"""
MAGIC: Exact influence function via metagradient computation.

Eliminates Hessian approximation error by computing exact influence through
the training trajectory. Serves as the upper bound for parameter-space TDA.

Cost: O(N * n * T) where N = training samples, n = test samples, T = training steps.
This is best-effort: may be infeasible at Pythia-1B scale (1.6TB checkpoint storage).

Implementation follows the metagradient approach:
1. Store all training checkpoints (or recompute via deterministic replay)
2. For each test sample, backpropagate through the training trajectory
3. Accumulate per-training-sample influence via chain rule
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Callable
from pathlib import Path


def magic_score_single_test(
    test_loss_fn: Callable[[torch.nn.Module], torch.Tensor],
    checkpoints: List[Dict],
    training_losses: List[Callable[[torch.nn.Module, int], torch.Tensor]],
    learning_rates: List[float],
    n_train: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute MAGIC exact influence for a SINGLE test sample against all training samples.

    This implements the metagradient chain rule:
    I(z_test, z_train_i) = sum_{t where z_train_i is used} [
        nabla_theta L_test(theta_T) *
        prod_{s=T}^{t+1} (I - lr_s * H_s) *
        lr_t * nabla_theta L_train_i(theta_t)
    ]

    Args:
        test_loss_fn: Function that takes model and returns scalar test loss.
        checkpoints: List of checkpoint dicts (state_dict) at each training step.
        training_losses: List of functions: training_losses[t](model, sample_idx) -> loss.
        learning_rates: Learning rate at each training step.
        n_train: Number of training samples.
        device: Computation device.

    Returns:
        scores: (n_train,) exact influence scores for this test sample.

    NOTE: This is a simplified skeleton. Full implementation requires:
    - Efficient Jacobian-vector products
    - Checkpoint loading/unloading to manage memory
    - Deterministic training replay
    """
    raise NotImplementedError(
        "MAGIC full implementation is best-effort and depends on feasibility assessment. "
        "See method-design.md §5 Component C for feasibility constraints. "
        "At Pythia-1B scale with DATE-LM toxicity (N=10K, T=200), "
        "estimated: ~3-5 hours per test sample on A6000, 1.6TB checkpoint storage."
    )


def magic_feasibility_check(
    model: torch.nn.Module,
    n_train: int,
    n_test: int,
    n_steps: int,
    gpu_memory_gb: float = 48.0,
    disk_space_gb: float = 500.0,
) -> Dict[str, object]:
    """
    Assess MAGIC feasibility for given configuration.

    Args:
        model: Model to evaluate.
        n_train: Number of training samples.
        n_test: Number of test samples.
        n_steps: Number of training steps.
        gpu_memory_gb: Available GPU memory in GB.
        disk_space_gb: Available disk space in GB.

    Returns:
        Dict with feasibility assessment:
        - feasible: bool
        - estimated_time_hours: float
        - estimated_disk_gb: float
        - estimated_gpu_memory_gb: float
        - bottleneck: str (what limits feasibility)
    """
    # Model size in GB (fp16)
    n_params = sum(p.numel() for p in model.parameters())
    model_size_gb = n_params * 2 / (1024 ** 3)  # fp16

    # Checkpoint storage: n_steps * model_size (with optimizer state ~3x)
    checkpoint_size_gb = n_steps * model_size_gb * 3  # model + optimizer state

    # GPU memory: model + gradients + optimizer overhead
    gpu_needed_gb = model_size_gb * 4  # model + gradients + backward buffers

    # Time estimate: ~3-5 hours per test sample for Pythia-1B
    # Scale linearly with n_params relative to Pythia-1B (1B params)
    time_per_test_hours = (n_params / 1e9) * 4.0  # ~4 hours per test for 1B
    total_time_hours = time_per_test_hours * n_test

    # Feasibility
    bottlenecks = []
    if checkpoint_size_gb > disk_space_gb:
        bottlenecks.append(f"disk: need {checkpoint_size_gb:.0f}GB, have {disk_space_gb:.0f}GB")
    if gpu_needed_gb > gpu_memory_gb:
        bottlenecks.append(f"GPU memory: need {gpu_needed_gb:.1f}GB, have {gpu_memory_gb:.1f}GB")
    if total_time_hours > 500:
        bottlenecks.append(f"time: {total_time_hours:.0f} hours exceeds practical budget")

    feasible = len(bottlenecks) == 0

    return {
        "feasible": feasible,
        "estimated_time_hours": total_time_hours,
        "estimated_disk_gb": checkpoint_size_gb,
        "estimated_gpu_memory_gb": gpu_needed_gb,
        "n_params": n_params,
        "bottleneck": "; ".join(bottlenecks) if bottlenecks else "none",
    }
