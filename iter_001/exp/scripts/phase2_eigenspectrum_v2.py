#!/usr/bin/env python3
"""
Phase 2a: Gradient & Representation Eigenspectrum Analysis (H4, H9) -- v2

On Pythia-70M (d=512):
1. Compute representation covariance eigenspectrum (exact, d=512).
2. Compute gradient covariance eigenspectrum via direct SVD of the gradient data matrix.
   For N=100 << D=6.3M, SVD of the N x D matrix is O(N^2 * D), much faster than Lanczos.
3. Measure r_eff(95%) for both.
4. Compute condition numbers.

Hypotheses:
- H4: r_eff(Sigma_g) in [256, 1024]
- H9: rep condition < 100, grad condition > 10^4

PILOT mode: 100 training samples, seed=42
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────────
TASK_ID = "phase2_eigenspectrum"
SEED = 42
PILOT_N = 100
MODEL_NAME = "EleutherAI/pythia-70m"
CHECKPOINT_CACHE = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-70m"
PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/CRA"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
PHASE2_DIR = f"{RESULTS_DIR}/phase2"
DEVICE = "cuda:0"

# Which layers to compute gradients for (last 2 layers, matching kfac_control)
TARGET_LAYERS = [4, 5]

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Helper functions ────────────────────────────────────────────────────────

def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def compute_r_eff(eigenvalues, threshold=0.95):
    """Minimum number of eigenvalues capturing threshold% of total variance."""
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
    total = sorted_eigs.sum()
    if total < 1e-15:
        return len(eigenvalues)
    cumsum = np.cumsum(sorted_eigs) / total
    r_eff = int(np.searchsorted(cumsum, threshold)) + 1
    return r_eff


def compute_condition_number(eigenvalues):
    """Condition number from eigenvalues (ratio of largest to smallest non-zero)."""
    abs_eigs = np.abs(eigenvalues)
    abs_eigs = abs_eigs[abs_eigs > 1e-15]
    if len(abs_eigs) < 2:
        return float('inf')
    return float(abs_eigs.max() / abs_eigs.min())


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    os.makedirs(PHASE2_DIR, exist_ok=True)

    # Write PID file
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    report_progress(TASK_ID, RESULTS_DIR, 0, 5, step=0, total_steps=5,
                    metric={"phase": "loading_model"})

    print(f"[{TASK_ID}] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CHECKPOINT_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CHECKPOINT_CACHE, dtype=torch.float32
    )
    model.to(DEVICE)
    model.eval()

    hidden_dim = model.config.hidden_size  # 512
    print(f"[{TASK_ID}] Hidden dim: {hidden_dim}, Layers: {model.config.num_hidden_layers}")

    # Identify target parameters
    target_params = []
    target_param_names = []
    for name, param in model.named_parameters():
        for l in TARGET_LAYERS:
            if f"layers.{l}." in name and param.requires_grad:
                target_params.append(param)
                target_param_names.append(name)
                break

    total_grad_dim = sum(p.numel() for p in target_params)
    print(f"[{TASK_ID}] Target layers: {TARGET_LAYERS}")
    print(f"[{TASK_ID}] Target parameters: {len(target_params)}, total dim: {total_grad_dim:,}")

    # ── Load training data ──────────────────────────────────────────────
    report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=1, total_steps=5,
                    metric={"phase": "loading_data"})

    print(f"[{TASK_ID}] Loading toxicity dataset for training samples")
    ds = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering",
                      "XSTest-response-Het", split="train")

    texts = []
    for i in range(min(PILOT_N, len(ds))):
        row = ds[i]
        text = row.get("text", row.get("prompt", "") + " " + row.get("response", ""))
        texts.append(text.strip())

    n_samples = len(texts)
    print(f"[{TASK_ID}] Loaded {n_samples} training samples")

    # Tokenize
    max_len = 256
    encodings = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len)

    # ── Part 1: Representation Covariance Eigenspectrum ─────────────────
    report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=2, total_steps=5,
                    metric={"phase": "representation_eigenspectrum"})

    print(f"\n[{TASK_ID}] === Part 1: Representation Covariance Eigenspectrum ===")
    rep_start = time.time()

    all_reps = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_ids = encodings["input_ids"][i:i+batch_size].to(DEVICE)
            batch_mask = encodings["attention_mask"][i:i+batch_size].to(DEVICE)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask,
                          output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (B, T, d)
            mask_expanded = batch_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            all_reps.append(pooled.cpu().float())

    reps = torch.cat(all_reps, dim=0).numpy()  # (N, d)
    print(f"[{TASK_ID}] Representations shape: {reps.shape}")

    # Center and compute exact covariance (d x d = 512 x 512)
    reps_centered = reps - reps.mean(axis=0, keepdims=True)
    cov_rep = (reps_centered.T @ reps_centered) / (n_samples - 1)

    # Exact eigendecomposition
    rep_eigenvalues, _ = np.linalg.eigh(cov_rep)
    rep_eigenvalues = rep_eigenvalues[::-1].copy()  # Sort descending

    rep_r_eff_95 = compute_r_eff(rep_eigenvalues, 0.95)
    rep_r_eff_99 = compute_r_eff(rep_eigenvalues, 0.99)
    rep_condition = compute_condition_number(rep_eigenvalues)

    rep_time = time.time() - rep_start
    print(f"[{TASK_ID}] Rep eigenspectrum computed in {rep_time:.1f}s")
    print(f"[{TASK_ID}] Rep r_eff(95%): {rep_r_eff_95}, r_eff(99%): {rep_r_eff_99}")
    print(f"[{TASK_ID}] Rep condition number: {rep_condition:.2e}")

    rep_total_var = np.abs(rep_eigenvalues).sum()
    rep_explained_cumulative = np.cumsum(np.abs(rep_eigenvalues)) / rep_total_var

    # ── Part 2: Gradient Covariance Eigenspectrum via Direct SVD ────────
    report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=3, total_steps=5,
                    metric={"phase": "gradient_eigenspectrum_direct_svd"})

    print(f"\n[{TASK_ID}] === Part 2: Gradient Covariance Eigenspectrum (Direct SVD) ===")
    print(f"[{TASK_ID}] Strategy: Collect all {n_samples} gradients (N x D = {n_samples} x {total_grad_dim:,})")
    print(f"[{TASK_ID}] Then SVD of N x D matrix => O(N^2 * D) = O({n_samples**2} * {total_grad_dim:,})")
    grad_start = time.time()

    # Collect all per-sample gradients
    all_grads = []
    for i in range(n_samples):
        model.zero_grad()
        ids = encodings["input_ids"][i:i+1].to(DEVICE)
        mask = encodings["attention_mask"][i:i+1].to(DEVICE)
        outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs.loss
        loss.backward()

        grads = []
        for p in target_params:
            if p.grad is not None:
                grads.append(p.grad.detach().clone().flatten().cpu())
            else:
                grads.append(torch.zeros(p.numel()))
        all_grads.append(torch.cat(grads))

        if (i + 1) % 20 == 0:
            elapsed = time.time() - grad_start
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"[{TASK_ID}] Gradient computation: {i+1}/{n_samples} ({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")
            report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=i+1, total_steps=n_samples,
                            metric={"phase": "computing_gradients", "eta_sec": round(eta)})

    grad_matrix = torch.stack(all_grads, dim=0).numpy()  # (N, D)
    del all_grads
    gc.collect()
    print(f"[{TASK_ID}] Gradient matrix shape: {grad_matrix.shape}, "
          f"memory: {grad_matrix.nbytes / 1e9:.2f} GB")

    # Center gradients
    grad_centered = grad_matrix - grad_matrix.mean(axis=0, keepdims=True)
    del grad_matrix
    gc.collect()

    # SVD of centered gradient matrix: G = U S V^T
    # Covariance eigenvalues = S^2 / (N-1)
    # Since N << D, the SVD is efficient: O(N^2 * D)
    print(f"[{TASK_ID}] Computing SVD of {n_samples} x {total_grad_dim:,} gradient matrix...")
    svd_start = time.time()
    U, S, Vt = np.linalg.svd(grad_centered, full_matrices=False)
    svd_time = time.time() - svd_start
    print(f"[{TASK_ID}] SVD completed in {svd_time:.1f}s, {len(S)} singular values")

    grad_eigenvalues = (S ** 2) / (n_samples - 1)  # Already sorted descending

    # Also save top eigenvectors (V) for potential TRAK-PCA later
    # Vt[:k, :] are the top-k right singular vectors (= eigenvectors of Sigma_g)
    # Save top-512 (= d) for use in phase2_trak_dim_sweep
    top_k_save = min(hidden_dim, len(S))
    grad_top_eigenvectors = Vt[:top_k_save, :]  # (k, D)

    del grad_centered, U, Vt
    gc.collect()

    grad_r_eff_95 = compute_r_eff(grad_eigenvalues, 0.95)
    grad_r_eff_99 = compute_r_eff(grad_eigenvalues, 0.99)
    grad_condition = compute_condition_number(grad_eigenvalues)

    grad_time = time.time() - grad_start
    print(f"[{TASK_ID}] Gradient eigenspectrum computed in {grad_time:.1f}s")
    print(f"[{TASK_ID}] Grad r_eff(95%): {grad_r_eff_95}, r_eff(99%): {grad_r_eff_99}")
    print(f"[{TASK_ID}] Grad condition number: {grad_condition:.2e}")
    print(f"[{TASK_ID}] Total gradient eigenvalues: {len(grad_eigenvalues)}")

    grad_total_var = np.abs(grad_eigenvalues).sum()
    grad_explained_cumulative = (
        np.cumsum(np.abs(grad_eigenvalues)) / grad_total_var
        if grad_total_var > 0 else np.zeros(len(grad_eigenvalues))
    )

    # ── Part 3: Hypothesis Testing ──────────────────────────────────────
    report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=4, total_steps=5,
                    metric={"phase": "hypothesis_testing"})

    print(f"\n[{TASK_ID}] === Hypothesis Testing ===")

    # H4: r_eff(Sigma_g) in [256, 1024] (= [0.5d, 2d] for d=512)
    h4_lower = 0.5 * hidden_dim  # 256
    h4_upper = 2 * hidden_dim    # 1024

    # With N=100 pilot, max rank is N-1=99, so we can't directly test H4.
    # Instead, we measure concentration: what fraction of variance is in top-k.
    max_possible_rank = min(n_samples - 1, total_grad_dim)
    h4_r_eff_frac = grad_r_eff_95 / max_possible_rank if max_possible_rank > 0 else 0

    # Key insight for pilot: if grad eigenvalues decay SLOWLY (r_eff close to N),
    # this suggests high intrinsic dimension, consistent with H4 at full scale.
    # If grad eigenvalues decay FAST (r_eff << N), this suggests low intrinsic dim.
    h4_note = (
        f"Pilot N={n_samples}, max rank={max_possible_rank}. "
        f"r_eff(95%)={grad_r_eff_95} ({h4_r_eff_frac:.1%} of max rank). "
        f"Full-scale target: r_eff in [{int(h4_lower)}, {int(h4_upper)}]."
    )
    # Extrapolation: if r_eff/N > 0.5, the spectrum is relatively flat => high intrinsic dim
    h4_directional = h4_r_eff_frac > 0.3  # loose threshold for pilot
    h4_note += f" Directional check (r_eff/N > 0.3): {'PASS' if h4_directional else 'FAIL'}."

    # H9: rep condition < 100, grad condition > 10^4
    h9_rep_pass = rep_condition < 100
    h9_grad_pass = grad_condition > 1e4
    h9_pass = h9_rep_pass and h9_grad_pass
    h9_condition_ratio = grad_condition / rep_condition if rep_condition > 0 else float('inf')

    print(f"[{TASK_ID}] H4: r_eff(Sigma_g) = {grad_r_eff_95}, r_eff/N = {h4_r_eff_frac:.3f}")
    print(f"[{TASK_ID}]     Directional: {'PASS' if h4_directional else 'FAIL'}")
    print(f"[{TASK_ID}] H9: rep_condition = {rep_condition:.2e} (<100? {h9_rep_pass})")
    print(f"[{TASK_ID}]     grad_condition = {grad_condition:.2e} (>10^4? {h9_grad_pass})")
    print(f"[{TASK_ID}]     condition ratio = {h9_condition_ratio:.2e}")

    # ── Part 4: Save Results ────────────────────────────────────────────
    report_progress(TASK_ID, RESULTS_DIR, 5, 5, step=5, total_steps=5,
                    metric={"phase": "saving_results"})

    # Save eigenspectrum data for visualization
    results = {
        "task_id": TASK_ID,
        "version": "v2",
        "model": MODEL_NAME,
        "hidden_dim": hidden_dim,
        "n_samples": n_samples,
        "target_layers": TARGET_LAYERS,
        "total_grad_dim": total_grad_dim,
        "seed": SEED,
        "mode": "pilot",

        "representation": {
            "covariance_shape": list(cov_rep.shape),
            "eigenvalues": rep_eigenvalues.tolist(),
            "n_eigenvalues": len(rep_eigenvalues),
            "r_eff_95": int(rep_r_eff_95),
            "r_eff_99": int(rep_r_eff_99),
            "condition_number": float(rep_condition),
            "top_10_eigenvalues": rep_eigenvalues[:10].tolist(),
            "bottom_10_eigenvalues": rep_eigenvalues[-10:].tolist(),
            "explained_var_at_d_over_4": float(rep_explained_cumulative[min(hidden_dim//4 - 1, len(rep_explained_cumulative)-1)]),
            "explained_var_at_d_over_2": float(rep_explained_cumulative[min(hidden_dim//2 - 1, len(rep_explained_cumulative)-1)]),
            "runtime_sec": round(rep_time, 2),
        },

        "gradient": {
            "method": "direct_svd",
            "n_eigenvalues": len(grad_eigenvalues),
            "eigenvalues": grad_eigenvalues.tolist(),  # All N-1 eigenvalues
            "singular_values": S.tolist(),
            "r_eff_95": int(grad_r_eff_95),
            "r_eff_99": int(grad_r_eff_99),
            "condition_number": float(grad_condition),
            "top_10_eigenvalues": grad_eigenvalues[:10].tolist(),
            "bottom_10_eigenvalues": grad_eigenvalues[-10:].tolist(),
            "explained_var_at_d": float(
                grad_explained_cumulative[min(hidden_dim-1, len(grad_explained_cumulative)-1)]
            ) if len(grad_explained_cumulative) > 0 else None,
            "svd_time_sec": round(svd_time, 2),
            "runtime_sec": round(grad_time, 2),
        },

        "hypotheses": {
            "H4": {
                "description": "r_eff(Sigma_g) in [0.5d, 2d] = [256, 1024]",
                "r_eff_95": int(grad_r_eff_95),
                "target_range": [int(h4_lower), int(h4_upper)],
                "max_possible_rank_pilot": max_possible_rank,
                "r_eff_fraction_of_max_rank": round(h4_r_eff_frac, 4),
                "directional_pass": h4_directional,
                "note": h4_note,
            },
            "H9": {
                "description": "rep_condition < 100, grad_condition > 10^4",
                "rep_condition": float(rep_condition),
                "grad_condition": float(grad_condition),
                "condition_ratio": float(h9_condition_ratio),
                "rep_pass": h9_rep_pass,
                "grad_pass": h9_grad_pass,
                "pass": h9_pass,
            },
        },

        "spectral_comparison": {
            "rep_r_eff_95": int(rep_r_eff_95),
            "grad_r_eff_95": int(grad_r_eff_95),
            "rep_condition": float(rep_condition),
            "grad_condition": float(grad_condition),
            "condition_ratio": float(h9_condition_ratio),
            "rep_top1_frac": float(np.abs(rep_eigenvalues[0]) / rep_total_var) if rep_total_var > 0 else 0,
            "grad_top1_frac": float(np.abs(grad_eigenvalues[0]) / grad_total_var) if grad_total_var > 0 else 0,
        },

        "qualitative_samples": {
            "rep_eigenvalue_decay": {
                "description": "How quickly representation eigenvalues decay",
                "top_5_ratio": float(np.abs(rep_eigenvalues[:5]).sum() / rep_total_var) if rep_total_var > 0 else 0,
                "top_10_ratio": float(np.abs(rep_eigenvalues[:10]).sum() / rep_total_var) if rep_total_var > 0 else 0,
                "top_50_ratio": float(np.abs(rep_eigenvalues[:50]).sum() / rep_total_var) if rep_total_var > 0 else 0,
                "top_100_ratio": float(np.abs(rep_eigenvalues[:min(100, len(rep_eigenvalues))]).sum() / rep_total_var) if rep_total_var > 0 else 0,
            },
            "grad_eigenvalue_decay": {
                "description": "How quickly gradient eigenvalues decay",
                "top_5_ratio": float(np.abs(grad_eigenvalues[:5]).sum() / grad_total_var) if grad_total_var > 0 else 0,
                "top_10_ratio": float(np.abs(grad_eigenvalues[:10]).sum() / grad_total_var) if grad_total_var > 0 else 0,
                "top_50_ratio": float(np.abs(grad_eigenvalues[:min(50, len(grad_eigenvalues))]).sum() / grad_total_var) if grad_total_var > 0 else 0,
                "top_100_ratio": float(np.abs(grad_eigenvalues[:min(100, len(grad_eigenvalues))]).sum() / grad_total_var) if grad_total_var > 0 else 0,
            },
        },

        "total_runtime_sec": round(time.time() - start_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save main results JSON
    out_path = Path(PHASE2_DIR) / "eigenspectrum.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{TASK_ID}] Results saved to {out_path}")

    # Save top gradient eigenvectors for TRAK-PCA (phase2_trak_dim_sweep)
    eigvec_path = Path(PHASE2_DIR) / "grad_top_eigenvectors.npy"
    np.save(str(eigvec_path), grad_top_eigenvectors)
    print(f"[{TASK_ID}] Top-{top_k_save} gradient eigenvectors saved to {eigvec_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"[{TASK_ID}] SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_NAME} (d={hidden_dim})")
    print(f"  Samples: {n_samples} (pilot)")
    print(f"  Gradient dim: {total_grad_dim:,} (layers {TARGET_LAYERS})")
    print(f"")
    print(f"  REPRESENTATION SPACE:")
    print(f"    r_eff(95%): {rep_r_eff_95}")
    print(f"    r_eff(99%): {rep_r_eff_99}")
    print(f"    Condition number: {rep_condition:.2e}")
    print(f"    Top-5 eigenvalue fraction: {results['qualitative_samples']['rep_eigenvalue_decay']['top_5_ratio']:.4f}")
    print(f"    Top-10 eigenvalue fraction: {results['qualitative_samples']['rep_eigenvalue_decay']['top_10_ratio']:.4f}")
    print(f"    Top-50 eigenvalue fraction: {results['qualitative_samples']['rep_eigenvalue_decay']['top_50_ratio']:.4f}")
    print(f"")
    print(f"  GRADIENT SPACE:")
    print(f"    r_eff(95%): {grad_r_eff_95}")
    print(f"    r_eff(99%): {grad_r_eff_99}")
    print(f"    Condition number: {grad_condition:.2e}")
    print(f"    Top-5 eigenvalue fraction: {results['qualitative_samples']['grad_eigenvalue_decay']['top_5_ratio']:.4f}")
    print(f"    Top-10 eigenvalue fraction: {results['qualitative_samples']['grad_eigenvalue_decay']['top_10_ratio']:.4f}")
    print(f"    Top-50 eigenvalue fraction: {results['qualitative_samples']['grad_eigenvalue_decay']['top_50_ratio']:.4f}")
    print(f"")
    print(f"  H4: {h4_note}")
    print(f"  H9: rep_cond={rep_condition:.2e}, grad_cond={grad_condition:.2e}, "
          f"ratio={h9_condition_ratio:.2e}, pass={h9_pass}")
    print(f"  Total runtime: {results['total_runtime_sec']:.1f}s")
    print(f"{'='*60}")

    # Mark done
    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=(f"Eigenspectrum pilot v2: rep_r_eff={rep_r_eff_95}, "
                            f"grad_r_eff={grad_r_eff_95}, "
                            f"rep_cond={rep_condition:.1e}, "
                            f"grad_cond={grad_condition:.1e}, "
                            f"H4_directional={'PASS' if h4_directional else 'FAIL'}, "
                            f"H9_pass={h9_pass}"))

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_task_done(TASK_ID, RESULTS_DIR, status="failed",
                       summary=f"Error: {str(e)}")
        sys.exit(1)
