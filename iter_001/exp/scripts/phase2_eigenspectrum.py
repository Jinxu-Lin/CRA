#!/usr/bin/env python3
"""
Phase 2a: Gradient & Representation Eigenspectrum Analysis (H4, H9)

On Pythia-70M (d=512):
1. Compute gradient covariance eigenspectrum via Gram matrix SVD (efficient for N << D).
2. Compute exact representation covariance eigenspectrum (d=512).
3. Measure r_eff(95%) for both.
4. Compute condition numbers.

Hypotheses:
- H4: r_eff(Sigma_g) in [256, 1024]
- H9: rep condition < 100, grad condition > 10^4

PILOT mode: 100 training samples, seed=42

Efficient approach: For N=100 samples with grad dim D >> N, we collect all N gradients,
form the N x N Gram matrix, and eigendecompose it. The non-zero eigenvalues of the
D x D covariance matrix equal those of the N x N Gram matrix (up to 1/(N-1) scaling).
This requires only N forward+backward passes total.
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
import torch.nn.functional as F
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

# Which layers to compute gradients for
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
    return int(np.searchsorted(cumsum, threshold)) + 1


def compute_condition_number(eigenvalues):
    """Condition number from eigenvalues."""
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

    report_progress(TASK_ID, RESULTS_DIR, 0, 5, metric={"phase": "loading_model"})

    print(f"[{TASK_ID}] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CHECKPOINT_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CHECKPOINT_CACHE, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()

    hidden_dim = model.config.hidden_size  # 512
    n_layers = model.config.num_hidden_layers
    print(f"[{TASK_ID}] Hidden dim: {hidden_dim}, Layers: {n_layers}")

    # ── Load training data ──────────────────────────────────────────────
    report_progress(TASK_ID, RESULTS_DIR, 1, 5, metric={"phase": "loading_data"})

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

    max_len = 256
    encodings = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len)

    # ── Part 1: Representation Covariance Eigenspectrum ─────────────────
    report_progress(TASK_ID, RESULTS_DIR, 2, 5, metric={"phase": "rep_eigenspectrum"})

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

    rep_eigenvalues, _ = np.linalg.eigh(cov_rep)
    rep_eigenvalues = rep_eigenvalues[::-1].copy()  # descending

    rep_r_eff_95 = compute_r_eff(rep_eigenvalues, 0.95)
    rep_r_eff_99 = compute_r_eff(rep_eigenvalues, 0.99)
    rep_condition = compute_condition_number(rep_eigenvalues)

    rep_time = time.time() - rep_start
    rep_total_var = np.abs(rep_eigenvalues).sum()
    rep_cumvar = np.cumsum(np.abs(rep_eigenvalues)) / rep_total_var

    print(f"[{TASK_ID}] Rep eigenspectrum: {rep_time:.1f}s")
    print(f"[{TASK_ID}] Rep r_eff(95%): {rep_r_eff_95}, r_eff(99%): {rep_r_eff_99}")
    print(f"[{TASK_ID}] Rep condition: {rep_condition:.2e}")

    # Also compute per-layer representation eigenspectra for richer analysis
    print(f"\n[{TASK_ID}] Computing per-layer representation eigenspectra...")
    layer_rep_stats = {}
    with torch.no_grad():
        # Re-run to get all hidden states
        all_layer_reps = {l: [] for l in range(n_layers + 1)}  # +1 for embeddings
        for i in range(0, n_samples, batch_size):
            batch_ids = encodings["input_ids"][i:i+batch_size].to(DEVICE)
            batch_mask = encodings["attention_mask"][i:i+batch_size].to(DEVICE)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask,
                          output_hidden_states=True)
            mask_expanded = batch_mask.unsqueeze(-1).float()
            for l_idx, hs in enumerate(outputs.hidden_states):
                pooled = (hs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                all_layer_reps[l_idx].append(pooled.cpu().float())

    for l_idx in range(n_layers + 1):
        lr = torch.cat(all_layer_reps[l_idx], dim=0).numpy()
        lr_centered = lr - lr.mean(axis=0, keepdims=True)
        cov_lr = (lr_centered.T @ lr_centered) / (n_samples - 1)
        eigs_lr, _ = np.linalg.eigh(cov_lr)
        eigs_lr = eigs_lr[::-1].copy()
        r95 = compute_r_eff(eigs_lr, 0.95)
        cond = compute_condition_number(eigs_lr)
        layer_name = "embedding" if l_idx == 0 else f"layer_{l_idx-1}"
        layer_rep_stats[layer_name] = {
            "r_eff_95": int(r95),
            "condition_number": float(cond),
            "top_eigenvalue": float(eigs_lr[0]),
        }
        if l_idx in [0, n_layers]:  # Print only first and last
            print(f"[{TASK_ID}] {layer_name}: r_eff(95%)={r95}, cond={cond:.2e}")

    del all_layer_reps
    gc.collect()

    # ── Part 2: Gradient Covariance Eigenspectrum (Gram matrix) ─────────
    report_progress(TASK_ID, RESULTS_DIR, 3, 5, metric={"phase": "grad_eigenspectrum"})

    print(f"\n[{TASK_ID}] === Part 2: Gradient Covariance Eigenspectrum ===")
    grad_start = time.time()

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
    print(f"[{TASK_ID}] Params: {len(target_params)}, grad dim D={total_grad_dim}")
    print(f"[{TASK_ID}] Sample count N={n_samples} << D={total_grad_dim}")
    print(f"[{TASK_ID}] Strategy: collect N gradients, form N×N Gram matrix, eigendecompose")

    # Collect all per-sample gradients
    # Store on CPU to avoid GPU OOM; only D floats per sample
    grad_matrix = torch.zeros(n_samples, total_grad_dim, dtype=torch.float32)

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
                grads.append(p.grad.detach().cpu().flatten())
            else:
                grads.append(torch.zeros(p.numel()))
        grad_matrix[i] = torch.cat(grads)

        if (i + 1) % 20 == 0:
            print(f"[{TASK_ID}] Gradient collection: {i+1}/{n_samples}")
            report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=i+1, total_steps=n_samples,
                            metric={"phase": "collecting_gradients"})

    grad_collect_time = time.time() - grad_start
    print(f"[{TASK_ID}] Gradient collection: {grad_collect_time:.1f}s")
    print(f"[{TASK_ID}] Gradient matrix shape: {grad_matrix.shape}, "
          f"memory: {grad_matrix.element_size() * grad_matrix.nelement() / 1e6:.1f} MB")

    # Center gradients
    grad_mean = grad_matrix.mean(dim=0)
    grad_centered = grad_matrix - grad_mean.unsqueeze(0)

    # Compute Gram matrix: G = (1/(N-1)) * grad_centered @ grad_centered^T  (N x N)
    # The non-zero eigenvalues of the D x D covariance equal those of G
    print(f"[{TASK_ID}] Computing N×N Gram matrix...")
    G_np = grad_centered.numpy()
    gram = (G_np @ G_np.T) / (n_samples - 1)  # (N, N)
    print(f"[{TASK_ID}] Gram matrix shape: {gram.shape}")

    # Eigendecompose the N x N Gram matrix
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(gram)
    # Sort descending
    gram_eigenvalues = gram_eigenvalues[::-1].copy()

    # These are the top-N eigenvalues of the D x D gradient covariance
    # (remaining D-N eigenvalues are zero)
    grad_eigenvalues = gram_eigenvalues.copy()

    # Filter out near-zero / negative eigenvalues (numerical noise)
    grad_eigenvalues_pos = grad_eigenvalues[grad_eigenvalues > 1e-12]
    n_significant = len(grad_eigenvalues_pos)
    print(f"[{TASK_ID}] Significant gradient eigenvalues: {n_significant}/{len(grad_eigenvalues)}")

    grad_r_eff_95 = compute_r_eff(grad_eigenvalues_pos, 0.95) if n_significant > 0 else n_samples
    grad_r_eff_99 = compute_r_eff(grad_eigenvalues_pos, 0.99) if n_significant > 0 else n_samples
    grad_condition = compute_condition_number(grad_eigenvalues_pos) if n_significant > 1 else float('inf')

    grad_total_var = np.abs(grad_eigenvalues_pos).sum() if n_significant > 0 else 0
    grad_cumvar = np.cumsum(np.abs(grad_eigenvalues_pos)) / grad_total_var if grad_total_var > 0 else np.array([])

    grad_time = time.time() - grad_start
    print(f"[{TASK_ID}] Grad eigenspectrum: {grad_time:.1f}s total")
    print(f"[{TASK_ID}] Grad r_eff(95%): {grad_r_eff_95}, r_eff(99%): {grad_r_eff_99}")
    print(f"[{TASK_ID}] Grad condition: {grad_condition:.2e}")

    # ── Part 2b: Full-parameter gradient eigenspectrum ──────────────────
    # Also compute for ALL layers to get the full picture
    print(f"\n[{TASK_ID}] === Part 2b: Full-model gradient eigenspectrum ===")
    full_grad_start = time.time()

    all_params = [p for p in model.parameters() if p.requires_grad]
    full_grad_dim = sum(p.numel() for p in all_params)
    print(f"[{TASK_ID}] Full model grad dim: {full_grad_dim}")

    # For full model, use the same Gram matrix approach
    full_grad_matrix = torch.zeros(n_samples, full_grad_dim, dtype=torch.float32)

    for i in range(n_samples):
        model.zero_grad()
        ids = encodings["input_ids"][i:i+1].to(DEVICE)
        mask = encodings["attention_mask"][i:i+1].to(DEVICE)
        outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs.loss
        loss.backward()

        grads = []
        for p in all_params:
            if p.grad is not None:
                grads.append(p.grad.detach().cpu().flatten())
            else:
                grads.append(torch.zeros(p.numel()))
        full_grad_matrix[i] = torch.cat(grads)

        if (i + 1) % 20 == 0:
            print(f"[{TASK_ID}] Full gradient collection: {i+1}/{n_samples}")
            report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=n_samples + i + 1,
                            total_steps=2*n_samples,
                            metric={"phase": "collecting_full_gradients"})

    full_grad_centered = full_grad_matrix - full_grad_matrix.mean(dim=0).unsqueeze(0)
    FG_np = full_grad_centered.numpy()
    full_gram = (FG_np @ FG_np.T) / (n_samples - 1)
    full_gram_eigs, _ = np.linalg.eigh(full_gram)
    full_gram_eigs = full_gram_eigs[::-1].copy()
    full_grad_eigs_pos = full_gram_eigs[full_gram_eigs > 1e-12]

    full_r_eff_95 = compute_r_eff(full_grad_eigs_pos, 0.95) if len(full_grad_eigs_pos) > 0 else n_samples
    full_r_eff_99 = compute_r_eff(full_grad_eigs_pos, 0.99) if len(full_grad_eigs_pos) > 0 else n_samples
    full_condition = compute_condition_number(full_grad_eigs_pos) if len(full_grad_eigs_pos) > 1 else float('inf')

    full_grad_time = time.time() - full_grad_start
    print(f"[{TASK_ID}] Full grad eigenspectrum: {full_grad_time:.1f}s")
    print(f"[{TASK_ID}] Full grad r_eff(95%): {full_r_eff_95}, condition: {full_condition:.2e}")

    del full_grad_matrix, full_grad_centered, FG_np, full_gram
    gc.collect()

    # ── Part 3: Hypothesis Testing ──────────────────────────────────────
    report_progress(TASK_ID, RESULTS_DIR, 4, 5, metric={"phase": "hypothesis_testing"})

    print(f"\n[{TASK_ID}] === Hypothesis Testing ===")

    # H4: r_eff(Sigma_g) in [256, 1024] (= [0.5d, 2d] for d=512)
    # IMPORTANT: With N=100 pilot, the covariance has rank <= N-1 = 99.
    # We cannot directly test H4 (which predicts r_eff in [256, 1024]).
    # Instead, we check: is the gradient spectrum concentrated (low r_eff relative to N)?
    # If r_eff << N, gradients have low intrinsic dim -- consistent with H4 direction.
    # Full test requires N >> 1024.
    max_rank = min(n_samples - 1, total_grad_dim)
    h4_r_eff_frac = grad_r_eff_95 / max_rank if max_rank > 0 else 0

    # Extrapolation hint: if r_eff/N is high (close to 1), signal is spread across
    # many directions. If low, signal is concentrated.
    # For full model gradients:
    full_max_rank = min(n_samples - 1, full_grad_dim)
    full_r_eff_frac = full_r_eff_95 / full_max_rank if full_max_rank > 0 else 0

    h4_note = (f"Pilot N={n_samples}: max rank={max_rank}. "
               f"Target-layer r_eff(95%)={grad_r_eff_95} ({h4_r_eff_frac:.1%} of max). "
               f"Full-model r_eff(95%)={full_r_eff_95} ({full_r_eff_frac:.1%} of max). "
               f"Full-scale test needs N >> 1024.")

    # H9: rep condition < 100, grad condition > 10^4
    h9_rep_pass = rep_condition < 100
    h9_grad_pass = grad_condition > 1e4
    h9_pass = h9_rep_pass and h9_grad_pass
    h9_ratio = grad_condition / rep_condition if rep_condition > 0 else float('inf')

    # Also check full-model gradient condition
    h9_full_grad_pass = full_condition > 1e4
    h9_full_ratio = full_condition / rep_condition if rep_condition > 0 else float('inf')

    print(f"[{TASK_ID}] H4: target-layer r_eff={grad_r_eff_95}/{max_rank} ({h4_r_eff_frac:.1%})")
    print(f"[{TASK_ID}]     full-model r_eff={full_r_eff_95}/{full_max_rank} ({full_r_eff_frac:.1%})")
    print(f"[{TASK_ID}] H9 (target layers): rep_cond={rep_condition:.2e}, grad_cond={grad_condition:.2e}")
    print(f"[{TASK_ID}]     ratio={h9_ratio:.2e}, pass={h9_pass}")
    print(f"[{TASK_ID}] H9 (full model): grad_cond={full_condition:.2e}, ratio={h9_full_ratio:.2e}")

    # ── Save Results ────────────────────────────────────────────────────
    report_progress(TASK_ID, RESULTS_DIR, 5, 5, metric={"phase": "saving"})

    # Decay analysis
    def decay_ratios(eigs, total):
        if total < 1e-15 or len(eigs) == 0:
            return {"top_5": 0, "top_10": 0, "top_50": 0}
        return {
            "top_5": float(np.abs(eigs[:5]).sum() / total),
            "top_10": float(np.abs(eigs[:10]).sum() / total),
            "top_50": float(np.abs(eigs[:min(50, len(eigs))]).sum() / total),
        }

    results = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": hidden_dim,
        "n_samples": n_samples,
        "seed": SEED,
        "mode": "pilot",

        "representation": {
            "covariance_shape": [hidden_dim, hidden_dim],
            "eigenvalues": rep_eigenvalues.tolist(),
            "n_eigenvalues": len(rep_eigenvalues),
            "r_eff_95": int(rep_r_eff_95),
            "r_eff_99": int(rep_r_eff_99),
            "condition_number": float(rep_condition),
            "top_10_eigenvalues": rep_eigenvalues[:10].tolist(),
            "decay": decay_ratios(rep_eigenvalues, rep_total_var),
            "runtime_sec": round(rep_time, 2),
        },

        "gradient_target_layers": {
            "target_layers": TARGET_LAYERS,
            "total_grad_dim": total_grad_dim,
            "param_names": target_param_names,
            "eigenvalues": grad_eigenvalues.tolist(),
            "n_significant_eigenvalues": n_significant,
            "r_eff_95": int(grad_r_eff_95),
            "r_eff_99": int(grad_r_eff_99),
            "condition_number": float(grad_condition),
            "top_10_eigenvalues": grad_eigenvalues[:10].tolist(),
            "decay": decay_ratios(grad_eigenvalues_pos, grad_total_var),
            "runtime_sec": round(grad_time, 2),
        },

        "gradient_full_model": {
            "total_grad_dim": full_grad_dim,
            "eigenvalues": full_gram_eigs.tolist(),
            "n_significant_eigenvalues": int(len(full_grad_eigs_pos)),
            "r_eff_95": int(full_r_eff_95),
            "r_eff_99": int(full_r_eff_99),
            "condition_number": float(full_condition),
            "top_10_eigenvalues": full_gram_eigs[:10].tolist(),
            "decay": decay_ratios(full_grad_eigs_pos,
                                  np.abs(full_grad_eigs_pos).sum() if len(full_grad_eigs_pos) > 0 else 0),
            "runtime_sec": round(full_grad_time, 2),
        },

        "layer_rep_stats": layer_rep_stats,

        "hypotheses": {
            "H4": {
                "description": "r_eff(Sigma_g) in [0.5d, 2d] = [256, 1024] for d=512",
                "target_layer_r_eff_95": int(grad_r_eff_95),
                "full_model_r_eff_95": int(full_r_eff_95),
                "target_range": [256, 1024],
                "max_rank_pilot": max_rank,
                "r_eff_fraction_target": round(h4_r_eff_frac, 4),
                "r_eff_fraction_full": round(full_r_eff_frac, 4),
                "pilot_limitation": "N=100 caps rank at 99; cannot directly test [256,1024] range",
                "directional_evidence": "Check if gradient spectrum is more concentrated than representation spectrum",
                "grad_decay_top5": decay_ratios(grad_eigenvalues_pos, grad_total_var).get("top_5", 0),
                "rep_decay_top5": decay_ratios(rep_eigenvalues, rep_total_var).get("top_5", 0),
            },
            "H9": {
                "description": "rep condition < 100, grad condition > 10^4",
                "rep_condition": float(rep_condition),
                "target_layer_grad_condition": float(grad_condition),
                "full_model_grad_condition": float(full_condition),
                "condition_ratio_target": float(h9_ratio),
                "condition_ratio_full": float(h9_full_ratio),
                "rep_pass": h9_rep_pass,
                "grad_pass_target": h9_grad_pass,
                "grad_pass_full": h9_full_grad_pass,
                "pass": h9_pass,
            },
        },

        "spectral_comparison": {
            "rep_r_eff_95": int(rep_r_eff_95),
            "grad_target_r_eff_95": int(grad_r_eff_95),
            "grad_full_r_eff_95": int(full_r_eff_95),
            "rep_condition": float(rep_condition),
            "grad_target_condition": float(grad_condition),
            "grad_full_condition": float(full_condition),
            "condition_ratio_target": float(h9_ratio),
            "condition_ratio_full": float(h9_full_ratio),
        },

        "total_runtime_sec": round(time.time() - start_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    out_path = Path(PHASE2_DIR) / "eigenspectrum.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{TASK_ID}] Results saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EIGENSPECTRUM PILOT SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_NAME} (d={hidden_dim})")
    print(f"  Samples: {n_samples}, Seed: {SEED}")
    print()
    print(f"  REPRESENTATION (last layer, d={hidden_dim}):")
    print(f"    r_eff(95%): {rep_r_eff_95}/{hidden_dim}")
    print(f"    Condition: {rep_condition:.2e}")
    print(f"    Top-5 variance: {decay_ratios(rep_eigenvalues, rep_total_var)['top_5']:.1%}")
    print()
    print(f"  GRADIENT (layers {TARGET_LAYERS}, D={total_grad_dim}):")
    print(f"    r_eff(95%): {grad_r_eff_95}/{max_rank} (pilot-bounded)")
    print(f"    Condition: {grad_condition:.2e}")
    print(f"    Top-5 variance: {decay_ratios(grad_eigenvalues_pos, grad_total_var)['top_5']:.1%}")
    print()
    print(f"  GRADIENT (full model, D={full_grad_dim}):")
    print(f"    r_eff(95%): {full_r_eff_95}/{full_max_rank} (pilot-bounded)")
    print(f"    Condition: {full_condition:.2e}")
    print()
    print(f"  H4: Cannot directly test with N={n_samples} (need N>>1024)")
    print(f"      Gradient concentration: top-5 explains "
          f"{decay_ratios(grad_eigenvalues_pos, grad_total_var)['top_5']:.1%} of variance")
    print(f"  H9: rep_cond={rep_condition:.2e} (<100? {h9_rep_pass})")
    print(f"      grad_cond={grad_condition:.2e} (>10^4? {h9_grad_pass})")
    print(f"      Ratio: {h9_ratio:.2e} (target layers), {h9_full_ratio:.2e} (full model)")
    print(f"  Runtime: {results['total_runtime_sec']:.1f}s")
    print(f"{'='*60}")

    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=(f"rep_r_eff95={rep_r_eff_95} cond={rep_condition:.1e} | "
                            f"grad_r_eff95={grad_r_eff_95} cond={grad_condition:.1e} | "
                            f"full_r_eff95={full_r_eff_95} cond={full_condition:.1e} | "
                            f"H9={h9_pass}"))
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_task_done(TASK_ID, RESULTS_DIR, status="failed", summary=str(e))
        sys.exit(1)
