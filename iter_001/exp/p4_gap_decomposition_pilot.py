#!/usr/bin/env python3
"""
P4: Gap Decomposition -- TRAK-PCA vs RepSim (H10) -- PILOT
============================================================
Systematically decompose the gap between TRAK-PCA at k=d and RepSim.

Factors:
  (a) Last-layer-only TRAK-PCA: restrict gradients to final transformer layer only
  (b) Cosine-normalized TRAK-PCA: L2-normalize gradient vectors before projection
  (c) Combined: last-layer-only + cosine-normalized
  (d) Residual: whatever gap remains

All on Pythia-1B, counterfact task, N=100 (pilot), seed=42, GPU=cuda:0.

OPTIMIZATION: Uses Gram-matrix PCA (N x N instead of D x D) to avoid OOM on
high-dimensional gradient matrices (D ~ 21M params).
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
from scipy.stats import kendalltau, spearmanr

# ── Configuration ──────────────────────────────────────────────────
TASK_ID = "p4_gap_decomposition"
CHECKPOINT_DIR = "EleutherAI/pythia-1b"
SEED = 42
PILOT_N_TRAIN = 100
BOOTSTRAP_B = 1000
DEVICE = "cuda:0"
MAX_LEN = 512

# Paths
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/CRA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
CACHE_DIR = RESULTS_DIR / "cache"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"

for d in [CACHE_DIR, PILOTS_DIR, FULL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Lifecycle helpers ──────────────────────────────────────────────
def _safe_write(path, content):
    try:
        Path(path).write_text(content)
    except OSError as e:
        print(f"[Warn] Cannot write {path}: {e}", flush=True)

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
_safe_write(pid_file, str(os.getpid()))


def report_progress(stage, detail="", pct=0.0, metric=None):
    _safe_write(
        RESULTS_DIR / f"{TASK_ID}_PROGRESS.json",
        json.dumps({
            "task_id": TASK_ID, "epoch": 0, "total_epochs": 1,
            "step": 0, "total_steps": 0, "loss": None,
            "metric": metric or {}, "stage": stage, "detail": detail,
            "pct": pct, "updated_at": datetime.now().isoformat(),
        })
    )


def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        try:
            pid_f.unlink()
        except OSError:
            pass
    fp = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final = {}
    if fp.exists():
        try:
            final = json.loads(fp.read_text())
        except Exception:
            pass
    _safe_write(
        RESULTS_DIR / f"{TASK_ID}_DONE",
        json.dumps({
            "task_id": TASK_ID, "status": status, "summary": summary,
            "final_progress": final,
            "timestamp": datetime.now().isoformat(),
        })
    )


# ── Evaluation helpers ─────────────────────────────────────────────
def compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50):
    recalls, mrrs = [], []
    for scores, fi in zip(scores_per_ref, fact_indices_per_ref):
        if not fi:
            continue
        si = np.argsort(-np.array(scores))
        topk = set(si[:k].tolist())
        recalls.append(len([i for i in fi if i in topk]) / len(fi))
        r2r = {idx: rank + 1 for rank, idx in enumerate(si)}
        ranks = [r2r[i] for i in fi if i in r2r]
        mrrs.append(1.0 / min(ranks) if ranks else 0.0)
    if not recalls:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(mrrs))


def compute_continuous_metrics(scores_per_ref, fact_indices_per_ref, n_train):
    taus, rhos = [], []
    for scores, fi in zip(scores_per_ref, fact_indices_per_ref):
        if not fi:
            continue
        binary_rel = np.zeros(n_train)
        binary_rel[fi] = 1.0
        s = np.array(scores)
        if np.std(s) < 1e-12:
            continue
        tau, _ = kendalltau(s, binary_rel)
        rho, _ = spearmanr(s, binary_rel)
        if not np.isnan(tau):
            taus.append(tau)
        if not np.isnan(rho):
            rhos.append(rho)
    return (float(np.mean(taus)) if taus else 0.0,
            float(np.mean(rhos)) if rhos else 0.0)


def evaluate_counterfact(scores_matrix, pilot_data, ref_data, n_train):
    """scores_matrix: [n_train, n_ref]"""
    n_ref = len(ref_data)
    scores_per_ref = []
    fact_indices_per_ref = []

    for j in range(n_ref):
        ref_sample = ref_data[j]
        scores_j = scores_matrix[:, j].tolist()
        fi = [
            i for i in range(n_train)
            if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
            and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
        ]
        scores_per_ref.append(scores_j)
        fact_indices_per_ref.append(fi)

    recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)
    tau, rho = compute_continuous_metrics(scores_per_ref, fact_indices_per_ref, n_train)

    rng_boot = np.random.RandomState(SEED + 2345)
    boot_recalls, boot_mrrs, boot_taus = [], [], []
    for _ in range(BOOTSTRAP_B):
        idx = rng_boot.choice(n_ref, n_ref, replace=True)
        boot_spr = [scores_per_ref[i] for i in idx]
        boot_fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(boot_spr, boot_fi, k=50)
        t, _ = compute_continuous_metrics(boot_spr, boot_fi, n_train)
        boot_recalls.append(r)
        boot_mrrs.append(m)
        boot_taus.append(t)

    n_with_facts = sum(1 for f in fact_indices_per_ref if f)
    return {
        "Recall@50": round(recall, 6),
        "Recall@50_CI": [round(float(np.percentile(boot_recalls, 2.5)), 6),
                         round(float(np.percentile(boot_recalls, 97.5)), 6)],
        "MRR": round(mrr, 6),
        "MRR_CI": [round(float(np.percentile(boot_mrrs, 2.5)), 6),
                    round(float(np.percentile(boot_mrrs, 97.5)), 6)],
        "kendall_tau": round(tau, 6),
        "kendall_tau_CI": [round(float(np.percentile(boot_taus, 2.5)), 6),
                           round(float(np.percentile(boot_taus, 97.5)), 6)],
        "spearman_rho": round(rho, 6),
        "refs_with_facts": n_with_facts,
        "n_ref": n_ref,
        "n_train": n_train,
    }


# ── Efficient Gram-matrix PCA for TRAK ────────────────────────────
def trak_pca_scores_gram(train_G, ref_train_G, k):
    """Compute TRAK-PCA scores via Gram matrix (avoids materializing D-dim projections).

    Args:
        train_G: [N_train, N_train] Gram matrix of centered train gradients
        ref_train_G: [N_ref, N_train] cross-Gram of centered ref vs train gradients
        k: number of PCA components

    The PCA projection score matrix is:
        S = ref_proj @ train_proj.T
    where train_proj = U_k * sqrt(Lambda_k), ref_proj = ref_centered @ V_k
    and V_k = X_centered.T @ U_k / sqrt(Lambda_k)

    So S = (ref_centered @ V_k) @ (U_k * sqrt(Lambda_k)).T
         = ref_centered @ X_centered.T @ U_k / sqrt(Lambda_k) @ (U_k * sqrt(Lambda_k)).T
         = ref_train_G @ U_k @ U_k.T

    This only requires the N_train x N_train eigendecomposition!

    Returns: scores [N_train, N_ref], actual_k, eigenvalues
    """
    # Eigendecompose Gram matrix
    eigenvalues, U = torch.linalg.eigh(train_G)
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]

    actual_k = min(k, U.shape[1])
    U_k = U[:, :actual_k]

    # Scores: [N_ref, N_train] = ref_train_G @ U_k @ U_k.T
    scores_ref_train = (ref_train_G @ U_k) @ U_k.T  # [N_ref, N_train]

    # Return as [N_train, N_ref] for consistency
    return scores_ref_train.T.numpy(), actual_k, eigenvalues[:actual_k].numpy()


# ── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Step 1: Load data ─────────────────────────────────────────────
    report_progress("loading_data", "Loading counterfact dataset", 0.05)
    print("[1/7] Loading DATE-LM counterfact data...", flush=True)

    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    fmt = lambda s: s["prompt"] + " " + s["response"]
    train_data_full = cf["train"]
    ref_data = cf["ref"]
    print(f"  Full: train={len(train_data_full)}, ref={len(ref_data)}", flush=True)

    rng = np.random.RandomState(SEED)
    n_total = len(train_data_full)
    pilot_idx = sorted(rng.choice(n_total, min(PILOT_N_TRAIN, n_total), replace=False).tolist())
    pilot_data = train_data_full.select(pilot_idx)
    n_train = len(pilot_data)
    n_ref = len(ref_data)
    print(f"  Pilot: train={n_train}, ref={n_ref}", flush=True)

    train_texts = [fmt(pilot_data[i]) for i in range(n_train)]
    ref_texts = [fmt(ref_data[i]) for i in range(n_ref)]

    # ── Step 2: Load model ────────────────────────────────────────────
    report_progress("loading_model", "Loading Pythia-1B", 0.10)
    print("[2/7] Loading Pythia-1B model...", flush=True)

    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    last_layer_idx = num_layers - 1
    print(f"  num_layers={num_layers}, hidden_dim={hidden_dim}, last_layer={last_layer_idx}", flush=True)

    # Layer patterns
    STANDARD_PATTERNS = [
        "layers.15.attention.dense.weight",
        "layers.15.mlp.dense_4h_to_h.weight",
    ]
    LASTLAYER_PATTERNS = [f"layers.{last_layer_idx}."]

    def get_target_params(patterns):
        params, names = [], []
        for name, p in model.named_parameters():
            if any(pat in name for pat in patterns):
                params.append(p)
                names.append(name)
        return params, names

    std_params, std_names = get_target_params(STANDARD_PATTERNS)
    ll_params, ll_names = get_target_params(LASTLAYER_PATTERNS)

    D_standard = sum(p.numel() for p in std_params)
    D_lastlayer = sum(p.numel() for p in ll_params)
    print(f"  Standard target: {std_names}, D={D_standard/1e6:.2f}M", flush=True)
    print(f"  Last-layer: {len(ll_names)} params, D={D_lastlayer/1e6:.2f}M", flush=True)

    # ── Step 3: Compute Gram matrices (memory-efficient) ──────────────
    # Instead of storing full [N, D] gradient matrices (e.g. 100 x 21M = 8GB each),
    # we compute Gram matrices [N, N] incrementally, one gradient at a time.
    report_progress("computing_grams", "Computing Gram matrices incrementally", 0.15)
    print("[3/7] Computing Gram matrices incrementally...", flush=True)

    def compute_gram_and_norms_incrementally(train_texts, ref_texts, target_params_list, desc=""):
        """Compute train-train Gram, ref-train cross-Gram, and gradient norms
        WITHOUT storing full gradient matrices.

        Returns:
            train_gram: [n_train, n_train] -- g_train @ g_train.T
            ref_train_gram: [n_ref, n_train] -- g_ref @ g_train.T
            train_norms: [n_train] -- ||g_i||
            ref_norms: [n_ref] -- ||g_j||
        """
        D = sum(p.numel() for p in target_params_list)
        nt = len(train_texts)
        nr = len(ref_texts)

        # Set requires_grad
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        for p in target_params_list:
            p.requires_grad_(True)

        # Phase 1: Extract train gradients (store on CPU in float32)
        # We DO need to store train grads to compute cross-Gram with ref.
        # But we use float32 which is 4 bytes per param.
        # For D=21M, 100 samples = 8.4 GB -- acceptable.
        print(f"    [{desc}] Phase 1: Extracting {nt} train gradients (D={D/1e6:.1f}M)...", flush=True)
        train_grads = torch.zeros(nt, D, dtype=torch.float32)

        for idx, text in enumerate(train_texts):
            inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            model.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                            labels=inp["input_ids"])
            out.loss.backward()

            offset = 0
            for p in target_params_list:
                g = p.grad.detach().flatten().float().cpu()
                train_grads[idx, offset:offset + g.shape[0]] = g
                offset += g.shape[0]
            model.zero_grad(set_to_none=True)

            if (idx + 1) % 25 == 0:
                torch.cuda.empty_cache()
                print(f"    [{desc}] train {idx+1}/{nt}", flush=True)

        # Compute train norms
        train_norms = train_grads.norm(dim=1)  # [nt]

        # Compute train Gram matrix
        print(f"    [{desc}] Computing train Gram matrix ({nt}x{nt})...", flush=True)
        train_gram = train_grads @ train_grads.T  # [nt, nt]

        # Phase 2: Compute ref-train cross-Gram incrementally
        print(f"    [{desc}] Phase 2: Computing ref-train cross-Gram ({nr} ref samples)...", flush=True)
        ref_train_gram = torch.zeros(nr, nt, dtype=torch.float32)
        ref_norms = torch.zeros(nr, dtype=torch.float32)

        for idx, text in enumerate(ref_texts):
            inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            model.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                            labels=inp["input_ids"])
            out.loss.backward()

            ref_grad = torch.zeros(D, dtype=torch.float32)
            offset = 0
            for p in target_params_list:
                g = p.grad.detach().flatten().float().cpu()
                ref_grad[offset:offset + g.shape[0]] = g
                offset += g.shape[0]
            model.zero_grad(set_to_none=True)

            ref_norms[idx] = ref_grad.norm()
            ref_train_gram[idx] = ref_grad @ train_grads.T  # [nt]

            if (idx + 1) % 25 == 0:
                torch.cuda.empty_cache()
                print(f"    [{desc}] ref {idx+1}/{nr}", flush=True)

        print(f"    [{desc}] Done. train_gram={train_gram.shape}, cross_gram={ref_train_gram.shape}", flush=True)
        return train_gram, ref_train_gram, train_norms, ref_norms, train_grads

    # Standard TRAK gradients
    t0 = time.time()
    std_train_gram, std_ref_train_gram, std_train_norms, std_ref_norms, std_train_grads = \
        compute_gram_and_norms_incrementally(train_texts, ref_texts, std_params, desc="standard")
    print(f"  Standard gradients: {time.time()-t0:.0f}s", flush=True)

    # Last-layer gradients
    report_progress("computing_grams", "Last-layer gradients", 0.35)
    t0 = time.time()
    ll_train_gram, ll_ref_train_gram, ll_train_norms, ll_ref_norms, ll_train_grads = \
        compute_gram_and_norms_incrementally(train_texts, ref_texts, ll_params, desc="lastlayer")
    print(f"  Last-layer gradients: {time.time()-t0:.0f}s", flush=True)

    # ── Step 4: Compute cosine-normalized Gram matrices ──────────────
    report_progress("cosine_grams", "Computing cosine-normalized Gram matrices", 0.50)
    print("[4/7] Computing cosine-normalized Gram matrices...", flush=True)

    def cosine_gram(train_grads, train_norms, ref_train_gram_raw, ref_norms):
        """Compute Gram matrices for cosine-normalized gradients.
        cosine_gram_ij = g_i.g_j / (||g_i|| * ||g_j||)
        """
        nt = train_grads.shape[0]
        # Train-train cosine Gram
        norm_outer = train_norms.unsqueeze(1) * train_norms.unsqueeze(0)  # [nt, nt]
        norm_outer = norm_outer.clamp(min=1e-8)
        train_gram_cos = (train_grads @ train_grads.T) / norm_outer

        # Ref-train cosine cross-Gram
        ref_norm_col = ref_norms.unsqueeze(1)  # [nr, 1]
        train_norm_row = train_norms.unsqueeze(0)  # [1, nt]
        denom = (ref_norm_col * train_norm_row).clamp(min=1e-8)
        ref_train_gram_cos = ref_train_gram_raw / denom

        return train_gram_cos, ref_train_gram_cos

    std_train_gram_cos, std_ref_train_gram_cos = cosine_gram(
        std_train_grads, std_train_norms, std_ref_train_gram, std_ref_norms
    )
    ll_train_gram_cos, ll_ref_train_gram_cos = cosine_gram(
        ll_train_grads, ll_train_norms, ll_ref_train_gram, ll_ref_norms
    )

    # Free large gradient matrices
    del std_train_grads, ll_train_grads
    gc.collect()

    # ── Step 5: Extract representations for RepSim ────────────────────
    report_progress("extracting_reps", "Extracting last-layer representations", 0.55)
    print("[5/7] Extracting last-layer representations for RepSim...", flush=True)

    for name, p in model.named_parameters():
        p.requires_grad_(False)

    def extract_reps(texts, desc=""):
        reps = []
        with torch.no_grad():
            for idx, text in enumerate(texts):
                inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
                out = model(**inp, output_hidden_states=True)
                last_h = out.hidden_states[-1][0, -1, :].float().cpu()
                reps.append(last_h)
                if (idx + 1) % 50 == 0:
                    print(f"  [{desc}] {idx+1}/{len(texts)}", flush=True)
        return torch.stack(reps)

    train_reps = extract_reps(train_texts, desc="train_rep")
    ref_reps = extract_reps(ref_texts, desc="ref_rep")
    print(f"  Representations: train={train_reps.shape}, ref={ref_reps.shape}", flush=True)

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 6: Compute PCA scores for all configs ────────────────────
    report_progress("computing_scores", "Computing PCA attribution scores", 0.65)
    print("[6/7] Computing PCA attribution scores...", flush=True)

    k = hidden_dim  # 2048 (or capped by rank)

    def centered_grams(train_gram, ref_train_gram):
        """Center Gram matrices (subtract train mean)."""
        nt = train_gram.shape[0]
        # Centering: G_centered = H @ G @ H where H = I - 1/n * 11^T
        # More efficient: g_centered_i = g_i - mean(g)
        # Gram(centered) = G - (1/n)*G@1*1.T - (1/n)*1*1.T@G + (1/n^2)*1.T@G@1*1*1.T
        row_means = train_gram.mean(dim=1, keepdim=True)  # [nt, 1]
        col_means = train_gram.mean(dim=0, keepdim=True)  # [1, nt]
        total_mean = train_gram.mean()
        train_gram_c = train_gram - row_means - col_means + total_mean

        # For cross-Gram: ref_centered @ train_centered.T
        # = (ref - train_mean) @ (train - train_mean).T
        # = ref @ train.T - train_mean @ train.T - ref @ train_mean.T + train_mean @ train_mean.T
        ref_row = ref_train_gram  # [nr, nt]
        train_col_means = ref_train_gram.mean(dim=0, keepdim=True)  # mean over ref dim -> not right
        # Actually: train_mean = (1/nt) * sum(g_train_i)
        # ref_centered @ train_centered.T
        #   = ref @ train.T - (1/nt) * (sum_j g_train_j) @ train.T
        #     - ref @ (1/nt) * sum_j g_train_j.T + (1/nt)^2 * sum_j g_j @ sum_k g_k.T
        #
        # In Gram terms:
        # train_mean contribution: (1/nt) * 1 @ train_gram  (summing rows of train_gram)
        # For ref: (1/nt) * ref_train_gram @ 1
        train_gram_col_sum = train_gram.sum(dim=0, keepdim=True) / nt  # [1, nt]: mean train gram row
        ref_gram_row_mean = ref_train_gram.mean(dim=1, keepdim=True)  # [nr, 1]: mean over train for each ref
        total_mean_cross = train_gram.sum() / (nt * nt)

        ref_train_gram_c = ref_train_gram - ref_gram_row_mean - train_gram_col_sum + total_mean_cross

        return train_gram_c, ref_train_gram_c

    configs_results = {}

    # (1) Standard TRAK-PCA (layer 15 attn+mlp, unnormalized)
    print("  Config 1: Standard TRAK-PCA...", flush=True)
    std_train_gram_c, std_ref_train_gram_c = centered_grams(std_train_gram, std_ref_train_gram)
    scores1, k1, eigs1 = trak_pca_scores_gram(std_train_gram_c, std_ref_train_gram_c, k)
    configs_results["standard_trak_pca"] = {
        "scores": scores1, "actual_k": k1,
        "description": "Standard TRAK-PCA (layer 15 attn+mlp, k=d, no normalization)",
        "factor": "baseline", "grad_dim": int(D_standard),
    }
    print(f"    actual_k={k1}", flush=True)

    # (2) Last-layer-only TRAK-PCA (all layer-15 params, unnormalized)
    print("  Config 2: Last-layer-only TRAK-PCA...", flush=True)
    ll_train_gram_c, ll_ref_train_gram_c = centered_grams(ll_train_gram, ll_ref_train_gram)
    scores2, k2, eigs2 = trak_pca_scores_gram(ll_train_gram_c, ll_ref_train_gram_c, k)
    configs_results["lastlayer_trak_pca"] = {
        "scores": scores2, "actual_k": k2,
        "description": f"Last-layer-only TRAK-PCA (all layer {last_layer_idx} params, k=d)",
        "factor": "a_layer_selection", "grad_dim": int(D_lastlayer),
    }
    print(f"    actual_k={k2}", flush=True)

    # (3) Cosine-normalized TRAK-PCA (standard layers)
    print("  Config 3: Cosine-normalized TRAK-PCA...", flush=True)
    std_cos_train_c, std_cos_ref_c = centered_grams(std_train_gram_cos, std_ref_train_gram_cos)
    scores3, k3, eigs3 = trak_pca_scores_gram(std_cos_train_c, std_cos_ref_c, k)
    configs_results["cosine_trak_pca"] = {
        "scores": scores3, "actual_k": k3,
        "description": "Cosine-normalized TRAK-PCA (layer 15, L2-normalized, k=d)",
        "factor": "b_cosine_normalization", "grad_dim": int(D_standard),
    }
    print(f"    actual_k={k3}", flush=True)

    # (4) Combined: last-layer + cosine-normalized
    print("  Config 4: Combined (last-layer + cosine-normalized)...", flush=True)
    ll_cos_train_c, ll_cos_ref_c = centered_grams(ll_train_gram_cos, ll_ref_train_gram_cos)
    scores4, k4, eigs4 = trak_pca_scores_gram(ll_cos_train_c, ll_cos_ref_c, k)
    configs_results["combined_trak_pca"] = {
        "scores": scores4, "actual_k": k4,
        "description": f"Combined: last-layer + cosine-norm (layer {last_layer_idx}, L2-normalized)",
        "factor": "c_combined", "grad_dim": int(D_lastlayer),
    }
    print(f"    actual_k={k4}", flush=True)

    # (5) RepSim reference (cosine similarity in representation space)
    print("  Config 5: RepSim...", flush=True)
    tn = train_reps / train_reps.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rn = ref_reps / ref_reps.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scores5 = (tn @ rn.T).numpy()  # [n_train, n_ref]
    configs_results["repsim"] = {
        "scores": scores5, "actual_k": int(hidden_dim),
        "description": f"RepSim (last-layer cosine similarity, d={hidden_dim})",
        "factor": "reference", "grad_dim": int(hidden_dim),
    }

    # (6) Bonus: raw dot product TRAK (no PCA, just centered inner product)
    # This gives us the "no projection" baseline
    print("  Config 6: Raw dot product (standard, no PCA)...", flush=True)
    scores6 = std_ref_train_gram_c.T.numpy()  # [n_train, n_ref]
    configs_results["raw_dot_trak"] = {
        "scores": scores6, "actual_k": int(D_standard),
        "description": "Raw dot product TRAK (layer 15, centered, no projection)",
        "factor": "no_projection_baseline", "grad_dim": int(D_standard),
    }

    # ── Step 7: Evaluate & analyze ────────────────────────────────────
    report_progress("evaluating", "Evaluating all configurations", 0.75)
    print("[7/7] Evaluating all configurations...", flush=True)

    results = {}
    for cname, cfg in configs_results.items():
        print(f"  Evaluating {cname}...", flush=True)
        metrics = evaluate_counterfact(cfg["scores"], pilot_data, ref_data, n_train)
        metrics["description"] = cfg["description"]
        metrics["factor"] = cfg["factor"]
        metrics["actual_k"] = cfg["actual_k"]
        metrics["grad_dim"] = cfg["grad_dim"]
        results[cname] = metrics
        print(f"    R@50={metrics['Recall@50']:.4f}, MRR={metrics['MRR']:.4f}, "
              f"tau={metrics['kendall_tau']:.4f}", flush=True)

    # ── Gap decomposition analysis ─────────────────────────────────
    report_progress("gap_analysis", "Computing gap decomposition", 0.90)
    print("\n=== Gap Decomposition Analysis ===", flush=True)

    repsim_r50 = results["repsim"]["Recall@50"]
    baseline_r50 = results["standard_trak_pca"]["Recall@50"]
    total_gap_pp = (repsim_r50 - baseline_r50) * 100

    factor_a_r50 = results["lastlayer_trak_pca"]["Recall@50"]
    factor_b_r50 = results["cosine_trak_pca"]["Recall@50"]
    factor_c_r50 = results["combined_trak_pca"]["Recall@50"]

    gap_a = (factor_a_r50 - baseline_r50) * 100
    gap_b = (factor_b_r50 - baseline_r50) * 100
    gap_c = (factor_c_r50 - baseline_r50) * 100
    residual_gap = (repsim_r50 - factor_c_r50) * 100

    print(f"  RepSim R@50:              {repsim_r50:.4f}", flush=True)
    print(f"  Standard TRAK-PCA R@50:   {baseline_r50:.4f}", flush=True)
    print(f"  Total gap:                {total_gap_pp:.1f}pp", flush=True)
    print(f"  (a) Last-layer-only:      R@50={factor_a_r50:.4f}, gap reduction={gap_a:+.1f}pp", flush=True)
    print(f"  (b) Cosine-normalized:    R@50={factor_b_r50:.4f}, gap reduction={gap_b:+.1f}pp", flush=True)
    print(f"  (c) Combined (a+b):       R@50={factor_c_r50:.4f}, gap reduction={gap_c:+.1f}pp", flush=True)
    print(f"  (d) Residual:             {residual_gap:.1f}pp", flush=True)

    # Kendall tau analysis
    repsim_tau = results["repsim"]["kendall_tau"]
    baseline_tau = results["standard_trak_pca"]["kendall_tau"]
    total_gap_tau = repsim_tau - baseline_tau
    gap_a_tau = results["lastlayer_trak_pca"]["kendall_tau"] - baseline_tau
    gap_b_tau = results["cosine_trak_pca"]["kendall_tau"] - baseline_tau
    gap_c_tau = results["combined_trak_pca"]["kendall_tau"] - baseline_tau
    residual_tau = repsim_tau - results["combined_trak_pca"]["kendall_tau"]

    # Gradient norm statistics
    grad_stats = {
        "standard": {
            "grad_dim": int(D_standard),
            "mean_norm": float(std_train_norms.mean()),
            "std_norm": float(std_train_norms.std()),
            "cv_norm": float(std_train_norms.std() / std_train_norms.mean()) if std_train_norms.mean() > 0 else 0,
        },
        "lastlayer": {
            "grad_dim": int(D_lastlayer),
            "mean_norm": float(ll_train_norms.mean()),
            "std_norm": float(ll_train_norms.std()),
            "cv_norm": float(ll_train_norms.std() / ll_train_norms.mean()) if ll_train_norms.mean() > 0 else 0,
        },
    }

    # Pass criteria
    pass_a_or_b = gap_a >= 5 or gap_b >= 5
    pass_combined = gap_c >= 10
    overall_pass = pass_a_or_b and pass_combined

    print(f"\n  Pass criteria:", flush=True)
    print(f"    Factor (a) or (b) >= 5pp: {'PASS' if pass_a_or_b else 'FAIL'} (a={gap_a:+.1f}pp, b={gap_b:+.1f}pp)", flush=True)
    print(f"    Combined >= 10pp:         {'PASS' if pass_combined else 'FAIL'} ({gap_c:+.1f}pp)", flush=True)
    print(f"    Overall:                  {'GO' if overall_pass else 'REFINE'}", flush=True)

    elapsed = time.time() - start_time

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "candidate_id": "cand_a",
        "mode": "pilot",
        "model": CHECKPOINT_DIR,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "task": "counterfact",
        "n_train": n_train,
        "n_ref": n_ref,
        "seed": SEED,
        "bootstrap_B": BOOTSTRAP_B,
        "target_layers_standard": std_names,
        "target_layers_lastlayer": ll_names,
        "D_standard": int(D_standard),
        "D_lastlayer": int(D_lastlayer),
        "configurations": {
            name: {k_: v for k_, v in res.items() if k_ != "scores"}
            for name, res in results.items()
        },
        "gap_decomposition": {
            "repsim_R50": float(repsim_r50),
            "baseline_R50": float(baseline_r50),
            "total_gap_pp": float(total_gap_pp),
            "factor_a_layer_selection": {
                "R50": float(factor_a_r50),
                "gap_reduction_pp": float(gap_a),
                "description": "Restrict gradients to ALL params in final transformer layer",
            },
            "factor_b_cosine_norm": {
                "R50": float(factor_b_r50),
                "gap_reduction_pp": float(gap_b),
                "description": "L2-normalize gradient vectors before PCA projection",
            },
            "factor_c_combined": {
                "R50": float(factor_c_r50),
                "gap_reduction_pp": float(gap_c),
                "description": "Last-layer-only + cosine-normalized",
            },
            "factor_d_residual": {
                "gap_pp": float(residual_gap),
                "description": "Remaining gap: nonlinear semantic features / method architecture",
            },
        },
        "gap_decomposition_tau": {
            "repsim_tau": float(repsim_tau),
            "baseline_tau": float(baseline_tau),
            "total_gap_tau": float(total_gap_tau),
            "factor_a_tau": float(gap_a_tau),
            "factor_b_tau": float(gap_b_tau),
            "factor_c_tau": float(gap_c_tau),
            "residual_tau": float(residual_tau),
        },
        "gradient_statistics": grad_stats,
        "pass_criteria": {
            "factor_a_or_b_ge_5pp": bool(pass_a_or_b),
            "combined_ge_10pp": bool(pass_combined),
            "overall_pass": bool(overall_pass),
        },
        "runtime_sec": round(elapsed, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    output_path = PILOTS_DIR / f"{TASK_ID}_pilot_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}", flush=True)

    # Pilot summary JSON
    pilot_summary = {
        "overall_recommendation": "GO" if overall_pass else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [
            {
                "candidate_id": "cand_a",
                "go_no_go": "GO" if overall_pass else "REFINE",
                "confidence": 0.7 if overall_pass else 0.5,
                "supported_hypotheses": ["H10"] if overall_pass else [],
                "failed_assumptions": [] if overall_pass else ["H10_gap_decomposition_partial"],
                "key_metrics": {
                    "total_gap_pp": float(total_gap_pp),
                    "factor_a_gap_reduction_pp": float(gap_a),
                    "factor_b_gap_reduction_pp": float(gap_b),
                    "factor_c_gap_reduction_pp": float(gap_c),
                    "residual_gap_pp": float(residual_gap),
                    "repsim_R50": float(repsim_r50),
                    "baseline_R50": float(baseline_r50),
                },
                "notes": (
                    f"Total gap: {total_gap_pp:.1f}pp. "
                    f"Factor (a) layer selection: {gap_a:+.1f}pp. "
                    f"Factor (b) cosine norm: {gap_b:+.1f}pp. "
                    f"Factor (c) combined: {gap_c:+.1f}pp. "
                    f"Residual: {residual_gap:.1f}pp. "
                    f"Pilot N={n_train}, PCA rank capped at {n_train}."
                ),
            }
        ],
        "pilot_limitations": [
            f"N={n_train} caps PCA rank at {n_train}",
            "Gradient norm distribution may differ at full scale (N=5473)",
            f"Standard uses D={D_standard:,} params; last-layer uses D={D_lastlayer:,}",
        ],
    }
    summary_path = PILOTS_DIR / f"{TASK_ID}_pilot_summary.json"
    summary_path.write_text(json.dumps(pilot_summary, indent=2))
    print(f"Pilot summary JSON saved to {summary_path}", flush=True)

    # Pilot summary markdown
    md_lines = [
        f"# P4 Gap Decomposition Pilot Summary",
        f"",
        f"**Task**: {TASK_ID}",
        f"**Model**: {CHECKPOINT_DIR} (hidden_dim={hidden_dim}, num_layers={num_layers})",
        f"**Mode**: PILOT (N={n_train})",
        f"**Runtime**: {elapsed:.0f}s",
        f"**Date**: {datetime.now().isoformat()[:10]}",
        f"",
        f"## Gap Decomposition Results (Recall@50)",
        f"",
        f"| Configuration | R@50 | Gap to RepSim (pp) | Gap Reduction (pp) | Kendall tau |",
        f"|---|---|---|---|---|",
        f"| RepSim (reference) | {repsim_r50:.4f} | 0.0 | -- | {repsim_tau:.4f} |",
        f"| Standard TRAK-PCA (baseline) | {baseline_r50:.4f} | {total_gap_pp:.1f} | 0.0 | {baseline_tau:.4f} |",
        f"| (a) Last-layer-only | {factor_a_r50:.4f} | {(repsim_r50-factor_a_r50)*100:.1f} | {gap_a:+.1f} | {results['lastlayer_trak_pca']['kendall_tau']:.4f} |",
        f"| (b) Cosine-normalized | {factor_b_r50:.4f} | {(repsim_r50-factor_b_r50)*100:.1f} | {gap_b:+.1f} | {results['cosine_trak_pca']['kendall_tau']:.4f} |",
        f"| (c) Combined (a+b) | {factor_c_r50:.4f} | {(repsim_r50-factor_c_r50)*100:.1f} | {gap_c:+.1f} | {results['combined_trak_pca']['kendall_tau']:.4f} |",
        f"| (d) Residual | -- | {residual_gap:.1f} | -- | -- |",
        f"| Raw dot product (no PCA) | {results['raw_dot_trak']['Recall@50']:.4f} | {(repsim_r50-results['raw_dot_trak']['Recall@50'])*100:.1f} | {(results['raw_dot_trak']['Recall@50']-baseline_r50)*100:+.1f} | {results['raw_dot_trak']['kendall_tau']:.4f} |",
        f"",
        f"## Pass Criteria",
        f"- Factor (a) or (b) >= 5pp gap reduction: **{'PASS' if pass_a_or_b else 'FAIL'}** (a={gap_a:+.1f}pp, b={gap_b:+.1f}pp)",
        f"- Combined >= 10pp gap reduction: **{'PASS' if pass_combined else 'FAIL'}** ({gap_c:+.1f}pp)",
        f"- Overall: **{'GO' if overall_pass else 'REFINE'}**",
        f"",
        f"## Gradient Statistics",
        f"| Config | Grad Dim | Mean Norm | CV(Norm) |",
        f"|---|---|---|---|",
        f"| Standard (attn+mlp) | {D_standard:,} | {grad_stats['standard']['mean_norm']:.6f} | {grad_stats['standard']['cv_norm']:.4f} |",
        f"| Last-layer (all params) | {D_lastlayer:,} | {grad_stats['lastlayer']['mean_norm']:.6f} | {grad_stats['lastlayer']['cv_norm']:.4f} |",
        f"",
        f"## Pilot Limitations",
        f"- N={n_train} caps PCA rank at {n_train}",
        f"- Full-scale (N=5473) will have higher effective rank for PCA",
    ]
    md_path = PILOTS_DIR / f"{TASK_ID}_pilot_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Pilot summary MD saved to {md_path}", flush=True)

    mark_done(
        status="success",
        summary=(
            f"Gap decomposition pilot complete in {elapsed:.0f}s. "
            f"Total gap: {total_gap_pp:.1f}pp. "
            f"a: {gap_a:+.1f}pp, b: {gap_b:+.1f}pp, c: {gap_c:+.1f}pp, "
            f"residual: {residual_gap:.1f}pp. "
            f"{'PASS' if overall_pass else 'REFINE'}"
        )
    )
    print(f"\nDone! Total runtime: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
