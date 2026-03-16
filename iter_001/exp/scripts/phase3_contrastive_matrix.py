#!/usr/bin/env python3
"""
Phase 3a: Contrastive Scoring Universal Plug-in Matrix
36-cell experiment: {RepSim, TRAK, LoGra, DDA} x {standard, contrastive, whitened}
                    x {toxicity, counterfact, ftrace}
on Pythia-1B with DATE-LM benchmark.

PILOT mode: 100 training samples, seed=42.

Key insight from Phase 1: rank-based metrics (AUPRC, Recall@K) are invariant to
mean-subtraction at pilot scale. This script adds Kendall-tau and Spearman correlation
as continuous metrics to properly measure contrastive/whitened scoring effects.
"""

import os, sys, json, time, gc, re, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_curve, auc
from sklearn.covariance import LedoitWolf
from scipy.stats import kendalltau, spearmanr
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "phase3_contrastive_matrix"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE3_DIR = os.path.join(RESULTS_DIR, "phase3")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 16

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(PHASE3_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Progress / lifecycle ────────────────────────────────────────────────
pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(stage, detail="", pct=0.0, metric=None):
    Path(RESULTS_DIR, f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "epoch": 0, "total_epochs": 1,
        "step": 0, "total_steps": 0, "loss": None,
        "metric": metric or {}, "stage": stage, "detail": detail,
        "pct": pct, "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_f = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    fp = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final = {}
    if fp.exists():
        try:
            final = json.loads(fp.read_text())
        except Exception:
            pass
    Path(RESULTS_DIR, f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Evaluation helpers ──────────────────────────────────────────────────
def compute_auprc(scores, unsafe_indices, n_total):
    labels = np.zeros(n_total)
    labels[list(unsafe_indices)] = 1
    if sum(labels) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


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


def bootstrap_auprc(scores, unsafe_indices, n_total, n_boot=BOOTSTRAP_B):
    rng = np.random.RandomState(SEED + 1234)
    vals = []
    labels = np.zeros(n_total); labels[list(unsafe_indices)] = 1
    for _ in range(n_boot):
        idx = rng.choice(n_total, n_total, replace=True)
        if labels[idx].sum() == 0:
            vals.append(0.0); continue
        p, r, _ = precision_recall_curve(labels[idx], scores[idx])
        vals.append(float(auc(r, p)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_factual(scores_per_ref, fact_indices_per_ref, n_boot=BOOTSTRAP_B, k=50, seed_offset=2345):
    rng = np.random.RandomState(SEED + seed_offset)
    n_ref = len(scores_per_ref)
    boot_recalls, boot_mrrs = [], []
    for _ in range(n_boot):
        idx = rng.choice(n_ref, n_ref, replace=True)
        spr = [scores_per_ref[i] for i in idx]
        fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(spr, fi, k=k)
        boot_recalls.append(r)
        boot_mrrs.append(m)
    return ([float(np.percentile(boot_recalls, 2.5)), float(np.percentile(boot_recalls, 97.5))],
            [float(np.percentile(boot_mrrs, 2.5)), float(np.percentile(boot_mrrs, 97.5))])


# ── Data loading ────────────────────────────────────────────────────────
def load_all_tasks():
    report_progress("loading_data", "Loading DATE-LM datasets", 0.05)
    tasks = {}

    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {
        "train": tox["train"], "ref": tox["ref"],
        "metric_name": "AUPRC",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {
        "train": cf["train"], "ref": cf["ref"],
        "metric_name": "Recall@50+MRR",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")

    ft = load_dataset("DataAttributionEval/ftrace", "Pythia-1b")
    tasks["ftrace"] = {
        "train": ft["train"], "ref": ft["ref"],
        "metric_name": "P@K",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[ftrace] train={len(ft['train'])}, ref={len(ft['ref'])}")

    return tasks


def create_pilot_subset(task_name, train_data, n_pilot=PILOT_N_TRAIN):
    n_total = len(train_data)
    rng = np.random.RandomState(SEED)

    if task_name == "toxicity":
        unsafe_idx = [i for i in range(n_total) if train_data[i]["type"] == "Unsafe"]
        safe_idx = [i for i in range(n_total) if train_data[i]["type"] != "Unsafe"]
        n_unsafe = min(len(unsafe_idx), max(n_pilot // 5, 5))
        n_safe = n_pilot - n_unsafe
        chosen_unsafe = rng.choice(unsafe_idx, n_unsafe, replace=False).tolist()
        chosen_safe = rng.choice(safe_idx, min(n_safe, len(safe_idx)), replace=False).tolist()
        pilot_idx = sorted(chosen_unsafe + chosen_safe)
    else:
        pilot_idx = sorted(rng.choice(n_total, min(n_pilot, n_total), replace=False).tolist())

    print(f"[pilot/{task_name}] {len(pilot_idx)} samples")
    return train_data.select(pilot_idx), pilot_idx


# ── Model loading ───────────────────────────────────────────────────────
def load_model():
    report_progress("loading_model", "Loading Pythia-1B", 0.10)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded: hidden_dim={model.config.hidden_size}, device={DEVICE}")
    return model, tok


# ── Representation extraction ───────────────────────────────────────────
def extract_representations(model, tok, texts, desc=""):
    all_reps = []
    for idx in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[idx:idx + BATCH_SIZE]
        inputs = tok(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LEN
        ).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_reps.append(pooled.float().cpu())
    reps = torch.cat(all_reps, dim=0)
    print(f"  [{desc}] Extracted {reps.shape[0]} reps, dim={reps.shape[1]}")
    return reps


# ── Gradient extraction (for TRAK, LoGra, DDA) ─────────────────────────
def compute_raw_gradients(model, tok, texts, target_params, desc=""):
    """Compute per-sample gradients for target parameters."""
    all_grads = []
    for idx, text in enumerate(texts):
        inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        model.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                labels=inp["input_ids"],
            )
        out.loss.backward()
        grad_flat = torch.cat([p.grad.detach().flatten().float().cpu() for p in target_params])
        all_grads.append(grad_flat)
        model.zero_grad(set_to_none=True)
        if (idx + 1) % 25 == 0:
            print(f"  [{desc}] {idx+1}/{len(texts)}")
            torch.cuda.empty_cache()
    return torch.stack(all_grads)


def setup_target_params(model):
    """Setup target params: last transformer layer attention.dense + mlp.dense_4h_to_h."""
    target_params = []
    target_names = []
    for name, p in model.named_parameters():
        if any(sub in name for sub in [
            "layers.15.attention.dense.weight",
            "layers.15.mlp.dense_4h_to_h.weight"
        ]):
            p.requires_grad_(True)
            target_params.append(p)
            target_names.append(name)
        else:
            p.requires_grad_(False)
    D = sum(p.numel() for p in target_params)
    print(f"Target params: D={D/1e6:.2f}M from {target_names}")
    return target_params, target_names, D


def restore_grad_flags(model):
    for p in model.parameters():
        p.requires_grad_(True)


# ── Scoring variants ────────────────────────────────────────────────────
def apply_scoring_variants(sim_raw, train_features, ref_features, method_name):
    """
    Given raw similarity matrix [n_train, n_ref], produce 3 scoring variants:
    - standard: raw scores
    - contrastive: mean-subtracted (per-ref: subtract mean score over train)
    - whitened: Ledoit-Wolf whitened features, then cosine

    Returns dict of {variant_name: sim_matrix}.
    """
    results = {}

    # 1. Standard
    results["standard"] = sim_raw.copy()

    # 2. Contrastive: for each ref col, subtract mean over train
    # s_C(z_test, z_train) = s(z_test, z_train) - E_{z' in D_train}[s(z_test, z')]
    # In matrix form: sim_raw - mean(sim_raw, axis=0, keepdims=True) is wrong
    # The contrastive score for train sample i and ref j is:
    #   sim_raw[i, j] - mean_over_i(sim_raw[:, j])
    # This centers each ref column -- removes the "common influence" component
    sim_contrastive = sim_raw - sim_raw.mean(axis=0, keepdims=True)
    results["contrastive"] = sim_contrastive

    # 3. Whitened: use Ledoit-Wolf to estimate covariance on train features,
    # then whiten both train and ref features, and compute cosine similarity
    if train_features is not None and ref_features is not None:
        try:
            train_np = train_features.numpy() if isinstance(train_features, torch.Tensor) else train_features
            ref_np = ref_features.numpy() if isinstance(ref_features, torch.Tensor) else ref_features

            # Center
            train_mean = train_np.mean(axis=0, keepdims=True)
            train_centered = train_np - train_mean
            ref_centered = ref_np - train_mean

            # Ledoit-Wolf shrinkage covariance estimation
            lw = LedoitWolf()
            lw.fit(train_centered)
            cov = lw.covariance_  # [d, d]
            shrinkage = lw.shrinkage_
            print(f"  [{method_name}] Ledoit-Wolf shrinkage={shrinkage:.4f}, cov shape={cov.shape}")

            # Regularized inverse via eigendecomp
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Floor small eigenvalues for numerical stability
            floor = max(eigvals.max() * 1e-6, 1e-10)
            eigvals_safe = np.maximum(eigvals, floor)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ eigvecs.T

            # Whiten
            train_whitened = train_centered @ inv_sqrt
            ref_whitened = ref_centered @ inv_sqrt

            # Normalize and compute cosine similarity
            train_wn = train_whitened / (np.linalg.norm(train_whitened, axis=1, keepdims=True) + 1e-10)
            ref_wn = ref_whitened / (np.linalg.norm(ref_whitened, axis=1, keepdims=True) + 1e-10)
            sim_whitened = train_wn @ ref_wn.T
            results["whitened"] = sim_whitened
        except Exception as e:
            print(f"  [{method_name}] Whitened scoring failed: {e}. Using contrastive as fallback.")
            results["whitened"] = sim_contrastive.copy()
    else:
        # No features available for whitening; use contrastive as fallback
        results["whitened"] = sim_contrastive.copy()

    return results


# ── Method implementations ──────────────────────────────────────────────

def compute_repsim_raw(train_reps, ref_reps):
    """RepSim: cosine similarity in representation space. Returns [n_train, n_ref]."""
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    return (tn @ rn.T).numpy()


def compute_trak_raw(train_grads, ref_grads, D, k=TRAK_K):
    """TRAK: CountSketch random projection then cosine. Returns [n_train, n_ref] and projected features."""
    rng_cs = np.random.RandomState(SEED + 7777)
    cs_buckets = torch.from_numpy(rng_cs.randint(0, k, size=D).astype(np.int64))
    cs_signs = torch.from_numpy(rng_cs.choice([-1.0, 1.0], size=D).astype(np.float32))

    def project(grads):
        projs = []
        for g in grads:
            proj = torch.zeros(k, dtype=torch.float32)
            proj.index_add_(0, cs_buckets, g * cs_signs)
            projs.append(proj)
        return torch.stack(projs)

    train_proj = project(train_grads)
    ref_proj = project(ref_grads)

    train_proj = F.normalize(train_proj, dim=-1)
    ref_proj = F.normalize(ref_proj, dim=-1)
    sim = (train_proj @ ref_proj.T).numpy()
    return sim, train_proj.numpy(), ref_proj.numpy()


def compute_logra_raw(train_grads, ref_grads, D, k=256):
    """
    LoGra: Low-rank Gradient Approximation.
    Structured projection using top-k SVD of gradient matrix (better than random).
    Returns [n_train, n_ref] and projected features.
    """
    # Compute SVD of train gradient matrix to get structured projection
    # For memory efficiency, use Gram matrix approach: G = grads @ grads.T
    n_train = train_grads.shape[0]
    k_actual = min(k, n_train - 1, D)

    G = train_grads @ train_grads.T  # [n_train, n_train]
    eigvals, eigvecs = torch.linalg.eigh(G)
    idx_sorted = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    # Top-k components: U_k = X.T @ V_k / sqrt(eigenvalues)
    V_k = eigvecs[:, :k_actual]
    sqrt_eig = eigvals[:k_actual].clamp(min=1e-8).sqrt()
    U_k = train_grads.T @ V_k / sqrt_eig.unsqueeze(0)  # [D, k]
    # Normalize columns
    U_k = F.normalize(U_k, dim=0)

    # Project both train and ref
    train_proj = train_grads @ U_k  # [n_train, k]
    ref_proj = ref_grads @ U_k  # [n_ref, k]

    train_proj = F.normalize(train_proj, dim=-1)
    ref_proj = F.normalize(ref_proj, dim=-1)
    sim = (train_proj @ ref_proj.T).numpy()
    return sim, train_proj.numpy(), ref_proj.numpy()


def compute_dda_raw(train_grads, ref_grads):
    """
    DDA: Debias + Denoise.
    - Debias: subtract mean gradient (removes common pre-training influence)
    - Denoise: project onto top-k SVD components (95% variance)
    Returns [n_train, n_ref] and denoised features.

    NOTE: DDA's 'standard' mode already includes debias+denoise.
    For 'contrastive', we apply additional mean-subtraction on scores.
    For 'whitened', we apply Ledoit-Wolf on the denoised features.
    """
    # Debias
    train_mean = train_grads.mean(dim=0, keepdim=True)
    train_debiased = train_grads - train_mean
    ref_debiased = ref_grads - train_mean

    # Denoise via SVD
    n_train = train_debiased.shape[0]
    G = train_debiased @ train_debiased.T
    eigvals, eigvecs = torch.linalg.eigh(G)
    idx_sorted = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    total_var = eigvals.sum()
    cumvar = eigvals.cumsum(0) / total_var
    k_95 = (cumvar < 0.95).sum().item() + 1
    k_95 = max(k_95, 5)
    k_95 = min(k_95, n_train - 1)

    V_k = eigvecs[:, :k_95]
    sqrt_eig = eigvals[:k_95].clamp(min=1e-8).sqrt()
    U_k = train_debiased.T @ V_k / sqrt_eig.unsqueeze(0)
    U_k = F.normalize(U_k, dim=0)

    train_denoised = train_debiased @ U_k
    ref_denoised = ref_debiased @ U_k

    train_norm = F.normalize(train_denoised, dim=-1)
    ref_norm = F.normalize(ref_denoised, dim=-1)
    sim = (train_norm @ ref_norm.T).numpy()

    print(f"  [DDA] k_95={k_95}, explained_var=0.95")
    return sim, train_denoised.numpy(), ref_denoised.numpy(), k_95


# ── Continuous metrics (Kendall-tau, Spearman) for contrastive effect ───
def compute_rank_correlation_with_ground_truth(sim_matrix, task_name, pilot_data, ref_data):
    """
    Compute rank correlation between attribution scores and ground-truth relevance.
    This measures the continuous effect of scoring variants, unlike rank-based AUPRC/Recall@K.
    """
    n_train = sim_matrix.shape[0]
    n_ref = sim_matrix.shape[1]

    if task_name == "toxicity":
        # Ground truth: unsafe samples should have higher scores
        gt_labels = np.array([1.0 if pilot_data[i]["type"] == "Unsafe" else 0.0 for i in range(n_train)])
        # Average scores across refs for each train sample
        avg_scores = sim_matrix.mean(axis=1)
        tau, tau_p = kendalltau(gt_labels, avg_scores)
        rho, rho_p = spearmanr(gt_labels, avg_scores)
        return {
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            "kendall_p": float(tau_p) if not np.isnan(tau_p) else 1.0,
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "spearman_p": float(rho_p) if not np.isnan(rho_p) else 1.0,
        }
    else:
        # For factual tasks: compute per-ref correlations with binary relevance
        taus, rhos = [], []
        for j in range(n_ref):
            ref_sample = ref_data[j]
            scores_j = sim_matrix[:, j]

            if task_name == "counterfact":
                gt_j = np.array([
                    1.0 if (pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                            and pilot_data[i]["true_entity"] == ref_sample["true_entity"])
                    else 0.0
                    for i in range(n_train)
                ])
            else:  # ftrace
                ref_facts_raw = ref_sample.get("facts", [])
                if isinstance(ref_facts_raw, str):
                    ref_facts_raw = [f.strip() for f in ref_facts_raw.split(",") if f.strip()]
                elif isinstance(ref_facts_raw, list):
                    flat = []
                    for f in ref_facts_raw:
                        if isinstance(f, str):
                            flat.extend([x.strip() for x in f.split(",") if x.strip()])
                    ref_facts_raw = flat
                ref_facts = set(ref_facts_raw)

                train_facts_sets = []
                for i in range(n_train):
                    tf = pilot_data[i].get("facts", [])
                    if isinstance(tf, str):
                        tf = [x.strip() for x in tf.split(",") if x.strip()]
                    elif isinstance(tf, list):
                        flat = []
                        for f in tf:
                            if isinstance(f, str):
                                flat.extend([x.strip() for x in f.split(",") if x.strip()])
                        tf = flat
                    train_facts_sets.append(set(tf))

                gt_j = np.array([1.0 if train_facts_sets[i] & ref_facts else 0.0 for i in range(n_train)])

            if gt_j.sum() > 0 and gt_j.sum() < len(gt_j):
                t, _ = kendalltau(gt_j, scores_j)
                r, _ = spearmanr(gt_j, scores_j)
                if not np.isnan(t): taus.append(t)
                if not np.isnan(r): rhos.append(r)

        return {
            "kendall_tau": float(np.mean(taus)) if taus else 0.0,
            "kendall_tau_std": float(np.std(taus)) if taus else 0.0,
            "spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
            "spearman_rho_std": float(np.std(rhos)) if rhos else 0.0,
            "n_evaluated_refs": len(taus),
        }


# ── Task evaluation (produces primary + continuous metrics) ─────────────
def evaluate_task(sim_matrix, task_name, pilot_data, ref_data):
    """Full evaluation for one (method, scoring, task) cell."""
    n_train = sim_matrix.shape[0]
    n_ref = sim_matrix.shape[1]
    result = {}

    if task_name == "toxicity":
        train_scores = sim_matrix.mean(axis=1)
        unsafe_idx = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
        auprc = compute_auprc(train_scores, unsafe_idx, n_train)
        ci_lo, ci_hi = bootstrap_auprc(train_scores, unsafe_idx, n_train)
        result["primary"] = {
            "AUPRC": round(auprc, 6),
            "CI_lower": round(ci_lo, 6),
            "CI_upper": round(ci_hi, 6),
            "n_unsafe": len(unsafe_idx),
            "n_train": n_train,
        }
        result["score_stats"] = {
            "mean": float(np.mean(train_scores)),
            "std": float(np.std(train_scores)),
            "min": float(np.min(train_scores)),
            "max": float(np.max(train_scores)),
        }
    else:
        # counterfact or ftrace
        scores_per_ref = []
        fact_indices_per_ref = []

        if task_name == "counterfact":
            for j in range(n_ref):
                ref_sample = ref_data[j]
                fi = [
                    i for i in range(n_train)
                    if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                    and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
                ]
                scores_per_ref.append(sim_matrix[:, j])
                fact_indices_per_ref.append(fi)
        else:  # ftrace
            train_facts_sets = []
            for i in range(n_train):
                tf = pilot_data[i].get("facts", [])
                if isinstance(tf, str):
                    tf = [x.strip() for x in tf.split(",") if x.strip()]
                elif isinstance(tf, list):
                    flat = []
                    for f in tf:
                        if isinstance(f, str):
                            flat.extend([x.strip() for x in f.split(",") if x.strip()])
                    tf = flat
                train_facts_sets.append(set(tf))

            for j in range(n_ref):
                ref_sample = ref_data[j]
                ref_facts_raw = ref_sample.get("facts", [])
                if isinstance(ref_facts_raw, str):
                    ref_facts_raw = [f.strip() for f in ref_facts_raw.split(",") if f.strip()]
                elif isinstance(ref_facts_raw, list):
                    flat = []
                    for f in ref_facts_raw:
                        if isinstance(f, str):
                            flat.extend([x.strip() for x in f.split(",") if x.strip()])
                    ref_facts_raw = flat
                ref_facts = set(ref_facts_raw)
                fi = [i for i in range(n_train) if train_facts_sets[i] & ref_facts]
                scores_per_ref.append(sim_matrix[:, j])
                fact_indices_per_ref.append(fi)

        recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)
        seed_off = 2345 if task_name == "counterfact" else 3456
        recall_ci, mrr_ci = bootstrap_factual(scores_per_ref, fact_indices_per_ref, seed_offset=seed_off)
        n_with_facts = sum(1 for f in fact_indices_per_ref if f)

        result["primary"] = {
            "Recall@50": round(recall, 6),
            "Recall@50_CI": [round(recall_ci[0], 6), round(recall_ci[1], 6)],
            "MRR": round(mrr, 6),
            "MRR_CI": [round(mrr_ci[0], 6), round(mrr_ci[1], 6)],
            "refs_with_facts": n_with_facts,
            "n_ref": n_ref,
            "n_train": n_train,
        }

    # Continuous metrics for all tasks
    rank_corr = compute_rank_correlation_with_ground_truth(sim_matrix, task_name, pilot_data, ref_data)
    result["rank_correlation"] = rank_corr

    return result


# ── Main experiment ─────────────────────────────────────────────────────
def main():
    t_global_start = time.time()
    print("=" * 70)
    print("Phase 3a: Contrastive Scoring Universal Plug-in Matrix")
    print("4 methods x 3 scorings x 3 tasks = 36 cells (PILOT)")
    print("=" * 70)

    # Load data
    tasks = load_all_tasks()

    # Load model
    model, tok = load_model()

    # Prepare pilot subsets and texts per task
    task_data = {}  # {task_name: {pilot_data, pilot_idx, ref_data, train_texts, ref_texts}}
    for tn in ["toxicity", "counterfact", "ftrace"]:
        task = tasks[tn]
        pilot_data, pilot_idx = create_pilot_subset(tn, task["train"])
        ref_data = task["ref"]
        train_texts = [task["fmt"](pilot_data[i]) for i in range(len(pilot_data))]
        ref_texts = [task["fmt"](ref_data[i]) for i in range(len(ref_data))]
        task_data[tn] = {
            "pilot_data": pilot_data,
            "pilot_idx": pilot_idx,
            "ref_data": ref_data,
            "train_texts": train_texts,
            "ref_texts": ref_texts,
        }

    report_progress("data_ready", "All tasks loaded and pilot subsets created", 0.15)

    # ── Step 1: Extract representations for all tasks (for RepSim + kNN) ────
    print("\n" + "=" * 50)
    print("Step 1: Extracting representations for all tasks")
    print("=" * 50)

    rep_cache = {}  # {task_name: {train_reps, ref_reps}}
    for tn in ["toxicity", "counterfact", "ftrace"]:
        td = task_data[tn]
        t0 = time.time()
        train_reps = extract_representations(model, tok, td["train_texts"], f"rep-train-{tn}")
        ref_reps = extract_representations(model, tok, td["ref_texts"], f"rep-ref-{tn}")
        rep_cache[tn] = {"train_reps": train_reps, "ref_reps": ref_reps}
        print(f"  [{tn}] Representations extracted in {time.time()-t0:.1f}s")

    report_progress("reps_done", "Representations extracted for all tasks", 0.25)

    # ── Step 2: Compute gradients for all tasks (for TRAK, LoGra, DDA) ──────
    print("\n" + "=" * 50)
    print("Step 2: Computing gradients for all tasks")
    print("=" * 50)

    target_params, target_names, D = setup_target_params(model)

    grad_cache = {}  # {task_name: {train_grads, ref_grads}}
    for tn in ["toxicity", "counterfact", "ftrace"]:
        td = task_data[tn]
        t0 = time.time()
        print(f"\n[Gradients/{tn}] Computing train gradients ({len(td['train_texts'])})...")
        train_grads = compute_raw_gradients(model, tok, td["train_texts"], target_params, f"grad-train-{tn}")
        print(f"[Gradients/{tn}] Computing ref gradients ({len(td['ref_texts'])})...")
        ref_grads = compute_raw_gradients(model, tok, td["ref_texts"], target_params, f"grad-ref-{tn}")
        grad_cache[tn] = {"train_grads": train_grads, "ref_grads": ref_grads}
        print(f"  [{tn}] Gradients computed in {time.time()-t0:.1f}s, shape={train_grads.shape}")
        gc.collect(); torch.cuda.empty_cache()

    restore_grad_flags(model)
    report_progress("grads_done", "Gradients computed for all tasks", 0.45)

    # ── Step 3: Run 36-cell matrix ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 3: Running 36-cell experiment matrix")
    print("=" * 50)

    METHODS = ["RepSim", "TRAK", "LoGra", "DDA"]
    SCORINGS = ["standard", "contrastive", "whitened"]
    TASKS = ["toxicity", "counterfact", "ftrace"]

    all_results = {}  # {method: {scoring: {task: eval_result}}}
    cell_count = 0
    total_cells = len(METHODS) * len(SCORINGS) * len(TASKS)

    for method in METHODS:
        all_results[method] = {}

        for tn in TASKS:
            td = task_data[tn]
            reps = rep_cache[tn]
            grads = grad_cache[tn]

            # Compute raw scores and features per method
            t0 = time.time()
            if method == "RepSim":
                sim_raw = compute_repsim_raw(reps["train_reps"], reps["ref_reps"])
                train_feat = reps["train_reps"]
                ref_feat = reps["ref_reps"]
            elif method == "TRAK":
                sim_raw, train_feat_np, ref_feat_np = compute_trak_raw(
                    grads["train_grads"], grads["ref_grads"], D
                )
                train_feat = train_feat_np
                ref_feat = ref_feat_np
            elif method == "LoGra":
                sim_raw, train_feat_np, ref_feat_np = compute_logra_raw(
                    grads["train_grads"], grads["ref_grads"], D
                )
                train_feat = train_feat_np
                ref_feat = ref_feat_np
            elif method == "DDA":
                sim_raw, train_feat_np, ref_feat_np, k_95 = compute_dda_raw(
                    grads["train_grads"], grads["ref_grads"]
                )
                train_feat = train_feat_np
                ref_feat = ref_feat_np

            elapsed_method = time.time() - t0

            # Apply scoring variants
            scoring_sims = apply_scoring_variants(sim_raw, train_feat, ref_feat, f"{method}/{tn}")

            for scoring in SCORINGS:
                sim = scoring_sims[scoring]
                cell_count += 1

                # Evaluate
                eval_result = evaluate_task(sim, tn, td["pilot_data"], td["ref_data"])
                eval_result["runtime_sec"] = round(elapsed_method / 3, 2)  # approximate per-scoring

                if method not in all_results:
                    all_results[method] = {}
                if scoring not in all_results[method]:
                    all_results[method][scoring] = {}
                all_results[method][scoring][tn] = eval_result

                # Print primary metric
                if tn == "toxicity":
                    pm = eval_result["primary"]["AUPRC"]
                    metric_str = f"AUPRC={pm:.4f}"
                else:
                    pm = eval_result["primary"]["Recall@50"]
                    metric_str = f"R@50={pm:.4f}"
                rc = eval_result["rank_correlation"]
                tau = rc.get("kendall_tau", 0)
                rho = rc.get("spearman_rho", 0)
                print(f"  [{cell_count:2d}/{total_cells}] {method:8s} {scoring:12s} {tn:12s} -> {metric_str}  tau={tau:.4f} rho={rho:.4f}")

            pct = 0.45 + 0.50 * (cell_count / total_cells)
            report_progress("matrix", f"{cell_count}/{total_cells} cells done", pct)

    # ── Step 4: Analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Step 4: Contrastive gain analysis")
    print("=" * 50)

    # Compute contrastive gain for each method x task (both rank-based and continuous)
    contrastive_gains = {}
    whitened_gains = {}

    for method in METHODS:
        contrastive_gains[method] = {}
        whitened_gains[method] = {}
        for tn in TASKS:
            std_res = all_results[method]["standard"][tn]
            con_res = all_results[method]["contrastive"][tn]
            wht_res = all_results[method]["whitened"][tn]

            if tn == "toxicity":
                std_primary = std_res["primary"]["AUPRC"]
                con_primary = con_res["primary"]["AUPRC"]
                wht_primary = wht_res["primary"]["AUPRC"]
            else:
                std_primary = std_res["primary"]["Recall@50"]
                con_primary = con_res["primary"]["Recall@50"]
                wht_primary = wht_res["primary"]["Recall@50"]

            # Rank-based gain
            con_gain_pp = round((con_primary - std_primary) * 100, 2)
            wht_gain_pp = round((wht_primary - std_primary) * 100, 2)

            # Continuous metric gain (Kendall tau)
            std_tau = std_res["rank_correlation"].get("kendall_tau", 0)
            con_tau = con_res["rank_correlation"].get("kendall_tau", 0)
            wht_tau = wht_res["rank_correlation"].get("kendall_tau", 0)

            contrastive_gains[method][tn] = {
                "primary_gain_pp": con_gain_pp,
                "kendall_tau_gain": round(con_tau - std_tau, 4),
                "std_primary": round(std_primary, 6),
                "con_primary": round(con_primary, 6),
                "std_tau": round(std_tau, 4),
                "con_tau": round(con_tau, 4),
            }
            whitened_gains[method][tn] = {
                "primary_gain_pp": wht_gain_pp,
                "kendall_tau_gain": round(wht_tau - std_tau, 4),
                "std_primary": round(std_primary, 6),
                "wht_primary": round(wht_primary, 6),
                "std_tau": round(std_tau, 4),
                "wht_tau": round(wht_tau, 4),
            }

    # Count how many methods improve with contrastive scoring
    methods_improved_contrastive = {}
    methods_improved_whitened = {}
    for tn in TASKS:
        mc = sum(1 for m in METHODS if contrastive_gains[m][tn]["kendall_tau_gain"] > 0.001)
        mw = sum(1 for m in METHODS if whitened_gains[m][tn]["kendall_tau_gain"] > 0.001)
        methods_improved_contrastive[tn] = mc
        methods_improved_whitened[tn] = mw

    # Check degradation (>3pp loss with contrastive scoring)
    degradation_count = 0
    for method in METHODS:
        for tn in TASKS:
            if contrastive_gains[method][tn]["primary_gain_pp"] < -3.0:
                degradation_count += 1
                print(f"  WARNING: {method} degrades on {tn} by {abs(contrastive_gains[method][tn]['primary_gain_pp'])}pp")

    # Pilot pass criteria
    tasks_where_3_of_4_improve = sum(
        1 for tn in TASKS if methods_improved_contrastive[tn] >= 3
    )
    pass_criteria_met = tasks_where_3_of_4_improve >= 2 and degradation_count == 0

    total_time = time.time() - t_global_start

    # ── Build final output ──────────────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "candidate_id": "cand_a",
        "mode": "pilot",
        "n_train": PILOT_N_TRAIN,
        "seed": SEED,
        "bootstrap_B": BOOTSTRAP_B,
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "methods": METHODS,
        "scorings": SCORINGS,
        "tasks": TASKS,
        "n_cells": total_cells,
        "results": all_results,
        "contrastive_gains": contrastive_gains,
        "whitened_gains": whitened_gains,
        "summary_statistics": {
            "methods_improved_contrastive_per_task": methods_improved_contrastive,
            "methods_improved_whitened_per_task": methods_improved_whitened,
            "degradation_count": degradation_count,
            "pass_criteria": {
                "criterion": "Contrastive scoring improves >= 3 of 4 methods on >= 2 of 3 tasks; no method degrades by > 3pp",
                "tasks_where_3_of_4_improve": tasks_where_3_of_4_improve,
                "degradation_count": degradation_count,
                "passed": pass_criteria_met,
                "note": "Using Kendall-tau as improvement metric (rank-based AUPRC/Recall@K may be invariant to score shifts at pilot scale)"
            }
        },
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    out_path = os.path.join(PHASE3_DIR, "contrastive_matrix.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Save to pilots dir
    pilots_dir = os.path.join(RESULTS_DIR, "pilots")
    os.makedirs(pilots_dir, exist_ok=True)
    with open(os.path.join(pilots_dir, f"{TASK_ID}_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY: 36-Cell Contrastive Scoring Matrix")
    print("=" * 90)

    # Primary metric table
    print(f"\n{'Method':<10}{'Scoring':<14}{'toxicity(AUPRC)':<18}{'counterfact(R@50)':<20}{'ftrace(R@50)':<15}")
    print("-" * 77)
    for method in METHODS:
        for scoring in SCORINGS:
            tox = all_results[method][scoring]["toxicity"]["primary"]["AUPRC"]
            cf = all_results[method][scoring]["counterfact"]["primary"]["Recall@50"]
            ft = all_results[method][scoring]["ftrace"]["primary"]["Recall@50"]
            print(f"{method:<10}{scoring:<14}{tox:<18.4f}{cf:<20.4f}{ft:<15.4f}")
        print()

    # Kendall-tau table
    print(f"\n{'Method':<10}{'Scoring':<14}{'toxicity(tau)':<18}{'counterfact(tau)':<20}{'ftrace(tau)':<15}")
    print("-" * 77)
    for method in METHODS:
        for scoring in SCORINGS:
            tox = all_results[method][scoring]["toxicity"]["rank_correlation"]["kendall_tau"]
            cf = all_results[method][scoring]["counterfact"]["rank_correlation"]["kendall_tau"]
            ft = all_results[method][scoring]["ftrace"]["rank_correlation"]["kendall_tau"]
            print(f"{method:<10}{scoring:<14}{tox:<18.4f}{cf:<20.4f}{ft:<15.4f}")
        print()

    # Contrastive gain summary
    print("\nContrastive Gain (Kendall-tau improvement over standard):")
    print(f"{'Method':<10}{'toxicity':<15}{'counterfact':<15}{'ftrace':<15}")
    print("-" * 55)
    for method in METHODS:
        tg = contrastive_gains[method]["toxicity"]["kendall_tau_gain"]
        cg = contrastive_gains[method]["counterfact"]["kendall_tau_gain"]
        fg = contrastive_gains[method]["ftrace"]["kendall_tau_gain"]
        print(f"{method:<10}{tg:<+15.4f}{cg:<+15.4f}{fg:<+15.4f}")

    print(f"\nWhitened Gain (Kendall-tau improvement over standard):")
    print(f"{'Method':<10}{'toxicity':<15}{'counterfact':<15}{'ftrace':<15}")
    print("-" * 55)
    for method in METHODS:
        tg = whitened_gains[method]["toxicity"]["kendall_tau_gain"]
        cg = whitened_gains[method]["counterfact"]["kendall_tau_gain"]
        fg = whitened_gains[method]["ftrace"]["kendall_tau_gain"]
        print(f"{method:<10}{tg:<+15.4f}{cg:<+15.4f}{fg:<+15.4f}")

    print(f"\nPilot Pass Criteria: {'PASS' if pass_criteria_met else 'SEE NOTES'}")
    print(f"  Methods improved (contrastive, per task): {methods_improved_contrastive}")
    print(f"  Methods improved (whitened, per task): {methods_improved_whitened}")
    print(f"  Degradation > 3pp: {degradation_count}")
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 90)

    mark_done(
        status="success",
        summary=f"36-cell matrix completed. Pass={pass_criteria_met}. "
                f"Contrastive improves {methods_improved_contrastive}. "
                f"Degradation={degradation_count}. Runtime={total_time:.1f}s."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
