#!/usr/bin/env python3
"""
P1: FM2 Continuous Metrics on Full Factorial (H2-revised) -- PILOT MODE
========================================================================
Full method tournament: {RepSim, TRAK, LoGra, DDA, Raw Dot IF, Diag IF, k-NN, BM25}
                      x {standard, contrastive}
on all 3 DATE-LM tasks with both rank-based AND continuous metrics.

Also: 2x2 factorial {TRAK, RepSim} x {standard, contrastive} with continuous metrics
to test whether Kendall tau can detect FM2 effects that rank-based metrics missed.

PILOT: N=100 training samples, seed=42, timeout=900s.
Pass criteria: Kendall tau computable for all method-task pairs; at least one method
shows Kendall tau difference > 0.01 between standard and contrastive scoring.
"""

import os, sys, json, time, gc, re, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_curve, auc, ndcg_score
from sklearn.covariance import LedoitWolf
from scipy.stats import kendalltau, spearmanr
from rank_bm25 import BM25Okapi
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p1_fm2_continuous_metrics"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048
LOGRA_K = 256
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 4  # Small batch to fit in limited GPU memory
KNN_K = 50

METHODS = ["RepSim", "TRAK", "LoGra", "DDA", "RawDotIF", "DiagIF", "kNN", "BM25"]
SCORINGS = ["standard", "contrastive"]
TASKS = ["toxicity", "counterfact", "ftrace"]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(FULL_DIR, exist_ok=True)
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
    labels = np.zeros(n_total)
    labels[list(unsafe_indices)] = 1
    for _ in range(n_boot):
        idx = rng.choice(n_total, n_total, replace=True)
        if labels[idx].sum() == 0:
            vals.append(0.0)
            continue
        p, r, _ = precision_recall_curve(labels[idx], scores[idx])
        vals.append(float(auc(r, p)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_factual(scores_per_ref, fact_indices_per_ref, n_boot=BOOTSTRAP_B, k=50):
    rng = np.random.RandomState(SEED + 2345)
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


def compute_ndcg(scores, relevance, k=50):
    """Compute NDCG@k given score array and relevance array."""
    try:
        return float(ndcg_score([relevance], [scores], k=k))
    except Exception:
        return 0.0


# ── Continuous metrics ──────────────────────────────────────────────────
def compute_continuous_metrics(sim_matrix, task_name, pilot_data, ref_data):
    """
    Compute Kendall tau, Spearman rho, and NDCG on raw attribution scores.
    These are continuous metrics that CAN detect mean-subtraction effects,
    unlike rank-based AUPRC/Recall@K.
    """
    n_train = sim_matrix.shape[0]
    n_ref = sim_matrix.shape[1]

    if task_name == "toxicity":
        gt_labels = np.array([1.0 if pilot_data[i]["type"] == "Unsafe" else 0.0
                              for i in range(n_train)])
        avg_scores = sim_matrix.mean(axis=1)
        tau_val, tau_p = kendalltau(gt_labels, avg_scores)
        rho_val, rho_p = spearmanr(gt_labels, avg_scores)
        ndcg_val = compute_ndcg(avg_scores, gt_labels, k=50)
        return {
            "kendall_tau": float(tau_val) if not np.isnan(tau_val) else 0.0,
            "kendall_p": float(tau_p) if not np.isnan(tau_p) else 1.0,
            "spearman_rho": float(rho_val) if not np.isnan(rho_val) else 0.0,
            "spearman_p": float(rho_p) if not np.isnan(rho_p) else 1.0,
            "ndcg_at_50": ndcg_val,
        }
    else:
        # For factual tasks: compute per-ref correlations
        taus, rhos, ndcgs = [], [], []
        for j in range(n_ref):
            ref_sample = ref_data[j]
            scores_j = sim_matrix[:, j]
            gt_j = _get_ground_truth(task_name, pilot_data, ref_sample, n_train)

            if gt_j.sum() > 0 and gt_j.sum() < len(gt_j):
                t, _ = kendalltau(gt_j, scores_j)
                r, _ = spearmanr(gt_j, scores_j)
                if not np.isnan(t):
                    taus.append(t)
                if not np.isnan(r):
                    rhos.append(r)
                ndcgs.append(compute_ndcg(scores_j, gt_j, k=50))

        return {
            "kendall_tau": float(np.mean(taus)) if taus else 0.0,
            "kendall_tau_std": float(np.std(taus)) if taus else 0.0,
            "spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
            "spearman_rho_std": float(np.std(rhos)) if rhos else 0.0,
            "ndcg_at_50": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "ndcg_at_50_std": float(np.std(ndcgs)) if ndcgs else 0.0,
            "n_evaluated_refs": len(taus),
        }


def _get_ground_truth(task_name, pilot_data, ref_sample, n_train):
    """Get binary ground-truth relevance for a given ref sample."""
    if task_name == "counterfact":
        return np.array([
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

        gt = np.zeros(n_train)
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
            if set(tf) & ref_facts:
                gt[i] = 1.0
        return gt


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
    gc.collect()
    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, dtype=torch.float16,
        device_map=DEVICE, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Model loaded: hidden_dim={model.config.hidden_size}, device={DEVICE}")
    free_mem = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"GPU free memory after model load: {free_mem:.0f} MiB")
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
        del inputs, outputs, hidden, mask, pooled
        torch.cuda.empty_cache()
    reps = torch.cat(all_reps, dim=0)
    print(f"  [{desc}] Extracted {reps.shape[0]} reps, dim={reps.shape[1]}")
    return reps


# ── Gradient extraction ─────────────────────────────────────────────────
def setup_target_params(model):
    """Target params: last transformer layer attention.dense + mlp.dense_4h_to_h."""
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


# ── Method implementations ──────────────────────────────────────────────

def compute_repsim_raw(train_reps, ref_reps):
    """RepSim: cosine similarity. Returns [n_train, n_ref]."""
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    return (tn @ rn.T).numpy()


def compute_knn_raw(train_reps, ref_reps, k=KNN_K):
    """k-NN in representation space: binary indicator within top-k. Returns [n_train, n_ref]."""
    # Use cosine similarity, then zero out everything outside top-k per ref column
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    sim = (tn @ rn.T).numpy()
    # For each ref, keep only top-k train scores
    sim_knn = np.zeros_like(sim)
    for j in range(sim.shape[1]):
        topk_idx = np.argsort(-sim[:, j])[:k]
        sim_knn[topk_idx, j] = sim[topk_idx, j]
    return sim_knn


def compute_trak_raw(train_grads, ref_grads, D, k=TRAK_K):
    """TRAK: CountSketch random projection then cosine. Returns [n_train, n_ref]."""
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
    return (train_proj @ ref_proj.T).numpy()


def compute_logra_raw(train_grads, ref_grads, D, k=LOGRA_K):
    """LoGra: Low-rank gradient approximation via SVD."""
    n_train = train_grads.shape[0]
    k_actual = min(k, n_train - 1, D)

    G = train_grads @ train_grads.T
    eigvals, eigvecs = torch.linalg.eigh(G)
    idx_sorted = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    V_k = eigvecs[:, :k_actual]
    sqrt_eig = eigvals[:k_actual].clamp(min=1e-8).sqrt()
    U_k = train_grads.T @ V_k / sqrt_eig.unsqueeze(0)
    U_k = F.normalize(U_k, dim=0)

    train_proj = train_grads @ U_k
    ref_proj = ref_grads @ U_k
    train_proj = F.normalize(train_proj, dim=-1)
    ref_proj = F.normalize(ref_proj, dim=-1)
    return (train_proj @ ref_proj.T).numpy()


def compute_dda_raw(train_grads, ref_grads):
    """DDA: Debias + Denoise (mean subtraction + SVD denoising)."""
    train_mean = train_grads.mean(dim=0, keepdim=True)
    train_debiased = train_grads - train_mean
    ref_debiased = ref_grads - train_mean

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
    print(f"  [DDA] k_95={k_95}")
    return (train_norm @ ref_norm.T).numpy()


def compute_raw_dot_if(train_grads, ref_grads):
    """Raw Dot IF: gradient dot product without Hessian inverse. Returns [n_train, n_ref]."""
    # No normalization -- raw dot product captures gradient norm signal
    return (train_grads @ ref_grads.T).numpy()


def compute_diag_if(train_grads, ref_grads):
    """
    Diagonal Fisher IF: g_train^T diag(F)^{-1} g_ref
    where diag(F) = diag of empirical Fisher = mean of g^2 over train.
    """
    # Compute diagonal Fisher approximation from training gradients
    fisher_diag = (train_grads ** 2).mean(dim=0)  # [D]
    # Regularize to avoid division by zero
    fisher_diag = fisher_diag.clamp(min=1e-10)
    inv_fisher_diag = 1.0 / fisher_diag  # [D]

    # Apply inverse diagonal Fisher: scale each gradient dimension
    train_scaled = train_grads * inv_fisher_diag.unsqueeze(0)  # [n_train, D]
    sim = (train_scaled @ ref_grads.T).numpy()  # [n_train, n_ref]
    return sim


def tokenize_simple(text):
    return re.findall(r'\b\w+\b', text.lower())


def compute_bm25_raw(train_texts, ref_texts):
    """BM25 lexical matching. Returns [n_train, n_ref]."""
    train_tokenized = [tokenize_simple(t) for t in train_texts]
    ref_tokenized = [tokenize_simple(t) for t in ref_texts]
    bm25 = BM25Okapi(train_tokenized)
    sim = np.zeros((len(train_texts), len(ref_texts)), dtype=np.float32)
    for j, ref_tok in enumerate(ref_tokenized):
        scores = bm25.get_scores(ref_tok)
        sim[:, j] = scores
    return sim


# ── Contrastive scoring ────────────────────────────────────────────────
def apply_contrastive(sim_raw):
    """Mean subtraction: for each ref column, subtract mean over train."""
    return sim_raw - sim_raw.mean(axis=0, keepdims=True)


# ── Full evaluation for one cell ────────────────────────────────────────
def evaluate_cell(sim_matrix, task_name, pilot_data, ref_data):
    """Full evaluation: rank-based + continuous metrics for one (method, scoring, task) cell."""
    n_train = sim_matrix.shape[0]
    n_ref = sim_matrix.shape[1]
    result = {}

    # -- Rank-based metrics --
    if task_name == "toxicity":
        train_scores = sim_matrix.mean(axis=1)
        unsafe_idx = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
        auprc = compute_auprc(train_scores, unsafe_idx, n_train)
        ci_lo, ci_hi = bootstrap_auprc(train_scores, unsafe_idx, n_train)
        result["rank_based"] = {
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
        scores_per_ref = []
        fact_indices_per_ref = []

        for j in range(n_ref):
            ref_sample = ref_data[j]
            if task_name == "counterfact":
                fi = [
                    i for i in range(n_train)
                    if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                    and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
                ]
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
                        flat2 = []
                        for f in tf:
                            if isinstance(f, str):
                                flat2.extend([x.strip() for x in f.split(",") if x.strip()])
                        tf = flat2
                    train_facts_sets.append(set(tf))
                fi = [i for i in range(n_train) if train_facts_sets[i] & ref_facts]

            scores_per_ref.append(sim_matrix[:, j])
            fact_indices_per_ref.append(fi)

        recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)
        recall_ci, mrr_ci = bootstrap_factual(scores_per_ref, fact_indices_per_ref)
        refs_with_facts = sum(1 for fi in fact_indices_per_ref if fi)
        result["rank_based"] = {
            "Recall_at_50": round(recall, 6),
            "Recall_at_50_CI": [round(x, 6) for x in recall_ci],
            "MRR": round(mrr, 6),
            "MRR_CI": [round(x, 6) for x in mrr_ci],
            "refs_with_facts": refs_with_facts,
            "n_ref": n_ref,
            "n_train": n_train,
        }

    # -- Continuous metrics --
    result["continuous"] = compute_continuous_metrics(sim_matrix, task_name, pilot_data, ref_data)

    return result


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    report_progress("init", "Starting P1 FM2 Continuous Metrics (PILOT)", 0.0)

    # Load data
    all_tasks = load_all_tasks()

    # Load model
    model, tok = load_model()

    # Per-task data prep
    task_data = {}
    for task_name in TASKS:
        task_info = all_tasks[task_name]
        pilot_train, pilot_idx = create_pilot_subset(task_name, task_info["train"])
        train_texts = [task_info["fmt"](pilot_train[i]) for i in range(len(pilot_train))]
        ref_texts = [task_info["fmt"](task_info["ref"][i]) for i in range(len(task_info["ref"]))]
        task_data[task_name] = {
            "pilot_train": pilot_train,
            "pilot_idx": pilot_idx,
            "ref_data": task_info["ref"],
            "train_texts": train_texts,
            "ref_texts": ref_texts,
        }
    report_progress("data_loaded", "All 3 tasks loaded", 0.15)

    # ── Extract representations (for RepSim, kNN) ──────────────────────
    print("\n=== Extracting representations ===")
    reps_cache = {}
    for task_name in TASKS:
        td = task_data[task_name]
        train_reps = extract_representations(model, tok, td["train_texts"], f"{task_name}/train")
        ref_reps = extract_representations(model, tok, td["ref_texts"], f"{task_name}/ref")
        reps_cache[task_name] = {"train": train_reps, "ref": ref_reps}
    report_progress("reps_extracted", "Representations extracted for all tasks", 0.25)

    # ── Extract gradients (for TRAK, LoGra, DDA, RawDotIF, DiagIF) ────
    print("\n=== Extracting gradients ===")
    target_params, target_names, D = setup_target_params(model)
    grads_cache = {}
    for task_name in TASKS:
        td = task_data[task_name]
        train_grads = compute_raw_gradients(model, tok, td["train_texts"], target_params, f"{task_name}/train_grad")
        ref_grads = compute_raw_gradients(model, tok, td["ref_texts"], target_params, f"{task_name}/ref_grad")
        grads_cache[task_name] = {"train": train_grads, "ref": ref_grads}
        gc.collect()
        torch.cuda.empty_cache()
    restore_grad_flags(model)
    report_progress("grads_extracted", "Gradients extracted for all tasks", 0.45)

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Compute raw similarity matrices for all methods ─────────────────
    print("\n=== Computing raw similarity matrices ===")
    all_results = {}  # {method: {scoring: {task: eval_result}}}
    method_timings = {}

    for task_name in TASKS:
        td = task_data[task_name]
        reps = reps_cache[task_name]
        grads = grads_cache[task_name]

        print(f"\n--- Task: {task_name} ---")

        # Compute raw sim matrices for each method
        raw_sims = {}

        # RepSim
        t_m = time.time()
        raw_sims["RepSim"] = compute_repsim_raw(reps["train"], reps["ref"])
        method_timings.setdefault("RepSim", {})[task_name] = time.time() - t_m
        print(f"  RepSim: {method_timings['RepSim'][task_name]:.2f}s")

        # kNN
        t_m = time.time()
        raw_sims["kNN"] = compute_knn_raw(reps["train"], reps["ref"])
        method_timings.setdefault("kNN", {})[task_name] = time.time() - t_m
        print(f"  kNN: {method_timings['kNN'][task_name]:.2f}s")

        # TRAK
        t_m = time.time()
        raw_sims["TRAK"] = compute_trak_raw(grads["train"], grads["ref"], D)
        method_timings.setdefault("TRAK", {})[task_name] = time.time() - t_m
        print(f"  TRAK: {method_timings['TRAK'][task_name]:.2f}s")

        # LoGra
        t_m = time.time()
        raw_sims["LoGra"] = compute_logra_raw(grads["train"], grads["ref"], D)
        method_timings.setdefault("LoGra", {})[task_name] = time.time() - t_m
        print(f"  LoGra: {method_timings['LoGra'][task_name]:.2f}s")

        # DDA
        t_m = time.time()
        raw_sims["DDA"] = compute_dda_raw(grads["train"], grads["ref"])
        method_timings.setdefault("DDA", {})[task_name] = time.time() - t_m
        print(f"  DDA: {method_timings['DDA'][task_name]:.2f}s")

        # RawDotIF
        t_m = time.time()
        raw_sims["RawDotIF"] = compute_raw_dot_if(grads["train"], grads["ref"])
        method_timings.setdefault("RawDotIF", {})[task_name] = time.time() - t_m
        print(f"  RawDotIF: {method_timings['RawDotIF'][task_name]:.2f}s")

        # DiagIF
        t_m = time.time()
        raw_sims["DiagIF"] = compute_diag_if(grads["train"], grads["ref"])
        method_timings.setdefault("DiagIF", {})[task_name] = time.time() - t_m
        print(f"  DiagIF: {method_timings['DiagIF'][task_name]:.2f}s")

        # BM25
        t_m = time.time()
        raw_sims["BM25"] = compute_bm25_raw(td["train_texts"], td["ref_texts"])
        method_timings.setdefault("BM25", {})[task_name] = time.time() - t_m
        print(f"  BM25: {method_timings['BM25'][task_name]:.2f}s")

        # ── Evaluate each method x scoring ──────────────────────────────
        pct_base = 0.45 + TASKS.index(task_name) * 0.15
        for method in METHODS:
            if method not in all_results:
                all_results[method] = {}
            sim_raw = raw_sims[method]

            for scoring in SCORINGS:
                if scoring not in all_results[method]:
                    all_results[method][scoring] = {}

                if scoring == "standard":
                    sim = sim_raw
                else:  # contrastive
                    sim = apply_contrastive(sim_raw)

                cell_result = evaluate_cell(
                    sim, task_name, td["pilot_train"], td["ref_data"]
                )
                cell_result["runtime_sec"] = round(method_timings[method][task_name], 2)
                all_results[method][scoring][task_name] = cell_result

        report_progress(
            f"task_{task_name}_done",
            f"Evaluated all methods on {task_name}",
            pct_base + 0.15
        )

    # ── Compute contrastive gains ───────────────────────────────────────
    print("\n=== Computing contrastive gains ===")
    contrastive_gains = {}
    for method in METHODS:
        contrastive_gains[method] = {}
        for task_name in TASKS:
            std_res = all_results[method]["standard"][task_name]
            con_res = all_results[method]["contrastive"][task_name]

            std_tau = std_res["continuous"]["kendall_tau"]
            con_tau = con_res["continuous"]["kendall_tau"]
            tau_gain = con_tau - std_tau

            std_rho = std_res["continuous"]["spearman_rho"]
            con_rho = con_res["continuous"]["spearman_rho"]
            rho_gain = con_rho - std_rho

            # Rank-based primary metric
            if task_name == "toxicity":
                std_primary = std_res["rank_based"]["AUPRC"]
                con_primary = con_res["rank_based"]["AUPRC"]
            else:
                std_primary = std_res["rank_based"]["Recall_at_50"]
                con_primary = con_res["rank_based"]["Recall_at_50"]
            primary_gain = (con_primary - std_primary) * 100  # pp

            contrastive_gains[method][task_name] = {
                "kendall_tau_gain": round(tau_gain, 6),
                "spearman_rho_gain": round(rho_gain, 6),
                "primary_metric_gain_pp": round(primary_gain, 2),
                "std_tau": round(std_tau, 6),
                "con_tau": round(con_tau, 6),
                "std_rho": round(std_rho, 6),
                "con_rho": round(con_rho, 6),
                "std_primary": round(std_primary, 6),
                "con_primary": round(con_primary, 6),
            }

    # ── 2x2 factorial focus: {TRAK, RepSim} x {standard, contrastive} ──
    print("\n=== 2x2 Factorial Analysis ===")
    factorial_2x2 = {}
    for method in ["TRAK", "RepSim"]:
        factorial_2x2[method] = {}
        for scoring in SCORINGS:
            factorial_2x2[method][scoring] = {}
            for task_name in TASKS:
                res = all_results[method][scoring][task_name]
                factorial_2x2[method][scoring][task_name] = {
                    "rank_based": res["rank_based"],
                    "continuous": res["continuous"],
                }

    # ── Pass criteria evaluation ────────────────────────────────────────
    # Pass: Kendall tau computable for all method-task pairs AND
    # at least one method shows |Kendall tau diff| > 0.01 between standard and contrastive
    all_tau_computable = True
    max_tau_diff = 0.0
    max_tau_diff_method = ""
    max_tau_diff_task = ""

    for method in METHODS:
        for task_name in TASKS:
            std_tau = all_results[method]["standard"][task_name]["continuous"]["kendall_tau"]
            con_tau = all_results[method]["contrastive"][task_name]["continuous"]["kendall_tau"]
            if np.isnan(std_tau) or np.isnan(con_tau):
                all_tau_computable = False
            diff = abs(con_tau - std_tau)
            if diff > max_tau_diff:
                max_tau_diff = diff
                max_tau_diff_method = method
                max_tau_diff_task = task_name

    pass_criteria = all_tau_computable and max_tau_diff > 0.01
    pass_detail = {
        "all_tau_computable": all_tau_computable,
        "max_tau_diff": round(max_tau_diff, 6),
        "max_tau_diff_method": max_tau_diff_method,
        "max_tau_diff_task": max_tau_diff_task,
        "threshold": 0.01,
        "passed": pass_criteria,
    }

    # ── Build final output ──────────────────────────────────────────────
    total_time = time.time() - t0
    final = {
        "task_id": TASK_ID,
        "candidate_id": "cand_a",
        "mode": "pilot",
        "n_train": PILOT_N_TRAIN,
        "seed": SEED,
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "gradient_params_D": D,
        "trak_k": TRAK_K,
        "logra_k": LOGRA_K,
        "bootstrap_B": BOOTSTRAP_B,
        "methods": METHODS,
        "scorings": SCORINGS,
        "tasks": TASKS,
        "results": all_results,
        "contrastive_gains": contrastive_gains,
        "factorial_2x2": factorial_2x2,
        "pass_criteria": pass_detail,
        "method_timings": method_timings,
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    out_path = os.path.join(FULL_DIR, f"{TASK_ID}.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also save to pilots dir
    pilots_dir = os.path.join(RESULTS_DIR, "pilots")
    os.makedirs(pilots_dir, exist_ok=True)
    with open(os.path.join(pilots_dir, f"{TASK_ID}_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    # ── Print summary tables ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FULL METHOD TOURNAMENT: Rank-Based Metrics")
    print("=" * 100)
    header = f"{'Method':<12}{'Scoring':<14}{'tox(AUPRC)':<14}{'cf(R@50)':<14}{'ft(R@50)':<14}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        for scoring in SCORINGS:
            tox = all_results[method][scoring]["toxicity"]["rank_based"]
            cf = all_results[method][scoring]["counterfact"]["rank_based"]
            ft = all_results[method][scoring]["ftrace"]["rank_based"]
            tox_v = tox.get("AUPRC", 0)
            cf_v = cf.get("Recall_at_50", 0)
            ft_v = ft.get("Recall_at_50", 0)
            print(f"{method:<12}{scoring:<14}{tox_v:<14.4f}{cf_v:<14.4f}{ft_v:<14.4f}")
        print()

    print("\n" + "=" * 100)
    print("FULL METHOD TOURNAMENT: Continuous Metrics (Kendall tau)")
    print("=" * 100)
    header = f"{'Method':<12}{'Scoring':<14}{'tox(tau)':<14}{'cf(tau)':<14}{'ft(tau)':<14}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        for scoring in SCORINGS:
            tox = all_results[method][scoring]["toxicity"]["continuous"]["kendall_tau"]
            cf = all_results[method][scoring]["counterfact"]["continuous"]["kendall_tau"]
            ft = all_results[method][scoring]["ftrace"]["continuous"]["kendall_tau"]
            print(f"{method:<12}{scoring:<14}{tox:<14.4f}{cf:<14.4f}{ft:<14.4f}")
        print()

    print("\n" + "=" * 100)
    print("CONTRASTIVE GAINS (Kendall tau: contrastive - standard)")
    print("=" * 100)
    header = f"{'Method':<12}{'tox(delta_tau)':<18}{'cf(delta_tau)':<18}{'ft(delta_tau)':<18}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        tg = contrastive_gains[method]["toxicity"]["kendall_tau_gain"]
        cg = contrastive_gains[method]["counterfact"]["kendall_tau_gain"]
        fg = contrastive_gains[method]["ftrace"]["kendall_tau_gain"]
        print(f"{method:<12}{tg:<+18.6f}{cg:<+18.6f}{fg:<+18.6f}")

    print(f"\nPass criteria: {pass_detail}")
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 100)

    mark_done(
        status="success",
        summary=f"8-method tournament completed. Pass={pass_criteria}. "
                f"Max tau diff={max_tau_diff:.6f} ({max_tau_diff_method}/{max_tau_diff_task}). "
                f"Runtime={total_time:.1f}s."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
