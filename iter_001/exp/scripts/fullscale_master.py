#!/usr/bin/env python3
"""
CRA Full-Scale Master Experiment Script
========================================
Runs all 12 tasks at full scale on a single GPU (GPU 3).
Adapts from pilot scripts but uses full training pools:
  - counterfact: N=5473
  - toxicity: N=10187 (train), plus ref samples
  - ftrace: full training set

Tasks executed sequentially:
  Phase 1: Extract representations + gradients at full scale (shared across tasks)
  Phase 2: P1 - FM2 continuous metrics (all methods tournament)
  Phase 3: P1 - FM2 contamination injection
  Phase 4: P1 - FM2 interaction analysis (CPU only, from P1 results)
  Phase 5: P2 - Eigenspectrum (already done at full scale - verify/skip)
  Phase 6: P2 - TRAK dimension sweep at full scale
  Phase 7: P2 - RepSim dimension sweep at full scale
  Phase 8: P3 - Retrieval baselines at full scale
  Phase 9: P4 - Gap decomposition
  Phase 10: P4 - Layer sweep
  Phase 11: P4 - Cosine vs Euclidean
  Phase 12: P5 - PCA whitened attribution
  Phase 13: Final analysis (CPU only)
"""

import os, sys, json, time, gc, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_curve, auc, ndcg_score
from sklearn.covariance import LedoitWolf
from scipy.stats import kendalltau, spearmanr, f_oneway
from scipy.linalg import eigh
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda:0"  # Will be mapped to GPU 3 via CUDA_VISIBLE_DEVICES
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
PYTHIA70M_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-70m"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048
LOGRA_K = 256
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 2  # Small batch for ~9GB available GPU memory
KNN_K = 50
SEEDS = [42, 123, 456]  # Full-scale uses 3 seeds

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Lifecycle helpers ───────────────────────────────────────────────────
def write_pid(task_id):
    Path(RESULTS_DIR, f"{task_id}.pid").write_text(str(os.getpid()))

def report_progress(task_id, stage, detail="", pct=0.0, metric=None):
    Path(RESULTS_DIR, f"{task_id}_PROGRESS.json").write_text(json.dumps({
        "task_id": task_id, "stage": stage, "detail": detail,
        "pct": pct, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(task_id, status="success", summary=""):
    pid_f = Path(RESULTS_DIR) / f"{task_id}.pid"
    if pid_f.exists(): pid_f.unlink()
    prog_f = Path(RESULTS_DIR) / f"{task_id}_PROGRESS.json"
    final = {}
    if prog_f.exists():
        try: final = json.loads(prog_f.read_text())
        except: pass
    Path(RESULTS_DIR, f"{task_id}_DONE").write_text(json.dumps({
        "task_id": task_id, "status": status, "summary": summary,
        "final_progress": final, "timestamp": datetime.now().isoformat(),
    }))

# ── Metric helpers ──────────────────────────────────────────────────────
def compute_auprc(scores, unsafe_indices, n_total):
    labels = np.zeros(n_total); labels[list(unsafe_indices)] = 1
    if sum(labels) == 0: return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))

def compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50):
    recalls, mrrs = [], []
    for scores, fi in zip(scores_per_ref, fact_indices_per_ref):
        if not fi: continue
        si = np.argsort(-np.array(scores))
        topk = set(si[:k].tolist())
        recalls.append(len([i for i in fi if i in topk]) / len(fi))
        r2r = {idx: rank+1 for rank, idx in enumerate(si)}
        ranks = [r2r[i] for i in fi if i in r2r]
        mrrs.append(1.0/min(ranks) if ranks else 0.0)
    if not recalls: return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(mrrs))

def bootstrap_auprc(scores, unsafe_indices, n_total, n_boot=BOOTSTRAP_B):
    rng = np.random.RandomState(SEED+1234)
    labels = np.zeros(n_total); labels[list(unsafe_indices)] = 1
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n_total, n_total, replace=True)
        if labels[idx].sum() == 0: vals.append(0.0); continue
        p, r, _ = precision_recall_curve(labels[idx], scores[idx])
        vals.append(float(auc(r, p)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def bootstrap_factual(scores_per_ref, fact_indices_per_ref, n_boot=BOOTSTRAP_B, k=50):
    rng = np.random.RandomState(SEED+2345)
    n_ref = len(scores_per_ref)
    boot_r, boot_m = [], []
    for _ in range(n_boot):
        idx = rng.choice(n_ref, n_ref, replace=True)
        spr = [scores_per_ref[i] for i in idx]
        fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(spr, fi, k=k)
        boot_r.append(r); boot_m.append(m)
    return ([float(np.percentile(boot_r,2.5)), float(np.percentile(boot_r,97.5))],
            [float(np.percentile(boot_m,2.5)), float(np.percentile(boot_m,97.5))])

def compute_ndcg(scores, relevance, k=50):
    try: return float(ndcg_score([relevance], [scores], k=k))
    except: return 0.0

def compute_continuous_metrics_toxicity(scores, labels):
    """Continuous metrics for toxicity (binary labels)."""
    tau_val, tau_p = kendalltau(labels, scores)
    rho_val, rho_p = spearmanr(labels, scores)
    ndcg_val = compute_ndcg(scores, labels, k=50)
    return {
        "kendall_tau": float(tau_val) if not np.isnan(tau_val) else 0.0,
        "kendall_p": float(tau_p) if not np.isnan(tau_p) else 1.0,
        "spearman_rho": float(rho_val) if not np.isnan(rho_val) else 0.0,
        "spearman_p": float(rho_p) if not np.isnan(rho_p) else 1.0,
        "ndcg_at_50": ndcg_val,
    }

def compute_continuous_metrics_factual(sim_matrix, train_data, ref_data, task_name):
    """Continuous metrics for factual tasks (per-ref correlations)."""
    n_train, n_ref = sim_matrix.shape
    taus, rhos, ndcgs = [], [], []
    for j in range(n_ref):
        gt_j = _get_ground_truth(task_name, train_data, ref_data[j], n_train)
        scores_j = sim_matrix[:, j]
        if gt_j.sum() > 0 and gt_j.sum() < len(gt_j):
            t, _ = kendalltau(gt_j, scores_j)
            r, _ = spearmanr(gt_j, scores_j)
            if not np.isnan(t): taus.append(t)
            if not np.isnan(r): rhos.append(r)
            ndcgs.append(compute_ndcg(scores_j, gt_j, k=50))
    return {
        "kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "kendall_tau_std": float(np.std(taus)) if taus else 0.0,
        "spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
        "spearman_rho_std": float(np.std(rhos)) if rhos else 0.0,
        "ndcg_at_50": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_evaluated_refs": len(taus),
    }

def _get_ground_truth(task_name, train_data, ref_sample, n_train):
    if task_name == "counterfact":
        return np.array([
            1.0 if (train_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                    and train_data[i]["true_entity"] == ref_sample["true_entity"])
            else 0.0 for i in range(n_train)
        ])
    else:  # ftrace
        ref_facts_raw = ref_sample.get("facts", [])
        if isinstance(ref_facts_raw, str):
            ref_facts_raw = [f.strip() for f in ref_facts_raw.split(",") if f.strip()]
        elif isinstance(ref_facts_raw, list):
            flat = []
            for f in ref_facts_raw:
                if isinstance(f, str): flat.extend([x.strip() for x in f.split(",") if x.strip()])
            ref_facts_raw = flat
        ref_facts = set(ref_facts_raw)
        gt = np.zeros(n_train)
        for i in range(n_train):
            tf = train_data[i].get("facts", [])
            if isinstance(tf, str): tf = [x.strip() for x in tf.split(",") if x.strip()]
            elif isinstance(tf, list):
                flat = []
                for f in tf:
                    if isinstance(f, str): flat.extend([x.strip() for x in f.split(",") if x.strip()])
                tf = flat
            if set(tf) & ref_facts: gt[i] = 1.0
        return gt

def _get_fact_indices(task_name, train_data, ref_data):
    """Get per-ref list of fact-sharing train indices for factual tasks."""
    fact_indices_per_ref = []
    for ref_sample in ref_data:
        gt = _get_ground_truth(task_name, train_data, ref_sample, len(train_data))
        fact_indices_per_ref.append([i for i in range(len(train_data)) if gt[i] > 0])
    return fact_indices_per_ref

# ── Data loading (FULL SCALE) ──────────────────────────────────────────
def load_all_tasks_full():
    """Load all DATE-LM tasks at FULL scale."""
    print("=" * 60)
    print("Loading DATE-LM datasets at FULL SCALE")
    print("=" * 60)
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

# ── Model loading ───────────────────────────────────────────────────────
def load_model_pythia1b():
    gc.collect(); torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, torch_dtype=torch.float16,
        device_map=DEVICE, low_cpu_mem_usage=True
    )
    model.eval()
    free_mem = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"Pythia-1B loaded. Hidden dim={model.config.hidden_size}. GPU free: {free_mem:.0f}MiB")
    return model, tok

def unload_model(model):
    del model; gc.collect(); torch.cuda.empty_cache()

# ── Representation extraction ───────────────────────────────────────────
def extract_representations(model, tok, texts, layer_idx=-1, batch_size=BATCH_SIZE, desc=""):
    """Extract representations from a specific layer. layer_idx=-1 means last layer."""
    all_reps = []
    for idx in range(0, len(texts), batch_size):
        batch_texts = texts[idx:idx+batch_size]
        inputs = tok(batch_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(input_ids=inputs["input_ids"],
                          attention_mask=inputs["attention_mask"],
                          output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_reps.append(pooled.float().cpu())
        del inputs, outputs, hidden, mask, pooled
        torch.cuda.empty_cache()
        if (idx // batch_size) % 100 == 0 and idx > 0:
            print(f"  [{desc}] {idx}/{len(texts)} extracted")
    reps = torch.cat(all_reps, dim=0)
    print(f"  [{desc}] Done: {reps.shape[0]} reps, dim={reps.shape[1]}")
    return reps

# ── Gradient extraction ─────────────────────────────────────────────────
def setup_target_params(model):
    target_params, target_names = [], []
    for name, p in model.named_parameters():
        if any(sub in name for sub in [
            "layers.15.attention.dense.weight",
            "layers.15.mlp.dense_4h_to_h.weight"
        ]):
            p.requires_grad_(True)
            target_params.append(p); target_names.append(name)
        else:
            p.requires_grad_(False)
    D = sum(p.numel() for p in target_params)
    print(f"Target params: D={D/1e6:.2f}M from {target_names}")
    return target_params, target_names, D

def restore_grad_flags(model):
    for p in model.parameters(): p.requires_grad_(True)

def compute_raw_gradients(model, tok, texts, target_params, desc="", max_samples=None):
    """Compute per-sample gradients. For large N, saves to disk in chunks."""
    n = len(texts) if max_samples is None else min(len(texts), max_samples)
    all_grads = []
    for idx in range(n):
        inp = tok(texts[idx], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        model.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(input_ids=inp["input_ids"],
                       attention_mask=inp["attention_mask"],
                       labels=inp["input_ids"])
        out.loss.backward()
        grad_flat = torch.cat([p.grad.detach().flatten().float().cpu() for p in target_params])
        all_grads.append(grad_flat)
        model.zero_grad(set_to_none=True)
        if (idx+1) % 100 == 0:
            print(f"  [{desc}] Gradients: {idx+1}/{n}")
            torch.cuda.empty_cache()
    return torch.stack(all_grads)

# ── Method implementations ──────────────────────────────────────────────
def compute_repsim_raw(train_reps, ref_reps):
    """RepSim: cosine similarity [n_train, n_ref]."""
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    return (tn @ rn.T).numpy()

def compute_repsim_euclidean(train_reps, ref_reps):
    """RepSim with negative Euclidean distance [n_train, n_ref]."""
    # -||a - b||^2 = 2 a.b - ||a||^2 - ||b||^2
    aa = (train_reps ** 2).sum(dim=-1, keepdim=True)  # [n_train,1]
    bb = (ref_reps ** 2).sum(dim=-1, keepdim=True)    # [n_ref,1]
    dist_sq = aa + bb.T - 2 * (train_reps @ ref_reps.T)
    return (-dist_sq).numpy()

def compute_repsim_dot(train_reps, ref_reps):
    """RepSim with dot product [n_train, n_ref]."""
    return (train_reps @ ref_reps.T).numpy()

def compute_knn_raw(train_reps, ref_reps, k=KNN_K):
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    sim = (tn @ rn.T).numpy()
    result = np.zeros_like(sim)
    for j in range(sim.shape[1]):
        topk_idx = np.argsort(-sim[:, j])[:k]
        result[topk_idx, j] = sim[topk_idx, j]
    return result

def compute_trak_raw(train_grads, ref_grads, k=TRAK_K, proj_type="random", proj_matrix=None):
    """TRAK with random or PCA projection."""
    D = train_grads.shape[1]
    if proj_type == "random" and proj_matrix is None:
        rng = np.random.RandomState(SEED)
        proj_matrix = torch.tensor(rng.randn(D, k) / np.sqrt(k), dtype=torch.float32)
    elif proj_type == "pca":
        if proj_matrix is None:
            raise ValueError("PCA projection requires pre-computed eigenvectors")

    train_proj = (train_grads @ proj_matrix).numpy()
    ref_proj = (ref_grads @ proj_matrix).numpy()
    # Normalize
    train_norm = train_proj / (np.linalg.norm(train_proj, axis=1, keepdims=True) + 1e-10)
    ref_norm = ref_proj / (np.linalg.norm(ref_proj, axis=1, keepdims=True) + 1e-10)
    return train_norm @ ref_norm.T

def compute_logra_raw(train_grads, ref_grads, k=LOGRA_K):
    """LoGra: low-rank gradient approximation."""
    # Use SVD to get top-k components
    U, S, Vt = torch.linalg.svd(train_grads, full_matrices=False)
    proj = Vt[:k].T  # [D, k]
    train_proj = (train_grads @ proj).numpy()
    ref_proj = (ref_grads @ proj).numpy()
    train_norm = train_proj / (np.linalg.norm(train_proj, axis=1, keepdims=True) + 1e-10)
    ref_norm = ref_proj / (np.linalg.norm(ref_proj, axis=1, keepdims=True) + 1e-10)
    return train_norm @ ref_norm.T

def compute_dda_raw(train_grads, ref_grads):
    """DDA: debias + denoise (mean-centered gradients + low-rank)."""
    train_centered = train_grads - train_grads.mean(dim=0, keepdim=True)
    ref_centered = ref_grads - ref_grads.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(train_centered, full_matrices=False)
    k = min(256, min(train_centered.shape) - 1)
    proj = Vt[:k].T
    train_proj = (train_centered @ proj).numpy()
    ref_proj = (ref_centered @ proj).numpy()
    train_norm = train_proj / (np.linalg.norm(train_proj, axis=1, keepdims=True) + 1e-10)
    ref_norm = ref_proj / (np.linalg.norm(ref_proj, axis=1, keepdims=True) + 1e-10)
    return train_norm @ ref_norm.T

def compute_rawdotif(train_grads, ref_grads):
    """Raw dot product IF (no Hessian)."""
    return (train_grads @ ref_grads.T).numpy()

def compute_diagif(train_grads, ref_grads):
    """Diagonal Fisher IF."""
    fisher_diag = (train_grads ** 2).mean(dim=0) + 1e-8
    scaled_train = train_grads / fisher_diag.sqrt().unsqueeze(0)
    return (scaled_train @ ref_grads.T).numpy()

def compute_bm25_scores(train_texts, ref_texts):
    """BM25 lexical baseline."""
    from rank_bm25 import BM25Okapi
    tokenized_train = [t.lower().split() for t in train_texts]
    bm25 = BM25Okapi(tokenized_train)
    sim = np.zeros((len(train_texts), len(ref_texts)))
    for j, ref_text in enumerate(ref_texts):
        scores = bm25.get_scores(ref_text.lower().split())
        sim[:, j] = scores
    return sim

# ── Evaluation wrapper ──────────────────────────────────────────────────
def evaluate_method(sim_matrix, task_name, train_data, ref_data, scoring="standard"):
    """Full evaluation: rank-based + continuous metrics."""
    n_train = sim_matrix.shape[0]

    # Apply contrastive scoring if requested
    if scoring == "contrastive":
        sim_matrix = sim_matrix - sim_matrix.mean(axis=0, keepdims=True)

    result = {"scoring": scoring}

    if task_name == "toxicity":
        unsafe_idx = [i for i in range(n_train) if train_data[i]["type"] == "Unsafe"]
        avg_scores = sim_matrix.mean(axis=1)

        auprc = compute_auprc(avg_scores, unsafe_idx, n_train)
        ci = bootstrap_auprc(avg_scores, unsafe_idx, n_train)
        labels = np.array([1.0 if train_data[i]["type"] == "Unsafe" else 0.0 for i in range(n_train)])
        cont = compute_continuous_metrics_toxicity(avg_scores, labels)

        result["rank_based"] = {"AUPRC": auprc, "CI_lower": ci[0], "CI_upper": ci[1],
                                "n_unsafe": len(unsafe_idx), "n_train": n_train}
        result["continuous"] = cont
        result["score_stats"] = {"mean": float(avg_scores.mean()), "std": float(avg_scores.std()),
                                 "min": float(avg_scores.min()), "max": float(avg_scores.max())}

    elif task_name in ["counterfact", "ftrace"]:
        fact_indices = _get_fact_indices(task_name, train_data, ref_data)
        scores_per_ref = [sim_matrix[:, j].tolist() for j in range(sim_matrix.shape[1])]

        r50, mrr = compute_factual_metrics(scores_per_ref, fact_indices, k=50)
        ci_r, ci_m = bootstrap_factual(scores_per_ref, fact_indices)
        cont = compute_continuous_metrics_factual(sim_matrix, train_data, ref_data, task_name)

        result["rank_based"] = {"R_at_50": r50, "MRR": mrr,
                                "R50_CI": ci_r, "MRR_CI": ci_m}
        result["continuous"] = cont

    return result

# ═══════════════════════════════════════════════════════════════════════
#  TASK P1: FM2 Continuous Metrics (Full Method Tournament)
# ═══════════════════════════════════════════════════════════════════════
def run_p1_fm2_continuous_metrics(tasks, model, tok):
    task_id = "p1_fm2_continuous_metrics"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "seed": SEED, "model": MODEL_NAME,
        "hidden_dim": 2048,
        "methods": ["RepSim", "TRAK", "LoGra", "DDA", "RawDotIF", "DiagIF", "kNN", "BM25"],
        "scorings": ["standard", "contrastive"],
        "tasks": ["toxicity", "counterfact", "ftrace"],
        "results": {},
    }

    for task_name, task_data in tasks.items():
        print(f"\n--- Task: {task_name} ---")
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        report_progress(task_id, f"processing_{task_name}", f"N_train={n_train}, N_ref={n_ref}")

        # Format texts
        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        # Cache key
        cache_reps_train = os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt")
        cache_reps_ref = os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt")
        cache_grads_train = os.path.join(CACHE_DIR, f"grads_train_{task_name}_full.pt")
        cache_grads_ref = os.path.join(CACHE_DIR, f"grads_ref_{task_name}_full.pt")

        # Extract representations (with caching)
        if os.path.exists(cache_reps_train):
            print(f"  Loading cached train reps: {cache_reps_train}")
            train_reps = torch.load(cache_reps_train, weights_only=True)
        else:
            train_reps = extract_representations(model, tok, train_texts, desc=f"{task_name}/train")
            torch.save(train_reps, cache_reps_train)

        if os.path.exists(cache_reps_ref):
            print(f"  Loading cached ref reps: {cache_reps_ref}")
            ref_reps = torch.load(cache_reps_ref, weights_only=True)
        else:
            ref_reps = extract_representations(model, tok, ref_texts, desc=f"{task_name}/ref")
            torch.save(ref_reps, cache_reps_ref)

        # Extract gradients (with caching)
        target_params, target_names, D = setup_target_params(model)

        if os.path.exists(cache_grads_train):
            print(f"  Loading cached train grads: {cache_grads_train}")
            train_grads = torch.load(cache_grads_train, weights_only=True)
        else:
            train_grads = compute_raw_gradients(model, tok, train_texts, target_params,
                                                desc=f"{task_name}/train_grads")
            torch.save(train_grads, cache_grads_train)

        if os.path.exists(cache_grads_ref):
            print(f"  Loading cached ref grads: {cache_grads_ref}")
            ref_grads = torch.load(cache_grads_ref, weights_only=True)
        else:
            ref_grads = compute_raw_gradients(model, tok, ref_texts, target_params,
                                              desc=f"{task_name}/ref_grads")
            torch.save(ref_grads, cache_grads_ref)

        restore_grad_flags(model)

        results["results"][task_name] = {"n_train": n_train, "n_ref": n_ref}

        # Compute each method
        method_sims = {}

        # RepSim (cosine)
        print(f"  Computing RepSim...")
        method_sims["RepSim"] = compute_repsim_raw(train_reps, ref_reps)

        # kNN
        print(f"  Computing kNN...")
        method_sims["kNN"] = compute_knn_raw(train_reps, ref_reps)

        # TRAK (random, k=2048)
        print(f"  Computing TRAK (random, k={TRAK_K})...")
        method_sims["TRAK"] = compute_trak_raw(train_grads, ref_grads, k=TRAK_K, proj_type="random")

        # LoGra
        print(f"  Computing LoGra (k={LOGRA_K})...")
        method_sims["LoGra"] = compute_logra_raw(train_grads, ref_grads, k=LOGRA_K)

        # DDA
        print(f"  Computing DDA...")
        method_sims["DDA"] = compute_dda_raw(train_grads, ref_grads)

        # Raw Dot IF
        print(f"  Computing RawDotIF...")
        method_sims["RawDotIF"] = compute_rawdotif(train_grads, ref_grads)

        # Diag IF
        print(f"  Computing DiagIF...")
        method_sims["DiagIF"] = compute_diagif(train_grads, ref_grads)

        # BM25
        print(f"  Computing BM25...")
        method_sims["BM25"] = compute_bm25_scores(train_texts, ref_texts)

        # Evaluate each method x scoring
        for method_name, sim in method_sims.items():
            results["results"][task_name][method_name] = {}
            for scoring in ["standard", "contrastive"]:
                sim_copy = sim.copy()
                eval_result = evaluate_method(sim_copy, task_name, train_data, ref_data, scoring)
                results["results"][task_name][method_name][scoring] = eval_result

                # Print key metric
                if task_name == "toxicity":
                    key_val = eval_result["rank_based"]["AUPRC"]
                    cont_val = eval_result["continuous"]["kendall_tau"]
                else:
                    key_val = eval_result["rank_based"]["R_at_50"]
                    cont_val = eval_result["continuous"]["kendall_tau"]
                print(f"    {method_name}/{scoring}: key={key_val:.4f}, tau={cont_val:.4f}")

        # Free gradient memory between tasks
        del train_grads, ref_grads
        gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    # Save
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")

    mark_done(task_id, summary=f"Full-scale FM2 continuous metrics completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P1: FM2 Contamination Injection
# ═══════════════════════════════════════════════════════════════════════
def run_p1_fm2_contamination_injection(tasks, model, tok):
    task_id = "p1_fm2_contamination_injection"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "seed": SEED, "model": MODEL_NAME, "alphas": alphas,
        "injection_modes": ["uniform", "structured", "magnitude_proportional"],
        "results": {},
    }

    for task_name in ["counterfact", "toxicity"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)

        report_progress(task_id, f"contamination_{task_name}", f"N={n_train}")

        n_ref = len(ref_data)
        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        # Load cached reps and grads
        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)

        results["results"][task_name] = {"n_train": n_train, "n_ref": n_ref}

        for method_name in ["RepSim", "TRAK"]:
            print(f"\n  {task_name}/{method_name} contamination injection")

            if method_name == "RepSim":
                base_sim = compute_repsim_raw(train_reps, ref_reps)
            else:
                train_grads = torch.load(os.path.join(CACHE_DIR, f"grads_train_{task_name}_full.pt"), weights_only=True)
                ref_grads = torch.load(os.path.join(CACHE_DIR, f"grads_ref_{task_name}_full.pt"), weights_only=True)
                base_sim = compute_trak_raw(train_grads, ref_grads, k=TRAK_K)
                del train_grads, ref_grads; gc.collect()

            method_results = {}

            for injection_mode in ["uniform", "structured", "magnitude_proportional"]:
                mode_results = {}

                for alpha in alphas:
                    sim_contaminated = base_sim.copy()

                    if injection_mode == "uniform":
                        mu = base_sim.mean(axis=0, keepdims=True)
                        sim_contaminated = base_sim + alpha * mu
                    elif injection_mode == "structured":
                        rng = np.random.RandomState(SEED + int(alpha * 100))
                        noise = rng.randn(*base_sim.shape) * base_sim.std()
                        sim_contaminated = base_sim + alpha * noise
                    elif injection_mode == "magnitude_proportional":
                        mag = np.abs(base_sim)
                        sim_contaminated = base_sim + alpha * mag * np.sign(base_sim.mean(axis=0, keepdims=True))

                    # Corrected (contrastive)
                    sim_corrected = sim_contaminated - sim_contaminated.mean(axis=0, keepdims=True)

                    # Evaluate both
                    eval_contam = evaluate_method(sim_contaminated, task_name, train_data, ref_data, "standard")
                    eval_correct = evaluate_method(sim_corrected, task_name, train_data, ref_data, "standard")

                    mode_results[str(alpha)] = {
                        "contaminated": eval_contam,
                        "corrected": eval_correct,
                    }

                    if task_name == "toxicity":
                        c_val = eval_contam["rank_based"]["AUPRC"]
                        r_val = eval_correct["rank_based"]["AUPRC"]
                    else:
                        c_val = eval_contam["rank_based"]["R_at_50"]
                        r_val = eval_correct["rank_based"]["R_at_50"]
                    print(f"    {injection_mode} alpha={alpha}: contam={c_val:.4f}, corrected={r_val:.4f}")

                method_results[injection_mode] = mode_results

            results["results"][task_name][method_name] = method_results

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Contamination injection completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P1: FM2 Interaction Analysis (CPU only)
# ═══════════════════════════════════════════════════════════════════════
def run_p1_fm2_interaction_analysis(p1_results):
    task_id = "p1_fm2_interaction_analysis"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE (CPU)")
    print("=" * 60)
    start_time = time.time()

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "anova_results": {},
    }

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_results = p1_results["results"].get(task_name, {})

        # 2x2 factorial: {RepSim, TRAK} x {standard, contrastive}
        cells = {}
        for method in ["RepSim", "TRAK"]:
            for scoring in ["standard", "contrastive"]:
                mr = task_results.get(method, {}).get(scoring, {})
                cont = mr.get("continuous", {})
                tau = cont.get("kendall_tau", 0.0)
                cells[f"{method}_{scoring}"] = tau

        # Extract values for ANOVA
        rep_std = cells.get("RepSim_standard", 0.0)
        rep_con = cells.get("RepSim_contrastive", 0.0)
        trak_std = cells.get("TRAK_standard", 0.0)
        trak_con = cells.get("TRAK_contrastive", 0.0)

        fm1_main = ((rep_std + rep_con) / 2) - ((trak_std + trak_con) / 2)
        fm2_main = ((rep_con + trak_con) / 2) - ((rep_std + trak_std) / 2)
        interaction = (rep_con - rep_std) - (trak_con - trak_std)

        grand_mean = (rep_std + rep_con + trak_std + trak_con) / 4
        ss_total = sum((v - grand_mean)**2 for v in [rep_std, rep_con, trak_std, trak_con])
        ss_fm1 = 2 * fm1_main**2
        ss_fm2 = 2 * fm2_main**2
        ss_int = interaction**2

        eta_fm1 = ss_fm1 / ss_total if ss_total > 0 else 0.0
        eta_fm2 = ss_fm2 / ss_total if ss_total > 0 else 0.0

        results["anova_results"][task_name] = {
            "cells": cells,
            "FM1_main_effect": fm1_main,
            "FM2_main_effect": fm2_main,
            "interaction": interaction,
            "eta_sq_FM1": eta_fm1,
            "eta_sq_FM2": eta_fm2,
            "ss_total": ss_total,
        }
        print(f"  {task_name}: FM1={fm1_main:.4f}, FM2={fm2_main:.4f}, interaction={interaction:.4f}, eta_FM1={eta_fm1:.4f}")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Interaction analysis completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P2: TRAK Dimension Sweep at Full Scale
# ═══════════════════════════════════════════════════════════════════════
def run_p2_trak_dim_sweep(tasks, model, tok):
    task_id = "p2_trak_dim_sweep_fullscale"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    # Use counterfact as primary task
    task_name = "counterfact"
    task_data = tasks[task_name]
    train_data = task_data["train"]
    ref_data = task_data["ref"]
    n_train = len(train_data)

    # Load cached
    train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
    ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
    train_grads = torch.load(os.path.join(CACHE_DIR, f"grads_train_{task_name}_full.pt"), weights_only=True)
    ref_grads = torch.load(os.path.join(CACHE_DIR, f"grads_ref_{task_name}_full.pt"), weights_only=True)

    D = train_grads.shape[1]
    repsim_sim = compute_repsim_raw(train_reps, ref_reps)
    repsim_eval = evaluate_method(repsim_sim, task_name, train_data, ref_data, "standard")

    random_k_values = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    pca_k_values = [32, 64, 128, 256, 512, 1024, 2048]

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "task_name": task_name, "n_train": n_train,
        "model": MODEL_NAME, "gradient_dim_D": D,
        "repsim_reference": repsim_eval,
        "random_projection": {},
        "pca_projection": {},
    }

    # Random projection sweep
    print(f"\n  Random projection sweep (D={D})...")
    for k in random_k_values:
        if k > D:
            # Pad or skip
            print(f"    k={k} > D={D}, using D")
            k_actual = D
        else:
            k_actual = k
        rng = np.random.RandomState(SEED)
        proj = torch.tensor(rng.randn(D, k_actual) / np.sqrt(k_actual), dtype=torch.float32)
        sim = compute_trak_raw(train_grads, ref_grads, k=k_actual, proj_type="random", proj_matrix=proj)
        ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
        results["random_projection"][str(k)] = ev
        key = ev["rank_based"]["R_at_50"]
        tau = ev["continuous"]["kendall_tau"]
        print(f"    k={k}: R@50={key:.4f}, tau={tau:.4f}")

    # PCA projection sweep
    print(f"\n  PCA projection sweep...")
    # Compute gradient covariance eigenvectors
    actual_rank = min(n_train, D)
    report_progress(task_id, "pca_eigenvectors", f"Computing SVD rank={actual_rank}")
    U, S, Vt = torch.linalg.svd(train_grads, full_matrices=False)

    for k in pca_k_values:
        k_actual = min(k, actual_rank)
        pca_proj = Vt[:k_actual].T  # [D, k_actual]
        sim = compute_trak_raw(train_grads, ref_grads, k=k_actual, proj_type="pca", proj_matrix=pca_proj)
        ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
        results["pca_projection"][str(k)] = {"k_actual": k_actual, **ev}
        key = ev["rank_based"]["R_at_50"]
        tau = ev["continuous"]["kendall_tau"]
        print(f"    k={k} (actual={k_actual}): R@50={key:.4f}, tau={tau:.4f}")

    del train_grads, ref_grads, U, S, Vt
    gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"TRAK dim sweep completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P2: RepSim Dimension Sweep at Full Scale
# ═══════════════════════════════════════════════════════════════════════
def run_p2_repsim_dim_sweep(tasks):
    task_id = "p2_repsim_dim_sweep_fullscale"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    k_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "model": MODEL_NAME, "k_values": k_values,
        "results": {},
    }

    for task_name in ["counterfact", "toxicity", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)

        d = train_reps.shape[1]

        # Full-dimensional baseline
        full_sim = compute_repsim_raw(train_reps, ref_reps)
        full_eval = evaluate_method(full_sim, task_name, train_data, ref_data, "standard")

        task_results = {"n_train": n_train, "d": d, "full_dim": full_eval, "pca_sweep": {}}

        # PCA on representations
        mean_reps = train_reps.mean(dim=0)
        centered = train_reps - mean_reps
        # Use torch.linalg.svd for PCA
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        for k in k_values:
            k_actual = min(k, min(n_train, d))
            pca_proj = Vt[:k_actual]  # [k, d]

            train_pca = centered @ pca_proj.T  # [n_train, k]
            ref_centered = ref_reps - mean_reps
            ref_pca = ref_centered @ pca_proj.T  # [n_ref, k]

            # Cosine similarity in PCA space
            tn = F.normalize(train_pca, dim=-1)
            rn = F.normalize(ref_pca, dim=-1)
            sim = (tn @ rn.T).numpy()

            ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
            task_results["pca_sweep"][str(k)] = {"k_actual": k_actual, **ev}

            if task_name == "toxicity":
                key = ev["rank_based"]["AUPRC"]
            else:
                key = ev["rank_based"]["R_at_50"]
            print(f"  {task_name} PCA k={k}: key={key:.4f}")

        results["results"][task_name] = task_results
        del U, S, Vt; gc.collect()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"RepSim dim sweep completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P3: Retrieval Baselines
# ═══════════════════════════════════════════════════════════════════════
def run_p3_retrieval_baselines(tasks, model, tok):
    task_id = "p3_retrieval_baselines"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "methods": ["Contriever", "GTR-T5", "BM25"],
        "results": {},
    }

    # Unload main model to free GPU memory for retrieval models
    # Actually we'll keep it and load retrieval models one at a time

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        results["results"][task_name] = {"n_train": n_train, "n_ref": n_ref}

        # BM25 (CPU only)
        print(f"\n  {task_name}: BM25...")
        bm25_sim = compute_bm25_scores(train_texts, ref_texts)
        bm25_eval = evaluate_method(bm25_sim, task_name, train_data, ref_data, "standard")
        results["results"][task_name]["BM25"] = bm25_eval
        if task_name == "toxicity":
            print(f"    BM25: AUPRC={bm25_eval['rank_based']['AUPRC']:.4f}")
        else:
            print(f"    BM25: R@50={bm25_eval['rank_based']['R_at_50']:.4f}")

    # Contriever
    print(f"\n  Loading Contriever...")
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModel
    contriever_tok = AutoTokenizer.from_pretrained("facebook/contriever")
    contriever_model = AutoModel.from_pretrained("facebook/contriever").half().to(DEVICE)
    contriever_model.eval()

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        print(f"\n  {task_name}: Contriever encoding {n_train} train + {n_ref} ref...")

        def encode_contriever(texts, batch_sz=8):
            all_emb = []
            for i in range(0, len(texts), batch_sz):
                batch = texts[i:i+batch_sz]
                inp = contriever_tok(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    out = contriever_model(**inp)
                    # Mean pooling
                    mask = inp["attention_mask"].unsqueeze(-1).float()
                    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    all_emb.append(emb.float().cpu())
                del inp, out; torch.cuda.empty_cache()
                if (i // batch_sz) % 50 == 0 and i > 0:
                    print(f"    {i}/{len(texts)}")
            return torch.cat(all_emb, dim=0)

        train_emb = encode_contriever(train_texts)
        ref_emb = encode_contriever(ref_texts)

        sim = (F.normalize(train_emb, dim=-1) @ F.normalize(ref_emb, dim=-1).T).numpy()
        ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
        results["results"][task_name]["Contriever"] = ev

        if task_name == "toxicity":
            print(f"    Contriever: AUPRC={ev['rank_based']['AUPRC']:.4f}")
        else:
            print(f"    Contriever: R@50={ev['rank_based']['R_at_50']:.4f}")

    del contriever_model, contriever_tok; gc.collect(); torch.cuda.empty_cache()

    # GTR-T5
    print(f"\n  Loading GTR-T5...")
    from sentence_transformers import SentenceTransformer
    gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device=DEVICE)

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        print(f"\n  {task_name}: GTR-T5 encoding {n_train} train + {n_ref} ref...")

        train_emb = torch.tensor(gtr.encode(train_texts, batch_size=8, show_progress_bar=False))
        ref_emb = torch.tensor(gtr.encode(ref_texts, batch_size=8, show_progress_bar=False))

        sim = (F.normalize(train_emb, dim=-1) @ F.normalize(ref_emb, dim=-1).T).numpy()
        ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
        results["results"][task_name]["GTR-T5"] = ev

        if task_name == "toxicity":
            print(f"    GTR-T5: AUPRC={ev['rank_based']['AUPRC']:.4f}")
        else:
            print(f"    GTR-T5: R@50={ev['rank_based']['R_at_50']:.4f}")

    del gtr; gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Retrieval baselines completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P4: Gap Decomposition
# ═══════════════════════════════════════════════════════════════════════
def run_p4_gap_decomposition(tasks, model, tok):
    task_id = "p4_gap_decomposition"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    task_name = "counterfact"
    task_data = tasks[task_name]
    train_data = task_data["train"]
    ref_data = task_data["ref"]
    n_train = len(train_data)

    train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
    ref_texts = [task_data["fmt"](ref_data[i]) for i in range(len(ref_data))]

    # Load cached full representations
    train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
    ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)

    # RepSim baseline
    repsim_sim = compute_repsim_raw(train_reps, ref_reps)
    repsim_eval = evaluate_method(repsim_sim, task_name, train_data, ref_data, "standard")

    # Standard TRAK-PCA at k=d
    train_grads = torch.load(os.path.join(CACHE_DIR, f"grads_train_{task_name}_full.pt"), weights_only=True)
    ref_grads = torch.load(os.path.join(CACHE_DIR, f"grads_ref_{task_name}_full.pt"), weights_only=True)
    D = train_grads.shape[1]
    d = train_reps.shape[1]  # 2048

    # Standard TRAK-PCA at k=d
    U, S, Vt = torch.linalg.svd(train_grads, full_matrices=False)
    k_d = min(d, min(n_train, D))
    pca_proj = Vt[:k_d].T
    standard_sim = compute_trak_raw(train_grads, ref_grads, k=k_d, proj_type="pca", proj_matrix=pca_proj)
    standard_eval = evaluate_method(standard_sim, task_name, train_data, ref_data, "standard")

    del U, S, Vt; gc.collect()

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "task_name": task_name, "n_train": n_train, "d": d, "D": D,
        "repsim": repsim_eval,
        "standard_trak_pca": standard_eval,
        "factors": {},
    }

    repsim_key = repsim_eval["rank_based"]["R_at_50"]
    standard_key = standard_eval["rank_based"]["R_at_50"]
    total_gap = repsim_key - standard_key
    print(f"  RepSim R@50={repsim_key:.4f}, TRAK-PCA R@50={standard_key:.4f}, Gap={total_gap:.4f}")

    # Factor (a): Last-layer-only gradients
    print(f"\n  Factor (a): Last-layer-only TRAK-PCA...")
    # Need to compute gradients from just the last transformer layer
    last_layer_params = []
    last_layer_names = []
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        # Pythia-1B has 16 layers (0-15), last layer is 15
        if "layers.15." in name and ("attention" in name or "mlp" in name):
            p.requires_grad_(True)
            last_layer_params.append(p)
            last_layer_names.append(name)

    D_last = sum(p.numel() for p in last_layer_params)
    print(f"  Last-layer params: D_last={D_last/1e6:.2f}M from {len(last_layer_names)} params")

    cache_ll_train = os.path.join(CACHE_DIR, f"grads_lastlayer_train_{task_name}_full.pt")
    cache_ll_ref = os.path.join(CACHE_DIR, f"grads_lastlayer_ref_{task_name}_full.pt")

    if os.path.exists(cache_ll_train):
        ll_train_grads = torch.load(cache_ll_train, weights_only=True)
    else:
        ll_train_grads = compute_raw_gradients(model, tok, train_texts, last_layer_params,
                                                desc="lastlayer/train")
        torch.save(ll_train_grads, cache_ll_train)

    if os.path.exists(cache_ll_ref):
        ll_ref_grads = torch.load(cache_ll_ref, weights_only=True)
    else:
        ll_ref_grads = compute_raw_gradients(model, tok, ref_texts, last_layer_params,
                                              desc="lastlayer/ref")
        torch.save(ll_ref_grads, cache_ll_ref)

    restore_grad_flags(model)

    # Last-layer TRAK-PCA at k=d
    U_ll, S_ll, Vt_ll = torch.linalg.svd(ll_train_grads, full_matrices=False)
    k_ll = min(d, min(n_train, D_last))
    ll_proj = Vt_ll[:k_ll].T
    ll_sim = compute_trak_raw(ll_train_grads, ll_ref_grads, k=k_ll, proj_type="pca", proj_matrix=ll_proj)
    ll_eval = evaluate_method(ll_sim, task_name, train_data, ref_data, "standard")
    ll_key = ll_eval["rank_based"]["R_at_50"]
    results["factors"]["a_last_layer"] = {"eval": ll_eval, "D_last": D_last, "gap_reduction": ll_key - standard_key}
    print(f"    Last-layer TRAK-PCA: R@50={ll_key:.4f}, reduction={ll_key - standard_key:.4f}pp")

    del U_ll, S_ll, Vt_ll; gc.collect()

    # Factor (b): Cosine-normalized TRAK-PCA
    print(f"\n  Factor (b): Cosine-normalized TRAK-PCA...")
    # Normalize gradients before projection
    train_grads_norm = F.normalize(train_grads, dim=-1)
    ref_grads_norm = F.normalize(ref_grads, dim=-1)
    U_n, S_n, Vt_n = torch.linalg.svd(train_grads_norm, full_matrices=False)
    k_n = min(d, min(n_train, D))
    norm_proj = Vt_n[:k_n].T
    norm_sim = compute_trak_raw(train_grads_norm, ref_grads_norm, k=k_n, proj_type="pca", proj_matrix=norm_proj)
    norm_eval = evaluate_method(norm_sim, task_name, train_data, ref_data, "standard")
    norm_key = norm_eval["rank_based"]["R_at_50"]
    results["factors"]["b_cosine_norm"] = {"eval": norm_eval, "gap_reduction": norm_key - standard_key}
    print(f"    Cosine-norm TRAK-PCA: R@50={norm_key:.4f}, reduction={norm_key - standard_key:.4f}pp")

    del U_n, S_n, Vt_n, train_grads_norm, ref_grads_norm; gc.collect()

    # Factor (c): Combined last-layer + cosine-norm
    print(f"\n  Factor (c): Combined last-layer + cosine-norm...")
    ll_train_norm = F.normalize(ll_train_grads, dim=-1)
    ll_ref_norm = F.normalize(ll_ref_grads, dim=-1)
    U_c, S_c, Vt_c = torch.linalg.svd(ll_train_norm, full_matrices=False)
    k_c = min(d, min(n_train, D_last))
    comb_proj = Vt_c[:k_c].T
    comb_sim = compute_trak_raw(ll_train_norm, ll_ref_norm, k=k_c, proj_type="pca", proj_matrix=comb_proj)
    comb_eval = evaluate_method(comb_sim, task_name, train_data, ref_data, "standard")
    comb_key = comb_eval["rank_based"]["R_at_50"]
    results["factors"]["c_combined"] = {"eval": comb_eval, "gap_reduction": comb_key - standard_key}
    print(f"    Combined TRAK-PCA: R@50={comb_key:.4f}, reduction={comb_key - standard_key:.4f}pp")

    del U_c, S_c, Vt_c, ll_train_norm, ll_ref_norm, ll_train_grads, ll_ref_grads
    gc.collect(); torch.cuda.empty_cache()

    # Residual
    residual = repsim_key - comb_key
    results["factors"]["d_residual"] = {
        "gap_pp": residual,
        "interpretation": "Nonlinear semantic features that representations capture but projected gradients do not"
    }
    results["summary"] = {
        "total_gap_pp": total_gap,
        "factor_a_pp": ll_key - standard_key,
        "factor_b_pp": norm_key - standard_key,
        "factor_c_pp": comb_key - standard_key,
        "factor_d_residual_pp": residual,
        "combined_explained_pct": ((comb_key - standard_key) / total_gap * 100) if total_gap > 0 else 0,
    }

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Gap decomposition completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P4: Layer Sweep
# ═══════════════════════════════════════════════════════════════════════
def run_p4_layer_sweep(tasks, model, tok):
    task_id = "p4_layer_sweep"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    # Pythia-1B has 16 transformer layers (0-15) + embedding layer
    # hidden_states indices: 0=embedding, 1-16=transformer layers
    layers = [0, 4, 8, 12, 16]  # Adjusted for Pythia-1B (16 layers = 0..15, hidden_states has 17 entries)
    # hidden_states[0] = embedding, hidden_states[1] = after layer 0, ..., hidden_states[16] = after layer 15
    layer_indices = [1, 5, 9, 13, 16]  # maps to after layer 0, 4, 8, 12, 15
    layer_names = ["layer_0", "layer_4", "layer_8", "layer_12", "layer_15"]

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "model": MODEL_NAME, "layers": layer_names,
        "results": {},
    }

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        task_results = {"n_train": n_train}

        for layer_idx, layer_name in zip(layer_indices, layer_names):
            cache_key = os.path.join(CACHE_DIR, f"reps_train_{task_name}_{layer_name}_full.pt")
            cache_ref = os.path.join(CACHE_DIR, f"reps_ref_{task_name}_{layer_name}_full.pt")

            if os.path.exists(cache_key):
                train_reps = torch.load(cache_key, weights_only=True)
            else:
                train_reps = extract_representations(model, tok, train_texts, layer_idx=layer_idx,
                                                     desc=f"{task_name}/{layer_name}/train")
                torch.save(train_reps, cache_key)

            if os.path.exists(cache_ref):
                ref_reps = torch.load(cache_ref, weights_only=True)
            else:
                ref_reps = extract_representations(model, tok, ref_texts, layer_idx=layer_idx,
                                                   desc=f"{task_name}/{layer_name}/ref")
                torch.save(ref_reps, cache_ref)

            sim = compute_repsim_raw(train_reps, ref_reps)
            ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
            task_results[layer_name] = ev

            if task_name == "toxicity":
                key = ev["rank_based"]["AUPRC"]
            else:
                key = ev["rank_based"]["R_at_50"]
            print(f"  {task_name}/{layer_name}: key={key:.4f}")

            del train_reps, ref_reps; gc.collect(); torch.cuda.empty_cache()

        results["results"][task_name] = task_results

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Layer sweep completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P4: Cosine vs Euclidean
# ═══════════════════════════════════════════════════════════════════════
def run_p4_cosine_vs_euclidean(tasks):
    task_id = "p4_cosine_vs_euclidean"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "model": MODEL_NAME,
        "similarity_functions": ["cosine", "euclidean", "dot_product"],
        "results": {},
    }

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)

        task_results = {"n_train": len(train_data)}

        # Cosine
        sim_cos = compute_repsim_raw(train_reps, ref_reps)
        ev_cos = evaluate_method(sim_cos, task_name, train_data, ref_data, "standard")
        task_results["cosine"] = ev_cos

        # Euclidean (negative distance)
        sim_euc = compute_repsim_euclidean(train_reps, ref_reps)
        ev_euc = evaluate_method(sim_euc, task_name, train_data, ref_data, "standard")
        task_results["euclidean"] = ev_euc

        # Dot product
        sim_dot = compute_repsim_dot(train_reps, ref_reps)
        ev_dot = evaluate_method(sim_dot, task_name, train_data, ref_data, "standard")
        task_results["dot_product"] = ev_dot

        # Rank correlations between similarity functions
        flat_cos = sim_cos.flatten()
        flat_euc = sim_euc.flatten()
        flat_dot = sim_dot.flatten()

        tau_ce, _ = kendalltau(flat_cos[:10000], flat_euc[:10000])  # Subsample for speed
        tau_cd, _ = kendalltau(flat_cos[:10000], flat_dot[:10000])

        task_results["rank_correlations"] = {
            "cosine_vs_euclidean_tau": float(tau_ce) if not np.isnan(tau_ce) else 0.0,
            "cosine_vs_dot_tau": float(tau_cd) if not np.isnan(tau_cd) else 0.0,
        }

        # Norm analysis
        norms = torch.norm(train_reps, dim=-1).numpy()
        task_results["norm_analysis"] = {
            "mean": float(norms.mean()), "std": float(norms.std()),
            "cv": float(norms.std() / norms.mean()) if norms.mean() > 0 else 0.0,
        }

        results["results"][task_name] = task_results

        if task_name == "toxicity":
            key_cos = ev_cos["rank_based"]["AUPRC"]
            key_euc = ev_euc["rank_based"]["AUPRC"]
            key_dot = ev_dot["rank_based"]["AUPRC"]
        else:
            key_cos = ev_cos["rank_based"]["R_at_50"]
            key_euc = ev_euc["rank_based"]["R_at_50"]
            key_dot = ev_dot["rank_based"]["R_at_50"]
        print(f"  {task_name}: cos={key_cos:.4f}, euc={key_euc:.4f}, dot={key_dot:.4f}")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Cosine vs Euclidean completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P5: PCA-Whitened Attribution
# ═══════════════════════════════════════════════════════════════════════
def run_p5_pca_whitened_attribution(tasks):
    task_id = "p5_pca_whitened_attribution"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    k_values = [16, 32, 64, 128, 256, 512]
    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "model": MODEL_NAME, "k_values": k_values,
        "results": {},
    }

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)

        d = train_reps.shape[1]

        # Standard RepSim baseline
        std_sim = compute_repsim_raw(train_reps, ref_reps)
        std_eval = evaluate_method(std_sim, task_name, train_data, ref_data, "standard")

        # PCA decomposition
        mean_reps = train_reps.mean(dim=0)
        centered = train_reps - mean_reps
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        task_results = {"n_train": n_train, "d": d, "standard_repsim": std_eval, "pca_whitened": {}}

        for k in k_values:
            k_actual = min(k, min(n_train - 1, d))
            pca_proj = Vt[:k_actual]  # [k, d]

            train_pca = centered @ pca_proj.T  # [n_train, k]
            ref_centered = ref_reps - mean_reps
            ref_pca = ref_centered @ pca_proj.T  # [n_ref, k]

            # Compute covariance in PCA space
            cov_pca = (train_pca.T @ train_pca) / (n_train - 1)

            # Method 1: Ledoit-Wolf shrinkage whitening
            try:
                lw = LedoitWolf()
                lw.fit(train_pca.numpy())
                cov_lw = torch.tensor(lw.covariance_, dtype=torch.float32)
                # Compute M = cov^{-1/2}
                eigvals, eigvecs = torch.linalg.eigh(cov_lw)
                eigvals = eigvals.clamp(min=1e-6)
                inv_sqrt = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T

                train_white = train_pca @ inv_sqrt
                ref_white = ref_pca @ inv_sqrt

                # Cosine in whitened space
                tn = F.normalize(train_white, dim=-1)
                rn = F.normalize(ref_white, dim=-1)
                sim_lw = (tn @ rn.T).numpy()
                ev_lw = evaluate_method(sim_lw, task_name, train_data, ref_data, "standard")
                lw_result = ev_lw
            except Exception as e:
                lw_result = {"error": str(e)}

            # Method 2: Ridge-regularized whitening with CV
            try:
                lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
                best_lambda = lambdas[0]
                best_cv_score = -np.inf

                # Simple 5-fold CV on a proxy metric
                n_fold = 5
                fold_size = n_train // n_fold
                for lam in lambdas:
                    fold_scores = []
                    for fold_i in range(n_fold):
                        val_start = fold_i * fold_size
                        val_end = min(val_start + fold_size, n_train)
                        val_idx = list(range(val_start, val_end))
                        tr_idx = list(range(0, val_start)) + list(range(val_end, n_train))

                        tr_pca = train_pca[tr_idx]
                        cov_fold = (tr_pca.T @ tr_pca) / (len(tr_idx) - 1) + lam * torch.eye(k_actual)
                        try:
                            ev_f, evec_f = torch.linalg.eigh(cov_fold)
                            ev_f = ev_f.clamp(min=1e-6)
                            inv_f = evec_f @ torch.diag(1.0 / ev_f.sqrt()) @ evec_f.T
                            val_pca = train_pca[val_idx]
                            # Score: average cosine similarity to nearest neighbor
                            val_w = val_pca @ inv_f
                            tr_w = tr_pca @ inv_f
                            val_n = F.normalize(val_w, dim=-1)
                            tr_n = F.normalize(tr_w, dim=-1)
                            sim_cv = (val_n @ tr_n.T).numpy()
                            fold_scores.append(float(np.mean(np.max(sim_cv, axis=1))))
                        except:
                            fold_scores.append(-np.inf)

                    avg = np.mean(fold_scores)
                    if avg > best_cv_score:
                        best_cv_score = avg
                        best_lambda = lam

                # Apply best lambda
                cov_ridge = cov_pca + best_lambda * torch.eye(k_actual)
                ev_r, evec_r = torch.linalg.eigh(cov_ridge)
                ev_r = ev_r.clamp(min=1e-6)
                inv_ridge = evec_r @ torch.diag(1.0 / ev_r.sqrt()) @ evec_r.T

                train_ridge = train_pca @ inv_ridge
                ref_ridge = ref_pca @ inv_ridge

                tn_r = F.normalize(train_ridge, dim=-1)
                rn_r = F.normalize(ref_ridge, dim=-1)
                sim_ridge = (tn_r @ rn_r.T).numpy()
                ev_ridge = evaluate_method(sim_ridge, task_name, train_data, ref_data, "standard")
                ridge_result = {"best_lambda": best_lambda, **ev_ridge}
            except Exception as e:
                ridge_result = {"error": str(e)}

            # SNR analysis
            try:
                diag_signal = torch.diag(cov_pca)
                noise_var = diag_signal.mean()
                snr_per_dim = diag_signal / noise_var
                snr_out = float(snr_per_dim.mean())
            except:
                snr_out = 0.0

            n_over_k = n_train / k_actual
            task_results["pca_whitened"][str(k)] = {
                "k_actual": k_actual, "N_over_k": n_over_k,
                "LW_whitened": lw_result,
                "ridge_cv": ridge_result,
                "SNR_mean": snr_out,
            }

            if task_name == "toxicity":
                lw_key = lw_result.get("rank_based", {}).get("AUPRC", "N/A") if isinstance(lw_result, dict) and "rank_based" in lw_result else "err"
                std_key = std_eval["rank_based"]["AUPRC"]
            else:
                lw_key = lw_result.get("rank_based", {}).get("R_at_50", "N/A") if isinstance(lw_result, dict) and "rank_based" in lw_result else "err"
                std_key = std_eval["rank_based"]["R_at_50"]
            print(f"  {task_name} k={k} (N/k={n_over_k:.1f}): LW={lw_key}, std={std_key}")

        results["results"][task_name] = task_results
        del U, S, Vt; gc.collect()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"PCA whitened attribution completed in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK: Final Analysis
# ═══════════════════════════════════════════════════════════════════════
def run_final_analysis():
    task_id = "final_analysis"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE (CPU)")
    print("=" * 60)
    start_time = time.time()

    # Load all results
    result_files = {
        "p1_continuous": "p1_fm2_continuous_metrics.json",
        "p1_contamination": "p1_fm2_contamination_injection.json",
        "p1_interaction": "p1_fm2_interaction_analysis.json",
        "p2_eigenspectrum": "p2_eigenspectrum_fullscale.json",
        "p2_trak_sweep": "p2_trak_dim_sweep_fullscale.json",
        "p2_repsim_sweep": "p2_repsim_dim_sweep_fullscale.json",
        "p3_retrieval": "p3_retrieval_baselines.json",
        "p4_gap": "p4_gap_decomposition.json",
        "p4_layer": "p4_layer_sweep.json",
        "p4_similarity": "p4_cosine_vs_euclidean.json",
        "p5_whitened": "p5_pca_whitened_attribution.json",
    }

    all_results = {}
    for key, fname in result_files.items():
        fpath = os.path.join(FULL_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                all_results[key] = json.load(f)
            print(f"  Loaded: {fname}")
        else:
            print(f"  MISSING: {fname}")

    # Build final analysis
    analysis = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "timestamp": datetime.now().isoformat(),
    }

    # Decision gates
    gates = {}

    # Gate 1: FM2
    p1 = all_results.get("p1_continuous", {})
    h2_passes = False
    max_tau_gain = 0.0
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tr = p1.get("results", {}).get(task_name, {})
        for method in ["TRAK", "RepSim"]:
            mr = tr.get(method, {})
            std_tau = mr.get("standard", {}).get("continuous", {}).get("kendall_tau", 0.0)
            con_tau = mr.get("contrastive", {}).get("continuous", {}).get("kendall_tau", 0.0)
            gain = con_tau - std_tau
            max_tau_gain = max(max_tau_gain, gain)
    h2_passes = max_tau_gain >= 0.05
    gates["gate_1_fm2"] = {"h2_passes": h2_passes, "max_tau_gain": max_tau_gain}

    # Gate 2: Retrieval
    p3 = all_results.get("p3_retrieval", {})
    matching_tasks = 0
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tr = p3.get("results", {}).get(task_name, {})
        # Compare retrieval vs RepSim from p1
        p1_tr = p1.get("results", {}).get(task_name, {})
        repsim_eval = p1_tr.get("RepSim", {}).get("standard", {})

        if task_name == "toxicity":
            repsim_key = repsim_eval.get("rank_based", {}).get("AUPRC", 0)
            for rm in ["Contriever", "GTR-T5"]:
                rm_key = tr.get(rm, {}).get("rank_based", {}).get("AUPRC", 0)
                if abs(rm_key - repsim_key) < 0.03:
                    matching_tasks += 1
                    break
        else:
            repsim_key = repsim_eval.get("rank_based", {}).get("R_at_50", 0)
            for rm in ["Contriever", "GTR-T5"]:
                rm_key = tr.get(rm, {}).get("rank_based", {}).get("R_at_50", 0)
                if abs(rm_key - repsim_key) < 0.03:
                    matching_tasks += 1
                    break
    gates["gate_2_retrieval"] = {"matching_tasks": matching_tasks, "triggers": matching_tasks >= 2}

    # Gate 3: Gap decomposition
    p4_gap = all_results.get("p4_gap", {})
    gap_summary = p4_gap.get("summary", {})
    combined_closes = gap_summary.get("factor_c_pp", 0)
    total_gap = gap_summary.get("total_gap_pp", 1)
    repsim_r50 = p4_gap.get("repsim", {}).get("rank_based", {}).get("R_at_50", 0)
    combined_r50 = p4_gap.get("factors", {}).get("c_combined", {}).get("eval", {}).get("rank_based", {}).get("R_at_50", 0)
    within_10pp = (repsim_r50 - combined_r50) <= 0.10
    gates["gate_3_gap"] = {"within_10pp": within_10pp, "gap_pp": repsim_r50 - combined_r50}

    # Gate 4: PCA whitening
    p5 = all_results.get("p5_whitened", {})
    whitening_passes = False
    best_gain = 0.0
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tr = p5.get("results", {}).get(task_name, {})
        std_eval = tr.get("standard_repsim", {})
        if task_name == "toxicity":
            std_key = std_eval.get("rank_based", {}).get("AUPRC", 0)
        else:
            std_key = std_eval.get("rank_based", {}).get("R_at_50", 0)

        for k_str, kv in tr.get("pca_whitened", {}).items():
            lw = kv.get("LW_whitened", {})
            if isinstance(lw, dict) and "rank_based" in lw:
                if task_name == "toxicity":
                    w_key = lw["rank_based"].get("AUPRC", 0)
                else:
                    w_key = lw["rank_based"].get("R_at_50", 0)
                gain = w_key - std_key
                if gain > best_gain:
                    best_gain = gain
    whitening_passes = best_gain >= 0.03
    gates["gate_4_whitening"] = {"passes": whitening_passes, "best_gain_pp": best_gain}

    analysis["decision_gates"] = gates

    # Method tournament summary
    tournament = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tr = p1.get("results", {}).get(task_name, {})
        task_tourn = {}
        for method in ["RepSim", "TRAK", "LoGra", "DDA", "RawDotIF", "DiagIF", "kNN", "BM25"]:
            mr = tr.get(method, {}).get("standard", {})
            if task_name == "toxicity":
                task_tourn[method] = mr.get("rank_based", {}).get("AUPRC", 0)
            else:
                task_tourn[method] = mr.get("rank_based", {}).get("R_at_50", 0)
        # Add retrieval baselines
        p3_tr = p3.get("results", {}).get(task_name, {})
        for rm in ["Contriever", "GTR-T5"]:
            rm_eval = p3_tr.get(rm, {})
            if task_name == "toxicity":
                task_tourn[rm] = rm_eval.get("rank_based", {}).get("AUPRC", 0)
            else:
                task_tourn[rm] = rm_eval.get("rank_based", {}).get("R_at_50", 0)
        tournament[task_name] = dict(sorted(task_tourn.items(), key=lambda x: -x[1]))

    analysis["method_tournament"] = tournament
    analysis["all_results_summary"] = {k: v.get("mode", "?") for k, v in all_results.items()}

    elapsed = time.time() - start_time
    analysis["elapsed_sec"] = elapsed

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Write summary markdown
    summary_path = os.path.join(FULL_DIR, "final_analysis_summary.md")
    with open(summary_path, "w") as f:
        f.write("# CRA Full-Scale Experiment Results Summary\n\n")
        f.write(f"## Decision Gates\n\n")
        for gname, gdata in gates.items():
            f.write(f"### {gname}\n")
            for k, v in gdata.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write(f"## Method Tournament\n\n")
        for task_name, tourn in tournament.items():
            f.write(f"### {task_name}\n")
            for method, score in tourn.items():
                f.write(f"- {method}: {score:.4f}\n")
            f.write("\n")

    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Final analysis completed in {elapsed:.0f}s")
    return analysis

# ═══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  CRA FULL-SCALE EXPERIMENT RUNNER")
    print(f"  GPU: {DEVICE} (CUDA_VISIBLE_DEVICES should be set to 3)")
    print(f"  Seed: {SEED}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Start time: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_mem = torch.cuda.mem_get_info()[0] / 1024**2
        total_mem = torch.cuda.mem_get_info()[1] / 1024**2
        print(f"GPU: {gpu_name}, Free: {free_mem:.0f}MiB / {total_mem:.0f}MiB")
    else:
        print("ERROR: No GPU available!")
        sys.exit(1)

    overall_start = time.time()

    # Load datasets at full scale
    tasks = load_all_tasks_full()

    # Load model
    model, tok = load_model_pythia1b()

    # ── Phase 1: P1 FM2 Continuous Metrics (biggest task - extracts all reps+grads) ──
    p1_results = run_p1_fm2_continuous_metrics(tasks, model, tok)

    # ── Phase 2: P1 FM2 Contamination Injection ──
    run_p1_fm2_contamination_injection(tasks, model, tok)

    # ── Phase 3: P1 FM2 Interaction Analysis ──
    run_p1_fm2_interaction_analysis(p1_results)

    # ── Phase 4: P2 TRAK Dimension Sweep ──
    run_p2_trak_dim_sweep(tasks, model, tok)

    # ── Phase 5: P2 RepSim Dimension Sweep ──
    run_p2_repsim_dim_sweep(tasks)

    # ── Phase 6: P3 Retrieval Baselines ──
    # Unload main model temporarily to fit retrieval models
    unload_model(model)
    run_p3_retrieval_baselines(tasks, None, None)
    # Reload main model for remaining tasks
    model, tok = load_model_pythia1b()

    # ── Phase 7: P4 Gap Decomposition ──
    run_p4_gap_decomposition(tasks, model, tok)

    # ── Phase 8: P4 Layer Sweep ──
    run_p4_layer_sweep(tasks, model, tok)

    # ── Phase 9: P4 Cosine vs Euclidean ──
    run_p4_cosine_vs_euclidean(tasks)

    # ── Phase 10: P5 PCA Whitened Attribution ──
    unload_model(model)
    run_p5_pca_whitened_attribution(tasks)

    # ── Phase 11: Final Analysis ──
    run_final_analysis()

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  ALL TASKS COMPLETE. Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
