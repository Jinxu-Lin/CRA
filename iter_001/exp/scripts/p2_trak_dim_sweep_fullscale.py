#!/usr/bin/env python3
"""
P2: TRAK Dimension Sweep at Full Scale (H5-revised) -- PILOT MODE
==================================================================
Run TRAK on Pythia-1B with counterfact task.
Random projection at k in {32, 64, 128, 256, 512, 1024, 2048, 4096}.
PCA projection at k in {32, 64, 128, 256, 512, 1024, 2048}.
Evaluate with BOTH rank-based (R@50, MRR) AND continuous (Kendall tau) metrics.

PILOT: N=100 training samples, seed=42.
Pass criteria: Saturation knee visible at k in [d/16, d/4];
TRAK-PCA outperforms TRAK-random at same k on at least 4/7 k values.
"""

import os, sys, json, time, gc, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p2_trak_dim_sweep_fullscale"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
BOOTSTRAP_B = 1000
MAX_LEN = 512
DEVICE = "cuda:0"

# Dimension sweep configurations (expanded from pilot)
RANDOM_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
PCA_DIMS = [32, 64, 128, 256, 512, 1024, 2048]

# Target layers: last transformer block (layer 15 for Pythia-1B with 16 layers)
TARGET_LAYER_PATTERNS = [
    "layers.15.attention.dense.weight",
    "layers.15.mlp.dense_4h_to_h.weight",
]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

for d in [FULL_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Progress / lifecycle ────────────────────────────────────────────────
def _safe_write(path, content):
    """Write to file, silently skip on disk quota errors."""
    try:
        Path(path).write_text(content)
    except OSError as e:
        print(f"[Warn] Cannot write {path}: {e}")

pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
_safe_write(pid_file, str(os.getpid()))


def report_progress(stage, detail="", pct=0.0, metric=None):
    _safe_write(
        Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json",
        json.dumps({
            "task_id": TASK_ID, "epoch": 0, "total_epochs": 1,
            "step": 0, "total_steps": 0, "loss": None,
            "metric": metric or {}, "stage": stage, "detail": detail,
            "pct": pct, "updated_at": datetime.now().isoformat(),
        })
    )


def mark_done(status="success", summary=""):
    pid_f = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pid_f.exists():
        try:
            pid_f.unlink()
        except OSError:
            pass
    fp = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final = {}
    if fp.exists():
        try:
            final = json.loads(fp.read_text())
        except Exception:
            pass
    _safe_write(
        Path(RESULTS_DIR) / f"{TASK_ID}_DONE",
        json.dumps({
            "task_id": TASK_ID, "status": status, "summary": summary,
            "final_progress": final,
            "timestamp": datetime.now().isoformat(),
        })
    )


# ── Evaluation helpers ──────────────────────────────────────────────────
def compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50):
    """Recall@K and MRR for counterfact."""
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
    """Kendall tau and Spearman rho on raw attribution scores vs binary relevance."""
    taus, rhos = [], []
    for scores, fi in zip(scores_per_ref, fact_indices_per_ref):
        if not fi:
            continue
        binary_rel = np.zeros(n_train)
        binary_rel[fi] = 1.0
        s = np.array(scores)
        # Skip if all scores are identical (can happen with degenerate projections)
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


def evaluate_counterfact(sim_matrix, pilot_data, ref_data, n_train):
    """Full evaluation: rank-based + continuous metrics with bootstrap CI."""
    n_ref = len(ref_data)

    scores_per_ref = []
    fact_indices_per_ref = []
    for j in range(n_ref):
        ref_sample = ref_data[j]
        scores_j = sim_matrix[:, j].tolist()
        fi = [
            i for i in range(n_train)
            if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
            and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
        ]
        scores_per_ref.append(scores_j)
        fact_indices_per_ref.append(fi)

    recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)
    tau, rho = compute_continuous_metrics(scores_per_ref, fact_indices_per_ref, n_train)

    # Bootstrap CI
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


# ── Data loading ────────────────────────────────────────────────────────
def load_counterfact_data():
    """Load counterfact dataset."""
    report_progress("loading_data", "Loading counterfact dataset", 0.05)
    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    fmt = lambda s: s["prompt"] + " " + s["response"]
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")
    return cf["train"], cf["ref"], fmt


def create_pilot_subset(train_data, n_pilot=PILOT_N_TRAIN):
    """Create pilot subset."""
    rng = np.random.RandomState(SEED)
    n_total = len(train_data)
    pilot_idx = sorted(rng.choice(n_total, min(n_pilot, n_total), replace=False).tolist())
    print(f"[pilot] Using {len(pilot_idx)} samples from {n_total} total")
    return train_data.select(pilot_idx), pilot_idx


# ── Model ───────────────────────────────────────────────────────────────
def load_model():
    report_progress("loading_model", "Loading Pythia-1B", 0.10)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    # Set requires_grad only for target layers
    target_params = []
    target_names = []
    for name, p in model.named_parameters():
        if any(pat in name for pat in TARGET_LAYER_PATTERNS):
            p.requires_grad_(True)
            target_params.append(p)
            target_names.append(name)
        else:
            p.requires_grad_(False)

    D = sum(p.numel() for p in target_params)
    print(f"Model loaded: hidden_dim={model.config.hidden_size}")
    print(f"Target params: {target_names}, D={D/1e6:.2f}M")
    return model, tok, target_params, target_names, D


# ── Gradient extraction (memory-efficient, one sample at a time) ────────
def extract_gradients(model, tok, target_params, texts, D, desc=""):
    """Extract gradients one sample at a time, store on CPU."""
    n = len(texts)
    all_grads = torch.zeros(n, D, dtype=torch.float32)  # CPU tensor

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

        # Collect gradients to CPU immediately
        offset = 0
        for p in target_params:
            g = p.grad.detach().flatten().float().cpu()
            all_grads[idx, offset:offset + g.shape[0]] = g
            offset += g.shape[0]

        model.zero_grad(set_to_none=True)

        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"  [{desc}] {idx+1}/{n}")

    print(f"[Grads/{desc}] Extracted: shape={all_grads.shape}")
    return all_grads


# ── CountSketch random projection ──────────────────────────────────────
def countsketch_project(grads, k, seed=SEED + 7777):
    """Apply CountSketch projection: D -> k. All on CPU."""
    D = grads.shape[1]
    rng_cs = np.random.RandomState(seed)
    cs_buckets = torch.from_numpy(rng_cs.randint(0, k, size=D).astype(np.int64))
    cs_signs = torch.from_numpy(rng_cs.choice([-1.0, 1.0], size=D).astype(np.float32))

    n = grads.shape[0]
    projected = torch.zeros(n, k, dtype=torch.float32)
    for i in range(n):
        projected[i].index_add_(0, cs_buckets, grads[i] * cs_signs)
    return projected


# ── PCA projection ──────────────────────────────────────────────────────
def pca_project(train_grads, ref_grads, k):
    """
    Project gradients onto top-k principal components of training gradient covariance.
    Uses Gram matrix approach (efficient when n << D).
    """
    n_train, D = train_grads.shape
    print(f"[PCA k={k}] Computing via Gram matrix ({n_train}x{n_train})...")

    # Center gradients
    mean_grad = train_grads.mean(dim=0, keepdim=True)
    centered_train = train_grads - mean_grad
    centered_ref = ref_grads - mean_grad

    # Gram matrix: n x n
    gram = centered_train @ centered_train.T
    eigenvalues, eigenvectors_gram = torch.linalg.eigh(gram)

    # Sort descending
    idx_sorted = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx_sorted]
    eigenvectors_gram = eigenvectors_gram[:, idx_sorted]

    # Take top-k (capped by n_train)
    k_actual = min(k, n_train)
    top_eigenvals = eigenvalues[:k_actual].clamp(min=1e-10)
    top_U = eigenvectors_gram[:, :k_actual]

    # Recover V = X^T U / sqrt(eigenvalues)
    V_topk = centered_train.T @ top_U / top_eigenvals.sqrt().unsqueeze(0)  # [D, k]
    V_topk = F.normalize(V_topk, dim=0)

    # Project
    train_proj = centered_train @ V_topk
    ref_proj = centered_ref @ V_topk

    # Explained variance
    total_var = eigenvalues.sum().item()
    explained_var = eigenvalues[:k_actual].sum().item() / max(total_var, 1e-10)

    print(f"[PCA k={k}] k_actual={k_actual}, explained_var={explained_var:.4f}")
    return train_proj, ref_proj, {
        "k_requested": k,
        "k_actual": int(k_actual),
        "explained_var_ratio": round(explained_var, 6),
        "top_eigenvalues": eigenvalues[:min(10, k_actual)].tolist(),
    }


# ── Similarity computation ─────────────────────────────────────────────
def compute_similarity_matrix(train_proj, ref_proj):
    """Cosine similarity matrix [n_train, n_ref]. CPU tensors."""
    train_norm = F.normalize(train_proj, dim=-1)
    ref_norm = F.normalize(ref_proj, dim=-1)
    sim = (train_norm @ ref_norm.T).numpy()
    return sim


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 70)
    print(f"P2: TRAK Dimension Sweep (H5-revised) -- PILOT MODE")
    print(f"Model: {MODEL_NAME}, N={PILOT_N_TRAIN}")
    print(f"Random dims: {RANDOM_DIMS}")
    print(f"PCA dims: {PCA_DIMS}")
    print(f"Metrics: Recall@50, MRR, Kendall tau, Spearman rho")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    # Load data
    train_data, ref_data, fmt_fn = load_counterfact_data()
    pilot_data, pilot_idx = create_pilot_subset(train_data)
    n_train = len(pilot_data)

    train_texts = [fmt_fn(pilot_data[i]) for i in range(n_train)]
    ref_texts = [fmt_fn(ref_data[i]) for i in range(len(ref_data))]

    # Load model
    model, tok, target_params, target_names, D = load_model()

    # ── Step 1: Extract raw gradients ────────────────────────────────────
    report_progress("gradient_extraction", "Extracting train gradients", 0.10)

    # Check cache
    cache_key = f"trak_dim_sweep_N{PILOT_N_TRAIN}_seed{SEED}"
    train_grad_cache = os.path.join(CACHE_DIR, f"{cache_key}_train_grads.pt")
    ref_grad_cache = os.path.join(CACHE_DIR, f"{cache_key}_ref_grads.pt")

    if os.path.exists(train_grad_cache) and os.path.exists(ref_grad_cache):
        print("[Cache] Loading cached gradients...")
        train_grads = torch.load(train_grad_cache, weights_only=True)
        ref_grads = torch.load(ref_grad_cache, weights_only=True)
        print(f"[Cache] Loaded: train={train_grads.shape}, ref={ref_grads.shape}")
    else:
        print(f"[Grads] Extracting train gradients ({n_train} samples)...")
        train_grads = extract_gradients(model, tok, target_params, train_texts, D, desc="train")

        report_progress("gradient_extraction", "Extracting ref gradients", 0.25)
        print(f"[Grads] Extracting ref gradients ({len(ref_texts)} samples)...")
        ref_grads = extract_gradients(model, tok, target_params, ref_texts, D, desc="ref")

        # Skip caching: disk quota too tight for large gradient tensors
        print(f"[Cache] Skipping gradient caching (disk quota constraint)")

    # Free model memory
    del model, tok, target_params
    gc.collect()
    torch.cuda.empty_cache()
    print("[Memory] Model freed; working with CPU gradients only")

    # ── Step 2: Load RepSim baseline for comparison ─────────────────────
    repsim_recall = None
    repsim_tau = None
    # Try full-scale p1 results first
    p1_path = os.path.join(FULL_DIR, "p1_fm2_continuous_metrics.json")
    if os.path.exists(p1_path):
        try:
            with open(p1_path) as f:
                p1_data = json.load(f)
            # Navigate the result structure to find RepSim counterfact metrics
            for task_key, task_data in p1_data.get("results_per_task", {}).items():
                if "counterfact" in task_key.lower():
                    methods = task_data.get("methods", {})
                    for method_key, method_data in methods.items():
                        if "repsim" in method_key.lower():
                            std = method_data.get("standard", {})
                            repsim_recall = std.get("Recall@50")
                            repsim_tau = std.get("kendall_tau")
                            break
        except Exception as e:
            print(f"[Warning] Could not load p1 results: {e}")

    # Fallback to pilot phase1 results
    if repsim_recall is None:
        repsim_path = os.path.join(RESULTS_DIR, "phase1", "repsim_standard.json")
        if os.path.exists(repsim_path):
            try:
                with open(repsim_path) as f:
                    repsim_data = json.load(f)
                repsim_recall = repsim_data["results_per_task"]["counterfact"]["metrics"]["Recall@50"]
            except Exception:
                pass

    if repsim_recall is not None:
        print(f"\n[Reference] RepSim Recall@50 = {repsim_recall:.4f}")
    if repsim_tau is not None:
        print(f"[Reference] RepSim Kendall tau = {repsim_tau:.4f}")

    # ── Step 3: Random projection dimension sweep ────────────────────────
    report_progress("random_sweep", "Running CountSketch dimension sweep", 0.35)
    random_results = {}

    for dim_idx, k in enumerate(RANDOM_DIMS):
        t_k = time.time()
        print(f"\n{'─'*50}")
        print(f"[Random] k={k} ({dim_idx+1}/{len(RANDOM_DIMS)})")

        train_proj = countsketch_project(train_grads, k)
        ref_proj = countsketch_project(ref_grads, k)

        sim = compute_similarity_matrix(train_proj, ref_proj)
        metrics = evaluate_counterfact(sim, pilot_data, ref_data, n_train)

        elapsed_k = time.time() - t_k
        metrics["runtime_sec"] = round(elapsed_k, 2)
        metrics["projection_type"] = "CountSketch"
        metrics["projection_dim"] = k
        metrics["grad_dim_D"] = D
        metrics["ratio_k_over_d"] = round(k / 2048, 4)

        random_results[str(k)] = metrics
        print(f"[Random] k={k}: R@50={metrics['Recall@50']:.4f}, MRR={metrics['MRR']:.4f}, "
              f"tau={metrics['kendall_tau']:.4f}, time={elapsed_k:.1f}s")

        pct = 0.35 + 0.30 * (dim_idx + 1) / len(RANDOM_DIMS)
        report_progress("random_sweep", f"k={k} done", pct,
                       metric={"k": k, "Recall@50": metrics["Recall@50"],
                               "kendall_tau": metrics["kendall_tau"]})

    # ── Step 4: PCA projection dimension sweep ───────────────────────────
    report_progress("pca_sweep", "Running PCA dimension sweep", 0.65)
    pca_results = {}

    for dim_idx, k in enumerate(PCA_DIMS):
        t_k = time.time()
        print(f"\n{'─'*50}")
        print(f"[PCA] k={k} ({dim_idx+1}/{len(PCA_DIMS)})")

        train_proj, ref_proj, pca_info = pca_project(train_grads, ref_grads, k)

        sim = compute_similarity_matrix(train_proj, ref_proj)
        metrics = evaluate_counterfact(sim, pilot_data, ref_data, n_train)

        elapsed_k = time.time() - t_k
        metrics["runtime_sec"] = round(elapsed_k, 2)
        metrics["projection_type"] = "PCA"
        metrics["projection_dim"] = k
        metrics["grad_dim_D"] = D
        metrics["ratio_k_over_d"] = round(k / 2048, 4)
        metrics["pca_info"] = pca_info

        pca_results[str(k)] = metrics
        print(f"[PCA] k={k}: R@50={metrics['Recall@50']:.4f}, MRR={metrics['MRR']:.4f}, "
              f"tau={metrics['kendall_tau']:.4f}, explained_var={pca_info['explained_var_ratio']:.4f}, "
              f"time={elapsed_k:.1f}s")

        pct = 0.65 + 0.25 * (dim_idx + 1) / len(PCA_DIMS)
        report_progress("pca_sweep", f"PCA k={k} done", pct,
                       metric={"k": k, "Recall@50": metrics["Recall@50"],
                               "kendall_tau": metrics["kendall_tau"], "type": "PCA"})

    # ── Step 5: Analysis ─────────────────────────────────────────────────
    report_progress("analysis", "Analyzing results", 0.92)

    # Random projection analysis
    random_recalls = {int(k): v["Recall@50"] for k, v in random_results.items()}
    random_taus = {int(k): v["kendall_tau"] for k, v in random_results.items()}
    max_random_recall = max(random_recalls.values())

    # Find saturation knee (90% of max)
    saturation_90_k = None
    for k in sorted(random_recalls.keys()):
        if random_recalls[k] >= 0.9 * max_random_recall:
            saturation_90_k = k
            break

    # Monotonicity check
    sorted_ks = sorted(random_recalls.keys())
    monotonic_violations = []
    for i in range(1, len(sorted_ks)):
        prev_k, curr_k = sorted_ks[i - 1], sorted_ks[i]
        if random_recalls[curr_k] < random_recalls[prev_k] - 0.02:
            monotonic_violations.append({
                "from_k": prev_k, "to_k": curr_k,
                "from_recall": random_recalls[prev_k],
                "to_recall": random_recalls[curr_k],
            })

    # PCA analysis
    pca_recalls = {int(k): v["Recall@50"] for k, v in pca_results.items()}
    pca_taus = {int(k): v["kendall_tau"] for k, v in pca_results.items()}

    # PCA vs Random comparison
    pca_vs_random = {}
    pca_wins = 0
    total_comparisons = 0
    for k_str in pca_results:
        k = int(k_str)
        if k_str in random_results:
            pca_r = pca_results[k_str]["Recall@50"]
            rand_r = random_results[k_str]["Recall@50"]
            pca_t = pca_results[k_str]["kendall_tau"]
            rand_t = random_results[k_str]["kendall_tau"]
            pca_vs_random[k_str] = {
                "pca_recall": round(pca_r, 6),
                "random_recall": round(rand_r, 6),
                "pca_advantage_pp": round((pca_r - rand_r) * 100, 2),
                "pca_tau": round(pca_t, 6),
                "random_tau": round(rand_t, 6),
                "pca_tau_advantage": round(pca_t - rand_t, 6),
            }
            if pca_r > rand_r:
                pca_wins += 1
            total_comparisons += 1

    # Smoking-gun test: TRAK-PCA at k=d vs RepSim
    pca_at_d = pca_results.get("2048", {})
    pca_at_d_recall = pca_at_d.get("Recall@50")
    pca_at_d_tau = pca_at_d.get("kendall_tau")
    pca_repsim_gap_recall = None
    pca_repsim_gap_tau = None

    if pca_at_d_recall is not None and repsim_recall is not None:
        pca_repsim_gap_recall = round(repsim_recall - pca_at_d_recall, 4)
    if pca_at_d_tau is not None and repsim_tau is not None:
        pca_repsim_gap_tau = round(repsim_tau - pca_at_d_tau, 4)

    smoking_gun = {
        "pca_at_d_recall": pca_at_d_recall,
        "pca_at_d_tau": pca_at_d_tau,
        "repsim_recall": repsim_recall,
        "repsim_tau": repsim_tau,
        "gap_recall_pp": round(pca_repsim_gap_recall * 100, 2) if pca_repsim_gap_recall is not None else None,
        "gap_tau": pca_repsim_gap_tau,
        "within_15pp_recall": abs(pca_repsim_gap_recall) <= 0.15 if pca_repsim_gap_recall is not None else None,
    }

    if pca_repsim_gap_recall is not None:
        if abs(pca_repsim_gap_recall) <= 0.05:
            smoking_gun["interpretation"] = (
                f"SMOKING GUN CONFIRMED: TRAK-PCA(k=d) matches RepSim within 5pp. "
                f"FM1 is the primary mechanism."
            )
        elif abs(pca_repsim_gap_recall) <= 0.15:
            smoking_gun["interpretation"] = (
                f"PARTIAL: TRAK-PCA(k=d) within 15pp of RepSim ({pca_repsim_gap_recall*100:.1f}pp gap). "
                f"FM1 explains significant portion but other factors remain."
            )
        else:
            smoking_gun["interpretation"] = (
                f"GAP PERSISTS: TRAK-PCA(k=d) differs from RepSim by {pca_repsim_gap_recall*100:.1f}pp. "
                f"FM1 alone does not explain the performance gap."
            )

    # H5 assessment
    h5_pass_criteria = {
        "saturation_knee_in_range": (saturation_90_k is not None and
                                      2048 // 16 <= saturation_90_k <= 2048 // 4),
        "pca_wins_4_of_7": pca_wins >= 4,
        "pca_wins_count": pca_wins,
        "total_comparisons": total_comparisons,
    }
    h5_overall = h5_pass_criteria["saturation_knee_in_range"] or h5_pass_criteria["pca_wins_4_of_7"]

    # Note about pilot limitations
    pilot_limitations = []
    if PILOT_N_TRAIN < 2048:
        pilot_limitations.append(
            f"N={PILOT_N_TRAIN} caps PCA rank at {PILOT_N_TRAIN}, making PCA projections "
            f"at k>{PILOT_N_TRAIN} identical (all project to rank-{PILOT_N_TRAIN} space)"
        )
    pilot_limitations.append(
        "Random projection Recall@50 may be noisy at pilot scale"
    )
    pilot_limitations.append(
        f"Bootstrap CIs from {len(ref_data)} ref queries may be wide"
    )

    total_time = time.time() - t_start

    # ── Compile final results ────────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "grad_dim_D": D,
        "target_layers": TARGET_LAYER_PATTERNS,
        "task": "counterfact",
        "pilot_n_train": PILOT_N_TRAIN,
        "n_ref": len(ref_data),
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "mode": "pilot",
        "random_projection_results": random_results,
        "pca_projection_results": pca_results,
        "pca_vs_random_comparison": pca_vs_random,
        "smoking_gun_test": smoking_gun,
        "h5_analysis": {
            "max_random_recall": round(max_random_recall, 6),
            "saturation_90_k": saturation_90_k,
            "saturation_90_ratio_k_over_d": round(saturation_90_k / 2048, 4) if saturation_90_k else None,
            "approximately_monotonic": len(monotonic_violations) == 0,
            "monotonic_violations": monotonic_violations,
            "pass_criteria": h5_pass_criteria,
            "overall_pass": h5_overall,
        },
        "pilot_limitations": pilot_limitations,
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save to full results dir (with disk error handling)
    out_path = os.path.join(FULL_DIR, f"{TASK_ID}.json")
    result_json = json.dumps(final, indent=2)
    try:
        with open(out_path, "w") as f:
            f.write(result_json)
        print(f"\nResults saved to {out_path}")
    except OSError as e:
        print(f"\n[CRITICAL] Cannot save results to disk: {e}")
        print("[FALLBACK] Printing JSON to stdout for capture:")
        print("===RESULT_JSON_START===")
        print(result_json)
        print("===RESULT_JSON_END===")

    # Also save pilot summary
    pilots_dir = os.path.join(RESULTS_DIR, "pilots")
    os.makedirs(pilots_dir, exist_ok=True)

    pilot_summary = {
        "overall_recommendation": "GO" if h5_overall else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if h5_overall else "REFINE",
            "confidence": 0.7 if h5_overall else 0.5,
            "supported_hypotheses": (["H5_revised"] if h5_overall else []),
            "failed_assumptions": [],
            "key_metrics": {
                "max_random_recall": round(max_random_recall, 4),
                "saturation_90_k": saturation_90_k,
                "pca_at_d_recall": pca_at_d_recall,
                "pca_repsim_gap_pp": smoking_gun.get("gap_recall_pp"),
                "pca_wins": pca_wins,
                "best_random_tau": round(max(random_taus.values()), 4) if random_taus else None,
                "best_pca_tau": round(max(pca_taus.values()), 4) if pca_taus else None,
            },
            "notes": (
                f"Random projection saturates at k={saturation_90_k} (k/d={saturation_90_k/2048:.3f}). "
                if saturation_90_k else "No clear saturation point. "
            ) + (
                f"PCA wins {pca_wins}/{total_comparisons} comparisons. "
                f"TRAK-PCA(k=d) vs RepSim gap: {smoking_gun.get('gap_recall_pp', 'N/A')}pp. "
            ) + (
                f"Pilot limitation: N={PILOT_N_TRAIN} caps PCA rank."
            ),
        }],
        "pilot_limitations": pilot_limitations,
    }

    _safe_write(
        os.path.join(pilots_dir, f"{TASK_ID}_pilot_summary.json"),
        json.dumps(pilot_summary, indent=2),
    )

    # Markdown summary
    md_lines = [
        f"# P2: TRAK Dimension Sweep (H5-revised) -- Pilot Summary",
        f"",
        f"## Configuration",
        f"- Model: {MODEL_NAME} (d=2048)",
        f"- N_train: {PILOT_N_TRAIN} (pilot), N_ref: {len(ref_data)}",
        f"- Target layers: {', '.join(TARGET_LAYER_PATTERNS)}",
        f"- Grad dim D: {D:,}",
        f"- Random projection k: {RANDOM_DIMS}",
        f"- PCA projection k: {PCA_DIMS}",
        f"",
        f"## Random Projection Results",
        f"| k | k/d | Recall@50 | MRR | Kendall tau |",
        f"|---|-----|-----------|-----|-------------|",
    ]
    for k in sorted(random_recalls.keys()):
        r = random_results[str(k)]
        md_lines.append(
            f"| {k} | {k/2048:.3f} | {r['Recall@50']:.4f} | {r['MRR']:.4f} | {r['kendall_tau']:.4f} |"
        )

    md_lines += [
        f"",
        f"## PCA Projection Results",
        f"| k | k/d | Recall@50 | MRR | Kendall tau | Explained Var |",
        f"|---|-----|-----------|-----|-------------|---------------|",
    ]
    for k_str in sorted(pca_results.keys(), key=int):
        r = pca_results[k_str]
        ev = r["pca_info"]["explained_var_ratio"]
        md_lines.append(
            f"| {k_str} | {int(k_str)/2048:.3f} | {r['Recall@50']:.4f} | {r['MRR']:.4f} | {r['kendall_tau']:.4f} | {ev:.4f} |"
        )

    md_lines += [
        f"",
        f"## PCA vs Random Comparison",
        f"| k | PCA R@50 | Random R@50 | PCA advantage (pp) |",
        f"|---|---------|------------|-------------------|",
    ]
    for k_str in sorted(pca_vs_random.keys(), key=int):
        c = pca_vs_random[k_str]
        md_lines.append(
            f"| {k_str} | {c['pca_recall']:.4f} | {c['random_recall']:.4f} | {c['pca_advantage_pp']:+.2f} |"
        )

    md_lines += [
        f"",
        f"## Smoking-Gun Test (TRAK-PCA at k=d vs RepSim)",
        f"- TRAK-PCA(k=d=2048) Recall@50: {pca_at_d_recall}",
        f"- RepSim Recall@50: {repsim_recall}",
        f"- Gap: {smoking_gun.get('gap_recall_pp', 'N/A')}pp",
        f"- Interpretation: {smoking_gun.get('interpretation', 'N/A')}",
        f"",
        f"## H5 Assessment",
        f"- Saturation at k={saturation_90_k} (k/d={saturation_90_k/2048:.3f})" if saturation_90_k else "- No clear saturation",
        f"- PCA wins: {pca_wins}/{total_comparisons}",
        f"- Overall: {'PASS' if h5_overall else 'NEEDS FULL SCALE'}",
        f"",
        f"## Pilot Limitations",
    ]
    for lim in pilot_limitations:
        md_lines.append(f"- {lim}")

    md_lines += [
        f"",
        f"## Runtime",
        f"- Total: {total_time:.1f}s",
        f"- GPU: {final['gpu']}",
    ]

    _safe_write(
        os.path.join(pilots_dir, f"{TASK_ID}_pilot_summary.md"),
        "\n".join(md_lines),
    )

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: TRAK Dimension Sweep (H5-revised)")
    print("=" * 70)

    print(f"\nRandom Projection:")
    print(f"{'k':<8}{'k/d':<8}{'R@50':<10}{'MRR':<10}{'tau':<10}{'Time':<8}")
    print("-" * 54)
    for k in sorted(random_recalls.keys()):
        r = random_results[str(k)]
        print(f"{k:<8}{k/2048:<8.3f}{r['Recall@50']:<10.4f}{r['MRR']:<10.4f}"
              f"{r['kendall_tau']:<10.4f}{r['runtime_sec']:<8.1f}")

    print(f"\nPCA Projection:")
    print(f"{'k':<8}{'k/d':<8}{'R@50':<10}{'MRR':<10}{'tau':<10}{'ExplVar':<10}{'Time':<8}")
    print("-" * 64)
    for k_str in sorted(pca_results.keys(), key=int):
        r = pca_results[k_str]
        ev = r["pca_info"]["explained_var_ratio"]
        print(f"{k_str:<8}{int(k_str)/2048:<8.3f}{r['Recall@50']:<10.4f}{r['MRR']:<10.4f}"
              f"{r['kendall_tau']:<10.4f}{ev:<10.4f}{r['runtime_sec']:<8.1f}")

    print(f"\nH5: {'PASS' if h5_overall else 'NEEDS FULL SCALE'}")
    print(f"Saturation: k={saturation_90_k}")
    print(f"PCA advantage: {pca_wins}/{total_comparisons} comparisons won")
    print(f"Smoking-gun gap: {smoking_gun.get('gap_recall_pp', 'N/A')}pp")
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 70)

    mark_done(
        "success",
        f"Dim sweep pilot done in {total_time:.0f}s. "
        f"Saturation@k={saturation_90_k}. "
        f"PCA wins {pca_wins}/{total_comparisons}. "
        f"Smoking-gun gap={smoking_gun.get('gap_recall_pp', 'N/A')}pp."
    )
    return final


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"FATAL: {e}\n{traceback.format_exc()}")
        mark_done("failed", str(e)[:300])
        sys.exit(1)
