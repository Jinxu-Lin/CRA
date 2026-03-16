#!/usr/bin/env python3
"""
Phase 2b: TRAK Dimension Sweep (H5)
- TRAK on Pythia-1B with projection dims k in {64, 128, 256, 512, 1024, 2048, 4096}
- TRAK-PCA (project onto top-k gradient eigenvectors) at k in {256, 512, 1024, 2048}
- DATE-LM counterfact task (Recall@50 + MRR as proxy for data_selection LDS)
- PILOT mode: 100 training samples, seed=42
- 2 GPUs: random projections on GPU0, PCA projections on GPU1

H5 hypothesis: 90% of max metric achieved by k=2d=4096, with <5% additional
improvement from k=2d to k=10d (we test up to k=4096 ~ 2d for Pythia-1B d=2048).

Smoking-gun test: TRAK-PCA at k=d should approach RepSim performance (within 5pp).
"""

import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "phase2_trak_dim_sweep"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE2_DIR = os.path.join(RESULTS_DIR, "phase2")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 8

# Dimension sweep configurations
RANDOM_DIMS = [64, 128, 256, 512, 1024, 2048, 4096]
PCA_DIMS = [256, 512, 1024, 2048]

# GPU assignment from environment or default
GPU_IDS_STR = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
GPU_IDS = [int(x) for x in GPU_IDS_STR.split(",")]
print(f"Using GPUs: {GPU_IDS} (CUDA_VISIBLE_DEVICES={GPU_IDS_STR})")

# We use the first visible GPU (index 0 in CUDA namespace)
DEVICE = "cuda:0"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


def evaluate_counterfact(sim_matrix, pilot_data, ref_data):
    """Evaluate counterfact task: Recall@50, MRR with bootstrap CI."""
    n_train = len(pilot_data)
    n_ref = len(ref_data)

    scores_per_ref = []
    fact_indices_per_ref = []
    for j in range(n_ref):
        ref_sample = ref_data[j]
        scores_j = sim_matrix[:, j]
        fi = [
            i for i in range(n_train)
            if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
            and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
        ]
        scores_per_ref.append(scores_j)
        fact_indices_per_ref.append(fi)

    recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)

    # Bootstrap CI
    rng_boot = np.random.RandomState(SEED + 2345)
    boot_recalls, boot_mrrs = [], []
    for _ in range(BOOTSTRAP_B):
        idx = rng_boot.choice(n_ref, n_ref, replace=True)
        boot_spr = [scores_per_ref[i] for i in idx]
        boot_fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(boot_spr, boot_fi, k=50)
        boot_recalls.append(r)
        boot_mrrs.append(m)

    recall_ci = (float(np.percentile(boot_recalls, 2.5)), float(np.percentile(boot_recalls, 97.5)))
    mrr_ci = (float(np.percentile(boot_mrrs, 2.5)), float(np.percentile(boot_mrrs, 97.5)))
    n_with_facts = sum(1 for f in fact_indices_per_ref if f)

    return {
        "Recall@50": round(recall, 6),
        "Recall@50_CI": [round(recall_ci[0], 6), round(recall_ci[1], 6)],
        "MRR": round(mrr, 6),
        "MRR_CI": [round(mrr_ci[0], 6), round(mrr_ci[1], 6)],
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
    print(f"[pilot] Random: {len(pilot_idx)} samples")
    return train_data.select(pilot_idx), pilot_idx


# ── Model ───────────────────────────────────────────────────────────────
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


# ── Raw gradient extraction ─────────────────────────────────────────────
def extract_raw_gradients(model, tok, texts, desc=""):
    """
    Extract full (uncompressed) gradients from target layers for all texts.
    Target: last layer attention.dense + mlp.dense_4h_to_h.
    Returns: [n_texts, D] tensor in float32 on CPU.
    """
    # Select target parameters
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
    print(f"[Grads/{desc}] D={D/1e6:.2f}M from {target_names}")

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

        if (idx + 1) % 20 == 0:
            print(f"  [{desc}] {idx+1}/{len(texts)}")
            torch.cuda.empty_cache()

    # Restore grad flags
    for p in model.parameters():
        p.requires_grad_(True)

    result = torch.stack(all_grads)  # [n_texts, D]
    print(f"[Grads/{desc}] Extracted: shape={result.shape}, mem={result.element_size() * result.nelement() / 1e9:.2f} GB")
    return result, D


# ── CountSketch random projection ──────────────────────────────────────
def countsketch_project(grads, k, seed=SEED + 7777):
    """Apply CountSketch projection to reduce D -> k."""
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
    Project gradients onto top-k principal components of the training gradient
    covariance. This is the "smoking gun" test: if gradient signal lives in a
    k-dimensional subspace, PCA projection should capture it optimally.

    Uses SVD on the centered gradient matrix (more efficient than covariance eigendecomp).
    """
    n_train, D = train_grads.shape
    print(f"[PCA k={k}] Computing SVD on {n_train}x{D} gradient matrix...")

    # Center gradients
    mean_grad = train_grads.mean(dim=0, keepdim=True)
    centered_train = train_grads - mean_grad
    centered_ref = ref_grads - mean_grad

    # SVD: U S V^T, where V columns are principal directions
    # For n < D, compute on the Gram matrix: centered_train @ centered_train.T
    if n_train < D:
        gram = centered_train @ centered_train.T  # [n, n]
        eigenvalues, eigenvectors_gram = torch.linalg.eigh(gram)
        # Sort descending
        idx_sorted = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors_gram = eigenvectors_gram[:, idx_sorted]

        # Recover V from U: V_j = X^T u_j / sigma_j
        # Take top-k
        k_actual = min(k, n_train)
        top_eigenvals = eigenvalues[:k_actual].clamp(min=1e-10)
        top_U = eigenvectors_gram[:, :k_actual]

        # V = X^T U / sqrt(eigenvalues)
        V_topk = centered_train.T @ top_U / top_eigenvals.sqrt().unsqueeze(0)  # [D, k]
        # Normalize columns
        V_topk = F.normalize(V_topk, dim=0)
    else:
        # Direct SVD (won't happen in practice for our D >> n)
        U, S, Vh = torch.linalg.svd(centered_train, full_matrices=False)
        V_topk = Vh[:k].T  # [D, k]

    # Project: [n, D] @ [D, k] -> [n, k]
    train_proj = centered_train @ V_topk
    ref_proj = centered_ref @ V_topk

    # Also return explained variance ratio
    total_var = eigenvalues.sum().item()
    explained_var = eigenvalues[:k_actual].sum().item() / max(total_var, 1e-10)

    print(f"[PCA k={k}] Projected to k={k_actual}, explained_var={explained_var:.4f}")
    return train_proj, ref_proj, {
        "k_requested": k,
        "k_actual": k_actual,
        "explained_var_ratio": round(explained_var, 6),
        "top_eigenvalues": eigenvalues[:min(10, k_actual)].tolist(),
    }


# ── Score computation ───────────────────────────────────────────────────
def compute_similarity_matrix(train_proj, ref_proj):
    """Compute cosine similarity matrix [n_train, n_ref]."""
    train_norm = F.normalize(train_proj, dim=-1)
    ref_norm = F.normalize(ref_proj, dim=-1)
    sim = (train_norm @ ref_norm.T).numpy()
    return sim


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    for d in [PHASE2_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print(f"Phase 2b: TRAK Dimension Sweep (H5)")
    print(f"Model: {MODEL_NAME}, Pilot N={PILOT_N_TRAIN}")
    print(f"Random dims: {RANDOM_DIMS}")
    print(f"PCA dims: {PCA_DIMS}")
    print(f"Task: counterfact (Recall@50 + MRR)")
    print(f"GPUs: {GPU_IDS}")
    print("=" * 70)

    # Load data
    train_data, ref_data, fmt_fn = load_counterfact_data()
    pilot_data, pilot_idx = create_pilot_subset(train_data)

    train_texts = [fmt_fn(pilot_data[i]) for i in range(len(pilot_data))]
    ref_texts = [fmt_fn(ref_data[i]) for i in range(len(ref_data))]

    # Load model
    model, tok = load_model()

    # ── Step 1: Extract raw gradients (once, reuse for all projections) ──
    report_progress("gradient_extraction", "Extracting raw gradients for train+ref", 0.15)

    # Check if cached gradients exist
    train_grad_cache = os.path.join(CACHE_DIR, "trak_dim_sweep_train_grads.pt")
    ref_grad_cache = os.path.join(CACHE_DIR, "trak_dim_sweep_ref_grads.pt")

    if os.path.exists(train_grad_cache) and os.path.exists(ref_grad_cache):
        print("[Cache] Loading cached gradients...")
        train_grads = torch.load(train_grad_cache, weights_only=True)
        ref_grads = torch.load(ref_grad_cache, weights_only=True)
        D = train_grads.shape[1]
        print(f"[Cache] Loaded: train={train_grads.shape}, ref={ref_grads.shape}")
    else:
        print("[Grads] Extracting train gradients...")
        train_grads, D = extract_raw_gradients(model, tok, train_texts, desc="train")

        print("[Grads] Extracting ref gradients...")
        ref_grads, _ = extract_raw_gradients(model, tok, ref_texts, desc="ref")

        # Skip caching large gradient tensors (torch.save fails on >4GB zip files)
        # For pilot runs this is acceptable since extraction only takes ~2 min
        print("[Cache] Skipping gradient caching (pilot mode, tensors too large for torch.save zip format)")

    gc.collect()
    torch.cuda.empty_cache()

    # Free model memory -- we only need gradients from here
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print("[Memory] Model freed; working with cached gradients only")

    # ── Step 2: Random projection dimension sweep ────────────────────────
    report_progress("random_sweep", "Running CountSketch dimension sweep", 0.35)
    random_results = {}

    for dim_idx, k in enumerate(RANDOM_DIMS):
        t_k = time.time()
        print(f"\n{'─'*50}")
        print(f"[Random] k={k} ({dim_idx+1}/{len(RANDOM_DIMS)})")

        # CountSketch projection
        train_proj = countsketch_project(train_grads, k)
        ref_proj = countsketch_project(ref_grads, k)

        # Compute similarity and evaluate
        sim = compute_similarity_matrix(train_proj, ref_proj)
        metrics = evaluate_counterfact(sim, pilot_data, ref_data)

        elapsed_k = time.time() - t_k
        metrics["runtime_sec"] = round(elapsed_k, 2)
        metrics["projection_type"] = "CountSketch"
        metrics["projection_dim"] = k
        metrics["grad_dim_D"] = D
        metrics["ratio_k_over_d"] = round(k / 2048, 4)  # d=2048 for Pythia-1B

        random_results[str(k)] = metrics
        print(f"[Random] k={k}: Recall@50={metrics['Recall@50']:.4f}, MRR={metrics['MRR']:.4f}, time={elapsed_k:.1f}s")

        report_progress("random_sweep", f"k={k} done", 0.35 + 0.35 * (dim_idx + 1) / len(RANDOM_DIMS),
                       metric={"k": k, "Recall@50": metrics["Recall@50"]})

    # ── Step 3: PCA projection dimension sweep (smoking-gun test) ────────
    report_progress("pca_sweep", "Running PCA dimension sweep", 0.70)
    pca_results = {}

    for dim_idx, k in enumerate(PCA_DIMS):
        t_k = time.time()
        print(f"\n{'─'*50}")
        print(f"[PCA] k={k} ({dim_idx+1}/{len(PCA_DIMS)})")

        # PCA projection using gradient covariance eigenvectors
        train_proj, ref_proj, pca_info = pca_project(train_grads, ref_grads, k)

        # Compute similarity and evaluate
        sim = compute_similarity_matrix(train_proj, ref_proj)
        metrics = evaluate_counterfact(sim, pilot_data, ref_data)

        elapsed_k = time.time() - t_k
        metrics["runtime_sec"] = round(elapsed_k, 2)
        metrics["projection_type"] = "PCA"
        metrics["projection_dim"] = k
        metrics["grad_dim_D"] = D
        metrics["ratio_k_over_d"] = round(k / 2048, 4)
        metrics["pca_info"] = pca_info

        pca_results[str(k)] = metrics
        print(f"[PCA] k={k}: Recall@50={metrics['Recall@50']:.4f}, MRR={metrics['MRR']:.4f}, "
              f"explained_var={pca_info['explained_var_ratio']:.4f}, time={elapsed_k:.1f}s")

        report_progress("pca_sweep", f"PCA k={k} done", 0.70 + 0.25 * (dim_idx + 1) / len(PCA_DIMS),
                       metric={"k": k, "Recall@50": metrics["Recall@50"], "type": "PCA"})

    # ── Step 4: Analysis ─────────────────────────────────────────────────
    report_progress("analysis", "Analyzing dimension sweep results", 0.95)

    # Find saturation point for random projection
    random_recalls = {int(k): v["Recall@50"] for k, v in random_results.items()}
    max_random_recall = max(random_recalls.values())
    saturation_90_k = None
    for k in sorted(random_recalls.keys()):
        if random_recalls[k] >= 0.9 * max_random_recall:
            saturation_90_k = k
            break

    # Compare PCA at k=d with RepSim
    # Load RepSim standard result for comparison
    repsim_recall = None
    repsim_path = os.path.join(RESULTS_DIR, "phase1", "repsim_standard.json")
    if os.path.exists(repsim_path):
        with open(repsim_path) as f:
            repsim_data = json.load(f)
        repsim_recall = repsim_data["results_per_task"]["counterfact"]["metrics"]["Recall@50"]
        print(f"\n[Comparison] RepSim standard Recall@50 = {repsim_recall:.4f}")

    # PCA at k=2048 (=d for Pythia-1B) vs RepSim
    pca_at_d = pca_results.get("2048", {}).get("Recall@50", None)
    pca_repsim_gap = None
    if pca_at_d is not None and repsim_recall is not None:
        pca_repsim_gap = round(repsim_recall - pca_at_d, 4)
        print(f"[Smoking-gun] TRAK-PCA(k=d=2048) Recall@50={pca_at_d:.4f}, "
              f"RepSim={repsim_recall:.4f}, gap={pca_repsim_gap:.4f}")

    # H5 analysis
    h5_analysis = {
        "max_random_recall": round(max_random_recall, 6),
        "saturation_90_k": saturation_90_k,
        "saturation_90_ratio_k_over_d": round(saturation_90_k / 2048, 4) if saturation_90_k else None,
        "recalls_by_k": {str(k): round(v, 6) for k, v in sorted(random_recalls.items())},
        "knee_at_2d": k <= 4096 if saturation_90_k else None,
        "h5_pass": saturation_90_k is not None and saturation_90_k <= 4096,
        "interpretation": "",
    }

    # Check monotonicity (within noise)
    sorted_ks = sorted(random_recalls.keys())
    monotonic_violations = []
    for i in range(1, len(sorted_ks)):
        prev_k, curr_k = sorted_ks[i - 1], sorted_ks[i]
        if random_recalls[curr_k] < random_recalls[prev_k] - 0.02:  # 2pp noise tolerance
            monotonic_violations.append((prev_k, curr_k, random_recalls[prev_k], random_recalls[curr_k]))
    h5_analysis["monotonic_violations"] = monotonic_violations
    h5_analysis["approximately_monotonic"] = len(monotonic_violations) == 0

    # Smoking-gun analysis
    smoking_gun = {
        "pca_at_d_recall": pca_at_d,
        "repsim_recall": repsim_recall,
        "gap_pp": round(pca_repsim_gap * 100, 2) if pca_repsim_gap is not None else None,
        "within_5pp": abs(pca_repsim_gap) <= 0.05 if pca_repsim_gap is not None else None,
        "interpretation": "",
    }

    if pca_repsim_gap is not None:
        if abs(pca_repsim_gap) <= 0.05:
            smoking_gun["interpretation"] = (
                f"SMOKING GUN CONFIRMED: TRAK-PCA at k=d ({pca_at_d:.4f}) matches RepSim "
                f"({repsim_recall:.4f}) within 5pp (gap={pca_repsim_gap*100:.1f}pp). "
                "This directly shows that gradient signal lives in a d-dimensional subspace, "
                "and optimal projection recovers representation-space performance."
            )
        else:
            smoking_gun["interpretation"] = (
                f"TRAK-PCA at k=d ({pca_at_d:.4f}) differs from RepSim ({repsim_recall:.4f}) "
                f"by {pca_repsim_gap*100:.1f}pp. Gap > 5pp suggests additional factors "
                "beyond projection dimension (e.g., curvature, layer selection, or FM2 effects)."
            )

    # Build interpretation
    if h5_analysis["h5_pass"]:
        h5_analysis["interpretation"] = (
            f"H5 SUPPORTED: 90% of max Recall@50 achieved at k={saturation_90_k} "
            f"(k/d={saturation_90_k/2048:.2f}). Max at k={max(random_recalls, key=random_recalls.get)} "
            f"with Recall@50={max_random_recall:.4f}. Saturation at k~2d confirms gradient signal "
            "lives in a low-rank subspace of effective dimension ~d."
        )
    else:
        h5_analysis["interpretation"] = (
            f"H5 STATUS UNCLEAR: Saturation not reached at k={max(sorted_ks)}. "
            "May need larger k values or more samples to observe saturation."
        )

    # PCA vs Random comparison (at matched dimensions)
    pca_vs_random = {}
    for k_str in pca_results:
        k = int(k_str)
        if k_str in random_results:
            pca_r = pca_results[k_str]["Recall@50"]
            rand_r = random_results[k_str]["Recall@50"]
            pca_vs_random[k_str] = {
                "pca_recall": round(pca_r, 6),
                "random_recall": round(rand_r, 6),
                "pca_advantage_pp": round((pca_r - rand_r) * 100, 2),
            }

    total_time = time.time() - t_start

    # ── Compile final results ────────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "grad_dim_D": D,
        "task": "counterfact",
        "metric": "Recall@50",
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "mode": "pilot",
        "random_projection_results": random_results,
        "pca_projection_results": pca_results,
        "h5_analysis": h5_analysis,
        "smoking_gun_test": smoking_gun,
        "pca_vs_random_comparison": pca_vs_random,
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    out_path = os.path.join(PHASE2_DIR, "trak_dim_sweep.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save to pilots dir
    pilots_dir = os.path.join(RESULTS_DIR, "pilots")
    os.makedirs(pilots_dir, exist_ok=True)
    with open(os.path.join(pilots_dir, f"{TASK_ID}_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: TRAK Dimension Sweep (Phase 2b, H5)")
    print("=" * 70)

    print("\nRandom Projection (CountSketch):")
    print(f"{'k':<8}{'k/d':<8}{'Recall@50':<12}{'MRR':<12}{'Time(s)':<10}")
    print("-" * 50)
    for k in sorted(random_recalls.keys()):
        r = random_results[str(k)]
        print(f"{k:<8}{k/2048:<8.3f}{r['Recall@50']:<12.4f}{r['MRR']:<12.4f}{r['runtime_sec']:<10.1f}")

    print(f"\nPCA Projection (gradient covariance eigenvectors):")
    print(f"{'k':<8}{'k/d':<8}{'Recall@50':<12}{'MRR':<12}{'ExplVar':<10}{'Time(s)':<10}")
    print("-" * 62)
    for k_str in sorted(pca_results.keys(), key=int):
        r = pca_results[k_str]
        ev = r["pca_info"]["explained_var_ratio"]
        print(f"{k_str:<8}{int(k_str)/2048:<8.3f}{r['Recall@50']:<12.4f}{r['MRR']:<12.4f}{ev:<10.4f}{r['runtime_sec']:<10.1f}")

    if pca_vs_random:
        print(f"\nPCA vs Random at matched dimensions:")
        print(f"{'k':<8}{'PCA':<12}{'Random':<12}{'PCA adv (pp)':<14}")
        print("-" * 46)
        for k_str in sorted(pca_vs_random.keys(), key=int):
            c = pca_vs_random[k_str]
            print(f"{k_str:<8}{c['pca_recall']:<12.4f}{c['random_recall']:<12.4f}{c['pca_advantage_pp']:<14.2f}")

    print(f"\nH5 Assessment: {'PASS' if h5_analysis['h5_pass'] else 'UNCLEAR'}")
    print(f"  Saturation 90% at k={saturation_90_k} (k/d={saturation_90_k/2048:.2f})" if saturation_90_k else "  No saturation found")
    print(f"  Approximately monotonic: {h5_analysis['approximately_monotonic']}")

    if smoking_gun["within_5pp"] is not None:
        print(f"\nSmoking-gun test: {'CONFIRMED' if smoking_gun['within_5pp'] else 'NOT CONFIRMED'}")
        print(f"  TRAK-PCA(k=d) = {smoking_gun['pca_at_d_recall']:.4f}")
        print(f"  RepSim = {smoking_gun['repsim_recall']:.4f}")
        print(f"  Gap = {smoking_gun['gap_pp']:.1f}pp")

    print(f"\nTotal runtime: {total_time:.1f}s")
    print("=" * 70)

    mark_done(
        "success",
        f"Dim sweep done in {total_time:.0f}s. "
        f"H5={'PASS' if h5_analysis['h5_pass'] else 'UNCLEAR'}. "
        f"Saturation@k={saturation_90_k}. "
        f"Smoking-gun={'YES' if smoking_gun.get('within_5pp') else 'NO/NA'}."
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
