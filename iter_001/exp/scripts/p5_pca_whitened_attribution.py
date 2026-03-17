#!/usr/bin/env python3
"""
P5: PCA-Reduced Whitened Attribution (H7-revised) -- PILOT MODE
================================================================
Project representations to top-k PCA components, then whiten with
the k x k covariance matrix in that subspace. Compare against
standard RepSim (M=I).

k in {16, 32, 64, 128, 256, 512}
At N=100 pilot: N/k ranges from 6.25 (k=16) to 0.2 (k=512).
Only k<=64 has N/k > 1 at pilot scale -- but we run all to
characterize the degradation curve.

Also tests ridge-regularized whitening with 5-fold cross-validated lambda.

Pass criteria (pilot): PCA-whitened at k=64 does not degrade RepSim by > 5pp
(unlike full-dimensional whitening which lost 8-11pp); positive SNR-accuracy
correlation maintained.
"""

import os, sys, json, time, gc, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, auc
from sklearn.covariance import LedoitWolf
from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p5_pca_whitened_attribution"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-1b"
HIDDEN_DIM = 2048
PCA_DIMS = [16, 32, 64, 128, 256, 512]
RIDGE_LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
CV_FOLDS = 5
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "repsim_standard_reps.pt")
PILOTS_DIR = os.path.join(RESULTS_DIR, "pilots")
BOOTSTRAP_B = 1000

np.random.seed(SEED)
torch.manual_seed(SEED)

for d in [FULL_DIR, CACHE_DIR, PILOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Progress / lifecycle ────────────────────────────────────────────────
def _safe_write(path, content):
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


# ── Evaluation helpers (same as p2_repsim_dim_sweep) ────────────────────
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


def compute_continuous_metrics_toxicity(train_scores, pilot_data, n_train):
    binary_rel = np.array([1.0 if pilot_data[i]["type"] == "Unsafe" else 0.0
                           for i in range(n_train)])
    if np.std(train_scores) < 1e-12 or np.sum(binary_rel) == 0:
        return 0.0, 0.0
    tau, _ = kendalltau(train_scores, binary_rel)
    rho, _ = spearmanr(train_scores, binary_rel)
    return (float(tau) if not np.isnan(tau) else 0.0,
            float(rho) if not np.isnan(rho) else 0.0)


def compute_continuous_metrics_factual(scores_per_ref, fact_indices_per_ref, n_train):
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


def get_per_query_accuracy(sim_matrix, pilot_data, ref_data, task_name, n_train):
    """Per-query R@50 accuracy for SNR correlation analysis."""
    n_ref = sim_matrix.shape[1]
    accuracies = []

    if task_name == "counterfact":
        for j in range(n_ref):
            ref_sample = ref_data[j]
            scores_j = sim_matrix[:, j]
            fi = [
                i for i in range(n_train)
                if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
            ]
            if not fi:
                continue
            si = np.argsort(-scores_j)
            topk = set(si[:50].tolist())
            acc = len([i for i in fi if i in topk]) / len(fi)
            accuracies.append(acc)
    elif task_name == "ftrace":
        train_facts_sets = []
        for i in range(n_train):
            facts_raw = pilot_data[i].get("facts", [])
            if isinstance(facts_raw, str):
                facts_raw = [f.strip() for f in facts_raw.split(",") if f.strip()]
            elif isinstance(facts_raw, list):
                flat = []
                for f in facts_raw:
                    if isinstance(f, str):
                        flat.extend([x.strip() for x in f.split(",") if x.strip()])
                facts_raw = flat
            train_facts_sets.append(set(facts_raw))

        for j in range(n_ref):
            ref_sample = ref_data[j]
            scores_j = sim_matrix[:, j]
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
            if not fi:
                continue
            si = np.argsort(-scores_j)
            topk = set(si[:50].tolist())
            acc = len([i for i in fi if i in topk]) / len(fi)
            accuracies.append(acc)

    return np.array(accuracies)


def evaluate_task(sim_matrix, pilot_data, ref_data, task_name, n_train):
    """Full evaluation for a task: returns metrics dict."""
    if task_name == "toxicity":
        train_scores = sim_matrix.mean(axis=1)
        unsafe_indices = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
        auprc = compute_auprc(train_scores, unsafe_indices, n_train)
        tau, rho = compute_continuous_metrics_toxicity(train_scores, pilot_data, n_train)
        return {
            "AUPRC": round(auprc, 6),
            "n_unsafe": len(unsafe_indices),
            "n_train": n_train,
            "kendall_tau": round(tau, 6),
            "spearman_rho": round(rho, 6),
        }
    else:
        # counterfact or ftrace
        n_ref = sim_matrix.shape[1]
        scores_per_ref = []
        fact_indices_per_ref = []

        if task_name == "counterfact":
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
        elif task_name == "ftrace":
            train_facts_sets = []
            for i in range(n_train):
                facts_raw = pilot_data[i].get("facts", [])
                if isinstance(facts_raw, str):
                    facts_raw = [f.strip() for f in facts_raw.split(",") if f.strip()]
                elif isinstance(facts_raw, list):
                    flat = []
                    for f in facts_raw:
                        if isinstance(f, str):
                            flat.extend([x.strip() for x in f.split(",") if x.strip()])
                    facts_raw = flat
                train_facts_sets.append(set(facts_raw))

            for j in range(n_ref):
                ref_sample = ref_data[j]
                scores_j = sim_matrix[:, j].tolist()
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
                scores_per_ref.append(scores_j)
                fact_indices_per_ref.append(fi)

        recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)
        tau, rho = compute_continuous_metrics_factual(scores_per_ref, fact_indices_per_ref, n_train)
        n_with_facts = sum(1 for f in fact_indices_per_ref if f)

        return {
            "Recall@50": round(recall, 6),
            "MRR": round(mrr, 6),
            "kendall_tau": round(tau, 6),
            "spearman_rho": round(rho, 6),
            "refs_with_facts": n_with_facts,
            "n_ref": n_ref,
            "n_train": n_train,
        }


# ── PCA Whitening methods ──────────────────────────────────────────────
def pca_reduce(train_reps_np, ref_reps_np, k):
    """Apply PCA to reduce from d to k. Fit on train only (avoid data leakage)."""
    d = train_reps_np.shape[1]
    n_train = train_reps_np.shape[0]
    max_components = min(n_train, d) - 1

    if k >= max_components:
        return train_reps_np.copy(), ref_reps_np.copy(), {
            "effective_k": d,
            "explained_variance_ratio": 1.0,
            "note": f"k={k} >= max_components={max_components}; using full"
        }, None

    pca = PCA(n_components=k, random_state=SEED)
    train_reduced = pca.fit_transform(train_reps_np)
    ref_reduced = pca.transform(ref_reps_np)
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    info = {
        "effective_k": k,
        "explained_variance_ratio": round(explained_var, 6),
        "top_eigenvalues": [round(float(v), 6) for v in pca.explained_variance_[:min(10, k)]],
        "n_fit_samples": n_train,
    }
    return train_reduced, ref_reduced, info, pca


def standard_repsim(train_reps, ref_reps):
    """Standard RepSim: cosine similarity (M=I)."""
    train_norm = train_reps / (np.linalg.norm(train_reps, axis=1, keepdims=True) + 1e-8)
    ref_norm = ref_reps / (np.linalg.norm(ref_reps, axis=1, keepdims=True) + 1e-8)
    return train_norm @ ref_norm.T


def whitened_repsim(train_reps, ref_reps, regularization="ledoit_wolf", ridge_lambda=0.0):
    """
    Whitened RepSim: phi^T Sigma^{-1} psi in PCA-reduced space.

    Sigma is estimated from train_reps only.
    regularization: 'ledoit_wolf', 'ridge', or 'none'
    """
    n_train, k = train_reps.shape

    if regularization == "ledoit_wolf":
        try:
            lw = LedoitWolf().fit(train_reps)
            cov = lw.covariance_
            shrinkage = float(lw.shrinkage_)
        except Exception as e:
            print(f"  [Warn] Ledoit-Wolf failed: {e}, falling back to ridge")
            cov = np.cov(train_reps, rowvar=False)
            shrinkage = -1.0
    elif regularization == "ridge":
        cov = np.cov(train_reps, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = cov + ridge_lambda * np.eye(k)
        shrinkage = ridge_lambda
    else:
        cov = np.cov(train_reps, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        shrinkage = 0.0

    # Compute condition number before inversion
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    condition_number = float(eigenvalues[-1] / eigenvalues[0])

    # Compute inverse
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        cov_inv = np.linalg.pinv(cov)

    # Whitened similarity: train^T @ Sigma^{-1} @ ref
    # For each ref query j: score_i = train_i^T @ Sigma^{-1} @ ref_j
    whitened_ref = ref_reps @ cov_inv.T  # (n_ref, k)
    sim_matrix = train_reps @ whitened_ref.T  # (n_train, n_ref)

    diag_info = {
        "shrinkage": round(shrinkage, 6),
        "condition_number": round(condition_number, 2),
        "cov_eigenvalue_range": [round(float(eigenvalues[0]), 8),
                                  round(float(eigenvalues[-1]), 8)],
        "sim_stats": {
            "mean": round(float(np.mean(sim_matrix)), 4),
            "std": round(float(np.std(sim_matrix)), 4),
            "min": round(float(np.min(sim_matrix)), 4),
            "max": round(float(np.max(sim_matrix)), 4),
        },
    }
    return sim_matrix, diag_info


def ridge_cv_whitened_repsim(train_reps, ref_reps, pilot_data, ref_data, task_name,
                              n_train, lambdas=RIDGE_LAMBDAS, n_folds=CV_FOLDS):
    """
    Ridge-regularized whitening with cross-validated lambda.

    CV strategy: split ref queries into folds, evaluate each lambda
    on held-out fold, pick best lambda.
    """
    n_ref = ref_reps.shape[0]
    k = train_reps.shape[1]

    if n_ref < n_folds:
        # Not enough ref samples for CV; fall back to default lambda
        print(f"  [Warn] n_ref={n_ref} < n_folds={n_folds}, using lambda=0.01")
        sim, info = whitened_repsim(train_reps, ref_reps, "ridge", ridge_lambda=0.01)
        info["cv_note"] = f"Insufficient ref samples for {n_folds}-fold CV"
        info["selected_lambda"] = 0.01
        return sim, info

    # For each lambda, compute average score across CV folds
    rng = np.random.RandomState(SEED + 777)
    fold_indices = rng.permutation(n_ref) % n_folds

    lambda_scores = {}
    for lam in lambdas:
        fold_metrics = []
        for fold in range(n_folds):
            val_mask = fold_indices == fold
            val_refs = ref_reps[val_mask]

            sim, _ = whitened_repsim(train_reps, val_refs, "ridge", ridge_lambda=lam)

            # Evaluate on validation fold
            val_ref_data_indices = np.where(val_mask)[0]

            if task_name == "toxicity":
                val_train_scores = sim.mean(axis=1)
                unsafe_idx = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
                m = compute_auprc(val_train_scores, unsafe_idx, n_train)
            elif task_name == "counterfact":
                val_recalls = []
                for jj, orig_j in enumerate(val_ref_data_indices):
                    ref_sample = ref_data[int(orig_j)]
                    scores_j = sim[:, jj]
                    fi = [
                        i for i in range(n_train)
                        if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                        and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
                    ]
                    if not fi:
                        continue
                    si = np.argsort(-scores_j)
                    topk = set(si[:50].tolist())
                    val_recalls.append(len([i for i in fi if i in topk]) / len(fi))
                m = float(np.mean(val_recalls)) if val_recalls else 0.0
            elif task_name == "ftrace":
                train_facts_sets = []
                for i in range(n_train):
                    facts_raw = pilot_data[i].get("facts", [])
                    if isinstance(facts_raw, str):
                        facts_raw = [f.strip() for f in facts_raw.split(",") if f.strip()]
                    elif isinstance(facts_raw, list):
                        flat = []
                        for f in facts_raw:
                            if isinstance(f, str):
                                flat.extend([x.strip() for x in f.split(",") if x.strip()])
                        facts_raw = flat
                    train_facts_sets.append(set(facts_raw))

                val_recalls = []
                for jj, orig_j in enumerate(val_ref_data_indices):
                    ref_sample = ref_data[int(orig_j)]
                    scores_j = sim[:, jj]
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
                    if not fi:
                        continue
                    si = np.argsort(-scores_j)
                    topk = set(si[:50].tolist())
                    val_recalls.append(len([i for i in fi if i in topk]) / len(fi))
                m = float(np.mean(val_recalls)) if val_recalls else 0.0
            else:
                m = 0.0

            fold_metrics.append(m)

        lambda_scores[lam] = float(np.mean(fold_metrics))

    # Select best lambda
    best_lambda = max(lambda_scores, key=lambda_scores.get)
    best_score = lambda_scores[best_lambda]

    # Re-run with best lambda on full data
    sim, info = whitened_repsim(train_reps, ref_reps, "ridge", ridge_lambda=best_lambda)
    info["cv_lambda_scores"] = {str(l): round(s, 6) for l, s in lambda_scores.items()}
    info["selected_lambda"] = best_lambda
    info["cv_best_score"] = round(best_score, 6)
    info["n_folds"] = n_folds

    return sim, info


def compute_snr(train_reps, pilot_data, ref_data, task_name, n_train, cov_inv):
    """
    Compute per-query output SNR for whitened attribution.

    SNR_out(q) = |mu_signal^T M mu_signal| / sqrt(mu_signal^T M Sigma_noise M mu_signal)

    Simplified for PCA-whitened: M = Sigma^{-1}, so
    SNR_out(q) = sqrt(mu_signal^T Sigma^{-1} mu_signal)
    """
    k = train_reps.shape[1]
    snr_values = []
    query_accuracies = []

    if task_name == "toxicity":
        # For toxicity, compute SNR per unsafe/safe split
        return [], []

    if task_name == "counterfact":
        n_ref = len(ref_data)
        for j in range(n_ref):
            ref_sample = ref_data[j]
            fi = [
                i for i in range(n_train)
                if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                and pilot_data[i]["true_entity"] == ref_sample["true_entity"]
            ]
            if not fi or len(fi) < 2:
                continue
            # Signal: mean of relevant training reps
            signal_reps = train_reps[fi]
            noise_reps = train_reps[[i for i in range(n_train) if i not in fi]]
            if noise_reps.shape[0] < 2:
                continue
            mu_signal = signal_reps.mean(axis=0)
            # SNR_out = sqrt(mu^T Sigma^{-1} mu)
            snr = float(np.sqrt(np.abs(mu_signal @ cov_inv @ mu_signal)))
            snr_values.append(snr)

    elif task_name == "ftrace":
        train_facts_sets = []
        for i in range(n_train):
            facts_raw = pilot_data[i].get("facts", [])
            if isinstance(facts_raw, str):
                facts_raw = [f.strip() for f in facts_raw.split(",") if f.strip()]
            elif isinstance(facts_raw, list):
                flat = []
                for f in facts_raw:
                    if isinstance(f, str):
                        flat.extend([x.strip() for x in f.split(",") if x.strip()])
                facts_raw = flat
            train_facts_sets.append(set(facts_raw))

        n_ref = len(ref_data)
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
            if not fi or len(fi) < 2:
                continue
            signal_reps = train_reps[fi]
            mu_signal = signal_reps.mean(axis=0)
            snr = float(np.sqrt(np.abs(mu_signal @ cov_inv @ mu_signal)))
            snr_values.append(snr)

    return snr_values, query_accuracies


# ── Data loading ────────────────────────────────────────────────────────
def load_task_data():
    report_progress("loading_data", "Loading DATE-LM datasets", 0.05)
    tasks = {}
    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {"train": tox["train"], "ref": tox["ref"]}
    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {"train": cf["train"], "ref": cf["ref"]}
    ft = load_dataset("DataAttributionEval/ftrace", "Pythia-1b")
    tasks["ftrace"] = {"train": ft["train"], "ref": ft["ref"]}
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
    return train_data.select(pilot_idx), pilot_idx


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 70)
    print(f"P5: PCA-Reduced Whitened Attribution -- PILOT MODE (H7-revised)")
    print(f"Model: {MODEL_NAME}, Hidden dim d={HIDDEN_DIM}")
    print(f"PCA dims: {PCA_DIMS}")
    print(f"Whitening methods: standard_repsim (M=I), pca_whitened (Ledoit-Wolf),")
    print(f"                   ridge_cv_whitened (5-fold CV lambda)")
    print(f"Tasks: toxicity, counterfact, ftrace")
    print(f"N_train (pilot): {PILOT_N_TRAIN}")
    print("=" * 70)

    # ── Load cached representations ─────────────────────────────────────
    report_progress("loading_cache", "Loading cached representations", 0.10)
    assert os.path.exists(CACHE_FILE), f"Cache file not found: {CACHE_FILE}"
    cache = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    print(f"Loaded cache keys: {list(cache.keys())}")

    task_reps = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        train_reps = cache[f"repsim_{task_name}_train"].numpy()
        ref_reps = cache[f"repsim_{task_name}_ref"].numpy()
        pilot_idx = cache[f"repsim_{task_name}_pilot_idx"]
        if isinstance(pilot_idx, torch.Tensor):
            pilot_idx = pilot_idx.tolist()
        task_reps[task_name] = {
            "train": train_reps, "ref": ref_reps, "pilot_idx": pilot_idx
        }
        print(f"  [{task_name}] train={train_reps.shape}, ref={ref_reps.shape}")

    # ── Load task data for evaluation labels ────────────────────────────
    tasks_data = load_task_data()

    pilot_datasets = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        pilot_data, pilot_idx = create_pilot_subset(task_name, tasks_data[task_name]["train"])
        cached_idx = task_reps[task_name]["pilot_idx"]
        assert pilot_idx == cached_idx, \
            f"Pilot index mismatch for {task_name}: {pilot_idx[:5]} vs {cached_idx[:5]}"
        pilot_datasets[task_name] = {
            "pilot_data": pilot_data,
            "ref_data": tasks_data[task_name]["ref"],
        }
    print("Pilot subsets verified: indices match cache.")

    # ── Step 1: Standard RepSim baselines (M=I, full dim) ──────────────
    report_progress("standard_repsim", "Computing standard RepSim baselines", 0.15)
    print("\n" + "=" * 70)
    print("STEP 1: Standard RepSim (M=I) Baselines -- Full Dimension")
    print("=" * 70)

    standard_results = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        train_reps = task_reps[task_name]["train"]
        ref_reps = task_reps[task_name]["ref"]
        sim = standard_repsim(train_reps, ref_reps)
        metrics = evaluate_task(
            sim, pilot_datasets[task_name]["pilot_data"],
            pilot_datasets[task_name]["ref_data"], task_name,
            len(pilot_datasets[task_name]["pilot_data"])
        )
        standard_results[task_name] = metrics
        primary = metrics.get("AUPRC", metrics.get("Recall@50", 0))
        print(f"  [{task_name}] Standard RepSim: primary={primary:.4f}, tau={metrics['kendall_tau']:.4f}")

    # ── Step 2: PCA-reduced standard RepSim (for comparison) ────────────
    report_progress("pca_standard", "Computing PCA-reduced standard RepSim", 0.25)
    print("\n" + "=" * 70)
    print("STEP 2: PCA-Reduced Standard RepSim (M=I in PCA space)")
    print("=" * 70)

    pca_standard_results = {}
    pca_info_all = {}

    for k_idx, k in enumerate(PCA_DIMS):
        print(f"\n--- PCA dim k={k} ---")
        pca_standard_results[k] = {}
        pca_info_all[k] = {}

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            train_reps = task_reps[task_name]["train"]
            ref_reps = task_reps[task_name]["ref"]

            train_reduced, ref_reduced, pca_info, _ = pca_reduce(train_reps, ref_reps, k)
            pca_info_all[k][task_name] = pca_info

            sim = standard_repsim(train_reduced, ref_reduced)
            metrics = evaluate_task(
                sim, pilot_datasets[task_name]["pilot_data"],
                pilot_datasets[task_name]["ref_data"], task_name,
                len(pilot_datasets[task_name]["pilot_data"])
            )
            pca_standard_results[k][task_name] = metrics
            primary = metrics.get("AUPRC", metrics.get("Recall@50", 0))
            print(f"  [{task_name}] k={k}: primary={primary:.4f}, tau={metrics['kendall_tau']:.4f}")

    # ── Step 3: PCA-Whitened RepSim (Ledoit-Wolf) ───────────────────────
    report_progress("pca_whitened_lw", "Computing PCA-whitened RepSim (Ledoit-Wolf)", 0.40)
    print("\n" + "=" * 70)
    print("STEP 3: PCA-Whitened RepSim (Ledoit-Wolf, M=Sigma_PCA^{-1})")
    print("=" * 70)

    whitened_lw_results = {}
    whitened_lw_diag = {}

    for k_idx, k in enumerate(PCA_DIMS):
        print(f"\n--- PCA dim k={k} (N/k = {PILOT_N_TRAIN/k:.1f}) ---")
        whitened_lw_results[k] = {}
        whitened_lw_diag[k] = {}

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            train_reps = task_reps[task_name]["train"]
            ref_reps = task_reps[task_name]["ref"]

            train_reduced, ref_reduced, pca_info, _ = pca_reduce(train_reps, ref_reps, k)
            effective_k = pca_info["effective_k"]

            if effective_k == HIDDEN_DIM:
                # Full dim -- skip whitening (we know it fails from pilot)
                whitened_lw_results[k][task_name] = {
                    "skipped": True,
                    "reason": f"k={k} fell back to full dim (d={HIDDEN_DIM}); whitening known to fail"
                }
                whitened_lw_diag[k][task_name] = {"skipped": True}
                print(f"  [{task_name}] k={k}: SKIPPED (fell back to full dim)")
                continue

            sim, diag = whitened_repsim(train_reduced, ref_reduced, "ledoit_wolf")
            metrics = evaluate_task(
                sim, pilot_datasets[task_name]["pilot_data"],
                pilot_datasets[task_name]["ref_data"], task_name,
                len(pilot_datasets[task_name]["pilot_data"])
            )
            whitened_lw_results[k][task_name] = metrics
            whitened_lw_diag[k][task_name] = diag
            primary = metrics.get("AUPRC", metrics.get("Recall@50", 0))
            print(f"  [{task_name}] k={k}: primary={primary:.4f}, tau={metrics['kendall_tau']:.4f}, "
                  f"cond={diag['condition_number']:.0f}, shrinkage={diag['shrinkage']:.4f}")

    # ── Step 4: Ridge-CV Whitened RepSim ────────────────────────────────
    report_progress("ridge_cv_whitened", "Computing ridge-CV whitened RepSim", 0.60)
    print("\n" + "=" * 70)
    print("STEP 4: Ridge-CV Whitened RepSim (lambda via 5-fold CV)")
    print("=" * 70)

    ridge_cv_results = {}
    ridge_cv_diag = {}

    for k_idx, k in enumerate(PCA_DIMS):
        print(f"\n--- PCA dim k={k} (N/k = {PILOT_N_TRAIN/k:.1f}) ---")
        ridge_cv_results[k] = {}
        ridge_cv_diag[k] = {}

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            train_reps = task_reps[task_name]["train"]
            ref_reps = task_reps[task_name]["ref"]

            train_reduced, ref_reduced, pca_info, _ = pca_reduce(train_reps, ref_reps, k)
            effective_k = pca_info["effective_k"]

            if effective_k == HIDDEN_DIM:
                ridge_cv_results[k][task_name] = {
                    "skipped": True,
                    "reason": f"k={k} fell back to full dim"
                }
                ridge_cv_diag[k][task_name] = {"skipped": True}
                print(f"  [{task_name}] k={k}: SKIPPED (fell back to full dim)")
                continue

            sim, diag = ridge_cv_whitened_repsim(
                train_reduced, ref_reduced,
                pilot_datasets[task_name]["pilot_data"],
                pilot_datasets[task_name]["ref_data"],
                task_name,
                len(pilot_datasets[task_name]["pilot_data"]),
            )
            metrics = evaluate_task(
                sim, pilot_datasets[task_name]["pilot_data"],
                pilot_datasets[task_name]["ref_data"], task_name,
                len(pilot_datasets[task_name]["pilot_data"])
            )
            ridge_cv_results[k][task_name] = metrics
            ridge_cv_diag[k][task_name] = diag
            primary = metrics.get("AUPRC", metrics.get("Recall@50", 0))
            best_lam = diag.get("selected_lambda", "?")
            print(f"  [{task_name}] k={k}: primary={primary:.4f}, tau={metrics['kendall_tau']:.4f}, "
                  f"best_lambda={best_lam}")

    # ── Step 5: SNR Analysis at best k ──────────────────────────────────
    report_progress("snr_analysis", "Computing per-query SNR analysis", 0.80)
    print("\n" + "=" * 70)
    print("STEP 5: Per-Query SNR Analysis")
    print("=" * 70)

    snr_analysis = {}
    # Find best k for each task (highest primary metric among whitened variants)
    for task_name in ["counterfact", "ftrace"]:
        best_k = None
        best_primary = -1
        for k in PCA_DIMS:
            res = whitened_lw_results[k].get(task_name, {})
            if res.get("skipped"):
                continue
            primary = res.get("Recall@50", res.get("AUPRC", 0))
            if primary > best_primary:
                best_primary = primary
                best_k = k

        if best_k is None:
            snr_analysis[task_name] = {"note": "No valid whitened result"}
            continue

        # Recompute PCA + whitening to get cov_inv
        train_reps = task_reps[task_name]["train"]
        ref_reps = task_reps[task_name]["ref"]
        train_reduced, ref_reduced, _, _ = pca_reduce(train_reps, ref_reps, best_k)

        # Fit covariance and invert
        lw = LedoitWolf().fit(train_reduced)
        cov_inv = np.linalg.inv(lw.covariance_)

        snr_vals, _ = compute_snr(
            train_reduced,
            pilot_datasets[task_name]["pilot_data"],
            pilot_datasets[task_name]["ref_data"],
            task_name, len(pilot_datasets[task_name]["pilot_data"]),
            cov_inv
        )

        # Get per-query accuracy for whitened method
        sim_wht, _ = whitened_repsim(train_reduced, ref_reduced, "ledoit_wolf")
        acc_wht = get_per_query_accuracy(
            sim_wht, pilot_datasets[task_name]["pilot_data"],
            pilot_datasets[task_name]["ref_data"],
            task_name, len(pilot_datasets[task_name]["pilot_data"])
        )

        # Also standard accuracy for comparison
        sim_std = standard_repsim(train_reduced, ref_reduced)
        acc_std = get_per_query_accuracy(
            sim_std, pilot_datasets[task_name]["pilot_data"],
            pilot_datasets[task_name]["ref_data"],
            task_name, len(pilot_datasets[task_name]["pilot_data"])
        )

        # Correlation between SNR and accuracy
        min_len = min(len(snr_vals), len(acc_wht), len(acc_std))
        if min_len >= 5:
            snr_arr = np.array(snr_vals[:min_len])
            acc_wht_arr = acc_wht[:min_len]
            acc_std_arr = acc_std[:min_len]

            corr_wht, _ = spearmanr(snr_arr, acc_wht_arr) if np.std(acc_wht_arr) > 1e-12 else (0, 1)
            corr_std, _ = spearmanr(snr_arr, acc_std_arr) if np.std(acc_std_arr) > 1e-12 else (0, 1)

            snr_analysis[task_name] = {
                "best_k": best_k,
                "n_queries": min_len,
                "mean_snr": round(float(np.mean(snr_arr)), 4),
                "median_snr": round(float(np.median(snr_arr)), 4),
                "std_snr": round(float(np.std(snr_arr)), 4),
                "snr_accuracy_corr_whitened": round(float(corr_wht) if not np.isnan(corr_wht) else 0, 4),
                "snr_accuracy_corr_standard": round(float(corr_std) if not np.isnan(corr_std) else 0, 4),
                "mean_acc_whitened": round(float(np.mean(acc_wht_arr)), 4),
                "mean_acc_standard": round(float(np.mean(acc_std_arr)), 4),
                "snr_percentiles": {
                    "p25": round(float(np.percentile(snr_arr, 25)), 4),
                    "p50": round(float(np.percentile(snr_arr, 50)), 4),
                    "p75": round(float(np.percentile(snr_arr, 75)), 4),
                },
                # Save sample data for scatter plot
                "snr_values_sample": [round(float(v), 4) for v in snr_arr[:20]],
                "acc_whitened_sample": [round(float(v), 4) for v in acc_wht_arr[:20]],
                "acc_standard_sample": [round(float(v), 4) for v in acc_std_arr[:20]],
            }
            print(f"  [{task_name}] best_k={best_k}: SNR mean={np.mean(snr_arr):.2f}, "
                  f"corr(SNR,acc_wht)={corr_wht:.3f}, corr(SNR,acc_std)={corr_std:.3f}")
        else:
            snr_analysis[task_name] = {
                "best_k": best_k,
                "note": f"Too few queries ({min_len}) for SNR analysis",
            }

    # ── Compile comparison table ────────────────────────────────────────
    report_progress("compiling", "Compiling comparison table", 0.90)
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: Standard vs PCA-Whitened vs Ridge-CV")
    print("=" * 70)

    comparison_table = {}
    for k in PCA_DIMS:
        n_over_k = round(PILOT_N_TRAIN / k, 1)
        row = {"k": k, "N_over_k": n_over_k, "tasks": {}}

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            std_m = pca_standard_results[k].get(task_name, {})
            wht_m = whitened_lw_results[k].get(task_name, {})
            rdg_m = ridge_cv_results[k].get(task_name, {})

            metric_key = "AUPRC" if task_name == "toxicity" else "Recall@50"
            std_val = std_m.get(metric_key, 0)

            if wht_m.get("skipped"):
                wht_val = None
                rdg_val = None
                gain_wht = None
                gain_rdg = None
            else:
                wht_val = wht_m.get(metric_key, 0)
                rdg_val = rdg_m.get(metric_key, 0) if not rdg_m.get("skipped") else None
                gain_wht = round((wht_val - std_val) * 100, 2)
                gain_rdg = round((rdg_val - std_val) * 100, 2) if rdg_val is not None else None

            full_std_val = standard_results[task_name].get(metric_key, 0)
            gain_vs_full = round((std_val - full_std_val) * 100, 2) if std_val else None

            row["tasks"][task_name] = {
                "metric": metric_key,
                "standard_pca": round(std_val, 6) if std_val else None,
                "whitened_lw": round(wht_val, 6) if wht_val is not None else None,
                "ridge_cv": round(rdg_val, 6) if rdg_val is not None else None,
                "full_standard": round(full_std_val, 6),
                "gain_wht_vs_std_pp": gain_wht,
                "gain_rdg_vs_std_pp": gain_rdg,
                "gain_pca_vs_full_pp": gain_vs_full,
            }

            print(f"  k={k:4d} [{task_name:11s}] std={std_val:.4f} "
                  f"wht={'SKIP' if wht_val is None else f'{wht_val:.4f}'} "
                  f"rdg={'SKIP' if rdg_val is None else f'{rdg_val:.4f}'} "
                  f"gain_wht={'N/A' if gain_wht is None else f'{gain_wht:+.2f}pp'} "
                  f"gain_rdg={'N/A' if gain_rdg is None else f'{gain_rdg:+.2f}pp'}")

        comparison_table[str(k)] = row

    # ── Pass criteria evaluation ────────────────────────────────────────
    # Check: PCA-whitened at k=64 does not degrade RepSim by > 5pp
    k64_pass = True
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        wht_m = whitened_lw_results.get(64, {}).get(task_name, {})
        std_m = pca_standard_results.get(64, {}).get(task_name, {})
        if wht_m.get("skipped"):
            continue
        metric_key = "AUPRC" if task_name == "toxicity" else "Recall@50"
        wht_val = wht_m.get(metric_key, 0)
        std_val = std_m.get(metric_key, 0)
        drop = (std_val - wht_val) * 100
        if drop > 5.0:
            k64_pass = False

    # Check: positive SNR-accuracy correlation maintained
    snr_positive = any(
        snr_analysis.get(t, {}).get("snr_accuracy_corr_whitened", 0) > 0
        for t in ["counterfact", "ftrace"]
    )

    # Check: any k outperforms standard by >= 3pp
    any_improvement = False
    best_improvement = {"task": None, "k": None, "gain_pp": 0}
    for k in PCA_DIMS:
        for task_name in ["toxicity", "counterfact", "ftrace"]:
            for method_name, method_results in [("lw", whitened_lw_results), ("ridge", ridge_cv_results)]:
                res = method_results.get(k, {}).get(task_name, {})
                if res.get("skipped"):
                    continue
                metric_key = "AUPRC" if task_name == "toxicity" else "Recall@50"
                wht_val = res.get(metric_key, 0)
                std_val = standard_results[task_name].get(metric_key, 0)
                gain = (wht_val - std_val) * 100
                if gain > best_improvement["gain_pp"]:
                    best_improvement = {"task": task_name, "k": k, "gain_pp": round(gain, 2),
                                        "method": method_name}
                if gain >= 3.0:
                    any_improvement = True

    pass_criteria = {
        "k64_no_degradation_gt_5pp": k64_pass,
        "positive_snr_correlation": snr_positive,
        "any_method_improves_3pp": any_improvement,
        "best_improvement": best_improvement,
    }

    total_time = time.time() - t_start

    # ── Compile final results ───────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": HIDDEN_DIM,
        "pca_dims": PCA_DIMS,
        "pilot_n_train": PILOT_N_TRAIN,
        "seed": SEED,
        "mode": "pilot",
        "standard_results_full_dim": standard_results,
        "pca_standard_results": {str(k): v for k, v in pca_standard_results.items()},
        "whitened_lw_results": {str(k): v for k, v in whitened_lw_results.items()},
        "whitened_lw_diagnostics": {str(k): v for k, v in whitened_lw_diag.items()},
        "ridge_cv_results": {str(k): v for k, v in ridge_cv_results.items()},
        "ridge_cv_diagnostics": {str(k): v for k, v in ridge_cv_diag.items()},
        "pca_info": {str(k): v for k, v in pca_info_all.items()},
        "comparison_table": comparison_table,
        "snr_analysis": snr_analysis,
        "pass_criteria": pass_criteria,
        "decision_gate": {
            "condition": "PCA-whitened at any k outperforms standard RepSim by >= 3pp on any task",
            "result": any_improvement,
            "if_true": "Matched filter theory rescued; framework is prescriptive",
            "if_false": "Honest negative; framework is taxonomic/organizational only",
        },
        "pilot_limitations": [
            f"N={PILOT_N_TRAIN} severely limits covariance estimation quality; "
            f"N/k ranges from {PILOT_N_TRAIN/PCA_DIMS[0]:.1f} to {PILOT_N_TRAIN/PCA_DIMS[-1]:.1f}",
            f"Only k<={PILOT_N_TRAIN-1} can be estimated at all; larger k fall back to full dim",
            "Ridge CV with small ref sets may overfit to CV folds",
            "SNR analysis limited by small number of queries",
        ],
        "total_runtime_sec": round(total_time, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # Save to full results dir
    out_path = os.path.join(FULL_DIR, f"{TASK_ID}.json")
    try:
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nResults saved to {out_path}")
    except OSError as e:
        print(f"\n[CRITICAL] Cannot save results: {e}")
        print("===RESULT_JSON_START===")
        print(json.dumps(final, indent=2))
        print("===RESULT_JSON_END===")

    # ── Pilot summary (machine-readable) ────────────────────────────────
    pilot_summary = {
        "overall_recommendation": "GO" if any_improvement else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if k64_pass else "GO_WITH_CAVEATS",
            "confidence": 0.65 if k64_pass and snr_positive else 0.40,
            "supported_hypotheses": ["H7_revised_partial"] if k64_pass else [],
            "failed_assumptions": [] if k64_pass else ["H7_revised_degradation"],
            "key_metrics": {
                "k64_no_5pp_degradation": k64_pass,
                "snr_positive_correlation": snr_positive,
                "any_3pp_improvement": any_improvement,
                "best_improvement": best_improvement,
            },
            "notes": (
                f"PCA-whitened attribution at pilot scale (N={PILOT_N_TRAIN}). "
                f"k64 pass (<5pp degradation): {k64_pass}. "
                f"SNR-accuracy positive correlation: {snr_positive}. "
                f"Best improvement: {best_improvement}. "
                f"Full-scale needed for definitive H7-revised test (N/k >> 40 at k=64)."
            ),
        }],
    }
    _safe_write(
        os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.json"),
        json.dumps(pilot_summary, indent=2),
    )

    # ── Pilot summary (markdown) ────────────────────────────────────────
    md_lines = [
        f"# P5: PCA-Reduced Whitened Attribution -- Pilot Summary (H7-revised)",
        f"",
        f"## Configuration",
        f"- Model: {MODEL_NAME} (d={HIDDEN_DIM})",
        f"- N_train: {PILOT_N_TRAIN} (pilot)",
        f"- PCA dims: {PCA_DIMS}",
        f"- Methods: standard_repsim (M=I), pca_whitened (Ledoit-Wolf), ridge_cv (5-fold CV)",
        f"",
        f"## Decision Gate",
        f"- **Condition**: PCA-whitened at any k outperforms standard RepSim by >= 3pp on any task",
        f"- **Result**: {'PASS' if any_improvement else 'FAIL (at pilot scale)'}",
        f"- **Best improvement**: {best_improvement}",
        f"",
        f"## Pass Criteria",
        f"- k=64 no >5pp degradation: **{'PASS' if k64_pass else 'FAIL'}**",
        f"- Positive SNR-accuracy correlation: **{'PASS' if snr_positive else 'FAIL'}**",
        f"- Any method +3pp improvement: **{'PASS' if any_improvement else 'FAIL'}**",
        f"",
        f"## Comparison Table (Standard vs PCA-Whitened vs Ridge-CV)",
        f"",
    ]

    for task_name in ["counterfact", "toxicity", "ftrace"]:
        metric_key = "AUPRC" if task_name == "toxicity" else "Recall@50"
        full_val = standard_results[task_name].get(metric_key, 0)
        md_lines += [
            f"### {task_name} ({metric_key}, full_dim_std={full_val:.4f})",
            f"| k | N/k | Standard PCA | Whitened (LW) | Ridge CV | Gain(LW) | Gain(Ridge) |",
            f"|---|-----|-------------|---------------|----------|----------|-------------|",
        ]
        for k in PCA_DIMS:
            ct = comparison_table[str(k)]["tasks"].get(task_name, {})
            std_v = ct.get("standard_pca")
            wht_v = ct.get("whitened_lw")
            rdg_v = ct.get("ridge_cv")
            g_w = ct.get("gain_wht_vs_std_pp")
            g_r = ct.get("gain_rdg_vs_std_pp")
            md_lines.append(
                f"| {k} | {PILOT_N_TRAIN/k:.1f} | "
                f"{'SKIP' if std_v is None else f'{std_v:.4f}'} | "
                f"{'SKIP' if wht_v is None else f'{wht_v:.4f}'} | "
                f"{'SKIP' if rdg_v is None else f'{rdg_v:.4f}'} | "
                f"{'N/A' if g_w is None else f'{g_w:+.2f}pp'} | "
                f"{'N/A' if g_r is None else f'{g_r:+.2f}pp'} |"
            )
        md_lines.append("")

    # SNR section
    md_lines += [
        f"## SNR Analysis",
        f"",
    ]
    for task_name in ["counterfact", "ftrace"]:
        sa = snr_analysis.get(task_name, {})
        if "mean_snr" in sa:
            md_lines += [
                f"### {task_name} (best_k={sa['best_k']})",
                f"- Mean SNR: {sa['mean_snr']:.4f}",
                f"- Corr(SNR, acc_whitened): {sa['snr_accuracy_corr_whitened']:.4f}",
                f"- Corr(SNR, acc_standard): {sa['snr_accuracy_corr_standard']:.4f}",
                f"- n_queries: {sa['n_queries']}",
                f"",
            ]
        else:
            md_lines.append(f"### {task_name}: {sa.get('note', 'N/A')}\n")

    md_lines += [
        f"## Pilot Limitations",
    ]
    for lim in final["pilot_limitations"]:
        md_lines.append(f"- {lim}")
    md_lines += [
        f"",
        f"## Runtime: {total_time:.1f}s",
    ]

    _safe_write(
        os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.md"),
        "\n".join(md_lines),
    )

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: PCA-Whitened Attribution (Pilot)")
    print("=" * 70)
    print(f"Pass criteria:")
    for k2, v in pass_criteria.items():
        print(f"  {k2}: {v}")
    print(f"Decision gate: {'PASS' if any_improvement else 'FAIL (at pilot scale)'}")
    print(f"Best improvement: {best_improvement}")
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 70)

    mark_done(
        "success",
        f"PCA-whitened attribution pilot done in {total_time:.0f}s. "
        f"k64_pass={k64_pass}, snr_positive={snr_positive}, "
        f"any_3pp={any_improvement}, best={best_improvement}"
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
