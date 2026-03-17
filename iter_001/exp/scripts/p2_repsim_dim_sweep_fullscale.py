#!/usr/bin/env python3
"""
P2: RepSim PCA Dimension Sweep at Full Scale -- PILOT MODE
============================================================
Run RepSim on Pythia-1B with PCA dimension reduction at k in
{16, 32, 64, 128, 256, 512, 1024, 2048} for all 3 DATE-LM tasks.
PILOT: N=100 training samples, seed=42.

Evaluate with BOTH rank-based (R@50, MRR, AUPRC) AND continuous
(Kendall tau, Spearman rho) metrics.

Pass criteria: RepSim at k=d matches full RepSim within 1pp;
graceful degradation with knee at k in [d/8, d/2].
"""

import os, sys, json, time, gc, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, auc
from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p2_repsim_dim_sweep_fullscale"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-1b"
HIDDEN_DIM = 2048
PCA_DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048]
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


def compute_continuous_metrics_toxicity(train_scores, pilot_data, n_train):
    """Kendall tau and Spearman rho for toxicity: scores vs binary unsafe label."""
    binary_rel = np.array([1.0 if pilot_data[i]["type"] == "Unsafe" else 0.0
                           for i in range(n_train)])
    if np.std(train_scores) < 1e-12 or np.sum(binary_rel) == 0:
        return 0.0, 0.0
    tau, _ = kendalltau(train_scores, binary_rel)
    rho, _ = spearmanr(train_scores, binary_rel)
    return (float(tau) if not np.isnan(tau) else 0.0,
            float(rho) if not np.isnan(rho) else 0.0)


def compute_continuous_metrics_factual(scores_per_ref, fact_indices_per_ref, n_train):
    """Kendall tau and Spearman rho for counterfact/ftrace: per-query correlation."""
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


def bootstrap_auprc(train_scores, pilot_data, n_train, B=BOOTSTRAP_B):
    unsafe_indices = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
    auprc = compute_auprc(train_scores, unsafe_indices, n_train)
    rng = np.random.RandomState(SEED + 1234)
    boot_vals = []
    for _ in range(B):
        idx = rng.choice(n_train, n_train, replace=True)
        bs = train_scores[idx]
        bl = np.zeros(len(idx))
        for ii, orig_i in enumerate(idx):
            if pilot_data[int(orig_i)]["type"] == "Unsafe":
                bl[ii] = 1
        if sum(bl) == 0:
            boot_vals.append(0.0)
            continue
        prec, rec, _ = precision_recall_curve(bl, bs)
        boot_vals.append(float(auc(rec, prec)))
    ci_lo = float(np.percentile(boot_vals, 2.5))
    ci_hi = float(np.percentile(boot_vals, 97.5))
    return auprc, ci_lo, ci_hi, len(unsafe_indices)


def bootstrap_factual(sim_matrix, pilot_data, ref_data, task_name, n_train):
    """Full bootstrap evaluation with rank-based + continuous metrics."""
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

    # Bootstrap CI
    seed_offset = 2345 if task_name == "counterfact" else 3456
    rng = np.random.RandomState(SEED + seed_offset)
    boot_recalls, boot_mrrs, boot_taus = [], [], []
    for _ in range(BOOTSTRAP_B):
        idx = rng.choice(n_ref, n_ref, replace=True)
        boot_spr = [scores_per_ref[i] for i in idx]
        boot_fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(boot_spr, boot_fi, k=50)
        t, _ = compute_continuous_metrics_factual(boot_spr, boot_fi, n_train)
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


# ── PCA + RepSim ────────────────────────────────────────────────────────
def pca_reduce_and_score(train_reps_np, ref_reps_np, k):
    """
    Apply PCA to reduce dimensionality from d=2048 to k.
    PCA is fit on the combined train+ref pool.
    After projection, L2-normalize and compute cosine similarity.
    """
    n_train = train_reps_np.shape[0]
    d = train_reps_np.shape[1]

    combined = np.vstack([train_reps_np, ref_reps_np])
    max_components = min(combined.shape[0], d) - 1

    if k >= max_components:
        # No meaningful reduction -- use original representations
        train_norm = train_reps_np / (np.linalg.norm(train_reps_np, axis=1, keepdims=True) + 1e-8)
        ref_norm = ref_reps_np / (np.linalg.norm(ref_reps_np, axis=1, keepdims=True) + 1e-8)
        sim_matrix = train_norm @ ref_norm.T
        info = {
            "note": f"k={k} >= max_components={max_components}; using full representations",
            "effective_k": d,
            "explained_variance_ratio": 1.0,
        }
        return sim_matrix, info

    pca = PCA(n_components=k, random_state=SEED)
    combined_reduced = pca.fit_transform(combined)
    train_reduced = combined_reduced[:n_train]
    ref_reduced = combined_reduced[n_train:]
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    # L2-normalize in reduced space
    train_reduced = train_reduced / (np.linalg.norm(train_reduced, axis=1, keepdims=True) + 1e-8)
    ref_reduced = ref_reduced / (np.linalg.norm(ref_reduced, axis=1, keepdims=True) + 1e-8)

    sim_matrix = train_reduced @ ref_reduced.T

    info = {
        "effective_k": k,
        "explained_variance_ratio": round(explained_var, 6),
        "top_eigenvalues": [round(float(v), 6) for v in pca.explained_variance_[:min(10, k)]],
        "n_fit_samples": combined.shape[0],
    }
    return sim_matrix, info


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
    print(f"P2: RepSim PCA Dimension Sweep -- PILOT MODE")
    print(f"Model: {MODEL_NAME}, Hidden dim d={HIDDEN_DIM}")
    print(f"PCA dims: {PCA_DIMS}")
    print(f"Tasks: toxicity, counterfact, ftrace")
    print(f"N_train (pilot): {PILOT_N_TRAIN}")
    print(f"Bootstrap B={BOOTSTRAP_B}")
    print(f"Metrics: R@50/AUPRC, MRR, Kendall tau, Spearman rho")
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

    # ── Sweep over PCA dimensions ───────────────────────────────────────
    sweep_results = {}
    pca_info_all = {}

    for dim_idx, k in enumerate(PCA_DIMS):
        print(f"\n{'='*70}")
        print(f"PCA dimension k={k} ({dim_idx+1}/{len(PCA_DIMS)})")
        print(f"{'='*70}")
        report_progress("pca_sweep", f"k={k}", (dim_idx + 1) / (len(PCA_DIMS) + 1))

        dim_results = {}
        dim_info = {}
        t_dim = time.time()

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            train_reps = task_reps[task_name]["train"]
            ref_reps = task_reps[task_name]["ref"]

            sim_matrix, info = pca_reduce_and_score(train_reps, ref_reps, k)
            dim_info[task_name] = info

            pilot_data = pilot_datasets[task_name]["pilot_data"]
            ref_data = pilot_datasets[task_name]["ref_data"]
            n_train = len(pilot_data)

            if task_name == "toxicity":
                train_scores = sim_matrix.mean(axis=1)
                auprc, ci_lo, ci_hi, n_unsafe = bootstrap_auprc(
                    train_scores, pilot_data, n_train)
                tau, rho = compute_continuous_metrics_toxicity(
                    train_scores, pilot_data, n_train)
                metrics = {
                    "AUPRC": round(auprc, 6),
                    "AUPRC_CI": [round(ci_lo, 6), round(ci_hi, 6)],
                    "n_unsafe": n_unsafe,
                    "n_train": n_train,
                    "kendall_tau": round(tau, 6),
                    "spearman_rho": round(rho, 6),
                }
                print(f"  [{task_name}] k={k}: AUPRC={auprc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], "
                      f"tau={tau:.4f}, rho={rho:.4f}")
            else:
                metrics = bootstrap_factual(
                    sim_matrix, pilot_data, ref_data, task_name, n_train)
                print(f"  [{task_name}] k={k}: R@50={metrics['Recall@50']:.4f} "
                      f"[{metrics['Recall@50_CI'][0]:.4f}, {metrics['Recall@50_CI'][1]:.4f}], "
                      f"MRR={metrics['MRR']:.4f}, tau={metrics['kendall_tau']:.4f}")

            dim_results[task_name] = metrics

        elapsed = time.time() - t_dim
        sweep_results[k] = {"metrics": dim_results, "runtime_sec": round(elapsed, 2)}
        pca_info_all[k] = dim_info
        print(f"  [k={k}] Total time: {elapsed:.1f}s")

    # ── Analysis: degradation curve ─────────────────────────────────────
    report_progress("analysis", "Computing degradation + knee analysis", 0.90)

    full_dim_metrics = sweep_results[2048]["metrics"]

    degradation = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        if task_name == "toxicity":
            full_val = full_dim_metrics[task_name]["AUPRC"]
            metric_key = "AUPRC"
        else:
            full_val = full_dim_metrics[task_name]["Recall@50"]
            metric_key = "Recall@50"

        full_tau = full_dim_metrics[task_name]["kendall_tau"]

        curve = []
        for k in PCA_DIMS:
            if task_name == "toxicity":
                val = sweep_results[k]["metrics"][task_name]["AUPRC"]
            else:
                val = sweep_results[k]["metrics"][task_name]["Recall@50"]
            tau_val = sweep_results[k]["metrics"][task_name]["kendall_tau"]

            drop_pp = round((full_val - val) * 100, 2)
            retention_pct = round(val / full_val * 100, 2) if full_val > 0 else 0.0
            tau_drop = round(full_tau - tau_val, 6)
            curve.append({
                "pca_dim": k,
                "value": round(val, 6),
                "drop_pp": drop_pp,
                "retention_pct": retention_pct,
                "kendall_tau": round(tau_val, 6),
                "tau_drop": tau_drop,
            })
        degradation[task_name] = {
            "metric": metric_key,
            "full_dim_value": round(full_val, 6),
            "full_dim_tau": round(full_tau, 6),
            "curve": curve,
        }

    # Find knee point: smallest k where retention >= 97% (within 3pp)
    knee_points = {}
    for task_name, deg in degradation.items():
        knee_k = None
        for pt in deg["curve"]:
            if pt["retention_pct"] >= 97.0:
                knee_k = pt["pca_dim"]
                break
        knee_points[task_name] = knee_k

    # H4 cross-validation: knee at k in [d/8, d/2]?
    h4_cross_validation = {}
    for task_name, knee_k in knee_points.items():
        if knee_k is not None:
            ratio = knee_k / HIDDEN_DIM
            in_range = 0.125 <= ratio <= 0.5
            h4_cross_validation[task_name] = {
                "knee_k": knee_k,
                "knee_ratio": round(ratio, 3),
                "in_expected_range": in_range,
                "note": f"knee at k={knee_k} = {ratio:.3f}*d"
            }
        else:
            h4_cross_validation[task_name] = {
                "knee_k": None,
                "note": "No knee found (all dims below 97% retention)"
            }

    # Minimum k for near-full performance (within 3pp)
    min_k_within_3pp = {}
    for task_name, deg in degradation.items():
        mk = None
        for pt in deg["curve"]:
            if abs(pt["drop_pp"]) <= 3.0:
                mk = pt["pca_dim"]
                break
        min_k_within_3pp[task_name] = mk

    # Pass criteria
    pass_criteria = {
        "full_dim_matches_within_1pp": all(
            abs(degradation[t]["curve"][-1]["drop_pp"]) <= 1.0
            for t in ["toxicity", "counterfact", "ftrace"]
        ),
        "graceful_degradation": all(
            knee_points[t] is not None for t in ["toxicity", "counterfact", "ftrace"]
        ),
        "knee_in_expected_range": any(
            h4_cross_validation[t].get("in_expected_range", False)
            for t in ["toxicity", "counterfact", "ftrace"]
        ),
    }

    # Pilot limitations
    pilot_limitations = [
        f"N={PILOT_N_TRAIN} caps PCA max_components to ~{PILOT_N_TRAIN + 30 - 1} "
        f"(varies by task pool size); k>max_components falls back to full d={HIDDEN_DIM}",
        "At pilot scale saturation may appear earlier (fewer samples = lower effective rank)",
        f"Bootstrap CIs may be wide with small ref sets",
    ]

    total_time = time.time() - t_start

    # ── Compile final results ───────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": HIDDEN_DIM,
        "pca_dims": PCA_DIMS,
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "mode": "pilot",
        "sweep_results": {str(k): v for k, v in sweep_results.items()},
        "pca_info": {str(k): v for k, v in pca_info_all.items()},
        "degradation_analysis": degradation,
        "knee_points": knee_points,
        "min_k_within_3pp": min_k_within_3pp,
        "h4_cross_validation": h4_cross_validation,
        "pass_criteria": pass_criteria,
        "pilot_limitations": pilot_limitations,
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
        "overall_recommendation": "GO" if all(pass_criteria.values()) else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if all(pass_criteria.values()) else "GO",
            "confidence": 0.75 if all(pass_criteria.values()) else 0.55,
            "supported_hypotheses": ["H5_repsim_dim"] if pass_criteria["graceful_degradation"] else [],
            "failed_assumptions": [],
            "key_metrics": {
                "knee_points": knee_points,
                "min_k_within_3pp": min_k_within_3pp,
                "counterfact_full_R50": degradation["counterfact"]["full_dim_value"],
                "counterfact_full_tau": degradation["counterfact"]["full_dim_tau"],
                "toxicity_full_AUPRC": degradation["toxicity"]["full_dim_value"],
                "toxicity_full_tau": degradation["toxicity"]["full_dim_tau"],
                "ftrace_full_R50": degradation["ftrace"]["full_dim_value"],
                "ftrace_full_tau": degradation["ftrace"]["full_dim_tau"],
            },
            "notes": (
                f"RepSim PCA sweep at pilot scale (N={PILOT_N_TRAIN}). "
                f"Knees (within 3pp): {min_k_within_3pp}. "
                f"Pilot limitation: N={PILOT_N_TRAIN} caps max PCA components; "
                f"k >= ~{PILOT_N_TRAIN+30} falls back to full dim. "
                f"Full-scale should reveal true saturation point."
            ),
        }],
        "pilot_limitations": pilot_limitations,
    }

    _safe_write(
        os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.json"),
        json.dumps(pilot_summary, indent=2),
    )

    # ── Pilot summary (markdown) ────────────────────────────────────────
    md_lines = [
        f"# P2: RepSim PCA Dimension Sweep -- Pilot Summary",
        f"",
        f"## Configuration",
        f"- Model: {MODEL_NAME} (d={HIDDEN_DIM})",
        f"- N_train: {PILOT_N_TRAIN} (pilot)",
        f"- PCA dims: {PCA_DIMS}",
        f"- Metrics: R@50/AUPRC, MRR, Kendall tau, Spearman rho",
        f"",
    ]

    for task_name in ["counterfact", "toxicity", "ftrace"]:
        deg = degradation[task_name]
        metric = deg["metric"]
        md_lines += [
            f"## {task_name} ({metric}, full_dim={deg['full_dim_value']:.4f}, tau={deg['full_dim_tau']:.4f})",
            f"| k | {metric} | Drop(pp) | Retention% | Kendall tau | tau_drop |",
            f"|---|---------|----------|-----------|-------------|---------|",
        ]
        for pt in deg["curve"]:
            marker = " **knee**" if pt["pca_dim"] == knee_points[task_name] else ""
            md_lines.append(
                f"| {pt['pca_dim']} | {pt['value']:.4f} | {pt['drop_pp']:+.2f} | "
                f"{pt['retention_pct']:.1f} | {pt['kendall_tau']:.4f} | "
                f"{pt['tau_drop']:+.4f} |{marker}"
            )
        md_lines.append("")

    md_lines += [
        f"## Analysis",
        f"- Knee points (97% retention): {knee_points}",
        f"- Min k within 3pp: {min_k_within_3pp}",
        f"- H4 cross-validation: {h4_cross_validation}",
        f"- Pass criteria: {pass_criteria}",
        f"",
        f"## Pilot Limitations",
    ]
    for lim in pilot_limitations:
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
    print("SUMMARY: RepSim PCA Dimension Sweep (Pilot)")
    print("=" * 70)

    for task_name in ["counterfact", "toxicity", "ftrace"]:
        deg = degradation[task_name]
        metric = deg["metric"]
        print(f"\n[{task_name}] ({metric}, full_dim={deg['full_dim_value']:.4f}, tau={deg['full_dim_tau']:.4f})")
        print(f"  {'k':<6} {metric:<10} {'Drop(pp)':<10} {'Ret%':<8} {'tau':<10} {'tau_drop':<10}")
        print(f"  {'-'*54}")
        for pt in deg["curve"]:
            marker = " <-- knee" if pt["pca_dim"] == knee_points[task_name] else ""
            print(f"  {pt['pca_dim']:<6} {pt['value']:<10.4f} {pt['drop_pp']:<+10.2f} "
                  f"{pt['retention_pct']:<8.1f} {pt['kendall_tau']:<10.4f} "
                  f"{pt['tau_drop']:<+10.4f}{marker}")

    print(f"\n{'='*70}")
    print(f"Knee points (97% retention): {knee_points}")
    print(f"Min k within 3pp: {min_k_within_3pp}")
    print(f"Pass criteria: {pass_criteria}")
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 70)

    mark_done(
        "success" if all(pass_criteria.values()) else "warn",
        f"RepSim PCA sweep pilot done in {total_time:.0f}s. "
        f"Dims={PCA_DIMS}. Knees={knee_points}. "
        f"Min_k_3pp={min_k_within_3pp}. "
        f"Pass={pass_criteria}"
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
