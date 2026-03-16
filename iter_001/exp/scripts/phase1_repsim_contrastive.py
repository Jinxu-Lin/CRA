#!/usr/bin/env python3
"""
Phase 1: RepSim Contrastive (Cell D of 2x2 factorial)
- RepSim with contrastive scoring (mean-subtracted cosine) on Pythia-1B
- All 3 DATE-LM tasks: toxicity (AUPRC), counterfact (Recall@50+MRR), ftrace (P@K)
- Reuses cached representations from repsim_standard
- Contrastive score: s_C(z_ref, z_train) = cosine(z_ref, z_train) - mean_over_train'[cosine(z_ref, z')]
- PILOT mode: 100 training samples, seed=42
- Bootstrap CI (B=1000)
- H2 asymmetry check: RepSim-C gain over RepSim should be smaller than TRAK-C over TRAK
"""

import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics import precision_recall_curve, auc

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "phase1_factorial_repsim_contrastive"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE1_DIR = os.path.join(RESULTS_DIR, "phase1")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
BOOTSTRAP_B = 1000

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
def compute_auprc(scores, unsafe_indices, n_total):
    """AUPRC for toxicity: higher score = more likely unsafe."""
    labels = np.zeros(n_total)
    labels[list(unsafe_indices)] = 1
    if sum(labels) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50):
    """Recall@K and MRR for counterfact/ftrace."""
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


# ── Data loading (minimal: only need metadata for evaluation) ───────────
def load_all_tasks():
    """Load all 3 DATE-LM tasks (only metadata needed, reps are cached)."""
    report_progress("loading_data", "Loading DATE-LM datasets for metadata", 0.05)
    tasks = {}

    # 1. Toxicity
    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {"train": tox["train"], "ref": tox["ref"]}
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    # 2. Counterfact
    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {"train": cf["train"], "ref": cf["ref"]}
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")

    # 3. Ftrace
    ft = load_dataset("DataAttributionEval/ftrace", "Pythia-1b")
    tasks["ftrace"] = {"train": ft["train"], "ref": ft["ref"]}
    print(f"[ftrace] train={len(ft['train'])}, ref={len(ft['ref'])}")

    return tasks


def create_pilot_subset(task_name, train_data, n_pilot=PILOT_N_TRAIN):
    """Create pilot subset with same random seed as standard variant."""
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


# ── Contrastive scoring ────────────────────────────────────────────────
def compute_contrastive_repsim(train_reps, ref_reps):
    """
    Compute contrastive RepSim scores.
    s_C(z_ref, z_train) = cosine(z_ref, z_train) - mean_over_train'[cosine(z_ref, z')]

    Both train_reps and ref_reps are L2-normalized, so dot product = cosine similarity.

    Returns:
        raw_sim: [n_train, n_ref] raw cosine similarity
        contrastive_sim: [n_train, n_ref] mean-subtracted similarity
        mean_per_ref: [n_ref] mean similarity per reference query
    """
    # Raw cosine similarity: [n_train, n_ref]
    raw_sim = (train_reps @ ref_reps.T).numpy()

    # Mean over all training samples for each reference query: [n_ref]
    mean_per_ref = raw_sim.mean(axis=0)

    # Contrastive: subtract mean (broadcast: [n_train, n_ref] - [1, n_ref])
    contrastive_sim = raw_sim - mean_per_ref[np.newaxis, :]

    return raw_sim, contrastive_sim, mean_per_ref


# ── Evaluation per task ─────────────────────────────────────────────────
def evaluate_toxicity(sim_matrix, pilot_data, ref_data):
    """Toxicity: score each training sample by mean similarity to refs."""
    train_scores = sim_matrix.mean(axis=1)  # [n_train]
    n_train = len(pilot_data)
    unsafe_indices = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]

    auprc = compute_auprc(train_scores, unsafe_indices, n_train)

    # Bootstrap CI
    rng_boot = np.random.RandomState(SEED + 1234)
    boot_vals = []
    for _ in range(BOOTSTRAP_B):
        idx = rng_boot.choice(n_train, n_train, replace=True)
        boot_scores = train_scores[idx]
        boot_labels = np.zeros(len(idx))
        for i, orig_i in enumerate(idx):
            if pilot_data[int(orig_i)]["type"] == "Unsafe":
                boot_labels[i] = 1
        if sum(boot_labels) == 0:
            boot_vals.append(0.0)
            continue
        precision, recall, _ = precision_recall_curve(boot_labels, boot_scores)
        boot_vals.append(float(auc(recall, precision)))
    ci_lo = float(np.percentile(boot_vals, 2.5))
    ci_hi = float(np.percentile(boot_vals, 97.5))

    print(f"[Toxicity] AUPRC={auprc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (unsafe={len(unsafe_indices)}/{n_train})")
    return {
        "AUPRC": round(auprc, 6),
        "CI_lower": round(ci_lo, 6),
        "CI_upper": round(ci_hi, 6),
        "n_unsafe": len(unsafe_indices),
        "n_train": n_train,
        "score_stats": {
            "mean": float(np.mean(train_scores)),
            "std": float(np.std(train_scores)),
            "min": float(np.min(train_scores)),
            "max": float(np.max(train_scores)),
        }
    }


def evaluate_counterfact(sim_matrix, pilot_data, ref_data):
    """Counterfact: for each ref query, rank training samples by score."""
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

    print(f"[Counterfact] Recall@50={recall:.4f} [{recall_ci[0]:.4f},{recall_ci[1]:.4f}], "
          f"MRR={mrr:.4f} [{mrr_ci[0]:.4f},{mrr_ci[1]:.4f}] (refs_with_facts={n_with_facts})")
    return {
        "Recall@50": round(recall, 6),
        "Recall@50_CI": [round(recall_ci[0], 6), round(recall_ci[1], 6)],
        "MRR": round(mrr, 6),
        "MRR_CI": [round(mrr_ci[0], 6), round(mrr_ci[1], 6)],
        "refs_with_facts": n_with_facts,
        "n_ref": n_ref,
        "n_train": n_train,
    }


def evaluate_ftrace(sim_matrix, pilot_data, ref_data):
    """Ftrace: for each ref query, rank training samples by score."""
    n_train = len(pilot_data)
    n_ref = len(ref_data)

    scores_per_ref = []
    fact_indices_per_ref = []

    # Parse facts for training samples
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
        scores_per_ref.append(scores_j)
        fact_indices_per_ref.append(fi)

    recall, mrr = compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50)

    # Bootstrap CI
    rng_boot = np.random.RandomState(SEED + 3456)
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

    print(f"[Ftrace] Recall@50={recall:.4f} [{recall_ci[0]:.4f},{recall_ci[1]:.4f}], "
          f"MRR={mrr:.4f} [{mrr_ci[0]:.4f},{mrr_ci[1]:.4f}] (refs_with_facts={n_with_facts})")
    return {
        "Recall@50": round(recall, 6),
        "Recall@50_CI": [round(recall_ci[0], 6), round(recall_ci[1], 6)],
        "MRR": round(mrr, 6),
        "MRR_CI": [round(mrr_ci[0], 6), round(mrr_ci[1], 6)],
        "refs_with_facts": n_with_facts,
        "n_ref": n_ref,
        "n_train": n_train,
    }


# ── Qualitative inspection ─────────────────────────────────────────────
def inspect_samples(sim_matrix, pilot_data, task_name, n_show=5):
    """Print top-scoring training samples for qualitative inspection."""
    mean_scores = sim_matrix.mean(axis=1)
    top_idx = np.argsort(-mean_scores)[:n_show]
    bot_idx = np.argsort(mean_scores)[:n_show]

    print(f"\n[Qualitative/{task_name}] Top-{n_show} highest-scoring (contrastive):")
    for rank, idx in enumerate(top_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:100]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} score={mean_scores[idx]:.4f} type={label}: {text}...")

    print(f"[Qualitative/{task_name}] Top-{n_show} lowest-scoring (contrastive):")
    for rank, idx in enumerate(bot_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:100]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} score={mean_scores[idx]:.4f} type={label}: {text}...")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs(PHASE1_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 1: RepSim Contrastive (Cell D of 2x2 factorial)")
    print(f"Model: {MODEL_NAME}, Pilot N={PILOT_N_TRAIN}")
    print("Tasks: toxicity, counterfact, ftrace")
    print(f"Bootstrap B={BOOTSTRAP_B}")
    print("Scoring: CONTRASTIVE (mean-subtracted cosine)")
    print("Using cached representations from repsim_standard")
    print("=" * 70)

    # ── Load cached representations ─────────────────────────────────────
    report_progress("loading_cache", "Loading cached representations", 0.05)
    cache_path = os.path.join(CACHE_DIR, "repsim_standard_reps.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}. Run repsim_standard first.")

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    print(f"Loaded cache from {cache_path}")
    print(f"Cache keys: {list(cache.keys())}")

    # ── Load metadata (needed for evaluation) ───────────────────────────
    tasks = load_all_tasks()

    # ── Load RepSim standard results for comparison ─────────────────────
    repsim_std_path = os.path.join(PHASE1_DIR, "repsim_standard.json")
    repsim_std_results = None
    if os.path.exists(repsim_std_path):
        with open(repsim_std_path) as f:
            repsim_std_results = json.load(f)
        print(f"Loaded RepSim standard results for comparison")

    # Also load TRAK contrastive results for H2 asymmetry comparison
    trak_std_path = os.path.join(PHASE1_DIR, "trak_standard.json")
    trak_c_path = os.path.join(PHASE1_DIR, "trak_contrastive.json")
    trak_std_results = None
    trak_c_results = None
    if os.path.exists(trak_std_path):
        with open(trak_std_path) as f:
            trak_std_results = json.load(f)
    if os.path.exists(trak_c_path):
        with open(trak_c_path) as f:
            trak_c_results = json.load(f)

    results_per_task = {}
    contrastive_analysis = {}
    all_timings = {}
    task_names = ["toxicity", "counterfact", "ftrace"]

    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*70}")
        print(f"Task {task_idx+1}/3: {task_name}")
        print(f"{'='*70}")

        report_progress("experiment", f"RepSim contrastive on {task_name}", (task_idx) / 3.0)
        t_task = time.time()

        # Get cached representations
        train_reps_key = f"repsim_{task_name}_train"
        ref_reps_key = f"repsim_{task_name}_ref"
        pilot_idx_key = f"repsim_{task_name}_pilot_idx"

        if train_reps_key not in cache:
            raise KeyError(f"Missing cache key: {train_reps_key}. Available: {list(cache.keys())}")

        train_reps = cache[train_reps_key]  # [n_train, hidden_dim], L2-normalized
        ref_reps = cache[ref_reps_key]      # [n_ref, hidden_dim], L2-normalized
        pilot_idx = cache[pilot_idx_key]
        if isinstance(pilot_idx, torch.Tensor):
            pilot_idx = pilot_idx.tolist()

        print(f"[RepSim-C/{task_name}] Loaded cached reps: train={train_reps.shape}, ref={ref_reps.shape}")

        # Recreate pilot subset with same indices
        pilot_data = tasks[task_name]["train"].select(pilot_idx)
        ref_data = tasks[task_name]["ref"]

        # Compute contrastive RepSim
        print(f"[RepSim-C/{task_name}] Computing contrastive similarity (mean-subtracted)...")
        raw_sim, contrastive_sim, mean_per_ref = compute_contrastive_repsim(train_reps, ref_reps)

        print(f"[RepSim-C/{task_name}] Raw sim stats: mean={raw_sim.mean():.4f}, std={raw_sim.std():.4f}")
        print(f"[RepSim-C/{task_name}] Mean per ref: mean={mean_per_ref.mean():.4f}, std={mean_per_ref.std():.4f}")
        print(f"[RepSim-C/{task_name}] Contrastive sim stats: mean={contrastive_sim.mean():.6f}, "
              f"std={contrastive_sim.std():.4f}, min={contrastive_sim.min():.4f}, max={contrastive_sim.max():.4f}")

        # Record contrastive analysis
        contrastive_analysis[task_name] = {
            "raw_sim_mean": float(np.mean(raw_sim)),
            "raw_sim_std": float(np.std(raw_sim)),
            "contrastive_sim_mean": float(np.mean(contrastive_sim)),
            "contrastive_sim_std": float(np.std(contrastive_sim)),
            "mean_subtraction_magnitude": float(np.mean(np.abs(mean_per_ref))),
            "mean_subtraction_std": float(np.std(mean_per_ref)),
            "mean_per_ref_range": [float(np.min(mean_per_ref)), float(np.max(mean_per_ref))],
        }

        elapsed = time.time() - t_task
        all_timings[task_name] = round(elapsed, 2)

        # Evaluate using contrastive similarity matrix
        if task_name == "toxicity":
            metrics = evaluate_toxicity(contrastive_sim, pilot_data, ref_data)
        elif task_name == "counterfact":
            metrics = evaluate_counterfact(contrastive_sim, pilot_data, ref_data)
        elif task_name == "ftrace":
            metrics = evaluate_ftrace(contrastive_sim, pilot_data, ref_data)

        # Qualitative inspection
        inspect_samples(contrastive_sim, pilot_data, task_name)

        results_per_task[task_name] = {
            "metrics": metrics,
            "runtime_sec": round(elapsed, 2),
            "n_train_pilot": len(pilot_data),
            "n_ref": len(ref_data),
        }

    # ── H2 comparison: RepSim-C gain vs TRAK-C gain ────────────────────
    report_progress("analysis", "Computing H2 asymmetry comparison", 0.90)
    h2_comparison = {}
    for tn in task_names:
        m_c = results_per_task[tn]["metrics"]

        # RepSim standard metric
        repsim_std_metric = 0.0
        if repsim_std_results and tn in repsim_std_results.get("results_per_task", {}):
            std_m = repsim_std_results["results_per_task"][tn]["metrics"]
            repsim_std_metric = std_m.get("AUPRC", std_m.get("Recall@50", 0))

        # RepSim contrastive metric
        repsim_c_metric = m_c.get("AUPRC", m_c.get("Recall@50", 0))
        repsim_gain = (repsim_c_metric - repsim_std_metric) * 100  # in pp

        # TRAK standard -> contrastive gain
        trak_gain = 0.0
        trak_std_metric = 0.0
        trak_c_metric = 0.0
        if trak_std_results and tn in trak_std_results.get("results_per_task", {}):
            trak_m = trak_std_results["results_per_task"][tn]["metrics"]
            trak_std_metric = trak_m.get("AUPRC", trak_m.get("Recall@50", 0))
        if trak_c_results and tn in trak_c_results.get("results_per_task", {}):
            trak_cm = trak_c_results["results_per_task"][tn]["metrics"]
            trak_c_metric = trak_cm.get("AUPRC", trak_cm.get("Recall@50", 0))
        trak_gain = (trak_c_metric - trak_std_metric) * 100

        metric_name = "AUPRC" if tn == "toxicity" else "Recall@50"
        h2_comparison[tn] = {
            "metric": metric_name,
            "repsim_standard": round(repsim_std_metric, 6),
            "repsim_contrastive": round(repsim_c_metric, 6),
            "repsim_gain_pp": round(repsim_gain, 2),
            "trak_standard": round(trak_std_metric, 6),
            "trak_contrastive": round(trak_c_metric, 6),
            "trak_gain_pp": round(trak_gain, 2),
            "h2_asymmetry_check": "PASS" if trak_gain > repsim_gain else "FAIL (rep gain >= param gain)",
        }

    # H2 asymmetry: parameter-space contrastive gain should be larger than rep-space gain
    h2_asymmetry_passes = sum(
        1 for tn in task_names
        if h2_comparison[tn]["trak_gain_pp"] > h2_comparison[tn]["repsim_gain_pp"]
    )
    h2_asymmetry_pass = h2_asymmetry_passes >= 2  # Pass on >= 2/3 tasks

    # ── Validity checks ─────────────────────────────────────────────────
    validity = {}
    for tn, res in results_per_task.items():
        m = res["metrics"]
        checks = {}
        if tn == "toxicity":
            checks["AUPRC_valid"] = 0 <= m["AUPRC"] <= 1
            checks["AUPRC_above_random"] = m["AUPRC"] > m["n_unsafe"] / m["n_train"]
        else:
            checks["Recall@50_valid"] = 0 <= m["Recall@50"] <= 1
            checks["MRR_valid"] = 0 <= m["MRR"] <= 1
        validity[tn] = checks

    all_valid = all(v for checks in validity.values() for v in checks.values())

    total_time = time.time() - t_start
    final = {
        "task_id": TASK_ID,
        "method": "RepSim_contrastive",
        "scoring": "contrastive",
        "contrastive_protocol": "mean_subtraction_over_train",
        "space": "representation",
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "extraction_layer": "last_hidden_state",
        "similarity": "cosine",
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "results_per_task": results_per_task,
        "contrastive_analysis": contrastive_analysis,
        "h2_comparison": h2_comparison,
        "h2_asymmetry_pass": h2_asymmetry_pass,
        "h2_asymmetry_passes_count": f"{h2_asymmetry_passes}/3",
        "timings": all_timings,
        "validity_checks": validity,
        "all_valid": all_valid,
        "total_runtime_sec": round(total_time, 2),
        "gpu": "N/A (cache-only, no GPU needed)",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    out_path = os.path.join(PHASE1_DIR, "repsim_contrastive.json")
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
    print("SUMMARY: RepSim Contrastive (Phase 1, Cell D)")
    print("=" * 70)
    print(f"{'Task':<15}{'Metric':<15}{'Standard':<12}{'Contrastive':<12}{'Gain(pp)':<10}{'TRAK gain':<10}{'H2':<8}")
    print("-" * 82)
    for tn in task_names:
        h2 = h2_comparison[tn]
        print(f"{tn:<15}{h2['metric']:<15}{h2['repsim_standard']:<12.4f}{h2['repsim_contrastive']:<12.4f}"
              f"{h2['repsim_gain_pp']:<10.2f}{h2['trak_gain_pp']:<10.2f}{h2['h2_asymmetry_check'][:6]:<8}")
    print("-" * 82)
    print(f"H2 asymmetry (rep-gain < param-gain): {'PASS' if h2_asymmetry_pass else 'FAIL'} ({h2_asymmetry_passes}/3 tasks)")
    print(f"Total runtime: {total_time:.1f}s  Validity: {'ALL PASS' if all_valid else 'ISSUES'}")
    print("=" * 70)

    # Build summary string
    tox_auprc = results_per_task['toxicity']['metrics']['AUPRC']
    cf_r50 = results_per_task['counterfact']['metrics']['Recall@50']
    ft_r50 = results_per_task['ftrace']['metrics']['Recall@50']
    mark_done(
        "success" if all_valid else "warn",
        f"RepSim contrastive done. Valid={all_valid}. H2 asymmetry={'PASS' if h2_asymmetry_pass else 'FAIL'}. "
        f"{total_time:.0f}s. Tox AUPRC={tox_auprc:.4f}, CF R@50={cf_r50:.4f}, FT R@50={ft_r50:.4f}."
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
