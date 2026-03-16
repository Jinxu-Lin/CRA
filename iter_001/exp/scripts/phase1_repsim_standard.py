#!/usr/bin/env python3
"""
Phase 1: RepSim Standard (Cell C of 2x2 factorial)
- RepSim (last-layer cosine similarity, standard scoring) on Pythia-1B
- All 3 DATE-LM tasks: toxicity (AUPRC), counterfact (Recall@50+MRR), ftrace (P@K)
- PILOT mode: 100 training samples, seed=42
- Bootstrap CI (B=1000)
- Caches extracted representations for reuse by contrastive variant
"""

import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_curve, auc

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "phase1_factorial_repsim_standard"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE1_DIR = os.path.join(RESULTS_DIR, "phase1")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 32  # RepSim only needs forward pass, can use larger batch

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


# ── Data loading ────────────────────────────────────────────────────────
def load_all_tasks():
    """Load all 3 DATE-LM tasks."""
    report_progress("loading_data", "Loading DATE-LM datasets", 0.05)
    tasks = {}

    # 1. Toxicity
    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {
        "train": tox["train"], "ref": tox["ref"],
        "metric_name": "AUPRC",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    # 2. Counterfact
    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {
        "train": cf["train"], "ref": cf["ref"],
        "metric_name": "Recall@50+MRR",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")

    # 3. Ftrace (factual tracing)
    ft = load_dataset("DataAttributionEval/ftrace", "Pythia-1b")
    tasks["ftrace"] = {
        "train": ft["train"], "ref": ft["ref"],
        "metric_name": "P@K",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[ftrace] train={len(ft['train'])}, ref={len(ft['ref'])}")

    return tasks


def create_pilot_subset(task_name, train_data, n_pilot=PILOT_N_TRAIN):
    """Create pilot subset with stratified sampling for toxicity."""
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
        print(f"[pilot/{task_name}] Stratified: {n_unsafe} unsafe + {len(chosen_safe)} safe = {len(pilot_idx)}")
    else:
        pilot_idx = sorted(rng.choice(n_total, min(n_pilot, n_total), replace=False).tolist())
        print(f"[pilot/{task_name}] Random: {len(pilot_idx)} samples")

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


# ── Representation extraction ───────────────────────────────────────────
@torch.no_grad()
def extract_representations(model, tok, texts, desc="", batch_size=BATCH_SIZE):
    """
    Extract last-layer hidden states (last non-pad token) from Pythia-1B.
    Returns: L2-normalized representation tensor [N, hidden_dim].
    """
    all_reps = []
    n = len(texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]

        inputs = tok(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        ).to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1]  # last layer: [batch, seq, hidden_dim]
        # Get position of last non-pad token for each sample
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1  # [batch]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        reps = hidden[batch_idx, seq_lens]  # [batch, hidden_dim]

        # Convert to float32 for stable normalization
        reps = reps.float().cpu()
        all_reps.append(reps)

        if (start + batch_size) % (batch_size * 4) == 0 or end == n:
            print(f"  [{desc}] {end}/{n} samples extracted")

    reps_tensor = torch.cat(all_reps, dim=0)  # [N, hidden_dim]
    # L2-normalize
    reps_tensor = F.normalize(reps_tensor, dim=-1)
    print(f"  [{desc}] Final shape: {reps_tensor.shape}, norm check: {reps_tensor.norm(dim=-1).mean():.4f}")
    return reps_tensor


def compute_repsim_scores(train_reps, ref_reps):
    """
    Compute RepSim (cosine similarity) scores.
    Standard scoring: no mean subtraction.
    Returns: similarity matrix [n_train, n_ref] as numpy array.
    """
    # Both are already L2-normalized, so dot product = cosine similarity
    sim = (train_reps @ ref_reps.T).numpy()  # [n_train, n_ref]
    return sim


# ── Evaluation per task ─────────────────────────────────────────────────
def evaluate_toxicity(sim_matrix, pilot_data, ref_data):
    """Toxicity: score each training sample by mean similarity to refs."""
    train_scores = sim_matrix.mean(axis=1)  # [n_train]
    n_train = len(pilot_data)
    unsafe_indices = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]

    auprc = compute_auprc(train_scores, unsafe_indices, n_train)

    # Bootstrap CI on AUPRC
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
    """Counterfact: for each ref query, rank training samples by RepSim score."""
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

    # Bootstrap CI: resample ref queries
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
    """Ftrace: for each ref query, rank training samples by RepSim score."""
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

    print(f"\n[Qualitative/{task_name}] Top-{n_show} highest-scoring training samples:")
    for rank, idx in enumerate(top_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:100]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} score={mean_scores[idx]:.4f} type={label}: {text}...")

    print(f"[Qualitative/{task_name}] Top-{n_show} lowest-scoring training samples:")
    for rank, idx in enumerate(bot_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:100]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} score={mean_scores[idx]:.4f} type={label}: {text}...")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    for d in [PHASE1_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print(f"Phase 1: RepSim Standard (Cell C of 2x2 factorial)")
    print(f"Model: {MODEL_NAME}, Pilot N={PILOT_N_TRAIN}")
    print(f"Tasks: toxicity, counterfact, ftrace")
    print(f"Bootstrap B={BOOTSTRAP_B}")
    print(f"Scoring: standard (no contrastive)")
    print("=" * 70)

    # Load data
    tasks = load_all_tasks()
    model, tok = load_model()

    results_per_task = {}
    all_timings = {}
    cached_reps = {}  # Will cache for contrastive variant

    for task_idx, (task_name, task_info) in enumerate(tasks.items()):
        print(f"\n{'='*70}")
        print(f"Task {task_idx+1}/3: {task_name} (metric: {task_info['metric_name']})")
        print(f"{'='*70}")

        report_progress("experiment", f"RepSim standard on {task_name}", (task_idx) / 3.0)
        t_task = time.time()

        # Create pilot subset (same random seed as TRAK tasks for direct comparison)
        pilot_data, pilot_idx = create_pilot_subset(task_name, task_info["train"])

        # Format texts
        fmt_fn = task_info["fmt"]
        train_texts = [fmt_fn(pilot_data[i]) for i in range(len(pilot_data))]
        ref_texts = [fmt_fn(task_info["ref"][i]) for i in range(len(task_info["ref"]))]

        # Extract representations
        print(f"[RepSim/{task_name}] Extracting train representations ({len(train_texts)} samples)...")
        train_reps = extract_representations(model, tok, train_texts, desc=f"train-{task_name}")

        print(f"[RepSim/{task_name}] Extracting ref representations ({len(ref_texts)} samples)...")
        ref_reps = extract_representations(model, tok, ref_texts, desc=f"ref-{task_name}")

        # Cache representations for contrastive variant
        cache_key = f"repsim_{task_name}"
        cached_reps[cache_key] = {
            "train_reps": train_reps,
            "ref_reps": ref_reps,
            "pilot_idx": pilot_idx,
        }

        # Compute RepSim (cosine similarity, standard scoring)
        print(f"[RepSim/{task_name}] Computing cosine similarity matrix...")
        sim_matrix = compute_repsim_scores(train_reps, ref_reps)
        print(f"[RepSim/{task_name}] Similarity matrix shape: {sim_matrix.shape}")
        print(f"[RepSim/{task_name}] Score stats: mean={sim_matrix.mean():.4f}, "
              f"std={sim_matrix.std():.4f}, min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}")

        elapsed = time.time() - t_task
        all_timings[task_name] = round(elapsed, 2)

        # Evaluate
        if task_name == "toxicity":
            metrics = evaluate_toxicity(sim_matrix, pilot_data, task_info["ref"])
        elif task_name == "counterfact":
            metrics = evaluate_counterfact(sim_matrix, pilot_data, task_info["ref"])
        elif task_name == "ftrace":
            metrics = evaluate_ftrace(sim_matrix, pilot_data, task_info["ref"])

        # Qualitative inspection
        inspect_samples(sim_matrix, pilot_data, task_name)

        results_per_task[task_name] = {
            "metrics": metrics,
            "runtime_sec": round(elapsed, 2),
            "n_train_pilot": len(pilot_data),
            "n_ref": len(task_info["ref"]),
        }

        gc.collect()
        torch.cuda.empty_cache()

    # ── Cache representations to disk ───────────────────────────────────
    report_progress("caching", "Saving cached representations", 0.90)
    cache_path = os.path.join(CACHE_DIR, "repsim_standard_reps.pt")
    cache_data = {}
    for key, val in cached_reps.items():
        cache_data[f"{key}_train"] = val["train_reps"]
        cache_data[f"{key}_ref"] = val["ref_reps"]
        cache_data[f"{key}_pilot_idx"] = val["pilot_idx"]
    torch.save(cache_data, cache_path)
    cache_size_mb = os.path.getsize(cache_path) / 1e6
    print(f"Cached representations saved to {cache_path} ({cache_size_mb:.1f} MB)")
    print(f"Keys: {list(cache_data.keys())}")

    # ── Aggregate results ───────────────────────────────────────────────
    report_progress("analysis", "Aggregating results", 0.95)

    # Validity checks
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

    # Pilot pass criteria: RepSim LDS > 0.3 (sanity)
    # We use a composite check across tasks
    repsim_sanity = True
    for tn, res in results_per_task.items():
        m = res["metrics"]
        primary = m.get("AUPRC", m.get("Recall@50", 0))
        if primary < 0.3:
            repsim_sanity = False
            print(f"WARNING: {tn} primary metric ({primary:.4f}) below 0.3 sanity threshold")

    total_time = time.time() - t_start
    final = {
        "task_id": TASK_ID,
        "method": "RepSim_standard",
        "scoring": "standard",
        "space": "representation",
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "extraction_layer": "last_hidden_state",
        "similarity": "cosine",
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "results_per_task": results_per_task,
        "timings": all_timings,
        "validity_checks": validity,
        "all_valid": all_valid,
        "repsim_sanity_check": repsim_sanity,
        "cache_file": cache_path,
        "cache_size_mb": round(cache_size_mb, 2),
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    out_path = os.path.join(PHASE1_DIR, "repsim_standard.json")
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
    print("SUMMARY: RepSim Standard (Phase 1, Cell C)")
    print("=" * 70)
    print(f"{'Task':<15}{'Primary Metric':<20}{'Value':<10}{'CI_lo':<10}{'CI_hi':<10}{'Time(s)':<8}")
    print("-" * 73)
    for tn, res in results_per_task.items():
        m = res["metrics"]
        if tn == "toxicity":
            print(f"{tn:<15}{'AUPRC':<20}{m['AUPRC']:<10.4f}{m['CI_lower']:<10.4f}{m['CI_upper']:<10.4f}{res['runtime_sec']:<8.1f}")
        else:
            ci_r = m.get("Recall@50_CI", [0, 0])
            print(f"{tn:<15}{'Recall@50':<20}{m['Recall@50']:<10.4f}{ci_r[0]:<10.4f}{ci_r[1]:<10.4f}{res['runtime_sec']:<8.1f}")
            ci_m = m.get("MRR_CI", [0, 0])
            print(f"{'':<15}{'MRR':<20}{m['MRR']:<10.4f}{ci_m[0]:<10.4f}{ci_m[1]:<10.4f}{'':<8}")
    print("-" * 73)
    print(f"Total runtime: {total_time:.1f}s  Validity: {'ALL PASS' if all_valid else 'ISSUES'}")
    print(f"Sanity (all >0.3): {'PASS' if repsim_sanity else 'FAIL'}")
    print(f"Representations cached: {cache_path} ({cache_size_mb:.1f} MB)")
    print("=" * 70)

    mark_done(
        "success" if (all_valid and repsim_sanity) else "warn",
        f"RepSim standard done. Valid={all_valid}, Sanity={repsim_sanity}. {total_time:.0f}s. "
        f"Tox AUPRC={results_per_task['toxicity']['metrics']['AUPRC']:.4f}, "
        f"CF R@50={results_per_task['counterfact']['metrics']['Recall@50']:.4f}, "
        f"FT R@50={results_per_task['ftrace']['metrics']['Recall@50']:.4f}. "
        f"Cache: {cache_size_mb:.1f}MB"
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
