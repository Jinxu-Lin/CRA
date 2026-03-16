#!/usr/bin/env python3
"""
Phase 1: TRAK Contrastive (Cell B of 2x2 factorial)
- TRAK (k=2048) with contrastive scoring (mean-subtracted) on Pythia-1B
- All 3 DATE-LM tasks: toxicity (AUPRC), counterfact (Recall@50+MRR), ftrace (P@K)
- Contrastive score: s_C(z_ref, z_train) = sim(z_ref, z_train) - mean_over_train[sim(z_ref, z')]
- PILOT mode: 100 training samples, seed=42
- Bootstrap CI (B=1000)
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
TASK_ID = "phase1_factorial_trak_contrastive"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE1_DIR = os.path.join(RESULTS_DIR, "phase1")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 8

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


# ── TRAK (CountSketch random projection) ───────────────────────────────
def compute_trak_scores_contrastive(model, tok, train_texts, ref_texts, task_name, k=TRAK_K):
    """
    TRAK with contrastive scoring:
    1. Compute projected gradient for each train and ref sample
    2. Compute raw similarity: sim[i,j] = <train_i, ref_j>
    3. Contrastive: s_C(ref_j, train_i) = sim[i,j] - mean_over_i[sim[i,j]]
       i.e., for each ref query, subtract the mean score across all training samples
    Returns: contrastive sim matrix [n_train, n_ref], raw sim matrix
    """
    t0 = time.time()

    # Select target parameters (last transformer layer)
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
    print(f"[TRAK-C/{task_name}] D={D/1e6:.2f}M from {target_names}")

    # CountSketch projection -- SAME seed as standard TRAK for fair comparison
    proj_seed = SEED + 7777
    rng_cs = np.random.RandomState(proj_seed)
    cs_buckets = torch.from_numpy(rng_cs.randint(0, k, size=D).astype(np.int64))
    cs_signs = torch.from_numpy(rng_cs.choice([-1.0, 1.0], size=D).astype(np.float32))
    print(f"[TRAK-C/{task_name}] CountSketch D={D} -> k={k}")

    def compute_projected_grads(texts, desc=""):
        all_proj = []
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
            proj = torch.zeros(k, dtype=torch.float32)
            proj.index_add_(0, cs_buckets, grad_flat * cs_signs)
            all_proj.append(proj)
            model.zero_grad(set_to_none=True)

            if (idx + 1) % 20 == 0:
                print(f"  [{desc}] {idx+1}/{len(texts)}")
                torch.cuda.empty_cache()

        return torch.stack(all_proj)

    print(f"[TRAK-C/{task_name}] Computing train projected grads ({len(train_texts)})...")
    train_proj = compute_projected_grads(train_texts, f"train-{task_name}")

    print(f"[TRAK-C/{task_name}] Computing ref projected grads ({len(ref_texts)})...")
    ref_proj = compute_projected_grads(ref_texts, f"ref-{task_name}")

    # Normalize and compute raw similarity
    train_proj_normed = F.normalize(train_proj, dim=-1)
    ref_proj_normed = F.normalize(ref_proj, dim=-1)
    raw_sim = (train_proj_normed @ ref_proj_normed.T).numpy()  # [n_train, n_ref]

    # ── Contrastive scoring ─────────────────────────────────────────────
    # For each ref query j: s_C(ref_j, train_i) = sim[i,j] - mean_i(sim[i,j])
    # This removes the "common influence" bias (FM2 correction)
    train_mean = raw_sim.mean(axis=0, keepdims=True)  # [1, n_ref]
    contrastive_sim = raw_sim - train_mean  # [n_train, n_ref]

    print(f"[TRAK-C/{task_name}] Contrastive scoring applied:")
    print(f"  Raw sim stats: mean={raw_sim.mean():.6f}, std={raw_sim.std():.6f}")
    print(f"  Contrastive sim stats: mean={contrastive_sim.mean():.6f}, std={contrastive_sim.std():.6f}")
    print(f"  Mean subtraction magnitude: mean={train_mean.mean():.6f}, std={train_mean.std():.6f}")

    # Restore grad flags
    for p in model.parameters():
        p.requires_grad_(True)
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"[TRAK-C/{task_name}] Done in {elapsed:.1f}s, sim shape={contrastive_sim.shape}")
    return contrastive_sim, raw_sim, elapsed


# ── Evaluation per task ─────────────────────────────────────────────────
def evaluate_toxicity(sim_matrix, pilot_data, ref_data):
    """
    Toxicity: score each training sample = mean attribution across ref queries.
    Unsafe ones should rank higher.
    """
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
    ci_lo, ci_hi = float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))

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
    """
    Counterfact: for each ref query, rank training samples by contrastive TRAK score.
    """
    n_train = len(pilot_data)
    n_ref = len(ref_data)

    scores_per_ref = []
    fact_indices_per_ref = []
    for j in range(n_ref):
        ref_sample = ref_data[j]
        scores_j = sim_matrix[:, j]  # [n_train]
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
    """
    Ftrace (factual tracing): for each ref query, rank training samples.
    """
    n_train = len(pilot_data)
    n_ref = len(ref_data)

    scores_per_ref = []
    fact_indices_per_ref = []

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
def inspect_samples(sim_matrix, raw_sim, pilot_data, task_name, n_show=5):
    """Print top-scoring training samples for qualitative inspection, both raw and contrastive."""
    # Contrastive scores
    c_scores = sim_matrix.mean(axis=1)
    top_idx = np.argsort(-c_scores)[:n_show]
    bot_idx = np.argsort(c_scores)[:n_show]

    # Raw scores
    r_scores = raw_sim.mean(axis=1)

    print(f"\n[Qualitative/{task_name}] Top-{n_show} contrastive-highest training samples:")
    for rank, idx in enumerate(top_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:100]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} c_score={c_scores[idx]:.4f} raw={r_scores[idx]:.4f} type={label}: {text}...")

    print(f"[Qualitative/{task_name}] Top-{n_show} contrastive-lowest training samples:")
    for rank, idx in enumerate(bot_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:100]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} c_score={c_scores[idx]:.4f} raw={r_scores[idx]:.4f} type={label}: {text}...")

    # Rank correlation between raw and contrastive
    from scipy.stats import spearmanr
    rho, pval = spearmanr(r_scores, c_scores)
    print(f"[Qualitative/{task_name}] Spearman(raw, contrastive) = {rho:.4f} (p={pval:.2e})")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    for d in [PHASE1_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print(f"Phase 1: TRAK Contrastive (Cell B of 2x2 factorial)")
    print(f"Model: {MODEL_NAME}, Pilot N={PILOT_N_TRAIN}, TRAK k={TRAK_K}")
    print(f"Scoring: CONTRASTIVE (mean-subtracted)")
    print(f"Tasks: toxicity, counterfact, ftrace")
    print(f"Bootstrap B={BOOTSTRAP_B}")
    print("=" * 70)

    # Load data
    tasks = load_all_tasks()
    model, tok = load_model()

    results_per_task = {}
    all_timings = {}
    contrastive_analysis = {}

    # Also load TRAK standard results for direct comparison
    trak_std_path = os.path.join(PHASE1_DIR, "trak_standard.json")
    trak_std_results = None
    if os.path.exists(trak_std_path):
        with open(trak_std_path) as f:
            trak_std_results = json.load(f)
        print(f"[INFO] Loaded TRAK standard results for comparison")
    else:
        print(f"[WARN] TRAK standard results not found at {trak_std_path}")

    for task_idx, (task_name, task_info) in enumerate(tasks.items()):
        print(f"\n{'='*70}")
        print(f"Task {task_idx+1}/3: {task_name} (metric: {task_info['metric_name']})")
        print(f"{'='*70}")

        report_progress("experiment", f"TRAK contrastive on {task_name}", (task_idx) / 3.0)

        # Create pilot subset -- SAME seed & strategy as standard for fair comparison
        pilot_data, pilot_idx = create_pilot_subset(task_name, task_info["train"])

        # Format texts
        fmt_fn = task_info["fmt"]
        train_texts = [fmt_fn(pilot_data[i]) for i in range(len(pilot_data))]
        ref_texts = [fmt_fn(task_info["ref"][i]) for i in range(len(task_info["ref"]))]

        # Compute TRAK contrastive scores
        contrastive_sim, raw_sim, elapsed = compute_trak_scores_contrastive(
            model, tok, train_texts, ref_texts, task_name
        )
        all_timings[task_name] = round(elapsed, 2)

        # Track contrastive analysis details
        contrastive_analysis[task_name] = {
            "raw_sim_mean": float(raw_sim.mean()),
            "raw_sim_std": float(raw_sim.std()),
            "contrastive_sim_mean": float(contrastive_sim.mean()),
            "contrastive_sim_std": float(contrastive_sim.std()),
            "mean_subtraction_magnitude": float(raw_sim.mean(axis=0).mean()),
            "mean_subtraction_std": float(raw_sim.mean(axis=0).std()),
        }

        # Evaluate using contrastive scores
        if task_name == "toxicity":
            metrics = evaluate_toxicity(contrastive_sim, pilot_data, task_info["ref"])
        elif task_name == "counterfact":
            metrics = evaluate_counterfact(contrastive_sim, pilot_data, task_info["ref"])
        elif task_name == "ftrace":
            metrics = evaluate_ftrace(contrastive_sim, pilot_data, task_info["ref"])

        # Qualitative inspection comparing raw vs contrastive
        inspect_samples(contrastive_sim, raw_sim, pilot_data, task_name)

        results_per_task[task_name] = {
            "metrics": metrics,
            "runtime_sec": round(elapsed, 2),
            "n_train_pilot": len(pilot_data),
            "n_ref": len(task_info["ref"]),
        }

        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate results ───────────────────────────────────────────────
    report_progress("analysis", "Aggregating results and comparing with TRAK standard", 0.95)

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

    # ── H2 directional check: TRAK-C should improve over TRAK standard ──
    h2_comparison = {}
    if trak_std_results:
        for tn in results_per_task:
            std_res = trak_std_results.get("results_per_task", {}).get(tn, {}).get("metrics", {})
            c_res = results_per_task[tn]["metrics"]
            if tn == "toxicity":
                std_val = std_res.get("AUPRC", 0)
                c_val = c_res["AUPRC"]
                metric_name = "AUPRC"
            else:
                std_val = std_res.get("Recall@50", 0)
                c_val = c_res["Recall@50"]
                metric_name = "Recall@50"
            gain = c_val - std_val
            h2_comparison[tn] = {
                "metric": metric_name,
                "trak_standard": round(std_val, 6),
                "trak_contrastive": round(c_val, 6),
                "gain_pp": round(gain * 100, 2),
                "improved": gain > 0,
            }
            print(f"[H2/{tn}] TRAK standard={std_val:.4f} -> TRAK-C={c_val:.4f} (gain={gain*100:+.2f}pp)")

    h2_pass = sum(1 for v in h2_comparison.values() if v["improved"]) >= 2 if h2_comparison else None
    print(f"\n[H2 directional check] TRAK-C improves on {sum(1 for v in h2_comparison.values() if v['improved'])}/3 tasks -> {'PASS' if h2_pass else 'INCONCLUSIVE/FAIL'}")

    total_time = time.time() - t_start
    final = {
        "task_id": TASK_ID,
        "method": "TRAK_contrastive",
        "scoring": "contrastive",
        "contrastive_protocol": "mean_subtraction_over_train",
        "space": "parameter",
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "trak_k": TRAK_K,
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "results_per_task": results_per_task,
        "contrastive_analysis": contrastive_analysis,
        "h2_comparison": h2_comparison,
        "h2_directional_pass": h2_pass,
        "timings": all_timings,
        "validity_checks": validity,
        "all_valid": all_valid,
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    out_path = os.path.join(PHASE1_DIR, "trak_contrastive.json")
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
    print("SUMMARY: TRAK Contrastive (Phase 1, Cell B)")
    print("=" * 70)
    print(f"{'Task':<15}{'Metric':<15}{'TRAK-C':<10}{'CI_lo':<10}{'CI_hi':<10}{'TRAK-std':<10}{'Gain(pp)':<10}{'Time(s)':<8}")
    print("-" * 88)
    for tn, res in results_per_task.items():
        m = res["metrics"]
        h2 = h2_comparison.get(tn, {})
        std_v = h2.get("trak_standard", "N/A")
        gain = h2.get("gain_pp", "N/A")
        if tn == "toxicity":
            std_s = f"{std_v:.4f}" if isinstance(std_v, float) else str(std_v)
            gain_s = f"{gain:+.2f}" if isinstance(gain, (int, float)) else str(gain)
            print(f"{tn:<15}{'AUPRC':<15}{m['AUPRC']:<10.4f}{m['CI_lower']:<10.4f}{m['CI_upper']:<10.4f}{std_s:<10}{gain_s:<10}{res['runtime_sec']:<8.1f}")
        else:
            ci_r = m.get("Recall@50_CI", [0, 0])
            std_s = f"{std_v:.4f}" if isinstance(std_v, float) else str(std_v)
            gain_s = f"{gain:+.2f}" if isinstance(gain, (int, float)) else str(gain)
            print(f"{tn:<15}{'Recall@50':<15}{m['Recall@50']:<10.4f}{ci_r[0]:<10.4f}{ci_r[1]:<10.4f}{std_s:<10}{gain_s:<10}{res['runtime_sec']:<8.1f}")
            ci_m = m.get("MRR_CI", [0, 0])
            print(f"{'':<15}{'MRR':<15}{m['MRR']:<10.4f}{ci_m[0]:<10.4f}{ci_m[1]:<10.4f}")
    print("-" * 88)
    print(f"H2 directional check: {'PASS' if h2_pass else 'INCONCLUSIVE/FAIL'}")
    print(f"Total runtime: {total_time:.1f}s  Validity: {'ALL PASS' if all_valid else 'ISSUES'}")
    print("=" * 70)

    mark_done(
        "success" if all_valid else "warn",
        f"TRAK contrastive done. Valid={all_valid}. H2={'PASS' if h2_pass else 'FAIL'}. {total_time:.0f}s. "
        f"Tox AUPRC={results_per_task['toxicity']['metrics']['AUPRC']:.4f}, "
        f"CF R@50={results_per_task['counterfact']['metrics']['Recall@50']:.4f}, "
        f"FT R@50={results_per_task['ftrace']['metrics']['Recall@50']:.4f}"
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
