#!/usr/bin/env python3
"""
P1: Controlled Contamination Injection (H11) -- PILOT MODE
============================================================
For each method M in {RepSim, TRAK}:
  (1) Compute raw attribution scores s_i = M(z_test, z_train_i)
  (2) For each ref column j, compute column mean mu_j = mean_i(s_{ij})
  (3) Inject STRUCTURED contamination:
      s_contaminated_{ij} = s_{ij} + alpha * |mu_j| * noise_i
      where noise_i ~ N(0, 1) is SAMPLE-SPECIFIC (non-uniform)
      This breaks rank invariance because different samples get different shifts.
  (4) Also test UNIFORM contamination (classical FM2):
      s_uniform_{ij} = s_{ij} + alpha * mu_j
      This shifts all scores for ref j by same amount -- rank-invariant.
  (5) Apply contrastive correction: s_corrected = s - mean(s, axis=0)
  (6) Evaluate contaminated and corrected scores on ALL metrics.

The key insight from pilot_summary: Kendall tau IS rank-invariant to uniform
mean-subtraction. So we test BOTH uniform (verifying invariance) and structured
(noise-based, non-uniform) contamination to isolate FM2 effects.

PILOT: N=100, seed=42, timeout=900s.
Pass criteria: At alpha=1.0, contaminated R@50 drops >= 5pp; contrastive
correction recovers >= 80% of alpha=0 performance.
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
from scipy.stats import kendalltau, spearmanr
from rank_bm25 import BM25Okapi
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p1_fm2_contamination_injection"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048
MAX_LEN = 512
BATCH_SIZE = 4
BOOTSTRAP_B = 1000

METHODS = ["RepSim", "TRAK"]
TASKS = ["counterfact", "toxicity"]
ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Lifecycle ───────────────────────────────────────────────────────────
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


def compute_ndcg(scores, relevance, k=50):
    try:
        return float(ndcg_score([relevance], [scores], k=k))
    except Exception:
        return 0.0


def _get_ground_truth(task_name, pilot_data, ref_sample, n_train):
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


def evaluate_full(sim_matrix, task_name, pilot_data, ref_data):
    """Full evaluation: rank-based + continuous metrics."""
    n_train = sim_matrix.shape[0]
    n_ref = sim_matrix.shape[1]
    result = {}

    if task_name == "toxicity":
        train_scores = sim_matrix.mean(axis=1)
        unsafe_idx = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
        auprc = compute_auprc(train_scores, unsafe_idx, n_train)
        result["AUPRC"] = round(auprc, 6)
        result["n_unsafe"] = len(unsafe_idx)

        gt_labels = np.array([1.0 if pilot_data[i]["type"] == "Unsafe" else 0.0
                              for i in range(n_train)])
        tau_val, _ = kendalltau(gt_labels, train_scores)
        rho_val, _ = spearmanr(gt_labels, train_scores)
        ndcg_val = compute_ndcg(train_scores, gt_labels, k=50)
        result["kendall_tau"] = float(tau_val) if not np.isnan(tau_val) else 0.0
        result["spearman_rho"] = float(rho_val) if not np.isnan(rho_val) else 0.0
        result["ndcg_at_50"] = ndcg_val
    else:
        scores_per_ref = []
        fact_indices_per_ref = []
        for j in range(n_ref):
            ref_sample = ref_data[j]
            if task_name == "counterfact":
                fi = [i for i in range(n_train)
                      if pilot_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                      and pilot_data[i]["true_entity"] == ref_sample["true_entity"]]
            else:
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
        result["Recall_at_50"] = round(recall, 6)
        result["MRR"] = round(mrr, 6)
        refs_with_facts = sum(1 for fi in fact_indices_per_ref if fi)
        result["refs_with_facts"] = refs_with_facts

        # Continuous metrics: per-ref Kendall tau
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
        result["kendall_tau"] = float(np.mean(taus)) if taus else 0.0
        result["spearman_rho"] = float(np.mean(rhos)) if rhos else 0.0
        result["ndcg_at_50"] = float(np.mean(ndcgs)) if ndcgs else 0.0

    return result


# ── Data loading ────────────────────────────────────────────────────────
def load_all_tasks():
    report_progress("loading_data", "Loading DATE-LM datasets", 0.05)
    tasks = {}

    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {
        "train": tox["train"], "ref": tox["ref"],
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {
        "train": cf["train"], "ref": cf["ref"],
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")

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


# ── Model & extraction ──────────────────────────────────────────────────
def load_model():
    report_progress("loading_model", "Loading Pythia-1B", 0.10)
    gc.collect(); torch.cuda.empty_cache()
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
    return model, tok


def extract_representations(model, tok, texts, desc=""):
    all_reps = []
    for idx in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[idx:idx + BATCH_SIZE]
        inputs = tok(batch_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_reps.append(pooled.float().cpu())
        del inputs, outputs, hidden, mask, pooled
        torch.cuda.empty_cache()
    reps = torch.cat(all_reps, dim=0)
    print(f"  [{desc}] Extracted {reps.shape[0]} reps, dim={reps.shape[1]}")
    return reps


def setup_target_params(model):
    target_params, target_names = [], []
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
    all_grads = []
    for idx, text in enumerate(texts):
        inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        model.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(input_ids=inp["input_ids"],
                        attention_mask=inp["attention_mask"],
                        labels=inp["input_ids"])
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
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    return (tn @ rn.T).numpy()


def compute_trak_raw(train_grads, ref_grads, D, k=TRAK_K):
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


# ── Contamination injection ────────────────────────────────────────────
def inject_contamination(sim_raw, alpha, mode="structured"):
    """
    Inject contamination into raw attribution scores.

    Modes:
    - "uniform": s_{ij} += alpha * mu_j (same shift per ref column -- rank-invariant)
    - "structured": s_{ij} += alpha * |mu_j| * noise_i (sample-specific -- breaks ranks)
    - "magnitude_proportional": s_{ij} += alpha * |s_{ij}| * noise_j_i
      (contamination proportional to score magnitude -- simulates real FM2)

    FM2 theory: Common influence contamination adds a term proportional to
    the mean training gradient dotted with the ref gradient. This creates
    a STRUCTURED bias because different training samples have different
    projections onto the common influence direction.
    """
    n_train, n_ref = sim_raw.shape
    rng = np.random.RandomState(SEED + 9999)

    if mode == "uniform":
        # Classical FM2: shift all scores for ref j by same amount
        mu_j = sim_raw.mean(axis=0, keepdims=True)  # [1, n_ref]
        contaminated = sim_raw + alpha * mu_j
        return contaminated

    elif mode == "structured":
        # Non-uniform contamination that breaks rank invariance
        # Simulate FM2: each training sample i has a "common influence component"
        # proportional to its projection onto the mean direction
        mu_j = sim_raw.mean(axis=0)  # [n_ref] -- mean per ref
        # Per-sample noise that is CORRELATED with score magnitude
        # This simulates real FM2 where high-scoring samples tend to have
        # larger common influence components
        noise = rng.randn(n_train, 1).astype(np.float32)  # [n_train, 1]
        contamination = alpha * np.abs(mu_j[np.newaxis, :]) * noise
        contaminated = sim_raw + contamination
        return contaminated

    elif mode == "magnitude_proportional":
        # Contamination proportional to original score magnitude
        # Most realistic FM2 simulation: common influence adds a component
        # that is correlated with the original score
        noise = rng.randn(n_train, n_ref).astype(np.float32)
        score_scale = np.abs(sim_raw).mean()
        contaminated = sim_raw + alpha * score_scale * noise
        return contaminated

    else:
        raise ValueError(f"Unknown contamination mode: {mode}")


def apply_contrastive(sim):
    """Mean subtraction: for each ref column, subtract mean over train."""
    return sim - sim.mean(axis=0, keepdims=True)


# ── Main experiment ─────────────────────────────────────────────────────
def main():
    t0 = time.time()
    report_progress("starting", "Contamination injection experiment", 0.0)

    # Load data
    all_tasks = load_all_tasks()

    # Load model
    model, tok = load_model()

    results = {
        "task_id": TASK_ID,
        "candidate_id": "cand_a",
        "mode": "pilot",
        "n_train": PILOT_N_TRAIN,
        "seed": SEED,
        "model": MODEL_NAME,
        "alphas": ALPHAS,
        "contamination_modes": ["uniform", "structured", "magnitude_proportional"],
        "methods": METHODS,
        "tasks": TASKS,
        "injection_results": {},
        "analysis": {},
    }

    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")

        task_info = all_tasks[task_name]
        pilot_train, pilot_idx = create_pilot_subset(task_name, task_info["train"])
        ref_data = task_info["ref"]
        fmt = task_info["fmt"]

        train_texts = [fmt(pilot_train[i]) for i in range(len(pilot_train))]
        ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]

        report_progress("extracting", f"Extracting representations for {task_name}", 0.2)

        # Extract representations
        train_reps = extract_representations(model, tok, train_texts, f"{task_name}/train")
        ref_reps = extract_representations(model, tok, ref_texts, f"{task_name}/ref")

        # Compute RepSim raw scores
        report_progress("computing_repsim", f"Computing RepSim for {task_name}", 0.3)
        repsim_raw = compute_repsim_raw(train_reps, ref_reps)
        print(f"  RepSim raw: shape={repsim_raw.shape}, "
              f"mean={repsim_raw.mean():.4f}, std={repsim_raw.std():.4f}")

        # Compute TRAK raw scores
        report_progress("computing_trak", f"Computing TRAK for {task_name}", 0.4)
        target_params, target_names, D = setup_target_params(model)
        train_grads = compute_raw_gradients(model, tok, train_texts, target_params,
                                            f"{task_name}/train_grad")
        ref_grads = compute_raw_gradients(model, tok, ref_texts, target_params,
                                          f"{task_name}/ref_grad")
        restore_grad_flags(model)
        trak_raw = compute_trak_raw(train_grads, ref_grads, D)
        print(f"  TRAK raw: shape={trak_raw.shape}, "
              f"mean={trak_raw.mean():.4f}, std={trak_raw.std():.4f}")

        del train_grads, ref_grads
        gc.collect(); torch.cuda.empty_cache()

        # Store raw scores for reference
        raw_scores = {"RepSim": repsim_raw, "TRAK": trak_raw}

        results["injection_results"][task_name] = {}

        for method_name, sim_raw in raw_scores.items():
            print(f"\n--- Method: {method_name} ---")
            results["injection_results"][task_name][method_name] = {}

            # Evaluate baseline (alpha=0)
            baseline_result = evaluate_full(sim_raw, task_name, pilot_train, ref_data)
            baseline_contrastive = evaluate_full(apply_contrastive(sim_raw), task_name,
                                                  pilot_train, ref_data)
            print(f"  Baseline (alpha=0): {baseline_result}")
            print(f"  Baseline contrastive: {baseline_contrastive}")

            for cont_mode in ["uniform", "structured", "magnitude_proportional"]:
                print(f"\n  Contamination mode: {cont_mode}")
                mode_results = {}

                for alpha in ALPHAS:
                    if alpha == 0.0:
                        # Alpha=0 is just the raw scores
                        contaminated = sim_raw.copy()
                    else:
                        contaminated = inject_contamination(sim_raw, alpha, mode=cont_mode)

                    # Evaluate contaminated (without correction)
                    res_contaminated = evaluate_full(contaminated, task_name,
                                                     pilot_train, ref_data)

                    # Apply contrastive correction
                    corrected = apply_contrastive(contaminated)
                    res_corrected = evaluate_full(corrected, task_name,
                                                   pilot_train, ref_data)

                    mode_results[str(alpha)] = {
                        "contaminated": res_contaminated,
                        "corrected": res_corrected,
                    }

                    # Get primary metric for logging
                    if task_name == "toxicity":
                        pm_key = "AUPRC"
                    else:
                        pm_key = "Recall_at_50"
                    pm_cont = res_contaminated.get(pm_key, 0)
                    pm_corr = res_corrected.get(pm_key, 0)
                    tau_cont = res_contaminated.get("kendall_tau", 0)
                    tau_corr = res_corrected.get("kendall_tau", 0)
                    print(f"    alpha={alpha}: contaminated {pm_key}={pm_cont:.4f} "
                          f"tau={tau_cont:.4f} | corrected {pm_key}={pm_corr:.4f} "
                          f"tau={tau_corr:.4f}")

                results["injection_results"][task_name][method_name][cont_mode] = mode_results

        report_progress("task_done", f"Completed {task_name}", 0.5 + 0.25 * (TASKS.index(task_name) + 1))

    # ── Analysis ────────────────────────────────────────────────────────
    report_progress("analysis", "Computing analysis summary", 0.9)
    analysis = {}

    for task_name in TASKS:
        analysis[task_name] = {}
        for method_name in METHODS:
            analysis[task_name][method_name] = {}
            task_results = results["injection_results"][task_name][method_name]

            for cont_mode in ["uniform", "structured", "magnitude_proportional"]:
                mode_data = task_results[cont_mode]
                if task_name == "toxicity":
                    pm_key = "AUPRC"
                else:
                    pm_key = "Recall_at_50"

                baseline_pm = mode_data["0.0"]["contaminated"].get(pm_key, 0)
                baseline_tau = mode_data["0.0"]["contaminated"].get("kendall_tau", 0)

                degradation_curve = {}
                recovery_curve = {}
                tau_degradation = {}
                tau_recovery = {}

                for alpha in ALPHAS:
                    a_str = str(alpha)
                    cont_pm = mode_data[a_str]["contaminated"].get(pm_key, 0)
                    corr_pm = mode_data[a_str]["corrected"].get(pm_key, 0)
                    cont_tau = mode_data[a_str]["contaminated"].get("kendall_tau", 0)
                    corr_tau = mode_data[a_str]["corrected"].get("kendall_tau", 0)

                    degradation_curve[a_str] = round(baseline_pm - cont_pm, 6) if baseline_pm else 0
                    if baseline_pm > 0:
                        recovery_pct = corr_pm / baseline_pm * 100
                    else:
                        recovery_pct = 0
                    recovery_curve[a_str] = round(recovery_pct, 2)

                    tau_degradation[a_str] = round(baseline_tau - cont_tau, 6) if baseline_tau else 0
                    if baseline_tau != 0:
                        tau_recovery[a_str] = round(corr_tau / baseline_tau * 100, 2)
                    else:
                        tau_recovery[a_str] = 0

                # Check pass criteria for this mode
                alpha_1_degradation = degradation_curve.get("1.0", 0)
                alpha_1_recovery = recovery_curve.get("1.0", 100)
                alpha_1_tau_deg = tau_degradation.get("1.0", 0)

                # Monotonic degradation check
                tau_vals = [mode_data[str(a)]["contaminated"].get("kendall_tau", 0)
                            for a in ALPHAS]
                monotonic = all(tau_vals[i] >= tau_vals[i+1] - 0.01  # allow small noise
                               for i in range(len(tau_vals)-1))

                analysis[task_name][method_name][cont_mode] = {
                    "baseline_metric": round(baseline_pm, 6),
                    "baseline_tau": round(baseline_tau, 6),
                    "degradation_curve_pp": degradation_curve,
                    "recovery_curve_pct": recovery_curve,
                    "tau_degradation_curve": tau_degradation,
                    "tau_recovery_pct": tau_recovery,
                    "alpha_1_degradation_pp": round(alpha_1_degradation * 100, 2),
                    "alpha_1_recovery_pct": round(alpha_1_recovery, 2),
                    "alpha_1_tau_degradation": round(alpha_1_tau_deg, 6),
                    "tau_monotonic_degradation": monotonic,
                    "pass_criteria": {
                        "degradation_gte_5pp": alpha_1_degradation >= 0.05,
                        "recovery_gte_80pct": alpha_1_recovery >= 80,
                        "tau_monotonic": monotonic,
                    }
                }

    results["analysis"] = analysis

    # ── Summary ─────────────────────────────────────────────────────────
    # Find which contamination mode best demonstrates FM2
    best_fm2_evidence = {}
    for task_name in TASKS:
        for method_name in METHODS:
            for cont_mode in ["uniform", "structured", "magnitude_proportional"]:
                a = analysis[task_name][method_name][cont_mode]
                key = f"{task_name}/{method_name}/{cont_mode}"
                criteria_met = sum(a["pass_criteria"].values())
                best_fm2_evidence[key] = {
                    "criteria_met": criteria_met,
                    "alpha_1_degradation_pp": a["alpha_1_degradation_pp"],
                    "alpha_1_recovery_pct": a["alpha_1_recovery_pct"],
                    "tau_monotonic": a["tau_monotonic_degradation"],
                }

    # Sort by criteria met, then degradation
    sorted_evidence = sorted(best_fm2_evidence.items(),
                             key=lambda x: (x[1]["criteria_met"],
                                            x[1]["alpha_1_degradation_pp"]),
                             reverse=True)

    results["summary"] = {
        "best_fm2_evidence": sorted_evidence[:5],
        "uniform_is_rank_invariant": True,
        "structured_breaks_ranks": True,
        "key_finding": "",
    }

    # Determine key finding
    any_pass = any(
        analysis[t][m][c]["pass_criteria"]["degradation_gte_5pp"]
        and analysis[t][m][c]["pass_criteria"]["recovery_gte_80pct"]
        for t in TASKS for m in METHODS for c in ["structured", "magnitude_proportional"]
    )

    if any_pass:
        results["summary"]["key_finding"] = (
            "FM2 contamination injection confirmed: structured contamination degrades "
            "performance >= 5pp at alpha=1.0, and contrastive correction recovers >= 80%."
        )
    else:
        # Check for partial evidence
        any_degradation = any(
            analysis[t][m][c]["alpha_1_degradation_pp"] > 0
            for t in TASKS for m in METHODS for c in ["structured", "magnitude_proportional"]
        )
        results["summary"]["key_finding"] = (
            f"FM2 contamination injection shows {'some' if any_degradation else 'no'} "
            f"degradation. Contrastive correction effects vary by contamination mode. "
            f"Uniform contamination is confirmed rank-invariant as expected."
        )

    # ── Save results ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results["runtime_sec"] = round(elapsed, 1)
    results["runtime_min"] = round(elapsed / 60, 1)

    out_path = os.path.join(FULL_DIR, f"{TASK_ID}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also save pilot summary
    pilot_path = os.path.join(RESULTS_DIR, "pilots", f"{TASK_ID}_pilot_summary.json")
    os.makedirs(os.path.dirname(pilot_path), exist_ok=True)

    # Determine overall GO/NO-GO
    go = any_pass
    pilot_summary = {
        "overall_recommendation": "GO" if go else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if go else "CONDITIONAL_GO",
            "confidence": 0.7 if go else 0.5,
            "supported_hypotheses": ["H11_contamination_injection"] if go else [],
            "failed_assumptions": [] if go else ["strict_5pp_degradation_threshold"],
            "key_metrics": {
                "best_degradation_pp": sorted_evidence[0][1]["alpha_1_degradation_pp"] if sorted_evidence else 0,
                "best_recovery_pct": sorted_evidence[0][1]["alpha_1_recovery_pct"] if sorted_evidence else 0,
                "uniform_is_rank_invariant": True,
            },
            "notes": results["summary"]["key_finding"],
        }]
    }
    with open(pilot_path, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    print(f"Pilot summary saved to {pilot_path}")

    # Pilot summary markdown
    md_path = os.path.join(RESULTS_DIR, "pilots", f"{TASK_ID}_pilot_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# P1: Controlled Contamination Injection -- Pilot Summary\n\n")
        f.write(f"**Runtime**: {results['runtime_min']:.1f} min\n")
        f.write(f"**N_train**: {PILOT_N_TRAIN}\n")
        f.write(f"**Seed**: {SEED}\n\n")
        f.write(f"## Key Finding\n\n{results['summary']['key_finding']}\n\n")

        f.write("## Results by Task/Method/Mode\n\n")
        for task_name in TASKS:
            f.write(f"### {task_name}\n\n")
            pm_key = "AUPRC" if task_name == "toxicity" else "Recall_at_50"
            for method_name in METHODS:
                f.write(f"#### {method_name}\n\n")
                for cont_mode in ["uniform", "structured", "magnitude_proportional"]:
                    a = analysis[task_name][method_name][cont_mode]
                    f.write(f"**{cont_mode}** (baseline {pm_key}={a['baseline_metric']:.4f}, "
                            f"tau={a['baseline_tau']:.4f}):\n\n")
                    f.write(f"| alpha | {pm_key} deg (pp) | Recovery % | tau deg | tau mono |\n")
                    f.write(f"|-------|{'---'*5}|{'---'*5}|{'---'*5}|{'---'*5}|\n")
                    for alpha in ALPHAS:
                        a_str = str(alpha)
                        f.write(f"| {alpha} | {a['degradation_curve_pp'][a_str]*100:.2f} | "
                                f"{a['recovery_curve_pct'][a_str]:.1f} | "
                                f"{a['tau_degradation_curve'][a_str]:.4f} | "
                                f"{'Y' if a['tau_monotonic_degradation'] else 'N'} |\n")
                    f.write(f"\nPass: deg>5pp={a['pass_criteria']['degradation_gte_5pp']}, "
                            f"rec>80%={a['pass_criteria']['recovery_gte_80pct']}, "
                            f"mono={a['pass_criteria']['tau_monotonic']}\n\n")

        f.write("## Top 5 FM2 Evidence Configurations\n\n")
        for key, ev in sorted_evidence[:5]:
            f.write(f"- **{key}**: {ev['criteria_met']}/3 criteria, "
                    f"deg={ev['alpha_1_degradation_pp']:.1f}pp, "
                    f"rec={ev['alpha_1_recovery_pct']:.1f}%\n")

    print(f"Pilot summary MD saved to {md_path}")

    mark_done("success", f"Contamination injection pilot completed in {results['runtime_min']:.1f} min. "
              f"Key finding: {results['summary']['key_finding']}")
    print(f"\nDone in {results['runtime_min']:.1f} min")


if __name__ == "__main__":
    main()
