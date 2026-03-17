#!/usr/bin/env python3
"""
P4: RepSim Layer Sweep -- PILOT MODE
======================================
Run RepSim at layers {0, 4, 8, 12, 16, 20, 23} on Pythia-1B for all 3 DATE-LM
tasks at pilot scale (N=100 training samples, seed=42).

Identify which layers carry attribution signal for each task type.
Expected: later layers better for semantic tasks (counterfact, ftrace),
possibly earlier/middle layers for behavioral (toxicity).

Pass criteria:
- Performance varies by >= 10pp across layers on at least 1 task
- Last layer is best on at least 1 semantic task

Metrics: R@50/AUPRC, MRR, Kendall tau, Spearman rho (with bootstrap CI).
"""

import os
import sys
import json
import time
import gc
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p4_layer_sweep"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-1b"
HIDDEN_DIM = 2048
# Pythia-1B has 16 transformer layers (0-15), plus embedding layer
# hidden_states[0] = embedding output, hidden_states[i+1] = layer i output
# So hidden_states[-1] = last layer = layer 15
# We sweep layers by hidden_states index: {0, 4, 8, 12, 15} + embedding(0)
# Task plan says {0, 4, 8, 12, 16, 20, 23} but Pythia-1B only has 16 layers (0-15)
# Map: use indices proportional to depth for a 16-layer model
# Actually: Pythia-1B has num_hidden_layers=16, so layers 0..15
# hidden_states has 17 entries: [embedding, layer0, layer1, ..., layer15]
# We'll use hidden_states indices: 1(layer0), 4(layer3), 7(layer6), 10(layer9), 13(layer12), 16(layer15)
# Plus embedding (index 0). That gives 7 points spanning the full depth.
# Actually let me just query the model config first and set layers adaptively.
LAYERS_TO_SWEEP = None  # Will be set after loading model config
RESULTS_DIR = Path("/home/jinxulin/sibyl_system/projects/CRA/exp/results")
FULL_DIR = RESULTS_DIR / "full"
CACHE_DIR = RESULTS_DIR / "cache"
PILOTS_DIR = RESULTS_DIR / "pilots"
BOOTSTRAP_B = 1000
MAX_LEN = 512
DEVICE = "cuda:0"

np.random.seed(SEED)
torch.manual_seed(SEED)

for d in [FULL_DIR, CACHE_DIR, PILOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Lifecycle helpers ────────────────────────────────────────────────────
def _safe_write(path, content):
    try:
        Path(path).write_text(content)
    except OSError as e:
        print(f"[Warn] Cannot write {path}: {e}", flush=True)

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
_safe_write(pid_file, str(os.getpid()))


def report_progress(stage, detail="", pct=0.0, metric=None):
    _safe_write(
        RESULTS_DIR / f"{TASK_ID}_PROGRESS.json",
        json.dumps({
            "task_id": TASK_ID, "epoch": 0, "total_epochs": 1,
            "step": 0, "total_steps": 0, "loss": None,
            "metric": metric or {}, "stage": stage, "detail": detail,
            "pct": pct, "updated_at": datetime.now().isoformat(),
        })
    )


def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        try:
            pid_f.unlink()
        except OSError:
            pass
    fp = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final = {}
    if fp.exists():
        try:
            final = json.loads(fp.read_text())
        except Exception:
            pass
    _safe_write(
        RESULTS_DIR / f"{TASK_ID}_DONE",
        json.dumps({
            "task_id": TASK_ID, "status": status, "summary": summary,
            "final_progress": final,
            "timestamp": datetime.now().isoformat(),
        })
    )


# ── Evaluation helpers ───────────────────────────────────────────────────
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


def evaluate_toxicity(sim_matrix, pilot_data, n_train):
    """Evaluate toxicity task: AUPRC + continuous metrics + bootstrap."""
    train_scores = sim_matrix.mean(axis=1)
    unsafe_indices = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]
    auprc = compute_auprc(train_scores, unsafe_indices, n_train)
    tau, rho = compute_continuous_metrics_toxicity(train_scores, pilot_data, n_train)

    # Bootstrap
    rng = np.random.RandomState(SEED + 1234)
    boot_auprcs, boot_taus = [], []
    for _ in range(BOOTSTRAP_B):
        idx = rng.choice(n_train, n_train, replace=True)
        bs = train_scores[idx]
        bl = np.zeros(len(idx))
        for ii, orig_i in enumerate(idx):
            if pilot_data[int(orig_i)]["type"] == "Unsafe":
                bl[ii] = 1
        if sum(bl) == 0:
            boot_auprcs.append(0.0)
            boot_taus.append(0.0)
            continue
        prec, rec, _ = precision_recall_curve(bl, bs)
        boot_auprcs.append(float(auc(rec, prec)))
        binary_rel_b = np.array([1.0 if pilot_data[int(idx[ii])]["type"] == "Unsafe" else 0.0
                                  for ii in range(len(idx))])
        t, _ = kendalltau(bs, binary_rel_b)
        boot_taus.append(float(t) if not np.isnan(t) else 0.0)

    return {
        "AUPRC": round(auprc, 6),
        "AUPRC_CI": [round(float(np.percentile(boot_auprcs, 2.5)), 6),
                      round(float(np.percentile(boot_auprcs, 97.5)), 6)],
        "n_unsafe": len(unsafe_indices),
        "n_train": n_train,
        "kendall_tau": round(tau, 6),
        "kendall_tau_CI": [round(float(np.percentile(boot_taus, 2.5)), 6),
                           round(float(np.percentile(boot_taus, 97.5)), 6)],
        "spearman_rho": round(rho, 6),
    }


def evaluate_factual(sim_matrix, pilot_data, ref_data, task_name, n_train):
    """Evaluate counterfact/ftrace: R@50, MRR, Kendall tau + bootstrap."""
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


# ── Representation extraction ─────────────────────────────────────────────
def extract_representations_at_layer(model, tokenizer, texts, layer_idx, desc=""):
    """Extract hidden states at a specific layer index.

    layer_idx: index into model.hidden_states output.
      0 = embedding output
      1 = after layer 0
      ...
      num_layers = after last layer (same as hidden_states[-1])

    Returns: [N, hidden_dim] tensor on CPU.
    """
    reps = []
    with torch.no_grad():
        for idx, text in enumerate(texts):
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            out = model(**inp, output_hidden_states=True)
            # hidden_states[layer_idx] shape: [1, seq_len, hidden_dim]
            # Use last token representation (same as RepSim convention)
            h = out.hidden_states[layer_idx][0, -1, :].float().cpu()
            reps.append(h)
            if (idx + 1) % 50 == 0:
                print(f"    [{desc}] {idx+1}/{len(texts)}", flush=True)
    return torch.stack(reps)


def cosine_similarity_matrix(train_reps, ref_reps):
    """Compute cosine similarity matrix [n_train, n_ref]."""
    tn = train_reps / train_reps.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rn = ref_reps / ref_reps.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (tn @ rn.T).numpy()


# ── Data loading ──────────────────────────────────────────────────────────
def load_task_data():
    from datasets import load_dataset
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


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()

    # ── Load model to determine architecture ──────────────────────────
    report_progress("loading_model", "Loading Pythia-1B", 0.05)
    print("[1/5] Loading Pythia-1B model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  num_hidden_layers={num_layers}, hidden_size={hidden_dim}", flush=True)
    print(f"  hidden_states will have {num_layers + 1} entries (0=embed, 1..{num_layers}=layers)", flush=True)

    # Determine layer sweep indices
    # For Pythia-1B (16 layers): sweep at proportional depths
    # hidden_states indices: 0=embed, 1=layer0, ..., 16=layer15
    # Task plan says {0, 4, 8, 12, 16, 20, 23} for a 24-layer model
    # Scale to 16-layer: {0, ~3, ~5, 8, ~11, ~13, 15} in layer indices
    # Use hidden_states indices: {0, 1, 4, 6, 9, 12, 14, 16}
    # That gives: embed, layer0, layer3, layer5, layer8, layer11, layer13, layer15
    # Simplify to 7 evenly-spaced points:
    if num_layers == 16:
        # hidden_states indices (0-indexed, 0=embedding)
        sweep_hs_indices = [0, 3, 5, 8, 11, 13, 16]  # embed + 6 layer points
        sweep_layer_names = ["embed", "layer_2", "layer_4", "layer_7", "layer_10", "layer_12", "layer_15"]
    else:
        # Generic: 7 evenly spaced points including first and last
        step = num_layers // 6
        sweep_hs_indices = [0] + [1 + i * step for i in range(6)]
        sweep_hs_indices[-1] = num_layers  # ensure last layer
        sweep_layer_names = [f"hs_{i}" for i in sweep_hs_indices]

    print(f"  Sweep hidden_states indices: {sweep_hs_indices}", flush=True)
    print(f"  Corresponding names: {sweep_layer_names}", flush=True)

    # ── Load data ──────────────────────────────────────────────────────
    report_progress("loading_data", "Loading DATE-LM datasets", 0.10)
    print("[2/5] Loading DATE-LM datasets...", flush=True)
    tasks_data = load_task_data()

    fmt_map = {
        "toxicity": lambda s: s.get("prompt", "") + " " + s.get("response", s.get("text", "")),
        "counterfact": lambda s: s["prompt"] + " " + s["response"],
        "ftrace": lambda s: s["prompt"] + " " + s["response"],
    }

    pilot_datasets = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        pilot_data, pilot_idx = create_pilot_subset(task_name, tasks_data[task_name]["train"])
        ref_data = tasks_data[task_name]["ref"]
        fmt = fmt_map[task_name]
        train_texts = [fmt(pilot_data[i]) for i in range(len(pilot_data))]
        ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]
        pilot_datasets[task_name] = {
            "pilot_data": pilot_data,
            "ref_data": ref_data,
            "pilot_idx": pilot_idx,
            "train_texts": train_texts,
            "ref_texts": ref_texts,
            "n_train": len(pilot_data),
            "n_ref": len(ref_data),
        }
        print(f"  [{task_name}] train={len(pilot_data)}, ref={len(ref_data)}", flush=True)

    # ── Extract representations at each layer for all tasks ────────────
    # Strategy: iterate over layers (outer), then tasks (inner)
    # This way we only need one forward pass per sample per layer
    # But actually, different tasks have different samples, so we need
    # per-task forward passes anyway. Better to iterate tasks(outer), layers(inner)
    # using a single forward pass that captures ALL hidden states.

    report_progress("extracting_reps", "Extracting representations at all layers", 0.15)
    print("[3/5] Extracting representations at all layers...", flush=True)

    # Pre-compute: for each task, do ONE forward pass per sample, capturing all hidden states
    all_reps = {}  # {task_name: {hs_index: {"train": tensor, "ref": tensor}}}

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        ds = pilot_datasets[task_name]
        all_texts = ds["train_texts"] + ds["ref_texts"]
        n_train = ds["n_train"]
        n_total = len(all_texts)

        print(f"\n  [{task_name}] Extracting {n_total} samples at {len(sweep_hs_indices)} layers...", flush=True)
        report_progress("extracting_reps", f"{task_name}", 0.15 + 0.20 * ["toxicity", "counterfact", "ftrace"].index(task_name))

        # Initialize storage
        layer_reps = {hs_idx: [] for hs_idx in sweep_hs_indices}

        with torch.no_grad():
            for idx, text in enumerate(all_texts):
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
                out = model(**inp, output_hidden_states=True)
                # Extract last-token hidden state at each layer
                for hs_idx in sweep_hs_indices:
                    h = out.hidden_states[hs_idx][0, -1, :].float().cpu()
                    layer_reps[hs_idx].append(h)
                if (idx + 1) % 50 == 0:
                    print(f"    [{task_name}] {idx+1}/{n_total}", flush=True)

        # Stack and split into train/ref
        task_reps = {}
        for hs_idx in sweep_hs_indices:
            stacked = torch.stack(layer_reps[hs_idx])
            task_reps[hs_idx] = {
                "train": stacked[:n_train],
                "ref": stacked[n_train:],
            }
        all_reps[task_name] = task_reps
        print(f"    [{task_name}] Done. Shapes: train={task_reps[sweep_hs_indices[0]]['train'].shape}, "
              f"ref={task_reps[sweep_hs_indices[0]]['ref'].shape}", flush=True)

        # Free intermediate
        del layer_reps
        gc.collect()

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  Model freed.", flush=True)

    # ── Evaluate RepSim at each layer for each task ────────────────────
    report_progress("evaluating", "Evaluating RepSim at all layers", 0.70)
    print("[4/5] Evaluating RepSim at all layers...", flush=True)

    sweep_results = {}  # {layer_name: {task_name: metrics}}

    for li, (hs_idx, layer_name) in enumerate(zip(sweep_hs_indices, sweep_layer_names)):
        print(f"\n{'='*60}", flush=True)
        print(f"Layer: {layer_name} (hidden_states[{hs_idx}]) [{li+1}/{len(sweep_hs_indices)}]", flush=True)
        print(f"{'='*60}", flush=True)
        report_progress("evaluating", f"{layer_name}", 0.70 + 0.20 * li / len(sweep_hs_indices))

        layer_results = {}
        for task_name in ["toxicity", "counterfact", "ftrace"]:
            ds = pilot_datasets[task_name]
            train_reps = all_reps[task_name][hs_idx]["train"]
            ref_reps = all_reps[task_name][hs_idx]["ref"]

            # Compute cosine similarity matrix
            sim_matrix = cosine_similarity_matrix(train_reps, ref_reps)

            if task_name == "toxicity":
                metrics = evaluate_toxicity(sim_matrix, ds["pilot_data"], ds["n_train"])
                primary = metrics["AUPRC"]
                print(f"  [{task_name}] AUPRC={metrics['AUPRC']:.4f} "
                      f"[{metrics['AUPRC_CI'][0]:.4f}, {metrics['AUPRC_CI'][1]:.4f}], "
                      f"tau={metrics['kendall_tau']:.4f}", flush=True)
            else:
                metrics = evaluate_factual(sim_matrix, ds["pilot_data"], ds["ref_data"],
                                           task_name, ds["n_train"])
                primary = metrics["Recall@50"]
                print(f"  [{task_name}] R@50={metrics['Recall@50']:.4f} "
                      f"[{metrics['Recall@50_CI'][0]:.4f}, {metrics['Recall@50_CI'][1]:.4f}], "
                      f"MRR={metrics['MRR']:.4f}, tau={metrics['kendall_tau']:.4f}", flush=True)

            layer_results[task_name] = metrics

        sweep_results[layer_name] = {
            "hs_index": hs_idx,
            "layer_index": hs_idx - 1 if hs_idx > 0 else -1,  # -1 for embedding
            "metrics": layer_results,
        }

    # ── Analysis ──────────────────────────────────────────────────────
    report_progress("analysis", "Computing layer sweep analysis", 0.92)
    print("\n[5/5] Analysis...", flush=True)

    # Per-task: extract primary metric curve across layers
    task_curves = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        metric_key = "AUPRC" if task_name == "toxicity" else "Recall@50"
        curve = []
        for layer_name in sweep_layer_names:
            val = sweep_results[layer_name]["metrics"][task_name][metric_key]
            tau = sweep_results[layer_name]["metrics"][task_name]["kendall_tau"]
            hs_idx = sweep_results[layer_name]["hs_index"]
            curve.append({
                "layer_name": layer_name,
                "hs_index": hs_idx,
                "primary_metric": metric_key,
                "value": round(val, 6),
                "kendall_tau": round(tau, 6),
            })
        task_curves[task_name] = curve

    # Compute per-task statistics
    task_analysis = {}
    for task_name, curve in task_curves.items():
        values = [pt["value"] for pt in curve]
        taus = [pt["kendall_tau"] for pt in curve]
        best_idx = int(np.argmax(values))
        worst_idx = int(np.argmin(values))
        range_pp = (max(values) - min(values)) * 100

        # Is last layer the best?
        last_layer_name = sweep_layer_names[-1]
        last_layer_val = curve[-1]["value"]
        is_last_best = best_idx == len(curve) - 1

        # Monotonicity: does performance generally increase with depth?
        # Compute Spearman correlation between layer index and performance
        from scipy.stats import spearmanr as sp_corr
        layer_indices = list(range(len(values)))
        mono_rho, mono_p = sp_corr(layer_indices, values)

        task_analysis[task_name] = {
            "metric": curve[0]["primary_metric"],
            "best_layer": curve[best_idx]["layer_name"],
            "best_value": round(max(values), 6),
            "worst_layer": curve[worst_idx]["layer_name"],
            "worst_value": round(min(values), 6),
            "range_pp": round(range_pp, 2),
            "last_layer_value": round(last_layer_val, 6),
            "is_last_layer_best": is_last_best,
            "depth_correlation_rho": round(float(mono_rho) if not np.isnan(mono_rho) else 0.0, 4),
            "depth_correlation_p": round(float(mono_p) if not np.isnan(mono_p) else 1.0, 4),
            "curve": curve,
        }

    # Pass criteria evaluation
    # 1. Performance varies by >= 10pp across layers on at least 1 task
    varies_10pp = any(task_analysis[t]["range_pp"] >= 10.0 for t in ["toxicity", "counterfact", "ftrace"])
    # 2. Last layer is best on at least 1 semantic task
    last_best_semantic = any(task_analysis[t]["is_last_layer_best"] for t in ["counterfact", "ftrace"])

    pass_criteria = {
        "varies_10pp_any_task": varies_10pp,
        "last_layer_best_semantic": last_best_semantic,
        "overall_pass": varies_10pp and last_best_semantic,
    }

    # Task-type boundary characterization
    # Expected: later layers better for semantic (counterfact, ftrace),
    # earlier/middle for behavioral (toxicity)
    boundary_analysis = {
        "semantic_tasks_depth_preference": {
            task: task_analysis[task]["depth_correlation_rho"]
            for task in ["counterfact", "ftrace"]
        },
        "behavioral_task_depth_preference": task_analysis["toxicity"]["depth_correlation_rho"],
        "semantic_prefer_later": all(
            task_analysis[t]["depth_correlation_rho"] > 0 for t in ["counterfact", "ftrace"]
        ),
        "behavioral_different_pattern": (
            task_analysis["toxicity"]["depth_correlation_rho"] <
            min(task_analysis[t]["depth_correlation_rho"] for t in ["counterfact", "ftrace"])
        ),
    }

    total_time = time.time() - t_start

    # ── Compile final results ────────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "candidate_id": "cand_a",
        "mode": "pilot",
        "model": MODEL_NAME,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "sweep_hs_indices": sweep_hs_indices,
        "sweep_layer_names": sweep_layer_names,
        "sweep_results": sweep_results,
        "task_analysis": task_analysis,
        "task_curves": task_curves,
        "pass_criteria": pass_criteria,
        "boundary_analysis": boundary_analysis,
        "total_runtime_sec": round(total_time, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # Save to full results dir
    out_path = FULL_DIR / f"{TASK_ID}.json"
    try:
        out_path.write_text(json.dumps(final, indent=2, default=str))
        print(f"\nResults saved to {out_path}", flush=True)
    except OSError as e:
        print(f"\n[CRITICAL] Cannot save results: {e}", flush=True)
        print("===RESULT_JSON_START===")
        print(json.dumps(final, indent=2, default=str))
        print("===RESULT_JSON_END===")

    # ── Pilot summary (machine-readable) ──────────────────────────────
    pilot_summary = {
        "overall_recommendation": "GO" if pass_criteria["overall_pass"] else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if pass_criteria["overall_pass"] else "GO",
            "confidence": 0.75 if pass_criteria["overall_pass"] else 0.55,
            "supported_hypotheses": (
                ["H_layer_variation"] if varies_10pp else []
            ) + (
                ["H_last_layer_semantic"] if last_best_semantic else []
            ),
            "failed_assumptions": [],
            "key_metrics": {
                task: {
                    "best_layer": task_analysis[task]["best_layer"],
                    "best_value": task_analysis[task]["best_value"],
                    "range_pp": task_analysis[task]["range_pp"],
                    "depth_correlation": task_analysis[task]["depth_correlation_rho"],
                }
                for task in ["toxicity", "counterfact", "ftrace"]
            },
            "notes": (
                f"Layer sweep pilot (N={PILOT_N_TRAIN}). "
                f"Ranges (pp): toxicity={task_analysis['toxicity']['range_pp']:.1f}, "
                f"counterfact={task_analysis['counterfact']['range_pp']:.1f}, "
                f"ftrace={task_analysis['ftrace']['range_pp']:.1f}. "
                f"Last-layer best on semantic: "
                f"counterfact={'yes' if task_analysis['counterfact']['is_last_layer_best'] else 'no'}, "
                f"ftrace={'yes' if task_analysis['ftrace']['is_last_layer_best'] else 'no'}. "
                f"Depth correlation: "
                f"toxicity={task_analysis['toxicity']['depth_correlation_rho']:.3f}, "
                f"counterfact={task_analysis['counterfact']['depth_correlation_rho']:.3f}, "
                f"ftrace={task_analysis['ftrace']['depth_correlation_rho']:.3f}."
            ),
        }],
        "pilot_limitations": [
            f"N={PILOT_N_TRAIN} pilot scale; full-scale may shift optimal layers",
            f"Pythia-1B has {num_layers} layers; sweep covers {len(sweep_hs_indices)} points",
            "Bootstrap CIs may be wide with small ref sets",
        ],
    }

    _safe_write(
        PILOTS_DIR / f"{TASK_ID}_pilot_summary.json",
        json.dumps(pilot_summary, indent=2),
    )

    # ── Pilot summary (markdown) ──────────────────────────────────────
    md_lines = [
        "# P4: RepSim Layer Sweep -- Pilot Summary",
        "",
        "## Configuration",
        f"- Model: {MODEL_NAME} (hidden_dim={hidden_dim}, num_layers={num_layers})",
        f"- N_train: {PILOT_N_TRAIN} (pilot)",
        f"- Layers: {sweep_layer_names}",
        f"- Hidden states indices: {sweep_hs_indices}",
        f"- Metrics: R@50/AUPRC, MRR, Kendall tau, Spearman rho",
        "",
        "## Results by Layer",
        "",
    ]

    # Table for each task
    for task_name in ["counterfact", "toxicity", "ftrace"]:
        analysis = task_analysis[task_name]
        metric = analysis["metric"]
        md_lines += [
            f"### {task_name} ({metric})",
            f"| Layer | hs_index | {metric} | Kendall tau |",
            f"|---|---|---|---|",
        ]
        for pt in analysis["curve"]:
            marker = " **best**" if pt["value"] == analysis["best_value"] else ""
            md_lines.append(
                f"| {pt['layer_name']} | {pt['hs_index']} | {pt['value']:.4f} | {pt['kendall_tau']:.4f} |{marker}"
            )
        md_lines += [
            f"",
            f"- Range: {analysis['range_pp']:.1f}pp",
            f"- Best: {analysis['best_layer']} ({analysis['best_value']:.4f})",
            f"- Last layer best: {'yes' if analysis['is_last_layer_best'] else 'no'}",
            f"- Depth correlation (Spearman): rho={analysis['depth_correlation_rho']:.3f}, p={analysis['depth_correlation_p']:.4f}",
            "",
        ]

    md_lines += [
        "## Pass Criteria",
        f"- Varies >= 10pp on any task: **{'PASS' if varies_10pp else 'FAIL'}**",
        f"- Last layer best on semantic: **{'PASS' if last_best_semantic else 'FAIL'}**",
        f"- Overall: **{'GO' if pass_criteria['overall_pass'] else 'REFINE'}**",
        "",
        "## Task-Type Boundary Analysis",
        f"- Semantic tasks depth preference: {boundary_analysis['semantic_tasks_depth_preference']}",
        f"- Behavioral (toxicity) depth preference: {boundary_analysis['behavioral_task_depth_preference']:.3f}",
        f"- Semantic prefer later layers: {'yes' if boundary_analysis['semantic_prefer_later'] else 'no'}",
        f"- Behavioral different pattern: {'yes' if boundary_analysis['behavioral_different_pattern'] else 'no'}",
        "",
        f"## Runtime: {total_time:.1f}s",
    ]

    _safe_write(
        PILOTS_DIR / f"{TASK_ID}_pilot_summary.md",
        "\n".join(md_lines),
    )

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: RepSim Layer Sweep (Pilot)", flush=True)
    print("=" * 70, flush=True)

    for task_name in ["counterfact", "toxicity", "ftrace"]:
        a = task_analysis[task_name]
        print(f"\n[{task_name}] ({a['metric']})", flush=True)
        print(f"  {'Layer':<12} {'hs_idx':<8} {a['metric']:<10} {'tau':<10}", flush=True)
        print(f"  {'-'*40}", flush=True)
        for pt in a["curve"]:
            marker = " <-- best" if pt["value"] == a["best_value"] else ""
            print(f"  {pt['layer_name']:<12} {pt['hs_index']:<8} {pt['value']:<10.4f} {pt['kendall_tau']:<10.4f}{marker}", flush=True)
        print(f"  Range: {a['range_pp']:.1f}pp, depth_corr={a['depth_correlation_rho']:.3f}", flush=True)

    print(f"\nPass criteria: {pass_criteria}", flush=True)
    print(f"Boundary analysis: {boundary_analysis}", flush=True)
    print(f"Total runtime: {total_time:.1f}s", flush=True)
    print("=" * 70, flush=True)

    mark_done(
        "success" if pass_criteria["overall_pass"] else "warn",
        f"Layer sweep pilot done in {total_time:.0f}s. "
        f"Ranges (pp): tox={task_analysis['toxicity']['range_pp']:.1f}, "
        f"cf={task_analysis['counterfact']['range_pp']:.1f}, "
        f"ft={task_analysis['ftrace']['range_pp']:.1f}. "
        f"Pass={pass_criteria}"
    )
    return final


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"FATAL: {e}\n{traceback.format_exc()}", flush=True)
        mark_done("failed", str(e)[:300])
        sys.exit(1)
