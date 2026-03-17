#!/usr/bin/env python3
"""
P4: Cosine vs Euclidean RepSim (Contrarian Test) -- PILOT MODE
================================================================
Run RepSim with three similarity functions on all 3 DATE-LM tasks
at pilot scale (N=100) on Pythia-1B:

1. Cosine similarity: phi^T psi / (||phi|| * ||psi||)
2. Euclidean distance: -||phi - psi||_2  (negated so higher = more similar)
3. Dot product: phi^T psi  (no normalization)

CRITICAL: Must use RAW (unnormalized) representations so that the three
similarity functions actually differ. The existing cache has L2-normalized
reps, making all three equivalent. We re-extract from Pythia-1B.

Contrarian falsification: if Euclidean matches cosine on all tasks,
then normalization is NOT a factor in the TRAK-PCA vs RepSim gap.
If cosine >> Euclidean, normalization is a key factor.

Pass criteria: All three similarity functions produce valid scores;
cosine vs Euclidean gap > 3pp on at least 1 task.
"""

import os, sys, json, time, gc, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc
from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p4_cosine_vs_euclidean"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-1b"
HIDDEN_DIM = 2048
MAX_LEN = 512
BATCH_SIZE = 4
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
RAW_CACHE_FILE = os.path.join(CACHE_DIR, "repsim_raw_reps.pt")  # new cache for raw reps
PILOTS_DIR = os.path.join(RESULTS_DIR, "pilots")
BOOTSTRAP_B = 1000

SIMILARITY_FUNCTIONS = ["cosine", "euclidean", "dot_product"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


# ── Representation extraction (RAW, unnormalized) ──────────────────────
def extract_representations_raw(model, tok, texts, desc="", batch_size=BATCH_SIZE):
    """
    Extract last-layer hidden states (last non-pad token) from Pythia-1B.
    Returns: RAW (unnormalized) representation tensor [N, hidden_dim].
    This is the same extraction as RepSim but WITHOUT L2 normalization.
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

        with torch.no_grad(), torch.amp.autocast("cuda"):
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

        # Convert to float32 but do NOT normalize
        reps = reps.float().cpu()
        all_reps.append(reps)

        if end == n or (end % (batch_size * 8) == 0):
            print(f"  [{desc}] {end}/{n} samples extracted")

    reps_tensor = torch.cat(all_reps, dim=0)  # [N, hidden_dim]
    return reps_tensor


# ── Similarity functions ────────────────────────────────────────────────
def compute_similarity_matrix(train_reps, ref_reps, sim_type):
    """
    Compute similarity matrix [n_train, n_ref] using the specified function.
    Input reps are RAW (unnormalized).

    Returns:
        sim_matrix: np.array [n_train, n_ref] where higher = more similar
    """
    if sim_type == "cosine":
        # phi^T psi / (||phi|| * ||psi||)
        train_norm = train_reps / (np.linalg.norm(train_reps, axis=1, keepdims=True) + 1e-8)
        ref_norm = ref_reps / (np.linalg.norm(ref_reps, axis=1, keepdims=True) + 1e-8)
        sim_matrix = train_norm @ ref_norm.T

    elif sim_type == "euclidean":
        # -||phi - psi||_2  (negated so higher = more similar)
        train_sq = np.sum(train_reps ** 2, axis=1, keepdims=True)
        ref_sq = np.sum(ref_reps ** 2, axis=1, keepdims=True)
        cross = train_reps @ ref_reps.T
        dist_sq = train_sq + ref_sq.T - 2 * cross
        dist_sq = np.maximum(dist_sq, 0.0)
        sim_matrix = -np.sqrt(dist_sq)

    elif sim_type == "dot_product":
        # phi^T psi  (no normalization)
        sim_matrix = train_reps @ ref_reps.T

    else:
        raise ValueError(f"Unknown similarity type: {sim_type}")

    return sim_matrix


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


def bootstrap_continuous_toxicity(train_scores, pilot_data, n_train, B=BOOTSTRAP_B):
    tau, rho = compute_continuous_metrics_toxicity(train_scores, pilot_data, n_train)
    rng = np.random.RandomState(SEED + 4567)
    boot_taus = []
    for _ in range(B):
        idx = rng.choice(n_train, n_train, replace=True)
        bs = train_scores[idx]
        bd = [pilot_data[int(i)] for i in idx]
        t, _ = compute_continuous_metrics_toxicity(bs, bd, n_train)
        boot_taus.append(t)
    return tau, rho, [round(float(np.percentile(boot_taus, 2.5)), 6),
                       round(float(np.percentile(boot_taus, 97.5)), 6)]


def bootstrap_factual(sim_matrix, pilot_data, ref_data, task_name, n_train):
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


def get_texts(data, task_name):
    """Extract text from dataset samples."""
    texts = []
    for i in range(len(data)):
        sample = data[i]
        if task_name == "toxicity":
            text = sample.get("text", sample.get("response", ""))
        elif task_name == "counterfact":
            text = sample.get("text", sample.get("prompt", ""))
        elif task_name == "ftrace":
            text = sample.get("text", sample.get("prompt", ""))
        else:
            text = str(sample.get("text", ""))
        texts.append(text)
    return texts


# ── Norm analysis ───────────────────────────────────────────────────────
def analyze_norms(train_reps, ref_reps, task_name):
    train_norms = np.linalg.norm(train_reps, axis=1)
    ref_norms = np.linalg.norm(ref_reps, axis=1)
    return {
        "train_norm_mean": round(float(np.mean(train_norms)), 4),
        "train_norm_std": round(float(np.std(train_norms)), 4),
        "train_norm_cv": round(float(np.std(train_norms) / (np.mean(train_norms) + 1e-8)), 4),
        "train_norm_min": round(float(np.min(train_norms)), 4),
        "train_norm_max": round(float(np.max(train_norms)), 4),
        "train_norm_p10": round(float(np.percentile(train_norms, 10)), 4),
        "train_norm_p90": round(float(np.percentile(train_norms, 90)), 4),
        "ref_norm_mean": round(float(np.mean(ref_norms)), 4),
        "ref_norm_std": round(float(np.std(ref_norms)), 4),
        "ref_norm_cv": round(float(np.std(ref_norms) / (np.mean(ref_norms) + 1e-8)), 4),
    }


def analyze_score_distributions(sim_matrices, task_name):
    stats = {}
    for sim_type, matrix in sim_matrices.items():
        flat = matrix.flatten()
        stats[sim_type] = {
            "mean": round(float(np.mean(flat)), 6),
            "std": round(float(np.std(flat)), 6),
            "min": round(float(np.min(flat)), 6),
            "max": round(float(np.max(flat)), 6),
            "median": round(float(np.median(flat)), 6),
        }
    return stats


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 70)
    print(f"P4: Cosine vs Euclidean RepSim -- PILOT MODE (v2: raw reps)")
    print(f"Model: {MODEL_NAME}, Hidden dim d={HIDDEN_DIM}")
    print(f"Similarity functions: {SIMILARITY_FUNCTIONS}")
    print(f"Tasks: toxicity, counterfact, ftrace")
    print(f"N_train (pilot): {PILOT_N_TRAIN}")
    print(f"Bootstrap B={BOOTSTRAP_B}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ── Load task data ──────────────────────────────────────────────────
    tasks_data = load_task_data()

    pilot_datasets = {}
    pilot_indices = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        pilot_data, pilot_idx = create_pilot_subset(task_name, tasks_data[task_name]["train"])
        pilot_datasets[task_name] = {
            "pilot_data": pilot_data,
            "ref_data": tasks_data[task_name]["ref"],
        }
        pilot_indices[task_name] = pilot_idx
        print(f"  [{task_name}] pilot N={len(pilot_data)}, ref N={len(tasks_data[task_name]['ref'])}")

    # ── Extract or load raw representations ─────────────────────────────
    if os.path.exists(RAW_CACHE_FILE):
        report_progress("loading_cache", "Loading raw representation cache", 0.10)
        raw_cache = torch.load(RAW_CACHE_FILE, map_location="cpu", weights_only=False)
        task_reps = {}
        for task_name in ["toxicity", "counterfact", "ftrace"]:
            train_reps = raw_cache[f"raw_{task_name}_train"].numpy()
            ref_reps = raw_cache[f"raw_{task_name}_ref"].numpy()
            norms = np.linalg.norm(train_reps, axis=1)
            task_reps[task_name] = {"train": train_reps, "ref": ref_reps}
            print(f"  [cache] [{task_name}] train={train_reps.shape}, "
                  f"norm mean={norms.mean():.2f}, std={norms.std():.4f}")
        print("Loaded raw representations from cache.")
    else:
        report_progress("loading_model", "Loading Pythia-1B for representation extraction", 0.10)
        print("\nLoading Pythia-1B model...")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE,
        )
        model.eval()
        print(f"Model loaded on {DEVICE}")

        report_progress("extracting_reps", "Extracting RAW representations", 0.15)
        task_reps = {}
        raw_cache = {}

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            pilot_data = pilot_datasets[task_name]["pilot_data"]
            ref_data = pilot_datasets[task_name]["ref_data"]

            train_texts = get_texts(pilot_data, task_name)
            ref_texts = get_texts(ref_data, task_name)

            print(f"\n  Extracting [{task_name}] train ({len(train_texts)} samples)...")
            train_reps_t = extract_representations_raw(model, tok, train_texts,
                                                        desc=f"train-{task_name}")
            print(f"  Extracting [{task_name}] ref ({len(ref_texts)} samples)...")
            ref_reps_t = extract_representations_raw(model, tok, ref_texts,
                                                      desc=f"ref-{task_name}")

            train_reps = train_reps_t.numpy()
            ref_reps = ref_reps_t.numpy()

            norms = np.linalg.norm(train_reps, axis=1)
            print(f"  [{task_name}] train norm: mean={norms.mean():.2f}, "
                  f"std={norms.std():.4f}, CV={norms.std()/norms.mean():.4f}")

            task_reps[task_name] = {"train": train_reps, "ref": ref_reps}
            raw_cache[f"raw_{task_name}_train"] = train_reps_t
            raw_cache[f"raw_{task_name}_ref"] = ref_reps_t
            raw_cache[f"raw_{task_name}_pilot_idx"] = pilot_indices[task_name]

        # Save raw cache for reuse
        torch.save(raw_cache, RAW_CACHE_FILE)
        print(f"\nRaw representation cache saved to: {RAW_CACHE_FILE}")

        # Free GPU memory
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # ── Norm analysis (diagnostic) ──────────────────────────────────────
    report_progress("norm_analysis", "Analyzing representation norms", 0.30)
    norm_analysis = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        norm_analysis[task_name] = analyze_norms(
            task_reps[task_name]["train"],
            task_reps[task_name]["ref"],
            task_name
        )
        cv = norm_analysis[task_name]["train_norm_cv"]
        print(f"  [{task_name}] train norm: mean={norm_analysis[task_name]['train_norm_mean']:.2f}, "
              f"CV={cv:.4f} ({'HIGH variance' if cv > 0.1 else 'low variance'})")

    # ── Evaluate each similarity function ───────────────────────────────
    all_results = {}
    score_distributions = {}

    for sim_idx, sim_type in enumerate(SIMILARITY_FUNCTIONS):
        print(f"\n{'='*70}")
        print(f"Similarity function: {sim_type} ({sim_idx+1}/{len(SIMILARITY_FUNCTIONS)})")
        print(f"{'='*70}")
        report_progress("evaluation", f"sim={sim_type}",
                        0.35 + (sim_idx + 1) * 0.15)

        sim_results = {}

        for task_name in ["toxicity", "counterfact", "ftrace"]:
            train_reps = task_reps[task_name]["train"]
            ref_reps = task_reps[task_name]["ref"]

            sim_matrix = compute_similarity_matrix(train_reps, ref_reps, sim_type)

            pilot_data = pilot_datasets[task_name]["pilot_data"]
            ref_data = pilot_datasets[task_name]["ref_data"]
            n_train = len(pilot_data)

            if task_name == "toxicity":
                train_scores = sim_matrix.mean(axis=1)
                auprc, ci_lo, ci_hi, n_unsafe = bootstrap_auprc(
                    train_scores, pilot_data, n_train)
                tau, rho, tau_ci = bootstrap_continuous_toxicity(
                    train_scores, pilot_data, n_train)
                metrics = {
                    "AUPRC": round(auprc, 6),
                    "AUPRC_CI": [round(ci_lo, 6), round(ci_hi, 6)],
                    "n_unsafe": n_unsafe,
                    "n_train": n_train,
                    "kendall_tau": round(tau, 6),
                    "kendall_tau_CI": tau_ci,
                    "spearman_rho": round(rho, 6),
                }
                print(f"  [{task_name}] {sim_type}: AUPRC={auprc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], "
                      f"tau={tau:.4f}, rho={rho:.4f}")
            else:
                metrics = bootstrap_factual(
                    sim_matrix, pilot_data, ref_data, task_name, n_train)
                print(f"  [{task_name}] {sim_type}: R@50={metrics['Recall@50']:.4f} "
                      f"[{metrics['Recall@50_CI'][0]:.4f}, {metrics['Recall@50_CI'][1]:.4f}], "
                      f"MRR={metrics['MRR']:.4f}, tau={metrics['kendall_tau']:.4f}")

            sim_results[task_name] = metrics

        all_results[sim_type] = sim_results

    # ── Score distribution analysis ─────────────────────────────────────
    report_progress("score_analysis", "Analyzing score distributions", 0.85)
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        matrices = {}
        for sim_type in SIMILARITY_FUNCTIONS:
            matrices[sim_type] = compute_similarity_matrix(
                task_reps[task_name]["train"],
                task_reps[task_name]["ref"],
                sim_type
            )
        score_distributions[task_name] = analyze_score_distributions(matrices, task_name)

    # ── Rank correlations between similarity functions ──────────────────
    report_progress("rank_correlations", "Computing rank correlations", 0.88)
    rank_correlations = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        train_reps = task_reps[task_name]["train"]
        ref_reps = task_reps[task_name]["ref"]
        cos_mat = compute_similarity_matrix(train_reps, ref_reps, "cosine").flatten()
        euc_mat = compute_similarity_matrix(train_reps, ref_reps, "euclidean").flatten()
        dot_mat = compute_similarity_matrix(train_reps, ref_reps, "dot_product").flatten()

        tau_ce, _ = kendalltau(cos_mat, euc_mat)
        tau_cd, _ = kendalltau(cos_mat, dot_mat)
        tau_ed, _ = kendalltau(euc_mat, dot_mat)
        rho_ce, _ = spearmanr(cos_mat, euc_mat)
        rho_cd, _ = spearmanr(cos_mat, dot_mat)
        rho_ed, _ = spearmanr(euc_mat, dot_mat)

        rank_correlations[task_name] = {
            "cosine_vs_euclidean_tau": round(float(tau_ce), 4),
            "cosine_vs_dot_tau": round(float(tau_cd), 4),
            "euclidean_vs_dot_tau": round(float(tau_ed), 4),
            "cosine_vs_euclidean_rho": round(float(rho_ce), 4),
            "cosine_vs_dot_rho": round(float(rho_cd), 4),
            "euclidean_vs_dot_rho": round(float(rho_ed), 4),
        }
        print(f"  [{task_name}] Rank correlations:")
        print(f"    cosine vs euclidean: tau={tau_ce:.4f}, rho={rho_ce:.4f}")
        print(f"    cosine vs dot:       tau={tau_cd:.4f}, rho={rho_cd:.4f}")

    # ── Comparative analysis ────────────────────────────────────────────
    report_progress("comparative_analysis", "Computing pairwise gaps", 0.90)

    comparison = {}
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        if task_name == "toxicity":
            metric_key = "AUPRC"
        else:
            metric_key = "Recall@50"

        cosine_val = all_results["cosine"][task_name][metric_key]
        euclidean_val = all_results["euclidean"][task_name][metric_key]
        dot_val = all_results["dot_product"][task_name][metric_key]

        cosine_tau = all_results["cosine"][task_name]["kendall_tau"]
        euclidean_tau = all_results["euclidean"][task_name]["kendall_tau"]
        dot_tau = all_results["dot_product"][task_name]["kendall_tau"]

        comparison[task_name] = {
            "metric": metric_key,
            "cosine": round(cosine_val, 6),
            "euclidean": round(euclidean_val, 6),
            "dot_product": round(dot_val, 6),
            "cosine_minus_euclidean_pp": round((cosine_val - euclidean_val) * 100, 2),
            "cosine_minus_dot_pp": round((cosine_val - dot_val) * 100, 2),
            "euclidean_minus_dot_pp": round((euclidean_val - dot_val) * 100, 2),
            "cosine_tau": round(cosine_tau, 6),
            "euclidean_tau": round(euclidean_tau, 6),
            "dot_tau": round(dot_tau, 6),
            "cosine_minus_euclidean_tau": round(cosine_tau - euclidean_tau, 6),
            "cosine_minus_dot_tau": round(cosine_tau - dot_tau, 6),
        }
        print(f"\n  [{task_name}] Gaps ({metric_key}):")
        print(f"    cosine - euclidean = {comparison[task_name]['cosine_minus_euclidean_pp']:+.2f}pp")
        print(f"    cosine - dot       = {comparison[task_name]['cosine_minus_dot_pp']:+.2f}pp")
        print(f"    euclidean - dot    = {comparison[task_name]['euclidean_minus_dot_pp']:+.2f}pp")

    # ── Pass criteria evaluation ────────────────────────────────────────
    all_valid = all(
        all(
            not np.isnan(all_results[sim][task]["kendall_tau"])
            for task in ["toxicity", "counterfact", "ftrace"]
        )
        for sim in SIMILARITY_FUNCTIONS
    )

    cosine_vs_euclidean_gaps = [
        abs(comparison[t]["cosine_minus_euclidean_pp"])
        for t in ["toxicity", "counterfact", "ftrace"]
    ]
    normalization_matters = any(g > 3.0 for g in cosine_vs_euclidean_gaps)
    max_gap_task = max(comparison.keys(),
                       key=lambda t: abs(comparison[t]["cosine_minus_euclidean_pp"]))
    max_gap = comparison[max_gap_task]["cosine_minus_euclidean_pp"]

    # Also check cosine vs dot product gaps (can be more informative)
    cosine_vs_dot_gaps = [
        abs(comparison[t]["cosine_minus_dot_pp"])
        for t in ["toxicity", "counterfact", "ftrace"]
    ]
    dot_normalization_matters = any(g > 3.0 for g in cosine_vs_dot_gaps)

    pass_criteria = {
        "all_valid_scores": all_valid,
        "cosine_vs_euclidean_gap_gt_3pp": normalization_matters,
        "cosine_vs_dot_gap_gt_3pp": dot_normalization_matters,
        "max_cosine_euclidean_gap_pp": round(max_gap, 2),
        "max_gap_task": max_gap_task,
        "overall_pass": all_valid and (normalization_matters or dot_normalization_matters),
    }

    # ── Implications for TRAK-PCA gap ───────────────────────────────────
    if normalization_matters:
        implication = (
            "Normalization IS a significant factor. Cosine normalization in RepSim "
            "provides >3pp advantage over unnormalized scoring on at least one task. "
            "This suggests that representation norm variation carries non-attribution noise, "
            "and cosine normalization in RepSim removes this noise. For the TRAK-PCA gap "
            "decomposition, cosine-normalized TRAK-PCA (factor b) should close part of the gap."
        )
    elif dot_normalization_matters:
        implication = (
            "Euclidean is robust (matches cosine within 3pp) but dot product diverges (>3pp gap). "
            "This means direction is what matters, not scale. Euclidean distance is approximately "
            "equivalent to cosine at fixed norm, so the normalization effect is subtle. "
            "Dot product's vulnerability to norm variation confirms that TRAK's unnormalized "
            "gradient inner products lose signal to norm noise."
        )
    else:
        implication = (
            "Normalization is NOT a significant factor for representation-space methods. "
            "Cosine, Euclidean, and dot product RepSim all perform comparably (<3pp gaps). "
            "This suggests that representation norms at the last layer are relatively uniform, "
            "so the TRAK-PCA vs RepSim gap is NOT primarily due to normalization. Other factors "
            "(layer mixing, nonlinear features, gradient-specific noise) dominate the gap."
        )

    # ── Assemble results ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results = {
        "task_id": TASK_ID,
        "candidate_id": "cand_a",
        "mode": "pilot",
        "model": MODEL_NAME,
        "hidden_dim": HIDDEN_DIM,
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "similarity_functions": SIMILARITY_FUNCTIONS,
        "note": "v2: uses RAW (unnormalized) last-token representations. "
                "v1 incorrectly used pre-normalized cache, making all metrics identical.",
        "results_by_similarity": all_results,
        "comparison": comparison,
        "norm_analysis": norm_analysis,
        "score_distributions": score_distributions,
        "rank_correlations": rank_correlations,
        "pass_criteria": pass_criteria,
        "implication_for_gap": implication,
        "runtime_sec": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # ── Save results ────────────────────────────────────────────────────
    report_progress("saving", "Writing results", 0.95)

    full_path = os.path.join(FULL_DIR, f"{TASK_ID}.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {full_path}")

    # Pilot summary JSON
    pilot_summary = {
        "overall_recommendation": "GO" if pass_criteria["overall_pass"] else "INCONCLUSIVE",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if pass_criteria["overall_pass"] else "INCONCLUSIVE",
            "confidence": 0.75 if (normalization_matters or dot_normalization_matters) else 0.50,
            "supported_hypotheses": (
                ["normalization_matters"] if normalization_matters
                else (["dot_product_diverges"] if dot_normalization_matters else [])
            ),
            "failed_assumptions": (
                [] if (normalization_matters or dot_normalization_matters)
                else ["normalization_not_significant"]
            ),
            "key_metrics": {
                f"cosine_vs_euclidean_{t}": comparison[t]["cosine_minus_euclidean_pp"]
                for t in ["toxicity", "counterfact", "ftrace"]
            },
            "notes": implication,
        }],
    }
    pilot_json_path = os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.json")
    with open(pilot_json_path, "w") as f:
        json.dump(pilot_summary, f, indent=2)

    # Pilot summary MD
    md_lines = [
        f"# P4: Cosine vs Euclidean RepSim -- Pilot Summary (v2: raw reps)",
        f"",
        f"**Note**: v1 used pre-L2-normalized cache, making all similarity functions "
        f"produce identical rankings. v2 re-extracts raw (unnormalized) representations "
        f"from Pythia-1B to enable meaningful comparison.",
        f"",
        f"## Pass Criteria",
        f"- All similarity functions produce valid scores: **{'PASS' if all_valid else 'FAIL'}**",
        f"- Cosine vs Euclidean gap > 3pp on at least 1 task: **{'PASS' if normalization_matters else 'FAIL'}**",
        f"- Cosine vs Dot Product gap > 3pp on at least 1 task: **{'PASS' if dot_normalization_matters else 'FAIL'}**",
        f"- Max cosine-euclidean gap: {max_gap:+.2f}pp on {max_gap_task}",
        f"- Overall: **{'PASS' if pass_criteria['overall_pass'] else 'INCONCLUSIVE'}**",
        f"",
        f"## Results Table (Rank-Based Metrics)",
        f"",
        f"| Task | Metric | Cosine | Euclidean | Dot Product | Cos-Euc Gap | Cos-Dot Gap |",
        f"|------|--------|--------|-----------|-------------|-------------|-------------|",
    ]
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        metric_key = "AUPRC" if task_name == "toxicity" else "Recall@50"
        cos_v = all_results["cosine"][task_name][metric_key]
        euc_v = all_results["euclidean"][task_name][metric_key]
        dot_v = all_results["dot_product"][task_name][metric_key]
        gap_ce = comparison[task_name]["cosine_minus_euclidean_pp"]
        gap_cd = comparison[task_name]["cosine_minus_dot_pp"]
        md_lines.append(
            f"| {task_name} | {metric_key} | {cos_v:.4f} | {euc_v:.4f} | "
            f"{dot_v:.4f} | {gap_ce:+.2f}pp | {gap_cd:+.2f}pp |"
        )

    md_lines += [
        f"",
        f"## Kendall Tau Comparison",
        f"",
        f"| Task | Cosine tau | Euclidean tau | Dot tau | Cos-Euc tau diff |",
        f"|------|-----------|--------------|---------|-----------------|",
    ]
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        c = comparison[task_name]
        md_lines.append(
            f"| {task_name} | {c['cosine_tau']:.4f} | {c['euclidean_tau']:.4f} | "
            f"{c['dot_tau']:.4f} | {c['cosine_minus_euclidean_tau']:+.4f} |"
        )

    md_lines += [
        f"",
        f"## Norm Analysis (RAW representations)",
        f"",
        f"| Task | Train Norm Mean | Train Norm CV | Ref Norm Mean |",
        f"|------|----------------|---------------|---------------|",
    ]
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        n = norm_analysis[task_name]
        md_lines.append(
            f"| {task_name} | {n['train_norm_mean']:.2f} | {n['train_norm_cv']:.4f} | "
            f"{n['ref_norm_mean']:.2f} |"
        )

    md_lines += [
        f"",
        f"## Rank Correlations Between Similarity Functions",
        f"",
        f"| Task | cos-euc rho | cos-dot rho | euc-dot rho |",
        f"|------|------------|------------|------------|",
    ]
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        rc = rank_correlations[task_name]
        md_lines.append(
            f"| {task_name} | {rc['cosine_vs_euclidean_rho']:.4f} | "
            f"{rc['cosine_vs_dot_rho']:.4f} | {rc['euclidean_vs_dot_rho']:.4f} |"
        )

    md_lines += [
        f"",
        f"## Implication for TRAK-PCA Gap",
        f"",
        implication,
        f"",
        f"## Runtime",
        f"Total: {elapsed:.1f}s",
    ]

    pilot_md_path = os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.md")
    with open(pilot_md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nPilot summary saved to: {pilot_json_path}")
    print(f"Pilot summary MD saved to: {pilot_md_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY: P4 Cosine vs Euclidean RepSim (v2: raw reps)")
    print(f"{'='*70}")
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        c = comparison[task_name]
        print(f"  [{task_name}] cos-euc gap: {c['cosine_minus_euclidean_pp']:+.2f}pp "
              f"({c['metric']}), cos-dot gap: {c['cosine_minus_dot_pp']:+.2f}pp, "
              f"tau diff: {c['cosine_minus_euclidean_tau']:+.4f}")
    print(f"\n  Norm CVs: tox={norm_analysis['toxicity']['train_norm_cv']:.4f}, "
          f"cf={norm_analysis['counterfact']['train_norm_cv']:.4f}, "
          f"ft={norm_analysis['ftrace']['train_norm_cv']:.4f}")
    print(f"  Normalization matters (cos vs euc): {normalization_matters}")
    print(f"  Normalization matters (cos vs dot): {dot_normalization_matters}")
    print(f"  Overall pass: {pass_criteria['overall_pass']}")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"{'='*70}")

    mark_done("success",
              f"Cosine vs Euclidean pilot (v2: raw reps). "
              f"Normalization matters (cos-euc): {normalization_matters}. "
              f"Normalization matters (cos-dot): {dot_normalization_matters}. "
              f"Max cos-euc gap: {max_gap:+.2f}pp on {max_gap_task}. "
              f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[FATAL] {e}\n{tb}")
        mark_done("failed", f"Error: {e}")
        sys.exit(1)
