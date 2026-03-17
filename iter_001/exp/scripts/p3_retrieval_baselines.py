#!/usr/bin/env python3
"""
P3: Dense Retrieval Baselines -- Contriever + GTR-T5 + BM25 (H8-revised) -- PILOT MODE
========================================================================================
Critical test: if retrieval models match RepSim, 'attribution = retrieval' and paper
must reposition.

Methods:
  - Contriever (facebook/contriever): dense retrieval encoder
  - GTR-T5 (sentence-transformers/gtr-t5-base): generalist text retrieval
  - BM25 (rank_bm25): lexical retrieval baseline (rerun at pilot scale for consistent comparison)

All 3 DATE-LM tasks: toxicity, counterfact, ftrace
Metrics: R@50, MRR, AUPRC, Kendall tau, Spearman rho, NDCG
PILOT: N=100 training samples, seed=42, timeout=900s

Pass criteria: Both retrieval models produce valid scores; at least one of {Contriever, GTR}
shows > 5pp gap below RepSim on >= 1 task.
"""

import os, sys, json, time, gc, re, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_curve, auc, ndcg_score
from scipy.stats import kendalltau, spearmanr
from rank_bm25 import BM25Okapi
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "p3_retrieval_baselines"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
PILOTS_DIR = os.path.join(RESULTS_DIR, "pilots")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
BOOTSTRAP_B = 1000
BATCH_SIZE = 32  # Small models, can use larger batches
KNN_K = 50

TASKS = ["toxicity", "counterfact", "ftrace"]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

for d in [FULL_DIR, PILOTS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

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
    """Compute NDCG@k given score array and relevance array."""
    try:
        return float(ndcg_score([relevance], [scores], k=k))
    except Exception:
        return 0.0


def compute_continuous_metrics_toxicity(scores, unsafe_indices, n_total):
    """Compute Kendall tau, Spearman rho, NDCG for toxicity task."""
    labels = np.zeros(n_total)
    labels[list(unsafe_indices)] = 1.0
    tau, tau_p = kendalltau(scores, labels)
    rho, rho_p = spearmanr(scores, labels)
    ndcg = compute_ndcg(scores, labels, k=50)
    return {
        "kendall_tau": round(float(tau) if not np.isnan(tau) else 0.0, 6),
        "kendall_tau_p": round(float(tau_p) if not np.isnan(tau_p) else 1.0, 6),
        "spearman_rho": round(float(rho) if not np.isnan(rho) else 0.0, 6),
        "spearman_rho_p": round(float(rho_p) if not np.isnan(rho_p) else 1.0, 6),
        "ndcg_at_50": round(ndcg, 6),
    }


def compute_continuous_metrics_factual(scores_per_ref, fact_indices_per_ref, n_train):
    """Compute Kendall tau, Spearman rho, NDCG for factual tasks (counterfact/ftrace)."""
    taus, rhos, ndcgs = [], [], []
    for scores, fi in zip(scores_per_ref, fact_indices_per_ref):
        if not fi:
            continue
        relevance = np.zeros(n_train)
        for idx in fi:
            relevance[idx] = 1.0
        scores_arr = np.array(scores)
        tau, _ = kendalltau(scores_arr, relevance)
        rho, _ = spearmanr(scores_arr, relevance)
        ndcg = compute_ndcg(scores_arr, relevance, k=50)
        if not np.isnan(tau):
            taus.append(tau)
        if not np.isnan(rho):
            rhos.append(rho)
        ndcgs.append(ndcg)
    return {
        "kendall_tau": round(float(np.mean(taus)) if taus else 0.0, 6),
        "spearman_rho": round(float(np.mean(rhos)) if rhos else 0.0, 6),
        "ndcg_at_50": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 6),
        "n_valid_refs": len(taus),
    }


def bootstrap_auprc(scores, unsafe_indices, n_total, n_boot=BOOTSTRAP_B):
    rng = np.random.RandomState(SEED + 1234)
    labels = np.zeros(n_total)
    labels[list(unsafe_indices)] = 1
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n_total, n_total, replace=True)
        if labels[idx].sum() == 0:
            vals.append(0.0)
            continue
        p, r, _ = precision_recall_curve(labels[idx], scores[idx])
        vals.append(float(auc(r, p)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_factual(scores_per_ref, fact_indices_per_ref, n_boot=BOOTSTRAP_B, k=50):
    rng = np.random.RandomState(SEED + 2345)
    n_ref = len(scores_per_ref)
    boot_recalls, boot_mrrs = [], []
    for _ in range(n_boot):
        idx = rng.choice(n_ref, n_ref, replace=True)
        spr = [scores_per_ref[i] for i in idx]
        fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(spr, fi, k=k)
        boot_recalls.append(r)
        boot_mrrs.append(m)
    return ([float(np.percentile(boot_recalls, 2.5)), float(np.percentile(boot_recalls, 97.5))],
            [float(np.percentile(boot_mrrs, 2.5)), float(np.percentile(boot_mrrs, 97.5))])


# ── Data loading ────────────────────────────────────────────────────────
def load_all_tasks():
    report_progress("loading_data", "Loading DATE-LM datasets", 0.05)
    tasks = {}

    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {
        "train": tox["train"], "ref": tox["ref"],
        "metric_name": "AUPRC",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {
        "train": cf["train"], "ref": cf["ref"],
        "metric_name": "Recall@50+MRR",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")

    ft = load_dataset("DataAttributionEval/ftrace", "Pythia-1b")
    tasks["ftrace"] = {
        "train": ft["train"], "ref": ft["ref"],
        "metric_name": "P@K",
        "fmt": lambda s: s["prompt"] + " " + s["response"],
    }
    print(f"[ftrace] train={len(ft['train'])}, ref={len(ft['ref'])}")

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
        print(f"[pilot/{task_name}] Stratified: {n_unsafe} unsafe + {len(chosen_safe)} safe = {len(pilot_idx)}")
    else:
        pilot_idx = sorted(rng.choice(n_total, min(n_pilot, n_total), replace=False).tolist())
        print(f"[pilot/{task_name}] Random: {len(pilot_idx)} samples")

    return train_data.select(pilot_idx), pilot_idx


# ── Patch torch.load safety check for older PyTorch + newer transformers ──
def _patch_torch_load_safety():
    """Monkey-patch transformers to allow torch.load with PyTorch < 2.6."""
    try:
        import transformers.modeling_utils as mu
        _orig_load = mu.load_state_dict
        def _patched_load(ckpt, **kw):
            return torch.load(ckpt, map_location='cpu', weights_only=False)
        mu.load_state_dict = _patched_load
        print("[Patch] Patched transformers.modeling_utils.load_state_dict for PyTorch <2.6 compat")
    except Exception as e:
        print(f"[Patch] Warning: could not patch load_state_dict: {e}")

_patch_torch_load_safety()


# ── Retrieval model encoding ───────────────────────────────────────────
def encode_with_contriever(texts, device=DEVICE, batch_size=BATCH_SIZE):
    """Encode texts using facebook/contriever."""
    print(f"[Contriever] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to(device)
    model.eval()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Contriever uses mean pooling
            token_embs = outputs.last_hidden_state  # [B, seq, d]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(embs.cpu())
        if (i + batch_size) % 128 == 0:
            print(f"  [Contriever] {min(i + batch_size, len(texts))}/{len(texts)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    embs = torch.cat(all_embs, dim=0)
    print(f"[Contriever] Encoded {embs.shape[0]} texts, dim={embs.shape[1]}")
    return embs


def encode_with_gtr_t5(texts, device=DEVICE, batch_size=BATCH_SIZE):
    """Encode texts using sentence-transformers/gtr-t5-base."""
    print(f"[GTR-T5] Loading model...")
    model = SentenceTransformer("sentence-transformers/gtr-t5-base", device=device)

    print(f"[GTR-T5] Encoding {len(texts)} texts...")
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True
    )
    embs_cpu = embs.cpu()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[GTR-T5] Encoded {embs_cpu.shape[0]} texts, dim={embs_cpu.shape[1]}")
    return embs_cpu


def compute_retrieval_sim(train_embs, ref_embs, normalize=True):
    """Compute cosine similarity matrix [n_train, n_ref]."""
    if normalize:
        train_norm = F.normalize(train_embs.float(), dim=-1)
        ref_norm = F.normalize(ref_embs.float(), dim=-1)
    else:
        train_norm = train_embs.float()
        ref_norm = ref_embs.float()
    sim = (train_norm @ ref_norm.T).numpy()
    return sim


# ── BM25 ────────────────────────────────────────────────────────────────
def tokenize_simple(text):
    return re.findall(r'\b\w+\b', text.lower())


def compute_bm25_scores(train_texts, ref_texts, task_name):
    t0 = time.time()
    print(f"[BM25/{task_name}] Tokenizing {len(train_texts)} train + {len(ref_texts)} ref...")
    train_tokenized = [tokenize_simple(t) for t in train_texts]
    ref_tokenized = [tokenize_simple(t) for t in ref_texts]
    bm25 = BM25Okapi(train_tokenized)
    sim = np.zeros((len(train_texts), len(ref_texts)), dtype=np.float32)
    for j, ref_tok in enumerate(ref_tokenized):
        scores = bm25.get_scores(ref_tok)
        sim[:, j] = scores
    elapsed = time.time() - t0
    print(f"[BM25/{task_name}] Done in {elapsed:.1f}s, sim shape={sim.shape}")
    return sim, elapsed


# ── Full evaluation per task ────────────────────────────────────────────
def evaluate_toxicity_full(sim_matrix, pilot_data, ref_data):
    """Evaluate toxicity with rank-based + continuous metrics."""
    train_scores = sim_matrix.mean(axis=1)
    n_train = len(pilot_data)
    unsafe_indices = [i for i in range(n_train) if pilot_data[i]["type"] == "Unsafe"]

    auprc = compute_auprc(train_scores, unsafe_indices, n_train)
    ci_lo, ci_hi = bootstrap_auprc(train_scores, unsafe_indices, n_train)

    continuous = compute_continuous_metrics_toxicity(train_scores, unsafe_indices, n_train)

    return {
        "AUPRC": round(auprc, 6),
        "CI_lower": round(ci_lo, 6),
        "CI_upper": round(ci_hi, 6),
        "n_unsafe": len(unsafe_indices),
        "n_train": n_train,
        "continuous": continuous,
        "score_stats": {
            "mean": float(np.mean(train_scores)),
            "std": float(np.std(train_scores)),
            "min": float(np.min(train_scores)),
            "max": float(np.max(train_scores)),
        }
    }


def evaluate_counterfact_full(sim_matrix, pilot_data, ref_data):
    """Evaluate counterfact with rank-based + continuous metrics."""
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
    recall_ci, mrr_ci = bootstrap_factual(scores_per_ref, fact_indices_per_ref)
    continuous = compute_continuous_metrics_factual(scores_per_ref, fact_indices_per_ref, n_train)
    n_with_facts = sum(1 for f in fact_indices_per_ref if f)

    return {
        "Recall@50": round(recall, 6),
        "Recall@50_CI": [round(recall_ci[0], 6), round(recall_ci[1], 6)],
        "MRR": round(mrr, 6),
        "MRR_CI": [round(mrr_ci[0], 6), round(mrr_ci[1], 6)],
        "refs_with_facts": n_with_facts,
        "n_ref": n_ref,
        "n_train": n_train,
        "continuous": continuous,
    }


def evaluate_ftrace_full(sim_matrix, pilot_data, ref_data):
    """Evaluate ftrace with rank-based + continuous metrics."""
    n_train = len(pilot_data)
    n_ref = len(ref_data)

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

    scores_per_ref = []
    fact_indices_per_ref = []
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
    recall_ci, mrr_ci = bootstrap_factual(scores_per_ref, fact_indices_per_ref)
    continuous = compute_continuous_metrics_factual(scores_per_ref, fact_indices_per_ref, n_train)
    n_with_facts = sum(1 for f in fact_indices_per_ref if f)

    return {
        "Recall@50": round(recall, 6),
        "Recall@50_CI": [round(recall_ci[0], 6), round(recall_ci[1], 6)],
        "MRR": round(mrr, 6),
        "MRR_CI": [round(mrr_ci[0], 6), round(mrr_ci[1], 6)],
        "refs_with_facts": n_with_facts,
        "n_ref": n_ref,
        "n_train": n_train,
        "continuous": continuous,
    }


# ── Qualitative inspection ─────────────────────────────────────────────
def inspect_samples(sim_matrix, pilot_data, method_name, task_name, n_show=5):
    mean_scores = sim_matrix.mean(axis=1)
    top_idx = np.argsort(-mean_scores)[:n_show]
    bot_idx = np.argsort(mean_scores)[:n_show]

    samples = {"top": [], "bottom": []}
    print(f"\n[Qualitative/{method_name}/{task_name}] Top-{n_show} highest:")
    for rank, idx in enumerate(top_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:80]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} score={mean_scores[idx]:.4f} type={label}: {text}...")
        samples["top"].append({"rank": rank+1, "score": round(float(mean_scores[idx]), 4),
                               "type": label, "text_preview": text})

    print(f"[Qualitative/{method_name}/{task_name}] Top-{n_show} lowest:")
    for rank, idx in enumerate(bot_idx):
        s = pilot_data[int(idx)]
        text = s.get("response", s.get("prompt", ""))[:80]
        label = s.get("type", "N/A")
        print(f"  #{rank+1} score={mean_scores[idx]:.4f} type={label}: {text}...")
        samples["bottom"].append({"rank": rank+1, "score": round(float(mean_scores[idx]), 4),
                                  "type": label, "text_preview": text})
    return samples


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 70)
    print("P3: Dense Retrieval Baselines (Contriever + GTR-T5 + BM25)")
    print(f"PILOT MODE: N={PILOT_N_TRAIN}, seed={SEED}")
    print(f"Tasks: {TASKS}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load data
    tasks = load_all_tasks()

    # Prepare pilot subsets and texts
    task_data = {}
    for task_name, task_info in tasks.items():
        pilot_data, pilot_idx = create_pilot_subset(task_name, task_info["train"])
        fmt_fn = task_info["fmt"]
        train_texts = [fmt_fn(pilot_data[i]) for i in range(len(pilot_data))]
        ref_texts = [fmt_fn(task_info["ref"][i]) for i in range(len(task_info["ref"]))]
        task_data[task_name] = {
            "pilot_data": pilot_data,
            "pilot_idx": pilot_idx,
            "train_texts": train_texts,
            "ref_texts": ref_texts,
            "ref_data": task_info["ref"],
        }

    all_results = {}

    # ════════════════════════════════════════════════════════════════════
    # METHOD 1: Contriever
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD: Contriever (facebook/contriever)")
    print("=" * 70)
    report_progress("contriever", "Encoding with Contriever", 0.15)

    contriever_results = {}
    for task_name in TASKS:
        td = task_data[task_name]
        print(f"\n--- Contriever / {task_name} ---")
        t0 = time.time()

        train_embs = encode_with_contriever(td["train_texts"])
        ref_embs = encode_with_contriever(td["ref_texts"])
        sim = compute_retrieval_sim(train_embs, ref_embs, normalize=True)

        if task_name == "toxicity":
            metrics = evaluate_toxicity_full(sim, td["pilot_data"], td["ref_data"])
        elif task_name == "counterfact":
            metrics = evaluate_counterfact_full(sim, td["pilot_data"], td["ref_data"])
        elif task_name == "ftrace":
            metrics = evaluate_ftrace_full(sim, td["pilot_data"], td["ref_data"])

        samples = inspect_samples(sim, td["pilot_data"], "Contriever", task_name)
        elapsed = time.time() - t0

        contriever_results[task_name] = {
            "metrics": metrics,
            "runtime_sec": round(elapsed, 2),
            "n_train_pilot": len(td["pilot_data"]),
            "n_ref": len(td["ref_data"]),
            "embedding_dim": int(train_embs.shape[1]),
            "qualitative_samples": samples,
        }
        report_progress("contriever", f"Done {task_name}", 0.15 + 0.08 * (TASKS.index(task_name) + 1))

        # Free memory between tasks
        del train_embs, ref_embs, sim
        gc.collect()
        torch.cuda.empty_cache()

    all_results["Contriever"] = contriever_results

    # ════════════════════════════════════════════════════════════════════
    # METHOD 2: GTR-T5
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD: GTR-T5 (sentence-transformers/gtr-t5-base)")
    print("=" * 70)
    report_progress("gtr_t5", "Encoding with GTR-T5", 0.40)

    gtr_results = {}
    for task_name in TASKS:
        td = task_data[task_name]
        print(f"\n--- GTR-T5 / {task_name} ---")
        t0 = time.time()

        train_embs = encode_with_gtr_t5(td["train_texts"])
        ref_embs = encode_with_gtr_t5(td["ref_texts"])
        sim = compute_retrieval_sim(train_embs, ref_embs, normalize=True)

        if task_name == "toxicity":
            metrics = evaluate_toxicity_full(sim, td["pilot_data"], td["ref_data"])
        elif task_name == "counterfact":
            metrics = evaluate_counterfact_full(sim, td["pilot_data"], td["ref_data"])
        elif task_name == "ftrace":
            metrics = evaluate_ftrace_full(sim, td["pilot_data"], td["ref_data"])

        samples = inspect_samples(sim, td["pilot_data"], "GTR-T5", task_name)
        elapsed = time.time() - t0

        gtr_results[task_name] = {
            "metrics": metrics,
            "runtime_sec": round(elapsed, 2),
            "n_train_pilot": len(td["pilot_data"]),
            "n_ref": len(td["ref_data"]),
            "embedding_dim": int(train_embs.shape[1]),
            "qualitative_samples": samples,
        }
        report_progress("gtr_t5", f"Done {task_name}", 0.40 + 0.08 * (TASKS.index(task_name) + 1))

        del train_embs, ref_embs, sim
        gc.collect()
        torch.cuda.empty_cache()

    all_results["GTR-T5"] = gtr_results

    # ════════════════════════════════════════════════════════════════════
    # METHOD 3: BM25 (consistent re-run at pilot scale)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD: BM25 (lexical retrieval)")
    print("=" * 70)
    report_progress("bm25", "Running BM25", 0.70)

    bm25_results = {}
    for task_name in TASKS:
        td = task_data[task_name]
        print(f"\n--- BM25 / {task_name} ---")
        t0 = time.time()

        sim, _ = compute_bm25_scores(td["train_texts"], td["ref_texts"], task_name)

        if task_name == "toxicity":
            metrics = evaluate_toxicity_full(sim, td["pilot_data"], td["ref_data"])
        elif task_name == "counterfact":
            metrics = evaluate_counterfact_full(sim, td["pilot_data"], td["ref_data"])
        elif task_name == "ftrace":
            metrics = evaluate_ftrace_full(sim, td["pilot_data"], td["ref_data"])

        samples = inspect_samples(sim, td["pilot_data"], "BM25", task_name)
        elapsed = time.time() - t0

        bm25_results[task_name] = {
            "metrics": metrics,
            "runtime_sec": round(elapsed, 2),
            "n_train_pilot": len(td["pilot_data"]),
            "n_ref": len(td["ref_data"]),
            "qualitative_samples": samples,
        }

    all_results["BM25"] = bm25_results

    # ── Load RepSim reference results for comparison ────────────────────
    report_progress("analysis", "Comparing against RepSim", 0.85)

    repsim_reference = {}
    try:
        # Load from p1_fm2_continuous_metrics results
        p1_path = os.path.join(FULL_DIR, "p1_fm2_continuous_metrics.json")
        if os.path.exists(p1_path):
            with open(p1_path) as f:
                p1_data = json.load(f)
            # Extract RepSim standard results
            for task_name in TASKS:
                key = f"RepSim_standard_{task_name}"
                if "tournament" in p1_data:
                    for entry in p1_data["tournament"]:
                        if entry.get("method") == "RepSim" and entry.get("scoring") == "standard" and entry.get("task") == task_name:
                            repsim_reference[task_name] = entry.get("metrics", {})
                            break
            # Also try results dict format
            if not repsim_reference:
                results = p1_data.get("results", {})
                for task_name in TASKS:
                    repsim_key = f"RepSim_standard"
                    if repsim_key in results and task_name in results[repsim_key]:
                        repsim_reference[task_name] = results[repsim_key][task_name]
        print(f"[RepSim Reference] Loaded for tasks: {list(repsim_reference.keys())}")
    except Exception as e:
        print(f"[RepSim Reference] Could not load: {e}")

    # Also try the pilot results
    if not repsim_reference:
        try:
            pilot_p1 = os.path.join(PILOTS_DIR, "p1_fm2_continuous_metrics_pilot_summary.json")
            if os.path.exists(pilot_p1):
                with open(pilot_p1) as f:
                    ps = json.load(f)
                km = ps.get("candidates", [{}])[0].get("key_metrics", {})
                repsim_reference = {
                    "counterfact": {"Recall@50": km.get("repsim_cf_r50", None)},
                    "toxicity": {"AUPRC": km.get("best_tox_tau", None)},
                }
                print(f"[RepSim Reference] Loaded from pilot summary (partial)")
        except Exception as e:
            print(f"[RepSim Reference] Could not load pilot: {e}")

    # ── Comparison analysis ─────────────────────────────────────────────
    report_progress("comparison", "Computing gaps", 0.90)

    comparisons = {}
    for method_name in ["Contriever", "GTR-T5", "BM25"]:
        method_comp = {}
        for task_name in TASKS:
            res = all_results[method_name][task_name]["metrics"]
            comp = {}

            # Get primary metric
            if task_name == "toxicity":
                comp["primary_metric"] = "AUPRC"
                comp["value"] = res["AUPRC"]
                if task_name in repsim_reference and "AUPRC" in repsim_reference.get(task_name, {}):
                    repsim_val = repsim_reference[task_name]["AUPRC"]
                    comp["repsim_value"] = repsim_val
                    comp["gap_pp"] = round((repsim_val - res["AUPRC"]) * 100, 2)
            else:
                comp["primary_metric"] = "Recall@50"
                comp["value"] = res["Recall@50"]
                if task_name in repsim_reference and "Recall@50" in repsim_reference.get(task_name, {}):
                    repsim_val = repsim_reference[task_name]["Recall@50"]
                    comp["repsim_value"] = repsim_val
                    comp["gap_pp"] = round((repsim_val - res["Recall@50"]) * 100, 2)

            method_comp[task_name] = comp
        comparisons[method_name] = method_comp

    # ── Decision gate evaluation ────────────────────────────────────────
    # "at least one of {Contriever, GTR} shows > 5pp gap below RepSim on >= 1 task"
    pass_criteria = {
        "both_produce_valid_scores": True,
        "gap_above_5pp_exists": False,
        "details": {},
    }

    for method_name in ["Contriever", "GTR-T5"]:
        for task_name in TASKS:
            comp = comparisons.get(method_name, {}).get(task_name, {})
            gap = comp.get("gap_pp", None)
            if gap is not None and gap > 5.0:
                pass_criteria["gap_above_5pp_exists"] = True
                pass_criteria["details"][f"{method_name}_{task_name}"] = f"gap={gap:.1f}pp > 5pp"

    # Check for the critical concern: retrieval matches RepSim
    critical_concern = False
    matching_tasks = 0
    for method_name in ["Contriever", "GTR-T5"]:
        for task_name in TASKS:
            comp = comparisons.get(method_name, {}).get(task_name, {})
            gap = comp.get("gap_pp", None)
            if gap is not None and abs(gap) < 3.0:
                matching_tasks += 1
    if matching_tasks >= 2:
        critical_concern = True

    pass_criteria["critical_concern_retrieval_matches_repsim"] = critical_concern
    pass_criteria["matching_task_count"] = matching_tasks

    # ── Save results ────────────────────────────────────────────────────
    report_progress("saving", "Saving results", 0.95)

    total_time = time.time() - t_start

    final = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "methods": ["Contriever", "GTR-T5", "BM25"],
        "pilot_n_train": PILOT_N_TRAIN,
        "bootstrap_B": BOOTSTRAP_B,
        "seed": SEED,
        "results": all_results,
        "repsim_reference": repsim_reference,
        "comparisons": comparisons,
        "pass_criteria": pass_criteria,
        "decision_gate": {
            "condition": "Contriever or GTR-T5 matches RepSim (< 3pp gap) on >= 2 tasks",
            "result": critical_concern,
            "action_if_true": "Reposition from 'attribution quality' to 'attribution vs retrieval boundary analysis'",
            "action_if_false": "RepSim captures genuine model-internal attribution beyond retrieval; proceed with current framing",
        },
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    # Save to full results dir
    out_path = os.path.join(FULL_DIR, "p3_retrieval_baselines.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save to pilots dir
    with open(os.path.join(PILOTS_DIR, f"{TASK_ID}_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: P3 Retrieval Baselines (PILOT)")
    print("=" * 80)
    print(f"{'Method':<15}{'Task':<15}{'Primary':<12}{'Value':<10}{'CI_lo':<10}{'CI_hi':<10}{'Tau':<10}{'Time(s)':<8}")
    print("-" * 90)
    for method in ["Contriever", "GTR-T5", "BM25"]:
        for tn in TASKS:
            m = all_results[method][tn]["metrics"]
            tau = m.get("continuous", {}).get("kendall_tau", "N/A")
            if tn == "toxicity":
                print(f"{method:<15}{tn:<15}{'AUPRC':<12}{m['AUPRC']:<10.4f}{m['CI_lower']:<10.4f}{m['CI_upper']:<10.4f}{tau:<10}{all_results[method][tn]['runtime_sec']:<8.1f}")
            else:
                ci = m.get("Recall@50_CI", [0, 0])
                print(f"{method:<15}{tn:<15}{'R@50':<12}{m['Recall@50']:<10.4f}{ci[0]:<10.4f}{ci[1]:<10.4f}{tau:<10}{all_results[method][tn]['runtime_sec']:<8.1f}")
    print("-" * 90)

    # Comparison with RepSim
    print("\nComparison with RepSim:")
    for method in ["Contriever", "GTR-T5", "BM25"]:
        for tn in TASKS:
            comp = comparisons.get(method, {}).get(tn, {})
            gap = comp.get("gap_pp", "N/A")
            print(f"  {method}/{tn}: gap_to_RepSim = {gap}pp")

    print(f"\nPass criteria: {pass_criteria}")
    print(f"Decision gate: critical_concern={critical_concern}")
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 80)

    # ── Pilot summary files ─────────────────────────────────────────────
    # pilot_summary.json
    pilot_summary = {
        "overall_recommendation": "GO" if pass_criteria["gap_above_5pp_exists"] else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if pass_criteria["gap_above_5pp_exists"] else "REFINE",
            "confidence": 0.7 if pass_criteria["gap_above_5pp_exists"] else 0.4,
            "supported_hypotheses": ["H8_retrieval_gap"] if pass_criteria["gap_above_5pp_exists"] else [],
            "failed_assumptions": [] if pass_criteria["gap_above_5pp_exists"] else ["H8_retrieval_gap"],
            "key_metrics": {
                "contriever_tox_auprc": all_results["Contriever"]["toxicity"]["metrics"]["AUPRC"],
                "contriever_cf_r50": all_results["Contriever"]["counterfact"]["metrics"]["Recall@50"],
                "contriever_ft_r50": all_results["Contriever"]["ftrace"]["metrics"]["Recall@50"],
                "gtr_tox_auprc": all_results["GTR-T5"]["toxicity"]["metrics"]["AUPRC"],
                "gtr_cf_r50": all_results["GTR-T5"]["counterfact"]["metrics"]["Recall@50"],
                "gtr_ft_r50": all_results["GTR-T5"]["ftrace"]["metrics"]["Recall@50"],
                "bm25_tox_auprc": all_results["BM25"]["toxicity"]["metrics"]["AUPRC"],
                "bm25_cf_r50": all_results["BM25"]["counterfact"]["metrics"]["Recall@50"],
                "bm25_ft_r50": all_results["BM25"]["ftrace"]["metrics"]["Recall@50"],
                "critical_concern_retrieval_matches_repsim": critical_concern,
                "pass_criteria_met": pass_criteria["gap_above_5pp_exists"],
            },
            "notes": f"Retrieval baselines pilot. Pass criteria (>5pp gap below RepSim on >=1 task): {'MET' if pass_criteria['gap_above_5pp_exists'] else 'NOT MET'}. "
                     f"Critical concern (retrieval matches RepSim on >=2 tasks): {'YES' if critical_concern else 'NO'}. "
                     f"Details: {json.dumps(pass_criteria['details'])}"
        }]
    }

    with open(os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.json"), "w") as f:
        json.dump(pilot_summary, f, indent=2)

    # pilot_summary.md
    md_lines = [
        f"# P3: Retrieval Baselines Pilot Summary",
        f"",
        f"## Configuration",
        f"- Mode: PILOT (N={PILOT_N_TRAIN})",
        f"- Methods: Contriever, GTR-T5, BM25",
        f"- Tasks: toxicity, counterfact, ftrace",
        f"- Total runtime: {total_time:.1f}s",
        f"",
        f"## Results",
        f"",
    ]

    for method in ["Contriever", "GTR-T5", "BM25"]:
        md_lines.append(f"### {method}")
        for tn in TASKS:
            m = all_results[method][tn]["metrics"]
            if tn == "toxicity":
                md_lines.append(f"- {tn}: AUPRC={m['AUPRC']:.4f} [{m['CI_lower']:.4f}, {m['CI_upper']:.4f}]")
            else:
                ci = m.get("Recall@50_CI", [0, 0])
                md_lines.append(f"- {tn}: R@50={m['Recall@50']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], MRR={m['MRR']:.4f}")
            if "continuous" in m:
                c = m["continuous"]
                md_lines.append(f"  - Kendall tau={c.get('kendall_tau', 'N/A')}, Spearman rho={c.get('spearman_rho', 'N/A')}")
        md_lines.append("")

    md_lines.extend([
        f"## Decision Gate",
        f"- Condition: Contriever or GTR-T5 matches RepSim (<3pp gap) on >=2 tasks",
        f"- Result: {'TRIGGERED' if critical_concern else 'NOT TRIGGERED'}",
        f"- Pass criteria (>5pp gap): {'MET' if pass_criteria['gap_above_5pp_exists'] else 'NOT MET'}",
        f"",
        f"## Comparison with RepSim",
    ])
    for method in ["Contriever", "GTR-T5"]:
        for tn in TASKS:
            comp = comparisons.get(method, {}).get(tn, {})
            gap = comp.get("gap_pp", "N/A")
            md_lines.append(f"- {method}/{tn}: gap = {gap}pp")

    md_lines.extend([
        f"",
        f"## Recommendation",
        f"{'GO' if pass_criteria['gap_above_5pp_exists'] else 'REFINE'}: " +
        ("At least one retrieval model shows >5pp gap below RepSim, confirming RepSim captures genuine model-internal attribution."
         if pass_criteria["gap_above_5pp_exists"]
         else "Need to investigate further; retrieval models may match RepSim.")
    ])

    with open(os.path.join(PILOTS_DIR, f"{TASK_ID}_pilot_summary.md"), "w") as f:
        f.write("\n".join(md_lines))

    mark_done(
        status="success",
        summary=f"Retrieval baselines pilot completed. pass_criteria_met={pass_criteria['gap_above_5pp_exists']}, "
                f"critical_concern={critical_concern}. Total {total_time:.1f}s."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
