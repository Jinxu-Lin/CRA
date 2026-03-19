#!/usr/bin/env python3
"""
CRA Full-Scale Master Experiment Script v2
===========================================
Memory-efficient version: gradient-based methods use streaming/projected computation
to avoid storing full N x D gradient matrices (which would require ~800GB for N=10K, D=21M).

Strategy:
- Representations: stored as N x d (d=2048), fits easily in RAM (~80MB per task)
- Gradient-based methods: compute projected gradients (N x k, k=2048) incrementally
  Store only the projected versions, not full D-dimensional gradients
- For DiagIF/RawDotIF: compute in streaming fashion (accumulate outer products)

All tasks on GPU 3, batch_size=2 for ~5GB available GPU memory.
"""

import os, sys, json, time, gc, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_curve, auc, ndcg_score
from sklearn.covariance import LedoitWolf
from scipy.stats import kendalltau, spearmanr
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
FULL_DIR = os.path.join(RESULTS_DIR, "full")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048
LOGRA_K = 256
BOOTSTRAP_B = 1000
MAX_LEN = 512
BATCH_SIZE = 2
KNN_K = 50

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Lifecycle helpers ───────────────────────────────────────────────────
def write_pid(task_id):
    Path(RESULTS_DIR, f"{task_id}.pid").write_text(str(os.getpid()))

def report_progress(task_id, stage, detail="", pct=0.0):
    Path(RESULTS_DIR, f"{task_id}_PROGRESS.json").write_text(json.dumps({
        "task_id": task_id, "stage": stage, "detail": detail,
        "pct": pct, "updated_at": datetime.now().isoformat(),
    }))

def mark_done(task_id, status="success", summary=""):
    pid_f = Path(RESULTS_DIR) / f"{task_id}.pid"
    if pid_f.exists(): pid_f.unlink()
    prog_f = Path(RESULTS_DIR) / f"{task_id}_PROGRESS.json"
    final = {}
    if prog_f.exists():
        try: final = json.loads(prog_f.read_text())
        except: pass
    Path(RESULTS_DIR, f"{task_id}_DONE").write_text(json.dumps({
        "task_id": task_id, "status": status, "summary": summary,
        "final_progress": final, "timestamp": datetime.now().isoformat(),
    }))

# ── Metric helpers ──────────────────────────────────────────────────────
def compute_auprc(scores, unsafe_indices, n_total):
    labels = np.zeros(n_total); labels[list(unsafe_indices)] = 1
    if sum(labels) == 0: return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))

def compute_factual_metrics(scores_per_ref, fact_indices_per_ref, k=50):
    recalls, mrrs = [], []
    for scores, fi in zip(scores_per_ref, fact_indices_per_ref):
        if not fi: continue
        si = np.argsort(-np.array(scores))
        topk = set(si[:k].tolist())
        recalls.append(len([i for i in fi if i in topk]) / len(fi))
        r2r = {idx: rank+1 for rank, idx in enumerate(si)}
        ranks = [r2r[i] for i in fi if i in r2r]
        mrrs.append(1.0/min(ranks) if ranks else 0.0)
    if not recalls: return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(mrrs))

def bootstrap_auprc(scores, unsafe_indices, n_total, n_boot=BOOTSTRAP_B):
    rng = np.random.RandomState(SEED+1234)
    labels = np.zeros(n_total); labels[list(unsafe_indices)] = 1
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n_total, n_total, replace=True)
        if labels[idx].sum() == 0: vals.append(0.0); continue
        p, r, _ = precision_recall_curve(labels[idx], scores[idx])
        vals.append(float(auc(r, p)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def bootstrap_factual(scores_per_ref, fact_indices_per_ref, n_boot=BOOTSTRAP_B, k=50):
    rng = np.random.RandomState(SEED+2345)
    n_ref = len(scores_per_ref)
    boot_r, boot_m = [], []
    for _ in range(n_boot):
        idx = rng.choice(n_ref, n_ref, replace=True)
        spr = [scores_per_ref[i] for i in idx]
        fi = [fact_indices_per_ref[i] for i in idx]
        r, m = compute_factual_metrics(spr, fi, k=k)
        boot_r.append(r); boot_m.append(m)
    return ([float(np.percentile(boot_r,2.5)), float(np.percentile(boot_r,97.5))],
            [float(np.percentile(boot_m,2.5)), float(np.percentile(boot_m,97.5))])

def compute_ndcg(scores, relevance, k=50):
    try: return float(ndcg_score([relevance], [scores], k=k))
    except: return 0.0

def compute_continuous_metrics_toxicity(scores, labels):
    tau_val, tau_p = kendalltau(labels, scores)
    rho_val, rho_p = spearmanr(labels, scores)
    ndcg_val = compute_ndcg(scores, labels, k=50)
    return {
        "kendall_tau": float(tau_val) if not np.isnan(tau_val) else 0.0,
        "kendall_p": float(tau_p) if not np.isnan(tau_p) else 1.0,
        "spearman_rho": float(rho_val) if not np.isnan(rho_val) else 0.0,
        "spearman_p": float(rho_p) if not np.isnan(rho_p) else 1.0,
        "ndcg_at_50": ndcg_val,
    }

def compute_continuous_metrics_factual(sim_matrix, train_data, ref_data, task_name):
    n_train, n_ref = sim_matrix.shape
    taus, rhos, ndcgs = [], [], []
    for j in range(n_ref):
        gt_j = _get_ground_truth(task_name, train_data, ref_data[j], n_train)
        scores_j = sim_matrix[:, j]
        if gt_j.sum() > 0 and gt_j.sum() < len(gt_j):
            t, _ = kendalltau(gt_j, scores_j)
            r, _ = spearmanr(gt_j, scores_j)
            if not np.isnan(t): taus.append(t)
            if not np.isnan(r): rhos.append(r)
            ndcgs.append(compute_ndcg(scores_j, gt_j, k=50))
    return {
        "kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "kendall_tau_std": float(np.std(taus)) if taus else 0.0,
        "spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
        "ndcg_at_50": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_evaluated_refs": len(taus),
    }

def _get_ground_truth(task_name, train_data, ref_sample, n_train):
    if task_name == "counterfact":
        return np.array([
            1.0 if (train_data[i]["counterfactual_entity"] == ref_sample["counterfactual_entity"]
                    and train_data[i]["true_entity"] == ref_sample["true_entity"])
            else 0.0 for i in range(n_train)
        ])
    else:  # ftrace
        ref_facts_raw = ref_sample.get("facts", [])
        if isinstance(ref_facts_raw, str):
            ref_facts_raw = [f.strip() for f in ref_facts_raw.split(",") if f.strip()]
        elif isinstance(ref_facts_raw, list):
            flat = []
            for f in ref_facts_raw:
                if isinstance(f, str): flat.extend([x.strip() for x in f.split(",") if x.strip()])
            ref_facts_raw = flat
        ref_facts = set(ref_facts_raw)
        gt = np.zeros(n_train)
        for i in range(n_train):
            tf = train_data[i].get("facts", [])
            if isinstance(tf, str): tf = [x.strip() for x in tf.split(",") if x.strip()]
            elif isinstance(tf, list):
                flat = []
                for f in tf:
                    if isinstance(f, str): flat.extend([x.strip() for x in f.split(",") if x.strip()])
                tf = flat
            if set(tf) & ref_facts: gt[i] = 1.0
        return gt

def _get_fact_indices(task_name, train_data, ref_data):
    fact_indices_per_ref = []
    for ref_sample in ref_data:
        gt = _get_ground_truth(task_name, train_data, ref_sample, len(train_data))
        fact_indices_per_ref.append([i for i in range(len(train_data)) if gt[i] > 0])
    return fact_indices_per_ref

# ── Data loading ────────────────────────────────────────────────────────
def load_all_tasks_full():
    print("=" * 60)
    print("Loading DATE-LM datasets at FULL SCALE")
    print("=" * 60)
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

# ── Model loading ───────────────────────────────────────────────────────
def load_model_pythia1b():
    gc.collect(); torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, dtype=torch.float16,
        device_map=DEVICE, low_cpu_mem_usage=True
    )
    model.eval()
    free_mem = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"Pythia-1B loaded. Hidden dim={model.config.hidden_size}. GPU free: {free_mem:.0f}MiB")
    return model, tok

def unload_model(model):
    del model; gc.collect(); torch.cuda.empty_cache()

# ── Representation extraction ───────────────────────────────────────────
def extract_representations(model, tok, texts, layer_idx=-1, batch_size=BATCH_SIZE, desc=""):
    all_reps = []
    for idx in range(0, len(texts), batch_size):
        batch_texts = texts[idx:idx+batch_size]
        inputs = tok(batch_texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(input_ids=inputs["input_ids"],
                          attention_mask=inputs["attention_mask"],
                          output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_reps.append(pooled.float().cpu())
        del inputs, outputs, hidden, mask, pooled
        torch.cuda.empty_cache()
        if (idx // batch_size) % 100 == 0 and idx > 0:
            print(f"  [{desc}] {idx}/{len(texts)} extracted")
    reps = torch.cat(all_reps, dim=0)
    print(f"  [{desc}] Done: {reps.shape[0]} reps, dim={reps.shape[1]}")
    return reps

def get_or_extract_reps(model, tok, texts, task_name, layer_suffix="", layer_idx=-1):
    """Get cached representations or extract them."""
    suffix = f"_{layer_suffix}" if layer_suffix else ""
    cache_path = os.path.join(CACHE_DIR, f"reps_train_{task_name}{suffix}_full.pt")
    if os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return torch.load(cache_path, weights_only=True)
    reps = extract_representations(model, tok, texts, layer_idx=layer_idx,
                                   desc=f"{task_name}{suffix}/train")
    torch.save(reps, cache_path)
    return reps

def get_or_extract_ref_reps(model, tok, texts, task_name, layer_suffix="", layer_idx=-1):
    suffix = f"_{layer_suffix}" if layer_suffix else ""
    cache_path = os.path.join(CACHE_DIR, f"reps_ref_{task_name}{suffix}_full.pt")
    if os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return torch.load(cache_path, weights_only=True)
    reps = extract_representations(model, tok, texts, layer_idx=layer_idx,
                                   desc=f"{task_name}{suffix}/ref")
    torch.save(reps, cache_path)
    return reps

# ── MEMORY-EFFICIENT Gradient computation ───────────────────────────────
def setup_target_params(model, layer_filter=None):
    """Setup target params. layer_filter=None means default (layer 15 attn+mlp).
    layer_filter="last_layer_all" means all params in layer 15."""
    target_params, target_names = [], []
    for name, p in model.named_parameters():
        p.requires_grad_(False)

    for name, p in model.named_parameters():
        if layer_filter is None:
            # Default: just attention.dense + mlp.dense_4h_to_h in last layer
            if any(sub in name for sub in [
                "layers.15.attention.dense.weight",
                "layers.15.mlp.dense_4h_to_h.weight"
            ]):
                p.requires_grad_(True)
                target_params.append(p)
                target_names.append(name)
        elif layer_filter == "last_layer_all":
            if "layers.15." in name and ("attention" in name or "mlp" in name):
                p.requires_grad_(True)
                target_params.append(p)
                target_names.append(name)

    D = sum(p.numel() for p in target_params)
    print(f"Target params: D={D/1e6:.2f}M from {len(target_names)} params")
    return target_params, target_names, D

def restore_grad_flags(model):
    for p in model.parameters(): p.requires_grad_(True)

def compute_single_gradient(model, tok, text, target_params):
    """Compute gradient for a single sample. Returns flat gradient vector on CPU."""
    inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    model.zero_grad()
    with torch.amp.autocast("cuda"):
        out = model(input_ids=inp["input_ids"],
                   attention_mask=inp["attention_mask"],
                   labels=inp["input_ids"])
    out.loss.backward()
    grad_flat = torch.cat([p.grad.detach().flatten().float().cpu() for p in target_params])
    model.zero_grad(set_to_none=True)
    del inp, out
    return grad_flat

def compute_projected_gradients(model, tok, texts, target_params, projector, desc=""):
    """Compute gradients and immediately project to low-dim space.
    projector: CountSketchProjector or torch.Tensor [D, k].
    Returns: [N, k] projected gradients (fits in RAM).
    """
    if isinstance(projector, CountSketchProjector):
        k = projector.k
    else:
        k = projector.shape[1]

    N = len(texts)
    projected = torch.zeros(N, k, dtype=torch.float32)

    for idx in range(N):
        grad = compute_single_gradient(model, tok, texts[idx], target_params)
        if isinstance(projector, CountSketchProjector):
            projected[idx] = projector.project(grad)
        else:
            projected[idx] = grad @ projector
        del grad

        if (idx + 1) % 200 == 0:
            print(f"  [{desc}] {idx+1}/{N} projected (dim={k})")
            torch.cuda.empty_cache()

    print(f"  [{desc}] Done: {N} projected grads, dim={k}")
    return projected

def compute_gradient_stats_streaming(model, tok, texts, target_params, desc=""):
    """Compute gradient statistics in streaming fashion:
    - Fisher diagonal (mean of squared gradients)
    - Gradient mean (for DDA centering)
    - Gradient norms (for RawDotIF)
    Does NOT store full gradient matrix.
    Returns fisher_diag [D], grad_mean [D], grad_norms [N]
    """
    N = len(texts)
    D = sum(p.numel() for p in target_params)
    fisher_diag = torch.zeros(D, dtype=torch.float64)
    grad_mean = torch.zeros(D, dtype=torch.float64)
    grad_norms = torch.zeros(N, dtype=torch.float32)

    for idx in range(N):
        grad = compute_single_gradient(model, tok, texts[idx], target_params)
        fisher_diag += grad.double() ** 2
        grad_mean += grad.double()
        grad_norms[idx] = grad.norm().item()
        del grad

        if (idx + 1) % 200 == 0:
            print(f"  [{desc}] {idx+1}/{N} stats")
            torch.cuda.empty_cache()

    fisher_diag /= N
    grad_mean /= N
    print(f"  [{desc}] Done: stats for {N} samples, D={D}")
    return fisher_diag.float(), grad_mean.float(), grad_norms

def compute_sim_matrix_streaming(model, tok, train_texts, ref_texts, target_params,
                                  method, proj_matrix=None, fisher_diag=None,
                                  grad_mean=None, desc=""):
    """Compute similarity matrix [N_train, N_ref] in a streaming fashion.
    For each ref sample, compute its gradient, then iterate over train samples
    computing the similarity. This uses O(D) memory per ref sample, not O(N*D).

    For TRAK/LoGra: use pre-computed projected train grads and project ref grads.
    For RawDotIF/DiagIF: compute ref grad once, then stream train grads.
    """
    N_train = len(train_texts)
    N_ref = len(ref_texts)
    sim = np.zeros((N_train, N_ref), dtype=np.float32)

    if method in ["TRAK", "LoGra", "DDA"]:
        # These should use pre-computed projected gradients
        raise ValueError(f"Use compute_projected_gradients for {method}")

    print(f"  [{desc}] Computing {method} sim matrix [{N_train}x{N_ref}] streaming...")

    for j in range(N_ref):
        ref_grad = compute_single_gradient(model, tok, ref_texts[j], target_params)

        # Process train samples in chunks to avoid recomputing
        # Actually, for RawDotIF/DiagIF on the ref side, we need each train grad too
        # This is O(N_train * N_ref) gradient computations -- too slow.
        # Better approach: pre-compute projected for these methods too.
        # For now, compute ref projection and use cached train projections.

        if method == "RawDotIF":
            # Dot product: need some form of projection
            # Use random projection to approximate
            if proj_matrix is not None:
                ref_proj = (ref_grad @ proj_matrix).numpy()
                # train_proj passed separately
            else:
                sim[:, j] = 0.0  # placeholder
        elif method == "DiagIF":
            if fisher_diag is not None and proj_matrix is not None:
                # Scale ref grad by Fisher, then project
                scaled_ref = ref_grad / (fisher_diag.sqrt() + 1e-8)
                ref_proj = (scaled_ref @ proj_matrix).numpy()
            else:
                sim[:, j] = 0.0

        del ref_grad
        torch.cuda.empty_cache()

    return sim

# ── Representation-based methods ────────────────────────────────────────
def compute_repsim_cosine(train_reps, ref_reps):
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    return (tn @ rn.T).numpy()

def compute_repsim_euclidean(train_reps, ref_reps):
    aa = (train_reps ** 2).sum(dim=-1, keepdim=True)
    bb = (ref_reps ** 2).sum(dim=-1, keepdim=True)
    dist_sq = aa + bb.T - 2 * (train_reps @ ref_reps.T)
    return (-dist_sq).numpy()

def compute_repsim_dot(train_reps, ref_reps):
    return (train_reps @ ref_reps.T).numpy()

def compute_knn_raw(train_reps, ref_reps, k=KNN_K):
    tn = F.normalize(train_reps, dim=-1)
    rn = F.normalize(ref_reps, dim=-1)
    sim = (tn @ rn.T).numpy()
    result = np.zeros_like(sim)
    for j in range(sim.shape[1]):
        topk_idx = np.argsort(-sim[:, j])[:k]
        result[topk_idx, j] = sim[topk_idx, j]
    return result

def compute_bm25_scores(train_texts, ref_texts):
    from rank_bm25 import BM25Okapi
    tokenized_train = [t.lower().split() for t in train_texts]
    bm25 = BM25Okapi(tokenized_train)
    sim = np.zeros((len(train_texts), len(ref_texts)))
    for j, ref_text in enumerate(ref_texts):
        scores = bm25.get_scores(ref_text.lower().split())
        sim[:, j] = scores
    return sim

# ── Projected gradient methods ──────────────────────────────────────────
def projected_sim(train_proj, ref_proj, normalize=True):
    """Compute similarity from projected gradients."""
    if normalize:
        tn = train_proj / (np.linalg.norm(train_proj, axis=1, keepdims=True) + 1e-10)
        rn = ref_proj / (np.linalg.norm(ref_proj, axis=1, keepdims=True) + 1e-10)
        return tn @ rn.T
    else:
        return train_proj @ ref_proj.T

class CountSketchProjector:
    """Sparse JL Transform (Count Sketch) for memory-efficient gradient projection.
    Maps D-dimensional vectors to k dimensions using hashing.
    O(D) time, O(D+k) memory (for hash tables, NOT O(D*k)).
    """
    def __init__(self, D, k, seed=SEED):
        self.D = D
        self.k = k
        self.seed = seed
        # Pre-compute hash tables: bucket assignment and sign
        rng = np.random.RandomState(seed)
        self.buckets = torch.tensor(rng.randint(0, k, size=D), dtype=torch.long)  # [D], ~160MB for D=21M
        self.signs = torch.tensor(rng.choice([-1, 1], size=D), dtype=torch.float32)  # [D], ~80MB
        self.scale = 1.0 / np.sqrt(k)
        print(f"  CountSketch projector: D={D/1e6:.1f}M -> k={k}, memory ~{(D*12)/(1024**2):.0f}MB")

    def project(self, grad_vec):
        """Project a single D-dimensional gradient to k dimensions.
        grad_vec: [D] tensor (float32)
        Returns: [k] tensor (float32)
        """
        signed = grad_vec.float() * self.signs  # [D]
        result = torch.zeros(self.k, dtype=torch.float32)
        result.scatter_add_(0, self.buckets, signed)
        return result * self.scale

    def project_batch(self, grads):
        """Project multiple gradients. grads: [N, D] tensor. Returns: [N, k]."""
        signed = grads.float() * self.signs.unsqueeze(0)  # [N, D]
        result = torch.zeros(grads.shape[0], self.k, dtype=torch.float32)
        buckets_exp = self.buckets.unsqueeze(0).expand_as(signed)
        result.scatter_add_(1, buckets_exp, signed)
        return result * self.scale

def make_random_projection(D, k, seed=SEED):
    """Create a CountSketch projector for memory-efficient random projection."""
    return CountSketchProjector(D, k, seed)

# ── Evaluation wrapper ──────────────────────────────────────────────────
def evaluate_method(sim_matrix, task_name, train_data, ref_data, scoring="standard"):
    n_train = sim_matrix.shape[0]
    if scoring == "contrastive":
        sim_matrix = sim_matrix - sim_matrix.mean(axis=0, keepdims=True)
    result = {"scoring": scoring}

    if task_name == "toxicity":
        unsafe_idx = [i for i in range(n_train) if train_data[i]["type"] == "Unsafe"]
        avg_scores = sim_matrix.mean(axis=1)
        auprc = compute_auprc(avg_scores, unsafe_idx, n_train)
        ci = bootstrap_auprc(avg_scores, unsafe_idx, n_train)
        labels = np.array([1.0 if train_data[i]["type"] == "Unsafe" else 0.0 for i in range(n_train)])
        cont = compute_continuous_metrics_toxicity(avg_scores, labels)
        result["rank_based"] = {"AUPRC": auprc, "CI_lower": ci[0], "CI_upper": ci[1],
                                "n_unsafe": len(unsafe_idx), "n_train": n_train}
        result["continuous"] = cont
        result["score_stats"] = {"mean": float(avg_scores.mean()), "std": float(avg_scores.std())}
    elif task_name in ["counterfact", "ftrace"]:
        fact_indices = _get_fact_indices(task_name, train_data, ref_data)
        scores_per_ref = [sim_matrix[:, j].tolist() for j in range(sim_matrix.shape[1])]
        r50, mrr = compute_factual_metrics(scores_per_ref, fact_indices, k=50)
        ci_r, ci_m = bootstrap_factual(scores_per_ref, fact_indices)
        cont = compute_continuous_metrics_factual(sim_matrix, train_data, ref_data, task_name)
        result["rank_based"] = {"R_at_50": r50, "MRR": mrr, "R50_CI": ci_r, "MRR_CI": ci_m}
        result["continuous"] = cont
    return result

# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1: Extract all representations + projected gradients
#  This is the expensive shared computation phase.
# ═══════════════════════════════════════════════════════════════════════
def phase1_extract_all(tasks, model, tok):
    """Extract representations and projected gradients for all tasks.
    Gradients are projected to TRAK_K dimensions to fit in RAM.
    Also computes Fisher diagonal and gradient stats for DiagIF/RawDotIF.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Extract representations + projected gradients")
    print("=" * 60)

    target_params, target_names, D = setup_target_params(model)

    # CountSketch projector for TRAK (memory-efficient: ~240MB for D=21M)
    proj_random = make_random_projection(D, TRAK_K)
    # Save metadata for reproducibility
    with open(os.path.join(CACHE_DIR, "proj_random_info.json"), "w") as pf:
        json.dump({"D": D, "k": TRAK_K, "seed": SEED, "type": "count_sketch"}, pf)

    for task_name, task_data in tasks.items():
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        print(f"\n--- {task_name}: N_train={n_train}, N_ref={n_ref} ---")

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        # Representations (last layer)
        train_reps = get_or_extract_reps(model, tok, train_texts, task_name)
        ref_reps = get_or_extract_ref_reps(model, tok, ref_texts, task_name)

        # Projected gradients: train
        cache_proj_train = os.path.join(CACHE_DIR, f"proj_grads_train_{task_name}_full.pt")
        if os.path.exists(cache_proj_train):
            print(f"  Cached projected train grads: {cache_proj_train}")
        else:
            print(f"  Computing projected train gradients ({n_train} samples, proj to k={TRAK_K})...")
            train_proj = compute_projected_gradients(
                model, tok, train_texts, target_params, proj_random,
                desc=f"{task_name}/train_proj")
            torch.save(train_proj, cache_proj_train)
            del train_proj

        # Projected gradients: ref
        cache_proj_ref = os.path.join(CACHE_DIR, f"proj_grads_ref_{task_name}_full.pt")
        if os.path.exists(cache_proj_ref):
            print(f"  Cached projected ref grads: {cache_proj_ref}")
        else:
            print(f"  Computing projected ref gradients ({n_ref} samples, proj to k={TRAK_K})...")
            ref_proj = compute_projected_gradients(
                model, tok, ref_texts, target_params, proj_random,
                desc=f"{task_name}/ref_proj")
            torch.save(ref_proj, cache_proj_ref)
            del ref_proj

        # Fisher diagonal + gradient stats (streaming, no full grad storage)
        cache_fisher = os.path.join(CACHE_DIR, f"fisher_diag_{task_name}_full.pt")
        cache_gnorms = os.path.join(CACHE_DIR, f"grad_norms_{task_name}_full.pt")
        cache_gmean = os.path.join(CACHE_DIR, f"grad_mean_{task_name}_full.pt")
        if os.path.exists(cache_fisher):
            print(f"  Cached Fisher diagonal: {cache_fisher}")
        else:
            print(f"  Computing gradient stats (streaming, {n_train} samples)...")
            fisher, gmean, gnorms = compute_gradient_stats_streaming(
                model, tok, train_texts, target_params, desc=f"{task_name}/stats")
            torch.save(fisher, cache_fisher)
            torch.save(gmean, cache_gmean)
            torch.save(gnorms, cache_gnorms)
            del fisher, gmean, gnorms

        gc.collect(); torch.cuda.empty_cache()

    restore_grad_flags(model)
    print("\nPHASE 1 COMPLETE: All representations and projected gradients cached.")

# ═══════════════════════════════════════════════════════════════════════
#  TASK P1: FM2 Continuous Metrics
# ═══════════════════════════════════════════════════════════════════════
def run_p1_fm2_continuous_metrics(tasks):
    task_id = "p1_fm2_continuous_metrics"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    # proj_random info not needed here -- we use pre-computed projected grads

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "seed": SEED, "model": MODEL_NAME, "hidden_dim": 2048,
        "methods": ["RepSim", "TRAK", "LoGra", "DDA", "RawDotIF", "DiagIF", "kNN", "BM25"],
        "scorings": ["standard", "contrastive"],
        "tasks": ["toxicity", "counterfact", "ftrace"],
        "results": {},
    }

    for task_name, task_data in tasks.items():
        print(f"\n--- Task: {task_name} ---")
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
        train_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_train_{task_name}_full.pt"), weights_only=True).numpy()
        ref_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_ref_{task_name}_full.pt"), weights_only=True).numpy()
        fisher_diag = torch.load(os.path.join(CACHE_DIR, f"fisher_diag_{task_name}_full.pt"), weights_only=True)
        grad_norms = torch.load(os.path.join(CACHE_DIR, f"grad_norms_{task_name}_full.pt"), weights_only=True)

        report_progress(task_id, f"evaluating_{task_name}", f"N={n_train}")
        results["results"][task_name] = {"n_train": n_train, "n_ref": n_ref}

        method_sims = {}

        # RepSim (cosine)
        print(f"  RepSim...")
        method_sims["RepSim"] = compute_repsim_cosine(train_reps, ref_reps)

        # kNN
        print(f"  kNN...")
        method_sims["kNN"] = compute_knn_raw(train_reps, ref_reps)

        # TRAK (random projection, k=2048)
        print(f"  TRAK (random, k={TRAK_K})...")
        method_sims["TRAK"] = projected_sim(train_proj, ref_proj, normalize=True)

        # LoGra: use SVD of projected gradients (since we can't do SVD on full D)
        # Approximate: SVD on the projected space
        print(f"  LoGra (approx)...")
        U_lo, S_lo, Vt_lo = np.linalg.svd(train_proj, full_matrices=False)
        k_lo = min(LOGRA_K, train_proj.shape[1])
        train_logra = train_proj @ Vt_lo[:k_lo].T
        ref_logra = ref_proj @ Vt_lo[:k_lo].T
        method_sims["LoGra"] = projected_sim(train_logra, ref_logra, normalize=True)
        del U_lo, S_lo, Vt_lo, train_logra, ref_logra

        # DDA: center in projected space, then low-rank
        print(f"  DDA...")
        train_centered = train_proj - train_proj.mean(axis=0, keepdims=True)
        ref_centered = ref_proj - ref_proj.mean(axis=0, keepdims=True)
        U_d, S_d, Vt_d = np.linalg.svd(train_centered, full_matrices=False)
        k_dda = min(256, train_centered.shape[1])
        train_dda = train_centered @ Vt_d[:k_dda].T
        ref_dda = ref_centered @ Vt_d[:k_dda].T
        method_sims["DDA"] = projected_sim(train_dda, ref_dda, normalize=True)
        del U_d, S_d, Vt_d, train_centered, ref_centered, train_dda, ref_dda

        # RawDotIF: approximate via projected dot product (not normalized)
        print(f"  RawDotIF (projected approx)...")
        method_sims["RawDotIF"] = projected_sim(train_proj, ref_proj, normalize=False)

        # DiagIF: approximate - scale projected grads by projected Fisher
        print(f"  DiagIF (projected approx)...")
        # Project Fisher diagonal: use variance of projected gradients as proxy
        proj_var = np.var(train_proj, axis=0) + 1e-8
        scaled_train = train_proj / np.sqrt(proj_var)[np.newaxis, :]
        method_sims["DiagIF"] = projected_sim(scaled_train, ref_proj, normalize=False)
        del scaled_train, proj_var

        # BM25
        print(f"  BM25...")
        method_sims["BM25"] = compute_bm25_scores(train_texts, ref_texts)

        # Evaluate all methods x scorings
        for method_name, sim in method_sims.items():
            results["results"][task_name][method_name] = {}
            for scoring in ["standard", "contrastive"]:
                sim_copy = sim.copy()
                eval_result = evaluate_method(sim_copy, task_name, train_data, ref_data, scoring)
                results["results"][task_name][method_name][scoring] = eval_result

                if task_name == "toxicity":
                    key_val = eval_result["rank_based"]["AUPRC"]
                else:
                    key_val = eval_result["rank_based"]["R_at_50"]
                tau_val = eval_result["continuous"]["kendall_tau"]
                print(f"    {method_name}/{scoring}: key={key_val:.4f}, tau={tau_val:.4f}")

        del method_sims, train_proj, ref_proj, fisher_diag, grad_norms
        gc.collect()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed
    results["timestamp"] = datetime.now().isoformat()

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Full-scale FM2 metrics in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P1: FM2 Contamination Injection
# ═══════════════════════════════════════════════════════════════════════
def run_p1_fm2_contamination_injection(tasks):
    task_id = "p1_fm2_contamination_injection"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "alphas": alphas, "results": {},
    }

    for task_name in ["counterfact", "toxicity"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
        train_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_train_{task_name}_full.pt"), weights_only=True).numpy()
        ref_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_ref_{task_name}_full.pt"), weights_only=True).numpy()

        results["results"][task_name] = {"n_train": n_train}

        for method_name in ["RepSim", "TRAK"]:
            if method_name == "RepSim":
                base_sim = compute_repsim_cosine(train_reps, ref_reps)
            else:
                base_sim = projected_sim(train_proj, ref_proj, normalize=True)

            method_results = {}
            for injection_mode in ["uniform", "structured", "magnitude_proportional"]:
                mode_results = {}
                for alpha in alphas:
                    sim_contam = base_sim.copy()
                    if injection_mode == "uniform":
                        mu = base_sim.mean(axis=0, keepdims=True)
                        sim_contam = base_sim + alpha * mu
                    elif injection_mode == "structured":
                        rng = np.random.RandomState(SEED + int(alpha * 100))
                        noise = rng.randn(*base_sim.shape) * base_sim.std()
                        sim_contam = base_sim + alpha * noise
                    elif injection_mode == "magnitude_proportional":
                        mag = np.abs(base_sim)
                        sim_contam = base_sim + alpha * mag * np.sign(base_sim.mean(axis=0, keepdims=True))

                    sim_corrected = sim_contam - sim_contam.mean(axis=0, keepdims=True)

                    eval_c = evaluate_method(sim_contam, task_name, train_data, ref_data, "standard")
                    eval_r = evaluate_method(sim_corrected, task_name, train_data, ref_data, "standard")
                    mode_results[str(alpha)] = {"contaminated": eval_c, "corrected": eval_r}

                    if task_name == "toxicity":
                        cv = eval_c["rank_based"]["AUPRC"]; rv = eval_r["rank_based"]["AUPRC"]
                    else:
                        cv = eval_c["rank_based"]["R_at_50"]; rv = eval_r["rank_based"]["R_at_50"]
                    print(f"    {task_name}/{method_name}/{injection_mode} a={alpha}: c={cv:.4f}, r={rv:.4f}")

                method_results[injection_mode] = mode_results
            results["results"][task_name][method_name] = method_results

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({elapsed:.0f}s)")
    mark_done(task_id, summary=f"Contamination injection in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P1: FM2 Interaction Analysis
# ═══════════════════════════════════════════════════════════════════════
def run_p1_fm2_interaction_analysis(p1_results):
    task_id = "p1_fm2_interaction_analysis"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE (CPU)")
    print("=" * 60)
    start_time = time.time()

    results = {"task_id": task_id, "candidate_id": "cand_a", "mode": "full", "anova_results": {}}

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tr = p1_results["results"].get(task_name, {})
        cells = {}
        for method in ["RepSim", "TRAK"]:
            for scoring in ["standard", "contrastive"]:
                tau = tr.get(method, {}).get(scoring, {}).get("continuous", {}).get("kendall_tau", 0.0)
                cells[f"{method}_{scoring}"] = tau

        rs, rc, ts, tc = cells.get("RepSim_standard",0), cells.get("RepSim_contrastive",0), \
                         cells.get("TRAK_standard",0), cells.get("TRAK_contrastive",0)
        fm1 = ((rs+rc)/2) - ((ts+tc)/2)
        fm2 = ((rc+tc)/2) - ((rs+ts)/2)
        inter = (rc-rs) - (tc-ts)
        gm = (rs+rc+ts+tc)/4
        ss_t = sum((v-gm)**2 for v in [rs,rc,ts,tc])
        eta1 = 2*fm1**2/ss_t if ss_t > 0 else 0
        eta2 = 2*fm2**2/ss_t if ss_t > 0 else 0

        results["anova_results"][task_name] = {
            "cells": cells, "FM1_main_effect": fm1, "FM2_main_effect": fm2,
            "interaction": inter, "eta_sq_FM1": eta1, "eta_sq_FM2": eta2,
        }
        print(f"  {task_name}: FM1={fm1:.4f}, FM2={fm2:.4f}, eta_FM1={eta1:.4f}")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"Interaction analysis in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P2: TRAK Dimension Sweep
# ═══════════════════════════════════════════════════════════════════════
def run_p2_trak_dim_sweep(tasks):
    task_id = "p2_trak_dim_sweep_fullscale"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    task_name = "counterfact"
    task_data = tasks[task_name]
    train_data = task_data["train"]
    ref_data = task_data["ref"]
    n_train = len(train_data)

    train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
    ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
    train_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_train_{task_name}_full.pt"), weights_only=True).numpy()
    ref_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_ref_{task_name}_full.pt"), weights_only=True).numpy()

    repsim_sim = compute_repsim_cosine(train_reps, ref_reps)
    repsim_eval = evaluate_method(repsim_sim, task_name, train_data, ref_data, "standard")

    random_k_values = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    pca_k_values = [32, 64, 128, 256, 512, 1024, 2048]

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "task_name": task_name, "n_train": n_train, "model": MODEL_NAME,
        "repsim_reference": repsim_eval,
        "random_projection": {}, "pca_projection": {},
    }

    # Random projection sweep: subsample from the full k=2048 projection
    # For k < 2048, just use first k columns of projected grads
    # For k > 2048, need new projection (but we only projected to 2048)
    print(f"\n  Random projection sweep (projected dim={TRAK_K})...")
    for k in random_k_values:
        if k <= TRAK_K:
            tp = train_proj[:, :k]
            rp = ref_proj[:, :k]
        else:
            # k=4096 > TRAK_K=2048: cannot do with stored projections
            # Skip or use full projection
            print(f"    k={k}: SKIPPED (> stored proj dim {TRAK_K})")
            continue

        sim = projected_sim(tp, rp, normalize=True)
        ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
        results["random_projection"][str(k)] = ev
        print(f"    k={k}: R@50={ev['rank_based']['R_at_50']:.4f}, tau={ev['continuous']['kendall_tau']:.4f}")

    # PCA projection sweep: PCA on the projected gradient space
    print(f"\n  PCA projection sweep (in projected space)...")
    U_p, S_p, Vt_p = np.linalg.svd(train_proj, full_matrices=False)

    for k in pca_k_values:
        k_actual = min(k, min(n_train, TRAK_K))
        pca_proj = Vt_p[:k_actual]  # [k, TRAK_K]
        tp_pca = train_proj @ pca_proj.T  # [N, k]
        rp_pca = ref_proj @ pca_proj.T

        sim = projected_sim(tp_pca, rp_pca, normalize=True)
        ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
        results["pca_projection"][str(k)] = {"k_actual": k_actual, **ev}
        print(f"    PCA k={k} (actual={k_actual}): R@50={ev['rank_based']['R_at_50']:.4f}, tau={ev['continuous']['kendall_tau']:.4f}")

    del U_p, S_p, Vt_p; gc.collect()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"TRAK dim sweep in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P2: RepSim Dimension Sweep
# ═══════════════════════════════════════════════════════════════════════
def run_p2_repsim_dim_sweep(tasks):
    task_id = "p2_repsim_dim_sweep_fullscale"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    k_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    results = {"task_id": task_id, "candidate_id": "cand_a", "mode": "full",
               "model": MODEL_NAME, "k_values": k_values, "results": {}}

    for task_name in ["counterfact", "toxicity", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
        d = train_reps.shape[1]

        full_sim = compute_repsim_cosine(train_reps, ref_reps)
        full_eval = evaluate_method(full_sim, task_name, train_data, ref_data, "standard")

        mean_reps = train_reps.mean(dim=0)
        centered = train_reps - mean_reps
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        task_results = {"n_train": n_train, "d": d, "full_dim": full_eval, "pca_sweep": {}}

        for k in k_values:
            k_actual = min(k, min(n_train, d))
            pca_proj = Vt[:k_actual]
            train_pca = centered @ pca_proj.T
            ref_pca = (ref_reps - mean_reps) @ pca_proj.T
            tn = F.normalize(train_pca, dim=-1)
            rn = F.normalize(ref_pca, dim=-1)
            sim = (tn @ rn.T).numpy()
            ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
            task_results["pca_sweep"][str(k)] = {"k_actual": k_actual, **ev}
            key = ev["rank_based"]["AUPRC"] if task_name == "toxicity" else ev["rank_based"]["R_at_50"]
            print(f"  {task_name} PCA k={k}: {key:.4f}")

        results["results"][task_name] = task_results
        del U, S, Vt; gc.collect()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"RepSim dim sweep in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P3: Retrieval Baselines
# ═══════════════════════════════════════════════════════════════════════
def run_p3_retrieval_baselines(tasks):
    task_id = "p3_retrieval_baselines"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    results = {"task_id": task_id, "candidate_id": "cand_a", "mode": "full",
               "methods": ["Contriever", "GTR-T5", "BM25"], "results": {}}

    all_texts = {}
    for task_name, task_data in tasks.items():
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)
        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]
        all_texts[task_name] = (train_texts, ref_texts, train_data, ref_data, n_train, n_ref)
        results["results"][task_name] = {"n_train": n_train, "n_ref": n_ref}

    # BM25 (CPU only)
    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tt, rt, td, rd, nt, nr = all_texts[task_name]
        print(f"\n  {task_name}: BM25 (N={nt})...")
        bm25_sim = compute_bm25_scores(tt, rt)
        ev = evaluate_method(bm25_sim, task_name, td, rd, "standard")
        results["results"][task_name]["BM25"] = ev
        key = ev["rank_based"]["AUPRC"] if task_name == "toxicity" else ev["rank_based"]["R_at_50"]
        print(f"    BM25: {key:.4f}")

    # Contriever
    print(f"\n  Loading Contriever...")
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModel
    ctok = AutoTokenizer.from_pretrained("facebook/contriever")
    cmod = AutoModel.from_pretrained("facebook/contriever").half().to(DEVICE)
    cmod.eval()

    def encode_contriever(texts, batch_sz=8):
        all_emb = []
        for i in range(0, len(texts), batch_sz):
            batch = texts[i:i+batch_sz]
            inp = ctok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                out = cmod(**inp)
                mask = inp["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                all_emb.append(emb.float().cpu())
            del inp, out; torch.cuda.empty_cache()
            if (i // batch_sz) % 100 == 0 and i > 0: print(f"    {i}/{len(texts)}")
        return torch.cat(all_emb, 0)

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tt, rt, td, rd, nt, nr = all_texts[task_name]
        print(f"\n  {task_name}: Contriever (N={nt})...")
        te = encode_contriever(tt); re = encode_contriever(rt)
        sim = (F.normalize(te, dim=-1) @ F.normalize(re, dim=-1).T).numpy()
        ev = evaluate_method(sim, task_name, td, rd, "standard")
        results["results"][task_name]["Contriever"] = ev
        key = ev["rank_based"]["AUPRC"] if task_name == "toxicity" else ev["rank_based"]["R_at_50"]
        print(f"    Contriever: {key:.4f}")

    del cmod, ctok; gc.collect(); torch.cuda.empty_cache()

    # GTR-T5
    print(f"\n  Loading GTR-T5...")
    from sentence_transformers import SentenceTransformer
    gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device=DEVICE)

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        tt, rt, td, rd, nt, nr = all_texts[task_name]
        print(f"\n  {task_name}: GTR-T5 (N={nt})...")
        te = torch.tensor(gtr.encode(tt, batch_size=8, show_progress_bar=False))
        re = torch.tensor(gtr.encode(rt, batch_size=8, show_progress_bar=False))
        sim = (F.normalize(te, dim=-1) @ F.normalize(re, dim=-1).T).numpy()
        ev = evaluate_method(sim, task_name, td, rd, "standard")
        results["results"][task_name]["GTR-T5"] = ev
        key = ev["rank_based"]["AUPRC"] if task_name == "toxicity" else ev["rank_based"]["R_at_50"]
        print(f"    GTR-T5: {key:.4f}")

    del gtr; gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"Retrieval baselines in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P4: Gap Decomposition
# ═══════════════════════════════════════════════════════════════════════
def run_p4_gap_decomposition(tasks, model, tok):
    task_id = "p4_gap_decomposition"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    task_name = "counterfact"
    task_data = tasks[task_name]
    train_data = task_data["train"]
    ref_data = task_data["ref"]
    n_train = len(train_data)
    n_ref = len(ref_data)
    d = 2048

    train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
    ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

    train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
    ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
    train_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_train_{task_name}_full.pt"), weights_only=True).numpy()
    ref_proj = torch.load(os.path.join(CACHE_DIR, f"proj_grads_ref_{task_name}_full.pt"), weights_only=True).numpy()

    # RepSim baseline
    repsim_sim = compute_repsim_cosine(train_reps, ref_reps)
    repsim_eval = evaluate_method(repsim_sim, task_name, train_data, ref_data, "standard")
    repsim_r50 = repsim_eval["rank_based"]["R_at_50"]

    # Standard TRAK-PCA at k=d (in projected space)
    U_p, S_p, Vt_p = np.linalg.svd(train_proj, full_matrices=False)
    k_d = min(d, min(n_train, TRAK_K))
    tp_pca = train_proj @ Vt_p[:k_d].T
    rp_pca = ref_proj @ Vt_p[:k_d].T
    standard_sim = projected_sim(tp_pca, rp_pca, normalize=True)
    standard_eval = evaluate_method(standard_sim, task_name, train_data, ref_data, "standard")
    standard_r50 = standard_eval["rank_based"]["R_at_50"]
    total_gap = repsim_r50 - standard_r50

    print(f"  RepSim R@50={repsim_r50:.4f}, TRAK-PCA R@50={standard_r50:.4f}, Gap={total_gap:.4f}")

    results = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "task_name": task_name, "n_train": n_train, "d": d,
        "repsim": repsim_eval, "standard_trak_pca": standard_eval, "factors": {},
    }

    # Factor (a): Last-layer-only gradients
    print(f"\n  Factor (a): Last-layer-only projected gradients...")
    # Compute last-layer projected gradients
    target_params_ll, _, D_ll = setup_target_params(model, layer_filter="last_layer_all")
    proj_ll = make_random_projection(D_ll, TRAK_K, seed=SEED+100)  # CountSketch projector for last-layer

    cache_ll_train = os.path.join(CACHE_DIR, f"proj_grads_lastlayer_train_{task_name}_full.pt")
    cache_ll_ref = os.path.join(CACHE_DIR, f"proj_grads_lastlayer_ref_{task_name}_full.pt")

    if os.path.exists(cache_ll_train):
        ll_train = torch.load(cache_ll_train, weights_only=True).numpy()
    else:
        ll_train_t = compute_projected_gradients(model, tok, train_texts, target_params_ll,
                                                  proj_ll, desc="lastlayer/train")
        torch.save(ll_train_t, cache_ll_train)
        ll_train = ll_train_t.numpy()
        del ll_train_t

    if os.path.exists(cache_ll_ref):
        ll_ref = torch.load(cache_ll_ref, weights_only=True).numpy()
    else:
        ll_ref_t = compute_projected_gradients(model, tok, ref_texts, target_params_ll,
                                                proj_ll, desc="lastlayer/ref")
        torch.save(ll_ref_t, cache_ll_ref)
        ll_ref = ll_ref_t.numpy()
        del ll_ref_t

    restore_grad_flags(model)

    # Last-layer PCA at k=d
    U_ll, S_ll, Vt_ll = np.linalg.svd(ll_train, full_matrices=False)
    k_ll = min(d, min(n_train, TRAK_K))
    ll_pca_t = ll_train @ Vt_ll[:k_ll].T
    ll_pca_r = ll_ref @ Vt_ll[:k_ll].T
    ll_sim = projected_sim(ll_pca_t, ll_pca_r, normalize=True)
    ll_eval = evaluate_method(ll_sim, task_name, train_data, ref_data, "standard")
    ll_r50 = ll_eval["rank_based"]["R_at_50"]
    results["factors"]["a_last_layer"] = {"eval": ll_eval, "D_last": D_ll,
                                           "gap_reduction": ll_r50 - standard_r50}
    print(f"    Last-layer TRAK-PCA: R@50={ll_r50:.4f}, delta={ll_r50-standard_r50:.4f}")

    # Factor (b): Cosine-normalized projected gradients
    print(f"\n  Factor (b): Cosine-normalized projected gradients...")
    train_norm = train_proj / (np.linalg.norm(train_proj, axis=1, keepdims=True) + 1e-10)
    ref_norm = ref_proj / (np.linalg.norm(ref_proj, axis=1, keepdims=True) + 1e-10)
    U_n, S_n, Vt_n = np.linalg.svd(train_norm, full_matrices=False)
    k_n = min(d, min(n_train, TRAK_K))
    tn_pca = train_norm @ Vt_n[:k_n].T
    rn_pca = ref_norm @ Vt_n[:k_n].T
    norm_sim = projected_sim(tn_pca, rn_pca, normalize=True)
    norm_eval = evaluate_method(norm_sim, task_name, train_data, ref_data, "standard")
    norm_r50 = norm_eval["rank_based"]["R_at_50"]
    results["factors"]["b_cosine_norm"] = {"eval": norm_eval, "gap_reduction": norm_r50 - standard_r50}
    print(f"    Cosine-norm TRAK-PCA: R@50={norm_r50:.4f}, delta={norm_r50-standard_r50:.4f}")

    # Factor (c): Combined
    print(f"\n  Factor (c): Combined last-layer + cosine-norm...")
    ll_tnorm = ll_train / (np.linalg.norm(ll_train, axis=1, keepdims=True) + 1e-10)
    ll_rnorm = ll_ref / (np.linalg.norm(ll_ref, axis=1, keepdims=True) + 1e-10)
    U_c, S_c, Vt_c = np.linalg.svd(ll_tnorm, full_matrices=False)
    k_c = min(d, min(n_train, TRAK_K))
    ct = ll_tnorm @ Vt_c[:k_c].T
    cr = ll_rnorm @ Vt_c[:k_c].T
    comb_sim = projected_sim(ct, cr, normalize=True)
    comb_eval = evaluate_method(comb_sim, task_name, train_data, ref_data, "standard")
    comb_r50 = comb_eval["rank_based"]["R_at_50"]
    results["factors"]["c_combined"] = {"eval": comb_eval, "gap_reduction": comb_r50 - standard_r50}
    print(f"    Combined TRAK-PCA: R@50={comb_r50:.4f}, delta={comb_r50-standard_r50:.4f}")

    residual = repsim_r50 - comb_r50
    results["factors"]["d_residual"] = {"gap_pp": residual}
    results["summary"] = {
        "total_gap_pp": total_gap,
        "factor_a_pp": ll_r50 - standard_r50,
        "factor_b_pp": norm_r50 - standard_r50,
        "factor_c_pp": comb_r50 - standard_r50,
        "factor_d_residual_pp": residual,
        "combined_explained_pct": ((comb_r50 - standard_r50) / total_gap * 100) if total_gap > 0 else 0,
    }

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"Gap decomposition in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P4: Layer Sweep
# ═══════════════════════════════════════════════════════════════════════
def run_p4_layer_sweep(tasks, model, tok):
    task_id = "p4_layer_sweep"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    # Pythia-1B: 16 transformer layers. hidden_states[0]=emb, [1]=after layer0, ..., [16]=after layer15
    layer_indices = [1, 5, 9, 13, 16]
    layer_names = ["layer_0", "layer_4", "layer_8", "layer_12", "layer_15"]

    results = {"task_id": task_id, "candidate_id": "cand_a", "mode": "full",
               "model": MODEL_NAME, "layers": layer_names, "results": {}}

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)
        n_ref = len(ref_data)

        train_texts = [task_data["fmt"](train_data[i]) for i in range(n_train)]
        ref_texts = [task_data["fmt"](ref_data[i]) for i in range(n_ref)]

        task_results = {"n_train": n_train}

        for li, ln in zip(layer_indices, layer_names):
            train_reps = get_or_extract_reps(model, tok, train_texts, task_name,
                                              layer_suffix=ln, layer_idx=li)
            ref_reps = get_or_extract_ref_reps(model, tok, ref_texts, task_name,
                                                layer_suffix=ln, layer_idx=li)

            sim = compute_repsim_cosine(train_reps, ref_reps)
            ev = evaluate_method(sim, task_name, train_data, ref_data, "standard")
            task_results[ln] = ev

            key = ev["rank_based"]["AUPRC"] if task_name == "toxicity" else ev["rank_based"]["R_at_50"]
            print(f"  {task_name}/{ln}: {key:.4f}")
            del train_reps, ref_reps; gc.collect(); torch.cuda.empty_cache()

        results["results"][task_name] = task_results

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"Layer sweep in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P4: Cosine vs Euclidean
# ═══════════════════════════════════════════════════════════════════════
def run_p4_cosine_vs_euclidean(tasks):
    task_id = "p4_cosine_vs_euclidean"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    results = {"task_id": task_id, "candidate_id": "cand_a", "mode": "full",
               "model": MODEL_NAME, "results": {}}

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)

        tr = {"n_train": len(train_data)}

        sim_cos = compute_repsim_cosine(train_reps, ref_reps)
        tr["cosine"] = evaluate_method(sim_cos, task_name, train_data, ref_data, "standard")

        sim_euc = compute_repsim_euclidean(train_reps, ref_reps)
        tr["euclidean"] = evaluate_method(sim_euc, task_name, train_data, ref_data, "standard")

        sim_dot = compute_repsim_dot(train_reps, ref_reps)
        tr["dot_product"] = evaluate_method(sim_dot, task_name, train_data, ref_data, "standard")

        # Rank correlations (subsample for speed)
        fc, fe, fd = sim_cos.flatten()[:10000], sim_euc.flatten()[:10000], sim_dot.flatten()[:10000]
        tau_ce, _ = kendalltau(fc, fe)
        tau_cd, _ = kendalltau(fc, fd)
        tr["rank_correlations"] = {
            "cosine_vs_euclidean_tau": float(tau_ce) if not np.isnan(tau_ce) else 0.0,
            "cosine_vs_dot_tau": float(tau_cd) if not np.isnan(tau_cd) else 0.0,
        }

        norms = torch.norm(train_reps, dim=-1).numpy()
        tr["norm_analysis"] = {"mean": float(norms.mean()), "std": float(norms.std()),
                               "cv": float(norms.std()/norms.mean()) if norms.mean() > 0 else 0}

        results["results"][task_name] = tr

        kc = tr["cosine"]["rank_based"]["AUPRC"] if task_name == "toxicity" else tr["cosine"]["rank_based"]["R_at_50"]
        ke = tr["euclidean"]["rank_based"]["AUPRC"] if task_name == "toxicity" else tr["euclidean"]["rank_based"]["R_at_50"]
        kd = tr["dot_product"]["rank_based"]["AUPRC"] if task_name == "toxicity" else tr["dot_product"]["rank_based"]["R_at_50"]
        print(f"  {task_name}: cos={kc:.4f}, euc={ke:.4f}, dot={kd:.4f}")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"Cosine vs Euclidean in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  TASK P5: PCA-Whitened Attribution
# ═══════════════════════════════════════════════════════════════════════
def run_p5_pca_whitened(tasks):
    task_id = "p5_pca_whitened_attribution"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    k_values = [16, 32, 64, 128, 256, 512]
    results = {"task_id": task_id, "candidate_id": "cand_a", "mode": "full",
               "model": MODEL_NAME, "k_values": k_values, "results": {}}

    for task_name in ["toxicity", "counterfact", "ftrace"]:
        task_data = tasks[task_name]
        train_data = task_data["train"]
        ref_data = task_data["ref"]
        n_train = len(train_data)

        train_reps = torch.load(os.path.join(CACHE_DIR, f"reps_train_{task_name}_full.pt"), weights_only=True)
        ref_reps = torch.load(os.path.join(CACHE_DIR, f"reps_ref_{task_name}_full.pt"), weights_only=True)
        d = train_reps.shape[1]

        std_sim = compute_repsim_cosine(train_reps, ref_reps)
        std_eval = evaluate_method(std_sim, task_name, train_data, ref_data, "standard")

        mean_reps = train_reps.mean(dim=0)
        centered = train_reps - mean_reps
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        task_results = {"n_train": n_train, "d": d, "standard_repsim": std_eval, "pca_whitened": {}}

        for k in k_values:
            k_actual = min(k, min(n_train-1, d))
            pca_proj = Vt[:k_actual]
            train_pca = centered @ pca_proj.T
            ref_pca = (ref_reps - mean_reps) @ pca_proj.T
            cov_pca = (train_pca.T @ train_pca) / (n_train - 1)

            # LW whitening
            try:
                lw = LedoitWolf()
                lw.fit(train_pca.numpy())
                cov_lw = torch.tensor(lw.covariance_, dtype=torch.float32)
                ev_lw, evec_lw = torch.linalg.eigh(cov_lw)
                ev_lw = ev_lw.clamp(min=1e-6)
                inv_sqrt = evec_lw @ torch.diag(1.0/ev_lw.sqrt()) @ evec_lw.T
                tw = train_pca @ inv_sqrt; rw = ref_pca @ inv_sqrt
                sim_lw = (F.normalize(tw, dim=-1) @ F.normalize(rw, dim=-1).T).numpy()
                lw_eval = evaluate_method(sim_lw, task_name, train_data, ref_data, "standard")
            except Exception as e:
                lw_eval = {"error": str(e)}

            # Ridge CV
            try:
                lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
                best_lam, best_cv = lambdas[0], -np.inf
                n_fold = 5; fold_sz = n_train // n_fold
                for lam in lambdas:
                    fscores = []
                    for fi in range(n_fold):
                        vs, ve = fi*fold_sz, min((fi+1)*fold_sz, n_train)
                        ti = list(range(0,vs)) + list(range(ve,n_train))
                        tp = train_pca[ti]
                        cf = (tp.T @ tp)/(len(ti)-1) + lam*torch.eye(k_actual)
                        evf, evcf = torch.linalg.eigh(cf)
                        evf = evf.clamp(min=1e-6)
                        inv_f = evcf @ torch.diag(1.0/evf.sqrt()) @ evcf.T
                        vw = train_pca[vs:ve] @ inv_f; tw2 = tp @ inv_f
                        sim_cv = (F.normalize(vw,dim=-1) @ F.normalize(tw2,dim=-1).T).numpy()
                        fscores.append(float(np.mean(np.max(sim_cv, axis=1))))
                    avg = np.mean(fscores)
                    if avg > best_cv: best_cv = avg; best_lam = lam

                cr = cov_pca + best_lam*torch.eye(k_actual)
                evr, evcr = torch.linalg.eigh(cr)
                evr = evr.clamp(min=1e-6)
                inv_r = evcr @ torch.diag(1.0/evr.sqrt()) @ evcr.T
                trr = train_pca @ inv_r; rrr = ref_pca @ inv_r
                sim_r = (F.normalize(trr, dim=-1) @ F.normalize(rrr, dim=-1).T).numpy()
                ridge_eval = {"best_lambda": best_lam, **evaluate_method(sim_r, task_name, train_data, ref_data, "standard")}
            except Exception as e:
                ridge_eval = {"error": str(e)}

            n_over_k = n_train / k_actual
            task_results["pca_whitened"][str(k)] = {
                "k_actual": k_actual, "N_over_k": n_over_k,
                "LW_whitened": lw_eval, "ridge_cv": ridge_eval,
            }

            if task_name == "toxicity":
                sk = std_eval["rank_based"]["AUPRC"]
                lk = lw_eval.get("rank_based",{}).get("AUPRC","err") if isinstance(lw_eval,dict) and "rank_based" in lw_eval else "err"
            else:
                sk = std_eval["rank_based"]["R_at_50"]
                lk = lw_eval.get("rank_based",{}).get("R_at_50","err") if isinstance(lw_eval,dict) and "rank_based" in lw_eval else "err"
            print(f"  {task_name} k={k} (N/k={n_over_k:.0f}): LW={lk}, std={sk:.4f}")

        results["results"][task_name] = task_results
        del U, S, Vt; gc.collect()

    elapsed = time.time() - start_time
    results["elapsed_sec"] = elapsed; results["timestamp"] = datetime.now().isoformat()
    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    mark_done(task_id, summary=f"PCA whitened in {elapsed:.0f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════
#  Final Analysis
# ═══════════════════════════════════════════════════════════════════════
def run_final_analysis():
    task_id = "final_analysis"
    write_pid(task_id)
    print("\n" + "=" * 60)
    print(f"TASK: {task_id} -- FULL SCALE")
    print("=" * 60)
    start_time = time.time()

    fmap = {
        "p1_continuous": "p1_fm2_continuous_metrics.json",
        "p1_contamination": "p1_fm2_contamination_injection.json",
        "p1_interaction": "p1_fm2_interaction_analysis.json",
        "p2_eigenspectrum": "p2_eigenspectrum_fullscale.json",
        "p2_trak_sweep": "p2_trak_dim_sweep_fullscale.json",
        "p2_repsim_sweep": "p2_repsim_dim_sweep_fullscale.json",
        "p3_retrieval": "p3_retrieval_baselines.json",
        "p4_gap": "p4_gap_decomposition.json",
        "p4_layer": "p4_layer_sweep.json",
        "p4_similarity": "p4_cosine_vs_euclidean.json",
        "p5_whitened": "p5_pca_whitened_attribution.json",
    }

    all_r = {}
    for k, fn in fmap.items():
        fp = os.path.join(FULL_DIR, fn)
        if os.path.exists(fp):
            with open(fp) as f: all_r[k] = json.load(f)
            print(f"  Loaded: {fn} (mode={all_r[k].get('mode','?')})")
        else:
            print(f"  MISSING: {fn}")

    analysis = {
        "task_id": task_id, "candidate_id": "cand_a", "mode": "full",
        "timestamp": datetime.now().isoformat(),
    }

    # Decision gates
    gates = {}

    # Gate 1: FM2
    p1 = all_r.get("p1_continuous", {})
    max_tau_gain = 0.0
    for tn in ["toxicity","counterfact","ftrace"]:
        for m in ["TRAK","RepSim"]:
            std = p1.get("results",{}).get(tn,{}).get(m,{}).get("standard",{}).get("continuous",{}).get("kendall_tau",0)
            con = p1.get("results",{}).get(tn,{}).get(m,{}).get("contrastive",{}).get("continuous",{}).get("kendall_tau",0)
            max_tau_gain = max(max_tau_gain, con - std)
    gates["gate_1_fm2"] = {"h2_passes": max_tau_gain >= 0.05, "max_tau_gain": max_tau_gain}

    # Gate 2: Retrieval
    p3 = all_r.get("p3_retrieval", {})
    matching = 0
    for tn in ["toxicity","counterfact","ftrace"]:
        rep_eval = p1.get("results",{}).get(tn,{}).get("RepSim",{}).get("standard",{})
        rk = rep_eval.get("rank_based",{}).get("AUPRC" if tn=="toxicity" else "R_at_50", 0)
        for rm in ["Contriever","GTR-T5"]:
            re = p3.get("results",{}).get(tn,{}).get(rm,{})
            rmk = re.get("rank_based",{}).get("AUPRC" if tn=="toxicity" else "R_at_50", 0)
            if abs(rmk - rk) < 0.03: matching += 1; break
    gates["gate_2_retrieval"] = {"matching_tasks": matching, "triggers": matching >= 2}

    # Gate 3: Gap
    g4 = all_r.get("p4_gap", {})
    gs = g4.get("summary", {})
    rep_r = g4.get("repsim",{}).get("rank_based",{}).get("R_at_50",0)
    comb_r = g4.get("factors",{}).get("c_combined",{}).get("eval",{}).get("rank_based",{}).get("R_at_50",0)
    gates["gate_3_gap"] = {"within_10pp": (rep_r - comb_r) <= 0.10, "remaining_gap": rep_r - comb_r}

    # Gate 4: Whitening
    p5 = all_r.get("p5_whitened", {})
    best_gain = 0
    for tn in ["toxicity","counterfact","ftrace"]:
        tr = p5.get("results",{}).get(tn,{})
        sk = tr.get("standard_repsim",{}).get("rank_based",{}).get("AUPRC" if tn=="toxicity" else "R_at_50", 0)
        for kv in tr.get("pca_whitened",{}).values():
            lw = kv.get("LW_whitened",{})
            if isinstance(lw, dict) and "rank_based" in lw:
                wk = lw["rank_based"].get("AUPRC" if tn=="toxicity" else "R_at_50", 0)
                best_gain = max(best_gain, wk - sk)
    gates["gate_4_whitening"] = {"passes": best_gain >= 0.03, "best_gain_pp": best_gain}

    analysis["decision_gates"] = gates

    # Method tournament
    tournament = {}
    for tn in ["toxicity","counterfact","ftrace"]:
        t = {}
        for m in ["RepSim","TRAK","LoGra","DDA","RawDotIF","DiagIF","kNN","BM25"]:
            e = p1.get("results",{}).get(tn,{}).get(m,{}).get("standard",{})
            t[m] = e.get("rank_based",{}).get("AUPRC" if tn=="toxicity" else "R_at_50", 0)
        for rm in ["Contriever","GTR-T5"]:
            e = p3.get("results",{}).get(tn,{}).get(rm,{})
            t[rm] = e.get("rank_based",{}).get("AUPRC" if tn=="toxicity" else "R_at_50", 0)
        tournament[tn] = dict(sorted(t.items(), key=lambda x: -x[1]))
    analysis["method_tournament"] = tournament
    analysis["results_modes"] = {k: v.get("mode","?") for k,v in all_r.items()}

    elapsed = time.time() - start_time
    analysis["elapsed_sec"] = elapsed

    out_path = os.path.join(FULL_DIR, f"{task_id}.json")
    with open(out_path, "w") as f: json.dump(analysis, f, indent=2)

    # Summary MD
    with open(os.path.join(FULL_DIR, "final_analysis_summary.md"), "w") as f:
        f.write("# CRA Full-Scale Results\n\n")
        for gn, gd in gates.items():
            f.write(f"## {gn}\n")
            for k,v in gd.items(): f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write("## Tournament\n\n")
        for tn, t in tournament.items():
            f.write(f"### {tn}\n")
            for m, s in t.items(): f.write(f"- {m}: {s:.4f}\n")
            f.write("\n")

    print(f"\nSaved: {out_path}")
    mark_done(task_id, summary=f"Final analysis in {elapsed:.0f}s")
    return analysis

# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  CRA FULL-SCALE EXPERIMENT RUNNER v2 (Memory-Efficient)")
    print(f"  GPU: {DEVICE} (CUDA_VISIBLE_DEVICES=3)")
    print(f"  Start: {datetime.now().isoformat()}")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, Free: {torch.cuda.mem_get_info()[0]/1024**2:.0f}MiB")
    else:
        print("ERROR: No GPU!"); sys.exit(1)

    overall_start = time.time()

    # Load datasets
    tasks = load_all_tasks_full()

    # Load model
    model, tok = load_model_pythia1b()

    # Phase 1: Extract all reps + projected grads (shared computation)
    phase1_extract_all(tasks, model, tok)

    # P1: FM2 Continuous Metrics
    p1_results = run_p1_fm2_continuous_metrics(tasks)

    # P1: Contamination Injection
    run_p1_fm2_contamination_injection(tasks)

    # P1: Interaction Analysis
    run_p1_fm2_interaction_analysis(p1_results)

    # P2: TRAK Dimension Sweep
    run_p2_trak_dim_sweep(tasks)

    # P2: RepSim Dimension Sweep
    run_p2_repsim_dim_sweep(tasks)

    # P3: Retrieval Baselines (unload main model, use retrieval models)
    unload_model(model)
    run_p3_retrieval_baselines(tasks)
    model, tok = load_model_pythia1b()

    # P4: Gap Decomposition
    run_p4_gap_decomposition(tasks, model, tok)

    # P4: Layer Sweep
    run_p4_layer_sweep(tasks, model, tok)

    # P4: Cosine vs Euclidean
    run_p4_cosine_vs_euclidean(tasks)

    # P5: PCA Whitened Attribution
    run_p5_pca_whitened(tasks)

    # Final Analysis
    unload_model(model)
    run_final_analysis()

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  ALL TASKS COMPLETE. Total: {total/60:.1f} min ({total/3600:.1f} hr)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
