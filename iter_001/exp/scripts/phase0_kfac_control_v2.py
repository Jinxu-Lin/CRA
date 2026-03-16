#!/usr/bin/env python3
"""
Phase 0b: K-FAC Hessian Control on Pythia-70M (H6 -- CRITICAL) -- v2

Fixes from v1:
  1. K-FAC factor accumulation: count samples correctly per layer
  2. Symmetrize covariance matrices and clamp eigenvalues to non-negative
  3. Add score distribution diagnostics to detect trivial correlations
  4. Multiple damping values to test sensitivity
  5. Add normalized (rank-based) AUPRC to guard against score scale artifacts

CRITICAL DECISION GATE:
  If K-FAC IF LDS >= RepSim LDS - 5pp → CRA thesis requires fundamental revision.
  If K-FAC IF LDS < RepSim LDS - 5pp → H6 not falsified, proceed with CRA.

Approach:
  1. Load Pythia-70M (d=512, 6 layers, ~70M params)
  2. Use DATE-LM Toxicity (AUPRC) and Counterfact (Recall@50, MRR) tasks
  3. Compute RepSim (last-layer cosine similarity) as the representation-space baseline
  4. Compute K-FAC IF with proper Kronecker factorization
  5. Also compute vanilla IF (diagonal Hessian) and TRAK as parameter-space controls
  6. Compare all methods
"""

import os, sys, json, time, gc, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_curve, auc

# ── Config ──────────────────────────────────────────────────────────────
TASK_ID = "phase0_kfac_control"
SEED = 42
PILOT_N_TRAIN = 100
MODEL_NAME = "EleutherAI/pythia-70m"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-70m/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE0_DIR = os.path.join(RESULTS_DIR, "phase0")
PILOTS_DIR = os.path.join(RESULTS_DIR, "pilots")

DEVICE = "cuda:0"

# K-FAC config
KFAC_TARGET_LAYERS = [4, 5]  # Last 2 transformer layers
KFAC_DAMPING_VALUES = [1e-2, 1e-3, 1e-4]  # Test sensitivity
KFAC_DAMPING_DEFAULT = 1e-3
TRAK_K = 2048

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


def mark_done(status="success", summary="", results=None):
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
    """Compute AUPRC: higher score = more likely unsafe."""
    labels = np.zeros(n_total)
    labels[list(unsafe_indices)] = 1
    if sum(labels) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def compute_auprc_rankbased(scores, unsafe_indices, n_total):
    """AUPRC on rank-transformed scores -- immune to score scale artifacts."""
    ranks = np.argsort(np.argsort(scores)).astype(float)  # rank transform
    return compute_auprc(ranks, unsafe_indices, n_total)


def compute_factual_metrics(scores_list, fact_indices_list, k=50):
    recalls, mrrs = [], []
    for scores, fi in zip(scores_list, fact_indices_list):
        if not fi:
            recalls.append(0.0)
            mrrs.append(0.0)
            continue
        si = np.argsort(-np.array(scores))
        topk = set(si[:k].tolist())
        recalls.append(len([i for i in fi if i in topk]) / len(fi))
        r2r = {idx: rank + 1 for rank, idx in enumerate(si)}
        ranks = [r2r[i] for i in fi if i in r2r]
        mrrs.append(1.0 / min(ranks) if ranks else 0.0)
    return float(np.mean(recalls)), float(np.mean(mrrs))


def score_diagnostics(scores, name, unsafe_indices=None):
    """Print score distribution diagnostics."""
    s = np.array(scores) if not isinstance(scores, np.ndarray) else scores
    info = {
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "median": float(np.median(s)),
        "pct_positive": float(np.mean(s > 0) * 100),
        "pct_negative": float(np.mean(s < 0) * 100),
    }
    if unsafe_indices is not None and len(unsafe_indices) > 0:
        safe_mask = np.ones(len(s), dtype=bool)
        safe_mask[list(unsafe_indices)] = False
        unsafe_scores = s[list(unsafe_indices)]
        safe_scores = s[safe_mask]
        info["unsafe_mean"] = float(np.mean(unsafe_scores))
        info["safe_mean"] = float(np.mean(safe_scores))
        info["separation"] = float(np.mean(unsafe_scores) - np.mean(safe_scores))
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(unsafe_scores) + np.var(safe_scores)) / 2)
        info["cohens_d"] = float(info["separation"] / max(pooled_std, 1e-10))
    print(f"  [{name}] mean={info['mean']:.4e}, std={info['std']:.4e}, "
          f"range=[{info['min']:.4e}, {info['max']:.4e}], %pos={info['pct_positive']:.0f}%")
    if "separation" in info:
        print(f"    unsafe_mean={info['unsafe_mean']:.4e}, safe_mean={info['safe_mean']:.4e}, "
              f"separation={info['separation']:.4e}, Cohen's d={info['cohens_d']:.2f}")
    return info


def fmt(sample):
    return sample["prompt"] + " " + sample["response"]


# ── Data loading ────────────────────────────────────────────────────────
def load_tasks():
    report_progress("loading_data", "Loading DATE-LM tasks", 0.02)
    tasks = {}

    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {"train": tox["train"], "ref": tox["ref"], "metric": "AUPRC"}
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {"train": cf["train"], "ref": cf["ref"], "metric": "Recall@50+MRR"}
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")

    return tasks


def create_pilot_subsets(tasks):
    pilot_subsets = {}
    for tn, ti in tasks.items():
        full_train = ti["train"]
        n_total = len(full_train)
        rng = np.random.RandomState(SEED)
        if tn == "toxicity":
            unsafe_idx = [i for i in range(n_total) if full_train[i]["type"] == "Unsafe"]
            safe_idx = [i for i in range(n_total) if full_train[i]["type"] != "Unsafe"]
            n_unsafe = min(len(unsafe_idx), max(PILOT_N_TRAIN // 5, 5))
            n_safe = PILOT_N_TRAIN - n_unsafe
            chosen_unsafe = rng.choice(unsafe_idx, n_unsafe, replace=False).tolist()
            chosen_safe = rng.choice(safe_idx, min(n_safe, len(safe_idx)), replace=False).tolist()
            pilot_idx = sorted(chosen_unsafe + chosen_safe)
        else:
            pilot_idx = sorted(rng.choice(n_total, min(PILOT_N_TRAIN, n_total), replace=False).tolist())
        pilot_subsets[tn] = {
            "data": full_train.select(pilot_idx),
            "indices": pilot_idx,
            "ref": ti["ref"],
        }
        print(f"[pilot/{tn}] {len(pilot_idx)} samples")
    return pilot_subsets


# ── Model loading ───────────────────────────────────────────────────────
def load_model():
    report_progress("loading_model", "Loading Pythia-70M (float32 for K-FAC)", 0.05)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, torch_dtype=torch.float32, device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded: hidden_dim={model.config.hidden_size}, layers={model.config.num_hidden_layers}")
    print(f"  Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    return model, tok


# ── RepSim (baseline) ──────────────────────────────────────────────────
def extract_reps(model, tok, texts, bs=16, max_len=512):
    """Extract last-layer, last-token representations."""
    reps = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEVICE)
        with torch.no_grad():
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]  # [B, seq, d]
            seq_lens = inp["attention_mask"].sum(1) - 1
            bi = torch.arange(h.size(0), device=DEVICE)
            r = F.normalize(h[bi, seq_lens].float(), dim=-1)
            reps.append(r.cpu())
    return torch.cat(reps, dim=0)


def run_repsim(model, tok, task_name, train_data, ref_data, n_train):
    report_progress("repsim", f"RepSim on {task_name}", 0.10)
    t0 = time.time()
    train_texts = [fmt(train_data[i]) for i in range(n_train)]
    ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]

    train_reps = extract_reps(model, tok, train_texts)
    ref_reps = extract_reps(model, tok, ref_texts)
    sim = train_reps @ ref_reps.T  # [n_train, n_ref]

    elapsed = time.time() - t0
    print(f"[RepSim/{task_name}] Done in {elapsed:.1f}s, sim shape={sim.shape}")

    if task_name == "counterfact":
        return [sim[:, j].numpy() for j in range(sim.shape[1])], elapsed
    return sim.mean(dim=1).numpy(), elapsed


# ── K-FAC Influence Functions (v2 -- corrected) ────────────────────────
class KFACInfluenceV2:
    """
    K-FAC IF with corrected factor accumulation.

    Key fixes:
      1. Separate sample counting per layer
      2. Properly symmetrize covariance matrices
      3. Clamp eigenvalues to non-negative after eigendecomp
      4. Support multiple damping values
    """

    def __init__(self, model, target_layer_indices, damping=1e-3, device="cuda:0"):
        self.model = model
        self.target_layers = target_layer_indices
        self.damping = damping
        self.device = device

        self.target_params = []
        self._hooks = []
        self._activations = {}
        self._grad_outputs = {}

        for li in self.target_layers:
            prefix = f"gpt_neox.layers.{li}"
            for suffix in ["attention.dense", "mlp.dense_4h_to_h"]:
                full_name = f"{prefix}.{suffix}"
                module = dict(model.named_modules())[full_name]
                weight = module.weight
                self.target_params.append({
                    "name": full_name,
                    "module": module,
                    "weight": weight,
                    "out_dim": weight.shape[0],
                    "in_dim": weight.shape[1],
                })
                print(f"  K-FAC target: {full_name} shape={weight.shape}")

        self.A_covs = {}
        self.G_covs = {}
        self.A_eig = {}
        self.G_eig = {}

    def _register_hooks(self):
        self._hooks = []
        for info in self.target_params:
            name = info["name"]
            module = info["module"]

            def make_fwd_hook(n):
                def hook(mod, inp, out):
                    self._activations[n] = inp[0].detach()
                return hook

            def make_bwd_hook(n):
                def hook(mod, grad_in, grad_out):
                    self._grad_outputs[n] = grad_out[0].detach()
                return hook

            h1 = module.register_forward_hook(make_fwd_hook(name))
            h2 = module.register_full_backward_hook(make_bwd_hook(name))
            self._hooks.extend([h1, h2])

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute_kronecker_factors(self, tok, texts, bs=8, max_len=512):
        """
        Compute A and G Kronecker factors.
        FIX: Track sample count per layer independently.
        Use outer product of per-sample-mean activations/gradients.
        """
        report_progress("kfac_factors", "Computing K-FAC Kronecker factors (v2)", 0.20)
        self._register_hooks()

        # Initialize accumulators with per-layer sample counts
        sample_counts = {}
        for info in self.target_params:
            name = info["name"]
            in_dim = info["in_dim"]
            out_dim = info["out_dim"]
            self.A_covs[name] = torch.zeros(in_dim, in_dim, device=self.device, dtype=torch.float64)
            self.G_covs[name] = torch.zeros(out_dim, out_dim, device=self.device, dtype=torch.float64)
            sample_counts[name] = 0

        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            inp = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(self.device)

            self.model.zero_grad()
            out = self.model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                labels=inp["input_ids"],
            )
            out.loss.backward()

            mask = inp["attention_mask"]

            for info in self.target_params:
                name = info["name"]
                act = self._activations[name]    # [B, T, in_dim]
                grad = self._grad_outputs[name]  # [B, T, out_dim]

                for b in range(act.shape[0]):
                    valid = mask[b].bool()
                    a_valid = act[b, valid].double()  # [T_valid, in_dim]
                    g_valid = grad[b, valid].double()  # [T_valid, out_dim]

                    # Mean over tokens for this sample
                    a_mean = a_valid.mean(0)
                    g_mean = g_valid.mean(0)

                    self.A_covs[name] += torch.outer(a_mean, a_mean)
                    self.G_covs[name] += torch.outer(g_mean, g_mean)
                    sample_counts[name] += 1

            self.model.zero_grad(set_to_none=True)
            self._activations.clear()
            self._grad_outputs.clear()

            if (i + bs) % 20 == 0 or i + bs >= len(texts):
                print(f"  K-FAC factors: {min(i + bs, len(texts))}/{len(texts)}")

        # Normalize per layer
        for info in self.target_params:
            name = info["name"]
            n = sample_counts[name]
            self.A_covs[name] /= n
            self.G_covs[name] /= n
            print(f"  {name}: {n} samples accumulated")

        self._remove_hooks()

    def eigendecompose(self):
        """Full eigendecomposition with proper symmetrization and PSD clamping."""
        report_progress("kfac_eigen", "Eigendecomposing K-FAC factors (v2)", 0.35)
        spectral_info = {}

        for info in self.target_params:
            name = info["name"]

            # A eigendecomp
            A = self.A_covs[name]
            A = (A + A.T) / 2  # Symmetrize
            eigvals_A, eigvecs_A = torch.linalg.eigh(A.float())
            # Clamp to non-negative (numerical noise fix)
            n_neg_A = int((eigvals_A < 0).sum())
            eigvals_A = eigvals_A.clamp(min=0)

            # G eigendecomp
            G = self.G_covs[name]
            G = (G + G.T) / 2
            eigvals_G, eigvecs_G = torch.linalg.eigh(G.float())
            n_neg_G = int((eigvals_G < 0).sum())
            eigvals_G = eigvals_G.clamp(min=0)

            self.A_eig[name] = (eigvals_A, eigvecs_A)
            self.G_eig[name] = (eigvals_G, eigvecs_G)

            # Compute effective rank
            def eff_rank(vals, threshold=0.95):
                s = vals.sum()
                if s < 1e-12:
                    return len(vals)
                cumfrac = vals.cumsum(0) / s
                return int((cumfrac < threshold).sum()) + 1

            cond_A = float(eigvals_A[-1] / max(eigvals_A[eigvals_A > 0].min().item(), 1e-15)) if (eigvals_A > 0).any() else float('inf')
            cond_G = float(eigvals_G[-1] / max(eigvals_G[eigvals_G > 0].min().item(), 1e-15)) if (eigvals_G > 0).any() else float('inf')

            spectral_info[name] = {
                "A_shape": list(A.shape),
                "G_shape": list(G.shape),
                "A_eigval_range": [float(eigvals_A.min()), float(eigvals_A.max())],
                "G_eigval_range": [float(eigvals_G.min()), float(eigvals_G.max())],
                "A_neg_count": n_neg_A,
                "G_neg_count": n_neg_G,
                "A_condition": cond_A,
                "G_condition": cond_G,
                "A_rank_95": eff_rank(eigvals_A),
                "G_rank_95": eff_rank(eigvals_G),
                "A_trace": float(eigvals_A.sum()),
                "G_trace": float(eigvals_G.sum()),
            }

            print(f"  {name}:")
            print(f"    A: shape={A.shape}, range=[{eigvals_A.min():.2e}, {eigvals_A.max():.2e}], "
                  f"neg_clamped={n_neg_A}, cond={cond_A:.2e}, rank95={eff_rank(eigvals_A)}")
            print(f"    G: shape={G.shape}, range=[{eigvals_G.min():.2e}, {eigvals_G.max():.2e}], "
                  f"neg_clamped={n_neg_G}, cond={cond_G:.2e}, rank95={eff_rank(eigvals_G)}")

        return spectral_info

    def compute_per_sample_grads(self, tok, texts, max_len=512):
        """Compute per-sample gradients for target parameters."""
        self._register_hooks()
        all_grads = []

        for idx in range(len(texts)):
            inp = tok(texts[idx], return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
            self.model.zero_grad()
            out = self.model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                labels=inp["input_ids"],
            )
            out.loss.backward()

            grad_dict = {}
            for info in self.target_params:
                name = info["name"]
                g = info["weight"].grad.detach().clone().float()  # [out, in]
                grad_dict[name] = g

            all_grads.append(grad_dict)
            self.model.zero_grad(set_to_none=True)
            self._activations.clear()
            self._grad_outputs.clear()

            if (idx + 1) % 20 == 0:
                print(f"    Grads: {idx + 1}/{len(texts)}")
                torch.cuda.empty_cache()

        self._remove_hooks()
        return all_grads

    def compute_kfac_if_scores(self, train_grads, ref_grads, damping=None):
        """
        K-FAC IF: s(test, train) = sum_l trace(G_ref_l^T G_inv_l G_train_l A_inv_l)
        """
        if damping is None:
            damping = self.damping

        n_train = len(train_grads)
        n_ref = len(ref_grads)
        scores = np.zeros((n_train, n_ref))

        for info in self.target_params:
            name = info["name"]
            eigvals_A, eigvecs_A = self.A_eig[name]
            eigvals_G, eigvecs_G = self.G_eig[name]

            inv_A = 1.0 / (eigvals_A + damping)
            inv_G = 1.0 / (eigvals_G + damping)

            # Precompute transformed train grads: U_G^T W U_A * (inv_G outer inv_A)
            train_transformed = []
            for tg in train_grads:
                W = tg[name].to(self.device)  # [out, in]
                W_t = eigvecs_G.T @ W @ eigvecs_A  # [out, in] in eigen basis
                W_inv = W_t * inv_G[:, None] * inv_A[None, :]
                train_transformed.append(W_inv.cpu())

            # Compute scores for each ref
            for j, rg in enumerate(ref_grads):
                W_ref = rg[name].to(self.device)
                W_ref_t = (eigvecs_G.T @ W_ref @ eigvecs_A).cpu()

                for i, W_train_inv in enumerate(train_transformed):
                    scores[i, j] += (W_ref_t * W_train_inv).sum().item()

        return scores

    def compute_diagonal_if_scores(self, train_grads, ref_grads, damping=None):
        """Diagonal IF: s = g_test^T diag(H)^{-1} g_train"""
        if damping is None:
            damping = self.damping

        fisher_diag = {}
        for info in self.target_params:
            name = info["name"]
            sq_sum = torch.zeros(info["weight"].shape, dtype=torch.float32)
            for tg in train_grads:
                sq_sum += tg[name].cpu() ** 2
            fisher_diag[name] = sq_sum / len(train_grads) + damping

        n_train = len(train_grads)
        n_ref = len(ref_grads)
        scores = np.zeros((n_train, n_ref))

        for info in self.target_params:
            name = info["name"]
            inv_diag = 1.0 / fisher_diag[name]
            for j, rg in enumerate(ref_grads):
                g_ref = rg[name].cpu()
                g_ref_scaled = g_ref * inv_diag
                for i, tg in enumerate(train_grads):
                    g_train = tg[name].cpu()
                    scores[i, j] += (g_ref_scaled * g_train).sum().item()

        return scores

    def compute_raw_dot_scores(self, train_grads, ref_grads):
        """Raw dot-product IF (no Hessian): s = g_test^T g_train"""
        n_train = len(train_grads)
        n_ref = len(ref_grads)
        scores = np.zeros((n_train, n_ref))

        for info in self.target_params:
            name = info["name"]
            for j, rg in enumerate(ref_grads):
                g_ref = rg[name].flatten().cpu()
                for i, tg in enumerate(train_grads):
                    g_train = tg[name].flatten().cpu()
                    scores[i, j] += (g_ref * g_train).sum().item()

        return scores


# ── TRAK ────────────────────────────────────────────────────────────────
def run_trak(model, tok, task_name, train_data, ref_data, n_train, k=TRAK_K):
    """TRAK with CountSketch projection on Pythia-70M."""
    report_progress("trak", f"TRAK on {task_name}", 0.70)
    t0 = time.time()

    train_texts = [fmt(train_data[i]) for i in range(n_train)]
    ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]

    target_params = []
    target_names = []
    for li in KFAC_TARGET_LAYERS:
        for suffix in ["attention.dense.weight", "mlp.dense_4h_to_h.weight"]:
            name = f"gpt_neox.layers.{li}.{suffix}"
            for n, p in model.named_parameters():
                if n == name:
                    p.requires_grad_(True)
                    target_params.append(p)
                    target_names.append(n)
                    break

    for n, p in model.named_parameters():
        if n not in target_names:
            p.requires_grad_(False)

    D = sum(p.numel() for p in target_params)
    print(f"[TRAK/{task_name}] D={D:,} from {target_names}")

    rng_cs = np.random.RandomState(SEED + 7777)
    cs_buckets = torch.from_numpy(rng_cs.randint(0, k, size=D).astype(np.int64))
    cs_signs = torch.from_numpy(rng_cs.choice([-1.0, 1.0], size=D).astype(np.float32))

    def compute_projected_grads(texts, desc=""):
        all_proj = []
        for idx, text in enumerate(texts):
            inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            model.zero_grad()
            out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                        labels=inp["input_ids"])
            out.loss.backward()
            grad_flat = torch.cat([p.grad.detach().flatten().float().cpu() for p in target_params])
            proj = torch.zeros(k, dtype=torch.float32)
            proj.index_add_(0, cs_buckets, grad_flat * cs_signs)
            all_proj.append(proj)
            model.zero_grad(set_to_none=True)
            if (idx + 1) % 20 == 0:
                print(f"  [{desc}] {idx + 1}/{len(texts)}")
                torch.cuda.empty_cache()
        return torch.stack(all_proj)

    train_proj = compute_projected_grads(train_texts, f"TRAK-train-{task_name}")
    ref_proj = compute_projected_grads(ref_texts, f"TRAK-ref-{task_name}")

    train_proj = F.normalize(train_proj, dim=-1)
    ref_proj = F.normalize(ref_proj, dim=-1)
    sim = train_proj @ ref_proj.T

    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.time() - t0

    if task_name == "counterfact":
        return [sim[:, j].numpy() for j in range(sim.shape[1])], elapsed
    return sim.mean(dim=1).numpy(), elapsed


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate(task_name, pilot_subset, scores, method):
    pdata = pilot_subset["data"]
    n = len(pdata)
    if task_name == "toxicity":
        unsafe = [i for i in range(n) if pdata[i]["type"] == "Unsafe"]
        # Score diagnostics
        diag = score_diagnostics(scores, f"{method}/{task_name}", unsafe)
        auprc = compute_auprc(scores, unsafe, n)
        auprc_rank = compute_auprc_rankbased(scores, unsafe, n)
        print(f"  [{method}/{task_name}] AUPRC={auprc:.4f} (rank-based={auprc_rank:.4f}) "
              f"(unsafe={len(unsafe)}/{n})")
        return {
            "AUPRC": round(auprc, 6),
            "AUPRC_rank": round(auprc_rank, 6),
            "n_unsafe": len(unsafe),
            "score_diagnostics": diag,
        }
    else:
        fi = []
        for r in pilot_subset["ref"]:
            indices = [
                i for i in range(n)
                if pdata[i].get("counterfactual_entity") == r.get("counterfactual_entity")
                and pdata[i].get("true_entity") == r.get("true_entity")
            ]
            fi.append(indices)
        recall, mrr = compute_factual_metrics(scores, fi, k=50)
        n_with_facts = sum(1 for f in fi if f)
        print(f"  [{method}/{task_name}] Recall@50={recall:.4f}, MRR={mrr:.4f} (refs={n_with_facts})")
        return {"Recall@50": round(recall, 6), "MRR": round(mrr, 6), "refs_with_facts": n_with_facts}


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    for d in [PHASE0_DIR, PILOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("Phase 0b: K-FAC Hessian Control (H6 -- CRITICAL) [v2]")
    print(f"Model: {MODEL_NAME} (d=512, 6 layers)")
    print(f"K-FAC target layers: {KFAC_TARGET_LAYERS}")
    print(f"Damping values to test: {KFAC_DAMPING_VALUES}")
    print(f"Pilot: N={PILOT_N_TRAIN}, seed={SEED}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    tasks = load_tasks()
    pilot_subsets = create_pilot_subsets(tasks)
    model, tok = load_model()

    all_results = {}
    kfac_spectral = {}
    damping_sweep_results = {}

    for task_name in ["toxicity", "counterfact"]:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        ps = pilot_subsets[task_name]
        n = len(ps["data"])
        train_texts = [fmt(ps["data"][i]) for i in range(n)]
        ref_texts = [fmt(ps["ref"][i]) for i in range(len(ps["ref"]))]

        task_results = {}

        # 1. RepSim
        print(f"\n--- RepSim ---")
        repsim_scores, repsim_time = run_repsim(model, tok, task_name, ps["data"], ps["ref"], n)
        repsim_metrics = evaluate(task_name, ps, repsim_scores, "RepSim")
        task_results["repsim"] = {"metrics": repsim_metrics, "runtime_sec": round(repsim_time, 2)}

        # 2. K-FAC IF (corrected)
        print(f"\n--- K-FAC IF (v2 corrected) ---")
        report_progress("kfac_if", f"K-FAC IF on {task_name}", 0.25 if task_name == "toxicity" else 0.55)
        t_kfac = time.time()

        kfac = KFACInfluenceV2(model, KFAC_TARGET_LAYERS, damping=KFAC_DAMPING_DEFAULT, device=DEVICE)
        kfac.compute_kronecker_factors(tok, train_texts, bs=4)
        spectral_info = kfac.eigendecompose()
        kfac_spectral[task_name] = spectral_info

        # Per-sample gradients
        print(f"  Computing train gradients ({n} samples)...")
        report_progress("kfac_train_grads", f"K-FAC train grads on {task_name}",
                        0.30 if task_name == "toxicity" else 0.60)
        train_grads = kfac.compute_per_sample_grads(tok, train_texts)

        print(f"  Computing ref gradients ({len(ref_texts)} samples)...")
        ref_grads = kfac.compute_per_sample_grads(tok, ref_texts)

        # K-FAC IF scores with default damping
        print(f"  Computing K-FAC IF scores (damping={KFAC_DAMPING_DEFAULT})...")
        kfac_scores_matrix = kfac.compute_kfac_if_scores(train_grads, ref_grads)

        if task_name == "counterfact":
            kfac_scores = [kfac_scores_matrix[:, j] for j in range(kfac_scores_matrix.shape[1])]
        else:
            kfac_scores = kfac_scores_matrix.mean(axis=1)

        kfac_time = time.time() - t_kfac
        kfac_metrics = evaluate(task_name, ps, kfac_scores, "K-FAC IF")
        task_results["kfac_if"] = {"metrics": kfac_metrics, "runtime_sec": round(kfac_time, 2)}

        # 2b. Damping sweep for K-FAC
        print(f"\n--- K-FAC Damping Sweep ---")
        damping_sweep_results[task_name] = {}
        for damp in KFAC_DAMPING_VALUES:
            if damp == KFAC_DAMPING_DEFAULT:
                # Already computed
                damping_sweep_results[task_name][str(damp)] = kfac_metrics
                continue
            kfac_matrix_d = kfac.compute_kfac_if_scores(train_grads, ref_grads, damping=damp)
            if task_name == "counterfact":
                kfac_scores_d = [kfac_matrix_d[:, j] for j in range(kfac_matrix_d.shape[1])]
            else:
                kfac_scores_d = kfac_matrix_d.mean(axis=1)
            damp_metrics = evaluate(task_name, ps, kfac_scores_d, f"K-FAC(d={damp})")
            damping_sweep_results[task_name][str(damp)] = damp_metrics

        # 3. Raw dot-product IF (no Hessian at all)
        print(f"\n--- Raw Dot-Product IF ---")
        report_progress("raw_dot_if", f"Raw Dot IF on {task_name}",
                        0.40 if task_name == "toxicity" else 0.70)
        t_raw = time.time()
        raw_scores_matrix = kfac.compute_raw_dot_scores(train_grads, ref_grads)

        if task_name == "counterfact":
            raw_scores = [raw_scores_matrix[:, j] for j in range(raw_scores_matrix.shape[1])]
        else:
            raw_scores = raw_scores_matrix.mean(axis=1)

        raw_time = time.time() - t_raw
        raw_metrics = evaluate(task_name, ps, raw_scores, "Raw-Dot")
        task_results["raw_dot_if"] = {"metrics": raw_metrics, "runtime_sec": round(raw_time, 2)}

        # 4. Diagonal IF
        print(f"\n--- Diagonal IF ---")
        report_progress("diag_if", f"Diagonal IF on {task_name}",
                        0.45 if task_name == "toxicity" else 0.73)
        t_diag = time.time()
        diag_scores_matrix = kfac.compute_diagonal_if_scores(train_grads, ref_grads)

        if task_name == "counterfact":
            diag_scores = [diag_scores_matrix[:, j] for j in range(diag_scores_matrix.shape[1])]
        else:
            diag_scores = diag_scores_matrix.mean(axis=1)

        diag_time = time.time() - t_diag
        diag_metrics = evaluate(task_name, ps, diag_scores, "Diag-IF")
        task_results["diag_if"] = {"metrics": diag_metrics, "runtime_sec": round(diag_time, 2)}

        # 5. TRAK
        print(f"\n--- TRAK ---")
        trak_scores, trak_time = run_trak(model, tok, task_name, ps["data"], ps["ref"], n)
        trak_metrics = evaluate(task_name, ps, trak_scores, "TRAK")
        task_results["trak"] = {"metrics": trak_metrics, "runtime_sec": round(trak_time, 2)}

        gc.collect()
        torch.cuda.empty_cache()
        del train_grads, ref_grads, kfac
        gc.collect()
        torch.cuda.empty_cache()

        all_results[task_name] = task_results

    # ── Analysis: H6 Decision Gate ──────────────────────────────────────
    report_progress("analysis", "H6 decision gate analysis", 0.90)
    print(f"\n{'='*70}")
    print("H6 DECISION GATE ANALYSIS (v2)")
    print(f"{'='*70}")

    h6_analysis = {}
    for task_name, task_results in all_results.items():
        if task_name == "toxicity":
            repsim_val = task_results["repsim"]["metrics"]["AUPRC"]
            kfac_val = task_results["kfac_if"]["metrics"]["AUPRC"]
            diag_val = task_results["diag_if"]["metrics"]["AUPRC"]
            trak_val = task_results["trak"]["metrics"]["AUPRC"]
            raw_val = task_results["raw_dot_if"]["metrics"]["AUPRC"]
            # Also check rank-based AUPRC
            repsim_rank = task_results["repsim"]["metrics"]["AUPRC_rank"]
            kfac_rank = task_results["kfac_if"]["metrics"]["AUPRC_rank"]
            diag_rank = task_results["diag_if"]["metrics"]["AUPRC_rank"]
            trak_rank = task_results["trak"]["metrics"]["AUPRC_rank"]
            raw_rank = task_results["raw_dot_if"]["metrics"]["AUPRC_rank"]
            metric_name = "AUPRC"
        else:
            rm = task_results["repsim"]["metrics"]
            km = task_results["kfac_if"]["metrics"]
            dm = task_results["diag_if"]["metrics"]
            tm = task_results["trak"]["metrics"]
            rwm = task_results["raw_dot_if"]["metrics"]
            repsim_val = rm.get("Recall@50", 0) + rm.get("MRR", 0)
            kfac_val = km.get("Recall@50", 0) + km.get("MRR", 0)
            diag_val = dm.get("Recall@50", 0) + dm.get("MRR", 0)
            trak_val = tm.get("Recall@50", 0) + tm.get("MRR", 0)
            raw_val = rwm.get("Recall@50", 0) + rwm.get("MRR", 0)
            repsim_rank = kfac_rank = diag_rank = trak_rank = raw_rank = None
            metric_name = "Recall@50+MRR"

        gap_kfac = repsim_val - kfac_val
        gap_diag = repsim_val - diag_val
        gap_trak = repsim_val - trak_val
        gap_raw = repsim_val - raw_val
        kfac_improvement = kfac_val - diag_val

        h6_pass = gap_kfac > 0.05
        h6_falsified = gap_kfac <= 0.05

        h6_entry = {
            "metric": metric_name,
            "repsim": round(repsim_val, 6),
            "kfac_if": round(kfac_val, 6),
            "diag_if": round(diag_val, 6),
            "trak": round(trak_val, 6),
            "raw_dot_if": round(raw_val, 6),
            "gap_repsim_kfac_pp": round(gap_kfac * 100, 2),
            "gap_repsim_diag_pp": round(gap_diag * 100, 2),
            "gap_repsim_trak_pp": round(gap_trak * 100, 2),
            "gap_repsim_raw_pp": round(gap_raw * 100, 2),
            "kfac_over_diag_pp": round(kfac_improvement * 100, 2),
            "h6_pass": h6_pass,
            "h6_falsified": h6_falsified,
        }

        if repsim_rank is not None:
            h6_entry["rank_based_auprc"] = {
                "repsim": round(repsim_rank, 6),
                "kfac_if": round(kfac_rank, 6),
                "diag_if": round(diag_rank, 6),
                "trak": round(trak_rank, 6),
                "raw_dot_if": round(raw_rank, 6),
            }

        h6_analysis[task_name] = h6_entry

        print(f"\n{task_name} ({metric_name}):")
        print(f"  RepSim:     {repsim_val:.4f}")
        print(f"  K-FAC IF:   {kfac_val:.4f}  (gap from RepSim: {gap_kfac*100:+.1f}pp)")
        print(f"  Diag IF:    {diag_val:.4f}  (gap from RepSim: {gap_diag*100:+.1f}pp)")
        print(f"  Raw Dot IF: {raw_val:.4f}  (gap from RepSim: {gap_raw*100:+.1f}pp)")
        print(f"  TRAK:       {trak_val:.4f}  (gap from RepSim: {gap_trak*100:+.1f}pp)")
        print(f"  K-FAC improvement over Diag: {kfac_improvement*100:+.1f}pp")
        if repsim_rank is not None:
            print(f"  Rank-based AUPRC: RepSim={repsim_rank:.4f}, K-FAC={kfac_rank:.4f}, "
                  f"Diag={diag_rank:.4f}, Raw={raw_rank:.4f}, TRAK={trak_rank:.4f}")
        print(f"  H6: {'PASS (gap>5pp -- K-FAC still fails)' if h6_pass else 'FALSIFIED (gap<5pp -- K-FAC closes gap!)'}")

    any_falsified = any(v["h6_falsified"] for v in h6_analysis.values())
    all_pass = all(v["h6_pass"] for v in h6_analysis.values())

    # For toxicity: if rank-based AUPRC tells a different story, note it
    tox_analysis = h6_analysis.get("toxicity", {})
    rank_auprc = tox_analysis.get("rank_based_auprc", {})
    if rank_auprc:
        rank_gap = rank_auprc.get("repsim", 0) - rank_auprc.get("kfac_if", 0)
        if abs(rank_gap * 100 - tox_analysis.get("gap_repsim_kfac_pp", 0)) > 10:
            print(f"\n  *** WARNING: Rank-based AUPRC gap ({rank_gap*100:+.1f}pp) differs significantly from "
                  f"raw AUPRC gap ({tox_analysis['gap_repsim_kfac_pp']:+.1f}pp) on toxicity.")
            print(f"      This suggests score scale artifacts. Using rank-based AUPRC for H6 decision.")
            # Override decision based on rank-based AUPRC
            rank_h6_pass = rank_gap > 0.05
            if rank_h6_pass != tox_analysis["h6_pass"]:
                print(f"      Rank-based H6: {'PASS' if rank_h6_pass else 'FALSIFIED'}")
                h6_analysis["toxicity"]["h6_pass_rank_based"] = rank_h6_pass
                h6_analysis["toxicity"]["h6_falsified_rank_based"] = not rank_h6_pass

    # Use both raw and rank-based for final decision
    # If rank-based tells a different story, prefer rank-based (immune to scale artifacts)
    tasks_pass_count = 0
    tasks_falsified_count = 0
    for tn, entry in h6_analysis.items():
        # Prefer rank-based if available
        if "h6_pass_rank_based" in entry:
            if entry["h6_pass_rank_based"]:
                tasks_pass_count += 1
            else:
                tasks_falsified_count += 1
        else:
            if entry["h6_pass"]:
                tasks_pass_count += 1
            else:
                tasks_falsified_count += 1

    if tasks_falsified_count == 0:
        decision = "PROCEED_CRA"
    elif tasks_pass_count == 0:
        decision = "PIVOT_CAND_B"
    else:
        decision = "PARTIAL_PASS"

    print(f"\n{'='*70}")
    print(f"H6 OVERALL DECISION: {decision}")
    print(f"  Tasks passed: {tasks_pass_count}, Tasks falsified: {tasks_falsified_count}")
    if decision == "PROCEED_CRA":
        print("  K-FAC IF still underperforms RepSim by >5pp on all tasks.")
        print("  FM1/FM2 are independent of Hessian quality. CRA thesis holds.")
    elif decision == "PIVOT_CAND_B":
        print("  K-FAC IF closes the gap with RepSim on all tasks!")
        print("  CRA thesis requires fundamental revision.")
    else:
        print("  Mixed results -- some tasks K-FAC closes the gap, others it doesn't.")
        print("  CRA thesis may hold with task-specific caveats.")
    print(f"{'='*70}")

    total_time = time.time() - t_start

    # Save results
    final = {
        "task_id": TASK_ID,
        "version": "v2",
        "model": MODEL_NAME,
        "hidden_dim": 512,
        "num_layers": 6,
        "kfac_target_layers": KFAC_TARGET_LAYERS,
        "kfac_damping_default": KFAC_DAMPING_DEFAULT,
        "pilot_n_train": PILOT_N_TRAIN,
        "trak_k": TRAK_K,
        "seed": SEED,
        "methods_per_task": all_results,
        "h6_analysis": h6_analysis,
        "h6_decision": decision,
        "damping_sweep": damping_sweep_results,
        "kfac_spectral": kfac_spectral,
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat(),
    }

    # Remove score_diagnostics from serialization (contains numpy)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    final = sanitize(final)

    for p in [
        os.path.join(PHASE0_DIR, "kfac_control.json"),
        os.path.join(PILOTS_DIR, f"{TASK_ID}_results.json"),
    ]:
        with open(p, "w") as f:
            json.dump(final, f, indent=2)
    print(f"\nResults saved to {PHASE0_DIR}/kfac_control.json")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Task':<15}{'Method':<14}{'Score':<10}{'Gap(pp)':<10}{'Time(s)':<8}")
    print("-" * 57)
    for tn in all_results:
        for method in ["repsim", "kfac_if", "raw_dot_if", "diag_if", "trak"]:
            r = all_results[tn][method]
            if tn == "toxicity":
                score = r["metrics"]["AUPRC"]
                repsim_score = all_results[tn]["repsim"]["metrics"]["AUPRC"]
            else:
                score = r["metrics"].get("Recall@50", 0) + r["metrics"].get("MRR", 0)
                rm = all_results[tn]["repsim"]["metrics"]
                repsim_score = rm.get("Recall@50", 0) + rm.get("MRR", 0)
            gap = (score - repsim_score) * 100
            print(f"{tn:<15}{method:<14}{score:<10.4f}{gap:+<10.1f}{r['runtime_sec']:<8.1f}")
    print(f"-" * 57)
    print(f"Total runtime: {total_time:.1f}s  Decision: {decision}")
    print(f"{'='*70}")

    mark_done(
        "success",
        f"H6 decision={decision} (v2). Total={total_time:.0f}s. "
        + "; ".join(f"{tn}: gap={v['gap_repsim_kfac_pp']}pp" for tn, v in h6_analysis.items()),
    )
    return final


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"FATAL: {e}\n{traceback.format_exc()}")
        mark_done("failed", str(e)[:500])
        sys.exit(1)
