#!/usr/bin/env python3
"""
Phase 0b: K-FAC Hessian Control on Pythia-70M (H6 -- CRITICAL)

CRITICAL DECISION GATE:
  If K-FAC IF LDS >= RepSim LDS - 5pp → CRA thesis requires fundamental revision.
  If K-FAC IF LDS < RepSim LDS - 5pp → H6 not falsified, proceed with CRA.

Approach:
  1. Load Pythia-70M (d=512, 6 layers, ~70M params)
  2. Use DATE-LM Toxicity (AUPRC) and Counterfact (Recall@50, MRR) tasks
     (Pythia-70M generates its own representations; data is model-agnostic text)
  3. Compute RepSim (last-layer cosine similarity) as the representation-space baseline
  4. Compute K-FAC IF:
     - Target: last 2 transformer layers (layers 4-5) weight matrices
     - K-FAC Kronecker factorization: H_W ≈ A ⊗ G
     - Full eigendecomposition of A (input cov) and G (output grad cov)
     - IF score: g_test^T H^{-1} g_train via Kronecker structure
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

# GPU assignment
GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
DEVICE = f"cuda:0"  # After CUDA_VISIBLE_DEVICES remapping

# K-FAC config
KFAC_TARGET_LAYERS = [4, 5]  # Last 2 transformer layers
KFAC_DAMPING = 1e-4  # Tikhonov damping for eigenvalue inversion
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
        "final_progress": final, "results": results,
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
    # Use float32 for K-FAC numerical stability
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, dtype=torch.float32, device_map=DEVICE
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


# ── K-FAC Influence Functions ───────────────────────────────────────────
class KFACInfluence:
    """
    K-FAC (Kronecker-Factored Approximate Curvature) influence functions.

    For weight matrix W of shape (out_dim, in_dim) in a linear layer:
      H_W ≈ G ⊗ A
    where:
      A = E[a a^T]  (input activation covariance, in_dim x in_dim)
      G = E[g g^T]  (output gradient covariance, out_dim x out_dim)

    IF score: g_test^T H^{-1} g_train
    Using Kronecker: H^{-1} = G^{-1} ⊗ A^{-1}
    So: vec(W_test)^T (G^{-1} ⊗ A^{-1}) vec(W_train)
      = trace(W_test^T G^{-1} W_train A^{-1})

    For full eigendecomp:
      A = U_A diag(λ_A) U_A^T
      G = U_G diag(λ_G) U_G^T
      A^{-1} = U_A diag(1/(λ_A + damping)) U_A^T
      G^{-1} = U_G diag(1/(λ_G + damping)) U_G^T
    """

    def __init__(self, model, target_layer_indices, damping=1e-4, device="cuda:0"):
        self.model = model
        self.target_layers = target_layer_indices
        self.damping = damping
        self.device = device

        # Identify target weight matrices
        self.target_params = []
        self._hooks = []
        self._activations = {}
        self._grad_outputs = {}

        for li in self.target_layers:
            prefix = f"gpt_neox.layers.{li}"
            # Target: attention.dense (out projection) and mlp.dense_4h_to_h (down projection)
            for suffix in ["attention.dense", "mlp.dense_4h_to_h"]:
                full_name = f"{prefix}.{suffix}"
                module = dict(model.named_modules())[full_name]
                weight = module.weight  # (out, in)
                self.target_params.append({
                    "name": full_name,
                    "module": module,
                    "weight": weight,
                    "out_dim": weight.shape[0],
                    "in_dim": weight.shape[1],
                })
                print(f"  K-FAC target: {full_name} shape={weight.shape}")

        # Storage for Kronecker factors
        self.A_covs = {}   # input activation covariances
        self.G_covs = {}   # output gradient covariances
        self.A_eig = {}    # eigendecompositions
        self.G_eig = {}    # eigendecompositions

    def _register_hooks(self):
        """Register forward/backward hooks to capture activations and gradients."""
        self._hooks = []
        for info in self.target_params:
            name = info["name"]
            module = info["module"]

            def make_fwd_hook(n):
                def hook(mod, inp, out):
                    # inp[0] is the input activation
                    self._activations[n] = inp[0].detach()
                return hook

            def make_bwd_hook(n):
                def hook(mod, grad_in, grad_out):
                    # grad_out[0] is the gradient w.r.t. output
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
        Compute A and G Kronecker factors from data.
        A_l = (1/N) Σ_n a_n a_n^T  (input activations, averaged over tokens)
        G_l = (1/N) Σ_n g_n g_n^T  (output gradients, averaged over tokens)
        """
        report_progress("kfac_factors", "Computing K-FAC Kronecker factors", 0.20)
        self._register_hooks()

        # Initialize accumulators
        for info in self.target_params:
            name = info["name"]
            in_dim = info["in_dim"]
            out_dim = info["out_dim"]
            self.A_covs[name] = torch.zeros(in_dim, in_dim, device=self.device)
            self.G_covs[name] = torch.zeros(out_dim, out_dim, device=self.device)

        n_samples = 0
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

            mask = inp["attention_mask"]  # [B, T]

            for info in self.target_params:
                name = info["name"]
                act = self._activations[name]   # [B, T, in_dim]
                grad = self._grad_outputs[name]  # [B, T, out_dim]

                # Average over valid tokens per sample, then accumulate
                for b in range(act.shape[0]):
                    valid = mask[b].bool()
                    a_valid = act[b, valid]  # [T_valid, in_dim]
                    g_valid = grad[b, valid]  # [T_valid, out_dim]

                    # Mean over tokens for this sample
                    a_mean = a_valid.mean(0)  # [in_dim]
                    g_mean = g_valid.mean(0)  # [out_dim]

                    self.A_covs[name] += torch.outer(a_mean, a_mean)
                    self.G_covs[name] += torch.outer(g_mean, g_mean)
                    n_samples += 1

            self.model.zero_grad(set_to_none=True)
            self._activations.clear()
            self._grad_outputs.clear()

            if (i + bs) % 20 == 0 or i + bs >= len(texts):
                print(f"  K-FAC factors: {min(i + bs, len(texts))}/{len(texts)}")

        # Normalize
        n_per_layer = n_samples // len(self.target_params)
        for info in self.target_params:
            name = info["name"]
            self.A_covs[name] /= n_per_layer
            self.G_covs[name] /= n_per_layer

        self._remove_hooks()
        print(f"  K-FAC factors computed from {n_per_layer} samples")

    def eigendecompose(self):
        """Full eigendecomposition of A and G for each target layer."""
        report_progress("kfac_eigen", "Eigendecomposing K-FAC factors", 0.35)
        for info in self.target_params:
            name = info["name"]

            # A eigendecomp
            A = self.A_covs[name]
            A = (A + A.T) / 2  # Symmetrize
            eigvals_A, eigvecs_A = torch.linalg.eigh(A)
            self.A_eig[name] = (eigvals_A, eigvecs_A)

            # G eigendecomp
            G = self.G_covs[name]
            G = (G + G.T) / 2
            eigvals_G, eigvecs_G = torch.linalg.eigh(G)
            self.G_eig[name] = (eigvals_G, eigvecs_G)

            print(f"  {name}:")
            print(f"    A: shape={A.shape}, eigval range=[{eigvals_A.min():.2e}, {eigvals_A.max():.2e}], "
                  f"condition={eigvals_A.max()/max(eigvals_A.min(), 1e-10):.2e}")
            print(f"    G: shape={G.shape}, eigval range=[{eigvals_G.min():.2e}, {eigvals_G.max():.2e}], "
                  f"condition={eigvals_G.max()/max(eigvals_G.min(), 1e-10):.2e}")

    def compute_per_sample_grads(self, tok, texts, bs=1, max_len=512):
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

            # Collect gradients for target params
            grad_dict = {}
            for info in self.target_params:
                name = info["name"]
                g = info["weight"].grad.detach().clone()  # [out, in]
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

    def compute_kfac_if_scores(self, train_grads, ref_grads):
        """
        Compute K-FAC IF scores: s(test, train) = Σ_l trace(G_test_l^T G_inv_l G_train_l A_inv_l)

        Using eigendecomp:
          A_inv = U_A diag(1/(λ_A + d)) U_A^T
          G_inv = U_G diag(1/(λ_G + d)) U_G^T
        """
        n_train = len(train_grads)
        n_ref = len(ref_grads)
        scores = np.zeros((n_train, n_ref))

        for info in self.target_params:
            name = info["name"]
            eigvals_A, eigvecs_A = self.A_eig[name]
            eigvals_G, eigvecs_G = self.G_eig[name]

            # Compute inverse eigenvalues with damping
            inv_A = 1.0 / (eigvals_A + self.damping)  # [in_dim]
            inv_G = 1.0 / (eigvals_G + self.damping)  # [out_dim]

            # Transform all gradients to eigen basis
            # For grad W (out, in):
            #   W_tilde = U_G^T W U_A  (in eigen basis)
            #   then IF = sum of W_tilde_test * W_tilde_train * inv_G_outer_inv_A

            # Precompute transformed train grads
            train_transformed = []
            for tg in train_grads:
                W = tg[name].to(self.device)  # [out, in]
                W_t = eigvecs_G.T @ W @ eigvecs_A  # [out, in] in eigen basis
                # Apply inverse: element-wise multiply by inv_G[:, None] * inv_A[None, :]
                W_inv = W_t * inv_G[:, None] * inv_A[None, :]
                train_transformed.append(W_inv)

            # Compute scores for each ref
            for j, rg in enumerate(ref_grads):
                W_ref = rg[name].to(self.device)
                W_ref_t = eigvecs_G.T @ W_ref @ eigvecs_A

                for i, W_train_inv in enumerate(train_transformed):
                    # trace(W_ref^T H^{-1} W_train) = sum(W_ref_tilde * W_train_inv)
                    scores[i, j] += (W_ref_t * W_train_inv).sum().item()

        return scores

    def compute_diagonal_if_scores(self, train_grads, ref_grads):
        """
        Vanilla diagonal IF: s = g_test^T diag(H)^{-1} g_train
        where diag(H) ≈ E[g^2] (empirical Fisher diagonal).
        """
        # Compute Fisher diagonal from train grads
        fisher_diag = {}
        for info in self.target_params:
            name = info["name"]
            sq_sum = torch.zeros_like(info["weight"])
            for tg in train_grads:
                sq_sum += tg[name].to(self.device) ** 2
            fisher_diag[name] = sq_sum / len(train_grads) + self.damping

        n_train = len(train_grads)
        n_ref = len(ref_grads)
        scores = np.zeros((n_train, n_ref))

        for info in self.target_params:
            name = info["name"]
            inv_diag = 1.0 / fisher_diag[name]
            for j, rg in enumerate(ref_grads):
                g_ref = rg[name].to(self.device)
                g_ref_scaled = g_ref * inv_diag
                for i, tg in enumerate(train_grads):
                    g_train = tg[name].to(self.device)
                    scores[i, j] += (g_ref_scaled * g_train).sum().item()

        return scores


# ── TRAK (parameter-space random projection baseline) ───────────────────
def run_trak(model, tok, task_name, train_data, ref_data, n_train, k=TRAK_K):
    """TRAK with CountSketch projection on Pythia-70M."""
    report_progress("trak", f"TRAK on {task_name}", 0.70)
    t0 = time.time()

    train_texts = [fmt(train_data[i]) for i in range(n_train)]
    ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]

    # Target: last 2 layers dense weights (same as K-FAC targets for fair comparison)
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

    # CountSketch projection
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

    # Restore requires_grad
    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.time() - t0
    print(f"[TRAK/{task_name}] Done in {elapsed:.1f}s")

    if task_name == "counterfact":
        return [sim[:, j].numpy() for j in range(sim.shape[1])], elapsed
    return sim.mean(dim=1).numpy(), elapsed


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate(task_name, pilot_subset, scores, method):
    pdata = pilot_subset["data"]
    n = len(pdata)
    if task_name == "toxicity":
        unsafe = [i for i in range(n) if pdata[i]["type"] == "Unsafe"]
        auprc = compute_auprc(scores, unsafe, n)
        print(f"  [{method}/{task_name}] AUPRC={auprc:.4f} (unsafe={len(unsafe)}/{n})")
        return {"AUPRC": round(auprc, 6), "n_unsafe": len(unsafe)}
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
    print("Phase 0b: K-FAC Hessian Control (H6 -- CRITICAL)")
    print(f"Model: {MODEL_NAME} (d=512, 6 layers)")
    print(f"K-FAC target layers: {KFAC_TARGET_LAYERS}")
    print(f"Damping: {KFAC_DAMPING}, TRAK k={TRAK_K}")
    print(f"Pilot: N={PILOT_N_TRAIN}, seed={SEED}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    tasks = load_tasks()
    pilot_subsets = create_pilot_subsets(tasks)
    model, tok = load_model()

    all_results = {}
    kfac_spectral = {}

    for task_name in ["toxicity", "counterfact"]:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        ps = pilot_subsets[task_name]
        n = len(ps["data"])
        train_texts = [fmt(ps["data"][i]) for i in range(n)]
        ref_texts = [fmt(ps["ref"][i]) for i in range(len(ps["ref"]))]

        task_results = {}

        # 1. RepSim (representation-space baseline)
        print(f"\n--- RepSim ---")
        repsim_scores, repsim_time = run_repsim(model, tok, task_name, ps["data"], ps["ref"], n)
        repsim_metrics = evaluate(task_name, ps, repsim_scores, "RepSim")
        task_results["repsim"] = {"metrics": repsim_metrics, "runtime_sec": round(repsim_time, 2)}

        # 2. K-FAC IF (full eigendecomp -- the critical test)
        print(f"\n--- K-FAC IF ---")
        report_progress("kfac_if", f"K-FAC IF on {task_name}", 0.25 if task_name == "toxicity" else 0.55)
        t_kfac = time.time()

        kfac = KFACInfluence(model, KFAC_TARGET_LAYERS, damping=KFAC_DAMPING, device=DEVICE)

        # Compute Kronecker factors from training data
        kfac.compute_kronecker_factors(tok, train_texts, bs=4)

        # Full eigendecomposition
        kfac.eigendecompose()

        # Record spectral info for analysis
        if task_name not in kfac_spectral:
            kfac_spectral[task_name] = {}
            for info in kfac.target_params:
                name = info["name"]
                eigA, _ = kfac.A_eig[name]
                eigG, _ = kfac.G_eig[name]
                kfac_spectral[task_name][name] = {
                    "A_eigvals": eigA.cpu().numpy().tolist(),
                    "G_eigvals": eigG.cpu().numpy().tolist(),
                    "A_condition": float(eigA.max() / max(eigA.min(), 1e-10)),
                    "G_condition": float(eigG.max() / max(eigG.min(), 1e-10)),
                    "A_rank_95": int((eigA.cumsum(0) / eigA.sum() < 0.95).sum()) + 1,
                    "G_rank_95": int((eigG.cumsum(0) / eigG.sum() < 0.95).sum()) + 1,
                }

        # Compute per-sample gradients
        print(f"  Computing train gradients ({n} samples)...")
        report_progress("kfac_train_grads", f"K-FAC train grads on {task_name}", 0.30 if task_name == "toxicity" else 0.60)
        train_grads = kfac.compute_per_sample_grads(tok, train_texts)

        print(f"  Computing ref gradients ({len(ref_texts)} samples)...")
        ref_grads = kfac.compute_per_sample_grads(tok, ref_texts)

        # K-FAC IF scores
        print(f"  Computing K-FAC IF scores...")
        report_progress("kfac_scores", f"K-FAC IF scores on {task_name}", 0.35 if task_name == "toxicity" else 0.65)
        kfac_scores_matrix = kfac.compute_kfac_if_scores(train_grads, ref_grads)

        if task_name == "counterfact":
            kfac_scores = [kfac_scores_matrix[:, j] for j in range(kfac_scores_matrix.shape[1])]
        else:
            kfac_scores = kfac_scores_matrix.mean(axis=1)

        kfac_time = time.time() - t_kfac
        kfac_metrics = evaluate(task_name, ps, kfac_scores, "K-FAC IF")
        task_results["kfac_if"] = {"metrics": kfac_metrics, "runtime_sec": round(kfac_time, 2)}

        # 3. Diagonal IF (vanilla parameter-space baseline)
        print(f"\n--- Diagonal IF ---")
        report_progress("diag_if", f"Diagonal IF on {task_name}", 0.40 if task_name == "toxicity" else 0.70)
        t_diag = time.time()
        diag_scores_matrix = kfac.compute_diagonal_if_scores(train_grads, ref_grads)

        if task_name == "counterfact":
            diag_scores = [diag_scores_matrix[:, j] for j in range(diag_scores_matrix.shape[1])]
        else:
            diag_scores = diag_scores_matrix.mean(axis=1)

        diag_time = time.time() - t_diag
        diag_metrics = evaluate(task_name, ps, diag_scores, "Diag-IF")
        task_results["diag_if"] = {"metrics": diag_metrics, "runtime_sec": round(diag_time, 2)}

        # 4. TRAK (random projection parameter-space)
        print(f"\n--- TRAK ---")
        trak_scores, trak_time = run_trak(model, tok, task_name, ps["data"], ps["ref"], n)
        trak_metrics = evaluate(task_name, ps, trak_scores, "TRAK")
        task_results["trak"] = {"metrics": trak_metrics, "runtime_sec": round(trak_time, 2)}

        gc.collect()
        torch.cuda.empty_cache()

        # Clean up per-sample grads to free memory
        del train_grads, ref_grads, kfac
        gc.collect()
        torch.cuda.empty_cache()

        all_results[task_name] = task_results

    # ── Analysis: H6 Decision Gate ──────────────────────────────────────
    report_progress("analysis", "H6 decision gate analysis", 0.90)
    print(f"\n{'='*70}")
    print("H6 DECISION GATE ANALYSIS")
    print(f"{'='*70}")

    h6_analysis = {}
    for task_name, task_results in all_results.items():
        if task_name == "toxicity":
            repsim_val = task_results["repsim"]["metrics"]["AUPRC"]
            kfac_val = task_results["kfac_if"]["metrics"]["AUPRC"]
            diag_val = task_results["diag_if"]["metrics"]["AUPRC"]
            trak_val = task_results["trak"]["metrics"]["AUPRC"]
            metric_name = "AUPRC"
        else:
            # Use Recall@50 + MRR as combined metric
            rm = task_results["repsim"]["metrics"]
            km = task_results["kfac_if"]["metrics"]
            dm = task_results["diag_if"]["metrics"]
            tm = task_results["trak"]["metrics"]
            repsim_val = rm.get("Recall@50", 0) + rm.get("MRR", 0)
            kfac_val = km.get("Recall@50", 0) + km.get("MRR", 0)
            diag_val = dm.get("Recall@50", 0) + dm.get("MRR", 0)
            trak_val = tm.get("Recall@50", 0) + tm.get("MRR", 0)
            metric_name = "Recall@50+MRR"

        gap_kfac = repsim_val - kfac_val
        gap_diag = repsim_val - diag_val
        gap_trak = repsim_val - trak_val
        kfac_improvement = kfac_val - diag_val  # Does K-FAC improve over diagonal?

        # H6 test: K-FAC IF should NOT close the gap with RepSim
        # Falsification: K-FAC IF LDS >= RepSim LDS - 5pp (gap < 5pp)
        h6_pass = gap_kfac > 0.05  # gap > 5pp means K-FAC still fails
        h6_falsified = gap_kfac <= 0.05

        h6_analysis[task_name] = {
            "metric": metric_name,
            "repsim": round(repsim_val, 6),
            "kfac_if": round(kfac_val, 6),
            "diag_if": round(diag_val, 6),
            "trak": round(trak_val, 6),
            "gap_repsim_kfac_pp": round(gap_kfac * 100, 2),
            "gap_repsim_diag_pp": round(gap_diag * 100, 2),
            "gap_repsim_trak_pp": round(gap_trak * 100, 2),
            "kfac_over_diag_pp": round(kfac_improvement * 100, 2),
            "h6_pass": h6_pass,
            "h6_falsified": h6_falsified,
        }

        print(f"\n{task_name} ({metric_name}):")
        print(f"  RepSim:   {repsim_val:.4f}")
        print(f"  K-FAC IF: {kfac_val:.4f}  (gap from RepSim: {gap_kfac*100:+.1f}pp)")
        print(f"  Diag IF:  {diag_val:.4f}  (gap from RepSim: {gap_diag*100:+.1f}pp)")
        print(f"  TRAK:     {trak_val:.4f}  (gap from RepSim: {gap_trak*100:+.1f}pp)")
        print(f"  K-FAC improvement over Diag: {kfac_improvement*100:+.1f}pp")
        print(f"  H6: {'PASS (gap>{0.05*100}pp -- K-FAC still fails)' if h6_pass else 'FALSIFIED (gap<5pp -- K-FAC closes gap!)'}")

    # Overall H6 decision
    any_falsified = any(v["h6_falsified"] for v in h6_analysis.values())
    all_pass = all(v["h6_pass"] for v in h6_analysis.values())

    decision = "PROCEED_CRA" if all_pass else ("PIVOT_CAND_B" if any_falsified else "PARTIAL_PASS")
    print(f"\n{'='*70}")
    print(f"H6 OVERALL DECISION: {decision}")
    if decision == "PROCEED_CRA":
        print("  K-FAC IF still underperforms RepSim by >5pp on all tasks.")
        print("  FM1/FM2 are independent of Hessian quality. CRA thesis holds.")
    elif decision == "PIVOT_CAND_B":
        print("  K-FAC IF closes the gap with RepSim on at least one task!")
        print("  CRA thesis may need revision. Consider pivoting to cand_b.")
    else:
        print("  Mixed results. Some tasks pass, some don't.")
    print(f"{'='*70}")

    total_time = time.time() - t_start

    # ── Save results ────────────────────────────────────────────────────
    final = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": 512,
        "num_layers": 6,
        "kfac_target_layers": KFAC_TARGET_LAYERS,
        "kfac_damping": KFAC_DAMPING,
        "pilot_n_train": PILOT_N_TRAIN,
        "trak_k": TRAK_K,
        "seed": SEED,
        "methods_per_task": all_results,
        "h6_analysis": h6_analysis,
        "h6_decision": decision,
        "kfac_spectral": kfac_spectral,
        "total_runtime_sec": round(total_time, 2),
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat(),
    }

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
    print(f"{'Task':<15}{'Method':<12}{'Score':<10}{'Gap(pp)':<10}{'Time(s)':<8}")
    print("-" * 55)
    for tn in all_results:
        for method in ["repsim", "kfac_if", "diag_if", "trak"]:
            r = all_results[tn][method]
            if tn == "toxicity":
                score = r["metrics"]["AUPRC"]
                repsim_score = all_results[tn]["repsim"]["metrics"]["AUPRC"]
            else:
                score = r["metrics"].get("Recall@50", 0) + r["metrics"].get("MRR", 0)
                rm = all_results[tn]["repsim"]["metrics"]
                repsim_score = rm.get("Recall@50", 0) + rm.get("MRR", 0)
            gap = (score - repsim_score) * 100
            print(f"{tn:<15}{method:<12}{score:<10.4f}{gap:+<10.1f}{r['runtime_sec']:<8.1f}")
    print(f"-" * 55)
    print(f"Total runtime: {total_time:.1f}s  Decision: {decision}")
    print(f"{'='*70}")

    mark_done(
        "success",
        f"H6 decision={decision}. Total={total_time:.0f}s. "
        + "; ".join(f"{tn}: gap={v['gap_repsim_kfac_pp']}pp" for tn, v in h6_analysis.items()),
        final
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
