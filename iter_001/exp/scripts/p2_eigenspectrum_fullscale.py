#!/usr/bin/env python3
"""
P2: Full-Scale Eigenspectrum at Multiple N (H4-revised, H9-revised)
On Pythia-70M (d=512, B~70M), compute gradient and representation
eigenspectra at N in {500, 1000, 2000, 5000} using CountSketch + SVD.
"""
import os, sys, json, time, gc, warnings
os.environ.setdefault("TMPDIR", "/home/jinxulin/tmp")
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.sparse.linalg import svds
warnings.filterwarnings("ignore")

TASK_ID = "p2_eigenspectrum_fullscale"
SEED = 42
MODEL_NAME = "EleutherAI/pythia-70m"
CHECKPOINT_CACHE = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-70m"
PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/CRA"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
FULL_DIR = f"{RESULTS_DIR}/full"
DEVICE = "cuda:0"
N_VALUES = [500, 1000, 2000, 5000]
TOP_K = 500
TARGET_LAYERS = [4, 5]
MAX_LEN = 256
BATCH_SIZE = 16

np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(FULL_DIR, exist_ok=True)

# ── Lifecycle helpers ───────────────────────────────────────────────────
def report_progress(metric=None):
    Path(RESULTS_DIR, f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "metric": metric or {},
        "updated_at": datetime.now().isoformat()}))

def mark_done(status="success", summary=""):
    pf = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pf.exists(): pf.unlink()
    fp = Path(RESULTS_DIR, f"{TASK_ID}_PROGRESS.json")
    final = json.loads(fp.read_text()) if fp.exists() else {}
    Path(RESULTS_DIR, f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final, "timestamp": datetime.now().isoformat()}))

def compute_r_eff(eigs, threshold=0.95):
    s = np.sort(np.abs(eigs))[::-1]
    t = s.sum()
    if t < 1e-15: return len(s)
    c = np.cumsum(s) / t
    return min(int(np.searchsorted(c, threshold)) + 1, len(s))

def compute_cond(eigs, eps=1e-12):
    a = np.abs(eigs); a = a[a > eps]
    return float(a.max() / a.min()) if len(a) >= 2 else float('inf')

def var_fracs(eigs):
    s = np.sort(np.abs(eigs))[::-1]; t = s.sum()
    if t < 1e-15: return {}
    c = np.cumsum(s) / t
    return {f"top_{k}": float(c[min(k-1, len(c)-1)]) for k in [1,5,10,20,50,100,200,500] if k <= len(c)}

# ── Representation eigenspectrum (exact) ────────────────────────────────
def rep_eigen(model, tokenizer, texts, n):
    print(f"  [Rep N={n}] Extracting representations...")
    t0 = time.time()
    reps = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            enc = tokenizer(texts[i:i+BATCH_SIZE], return_tensors="pt",
                            padding=True, truncation=True, max_length=MAX_LEN)
            out = model(input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE),
                        output_hidden_states=True)
            h = out.hidden_states[-1]
            m = enc["attention_mask"].to(DEVICE).unsqueeze(-1).float()
            reps.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().float())
    R = torch.cat(reps).numpy()
    d = R.shape[1]
    Rc = R - R.mean(0, keepdims=True)
    cov = (Rc.T @ Rc) / (n - 1)
    eigs = np.linalg.eigh(cov)[0][::-1].copy()
    elapsed = time.time() - t0
    print(f"  [Rep N={n}] Done in {elapsed:.1f}s, r_eff95={compute_r_eff(eigs)}")
    return {"eigenvalues": eigs.tolist(), "d": d, "n": n,
            "r_eff_90": compute_r_eff(eigs, 0.90), "r_eff_95": compute_r_eff(eigs, 0.95),
            "r_eff_99": compute_r_eff(eigs, 0.99), "condition_number": compute_cond(eigs),
            "condition_number_top100": compute_cond(eigs[:min(100, len(eigs))]),
            "explained_var": var_fracs(eigs),
            "n_nonzero": int(np.sum(np.abs(eigs) > 1e-12)), "runtime_sec": round(elapsed, 2)}

# ── Gradient eigenspectrum via CountSketch + SVD ────────────────────────
def grad_eigen(model, tokenizer, texts, n, param_filter=None, label="full"):
    """CountSketch the per-sample gradients, then truncated SVD of sketch."""
    print(f"  [Grad-{label} N={n}] Starting CountSketch gradient eigenspectrum...")
    t0 = time.time()

    # Select parameters
    if param_filter == "target_layers":
        params = []
        for name, p in model.named_parameters():
            if p.requires_grad and any(f"layers.{l}." in name for l in TARGET_LAYERS):
                params.append(p)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    total_dim = sum(p.numel() for p in params)
    print(f"  [Grad-{label}] Param dim B={total_dim:,}, {len(params)} tensors")

    # CountSketch setup
    sketch_dim = min(max(2000, TOP_K * 4), n - 1, total_dim)
    rng = np.random.RandomState(SEED + 999)
    hash_arr = torch.from_numpy(rng.randint(0, sketch_dim, size=total_dim, dtype=np.int32)).to(DEVICE)
    sign_arr = torch.from_numpy(rng.choice([-1.0, 1.0], size=total_dim).astype(np.float32)).to(DEVICE)
    sketch = np.zeros((n, sketch_dim), dtype=np.float32)

    for i in range(n):
        model.zero_grad()
        enc = tokenizer([texts[i]], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN)
        out = model(input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE), labels=enc["input_ids"].to(DEVICE))
        out.loss.backward()

        parts = []
        for p in params:
            parts.append(p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.numel(), device=DEVICE))
        g = torch.cat(parts)
        row = torch.zeros(sketch_dim, device=DEVICE)
        row.scatter_add_(0, hash_arr.long(), g * sign_arr)
        sketch[i] = row.cpu().numpy()
        del g, row, parts

        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  [Grad-{label} N={n}] {i+1}/{n} ({elapsed:.0f}s, ETA {eta:.0f}s, {rate:.1f} s/s)")
            report_progress({"phase": f"grad_{label}_N{n}", "done": i+1, "total": n, "eta": round(eta)})

    del hash_arr, sign_arr
    torch.cuda.empty_cache(); gc.collect()

    sketch_time = time.time() - t0
    sketch_c = sketch - sketch.mean(0, keepdims=True)
    k = min(TOP_K, sketch_dim - 1, n - 2)
    print(f"  [Grad-{label} N={n}] Sketch done in {sketch_time:.0f}s, SVD {n}x{sketch_dim} -> top-{k}...")

    _, S, _ = svds(sketch_c.astype(np.float64), k=k, which='LM')
    S = np.sort(S)[::-1]
    eigs = (S ** 2) / (n - 1)
    total_time = time.time() - t0

    print(f"  [Grad-{label} N={n}] Done in {total_time:.1f}s, r_eff95={compute_r_eff(eigs)}")
    return {"eigenvalues": eigs.tolist(), "method": "countsketch_svd",
            "sketch_dim": sketch_dim, "top_k_computed": k,
            "total_param_dim": total_dim, "n": n,
            "r_eff_90": compute_r_eff(eigs, 0.90), "r_eff_95": compute_r_eff(eigs, 0.95),
            "r_eff_99": compute_r_eff(eigs, 0.99), "condition_number_top": compute_cond(eigs),
            "explained_var": var_fracs(eigs),
            "n_nonzero": int(np.sum(np.abs(eigs) > 1e-12)),
            "runtime_sec": round(total_time, 2),
            "note": "CountSketch approximation; top eigenvalues reliable, tail may be distorted."}

# ── Main ────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    Path(RESULTS_DIR, f"{TASK_ID}.pid").write_text(str(os.getpid()))
    report_progress({"phase": "loading"})

    print(f"[{TASK_ID}] Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CHECKPOINT_CACHE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CHECKPOINT_CACHE,
                                                  dtype=torch.float32)
    model.to(DEVICE); model.eval()
    d = model.config.hidden_size
    B = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{TASK_ID}] d={d}, layers={model.config.num_hidden_layers}, B={B:,}")

    print(f"[{TASK_ID}] Loading data...")
    ds = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering",
                      "XSTest-response-Het", split="train")
    texts = []
    for i in range(min(max(N_VALUES), len(ds))):
        row = ds[i]
        t = row.get("text", "")
        if not t:
            t = (row.get("prompt", "") + " " + row.get("response", "")).strip()
        if not t:
            t = "empty"  # fallback to avoid 0-length tokenization
        texts.append(t)
    print(f"[{TASK_ID}] {len(texts)} samples loaded")

    n_values = [n for n in N_VALUES if n <= len(texts)]
    results = {"task_id": TASK_ID, "model": MODEL_NAME, "hidden_dim": d,
               "total_params": B, "n_values": n_values, "target_layers": TARGET_LAYERS,
               "seed": SEED, "representation": {}, "gradient_target_layers": {},
               "gradient_full_model": {}, "r_eff_scaling": {}, "spectral_concentration": {},
               "hypotheses": {}}

    for ni, n in enumerate(n_values):
        print(f"\n{'='*60}\n[{TASK_ID}] N={n} ({ni+1}/{len(n_values)})\n{'='*60}")

        results["representation"][str(n)] = rep_eigen(model, tok, texts, n)

        results["gradient_target_layers"][str(n)] = grad_eigen(
            model, tok, texts, n, param_filter="target_layers", label="target")

        results["gradient_full_model"][str(n)] = grad_eigen(
            model, tok, texts, n, param_filter=None, label="full")

        torch.cuda.empty_cache(); gc.collect()

    # ── Scaling analysis ────────────────────────────────────────────────
    scaling = {"N": n_values}
    for space, key in [("rep", "representation"), ("target", "gradient_target_layers"),
                       ("full", "gradient_full_model")]:
        for pct in ["90", "95", "99"]:
            scaling[f"{space}_r_eff_{pct}"] = [results[key][str(n)][f"r_eff_{pct}"] for n in n_values]
    results["r_eff_scaling"] = scaling

    print(f"\n{'='*60}\n[{TASK_ID}] r_eff Scaling\n{'='*60}")
    print(f"  {'N':>6} | {'rep_95':>8} | {'tgt_95':>8} | {'full_95':>8}")
    for i, n in enumerate(n_values):
        print(f"  {n:>6} | {scaling['rep_r_eff_95'][i]:>8} | "
              f"{scaling['target_r_eff_95'][i]:>8} | {scaling['full_r_eff_95'][i]:>8}")

    # ── Spectral concentration ratio ────────────────────────────────────
    conc = {}
    for n in n_values:
        ns = str(n)
        r95_rep = results["representation"][ns]["r_eff_95"]
        r95_full = results["gradient_full_model"][ns]["r_eff_95"]
        B_full = results["gradient_full_model"][ns]["total_param_dim"]
        rep_frac = r95_rep / d
        full_frac = r95_full / B_full if B_full > 0 else 0
        conc[ns] = {"rep_r95_over_d": round(rep_frac, 4),
                     "full_r95_over_B": round(full_frac, 10),
                     "ratio": round(rep_frac / full_frac, 1) if full_frac > 0 else float('inf')}
        print(f"  N={n}: rep_r95/d={rep_frac:.4f}, full_r95/B={full_frac:.2e}, ratio={conc[ns]['ratio']:.0f}x")
    results["spectral_concentration"] = conc

    # ── Hypotheses ──────────────────────────────────────────────────────
    ns_max = str(n_values[-1])
    full_max = results["gradient_full_model"][ns_max]
    rep_max = results["representation"][ns_max]
    conc_max = conc[ns_max]

    h4 = {"description": "Gradient signal in low-rank subspace (r_eff << B)",
           "full_r_eff_95": full_max["r_eff_95"], "B": full_max["total_param_dim"],
           "fraction": round(full_max["r_eff_95"] / full_max["total_param_dim"], 8),
           "pass_criteria": "full_model r_eff(95%) < 100", "pass": full_max["r_eff_95"] < 100,
           "trend": [results["gradient_full_model"][str(n)]["r_eff_95"] for n in n_values]}

    h9 = {"description": "rep_r_eff/d >> grad_r_eff/B (>= 10x)",
           "ratio": conc_max["ratio"], "pass": conc_max["ratio"] >= 10}

    results["hypotheses"] = {"H4_revised": h4, "H9_revised": h9}

    # ── Summary ─────────────────────────────────────────────────────────
    total = time.time() - start
    results["summary"] = {
        "runtime_sec": round(total, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
        "findings": [
            f"Full-model gradient r_eff(95%)={full_max['r_eff_95']} at N={n_values[-1]}, "
            f"fraction of B: {h4['fraction']:.2e}",
            f"Rep r_eff(95%)={rep_max['r_eff_95']} at N={n_values[-1]}, fraction of d: {conc_max['rep_r95_over_d']:.4f}",
            f"Concentration ratio: {conc_max['ratio']:.0f}x ({'PASS' if h9['pass'] else 'FAIL'})",
            f"H4: {'PASS' if h4['pass'] else 'FAIL'}, H9: {'PASS' if h9['pass'] else 'FAIL'}"
        ]}

    print(f"\n{'='*60}\n[{TASK_ID}] DONE ({total:.0f}s = {total/60:.1f}min)")
    for f in results["summary"]["findings"]: print(f"  * {f}")
    print(f"{'='*60}")

    Path(FULL_DIR, f"{TASK_ID}.json").write_text(json.dumps(results, indent=2))
    mark_done("success", f"r_eff95_full={full_max['r_eff_95']}, ratio={conc_max['ratio']:.0f}x, "
              f"H4={'PASS' if h4['pass'] else 'FAIL'}, H9={'PASS' if h9['pass'] else 'FAIL'}, "
              f"runtime={total:.0f}s")
    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        mark_done("failed", str(e))
        sys.exit(1)
