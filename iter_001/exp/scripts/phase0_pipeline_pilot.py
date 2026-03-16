#!/usr/bin/env python3
"""
Phase 0a: Pipeline Validation Pilot
- RepSim (last-layer cosine similarity) on Pythia-1B
- TRAK (k=2048 random projection of per-sample gradients) on Pythia-1B
- Evaluate on DATE-LM toxicity (XSTest-response-Het) and counterfact (Pythia-1b)
- Pilot: 100 training samples, seed=42
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

TASK_ID = "phase0_pipeline_pilot"
SEED = 42
PILOT_N_TRAIN = 100
DEVICE = "cuda:0"
MODEL_NAME = "EleutherAI/pythia-1b"
CHECKPOINT_DIR = "/home/jinxulin/sibyl_system/shared/checkpoints/pythia-1b/models--EleutherAI--pythia-1b/snapshots/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
RESULTS_DIR = "/home/jinxulin/sibyl_system/projects/CRA/exp/results"
PHASE0_DIR = os.path.join(RESULTS_DIR, "phase0")
PILOTS_DIR = os.path.join(RESULTS_DIR, "pilots")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
TRAK_K = 2048

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


# ── Evaluation ──────────────────────────────────────────────────────────
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


# ── Data ────────────────────────────────────────────────────────────────
def load_tasks():
    report_progress("loading_data", "Loading DATE-LM", 0.05)
    tasks = {}
    tox = load_dataset("DataAttributionEval/Toxicity-Bias-Filtering", "XSTest-response-Het")
    tasks["toxicity"] = {"train": tox["train"], "ref": tox["ref"], "metric": "AUPRC"}
    print(f"[toxicity] train={len(tox['train'])}, ref={len(tox['ref'])}")

    cf = load_dataset("DataAttributionEval/Counterfact", "Pythia-1b")
    tasks["counterfact"] = {"train": cf["train"], "ref": cf["ref"], "metric": "Recall@50+MRR"}
    print(f"[counterfact] train={len(cf['train'])}, ref={len(cf['ref'])}")
    return tasks


def load_model():
    report_progress("loading_model", "Loading Pythia-1B", 0.10)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded: hidden_dim={model.config.hidden_size}")
    return model, tok


def fmt(sample):
    return sample["prompt"] + " " + sample["response"]


# ── RepSim ──────────────────────────────────────────────────────────────
def extract_reps(model, tok, texts, bs=8, max_len=512):
    reps = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            sl = inp["attention_mask"].sum(1) - 1
            bi = torch.arange(h.size(0), device=DEVICE)
            r = F.normalize(h[bi, sl].float(), dim=-1)
            reps.append(r.cpu())
    return torch.cat(reps, dim=0)


def run_repsim(model, tok, task_name, train_data, ref_data, n_train):
    report_progress("repsim", f"RepSim on {task_name}", 0.20)
    t0 = time.time()
    train_texts = [fmt(train_data[i]) for i in range(n_train)]
    ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]

    print(f"[RepSim/{task_name}] Extracting reps...")
    train_reps = extract_reps(model, tok, train_texts)
    ref_reps = extract_reps(model, tok, ref_texts)

    sim = train_reps @ ref_reps.T
    elapsed = time.time() - t0
    print(f"[RepSim/{task_name}] Done in {elapsed:.1f}s, sim shape={sim.shape}")

    if task_name == "counterfact":
        return [sim[:, j].numpy() for j in range(sim.shape[1])], elapsed
    return sim.mean(dim=1).numpy(), elapsed


# ── TRAK ────────────────────────────────────────────────────────────────
def run_trak(model, tok, task_name, train_data, ref_data, n_train, k=TRAK_K):
    """
    TRAK: per-sample gradient similarity via random projection.
    Streaming chunked projection to avoid materializing full [D, k] matrix.
    Target: last layer attention.dense + mlp.dense_4h_to_h (~21M params).
    """
    report_progress("trak", f"TRAK on {task_name}", 0.50)
    t0 = time.time()

    train_texts = [fmt(train_data[i]) for i in range(n_train)]
    ref_texts = [fmt(ref_data[i]) for i in range(len(ref_data))]

    # Select target parameters
    target_params = []
    target_names = []
    for name, p in model.named_parameters():
        if any(
            sub in name
            for sub in ["layers.15.attention.dense.weight", "layers.15.mlp.dense_4h_to_h.weight"]
        ):
            p.requires_grad_(True)
            target_params.append(p)
            target_names.append(name)
        else:
            p.requires_grad_(False)

    D = sum(p.numel() for p in target_params)
    print(f"[TRAK/{task_name}] D={D / 1e6:.2f}M from {target_names}")

    # CountSketch projection: O(D) time, O(k) space per sample -- no matrix needed
    proj_seed = SEED + 7777
    rng_cs = np.random.RandomState(proj_seed)
    cs_buckets = torch.from_numpy(rng_cs.randint(0, k, size=D).astype(np.int64))
    cs_signs = torch.from_numpy(rng_cs.choice([-1.0, 1.0], size=D).astype(np.float32))
    print(f"[TRAK/{task_name}] CountSketch D={D} -> k={k}")

    def compute_projected_grads(texts, desc=""):
        all_proj = []
        for idx, text in enumerate(texts):
            inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
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

            if (idx + 1) % 10 == 0:
                print(f"  [{desc}] {idx + 1}/{len(texts)}")
                torch.cuda.empty_cache()

        return torch.stack(all_proj)

    print(f"[TRAK/{task_name}] Train projected grads ({n_train})...")
    train_proj = compute_projected_grads(train_texts, f"TRAK-train-{task_name}")
    print(f"[TRAK/{task_name}] Ref projected grads ({len(ref_texts)})...")
    ref_proj = compute_projected_grads(ref_texts, f"TRAK-ref-{task_name}")

    train_proj = F.normalize(train_proj, dim=-1)
    ref_proj = F.normalize(ref_proj, dim=-1)
    sim = train_proj @ ref_proj.T

    for p in model.parameters():
        p.requires_grad_(True)
    gc.collect()

    elapsed = time.time() - t0
    print(f"[TRAK/{task_name}] Done in {elapsed:.1f}s")

    if task_name == "counterfact":
        return [sim[:, j].numpy() for j in range(sim.shape[1])], elapsed
    return sim.mean(dim=1).numpy(), elapsed


# ── Evaluate ────────────────────────────────────────────────────────────
def evaluate_pilot(task_name, pilot_subset, scores, method):
    """Evaluate on pilot subset. pilot_subset has 'data', 'indices', 'ref'."""
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
                if pdata[i]["counterfactual_entity"] == r["counterfactual_entity"]
                and pdata[i]["true_entity"] == r["true_entity"]
            ]
            fi.append(indices)
        recall, mrr = compute_factual_metrics(scores, fi, k=50)
        n_with_facts = sum(1 for f in fi if f)
        print(f"  [{method}/{task_name}] Recall@50={recall:.4f}, MRR={mrr:.4f} (refs_with_facts={n_with_facts})")
        return {"Recall@50": round(recall, 6), "MRR": round(mrr, 6), "refs_with_facts": n_with_facts}


# ── Main ────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    for d in [PHASE0_DIR, PILOTS_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print(f"Phase 0a: Pipeline Validation Pilot")
    print(f"Model: {MODEL_NAME}, N={PILOT_N_TRAIN}, TRAK k={TRAK_K}")
    print("=" * 70)

    tasks = load_tasks()
    model, tok = load_model()

    # Create pilot subsets with stratified sampling for toxicity
    pilot_subsets = {}
    for tn, ti in tasks.items():
        full_train = ti["train"]
        n_total = len(full_train)
        if tn == "toxicity":
            # Stratified: ensure unsafe samples are included
            unsafe_idx = [i for i in range(n_total) if full_train[i]["type"] == "Unsafe"]
            safe_idx = [i for i in range(n_total) if full_train[i]["type"] != "Unsafe"]
            rng_pilot = np.random.RandomState(SEED)
            n_unsafe = min(len(unsafe_idx), max(PILOT_N_TRAIN // 5, 5))
            n_safe = PILOT_N_TRAIN - n_unsafe
            chosen_unsafe = rng_pilot.choice(unsafe_idx, n_unsafe, replace=False).tolist()
            chosen_safe = rng_pilot.choice(safe_idx, min(n_safe, len(safe_idx)), replace=False).tolist()
            pilot_idx = sorted(chosen_unsafe + chosen_safe)
            print(f"[pilot/{tn}] Stratified: {n_unsafe} unsafe + {len(chosen_safe)} safe = {len(pilot_idx)}")
        else:
            rng_pilot = np.random.RandomState(SEED)
            pilot_idx = sorted(rng_pilot.choice(n_total, min(PILOT_N_TRAIN, n_total), replace=False).tolist())
            print(f"[pilot/{tn}] Random: {len(pilot_idx)} samples")
        pilot_subsets[tn] = {
            "data": full_train.select(pilot_idx),
            "indices": pilot_idx,
            "ref": ti["ref"],
        }

    results = {"repsim": {}, "trak": {}}
    timings = {}
    raw = {}

    # RepSim
    for tn, ti in tasks.items():
        ps = pilot_subsets[tn]
        n = len(ps["data"])
        scores, t = run_repsim(model, tok, tn, ps["data"], ps["ref"], n)
        met = evaluate_pilot(tn, ps, scores, "RepSim")
        results["repsim"][tn] = {"metrics": met, "runtime_sec": round(t, 2)}
        timings[f"repsim_{tn}"] = round(t, 2)
        raw[f"repsim_{tn}"] = scores
    gc.collect()
    torch.cuda.empty_cache()

    # TRAK
    for tn, ti in tasks.items():
        ps = pilot_subsets[tn]
        n = len(ps["data"])
        scores, t = run_trak(model, tok, tn, ps["data"], ps["ref"], n)
        met = evaluate_pilot(tn, ps, scores, "TRAK")
        results["trak"][tn] = {"metrics": met, "runtime_sec": round(t, 2)}
        timings[f"trak_{tn}"] = round(t, 2)
        raw[f"trak_{tn}"] = scores
        gc.collect()
        torch.cuda.empty_cache()

    # Sanity checks
    report_progress("analysis", "Sanity checks", 0.90)
    sanity = {}
    for tn in tasks:
        rm = results["repsim"][tn]["metrics"]
        tm = results["trak"][tn]["metrics"]
        if tn == "toxicity":
            rv, tv, mn = rm["AUPRC"], tm["AUPRC"], "AUPRC"
        else:
            rv = rm.get("Recall@50", 0) + rm.get("MRR", 0)
            tv = tm.get("Recall@50", 0) + tm.get("MRR", 0)
            mn = "Recall@50+MRR"
        gap = rv - tv
        ok = rv > tv
        sanity[tn] = {"repsim": round(rv, 6), "trak": round(tv, 6), "gap": round(gap, 6), "metric": mn, "pass": ok}
        print(f"[Sanity/{tn}] RepSim={rv:.4f} vs TRAK={tv:.4f} gap={gap:+.4f} [{'PASS' if ok else 'WARN'}]")

    overall = all(s["pass"] for s in sanity.values())

    # Qualitative: toxicity top-5
    if "repsim_toxicity" in raw and isinstance(raw["repsim_toxicity"], np.ndarray):
        rs = raw["repsim_toxicity"]
        top5 = np.argsort(-rs)[:5]
        print(f"\n[Qualitative] Toxicity RepSim top-5:")
        pdata = pilot_subsets["toxicity"]["data"]
        for rank, idx in enumerate(top5):
            s = pdata[int(idx)]
            print(f"  #{rank + 1} score={rs[idx]:.4f} type={s['type']}: {s['response'][:80]}...")

    # Score stats
    stats = {}
    for key, sc in raw.items():
        if isinstance(sc, np.ndarray):
            stats[key] = {
                "mean": float(np.mean(sc)),
                "std": float(np.std(sc)),
                "min": float(np.min(sc)),
                "max": float(np.max(sc)),
            }
        elif isinstance(sc, list) and sc:
            a = np.concatenate(sc)
            stats[key] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "n_ref": len(sc),
            }

    total = time.time() - t_start
    final = {
        "task_id": TASK_ID,
        "model": MODEL_NAME,
        "hidden_dim": 2048,
        "pilot_n_train": PILOT_N_TRAIN,
        "trak_k": TRAK_K,
        "seed": SEED,
        "methods": results,
        "sanity_checks": sanity,
        "overall_sanity_pass": overall,
        "score_stats": stats,
        "timings": timings,
        "total_runtime_sec": round(total, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }

    for p in [
        os.path.join(PHASE0_DIR, "pipeline_pilot.json"),
        os.path.join(PILOTS_DIR, f"{TASK_ID}_results.json"),
    ]:
        with open(p, "w") as f:
            json.dump(final, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Task':<15}{'Method':<10}{'Metric':<15}{'Value':<10}{'Time(s)':<8}")
    print("-" * 58)
    for tn in tasks:
        for m in ["repsim", "trak"]:
            for mk, mv in results[m][tn]["metrics"].items():
                if isinstance(mv, float):
                    print(f"{tn:<15}{m:<10}{mk:<15}{mv:<10.4f}{results[m][tn]['runtime_sec']:<8.1f}")
    print("-" * 58)
    print(f"Total: {total:.1f}s  Sanity: {'PASS' if overall else 'WARN'}")
    print("=" * 70)

    mark_done("success" if overall else "warn", f"Done. Sanity={'PASS' if overall else 'WARN'}. {total:.0f}s", final)
    return final


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"FATAL: {e}\n{traceback.format_exc()}")
        mark_done("failed", str(e)[:200])
        sys.exit(1)
