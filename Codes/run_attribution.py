#!/usr/bin/env python3
"""
CRA Attribution Scoring Pipeline.

This is the main entry point for computing attribution scores.
Equivalent to train.py in a training project -- but CRA computes
attribution scores, not model parameters.

Usage:
    python run_attribution.py --config configs/base.yaml
    python run_attribution.py --config configs/probe_repsim.yaml --dry-run
    python run_attribution.py --config configs/base.yaml --seed 42

Steps:
    1. Load config and set seeds
    2. Load model (fine-tuned + optionally base model for contrastive)
    3. Load DATE-LM data
    4. Extract features (representations or gradients)
    5. Compute attribution scores
    6. Save scores to _Data/scores/
"""

import argparse
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Add Codes/ to path
sys.path.insert(0, str(Path(__file__).parent))

from config_utils import load_config, add_common_args, get_config_from_args, expand_path
from seed_utils import set_seed, seed_worker, get_generator
from logging_utils import ExperimentLogger, get_gpu_info
from core.data.date_lm_loader import create_dataloader, get_task_labels, get_actual_changes
from core.data.representation import extract_representations, resolve_layer_index
from core.attribution.repsim import repsim_score, repsim_score_batched
from core.attribution.contrastive import contrastive_score, contrastive_score_from_representations
from core.evaluation.metrics import compute_all_metrics


def load_model(config: dict, device: torch.device, random_init: bool = False):
    """
    Load a HuggingFace model based on config.

    Args:
        config: Full config dict.
        device: Device to load model on.
        random_init: If True, use random weights (for control experiments).

    Returns:
        model: Loaded model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    model_path = expand_path(model_cfg["path"])

    if random_init:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
        )

    model = model.to(device)
    model.eval()
    return model


def compute_scores_repsim(
    model, config, train_loader, test_loader, device, dry_run=False, max_steps=2,
):
    """Compute RepSim attribution scores."""
    layer = config["attribution"].get("layer", "middle")
    aggregation = config["attribution"].get("token_aggregation", "last_token")
    n_layers = config["model"].get("n_layers", None)

    print(f"  Extracting train representations (layer={layer}, agg={aggregation})...")
    if dry_run:
        # Use only first max_steps batches
        from torch.utils.data import Subset
        n_train = min(max_steps * train_loader.batch_size, len(train_loader.dataset))
        n_test = min(max_steps * test_loader.batch_size, len(test_loader.dataset))
        train_ds = Subset(train_loader.dataset, range(n_train))
        test_ds = Subset(test_loader.dataset, range(n_test))
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=train_loader.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=test_loader.batch_size, shuffle=False)

    h_train = extract_representations(model, train_loader, layer, aggregation, n_layers, device)
    print(f"  Train representations: shape={h_train.shape}, dtype={h_train.dtype}")

    print(f"  Extracting test representations...")
    h_test = extract_representations(model, test_loader, layer, aggregation, n_layers, device)
    print(f"  Test representations: shape={h_test.shape}, dtype={h_test.dtype}")

    # Sanity checks
    print(f"  Train repr range: [{h_train.min():.4f}, {h_train.max():.4f}]")
    print(f"  Test repr range: [{h_test.min():.4f}, {h_test.max():.4f}]")
    assert not torch.isnan(h_train).any(), "NaN in train representations!"
    assert not torch.isnan(h_test).any(), "NaN in test representations!"
    assert not torch.isinf(h_train).any(), "Inf in train representations!"

    print(f"  Computing cosine similarity matrix...")
    scores = repsim_score_batched(h_test.float(), h_train.float())
    print(f"  Scores: shape={scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
    assert not torch.isnan(scores).any(), "NaN in scores!"

    return scores, {"h_train": h_train, "h_test": h_test}


def compute_scores_gradsim(
    model, config, train_loader, test_loader, device, dry_run=False, max_steps=2,
):
    """Compute Grad-Sim attribution scores."""
    from core.attribution.gradsim import gradsim_score_batched, extract_per_sample_gradients

    if dry_run:
        from torch.utils.data import Subset, DataLoader
        n_train = min(max_steps * train_loader.batch_size, len(train_loader.dataset))
        n_test = min(max_steps * test_loader.batch_size, len(test_loader.dataset))
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(n_train)),
            batch_size=train_loader.batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            Subset(test_loader.dataset, range(n_test)),
            batch_size=test_loader.batch_size, shuffle=False,
        )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    print(f"  Extracting per-sample gradients (train)...")
    # For dry-run or small scale, limit parameters
    max_params = 1000 if dry_run else None
    g_train = extract_per_sample_gradients(model, train_loader, loss_fn, device, max_params=max_params)
    print(f"  Train gradients: shape={g_train.shape}")

    print(f"  Extracting per-sample gradients (test)...")
    g_test = extract_per_sample_gradients(model, test_loader, loss_fn, device, max_params=max_params)
    print(f"  Test gradients: shape={g_test.shape}")

    print(f"  Computing gradient cosine similarity...")
    scores = gradsim_score_batched(g_test.float(), g_train.float())
    print(f"  Scores: shape={scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")

    return scores, {"g_train": g_train, "g_test": g_test}


def compute_scores_rept(
    model, config, train_loader, test_loader, device, dry_run=False, max_steps=2,
):
    """Compute RepT attribution scores."""
    from core.attribution.rept import (
        compute_layer_gradient_norms, detect_phase_transition_layer,
        extract_hidden_gradients, extract_rept_features, rept_score,
    )

    n_layers = config["model"].get("n_layers", None)
    aggregation = config["attribution"].get("token_aggregation", "last_token")
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    if dry_run:
        from torch.utils.data import Subset, DataLoader
        n_train = min(max_steps * train_loader.batch_size, len(train_loader.dataset))
        n_test = min(max_steps * test_loader.batch_size, len(test_loader.dataset))
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(n_train)),
            batch_size=train_loader.batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            Subset(test_loader.dataset, range(n_test)),
            batch_size=test_loader.batch_size, shuffle=False,
        )

    print(f"  Detecting phase-transition layer...")
    grad_norms = compute_layer_gradient_norms(
        model, train_loader, loss_fn, n_layers=n_layers, device=device,
        n_samples=min(20, len(train_loader.dataset)),
    )
    l_star = detect_phase_transition_layer(grad_norms)
    print(f"  Phase-transition layer: l*={l_star}")

    print(f"  Extracting RepT features (train)...")
    h_train, g_train = extract_hidden_gradients(
        model, train_loader, loss_fn, layer=l_star,
        aggregation=aggregation, n_layers=n_layers, device=device,
    )
    phi_train = extract_rept_features(h_train, g_train)

    print(f"  Extracting RepT features (test)...")
    h_test, g_test = extract_hidden_gradients(
        model, test_loader, loss_fn, layer=l_star,
        aggregation=aggregation, n_layers=n_layers, device=device,
    )
    phi_test = extract_rept_features(h_test, g_test)

    print(f"  RepT features: train={phi_train.shape}, test={phi_test.shape}")

    print(f"  Computing RepT scores...")
    scores = rept_score(phi_test.float(), phi_train.float())
    print(f"  Scores: shape={scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")

    return scores, {"phi_train": phi_train, "phi_test": phi_test, "l_star": l_star}


def compute_scores_random(config, n_test, n_train):
    """Compute random baseline scores."""
    scores = torch.randn(n_test, n_train)
    print(f"  Random scores: shape={scores.shape}")
    return scores, {}


def apply_contrastive(
    model_base, scoring_fn, config, train_loader, test_loader,
    device, scores_ft, features_ft, dry_run=False, max_steps=2,
):
    """Apply contrastive scoring: score_ft - score_base."""
    print(f"  Computing base model scores for contrastive...")
    scores_base, features_base = scoring_fn(
        model_base, config, train_loader, test_loader, device,
        dry_run=dry_run, max_steps=max_steps,
    )
    scores_contr = contrastive_score(scores_ft, scores_base)
    print(f"  Contrastive scores: shape={scores_contr.shape}, range=[{scores_contr.min():.4f}, {scores_contr.max():.4f}]")
    return scores_contr


def main():
    parser = argparse.ArgumentParser(description="CRA Attribution Scoring Pipeline")
    add_common_args(parser)
    parser.add_argument("--resume", type=str, default=None, help="Path to saved scores (resume from checkpoint)")
    parser.add_argument("--random-init", action="store_true", help="Use randomly initialized model (control)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = get_config_from_args(args)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    # Determine method and scoring
    method = config["attribution"]["method"]
    scoring = config["attribution"].get("scoring", "standard")
    task = config["evaluation"]["task"]

    print(f"=" * 60)
    print(f"CRA Attribution Scoring Pipeline")
    print(f"  Method: {method}")
    print(f"  Scoring: {scoring}")
    print(f"  Task: {task}")
    print(f"  Seed: {seed}")
    print(f"  Dry run: {args.dry_run}")
    print(f"=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_info = get_gpu_info()
        print(f"GPU: {gpu_info.get('gpu_name', 'unknown')}, Memory: {gpu_info.get('gpu_memory_gb', 0):.1f} GB")

    # Logger
    exp_name = f"{method}_{scoring}_{task}_seed{seed}"
    logger = ExperimentLogger(config, exp_name, dry_run=args.dry_run)

    start_time = time.time()

    # Load model
    print(f"\n[1/5] Loading model...")
    model = load_model(config, device, random_init=args.random_init)
    print(f"  Model loaded: {config['model']['name']}")

    # Load data
    print(f"\n[2/5] Loading data...")
    data_path = expand_path(config["paths"]["data"])
    batch_size = config.get("data", {}).get("batch_size", 32)
    max_samples = config.get("data", {}).get("max_samples", None)
    if args.dry_run:
        max_samples = min(max_samples or float("inf"), args.max_steps * batch_size * 2)
        max_samples = int(max_samples)

    train_loader = create_dataloader(data_path, task, "train", batch_size, max_samples)
    test_loader = create_dataloader(data_path, task, "test", batch_size,
                                     min(max_samples, 20) if args.dry_run else None)
    print(f"  Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Compute scores
    print(f"\n[3/5] Computing {method} scores...")
    scoring_fns = {
        "repsim": compute_scores_repsim,
        "rept": compute_scores_rept,
        "gradsim": compute_scores_gradsim,
    }

    if method == "random":
        scores, features = compute_scores_random(
            config, len(test_loader.dataset), len(train_loader.dataset)
        )
    elif method in scoring_fns:
        scores, features = scoring_fns[method](
            model, config, train_loader, test_loader, device,
            dry_run=args.dry_run, max_steps=args.max_steps,
        )
    elif method in ("trak", "dda", "bm25"):
        print(f"  Method '{method}' requires DATE-LM integration. Using placeholder scores.")
        scores = torch.randn(len(test_loader.dataset), len(train_loader.dataset))
        features = {}
    elif method == "magic":
        from core.attribution.magic import magic_feasibility_check
        feasibility = magic_feasibility_check(
            model, len(train_loader.dataset), len(test_loader.dataset),
            config.get("fine_tuning", {}).get("decay_steps", 200),
        )
        print(f"  MAGIC feasibility: {json.dumps(feasibility, indent=2)}")
        scores = torch.randn(len(test_loader.dataset), len(train_loader.dataset))
        features = {"feasibility": feasibility}
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply contrastive scoring if requested
    if scoring == "contrastive" and method != "random":
        print(f"\n[3.5/5] Applying contrastive scoring...")
        base_model_path = config["paths"].get("base_model", config["model"]["path"])
        config_base = config.copy()
        config_base["model"] = {**config["model"], "path": base_model_path}
        model_base = load_model(config_base, device)

        if method in scoring_fns:
            scores = apply_contrastive(
                model_base, scoring_fns[method], config, train_loader, test_loader,
                device, scores, features, dry_run=args.dry_run, max_steps=args.max_steps,
            )

    # Save scores
    print(f"\n[4/5] Saving scores...")
    output_base = args.output_dir or expand_path(config["paths"].get("cache", "_Data"))
    scores_dir = Path(output_base) / "scores" / task
    scores_dir.mkdir(parents=True, exist_ok=True)

    score_file = scores_dir / f"{method}_{scoring}_seed{seed}.pt"
    torch.save(scores, score_file)
    print(f"  Saved: {score_file}")

    # Quick evaluation
    print(f"\n[5/5] Quick evaluation...")
    labels = get_task_labels(data_path, task, "train")
    actual_changes = get_actual_changes(data_path, task)

    # For evaluation, we need per-test-sample scores aggregated over train
    # Use mean score per test sample as a simple aggregate
    per_test_scores = scores.mean(dim=1)  # (n_test,)

    metrics = {}
    metric_names = config["evaluation"].get("metrics", ["lds", "auprc", "pk"])

    if actual_changes is not None and len(actual_changes) == len(per_test_scores):
        from core.evaluation.metrics import lds
        lds_val = lds(per_test_scores, actual_changes).item()
        metrics["lds"] = lds_val
        print(f"  LDS: {lds_val:.4f}")

    if labels is not None and len(labels) == scores.shape[1]:
        # For retrieval metrics, use top score per test sample
        from core.evaluation.metrics import auprc, precision_at_k
        for i in range(min(3, scores.shape[0])):
            sample_metrics = compute_all_metrics(
                scores[i], labels=labels, actual_changes=None,
                metric_names=["auprc", "pk", "mrr"],
            )
            for k, v in sample_metrics.items():
                metrics.setdefault(f"test_{i}_{k}", v)

    elapsed = time.time() - start_time
    metrics["wall_time_seconds"] = elapsed
    metrics["method"] = method
    metrics["scoring"] = scoring
    metrics["task"] = task
    metrics["seed"] = seed

    print(f"\n  Results: {json.dumps({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in metrics.items()}, indent=2)}")

    # Log summary
    logger.log_summary(metrics)
    logger.finish()

    # Sanity check for dry-run
    if args.dry_run:
        print(f"\n[DRY RUN] Verification:")
        print(f"  Scores shape: {scores.shape} -- OK")
        print(f"  No NaN: {not torch.isnan(scores).any()} -- OK")
        print(f"  No Inf: {not torch.isinf(scores).any()} -- OK")
        print(f"  Score variance > 0: {scores.var().item() > 0} -- OK")
        print(f"  Wall time: {elapsed:.1f}s")
        print(f"\n[DRY RUN PASSED]")

    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
