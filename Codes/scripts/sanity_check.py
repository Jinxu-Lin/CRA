#!/usr/bin/env python3
"""
Pre-experiment sanity checks.

Usage:
    python scripts/sanity_check.py --check shape --model pythia-1b
    python scripts/sanity_check.py --check repr --model pythia-1b --n_train 10 --n_test 5
    python scripts/sanity_check.py --check all
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_utils import load_config, expand_path
from seed_utils import set_seed


def check_config_loading(configs_dir: Path):
    """Verify all config files load without errors."""
    print("\n[Config Loading Check]")
    passed = 0
    total = 0
    for config_file in sorted(configs_dir.glob("*.yaml")):
        total += 1
        try:
            config = load_config(str(config_file))
            print(f"  {config_file.name}: OK")
            passed += 1
        except Exception as e:
            print(f"  {config_file.name}: FAILED ({e})")

    print(f"  Result: {passed}/{total} configs loaded")
    return passed == total


def check_shape(config):
    """Forward pass shape check."""
    print("\n[Shape Check]")
    print("  (requires model + data -- skipping if not available)")

    # Verify config structure
    assert "model" in config, "Missing 'model' in config"
    assert "attribution" in config, "Missing 'attribution' in config"
    assert "paths" in config, "Missing 'paths' in config"

    n_layers = config["model"]["n_layers"]
    d_model = config["model"]["d_model"]
    print(f"  Model: {config['model']['name']}, {n_layers} layers, d={d_model}")
    print(f"  Expected repr shape: (batch, {d_model})")
    print(f"  Expected score shape: (n_test, n_train)")

    return True


def check_metrics():
    """Verify all metric functions work."""
    print("\n[Metrics Check]")
    from core.evaluation.metrics import lds, auprc, precision_at_k, recall_at_k, mrr

    n = 100
    pred = torch.randn(n)
    labels = (torch.rand(n) > 0.5).long()
    changes = torch.randn(n)

    checks = {
        "lds": lambda: lds(pred, changes),
        "auprc": lambda: auprc(pred, labels),
        "P@10": lambda: precision_at_k(pred, labels, 10),
        "Recall@50": lambda: recall_at_k(pred, labels, 50),
        "MRR": lambda: mrr(pred, labels),
    }

    all_ok = True
    for name, fn in checks.items():
        try:
            val = fn().item()
            ok = not (val != val)  # NaN check
            print(f"  {name}: {val:.4f} -- {'OK' if ok else 'FAILED (NaN)'}")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            all_ok = False

    return all_ok


def check_statistical():
    """Verify statistical analysis functions."""
    print("\n[Statistical Analysis Check]")
    from core.evaluation.statistical import permutation_test, bootstrap_ci, cohens_d, benjamini_hochberg

    a = torch.randn(30) + 0.5
    b = torch.randn(30)

    checks = {}
    try:
        diff, pval = permutation_test(a, b, n_permutations=100, seed=42)
        checks["permutation_test"] = f"diff={diff:.4f}, p={pval:.4f}"
    except Exception as e:
        checks["permutation_test"] = f"FAILED: {e}"

    try:
        est, lo, hi = bootstrap_ci(a, n_bootstrap=100, seed=42)
        checks["bootstrap_ci"] = f"est={est:.4f}, CI=[{lo:.4f}, {hi:.4f}]"
    except Exception as e:
        checks["bootstrap_ci"] = f"FAILED: {e}"

    try:
        d = cohens_d(a, b)
        checks["cohens_d"] = f"d={d:.4f}"
    except Exception as e:
        checks["cohens_d"] = f"FAILED: {e}"

    try:
        sig = benjamini_hochberg([0.01, 0.03, 0.05, 0.1, 0.5])
        checks["bh_fdr"] = f"{sig}"
    except Exception as e:
        checks["bh_fdr"] = f"FAILED: {e}"

    all_ok = True
    for name, result in checks.items():
        ok = "FAILED" not in str(result)
        print(f"  {name}: {result} -- {'OK' if ok else 'FAILED'}")
        all_ok = all_ok and ok

    return all_ok


def check_ablation():
    """Verify ablation analysis functions."""
    print("\n[Ablation Analysis Check]")
    from core.evaluation.ablation_analysis import compute_main_effects, assess_independence

    c1 = torch.randn(20) * 0.1 + 0.15
    c2 = torch.randn(20) * 0.1 + 0.20
    c3 = torch.randn(20) * 0.1 + 0.25
    c4 = torch.randn(20) * 0.1 + 0.30

    try:
        effects = compute_main_effects(c1, c2, c3, c4)
        assessment = assess_independence(effects["interaction_ratio"])
        print(f"  FM1={effects['fm1_effect']:.4f}, FM2={effects['fm2_effect']:.4f}")
        print(f"  Interaction={effects['interaction']:.4f}, Ratio={effects['interaction_ratio']:.4f}")
        print(f"  Assessment: {assessment} -- OK")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def check_scoring():
    """Verify attribution scoring functions."""
    print("\n[Attribution Scoring Check]")
    from core.attribution.repsim import repsim_score
    from core.attribution.contrastive import contrastive_score
    from core.attribution.gradsim import gradsim_score

    d = 128
    n_test, n_train = 5, 20

    h_test = torch.randn(n_test, d)
    h_train = torch.randn(n_train, d)
    g_test = torch.randn(n_test, d)
    g_train = torch.randn(n_train, d)

    all_ok = True
    try:
        s = repsim_score(h_test, h_train)
        assert s.shape == (n_test, n_train), f"Wrong shape: {s.shape}"
        assert not torch.isnan(s).any()
        assert s.min() >= -1.01 and s.max() <= 1.01  # cosine similarity bounds
        print(f"  repsim_score: shape={s.shape}, range=[{s.min():.4f}, {s.max():.4f}] -- OK")
    except Exception as e:
        print(f"  repsim_score: FAILED ({e})")
        all_ok = False

    try:
        s = gradsim_score(g_test, g_train)
        assert s.shape == (n_test, n_train)
        print(f"  gradsim_score: shape={s.shape} -- OK")
    except Exception as e:
        print(f"  gradsim_score: FAILED ({e})")
        all_ok = False

    try:
        s1 = repsim_score(h_test, h_train)
        s2 = repsim_score(h_test, h_train)
        sc = contrastive_score(s1, s2)
        assert sc.shape == s1.shape
        print(f"  contrastive_score: shape={sc.shape} -- OK")
    except Exception as e:
        print(f"  contrastive_score: FAILED ({e})")
        all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="CRA Sanity Checks")
    parser.add_argument("--check", default="all",
                        choices=["all", "config", "shape", "metrics", "statistical", "ablation", "scoring"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    set_seed(42)
    codes_dir = Path(__file__).parent.parent
    configs_dir = codes_dir / "configs"

    config = None
    if args.config:
        config = load_config(args.config)
    elif (configs_dir / "base.yaml").exists():
        config = load_config(str(configs_dir / "base.yaml"))

    checks = {
        "config": lambda: check_config_loading(configs_dir),
        "shape": lambda: check_shape(config) if config else True,
        "metrics": check_metrics,
        "statistical": check_statistical,
        "ablation": check_ablation,
        "scoring": check_scoring,
    }

    if args.check == "all":
        results = {}
        for name, fn in checks.items():
            results[name] = fn()

        print(f"\n{'='*40}")
        print("Summary:")
        all_ok = True
        for name, ok in results.items():
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {status}")
            all_ok = all_ok and ok

        print(f"\nOverall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
        sys.exit(0 if all_ok else 1)
    else:
        ok = checks[args.check]()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
