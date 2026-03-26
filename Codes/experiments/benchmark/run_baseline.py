#!/usr/bin/env python3
"""
Experiment 1: Run a single baseline method for reproduction verification.
"""

import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Run Single Baseline")
    parser.add_argument("--method", required=True, help="Method name (trak, bm25, gradsim)")
    parser.add_argument("--task", default="toxicity")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    codes_dir = Path(__file__).parent.parent.parent
    config = args.config or str(codes_dir / "configs" / f"benchmark_{args.task}.yaml")
    run_script = codes_dir / "run_attribution.py"

    cmd = [
        sys.executable, str(run_script),
        "--config", config,
        "--override", f"attribution.method={args.method}",
        f"evaluation.task={args.task}",
        "evaluation.n_seeds=1",
    ]
    if args.dry_run:
        cmd.extend(["--dry-run", "--max-steps", str(args.max_steps)])

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
