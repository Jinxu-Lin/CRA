#!/bin/bash
# Run pilot experiment (1/10 scale, quick verification)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== CRA Pilot Run ==="

python run_attribution.py --config configs/pilot.yaml --dry-run --max-steps 2
python evaluate.py --config configs/pilot.yaml --dry-run

echo "=== Pilot Complete ==="
