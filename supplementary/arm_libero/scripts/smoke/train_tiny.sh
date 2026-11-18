#!/usr/bin/env bash
# Training-chain smoke and environment self-check of the model environment: a tiny random LingBot-VA
# builds a teacher cache from synthetic journals, then each of the five students of configs/students/
# trains a few updates, checkpoints, resumes and exports. CPU, about 10 s, no weights, no physics.
#   bash scripts/smoke/train_tiny.sh [OUTPUT_DIR] [cpu|cuda]
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output=${1:-$(mktemp -d)/arm-opsd-tiny-train}
cd "$root"
PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python}" -m arm_opsd.cli tiny-smoke --output "$output" --device "${2:-cpu}"
echo "report: $output/report.json"
