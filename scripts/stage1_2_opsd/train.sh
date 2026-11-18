#!/usr/bin/env bash
# Stage 1.2 OPSD training on one node (torchrun learner + one QLM CPU worker per rank).
#   scripts/stage1_2_opsd/train.sh OUTPUT_DIR [CONFIG] [extra options: --resume CKPT, --set ..., --resource ...]
# CONFIG defaults to configs/stage1_2_opsd/opsd.json (ablations: configs/stage1_2_opsd/ablations/*.json).
# The paper student is OUTPUT_DIR/learner/students/step_00000300/ema.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}; shift
CONFIG=${1:-$ROOT/configs/stage1_2_opsd/opsd.json}; [ $# -gt 0 ] && shift
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m wam_policy.opsd train --config "$CONFIG" --output "$OUT" "$@"
