#!/usr/bin/env bash
# Re-export the student of a resume checkpoint (release student format; Stage 1.2 and 2.2).
#   scripts/stage1_2_opsd/export.sh RUN_DIR STEP OUTPUT_DIR [raw|ema]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?run directory}; STEP=${2:?update}; OUT=${3:?output directory}; VARIANT=${4:-ema}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m wam_policy.opsd export --config "$RUN/config.json" \
    --checkpoint "$RUN/learner/checkpoints/$(printf 'step_%08d' "$STEP")" --variant "$VARIANT" --output "$OUT"
