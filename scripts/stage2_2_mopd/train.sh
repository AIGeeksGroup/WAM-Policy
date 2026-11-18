#!/usr/bin/env bash
# Stage 2.2 M-OPD training: OPSD S300 EMA student (OPSD_STUDENT) distilled from the task-routed teacher.
#   scripts/stage2_2_mopd/train.sh OUTPUT_DIR [extra options]
# The paper student is OUTPUT_DIR/learner/students/step_00000300/ema.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}; shift
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m wam_policy.mopd train --config "$ROOT/configs/stage2_2_mopd/mopd.json" --output "$OUT" "$@"
