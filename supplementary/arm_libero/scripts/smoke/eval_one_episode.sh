#!/usr/bin/env bash
# Evaluation-chain smoke with the real model and simulator: the released OPSD student plays one standard
# LIBERO-Long episode (CPU LIBERO worker). Needs a host prepared by scripts/eval/setup.sh (EVAL_ROOT).
#   EVAL_ROOT=/data/arm-eval bash scripts/smoke/eval_one_episode.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${EVAL_ROOT:?EVAL_ROOT required}"
MODEL_PY=${MODEL_PY:-$EVAL_ROOT/envs/model/bin/python}
student=${STUDENT:-}
if [[ -z "$student" ]]; then
    student=$(PYTHONPATH="$root" "$MODEL_PY" -m arm_opsd.evaluation.cli fetch-student --output "$EVAL_ROOT/students" | tail -1)
fi
out=${OUT:-$EVAL_ROOT/smoke/one-episode-$(date +%Y%m%dT%H%M%S)}
POLICY=student_1_2 STUDENT=$student OUT=$out MODEL_PY=$MODEL_PY bash "$root/scripts/eval/libero_long.sh" 1
ls "$out/libero_long/student_1_2"/*.json
