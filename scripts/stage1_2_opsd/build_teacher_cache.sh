#!/usr/bin/env bash
# Stage 1.2 teacher target cache (8 GPUs + 8 CPU simulation workers on one node).
#   scripts/stage1_2_opsd/build_teacher_cache.sh OUTPUT_DIR [extra options, e.g. --set world_size=4]
# The cache is OUTPUT_DIR/learner/cache; use it as OPSD_TEACHER_CACHE. An interrupted build can be
# completed from its journal: python -m wam_policy.opsd finalize-cache --cache OUTPUT_DIR/learner/cache
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}; shift
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m wam_policy.opsd cache --config "$ROOT/configs/stage1_2_opsd/teacher_cache.json" --output "$OUT" "$@"
