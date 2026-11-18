#!/usr/bin/env bash
# Stage 2.2 M-OPD teacher cache: the specialist teachers collect only their tasks, which replace those
# tasks' entries of the OPSD teacher cache (build that first: scripts/stage1_2_opsd/build_teacher_cache.sh).
#   scripts/stage2_2_mopd/build_specialist_cache.sh OUTPUT_DIR [extra options]
# Uses OPSD_TEACHER_CACHE as the previous cache; the merged cache is OUTPUT_DIR/cache (MOPD_TEACHER_CACHE).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}; shift
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m wam_policy.mopd cache --config "$ROOT/configs/stage2_2_mopd/mopd.json" \
    --previous-cache "${OPSD_TEACHER_CACHE:?set OPSD_TEACHER_CACHE}" --output "$OUT" "$@"
