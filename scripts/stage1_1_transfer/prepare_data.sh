#!/usr/bin/env bash
# Stage 1.1 data: download the ten QLM-Bench releases -> fit the train-split action normalizer ->
# build the frozen dual-view VAE latent + T5 text cache (resumable; re-run to continue).
#   scripts/stage1_1_transfer/prepare_data.sh [snapshot_dir] [cache_dir] [base_dir]
# Defaults: $QLM_BENCH_SNAPSHOT, $SFT_CACHE, $WAM_BASE_MODEL (else under $WAM_ROOT).
# The cache step needs one GPU (DEVICE=cuda) plus ffmpeg/PyAV for video decoding.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
SNAPSHOT=${1:-${QLM_BENCH_SNAPSHOT:-${WAM_ROOT:?set WAM_ROOT}/qlm-bench-data}}
CACHE=${2:-${SFT_CACHE:-$WAM_ROOT/stage1_1/frozen-cache}}
BASE=${3:-${WAM_BASE_MODEL:?set WAM_BASE_MODEL}}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m wam_policy.data download --snapshot "$SNAPSHOT"
python -m wam_policy.data inspect --snapshot "$SNAPSHOT"
python -m wam_policy.data fit --snapshot "$SNAPSHOT" --cache "$CACHE"
python -m wam_policy.data cache --snapshot "$SNAPSHOT" --cache "$CACHE" --base-model "$BASE" --device "${DEVICE:-cuda}"
python -m wam_policy.data inspect --snapshot "$SNAPSHOT" --cache "$CACHE"
