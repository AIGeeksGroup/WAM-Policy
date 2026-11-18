#!/usr/bin/env bash
# Stage 1.1 teacher SFT on one node (CUDA or ROCm). Paper: NPROC=8 (8x MI355X), 5000 steps.
#   scripts/stage1_1_transfer/train.sh [run_dir] [extra args, e.g. --resume latest]   (run_dir is optional)
# Inputs: $WAM_ROOT/stage1_1/{frozen-cache,native9-transformer} (prepare_data.sh, convert_base.sh)
# and $WAM_BASE_MODEL (VAE for the optional prediction videos). With NPROC < 8 add
# --accumulation $((8 / NPROC)) to keep the global batch of 8.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then RUN=$1; shift; else RUN=${WAM_ROOT:?set WAM_ROOT}/stage1_1/run; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun --standalone --nproc_per_node "${NPROC:-8}" -m wam_policy.transfer train \
  --config "${CONFIG:-$ROOT/configs/stage1_1_transfer/teacher_sft.json}" \
  --cache-root "${SFT_CACHE:-$WAM_ROOT/stage1_1/frozen-cache}" \
  --transformer "${SFT_TRANSFORMER:-$WAM_ROOT/stage1_1/native9-transformer}" \
  --base-model "${WAM_BASE_MODEL:?set WAM_BASE_MODEL}" --output-dir "$RUN" "$@"
