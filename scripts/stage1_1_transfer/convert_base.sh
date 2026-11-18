#!/usr/bin/env bash
# Stage 1.1 initialization: LingBot-VA base transformer (30D action head) -> native 9D transformer.
#   scripts/stage1_1_transfer/convert_base.sh [base_dir] [output_transformer_dir]
# Defaults: base_dir=$WAM_BASE_MODEL, output=$SFT_TRANSFORMER (else $WAM_ROOT/stage1_1/native9-transformer).
# Downloads $WAM_BASE_MODEL_REPO@$WAM_BASE_MODEL_REVISION (paper: robbyant/lingbot-va-base@68b7bc1)
# into base_dir first if its transformer/ is missing. CPU is enough.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
BASE=${1:-${WAM_BASE_MODEL:?set WAM_BASE_MODEL or pass base_dir}}
OUT=${2:-${SFT_TRANSFORMER:-${WAM_ROOT:?set WAM_ROOT or pass output}/stage1_1/native9-transformer}}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -f "$BASE/transformer/config.json" ]; then
  huggingface-cli download "${WAM_BASE_MODEL_REPO:-robbyant/lingbot-va-base}" \
    --revision "${WAM_BASE_MODEL_REVISION:-68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c}" --local-dir "$BASE"
fi
python -m wam_policy.models.native9 --source "$BASE/transformer" --output "$OUT" --seed 0
