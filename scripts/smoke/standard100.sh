#!/usr/bin/env bash
# Standard100 evaluation smoke / environment self-check: a random tiny teacher (release format) served by
# `wam_policy.inference.serve` and driven by one QLM-Bench client (`qlm_bench.benchmark run`) for ONE
# Standard100 episode. Checks the model env, the QLM env (MuJoCo/EGL), the real VAE and prompt embeddings,
# and the qlm-live-v1 protocol end to end. The outcome itself is meaningless (random policy).
#   scripts/smoke/standard100.sh [work_dir] [case_id]
# Needs configs/site.env: WAM_BASE_MODEL (vae/), WAM_PROMPTS, QLM_RESOURCES, QLM_PYTHON, QLM_BENCH_SRC.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
WORK=${1:-$(mktemp -d)}; CASE=${2:-push100-20261014-091}
PY=${PYTHON:-python}
DEVICE=${DEVICE:-$("$PY" -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m wam_policy.testing.tiny_teacher --output "$WORK/tiny_teacher"
"$PY" -m wam_policy.evaluation.standard100 --policy teacher --teacher "$WORK/tiny_teacher" \
  --base-model "$WAM_BASE_MODEL" --prompt-root "$WAM_PROMPTS" --device "$DEVICE" --vae-device "$DEVICE" \
  --resource-root "$QLM_RESOURCES" --qlm-python "$QLM_PYTHON" --qlm-src "$QLM_BENCH_SRC" \
  --seeds 42 --cases "$CASE" --port "${PORT:-8879}" --output "$WORK/eval"
"$PY" - "$WORK/eval/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
assert s["trials"] == 1 and s["complete"], s
print("Standard100 smoke passed: 1 episode with a terminal outcome", list(s["tasks"].values())[0]["outcomes"])
PYEOF
