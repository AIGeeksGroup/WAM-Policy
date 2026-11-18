#!/usr/bin/env bash
# Stage 1.1 smoke / environment self-check (CPU, tiny model, synthetic data; about a minute):
# native9 conversion of a tiny 30D transformer -> 2-rank (gloo) SFT that stops at step 2 ->
# resume to step 4 -> teacher export -> load_teacher.
#   scripts/smoke/stage1_1_sft.sh [work_dir]     (PYTHON=... selects the interpreter)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-$(mktemp -d)}
PY=${PYTHON:-python}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=2
CONFIG="$ROOT/configs/stage1_1_transfer/tiny.json"
TRAIN=(-m torch.distributed.run --standalone --nproc_per_node 2 -m wam_policy.transfer train --config "$CONFIG"
       --cache-root "$WORK/cache" --transformer "$WORK/transformer" --output-dir "$WORK/run")
echo "== synthetic inputs in $WORK"
"$PY" -m wam_policy.testing.synthetic --output "$WORK"
echo "== native9 conversion"
"$PY" -m wam_policy.models.native9 --source "$WORK/source/transformer" --output "$WORK/transformer" --seed 0
echo "== train to step 2"
"$PY" "${TRAIN[@]}" --stop-after 2
echo "== resume to step 4"
"$PY" "${TRAIN[@]}" --resume "$WORK/run/checkpoints/step_00000002"
echo "== export + load_teacher"
"$PY" -m wam_policy.transfer export --checkpoint "$WORK/run/checkpoints/step_00000004" --output "$WORK/teacher"
"$PY" - "$WORK/teacher" <<'PYEOF'
import sys
from wam_policy.inference.loader import load_teacher
model, normalizer, meta = load_teacher(sys.argv[1], device="cpu", dtype="float32", attn_mode="torch")
print("load_teacher ok:", meta["kind"], meta["sampling"], "action_dim", model.config.action_dim,
      "params", sum(p.numel() for p in model.parameters()))
PYEOF
tail -n 2 "$WORK/run/metrics.jsonl"
echo "stage1.1 smoke passed ($WORK)"
