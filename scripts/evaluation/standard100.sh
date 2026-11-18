#!/usr/bin/env bash
# Default Standard100 evaluation (one service, one client per seed). Usage:
#   scripts/evaluation/standard100.sh <policy> <output> [adapter_dir] [extra args for wam_policy.evaluation.standard100]
#   policy: teacher | specialist | opsd_student | mopd_student
# Paths come from configs/site.env (WAM_WEIGHTS, WAM_BASE_MODEL, WAM_PROMPTS, QLM_RESOURCES, QLM_PYTHON, QLM_BENCH_SRC).
# Examples:
#   scripts/evaluation/standard100.sh teacher runs/eval/teacher-25-50
#   scripts/evaluation/standard100.sh teacher runs/eval/teacher-1-2 "" --video-steps 1 --action-steps 2
#   scripts/evaluation/standard100.sh opsd_student runs/eval/opsd $WAM_WEIGHTS/stage1_2_opsd_student
#   scripts/evaluation/standard100.sh specialist runs/eval/push $WAM_WEIGHTS/stage2_1_specialists/push_box
#     (the paper evaluates each specialist on all ten tasks, seeds 42/43/44; quick check: --cases <ids> --seeds 42)
#   scripts/evaluation/standard100.sh proprioception_teacher runs/eval/proprio "" --teacher <ablation teacher> --seeds 42
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
source "$ROOT/configs/evaluation/standard100_default.env"
policy=$1; output=$2; adapter=${3:-}; shift $(( $# < 3 ? $# : 3 ))
args=()
[ -n "$adapter" ] && args+=(--adapter "$adapter")
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m wam_policy.evaluation.standard100 --policy "$policy" \
  --teacher "$WAM_WEIGHTS/stage1_1_teacher" --base-model "$WAM_BASE_MODEL" --prompt-root "$WAM_PROMPTS" \
  --resource-root "$QLM_RESOURCES" --qlm-python "$QLM_PYTHON" --qlm-src "$QLM_BENCH_SRC" \
  --seeds "$SEEDS" --vae-device "$VAE_DEVICE" --offload-blocks "$OFFLOAD_BLOCKS" --port "$PORT" \
  --output "$output" "${args[@]}" "$@"
