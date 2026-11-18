#!/usr/bin/env bash
# Accelerated Standard100 evaluation (resident multi-session service, VAE on the GPU, one QLM client
# per (case, seed), CLIENTS at a time). Usage:
#   scripts/evaluation/standard100_accelerated.sh <policy> <output> [adapter_dir] [extra args for wam_policy.evaluation.run_parallel]
#   policy: teacher | specialist (row-batched service) | opsd_student | mopd_student (multi-session service)
# Paths come from configs/site.env (WAM_WEIGHTS, WAM_BASE_MODEL, WAM_PROMPTS, QLM_RESOURCES, QLM_PYTHON, QLM_BENCH_SRC).
# Paper settings (overridable by environment): SEEDS=42,43,44 (300 episodes), CLIENTS=8, MAX_BATCH=8, MAX_WAIT_S=6
# (configs/evaluation/standard100_accelerated.env).
# Results differ from scripts/evaluation/standard100.sh only by bf16 rounding of batched GEMMs and the VAE device.
# The run resumes: finished (case, seed) runs under <output> are skipped.
# Examples:
#   scripts/evaluation/standard100_accelerated.sh teacher runs/eval/teacher-25-50-accelerated
#   scripts/evaluation/standard100_accelerated.sh specialist runs/eval/push $WAM_WEIGHTS/stage2_1_specialists/push_box
#     (all ten tasks x seeds 42/43/44 as in the paper; quick check: add --tasks push_box --seeds 42)
#   scripts/evaluation/standard100_accelerated.sh mopd_student runs/eval/mopd $WAM_WEIGHTS/stage2_2_mopd_student
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
_env=$(declare -p SEEDS CLIENTS MAX_BATCH MAX_WAIT_S PORT 2>/dev/null || true)
source "$ROOT/configs/evaluation/standard100_accelerated.env"
eval "$_env"   # values already set in the environment win over the config file
policy=$1; output=$2; adapter=${3:-}; shift $(( $# < 3 ? $# : 3 ))
args=()
[ -n "$adapter" ] && args+=(--adapter "$adapter")
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m wam_policy.evaluation.run_parallel --policy "$policy" \
  --teacher "$WAM_WEIGHTS/stage1_1_teacher" --base-model "$WAM_BASE_MODEL" --prompt-root "$WAM_PROMPTS" \
  --resource-root "$QLM_RESOURCES" --qlm-python "$QLM_PYTHON" --qlm-src "$QLM_BENCH_SRC" \
  --seeds "$SEEDS" --clients "$CLIENTS" --max-batch "$MAX_BATCH" --max-wait-s "$MAX_WAIT_S" --port "$PORT" \
  --output "$output" "${args[@]}" "$@"
