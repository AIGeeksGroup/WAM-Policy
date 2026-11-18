#!/usr/bin/env bash
# Standard LIBERO-Long (10 tasks x 50 official init states) of one policy on one visible GPU.
#   POLICY=base_1_2 OUT=$EVAL_ROOT/results/base bash scripts/eval/libero_long.sh
#   POLICY=student_1_2 STUDENT=<student dir> OUT=$EVAL_ROOT/results/opsd bash scripts/eval/libero_long.sh
# The optional argument caps the new episodes of this invocation (a smoke). A rerun resumes.
# Results: $OUT/libero_long/<policy>/<case>.json, summary.json, result.txt. About 7 h on an RTX 5090.
set -euo pipefail
max_episodes=${1:-}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${EVAL_ROOT:?EVAL_ROOT required}" "${OUT:?OUT required: results directory of this policy}"
policy=${POLICY:-base_1_2}
MODEL_PY=${MODEL_PY:-$EVAL_ROOT/envs/model/bin/python}
long_env=$EVAL_ROOT/envs/libero-cpu
resets=$EVAL_ROOT/resources/libero-long-resets/manifest.json
prompts=$EVAL_ROOT/libero-long/prompts.pt
student_args=()
if [[ "$policy" == student_1_2 ]]; then
    : "${STUDENT:?student_1_2 needs STUDENT (directory with arm_libero.json and student_adapter.pt)}"
    student_args=(--student "$STUDENT")
fi
for path in "$long_env/bin/python" "$long_env/worker-env.json" "$resets" "$prompts"; do
    [[ -e "$path" ]] || { echo "Missing prepared input: $path (run scripts/eval/setup.sh)" >&2; exit 2; }
done
args=(--policy "$policy" "${student_args[@]}" --model-root "$EVAL_ROOT/resources/lingbot-va-posttrain-libero-long"
      --prompts "$prompts" --reset-manifest "$resets" --cpu-python "$long_env/bin/python"
      --worker-env "$long_env/worker-env.json" --output "$OUT/libero_long")
[[ -n "$max_episodes" ]] && args+=(--max-episodes "$max_episodes")
PYTHONPATH="$root" "$MODEL_PY" -m arm_opsd.evaluation.cli evaluate-long "${args[@]}"
