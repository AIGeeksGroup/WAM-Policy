#!/usr/bin/env bash
# LIBERO-Pro2.0 (2,520 episodes) of one policy on one visible GPU, then its official group rates.
#   POLICY=base_1_2 OUT=$EVAL_ROOT/results/base bash scripts/eval/libero_pro2.sh
#   POLICY=student_1_2 STUDENT=<student dir> OUT=$EVAL_ROOT/results/opsd bash scripts/eval/libero_pro2.sh
# The optional argument caps the new episodes of this invocation (a smoke). A rerun resumes the same run.
# Results: $OUT/libero_pro2/<policy>/<episode>.json, $OUT/libero_pro2/official-summary.{json,txt}.
# About 14 h on an RTX 5090 for 2,520 episodes. Never prints server.env/client.env.
set -euo pipefail
max_episodes=${1:-}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/scripts/eval/protocol.sh"
: "${EVAL_ROOT:?EVAL_ROOT required}" "${OUT:?OUT required: results directory of this policy}"
policy=${POLICY:-base_1_2}
MODEL_PY=${MODEL_PY:-$EVAL_ROOT/envs/model/bin/python}
pro=$EVAL_ROOT/resources/libero-pro2.0-eval/instance/LIBERO-Pro-evaluation
pro_py=$EVAL_ROOT/envs/libero-pro2/bin/python
plan_dir=$EVAL_ROOT/libero-pro2/plan
out=$OUT/libero_pro2
student_args=()
if [[ "$policy" == student_1_2 ]]; then
    : "${STUDENT:?student_1_2 needs STUDENT (directory with arm_libero.json and student_adapter.pt)}"
    student_args=(--student "$STUDENT")
fi
for path in "$pro/server.env" "$pro/client.env" "$pro_py" "$plan_dir/plan.json" "$plan_dir/instructions.json" "$plan_dir/prompts.pt"; do
    [[ -e "$path" ]] || { echo "Missing prepared input: $path (run scripts/eval/setup.sh)" >&2; exit 2; }
done
gpus=$("$MODEL_PY" -I -c 'import torch; print(torch.cuda.device_count())')
[[ "$gpus" == 1 ]] || { echo "Expected exactly one visible GPU (CUDA_VISIBLE_DEVICES), found $gpus" >&2; exit 2; }
# One server process has hit "Too many open files" after ~500 episodes under the default soft limit.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
mkdir -p "$out/logs"
bash "$pro/stop.sh"
rmdir "$pro/runtime/start.lock" 2>/dev/null || true
trap 'bash "$pro/stop.sh"' EXIT
bash "$pro/start.sh"
bash "$pro/healthcheck.sh"
submit=$out/submit
if [[ ! -f "$submit/submission.json" ]]; then
    (set -a; source "$pro/client.env"; set +a
     "$pro_py" "$pro/service-source/scripts/run_grouped_eval.py" --base-url "$LIBERO_REMOTE_URL" submit \
       "${PLAN_ARGS[@]}" --output-dir "$submit")
fi
run_id=$("$MODEL_PY" -I - "$plan_dir/plan.json" "$submit/plan.json" "$submit/submission.json" <<'PY'
import json, sys
shared, submitted, submission = (json.load(open(p)) for p in sys.argv[1:])
if shared["plan_id"] != submitted["plan_id"]:
    raise SystemExit(f"submitted plan {submitted['plan_id']} differs from the prepared plan {shared['plan_id']}")
print(submission["run_id"])
PY
)
args=(--policy "$policy" "${student_args[@]}" --model-root "$EVAL_ROOT/resources/lingbot-va-posttrain-libero-long"
      --prompts "$plan_dir/prompts.pt" --plan "$plan_dir/plan.json" --instructions "$plan_dir/instructions.json"
      --run-id "$run_id" --pro-python "$pro_py" --output "$out")
[[ -n "$max_episodes" ]] && args+=(--max-episodes "$max_episodes")
(set -a; source "$pro/client.env"; set +a
 PYTHONPATH="$root" "$MODEL_PY" -m arm_opsd.evaluation.cli evaluate "${args[@]}")
if [[ -z "$max_episodes" && ! -f "$out/official-summary.json" ]]; then
    PYTHONPATH="$root" "$MODEL_PY" -m arm_opsd.evaluation.cli official-summary --results "$out/$policy" \
        --plan "$plan_dir/plan.json" --output "$out/official-summary.json"
fi
