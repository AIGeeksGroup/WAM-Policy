#!/usr/bin/env bash
# Start the WAM-Policy server for one task on this machine, for the robot bridge:
#
#   scripts/wam/policy_server.sh <task> [extra wam_policy.inference.serve arguments]
#
# Runs this repository's `python -m wam_policy.inference.serve` unchanged, listening on
# 127.0.0.1:$WAM_PORT (qlm-live-v1), with only the task's cached prompt embedding loaded.
# Model paths come from configs/site.env (WAM_TEACHER, WAM_WEIGHTS, OPSD_STUDENT, WAM_BASE_MODEL,
# WAM_PROMPTS, optionally QLM_BENCH_SRC), robot-side settings from deploy/config/deploy.env. The model
# environment needs this repository's policy dependencies and the QLM-Bench live package
# (pip install -e "<QLM-Bench>[live]").
#
# Env: WAM_POLICY      opsd_student (default) | mopd_student | teacher | specialist
#      WAM_ADAPTER     adapter directory (default: the released one of WAM_POLICY)
#      WAM_PORT        8862
#      WAM_POLICY_PYTHON  Python of the model environment (default python3)
#      WAM_SERVER_ARGS extra serve arguments, e.g. "--vae-device cpu --offload-blocks 8"
#      WAM_SERVER_LOG_DIR  passed as --log-dir (request and event logs)
set -euo pipefail
# shellcheck source=../load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../load_config.sh"

die() { echo "policy_server: $*" >&2; exit 1; }
[[ $# -ge 1 && "$1" != -* ]] || { awk 'NR > 1 { if (!/^#/) exit; print substr($0, 3) }' "${BASH_SOURCE[0]}"; exit 2; }
TASK="$1"; shift

WEIGHTS="${WAM_WEIGHTS:-}"
TEACHER="${WAM_TEACHER:-${WEIGHTS:+$WEIGHTS/stage1_1_teacher}}"
PROMPTS="${WAM_PROMPTS:-${WEIGHTS:+$WEIGHTS/prompt_embeddings}}"
case "$WAM_POLICY" in
  opsd_student) ADAPTER="${WAM_ADAPTER:-${OPSD_STUDENT:-${WEIGHTS:+$WEIGHTS/stage1_2_opsd_student}}}" ;;
  mopd_student) ADAPTER="${WAM_ADAPTER:-${WEIGHTS:+$WEIGHTS/stage2_2_mopd_student}}" ;;
  specialist)   site_var="MOPD_$(printf '%s' "$TASK" | tr '[:lower:]' '[:upper:]')_SPECIALIST"   # configs/site.env
                ADAPTER="${WAM_ADAPTER:-${!site_var:-${WEIGHTS:+$WEIGHTS/stage2_1_specialists/$TASK}}}" ;;
  teacher)      ADAPTER="" ;;
  *) die "WAM_POLICY=$WAM_POLICY: must be opsd_student, mopd_student, teacher or specialist" ;;
esac
[[ -n "$TEACHER" ]] || die "WAM_TEACHER (or WAM_WEIGHTS) is not set: see configs/site.env"
[[ -n "$PROMPTS" ]] || die "WAM_PROMPTS (or WAM_WEIGHTS) is not set: see configs/site.env"
[[ -n "${WAM_BASE_MODEL:-}" ]] || die "WAM_BASE_MODEL is not set: see configs/site.env"
[[ "$WAM_POLICY" == teacher || -n "$ADAPTER" ]] || die "no adapter for $WAM_POLICY: set WAM_WEIGHTS or WAM_ADAPTER"
[[ -f "$WAM_REPO_DIR/src/wam_policy/inference/serve.py" ]] \
  || die "wam_policy.inference.serve not found under $WAM_REPO_DIR/src (deploy/ must sit in the WAM-Policy repository)"

PYTHONPATH="$WAM_DEPLOY_DIR" python3 -m wam_go2.wam.model_check --policy "$WAM_POLICY" --teacher "$TEACHER" \
  ${ADAPTER:+--adapter "$ADAPTER"} --prompt-root "$PROMPTS" --task "$TASK" >&2 \
  || die "the model directories do not fit this run (see above)"

args=(--policy "$WAM_POLICY" --teacher "$TEACHER" --base-model "$WAM_BASE_MODEL" --prompt-root "$PROMPTS"
      --tasks "$TASK" --host 127.0.0.1 --port "$WAM_PORT")
[[ -z "$ADAPTER" ]] || args+=(--adapter "$ADAPTER")
[[ -z "${WAM_SERVER_LOG_DIR:-}" ]] || args+=(--log-dir "$WAM_SERVER_LOG_DIR")
extra=()
if [[ -n "$WAM_SERVER_ARGS" ]]; then set -f; read -r -a extra <<<"$WAM_SERVER_ARGS"; set +f; fi

QLM_SRC="${QLM_BENCH_SRC:-}"
[[ -n "$QLM_SRC" && -d "$QLM_SRC/qlm_bench" ]] || QLM_SRC=""
export PYTHONPATH="$WAM_REPO_DIR/src${QLM_SRC:+:$QLM_SRC}${PYTHONPATH:+:$PYTHONPATH}"
"$WAM_POLICY_PYTHON" -c 'import qlm_bench.live' >/dev/null 2>&1 \
  || die "the QLM-Bench live package is not importable by $WAM_POLICY_PYTHON: pip install -e \"<QLM-Bench>[live]\" there"
echo "policy_server: $WAM_POLICY for $TASK on 127.0.0.1:$WAM_PORT (teacher $TEACHER${ADAPTER:+, adapter $ADAPTER})"
exec "$WAM_POLICY_PYTHON" -m wam_policy.inference.serve "${args[@]}" "${extra[@]}" "$@"
