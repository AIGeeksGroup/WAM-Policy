#!/usr/bin/env bash
# Stage 2.1 Task-Specific RL on one node: prepare the scenes -> start pool -> start split -> rounds -> export.
# Every step resumes from what is on disk; rerun the same command after an interruption.
#
#   TASK=push_box|open_room_door RUN=<run dir> QLM_BENCH=<QLM-Bench checkout> QLM_PYTHON=<QLM env python> \
#   TEACHER=<Stage 1.1 teacher dir> BASE_MODEL=<lingbot-va-base dir> PROMPT_ROOT=<prompt_embeddings dir> \
#   [CONFIG=configs/stage2_1_task_rl/$TASK.json] [GPUS=1] [CORES_PER_GPU=0] [FIRST=0] [LAST=<max_rounds - 1>] \
#   [DATA_ROOT=$RUN/../data] [DATA_REUSE=<existing tree with the scene files>] [GL=osmesa|egl] [OSMESA_LIB=<dir>] \
#   [DEADLINE=<epoch seconds>] [EXPORT_ROUND=rNN] scripts/stage2_1_task_rl/run_task_rl.sh
#
# The paper runs: push_box (configs/stage2_1_task_rl/push_box.json, specialist = r14) and open_room_door
# (open_room_door.json, specialist = r04, chosen by the in-training evaluation), each on 8 GPUs (GPUS=8, OSMesa).
# Optional: WANDB_MODE=online|offline (+ WANDB_ENTITY / WANDB_PROJECT) mirrors the round table to W&B at the end.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO/configs/site.env" ] && source "$REPO/configs/site.env"
TEACHER="${TEACHER:-${WAM_TEACHER:-}}"; BASE_MODEL="${BASE_MODEL:-${WAM_BASE_MODEL:-}}"
PROMPT_ROOT="${PROMPT_ROOT:-${WAM_PROMPTS:-}}"; OSMESA_LIB="${OSMESA_LIB:-${QLM_OSMESA_LIB:-}}"
[ -z "${QLM_BENCH:-}" ] && [ -n "${QLM_BENCH_SRC:-}" ] && QLM_BENCH="$(dirname "$QLM_BENCH_SRC")"
: "${TASK:?}" "${RUN:?}" "${QLM_BENCH:?}" "${QLM_PYTHON:?}" "${TEACHER:?}" "${BASE_MODEL:?}" "${PROMPT_ROOT:?}"
PY="${POLICY_PYTHON:-python}"
CONFIG="${CONFIG:-$REPO/configs/stage2_1_task_rl/$TASK.json}"
GPUS="${GPUS:-1}"; CORES="${CORES_PER_GPU:-0}"; GL="${GL:-osmesa}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$RUN")/data}"
export PYTHONPATH="$REPO/src:$QLM_BENCH/src${PYTHONPATH:+:$PYTHONPATH}"
T() { "$PY" -m wam_policy.task_rl "$@"; }

CASES="$DATA_ROOT/cases-$TASK"
if [ ! -e "$CASES/cases-index.json" ]; then
  if [ -n "${DATA_REUSE:-}" ]; then FETCH=(--reuse "$DATA_REUSE"); else FETCH=(--download); fi
  T prepare --task "$TASK" --inputs "$QLM_BENCH/inputs/task_rl/$TASK" --data-root "$DATA_ROOT" --out "$CASES" "${FETCH[@]}"
fi
T init --run "$RUN" --task "$TASK" --config "$CONFIG" --teacher "$TEACHER" --base-model "$BASE_MODEL" \
  --prompt-root "$PROMPT_ROOT" --cases "$CASES" --qlm-python "$QLM_PYTHON" --qlm-src "$QLM_BENCH/src" \
  --gl "$GL" ${OSMESA_LIB:+--osmesa-lib "$OSMESA_LIB"}
[ -e "$RUN/pool/meta.json" ] || T pool --run "$RUN" --gpus "$GPUS" --cores-per-gpu "$CORES"
[ -e "$RUN/pool/selection.json" ] || T select --run "$RUN"
LAST="${LAST:-$(( $("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['config']['max_rounds'])" "$RUN/run.json") - 1 ))}"
T rounds --run "$RUN" "${FIRST:-0}" "$LAST" --gpus "$GPUS" --cores-per-gpu "$CORES" ${DEADLINE:+--deadline "$DEADLINE"}
T track --run "$RUN" > "$RUN/logs/track.log" 2>&1 || true
if [ -n "${EXPORT_ROUND:-}" ]; then
  T export --run "$RUN" --round "$EXPORT_ROUND" --output "$RUN/specialist-$EXPORT_ROUND"
fi
cat "$RUN/results/round-table.md"
