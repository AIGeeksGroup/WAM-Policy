#!/usr/bin/env bash
#SBATCH --job-name=wam-v2-full-pipeline-admin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-wam-v2-full-pipeline-admin-%j.out
#SBATCH --error=slurm-wam-v2-full-pipeline-admin-%j.err

set -euo pipefail

# Admin submission example:
#   export REPO_ROOT=/shared/amdgpu/home/<user>/WAM-Policy
#   export TEACHER_CKPT=$REPO_ROOT/experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/transformer
#   export VIDEO_DEBUG_MANIFEST=$REPO_ROOT/datasets/manifests/video_latents_debug.jsonl
#   export VIDEO_FULL_MANIFEST=$REPO_ROOT/datasets/manifests/video_latents_full.jsonl
#   export ACTION_DEBUG_MANIFEST=$REPO_ROOT/datasets/manifests/action_chunks_debug.jsonl
#   export ACTION_FULL_MANIFEST=$REPO_ROOT/datasets/manifests/action_chunks_full.jsonl
#   export ROLLOUT_DEBUG_MANIFEST=$REPO_ROOT/datasets/manifests/student_rollouts_debug.jsonl
#   export ROLLOUT_FULL_MANIFEST=$REPO_ROOT/datasets/manifests/student_rollouts_full.jsonl
#   export DMD_TEACHER_SCORE_SOURCE=cached_label
#   sbatch --export=ALL scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch
#
# DMD_TEACHER_SCORE_SOURCE defaults to not_implemented. Set it to cached_label
# when rollout manifests contain teacher labels, or lingbot_velocity_forward only
# after the server-side LingBot teacher score binding probe succeeds.

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT is required." >&2
  exit 2
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VIDEO_FFM_CONFIG="${VIDEO_FFM_CONFIG:-configs/experiments/v2/anyflow_video_ffm.yaml}"
ACTION_FFM_CONFIG="${ACTION_FFM_CONFIG:-configs/experiments/v2/anyflow_action_ffm.yaml}"
VIDEO_DMD_CONFIG="${VIDEO_DMD_CONFIG:-configs/experiments/v2/anyflow_video_dmd_opd.yaml}"
ACTION_DMD_CONFIG="${ACTION_DMD_CONFIG:-configs/experiments/v2/anyflow_action_dmd_opd.yaml}"

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$REPO_ROOT/logs/v2_full_pipeline_admin/$JOB_ID"
STATUS_FILE="$LOG_DIR/status.txt"
RUN_BASE="$REPO_ROOT/experiments/runs/v2"
CHECKPOINT_BASE="$REPO_ROOT/experiments/checkpoints/v2"
SLURM_OUTPUT_DIR="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
SLURM_OUT_FILE="$SLURM_OUTPUT_DIR/slurm-wam-v2-full-pipeline-admin-${JOB_ID}.out"
SLURM_ERR_FILE="$SLURM_OUTPUT_DIR/slurm-wam-v2-full-pipeline-admin-${JOB_ID}.err"

VIDEO_DEBUG_MANIFEST="${VIDEO_DEBUG_MANIFEST:-${DEBUG_MANIFEST:-}}"
VIDEO_FULL_MANIFEST="${VIDEO_FULL_MANIFEST:-${FULL_MANIFEST:-}}"
ACTION_DEBUG_MANIFEST="${ACTION_DEBUG_MANIFEST:-}"
ACTION_FULL_MANIFEST="${ACTION_FULL_MANIFEST:-}"
ROLLOUT_DEBUG_MANIFEST="${ROLLOUT_DEBUG_MANIFEST:-}"
ROLLOUT_FULL_MANIFEST="${ROLLOUT_FULL_MANIFEST:-}"
TEACHER_CKPT="${TEACHER_CKPT:-}"

TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-5000}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_SMOKE_MAX_STEPS="${TRAIN_SMOKE_MAX_STEPS:-2}"
TRAIN_SMOKE_SAVE_EVERY="${TRAIN_SMOKE_SAVE_EVERY:-1}"
TRAIN_SMOKE_LOG_EVERY="${TRAIN_SMOKE_LOG_EVERY:-1}"
TRAIN_SMOKE_BATCH_SIZE="${TRAIN_SMOKE_BATCH_SIZE:-1}"

DMD_TEACHER_SCORE_SOURCE="${DMD_TEACHER_SCORE_SOURCE:-not_implemented}"
DMD_SURROGATE_STATE="${DMD_SURROGATE_STATE:-pred_u}"
DMD_STUDENT_SCORE_SOURCE="${DMD_STUDENT_SCORE_SOURCE:-flowmap_small_interval}"
DMD_SMALL_DELTA_S="${DMD_SMALL_DELTA_S:-0.02}"

WANDB_ENABLED="${WANDB_ENABLED:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-WAM-Policy}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-v2-full-pipeline-${JOB_ID}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-wam-v2-full-pipeline-${JOB_ID}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-wandb}"
WANDB_UPLOAD_LOGS="${WANDB_UPLOAD_LOGS:-true}"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

stage_run_dir() {
  printf '%s/%s_admin/%s\n' "$RUN_BASE" "$1" "$JOB_ID"
}

stage_ckpt_dir() {
  printf '%s/%s_admin/%s\n' "$CHECKPOINT_BASE" "$1" "$JOB_ID"
}

for stage_name in video_ffm action_ffm video_dmd action_dmd; do
  mkdir -p "$(stage_run_dir "$stage_name")" "$(stage_ckpt_dir "$stage_name")"
done

mark_failed() {
  local stage="$1"
  local exit_code="$2"
  echo "FAILED:${stage}:${exit_code}" | tee -a "$STATUS_FILE"
  echo "First failing stage: ${stage}" | tee -a "$STATUS_FILE"
  echo "See log: ${LOG_DIR}/${stage}.log" | tee -a "$STATUS_FILE"
  exit "$exit_code"
}

run_stage() {
  local stage="$1"
  shift
  local log_file="$LOG_DIR/${stage}.log"

  echo "START:${stage}:$(date -Is)" | tee -a "$STATUS_FILE"
  {
    echo "stage: ${stage}"
    echo "started_at: $(date -Is)"
    echo "workdir: $(pwd)"
    echo "command: $*"
    echo
    set +e
    "$@"
    local command_status=$?
    set -e
    echo
    echo "finished_at: $(date -Is)"
    echo "exit_code: ${command_status}"
    exit "$command_status"
  } >"$log_file" 2>&1 || mark_failed "$stage" "$?"

  echo "PASSED:${stage}:$(date -Is)" | tee -a "$STATUS_FILE"
}

run_shell_stage() {
  local stage="$1"
  local script="$2"
  run_stage "$stage" bash -lc "$script"
}

echo "STARTED:${JOB_ID}:$(date -Is)" | tee "$STATUS_FILE"
echo "repo_root: $REPO_ROOT" | tee -a "$STATUS_FILE"
echo "log_dir: $LOG_DIR" | tee -a "$STATUS_FILE"
echo "checkpoint_base: $CHECKPOINT_BASE" | tee -a "$STATUS_FILE"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SETUP"
fi

COMMON_OVERRIDES=(
  --set runtime.device=rocm
  --set runtime.lingbot_attn_mode=torch
  --set "teacher.checkpoint_path=${TEACHER_CKPT}"
)

DMD_OVERRIDES=(
  --set "opd.dmd.teacher_score_source=${DMD_TEACHER_SCORE_SOURCE}"
  --set "opd.dmd.surrogate_state=${DMD_SURROGATE_STATE}"
  --set "opd.dmd.student_score_source=${DMD_STUDENT_SCORE_SOURCE}"
  --set "opd.dmd.small_delta_s=${DMD_SMALL_DELTA_S}"
)

WANDB_OVERRIDES=()
WANDB_EFFECTIVE_ENABLED=false
case "$WANDB_ENABLED" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    WANDB_EFFECTIVE_ENABLED=true
    ;;
  auto|Auto|AUTO)
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      WANDB_EFFECTIVE_ENABLED=true
    fi
    ;;
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    WANDB_OVERRIDES=(--set logging.wandb.enabled=false)
    ;;
esac

if [[ "$WANDB_EFFECTIVE_ENABLED" == "true" ]]; then
  WANDB_OVERRIDES=(
    --set logging.wandb.enabled=true
    --set "logging.wandb.project=${WANDB_PROJECT}"
    --set "logging.wandb.name=${WANDB_RUN_NAME}"
    --set "logging.wandb.id=${WANDB_RUN_ID}"
    --set "logging.wandb.resume=${WANDB_RESUME}"
    --set "logging.wandb.mode=${WANDB_MODE}"
    --set "logging.wandb.dir=${WANDB_DIR}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_OVERRIDES+=(--set "logging.wandb.entity=${WANDB_ENTITY}")
  fi
fi

upload_wandb_logs() {
  local exit_code="$1"
  if [[ "$WANDB_EFFECTIVE_ENABLED" != "true" ]]; then
    return 0
  fi
  case "$WANDB_UPLOAD_LOGS" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
      return 0
      ;;
  esac

  set +e
  local upload_args=(
    scripts/tools/upload_admin_logs_to_wandb.py
    --job-id "$JOB_ID"
    --log-dir "$LOG_DIR"
    --checkpoint-dir "$CHECKPOINT_BASE"
    --project "$WANDB_PROJECT"
    --name "$WANDB_RUN_NAME"
    --run-id "$WANDB_RUN_ID"
    --resume "$WANDB_RESUME"
    --mode "$WANDB_MODE"
    --dir "$WANDB_DIR"
    --exit-code "$exit_code"
    --slurm-out "$SLURM_OUT_FILE"
    --slurm-err "$SLURM_ERR_FILE"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    upload_args+=(--entity "$WANDB_ENTITY")
  fi
  "$PYTHON_BIN" "${upload_args[@]}"
  local upload_status=$?
  set -e
  if [[ "$upload_status" -ne 0 ]]; then
    echo "WARN: failed to upload admin logs to wandb; upload_exit_code=${upload_status}" | tee -a "$STATUS_FILE"
  fi
  return 0
}

on_exit() {
  local exit_code="$1"
  trap - EXIT
  upload_wandb_logs "$exit_code" || true
  exit "$exit_code"
}

trap 'on_exit $?' EXIT

run_train_stage() {
  local stage="$1"
  local config="$2"
  local manifest_key="$3"
  local manifest="$4"
  local run_dir="$5"
  local ckpt_dir="$6"
  local max_steps="$7"
  local save_every="$8"
  local log_every="$9"
  local batch_size="${10}"
  shift 10

  run_stage "$stage" \
    "$PYTHON_BIN" scripts/train.py \
    --config "$config" \
    "${COMMON_OVERRIDES[@]}" \
    "${WANDB_OVERRIDES[@]}" \
    --set "${manifest_key}=${manifest}" \
    --set "paths.output_dir=${run_dir}" \
    --set "paths.checkpoint_dir=${ckpt_dir}" \
    --set "data.batch_size=${batch_size}" \
    --set data.num_workers=0 \
    --set "train.max_steps=${max_steps}" \
    --set "train.save_every=${save_every}" \
    --set "train.log_every=${log_every}" \
    "$@"
}

run_shell_stage "00_env" "
set -euo pipefail
echo hostname: \$(hostname)
echo date: \$(date -Is)
echo slurm_job_id: ${SLURM_JOB_ID:-}
echo slurm_nodelist: ${SLURM_NODELIST:-}
echo repo_root: ${REPO_ROOT@Q}
echo teacher_ckpt: ${TEACHER_CKPT@Q}
echo video_debug_manifest: ${VIDEO_DEBUG_MANIFEST@Q}
echo video_full_manifest: ${VIDEO_FULL_MANIFEST@Q}
echo action_debug_manifest: ${ACTION_DEBUG_MANIFEST@Q}
echo action_full_manifest: ${ACTION_FULL_MANIFEST@Q}
echo rollout_debug_manifest: ${ROLLOUT_DEBUG_MANIFEST@Q}
echo rollout_full_manifest: ${ROLLOUT_FULL_MANIFEST@Q}
echo dmd_teacher_score_source: ${DMD_TEACHER_SCORE_SOURCE@Q}
echo dmd_surrogate_state: ${DMD_SURROGATE_STATE@Q}
echo dmd_student_score_source: ${DMD_STUDENT_SCORE_SOURCE@Q}
echo dmd_small_delta_s: ${DMD_SMALL_DELTA_S@Q}
echo train_max_steps: ${TRAIN_MAX_STEPS@Q}
echo train_smoke_max_steps: ${TRAIN_SMOKE_MAX_STEPS@Q}
echo wandb_enabled: ${WANDB_ENABLED@Q}
echo wandb_effective_enabled: ${WANDB_EFFECTIVE_ENABLED@Q}
echo wandb_project: ${WANDB_PROJECT@Q}
echo wandb_entity: ${WANDB_ENTITY:-}
echo wandb_run_name: ${WANDB_RUN_NAME@Q}
echo wandb_run_id: ${WANDB_RUN_ID@Q}
echo python_bin: ${PYTHON_BIN@Q}
echo
test -d ${REPO_ROOT@Q}
test -f ${VIDEO_FFM_CONFIG@Q}
test -f ${ACTION_FFM_CONFIG@Q}
test -f ${VIDEO_DMD_CONFIG@Q}
test -f ${ACTION_DMD_CONFIG@Q}
test -e ${TEACHER_CKPT@Q}
test -f ${VIDEO_DEBUG_MANIFEST@Q}
test -f ${VIDEO_FULL_MANIFEST@Q}
test -f ${ACTION_DEBUG_MANIFEST@Q}
test -f ${ACTION_FULL_MANIFEST@Q}
test -f ${ROLLOUT_DEBUG_MANIFEST@Q}
test -f ${ROLLOUT_FULL_MANIFEST@Q}
command -v ${PYTHON_BIN@Q}
${PYTHON_BIN@Q} - <<'PY'
import torch
print('torch.__version__:', torch.__version__)
print('torch.version.hip:', getattr(torch.version, 'hip', None))
print('torch.version.cuda:', getattr(torch.version, 'cuda', None))
print('torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch.cuda.device_count:', torch.cuda.device_count())
    print('torch.cuda.get_device_name:', torch.cuda.get_device_name(0))
PY
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi
else
  echo 'rocm-smi not found'
fi
echo
env | sort | sed -E 's/^(.*(TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_KEY).*=).*/\1<redacted>/'
"

run_shell_stage "01_static_all" "
set -euo pipefail
${PYTHON_BIN@Q} scripts/train.py --config ${VIDEO_FFM_CONFIG@Q} --dry-run static
${PYTHON_BIN@Q} scripts/train.py --config ${ACTION_FFM_CONFIG@Q} --dry-run static
${PYTHON_BIN@Q} scripts/train.py --config ${VIDEO_DMD_CONFIG@Q} --dry-run static
${PYTHON_BIN@Q} scripts/train.py --config ${ACTION_DMD_CONFIG@Q} --dry-run static
${PYTHON_BIN@Q} scripts/tools/probe_video_ffm_runtime.py probe-static-source --config ${VIDEO_FFM_CONFIG@Q}
${PYTHON_BIN@Q} scripts/tools/probe_action_ffm_runtime.py probe-static-source --config ${ACTION_FFM_CONFIG@Q}
${PYTHON_BIN@Q} scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config ${VIDEO_DMD_CONFIG@Q}
${PYTHON_BIN@Q} scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config ${ACTION_DMD_CONFIG@Q}
${PYTHON_BIN@Q} -m py_compile \
  scripts/train.py \
  scripts/tools/probe_video_ffm_runtime.py \
  scripts/tools/probe_action_ffm_runtime.py \
  scripts/tools/probe_dmd_opd_runtime.py \
  scripts/tools/upload_admin_logs_to_wandb.py \
  src/wam_policy/adapters/anyflow.py \
  src/wam_policy/adapters/lingbot_va.py \
  src/wam_policy/adapters/robotwin.py \
  src/wam_policy/data/action_dataset.py \
  src/wam_policy/data/rollout_manifest_dataset.py \
  src/wam_policy/opd/losses.py \
  src/wam_policy/opd/stages.py \
  src/wam_policy/opd/trainer.py
"

run_stage "02_import" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-import \
  --config "$VIDEO_FFM_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  --set "data.manifest=${VIDEO_DEBUG_MANIFEST}"

run_stage "03_load_model" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-load-model \
  --config "$VIDEO_FFM_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  --set "data.manifest=${VIDEO_DEBUG_MANIFEST}"

run_stage "04_video_manifest_debug" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-manifest-one-sample \
  --config "$VIDEO_FFM_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  --set "data.manifest=${VIDEO_DEBUG_MANIFEST}"

run_stage "05_video_forward_debug" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-video-forward \
  --config "$VIDEO_FFM_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  --set "data.manifest=${VIDEO_DEBUG_MANIFEST}"

run_stage "06_action_manifest_debug" \
  "$PYTHON_BIN" scripts/tools/probe_action_ffm_runtime.py \
  probe-manifest-one-sample \
  --config "$ACTION_FFM_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  --set "data.manifest=${ACTION_DEBUG_MANIFEST}"

run_stage "07_action_forward_debug" \
  "$PYTHON_BIN" scripts/tools/probe_action_ffm_runtime.py \
  probe-action-forward \
  --config "$ACTION_FFM_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  --set "data.manifest=${ACTION_DEBUG_MANIFEST}"

run_shell_stage "08_rollout_manifest_debug" "
set -euo pipefail
${PYTHON_BIN@Q} scripts/tools/probe_dmd_opd_runtime.py probe-rollout-manifest \
  --config ${VIDEO_DMD_CONFIG@Q} \
  --require-external-exists \
  --set runtime.device=rocm \
  --set runtime.lingbot_attn_mode=torch \
  --set teacher.checkpoint_path=${TEACHER_CKPT@Q} \
  --set opd.dmd.teacher_score_source=${DMD_TEACHER_SCORE_SOURCE@Q} \
  --set opd.dmd.surrogate_state=${DMD_SURROGATE_STATE@Q} \
  --set opd.dmd.student_score_source=${DMD_STUDENT_SCORE_SOURCE@Q} \
  --set opd.dmd.small_delta_s=${DMD_SMALL_DELTA_S@Q} \
  --set opd.rollout_manifest=${ROLLOUT_DEBUG_MANIFEST@Q}
${PYTHON_BIN@Q} scripts/tools/probe_dmd_opd_runtime.py probe-rollout-manifest \
  --config ${ACTION_DMD_CONFIG@Q} \
  --require-external-exists \
  --set runtime.device=rocm \
  --set runtime.lingbot_attn_mode=torch \
  --set teacher.checkpoint_path=${TEACHER_CKPT@Q} \
  --set opd.dmd.teacher_score_source=${DMD_TEACHER_SCORE_SOURCE@Q} \
  --set opd.dmd.surrogate_state=${DMD_SURROGATE_STATE@Q} \
  --set opd.dmd.student_score_source=${DMD_STUDENT_SCORE_SOURCE@Q} \
  --set opd.dmd.small_delta_s=${DMD_SMALL_DELTA_S@Q} \
  --set opd.rollout_manifest=${ROLLOUT_DEBUG_MANIFEST@Q}
"

run_stage "09_video_dmd_forward_debug" \
  "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py \
  probe-dmd-forward \
  --config "$VIDEO_DMD_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  "${DMD_OVERRIDES[@]}" \
  --set "opd.rollout_manifest=${ROLLOUT_DEBUG_MANIFEST}"

run_stage "10_action_dmd_forward_debug" \
  "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py \
  probe-dmd-forward \
  --config "$ACTION_DMD_CONFIG" \
  --require-external-exists \
  "${COMMON_OVERRIDES[@]}" \
  "${DMD_OVERRIDES[@]}" \
  --set "opd.rollout_manifest=${ROLLOUT_DEBUG_MANIFEST}"

run_train_stage \
  "11_video_ffm_train_smoke" \
  "$VIDEO_FFM_CONFIG" \
  data.manifest \
  "$VIDEO_DEBUG_MANIFEST" \
  "$(stage_run_dir video_ffm)" \
  "$(stage_ckpt_dir video_ffm)" \
  "$TRAIN_SMOKE_MAX_STEPS" \
  "$TRAIN_SMOKE_SAVE_EVERY" \
  "$TRAIN_SMOKE_LOG_EVERY" \
  "$TRAIN_SMOKE_BATCH_SIZE"

run_train_stage \
  "12_action_ffm_train_smoke" \
  "$ACTION_FFM_CONFIG" \
  data.manifest \
  "$ACTION_DEBUG_MANIFEST" \
  "$(stage_run_dir action_ffm)" \
  "$(stage_ckpt_dir action_ffm)" \
  "$TRAIN_SMOKE_MAX_STEPS" \
  "$TRAIN_SMOKE_SAVE_EVERY" \
  "$TRAIN_SMOKE_LOG_EVERY" \
  "$TRAIN_SMOKE_BATCH_SIZE"

run_train_stage \
  "13_video_dmd_train_smoke" \
  "$VIDEO_DMD_CONFIG" \
  opd.rollout_manifest \
  "$ROLLOUT_DEBUG_MANIFEST" \
  "$(stage_run_dir video_dmd)" \
  "$(stage_ckpt_dir video_dmd)" \
  "$TRAIN_SMOKE_MAX_STEPS" \
  "$TRAIN_SMOKE_SAVE_EVERY" \
  "$TRAIN_SMOKE_LOG_EVERY" \
  "$TRAIN_SMOKE_BATCH_SIZE" \
  "${DMD_OVERRIDES[@]}"

run_train_stage \
  "14_action_dmd_train_smoke" \
  "$ACTION_DMD_CONFIG" \
  opd.rollout_manifest \
  "$ROLLOUT_DEBUG_MANIFEST" \
  "$(stage_run_dir action_dmd)" \
  "$(stage_ckpt_dir action_dmd)" \
  "$TRAIN_SMOKE_MAX_STEPS" \
  "$TRAIN_SMOKE_SAVE_EVERY" \
  "$TRAIN_SMOKE_LOG_EVERY" \
  "$TRAIN_SMOKE_BATCH_SIZE" \
  "${DMD_OVERRIDES[@]}"

run_train_stage \
  "15_video_ffm_train_long_full_manifest" \
  "$VIDEO_FFM_CONFIG" \
  data.manifest \
  "$VIDEO_FULL_MANIFEST" \
  "$(stage_run_dir video_ffm)" \
  "$(stage_ckpt_dir video_ffm)" \
  "$TRAIN_MAX_STEPS" \
  "$TRAIN_SAVE_EVERY" \
  "$TRAIN_LOG_EVERY" \
  "$TRAIN_BATCH_SIZE"

run_train_stage \
  "16_action_ffm_train_long_full_manifest" \
  "$ACTION_FFM_CONFIG" \
  data.manifest \
  "$ACTION_FULL_MANIFEST" \
  "$(stage_run_dir action_ffm)" \
  "$(stage_ckpt_dir action_ffm)" \
  "$TRAIN_MAX_STEPS" \
  "$TRAIN_SAVE_EVERY" \
  "$TRAIN_LOG_EVERY" \
  "$TRAIN_BATCH_SIZE"

run_train_stage \
  "17_video_dmd_train_long_full_manifest" \
  "$VIDEO_DMD_CONFIG" \
  opd.rollout_manifest \
  "$ROLLOUT_FULL_MANIFEST" \
  "$(stage_run_dir video_dmd)" \
  "$(stage_ckpt_dir video_dmd)" \
  "$TRAIN_MAX_STEPS" \
  "$TRAIN_SAVE_EVERY" \
  "$TRAIN_LOG_EVERY" \
  "$TRAIN_BATCH_SIZE" \
  "${DMD_OVERRIDES[@]}"

run_train_stage \
  "18_action_dmd_train_long_full_manifest" \
  "$ACTION_DMD_CONFIG" \
  opd.rollout_manifest \
  "$ROLLOUT_FULL_MANIFEST" \
  "$(stage_run_dir action_dmd)" \
  "$(stage_ckpt_dir action_dmd)" \
  "$TRAIN_MAX_STEPS" \
  "$TRAIN_SAVE_EVERY" \
  "$TRAIN_LOG_EVERY" \
  "$TRAIN_BATCH_SIZE" \
  "${DMD_OVERRIDES[@]}"

run_shell_stage "19_collect_artifacts" "
set -euo pipefail
echo status_file: ${STATUS_FILE@Q}
echo log_dir: ${LOG_DIR@Q}
echo checkpoint_base: ${CHECKPOINT_BASE@Q}
find ${CHECKPOINT_BASE@Q} -path '*/${JOB_ID}/*' -type f | sort > ${LOG_DIR@Q}/checkpoint_files_${JOB_ID}.txt
cat ${LOG_DIR@Q}/checkpoint_files_${JOB_ID}.txt
"

echo "SUCCESS:${JOB_ID}:$(date -Is)" | tee -a "$STATUS_FILE"
echo "Full v2 admin pipeline completed. Logs: $LOG_DIR" | tee -a "$STATUS_FILE"
echo "Checkpoint base: $CHECKPOINT_BASE" | tee -a "$STATUS_FILE"
