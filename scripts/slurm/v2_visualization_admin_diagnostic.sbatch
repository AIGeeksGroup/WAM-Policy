#!/usr/bin/env bash
#SBATCH --job-name=wam-v2-visualization-admin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-wam-v2-visualization-admin-%j.out
#SBATCH --error=slurm-wam-v2-visualization-admin-%j.err

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT is required." >&2
  exit 2
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VIS_CONFIG="${VIS_CONFIG:-configs/experiments/v2/anyflow_visualization.yaml}"
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$REPO_ROOT/logs/v2_visualization_admin/$JOB_ID"
STATUS_FILE="$LOG_DIR/status.txt"
OUTPUT_DIR="$REPO_ROOT/results/v2/visualization_admin/$JOB_ID"

TEACHER_CKPT="${TEACHER_CKPT:-}"
VIDEO_FULL_MANIFEST="${VIDEO_FULL_MANIFEST:-${FULL_MANIFEST:-}}"
VIDEO_STUDENT_CKPT="${VIDEO_STUDENT_CKPT:-$REPO_ROOT/experiments/checkpoints/v2/video_dmd/latest.pt}"
ACTION_STUDENT_CKPT="${ACTION_STUDENT_CKPT:-$REPO_ROOT/experiments/checkpoints/v2/action_dmd/latest.pt}"
RUN_OFFLINE_VISUALIZATION="${RUN_OFFLINE_VISUALIZATION:-true}"
RUN_ROBOTWIN_VISUALIZATION="${RUN_ROBOTWIN_VISUALIZATION:-true}"
ROBOTWIN_ROLLOUT_COMMAND="${ROBOTWIN_ROLLOUT_COMMAND:-}"
ROBOTWIN_ROLLOUT_WORKDIR="${ROBOTWIN_ROLLOUT_WORKDIR:-}"
VIS_MAX_CHUNKS="${VIS_MAX_CHUNKS:-16}"
VIS_ROBOTWIN_MAX_CHUNKS="${VIS_ROBOTWIN_MAX_CHUNKS:-32}"
VIS_FPS="${VIS_FPS:-10}"
VIS_TEACHER_STEPS="${VIS_TEACHER_STEPS:-25}"
VIS_STUDENT_STEPS="${VIS_STUDENT_STEPS:-4}"

WANDB_ENABLED="${WANDB_ENABLED:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-WAM-Policy}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-v2-visualization-${JOB_ID}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-wam-v2-visualization-${JOB_ID}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-wandb}"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
echo "STARTED:${JOB_ID}:$(date -Is)" | tee "$STATUS_FILE"
echo "log_dir: $LOG_DIR" | tee -a "$STATUS_FILE"
echo "output_dir: $OUTPUT_DIR" | tee -a "$STATUS_FILE"

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

COMMON_OVERRIDES=(
  --set runtime.device=rocm
  --set runtime.lingbot_attn_mode=torch
  --set "teacher.checkpoint_path=${TEACHER_CKPT}"
  --set "data.manifest=${VIDEO_FULL_MANIFEST}"
  --set "visualization.offline_video_stream.student_checkpoint_path=${VIDEO_STUDENT_CKPT}"
  --set "visualization.offline_video_stream.output_dir=${OUTPUT_DIR}/offline_video_stream"
  --set "visualization.offline_video_stream.max_chunks=${VIS_MAX_CHUNKS}"
  --set "visualization.offline_video_stream.fps=${VIS_FPS}"
  --set "visualization.offline_video_stream.teacher_steps=${VIS_TEACHER_STEPS}"
  --set "visualization.offline_video_stream.student_steps=${VIS_STUDENT_STEPS}"
  --set "visualization.robotwin_rollout.video_checkpoint_path=${VIDEO_STUDENT_CKPT}"
  --set "visualization.robotwin_rollout.action_checkpoint_path=${ACTION_STUDENT_CKPT}"
  --set "visualization.robotwin_rollout.output_dir=${OUTPUT_DIR}/robotwin_rollout"
  --set "visualization.robotwin_rollout.max_chunks=${VIS_ROBOTWIN_MAX_CHUNKS}"
  --set "visualization.robotwin_rollout.fps=${VIS_FPS}"
)

if [[ -n "$ROBOTWIN_ROLLOUT_COMMAND" ]]; then
  COMMON_OVERRIDES+=(
    --set visualization.robotwin_rollout.runner=external_command
    --set "visualization.robotwin_rollout.command=${ROBOTWIN_ROLLOUT_COMMAND}"
  )
  if [[ -n "$ROBOTWIN_ROLLOUT_WORKDIR" ]]; then
    COMMON_OVERRIDES+=(--set "visualization.robotwin_rollout.workdir=${ROBOTWIN_ROLLOUT_WORKDIR}")
  fi
fi

run_shell_stage "00_env" "
set -euo pipefail
echo hostname: \$(hostname)
echo date: \$(date -Is)
echo repo_root: ${REPO_ROOT@Q}
echo teacher_ckpt: ${TEACHER_CKPT@Q}
echo video_full_manifest: ${VIDEO_FULL_MANIFEST@Q}
echo video_student_ckpt: ${VIDEO_STUDENT_CKPT@Q}
echo action_student_ckpt: ${ACTION_STUDENT_CKPT@Q}
echo robotwin_rollout_command: ${ROBOTWIN_ROLLOUT_COMMAND@Q}
echo wandb_enabled: ${WANDB_ENABLED@Q}
command -v ${PYTHON_BIN@Q}
${PYTHON_BIN@Q} - <<'PY'
import torch
print('torch.__version__:', torch.__version__)
print('torch.version.hip:', getattr(torch.version, 'hip', None))
print('torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch.cuda.device_count:', torch.cuda.device_count())
    print('torch.cuda.get_device_name:', torch.cuda.get_device_name(0))
PY
if command -v rocm-smi >/dev/null 2>&1; then rocm-smi; else echo 'rocm-smi not found'; fi
test -f ${VIS_CONFIG@Q}
test -e ${TEACHER_CKPT@Q}
test -f ${VIDEO_FULL_MANIFEST@Q}
test -f ${VIDEO_STUDENT_CKPT@Q}
"

run_stage "01_import" \
  "$PYTHON_BIN" scripts/tools/probe_visualization_runtime.py probe-import \
  --config "$VIS_CONFIG" --require-external-exists \
  --output-dir "$OUTPUT_DIR/probes" "${COMMON_OVERRIDES[@]}"

run_stage "02_video_encoder_probe" \
  "$PYTHON_BIN" scripts/tools/probe_visualization_runtime.py probe-video-encoder \
  --config "$VIS_CONFIG" --output-dir "$OUTPUT_DIR/probes" "${COMMON_OVERRIDES[@]}"

run_stage "03_vae_decode_probe" \
  "$PYTHON_BIN" scripts/tools/probe_visualization_runtime.py probe-vae-decode \
  --config "$VIS_CONFIG" --require-external-exists \
  --output-dir "$OUTPUT_DIR/probes" "${COMMON_OVERRIDES[@]}"

run_stage "04_teacher_student_sampling_probe" \
  "$PYTHON_BIN" scripts/tools/probe_visualization_runtime.py probe-teacher-student-sampling \
  --config "$VIS_CONFIG" --require-external-exists \
  --output-dir "$OUTPUT_DIR/probes" "${COMMON_OVERRIDES[@]}"

run_stage "05_decode_and_encode_probe" \
  "$PYTHON_BIN" scripts/tools/probe_visualization_runtime.py probe-decode-and-encode \
  --config "$VIS_CONFIG" --require-external-exists \
  --output-dir "$OUTPUT_DIR/probes" "${COMMON_OVERRIDES[@]}"

run_shell_stage "06_visualization_full" "
set -euo pipefail
case ${RUN_OFFLINE_VISUALIZATION@Q} in
  0|false|False|FALSE|no|No|NO|off|Off|OFF) echo 'offline visualization skipped' ;;
  *) ${PYTHON_BIN@Q} scripts/visualize_v2.py offline-video-stream --config ${VIS_CONFIG@Q} \
       --require-external-exists --set logging.wandb.enabled=false ${COMMON_OVERRIDES[*]@Q} ;;
esac
case ${RUN_ROBOTWIN_VISUALIZATION@Q} in
  0|false|False|FALSE|no|No|NO|off|Off|OFF) echo 'RoboTwin visualization skipped' ;;
  *) ${PYTHON_BIN@Q} scripts/visualize_v2.py robotwin-rollout --config ${VIS_CONFIG@Q} \
       --require-external-exists --set logging.wandb.enabled=false ${COMMON_OVERRIDES[*]@Q} ;;
esac
"

run_stage "07_wandb_upload" \
  "$PYTHON_BIN" scripts/tools/upload_visualization_outputs_to_wandb.py \
  --output-dir "$OUTPUT_DIR" \
  --summary "$OUTPUT_DIR/offline_video_stream/offline_video_stream_metadata.json" \
  --summary "$OUTPUT_DIR/robotwin_rollout/robotwin_rollout_metadata.json" \
  --project "$WANDB_PROJECT" \
  --name "$WANDB_RUN_NAME" \
  --run-id "$WANDB_RUN_ID" \
  --resume "$WANDB_RESUME" \
  --mode "$WANDB_MODE" \
  --dir "$WANDB_DIR" \
  --enabled "$WANDB_ENABLED" \
  ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"}

run_shell_stage "08_collect_artifacts" "
set -euo pipefail
find ${OUTPUT_DIR@Q} -type f | sort > ${LOG_DIR@Q}/visualization_files_${JOB_ID}.txt
cat ${LOG_DIR@Q}/visualization_files_${JOB_ID}.txt
echo output_dir: ${OUTPUT_DIR@Q}
echo tar_gz: ${OUTPUT_DIR@Q}.tar.gz
"

echo "SUCCESS:${JOB_ID}:$(date -Is)" | tee -a "$STATUS_FILE"
echo "Visualization outputs: $OUTPUT_DIR" | tee -a "$STATUS_FILE"
echo "Visualization tar.gz: ${OUTPUT_DIR}.tar.gz" | tee -a "$STATUS_FILE"
