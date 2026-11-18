#!/usr/bin/env bash
# Stage 2.1 Task-Specific RL smoke (push_box, tiny scale; doubles as an environment check of the policy env, the QLM-Bench
# scene worker and the renderer): prepare the scenes -> teacher episodes from reset on 2 scenes -> start split -> round
# r00 (teacher probe x2, 2 starts x 8 candidates, window reward and rho, one DiffusionNFT step + EMA, probe with the new
# old adapters, readback, round table) -> export r00 as a release specialist -> attach it to the teacher.
#
# Without TEACHER a tiny randomly initialised teacher is built (no weights download; a random policy does not touch the
# box, so the smoke then takes starts before contact and keeps constant reward groups -- a plumbing check, not training).
#
#   QLM_BENCH=<QLM-Bench checkout> QLM_PYTHON=<QLM env python> BASE_MODEL=<lingbot-va-base dir> \
#   PROMPT_ROOT=<prompt_embeddings dir> [TEACHER=<Stage 1.1 teacher dir>] [WORK=<new dir>] \
#   [DATA_REUSE=<tree holding the scene meshes>] [GL=egl|osmesa] [OSMESA_LIB=<dir>] [ROWS=2] \
#   scripts/smoke/task_rl_push_box.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO/configs/site.env" ] && source "$REPO/configs/site.env"
BASE_MODEL="${BASE_MODEL:-${WAM_BASE_MODEL:-}}"
PROMPT_ROOT="${PROMPT_ROOT:-${WAM_PROMPTS:-}}"; OSMESA_LIB="${OSMESA_LIB:-${QLM_OSMESA_LIB:-}}"
[ -z "${QLM_BENCH:-}" ] && [ -n "${QLM_BENCH_SRC:-}" ] && QLM_BENCH="$(dirname "$QLM_BENCH_SRC")"
: "${QLM_BENCH:?}" "${QLM_PYTHON:?}" "${BASE_MODEL:?}" "${PROMPT_ROOT:?}"
PY="${POLICY_PYTHON:-python}"
WORK="${WORK:-$(mktemp -d -t task-rl-smoke-XXXX)}"
GL="${GL:-osmesa}"; ROWS="${ROWS:-2}"
export PYTHONPATH="$REPO/src:$QLM_BENCH/src${PYTHONPATH:+:$PYTHONPATH}"
T() { "$PY" -m wam_policy.task_rl "$@"; }
mkdir -p "$WORK"; echo "work dir: $WORK"
t0=$(date +%s)

# 1. scenes
if [ ! -e "$WORK/cases/cases-index.json" ]; then
  if [ -n "${DATA_REUSE:-}" ]; then FETCH=(--reuse "$DATA_REUSE"); else FETCH=(--download); fi
  T prepare --task push_box --inputs "$QLM_BENCH/inputs/task_rl/push_box" --data-root "$WORK/data" --out "$WORK/cases" "${FETCH[@]}"
fi

# 2. teacher
EXTRA=()
if [ -z "${TEACHER:-}" ]; then
  TEACHER="$WORK/tiny-teacher"
  [ -e "$TEACHER/wam_policy.json" ] || "$PY" - "$TEACHER" <<'PYEOF'
import sys, torch
from pathlib import Path
from wam_policy.models import metadata
from wam_policy.testing.fixtures import TINY_CONFIG
from wan_va.modules.model import WanTransformer3DModel
out = Path(sys.argv[1])
torch.manual_seed(0)
WanTransformer3DModel(**TINY_CONFIG, attn_mode='torch').save_pretrained(out / 'transformer')
# the action normalizer of the released Stage 1.1 teacher (native9: vx, vy, yaw rate, FL foot xyz, zero force)
metadata.write(out, {'kind': 'teacher', 'rgb_profile': 'dual-rgb-360p-v1', 'sampling': {'video_steps': 25, 'action_steps': 50},
                     'action_normalizer': {'offset': [0.058469385951757366, -0.014171185027807964, -0.04378169694915417,
                                                      0.349012840911746, 0.11562707189470528, 0.17840446569025514, 0, 0, 0],
                                           'scale': [0.08413853824138635, 0.048443782050162545, 0.08683070188388227,
                                                     0.19649447090923786, 0.08942297752946615, 0.16226205222308632, 1, 1, 1]},
                     'notes': {'smoke': 'tiny randomly initialised transformer'}})
print('tiny teacher', out)
PYEOF
  EXTRA=(--set 'pool={"probe_scenes": 1, "train_scenes": 1, "probe_per_scene": 1, "train_per_scene": 3, "train_min": 0, "start_after_contact": false}'
         --set skip_range_m=0)
fi

# 3. run
T init --run "$WORK/run" --task push_box --config "$REPO/configs/stage2_1_task_rl/tiny.json" --set rows="$ROWS" "${EXTRA[@]}" \
  --teacher "$TEACHER" --base-model "$BASE_MODEL" --prompt-root "$PROMPT_ROOT" --cases "$WORK/cases" \
  --qlm-python "$QLM_PYTHON" --qlm-src "$QLM_BENCH/src" --gl "$GL" ${OSMESA_LIB:+--osmesa-lib "$OSMESA_LIB"}
[ -e "$WORK/run/pool/meta.json" ] || T pool --run "$WORK/run"
[ -e "$WORK/run/pool/selection.json" ] || T select --run "$WORK/run"
T rounds --run "$WORK/run" 0 0
T export --run "$WORK/run" --round r00 --output "$WORK/specialist"

# 4. read back
"$PY" - "$WORK" "$TEACHER" <<'PYEOF'
import json, sys
from pathlib import Path
from wam_policy.inference.loader import load_teacher
from wam_policy.task_rl.adapters import attach_specialist
work, teacher = Path(sys.argv[1]), sys.argv[2]
rb = json.loads((work / 'run/rounds/r00/readback.json').read_text())
lm = json.loads((work / 'run/rounds/r00/learn-metrics.json').read_text())
sc = json.loads((work / 'run/rounds/r00/scores.json').read_text())['aggregate']
model, _, _ = load_teacher(teacher, device='cpu', dtype='float32')
meta = attach_specialist(model, work / 'specialist')
dg = rb['diagnostics']
ok = rb['ok'] and lm['steps_this_round'] >= 1
print('TASK RL SMOKE', 'PASS' if ok else 'FAIL', '| candidates', rb['candidates'], '| groups', sc['groups'],
      'skipped', sc['skipped_groups'], '| samples', dg['training_samples'], '/ expected', dg['expected_training_samples'],
      '| reward recompute diff', dg['reward_recompute_max_abs_diff'], '| candidates with diagnostics', len(dg['candidates_with_differences']), '| steps', lm['steps_this_round'],
      '| grad_norm', round(lm['steps'][0]['grad_norm'], 4) if lm['steps'] else None,
      '| displacement', round(lm['displacement_current'], 5), '| specialist', meta['task'], meta['round'],
      'attached displacement', round(meta['displacement'], 5), '| weights_sha256', {k: v[:12] for k, v in meta['weights_sha256'].items()})
sys.exit(0 if ok else 1)
PYEOF
echo "elapsed $(( $(date +%s) - t0 )) s"
