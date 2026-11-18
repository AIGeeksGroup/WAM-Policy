#!/usr/bin/env bash
# Stage 1.2 OPSD smoke / environment self-check (CPU, tiny synthetic transformer, no weights, no simulator;
# a few seconds): teacher target cache through the teacher-actor path -> streaming updates with cached
# contexts (full prefix replay and history reuse, compared) -> checkpoint + resume -> export in the
# release student format -> deployment sampler (wam_policy.opsd.deploy.student_actor) on that export.
#   scripts/smoke/opsd.sh [work_dir]     (PYTHON=... selects the interpreter)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-$(mktemp -d)}
PY=${PYTHON:-python}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=2
"$PY" -m wam_policy.opsd smoke --output "$WORK/opsd-smoke"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); print('OPSD smoke passed:', r['passed']); sys.exit(0 if r['passed'] else 1)" "$WORK/opsd-smoke/report.json"
