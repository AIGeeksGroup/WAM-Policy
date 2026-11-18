#!/usr/bin/env bash
# Stage 2.2 M-OPD smoke (CPU, tiny synthetic transformer, a few seconds): synthetic specialists in the
# release format -> task routing on the teacher -> specialist teacher cache (routed teacher actor on the
# expert tasks) merged into an OPSD cache -> student initialized from an export -> updates -> export.
#   scripts/smoke/mopd.sh [work_dir]     (PYTHON=... selects the interpreter)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-$(mktemp -d)}
PY=${PYTHON:-python}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=2
"$PY" -m wam_policy.mopd smoke --output "$WORK/mopd-smoke"
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); print('M-OPD smoke passed:', r['passed']); sys.exit(0 if r['passed'] else 1)" "$WORK/mopd-smoke/report.json"
