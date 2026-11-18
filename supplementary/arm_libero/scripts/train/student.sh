#!/usr/bin/env bash
# Train one student on one node (torchrun, method.world_size ranks).
#   SITE=site.json PROMPTS=/data/arm/prompts.pt OUTPUT=/data/arm/runs/opsd \
#     bash scripts/train/student.sh configs/students/opsd.json
# OUTPUT overrides machine.output_dir. RESUME=<OUTPUT>/checkpoints/round-NNNNNN continues a run.
# Students land in $OUTPUT/students/update-NNNNNN[-ema] (every experiment.student_snapshots update).
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
config=$(realpath "${1:?config required, e.g. configs/students/opsd.json}")
: "${SITE:?SITE required}" "${PROMPTS:?PROMPTS required}"
PYTHON=${PYTHON:-python}
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
site=$SITE
if [[ -n "${OUTPUT:-}" ]]; then  # A copy of the site file with this run's output directory.
    site=$(mktemp --suffix=.json)
    "$PYTHON" - "$SITE" "$OUTPUT" "$site" <<'PY'
import json, os, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
site = json.loads(source.read_text())
for key, value in site.get("machine", {}).items():
    if isinstance(value, str) and value and key in ("model_root", "cpu_python", "worker_config", "reset_manifest",
                                                    "offline_cache", "sft_dataset"):
        site["machine"][key] = os.path.abspath(source.parent / value)
site["machine"]["output_dir"] = os.path.abspath(sys.argv[2])
Path(sys.argv[3]).write_text(json.dumps(site, indent=2))
PY
fi
read -r world output < <("$PYTHON" -c "import sys; from arm_opsd.config import ArmConfig; c = ArmConfig.load(sys.argv[1], site=sys.argv[2]); print(c.method.world_size, c.machine.output_dir)" "$config" "$site")
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" --max_restarts=0 \
    -m arm_opsd.cli train --config "$config" --site "$site" --prompts "$PROMPTS" ${RESUME:+--resume "$RESUME"}
echo "Students: $output/students"
