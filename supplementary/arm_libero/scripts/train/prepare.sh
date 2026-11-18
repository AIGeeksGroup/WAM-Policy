#!/usr/bin/env bash
# Inputs of every training run, from the machine paths of a site file (configs/site.example.json):
# official weights, CPU LIBERO environment check, reset cases (train/validation/test), the prompt bank,
# and for SFT only the empty-prompt embedding next to the demonstrations. Existing outputs are kept.
#   SITE=site.json PROMPTS=/data/arm/prompts.pt bash scripts/train/prepare.sh
# The CPU LIBERO environment itself comes from scripts/prepare/libero_cpu_env.sh, the demonstrations of
# SFT only from `hf download robbyant/libero-long-lerobot --repo-type dataset` (see README).
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${SITE:?SITE required}" "${PROMPTS:?PROMPTS required: where to write the prompt bank}"
PYTHON=${PYTHON:-python}
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
config=$root/configs/students/opsd.json
machine() { "$PYTHON" -c "import sys; from arm_opsd.config import ArmConfig; print(getattr(ArmConfig.load(sys.argv[1], site=sys.argv[2]).machine, sys.argv[3]))" "$config" "$SITE" "$1"; }
model_root=$(machine model_root); cpu_python=$(machine cpu_python); worker_env=$(machine worker_config)
resets=$(machine reset_manifest); sft_dataset=$(machine sft_dataset)
libero_root=${LIBERO_ROOT:?LIBERO_ROOT required: <sources>/LIBERO-<commit>/libero/libero of the CPU LIBERO environment}
[[ -d "$model_root/transformer" ]] || "$PYTHON" -m arm_opsd.cli download-base-model --output "$model_root"
"$PYTHON" "$root/scripts/prepare/check_libero_env.py" --python "$cpu_python" --worker-env "$worker_env" \
    --libero-root "$libero_root" --output "$(dirname "$resets")/env-check"
[[ -f "$resets" ]] || "$PYTHON" "$root/scripts/prepare/libero_resets.py" --python "$cpu_python" \
    --worker-env "$worker_env" --libero-root "$libero_root" --output "$(dirname "$resets")"
[[ -f "$PROMPTS" ]] || "$PYTHON" -m arm_opsd.cli prepare-prompts --config "$config" --site "$SITE" --output "$PROMPTS"
if [[ -n "$sft_dataset" && -d "$sft_dataset" ]]; then
    "$PYTHON" "$root/scripts/prepare/sft_empty_embedding.py" --config "$root/configs/students/sft_only.json" --site "$SITE"
fi
echo "Prepared: $model_root, $resets, $PROMPTS"
