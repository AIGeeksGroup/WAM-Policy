#!/usr/bin/env bash
# Download the RAMBO controller checkpoint from the QLM-Bench dataset into policies/quadruped/:
#
#   scripts/fetch_controller.sh
#
# Source: $QLM_DATASET_REPO@$QLM_DATASET_REVISION (configs/site.env; default AIGeeksGroup/QLM-Bench@main),
# assets/rambo-go2-controller/iteration-2000/runtime/model_2000.pt. CC BY-NC 4.0 (see the dataset card).
set -euo pipefail
# shellcheck source=load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_config.sh"

REPO="${QLM_DATASET_REPO:-AIGeeksGroup/QLM-Bench}"
REVISION="${QLM_DATASET_REVISION:-main}"
FILE="assets/rambo-go2-controller/iteration-2000/runtime/model_2000.pt"
SHA256="1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706"
OUT="$WAM_DEPLOY_DIR/policies/quadruped/model_2000.pt"

if [[ -f "$OUT" ]] && echo "$SHA256  $OUT" | sha256sum -c --status; then
  echo "fetch_controller: $OUT is already there"
  exit 0
fi
mkdir -p "$(dirname "$OUT")"
url="https://huggingface.co/datasets/$REPO/resolve/$REVISION/$FILE"
echo "fetch_controller: $url"
curl -fsSL --retry 3 -o "$OUT.part" "$url"
echo "$SHA256  $OUT.part" | sha256sum -c --status \
  || { rm -f "$OUT.part"; echo "fetch_controller: sha256 mismatch, file removed" >&2; exit 1; }
mv "$OUT.part" "$OUT"
echo "fetch_controller: $OUT"
