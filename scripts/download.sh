#!/usr/bin/env bash
# Download released weights, the LingBot-VA base and (optionally) the Standard100 resources.
#   scripts/download.sh weights      # final checkpoints and prompts -> $WAM_WEIGHTS
#   scripts/download.sh base         # $WAM_BASE_MODEL_REPO@$WAM_BASE_MODEL_REVISION -> $WAM_BASE_MODEL
#   scripts/download.sh standard100  # QLM-Bench Standard100 cases + assets -> $QLM_RESOURCES (QLM-Bench env)
#   scripts/download.sh prompts      # only prompt_embeddings/ of $WAM_HF_MODEL_REPO -> $WAM_WEIGHTS/prompt_embeddings
#   scripts/download.sh reset_pool   # train800 reset pool (Stages 1.2/2.2) -> $QLM_BENCH_SNAPSHOT/reset_pools/train800-reset-pool/
# Variables come from configs/site.env (see configs/site.example.env).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/configs/site.env" ] && source "$ROOT/configs/site.env"
case ${1:-} in
  weights)
    huggingface-cli download "$WAM_HF_MODEL_REPO" --revision "$WAM_HF_MODEL_REVISION" \
      --exclude 'ablations/*' --local-dir "$WAM_WEIGHTS" ;;
  base)
    # The Stage 1.1 conversion needs transformer/; inference only needs vae/ (and text_encoder/ +
    # tokenizer/ when prompts are encoded on the fly instead of using prompt_embeddings/).
    huggingface-cli download "$WAM_BASE_MODEL_REPO" --revision "$WAM_BASE_MODEL_REVISION" --local-dir "$WAM_BASE_MODEL" ;;
  standard100)
    PYTHONPATH="$QLM_BENCH_SRC${PYTHONPATH:+:$PYTHONPATH}" "$QLM_PYTHON" -m qlm_bench.benchmark fetch --output "$QLM_RESOURCES" ;;
  prompts)
    huggingface-cli download "$WAM_HF_MODEL_REPO" --revision "$WAM_HF_MODEL_REVISION" \
      --include 'prompt_embeddings/*' --local-dir "$WAM_WEIGHTS" ;;
  reset_pool)
    # One tar of the OPSD/M-OPD training reset pool; it holds hard links, so extract with GNU tar as is
    # (about 126 GiB without them). The manifest is $QLM_TRAIN_POOL; its sha256 is part of the teacher-cache key.
    huggingface-cli download "$QLM_DATASET_REPO" reset_pools/train800-reset-pool.tar --repo-type dataset \
      --revision "$QLM_DATASET_REVISION" --local-dir "$QLM_BENCH_SNAPSHOT"
    tar -xf "$QLM_BENCH_SNAPSHOT/reset_pools/train800-reset-pool.tar" -C "$QLM_BENCH_SNAPSHOT/reset_pools"
    ls "$QLM_BENCH_SNAPSHOT/reset_pools/train800-reset-pool/manifest.json" ;;
  *) echo "usage: $0 {weights|prompts|base|standard100|reset_pool}" >&2; exit 2 ;;
esac
