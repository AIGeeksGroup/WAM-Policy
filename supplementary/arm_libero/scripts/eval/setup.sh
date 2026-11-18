#!/usr/bin/env bash
# One-shot preparation of an evaluation host (uv, no conda) under EVAL_ROOT:
#   model env -> [official weights | LIBERO-Pro2.0 env, server smoke, plan + instructions | CPU LIBERO env,
#   environment check, test resets | GPU check + tiny smokes] -> prompt embeddings of both benchmarks.
# Every finished stage leaves $EVAL_ROOT/setup-state/<stage>.done (a rerun skips it) and logs to
# $EVAL_ROOT/setup-logs/<stage>.log. Needs one NVIDIA GPU with EGL, libOSMesa.so.8 and Hugging Face read access.
#   EVAL_ROOT=/data/arm-eval bash scripts/eval/setup.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/scripts/eval/protocol.sh"
: "${EVAL_ROOT:?EVAL_ROOT required: where this host keeps environments, weights and results}"
export EVAL_ROOT
export UV_CACHE_DIR=${UV_CACHE_DIR:-$EVAL_ROOT/cache/uv}
export MODEL_PY=$EVAL_ROOT/envs/model/bin/python
export PYTHONPATH=$root
model_root=$EVAL_ROOT/resources/lingbot-va-posttrain-libero-long
plan_dir=$EVAL_ROOT/libero-pro2/plan
pkg=$EVAL_ROOT/resources/libero-pro2.0-eval
pro=$pkg/instance/LIBERO-Pro-evaluation
pro_env=$EVAL_ROOT/envs/libero-pro2
pro_py=$pro_env/bin/python
long_env=$EVAL_ROOT/envs/libero-cpu
long_sources=$EVAL_ROOT/resources/libero-cpu-sources
long_resets=$EVAL_ROOT/resources/libero-long-resets
long_dir=$EVAL_ROOT/libero-long
state=$EVAL_ROOT/setup-state
logs=$EVAL_ROOT/setup-logs
mkdir -p "$EVAL_ROOT"/{resources,envs} "$state" "$logs" "$long_dir"

stage() {  # stage NAME FUNCTION: run once, log to file, mark done
    local name=$1
    if [[ -f "$state/$name.done" ]]; then echo "[skip] $name"; return 0; fi
    echo "[$(date +%T)] start $name (log: $logs/$name.log)"
    local rc=0
    set +e
    (set -e; "$2") > "$logs/$name.log" 2>&1
    rc=$?
    set -e
    if [[ $rc == 0 ]]; then
        touch "$state/$name.done"; echo "[$(date +%T)] done  $name"
    else
        echo "[$(date +%T)] FAILED $name (exit $rc); last lines of $logs/$name.log:" >&2
        tail -20 "$logs/$name.log" >&2; return 1
    fi
}

model_env() {
    [[ -x "$MODEL_PY" ]] || uv venv --python 3.10 "$EVAL_ROOT/envs/model"
    uv pip install --python "$MODEL_PY" torch==2.9.0 torchvision==0.24.0 \
        --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    uv pip install --python "$MODEL_PY" --no-deps -r "$root/requirements/model.txt"
}

weights() {  # Every student is a LoRA on the official weights, so they are needed for every policy.
    "$MODEL_PY" -m arm_opsd.cli download-base-model --output "$model_root"
}

pro_chain() {
    [[ -f "$pkg/download/release.json" ]] || "$MODEL_PY" -m arm_opsd.evaluation.cli fetch-libero-pro --output "$pkg/download"
    [[ -f "$pro/SOURCE_BASELINE.json" ]] || python3 -I "$pkg/download/scripts/prepare_evaluation.py" \
        --download-dir "$pkg/download" --output-dir "$pkg/instance"
    [[ -x "$pro_py" ]] || uv venv --python 3.8 "$pro_env"
    uv pip install --python "$pro_py" "cmake<4"  # build-only, for egl_probe
    PATH="$pro_env/bin:$PATH" uv pip install --python "$pro_py" --index-strategy unsafe-best-match \
        -r "$pro/requirements-lock.txt"
    uv pip install --python "$pro_py" --no-deps --no-build-isolation -e "$pro/simulator-source" -e "$pro/service-source"
    # configure.sh takes the render GPU from LIBERO_PRO_GPU_ID and LIBERO_PRO_EGL_DEVICE_ID (default 0).
    [[ -f "$pro/server.env" ]] || bash "$pro/configure.sh" "$pro_env"
    bash "$pro/stop.sh"
    rmdir "$pro/runtime/start.lock" 2>/dev/null || true
    trap 'bash "$pro/stop.sh"' EXIT  # the stage runs in its own subshell
    bash "$pro/start.sh"
    bash "$pro/healthcheck.sh"
    (set -a; source "$pro/client.env"; set +a; "$pro_py" "$pro/smoke_test.py")
}

official_plan() {
    bash "$pro/stop.sh"
    rmdir "$pro/runtime/start.lock" 2>/dev/null || true
    trap 'bash "$pro/stop.sh"' EXIT
    bash "$pro/start.sh"
    bash "$pro/healthcheck.sh"
    [[ -f "$plan_dir/plan.json" ]] || (set -a; source "$pro/client.env"; set +a
        "$pro_py" "$pro/service-source/scripts/run_grouped_eval.py" --base-url "$LIBERO_REMOTE_URL" plan \
          "${PLAN_ARGS[@]}" --output-dir "$plan_dir")
    [[ -f "$plan_dir/instructions.json" ]] || (set -a; source "$pro/server.env"; set +a
        "$pro_py" "$root/arm_opsd/sim/libero_pro/instructions.py" --plan "$plan_dir/plan.json" \
          --task-manifest "$pro/service-source/assets/task_manifest.json" --output "$plan_dir/instructions.json")
    # Print what the plan holds; the paper's plan has $PLAN_ROLLOUTS episodes, $PLAN_TEXTS instructions and the plan id
    # in protocol.sh (for comparison only).
    "$MODEL_PY" - "$plan_dir" <<'PY'
import sys
from pathlib import Path
from arm_opsd.sim.libero_pro.plan import load_plan
from arm_opsd.evaluation.libero_pro import load_instructions
plan = load_plan(Path(sys.argv[1]) / "plan.json")
texts = load_instructions(Path(sys.argv[1]) / "instructions.json", plan)["texts"]
print("plan_id", plan["plan_id"], "episodes", plan["rollout_count"], "instructions", len(texts))
PY
    echo "paper: plan_id $PLAN_ID, $PLAN_ROLLOUTS episodes, $PLAN_TEXTS instructions"
}


long_cpu() {  # The CPU LIBERO worker of standard LIBERO-Long (OSMesa/llvmpipe), separate from the LIBERO-Pro2.0 env.
    local osmesa=${OSMESA_LIB_DIR:-}
    if [[ -z "$osmesa" ]]; then
        osmesa=$(ldconfig -p | awk '/libOSMesa\.so\.8/ {print $NF; exit}')
        [[ -n "$osmesa" ]] || { echo "libOSMesa.so.8 not found: install libosmesa6 or set OSMESA_LIB_DIR" >&2; return 1; }
        osmesa=$(dirname "$osmesa")
    fi
    [[ -f "$long_env/worker-env.json" ]] || bash "$root/scripts/prepare/libero_cpu_env.sh" \
        "$long_env" "$long_sources" "$(uv python find 3.10)" "$osmesa"
    local libero=$long_sources/LIBERO-$LIBERO_COMMIT/libero/libero
    "$MODEL_PY" "$root/scripts/prepare/check_libero_env.py" --python "$long_env/bin/python" \
        --worker-env "$long_env/worker-env.json" --libero-root "$libero" --output "$long_dir/env-check"
    if [[ ! -f "$long_resets/manifest.json" ]]; then
        rm -rf "$long_resets.partial"
        "$MODEL_PY" "$root/scripts/prepare/libero_resets.py" --python "$long_env/bin/python" \
            --worker-env "$long_env/worker-env.json" --libero-root "$libero" --output "$long_resets" --splits test
    fi
}

checks() {
    "$MODEL_PY" -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); \
a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16); print('matmul', (a @ a).float().abs().mean().item())"
    PYTHON=$MODEL_PY bash "$root/scripts/smoke/train_tiny.sh" "$EVAL_ROOT/smoke/train-tiny-$(date +%s)"
    "$MODEL_PY" "$root/scripts/smoke/eval_tiny.py" --output "$EVAL_ROOT/smoke/eval-tiny-$(date +%s)"
}

official_prompts() {
    "$MODEL_PY" -m arm_opsd.evaluation.cli prepare-prompts --model-root "$model_root" \
        --plan "$plan_dir/plan.json" --instructions "$plan_dir/instructions.json" --output "$plan_dir/prompts.pt"
}

long_prompts() {
    [[ -f "$long_dir/prompts.pt" ]] || "$MODEL_PY" -m arm_opsd.evaluation.cli prepare-long-prompts \
        --model-root "$model_root" --reset-manifest "$long_resets/manifest.json" --output "$long_dir/prompts.pt"
}

echo "EVAL_ROOT=$EVAL_ROOT"
stage model_env model_env
stage weights weights & w=$!
{ stage pro_chain pro_chain && stage official_plan official_plan; } & p=$!
stage checks checks & c=$!
stage long_cpu long_cpu & l=$!
failed=0
for pid in $w $p $c $l; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "Fix the failed stage(s) above and rerun; finished stages are skipped." >&2; exit 1; }
stage official_prompts official_prompts
stage long_prompts long_prompts
echo "Preparation complete."
