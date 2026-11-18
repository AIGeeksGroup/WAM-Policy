#!/usr/bin/env bash
# The CPU LIBERO environment (Python 3.10, MuJoCo 2.3.7, OSMesa/llvmpipe) of the simulation workers.
# Usage: libero_cpu_env.sh NEW_ENV_PATH SOURCE_DIR PYTHON310 OSMESA_LIB_DIR. Never modifies an existing env.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_path=${1:?Usage: libero_cpu_env.sh NEW_ENV_PATH ASSET_ROOT PYTHON310 OSMESA_LIB_DIR}
asset_root=${2:?fixed-source asset directory required}
python_bin=${3:?Python 3.10 interpreter required}
osmesa_dir=${4:?directory containing libOSMesa.so.8 required}
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 10), "the CPU LIBERO environment uses Python 3.10"'
libero_commit=8f1084e3132a39270c3a13ebe37270a43ece2a01
robosuite_commit=fbee5844ff5632f5b5698e204ec5357ca50be0df
if [[ -e "$env_path" ]]; then
    echo "Refusing to replace an existing environment: $env_path" >&2
    exit 2
fi
mkdir -p "$asset_root"
for package in LIBERO robosuite; do
    if [[ "$package" == LIBERO ]]; then
        owner=Lifelong-Robot-Learning
        commit=$libero_commit
    else
        owner=ARISE-Initiative
        commit=$robosuite_commit
    fi
    if [[ ! -d "$asset_root/$package-$commit" ]]; then
        curl -L --fail "https://codeload.github.com/$owner/$package/tar.gz/$commit" -o "$asset_root/$package-$commit.tar.gz"
        tar -xzf "$asset_root/$package-$commit.tar.gz" -C "$asset_root"
    fi
done
"$python_bin" -m venv "$env_path"
"$env_path/bin/python" -m pip install -r "$repo_root/requirements/libero-cpu.txt"
"$env_path/bin/python" -m pip install --no-deps "$asset_root/robosuite-$robosuite_commit" "$asset_root/LIBERO-$libero_commit"
mkdir -p "$env_path/libero-config"
"$env_path/bin/python" - "$asset_root/LIBERO-$libero_commit/libero/libero" "$env_path" "$repo_root" "$osmesa_dir" <<'PY'
import json, pathlib, sys
import yaml
root, env, repo, osmesa = map(pathlib.Path, sys.argv[1:])
config = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
          "init_states": str(root / "init_files"), "datasets": str(root / "../datasets"),
          "assets": str(root / "assets")}
(env / "libero-config/config.yaml").write_text(yaml.safe_dump(config))
worker_env = {"LIBERO_CONFIG_PATH": str(env / "libero-config"),
              "LD_LIBRARY_PATH": str(osmesa), "MUJOCO_GL": "osmesa",
              "PYOPENGL_PLATFORM": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "true",
              "GALLIUM_DRIVER": "llvmpipe", "PYTHONPATH": str(repo) + ":" + str(root.parent.parent),
              "NUMBA_DISABLE_JIT": "1", "PYTHONFAULTHANDLER": "1",
              "MPLCONFIGDIR": str(env / "matplotlib-cache"),
              "NUMBA_CACHE_DIR": str(env / "numba-cache")}
(env / "worker-env.json").write_text(json.dumps(worker_env, indent=2))
PY
"$env_path/bin/python" -m pip freeze > "$env_path/installed.freeze.txt"
echo "Prepared $env_path; machine.worker_config = $env_path/worker-env.json"
