#!/usr/bin/env bash
# Build this package into its own colcon workspace and verify the install.
#
#   scripts/build.sh
#
# The robot scripts start the controller installed in DEPLOY_WS, not the source next to them.
# This script links DEPLOY_WS/src/wam_go2_deploy to this directory (another checkout already
# linked there is refused, never replaced), builds with the deploy venv's python (so the
# installed entry points use it), then checks that the installed package is byte-identical to
# this source and is the one a sourced shell resolves.
#
# Settings (deploy/config/deploy.env): DEPLOY_WS, UNITREE_MSGS_WS, DEPLOY_VENV.
set -euo pipefail
# shellcheck source=load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_config.sh"

PKG="$WAM_DEPLOY_DIR"
NAME=wam_go2_deploy
WS="$DEPLOY_WS"

say() { echo "[build] $*"; }
die() { echo "[build] error: $*" >&2; exit 1; }
[[ $# -eq 0 ]] || { awk 'NR > 1 { if (!/^#/) exit; print substr($0, 3) }' "${BASH_SOURCE[0]}"; exit 2; }

# 1. the deploy venv: its python is baked into the entry points
[[ -x "$DEPLOY_VENV/bin/python" ]] || die "deploy venv missing: $DEPLOY_VENV (README: Installation)"
"$DEPLOY_VENV/bin/python" -c 'import numpy, scipy, osqp, torch' >/dev/null 2>&1 \
  || die "the deploy venv lacks numpy/scipy/osqp/torch: $DEPLOY_VENV (requirements.txt)"
DEPLOY_VENV="$(cd "$DEPLOY_VENV" && pwd)"
VENV_PY="$DEPLOY_VENV/bin/python"

# 2. the workspace's src links to this directory (a dangling link is replaced)
link="$WS/src/$NAME"
if [[ -L "$link" && ! -e "$link" ]]; then
  rm "$link"
fi
if [[ -e "$link" || -L "$link" ]]; then
  [[ "$(readlink -f "$link")" == "$PKG" ]] \
    || die "$link points to $(readlink -f "$link" || echo '?'), not $PKG: set DEPLOY_WS to another workspace"
else
  mkdir -p "$WS/src"
  ln -s "$PKG" "$link"
  say "new workspace: $link -> $PKG"
fi
WS="$(cd "$WS" && pwd -P)"
[[ -r /opt/ros/jazzy/setup.bash ]] || die "ROS 2 Jazzy not found (/opt/ros/jazzy/setup.bash)"
[[ -r "$UNITREE_MSGS_WS/install/setup.bash" ]] \
  || die "$UNITREE_MSGS_WS/install/setup.bash missing (scripts/install_unitree_msgs.sh)"
UNITREE_MSGS_WS="$(cd "$UNITREE_MSGS_WS" && pwd -P)"

# 3. build in a clean environment (no other workspace as underlay), colcon under the venv's python
mkdir -p "$WS/log"
build_log="$WS/log/wam_go2_build_last.log"
say "colcon build $NAME in $WS (log: $build_log)"
if ! env -i HOME="$HOME" PATH="$DEPLOY_VENV/bin:/usr/local/bin:/usr/bin:/bin" VIRTUAL_ENV="$DEPLOY_VENV" \
    LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 WS="$WS" UMW="$UNITREE_MSGS_WS" NAME="$NAME" VENV_PY="$VENV_PY" bash -c '
      source /opt/ros/jazzy/setup.bash && source "$UMW/install/setup.bash" &&
      cd "$WS" && "$VENV_PY" -m colcon build --packages-select "$NAME"' >"$build_log" 2>&1; then
  tail -n 40 "$build_log" >&2
  die "build failed (log: $build_log)"
fi

# 4. the installed package must be this source, byte for byte (colcon never deletes files that
#    left the source, so a stale module would otherwise stay importable)
inst="$WS/install/$NAME"
site="$(find "$inst/lib" -type d -name wam_go2 -path '*/site-packages/*' -print -quit 2>/dev/null || true)"
[[ -n "$site" ]] || die "no installed wam_go2 under $inst"
bad=()
while IFS= read -r -d '' f; do
  rel="${f#"$PKG"/wam_go2/}"
  cmp -s "$f" "$site/$rel" || bad+=("wam_go2/$rel")
done < <(find "$PKG/wam_go2" -type f -name '*.py' -print0)
while IFS= read -r -d '' f; do
  rel="${f#"$site"/}"
  [[ -f "$PKG/wam_go2/$rel" ]] || bad+=("wam_go2/$rel (installed only)")
done < <(find "$site" -type f -name '*.py' -print0)
for f in "$PKG"/package.xml "$PKG"/config/*.yaml "$PKG"/config/*.xml "$PKG"/launch/*.py "$PKG"/policies/quadruped/*; do
  [[ -e "$f" ]] || continue
  rel="${f#"$PKG"/}"
  cmp -s "$f" "$inst/share/$NAME/$rel" || bad+=("$rel")
done
for ep in controller wam_bridge ego_stream; do     # the entry points run the deploy venv's python
  [[ "$(head -n 1 "$inst/lib/$NAME/$ep" 2>/dev/null)" == "#!$VENV_PY" ]] || bad+=("lib/$NAME/$ep (not #!$VENV_PY)")
done
(( ${#bad[@]} == 0 )) || die "the installed package differs from this source: ${bad[*]}
  clean rebuild: rm -rf '$WS/build/$NAME' '$WS/install/$NAME' and run this again"

# 5. ... and it is the one a shell resolves after sourcing the workspaces as the robot scripts do
export PYTHONDONTWRITEBYTECODE=1
resolved="$(cd / && set +u && source /opt/ros/jazzy/setup.bash && source "$UNITREE_MSGS_WS/install/setup.bash" \
  && source "$WS/install/setup.bash" && ros2 pkg prefix "$NAME" \
  && "$VENV_PY" -c 'import os, wam_go2; print(os.path.dirname(os.path.realpath(wam_go2.__file__)))')" \
  || die "cannot resolve $NAME after sourcing $WS"
prefix="$(readlink -f "$(sed -n 1p <<<"$resolved")")"
module="$(sed -n 2p <<<"$resolved")"
[[ "$prefix" == "$(readlink -f "$inst")" && "$module" == "$(readlink -f "$site")" ]] \
  || die "another install would run: prefix $prefix, wam_go2 $module (expected $inst)"
say "ok: $inst"
