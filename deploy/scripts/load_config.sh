# shellcheck shell=bash
# Sourced by the deploy scripts: loads the repository's configs/site.env (model paths, RAMBO_CONTROLLER) and
# deploy/config/deploy.env (robot-side settings), then fills in the defaults. A variable that is already
# set when a script starts (e.g. `RAMBO_CONTROLLER=/x/model_2000.pt tasks/push_box.sh`) keeps its value.
#   WAM_SITE_ENV / WAM_DEPLOY_ENV point at other files; WAM_NO_CONFIG=1 skips both (tests).

WAM_DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WAM_REPO_DIR="$(cd "$WAM_DEPLOY_DIR/.." && pwd -P)"

if [[ "${WAM_NO_CONFIG:-0}" != 1 ]]; then
  _wam_site="${WAM_SITE_ENV:-$WAM_REPO_DIR/configs/site.env}"
  _wam_deploy="${WAM_DEPLOY_ENV:-$WAM_DEPLOY_DIR/config/deploy.env}"
  # the variables the files assign that are already set: the caller's values win
  _wam_keep=()
  for _wam_file in "$_wam_site" "$_wam_deploy"; do
    [[ -r "$_wam_file" ]] || continue
    while IFS= read -r _wam_name; do
      [[ -n "${!_wam_name+x}" ]] && _wam_keep+=("$_wam_name=${!_wam_name}")
    done < <(sed -n 's/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}\([A-Za-z_][A-Za-z0-9_]*\)=.*/\2/p' "$_wam_file")
  done
  # shellcheck disable=SC1090
  [[ ! -r "$_wam_site" ]] || { set +u; source "$_wam_site"; set -u; }
  # shellcheck disable=SC1090
  [[ ! -r "$_wam_deploy" ]] || { set +u; source "$_wam_deploy"; set -u; }
  for _wam_kv in "${_wam_keep[@]}"; do export "${_wam_kv%%=*}=${_wam_kv#*=}"; done
  unset _wam_site _wam_deploy _wam_keep _wam_file _wam_name _wam_kv
fi

: "${DEMO_NIC:=}"
: "${DEMO_LAPTOP_IP:=}"
: "${DEMO_ROBOT_IP:=}"
: "${X5_IP:=}"
: "${DEPLOY_VENV:=$HOME/venvs/wam-go2}"
: "${UNITREE_MSGS_WS:=$HOME/ros2_ws/unitree_msgs}"
: "${DEPLOY_WS:=$HOME/ros2_ws/wam_go2}"
: "${WAM_POLICY_PYTHON:=python3}"
: "${RAMBO_CONTROLLER:=}"
: "${WAM_POLICY:=opsd_student}"
: "${WAM_PORT:=8862}"
: "${WAM_SERVER_ARGS:=}"
export DEMO_NIC DEMO_LAPTOP_IP DEMO_ROBOT_IP X5_IP DEPLOY_VENV UNITREE_MSGS_WS DEPLOY_WS WAM_POLICY_PYTHON \
       RAMBO_CONTROLLER WAM_POLICY WAM_PORT WAM_SERVER_ARGS
