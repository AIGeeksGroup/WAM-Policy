# Laptop terminal environment for LOOKING AT the camera topics (RViz2, ros2
# topic, bag record). Same domain and wired-only DDS binding as the robot
# tooling. Usage:  source scripts/x5/laptop_camera_env.bash
# The DDS profile is config/fastdds_wired.xml with DEMO_LAPTOP_IP filled in (written to
# $XDG_RUNTIME_DIR). CAMERA_DDS_SHM=1 (opt-in) adds a host-local shared-memory transport to the same
# wired whitelist (config/fastdds_wired_shm.xml), for large images passed between processes on the
# laptop, e.g. scripts/x5/d435i_sim_camera.py and its consumers.
# Needs ROS 2 Jazzy, the workspaces UNITREE_MSGS_WS (the unitree_go/unitree_api messages) and
# DEPLOY_WS (this package), and DEPLOY_VENV (a venv with av/numpy for the camera tools; cv2 is the
# apt python3-opencv, seen through the venv's system site packages), from config/deploy.env as for
# every other script (scripts/load_config.sh).
_camera_deploy="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_camera_nounset=0; [[ $- == *u* ]] && _camera_nounset=1
# shellcheck source=../load_config.sh
source "$_camera_deploy/scripts/load_config.sh"
(( _camera_nounset )) || set +u
source /opt/ros/jazzy/setup.bash
for _camera_ws in "$UNITREE_MSGS_WS" "$DEPLOY_WS"; do
  if [ -f "$_camera_ws/install/setup.bash" ]; then
    source "$_camera_ws/install/setup.bash"
  else
    echo "camera env: WARNING no $_camera_ws/install/setup.bash (unitree messages / this package not sourced)" >&2
  fi
done
_camera_venv="$DEPLOY_VENV"
if [ -f "$_camera_venv/bin/activate" ]; then
  source "$_camera_venv/bin/activate"
else
  echo "camera env: WARNING no deploy venv $_camera_venv (python3 without av/numpy pins)" >&2
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" ROS_LOCALHOST_ONLY=0
unset CYCLONEDDS_URI RMW_IMPLEMENTATION
_camera_dds=fastdds_wired
[ "${CAMERA_DDS_SHM:-0}" = "1" ] && _camera_dds=fastdds_wired_shm
if [ -n "$DEMO_LAPTOP_IP" ]; then
  # the template with the laptop's wired address (DEMO_LAPTOP_IP, config/deploy.env) filled in
  export FASTRTPS_DEFAULT_PROFILES_FILE="${XDG_RUNTIME_DIR:-/tmp}/wam_go2_${_camera_dds}_${DEMO_LAPTOP_IP}.xml"
  sed "s/@LAPTOP_IP@/${DEMO_LAPTOP_IP}/g" "$_camera_deploy/config/$_camera_dds.xml" > "$FASTRTPS_DEFAULT_PROFILES_FILE"
  echo "camera env: domain $ROS_DOMAIN_ID, DDS whitelist from $FASTRTPS_DEFAULT_PROFILES_FILE"
else
  unset FASTRTPS_DEFAULT_PROFILES_FILE
  echo "camera env: WARNING DEMO_LAPTOP_IP is not set (config/deploy.env): DDS is not bound to the wired interface" >&2
fi
unset _camera_ws _camera_deploy _camera_venv _camera_nounset _camera_dds
ros2 daemon stop >/dev/null 2>&1
