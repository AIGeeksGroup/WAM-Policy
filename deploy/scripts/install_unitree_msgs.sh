#!/usr/bin/env bash
# Build the unitree_go / unitree_api ROS2 message packages (runtime
# dependency of wam_go2_deploy) into a standalone workspace, so the
# unitree_ros2 reference repository can be deleted afterwards.
#
# Tested on Ubuntu 24.04 with ROS 2 Jazzy.
#
# Usage:
#   ./scripts/install_unitree_msgs.sh [WS_DIR] [UNITREE_ROS2_SRC]
#     WS_DIR            target workspace (default: $UNITREE_MSGS_WS, else
#                       $HOME/ros2_ws/unitree_msgs)
#     UNITREE_ROS2_SRC  existing unitree_ros2 checkout to copy the msg
#                       packages from; if absent, the official repo is cloned.
#
# Known-good versions: unitree_ros2 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88 (master),
# rosidl_dds 8d6fdb365792c81861cf1d49da2d265e91dca86d (jazzy branch). The clone below takes the
# newest master; to build exactly those messages:
#   git clone https://github.com/unitreerobotics/unitree_ros2.git <dir> && git -C <dir> checkout 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
#   ./scripts/install_unitree_msgs.sh "$HOME/ros2_ws/unitree_msgs" <dir>
set -euo pipefail

WS_DIR="${1:-${UNITREE_MSGS_WS:-$HOME/ros2_ws/unitree_msgs}}"
UNITREE_SRC="${2:-}"
ROS_DISTRO=jazzy

# Jazzy's setup scripts read unset variables (AMENT_TRACE_SETUP_FILES, ...): source them without -u
set +u; source /opt/ros/jazzy/setup.bash; set -u

mkdir -p "${WS_DIR}/src"
cd "${WS_DIR}/src"

# rosidl_generator_dds_idl (required by unitree_go's CMakeLists; not shipped
# in the ros-jazzy apt repos as a standalone package on all systems)
if [ ! -d rosidl_dds ]; then
    git clone -b "${ROS_DISTRO}" https://github.com/ros2/rosidl_dds.git
fi

if [ ! -d unitree_go ] || [ ! -d unitree_api ]; then
    if [ -n "${UNITREE_SRC}" ] && [ -d "${UNITREE_SRC}/cyclonedds_ws/src/unitree" ]; then
        cp -r "${UNITREE_SRC}/cyclonedds_ws/src/unitree/unitree_go" .
        cp -r "${UNITREE_SRC}/cyclonedds_ws/src/unitree/unitree_api" .
    else
        tmp="$(mktemp -d)"
        git clone --depth 1 https://github.com/unitreerobotics/unitree_ros2.git "${tmp}/unitree_ros2"
        cp -r "${tmp}/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go" .
        cp -r "${tmp}/unitree_ros2/cyclonedds_ws/src/unitree/unitree_api" .
        rm -rf "${tmp}"
    fi
fi

cd "${WS_DIR}"
# unitree packages do not declare the dds_idl generator in package.xml, so
# build it first, then the message packages
colcon build --packages-select rosidl_generator_dds_idl
set +u; source install/setup.bash; set -u
colcon build --packages-select unitree_api unitree_go --cmake-args -DCMAKE_BUILD_TYPE=Release

echo
echo "Done. Add to your shell setup:"
echo "  source ${WS_DIR}/install/setup.bash"
