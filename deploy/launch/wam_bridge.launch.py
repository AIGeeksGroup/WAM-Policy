"""wam_bridge.launch.py: the WAM-Policy bridge and the ego camera stream.

Starts ``ego_stream`` (Unitree's /frontvideostream decoded to 640x360 rgb8 on
/camera/go2/ego/image_raw, ~14 fps) and ``wam_bridge`` (the qlm-live-v1 client
driving the controller). The controller and the real-to-sim D435i stream run
separately (scripts/demo/wam_task.sh starts all of them in order).
Parameters come from config/wam_bridge.yaml plus the launch arguments below.

DDS: large local image streams need the shared-memory profile. Unless the
operator bound DDS already (FASTRTPS_DEFAULT_PROFILES_FILE / CycloneDDS), this
launch points FastDDS at config/fastdds_wired_shm.xml: the wired whitelist of
config/fastdds_wired.xml plus host-local shared memory.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def dds_actions(share):
    if os.environ.get("RMW_IMPLEMENTATION", "") == "rmw_cyclonedds_cpp" and os.environ.get("CYCLONEDDS_URI"):
        return [LogInfo(msg="DDS: CycloneDDS with CYCLONEDDS_URI (operator-bound)")]
    if os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"):
        return [LogInfo(msg="DDS: FastDDS profile from the environment: " + os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"])]
    laptop_ip = os.environ.get("DEMO_LAPTOP_IP", "")
    if not laptop_ip:
        return [LogInfo(msg="WARNING: DDS is not bound to the wired interface (set DEMO_LAPTOP_IP or "
                            "FASTRTPS_DEFAULT_PROFILES_FILE, docs/network.md)")]
    template = os.path.join(share, "config", "fastdds_wired_shm.xml")
    profile = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"wam_go2_fastdds_wired_shm_{laptop_ip}.xml")
    with open(template, encoding="utf-8") as src, open(profile, "w", encoding="utf-8") as dst:
        dst.write(src.read().replace("@LAPTOP_IP@", laptop_ip))
    return [SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", profile),
            LogInfo(msg="DDS: wired whitelist + shared memory (FastDDS): " + profile)]


def generate_launch_description():
    share = get_package_share_directory("wam_go2_deploy")
    config = os.path.join(share, "config", "wam_bridge.yaml")
    args = [
        DeclareLaunchArgument("host", default_value="127.0.0.1", description="policy server host (localhost only)"),
        DeclareLaunchArgument("port", default_value="8862", description="policy server port"),
        DeclareLaunchArgument("instruction", default_value="Push the box inside the outlined area.",
                              description="task instruction sent at reset (config/wam_tasks.json; a server with "
                                          "--prompt-root knows only the tasks' instructions)"),
        DeclareLaunchArgument("task_id", default_value="push_box",
                              description="task id reported at reset (config/wam_tasks.json)"),
        DeclareLaunchArgument("task_version", default_value="mujoco-push_box-runtime-v1",
                              description="task profile reported at reset (the task's training profile, "
                                          "config/wam_tasks.json)"),
        DeclareLaunchArgument("session_id", default_value="go2-real"),
        DeclareLaunchArgument("max_chunks", default_value="1000"),
        DeclareLaunchArgument("max_wall_s", default_value="0.0", description="0 = no wall-clock budget"),
        DeclareLaunchArgument("save_images", default_value="false", description="PNG of every sample pair"),
        DeclareLaunchArgument("go2_latency_ms", default_value="0.0",
                              description="calibrated Go2 ego pipeline latency (0 = receive-time estimate)"),
        DeclareLaunchArgument("calib", default_value="", description="sync_check / pair_report calibration JSON"),
        DeclareLaunchArgument("output_dir", default_value="~/.ros/wam_bridge"),
        DeclareLaunchArgument("start_timeout_s", default_value="15.0",
                              description="wait this long for the streams, POLICY and the entry sequence's exit "
                                          "(the demo starts the bridge before the sequence and passes 90)"),
        DeclareLaunchArgument("settle_s", default_value="2.0",
                              description="the controller must have been in POLICY this long before the cold start"),
    ]

    ego = Node(package="wam_go2_deploy", executable="ego_stream", name="ego_stream", output="screen",
               parameters=[{"res": 360,
                            "go2_latency_ms": ParameterValue(LaunchConfiguration("go2_latency_ms"), value_type=float),
                            "calib": ParameterValue(LaunchConfiguration("calib"), value_type=str)}])
    # ego_stream never exits on its own: the bridge's exit ends the launch, so the demo script waiting
    # for it can go on (DAMP, cleanup, stopping the policy server)
    bridge = Node(package="wam_go2_deploy", executable="wam_bridge", name="wam_bridge", output="screen",
                  on_exit=[Shutdown(reason="wam_bridge exited")],
                  parameters=[config, {
                      "host": ParameterValue(LaunchConfiguration("host"), value_type=str),
                      "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
                      "instruction": ParameterValue(LaunchConfiguration("instruction"), value_type=str),
                      "task_id": ParameterValue(LaunchConfiguration("task_id"), value_type=str),
                      "task_version": ParameterValue(LaunchConfiguration("task_version"), value_type=str),
                      "session_id": ParameterValue(LaunchConfiguration("session_id"), value_type=str),
                      "max_chunks": ParameterValue(LaunchConfiguration("max_chunks"), value_type=int),
                      "max_wall_s": ParameterValue(LaunchConfiguration("max_wall_s"), value_type=float),
                      "save_images": ParameterValue(LaunchConfiguration("save_images"), value_type=bool),
                      "output_dir": ParameterValue(LaunchConfiguration("output_dir"), value_type=str),
                      "start_timeout_s": ParameterValue(LaunchConfiguration("start_timeout_s"), value_type=float),
                      "settle_s": ParameterValue(LaunchConfiguration("settle_s"), value_type=float)}])
    return LaunchDescription(args + dds_actions(share) + [ego, bridge])
