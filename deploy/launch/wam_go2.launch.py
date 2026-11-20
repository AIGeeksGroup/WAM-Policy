import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def wired_dds_actions(share):
    """Host <-> robot over the Ethernet cable ONLY.

    With the default FastDDS participant the host announces its WiFi address
    too and a robot that is also on WiFi may stream /lowstate over WiFi:
    20-250 ms bursts of lost frames. Unless the operator already bound DDS
    (FASTRTPS_DEFAULT_PROFILES_FILE set, or CycloneDDS with CYCLONEDDS_URI),
    point FastDDS at config/fastdds_wired.xml with the host's wired address
    (DEMO_LAPTOP_IP) as its interfaceWhiteList. The node additionally refuses
    to move unless the LowState bytes arrive on `wired_nic`."""
    rmw = os.environ.get("RMW_IMPLEMENTATION", "")
    if rmw == "rmw_cyclonedds_cpp":
        if os.environ.get("CYCLONEDDS_URI"):
            return [LogInfo(msg="DDS: CycloneDDS with CYCLONEDDS_URI (operator-bound)")]
        return [LogInfo(msg="WARNING: rmw_cyclonedds_cpp without CYCLONEDDS_URI - DDS is NOT "
                            "bound to the wired NIC (docs/network.md); the node will refuse to move "
                            "unless /lowstate arrives on wired_nic")]
    if os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"):
        return [LogInfo(msg="DDS: FastDDS profile from the environment: "
                            + os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"])]
    laptop_ip = os.environ.get("DEMO_LAPTOP_IP", "")
    if not laptop_ip:
        return [LogInfo(msg="WARNING: DDS is not bound to the wired NIC (set DEMO_LAPTOP_IP or "
                            "FASTRTPS_DEFAULT_PROFILES_FILE, docs/network.md); the node will refuse "
                            "to move unless /lowstate arrives on wired_nic")]
    template = os.path.join(share, "config", "fastdds_wired.xml")
    profile = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"wam_go2_fastdds_wired_{laptop_ip}.xml")
    with open(template, encoding="utf-8") as src, open(profile, "w", encoding="utf-8") as dst:
        dst.write(src.read().replace("@LAPTOP_IP@", laptop_ip))
    return [SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", profile),
            LogInfo(msg="DDS bound to the wired NIC (FastDDS interfaceWhiteList): " + profile)]


def generate_launch_description():
    share = get_package_share_directory("wam_go2_deploy")
    config = os.path.join(share, "config", "wam_go2.yaml")

    task_arg = DeclareLaunchArgument(
        "task",
        default_value="quadruped",
        description="RAMBO task: quadruped (three-legged walk, FL arm)")

    policy_arg = DeclareLaunchArgument(
        "policy_checkpoint",
        default_value="",
        description="Path to the RAMBO policy checkpoint (.pt or .onnx). "
                    "Empty: the node loads the task's checkpoint from the package's policies/ "
                    "directory (policies/quadruped/model_2000.pt, downloaded separately)")

    heading_arg = DeclareLaunchArgument(
        "heading_hold",
        default_value="true",
        description="Quadruped straight walks: hold the IMU yaw the walk started with by "
                    "commanding the policy's yaw rate (PI, |wz| <= heading_hold_max_wz); the "
                    "three-leg gait yaws right on its own otherwise. false = raw operator "
                    "command (wam_go2/control/heading_hold.py)")

    in_place_arg = DeclareLaunchArgument(
        "heading_hold_in_place",
        default_value="false",
        description="Quadruped: also hold the IMU yaw while the whole command is ~zero (the "
                    "three-leg gait turns right while stepping in place); any command releases "
                    "it. WAM runs set true (wam_go2/control/heading_hold.py)")

    handling_arg = DeclareLaunchArgument(
        "command_handling",
        default_value="rambo",
        description="Native9 commands: rambo (clipped to RAMBO's training ranges, zeroed in all-stance) "
                    "or sim (unclipped, exactly as the QLM-Bench MuJoCo simulation of the WAM training "
                    "data; WAM runs set it)")

    # Explicitly typed overrides reach Node.parameters; undeclared launch
    # text is not a running controller setting.
    safety_settings = {
        "wired_nic": ("eth0", str, "Required robot Ethernet NIC (your wired interface, "
                                   "see `ip -br link`; none = mock robot)"),
        "ee_force_cmd_timeout": ("0.5", float, "Force command silence timeout [s]"),
        "source_stale_timeout": ("0.05", float, "Maximum source state age [s]"),
        "control_gap_timeout": ("0.05", float, "Maximum host control scheduling gap [s]"),
        "estimator_timeout": ("0.05", float, "Maximum age of successful state estimation [s]"),
    }
    safety_args = [DeclareLaunchArgument(name, default_value=default, description=desc)
                   for name, (default, _, desc) in safety_settings.items()]
    safety_parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=value_type)
        for name, (_, value_type, _) in safety_settings.items()}

    controller = Node(
        package="wam_go2_deploy",
        executable="controller",
        name="wam_go2_controller",
        output="screen",
        parameters=[
            config,
            safety_parameters,
            {"task": LaunchConfiguration("task"),
             "policy_checkpoint": LaunchConfiguration("policy_checkpoint"),
             "heading_hold": ParameterValue(LaunchConfiguration("heading_hold"), value_type=bool),
             "heading_hold_in_place": ParameterValue(
                 LaunchConfiguration("heading_hold_in_place"), value_type=bool),
             "command_handling": LaunchConfiguration("command_handling")},
        ],
    )

    return LaunchDescription(wired_dds_actions(share) + safety_args
                             + [task_arg, policy_arg, heading_arg, in_place_arg, handling_arg, controller])
