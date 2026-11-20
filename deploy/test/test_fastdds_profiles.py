"""Static tests for the laptop FastDDS profiles.

config/fastdds_wired.xml binds every robot/camera participant to the wired NIC only (the laptop and
the robot must talk over the cable). The sim-camera republisher (scripts/x5/d435i_sim_camera.py)
sends 2.76 MB images to consumers on the laptop; over UDP loopback those bursts overflow the default
212 kB socket buffer (measured: 1033 receive-buffer errors in 20 s, 0.7 of 15 fps delivered).
config/fastdds_wired_shm.xml is the opt-in fix: the SAME wired UDP whitelist plus a host-local
shared-memory transport (no network interface at all), selected with CAMERA_DDS_SHM=1 in
scripts/x5/laptop_camera_env.bash. The default stays fastdds_wired.xml.

The X5's Humble realsense node cannot parse the Jazzy laptop's ros_discovery_info (node-name)
messages and runs out of memory on them (0.8-4.4 GB per laptop process start / exit; OOM kill, no
D435i frames). Every laptop profile therefore puts that one topic's writer and reader in the same
non-default partition; the X5's profile must not.
"""
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
NS = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
WIRED = PKG / "config" / "fastdds_wired.xml"
SHM = PKG / "config" / "fastdds_wired_shm.xml"
ENV = PKG / "scripts" / "x5" / "laptop_camera_env.bash"
BASH = shutil.which("bash")


def _transports(path):
    root = ET.parse(path).getroot()
    out = {}
    for td in root.findall("f:transport_descriptors/f:transport_descriptor", NS):
        tid = td.find("f:transport_id", NS).text
        out[tid] = {
            "type": td.find("f:type", NS).text,
            "whitelist": [a.text for a in td.findall("f:interfaceWhiteList/f:address", NS)],
            "segment_size": int(td.find("f:segment_size", NS).text) if td.find("f:segment_size", NS) is not None else None,
        }
    parts = root.findall("f:participant", NS)
    assert len(parts) == 1 and parts[0].get("is_default_profile") == "true"
    rtps = parts[0].find("f:rtps", NS)
    used = [t.text for t in rtps.findall("f:userTransports/f:transport_id", NS)]
    builtin = rtps.find("f:useBuiltinTransports", NS).text
    return out, used, builtin


def test_shm_profile_keeps_the_wired_udp_whitelist():
    wired, wired_used, wired_builtin = _transports(WIRED)
    shm, used, builtin = _transports(SHM)
    assert builtin == wired_builtin == "false"           # no builtin UDP on every NIC
    udp = {k: v for k, v in shm.items() if v["type"] == "UDPv4"}
    wired_udp = {k: v for k, v in wired.items() if v["type"] == "UDPv4"}
    assert [v["whitelist"] for v in udp.values()] == [v["whitelist"] for v in wired_udp.values()]
    assert all(v["whitelist"] for v in udp.values())     # every UDP transport is whitelisted
    assert {v["type"] for v in shm.values()} == {"UDPv4", "SHM"}
    assert sorted(used) == sorted(shm)                   # both declared transports are in use


def test_shm_segment_holds_whole_1280x720_frames():
    shm, _, _ = _transports(SHM)
    seg = [v["segment_size"] for v in shm.values() if v["type"] == "SHM"]
    assert len(seg) == 1 and seg[0] >= 4 * 1280 * 720 * 3


LAPTOP_IP = "192.0.2.10"                                   # a reserved documentation address


def _filled(path):
    """_transports of a template with @LAPTOP_IP@ replaced by LAPTOP_IP."""
    out, used, builtin = _transports(path)
    out = {k: {**v, "whitelist": [LAPTOP_IP if a == "@LAPTOP_IP@" else a for a in v["whitelist"]]}
           for k, v in out.items()}
    return out, used, builtin


def _env_profile(tmp_path, extra="", laptop_ip=LAPTOP_IP):
    """FASTRTPS_DEFAULT_PROFILES_FILE and stderr after sourcing the env script (ROS setup and ros2 stubbed)."""
    script = "\n".join([
        "source() { :; }",                               # the ROS setup files: not needed here
        "ros2() { :; }",
        extra,
        f"builtin source '{ENV}' >/dev/null",
        'echo "profile=${FASTRTPS_DEFAULT_PROFILES_FILE:-}"',
    ])
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("DEMO_") and k != "FASTRTPS_DEFAULT_PROFILES_FILE"}
    env.update(XDG_RUNTIME_DIR=str(tmp_path), DEMO_LAPTOP_IP=laptop_ip, WAM_NO_CONFIG="1")
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 0, r.stderr
    profile = r.stdout.strip().splitlines()[-1].removeprefix("profile=")
    return (Path(profile) if profile else None), r.stderr


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_env_default_is_the_plain_wired_profile(tmp_path):
    profile, _ = _env_profile(tmp_path)
    assert profile.parent == tmp_path and _transports(profile) == _filled(WIRED)


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_env_shm_opt_in(tmp_path):
    profile, _ = _env_profile(tmp_path, "export CAMERA_DDS_SHM=1")
    assert _transports(profile) == _filled(SHM)
    profile, _ = _env_profile(tmp_path, "export CAMERA_DDS_SHM=0")
    assert _transports(profile) == _filled(WIRED)


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_env_without_the_laptop_address_binds_nothing_and_says_so(tmp_path):
    profile, err = _env_profile(tmp_path, laptop_ip="")
    assert profile is None and "DEMO_LAPTOP_IP is not set" in err


X5_PROFILE = PKG / "scripts" / "x5" / "fastdds_x5_wired.xml"
X5_SCRIPT = PKG / "scripts" / "x5" / "go2_camera.sh"       # writes the X5's profile on the X5
COMMON = PKG / "scripts" / "demo" / "common.sh"
TASK_SH = PKG / "scripts" / "demo" / "wam_task.sh"
GRAPH_TOPIC = "ros_discovery_info"
GRAPH_PARTITION = ["wam_jazzy_graph"]


def _graph_partitions(path):
    """Partitions of the ros_discovery_info writer / reader profiles, one list per profile."""
    root = ET.parse(path).getroot()
    return {kind: [[n.text for n in p.findall("f:qos/f:partition/f:names/f:name", NS)]
                   for p in root.findall(f"f:{kind}", NS) if p.get("profile_name") == GRAPH_TOPIC]
            for kind in ("data_writer", "data_reader")}


def _shell_function(path, name):
    m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", path.read_text(encoding="utf-8"), re.S | re.M)
    assert m, f"{name} not found in {path}"
    return m.group(0)


def _generated_profiles(tmp_path):
    """The two profiles the demo scripts write: wired (scripts/demo/common.sh: controller, ros2 CLI)
    and wired + SHM (scripts/demo/wam_task.sh: bridge, ego_stream, the camera view, the D435i relay)."""
    script = "\n".join([
        "set -euo pipefail",
        f"source '{COMMON}'",
        _shell_function(TASK_SH, "wam_shm_profile"),
        "ros2() { :; }",                                 # demo_restart_ros2_daemon: no real daemon
        "demo_bind_fastdds_wired >/dev/null",
        'echo "$FASTRTPS_DEFAULT_PROFILES_FILE"',
        "wam_shm_profile",
        'echo "$WAM_SHM_PROFILE"',
    ])
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("DEMO_") and k != "FASTRTPS_DEFAULT_PROFILES_FILE"}
    env["XDG_RUNTIME_DIR"] = str(tmp_path)
    env["DEMO_LAPTOP_IP"] = LAPTOP_IP
    env["WAM_NO_CONFIG"] = "1"
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    wired, shm = (Path(p) for p in r.stdout.split())
    assert wired.parent == shm.parent == tmp_path
    return wired, shm


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_every_laptop_profile_puts_ros_discovery_info_in_one_partition(tmp_path):
    want = {"data_writer": [GRAPH_PARTITION], "data_reader": [GRAPH_PARTITION]}
    for path in (WIRED, SHM, *_generated_profiles(tmp_path)):
        assert _graph_partitions(path) == want, path


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_generated_profiles_match_the_config_templates(tmp_path):
    wired, shm = _generated_profiles(tmp_path)        # the templates with @LAPTOP_IP@ filled in
    assert _transports(wired) == _filled(WIRED)
    assert _transports(shm) == _filled(SHM)


def test_x5_profile_stays_in_the_default_partition():
    # the point is the asymmetry: the X5 (Humble) must never match the laptop's graph topic
    assert _graph_partitions(X5_PROFILE) == {"data_writer": [], "data_reader": []}
    for path in (X5_PROFILE, X5_SCRIPT):
        text = path.read_text(encoding="utf-8")
        assert "<partition>" not in text and "wam_jazzy_graph" not in text, path
