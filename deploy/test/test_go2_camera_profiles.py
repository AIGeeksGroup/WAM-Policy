"""Offline tests for the stream profiles of scripts/x5/go2_camera.sh (RDK X5 camera host).

The colour stream follows the simulation camera while depth stays small, so CAMERA_DEPTH_PROFILE
is an optional camera.conf key; when it is absent the depth stream follows CAMERA_PROFILE (a
camera.conf without the key launches the same streams for both).
No X5, no ROS: the shell functions are cut out of the script and run by bash.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "x5" / "go2_camera.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _src():
    return SCRIPT.read_text()


def _extract(name):
    lines = _src().splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"^{name}\(\)\s*\{{", line):
            if line.rstrip().endswith("}") and line.count("{") == line.count("}"):
                return line
            for j in range(i + 1, len(lines)):
                if lines[j] == "}":
                    return "\n".join(lines[i:j + 1])
    raise AssertionError(f"function {name} not found in {SCRIPT}")


def _launch_args(tmp_path, conf_lines):
    """launch_args after the defaults block and a camera.conf with `conf_lines`."""
    head = _src().split('[ -f "$CAMERA_CONF" ] && source "$CAMERA_CONF"')[0]
    defaults = [ln for ln in head.splitlines() if re.match(r"^(CAMERA_[A-Z_]*|ROS_DOMAIN_ID)=", ln)]
    conf = tmp_path / "camera.conf"
    conf.write_text("\n".join(conf_lines) + "\n")
    script = "\n".join(["set -uo pipefail", f"CONF='{tmp_path}'", *defaults, f"source '{conf}'",
                        _extract("depth_profile"), _extract("write_camera_params"), _extract("launch_args"),
                        "launch_args"])
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _profiles(out):
    colour = re.search(r"rgb_camera\.color_profile:=(\S+)", out).group(1)
    depth = re.search(r"depth_module\.depth_profile:=(\S+)", out).group(1)
    return colour, depth


def test_camera_conf_without_depth_key_launches_depth_like_colour(tmp_path):
    """A camera.conf without the depth key: depth follows the colour profile."""
    out = _launch_args(tmp_path, ["CAMERA_IMU=1", "CAMERA_PROFILE=640x360x30"])
    assert _profiles(out) == ("640x360x30", "640x360x30")
    assert "enable_depth:=true" in out


def test_depth_profile_can_differ_from_colour(tmp_path):
    out = _launch_args(tmp_path, ["CAMERA_IMU=1", "CAMERA_PROFILE=1280x720x15", "CAMERA_DEPTH_PROFILE=640x360x15"])
    assert _profiles(out) == ("1280x720x15", "640x360x15")


def test_empty_depth_key_means_follow_colour(tmp_path):
    out = _launch_args(tmp_path, ["CAMERA_PROFILE=1280x720x15", "CAMERA_DEPTH_PROFILE="])
    assert _profiles(out) == ("1280x720x15", "1280x720x15")


def test_depth_default_is_set_before_camera_conf_and_documented():
    src = _src()
    assert re.search(r"^CAMERA_DEPTH_PROFILE=$", src, re.M), "empty default in the defaults block"
    assert src.index("\nCAMERA_DEPTH_PROFILE=") < src.index('source "$CAMERA_CONF"')
    header = src.split("set -uo pipefail")[0]
    assert "CAMERA_DEPTH_PROFILE" in header
    # the start log names both profiles
    assert re.search(r'log "starting realsense2_camera: profile=\$CAMERA_PROFILE depth=\$\(depth_profile\)', src)


def test_example_conf_launches_the_live_streams(tmp_path):
    """camera.conf.example = the deployed profile (colour 640x360x60, depth 640x360x15)."""
    example = SCRIPT.parent / "camera.conf.example"
    lines = [ln for ln in example.read_text().splitlines() if ln and not ln.startswith("#")]
    out = _launch_args(tmp_path, lines)
    assert _profiles(out) == ("640x360x60", "640x360x15")
