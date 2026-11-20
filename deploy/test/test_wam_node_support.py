"""wam_go2/wam/node_support.py: ROS-free helpers of the wam_bridge node."""
import numpy as np
import pytest

from wam_go2.wam import node_support as ns
from wam_go2.wam.limits import Bounds
from wam_go2.wam.observation import Tolerances


def test_image_to_rgb_handles_rgb8_bgr8_and_row_padding():
    h, w = 4, 6
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    out = ns.image_to_rgb(height=h, width=w, step=w * 3, encoding="rgb8", data=rgb.tobytes())
    assert np.array_equal(out, rgb) and out.flags["C_CONTIGUOUS"]
    out = ns.image_to_rgb(height=h, width=w, step=w * 3, encoding="bgr8", data=rgb.tobytes())
    assert np.array_equal(out, rgb[:, :, ::-1])
    padded = np.zeros((h, w * 3 + 5), np.uint8)
    padded[:, :w * 3] = rgb.reshape(h, w * 3)
    out = ns.image_to_rgb(height=h, width=w, step=w * 3 + 5, encoding="rgb8", data=padded.tobytes())
    assert np.array_equal(out, rgb)
    with pytest.raises(ValueError):
        ns.image_to_rgb(height=h, width=w, step=w * 3, encoding="mono8", data=rgb.tobytes())
    with pytest.raises(ValueError):
        ns.image_to_rgb(height=h, width=w, step=w * 3, encoding="rgb8", data=rgb.tobytes()[:-1])


def test_stamp_and_frame_id_helpers():
    assert ns.stamp_to_float(1_700_000_000, 500_000_000) == pytest.approx(1.7e9 + 0.5)
    assert ns.frame_id_from_header("tf:123456", 1, 2) == 123456
    assert ns.frame_id_from_header("", 1, 2) == 1_000_000_002        # falls back to the stamp in ns
    assert ns.frame_id_from_header("camera_color_optical_frame", 3, 4) == 3_000_000_004


def test_frame_id_from_header_falls_back_when_the_counter_is_not_a_number():
    assert ns.frame_id_from_header("tf:x", 1, 2) == 1_000_000_002


def test_bounds_and_tolerances_from_parameters():
    params = {"bound_vx": 0.3, "bound_vy": 0.2, "bound_wz": 0.4, "bound_ee_x": [0.1, 0.5],
              "bound_ee_y": [0.0, 0.3], "bound_ee_z": [0.0, 0.35]}
    b = ns.bounds_from_params(params)
    assert b == Bounds(vx=0.3, vy=0.2, wz=0.4, ee_x=(0.1, 0.5), ee_y=(0.0, 0.3), ee_z=(0.0, 0.35))
    t = ns.tolerances_from_params({"ego_tolerance_s": 0.05, "d435i_tolerance_s": 0.02, "state_tolerance_s": 0.01})
    assert t == Tolerances(ego_s=0.05, d435i_s=0.02, state_s=0.01)
    with pytest.raises(ValueError):
        ns.bounds_from_params(dict(params, bound_ee_x=[0.5, 0.1]))


def test_episode_identity_defaults():
    ident = ns.episode_identity(session_id="go2-real", episode_id="", reset_epoch=0, now=1_700_000_123.9)
    assert ident == ("go2-real", "go2-real:1700000123", 1700000123)
    ident = ns.episode_identity(session_id="s", episode_id="custom", reset_epoch=5, now=0.0)
    assert ident == ("s", "custom", 5)
    with pytest.raises(ValueError):
        ns.episode_identity(session_id="", episode_id="", reset_epoch=0, now=1.0)


def test_start_gate_order_streams_policy_publishers_settle():
    fresh = {"ego": 99.95, "task_centric": 99.98, "state": 99.99}
    ok = dict(newest=fresh, mode="POLICY", tripped=False, now=100.0, policy_since=97.0, settle_s=2.0,
              other_cmd_vel_publishers=0)
    assert ns.start_gate(**ok) == ""
    # a missing or stale source comes first, every one named
    r = ns.start_gate(**dict(ok, newest={"ego": None, "task_centric": 98.5, "state": 99.99}))
    assert r.startswith("no recent") and "ego" in r and "task_centric" in r and "state" not in r
    # then the controller mode
    assert ns.start_gate(**dict(ok, mode=None)) == "controller not in POLICY (None)"
    assert ns.start_gate(**dict(ok, mode="STAND")) == "controller not in POLICY (STAND)"
    assert "latched" in ns.start_gate(**dict(ok, tripped=True))
    # then another cmd_vel publisher (the entry sequence still running)
    assert "1 other cmd_vel publisher" in ns.start_gate(**dict(ok, other_cmd_vel_publishers=1))
    # then the settle time counted from the first POLICY report
    assert "settling" in ns.start_gate(**dict(ok, policy_since=None))
    assert "settling" in ns.start_gate(**dict(ok, policy_since=98.5))
    assert ns.start_gate(**dict(ok, policy_since=98.5, settle_s=0.0)) == ""
    assert ns.start_gate(**dict(ok, policy_since=98.0)) == ""


# --- frame size against the RGB profile (dual-rgb-360p-v1: 640x360, both views) --------------------

def test_frame_size_problem_names_expected_and_received():
    assert ns.frame_size_problem(640, 360, "task_centric") == ""
    assert ns.frame_size_problem(640, 360, "ego") == ""
    problem = ns.frame_size_problem(1280, 720, "task_centric")
    assert "1280x720" in problem and "640x360" in problem and "dual-rgb-360p-v1" in problem
    assert "d435i_sim_camera.py" in problem and "--scale 1.0" in problem      # the full export, by hand
    assert "640x360" in ns.frame_size_problem(1280, 720, "ego") and "d435i_sim" not in ns.frame_size_problem(1280, 720, "ego")


def test_start_gate_says_why_a_camera_has_no_frames():
    newest = {"ego": 10.0, "task_centric": None, "state": 10.0}
    reason = ns.start_gate(newest=newest, mode="POLICY", tripped=False, now=10.1, policy_since=5.0, settle_s=2.0,
                           other_cmd_vel_publishers=0,
                           size_problems={"task_centric": "frames are 1280x720, dual-rgb-360p-v1 needs 640x360"})
    assert reason.startswith("no recent task_centric") and "frames are 1280x720" in reason
