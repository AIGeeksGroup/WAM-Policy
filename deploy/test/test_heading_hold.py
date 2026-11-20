"""HeadingHold: PI yaw-rate command that keeps the straight walk straight
(the three-leg gait yaws right ~1-2 deg/s on the real robot)."""

import math

import pytest

from wam_go2.control.heading_hold import HeadingHold, wrap_angle, yaw_from_quat


def quat_yaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))   # w, x, y, z


def test_yaw_from_quat_and_wrap():
    assert yaw_from_quat((1, 0, 0, 0)) == pytest.approx(0.0)
    assert yaw_from_quat(quat_yaw(0.7)) == pytest.approx(0.7)
    assert yaw_from_quat(quat_yaw(-2.5)) == pytest.approx(-2.5)
    assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle(-3.2) == pytest.approx(-3.2 + 2 * math.pi)


def test_disabled_is_passthrough():
    h = HeadingHold(enabled=False)
    assert h.update([0.15, 0.0, 0.0], 0.3) == [0.15, 0.0, 0.0]
    assert not h.active and h.event is None


def test_standing_still_does_not_hold():
    h = HeadingHold()
    for k in range(50):
        out = h.update([0.0, 0.0, 0.0], 0.01 * k)      # robot yawing while standing
        assert out == [0.0, 0.0, 0.0]
    assert not h.active


def test_latches_on_walk_start_and_counters_drift():
    h = HeadingHold(kp=1.0, ki=0.0, max_wz=0.3, deadband_deg=1.0)
    out = h.update([0.15, 0.0, 0.0], 0.20)
    assert h.event == "latched" and h.yaw_ref == pytest.approx(0.20)
    assert out == [0.15, 0.0, 0.0]                       # zero error at latch
    # the robot drifts right (yaw decreasing) by 5 deg: command a left turn
    out = h.update([0.15, 0.0, 0.0], 0.20 - math.radians(5.0))
    expected = 1.0 * math.radians(5.0 - 1.0)             # P on the error beyond the deadband
    assert out[2] == pytest.approx(expected)
    assert out[0] == 0.15 and out[1] == 0.0
    # inside the deadband: no P action
    out = h.update([0.15, 0.0, 0.0], 0.20 - math.radians(0.5))
    assert out[2] == pytest.approx(0.0)


def test_saturation_and_integral_antiwindup():
    h = HeadingHold(kp=1.0, ki=0.3, max_wz=0.3, deadband_deg=0.0, dt=0.01)
    h.update([0.15, 0.0, 0.0], 0.0)
    for _ in range(200):
        out = h.update([0.15, 0.0, 0.0], -1.0)           # 57 deg error: saturated
        assert out[2] == pytest.approx(0.3)
    assert abs(h.integral) < 1e-9                        # did not wind up while saturated
    # a small steady error is integrated away over time
    h2 = HeadingHold(kp=1.0, ki=0.3, max_wz=0.3, deadband_deg=1.0, dt=0.01)
    h2.update([0.15, 0.0, 0.0], 0.0)
    e = math.radians(0.8)                                # inside the deadband: P does nothing
    wz = [h2.update([0.15, 0.0, 0.0], -e)[2] for _ in range(300)]
    assert wz[0] == pytest.approx(0.3 * e * 0.01)
    assert wz[-1] == pytest.approx(0.3 * e * 3.0)        # integral alone, growing
    assert wz[-1] > 0.0                                  # right drift -> left turn command


def test_operator_yaw_input_releases_and_relatches():
    h = HeadingHold()
    h.update([0.15, 0.0, 0.0], 0.0)
    assert h.active
    out = h.update([0.15, 0.0, 0.2], 0.1)                # operator steers
    assert out == [0.15, 0.0, 0.2] and not h.active and h.event == "released"
    out = h.update([0.15, 0.0, 0.0], 0.5)                # straight again: new reference
    assert h.event == "latched" and h.yaw_ref == pytest.approx(0.5) and out[2] == 0.0


def test_stop_releases_and_reset_clears():
    h = HeadingHold()
    h.update([0.15, 0.0, 0.0], 0.0)
    h.update([0.15, 0.0, 0.0], -0.1)
    assert h.integral != 0.0
    out = h.update([0.0, 0.0, 0.0], -0.1)
    assert out == [0.0, 0.0, 0.0] and not h.active and h.integral == 0.0
    h.update([0.15, 0.0, 0.0], 1.0)
    h.reset()
    assert not h.active and h.integral == 0.0 and h.last_wz == 0.0


def test_wraparound_error():
    h = HeadingHold(kp=1.0, ki=0.0, max_wz=0.5, deadband_deg=0.0)
    h.update([0.15, 0.0, 0.0], math.pi - 0.05)           # reference near +pi
    out = h.update([0.15, 0.0, 0.0], -math.pi + 0.05)    # yaw wrapped to -pi side: only 0.1 rad off
    assert out[2] == pytest.approx(-0.1)


def test_backward_walk_also_holds():
    h = HeadingHold(kp=1.0, ki=0.0, deadband_deg=0.0)
    h.update([-0.15, 0.0, 0.0], 0.0)
    assert h.active
    assert h.update([-0.15, 0.0, 0.0], -0.1)[2] == pytest.approx(0.1)


def test_bad_gains_rejected():
    with pytest.raises(ValueError):
        HeadingHold(kp=-1.0)


# --- in place: stepping in place, the robot slowly turns right ------------------------------------
# The three-leg gait turns right with a ZERO command too (real -0.9..-1.0 deg/s, MuJoCo -1.1 deg/s).
# WAM runs: walking hold OFF (training had none), in-place hold ON (the training sim was frozen
# while the model predicted; the real robot steps in place then).

from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def wam_hold(**kw):
    return HeadingHold(enabled=False, in_place=True, **kw)


def test_in_place_latches_at_zero_command_and_counters_the_right_turn():
    h = wam_hold(kp=1.0, ki=0.0, deadband_deg=1.0)
    assert h.update([0.0, 0.0, 0.0], 0.30) == [0.0, 0.0, 0.0]
    assert h.event == "latched" and h.mode == "in_place" and h.yaw_ref == pytest.approx(0.30)
    out = h.update([0.0, 0.0, 0.0], 0.30 - math.radians(4.0))     # turned right by 4 deg
    assert out[2] == pytest.approx(math.radians(3.0)) and out[:2] == [0.0, 0.0]
    # a tiny command below the thresholds is kept and the correction added to it
    out = h.update([0.005, -0.004, 0.003], 0.30 - math.radians(4.0))
    assert out[0] == 0.005 and out[1] == -0.004 and out[2] == pytest.approx(0.003 + math.radians(3.0))


@pytest.mark.parametrize("cmd", [[0.05, 0.0, 0.0], [0.0, -0.05, 0.0], [0.0, 0.0, 0.1], [-0.1, 0.0, 0.0]])
def test_in_place_any_command_releases_and_passes_unchanged(cmd):
    h = wam_hold()
    h.update([0.0, 0.0, 0.0], 0.0)
    h.update([0.0, 0.0, 0.0], -0.2)
    out = h.update(cmd, -0.2)
    assert out == cmd and h.event == "released" and not h.active


def test_wam_setting_never_holds_while_walking():
    h = wam_hold()
    for k in range(20):
        assert h.update([0.15, 0.0, 0.0], -0.01 * k) == [0.15, 0.0, 0.0]
    assert not h.active


def test_in_place_integral_is_carried_to_the_next_pause_and_cleared_by_reset():
    h = wam_hold(kp=1.0, ki=0.3, deadband_deg=1.0, dt=0.01)
    h.update([0.0, 0.0, 0.0], 0.0)
    for _ in range(300):                                          # a 3 s pause, 2 deg behind
        h.update([0.0, 0.0, 0.0], -math.radians(2.0))
    carried = h.integral
    assert carried > 0.0
    h.update([0.1, 0.0, 0.0], -0.5)                               # a chunk: released
    assert h.event == "released" and not h.active
    out = h.update([0.0, 0.0, 0.0], -0.5)                         # next pause: new reference, same bias
    assert h.event == "latched" and h.yaw_ref == pytest.approx(-0.5)
    assert h.integral == pytest.approx(carried + 0.0) and out[2] == pytest.approx(carried)
    h.reset()                                                     # POLICY entry: nothing carried over
    h.update([0.0, 0.0, 0.0], 0.0)
    assert h.integral == 0.0


def test_switching_between_walking_and_in_place_relatches():
    h = HeadingHold(enabled=True, in_place=True)
    h.update([0.15, 0.0, 0.0], 0.10)
    assert h.mode == "walking" and h.yaw_ref == pytest.approx(0.10)
    h.update([0.0, 0.0, 0.0], 0.25)                               # stopped: hold where it stands now
    assert h.event == "latched" and h.mode == "in_place" and h.yaw_ref == pytest.approx(0.25)


def test_wam_runs_switch_the_in_place_hold_on_and_the_walking_hold_off():
    script = (REPO / "scripts/demo/wam_task.sh").read_text(encoding="utf-8")
    assert 'WAM_HEADING_HOLD="${WAM_HEADING_HOLD:-false}"' in script
    assert 'WAM_IN_PLACE_HOLD="${WAM_IN_PLACE_HOLD:-true}"' in script
    assert "heading_hold:=$WAM_HEADING_HOLD heading_hold_in_place:=$WAM_IN_PLACE_HOLD" in script
    launch = (REPO / "launch/wam_go2.launch.py").read_text(encoding="utf-8")
    assert '"heading_hold_in_place"' in launch and "in_place_arg," in launch
    node = (REPO / "wam_go2/nodes/controller_node.py").read_text(encoding="utf-8")
    assert 'declare("heading_hold_in_place", False)' in node
    assert 'in_place=bool(p("heading_hold_in_place")))' in node
    yaml = (REPO / "config/wam_go2.yaml").read_text(encoding="utf-8")
    assert "heading_hold_in_place: false" in yaml                # the node default: unchanged
