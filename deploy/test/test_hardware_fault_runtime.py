"""Hardware status takes precedence over every movement/rearm input."""
import pytest

from test_handoff_runtime import tick_node  # noqa: F401 - shared real tick harness


@pytest.mark.parametrize("mode", ["DAMP", "STAND", "POLICY"])
@pytest.mark.parametrize("fault", ["motor", "MCU", "battery"])
def test_hardware_fault_damps_and_latches_before_action(tick_node, mode, fault):
    node = tick_node(mode=mode)
    if fault == "motor":
        node.state.motor_errors[4] = 1
    else:
        node.state.bit_flag = 0x40 if fault == "MCU" else 0x10
    node._control_tick()
    assert node.safety.tripped and node.mode == "DAMP"
    assert node.sends == ["DAMP"]
    assert ("motor" if fault == "motor" else "bit_flag") in node.safety.reason


def test_fault_rearm_cannot_clear_latch_or_replay_after_recovery(tick_node):
    node = tick_node(mode="STAND")
    node.state.bit_flag = 0x40
    node._control_tick()
    node.request("REARM")
    node._control_tick()
    assert node.safety.tripped
    node.state.bit_flag = 0
    node._control_tick()
    assert node.safety.tripped and node.mode == "DAMP"
    node.request("STAND")
    node._control_tick()
    assert node.safety.tripped and node.mode == "DAMP"
    node.request("REARM")
    node._control_tick()
    assert not node.safety.tripped and node.mode == "DAMP"
    node.request("STAND")
    node._control_tick()
    assert node.mode == "STAND" and node.sends[-1] == "POSE"


def test_fault_consumes_motion_request(tick_node):
    node = tick_node()
    node.state.motor_errors[0] = 0x100
    node.request("STAND")
    node._control_tick()
    assert node.safety.tripped and node.mode == "DAMP"
    assert node._fsm_request is None
    node.state.motor_errors[0] = 0
    node.request("REARM")
    node._control_tick()
    node._control_tick()
    assert not node.safety.tripped and node.mode == "DAMP"
    assert "POSE" not in node.sends


def test_no_remote_is_not_a_fault(tick_node):
    node = tick_node(mode="STAND")
    node.state.bit_flag = 0x25          # what a Go2 without a connected remote reports
    node._control_tick()
    assert not node.safety.tripped and node.sends == ["POSE"]
    node.state.motor_errors[0] = 1
    node._control_tick()
    assert node.safety.tripped and node.mode == "DAMP" and node.sends[-1] == "DAMP"


def test_hardware_fault_before_handoff_latches_without_lowcmd(tick_node):
    node = tick_node(confirmed=False)
    node.state.bit_flag = 0x80
    node.request("REARM")
    node._control_tick()
    assert node.safety.tripped and node.sends == []
    node.motion_switcher.release_state = "confirmed"
    node._control_tick()
    assert node._released_sport_mode and node.safety.tripped
    assert node.sends == ["DAMP"]
