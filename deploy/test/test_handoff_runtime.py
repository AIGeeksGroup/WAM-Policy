"""Production tick/FSM and shutdown behavior with no ROS or robot transport."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_runtime_faults import node_methods
from wam_go2.constants import DEFAULT_JOINT_POS
from wam_go2.go2.lowstate import Go2State
from wam_go2.safety.safety_manager import SafetyManager


class TickNode:
    locals().update(node_methods("_control_tick", "_on_fsm_cmd",
                                "_apply_fsm_request", "_to_damp", "_to_stand",
                                "_wired_ok_or_refuse"))

    def __init__(self, *, confirmed=True, has_state=True, mode="DAMP"):
        self.sends, self.messages, self.polls = [], [], []
        self.mode = mode
        self._released_sport_mode = confirmed
        # confirmed + state models an already running node; other combinations
        # model startup before its first usable LowState tick.
        self._first_state_seen = confirmed and has_state
        self.motion_switcher = NS(release_state="confirmed" if confirmed else "checking",
                                  poll_release=self.poll_release)
        self.state = Go2State(q=DEFAULT_JOINT_POS.copy())
        self.state.motor_errors = np.zeros(12, dtype=np.uint32)
        self.go2 = NS(last_state=self.state if has_state else None, state_count=500,
                      source_fault=None, seconds_since_last_state=lambda: .002,
                      send_damping=lambda: self.sends.append("DAMP"))
        self.safety = SafetyManager()
        self._last_control_time = None
        self._fsm_request = None
        self._damp_requested = False
        self._freshness = NS(update=lambda _: None, last_stale=False)
        self._wired = NS(update=lambda *_: None, ok=True)
        self._make_robot_state = lambda _: NS(
            projected_gravity_b=np.array([0., 0., -1.]), joint_pos=self.state.q,
            joint_vel=np.zeros(12), base_lin_vel_b=np.zeros(3), base_height=.3,
            base_ang_vel_b=np.zeros(3))
        self.estimator = NS(age=lambda _: .002)
        self._track_mode_exit = lambda: None
        self._publish_mode = lambda: None
        self._standing = lambda _: True
        self._send_pose = lambda *_: self.sends.append("POSE")
        self._dt = .01
        self._stand_duration = 2.
        self._gather_progress = 1.
        self._stand_progress = 0.
        self._stand_start_q = self.state.q.copy()
        self._stand_hold_q = None
        self._stand_track_bad = 0
        self._topic_velocity = lambda: np.zeros(3)
        self._heading_hold = NS(update=lambda vel, _: vel, event=None)
        self._force_step = lambda *_: pytest.fail("fault reached policy computation")

    def poll_release(self, _):
        self.polls.append(self.motion_switcher.release_state)
        return self.motion_switcher.release_state == "confirmed"

    def get_logger(self):
        return NS(info=self.messages.append, warn=self.messages.append,
                  error=self.messages.append)

    def request(self, name):
        self._on_fsm_cmd(NS(data=name))


@pytest.fixture
def tick_node():
    # Real state and SafetyManager; fake only clock, estimator output and I/O.
    TickNode._control_tick.__globals__.update(
        time=NS(monotonic=lambda: 10.), DEFAULT_JOINT_POS=DEFAULT_JOINT_POS,
        Float32MultiArray=lambda **kw: NS(**kw), yaw_from_quat=lambda _: 0.)
    return TickNode


@pytest.mark.parametrize("release_state", ["checking", "failed"])
@pytest.mark.parametrize("command", [None, "DAMP", "STAND"])
def test_no_lowcmd_before_confirmed_handoff(tick_node, release_state, command):
    node = tick_node(confirmed=False)
    node.motion_switcher.release_state = release_state
    if command:
        node.request(command)
    for _ in range(3):
        node._control_tick()
    assert node.sends == []
    assert len(node.polls) == 3
    assert node.mode == "DAMP"


def test_waiting_commands_are_consumed_even_on_confirmation_tick(tick_node):
    node = tick_node(confirmed=False)
    node.request("STAND")
    node.motion_switcher.release_state = "confirmed"
    node._control_tick()
    node._control_tick()
    assert node._released_sport_mode
    assert node.mode == "DAMP" and "POSE" not in node.sends
    node.request("STAND")
    node._control_tick()
    assert node.mode == "STAND" and node.sends[-1] == "POSE"


def test_no_first_state_drops_pending_action_but_keeps_polling(tick_node):
    node = tick_node(confirmed=False, has_state=False)
    node.request("STAND")
    node._control_tick()
    assert node.sends == [] and node.polls == ["checking"]
    assert node._fsm_request is None
    node.motion_switcher.release_state = "confirmed"
    node._control_tick()
    node.go2.last_state = node.state
    node._control_tick()
    assert node.mode == "DAMP" and "POSE" not in node.sends


@pytest.mark.parametrize("command,tripped", [("STAND", False), ("REARM", True)])
def test_first_state_after_handoff_drops_earlier_requests(tick_node, command, tripped):
    node = tick_node(confirmed=False, has_state=False)
    if tripped:
        node.safety._trip("previous fault")
    node.request(command)
    node.motion_switcher.release_state = "confirmed"
    node._control_tick()
    assert node._released_sport_mode and node.sends == []
    node.go2.last_state = node.state
    node._control_tick()
    assert node.mode == "DAMP" and node.safety.tripped == tripped
    assert "POSE" not in node.sends
    node.request(command)
    node._control_tick()
    if command == "STAND":
        assert node.mode == "STAND" and node.sends[-1] == "POSE"
    else:
        assert not node.safety.tripped


def test_damp_request_wins_over_a_later_stand(tick_node):
    node = tick_node(mode="STAND")
    node.request("DAMP")
    node.request("STAND")
    node._control_tick()
    assert node._exit_reason == "topic request"
    assert node.mode == "DAMP" and node.sends == ["DAMP"]


@pytest.mark.parametrize("command,tripped", [("STAND", False), ("REARM", True)])
def test_first_state_discards_topic_command_received_while_waiting(tick_node, command, tripped):
    node = tick_node(confirmed=False, has_state=False)
    if tripped:
        node.safety._trip("previous fault")
    node.motion_switcher.release_state = "confirmed"
    node._control_tick()
    node.request(command)
    node.go2.last_state = node.state
    node._control_tick()
    assert node.mode == "DAMP" and node.safety.tripped == tripped
    assert node._fsm_request is None and "POSE" not in node.sends


def test_confirmed_controller_keeps_estop_damping(tick_node):
    node = tick_node(mode="STAND")
    node.request("DAMP")
    node._control_tick()
    assert node.mode == "DAMP" and node.sends == ["DAMP"]


@pytest.mark.parametrize("confirmed,has_state,expected", [
    (False, False, []), (False, True, []), (True, False, []),
    (True, True, ["DAMP", "DAMP", "DAMP"]),
])
def test_shutdown_only_damps_after_handoff_and_first_state(confirmed, has_state, expected):
    source = Path(__file__).parents[1] / "wam_go2/nodes/controller_node.py"
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    sent, frozen = [], []
    node = NS(_released_sport_mode=confirmed, mode="DAMP",
              go2=NS(last_state=object() if has_state else None,
                     send_damping=lambda: sent.append("DAMP")),
              _save_policy_trace=lambda: None, destroy_node=lambda: None,
              get_logger=lambda: NS(info=lambda *_: None))
    env = dict(RamboGo2Controller=lambda: node, POLICY="POLICY",
               freeze_startup_heap=lambda: frozen.append(1) or (0, 0.0),     # wam_go2/realtime.py
               rclpy=NS(init=lambda **_: None, ok=lambda: False),
               signal=NS(SIGINT=2, SIGTERM=15, SIGHUP=1, signal=lambda *_: None),
               time=NS(sleep=lambda _: None), SignalHandlerOptions=NS(NO=0))
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), env)
    env["main"]()
    assert sent == expected
    assert frozen == [1]                                   # the start-up heap is frozen once, before spinning
