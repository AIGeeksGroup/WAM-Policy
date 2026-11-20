"""Failure injection for the production node methods, without a ROS runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_go2.constants import DEFAULT_JOINT_POS
from wam_go2.estimator.state_estimator import StateEstimator
from wam_go2.go2.lowstate import Go2State
from wam_go2.safety.hardware_health import hardware_fault_reason
from wam_go2.safety.safety_manager import SafetyManager


def node_methods(*names):
    path = Path(__file__).resolve().parents[1] / "wam_go2/nodes/controller_node.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RamboGo2Controller")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    env = dict(np=np, time=time, DAMP="DAMP", STAND="STAND", POLICY="POLICY",
               hardware_fault_reason=hardware_fault_reason)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), env)
    return {name: env[name] for name in names}


class RequestNode:
    locals().update(node_methods("_on_fsm_cmd", "_apply_fsm_request"))

    def __init__(self):
        self.mode = "DAMP"
        self._fsm_request = None
        self._damp_requested = False
        self.safety = SafetyManager()
        self.movements = []

    def _to_damp(self, reason):
        self.mode = "DAMP"

    def _to_stand(self):
        self.mode = "STAND"
        self.movements.append("STAND")

    def _to_policy(self):
        self.mode = "POLICY"
        self.movements.append("POLICY")

    def get_logger(self):
        return SimpleNamespace(info=lambda *_: None)


@pytest.mark.parametrize("commands", [("DAMP", "STAND"), ("STAND", "DAMP"), ("DAMP", "REARM", "STAND"),
                                      ("DAMP", "POLICY", "STAND")])
def test_topic_damp_survives_later_requests(commands):
    node = RequestNode()
    for cmd in commands:
        node._on_fsm_cmd(SimpleNamespace(data=cmd))
    node._apply_fsm_request()
    assert node.mode == "DAMP" and not node.movements
    node._apply_fsm_request()
    assert node.mode == "DAMP"  # no queued motion after the damp


def update_estimator(est, now, accel=None, gyro=None):
    return est.update(.002, np.array([1., 0, 0, 0]),
                      np.zeros(3) if gyro is None else gyro,
                      np.array([0., 0, 9.81]) if accel is None else accel,
                      DEFAULT_JOINT_POS, np.zeros(12), np.ones(4, dtype=bool), now=now)


def test_invalid_estimation_does_not_refresh_success_time():
    est = StateEstimator()
    assert update_estimator(est, 10.0)
    original = est.x.copy()
    for i in range(1, 1001):
        assert not update_estimator(est, 10.0 + .002*i, accel=np.array([np.nan, 0, 9.81]))
    np.testing.assert_array_equal(est.x, original)
    assert not est.valid and est.consecutive_failures == 1000
    assert est.last_success_time == 10.0
    assert est.age(12.0) == pytest.approx(2.0)
    assert update_estimator(est, 12.002)
    assert est.valid and est.consecutive_failures == 0


def check(sm, **kw):
    return sm.check(.002, np.array([0., 0, -1.]), np.zeros(12), np.zeros(12), **kw)


def test_quad_rejects_bad_gyro_and_stale_estimator_then_latches():
    sm = SafetyManager()
    assert not check(sm, base_ang_vel_b=np.array([0., np.nan, 0.]))
    assert "non-finite" in sm.reason
    sm.rearm()
    assert check(sm, estimator_age=.002)
    assert not check(sm, estimator_age=.051)
    assert "estimator" in sm.reason
    assert not check(sm, estimator_age=.0)  # restored input is not REARM


@pytest.mark.parametrize("gap", [.05, .1, .17])
def test_source_stale_runs_trip_before_250ms(gap):
    sm = SafetyManager(lowstate_timeout=.25)
    assert not sm.check(gap, np.array([0., 0, -1.]), np.zeros(12), np.zeros(12), active=True)
    assert "source" in sm.reason


def test_runtime_link_failure_and_host_pause_have_distinct_reasons():
    sm = SafetyManager()
    assert not check(sm, active=True, wired_ok=False)
    assert "wired" in sm.reason
    sm.rearm()
    assert not check(sm, active=True, control_gap=.051)
    assert "control loop" in sm.reason
    sm.rearm()
    assert check(sm, active=True, control_gap=.01, wired_ok=True)


def test_good_wired_stream_and_short_loss_tolerated():
    sm = SafetyManager(lowstate_timeout=.25)
    for age in [.002, .008, .02, .049, .002]:
        assert sm.check(age, np.array([0., 0, -1.]), np.zeros(12), np.zeros(12),
                        active=True, control_gap=.01, wired_ok=True, estimator_age=.002)


def test_first_policy_failure_saves_raw_evidence_and_never_sends_active_command(tmp_path):
    methods = node_methods('_control_tick', '_save_policy_trace', '_to_damp', '_publish_mode')
    for method in methods.values():
        method.__globals__.update(os=__import__('os'), json=__import__('json'),
            Float32MultiArray=lambda **kw: SimpleNamespace(**kw),
            String=lambda: SimpleNamespace(data=''), yaw_from_quat=lambda _: 0.)
    class FaultNode:
        locals().update(methods)
    node = FaultNode()
    node.mode = 'POLICY'
    node._last_control_time = None
    state = Go2State()
    sends = []
    node.go2 = SimpleNamespace(last_state=state, state_count=1, source_fault=None,
        seconds_since_last_state=lambda: .002, send_damping=lambda: sends.append('DAMP'),
        send=lambda *_: sends.append('ACTIVE'))
    node._freshness = SimpleNamespace(update=lambda _: None, last_stale=False)
    node._wired = SimpleNamespace(update=lambda *_: None, ok=True)
    node._released_sport_mode = True
    node._first_state_seen = True
    node._apply_fsm_request = lambda: False
    node._make_robot_state = lambda _: SimpleNamespace(projected_gravity_b=np.array([0.,0,-1.]),
        joint_pos=np.zeros(12), joint_vel=np.zeros(12), base_lin_vel_b=np.zeros(3),
        base_height=.3, base_ang_vel_b=np.zeros(3))
    node.safety = SafetyManager()
    node.estimator = SimpleNamespace(age=lambda _: .002)
    node._topic_velocity = lambda: np.zeros(3)
    node._heading_hold = SimpleNamespace(update=lambda vel,_: vel, event=None)
    node._force_step = lambda *_: None
    def broken(*_):
        raise ValueError('policy action contains NaN')
    node.controller = SimpleNamespace(set_commands=lambda **_: None, step=broken,
                                     last_raw_action=np.full(18,np.nan))
    node._policy_time = 0.
    node._policy_fault = None
    node._trace_dir = str(tmp_path)
    node._trace = []
    node._trace_metadata = '{"test":true}'
    node._set_ee_compliance = lambda _: None
    node._track_mode_exit = lambda: None
    node._mode_pub = SimpleNamespace(publish=lambda _: None)
    node.get_logger = lambda: SimpleNamespace(info=lambda *_: None, warn=lambda *_: None,
                                              error=lambda *_: None)
    node._control_tick()
    assert sends == ['DAMP'] and node.mode == 'DAMP' and node.safety.tripped
    paths = list(tmp_path.glob('policy_trace_*.npz'))
    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=False) as trace:
        assert np.isnan(trace['fault_raw_action']).all()
        assert 'NaN' in str(trace['fault_reason'])
        assert 'POLICY fault' in str(trace['exit_reason'])


def test_release_confirmation_progresses_without_state_or_while_stopped():
    method = node_methods('_control_tick')['_control_tick']
    for has_state in (False, True):
        calls = []
        node = SimpleNamespace(mode='DAMP', _last_control_time=None,
            _released_sport_mode=False, _first_state_seen=False,
            motion_switcher=SimpleNamespace(poll_release=lambda _: calls.append('poll') or True),
            go2=SimpleNamespace(last_state=Go2State() if has_state else None, state_count=1,
                               send_damping=lambda: calls.append('damp')),
            _freshness=SimpleNamespace(update=lambda _: None),
            _wired=SimpleNamespace(update=lambda *_: None, ok=True),
            _apply_fsm_request=lambda: True,
            _track_mode_exit=lambda: None, _publish_mode=lambda: None)
        method(node)
        assert node._released_sport_mode
        assert calls == (['poll', 'damp'] if has_state else ['poll'])
