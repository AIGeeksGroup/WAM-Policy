"""Offline transport regressions; only ROS message/transport boundaries are fake."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import pytest


class Clock:
    now = 10.0
    def monotonic(self):
        return self.now
    def monotonic_ns(self):
        return int(self.now * 1e9)


class Request:
    def __init__(self):
        self.header = NS(identity=NS(id=0, api_id=0))
        self.parameter = ""


class FakeNode:
    def __init__(self, *args):
        self.sent = []
    def create_publisher(self, cls, topic, qos):
        return NS(publish=lambda msg: self.sent.append((topic, msg)))
    def create_subscription(self, cls, topic, callback, qos):
        return callback
    def get_clock(self):
        # Source age must not depend on ROS time.
        return NS(now=lambda: NS(nanoseconds=123_000_000_000))


@pytest.fixture
def modules(monkeypatch):
    messages = ModuleType("unitree_api.msg")
    messages.Request, messages.Response = Request, object
    monkeypatch.setitem(sys.modules, "unitree_api.msg", messages)
    go_msgs = ModuleType("unitree_go.msg")
    go_msgs.LowCmd, go_msgs.LowState = object, object
    monkeypatch.setitem(sys.modules, "unitree_go.msg", go_msgs)
    qos = ModuleType("rclpy.qos")
    for name in ("DurabilityPolicy", "HistoryPolicy", "ReliabilityPolicy"):
        setattr(qos, name, NS(KEEP_LAST=1, BEST_EFFORT=1, VOLATILE=1))
    qos.QoSProfile = lambda **kwargs: NS(**kwargs)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)
    loaded = []
    for name in ("motion_switcher", "interface"):
        fullname = "wam_go2.go2._test_" + name
        path = Path(__file__).parents[1] / "wam_go2" / "go2" / (name + ".py")
        spec = importlib.util.spec_from_file_location(fullname, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    clock = Clock()
    monkeypatch.setattr(loaded[0], "time", clock)
    if hasattr(loaded[1], "time"):
        monkeypatch.setattr(loaded[1], "time", clock)
    monkeypatch.setattr(loaded[1].Go2Interface, "_init_lowcmd", staticmethod(lambda: None))
    monkeypatch.setattr(loaded[1], "parse_lowstate", lambda msg: NS(tick_ms=msg.tick, value=msg.value))
    return *loaded, clock


def response(req, data='{"name": ""}', code=0, api_id=None):
    return NS(header=NS(identity=NS(id=req.header.identity.id,
                                  api_id=req.header.identity.api_id if api_id is None else api_id),
                        status=NS(code=code)), data=data)


@pytest.mark.parametrize("data,code", [
    ('{"name": ""}', 1), ("{}", 0), ("", 0), ("not-json", 0),
    ("[]", 0), ("null", 0), ('{"name": null}', 0), ('{"name": 3}', 0),
])
def test_invalid_mode_response_never_confirms_release(modules, data, code):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    def spin():
        client._on_response(response(node.sent[-1][1], data, code))
        clock.now += 0.001
    assert client.check_mode(spin) is None
    assert not client.release_until_clear(spin, max_tries=1, settle_s=0.0)


@pytest.mark.parametrize("name", ["", "sport_mode"])
def test_valid_mode_response_returns_exact_name(modules, name):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    def spin():
        client._on_response(response(node.sent[-1][1], '{"name": "' + name + '"}'))
        clock.now += 0.001
    assert client.check_mode(spin) == name


def test_missing_response_is_unknown(modules):
    motion, _, clock = modules
    client = motion.MotionSwitcherClient(FakeNode())
    assert client.check_mode(lambda: setattr(clock, "now", clock.now + 0.1), timeout=0.2) is None


def test_async_handoff_waits_for_confirming_check_after_release(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    assert hasattr(client, "start_release"), "confirmed asynchronous handoff is missing"
    client.start_release(now=clock.now, settle_s=0.0)
    assert not client.poll_release(now=clock.now)
    check = node.sent[-1][1]
    assert check.header.identity.api_id == motion.API_CHECK_MODE
    client._on_response(response(check, '{"name": "sport_mode"}'))
    assert not client.poll_release(now=clock.now)
    release = node.sent[-1][1]
    assert release.header.identity.api_id == motion.API_RELEASE_MODE
    client._on_response(response(release))
    assert not client.poll_release(now=clock.now)
    assert not client.poll_release(now=clock.now)
    check2 = node.sent[-1][1]
    assert check2.header.identity.api_id == motion.API_CHECK_MODE
    client._on_response(response(check2))
    assert client.poll_release(now=clock.now)
    assert client.release_state == "confirmed"


@pytest.mark.parametrize("data,code", [("{}", 0), ('{"name": ""}', 1), ("not-json", 0)])
def test_async_invalid_response_fails_closed(modules, data, code):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    assert hasattr(client, "start_release"), "confirmed asynchronous handoff is missing"
    client.start_release(now=clock.now, allow_release=False)
    client.poll_release(now=clock.now)
    client._on_response(response(node.sent[-1][1], data, code))
    assert not client.poll_release(now=clock.now)
    assert client.release_state == "failed"
    assert client.release_error
    assert all(req.header.identity.api_id == motion.API_CHECK_MODE for _, req in node.sent)


def test_async_timeout_is_bounded_and_late_response_cannot_confirm(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    assert hasattr(client, "start_release"), "confirmed asynchronous handoff is missing"
    client.start_release(now=clock.now, allow_release=False, timeout=0.1, max_tries=1)
    client.poll_release(now=clock.now)
    req = node.sent[-1][1]
    for _ in range(10):
        assert not client.poll_release(now=clock.now)
    assert len(node.sent) == 1
    assert not client.poll_release(now=clock.now + 0.11)
    assert client.release_state == "failed"
    client._on_response(response(req))
    assert not client.poll_release(now=clock.now + 0.12)


def test_async_confirmation_only_accepts_clear_and_never_sends_release(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    assert hasattr(client, "start_release"), "confirmed asynchronous handoff is missing"
    client.start_release(now=clock.now, allow_release=False)
    client.poll_release(now=clock.now)
    client._on_response(response(node.sent[-1][1]))
    assert client.poll_release(now=clock.now)
    assert len(node.sent) == 1


def interface(modules):
    _, module, clock = modules
    consumed = []
    go2 = module.Go2Interface(FakeNode(), state_callback=consumed.append)
    return go2, consumed, clock


def feed(go2, tick, value=0):
    go2._on_lowstate(NS(tick=tick, value=value))


def test_duplicate_ticks_never_refresh_state_or_estimator(modules):
    go2, consumed, clock = interface(modules)
    feed(go2, 12345, 1)
    for i in range(100):
        clock.now += 0.002
        feed(go2, 12345, 999)
    assert go2.state_count == 1
    assert len(consumed) == 1
    assert go2.last_state.value == 1
    assert go2.receive_count == 101
    assert go2.seconds_since_last_state() == pytest.approx(0.2)
    assert go2.seconds_since_last_receive() == pytest.approx(0.0)


def test_single_out_of_order_frame_is_ignored_then_forward_stream_resumes(modules):
    go2, consumed, clock = interface(modules)
    feed(go2, 1000)
    clock.now += 0.002
    feed(go2, 998, 999)
    assert go2.state_count == 1
    clock.now += 0.002
    feed(go2, 1002, 2)
    assert go2.state_count == 2
    assert len(consumed) == 2
    assert go2.source_fault is None
    assert go2.accepted_dt == pytest.approx(0.002)


def test_uint32_wrap_is_forward_progress_with_two_millisecond_dt(modules):
    go2, consumed, clock = interface(modules)
    feed(go2, 4294967295)
    clock.now += 0.002
    feed(go2, 1)
    assert len(consumed) == 2
    assert hasattr(go2, "accepted_dt"), "source dt must account for uint32 wrap"
    assert go2.accepted_dt == pytest.approx(0.002)
    assert go2.source_fault is None


def test_robot_reboot_latches_fault_and_cannot_self_recover(modules):
    go2, consumed, clock = interface(modules)
    feed(go2, 120000)
    for tick in (0, 2, 4, 120002):
        clock.now += 0.002
        feed(go2, tick)
    assert len(consumed) == 1
    assert go2.source_fault
    assert go2.state_count == 1
    assert go2.seconds_since_last_state() == pytest.approx(0.008)


def test_small_early_reboot_detected_from_progressing_regressed_stream(modules):
    go2, consumed, clock = interface(modules)
    feed(go2, 100)
    feed(go2, 0)
    feed(go2, 2)
    feed(go2, 102)
    assert len(consumed) == 1
    assert go2.source_fault


def test_normal_500hz_stream_updates_every_source_frame(modules):
    go2, consumed, clock = interface(modules)
    for i in range(100):
        clock.now += 0.002
        feed(go2, 20000 + i * 2)
    assert go2.state_count == 100
    assert len(consumed) == 100
    assert go2.seconds_since_last_state() == pytest.approx(0.0)
    assert hasattr(go2, "receive_count"), "receive and source progress must be separate"
    assert go2.receive_count == 100
    assert go2.source_fault is None


def test_check_response_received_after_deadline_is_unknown(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    def spin():
        clock.now += 0.2
        client._on_response(response(node.sent[-1][1]))
    assert client.check_mode(spin, timeout=0.1) is None


def test_async_response_received_after_deadline_cannot_confirm(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, allow_release=False, timeout=0.1, max_tries=1)
    req = node.sent[-1][1]
    clock.now += 0.2
    client._on_response(response(req))
    assert not client.poll_release(now=clock.now)
    assert client.release_state == "failed"


def test_async_silent_service_exhausts_bounded_release_attempts(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, max_tries=2, timeout=0.1, settle_s=0.0)
    for _ in range(30):
        clock.now += 0.11
        assert not client.poll_release(now=clock.now)
    assert client.release_state == "failed"
    releases = [req for topic, req in node.sent
                if topic == motion.MOTION_SWITCHER_REQUEST_TOPIC
                and req.header.identity.api_id == motion.API_RELEASE_MODE]
    assert len(releases) == 2
    assert not client._pending


def test_response_for_different_api_cannot_confirm_check(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, allow_release=False, timeout=0.1, max_tries=1)
    req = node.sent[-1][1]
    client._on_response(response(req, api_id=motion.API_RELEASE_MODE))
    assert not client.poll_release(now=clock.now)
    assert not client.poll_release(now=clock.now + 0.2)
    assert client.release_state == "failed"


def test_restarted_handoff_ignores_previous_clear_response(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, allow_release=False)
    old_request = node.sent[-1][1]
    client.start_release(now=clock.now, allow_release=False)
    client._on_response(response(old_request))
    assert not client.poll_release(now=clock.now)
    client._on_response(response(node.sent[-1][1]))
    assert client.poll_release(now=clock.now)


@pytest.mark.parametrize("previous,tick", [(3000000000, 0), (100000, 110000)])
def test_ambiguous_reboot_or_implausible_forward_jump_latches(modules, previous, tick):
    go2, consumed, clock = interface(modules)
    feed(go2, previous)
    clock.now += 0.002
    feed(go2, tick)
    assert go2.source_fault
    assert len(consumed) == 1


def test_unreceived_state_age_is_infinite(modules):
    go2, _, _ = interface(modules)
    assert go2.seconds_since_last_state() == float("inf")
    assert go2.seconds_since_last_receive() == float("inf")


@pytest.mark.parametrize("data,code,exit_code", [('{"name": ""}', 0, 0),
                                                ('{"name": ""}', 1, 1),
                                                ('{}', 0, 1), ('not-json', 0, 1)])
def test_release_cli_exit_status_requires_confirmed_empty_mode(modules, monkeypatch,
                                                              data, code, exit_code):
    motion, _, clock = modules
    node = FakeNode()
    node.get_logger = lambda: NS(info=lambda msg: None, error=lambda msg: None)
    node.destroy_node = lambda: None
    client = motion.MotionSwitcherClient(node)
    # Retain all production handoff logic; replace ROS transport with replies.
    ros = ModuleType("rclpy")
    ros.init = lambda **kwargs: None
    ros.ok = lambda: True
    ros.shutdown = lambda: None
    def spin_once(node, timeout_sec):
        clock.now += 0.1
        client._on_response(response(node.sent[-1][1], data, code))
    ros.spin_once = spin_once
    ros_node = ModuleType("rclpy.node")
    ros_node.Node = lambda name: node
    monkeypatch.setitem(sys.modules, "rclpy", ros)
    monkeypatch.setitem(sys.modules, "rclpy.node", ros_node)
    monkeypatch.setitem(sys.modules, "wam_go2.go2.motion_switcher", motion)
    path = Path(__file__).parents[1] / "wam_go2/nodes/release_sport_mode.py"
    spec = importlib.util.spec_from_file_location("wam_go2.nodes._test_release", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "MotionSwitcherClient", lambda node: client)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == exit_code


def test_mock_robot_confirms_check_mode_without_a_real_service(modules, monkeypatch):
    motion, _, clock = modules
    ros = ModuleType("rclpy")
    ros_node = ModuleType("rclpy.node")
    ros_node.Node = FakeNode
    std_msgs = ModuleType("std_msgs.msg")
    std_msgs.Bool, std_msgs.String, std_msgs.Float64MultiArray = object, object, object
    monkeypatch.setitem(sys.modules, "rclpy", ros)
    monkeypatch.setitem(sys.modules, "rclpy.node", ros_node)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs)
    monkeypatch.setitem(sys.modules, "wam_go2.go2.motion_switcher", motion)
    class Response(Request):
        def __init__(self):
            super().__init__()
            self.header.status = NS(code=0)
            self.data = ""
    monkeypatch.setattr(sys.modules["unitree_api.msg"], "Response", Response)
    monkeypatch.setattr(FakeNode, "create_timer", lambda *args: None, raising=False)
    path = Path(__file__).parent / "integration/mock_robot_test.py"
    spec = importlib.util.spec_from_file_location("test_transport_mock_robot", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    robot = fixture.MockRobot()
    client_node = FakeNode()
    client = motion.MotionSwitcherClient(client_node)
    client.start_release(now=clock.now, allow_release=False)
    assert hasattr(robot, "_on_motion_request"), "mock must answer ownership queries"
    robot._on_motion_request(client_node.sent[-1][1])
    topic, reply = robot.sent[-1]
    assert topic == motion.MOTION_SWITCHER_RESPONSE_TOPIC
    client._on_response(reply)
    assert client.poll_release(now=clock.now)


def _silent_step(client, clock, timeout):
    clock.now += timeout + 0.01
    assert not client.poll_release(now=clock.now)


def test_check_only_handoff_re_asks_after_a_lost_first_request(modules):
    """The very first CheckMode can be lost to DDS discovery (seen in
    mock_demo_test); check-only mode must not give up on that single silence.
    Silence is not evidence of an active service."""
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, allow_release=False, timeout=0.1)
    _silent_step(client, clock, 0.1)                 # first request lost
    assert client.release_state == "checking"
    requests = [req for _, req in node.sent]
    assert len(requests) == 2
    assert all(r.header.identity.api_id == motion.API_CHECK_MODE for r in requests)
    client._on_response(response(requests[-1]))
    assert client.poll_release(now=clock.now)
    assert client.release_state == "confirmed"
    assert len(node.sent) == 2                       # still never a ReleaseMode


def test_check_only_handoff_fails_after_max_tries_silences(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, allow_release=False, max_tries=3, timeout=0.1)
    for _ in range(3):
        _silent_step(client, clock, 0.1)
    assert client.release_state == "failed"
    assert len(node.sent) == 3
    assert all(r.header.identity.api_id == motion.API_CHECK_MODE for _, r in node.sent)


def test_release_allowed_gives_one_discovery_grace_before_the_official_loop(modules):
    motion, _, clock = modules
    node = FakeNode()
    client = motion.MotionSwitcherClient(node)
    client.start_release(now=clock.now, allow_release=True, max_tries=5, timeout=0.1, settle_s=0.0)
    _silent_step(client, clock, 0.1)                 # grace: ask again, no release yet
    kinds = [r.header.identity.api_id for t, r in node.sent
             if t == motion.MOTION_SWITCHER_REQUEST_TOPIC]
    assert kinds == [motion.API_CHECK_MODE, motion.API_CHECK_MODE]
    _silent_step(client, clock, 0.1)                 # second silence: fallback + ReleaseMode
    kinds = [r.header.identity.api_id for t, r in node.sent
             if t == motion.MOTION_SWITCHER_REQUEST_TOPIC]
    assert kinds[-1] == motion.API_RELEASE_MODE
    assert client.release_state == "releasing"
