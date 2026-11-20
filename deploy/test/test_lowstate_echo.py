"""Communication diagnostics must not label frozen samples as live 500 Hz."""
import importlib.util
from enum import IntEnum
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from wam_go2.go2.lowstate import Go2State


class FakeClockType(IntEnum):
    SYSTEM_TIME = 1
    ROS_TIME = 2
    STEADY_TIME = 3


class FakeClock:
    def __init__(self, *, clock_type=FakeClockType.SYSTEM_TIME):
        self.clock_type = clock_type


class FakeNode:
    def __init__(self, name):
        self.logs = []
        self.timers = []

    def create_subscription(self, *args):
        pass

    def create_timer(self, timer_period_sec, callback, callback_group=None, clock=None):
        timer = NS(timer_period_sec=timer_period_sec, callback=callback,
                   callback_group=callback_group, clock=clock)
        self.timers.append(timer)
        return timer

    def get_logger(self):
        return NS(info=self.logs.append, warn=self.logs.append)


@pytest.fixture
def echo(monkeypatch):
    rclpy = ModuleType("rclpy")
    node_module = ModuleType("rclpy.node")
    node_module.Node = FakeNode
    clock_module = ModuleType("rclpy.clock")
    clock_module.Clock, clock_module.ClockType = FakeClock, FakeClockType
    messages = ModuleType("unitree_go.msg")
    messages.LowState = object
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.clock", clock_module)
    monkeypatch.setitem(sys.modules, "unitree_go.msg", messages)
    path = Path(__file__).parents[1] / "wam_go2/nodes/lowstate_echo.py"
    spec = importlib.util.spec_from_file_location("wam_go2.nodes._test_echo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = NS(now=0.)
    monkeypatch.setattr(module, "time", NS(monotonic=lambda: clock.now), raising=False)
    monkeypatch.setattr(module, "parse_lowstate", lambda m: Go2State(tick_ms=m.tick))
    node = module.LowStateEcho()
    return node, clock


def feed(node, clock, ticks, interval=.002):
    for tick in ticks:
        clock.now += interval
        node._on_state(NS(tick=tick))


def report(node):
    node._print()
    return node.logs[-1].splitlines()[1]


def rate(text, name):
    match = re.search(rf"{name}=\s*([0-9.]+) Hz", text)
    assert match, f"missing numeric {name}: {text}"
    return float(match.group(1))


def test_received_rate_is_separate_from_source_tick_rate(echo):
    node, clock = echo
    feed(node, clock, range(0, 200, 2), interval=.02)
    text = report(node)
    assert rate(text, "rx_rate") == pytest.approx(50.)
    assert rate(text, "tick_rate") == pytest.approx(500., abs=.1)
    assert "LIVE" in text and "age=0.000s" in text


def test_stopped_stream_reports_zero_receive_rate_and_age(echo):
    node, clock = echo
    feed(node, clock, range(0, 500, 2))
    report(node)
    clock.now += 1.
    text = report(node)
    assert "STALE" in text and "age=1.000s" in text
    assert rate(text, "rx_rate") == 0.


def test_resume_does_not_reuse_old_interval_or_report_a_rate_spike(echo):
    node, clock = echo
    feed(node, clock, range(0, 500, 2))
    report(node)
    clock.now += 1.
    report(node)
    feed(node, clock, [1500])
    text = report(node)
    assert "LIVE" in text and "tick_rate=unknown" in text
    feed(node, clock, range(1502, 2002, 2))
    text = report(node)
    assert rate(text, "rx_rate") == pytest.approx(500., abs=.1)
    assert rate(text, "tick_rate") == pytest.approx(500., abs=.1)


def test_uint32_wrap_preserves_source_tick_estimate(echo):
    node, clock = echo
    feed(node, clock, [0xFFFFFFFC, 0xFFFFFFFE, 0, 2])
    assert rate(report(node), "tick_rate") == pytest.approx(500., abs=.1)


@pytest.mark.parametrize("ticks", [
    [1000000, 1000002, 0],   # source restarted
    [100, 102, 102],         # duplicate source sample
    [0, 2, 500000],          # impossible forward jump at this receive interval
    [3000000000, 3000000002, 0],  # reboot must not masquerade as uint32 wrap
])
def test_discontinuous_ticks_have_no_source_rate_until_new_interval(echo, ticks):
    node, clock = echo
    feed(node, clock, ticks)
    assert "tick_rate=unknown" in report(node)
    feed(node, clock, [ticks[-1] + 2])
    assert rate(report(node), "tick_rate") == pytest.approx(500., abs=.1)
