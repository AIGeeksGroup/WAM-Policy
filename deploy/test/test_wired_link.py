"""WiredLinkMonitor: the LowState stream must arrive on the wired NIC
(the host and the robot talk over the Ethernet cable only)."""

import os

import pytest

from wam_go2.safety.wired_link import WiredLinkMonitor


def fake_nic(tmp_path, nic, rx_bytes):
    d = tmp_path / nic / "statistics"
    d.mkdir(parents=True, exist_ok=True)
    (d / "rx_bytes").write_text(f"{rx_bytes}\n")


def set_rx(tmp_path, nic, rx_bytes):
    (tmp_path / nic / "statistics" / "rx_bytes").write_text(f"{rx_bytes}\n")


def test_disabled_when_nic_empty_or_none():
    for nic in ("", "none", "NONE", " off "):
        m = WiredLinkMonitor(nic)
        assert not m.enabled and m.ok is True and m.reason == ""
        m.update(0.0, 10)
        assert m.ok is True


def test_missing_nic_is_not_ok(tmp_path):
    m = WiredLinkMonitor("enxMISSING", sysfs=str(tmp_path))
    assert m.ok is None
    m.update(0.0, 0)
    assert m.ok is False and "not found" in m.reason


def test_stream_on_the_wire_is_ok(tmp_path):
    fake_nic(tmp_path, "enxW", 1_000_000)
    m = WiredLinkMonitor("enxW", min_mbps=0.3, period=1.0, sysfs=str(tmp_path))
    m.update(0.0, 0)
    assert m.ok is None                      # first sample only
    set_rx(tmp_path, "enxW", 1_650_000)      # 0.65 MB in 1 s
    m.update(1.0, 500)
    assert m.ok is True and m.reason == ""
    assert m.rate_mbps == pytest.approx(0.65)


def test_frames_flowing_but_nothing_on_the_wire_is_refused(tmp_path):
    fake_nic(tmp_path, "enxW", 5_000)
    m = WiredLinkMonitor("enxW", min_mbps=0.3, period=1.0, sysfs=str(tmp_path))
    m.update(0.0, 0)
    set_rx(tmp_path, "enxW", 6_000)          # 1 kB in 1 s: WiFi carries the stream
    m.update(1.0, 500)
    assert m.ok is False
    assert "NOT on the robot cable" in m.reason and "500 Hz" in m.reason


def test_no_frames_means_no_verdict_yet(tmp_path):
    fake_nic(tmp_path, "enxW", 0)
    m = WiredLinkMonitor("enxW", sysfs=str(tmp_path))
    m.update(0.0, 0)
    m.update(1.5, 3)                         # 3 frames: robot not streaming
    assert m.ok is None and "not verified" in m.reason


def test_samples_at_the_period_not_every_tick(tmp_path):
    fake_nic(tmp_path, "enxW", 0)
    m = WiredLinkMonitor("enxW", period=1.0, sysfs=str(tmp_path))
    m.update(0.0, 0)
    for k in range(1, 99):
        set_rx(tmp_path, "enxW", k * 10_000)
        m.update(k * 0.01, k * 5)
        assert m.ok is None                  # inside the first period
    set_rx(tmp_path, "enxW", 1_000_000)
    m.update(1.0, 500)
    assert m.ok is True


def test_verdict_flips_when_the_path_changes(tmp_path):
    fake_nic(tmp_path, "enxW", 0)
    m = WiredLinkMonitor("enxW", sysfs=str(tmp_path))
    m.update(0.0, 0)
    set_rx(tmp_path, "enxW", 700_000)
    m.update(1.0, 500)
    assert m.ok is True
    m.update(2.0, 1000)                      # counter frozen: stream moved off the wire
    assert m.ok is False
