"""wam_go2/wam/episode.py: the client-side episode driver (reset -> predict -> execute -> feedback).

Exercised against a fake real-time runtime with a virtual clock: cameras and
controller states are generated as the clock advances, commands are logged with
the virtual time they were issued, and the policy is the protocol-faithful mock
service called in-process. What is checked is the contract the robot node then
has to honour: commands on a 20 ms schedule, the logical tick clock frozen while
the policy computes, feedback samples at the +80/+160/+240/+320 ms instants,
holds between chunks, and the terminal path for every stop reason.
"""
import numpy as np
import pytest

pytest.importorskip("qlm_bench.live", reason="QLM-Bench live package not on PYTHONPATH")

from wam_go2.wam import episode as ep  # noqa: E402
from wam_go2.wam import mock_service as ms  # noqa: E402
from wam_go2.wam.limits import Bounds  # noqa: E402
from wam_go2.wam.observation import (PROFILE_HW, FrameRecord,  # noqa: E402
                                       ObservationBuffers, StateRecord, Tolerances)

INSTRUCTION = "Push the box inside the outlined area."


class FakeRuntime:
    """Virtual-clock world: free-running cameras and a 100 Hz controller state."""

    def __init__(self, *, ego_period=0.070, d435i_period=1 / 30, state_period=0.010,
                 mode_of_time=None, ego_stop_at=None):
        self.t = 100.0
        self.buffers = ObservationBuffers(keep_s=3.0)
        self.d435i_hw = PROFILE_HW
        self.ego_period, self.d435i_period, self.state_period = ego_period, d435i_period, state_period
        self.mode_of_time = mode_of_time or (lambda t: "POLICY")
        self.ego_stop_at = ego_stop_at
        self._next = {"ego": self.t - 0.5, "task_centric": self.t - 0.5, "state": self.t - 0.5}
        self._ids = {"ego": 100, "task_centric": 500}
        self.commands = []          # (time, native9 or None)
        self.events = []
        self._interrupt_at = None
        self._generate(self.t)

    # world ----------------------------------------------------------------
    def _generate(self, until):
        while self._next["ego"] <= until:
            t = self._next["ego"]
            if self.ego_stop_at is None or t < self.ego_stop_at:
                self._ids["ego"] += 1
                self.buffers.add_frame("ego", FrameRecord(t, self._ids["ego"], np.full((360, 640, 3), 10, np.uint8), t + 0.05))
            self._next["ego"] += self.ego_period
        while self._next["task_centric"] <= until:
            t = self._next["task_centric"]
            self._ids["task_centric"] += 1
            self.buffers.add_frame("task_centric", FrameRecord(t, self._ids["task_centric"],
                                                               np.full((*self.d435i_hw, 3), 20, np.uint8), t + 0.08))
            self._next["task_centric"] += self.d435i_period
        while self._next["state"] <= until:
            t = self._next["state"]
            self.buffers.add_state(StateRecord(time=t, height=0.30, quat_wxyz=(1, 0, 0, 0), lin_vel_b=(0.1, 0, 0),
                                               ang_vel_b=(0, 0, 0), mode=self.mode_of_time(t), fresh=True,
                                               action18=[0.5] * 18))
            self._next["state"] += self.state_period

    # Runtime protocol ----------------------------------------------------------
    def now(self):
        return self.t

    def sleep_until(self, t):
        if t > self.t:
            self.t = t
            self._generate(self.t)

    def command(self, native9):
        self.commands.append((self.t, None if native9 is None else list(native9)))

    def step_ok(self):
        mode = self.mode_of_time(self.t)
        return (mode == "POLICY", "" if mode == "POLICY" else f"controller mode {mode}")

    def ready(self, grid_wall, tolerances):
        return self.buffers.ready(grid_wall, tolerances)

    def resolve(self, **kw):
        return self.buffers.resolve(**kw)

    def last_action18(self):
        return [0.5] * 18

    def latest_ego_time(self):
        return self.buffers.latest_ego_time()

    def record(self, event, payload):
        self.events.append((event, payload))

    def interrupted(self):
        return self._interrupt_at is not None and self.t >= self._interrupt_at


def make(policy=None, *, latency_s=0.0, runtime=None, max_chunks=3, fail_on=None, bounds=None, **kw):
    runtime = runtime or FakeRuntime()
    service = ms.MockPolicyService(policy or ms.ConstantPolicy(vx=0.15, ee=(0.44, 0.142, 0.16)),
                                   instruction=INSTRUCTION, fail_on=fail_on)
    client = ms.LocalClient(service, latency_s=latency_s,
                            advance_clock=lambda dt: runtime.sleep_until(runtime.now() + dt))
    driver = ep.EpisodeDriver(runtime, client, session_id="go2-test", episode_id="ep-1", reset_epoch=1,
                              instruction=INSTRUCTION, policy_seed=7, tolerances=Tolerances(),
                              bounds=bounds or Bounds(), sample_wait_s=0.30, max_chunks=max_chunks, **kw)
    return runtime, service, client, driver


def test_happy_path_three_chunks_then_budget():
    rt, svc, client, drv = make()
    reply = drv.reset_episode()
    assert reply["status"] == "OBSERVED" and drv.history.tick == 0
    result = drv.run()
    assert result.reason == "chunk_budget" and result.exit_code == 0
    assert result.chunks == 3 and result.commands_completed == 48 and drv.clock.tick == 480
    cmds = [c for c in rt.commands if c[1] is not None]
    holds = [c for c in rt.commands if c[1] is None]
    assert len(cmds) == 48
    # the session rounds predictions to float32 (as the real one does)
    assert cmds[0][1] == pytest.approx([0.15, 0.0, 0.0, 0.44, 0.142, 0.16, 0.0, 0.0, 0.0], abs=1e-6)
    gaps = np.diff([c[0] for c in cmds[:16]])
    assert np.allclose(gaps, 0.02)
    assert len(holds) >= 4            # initial hold + after every chunk
    # the service saw the whole protocol: reset, 3 predictions, 3 feedback commits, end
    kinds = [e["event"] for e in svc.events]
    assert kinds == ["reset", "prediction", "feedback_commit", "prediction", "feedback_commit",
                     "prediction", "feedback_commit", "end"]
    fb = [e for e in svc.events if e["event"] == "feedback_commit"]
    assert [s["tick"] for s in fb[0]["samples"]] == [40, 80, 120, 160]
    assert [s["tick"] for s in fb[2]["samples"]] == [360, 400, 440, 480]
    assert [m for m, _ in client.calls] == ["reset", "predict", "feedback", "predict", "feedback", "predict",
                                            "feedback", "end"]
    # frame ids keep increasing across the whole episode
    ids = [s["frame_ids"]["go2_ego"] for e in fb for s in e["samples"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_latency_is_a_hold_and_freezes_the_logical_clock():
    rt, svc, client, drv = make(latency_s=0.5, max_chunks=2)
    drv.reset_episode()
    result = drv.run()
    assert result.chunks == 2 and drv.clock.tick == 320
    starts = [e[1] for e in rt.events if e[0] == "execute"]
    assert len(starts) == 2
    # chunk 2 starts after feedback + predict latencies (1.0 s) plus the sample wait
    gap = starts[1]["start_wall"] - starts[0]["end_wall"]
    assert 1.0 <= gap <= 1.0 + 0.30 + 0.05
    holds = [e[1] for e in rt.events if e[0] == "hold"]
    assert holds and abs(holds[-1]["duration_s"] - gap) < 1e-6
    # during the hold the robot was commanded zero velocity (None) and no ticks advanced
    none_cmds = [c for c in rt.commands if c[1] is None and starts[0]["end_wall"] <= c[0] <= starts[1]["start_wall"]]
    assert none_cmds
    assert svc.events[-2]["event"] == "feedback_commit" and svc.events[-2]["timestamp_ns"] == 320 * 2_000_000


def test_operator_stop_mid_chunk_is_a_terminal_prefix():
    t_stop = [None]

    def mode(t):
        return "DAMP" if (t_stop[0] is not None and t >= t_stop[0]) else "POLICY"
    rt = FakeRuntime(mode_of_time=mode)
    rt, svc, client, drv = make(runtime=rt, max_chunks=5)
    drv.reset_episode()
    # stop 0.105 s into the second chunk: the first 5 commands (10 steps) completed, the 6th not

    def on_record(event, payload):
        if event == "execute" and payload.get("chunk_index") == 1 and t_stop[0] is None:
            t_stop[0] = payload["start_wall"] + 0.105
    drv.on_event = on_record
    result = drv.run()
    assert result.reason == "controller_left_policy" and result.exit_code == 2
    assert result.chunks == 1 and result.commands_completed == 16 + 5
    assert svc.state == "ENDED" and svc.events[-2]["event"] == "terminal"
    assert svc.events[-2]["completed_commands"] == 21 and svc.events[-1]["event"] == "end"
    assert drv.clock.tick == 160 + 50               # ten confirmed steps; the step ending after the stop is not
    assert rt.commands[-1][1] is None              # the robot was told to hold


def test_observation_gap_ends_the_episode_after_a_complete_chunk():
    rt = FakeRuntime(ego_stop_at=100.0 + 0.15)   # ego frames stop 150 ms into the first chunk
    rt, svc, client, drv = make(runtime=rt, max_chunks=5)
    drv.reset_episode()
    result = drv.run()
    assert result.reason.startswith("observation_gap") and result.exit_code == 6
    assert result.chunks == 0 and result.commands_completed == 16
    assert svc.state == "ENDED" and svc.events[-2]["event"] == "terminal"
    assert [m for m, _ in client.calls] == ["reset", "predict", "feedback", "end"]


def test_out_of_bounds_prediction_is_refused_before_any_command():
    rt, svc, client, drv = make(policy=ms.ConstantPolicy(vx=2.0), max_chunks=3)
    drv.reset_episode()
    n_before = len(rt.commands)
    result = drv.run()
    assert result.reason == "prediction_rejected" and result.exit_code == 8
    assert all(c[1] is None for c in rt.commands[n_before:])
    assert "base_vx" in result.detail
    assert client.calls[-1][0] == "end"


def test_rpc_failure_holds_and_ends():
    rt, svc, client, drv = make(fail_on=lambda r: r["method"] == "predict" and r["request_id"].endswith(":obs:1:160"),
                                max_chunks=3)
    drv.reset_episode()
    result = drv.run()
    assert result.reason == "rpc_error" and result.exit_code == 7 and result.chunks == 1
    assert rt.commands[-1][1] is None


def test_interrupt_during_a_chunk_is_a_terminal_prefix_with_exit_130():
    rt = FakeRuntime()
    rt, svc, client, drv = make(runtime=rt, max_chunks=3)
    drv.reset_episode()

    def on_record(event, payload):
        if event == "execute" and payload.get("chunk_index") == 0:
            rt._interrupt_at = payload["start_wall"] + 0.045
    drv.on_event = on_record
    result = drv.run()
    assert result.reason == "interrupted" and result.exit_code == 130
    assert result.commands_completed == 2 and svc.state == "ENDED"


def test_cold_start_is_anchored_to_the_newest_ego_frame():
    rt, svc, client, drv = make()
    drv.reset_episode()
    cold = [p for e, p in rt.events if e == "sample"][0]
    assert cold["tick"] == 0 and abs(cold["ego"]["offset_s"]) < 1e-9        # the anchor frame itself
    assert cold["grid_wall"] <= 100.0 and cold["grid_wall"] >= 100.0 - 0.071


def test_reset_requires_a_resolvable_cold_start():
    rt = FakeRuntime(ego_stop_at=99.0)       # no ego frames at all
    rt, svc, client, drv = make(runtime=rt)
    with pytest.raises(ep.EpisodeAbort) as e:
        drv.reset_episode()
    assert e.value.reason.startswith("observation_gap")


# --- what the packets carry: WAM-Policy's input, 360p, no body state --------------------------------

class _SpyService(ms.MockPolicyService):
    """The mock's state machine with only QLM's packet check, recording every packet."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.packets = []

    def _validate(self, packet, *, origin_ns, expected_ticks):
        from qlm_bench.live import validate_live_observation
        self.packets.append(packet)
        validate_live_observation(packet, origin_ns=origin_ns, expected_capture_ticks=expected_ticks)
        return []


def test_packets_have_no_body_state_and_the_360p_profile():
    from qlm_bench.contracts.command.validation import RGB_360_PROFILE_HASHES
    runtime = FakeRuntime(ego_period=1 / 60, d435i_period=1 / 60)
    service = _SpyService(ms.ConstantPolicy(vx=0.1, ee=(0.44, 0.142, 0.16)), instruction=INSTRUCTION)
    client = ms.LocalClient(service, advance_clock=lambda dt: runtime.sleep_until(runtime.now() + dt))
    driver = ep.EpisodeDriver(runtime, client, session_id="go2-test", episode_id="ep-1", reset_epoch=1,
                              instruction=INSTRUCTION, max_chunks=1)
    driver.reset_episode()
    result = driver.run()
    assert result.exit_code == 0 and result.chunks == 1
    assert [len(p["samples"]) for p in service.packets] == [1, 4]
    for packet in service.packets:
        assert packet["state_semantics"] == {}
        assert packet["observation"]["profiles"] == RGB_360_PROFILE_HASHES
        for sample in packet["samples"]:
            assert sample["root_state"] == {}
            assert {k: v.shape for k, v in sample["rgb"].items()} == {"go2_ego": (360, 640, 3),
                                                                      "d435i_rgb_task": (360, 640, 3)}


def test_the_episode_has_no_profile_choice():
    # one RGB profile only (dual-rgb-360p-v1)
    with pytest.raises(TypeError):
        make(rgb_profile="dual-rgb-720p-v1")
