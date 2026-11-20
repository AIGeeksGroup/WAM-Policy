"""wam_go2/wam/mock_service.py: a protocol-faithful stand-in for the WAM-Policy policy server.

It must accept what the server's ``OnlineSession`` (``python -m wam_policy.inference.serve``)
accepts (same state machine, ``PolicyHistory``, packet validation, 4-sample feedback with the
expected ticks, idempotent duplicates, the 640x360 image profile) and refuse the same mistakes,
plus the mock's own rules: the reset instruction is the one it serves, and the packets carry no
body state (WAM-Policy takes none). Its replies have the server's shape. Needs the pinned QLM
live package.
"""
import threading

import numpy as np
import pytest

qlm = pytest.importorskip("qlm_bench.live", reason="QLM-Bench live package not found (set QLM_BENCH_SRC)")
from wam_go2.wam.ledger import CommandLedger  # noqa: E402
from qlm_bench.contracts.command.validation import PROFILE_HASHES  # noqa: E402
from qlm_bench.live import RPCClient, RemoteError, make_message, make_observation  # noqa: E402

from wam_go2.wam import mock_service as ms  # noqa: E402
from wam_go2.wam.rootlink import ROOT_FIELDS  # noqa: E402

IDS = dict(session_id="s", episode_id="e", reset_epoch=1)
INSTRUCTION = "Push the box inside the outlined area."
STATE_SEMANTICS = {"reference": "root_link", "quaternion": "xyzw", "velocity_frame": "world",
                   "projected_gravity_frame": "body", "ground_z": 0.0,
                   "ground_reference": "push-box-v2-plane-world-z0"}      # a packet that DOES carry state


def reset_message(epoch=1, instruction=INSTRUCTION):
    ids = dict(IDS, reset_epoch=epoch)
    return make_message("ResetRequest", dict(timeout_s=60.0, origin_sim_ns=0,
                        task=dict(id="push_box", version="mujoco-push_box-runtime-v1", instruction=instruction, language="en")),
                        message_id=f"reset:{epoch}", extensions={"policy_seed": 42}, **ids)


def snapshot(tick, epoch=1, value=7, height=0.3, hw=(360, 640)):
    state = dict(zip(ROOT_FIELDS, ([0.0, 0.0, height], [0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0],
                                   [0.0, 0.0, 0.0], [0.0, 0.0, -1.0])))
    snap = {"physics_step": tick, "simulation_time_ns": tick * 2_000_000, "reset_epoch": epoch, "state": state,
            "sensor_frame_ids": {"ego": tick + 1, "task_centric": tick + 1},
            "rgb": {k: np.full((*hw, 3), value, np.uint8) for k in ("ego", "task_centric")}}
    if hw == (360, 640):
        snap["rgb_spatial_profile"] = "dual-rgb-360p-v1"
    return snap


def observation(ticks, records, epoch=1, instruction=INSTRUCTION, hw=(360, 640), state_semantics=None):
    ids = dict(IDS, reset_epoch=epoch)
    return make_observation([snapshot(t, epoch, hw=hw) for t in ticks], message_id=f"obs:{epoch}:{ticks[-1]}",
                            instruction=instruction, executed_history=records,
                            state_semantics=state_semantics or {}, **ids)


def request(method, payload, request_id, epoch=1):
    return dict(protocol="qlm-live-v1", method=method, payload=payload, request_id=request_id,
                session_id="s", episode_id="e", reset_epoch=epoch)


def ack_for(chunk, steps=32, reason="completed", epoch=1):
    body = chunk["body"]
    ledger = CommandLedger(body["actions"], body["chunk_id"], body["start_tick"])
    for _ in range(steps):
        r = ledger.next_step()
        r.update(residual_observation_command_id=None, fl_ik_command_id=r["command_id"],
                 fl_ik_computed_tick=r["start_tick"], residual_action_18d=[0.0] * 18, extensions={})
        ledger.confirm(r)
    ids = dict(IDS, reset_epoch=epoch)
    return make_message("ExecutionAck", ledger.acknowledgement(reason), message_id=f"ack:{epoch}:{body['chunk_id']}", **ids), ledger


def test_constant_policy_shape_and_zero_force():
    policy = ms.ConstantPolicy(vx=0.15, ee=(0.44, 0.142, 0.16))
    chunk = policy(group_index=1, history=[], samples=[])
    assert len(chunk) == 16 and all(len(a) == 9 for a in chunk)
    assert chunk[0] == [0.15, 0.0, 0.0, 0.44, 0.142, 0.16, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        ms.ScriptedPolicy([[[0.0] * 9] * 15])


def test_full_session_cycle_with_the_servers_replies():
    svc = ms.MockPolicyService(ms.ConstantPolicy(vx=0.1), instruction=INSTRUCTION)
    r = svc(request("reset", {"reset": reset_message(), "observation": observation([0], [])}, "reset:1"))
    assert r["status"] == "OBSERVED" and r["duplicate"] is False
    assert r["policy_info"] == {"mock": True, "policy": "ConstantPolicy", "rgb_profile": "dual-rgb-360p-v1"}
    records = []
    for group in range(1, 4):
        start = (group - 1) * 160
        chunk = svc(request("predict", {"observation_id": f"obs:1:{start}"}, f"predict:{group}"))
        assert chunk["type"] == "ActionChunk" and chunk["body"]["start_tick"] == start
        assert chunk["profiles"] == PROFILE_HASHES and chunk["protocol_version"] == "2.0.0"
        assert chunk["body"]["chunk_id"] == f"e:1:{start}" and chunk["message_id"] == f"action:e:1:{start}"
        assert len(chunk["body"]["actions"]) == 16 and chunk["body"]["action_period_ticks"] == 10
        assert svc(request("predict", {"observation_id": f"obs:1:{start}"}, f"predict:{group}")) == chunk
        ack, ledger = ack_for(chunk)
        records = records + [row for row in ack["body"]["commands"] if row["status"] == "completed"]
        ticks = [start + 40 * k for k in range(1, 5)]
        fb = svc(request("feedback", {"ack": ack, "observation": observation(ticks, records), "terminal": None},
                         f"feedback:{group}"))
        assert fb == {"status": "OBSERVED", "tick": start + 160, "completed_count": 16 * group}
    assert svc.group_index == 4 and len(svc.history.records) == 48
    assert [e["event"] for e in svc.events][:3] == ["reset", "prediction", "feedback_commit"]
    assert svc(request("end", {"reason": "done"}, "end:1")) == {"status": "ENDED"}


def test_wrong_feedback_is_refused_like_the_real_session():
    svc = ms.MockPolicyService(ms.ConstantPolicy(), instruction=INSTRUCTION)
    svc(request("reset", {"reset": reset_message(), "observation": observation([0], [])}, "reset:1"))
    chunk = svc(request("predict", {"observation_id": "obs:1:0"}, "predict:1"))
    ack, _ = ack_for(chunk)
    rows = [r for r in ack["body"]["commands"] if r["status"] == "completed"]
    with pytest.raises(ValueError):            # three samples instead of four
        svc(request("feedback", {"ack": ack, "observation": observation([40, 80, 120], rows), "terminal": None}, "fb:x"))
    with pytest.raises(ValueError):            # a prefix without terminal evidence
        short, _ = ack_for(chunk, steps=6, reason="fall")
        svc(request("feedback", {"ack": short, "observation": None, "terminal": None}, "fb:y"))
    with pytest.raises(ValueError):            # stale identity
        svc(request("predict", {"observation_id": "obs:1:0"}, "predict:z", epoch=2))
    assert svc.state == "PREDICTED_WAIT_EXECUTION"


def test_terminal_prefix_ends_the_session():
    svc = ms.MockPolicyService(ms.ConstantPolicy(), instruction=INSTRUCTION)
    svc(request("reset", {"reset": reset_message(), "observation": observation([0], [])}, "reset:1"))
    chunk = svc(request("predict", {"observation_id": "obs:1:0"}, "predict:1"))
    ack, ledger = ack_for(chunk, steps=7, reason="operator")        # 3 completed, 1 partial
    terminal = {"tick": ledger.tick, "reset_epoch": 1, "simulation_time_ns": ledger.tick * 2_000_000,
                "reason": "operator", "captured_before_reset": True}
    r = svc(request("feedback", {"ack": ack, "observation": None, "terminal": terminal}, "fb:t"))
    assert r == {"status": "ENDED", "tick": ledger.tick, "completed_count": 3}
    assert svc(request("predict", {"observation_id": "obs:1:0"}, "predict:1")) == chunk   # cached, as the real session
    with pytest.raises(ValueError):
        svc(request("predict", {"observation_id": "obs:1:40"}, "predict:2"))


def test_instruction_must_be_the_one_the_mock_serves():
    svc = ms.MockPolicyService(ms.ConstantPolicy(), instruction=INSTRUCTION)
    packet = observation([0], [], instruction="Push the box into the target area.")
    with pytest.raises(ValueError):
        svc(request("reset", {"reset": reset_message(instruction="Push the box into the target area."),
                              "observation": packet}, "reset:1"))


def test_packets_with_body_state_are_refused():
    svc = ms.MockPolicyService(ms.ConstantPolicy(), instruction=INSTRUCTION)
    packet = observation([0], [], state_semantics=STATE_SEMANTICS)
    with pytest.raises(ValueError):
        svc(request("reset", {"reset": reset_message(), "observation": packet}, "reset:1"))


def test_images_must_have_the_360p_profile():
    svc = ms.MockPolicyService(ms.ConstantPolicy(), instruction=INSTRUCTION)          # dual-rgb-360p-v1 only
    with pytest.raises(ValueError):
        svc(request("reset", {"reset": reset_message(), "observation": observation([0], [], hw=(720, 1280))}, "reset:1"))
    with pytest.raises(TypeError):                                                     # no profile choice
        ms.MockPolicyService(ms.ConstantPolicy(), instruction=INSTRUCTION, rgb_profile="dual-rgb-720p-v1")


def test_over_tcp_with_the_qlm_client_and_simulated_latency():
    svc = ms.MockPolicyService(ms.ConstantPolicy(vx=0.05), instruction=INSTRUCTION, latency_s=0.05)
    server = ms.serve(svc, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RPCClient("127.0.0.1", server.address[1], timeout_s=5.0, retries=1)
        r = client.call("reset", {"reset": reset_message(), "observation": observation([0], [])},
                        request_id="reset:1", **IDS)
        assert r["status"] == "OBSERVED"
        chunk = client.call("predict", {"observation_id": "obs:1:0"}, request_id="predict:1", **IDS)
        assert isinstance(chunk["body"]["actions"][0][0], float)
        with pytest.raises(RemoteError):
            client.call("feedback", {"ack": ack_for(chunk)[0], "observation": None, "terminal": None},
                        request_id="fb:bad", **IDS)
        assert svc.durations["predict"]["max_s"] >= 0.05
    finally:
        server.shutdown()
        server.close()


def test_recording_service_appends_every_event_with_the_state(tmp_path):
    path = tmp_path / "events.jsonl"
    svc = ms.MockPolicyService(ms.ConstantPolicy(vx=0.1), instruction=INSTRUCTION)
    rec = ms.RecordingService(svc, str(path))
    rec(request("reset", {"reset": reset_message(), "observation": observation([0], [])}, "reset:1"))
    chunk = rec(request("predict", {"observation_id": "obs:1:0"}, "predict:1"))
    with pytest.raises(ValueError):                      # a refused request records nothing new
        rec(request("predict", {"observation_id": "obs:1:40"}, "predict:x"))
    ack, _ = ack_for(chunk)
    rows = [r for r in ack["body"]["commands"] if r["status"] == "completed"]
    rec(request("feedback", {"ack": ack, "observation": observation([40, 80, 120, 160], rows), "terminal": None}, "fb:1"))
    rec(request("end", {"reason": "done"}, "end:1"))
    events = ms.read_events(str(path))
    assert [e["event"] for e in events] == ["reset", "prediction", "feedback_commit", "end"]
    assert [e["state"] for e in events] == ["OBSERVED", "PREDICTED_WAIT_EXECUTION", "OBSERVED", "ENDED"]
    assert [s["tick"] for s in events[2]["samples"]] == [40, 80, 120, 160]
    rec(request("reset", {"reset": reset_message(epoch=2), "observation": observation([0], [], epoch=2)}, "reset:2", epoch=2))
    assert [e["event"] for e in ms.read_events(str(path))][-2:] == ["end", "reset"]


def test_cli_defaults():
    args = ms.parse_args([])
    assert (args.instruction, args.port) == (INSTRUCTION, 8862) and not hasattr(args, "rgb_profile")
    with pytest.raises(SystemExit):
        ms.parse_args(["--rgb-profile", "dual-rgb-720p-v1"])
