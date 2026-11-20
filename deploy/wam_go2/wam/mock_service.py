"""A protocol-faithful stand-in for the WAM-Policy policy server.

The policy server (``python -m wam_policy.inference.serve`` of this repository,
its ``wam_policy.inference.session.OnlineSession``) serves qlm-live-v1 requests
(reset, predict, feedback, end) for a model that needs a large GPU. This module
serves the same requests with the same state machine and the same acceptance
rules, but a scripted policy instead of the model: the bridge, the mock
integration test and the MuJoCo closed loop can run the real protocol on any
laptop.

Mirrors ``OnlineSession`` (NEW -> OBSERVED -> PREDICTED_WAIT_EXECUTION -> ... ->
ENDED, ERROR after a failed mutation), its packet checks
(``validate_live_observation``, 1 or 4 samples, both views uint8 640x360 -
``dual-rgb-360p-v1`` -, nonterminal feedback with all 16 commands and four
samples at +40/+80/+120/+160 ticks, idempotent duplicates, conflicting
duplicates refused) and its reply shapes (reset answers with ``policy_info``).
Two rules are the mock's own: the reset instruction must equal the mock's
``instruction`` (a server started with --prompt-root likewise refuses an
instruction it has no cached prompt embedding for), and a packet carrying body
state is refused (WAM-Policy takes none, so the bridge must send none). What it
does NOT do: encode video, keep a KV cache, or produce anything a model would.
"""
from __future__ import annotations

import copy
import json
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from qlm_bench.contracts.command.validation import PROFILE_HASHES, VERSION, validate_record
from qlm_bench.live import PolicyHistory, RPCServer, validate_live_observation

from . import settings
from .observation import PROFILE_HW, RGB_360
from .timeline import GROUP_COMMANDS, SAMPLE_TICKS

CAMERA_KEYS = ("go2_ego", "d435i_rgb_task")
DEFAULT_INSTRUCTION = settings.INSTRUCTION


def _fingerprint(value) -> str:
    """Content identity of a request part, with RGB arrays by hash (as the server's packet_digest)."""
    import hashlib
    import json

    def normalize(obj):
        if isinstance(obj, np.ndarray):
            return {"dtype": str(obj.dtype), "shape": list(obj.shape),
                    "sha256": hashlib.sha256(obj.tobytes()).hexdigest()}
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [normalize(v) for v in obj]
        return obj
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------- policies
class ConstantPolicy:
    """Every chunk is the same command: a velocity and an FL target (force zero)."""

    def __init__(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
                 ee: Sequence[float] = (0.44, 0.142, 0.16)):
        self.command = [float(vx), float(vy), float(wz)] + [float(v) for v in ee] + [0.0, 0.0, 0.0]

    def __call__(self, *, group_index: int, history: list, samples: list) -> List[List[float]]:
        return [list(self.command) for _ in range(GROUP_COMMANDS)]


class ScriptedPolicy:
    """A fixed list of chunks (each 16 x 9); the last one repeats once exhausted."""

    def __init__(self, chunks: Sequence[Sequence[Sequence[float]]]):
        chunks = [[[float(v) for v in a] for a in chunk] for chunk in chunks]
        if not chunks or any(len(c) != GROUP_COMMANDS or any(len(a) != 9 for a in c) for c in chunks):
            raise ValueError("ScriptedPolicy needs chunks of exactly 16 native9 commands")
        self.chunks = chunks

    def __call__(self, *, group_index: int, history: list, samples: list) -> List[List[float]]:
        return copy.deepcopy(self.chunks[min(group_index - 1, len(self.chunks) - 1)])


# ----------------------------------------------------------------------------- service
class MockPolicyService:
    """qlm-live-v1 request handler with the WAM session's acceptance rules."""

    def __init__(self, policy: Callable, *, instruction: str = DEFAULT_INSTRUCTION,
                 latency_s: float = 0.0, fail_on: Optional[Callable[[dict], bool]] = None):
        self.policy = policy
        self.instruction = instruction
        self.rgb_profile = RGB_360
        self.image_hw = PROFILE_HW
        self.latency_s = float(latency_s)
        self.fail_on = fail_on
        self.history = PolicyHistory()
        self.state = "NEW"
        self.group_index = 0
        self.pending = None
        self.last_timestamp_ns = None
        self.predict_replies = {}
        self.feedback_replies = {}
        self._reset_fingerprint = None
        self.events = []
        self.samples_seen = []          # compact receipts of every accepted sample
        self.durations = {}
        self.calls = 0

    # dispatch (OnlineSession.handle) --------------------------------------
    def __call__(self, request: dict):
        started = time.perf_counter()
        method = request["method"]
        try:
            if self.fail_on is not None and self.fail_on(request):
                raise RuntimeError("injected policy service failure")
            if self.latency_s > 0.0:       # inference stand-in; this is a plain RPC process, not a ROS node
                time.sleep(self.latency_s)
            payload = request["payload"]
            ident = (request["session_id"], request["episode_id"], request["reset_epoch"])
            if method == "reset":
                reset = payload["reset"]
                if ident != (reset["session_id"], reset["episode_id"], reset["reset_epoch"]):
                    raise ValueError("Transport/reset identities differ")
                return self.reset(reset, payload["observation"])
            if ident != self.history.identity:
                raise ValueError("Stale transport episode/reset identity")
            if method == "predict":
                return self.predict(payload["observation_id"])
            if method == "feedback":
                return self.feedback(payload["ack"], payload.get("observation"), payload.get("terminal"))
            if method == "end":
                return self.end(payload["reason"])
            raise ValueError("Unknown policy operation")
        finally:
            elapsed = time.perf_counter() - started
            self.calls += 1
            stats = self.durations.setdefault(method, {"calls": 0, "total_s": 0.0, "max_s": 0.0})
            stats["calls"] += 1
            stats["total_s"] += elapsed
            stats["max_s"] = max(stats["max_s"], elapsed)

    # packet checks (the server's LiveObservationPipeline.validate) ---------
    def _validate(self, packet, *, origin_ns, expected_ticks):
        validate_live_observation(packet, origin_ns=origin_ns, expected_capture_ticks=expected_ticks)
        if packet.get("state_semantics") or any(sample["root_state"] for sample in packet["samples"]):
            raise ValueError("WAM-Policy takes no body state: the packet must carry none")
        if len(packet["samples"]) not in (1, 4):
            raise ValueError("Initial observation needs one frame; ordinary feedback needs four")
        for sample in packet["samples"]:
            for key in CAMERA_KEYS:
                image = sample["rgb"][key]
                if image.dtype != np.uint8 or image.shape != (*self.image_hw, 3):
                    raise ValueError(f"Live RGB differs from the {self.rgb_profile} profile: {key} {image.shape}")
        return [{"tick": s["capture_tick"], "sim_ns": s["capture_sim_ns"], "frame_ids": dict(s["sensor_frame_ids"]),
                 "rgb_mean": {k: float(s["rgb"][k].mean()) for k in CAMERA_KEYS}} for s in packet["samples"]]

    def _message(self, kind, body, message_id):
        session, episode, epoch = self.history.identity
        value = {"protocol_version": VERSION, "session_id": session, "episode_id": episode,
                 "reset_epoch": epoch, "message_id": message_id, "profiles": copy.deepcopy(PROFILE_HASHES),
                 "extensions": {}, "type": kind, "body": body}
        validate_record(kind, value)
        return value

    # operations (OnlineSession) -------------------------------------------
    def reset(self, request, packet):
        value = _fingerprint({"reset": request, "observation": packet})
        if self._reset_fingerprint == value:
            return {"status": self.state, "duplicate": True, "policy_info": self._policy_info()}
        candidate = copy.deepcopy(self.history)
        candidate.reset(request)
        if candidate.instruction != self.instruction:
            raise ValueError("Instruction differs from the instruction this mock serves")
        candidate.observe(packet["observation"])
        receipts = self._validate(packet, origin_ns=candidate.origin_ns, expected_ticks=[0])
        seed = request.get("extensions", {}).get("policy_seed", 42)
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("Explicit finite nonnegative policy seed required")
        self.state = "ERROR"
        self.history = candidate
        self.group_index = 1
        self.pending = None
        self.predict_replies = {}
        self.feedback_replies = {}
        self.last_timestamp_ns = candidate.origin_ns
        self._reset_fingerprint = value
        self.samples_seen = receipts
        self.events = [{"event": "reset", "timestamp_ns": self.last_timestamp_ns, "measured_states": 1, "seed": seed}]
        self.state = "OBSERVED"
        return {"status": self.state, "duplicate": False, "policy_info": self._policy_info()}

    def _policy_info(self):
        return {"mock": True, "policy": type(self.policy).__name__, "rgb_profile": self.rgb_profile}

    def predict(self, observation_id):
        if self.state == "ERROR":
            raise ValueError("Failed session requires a new reset epoch")
        if observation_id in self.predict_replies:
            return copy.deepcopy(self.predict_replies[observation_id])
        if self.state != "OBSERVED" or observation_id != self.history.observation_id:
            raise ValueError("Prediction requires the current committed physical observation")
        self.state = "ERROR"
        actions = self.policy(group_index=self.group_index, history=list(self.history.records),
                              samples=list(self.samples_seen))
        commands = np.asarray(actions, np.float32).reshape(GROUP_COMMANDS, 9)
        if not np.isfinite(commands).all() or np.any(commands[:, 6:] != 0):
            raise ValueError("Invalid finite zero-force native9 prediction")
        _, episode, epoch = self.history.identity
        chunk_id = f"{episode}:{epoch}:{self.history.tick}"            # as the server's OnlineSession
        message = self._message("ActionChunk", {"chunk_id": chunk_id, "start_tick": self.history.tick,
                                                "observation_id": observation_id, "action_period_ticks": 10,
                                                "actions": commands.astype(np.float64).tolist()},
                                "action:" + chunk_id)
        self.history.expect_chunk(message)
        self.pending = copy.deepcopy(message)
        self.predict_replies[observation_id] = copy.deepcopy(message)
        self.state = "PREDICTED_WAIT_EXECUTION"
        self.events.append({"event": "prediction", "group": self.group_index,
                            "conditioned_until_ns": self.last_timestamp_ns,
                            "observed_history_commands": len(self.history.records)})
        return message

    def feedback(self, ack, packet=None, terminal=None):
        key = ack["message_id"]
        sha = _fingerprint({"ack": ack, "observation": packet, "terminal": terminal})
        if key in self.feedback_replies:
            prior, result = self.feedback_replies[key]
            if prior != sha:
                raise ValueError("Conflicting duplicate feedback")
            return copy.deepcopy(result)
        if self.state != "PREDICTED_WAIT_EXECUTION" or self.pending is None:
            raise ValueError("Feedback requires a pending action chunk")
        candidate = copy.deepcopy(self.history)
        candidate.accept_ack(ack)
        ended = candidate.state == "ENDED" or terminal is not None
        if ended:
            if terminal is None:
                raise ValueError("Terminal prefix requires terminal evidence")
            if packet is not None or terminal.get("tick") != candidate.tick or \
                    terminal.get("reset_epoch") != candidate.identity[2] or \
                    terminal.get("simulation_time_ns") != candidate.origin_ns + candidate.tick * 2_000_000 or \
                    terminal.get("captured_before_reset") is not True or \
                    not isinstance(terminal.get("reason"), str) or not terminal["reason"] or \
                    terminal["reason"] != ack["body"]["reason"]:
                raise ValueError("Terminal evidence must match acknowledged physical time before reset")
            self.history = candidate
            self.pending = None
            self.state = "ENDED"
            result = {"status": self.state, "tick": candidate.tick, "completed_count": len(candidate.records)}
            self.events.append({"event": "terminal", "tick": candidate.tick,
                                "completed_commands": len(candidate.records)})
        else:
            start = self.pending["body"]["start_tick"]
            expected = [start + i * SAMPLE_TICKS for i in range(1, 5)]
            if ack["body"]["end_tick"] != start + 160 or ack["body"]["completed_count"] != 16 or packet is None:
                raise ValueError("Nonterminal feedback must confirm all 16 commands and four RGB/state samples")
            receipts = self._validate(packet, origin_ns=candidate.origin_ns, expected_ticks=expected)
            candidate.observe(packet["observation"])
            if packet["samples"][0]["capture_sim_ns"] <= self.last_timestamp_ns:
                raise ValueError("Feedback repeated an already committed boundary")
            self.state = "ERROR"
            self.last_timestamp_ns = candidate.origin_ns + candidate.tick * 2_000_000
            self.history = candidate
            self.pending = None
            self.group_index += 1
            self.samples_seen = receipts
            self.state = "OBSERVED"
            result = {"status": self.state, "tick": candidate.tick, "completed_count": len(candidate.records)}
            self.events.append({"event": "feedback_commit", "group": self.group_index - 1,
                                "timestamp_ns": self.last_timestamp_ns, "samples": receipts})
        self.feedback_replies[key] = sha, copy.deepcopy(result)
        return result

    def end(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError("Explicit end reason required")
        self.state = "ENDED"
        self.pending = None
        self.events.append({"event": "end", "reason": reason})
        return {"status": "ENDED"}


class LocalClient:
    """In-process stand-in for ``qlm_bench.live.RPCClient``: same request envelope,
    handler exceptions surface as ``RemoteError`` (as the TCP server reports them),
    an optional clock hook models the wall time a call takes."""

    def __init__(self, handler: Callable, *, latency_s: float = 0.0, advance_clock: Optional[Callable[[float], None]] = None):
        self.handler = handler
        self.latency_s = float(latency_s)
        self.advance_clock = advance_clock
        self.calls = []

    def call(self, method, payload, *, request_id, session_id, episode_id, reset_epoch):
        from qlm_bench.live import RemoteError
        from qlm_bench.live.transport import validate_request
        request = validate_request(dict(protocol="qlm-live-v1", method=method, payload=payload,
                                        request_id=request_id, session_id=session_id,
                                        episode_id=episode_id, reset_epoch=reset_epoch))
        self.calls.append((method, request_id))
        if self.advance_clock is not None and self.latency_s > 0.0:
            self.advance_clock(self.latency_s)
        try:
            return copy.deepcopy(self.handler(copy.deepcopy(request)))
        except Exception as error:
            raise RemoteError(type(error).__name__ + ": " + str(error)) from error


class RecordingService:
    """Wraps a ``MockPolicyService``: after every request, each new service event is
    appended to ``path`` as one JSON line carrying the session state, so another
    process can read what the service accepted (the integration test runs the
    service in its own process - its 30 MB feedback messages must not share a
    Python process with the mock robot's 500 Hz LowState timer)."""

    def __init__(self, service: MockPolicyService, path: str):
        self.service = service
        self.path = path
        self._events_ref = None
        self._written = 0
        open(path, "w", encoding="utf-8").close()

    def __call__(self, request: dict):
        try:
            return self.service(request)
        finally:
            self.flush()

    def flush(self) -> None:
        events = self.service.events
        if events is not self._events_ref:        # a reset starts a new event list
            self._events_ref, self._written = events, 0
        if self._written >= len(events):
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for event in events[self._written:]:
                f.write(json.dumps({"state": self.service.state, **event}, default=str) + "\n")
        self._written = len(events)


def read_events(path: str) -> List[dict]:
    """The events a ``RecordingService`` wrote (an empty list for a missing file)."""
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def serve(handler, host: str = "127.0.0.1", port: int = 8862, **kwargs) -> RPCServer:
    """An RPCServer for ``handler``; ``port=0`` picks a free port (``server.address``)."""
    return RPCServer(handler, host=host, port=port, **kwargs)


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Mock WAM-Policy policy server (qlm-live-v1) with a scripted policy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8862)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="the only instruction reset accepts (default: push_box's)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--ee", type=float, nargs=3, default=(0.1934, 0.142, 0.05), metavar=("X", "Y", "Z"))
    ap.add_argument("--latency", type=float, default=0.0, help="seconds slept per request (inference stand-in)")
    ap.add_argument("--record", default="", metavar="PATH",
                    help="append every accepted event (reset, prediction, feedback_commit, terminal, end) as a JSON line")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    service = MockPolicyService(ConstantPolicy(args.vx, args.vy, args.wz, args.ee), instruction=args.instruction,
                                latency_s=args.latency)
    handler = RecordingService(service, args.record) if args.record else service
    server = serve(handler, args.host, args.port)
    print(f"mock policy service on {server.address[0]}:{server.address[1]} ({RGB_360}, "
          f"vx {args.vx}, vy {args.vy}, wz {args.wz}, ee {tuple(args.ee)}, latency {args.latency} s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
