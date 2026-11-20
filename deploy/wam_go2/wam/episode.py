"""The real-robot client of the qlm-live-v1 protocol: one episode, chunk by chunk.

This is the role QLM's ``SynchronousDriver`` plays in simulation, rewritten for
a world whose time cannot be frozen. The structure, message identities and
validation steps mirror ``qlm_bench.live.runner`` (reset -> predict ->
execute -> feedback ... -> end, with ``PolicyHistory`` and ``CommandLedger``
doing the bookkeeping), but execution runs on a wall-clock schedule through an
abstract ``Runtime``:

* the logical tick clock (``timeline.LogicalClock``) advances only while a
  command is held - exactly like the simulator's physics during the live loop;
* a chunk's 16 commands are issued at 20 ms spacing; each 10 ms controller
  step is confirmed only if the controller was still executing (POLICY mode,
  fresh state) when the step ended;
* the four feedback samples are the frames/state nearest the +80/+160/+240/
  +320 ms instants, resolved after a bounded wait for late frames;
* between chunks (feedback + predict latency) the robot HOLDS: zero velocity,
  last FL target. The hold is recorded per chunk; it is the one thing the model
  never saw in simulation.

Stop reasons -> exit codes follow scripts/demo_sequence.py where they overlap:
0 budget reached, 1 execution error, 2 the controller left POLICY (a DAMP
request or a safety trip), 6 an observation could not be resolved, 7 the policy
server failed, 8 a prediction was refused (non-finite / out of bounds),
130 interrupted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, Tuple

from .ledger import CommandLedger
from qlm_bench.contracts.command.validation import ContractError, validate_profiles, validate_record
from qlm_bench.live import PolicyHistory, make_message, make_observation

from .limits import Bounds, first_violation
from .observation import ObservationGap, ResolvedSample, Tolerances
from .rootlink import GROUND_Z
from .timeline import GROUP_COMMANDS, ChunkSchedule, LogicalClock

EXIT_CODES = {"chunk_budget": 0, "wall_budget": 0, "controller_left_policy": 2, "observation_gap": 6,
              "rpc_error": 7, "prediction_rejected": 8, "interrupted": 130, "execute_error": 1}
POLL_S = 0.005          # sample-wait polling step


class Runtime(Protocol):
    """What the episode driver needs from the world (the ROS node, MuJoCo, or a fake)."""

    def now(self) -> float: ...
    def sleep_until(self, t: float) -> None: ...
    def command(self, native9: Optional[Sequence[float]]) -> None: ...   # None = hold
    def step_ok(self) -> Tuple[bool, str]: ...
    def ready(self, grid_wall: float, tolerances: Tolerances) -> bool: ...
    def resolve(self, **kwargs) -> ResolvedSample: ...
    def last_action18(self) -> Optional[Sequence[float]]: ...
    def record(self, event: str, payload: dict) -> None: ...
    def interrupted(self) -> bool: ...
    def latest_ego_time(self) -> Optional[float]: ...     # optional: anchors the cold start


class EpisodeAbort(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class EpisodeResult:
    reason: str
    exit_code: int
    detail: str = ""
    chunks: int = 0                    # chunks executed AND fed back (nonterminal)
    commands_completed: int = 0
    ticks: int = 0
    wall_s: float = 0.0
    holds_s: list = field(default_factory=list)


def _exit_code(reason: str) -> int:
    return EXIT_CODES.get(reason.split(":", 1)[0], 1)


class EpisodeDriver:
    def __init__(self, runtime: Runtime, client, *, session_id: str, episode_id: str, reset_epoch: int,
                 instruction: str, task_id: str = "push_box", task_version: str = "mujoco-push_box-runtime-v1",
                 policy_seed: int = 42, service_timeout_s: float = 600.0, tolerances: Tolerances = Tolerances(),
                 bounds: Bounds = Bounds(), sample_wait_s: float = 0.30, max_chunks: int = 1000,
                 max_wall_s: Optional[float] = None):
        # Both views go as 640x360 (dual-rgb-360p-v1, the policy's input). The packets carry no
        # body state (WAM-Policy takes none); the measured root state only places the camera frames.
        if type(reset_epoch) is not int or reset_epoch < 0:
            raise ValueError("reset_epoch must be a nonnegative int")
        if type(policy_seed) is not int or policy_seed < 0:
            raise ValueError("policy_seed must be a nonnegative int")
        if type(max_chunks) is not int or max_chunks < 1:
            raise ValueError("max_chunks must be a positive int")
        self.runtime = runtime
        self.client = client
        self.ids = dict(session_id=session_id, episode_id=episode_id, reset_epoch=reset_epoch)
        self.instruction = instruction
        self.task_id, self.task_version = task_id, task_version
        self.policy_seed = policy_seed
        self.service_timeout_s = float(service_timeout_s)
        self.tolerances = tolerances
        self.bounds = bounds
        self.sample_wait_s = float(sample_wait_s)
        self.max_chunks = max_chunks
        self.max_wall_s = None if max_wall_s is None else float(max_wall_s)
        self.ground_z = GROUND_Z
        self.on_event: Callable[[str, dict], None] = lambda event, payload: None
        self.history = PolicyHistory()
        self.clock = LogicalClock()
        self.observation = None            # the last accepted Observation packet
        self.last_ids: dict = {}
        self.chunks_done = 0
        self.commands_completed = 0
        self.holds_s: list = []
        self._started_wall = None
        self._last_chunk_end_wall = None
        self._ended = False

    # helpers -----------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self.ids["reset_epoch"]

    def _emit(self, event: str, payload: dict) -> None:
        self.runtime.record(event, payload)
        self.on_event(event, payload)

    def _rpc(self, method: str, payload: dict, request_id: str):
        t0 = self.runtime.now()
        tick = self.clock.tick
        try:
            result = self.client.call(method, payload, request_id=request_id, **self.ids)
        finally:
            self._emit("rpc", {"method": method, "request_id": request_id, "wall_s": self.runtime.now() - t0,
                               "tick_before": tick, "tick_after": self.clock.tick})
        return result

    def _wait_ready(self, grid_wall: float) -> bool:
        deadline = grid_wall + self.sample_wait_s
        while not self.runtime.ready(grid_wall, self.tolerances):
            now = self.runtime.now()
            if now >= deadline:
                return False
            self.runtime.sleep_until(min(now + POLL_S, deadline))
        return True

    def _resolve(self, tick: int, grid_wall: float) -> ResolvedSample:
        self._wait_ready(grid_wall)          # a late source is reported by resolve() itself
        sample = self.runtime.resolve(tick=tick, sim_time_ns=self.clock.sim_time_ns(tick), reset_epoch=self.epoch,
                                      grid_wall=grid_wall, tolerances=self.tolerances, last_ids=self.last_ids,
                                      ground_z=self.ground_z)
        self.last_ids = {"ego": sample.ego_frame_id, "task_centric": sample.d435i_frame_id}
        self._emit("sample", sample.receipt())
        return sample

    def _observation(self, samples: Sequence[ResolvedSample], end_tick: int) -> dict:
        packet = make_observation([s.snapshot() for s in samples], message_id=f"obs:{self.epoch}:{end_tick}",
                                  instruction=self.instruction, executed_history=self.history.records,
                                  state_semantics={}, origin_ns=self.clock.origin_ns, **self.ids)
        self.history.observe(packet["observation"])
        self.observation = packet
        return packet

    # protocol ----------------------------------------------------------------
    def reset_episode(self) -> dict:
        self._started_wall = self.runtime.now()
        self.runtime.command(None)                       # hold while the cold start is captured
        reset = make_message("ResetRequest", dict(timeout_s=self.service_timeout_s, origin_sim_ns=self.clock.origin_ns,
                             task=dict(id=self.task_id, version=self.task_version, instruction=self.instruction,
                                       language="en")),
                             message_id=f"reset:{self.epoch}", extensions={"policy_seed": self.policy_seed}, **self.ids)
        self.history.reset(reset)
        # The cold start is anchored to the newest ego frame (the slowest stream, <= one
        # period old) instead of an arbitrary instant: the other sources are then nearest
        # to a frame that exists, and the sample is the state of the world at that frame.
        t0 = self.runtime.now()
        anchor = getattr(self.runtime, "latest_ego_time", None)
        if anchor is not None:
            latest = anchor()
            if latest is not None and 0.0 <= t0 - latest <= 1.0:
                t0 = latest
        try:
            sample = self._resolve(0, t0)
        except ObservationGap as gap:
            raise EpisodeAbort("observation_gap:" + gap.reason, gap.detail) from gap
        packet = self._observation([sample], 0)
        self._emit("reset", {"reset": reset["message_id"], "cold_start": sample.receipt()})
        try:
            reply = self._rpc("reset", {"reset": reset, "observation": packet}, f"reset:{self.epoch}")
        except Exception as error:
            raise EpisodeAbort("rpc_error", f"reset: {type(error).__name__}: {error}") from error
        self._emit("policy_reset", {"reply": reply})
        self._last_chunk_end_wall = self.runtime.now()
        return reply

    def run(self) -> EpisodeResult:
        if self.observation is None:
            raise EpisodeAbort("state", "reset_episode() first")
        reason, detail = "chunk_budget", ""
        try:
            while True:
                if self.chunks_done >= self.max_chunks:
                    reason = "chunk_budget"
                    break
                if self.max_wall_s is not None and self.runtime.now() - self._started_wall >= self.max_wall_s:
                    reason = "wall_budget"
                    break
                if self.runtime.interrupted():
                    reason = "interrupted"
                    break
                stop = self._one_chunk()
                if stop is not None:
                    reason, detail = stop
                    break
        except Exception as error:                     # never leave the robot commanded
            self.runtime.command(None)
            reason, detail = "execute_error", f"{type(error).__name__}: {error}"
            self._emit("error", {"reason": reason, "detail": detail})
        self._end(reason)
        return EpisodeResult(reason=reason, exit_code=_exit_code(reason), detail=detail, chunks=self.chunks_done,
                             commands_completed=self.commands_completed, ticks=self.clock.tick,
                             wall_s=self.runtime.now() - self._started_wall, holds_s=list(self.holds_s))

    def _end(self, reason: str) -> None:
        if self._ended:
            return
        self._ended = True
        self.runtime.command(None)
        try:
            self._rpc("end", {"reason": reason}, f"end:{self.epoch}")
        except Exception as error:
            self._emit("end_failed", {"reason": reason, "error": f"{type(error).__name__}: {error}"})

    # one chunk -------------------------------------------------------------------
    def _one_chunk(self) -> Optional[Tuple[str, str]]:
        """Predict, execute, feed back. Returns (reason, detail) to stop, else None."""
        observation_id = self.observation["observation"]["message_id"]
        try:
            chunk = self._rpc("predict", {"observation_id": observation_id},
                              f"predict:{self.epoch}:{observation_id}")
        except Exception as error:
            return "rpc_error", f"predict: {type(error).__name__}: {error}"
        try:
            validate_record("ActionChunk", chunk)
            validate_profiles(chunk["profiles"])
            if tuple(chunk[k] for k in ("session_id", "episode_id", "reset_epoch")) != tuple(self.ids.values()):
                raise ContractError("epoch", "stale ActionChunk epoch")
            if len(chunk["body"]["actions"]) != GROUP_COMMANDS:
                raise ContractError("action", "the live profile executes exactly sixteen commands")
            self.history.expect_chunk(chunk)
        except (ContractError, ValueError, KeyError, TypeError) as error:
            return "rpc_error", f"invalid ActionChunk: {error}"
        body = chunk["body"]
        actions = body["actions"]
        bad, problems = first_violation(actions, self.bounds)
        if bad is not None:
            self._emit("rejected", {"chunk_id": body["chunk_id"], "command": bad, "problems": problems})
            return "prediction_rejected", f"command {bad}: " + "; ".join(problems)

        chunk_id = body["chunk_id"]
        ledger = CommandLedger(actions, chunk_id, body["start_tick"])
        start_wall = self.runtime.now()
        sched = ChunkSchedule(body["start_tick"], start_wall)
        hold_s = start_wall - self._last_chunk_end_wall
        self.holds_s.append(hold_s)
        self._emit("hold", {"chunk_index": self.chunks_done, "duration_s": hold_s})
        self._emit("execute", {"chunk_index": self.chunks_done, "chunk_id": chunk_id, "start_tick": body["start_tick"],
                               "start_wall": start_wall, "end_wall": sched.end_wall, "actions": actions})
        stop_reason = None
        stop_detail = ""
        for step in sched.steps():
            if step.index % 2 == 0:
                self.runtime.command(actions[step.command_index])
            self.runtime.sleep_until(step.end_wall)
            ok, why = self.runtime.step_ok()
            if not ok:
                stop_reason, stop_detail = "controller_left_policy", why
                break
            if self.runtime.interrupted():
                stop_reason, stop_detail = "interrupted", "interrupted during execution"
                break
            record = ledger.next_step()
            action18 = self.runtime.last_action18()
            record.update(residual_observation_command_id=None, fl_ik_command_id=record["command_id"],
                          fl_ik_computed_tick=record["start_tick"],
                          residual_action_18d=[float(v) for v in (action18 if action18 is not None else [0.0] * 18)],
                          extensions={})
            ledger.confirm(record)
            self.clock.advance_steps(1)
        self.runtime.command(None)                                   # hold from here on
        self._last_chunk_end_wall = self.runtime.now()

        samples = []
        if stop_reason is None:
            try:
                for tick, grid_wall in zip(sched.sample_ticks, sched.sample_walls):
                    samples.append(self._resolve(tick, grid_wall))
            except ObservationGap as gap:
                stop_reason, stop_detail = "observation_gap:" + gap.reason, gap.detail
        ack_reason = "completed" if stop_reason is None else stop_reason
        ack = make_message("ExecutionAck", ledger.acknowledgement(ack_reason),
                           message_id=f"ack:{self.epoch}:{chunk_id}", **self.ids)
        try:
            self.history.accept_ack(ack)
        except ContractError as error:
            return "execute_error", f"acknowledgement rejected locally: {error}"
        completed = ack["body"]["completed_count"]
        self.commands_completed += completed
        self._emit("executed", {"chunk_id": chunk_id, "completed": completed, "end_tick": ack["body"]["end_tick"],
                                "end_wall": self._last_chunk_end_wall, "reason": ack_reason})
        if stop_reason is None:
            try:
                packet = self._observation(samples, ack["body"]["end_tick"])
            except (ContractError, ValueError) as error:
                stop_reason, stop_detail = "execute_error", f"observation packet rejected locally: {error}"
            else:
                feedback = {"ack": ack, "observation": packet, "terminal": None}
                try:
                    reply = self._rpc("feedback", feedback, f"feedback:{self.epoch}:{chunk_id}")
                except Exception as error:
                    return "rpc_error", f"feedback: {type(error).__name__}: {error}"
                self._emit("feedback", {"chunk_id": chunk_id, "reply": reply})
                self.chunks_done += 1
                return None
        # terminal path: the prefix is acknowledged with terminal evidence, then the episode ends
        terminal = {"tick": ack["body"]["end_tick"], "reset_epoch": self.epoch,
                    "simulation_time_ns": self.clock.sim_time_ns(ack["body"]["end_tick"]), "reason": ack_reason,
                    "captured_before_reset": True, "success": False, "fallen": False, "timed_out": False}
        try:
            reply = self._rpc("feedback", {"ack": ack, "observation": None, "terminal": terminal},
                              f"feedback:{self.epoch}:{chunk_id}")
            self._emit("feedback_terminal", {"chunk_id": chunk_id, "reply": reply})
        except Exception as error:
            self._emit("feedback_terminal_failed", {"chunk_id": chunk_id, "error": f"{type(error).__name__}: {error}"})
        return stop_reason, stop_detail
