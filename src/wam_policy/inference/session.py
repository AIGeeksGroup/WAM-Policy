"""Synchronous policy session over QLM physical observations and ExecutionAck.

All future conditioning comes from live feedback. This module has no dataset
reader, simulator imports, clock stepping, or demonstration trajectory access.
"""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np

from wam_policy.common.action_space import CAMERA_KEYS
from wam_policy.common.video import preprocessing
from wam_policy.interface import PolicyHistory
from wam_policy.interface.validation import digest


def packet_digest(packet):
    def encode(value):
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, dict):
            return {key: encode(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value
    return digest(encode(packet))


class LiveObservationPipeline:
    def __init__(self, codec):
        self.codec = codec

    def reset(self):
        self.codec.reset()

    def validate(self, packet, history, *, initial=False):
        from qlm_bench.live import validate_live_observation
        start = 0 if initial else history.tick - 120
        ticks = [0] if initial else list(range(start, history.tick + 1, 40))
        validate_live_observation(packet, origin_ns=history.origin_ns, expected_capture_ticks=ticks)
        if len(packet["samples"]) != (1 if initial else 4):
            raise ValueError("Initial observation needs one frame; ordinary feedback needs four")
        if packet["observation"]["body"]["tick"] != history.tick:
            raise ValueError("RGB packet endpoint differs from confirmed execution")
        for sample in packet["samples"]:
            for image in sample["rgb"].values():
                if image.shape != (*preprocessing(getattr(self.codec,"rgb_profile","dual-rgb-360p-v1"))[0]["source_hw"], 3):
                    raise ValueError("Live RGB differs from selected codec profile")
        return ticks

    def encode(self, packet):
        views = {k: np.stack([sample["rgb"][k] for sample in packet["samples"]]) for k in CAMERA_KEYS}
        latent = self.codec.push(views)
        if latent is None or latent.shape != (1, 48, 1, 24, 20):
            raise ValueError("Live RGB did not encode exactly one causal group")
        return latent


class OnlineSession:
    def __init__(self, sampler, codec, prompt_encoder, *, policy_seed=42, info=None):
        self.sampler = sampler
        self.pipeline = LiveObservationPipeline(codec)
        self.prompt_encoder = prompt_encoder
        self.policy_seed = policy_seed
        self.info = copy.deepcopy(info or {})
        self.history = PolicyHistory()
        self.predictions = {}
        self.feedbacks = {}
        self.reset_signature = None
        self.events = []

    def _abort(self):
        self.sampler.clear()
        self.pipeline.reset()
        self.history.state = "ERROR"

    def reset(self, request, observation):
        signature = packet_digest({"reset": request, "observation": observation})
        if self.history.identity == (request["session_id"], request["episode_id"], request["reset_epoch"]):
            if signature != self.reset_signature:
                raise ValueError("Conflicting reset in the same epoch")
            return {"status": self.history.state, "duplicate": True, "policy_info": self.info}
        trial = copy.deepcopy(self.history)
        trial.reset(request)
        trial.observe(observation["observation"])
        self.pipeline.validate(observation, trial, initial=True)
        seed = request.get("extensions", {}).get("policy_seed", self.policy_seed)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Per-episode policy seed must be a nonnegative integer")
        try:
            self.pipeline.reset()
            latent = self.pipeline.encode(observation)
            prompt = self.prompt_encoder(trial.instruction)
            self.sampler.reset(latent, prompt["text_emb"], seed=seed,
                               empty_text_emb=prompt.get("empty_text_emb"))
        except BaseException:
            self._abort()
            raise
        self.history = trial
        self.predictions = {}
        self.feedbacks = {}
        self.reset_signature = signature
        self.events = [{"event": "reset", "seed": seed, "observation_sha256": packet_digest(observation)}]
        return {"status": "OBSERVED", "duplicate": False, "policy_info": self.info}

    def predict(self, observation_id):
        if self.history.state == "ERROR":
            raise ValueError("Failed session requires a new reset epoch")
        if observation_id in self.predictions:
            return copy.deepcopy(self.predictions[observation_id])
        if self.history.state != "OBSERVED" or observation_id != self.history.observation_id:
            raise ValueError("Prediction must reference the latest confirmed observation")
        started = time.perf_counter()
        try:
            commands = np.asarray(self.sampler.predict(), dtype=np.float32)
            if commands.shape != (16, 9) or not np.isfinite(commands).all() or np.any(commands[:, 6:] != 0):
                raise ValueError("Sampler must return sixteen finite force-zero native9 commands")
            session, episode, epoch = self.history.identity
            chunk_id = f"{episode}:{epoch}:{self.history.tick}"
            response = {"protocol_version": "2.0.0", "type": "ActionChunk", "session_id": session,
                        "episode_id": episode, "reset_epoch": epoch, "message_id": "action:" + chunk_id,
                        "profiles": copy.deepcopy(self.profiles), "extensions": {},
                        "body": {"observation_id": observation_id, "chunk_id": chunk_id,
                                 "start_tick": self.history.tick, "action_period_ticks": 10,
                                 "actions": commands.tolist()}}
            self.history.expect_chunk(response)
        except BaseException:
            self._abort()
            raise
        self.predictions[observation_id] = copy.deepcopy(response)
        self.events.append({"event": "predict", "chunk_id": chunk_id, "start_tick": self.history.tick,
                            "duration_s": time.perf_counter() - started,
                            "requested_sha256": digest(response), "cache": self.sampler.cache_summary()})
        return response

    @property
    def profiles(self):
        # These are the byte-frozen public QLM physical compatibility profiles.
        from wam_policy.interface.validation import PROFILE_HASHES
        return PROFILE_HASHES

    def feedback(self, ack, observation=None, terminal=None):
        if self.history.state == "ERROR":
            raise ValueError("Failed session requires a new reset epoch")
        key = ack["message_id"]
        signature = packet_digest({"ack": ack, "observation": observation, "terminal": terminal})
        if key in self.feedbacks:
            old, response = self.feedbacks[key]
            if old != signature:
                raise ValueError("Conflicting duplicate execution feedback")
            return copy.deepcopy(response)
        # All ledger/timeline checks occur on a candidate. Invalid packets cannot
        # partially commit the real ledger or touch streaming VAE state.
        trial = copy.deepcopy(self.history)
        old_count = len(trial.records)
        trial.accept_ack(ack)
        ended = trial.state == "ENDED" or terminal is not None
        if ended:
            if terminal is None or not isinstance(terminal, dict) or observation is not None:
                raise ValueError("Interrupted execution requires explicit pre-reset terminal evidence")
            if (terminal.get("tick") != trial.tick or terminal.get("reset_epoch") != trial.identity[2]
                    or terminal.get("captured_before_reset") is not True
                    or terminal.get("simulation_time_ns") != trial.origin_ns + trial.tick * 2_000_000
                    or terminal.get("reason") != ack["body"]["reason"]):
                raise ValueError("Terminal evidence must match the acknowledged endpoint and epoch")
            # Partial tail data is diagnostic only: never pad RGB or commit suffix.
            self.sampler.clear()
            self.pipeline.reset()
            trial.state = "ENDED"
        else:
            if observation is None:
                raise ValueError("A complete execution group requires actual RGB feedback")
            trial.observe(observation["observation"])
            self.pipeline.validate(observation, trial)
            executed = [r["executed"] for r in trial.records[old_count:]]
            if len(executed) != 16:
                raise ValueError("Ordinary feedback must confirm exactly sixteen commands")
            try:
                latent = self.pipeline.encode(observation)
                self.sampler.commit(latent, executed)
            except BaseException:
                self._abort()
                raise
        self.history = trial
        response = {"status": trial.state, "tick": trial.tick, "completed_count": len(trial.records)}
        self.feedbacks[key] = (signature, copy.deepcopy(response))
        self.events.append({"event": "terminal" if ended else "feedback_commit", "tick": trial.tick,
                            "ack_sha256": digest(ack), "feedback_sha256": signature,
                            "completed_count": len(trial.records), "cache": self.sampler.cache_summary()})
        return response

    def end(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError("Explicit end reason required")
        self.sampler.clear()
        self.pipeline.reset()
        self.history.state = "ENDED"
        self.events.append({"event": "end", "reason": reason})
        return {"status": "ENDED"}

    def handle(self, request):
        if request.get("protocol") != "qlm-live-v1":
            raise ValueError("Unsupported QLM protocol")
        ident = tuple(request[k] for k in ("session_id", "episode_id", "reset_epoch"))
        method, payload = request["method"], request["payload"]
        if method == "reset":
            reset = payload["reset"]
            if ident != tuple(reset[k] for k in ("session_id", "episode_id", "reset_epoch")):
                raise ValueError("RPC and reset identity differ")
            return self.reset(reset, payload["observation"])
        if ident != self.history.identity:
            raise ValueError("RPC belongs to a stale or unknown epoch")
        if method == "predict":
            return self.predict(payload["observation_id"])
        if method == "feedback":
            return self.feedback(payload["ack"], payload.get("observation"), payload.get("terminal"))
        if method == "end":
            return self.end(payload["reason"])
        raise ValueError("Unknown QLM policy operation")
