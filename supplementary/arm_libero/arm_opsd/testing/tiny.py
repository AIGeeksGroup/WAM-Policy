"""Tiny-model smoke of the training chain; no weights download and no physics.

`tiny_smoke` builds a teacher cache from synthetic journals, then trains each of the five students
(configs/students/*.json, shrunk to one rank, a few updates, a 4/8-step teacher and a 2-block model)
for a few rounds, saves a checkpoint, resumes one more round from it and exports and reloads the
student. It also checks that the zero-initialized LoRA student reproduces the teacher exactly when
it samples with the teacher's steps (zero regression loss)."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from arm_opsd.checkpoint import export_student, load_student, load_training_checkpoint, save_training_checkpoint
from arm_opsd.config import ArmConfig, MachineConfig
from arm_opsd.model.history import EpisodeJournal, HistoryStore
from arm_opsd.model.loading import roles_from_model
from arm_opsd.model.sampler import NativeSampler
from arm_opsd.opsd.learner import StreamingLearner
from arm_opsd.opsd.targets import HeldoutSet, TargetCache, TargetCacheWriter, TargetGenerator, stratified_heldout

TEACHER_STEPS = (4, 8)
STUDENTS = ("opsd", "teacher_cache_only", "streaming_only", "collected_once", "sft_only")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/students"


def tiny_model(device="cpu", dtype=torch.float32, layers=2, seed=42):
    from wan_va.modules.model import WanTransformer3DModel
    torch.manual_seed(seed)
    return WanTransformer3DModel(patch_size=[1, 2, 2], num_attention_heads=2,
        attention_head_dim=12, in_channels=48, out_channels=48, action_dim=30,
        text_dim=16, freq_dim=12, ffn_dim=48, num_layers=layers, attn_mode="torch").to(device, dtype)


class SyntheticDemonstrations:
    """Random whole-episode samples with the SFT-only tensor layout, for the tiny models."""

    def __init__(self, frames=4, text=(4, 16)):
        self.frames, self.text = frames, text

    def __len__(self):
        return 8

    def empty_text(self):
        return torch.zeros(*self.text)

    def draw(self, count, generator, longest=False):
        device = generator.device
        out = []
        for _ in range(count):
            mask = torch.zeros(30, self.frames, 4, 1, dtype=torch.bool)
            mask[:7] = True
            actions = torch.randn(30, self.frames, 4, 1, generator=generator, device=device).clamp(-1.5, 1.5).cpu() * mask
            out.append({"latents": torch.randn(48, self.frames, 8, 16, generator=generator, device=device).cpu(),
                        "actions": actions, "actions_mask": mask,
                        "text_emb": torch.randn(*self.text, generator=generator, device=device).cpu()})
        return out


def smoke_config(student, *, rounds=2, device="cpu", cache="", heldout=4, collect_once_rounds=2):
    """configs/students/<student>.json, shrunk to one rank, `rounds` updates and a 4/8-step teacher."""
    config = ArmConfig.load(CONFIG_DIR / f"{student}.json")
    method = replace(config.method, world_size=1, student_updates=rounds, teacher_video_steps=TEACHER_STEPS[0],
                     teacher_action_steps=TEACHER_STEPS[1],
                     collect_once_rounds=min(config.method.collect_once_rounds, collect_once_rounds))
    return replace(config, method=method, experiment=replace(config.experiment, heldout_contexts=heldout),
                   machine=MachineConfig(device=device, offline_cache=cache), profile="tiny").validate()


class SyntheticStream:
    """Three snapshots per round from one synthetic episode; random facts, no physics."""

    def __init__(self, history, *, stream="tiny", task=0, seed=0):
        self.generator = torch.Generator().manual_seed(seed)
        self.journal = EpisodeJournal(stream_id=stream, session_id="synthetic-session",
            episode_id=f"synthetic-episode-{stream}-{seed}", task_id=task, instruction="synthetic fixture",
            initial_latent=torch.randn(1, 48, 1, 8, 16, generator=self.generator), resource_hash="synthetic-no-physics")
        history.add(self.journal)
        self.history = history

    def collect(self, *, actor_version, round_id, count=3):
        contexts = []
        for slot in range(count):
            context = self.journal.snapshot(actor_version=actor_version, round_id=round_id, slot_id=slot)
            contexts.append(context)
            actions = torch.randn(1, 30, 4, 4, 1, generator=self.generator)
            actions[:, 7:] = 0
            if context.first:
                actions[:, :, 0] = 0
            self.journal.append(receipt_id=f"synthetic-{round_id}-{slot}",
                latents=torch.randn(1, 48, 3 if context.first else 4, 8, 16, generator=self.generator),
                normalized_actions=actions, applied_actions=torch.randn(12 if context.first else 16, 7, generator=self.generator),
                actor_version=actor_version)
        return contexts

    def live_prefix(self):
        return (self.journal.journal_id, len(self.journal.records))


def prompt_lookup_factory(seed=7):
    generator = torch.Generator().manual_seed(seed)
    positive, negative = torch.randn(1, 4, 16, generator=generator), torch.randn(1, 4, 16, generator=generator)
    return lambda task, instruction: (positive, negative)


def build_synthetic_cache(directory, roles, config, prompt_lookup, *, journals=2, per_journal=3, seed=11):
    """Teacher-labelled synthetic contexts written through the cache writer (source teacher_rollout)."""
    history = HistoryStore()
    sampler = NativeSampler(config.native, state_steps=config.method.teacher_video_steps,
                            action_steps=config.method.teacher_action_steps)
    generator = TargetGenerator(roles, sampler, history, prompt_lookup,
                                generator=torch.Generator(device=roles.teacher.device).manual_seed(seed))
    writer = TargetCacheWriter(directory, config=config, source="teacher_rollout", teacher_version=0)
    for index in range(journals):
        stream = SyntheticStream(history, stream=f"cache-{index}", seed=seed + index)
        for context in stream.collect(actor_version=0, round_id=1, count=per_journal):
            writer.add(generator.label(context, source="offline"), stream.journal)
    return writer.finish()


def attach_demonstrations(learner, prompt_lookup):
    if learner.flow_matching:
        learner.flow_matching_data = SyntheticDemonstrations()
        learner.flow_matching_empty_text = learner.flow_matching_data.empty_text()


def run_student(roles, config, prompt_lookup, *, rounds, output, cache_dir):
    """Rounds of one student on the synthetic stream; checkpoint, resume one more round, export, reload."""
    history = HistoryStore()
    full = TargetCache(cache_dir, config=config)
    # As runtime.train: every student holds out the same stratified cache entries and never trains on them.
    heldout_ids = stratified_heldout(full.manifest["entries"], config.experiment.heldout_contexts, config.experiment.seed)
    cache = TargetCache(cache_dir, config=config, exclude=heldout_ids) if heldout_ids else full
    learner = StreamingLearner(roles, history, prompt_lookup, config, offline_cache=cache)
    attach_demonstrations(learner, prompt_lookup)
    stream = SyntheticStream(history, stream=config.method.data_mode, seed=100) if config.method.collects else None
    if heldout_ids:
        learner.heldout = HeldoutSet([full.load(e, source="heldout") for e in full.manifest["entries"] if e["id"] in set(heldout_ids)])
        learner.heldout.attach(history)
    heldout = [learner.evaluate(learner.heldout.pairs)] if learner.heldout else []
    save_training_checkpoint(output / "round-0", learner)
    metrics = []
    while learner.counters.student_updates < rounds:
        round_id = learner.counters.attempted_rounds + 1
        contexts = stream.collect(actor_version=learner.roles.student.version, round_id=round_id) if learner.collecting else []
        metrics.append(learner.train_round(contexts, round_id=round_id))
        learner.release([stream.live_prefix()] if stream is not None else [])
        if learner.heldout and not metrics[-1]["collect_round"]:
            heldout.append(learner.evaluate(learner.heldout.pairs))
    checkpoint = save_training_checkpoint(output / "final", learner)
    export = export_student(output / "student", roles.student, config, update=rounds,
                            adapter=learner.ema_state(), variant="ema" if learner.ema_state() is not None else "raw")
    # Resume into fresh roles from the same base and continue exactly one more round.
    rank, alpha = config.method.lora_rank, config.method.lora_alpha
    fresh = roles_from_model(roles.teacher.model, rank=rank, alpha=alpha)
    resumed_history = HistoryStore()
    resumed = StreamingLearner(fresh, resumed_history, prompt_lookup, config, offline_cache=cache)
    attach_demonstrations(resumed, prompt_lookup)
    load_training_checkpoint(checkpoint, resumed)
    if asdict(resumed.counters) != asdict(learner.counters):
        raise AssertionError("Resumed counters differ")
    if stream is not None:
        # The live synthetic episode continues after the resume (a real run starts a new episode);
        # a pooled copy of it restored from the checkpoint is superseded by the live object.
        resumed_history.merge(stream.journal)
        resumed_history.journals[stream.journal.journal_id] = stream.journal
    round_id = resumed.counters.attempted_rounds + 1
    contexts = stream.collect(actor_version=resumed.roles.student.version, round_id=round_id) if resumed.collecting else []
    after = resumed.train_round(contexts, round_id=round_id)
    from arm_opsd.model.kv_views import NativeRole
    deployment = NativeRole(roles_from_model(roles.teacher.model).teacher.model, "student")
    metadata = load_student(export, deployment)
    return {"rounds": [{"update": m["counters"]["student_updates"], "collect_round": m["collect_round"], "data": m["data"],
                        "losses": m["losses"]} for m in metrics],
            "heldout": heldout, "resumed_round": {"update": after["counters"]["student_updates"], "data": after["data"],
                                                   "losses": after["losses"]},
            "export": {"path": str(export), "format": metadata["format"], "variant": metadata["variant"]}}


def self_check(roles, config, prompt_lookup, *, seed=3):
    history = HistoryStore()
    learner = StreamingLearner(roles, history, prompt_lookup, config)
    stream = SyntheticStream(history, stream="selfcheck", seed=seed)
    pairs = learner.label(stream.collect(actor_version=0, round_id=1), "heldout")
    return learner.self_check(pairs)


def tiny_smoke(output, *, device="cpu", rounds=2):
    if rounds < 2:
        raise ValueError("The smoke needs at least two rounds")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    base = tiny_model(device, dtype)
    prompt_lookup = prompt_lookup_factory()
    cache_dir = output / "teacher-cache"
    cache_config = smoke_config("opsd", device=device, cache=str(cache_dir))
    cache_manifest = build_synthetic_cache(cache_dir, roles_from_model(base), cache_config, prompt_lookup)
    check = self_check(roles_from_model(base), cache_config, prompt_lookup)
    students = {}
    for student in STUDENTS:
        config = smoke_config(student, rounds=rounds, device=device, cache=str(cache_dir))
        students[student] = run_student(roles_from_model(base, rank=config.method.lora_rank, alpha=config.method.lora_alpha),
                                        config, prompt_lookup, rounds=rounds, output=output / student, cache_dir=cache_dir)
    passed = check["teacher_steps_total"] < 1e-6 and check["student_steps_total"] > 0
    report = {"passed": passed, "elapsed_s": time.perf_counter() - started, "device": device, "dtype": str(dtype),
              "model": "tiny LingBot-VA transformer, 2 blocks", "teacher_steps": list(TEACHER_STEPS),
              "teacher_cache_entries": cache_manifest["count"], "self_check": check,
              "students": {name: {"export": value["export"], "final_losses": value["resumed_round"]["losses"]}
                           for name, value in students.items()}}
    (output / "report.json").write_text(json.dumps({**report, "details": students}, indent=2) + "\n")
    if not passed:
        raise AssertionError(f"tiny smoke failed; see {output / 'report.json'}")
    return report
