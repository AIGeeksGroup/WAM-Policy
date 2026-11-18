"""Same-noise teacher ODE endpoints as fixed regression targets, and the teacher cache.

For a context H the target is (eps_Z, eps_A, Z_T = teacher video ODE from eps_Z, A_T = teacher
action ODE from eps_A conditioned on Z_T). A target never depends on the student, so labelled
contexts can be cached on disk (the teacher cache), pooled (Collected once) or kept aside (held-out
set). Every container here holds CPU tensors only.

Teacher-cache reuse key (checked when a cache is opened): teacher sampler steps and digest, the
base model repo/revision and the longest cached prefix."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from uuid import uuid4
import torch
from arm_opsd.config import canonical_hash
from arm_opsd.config import BASE_REPO, BASE_REVISION
from arm_opsd.records import ContextSnapshot, owned_cpu
from arm_opsd.model.history import EpisodeJournal
from arm_opsd.model.codec import free_mask

CACHE_SCHEMA = "arm-opsd-teacher-cache-v1"
ROUTE = "arm-libero-long"
SOURCES = ("onpolicy", "offline", "replay", "heldout")
TENSOR_FIELDS = ("eps_video", "eps_action", "target_video", "target_action", "mask_video", "mask_action")


@dataclass
class TargetSample:
    context: ContextSnapshot
    source: str
    eps_video: torch.Tensor
    eps_action: torch.Tensor
    target_video: torch.Tensor
    target_action: torch.Tensor
    mask_video: torch.Tensor
    mask_action: torch.Tensor
    teacher_version: int
    created_utc: float
    entry_id: str = None

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError("Unknown target source")
        for name in TENSOR_FIELDS:
            value = getattr(self, name)
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError(f"Target sample tensors must be detached CPU tensors: {name}")
        if self.eps_video.shape != self.target_video.shape or self.eps_action.shape != self.target_action.shape:
            raise ValueError("Noise/target shapes differ")
        if self.mask_video.shape != self.target_video.shape or self.mask_action.shape != self.target_action.shape:
            raise ValueError("Mask/target shapes differ")

    @classmethod
    def from_joint(cls, context, joint, *, teacher_version, source):
        """A teacher rollout chunk: its own noises and endpoints are the targets."""
        if joint.state_noise is None or joint.action_noise is None:
            raise ValueError("Joint sample does not expose its initial noises")
        return cls(context, source, owned_cpu(joint.state_noise), owned_cpu(joint.action_noise),
                   owned_cpu(joint.state), owned_cpu(joint.action),
                   free_mask(joint.state, action=False, first=context.first).cpu(),
                   free_mask(joint.action, action=True, first=context.first).cpu(),
                   int(teacher_version), time.time())

    def with_source(self, source):
        return TargetSample(self.context, source, self.eps_video, self.eps_action, self.target_video,
                            self.target_action, self.mask_video, self.mask_action, self.teacher_version,
                            self.created_utc, self.entry_id)

    def state(self):
        return {"snapshot": asdict(self.context), **{name: getattr(self, name).clone() for name in TENSOR_FIELDS},
                "teacher_version": self.teacher_version, "created_utc": self.created_utc}

    @classmethod
    def from_state(cls, state, *, source, entry_id=None):
        return cls(ContextSnapshot(**state["snapshot"]), source, *(owned_cpu(state[n]) for n in TENSOR_FIELDS),
                   int(state["teacher_version"]), float(state["created_utc"]), entry_id)


class TargetGenerator:
    """Teacher-side labelling of contexts: no gradients, the teacher's deployment sampler steps."""

    def __init__(self, roles, teacher_sampler, history, prompt_lookup, *, generator):
        self.roles, self.sampler, self.history, self.prompt_lookup = roles, teacher_sampler, history, prompt_lookup
        self.generator = generator

    def _view(self, context):
        positive, negative = self.prompt_lookup(context.task_id, context.instruction)
        return self.history.view(self.roles.teacher, context, text_emb=positive, negative_text_emb=negative)

    def _initial(self, context):
        teacher = self.roles.teacher
        return self.history.journals[context.journal_id].initial_latent.to(teacher.device, teacher.dtype)

    def noise(self, shape):
        teacher = self.roles.teacher
        return torch.randn(shape, generator=self.generator, device=teacher.device, dtype=teacher.dtype)

    @torch.no_grad()
    def label(self, context, *, source, eps_video=None, eps_action=None):
        teacher, contract = self.roles.teacher, self.sampler.contract
        eps_video = self.noise(contract.state_shape) if eps_video is None else eps_video.to(teacher.device, teacher.dtype)
        eps_action = self.noise(contract.action_shape) if eps_action is None else eps_action.to(teacher.device, teacher.dtype)
        joint = self.sampler.joint(teacher, self._view(context), first=context.first,
                                   initial_latent=self._initial(context), state_noise=eps_video,
                                   action_noise=eps_action, actor_clean_action=False)
        return TargetSample.from_joint(context, joint, teacher_version=teacher.version, source=source)


def _manifest_body(*, source, config, max_prefix, entries, extra=None):
    return {"schema": CACHE_SCHEMA, "route": ROUTE, "source": source,
            "teacher_video_steps": config.method.teacher_video_steps,
            "teacher_action_steps": config.method.teacher_action_steps,
            "teacher_sampler_digest": config.teacher_sampler_digest,
            "base_model": {"repo": BASE_REPO, "revision": BASE_REVISION},
            "max_prefix": max_prefix, "count": len(entries), "entries": entries, **(extra or {})}


def write_manifest(directory, body):
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


class TargetCacheWriter:
    """Writes entries/<id>.pt; a run with several ranks merges their entry lists into one manifest."""

    def __init__(self, directory, *, config, source="teacher_rollout", max_prefix=None,
                 prefix="", teacher_version=None):
        self.directory = Path(directory)
        self.entries_dir = self.directory / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.config, self.source = config, source
        self.max_prefix = config.experiment.offline_max_prefix if max_prefix is None else max_prefix
        self.prefix, self.entries, self.skipped = prefix, [], 0
        self.teacher_version = teacher_version

    def add(self, sample, journal):
        if journal.journal_id != sample.context.journal_id:
            raise ValueError("Sample and journal belong to different episodes")
        journal.validate_snapshot(sample.context)
        if sample.context.prefix > self.max_prefix:
            self.skipped += 1
            return None
        if self.teacher_version is not None and sample.teacher_version != self.teacher_version:
            raise ValueError("Cache entries must come from one teacher version")
        entry_id = f"{self.prefix}{len(self.entries):06d}-{uuid4().hex[:8]}"
        state = journal.state()
        state["records"] = state["records"][:sample.context.prefix]  # Facts after the context are not its history.
        state["terminal_receipt"] = None
        path = self.entries_dir / f"{entry_id}.pt"
        torch.save({"journal": state, **sample.state()}, path)
        self.entries.append({"id": entry_id, "task_id": sample.context.task_id, "prefix": sample.context.prefix})
        return entry_id

    def finish(self, entries=None, extra=None):
        body = _manifest_body(source=self.source, config=self.config, max_prefix=self.max_prefix,
                              entries=sorted(entries if entries is not None else self.entries, key=lambda e: e["id"]),
                              extra=extra)
        write_manifest(self.directory, body)
        return body


class TargetCache:
    """Reader of a teacher cache; refuses a cache whose reuse key differs from the configuration."""

    def __init__(self, directory, *, config, exclude=()):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Target cache manifest missing: {path}")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != CACHE_SCHEMA or manifest.get("route") != ROUTE:
            raise ValueError("Unknown target cache schema/route")
        if (manifest.get("teacher_video_steps"), manifest.get("teacher_action_steps")) != (
                config.method.teacher_video_steps, config.method.teacher_action_steps):
            raise ValueError("Target cache teacher steps differ from the configuration")
        if manifest.get("teacher_sampler_digest") != config.teacher_sampler_digest:
            raise ValueError("Target cache teacher sampler differs from the configuration")
        if manifest.get("source") == "heldout":
            if manifest.get("max_prefix", 0) > config.experiment.offline_max_prefix:
                raise ValueError("Held-out prefixes exceed offline_max_prefix")
        elif manifest.get("max_prefix") != config.experiment.offline_max_prefix:
            # A cache built under another prefix limit covers another part of the episode.
            raise ValueError("Target cache prefix limit differs from offline_max_prefix")
        base = manifest.get("base_model") or {}
        if {k: base.get(k) for k in ("repo", "revision")} != {"repo": BASE_REPO, "revision": BASE_REVISION}:
            raise ValueError("Target cache was generated by another base model")
        if len(manifest["entries"]) != manifest.get("count") or not manifest["entries"]:
            raise ValueError("Target cache entry count differs from its manifest")
        self.manifest = manifest
        self.source = manifest["source"]
        excluded = set(exclude)
        self.entries = [e for e in manifest["entries"] if e["id"] not in excluded]
        if not self.entries:
            raise ValueError("No target cache entries remain after exclusion")

    def __len__(self):
        return len(self.entries)

    def load(self, entry, *, source="offline"):
        entry = entry if isinstance(entry, dict) else next(e for e in self.manifest["entries"] if e["id"] == entry)
        path = self.directory / "entries" / f"{entry['id']}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        journal = EpisodeJournal.from_state(state["journal"])
        sample = TargetSample.from_state(state, source=source, entry_id=entry["id"])
        journal.validate_snapshot(sample.context)
        if sample.context.task_id != entry["task_id"] or sample.context.prefix != entry["prefix"]:
            raise ValueError("Target cache entry differs from its manifest row")
        return journal, sample

    def draw(self, count, generator):
        """`count` uniform draws with replacement, deterministic under the rank's generator."""
        if count <= 0:
            return []
        indices = torch.randint(len(self.entries), (count,), generator=generator, device=generator.device).tolist()
        return [self.load(self.entries[i]) for i in indices]


class SamplePool:
    """Labelled on-policy samples kept for later updates: the replay window (last N rounds) or
    the whole collect_once set (unbounded). Journals are kept alive for the history store."""

    def __init__(self, rounds=None):
        self.rounds = rounds
        self.items = {}

    def __len__(self):
        return sum(len(v) for v in self.items.values())

    def add(self, round_id, pairs):
        if round_id in self.items:
            raise ValueError("Round already pooled")
        self.items[round_id] = [(journal, sample.with_source("replay")) for journal, sample in pairs]
        if self.rounds is not None:
            for key in sorted(self.items)[:-self.rounds] if len(self.items) > self.rounds else []:
                del self.items[key]

    def journal_ids(self):
        return {journal.journal_id for pairs in self.items.values() for journal, _ in pairs}

    def draw(self, count, generator):
        flat = [pair for key in sorted(self.items) for pair in self.items[key]]
        if count <= 0 or not flat:
            return []
        indices = torch.randint(len(flat), (count,), generator=generator, device=generator.device).tolist()
        return [flat[i] for i in indices]

    def state(self):
        """One copy of each journal: a collect_once set has about 16 contexts per episode."""
        journals = {j.journal_id: j.state() for pairs in self.items.values() for j, _ in pairs}
        return {"rounds": self.rounds, "journals": journals,
                "items": {str(k): [{"journal_id": j.journal_id, "sample": s.state()} for j, s in v]
                          for k, v in self.items.items()}}

    def load_state(self, state, history):
        self.rounds = state["rounds"]
        journals = {key: history.merge(EpisodeJournal.from_state(value)) for key, value in state.get("journals", {}).items()}
        self.items = {int(key): [(journals[row["journal_id"]], TargetSample.from_state(row["sample"], source="replay"))
                                 for row in rows] for key, rows in state["items"].items()}


def stratified_heldout(rows, count, seed):
    """Held-out entry ids from cache manifest rows, stratified by task and episode position.

    The `count` places are dealt across the tasks present (equal shares; the remainder goes to tasks
    in a seeded order). Inside a task the rows are ordered by (prefix, id) and the evenly spaced
    quantiles (i + 0.5) / n are taken, so every task contributes early, middle and late contexts up
    to its longest cached prefix. A task with fewer rows than its share gives all of them and the
    rest is dealt to the other tasks. Only manifest rows are read; the choice is deterministic for
    one cache, so every experiment that uses the cache holds out the same contexts."""
    if count <= 0:
        return []
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    if sum(len(v) for v in by_task.values()) <= count:
        raise ValueError("Target cache is too small for the held-out set")
    for task_rows in by_task.values():
        task_rows.sort(key=lambda r: (r["prefix"], r["id"]))
    order = sorted(by_task, key=lambda t: canonical_hash([seed, "heldout", str(t)]))
    quota = {task: 0 for task in order}
    remaining = count
    while remaining:
        open_tasks = [t for t in order if quota[t] < len(by_task[t])]
        share, extra = divmod(remaining, len(open_tasks))
        for index, task in enumerate(open_tasks):
            quota[task] += min(share + (index < extra), len(by_task[task]) - quota[task])
        remaining = count - sum(quota.values())
    chosen = []
    for task in order:
        task_rows, n = by_task[task], quota[task]
        chosen += [task_rows[int((i + .5) * len(task_rows) / n)]["id"] for i in range(n)]
    return sorted(chosen)


class HeldoutSet:
    """Fixed labelled contexts for the held-out regression loss; saved in the cache format."""

    def __init__(self, pairs):
        self.pairs = [(journal, sample.with_source("heldout")) for journal, sample in pairs]

    def __len__(self):
        return len(self.pairs)

    def journal_ids(self):
        return {journal.journal_id for journal, _ in self.pairs}

    def save(self, directory, *, config):
        writer = TargetCacheWriter(directory, config=config, source="heldout",
                                   max_prefix=max([s.context.prefix for _, s in self.pairs] + [0]))
        for journal, sample in self.pairs:
            if writer.add(sample, journal) is None:
                raise ValueError("Held-out entry was skipped")
        return writer.finish()

    @classmethod
    def load(cls, directory, *, config):
        cache = TargetCache(directory, config=config)
        if cache.source != "heldout":
            raise ValueError("Directory is not a held-out set")
        return cls([cache.load(entry, source="heldout") for entry in cache.manifest["entries"]])

    def attach(self, history):
        """Register the held-out journals with a history store (an identical live journal is reused)."""
        for index, (journal, sample) in enumerate(self.pairs):
            self.pairs[index] = (history.merge(journal), sample)
