"""Regression targets: same-noise teacher ODE endpoints per context, and the teacher target cache.

A target sample is (context H, eps_video, eps_action, Z_T, A_T, masks). Z_T is the frozen
teacher's multi-step video ODE endpoint from eps_video; A_T is the teacher's multi-step action
endpoint from eps_action conditioned on Z_T. Targets do not depend on the student, so cached and
held-out contexts stay valid supervision pairs for the whole run.

Cache layout: <root>/manifest.json (header + rows {id, task_id, prefix}) and entries/<id>.pt holding
the rebuildable journal prefix, the snapshot, the noise, the targets and the masks. The header is
the cache's reuse key (cache_key): a run whose teacher, teacher/student samplers, train pool or
M-OPD specialist teachers differ refuses the cache.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import torch
from wam_policy.opsd.config import digest, write_json
from wam_policy.opsd.history import STATE_SHAPE, ACTION_SHAPE, ContextRecord, EpisodeJournal, HistorySnapshot, cpu_copy
from wam_policy.opsd.sampling import generate, teacher_sampler_manifest, sampler_manifest

SCHEMA = "wam-policy-teacher-cache-v1"
SOURCES = ("teacher_rollout", "synthetic")
SAMPLE_SOURCES = ("onpolicy", "offline", "heldout")
KEY_FIELDS = ("teacher", "train_pool_sha256", "teacher_video_steps", "teacher_action_steps",
              "teacher_sampler_digest", "student_sampler_digest", "experts")


def action_mask(config):
    mask = torch.zeros(ACTION_SHAPE, dtype=torch.bool)
    mask[:, :config.action_valid_dims] = True
    return mask


def video_mask():
    return torch.ones(STATE_SHAPE, dtype=torch.bool)


@dataclass(frozen=True)
class TargetSample:
    record: ContextRecord
    source: str
    eps_video: torch.Tensor
    eps_action: torch.Tensor
    target_video: torch.Tensor
    target_action: torch.Tensor
    mask_video: torch.Tensor
    mask_action: torch.Tensor
    teacher_version: int
    created_utc: str
    id: str

    @property
    def task_id(self):
        return self.record.journal.contract.get("task", "unspecified")

    @property
    def key(self):
        return self.record.snapshot.key

    def validate(self):
        if self.source not in SAMPLE_SOURCES:
            raise ValueError("Unknown target sample source")
        shapes = ((self.eps_video, STATE_SHAPE), (self.target_video, STATE_SHAPE), (self.mask_video, STATE_SHAPE),
                  (self.eps_action, ACTION_SHAPE), (self.target_action, ACTION_SHAPE), (self.mask_action, ACTION_SHAPE))
        if any(tuple(t.shape) != shape for t, shape in shapes):
            raise ValueError("Target sample geometry mismatch")
        if self.mask_video.dtype != torch.bool or self.mask_action.dtype != torch.bool:
            raise ValueError("Boolean target masks required")
        for t in (self.eps_video, self.eps_action, self.target_video, self.target_action):
            if not torch.isfinite(t).all():
                raise FloatingPointError("Nonfinite target sample")
        self.record.validate()
        return self

    def payload(self):
        """CPU tensors of a cache entry; the journal is stored up to the context prefix only."""
        return {"journal": self.record.journal.prefix_state(self.record.snapshot.prefix_end),
                "snapshot": self.record.snapshot.to_dict(),
                "eps_video": cpu_copy(self.eps_video), "eps_action": cpu_copy(self.eps_action),
                "target_video": cpu_copy(self.target_video), "target_action": cpu_copy(self.target_action),
                "mask_video": cpu_copy(self.mask_video), "mask_action": cpu_copy(self.mask_action),
                "teacher_version": self.teacher_version, "created_utc": self.created_utc}

    @classmethod
    def from_payload(cls, value, *, source, id):
        journal = EpisodeJournal.from_prefix_state(value["journal"])
        snapshot = HistorySnapshot.from_dict(value["snapshot"])
        record = ContextRecord(snapshot, journal)
        return cls(record, source, value["eps_video"], value["eps_action"], value["target_video"],
                   value["target_action"], value["mask_video"], value["mask_action"],
                   int(value["teacher_version"]), str(value["created_utc"]), id).validate()


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TargetGenerator:
    """No-grad teacher ODE endpoints for one context under the teacher sampler steps."""

    def __init__(self, teacher, config):
        self.teacher, self.config = teacher, config
        self.video_mask = video_mask()
        self.action_mask = action_mask(config)

    @torch.no_grad()
    def generate(self, record, eps_video, eps_action, *, source, id):
        config = self.config
        with self.teacher.prepare(record) as c:
            z = generate(c, None, initial_noise=eps_video, steps=config.teacher_video_steps)
            c.condition(z)
            a = generate(c, None, action=True, initial_noise=eps_action, steps=config.teacher_action_steps)
        return TargetSample(record, source, cpu_copy(eps_video), cpu_copy(eps_action), cpu_copy(z), cpu_copy(a),
                            self.video_mask, self.action_mask, self.teacher.version, now_utc(), id).validate()

    def from_actor_record(self, record, *, source, id):
        """A teacher actor's own chunk: its noise and sample are the target (zero-cost labels)."""
        if record.noise is None or record.plan is None or record.model_normalized is None:
            raise ValueError("Actor record lacks its noise/plan/sample")
        eps_video, eps_action = record.noise
        return TargetSample(record, source, cpu_copy(eps_video), cpu_copy(eps_action), cpu_copy(record.plan),
                            cpu_copy(record.model_normalized), self.video_mask, self.action_mask,
                            self.teacher.version, now_utc(), id).validate()


def cache_key(config):
    """Reuse key of a teacher target cache built or read under this config: the teacher, the train
    pool, the teacher/student samplers and (M-OPD) the specialist teachers of the expert tasks."""
    from wam_policy.opsd.runtime import teacher_key, pool_sha256
    from wam_policy.mopd.experts import expert_key
    return {"teacher": teacher_key(config), "train_pool_sha256": pool_sha256(config),
            "teacher_video_steps": config.teacher_video_steps, "teacher_action_steps": config.teacher_action_steps,
            "teacher_sampler_digest": digest(teacher_sampler_manifest(config)),
            "student_sampler_digest": digest(sampler_manifest(config)), "experts": expert_key(config)}


def cache_header(config, *, source, key=None):
    if source not in SOURCES:
        raise ValueError("Unknown target cache source")
    return {"schema": SCHEMA, "source": source, **(cache_key(config) if key is None else key)}


def write_entries(root, samples):
    """Write entries/<id>.pt for these samples; returns their manifest rows (any rank)."""
    root = Path(root)
    (root / "entries").mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        sample.validate()
        if not sample.id or "/" in sample.id or sample.id.startswith("."):
            raise ValueError("Safe target entry id required")
        path = root / "entries" / (sample.id + ".pt")
        if path.exists():
            raise FileExistsError(path)
        fd, staging = tempfile.mkstemp(prefix="entry-", suffix=".partial", dir=path.parent)
        os.close(fd)
        try:
            torch.save(sample.payload(), staging)
            with open(staging, "rb") as f:
                os.fsync(f.fileno())
            os.replace(staging, path)
        except BaseException:
            Path(staging).unlink(missing_ok=True)
            raise
        rows.append({"id": sample.id, "task_id": sample.task_id, "prefix": sample.record.snapshot.prefix_end})
    return rows


def write_manifest(root, header, rows):
    root = Path(root)
    manifest = root / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    rows = sorted(rows, key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate target entry ids")
    write_json(manifest, {**header, "count": len(rows), "entries": rows})
    return {"path": str(root), "count": len(rows)}


def append_rows(root, rank, rows):
    """Durable per-rank row journal of entries already on disk: a build that fails later can
    still be finalized from it (finalize_cache) instead of being collected again."""
    path = Path(root) / f"rows-rank-{rank:02d}.jsonl"
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def journal_rows(root):
    """Journaled rows whose entry file exists; a torn last line of a killed rank is ignored."""
    root, rows = Path(root), {}
    for path in sorted(root.glob("rows-rank-*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = root / "entries" / (row["id"] + ".pt")
            if entry.is_file():
                rows[row["id"]] = row
    return [rows[key] for key in sorted(rows)]


def finalize_cache(root):
    """Write the manifest of a build that stopped before its own manifest: the header the build
    recorded at its start plus every journaled entry."""
    root = Path(root)
    header = json.loads((root / "header.json").read_text())
    if header.get("schema") != SCHEMA:
        raise ValueError("Not a target cache build header")
    rows = journal_rows(root)
    if not rows:
        raise ValueError("No journaled target entries to finalize")
    return write_manifest(root, header, rows)


def stratified_heldout(rows, count, seed):
    """Held-out ids spread over tasks and episode positions, chosen from manifest rows only.

    `count` is split evenly over the tasks present; the remainder goes to the first tasks of a
    seeded task order, and a task with too few rows passes its surplus on in the same order.
    Within a task the rows sorted by (prefix, id) are taken at the quantiles (i + 0.5) / n, so
    the held-out set reaches from the earliest to the latest cached position of every task.
    The choice depends only on the cache and the seed: every arm on one cache holds out the same ids."""
    rows = list(rows)
    if count <= 0 or not rows:
        return []
    by_task = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    order = sorted(by_task, key=lambda task: digest([seed, "heldout-task-order", task]))
    base, extra = divmod(min(count, len(rows)), len(order))
    quota = {task: base + (index < extra) for index, task in enumerate(order)}
    surplus = 0
    for task in order:
        surplus += max(0, quota[task] - len(by_task[task]))
        quota[task] = min(quota[task], len(by_task[task]))
    while surplus:
        for task in order:
            if surplus and quota[task] < len(by_task[task]):
                quota[task] += 1
                surplus -= 1
    chosen = []
    for task, n in quota.items():
        ranked = sorted(by_task[task], key=lambda row: (row["prefix"], row["id"]))
        chosen += [ranked[int((i + .5) * len(ranked) / n)]["id"] for i in range(n)]
    return sorted(chosen)


def write_cache(root, config, samples, *, source, key=None):
    """Single-process cache (smoke)."""
    return write_manifest(root, cache_header(config, source=source, key=key), write_entries(root, samples))


class TargetCache:
    """Read side of a target cache; entries load lazily. `key` overrides cache_key(config)."""

    def __init__(self, root, config, *, source, key=None):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("source") not in SOURCES:
            raise ValueError(f"{self.root} is not a teacher target cache ({SCHEMA})")
        expected = cache_key(config) if key is None else key
        differing = [k for k in KEY_FIELDS if m.get(k) != expected.get(k)]
        if differing:
            raise ValueError(f"Target cache {self.root} was built under other settings ({', '.join(differing)}); "
                             "it cannot be reused for this run")
        rows = m.get("entries", [])
        if m.get("count") != len(rows) or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Target cache inventory mismatch")
        self.rows = {r["id"]: r for r in rows}
        self.source = source
        self.reserved = set()

    def __len__(self):
        return len(self.rows) - len(self.reserved)

    @property
    def ids(self):
        return sorted(i for i in self.rows if i not in self.reserved)

    def reserve(self, count, seed):
        """Exclude the held-out ids (stratified_heldout) from training draws and return them."""
        chosen = stratified_heldout(self.rows.values(), count, seed)
        self.reserved.update(chosen)
        return chosen

    def read(self, id, *, source=None):
        row = self.rows[id]
        path = self.root / "entries" / (id + ".pt")
        value = torch.load(path, map_location="cpu", weights_only=False)
        sample = TargetSample.from_payload(value, source=source or self.source, id=id)
        if sample.task_id != row["task_id"] or sample.record.snapshot.prefix_end != row["prefix"]:
            raise ValueError("Target cache entry disagrees with its manifest row: " + id)
        return sample

    def sample(self, count, generator):
        ids = self.ids
        count = min(count, len(ids))
        if not count:
            return []
        chosen = torch.randperm(len(ids), generator=generator, device=generator.device)[:count].tolist()
        return [self.read(ids[i]) for i in chosen]


def open_cache(config, *, source="offline"):
    """The configured offline target cache, or None when the config names none."""
    path = config.resources.get("offline_cache")
    return None if path is None else TargetCache(path, config, source=source)
