"""The QLM-Bench demonstration set used by Stage 1.1: ten LeRobot v2.1 releases (one per task),
100 episodes each, split 80/10/10 per release (800/100/100 in total).

The release list lives in ``qlm_bench_catalog.json``; the dataset repository and revision come
from the site configuration (``QLM_DATASET_REPO``, ``QLM_DATASET_REVISION``).
"""
from __future__ import annotations

import json
from pathlib import Path

from wam_policy.common import site
from wam_policy.common.hub import dataset_repo_id

_CATALOG = json.loads(Path(__file__).with_name("qlm_bench_catalog.json").read_text())

SPLITS = ("train", "validation", "test")
SPLIT_PER_RELEASE = dict(_CATALOG["split_per_release"])
EPISODES_PER_RELEASE = _CATALOG["episodes_per_release"]


def releases(names=None):
    """Catalog entries ``{task, release, canonical_root, instruction}`` in catalog order.

    ``names`` optionally selects releases by release name or task name.
    """
    entries = [dict(e) for e in _CATALOG["releases"]]
    if names:
        wanted = set(names)
        entries = [e for e in entries if e["release"] in wanted or e["task"] in wanted]
        if len(entries) != len(wanted):
            raise ValueError(f"Unknown release/task names: {sorted(wanted)}")
    return entries


def dataset_repo(repo=None, revision=None):
    """(repo_id, revision) of the QLM-Bench dataset on the Hugging Face Hub."""
    return (dataset_repo_id(repo or site.get("QLM_DATASET_REPO")),
            revision or site.get("QLM_DATASET_REVISION"))


def download(snapshot, *, repo=None, revision=None, names=None):
    """Download the selected releases into ``snapshot`` (resumable ``snapshot_download``)."""
    from huggingface_hub import snapshot_download
    repo_id, revision = dataset_repo(repo, revision)
    patterns = [e["canonical_root"] + "/*" for e in releases(names)]
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=patterns, local_dir=str(snapshot))
    return {"repo_id": repo_id, "revision": revision, "snapshot": str(snapshot),
            "releases": [e["release"] for e in releases(names)]}
