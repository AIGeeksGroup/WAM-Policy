"""Inspect the downloaded QLM-Bench releases and fit the train-split action normalizer."""
from __future__ import annotations

import json
from pathlib import Path

from wam_policy.data.lerobot import LeRobotRelease, fit_training_normalizer
from wam_policy.data.qlm_bench import SPLITS, releases

NORMALIZER = "action_normalizer.json"
NORMALIZER_STATS = "action_normalizer_fit.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def open_releases(snapshot, names=None):
    """[(catalog entry, LeRobotRelease)] in catalog order."""
    result = []
    for entry in releases(names):
        root = Path(snapshot) / entry["canonical_root"]
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(f"Release {entry['release']} is missing under {snapshot}; "
                                    "run `python -m wam_policy.data download` first")
        result.append((entry, LeRobotRelease(root)))
    return result


def inspect(snapshot, names=None):
    counts = {s: {"episodes": 0, "actions": 0, "grouped_actions": 0} for s in SPLITS}
    per_release = {}
    for entry, release in open_releases(snapshot, names):
        mine = {s: 0 for s in SPLITS}
        for index, episode in enumerate(release.episodes):
            split = release.split_of[episode["episode_uid"]]
            n = episode["length"]
            if episode["task_text"] != entry["instruction"]:
                raise ValueError(f"{entry['release']}: unexpected instruction {episode['task_text']!r}")
            if not release.parquet_path(index).is_file():
                raise FileNotFoundError(release.parquet_path(index))
            mine[split] += 1
            counts[split]["episodes"] += 1
            counts[split]["actions"] += n
            counts[split]["grouped_actions"] += (n // 16) * 16
        per_release[entry["release"]] = {"task": entry["task"], "rgb_profile": release.rgb_profile, **mine}
    return {"splits": counts, "releases": per_release}


def fit(snapshot, cache_root, names=None):
    """Fit the normalizer on the train-split executed actions; writes ``<cache>/action_normalizer.json``."""
    actions = [release.actions(i)
               for _, release in open_releases(snapshot, names)
               for i, episode in enumerate(release.episodes)
               if release.split_of[episode["episode_uid"]] == "train"]
    normalizer, stats = fit_training_normalizer(actions)
    cache_root = Path(cache_root)
    write_json(cache_root / NORMALIZER, {"offset": list(normalizer.offset), "scale": list(normalizer.scale)})
    write_json(cache_root / NORMALIZER_STATS, stats)
    return {"normalizer": {"offset": list(normalizer.offset), "scale": list(normalizer.scale)}, **stats}
