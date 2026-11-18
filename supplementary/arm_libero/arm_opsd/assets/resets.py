"""LIBERO-Long reset manifests: train / validation / test cases per task. Imported without a model dependency.

Training streams draw train cases; the standard LIBERO-Long evaluation uses the first 50 official
init states of each task (test split); the rank-0 prediction video uses a validation case."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from arm_opsd.sim.libero.worker_protocol import digest

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ROBOSUITE_COMMIT = "fbee5844ff5632f5b5698e204ec5357ca50be0df"
SPLIT_COUNTS = {"train": 100, "validation": 20, "test": 50}
SCHEMA = "arm-libero-resets-v1"


def reset_seed(master_seed, split, task_id, index):
    # The key string is part of the seeding rule of the paper's reset cases; do not change it.
    key = f"opdv4/reset/{int(master_seed)}/{split}/{int(task_id)}/{int(index)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def task_descriptors(libero_root):
    """Read the fixed native task order without importing benchmark/torch."""
    root = Path(libero_root).resolve()
    if not (root / "benchmark/libero_suite_task_map.py").exists():
        root = root / "libero/libero"
    tree = ast.parse((root / "benchmark/libero_suite_task_map.py").read_text())
    mapping = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets))
    result = []
    for index, name in enumerate(mapping["libero_10"]):
        offset = name.find("SCENE") + (8 if "SCENE10" in name else 7)
        result.append({"task_id": index, "task_name": name,
                       "instruction": " ".join(name[offset:].split("_")),
                       "bddl_file": str(root / "bddl_files/libero_10" / f"{name}.bddl"),
                       "official_init_file": str(root / "init_files/libero_10" / f"{name}.pruned_init")})
    if len(result) != 10:
        raise ValueError("expected ten LIBERO-Long tasks")
    return result


def load_official_states(path):
    """Preparation-only import: worker does not import torch or deserialize pickle."""
    import torch
    states = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(states, "detach"):
        states = states.detach().cpu().numpy()
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError(f"invalid official init states: {path}")
    # Flattened simulator state starts with time; time alone cannot define a distinct case.
    if len({digest(x[1:]) for x in states}) != len(states):
        raise ValueError(f"duplicate official init states: {path}")
    return states


def materialize_case(case, manifest_directory):
    result = dict(case)
    root = Path(manifest_directory).resolve()
    relative = Path(result.pop("init_state_file"))
    state_path = (root / relative).resolve()
    if relative.is_absolute() or not state_path.is_relative_to(root) or state_path.suffix != ".npy":
        raise ValueError("unsafe reset state path")
    result["init_state"] = np.load(state_path, allow_pickle=False)
    state = result["init_state"]
    if state.ndim != 1 or state.dtype != np.dtype("float64") or not np.isfinite(state).all():
        raise ValueError("reset state must be finite float64 vector")
    if "fixture_file" in result:
        relative = Path(result.pop("fixture_file"))
        fixture_path = (root / relative).resolve()
        if relative.is_absolute() or not fixture_path.is_relative_to(root) or fixture_path.suffix != ".npz":
            raise ValueError("unsafe reset fixture path")
        with np.load(fixture_path, allow_pickle=False) as saved:
            result["fixtures"] = {name: saved[name].copy() for name in ("body_pos", "body_quat")}
        positions, quaternions = result["fixtures"]["body_pos"], result["fixtures"]["body_quat"]
        if (positions.ndim != 2 or positions.shape[1] != 3 or
                quaternions.shape != (len(positions), 4) or
                any(x.dtype != np.dtype("float64") or not np.isfinite(x).all()
                    for x in (positions, quaternions))):
            raise ValueError("invalid measured fixture array contract")
    if "bddl_file" in result:
        relative = Path(result["bddl_file"])
        bddl_path = (root / relative).resolve()
        if relative.is_absolute() or not bddl_path.is_relative_to(root) or bddl_path.suffix != ".bddl":
            raise ValueError("unsafe reset BDDL path")
        result["bddl_file"] = str(bddl_path)
    return result


def save_manifest(output, cases, *, counts=None, master_seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requested = counts or SPLIT_COUNTS
    seen = set()
    serialized = []
    for case in cases:
        case = dict(case)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", case["case_id"]):
            raise ValueError("unsafe case ID")
        key = case["reset_key"]
        if key in seen:
            raise ValueError("duplicate physical reset across splits/cases")
        seen.add(key)
        state = np.asarray(case.pop("init_state"), dtype=np.float64)
        path = Path("states") / f"{case['case_id']}.npy"
        (output / path).parent.mkdir(exist_ok=True)
        np.save(output / path, state, allow_pickle=False)
        fixtures = case.pop("fixtures", None)
        if fixtures is not None:
            fixture_path = Path("states") / f"{case['case_id']}.fixtures.npz"
            np.savez(output / fixture_path, **fixtures)
            case["fixture_file"] = str(fixture_path)
        case["init_state_file"] = str(path)
        if "bddl_file" in case:
            source = Path(case["bddl_file"])
            relative = Path("bddl") / source.name
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            case["bddl_file"] = str(relative)
        if "official_init_file" in case:
            case["official_init_file"] = "libero_10/" + Path(case["official_init_file"]).name
        serialized.append(case)
    for split, count in requested.items():
        for task_id in range(10):
            actual = sum(c["split"] == split and c["task_id"] == task_id for c in serialized)
            if actual != count:
                raise ValueError(f"{split}/task{task_id}: expected {count}, got {actual}")
    manifest = {"schema": SCHEMA, "suite": "libero_10", "task_order": 0,
                "libero_commit": LIBERO_COMMIT, "robosuite_commit": ROBOSUITE_COMMIT,
                "horizon": "native_chunk800_v1", "master_seed": master_seed,
                "split_counts_per_task": requested, "cases": serialized}
    manifest["content_key"] = digest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(path, *, require_full_splits=False):
    path = Path(path)
    result = json.loads(path.read_text())
    # content_key keys the episode schedule of a run (a resumed run must use the same manifest).
    expected = result.pop("content_key")
    if (result.get("schema") != SCHEMA or result.get("suite") != "libero_10"
            or result.get("task_order") != 0 or result.get("horizon") != "native_chunk800_v1"):
        raise ValueError("unsupported reset schema/suite/order/horizon")
    counts = result["split_counts_per_task"]
    if require_full_splits and counts != SPLIT_COUNTS:
        raise ValueError("training needs train 100 / validation 20 / test 50 reset cases per task")
    if not counts or not set(counts).issubset(SPLIT_COUNTS) or any(not isinstance(v, int) or v <= 0 for v in counts.values()):
        raise ValueError("invalid reset split counts")
    ids, keys, paths = set(), set(), set()
    actual = {(split, task): 0 for split in counts for task in range(10)}
    for case in result["cases"]:
        key = (case["split"], case["task_id"])
        if key not in actual:
            raise ValueError("unknown case split/task")
        if require_full_splits and "fixture_file" not in case:
            raise ValueError("training reset cases need their measured fixtures")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", case["case_id"]):
            raise ValueError("unsafe case ID")
        if not isinstance(case["reset_seed"], int) or not 0 <= case["reset_seed"] < 2**32:
            raise ValueError("invalid reset RNG seed")
        if require_full_splits:
            if case["split"] == "test" and (not isinstance(case.get("official_index"), int)
                    or not 0 <= case["official_index"] < SPLIT_COUNTS["test"]):
                raise ValueError("test case must name one of the fifty distinct official indices")
        for seen, value, label in ((ids, case["case_id"], "case ID"),
                                    (keys, case["reset_key"], "physical reset"),
                                    (paths, case["init_state_file"], "state path")):
            if value in seen:
                raise ValueError(f"duplicate {label}")
            seen.add(value)
        materialize_case(case, path.parent)
        actual[key] += 1
    if require_full_splits:
        for task in range(10):
            indices = [c["official_index"] for c in result["cases"]
                       if c["split"] == "test" and c["task_id"] == task]
            if set(indices) != set(range(50)):
                raise ValueError("official test indices repeat or omit a case")
    if any(n != counts[split] for (split, _), n in actual.items()):
        raise ValueError("reset split/task count mismatch")
    result["content_key"] = expected
    return result


def load_test_manifest(path, *, episodes_per_task=SPLIT_COUNTS["test"]):
    """A manifest usable for the standard evaluation: every task's first official init states, with
    measured fixtures. It may hold the test split alone, since evaluation needs no training resets."""
    result = load_manifest(path)
    if result["split_counts_per_task"].get("test") != episodes_per_task:
        raise ValueError(f"Standard evaluation requires {episodes_per_task} test cases per task")
    for task in range(10):
        rows = [c for c in result["cases"] if c["split"] == "test" and c["task_id"] == task]
        if {c.get("official_index") for c in rows} != set(range(episodes_per_task)):
            raise ValueError("Test cases must be the leading official init states of each task")
        if any("fixture_file" not in c or "instruction" not in c for c in rows):
            raise ValueError("Test cases require measured fixtures and the task instruction")
    return result
