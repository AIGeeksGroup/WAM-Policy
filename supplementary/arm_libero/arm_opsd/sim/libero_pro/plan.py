"""The official LIBERO-Pro2.0 grouped-evaluation plan; shared by both Python environments (no torch)."""
from __future__ import annotations

import json
from pathlib import Path

PLAN_SCHEMA = "libero-pro-grouped-eval-plan-v1"
HELD_OUT = {"libero_spatial": (0, 8), "libero_object": (1, 8), "libero_goal": (3, 6), "libero_10": (5, 8)}


def episode_key(spec):
    """Stable, file-safe episode key shared by every policy evaluated on one plan."""
    return "{}-t{:02d}-{}-i{}-s{}".format(spec["suite"], int(spec["task_id"]), str(spec["variant"]).upper(),
                                          int(spec["init_state_id"]), int(spec["seed"]))


def variant_group(variant):
    variant = str(variant).upper()
    return "base" if variant == "BASE" else "static" if variant.startswith("S") else "dynamic"


def load_plan(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Not an official LIBERO-Pro grouped-evaluation plan")
    keys = [episode_key(spec) for spec in plan["episodes"]]
    if len(keys) != len(set(keys)) or len(keys) != plan["rollout_count"]:
        raise ValueError("Plan episodes are not a distinct complete denominator")
    return {**plan, "keys": keys}


def spec_matches(plan_spec, leased_spec):
    """A leased episode must be exactly one planned episode (server may add fields)."""
    return all(leased_spec.get(name) == value for name, value in plan_spec.items())
