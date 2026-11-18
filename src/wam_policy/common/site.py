"""Site configuration: every account name, repository id and local path comes from here.

Values are read from environment variables (see ``configs/site.example.env``); nothing else in the
code base names a person, organisation, cluster or host.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULTS = {
    # Released weights (model repository) and its revision.
    "WAM_HF_MODEL_REPO": "AIGeeksGroup/WAM-Policy",
    "WAM_HF_MODEL_REVISION": "main",
    # Upstream LingBot-VA base (VAE, T5 and the 30D transformer converted in Stage 1.1).
    "WAM_BASE_MODEL_REPO": "robbyant/lingbot-va-base",
    "WAM_BASE_MODEL_REVISION": "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c",
    # QLM-Bench dataset repository (read by QLM-Bench's own `qlm_bench.config`, mirrored here).
    "QLM_DATASET_REPO": "AIGeeksGroup/QLM-Bench",
    "QLM_DATASET_REVISION": "main",
    # Optional experiment tracking; disabled unless WANDB_MODE is online/offline.
    "WANDB_MODE": "disabled",
    "WANDB_ENTITY": "",
    "WANDB_PROJECT": "wam-policy",
    # Optional upload target for newly trained checkpoints; empty disables uploading.
    "WAM_UPLOAD_REPO": "",
}


def get(name, default=None):
    """Environment value, falling back to the documented default."""
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    return DEFAULTS.get(name, default) if default is None else default


def path(name, default=None):
    value = get(name, default)
    return None if value in (None, "") else Path(value).expanduser()


def wandb_settings(mode=None, entity=None, project=None):
    """(mode, entity or None, project) with explicit arguments taking precedence."""
    mode = mode or get("WANDB_MODE")
    if mode not in ("disabled", "offline", "online"):
        raise ValueError("WANDB_MODE must be disabled, offline or online")
    return mode, (entity or get("WANDB_ENTITY") or None), (project or get("WANDB_PROJECT"))
