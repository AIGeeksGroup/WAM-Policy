from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import wam_policy.opd.pipelines.v2_anyflow_dual_head as pipeline
from wam_policy.opd.pipelines.v2_anyflow_dual_head import run
from wam_policy.utils.config import load_config


CONFIG_PATH = Path("configs/experiments/v2/anyflow_video_ffm.yaml")
ACTION_CONFIG_PATH = Path("configs/experiments/v2/anyflow_action_ffm.yaml")
VIDEO_DMD_CONFIG_PATH = Path("configs/experiments/v2/anyflow_video_dmd_opd.yaml")
ACTION_DMD_CONFIG_PATH = Path("configs/experiments/v2/anyflow_action_dmd_opd.yaml")


def test_video_ffm_static_dispatch_succeeds() -> None:
    cfg = load_config(CONFIG_PATH)

    result = run(cfg, dry_run=True)

    assert result is not None
    assert result["pipeline"] == "v2_anyflow_dual_head"
    assert result["stage"] == "video_ffm"


@pytest.mark.parametrize(
    ("config_path", "stage"),
    [
        (ACTION_CONFIG_PATH, "action_ffm"),
        (VIDEO_DMD_CONFIG_PATH, "video_dmd"),
        (ACTION_DMD_CONFIG_PATH, "action_dmd"),
    ],
)
def test_new_stage_static_dispatch_succeeds(config_path: Path, stage: str) -> None:
    cfg = load_config(config_path)

    result = run(cfg, dry_run=True)

    assert result is not None
    assert result["pipeline"] == "v2_anyflow_dual_head"
    assert result["stage"] == stage


def test_unknown_stage_raises_value_error() -> None:
    cfg = load_config(CONFIG_PATH)
    cfg = deepcopy(cfg)
    cfg["experiment"]["stage"] = "unknown_stage"

    with pytest.raises(ValueError, match="Unknown stage"):
        run(cfg, dry_run=True)


def test_video_ffm_non_dry_run_dispatches_to_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(CONFIG_PATH)
    called = {}

    def fake_train_video_forward_flowmap(received_cfg):
        called["cfg"] = received_cfg
        return {"step": 1}

    monkeypatch.setattr(
        pipeline,
        "train_video_forward_flowmap",
        fake_train_video_forward_flowmap,
    )

    result = run(cfg, dry_run=False)

    assert result == {"step": 1}
    assert called["cfg"] is cfg
