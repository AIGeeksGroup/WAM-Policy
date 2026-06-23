from __future__ import annotations

from pathlib import Path

from wam_policy.utils.config import apply_overrides, load_config
from wam_policy.utils.paths import resolve_repo_root


CONFIG_PATH = Path("configs/experiments/v2/anyflow_video_ffm.yaml")


def test_anyflow_video_ffm_config_loads() -> None:
    cfg = load_config(CONFIG_PATH)

    assert cfg["experiment"]["stage"] == "video_ffm"
    assert cfg["paths"]["lingbot_repo"] == "external/lingbot-va"
    assert cfg["paths"]["anyflow_repo"] == "external/AnyFlow"
    assert cfg["model"]["time_condition"] == "s_src_s_tgt_delta"
    assert cfg["logging"]["wandb"]["enabled"] == "auto"
    assert cfg["logging"]["wandb"]["project"] == "WAM-Policy"
    assert cfg["logging"]["wandb"]["id"] is None
    assert cfg["logging"]["wandb"]["resume"] is None


def test_apply_overrides_updates_nested_values() -> None:
    cfg = load_config(CONFIG_PATH)
    updated = apply_overrides(
        cfg,
        [
            "runtime.device=cuda",
            "train.max_steps=10",
            "runtime.allow_tf32=false",
            "teacher.checkpoint_path=experiments/pretrained_models/lingbot-va/test.pt",
        ],
    )

    assert updated["runtime"]["device"] == "cuda"
    assert updated["train"]["max_steps"] == 10
    assert updated["runtime"]["allow_tf32"] is False
    assert updated["teacher"]["checkpoint_path"].endswith("test.pt")
    assert cfg["runtime"]["device"] == "auto"


def test_resolve_repo_root_from_config() -> None:
    cfg = load_config(CONFIG_PATH)
    repo_root = resolve_repo_root(cfg)

    assert repo_root.name == "WAM-Policy"
    assert (repo_root / "pyproject.toml").is_file()
