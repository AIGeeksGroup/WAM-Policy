from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from wam_policy.utils.logging import TrainingLogger


def _cfg(tmp_path: Path, enabled: bool | str) -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": str(tmp_path),
        },
        "logging": {
            "wandb": {
                "enabled": enabled,
                "project": "WAM-Policy",
                "entity": "AIGeeksGroup",
                "name": "unit-test-run",
                "id": "unit-test-id",
                "resume": "allow",
                "mode": "offline",
                "dir": "wandb",
                "tags": ["video_ffm", "test"],
            },
        },
    }


def test_training_logger_noops_when_wandb_disabled(tmp_path: Path) -> None:
    logger = TrainingLogger(_cfg(tmp_path, enabled=False))

    logger.log_metrics(3, {"loss/video_ffm": 1.25})
    logger.finish()

    assert not (tmp_path / "wandb").exists()


def test_training_logger_noops_when_wandb_auto_without_api_key(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    logger = TrainingLogger(_cfg(tmp_path, enabled="auto"))

    logger.log_metrics(3, {"loss/video_ffm": 1.25})
    logger.finish()

    assert not (tmp_path / "wandb").exists()


def test_training_logger_auto_initializes_when_api_key_is_set(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls = _install_fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    logger = TrainingLogger(_cfg(tmp_path, enabled="auto"))
    logger.log_metrics(7, {"loss/video_ffm": 0.5})
    logger.finish()

    assert calls["init"]["project"] == "WAM-Policy"
    assert calls["init"]["entity"] == "AIGeeksGroup"
    assert calls["init"]["name"] == "unit-test-run"
    assert calls["init"]["id"] == "unit-test-id"
    assert calls["init"]["resume"] == "allow"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["dir"] == str((tmp_path / "wandb").resolve())
    assert calls["init"]["tags"] == ["video_ffm", "test"]
    assert calls["log"] == [(7, {"loss/video_ffm": 0.5})]
    assert calls["finish"] == 1


def test_training_logger_initializes_and_logs_to_wandb(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls = _install_fake_wandb(monkeypatch)

    logger = TrainingLogger(_cfg(tmp_path, enabled=True))
    logger.log_metrics(7, {"loss/video_ffm": 0.5})
    logger.finish()

    assert calls["init"]["project"] == "WAM-Policy"
    assert calls["init"]["entity"] == "AIGeeksGroup"
    assert calls["init"]["name"] == "unit-test-run"
    assert calls["init"]["id"] == "unit-test-id"
    assert calls["init"]["resume"] == "allow"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["dir"] == str((tmp_path / "wandb").resolve())
    assert calls["init"]["tags"] == ["video_ffm", "test"]
    assert calls["log"] == [(7, {"loss/video_ffm": 0.5})]
    assert calls["finish"] == 1


def test_training_logger_logs_wandb_videos(tmp_path: Path, monkeypatch: Any) -> None:
    calls = _install_fake_wandb(monkeypatch)
    video_path = tmp_path / "teacher.mp4"
    video_path.write_bytes(b"fake-video")

    logger = TrainingLogger(_cfg(tmp_path, enabled=True))
    logger.log_videos({"offline_video_stream/teacher_25step": video_path}, fps=10)
    assert logger.run_url == "https://wandb.test/run"
    logger.finish()

    assert calls["video"] == [(str(video_path), 10, "mp4")]
    assert calls["log"] == [
        (
            None,
            {
                "offline_video_stream/teacher_25step": {
                    "path": str(video_path),
                    "fps": 10,
                    "format": "mp4",
                }
            },
        )
    ]
    assert calls["finish"] == 1


def _install_fake_wandb(monkeypatch: Any) -> dict[str, Any]:
    calls: dict[str, Any] = {"init": None, "log": [], "finish": 0, "video": []}

    class FakeRun:
        url = "https://wandb.test/run"

    def fake_init(**kwargs: Any) -> FakeRun:
        calls["init"] = kwargs
        return FakeRun()

    def fake_log(metrics: dict[str, Any], step: int | None = None) -> None:
        calls["log"].append((step, metrics))

    def fake_video(path: str, fps: int, format: str) -> dict[str, Any]:
        calls["video"].append((path, fps, format))
        return {"path": path, "fps": fps, "format": format}

    def fake_finish() -> None:
        calls["finish"] += 1

    fake_wandb = SimpleNamespace(
        init=fake_init,
        log=fake_log,
        Video=fake_video,
        finish=fake_finish,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return calls
