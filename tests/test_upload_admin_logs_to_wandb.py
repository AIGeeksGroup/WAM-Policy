from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.tools.upload_admin_logs_to_wandb import upload_admin_logs_to_wandb


def test_upload_admin_logs_to_wandb_uploads_logs_and_checkpoint_listing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    log_dir = tmp_path / "logs/admin_diagnostic/123"
    checkpoint_dir = tmp_path / "experiments/checkpoints/123"
    wandb_dir = tmp_path / "wandb"
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    (log_dir / "status.txt").write_text("FAILED:04_manifest:1\n", encoding="utf-8")
    (log_dir / "04_manifest.log").write_text("manifest failed\n", encoding="utf-8")
    checkpoint_file = checkpoint_dir / "latest.pt"
    checkpoint_file.write_bytes(b"checkpoint")
    slurm_out = tmp_path / "slurm-wam-video-ffm-admin-123.out"
    slurm_err = tmp_path / "slurm-wam-video-ffm-admin-123.err"
    slurm_out.write_text("stdout\n", encoding="utf-8")
    slurm_err.write_text("stderr\n", encoding="utf-8")

    calls = _install_fake_wandb(monkeypatch)

    uploaded = upload_admin_logs_to_wandb(
        job_id="123",
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        project="WAM-Policy",
        name="video-ffm-123",
        run_id="wam-video-ffm-123",
        mode="offline",
        wandb_dir=wandb_dir,
        resume="allow",
        exit_code=1,
        entity="AIGeeksGroup",
        slurm_out=slurm_out,
        slurm_err=slurm_err,
    )

    uploaded_names = {path.name for path in uploaded}
    assert uploaded_names == {
        "status.txt",
        "04_manifest.log",
        "checkpoint_files_123.txt",
        "slurm-wam-video-ffm-admin-123.out",
        "slurm-wam-video-ffm-admin-123.err",
    }
    assert checkpoint_file.name not in uploaded_names

    checkpoint_listing = log_dir / "checkpoint_files_123.txt"
    assert str(checkpoint_file) in checkpoint_listing.read_text(encoding="utf-8")

    assert calls["init"]["project"] == "WAM-Policy"
    assert calls["init"]["entity"] == "AIGeeksGroup"
    assert calls["init"]["name"] == "video-ffm-123"
    assert calls["init"]["id"] == "wam-video-ffm-123"
    assert calls["init"]["resume"] == "allow"
    assert calls["init"]["mode"] == "offline"
    assert calls["artifact"].name == "wam-video-ffm-admin-logs-123"
    assert calls["artifact"].type == "admin-logs"
    assert {file_name for _, file_name in calls["artifact"].files} == uploaded_names
    assert calls["logged_artifact"] is calls["artifact"]
    assert calls["finish"] == 1


def _install_fake_wandb(monkeypatch: Any) -> dict[str, Any]:
    calls: dict[str, Any] = {"init": None, "artifact": None, "logged_artifact": None, "finish": 0}

    class FakeArtifact:
        def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files: list[tuple[str, str]] = []
            calls["artifact"] = self

        def add_file(self, path: str, name: str) -> None:
            self.files.append((path, name))

    class FakeRun:
        def log_artifact(self, artifact: FakeArtifact) -> None:
            calls["logged_artifact"] = artifact

    def fake_init(**kwargs: Any) -> FakeRun:
        calls["init"] = kwargs
        return FakeRun()

    def fake_finish() -> None:
        calls["finish"] += 1

    fake_wandb = SimpleNamespace(init=fake_init, Artifact=FakeArtifact, finish=fake_finish)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return calls
