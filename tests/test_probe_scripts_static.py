from __future__ import annotations

import subprocess
import sys


def test_probe_action_ffm_static_source_runs_without_external_repo() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/probe_action_ffm_runtime.py",
            "probe-static-source",
            "--config",
            "configs/experiments/v2/anyflow_action_ffm.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "WAM Action Adapter Alignment" in result.stdout
    assert "adapter exposes forward_action_flowmap" in result.stdout


def test_probe_dmd_opd_static_source_runs_without_external_repo() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/probe_dmd_opd_runtime.py",
            "probe-static-source",
            "--config",
            "configs/experiments/v2/anyflow_video_dmd_opd.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "WAM DMD OPD Static Alignment" in result.stdout
    assert "video DMD trainer exists" in result.stdout
