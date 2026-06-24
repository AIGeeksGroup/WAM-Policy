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
