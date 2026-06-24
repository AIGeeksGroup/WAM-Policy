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
    assert "action_dict" in result.stdout
    assert "current_wam_contract" not in result.stdout
    assert "WAM action input contract" in result.stdout


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
    assert "DMD teacher_score_source config is consumed" in result.stdout
    assert "cached_label" in result.stdout


def test_probe_action_dmd_static_source_runs_without_external_repo() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/probe_dmd_opd_runtime.py",
            "probe-static-source",
            "--config",
            "configs/experiments/v2/anyflow_action_dmd_opd.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "WAM DMD OPD Static Alignment" in result.stdout
    assert "action DMD trainer exists" in result.stdout


def test_probe_action_forward_diagnostic_formatter_includes_fix_location() -> None:
    from scripts.tools.probe_action_ffm_runtime import _format_action_forward_diagnostics

    class FakeModel:
        def forward(self, action_dict, update_cache=0, action_mode=True):  # noqa: ANN001
            del action_dict, update_cache, action_mode

    diagnostic = _format_action_forward_diagnostics(
        model=FakeModel(),
        input_dict={"noisy_actions": "a", "timesteps": "t", "z_condition": None},
        a_src="a_src",
        z_condition=None,
    )

    assert "input_keys" in diagnostic
    assert "model_forward_signature" in diagnostic
    assert "fix_location: src/wam_policy/adapters/lingbot_va.py" in diagnostic
    assert "current_wam_contract" in diagnostic
