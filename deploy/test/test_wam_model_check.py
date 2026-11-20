"""wam_go2/wam/model_check.py: the released model directories are checked before the server loads them."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wam_go2.wam import model_check as mc

DEPLOY = Path(__file__).resolve().parents[1]


def meta(directory, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "wam_policy.json").write_text(json.dumps({"format": mc.FORMAT, **fields}))
    return directory


@pytest.fixture
def release(tmp_path):
    teacher = meta(tmp_path / "stage1_1_teacher", kind="teacher", rgb_profile="dual-rgb-360p-v1")
    student = meta(tmp_path / "stage1_2_opsd_student", kind="student", method="opsd")
    (student / "student_adapter.pt").write_bytes(b"x")
    mopd = meta(tmp_path / "stage2_2_mopd_student", kind="student", method="mopd")
    (mopd / "student_adapter.pt").write_bytes(b"x")
    spec = meta(tmp_path / "stage2_1_specialists" / "push_box", kind="specialist", task="push_box")
    (spec / "adapter.pt").write_bytes(b"x")
    prompts = tmp_path / "prompt_embeddings"
    for task in ("push_box", "open_room_door"):
        (prompts / task).mkdir(parents=True)
        (prompts / task / "embedding.pt").write_bytes(b"x")
    return {"teacher": teacher, "opsd": student, "mopd": mopd, "spec": spec, "prompts": prompts}


def test_released_layout_passes_for_every_policy(release):
    r = release
    assert mc.problems("opsd_student", r["teacher"], r["opsd"], r["prompts"], "push_box") == []
    assert mc.problems("mopd_student", r["teacher"], r["mopd"], r["prompts"], "open_room_door") == []
    assert mc.problems("teacher", r["teacher"], None, r["prompts"], "push_box") == []
    assert mc.problems("specialist", r["teacher"], r["spec"], r["prompts"], "push_box") == []


def test_wrong_directories_are_reported(release, tmp_path):
    r = release
    assert any("mopd student" in p for p in mc.problems("mopd_student", r["teacher"], r["opsd"], r["prompts"], "push_box"))
    assert any("not the Stage 1.1 teacher" in p for p in mc.problems("opsd_student", r["opsd"], r["opsd"], r["prompts"], "push_box"))
    assert any("trained for 'push_box'" in p for p in mc.problems("specialist", r["teacher"], r["spec"], r["prompts"], "open_room_door"))
    assert any("prompt embedding" in p for p in mc.problems("opsd_student", r["teacher"], r["opsd"], r["prompts"], "press_button"))
    assert any("needs an adapter" in p for p in mc.problems("opsd_student", r["teacher"], None, r["prompts"], "push_box"))
    assert any("takes no adapter" in p for p in mc.problems("teacher", r["teacher"], r["opsd"], r["prompts"], "push_box"))
    assert any("no wam_policy.json" in p for p in mc.problems("teacher", tmp_path / "empty", None, r["prompts"], "push_box"))
    assert any("unknown task" in p for p in mc.problems("teacher", r["teacher"], None, r["prompts"], "fly"))
    assert any("not one of" in p for p in mc.problems("sft", r["teacher"], None, r["prompts"], "push_box"))


def test_a_teacher_for_other_images_is_refused(release):
    meta(release["teacher"], kind="teacher", rgb_profile="dual-rgb-720p-v1")
    problems = mc.problems("teacher", release["teacher"], None, release["prompts"], "push_box")
    assert len(problems) == 1 and "dual-rgb-720p-v1" in problems[0]


def test_missing_adapter_weights_are_reported(release):
    (release["opsd"] / "student_adapter.pt").unlink()
    assert any("student_adapter.pt missing" in p
               for p in mc.problems("opsd_student", release["teacher"], release["opsd"], release["prompts"], "push_box"))


def test_cli_exit_codes(release):
    r = release
    base = [sys.executable, "-m", "wam_go2.wam.model_check", "--teacher", str(r["teacher"]),
            "--prompt-root", str(r["prompts"]), "--task", "push_box"]
    ok = subprocess.run(base + ["--policy", "opsd_student", "--adapter", str(r["opsd"])], cwd=DEPLOY,
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = subprocess.run(base + ["--policy", "opsd_student"], cwd=DEPLOY, capture_output=True, text=True)
    assert bad.returncode == 3 and "model_check:" in bad.stdout
