"""wam_go2/wam/settings.py and config/wam_tasks.json against the released WAM-Policy code."""
import json
from pathlib import Path

import pytest

from wam_go2.wam import observation as ob
from wam_go2.wam import settings, tasks

REPO = Path(__file__).resolve().parents[2]          # the WAM-Policy repository holding deploy/


def test_settings():
    assert settings.INSTRUCTION == "Push the box inside the outlined area."
    assert settings.RGB_PROFILE == ob.RGB_360 == "dual-rgb-360p-v1"
    assert settings.ROOT_STATE is False
    assert settings.DEFAULT_POLICY == "opsd_student" and settings.DEFAULT_POLICY in settings.POLICIES


def test_the_policies_are_the_servers():
    serve = REPO / "src" / "wam_policy" / "inference" / "policies.py"
    if not serve.is_file():
        pytest.skip("deploy/ is not inside the WAM-Policy repository")
    text = serve.read_text()
    assert 'POLICIES = ("teacher", "specialist", "opsd_student", "mopd_student")' in text
    assert sorted(settings.POLICIES) == sorted(("teacher", "specialist", "opsd_student", "mopd_student"))


def test_the_catalogue_is_the_ten_released_tasks_with_their_exact_prompts():
    common = REPO / "src" / "wam_policy" / "common" / "tasks.py"
    if not common.is_file():
        pytest.skip("deploy/ is not inside the WAM-Policy repository")
    released = {}
    exec(compile(common.read_text(), str(common), "exec"), released)   # standard library only
    catalogue = tasks.load()
    assert tasks.ids(catalogue) == list(released["TASKS"])
    for entry in catalogue["tasks"]:
        assert entry["instruction"] == released["INSTRUCTIONS"][entry["id"]], entry["id"]


def test_every_task_enters_policy_at_the_training_start_command():
    for entry in tasks.load()["tasks"]:
        assert entry["entry_ee"] == [0.1934, 0.142, 0.05], entry["id"]      # RAMBO's default command
        assert entry["task_profile_version"] == f"mujoco-{entry['id']}-runtime-v1"
        assert set(entry) == {"id", "instruction", "entry_ee", "task_profile_version"}


def test_tasks_cli(capsys):
    assert tasks.main(["get", "open_room_door", "instruction"]) == 0
    assert capsys.readouterr().out.strip() == "Press down the lever handle, then push the door open."
    assert tasks.main(["get", "push_box", "entry_ee"]) == 0
    assert capsys.readouterr().out.strip() == "0.1934 0.142 0.05"
    assert tasks.main(["get", "nope", "instruction"]) == 2
    assert tasks.main(["list"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 10


def test_the_catalogue_is_valid_json_with_a_schema():
    data = json.loads(tasks.CATALOGUE.read_text())
    assert data["schema"] == "wam-go2-tasks-v1" and len(data["tasks"]) == 10
