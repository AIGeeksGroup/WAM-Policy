from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from wam_policy.adapters.robotwin import RoboTwinAdapter
from wam_policy.visualization.offline import select_offline_chunks
from wam_policy.visualization.static import static_visualization_check
from wam_policy.visualization.video import contact_sheet_video, side_by_side, stitch_chunks


def test_select_offline_chunks_uses_continuous_episode_metadata() -> None:
    items = [
        {"sample_id": "b", "episode_index": 1, "chunk_idx": 1},
        {"sample_id": "a", "episode_index": 1, "chunk_idx": 0},
        {"sample_id": "c", "episode_index": 1, "chunk_idx": 2},
    ]

    selection = select_offline_chunks(items, max_chunks=2)

    assert selection.composition_mode == "stitch"
    assert selection.indices == [1, 0]
    assert selection.episode_key == "episode_index"
    assert selection.chunk_key == "chunk_idx"


def test_select_offline_chunks_falls_back_to_grid_without_continuity() -> None:
    items = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]

    selection = select_offline_chunks(items, max_chunks=2)

    assert selection.composition_mode == "grid"
    assert selection.indices == [0, 1]


def test_video_composition_shapes() -> None:
    chunk_a = torch.zeros(2, 4, 5, 3, dtype=torch.uint8)
    chunk_b = torch.ones(3, 4, 5, 3, dtype=torch.uint8)

    stitched = stitch_chunks([chunk_a, chunk_b])
    sheet = contact_sheet_video([chunk_a, chunk_b], columns=2)
    comparison = side_by_side(chunk_a, chunk_b, pad_px=2)

    assert stitched.shape == (5, 4, 5, 3)
    assert sheet.shape == (3, 4, 10, 3)
    assert comparison.shape == (3, 4, 12, 3)


def test_visualization_config_static_checks() -> None:
    cfg = {
        "paths": {
            "repo_root": ".",
            "lingbot_repo": "external/lingbot-va",
            "anyflow_repo": "external/AnyFlow",
        },
        "teacher": {"checkpoint_path": "teacher/transformer"},
        "model": {},
        "data": {"manifest": "datasets/manifests/video.jsonl"},
        "flowmap": {},
        "runtime": {},
        "logging": {"wandb": {"enabled": False}},
        "visualization": {
            "offline_video_stream": {
                "student_checkpoint_path": "experiments/checkpoints/video/latest.pt"
            },
            "robotwin_rollout": {
                "video_checkpoint_path": "experiments/checkpoints/video/latest.pt",
                "action_checkpoint_path": "experiments/checkpoints/action/latest.pt",
            },
        },
    }

    offline = static_visualization_check(cfg, "offline-video-stream")
    robotwin = static_visualization_check(cfg, "robotwin-rollout")

    assert offline["output_dir"] == "results/v2/visualization/offline_video_stream"
    assert robotwin["output_dir"] == "results/v2/visualization/robotwin_rollout"


def test_robotwin_adapter_external_command_contract(tmp_path: Path) -> None:
    runner = tmp_path / "fake_robotwin_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "out = Path(os.environ['WAM_ROBOTWIN_OUTPUT_DIR'])",
                "result = Path(os.environ['WAM_ROBOTWIN_RESULT_JSON'])",
                "paths = {}",
                "keys = ['actual_observation_video', 'student_imagined_video']",
                "keys.append('actual_vs_imagined_video')",
                "for key in keys:",
                "    path = out / f'{key}.mp4'",
                "    path.write_bytes(b'fake')",
                "    paths[key] = str(path)",
                "result.write_text(json.dumps(paths), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    cfg = {
        "paths": {"repo_root": str(tmp_path)},
        "teacher": {"checkpoint_path": "teacher/transformer"},
        "visualization": {
            "robotwin_rollout": {
                "runner": "external_command",
                "command": [sys.executable, str(runner)],
                "video_checkpoint_path": "video.pt",
                "action_checkpoint_path": "action.pt",
            }
        },
    }

    output_dir = tmp_path / "rollout"
    result = RoboTwinAdapter(cfg).rollout_student_episode(
        output_dir=output_dir,
        max_chunks=2,
        fps=10,
        task_name="adjust_bottle",
        seed=0,
    )

    assert Path(result["actual_observation_video"]).is_file()
    assert Path(result["student_imagined_video"]).is_file()
    assert Path(result["actual_vs_imagined_video"]).is_file()
    request = json.loads((output_dir / "robotwin_rollout_request.json").read_text())
    assert request["max_chunks"] == 2
    assert request["task_name"] == "adjust_bottle"
