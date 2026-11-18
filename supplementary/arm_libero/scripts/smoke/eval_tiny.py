"""Evaluation-chain smoke without weights or simulator: a tiny random LingBot-VA plays standard
LIBERO-Long episodes against a synthetic worker (images only, a fixed 15-step episode), through the
same episode loop, result files, summary and paired report as the real evaluation.

    python scripts/smoke/eval_tiny.py --output DIR
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arm_opsd.assets.resets import reset_seed, save_manifest  # noqa: E402
from arm_opsd.evaluation.libero_long import evaluate_libero_long, libero_long_report  # noqa: E402
from arm_opsd.model.kv_views import NativeRole  # noqa: E402
from arm_opsd.sim.libero.cpu_worker import extract_observation  # noqa: E402

CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
TERMINAL_STEP = 15


def frame(step, size=128):
    rows = (np.arange(size, dtype=np.uint16) * 255 // (size - 1)).astype(np.uint8)
    image = np.repeat(rows[:, None, None], size, axis=1).repeat(3, axis=2)
    image[..., 2] = step % 256
    return {"agentview_image": image, "robot0_eye_in_hand_image": image.copy()}


class SyntheticWorker:
    """The CPU worker's surface (reset / step_chunk) with synthetic images; succeeds on even tasks."""
    session_id = "synthetic"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reset(self, case, episode_id):
        self.step, self.success = 0, case["task_id"] % 2 == 0
        return {"episode_id": episode_id, "observation": extract_observation(frame(0)), "reset_info": {}}

    def step_chunk(self, commands, *, episode_id, chunk_id, actor_version):
        applied, observations, terminal = [], [], False
        for command in commands:
            applied.append(np.asarray(command).copy())
            self.step += 1
            observations.append(extract_observation(frame(self.step)))
            if self.step == TERMINAL_STEP:
                terminal = True
                break
        return {"applied_actions": np.stack(applied), "observations": observations, "terminal": terminal,
                "success": terminal and self.success, "timeout": terminal and not self.success,
                "receipt_id": f"{episode_id}-{chunk_id}"}


class SyntheticEncoder:
    """Stands in for the VAE: one latent group for the reset frame, one per four later frames."""
    def reset(self):
        self.frames = 0

    def encode(self, observations):
        observations = [observations] if isinstance(observations, dict) else observations
        groups = 1 if self.frames == 0 else len(observations) // 4
        self.frames += len(observations)
        return torch.full((1, 48, groups, 8, 16), float(observations[-1][CAMERAS[0]][0, 0, 2]) / 255.)


def tiny_role(seed):
    from wan_va.modules.model import WanTransformer3DModel
    torch.manual_seed(seed)
    return NativeRole(WanTransformer3DModel(patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=12,
                                            in_channels=48, out_channels=48, action_dim=30, text_dim=4096, freq_dim=12,
                                            ffn_dim=48, num_layers=1, attn_mode="torch").eval(), "student")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    cases = []
    for task in range(10):
        state = np.array([0., task, 0.])
        fixtures = {"body_pos": np.zeros((1, 3)), "body_quat": np.ones((1, 4))}
        cases.append({"case_id": f"test-task{task:02d}-case0000", "split": "test", "task_id": task,
                      "instruction": f"task {task}", "task_name": f"TASK_{task}", "official_index": 0,
                      "reset_seed": reset_seed(42, "test", task, 0), "init_state": state,
                      "reset_key": f"synthetic-{task}", "fixtures": fixtures})
    save_manifest(output / "resets", cases, counts={"test": 1})
    generator = torch.Generator().manual_seed(9)
    prompts = torch.randn(1, 512, 4096, generator=generator), torch.randn(1, 512, 4096, generator=generator)
    summaries = {}
    for name, seed in (("baseline", 1), ("candidate", 2)):
        summaries[name] = evaluate_libero_long(model_root=None, prompts=None, reset_manifest=output / "resets/manifest.json",
            cpu_python=None, worker_env=None, policy="base_1_2", output_dir=output / name, episodes_per_task=1,
            video_cases_per_task=0, role_override=tiny_role(seed), encoder_override=SyntheticEncoder(),
            prompt_override=lambda task, instruction: prompts, worker_factory=lambda stderr: SyntheticWorker())
    report = libero_long_report(output / "baseline/base_1_2", output / "candidate/base_1_2")
    print(report["text"])
    result = {"episodes": {k: v["completed"] for k, v in summaries.items()}, "macro_sr": {k: v["macro_sr"] for k, v in summaries.items()},
              "paired": {k: report[k] for k in ("n", "macro_delta", "improved", "regressed", "mcnemar_exact_p")}}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
