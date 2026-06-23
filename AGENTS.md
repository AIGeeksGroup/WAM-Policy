# WAM-Policy Agent Guide

This file is the operating guide for future agents working in this repository.
Treat it as the short, repo-local source of truth for boundaries, design intent,
and current implementation state.

## Environment Rules

Allowed in lightweight agent sessions:

- Read and edit files inside this `WAM-Policy` repository.
- Run lightweight commands such as `pwd`, `ls`, `tree -L 2/3`, `find . -maxdepth 3`,
  `git status`, `git diff`, `git log`, and tightly scoped `rg` searches.
- Prepare configs, scripts, debug probes, and Slurm `sbatch`/`srun` templates.
- Run very small syntax or import checks only when they are clearly lightweight.

Do not run in lightweight agent sessions or shared login environments:

- Training, finetuning, evaluation, RoboTwin simulation, model inference, or GPU work.
- `torchrun`, `deepspeed`, `accelerate launch`, distributed jobs, or long Python jobs.
- Full `pytest`, benchmarks, rollouts, or environment simulation.
- Heavy `pip install`, large builds, `make -j`, `ninja`, or dependency compilation.
- Recursive scans of `/shared`, external repos, checkpoints, datasets, logs, wandb,
  `.venv`, `__pycache__`, or other large directories.
- Large downloads, copies, extracts, checkpoint moves, `nohup`, background jobs,
  `tmux`, `screen`, long shell loops, or `sudo`.

If a task needs GPU, LingBot runtime loading, checkpoint loading, training,
RoboTwin, or large debug, prepare a compute-node `sbatch`/`srun` script and
explain how to submit it.

## Repository Boundaries

WAM-Policy is a single Git repository rooted at this directory.

Tracked project content:

- `src/`, `configs/`, `scripts/`, `docs/`, `tests/`, `assets/`
- `README.md`, `pyproject.toml`, `.gitignore`
- repo maintenance docs such as `WAM-Policy_repo_structure.md`

Ignored local/runtime content:

- `external/`, `experiments/`, `datasets/`, `results/`, `logs/`, `cache/`, `wandb/`

Rules:

- Do not commit external source repos, checkpoints, model weights, datasets,
  generated results, logs, caches, or wandb files.
- Do not add `.gitkeep`, placeholder files, or empty stubs just to track empty
  ignored directories.
- External source repos are expected at:
  - `external/lingbot-va`
  - `external/AnyFlow`
- WAM-Policy code should not be mixed into `external/`, and external repo edits
  should be handled through a fork or patch strategy.

## WAM-Policy v2 Design Intent

WAM-Policy_v2 is a LingBot-VA-compatible, AnyFlow-style dual-head flow-map
student.

High-level design:

- Teacher: frozen RoboTwin post-trained LingBot-VA.
- Student: initialized from the teacher checkpoint, architecture-compatible with
  LingBot-VA, but operator semantics change from instantaneous velocity fields
  to two-time flow-map transitions or average velocities.
- Goal: reduce video/action sampling steps while avoiding naive coarse Euler
  step reduction.
- Keep LingBot-VA interfaces intact where possible: video/action token shapes,
  VAE/text encoder assumptions, causal ordering, KV cache, chunk interleaving,
  and RoboTwin action format.

Design stages:

- Stage 0: student initialization from teacher checkpoint.
- Stage 1: dual-head Forward Flow-map Training.
- Stage 2: dual-head On-Policy Flow-map Distillation with DMD correction.

Current implemented scope is narrower than the full design:

- Implemented: Stage 1 video-head Forward Flow-map Training (`video_ffm`).
- Not implemented yet: action-head FFM, video/action DMD, OPD rollout,
  RoboTwin online evaluation, distributed training, FSDP/DeepSpeed/ZeRO.

## Time Convention and Math Invariants

The design document may describe time using `tau/rho`, but current WAM-Policy
code intentionally uses LingBot-style `s` naming at trainer-facing boundaries.

Code convention:

- `s = 0`: noise.
- `s = 1`: clean data.
- Forward flow-map pair: `0 <= s_src < s_tgt <= 1`.
- Use `s_src`, `s_tgt`, and `delta_s = s_tgt - s_src` in code-facing APIs.
- Do not introduce `tau`/`rho` names into trainer, dataset, loss, or pipeline
  APIs unless deliberately writing design-only documentation.

Video FFM construction:

- `Z_s = (1 - s) * eps + s * Z_clean`
- `target_u = (Z_tgt - Z_src) / (s_tgt - s_src)`
- `pred_u.shape == target_u.shape == Z_clean.shape`
- First implementation uses average-velocity prediction, not endpoint
  prediction.

## Current Implementation State

Main entrypoint:

- `scripts/train.py`
- config: `configs/experiments/v2/anyflow_video_ffm.yaml`
- pipeline: `src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py`

Current non-dry-run path:

```text
config
-> bootstrap external repos
-> v2_anyflow_dual_head pipeline
-> VideoFFMTrainer
-> video latent manifest dataset
-> s_src/s_tgt flow-map pair
-> LingBotVideoFlowMapStudent
-> LingBotVAAdapter video hook
-> average-velocity loss
-> optimizer step
-> checkpoint save
```

Current code already includes:

- Config loading, `--set` overrides, and static dry-run dispatch.
- AnyFlow-style `s_src/s_tgt` pair utilities.
- Video latent JSONL manifest dataset and collate.
- Video FFM average-velocity loss.
- LingBot-VA adapter boundary and student wrapper.
- WAM-side video forward hook installation.
- Two-time conditioner injection into LingBot `_time_embed`.
- Latent/context shape validation for the fake-model smoke path.
- Checkpoint saving helper.

Known server gaps:

- Real LingBot runtime dependencies and ROCm/PyTorch environment are not solved.
- Real `wan_va.modules.utils.load_transformer` and checkpoint loading are not
  validated.
- Real LingBot model `patch_size`, `_time_embed` signature,
  `condition_embedder.time_proj.out_features`, and forward return shape may
  differ from the fake test model.
- Real `z_clean` and `text_emb` manifest shapes still need server validation.

## Adapter and Dependency Rules

External repo internals must stay behind adapter boundaries.

- Only `src/wam_policy/adapters/lingbot_va.py` should directly bind to
  LingBot-VA internals.
- Only `src/wam_policy/adapters/anyflow.py` should directly bind to AnyFlow
  internals if such imports become necessary.
- Trainer, stages, losses, datasets, and pipelines must not directly import
  external repo packages.
- Use `bootstrap_external_repos(cfg)` to add `external/lingbot-va` and
  `external/AnyFlow` to `sys.path`.

When real LingBot binding mismatches appear, fix the adapter rather than
spreading LingBot-specific logic into trainer or pipeline code.

## Safe Next Steps

The next useful work should be compute-node oriented:

1. Prepare or run a minimal `sbatch`/`srun` probe to confirm `wan_va` import.
2. Probe `LingBotVAAdapter.load_pretrained_video_action_model()` with the real
   checkpoint on a compute node.
3. Inspect real LingBot `patch_size`, `_time_embed` signature,
   `condition_embedder.time_proj.out_features`, and video forward return shape.
4. Validate one tiny manifest sample: `z_clean` shape and `context.text_emb`
   shape.
5. Only after those pass, run a tiny compute-node training job such as
   `train.max_steps=2`.

Run these through compute-node `sbatch`/`srun` jobs rather than directly from
lightweight agent sessions.

## Lightweight Verification Policy

For lightweight verification, prefer:

- Static dry-run only when needed:
  `python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static`
- Small, targeted file reads or scoped searches.
- Targeted unit tests only if they are clearly lightweight and relevant.

Avoid full test suites or commands that may load real models, datasets, external
runtime dependencies, or GPUs.
