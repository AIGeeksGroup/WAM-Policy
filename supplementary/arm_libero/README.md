# Generalization to an Arm WAM on LIBERO

Part of [WAM-Policy](../../README.md) (supplementary material); the quadruped code and the paper overview are in the root README.

Code of the supplementary experiment *Generalization to an Arm WAM on LIBERO* of WAM-Policy: Stage 1.2
(OPSD) applied to the arm WAM [LingBot-VA](https://github.com/Robbyant/lingbot-va) on LIBERO-Long and
LIBERO-Pro2.0, with the ablations of the supplementary material. The supplementary tables are referred to by
their captions: *Success rate (%) of an arm WAM on LIBERO* (main arm table), *Ablation studies on LIBERO-Pro2.0*,
*Success rate (%) on LIBERO after longer training*, and *Paired comparisons on LIBERO* (section Statistical Analysis).

This directory is a self-contained project with its own environment. It vendors a subset of LingBot-VA
as `wan_va/`; the quadruped code of WAM-Policy has a package of the same name with different KV-cache
semantics, so **do not install the two into one environment or put both on `PYTHONPATH`**.

## Contents

```
arm_opsd/                 OPSD on LingBot-VA
  config.py               configuration (tensor contract, method, experiment, machine)
  opsd/                   teacher targets and teacher cache, regression losses, learner
  streaming/              streaming contexts: per-rank CPU LIBERO stream, global episode schedule
  sft_only/               SFT only: demonstration reader and the upstream flow-matching loss
  model/                  loading, LoRA, sampler, action/observation codec, KV-cache views
  evaluation/             LIBERO-Long and LIBERO-Pro2.0 evaluation, paired statistics
  sim/                    simulation workers (CPU LIBERO; LIBERO-Pro2.0 lease client)
  runtime.py, cli.py      teacher cache, training, export
  testing/tiny.py         tiny-model smoke of the training chain
wan_va/                   vendored LingBot-VA (modules/model.py, modules/utils.py, utils/scheduler.py)
configs/students/         the five students of the paper (300 updates; SFT only 692)
configs/ablations/        longer training (OPSD 900, Teacher cache only 2,000 updates)
configs/evaluation/       published student checkpoints, comparisons of the paired table
configs/site.example.json machine paths, optional W&B logging and Hugging Face upload settings
scripts/prepare/          CPU LIBERO environment, environment check, reset cases, SFT empty prompt
scripts/train/            prepare, teacher cache, one student (single node, torchrun)
scripts/slurm/            optional Slurm wrapper
scripts/eval/             evaluation host setup, LIBERO-Pro2.0 and LIBERO-Long runners
scripts/smoke/            minimal smokes (also an environment self-check)
```

## Paper terms and implementation

| Paper | Config | Implementation |
|---|---|---|
| Teacher (high-NFE) | — | official weights θ₀, 20 video / 50 action steps, guidance 5.0 / 1.0 (generates the targets) |
| Student w/o distillation | — | θ₀ with the 1/2-NFE sampler (`--policy base_1_2`) |
| OPSD (Ours), η = 0.5 | `students/opsd.json` | `data_mode=streaming`, per update 24 streaming + 24 teacher-cache contexts |
| Teacher cache only, η = 1 | `students/teacher_cache_only.json` | `data_mode=offline_only`, 48 teacher-cache contexts |
| Streaming only, η = 0 | `students/streaming_only.json` | `data_mode=streaming`, 24 streaming contexts, no cache |
| Collected once | `students/collected_once.json` | `data_mode=collect_once`: 209 rounds × 24 = 5,016 contexts collected once by the initial student, then 48 per update from this set |
| SFT only (no teacher) | `students/sft_only.json` | `objective=sft_fm`: official LingBot-VA LIBERO flow-matching recipe, LoRA-256, one demonstration episode per rank and update, 692 updates |
| Teacher cache | `scripts/train/teacher_cache.sh` | the teacher acts in 8 CPU LIBERO streams; about 5,000 (noise, ODE endpoint) targets of whole episodes |
| Student (1/2 NFE) | — | `--policy student_1_2 --student <export>` |

The paper evaluates S300 EMA for the main distillation comparisons and S692 EMA for SFT only.

## Environments

Tested with Python 3.10 on Linux. Two or three environments:

1. **Model environment** (training and evaluation): torch ≥ 2.9 first (paper evaluation: `torch==2.9.0`,
   `torchvision==0.24.0`, CUDA 12.8; training: the site's ROCm build of torch 2.9), then
   ```bash
   pip install --no-deps -r requirements/model.txt
   pip install --no-deps -e .          # or: export PYTHONPATH=$PWD
   ```
2. **CPU LIBERO environment** (training streams, LIBERO-Long evaluation): Python 3.10, MuJoCo 2.3.7,
   OSMesa/llvmpipe rendering (`apt install libosmesa6`). LIBERO and robosuite are fetched at fixed commits:
   ```bash
   bash scripts/prepare/libero_cpu_env.sh <new-env> <source-dir> "$(which python3.10)" <dir-of-libOSMesa.so.8>
   ```
   Its `worker-env.json` is `machine.worker_config`, its `bin/python` is `machine.cpu_python`.
3. **LIBERO-Pro2.0 environment** (evaluation only): the official evaluation package
   ([`zhouxueyang/LIBERO-Pro2.0-Eval`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Eval),
   revision `d1933228`), Python 3.8 and EGL; `scripts/eval/setup.sh` builds it.

External resources (downloaded, not redistributed; see NOTICE for licenses):
[`robbyant/lingbot-va-posttrain-libero-long`](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long)
at revision `0e89d1e7` (θ₀; `python -m arm_opsd.cli download-base-model --output <dir>`), and for SFT only
[`robbyant/libero-long-lerobot`](https://huggingface.co/datasets/robbyant/libero-long-lerobot) at revision
`8c0313b1` (CC BY-NC-SA 4.0; `hf download robbyant/libero-long-lerobot --repo-type dataset --revision 8c0313b1c7cd9fa3798798479cbf59b11af8979d --local-dir <dir>`).

## Smoke tests

```bash
bash scripts/smoke/train_tiny.sh            # training chain on a tiny model: teacher cache -> 5 students -> resume -> export (CPU, ~10 s)
python scripts/smoke/eval_tiny.py --output /tmp/arm-eval-tiny     # evaluation chain on a tiny model and a synthetic environment
EVAL_ROOT=<prepared host> bash scripts/smoke/eval_one_episode.sh  # released OPSD student, one real LIBERO-Long episode
```

## Training

The paper's students were trained on one node with 8 AMD Instinct MI355X GPUs (ROCm, 288 GB per GPU).
Each rank holds a teacher and a student copy of the 5.1B-parameter transformer and drives one CPU LIBERO worker.
Run the commands from `supplementary/arm_libero` with the model environment active.
Copy `configs/site.example.json`, fill in the machine paths, then:

```bash
export SITE=my-site.json PROMPTS=/data/arm/prompts.pt
LIBERO_ROOT=<source-dir>/LIBERO-8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero bash scripts/train/prepare.sh
bash scripts/train/teacher_cache.sh                                   # -> machine.offline_cache (shared by all students)
OUTPUT=/data/arm/runs/opsd bash scripts/train/student.sh configs/students/opsd.json
OUTPUT=/data/arm/runs/teacher_cache_only bash scripts/train/student.sh configs/students/teacher_cache_only.json
OUTPUT=/data/arm/runs/streaming_only bash scripts/train/student.sh configs/students/streaming_only.json
OUTPUT=/data/arm/runs/collected_once bash scripts/train/student.sh configs/students/collected_once.json
OUTPUT=/data/arm/runs/sft_only bash scripts/train/student.sh configs/students/sft_only.json   # needs machine.sft_dataset
OUTPUT=/data/arm/runs/opsd_900 bash scripts/train/student.sh configs/ablations/longer_training/opsd_900.json
OUTPUT=/data/arm/runs/tc_2000 bash scripts/train/student.sh configs/ablations/longer_training/teacher_cache_only_2000.json
```

Students are exported to `<OUTPUT>/students/update-NNNNNN[-ema]/` (`arm_libero.json` + `student_adapter.pt`).
`RESUME=<OUTPUT>/checkpoints/round-NNNNNN` continues an interrupted run; the checkpoint must come from the
same method/experiment configuration. Slurm (submit from `supplementary/arm_libero`, or add
`ARM_LIBERO_ROOT=<path to supplementary/arm_libero>` to `--export`):
`sbatch --export=ALL,SITE=...,PROMPTS=...,OUTPUT=... scripts/slurm/student.sbatch <config>` (`STEP=teacher_cache`
for the cache). Each rank writes `rank-NN/events.jsonl`; the `wandb` section of the
site file mirrors the training curves to W&B, the `hub` section uploads each exported student to a
Hugging Face model repository (both off by default).

## Evaluation

One GPU per process (the paper: RTX 5090). Run everything from `supplementary/arm_libero`. `EVAL_ROOT` keeps
environments, weights, plans and prompts; `setup.sh` builds the model environment `$EVAL_ROOT/envs/model`,
whose Python runs every evaluation command below:

```bash
export EVAL_ROOT=/data/arm-eval PYTHONPATH=$PWD
bash scripts/eval/setup.sh
$EVAL_ROOT/envs/model/bin/python -m arm_opsd.evaluation.cli fetch-student --output $EVAL_ROOT/students   # released OPSD S300 EMA
```

`fetch-student --name` selects one checkpoint from [`released_students.json`](configs/evaluation/released_students.json).
The following command downloads the teacher-cache-only S300 EMA for the evaluation example below.

```bash
$EVAL_ROOT/envs/model/bin/python -m arm_opsd.evaluation.cli fetch-student \
  --name teacher_cache_only_s300_ema --output "$EVAL_ROOT/students"
```

Then per policy (`OUT` is that policy's results directory):

```bash
POLICY=base_1_2 OUT=$EVAL_ROOT/results/base bash scripts/eval/libero_pro2.sh     # 2,520 episodes, ~14.5 h
POLICY=base_1_2 OUT=$EVAL_ROOT/results/base bash scripts/eval/libero_long.sh     # 500 episodes, ~7 h
POLICY=student_1_2 STUDENT=$EVAL_ROOT/students/supplementary/arm_libero_opsd_student \
  OUT=$EVAL_ROOT/results/opsd bash scripts/eval/libero_pro2.sh
POLICY=student_1_2 STUDENT=$EVAL_ROOT/students/ablations/arm_libero/teacher_cache_only/s300_ema \
  OUT=$EVAL_ROOT/results/teacher_cache_only bash scripts/eval/libero_pro2.sh
```

`libero_pro2.sh` ends with `official-summary.txt` (BASE / static / runtime, the official group rates at the
800-step limit of LingBot-VA's client; the 520-step package default is reported as well). Paired statistics:

```bash
$EVAL_ROOT/envs/model/bin/python -m arm_opsd.evaluation.cli paired-table --results results.json \
  --plan $EVAL_ROOT/libero-pro2/plan/plan.json --output paired.json
```

with `results.json` mapping the policy names of `configs/evaluation/paired_libero.json` to results
directories, e.g. `{"base": {"libero_pro2": "<OUT>/libero_pro2/base_1_2", "libero_long": "<OUT>/libero_long/base_1_2"}, ...}`.
It prints every row of *Paired comparisons on LIBERO*: Δ (first minus second, task-macro), +/− discordant pairs, a
bootstrap 95% interval (LIBERO-Pro2.0: whole task instances resampled within each task; LIBERO-Long: cases
within each task) and the exact two-sided McNemar p. Single comparisons: `report` / `report-long`.

## Tables

| Supplementary table | Rows | Configs / students | Evaluation |
|---|---|---|---|
| Success rate (%) of an arm WAM on LIBERO | Student w/o distillation | θ₀ (`base_1_2`) | Pro2.0 + Long |
| Success rate (%) of an arm WAM on LIBERO | Teacher cache only, Streaming only, OPSD | `students/{teacher_cache_only,streaming_only,opsd}.json`, S300 EMA (OPSD released) | Pro2.0 + Long (Streaming only: Pro2.0) |
| Ablation studies on LIBERO-Pro2.0 | Collected once, SFT only | `students/collected_once.json` S300 EMA, `students/sft_only.json` S692 EMA | Pro2.0 |
| Success rate (%) on LIBERO after longer training | Teacher cache only 2,000, OPSD 900 | `ablations/longer_training/*.json`, S2000 / S900 EMA | Pro2.0 + Long |
| Paired comparisons on LIBERO | all comparisons | `configs/evaluation/paired_libero.json` | `paired-table` |

The Hugging Face model repository publishes the OPSD S300 EMA student in `supplementary/arm_libero_opsd_student/`
and the six ablation checkpoints under `ablations/arm_libero/` on `main`. The table above lists the training
configs for reproducing their exports. Teacher (high-NFE) values in the main arm table are taken from
the LingBot-VA and LIBERO-Pro2.0 papers. [WAM-Policy model repository](https://huggingface.co/AIGeeksGroup/WAM-Policy)

## License

Apache License 2.0 (`LICENSE`); derived from LingBot-VA, see `NOTICE`.
