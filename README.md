# WAM-Policy: World Action Models with On-Policy Distillation for Quadrupedal Loco-Manipulation

Code repository for WAM-Policy training and evaluation.
The `main` branch contains the WAM-Policy code, `qlm-bench` contains the QLM-Bench simulator and benchmark, and `website` hosts the project page.

> **WAM-Policy: World Action Models with On-Policy Distillation for Quadrupedal Loco-Manipulation**
>
> [Junlin Long](https://github.com/JalenLoong)\*, [Xu Deng](https://github.com/Ethan-Xu-Deng)\*, [Yiran Wang](https://scholar.google.com/citations?user=eg_nBF0AAAAJ&hl=en), [Yue Yang](https://au.linkedin.com/in/yue-yang-982714286)†, [Yang Zhao](https://yangyangkiki.github.io/)†, and [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)‡
>
> \*Equal contribution. †Corresponding authors. ‡Project lead.
>
> #### [Website](https://aigeeksgroup.github.io/WAM-Policy/) | [WAM-Policy code](https://github.com/AIGeeksGroup/WAM-Policy) | [WAM-Policy weights](https://huggingface.co/AIGeeksGroup/WAM-Policy) | [QLM-Bench code](https://github.com/AIGeeksGroup/WAM-Policy/tree/qlm-bench) | [QLM-Bench dataset](https://huggingface.co/datasets/AIGeeksGroup/QLM-Bench)

## ✏️ Citation

Citation will be added after publication.

---

## 🏃 Introduction

Arm-free quadrupeds can perform loco-manipulation, walking on three legs while manipulating with a front leg, yet no existing policy for it follows language instructions. World action models (WAMs) have emerged as effective language-conditioned manipulation policies, but are pretrained for robot arms and sample slowly. WAM-Policy turns one WAM into a quadrupedal loco-manipulation policy via two stages. The first, Cross-Morphology Policy Adaptation and Acceleration, adapts the arm-centric WAM to quadrupeds and distills the transferred policy into a student with one video and two action steps by on-policy self-distillation (OPSD). This student, however, is bounded by its teacher, which only imitates demonstrations. The second stage, Task-Specific Improvement and Merging, therefore tunes the teacher by reinforcement learning (RL) and merges the slow specialists into one few-step student by multi-teacher on-policy distillation (M-OPD). On QLM-Bench, ten simulated tasks with 1,000 demonstrations synthesized without teleoperation, the transferred policy reaches 77.0%, and the OPSD student keeps 76.7% while sampling 14 times faster. Task-specific RL and M-OPD lift it to 85.0%. We also deploy the OPSD student on a real Unitree Go2 Air. WAM-Policy thus achieves a stronger quadrupedal policy at a fraction of the inference cost.

![The OPSD student on the real robot (top) and in simulation (bottom)](./assets/teaser.png)

- **Cross-morphology adaptation and acceleration.** Stage 1 replaces only the visual and action interfaces of a WAM, fine-tunes it on quadruped demonstrations, and accelerates it by a few-step form of on-policy self-distillation (OPSD), in which the same network with more sampling steps labels the student's own samples. The transferred policy reaches 77.0%, and the OPSD student keeps 76.7% (69.3% without distillation) while sampling 14 times faster.
- **Task-specific improvement and merging.** Stage 2 tunes specialist teachers by per-task reinforcement learning and merges them by multi-teacher on-policy distillation (M-OPD), which routes each context to the teacher of its task. On pushing a box and opening a door, the specialists succeed in every episode, against 47% and 53% for the OPSD teacher, and the M-OPD student reaches 90% and 87%, raising the ten-task success from 76.7% to 85.0%, above the OPSD teacher (77.0%).
- **QLM-Bench**, ten simulated loco-manipulation tasks with 1,000 language-annotated demonstrations synthesized without teleoperation, and evaluation in simulation and on a real Unitree Go2 Air, where the OPSD student runs without real-robot data.

### Method

![Stage 1: Cross-Morphology Policy Adaptation and Acceleration](./assets/transfer.png)

![On-Policy Self-Distillation (Stage 1.2, top) and Stage 2 (bottom)](./assets/opd.png)

### QLM-Bench

See the [QLM-Bench code README](https://github.com/AIGeeksGroup/WAM-Policy/tree/qlm-bench) for data preparation and evaluation setup,
and the [QLM-Bench dataset card](https://huggingface.co/datasets/AIGeeksGroup/QLM-Bench) for downloads, formats and splits.

![The ten tasks of QLM-Bench](./assets/qlm_bench_tasks.png)

## ⚙️ Environment Setup

### Installation

Use separate Python environments for the policy and the simulator.

1. **WAM-Policy** (Python ≥ 3.10, CUDA or ROCm). Install PyTorch for your GPU, then install the policy and its data dependencies.

   ```bash
   pip install -e ".[policy,data]"
   pip install --no-deps lerobot==0.3.3
   ```

   The paper uses ROCm for training and the [CUDA 12.8 lockfile](./requirements/policy-cuda128.lock) for evaluation.
2. **QLM-Bench** (Python ≥ 3.11). From the parent directory of the WAM-Policy checkout, clone the simulator into a separate `QLM-Bench` directory.

   ```bash
   git clone -b qlm-bench https://github.com/AIGeeksGroup/WAM-Policy.git QLM-Bench
   ```

   Install the simulator in its own environment with the [QLM-Bench setup guide](https://github.com/AIGeeksGroup/WAM-Policy/tree/qlm-bench#installation). Install the live protocol in the WAM environment with `pip install -e "/path/to/QLM-Bench[live]"`.

Copy the [site configuration](./configs/site.example.env), set `QLM_BENCH_SRC` to the QLM-Bench checkout's `src/` directory and `QLM_PYTHON` to its interpreter, then source the file.

```bash
cp configs/site.example.env configs/site.env
# Edit configs/site.env before sourcing it.
source configs/site.env
```

### Pretrained Checkpoints

The [Hugging Face model repository](https://huggingface.co/AIGeeksGroup/WAM-Policy) documents the teacher, adapters, ablations and prompt embeddings. Quadruped student and specialist adapters load on top of the Stage 1.1 teacher.

```bash
scripts/download.sh weights
scripts/download.sh base
```

The [smoke scripts](./scripts/smoke/) check each stage on a small example. Use `scripts/download.sh prompts` to fetch only the prompt embeddings.

## 🚀 Training

Run the commands from the WAM-Policy checkout after sourcing `configs/site.env`. Later stages use the released checkpoints by default. To chain your own runs, set the input paths in the [site configuration](./configs/site.example.env). The paper trains on one node with eight AMD MI355X GPUs, and the OPSD and M-OPD paper recipes use eight ranks.

### Stage 1.1 Cross-Morphology Policy Transfer

```bash
scripts/stage1_1_transfer/convert_base.sh
scripts/stage1_1_transfer/prepare_data.sh
NPROC=8 scripts/stage1_1_transfer/train.sh
```

Convert the LingBot-VA action head to native 9-D commands and fine-tune on QLM-Bench demonstrations. The [teacher config](./configs/stage1_1_transfer/teacher_sft.json) specifies the paper's SFT run. Set `WAM_TEACHER` to the exported `<run>/teacher/` directory to use your own teacher downstream.

### Stage 1.2 OPSD

```bash
scripts/download.sh reset_pool
scripts/stage1_2_opsd/build_teacher_cache.sh "$WAM_ROOT/runs/opsd-teacher-cache"
scripts/stage1_2_opsd/train.sh "$WAM_ROOT/runs/opsd"
```

OPSD distills the teacher into a 1/2-NFE student using cached and streaming contexts. See the [cache config](./configs/stage1_2_opsd/teacher_cache.json), [paper config](./configs/stage1_2_opsd/opsd.json), and [ablation configs](./configs/stage1_2_opsd/ablations/). The paper uses the S300 EMA export.

### Stage 2.1 Task-Specific RL

```bash
TASK=push_box RUN=runs/task_rl/push_box GPUS=8 EXPORT_ROUND=r14 \
  scripts/stage2_1_task_rl/run_task_rl.sh
TASK=open_room_door RUN=runs/task_rl/open_room_door GPUS=8 EXPORT_ROUND=r04 \
  scripts/stage2_1_task_rl/run_task_rl.sh
```

Train one specialist adapter per task from the Stage 1.1 teacher. The driver prepares scenes from the QLM-Bench checkout. The task configs are [push box](./configs/stage2_1_task_rl/push_box.json) and [open door](./configs/stage2_1_task_rl/open_room_door.json).

### Stage 2.2 M-OPD

```bash
scripts/stage2_2_mopd/build_specialist_cache.sh "$WAM_ROOT/runs/mopd-teacher-cache"
scripts/stage2_2_mopd/train.sh "$WAM_ROOT/runs/mopd"
```

The [M-OPD config](./configs/stage2_2_mopd/mopd.json) starts from the OPSD student and routes specialist teachers by task. Build the Stage 1.2 cache first. The specialist cache replaces the entries for the specialist tasks.

## 📊 Evaluation

### QLM-Bench Standard100 evaluation

```bash
scripts/download.sh standard100
scripts/evaluation/standard100.sh opsd_student runs/eval/opsd \
  "$WAM_WEIGHTS/stage1_2_opsd_student" --seeds 42
```

Use `--seeds 42,43,44` for the paper's 300-episode protocol and see `--help` in the [evaluation CLIs](./src/wam_policy/evaluation/) for runner, trial merging and latency options.

### Reproducing the Paper

| Paper table | Entry point or specification |
|---|---|
| Tab. 1: Success rate (%) on QLM-Bench Standard100 | [Default runner](./scripts/evaluation/standard100.sh) and [accelerated runner](./scripts/evaluation/standard100_accelerated.sh) |
| Tab. 1: Leg-Manip baseline | [Leg-Manip implementation and Standard100 adaptation status](https://github.com/AIGeeksGroup/WAM-Policy/tree/qlm-bench/baselines/leg_manip) |
| Tab. 2: Training contexts | [OPSD config](./configs/stage1_2_opsd/opsd.json) and [OPSD ablation configs](./configs/stage1_2_opsd/ablations/) |
| Tab. 3: Longer training | [Longer training config](./configs/stage1_2_opsd/ablations/longer_training.json) and [teacher cache only config](./configs/stage1_2_opsd/ablations/teacher_cache_only.json) |
| Tab. 4: Distillation objective | [Standard100 runner](./scripts/evaluation/standard100.sh) for the 1/2-NFE teacher at seed 42 and the [900-update OPSD config](./configs/stage1_2_opsd/ablations/longer_training.json). |
| Tab. 5: Sampling steps without distillation | [Standard100 runner](./scripts/evaluation/standard100.sh) with `teacher`, `--video-steps` and `--action-steps` |
| Tab. 6: Proprioception in Stage 1.1 | [SFT config](./configs/ablations/proprioception/sft5000.json) and [training script](./scripts/ablations/proprioception/train.sh). Evaluate with `proprioception_teacher` and set `--teacher` to the [SFT5000 checkpoint](https://huggingface.co/AIGeeksGroup/WAM-Policy/tree/main/ablations/quadruped/rgb_proprioception/s5000). |
| Tab. S1: Ablations on QLM-Bench Standard100 per task | Use the configurations and evaluation entry points for Tabs. 2–6 and report the results by task. |
| Tab. S2: Success rate (%) of an arm WAM on LIBERO | [Arm evaluation](./supplementary/arm_libero/README.md#evaluation) |
| Tab. S3: Ablation studies on LIBERO-Pro2.0: success rate (%) | [Arm configs and tables](./supplementary/arm_libero/README.md#tables) |
| Tab. S4: Success rate (%) on LIBERO after longer training | [Longer training configs](./supplementary/arm_libero/configs/ablations/longer_training/) |
| Tab. S5: Paired comparisons on QLM-Bench | [Paired evaluation CLI](./src/wam_policy/evaluation/compare.py) |
| Tab. S6: Paired comparisons on LIBERO | [Arm paired evaluation](./supplementary/arm_libero/README.md#evaluation) |

## License

- WAM-Policy code on `main` is licensed under [Apache-2.0](./LICENSE). Third-party attributions (LingBot-VA, Wan, torchtitan) are in
  `NOTICE`. `supplementary/arm_libero/` carries its own LICENSE and NOTICE.
- QLM-Bench code on [`qlm-bench`](https://github.com/AIGeeksGroup/WAM-Policy/tree/qlm-bench) is licensed under [CC BY-NC 4.0](https://github.com/AIGeeksGroup/WAM-Policy/blob/qlm-bench/LICENSE).
- Released weights (`AIGeeksGroup/WAM-Policy`): Apache License 2.0. They are derived from the
  Apache-2.0 models `robbyant/lingbot-va-base` (quadruped) and `robbyant/lingbot-va-posttrain-libero-long` (arm).
- The QLM-Bench dataset and the RAMBO controller weights are licensed CC BY-NC 4.0. See the
  [`qlm-bench` branch](https://github.com/AIGeeksGroup/WAM-Policy/tree/qlm-bench) and the dataset card.

## 😘 Acknowledgement

This repository contains code derived from the following third-party works.

- [LingBot-VA](https://github.com/Robbyant/lingbot-va)
- [Wan](https://github.com/Wan-Video/Wan2.1) (via LingBot-VA)

See [NOTICE](./NOTICE) for third-party attribution.
