# WAM-Policy Repo Structure

本文档说明 `WAM-Policy/` repo 的文件结构、目录职责和 Git 跟踪逻辑。它是长期维护文档，不是一次性初始化脚本说明。

核心原则：

- `WAM-Policy/` 本身就是项目根目录，也是唯一的 Git repository。
- WAM-Policy 自己的源码、配置、脚本、文档、测试和小型静态资源进入 Git。
- 外部源码 repo、预训练模型、数据集、实验输出、日志和缓存保留在 repo 内的固定目录下，但通过 `.gitignore` 忽略，不提交到 GitHub。
- 不通过复制整个项目目录来区分 v1、v2、v3；版本差异应放在 `configs/experiments/` 和 `src/wam_policy/opd/pipelines/` 中。

---

## 1. 顶层结构

```text
WAM-Policy/
├── src/
├── configs/
├── scripts/
├── docs/
├── tests/
├── assets/
├── external/
├── experiments/
├── datasets/
├── results/
├── logs/
├── cache/
├── wandb/
├── pyproject.toml
├── README.md
├── .gitignore
├── WAM-Policy_repo_structure.md
└── WAM-Policy_repo_structure_handoff.md
```

`WAM-Policy_repo_structure_handoff.md` 是初始化阶段的交接说明，可以作为历史参考；日常维护以本文档为准。

---

## 2. Git 跟踪边界

### 2.1 应进入 Git

这些目录和文件属于 WAM-Policy 自己的项目内容，应该提交到 Git：

```text
src/
configs/
scripts/
docs/
tests/
assets/
README.md
pyproject.toml
.gitignore
WAM-Policy_repo_structure.md
WAM-Policy_repo_structure_handoff.md
```

用途：

- `src/`：项目源码。
- `configs/`：实验和组件配置。
- `scripts/`：训练、评估、rollout、setup 等命令入口。
- `docs/`：设计文档、实验记录、setup 说明。
- `tests/`：测试代码。
- `assets/`：可提交的小型静态资源，例如图表、小示例和 README 图片。

### 2.2 不应进入 Git

这些目录用于本地或服务器运行资源，内容必须被 `.gitignore` 忽略：

```text
external/
experiments/
datasets/
results/
logs/
cache/
wandb/
```

用途：

- `external/`：外部源码 repo，例如 LingBot-VA、AnyFlow、RoboTwin。
- `experiments/`：预训练模型、训练 checkpoint、运行记录。
- `datasets/`：原始数据、预处理数据、latent、rollout、manifest。
- `results/`：评估结果、可视化、视频、metrics。
- `logs/`：训练和调试日志。
- `cache/`：HuggingFace、下载、预处理等缓存。
- `wandb/`：Weights & Biases 本地运行目录。

注意：空目录不会被 Git 跟踪。本 repo 不使用 `.gitkeep` 强行保留空目录，目录结构由本文档说明。

---

## 3. 目录设计

```text
WAM-Policy/
├── src/
│   └── wam_policy/
│       ├── models/
│       ├── adapters/
│       ├── opd/
│       │   └── pipelines/
│       ├── data/
│       ├── eval/
│       └── utils/
│
├── configs/
│   ├── experiments/
│   │   ├── v1/
│   │   └── v2/
│   ├── components/
│   │   ├── paths/
│   │   ├── model/
│   │   ├── teacher/
│   │   ├── data/
│   │   ├── env/
│   │   └── runtime/
│   └── sweeps/
│
├── scripts/
│
├── docs/
│   ├── designs/
│   └── experiments/
│
├── tests/
├── assets/
│
├── external/
│
├── experiments/
│   ├── pretrained_models/
│   │   ├── lingbot-va/
│   │   ├── anyflow/
│   │   ├── wan2.1/
│   │   └── wan2.2/
│   ├── checkpoints/
│   │   ├── v1/
│   │   └── v2/
│   └── runs/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── latents/
│   ├── rollouts/
│   └── manifests/
│
├── results/
├── logs/
├── cache/
└── wandb/
```

---

## 4. 源码目录

### 4.1 `src/wam_policy/`

WAM-Policy 自己的 Python package 根目录。

```text
src/wam_policy/
├── models/
├── adapters/
├── opd/
│   └── pipelines/
├── data/
├── eval/
└── utils/
```

职责：

- `models/`：student model、dual-head model、flow-map heads、action heads 等。
- `adapters/`：LingBot-VA、AnyFlow、RoboTwin、checkpoint、dataset 等外部系统适配层。
- `opd/`：on-policy distillation 相关训练流程。
- `opd/pipelines/`：v1、v2、v3 等算法版本的 pipeline。
- `data/`：dataset loader、latent loader、rollout loader、manifest parser 等。
- `eval/`：RoboTwin、LIBERO 或自定义任务评估逻辑。
- `utils/`：路径、配置、日志、随机种子、设备管理等通用工具。

---

## 5. 配置目录

### 5.1 `configs/`

实验配置和可复用组件配置目录。

```text
configs/
├── experiments/
│   ├── v1/
│   └── v2/
├── components/
│   ├── paths/
│   ├── model/
│   ├── teacher/
│   ├── data/
│   ├── env/
│   └── runtime/
└── sweeps/
```

职责：

- `configs/experiments/v1/`：v1 最小可行实验 recipe。
- `configs/experiments/v2/`：v2 AnyFlow-style dual-head flow-map OPD recipe。
- `configs/components/paths/`：路径配置片段。
- `configs/components/model/`：student、teacher、head 等模型配置片段。
- `configs/components/teacher/`：LingBot-VA、AnyFlow 或其他 teacher 配置。
- `configs/components/data/`：dataset、latent、rollout、manifest 配置。
- `configs/components/env/`：RoboTwin、LIBERO 或 custom env 配置。
- `configs/components/runtime/`：device、precision、distributed、seed、logging 配置。
- `configs/sweeps/`：ablation 和 sweep 配置。

配置文件中不要写死个人机器路径，例如 `/Users/<username>/...` 或 `/home/<username>/...`。需要机器相关路径时，优先使用环境变量或本地 ignored 配置。

---

## 6. 文档、脚本、测试和资源

### 6.1 `scripts/`

命令行入口和 setup 脚本目录。后续可放：

```text
scripts/train.py
scripts/eval.py
scripts/rollout.py
scripts/setup_local_dirs.sh
scripts/fetch_external_repos.sh
scripts/download_pretrained_models.sh
```

### 6.2 `docs/`

项目文档目录。

```text
docs/
├── designs/
└── experiments/
```

职责：

- `docs/designs/`：算法设计文档，例如 v1 velocity-field MSE、v2 AnyFlow dual-head flow-map OPD。
- `docs/experiments/`：实验记录、ablation 说明、失败日志总结。

### 6.3 `tests/`

测试目录。后续可放：

```text
tests/test_config_loading.py
tests/test_path_resolution.py
tests/test_pipeline_dispatch.py
```

### 6.4 `assets/`

小型静态资源目录。可以提交图表、README 图片、小示例等。不要放大型视频、模型权重或数据集。

---

## 7. 本地资源目录

### 7.1 `external/`

外部源码 repo 目录，不进入 Git。

推荐结构：

```text
external/
├── lingbot-va/
├── AnyFlow/
└── RoboTwin/
```

规则：

- 只放外部源码 repo。
- 不放模型权重。
- 不放 WAM-Policy 自己的源码。
- 如果需要修改外部 repo，优先维护 fork 或 patch，不把外部源码混进 `src/wam_policy/`。

### 7.2 `experiments/`

实验相关本地目录，不进入 Git。

```text
experiments/
├── pretrained_models/
│   ├── lingbot-va/
│   ├── anyflow/
│   ├── wan2.1/
│   └── wan2.2/
├── checkpoints/
│   ├── v1/
│   └── v2/
└── runs/
```

职责：

- `experiments/pretrained_models/`：外部预训练模型和 teacher checkpoints。
- `experiments/checkpoints/`：WAM-Policy 自己训练出的 student checkpoints。
- `experiments/runs/`：训练 run 的 resolved config、metrics、stdout、stderr、manifest 等。

### 7.3 `datasets/`

数据目录，不进入 Git。

```text
datasets/
├── raw/
├── processed/
├── latents/
├── rollouts/
└── manifests/
```

职责：

- `raw/`：原始数据。
- `processed/`：预处理后的数据。
- `latents/`：VAE、video latent、action latent。
- `rollouts/`：student rollout 数据。
- `manifests/`：数据清单、路径索引、checksum、split 文件。

### 7.4 `results/`

评估输出和可视化结果目录，不进入 Git。可放：

```text
results/demo/
results/eval/
results/robotwin/
results/libero/
results/videos/
results/metrics/
```

### 7.5 `logs/`

训练和调试日志目录，不进入 Git。

### 7.6 `cache/`

缓存目录，不进入 Git。后续可以把 HuggingFace 缓存指到这里：

```bash
export HF_HOME=cache/huggingface
```

### 7.7 `wandb/`

Weights & Biases 本地运行目录，不进入 Git。

---

## 8. `.gitignore` 逻辑

当前 `.gitignore` 的核心规则：

```gitignore
/external/*
/datasets/*
/experiments/*
/results/*
/logs/*
/cache/*
/wandb/*
```

这表示这些目录本身可以存在于本地，但目录内部内容不会进入 Git。

同时忽略 Python 缓存、build 输出、测试缓存、虚拟环境、本地环境变量文件、IDE 文件、macOS 文件和 notebook checkpoints。

不要提交以下内容：

- 外部源码 repo 内容。
- 模型权重和 checkpoints。
- 原始数据、预处理数据、latent、rollout。
- 大型视频和评估输出。
- wandb 本地运行文件。
- 机器相关的绝对路径配置。
- `.env`、token、密钥或私有凭证。

---

## 9. 版本迭代规则

不要创建：

```text
WAM-Policy_v1/
WAM-Policy_v2/
WAM-Policy_v3/
src/wam_policy_v1/
src/wam_policy_v2/
opd_v1/
opd_v2/
trainer_v1.py
trainer_v2.py
trainer_final.py
```

正确方式：

```text
configs/experiments/v1/
configs/experiments/v2/
src/wam_policy/opd/pipelines/
docs/designs/
docs/experiments/
```

建议定位：

- v1：LingBot-VA teacher + LingBot-style student + video/action velocity-field MSE + simple dual-head OPD。
- v2：LingBot-VA teacher + AnyFlow-style dual-head flow-map OPD。

---

## 10. Git 使用建议

### 10.1 单 repo 原则

只有一个 Git repo：

```text
WAM-Policy/
```

不要创建外层 `WAM-Policy_workspace/`，也不要把 `external/` 中的外部 repo 内容加入当前 Git。

### 10.2 分支和 tag

建议主线分支：

```text
main
```

建议实验分支：

```text
exp/v1-velocity-mse
exp/v2-anyflow-dual-head
exp/path-and-config-cleanup
```

建议稳定节点 tag：

```text
v1-smoke
v2-anyflow-debug
v2-anyflow-video-only
```

### 10.3 commit 粒度

每次 commit 只做一类事情，例如：

```text
initialize repo skeleton
add v1 experiment configs
add v2 anyflow pipeline skeleton
add lingbot adapter skeleton
add anyflow adapter skeleton
```

避免在一个 commit 中同时修改目录结构、训练逻辑、loss、评估脚本和 README。

---

## 11. 维护清单

新增文件前先判断：

- 是否是 WAM-Policy 自己的源码、配置、脚本、文档、测试或小型资源？如果是，可以进入 Git。
- 是否是外部源码、模型、数据、运行输出、日志或缓存？如果是，应放入 ignored 目录。
- 是否包含个人路径、密钥、token、内部数据路径或敏感信息？如果是，不应提交。
- 是否只是为了让空目录被 Git 跟踪？如果是，不要添加 `.gitkeep` 或 placeholder 文件。

初始化阶段不应该创建：

```text
.gitkeep
placeholder.py
empty.py
TODO.py
placeholder.yaml
placeholder.json
```

后续实现阶段再按实际需要添加源码、配置、测试和脚本。
