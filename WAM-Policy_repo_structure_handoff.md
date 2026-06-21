# WAM-Policy Repo Structure

本文档用于 hand off 给 Codex，在 macOS 本地初始化全新的 `WAM-Policy/` 项目骨架，并约定后续如何使用、迭代和维护这个 repo。

当前决策是：

> `WAM-Policy/` 本身就是项目根目录，也是唯一的 Git repository。  
> 不再额外创建 `WAM-Policy_workspace/` 外层目录。  
> 外部源码 repo、预训练模型、数据集、实验输出、日志和缓存都放在 `WAM-Policy/` 内部的固定入口目录下，但通过 `.gitignore` 忽略，不提交到 GitHub。

这版结构参考 AnyFlow / LingBot-VA 这类 research repo 的组织方式：源码、配置、脚本、文档留在 repo 内并进入 Git；大模型、数据和运行产物留在 repo 内的标准路径下，但不进入 Git。

---

## 1. 初始化任务边界

Codex 当前只需要初始化 **文件夹骨架和少量非代码文档/配置入口**。

### 1.1 必须做

在一个空的 `WAM-Policy/` 文件夹中创建目录结构。

允许创建以下非代码文件：

```text
README.md
.gitignore
pyproject.toml
WAM-Policy_repo_structure.md
```

如果用户要求，也可以创建 `docs/` 下的说明文档，例如：

```text
docs/setup.md
docs/server_setup.md
docs/repo_structure.md
```

### 1.2 当前不要做

初始化骨架时 **不要创建任何 Python 代码文件**，包括但不限于：

```text
src/wam_policy/*.py
src/wam_policy/**/**/*.py
scripts/train.py
scripts/eval.py
scripts/rollout.py
scripts/setup_local_dirs.sh
scripts/fetch_external_repos.sh
scripts/download_pretrained_models.sh
```

这些文件是后续实现阶段才创建的，不属于当前 folder-only skeleton 初始化任务。

### 1.3 不要创建 placeholder 文件

不要为了让 Git 跟踪空目录而创建：

```text
.gitkeep
placeholder.py
empty.py
TODO.py
placeholder.yaml
placeholder.json
README.md inside every folder
```

空目录在 Git 中不会被跟踪，这是可以接受的。repo 结构由本文档和 README 说明，不需要用 `.gitkeep` 强行保留所有空目录。

---

## 2. 顶层结构

目标项目根目录为：

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
└── WAM-Policy_repo_structure.md
```

这些目录分成两类。

### 2.1 进入 Git 的源码和说明目录

这些目录用于放 WAM-Policy 自己的源码、配置、脚本、文档和测试，应该进入 Git：

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
```

### 2.2 不进入 Git 的本地资源目录

这些目录用于放外部源码、预训练模型、数据、实验输出、日志和缓存，必须被 `.gitignore` 忽略：

```text
external/
experiments/
datasets/
results/
logs/
cache/
wandb/
```

这些目录可以存在于本地和服务器，但内容不应该提交到 GitHub。

---

## 3. 完整目录设计

初始化时只创建目录，不创建代码文件。

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
├── wandb/
│
├── pyproject.toml
├── README.md
├── .gitignore
└── WAM-Policy_repo_structure.md
```

---

## 4. 各目录职责

### 4.1 `src/wam_policy/`

WAM-Policy 自己的 Python package。当前只创建目录，不创建代码文件。

后续实现阶段预计结构如下：

```text
src/wam_policy/
├── models/
├── adapters/
├── opd/
│   ├── pipelines/
│   ├── trainer.py
│   ├── stages.py
│   ├── losses.py
│   └── buffers.py
├── data/
├── eval/
└── utils/
```

职责说明：

- `models/`：WAM-Policy student model、dual-head model、flow-map heads、action heads 等。
- `adapters/`：LingBot-VA、AnyFlow、RoboTwin、checkpoint、dataset 等外部系统的适配层。
- `opd/`：on-policy distillation 相关训练流程。
- `opd/pipelines/`：v1、v2、v3 等算法版本的主流程文件。
- `data/`：dataset loader、latent loader、rollout loader、manifest parser 等源码。
- `eval/`：RoboTwin / LIBERO / 自定义任务评估逻辑。
- `utils/`：路径、配置、日志、随机种子、设备管理等通用工具。

### 4.2 `configs/`

实验配置目录。当前只创建目录，不创建 YAML 文件。

建议后续结构：

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

职责说明：

- `configs/experiments/v1/`：WAM-Policy v1 实验 recipe。
- `configs/experiments/v2/`：WAM-Policy v2 AnyFlow-style dual-head flow-map distillation recipe。
- `configs/components/paths/`：本地路径、服务器路径、默认路径配置。
- `configs/components/model/`：student / teacher / head 相关配置片段。
- `configs/components/teacher/`：LingBot-VA teacher、AnyFlow teacher 或其他 teacher 配置。
- `configs/components/data/`：dataset、latent、rollout、manifest 配置。
- `configs/components/env/`：RoboTwin / LIBERO / custom env 配置。
- `configs/components/runtime/`：device、precision、distributed、seed、logging 配置。
- `configs/sweeps/`：ablation / sweep 配置。

### 4.3 `scripts/`

命令行入口和 setup 脚本目录。当前只创建目录，不创建脚本文件。

后续实现阶段预计会创建：

```text
scripts/train.py
scripts/eval.py
scripts/rollout.py
scripts/setup_local_dirs.sh
scripts/fetch_external_repos.sh
scripts/download_pretrained_models.sh
```

职责说明：

- `train.py`：训练入口。
- `eval.py`：评估入口。
- `rollout.py`：rollout / relabel / teacher labeling 入口。
- `setup_local_dirs.sh`：创建本地 ignored 目录。
- `fetch_external_repos.sh`：clone LingBot-VA、AnyFlow、RoboTwin 等外部源码 repo。
- `download_pretrained_models.sh`：下载 LingBot-VA、AnyFlow、Wan2.1 / Wan2.2 等模型权重。

### 4.4 `docs/`

项目文档目录。

```text
docs/
├── designs/
└── experiments/
```

职责说明：

- `docs/designs/`：算法设计文档，例如 v1 velocity-field MSE、v2 AnyFlow dual-head flow-map OPD。
- `docs/experiments/`：实验记录、ablation 说明、失败日志总结。
- `docs/setup.md`：后续可以写本地 setup 说明。
- `docs/server_setup.md`：后续可以写服务器 setup 说明。
- `docs/repo_structure.md`：后续可以把本文档整理成 repo 内正式结构说明。

### 4.5 `tests/`

测试目录。当前只创建目录，不创建测试文件。

后续可以放：

```text
tests/test_config_loading.py
tests/test_path_resolution.py
tests/test_pipeline_dispatch.py
```

### 4.6 `assets/`

小型静态资源目录。可以进入 Git。

可放内容：

```text
assets/figures/
assets/readme_images/
assets/small_examples/
```

不要放大型视频、模型权重、数据集。

### 4.7 `external/`

外部源码 repo 目录，不进入 Git。

后续服务器或本地可以 clone：

```text
external/
├── lingbot-va/
├── AnyFlow/
└── RoboTwin/
```

规则：

- `external/` 只放外部源码 repo。
- 不放模型权重。
- 不放 WAM-Policy 自己的源码。
- 不把 external 里的任何内容提交到 Git。

### 4.8 `experiments/`

实验相关本地目录，不进入 Git。

```text
experiments/
├── pretrained_models/
├── checkpoints/
└── runs/
```

职责说明：

- `experiments/pretrained_models/`：外部预训练模型和 teacher checkpoints。
- `experiments/checkpoints/`：WAM-Policy 自己训练出的 student checkpoints。
- `experiments/runs/`：训练 run 的 resolved config、metrics、stdout、stderr、manifest 等。

建议模型路径：

```text
experiments/pretrained_models/lingbot-va/lingbot-va-base/
experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/
experiments/pretrained_models/anyflow/AnyFlow-Wan2.1-T2V-1.3B-Diffusers/
experiments/pretrained_models/wan2.1/
experiments/pretrained_models/wan2.2/
```

### 4.9 `datasets/`

数据目录，不进入 Git。

```text
datasets/
├── raw/
├── processed/
├── latents/
├── rollouts/
└── manifests/
```

职责说明：

- `raw/`：原始数据。
- `processed/`：预处理后的数据。
- `latents/`：VAE / video latent / action latent。
- `rollouts/`：student rollout 数据。
- `manifests/`：数据清单、路径索引、checksum、split 文件。

### 4.10 `results/`

评估输出和可视化结果，不进入 Git。

可放：

```text
results/demo/
results/eval/
results/robotwin/
results/libero/
results/videos/
results/metrics/
```

### 4.11 `logs/`

训练和调试日志，不进入 Git。

### 4.12 `cache/`

缓存目录，不进入 Git。

建议后续将 HuggingFace 缓存指到这里：

```bash
export HF_HOME=cache/huggingface
```

### 4.13 `wandb/`

Weights & Biases 本地运行目录，不进入 Git。

---

## 5. `.gitignore` 规范

Codex 需要创建 `.gitignore`，内容建议为：

```gitignore
# External source repositories
/external/*

# Local datasets and processed data
/datasets/*

# Experiments, checkpoints, pretrained models, and run outputs
/experiments/*

# Evaluation outputs
/results/*

# Logs, caches, and local trackers
/logs/*
/cache/*
/wandb/*

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Python packaging/build outputs
build/
dist/
*.egg-info/
.eggs/

# Test/type-check cache
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual environments and local env files
.env
.venv/
venv/
ENV/

# IDE / OS files
.DS_Store
.idea/
.vscode/

# Notebook checkpoints
.ipynb_checkpoints/
```

注意：

- `/external/*` 忽略 external 里的外部 repo 内容。
- `/experiments/*` 忽略预训练模型、checkpoints 和 runs。
- `/datasets/*` 忽略所有数据。
- 不要把模型权重、数据集、rollout、latent、视频、wandb 输出提交到 Git。

---

## 6. `pyproject.toml` 最小规范

初始化阶段允许创建 `pyproject.toml`，但不要写复杂依赖。建议只写最小项，后续再扩展。

建议内容：

```toml
[project]
name = "wam-policy"
version = "0.0.0"
description = "WAM-Policy: On-policy distillation for LingBot-VA style world-action models."
requires-python = ">=3.10"

[tool.ruff]
line-length = 100
```

如果用户要求彻底 folder-only，也可以暂时不创建 `pyproject.toml`。

---

## 7. `README.md` 最小规范

初始化阶段允许创建一个很短的 `README.md`，用于说明项目入口。不要写大量未实现的运行命令。

建议内容包括：

```text
# WAM-Policy

WAM-Policy is a research repo for distilling LingBot-VA style video-action world-action models with on-policy distillation.

This repo uses a single-root layout. Source code, configs, scripts, docs, and tests are tracked by Git. External source repos, pretrained models, datasets, checkpoints, runs, logs, and caches live under fixed local directories but are ignored by Git.

See WAM-Policy_repo_structure.md for the current repo structure and maintenance rules.
```

---

## 8. 路径约定

所有路径都以 `WAM-Policy/` repo 根目录为基准。

建议后续 config 中使用：

```yaml
paths:
  repo_root: .
  external_root: external
  pretrained_model_root: experiments/pretrained_models
  checkpoint_root: experiments/checkpoints
  run_root: experiments/runs
  dataset_root: datasets
  result_root: results
  log_root: logs
  cache_root: cache

  lingbot_va_repo: external/lingbot-va
  anyflow_repo: external/AnyFlow
  robotwin_repo: external/RoboTwin

  lingbot_va_base_ckpt: experiments/pretrained_models/lingbot-va/lingbot-va-base
  lingbot_va_robotwin_ckpt: experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin
  anyflow_wan21_ckpt: experiments/pretrained_models/anyflow/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
```

不要在 config 里写死个人机器路径，例如：

```text
/Users/<username>/...
/home/<username>/...
```

如果确实需要机器相关路径，放到本地 ignored config 或环境变量里，不提交到 Git。

---

## 9. v1 / v2 迭代规则

WAM-Policy 不通过复制完整项目文件夹来迭代。

不要创建：

```text
WAM-Policy_v1/
WAM-Policy_v2/
WAM-Policy_v3/
```

也不要在 repo 内创建：

```text
src/wam_policy_v1/
src/wam_policy_v2/
opd_v1/
opd_v2/
trainer_v1.py
trainer_v2.py
trainer_final.py
```

正确方式是：

```text
configs/experiments/v1/
configs/experiments/v2/
src/wam_policy/opd/pipelines/
```

后续代码实现阶段建议：

```text
src/wam_policy/opd/pipelines/v1_velocity_mse.py
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py
```

但初始化阶段不要创建这些 `.py` 文件。

### 9.1 v1 定位

v1 是最小可行实验：

```text
LingBot-VA teacher
+ LingBot-style student
+ video/action velocity-field MSE
+ simple dual-head OPD
```

配置后续放在：

```text
configs/experiments/v1/
```

### 9.2 v2 定位

v2 是 AnyFlow-style dual-head flow-map OPD：

```text
LingBot-VA teacher
+ flow-map student heads
+ video head forward flow-map training
+ action head forward flow-map training
+ video head on-policy flow-map distillation
+ action head on-policy flow-map distillation
```

配置后续放在：

```text
configs/experiments/v2/
```

pipeline 后续放在：

```text
src/wam_policy/opd/pipelines/
```

---

## 10. 外部 repo 管理规则

外部源码 repo 放在：

```text
external/
```

推荐后续使用脚本 clone：

```text
external/lingbot-va/
external/AnyFlow/
external/RoboTwin/
```

规则：

- 外部源码不提交到 WAM-Policy Git。
- 不在 WAM-Policy 内复制大段外部源码。
- WAM-Policy 自己只写 adapter，调用或引用外部 repo。
- 如果需要修改外部 repo，优先记录 patch 或 fork，不要直接把外部 repo 内容混进 WAM-Policy 源码。

---

## 11. 预训练模型和 checkpoint 管理规则

外部模型权重放在：

```text
experiments/pretrained_models/
```

WAM-Policy 自己训练出来的 checkpoint 放在：

```text
experiments/checkpoints/
```

不要把任何权重提交到 Git。

推荐路径：

```text
experiments/pretrained_models/lingbot-va/lingbot-va-base/
experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/
experiments/pretrained_models/anyflow/AnyFlow-Wan2.1-T2V-1.3B-Diffusers/
experiments/pretrained_models/wan2.1/
experiments/pretrained_models/wan2.2/

experiments/checkpoints/v1/
experiments/checkpoints/v2/
```

---

## 12. 数据管理规则

所有真实数据、预处理数据、latent、rollout 和 relabel 数据都放在：

```text
datasets/
```

不要把数据提交到 Git。

建议路径：

```text
datasets/raw/
datasets/processed/
datasets/latents/
datasets/rollouts/
datasets/manifests/
```

`datasets/manifests/` 中的 manifest 如果很小、且不包含私有路径或敏感信息，未来可以考虑复制一份到 `configs/` 或 `docs/` 中作为可追踪示例。但初始化阶段不创建 manifest 文件。

---

## 13. 实验输出管理规则

训练 run 输出放在：

```text
experiments/runs/<run_id>/
```

评估结果放在：

```text
results/<run_id>/
```

日志放在：

```text
logs/<run_id>/
```

建议每次运行保存：

```text
resolved_config.yaml
metrics.jsonl
manifest.json
stdout.log
stderr.log
```

这些文件不进入 Git。重要结论用简短 markdown 总结到：

```text
docs/experiments/
```

---

## 14. Git 使用和维护规则

### 14.1 只有一个 Git repo

Git repo 根目录就是：

```text
WAM-Policy/
```

不要再创建外层：

```text
WAM-Policy_workspace/
```

也不要在 `external/` 里把外部 repo 的内容加入当前 Git。

### 14.2 推荐分支策略

稳定主线：

```text
main
```

实验分支：

```text
exp/v1-velocity-mse
exp/v2-anyflow-dual-head
exp/path-and-config-cleanup
```

稳定节点打 tag：

```text
v1-smoke
v2-anyflow-debug
v2-anyflow-video-only
```

### 14.3 commit 粒度

每次 commit 只做一类事情，例如：

```text
initialize repo skeleton
add v1 experiment configs
add v2 anyflow pipeline skeleton
add lingbot adapter skeleton
add anyflow adapter skeleton
```

不要一个 commit 里同时改目录结构、训练逻辑、loss、评估脚本和 README。

---

## 15. 初始化命令

在 macOS 本地，用户会删除旧的 `WAM-Policy_workspace/`，然后新建空目录：

```bash
mkdir WAM-Policy
cd WAM-Policy
```

Codex 应在 `WAM-Policy/` 内执行等价于以下命令的目录初始化。

```bash
mkdir -p src/wam_policy/models
mkdir -p src/wam_policy/adapters
mkdir -p src/wam_policy/opd/pipelines
mkdir -p src/wam_policy/data
mkdir -p src/wam_policy/eval
mkdir -p src/wam_policy/utils

mkdir -p configs/experiments/v1
mkdir -p configs/experiments/v2
mkdir -p configs/components/paths
mkdir -p configs/components/model
mkdir -p configs/components/teacher
mkdir -p configs/components/data
mkdir -p configs/components/env
mkdir -p configs/components/runtime
mkdir -p configs/sweeps

mkdir -p scripts

mkdir -p docs/designs
mkdir -p docs/experiments

mkdir -p tests
mkdir -p assets

mkdir -p external

mkdir -p experiments/pretrained_models/lingbot-va
mkdir -p experiments/pretrained_models/anyflow
mkdir -p experiments/pretrained_models/wan2.1
mkdir -p experiments/pretrained_models/wan2.2
mkdir -p experiments/checkpoints/v1
mkdir -p experiments/checkpoints/v2
mkdir -p experiments/runs

mkdir -p datasets/raw
mkdir -p datasets/processed
mkdir -p datasets/latents
mkdir -p datasets/rollouts
mkdir -p datasets/manifests

mkdir -p results
mkdir -p logs
mkdir -p cache
mkdir -p wandb
```

然后创建允许的非代码文件：

```text
README.md
.gitignore
pyproject.toml
WAM-Policy_repo_structure.md
```

不要创建任何 `.py` 文件，不要创建 `.yaml` 文件，不要创建 `.gitkeep` 文件。

---

## 16. 初始化验收清单

Codex 完成后，应满足：

```text
[ ] 当前项目根目录名为 WAM-Policy/。
[ ] 不存在 WAM-Policy_workspace/ 外层目录。
[ ] src/wam_policy/models/ 存在。
[ ] src/wam_policy/adapters/ 存在。
[ ] src/wam_policy/opd/pipelines/ 存在。
[ ] src/wam_policy/data/ 存在。
[ ] src/wam_policy/eval/ 存在。
[ ] src/wam_policy/utils/ 存在。
[ ] configs/experiments/v1/ 存在。
[ ] configs/experiments/v2/ 存在。
[ ] configs/components/paths/ 存在。
[ ] configs/components/model/ 存在。
[ ] configs/components/teacher/ 存在。
[ ] configs/components/data/ 存在。
[ ] configs/components/env/ 存在。
[ ] configs/components/runtime/ 存在。
[ ] configs/sweeps/ 存在。
[ ] scripts/ 存在，但没有任何脚本文件。
[ ] docs/designs/ 存在。
[ ] docs/experiments/ 存在。
[ ] tests/ 存在。
[ ] assets/ 存在。
[ ] external/ 存在，但内容被 .gitignore 忽略。
[ ] experiments/pretrained_models/ 存在，但内容被 .gitignore 忽略。
[ ] experiments/checkpoints/ 存在，但内容被 .gitignore 忽略。
[ ] experiments/runs/ 存在，但内容被 .gitignore 忽略。
[ ] datasets/ 存在，但内容被 .gitignore 忽略。
[ ] results/ 存在，但内容被 .gitignore 忽略。
[ ] logs/ 存在，但内容被 .gitignore 忽略。
[ ] cache/ 存在，但内容被 .gitignore 忽略。
[ ] wandb/ 存在，但内容被 .gitignore 忽略。
[ ] README.md 存在。
[ ] .gitignore 存在。
[ ] pyproject.toml 存在，或用户明确选择不创建。
[ ] WAM-Policy_repo_structure.md 存在。
[ ] 没有创建任何 Python 代码文件。
[ ] 没有创建任何 placeholder YAML/JSON 文件。
[ ] 没有创建 .gitkeep 文件。
[ ] 没有 clone 任何外部 repo。
[ ] 没有下载任何模型或数据。
```

---

## 17. 后续维护流程

### 17.1 本地新机器 setup

后续当 repo 已经推到 GitHub 后，新机器上应按以下流程：

```bash
git clone <WAM-Policy GitHub URL>
cd WAM-Policy
```

然后根据 README 执行后续 setup 脚本。

初始化阶段暂时不要创建这些脚本；等用户确认要实现 setup 自动化时，再创建：

```text
scripts/setup_local_dirs.sh
scripts/fetch_external_repos.sh
scripts/download_pretrained_models.sh
```

### 17.2 外部 repo setup

后续建议：

```text
scripts/fetch_external_repos.sh
```

负责 clone：

```text
external/lingbot-va/
external/AnyFlow/
external/RoboTwin/
```

### 17.3 预训练模型 setup

后续建议：

```text
scripts/download_pretrained_models.sh
```

负责下载：

```text
experiments/pretrained_models/lingbot-va/...
experiments/pretrained_models/anyflow/...
experiments/pretrained_models/wan2.1/...
experiments/pretrained_models/wan2.2/...
```

### 17.4 v1 到 v2 迭代

不要复制 repo。

正确做法是：

```text
新增 configs/experiments/v2/ 下的实验配置
新增 src/wam_policy/opd/pipelines/ 下的 v2 pipeline
必要时新增 adapters、models、losses 中的实现
```

每个版本的算法设计写入：

```text
docs/designs/
```

每个重要实验的结论写入：

```text
docs/experiments/
```

---

## 18. 非协商约束

Codex 在初始化骨架时必须遵守：

```text
不要创建 WAM-Policy_workspace/。
不要创建 WAM-Policy_v1/、WAM-Policy_v2/、WAM-Policy_v3/。
不要创建任何 .py 代码文件。
不要创建任何 YAML/JSON placeholder 配置。
不要创建 .gitkeep。
不要 clone 外部 repo。
不要下载模型。
不要下载数据。
不要把 external/、experiments/、datasets/、results/、logs/、cache/、wandb/ 的内容加入 Git。
```

本次任务的目标只是让 `WAM-Policy/` 成为一个干净、可维护、AnyFlow-style 的 research repo 根目录。
