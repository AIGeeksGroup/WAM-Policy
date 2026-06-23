# WAM-Policy Video FFM Admin Setup Summary

日期：2026-06-23

本文整理本轮 chat session 中对 WAM-Policy Video-stream Forward Flow-map Training 做过的改动。重点是让管理员在服务器上准备 LingBot-VA teacher、video latent manifest、Weights & Biases，并通过一次 `sbatch` 自动串起诊断和长训练。

## 1. GitHub remote 和本地仓库状态

Mac 本地 WAM-Policy repo 的 `origin` 已从个人 private repo 切换到团队 repo：

```text
git@github.com:AIGeeksGroup/WAM-Policy.git
```

当前本地分支是 `main`。本轮后续改动尚未提交，主要集中在 README、Slurm 管理脚本、W&B logging、latent converter、测试和管理员 handoff 文档。

## 2. README 管理员 setup 流程

`README.md` 已新增并整理为 6 个步骤，面向管理员从空 clone 到提交训练任务：

1. Clone WAM-Policy：

```bash
git clone git@github.com:AIGeeksGroup/WAM-Policy.git
cd WAM-Policy
```

2. 按 `WAM-Policy_repo_structure.md` 创建被 `.gitignore` 忽略的 runtime 目录，包括 `external/`、`experiments/`、`datasets/`、`logs/`、`cache/`、`wandb/` 等。

3. Clone LingBot-VA 源码到固定位置：

```bash
git clone https://github.com/Robbyant/lingbot-va.git external/lingbot-va
test -d external/lingbot-va/wan_va
```

4. 从 Hugging Face 下载 LingBot-VA RoboTwin teacher checkpoint：

```bash
python3 -m pip install -U huggingface_hub
export HF_HOME="$PWD/cache/huggingface"
```

下载目标：

```text
experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/
```

训练时传入的 teacher checkpoint path 是：

```bash
export TEACHER_CKPT="$PWD/experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/transformer"
```

5. 下载 LingBot 官方 RoboTwin dataset，并把其中已经预计算好的 WAN 2.2 video latents 转成 WAM 当前 loader 可读的 `z_clean.pt`、`context.pt` 和 JSONL manifest。

6. 设置 `WANDB_API_KEY` 后提交一次 Slurm job。脚本会先用 debug manifest 跑诊断，通过后自动用 full manifest 进入长训练。

## 3. Video latent manifest 转换

新增脚本：

```text
scripts/tools/convert_lingbot_robotwin_latents.py
```

用途：把 LingBot-VA 官方 RoboTwin dataset 中的 `.pth` latent 转成 WAM-Policy 当前 `VideoLatentManifestDataset` 能直接读取的 split files。

输入默认匹配：

```text
<source-root>/latents/**/*.pth
```

每个源 `.pth` 必须包含：

```text
latent
latent_num_frames
latent_height
latent_width
text_emb
```

核心转换：

```python
latent: [F * H * W, C]
z_clean = latent.reshape(F, H, W, C).permute(3, 0, 1, 2).contiguous()
# z_clean: [C, F, H, W]
```

debug manifest：

```bash
python3 scripts/tools/convert_lingbot_robotwin_latents.py \
  --repo-root "$PWD" \
  --source-root datasets/raw/robotwin-clean-and-aug-lerobot \
  --output-root datasets/latents/robotwin_video_ffm_debug \
  --manifest datasets/manifests/video_latents_debug.jsonl \
  --max-samples 16 \
  --overwrite

export DEBUG_MANIFEST="$PWD/datasets/manifests/video_latents_debug.jsonl"
```

full manifest：

```bash
python3 scripts/tools/convert_lingbot_robotwin_latents.py \
  --repo-root "$PWD" \
  --source-root datasets/raw/robotwin-clean-and-aug-lerobot \
  --output-root datasets/latents/robotwin_video_ffm_full \
  --manifest datasets/manifests/video_latents_full.jsonl \
  --max-samples 0 \
  --overwrite

export FULL_MANIFEST="$PWD/datasets/manifests/video_latents_full.jsonl"
```

脚本不 import LingBot-VA，不加载 teacher checkpoint，不需要 GPU；只依赖 `torch` 和标准库。Mac 本地不需要下载 443GB dataset，也不需要本地跑 LingBot-VA。

## 4. Debug 诊断和 full training 的 Slurm 串联

更新脚本：

```text
scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

现在脚本要求三个核心变量：

```text
TEACHER_CKPT
DEBUG_MANIFEST
FULL_MANIFEST
```

提交方式：

```bash
sbatch --export=ALL,TEACHER_CKPT="$TEACHER_CKPT",DEBUG_MANIFEST="$DEBUG_MANIFEST",FULL_MANIFEST="$FULL_MANIFEST" \
  scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

执行阶段：

```text
00_env -> 01_static -> 02_import -> 03_load_model -> 04_manifest -> 05_video_forward -> 06_train_long
```

`00_env` 到 `05_video_forward` 使用 `DEBUG_MANIFEST`。任一诊断阶段失败就停止并记录 first failing stage。全部通过后，`06_train_long` 自动使用 `FULL_MANIFEST` 进入长训练。

默认长训练参数：

```text
TRAIN_MAX_STEPS=5000
TRAIN_SAVE_EVERY=500
TRAIN_LOG_EVERY=10
TRAIN_BATCH_SIZE=1
```

如需先短跑，可以提交时覆盖：

```bash
sbatch --export=ALL,TEACHER_CKPT="$TEACHER_CKPT",DEBUG_MANIFEST="$DEBUG_MANIFEST",FULL_MANIFEST="$FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

## 5. W&B metrics 和 log artifact

配置文件新增默认 W&B 配置：

```text
configs/experiments/v2/anyflow_video_ffm.yaml
```

默认行为：

```yaml
logging:
  wandb:
    enabled: auto
    project: WAM-Policy
```

`enabled: auto` 的含义是：环境里存在 `WANDB_API_KEY` 时自动启用 W&B；没有 key 时保持关闭。因此管理员只需要在提交前设置：

```bash
export WANDB_API_KEY="<private-api-key>"
```

可选：

```bash
export WANDB_ENTITY="<team-or-user>"
export WANDB_PROJECT="WAM-Policy"
```

新增运行 logger：

```text
src/wam_policy/utils/logging.py
```

训练器现在通过 `TrainingLogger` 同时打印 stdout metrics，并在 W&B 启用时调用 `wandb.log(...)`。W&B config 会对 key 名包含 `api_key`、`token`、`password`、`secret` 的字段做 `<redacted>` 处理。

新增 log artifact 上传脚本：

```text
scripts/tools/upload_admin_logs_to_wandb.py
```

Slurm job 退出时，如果 W&B 已启用且 `WANDB_UPLOAD_LOGS=true`，会向同一个 run 上传 `admin-logs` artifact，包含：

```text
logs/admin_diagnostic/<JOB_ID>/status.txt
logs/admin_diagnostic/<JOB_ID>/*.log
slurm-wam-video-ffm-admin-<JOB_ID>.out
slurm-wam-video-ffm-admin-<JOB_ID>.err
logs/admin_diagnostic/<JOB_ID>/checkpoint_files_<JOB_ID>.txt
```

artifact 只上传 checkpoint 文件列表，不上传 checkpoint tensor 本体。若只想上报 metrics、不上传日志 artifact，设置：

```bash
export WANDB_UPLOAD_LOGS=false
```

## 6. 相关文件改动

主要修改：

```text
README.md
configs/experiments/v2/anyflow_video_ffm.yaml
docs/agent_handoff/2026-06-22/admin_training_output_guide.md
pyproject.toml
scripts/slurm/video_ffm_admin_diagnostic.sbatch
src/wam_policy/opd/trainer.py
src/wam_policy/utils/logging.py
tests/test_config_static.py
```

新增文件：

```text
scripts/tools/convert_lingbot_robotwin_latents.py
scripts/tools/upload_admin_logs_to_wandb.py
tests/test_lingbot_robotwin_latent_conversion.py
tests/test_logging.py
tests/test_upload_admin_logs_to_wandb.py
docs/agent_handoff/2026-06-23/WAM-Policy_video_ffm_admin_setup_summary.md
```

`pyproject.toml` 新增可选依赖：

```text
wandb = ["wandb>=0.16"]
```

因此需要 W&B 的环境可安装：

```bash
python3 -m pip install -e '.[wandb]'
```

## 7. 本轮已做验证

本轮已跑过以下轻量验证：

```bash
bash -n scripts/slurm/video_ffm_admin_diagnostic.sbatch

python3 -m py_compile \
  scripts/tools/upload_admin_logs_to_wandb.py \
  scripts/tools/convert_lingbot_robotwin_latents.py \
  scripts/train.py \
  scripts/tools/probe_video_ffm_runtime.py \
  src/wam_policy/utils/logging.py

python3 scripts/train.py \
  --config configs/experiments/v2/anyflow_video_ffm.yaml \
  --dry-run static

python3 -m pytest \
  tests/test_logging.py \
  tests/test_upload_admin_logs_to_wandb.py \
  tests/test_config_static.py \
  tests/test_video_ffm_trainer.py \
  tests/test_lingbot_robotwin_latent_conversion.py \
  tests/test_video_latent_dataset.py \
  tests/test_manifest_schema.py
```

结果：上述 pytest 集合当时通过，共 36 个测试 passed。

## 8. 仍需管理员在服务器验证的点

这些事情不能在 Mac 或登录节点上完成，需要管理员在允许下载和 GPU 运行的服务器/计算节点上验证：

- `external/lingbot-va` 的真实 import 是否通过。
- `wan_va.modules.utils.load_transformer` 是否能加载 `TEACHER_CKPT`。
- 真实 LingBot model 的 `patch_size`、`_time_embed` signature、`condition_embedder.time_proj.out_features` 和 video forward return shape 是否与当前 adapter 假设一致。
- 官方 dataset 实际下载后的 latent 路径是否匹配 `latents/**/*.pth`。
- debug manifest 的一条样本是否能通过 `04_manifest` 和 `05_video_forward`。
- 诊断通过后，`06_train_long` 是否能在 full manifest 上稳定训练并写 checkpoint。

## 9. 不应提交或外传的内容

本轮没有把以下内容加入 Git：

```text
WANDB_API_KEY
GitHub token
Hugging Face token
LingBot checkpoint
RoboTwin dataset
converted .pt/.pth latent files
generated JSONL manifest
training logs
wandb local files
checkpoint tensors
```

这些仍应只通过服务器本地路径、环境变量或 W&B run/artifact 机制传递。
