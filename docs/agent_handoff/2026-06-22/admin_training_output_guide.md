# Admin Training Output Guide

日期：2026-06-22  
脚本：`scripts/slurm/video_ffm_admin_diagnostic.sbatch`

本文说明管理员提交 AMD 计算节点任务后，如何查看输出、定位失败阶段、确认是否进入长训练。

## 1. 管理员提交命令

基础提交：

```bash
sbatch --export=ALL,TEACHER_CKPT=/abs/path/to/transformer,MANIFEST=/abs/path/to/video_latents_debug.jsonl \
  /shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

如果管理员只给较短窗口：

```bash
sbatch --export=ALL,TEACHER_CKPT=/abs/path/to/transformer,MANIFEST=/abs/path/to/video_latents_debug.jsonl,TRAIN_MAX_STEPS=1000 \
  /shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

如果想尽量长跑：

```bash
sbatch --export=ALL,TEACHER_CKPT=/abs/path/to/transformer,MANIFEST=/abs/path/to/video_latents_debug.jsonl,TRAIN_MAX_STEPS=10000 \
  /shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

可选环境变量：

```text
PYTHON_BIN=/path/to/python
ENV_SETUP=/path/to/env_setup.sh
TRAIN_MAX_STEPS=5000
TRAIN_SAVE_EVERY=500
TRAIN_LOG_EVERY=10
TRAIN_BATCH_SIZE=1
```

## 2. 输出目录

脚本会使用 Slurm job id 作为目录名：

```text
JOB_ID=${SLURM_JOB_ID}
```

主日志目录：

```text
/shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/logs/admin_diagnostic/<JOB_ID>/
```

训练 run 目录：

```text
/shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/experiments/runs/v2/video_ffm_admin_debug/<JOB_ID>/
```

checkpoint 目录：

```text
/shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/experiments/checkpoints/v2/video_ffm_admin_debug/<JOB_ID>/
```

Slurm 默认 stdout/stderr：

```text
slurm-wam-video-ffm-admin-<JOB_ID>.out
slurm-wam-video-ffm-admin-<JOB_ID>.err
```

这些默认 Slurm 文件只负责记录脚本顶层输出；真正有用的信息主要在 `logs/admin_diagnostic/<JOB_ID>/`。

## 3. 最先查看的文件

先看：

```bash
cat logs/admin_diagnostic/<JOB_ID>/status.txt
```

成功时应该类似：

```text
STARTED:<JOB_ID>:...
PASSED:00_env:...
PASSED:01_static:...
PASSED:02_import:...
PASSED:03_load_model:...
PASSED:04_manifest:...
PASSED:05_video_forward:...
PASSED:06_train_long:...
SUCCESS:<JOB_ID>:...
```

失败时会类似：

```text
FAILED:03_load_model:1
First failing stage: 03_load_model
See log: .../03_load_model.log
```

这时直接打开对应阶段日志。

## 4. 阶段日志说明

每个阶段单独写一个 log：

```text
00_env.log
01_static.log
02_import.log
03_load_model.log
04_manifest.log
05_video_forward.log
06_train_long.log
```

### 00_env.log

用途：确认计算节点基本环境。

重点看：

```text
hostname
slurm_job_id
slurm_nodelist
teacher_ckpt
manifest
rocm-smi output
PYTHON_BIN
```

常见问题：

- `TEACHER_CKPT is required`：提交时没有传 checkpoint 路径。
- `MANIFEST is required`：提交时没有传 manifest 路径。
- `test -e TEACHER_CKPT` 失败：checkpoint 路径不存在。
- `test -f MANIFEST` 失败：manifest 路径不存在或不是文件。
- `rocm-smi not found`：不一定致命，但说明环境模块可能没配好。

### 01_static.log

用途：确认 WAM config、源码静态对齐和 Python 语法。

重点看：

```text
Static dry-run passed
WAM Adapter Alignment
OK: ...
```

如果这里失败，多半是仓库文件不完整、路径错、Python 版本太低或脚本语法问题。

### 02_import.log

用途：确认 PyTorch ROCm 和 LingBot runtime dependencies。

重点看：

```text
torch.__version__
torch.version.hip
torch.cuda.is_available
torch.cuda.get_device_name
torch: available
diffusers: available
transformers: available
wan_va.__file__
load_transformer_signature
```

判断：

- `torch.version.hip` 应该非空。
- ROCm PyTorch 中 `torch.cuda.is_available()` 通常仍应为 `True`。
- 如果缺 `diffusers`、`transformers`、`easydict` 等，管理员需要修环境。

### 03_load_model.log

用途：加载 LingBot transformer checkpoint。

重点看：

```text
checkpoint_path
device
torch_dtype
model_class
model_forward_signature
patch_size
_time_embed_signature
time_proj_out_features
first_param_device
first_param_dtype
```

常见问题：

- checkpoint path 不对。
- checkpoint 应该指向 `transformer` 子目录，但传了父目录。
- 缺模型依赖或 config 文件。
- ROCm dtype/device 不支持。
- flash-attn import 问题；当前默认 `runtime.lingbot_attn_mode=torch` 应尽量避开。

### 04_manifest.log

用途：读取 manifest 的第一条样本，确认 tensor/context shape。

重点看：

```text
manifest_path
dataset_len
sample_id
z_clean: shape=...
text_emb: shape=...
```

预期：

```text
z_clean -> batch 后应为 [B, C, F, H, W]
text_emb -> [B, L_text, D]
```

常见问题：

- manifest JSONL 格式错误。
- `z_clean` 文件不存在。
- `context` 文件不存在。
- context 内没有 `text_emb`。
- `text_emb` batch size 与 `z_clean` batch size 不一致。

### 05_video_forward.log

用途：跑一次 no-grad video flow-map forward，确认真实 LingBot forward contract。

重点看：

```text
LingBot Video Input Contract
noisy_latents: shape=...
timesteps: shape=...
grid_id: shape=...
text_emb: shape=...
No-Grad Video Forward
pred_u: shape=...
pred_u_matches_z_src_shape: True
pred_u_all_finite: True
```

如果失败，优先检查：

- `patch_size` 与 latent `F/H/W` 是否整除。
- `grid_id` shape 是否符合 LingBot model。
- forward output 是否为 WAM adapter 支持的 sequence/latent shape。
- attention backend 是否仍然触发了不可用的 flash/flex kernel。

### 06_train_long.log

用途：真正进入 Video-stream Forward Flow-map Training。

重点看：

```text
step=...
loss/video_ffm=...
```

预期：

- loss 是 finite 数字。
- 每 `TRAIN_LOG_EVERY` 步有日志。
- 每 `TRAIN_SAVE_EVERY` 步有 checkpoint。
- 训练结束时保存 final checkpoint。

如果失败，判断层级：

- forward 已过但 train 失败：可能是 backward、requires_grad、optimizer、dtype 或 checkpoint save。
- loss NaN/Inf：可能是 dtype、target scale、latent 数据异常或 model output 异常。
- no trainable params：检查 `model.freeze_backbone` 和 `model.trainable_modules` 是否命中真实 LingBot 参数名。

## 5. 查看 checkpoint

checkpoint 目录：

```bash
ls -lh experiments/checkpoints/v2/video_ffm_admin_debug/<JOB_ID>/
```

预期至少有：

```text
latest.pt
step_000500.pt
step_001000.pt
...
```

如果 `TRAIN_MAX_STEPS=5000` 且 `TRAIN_SAVE_EVERY=500`，应有多个 `step_*.pt`。

不要在登录节点用 torch 加载 checkpoint。需要检查 checkpoint 内容时，应让管理员在计算节点环境里运行轻量 `torch.load(..., map_location="cpu")`。

## 6. 快速定位表

| 失败阶段 | 最可能问题 | 下一步 |
|---|---|---|
| `00_env` | 路径、环境变量、ROCm module | 修提交命令或环境 |
| `01_static` | repo/config/script 静态问题 | 回到登录节点修 WAM 源码 |
| `02_import` | Python/ROCm/LingBot dependencies | 管理员修环境 |
| `03_load_model` | checkpoint path/format/model deps | 确认 checkpoint 目录和 LingBot load_transformer |
| `04_manifest` | manifest 或 tensor/context shape | 修 manifest 或预处理数据 |
| `05_video_forward` | LingBot input/output contract mismatch | 修 `LingBotVAAdapter` |
| `06_train_long` | backward/optimizer/checkpoint/training stability | 根据日志修 trainer/model/loss/optimizer |

## 7. 管理员需要回传什么

最少回传：

```text
logs/admin_diagnostic/<JOB_ID>/status.txt
```

如果失败，回传：

```text
logs/admin_diagnostic/<JOB_ID>/<FAILED_STAGE>.log
```

如果进入训练，再回传：

```text
logs/admin_diagnostic/<JOB_ID>/06_train_long.log
experiments/checkpoints/v2/video_ffm_admin_debug/<JOB_ID>/ 目录列表
```

如果全部成功，回传：

```text
logs/admin_diagnostic/<JOB_ID>/
experiments/checkpoints/v2/video_ffm_admin_debug/<JOB_ID>/latest.pt
```

## 8. 当前默认训练参数

管理员脚本默认：

```text
TRAIN_MAX_STEPS=5000
TRAIN_SAVE_EVERY=500
TRAIN_LOG_EVERY=10
TRAIN_BATCH_SIZE=1
runtime.device=rocm
runtime.lingbot_attn_mode=torch
data.num_workers=0
```

这些默认值偏保守，目标是优先暴露 AMD 平台上真实 runtime/training 问题，而不是追求吞吐最大化。
