# WAM-Policy Progress Summary

日期：2026-06-22  
范围：从 WAM-Policy 拉到本地/建立 repo 骨架，到当前准备由管理员提交 AMD 计算节点诊断训练任务。

## 1. 当前目标

WAM-Policy 当前聚焦 **WAM-Policy_v2 / Stage 1 video-head Forward Flow-map Training**。

核心目标是把 LingBot-VA 的 video stream 从原始 instantaneous velocity-field sampling，改造成 LingBot-compatible 的 two-time flow-map / average-velocity student：

```text
Z_s = (1 - s) * eps + s * Z_clean
0 <= s_src < s_tgt <= 1
target_u = (Z_tgt - Z_src) / (s_tgt - s_src)
pred_u = u_S^v(Z_src, s_src, s_tgt | context)
loss = MSE(pred_u, target_u)
```

代码约定保持 LingBot-style `s`：

```text
s = 0: noise
s = 1: clean data
```

当前不做 action-head FFM、DMD、OPD rollout、RoboTwin online eval、distributed/FSDP/DeepSpeed。

## 2. Repo 和文档进度

已经建立并整理了 WAM-Policy repo 的基本边界：

```text
src/
configs/
scripts/
tests/
docs/
assets/
external/       # ignored, for external LingBot/AnyFlow repos
experiments/    # ignored, for runs/checkpoints/pretrained models
datasets/       # ignored
logs/           # ignored
```

重要文档：

- `AGENTS.md`：repo-local agent guide，规定登录节点限制、external repo 边界、当前实现状态和安全下一步。
- `docs/agent_handoff/2026-06-21/Video-head_Forward_Flow-map_plan.md`：video-head FFM 初始实现计划。
- `docs/agent_handoff/2026-06-21/Video-head_FFM_today_work_summary.md`：从空骨架到 fake LingBot smoke path 的实现总结。
- `docs/design/WAM-Policy_v2_dual-head_AnyFlow.md`：WAM-Policy_v2 dual-head AnyFlow design 文档。

注意：当前 git worktree 里有一些文档移动/删除状态，尚未最终整理提交；不要误删用户或前序 agent 留下的 handoff 文件。

## 3. 已实现的 video_ffm 主路径

当前 non-dry-run 目标路径是：

```text
scripts/train.py
-> bootstrap external repos
-> v2_anyflow_dual_head pipeline
-> VideoFFMTrainer
-> VideoLatentManifestDataset
-> sample/build s_src/s_tgt pair
-> LingBotVideoFlowMapStudent
-> LingBotVAAdapter video hook
-> average-velocity loss
-> optimizer step
-> checkpoint save
```

已实现模块：

- config loading 和 `--set key=value` overrides。
- `--dry-run static` 静态 dispatch。
- AnyFlow-style `s_src/s_tgt` pair construction。
- video latent JSONL manifest loader 和 collate。
- video FFM average velocity loss。
- LingBot student wrapper。
- LingBot adapter 边界。
- two-time conditioner。
- trainer、optimizer、checkpoint helper。
- fake LingBot-like model smoke test path。

## 4. 登录节点静态路径修复

AMD 服务器当前只能使用登录节点，不能直接跑 GPU/checkpoint/training。

已经修复 `--dry-run static`，使它不在 pipeline 顶层 import trainer/torch：

```text
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py
```

验证命令：

```bash
python3 scripts/train.py \
  --config configs/experiments/v2/anyflow_video_ffm.yaml \
  --dry-run static
```

该命令已经在登录节点通过。

## 5. AMD/ROCm 适配进度

已按 ROCm PyTorch 约定实现：

- WAM config 支持 `runtime.device=rocm|hip`，内部映射到 PyTorch `"cuda"` device。
- 若显式使用 `rocm/hip`，会要求 `torch.version.hip` 存在。
- 默认 LingBot attention backend 改为：

```yaml
runtime:
  lingbot_attn_mode: torch
```

原因：`external/lingbot-va` 的 `modules/model.py` 顶层会尝试 import `flash_attn_interface` 或 `flash_attn`；AMD/ROCm 环境下这些依赖很可能不可用或不兼容。当前 WAM adapter 在 `attn_mode=torch` 时安装安全 stub，只为允许 LingBot 源码 import，真正误用 flash-attn 时会抛清晰错误。

相关文件：

- `src/wam_policy/utils/runtime.py`
- `src/wam_policy/adapters/lingbot_va.py`
- `configs/experiments/v2/anyflow_video_ffm.yaml`

## 6. LingBot 源码静态对齐

已经静态阅读当前 `external/lingbot-va` 的真实源码，关键事实如下：

```text
load_transformer(transformer_path, torch_dtype, torch_device, **kwargs)
WanTransformer3DModel.forward(input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False)
WanTransformer3DModel._time_embed(timesteps, H, W, dtype, action_mode=False)
```

真实 `_time_embed` 返回：

```text
temb: [B, L, C]
timestep_proj: [B, L, 6, C]
```

当前 WAM adapter 已对齐 video-only FFM 到 LingBot inference 单流接口：

```python
model(
    input_dict,
    update_cache=0,
    cache_name="pos",
    action_mode=False,
    train_mode=False,
)
```

而不是 LingBot train path 的：

```text
latent_dict/action_dict/chunk_size/window_size
```

video FFM 输入契约：

```text
noisy_latents: [B, C, F, H, W]
timesteps:     [B, F / patch_f]
grid_id:       [B, 4, token_count]
text_emb:      [B, L_text, D]
```

## 7. Probe 脚本进度

新增：

```text
scripts/tools/probe_video_ffm_runtime.py
```

子命令：

```text
probe-static-source          # torch-free, login-node safe
probe-import                 # compute-node/runtime env
probe-load-model             # checkpoint load
probe-manifest-one-sample    # one manifest sample
probe-video-forward          # no-grad LingBot video forward
```

登录节点已验证：

```bash
python3 scripts/tools/probe_video_ffm_runtime.py \
  probe-static-source \
  --config configs/experiments/v2/anyflow_video_ffm.yaml
```

结果：源码签名和 WAM adapter alignment check 均为 `OK`。

## 8. Slurm / 管理员提交脚本进度

我们尝试在登录节点提交轻量 `sbatch` 测试，Slurm 返回：

```text
QOSGrpSubmitJobsLimit
Batch job submission failed: Job violates accounting/QOS policy
```

`squeue --me` 为空，因此更像是账户/QOS 组级别限制或维护期间限制。当前无法自行提交计算节点任务。

因此已准备管理员提交脚本：

```text
scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

设计为：

```text
00_env
01_static
02_import
03_load_model
04_manifest
05_video_forward
06_train_long
```

前五个 probe 全部通过后，才进入长训练：

```text
TRAIN_MAX_STEPS default: 5000
TRAIN_SAVE_EVERY default: 500
TRAIN_LOG_EVERY default: 10
TRAIN_BATCH_SIZE default: 1
```

管理员提交命令示例：

```bash
sbatch --export=ALL,TEACHER_CKPT=/abs/path/to/transformer,MANIFEST=/abs/path/to/video_latents_debug.jsonl \
  /shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

## 9. 已通过的登录节点验证

以下命令已在登录节点通过：

```bash
bash -n scripts/slurm/video_ffm_admin_diagnostic.sbatch
python3 scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
python3 scripts/tools/probe_video_ffm_runtime.py probe-static-source --config configs/experiments/v2/anyflow_video_ffm.yaml
python3 -m py_compile scripts/train.py scripts/tools/probe_video_ffm_runtime.py src/wam_policy/adapters/lingbot_va.py src/wam_policy/utils/runtime.py
```

没有在登录节点运行：

- torch import。
- checkpoint loading。
- LingBot model forward。
- real manifest tensor loading。
- training。
- pytest full suite。

原因：登录节点缺 torch/pytest，且 repo-local guide 禁止在登录节点跑模型、训练、GPU 或重依赖任务。

## 10. 当前已知风险

1. Real LingBot checkpoint path 是否应该指向 `.../transformer` 目录，仍需管理员运行 `03_load_model` 确认。
2. Real manifest 的 `z_clean` 和 `context.text_emb` shape 仍需 `04_manifest` 确认。
3. `05_video_forward` 可能暴露真实 LingBot forward output shape 或 cache/attention backend 问题。
4. `06_train_long` 可能暴露 backward、optimizer、checkpoint save、ROCm dtype/attention kernel 问题。
5. 当前 optimizer 是保守 AdamW 路径，没有使用 fused AdamW；这是为了 AMD/ROCm 兼容和诊断稳定性。

## 11. 下一步

把以下路径交给管理员：

```text
/shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

并提供：

```text
TEACHER_CKPT=/abs/path/to/lingbot/transformer_or_checkpoint
MANIFEST=/abs/path/to/video_latents_debug.jsonl
```

管理员提交后，查看输出方法见：

```text
docs/agent_handoff/2026-06-22/admin_training_output_guide.md
```
