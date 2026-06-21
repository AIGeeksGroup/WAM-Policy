# Video-head Forward Flow-map Training 今日工作总结

日期：2026-06-21  
范围：WAM-Policy repo 从空骨架到当前 video-head Forward Flow-map Training 主路径补齐。

## 总体状态

今天完成的核心目标是：把 WAM-Policy 的 `video_ffm` 训练路径从“只有计划文档”推进到“Mac 端代码逻辑完整、可本地测试、可通过 fake LingBot-like model 走到 video stream forward hook”的状态。

当前已经具备：

- 配置入口和 CLI dry-run。
- AnyFlow 风格的 `s_src/s_tgt` flow-map 数学核心。
- video latent manifest dataset 和 collate。
- video FFM average velocity loss。
- LingBot-VA adapter/student wrapper 边界。
- trainer、optimizer、checkpoint、pipeline runtime dispatch。
- LingBot video stream forward hook 的 WAM 侧实现。
- two-time conditioner 对 LingBot `_time_embed` 的非侵入式注入。
- latent/context shape 对齐逻辑。
- 本地 fake model smoke test 覆盖完整训练路径。

当前仍未尝试解决：

- AMD server 环境问题。
- ROCm/PyTorch 适配。
- 真实 LingBot checkpoint 加载。
- 真实 `diffusers/transformers/easydict` 等 LingBot runtime dependency。
- action-head、DMD、RoboTwin rollout、distributed/FSDP/DeepSpeed。

## 第一批：Static Dry-run 主干

新增 `configs/experiments/v2/anyflow_video_ffm.yaml`，作为 video-head FFM 的主实验配置。

配置结构包含：

- `experiment`
- `paths`
- `teacher`
- `model`
- `data`
- `flowmap`
- `runtime`
- `train`

新增 `scripts/train.py`，提供统一训练入口：

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
```

CLI 支持：

- `--config`
- 可重复 `--set key.path=value`
- `--dry-run static`

新增轻量配置工具：

- `wam_policy.utils.config.load_config`
- `wam_policy.utils.config.apply_overrides`
- `wam_policy.utils.paths.resolve_repo_root`
- `wam_policy.utils.external.bootstrap_external_repos`

新增 pipeline dispatch：

- `wam_policy.opd.pipelines.v2_anyflow_dual_head.run`
- `video_ffm` dry-run 返回静态检查结果。
- `action_ffm`、`video_dmd`、`action_dmd` 继续抛 `NotImplementedError`。

`pyproject.toml` 增加了：

```toml
PyYAML>=6.0
```

没有引入 Hydra/OmegaConf，保持第一批配置系统轻量。

## 第二批：AnyFlow `s_src/s_tgt` 数学核心

新增 `src/wam_policy/adapters/anyflow.py`。

实现了：

- `VideoFlowMapPair`
- `sample_s_pairs`
- `expand_time_like`
- `build_video_flowmap_pair`
- `average_velocity_target_s`

核心约束：

- 只使用 `s_src/s_tgt/delta_s`。
- `s_src < s_tgt`。
- `delta_s >= min_delta_s`。
- `Z_s = (1 - s) * eps + s * Z_clean`。
- `target_u = (Z_tgt - Z_src) / delta_s`。

本批没有 import AnyFlow 外部 repo 的 `far.*`，所以本地测试不受外部依赖影响。

## 第三批：Video Latent Dataset + Loss

新增 `src/wam_policy/data/` 包。

`manifests.py` 负责：

- JSONL manifest 读取。
- 逐行 schema 校验。
- line number 错误提示。
- repo-root 相对路径解析。
- 绝对路径保持原样。

manifest schema：

- 必填：`sample_id: str`、`z_clean: str`
- 可选：`context`、`prompt`、`task`、`chunk_idx`、`source`、`shape`、`dtype`

`video_latent_dataset.py` 负责：

- `torch.load(..., map_location="cpu", weights_only=False)` 加载预处理 latent。
- 返回固定格式：

```python
{
    "z_clean": Tensor,
    "context": Any | None,
    "prompt": str,
    "meta": dict,
}
```

新增 `video_latent_collate_fn`：

- `z_clean` 使用 `torch.stack`。
- `context` 保持 list。
- `prompt` 保持 list[str]。
- `meta` 保持 list[dict]。

新增 `src/wam_policy/opd/losses.py`：

- `video_ffm_average_velocity_loss(pred_u, target_u, mask=None)`

loss 行为：

- shape/device 校验。
- NaN/Inf 非有限值校验。
- mask 支持同 shape、可广播 shape、或 `[B]`。
- 使用 `diff = diff * mask; diff.square().mean()`。
- metrics 返回 Python float。

## 第四批：LingBot Adapter + Student Wrapper 骨架

新增 `src/wam_policy/adapters/lingbot_va.py`。

它成为 WAM-Policy 与 LingBot-VA internals 的唯一边界。trainer/student 不直接 import `wan_va`。

实现内容：

- `LingBotVAAdapter(cfg)`
- 延迟 import `wan_va.modules.utils.load_transformer`
- `load_pretrained_video_action_model`
- `build_video_context`
- `attach_video_time_pair_conditioner`
- `forward_video_flowmap`

本地缺 LingBot runtime dependency 时，会抛清晰 `RuntimeError`，而不是在 import 阶段直接炸掉整个 WAM-Policy。

新增 `src/wam_policy/models/flow_map_heads.py`：

- `FlowTimePairEmbedding`
- 输入 `s_src/s_tgt`
- 内部构造 `delta_s`
- zero-init 输出层，初始 conditioning 近似零扰动。

新增 `src/wam_policy/models/lingbot_student.py`：

- `LingBotVideoFlowMapStudent`
- `build()` 加载 base model 并 attach conditioner。
- `freeze_by_config()` 支持冻结 backbone，仅保留 conditioner 和命中的 trainable modules。
- `forward_video_flowmap()` 委托 adapter，并校验 `pred_u.shape == z_src.shape`。

## 第五批：Trainer + Checkpoint + Runtime Pipeline

新增 runtime helper：

- `resolve_device`
- `resolve_torch_dtype`
- `set_seed`
- `log_step_metrics`
- `build_adamw_optimizer`

`build_adamw_optimizer` 包含 fallback optimizer，用于绕过本地 Torch 环境里 `torch.optim.AdamW` 触发缺失 `sympy` 的问题。

新增 checkpoint helper：

- `save_training_checkpoint`

checkpoint schema：

```python
{
    "step": step,
    "model": model_state_dict,
    "optimizer": optimizer_state_dict,
    "cfg": cfg,
    "metrics": metrics,
}
```

保存文件：

- `latest.pt`
- `step_000500.pt` 这种 step 文件。

新增 trainer：

- `VideoFFMTrainer`
- `train_video_forward_flowmap`

训练 step 顺序：

1. 从 DataLoader 取 batch。
2. 读取 `z_clean`。
3. `adapter.build_video_context(batch)`。
4. sample `s_src/s_tgt`。
5. `build_video_flowmap_pair`。
6. `student.forward_video_flowmap`。
7. `video_ffm_average_velocity_loss`。
8. backward。
9. optional grad clip。
10. optimizer step。
11. log/save checkpoint。

pipeline dispatch 更新为：

- `video_ffm` 非 dry-run 时调用 `train_video_forward_flowmap(cfg)`。
- 其他 stage 仍未实现。

## 第六批：Mac 端补齐真实 Video-head FFM 代码路径

这是今天最后一批关键补齐，目标是：不考虑 AMD 环境和真实 checkpoint，在 Mac 端把代码逻辑补齐到“可以调用 LingBot-VA video stream forward”的程度。

### LingBot video forward hook

`LingBotVAAdapter.attach_video_time_pair_conditioner()` 不再只是挂属性。

现在会：

1. 挂载 `model.wam_time_pair_conditioner`。
2. 根据 LingBot model 推断 conditioner 输出维度。
3. monkey-patch LingBot model 的 `_time_embed`。
4. 安装 `model.wam_forward_video_flowmap(...)`。

新增 public interface：

- `install_video_flowmap_hook`
- `build_lingbot_video_input`
- `normalize_video_context`
- `lingbot_output_to_latent`
- `video_sequence_to_latent`

hook 内部做的事情：

1. 将 WAM 语义输入：

```python
z_src [B, C, F, H, W]
s_src [B]
s_tgt [B]
context
```

转换成 LingBot video stream input：

```python
{
    "noisy_latents": z_src,
    "timesteps": (1 - s_src) * num_train_timesteps,
    "grid_id": grid_id,
    "text_emb": text_emb,
}
```

2. 调用：

```python
model(input_dict, action_mode=False, train_mode=False)
```

3. 将 LingBot 输出转回 WAM latent shape。

### LingBot video input builder

`build_lingbot_video_input()` 负责：

- 校验 `z_src` 必须是 `[B,C,F,H,W]`。
- 根据 `model.patch_size` 检查 `F/H/W` 可整除。
- 构造 `timesteps`，shape 为 `[B, F / patch_f]`。
- 构造 `grid_id`，shape 为 `[B, 4, L]`。
- 规范化 `text_emb`，shape 为 `[B, L_text, D_text]`。

`flowmap.lingbot_num_train_timesteps` 默认是 `1000`。

### Two-time conditioner 真实注入

`FlowTimePairEmbedding` 已扩展：

- 支持 `[B]` 时间输入。
- 支持 `[B,L]` token-level 时间输入。
- 新增 `forward_conditioning()`，返回：

```python
temb_delta
timestep_proj_delta
```

adapter 会 patch LingBot model 的 `_time_embed`：

1. 先调用原始 `_time_embed`。
2. 如果 `action_mode=True`，直接返回原结果。
3. 如果是 video stream，并且存在当前 `s_src/s_tgt`，则扩展到 token level。
4. 调用 conditioner。
5. 把 zero-init delta 加到 `temb` 和 `timestep_proj`。

因为 conditioner 最后一层 zero-init，所以初始时数值不变，但训练中可以学习 two-time conditioning。

### Latent/context shape 对齐

trainer 新增：

- `normalize_video_latent_batch`

支持：

- `[B,C,F,H,W]`：直接通过。
- `[B,1,C,F,H,W]`：squeeze 第 1 维。
- 其他 rank：直接报错。

adapter 新增：

- `normalize_video_context`

支持 context：

- `dict`
- `list[dict]`
- tensor

但最终必须得到：

```python
text_emb [B, L_text, D_text]
```

如果没有 `text_emb`，会明确报错，不会静默造 dummy context。

### LingBot output 转 latent

支持两类 LingBot output：

- `[B,C,F,H,W]`：直接返回。
- `[B,N,C]` token sequence：用 WAM 内部 `video_sequence_to_latent()` 还原成 `[B,C,F,H,W]`。

`video_sequence_to_latent()` 的逻辑等价于 LingBot `data_seq_to_patch()`，但放在 WAM adapter 内，避免 trainer 依赖 external repo。

## 测试覆盖

当前测试覆盖了：

- config YAML 加载。
- `--set` 嵌套 override。
- repo root 解析。
- external repo bootstrap。
- pipeline dry-run dispatch。
- AnyFlow `s_src/s_tgt` 采样和公式。
- manifest schema 校验。
- video latent dataset/collate。
- video FFM loss。
- checkpoint 保存。
- trainer CPU smoke。
- student wrapper build/freeze/forward。
- LingBot adapter delayed import。
- LingBot video forward hook 安装。
- LingBot input dict 构造。
- `(1 - s_src) * 1000` timesteps。
- grid shape。
- context `text_emb` 规范化。
- sequence output 转回 latent。
- two-time conditioner `[B]` 和 `[B,L]` 输入。
- zero-init 注入保持初始数值不变。
- action-mode `_time_embed` 不被注入。
- trainer 通过 adapter hook path 的 fake smoke test。

## 验收命令

今天最后一轮执行并通过：

```bash
python -m compileall src scripts tests
python -m ruff check src scripts tests
python -m pytest tests
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
```

pytest 结果：

```text
74 passed
```

dry-run 输出确认：

```text
Static dry-run passed
pipeline: v2_anyflow_dual_head
stage: video_ffm
```

## 当前代码边界

已经补齐的 Mac 端代码路径：

```text
manifest dataset
  -> dataloader collate
  -> normalize_video_latent_batch
  -> sample_s_pairs
  -> build_video_flowmap_pair
  -> LingBotVideoFlowMapStudent
  -> LingBotVAAdapter.wam_forward_video_flowmap hook
  -> LingBot video stream model(input_dict, action_mode=False)
  -> output to latent
  -> video_ffm_average_velocity_loss
  -> optimizer
  -> checkpoint
```

仍留到 AMD server debug 的内容：

- 真实 LingBot checkpoint 是否能被 `load_transformer` 正确加载。
- 真实 LingBot model 的 exact `patch_size`、`_time_embed` signature、`condition_embedder.time_proj.out_features` 是否与 fake model 完全一致。
- 真实训练 manifest 的 `z_clean` shape 和 `text_emb` shape 是否与 adapter 预期一致。
- ROCm dtype/device 行为。
- server 路径和外部依赖安装。

## 修改文件概览

主要新增/修改的代码文件：

- `pyproject.toml`
- `configs/experiments/v2/anyflow_video_ffm.yaml`
- `scripts/train.py`
- `src/wam_policy/utils/config.py`
- `src/wam_policy/utils/paths.py`
- `src/wam_policy/utils/external.py`
- `src/wam_policy/utils/runtime.py`
- `src/wam_policy/utils/seed.py`
- `src/wam_policy/utils/logging.py`
- `src/wam_policy/utils/optim.py`
- `src/wam_policy/adapters/anyflow.py`
- `src/wam_policy/adapters/lingbot_va.py`
- `src/wam_policy/adapters/checkpoint.py`
- `src/wam_policy/data/manifests.py`
- `src/wam_policy/data/video_latent_dataset.py`
- `src/wam_policy/models/flow_map_heads.py`
- `src/wam_policy/models/lingbot_student.py`
- `src/wam_policy/opd/losses.py`
- `src/wam_policy/opd/trainer.py`
- `src/wam_policy/opd/stages.py`
- `src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py`

主要测试文件：

- `tests/test_config_static.py`
- `tests/test_external_bootstrap.py`
- `tests/test_pipeline_static_dispatch.py`
- `tests/test_anyflow_s_pairs.py`
- `tests/test_manifest_schema.py`
- `tests/test_video_latent_dataset.py`
- `tests/test_video_ffm_loss.py`
- `tests/test_lingbot_adapter.py`
- `tests/test_lingbot_student.py`
- `tests/test_checkpoint.py`
- `tests/test_video_ffm_trainer.py`

## 下一步建议

下一阶段上 AMD server 时，优先按这个顺序 debug：

1. 确认 `bootstrap_external_repos` 后可以 import `wan_va`。
2. 单独调用 `LingBotVAAdapter.load_pretrained_video_action_model()`。
3. inspect 真实 LingBot model 的：
   - `patch_size`
   - `_time_embed` signature
   - `condition_embedder.time_proj.out_features`
   - forward 返回 shape
4. 用 1 个 fake manifest 样本跑非 dry-run。
5. 对齐真实 `z_clean` 和 `text_emb` shape。
6. 再跑小步数训练，例如 `train.max_steps=2`。
