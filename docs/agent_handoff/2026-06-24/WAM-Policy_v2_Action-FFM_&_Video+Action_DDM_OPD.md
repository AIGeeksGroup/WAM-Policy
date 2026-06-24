# WAM-Policy_v2 Action-FFM 与 Video+Action DMD-OPD 本地静态实现计划

> 面向 Codex 的 handoff 文档。  
> 当前开发方式：**Mac 本地只做代码搭建、接口设计、静态 debug、轻量 fake-test**；不要求在 Mac 上下载 checkpoint、manifest、RoboTwin dataset，也不要求在 Mac 上加载 LingBot-VA teacher 或真正跑训练。  
> 服务器目标：延续当前 Video Forward Flow-map Training 的工程风格，保证后续阶段可以通过 Slurm/ROCm 诊断脚本放到 AMD 服务器上逐步验证。

---

## 0. 当前背景与目标边界

当前 WAM-Policy_v2 已经完成了一版 **Video-stream Forward Flow-map Training**，并且服务器已经可以跑通。接下来的目标不是在 Mac 上彻底跑通所有训练，而是在现有 video FFM 代码基础上，继续补齐剩余三个阶段的 **接口、config、adapter boundary、trainer skeleton、loss/data schema、probe/slurm 入口和轻量测试**：

1. `action_ffm`：Action-stream Forward Flow-map Training。
2. `video_dmd`：Video-stream DMD-style On-policy Distillation。
3. `action_dmd`：Action-stream DMD-style On-policy Distillation。

本计划中的“实现”含义是：

- Mac 本地：配置能加载、pipeline 能 dispatch、接口能静态通过、fake tensor 测试能覆盖 shape/loss/dispatch，不依赖真实 checkpoint 和真实 manifest。
- AMD/ROCm 服务器：提供和当前 `video_ffm_admin_diagnostic.sbatch` 类似的诊断脚本，让管理员在 compute node 上传入真实 checkpoint、manifest、rollout manifest 后逐级验证。

本计划中的“暂不要求”包括：

- 不在 Mac 本地下载 LingBot-VA checkpoint。
- 不在 Mac 本地下载 RoboTwin dataset / manifest / latent。
- 不在 Mac 本地 import 或加载真实 `wan_va` 大模型。
- 不在 Mac 本地跑 RoboTwin simulation。
- 不在 Mac 本地跑 GPU / ROCm / torch distributed / deepspeed / FSDP。
- 不直接修改 `external/lingbot-va`、`external/AnyFlow`、`external/RoboTwin` 源码。

---

## 1. 必须保持的现有设计约束

### 1.1 时间变量约束

继续沿用当前 WAM-Policy 代码约定：

```text
s = 0: noise
s = 1: clean data
0 <= s_src < s_tgt <= 1
```

代码 API 中继续使用：

```text
s_src
s_tgt
delta_s = s_tgt - s_src
```

不要在 trainer、dataset、loss、pipeline、adapter public API 中引入 `tau/rho`。如果文档中需要解释 AnyFlow/LingBot 设计，可以说明它们是 design notation，但代码边界保持 `s_src/s_tgt`。

### 1.2 Flow-map target 约束

Video FFM 已经使用 average velocity parameterization：

```text
Z_s = (1 - s) * eps + s * Z_clean
target_u = (Z_tgt - Z_src) / (s_tgt - s_src)
```

Action FFM 必须完全复用这个方向：

```text
A_s = (1 - s) * eps + s * A_clean
target_u = (A_tgt - A_src) / (s_tgt - s_src)
```

不要把 action 写成反向 `tau/rho` 公式，也不要在 action 阶段改成 endpoint prediction。第一版 action 也使用 average velocity prediction。

### 1.3 Two-time conditioner 约束

当前 video FFM 使用的是 **MLP 式外挂 two-time conditioner**：

```text
FlowTimePairEmbedding(s_src, s_tgt, delta_s)
→ 生成 temb_delta / timestep_proj_delta
→ patch LingBot condition/time embedding
```

后续 action FFM、video DMD、action DMD 暂时也保持这种设计，不要重构成复杂 LoRA、FiLM、adapter block、额外 transformer head 或多套深层结构。

### 1.4 Adapter boundary 约束

外部 repo 逻辑必须继续收口在 adapter 内：

```text
LingBot-VA 相关: src/wam_policy/adapters/lingbot_va.py
AnyFlow 相关:   src/wam_policy/adapters/anyflow.py
RoboTwin 相关:  src/wam_policy/adapters/robotwin.py
```

`trainer.py`、`stages.py`、`losses.py`、`datasets.py`、`pipelines/` 不应该直接 import `wan_va`、AnyFlow runtime 或 RoboTwin runtime。

Mac 本地测试不应该因为没有 `external/lingbot-va` 或 checkpoint 就失败。只有服务器 probe / non-dry-run runtime 阶段才允许要求真实外部依赖。

---

## 2. 目标文件总览

建议 Codex 分批修改/新增以下文件。

### 2.1 Configs

新增或补齐：

```text
configs/experiments/v2/anyflow_action_ffm.yaml
configs/experiments/v2/anyflow_video_dmd_opd.yaml
configs/experiments/v2/anyflow_action_dmd_opd.yaml
configs/experiments/v2/anyflow_dual_head_dmd_debug.yaml   # 可选
```

保留已有：

```text
configs/experiments/v2/anyflow_video_ffm.yaml
```

### 2.2 Pipeline / stages / trainer

修改：

```text
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py
src/wam_policy/opd/stages.py
src/wam_policy/opd/trainer.py
src/wam_policy/opd/losses.py
src/wam_policy/opd/buffers.py
```

如果 `trainer.py` 变得太长，可以新增：

```text
src/wam_policy/opd/trainers_video.py
src/wam_policy/opd/trainers_action.py
src/wam_policy/opd/trainers_dmd.py
```

但第一版可以先在 `trainer.py` 内追加类，避免过早重构。

### 2.3 Models / adapters

修改或新增：

```text
src/wam_policy/models/lingbot_student.py
src/wam_policy/models/flow_map_heads.py
src/wam_policy/adapters/anyflow.py
src/wam_policy/adapters/lingbot_va.py
src/wam_policy/adapters/robotwin.py
src/wam_policy/adapters/checkpoint.py
```

### 2.4 Data

新增：

```text
src/wam_policy/data/action_dataset.py
src/wam_policy/data/rollout_manifest_dataset.py
```

可选地更新：

```text
src/wam_policy/data/manifests.py
```

### 2.5 Probes / Slurm

新增：

```text
scripts/tools/probe_action_ffm_runtime.py
scripts/tools/probe_dmd_opd_runtime.py
scripts/slurm/action_ffm_admin_diagnostic.sbatch
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

也可以把 Slurm 合并成一个通用脚本：

```text
scripts/slurm/v2_stage_admin_diagnostic.sbatch
```

但为了延续当前 video FFM 的清晰风格，建议第一版使用 stage-specific 脚本。

### 2.6 Tests

新增或扩展：

```text
tests/test_pipeline_static_dispatch.py
tests/test_action_flowmap_pair.py
tests/test_action_dataset.py
tests/test_action_ffm_loss.py
tests/test_action_student_static.py
tests/test_dmd_losses.py
tests/test_rollout_manifest_dataset.py
tests/test_probe_scripts_static.py
```

测试必须使用临时目录、fake tensor、fake adapter、fake model，不依赖真实 checkpoint、manifest、WAN/LingBot/RoboTwin。

---

## 3. 阶段 A：先补齐 config 与 pipeline static dispatch

这是第一批 commit。目标是让三个新 stage 在 Mac 本地都能被 `scripts/train.py --dry-run static` 识别。

### 3.1 修改 pipeline

更新：

```text
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py
```

当前状态类似：

```python
UNIMPLEMENTED_STAGES = {"action_ffm", "video_dmd", "action_dmd"}
```

需要改成显式 dispatch：

```python
def run(cfg, dry_run=False):
    stage = cfg["experiment"]["stage"]

    if stage == "video_ffm":
        ...
    if stage == "action_ffm":
        if dry_run:
            return _static_action_ffm_check(cfg)
        return train_action_forward_flowmap(cfg)
    if stage == "video_dmd":
        if dry_run:
            return _static_video_dmd_check(cfg)
        return train_video_dmd_opd(cfg)
    if stage == "action_dmd":
        if dry_run:
            return _static_action_dmd_check(cfg)
        return train_action_dmd_opd(cfg)
```

注意：

- dry-run static 不应该加载 checkpoint。
- dry-run static 不应该读取真实 manifest 内容。
- dry-run static 只检查 config section、paths 字段、stage 名字、必要 key 是否存在。
- 如果 non-dry-run 的 trainer 还没写完，允许临时 raise 清晰 `NotImplementedError`，但最终本计划要求 trainer 类存在并能用 fake adapter 测试。

### 3.2 新增 config

`anyflow_action_ffm.yaml` 建议结构：

```yaml
experiment:
  name: anyflow_action_ffm
  version: v2
  stage: action_ffm

paths:
  repo_root: .
  lingbot_repo: external/lingbot-va
  anyflow_repo: external/AnyFlow
  output_dir: experiments/runs/v2/action_ffm
  checkpoint_dir: experiments/checkpoints/v2/action_ffm

teacher:
  name: lingbot_va
  checkpoint_path: experiments/pretrained_models/lingbot-va/PLACEHOLDER
  frozen: true

model:
  backend: lingbot_va
  student_type: action_flowmap
  output_type: average_velocity
  time_condition: s_src_s_tgt_delta
  freeze_backbone: true
  trainable_modules:
    - action_time_pair_adapter
    - action_head_adapter

# Mac 本地可以只使用 placeholder；服务器再通过 --set 覆盖 manifest。
data:
  mode: action_chunk_manifest
  manifest: datasets/manifests/action_chunks_debug.jsonl
  action_root: datasets/actions
  context_mode: precomputed_or_rebuild
  video_condition_mode: teacher_video_or_manifest
  batch_size: 1
  num_workers: 0

flowmap:
  time_convention: lingbot_s_noise0_data1
  s_min: 0.0
  s_max: 1.0
  min_delta_s: 0.02
  sample_strategy: uniform
  target_type: average_velocity
  eps_shared_between_src_tgt: true

runtime:
  device: auto
  precision: bf16
  lingbot_attn_mode: torch
  allow_tf32: true
  compile: false

train:
  seed: 42
  lr: 1.0e-5
  weight_decay: 0.0
  max_steps: 1000
  grad_clip_norm: 1.0
  log_every: 10
  save_every: 500
```

`anyflow_video_dmd_opd.yaml` 与 `anyflow_action_dmd_opd.yaml` 应该加入：

```yaml
opd:
  rollout_manifest: datasets/manifests/student_rollouts_debug.jsonl
  teacher_relabel_mode: online_or_cached
  dmd:
    enabled: true
    weight: 0.1
    score_source: teacher_velocity_adapter
    student_score_source: flowmap_small_interval
  anchor_ffm:
    enabled: true
    weight: 1.0
```

Mac 本地只检查 schema，不读取真实 rollout manifest。

### 3.3 Mac 验收命令

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static
```

这些命令必须在没有真实 checkpoint、没有真实 manifest 的 Mac 本地通过。

---

## 4. 阶段 B：Action Flow-map pair、dataset、loss

这是第二批 commit。目标是让 action FFM 的非 LingBot 部分在 Mac 本地完全可测。

### 4.1 扩展 AnyFlow adapter

修改：

```text
src/wam_policy/adapters/anyflow.py
```

新增：

```python
@dataclass(frozen=True)
class ActionFlowMapPair:
    a_clean: torch.Tensor
    eps: torch.Tensor
    s_src: torch.Tensor
    s_tgt: torch.Tensor
    a_src: torch.Tensor
    a_tgt: torch.Tensor
    target_u: torch.Tensor


def build_action_flowmap_pair(
    a_clean: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> ActionFlowMapPair:
    ...
```

实现原则：

- 不假设 action tensor 具体 rank，只要求第一维是 batch。
- 支持常见形状：`[B, T, D]`、`[B, D]`、`[B, C, T]`。
- `target_u = (a_tgt - a_src) / (s_tgt - s_src)`。
- 复用 `expand_time_like()` 和已有 time validation。
- 不 import AnyFlow 外部 repo；当前实现只需要 torch。

### 4.2 新增 action dataset

新增：

```text
src/wam_policy/data/action_dataset.py
```

建议类名：

```python
class ActionChunkManifestDataset(Dataset[dict[str, Any]]):
    @classmethod
    def from_config(cls, cfg): ...
```

JSONL manifest 第一版 schema：

```json
{
  "sample_id": "task0_ep000_chunk000",
  "a_clean_path": "datasets/actions/sample_000/a_clean.pt",
  "context_path": "datasets/actions/sample_000/context.pt",
  "z_condition_path": "datasets/actions/sample_000/z_condition.pt",
  "shape": [4, 14],
  "dtype": "float32",
  "prompt": "adjust the bottle",
  "source": "teacher_or_demo"
}
```

说明：

- `a_clean_path` 必须存在于真实服务器 runtime；Mac 测试使用 temp file。
- `context_path` 可选。
- `z_condition_path` 可选。Action head 训练早期可以使用 teacher video condition 或 manifest 里预计算好的 video condition。
- loader 不应该在 import 阶段读文件，只在 `__getitem__` 读。
- collate 返回：

```python
{
  "a_clean": Tensor,
  "context": list[Any],
  "z_condition": Tensor | list | None,
  "prompt": list[str],
  "meta": list[dict]
}
```

### 4.3 新增 action FFM loss

修改：

```text
src/wam_policy/opd/losses.py
```

新增：

```python
def action_ffm_average_velocity_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

可以把 video/action 的 MSE 公共逻辑抽成：

```python
def average_velocity_mse_loss(pred_u, target_u, mask=None, metric_prefix="loss/ffm"):
    ...
```

但不要破坏现有 `video_ffm_average_velocity_loss` 的 public API。

### 4.4 Mac 验收测试

新增测试：

```text
tests/test_action_flowmap_pair.py
tests/test_action_dataset.py
tests/test_action_ffm_loss.py
```

测试覆盖：

- `build_action_flowmap_pair` 对 `[B,T,D]` 和 `[B,D]` 都能返回同 shape。
- `s_src >= s_tgt` 时抛清晰错误。
- `eps` shape 不匹配时抛清晰错误。
- temp JSONL manifest 能加载 action tensor。
- action loss shape mismatch / NaN 能抛错。

---

## 5. 阶段 C：Action student 与 LingBot action adapter 接口

这是第三批 commit。目标是在 Mac 本地建立 action FFM 的接口边界，但不要求真实 LingBot action forward 在 Mac 上可运行。

### 5.1 Student wrapper

修改：

```text
src/wam_policy/models/lingbot_student.py
```

新增：

```python
class LingBotActionFlowMapStudent(nn.Module):
    def __init__(self, cfg, adapter): ...
    def build(self) -> None: ...
    def freeze_by_config(self) -> None: ...
    def forward_action_flowmap(self, a_src, s_src, s_tgt, context, z_condition=None) -> torch.Tensor: ...
```

实现原则：

- 结构尽量镜像 `LingBotVideoFlowMapStudent`。
- 使用同一个 `FlowTimePairEmbedding` 类，不新增复杂 conditioner。
- 默认 `freeze_backbone: true`。
- `time_pair_conditioner` 参数永远 trainable。
- `pred_u.shape == a_src.shape`，不符合就抛错。

### 5.2 LingBot adapter action hook

修改：

```text
src/wam_policy/adapters/lingbot_va.py
```

新增 public methods：

```python
def build_action_context(self, batch: Mapping[str, Any]) -> Any: ...

def attach_action_time_pair_conditioner(
    self,
    model: nn.Module,
    conditioner: nn.Module,
) -> nn.Module: ...

def install_action_flowmap_hook(
    self,
    model: nn.Module,
    conditioner: nn.Module,
) -> nn.Module: ...

def forward_action_flowmap(
    self,
    model: nn.Module,
    a_src: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    context: Any,
    z_condition: torch.Tensor | None = None,
) -> torch.Tensor: ...
```

### 5.3 关键：mode-specific two-time conditioner

当前 video hook 使用类似：

```text
model._wam_current_s_src
model._wam_current_s_tgt
```

为了同时支持 video/action，建议改成 mode-specific，但必须保持 backward compatibility：

```text
model._wam_current_flowmap_mode = "video" | "action" | None
model._wam_current_video_s_src
model._wam_current_video_s_tgt
model._wam_current_action_s_src
model._wam_current_action_s_tgt
model.wam_video_time_pair_conditioner
model.wam_action_time_pair_conditioner
```

兼容旧 video 代码：

```text
model.wam_time_pair_conditioner 仍可作为 video alias
```

patch `condition_embedder.forward` / `_time_embed` 时，不要无条件使用同一个 conditioner。应该根据 `model._wam_current_flowmap_mode` 选择：

```python
if mode == "video":
    conditioner = model.wam_video_time_pair_conditioner
    s_src = model._wam_current_video_s_src
    s_tgt = model._wam_current_video_s_tgt
elif mode == "action":
    conditioner = model.wam_action_time_pair_conditioner
    s_src = model._wam_current_action_s_src
    s_tgt = model._wam_current_action_s_tgt
else:
    return original_temb, original_timestep_proj
```

不要让 action FFM 改动破坏已有 video FFM 服务器路径。

### 5.4 `build_lingbot_action_input` 的策略

由于 Mac 本地不加载真实 LingBot-VA action path，第一版允许 `build_lingbot_action_input` 只定义接口和做 shape validation：

```python
def build_lingbot_action_input(
    self,
    model,
    a_src,
    s_src,
    s_tgt,
    context,
    z_condition=None,
) -> dict[str, Any]:
    ...
```

要求：

- 不在 import 阶段依赖 `wan_va`。
- 能被 fake-model test 调用。
- 如果真实 LingBot action-mode input contract 尚未确认，non-fake runtime 中可以 raise 清晰错误，例如：

```python
raise NotImplementedError(
    "LingBot action flow-map binding needs server-side probe against real action_mode forward."
)
```

但同时必须提供 server probe 脚本帮助管理员确认真实 contract。

### 5.5 Action runtime probe

新增：

```text
scripts/tools/probe_action_ffm_runtime.py
```

参考现有 `probe_video_ffm_runtime.py`，至少支持：

```text
probe-static-source
probe-import
probe-load-model
probe-manifest-one-sample
probe-action-forward
```

Mac 本地只跑：

```bash
python scripts/tools/probe_action_ffm_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_ffm.yaml
```

服务器上再跑 import/load/manifest/forward。

---

## 6. 阶段 D：ActionFFMTrainer

这是第四批 commit。目标是 action FFM 的 trainer 结构和 video FFM 对齐。

### 6.1 stages.py

修改：

```text
src/wam_policy/opd/stages.py
```

新增：

```python
def train_action_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = ActionFFMTrainer(cfg)
    return trainer.train()
```

### 6.2 trainer.py

新增：

```python
class ActionFFMTrainer:
    def __init__(self, cfg, adapter=None, student=None, dataset=None): ...
    def train(self) -> dict[str, Any]: ...
    def _build_student(self) -> None: ...
    def _build_optimizer(self) -> torch.optim.Optimizer: ...
    def _build_dataloader(self) -> DataLoader: ...
    def _train_step(self, batch) -> dict[str, float]: ...
    def _save_checkpoint(self, step, metrics) -> None: ...
```

训练 step mirror video FFM：

```python
a_clean = normalize_action_chunk_batch(batch["a_clean"]).to(device, dtype)
context = adapter.build_action_context(batch)
z_condition = batch.get("z_condition")

s_src, s_tgt = sample_s_pairs(...)
flowmap_pair = build_action_flowmap_pair(a_clean, s_src, s_tgt)

pred_u = student.forward_action_flowmap(
    flowmap_pair.a_src,
    flowmap_pair.s_src,
    flowmap_pair.s_tgt,
    context,
    z_condition=z_condition,
)

loss, metrics = action_ffm_average_velocity_loss(pred_u, flowmap_pair.target_u)
```

### 6.3 normalize helper

新增：

```python
def normalize_action_chunk_batch(a_clean: torch.Tensor) -> torch.Tensor:
    ...
```

第一版规则：

- 接受 `[B,T,D]`。
- 接受 `[B,D]`。
- 接受 `[B,1,T,D]` 时 squeeze 第二维。
- 其他 shape 抛清晰错误。

### 6.4 checkpoint path

建议 checkpoint default：

```text
experiments/checkpoints/v2/action_ffm
```

不要复用 video checkpoint dir。

### 6.5 Mac fake test

新增测试：

```text
tests/test_action_ffm_trainer_static.py
```

用 fake adapter、fake student、temp dataset，验证：

- `_train_step` 能跑一个 CPU step。
- loss metrics key 包含 `loss/action_ffm`。
- checkpoint save helper 可被 monkeypatch，避免写大文件。

---

## 7. 阶段 E：DMD-style OPD 的本地接口与 buffer schema

这是第五批 commit。目标不是在 Mac 上跑 RoboTwin，而是先定义 on-policy rollout / relabel / DMD correction 的数据边界。

### 7.1 OPD rollout record schema

修改或新增：

```text
src/wam_policy/opd/buffers.py
src/wam_policy/data/rollout_manifest_dataset.py
```

建议 dataclass：

```python
@dataclass(frozen=True)
class OnPolicyChunkRecord:
    sample_id: str
    task: str
    chunk_index: int
    context_path: str | None
    z_student_path: str | None
    a_student_path: str | None
    z_condition_path: str | None
    teacher_video_label_path: str | None
    teacher_action_label_path: str | None
    env_meta: dict[str, Any]
```

第一版 rollout manifest JSONL 示例：

```json
{
  "sample_id": "student_rollout_ep000_chunk003",
  "task": "adjust_bottle",
  "chunk_index": 3,
  "context_path": "datasets/rollouts/ep000/chunk003/context.pt",
  "z_student_path": "datasets/rollouts/ep000/chunk003/z_student.pt",
  "a_student_path": "datasets/rollouts/ep000/chunk003/a_student.pt",
  "z_condition_path": "datasets/rollouts/ep000/chunk003/z_condition.pt",
  "teacher_video_label_path": null,
  "teacher_action_label_path": null,
  "env_meta": {
    "success": false,
    "seed": 0
  }
}
```

Mac 本地测试只用 temp manifest 和 fake tensors。

### 7.2 RoboTwin adapter stub

新增或补齐：

```text
src/wam_policy/adapters/robotwin.py
```

第一版只定义接口：

```python
class RoboTwinAdapter:
    def __init__(self, cfg): ...
    def make_env(self): ...
    def rollout_student_episode(...): ...
    def save_rollout_manifest(...): ...
```

Mac 本地默认不 import RoboTwin。如果调用真实 env 方法，抛：

```python
NotImplementedError(
    "RoboTwin rollout must run on the server/compute node with external/RoboTwin installed."
)
```

### 7.3 DMD loss 接口

修改：

```text
src/wam_policy/opd/losses.py
```

新增一个 generic DMD-style surrogate。第一版使用显式 score/velocity correction 输入，不在 loss 内部调用 teacher/student：

```python
def dmd_score_surrogate_loss(
    student_state: torch.Tensor,
    teacher_score: torch.Tensor,
    student_score: torch.Tensor,
    weight: torch.Tensor | float | None = None,
    metric_prefix: str = "video_dmd",
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

建议第一版语义：

```text
correction = stop_grad(student_score - teacher_score)
loss = mean(sum_flat(student_state * correction) * weight)
```

注意：

- `teacher_score`、`student_score` 必须 detach，不允许梯度流入 teacher。
- `student_state` 保持可导。
- 这是 DMD-style gradient surrogate，不是普通 MSE。
- 真实 score/velocity 如何从 LingBot teacher 和 flow-map student 中取，由 adapter/trainer 负责，不放进 loss。

同时新增：

```python
def ffm_anchor_loss(...)
```

或直接复用已有 video/action FFM average velocity loss 作为 anchor。

### 7.4 AnyFlow backward-simulation utilities

扩展：

```text
src/wam_policy/adapters/anyflow.py
```

新增轻量工具，不 import AnyFlow runtime：

```python
@dataclass(frozen=True)
class FlowMapSegment:
    s_src: torch.Tensor
    s_mid: torch.Tensor | None
    s_tgt: torch.Tensor
    step_index: int | None
    num_steps: int | None


def sample_flowmap_backward_segment(...): ...
def estimate_instant_velocity_from_flowmap(pred_u, ...): ...
def flowmap_dmd_weight(...): ...
```

Mac 测试只检查 shape 和边界条件。

---

## 8. 阶段 F：VideoDMDOPDTrainer

这是第六批 commit。目标是建立 video DMD OPD 的 trainer，并能通过 fake adapter/fake dataset 的 CPU 测试。

### 8.1 stages.py

新增：

```python
def train_video_dmd_opd(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = VideoDMDOPDTrainer(cfg)
    return trainer.train()
```

### 8.2 Trainer 结构

新增：

```python
class VideoDMDOPDTrainer:
    def __init__(self, cfg, adapter=None, student=None, rollout_dataset=None): ...
    def train(self) -> dict[str, Any]: ...
    def _train_step(self, batch) -> dict[str, float]: ...
```

训练 step 第一版逻辑：

```text
rollout batch
→ load on-policy context C_k^S
→ load or reconstruct z_student / z_condition
→ sample s_src/s_tgt segment
→ student.forward_video_flowmap(...)
→ adapter.teacher_video_score_or_velocity(...)
→ adapter.student_video_score_or_velocity(...) 或 small-interval flowmap approximation
→ dmd_score_surrogate_loss(...)
→ optional video_ffm anchor
→ optimizer step
```

Mac 本地：

- 用 fake adapter 提供 `teacher_video_score_or_velocity`。
- 用 fake student 输出同 shape tensor。
- 不加载真实 LingBot。
- 不读取真实 rollout manifest。

真实服务器：

- 如果 teacher score binding 还没实现，probe 应该在 `05_video_dmd_forward` 阶段给出清晰错误。
- 不要让错误出现在 `import` 或 config load 阶段。

### 8.3 Teacher/student score binding 占位接口

在 `LingBotVAAdapter` 中新增：

```python
def teacher_video_score_or_velocity(...):
    raise NotImplementedError(...)

def student_video_score_or_velocity_from_flowmap(...):
    ...
```

第一版可以用 small interval flow-map approximation：

```text
s_eps = min(s_tgt, s_src + dmd.small_delta_s)
u_small = student.forward_video_flowmap(x_s, s_src, s_eps, context)
```

但 teacher side 的真实 velocity/score 需要服务器 probe 结合 LingBot 原始 velocity forward 确认。

---

## 9. 阶段 G：ActionDMDOPDTrainer

这是第七批 commit。目标是建立 action DMD OPD 的 trainer，并确保 action DMD 条件在 student video condition 上，而不是 teacher video condition。

### 9.1 stages.py

新增：

```python
def train_action_dmd_opd(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = ActionDMDOPDTrainer(cfg)
    return trainer.train()
```

### 9.2 Trainer 结构

新增：

```python
class ActionDMDOPDTrainer:
    def __init__(self, cfg, adapter=None, student=None, rollout_dataset=None): ...
    def train(self) -> dict[str, Any]: ...
    def _train_step(self, batch) -> dict[str, float]: ...
```

训练 step 第一版逻辑：

```text
rollout batch
→ load C_k^S
→ load A_student or A_clean/action rollout target
→ load z_condition = student video prediction from rollout
→ sample s_src/s_tgt
→ student.forward_action_flowmap(A_src, s_src, s_tgt, C_k^S, z_condition=Z_student)
→ teacher_action_score_or_velocity(..., z_condition=Z_student)
→ student_action_score_or_velocity_from_flowmap(...)
→ dmd_score_surrogate_loss(...)
→ optional action_ffm anchor
→ optimizer step
```

关键约束：

```text
Action DMD 的 z_condition 必须默认来自 student video rollout，而不是 teacher video。
```

否则 action head 部署时仍然会遇到 condition mismatch。

### 9.3 Adapter 占位接口

在 `LingBotVAAdapter` 中新增：

```python
def teacher_action_score_or_velocity(...):
    raise NotImplementedError(...)

def student_action_score_or_velocity_from_flowmap(...):
    ...
```

Mac 本地 fake-test 使用 fake adapter。

---

## 10. 阶段 H：ROCm/AMD 服务器诊断脚本

这是第八批 commit。目标是让管理员像当前 video FFM 一样，用 Slurm 在 AMD/ROCm 服务器逐阶段验证。

### 10.1 Action FFM diagnostic

新增：

```text
scripts/slurm/action_ffm_admin_diagnostic.sbatch
```

建议阶段：

```text
00_env
01_static
02_import
03_load_model
04_manifest
05_action_forward
06_train_long
```

必需环境变量：

```text
TEACHER_CKPT=/abs/path/to/lingbot/transformer
ACTION_DEBUG_MANIFEST=/abs/path/to/action_chunks_debug.jsonl
ACTION_FULL_MANIFEST=/abs/path/to/action_chunks_full.jsonl
```

可选：

```text
REPO_ROOT
CONFIG=configs/experiments/v2/anyflow_action_ffm.yaml
TRAIN_MAX_STEPS
TRAIN_SAVE_EVERY
TRAIN_LOG_EVERY
TRAIN_BATCH_SIZE
WANDB_ENABLED
WANDB_PROJECT
WANDB_ENTITY
WANDB_RUN_NAME
WANDB_RUN_ID
```

必须默认覆盖：

```bash
--set runtime.device=rocm
--set runtime.lingbot_attn_mode=torch
--set teacher.checkpoint_path="$TEACHER_CKPT"
--set data.manifest="$ACTION_DEBUG_MANIFEST" 或 "$ACTION_FULL_MANIFEST"
```

### 10.2 DMD OPD diagnostic

新增：

```text
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

建议通过 `STAGE` 选择：

```text
STAGE=video_dmd | action_dmd
```

必需环境变量：

```text
TEACHER_CKPT=/abs/path/to/lingbot/transformer
ROLLOUT_DEBUG_MANIFEST=/abs/path/to/student_rollouts_debug.jsonl
ROLLOUT_FULL_MANIFEST=/abs/path/to/student_rollouts_full.jsonl
```

建议阶段：

```text
00_env
01_static
02_import
03_load_model
04_rollout_manifest
05_dmd_forward
06_train_short_or_long
```

第一版可以让 `06_train_short_or_long` 默认 `TRAIN_MAX_STEPS=2`，等 teacher score binding 稳定后再放开长训练。

### 10.3 ROCm 注意事项

- `runtime.device=rocm` 应该被解析到 torch 可用的 HIP/CUDA-compatible device。
- `runtime.lingbot_attn_mode=torch` 是默认安全选择，不要默认启用 flash-attn。
- probe 脚本里可以打印：

```python
torch.__version__
torch.version.hip
torch.version.cuda
torch.cuda.is_available()
torch.cuda.get_device_name(0)
```

- 不要在 Mac 本地要求 `rocm-smi` 存在。

---

## 11. Mac 本地验收标准

Codex 完成本计划后，Mac 本地至少应该能通过以下轻量命令。

### 11.1 Static dispatch

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static
```

### 11.2 Probe static source

```bash
python scripts/tools/probe_action_ffm_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_ffm.yaml

python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_video_dmd_opd.yaml

python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_dmd_opd.yaml
```

这些命令不应该要求真实 external repo 存在。如果脚本提供 `--require-external-exists`，只有显式传入时才检查 external repo。

### 11.3 Py compile

```bash
python -m py_compile \
  scripts/train.py \
  scripts/tools/probe_action_ffm_runtime.py \
  scripts/tools/probe_dmd_opd_runtime.py \
  src/wam_policy/adapters/anyflow.py \
  src/wam_policy/adapters/lingbot_va.py \
  src/wam_policy/adapters/robotwin.py \
  src/wam_policy/data/action_dataset.py \
  src/wam_policy/data/rollout_manifest_dataset.py \
  src/wam_policy/opd/losses.py \
  src/wam_policy/opd/stages.py \
  src/wam_policy/opd/trainer.py
```

### 11.4 Targeted tests

只跑轻量 targeted tests：

```bash
pytest \
  tests/test_pipeline_static_dispatch.py \
  tests/test_action_flowmap_pair.py \
  tests/test_action_dataset.py \
  tests/test_action_ffm_loss.py \
  tests/test_dmd_losses.py \
  tests/test_rollout_manifest_dataset.py \
  tests/test_probe_scripts_static.py
```

不要在 Mac 本地跑 full pytest，如果 full pytest 中包含会加载真实 LingBot/RoboTwin/GPU 的测试。

---

## 12. AMD/ROCm 服务器验收标准

服务器验收由管理员通过 Slurm 执行。Mac 本地不需要满足这些条件。

### 12.1 Action FFM server smoke

```bash
sbatch --export=ALL,TEACHER_CKPT="$TEACHER_CKPT",ACTION_DEBUG_MANIFEST="$ACTION_DEBUG_MANIFEST",ACTION_FULL_MANIFEST="$ACTION_FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/action_ffm_admin_diagnostic.sbatch
```

预期：

```text
00_env PASSED
01_static PASSED
02_import PASSED
03_load_model PASSED
04_manifest PASSED
05_action_forward PASSED 或给出明确 action binding mismatch
06_train_long 在前面通过后能跑 2 step
```

如果 `05_action_forward` 失败，Codex 下一轮只修改 `LingBotVAAdapter` action binding，不要把 LingBot-specific 逻辑扩散到 trainer。

### 12.2 Video DMD server smoke

```bash
sbatch --export=ALL,STAGE=video_dmd,TEACHER_CKPT="$TEACHER_CKPT",ROLLOUT_DEBUG_MANIFEST="$ROLLOUT_DEBUG_MANIFEST",ROLLOUT_FULL_MANIFEST="$ROLLOUT_FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

预期：

```text
config/static/manifest/model load 先通过
teacher score / velocity binding 如未完成，必须在 05_dmd_forward 清晰失败
不能在 import/config 阶段失败
```

### 12.3 Action DMD server smoke

```bash
sbatch --export=ALL,STAGE=action_dmd,TEACHER_CKPT="$TEACHER_CKPT",ROLLOUT_DEBUG_MANIFEST="$ROLLOUT_DEBUG_MANIFEST",ROLLOUT_FULL_MANIFEST="$ROLLOUT_FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

关键检查：

```text
z_condition 默认来自 student rollout manifest
不是 teacher video condition
```

---

## 13. 推荐 commit 切片顺序

### Commit 1：Configs + pipeline dry-run dispatch

包含：

```text
configs/experiments/v2/anyflow_action_ffm.yaml
configs/experiments/v2/anyflow_video_dmd_opd.yaml
configs/experiments/v2/anyflow_action_dmd_opd.yaml
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py
tests/test_pipeline_static_dispatch.py
```

验收：四个 stage dry-run static 都通过。

### Commit 2：Action flow-map pair + action dataset + action loss

包含：

```text
src/wam_policy/adapters/anyflow.py
src/wam_policy/data/action_dataset.py
src/wam_policy/opd/losses.py
tests/test_action_flowmap_pair.py
tests/test_action_dataset.py
tests/test_action_ffm_loss.py
```

验收：纯 CPU fake tensor tests 通过。

### Commit 3：Action student + LingBot action adapter skeleton

包含：

```text
src/wam_policy/models/lingbot_student.py
src/wam_policy/adapters/lingbot_va.py
scripts/tools/probe_action_ffm_runtime.py
tests/test_action_student_static.py
tests/test_probe_scripts_static.py
```

验收：Mac 上 probe-static-source 和 fake adapter tests 通过；video FFM 原路径不被破坏。

### Commit 4：ActionFFMTrainer

包含：

```text
src/wam_policy/opd/stages.py
src/wam_policy/opd/trainer.py
src/wam_policy/adapters/checkpoint.py  # 如需 stage-specific checkpoint helper
scripts/slurm/action_ffm_admin_diagnostic.sbatch
```

验收：fake dataset/fake student/fake adapter 训练 1 step；Slurm 脚本可 py_compile/shell 静态检查。

### Commit 5：OPD rollout schema + DMD losses/utilities

包含：

```text
src/wam_policy/opd/buffers.py
src/wam_policy/data/rollout_manifest_dataset.py
src/wam_policy/adapters/robotwin.py
src/wam_policy/adapters/anyflow.py
src/wam_policy/opd/losses.py
tests/test_rollout_manifest_dataset.py
tests/test_dmd_losses.py
```

验收：Mac fake rollout manifest 和 fake DMD loss 通过。

### Commit 6：VideoDMDOPDTrainer + DMD probe

包含：

```text
src/wam_policy/opd/stages.py
src/wam_policy/opd/trainer.py
src/wam_policy/adapters/lingbot_va.py
scripts/tools/probe_dmd_opd_runtime.py
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

验收：video_dmd dry-run static、fake trainer step、probe-static-source 通过。

### Commit 7：ActionDMDOPDTrainer

包含：

```text
src/wam_policy/opd/stages.py
src/wam_policy/opd/trainer.py
src/wam_policy/adapters/lingbot_va.py
tests/test_action_dmd_trainer_static.py
```

验收：action_dmd dry-run static、fake trainer step 通过；默认 z_condition 使用 student rollout。

### Commit 8：文档与 README 更新

包含：

```text
docs/agent_handoff/<date>/WAM-Policy_v2_Action-FFM_&_Video+Action_DDM_OPD.md
README.md 或 docs/experiments/v2_runtime.md
```

验收：管理员能按文档在 AMD/ROCm 服务器提交 action FFM 和 DMD OPD smoke job。

---

## 14. 实现时的红线

Codex 不要做以下事情：

1. 不要把 checkpoint、manifest、latent、rollout、wandb、log 文件加入 Git。
2. 不要在 Mac 本地尝试下载 LingBot checkpoint 或 RoboTwin dataset。
3. 不要在 module import 阶段 import `wan_va`、RoboTwin 或 AnyFlow runtime。
4. 不要在轻量测试中调用真实 GPU、ROCm、torchrun、RoboTwin simulation。
5. 不要改 `external/lingbot-va` 源码；真实绑定不匹配时，通过 `LingBotVAAdapter` 修复。
6. 不要把 action DMD 的 video condition 默认写成 teacher video。必须默认 student rollout video。
7. 不要用 `tau/rho` 替换现有 `s_src/s_tgt` API。
8. 不要让新 action conditioner 破坏已有 video FFM 的 conditioner patch。
9. 不要把 DMD loss 写成普通 teacher-student MSE 后声称是 DMD。第一版可以是 DMD-style surrogate，但必须在函数名和 docstring 中说明梯度语义。
10. 不要要求 Mac 本地完整跑通真实训练。Mac 本地只负责接口和静态 debug。

---

## 15. 最小完成定义

本轮 Codex 完成后，应该达到：

```text
[ ] video_ffm 现有 dry-run 和服务器路径不被破坏。
[ ] action_ffm config 存在，dry-run static 通过。
[ ] video_dmd config 存在，dry-run static 通过。
[ ] action_dmd config 存在，dry-run static 通过。
[ ] action flow-map pair/loss/dataset 有 CPU fake tests。
[ ] action student wrapper 和 adapter hook skeleton 存在。
[ ] mode-specific video/action two-time conditioner 不互相污染。
[ ] ActionFFMTrainer 能用 fake adapter 跑 1 个 CPU step。
[ ] rollout manifest dataset 和 DMD surrogate loss 有 CPU fake tests。
[ ] VideoDMDOPDTrainer / ActionDMDOPDTrainer 有 fake trainer tests 或至少清晰接口。
[ ] action FFM Slurm diagnostic script 存在。
[ ] DMD OPD Slurm diagnostic script 存在。
[ ] 所有真实 LingBot/RoboTwin/ROCm 依赖都延后到服务器 probe 阶段。
```

