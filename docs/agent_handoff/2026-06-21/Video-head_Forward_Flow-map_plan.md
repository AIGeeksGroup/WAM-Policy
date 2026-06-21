# Video-head Forward Flow-map Training Implementation Plan

本文档用于 hand off 给 Codex，在刚创建好的 `WAM-Policy/` repo 骨架上，实现 **WAM-Policy_v2 的 video-head Forward Flow-map Training** 代码路径。

本阶段只关注 **video head 的 Forward Flow-map Training**。目标不是在 macOS 本地跑通真实训练，而是写出结构清晰、静态检查可过、后续可迁移到 AMD 服务器进行 runtime debug 的代码。

---

## 0. 已确定的三个前提

### 0.1 外部 repo 接入方式

`LingBot-VA` 和 `AnyFlow` repo 会被拉到：

```text
WAM-Policy/external/lingbot-va/
WAM-Policy/external/AnyFlow/
```

这两个 external repo 不进入 WAM-Policy 的 Git 跟踪。

代码实现采用 **adapter 层 direct import** 策略：

```text
trainer / loss / dataset / pipeline
    ↓
WAM-Policy adapter
    ↓
direct import external/lingbot-va and external/AnyFlow
```

要求：

- 只允许 `src/wam_policy/adapters/lingbot_va.py` 直接 import LingBot-VA。
- 只允许 `src/wam_policy/adapters/anyflow.py` 直接 import AnyFlow。
- `trainer.py`、`stages.py`、`losses.py`、`data/*.py`、`pipelines/*.py` 不直接 import 外部 repo。
- 通过 `bootstrap_external_repos()` 统一把 external repo 路径加入 `sys.path`。
- 不使用复杂的动态 import 猜测逻辑；如果真实 package 名不匹配，就在 adapter 层修复。

注意：`external/lingbot-va` 目录名中有 hyphen，Python 不能写 `import lingbot-va`。Codex 需要在 external repo 拉下来后检查其内部真实 package 名，例如通过：

```bash
find external/lingbot-va -maxdepth 2 -name "__init__.py"
find external/AnyFlow -maxdepth 2 -name "__init__.py"
```

---

### 0.2 时间方向

WAM-Policy_v2 代码内部统一使用 LingBot-VA 的 flow time convention：

```text
s = 0: noise
s = 1: clean data
```

Forward Flow-map 使用 two-time pair：

```text
0 <= s_src < s_tgt <= 1
```

不要在本阶段代码中使用 `tau/rho` 作为变量名。所有 trainer、loss、dataset、adapter 统一使用：

```text
s_src
s_tgt
delta_s = s_tgt - s_src
```

对应插值为：

```text
Z_s = (1 - s) * eps + s * Z_clean
```

video flow-map student 预测 average velocity：

```text
u_S^v(Z_src, s_src, s_tgt | C_k)
```

supervised target：

```text
target_u = (Z_tgt - Z_src) / (s_tgt - s_src)
```

采样更新关系：

```text
Z_tgt = Z_src + pred_u * (s_tgt - s_src)
```

---

### 0.3 当前代码起点

计划起点是：**刚创建好的 WAM-Policy repo 骨架**。

不要假设已有旧实现，也不要规划“删除旧实现”。本计划只描述从骨架开始需要新增和完善的真实 LingBot-bound video FFM 训练路径。

---

## 1. 今日目标

实现一条 server-ready 的 video-head Forward Flow-map Training 主路径：

```text
config
→ scripts/train.py
→ bootstrap external repos
→ v2_anyflow_dual_head pipeline
→ video_ffm stage
→ video latent manifest dataset
→ s_src/s_tgt flow-map pair construction
→ LingBotVideoFlowMapStudent
→ video FFM average-velocity loss
→ optimizer step
→ checkpoint save
```

今天不追求 macOS 上加载真实 LingBot checkpoint 或真实训练成功。macOS 本地只做静态检查和 config dry-run。

---

## 2. 本阶段不做的内容

本阶段不实现：

```text
action-head Forward Flow-map Training
video-head DMD / OPD correction
action-head DMD / OPD correction
RoboTwin online rollout
teacher relabeling
success-rate evaluation
distributed training
FSDP / DeepSpeed / ZeRO
完整部署 sampler
```

---

## 3. 目标训练公式

给定 clean video latent chunk：

```text
Z_clean = Z_k^*
```

采样高斯噪声：

```text
eps ~ N(0, I)
```

采样 flow-map 时间对：

```text
0 <= s_src < s_tgt <= 1
```

构造：

```text
Z_src = (1 - s_src) * eps + s_src * Z_clean
Z_tgt = (1 - s_tgt) * eps + s_tgt * Z_clean
```

目标 average velocity：

```text
target_u = (Z_tgt - Z_src) / (s_tgt - s_src)
```

student video flow-map head 预测：

```text
pred_u = u_S^v(Z_src, s_src, s_tgt | C_k)
```

loss：

```text
L_video_ffm = mean((pred_u - target_u)^2)
```

要求：

- `pred_u.shape == target_u.shape == Z_clean.shape`
- `s_src.shape == s_tgt.shape == [B]`
- trainer 内只使用 `s_src/s_tgt`，不使用反向时间记法。

---

## 4. 目录和文件清单

本阶段建议新增或完善以下文件：

```text
configs/experiments/v2/anyflow_video_ffm.yaml

scripts/train.py
scripts/tools/inspect_external_bindings.py

src/wam_policy/utils/config.py
src/wam_policy/utils/paths.py
src/wam_policy/utils/runtime.py
src/wam_policy/utils/external.py
src/wam_policy/utils/logging.py
src/wam_policy/utils/seed.py

src/wam_policy/adapters/lingbot_va.py
src/wam_policy/adapters/anyflow.py
src/wam_policy/adapters/checkpoint.py

src/wam_policy/models/flow_map_heads.py
src/wam_policy/models/lingbot_student.py
src/wam_policy/models/dual_head_student.py

src/wam_policy/data/manifests.py
src/wam_policy/data/video_latent_dataset.py

src/wam_policy/opd/losses.py
src/wam_policy/opd/trainer.py
src/wam_policy/opd/stages.py
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py

tests/test_config_static.py
tests/test_external_bootstrap.py
tests/test_anyflow_s_pairs.py
tests/test_manifest_schema.py
tests/test_pipeline_static_dispatch.py
```

---

## 5. Step-by-step implementation plan

## Step 1: Implement external repo bootstrap

Create:

```text
src/wam_policy/utils/external.py
```

Responsibilities:

```text
1. Resolve WAM-Policy repo root.
2. Read external repo paths from config.
3. Add external/lingbot-va to sys.path.
4. Add external/AnyFlow to sys.path.
5. Optionally check that the paths exist.
6. Do not load checkpoints.
7. Do not instantiate models.
```

Suggested function:

```python
def bootstrap_external_repos(cfg: Mapping[str, Any]) -> None:
    """Add configured external repos to sys.path."""
```

Expected call order in `scripts/train.py`:

```text
load config
→ apply --set overrides
→ bootstrap_external_repos(cfg)
→ import selected pipeline
→ run pipeline
```

Source classification:

| Part | Source |
|---|---|
| `sys.path`, `pathlib`, environment variables | Python API |
| external repo path convention | WAM-Policy repo design |
| bootstrap logic | Implement in WAM-Policy |

---

## Step 2: Write video FFM experiment config

Create:

```text
configs/experiments/v2/anyflow_video_ffm.yaml
```

Recommended config:

```yaml
experiment:
  name: anyflow_video_ffm
  version: v2
  stage: video_ffm

paths:
  repo_root: .
  lingbot_repo: external/lingbot-va
  anyflow_repo: external/AnyFlow
  output_dir: experiments/runs/v2/video_ffm
  checkpoint_dir: experiments/checkpoints/v2/video_ffm

teacher:
  name: lingbot_va
  checkpoint_path: experiments/pretrained_models/lingbot-va/PLACEHOLDER
  frozen: true

model:
  backend: lingbot_va
  student_type: video_flowmap
  output_type: average_velocity
  time_condition: s_src_s_tgt_delta
  freeze_backbone: true
  trainable_modules:
    - video_time_pair_adapter
    - video_head_adapter

data:
  mode: video_latent_manifest
  manifest: datasets/manifests/video_latents_debug.jsonl
  latent_root: datasets/latents
  context_mode: precomputed_or_rebuild
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

Source classification:

| Part | Source |
|---|---|
| YAML loading | Python/YAML API |
| `s=0 noise, s=1 clean` | LingBot-VA convention |
| two-time flow-map fields | AnyFlow / WAM-Policy_v2 design |
| config schema | Implement in WAM-Policy |

---

## Step 3: Implement config, CLI, and static dry-run

Create or complete:

```text
scripts/train.py
src/wam_policy/utils/config.py
src/wam_policy/utils/paths.py
src/wam_policy/utils/runtime.py
src/wam_policy/utils/seed.py
src/wam_policy/utils/logging.py
```

`train.py` should support:

```bash
python scripts/train.py \
  --config configs/experiments/v2/anyflow_video_ffm.yaml \
  --dry-run static
```

`--dry-run static` checks:

```text
config can be loaded
--set overrides can be applied
repo root can be resolved
external repo path fields exist in config
experiment.stage == video_ffm
pipeline can be located
no checkpoint loading
no LingBot model construction
```

Real server command, later:

```bash
python scripts/train.py \
  --config configs/experiments/v2/anyflow_video_ffm.yaml \
  --set teacher.checkpoint_path=experiments/pretrained_models/lingbot-va/<ckpt> \
  --set data.manifest=datasets/manifests/video_latents_debug.jsonl \
  --set runtime.device=cuda \
  --set runtime.precision=bf16
```

Source classification:

| Part | Source |
|---|---|
| `argparse`, `yaml`, `pathlib` | Python API |
| `video_ffm` stage naming | WAM-Policy_v2 design |
| config override, dry-run, path resolution | Implement in WAM-Policy |

---

## Step 4: Implement AnyFlow adapter for Forward Flow-map utilities

Create:

```text
src/wam_policy/adapters/anyflow.py
```

Responsibilities:

1. Centralize direct imports from `external/AnyFlow`.
2. Provide WAM-Policy's `s`-direction flow-map utilities.
3. Keep AnyFlow time convention mapping inside this adapter if needed.

Suggested dataclass:

```python
@dataclass
class VideoFlowMapPair:
    z_clean: torch.Tensor
    eps: torch.Tensor
    s_src: torch.Tensor
    s_tgt: torch.Tensor
    z_src: torch.Tensor
    z_tgt: torch.Tensor
    target_u: torch.Tensor
```

Suggested functions:

```python
def sample_s_pairs(
    batch_size: int,
    s_min: float,
    s_max: float,
    min_delta_s: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return s_src, s_tgt with s_src < s_tgt."""


def expand_time_like(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B] time tensor to match x dimensions."""


def build_video_flowmap_pair(
    z_clean: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> VideoFlowMapPair:
    """Build Z_src, Z_tgt and average-velocity target."""


def average_velocity_target_s(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
) -> torch.Tensor:
    """Compute (z_tgt - z_src) / (s_tgt - s_src)."""
```

Implementation constraints:

```text
s_src < s_tgt
s_tgt - s_src >= min_delta_s
no tau/rho naming in trainer-facing APIs
all returned tensors must stay on the same device as z_clean
```

Source classification:

| Part | Source |
|---|---|
| tensor broadcast and noise sampling | PyTorch API |
| arbitrary time-pair transition | AnyFlow idea |
| `s_src/s_tgt` convention and dataclass | Implement in WAM-Policy |

---

## Step 5: Implement LingBot-VA adapter

Create:

```text
src/wam_policy/adapters/lingbot_va.py
```

This is the only file that should directly bind to LingBot-VA internals.

Suggested structure:

```python
class LingBotVAAdapter:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.repo_path = ...
        self.checkpoint_path = ...

    def load_pretrained_video_action_model(self) -> torch.nn.Module:
        """Load LingBot-VA model and checkpoint."""

    def build_video_context(self, batch: Mapping[str, Any]) -> Any:
        """Build or load C_k for video prediction."""

    def attach_video_time_pair_conditioner(
        self,
        model: torch.nn.Module,
        conditioner: torch.nn.Module,
    ) -> torch.nn.Module:
        """Attach two-time flow-map conditioning to LingBot video pathway."""

    def forward_video_flowmap(
        self,
        model: torch.nn.Module,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """Return average velocity pred_u in LingBot s direction."""
```

Direct import examples, to be corrected after inspecting the actual repo:

```python
# Example only; Codex must replace these with real package names.
from lingbot_va.models import build_model
from lingbot_va.checkpoint import load_checkpoint
```

Binding rule:

```text
Do not guess LingBot internals in trainer.py.
All mismatch between WAM-Policy and LingBot-VA must be handled in this adapter.
```

Source classification:

| Part | Source |
|---|---|
| model builder, checkpoint loader, video forward, context/KV-cache API | LingBot-VA |
| adapter class and unified signature | Implement in WAM-Policy |
| two-time flow-map binding | Modify/wrap LingBot video head for WAM-Policy_v2 |

---

## Step 6: Implement two-time condition module

Create:

```text
src/wam_policy/models/flow_map_heads.py
```

Main class:

```python
class FlowTimePairEmbedding(nn.Module):
    """
    Embed (s_src, s_tgt, delta_s) for LingBot video flow-map training.
    """
```

Inputs:

```text
s_src
s_tgt
delta_s = s_tgt - s_src
```

Output:

```text
time_pair_emb
```

Usage:

```text
Inject into LingBot-VA's original video timestep embedding path.
```

Initialization recommendation:

```text
1. s_src branch should be compatible with the original LingBot timestep embedding if possible.
2. s_tgt and delta_s branches should be zero-init or small-init.
3. Initial student behavior should stay close to original LingBot velocity model.
```

Source classification:

| Part | Source |
|---|---|
| `nn.Module`, MLP, zero-init | PyTorch API |
| original timestep embedding injection point | LingBot-VA |
| `(s_src, s_tgt, delta_s)` embedding | Implement in WAM-Policy |
| two-time conditioning | AnyFlow / WAM-Policy_v2 |

---

## Step 7: Implement LingBot video flow-map student wrapper

Create:

```text
src/wam_policy/models/lingbot_student.py
```

Suggested class:

```python
class LingBotVideoFlowMapStudent(nn.Module):
    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: LingBotVAAdapter,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter
        self.base_model: nn.Module | None = None
        self.time_pair_conditioner = FlowTimePairEmbedding(...)

    def build(self) -> None:
        self.base_model = self.adapter.load_pretrained_video_action_model()
        self.adapter.attach_video_time_pair_conditioner(
            self.base_model,
            self.time_pair_conditioner,
        )
        self.freeze_by_config()

    def forward_video_flowmap(
        self,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """Return pred_u with the same shape as z_src."""
```

Output semantics:

```text
pred_u = average velocity in LingBot s direction
pred_u ≈ (z_tgt - z_src) / (s_tgt - s_src)
```

Source classification:

| Part | Source |
|---|---|
| `nn.Module`, freeze/unfreeze, trainable parameter selection | PyTorch API |
| base model, video head, checkpoint | LingBot-VA |
| wrapper and forward signature | Implement in WAM-Policy |
| flow-map operator semantics | AnyFlow / WAM-Policy_v2 |

---

## Step 8: Implement video latent manifest dataset

Create:

```text
src/wam_policy/data/manifests.py
src/wam_policy/data/video_latent_dataset.py
```

Suggested JSONL manifest item:

```json
{
  "sample_id": "adjust_bottle_000001_chunk_000",
  "task": "adjust_bottle",
  "z_clean": "datasets/latents/adjust_bottle/000001/chunk_000_z.pt",
  "context": "datasets/latents/adjust_bottle/000001/chunk_000_context.pt",
  "prompt": "adjust the bottle",
  "chunk_idx": 0,
  "source": "demo_or_teacher",
  "shape": [1, 16, 21, 32, 32],
  "dtype": "bf16"
}
```

Dataset return format:

```python
{
    "z_clean": Tensor,
    "context": Any | None,
    "prompt": str,
    "meta": dict,
}
```

Scope:

```text
This dataset only loads already-prepared video latents and optional context.
It does not generate latents.
It does not run RoboTwin.
It does not call the teacher.
```

Source classification:

| Part | Source |
|---|---|
| JSONL parser, `torch.load`, DataLoader | Python / PyTorch API |
| video latent shape and context meaning | LingBot-VA |
| manifest schema and path resolution | Implement in WAM-Policy |

---

## Step 9: Implement video FFM loss

Create or complete:

```text
src/wam_policy/opd/losses.py
```

Function:

```python
def video_ffm_average_velocity_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

Minimum behavior:

```python
diff = pred_u - target_u
if mask is not None:
    diff = diff * mask
loss = diff.square().mean()
```

Metrics:

```python
{
    "loss/video_ffm": float(loss.detach()),
    "stat/pred_u_abs_mean": ...,
    "stat/target_u_abs_mean": ...,
}
```

Source classification:

| Part | Source |
|---|---|
| MSE and finite checks | PyTorch API |
| average velocity target | AnyFlow-style FFM |
| metrics dict, mask handling, NaN guard | Implement in WAM-Policy |

---

## Step 10: Implement video-only FFM trainer

Create or complete:

```text
src/wam_policy/opd/trainer.py
src/wam_policy/opd/stages.py
```

Main entry:

```python
def train_video_forward_flowmap(cfg: Mapping[str, Any]) -> None:
    ...
```

Training flow:

```text
1. Set seed.
2. Resolve device and dtype.
3. Build LingBotVAAdapter.
4. Build LingBotVideoFlowMapStudent.
5. Call student.build().
6. Build VideoLatentManifestDataset.
7. Build DataLoader.
8. Build optimizer from trainable parameters.
9. For each step:
   a. Get batch.
   b. Load z_clean.
   c. Build context through adapter.build_video_context(batch).
   d. Sample s_src, s_tgt.
   e. Build VideoFlowMapPair.
   f. Call student.forward_video_flowmap(...).
   g. Compute video_ffm_average_velocity_loss.
   h. Backward.
   i. Clip grad.
   j. Optimizer step.
   k. Log metrics.
   l. Save checkpoint by schedule.
```

Trainer-facing variable names:

```text
z_clean
z_src
z_tgt
s_src
s_tgt
target_u
context
pred_u
```

Do not use alternate time names in trainer code.

Source classification:

| Part | Source |
|---|---|
| optimizer, backward, grad clipping, DataLoader | PyTorch API |
| video forward and context | LingBot-VA adapter |
| flow-map pair construction | AnyFlow adapter |
| loop, logging, checkpoint schedule | Implement in WAM-Policy |

---

## Step 11: Implement checkpoint helper

Create:

```text
src/wam_policy/adapters/checkpoint.py
```

Save format:

```python
{
    "step": step,
    "model": student.state_dict(),
    "optimizer": optimizer.state_dict(),
    "cfg": resolved_cfg,
    "metrics": metrics,
}
```

Paths:

```text
experiments/checkpoints/v2/video_ffm/latest.pt
experiments/checkpoints/v2/video_ffm/step_000500.pt
```

Source classification:

| Part | Source |
|---|---|
| `torch.save`, `torch.load` | PyTorch API |
| LingBot state dict compatibility | LingBot-VA |
| checkpoint schema and metadata | Implement in WAM-Policy |

---

## Step 12: Implement v2 pipeline dispatch

Create or complete:

```text
src/wam_policy/opd/pipelines/v2_anyflow_dual_head.py
```

Logic:

```python
def run(cfg: Mapping[str, Any]) -> None:
    stage = cfg["experiment"]["stage"]

    if stage == "video_ffm":
        return train_video_forward_flowmap(cfg)

    if stage in {"action_ffm", "video_dmd", "action_dmd"}:
        raise NotImplementedError(
            f"{stage} is not implemented yet. Current scope: video_ffm."
        )

    raise ValueError(f"Unknown stage: {stage}")
```

Source classification:

| Part | Source |
|---|---|
| Python dispatch | Python API |
| stage names | WAM-Policy_v2 design |
| pipeline structure | Implement in WAM-Policy |

---

## Step 13: Implement external binding inspection script

Create:

```text
scripts/tools/inspect_external_bindings.py
```

Purpose: inspect external repo package names and likely binding points before launching training.

Responsibilities:

```text
1. Print whether external/lingbot-va exists.
2. Print whether external/AnyFlow exists.
3. Run bootstrap_external_repos().
4. Try importing LingBot-VA's real package.
5. Try importing AnyFlow's real package.
6. Inspect candidate model builder, checkpoint loader, video head, and timestep embedding modules.
7. Write a readable binding report.
```

Server command:

```bash
python scripts/tools/inspect_external_bindings.py \
  --lingbot-repo external/lingbot-va \
  --anyflow-repo external/AnyFlow
```

Source classification:

| Part | Source |
|---|---|
| `inspect`, `importlib`, `pathlib` | Python API |
| inspected modules | LingBot-VA / AnyFlow |
| report format | Implement in WAM-Policy |

---

## Step 14: Write static tests

Create:

```text
tests/test_config_static.py
tests/test_external_bootstrap.py
tests/test_anyflow_s_pairs.py
tests/test_manifest_schema.py
tests/test_pipeline_static_dispatch.py
```

Test coverage:

```text
config can load
repo root can resolve
external repo paths can be appended to sys.path
sample_s_pairs returns s_src < s_tgt
s_tgt - s_src >= min_delta_s
build_video_flowmap_pair returns correct shapes
manifest required keys are validated
video_ffm stage dispatches correctly
unsupported stages raise clear NotImplementedError
```

Static check commands:

```bash
python -m compileall src scripts tests
ruff check src scripts tests
mypy src/wam_policy --ignore-missing-imports
```

`mypy --ignore-missing-imports` is expected because external repos may not ship type stubs.

---

## 6. P0 / P1 / P2 priority

## P0: Must complete today

```text
1. anyflow_video_ffm.yaml
2. train.py + config loader + --set override + --dry-run static
3. bootstrap_external_repos()
4. adapters/anyflow.py with s_src/s_tgt flow-map utilities
5. adapters/lingbot_va.py direct import binding skeleton
6. FlowTimePairEmbedding
7. LingBotVideoFlowMapStudent wrapper
8. VideoLatentManifestDataset
9. video_ffm_average_velocity_loss
10. train_video_forward_flowmap
11. v2_anyflow_dual_head.py dispatch
12. compileall / ruff / mypy static-check path
```

## P1: Complete if time allows

```text
1. inspect_external_bindings.py
2. checkpoint helper
3. manifest schema checker
4. clear external package import error messages
5. docs/migration/lingbot_va_binding_notes.md
```

## P2: Not in today's scope

```text
1. action-head Forward Flow-map Training
2. video-head DMD
3. action-head DMD
4. RoboTwin online rollout
5. teacher relabeling
6. success-rate evaluation
7. distributed training
```

---

## 7. Expected local commands

Mac local static check:

```bash
python -m compileall src scripts tests
ruff check src scripts tests
mypy src/wam_policy --ignore-missing-imports
```

Config dry-run:

```bash
python scripts/train.py \
  --config configs/experiments/v2/anyflow_video_ffm.yaml \
  --dry-run static
```

External binding inspection, after external repos are cloned:

```bash
python scripts/tools/inspect_external_bindings.py \
  --lingbot-repo external/lingbot-va \
  --anyflow-repo external/AnyFlow
```

AMD server first real runtime debug:

```bash
python scripts/train.py \
  --config configs/experiments/v2/anyflow_video_ffm.yaml \
  --set teacher.checkpoint_path=experiments/pretrained_models/lingbot-va/<ckpt> \
  --set data.manifest=datasets/manifests/video_latents_debug.jsonl \
  --set runtime.device=cuda \
  --set runtime.precision=bf16 \
  --set train.max_steps=10
```

---

## 8. Key implementation constraints

1. **Adapter-only direct imports**

   External repo imports should only appear in:

   ```text
   src/wam_policy/adapters/lingbot_va.py
   src/wam_policy/adapters/anyflow.py
   ```

2. **Use LingBot `s` convention everywhere**

   Use:

   ```text
   s_src
   s_tgt
   delta_s
   ```

   Do not use alternate time naming in trainer-facing APIs.

3. **Trainer stays independent of external repo internals**

   Trainer calls only WAM-Policy classes/functions:

   ```text
   LingBotVAAdapter
   LingBotVideoFlowMapStudent
   build_video_flowmap_pair
   video_ffm_average_velocity_loss
   ```

4. **All LingBot binding mismatch is isolated in the adapter**

   If LingBot-VA actual package names, checkpoint loaders, context APIs, or video forward methods differ, fix only:

   ```text
   src/wam_policy/adapters/lingbot_va.py
   ```

5. **Mac is for static checks; AMD server is for runtime debug**

   Do not require local checkpoint loading or CUDA execution to validate today's implementation.

---

## 9. Final target state

After this implementation, the WAM-Policy repo should contain a coherent video-only Forward Flow-map Training path:

```text
fresh WAM-Policy repo skeleton
→ adapter-layer direct import for LingBot-VA / AnyFlow
→ LingBot-style s convention
→ video latent manifest dataset
→ s_src/s_tgt Forward Flow-map pair
→ LingBotVideoFlowMapStudent
→ video FFM average-velocity loss
→ checkpointable trainer
→ static checks and server-ready debug entry
```

This is the minimum implementation needed before moving to AMD server runtime debugging.
