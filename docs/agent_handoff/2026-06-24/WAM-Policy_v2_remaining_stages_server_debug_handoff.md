# WAM-Policy v2 剩余三阶段服务器级 Debug Handoff

目标文件建议保存到 repo：

```text
WAM-Policy/docs/agent_handoff/2026-06-24/WAM-Policy_v2_remaining_stages_server_debug_handoff.md
```

## 0. 当前背景

上一轮 Mac 本地静态测试已经通过。当前 repo 已经具备以下基础：

- `video_ffm` 旧路径仍然存在；
- 新增 `action_ffm`、`video_dmd`、`action_dmd` 的 pipeline dispatch；
- 新增 action flow-map pair、action manifest dataset、action FFM loss、Action FFM trainer；
- 新增 rollout manifest dataset、DMD surrogate loss、Video/Action DMD OPD trainer；
- 新增 action/DMD probe scripts 和分开的 Slurm diagnostic 脚本；
- Mac 本地 fake tensor / static-source tests 已通过。

接下来的目标不是继续只做 Mac 静态测试，而是把整个 repo debug 到类似当前 Video-stream FFM 的管理员可运行状态：

1. 管理员 clone repo；
2. 按 README 准备 external repos、teacher checkpoint、manifest、W&B；
3. 提交一个尽量统一的 `sbatch`；
4. 这个 `sbatch` 自动执行各阶段诊断；
5. 诊断通过后，在 full manifest 上运行 long training；
6. 每个阶段失败时留下清晰日志、first failing stage 和下一步 debug 指向。

当前仍然不要在 Mac 本地下载 checkpoint、full manifest、RoboTwin dataset，也不要在 Mac 本地跑真实 LingBot/RoboTwin/GPU 任务。Mac 本地只允许继续做轻量 static/fake tests。

---

## 1. 本轮必须反馈并修复的 4 个问题

### 问题 1：DMD trainer 当前把 `pred_u` 当作 `student_state`，语义需要修正或显式标注

当前实现中，Video/Action DMD trainer 调用：

```python
loss = dmd_score_surrogate_loss(
    student_state=pred_u,
    teacher_score=teacher_score,
    student_score=student_score,
)
```

这意味着当前 loss 实际是：

```text
< stopgrad(student_score - teacher_score), pred_u >
```

而我们 WAM-Policy_v2 设计中的 DMD-style surrogate 更接近：

```text
< stopgrad(student_score - teacher_score), student_clean_sample >
```

也就是：

- video DMD 应该优先对应 `z_student` 或当前 differentiable student-generated clean video chunk；
- action DMD 应该优先对应 `a_student` 或当前 differentiable student-generated clean action chunk；
- `pred_u` 只能被视为一个 velocity-level debug surrogate。

#### Codex 任务

先不要强行重构成完整 on-policy differentiable rollout graph。请做一个清晰的两层实现：

1. 保留当前可跑的 `pred_u` 路径作为 debug surrogate；
2. 在 config 中显式加入：

```yaml
opd:
  dmd:
    surrogate_state: pred_u
```

3. 在 trainer 和 loss docstring 中明确：

```text
surrogate_state: pred_u 只用于当前服务器接口 debug，不等价于最终 clean-sample DMD objective。
最终算法目标应切换为 surrogate_state: clean_sample。
```

4. 如果实现成本不高，增加 `surrogate_state: clean_sample` 分支：

```python
if surrogate_state == "pred_u":
    student_state = pred_u
elif surrogate_state == "clean_sample":
    student_state = z_student or a_student
else:
    raise ValueError(...)
```

5. 对两个分支分别写 fake CPU unit tests，至少验证：

- `pred_u` 分支保持现有测试通过；
- `clean_sample` 分支能把 gradient 施加到传入的 clean tensor；
- 默认配置如果仍用 `pred_u`，必须在 config 注释里说明这是 debug surrogate。

验收标准：

```bash
python -m pytest \
  tests/test_dmd_losses.py \
  tests/test_video_dmd_trainer_static.py \
  tests/test_action_dmd_trainer_static.py
```

只允许 fake tensor 测试，不允许本地加载真实 LingBot。

---

### 问题 2：`opd.dmd.small_delta_s` 写进 config 但 trainer 没有真正使用

当前 config 中有：

```yaml
opd:
  dmd:
    student_score_source: flowmap_small_interval
    small_delta_s: 0.02
```

但 trainer 实际上直接复用 `sample_s_pairs(...)` 得到的同一个 `s_src/s_tgt` interval，然后把当前 interval 的 `pred_u` 当作 student score/velocity proxy。也就是说当前实现更像：

```text
student_score_source = current_interval_average_velocity
```

不是：

```text
student_score_source = flowmap_small_interval
```

#### Codex 任务

二选一，优先采用方案 A。

#### 方案 A：实现真正的 small-interval student score probe

在 `VideoDMDOPDTrainer` / `ActionDMDOPDTrainer` 中加入工具函数：

```python
build_small_interval_pair(x_state, s_probe, small_delta_s)
```

逻辑：

1. 采样一个 probe time `s_probe`；
2. 构造：

```text
s_small_src = s_probe
s_small_tgt = min(s_probe + small_delta_s, 1.0)
```

3. 如果 `s_small_tgt - s_small_src` 太小，则 clamp 或重新采样；
4. 调用 student flow-map forward 得到 `student_score`；
5. DMD correction 使用这个 small interval 的 student velocity proxy。

同时保证 teacher score/velocity 在同一个 `x_state` 和同一个 `s_probe` 上查询。

#### 方案 B：如果暂时不实现 small interval，则改名避免误导

把 config 改成：

```yaml
opd:
  dmd:
    student_score_source: current_interval_average_velocity
```

并删除或注释 `small_delta_s`。

#### 验收标准

- 如果采用方案 A，新增 tests 验证 `small_delta_s` 被读取且生成的小区间满足：

```text
0 <= s_small_src < s_small_tgt <= 1
s_small_tgt - s_small_src ~= small_delta_s 或被合理 clamp
```

- 如果采用方案 B，所有 config、docstring、README 不再声称当前 DMD 使用 small-interval student score。

---

### 问题 3：Action LingBot runtime binding 现在是推测性实现，需要服务器级 probe 和更清晰错误信息

当前 `LingBotVAAdapter.build_lingbot_action_input(...)` 返回：

```python
{
    "noisy_actions": a_src,
    "timesteps": timesteps,
    "context": context,
    "z_condition": z_condition,
}
```

然后 `install_action_flowmap_hook(...)` 用 `action_mode=True` 调真实 LingBot forward。

这在 Mac 本地 fake tests 中可以通过，但真实 LingBot-VA action path 很可能要求不同的 input contract，例如 `action_dict`、`latent_dict`、cache、grid、mask 或其他内部字段。

#### Codex 任务

1. 保留当前 action trainer / dataset / loss，不要把 LingBot 细节扩散到 trainer。
2. 强化 `scripts/tools/probe_action_ffm_runtime.py` 的 `probe-static-source`：
   - 读取 LingBot `wan_va/train.py`、`wan_va/modules/model.py`、server inference 相关文件；
   - 打印真实 action path 中出现的关键字段，如 `action_dict`、`latent_dict`、`action_mode`、`forward_train`、`update_cache`；
   - 输出 “WAM 当前 action input contract” 和 “LingBot 源码中观察到的 action input hints”。
3. 强化 `probe-action-forward`：
   - 如果真实 forward 失败，捕获异常并打印：
     - 当前 `input_dict.keys()`；
     - `a_src` shape；
     - `z_condition` shape；
     - model forward signature；
     - 失败函数位置建议：`LingBotVAAdapter.build_lingbot_action_input` / `install_action_flowmap_hook`。
4. 不要在 trainer 中 hard-code LingBot action internals。

#### 验收标准

服务器上管理员可运行：

```bash
MODE=05_action_forward sbatch scripts/slurm/action_ffm_admin_diagnostic.sbatch
```

如果失败，日志必须能明确说明是 action adapter contract mismatch，而不是模糊的 tensor shape / forward error。

---

### 问题 4：DMD teacher score/velocity binding 仍然是 `NotImplementedError`，需要变成服务器可诊断路径

当前：

```python
def teacher_video_score_or_velocity(...):
    raise NotImplementedError(...)

def teacher_action_score_or_velocity(...):
    raise NotImplementedError(...)
```

这符合上一轮 Mac 本地边界，但下一步要进入服务器级 debug，需要做到：

1. static/dry-run 不加载真实模型；
2. server probe 可以清晰触发并定位 teacher velocity binding；
3. 如果真实 LingBot velocity binding 还没写好，`05_dmd_forward` 必须清晰失败；
4. 最终 long training 前必须通过 teacher score/velocity probe。

#### Codex 任务

1. 给 `teacher_video_score_or_velocity` 和 `teacher_action_score_or_velocity` 增加两层模式：

```yaml
opd:
  dmd:
    teacher_score_source: not_implemented | lingbot_velocity_forward | cached_label
```

2. Mac 默认仍可以是 `not_implemented`，但 server long training 必须要求：

```yaml
teacher_score_source: lingbot_velocity_forward
```

或显式使用 `cached_label`。

3. 如果 `teacher_score_source=not_implemented` 且进入 non-dry-run DMD trainer，必须立刻报错，错误信息包含：

```text
DMD teacher score/velocity binding is not configured.
Use teacher_score_source=lingbot_velocity_forward after server probe succeeds,
or use teacher_score_source=cached_label with teacher labels in rollout manifest.
```

4. 实现 `cached_label` fallback：
   - video DMD 可以读取 `teacher_video_label`；
   - action DMD 可以读取 `teacher_action_label`；
   - shape 必须和 student/fake score shape 一致；
   - 这可以帮助管理员先跑 DMD trainer，不必立刻完成 real-time teacher forward。

5. `probe_dmd_opd_runtime.py probe-dmd-forward` 必须分别验证：
   - video teacher score shape；
   - action teacher score shape；
   - teacher score finite；
   - action DMD 使用 student rollout `z_condition`。

---

## 2. 统一管理员 Slurm 脚本目标

当前已经有分开的：

```text
scripts/slurm/action_ffm_admin_diagnostic.sbatch
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

但是为了让管理员操作接近 Video-stream FFM 的体验，建议新增一个统一脚本：

```text
scripts/slurm/v2_remaining_stages_admin_diagnostic.sbatch
```

这个脚本负责串起 action FFM、video DMD、action DMD 的诊断和 long training。

### 2.1 必要环境变量

脚本应要求或支持以下变量：

```bash
REPO_ROOT=/abs/path/to/WAM-Policy
TEACHER_CKPT=/abs/path/to/lingbot/transformer

ACTION_DEBUG_MANIFEST=/abs/path/to/action_chunks_debug.jsonl
ACTION_FULL_MANIFEST=/abs/path/to/action_chunks_full.jsonl

ROLLOUT_DEBUG_MANIFEST=/abs/path/to/student_rollouts_debug.jsonl
ROLLOUT_FULL_MANIFEST=/abs/path/to/student_rollouts_full.jsonl

WANDB_API_KEY=...
WANDB_PROJECT=WAM-Policy
WANDB_ENTITY=...
```

可选变量：

```bash
PYTHON_BIN=python3
ENV_SETUP=/path/to/env_setup.sh
TRAIN_MAX_STEPS_ACTION_FFM=5000
TRAIN_MAX_STEPS_VIDEO_DMD=5000
TRAIN_MAX_STEPS_ACTION_DMD=5000
TRAIN_BATCH_SIZE=1
TRAIN_SAVE_EVERY=500
TRAIN_LOG_EVERY=10
RUN_DMD_LONG=true
RUN_ACTION_FFM_LONG=true
```

### 2.2 推荐阶段顺序

统一脚本应按以下顺序执行。任何阶段失败都要停止，并在 status 文件中写入 first failing stage。

```text
00_env
01_static_all
02_import
03_load_model
04_action_manifest_debug
05_action_forward_debug
06_rollout_manifest_debug
07_video_dmd_forward_debug
08_action_dmd_forward_debug
09_action_ffm_train_smoke
10_video_dmd_train_smoke
11_action_dmd_train_smoke
12_action_ffm_train_long_full_manifest
13_video_dmd_train_long_full_manifest
14_action_dmd_train_long_full_manifest
15_collect_artifacts
```

### 2.3 各阶段职责

#### 00_env

- 打印 hostname、date、SLURM_JOB_ID、ROCm 信息；
- 打印 PyTorch version、HIP/CUDA 状态；
- 检查 `REPO_ROOT`、`TEACHER_CKPT`、manifest 文件是否存在；
- 隐去 token/API key。

#### 01_static_all

运行：

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static

python scripts/tools/probe_action_ffm_runtime.py probe-static-source --config configs/experiments/v2/anyflow_action_ffm.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config configs/experiments/v2/anyflow_video_dmd_opd.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config configs/experiments/v2/anyflow_action_dmd_opd.yaml

python -m py_compile scripts/train.py scripts/tools/probe_action_ffm_runtime.py scripts/tools/probe_dmd_opd_runtime.py
```

#### 02_import

- bootstrap external repos；
- import `wan_va`；
- import WAM modules；
- 不加载 checkpoint。

#### 03_load_model

- 加载 LingBot teacher checkpoint；
- 打印 model class、forward signature、patch size、condition embedder metadata；
- 不训练。

#### 04_action_manifest_debug

- 读取 `ACTION_DEBUG_MANIFEST` 第一条；
- 打印 `a_clean`、`context`、`z_condition` shape；
- 不加载模型。

#### 05_action_forward_debug

- 加载模型；
- attach action two-time conditioner；
- 构造 action flow-map pair；
- 调 `adapter.forward_action_flowmap(...)`；
- 检查 `pred_u.shape == a_src.shape` 和 finite。

#### 06_rollout_manifest_debug

- 读取 `ROLLOUT_DEBUG_MANIFEST` 第一条；
- 检查 video DMD 需要 `z_student`；
- 检查 action DMD 需要 `a_student` 和 `z_condition`；
- 如果 `teacher_score_source=cached_label`，还要检查 teacher label 是否存在。

#### 07_video_dmd_forward_debug

- 加载模型；
- attach video two-time conditioner；
- 构造 video DMD flow-map pair；
- 查询 student score proxy；
- 查询 teacher score/velocity 或 cached label；
- 检查 shape/finite。

#### 08_action_dmd_forward_debug

- 加载模型；
- attach action two-time conditioner；
- 使用 student rollout `z_condition`；
- 查询 action teacher score/velocity 或 cached label；
- 检查 shape/finite。

#### 09–11 smoke train

每个阶段用 debug manifest 跑 2 step：

```bash
train.max_steps=2
train.save_every=1
train.log_every=1
```

目的是验证 trainer、optimizer、checkpoint、W&B logging 不崩。

#### 12–14 long train

诊断全部通过后才运行 full manifest long training：

- `12_action_ffm_train_long_full_manifest` 使用 `ACTION_FULL_MANIFEST`；
- `13_video_dmd_train_long_full_manifest` 使用 `ROLLOUT_FULL_MANIFEST`；
- `14_action_dmd_train_long_full_manifest` 使用 `ROLLOUT_FULL_MANIFEST`。

每个阶段写到独立 output/checkpoint dir：

```text
experiments/runs/v2/action_ffm_admin/<JOB_ID>/
experiments/runs/v2/video_dmd_admin/<JOB_ID>/
experiments/runs/v2/action_dmd_admin/<JOB_ID>/

experiments/checkpoints/v2/action_ffm_admin/<JOB_ID>/
experiments/checkpoints/v2/video_dmd_admin/<JOB_ID>/
experiments/checkpoints/v2/action_dmd_admin/<JOB_ID>/
```

#### 15_collect_artifacts

- 写 `status.txt`；
- 列出 checkpoint 文件名，但不要上传 checkpoint tensor；
- 如果 W&B enabled，上传 logs artifact；
- 不上传 manifest、dataset、checkpoint 本体。

---

## 3. README 需要补充的管理员说明

README 应新增一个章节：

```text
Run WAM-Policy v2 Remaining Stages: Action FFM + Video/Action DMD OPD
```

内容包括：

1. 准备 external repos；
2. 准备 teacher checkpoint；
3. 准备 action debug/full manifest；
4. 准备 student rollout debug/full manifest；
5. 设置 W&B；
6. 提交统一 sbatch。

示例命令：

```bash
export REPO_ROOT="$PWD"
export TEACHER_CKPT="$PWD/experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/transformer"

export ACTION_DEBUG_MANIFEST="$PWD/datasets/manifests/action_chunks_debug.jsonl"
export ACTION_FULL_MANIFEST="$PWD/datasets/manifests/action_chunks_full.jsonl"

export ROLLOUT_DEBUG_MANIFEST="$PWD/datasets/manifests/student_rollouts_debug.jsonl"
export ROLLOUT_FULL_MANIFEST="$PWD/datasets/manifests/student_rollouts_full.jsonl"

export WANDB_PROJECT="WAM-Policy"
export WANDB_UPLOAD_LOGS=true

sbatch --export=ALL scripts/slurm/v2_remaining_stages_admin_diagnostic.sbatch
```

还要明确：

- Action FFM 可以只依赖 action manifest；
- Video/Action DMD 依赖 student rollout manifest；
- 如果没有 rollout manifest，不能直接跑 DMD long training；
- 当前 RoboTwin online rollout generation 如果还未实现，需要管理员先提供 rollout manifest，或者后续单独实现 RoboTwin rollout collection stage。

---

## 4. 本轮不要做的事

不要做：

- 不要在 Mac 本地下载 checkpoint / full dataset / manifest；
- 不要在 Mac 本地跑 LingBot model loading；
- 不要在 Mac 本地跑 RoboTwin；
- 不要把 LingBot internals 写进 trainer；
- 不要把 checkpoint、manifest、logs、wandb、datasets 加入 git；
- 不要把当前 DMD debug surrogate 说成 final DMD 算法实现；
- 不要把 action DMD 的 `z_condition` 改成 teacher video condition。

---

## 5. Codex 本地验收命令

Mac 本地只需要跑轻量测试：

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static

python scripts/tools/probe_action_ffm_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_ffm.yaml

python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_video_dmd_opd.yaml

python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_dmd_opd.yaml

python -m pytest \
  tests/test_pipeline_static_dispatch.py \
  tests/test_action_flowmap_pair.py \
  tests/test_action_ffm_trainer_static.py \
  tests/test_dmd_losses.py \
  tests/test_video_dmd_trainer_static.py \
  tests/test_action_dmd_trainer_static.py \
  tests/test_probe_scripts_static.py
```

如果这些通过，说明 Mac 本地接口和 fake tests 通过。真实 action forward、teacher score/velocity、ROCm、checkpoint loading、full manifest long training 必须交给服务器 sbatch 验证。

---

## 6. 服务器验收目标

最终管理员只需要做：

```bash
git clone <repo>
cd WAM-Policy
# 按 README 准备 external repos / checkpoint / manifests / env vars
sbatch --export=ALL scripts/slurm/v2_remaining_stages_admin_diagnostic.sbatch
```

成功标准：

1. `00_env` 到 `11_action_dmd_train_smoke` 全部通过；
2. action forward binding 在真实 LingBot 上可运行，或失败信息明确指向 adapter contract；
3. DMD teacher score/velocity binding 可运行，或使用 cached label fallback；
4. 诊断全部通过后自动进入 full manifest long training；
5. 每个 long training 阶段都有独立 checkpoint dir；
6. W&B 有 metrics；
7. logs artifact 不包含 checkpoint tensor 本体；
8. `status.txt` 明确记录 SUCCESS 或 first failing stage。

---

## 7. 推荐 commit 切分

### Commit 1：DMD surrogate 语义修正

- 增加 `opd.dmd.surrogate_state`；
- 明确 `pred_u` vs `clean_sample`；
- 更新 tests。

### Commit 2：small_delta_s 或 config 命名修复

- 实现 small-interval student score probe，或把 config 改成 current-interval；
- 更新 docs/tests。

### Commit 3：Action runtime probe 强化

- 强化 `probe_action_ffm_runtime.py`；
- 增加清晰错误信息；
- 不改 trainer。

### Commit 4：DMD teacher score/velocity source 配置化

- 增加 `teacher_score_source`；
- 实现 `cached_label` fallback；
- `lingbot_velocity_forward` 先接 adapter/probe；
- 保持 Mac static 可过。

### Commit 5：统一管理员 sbatch

- 新增 `scripts/slurm/v2_remaining_stages_admin_diagnostic.sbatch`；
- 串起 action FFM + video DMD + action DMD；
- 输出 status/logs/checkpoint listing；
- 支持 W&B logs artifact。

### Commit 6：README 管理员指南

- 增加剩余三阶段的 setup 和提交说明；
- 明确必需 manifest；
- 明确 DMD 依赖 rollout manifest。

