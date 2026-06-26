# WAM-Policy_v2: Dual-Head AnyFlow OPD for LingBot-VA

> 设计主题：以 **LingBot-VA** 作为 frozen teacher model，使用 **AnyFlow-style On-Policy Flow-Map Distillation**，将 LingBot-VA 的 video head 与 action head 从多步 flow-matching 采样蒸馏为少步 flow-map 采样。  
> 目标任务背景：RoboTwin post-trained LingBot-VA，例如 `adjust_bottle` 任务。  
> 目标模型定位：保留 LingBot-VA 的 video-action causal world model 结构，同时显著减少 video/action 两个生成过程的 sampling steps。

---

## 1. 核心结论

WAM-Policy_v2 的目标不是简单地复制一个 LingBot-VA checkpoint，然后把：

\[
N_v = 25 \rightarrow 4/8
\]

\[
N_a = 50 \rightarrow 8/16
\]

这种做法只是 **naive step reduction**：原模型仍然是 instantaneous velocity-field model，只是用更粗的 Euler step 采样，因此会产生较大的 discretization error。

WAM-Policy_v2 的真正目标是：

\[
\boxed{
\text{把 LingBot-VA 的 video/action 两个 flow-matching sampling operators 蒸馏成 two-time flow-map operators。}
}
\]

也就是说，student 不再只是原始 LingBot-VA 的低步数 sampler，而是一个 **LingBot-VA-compatible flow-map student**。

它保留 LingBot-VA 的 video-action causal interface、MoT / shared-backbone 结构、KV cache 机制和 chunk-level interleaving，但 video head 与 action head 的输出语义从：

\[
v(x_\tau,\tau)
\]

变成：

\[
F(x_\tau,\tau,\rho) \approx x_\rho
\]

或等价的 average velocity：

\[
u(x_\tau,\tau,\rho)
\approx
\frac{x_\tau-x_\rho}{\tau-\rho}
\]

其中：

\[
1 \geq \tau > \rho \geq 0
\]

---

## 2. Teacher 与 Student 的关系

### 2.1 Teacher model

Teacher 是原始 RoboTwin post-trained LingBot-VA。

它包含两个 iterative flow-matching generation processes：

1. **Video head / video mode**

   \[
   v_T^v(Z_\tau,\tau \mid C_k)
   \]

   原始 RoboTwin 配置中通常使用：

   \[
   N_v^T = 25
   \]

2. **Action head / action mode**

   \[
   v_T^a(A_\tau,\tau \mid C_k,\hat Z_k)
   \]

   原始 RoboTwin 配置中通常使用：

   \[
   N_a^T = 50
   \]

其中：

- \(Z_k\)：第 \(k\) 个 video chunk latent；
- \(A_k\)：第 \(k\) 个 action chunk；
- \(C_k\)：第 \(k\) 个 chunk 开始时的 causal context，包括 text embedding、真实 video/action history、KV cache 等；
- \(\hat Z_k\)：当前 video head 预测出的 future visual transition，是 action head 做 inverse dynamics 的关键条件。

Teacher 在训练中保持 frozen，主要作用是：

1. 提供高质量 25-step video chunk samples；
2. 提供高质量 50-step action chunk samples；
3. 在 DMD correction 中提供 teacher distribution / teacher score / teacher velocity signal；
4. 作为 on-policy distillation 的 reference model。

---

### 2.2 Student model

Student 从 teacher checkpoint 初始化：

\[
\theta_S \leftarrow \theta_T
\]

\[
\psi_S \leftarrow \psi_T
\]

但 student 不是简单地调小 sampling steps。

Student 要被训练成 **dual-head flow-map model**：

1. **Video flow-map head**

   \[
   F_S^v(Z_\tau,\tau,\rho \mid C_k)
   \approx
   Z_\rho
   \]

   或：

   \[
   u_S^v(Z_\tau,\tau,\rho \mid C_k)
   \approx
   \frac{Z_\tau-Z_\rho}{\tau-\rho}
   \]

2. **Action flow-map head**

   \[
   F_S^a(A_\tau,\tau,\rho \mid C_k,\hat Z_k)
   \approx
   A_\rho
   \]

   或：

   \[
   u_S^a(A_\tau,\tau,\rho \mid C_k,\hat Z_k)
   \approx
   \frac{A_\tau-A_\rho}{\tau-\rho}
   \]

因此，student 和 teacher 的关系应该表述为：

\[
\boxed{
\text{architecture-level near-isomorphic, operator-level non-isomorphic}
}
\]

也就是：

| 层面 | Teacher LingBot-VA | Student LingBot-AnyFlow |
|---|---|---|
| video/action token shape | 相同 | 相同 |
| VAE / text encoder | 相同 | 相同 |
| KV cache 机制 | 相同 | 相同 |
| video-action causal order | 相同 | 相同 |
| video head 输出语义 | instantaneous velocity | flow-map transition / average velocity |
| action head 输出语义 | instantaneous velocity | flow-map transition / average velocity |
| 时间条件 | \(\tau\) | \((\tau,\rho)\) 或 \((\tau,\Delta\tau)\) |
| 采样方式 | many-step Euler integration | few-step flow-map jumps |
| video sampling steps | 25 | 例如 4 / 8 |
| action sampling steps | 50 | 例如 8 / 16 |

---

## 3. 为什么不能只 copy LingBot-VA 然后调小 sampling steps？

原始 LingBot-VA 的 video/action heads 学的是 instantaneous velocity field：

\[
v(x_\tau,\tau)
=\frac{dx_\tau}{d\tau}
\]

采样时依赖多步 Euler-style integration：

\[
x_{\tau_{i+1}}
=
x_{\tau_i}
+
v(x_{\tau_i},\tau_i)\Delta\sigma_i
\]

如果直接把 steps 调小，例如：

\[
25\to4,
\quad
50\to8
\]

本质上只是把 Euler step size 变大。模型并没有学过：

\[
x_\tau \to x_\rho
\]

这种大跨度 transition。

所以它可能出现：

1. video chunk 运动不准；
2. video latent 变糊或语义漂移；
3. action head 因为依赖 predicted video transition 而进一步失稳；
4. long-horizon rollout 中误差通过 cache 和环境反馈累积；
5. task success rate 明显下降。

因此，直接调小 sampling steps 可以作为 baseline，但不是 AnyFlow-style OPD。

---

## 4. LingBot-VA 原始执行过程中的 AnyFlow 插入点

以 RoboTwin post-trained LingBot-VA 为例，一轮执行大致是：

\[
o_k
\rightarrow
z_k=E_{\text{VAE}}(o_k)
\rightarrow
\hat Z_k
\rightarrow
\hat A_k
\rightarrow
\text{execute}
\rightarrow
o_{k+1}
\]

更具体地说：

1. 真实 observation 被 VAE encode 成 video latent；
2. video head 预测一个短 horizon 的 future video chunk；
3. predicted video chunk 写入 predicted video cache；
4. action head 基于 predicted video transition 做 inverse dynamics decoding；
5. action chunk 发送给 RoboTwin 环境执行；
6. 环境返回新 observation 和 robot state；
7. predicted cache 被清除；
8. 真实 observation/action 被写入 persistent real cache；
9. 下一轮基于 real cache 继续。

AnyFlow 应该插入到两个 iterative generation loops 中：

1. **video chunk denoising loop**  
   从原始 25-step velocity integration 改为 few-step video flow-map transition。

2. **action chunk denoising loop**  
   从原始 50-step velocity integration 改为 few-step action flow-map transition。

不建议改动：

- VAE encode / decode；
- text embedding；
- video/action token shape；
- predicted cache 写入逻辑；
- real cache update 逻辑；
- RoboTwin closed-loop execution 逻辑。

因为 LingBot-VA 的 closed-loop real cache correction 正好可以为 OPD 提供 on-policy contexts。

---

## 5. WAM-Policy_v2 的总体训练框架

WAM-Policy_v2 建议分为三个阶段：

\[
\boxed{
\text{Stage 0: Student initialization}
}
\]

\[
\boxed{
\text{Stage 1: Dual-head Forward Flow-Map Training}
}
\]

\[
\boxed{
\text{Stage 2: Dual-head On-Policy Flow-Map Distillation with DMD Correction}
}
\]

部署阶段只使用 trained student，不再使用 teacher，也不再跑 DMD。

---

# 6. Stage 0: Student Initialization

Student 从 teacher checkpoint 初始化。

保留：

- video/action MoT 或 shared-backbone architecture；
- video/action embedding；
- video/action output projection；
- text cross-attention；
- causal attention mask；
- KV cache；
- chunk-level video-action interleaving；
- RoboTwin action format。

需要新增或调整：

1. video head 接收 two-time condition：

   \[
   (\tau,\rho)
   \]

2. action head 接收 two-time condition：

   \[
   (\tau,\rho)
   \]

3. timestep embedding 从单时间条件变成双时间条件，例如：

   \[
   e(\tau,\rho)
   =
   e(\tau)+e(\rho)+e(\tau-\rho)
   \]

或者使用：

\[
(\tau,\Delta\tau)
\]

其中：

\[
\Delta\tau=\tau-\rho
\]

推荐第一版使用 average velocity parameterization，因为它与原始 flow-matching velocity prediction 更接近。

---

# 7. Stage 1: Dual-head Forward Flow-Map Training

Stage 1 的目标是把原始 LingBot-VA student cold-start 成一个可以做 arbitrary-time transition 的 flow-map model。

这一阶段不是 on-policy，而是用 teacher samples、demo samples 或 teacher rollout samples 做 supervised flow-map transition learning。

---

## 7.1 Video head forward flow-map training

给定一个 clean video chunk target：

\[
Z_k^*
\]

它可以来自：

1. RoboTwin demonstration 的真实未来 observation，经 VAE encode；
2. teacher LingBot-VA 25-step 生成的 video chunk；
3. teacher rollout 后环境返回的真实 observation chunk。

对于每个 target chunk，采样噪声：

\[
\epsilon^v\sim\mathcal{N}(0,I)
\]

采样两个 flow time：

\[
1\geq \tau > \rho \geq 0
\]

构造：

\[
Z_\tau
=
(1-\tau)Z_k^*+\tau\epsilon^v
\]

\[
Z_\rho
=
(1-\rho)Z_k^*+\rho\epsilon^v
\]

训练 video flow-map：

\[
F_S^v(Z_\tau,\tau,\rho\mid C_k)
\approx
Z_\rho
\]

如果使用 endpoint loss：

\[
\mathcal{L}_{v,\text{ffm}}
=
\left\|
F_S^v(Z_\tau,\tau,\rho\mid C_k)-Z_\rho
\right\|^2
\]

如果使用 average velocity loss：

\[
\mathcal{L}_{v,\text{ffm}}
=
\left\|
u_S^v(Z_\tau,\tau,\rho\mid C_k)
-
\frac{Z_\tau-Z_\rho}{\tau-\rho}
\right\|^2
\]

推荐第一版使用 average velocity loss。

---

## 7.2 Action head forward flow-map training

给定一个 clean action chunk target：

\[
A_k^*
\]

它可以来自：

1. RoboTwin demonstration executed action chunk；
2. teacher LingBot-VA 50-step 生成的 action chunk；
3. teacher rollout 中实际执行的 action chunk。

采样噪声：

\[
\epsilon^a\sim\mathcal{N}(0,I)
\]

采样两个 flow time：

\[
1\geq \tau > \rho \geq 0
\]

构造：

\[
A_\tau
=
(1-\tau)A_k^*+\tau\epsilon^a
\]

\[
A_\rho
=
(1-\rho)A_k^*+\rho\epsilon^a
\]

训练 action flow-map：

\[
F_S^a(A_\tau,\tau,\rho\mid C_k,\hat Z_k)
\approx
A_\rho
\]

或：

\[
\mathcal{L}_{a,\text{ffm}}
=
\left\|
u_S^a(A_\tau,\tau,\rho\mid C_k,
\hat Z_k)
-
\frac{A_\tau-A_\rho}{\tau-\rho}
\right\|^2
\]

Stage 1 早期建议让 action head 条件在 teacher-quality video chunk 上：

\[
\hat Z_k=\hat Z_k^T
\]

等 video flow-map student 稳定后，再逐渐切换到：

\[
\hat Z_k=\hat Z_k^S
\]

原因是 action head 的输入分布强依赖 predicted video transition。若一开始就使用 unstable student video，会同时引入 video distribution drift 和 action distribution drift。

---

## 7.3 Stage 1 结束后的 student 能力

Stage 1 结束后，student 已经可以用 few-step flow-map 采样：

Video：

\[
Z_1
\rightarrow
Z_{3/4}
\rightarrow
Z_{1/2}
\rightarrow
Z_{1/4}
\rightarrow
Z_0
\]

Action：

\[
A_1
\rightarrow
A_{7/8}
\rightarrow
A_{6/8}
\rightarrow
\cdots
\rightarrow
A_0
\]

但是它还没有完全解决 on-policy rollout distribution mismatch。

因此需要 Stage 2。

---

# 8. Stage 2: Dual-head On-Policy Flow-Map Distillation with DMD Correction

Stage 2 的目标是让 student 在自己真实执行后产生的 context 上被 teacher 纠偏。

OPD 的 on-policy context 来自 student 在 RoboTwin 中的 closed-loop rollout：

\[
C_0^S
\rightarrow
\hat Z_0^S
\rightarrow
\hat A_0^S
\rightarrow
o_1^S
\rightarrow
C_1^S
\rightarrow
\hat Z_1^S
\rightarrow
\hat A_1^S
\rightarrow
o_2^S
\]

其中每个：

\[
C_k^S
\]

都是 student 自己诱导出来的 real history cache，包括：

- 真实 observation latent；
- executed action history；
- prompt embedding；
- causal KV cache。

这一步是 WAM-Policy_v2 与普通 offline distillation 的关键区别。

---

## 8.1 Video flow-map backward simulation

给定一个 on-policy context：

\[
C_k^S
\]

假设目标 video sampling budget 是：

\[
N_v^S=4
\]

则每一步间隔：

\[
\Delta_v=\frac{1}{4}
\]

随机选择一个 transition：

\[
\tau>\rho,
\quad
\tau-\rho=\Delta_v
\]

例如：

\[
\tau=\frac{3}{4},
\quad
\rho=\frac{1}{2}
\]

flow-map backward simulation 将完整生成轨迹分解成三段：

\[
Z_1
\xrightarrow{F_S^v(1,\tau)}
Z_\tau^S
\]

\[
Z_\tau^S
\xrightarrow{F_S^v(\tau,\rho)}
Z_\rho^S
\]

\[
Z_\rho^S
\xrightarrow{F_S^v(\rho,0)}
\hat Z_{k,0}^S
\]

其中：

- \(F_S^v(1,\tau)\)：shortcut transition；
- \(F_S^v(\tau,\rho)\)：当前训练的 local transition；
- \(F_S^v(\rho,0)\)：shortcut transition；
- \(\hat Z_{k,0}^S\)：student 在当前 on-policy context 下生成的 clean video chunk。

这样不需要完整跑 \(N_v^S\) 步，也能模拟某个 student few-step transition 对最终 sample 的影响。

---

## 8.2 Video DMD correction

得到 student clean video chunk：

\[
\hat Z_{k,0}^S
\]

重新加噪到随机时间 \(\alpha\)：

\[
\tilde Z_{k,\alpha}^S
=
(1-\alpha)\hat Z_{k,0}^S+
\alpha\epsilon
\]

然后分别用 teacher 与 student 在同一个 context 下评估 score / velocity signal：

\[
s_T^v
=
s_T^v(\tilde Z_{k,\alpha}^S,
\alpha
\mid C_k^S)
\]

\[
s_S^v
=
s_S^v(\tilde Z_{k,\alpha}^S,
\alpha
\mid C_k^S)
\]

DMD correction 的方向是：

\[
s_T^v-s_S^v
\]

它反传到整个 video flow-map backward simulation chain：

\[
F_S^v(1,\tau),
\quad
F_S^v(\tau,\rho),
\quad
F_S^v(\rho,0)
\]

概念上，这一步在做：

\[
\boxed{
\text{让 student 在自己 rollout 到的 real context 下生成的 video chunk 分布接近 teacher video distribution。}
}
\]

---

## 8.3 Action flow-map backward simulation

Action OPD 必须条件在 video prediction 上。

给定同一个 on-policy context：

\[
C_k^S
\]

先用 student video flow-map 得到：

\[
\hat Z_k^S
\]

将 action condition 定义为：

\[
D_k^S
=
\{C_k^S,\hat Z_k^S\}
\]

假设目标 action sampling budget 是：

\[
N_a^S=8
\]

则：

\[
\Delta_a=\frac{1}{8}
\]

随机选择：

\[
\tau>\rho,
\quad
\tau-\rho=\Delta_a
\]

例如：

\[
\tau=\frac{5}{8},
\quad
\rho=\frac{4}{8}
\]

做 action flow-map backward simulation：

\[
A_1
\xrightarrow{F_S^a(1,\tau)}
A_\tau^S
\]

\[
A_\tau^S
\xrightarrow{F_S^a(\tau,\rho)}
A_\rho^S
\]

\[
A_\rho^S
\xrightarrow{F_S^a(\rho,0)}
\hat A_{k,0}^S
\]

所有 transition 都条件在：

\[
D_k^S=\{C_k^S,\hat Z_k^S\}
\]

---

## 8.4 Action DMD correction

得到 student action chunk：

\[
\hat A_{k,0}^S
\]

重新加噪：

\[
\tilde A_{k,\alpha}^S
=
(1-\alpha)\hat A_{k,0}^S+
\alpha\epsilon^a
\]

teacher 与 student 评估 action score / velocity signal：

\[
s_T^a
=
s_T^a(
\tilde A_{k,\alpha}^S,
\alpha
\mid C_k^S,
\hat Z_k^S
)
\]

\[
s_S^a
=
s_S^a(
\tilde A_{k,\alpha}^S,
\alpha
\mid C_k^S,
\hat Z_k^S
)
\]

DMD correction direction：

\[
s_T^a-s_S^a
\]

并反传到：

\[
F_S^a(1,\tau),
\quad
F_S^a(\tau,\rho),
\quad
F_S^a(\rho,0)
\]

这里推荐 teacher action score 也条件在 student video chunk：

\[
\hat Z_k^S
\]

而不是 teacher video chunk：

\[
\hat Z_k^T
\]

原因是部署时 action head 真实看到的是 student few-step video imagination。如果训练时 action teacher 总是基于 teacher video chunk 提供监督，会造成新的 condition mismatch。

---

## 8.5 Stage 2 的总目标

Stage 2 通常不应该只做 DMD，也应该继续保留 Stage 1 的 forward flow-map training 作为 anchor。

总目标可以写成：

\[
\mathcal{L}
=
\lambda_v\mathcal{L}_{v,\text{ffm}}
+
\lambda_a\mathcal{L}_{a,\text{ffm}}
+
\beta_v\mathcal{L}_{v,\text{DMD}}
+
\beta_a\mathcal{L}_{a,\text{DMD}}
+
\eta\mathcal{L}_{\text{anchor}}
\]

其中：

- \(\mathcal{L}_{v,\text{ffm}}\)：video forward flow-map supervised loss；
- \(\mathcal{L}_{a,\text{ffm}}\)：action forward flow-map supervised loss；
- \(\mathcal{L}_{v,\text{DMD}}\)：video on-policy distribution correction；
- \(\mathcal{L}_{a,\text{DMD}}\)：action on-policy distribution correction；
- \(\mathcal{L}_{\text{anchor}}\)：防止 student 早期训练偏离 teacher / demo manifold 的稳定项。

可以使用轻量 anchor：

\[
\mathcal{L}_{v,\text{anchor}}
=
\left\|
\hat Z_k^S-
\text{sg}(\hat Z_k^T)
\right\|^2
\]

\[
\mathcal{L}_{a,\text{anchor}}
=
\left\|
\hat A_k^S-
\text{sg}(\hat A_k^T)
\right\|^2
\]

其中 \(\text{sg}\) 表示 stop-gradient。

---

# 9. 一次完整 training iteration 的抽象流程

## Step A: Student rollout

用当前 student 在 RoboTwin 中执行：

\[
C_0^S
\rightarrow
\hat Z_0^S
\rightarrow
\hat A_0^S
\rightarrow
o_1^S
\]

\[
C_1^S
\rightarrow
\hat Z_1^S
\rightarrow
\hat A_1^S
\rightarrow
o_2^S
\]

收集一批 on-policy contexts：

\[
\{C_k^S\}_{k=0}^{K}
\]

---

## Step B: Video OPD update

从 buffer 中采样 \(C_k^S\)。

做 video flow-map backward simulation：

\[
Z_1
\to
Z_\tau^S
\to
Z_\rho^S
\to
\hat Z_{k,0}^S
\]

计算 video DMD correction：

\[
s_T^v-s_S^v
\]

更新 video flow-map parameters。

---

## Step C: Action OPD update

用 student video flow-map 得到：

\[
\hat Z_k^S
\]

构造 action condition：

\[
D_k^S=\{C_k^S,\hat Z_k^S\}
\]

做 action flow-map backward simulation：

\[
A_1
\to
A_\tau^S
\to
A_\rho^S
\to
\hat A_{k,0}^S
\]

计算 action DMD correction：

\[
s_T^a-s_S^a
\]

更新 action flow-map parameters。

---

## Step D: Forward flow-map regularization

同时从 teacher/demo data 中采样 clean video/action chunks，继续优化：

\[
\mathcal{L}_{v,\text{ffm}}
+
\mathcal{L}_{a,\text{ffm}}
\]

避免 student 只跟随 DMD signal 造成训练不稳定。

---

# 10. 训练完成后的推理流程

部署时不再使用 teacher，也不再运行 DMD correction。

第 \(k\) 个 chunk：

1. 从环境获得真实 observation：

   \[
   o_k
   \]

2. VAE encode：

   \[
   z_k=E_{\text{VAE}}(o_k)
   \]

3. 基于 real history cache 构造 context：

   \[
   C_k^S
   \]

4. video few-step flow-map sampling：

   \[
   Z_1
   \rightarrow
   Z_{\tau_1}
   \rightarrow
   \cdots
   \rightarrow
   \hat Z_k^S
   \]

   例如：

   \[
   N_v^S=4
   \]

5. 将 \(\hat Z_k^S\) 写入 predicted video cache；

6. action few-step flow-map sampling：

   \[
   A_1
   \rightarrow
   A_{\tau_1}
   \rightarrow
   \cdots
   \rightarrow
   \hat A_k^S
   \]

   例如：

   \[
   N_a^S=8
   \]

7. 执行 action chunk：

   \[
   \hat A_k^{S,exec}
   \]

8. 环境返回：

   \[
   o_{k+1}
   \]

9. 清除 predicted cache，写入真实 observation/action cache；

10. 进入下一轮。

最终推理结构保持 LingBot-VA 的 closed-loop world-action model 形式：

\[
\boxed{
\text{real observation}
\rightarrow
\text{few-step video flow-map}
\rightarrow
\text{few-step action flow-map}
\rightarrow
\text{execute}
\rightarrow
\text{real cache update}
}
\]

---

# 11. 与原始 AnyFlow 的对应关系

| AnyFlow 原始 video diffusion | WAM-Policy_v2 / LingBot-VA |
|---|---|
| 单一 video latent trajectory | video chunk trajectory + action chunk trajectory |
| condition \(c\) | text + real video/action history cache |
| flow-map \(f_\theta(z_t,t,r)\) | video flow-map \(F_S^v(Z_\tau,\tau,\rho\mid C_k)\) |
| flow-map \(f_\theta(z_t,t,r)\) | action flow-map \(F_S^a(A_\tau,\tau,\rho\mid C_k,\hat Z_k)\) |
| forward flow-map training | dual-head forward flow-map training |
| flow-map backward simulation | video/action 分别做 backward simulation |
| DMD correction | video DMD + action DMD |
| model rollout | RoboTwin closed-loop student rollout |
| exposure bias | student executed trajectory 诱导的 cache distribution shift |

---

# 12. 推荐的实验路线

## Experiment 0: Naive step reduction baseline

直接 copy LingBot-VA，只改 sampling steps：

\[
N_v=25\to4/8
\]

\[
N_a=50\to8/16
\]

目的：建立粗步长 velocity sampler 的 baseline。

---

## Experiment 1: Video-only flow-map cold start

固定 action head 仍使用 teacher / 原始 50-step。

只训练 video flow-map：

\[
N_v^S=8
\]

观察：

- predicted video latent quality；
- video transition 是否 task-relevant；
- action head 在 student video condition 下是否还能正常工作。

---

## Experiment 2: Action-only flow-map cold start

固定 video head 使用 teacher 25-step predicted video。

只训练 action flow-map：

\[
N_a^S=16
\]

观察：

- action chunk MSE；
- gripper timing；
- EEF trajectory smoothness；
- RoboTwin task success rate。

---

## Experiment 3: Dual-head forward flow-map

同时训练：

\[
N_v^S=8
\]

\[
N_a^S=16
\]

先不加 DMD，只验证 dual-head flow-map cold start 是否稳定。

---

## Experiment 4: Dual-head on-policy DMD

加入 RoboTwin student rollout：

\[
C_k^S
\rightarrow
\hat Z_k^S
\rightarrow
\hat A_k^S
\rightarrow
o_{k+1}^S
\]

然后做：

\[
\mathcal{L}_{v,\text{DMD}}
+
\mathcal{L}_{a,\text{DMD}}
\]

---

## Experiment 5: Step compression

如果 \(8/16\) 稳定，再尝试：

\[
N_v^S=4
\]

\[
N_a^S=8
\]

进一步再考虑：

\[
N_v^S=2
\]

\[
N_a^S=4
\]

---

# 13. 风险与关键问题

## 13.1 Video-action condition mismatch

Action head 的输入依赖 predicted video chunk。

如果 action OPD 只在 teacher video condition 上训练，部署时看到 student video condition，仍然会失配。

因此 Stage 2 中 action DMD 应尽量条件在：

\[
\hat Z_k^S
\]

---

## 13.2 Student video drift 会放大 action drift

Video flow-map student 质量不足时，action head 的 inverse dynamics 会被错误 visual transition 误导。

因此建议训练顺序是：

1. video-only；
2. action-only with teacher video；
3. dual-head forward flow-map；
4. dual-head on-policy DMD。

---

## 13.3 DMD signal 不应完全替代 supervised flow-map loss

DMD 是 distribution correction，不是稳定的 point-wise supervised target。

Stage 2 中建议保留：

\[
\mathcal{L}_{v,\text{ffm}}
+
\mathcal{L}_{a,\text{ffm}}
\]

作为 anchor。

---

## 13.4 Teacher 不需要变成 flow-map model

Teacher 保持原始 LingBot-VA flow-matching model 即可。

Student 才是 flow-map model。

---

## 13.5 不是每个 chunk 都临时跑 AnyFlow 两阶段

AnyFlow 的 Stage 1 和 Stage 2 是训练阶段，而不是每次 inference 的临时过程。

部署时只使用训练好的 student：

\[
\text{few-step video flow-map}
+
\text{few-step action flow-map}
\]

---

# 14. 最终定位

WAM-Policy_v2 可以定义为：

\[
\boxed{
\text{A dual-head flow-map on-policy distilled version of LingBot-VA.}
}
\]

它保留 LingBot-VA 的核心世界模型结构：

\[
\text{video imagination}
\rightarrow
\text{inverse dynamics action decoding}
\rightarrow
\text{closed-loop real cache correction}
\]

同时把原始高成本采样：

\[
25 \text{ video steps} + 50 \text{ action steps}
\]

压缩为例如：

\[
4 \text{ video flow-map steps} + 8 \text{ action flow-map steps}
\]

因此，WAM-Policy_v2 不是 action-only VLA，也不是简单 low-step LingBot-VA，而是：

\[
\boxed{
\text{LingBot-VA-compatible dual-head flow-map policy distilled by AnyFlow-style OPD.}
}
\]
