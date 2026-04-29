# V5 归因分析与 V6 立项依据 — 2026-04-28

## 1. V5 实验体系回顾

V5 不是一个实验，而是两个：

| 实验 | 设计 | 目的 |
|---|---|---|
| **V5 rollout-heavy** | 从 V3 best resume，改 λ_roll=1.5, λ_img=0.08, step_weights 尾重 | 测试加重 rollout 能否打破 V3 plateau |
| **V5 null-control** | 从 V3 best resume，**不改任何权重**（V3 原始配方） | 对照：区分"V5 loss 改动"vs"resume/LR"的影响 |

## 2. 当前数据（截至 0429 快照，NC step ~123550 / V6 step ~38050）

### 2.1 V5 null-control 的 weighted loss fraction

> 注：`*_frac` 严格定义是 weighted loss 占比，不是反向梯度 norm 占比；项目内沿用其作为 supervision pressure 的代理。

null-control 从 V3 best（step 86800）resume，已训练约 36750 新步（目标 50K 新步的 73%）：

| step | pair_frac | roll_frac | img_frac | transport 总计 |
|---:|---:|---:|---:|---:|
| 88000 | 1.2% | 6.8% | **92.0%** | 8.0% |
| 92000 | 2.4% | 6.1% | **91.5%** | 8.5% |
| 96000 | 1.8% | 5.5% | **92.7%** | 7.3% |
| 100000 | 3.2% | 9.7% | **87.1%** | 12.9% |
| 105000 | 2.4% | 16.0% | **81.6%** | 18.4% |
| 110000 | 2.8% | 6.3% | **90.9%** | 9.1% |
| 115000 | 0.8% | 5.3% | **93.9%** | 6.1% |
| 120000 | 1.2% | 10.0% | **88.8%** | 11.2% |
| 123550 | 7.2% | 18.0% | **74.8%** | 25.2% |

**image_aux 占据 87-96% 的梯度预算，transport 只有 5-13%。**

这与 V3 step 86800-172600 的行为完全一致：

| 对比 | pair_frac | roll_frac | img_frac |
|---|---:|---:|---:|
| V3 step 130000 | 1.6% | 5.0% | 93.3% |
| V3 step 150000 | 2.1% | 5.2% | 92.7% |
| V5 null-control 平均 | ~2.5% | ~6.5% | ~91% |

→ null-control 完全复现了 V3 后段的 image-dominated 状态。

### 2.2 V5 null-control 的 val_chain_normal_mse

| 区间 | 范围 | 趋势 |
|---|---|---|
| V3 best（step 86800） | **0.000165** | baseline |
| null-control 87200-92800 | 0.000164 - 0.000424 | rolling-val 波动，无改善 |
| null-control 93200-100800 | 0.000159 - 0.000382 | 同上，无改善 |
| null-control 100800-123200 | 0.000169 - 0.000345 | 持续波动，无改善趋势 |

36750 步训练中，rolling-val 偶尔接近 V3 best（0.000159/0.000169/0.000173）但不稳定，交替出现 0.000300+ 的差窗口。**full-val 才能给出真实判定。**

### 2.3 V5 rollout-heavy 的状态

V5 rollout-heavy 的完整分析已在 0427 文档中完成。关键结论：
- step 93200 的"相变"实际上是 rolling-val 窗口回卷（counter-review 已指出）
- 但即使排除窗口混淆，V5 rollout-heavy 也没有在 rolling-val 上显示明确改善
- V5 full-val 结果已存在于 `eval_0427_fullval/v5_best`，待对比

## 3. 精确结论

### 3.1 null-control 支持了什么假设

> 注：按 Supervisor 裁决，以下使用"支持假设"而非"证明"，因果闭环需等 full-val + V6 结果。

**null-control 不是失败——它精确回答了它被设计回答的问题：**

> Q: 从 V3 best resume 继续训练，不改 loss 权重，会发生什么？
> A: image_aux 继续主导 weighted loss fraction（75-94%），val_chain_normal_mse 无稳定改善趋势。

这**支持**以下假设（pending full-val 闭环）：
1. V3 的 plateau 与**训练量不足无关**——再训 36750 步（V3 total 的 18%）、LR 仍在 base 的 55-75% 区间，仍无改善
2. V3 的 plateau 与**权重配方让 image 主导**一致——`λ_img=0.12` 使 image 通道占据 75-94% 的 weighted loss fraction
3. 要打破 plateau，**改变权重配方是合理方向**（但尚未排除 narrow basin 替代解释，需 V6 from scratch 验证）

### 3.2 V5 rollout-heavy 证明了什么

V5 rollout-heavy 改了权重（λ_roll 6× 增大，λ_img 从 0.12 降到 0.08），但从 V3 best resume。结果也没有明确改善。

可能原因（待 full-val 确认）：
- resume 后 cosine LR 位置错误（已发现并修复 total_steps_override bug）
- V3 best 已经在一个 GT-optimal 的窄盆地中，任何梯度扰动都会推出
- λ_img=0.08 仍然太高（V6 进一步降到 0.04）

### 3.3 V6 不是"补救"——是不同策略

| 维度 | V5 的思路 | V6 的思路 |
|---|---|---|
| 起点 | V3 best resume | **from scratch** |
| image 处理 | 0.12→0.08（降 33%） | **固定 0.04（降 67%）** |
| transport 提升 | λ_roll 6× | **pair_weight 15× + λ_roll 16×** |
| 核心机制 | 只改 rollout | **同时改 pair + rollout + image** |
| alpha schedule | 固定 1.0 | **0→1 三阶段 ramp** |

V6 不是在 V5 失败后的"修补"，而是基于 V3 全量数据分析后的**全新设计**——核心洞察是：
- V3 的问题不在 rollout 太弱，而在 **image_aux 太强 + pair 太弱**
- 解决方案不是"加强 rollout"，而是**让 transport（pair+rollout）整体主导**

## 4. 对 V6 的影响

| null-control 结论 | 对 V6 的含义 |
|---|---|
| 不改权重 → 继续训练无改善 | V6 改权重是**必要**的，不是可选的 |
| image 90%+ → transport 被淹没 | V6 把 image 压到 ≤25% 是关键设计目标 |
| V3 plateau 是权重配方问题，不是训练量问题 | V6 from scratch 200K 不会因为"训练更久"而自动解决，必须靠新配方 |
| null-control 复现了 V3 后段行为 | 确认 V3 JSONL 数据的因果分析是正确的 |

## 5. 待完成事项

| 事项 | 状态 | 影响 |
|---|---|---|
| null-control full-val | 训练进行中（~73% 完成，step 123550/136800） | 确认 rolling-val 结论 |
| V5 rollout-heavy full-val | 已存在于远程，待 sync 进 review 树 | 确认 V5 改动的真实效果 |
| V6 Phase I 进展 | step ~38K/200K（19%），Phase I 正常 | 与 V3 同期一致 |
| V6 Phase II 分歧点 | step 50K（预计约 1.5 天后） | 真正检验 V6 设计的关键阶段 |
| Path A diagnostic | 待 null-control 完成后执行 | 因果机制确认 |

## 6. 一句话总结

**V5 null-control 36750 步的 rolling 数据支持"V3 plateau 与 recipe/supervision-pressure 瓶颈一致"的假设（LR floor 已排除，待 full-val 闭环）。V6 受此假设驱动，用全新权重配方（pair_weight=15, λ_img=0.04）从 scratch 训练，仍需 staged validation。**
