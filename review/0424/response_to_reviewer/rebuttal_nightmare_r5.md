# Rebuttal — Response to Nightmare Round 5 Review

**Date**: 2026-04-24  
**Responding to**: `review/0424/claude-reviewer/nightmare/05_final_consensus_and_roadmap.md`  
**Score received**: 3.05 / 10

---

## 总体回应

我们接受 Nightmare 审查的大部分裁决。审查严厉但公正。以下逐条回应致命发现和关键建议。

---

## 对 5 个致命发现的回应

### F1: "Hop0 机制效应 0.083 dB 无统计学意义"

**接受。** 我们承认在没有 σ_seed 测量的情况下，0.083 dB 的效应确实无法与随机波动区分。

**我们的行动**：
- Path A 诊断实验（TF vs RO gap）已设计并可执行（0.5 GPU-day）
- 这不是为了"挽救"hop0 叙事，而是为了定位 10 dB gap 的机制原因（exposure bias vs velocity capacity）
- 如果 Path A 显示 exposure bias 不主导，我们接受 hop0 路线的负结论

### F2: "故事与证据方向相反——tail 增益 > head 增益"

**部分接受。** tail 增益确实大于 head 增益，这与"hop0 是瓶颈"的叙事不一致。

**但需要补充说明**：
- 这恰恰印证了我们 0424 的分析——latent MSE 被 decoder **非线性放大**，越靠近 chain 末端放大倍数越大（每 0.0001 MSE: D20 = 5.4 dB, NORMAL = 9.4 dB）
- tail 增益大不代表 hop0 不重要，而是 decoder 对 tail 误差更敏感
- 我们同意需要 reframe 叙事，从"hop0 瓶颈"改为"级联 transport 的误差传播与 decoder 非线性放大"

### F3: "5 个 config 全在 36.12-36.21 dB plateau"

**接受。** 这是事实。我们所有 init fix、EMA、LR 调整、pixel forcing 开关都未突破此 plateau。

**我们的理解**：
- Plateau 不是因为"什么都不 work"，而是因为 **latent MSE（~0.0002）已经是当前 pair loss + rollout loss 在 50K 步下能达到的精度极限**
- 突破 plateau 需要两个方向之一：
  1. 更长训练（200K 步 = 15 遍数据 vs 当前 3.8 遍）→ 正在进行
  2. 更好的 loss 信号（image-space loss 覆盖全 hop）→ 方案已设计

### F4: "10 dB oracle 差距无机制解释"

**不完全接受。** 我们的 0416 E1 诊断已将 gap 归因到 velocity 端（96-99%），这在 `review/0416/conclusion.md` 中有 n=7403 的 full-val 证据。

**我们承认不足之处**：
- E1 只做了一层分解（transport vs decoder），没有进一步区分 exposure bias vs velocity capacity
- Path A（已设计）正是为了完成这第二层分解
- `breakthrough_analysis_10dB_gap.md` 已系统化了这个诊断链

### F5: "零外部 baseline"

**接受。** 这是硬伤。36.2 dB 没有参照系，无法判断水平。

**我们的计划**：
- 在后续投稿准备中加入至少 1 个外部 baseline（Pix2Pix 或 PET-DDPM）
- 当前阶段优先完成 Path A 诊断 + 200K 训练，baseline 在投稿准备阶段补充

---

## 对结构发现的回应

### F6: "组件皆为已发表方法的微小特化"

**接受。** 如果 hop0 机制在统计学上不可辨，那这些组件确实不构成方法贡献。

**可能的 reframe**：贡献在于系统性诊断框架（A/B/C/D 分解），而非单个组件。但需要跨域验证（如 Reviewer A 所指出）。

### F7: "Checkpoint-selection 污染 leaderboard"

**接受并已修复。** N1 last > N1 best 的反转已被记录。我们的 full-val rerank 协议（0423 已建立）正是为了解决这个问题。所有后续结论均以 full-val rerank 为准。

### F8: "26 configs + 2 个死代码"

**接受。** iREPA Spatial Projector 和 SeamRefiner 在主结果配置中均未启用。这是 scope creep 的结果。

**说明**：这些模块是探索过程中的产物，不应作为方法贡献。

### F9-F10: 被证伪的攻击

我们注意到 Tribunal 独立验证了：
- N1 vs C 确实只差一个变量（`pixel_forcing_disabled`）✓
- patient-level split 确实在 upstream 实现 ✓

---

## 对 Nightmare 建议的回应

### 关于"终止 200K run"

**不接受此建议。** 理由：

Nightmare 的论证是"stop criterion 容差 > 配置方差"。但 200K 和 hop0 机制是**不同的问题**：

| 实验 | 回答的问题 |
|------|----------|
| σ_seed 实验 | hop0 机制的效应是否真实？ |
| **200K run** | **更长训练能否降低 latent MSE 从而突破 36.2 plateau？** |

200K run 不是在测 hop0 效应——它是在测 **transport backbone 在 v3 权重修正 + 15 遍数据后是否还能继续收敛**。即使 hop0 机制效应为零，transport_avg 从 36.2 → 37.0 dB 也是有价值的发现。

**可证伪的预测**（按 Nightmare 要求记录）：
> 200K transport_avg 将 > 36.35 dB（比 50K best 36.206 提升 > 0.15 dB）。  
> 如果 150K 处 transport_avg 与 50K 相比 < 0.05 dB 且曲线明显 plateau，提前终止。

### 关于 Path A

**完全接受。** Path A（TF vs RO gap）是最高优先级诊断实验。

- 脚本已实现：`scripts/diagnose_tf_rollout_gap.py`（488 行）
- 0.5 GPU-day，零训练
- 回答 velocity 误差是来自 exposure bias 还是 capacity

### 关于 Path C (Decoder FT 诊断)

**接受为诊断实验。** 我们同意 Path C 不是方法，而是估计 RAE 上界的诊断。待 Path A 结果出来后执行。

### 关于 σ_seed 实验

**接受其重要性，但延后优先级。** 理由：
- σ_seed 回答的是 hop0 机制的统计显著性
- 我们当前已接受 hop0 机制可能无效，重心已转向 transport backbone 本身
- 如果 200K 或 Path A 产生显著突破，σ_seed 的优先级自然提升
- 4 GPU-day 的成本不低（6 个 50K runs），在 transport 问题明确前不急于投入

---

## 我们当前的设计决策

### 正在执行的实验

| 实验 | GPU | 目的 | 状态 |
|------|-----|------|------|
| v3 50K | GPU-3 | 测试权重反转 + velocity rebalance | 进行中 |
| v3 200K | 待启动 | 测试更长训练能否突破 plateau | config 已就绪 |

### 下一批实验（按 v3 结果决定）

| 条件 | 下一步 |
|------|--------|
| v3-50K transport_avg > 36.30 → 权重修正有效 | 继续 200K，关注收敛曲线 |
| v3-50K ≈ 36.20 → 权重修正无效 | 立即跑 Path A，定位是 exposure bias 还是 capacity |
| 200K transport_avg > 36.50 → 训练步数是瓶颈 | 考虑 400K 或 grad_accum |
| 200K ≈ 36.20 → 训练步数不是瓶颈 | 转向 Path B（随机 hop image loss）或 Path C |

### 叙事 reframe 方向（已接受）

从 "hop0 pixel forcing 突破首跳瓶颈" 改为：
> "系统性诊断 4-hop latent flow matching 在低剂量 PET 重建中的误差传播与性能边界"

这与 Nightmare Round 5 的建议一致。

---

## 对评分的回应

3.05/10 的评分我们不争辩——在当前数据下这是公正的。但我们认为 breakthrough_analysis 中提出的 Path A + 200K 组合有潜力将分数推到 3.7-4.5 范围（workshop viable），前提是：

1. Path A 能给出 clear verdict（exposure vs capacity）
2. 200K 能展示 plateau 是否可打破
3. 至少 1 个外部 baseline 提供参照系

这 3 个条件都在 2 周内可验证。
