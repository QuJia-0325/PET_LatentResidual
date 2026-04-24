# Reviewer Reply — Response to the Rebuttal

**Date**: 2026-04-24  
**Responding to**: `review/0424/response_to_reviewer/rebuttal_nightmare_r5.md`  
**Reviewer**: Nightmare Panel (synthesized)  
**Verdict**: **Rebuttal partially accepted. 3 specific pushbacks + 1 concept correction + 1 compromise.**

---

## 0. Overall Assessment of the Rebuttal

Rebuttal tone is professional and most concessions are clean (F1, F3, F5, F6-F10). Panel特别认可：
- 诚实接受 hop0 可能无效 (F1)
- 接受 Path A 为最高优先级诊断 (zero friction)
- 接受 scope creep / dead code / leaderboard 污染 (F7/F8)
- 接受叙事 reframe 到 "systematic diagnosis of 4-hop latent flow matching error propagation"

这些是这份 rebuttal **显著加分的地方**——和典型的 "dispute everything" rebuttal 不同，proposer 表现出真实的科学诚实度。

但下面 3 条必须反驳。**这不是形式性争辩**——其中两条涉及方法论错误，如果带进 rebuttal 到实际 venue 会被外部 reviewer 抓住。

---

## 1. 反驳 · 200K Run 的新预测仍有致命缺陷

### 1.1 Rebuttal 的新论证

> "200K 和 hop0 是不同问题。200K 测 transport backbone 的收敛性。"
>
> 可证伪的预测：`200K transport_avg > 36.35 dB` (比 50K best 36.206 提升 > 0.15 dB)。
>
> 停止准则：150K vs 50K < 0.05 dB 且曲线 plateau 则终止。

### 1.2 Panel 回应

**Proposer 这次的论证实际上是升级版**——把 ±0.05 dB 容忍窗口换成了 "> 0.15 dB" 的硬预测，0.15 dB **确实**已经大于 5-config 跨度 (0.083 dB)。这是进步。**但仍有两处崩溃**：

#### 崩溃 1: 没有 σ_seed 就无法解释 0.15 dB

假设 200K 跑出 transport_avg = 36.40 dB (比 50K 的 36.206 高 0.194 dB)。

- 如果 **σ_seed ≈ 0.03 dB** → 这是 +6σ 提升，方法论上成立。
- 如果 **σ_seed ≈ 0.08 dB** → 这是 +2.4σ 提升，边缘显著性，CI 下界可能接近 0。
- 如果 **σ_seed ≈ 0.12 dB** → 这是 +1.6σ 提升，完全可以是噪声。

**Proposer 的预测在没有噪声尺度的情况下无法被任何外部 reviewer 相信。** "我们预测了 0.15 dB 且跑出了 0.19 dB" 在 2026 年的顶会是**立即被质询 σ_seed**的点。

#### 崩溃 2: 停止准则仍然不可证伪

"< 0.05 dB 且曲线 plateau" 这个准则里的 "plateau" 是人类视觉判断，不是定量门槛。
- 如果 150K = 50K + 0.04 dB → 触发 "abandon"
- 如果 150K = 50K + 0.10 dB 但后半曲线平缓 → "abandon"? "continue"?
- 如果 150K = 50K + 0.03 dB 但方差比 50K 大 → 哪种情况？

这条停止准则的操作定义缺失。

### 1.3 Panel 修订的妥协方案

Panel **不再要求 kill 200K**，但要求以下 3 条同时满足才允许启动：

1. **先跑 Exp 1 σ_seed**（3 seeds × 1 config = 2 GPU-day，砍掉 N1 arm），给 σ_seed 一个 point estimate。
2. **把 200K 的成功准则重写成 σ-units**：
   ```
   PASS = mean(200K) - mean(50K) > 3·σ_seed  AND
          paired bootstrap 95% CI lower bound > σ_seed
   ```
3. **把 stop criterion 定量化**：
   ```
   ABANDON at 150K if:
     ΔPSNR(150K vs 50K) < σ_seed  OR
     PSNR(150K) - PSNR(100K) < 0.5·σ_seed (curve flat)
   ```

成本：Exp 1 简化版 2 GPU-day + 200K 2.7 GPU-day = **4.7 GPU-day**，比 Proposer 原计划的 200K (2.7) + σ_seed (4) = 6.7 GPU-day **反而省 2 GPU-day**。

### 1.4 Panel 立场

- **撤回**："kill 200K now" 原裁决
- **新裁决**：200K 允许跑，但必须前置 σ_seed-lite (2 GPU-day)
- **原裁决保留的部分**：stop criterion 必须用 σ-units 重写

---

## 2. 反驳 · σ_seed 不只为 hop0 服务

### 2.1 Rebuttal 原文

> "σ_seed 回答的是 hop0 机制的统计显著性。我们已接受 hop0 可能无效，重心转向 transport backbone，σ_seed 可延后。"

### 2.2 Panel 回应

**这是方法论错误。σ_seed 不是 "hop0 专用指标"，它是整个项目所有 PSNR 声明的 noise floor。**

三个反例：

| Proposer 仍想做的声明 | 为什么需要 σ_seed |
|---|---|
| "200K 比 50K 高 0.19 dB → 训练步数有效" | 必须用 σ_seed 判断 0.19 dB 是否显著 |
| "Path A 显示 exposure_gap_dB = 0.3" | 0.3 dB 是否 > 2σ？否则无法判定有 exposure bias |
| "外部 baseline (Pix2Pix) 低 0.4 dB" | 0.4 dB 是否 > 2σ？否则"击败 baseline" 不成立 |

**σ_seed 是所有这些声明的共同分母。** 不测 σ_seed 就像物理实验室里没校准天平。

Proposer 把 σ_seed 归类为 "hop0 问题"是**思路错误**——σ_seed 是 PSNR 本身的仪表读数精度，与任何具体实验无关。

### 2.3 Panel 立场

σ_seed 的优先级 **不能延后**。但 Panel 接受简化：
- **原要求**：3 seeds × 2 configs = 6 runs = 4 GPU-day
- **修订要求**：**3 seeds × 1 config (imgaux_boost only) = 3 runs = 2 GPU-day**。把另一半 2 GPU-day 让给 200K。

这样 σ_seed 立刻得到一个 point estimate，足以解释 200K 的结果。如果日后需要 C vs N1 的对比显著性，再补 N1 的 3 seeds。

---

## 3. 概念纠正 · F2 的 decoder 非线性放大论证

### 3.1 Rebuttal 原文

> tail 增益大不代表 hop0 不重要，而是 decoder 对 tail 误差更敏感。latent MSE 被 decoder 非线性放大，越靠近 chain 末端放大倍数越大 (每 0.0001 MSE: D20 = 5.4 dB, NORMAL = 9.4 dB)

### 3.2 Panel 纠正

**这个论证把两种不同的 "amplification" 混淆了。**

| 概念 | 定义 | 项目内值 |
|---|---|---|
| **Decoder gain** | ΔPSNR / Δ(latent MSE) at any operating point | 确实 NORMAL 端更高（因为动态范围更大） |
| **Off-manifold amplification** | PSNR(decode(z_pred) vs x_gt) − PSNR(decode(z_pred) vs decode(z_gt)) | E1 实测 D20: 0.41 dB, NORMAL: 0.19 dB —— **反而 NORMAL 更低** |

两者的物理含义完全不同：
- **Decoder gain** 大 → 同样 latent MSE 在 output 端看起来更大 → 这是 **标定问题**，不是机制问题。
- **Off-manifold amplification** 大 → decode(z_pred) 偏离 manifold 比 decode(z_gt) 严重 → 这才是 "decoder 敏感" 的机制。

E1 表明 **off-manifold amplification 在所有 hop 上都 < 1 dB，占 gap < 4%**。也就是说 decoder 对 prediction 相对 GT 是相当**鲁棒**的。

### 3.3 所以 F2 的真正机制是什么

tail 增益 (+0.66 dB NORMAL) > head 增益 (+0.18 dB D20) 的真实解释是：

```
Gap_Transport 在 chain 末端更大（10.74 → 15.70 dB），绝对值空间更大，
所以即使 same small relative improvement 也体现为更大的 dB 数字。
```

用 **relative improvement in latent MSE** 看可能更清楚：
- D20: latent MSE 改善 x% → +0.18 dB
- NORMAL: latent MSE 改善 x% → +0.66 dB

**这不支持 "hop0 是瓶颈"，因为 NORMAL 端本身有更大的改进空间。**

### 3.4 Panel 建议

把 F2 的解释改写成：

> tail 增益大于 head 增益，本质上是**因为 chain 末端的 transport gap 本身更大（15.70 dB vs 10.74 dB）**。用 absolute dB 衡量会系统性偏袒 tail。为了公平对比，未来分析应报告 **normalized improvement** (ΔPSNR / Gap_Transport)，或直接报 **latent MSE 的相对改善**。

这会让 story 更诚实，也更经得起外部 reviewer 的质询。

---

## 4. 妥协 · 时间线与执行清单

Panel 接受 rebuttal 对研究路径的整体规划，提出以下时间线（**替换 Nightmare Round 5 §3 的原清单**）：

| 日程 | 实验 | GPU-day | 阻塞 |
|---|---|---:|---|
| Day 1-2 | **σ_seed-lite**: imgaux_boost × 3 seeds × 50K | 2.0 | 解锁所有 PSNR 声明 |
| Day 1-2 并行 | **Path A** 在 imgaux_boost best.pt 上 | 0.5 | 独立于 σ_seed |
| Day 3-5 | **200K v3 run** (σ-unit stop criterion) | 2.7 | 需要 σ_seed 结果 |
| Day 3-5 并行 | **Path C decoder-FT 诊断** (5K steps, last 2 upsample blocks) | 1.0 | 独立 |
| Day 6-7 | 读数 + 写作 kickoff | 0 | — |
| Day 8-14 | workshop 短文 8 页 + Pix2Pix baseline | 2.0 | — |

**总计**: ~8 GPU-day + 1 周写作。

### 分叉决策

| Day 3 读数 | 下一步 |
|---|---|
| σ_seed > 0.08 dB | **砍 200K**，立即写负结果论文（省 2.7 GPU-day） |
| σ_seed ∈ [0.04, 0.08] dB | 200K 跑，但 stop criterion 必须收紧 |
| σ_seed < 0.04 dB | 全流程继续，有机会争取 workshop 正结果叙事 |

---

## 5. 分数调整 (post-rebuttal)

| 维度 | Nightmare R5 | 基于 rebuttal 诚实度 | 条件 (若执行新清单) |
|---|---:|---:|---:|
| Novelty | 3.0 | 3.0 | 3.5 (若诊断框架跨域验证) |
| Story | 3.5 | **3.8** (诚实接受 reframe +0.3) | 4.5 (若 F2 机制论重写诚实) |
| Implementation | 5.8 | 5.8 | 7.0 (修复 D50 列, 删死代码, σ-unit 停止准则) |
| Causal | 2.0 | **2.3** (接受 Path A/C +0.3) | 5.5 (执行 σ_seed + Path A + Path C) |
| Submission | 2.5 | 2.5 | 4.5 (加 Pix2Pix baseline) |

**Rebuttal 后即时分数**: 3.25 / 10 (↑0.20)  
**若执行修订清单**: 4.4 / 10 (workshop-viable 区间)  
**若加诊断框架跨域验证 + 外部 baseline**: 4.9 / 10

---

## 6. Panel 最终立场 (3 句话)

1. **Proposer 的诚实接受度高于平均 rebuttal**；接受 F1/F3/F5/F6-F10 且不硬辩 hop0，赢得 panel 尊重。
2. **但 200K 防御和 σ_seed 延后是方法论错误**，必须按本文 §1-§2 的修订清单执行；若拒绝，分数不变且 Causal/Submission 进一步下调。
3. **F2 的 decoder gain 论证是写作陷阱**，外部 reviewer 会抓到；改写成 "gap 绝对值不同导致 dB 系统性偏袒 tail" 会让故事立住。

---

## 7. 本回复落盘后 proposer 可选动作

- **接受 (推荐)**: 按本回复 §4 清单执行；panel 随后可以重新评审 (Round 6) 并公开承认分数上调到 ~4.4/10。
- **部分接受**: 执行 Path A + σ_seed-lite 但跳过 Path C 诊断。Panel 接受，但分数封顶 3.8/10。
- **拒绝 §1-§2**: 继续按 rebuttal 原计划跑 200K + 不做 σ_seed。Panel 立场是"祝好运"，但不再 endorse。

---

*End of Reviewer Reply. The floor returns to Proposer.*
