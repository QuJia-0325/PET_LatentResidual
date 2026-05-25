# Peer Review Round 18-Codex-Audit — Audit of Codex's Independent Design Analysis

- date: 2026-05-26
- branch: foc_lite_hop0 (commit 3b62b59)
- 主审对象: [codex_x1_x3_design_analysis_20260525.md](./server/codex_x1_x3_design_analysis_20260525.md)
- 触发: codex 在 launch X1-lite + X3 期间, **独立**做了一份 design analysis 并 push, 提出 4 个新 limitation (CL1-CL4) + 1 个 architecture-level 观察. 这是 codex 首次主动做 design-level audit, 内容非纯执行汇报. 需要外部 reviewer 评估这份分析的论点是否站得住, 是否影响 Round 19 决策.
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 18 战略层 + Round 18-Prep 代码层 已签. X1-lite + X3 正在跑 (X1 GPU1, X3 GPU3). 本轮**只审 codex 这份 design analysis 的论点质量**:

**本轮审**:
- CL1 (gradient-budget confound) 论点是否成立? 严重性多高?
- CL2 (X1-lite 无法分 M1 vs M4) 是否真不能分? 是否有便宜补救?
- CL3 (X3 10K 短窗) 是否被 codex 正确 framing?
- CL4 (slice-level + single dataset) 是否仅复述 Round 17-Stats, 还是有新内容?
- codex architecture-level 观察 (transport 训练 objective 不够 decoder-aware) 是否合理?
- codex 解读阈值 (0.02 dB / 0.05 dB / 0.10 dB) 是否符合现有 substrate (V14 noise, A4-mid signal magnitude)?
- codex 推荐的 future direction (decoder-pullback / multi-hop chain / dose-conditional / 完全 pivot) 排序是否合理?
- codex 非 staggered launch (user override) 是否带来未识别风险?

**本轮不审**:
- Round 18 战略选 X1-lite + X3 vs 其它 (Round 18 4/4 已签)
- Round 18-Prep 4 HIGH + 6 MED fixes 实施质量 (commit a764742, codex 已 verify runtime active)
- 是否启动 X1-v2-balanced (是 Round 19 TODO-3, 等 X1-lite outcome 出来再决定)
- paper venue / 时间表 (Round 18 集成已签 MICCAI 2027)

---

## 1. Codex 提的 4 个 limitation (CL1-CL4) 概要

### CL1 — X1-lite 不是 perfectly clean mechanism decomposition

**Codex 论点**:
> "Setting ssim_weight=0 and seam_weight=0 removes SSIM/seam terms, but it also changes the total image_aux gradient scale and gradient spectrum. If X1-lite is below A4-mid, we cannot immediately conclude that SSIM/seam are mechanistically essential. The drop may partly reflect reduced total supervision strength."

**数学验证 (claude 已查)**:
- A4-mid: `image_aux_total = 0.08 × (1.0 × l1 + 0.25 × ssim + 0.10 × seam)` = `0.08 × (l1 + 0.25 × ssim + 0.10 × seam)` ≈ `0.08 × 1.35 × |loss|` (若所有 loss 同量级)
- X1-lite: `image_aux_total = 0.08 × (1.0 × l1 + 0 × ssim + 0 × seam)` = `0.08 × l1` ≈ `0.08 × 1.0 × |loss|`
- 总梯度 nominal 差 ~35% (1.35 vs 1.0)

claude self-review (4 + 3 reviewer 视角) **全部漏了**这条. codex 第一次提.

### CL2 — X1-lite 无法分 M1 vs M4

**Codex 论点**:
> "X1-lite can test whether L1 through the decoder is enough, but it cannot distinguish pure latent-transport undertraining from decoder-manifold anchoring."

**M1**: latent transport DiT 在纯 latent MSE 下 undertrained, pixel-space gradient 提供更强信号
**M4**: image_aux 把 z_pred 隐式 anchor 到 decoder 可解码 manifold

(M1 + M4 是 Round 18 4/4 reviewer 主推机制, 但 X1-lite 不区分它们.)

### CL3 — X3 只测 short-window LoRA additivity

**Codex 论点**:
> "X3 follows an A3-style 10k adaptation window from V7 best. If it does not exceed A4-mid, it rules out short-window additivity, not all possible long-training LoRA benefits."

(已是 Round 18-Prep D1 设计 limitation, codex 重申.)

### CL4 — single-dataset + slice-level

**Codex 论点**:
> "Without patient IDs or external datasets, the paper should avoid patient-level statistical claims."

(已是 Round 17-Stats 已签结论, codex 重申.)

---

## 2. Codex architecture-level 观察

> "The biggest architecture signal from A4-mid is not that LoRA is the missing piece. The signal is that pure latent transport loss is poorly aligned with the final decoded image metric."

提议的 4 个 future direction:

| Direction | Codex priority |
|---|---|
| Decoder-aware latent metric / pullback loss | High |
| Multi-hop decoder-aware chain supervision | Medium |
| Dose-conditioned transport | Medium |
| Full architecture pivot | Not recommended |

---

## 3. Codex 解读阈值

X1-lite outcome → mechanism verdict:
| Δ vs A4-mid | Codex interpretation |
|---|---|
| within 0.02 dB | L1 sufficient |
| 0.02-0.05 dB below | L1 partial, M2/M3 maybe |
| near V7 or below | L1 insufficient |

X3 outcome → V18 verdict:
| Δ vs A4-mid | Codex interpretation |
|---|---|
| clearly above | additive |
| ≈ A4-mid | redundant |
| below | conflict |

---

## 4. 给 reviewer 的 8 个深审问题

### Q1 — CL1 严重性评估

claude 漏了, codex catch. 你怎么看?

- CL1 论点是否完全成立? 还是 claude 可以 push back 说 "0.08 × 0.35 = 0.028 的额外 gradient 在 transport DiT 上量级太小, 不会改变最终 chain PSNR"?
- 严重性: HIGH (paper 必修, 否则 reviewer 必挑) / MED (limitation 写进 Discussion 即可) / LOW (无需提)?
- 是否值得现在就启 X1-v2-balanced (l1_weight=1.35) 用 slot 3 (打破 Round 18 stop rule)? 还是按 user 选项 A 等 outcome?

### Q2 — CL2 可补救性

X1-lite 不能分 M1 vs M4. 是否有 ≤ 1 个新 experiment 能区分?

候选:
- **M1-test**: 在 frozen decoder 输出上加 L1 + L2 + perceptual, 看是否 transport 收敛更快 (测 transport undertraining)
- **M4-test**: 测 X1-lite 训完后 z_pred 在 decoder manifold 上的距离 vs V7 z_pred (是否 anchor 更近)
- 这两个测都不需要新 from-scratch run, 都是 inference-time probe

是否应该写进 Round 19 TODO 作 future probe?

### Q3 — CL3 framing 是否过保守

codex 说 "X3 不否定 long-training LoRA". 这是否给 V18 family 留了**过多**复活空间? 

- V18 实测 last (40K LoRA) vs best (5K LoRA) 提升只 +0.0314 dB. 按比例 X3 在 10K (5K+5K) 应 ≈ V18.best 区域
- 若 X3 ≈ A4-mid (= V18.best + ~0.08), 已经超越 V18.best ~80%. long-training 是否真有更多空间?
- codex 的 framing 是否过严, 实际上 X3 outcome 应该已经足以判 V18 family 死活?

### Q4 — Architecture observation 论证质量

codex 说 "transport 训练 objective 不够 decoder-aware". 论据:
- A4-mid +0.113 dB 比 V18 +0.062 dB 大近 2 倍
- image_aux (post-hoc auxiliary) 主导 vs decoder LoRA (capacity-side)

是否成立? 是否漏 alternative explanation?
- alt-A: image_aux 只是 inductive bias, transport 实际并没 undertrained (用 V14 noise 0.0004 看, V7 已经接近极限)
- alt-B: image_aux 等价于"在 latent 上加 SGD 的 effective regularizer", 与 decoder-awareness 无关
- alt-C: image_aux 实际上是 **outlier suppression** mechanism (decoder backprop 让 tail-distribution slice 不被纯 latent loss 拉跑)

### Q5 — Codex 4 个 future direction 优先级排序

| 方向 | 你认同 codex 排序? | 修改建议 |
|---|---|---|
| Decoder-aware latent metric (High) | | |
| Multi-hop chain supervision (Med) | | |
| Dose-conditional transport (Med) | | |
| Full architecture pivot (Not recommended) | | |

是否漏关键方向 (e.g. token-level perceptual loss / SSIM-on-latent / contrastive transport)?

### Q6 — Codex 解读阈值的统计基础

- "within 0.02 dB" = 50× V14 noise (0.0004 dB). 是否过严? V18.best - V7 = +0.030 是 75× noise, codex 阈值会把 V18 信号判为"显著区别"
- "0.02-0.05 dB below" 区间是否对 X1-lite 实际可能 outcome 太窄? gradient-budget 35% 降低可能让 X1-lite 落在 0.02-0.08 dB 区间
- 阈值应该用 V14 noise multiplier 重新校准还是用 A4-mid magnitude 比例?

### Q7 — Codex 非 staggered launch 风险

user override 让 codex 并行 launch (不等 X1 +30min health check). codex 报告 "no OOM, both healthy".

- 是否有未识别 race condition (dataloader 同时争数据, 影响 X3 warmstart 行为)?
- 是否有 GPU memory share / NCCL bandwidth contention 影响某个实验的训练速度?
- 是否应该在 Round 19 TODO 标注 "X1-lite 和 X3 实际跑了 parallel-not-staggered, 可能影响 wall-clock 但不影响 outcome"?

### Q8 — Codex 这份分析的元评估

- codex 主动写 design analysis 是好行为还是过界 (codex 应该只执行)?
- codex 分析 quality 是否达到一个独立 reviewer 水准? 是否值得未来每个大型 task 都让 codex 写一份 launch-time analysis?
- codex 是否引入新偏差? (e.g. "transport not decoder-aware" 论点是否 anchor 到 codex 自己的架构偏好)

---

## 5. 资料目录

### 5.1 本轮主审 (codex 文档)

- [codex_x1_x3_design_analysis_20260525.md](./server/codex_x1_x3_design_analysis_20260525.md) (127 行)

### 5.2 上游 context

- [REVIEW_INTEGRATION_round18_20260525.md](./REVIEW_INTEGRATION_round18_20260525.md) (战略 4/4 共识)
- [REVIEW_INTEGRATION_round18_prep_20260525.md](./REVIEW_INTEGRATION_round18_prep_20260525.md) (code-level 3/3 MODIFY-BEFORE-PUSH)
- [CODEX_TASK_ROUND18_X1_X3_20260525.md](./CODEX_TASK_ROUND18_X1_X3_20260525.md) (任务文档, 含全部修复)
- [Round19-TODO.md](./Round19-TODO.md) (含 CL1 选项 A/B/C, user 已选 A)

### 5.3 已知 substrate (作信号校准用)

| run | image_aux λ | LoRA | NORMAL |
|---|---:|---|---:|
| V13 | 0.00 | 无 | 36.4943 |
| A4-low | 0.02 | 无 | 36.7010 |
| V7 | 0.04 | 无 | 36.7810 |
| V14 | 0.04 | 无 | 36.7806 (单 seed noise 估计) |
| V18.best | 0.04 | r32 + KL | 36.8112 |
| V18.last | 0.04 | r32 + KL | 36.8426 |
| A3 (V18-cap) | 0.04 | r32 (no KL) | ≈ V18.step170k chain |
| A4-mid | 0.08 | 无 | 36.8939 |

---

## 6. 输出格式

每位 reviewer 独立产出 markdown:

### 6.1 8 个问题逐条 (AGREE / DISAGREE / MODIFY + 论据)

### 6.2 CL1 严重性 verdict
- **HIGH** (打破 Round 18 stop rule 启 X1-v2-balanced)
- **MED** (paper 写 limitation, 不启)
- **LOW** (无需写 limitation, 直接归因)

### 6.3 X3 outcome 解读 framing 评估
codex framing 过保守 / 合理 / 过激进?

### 6.4 architecture observation 评估
codex "transport not decoder-aware" 论点: 强支持 / 弱支持 / 不支持

### 6.5 立即可执行建议
若 reviewer 找到 must-act-before-X-completes 的事项, 列出.

### 6.6 新偏差 (B97+)
若发现 codex 分析引入新 anchor / overreach.

---

## 7. 角色提示

你不在审 task md (Round 18-Prep 已签). 你不在审战略 (Round 18 已签). 你在审一份**codex 主动生成的设计分析文档**:

- codex 第一次做这种 design-level audit, 它的论点质量决定 Round 19 决策框架
- CL1 是 claude + 4 reviewer + 3 reviewer 全部漏的 confound, codex 一人 catch — 这是 humbling 但也 valuable 信号
- 你的 verdict 直接决定 user 是否要打破 stop rule 启 X1-v2-balanced (7d × 1 GPU)
- 你的 verdict 也决定未来是否让 codex 每次都做 design analysis (是否把 codex 抬到 reviewer 角色)

请极其严格. 不要为了 "stay polite to codex" 接受不严论点; 也不要为了 "show critical" 否定 codex 真正的发现.
