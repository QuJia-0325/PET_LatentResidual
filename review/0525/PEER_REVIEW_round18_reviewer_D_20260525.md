# Peer Review Round 18 — Reviewer D

- reviewer name: D
- date: 2026-05-25
- scope: Strategic Next Step (post A4 bracket)
- independence: 本报告独立撰写，不参考其他 reviewer 草稿

---

## 0. 机械核对结论（只读）

我先核对了本轮核心 substrate 是否自洽：

1. A4 bracket 核心数字一致
- A4-mid NORMAL = 36.893917...
- V7 baseline NORMAL = 36.780951...
- delta = +0.112966 dB（可四舍五入为 +0.1130）
- A4-low delta = -0.079994 dB（可四舍五入为 -0.0800）

2. “A4-mid > V18”结论成立
- V18.best NORMAL = 36.8112
- V18.last NORMAL = 36.8426
- A4-mid = 36.8939
- A4-mid 相对 V18.last 仍有 +0.0513 dB

3. 当前证据边界仍然成立
- single-seed 训练证据
- slice-level full-val 证据（非 patient-level）

以上三点成立后，我认可 Round 18 进入“战略重排”是合理的。

---

## 1. TL;DR（Reviewer D）

- A4-mid +0.113 dB 不是噪声，足以改写项目主线。
- 但“立即大规模架构换代（X2）”在当前证据阶段风险过高，会把 7 个月窗口变成高失败率探索。
- 主推策略应是：
  - 主战略：Hybrid X1 + X3（机制拆解 + 一次联合加成判定）
  - 备选：X2（仅在 X1/X3 给出明确天花板后再开分支）
- V18 角色应降为次要 ablation，不应继续作为主叙事。
- paper 现在应立即启动（至少完成 outline + methods + results 主体草稿），实验与写作并行。

---

## 2. Q1-Q7 逐条判定（APPROVE / MODIFY / REJECT）

### Q1 — A4-mid +0.113 dB 真不真？
**Verdict: APPROVE（带边界）**

理由：
- 信号量级对比历史 noise floor 显著，不属于“边缘漂移”。
- D20/D10/D4/NORMAL 同向改善，方向一致性强。
- 即便是 single-seed，当前量级已经足以作为“主结果候选”。

边界：
- 论文措辞应写成“slice-level full-val improvement”而非 patient-level significance。
- 需补一个轻量稳健性附件（例如 per-slice win-rate 与 bootstrap CI），不用重新定义 patient-level p-value。

### Q2 — M1-M4 哪个机制最可能？最小 disambig 实验？
**Verdict: MODIFY**

主推机制：
- 首选 M1 + M4 组合解释。
  - M1：latent-only 监督不足，pixel-space 梯度提供更直接可学习信号。
  - M4：image_aux 把 z_pred 锚定在 decoder 可解码流形附近。
- M2/M3 更像“贡献子项”而非主因。

最小可区分实验（满足 ≤1 run + 1 probe）：
1. 单 run：在 λ=0.08 下做 l1-only（关闭 ssim/seam）
2. 单 probe：记录 z_pred 到 decode(z_pred) 的误差分布与频谱/结构统计，和基线 λ=0.04 对比

判别逻辑：
- 若 l1-only 仍保留大部分增益，主因偏 M1/M4；
- 若显著掉点，再分解到 M2/M3（下一轮再做，不在本轮强行全做 3-run）。

### Q3 — 不调小参约束下，X1-X5 选哪 1-2 个？
**Verdict: APPROVE（主推两项）**

主推 1：X1（机制深拆，轻量版）
- 不是“调小参”，而是把主结果从经验现象提升为可解释机制。
- 但我不支持直接 3-run 全拆；先做“1 run + 1 probe”版本，减少被视为小修小补的风险。

主推 2：X3（image_aux + LoRA additive 判定）
- 1 个 run 就能回答“V18 是否彻底冗余”这一高价值问题。
- 这不是 V18.v2 式扩张，而是战略收口实验：做完即可决定 V18 线去留。

拒绝项：
- X2：当前阶段过重（4-8 周，失败率高），不应作为马上主航道。
- X4：会破坏既有可比性，且收益不确定。
- X5：价值高但强依赖数据获取，不适合作为当前主计划。

### Q4 — V18 在 paper 里的角色？
**Verdict: APPROVE (b)**

选择：
- (b) 次要 ablation

解释：
- A4-mid 已经成为主信号；V18 不再是 headline。
- 但 V18 不应“完全撤出”，因为它仍是“我们尝试过更重工程路径，收益较小”的关键对照，能增强论文可信度与工程完整性。

### Q5 — 是否漏关键方向 X6+？
**Verdict: MODIFY（补充 1 个 X6）**

建议 X6：Dose-aware conditional transport（非小调参）
- 在 transport 输入引入 dose-level 条件嵌入（而非仅靠固定链式 timepoints）
- 目标：把“不同剂量恢复难度”显式建模，减少统一模型在极低剂量段的欠拟合

为什么优于 X7/X8/X9 当前阶段：
- 不依赖新增跨模态/跨数据资产，实施门槛低于 X8/X9；
- 相比纯文本条件 X7，更贴近当前主任务信号路径。

### Q6 — paper 时间表（MICCAI 2027 / TMI / MedIA）
**Verdict: APPROVE（双轨）**

时间判断：
- MICCAI 2027：可行，但前提是 6-8 周内完成机制最小闭环与叙事收敛。
- TMI/MedIA：作为并行备选，不应阻断当前会议稿节奏。

写作启动建议：
- 立刻启动到“methods + results 主体可投状态”，不是只停留在 outline。
- 具体：1 周内定稿图表框架与主结论句式；实验补充在后续版本增量合并。

### Q7 — 偏差审查（B87+）
**Verdict: APPROVE（存在偏差，需纠偏）**

- B87 [MED] A4-mid 叙事浪漫化
  - 风险：把 +0.113 当“最终答案”，低估 single-seed 与外部泛化问题。

- B88 [MED] anti-tuning 教条化
  - 风险：把所有机制拆解都归为“小打小闹”，导致论文缺关键可解释性证据。

- B89 [MED] architecture pivot 过早承诺
  - 风险：在未完成当前主线闭环前转 X2，增加高失败率路径依赖。

- B90 [LOW-MED] V18 全盘否定偏差
  - 风险：把 V18 彻底清零会丢失负结果价值与对照叙事完整性。

---

## 3. 主战略 verdict（单选 + 备选）

主战略（单选）：Hybrid = X1(light) + X3
- X1(light): 先做最小机制区分（1 run + 1 probe），避免 21d 全拆。
- X3: 做 1 次 additive 判定 run，快速决定 V18 在新主线下是否保留。

备选：X2
- 仅在 X1/X3 完成后，若出现“image_aux 已接近天花板且 V18 不可加成”的明确信号，再开启 X2 分支。

---

## 4. V18 角色判定

选择：(b) 次要 ablation

建议文稿定位：
- 主结论：image_aux strength shift drives dominant gain.
- 次结论：decoder LoRA path offers smaller incremental gain and does not redefine the main story.

---

## 5. image_aux mechanism 主推 + 最小实验

主推机制：M1 + M4

最小 disambig experiment：
1. 单 run（λ=0.08, l1-only）
2. 单 probe（latent manifold proximity / reconstruction structure stats 对比 λ=0.04）

成功标准：
- 若增益基本保留：确认主因偏 M1/M4，论文可写“pixel-anchor mechanism”。
- 若明显下滑：再进入 M2/M3 细分，不在本轮一次性铺开。

---

## 6. paper 时间表建议

- 目标：MICCAI 2027 主线可行，TMI/MedIA 作为延伸稿轨道。
- 立即动作：进入“可投草稿阶段”而非仅 outline。
- 建议节奏：
  - Week 1-2：methods/results 主体 + 主图（A4 对照 + V18 降级对照）
  - Week 3-6：X1(light)+X3 完成并并入
  - Week 7-8：统一叙事与补充材料（limitations, robustness appendix）

---

## 7. 执行口径（给决策者）

如果你只接受“非小打小闹”路径，我建议不要直接跳 X2 全量重构。更优解是先用 X1(light)+X3 在 2-3 周内把主线闭环、把 V18 定位定死，再决定是否开 X2 作为下一阶段项目。这样既不陷入参数微调循环，也不把当前已成形的 paper 机会押在高风险重构上。

---

## 8. What I did NOT review

- 不重审 Round 12-17 已签结论。
- 不重审训练执行细节与脚本正确性（仅使用已有报告的核对数据）。
- 不做任何代码修改、配置改写或实验重跑建议落地。