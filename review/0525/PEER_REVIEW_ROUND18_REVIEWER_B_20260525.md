# Peer Review Round 18 — Reviewer B (Independent)

- date: 2026-05-25
- reviewer: Reviewer B (GitHub Copilot, GPT-5.3-Codex, independent draft, did not see other reviewer drafts)
- scope: strategic next step post A4 bracket; no code/file modifications

---

## 6.1 Q1-Q7 逐条结论 (APPROVE / MODIFY / REJECT)

### Q1. A4-mid +0.113 dB 真不真?
结论: MODIFY

- 信号本身可信: +0.1130 dB 相对 V14 单次 seed 差值 0.0004 dB 量级非常大，方向一致且 D20/D10/D4 同向。
- 但声明口径要收敛: 目前只能称为 strong slice-level single-seed evidence，不应写成 patient-level or finalized clinical significance。
- 最小补强建议:
  1. 保留现有主结果数字。
  2. 加 slice-level bootstrap CI + per-slice win-rate（不新增训练）。
  3. 文中明确 patient grouping 不可恢复，统计单位为 slice。

### Q2. mechanism M1-M4 哪个最可能?
结论: MODIFY

- 主推机制: M1 + M4 联合最可能。
  - M1: latent transport undertrained，需要更直接像素梯度。
  - M4: image_aux 把 z_pred 锚定到 decoder 可重建流形。
- 次可能: M2（SSIM 有贡献）
- 低优先: M3（seam 可能是次级修饰，不太像主因）
- 最小 disambig（<=1 run + 1 probe）:
  1. 1 run: λ=0.08, seam_weight=0（只去 seam，保留 l1+ssim）。
  2. 1 probe: 对比 λ=0.04 vs 0.08 的 z_pred 分布与 decode residual map（按 timepoint 分层）。
- 若 seam 去掉后几乎不掉点，则 M3 基本可降级。

### Q3. 在“不调小参”约束下，X1-X5 选哪 1-2 个?
结论: APPROVE (主推 X3 + X5)

排序（EV × 可实现性 / 成本）:
1. X3（image_aux + decoder LoRA additive test）
2. X5（cross-dataset/cross-tracer generalization）
3. X1（机制深拆）
4. X2（架构升级）
5. X4（chain redesign）

取舍理由:
- 选 X3: 1 个 7d run 就能回答“V18 还有没有独立价值”，战略信息密度最高。
- 选 X5: 决定论文档次上限（可泛化 vs 单数据集技巧）。
- 不主推 X1: 科学价值高，但在 user 约束下容易被理解为“调子项”。可作为 X3 后补充。
- 不主推 X2: 风险和周期太大，当前不适合作为下半场主线起点。
- 不主推 X4: 会破坏历史可比性，且收益不确定。

### Q4. V18 在 paper 里的角色?
结论: APPROVE 选 (b) 次要 ablation

- 头条必须转为 image_aux schedule。
- V18保留为 “we also tested decoder-side intervention; gains are smaller and condition-dependent”。
- 不建议 (a) 完全删除：保留可增强“我们尝试过替代路径”的可信度。
- 不建议当前就 (c) 主 ablation：在 A4-mid 强信号后，V18 已不应占主轴。

### Q5. 是否漏关键方向 X6+?
结论: APPROVE（补充 X6）

新增 X6: training-time dose curriculum / uncertainty-aware weighting
- 核心: 不是继续扫 image_aux λ，而是按 timepoint 或样本难度自适应分配监督强度。
- 价值: 把“更强像素监督有效”提升为方法论，不停留在单一 λ 常数。
- 成本: 中等（1-2 周实现 + 1 run 验证），风险低于 X2。

### Q6. paper 时间表?
结论: MODIFY

建议双轨:
- 轨道 A（主线，保进度）:
  1. 2 周内完成 X3。
  2. 同步启动 paper outline + Methods + Results 初稿。
  3. 目标: MICCAI 2027 可行。
- 轨道 B（抬上限）:
  1. 若 X5 数据可获取，随后 4-8 周补泛化段。
  2. 若 X5 不可得，转为 stronger mechanism section（轻量 X1 probe）。
  3. 目标: TMI/MedIA 作为并行或后续延展。

### Q7. 偏差审查 B87+?
结论: APPROVE（存在）

- B87 A4-mid hero bias: 把单次大增益过度人格化为“终极答案”。
- B88 anti-tuning dogma: 把所有低成本机制实验都误判为“调参”，导致拒绝必要证据。
- B89 architecture jump bias: 在信号刚出现时过早跳到高风险 X2，牺牲时间确定性。
- B90 narrative collapse bias: 因 A4 强势而过度贬低 V18，失去“替代路径已验证”的说服力。

---

## 6.2 主战略 verdict (单选 + 备选)

主选: Hybrid = X3 + X5
- 先做 X3（1 run，7d）回答 additive/冗余问题。
- 并行准备 X5 数据可达性；可达则上，不卡主线。

备选: Hybrid = X3 + 轻量 X1（1 run + 1 probe）
- 当 X5 数据暂不可得时，用机制证据补论文可信度。

拒绝项说明:
- 不主推 X2 作为当前主线（风险/周期过大）。
- 不主推 X4（可比性破坏风险）。

---

## 6.3 V18 角色判定

选择: (b) 次要 ablation

推荐写法:
- image_aux 是主结果。
- V18 是次要结构性对照：decoder-side intervention 有增益，但小于 image_aux 主杠杆，且需在新监督设定下重判。

---

## 6.4 image_aux mechanism 主推 + 最小 disambig

主推: M1 + M4

最小 disambig experiment:
1. 一个训练 run: λ=0.08 且 seam_weight=0，检验 M3 必要性。
2. 一个分析 probe: 对 λ=0.04/0.08 的 z_pred 分布、reconstruction residual 进行 timepoint 分解对比。

判定规则:
- 若去 seam 影响很小，M3 降级。
- 若 residual 显著缩小且 latent 分布更贴近 decoder manifold，支持 M4。
- 若主要提升集中于高误差区域且跨 timepoint 一致，支持 M1。

---

## 6.5 paper 时间表

- 目标一: MICCAI 2027（主推，进度可控）
- 目标二: TMI/MedIA（作为增强轨道，不阻塞主投稿）

当前应立即启动:
1. Paper outline（当天）。
2. Methods + Experiment protocol 初稿（本周）。
3. 结果章节先写已定结果（V13/V7/A4-low/A4-mid + V18定位），把 X3/X5 作为 pending slots。

---

## 6.6 新偏差 (B87+)

- B87: A4-mid hero bias
- B88: anti-tuning dogma overreach
- B89: premature architecture-pivot bias
- B90: over-pruning of non-headline evidence (V18)

治理建议:
- 任何“转主线”决策必须附带一个 7-14 天内可验证的低风险判别实验（本轮即 X3）。
- 任何“拒绝小实验”决策需说明会损失哪条可发表证据链。

---

## Reviewer B 一句话结论

A4-mid 是可信且足够改变战略重心的信号，但当前最优路径不是立刻大架构跳跃，而是先用 X3 低成本判别 V18 是否仍有独立价值，再视数据可得性推进 X5，把论文从“单点强结果”升级为“可泛化且机制可解释”的完整故事。