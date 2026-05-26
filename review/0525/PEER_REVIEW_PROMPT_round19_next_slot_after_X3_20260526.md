# Peer Review Round 19-Pre — Next Slot After X3 Non-Additive Result

- date: 2026-05-26
- branch: foc_lite_hop0
- 主审对象: **X3 已完成且非 additive; X1-lite 仍在跑; 当前空出 1 个训练 slot, 下一步是否启动一个新实验以及启动哪个**
- 上游:
  - [Round19-TODO.md](./Round19-TODO.md) TODO-1 / TODO-1b
  - [X3_FULLVAL_EVAL_REPORT_20260526.md](./X3_image_aux_lora/fullval_eval/X3_FULLVAL_EVAL_REPORT_20260526.md)
  - [codex_x1_x3_design_analysis_20260525.md](./server/codex_x1_x3_design_analysis_20260525.md)
  - [REVIEW_INTEGRATION_round18_20260525.md](./REVIEW_INTEGRATION_round18_20260525.md)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

X3 已经完成, 结果明确 non-additive. X1-lite 仍在训练. 用户确认: X5 / X6 当前不可用 (无外部数据 / 无 clinical metadata), 但现在有一个训练 slot 空出来.

**本轮唯一问题**:

> 在不延误 X1-lite 和 paper draft 的前提下, 是否应该使用空出的 1 个训练 slot? 如果用, 跑哪个实验?

**本轮不审**:
- X3 结果是否有效 (已由 canonical full-val report 确认)
- X1-lite 是否值得继续 (它仍是 Round 18 4/4 共识必做)
- X5 / X6 (用户已确认当前不可用)
- V18 是否做 headline (Round 18 + X3 结果已否定)

---

## 1. X3 结果事实锚点

来源: `review/0525/X3_image_aux_lora/fullval_eval/X3_FULLVAL_EVAL_REPORT_20260526.md`

| run | NORMAL PSNR_clip3 | Δ vs V7 | Δ vs A4-mid |
|---|---:|---:|---:|
| V7.best | 36.7810 | 0 | -0.1130 |
| A4-mid.best | 36.8939 | +0.1130 | 0 |
| V18.best | 36.8112 | +0.0302 | -0.0828 |
| V18.last | 36.8426 | +0.0617 | -0.0513 |
| X3.best | 36.8104 | +0.0295 | -0.0835 |
| X3.last | 36.8288 | +0.0478 | -0.0651 |

X3.last recovers only ~42.3% of A4-mid's gain over V7. It remains in V18 range, not A4-mid range.

**Locked interpretation**:
- X3 is **non-additive** in the A3-matched 10K warmstart window.
- V18/LoRA remains secondary ablation only.
- No X3-v2 / X3-extend / V19 / V18-clean unless user explicitly overturns Round 18 stop rule after new peer review.

---

## 2. 当前开放 slot 的候选实验

### Option A — A4-mid-seed1337 (recommended candidate)

**Design**: clone A4-mid (`λ_img=0.08`, no LoRA, full image_aux), change only `seed: 42 → 1337`, from-scratch to 160K.

**Question**: Is the main paper headline A4-mid (+0.113 dB over V7) robust to seed?

**Why it matters**:
- A4-mid is now the paper headline.
- Patient-level p-value impossible; seed robustness is the cleanest available robustness evidence.
- We already have V14 (V7 + seed=1337) showing V7 is seed-stable: V14 − V7 = -0.0004 dB. But A4-mid may or may not be seed-stable.

**Outcome interpretation**:
- If A4-mid-seed1337 ≈ A4-mid (within 0.02 dB) → headline robust; paper much stronger.
- If it remains > V7 + 0.05 but below A4-mid by >0.02 → effect real but magnitude seed-sensitive.
- If it falls to V7 range → A4-mid was seed luck; paper must downgrade.

**Cost**: 7d × 1 GPU. Does not interfere with X1-lite.

### Option B — X1-v2-balanced (CL1 gradient-budget control)

**Design**: clone X1-lite (`λ=0.08`, ssim=0, seam=0), set `loss.image_aux.l1_weight: 1.0 → 1.35` to compensate for removing SSIM/seam nominal gradient budget.

**Question**: If X1-lite underperforms A4-mid, is it because SSIM/seam mechanisms matter, or because total gradient budget was reduced?

**Why it matters**:
- Codex CL1: X1-lite removes SSIM/seam and reduces total nominal image_aux gradient by ~35%.
- If X1-lite lands in the interior region, X1-v2-balanced resolves the ambiguity.

**Risk**:
- Running it **before** X1-lite result may be unnecessary.
- It partially breaks Round 18 X1-lite hard-stop.
- It is close to component/weight tuning, which user dislikes.

**Cost**: 7d × 1 GPU.

### Option C — No new training, wait for X1-lite

**Design**: keep slot idle, start paper outline now, wait for X1-lite.

**Question**: Does governance value / stop-rule preservation outweigh using the idle slot?

**Why it matters**:
- Prevents scope creep.
- X1-lite outcome may make X1-v2 unnecessary.
- Paper writing can begin immediately without another run.

**Cost**: 0 GPU, but loses one 7-day opportunity window.

### Option D — X3-extend to 200K

**Design**: resume X3.last@170K, continue to 200K.

**Question**: Does LoRA require long-window training to become additive under `λ_img=0.08`?

**Why it matters**:
- X3 10K only rules out short-window additivity.
- V18.last improved over V18.best over longer training.

**Risk**:
- Directly violates X3 hard-stop.
- Very likely V18 sunk-cost revival.
- Even if it gains +0.03, it still may not beat A4-mid enough to matter.

**Cost**: 5-7d × 1 GPU.

---

## 3. 给 reviewer 的 7 个问题

### Q1 — 空 slot 是否应该使用?

请明确:
- Should we use the freed X3 slot for one more experiment now?
- Or preserve stop rules and wait for X1-lite + paper draft?

给主 verdict 和理由.

### Q2 — Option A (A4-mid-seed1337) 是否是最高 EV?

请评估:
- seed robustness 对 paper 主结果的价值是否 > mechanism disambiguation 的价值?
- A4-mid 作为 headline 是否必须有 seed replicate?
- V14 已证明 V7 seed-stable, 是否足以推断 A4-mid seed-stable? 还是必须直接测?

### Q3 — Option B (X1-v2-balanced) 是否应该提前跑?

请评估:
- CL1 是否严重到值得现在就跑 balanced control?
- 如果 X1-lite outcome 非 interior, X1-v2 是否会变成浪费?
- 是否应该等 X1-lite 结果再决定, 即使那会推迟 7 天?

### Q4 — Option C (no new training) 是否过于保守?

请评估:
- idle slot 是否有正 option value (buffer / paper time / avoid scope creep)?
- 还是在 GPU 已可用且 paper 主结果仍 single-seed 的情况下, 不跑就是浪费?

### Q5 — Option D (X3-extend) 是否应直接排除?

请评估:
- X3 non-additive 是否已经足够判定 V18 family 对当前 paper无主线价值?
- 继续到 200K 是否只是 V18 sunk-cost?
- 有没有 scenario 让 X3-extend 是合理 next-slot use?

### Q6 — 是否漏掉更好的 Option E?

请提出最多 1 个新 candidate, 必须满足:
- 1 个 run
- 不需要外部数据 / clinical metadata
- 不违反 Round 18 的 no-small-tuning spirit
- 不延误 X1-lite / paper draft

### Q7 — 偏差审查 (B97+)

请检查本 prompt 是否引入:
- A4-mid seed robustness fetish (过度重视 replicate)
- CL1 overreaction (过早启动 X1-v2)
- V18 sunk-cost (给 X3-extend 留口子)
- slot-utilization bias (因为有空 GPU 就想用)

---

## 4. 输出格式

每位 reviewer 独立产出 markdown:

### 4.1 Q1-Q7 逐条回答 (APPROVE / MODIFY / REJECT)

### 4.2 主 verdict (单选 + 备选)
- A: A4-mid-seed1337
- B: X1-v2-balanced
- C: No new training, wait for X1-lite
- D: X3-extend to 200K
- E: 自定义 (只能一个)

### 4.3 If choose a run, minimal design spec
列 4-6 个 yaml diff 字段 + stop rule.

### 4.4 If choose no run, paper/task action
说明 idle slot 期间应该做什么 (paper outline / analysis / monitoring).

### 4.5 新偏差 (B97+)

---

## 5. 角色提示

这不是大方向战略轮. 大方向已经是 image_aux 主线 + V18 secondary + X1-lite mechanism. 你只是在判断一个空出来的 GPU slot 是否该用.

关键 trade-off:
- A4-mid-seed1337 强化 paper headline robustness
- X1-v2-balanced 强化 mechanism attribution, 但可能等 X1-lite 后才需要
- No run 守 stop rule, 但放弃 7d 机会
- X3-extend 诱人但可能是 sunk-cost

请直接给选择, 不要把四个方案平均放回 user 手里.
