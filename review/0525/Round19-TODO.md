# Round 19 — TODO (X1-lite + X3 完成后启动)

- date: 2026-05-26
- branch: foc_lite_hop0
- 触发条件: X1-lite + X3 任一完成且 full-val canonical eval 出来
- 上游: [REVIEW_INTEGRATION_round18_prep_20260525.md](./REVIEW_INTEGRATION_round18_prep_20260525.md) + [codex_x1_x3_design_analysis_20260525.md](./server/codex_x1_x3_design_analysis_20260525.md)

---

## 0. 即时状态 (2026-05-26)

| run | GPU | session | 状态 | 预计完成 |
|---|---|---|---|---|
| X1-lite | GPU1 | round18_x1_gpu1_0525 | training (from-scratch to 160K) | T+7d ≈ 2026-06-01 |
| X3 | GPU3 | round18_x3_gpu3_0525 | training (warmstart V7.best → 170K) | T+1-2d ≈ 2026-05-27 |

健康状态: 两者已通过 +5min smoke check (λ_img=0.08 / λ_kl=0 / resume @160K 都已 verify, no OOM). 详 [codex_x1_x3_design_analysis_20260525.md](./server/codex_x1_x3_design_analysis_20260525.md).

---

## 1. 已记录待办 (按依赖序)

### TODO-1 (X3 完成后立即) — 写 X3 outcome 解读

trigger: codex push `review/0525/X3_image_aux_lora/fullval_eval/artifacts/*.json`

action:
- 提取 X3.best/last NORMAL PSNR_clip3
- 比较 X3 vs A4-mid (绝对值), X3 vs A3 (paired, 仅 λ_img 差), X3 vs V18.step170k (matched-step)
- 按 codex 解读规则给 verdict: **clearly above A4-mid** (additive) / **≈ A4-mid** (redundant) / **< A4-mid** (conflict)
- 若 verdict = redundant or conflict → paper 把 V18 family 完全降为 negative ablation, **取消任何 V19 / X3 v2 计划**

decision rule (Round 18 stop rule + Round 18-Prep M2 confirmed):
- X3 是 1 run hard-stop
- 不允许 X3 extend to 200K, 不论 outcome
- 不允许 X3 v2 (LoRA rank/blocks variant)

### TODO-2 (X1-lite 完成后立即) — 写 X1-lite outcome 解读 + CL1 confound 判定

trigger: codex push `review/0525/X1_lite_l1_only/fullval_eval/artifacts/*.json`

action:
- 提取 X1-lite.best/last NORMAL PSNR_clip3
- 按 codex 解读规则:
  - **within 0.02 dB of A4-mid** → M1+M4 主导, paper 写 "pixel L1 through frozen decoder is sufficient", CL1 confound moot
  - **0.02-0.05 dB below A4-mid** → ambiguous (M2/M3 possibly OR gradient-budget) → **触发 CL1 决策点**
  - **near V7 or below** → l1 单独不够 → 触发 CL1 决策点

### TODO-3 (X1-lite outcome = interior 时触发) — CL1 决策点

**问题**: X1-lite 把 ssim_weight 和 seam_weight 都设 0，**同时** total image_aux gradient 降了 ~35% (ssim 默认权重 0.25 + seam 0.10 = 0.35 vs L1 1.0). 若 X1-lite < A4-mid, **无法区分**:
- 是 SSIM/seam **机制**重要 (要保留这些 component)
- 还是只是 **gradient budget** 不够 (L1 单独够, 只需要权重补偿)

**3 个选项**:

#### 选项 A (推荐, user 已选) — 不启新 run, 写 paper limitation

若 X1-lite outcome 落在干净区域 (within 0.02 dB OR near V7), CL1 是 moot. 若落在 interior:
- paper 写 "X1-lite ablates SSIM and seam components but also reduces total image_aux gradient by ~35%; we cannot fully separate component-mechanism contribution from gradient-budget contribution within this paper"
- limitation 进 paper "Methods" 或 "Discussion" 章节
- cost: 0 GPU, 仅写作工作
- 时间: 立即可做

#### 选项 B — 启 X1-v2-balanced (gradient-budget control)

clone X1-lite yaml + `loss.image_aux.l1_weight: 1.0 → 1.35` (补偿 ssim+seam 移除的总量), 其它一致.
- 若 X1-v2-balanced ≈ A4-mid → 真 confirmed L1 sufficient (M1+M4 强证据)
- 若 X1-v2-balanced < A4-mid → SSIM/seam **机制**真的重要 (M2/M3 部分必要)
- cost: 7d × 1 GPU (slot 3, 仍在 ≤3 并行约束内)
- 风险: **打破 Round 18 X1-lite "1 run hard-stop" 共识**, B91 sunk-cost framing 风险, 应明确为 "single follow-up disambiguation run, not a sweep"

#### 选项 C — 不启 + 不写 limitation, 接受默认归因

若选 C, paper 直接写 "SSIM and seam are required components". user 同意接受这个不严格归因.
- cost: 0
- 风险: reviewer 在 paper review 时可能挑出 confound, 要求 rebuttal 跑 X1-v2 (届时 deadline 压力大)

**user 已选**: **选项 A**. 等 X1-lite outcome, 若落 interior region 在 paper 写 limitation.

### TODO-4 (任一 outcome 出来后) — 启动 paper outline + Methods skeleton 第 1 稿

paper 章节框架 (Round 18 集成 §5.2 / 5.3 已签):
- **Introduction**: low-dose PET denoising, latent flow as alternative to direct pixel diffusion
- **Methods**:
  - Sec X: transport DiT + frozen RAE decoder framework
  - Sec X+1: image_aux loss (decoder-aware pixel supervision via frozen decoder backprop)
  - Sec X+2: decoder LoRA + KL pullback (V18 family, **降级为 ablation**)
- **Results**:
  - 主图: 4-point image_aux response curve (V13 / A4-low / V7 / A4-mid)
  - 主表: V13 / V14 / V7 / A4-low / A4-mid full-val canonical PSNR
  - ablation 表: V18 / V18.last / A3 / X3 (decoder-side interventions)
  - mechanism ablation: A4-mid vs X1-lite (component attribution)
- **Discussion**:
  - mechanism story: pixel L1 through frozen decoder is the dominant lever
  - limitations: single-dataset, slice-level only, image_aux components not fully decoupled (CL1)
- **Future Work**:
  - decoder-aware latent metric / pullback loss (codex 提议)
  - cross-tracer / cross-dataset generalization (X5)
  - dose-conditional transport (codex X7)

### TODO-5 (X1-lite + X3 都完成后) — Round 19 战略整合

trigger: 两个 outcome 都 push 完成
- 写 `REVIEW_INTEGRATION_round19_20260601.md` (日期视实际完成日)
- 整合 X1-lite mechanism verdict + X3 additivity verdict + 任何 CL1 选项决策
- 决定项目是否进入纯写作期 (no more training) 还是需要 X5/X6/未列出实验

### TODO-6 (paper draft 启动后) — 启动 cross-AI peer review on paper draft

复用 Round 14/15/16/17/18 reviewer 流程, 但 substrate 换成 paper draft 而不是 task md.

---

## 2. 已签约束 (Round 19 不允许触碰)

| ❌ | 理由 |
|---|---|
| 启动 X1-lite v2 (除非选项 B 被显式启用) | Round 18 stop rule, 4/4 reviewer 共识 |
| 启动 X3 v2 / extend X3 到 200K | Round 18-Prep stop rule, 3/3 reviewer 共识 |
| 启动任何 V19 / V18-clean / V18-rank-sweep | Round 16-18 一致拒 |
| 启动 image_aux λ ∉ {0.02, 0.08} 新 A4 变体 | Round 17-Slots stop rule |
| 启动 X2 (architecture pivot) | Round 18 4/4 拒 immediate next step |
| 启动 X4 (chain redesign) | Round 18 4/4 拒, break baseline |
| 启动 V14b/c (多 seed) | Round 17-Stats slice-level only |
| 在 paper 主图保留 V18 为头条 | Round 18 4/4 选 V18 (b) secondary ablation |
| paper 主文使用 patient-level p-value | Round 17-Stats patient_id 不可恢复 |

---

## 3. 偏差跟踪 (B87-B95 + 新增)

Round 18 + Round 18-Prep 累积偏差全清单 (claude 自审常犯, 任何新 round 必查):

| ID | 偏差 | manifestation |
|---|---|---|
| B87 | A4-mid 单 seed 信号过度浪漫化 | claude 倾向把 +0.113 dB 当"项目救星" |
| B88 | image_aux fetish | 其它方向被低估 |
| B89 | "不调小参" 过度教条化 | mechanism ablation 被误归类 tuning |
| B90 | Architecture pivot optimism | X2 失败率高被低估 |
| B91 | V18 sunk-cost | X3 可能被悄悄变 V18.v2 |
| B92 | PSNR monoculture | 无 clinical metric |
| B93 | self-review tunnel vision | claude 验 yaml 不验 launcher shell |
| B94 | matched-comparison overclaim | X3 vs A3 "只差 λ" 实际多字段差 |
| B95 | health-check theater | grep 看起来全面但读错 source |
| **B96 (新, codex CL1)** | **gradient-budget vs component-mechanism confound** | **X1-lite 关 SSIM/seam 同时降 gradient 总量, 归因不严** |

---

## 4. Round 19 trigger 时间表预估

| 事件 | 预计日期 |
|---|---|
| X3 完成 + eval | 2026-05-27 |
| X1-lite 完成 + eval | 2026-06-01 |
| Round 19 集成启动 (≥ 2 outcome) | 2026-06-01 |
| paper outline 第 1 稿 | 2026-06-03 |
| paper Methods skeleton | 2026-06-08 |
| MICCAI 2027 abstract 投递 | TBD (deadline ~2026-12) |
