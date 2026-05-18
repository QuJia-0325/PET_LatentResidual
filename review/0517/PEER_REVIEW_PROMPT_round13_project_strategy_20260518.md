# Peer Review Round 13 — Project Strategy Re-evaluation (V18 KL drift inverse + 历史 PSNR 演进)

- date: 2026-05-18
- branch: foc_lite_hop0 (commit 9c5c97b, KL drift 数据已 merge)
- 主审对象 (3 个):
  1. **V18 KL drift inverse-direction finding** (Round 12 假设被反驳)
  2. **历史 PSNR 演进全景** (V6 → V18 涨 +0.07 dB, 整个项目优化空间几乎耗尽?)
  3. **项目战略路径选择** (continue transport / continue decoder / pivot to architecture / pivot to data / paper as-is)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**
- 触发: user 观察 "transport 没大涨, decoder 也没大涨" + Round 12 KL drift 反预期发现

---

## 0. 本轮范围与边界

Round 1-11 审 design/execution/audit prep (markdown), Round 12 审 V18 final substrate. **Round 13 首次审 project strategy** — 不是某次实验对错, 是**整个 PET_LatentResidual 框架是否值得继续推**.

**本轮审**:
- V18 KL drift signed-negative 发现的物理意义 (是 V18 改进 GT manifold 还是测量伪影?)
- 历史 PSNR 演进 (V6 → V18) 累积 ΔPSNR 是否真如此 marginal?
- 项目战略 5 选项: 继续 transport 优化 / 继续 decoder 优化 / pivot architecture / pivot data / 直接 paper as-is + 接受 limitation
- 是否项目本身的 ceiling 已逼近, 应转 follow-up 方向

**本轮不审**:
- Round 1-12 已签内容 (设计 / 执行 / audit prep / V18 outcome)
- 阶段 A v3 是否 push (与 strategy 独立)
- KL drift probe 代码质量 (Round 12 reviewer 已 verify)

---

## 1. 历史 PSNR 演进 (本地 grep verify)

### 1.1 V6 → V18 累积变化 (canonical PSNR_clip3, n=7403, NORMAL chain)

来源: PLANF_FINAL_ANALYSIS_20260516.md line 68-71 + V18 final eval

| 版本 | NORMAL PSNR_clip3 | Δ vs V6_NOISE | 备注 |
|---|---:|---:|---|
| V6_NOISE.best (step 156400) | **36.7437** | (baseline) | seed=1337, V6 step_weights, image_aux on |
| V6_NOISE.last (step 160000) | **36.7550** | +0.011 | |
| V8.best (step 160000) | **36.4729** | **-0.271** | **REGRESSION**: Grönwall + image_aux off |
| V7.best (step 160000) | **36.7810** | +0.037 | Grönwall + image_aux on (主线) |
| V7.last (step 160000) | **36.7810** | +0.037 | (V7 best=last, plateau early) |
| V18.best (step 165000) | **36.8112** | +0.068 | rank=32 LoRA, +5K from V7 |
| V18.last (step 200000) | **36.8426** | **+0.099** | +40K from V7 |

### 1.2 关键差分 (按 paired comparison)

| 比较 | NORMAL Δ | 解读 |
|---|---:|---|
| V8 → V7 (image_aux + step_weights 联合 effect) | **+0.308 dB** | Plan F 主成果, 但 PEER_REVIEW 已警告这是 confound (image_aux + step_weights 一起改, 不可分离) |
| V6_NOISE → V7 (transport 联合优化) | **+0.026 dB** | seed=1337 → 42 + V6 step_weights → Grönwall + image_aux 全开 |
| V7 → V18.best (decoder LoRA, +5K step) | **+0.030 dB** | sub-PARTIAL |
| V7 → V18.last (decoder LoRA, +40K step) | **+0.062 dB** | PARTIAL 下沿 |
| **V6_NOISE → V18.last (项目总优化)** | **+0.088 dB** | (~0.09 dB 总优化空间, 1 年工作量) |

### 1.3 透视

- V8 → V7 +0.308 dB 是 **联合 effect**, 不是单 image_aux 贡献 (V13 设计就是为分离, 但 V13 尚未 launch)
- V8 是 **regression** (相对 V6_NOISE -0.27 dB), 说明 "Grönwall step_weights 但去掉 image_aux" 比基线还差
- V6_NOISE → V18 总变化 **+0.088 dB** (V7 +0.04 + V18 +0.05 last-vs-V7)
- 当前 attackable gap (decoder ceiling ~46.6 dB - transport ~35.4 dB = ~11 dB) 实测**只能吃到 ~0.1 dB** (0.9%)

→ 这是项目 **ceiling 信号**: latent transport + frozen decoder 框架在 PET 数据上的总优化上限约 ~0.1 dB?

---

## 2. V18 KL drift 反预期发现 (再审)

### 2.1 数据 (本地 verify, n=7403, NORMAL chain)

来源: kl_drift_20260518_123834/KL_DRIFT_REPORT.md (codex commit 9c5c97b)

| ckpt | step | PSNR(decode(z_GT), x_target) on NORMAL |
|---|---:|---:|
| V7.best | 160000 | **52.6341 dB** |
| V18.best | 165000 | **52.7314 dB** |
| V18.last | 200000 | **52.7980 dB** |

**Signed drift (V7 - V18.X)**:
- V18.best: **-0.0973 dB** (V18 better on GT manifold)
- V18.last: **-0.1639 dB** (V18 even better)

**全部 chain (D20/D10/D4/NORMAL) signed drift 全 negative**.

### 2.2 解读 (3 个候选)

**A. V18 decoder LoRA 真在 GT manifold 改进** (claude Round 12 整合解读):
- KL pullback `use_pred_latent=true` **不**让 decoder 漂离 GT manifold
- B9 假设被反驳
- V18 decoder 有 +0.10 dB GT 余量, 但 transport pipeline 吞噬 70%, 只剩 +0.030 dB
- Stage C bottleneck 在 transport 而非 decoder

**B. KL drift probe 测量伪影** (alternative):
- probe 直接 `decode_crop(z_GT)`, 但 V18 训练时 KL pullback 用 `decode(z_pred)`, 不是 `decode(z_GT)`
- `z_GT` 路径在 V18 训练中 from-scratch 没见过 (KL only ramp z_pred path)
- V18 decoder 在 unseen z_GT 上的 "+0.10 dB" 可能是 LoRA 通用 capacity 增加 (rank=32 trainable +589K), 不是 KL pullback alignment 工作
- 同等 frozen V7 + decoder rank=32 (无 LoRA-finetune, 纯 random init noise) 可能也涨 +0.05 dB (待 control 实验)

**C. canonical evaluator artifact** (alternative):
- decode_crop 函数实现 (clip [0,3] + crop) 可能 V18 与 V7 路径有微差
- `model.rae.decode(z_GT)` 是否 V7 / V18 调相同 path? 还是 V18 因 LoRA wrap 走 LinearWithLoRA 路径多了一步 (e.g. dropout / activation 略不同)
- +0.10 dB 量级在 evaluator noise 内?

### 2.3 关键 follow-up 问题

- 如果 A 真, 应集中 transport-side intervention (Stage C 选项 D)
- 如果 B 真, decoder LoRA 是 generic capacity 增加, 与 KL pullback 设计无关, V18 framework 失效
- 如果 C 真, +0.10 dB 是 noise, V18 与 V7 在 GT manifold 实质 tie, 整个 V18 框架 marginal

---

## 3. 给 reviewer 的 7 个问题

### Q1 — V18 KL drift inverse 真实物理意义 (3 候选选哪个?)

请评估 §2.2 A/B/C 三个候选, 给主推 + 反对理由:
- 是否需要新 control 实验来 disambiguate (e.g. V18 rank=32 但 lambda_kl=0 / V18 rank=32 from scratch random init no LoRA-train)?
- decode_crop 是否真 deterministic / V7 V18 同 path? (需读 model_first_hop.py `decode_crop` 代码)
- A 是 claude 的 default 解读, 你是否同意?

### Q2 — 整个项目优化空间是否已逼近 ceiling?

V6_NOISE → V18.last NORMAL +0.088 dB (1 年工作). attackable gap ~11 dB 但实际只吃 ~1%.

请评估:
- 0.088 dB 是 PET latent transport + frozen decoder framework 的真实 ceiling, 还是有 unexplored space?
- 如果是 ceiling, 应该:
  - (a) Pivot architecture (跳出 latent transport, 改 diffusion / score-based / end-to-end pixel)
  - (b) Pivot data (V6/V7 split 不够, 需更多 PET 数据)
  - (c) 接受 +0.088 dB 写 paper "PET latent transport feasibility study"
  - (d) Continue tinkering (低效但谨慎)
- 如果有 unexplored space, 哪个方向 EV 最高?

### Q3 — V8 regression -0.27 dB 是否反向证明 image_aux 是项目支柱?

V8 = "Grönwall step_weights + image_aux off" → -0.27 dB vs V6_NOISE. 这是否说明:
- image_aux 是项目 -0.27 dB 的隐性贡献 (V13 应该能 isolate)
- Grönwall step_weights 单独无用 / 害? (V8 ≠ V6 setup 差异除 step_weights 还有 V6 vs Plan F train config 全差异)
- V13 (Grönwall + image_aux off) 仍未 launch — 是否**现在**就该 launch 来分离, 然后再决策项目方向?

### Q4 — Decoder LoRA 在 GT manifold 改进 +0.10 但 transport 吞噬 70%, 含义?

如果 §2.2 A 真:
- transport pipeline error 占总 error 92% (Round 7 §2.4 给的 MSE 分解)
- decoder LoRA 在 8% 部分 (decoder ceiling) 改进 +0.10 dB → 这本身是 1.25 dB / 8% scaled-up 等价的 decoder capacity
- transport-side intervention 真要解 92% 部分, 需要 backbone 重设计 / data augmentation / multi-step refinement, 不是 hyperparam tinkering
- 当前 V7 transport (latent DiT) 是否到性能上限?

请评估 transport-side intervention 的 5 个候选优先级:
- (A) backbone scale-up (DiT base → large)
- (B) backbone architecture change (DiT → flow matching / consistency models)
- (C) data 增加 (额外 PET split)
- (D) image_aux schedule tuning (image_aux 已饱和 V7, 但可能 alternative schedule 释放 ~0.05 dB?)
- (E) multi-step refinement (chain length / sampling steps 增加)

### Q5 — Paper 现状是否值得写?

如果项目真 ceiling 是 +0.088 dB, paper narrative:
- "PET latent transport: feasibility study" (low-ambition, 仍 publish)
- "PET latent transport: a thorough negative result" (honest)
- "PET decoder LoRA: a partial improvement framework" (V18 +0.06 last-vs-best)

请评估:
- 哪个 narrative 最 publishable?
- conference (MICCAI / ISBI / IPMI) vs journal (Med Image Anal / TMI) 哪个适合?
- 是否需要 V13 + V14 + V18-clean 全跑完才能 paper-ready?

### Q6 — Stage C 真应该走哪条?

按 Round 12 claude 推荐 D (transport-side intervention), 但 §2.2 B/C 候选未排除. 请评估:
- A: V18-clean (use_pred_latent=false) — KL drift 反预期后, **预期 V18-clean ≤ V18** (B9 假设被反驳), 不推
- B: V18-r64 / V18-last4 sweep — GT 余量仅 +0.10 dB, sweep marginal
- C: paper draft + V18 PARTIAL limitation — 可行但放弃后续优化
- D: transport-side intervention — 需新 design + ~7 天 GPU
- **E (新)**: V13 + V14 必须跑 (V7-V8 disambiguation + d_pure 真值), 在决策 D vs C 之前必须有 V13/V14 数据

是否应**先 launch V13 + V14** (~2 个 7 天 task, slot 2+3) 再做 Stage C 决策? 还是 V13/V14 outcome 对 project strategy 影响小, 跳过?

### Q7 — Round 13 + 之后的 review cadence?

Round 1-12 共 12 轮 review, 大量精力. 是否到了"减 review, 加实验"的转折点?

请评估:
- Round 13 之后 (e.g. V13/V14 完成后), 是否应该减少 reviewer 介入, user + claude 直接判定?
- 或继续 reviewer-heavy mode (substrate 上, reviewer 比 claude 自纠更可靠, Round 12 已 100% 证明)?
- standing rule #N 立: "战略转折点 (e.g. ceiling 信号) 必须 reviewer 介入" vs "战术细节 (e.g. yaml 字段) user+claude 直接"?

---

## 4. 资料目录

### 4.1 本轮主审对象

- [KL_DRIFT_REPORT.md](PET_LatentResidual/review/0517/V18_decoder_lora/kl_drift_20260518_123834/KL_DRIFT_REPORT.md) (V18 KL drift 反预期数据)
- [KL_DRIFT_SUMMARY.json](PET_LatentResidual/review/0517/V18_decoder_lora/kl_drift_20260518_123834/KL_DRIFT_SUMMARY.json)
- [KL_DRIFT_PER_SLICE.csv](PET_LatentResidual/review/0517/V18_decoder_lora/kl_drift_20260518_123834/KL_DRIFT_PER_SLICE.csv) (88837 行)
- [probe_v18_kl_drift.py](PET_LatentResidual/tools/probe_v18_kl_drift.py) (390 行新 script)

### 4.2 历史 PSNR context

- [PLANF_FINAL_ANALYSIS_20260516.md](PET_LatentResidual/review/0516/PLANF_FINAL_ANALYSIS_20260516.md) (V6_NOISE / V7 / V8 baseline)
- [V18_FINAL_RESULTS_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md) (V18 final canonical)
- [V18_METRIC_CORRECTION_REPORT_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_METRIC_CORRECTION_REPORT_20260518.md) (45.6→36.8 修正)

### 4.3 项目设计 context (本轮不重审, 仅 context)

- [V18_design_rationale.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_design_rationale.md) (V18 设计原因)
- [REVIEW_INTEGRATION_round12_20260518.md](PET_LatentResidual/review/0517/REVIEW_INTEGRATION_round12_20260518.md) (Round 12 整合, 3×3 矩阵)
- [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](PET_LatentResidual/review/0517/NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) (三阶段路线)

### 4.4 代码 context (用于 Q1 verify)

- [model_first_hop.py `decode_crop`](PET_LatentResidual/pet_lr/model_first_hop.py) (V7/V18 是否同 path)
- [decoder_lora.py](PET_LatentResidual/pet_lr/decoder_lora.py) (LinearWithLoRA 实现, init_scale_zero 行为)
- [eval_first_hop_224_clip3.py](PET_LatentResidual/eval_first_hop_224_clip3.py) (canonical evaluator)

---

## 5. 输出格式

每位 reviewer 独立产出 markdown:

### 5.1 7 个问题逐条回答 (APPROVE / MODIFY / REJECT)

### 5.2 战略推荐 (主 verdict)
明确选 1 个 Stage C 主路径:
- A: V18-clean
- B: V18-family sweep (rank/blocks)
- C: paper draft as-is + V18 PARTIAL limitation
- D: transport-side intervention (具体哪个 sub-option 优先?)
- E: 先 launch V13+V14 disambig, 再决策
- F: pivot to architecture (跳出 latent transport framework)
- G: pivot to data (PET 数据扩展)
- H: 项目 terminate / paper as feasibility study

### 5.3 V18 KL drift 解读 (Q1)
- §2.2 A/B/C 哪个真? 是否需新 control 实验?

### 5.4 项目 ceiling 判断 (Q2)
- 0.088 dB 是 ceiling 还是有 unexplored?

### 5.5 (可选) 新偏差 (B39+)
若发现 claude 在本 prompt 引入新 anchor / cherry-pick / 简化 narrative.

---

## 6. 约束与提醒

- **本轮不重审** Round 1-12 已签内容
- **不假设** 你能跑代码 — 但可读 CSV / JSON artifact / grep markdown
- **特别关注**:
  - Q1 §2.2 B/C alternative: V18 KL drift 反预期可能不是 "decoder LoRA 改进", 而是 LoRA generic capacity / evaluator artifact
  - Q4 transport-side intervention: 92% error 在 transport, decoder 8% 已被吃了 1.25 dB / scaled, 真要解需 transport-side 重设计
  - Q5 paper narrative: 是否承认 +0.088 dB ceiling 是诚实但 publishable?
- **优先质疑**:
  - claude default 解读 §2.2 A 是否 anchor 到"decoder LoRA 是有效设计" (sunk cost bias)
  - 历史 +0.088 dB 真累积优化, 还是 confound (V8 regression 暗示 image_aux 是支柱, V7 不是 transport 设计胜利)

---

## 7. Standing constraints

1. ≤ 3 并行训练任务
2. V18 不动 (已完成)
3. V18b 永久撤销
4. 阈值不改 (Round 7 锁定)
5. user 决策 (R7/R8/R9/R10/R12) 不可推翻
6. Round 13 是战略 review, 不是 execution review (区别于 Round 1-12)
7. 不审 audit DRAFT 是否 release (user 已请求, 但 release 时机依 Round 13 战略 outcome)
8. 不审 v3 task md 是否 push (已 commit 7486776 待 push; 但 Round 13 战略可能改变是否需 V13 launch)
9. **新**: Round 13 verdict 必须 grep verify 历史 PSNR 数字, 不允许凭叙事 anchor

---

## 8. 本轮元说明

| 轮次 | 范围 | substrate | 出处 |
|---|---|---|---|
| 1-7 | design | spec/yaml/历史 | claude+user+reviewer |
| 8-11 | execution v1→v3 | markdown | reviewer focus |
| 12 | V18 substrate | PSNR数字 | substrate review |
| **13** | **project strategy** | **历史 PSNR 全景 + KL drift 反预期 + project ceiling 信号** | **战略 review** |

预期 Round 13 outcome:
- 最佳: 3/3 共识 Stage C 路径 (e.g. 全选 E = V13/V14 先 launch, 或 D = transport intervention)
- 中等: 3/3 verdict 分歧但 ceiling 判断一致 (e.g. 都说项目 ceiling 已近, 但路径不一)
- 最差: 3/3 各推不同战略, user 必须强决策
