# Peer Review Round 12 — V18 Final Results Interpretation (Claude self-correction included)

- date: 2026-05-18
- branch: foc_lite_hop0 (本地已 merge gitee/foc_lite_hop0; v3 task md 已 commit 未 push)
- 主审对象:
  1. **V18 final result interpretation** (canonical PSNR_clip3 vs V7 baseline 的判定)
  2. **claude self-correction**: 我刚才在第一次解读 V18 数据时犯了 **B23 同型偏差** (用 V7 D20 = 35.44 dB 当 NORMAL baseline, 错算 ΔPSNR = +1.37 dB; 实际 V7 NORMAL = 36.78 dB → ΔPSNR = +0.06 dB)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿** + **不互见 claude 修正**
- 目标: 独立判定 V18 outcome (SUCCESS / PARTIAL / KILL), 验证 claude 自纠是否正确, 决定阶段 C 路径

---

## 0. 本轮范围

Round 1-11 在 V18 数据出来**之前**审 design / execution / audit prep, 0 代码改动. **Round 12 是 V18 数据出来后首次 substrate review** — 审的不是 markdown 文档, 是真实 PSNR 数字 + 判定逻辑.

**本轮只审**:
- V18 final PSNR_clip3 数字与 V7 baseline 对比是否正确
- claude self-correction (B23 同型) 是否真自纠正确, 还是又造新 bias
- V18 PARTIAL outcome 下阶段 C 决策路径选择 (按 NEXT_STAGE 3×3 矩阵)
- V18 metric correction 报告 (45.6 dB → 36.8 dB) 是否可信

**本轮不审**:
- Round 1-11 已签内容 (设计 / 执行 / audit prep)
- 阶段 A v3 是否 push (与 V18 outcome 独立)
- audit DRAFT 是否 release

---

## 1. 关键数据 (codex 服务器实测, 已 merge 到本地)

### 1.1 V18 canonical full-val PSNR_clip3 (n=7403)

| ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| best.pt | 165000 | 35.4382 | 35.8346 | 36.3944 | **36.8112** |
| last.pt | 200000 | 35.4212 | 35.8439 | 36.4124 | **36.8426** |

来源: [V18_FULLVAL_EVAL_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/fullval_eval_20260518/V18_FULLVAL_EVAL_20260518.md)

### 1.2 V7 baseline (历史 full-val, n=7403)

| ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7 best | 160000 | 35.4354 | 35.8194 | 36.3736 | **36.7810** |
| V7 last | 160000 | 35.4354 | 35.8194 | 36.3736 | **36.7810** |

来源: [PLANF_FINAL_ANALYSIS_20260516.md](PET_LatentResidual/review/0516/PLANF_FINAL_ANALYSIS_20260516.md) line 68-69 + [PEER_REVIEW_PROMPT_image_aux_direction_20260516.md](PET_LatentResidual/review/0516/PEER_REVIEW_PROMPT_image_aux_direction_20260516.md) line 79

### 1.3 ΔPSNR(V18 vs V7) 实测

| chain | V18 best | V18 last | V7 best | Δ(V18.best - V7.best) | Δ(V18.last - V7.best) |
|---|---:|---:|---:|---:|---:|
| D20 | 35.4382 | 35.4212 | 35.4354 | +0.0028 | -0.0142 |
| D10 | 35.8346 | 35.8439 | 35.8194 | +0.0152 | +0.0245 |
| D4 | 36.3944 | 36.4124 | 36.3736 | +0.0208 | +0.0388 |
| **NORMAL** | **36.8112** | **36.8426** | **36.7810** | **+0.0302** | **+0.0616** |

### 1.4 预注册阈值 (V18_design_rationale §2.3, Round 7 user 签 "全推荐" 不改; Round 12 补 KL drift 维度)

**1D 主阈值 (NORMAL ΔPSNR best-vs-best)**:

| 判定 | 阈值 |
|---|---|
| PRIMARY SUCCESS | ≥ +0.30 dB |
| PARTIAL | [+0.05, +0.30] dB |
| KILL (formal) | ΔPSNR < +0.05 dB **AND** KL drift > 0.05 (§2.3 row 4 原双触发) |
| sub-PARTIAL gray | ΔPSNR < +0.05 dB **AND** KL drift 未测 / < 0.05 (灰区, 需 user 决策) |

**重要** (Round 12 B38 修订): 原 §2.3 row 4 是 2D 阈值 (ΔPSNR + KL drift), Round 12 初稿简化为 1D 是错的, 丢了灰区选项. KILL 需 2D 双触发 confirm. 当前 V18 best-vs-best Δ=+0.030 < +0.05 ✓条件 1, 但 KL drift 未测 → **formal KILL pending KL drift**.

---

## 2. claude 自纠史 (透明披露, Round 12 修订后)

我第一次解读 V18 数据时 **犯了 B23 同型偏差** (premature claim) + **B36 同型** (Comparison-label mismatch). 三轮错误轨迹:

| 时刻 | 错误 | 修正 |
|---|---|---|
| 第 1 次 (V18 数据刚出时) | V7 NORMAL ≈ 35.44 dB → ΔPSNR=+1.37 dB → SUCCESS | 35.44 是 V7 **D20**, V7 NORMAL 实是 **36.78 dB** |
| 第 2 次 (Round 12 prompt 初稿) | "+0.06 dB → PARTIAL (**best vs best 比较**)" | +0.06 是 **last-vs-best**, best-vs-best 实测 **+0.030 dB → KILL** |
| 第 3 次 (Round 12 修订 = 本版本) | 按下表双轨披露, 不隐含主报口径 | 3/3 reviewer Round 12 verify CSV 后修正 |

### 正确双轨表 (Round 12 reviewer per-slice CSV 实算)

| 比较口径 | NORMAL ΔPSNR | 预注册 band (§1.4) |
|---|---:|---|
| **best-vs-best** (V18.best 165K vs V7.best 160K) | **+0.0302 dB** | **KILL** (< +0.05) |
| **last-vs-best** (V18.last 200K vs V7.best 160K) | **+0.0617 dB** | **PARTIAL** ([+0.05, +0.30] 下沿) |

**主报口径 (user Round 12 决策 = A)**: best-vs-best. 理由: V18 best.pt selection criterion = `val_multi_objective(0.5×d20 + 0.45×d10 + 0.9×d4 + 1.5×normal)` (NORMAL 加权最高), best.pt 是 commit-time 选 ckpt, last.pt 只是 plateau 时刻的 noisy snapshot.

→ V18 outcome = **KILL (formal pending KL drift confirm)**, last-vs-best 仅作辅助披露.

**阶段 C 路径 (user Round 12 决策 = A)**: 先测 KL drift (~30 min, slot 1 已 free, 0 GPU 增量) → 按 §2.3 row 4 双触发 (ΔPSNR + KL drift) 确认 KILL.

### 根因 (Round 12 元教训第 N+2 次复发)

claude 整个 Round 1-11 在防 LLM 滑坡 (修偏差 PR 自身造同型偏差), 但 V18 substrate review 第一刻就滑坡:
- 第一次错 anchor 到历史叙事 (D20=35.44, 11.2 dB attackable gap)
- 第二次错 anchor 到 "PARTIAL 标签" + 走捷径选让叙事 clean 的数字 (last-vs-best +0.06 dB)

**防御 (Round 12 standing rule #10 新立)**: 任何 V18-family ΔPSNR 报告必须同时列 4 chain × 2 ckpt = 8 个数字, 每个配独立 band 标签. 禁止单写一个 ΔPSNR 不标比较口径.

---

## 3. 给 reviewer 的 6 个问题

### Q1 — V18 实际 outcome 是 SUCCESS 还是 PARTIAL?

按 §1.3 实测 ΔPSNR + §1.4 预注册阈值, 用 best-vs-best 或 last-vs-best, V18 outcome 落在哪一档?

请明确:
- 用哪个比较 (best vs best / last vs best / 其他)?
- 该比较的 ΔPSNR(NORMAL) 数值?
- 按预注册阈值落在 SUCCESS / PARTIAL / KILL 哪档?
- claude 自纠 (PARTIAL not SUCCESS) 是否正确?

### Q2 — V7 NORMAL = 36.78 dB 数据可信吗?

§1.2 V7 baseline 来源是 [PLANF_FINAL_ANALYSIS_20260516.md](PET_LatentResidual/review/0516/PLANF_FINAL_ANALYSIS_20260516.md).

请 verify:
- 该文档是否真用 canonical `calc_psnr_clip3` 评估 (与 V18 同 evaluator)?
- 是否 same val split (n=7403)?
- 是否同 chain definition (NORMAL = full chain 末端 = identity transport)?
- 若 V7 是用不同 evaluator/split 测的, ΔPSNR 比较就 invalid

### Q3 — V18 metric correction (45.6 → 36.8 dB) 是否真合理?

[V18_METRIC_CORRECTION_REPORT_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_METRIC_CORRECTION_REPORT_20260518.md) 说之前 V18 报告的 45.6 dB 是**错的** (post-hoc `10*log10(9/MSE)` 从 `val_chain_*_mse` 算), canonical `calc_psnr_clip3` 实测 36.8 dB.

请评估:
- 45.6 dB → 36.8 dB 差 8.8 dB, 是否合理 (clip [0,3] 后 dynamic range 压缩可解释)?
- `10*log10(9/MSE)` 公式哪来? 它和 PSNR_clip3 关系是什么?
- canonical PSNR_clip3 评估流程 (decode → SUV → clip [0,3] → per-slice PSNR) 是否真是 V7 用过的?
- 是否仍有可能 V18 实际更好但 evaluator 隐含 bias 导致 36.8 dB 低估?

### Q4 — D20/D10/D4/NORMAL 趋势暗示什么? (Round 12 修订: 双口径独立报告, 不 cherry-pick)

**best-vs-best 口径** (主报口径):

| chain | V18.best - V7.best | 走向 |
|---|---:|---|
| D20 | **+0.0028 dB** | 改善 |
| D10 | **+0.0152 dB** | 改善 |
| D4 | **+0.0208 dB** | 改善 |
| NORMAL | **+0.0302 dB** | 改善 (全在 KILL band, < +0.05) |

best-vs-best 下是 **严格 monotonic ascent**, 但全部 < +0.05 阈值 → 支持 "V18 击長 chain robustness" 假设, 但 effect size 不足.

**last-vs-best 口径** (辅助披露):

| chain | V18.last - V7.best | 走向 |
|---|---:|---|
| D20 | **-0.0142 dB** | **REGRESSION** ⚠️ |
| D10 | **+0.0245 dB** | 改善 |
| D4 | **+0.0388 dB** | 改善 |
| NORMAL | **+0.0617 dB** | PARTIAL 下沿 |

last-vs-best **不是 monotonic** (D20 退化). 暗示 V18 训练后期 over-fit long chain, 牺牲了 short chain D20.

请评估:
- V18 击的是 long-chain regime (transport-error-accumulated decoder slack) 还是 decoder 内在 ceiling?
- D20 regression 是 over-fit 信号 还是 noise (per-slice CSV win rate = 51.5%, 近 coin flip)?
- 对 paper narrative 重要 (V18 framework 是 "long-chain robustness" 而非 "uniform decoder gain"), 但 effect size 太小 paper claim 风险高.

### Q5 — best vs last 几乎平局 (Δ +0.0315 dB), 对阶段 C 有何意义?

V18 训练 165K-200K 期间 PSNR 几乎 plateau. 含义:
- V18-clean (若 launched) 是否 max_steps=180K 即可, 不需 200K?
- best.pt 选择 metric (val_chain_normal_mse 最低) 是否真选到了最好 ckpt? 还是 last.pt 略好暗示选择 criterion 有 bias?
- V18 family sweep (rank/blocks) 是否值得做 (PARTIAL outcome 触发 §2.3 row 1/2)?

### Q6 — 阶段 C 路径选择 (按 NEXT_STAGE 3×3 决策矩阵)

V18 outcome = PARTIAL 落在矩阵第 2 行. 第 2 行需 KL drift 数据决定:

| V18 outcome \ KL drift | NEGLIGIBLE | MODERATE | SIGNIFICANT |
|---|---|---|---|
| **PARTIAL** ([+0.05, +0.30]) | **slot 3 = backbone 方向** | 暂缓决策 | **slot 3 = V18-clean launch** |

请评估:
- 阶段 C 是否必须先做 KL drift 测量 (~30 min, slot 1 已 free)?
- 或可以省 KL drift, 直接按"V18 PARTIAL 而非 SUCCESS"决定不跑 V18-clean (改 paper "limitation: V18 attain PARTIAL not SUCCESS")?
- V18-clean 若 launch, 真实成本 ~1.4 天 (B20 修正), ROI 高吗 (V18-clean 期望 ΔPSNR > V18 by ≥+0.05 dB)?

---

## 4. 资料目录

### 4.1 本轮主审对象

- [V18_FINAL_RESULTS_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md)
- [V18_METRIC_CORRECTION_REPORT_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_METRIC_CORRECTION_REPORT_20260518.md)
- [V18_FULLVAL_EVAL_20260518.md](PET_LatentResidual/review/0517/V18_decoder_lora/fullval_eval_20260518/V18_FULLVAL_EVAL_20260518.md)
- per-slice CSV (n=7403): [v18_best_fullval_psnr_chain_mse_per_slice.csv](PET_LatentResidual/review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv)

### 4.2 V7 baseline 来源

- [PLANF_FINAL_ANALYSIS_20260516.md](PET_LatentResidual/review/0516/PLANF_FINAL_ANALYSIS_20260516.md) line 68-69
- [PEER_REVIEW_PROMPT_image_aux_direction_20260516.md](PET_LatentResidual/review/0516/PEER_REVIEW_PROMPT_image_aux_direction_20260516.md) line 79

### 4.3 预注册阈值 + 决策矩阵

- [V18_design_rationale.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_design_rationale.md) §2.3 (4 outcome 表) + §2.4 (Round 5/6/7 EV 修正记录, [0.05, 3] dB 区间)
- [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](PET_LatentResidual/review/0517/NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) §3.1 3×3 决策矩阵
- [REVIEW_INTEGRATION_round7_20260517.md](PET_LatentResidual/review/0517/REVIEW_INTEGRATION_round7_20260517.md) (user 决策 "全推荐", 不改阈值)

### 4.4 历史偏差 context

- B23 = premature claim (Round 10 立). claude 本次自纠的就是 B23 同型 (用 D20 数字推 NORMAL 判定)
- B10/B11 = 用单一 scalar 当 attackable gap. claude 本次也犯 B10 同型 (用 "11.2 dB attackable" 历史叙事 anchor 到 +1.37 dB ΔPSNR 判定)

---

## 5. 输出格式 (请 reviewer 严格遵守)

每位 reviewer 独立产出 markdown:

### 5.1 6 个问题逐条回答 (APPROVE / MODIFY / REJECT)

### 5.2 整体 verdict — V18 outcome
- `SUCCESS` (ΔPSNR ≥ +0.30)
- `PARTIAL` ([+0.05, +0.30])
- `KILL` (<+0.05)
- 必须明示 best-vs-best 或 last-vs-best 比较

### 5.3 阶段 C 决策建议
- A: KL drift 测量后按 3×3 矩阵决定
- B: 直接 paper draft + "V18 PARTIAL limitation" 章节, 不跑 V18-clean
- C: 直接 launch V18-clean 不等 KL drift
- D: backbone/data 方向, V18 family retire
- 选项 + 理由

### 5.4 claude 自纠评估
- claude 自报犯 B23 同型 (用 D20 baseline 推 NORMAL 判定), 自纠 SUCCESS → PARTIAL
- 自纠是否正确?
- 是否有更深错误 claude 未识别?

### 5.5 新偏差 (B36+, 若发现)

---

## 6. 约束与提醒

- **本轮不重审** Round 1-11 已签内容
- **不假设** 你能跑代码 — 但可读 csv / json artifact + grep markdown
- **特别关注**:
  - Q2: V7 baseline 是否真同 evaluator (这决定 ΔPSNR 是否 valid)
  - Q4: D20/D10/D4/NORMAL 单调上升的物理含义
  - claude 自纠是否真完全, 或还有 hidden 同型偏差未识别
- **优先质疑**:
  - claude 自纠声称 "V7 NORMAL = 36.78 dB" 数据是否可信 (二次 verify grep + 读 PLANF report)
  - 36.8 dB canonical 是否真在 V18 attackable gap 区间内 (而非超出, e.g. V18 已 saturate decoder ceiling = ~36.84 dB, 即使 V18-clean 也无法再涨)

---

## 7. Standing constraints (Round 11 #7 已立 "Round 11 是 execution+audit-prep 审稿绝对终点", 本 Round 12 是 substrate review 不违反)

1. ≤ 3 并行训练任务
2. V18 不动 (已完成)
3. V18b 永久撤销
4. 阈值不改 (Round 7 全推荐, V18_design_rationale §2.3 锁定)
5. V18-clean 阶段 C 才 pre-register
6. user 决策不可推翻 (R7/R8/R9/R10)
7. **本 Round 12 是数据出来后 substrate review 终点**. 阶段 C 决策 commit + push 之后 Round 13+ 仅在 V18-clean / V19 / paper draft 出新 substrate 时才起.
8. 不审 audit 是否该做 (user 已请求)
9. **新**: V18 outcome 判定必须基于 grep verify 后的 V7/V18 数字, 不允许凭历史叙事 anchor

---

## 8. 本轮元说明

| 轮次 | 类型 | substrate |
|---|---|---|
| 1-7 | design | spec / yaml / 历史数据 |
| 8-11 | execution / audit prep | markdown 文档 (0 代码改动) |
| **12** | **V18 final substrate review** | **canonical PSNR_clip3 数字** + claude self-correction verification |

预期 Round 12 outcome:
- 最佳: 3/3 confirm V18 = PARTIAL + 阶段 C 路径共识 (e.g. 全推荐 A: 先做 KL drift)
- 中等: 3/3 同意 PARTIAL 但路径分歧 (A/B/C 各有支持), user 决策
- 最差: reviewer 发现 claude 自纠仍有错 (e.g. V7 baseline 不 comparable), 起 Round 13 重新 verify
