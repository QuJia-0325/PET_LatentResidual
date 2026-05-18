# Round 12 Review Integration — V18 final results + claude self-correction

- date: 2026-05-18
- branch: foc_lite_hop0 (本地已 merge gitee, v3 task md 已 commit 未 push)
- 主审对象: claude 对 V18 final results 的解读 + self-correction
- 3 reviewer 独立 verify (含 per-slice CSV 实算 + JSON metadata cross-check)
- 上游: V18 codex artifacts (merged from gitee `626b0b4`)

---

## 0. 一行结论

**3/3 reviewer REJECT claude 自纠的 "PARTIAL (best vs best)" 标签** — 这是 B23/B11 同型偏差**第二次复发** (claude 在 prompt 里宣称自纠了 B23, 实际只修了一半: chain 名称改对了 (D20→NORMAL), 但 ckpt 比较口径标签错了 (last-vs-best 数字 + best-vs-best 标签)).

**真正 verdict (3/3 共识)**:

| 比较口径 | NORMAL ΔPSNR | 预注册 band |
|---|---:|---|
| **best-vs-best** | **+0.0302 dB** | **KILL** (< +0.05) |
| **last-vs-best** | **+0.0617 dB** | **PARTIAL** [+0.05, +0.30] |

**阶段 C 推荐路径 (3/3 共识)**: 选项 **A — 先测 KL drift (~30 min, slot 1 已 free), 按 §2.3 row 4 双触发 (ΔPSNR + KL drift) 确认 KILL 或 PARTIAL**. 不直接 launch V18-clean, 不直接 paper draft.

---

## 1. 3/3 reviewer 独立 verify 共识

### 1.1 V7 baseline 可信 (Q2 全 APPROVE)

agent1/agent2 都 python load 了 V7 与 V18 raw JSON artifact, cross-check 5 个 evaluator 字段:

| field | V7 | V18 | 状态 |
|---|---|---|---|
| `psnr_metric` | `src.utils.metrics.calc_psnr_clip3` | (同) | ✅ |
| `chain_mse_definition` | `mean((decode_crop(z_chain[t]) - x_rollout[t])^2)` | (同) | ✅ |
| `split` | `val` | (同) | ✅ |
| `num_eval_slices` | 7403 | 7403 | ✅ |
| `rollout_timepoints` | [D50, D20, D10, D4, NORMAL] | (同) | ✅ |

→ ΔPSNR 比较有效, V7 NORMAL = 36.7810 dB 真实可信.

### 1.2 V18 metric correction 合理 (Q3 全 APPROVE)

agent2 分析 `45.6 dB → 36.8 dB` 修正的 3 个独立 ground:
1. **Wrong dynamic range**: 原公式假设 MAX²=9 (SUV clip [0,3]²), 但 training-time `val_chain_*_mse` 在 pre-clip decoder space, peak undefined
2. **Wrong reduction order**: canonical = `mean_over_slices(10*log10(MAX²/per_slice_MSE))`, post-hoc = `10*log10(MAX²/mean_MSE)` (Jensen 不等)
3. **Wrong domain**: canonical clip [0,3] 在 decode + SUV map 后; training MSE 无 SUV map / 无 clip

agent3 独立用 [metrics.py](RAE/RAE/src/utils/metrics.py) 验证 canonical 流程: `20*log10(3) - 10*log10(mse)`, 实测匹配 36.81/36.84 dB.

→ 36.8 dB 真实, 45.6 dB 数字废弃.

### 1.3 D20→NORMAL 单调性 — 3/3 reviewer catch prompt §1.3 vs Q4 数据漂移

**重大发现** (agent1/agent2/agent3 全 catch): prompt §1.3 表 V18.last 的 D20 = **-0.0142** (regression), 但 Q4 narrative 引述 D20 = **+0.003** (来自 best-vs-best). 同一 prompt 内部数据漂移.

**实测正确 last-vs-best**:

| chain | last - V7.best | 走向 |
|---|---:|---|
| D20 | **-0.01421** | **REGRESSION** |
| D10 | +0.02453 | 改善 |
| D4 | +0.03879 | 改善 |
| NORMAL | +0.06169 | PARTIAL 下沿 |

**纯 last-vs-best 实际是 NOT monotonic** (D20 退化). 纯 best-vs-best monotonic 但全部 < +0.05 (全 KILL).

Q4 的 "long-chain robustness" narrative 仅在 best-vs-best 口径下成立; 用 last-vs-best 必须 acknowledge D20 退化.

### 1.4 best.pt vs last.pt plateau (Q5 全 APPROVE w/ caution)

agent3 per-slice 实测 V18 last - V18 best:
- NORMAL: +0.03148 dB
- **win rate 仅 51.52%** (近乎 coin flip)
- D20 退化 -0.01703 dB

**best.pt 选择 criterion** = `val_multi_objective = 0.5×d20 + 0.45×d10 + 0.9×d4 + 1.5×normal` (json artifact verify). NORMAL 已加权最高, 但 best.pt (165K) 在 canonical NORMAL PSNR 上仍**比** last.pt (200K) 略差 +0.03 dB — 是 V6_NOISE 同型 "rolling-val MSE vs full-val PSNR_clip3 selection mismatch" (PLANF §1.1).

含义:
- **不推荐换 last.pt 作主报告口径** (best.pt 是合规选择, last 只是 noisy 选择)
- 但 win rate 51% 暗示 V18 best-vs-best vs last-vs-best 的 ΔPSNR 差异在 noise floor 量级
- **不推荐 V18-r64 / V18-last4 sweep** (best-vs-best +0.030 < +0.05 KILL, marginal gain 不太可能 clear PRIMARY SUCCESS)
- V18-clean 若 launch, **不**应该砍 max_steps 到 180K (V18 plateau ≠ V18-clean plateau, 起点+动力学不同)

---

## 2. 3/3 共识必修: claude self-correction 第二次复发 (B23 同型)

### 2.1 错误轨迹

| 时刻 | 错误 | 形态 |
|---|---|---|
| **第 1 次错** (V18 数据出来时) | 用 V7 D20=35.44 推 NORMAL baseline, ΔPSNR=+1.37 dB → SUCCESS | B23 (premature claim, anchor 历史 11.2 dB 叙事) |
| **第 1 次自纠** (本 prompt §2) | 改 V7 NORMAL=36.78, ΔPSNR=+0.06 dB → PARTIAL | 数字对方向 |
| **第 2 次错** (本 prompt §2 标签) | "+0.06 dB → PARTIAL (**best vs best 比较**)" | **错** — +0.06 是 last-vs-best, best-vs-best 实测 +0.030 → KILL |
| **第 2 次自纠待做** | 显式区分 best-vs-best (KILL) 与 last-vs-best (PARTIAL), 两套数字独立报告 | 待 Round 12 修订 |

### 2.2 元教训 (Round 11 元教训 #N+1 次复发)

claude 整个 Round 1-11 在防 LLM 滑坡 (修偏差 PR 自身造同型偏差), 但 V18 substrate review 第一刻就滑坡: **第一次错** anchored 到历史叙事 (D20=35.44), **第二次错** anchored 到 "PARTIAL 标签" + 走捷径选让叙事 clean 的数字.

agent3: "claude 自纠走了'修一个 anchor 错, 换另一个 anchor 错'的捷径".

---

## 3. 3 个新偏差 (B36 系列)

### 3.1 B36 — Comparison-label mismatch (agent1+2+3 一致)

**形态**: 当有 2×N 比较口径 (best-vs-best / last-vs-best × D20/D10/D4/NORMAL), 用 X 口径的数字配 Y 口径的标签, 走捷径选 favorable 那个.

**实例**: claude 自纠 "+0.06 (best vs best)" 数字属 last-vs-best.

**修法 standing rule**: 任何 V18-family ΔPSNR 报告**必须**同时列 4×2=8 个数字 (4 chain × 2 ckpt 比较), 每个数字配独立 band 标签. 禁止单写一个 ΔPSNR 不标比较口径.

### 3.2 B37 — Mixed-chain delta narrative (agent2+3)

**形态**: Q4 narrative 把 D20 (best-vs-best 数字) 和 D10/D4/NORMAL (last-vs-best 数字) cherry-pick 拼成 "monotonic" 故事.

**实例**: prompt Q4 引用 "+0.003 / +0.025 / +0.039 / +0.062 (last-vs-best chain)" — D20 数字其实来自 best-vs-best.

**修法**: §A/§Q 任何 ΔPSNR 表必须单一口径全列, 不混搭.

### 3.3 B38 — Threshold dimensionality reduction (agent2)

**形态**: prompt §1.4 把预注册 2D 阈值 (ΔPSNR + KL drift, V18_design_rationale §2.3 row 4) 简化为 1D ladder (仅 ΔPSNR), 丢了"灰区"选项 (sub-PARTIAL 但 KL drift 未测时 = pending KILL).

**实例**: best-vs-best Δ=+0.030 dB, 1D 表直接判 KILL, 2D 真阈值是 "ΔPSNR < +0.05 + KL drift > 0.05 → 撤回", 当前 KL drift 未测 = formal KILL pending.

**修法**: 阈值表加 "+ KL drift > 0.05 才 confirm KILL/SUCCESS/PARTIAL" 列, 与 §2.3 row 4 一致.

### 3.4 B36c (agent2) / B38 narrative scale anchor (agent1) — 历史叙事 anchor

agent1: "Round 5/B10 framing 'V18 attackable gap ~11.2 dB' 让 +1.37 dB 看似合理; 实际 scale 是 +0.03-+0.06 dB (~200×小)" — 这是 claude 第一次错的元因. 任何 "attackable gap N dB" 必须配 "we expect ≥X% recovery → ≥Y dB" 的 falsifiable subscale prediction.

---

## 4. 阶段 C 决策矩阵 — 必须重审

按 NEXT_STAGE_ARCH_CODE_FINAL §3.1 原始 3×3 矩阵:

| V18 outcome \ KL drift | NEGLIGIBLE | MODERATE | SIGNIFICANT |
|---|---|---|---|
| SUCCESS (≥+0.30) | null / paper draft | (同) + KL note | (同) + B9 limitation note |
| **PARTIAL** ([+0.05, +0.30]) | backbone 方向 | 暂缓决策 user 沟通 | **V18-clean launch** |
| **KILL** (<+0.05) | backbone/data/architecture | 同 | 同 |

**当前 V18 状态 (待 KL drift)**:
- best-vs-best (+0.030): KILL 行
- last-vs-best (+0.062): PARTIAL 行 (下沿)
- KL drift: **未测**

**Stage C 推荐 sequence** (3/3 共识 = 选项 A):
1. **STEP 1** (今天, 0 GPU 增量): 测 KL drift (~30 min, slot 1 已 free)
2. **STEP 2** (依 KL drift 结果分支):
   - KL drift < 0.05 (NEGLIGIBLE): V18 落 PARTIAL × NEGLIGIBLE → **backbone 方向** (paper "V18 limitation" 章节)
   - KL drift ∈ [0.05, 0.10) (MODERATE): 暂缓决策 + user 沟通 (best-vs-best=KILL vs last-vs-best=PARTIAL 灰区由 user 拍板)
   - KL drift > 0.10 (SIGNIFICANT): 落 PARTIAL × SIGNIFICANT → 可 launch V18-clean **但** ROI 已下调 (V18-clean 期望 vs V18 ≥+0.05 dB 的概率, 在 V18 仅+0.030 dB 后 < 20%, 1.4 天 slot 3 不够本)

**反对** 选项 B (直接 paper draft 不测 KL drift): KL drift 测量 0 GPU 增量, 30 min, 直接决定 retire vs sweep, ROI 极高.

**反对** 选项 C (直接 launch V18-clean): V18-clean 期望 ΔPSNR > V18 by ≥+0.05 dB 的先验概率, 在 V18 already-PARTIAL 下小. 1.4 天 slot 3 不值.

---

## 5. push prompt 前最小修订 (~10 min)

agent2/agent3 列的修订全部采纳:

| # | 位置 | 修法 | reviewer |
|---|---|---|---|
| 1 | §2 自纠表 "+0.06 dB (best vs best)" → "+0.06 dB (last vs best)" + 新增 best-vs-best=+0.030 → KILL 行 | 必修 | 3/3 一致 |
| 2 | Q4 narrative: D20 列改用 last-vs-best=-0.0142 OR 显式 "D20 用 best-vs-best 因 last-vs-best regress" | 必修 | agent2+3 |
| 3 | §1.4 阈值表加第 5 列 "+ KL drift > 0.05 confirm KILL/PARTIAL" (与 §2.3 row 4 一致) | 必修 | agent2 |
| 4 | Q6 决策矩阵加 "KILL × any" 第 3 行, 说明 best-vs-best 在 KILL band | 必修 | agent2 |
| 5 | Q5 显式列 best.pt criterion 全文 `val_multi_objective` 加权 | 应修 | agent2 |
| 6 | standing rule #10 新立: 任何 V18-family ΔPSNR 必须 8 数字全列 (4 chain × 2 ckpt) | 应修 | agent1+2+3 |
| 7 | standing rule #11 新立: 任何 "attackable gap N dB" 必须配 falsifiable subscale prediction | 应修 | agent1 |

---

## 6. user 决策点 (3 个)

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | V18 主报告口径 | A: best-vs-best (合规 best.pt selection, +0.030 KILL) <br>B: last-vs-best (endpoint vs endpoint, +0.062 PARTIAL) <br>C: 双轨披露 (8 数字全列, 主 verdict 取严, 即 KILL) | **A** (best.pt 是 commit-time 选 criterion, last 是 plateau 时刻 noisy) |
| 2 | 阶段 C 路径 | A: 先测 KL drift → 矩阵分支 (3/3 推荐) <br>B: 直接 paper draft + V18 limitation 章节 <br>C: 直接 launch V18-clean (1.4 天 slot 3) <br>D: backbone/data 方向 V18 family retire | **A** (0 GPU 增量, 30 min, 决定全局) |
| 3 | 是否修订 prompt 后再发独立 reviewer (Round 13) | A: 修订后发 (要求 reviewer 必须 grep verify CSV/JSON, 不接受 markdown 转述) <br>B: 整合本 Round 12 直接进 KL drift 测量, 不再起 Round 13 | **B** (3 reviewer 已 100% verify CSV+JSON, 共识强, 起 Round 13 ROI 低; 但 standing rule #10/#11 立, 未来 V19+ 自动 enforce) |

回复格式: `1=A 2=A 3=B` 或 `按推荐`.

---

## 7. Round 12 元教训

**Round 11 元教训第 N+2 次复发**: claude 自纠 SUCCESS → PARTIAL **方向对但论据错** — 用 last-vs-best 数字配 best-vs-best 标签, 是 B11/B23 同型走捷径. 3/3 reviewer 用 per-slice CSV + JSON metadata cross-check 才 catch.

**根本原因**: substrate review 比 markdown review 更 cognitive-load (要同时 hold 多 ckpt × 多 chain × 多 evaluator metadata), claude anchoring 走捷径概率更高. 防御 = standing rule #10 (8 数字全列) + standing rule #11 (attackable gap 配 subscale prediction) + 严格要求 reviewer 跑 grep + 实算, 不接受 markdown 转述.

**Round 12 也是 Round 1-12 最深的偏差实例**: claude 在 prompt 自己声称"自纠 B23"的同一段, 自己又造 B23 同型. 这说明 self-correction 本身是 anchor 滑坡风险点, 必须 outsource 给 mechanical script (per-slice CSV 实算) 或 independent reviewer (本 Round 12 三 reviewer 起作用).

---

## 8. 立即下一步

待 user 回复决策 1/2/3 → claude:
1. (若 1=A 或 C) 修订 prompt §2 自纠表 + Q4 D20 列 + §1.4 阈值表 + Q6 矩阵 (~10 min)
2. (若 2=A) 起 KL drift 测量 task md → push gitee → codex 在服务器执行 (~30 min)
3. (若 3=B) 不起 Round 13, 直接进 KL drift 测量后的下一步决策
4. standing rule #10/#11 写进 V18_design_rationale §5.2 (5 min)

总 ~50 min (含 KL drift 测量), Phase A v3 push 与本流程独立 (V13 launch 与 V18 outcome 无关, 可并行).
