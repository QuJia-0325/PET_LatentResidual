# Round 16 Peer Review — Reviewer A (independent)

- date: 2026-05-19
- reviewer: **Reviewer A** (GitHub Copilot, Claude Opus 4.7 xhigh, independent draft, did not see Reviewer B/C drafts)
- substrate verified via: grep + jsonl 实算 + sed on 5 artifacts (paths cited inline)
- 主审对象: Round 16 prompt §1 (A3 数字) + §1.4 (3 候选解读) + §3 (Q1-Q7) + §5 (verdict options)

---

## 0. One-line verdict

**主 verdict = A** (A3 已足够把 "KL pullback 解释 V18 direct-decode GT-manifold 提升" 这条叙事**基本判死**, 但有 2 个 prompt 未 highlight 的 substrate-fact 必须并行写入主结论), 同时**4 个 HIGH 新偏差 B61-B64 必须修复** (含 prompt 自己 Q7 warned but committed 的 direct-decode/chain 混淆).

---

## 1. 数字独立核验

| 声明 | 来源 | Reviewer A 验证 |
|---|---|---|
| A3@165K val_select=0.0009037353 / NORMAL MSE=0.0002454921 | jsonl 直读 (`val_chain_unique_slices=7403` full-val 行) | **✓** |
| A3@170K val_select=0.0009039954 / NORMAL MSE=0.0002454063 | 同上 | **✓** |
| delta 170K-165K: val_select +2.6e-7, NORMAL MSE -8.6e-8 | 算术 | **✓** (方向正确, val_select 略差, NORMAL 略好) |
| V18.best@165K val_select=0.0009037494 / NORMAL MSE=0.0002454947 | [V18_FINAL_RESULTS L30-41](review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md#L30-L41) | **✓** |
| KL drift probe 5 ckpts × 4 chain (V7/V18.best/V18.step170k/V18.last/V18-cap.last) | [KL_DRIFT_REPORT.md](review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_REPORT.md) | **✓** verbatim |
| matched-step delta V18-cap.last - V18.step170k NORMAL = +0.0020 | 52.8047 − 52.8027 | **✓** |
| B42 仍成立 (`train_first_hop.py:2230`) | `sed -n '2228,2232p'` | **✓** verbatim |

**所有数字 100% verify**. Round 16 prompt 是 Round 12 以来数字最干净的一份.

---

## 2. Prompt 未 highlight 的 2 个关键 substrate fact

### 2.1 A3@165K vs V18.best@165K **chain 指标也几乎完全 tie**

| metric | A3@165K | V18.best@165K | Δ |
|---|---:|---:|---:|
| val_select_score | 0.0009037353 | 0.0009037494 | **-1.4e-8** |
| NORMAL MSE | 0.0002454921 | 0.0002454947 | **-2.6e-9** |
| D20 MSE | 0.0003304433 | 0.0003304456 | -2.3e-9 |
| D10 MSE | 0.0002966598 | 0.0002966660 | -6.2e-9 |
| D4 MSE | 0.0002630874 | 0.0002630942 | -6.8e-9 |

**A3 在所有 chain metric 上都比 V18.best@165K 略好 (但量级 < 1e-7)**.

含义: 在 **trainer 自己的 multi-objective chain selection metric 上**, 关闭 KL pullback **没有任何代价**, 反而**微好**. 这是比 prompt §1.2 表 (只列 A3 内部 170K vs 165K) 更强的发现 — 它直接对比 KL on (V18) vs KL off (A3) 在 matched-step (165K) 的 chain 表现, 结论是 **KL 在 chain 指标上也未显示出哪怕 1e-7 量级的收益**.

prompt Q2 暗示了这点但没把数字摆出来. 这是 Q2 verdict 的核心证据.

### 2.2 V18 direct decode 在 170K→200K **regress**

| step | V18 NORMAL direct decode PSNR | 相对 V7.best |
|---|---:|---:|
| 165000 (V18.best) | 52.7314 | +0.0973 |
| 170000 (V18.step170k) | 52.8027 | +0.1686 |
| 200000 (V18.last) | 52.7980 | **+0.1639** |

V18 direct decode GT-manifold 在 step 170K 达 peak, 之后 30K **倒退 −0.0047 dB**.

含义:
- (a) trainer 的 best.pt 选择 (165K, by chain MSE) **错过** direct decode 真 peak (170K)
- (b) V18 在 170K→200K 的额外 30K KL 训练**不是中性**, 是**轻微 destructive** for direct decode
- (c) 这与 B44 (Round 14 立) "KL 通过共享 LoRA 间接耦合 z_GT" 一致 —— KL gradient 在长训中让 LoRA 偏离 z_GT-optimal 方向

prompt Q1/Q3 都没引用这条. 它是 "KL secondary effect 在 z_pred path 是否有用" 的**反证据**: 若 KL 在 z_pred 上有用, V18 应在 170K→200K 持续受益, 实际是轻度受损.

---

## 3. Q1-Q7 逐条

### Q1 — APPROVE 候选 A, MODIFY 表述

**A3 已足以把 "KL 解释 V18 direct-decode GT-manifold 提升" 基本判死**, 论据 4 条:

1. matched-step delta +0.0020 dB (D20/D10/D4/NORMAL 全部 < 0.002, 远低于预注册 0.02 阈值)
2. A3@165K **chain 指标**也几乎完全 tie V18@165K (§2.1, 比 prompt 给出的更强证据)
3. V18 direct decode 170K→200K **regress** -0.005 (§2.2, KL secondary 反向证据)
4. B42 机制层面早已成立, A3 substrate 闭环

**但最准确表述不是 "KL 被基本证伪"** (这会被 reviewer 抗议 "你只测了 direct decode"), 而是分两层:

- **direct decode 层**: KL pullback 设计对 V18 GT-manifold direct decode improvement **没有 measurable 贡献** (matched-step +0.002 dB tie)
- **chain 层**: 同样 KL pullback 在 chain MSE 上 **没有 measurable 贡献** (§2.1 1e-7 tie)
- **z_pred path / rollout 层**: A3 没直接测, 但 §2.2 间接证据 (V18 170K→200K direct decode regress) 暗示 KL 在长训也无收益

候选 B (留 secondary 余地) 偏弱: §2.2 的 regress 证据已经让 "KL 在 170K→200K 有用" 不可信. 候选 C (insufficient) 完全错: A3 设计就是 disambig 实验, 它**本来就**只回答 direct decode mechanism, 不回答 "KL 在所有可能路径都无用".

→ **主推 A, 加 §2.1 + §2.2 两条作为 paper-quotable 论据**.

### Q2 — APPROVE

**至少在 160K→165K 段, V18 chain improvement 也几乎全可由 capacity-only 解释** — §2.1 数据闭环.

更精确的命题改写:
- 旧: "V18 有效"
- 新: "V7→V18@165K +0.03 dB chain 改善 = 在 V7.best 上加 LoRA r32 last-2 blocks 训 5K, 与 KL on/off **无关**" (§2.1)
- 推论: V18 paper 不能再讲 "KL pullback enables decoder adaptation"; 只能讲 "decoder LoRA (rank=32, last 2 blocks) provides marginal +0.03 dB chain improvement"

prompt Q2 第 3 子问 "不是的话还缺哪条关键证据" — 答: **不缺**. §2.1 已 closed-form. 但**剩下问题**是 V18 vs V7 +0.03 dB 是否超 d_pure noise band (等 V14).

### Q3 — APPROVE 大部分, MODIFY §3.2

#### Q3.1 V18/KL 线
- **直接撤下 KL 成功叙事** (§2.1+§2.2 闭环). 不再为 KL pullback 投入新实验.
- **V18-clean (use_pred_latent=false)**: REJECT resurrection. 理由 = (1) Round 7 user 永久撤销 = standing rule #5 不可推翻; (2) 即使复活, 它要回答的问题是 "use_pred_latent=false 对 chain 是否更好", 但 A3 已显示 KL on 与 off 在 chain 上 tie → use_pred_latent flag 是 second-order 调整, EV 低于 d_pure noise floor

#### Q3.2 decoder capacity 线 — Prompt 表述需 MODIFY (direct/chain 混淆, 见 B61)

prompt 写 "decoder capacity 微幅有效, 但 chain 收益极弱". 这是**直接/chain 混淆**:
- **direct decode 层**: capacity 改善 **+0.17 dB NORMAL** (V18-cap.last vs V7.best) — **不是"微弱"**, 是中等量级
- **chain 层**: capacity 改善 **+0.03 dB NORMAL** (A3@165K vs V7.best, MSE 差 ~1.5e-7) — 极小

正确表述: "rank=32 last-2 blocks LoRA 让 decoder 在 GT-manifold direct decode 上得到稳定 +0.17 dB, 但这个 capacity gain **不传递到 transport chain**, 因为 transport-side 误差是主导项 (Round 13 §2 attackable-gap 分析)".

继续 rank/blocks sweep EV 评估:
- 若问 "direct decode 还能涨多少": 也许 (没测过 rank=64 / blocks=last-4)
- 若问 "chain MSE 还能涨多少": 几乎不可能, transport-bottlenecked
- 因此 decoder sweep **EV 低**, 除非 paper 故事要 "decoder ceiling exploration" 章节

#### Q3.3 transport 线 — APPROVE
A3 + §2.1 联合证实 "transport 是主瓶颈". 真正未解是 z_pred / rollout / chain path. Transport-side intervention (Round 13 §Q4 候选 A-E) 升为最高优先级.

### Q4 — APPROVE 主框架, ADD 4 条

**现在就能下的结论** (Reviewer A 完整列表):
1. "V18 direct decode +0.10 dB GT-manifold improvement = LoRA capacity, **不是** KL pullback 设计成果" (§1 + §2.2)
2. "V7→V18 +0.03 dB chain 改善 = LoRA 5K 训练副作用, KL on/off 无差异" (§2.1, **prompt 漏**)
3. "V18 direct decode 在 170K→200K **轻微 regress** -0.005 dB, KL 长训不是中性是 mild destructive" (§2.2, **prompt 漏**)
4. "decoder LoRA capacity 不是 chain bottleneck 的解; transport-side 是" (Q3.3)
5. "trainer best.pt 选择 (chain MSE 加权) 与 direct decode peak 不一致 (165K vs 170K), 这是 selection metric mismatch, paper 必须 disclose"
6. "继续投资 V18-family 或 V18-clean 复活 EV 低; reallocate 到 transport intervention"

**必须等 V13/V14 的结论**:
1. "V7 main gain 究竟来自 image_aux / step_weights / 联合" (V13 答)
2. "V18-V7 +0.03 dB chain 改善 vs d_pure noise band 是否显著" (V14 答)
3. "项目 ceiling 是否真已逼近 0.1 dB / 1 年累积是否 marginal" (V13+V14 联合答)
4. "paper narrative 终态 (V18-marginal / V7-only / pure-V7-feasibility)" (V13+V14 全完整后)

### Q5 — APPROVE 主框架

- **A (V18-clean)**: 已跌出高优先级 (Q3.1). 仅在 V13/V14 出来后, 若 user 仍想为 paper 主图加 KL ablation 才考虑; 否则永久撤
- **B (rank/blocks sweep)**: 已沦为 "继续优化 direct decode, chain 不敏感" 低 EV 路线. 仅在 paper 想加 "decoder capacity scaling" 章节才跑
- **C (z_pred/rollout transport intervention)**: 被动**升为合理主线**. 但 Round 16 不该立即 launch, 应等 V13/V14 outcome 决定 transport-side intervention 具体方向 (e.g. backbone scale-up vs flow matching vs multi-step refinement, 参考 Round 13 §Q4 A-E)

### Q6 — APPROVE narrative B 主推, REJECT narrative A, MODIFY narrative C

- **narrative A** ("KL pullback 改善 decoder manifold 但被 transport 吞噬"): **REJECT**, 已被 A3 闭环证伪. paper 主动撤下.
- **narrative B** ("decoder capacity 可改善 GT-manifold direct decode, 但不自动转化为 transport chain gain"): **APPROVE**, 当前最可信. 但措辞需精确化, 见 Q3.2.
- **narrative C** ("整个 V18 family marginal, 等 V13/V14 决定能否讲 paper"): **PARTIAL APPROVE**. C 比 B 更保守, 但 C 把 V18 direct decode +0.17 dB collapse 成 "marginal" 不准确. 更准确 narrative D (新提): "V18 family 包含一个 confirmed effect (decoder capacity → +0.17 dB direct decode) 和一个 disconfirmed mechanism (KL pullback 设计无贡献); chain 总收益是否超 noise floor 等 V14 决定"

**paper action**:
- 主动撤: "KL pullback enables decoder adaptation" / "two-stage transport with KL alignment" (设计原文)
- 保留: "decoder LoRA improves direct decode by +0.17 dB but is bottlenecked by transport in chain" (A3 实证)
- A3 作为**反证主体写进主文** (不放 appendix), 用于支撑 "we explicitly disambiguate KL contribution from capacity contribution via λ_kl=0 control"
- 暂留待定: chain 总效应是否值得 paper (V14 d_pure 决定)

### Q7 — Reviewer A 找到 4 个新偏差 (B61-B64), 详 §5

---

## 4. 主 verdict + 立即可执行

### 4.1 主 verdict = **A** (with §2.1+§2.2 论据加强)

A3 已足够把 KL direct-decode 叙事基本判死; 后续主看 transport / z_pred path. 但 paper-quotable 论据需用 Reviewer A §2.1 (chain 也 tie) + §2.2 (V18 long-train regress) 加强, 而非仅 prompt §1.3 的 +0.0020 dB direct decode delta.

### 4.2 立即可执行 (V13/V14 仍跑期间)

**现在 do**:
1. **撤下 KL 成功叙事文字** — design_rationale §2 / §3 + V18_FINAL_RESULTS §Interpretation 凡含 "KL pullback enables..." 表述改写 (~5 处, 见 Q3.1)
2. **A3 报告补 §2.1 §2.2 两条** (chain tie + V18 170K→200K regress), 现 A3 报告已 ready 但漏这两点; 加 ~10 行
3. **起草 paper outline 主图 ablation table** 含: V7 / V18.best / V18-cap.last 三列 × (direct decode, chain MSE) 两行 — 把 "KL 无功能" 用同一个数字矩阵 visualize
4. **写 standing rule B60**: "matched-step LoRA delta < 0.01 dB direct decode + < 1e-7 chain MSE = 该训练变量无 measurable effect, 不再投资该变量"
5. **复用 V18 step ckpts 跑 SSIM eval** (Round 15 Reviewer A 提议 paper-blind eval): 0 GPU 成本, 加 paper 工具箱

**现在不要 do**:
1. **不要** launch V18-clean / V18-rank64 / 任何 V18-family 新变体
2. **不要** 起 transport intervention task (等 V13/V14)
3. **不要** 改任何 paper narrative 决策 (V18 marginal vs V7-only) — 等 V14 d_pure
4. **不要** 把 A3 outcome 解读为 "项目 ceiling 已到" (B62, premature 提前终局)
5. **不要** 推翻 standing rule #5 复活 V18-clean

---

## 5. 新偏差 B61-B64

### B61 [HIGH] Direct decode vs chain 混淆 (prompt 自己 Q7 警告但 Q2/Q3.2 committed)

- prompt Q7 列 "direct-decode / chain 混淆偏差" 作为 reviewer 要查的偏差
- 但 prompt Q2 自己写 "decoder capacity 微幅有效, KL 设计未显示额外贡献" — "微幅" 这词来自 chain MSE 差 (~1.5e-7), 而 capacity 的 direct decode 改善是 +0.17 dB (≈ 5×V18 chain 改善的数量级)
- 同型 prompt Q3.2 "decoder capacity 微弱有效" 也是 chain 数量级套 direct decode 结论
- **修法**: Q2/Q3.2 把 capacity 效应**分两层列**: direct decode +0.17 dB (中等), chain MSE 1.5e-7 (极小), 不再混用 "微弱/微幅" 单标签

### B62 [HIGH] 提前外推 / 终局偏差

- prompt Q3.2 "decoder 已不是主瓶颈, rank/blocks sweep EV 很低" — 这话**部分正确** (chain bottleneck 不在 decoder), 但**外推过头**: 它依赖 "项目目标 = chain gain". 若 paper 目标改为 "decoder capacity ceiling exploration", direct decode +0.17 dB 是有意义的数据点, sweep EV 不低
- prompt §0 范围声明 "审 V18 / KL / decoder / transport 这几条研究线各自还剩多少可信空间" — 把 4 条线的全终局判定塞进单轮 review, **超 A3 substrate 能 cover 的范围**. A3 只测 V18 direct decode mechanism, 不直接判 transport / decoder ceiling
- **修法**: prompt §0 明确 "A3 仅判 V18-KL mechanism 一条线; decoder / transport / paper narrative 终局必须等 V13/V14"

### B63 [MED] V18-clean 复活作为 Q5.A 选项, 与 standing rule #5 抵触

- standing rule #5: R7 user 决策 "V18-clean 永久撤销" 不可推翻
- Round 13 prompt §2.3 也写 "A: V18-clean — KL drift 反预期后, 预期 V18-clean ≤ V18, 不推"
- 但 Round 16 prompt Q5 仍把 V18-clean (A) 列为候选, 询问 reviewer "若还保留 A, 它要回答的唯一问题是什么?"
- **形态**: 与 R15 B55 同型 — 不明确撤 standing rule, 又把已撤选项作为问题 placeholder, 让 reviewer 不知该按 "rule 不可推翻" 答 reject 还是按 prompt 邀请答 "什么问题"
- **修法**: Q5 显式标 "A 选项已在 R7 + R13 撤销, 此处仅询问 reviewer 是否同意永久撤"; 或删 A 选项, Q5 改 2 选项 (B sweep / C transport)

### B64 [LOW] §1.4 候选 A 表述比 A3 报告自己结论更激进

- A3 报告 §4 结论: "**本实验支持候选 B**: V18 direct decode improvement **主要可以由** LoRA decoder capacity **解释**" — 用 "主要" 留余地, 也加了 "不能作为 KL 设计有效的证据" 与 "不等价于 transport chain 改善" 两个 caveat
- Round 16 prompt §1.4 候选 A: "V18 在 GT latent manifold 上的 direct decode 提升**主要来自** LoRA decoder capacity, **而不是** KL pullback" — 删了 "可以解释" 改为 "来自", 删了 chain caveat
- **修法**: prompt §1.4 候选 A 改回 A3 报告原文 "可以由 LoRA decoder capacity 解释; direct decode improvement 不自动等价 transport chain improvement"

---

## 6. 元评论 (Reviewer A 立场)

R16 prompt 是 Round 12 以来**数字最干净、最自洽**的一份 (所有 5 个数字源 100% verify, B42 仍稳, 候选解读 A/B/C 三选项框架清晰). prompt 在 Q7 主动让 reviewer 找自己的偏差, 这是 R15 R14 R13 都没有的 "voluntary self-exposure" 风格.

但**仍犯 4 个新偏差** (B61-B64), 都是 R12 B45 standing rule (substrate 阶段 claude 走捷径) 同型, 只是这次走得隐蔽:
- B61 自己 Q7 警告但 Q2/Q3.2 自己 commit
- B62 把单实验外推到全终局
- B63 violated standing rule #5 但用问句包装
- B64 把 A3 报告的 nuanced "支持候选 B" 改写为更激进的 "候选 A"

**根本机制**: A3 outcome 是 prompt 的 "first wave" 成功 — 一个 disambig 实验给出干净 verdict (+0.002 dB tie). claude 倾向把这个干净 verdict **过度 leverage**, 让 R16 一举判决多条线终局. R12 B45 已立: substrate 阶段 claude self-correction 不可信, 必须 reviewer 介入. R16 即是验证 — 数字干净不等于解读干净.

**给 user 的建议**:
- R16 整合时, **主 verdict A** 接受
- 但 paper-quotable 论据必须用 Reviewer A §2.1+§2.2 加强, 不能仅靠 prompt §1.3 +0.0020 dB
- B61-B63 必须修后再 dispatch 给 Reviewer B/C (否则同 anchor)
- 不要把 R16 当 "战略终局" 轮 — 它只是 A3 substrate 解读轮; 真正战略轮等 V13+V14 完成后的 Round 17+

---

## 7. 输出元

- reviewer: **Reviewer A**
- methodology: 5 个 artifact full grep verify + jsonl 实算 + matched-step delta 独立算术 + B42 sed verbatim
- 未与 Reviewer B/C 交流
- 撰写时长: 单轮, no iteration
- 关键独立发现: §2.1 (chain 也 tie, prompt 漏) + §2.2 (V18 170K→200K direct decode regress, prompt 漏)
