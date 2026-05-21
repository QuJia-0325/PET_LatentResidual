# Round 17 Peer Review — Reviewer A (independent)

- date: 2026-05-21
- reviewer: **Reviewer A** (GitHub Copilot, Claude Opus 4.7 xhigh, independent draft, did not see Reviewer B/C drafts)
- substrate verified via: jsonl 实算 + grep + 与 PLANF_FINAL_ANALYSIS / Round 16 整合文件 cross-check
- 主审对象: Round 17 prompt §1 (V13/V14 数字) + §2 (SNR analysis) + §3 (5 选项) + §4 (Q1-Q7) + §5/§6 (narrative)

---

## 0. One-line verdict

**主战略 = Hybrid B+A4** with **强制 prerequisite F0 (零成本 per-slice paired-t analysis on V18 vs V7 from existing JSON)** before committing to paper draft. Without paired-t, "V18 +0.06 dB = real signal" 这个 paper-critical 命题**没有被证据支撑**, prompt §2 SNR 表用的 noise estimator (V14 单次 |Δ|=0.0004) **方法论错误**.

**V18 信号判定 = "borderline, needs free paired-t analysis FIRST"** (不是 prompt §6.3 任何一个选项).

**新偏差**: 6 个 (B61 HIGH, B62 HIGH, B63 MED, B64 MED, B65 LOW, B66 LOW).

---

## 1. 数字独立核验

| 声明 | 来源 | Reviewer A 验证 |
|---|---|---|
| V13.best=last NORMAL=36.4943 | [v13_best JSON](review/0516/full_eval_json/v13_true_image_aux_off_best_fullval_psnr_chain_mse.json) | **✓** (D20=35.2172, D10=35.5971, D4=36.1122, NORMAL=36.4943) |
| V14.best=last NORMAL=36.7806 | [v14_best JSON](review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json) | **✓** (D20=35.4249, D10=35.8101, D4=36.3677, NORMAL=36.7806) |
| V13 lambda_img=0, lambda_kl=0 | V13 train log grep | **✓** verbatim |
| V14 config = v7_seed1337.yaml | V14 train log grep | **✓** |
| V14 − V7 = −0.0004 dB | 36.7806 − 36.7810 | **✓** |
| V13 ≠ V8 train config (Plan F vs V7) | [PLANF L70 / L72](review/0516/PLANF_FINAL_ANALYSIS_20260516.md) + V13 train log | **✓** prompt 自己 §1.4 X3 也已 acknowledge |
| V6_NOISE.best=36.7437, V8.best=36.4729, V7.best=36.7810 | PLANF Table L68-L72 | **✓** verbatim |
| V18.best=36.8112, V18.last=36.8426 | Round 16 §1.2 (上游已签) | **✓** (信任上游 ledger) |
| **PLANF 已有 paired t = 74.6 (V7−V8 NORMAL) over 7403 slices, win rate 91.5%** | [PLANF L43](review/0516/PLANF_FINAL_ANALYSIS_20260516.md) | **✓** ← 本轮 prompt 漏的关键武器 |

**所有 §1 数字 100% verify**. 但 **prompt §2 SNR 方法论错误**, 见 §2.

---

## 2. Prompt §2 SNR analysis 的方法论问题 (HIGH)

### 2.1 Prompt 自己引用的 PLANF 已经有更强的 noise estimator

PLANF_FINAL_ANALYSIS L43 (V7−V8) 用的是 **per-slice paired t-statistic**:

> NORMAL PSNR_clip3: V7 36.7810 dB vs V8 36.4729 dB (Δ +0.30803 dB; **paired t = 74.6 over 7403 slices; per-slice win rate 91.5%**).

即 prompt 的 substrate **已经包含** per-slice 7403-sample paired comparison 的工具链 (eval 脚本 dump per-slice JSON, n=7403 paired). 但 Round 17 prompt §2 选择用 **single global mean delta 单点 |V14 − V7| = 0.0004 dB** 作为 noise scale.

这是**方法论降级**: 7403-sample paired-t → 1-sample one-shot delta.

### 2.2 实际 SEM 估计 (来自 V13/V14 JSON 自己的 per-slice std)

| run | NORMAL per-slice std (n=7403) | SEM of global mean = std/√n |
|---|---:|---:|
| V13 | 7.9132 dB | 0.0920 dB |
| V14 | 7.9777 dB | 0.0927 dB |
| (V18 估计 ≈ 同量级) | ~7.9 dB | **~0.092 dB** |

**含义**:
- V18.best − V7.best = +0.0302 dB **< 1/3 × SEM (0.092)** → unpaired 看是噪声内
- V18.last − V7.best = +0.0617 dB **< 1 × SEM (0.092)** → unpaired 看也在噪声内

但 **paired-sample** 分析可以救回来, 因为 V18 / V7 用 same slice. 关键问题: prompt 用 unpaired single-seed (|Δ|=0.0004) 当 noise std, 这是**两个错的方向都犯了**:
- 既不是 unpaired global SEM (= 0.092 dB, 大得多)
- 也不是 paired-sample σ_diff (需要算, 但有方法)

### 2.3 数学上 prompt §2 表的 "75× SNR" 含义

Prompt §2 表:
> V18.best − V7.best = +0.0302 dB → SNR ~75× (Δ / noise)

数学上 = 0.0302 / 0.0004 = 75. 但 **0.0004 不是 σ, 是 one-shot 估计自身的值**. 把一个单点估计当作分布的 σ, 在统计上不成立. 正确表述:

- "V14 给出 1-shot 估计 |Δ_seed| ≤ ~0.0004 dB; 不能用单点估计 σ" (correct)
- "若假设 σ_seed = O(0.001 dB), 则 V18.best Δ 是 O(30σ); 若 σ_seed = 0.01 dB (大致 SEM/3), 则 V18.best Δ 是 O(3σ)" (correct interpretation)
- Prompt 表说的 "SNR 75×" **隐含 σ = 0.0004**, 不成立

### 2.4 Reviewer A 推荐的免费修复

**零 GPU 成本**:
1. 重跑 V18 / V7 eval (其实不用重跑, JSON 已经存在), 提取 per-slice PSNR_clip3
2. 对 7403 个 paired (V18.last_slice_i − V7.best_slice_i) 算: mean, σ_diff, paired-t, win-rate
3. 与 PLANF V7−V8 (paired t=74.6) 同型 report

预期产出:
- **若 paired t > 30**: V18 +0.06 是 robust 真信号, paper 可写
- **若 paired t ∈ [3, 30]**: V18 +0.06 是 weak 但 statistically significant 信号
- **若 paired t < 3**: V18 +0.06 在 paired-sample 内也不显著, 必须降为 ablation 或不写

**这一步不做完, 任何 V18 写不写的决策都没有 evidentiary base**.

---

## 3. Q1-Q7 逐条

### Q1 — MODIFY (prompt 自己的命题方向对, 但漏了 paired-t 这条免费证据)

- V14 单次 seed → 严格命题: **"V18 信号超过单次 seed 扰动"** (这条 prompt 自己说的正确)
- 但 prompt 把 V14 |Δ| 当 σ 写进 SNR 表 → **方法论错误** (§2.1-§2.4)
- **不需要** 先跑 V14b/V14c 拿多 seed std; **第一步是免费 paired-t** (§2.4)
- 多 seed 是第二阶段, 仅在 paired-t < 3 时才需要

**最严格可说命题** (Reviewer A 推荐): "V18 信号 +0.06 dB 超过单次 V7-seed perturbation by ≥150×, 但尚未经 per-slice paired-t 检验; 在 paired-t 完成前不应作为 paper-critical claim".

### Q2 — APPROVE 主命题, **REJECT** V13-vs-V8 Grönwall 推论

**V13 vs V14 (= V13 vs V7) → image_aux 单变量贡献 +0.29 dB**: ✓ APPROVE.
这是**唯一干净的 disambig**, 因为 V13 / V14 share train config + Grönwall step_weights, 只差 image_aux 一个变量.

**V13 vs V8 → Grönwall 单独效应**: **REJECT prompt §1.4 X3 表述**.

数据状态 (PLANF 自己说的 + prompt §1.4 X3 自己 acknowledge):
- V13 = V7 train config + Grönwall step_weights + image_aux **OFF**
- V8 = **Plan F train config** + Grönwall step_weights + image_aux **OFF**

V13 vs V8 还有 train config (Plan F vs V7) confound. 单个 0.02 dB delta **不能**归因 Grönwall — 它混合了 (1) Grönwall 是否单独有效, (2) train config 改变是否独立有效. Prompt §1.4 X3 acknowledge 这点然后还是说 "已足够说明 Grönwall 不是主线收益来源" — **这是 acknowledgment-then-overreach 模式**.

**正确表述**: "V13 vs V8 ~0.02 dB 量级反映 (Grönwall + train_config) 联合微弱效应; Grönwall 单独贡献需要新 run (V13 train config + V8 step_weights) 才能 disambig".

**是否值得跑 Grönwall disambig**: NO. V8 train config 自身可能已 sub-optimal, 不必为退役变量 budget GPU.

**是否撤旧叙事**: YES, 但表述要精确 — 撤 "Grönwall step_weights 是 transport 设计胜利" 没问题; 不要替换为 "Grönwall 在噪声内" (这是 unverified 推论).

### Q3 — MODIFY (合理但 contingent on paired-t)

**V18 信号判定** (Reviewer A 立场):
- "≈ 150× single-seed delta" → 可能真 effect, **但量级 < 1/2 unpaired SEM**
- "image_aux +0.29 dB 的参照系" → 量级反差 ~5×
- "paper 主结果 vs ablation" → **取决于 paired-t**:
  - 若 paired-t > 30: V18 可作为 secondary 结果写
  - 若 paired-t ∈ [3, 30]: V18 写为 ablation 一行, 不上主图
  - 若 paired-t < 3: V18 + KL 整条线全部 ablation 化, paper 主体讲 image_aux

**V18 机制最合理解释** (假设真有 +0.06): "**LoRA decoder capacity (rank=32 last 2 blocks) + 从 V7.best step_160000 起的 35K 普通 fine-tune 训练**". Round 16 A3 already confirms KL 设计无 measurable 贡献; 剩余 +0.03→+0.06 dB increment (165K→200K) 与 "继续 35K 训练" 不可区分.

**反证测试** (低优先级): 跑 "V7 + LoRA(无 KL) + warm-start V7.best + 35K 训练到 step 200K" → 若 +0.06 dB 复现, 证机制为 capacity+train_time. 但 A3 已经测了 capacity-only 到 170K (matched-step tie), 多跑 30K 才能严格证明; 不强推.

### Q4 — 主战略 Hybrid B+A4 (with F0 prerequisite); 备选 = pure B

**主推: Hybrid B+A4 + F0 prerequisite**:

执行序列:
- **F0 (now, 0 GPU)**: 跑 paired-t analysis V18.last vs V7.best per-slice on 7403 slices. 1-2 小时 CPU 工作. 输出 = paired_t, σ_diff, win_rate, p-value. **决定后续所有走向**.
- **F1 (after F0)**:
  - 若 paired-t > 30: 启动 B (paper draft) + 并行 A4 (image_aux schedule tuning, 7 days)
  - 若 paired-t ∈ [3, 30]: 启动 B (paper draft, V18 降为 ablation row), 暂不 A4
  - 若 paired-t < 3: 启动 B (paper draft 不含 V18 主结果), 跳过 A4, 直接 image_aux 主线
- **F2 (parallel to F1)**: V18 + seed=1337 跑 (V18-seed-control), 7 days. 仅在 F0 paired-t > 30 时启动, 用于 final paper robustness check

**拒绝**:
- **C (architecture pivot)**: 4-8 周新项目, 与现 V7 系列无关; 在没用尽现有数据 (paired-t 都没跑) 之前 pivot **方法论早**
- **D (data pivot)**: 数据获取通常涉及 PHI / clinical 协议, **超 PSNR optimization 范畴**, 不是 Round 17 该决策
- **A2 (DiT → flow matching/consistency)**: arch 跨度过大, 等同 C 的轻量版, 与 A1/A4 不在一个 EV 量级
- **E (terminate)**: 现在终止**浪费 image_aux +0.29 dB 这个 publishable 强信号**; 至少完成 paper 再考虑

**Hybrid 合理性**: prompt §3.6 提出 B+A4. Reviewer A 改造 **B+F0+A4** — 把 F0 (paired-t) 作为前置 gate. F0 失败 → B but no A4 (因为 A4 假设 image_aux 还能 squeeze, 但 image_aux 主导性如果 V18 都不显著, 已经够 paper). F0 通过 → B+A4 paralleled.

**第 6 选项 F (V19 = use_pred_latent=false corrected KL)**: **REJECT** — 与 Round 16 + Round 7 standing rule (V18-clean 永久撤销) 抵触, 不该作为 Round 17 选项重提.

### Q5 — APPROVE narrative B' as main, ADD F' as candidate, REJECT D'/E'

**narrative B'** (image_aux 主结果): **APPROVE 作为主候选**.
- 强度: V13 vs V14 paired-t (没跑但可跑, V7 vs V8 paired-t = 74.6 同型) → 强 evidence
- venue fit: MICCAI / Med Image Anal / IEEE TMI 接受 "ablation-driven design study", 这条 narrative 直接命中
- 风险: 项目 1 年只 +0.29 dB 量级 contribution 偏小, 但 PET-specific niche 通常接受

**narrative C'** (decoupling 主结果): PARTIAL APPROVE 作为 SECONDARY angle.
- 强度: Round 16 A3 decode/chain decoupling 是干净结果 (+0.17 direct decode → +0.03 chain), 配合 paired-t 可写
- 但 decoupling 单独不够 paper, 需要配合 B' 或 D'

**narrative D'** (negative result ceiling): **REJECT 作为主**.
- "0.1 dB ceiling" 是**绝对值描述**, 没有 baseline normalization, 投出去 reviewer 会 reject "你做的是个 PET vs natural image 量级不同的 dataset, 不能跨域比较"
- "image_aux dominates" 是 positive finding, 不是 negative; D' 把它 frame 成负面是 PR 错误

**narrative E'** (feasibility study): MARGINAL APPROVE 作为 fallback.
- 适合 MICCAI workshop / ISBI short paper, 不适合 TMI / Med Image Anal
- 若 paired-t < 3, E' 是唯一可走路径

**Reviewer A 新候选 F'** (combined): 
> "Auxiliary supervision design is the dominant lever in PET latent transport (+0.29 dB via image_aux ablation); decoder capacity / KL pullback / additional fine-tune contribute marginal +0.03-0.06 dB structurally decoupled from transport bottleneck"

**venue 推荐**:
- 主 target: **MICCAI 2026** (deadline ~3 月, 现在写赶得上 if submit by Feb 2026)
- 备选: **Med Image Anal** (rolling, 2-month review) — 更慢但接受更系统的 study
- 不推荐: TMI (要求 broader impact, V18 量级不够), IPMI (theoretical 重)

**额外实验需求 to strengthen**:
- F0 (paired-t) — REQUIRED (free)
- V14b/V14c (multi-seed for image_aux too) — OPTIONAL, 让 +0.29 dB 上 multi-seed std → strengthens MICCAI narrative
- V18+seed=1337 (V18-seed-control) — OPTIONAL, 仅在 V18 上主图时需要

### Q6 — Reviewer A 控制优先级

| 控制 | EV | Cost | 建议 |
|---|---|---|---|
| **F0: paired-t V18 vs V7 per-slice** | **HIGH** | **0 GPU, 1-2 hr CPU** | **MUST DO NOW** |
| F0b: paired-t V13 vs V14 per-slice | HIGH | 0 GPU, 1 hr CPU | DO if F0 confirms paired-t methodology adequate |
| V18 + seed=1337 (V18-seed-control) | MEDIUM | 7 days GPU | Only if F0 paired-t > 30 AND V18 is main result candidate |
| V14b / V14c (multi-seed V7) | MEDIUM | 14 days GPU | Only if F0 paired-t inconclusive AND seed-distribution becomes paper-critical |
| V13 + LoRA (image_aux off + capacity) | LOW | 7 days GPU | **REJECT** — overkill; V18 信号已经 marginal, 在 image_aux=off regime 加 capacity 更弱 |
| V19 (V18 + use_pred_latent=false) | -∞ | 7 days GPU | **REJECT** — standing rule #5 (R7 user 永久撤销) 不可推翻; Round 16 已二次签 |

**优先级**: F0 → F0b → 决定是否启动其余.

### Q7 — Reviewer A 找到 6 个新偏差 (B61-B66), 详 §5

---

## 4. 立即可执行 (without GPU)

**现在 do**:
1. **F0 paired-t analysis**: write script to load V18.last / V7.best per-slice PSNR JSON (or re-extract from eval_first_hop_fullval_psnr_chain_mse outputs), align by slice_id, compute paired-t, σ_diff, win rate. **Reviewer A 估计 1-2 小时 implementation**.
2. **撤旧叙事 (但表述要精确)**: V18_design_rationale / Plan F / V7 历史叙事 — 撤 "Grönwall step_weights 是 transport 设计胜利" (B62), 替换为 "Grönwall + V7 train config 联合 ≈ image_aux=on 时与 Plan F 联合 在噪声内; image_aux 是主导"
3. **写 standing rule B67**: "项目 PSNR delta 报告必须 paired-sample per-slice + paired-t; 不允许用单次 seed |Δ| 作为 noise σ"
4. **复用现有 JSON 跑 SSIM eval** (Round 15/16 reviewer A 已提): 0 GPU, paper figure 工具箱扩展
5. **paper draft skeleton**: image_aux 主线 + decoupling 副线; V18 暂留 placeholder, paired-t 出来再决定是否上主图

**现在 不 do**:
1. **不要** 启动 V18-seed-control / V14b / V14c 任何 GPU 实验 (除非 F0 paired-t 完成且明确需要)
2. **不要** 推 architecture pivot / data pivot
3. **不要** 复活 V19 / V18-clean (standing rule #5)
4. **不要** publish 任何 "V18 +0.06 dB 显著" 表述 in paper draft (paired-t 之前)
5. **不要** terminate project (E)
6. **不要** 把 §1.4 X3 (Grönwall noise) 当作可引用结论 (B62)

---

## 5. 新偏差 B61-B66

### B61 [HIGH] §2 SNR analysis 方法论错误 (single-shot delta 当 σ)

- prompt §2 表用 V14 single global mean |Δ|=0.0004 作为 noise scale, 报 SNR = 75× / 150×
- 实际 (a) one-shot 估计不是 σ, (b) unpaired SEM ≈ 0.092 dB (大得多), (c) paired-sample σ_diff 需要 per-slice 算 (现有 JSON 有数据)
- prompt 自己 §2 末尾 acknowledge "这是一次估计, 不是分布... 当 noise floor std 不合理" — **acknowledgment-then-still-uses-it 模式**, 同 R16 B61 同型
- **修法**: 删 §2 SNR 表; 替换为 F0 paired-t TBD + caveat "paired-t 出来前 V18 信号 borderline"

### B62 [HIGH] §1.4 X3 Grönwall 推论 overreach

- prompt §1.4 X3 claim "Grönwall step_weights 单独效应在噪声内"
- 但同段落自己 acknowledge: V13/V8 还有 train config (Plan F vs V7) confound
- **acknowledgment-then-overreach** — 与 R16 B61 / R17 B61 同型
- **修法**: §1.4 X3 改为 "V13-V8 0.02 dB 反映 (Grönwall + train_config) 联合微弱效应; Grönwall 单独贡献无 disambig"; 不再用此推论支持 paper 撤旧叙事 (paper 撤旧叙事可以基于 V13-V7 的 +0.29 dB 直接, 不需要 V13-V8)

### B63 [MED] Narrative anchor 偏差 (Q5 4 候选)

- §5 Q5 列 narrative B' / C' / D' / E' 四个候选
- 全部围绕**已知 evidence reframing**; 没有候选问 "image_aux saturation point", "decoder vs RAE bottleneck split"
- D' "ceiling" framing 是 negative 包装 positive (image_aux dominates) finding, PR 错误
- **修法**: 加 narrative F' (image_aux + decoupling combined, §3-Q5 答案); 删 D'

### B64 [MED] §3 战略选项命名冲突 (A3 复用)

- §3.1 列 transport intervention 子选项 A1-A5, 其中 **A3 = "multi-step refinement (rollout 步数增加)"**
- 但项目 working memory 中 **A3 = V18-capacity-only experiment** (Round 13 命名, Round 16 全名 = "V18_capacity_only / Stage C A3")
- 这是命名 collision, 会让后续 reviewer 误读 (e.g. "A3 capacity-only" vs "A3 multi-step refinement")
- **修法**: §3.1 transport 子选项重命名 (T1-T5 或 A.1-A.5 with sub-dot), 避免与已签实验同名

### B65 [LOW] §1.4 "事实" framing 

- §1.4 标题 "三条立即派生的事实"
- 但 §2 自己也说 "V14 是一次估计, 不是 noise std"
- 把单点估计 frame 为 "事实" → reviewer 读到 X2 "image_aux +0.29 dB 是项目单一最大设计决策收益" 时不会问 "vs 多 seed 分布是不是 robust"
- **修法**: §1.4 改为 "三条立即派生的 1-shot observations"

### B66 [LOW] Hybrid B+A4 "低风险" 表述 

- §3.6 "Hybrid B+A4 (低成本试探)"
- 低**成本** 是真的 (7 day GPU + paper draft 并行); 但 "低风险" 是 **outcome 描述**, 不是 cost 描述
- A4 outcome = +0.05 dB / 0 dB / -0.05 dB 都可能. 若 0/-0.05, A4 浪费了 7 days
- **修法**: §3.6 改为 "低 GPU 成本试探, outcome 仍未知"

---

## 6. 元评论 (Reviewer A 立场)

R17 prompt 是 Round 12 以来**战略层最完整**的一份 (5 选项 + Hybrid + Q4-Q6 互补 + §0 边界声明清晰). 数字层 100% verify. 它继承了 R16 prompt 的 voluntary self-exposure 风格 (Q7 让 reviewer 找偏差), 而且 §2 末尾自己 acknowledge V14 单次 seed 局限.

但**犯 2 个 HIGH 偏差** (B61 SNR 方法论, B62 Grönwall overreach) 是同型 R16 B61 (acknowledgment-then-still-commit). 这暗示**根本机制**: claude 在 substrate 干净 + 时间压力下 (V13/V14 完成 + GPU 全空闲) 倾向**用现有数据做出最大 narrative leverage**, 即使 acknowledge 了局限也照样推论.

**给 user 的建议**:
- **R17 整合时 强制 prerequisite F0** (paired-t paired-sample analysis), 不要 skip
- R17 决策 ≠ paper 决策. F0 paired-t 才是 paper 决策真正前置. 没 F0, paper draft 写出来全部 V18 + 改 V18 都是空话
- B61/B62 必须修后再 dispatch 给 Reviewer B/C (否则同 anchor)
- 主战略 = Hybrid B+F0+A4, 不是 prompt 提的 B+A4 — F0 是 free + gate-keeping, 必须前置
- standing rule #5 (V18-clean 永久撤销) + standing rule (Reviewer A 新立) "delta 报告必须 paired-t" 必须 mark, R18 必须遵守

**Reviewer A 战略立场 summary**:
| 维度 | 立场 |
|---|---|
| paper 主线 | image_aux +0.29 dB (narrative B' / F') |
| V18 写不写 | F0 paired-t 决定 |
| 战略选项 | Hybrid B+F0+A4 |
| 项目终止 | NO (尚有 publishable 主线) |
| 继续 transport intervention | 仅 A4 (image_aux schedule), 其余 wait |
| arch / data pivot | REJECT |

---

## 7. 输出元

- reviewer: **Reviewer A**
- methodology: 4 个 JSON full verify + PLANF historical cross-check + Round 16 整合 cross-check + per-slice std → SEM 算术 + standing-rule audit
- 未与 Reviewer B/C 交流
- 撰写时长: 单轮, no iteration
- 关键独立发现:
  1. §2.1-§2.4 — prompt §2 SNR 方法论错误 (V14 单点当 σ); paired-t 是免费 fix, prompt substrate 自己已有 PLANF V7-V8 paired-t=74.6 示例
  2. §3.Q4 — Hybrid B+F0+A4 (F0 paired-t prerequisite); prompt §3.6 原 Hybrid 漏 F0 gate
  3. §3.Q5 — narrative F' (image_aux + decoupling combined) 比 prompt 提的 4 个候选更准确
  4. B61-B66 — 6 个新偏差含 2 个 HIGH (B61 / B62 SNR + Grönwall, 同 acknowledgment-then-overreach 模式)
