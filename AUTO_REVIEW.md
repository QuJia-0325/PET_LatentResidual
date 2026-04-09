# AUTO_REVIEW.md

## Purpose
对 **idea 本身**（不是只看单次实验结果）做多维度打分，并执行多模型对抗评审，专门审查：
- 224 级联异常：`D50→NORMAL < baseline` 且 `D20→NORMAL > baseline`
- hop0（`D50→D20`）latent forcing + pixel structure injection 是否真正解决首跳结构问题

---

## 1) Review Modes

### HARD（默认研究对抗）
- 多维度量化评分（见第 3 节）
- 双评审模型对抗：
  - Reviewer-A: `gpt-5.4`（主审）
  - Reviewer-B: `o3`（反对审）
- 必须执行分歧仲裁协议（第 5 节）

### NIGHTMARE（最高强度）
在 HARD 基础上增加：
- Reviewer-C: 仓库直接核查代理（独立读代码/JSON/CSV/log，不接受口头结论）
- 强制“最不利切片桶”判决（high seam-risk）
- 强制 claim-verification：结论逐条绑定证据文件路径

---

## 2) Reviewer Matrix（多模型对抗矩阵）

| Reviewer | Model/Role | 主要职责 | 允许否决项 |
|---|---|---|---|
| A | gpt-5.4（主审） | 给总分、主要缺陷、最小修复集 | 可否决“可投稿” |
| B | o3（反对审） | 专找漏洞、质疑归因、找 cherry-pick | 可否决“因果成立” |
| C | Repo Verifier（nightmare） | 独立核对代码与结果一致性 | 可否决“结果可信” |

**最终通过条件**：A/B/C（nightmare）均不触发否决，且总分达标。

---

## 3) Multi-Dimensional Idea Scoring（10 分制）

> 不是单指标打分；按维度加权汇总。

### 评分维度与权重
1. **Problem-Mechanism Fit（20%）**
   - idea 是否直接命中 `D50→D20` 瓶颈，而非间接改善 tail。
2. **Causal Evidence Strength（20%）**
   - 是否有 TF vs pure-pred、A/B/C/D 或等价分解证明“为何有效”。
3. **Robustness & Reproducibility（15%）**
   - seed、重复 eval、max-slices 稳定性。
4. **Adversarial Survivability（15%）**
   - 在 hard/nightmare 审核下是否仍成立。
5. **Artifact/Seam Mitigation Specificity（15%）**
   - 对 padding/seam 现象是否有分层证据（high seam-risk 桶）。
6. **Engineering Feasibility under Constraints（10%）**
   - 是否满足“不改代码/或仅最小改动需批准”与 clip3 规范。
7. **Novelty & Positioning（5%）**
   - 新意是否明确（方法/证据协议/问题定义层面）。

### 计算公式
`TotalScore = Σ(weight_i × score_i)`，每维 0-10。

### 判定阈值
- **PASS**: `TotalScore >= 7.0` 且无否决
- **GRAY**: `5.5 <= TotalScore < 7.0` 或存在轻度分歧
- **FAIL**: `< 5.5` 或任一硬否决触发

---

## 4) Hard Gates（硬门）

任一不满足，直接 FAIL（不进入主观讨论）：

1. **Metric Gate**
   - 评估必须是 `src.utils.metrics.calc_psnr_clip3`
2. **Evidence Gate**
   - 必须有 hop-level 结果；禁止只报 NORMAL 总均值
3. **Causal Gate**
   - 若 `D20↑` 且 `D50→NORMAL↓`，必须给出 hop1 gap 归因证据
4. **Repro Gate**
   - 至少 2 seeds + 2 次重复 eval
5. **Anti-cherry-pick Gate**
   - 同时报告 best/last；必须保留失败实验

---

## 5) Disagreement Protocol（分歧仲裁）

当 A/B（或 C）结论冲突时，按以下步骤：

1. **Issue Lock**：把冲突问题编号（D1, D2, ...）
2. **Evidence Binding**：每方必须引用文件路径+字段（JSON key/CSV列/代码行）
3. **Rebuttal Round（最多 1 轮）**：
   - 作者可提交最多 3 条反驳
   - 评审仅可基于证据裁决：`SUSTAINED / OVERRULED / PARTIAL`
4. **Ruling**：
   - 若仍冲突，以 **Verifier(C)** 结果优先（nightmare）
   - hard 模式下以“更保守结论”优先

---

## 6) Required Artifacts（缺任意一项=FAIL）

1. `outputs/clip3_eval/*/baseline/*.json`
2. `outputs/clip3_eval/*/variant/*.json`
3. `outputs/clip3_eval/*/baseline/*.csv`
4. `outputs/clip3_eval/*/variant/*.csv`
5. `tf_vs_pure_gap.json`
6. `seam_strata_summary.csv`
7. baseline/variant 配置快照
8. 审核轮次日志（本文件 Round Entries）

---

## 7) Round Entry Template

## Round {N} — Mode: {HARD|NIGHTMARE}

### Idea Under Review
- Variant: {V1|V2|V3|Custom}
- Core hypothesis: {...}

### Reviewer Scores (0-10)
| Dimension | Weight | A(gpt-5.4) | B(o3) | C(verifier, nightmare) | Fused |
|---|---:|---:|---:|---:|---:|
| Problem-Mechanism Fit | 0.20 |  |  |  |  |
| Causal Evidence Strength | 0.20 |  |  |  |  |
| Robustness & Reproducibility | 0.15 |  |  |  |  |
| Adversarial Survivability | 0.15 |  |  |  |  |
| Artifact/Seam Mitigation | 0.15 |  |  |  |  |
| Feasibility under Constraints | 0.10 |  |  |  |  |
| Novelty & Positioning | 0.05 |  |  |  |  |

- **TotalScore**: {x.xx}/10
- **Vetoes**: {none / A:... / B:... / C:...}
- **Verdict**: {PASS|GRAY|FAIL}

### Key Conflicts
- D1: ...
- D2: ...

### Rebuttal + Ruling
- D1 ruling: {SUSTAINED|OVERRULED|PARTIAL}
- D2 ruling: {...}

### Minimum Fixes for Next Round
1. ...
2. ...

### Evidence Index
- `.../file.json` key: `summary_psnr_clip3.D20.mean`
- `.../file.csv` column: `psnr_NORMAL`
- `pet_lr/model_first_hop.py:{line}`

---

## 8) Final Decision Policy

- **Submission-ready** only if:
  1) 连续两轮 `PASS`
  2) nightmare 模式通过一次
  3) high seam-risk 桶不退化
  4) 无 reviewer veto

否则输出：
- 未解决 blocker 列表
- 每个 blocker 的最小成本修复建议
- 是否建议继续本方向或转向

---

## Round 1 — Mode: NIGHTMARE

### Idea Under Review
- Variant: V1 + V3 (pre-execution audit)
- Core hypothesis: 前跳重平衡可改善 D50→D20；seam 分层与 TF/PURE gap 可验证 hop0 因果链。

### Reviewer Raw Responses

#### Reviewer-A (gpt-5.4 role)
- Strengths: 问题-机制对齐较好，配置层可执行，clip3 口径明确。
- Concerns: 目前仍是计划与假设，缺少 variant 实验结果与高风险桶证据。
- Decision: 进入执行轮，但本轮不能判定 claim 成立。

#### Reviewer-B (o3 opposition role)
- Objection-1: `D20↑, D50→NORMAL↓` 可能仅是 tail 偏置与 checkpoint 选择效应，未被排除。
- Objection-2: 未见双 seed + 重复 eval + best/last 同报，存在 cherry-pick 风险。
- Decision: 因果未建立，反对通过。

#### Reviewer-C (repo verifier role)
- Verified present evidence:
  - `outputs/diagnostics/strict_best_full/hop_difficulty_clip3_val_strict_best_full.json`
  - `outputs/diagnostics/chainstable50k_best_full/hop_difficulty_clip3_val_chainstable50k_best_full.json`
  - `outputs/first_hop_224_10k_strict_eval_clip3_best/first_hop_224_val_clip3_eval.json`
- Missing required artifacts (hard fail now):
  - `outputs/clip3_eval/*/baseline/*.json`
  - `outputs/clip3_eval/*/variant/*.json`
  - `outputs/clip3_eval/*/baseline/*.csv`
  - `outputs/clip3_eval/*/variant/*.csv`
  - `tf_vs_pure_gap.json`
  - `seam_strata_summary.csv`
- Decision: Evidence Gate fail（结果可信性不足）。

### Reviewer Scores (0-10)
| Dimension | Weight | A(gpt-5.4) | B(o3) | C(verifier, nightmare) | Fused |
|---|---:|---:|---:|---:|---:|
| Problem-Mechanism Fit | 0.20 | 8.0 | 7.0 | 6.5 | 7.2 |
| Causal Evidence Strength | 0.20 | 5.0 | 3.0 | 2.0 | 3.3 |
| Robustness & Reproducibility | 0.15 | 4.5 | 3.0 | 2.0 | 3.2 |
| Adversarial Survivability | 0.15 | 5.0 | 3.5 | 2.0 | 3.5 |
| Artifact/Seam Mitigation | 0.15 | 6.0 | 5.0 | 4.0 | 5.0 |
| Feasibility under Constraints | 0.10 | 9.0 | 8.0 | 9.0 | 8.7 |
| Novelty & Positioning | 0.05 | 4.5 | 3.5 | 3.0 | 3.7 |

- **TotalScore**: **4.84/10**
- **Vetoes**: `B: causal not established`; `C: evidence gate failed`
- **Verdict**: **FAIL**

### Key Conflicts
- D1: “是否已证明 hop0 机制改善了首跳因果瓶颈？”
- D2: “padding/seam 改善是否真实而非切片选择偏差？”

### Rebuttal + Ruling
- D1 ruling: **SUSTAINED**（当前仅有 baseline/strict 诊断，未有 V1/V2 结果）
- D2 ruling: **SUSTAINED**（缺少 `seam_strata_summary.csv` 与 variant 对照）

### Minimum Fixes for Next Round
1. 产出 baseline 与 V1（必要时含 V2）全量 val `clip3` JSON/CSV（best + last；至少 2 seeds）。
2. 生成 `tf_vs_pure_gap.json`（含 hop1 与 final gap，baseline vs variant）。
3. 生成 `seam_strata_summary.csv`（low/mid/high 桶，报告 D50→D20、D50→NORMAL、D20→NORMAL）。
4. 按 anti-cherry-pick 规则同报失败实验与 checkpoint 选择逻辑。

### Evidence Index
- `outputs/diagnostics/strict_best_full/hop_difficulty_clip3_val_strict_best_full.json` key: `summary.derived.tf_hop0_minus_no_transport_d20_db`
- `outputs/diagnostics/chainstable50k_best_full/hop_difficulty_clip3_val_chainstable50k_best_full.json` key: `summary.autoregressive_chain.NORMAL.mean`
- `outputs/first_hop_224_10k_strict_eval_clip3_best/first_hop_224_val_clip3_eval.json` key: `summary_psnr_clip3.D20.mean`
- `IDEA_REPORT.md` lines: 16-27, 46-55 (idea/gate definitions)
