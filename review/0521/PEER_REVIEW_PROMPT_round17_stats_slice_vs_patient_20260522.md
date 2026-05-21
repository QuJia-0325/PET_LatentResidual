# Peer Review Round 17-Stats — Slice-Level vs Patient-Level Paired-t (F0 Methodology Audit)

- date: 2026-05-22
- branch: foc_lite_hop0 (commit 33a63a2)
- 主审对象: Codex 在 [CODEX_TASK_ROUND17_EXEC_FIXES_20260522.md](./CODEX_TASK_ROUND17_EXEC_FIXES_20260522.md) §3 提出的 F0 统计降级 ("slice-level paired statistics, not patient-level independent inference")
- 触发: Codex 自己在执行前发现原 task md 把 paired-t 当 patient-level 显著性, 主动降级为 slice-level + 加 caveat
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**, **不互见 codex EXEC_FIXES 之外的本仓库 reviewer 文档**
- 范围: **纯统计学/方法论审计**. 不审战略 (Round 17 已签), 不审 F0/A4 执行 (Round 17-Prep 已签), 不审 codex 6 处 fix 的其余 5 项 (path_guard / A4 output / git remote / V7 baseline / V18-cap guard — 这些都是非争议的执行正确性修复)

---

## 0. 本轮范围与边界

**本轮唯一审**:

> Codex 把 F0 paired-t 从 "patient-level significance" 降级为 "slice-level paired statistics + bootstrap CI + win-rate"; **这个降级是否方法论上站得住** 让 V18 信号判定可以入 paper, 还是必须继续降级 (例如完全去掉 p-value 只报 effect-size), 还是反过来 — 通过某种 cluster-robust 矫正能恢复到 patient-level inference.

**本轮不审**:
- F0 / A4 是否值得做 (Round 17 已签)
- F0 数据源 / 路径 / CLI flag (Round 17-Prep 已签 + codex EXEC_FIXES 已修)
- A4 yaml 字段 (Round 17-Prep 已签)
- codex EXEC_FIXES 的其余 5 项 (非争议执行修复)

---

## 1. 事实锚点 (本仓库本地 verify, 不要在审计中重 grep)

### 1.1 数据集 grouping 现状 (verify 完成)

- 数据集: PET low-dose chain rollout, `val` split, **7403 slices**
- 加载源: `pet_lr/data_first_hop.py` reads `raw_data_dir/preprocessed_data_<timepoint>.pt` (tensor blobs)
- **patient / subject / series / case ID 在数据管线里完全不存在**: `pet_lr/data_first_hop.py` 全文 grep 无 `patient|subject|series|case|study|exam` 标识符
- per-slice CSV header: `slice_idx, mse_D10, mse_D20, mse_D4, mse_D50, mse_NORMAL, mse_raw_*, psnr_D10, psnr_D20, psnr_D4, psnr_D50, psnr_NORMAL, psnr_raw_*`
- `slice_idx` 是 0-indexed 顺序位置, **不携带 patient grouping**
- 数据集来源是 PET 体数据, 典型每个病人 ~64-128 张 axial slices → 7403 slice 可能来自 **~58-115 个 patient**, 但 patient 边界目前无法恢复

### 1.2 当前 F0 实际计算 (codex 已写入脚本模板)

对每个 ckpt (V13 / V14 / V18.best / V18.last) vs V7.best, 对每个 timepoint (D20 / D10 / D4 / NORMAL), 在 7403 slice 上算:

| 量 | 公式 | 解读 |
|---|---|---|
| `mean Δ (dB)` | `np.mean(psnr_X - psnr_V7)` | effect size point estimate |
| `std Δ (dB)` | `np.std(delta, ddof=1)` | per-slice 离散度 |
| `paired SEM` | `std / sqrt(7403)` | **slice-level** SEM, 假设 slice 独立 |
| `paired t` | `mean / SEM` | **slice-level** t 统计量 |
| `two-sided p` | `scipy.stats.ttest_rel(arr, ref)` | **slice-level** p 值 |
| `Cohen's d` | `mean / std` | standardized effect size |
| `win-rate` | `(delta > 0).mean()` | slice-level 胜率 |
| `bootstrap 95% CI` | 5000-iter slice resample | **slice-level** bootstrap CI |

### 1.3 Codex 自己的 caveat (写入 F0 报告 markdown header)

> "patient/volume IDs are unavailable, so t/p/bootstrap are slice-level statistics, not patient-level independent inference."

### 1.4 Round 17 substrate 上的关键 V18 数字 (作为这次审计要给意见的目标对象)

| comparison | mean Δ NORMAL (dB) | 量级 |
|---|---:|---|
| V18.best − V7.best | +0.0302 | 小 |
| V18.last − V7.best | +0.0617 | 小 |
| V13 − V7 (image_aux off) | −0.2867 | 大 (作为 sanity, paired-t p 必然 < 1e-50) |
| V14 − V7 (d_pure) | −0.0004 | 0 (作为 sanity, paired-t p 应不显著) |

---

## 2. 关键统计学问题

### 2.1 Slice-level paired-t 真的高估显著性吗?

PET val 集 7403 slice 来自 ~58-115 病人. 同一病人的相邻 slice 在 PSNR 上**高度相关** (相同 anatomy, 相同 scanner noise pattern, 相同 lesion). 这意味着:

- 真有效 sample size N_eff ≪ 7403 (可能在 100-500 量级)
- slice-level SEM 比真 patient-level SEM **小 ~4-8×**
- slice-level t 比 patient-level t **大 ~4-8×**
- slice-level p **远小于** patient-level p

如果 V18 真 effect 是 +0.03 dB, slice-level paired-t 可能给出 `p < 1e-8` (虚假强显著), 而 patient-level paired-t 在 ~100 病人上可能只给 `p ≈ 0.05` 或不显著. 

### 2.2 是否有不需要 patient ID 的 cluster-robust 矫正?

候选方法 (请 reviewer 评估):

- **A. 完全放弃 p-value**, 只报 mean Δ + bootstrap CI + win-rate + 与 V14 noise floor 比较
- **B. 设计有效 N_eff 估计**: 用 slice 间空间自相关 (e.g. autocorrelation function on slice_idx) 估 effective sample size, 然后 inflate SEM
- **C. Cluster-by-slice-block bootstrap**: 假设每个 patient 连续 ~64 slice, 用 64-slice block bootstrap 代替 i.i.d. slice bootstrap; 当作 ad-hoc patient-level 近似
- **D. 用 V14 noise floor 替代 paired-t**: V14 也是 7403 slice, 它的 |mean Δ| = 0.0004 是 "完全相同设置 + 不同 seed" 的对照, 直接当 minimal detectable effect 用 — V18 +0.03 dB 是 ~75× V14 的 mean Δ, 这本身比 slice-level p 更诚实
- **E. 重跑 V7 + V18 evaluator 让它保留 patient grouping**: 修 eval 脚本, 让 per-slice CSV 多一列 `patient_id` (从 raw blob 的 slice → patient 映射推回, 如果 blob 里有这个 metadata)
- **F. 接受 slice-level + 强 caveat + 强 effect-size 报告, 不在 paper 主文用 p-value 说话**

### 2.3 paper venue 标准

MICCAI / TMI / Med Image Anal 等 venue 对统计 inference 的常规要求:
- 主表里 ΔPSNR 的 inference 通常要 **patient-level** (paired Wilcoxon / paired t on per-patient mean)
- slice-level 统计可以作为 supplementary 但不作为主结果
- 如果只能 slice-level, 一般要求显式 caveat + per-patient bootstrap (即上面 C 方案)

### 2.4 V18 信号的最严判定

如果方法论按 patient-level (或合理 cluster-robust) 矫正, V18.last − V7.best = +0.0617 dB 是否仍能被宣称为 "real signal worth ablation reporting"?

---

## 3. 给 reviewer 的 6 个问题

### Q1 — Codex 的 "slice-level" 降级是否够稳?

请评估:

- Codex 把 paired-t 降级为 "slice-level paired statistics" 是否方法论上**充分**? 还是仍然在 paper 里会被 reviewer 拒?
- 添加 bootstrap CI + win-rate 是否真的弥补了 slice-level p 的虚假强度? 还是 bootstrap 本身也是 slice-level (是), 同样虚假强?
- 文末的 caveat 一行能否充当 venue 接受的免责声明? 还是必须有正式 cluster-robust 矫正?

### Q2 — slice-level paired-t 究竟高估多少?

PET val 7403 slice from ~58-115 patients. 请粗估 (不要求精确):

- N_eff 实际可能在什么量级 (100-200? 50-100? 500-1000?)
- slice-level SEM 比 patient-level SEM 小多少倍
- 对 V18 +0.03 dB / +0.06 dB 信号, slice-level p 与 patient-level p 的预期 gap 多大
- 这个 gap 是否会让 V18 从 "p<1e-10" 跌到 "p≈0.05" 甚至 "p>0.1"

### Q3 — 选项 A-F 哪个最稳?

请在 §2.2 的 A-F 中选**主推**+**备选**:

- A 完全放弃 p-value
- B 用 ACF 估 N_eff
- C block bootstrap (~64-slice 块)
- D 用 V14 noise floor 替代 paired-t
- E 重跑 eval 加 patient_id 列
- F 接受 slice-level + caveat + effect-size only

对每个未选项, 给出 1 句拒绝理由.

### Q4 — V18 信号在最严矫正后还能不能进 paper?

假设按你 §Q3 主推方案做了矫正后:

- V18.best − V7.best = +0.0302 dB 是否仍能宣称 "real"?
- V18.last − V7.best = +0.0617 dB 是否仍能宣称 "real"?
- 如果"real" 不成立, paper 里 V18 ablation 应改成什么措辞?
  - "decoder LoRA at this scale produces no detectable chain improvement"
  - "decoder LoRA improvement is within seed-perturbation magnitude"
  - "decoder LoRA improvement is below patient-level statistical detection"

### Q5 — V14 当 noise estimator 的 paper 措辞

无论 §Q3 选哪个, V14 (single-seed V7 reseed) 仍是项目唯一的 noise floor 直接测量. 请评估在 paper 里:

- V14 应该作为 "noise floor proxy" 写在 main text, 还是 supplementary?
- V14 一次 seed 的 |Δ| = 0.0004 dB 是否应明确说 "upper bound, not std"?
- 是否值得 paper 里专门有一段 "Statistical methodology" 章节讨论 slice-level vs patient-level + V14 角色?

### Q6 — 是否需要 codex 在 F0 之外补一个 patient-id recovery sub-task?

数据集源是 PET 体数据, raw blob 是 `preprocessed_data_<timepoint>.pt`. 请评估:

- raw blob 是否大概率含有可恢复的 patient/series boundary (e.g. 每 64-128 slice 一组)?
- 让 codex 写一个 ~50 行的 patient_id recovery script (从 raw blob 的 metadata / slice ordering / 维度推断 patient 边界), 是否值得?
- 如果 patient_id 不可恢复, 是否应该接受 slice-level + 在 paper 显式说 "patient grouping was not recoverable in the preprocessed pipeline; this is a limitation"?

---

## 4. 资料目录

### 4.1 本轮主审

- [CODEX_TASK_ROUND17_EXEC_FIXES_20260522.md](./CODEX_TASK_ROUND17_EXEC_FIXES_20260522.md) §3 + §"Remaining Caveat"
- [CODEX_TASK_ROUND17_F0_A4_20260522.md](./CODEX_TASK_ROUND17_F0_A4_20260522.md) §F0.2 (脚本模板) + §F0.3 (pass 准则)

### 4.2 substrate (本轮不重审, 仅 verify reference 正确)

- [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md)
- [V13_V14_full_eval_analysis_20260521.md](../0516/V13_V14_full_eval_analysis_20260521.md)
- [V18_FINAL_RESULTS_20260518.md](../0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md)
- [PLANF_FINAL_ANALYSIS_20260516.md](../0516/PLANF_FINAL_ANALYSIS_20260516.md) (含 V7-V8 paired-t = 74.6 over 7403 slice 的同型先例)

### 4.3 代码 (verify patient ID 不可恢复用)

- [pet_lr/data_first_hop.py](../../pet_lr/data_first_hop.py) (数据加载, 全文无 patient/subject/case/series id)
- [review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py](../0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py) (eval 脚本, 输出 per-slice CSV, 只含 slice_idx)

---

## 5. 输出格式

每位 reviewer 独立产出 markdown:

### 5.1 6 个问题逐条 (APPROVE / MODIFY / REJECT)

### 5.2 主 verdict
请明确选 1:

- **ACCEPT-SLICE-LEVEL**: Codex 现稿 slice-level + bootstrap + win-rate + caveat 已够稳, 可直接用做 paper ablation 显著性依据
- **MODIFY-WITH-CLUSTER-CORRECTION**: 必须加 §2.2 中某 1 个矫正 (具体哪个) 才能 paper-defensible
- **MODIFY-DROP-PVALUE**: 完全去掉 p-value, 只报 effect-size + V14 noise + bootstrap CI; V18 信号判定靠 magnitude ratio 不靠 p
- **REJECT-NEEDS-PATIENT-ID**: 必须先恢复 patient ID, 否则 paper 里 V18 不能宣称显著

### 5.3 V18 信号最终判定
按你 §5.2 主 verdict 后:

- "real and worth headlining"
- "real but ablation-only"
- "real but should be reframed as decoupling diagnosis"
- "borderline below patient-level detection threshold"
- "not significant at patient level"

### 5.4 paper 写法建议
1-3 句话, V18 ablation 在 paper 里该怎么写措辞.

### 5.5 新偏差 (B83+)
若本 prompt 自身在统计学讨论上引入 anchor / overreach / 简化 narrative.

---

## 6. 角色提示

你不在审战略, 也不在审执行. 你在审**一个具体的统计学决策**: F0 paired-t 用 slice-level 行不行.

这个决策直接决定 paper 主文里 V18 部分能写多硬:
- 如果 slice-level 够稳 → "V18 produces a small but statistically significant +0.06 dB improvement (paired t, n=7403 slices)"
- 如果 slice-level 不稳 → "V18 produces a +0.06 dB improvement at slice level; patient-level inference was not feasible due to missing grouping in the preprocessed pipeline"

这两个写法在 reviewer 眼里完全不同等级. **本轮 review 决定 paper 写哪个**.

请直接给观点, 不要因为"稳妥"两个都列推给 user 决策疲劳.
