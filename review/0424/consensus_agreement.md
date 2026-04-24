# 意见统一文档 — Author × Reviewer Round 7 共识

**Date**: 2026-04-24  
**状态**: 双方达成一致

---

## 1. 审查流程回顾

| Round | 文档 | 核心事件 |
|-------|------|---------|
| R1-R5 | `nightmare/00-05` | 5 轮独立 Opus 4.5 对抗评审 → 3.05/10 |
| R5 Rebuttal | `rebuttal_nightmare_r5.md` | Author 接受 F1/F3/F5-F10，争辩 200K 和 σ_seed |
| R6 Reply | `reviewer_reply_to_rebuttal.md` | Reviewer 纠正 3 点 → 3.25/10 |
| R6 Response | `proposer_final_response_r6.md` | Author 全面接受 3 个纠正 |
| R6 Signoff | `reviewer_signoff_r6.md` | Reviewer ACCEPTED + 指出 3 个 trap |
| R6 Dossier | `06_reasoning_dossier_on_rebuttal.md` | Reviewer 公开推理链 |
| R7 Verdict Map | `07_pre_registered_verdict_map.md` | Reviewer 预注册评分规则 |
| R7 Joint Spec | `joint_execution_spec_round7.md` | 双方联合执行规范 |

---

## 2. 双方已达成一致的核心决策

### 2.1 σ_seed 是最高优先级（双方一致 ✓）

- **4 seeds** × imgaux_boost × 50K（seeds: 42, 123, 456, 789）
- **全部在同一 codebase 重跑**（不复用旧 seed=42 结果）
- σ_seed 是**所有 PSNR 声明的全局校准**，不只是 hop0 专用
- 使用 `σ_upper ≈ 1.5 × σ_hat` 作为保守操作界

### 2.2 200K 条件化执行（双方一致 ✓）

- **必须在 σ_seed 结果出来后才能启动**
- σ_hat > 0.08 → 砍 200K
- σ_hat ≤ 0.08 → 200K 用 σ-unit gate 运行
- PASS: `mean(200K) - mean(50K) > 3·σ_upper AND CI lower > σ_upper`
- ABANDON at 150K: `Δ(150K-50K) < σ_upper OR Δ(150K-100K) < 0.5·σ_upper`

### 2.3 Path A 最高信息价值（双方一致 ✓）

- TF vs RO gap 诊断，跑 2 个 checkpoint（C + N1）
- 0.5 GPU-day，与 σ_seed 并行
- verdict matrix 预注册（exposure / capacity / mixed / near-ceiling）

### 2.4 Path C 作为诊断（双方一致 ✓）

- Decoder FT 不是方法，是估 RAE bound 的诊断
- 与 Path A 独立，可并行
- 需要 `model_first_hop.py` 添加 partial decoder unfreeze 支持

### 2.5 F2 概念纠正（双方一致 ✓）

- "decoder 非线性放大 tail" → 错误
- 正确表述："Gap_Transport 单调递增（10.74→15.70 dB），同比例改善在 dB 绝对值上表现为 tail 更大"
- 报告使用 normalized improvement 或 latent MSE 相对改善

### 2.6 叙事 reframe（双方一致 ✓）

- 从 "Pixel Injection Breaks First-Hop Bottleneck" 
- 到 "Systematic Diagnosis of Error Propagation in Multi-Hop Latent Flow Matching"

---

## 3. 执行时间线（双方统一版本）

| Day | 任务 | GPU-day | 产出 |
|-----|------|---------|------|
| 0 | Preflight：codebase hash 冻结、数据验证 | 0 | `round7_*.txt` |
| 1-2 | **A1 σ_seed**：4 seeds 并行 | 4 | `round7_sigma/summary.json` |
| 1-2 并行 | **A2 Path A**：C + N1 TF-vs-RO | 1 | `round7_pathA/{C,N1}/tf_rollout_gap_val.json` |
| 3 | **σ_seed 读数 → 200K 决策** | 0 | `200K_decision.md` |
| 3-5 | **A3 Path C**：decoder FT 诊断 | 1 | `round7_pathC/pathC_val.json` |
| 3-5 | **A6 200K**（条件化） | 2.7 | 有/无 gate 记录 |
| 6+ | **A4 F2 rewrite** + **A5 Pix2Pix**（可选） | 2 | CLAUDE.md 修正 |
| **总计** | | **~10 GPU-day** | |

---

## 4. Scripts 目录说明

| 脚本 | 对应诊断 | 用途 | 产出 |
|------|---------|------|------|
| `diagnose_error_budget.py` | **E1** | 将 PSNR gap 分解为 transport vs decoder 两部分。已在 0416 运行（n=7403），结论：transport 占 96-99% | per-hop gap 分解 JSON |
| `diagnose_foc_gap.py` | **E2** | 测量 full-step vs half-step 的 FOC 差异。验证 ODE 积分精度。已在 0416 运行，结论：差异 < 0.17%，FOC 假设被证伪 | foc_gap JSON |
| `eval_gt_latent_decoder_ceiling_clip3.py` | **Oracle** | 用 GT latent 直接 decode，测量 decoder PSNR 上限。已在 0422 运行，结论：ceiling 46-52 dB，decoder 非主导瓶颈 | per-timepoint ceiling JSON + CSV |
| `diagnose_tf_rollout_gap.py` | **Path A** | TF vs RO gap：区分 exposure bias 和 velocity capacity。488 行，预注册 verdict matrix | per-hop exposure/ceiling gap JSON + per-slice CSV |

**诊断链**：E1 → 定位到 velocity → E2 → 排除 ODE → Oracle → 排除 decoder → **Path A → 区分 exposure vs capacity**

---

## 5. 需要 Author 在启动前完成的代码改动

### 5.1 创建 4 个 σ_seed configs

基于 `imgaux_boost.yaml`，只改 `run_name` 和 `seed`：
- `pet_flow_first_hop_224_50k_sigma_s42.yaml` (seed=42)
- `pet_flow_first_hop_224_50k_sigma_s123.yaml` (seed=123)
- `pet_flow_first_hop_224_50k_sigma_s456.yaml` (seed=456)
- `pet_flow_first_hop_224_50k_sigma_s789.yaml` (seed=789)

注意：之前创建的 `sigma_seed123.yaml` 和 `sigma_seed456.yaml` 需要删除或替换（命名规范不同）。

### 5.2 Path C decoder partial unfreeze（~5 行代码）

在 `model_first_hop.py` 或 `train_first_hop.py` 中添加支持：
```python
# 只解冻 decoder 最后 N 个 block
rae_unfreeze_n = int(train_cfg.get("rae_unfreeze_last_n_blocks", 0))
if rae_unfreeze_n > 0:
    decoder_blocks = list(model.rae.decoder.blocks)  # 适配实际结构
    for block in decoder_blocks[-rae_unfreeze_n:]:
        block.requires_grad_(True)
```

### 5.3 Path C config

基于 `imgaux_boost.yaml`，改动：
- `freeze_rae: false`（或保持 true + 新的 partial unfreeze 开关）
- `max_steps: 5000`
- `--resume imgaux_boost_best.pt`

---

## 6. Reviewer 预注册的评分路径

| 总分区间 | 含义 | 达成条件 |
|---------|------|---------|
| **< 3.0** | 不可投稿 | 任何 KILL list 违规 |
| **3.25** (当前) | 需更多数据 | — |
| **3.8** | σ_seed + Path A 完成 | Causal +0.5 |
| **4.4** | + Path C + F2 rewrite | Novelty +0.5, Story +0.7 |
| **4.9** | + Pix2Pix baseline | Submission +1.0 |
| **5.2** (ceiling) | 所有 6 个 artifact 完成 | 最大值 |

---

## 7. 不再争辩的已关闭议题

以下由双方共同确认关闭，Round 7 不再重新审查：

- ✅ C vs N1 是单变量 clean ablation（Tribunal 验证）
- ✅ Patient-level split 正确（R4 C-auditor 验证）
- ✅ Hop0 不再声称为 "bottleneck"（F1 接受）
- ✅ iREPA / SeamRefiner 是死代码不作为贡献
- ✅ Mediation 实验不强制（nice-to-have）
