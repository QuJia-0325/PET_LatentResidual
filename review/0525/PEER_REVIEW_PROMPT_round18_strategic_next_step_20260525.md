# Peer Review Round 18 — Strategic Next Step (Post-A4-Bracket, No-More-Tuning Mandate)

- date: 2026-05-25
- branch: foc_lite_hop0 (commit f2206cc)
- 主审对象: **A4 bracket 结果 + Round 18 战略选项 X1-X5**
- 触发: A4-mid 出现项目最大单实验信号 (+0.113 dB, 280× V14 noise); user 明确"不想是调调参数之类的小打小闹"
- 上游分析: [ROUND_18_A4_BRACKET_ANALYSIS_20260525.md](./ROUND_18_A4_BRACKET_ANALYSIS_20260525.md)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

A4 bracket 结果已成为项目重心. Round 17 / 17-Prep / 17-Slots / 17-Stats 全部已签. Round 18 是**战略重排**:

**本轮审**:
- A4 +0.113 dB 信号的真假与权重 (vs single-seed limitation, slice-level only)
- image_aux mechanism (M1-M4) 哪个候选合理 / 怎么 disambig
- user "不调小参" 约束下 5 个 substantive 方向 X1-X5 的优先级
- paper narrative 是否立即重写 (image_aux 为主, V18 为 ablation)
- 是否漏关键方向 X6+

**本轮不审**:
- A4 yaml / training execution (已 push 完成)
- F0 paired-t 协议 (Round 17-Stats 已签 slice-level only)
- V18 / KL / decoder LoRA 个体决策 (Round 13-17 已签)

---

## 1. A4 substrate (raw verify, 不要求重读)

来源: `review/0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_REPORT_20260525.md`

### 1.1 4 点 image_aux response curve

| run | image_aux λ | NORMAL PSNR_clip3 | Δ vs V7 | Δ / V14 noise |
|---|---:|---:|---:|---:|
| V13 | 0.00 | 36.4943 | −0.2867 | −717× |
| A4-low | 0.02 | 36.7010 | −0.0800 | −200× |
| V7 | 0.04 | 36.7810 | 0 | 0 |
| **A4-mid** | **0.08** | **36.8939** | **+0.1130** | **+283×** |

D20 / D10 / D4 全部同向, A4-mid best.pt = last.pt @ step 160000.

### 1.2 与 V18 系列对比

| 工程 | 投入 | NORMAL Δ vs V7 | 与 A4-mid 比 |
|---|---|---:|---|
| V18.best (decoder LoRA + KL) | 6w 设计 + 7 review 轮 | +0.0302 | 27% |
| V18.last (训到 200K) | + 7d GPU | +0.0617 | 55% |
| **A4-mid (image_aux λ 0.08)** | **1 yaml 字段, 0 代码** | **+0.1130** | **100%** |

A4-mid 单一字段调整 = V18 整套 + 35K 额外训练的 1.8 倍效果.

---

## 2. 4 个 image_aux mechanism 候选 (reviewer 选)

`image_aux` 实现 = `l1(decode(z_pred), x_target) + 0.25 * ssim + 0.1 * seam`. 把 pixel-space supervision 从 frozen RAE decoder 反传回 transport latent. 4 个候选机制:

- **M1**: transport DiT 在 latent space undertrained, pixel-gradient 提供更强 signal (类 GAN perceptual loss)
- **M2**: SSIM 子项捕获 structure, 而 latent MSE 只对 magnitude 敏感
- **M3**: seam 子项防 patch boundary, 改善 chain 累积 error
- **M4**: image_aux 隐式 regularizer, anchor z_pred 到 decoder manifold (机制类 A3 capacity 但反向)

---

## 3. 5 个 substantive 方向 (reviewer 选 1-2)

### Option X1 — image_aux 机制深拆

**实验**: 3 个 λ=0.08 ablation runs (l1-only / ssim-only / seam-only) + activation probe.
**成本**: 21d GPU + ~1 GPU day probe.
**EV**: 高 (paper mechanism story).
**风险**: 可能被 user 当"调子项 = 小打小闹".

### Option X2 — 架构升级: pixel-aware transport

**实验**: 替换 transport DiT → consistency model / shortcut model / flow matching with stochastic interpolant.
**成本**: 4-8w 设计 + 训练.
**EV**: 中-高 (如果 X2 出 +0.3 dB 是主刊级结果).
**风险**: 新项目级开销; 失败率 ~50%.

### Option X3 — 端到端联合: image_aux + decoder LoRA additive

**实验**: V18-like 但 λ_kl=0, λ_img=0.08, LoRA rank=32 / blocks=last-2.
**成本**: 1 run × 7d GPU.
**EV**: 中 (决定 V18 是否冗余).
**风险**: 看作"V18.v2 微改", 视设计 framing.

### Option X4 — Multi-step refinement / chain redesign

**实验**: 加 second hop / inference-time iterative refinement / chain dose schedule 改.
**成本**: 2-4w 设计 + 训练.
**EV**: 不确定.
**风险**: 改 chain 几何动 V13/V14/V18 可比性.

### Option X5 — Cross-anatomy / cross-tracer generalization

**实验**: 另一个 PET 数据集 (FDG / DOTATATE / PSMA) 重跑 V13 + A4-mid + V18.
**成本**: 4-8w (含数据获取).
**EV**: 高 if 有数据.
**风险**: 数据获取本身可能 block.

---

## 4. 给 reviewer 的 7 个问题

### Q1 — A4-mid +0.113 dB 真不真?

它是 single-seed, slice-level (patient ID 不可恢复). vs V14 noise = 283×. 请评估:
- A4-mid 是真信号还是 single-seed luck?
- 在 patient-level inference 不可达情况下, paper 写"+0.113 dB" 是否需要任何额外 ablation 才稳?
- 是否 V14 noise floor (单次 |Δ| = 0.0004) 真足以支持 +0.113 dB 作为"main result"声明?

### Q2 — image_aux mechanism M1-M4 哪个真?

请评估 §2 的 4 个候选:
- 哪个最可能真? 哪个被现有 substrate 已经间接否定?
- 区分 M1 vs M2 vs M3 vs M4 的最小 experiment 是什么 (≤ 1 run + 1 probe)?
- 是否还有 M5+ (我们没列的机制)?

### Q3 — user 约束"不调小参"下, X1-X5 哪 1-2 个最值得?

请按 (EV × 实现可能性 / 成本) 排序; 拒绝其它 3-4 个并给理由.

**特别评估**:
- X1 是不是 disguised parameter tuning (3 个 ablation runs 也算"调")?
- X2 是不是 over-ambitious (4-8w 设计 + 50% 失败率, 是否对 paper 总进度净负)?
- X3 是不是 disguised V18.v2 (V18 family 已经被 Round 16 降级, 复活是否合理)?
- X4 是不是 改 chain 后动 baseline (V13/V14/V18 数据失效)?
- X5 是不是 现实不可达 (是否真有 cross-tracer 数据)?

### Q4 — V18 在 paper 里现在的角色?

A4-mid (+0.113) > V18.last (+0.062) 接近 2 倍. V18 整套 KL + LoRA 已被 Round 16 证伪 KL 部分. 请评估:
- paper 头条是否必须是 image_aux schedule? 
- V18 应该:
  - (a) 完全撤出 paper
  - (b) 作为 "we also tried decoder LoRA, smaller effect" 一段
  - (c) 作为 main ablation 与 image_aux 联合分析
- 是否应该 cancel V18 family 全部 follow-up 实验 (X3 也包括)?

### Q5 — 是否漏关键方向 X6+?

reviewer 自己想 1-2 个我们没列的 substantive direction. 例如:
- X6: text-conditioned latent? (PET 临床报告 + image 联合)
- X7: dose-conditional inference-time control (CFG-style on dose scale)?
- X8: 联合 CT-PET dual modal transport?
- X9: PET tracer-conditional embedding?
- 其它?

### Q6 — paper 时间表

假设 X1 (机制) + X3 (additive 测试) 都做完后, paper 投递时间表:
- MICCAI 2027 (deadline ~2026-12, 还 ~7 个月) 是否可行?
- 是否值得追 TMI/MedIA 较长 cycle 换更深 mechanism + cross-dataset?
- 现在 paper draft 应该启动到哪个阶段 (outline / methods / 全文)?

### Q7 — 偏差审查 (B87+)

请特别检查本 prompt 是否引入:
- **A4-mid 过度浪漫化**: 把 +0.113 dB 当"项目救星", 忽略 single-seed limitation
- **image_aux fetish**: A4 = winning, 所以其它方向都被 anchor 为不重要
- **architecture pivot 过度乐观**: X2 失败率 50% 但被列为高 EV
- **"不调小参" 教条化**: X1 mechanism ablation 被定为"调小参", 但 paper 实际需要它

---

## 5. 资料目录

### 5.1 本轮主审

- [ROUND_18_A4_BRACKET_ANALYSIS_20260525.md](./ROUND_18_A4_BRACKET_ANALYSIS_20260525.md) (claude 分析)
- [A4_BRACKET_FULLVAL_REPORT_20260525.md](../0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_REPORT_20260525.md) (codex 报告)
- [A4_BRACKET_FULLVAL_SUMMARY_20260525.json](../0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_SUMMARY_20260525.json) (raw JSON)

### 5.2 上游 context (本轮不重审)

- [REVIEW_INTEGRATION_round17_20260522.md](../0521/REVIEW_INTEGRATION_round17_20260522.md) (Hybrid B+F0+A4-light 战略)
- [REVIEW_INTEGRATION_round17_slots_20260522.md](../0521/REVIEW_INTEGRATION_round17_slots_20260522.md) (A4-bracket 扩展)
- [REVIEW_INTEGRATION_round16_20260519.md](../0517/REVIEW_INTEGRATION_round16_20260519.md) (V18 KL 死)
- [V13_V14_full_eval_analysis_20260521.md](../0516/V13_V14_full_eval_analysis_20260521.md) (image_aux off / d_pure substrate)
- [V18_FINAL_RESULTS_20260518.md](../0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md) (V18 canonical)

### 5.3 代码 context

- [V7_gronwall_raw.yaml](../0505/local/configs/V7_gronwall_raw.yaml) (image_aux 配置位置)
- [train_first_hop.py](../../train_first_hop.py) (image_aux loss 实现)

---

## 6. 输出格式

每位 reviewer 独立产出 markdown:

### 6.1 7 个问题逐条 (APPROVE / MODIFY / REJECT)

### 6.2 主战略 verdict (单选 + 备选)
- X1 (机制深拆)
- X2 (架构升级)
- X3 (image_aux + LoRA additive)
- X4 (chain redesign)
- X5 (cross-dataset)
- 自定义 X6+ (说明)
- Hybrid (说明组合)

### 6.3 V18 角色判定
- (a) 完全撤出 paper
- (b) 次要 ablation
- (c) 主 ablation 与 image_aux 联合分析

### 6.4 image_aux mechanism 主推
- M1 / M2 / M3 / M4 / M5+ (说明)
- 最小 disambig experiment

### 6.5 paper 时间表
- MICCAI 2027 / TMI / MedIA / 其它
- 现在 paper 应启动到哪一步

### 6.6 新偏差 (B87+)

---

## 7. 约束与提醒

- **本轮不重审** Round 12-17 已签内容
- **user 硬约束**: "不想是调调参数之类的小打小闹". 如果 reviewer 主推 X1 (子项 ablation) / X3 (image_aux+LoRA), 必须有强论证为何不算"调小参".
- **特别关注**:
  - A4 single-seed + slice-level limitation 是否影响"+0.113 dB 是 main result"声明
  - V18 在 Round 16 已被部分降级, 现在是否应彻底撤
  - X2 (architecture pivot) 是否成熟到 reviewer 敢推 vs 风险太大

---

## 8. 角色提示

你不在审实验细节. 你在帮 user + claude 做**项目下半场战略**:

- A4-mid 已经把 paper 从"marginal +0.06 dB" 提升到"meaningful +0.11 dB"
- 但 user 拒绝继续微调
- 选 X1-X5 的哪个组合, 决定接下来 1-3 个月的所有 GPU + 设计精力分配
- 错选 (e.g. X2 失败) = paper 推迟 6 个月; 漏选 (e.g. 没做 mechanism story) = paper venue 降两级

请给清晰观点, 不要稳妥列 4 个让 user 决策疲劳.
