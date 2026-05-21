# Peer Review Round 17 — Post-V13/V14 Strategic Decision

- date: 2026-05-21
- branch: foc_lite_hop0 (commit 52406e7, V13/V14 full-val eval 已 merge)
- 主审对象: **V13/V14 完成后, 项目战略下一步**
- 上游: Round 16 整合 (REVIEW_INTEGRATION_round16_20260519.md) 已确立 A3 杀掉 KL direct-decoder 叙事; Round 16 留下 4 条 "等 V13/V14" 决策, 现在 3 条可解
- 当前状态: **A3 / V13 / V14 全部完成**; **GPU slot 全空闲**
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 16 是 A3 substrate 解读. **Round 17 是 V13/V14 substrate 解读 + 项目战略决策**:

**本轮审**:
- V14 d_pure noise floor 的实际宽度 + 它对 V18 deltas 显著性的判定
- V13 真 image_aux 单变量贡献 + 它对历史 confound 的清算
- V8 vs V13 几乎相等 (~36.49 dB) 意味着什么 (Grönwall step_weights 单独效应是否在噪声内)
- 在 V18 信号被确认为 "real but tiny" 后, 项目应该走哪条路: continue transport / pivot architecture / paper-as-is / 终止
- paper 叙事是否应该把 image_aux 升为主结果, V18 降为 ablation

**本轮不审**:
- A3 结果解释 (Round 16 已签)
- V13/V14 实验配置 / 执行质量 (本轮接受 Codex 提交的 canonical full-val 数字)
- KL pullback 设计本身 (Round 16 已签 KL direct-decoder narrative 死)

---

## 1. V13/V14 事实锚点 (已 merge, raw JSON 独立 verify)

来源:
- `review/0516/V13_V14_full_eval_analysis_20260521.md`
- `review/0516/full_eval_json/v13_true_image_aux_off_best_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v13_true_image_aux_off_last_fullval_psnr_chain_mse.json`
- `review/0516/full_eval_json/v14_true_d_pure_last_fullval_psnr_chain_mse.json`

### 1.1 V13 / V14 实验定义

- **V13 = true_image_aux_ablation**: 在 V7 配置基础上, 仅关闭 image auxiliary supervision (`lambda_img = 0`), 其它一切 (Grönwall step_weights, image_aux 之外的所有 V7 设置, seed=42) 保持不变. 这是 V7→V8 confound 的**真正**单变量替代品.
- **V14 = true_d_pure**: 在 V7 配置基础上, 仅改 seed=42 → seed=1337, 其它一切保持不变. 这是 d_pure / 单种子噪声地板的直接测度.
- 两个 run 都 from-scratch 训到 step 160000.
- `best.pt == last.pt` (step 160000 同时是 best, by `val_multi_objective` selector).

### 1.2 canonical full-val PSNR_clip3 (n=7403)

来自 `eval_first_hop_fullval_psnr_chain_mse.py`, 与 V7/V18 同一评测脚本.

| run | step | D20 | D10 | D4 | NORMAL | NORMAL MSE |
|---|---:|---:|---:|---:|---:|---:|
| V13 best=last | 160000 | 35.2172 | 35.5971 | 36.1122 | **36.4943** | 0.0002606 |
| V14 best=last | 160000 | 35.4249 | 35.8101 | 36.3677 | **36.7806** | 0.0002455 |

### 1.3 与历史基线放在同一坐标 (canonical NORMAL PSNR_clip3)

| run | NORMAL | vs V7.best (Δ dB) |
|---|---:|---:|
| V6_NOISE.best (seed=1337) | 36.7437 | −0.0373 |
| **V8.best (旧 image_aux off + Grönwall step_weights)** | **36.4729** | **−0.3081** |
| **V13.best (true image_aux off, V7 config + seed=42)** | **36.4943** | **−0.2867** |
| V7.best (seed=42) | 36.7810 | 0 |
| **V14.best (V7 + seed=1337)** | **36.7806** | **−0.0004** |
| V18.best @165K (LoRA + KL) | 36.8112 | +0.0302 |
| V18.last @200K (LoRA + KL) | 36.8426 | +0.0617 |
| A3 V18-cap @170K (matched-step, KL off) | ≈ V18@170K | (Round 16 已签) |

### 1.4 三条立即派生的事实

**事实 X1 — d_pure 噪声地板**: V14 − V7 = −0.0004 dB ≈ 0. 单种子 perturbation 在 canonical NORMAL PSNR_clip3 上的扰动 < 0.001 dB.

**事实 X2 — 真 image_aux 单变量贡献**: V14 − V13 = +0.2863 dB ≈ V7 − V13 = +0.2867 dB. 这是项目目前**单一最大的设计决策收益**, 比 V18 整套 decoder LoRA + KL 工程加起来 (+0.06 dB) 大约 5×.

**事实 X3 — Grönwall step_weights 单独效应在噪声内**: V13 = 36.4943 vs V8 = 36.4729, 相差 0.0214 dB ≈ V14 噪声地板的 ~20×, 但绝对量级仍很小. 由于 V8 与 V13 还差一个 train config (V8 用旧 Plan F train config, V13 用 V7 train config), 严格说这个 0.02 dB 不是纯 Grönwall 单变量效应. 但它已足够弱到可以说: **Grönwall step_weights 不是 V7 主线收益的来源, image_aux 才是**.

---

## 2. V18 deltas 的显著性 (SNR analysis)

把 V14 当作 single-seed perturbation noise estimator (|Δ| < 0.001 dB):

| comparison | Δ NORMAL (dB) | SNR (Δ / noise) | 解读 |
|---|---:|---:|---|
| V18.best − V7.best | +0.0302 | ~75× | 真信号, 但量级极小 |
| V18.last − V7.best | +0.0617 | ~150× | 真信号, 但仍 < d_pure → V13 image_aux 量级的 1/5 |
| V18.last − V18.best | +0.0314 | ~75× | 35K 额外训练带来真增长 |
| V7.best − V6_NOISE.best | −0.0373 | ~90× | 真信号 (注: V7 用 seed=42, V6_NOISE 用 seed=1337, 所以这个 delta 与 d_pure 同号且同量级, **可能 ≈ 纯种子效应**) |

**警告**: V14 只测了 V7 配置在 seed=42 → 1337 的一次扰动. 这是**一次估计**, 不是分布. 真正的 d_pure 噪声地板可能更宽 (e.g. 多 seed 跑会显示 std ~0.005-0.02 dB). 把 V14 单次 |Δ|=0.0004 dB 当作 "noise floor 上限" 是合理的, 当作 "noise floor 标准差" 不合理.

---

## 3. 战略选项 (5 选项 + 1 hybrid)

### 3.1 选项 A — Transport-side intervention

新设计一个针对 z_pred / rollout / chain path 的实验. 候选:

- A1: backbone scale-up (DiT base → large)
- A2: backbone architecture change (DiT → flow matching / consistency)
- A3: multi-step refinement (rollout 步数增加)
- A4: image_aux schedule tuning (V14 已确认 image_aux 关键, 但可能 schedule 没饱和)
- A5: 数据增强 / 额外 PET split

预期收益: 未知; 历史经验 +0.03-0.10 dB 一个轮次.
代价: 7-14 天 GPU + design + 一轮 review.

### 3.2 选项 B — Paper as-is (诚实负/边缘结果)

接受 V6_NOISE → V18.last = +0.099 dB 作为项目总优化量, 写 paper:

- 主结果: image_aux 是 +0.29 dB 关键设计 (V13 confirmed)
- 次结果: decoder LoRA capacity 在 GT-manifold direct decode 上 +0.17 dB, 链路只传 18% (A3 confirmed)
- KL pullback 设计无 measurable 效应 (A3 negative control)
- 项目 narrative: "PET latent transport feasibility study with structural decoder/transport decoupling analysis"

预期收益: 直接进入写作期, 1-2 个月 paper 可投.
代价: 接受总优化量边缘, 不再追加实验.

### 3.3 选项 C — Pivot architecture

跳出 latent transport + frozen decoder 框架, 改 diffusion / score-based / end-to-end pixel / autoencoder co-train.

预期收益: 可能 +1-3 dB, 也可能 -1 dB.
代价: 4-8 周设计 + 训练, 与现有 V7 系列基本无关, 等于新项目.

### 3.4 选项 D — Pivot data

获取额外 PET split / 增强 train set / 增加 modality. 现有数据是否真的不够仍未量化.

预期收益: 无 prior estimate.
代价: 数据获取 + 重训, 2-4 周.

### 3.5 选项 E — 终止 / paper as feasibility study

接受当前 V6→V18 +0.099 dB 是项目天花板, paper 写为 feasibility / negative result, 之后转新方向.

预期收益: 时间节约最大.
代价: 不进一步推进任何已识别但未尝试的方向.

### 3.6 Hybrid B+A4 (低成本试探)

先按选项 B 起 paper draft, 同时**并行**跑 1 个最便宜的 transport 探针 (A4 image_aux schedule tuning, 复用 V7 框架, ~7 天). 如果 A4 出 +0.05 dB 以上, paper 改写; 否则按 B 投.

---

## 4. 给 reviewer 的 7 个问题

### Q1 — V14 是否真的足以做 d_pure noise floor estimator?

V14 是 V7 + seed=1337 的**单次** rerun. 请评估:

- 单次 seed perturbation 是否足以判定 V18 deltas (+0.03 / +0.06 dB) 显著? 还是需要 ≥3 个 seed 才能给出 std?
- 如果只看 V14, 我们能说的最严格命题是什么? ("V18 信号超过单次 seed 扰动" vs "V18 信号统计显著")
- 是否需要再跑 V14b (V7 + seed=2024) / V14c (V7 + seed=7) 来加宽 noise estimator? 还是已经够了?

### Q2 — V13 +0.29 dB image_aux 单变量是否真清算了 V7-V8 confound?

V13 与 V8 相差 ~0.02 dB. V13 与 V14 相差 +0.29 dB. 请评估:

- 这是否足以宣判: "V7-V8 -0.31 dB 几乎全部由 image_aux 关闭引起, Grönwall step_weights 单独效应在噪声内"?
- V13 与 V8 之间还有 train config 差异 (V8 用 Plan F train config, V13 用 V7 train config), 这个 ~0.02 dB 差异是否需要进一步分离? 还是可以 close 这个 confound?
- 是否应该立即更新 V18_design_rationale / Plan F 历史叙事, 把 "Grönwall step_weights 是 transport 设计胜利" 这个旧说法撤掉?

### Q3 — V18 信号 +0.06 dB 在 V14 噪声地板下意味着什么?

V18.last − V7.best = +0.0617 dB ≈ 150× V14 单次 noise. 请评估:

- 这是 "真 effect, 量级小" 还是 "可能仍在多 seed std 内, 需要更多 seed"?
- 在 image_aux +0.29 dB 的参照系下, V18 的 +0.06 dB 是否值得写为 paper 主结果? 还是降为 ablation?
- 如果 V18 信号是真的, 它的机制现在最合理解释是什么? (decoder LoRA capacity + 35K 额外 from-V7-best 训练 = 35K 普通 fine-tune 的等效效果?)

### Q4 — 5 个战略选项中, 你主推哪个?

按 §3 列出 (A transport / B paper-as-is / C arch pivot / D data pivot / E terminate / Hybrid B+A4). 请明确:

- 主推 + 理由
- 拒绝 + 理由
- 是否同意 Hybrid B+A4 是一个合理低风险路径?
- 是否有 §3 没列的第 6 选项 (e.g. F: 重做 V18 但补全 KL 设计的修正版 V19)?

### Q5 — 论文叙事应该如何重写?

当前已知 publish-able 内容:

- image_aux +0.29 dB (V13 disambig)
- decoder LoRA capacity +0.17 dB direct decode, 18% 链路传导 (A3 disambig)
- KL pullback 无 measurable 效应 (A3 negative control)
- 项目总优化量 V6→V18 +0.099 dB (1 年)
- d_pure 单次 seed 噪声 < 0.001 dB (V14)
- transport chain MSE 92% 主导, decoder ceiling 8% (Round 7/13 attackable-gap 分析)

请评估哪种叙事最 publishable 且诚实:

- narrative B' (image_aux 主结果): "Image auxiliary supervision is the dominant design choice in PET latent transport (+0.29 dB); decoder/transport bottlenecks are structurally decoupled"
- narrative C' (decoupling 主结果): "Decoder capacity and transport quality are structurally decoupled in latent flow models for PET"
- narrative D' (negative result): "PET latent transport with frozen decoder hits a ~+0.1 dB ceiling; image_aux dominates while KL/decoder LoRA do not transfer through chain"
- narrative E' (feasibility study): "First systematic ablation of design choices in PET low-dose latent transport"

- 哪个最适合 MICCAI / ISBI / IPMI / Med Image Anal / TMI?
- 是否需要任何额外实验 (e.g. V14b/V14c 多 seed, A4 image_aux schedule) 才能 strengthen?

### Q6 — 是否漏关键控制 (post-V13/V14)?

V13/V14 完成后, 哪些控制现在变成 "如果跑会 strengthen paper" 的 high-EV 候选:

- V14b / V14c (多 seed Variance estimator)?
- V18 + seed=1337 (V18-seed-control, 检验 V18 +0.06 是否 seed-robust)?
- V13 + decoder LoRA (image_aux off + capacity, 检验 capacity 是否在 image_aux 缺失时反向变重要)?
- V19 = V18 但 use_pred_latent=false (corrected KL 重启, 终结 KL 故事)?

请给优先级排序 + 是否值得做.

### Q7 — Round 17 prompt 是否引入新偏差 (B61+)?

请特别检查:

- **过度精确的 SNR 偏差**: 把 V14 单次 |Δ|=0.0004 当 noise std, 而不是 upper bound
- **image_aux 过度收益归因偏差**: V13 与 V8 还有 train config 差异, 0.29 dB 是否真的全是 image_aux
- **paper narrative anchor 偏差**: §5 列了 4 个 narrative 候选, 是否漏更优 narrative
- **过早战略锁定偏差**: §3 列了 5 个选项, 是否锁定过早 (例如还有更微调的 B+任何具体 X hybrid)

---

## 5. 资料目录

### 5.1 本轮主审 substrate

- [V13_V14_full_eval_analysis_20260521.md](review/0516/V13_V14_full_eval_analysis_20260521.md)
- [v13_best JSON](review/0516/full_eval_json/v13_true_image_aux_off_best_fullval_psnr_chain_mse.json)
- [v14_best JSON](review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json)
- [V13 train log](review/0516/V13_true_image_aux_ablation/V13_train_20260518_183547.log)
- [V14 train log](review/0516/V14_true_d_pure/V14_train_20260518_183547.log)

### 5.2 上游已签结论 (本轮不重审)

- [REVIEW_INTEGRATION_round16_20260519.md](review/0517/REVIEW_INTEGRATION_round16_20260519.md) (A3 + KL direct-decoder narrative 死)
- [V18_FINAL_RESULTS_20260518.md](review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md)
- [V18_CAPACITY_ONLY_A3_REPORT_20260519.md](review/0517/V18_capacity_only/V18_CAPACITY_ONLY_A3_REPORT_20260519.md)
- [PLANF_FINAL_ANALYSIS_20260516.md](review/0516/PLANF_FINAL_ANALYSIS_20260516.md) (V6/V7/V8 历史 PSNR)

---

## 6. 输出格式

每位 reviewer 独立产出 markdown:

### 6.1 7 个问题逐条 (APPROVE / MODIFY / REJECT)

### 6.2 战略主 verdict
明确选 1 个 + 1 个备选:
- A transport intervention
- B paper-as-is
- C architecture pivot
- D data pivot
- E terminate / feasibility paper
- Hybrid B+A4
- 自定义 F (说明)

### 6.3 V18 信号判定
- "real and worth headlining"
- "real but ablation-only"
- "real but should be reframed as decoupling diagnosis"
- "borderline, need V18-seed control"

### 6.4 立即可执行 (now without GPU)
- design_rationale / paper draft / claim ledger 哪些立即更新
- 哪些动作暂停等 reviewer 共识

### 6.5 新偏差 (B61+)
若发现 prompt 引入 anchor / overreach / cherry-pick

---

## 7. 约束与提醒

- **本轮不重审** A3 / V13/V14 yaml 或 eval 协议
- **不假设** 你能跑代码 — 可读 markdown / JSON / log
- **特别关注**:
  - V14 单次 seed 是否 noise floor estimator 还是 upper bound (Q1)
  - image_aux +0.29 vs V18 +0.06 量级反差, paper narrative 是否应重排 (Q3+Q5)
  - 5 战略选项哪个 EV 最高 (Q4)
- **优先质疑**:
  - "V18 +0.06 dB 显著" 是否过度依赖单次 V14 seed perturbation
  - V13 与 V8 间 0.02 dB 差异是否真可忽略
  - paper narrative 是否陷入 sunk cost (V18 一定要写)

---

## 8. 本轮元说明

| 轮次 | 范围 | substrate |
|---|---|---|
| 12 | V18 final PSNR substrate | canonical eval |
| 13 | strategy re-evaluation (KL drift inverse) | KL drift + 历史 PSNR |
| 14 | A3 execution prep | yaml + task md |
| 15 | 3-slot launch decision | slot allocation |
| 16 | A3 结果解释 | A3 report + matched-step probe |
| **17** | **V13/V14 + 战略决策** | **V13/V14 canonical eval + V18/A3 已签结论** |

预期 Round 17 outcome:

- 最佳: 3 reviewer 对战略选项达成 ≥ 2/3 共识, 同时 V18 信号判定清晰
- 中等: 共识为 Hybrid B+A4 (低成本试探 + 写作并行)
- 最差: reviewer 发现 V14 单次 seed 不足以判 V18 信号显著, 必须先跑 V14b/V14c

---

## 9. 给 reviewer 的角色提示

你不是在审执行细节. 你在帮 user + claude 做一个 **战略决策**: 用 1 年时间换 +0.099 dB 总优化量后, 项目应该继续 (哪个方向) 还是 paper-and-pivot.

如果你认为证据已经足够支持某个清晰选择, 请直接给; 不要因为 "稳妥" 而推荐再多跑 2 个实验.
如果你认为证据真不够, 请明确说 "需要 X 实验才能决策", 并给出 X 的最小可行版本.
