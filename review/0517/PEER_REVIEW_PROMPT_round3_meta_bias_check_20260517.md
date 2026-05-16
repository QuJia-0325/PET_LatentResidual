# Peer Review Round 3 — V17 design + 5 P0 diagnostic plan

- date: 2026-05-17
- branch: foc_lite_hop0
- HEAD when written: efa272c (will bump after this commit)
- purpose: 第三轮独立 peer review。Round 2 的三份 review 整合后让 claude 撤回 V17 原设计，转而推荐 5 个 P0 离线诊断。**本轮要求 reviewer 对这次的整合 + 修正方向本身做对抗式评估**，避免把"修正"当作"修对了"。
- target reviewers: 任意 2-3 个 AI（codex / gemini / 其它），**完全独立运行**，不互相看草稿。
- **审稿人执行环境**：数据与 checkpoint 全在 GPU 服务器（`/home/qujiaxiang/project/PET_LatentResidual` & `/data_2/qujiaxiang/outputs/...`）。Reviewer 在本地只能读 git 已 commit 的文档、配置、代码、CSV/JSON 报告。无法亲自运行 V7/V8 forward。

> 这是项目第三轮 review。前两轮 review 的整合 + claude 自己的迭代见 §2 "已 commit 的资料目录"。**Reviewer 必须先读那些文件再做评审**，否则会重复前两轮已经解决的问题。

---

## 0. Round 3 区别于 Round 2 的关键不同

Round 2 是"判定 V17 是否值得做"。**Round 3 是"判定 claude 在 Round 2 整合后的修正本身有没有新的 confirmation bias"**。

Round 2 三位 reviewer 共识动作（agent1 / agent2 / agent3）：
1. V9a (β=2.5 重选 V7 ckpt，0 训练) — 三人独立推出
2. 撤回 V17 当前设计 (top-5% × L1 × α=4) — agent3 主导，agent1 部分同意
3. 5 个 P0 离线诊断（V9a + L1/L2 gradient norm + off-manifold single-step + NORMAL-mask ROI 重算 + decoder Jacobian probe）≤ 1 天完成

Claude 整合后准备执行这个 P0 计划。**Round 3 reviewer 要回答的不是"V17 该不该做"，而是"这次撤回 V17 + 推 P0 的决定本身是不是第四次 confirmation bias"。**

---

## 1. 给 reviewer 的 4 个新问题

### Q1：5 个 P0 实验是否够诊断当前争议？

Claude 列出的 P0:
- **P0-1** V9a：用 V7 中间 ckpt + β_NORMAL=2.5 selector 重选 best.pt，跑 full-val
- **P0-2** L1/L2/SUVmax gradient norm 离线对比（V7 ckpt，1 batch）
- **P0-3** Off-manifold single-step：V8 cascade midpoint z_pred 喂 V7 和 V8，比较 PSNR
- **P0-4** NORMAL-mask ROI 重算：用 clean x_NORMAL slice-aligned 计算 top-K mask，应用到 D20 重算 paired t
- **P0-5** Decoder Jacobian probe：在 z_D20 vs z_NORMAL 周围测 ‖∂x/∂z‖_F

**请评估**：
- 哪些 P0 是真正能否决 / 确认下一步训练设计的（高 leverage）？
- 哪些 P0 在 *任意结果* 下都不能 falsify Claude 的方向（低 leverage，浪费时间）？
- 缺少哪个关键诊断？（agent3 提了 decoder Jacobian，agent1 提了 latent-channel ROI loss 的可行性，有别的吗？）
- 5 个 P0 总时间 "≤ 1 天" 估计是否乐观？

### Q2：撤回 V17 是否过度？

Round 2 三位都说 V17 当前设计 (top-5% × L1 × α=4) 大概率无效，主要 3 个理由：
1. agent1: α=4 mass shift 仅 17.4%，SSIM/seam 抵消
2. agent2: D20 top-1% 负 Δ 可能是 mask artifact 不是 hop0 image_aux 无效
3. agent3: **L1 中位数偏置 + SUVmax underestimation 是真根因，ROI × L1 治不了**

Claude 接受了 3，撤回 V17。**请评估**：
- 这 3 个理由是否真的足够证伪 V17？或者只是说"V17 收益预期低"？
- 即便 V17 收益 < d_pure，跑 V17 是否仍有诊断价值（证伪 ROI-loss-shape 这条 hypothesis）？
- agent3 推的 V17' = SUVmax penalty + ROI-L2，比 V17 原设计的预期改善有多少？是否 *也* 治不了 L1 median bias？（L2 也有中位数收缩问题，只是更轻）
- 是否应该**保留** V17 原设计作为 negative control 反而能提供信息？

### Q3：claude 的 "P0 ≤ 1 天，先做 P0 再决定" 是否是用 review 推迟决定？

Round 2 prompt §10 已经自警："持续 review 而不执行是另一种 cherry-pick"。但 claude 在 Round 3 的整合里又抛出 5 个 P0 + 准备发 Round 3 prompt。

**请评估**：
- "先做 P0 再决定 V17/V11/V9" 是否合理推迟，还是 analysis paralysis？
- 如果 P0 全部跑完结果模糊（混合信号），claude 会再要 P1 吗？停止点在哪？
- 给一个 *绝对* 时间盒：从今天起 N 天内不论 P0 结果如何，必须 launch 至少一个 160K 训练。N 是多少？

### Q4：缺失的"baseline / null hypothesis"

整套 V11 / V11' / V17 / V17' / V15 / V9 / V13 / V14 / V16 大家谁也没说过的：**如果 V7 训练设置已经接近 image_aux 路径的局部最优，*什么都不动* 也是合理选项**。

**请评估**：
- 数据上有没有迹象表明 V7 已经到 image_aux 路径的边际收益饱和点？（agent1 说 frozen RAE decoder Jacobian 是结构瓶颈）
- 如果是，下一步应该 *跳出* image_aux 体系，去：
  - (a) 部分解冻 RAE decoder（agent1 提）
  - (b) 增加 backbone 容量（深度 / 宽度）
  - (c) 数据增强 / 多 dataset finetune
  - (d) 换 transport 架构（CCT 路径、Schrodinger bridge、direct probabilistic transport）
- 这些 *非 image_aux* 方向应不应该插到当前的 V11/V17 队列前面？

---

## 2. Reviewer 必读的已 commit 资料

按读的顺序：

### 2.1 Round 2 三份原始 review

- [PEER_REVIEW_PROMPT_round2_V17_design_20260517.md](./PEER_REVIEW_PROMPT_round2_V17_design_20260517.md) — Round 2 prompt（Claude 当时的推荐 + 自警）
- **Round 2 三位 reviewer 的回复**：这次不在 git 里，是用户直接粘贴给 Claude 的 chat 输入。Claude 已把整合写在下一份文件中（见 §2.3）。**如果你能拿到 Round 2 的原始 reviewer 文本，请优先读那个；否则读 §2.3 的整合。**

### 2.2 Round 1 已修复的事实

- [REVIEW_INTEGRATION_20260516.md](../0516/REVIEW_INTEGRATION_20260516.md) — Round 1 整合：V7/V8/V6_NOISE 不是单变量 ablation
- 已有 corrections in：[PLANF_FINAL_ANALYSIS_20260516.md](../0516/PLANF_FINAL_ANALYSIS_20260516.md)、[IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md](../0516/IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md)

### 2.3 Round 2 整合结论（claude 写的，需要被 review 的对象）

- *本 prompt 顶部 §0 即为整合后结论*
- 没有单独写 `REVIEW_INTEGRATION_v3` 文档，因为 reviewer 看 prompt 本身就足够；写 v3 反而是 analysis paralysis 的开始。

### 2.4 实验数据（在 git 里，reviewer 可直接读）

#### Disambiguation §0/§1/§2 报告

- [lipschitz/LIPSCHITZ_REPORT.md](./disambig/lipschitz/LIPSCHITZ_REPORT.md) — V7 alignment 0.57%, max L=1.009
- [per_hop_singlestep/SINGLESTEP_REPORT.md](./disambig/per_hop_singlestep/SINGLESTEP_REPORT.md) — channel A 主导（但 agent3 指出测试有结构盲点）
- [roi_psnr/ROI_REPORT.md](./disambig/roi_psnr/ROI_REPORT.md) — D20 top-1% Δ=-0.017 dB（可能是 mask artifact）
- 每份 report 旁边都有 per-slice CSV，可重算

#### Plan F 训练日志（160K step 完整）

- [review/0511/log_snapshots_20260516_163900/main_training/](../0511/log_snapshots_20260516_163900/main_training/) — V7/V8/V6_NOISE 完整训练 log
- [training_metrics_table_20260516_163900.csv](../0511/log_snapshots_20260516_163900/status/training_metrics_table_20260516_163900.csv) — 全 val 指标汇总

#### Full-val PSNR_clip3 评估

- [review/0511/fullval_psnr_clip3_20260516_173941/](../0511/fullval_psnr_clip3_20260516_173941/) — 6 个 checkpoint × 全 val 评估 + 每 slice CSV

### 2.5 当前代码（reviewer 需读，但不要在本地修改）

- [pet_lr/model_first_hop.py](../../pet_lr/model_first_hop.py) — `_apply_hop0_pixel_forcing`, `predict_latent_step`
- [pet_lr/losses_first_hop.py](../../pet_lr/losses_first_hop.py) — `compute_first_hop_image_loss` (image_aux loss 入口)
- [pet_lr/losses.py](../../pet_lr/losses.py) — `weighted_l1_loss`, `make_border_weight_map`, `ssim_loss`, `seam_consistency_loss`
- [pet_lr/rollout_first_hop.py](../../pet_lr/rollout_first_hop.py) — `rollout_multistep_losses_first_hop` (chain forward)
- [train_first_hop.py](../../train_first_hop.py) — `compute_hop0_image_losses` (line 786), total_loss 组合 (line ~2066)
- [tools/eval_per_hop_singlestep_clip3.py](../../tools/eval_per_hop_singlestep_clip3.py) — §1 测试代码（agent3 指出用 GT z_src 的问题）
- [tools/eval_roi_psnr.py](../../tools/eval_roi_psnr.py) — §2 测试代码（agent3 指出 D20-mask 用 noisy x_D20 计算 quantile 的问题）
- [tools/estimate_per_hop_lipschitz.py](../../tools/estimate_per_hop_lipschitz.py) — §0

### 2.6 待 launch 的 yaml drafts

- [review/0516/V9_normal_emphasis/V9_normal_emphasis.yaml](../0516/V9_normal_emphasis/V9_normal_emphasis.yaml) — V9 (β=2.5)
- [review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) — V13
- [review/0516/V14_true_d_pure/V14_v7_seed1337.yaml](../0516/V14_true_d_pure/V14_v7_seed1337.yaml) — V14
- V17 / V17' / V11 / V11' / V15 / V16 尚无 yaml

---

## 3. 给 reviewer 的输出格式

每位 reviewer 独立给：

```
Q1 P0 leverage assessment:
  P0-1 V9a: [high / medium / low]  why
  P0-2 gradient norm: [high / medium / low]  why
  P0-3 off-manifold single-step: [high / medium / low]  why
  P0-4 NORMAL-mask ROI: [high / medium / low]  why
  P0-5 decoder Jacobian: [high / medium / low]  why
  missing_diagnostic: <if any>
  total_time_realistic: <hours estimate, with reasoning>

Q2 V17 retraction verdict: [overdone / appropriate / underdone]
Q2 reasoning: <2-4 sentences>
Q2 V17' (SUVmax penalty + ROI-L2) verdict: <will it work better?>

Q3 P0 timeline verdict: [reasonable / analysis-paralysis / cant-tell]
Q3 hard timebox: <N days from today, after which must launch>

Q4 null hypothesis:
  V7-already-saturated likelihood: [high / medium / low]
  if-saturated next direction: <ranked list of (a)/(b)/(c)/(d) or new option>

Meta-bias check:
  is_claude_committing_4th_bias: [yes / no]
  if_yes: <what bias, specifically>

Overall recommendation:
  do_round_3_change: <single concrete change to the P0 plan>
  confidence: [low / medium / high]
  kill_switch: <what data invalidates this recommendation>
```

---

## 4. 给 reviewer 的对抗式说明

Round 1 / Round 2 claude 已经被指出 3 次 confirmation bias：
- Round 1 第 1 次：用 ablation delta 推未挖掘空间（V11 误推荐）
- Round 1 第 2 次：用 selector-biased ckpt 推 NORMAL 饱和（V11' 误推荐）
- Round 2 整合：用 single-checkpoint × on-manifold × global PSNR 当作世界本身性质（V17 误推荐）

**Round 3 的本质问题**：Claude 这次 "撤回 V17 + 推 P0" 是不是第 4 次同型错误？特别注意：
- 把 "我之前错了所以这次更小心" 当作 over-confidence 的理由
- 把 P0 数量多（5 个）当作 "更全面" 的伪保证，实际上是稀释 leverage
- 把 "等更多数据" 当作 "做更对的决定"

请尽量对抗式批判。我倾向于 "被指出第 4 种解读 / 第 5 个盲点" 胜过 "被告知 P0 计划合理"。

---

## 5. Claude 自己未消除的偏见（坦白）

1. **P0-1 V9a 的"0 GPU-hour"估计**：实际需要 V7 训练过程中保留的中间 ckpt。如果只有 best.pt + last.pt 两个，V9a 无法做 selector 重选（无中间样本可选）。Claude 没在 prompt 里 disclaim 这个先决条件。
2. **P0-3 off-manifold single-step**：claude 说"复用现有工具半天可做"，但实际需要先跑 V7/V8 cascade 收集 z_pred[k]，再用作 next-step input。代码量比"复用"大。
3. **5 个 P0 都是"诊断"** —— 没有任何一个 P0 *本身* 产生新的训练模型。如果 P0 全跑完结果是 "V7 已经很好别动了"，claude 会接受这个结果还是会继续找新 P1？没预先承诺。
4. **撤回 V17 之后的 fallback "V17' = SUVmax penalty + ROI-L2"** 是 agent3 提的，claude 没验证 V17' 的 implementation 复杂度（SSIM 是否要也加 SUVmax 项？SUVmax 是 per-image scalar，怎么写 backprop-friendly 的 loss？）
