# Peer Review Round 5 — V18 + ?? + ?? GPU slot allocation

- date: 2026-05-17 深夜
- branch: foc_lite_hop0
- HEAD when written: d59b9cf
- context: 用户硬约束 = **同时最多 3 个 task**。Slot 1 已确定 = V18 (rank=32 + KL fix)，正在跑。本轮 review 只评估 **slot 2 + slot 3 的最优选择**。
- target reviewers: 2-3 个 AI 独立评审，**不互见草稿**
- 数据/checkpoint 全在服务器，reviewer 凭 git artifacts 评估

> 前 4 轮 review 的整合 + 7 次 confirmation bias 记录见 [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) 与 [AUDIT_LORA_PARAM_COUNT_20260517.md](./V18_decoder_lora/AUDIT_LORA_PARAM_COUNT_20260517.md)。本轮**不重审 V18 设计**，三轮 review + codex audit 已经穷尽。

---

## 0. Round 5 与前 4 轮的根本不同

**前 4 轮**审 "应该跑什么实验"。  
**Round 5** 是首次审 **"3 个并行 GPU slot 的最优分配"** — 一个资源约束决策，不是设计决策。

V18 占 Slot 1（既成事实）。Slot 2/3 的候选有 5 个 + null：

| 候选 | 类型 | 训练时间 | 已有 yaml | 期望产出 |
|---|---|---|---|---|
| V14 | true d_pure (V7 + seed 1337) | 7 天 | ✅ | 真 d_pure 标尺，影响所有未来 SNR 判定 |
| V9 | β_NORMAL=2.5 full training | 7 天 | ✅ | β-axis 是否能改 NORMAL（Round 4 偏负面） |
| V9a | β=2.5 重选 V7 中间 ckpt + 1 次 full-val | ≤1 GPU-hour | 命令在 runbook | 0 训练 verify V9 是否值得做 |
| V13 | true image_aux ablation (V7 + image_aux off) | 7 天 | ✅ | 第一次纯 image_aux ablation，影响 V7-V8 解读 |
| V21 | conv head on frozen V7 | 2-3 天 + 1 天开发 | ❌ 待写 | 替代 V18 的便宜 fallback 路径 |
| **null** | 不 launch slot 2/3，等 V18 结果 | 0 | — | 保守，等数据 |

claude 在上一轮回答推荐 **slot 2 = V14, slot 3 = V9a**，**slot 3 V9a 之后空闲等 V18**。本轮请审这个推荐是否最优。

---

## 1. 给 reviewer 的 6 个新问题

### Q1 — slot 2 是否 V14？

V14 = V7 + seed 1337。Round 4 reviewer 都说 V14 必须做（任何 V18 SNR 判定都需要真 d_pure）。**但 Round 4 没考虑 GPU 限制**。

请评估：
- V14 与 V18 完全独立，无干扰 ✓
- V14 7 天与 V18 5-6 天完成时间错位 1-2 天，**正好用作 V18 评估时的 d_pure 标尺** ✓
- **但** V14 自身的 EV 是 "更精确的 d_pure 估计"，**当前已用 V7-V6_NOISE = 0.026 dB**。如果 V14 出来 = 0.020-0.030 dB（与上界相近），V14 的边际信息几乎为 0。**最 likely 情况是 V14 几乎没改变结论**。
- 替代：把 V14 slot 给 V13 (true image_aux ablation) 是否更高 EV？V13 第一次能纯隔离 image_aux 贡献，回答 "V7-V8 的 +0.308 dB 中 image_aux 真正贡献多少"。

### Q2 — slot 3 是否 V9a + idle？

claude 推荐 V9a (1 GPU-hour) 然后空闲 (slot 3 等 V18 完成)。

请评估：
- V9a 的产出：要么 V9 selector 与 V7 选同一 ckpt（V9 死），要么不同（V9 可能值得 7 天）
- **如果 V9a 显示不同 step**，是否应该立刻 launch V9 在 slot 3？这样 slot 3 在 V18 完成时也已跑了 1-2 天，proportionally 更高利用率
- **如果 V9a 显示同 step**，slot 3 还是空着。是否应该用作 V13 或别的？

### Q3 — V13 是否被 Round 3/4 错误降级？

Round 3/4 整合说 V13 "image_aux 已饱和，期望 ΔPSNR < 0.05 dB，性价比差"。

请评估：
- 这个判断基于 [DIAGNOSTIC_FINDING_20260517.md M1](./DIAGNOSTIC_FINDING_20260517.md)：val_hop0_img_total 末期/早期 = 1.028（仅改 2.7%）。**这是 V7 自己的 image_aux loss trajectory，不是 V13 (image_aux off) 的预期表现**。
- V13 的真正诊断价值是验证 **V7 vs V8 的 +0.308 dB NORMAL Δ 中 image_aux 贡献多少**。如果 V13 显示 ΔPSNR(V7-V13) ≈ 0，整套 image_aux 路径就被证伪。这是 negative-result 的强诊断价值。
- 7 天 GPU 换一个能 close 整族 image_aux 问题的实验，**性价比是否被低估**？

### Q4 — V21 (conv head) 是否仍是合理候选？

Round 4 决策树说："若 gap_decomposition probe 显示 attackable_gap < 2 dB → launch V21 instead of V18"。**probe 实际显示 11.2 dB，所以 V18 被选**。但 V21 仍未被否决，只是 V18 优先。

请评估：
- V21 是 "freeze V7 整体 + 在 decode(z_pred^V7) 之上训 ConvDecoderHead"
- 与 V18 完全正交（V18 改 decoder Linear，V21 加 post-decode conv head）
- 2-3 天训练，比 V18/V14 短一半
- 若 V18 成功，V21 上 V18 输出可能进一步改善；若 V18 失败，V21 是独立的 fallback
- 但 V21 yaml 待写 + 需要 RAE conv_head wrapper module 集成（1 天开发）
- 当前是否值得花 1 天开发 + 占 slot 3 跑？

### Q5 — 是否所有候选都不如 "保守 null"？

3 GPU slot 满载有机会成本：
- 训练时 GPU 满载 → 监控/评估慢
- V18 关键 checkpoint (step 180K, 200K) 需要 eval，需要 GPU
- 如果 slot 2/3 跑训练，V18 评估只能等

请评估：
- 是否应该 slot 2 = V14（与 V18 互不抢评估资源），slot 3 = **永久留空**给 V18 评估？
- 或 slot 2 = V14, slot 3 = V9a (1 GPU-hour) → 空闲 → V18 评估？

### Q6 — 启发式与硬约束的冲突

Round 4 reviewer 共识："不发新 review prompt"。本轮 prompt 是否违反这条原则？

请评估：
- Round 4 的 "不发新 review" 针对 V18 设计本身，不针对 GPU slot 调度
- GPU 限制是用户后引入的硬约束，前 4 轮未考虑
- 但本轮 prompt 是否在 anchoring 偏差（已经 sunk cost 在 V18 设计，所以推荐 V14/V9a 而非 reconsider V18 整体）？

---

## 2. 资料目录

按读的顺序：

### 2.1 V18 当前状态（slot 1）

- [V18_EXECUTION_REPORT_20260517.md](./V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md) — V18 launch 报告
- [AUDIT_LORA_PARAM_COUNT_20260517.md](./V18_decoder_lora/AUDIT_LORA_PARAM_COUNT_20260517.md) — V18 实施验证
- [V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) — V18 配置
- 当前 step ≈ 180K（截至 prompt 写作时），距 max_steps=200K 还约 20K step

### 2.2 候选 yaml（已就绪）

- [V14_v7_seed1337.yaml](../0516/V14_true_d_pure/V14_v7_seed1337.yaml) — V14 (7 天, GPU)
- [V9_normal_emphasis.yaml](../0516/V9_normal_emphasis/V9_normal_emphasis.yaml) — V9 full train (7 天, GPU)
- [V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) — V13 (7 天, GPU)

### 2.3 V21 待开发

- 基础组件：[RAE/RAE/src/stage1/decoders/conv_head.py](../../../RAE/RAE/src/stage1/decoders/conv_head.py)（已存在）
- V21 yaml: 待写
- V21 wrapper: 需新文件 `pet_lr/decoder_conv_head.py` (~200 行)

### 2.4 Round 1-4 关键事实

- [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) — Round 4 整合
- [DIAGNOSTIC_FINDING_20260517.md](./DIAGNOSTIC_FINDING_20260517.md) — image_aux 饱和的 M1/M2/M3 数据
- gap decomposition probe: [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) — attackable_gap = 11.2 dB
- V7/V8/V6_NOISE 多轴混淆事实：[REVIEW_INTEGRATION_20260516.md](../0516/REVIEW_INTEGRATION_20260516.md)

### 2.5 7 次 confirmation bias 记录

见 [/memories/repo/pet_latent_residual_eval.md](memories/repo/pet_latent_residual_eval.md)：B1-B7 形态。

---

## 3. Reviewer 输出格式

```
Q1 V14 as slot 2:
  decision: [yes / replace_with_V13 / replace_with_other]
  reasoning: <2-4 sentences>
  d_pure_value_when_V14_completes: <expected dB range>

Q2 V9a as slot 3 starter:
  decision: [yes_keep_idle_after / launch_V9_if_different_step / replace_entirely]
  reasoning: <2-4 sentences>

Q3 V13 mistakenly downgraded?
  verdict: [yes_should_promote / no_correct_downgrade / depends_on_X]
  V13_value_proposition: <one sentence>

Q4 V21 should-launch:
  verdict: [yes / no / wait_for_V18]
  cost_of_dev: <day estimate>

Q5 null option:
  verdict: [keep_slot_idle_for_V18_eval / fill_with_training / mix]

Q6 meta:
  is_round_5_legitimate: [yes / no_this_is_8th_bias]
  reasoning: <2-4 sentences>

Overall recommendation:
  slot_2: <V14 / V13 / V9 / V21 / null>
  slot_3: <V14 / V13 / V9 / V21 / V9a_then_null / V9a_then_V9>
  confidence: [low / medium / high]
  kill_switch: <what changes the recommendation>
```

---

## 4. Claude 自警的 4 处偏见

1. **Anchoring on V18**: 推荐 V14 + V9a 都是 "V18-supporting" 实验。如果 V18 是错方向，V14/V9a 也无用。是否应该 slot 2/3 用作 V18-orthogonal hedge (例如 V21)？

2. **EV 估算未量化**: V14 期望产出 "更精确的 d_pure"，但没说精确到什么 ΔdB 才有决策价值。如果 d_pure 上界已是 0.026 dB（V7-V6_NOISE），V14 误差区间多大才值 7 GPU-day？

3. **V9a 决策树过简**: claude 设计 "如果 V9a 同 step → V9 死" 但没考虑 V9a selector 改变可能在 NORMAL chain MSE 上几乎无差异（即使选不同 step）。需 v9a-decision-rule pre-register。

4. **V13 EV 估算可能错**: claude 用 M1 数据 (V7 自己 image_aux loss trajectory) 推 V13 期望 ΔPSNR，但这俩是不同实验。V13 真实价值是 "validate V7-V8 image_aux causal contribution"，可能值 7 天。

---

## 5. 请尽量对抗式批判

我倾向于 "被指出第 5 个 slot 选项 / 推 V14 → V13 / 推 V9a → V21 / null option 比训练好" 等任何挑战。**特别欢迎**：
- 指出 round 5 本身就是 8th bias
- 指出 V14/V9a 整体都不该跑，只该跑 V21
- 指出"3 GPU slot 不该满载"的硬约束理由
