# Peer Review Round 2 — V17 ROI-weighted hop0 image_aux

- date: 2026-05-17
- branch: foc_lite_hop0
- HEAD when prompt written: 9ce6ecf
- purpose: 对 disambiguation §0/§1/§2 完成后我（claude）给出的下一步推荐（V17 = hop0 image_aux + top-5% SUV ROI 加权 优先于 V9/V13/V11/V15）进行**第二轮外部 peer review**。
- target reviewers: codex (implementation feasibility) + gemini (statistical / experimental design / cherry-pick risk) — 两份独立 review，**不要互相看对方的草稿**。

> 这是项目第二轮 peer review。第一轮（[REVIEW_INTEGRATION_20260516.md](./REVIEW_INTEGRATION_20260516.md)）的整合判定催生了三个零训练 disambiguation 实验，结果在 [review/0517/disambig/](../0517/disambig/) —— **请先读那三份 report 再做评审**。

---

## 给两位 reviewer 的 4 个具体问题

### Q1：单步 PSNR 解耦的强度

§1 [SINGLESTEP_REPORT.md](../0517/disambig/per_hop_singlestep/SINGLESTEP_REPORT.md) 显示 hop1/2/3 的单步 V7−V8 ΔPSNR 几乎为 0：
- D20→D10: +0.021 dB (t=5.29 仍显著但 mean 极小)
- D10→D4: −0.000 dB (t=-0.17)
- D4→NORMAL: −0.019 dB (t=-3.41)

我据此判定 "channel B (shared backbone) 不存在，channel A (chain coherence) 主导"，进而撤回 multi-hop image_aux (V11/V15)。

**请评估**：
- 这个解耦在已收敛 checkpoint 上测一次能否推广？训练动力学里 channel B 可能曾经存在但被 fine-tuning 抹平 — 我没考虑这点。
- D4→NORMAL 的 −0.019 dB 是 t=-3.41 显著的，能否反向说 *hop3 上 image_aux 反而有害*？还是 d_pure (V14 没做) 的 noise floor？
- 在 t=5.29 处的 D20→D10 微弱差异 +0.021 dB 是不是 channel B 的残留信号被 stat 检出但被我忽略？

### Q2：ROI 加权 vs λ-sweep 的优先级

§2 [ROI_REPORT.md](../0517/disambig/roi_psnr/ROI_REPORT.md) 显示 V7-V8 在 D20 top-1% SUV 上 Δ = −0.017 dB。我据此说"hop0 image_aux 对最难像素无效 → V11c (ROI 加权) 优于 V11' (λ sweep)"。

**请评估**：
- "hop0 λ_max 当前 0.04"，加倍到 0.08 是否真的不能改 top-1%？还是只是"现有 λ 下没改"？V7-V8 是 0 vs 0.04 的对比，不能外推到 0.04 vs 0.08。
- 如果同时跑 V17 (ROI) 和 V11' (λ sweep) 互为对照，是不是比单选 V17 更好？我的"V11' 信号有限"判定基于 V7-V8 单次对比，confounded by step_weights 差异。
- ROI mask 用 GT x_gt 的 quantile 算 → 每个 batch / 每个 slice 内 normalized。这是不是会让"高 SUV"在低 SUV 切片里也被放大（noise / 背景被当成 hot spot）？

### Q3：V14 (true d_pure) 是否还有必要做？

§2 用 V7-V6_NOISE 估算 d_pure 在大多数指标上 ΔPSNR < 0.04 dB，看起来很小。我之前推荐保留 V14，但 ROI 数据已经给出"V7-V6_NOISE 几乎没差"的事实证据。

**请评估**：
- V14 还有必要做吗？或者能否用现有 V6_NOISE-V6 比较推算（如果 V6 seed42 checkpoint 存在）？
- V14 的真正价值：当 V17 跑完后，V17-V7 必须用 d_pure 做 SNR 判定。这个时候 V14 才有用 — 但需要 ~7 天训练。能否用 V7-V6_NOISE 做 upper-bound 的近似 d_pure 先用着？

### Q4：V9 (β_NORMAL=2.5) 降级是否过度？

我推荐 V9 降级，理由："NORMAL 已经是 PSNR 最高的阶段（36.78 dB），加重已饱和目标"。但：
- Lipschitz gate 通过（§0 alignment 0.57%）
- §2 ROI 数据没有直接说 β_NORMAL=2.5 会失败
- "NORMAL PSNR 最高" 一部分原因是 best.pt selector 用 β_NORMAL=1.5 选的 ckpt — 改 β 会改 selector 行为

**请评估**：
- V9 降级是否过度保守？或者应该跟 V17 并行做（互为正交轴）？
- V9 的实际预测：增加 β_NORMAL 在已收敛模型上会主要影响 best.pt 选择，而不是模型本身 — 是不是不需要重训练，只需用新 β 重选 ckpt？

---

## 给 codex 的额外 implementation 问题

### C1：V17 ROI 加权的具体实现

我打算改 [pet_lr/losses_first_hop.py:`compute_first_hop_image_loss`](../../pet_lr/losses_first_hop.py)，在现有 `border_weight_map` 基础上叠加 SUV mask。伪代码：

```python
suv_thresh = torch.quantile(x_gt.flatten(2), 1 - 0.05, dim=2, keepdim=True).view(B, 1, 1, 1)
suv_mask = (x_gt >= suv_thresh).float()  # [B, 1, H, W]
roi_weight = 1.0 + (roi_weight_alpha - 1.0) * suv_mask  # alpha=4 → top-5% gets 4×
combined = border_map * roi_weight
loss_l1 = weighted_l1_loss(x_pred, x_gt, combined)
# SSIM / seam 暂不改
```

**问题**：
- `weighted_l1_loss` 现在的实现能直接吃 `combined` 而不溢出吗？需不需要 re-normalize 让 `mean(combined) = 1`？
- SSIM 是否也该加 ROI（SSIM 是 windowed，加 mask 在边界会有问题）？
- top-5% 切片内的 quantile 在低 SUV 切片（背景占多数）上是不是会把噪声拉成"伪 hot spot"？是不是该加一个 absolute SUV 下限（e.g. SUV ≥ 1.0 且 quantile）？

### C2：V14 实施

V14 的 yaml 已在 [V14_v7_seed1337.yaml](./V14_true_d_pure/V14_v7_seed1337.yaml) draft。`require_fresh_output_dir: true` 已设。需要确认：
- 服务器上 `review_0516_runs/V14_true_d_pure/run/` 不存在
- GPU 可用情况下 V14 是否能与 V17 并行（每个一张卡）

### C3：单步 PSNR 工具能否用到 V17 评估？

我希望 V17 跑完后**重复 §1 单步 PSNR 测试**：如果 V17 vs V7 的单步 Δ 在 hop0 上集中（且 ≥ V7-V8 hop0 Δ），证明 ROI 加权改进了 hop0；如果在多 hop 上分散，说明 ROI 加权产生 shared-backbone 副作用。

[tools/eval_per_hop_singlestep_clip3.py](../../tools/eval_per_hop_singlestep_clip3.py) 能直接复用吗？

---

## 给 gemini 的额外 statistical 问题

### G1：Confirmation bias 风险

我在第一轮 review 已经被指出过两次（推 V11、撤回 V11→V11'）confirmation bias。这次推 V17 + 降 V9 的论证里有没有第三次同型错误？

特别注意：
- §1 单步 PSNR 的"channel A 主导"叙事是不是过于干净？(在已收敛 ckpt 上的单点测量推广到训练动力学的隐含假设)
- §2 ROI 数据里 D4/NORMAL 的 top-1% Δ 其实是 +0.163 / +0.183 dB（不小）。我只挑 D20 的 -0.017 dB 说"hop0 image_aux 对最难像素无效"是不是 cherry pick？

### G2：V17 的成功阈值预注册

V17 是 1 次 160K 训练。成功阈值该怎么定？候选：
- Δ NORMAL PSNR_clip3 (V17 - V7) ≥ 3·d_pure（但 d_pure 在 V14 跑完前只有 V7-V6_NOISE 的 +0.026 dB 估计，可能 inflated 也可能 underestimated）
- Δ top-5% SUV PSNR (V17 - V7) ≥ 某个 dB 阈值
- Δ SUVmax error (V17 - V7) 收窄某个比例

应该怎么定？给具体数值阈值。

### G3：multiple comparison 与 selector bias

我现在跑了 V7/V8/V6_NOISE 三个 model × 4 个 hop × 7+ 个 ROI 指标 = 几十个 paired test。FDR / Bonferroni 校正后还有多少 "显著" 结果留下？

并且：ckpt 是按 β 加权 selector 选的。挑 metric (top-1% vs top-5% vs high-grad) 是 post-hoc。这两层 bias 加起来，§2 哪些数字能信？

---

## 期望的 reviewer 输出格式

每位 reviewer 单独给：

```
Q1 verdict: [accepted / partially-accepted / rejected]
Q1 reasoning: <2-4 sentences>
Q1 alternative-hypothesis: <if any>

(同 Q2/Q3/Q4)

(codex 额外答 C1/C2/C3 — 给 yes/no + 具体 issue)
(gemini 额外答 G1/G2/G3 — 给 explicit threshold / pass rate / bias quantification)

Overall recommendation:
  - next-experiment: <single concrete next run, with config sketch>
  - confidence: [low / medium / high]
  - kill-switch: <what data could falsify the recommendation in 1 week>
```

---

## 打包资料（粘贴给 reviewer）

### 资料 1：本轮 disambiguation 三份 report

- [LIPSCHITZ_REPORT.md](../0517/disambig/lipschitz/LIPSCHITZ_REPORT.md)
- [SINGLESTEP_REPORT.md](../0517/disambig/per_hop_singlestep/SINGLESTEP_REPORT.md)
- [ROI_REPORT.md](../0517/disambig/roi_psnr/ROI_REPORT.md)

### 资料 2：第一轮 review 整合（含背景与 confounder caveat）

- [REVIEW_INTEGRATION_20260516.md](./REVIEW_INTEGRATION_20260516.md)
- [STEP_WEIGHTS_THEORY_REFERENCE.md](./STEP_WEIGHTS_THEORY_REFERENCE.md)（multi-hop Grönwall 闭式解）
- [IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md](./IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md)（顶部有 CORRECTION）
- [PLANF_FINAL_ANALYSIS_20260516.md](./PLANF_FINAL_ANALYSIS_20260516.md)（顶部有 CORRECTION）

### 资料 3：相关代码

- 现有 image_aux loss：[pet_lr/losses_first_hop.py](../../pet_lr/losses_first_hop.py)
- hop0 pixel forcing：[pet_lr/model_first_hop.py:444-476](../../pet_lr/model_first_hop.py)
- single-step 评估：[tools/eval_per_hop_singlestep_clip3.py](../../tools/eval_per_hop_singlestep_clip3.py)
- ROI 评估：[tools/eval_roi_psnr.py](../../tools/eval_roi_psnr.py)
- Lipschitz 测量：[tools/estimate_per_hop_lipschitz.py](../../tools/estimate_per_hop_lipschitz.py)
- 训练循环：[train_first_hop.py:786-825 `compute_hop0_image_losses`](../../train_first_hop.py#L786)、[train_first_hop.py:2066-2075 `total_loss`](../../train_first_hop.py#L2066)

### 资料 4：项目历史相关结论

- [review/0430/reviewer/literature_architecture_image_aux_review_20260501.md §3](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)：2024+ PET 文献，**推荐 ROI-PSNR / SUVmax / SBR / CNR 而不仅是全图 PSNR**
- [review/plan/ARCHITECTURE_ANALYSIS_20260501.md §16/§18.3/§20.6](../plan/ARCHITECTURE_ANALYSIS_20260501.md)：Grönwall 闭式解 + σ-normalize ablation 优先级

### 资料 5：claude 这次的具体推荐

| 实验 | 我的优先级 | 我的理由 |
|---|---|---|
| **V17 = hop0 image_aux + top-5% SUV ROI 加权 (4×)** | **首发** | §2 直接证据 |
| **V14 = V7 + seed 1337** | 并行 | 需要 d_pure 给后续 SNR 判定 |
| V9 = β_NORMAL=2.5 | 降级 | NORMAL 已饱和；selector circular |
| V13 = V7 image_aux off | 降级 | §1 已隐式回答 |
| ~~V11/V15 = multi-hop image_aux~~ | **撤回** | §1 channel B 证伪 |

---

## 给 reviewer 的对抗式说明

我（claude）已经被第一轮 review 指出过两次 confirmation bias。这次推 V17 的论证里**至少**有以下未自动消除的偏见：

1. 我把 V17 的"信号密度高"当公理，没量化 V17 vs V11' 的边际信息。
2. 我把 §1 的 "channel A 主导" 当 strong evidence，但它只是 1 个 checkpoint × 1 次测量。
3. 我把 §2 的 D20 top-1% 负 Δ 当作 "hop0 image_aux 无效" 的证据，但同样的 ROI 在 D4/NORMAL 上是显著正 Δ。

请尽量对抗式批判。我倾向于"被指出第三种解读"胜过"被告知你的推荐对"。
