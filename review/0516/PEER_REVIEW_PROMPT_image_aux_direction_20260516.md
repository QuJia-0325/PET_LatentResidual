# Peer Review Prompt — image_aux 扩展方向（20260516）

> 把这份 prompt（包含下面的"打包资料"部分）发给 Codex / Gemini / 任一外部 AI。要求它扮演 critical reviewer，专门检查我们 image_aux 扩展方向的推理是否有偏。

---

## 我们想要 reviewer 回答的三个具体问题

### Q1：cascade PSNR 单调上升的解释

我们的 V7 (image_aux ON) 在 full-val (n=7403) 的 chain PSNR_clip3 沿 D20→NORMAL **单调上升**：35.44 → 35.82 → 36.37 → 36.78 dB；chain MSE 单调下降：3.30e-4 → 2.96e-4 → 2.63e-4 → 2.46e-4。

我们的解释是："沿链 pair-level v_std 从 0.00963 (hop0) 衰减到 0.000141 (hop3)，下降 68×，比误差累积衰减更快；归一化后下游目标变得更容易；因此链下游 PSNR 反而更高。"

**请评估这个解释是否成立**。如果不成立，给出替代假设。如果成立，说明这个性质对 image_aux 扩展决策的含义。

### Q2：哪一跳是真正的瓶颈？

数据上 D20 是 PSNR 最低、MSE 最高的端点；NORMAL 是 PSNR 最高、MSE 最低。但 V7 vs V8 (image_aux on/off) 的 paired PSNR delta 在 NORMAL 最大（+0.308 dB）、D20 最小（+0.226 dB）。

我们最初的（错误的）推理是："NORMAL delta 最大 → NORMAL 有未挖掘的监督空间 → 应该在 NORMAL 加 image_aux。"

我们修正后的推理是："NORMAL delta 最大是 hop0 改善通过 chain coherence 间接传播的结果，不代表 NORMAL 本身有监督空间。D20 是真瓶颈，正确动作是强化 hop0 image_aux。"

**请判定哪个推理正确**。如果都不对，给第三种解读。

### Q3：在 D20 是 PSNR 最低端点的前提下，下面三个方向哪个边际收益最高？

| 方向 | 描述 | 代价 |
|---|---|---|
| (A) V11': 强化 hop0 image_aux | λ_img 0.04→0.08+0.12 sweep、`use_extended_seam: true`、高 SUV ROI 加权 | 0 代码或 ~50 行 |
| (B) V11: multi-hop image_aux | 把 image_aux 复制到所有 hop，权重用 Grönwall 闭式解 | 训练时间 +30-50% |
| (C) V9: β_NORMAL 1.5→2.5 | 提高 NORMAL 在 selector 中的权重 | 1 次完整训练 |

我们当前倾向 (A)。**请评估** (A)/(B)/(C) 的预期收益顺序，特别说明：
- (B) 的预期收益是否被 hop1/2/3 v_std 衰减 7-68× 抵消？
- (C) 是否在加权一个已经饱和的目标（NORMAL 已经是 PSNR 最高的阶段）？

---

## 我们已知的混淆变量 / 偏见来源

- 我（claude）刚承认在同一份数据上推出过相反结论，confirmation bias 风险高。
- d_pure (V7 vs V6_NOISE seed 噪声) 只有 0.026 dB / 0.37% MSE，所以小差距判定需要 paired t-test 而不是 mean 直接比较。
- best.pt 选择 metric 受 selector β 影响；V9 的"改善"可能来自更宽容的选择标准而非真模型变好。

---

## 我们希望 reviewer 输出的格式

```
Q1 verdict: [accepted / partially-accepted / rejected]
Q1 reasoning: <2-4 sentences>
Q1 alternative-hypothesis: <if any>

Q2 verdict: [original / corrected / neither]
Q2 reasoning: <2-4 sentences>

Q3 ranking: <e.g. A > C > B>
Q3 reasoning: <2-5 sentences>
Q3 risk-flag: <what could make this ranking wrong>

Overall recommendation:
  - next-experiment: <single concrete next run>
  - confidence: [low / medium / high]
  - what-to-measure-first: <metric or test that should run before launch>
```

---

## 打包资料（粘贴给 reviewer）

### 资料 1：V7/V8/V6_NOISE full-val PSNR_clip3 (n=7403, paired)

来自 [review/0511/fullval_psnr_clip3_20260516_173941/status/planf_fullval_psnr_clip3_summary_20260516_173941.csv](../0511/fullval_psnr_clip3_20260516_173941/status/planf_fullval_psnr_clip3_summary_20260516_173941.csv)。

| tag | exp | ckpt | step | D20 | D10 | D4 | NORMAL |
|---|---|---|---:|---:|---:|---:|---:|
| planf_v7_last | V7 (image_aux ON) | last | 160000 | 35.4354 | 35.8194 | 36.3736 | 36.7810 |
| planf_v8_last | V8 (image_aux OFF) | last | 160000 | 35.2094 | 35.5897 | 36.0963 | 36.4729 |
| planf_v6noise_last | V6 seed1337 (image_aux ON) | last | 160000 | 35.4217 | 35.8110 | 36.3596 | 36.7550 |

NORMAL chain MSE: V7=0.000246, V8=0.000262, V6_NOISE=0.000247。

### 资料 2：paired stats (V7 - V8, V7 - V6_NOISE)

| stage | Δ(V7-V8) PSNR dB | t | win % | Δ(V7-V6_NOISE) PSNR dB | t | win % |
|---|---:|---:|---:|---:|---:|---:|
| D20 | +0.226 | 43.1 | 85.8 | +0.014 | 6.6 | 54.6 |
| D10 | +0.230 | 72.4 | 90.0 | +0.008 | 3.6 | 53.5 |
| D4 | +0.277 | 79.4 | 91.8 | +0.014 | 4.9 | 54.0 |
| NORMAL | +0.308 | 74.6 | 91.5 | +0.026 | 7.8 | 55.8 |

### 资料 3：pair-level v_std（chain dynamics 物理常数）

| pair | v_std | 相对 hop0 |
|---|---:|---:|
| D50→D20 | 0.009634 | 1× |
| D20→D10 | 0.002946 | 1/3.3 |
| D10→D4 | 0.000775 | 1/12 |
| D4→NORMAL | 0.000141 | 1/68 |

### 资料 4：架构约束

- `_apply_hop0_pixel_forcing` 在 `hop_idx==0` 时把 D50 真实图像注入 z_D50 (pet_lr/model_first_hop.py:444-476)。**只能 hop0**，其他跳没有同源真实图像。
- `compute_hop0_image_losses` 在 hop0_batch 上 decode(z_pred) → 与 GT x_dst 比较 (train_first_hop.py:786)。**理论上可扩展**到其他 hop。
- 数据集已支持 `include_full_x_rollout`（V7/V8 val 已开），训练侧未开。
- image_aux 当前权重：`lambda_max: 0.04`、`l1_weight: 1.0`、`ssim_weight: 0.25`、`seam_weight: 0.10`、`use_extended_seam: false`。

### 资料 5：之前推导过的相关闭式解

[review/0516/STEP_WEIGHTS_THEORY_REFERENCE.md](./STEP_WEIGHTS_THEORY_REFERENCE.md) §2.2:

```
w_j^{*, multi-hop} ∝ Σ_{k=j+1}^{K} β_k · Π_{i=j+1}^{k-1} L_i²
```

V7 step_weights 已是该闭式解在 β=[0.5, 0.45, 0.9, 1.5] + L_i≈1 下的实例化。L_i≈1 是反解结果，未实测；[tools/estimate_per_hop_lipschitz.py](../../tools/estimate_per_hop_lipschitz.py) 是 10 分钟可跑的脚本骨架。

### 资料 6：历史 review

[review/0430/reviewer/literature_architecture_image_aux_review_20260501.md](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) §6 已明确指出当前 image_aux "hop0-only, low-weight, low-level"，建议加 ROI-PSNR 评估、SUVmax 指标、LPIPS。

### 资料 7：我们刚撤回的错误推理

[review/0516/IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md §10](./IMAGE_AUX_ARCHITECTURE_ANALYSIS_20260516.md) 完整记录了我们如何从"NORMAL delta 最大 → 推荐 V11 multi-hop"，到看到绝对 PSNR 表（NORMAL 是最高不是最低）后撤回为 V11' (强化 hop0)。请 reviewer 也独立判定这个撤回是否过度修正。

---

## 期望的反馈强度

请尽可能**对抗式**地批判。不必维护我们的面子。我们偏好"被指出第三种解释"胜过"被告知你们想对了"。
