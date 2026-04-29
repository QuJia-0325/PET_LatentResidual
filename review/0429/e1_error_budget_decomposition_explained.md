# E1 误差预算分解：Transport vs Decoder Gap 计算详解（修订版）

**日期**：2026-04-29（修订）
**关联脚本**：[`scripts/diagnose_error_budget.py`](../../scripts/diagnose_error_budget.py)
**关联数据**：[`review/0422/exp/gt_latent_decoder_ceiling_clip3_val_20260422.md`](../0422/exp/gt_latent_decoder_ceiling_clip3_val_20260422.md)
**关联文档**：[`docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md`](../../docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md)

---

## 0. 关于本文档的数学严谨性说明

本文档的分解是 **PSNR 对数域诊断性分解**（PSNR-domain diagnostic decomposition），不是严格的像素误差能量（MSE-domain）线性贡献分解。

PSNR 定义为：

$$\text{PSNR} = 10 \log_{10} \frac{\text{MAX}^2}{\text{MSE}}$$

因此 PSNR 差值（dB gap）实质上是 MSE 比值的对数：

$$\text{PSNR}_A - \text{PSNR}_B = 10 \log_{10} \frac{\text{MSE}_B}{\text{MSE}_A}$$

这意味着 dB gap **不能被解读为线性误差贡献比例**。本文中的"诊断占比"是 dB 域的分配，用于工程诊断方向判断；如需严格的误差归因，应参考 §5 的 MSE 域分析。

---

## 1. 问题

当前系统（V3 best）的 E2E PSNR 为 36.87 dB（NORMAL 档），而 decoder 对 GT latent 的重建 ceiling 为 52.63 dB。总 gap ≈ 15.76 dB。

这个 gap 中多少与 **transport（速度场预测不准）** 相关，多少与 **decoder（解码器自身不完美）** 相关？

## 2. 测量方法

E1 脚本 `diagnose_error_budget.py` 对每个 val slice 计算 **三个 PSNR**：

### 2.1 三个测量

```python
# 测量 1: Decoder Ceiling
# GT latent → decoder → 和 GT 原始像素比较
# 回答：decoder 拿到完美 latent 能做到多好？
psnr_ceil = calc_psnr_clip3(decoder(z_GT), x_GT)

# 测量 2: E2E 实际
# transport 预测的 latent → decoder → 和 GT 原始像素比较
# 回答：整个系统端到端的实际表现？
psnr_e2e = calc_psnr_clip3(decoder(z_pred), x_GT)

# 测量 3: Pred vs GT Decode
# transport 预测的 latent → decoder → 和 decoder(GT latent) 比较
# 回答：decoder-mediated transport discrepancy 有多大？
psnr_pred_gt = calc_psnr_clip3(decoder(z_pred), decoder(z_GT))
```

### 2.2 第三个测量的精确含义

第三个测量 **不是** "纯 transport 误差"。更准确地说：

$$\text{PSNR}(\text{decoder}(z_{pred}), \text{decoder}(z_{GT}))$$

测量的是 **transport latent error 经过 decoder 映射后的 decoded-space discrepancy**：

$$\text{decoder}(z_{pred}) - \text{decoder}(z_{GT}) \approx J_D(z_{GT}) \cdot (z_{pred} - z_{GT})$$

其中 $J_D$ 是 decoder 的 Jacobian。因此这个量受到 decoder 对 latent perturbation 的局部敏感性影响，是 **decoder-mediated transport error**，不是 latent-space transport error 本身。

## 3. PSNR 域诊断分解

```
gap_total     = psnr_ceil    - psnr_e2e       （PSNR 总 gap）
gap_transport = psnr_ceil    - psnr_pred_gt   （transport 相关 PSNR gap）
gap_decoder   = psnr_pred_gt - psnr_e2e       （decoder reconstruction floor 相关 PSNR gap）
```

代数恒等式 $A - C = (A - B) + (B - C)$ 保证 `gap_transport + gap_decoder = gap_total`。

> **重要**：这是 PSNR 对数域的代数分解，不是 MSE 域的线性误差贡献分解。dB gap 的"占比"是诊断性指标，不应被解读为严格的误差能量归因。

### 3.1 直觉理解

```
                psnr_ceil (46.64)
                    │
     gap_transport  │  = 46.64 - 35.90 = 10.74 dB
     (decoder-      │  decoder(z_pred) vs decoder(z_GT) 的 PSNR 落差
      mediated      │  → latent 质量差异经 decoder 映射后的表现
      transport     │
      discrepancy)  │
                    ↓
              psnr_pred_gt (≈ 35.90)
                    │
     gap_decoder    │  = 35.90 - 35.49 = 0.41 dB
     (decoder       │  decoder(z_GT) ≠ x_GT 造成的参照偏移
      reconstruction│  → decoder 自身重建不完美
      floor)        │
                    ↓
              psnr_e2e (35.49)
```

### 3.2 gap_decoder 的物理含义

$$\text{gap\_decoder} = \text{PSNR}(\text{dec}(z_{pred}), \text{dec}(z_{GT})) - \text{PSNR}(\text{dec}(z_{pred}), x_{GT})$$

- 如果 decoder 完美：$\text{dec}(z_{GT}) = x_{GT}$，则 gap_decoder = 0
- 实际 decoder 有重建损失，所以 $\text{dec}(z_{GT})$ 比 $x_{GT}$ "更接近" $\text{dec}(z_{pred})$
- 这个 0.41 dB 量化了 decoder reconstruction floor 导致的参照偏移

## 4. D20 档 PSNR 域数值（7403 val slices）

| 测量 | 公式 | 值 |
|---|---|---:|
| Decoder Ceiling (psnr_ceil) | PSNR(dec(z_GT_D20), x_GT_D20) | **46.64 dB** |
| Pred vs GT Decode (psnr_pred_gt) | PSNR(dec(z_pred_D20), dec(z_GT_D20)) | **≈ 35.90 dB**（推导值） |
| E2E (psnr_e2e) | PSNR(dec(z_pred_D20), x_GT_D20) | **35.49 dB** |

| PSNR 域分解 | 公式 | 值 | PSNR-gap diagnostic ratio |
|---|---|---:|---:|
| gap_total | ceil - e2e | 11.15 dB | 100% |
| gap_transport | ceil - pred_gt | **10.74 dB** | **96.3%** |
| gap_decoder | pred_gt - e2e | **0.41 dB** | **3.7%** |

> 注：psnr_pred_gt ≈ 35.90 dB 是从 46.64 - 10.74 推导的，原始 E1 JSON 在远程服务器未 sync。

## 5. MSE 域分析（更严格的误差量化）

PSNR 域的 dB gap 本质是 MSE 比值的对数。以 D20 为例：

### 5.1 MSE 比值

$$\frac{\text{MSE}_{e2e}}{\text{MSE}_{ceil}} = 10^{\text{gap\_total}/10} = 10^{11.15/10} \approx 13.03$$

$$\frac{\text{MSE}_{pred\_gt}}{\text{MSE}_{ceil}} = 10^{\text{gap\_transport}/10} = 10^{10.74/10} \approx 11.86$$

### 5.2 物理解读

| MSE 比值 | 数值 | 含义 |
|---|---:|---|
| MSE_e2e / MSE_ceil | **13.03×** | E2E 像素误差是 decoder ceiling 误差的 13 倍 |
| MSE_pred_gt / MSE_ceil | **11.86×** | decoder-mediated transport discrepancy 是 decoder ceiling 误差的 12 倍 |
| MSE_e2e / MSE_pred_gt | **1.10×** | 参照从 dec(z_GT) 换到 x_GT 后误差仅增大 10% |

### 5.3 全部 4 跳 MSE 比值

| 跳 | gap_total (dB) | MSE_e2e/MSE_ceil | gap_transport (dB) | MSE_pred_gt/MSE_ceil | gap_decoder (dB) | MSE_e2e/MSE_pred_gt |
|---|---:|---:|---:|---:|---:|---:|
| D50→D20 | 11.15 | **13.03×** | 10.74 | 11.86× | 0.41 | 1.10× |
| →D10 | 12.88 | **19.38×** | 12.33 | 17.10× | 0.55 | 1.13× |
| →D4 | 14.41 | **27.61×** | 13.98 | 25.01× | 0.43 | 1.10× |
| →NORMAL | 15.89 | **38.80×** | 15.70 | 37.15× | 0.19 | 1.04× |

**关键观察**：
- E2E MSE 是 decoder ceiling MSE 的 **13-39 倍**——transport 引入的像素误差远大于 decoder 自身误差
- 参照偏移（MSE_e2e/MSE_pred_gt）仅 1.04-1.13×——decoder reconstruction floor 对总误差的边际贡献很小
- MSE 比值随级联跳数指数增长——exposure bias 的 MSE 累积效应

### 5.4 与 PSNR 域诊断占比的关系

PSNR 域的 96.3% "diagnostic ratio" 对应 MSE 域中 MSE_pred_gt/MSE_ceil = 11.86×（decoder-mediated transport error 是 decoder ceiling error 的近 12 倍）。

两种表述方式传达同一个信息：**transport 相关的误差在量级上远超 decoder 自身误差**。但 MSE 比值是更严格的定量陈述。

## 6. Latent-Space 指标（补充）

以上分析均在 **decoded pixel space** 进行，受 decoder Jacobian 的放大/压缩效应影响。E1 脚本同时记录了 **latent-space MSE**：

```python
latent_mse = ((z_pred - z_GT).pow(2)).mean()  # per-slice latent MSE
```

E1 按 hop 记录了 `latent_mse_per_tp`（代码 L143）。这是不经过 decoder 的直接 transport 误差，不受 decoder 非线性影响。

完整的瓶颈归因应包含：

| 层面 | 指标 | 意义 |
|---|---|---|
| Pixel (decoded) space | PSNR gap + MSE 比值 | 用户可感知的最终质量 |
| Latent space | latent MSE per hop | transport 速度场精度的直接衡量 |
| Decoder sensitivity | $\|J_D \cdot \delta z\| / \|\delta z\|$ | decoder 对 latent perturbation 的放大率 |

> latent_mse 的原始 JSON 在远程服务器。如需完整归因，需将 E1 输出 sync 进 review 树。

## 7. 全部 4 跳 PSNR 域结果

| 跳 | psnr_ceil | psnr_e2e | gap_total | gap_transport | gap_decoder | PSNR-gap diagnostic ratio |
|---|---:|---:|---:|---:|---:|---:|
| D50→D20 | 46.64 | 35.49 | 11.15 | 10.74 | 0.41 | transport 96.3% |
| →D10 | 48.74 | 35.86 | 12.88 | 12.33 | 0.55 | transport 95.7% |
| →D4 | 50.83 | 36.42 | 14.41 | 13.98 | 0.43 | transport 97.0% |
| →NORMAL | 52.63 | 36.74 | 15.89 | 15.70 | 0.19 | transport 98.8% |

注意：
- gap_transport 随级联跳数**单调递增**（10.74 → 15.70 dB）——exposure bias 逐级累积
- gap_decoder 几乎不变（0.19-0.55 dB）——decoder self floor 与 transport 误差基本无关
- PSNR-gap diagnostic ratio 只是 dB 域比例，不可解读为 MSE 线性占比

## 8. 对 V6 的意义

### 8.1 Transport 是当前主要优化方向

在当前 decoder ceiling 远高于 E2E PSNR 的条件下（46-53 dB vs 35-37 dB），优先优化 transport 的收益显著高于继续优化 decoder。

| 改善方向 | MSE 比值参考 | 可行收益 |
|---|---|---|
| 优化 transport | MSE_e2e/MSE_ceil ≈ 13-39× | 大：缩小比值即可提升 PSNR |
| 优化 decoder | MSE_e2e/MSE_pred_gt ≈ 1.04-1.13× | 小：decoder floor 影响已很低 |

**Decoder 暂不应作为 V6 主攻方向，但仍需监控**：
- patch artifact（视觉质量，PSNR 不完全捕捉）
- latent perturbation amplification（decoder Jacobian 放大效应）
- 高频纹理重建质量

### 8.2 V6 的三个机制如何作用

```
Transport-decoder gap（E2E MSE 是 ceiling MSE 的 13-39×）来源：
├── hop0 velocity 精度不足 → pair_weight=15 + pair_loss_weights=[2.5,1,1,1]
│   └── 强化 GT velocity 学习，减少 latent MSE at hop0
├── hop1-3 exposure bias 累积 → rollout ramp 0→4.0 + step_weights=[0.5,2,1.5,1]
│   └── 抑制级联 MSE 指数增长
└── image_aux 挤占 weighted loss fraction → λ_img 从 0.12 固定到 0.04
    └── 释放 loss 预算给 transport 通道
```

## 9. 脚本代码核心（简化版）

```python
# scripts/diagnose_error_budget.py 核心逻辑（L131-L151）

for each val slice:
    z_GT   = encoder(x_GT)           # GT latent（预计算，从 latent file 加载）
    z_pred = transport_chain(z_D50)   # 4-hop 级联预测

    x_from_gt   = decoder(z_GT)       # decoder 重建
    x_from_pred = decoder(z_pred)     # E2E 输出

    # 三个 PSNR（per-slice）
    psnr_ceil    = calc_psnr_clip3(x_from_gt,   x_GT)       # Decoder Ceiling
    psnr_e2e     = calc_psnr_clip3(x_from_pred, x_GT)       # E2E
    psnr_pred_gt = calc_psnr_clip3(x_from_pred, x_from_gt)  # Decoder-mediated transport discrepancy

    # Latent-space MSE（per-slice, per-hop）
    latent_mse   = ((z_pred - z_GT).pow(2)).mean()

# 分解（对 mean PSNR 操作）
gap_total     = mean(psnr_ceil)    - mean(psnr_e2e)       # Total PSNR gap
gap_transport = mean(psnr_ceil)    - mean(psnr_pred_gt)   # Transport-related PSNR gap
gap_decoder   = mean(psnr_pred_gt) - mean(psnr_e2e)       # Decoder floor PSNR gap

# 代数验证
assert abs(gap_transport + gap_decoder - gap_total) < 0.01  # ✓ (A-C) = (A-B) + (B-C)
```

## 10. 术语对照表

| 本文术语 | 含义 | 注意事项 |
|---|---|---|
| PSNR-gap diagnostic ratio | dB 差值占总 dB gap 的百分比 | 不是 MSE 域线性误差贡献比 |
| Decoder Ceiling | dec(z_GT) vs x_GT 的 PSNR | encoder-decoder 重建上界 |
| Decoder-mediated transport discrepancy | dec(z_pred) vs dec(z_GT) 的 PSNR | 含 decoder Jacobian 影响 |
| gap_decoder | 参照从 dec(z_GT) 换到 x_GT 后的 PSNR 落差 | decoder reconstruction floor 的量化 |
| MSE 比值 | $10^{\text{dB gap}/10}$ | 比 dB gap 更有物理意义 |
