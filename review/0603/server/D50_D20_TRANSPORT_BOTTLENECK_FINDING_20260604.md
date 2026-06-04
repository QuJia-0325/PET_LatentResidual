# D50→D20 Transport Bottleneck Finding

日期: 2026-06-04  
分支: `foc_lite_hop0`  
目的: 固化“D50→D20 是 PET latent transport 首跳信息瓶颈”的数据证据、物理解释和论文表述边界。

## 结论

当前数据支持一个谨慎但有力的结论：

> `D50→D20` 是本项目 transport chain 中证据最强的首跳瓶颈 / 信息瓶颈。它在 latent 位移、`σ·Δt`、single-step PSNR 和部分 seam 指标上排名最差；但现有数据不支持“数量级最坏”的强叙事。

这意味着论文中可以把 `D50→D20` 写成 dominant information bottleneck，但不应写成 order-of-magnitude worse hop。

## 1. 本地逐跳数据证据

### 1.1 S2.0 single-step 逐跳表

来源: `review/0603/server/S2_singlehop_seam_evidence_report_20260604.md` 与 `review/0603/server/s2_singlehop_seam_evidence.csv`。

该实验对每个 hop 使用 GT source latent 做单步预测，避免 cascade endpoint 把早期误差传播到后续 timepoint 后造成归因混淆。

| hop | latent RMS | `σ·Δt` | V13 seam | A4 seam | V13 PSNR | A4 PSNR |
|---|---:|---:|---:|---:|---:|---:|
| `D50→D20` | 0.033605 | 0.028902 | 0.025910 | 0.019785 | 35.217159 | 35.491798 |
| `D20→D10` | 0.016295 | 0.014730 | 0.023910 | 0.024026 | 39.209922 | 39.210833 |
| `D10→D4` | 0.012672 | 0.011625 | 0.021784 | 0.020862 | 41.052393 | 41.213092 |
| `D4→NORMAL` | 0.009910 | 0.010575 | 0.019263 | 0.018924 | 42.978035 | 43.051446 |

关键比例：

- 首跳 latent RMS 是第二高 hop 的 `2.06×`。
- 首跳 `σ·Δt` 是第二高 hop 的 `1.96×`。
- 首跳 V13 seam 是第二高 hop 的 `1.08×`。
- 首跳 V13 ext-seam 是第二高 hop 的 `1.49×`。
- 首跳 V13 PSNR 比第二低 hop 低 `3.99 dB`。
- 首跳 A4 PSNR 比第二低 hop 低 `3.72 dB`。

判读：`D50→D20` 在位移量级和单步重建质量上是最坏 hop；seam 也偏高，但 seam 证据是弱到中等，不是数量级差异。

### 1.2 历史 Path A teacher-forcing / rollout 诊断

来源: `review/0425/logs_diag/pathA_C_multi_ckpt_gpu0.log`。

历史 Path A 表显示，`D50→D20` 的 single-step PSNR 在多个 checkpoint 下持续最低：

| checkpoint | `D50→D20` PSNR | `D20→D10` PSNR_TF | `D10→D4` PSNR_TF | `D4→NORMAL` PSNR_TF |
|---|---:|---:|---:|---:|
| step 10k | 35.0966 | 38.3849 | 40.1067 | 36.4053 |
| step 30k | 35.4822 | 39.0446 | 40.8981 | 42.0085 |
| best | 35.5690 | 39.3231 | 41.2132 | 43.1281 |

判读：即使在 GT source latent 的 teacher-forcing 口径下，首跳仍是绝对 PSNR 最低的 hop，说明困难不是单纯由 open-loop cascade 曝光偏差造成。

### 1.3 V7/V8 per-hop single-step 解耦

来源: `review/0517/disambig/per_hop_singlestep/SINGLESTEP_REPORT.md`。

| hop | V7_last single-step | V8_last single-step | Δ V7-V8 | paired t | win % |
|---|---:|---:|---:|---:|---:|
| `D50→D20` | 35.4354 | 35.2094 | +0.2260 | 43.10 | 85.8% |
| `D20→D10` | 39.2081 | 39.1875 | +0.0206 | 5.29 | 71.0% |
| `D10→D4` | 41.1100 | 41.1104 | -0.0004 | -0.17 | 52.3% |
| `D4→NORMAL` | 43.0138 | 43.0325 | -0.0187 | -3.41 | 52.2% |

判读：V7/V8 的主要 single-step 改进集中在 `D50→D20`。后续 hop 的 direct single-step gain 接近 0 或为负，说明最终 NORMAL 改善更像是首跳更准后沿 open-loop chain 传播出来的 coherence gain，而不是每个 hop 都独立变强。

### 1.4 ROI / SUV 证据

来源: `review/0517/disambig/roi_psnr/ROI_REPORT.md`。

ROI 报告指出：

- `D20` 阶段 SUVmax underestimation 最大，之后向 `NORMAL` 逐步减小。
- 最困难区域是 high-SUV / top-1% uptake 区域，而不是全图平均 PSNR 能完全反映的问题。
- 后续建议是优先做 hop0/control 或 D20/top-SUV ROI-weighted loss，而不是仅根据 NORMAL 指标强化 multi-hop 或 β_NORMAL。

这与“首跳 / 早期高噪声输入是主要瓶颈”的解释一致。

## 2. 物理解释

PET 低剂量 / 低计数图像的噪声主要来自 count statistics。计数越低，观测中的结构和强度信息越不可靠，恢复问题越不适定。

在本项目的剂量链条中：

- `D50` 是最高噪声 / 最低可靠信息状态。
- `D20` 已经要求恢复出更接近低噪声轨迹的结构和 uptake 分布。
- 因此 `D50→D20` 是从最差 SNR 输入中恢复中噪声目标的第一步，本质上承担了最多的信息恢复压力。

这解释了为什么首跳 latent RMS 和 `σ·Δt` 最大，也解释了为什么首跳 single-step PSNR 显著低于后续 hop。

## 3. Transport 机制中的级联影响

Transport chain 可写成：

```text
z_D20_pred = F0(z_D50)
z_D10_pred = F1(z_D20_pred)
z_D4_pred  = F2(z_D10_pred)
z_N_pred   = F3(z_D4_pred)
```

如果首跳产生误差：

```text
z_D20_pred = z_D20_gt + δ0
```

则后续 hop 的输入不再位于训练时 GT source latent 分布上，而是位于偏移后的 latent manifold 邻域。后续 transport 不只是继续降噪，而是在错误起点上积分速度场。因此首跳误差会通过三条路径影响最终图像：

1. **状态偏移传播**：`δ0` 被后续 `F1/F2/F3` 继续传播。
2. **off-manifold 解码风险**：偏移 latent 经 decoder 可能放大为纹理噪声、seam 或热点伪影。
3. **SUV bias 累积**：早期高摄取区域恢复不足会影响后续 uptake refinement，导致 SUVmax / top-SUV 区域系统性低估。

因此，首跳损失不是局部指标损失，而是整个 open-loop transport 轨迹的初始条件误差。

## 4. 文献支撑方向

外部 PET 文献可以支持“低计数 / 低剂量 PET 更难恢复”的物理背景：

- PET 图像质量受 detected counts、注射活度、采集时长和统计噪声影响；低 count level 会降低 SNR 并增加重建不确定性。
- 低剂量 / ultra-low-dose PET denoising 文献通常把不同 count level 当作不同难度条件，说明模型需要适配不同噪声强度。
- 这些文献支持“从高噪声 `D50` 恢复到 `D20` 更困难”的物理直觉。

但注意：文献不能直接证明本项目离散 chain 中 `D50→D20` 必然最难。该结论主要来自本项目 S2.0、Path A、per-hop single-step 和 ROI 数据。

## 5. 推荐论文表述

推荐强度适中的版本：

> Empirically, the first transport step `D50→D20` is the dominant information bottleneck of our PET latent transport chain. It has the largest latent displacement, the largest `σ·Δt`, the lowest single-step PSNR, and the strongest early-stage uptake underestimation. This is consistent with PET count-statistics intuition: the noisiest low-count input provides the least reliable structural and uptake information. Because subsequent steps are open-loop transports initialized from the predicted `D20` latent, hop0 errors are especially consequential and can propagate as texture noise, seam artifacts, and SUV bias in the final reconstruction.

不推荐的过强版本：

> `D50→D20` is an order-of-magnitude harder than all other hops.

原因：S2.0 中首跳 seam 只比第二高 hop 高 `1.08×`，ext-seam 高 `1.49×`，不支持数量级叙事。

## 6. 后续实验建议

若要把该结论进一步增强为论文核心论点，建议补充：

1. **多 seed single-step panel**：确认 `D50→D20` lowest-PSNR ranking 不依赖单 seed。
2. **patient-level grouped statistics**：避免 slice-level 统计夸大显著性。
3. **D20/top-SUV ROI-weighted hop0 loss**：直接测试“修首跳高摄取区域是否改善最终 NORMAL”。
4. **error propagation sensitivity**：向 GT `z_D20` 注入不同幅度扰动，测 NORMAL PSNR/SUVmax 对 `δ0` 的敏感度。
5. **文献段落补强**：在论文 related work / motivation 中引用低计数 PET count statistics 与 low-dose PET denoising 文献，作为物理背景，而非替代本地逐跳证据。
