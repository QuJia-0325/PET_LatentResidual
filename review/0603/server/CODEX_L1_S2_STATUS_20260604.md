# L1-only 与 S2 任务执行说明

日期: 2026-06-04

## 结论摘要

1. L1-only 已完成同口径 full-val 评估。NORMAL PSNR 从 V13 的 36.494330 提升到 36.715786，增益为 +0.221455 dB；A4-mid 为 36.893917，增益为 +0.399587 dB。
2. L1-only 复现了 A4-mid NORMAL PSNR 增益的 55.42%，低于任务文件中“≥70% 说明 L1 基本复现 A4”的阈值。因此当前证据支持：冻结 decoder 下的像素域 L1 约束贡献显著，但 SSIM/seam 复合项仍提供额外、可测的增益。
3. Copilot/Supervisor 提出的 S2 机制实验方向是合理的，因为它直接检验论文中 `M=J_G^T J_G` pulled-back metric 是否只是动机，还是被数据支持的机制证据。
4. 但 S2 原始方案存在两个风险：第一，`bottom-k` 奇异值和条件数在 196608 维 latent 空间中用 matrix-free 方法很难可靠估计；第二，S2.0 结果只弱支持“首跳最坏”，不支持“数量级最坏”叙事。
5. 因此我没有直接启动更重的 S2.a/S2.b/S2.c 全量 Jacobian 谱实验；我先完成了任务文件要求的 S2.0，并补跑了更严格的 single-step S2.0。当前建议是：论文可写“D50→D20 排名最坏但差距有限”，不应写“数量级最坏”。

## L1-only 结果

| metric | V13 | L1-only | A4-mid | L1-vs-V13 | A4-vs-V13 | A4-vs-L1 |
|---|---:|---:|---:|---:|---:|---:|
| PSNR | 36.494330 | 36.715786 | 36.893917 | +0.221455 | +0.399587 | +0.178132 |
| MS-SSIM | 0.982886 | 0.983754 | 0.984347 | +0.000868 | +0.001461 | +0.000593 |
| seam | 0.021355 | 0.019526 | 0.016552 | -0.001829 | -0.004803 | -0.002974 |
| ext-seam | 0.010044 | 0.009512 | 0.008924 | -0.000532 | -0.001120 | -0.000588 |

判读：

- L1-only 不是失败对照；它本身带来稳定增益，说明“通过冻结 decoder 的像素域约束”是有效信号。
- A4-mid 相对 L1-only 仍有 +0.178132 dB NORMAL PSNR、-15.23% seam、-6.18% ext-seam 的增量，因此不能把 A4 全部解释为 L1；复合结构有实证必要性。
- 论文措辞应避免“seam/SSIM 是唯一主因”。更准确表述是：L1 提供主要像素域锚定，SSIM/seam 进一步改善结构一致性与 patch-boundary 平滑。

## S2 实验合理性

S2 的核心问题是：论文使用 `M=J_G^T J_G` 解释 decoder-aware supervision，为何能优先修正对像素域影响更大的 latent 误差方向。这个问题确实必须验证，否则该理论段落容易被审稿人视为装饰性解释。

合理部分：

- S2.0 用逐跳数据先确认“最坏跳”经验基础，顺序正确。
- S2.a 用 matrix-free `M v = J_G^T J_G v` 验证 decoder metric 是否各向异性，能直接支撑或否定理论前提。
- S2.b 把 A4/V13 增益投影到高-M 与低-M 子空间，能检验“高-M 优先修正”这一机制 claim。
- S2.c 检验 top-M 图像响应是否贴合 14px patch grid，是连接 `M` 与 seam 现象的关键桥。

主要问题：

- 原方案要求 bottom-k 奇异值、条件数和有效秩。对 768×16×16 latent，显式矩阵不可行；matrix-free Lanczos 可较可靠估 top spectrum，但 bottom spectrum/条件数很容易不稳定。后续若跑 S2.a，建议先改成 top spectrum、Hutchinson trace、spectral concentration pilot，不要强报 bottom-k。
- S2.b 的一阶近似只在局部小扰动下成立；如果 `z_pred-z_gt` 较大，`δ^T M δ` 与真实像素误差可能相关性不足。必须把相关性作为先验检验，而不是默认成立。
- S2.c 的“top-M = patch boundary”是强断言，可能被证伪。若证伪，论文应保留 seam 的经验观察，但不能把 seam 现象归因于 top-M patch-grid 对齐。

## S2.0 已运行结果

我运行了两个只读版本：

1. cascade endpoint 口径：读取已有 B-ext full-val per-slice CSV 和 GT latent 差分。
2. single-step 口径：V13/A4 分别从每个 hop 的 GT source latent 单步预测 GT destination，避免 cascade 误差传播混淆。

single-step 是更严格口径，结论如下：

| hop | latent_rms | sigma_dt | V13 seam | A4 seam | seam delta | V13 PSNR | A4 PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| D50->D20 | 0.033605 | 0.028902 | 0.025910 | 0.019785 | -23.64% | 35.217159 | 35.491798 |
| D20->D10 | 0.016295 | 0.014730 | 0.023910 | 0.024026 | +0.49% | 39.209922 | 39.210833 |
| D10->D4 | 0.012672 | 0.011625 | 0.021784 | 0.020862 | -4.23% | 41.052393 | 41.213092 |
| D4->NORMAL | 0.009910 | 0.010575 | 0.019263 | 0.018924 | -1.76% | 42.978035 | 43.051446 |

判读：

- D50->D20 在 latent RMS、sigma_dt、V13 seam、V13 ext-seam 上均排名第一。
- 但首跳 seam 只比第二高的 D20->D10 高 1.084 倍，ext-seam 高 1.487 倍；这不是“数量级最坏”。
- 因此 S2.0 未证伪“首跳最坏”，但证伪了强叙事：“D50->D20 seam 数量级最重”。后续论文必须写成弱叙事。

## 是否继续 S2.a/S2.b/S2.c

当前我不建议直接启动全量 S2.a/S2.b/S2.c，理由是：

1. S2.0 只给弱支持，强机制实验即使跑通，也不能支撑“首跳数量级最坏”的论文开场。
2. 原始 S2.a 的 bottom-k/条件数要求计算风险高，容易产出不稳定数字。
3. 更合理的下一步是先做小样本 pilot：N=16 或 32，只估 top-M spectrum、top-M seam energy enrichment、以及 `δ^T M δ` 与像素 MSE 的相关性；如果 pilot 支持，再扩到 N=64。

建议的下一步实验定义：

- `S2.a-pilot`: N=16，NORMAL 与 D20 各取 8 个 slice，只报告 top singular values、Hutchinson trace、top-energy concentration。
- `S2.c-pilot`: 对 top-8 M 方向做 JVP，量化 patch-grid mask 能量占比，与随机方向对比。
- `S2.b-pilot`: 对同一 N=16 slice，比较 A4/V13 的 `δ^T M δ` 与 pixel MSE 相关性；若相关性低，停止 M 机制叙事。

## 输出文件

- L1-only eval JSON: `review/0603/server/c_l1_only_first_hop_224_val_clip3_eval.json`
- L1-only eval CSV: `review/0603/server/c_l1_only_first_hop_224_val_clip3_eval.csv`
- L1-only eval log: `review/0603/server/c_l1_only_eval_gpu0_20260604.log`
- L1 三联报告: `review/0603/server/l1_only_analysis_20260604.md`
- L1 三联表: `review/0603/server/l1_only_three_way_metrics.csv`
- L1 seam 分层表: `review/0603/server/l1_only_seam_strata_summary.csv`
- S2.0 endpoint 表: `review/0603/server/per_hop_seam_evidence.csv`
- S2.0 endpoint JSON: `review/0603/server/per_hop_seam_evidence.json`
- S2.0 endpoint 报告: `review/0603/server/S2_per_hop_seam_evidence_report_20260604.md`
- S2.0 single-step 表: `review/0603/server/s2_singlehop_seam_evidence.csv`
- S2.0 single-step JSON: `review/0603/server/s2_singlehop_seam_evidence.json`
- S2.0 single-step 报告: `review/0603/server/S2_singlehop_seam_evidence_report_20260604.md`
- S2.0 single-step log: `review/0603/server/s2_singlehop_gpu0_20260604.log`
