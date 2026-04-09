# PLAN_V3_Seam_Stress_and_Hop0_Attribution.md

## Goal
针对“padding/拼接割裂”现象做**应力测试 + 归因验证**，确认问题来自 hop0 表征缺口而非评估偶然性。

## Scope
- 默认不改代码；先做评估与数据分层。
- 若 V1/V2 均失败，再进入“需批准”的最小实现变更。

## Hypothesis
D50 输入噪声更重，DINO 语义化后局部结构损失更明显，导致 hop0 预测 latent 在 decode 时更依赖 patch 先验，形成接缝割裂。D20 起点信息更充分，因此到 NORMAL 可优于 baseline。

## Experiment Blocks

### Block A: Seam-aware Slice Stratification (no code change)
按边界梯度不连续度将 val slices 分三桶（low/mid/high seam-risk），分别统计：
- `D50→D20`
- `D50→NORMAL`
- `D20→NORMAL`

若异常主要出现在 high seam-risk 桶，支持“结构割裂驱动”假设。

### Block B: Hop0 Forcing Ablation (minimal config toggles)
在相同 checkpoint 训练预算下对比：
- B1: 默认 gate/lambda
- B2: 减弱 hop0 forcing（更小 `pixel_gate_init`, `lambda_hop_init`）
- B3: 增强 hop0 forcing（适度增大 gate/lambda）

观察 seam-risk 高桶中 D50→NORMAL 的响应曲线是否单峰（过弱/过强都差）。

### Block C: Teacher-forced vs Pure-pred divergence
统一输出每 hop 的 TF 与 pure-pred 差值：
- `gap_h1 = PSNR_TF(D20) - PSNR_PURE(D20)`
- `gap_final = PSNR_TF(NORMAL) - PSNR_PURE(NORMAL)`

若 V1/V2 降低 `gap_h1` 且同步提升 D50→NORMAL，则可判定问题核心在 hop0 级联放大。

## Optional (requires explicit approval)
若以上仍不能收敛，可申请最小代码实验：
- 增加 hop0 seam-aware aux loss 权重调度（仅 hop0）
- 仍保持 latent-only rollout state，不引入 dual-state

## Gate
- PASS: high seam-risk 桶中 `D50→NORMAL +0.30 dB` 且 `D20→NORMAL` 不退化
- FAIL: 各桶趋势不一致或改动只带来随机波动

## Deliverables
- `seam_strata_summary.csv`
- `hop0_ablation_grid.csv`
- `tf_vs_pure_gap.json`
- 每个实验附 `clip3` 口径 meta 字段
