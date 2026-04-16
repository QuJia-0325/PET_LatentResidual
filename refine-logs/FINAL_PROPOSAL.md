# Final Proposal: Post-Idea2 Refined Route

## Problem Anchor
- Bottom-line problem:
  在固定纯预测级联 `D50->D20->D10->D4->NORMAL` 下，提升全链路质量，尤其是首跳 `D50->D20`，并抑制 decoder 输出分块/缝隙纹理。
- Must-solve bottleneck:
  1) 首跳漂移导致后续误差累积；2) 训练目标与部署目标不一致时，checkpoint 选择容易失真；3) pixel 辅助与链路目标存在拉扯。
- Non-goals:
  不引入 dual-state transport、不改主干为 3D、不扩大为复杂多分支大系统。
- Constraints:
  保持 `latent-only` 主状态、单卡 50k 预算约 16-29h、评估必须 `calc_psnr_clip3`、主指标以 full-val chain 为准。
- Success condition:
  在对称 full-val 评估下，相比 `chainstable(last_full)` 在 `tail_avg` 与 `NORMAL` 上稳定提升，并且首跳 D20 不退化。

## What Idea2 Taught Us (Frozen Lessons)
1. `idea2_best` 略低于 `chainstable_best`，但 `idea2_last` 可高于 `chainstable_best`，说明方法方向可救但选点有偏差。
2. `chainstable_last` 进一步优于 `chainstable_best`，证明 rolling-window best 不能作为最终结论依据。
3. 高复杂度 CCT 组合（matching + conflict_damp）在当前实现下并未形成稳健主收益。

## Final Method Thesis
以 **数值稳定优先** 的最小机制推进：

> 在 `chainstable(last)` 的稳定底座上，加入 **Hop0 First-Order ODE Calibration (FOC-lite)** 作为唯一主创新，先修首跳局部积分误差；然后仅在该主线通过后，叠加一个 **pred-latent seam adaptor (PLPP-lite)** 作为可选视觉修复支线。

## Contribution Focus
- Dominant contribution:
  **FOC-lite**：在 hop0 施加 `1-step vs 2 half-steps` 一致性约束，直接抑制首跳数值跃迁误差，目标是提升 D20 并减少尾段放大。
- Optional supporting contribution:
  **PLPP-lite**：仅针对 `predicted latent decode` 的 seam-aware 小残差修复头，用于降低分块风险，在不伤害链路指标前提下提升视觉连续性。
- Explicitly rejected complexity:
  - 继续堆叠 Alpha-CCT 的 matching/conflict_damp 变体作为主线。
  - 直接上样本级复杂自适应控制器（DSGC）作为下一步主实验。
  - 引入第二条 rollout state 或 dual transport。

## Method Design (Implementation-Oriented)

### M1. FOC-lite (Primary)
- Scope:
  仅 `hop_idx==0` 触发。
- Core loss:
  `L_foc = ||Phi_1(z_D50) - Phi_1/2(Phi_1/2(z_D50))||`。
- Integration:
  在现有 `pair + rollout + image_aux` 之上添加小权重正则项，默认不改变主干结构。
- Why minimal:
  不增加新网络，仅增加一次 hop0 额外前向路径。

### M2. PLPP-lite (Optional)
- Scope:
  仅对 `Dec(z_pred)` 输出做小残差修复，训练分布限制为 `pred-latent decode`。
- Loss:
  seam/high-frequency 优先，避免全图均匀平滑。
- Guardrail:
  任何引入若导致 chain 指标退化则立即回退。

## Frontier Primitive Position
- Frontier primitive is **optional, not central**.
- 本轮不以“新扩散/新引导机制”为主贡献，重点是可验证的数值稳定与部署一致性。

## Claims (Must-Prove)
- C1 (primary): FOC-lite 在 full-val 上提升 `D20` 并带动 `D10/D4/NORMAL` 至少不退化。
- C2 (supporting): 在 C1 成立后，PLPP-lite 能降低 seam 风险分桶指标且不损害 chain PSNR。

## Final Verdict
**READY**（进入受控实验阶段）
- 条件：严格执行对称 full-val 评估与固定 checkpoint 选择协议。
