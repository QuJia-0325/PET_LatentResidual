# Refinement Report

**Date**: 2026-04-14  
**Scope**: 基于 Idea2 实现问题，对剩余 ideas 进行方法收敛与风险裁剪。

## Phase 0: Triage
- `refine-logs/` 在本仓库此前不存在，已新建。
- 关键补洞已完成：新增 `chainstable last.pt` 的 full-val eval。
  - [chainstable last full eval](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_formal_v3_chainstable_eval_clip3_last_full/first_hop_224_val_clip3_eval.json)
- 关键观测：
  - `chainstable_last_full` > `chainstable_best_full`
  - `idea2_last_full` > `idea2_best_full`
  - 但 `idea2_last_full` 仍低于 `chainstable_last_full`（主要在 NORMAL）

## Phase 1: Method Refinement Decision

候选机制与裁决：
1. Alpha-CCT 扩展（matching/conflict 继续堆叠）
- 结论：不作为主线。
- 原因：可解释性与稳定性不足，且增益对 checkpoint 选择敏感。

2. FOC-lite（Hop0 数值一致性）
- 结论：主线保留。
- 原因：最小机制、直接命中首跳、计算可控、归因清晰。

3. PLPP-lite（pred-latent seam adaptor）
- 结论：保留为次级支线。
- 原因：对分块问题针对性强，但必须受 chain 指标约束。

4. MAP-hop-lite / DSGC
- 结论：暂缓。
- 原因：额外自由度偏高，先等主线证据稳定。

## Phase 2: Planning Gate Check

- Final method thesis?
  `chainstable(last)` + `FOC-lite`，必要时再加 `PLPP-lite`。
- Dominant contribution?
  Hop0 数值稳定校准（FOC-lite）。
- Explicitly rejected complexity?
  Alpha-CCT 复杂扩展、复杂自适应控制器、dual-state。
- Reviewer concerns still critical?
  checkpoint 选择偏差、3-seed 稳定性、seam 桶泛化。
- Frontier primitive central/optional/absent?
  **Optional**（非中心贡献）。

Gate verdict: **PASS**（可进入实验计划阶段）

## Phase 3: Claim Set Freeze

- C1: FOC-lite 可提升 D20 且不伤害尾段（D10/D4/NORMAL）
- C2: PLPP-lite 可降低 seam 风险且不损害 chain 指标
- Anti-claim to rule out:
  - 结果仅来自 checkpoint 选择噪声
  - 视觉提升但链路退化

## Key Risk Ledger
1. 风险：FOC 权重过大导致过平滑。
- 缓解：权重网格小步扫描（0/0.03/0.06/0.10）+ tail guardrail。

2. 风险：PLPP 仅修图不修链路。
- 缓解：强制报告 chain 与 seam 双指标，若 tail 退化立即回退。

3. 风险：单 seed 偶然性。
- 缓解：通过单 seed gate 后再扩展 3 seeds。
