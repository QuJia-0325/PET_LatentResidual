# IDEA_REPORT.md

**Direction**: PET_LatentResidual 224 first-hop next-step ideation under strict constraints (no code changes without approval).
**Generated**: 2026-04-09
**Ideas evaluated**: 3 Gate-1 candidates → 2 primary recommendations
**Review protocol**: 每个 idea 必须包含 novelty 量化评分 + 三方对抗审查（Proposer / Novelty Skeptic / Engineering Skeptic）。

## Executive Summary
当前最可发表且与现有实现最兼容的路线是：
1) 先做 **CCT-224**（一致性训练，直接打分布偏移），
2) 再叠加 **ΔB-aware Adaptive Reweighting**（把 A/B/C/D 诊断闭环到训练）。

## Implementation Status
- `2026-04-09` 版本控制状态：`Idea1 / CCT-224` 已迁移到测试分支 `idea1_cct_test`；master 恢复为原始 D50 hop0 baseline。
- 注意：主分支当前不包含 `training.cct`；如需复现实验性 CCT 线路，请切换到测试分支 `idea1_cct_test`。

## Gate-1 Ranked Ideas (2026-04-09)

### 🏆 Idea 1: CCT-224（Counterfactual Consistency Training） — RECOMMENDED
- **Hypothesis**: 级联漂移的主因是训练时 teacher-forced 输入分布与推理时 pure-pred 输入分布不一致；对两条路径施加一致性约束可降低漂移。
- **How to integrate**: 在现有 `pair + rollout + hop0 img aux` 不变前提下新增 consistency term。
- **Expected signal**: first-hop 与 end-to-end 同时改善，且 A/B gap 收缩。
- **Risk**: 中等（loss 竞争导致不稳），可由现有 warmup/ramp + fail-fast 控制。
- **Novelty positioning**: 方法原型可追溯到一致性/蒸馏思想，但在 PET 固定 4-hop 级联与 hop0 条件场景下具备可发表的任务特异性。
- **Novelty score**: **8.4 / 10**
  - **Scoring basis**: 任务特异性 3.0/3 + 与现有实现差异性 2.8/3 + 可验证性 2.6/4。

#### Three-party adversarial review
- **Proposer**: CCT 直接命中“训练/推理分布偏移”这一主因，且不破坏当前统一 `predict_latent_step(...)` 入口，适合做第一阶段主路线。
- **Novelty Skeptic**: 一致性训练本身并非新范式，若只写成普通 consistency regularization，容易被审稿人视为“常规蒸馏变体”。
- **Engineering Skeptic**: 双路径 loss 可能与现有 pair/rollout/img loss 竞争，导致 warmup 早期不稳或 first-hop 指标短期波动。
- **Gate decision**: **PASS (Phase-1)**。理由：收益面最大、与当前代码路径兼容度最高；通过严格 ablation（仅 CCT / CCT+rollout / CCT+img）应对 novelty 与稳定性质疑。

### Idea 2: ΔB-aware Adaptive Reweighting — RECOMMENDED (Phase-2)
- **Hypothesis**: 静态权重无法持续对准瓶颈；用 `ΔB=B-A` 作为在线反馈做权重调节，可持续压低首跳误差传播。
- **How to integrate**: 动态调整 pair/rollout/hop0-loss 的权重或采样分布，不改主干结构。
- **Expected signal**: `ΔB` 下降，首跳提升且 tail 不退化。
- **Risk**: 指标噪声引发权重抖动，需 EMA 与更新阈值。
- **Novelty positioning**: 不是新网络，而是“因果诊断驱动训练闭环”的工程-方法结合点。
- **Novelty score**: **7.6 / 10**
  - **Scoring basis**: 诊断闭环独特性 2.7/3 + 方法新颖度 2.2/3 + 可发表性/可复现性 2.7/4。

#### Three-party adversarial review
- **Proposer**: 该方案把 A/B/C/D 诊断从“离线分析”升级为“在线控制信号”，能持续把训练资源推向真实瓶颈。
- **Novelty Skeptic**: 若只体现为 heuristic reweighting，可能被认为是调参策略而非方法贡献；需强调诊断-控制闭环的可解释性与可迁移性。
- **Engineering Skeptic**: ΔB 度量噪声会导致权重抖动，需 EMA、更新频率限制、最小变更阈值，避免训练震荡。
- **Gate decision**: **PASS (Phase-2)**。理由：在 CCT 稳定后叠加，能形成“误差定位→权重纠偏”的闭环；先保证控制器稳定性再追求收益上限。

### Idea 3: Uncertainty-Gated Hop0 Forcing — BACKUP
- **Hypothesis**: 困难样本应获得更强 hop0 像素注入。
- **Risk**: 易被视为局部门控trick，且与现有门控/残差头耦合，调参成本较高。
- **Novelty score**: **6.3 / 10**
  - **Scoring basis**: 机制增量性较强（2.0/3），但可操作性较好（2.3/4），任务相关性中等（2.0/3）。

#### Three-party adversarial review
- **Proposer**: 通过不确定性驱动的样本级门控，理论上可把 hop0 注入集中到“真正困难样本”，提高参数利用效率。
- **Novelty Skeptic**: 容易被归类为 gating trick，缺少独立方法贡献；若没有明显超过 CCT 路线，发表价值偏弱。
- **Engineering Skeptic**: 与现有 `g_pix` / `lambda_hop` 耦合强，调参面扩大，训练稳定性和可复现性风险高。
- **Gate decision**: **HOLD (Backup only)**。理由：仅当前两项收益不足或出现上限瓶颈时再启用，避免分散主线资源。

## Execution Order
1. Phase-1: CCT-224
2. Phase-2: CCT-224 + ΔB-aware adaptive reweighting
3. Phase-3: uncertainty-gated hop0 (only if needed)

## Governance
- Canonical implementation constraints: `docs/main.md`
- Idea backlog and ranking: this file (`IDEA_REPORT.md`)
- No architecture/code modification before explicit user approval.
- Gate 评审硬要求：每个 idea 必须同步更新 **Novelty score + Three-party adversarial review + Gate decision**。
