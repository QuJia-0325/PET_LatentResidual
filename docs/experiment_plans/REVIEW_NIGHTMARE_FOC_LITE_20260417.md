# NIGHTMARE Review: FOC-lite (2026-04-17)

**Mode**: NIGHTMARE (三方对抗)
**Target**: FOC-lite (Hop0 First-Order ODE Calibration)
**Branch**: `foc_lite_hop0`

---

## Reviewer Matrix

| Reviewer | Role | Veto Target | Verdict |
|---|---|---|---|
| A (Primary) | 总评打分 | 可否决"可投稿" | **VETO** |
| B (Opposition) | 攻击漏洞 | 可否决"因果成立" | **VETO** |
| C (Verifier) | 代码核查 | 可否决"结果可信" | **NO VETO**（附修复） |

**Final**: **FAIL (3.53/10)**，双 VETO (A+B)

---

## Reviewer-A 评分

| 维度 | 权重 | 分数 | 加权 | 理由 |
|------|------|------|------|------|
| Problem-Mechanism Fit | 20% | 6.5 | 1.30 | 确实针对 hop0，但攻击的是症状（积分误差）而非根因（encoder 信息瓶颈+decoder 冻结） |
| Causal Evidence | 20% | 2.0 | 0.40 | 零实验证据。hop0 gap 中 ODE 积分误差的占比未测量 |
| Robustness | 15% | 1.0 | 0.15 | 无运行、无 seed、无重复 eval |
| Adversarial Survivability | 15% | 3.5 | 0.53 | t=3.5 无物理意义、stop-gradient 单向偏差、CCT 负先验 |
| Artifact/Seam | 15% | 1.0 | 0.15 | 纯 latent 操作，对 14px 网格伪影零影响 |
| Engineering Feasibility | 10% | 8.0 | 0.80 | 实现干净，~110 行，config-driven |
| Novelty | 5% | 4.0 | 0.20 | Richardson extrapolation 应用到单 hop |
| **Total** | | | **3.53** | |

---

## Reviewer-B 攻击摘要

1. **因果链循环**：velocity inconsistency → PSNR 是假设，encoder bottleneck 是更简洁的解释
2. **t=3.5 任意性**：midpoint 无物理剂量对应，无子步位置消融
3. **Stop-gradient 单向偏差**：只约束 full→half，half 可能更差
4. **CCT 负先验**：更强方法 ±0.015 dB，更弱版本为何能成功？
5. **零新意**：Richardson extrapolation + 应用
6. **Cherry-pick 风险**：3 个 schedule 自由参数 + 单 seed
7. **可证伪条件**：6 条具体 falsification criteria

---

## Reviewer-C 代码核查

| 项 | 结果 |
|---|---|
| Sub-stepping math | PASS |
| Stop-gradient | PASS |
| Backward compat | PASS |
| Loss integration | PASS |
| Silent failures | WARN（λ=0 前 5k 步） |
| Config resolution | PASS |
| Val averaging | **FAIL → 已修复**（`val_foc_*` 改用 hop0_count 平均） |

---

## 共识结论

FOC-lite 是**合理假设**，工程实现无重大问题，但**科学证据为零**。

**在投入训练 GPU 之前，必须先完成两个零成本诊断实验：**

### E1: 误差预算分解（eval only, ~1 GPU-hr）

脚本：`scripts/diagnose_error_budget.py`

测量 3 个量：
- **Decoder ceiling**: `PSNR(decode(z_gt), x_gt)` — decoder 的天花板
- **Transport gap**: `MSE(z_pred, z_gt)` in latent space
- **Off-manifold amplification**: `PSNR(decode(z_pred), decode(z_gt))` vs `PSNR(decode(z_pred), x_gt)`

**Gate**: ODE 积分误差占 hop0 gap ≥ 30%，否则 FOC-lite 应降低优先级

### E2: 预训练 FOC Gap 测量（eval only, ~0.5 GPU-hr）

脚本：`scripts/diagnose_foc_gap.py`

用现有 checkpoint 测量 `‖z_full − z_half‖`：
- **Gate**: gap > 5% of `‖z_full‖`，否则 ODE 误差不显著

---

## Exit FAIL 路径

| 优先级 | 实验 | 成本 | Pass 条件 |
|--------|------|------|-----------|
| P0 | E1 误差分解 | ~1 hr eval | ODE 误差占 hop0 gap ≥ 30% |
| P0 | E2 Gap 测量 | ~0.5 hr eval | gap/norm > 5% |
| P1 | E3 FOC 训练 2 seeds | ~48 hr | hop0 ≥ +0.15 dB, tail 不退化 |
| P2 | E4 λ 扫描 | ~96 hr | 单调响应 |
| P2 | E5 gap_norm 收敛 | 内含 E3 | 单调下降 |
