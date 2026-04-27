# V6 Plan Supervisor Final Review — 整合版

**日期**：2026-04-28
**作者**：Supervisor (Copilot)
**评审范围**：V6 plan + 我方初评 + 回复 + 回复深入分析 的整合复盘
**关联文档**：
- [`v6_plan_supervisor_review_20260427.md`](v6_plan_supervisor_review_20260427.md)（初评）
- [`to_supervisor_v6_response_20260427.md`](to_supervisor_v6_response_20260427.md)（团队回复）
- [`review/plan/transport_breakthrough_research_v6.md`](../../plan/transport_breakthrough_research_v6.md)（V6 plan）
**关联代码**：[`train_first_hop.py`](../../../train_first_hop.py)、[`pet_lr/rollout_first_hop.py`](../../../pet_lr/rollout_first_hop.py)
**关联实测**：[`v3_200k_transport_metrics_snapshot_20260427_1916.jsonl`](../logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl)
**关联 config**：
- V3 production：[`pet_flow_first_hop_224_200k_transport_v3.yaml`](../../../configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml)
- V5 rollout_heavy：[`pet_flow_first_hop_224_v5_rollout_heavy.yaml`](../../../configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml)
- V4 SF pilot：[`pet_flow_first_hop_224_v4_sf_pilot.yaml`](../../../configs/pet_flow/pet_flow_first_hop_224_v4_sf_pilot.yaml)

---

## 0. 整体结论

V6 plan 经过初评 → 回复 → 深入分析 → config 复核四轮论证后，**可以启动 200K from-scratch 训练**，**无硬阻塞项**。所有 Critical 级原始指控均已撤回或降级为 reactive 监控，唯一真正需要确认的事项是 V6 config 与 V3 200K production 的兼容性（保持 `velocity_rebalance.enabled: true`）。

### 三轮审议的最终判定

| 编号 | 初评指控 | 回复方论据 | 我方深入分析 | 最终判定 |
|:-:|---|---|---|:-:|
| #1 | image_aux 必须预设退火 | V3 img_raw 上升是 λ_img=0.12 高 + transport 弱的耦合后果 | 因果链成立但漏掉 VAE-floor 机制 | ⚠️ **降级为 reactive 监控**（+130K img_frac + latent-norm drift） |
| #2 | pair_weight=15 cold-start 不稳定 | AdamW 尺度不变性 + bias correction | 数学论证完全严密 | ✅ **撤回**（保留 +1K NaN 安全网） |
| #3 | §1.5 用错 V3 step 区间 | V6 raw loss 轨迹 ≠ V3，外推不可靠 | 归因正确，但留下循环论证 | ⚠️ **撤回 "算错"**，但要求 §5 双阈值 |
| #4 | Phase I 0-50K 过长 | 设计上让 velocity 精度先收敛 | 缺乏 GT-overfitting 量化反驳 | ⚠️ **保留**，pilot 后做 GT 噪声敏感性诊断 |
| #5 | step_weights[0]=0.5 与 §1.4 自相矛盾 | rollout[hop0] 是全步 endpoint，pair 是随机子段 | 经代码（`compute_pair_losses` L302-315）核实，回复正确 | ✅ **撤回** |
| #6 | `lambda_scale_mode: none` 与 alpha 解耦 | pilot 先用 none，60-80K 浪费现象出现再切 alpha | 两阶段验证策略合理 | ⏸ **暂搁置**，pilot 数据决定 |
| 新增 | velocity_rebalance 与 pair_weight 复合放大 | （初次提出后）核实 V3 production 一直 enabled=true，clip_max=4 | 我方初次推理基于错误的 clip_max=10 + 误把两者当独立放大复合 | ✅ **撤回**，V6 应继承 V3 设置 |

**最终阻塞项数：0。** 全部初评指控经辩证后无一保留为硬阻塞，但保留 4 个 reactive 监控点。

---

## 1. velocity_rebalance — 我方误判的复盘

这是这一轮审议中我方提出后又自行撤回的一项，单独说明。

### 1.1 V3/V4/V5 历史设置

| Config | enabled | clip_min | clip_max | 实验结果 |
|---|:-:|:-:|:-:|---|
| V3 200K production | **true** | 1.0 | **4.0** | best step 86800 |
| V4 SF pilot | false | — | — | 偏离实验，不作 baseline |
| V5 null_control | **true** | 1.0 | 4.0 | 恢复 V3 |
| V5 rollout_heavy | **true** | 1.0 | 4.0 | 恢复 V3 |

V3 best=86800 这个 V6 比对的核心 milestone，是在 rebalance ON、clip_max=4 下取得的。

### 1.2 我方初次推理的两个错误

**错误 A：误用代码默认 clip_max=10**

`compute_pair_losses` 第 347 行：

```python
clip_max = float(rebalance_cfg.get("clip_max", 10.0))
```

代码默认是 10，但所有 V3/V5 production config 都显式设 4。我方初次推理用了默认 10，于是得到"15 × 10 = 150× velocity"的灾难性数字。实际上限是 15 × 4 = 60×，且这只是把 V3 已经在用的 4× velocity 缩放整体放大 15×（pair_weight 的设计意图）。

**错误 B：把 pair_weight 与 rebalance 当作独立放大相乘**

完整等式（[`train_first_hop.py` L347-348, L1919](train_first_hop.py#L347)）：

```
total_pair  = (v_weight · scale_rebal) · L_vel  +  endpoint_weight · L_end
total_loss  = pair_weight · total_pair  +  λ_roll · L_roll  +  λ_img · L_img
```

- `scale_rebal` 调的是 pair **内部** V/E 比例
- `pair_weight` 调的是 pair **整体**相对 roll/img 的比例
- 两者**正交**，不在单一项上复合放大

V3 已经在 V:E ≈ 4:1 比例下成功，V6 保持同样 4:1（rebalance 配置不变）只是把整个 pair 通道权重提高 15 倍。

### 1.3 正确判定

| 决策 | 依据 |
|---|---|
| V6 config 保持 `velocity_rebalance.enabled: true`，clip_max=4 | 与 V3 production 完全一致 |
| 不引入"V6 关闭 rebalance"作为阻塞项 | 我方初次推理基于错误数值 |
| rebalance 的 `scale_rebal` 用 `.detach()`，不进入梯度图 | 行为可预测，不会 amplify gradient noise |

**教训**：评审涉及 config 行为时必须先核 production yaml，不能仅看代码默认值。

---

## 2. V3 实测数据复盘 — 设计判断的事实基础

V6 plan 的所有定量预测都建立在 V3 200K JSONL 的几个关键 step 上。以下是事实档案，便于后续对照：

### 2.1 V3 200K 关键 step 的原始 loss

| step | 阶段 | pair_raw | roll_raw | img_raw | pair_frac | roll_frac | img_frac | val_chain_normal_mse |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000   | 早期 | 1.01e-4 | 5.55e-5 | 4.30e-3 | 54% | 1% | 46% | — |
| 5000   | warmup 后 | 8.5e-5 | 1.2e-4 | 3.1e-3 | 45% | 8% | 47% | — |
| 86800  | **best** | 7.16e-5 | 6.90e-4 | 2.51e-3 | 13% | 32% | 55% | **0.000165** |
| 130000 | plateau | 1.11e-5 | 1.39e-4 | 5.35e-3 | 2% | 5% | 92% | 0.000172 |
| 150000 | plateau | 1.36e-5 | 1.38e-4 | 5.08e-3 | 2% | 5% | 93% | 0.000170 |
| 172600 | plateau | 1.48e-5 | 1.07e-4 | 9.40e-3 | 1% | 2% | 96% | 0.000169 |

### 2.2 两段轨迹的物理含义

**Phase I（0-86800）：transport-active**
- pair_raw 缓慢降至 7e-5，rollout_raw 增至 7e-4
- img_raw 维持 ~2.5e-3（VAE-floor 附近）
- val_mse 单调下降到 0.000165（best）

**Phase II（86800-200000）：image-dominated plateau**
- pair_raw 进一步降至 1.5e-5（**4.8× 衰减**）
- rollout_raw 也降至 1e-4（**7× 衰减**）
- img_raw 反向上升至 9.4e-3（**3.7× 上升**）
- val_mse 几乎不动（86800 的 0.000165 → 200000 的 0.000169）

### 2.3 V6 的核心赌注

V6 押的就是：通过 pair_weight=15 + λ_roll=4 让 transport 通道在整个 200K 中保持类似 step 86800 的活跃状态，避免坠入 Phase II plateau。

如果 V6 成功，预测的 raw loss 轨迹应该接近 step 86800 而非 step 172600；如果 V6 失败，会复现 V3 Phase II。两种轨迹下 V6 的 fraction 预测完全不同，这是 §1.5 算"对"算"错"争论的根源——本质上 §1.5 是 conditional prediction，未明确条件。

### 2.4 plan §1.5 的修订建议

把 §1.5 改写为**双场景预测表**：

```markdown
| 场景 | 假设 V6 raw loss | pair_frac | roll_frac | img_frac |
|---|---|---:|---:|---:|
| 成功（接近 V3 step 86800） | pair=7e-5, roll=7e-4, img=2.5e-3 | 28% | 70% | 2.6% |
| 失败（接近 V3 step 172600） | pair=1.5e-5, roll=1e-4, img=9.4e-3 | 22% | 41% | 37% |
```

并把 §5 的"transport ≥ 75%"改为：
- **主判据**：val_chain_normal_mse < 0.000150（绝对 metric）
- **辅助判据**：transport ≥ 60%（保守红线，触发 reactive image_aux 退火）

这样避免 fraction 预测变成自我验证的循环论证。

---

## 3. 4 个 reactive 监控点

最终保留以下 reactive 检查，不阻塞启动，但训练中必须执行：

### 3.1 +1K — cold-start 安全网

**目标**：确认 pair_weight=15 没有让 AdamW 行为偏离尺度不变性区域

**指标**：
- pair_loss < 5e-4（绝对值）
- 无 NaN / Inf
- velocity head 输出 norm 在 [0.001, 0.1] 范围（log-scale 监控）

**触发动作**：abort 训练，回滚到 pair_weight ramp（1→15 over 0-10K）

**预期**：不会触发（AdamW 尺度不变性论证）

### 3.2 +50K — Phase I 收敛检查

**目标**：确认 GT-only velocity 学习收敛，且未过拟合到 GT segment 几何

**指标**：
- pair_frac > 70%（plan §4 已要求）
- pair_raw 稳定在 ~5e-5 ~ 1e-4
- **新增**：在 GT segment 输入上加 N(0, 0.001) 噪声，pair_loss 增加 < 3×

**触发动作**：若噪声敏感度 > 3×，下次实验把 warmup_ratio 从 0.25 降到 0.10

### 3.3 +130K — image_aux 失控检测

**目标**：捕获 V3 Phase II plateau 复现的早期信号

**指标**：
- img_frac < 30%
- img_raw 不出现连续 30K step 的单调上升
- **新增（应对 VAE-floor 盲区）**：z_pred_NORMAL 的 mean / std 与 z_GT_NORMAL 分布的 KL 散度 < 0.1

**触发动作**：reactive 切换 λ_img 0.04 → 0.01（通过 SIGNAL_FILE 机制）

### 3.4 +150K — Go/No-Go 主判定

**目标**：决定是否完成 200K 全程训练

**指标**：
- val_chain_normal_mse 趋势改善（vs +100K snapshot）
- transport gradient ≥ 60%
- 无 reactive 退火事件

**触发动作**：
- 全部满足 → 完成 200K
- val 趋势恶化 → abort，分析 +130K 监控数据

---

## 4. V6 设计的我方最终评价

### 4.1 设计强项

1. **正确诊断 V3 失败原因**：V3 的 86800 plateau 是 image-dominated 正反馈的产物，不是 transport 收敛能力上限。V6 用 pair_weight=15 切断这个正反馈起点，方向正确。

2. **通道-跳分离加权设计在数学上 well-founded**：
   - pair_loss_weights=[2.5, 1.0, 1.0, 1.0] 头重，对应 hop0 v_std=0.0096 的 GT-input velocity 独占信号
   - step_weights=[0.5, 2.0, 1.5, 1.0] 中重，押 ExpoGap × v_std 的 hop1 sweet spot
   - 两者在不同通道的不同跳上发力，无重复

3. **AdamW 尺度不变性的运用**：pair_weight=15 直接启用而非 ramp，依赖 Adam bias correction + 尺度不变性，省去 ramp 复杂度，是理论严密的简化。

### 4.2 设计弱项

1. **§1.5 fraction 预测含循环论证**：把"V6 假设成功"作为 fraction 预测前提，又用同样的 fraction 阈值作为成功标准。建议改为绝对 val metric 主判据。

2. **Phase I 50K GT-only 缺少量化论证**：相比 V3 的 5K-20K，延长到 50K 是 2.5× 扩展，叠加 pair_weight=15 后 GT-overfit 风险存在，但 plan 没给量化评估。

3. **image_aux 退火问题留下了 reactive 处理**：理论分析支持"V6 成功 → img_raw 不会失控"，但 VAE-floor 机制不是 transport-vs-image 的零和博弈，可能独立产生 img_raw 漂移。这部分必须用 +130K 的 latent-norm drift 监控兜底。

### 4.3 总体评价

V6 是**经过深度论证、设计内部一致、有明确赌注与 fallback 监控的方案**。比 V3/V5 的迭代有清晰理论进步（pair_weight 机制的引入 + AdamW 尺度不变性的运用 + 通道-跳分离加权）。三轮审议过程也证明设计经得起质疑。

**Supervisor 决议：可启动 V6 200K from-scratch 训练。**

---

## 5. 启动前 Checklist

| 项目 | 状态 | 行动 |
|---|:-:|---|
| `loss.pair_weight` 代码实现 | ✅ 已完成（团队回复 §7） | — |
| V6 config 创建 | ⏸ 未确认 | 从 V3 200K config 派生，仅改 plan §3 列出的 9 个参数 |
| V6 config 保持 `velocity_rebalance.enabled: true, clip_max=4` | ⏸ **必须确认** | 与 V3 production 一致 |
| V6 config 列出 align/foc/regularizer 配置（继承 V3） | ⏸ 待 plan §1.5 附录补全 | — |
| §1.5 改为双场景预测 + §5 改为 val 主判据 | ⏸ 建议修订 | 不阻塞启动，可在 pilot 期间补 |
| Dry-run 50-200 步验证日志 | ⏸ 待执行 | — |
| Reactive 监控 hook（+1K/+50K/+130K/+150K） | ⏸ 待集成 | 沿用 V5 已有的 SIGNAL_FILE 机制 |

启动前**必须确认**的只有第 3 项（rebalance config）。其他都是 nice-to-have 或 pilot 后修订。

---

## 6. 给后续 milestones 的建议

V6 之后无论成败，几个可验证的科学问题值得记录：

1. **AdamW 尺度不变性在多 loss 训练中的边界条件**：V6 是首次在 PET 项目中用大 loss-weight 倍数（15×）启动 from scratch。结果（无论成败）应当用作其他通道（image_aux λ、roll λ）将来调整的参考。

2. **VAE-floor 机制的隔离实验**：如果 V6 +130K 出现 img_raw 失控但 transport 健康，那是 VAE-floor 机制的实证；如果两者同步退化，仍是正反馈。区分这两种情况需要单独跑一个"transport-only no-image"的对照。

3. **velocity_rebalance 的必要性回归测试**：V4 SF pilot 是唯一关闭 rebalance 的实验，但被其他变量干扰未单独评估。可在 V6 之后做一个 V6+rebalance OFF 的 50K pilot，独立验证 rebalance 在 V6 配方下是否仍必要。

---

## 7. 修订记录

| 版本 | 日期 | 主要修订 |
|---|---|---|
| 初评 v1 | 2026-04-27 | 提出 6 项指控（3 Critical + 3 Important） |
| 回复 v1 | 2026-04-27 | 团队针对 6 项逐一回应 |
| 深入分析 v1 | 2026-04-27 | 评估回复有效性，新增"velocity_rebalance 阻塞" |
| **本文档（终评）** | 2026-04-28 | 撤回 velocity_rebalance 阻塞（误判），整合所有撤回/降级判定 |
