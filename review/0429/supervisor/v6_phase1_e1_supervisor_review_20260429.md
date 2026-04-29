# Supervisor 0429 — E1 误差预算 + V6 Phase I 状态裁决

**日期**：2026-04-29
**角色**：Supervisor
**评审范围**：
- [`review/0429/e1_error_budget_decomposition_explained.md`](../e1_error_budget_decomposition_explained.md)（Author 修订版）
- [`review/0429/e1_core_hypothesis_viewer_assessment_20260429.md`](../e1_core_hypothesis_viewer_assessment_20260429.md)（Reviewer assessment）
- V6 Phase I 训练状态（step 38050，[`v6_transport_first_gpu1_train_snapshot_20260429_121604.log`](../../0428/operator/log_snapshots/v6_transport_first_gpu1_train_snapshot_20260429_121604.log)）
- V5 null-control 训练状态（step 123550，[`v5_null_control_gpu3_train_snapshot_20260429_121604.log`](../../0428/operator/log_snapshots/v5_null_control_gpu3_train_snapshot_20260429_121604.log)）
**关联 supervisor 历史**：
- [`0427/supervisor/v6_plan_supervisor_review_20260427.md`](../../0427/supervisor/v6_plan_supervisor_review_20260427.md)
- [`0427/supervisor/v6_plan_supervisor_final_review_20260428.md`](../../0427/supervisor/v6_plan_supervisor_final_review_20260428.md)
- [`0428/supervisor/v5_attribution_supervisor_verdict_20260428.md`](../../0428/supervisor/v5_attribution_supervisor_verdict_20260428.md)
- [`0428/supervisor/v5_v6_supervisor_addendum_20260429.md`](../../0428/supervisor/v5_v6_supervisor_addendum_20260429.md)

---

## 0. 总结论

| 议题 | 0429 状态 | 是否阻塞 |
|---|---|:-:|
| E1 误差预算分解（修订版）的方法论严谨性 | **通过** — author 修订版正确加入 PSNR-vs-MSE 域区分，reviewer 边界条件合理 | 否 |
| V6 Phase I "chain quality 落后 V3 同期" 的因果归因 | **已闭环** — 不是 V6 设计 bug，是 V3 同期数字本身就含 chain 子任务监督 | 否 |
| V6 50K-75K 关键 Gate 监控 plan | **已落地**（0429 addendum）+ 新增 V7 fallback 候选 | 否 |
| V5 attribution full-val + Path A 闭环 | 仍 pending | 是（仅对 V6 200K full 完成判定） |
| Decoder 优化在 V6 milestone 是否需要触发 | **否** — E1 显示 decoder floor 占 MSE 比例 < 13% | 否 |

**Supervisor 整体判断**：当前文档 + 训练状态构成**一致的科学叙事**：
1. **方向**：transport 是当前主瓶颈（E1 已 close）
2. **诊断**：V6 chain 落后是 strict λ_roll=0 设计的预期副作用，不是 bug
3. **赌注**：V6 Phase II（50K-150K）能否快速建立 chain composition 能力 — **未验证**
4. **fallback**：若 65K 时 chain 无明显回收，启用 V3-style λ_roll floor=0.002 作为热修

V6 pilot 继续推进，无新增阻塞项。

---

## 1. E1 误差预算文档评审

### 1.1 Author 修订版的正面修正

Author 0429 修订版对 0416 原版的两个改动是关键的：

| 0416 原版问题 | 0429 修订版修复 |
|---|---|
| 把 dB 域占比当作"误差贡献占比"（"transport 占 96%"读作能量分解） | 显式声明这是 **PSNR-domain diagnostic decomposition**，不是 MSE-domain linear contribution |
| `psnr_pred_gt = decoder(z_pred) vs decoder(z_GT)` 被读作"纯 transport 误差" | 明确这是 **decoder-mediated transport discrepancy**（含 decoder Jacobian），不是 latent-space transport error |

加上 §5 的 MSE 比值表（13-39× E2E vs ceiling, 1.04-1.13× e2e vs pred_gt）作为 **严格定量陈述**，文档的方法论现已严谨。

### 1.2 Reviewer 的边界条件合理且可执行

Reviewer §6 caveats 表列了 5 项关键限制，全部成立。其中两项最关键：

- "dB-gap ratio is not MSE-linear contribution" — 与 author §0 的免责声明对应
- "psnr_pred_gt is decoder-mediated" — author §2.2 已用 Jacobian 一阶展开承认这点

Reviewer §7 的 "use this / avoid this" 措辞对照表是有效的工程化指引。**Supervisor 采纳 reviewer §7 措辞，作为后续 PR/paper draft 的标准表达**。

### 1.3 一处轻微遗漏

Author §6 提到 latent-space MSE，但仅作"补充"。从 V6 当前状况看，这其实是**关键诊断**：

V6 step 38000 的 val 数据：
- `val_pair_total = 0.000002`（latent-space pair MSE，已收敛）
- `val_chain_normal_mse = 0.000975`（latent-space chain MSE，远未收敛）

这两个 latent-space 数字本身就足以诊断 V6 Phase I 的 pair-vs-chain 不对称，**不需要**经过 decoder 放大才能看出问题。Author §6 应把 latent_mse 提升为 §3 等级的"主要诊断量"，而不是补充。

**修订建议（不阻塞）**：Author 在 §6 标题下加一句：

> latent-space MSE 是不依赖 decoder 的直接 transport 诊断量。在 V6 Phase I 这种 pair 已饱和但 chain 未训练的状态下，latent-space `val_pair_total` 与 `val_chain_normal_mse` 的差距比 PSNR 域分解更能直接说明问题。

### 1.4 Reviewer Finding 3 的关键提醒

> "E1 says 'transport-side decoded discrepancy dominates'; it does not by itself say 'FOC/integration is the cause' or 'pair_weight alone fixes it.'"

这点对**V6 plan §1.2** 的措辞有实际约束：V6 plan 说 "V3 200K 训练数据证明 rollout 梯度占比高（32%）时模型达到 best" — 这个陈述把 E1 + V3 JSONL 两个独立证据链打包，但 E1 并不能单独支持 "rollout fraction" 与 "best chain quality" 之间的因果。

V6 plan 的逻辑严格说是：
- E1 → "transport 是瓶颈"（已 close）
- V3 JSONL → "rollout 通道梯度占比高时 best 出现"（相关性证据）
- V6 hypothesis → "提高 rollout 梯度占比 + 削弱 image 通道 → 突破 best"

E1 只支持第一条，第二条是相关性观察，第三条是 V6 自己的赌注。**Supervisor 采纳 reviewer Finding 3**，要求后续 V6 中期总结明确区分这三层论证强度。

---

## 2. V6 Phase I 状态归因（与之前 supervisor 文档的连贯性）

0428 supervisor addendum 把 V6 chain 退化标记为"新增风险"。0429 的进一步分析（用户与 Copilot 的对话）已把这个"风险"**降级为"预期副作用"**。

### 2.1 V6 chain 落后 V3 同期的根本原因

V3 同期（step 1000 起）λ_roll = 0.002 持续在监督 chain composition；V6 严格 λ_roll = 0 把这个监督完全切断。

这意味着 V3 step 38K 的 chain ≈ 0.0005 数字 **不是 GT-only 训练的产物**，而是 38K 步 chain 子任务监督的累积。V6 step 38K 的 chain ≈ 0.00097 是 0 步 chain 子任务监督的产物。

**两者在不同任务上对比，结果不可直接比较。**

### 2.2 V6 当前状态的二维诊断

| 维度 | V6 step 38K | V3 step 38K（同期）| V3 best (86800) | 评价 |
|---|---:|---:|---:|---|
| pair velocity 拟合精度 | 0.000002 | 0.000087 | 待 sync | V6 **强 40×** |
| chain composition 质量 | 0.000975 | ~0.000470 | 0.000165 | V6 **弱 2.07×** |
| pair_frac | 0.78 均值 | 0.28 均值 | 0.13 | V6 transport-led ✓ |
| img_frac | 0.22 均值 | 0.71 均值 | 0.55 | V6 抑制了 image domination ✓ |

V6 实现了"训练压力的重分配"目标；但还没实现"端到端 chain quality 改善"目标，因为后者依赖 50K 后才启动的 rollout 监督。

### 2.3 从 E1 框架看 V6 当前状态

把 E1 §5 的 MSE 比值套用到 V6 训练中（latent-space 类比）：

V6 step 38K：
- pair MSE / pair MSE_ceil（V3 同期 baseline）≈ 0.000002 / 0.000087 = **0.023×**（V6 已优化到 V3 同期 baseline 的 1/43）
- chain MSE / chain MSE_baseline（V3 同期）≈ 0.000975 / 0.000470 = **2.07×**（V6 chain 比 V3 同期差 2 倍）

V6 把"pair 子任务"过度拟合，"chain 子任务"还没开始。这与 E1 揭示的 "transport-related decoded discrepancy dominates"完全一致 — V6 当前的 chain 子任务正是 E1 中"被 decoder 放大成 13-39× MSE ratio"的那个根源。

**V6 Phase II 的 rollout ramp 起作用时**，应当看到：
1. chain MSE 在 latent space 直接下降（无需经 decoder）
2. 经 decoder 后 PSNR 提升量级与 E1 的 gap_transport 对应

如果 50K-75K 内 chain MSE 不显著下降，说明 V6 假设"strict λ_roll=0 + 大 pair_weight"配方能后期快速建立 chain 是错的。

---

## 3. V7 Fallback 候选的预先确认

0429 addendum 提了两个 V7 fallback 候选；这里 supervisor 进一步评估。

### 3.1 候选 A：恢复 V3-style λ_roll floor = 0.002

```yaml
rollout:
  lambda_start: 0.002    # V3 floor，非零
  lambda_end: 4.0
  warmup_ratio: 0.25
```

**论据**：V3 同期 chain quality 实证显示 0.002 这个非零 floor 即可建立 chain composition 能力（即使数值小，AdamW 尺度不变性让方向信号进入梯度）。

**风险**：在 V6 的 pair_weight=15 配方下，0.002 × roll_raw 可能被 15 × pair_raw 完全淹没，等价于"几乎仍是 strict 0"。需要量化：

V3 step 38K 时：pair_raw ≈ 1.5e-5（推断），roll_raw ≈ 6e-4（推断）
- V3 weighted: 1.0 × 1.5e-5 = 1.5e-5（pair）, 0.002 × 6e-4 = 1.2e-6（roll），比例 12:1
- V6 weighted（如 floor=0.002）: 15 × 1.5e-5 = 2.25e-4（pair）, 0.002 × 6e-4 = 1.2e-6（roll），比例 187:1

V6 配方下 0.002 floor 可能不够。建议**候选 A 改为 λ_roll floor = 0.05**：
- V6 weighted: pair = 2.25e-4, roll = 0.05 × 6e-4 = 3e-5，比例 7.5:1（与 V3 同期 12:1 同量级）

### 3.2 候选 B：缩短 warmup_ratio 到 0.10

```yaml
rollout:
  warmup_ratio: 0.10     # 50K → 20K
  ramp_ratio: 0.65
```

**论据**：把 chain 子任务的"零监督期"从 50K 缩短到 20K，减少 backbone 在 pair-only manifold 上累积偏移的时间。

**风险**：早期 pair 拟合时间被压缩，可能影响 hop0 GT-input velocity 的学习深度。但 V6 当前 step 14K 时 val_pair_total 已经降到 ~5e-6 量级，说明 pair 拟合 14K 步就能进入 saturation 区间，warmup_ratio=0.10（20K）已足够。

### 3.3 优先级

| 候选 | 修复成本 | 预期效果 | Supervisor 推荐顺序 |
|---|---|---|:-:|
| A（floor=0.05） | 1 行 yaml | 让 chain 子任务从 step 1 就有有效梯度信号，量级匹配 V3 | **首选** |
| B（warmup=0.10） | 1 行 yaml | 减少 chain-blind 期 30K 步 | 备选 |
| A + B | 2 行 yaml | 双重防护 | 仅在单独 A 或 B 都失败时考虑 |

**重要**：候选 A/B 是**V6 失败后的 V7 设计**，不是 V6 当前训练的临时干预。当前 V6 训练**继续按 plan 推进**，等到 step 65K 决策点再判断是否启用 V7。

### 3.4 临时干预 vs V7 的决策树

```
step 50K  : V6 进入 Phase II，λ_roll 开始 ramp
step 60K  : 第一次 chain quality 检查
            ├── chain < 0.0007 → 继续按 plan
            └── chain ≥ 0.0007 → 提前到 step 65K 决策
step 65K  : 关键 Gate
            ├── chain < 0.0005 → V6 设计成立，继续到 200K
            ├── 0.0005 ≤ chain < 0.0008 → 边缘可接受，加密监控到 75K
            └── chain ≥ 0.0008 → V6 失败，停止训练
                                  └── V7 启动（候选 A，floor=0.05）
```

**临时干预（边训练边改 λ_roll floor）不被推荐** — 会污染 V6 vs V7 的对照实验设计。失败应当干净失败，重新启动 V7。

---

## 4. 与之前 supervisor 文档的连贯性核查

| 之前 supervisor 立场 | 0429 状态 | 是否需要修订 |
|---|---|:-:|
| 0427 final review：V6 可启动 | ✅ V6 已启动并推进到 step 38K | — |
| 0427 final review：必须确认 velocity_rebalance.enabled=true | ✅ V6 config 已确认（继承 V3 200K） | — |
| 0428 verdict：V5 attribution author 措辞需降级 | ✅ Reviewer 0429 update 重申，author 待修订 | — |
| 0428 verdict：V6 200K 放行延至 +50K 决策点 | ⚠️ 0429 addendum 已扩展为多 Gate 决策（60K/65K/75K）| 已生效 |
| 0428 addendum：chain quality 退化是"新增风险" | ✅ **降级为"预期副作用"**（0429 用户对话+本文档）| 立场更新 |

**关键立场迁移**：
- 0428 addendum 把 V6 chain 退化标记为风险信号
- 0429 经因果分析确认这是 strict λ_roll=0 设计的预期副作用
- 风险**没消失**，只是性质从"V6 现在就有 bug"改为"V6 假设是否成立要等 Phase II 验证"

这个迁移不是 supervisor 立场反复，而是**证据增量带来的诊断精度提升**。

---

## 5. 当前阻塞清单（更新版）

| 阻塞项 | 状态 | 阻塞对象 |
|---|:-:|---|
| velocity_rebalance config 确认 | ✅ done | V6 启动 |
| V6 Phase I 启动 | ✅ done | — |
| V6 60K/65K/75K chain 监控 | ⏸ 待执行 | V6 Phase II 决策 |
| NC 训到 +50K matched budget (~136K) | 🔄 73% 进度 | V5 归因对照 |
| sync V5 rollout-heavy + NC full-val JSON | ⏸ pending | V6 100K mid-run 决策 |
| Path A diagnostic | ⏸ pending | V5 因果机制 close |
| V6 200K full 完成判定 | ⏸ 远期 | 项目交付 |
| Author E1 文档 §6 latent_mse 提级建议 | 🟢 可异步 | 无阻塞 |
| Author/Reviewer V5 attribution 措辞统一 | 🟢 可异步 | 无阻塞 |

**当前无新增阻塞项**。V6 训练继续，按 0429 addendum 的 Gate plan 在 50K-75K 监控。

---

## 6. 给三方的指令（0429 更新）

### 6.1 给 Author

- E1 文档 §6 latent_mse 部分**建议提级**为主要诊断量（章节顺序提到 §3 之前）
- V5 attribution 文档采纳 reviewer §7 措辞对照表
- V6 plan 后续若被引用，注意 reviewer Finding 3 的三层论证强度区分

### 6.2 给 Reviewer

- E1 assessment §6 caveats 已采纳，**作为项目标准 caveat 列表**用于后续 V6/V7 文档
- V5/V6 拆分文档结构正确，建议保持
- 下一份 viewer update 建议追加 60K-75K chain trend curve（图或表）

### 6.3 给 Operator

- V6 50K-75K 期间，**每 5K snapshot** 必须执行（不是每 10K）
- 每个 snapshot 输出三项关键指标：`val_pair_total`、`val_chain_normal_mse`、`val_chain_tail_mse`
- NC 不要在 +50K matched budget（step ~136K）之前停止，不要被 73% 进度误判为已完成
- V5 full-val sync 应在 V6 step 75K 前完成（约 7-10 天后）

### 6.4 V6 团队 — V7 预案

- **不要**做临时干预（不要边训练边改 λ_roll floor）
- 若 V6 在 step 65K 失败，按候选 A（λ_roll floor=0.05）启动 V7 from scratch
- V7 与 V6 仅一个变量差异（λ_roll start），可作为干净的 ablation 实验

---

## 7. 修订记录

| 日期 | supervisor 文档 | 主要修订 |
|---|---|---|
| 0427 | initial review | 6 项指控 |
| 0427 | final review | 全部撤回/降级；启动 checklist |
| 0428 | V5 attribution verdict | 措辞降级、5 项分歧仲裁 |
| 0429 | V5/V6 addendum | V6 chain 退化作为新风险 |
| **0429 本文档** | E1 + V6 Phase I 裁决 | E1 方法论通过；V6 chain 退化降级为"预期副作用"；V7 候选 A 量级修正（0.002→0.05）|
