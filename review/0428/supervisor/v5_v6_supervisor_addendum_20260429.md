# Supervisor 0429 Addendum — V5 归因 + V6 Phase I 状态更新

**日期**：2026-04-29
**角色**：Supervisor
**性质**：对 [`v5_attribution_supervisor_verdict_20260428.md`](v5_attribution_supervisor_verdict_20260428.md) 的增量更新（不替换原文）
**触发**：
- null-control 训练推进到 step 123550（+22.7K 新步）
- V6 Phase I 推进到 step 38050（接近 50K 分歧点）
- Reviewer 拆分出 [`v6_phase1_viewer_analysis_20260429.md`](../reviewer/v6_phase1_viewer_analysis_20260429.md) 并更新 [`v5_attribution_viewer_update_20260429.md`](../reviewer/v5_attribution_viewer_update_20260429.md)

**关联日志**：
- [`v5_null_control_gpu3_train_snapshot_20260429_121604.log`](../operator/log_snapshots/v5_null_control_gpu3_train_snapshot_20260429_121604.log)
- [`v6_transport_first_gpu1_train_snapshot_20260429_121604.log`](../operator/log_snapshots/v6_transport_first_gpu1_train_snapshot_20260429_121604.log)

---

## 0. 是否需要更新 0428 verdict

**结论**：**部分更新**——V5 归因部分裁决保留，新增一个 V6 Phase I 风险议题。

| 议题 | 0428 verdict 状态 | 0429 更新 |
|---|---|---|
| V5 归因结论强度 | author 需降级措辞 | ✅ 保持。NC 数据从 14K→36.7K 步只是**强化**原方向，不改变结论 |
| V5 full-val + Path A 闭环 | pending | ✅ 仍 pending，未 sync |
| V6 pilot 是否阻塞 | 不阻塞 | ⚠️ **新增风险**：V6 Phase I chain quality 退化，需 +50K Gate 加严 |
| V6 200K full 放行时机 | 延至 +50K 分歧点 | ✅ 仍延至 +50K，但**判据需扩展**（不只 pair_frac，还要 chain quality 恢复信号） |

**核心新发现（0428 verdict 未覆盖）**：V6 Phase I 后段 `val_chain_normal_mse` 显著退化，最新 step 38000 达到 0.000975，是 V3 同期（0.000470）的 2.07× 和 V3 best（0.000165）的 5.9×。这是 0428 时点尚未浮现的信号。

---

## 1. V5 Attribution 增量证据

### 1.1 NC 训练状态对比

| 维度 | 0428 verdict 时 | 0429 现状 |
|---|---|---|
| NC 最新 step | 100850 | **123550** |
| 相对 V3 best (86800) 的新步数 | +14050 | **+36750** |
| 占 +50K matched budget 进度 | 28% | **73%** |
| pair_frac 范围 | 0.9-3.2% | 0.8-7.2% |
| roll_frac 范围 | 4-9.7% | 5.3-18.0% |
| img_frac 范围 | 87-94% | 74.8-93.9% |
| rolling-val 范围 | 0.000159-0.000382 | 0.000159-0.000362 |

### 1.2 一个有意思的新观察

NC 最新 step 123550 单行：`pair_frac=0.072, roll_frac=0.180, img_frac=0.748`，transport 总计 25.2%，比之前任何窗口都高。

这是**单 batch 噪声还是 trend 起点**？目前数据不足判断：
- 之前 step 105000 也出现过 16% roll_frac，但下一行回到 6%
- 仅凭单行不能改变"主要 image-dominated"的总体描述

但提示一个潜在替代解释：V3 recipe 在 EMA + 长期训练后，**晚期可能存在 transport pressure 缓慢上升的现象**（不是 plateau 永恒）。这会削弱"V3 plateau 是权重配方病"的强 claim 支持度——如果 NC 继续训到 +50K matched budget 时 transport 能爬到 30%+ 且 rolling-val 改善，那 V3 之前的 plateau 可能只是 LR cosine 末段的偶然现象，而非 recipe 病。

### 1.3 Reviewer 0429 的措辞调整

reviewer 把 0428 的 "always 90% image" 修正为 "mostly image-dominated, with late batch-level transport-pressure spikes"。这个修正合理。

**Supervisor 立场**：保留 0428 verdict 的所有裁决（author 措辞降级、full-val + Path A 仍是闭环阻塞），但**追加一条**：

> NC 在 step 105K-123K 区间出现 transport-pressure 上升的零星 batch（最高 25.2%），需要 NC 训到 +50K matched budget 后做完整窗口统计，确认这是噪声还是 trend。如果是 trend，"V3 plateau 仅是 recipe 病"的 claim 强度需进一步降级。

---

## 2. V6 Phase I 新议题（0428 verdict 未覆盖）

### 2.1 关键事实

V6 当前 step 38050（Phase I 76% 进度，距 50K 分歧点 12K 步）：

| 指标 | V6 当前 | V3 同期（≤40K） | V3 best (86800) | 评价 |
|---|---:|---:|---:|---|
| pair_frac 均值 | 0.775 | 0.280 | 0.13 | ✓ V6 实现了 transport-first |
| img_frac 均值 | 0.225 | 0.711 | 0.55 | ✓ V6 抑制了 image domination |
| val_pair_total（晚期） | 0.000002 | 0.000087 | n/a | ✓ pair 任务拟合 |
| **val_chain_normal_mse（晚期）** | **0.000864-0.000975** | 0.000470 | 0.000165 | ✗ **chain 质量显著退化** |

V6 的 pair supervision 拟合得"过于成功"（val_pair_total 比 V3 同期低 30×），但 open-loop chain 推理质量比 V3 同期还差 1.84×，比 V3 best 差 5.9×。

### 2.2 退化的物理机制（候选）

V6 Phase I 设计上 `lambda_roll=0, alpha=0`，chain 监督完全缺席。可能机制：

1. **几何折叠**：pair velocity 过度拟合 GT segment 上的随机 t 子段，hop 之间的 endpoint 一致性没有任何约束，导致 4-hop chain composition 时误差累积放大
2. **velocity scale shrinkage**：v_hop_abs 在 V6 step 38050 = 5.16e-4，远小于 GT v_std=9.6e-3（hop0），velocity 输出可能被 endpoint loss 项主导而 collapse 到 endpoint-direct 模式
3. **VAE manifold 漂移**：transport-first 把 latent 推到了 pair-loss 最优但 VAE decoder 失真大的区域

机制 1 是最可能的——这正是为什么 V6 设计 50K-150K alpha ramp（让 rollout 进入纠正 chain composition），但 Phase I 后段 chain metric **越拉越差**意味着到 50K 时进入 Phase II 的起点已经偏离 V3 同期 baseline 不少。

### 2.3 风险评估

**乐观情景**：rollout/alpha ramp 在 50K-75K 启动后，chain composition 开始受监督，val_chain_normal_mse 快速回收到 V3 同期水平甚至以下。这是 V6 设计的预期。

**悲观情景**：Phase I 退化已让模型陷入"velocity 拟合好但 chain composition 错乱"的局部 optima，rollout 监督只能轻微纠正，无法抵消 50K 步的累积偏离。

**中性情景**：Phase II ramp 起作用，但慢；到 100K-150K 才能追平 V3 同期，整体 200K 收益边际。

当前数据**无法区分**这三种情景，必须等 60K-75K 窗口的 rolling-val 趋势。

### 2.4 0428 verdict 中"V6 +50K Phase II 分歧点"判据需要扩展

0428 verdict §3.4 中的 "+50K 分歧点" 决策矩阵需要补充：

| 条件 | 0428 verdict | 0429 updated |
|---|---|---|
| 主判据 | V6 内生 metric + V5 归因闭环 | **追加：50K-75K 窗口 val_chain_normal_mse 是否从 ~0.0009 回收到 ≤ 0.0005** |
| 次判据 | pair_frac 维持 transport-led | 保留 |
| 新增 | — | **chain quality 趋势监控**：60K/65K/70K/75K 四个 5K 间隔的 val_chain_normal_mse 是否单调下降 |

如果 75K 时 val_chain_normal_mse 仍 > 0.0005，应**停止训练并复盘 Phase I 设计**（warmup_ratio=0.25 是否过长、step_weights[0]=0.5 是否需要调高、image_aux 是否在 Phase I 应该更高以维持 latent manifold 锚定）。

---

## 3. 双 reviewer 文档拆分的方法论评价

reviewer 把 V5 attribution 和 V6 Phase I 拆成独立文档是正确选择：

- V5 attribution 是 V3-recipe 复盘 + null-control 对照实验
- V6 Phase I 是 from-scratch 独立实验

把两者放一份会导致一个文档承载两套独立的因果模型和决策矩阵。拆分后：
- V5 文档可继续等 full-val + Path A 闭环
- V6 文档可独立按 Phase Gate 推进

**Supervisor 采纳此拆分**，0428 verdict 的"V6 200K 全程放行延至 +50K"决策**不再绑定 V5 attribution closure**，改为：

| 决策 | 依据 |
|---|---|
| V6 +50K Phase II 进入决策 | 仅看 V6 自身（pair_frac + chain quality 趋势 + 无 NaN） |
| V6 100K mid-run 决策 | V6 Phase II 表现 + V5 归因 closure（如已 sync） |
| V6 200K full 完成判定 | V6 全程 + V5/null-control 同 budget full-val 对照 |

这种解耦让 V6 决策不被 V5 操作进度阻塞。

---

## 4. 修订后的优先级

| 优先级 | 任务 | 阻塞 | 状态 |
|:-:|---|:-:|:-:|
| 🔴 P0 (新增) | V6 步 50K-75K rolling val_chain_normal_mse 监控（每 5K snapshot） | V6 Phase II 是否继续 | 当前 step 38K, ~12K 步后到 50K |
| 🔴 P1 | NC 继续训到 +50K matched budget (step ~136800) | V5 归因对照 | 73% 完成，~13K 步后达成 |
| 🟡 P2 | sync V5 rollout-heavy + null-control full-val JSON | V5 归因 closure | pending |
| 🟡 P3 | 执行 Path A diagnostic | V5 因果机制 | pending |
| 🟢 P4 | author 文档采纳 reviewer 措辞降级表 | 文档质量 | 可异步 |

**新优先级 P0 的提出**：0428 verdict 时 V6 才 step ~7K，chain quality 数据不足；0429 已有 38K 数据显示 chain 退化，必须把 50K-75K 的 chain 趋势监控提级为最高优先级。

---

## 5. 给三方的更新指令

### 5.1 给 Author

- V5 归因部分：保持 0428 supervisor 给出的措辞降级（reviewer §4 表）
- V6 Phase I：**新增**章节描述 chain quality 退化的事实和 50K-75K 监控 plan
- 不要把 V6 Phase I chain 退化解读为"V6 失败"——这是 Phase I 设计的内在副作用，等 Phase II 验证

### 5.2 给 Reviewer

- V5/V6 文档拆分正确，建议 0429 之后保持这种结构
- V6 Phase I 分析很到位，特别是"pair fitting 但 chain quality 差"这个二维评价框架
- 建议下一次更新追加：60K/65K/70K/75K 这四个间隔的 chain quality trend curve（图或表），便于一眼判断

### 5.3 给 Operator

- V6 50K-75K 期间，每 5K 必须输出一次 rolling-val snapshot
- NC 不要在 +50K matched budget 之前停（操作风险点：当前 73% 进度容易被误判为已完成）
- V5 full-val sync 是 P2，但应在 V6 step 75K 前完成，否则 100K mid-run 决策缺数据

---

## 6. 关于 0428 supervisor verdict 是否被 0429 推翻

**没有**。0428 verdict 的所有结论在 0429 仍成立：

- ✅ V5 归因 author 措辞需降级 → 仍成立
- ✅ V5 归因需 full-val + Path A 闭环 → 仍 pending
- ✅ V6 pilot 不阻塞 → 仍成立
- ✅ V6 200K 放行延至 Phase II 分歧点 → 仍成立

0429 的新增内容是**风险层叠**而非结论翻转：
- 在"V5 归因 pending"基础上，叠加了"NC 后段出现 transport-pressure 上升零星信号"的新假设候选
- 在"V6 +50K 分歧点决策"基础上，叠加了"chain quality 退化"的新判据要求

整体上 0428 verdict 是 0429 addendum 的基础，二者**累加阅读**才是当前完整 supervisor 立场。
