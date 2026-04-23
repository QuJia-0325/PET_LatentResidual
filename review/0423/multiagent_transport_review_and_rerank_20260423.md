# 0423 Multi-Agent 调研记录：Full-val Rerank + EMA + D50->D20目标对齐 + Rollout Trade-off

日期：2026-04-23  
分支：`foc_lite_hop0`

---

## 1. 本次执行内容

按你的要求，完成了四件事：

1. 用统一协议对 `C/N1/v2` 的 `best/last` 做 full-val rerank（clip3）。
2. 调用 agent 评估 `EMA` 是否有效。
3. 调用 agent 分析“训练目标/选优偏 tail，与 D50->D20 主瓶颈叙事不一致”的修正方案。
4. 调用 agent 分析 rollout 前期非 open-loop 与 eval open-loop 的 trade-off 是否合理、如何优化。

---

## 2. Full-val Rerank（统一协议）

### 2.1 协议说明

- 指标：`clip_max=3` 的 per-sample PSNR（公式与 `calc_psnr_clip3` 等价）。
- split：`val` 全集（`n=7403`）。
- 解码：默认 decode 路径（与训练配置一致；若模型设置 `skip_first_tp` 则保持一致逻辑）。
- 输出 artifact：
  - 合并 JSON：`review/0423/artifacts/fullval_rerank_c_n1_v2_clip3_fast_unified_20260423_143907.json`
  - 合并 CSV：`review/0423/artifacts/fullval_rerank_c_n1_v2_clip3_fast_unified_20260423_143907.csv`

### 2.2 结果总表（full-val）

| Scheme | Ckpt | D20 | D10 | D4 | NORMAL | transport_avg(D20,D10,D4,NORMAL) | all_avg |
|---|---|---:|---:|---:|---:|---:|---:|
| C | best | 35.569030 | 35.942137 | 36.505393 | 36.805528 | **36.205522** | **37.488894** |
| C | last | 35.570782 | 35.948680 | 36.504914 | 36.793371 | **36.204437** | **37.488026** |
| N1 | best | 35.558228 | 35.874531 | 36.406767 | 36.654230 | **36.123439** | **37.423227** |
| N1 | last | 35.556761 | 35.896565 | 36.456764 | 36.734590 | **36.161170** | **37.453412** |
| v2 | best | 35.566331 | 35.924282 | 36.499671 | 36.798291 | **36.197144** | **37.482191** |
| v2 | last | 35.569393 | 35.930204 | 36.493663 | 36.758025 | **36.187821** | **37.474733** |

### 2.3 排名（按 transport_avg）

`C_best > C_last > v2_best > v2_last > N1_last > N1_best`

### 2.4 关键观察

1. `N1` 出现了训练期 best 与 full-val 反转：
   - `N1_last (36.161170)` 明显优于 `N1_best (36.123439)`，差值 `+0.037731`。
   - 支持“rolling-window best 可能与 full-val 最优不一致”的判断。

2. `v2` 相比 `C`：
   - `v2_best (36.197144)` 低于 `C_best (36.205522)`，差值约 `-0.008378`。
   - 在当前统一 full-val 协议下，`v2` 未超过 `C`。

---

## 3. EMA 专项结论（Agent）

### 3.1 结论

- 证据强度：**弱**。
- 目前不能得出“EMA 有稳定收益”的结论。
- 当前更稳妥建议：**不要把 EMA 作为默认训练有效性结论**；代码可保留，但需先做严格 A/B 才能定论。

### 3.2 依据（核心）

1. 已执行 run 的对比存在混杂项（不仅仅是 EMA 开关）。
2. 文档/配置与归档产物对 EMA 状态存在不一致描述（需谨慎解读）。
3. 本次统一 full-val 下，`v2` 未优于 `C`。

### 3.3 建议

- 若要严谨验证 EMA：跑一个仅切换 `ema.enabled` 的 clean A/B（其余完全不变），并做同协议 full-val 对比。

---

## 4. 训练目标/选优偏 tail vs D50->D20 瓶颈（Agent）

### 4.1 结论

当前系统是“主干偏 tail + hop0 辅助分支补偿”的混合设计。若主叙事是“D50->D20 是核心瓶颈”，应先对齐 `best_metric`，再调整 rollout/pair 权重。

### 4.2 最小改动方案（推荐顺序）

1. **方案 A（先做）**：只改选优，不改训练梯度
   - `best_metric = val_chain_d20_mse`（或 `val_rollout_step_0`）

2. **方案 B**：A + 前移 rollout step 权重
   - 例如：`[1.50, 1.20, 1.00, 1.00]`

3. **方案 C**：B + 前移 pair sampling / pair loss 权重
   - 例如：`pair_sample_probs=[0.40,0.25,0.20,0.15]`
   - `pair_loss_weights=[1.80,1.20,1.00,1.00]`

### 4.3 评估门槛（建议）

- 主看：`full-val D20` 与 `transport_avg`。
- 约束：不要让 `D10/D4/NORMAL` 明显退化。
- 同时跟踪：`val_rollout_step_0..3`、`val_foc_gap`、loss fractions。

---

## 5. Rollout Trade-off（Agent）

### 5.1 结论

你当前“前期 GT 稳定训练、后期提高 Pred 比例实现级联”的设计是**合理**的，不建议推翻。
问题主要是：分布差距监控不够显式，而不是方向错误。

### 5.2 低风险优化建议

1. **R1: 更快交棒（不改机制）**
   - `warmup_ratio: 0.10 -> 0.05`
   - `ramp_ratio: 0.30 -> 0.20`

2. **R2: 保持当前过渡 + 启用/强化 FOC-lite**
   - 用一致性约束降低 open-loop 级联漂移风险。

3. **R3: 双-lane 验证（诊断）**
   - 同时记录 mixed lane 和 open-loop lane 的 gap；
   - best 选择仍只看 open-loop 主指标。

---

## 6. 建议执行顺序（可直接落地）

1. 固化 selection 协议：训练中窗口指标只作监控，报告与结论以 full-val rerank 为准。  
2. 先做目标对齐最小改动（方案 A）：`best_metric` 切到 D20 导向，复跑并 full-val。  
3. 若 A 仍无提升，再做方案 B（rollout 前移）；必要时再做方案 C（pair 前移）。  
4. EMA 单独做 clean A/B，不再和其他改动混在同一个结论里。

---

## 7. Agent 任务对应

- EMA 专项：agent 独立审查（证据强度评估 + 混杂项分析）。
- D50->D20 目标对齐：agent 独立审查（代码路径 + 最小改动方案）。
- Rollout trade-off：agent 独立审查（机制合理性 + 低风险优化矩阵）。

本文件汇总了三路 agent 结论，并结合本次 full-val rerank 实测结果给出统一建议。
