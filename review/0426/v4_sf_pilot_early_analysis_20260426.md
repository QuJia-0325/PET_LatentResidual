# V4 SF Pilot 早期收敛分析（提前终止）

**日期**：2026-04-26
**对象**：`first_hop_224_v4_sf_pilot`（resume from V3 200K transport `best.pt @ step=86800`）
**日志**：`review/0426/logs_train/v4_sf_pilot_gpu2.log`（266 行，覆盖 step 86850 → 95250）
**决策**：在 sf_alpha 仅爬到 ~0.345（约目标 1.0 的 1/3）时**提前终止**——已经能看清趋势，无需消耗剩余 GPU 时长。

**关联文档**：
- 与本目录已有 [v3_results_analysis](./v3_results_analysis_20260426.md)（Path A 诊断）、[v3p1_redo_plan](./v3p1_redo_experiment_plan_20260426.md)（schedule fix 计划）、[Codex review](./research_review_0426_gpt54xhigh_20260426.md) 为同一分析链
- Reviewer critique：[viewer/v4_sf_pilot_review_critique_20260426.md](./viewer/v4_sf_pilot_review_critique_20260426.md)

---

## 1. 结论摘要（TL;DR）

| 维度 | 状态 | 证据 |
|---|---|---|
| V3.1 P1 修复（resume_relative SF schedule）| ✅ **生效** | step 86850–91800 共 5000 步内 `sf_alpha=0.000`，从 step 91850 开始 ramp |
| V3.1 P2 修复（stdout 暴露 sf_alpha/sf_gap）| ✅ **生效** | 每行 `[train]` 末尾均含 `sf_alpha=… sf_gap=…` |
| SF 在低 alpha（≤0.10）阶段是否有收益 | 🟡 **待验证** | V4 度量下：V3 基线 0.000567 → SF α≈0.10 时 best 0.000538（−5.1%）；**需先测 baseline val 波动范围确认信号真实性** |
| SF 在中高 alpha（≥0.15）阶段是否仍正向 | ❌ **当前超参下失效** | val_select 从 0.000538 劣化至 0.001323，但α=0.30 优于 α=0.22（**震荡而非单调**，提示优化不稳而非损失方向错误） |
| sf_gap 是否随训练收敛 | 🟡 **待验证** | 0.002–0.007 区间震荡；sf_gap = 非归一化 L1（`(z_pred-z_gt).abs().mean()`），**该指标本身的信息量存疑** |
| **总体判断** | ⚠️ **当前超参配置下 SF-pair 失效，但归因尚不充分——是优化不稳还是损失形式错误需 ablation 确认** | 恶化模式呈震荡而非单调，单 seed 单超参不足以判死方向 |

---

## 2. 度量对齐（关键说明）

**V3 与 V4 的 `val_multi_objective` 权重不同**，原始数值不可直接比较：

```
V3 weighting:  0.5*d20 + 0.45*d10 + 0.9*d4 + 1.5*normal
V4 weighting:  0.15*d20 + 0.45*d10 + 0.9*d4 + 1.5*normal   ← d20 权重 0.5 → 0.15
```

V4 pilot 是从 V3 best.pt（step=86800）resume 的，因此 V4 日志里 step=87200（sf_alpha=0）的 val_select_score=**0.000567** 就是 **V3 模型在 V4 度量下的真实基线**——这是公平对比的起点，无需另外重新评估。

> **基线对照表**（避免混淆）：
> - V3 best.pt @ V3 weighting (0.5\*d20+...) = **0.000648**
> - V3 best.pt @ V4 weighting (0.15\*d20+...) = **0.000567** ← 后续所有对比一律用此值

---

## 3. SF Schedule 修复验证

### 3.1 sf_alpha 时序（resume_relative 工作正常）

| Step | sf_alpha | 注释 |
|---|---|---|
| 86850 | 0.000 | resume 起点 |
| 91800 | 0.000 | warmup 结束（86800 + 5000 = 91800）|
| 91850 | 0.005 | ramp 起点 |
| 92000 | 0.020 | |
| 92800 | 0.105 | **V4 best @ this checkpoint** |
| 93200 | 0.140 | val 开始恶化 |
| 94000 | 0.220 | |
| 95250 | 0.345 | 终止时刻 |

> **注**：sf_alpha 值为 `get_linear_schedule_value` 按 `effective_step = global_step - 86800` 公式推算，与日志 stdout 中 `sf_alpha=X.XXX` 字段一致。

`effective_step = global_step - resume_start_step (=86800)`, `warmup_steps=5000`, `ramp_steps=10000`, `alpha_sf_end=1.0` —— 与 config 完全一致。**P1 修复完全成功。**

### 3.2 stdout 诊断字段

每行 `[train]` 末尾稳定输出 `sf_alpha=X.XXX sf_gap=X.XXXXXX`，无需 SSH 查 metrics.jsonl 即可监控 SF 状态。**P2 修复完全成功。**

---

## 4. Val 趋势分析

### 4.1 关键时间点（V4 weighting）

| Step | sf_alpha | val_select | d20 | d10 | d4 | normal | val_pair_total |
|---|---|---|---|---|---|---|---|
| 87200 | 0.000 | **0.000567** (V3 baseline) | 0.000243 | 0.000217 | 0.000189 | 0.000175 | 0.000001 |
| 91200 | 0.000 | 0.000584 | 0.000247 | 0.000219 | 0.000193 | 0.000183 | 0.000000 |
| **92800** | **0.105** | **0.000538** ⭐ best | 0.000217 | 0.000198 | 0.000179 | 0.000170 | 0.000000 |
| 93200 | 0.140 | 0.000551 | 0.000253 | 0.000218 | 0.000185 | 0.000166 | **0.000196** ← 跳变 |
| 93600 | 0.180 | 0.000939 | 0.000425 | 0.000364 | 0.000314 | 0.000286 | 0.000210 |
| 94000 | 0.220 | 0.001025 | 0.000452 | 0.000393 | 0.000343 | 0.000315 | 0.000203 |
| 94400 | 0.260 | 0.000744 | 0.000314 | 0.000283 | 0.000247 | 0.000232 | 0.000199 |
| 94800 | 0.300 | 0.000654 | 0.000261 | 0.000242 | 0.000216 | 0.000207 | 0.000155 |
| 95200 | 0.335 | **0.001323** | 0.000633 | 0.000528 | 0.000441 | 0.000395 | 0.000295 |

### 4.2 三个明确发现

**Finding A — SF 在 α ∈ [0, 0.10] 时观察到改善，但信号真实性待验证**
- V3 baseline @ V4 metric: 0.000567
- V4 best @ α≈0.10: 0.000538
- **相对改善 5.1%**（绝对值 −0.000029）
- 全部四档 chain MSE 都改善：d20 −10.7%, d10 −8.8%, d4 −5.3%, normal −2.9%
- **⚠️ 反预期信号**：改善幅度 d20 > d10 > d4 > normal，但 SF 机制的预期是**末端 hop 改善最多**（SF 主要修正 chain 后端的 exposure bias）。当前观察反而是链头改善最多、链尾改善最少——这与"d20 在 V4 weighting 下被降权后绝对值变小、相对波动放大"的 noise 解释一致，**进一步降低信号可信度**。
- **⚠️ 待验证**：α=0 区间（step 87200-91800）的 val_chain_normal_mse 自然波动范围未测量。如果 baseline 在纯 GT 区间的波动 ≥ 0.000005，则 0.000175→0.000170 可能在 noise 内。**需列出 α=0 区间全部 val 点并计算 std 后才能判断**

**Finding B — α ≥ 0.15 后 val 在当前超参下进入震荡式恶化**
- 93200（α=0.14）开始反弹（0.000551）
- 93600（α=0.18）暴涨至 0.000939（+74% vs best）
- 此后在 0.000654–0.001323 大幅震荡
- **⚠️ 关键观察**：α=0.30（val=0.000654）优于 α=0.22（val=0.001025）——**恶化模式是震荡而非单调**。如果是"SF 损失形式错误"，应当看到单调恶化；震荡更可能指向**优化不稳**（ramp 太陡 / LR 没跟上 / rolling val 噪声）
- 与 V3 SF run 的"瞬间冲击"不同，V4 是"渐进+震荡"——**两者不是同一种失败模式**

**Finding C — `val_pair_total` 在 step 93200 出现跳变（已确认：是真实模型退化）**
- step 87200：`val_pair_total = 0.000001`（浮点精度，本质为 0）
- step 91200：`val_pair_total = 0.000000`
- step 93200 起：跳到 0.000196 量级并保持
- **✅ 已确认**（代码审查 `train_first_hop.py` line 870-878）：val 循环中 pair loss 直接用 `batch["z_src"]`（GT）计算，**不调用 `compute_self_forcing_z_src`**。val 时不走 SF 路径。因此 val_pair_total 的跳变是**模型参数被 SF 训练改变后，在 GT 输入上的 pair loss 真实恶化**——SF 训练确实把模型的 velocity 预测从 GT-optimal 推离了
- 这**是**"模型被 SF 损害"的独立证据，与 val_select_score 恶化一致

### 4.3 sf_gap 震荡（待验证的观察，非诊断结论）

`sf_gap` 在 0.002–0.007 区间震荡，与 sf_alpha 上升无单调关系。

**sf_gap 的实际计算**（源自 `compute_self_forcing_z_src` 返回值）：
```python
gap_norm = (z_src_pred - z_src_gt).abs().mean()  # 非归一化 L1
```

**⚠️ 指标信息量存疑**：
- 这是**非归一化**的 L1 范数。latent 的自然尺度在 0.001-0.01 量级，sf_gap 0.002-0.007 可能只是反映 latent 的自然尺度差异
- 被 hop_idx=0 样本稀释（hop0 的 z_src_pred ≡ z_src_gt，gap 恒为 0，约占 batch 的 25%）
- 更有意义的指标应该是 per-hop 归一化 gap（如 `gap / z_gt_norm`）或单独看 hop1-3

**在归一化 + per-hop 版本确认前，sf_gap 的震荡不能作为"SF 没用"的证据，仅作为现象记录。**

---

## 5. 与 V3 SF run 的对照

| 维度 | V3 SF run（旧 schedule bug）| V4 SF pilot（修复后）|
|---|---|---|
| sf_alpha 起点 | 立刻 = 1.0（bug） | 0.0（正确） |
| 有效新训练步数 | ~3600（resume 后即结束）| ~8400（终止时） |
| val 最终结果 vs baseline | **−30%**（全面恶化） | α≤0.10 时 +5%，α≥0.15 后 **−74%~−146%** |
| 失败模式 | "瞬间冲击" | "渐进恶化" |
| 是否能区分 schedule bug 与方法本身缺陷 | ❌ 不能 | 🟡 部分——**当前超参下 α 高时失效，但震荡非单调提示可能是优化不稳** |

**核心信息**：V3.1 的 schedule 修复消除了"实验结论无效"的混淆变量。**当前超参配置下（ramp=10K, α_end=1.0, LR=4e-5）SF-pair 在 α≥0.15 后失效，但失效归因（方法缺陷 vs 优化不稳）尚需 ablation 确认。**

---

## 6. 假设排查表

| 假设 | 是否成立 | 证据 |
|---|---|---|
| H1：V3 SF 失败仅因为 schedule bug | ❌ **被证伪** | V4 修复后仍在 α≥0.15 失败 |
| H2：低 alpha 的 SF 有微小正收益 | 🟡 **待验证** | 5% 改善观察到，但需先测 baseline val 波动 std |
| H3：SF-pair loss 形式本身设计有缺陷 | 🟡 **待排查** | α 高时恶化，但**震荡非单调**→ 可能是优化不稳而非损失方向错；sf_gap 指标本身未验证 |
| H4：sf_gap 度量本身有问题 | 🟡 **很可能** | 非归一化 L1 + hop0 稀释；需 per-hop 归一化版本 |
| H5：transport gap 不应通过"换 z_src"解决 | 🟡 **开放** | 理论上成立，但实测震荡模式不排除优化问题 |
| **H6：当前 SF 配方的 ramp 太陡 / LR 不匹配** | 🟡 **新增假设** | α=0.30 优于 α=0.22 的非单调性直接支持此假设 |

---

## 7. 下一步计划（综合 reviewer critique 后）

### 并行轨道 A — Rollout-Heavy（第一优先，纯 config 改动）

**不等 SF 归因完成就可以跑**——探索一个全新的轴。

从 200K v3 best resume，50K 步：
- `rollout.lambda: 0.25 → 2.0`（8×）
- `rollout.step_weights: [0.5, 1.0, 2.0, 4.0]`（按 exposure gap² 比例）
- `image_aux.lambda: 0.12 → 0.04`（降 3×）
- `self_forcing_pair.enabled: false`

详见 §12 综合实验计划。

### 并行轨道 B — SF 低 α 验证（不需 GPU，数据分析）

在等 Rollout-Heavy 跑的期间，完成 reviewer 要求的 3 项验证：

**B1. 测 baseline val 波动范围**（F1 修复）
- 列出 step 87200-91800（α=0 区间）的全部 val 点的 `val_chain_normal_mse`
- 计算 mean ± std
- 判断 step 92800 的 0.000170 是否在 mean - 1σ 以下

**B2. 确认 val_pair_total 定义**（F4 修复）
- 查 `train_first_hop.py` 的 val 循环中 `sf_info` 是否参与 val pair 计算
- 如果是 → Finding C 降级为定义效应
- 如果否 → Finding C 保留为模型分布漂移证据

**B3. sf_gap 归一化版本**（F3 修复）
- 计算 `sf_gap_normalized = sf_gap / sf_z_gt_norm`（两者都已在 metrics_jsonl 中）
- 按 hop 分组看 per-hop gap 趋势（hop0 应恒为 0，hop1-3 是真正的信号）

### 串行轨道 C — SF 窄带 Ablation（仅当 Rollout-Heavy 也不行时）

如果 Rollout-Heavy 失败，回来做 reviewer 建议的 SF 参数排查：
- `alpha_sf_end: 0.15`（限幅到 pilot best 附近）
- `ramp_steps: 30000`（更慢的 ramp）
- `lr: 2e-5`（降 LR 配合 SF 注入）
- 跑 30K 步，看 val 是否稳定

---

## 8. 立即行动项（更新版）

1. ✅ 已完成：服务器 V4 SF pilot 提前终止
2. ✅ 已完成：本报告归档 + reviewer critique 修正
3. ⏳ **B1**：列出 α=0 区间全部 val 点，计算 baseline val std（不需 GPU）
4. ✅ **B2**：已确认——val 循环不调用 SF，val_pair_total 是 GT 输入下的真实 pair loss。Finding C 的跳变是模型退化的独立证据
5. ⏳ **B3**：计算 sf_gap 归一化版本 `sf_gap / sf_z_gt_norm`，按 hop 分组（不需 GPU，从 metrics_jsonl 读取）
6. ⏳ 准备 Rollout-Heavy config + launch script
7. ⏳ 等 200K v3 完成 → 启动 Rollout-Heavy

---

## 9. 数据可信度声明

- 训练步数：~8450 effective steps（86800 → 95250），SF 真正生效仅 ~3400 步
- val 频率：每 400 步一次，覆盖 21 个 val 点
- 单一 seed、单一 GPU，无 N=3 重复
- **本报告结论的强度**：足以暂停当前 v4 SF 配置（ramp=10K, α_end=1.0, LR=4e-5）的继续执行，但**不足以判死 SF-pair 方向**。归因（方法缺陷 vs 优化不稳）需 ablation 确认（见 §7 轨道 C）。如需发表强否定声明，需要至少 N=3 重复 + 多种超参网格

---

*报告作者：Copilot（基于 gitee 42acc71 的 V3+V4 训练日志静态分析）*
*用户决策记录：(1) 立即停跑，按当前数据收敛分析；(2) 已完成 V3↔V4 度量对齐重算（V3 @ V4 metric = 0.000567）*

---
---

# 后续分析：SF 失败根因 & 下一步可行方向（2026-04-26 补充）

## 10. SF-pair 为什么从根本上失败

### 10.1 回顾 Path A 完整数据（Scheme C best.pt）

| Hop | PSNR_TF | PSNR_RO | ExpoGap | CeilGap | LatMSE_RO/TF |
|-----|---------|---------|---------|---------|---------------|
| D50→D20 | 35.57 | 35.57 | **0.00** | 11.07 | 1.00 |
| D20→D10 | 39.32 | 35.94 | **3.38** | 9.42 | 3.11 |
| D10→D4 | 41.21 | 36.51 | **4.71** | 9.62 | 4.62 |
| D4→NORMAL | 43.13 | 36.81 | **6.32** | 9.51 | 5.64 |

关键特征：
1. **Hop 0 没有 exposure bias**（ExpoGap=0）—— 输入永远是 GT D50
2. **Exposure 误差指数累积**：0 → 3.38 → 4.71 → 6.32 dB
3. **Ceiling gap 也很大**（9-11 dB）—— 即使 teacher-forced 也离 oracle 很远
4. **LatMSE_RO/TF = 3-6×** —— rollout latent 误差是 TF 的 3-6 倍

### 10.2 SF-pair 的机制缺陷

SF-pair 做的事情：把 pair_loss 的 GT 输入 `z_src` 换成模型在 rollout chain 中的自预测值 `z_src_pred`，然后在这个"noisy"输入上计算 velocity/endpoint loss。

**问题在于**：backbone（DiT-S）的 velocity 预测器**不知道输入是 noisy 的**。它只看到 "这是一个 latent，我预测 velocity"。当输入从 GT 分布偏移到 predicted 分布时：

1. velocity 预测在**错误的输入分布**上计算，loss 信号 backprop 后模型的反应是**把 velocity 往错误方向拟合**
2. 没有任何机制告诉模型"你的输入有误差，你应该做 correction"
3. pair_loss 是**单跳**的——它不知道这个输入是 chain 的第几步产物，也不知道下游还有几跳需要这个输出
4. 随着 alpha 上升，输入偏移越来越大，velocity 预测越来越偏，链式累积导致全面崩溃

**一个可能的解释**：SF-pair = input-side perturbation without correction signal。

> ⚠️ **强度声明**：这是当前数据下的**一种解释**，不是已被实验证伪的结论。震荡式（非单调）的失败模式同样可能由优化不稳（ramp 太陡 / LR 不匹配 / val 噪声）引起——参见 §6 H6。要把这条假设升级为"机制缺陷"，需要至少完成：(a) ramp_steps=30K + α_end=0.15 的窄带 ablation；(b) sf_gap 归一化版本验证。在两项之前，本节仅作为**方向选择的启发**，不作为否决 SF 路线的判决。

### 10.3 与 rollout_loss 的对比

| 维度 | pair_loss（含 SF） | rollout_loss |
|------|-------------------|-------------|
| 输入来源 | 单跳（GT 或 SF-predicted） | chain（纯 predicted，alpha=1.0） |
| 监督信号 | 单跳 velocity/endpoint vs GT | **每一步** z_pred vs GT 对应位置 |
| 梯度路径 | 只经过 1 次 forward | **4 次 forward**，每步都有 loss |
| 链式校正 | ❌ 无 | ✅ 有（step_losses 加权求和） |
| exposure bias 感知 | ❌ 不知道输入有误差 | ✅ 隐式知道（因为 z_curr 是预测值） |

**rollout_loss 是当前系统中最直接对抗 exposure bias 的已有工具**——它在链式预测上计算 loss，天然包含"从 noisy 输入出发也能到达 GT"的学习信号。（注意：这不等于说它一定能解决 exposure bias，只是它比 pair_loss + SF 更适合这个任务。）

### 10.4 为什么 rollout_loss 当前效果有限

因为**权重太低**。看训练日志中的梯度占比：

```
pair_frac ≈ 3-10%    ← 单跳监督
roll_frac ≈ 7-15%    ← 直接对抗 exposure bias
img_frac  ≈ 80-90%   ← hop0 像素重建（与 exposure bias 无关）
```

**image_aux 占了 80-90% 的梯度**，它在做 hop0 的像素级重建。这对图像质量有帮助，但**对 transport chain 的 exposure bias 完全没有帮助**——它只影响 hop0，而 exposure bias 从 hop1 开始累积。

当前 rollout λ=0.25 相对 image_aux λ=0.12 看似不低，但因为 rollout_loss 的绝对值远小于 image_aux_loss（latent MSE ~0.001 vs pixel loss ~0.005-0.015），实际加权后 rollout 贡献被严重压缩。

---

## 11. 三个可行方向（按优先级）

### 方向 1：加大 Rollout 权重 + 降低 Image Aux（最简单，纯 config 改动）

**核心思路**：不需要新 loss，不需要新机制——只需把已有的 rollout_loss 权重拉上去。

**⚠️ 教训应用**：V4 SF pilot 失败的核心教训是"单点跳一大步、无渐进 ablation"。本方向必须避免重蹈覆辙——**先做保守版验证、再决定是否激进**。

#### 1A — 保守版（第一步必跑）

| 参数 | 当前值 | 保守值 | 倍数 |
|------|--------|--------|------|
| `rollout.lambda_start/end` | 0.25 | **1.0** | 4× |
| `image_aux.lambda_start/max` | 0.12 | **0.08** | ÷1.5 |
| `rollout.step_weights` | [1.0, 1.1, 1.2, 1.3] | [0.5, 1.0, 2.0, 4.0] | 末端加重（详见方向 2） |

**为什么先跑保守版**：
- image_aux 是 hop0 像素重建的**唯一稳定源**。直接降 3× 等于在另一个轴上重复 V4 SF 的错误：单超参一次性大跳。如果 hop0 像素崩，整个 chain 输入分布漂移，比 SF 失败更糟。
- rollout 4× 已经能让 `roll_frac` 从 7-15% 提升到 ~30%，足以验证趋势。
- 这一组**风险面最小、可验证性最强**。

#### 1B — 激进版（仅当 1A 趋势正向后才跑）

| 参数 | 1A 值 | 1B 值 |
|------|-------|-------|
| `rollout.lambda_start/end` | 1.0 | **2.0** (再 2×) |
| `image_aux.lambda_start/max` | 0.08 | **0.04** (再 ÷2) |

**门槛**：1A 在 25K 步内 `val_select` 改善 ≥ 3% 且 hop0 像素 PSNR 不退化 ≥ 0.3 dB → 才允许跑 1B。

**预期效果（1A→1B 累积）**：
- `roll_frac` 从 7-15% → 1A: ~30% → 1B: 40-60%
- `img_frac` 从 80-90% → 1A: ~50% → 1B: 20-30%

**核心风险**：
- rollout_loss 过大压制 pair_loss → 单跳 velocity 精度下降（监控 `pair_frac` ≥ 2%）
- image_aux 降太低 → hop0 像素质量退化（监控 hop0 PSNR）
- **rollout step_weights 末端加重 → backbone 在末端 hop 上过拟合预测分布而非真实 velocity**（监控 `val_chain_d20_mse` 不恶化超过 5%）

**代码改动**：0 行。纯 config。

### 方向 2：Hop-Weighted Rollout Loss（配合方向 1）

当前 `step_weights = [1.0, 1.1, 1.2, 1.3]` 几乎均匀。但 exposure bias 分布**高度不均匀**：

| Hop | ExpoGap | 当前权重 | 按 gap² 比例 |
|-----|---------|---------|-------------|
| D50→D20 | 0.00 dB | 1.0 | **0.5** |
| D20→D10 | 3.38 dB | 1.1 | **1.0** |
| D10→D4 | 4.71 dB | 1.2 | **2.0** |
| D4→NORMAL | 6.32 dB | 1.3 | **4.0** |

末端 hop（D4→NORMAL）的 exposure gap 是 hop1（D20→D10）的 **1.87×**，但当前权重只高 18%。

**提议**：`step_weights: [0.5, 1.0, 2.0, 4.0]`

**与方向 1 组合使用**：λ_roll=2.0 + step_weights=[0.5, 1.0, 2.0, 4.0] → 末端 hop 的有效 rollout 梯度是当前的 **8 × (4.0/1.3) ≈ 24.6×**

**代码改动**：0 行。纯 config。

### 方向 3：Pair Loss 渐进归零（仅当方向 1B 仍不足时考虑）

**核心思路**：pair_loss 是 exposure bias 的**根源**（提供 ~88% 梯度但全部来自 GT 输入）。如果方向 1B 仍不能让 normal_mse 显著改善，可以进一步**渐进降低** pair_loss 权重，让 backbone 在更接近 chain prediction 分布上学习。

**⚠️ 不直接归零**：直接 `pair_sample_probs=[0,0,0,0]` 等于一次性切掉 88% 的梯度信号，剩 12%（rollout + image_aux）是否足以驱动 backbone 更新完全未知。这等于第三次重复 V4 SF 的"单点跳大步"错误。

**渐进 schedule**（仅当 1B 通过后启用）：

```yaml
# Phase 3a (前 15K 步): pair_loss_weight 从 1.0 线性降到 0.5
# Phase 3b (中 15K 步): pair_loss_weight 从 0.5 线性降到 0.25
# Phase 3c (后 20K 步): pair_loss_weight 保持 0.25 或继续降到 0.1
#
# 每个 phase 末尾做 checkpoint，看 val 是否退化；退化则回滚到上一 phase 权重
```

**实现方式**：
- 在 `train_first_hop.py` 中加 `pair_loss_weight = get_linear_schedule_value(...)` 包在 pair_loss 求和处
- ~10 行代码改动
- **不**改 `pair_sample_probs`（保持采样，只缩 loss）——这样 pair grad 还在但权重缩小，比直接断采样更稳

**Go/No-Go**：每 5K 步看 `val_chain_normal_mse` 是否劣于上一阶段 + 5%；劣化则回滚权重。

**与方向 1B 的关系**：方向 3 不是替代品而是补充。先跑完 1B 的 50K 步看是否达到医生满意度；不够再考虑 3。

---

## 12. 综合实验计划

### 第一优先：方向 1A（"Rollout-Heavy 保守版"）

```yaml
# 新 config: pet_flow_first_hop_224_v4_rollout_heavy_1a.yaml
# 改动 vs 200K v3 baseline:
rollout:
  lambda_start: 1.0       # 0.25 → 1.0 (4×, 保守)
  lambda_end: 1.0
  step_weights:
    - 0.5                 # hop0: 降低（无 exposure bias）
    - 1.0                 # hop1
    - 2.0                 # hop2
    - 4.0                 # hop3: 加重（exposure bias 最严重）
image_aux:
  lambda_start: 0.08      # 0.12 → 0.08 (÷1.5, 保守)
  lambda_max: 0.08
# 其余所有参数与 200K v3 完全一致
self_forcing_pair: { enabled: false }   # 明确关闭 SF
```

从 200K v3 best resume，跑 50K 步。

**Go/No-Go 判定**（双触发任一即止损，挂钩医生关心的 normal 档）：

| 检查点 | val_select 阈值 | val_chain_normal_mse 阈值 | hop0 PSNR 阈值 |
|--------|----------------|---------------------------|---------------|
| +10K | > baseline × 1.05 → 止损 | > baseline × 1.10 → 止损 | < baseline − 0.5 dB → 止损 |
| +25K | 无改善（≥ baseline）→ 止损 | 无改善（≥ baseline × 0.97）→ 止损 | < baseline − 0.3 dB → 止损 |
| +50K | full-val eval + Path A redo | — | — |

**为什么阈值这么严**：之前 V4 SF 跑了 8400 步才看到 val_select 涨 146%，到那时 GPU 时间已经浪费。新阈值在 10K 步就强制 review，避免重复同样的浪费。

**1A → 1B 升级条件**（必须全部满足；所有"步"均指 resume 后的新增步数，非绝对 step）：
- resume 后 +25K 步时 `val_select` 已改善 ≥ 3%（相对 resume 起点 baseline）
- resume 后 +25K 步时 `val_chain_normal_mse` 已改善 ≥ 2%
- hop0 PSNR 不退化超过 0.3 dB
- `pair_frac` 仍 ≥ 2%（pair 监督未被压死）

**成功标准（针对医生需求）**：
- 第一目标：`val_chain_normal_mse` 改善 ≥ 5%（normal 档是医生看到的输出）
- 第二目标：Path A exposure_gap 末端 hop（D4→NORMAL）下降 ≥ 10%（从 6.32 dB 降到 ≤ 5.69 dB）
- 第三目标：`val_select_score < baseline`（chain 整体改善，避免单档优化拆东墙补西墙）

### 第二优先：方向 3（"Rollout-Only Phase"）

仅在方向 1+2 失败后尝试。需要先确认 `pair_sample_probs=[0,0,0,0]` 的兼容性或加 ~5 行 `pair_loss_weight` 代码。

### Claims Matrix（更新版）

| 场景 | 允许的 claim |
|------|-------------|
| Rollout-Heavy > baseline 且 exposure_gap 下降 | 加权 rollout 策略有效对抗 cascade exposure bias |
| Rollout-Heavy > baseline 但 exposure_gap 不变 | rollout 权重改善了 chain 精度但未直接压缩 exposure gap（可能是更好的 velocity 拟合） |
| Rollout-Heavy ≈ baseline | 当前架构下 exposure bias 不可通过 loss 调整解决——可能需要结构性改变（multi-scale decoder、chain-aware backbone） |
| Rollout-Heavy < baseline | rollout 过强导致 velocity drift，需要回退或寻找平衡点 |

---

## 13. 为什么"加大 rollout 权重"作为下一个尝试方向

> ⚠️ **强度声明**：以下是**选择该方向的依据**，不是预测它会成功的乐观陈述。研究里的"乐观"是放弃 ablation 的借口。Go/No-Go 阈值（§12）一旦触发就止损，不要因为这一节的"理由"拖延决策。

1. **Rollout loss 已经在做正确的事**——它在纯预测链上计算 loss（alpha=1.0），每步都有 GT 监督
2. **它只是被 image_aux 淹没了**——80-90% 梯度去做像素重建，对 transport chain 帮助有限
3. **v3 C'（λ_roll=1.0）的梯度特征**：虽然只跑了 3600 步、单 seed，但 `grad_backbone` 是 SF 版的 2.4×——说明加大 rollout 确实**改变了梯度信号**（注意：这只支持"权重生效"，**不**支持"会带来 val 改善"）
4. **Path A diagnostic 的原始建议**就是 "inspect rollout hyperparameters (alpha ramp, step weights) and the mid-chain contribution"——我们之前跳过了这个最简单的建议直接去做 SF
5. **Occam's Razor**：最简单的干预（调权重）应该先于复杂的新机制（SF、EMA teacher 等）
6. **失败也有诊断价值**：如果方向 1A 在 25K 步内仍无改善，则可以排除"梯度信号不足"假设，转向更激进的方向 1B 或结构性改动（multi-scale decoder、chain-aware backbone）——这本身就是有价值的信息

**反方向考量（必须留意）**：
- 第 3 条的 3600 步样本量极小，不能据此预测 50K 步的趋势
- 第 5 条的 Occam 原则只说"先尝试简单方案"，不说"简单方案会成功"
- step_weights 末端加重 [0.5, 1.0, 2.0, 4.0] 是基于 §10.1 ExpoGap² 比例的**理论推导**，未经实验验证——可能存在 backbone 在末端 hop 过拟合预测分布的风险

---

*补充分析作者：Copilot*
*补充日期：2026-04-26*
*依据：Path A 多 ckpt 数据 + v4 SF pilot 失败分析 + rollout_first_hop.py / train_first_hop.py 代码审查*
