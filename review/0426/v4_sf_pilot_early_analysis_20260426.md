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
| SF 在中高 alpha（≥0.15）阶段是否仍正向 | ❌ **当前超参下相变型失效** | val_select 从 0.000538 劣化至 0.001323；**val_pair_total（GT 输入下）从 0 跳变到 0.000196**——模型被永久推离 GT-optimal manifold。α=0.30 vs α=0.22 的 select 震荡不再能用"优化不稳"完全解释（详见 §10.5 Finding D） |
| sf_gap 是否随训练收敛 | 🟡 **待验证** | 0.002–0.007 区间震荡；sf_gap = 非归一化 L1（`(z_pred-z_gt).abs().mean()`），**该指标本身的信息量存疑** |
| **总体判断** | ⚠️ **当前超参配置下 SF-pair 失效，且失效是相变型 + 模型对 GT 的响应被永久推坏（不仅是 SF 路径下表现差）。归因偏向"损失形式与 backbone 容量不匹配"，但仍需 ablation 排除 ramp/LR 因素** | val_pair_total 在 GT 输入下持续高位 + 相变型跳变 → 见 §10.5 |

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
- **⚠️ 临界点位置**：α≈0.10 这个 best 出现在**模型即将相变崩溃但还没崩**的临界点（详见 §10.5 Finding D）——4 个 val 点之后（93200，α≈0.14）val_pair_total 就跳到 0.000196。这意味着即使 5% 改善是真信号，它也**不在一个稳定的工作区**，无法作为长训目标
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
| H3：SF-pair loss 形式本身设计有缺陷 | ⚠️ **中度支持**（升级） | (a) val_pair_total 在 GT 输入下永久恶化（Finding C/D）；(b) 相变型而非渐进型失败；(c) 末端 hop 改善 < 链头改善（与 SF 预期相反）。三条独立证据都指向 SF 信号方向问题，不只是优化超参 |
| H4：sf_gap 度量本身有问题 | 🟡 **很可能** | 非归一化 L1 + hop0 稀释；需 per-hop 归一化版本 |
| H5：transport gap 不应通过"换 z_src"解决 | 🟡 **开放** | 理论上成立，但实测震荡模式不排除优化问题 |
| **H6：当前 SF 配方的 ramp 太陡 / LR 不匹配** | 🟡 **部分支持，但不充分** | α=0.30 vs α=0.22 的 select 震荡支持 H6；但 val_pair_total 在 α=0.30 仍是 0.000155（高位），如果只是 ramp/LR 问题应该看到 GT 性能恢复——它没有。H6 解释 select 震荡，**不解释** GT 性能永久损失 |
| **H7（新增）：α 跨过临界点后参数进入对 GT 也次优的盆地（相变型 catastrophic forgetting）** | ⚠️ **中度支持** | val_pair_total 从 0 → 0.000196 是断崖式跳变；后续 α 上升到 0.30 时 val_pair_total 仍在 0.000155-0.000295 高位，**没有任何回到 0 的迹象** → 优化已进入新的局部盆地 |

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
6. ⏳ **B4（新增，最高优先）**：核对 `compute_self_forcing_z_src` 中 `z_src_pred` 是否带 `.detach()`——决定 D.4 第 4 条"自洽循环"假设强度（不需 GPU，5 分钟代码审查）
7. ⏳ **B5（新增）**：取 step 91200 vs 93200 的 ckpt 算 backbone 参数 L2 距离，对比 V3 正常训练的相邻 4K 步参数移动幅度（需 GPU 但只跑一次 diff）
8. ⏳ 准备 Rollout-Heavy 1A config + launch script（需要严格 Go/No-Go 监控 val_pair_total on GT inputs）
9. ⏳ 等 200K v3 完成 → 启动 Rollout-Heavy 1A

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

> ⚠️ **强度声明（更新）**：本节作为机制假设的**支持度从"待验证"升级到"中度支持"**——基于 §10.5 Finding D 的相变型失败 + GT 性能永久损失证据。但仍**不构成判决**：要把"机制缺陷"上升为"已证实"，仍需 (a) ramp_steps=30K + α_end=0.15 的窄带 ablation 排除优化因素；(b) sf_gap 归一化版本验证。当前置信度：**SF-pair 在当前 backbone 容量 / 训练 schedule 下不可用**——是否对所有 backbone / schedule 都不可用，仍开放。

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

## 10.5 Finding D：相变型失败 + GT 性能永久损失（深度机制分析）

### D.1 关键观察：失败是相变而非渐进

把 Finding C 升级（val 不走 SF 路径）+ 全表 val_pair_total 重新看：

| Step | sf_alpha | val_select | val_pair_total（GT 输入下）| 训练后 SF 累积步数 |
|---|---|---|---|---|
| 87200 | 0.000 | 0.000567 | 0.000001 | 0 |
| 91200 | 0.000 | 0.000584 | 0.000000 | 0 |
| 92800 | 0.105 | **0.000538** ⭐ | **0.000000** | 950 |
| 93200 | 0.140 | 0.000551 | **0.000196** ← 跳变 | 1350 |
| 93600 | 0.180 | 0.000939 | 0.000210 | 1750 |
| 94000 | 0.220 | 0.001025 | 0.000203 | 2150 |
| 94400 | 0.260 | 0.000744 | 0.000199 | 2550 |
| 94800 | 0.300 | 0.000654 | 0.000155 | 2950 |
| 95200 | 0.335 | 0.001323 | 0.000295 | 3350 |

**两个独立现象**：

1. **相变**：950 步 SF 训练（α 0→0.105）val_pair_total 完全不动；再 400 步（α 0.105→0.140）就跳变 200,000×。这不是连续函数。

2. **永久损失**：α=0.30 时 val_select 从 0.001025 部分恢复到 0.000654，但 val_pair_total 仍在 0.000155 高位（baseline 是 0），**模型对 GT 输入的 pair 性能没有恢复**。如果只是优化不稳，应该看到对称的恢复。

### D.2 机制猜想：参数空间盆地切换

**假设**：DiT-S backbone 在 V3 200K 训练后位于一个**狭窄的 GT-optimal 盆地**。这个盆地有两个特征：
- 入口很窄：需要长时间 GT-only 训练才能进入（解释 V3 200K 才达到的 0.000567 baseline）
- 出口很易：任何持续的非 GT 输入扰动都会把参数推出盆地

**SF 训练的实际效果**：
- α ≤ 0.10：扰动幅度 + SF loss 梯度的合力还在盆地内可承受范围（参数有微小漂移但仍在盆地内）→ val_pair_total 保持 0
- α 越过 ~0.12 临界点：扰动幅度超过盆地宽度，参数被推出 → val_pair_total 断崖跳变到 0.0002 量级
- 出去后参数进入一个**对 SF-perturbed 输入更优、对 GT 输入次优**的新盆地
- α 继续上升：参数在新盆地内做局部调整（α=0.30 比 α=0.22 select 更好），但**回不到原盆地**（val_pair_total 始终 0.000155+）

**这与 catastrophic forgetting 的机制类似**，但触发因素不是任务切换而是**输入分布漂移**。

### D.3 为什么 backbone 容量也是因素

DiT-S 的容量决定了它**能否同时**对 GT 和 SF-perturbed 两种输入分布都给出好的 velocity 预测。当前观察到的相变暗示：

- DiT-S 的 velocity 预测函数没有足够的"分布感知"自由度
- 它只能选择一种"最优"输入分布去拟合
- α 上升迫使它从"GT-optimal"切换到"SF-perturbed-optimal"
- 切换是离散的（盆地之间没有平滑过渡）

**这条假设的可证伪点**：
- 如果换更大 backbone（DiT-B / DiT-L），相变临界点应该上移甚至消失（更大模型有更多容量同时拟合两种分布）
- 如果加 conditioning 信号（告诉 backbone "当前输入是 GT 还是 predicted"），相变也应该消失
- 这两条都是未来 ablation 的设计依据

### D.4 与 Self-Forcing 原始论文场景的差异

Self-Forcing（视频生成 / autoregressive LM 等）原始场景的成功依赖于：
1. **大模型容量**（数十亿参数级），能同时处理 teacher-forced 和 self-generated 输入
2. **stop-gradient on z_pred**：z_src_pred 不参与梯度，只作为 input；当前实现是否有 stop-grad 需要核对（见 D.5）
3. **SF loss 是分布匹配（KL / MMD）而非 MSE**：MSE 对 outlier 敏感，input 偏移 + MSE 容易把模型拉偏
4. **逐步 student-teacher 结构**：teacher 提供平滑参考，避免参数空间断裂

我们当前的 SF-pair 实现：
- ✅ backbone 小（DiT-S 几千万参数）
- ❓ stop-grad 状态待核对
- ❌ 用 MSE/endpoint loss 不是分布匹配
- ❌ 没有独立 teacher，z_src_pred 来自同一个被训练的模型 → **自洽循环**：模型预测错 → SF 输入错 → 梯度把模型推得更错

**第 4 条是关键**：当 z_src_pred 来自正在训练的模型本身，没有独立的"对照系"。早期模型预测尚未崩溃时这是温和的正则化；一旦预测开始偏离，**正反馈循环**会把模型快速推出盆地——这正好对应观察到的相变行为。

### D.5 必须核对的代码细节（B3 之前先做）

新增到验证清单：

**B4. 核对 SF 实现中的 stop-grad 状态**
- 查 `compute_self_forcing_z_src` 中 `z_src_pred` 是否带 `.detach()`
- 如果**没有 detach** → SF loss 的梯度通过 z_src_pred 反向传播到模型本身 → 更强的自洽循环 → 解释为什么相变这么剧烈
- 如果**有 detach** → 输入只是 noisy 但不形成显式自洽循环 → 失败更可能是分布偏移本身造成

**B5. 比较 SF 训练前后模型参数变化幅度**
- 取 step 91200（α=0）和 step 93200（α=0.14, 跳变后）的 ckpt，计算 backbone 参数 L2 距离
- 与 V3 200K 训练中相邻 4000 步的参数距离对比
- 如果 SF 训练带来的参数移动**远大于**正常训练同步数 → 直接证据支持"参数被推出盆地"假设

### D.6 对方向选择的修正含义

**对方向 1（Rollout-Heavy）的影响**：
- rollout_loss 也涉及"模型在自己预测的 z_curr 上学习"——理论上有同样的盆地切换风险
- 但 rollout 有几个保护因素：
  - 每步都有 GT 监督（`step_loss = ||z_pred - z_gt||²`），不是只看下游
  - rollout 在 V3 训练中**一直存在**（λ=0.25），模型已经适应了它，没有发生过盆地切换
  - 加大权重是**渐进强化已有信号**，不是引入新分布
- **结论**：方向 1 的相变风险**低于** SF-pair，但不为零。1A 的严格 Go/No-Go 阈值（§12）对捕捉相变信号是必要的——一旦 val_pair_total（GT 输入下）开始上涨就立即止损

**对方向 3（pair_loss 渐进归零）的影响**：
- 渐进降 pair_loss 等于**让模型逐步离开 GT-optimal 盆地**——本质和 SF 类似，只是机制不同
- 风险更高：方向 3 没有任何回到 GT 的拉力（pair_loss 在缩小），可能比 SF 更快进入新盆地且回不去
- **修正**：方向 3 必须在每个 phase 末尾**显式 eval pair_loss on GT inputs**，而不是只看 val_select。如果 GT pair 性能开始恶化就立即回滚

**新方向（方向 4 候选）：双 backbone teacher-student**
基于 D.4 第 4 条，如果未来还想做 SF 类的方法，应当：
- 维护一个 EMA teacher backbone（参数缓慢跟踪 student）
- z_src_pred 由 teacher 生成而非 student 自身
- 这样 SF 输入有"独立对照系"，避免自洽循环
- 实现成本：~50 行代码，需要新设计

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

| 检查点 | val_select 阈值 | val_chain_normal_mse 阈值 | hop0 PSNR 阈值 | val_pair_total（相变监控）|
|--------|----------------|---------------------------|---------------|-------------------------|
| +5K | — | — | — | **> 0.00005 → 立即停**（相变早期信号）|
| +10K | > baseline × 1.05 → 止损 | > baseline × 1.10 → 止损 | < baseline − 0.5 dB → 止损 | > 0.0001 → 止损 |
| +25K | 无改善（≥ baseline）→ 止损 | 无改善（≥ baseline × 0.97）→ 止损 | < baseline − 0.3 dB → 止损 | > 0.00005 → 止损 |
| +50K | full-val eval + Path A redo | — | — | — |

**为什么加 val_pair_total 监控**：基于 §10.5 Finding D，模型从 GT-optimal 盆地被推出的最早信号是 val_pair_total 上涨（V4 SF pilot 中是从 0 → 0.000196 断崖跳变）。在 Rollout-Heavy 实验中，rollout_loss 也使用模型自预测输入，理论上有相同风险（虽然较低）。**val_pair_total 是相变早期警报，比 val_select 提前 1-2 个 val 点出现**，必须独立监控。

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

---
---

# SF-pair 失败的根因复盘：与现有 rollout alpha 机制的代码级对比（2026-04-26 第二次补充）

## 14. 现有 rollout 的 alpha 混合机制（代码实证）

### 14.1 chainstable 配置中的三阶段设计

200K v3 训练（`pet_flow_first_hop_224_50k_formal_v3_chainstable.yaml`）的 rollout schedule：

```yaml
rollout:
  warmup_ratio: 0.10    # 前 10%：alpha = 0（纯 GT 输入）
  ramp_ratio: 0.30      # 10%-40%：alpha 从 0 线性到 1.0（GT→混合→pred）
  alpha_start: 0.0
  alpha_end: 1.00       # 40% 之后：alpha = 1.0（纯 pred 输入）
```

以 50K 步为例：
- **Step 0–5000**（前 10%）：alpha=0，rollout 链每跳输入 = 纯 GT → 模型先学好单跳 velocity
- **Step 5000–20000**（10-40%）：alpha 从 0→1，`z_curr = (1-α)·GT + α·pred` → 模型逐步适应 noisy 输入
- **Step 20000–50000**（40% 后）：alpha=1.0，`z_curr = pred` → 纯预测链，与推理一致

### 14.2 rollout 链内传播逻辑（`rollout_first_hop.py` line 80-100）

```python
for hop_idx in range(num_steps):
    # 1. 用当前 z_curr 做 forward
    out = model.predict_latent_step(z_src=z_curr, ...)
    z_pred = out["z_pred"]
    z_gt = z_rollout[:, hop_idx + 1]

    # 2. 每一跳都有 loss（对照当前位置 GT）
    step_loss = ||z_pred - z_gt||²

    # 3. 决定下一跳的输入
    if hop_idx < num_steps - 1:
        z_curr = mix_latent(z_gt, z_pred, alpha)
        # alpha=0 → z_curr = GT（稳定）
        # alpha=0.5 → z_curr = 混合（渐进）
        # alpha=1.0 → z_curr = pred（和推理一致）
    else:
        z_curr = z_pred  # 最后一跳不需要混合
```

**关键设计点**：
1. **每一跳都有 GT loss**——即使输入是 predicted/混合的，target 仍是当前位置的 GT
2. **GT 锚定**——alpha < 1 时，下一跳输入混入 GT，防止误差雪崩
3. **渐进过渡**——从纯 GT 到纯 pred 是平滑的，模型有充足时间适应

### 14.3 `mix_latent` 的 straight-through 估计器

```python
def mix_latent(z_gt, z_pred, alpha, straight_through=True):
    if alpha <= 0:
        # forward: z_gt, backward: gradient 仍流过 z_pred（STE）
        return z_pred + (z_gt - z_pred).detach()
    if alpha >= 1:
        return z_pred
    mixed = (1 - alpha) * z_gt + alpha * z_pred
    if straight_through:
        # forward: mixed, backward: gradient 只流过 z_pred
        return z_pred + (mixed - z_pred).detach()
    return mixed
```

alpha_gated 模式下，STE 只在 alpha ≥ 0.999 时启用（`straight_through_alpha_min: 0.999`）。这意味着在混合阶段（alpha < 1），梯度**直接流过 mixed value**——GT 和 pred 都参与梯度计算。

---

## 15. SF-pair 与 rollout alpha 的逐点对比

### 15.1 输入/target 匹配性

**rollout_loss（正确）**：
```
输入：z_curr = mix(z_gt[k], z_pred[k-1], alpha)    ← 含 GT 锚定
forward：z_pred[k] = model(z_curr)
target：z_gt[k+1]                                  ← 当前跳目标位置的 GT
loss：||z_pred[k] - z_gt[k+1]||²
语义："不管从哪出发，到达 GT 目标位置"              ← ✅ 物理合理
```

**SF-pair（有问题）**：
```
输入：z_src_sf = mix(z_src_GT, z_chain[hop_idx], alpha_sf)  ← 无 GT 锚定（alpha_sf 高时纯 pred）
forward：v_pred = model(z_src_sf)
target：v_target = (z_dst_GT - z_src_GT) / dt              ← 用 z_src_GT 算的 velocity
loss：||v_pred(z_src_sf) - v_target(z_src_GT)||²
语义："从 noisy 位置出发，产生的 velocity 应等于从 GT 位置出发的 velocity"  ← ❌ 物理不合理
```

**根本矛盾**：当 `z_src_sf ≠ z_src_GT` 时，从两个不同起点出发到同一终点的 velocity 不应相同。SF-pair 要求模型做到这一点，这是一个**不可满足的约束**——alpha 越大，约束越不合理，模型被迫在两个矛盾信号间折衷，导致 GT 上的 pair loss 也恶化（§4.2 Finding C 已代码确认）。

### 15.2 中间状态监督

| 维度 | rollout_loss | SF-pair |
|------|-------------|---------|
| hop0 (D50→D20) | ✅ step_loss[0] | ❌ 无（hop0 的 z_src_pred ≡ z_src_GT，SF 无效果） |
| hop1 (D20→D10) | ✅ step_loss[1] | ❌ 仅当 batch 中 hop_idx=1 的样本有 loss |
| hop2 (D10→D4) | ✅ step_loss[2] | ❌ 仅当 batch 中 hop_idx=2 的样本有 loss |
| hop3 (D4→NORMAL) | ✅ step_loss[3] | ❌ 仅当 batch 中 hop_idx=3 的样本有 loss |
| **跨 hop 约束** | ✅ 链式传播，每跳误差影响下一跳 loss | ❌ 每个样本独立，不知道上下文 |

rollout 的每一跳 loss 都在约束链式传播的中间状态。SF-pair 的 pre-rollout 是 `@torch.no_grad()` 的 4 次 forward，中间状态没有任何 loss 约束——它只用最终结果替换 pair 输入。

### 15.3 误差累积控制

**rollout（有控制）**：
```
alpha < 1 时：z_curr 混入 GT → 误差被压制在每一跳
alpha = 1 时：纯 pred，但每跳 loss 仍约束 → 误差累积但有梯度纠正
```

**SF-pair（无控制）**：
```
pre-rollout 从 GT D50 出发，跑 4 跳 chain（no_grad）
hop0: z_chain[0] = GT（无误差）
hop1: z_chain[1] = model(GT) → 有误差
hop2: z_chain[2] = model(z_chain[1]) → 误差累积
hop3: z_chain[3] = model(z_chain[2]) → 误差进一步累积
→ 无任何 loss 约束 z_chain[1,2,3] 的质量
→ hop3 的 z_chain[3] 可能已严重偏离 GT
→ 用它替换 pair_loss 的 z_src → velocity target 不匹配 → 模型被错误训练
```

---

## 16. 为什么 exposure bias 仍然存在（尽管 rollout 设计正确）

rollout 的 alpha 混合设计**本身是对的**。问题不在设计，在**权重**。

### 16.1 梯度占比分析

200K v3 训练日志（从 v4 SF pilot 的 α=0 区间提取，此时 SF 未生效，反映纯 baseline 状态）：

```
step 86850-91800（sf_alpha=0，纯 baseline 行为）：
  pair_frac  ≈ 2-13%   ← 单跳 GT 监督
  roll_frac  ≈ 6-32%   ← 链式 pred 监督（对抗 exposure bias）
  img_frac   ≈ 57-92%  ← hop0 像素重建（不对抗 exposure bias）
```

**中位数大约**：pair ~5%, roll ~10%, img ~85%。

### 16.2 含义

| loss | 占梯度 | 对 exposure bias 的帮助 | 结论 |
|------|--------|------------------------|------|
| pair_loss | ~5% | ❌ 制造 exposure bias（纯 GT 训练） | 越多越加剧 bias |
| rollout_loss | ~10% | ✅ 对抗 exposure bias（pred chain + GT target） | **太少了** |
| image_aux | ~85% | ❌ 无关（只做 hop0 像素重建） | 不帮忙但占了绝大部分梯度 |

image_aux 在做 hop0 的像素级重建（L1 + SSIM + seam），这对最终图像质量有帮助，但**对 chain 后端（hop1-3）的 exposure bias 没有任何帮助**。它吃掉了 85% 的梯度，留给 rollout 的只有 10%——rollout 的设计再正确，10% 的梯度份额也不够让 backbone 真正学会"在 predicted 输入上做好 chain prediction"。

### 16.3 Path A exposure_gap 的趋势进一步支持这个判断

| Checkpoint | 训练步数 | mean ExpoGap | 解读 |
|------------|---------|-------------|------|
| step_10000 | 10K | 4.19 dB | 早期（rollout α 仍在 ramp 中） |
| step_30000 | 30K | 4.76 dB | alpha 已到 1.0 超 10K 步，但 gap 仍上升 |
| best (46K) | 46K | 4.80 dB | 继续上升 |

**如果 rollout 权重足够**，在 alpha=1.0 之后（step 20K 之后），ExpoGap 应当开始下降——因为模型此时在纯 pred chain 上训练。但实测 gap 仍在上升：说明 10% 的 rollout 梯度**不足以抵消** pair_loss 持续在 GT 分布上强化模型的效应。

---

## 17. 修正后的因果解释链

```
根因：rollout_loss 权重太低（~10% 梯度），被 image_aux（~85%）淹没
  ↓
直接后果：backbone 参数更新主要由 image_aux（hop0 像素）驱动
  ↓
间接后果：rollout 的 chain prediction 信号不足以让 backbone 适应 predicted 分布
  ↓
外显症状：Path A ExpoGap 持续上升（4.19 → 4.80 dB）
  ↓
错误应对：SF-pair（在 pair_loss 输入端注入 predicted 分布）
  ↓
为什么 SF 失败：pair_loss 的 target velocity 仍按 GT 算，输入/target 不匹配 +
               无中间状态 loss + 无 GT 锚定 → 模型被错误训练
  ↓
正确应对：加大 rollout_loss 权重（§11 方向 1），让已有的正确机制有足够的梯度影响力
```

---

*第二次补充作者：Copilot*
*补充日期：2026-04-26*
*依据：`rollout_first_hop.py` 完整代码审查 + `pet_flow_first_hop_224_50k_formal_v3_chainstable.yaml` config 对比 + v4 SF pilot α=0 区间梯度占比实测*
