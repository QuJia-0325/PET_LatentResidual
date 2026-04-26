# V4 SF Pilot 早期收敛分析（提前终止）

**日期**：2026-04-26
**对象**：`first_hop_224_v4_sf_pilot`（resume from V3 200K transport `best.pt @ step=86800`）
**日志**：`review/0426/logs_train/v4_sf_pilot_gpu2.log`（266 行，覆盖 step 86850 → 95250）
**决策**：在 sf_alpha 仅爬到 ~0.345（约目标 1.0 的 1/3）时**提前终止**——已经能看清趋势，无需消耗剩余 GPU 时长。

---

## 1. 结论摘要（TL;DR）

| 维度 | 状态 | 证据 |
|---|---|---|
| V3.1 P1 修复（resume_relative SF schedule）| ✅ **生效** | step 86850–91800 共 5000 步内 `sf_alpha=0.000`，从 step 91850 开始 ramp |
| V3.1 P2 修复（stdout 暴露 sf_alpha/sf_gap）| ✅ **生效** | 每行 `[train]` 末尾均含 `sf_alpha=… sf_gap=…` |
| SF 在低 alpha（≤0.10）阶段是否有收益 | 🟡 **边际正向** | V4 度量下：V3 基线 0.000567 → SF α≈0.10 时 best 0.000538（−5.1%） |
| SF 在中高 alpha（≥0.15）阶段是否仍正向 | ❌ **明确负向** | val_select 从 0.000538 单调劣化至 0.001323（α≈0.30 时已 +146%） |
| sf_gap 是否随训练收敛 | ❌ **不收敛** | 0.002–0.007 区间持续震荡，与 alpha 上升无单调关系 |
| **总体判断** | ❌ **当前 SF-pair 设计不能作为 transport 突破方向** | 与 V3 SF run 完全相同的恶化模式重现 |

---

## 2. 度量对齐（关键说明）

**V3 与 V4 的 `val_multi_objective` 权重不同**，原始数值不可直接比较：

```
V3 weighting:  0.5*d20 + 0.45*d10 + 0.9*d4 + 1.5*normal
V4 weighting:  0.15*d20 + 0.45*d10 + 0.9*d4 + 1.5*normal   ← d20 权重 0.5 → 0.15
```

V4 pilot 是从 V3 best.pt（step=86800）resume 的，因此 V4 日志里 step=87200（sf_alpha=0）的 val_select_score=**0.000567** 就是 **V3 模型在 V4 度量下的真实基线**——这是公平对比的起点，无需另外重新评估。

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

**Finding A — SF 在 α ∈ [0, 0.10] 时有边际正收益**
- V3 baseline @ V4 metric: 0.000567
- V4 best @ α≈0.10: 0.000538
- **相对改进 5.1%**（绝对值 −0.000029）
- 全部四档 chain MSE 都改善：d20 −10.7%, d10 −8.8%, d4 −5.3%, normal −2.9%
- 这是真实但非常微弱的信号，可能仅为 finetune noise 范围

**Finding B — α ≥ 0.15 后 val 进入震荡式恶化**
- 93200（α=0.14）已开始反弹（0.000551）
- 93600（α=0.18）暴涨至 0.000939（+74% vs best）
- 此后在 0.000654–0.001323 大幅震荡，再无任何接近 best 的点
- 模式与 V3 SF run（旧 schedule bug 下 α=1.0）的"30% 全面退化"高度一致

**Finding C — `val_pair_total` 在 step 93200 出现"质变跳跃"**
- step ≤ 92800：`val_pair_total = 0.000000`（完全不参与 val loss）
- step 93200 起：跳到 0.0002 量级并保持
- 时间点正好在 sf_alpha 越过某个阈值（约 0.10–0.14 之间）时出现
- 推测 val 阶段也走 `compute_self_forcing_z_src` 路径，pair 损失被 SF 偏移污染——这本身**不是 bug**，但说明 SF 已经把模型预测分布推离 GT 分布到肉眼可见的程度

### 4.3 sf_gap 不收敛的负面信号

`sf_gap = ||z_pred − z_src_GT||` 在整段训练中保持 0.002–0.007 震荡，**与 sf_alpha 上升无任何收敛趋势**。

如果 SF loss 真在驱动模型把 rollout 预测拉近 teacher 分布，gap 应当随训练单调下降；现状说明：

- 要么 `sf_gap` 计算口径有问题（仅看绝对值，未归一化）；
- 要么模型确实没在学习消化 SF 信号——它只是被 SF loss 推得越来越偏，而 image / rollout 损失在反向牵引，最终两边都没赢。

---

## 5. 与 V3 SF run 的对照

| 维度 | V3 SF run（旧 schedule bug）| V4 SF pilot（修复后）|
|---|---|---|
| sf_alpha 起点 | 立刻 = 1.0（bug） | 0.0（正确） |
| 有效新训练步数 | ~3600（resume 后即结束）| ~8400（终止时） |
| val 最终结果 vs baseline | **−30%**（全面恶化） | α≤0.10 时 +5%，α≥0.15 后 **−74%~−146%** |
| 失败模式 | "瞬间冲击" | "渐进恶化" |
| 是否能区分 schedule bug 与方法本身缺陷 | ❌ 不能 | ✅ 能——**方法本身在 α 偏高时即失效** |

**核心信息**：V3.1 的 schedule 修复消除了"实验结论无效"的混淆变量，**但 SF-pair 损失本身在当前权重 / 时序下不能跨越 transport gap**。

---

## 6. 假设排查表

| 假设 | 是否成立 | 证据 |
|---|---|---|
| H1：V3 SF 失败仅因为 schedule bug | ❌ **被证伪** | V4 修复后仍在 α≥0.15 失败 |
| H2：低 alpha 的 SF 有微小正收益 | 🟡 **弱支持** | 5% 改进，但样本仅 1 个 ckpt，不能排除 noise |
| H3：SF-pair loss 形式本身设计有缺陷 | 🟡 **强烈怀疑** | sf_gap 不收敛 + alpha 一上去 val 就坏 |
| H4：sf_gap 度量本身有问题 | 🟡 **可能** | 需要看 gap_norm 的归一化逻辑 |
| H5：transport gap 不应通过"换 z_src"解决 | 🟡 **倾向支持** | 替换 pair 输入只改变 distribution shift，不直接改进 hop-wise velocity 学习 |

---

## 7. 下一步建议（路线选择）

### 选项 A — **放弃 SF-pair 路线**（推荐，置信度高）
SF 在 V3 和 V4 两次都失败，且失败模式从"schedule bug"迁移到"方法缺陷"后仍存在。继续在这个方向调参（更小 ramp_end、更慢 ramp、加 EMA 等）边际收益预期低，建议把 GPU 预算投到其他 idea。

**对 v4 plan 的影响**：`review/plan/transport_breakthrough_research_v4.md` 里如果还把 SF-pair 列为主推方向，需要降级或剥离。

### 选项 B — **小剂量 SF 限幅版**（仅当你想做完整性 ablation 时）
- `alpha_sf_end = 0.10`（不要 1.0）
- `ramp_steps = 30000`（更慢）
- 训 30K 步看是否能稳定保持当前 best 0.000538
- 风险：即使成功，也只是 5% 改进，不足以作为论文主卖点

### 选项 C — **重新设计 SF 形式**
当前 SF-pair 把 GT `z_src` 换成模型自己的 `z_pred`，本质上是 input-side perturbation，没有显式约束 hop-velocity 学习。可考虑：
- 把 SF 信号施加在 velocity 输出端而非 input 端
- 引入 stop-grad 的 EMA teacher 而非当前 step 的自预测
- 但这等于做新 idea，需要重新走 plan/discuss/execute 流程

---

## 8. 立即行动项

1. ✅ 已完成：服务器 V4 SF pilot 提前终止
2. ⏳ 本报告归档至 `review/0426/`
3. ⏳ 更新 `review/plan/transport_breakthrough_research_v4.md`：在 Idea 1 (SF-pair) 标注"已证伪"，根据用户决策选择 A/B/C
4. ⏳ 向 reviewer 通报：V3.1 schedule fix 验证成功，但方法本身需要重新设计；当前没有候选方案能突破 V3 200K transport 基线

---

## 9. 数据可信度声明

- 训练步数：~8450 effective steps（86800 → 95250），SF 真正生效仅 ~3400 步
- val 频率：每 400 步一次，覆盖 21 个 val 点
- 单一 seed、单一 GPU，无 N=3 重复
- **本报告结论的强度**：足以否决"继续按当前 v4 plan 跑完 50K"，但不足以发表"SF-pair 永远不行"的强声明；如需后者，需要至少 N=3 重复 + 多种超参网格

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

**本质上 SF-pair = input-side perturbation without correction signal**

### 10.3 与 rollout_loss 的对比

| 维度 | pair_loss（含 SF） | rollout_loss |
|------|-------------------|-------------|
| 输入来源 | 单跳（GT 或 SF-predicted） | chain（纯 predicted，alpha=1.0） |
| 监督信号 | 单跳 velocity/endpoint vs GT | **每一步** z_pred vs GT 对应位置 |
| 梯度路径 | 只经过 1 次 forward | **4 次 forward**，每步都有 loss |
| 链式校正 | ❌ 无 | ✅ 有（step_losses 加权求和） |
| exposure bias 感知 | ❌ 不知道输入有误差 | ✅ 隐式知道（因为 z_curr 是预测值） |

**rollout_loss 才是直接对抗 exposure bias 的正确工具**——它已经在链式预测上计算 loss，天然包含"从 noisy 输入出发也能到达 GT"的学习信号。

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

**配置变化**：

| 参数 | 当前值 | 提议值 | 理由 |
|------|--------|--------|------|
| `rollout.lambda_start/end` | 0.25 | **2.0** (8×) | rollout 是唯一直接对抗 exposure bias 的 loss |
| `image_aux.lambda_start/max` | 0.12 | **0.04** (降 3×) | 减少 image 对梯度的主导地位 |
| `rollout.step_weights` | [1.0, 1.1, 1.2, 1.3] | 保持不变或微调 | 先验证权重拉升效果 |

**预期效果**：
- `roll_frac` 从 7-15% → 40-60%
- `img_frac` 从 80-90% → 20-30%
- backbone 梯度将主要来自 chain prediction，而非 hop0 像素重建

**风险**：
- rollout_loss 过大可能压制 pair_loss → 单跳 velocity 精度下降
- image_aux 降太低 → hop0 像素质量退化
- **缓解**：监控 `pair_frac` 不低于 2%，`val_chain_d20_mse` 不恶化

**代码改动**：0 行。纯 config。

**验证成本**：~30h（50K 步 from 200K v3 best resume）。

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

### 方向 3：Rollout-Only Fine-tuning Phase（需少量代码改动）

**核心思路**：pair_loss 是 exposure bias 的**根源**（88% 梯度来自 GT-only 训练）。在 200K 标准训练后，进入 Phase 2，**完全去掉 pair_loss**，只用 rollout_loss + image_aux 训练。

**理由**：
- 去掉 pair_loss 后，模型只在链式预测上学习，**从根源消除 exposure bias**
- rollout_loss 的每一步 `step_loss = ||z_pred - z_gt||²` 本身就是单跳监督——只是输入是预测值而非 GT
- image_aux 保留以稳定 hop0 的像素质量

**风险**：
- 没有 pair_loss 的单跳监督，模型可能"忘记" velocity 预测精度
- 但 rollout 每步 loss 隐含了单跳监督——只是输入分布不同

**实现方式**：

```yaml
# 最简实现（0 代码改动）：
transport:
  pair_sample_probs: [0.0, 0.0, 0.0, 0.0]  # pair 不再采样
# 或新增 flag：
training:
  pair_loss_enabled: false   # 需 ~5 行代码
```

需确认 `pair_sample_probs = [0,0,0,0]` 在 dataloader 中是否会报错——如果不行，加一个 `pair_loss_weight: 0.0` 乘在 total_loss 中即可。

**验证成本**：~30h（50K 步）。

---

## 12. 综合实验计划

### 第一优先：方向 1+2 组合（"Rollout-Heavy"）

```yaml
# 新 config: pet_flow_first_hop_224_v4_rollout_heavy.yaml
# 改动 vs 200K v3 baseline:
rollout:
  lambda_start: 2.0      # 0.25 → 2.0 (8×)
  lambda_end: 2.0
  step_weights:
    - 0.5                 # hop0: 降低（无 exposure bias）
    - 1.0                 # hop1
    - 2.0                 # hop2
    - 4.0                 # hop3: 加重（exposure bias 最严重）
image_aux:
  lambda_start: 0.04      # 0.12 → 0.04 (降 3×)
  lambda_max: 0.04
# 其余所有参数与 200K v3 完全一致
# self_forcing_pair: { enabled: false }  ← 明确关闭 SF
```

从 200K v3 best resume，跑 50K 步。

**Go/No-Go 判定**：
- +10K: 若 val_select_score > baseline × 1.20 → 止损
- +25K: 若 val 趋势无改善 → 止损
- +50K: full-val eval + Path A redo

**成功标准**：
- `val_select_score < baseline`（chain 整体改善）
- Path A exposure_gap 下降 ≥ 10%（从 4.80 降到 ≤ 4.32 dB）
- `val_chain_normal_mse` 改善 ≥ 5%（末端 hop 受益最大）

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

## 13. 为什么"加大 rollout 权重"值得乐观

1. **Rollout loss 已经在做正确的事**——它在纯预测链上计算 loss（alpha=1.0），每步都有 GT 监督
2. **它只是被 image_aux 淹没了**——80-90% 梯度去做像素重建，对 transport chain 帮助有限
3. **v3 C'（λ_roll=1.0）的梯度特征**：虽然只跑了 3600 步，但 `grad_backbone` 是 SF 版的 2.4×——说明加大 rollout 确实大幅改变了梯度信号
4. **Path A diagnostic 的原始建议**就是 "inspect rollout hyperparameters (alpha ramp, step weights) and the mid-chain contribution"——我们之前跳过了这个最简单的建议直接去做 SF
5. **Occam's Razor**：最简单的干预（调权重）应该先于复杂的新机制（SF、EMA teacher 等）

---

*补充分析作者：Copilot*
*补充日期：2026-04-26*
*依据：Path A 多 ckpt 数据 + v4 SF pilot 失败分析 + rollout_first_hop.py / train_first_hop.py 代码审查*
