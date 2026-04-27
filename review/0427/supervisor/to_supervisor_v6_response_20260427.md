# V6 Supervisor Review 回复

**日期**：2026-04-27
**回复对象**：[`v6_plan_supervisor_review_20260427.md`](v6_plan_supervisor_review_20260427.md)
**关联文档**：
- [`review/plan/transport_breakthrough_research_v6.md`](../../plan/transport_breakthrough_research_v6.md)（已更新）
- [`train_first_hop.py`](../../../train_first_hop.py)（已修改）
- [`review/0427/logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl`](../logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl)

---

## 0. 总体回应

感谢 Supervisor 的细致审查。6 个问题逐一回复如下，附数学论证。

| # | 问题 | Supervisor 判定 | 我方决策 | 理由简述 |
|:-:|---|:-:|:-:|---|
| 1 | image_aux λ 固定 0.04 | 🔴 Critical | **不预设退火，加 +130K 监控** | 因果反转：V3 img_raw 上升是 λ_img=0.12 的后果，非前提 |
| 2 | pair_weight=15 冷启动 | 🔴 Critical | **不 ramp，加 +1K 安全检查** | AdamW 尺度不变性保证 |
| 3 | §1.5 预测用错 step | 🔴 Critical | **降级为 V5 归因问题** | V6 from scratch 不能用 V3 raw loss 外推 |
| 4 | Phase I 过长 | 🟡 Important | **保持原设计** | 此前已论证 |
| 5 | step_weights[0]=0.5 矛盾 | 🟡 Important | **保持 0.5，修正措辞** | rollout[hop0] 非完全退化 |
| 6 | lambda_scale_mode: none | 🟡 Important | **pilot 保持 none** | 更简单，风险可控 |

**已完成的代码改动**：`loss.pair_weight` 在 `train_first_hop.py` 中实现（7 行），默认 1.0 向后兼容。

---

## 1. image_aux λ 固定 0.04 — 不预设退火

### 1.1 Supervisor 的论据

用 V3 step 172600 的 `img_raw=9.40e-3` 代入 V6 权重，得到 img_frac=36.7%，超过 25% 红线。

### 1.2 因果链分析

Supervisor 的推理隐含一个假设：**V6 的 img_raw 会像 V3 一样在后段上升**。但 V3 后段 img_raw 上升有明确的因果链：

```
V3: λ_img=0.12（大）
  → image 梯度占 93%（主导）
    → backbone 追像素高频细节
      → latent 偏离 transport manifold
        → transport loss 下降更慢
          → image 占比继续上升（正反馈）
            → img_raw 从 2.51e-3 涨到 9.40e-3
```

V6 切断了这个正反馈的**起点**：

```
V6: λ_img=0.04（V3 的 1/3），pair_weight=15
  → Phase I: pair 占 ~80%，image 占 ~20%
    → backbone 主要服务 transport
      → latent 保持在 transport manifold 上
        → 正反馈循环不成立
```

**关键区别**：V3 的 λ_img=0.12 给了 image 通道**从一开始就足够大的梯度预算**去主导 backbone，这是正反馈的前提。V6 的 λ_img=0.04 + pair_weight=15 意味着 image 从来没有机会主导。

### 1.3 两个场景的数学推导

**场景 A：V6 transport-first 成功（transport 保持在 V3-best 水平）**

假设 pair_raw ≈ 7e-5，roll_raw ≈ 7e-4（与 V3 step 86800 一致）：

$$W_{transport} = 15 \times 7 \times 10^{-5} + 4 \times 7 \times 10^{-4} = 3.85 \times 10^{-3}$$

img_frac 达到 25% 的临界 img_raw：

$$\frac{0.04 \times L_i^{crit}}{3.85 \times 10^{-3} + 0.04 \times L_i^{crit}} = 0.25 \implies L_i^{crit} = \frac{0.25 \times 3.85 \times 10^{-3}}{0.75 \times 0.04} = 3.21 \times 10^{-2}$$

V3 最差 img_raw = 9.40e-3，仅为临界值的 **0.29×**。**即使 img_raw 翻倍也远低于 25%。**

**场景 B：V6 transport 坍塌（降到 V3 后段水平）**

假设 pair_raw ≈ 1.4e-5，roll_raw ≈ 1.2e-4：

$$W_{transport} = 15 \times 1.4 \times 10^{-5} + 4 \times 1.2 \times 10^{-4} = 6.90 \times 10^{-4}$$

$$L_i^{crit} = \frac{0.25 \times 6.90 \times 10^{-4}}{0.75 \times 0.04} = 5.75 \times 10^{-3}$$

此时 V3 最差 img_raw（9.40e-3）确实超过临界值 1.63×，img_frac ≈ 37%。

**但**：如果 transport 坍塌到 V3 后段水平，说明 V6 的 transport-first 策略本身失败了——此时应停止实验分析原因，而非试图用 image 退火补救一个已经失败的策略。

### 1.4 决策

| 决策 | 依据 |
|---|---|
| **不预设 image_aux 退火 schedule** | 退火解决的是"transport 成功后 image 仍主导"的场景，但数学推导表明该场景下 img_frac < 3% |
| **在 Go/No-Go 加 +130K img_frac 检查** | 若 img_frac > 30% 连续上升，说明 transport 已坍塌，应停止实验 |

### 1.5 V3 实测数据补充

| step | pair_raw | roll_raw | img_raw | V6 权重下 img_frac |
|---:|---:|---:|---:|---:|
| 86800 | 7.16e-5 | 6.90e-4 | 2.51e-3 | **2.6%** |
| 130000 | 1.11e-5 | 1.39e-4 | 5.35e-3 | 22.9% |
| 150000 | 1.36e-5 | 1.38e-4 | 5.08e-3 | 21.2% |
| 172600 | 1.48e-5 | 1.07e-4 | 9.40e-3 | 36.6% |

注意：130K-172K 行对应的是 **V3 transport 已经坍塌后的状态**（pair_raw 从 86800 的 7.16e-5 降到 1.48e-5，下降 4.8×）。V6 的设计目标正是防止这种坍塌。如果 V6 成功，这些行不适用；如果 V6 失败出现相同坍塌，则实验本身需要重新设计。

---

## 2. pair_weight=15 冷启动 — 不需要 ramp

### 2.1 Supervisor 的论据

> random init 下 velocity head 输出 ~ N(0, σ²)，pair_weight=15 让有效 LR 拉到 ~1.2e-3

### 2.2 AdamW 尺度不变性

AdamW 的参数更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

对 loss 乘以常数 $c$（即 pair_weight）：梯度 $g' = c \cdot g$

$$\hat{m}'_t = c \cdot \hat{m}_t, \quad \hat{v}'_t = c^2 \cdot \hat{v}_t$$

$$\Delta\theta' = \eta \cdot \frac{c \cdot \hat{m}_t}{\sqrt{c^2 \cdot \hat{v}_t} + \epsilon} = \eta \cdot \frac{c \cdot \hat{m}_t}{|c| \sqrt{\hat{v}_t} + \epsilon}$$

当 $|c| \sqrt{\hat{v}_t} \gg \epsilon$（$\epsilon = 10^{-8}$）时：

$$\Delta\theta' \approx \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}} = \Delta\theta$$

### 2.3 数值验证

V3 step 50 的 pair_raw = 1.01e-4。乘以 pair_weight=15 得 1.52e-3。

pair 梯度尺度 $|g| \sim \sqrt{L_p} \approx \sqrt{1.52 \times 10^{-3}} \approx 0.039$。

$$|c| \sqrt{\hat{v}_t} \sim 0.039 \gg \epsilon = 10^{-8}$$

AdamW 完全处于尺度不变区域。**pair_weight=15 不会改变步长大小，只改变梯度方向**——让 pair 通道占 ~80% 的方向权重（vs V3 的 ~50%），这正是设计意图。

### 2.4 关于 β₂ bias correction

Supervisor 提到"AdamW β₂=0.999 二阶动量未收敛期"。但 bias correction 恰好处理了这一点：

$$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{\beta_2 \cdot 0 + (1-\beta_2) \cdot g_1^2}{1 - \beta_2} = g_1^2$$

从 step 1 开始，$\hat{v}_1 = g_1^2$，$\sqrt{\hat{v}_1} = |g_1|$，更新量 $\approx \eta \cdot \text{sign}(g_1)$，与 loss scale 完全无关。

### 2.5 决策

| 决策 | 依据 |
|---|---|
| **pair_weight=15 从 step 0 启用，不 ramp** | AdamW 尺度不变性 + bias correction 保证 |
| **加 +1K Go/No-Go：pair_loss < 5e-4 且无 NaN** | 纯安全网，预期不会触发 |

---

## 3. §1.5 预测用错 step — 降级为 V5 归因问题

### 3.1 Supervisor 的论据

> §1.5 用 V3 step 86800 raw loss 预测 V6 Phase III 分布，得到 image=3%；实际 Phase III 应用 V3 step 150K-172K 数据，得到 image=37%。差距 12×。

### 3.2 为什么这是 V5 归因问题而非 V6 设计缺陷

V6 是 **from scratch 200K 训练**，有完全不同的权重配方：

| 参数 | V3 | V6 |
|---|---|---|
| pair_weight | 1.0 | **15.0** |
| λ_roll 范围 | 0.02→0.25 | **0→4.0** |
| λ_img 范围 | 0.005→0.12 | **固定 0.04** |
| alpha 0→1 窗口 | 20K-80K | **50K-150K** |

V3 后段的 raw loss 轨迹是在 V3 特定权重配方下产生的。用 V3 的 raw loss 直接代入 V6 的 λ 做 fraction 预测，等同于假设 **V6 的 raw loss 轨迹与 V3 完全相同**。但 V6 的核心改动恰恰就是要改变这个轨迹。

具体地：V3 后段 pair_raw 从 7.16e-5 降到 1.48e-5（4.8×），是因为 V3 的 image 主导让 backbone 偏离了 transport manifold。V6 用 pair_weight=15 防止这种偏离，所以 V6 的 pair_raw 后段轨迹预期会与 V3 截然不同。

这是一个**二级归因问题**：只有在 V6 pilot 或 full 实验跑完后，才能回答"V6 的 raw loss 后段是否像 V3 一样恶化"。在实验之前用 V3 数据做外推是不可靠的。

### 3.3 决策

| 决策 | 依据 |
|---|---|
| §1.5 保留 step 86800 作为 **V6 Phase II 中段**的参考 | 86800 的 transport 活跃状态更接近 V6 设计目标 |
| 不用 V3 150K-172K 数据"修正"预测 | 那些数据反映的是 V3 失败状态，不是 V6 的预期轨迹 |
| **V6 Phase III 的实际 fraction 由 pilot 实测确定** | +130K Go/No-Go 检查点 |

---

## 4. Phase I（0-50K）GT-only — 保持原设计

此前已论证。V6 延长 GT-only 到 50K 是有意设计：让模型在纯 GT 输入下充分学习 velocity 精度，避免过早引入 pred-input 噪声干扰初期学习。

---

## 5. step_weights[0]=0.5 — 保持，修正措辞

### 5.1 Supervisor 的论据

> §1.4 说 rollout[hop0] 与 pair[hop0] "退化重叠"，但还给了 0.5 权重。

### 5.2 精确区分

| 通道 | hop0 输入 | 监督方式 | 信息内容 |
|---|---|---|---|
| pair[hop0] | GT segment 随机 (t_src, t_dst) | velocity + endpoint | **多子段**监督 |
| rollout[hop0] | 固定 t_D50 → t_D20（全步） | endpoint MSE only | **全步 endpoint** 一致性 |

pair 在 mini-batch 中随机采样时间子段，不一定每 step 都覆盖全步。rollout[hop0] 保证每 step 都有**完整段** D50→D20 的 endpoint 校验，这是 pair 的 endpoint 项未必覆盖的。

但两者确实高度重叠——pair 的 endpoint 项（`endpoint_weight=1.0`）也在做类似的事。所以 0.5/5.0 = 10% 的 rollout 预算分给 hop0，影响极小，不值得为此改动代码。

### 5.3 决策

| 决策 | 依据 |
|---|---|
| **保持 step_weights[0]=0.5** | 10% 预算，影响边际 |
| **V6 plan §1.4 措辞修正** | "退化重叠" → "高度重叠，仅保留全步 endpoint 一致性辅助" |

---

## 6. lambda_scale_mode: none — pilot 保持

### 6.1 两种模式对比

V6 的 alpha 和 lambda_roll 都线性 ramp（50K-150K）：

| step | α | λ_roll (none) | λ_roll (alpha) |
|---:|---:|---:|---:|
| 60K | 0.1 | 0.4 | 0.04 |
| 80K | 0.3 | 1.2 | 0.36 |
| 100K | 0.5 | 2.0 | 1.0 |
| 150K | 1.0 | 4.0 | 4.0 |

mode=alpha：$\lambda_{eff} = \lambda_{base}(t) \times \alpha(t)$，两个线性的乘积 = **二次增长**。

优点：50K-80K 区间（α<0.3）rollout 信号与 pair 高度同源时，λ_eff 很小，不浪费梯度预算。
缺点：rollout 到 100K 才有实质权重（λ_eff=1.0），有效 ramp 窗口缩短。

### 6.2 决策

| 决策 | 依据 |
|---|---|
| **pilot 保持 lambda_scale_mode: none** | 更简单，行为更可预测 |
| 若 pilot 中 60-80K 区间 roll_frac > 30% 且 α < 0.3 | 全量实验考虑切换为 alpha 模式 |

---

## 7. 已完成的代码改动

`loss.pair_weight` 在 [`train_first_hop.py`](../../../train_first_hop.py) 中实现：

| 位置 | 改动 | 行号 |
|---|---|---:|
| config 解析 + 有效性校验 | `pair_loss_weight = float(cfg["loss"].get("pair_weight", 1.0))`，含 `math.isfinite` 和非负检查 | L1361-1365 |
| total_loss | `pair_loss_weight * pair_losses["total"]` | L1919 |
| logging fraction | `pair_weighted = pair_loss_weight * pair_losses["total"]` | L1936 |
| metrics JSONL | `"pair_loss_weight": pair_loss_weight` | L2138 |

默认值 1.0，对所有现有 config 完全向后兼容。

---

## 8. V6 plan 已更新内容

| 更新项 | 位置 |
|---|---|
| §2 代码改动：改为实际 7 行实现 | §2 |
| §1.4 措辞："退化重叠" → "高度重叠，仅保留 endpoint 一致性辅助" | §1.4 |
| Go/No-Go：+1K 冷启动安全检查 | §4 |
| Go/No-Go：+130K img_frac 上升趋势检测 | §4 |

---

## 9. 下一步

1. **创建 V6 config** — 从 V3 200K config 派生，填入 V6 全部参数
2. **dry-run 50-200 步** — 验证 pair_loss_weight 日志正确、无 NaN
3. **启动 V6-P1 pilot（50K）** — 直接用正式权重，不需要保守 P0
4. **并行**：V5 full-val / null-control 归因实验继续运行，结果不阻塞 pilot
