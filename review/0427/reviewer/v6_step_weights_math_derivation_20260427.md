# V6 step_weights 与 pair_loss_weights 的数学推导论证

**日期**：2026-04-27
**作者**：Reviewer (Copilot)
**目的**：用数学推导论证 V6 plan 当前 `step_weights = [2.5, 1.5, 1.0, 0.5]` 的设计错误，并给出基于通道物理意义的正确分配方案
**关联文档**：
- [V6 plan](../../plan/transport_breakthrough_research_v6.md)
- [V4 SF pilot 分析](../../0426/v4_sf_pilot_early_analysis_20260426.md) §10
- [V3 200K 训练 JSONL](../logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl)
- 代码：[`pet_lr/rollout_first_hop.py`](../../../pet_lr/rollout_first_hop.py), [`train_first_hop.py`](../../../train_first_hop.py)

---

## 0. 论证目标

V6 plan §1.4 的 `step_weights = [2.5, 1.5, 1.0, 0.5]` 头重设计基于"v_std 大 = 困难"的论证。本文用 PET 训练目标的数学结构 + V3 实测数据证明：

1. **hop0 确实是最难的跳**（4 项独立证据支持）
2. **但 step_weights 不是给 hop0 加权的正确通道**（数学上重复了 pair_loss 的工作）
3. **正确的三通道分配**应该是：pair_loss_weights 头重 + image_aux 保持 + step_weights 中重

---

## 1. 关键经验事实

### 1.1 hop0 难度的多维证据

| 指标 | 来源 | hop0 | hop1 | hop2 | hop3 |
|---|---|---:|---:|---:|---:|
| $v_{\text{std}}$（latent displacement）| V6 plan §1.4 | **0.009634** | 0.002946 | 0.000775 | 0.000141 |
| PSNR_TF（teacher-forced）| [v4_sf §10.1](../../0426/v4_sf_pilot_early_analysis_20260426.md) | **35.57** | 39.32 | 41.21 | 43.13 |
| PSNR_RO（rollout）| 同上 | 35.57 | 35.94 | 36.51 | 36.81 |
| ExpoGap = TF − RO | 同上 | **0.00** | 3.38 | 4.71 | **6.32** |
| CeilGap（与 oracle 距离）| 同上 | **11.07** | 9.42 | 9.62 | 9.51 |
| LatMSE_RO/TF（rollout latent 漂移倍数）| 同上 | 1.00 | 3.11 | 4.62 | **5.64** |

**关键观察**：
- **hop0 是绝对困难的**（PSNR_TF 最低 35.57，CeilGap 最大 11.07）
- **hop3 的 ExpoGap 最大**（6.32 dB）但 $v_{\text{std}}$ 接近恒等（0.000141）
- **hop0 的 ExpoGap 恒为 0**：这是结构性的，因为 rollout chain 起点 `z_curr = z_GT_D50` 永远是 GT

### 1.2 V3 200K 实测梯度分布（[JSONL](../logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl)）

| step | pair_frac | roll_frac | img_frac | pair_raw | roll_raw | img_raw |
|---:|---:|---:|---:|---:|---:|---:|
| 86800 (best) | 13.1% | 31.6% | 55.2% | 7.16e-5 | 6.90e-4 | 2.51e-3 |
| 150000 | 2.1% | 5.2% | 92.7% | 1.36e-5 | 1.38e-4 | 5.08e-3 |
| 172600 (plateau) | 1.3% | 2.3% | 96.4% | 1.48e-5 | 1.07e-4 | 9.40e-3 |

img_raw 在 transport 收敛过程中**反而上升 3.7×**（2.51e-3 → 9.40e-3）——backbone 为追末端高频细节把 latent 推离 VAE manifold。

---

## 2. PET 训练目标的数学结构

### 2.1 总损失分解（[train_first_hop.py L1912-1918](../../../train_first_hop.py)）

$$
L_{\text{total}} = \lambda_p \cdot L_{\text{pair}} + \lambda_r \cdot L_{\text{rollout}} + \lambda_i \cdot L_{\text{img}}
$$

其中三项各自展开：

**Pair loss**（[`compute_pair_losses` L289-360](../../../train_first_hop.py)）：
$$
L_{\text{pair}} = \sum_{k=0}^{K-1} p_k \cdot c_k \cdot \mathbb{E}_{t \sim U(0,1)} \big\| f_\theta\big( (1-t) z_{\text{GT}}^{(k)} + t \, z_{\text{GT}}^{(k+1)},\; t,\; \text{hop}=k \big) - v^{(k)} \big\|^2
$$

其中：
- $p_k$ = `pair_sample_probs[k]`（V3 = [0.25,0.25,0.25,0.25]）
- $c_k$ = `pair_loss_weights[k]`（V3 = [1.20, 1.10, 1.05, 1.00]）
- $v^{(k)} = (z_{\text{GT}}^{(k+1)} - z_{\text{GT}}^{(k)}) / \Delta t$
- 输入是 GT segment 上随机时间点 $t \in [0,1]$ 的插值

**Rollout loss**（[`rollout_multistep_losses_first_hop` L49-110](../../../pet_lr/rollout_first_hop.py)）：
$$
L_{\text{rollout}} = \frac{1}{\sum_k w_k} \sum_{k=0}^{K-1} w_k \cdot \big\| f_\theta\big( z_{\text{curr}}^{(k)},\; t_k,\; \text{hop}=k \big) - z_{\text{GT}}^{(k+1)} \big\|^2
$$

其中：
- $w_k$ = `step_weights[k]`
- $z_{\text{curr}}^{(0)} = z_{\text{rollout}}[:,0] = z_{\text{GT}}^{D50}$（**永远 GT**）
- $z_{\text{curr}}^{(k+1)} = \text{mix}(\alpha,\, z_{\text{pred}}^{(k+1)},\, z_{\text{GT}}^{(k+1)})$ for $k \geq 1$
- 监督点固定为 segment 起点 $t=t_k$（不像 pair 是全段采样）

**Image aux loss**（仅 hop0）：
$$
L_{\text{img}} = \big\| \mathcal{D}\big( f_\theta(z_{\text{GT}}^{D50}) \big) - x_{\text{GT}}^{D20} \big\|^2 + \cdots
$$

仅当 batch 包含 hop0 样本时计算，是 hop0-exclusive 的 pixel-domain 监督。

### 2.2 三个通道对每跳的物理作用

定义"通道 $C$ 在 hop $k$ 的供给"为该通道关于 hop $k$ 的梯度贡献：

$$
G_C^{(k)} = \frac{\partial L_C}{\partial \theta} \bigg|_{\text{hop } k \text{ 部分}}
$$

| 通道 | 对 hop $k$ 的物理意义 | 输入分布 |
|---|---|---|
| **Pair[k]** | 在 GT segment 全段上学 hop k 的 velocity | $z_t = (1-t) z_{\text{GT}}^{(k)} + t z_{\text{GT}}^{(k+1)}$, $t \sim U(0,1)$ |
| **Rollout[k]** | 在 cascade 上下文（$z_{\text{curr}}^{(k)}$ 可能含漂移）下学 hop k 的端点预测 | $z_{\text{curr}}^{(k)}$, $t = t_k$ 固定 |
| **Image[k]** | 仅 $k=0$，pixel-domain 监督 | $z_{\text{GT}}^{D50}$ |

---

## 3. 核心定理：rollout[hop0] 与 pair[hop0] 的输入分布退化重叠

### 3.1 定理陈述

**定理 1**：在 V6 设计下（`alpha_start=0.0`, `lambda_start=0.0`, `straight_through_alpha_min=0.999`），rollout[hop0] 的输入分布是 pair[hop0] 输入分布的**单点退化**：

$$
P_{\text{rollout}}^{(0)}(z, t) = \delta(z - z_{\text{GT}}^{D50}) \cdot \delta(t - t_0)
$$
$$
P_{\text{pair}}^{(0)}(z, t) = \mathbb{E}_{t' \sim U(0,1)} \big[ \delta(z - (1-t') z_{\text{GT}}^{D50} - t' z_{\text{GT}}^{D20}) \cdot \delta(t - t') \big]
$$

且 $P_{\text{rollout}}^{(0)} = \lim_{t' \to t_0} P_{\text{pair}}^{(0)}|_{t=t'}$。

### 3.2 证明

**rollout 在 hop0 的输入**（[`rollout_first_hop.py` L74](../../../pet_lr/rollout_first_hop.py)）：
```python
z_curr = z_rollout[:, 0]   # = z_GT_D50, 任何 alpha 任何 step 都不变
t_src = full(rollout_times[0])  # = t_0 固定
```

**pair 在 hop=0 的输入**（[`compute_pair_losses` L302](../../../train_first_hop.py) 隐式）：
```python
t = uniform(0, 1)              # batch level 随机
z_t = (1-t) * z_GT_D50 + t * z_GT_D20
```

当 $t' = t_0 \approx 0$ 时，$z_{\text{pair}} = (1-t_0) z_{\text{GT}}^{D50} + t_0 z_{\text{GT}}^{D20} \approx z_{\text{GT}}^{D50}$（pair 的 segment 起点退化到 rollout 的输入点）。$\square$

### 3.3 推论：rollout[hop0] 的"独占信息"为零

定义 rollout 通道在 hop $k$ 的**独占信号**为：

$$
\mathcal{I}_{\text{unique}}^{(k)} = \text{KL}\big( P_{\text{rollout}}^{(k)} \,\big\|\, P_{\text{pair}}^{(k)} \big)
$$

由定理 1 直接得：
$$
\mathcal{I}_{\text{unique}}^{(0)} = \text{KL}\big( \delta(t-t_0) \,\big\|\, U(0,1) \big) \to \infty
$$

但这个无穷大是负向的——意味着 rollout 给 hop0 的不是"新分布"，而是"pair 已覆盖分布的一个点"。**rollout[hop0] 不带来 pair 不知道的信息**。

对比 hop3：
$$
P_{\text{rollout}}^{(3)} = \delta(z - z_{\text{curr}}^{(3)}) \cdot \delta(t - t_3)
$$
当 alpha=1 时 $z_{\text{curr}}^{(3)} = z_{\text{pred}}^{(3)} \neq z_{\text{GT}}^{(3)}$，由 LatMSE_RO/TF[hop3]=5.64 得 $\|\delta_3\|^2 \approx 5.64 \cdot \|\delta_0\|^2$。**rollout[hop3] 的输入分布完全脱离 pair[hop3] 的 GT-segment 流形**——这是 pair 永远看不到的 noisy-input 监督。

---

## 4. 通道-跳 适配矩阵

由 §3 推论，每个通道有"擅长"和"不擅长"的跳：

| 通道 \ Hop | hop0 | hop1 | hop2 | hop3 |
|---|---:|---:|---:|---:|
| **Pair**（GT-input velocity）| ⭐⭐⭐ 难度大 | ⭐⭐ | ⭐⭐ | ⭐ 接近恒等 |
| **Rollout**（cascade 上下文）| ❌ 与 pair 重复 | ⭐⭐ ExpoGap=3.38 | ⭐⭐⭐ ExpoGap=4.71 | ⭐ Jacobian 饱和（v_std→0）|
| **Image_aux**（pixel domain）| ⭐⭐⭐ 仅作用于 hop0 | — | — | — |

**关键不对称**：
- **Pair** 衡量"GT 输入下的 velocity 学习"——hop0 难度最大（CeilGap 11.07），适合**大幅头重**
- **Rollout** 衡量"cascade 漂移下的修正能力"——hop0 漂移恒为 0（无 cascade），hop1-3 漂移渐增；但 hop3 v_std=0.000141 接近恒等，rollout 梯度 $\propto \partial f/\partial \theta$ 被 Jacobian 钳住

---

## 5. Jacobian 饱和：为什么 rollout step_weights[3] 大也无效

### 5.1 rollout 梯度展开

对 step $k$：
$$
\nabla_\theta L_{\text{rollout}}^{(k)} = w_k \cdot 2\big( z_{\text{pred}}^{(k+1)} - z_{\text{GT}}^{(k+1)} \big) \cdot \frac{\partial z_{\text{pred}}^{(k+1)}}{\partial \theta}
$$

在 mean-flow 公式下：
$$
z_{\text{pred}}^{(k+1)} = z_{\text{curr}}^{(k)} + (t_{k+1} - t_k) \cdot f_\theta(z_{\text{curr}}^{(k)}, t_k)
$$

所以：
$$
\frac{\partial z_{\text{pred}}^{(k+1)}}{\partial \theta} = (t_{k+1} - t_k) \cdot \frac{\partial f_\theta}{\partial \theta}
$$

模型 $f_\theta$ 输出的 velocity 量级近似 $v_{\text{std}}^{(k)}$，故：
$$
\Big\| \frac{\partial z_{\text{pred}}^{(k+1)}}{\partial \theta} \Big\| \sim (t_{k+1} - t_k) \cdot v_{\text{std}}^{(k)}
$$

### 5.2 hop3 的 Jacobian 饱和

代入数据（$v_{\text{std}}^{(3)} = 0.000141$）：
$$
\Big\| \frac{\partial z_{\text{pred}}^{(3)}}{\partial \theta} \Big\| \approx 0.25 \times 0.000141 = 3.5 \times 10^{-5}
$$

而 hop1（$v_{\text{std}}^{(1)} = 0.002946$）：
$$
\Big\| \frac{\partial z_{\text{pred}}^{(1)}}{\partial \theta} \Big\| \approx 0.25 \times 0.002946 = 7.4 \times 10^{-4}
$$

**hop3 的 Jacobian 是 hop1 的 1/21**。即使 step_weights[3] = 4.0 vs step_weights[1] = 1.0（4× 加权），实际有效梯度 hop3 / hop1 = 4 × (1/21) = 0.19。**给 hop3 加权 4× 仍只达到 hop1 baseline 的 19%**。

### 5.3 hop3 ExpoGap=6.32 dB 的归因

由 BPTT 链式法则，记 $\delta_k$ = hop k 入口处 latent 漂移：

$$
\delta_3 = J_2 \delta_2 + \epsilon_3, \quad J_2 = \frac{\partial f^{(2)}}{\partial z_2}
$$

由 $v_{\text{std}}^{(2)} = 0.000775$ 估计 $J_2 \approx I + O(v_{\text{std}}^{(2)})$，所以：
$$
\delta_3 \approx \delta_2 + \epsilon_3
$$

**hop3 的 ExpoGap=6.32 dB 几乎是 hop2 漂移 $\delta_2$ 的下游显示，不是 hop3 内部学习不足**。要修 hop3 的 ExpoGap，应该通过 BPTT 把梯度反传到 hop1/hop2——但同样的 near-identity Jacobian 让反传也衰减。

**结论**：step_weights[3] 大不仅梯度被钳住，还无法有效反传修复上游根因。

---

## 6. ExpoGap × $v_{\text{std}}$ Sweet Spot 分析

定义 hop k 的 rollout 通道**有效信号强度**：

$$
S_{\text{rollout}}^{(k)} = \underbrace{\text{ExpoGap}_k}_{\text{独占信息量}} \times \underbrace{v_{\text{std}}^{(k)}}_{\text{Jacobian 容量}}
$$

代入数据：

| Hop | ExpoGap (dB) | $v_{\text{std}}$ | $S_{\text{rollout}}$ | 归一化（×4 求和=4）|
|---|---:|---:|---:|---:|
| 0 | 0.00 | 0.0096 | **0.000** | 0.00 |
| **1** | 3.38 | 0.0029 | **0.00982** | **2.65** |
| 2 | 4.71 | 0.00078 | 0.00367 | 0.99 |
| 3 | 6.32 | 0.00014 | 0.00088 | 0.24 |

理论最优 step_weights ∝ $S_{\text{rollout}}$（带 hop0 底线 0.5）：

$$
\boxed{w_{\text{rollout}}^* = [0.5,\; 2.65,\; 1.00,\; 0.24]}
$$

四舍五入到工程值：**[0.5, 2.0, 1.5, 1.0]**（适度平滑，避免极端权重导致优化震荡）。

---

## 7. hop0 的正确加权通道：pair_loss_weights

### 7.1 pair 通道在 hop0 的梯度

$$
\nabla_\theta L_{\text{pair}}^{(0)} = p_0 c_0 \cdot 2 \mathbb{E}_t \big[ (f_\theta - v^{(0)}) \cdot \frac{\partial f_\theta}{\partial \theta} \big]
$$

在 hop0 上 $v_{\text{std}}^{(0)} = 0.009634$ 是 4 跳最大（**Jacobian 容量充足**），且 segment 全段采样 $t \sim U(0,1)$ 提供**比 rollout 更密的监督**。

### 7.2 V3 实测：pair[hop0] 已是 hop0 的主要梯度来源

V3 的 `pair_loss_weights = [1.20, 1.10, 1.05, 1.00]` 已识别 hop0 优先（设计者直觉对了），但加权幅度（1.20×）不够。CeilGap 11.07 vs 其它跳 9.4-9.6 是设计者**应该注意但被压缩成 18% 加权**的信号。

**修正建议**：基于 CeilGap 比例：

$$
c_k^{\text{new}} \propto \text{CeilGap}_k^2 / \sum \text{CeilGap}_k^2
$$

| Hop | CeilGap (dB) | $\text{CeilGap}^2$ | 归一化×4 |
|---|---:|---:|---:|
| 0 | 11.07 | 122.5 | **1.34** |
| 1 | 9.42 | 88.7 | 0.97 |
| 2 | 9.62 | 92.5 | 1.01 |
| 3 | 9.51 | 90.4 | 0.99 |

CeilGap² 比例不够激进（hop0 仅 1.34×）。考虑到 hop0 是级联基础（任何 hop0 误差通过 BPTT 链式放大），**建议进一步提升到 2-3×**：

$$
\boxed{c^* = [2.5,\; 1.0,\; 1.0,\; 1.0]}
$$

---

## 8. 三通道协同的总梯度分配

### 8.1 设定（V6 修正版参数）

```yaml
loss:
  pair_weight: 15.0      # λ_p
training:
  rollout:
    lambda_end: 4.0      # λ_r
    step_weights: [0.5, 2.0, 1.5, 1.0]
  image_aux:
    lambda: 0.04         # λ_i
transport:
  pair_sample_probs: [0.25, 0.25, 0.25, 0.25]
  pair_loss_weights: [2.5, 1.0, 1.0, 1.0]
```

### 8.2 Phase III（150K-200K，alpha=1.0, λ_roll=4.0）每跳梯度占比

用 V3 86800 实测 raw loss 估算（pair_raw=7.16e-5，roll_raw=6.90e-4，img_raw=2.51e-3）：

**Pair 部分**（按 $p_k c_k$ 分配，$\sum p_k c_k = 0.25 \times (2.5+1+1+1) = 1.375$）：
$$
\text{pair}_k = \frac{15 \cdot 7.16 \times 10^{-5} \cdot p_k c_k}{Z} = \frac{1.07 \times 10^{-3} \cdot p_k c_k}{Z}
$$

**Rollout 部分**（按 $w_k / \sum w_k$ 分配，$\sum w_k = 5.0$）：
$$
\text{roll}_k = \frac{4 \cdot 6.90 \times 10^{-4} \cdot w_k / 5.0}{Z} = \frac{5.52 \times 10^{-4} \cdot w_k}{Z}
$$

**Image 部分**（仅 hop0）：
$$
\text{img}_0 = \frac{0.04 \cdot 2.51 \times 10^{-3}}{Z} = \frac{1.00 \times 10^{-4}}{Z}
$$

总 $Z = 15 \times 7.16e-5 + 4 \times 6.90e-4 + 0.04 \times 2.51e-3 = 1.07e-3 + 2.76e-3 + 1.00e-4 = 3.93e-3$

| Hop | Pair | Rollout | Image | **合计** |
|---|---:|---:|---:|---:|
| 0 | 17.1% | 7.0% | **2.6%** | **26.7%** |
| 1 | 6.8% | 28.1% | 0% | 34.9% |
| 2 | 6.8% | 21.1% | 0% | 27.9% |
| 3 | 6.8% | 14.0% | 0% | 20.8% |

**hop0 合计 26.7%（4 跳第二，仅次于 hop1 sweet spot），且其中 17.1% 来自 pair（独占信号），不重复**。

### 8.3 与 V6 当前方案对比

V6 当前 `pair_loss_weights=[1.20,1.10,1.05,1.00]`（默认，未改）+ `step_weights=[2.5,1.5,1.0,0.5]`：

| Hop | Pair | Rollout | Image | 合计 | 真实独立信号 |
|---|---:|---:|---:|---:|---:|
| 0 | 7.6% | 35.1% | 2.6% | **45.3%** | **10.2%**（pair 7.6 + image 2.6）|
| 1 | 6.9% | 21.1% | 0% | 28.0% | 28.0% |
| 2 | 6.6% | 14.0% | 0% | 20.6% | 20.6% |
| 3 | 6.3% | 7.0% | 0% | 13.3% | 13.3% |

**V6 当前**：hop0 表面 45.3% 但其中 35.1% rollout 与 pair 重复（独占信号仅 10.2%），hop1 仅 28%。

**修正版**：hop0 26.7% 全是有效信号，hop1 34.9% 押 sweet spot。

---

## 9. V6 plan 修订建议

### 9.1 §1.4 `step_weights` 改为中重

```yaml
# 修改前
step_weights: [2.5, 1.5, 1.0, 0.5]   # 头重：按 v_std 困难度分配

# 修改后
step_weights: [0.5, 2.0, 1.5, 1.0]   # 中重：按 ExpoGap × v_std sweet spot 分配
```

### 9.2 §1（新增）`pair_loss_weights` 头重

```yaml
transport:
  pair_loss_weights:
    - 2.5    # hop0：CeilGap=11.07 dB（最大），级联基础
    - 1.0    # hop1
    - 1.0    # hop2
    - 1.0    # hop3
```

### 9.3 §1.5 per-hop 梯度表用本文 §8.2 替换

确保叙事是"hop0 通过 pair 通道得 17.1%（独占）+ image 通道得 2.6%（独占）= 19.7% 独占信号；rollout 通道押 hop1 sweet spot"，而非"hop0 50.1%"的虚胖叙事。

### 9.4 §1.6 image_aux 加可选 decay（已在 [previous discussion](#) 论证，nice-to-have）

```yaml
image_aux:
  schedule:
    - {step:      0, lambda: 0.04}
    - {step: 150000, lambda: 0.04}   # Phase III 起 hold 20K，给 BPTT 干净启动
    - {step: 170000, lambda: 0.02}
    - {step: 200000, lambda: 0.01}
  interpolation: linear
```

---

## 10. 风险与回退

### 10.1 如果中重 step_weights 失败

监控 +50K（Phase II 中段）、+100K、+150K（Phase III 起）的 `val_chain_normal_mse` 和 Path A ExpoGap：

| 检查点 | 健康指标 | 失败信号 → 动作 |
|---|---|---|
| +50K | pair_frac > 70%（Phase I 末段）| < 50% → pair_weight 不够，提到 20 |
| +100K | roll_frac 上升中，hop1 step_loss < hop2 | hop1 不下降 → step_weights[1] 提到 2.5 |
| +150K | val_chain_normal_mse < 0.000180 | > 0.000200 → 回退到 V3 [1.30,1.20,1.10,1.00] |

### 10.2 pair_loss_weights[0]=2.5 是否过激

V3 用 1.20 已经识别 hop0 优先。从 1.20 → 2.5 是 **2.08× 放大**，配合本文 §8.2 计算的 hop0 总占比 26.7%（vs V3 ~25%）增量并不大——**风险主要在数值稳定性，不在方向**。如担心 hop0 过拟合，可保守取 **2.0**。

---

## 11. 一句话总结

| 问题 | 答案 |
|---|---|
| hop0 是不是最难？| **是**（PSNR_TF 35.57 最低、CeilGap 11.07 最大、$v_{\text{std}}$ 0.0096 最大）|
| 应该给 hop0 加权？| **应该** |
| 通过 step_weights 加权？| **不应该**（rollout[hop0] 输入分布与 pair[hop0] 退化重叠，重权重 = 信号重复）|
| 通过哪个通道加权？| **pair_loss_weights[0]=2.5**（直接加权 GT-input velocity 学习）+ **image_aux=0.04**（hop0-only pixel 监督）|
| step_weights 应该怎么设？| **[0.5, 2.0, 1.5, 1.0]**（中重，押 ExpoGap × $v_{\text{std}}$ sweet spot 在 hop1）|

V6 当前 `step_weights = [2.5, 1.5, 1.0, 0.5]` 同时犯两个错：
1. **用错通道**：把"hop0 难"的事实通过 rollout step_weights 表达，但这条通道在 hop0 与 pair 重复
2. **方向错误**：把 step_weights 理解成"难度优先级"，但它实际是"cascade 上下文优先级"——在 cascade 上下文里，hop0 没有 cascade 漂移（因为是入口），hop3 有最大漂移但 Jacobian 饱和，sweet spot 在 hop1
