# Transport 突破实验计划 v6 — 2026-04-27（修订版）

**Supersedes**: v5（rollout-heavy resume 实验）
**核心发现**: V3 200K 训练数据证明 **rollout 梯度占比高（32%）时模型达到 best（step 86800），之后 image_aux 重新主导（88-93%），模型不再改善**
**V6 核心方向**: Transport-First 从 scratch 训练，方案 A（给 pair 加 λ_p 乘数）+ 延长 alpha schedule + 头重 step_weights

---

## 0. V3 200K 训练的关键教训

```
step  1000: pair=54%  roll= 1%  img=46%   (alpha=0, 早期)
step 86800: pair=13%  roll=32%  img=55%   ← BEST (rollout 影响力最大)
step150000: pair= 2%  roll= 5%  img=93%   ← 退化 (image_aux 重新淹没 rollout)
```

| 观察 | 含义 |
|------|------|
| Best 出现在 roll_frac=32% 的窗口 | rollout 梯度影响力与模型质量正相关 |
| 86800 之后 rollout_loss 降低但 image_aux 没降 | 梯度比例自然偏移，不是 config 变了 |
| 86800 后训了 86K 步无改善 | image_aux 主导的训练已经 plateau |

---

## 1. V6 设计

### 1.1 方案 A：给 pair_loss 加 λ_p 乘数（~3 行代码）

当前代码：
```python
total = pair_loss + λ_roll × rollout_loss + λ_img × image_loss
```

修改为：
```python
total = λ_pair × pair_loss + λ_roll × rollout_loss + λ_img × image_loss
```

这让三个 loss 项都有独立的权重控制，配合 V3 实际 loss 数据的数学推导（见 §1.2）确定参数。

### 1.2 参数推导（基于 V3 JSONL 实测数据）

从 V3 训练日志反推 loss 比例：
- $L_r / L_p \approx 10$（**训练全程稳定**）
- $L_i / L_p = 35 \sim 370$（波动 10×，是梯度不稳定的来源）

**推导目标**：transport（pair + rollout）占梯度 ≥ 75%，image_aux ≤ 25%

**最终参数**：

$$\boxed{\lambda_p = 15.0, \quad \lambda_r = 4.0, \quad \lambda_i = 0.04}$$

### 1.3 三阶段训练 + 延长 alpha schedule

| 阶段 | Step | alpha | λ_roll 有效值 | 设计意图 |
|------|------|-------|-------------|---------|
| **I. 精度** | 0-50K | 0（纯 GT） | 0 | 模型专心学 velocity 精度 |
| **II. 过渡** | 50K-150K | 0→1 | 0→4.0 | alpha 和 λ_roll 同步 ramp |
| **III. 精修** | 150K-200K | 1.0 | 4.0 | 纯 Pred chain，和推理一致 |

rollout config:
```yaml
rollout:
  lambda_start: 0.0       # Phase I: rollout 不参与
  lambda_end: 4.0          # Phase III: rollout 满权重
  warmup_ratio: 0.25       # 0-50K: lambda=0
  ramp_ratio: 0.50         # 50K-150K: lambda 0→4.0
  alpha_start: 0.0
  alpha_end: 1.0
  # alpha 与 rollout lambda 同步 ramp
```

### 1.4 头重 step_weights（按困难度分配）

| Hop | v_std | PSNR gap | 困难度 | step_weight |
|-----|-------|----------|--------|-------------|
| hop0 (D50→D20) | 0.009634 | 10.79 dB | **最难** | **2.5** |
| hop1 (D20→D10) | 0.002946 | ~10 dB | 中高 | **1.5** |
| hop2 (D10→D4) | 0.000775 | ~10 dB | 中低 | **1.0** |
| hop3 (D4→NORMAL) | 0.000141 | ~10 dB | **最简单** | **0.5** |

```yaml
step_weights: [2.5, 1.5, 1.0, 0.5]   # 头重：按困难度分配
```

**为什么头重而非尾重**：
- V3 原始设计 [1.30, 1.20, 1.10, 1.00] 就是头重（对的）
- hop0 的 v_std 是 hop3 的 68×，是全链最大瓶颈
- hop3 的 PSNR gap 只有 0.31 dB，几乎不需要额外关注
- image_aux 已经给 hop0 额外像素监督，step_weights 头重让 rollout 也集中在 hop0

### 1.5 预期梯度分布

**Phase I（0-50K，λ_roll=0）**：

| Step | pair% | roll% | img% |
|------|-------|-------|------|
| 典型 | **69-91%** | 0% | 9-31% |

**Phase III（150K-200K）**：

全局分布：pair ≈ 17-22%，rollout ≈ 57-70%，image ≈ 5-21%

Per-hop 分布（step_weights [2.5, 1.5, 1.0, 0.5]，Σw=5.5）：

| Hop | 来自 rollout | 来自 pair | 来自 image | 合计 |
|-----|-------------|----------|----------|------|
| hop0 (D50→D20) | 31.8% | 4.3% | 14% | **50.1%** |
| hop1 (D20→D10) | 19.1% | 4.3% | 0% | **23.4%** |
| hop2 (D10→D4) | 12.7% | 4.3% | 0% | **17.0%** |
| hop3 (D4→NORMAL) | 6.4% | 4.3% | 0% | **10.7%** |

**hop0 拿到最多梯度（50.1%）** — 最难的跳得到最多资源
**hop3 拿到最少梯度（10.7%）** — 最简单的跳不浪费资源
**transport 总占比 ≥ 79%** — transport 是主力

### 1.6 image_aux 配置

```yaml
image_aux:
  enabled: true
  warmup_ratio: 0.0       # 从 step 0 就启动
  ramp_ratio: 0.0         # 不 ramp，固定 λ
  lambda_start: 0.04
  lambda_max: 0.04         # 全程固定 0.04
```

不做 image_aux 的 ramp——它是 hop0 的固定辅助，不需要随训练阶段变化。

---

## 2. 代码改动

| 文件 | 改动 | 行数 |
|------|------|------|
| `train_first_hop.py` | `total_loss` 加 `pair_loss_weight` 乘数 | **~3 行** |
| `train_first_hop.py` | config 解析 `loss.pair_weight` | ~2 行 |
| config: `pet_flow_first_hop_224_v6_transport_first.yaml` | V6 config | 新文件 |

**总共 ~5 行代码改动 + 1 个新 config 文件。不引入新机制。**

### 2.1 代码改动位置

```python
# train_first_hop.py, total_loss 计算处
pair_loss_weight = float(cfg["loss"].get("pair_weight", 1.0))  # 新增，默认 1.0 向后兼容

total_loss = (
    pair_loss_weight * pair_losses["total"]               # ← 加 pair_loss_weight
    + rollout_losses["lambda_roll"] * rollout_losses["loss_total"]
    + float(lambda_img) * loss_img
    + foc_losses["lambda_foc"] * foc_losses["loss_total"]
    + float(lambda_align) * loss_align
)
```

---

## 3. V6 Config 关键参数

```yaml
# V6 Transport-First Config
training:
  max_steps: 200000

  rollout:
    warmup_ratio: 0.25      # 0-50K: alpha=0, lambda=0
    ramp_ratio: 0.50         # 50K-150K: alpha 0→1, lambda 0→4.0
    alpha_start: 0.0
    alpha_end: 1.0
    lambda_start: 0.0
    lambda_end: 4.0
    step_weights: [2.5, 1.5, 1.0, 0.5]   # 头重

  image_aux:
    lambda_start: 0.04
    lambda_max: 0.04          # 固定，不 ramp

loss:
  pair_weight: 15.0           # V6 核心改动

optimizer:
  lr: 8.0e-5                  # 与 V3 一致

lr_schedule:
  warmup_ratio: 0.15          # 与 V3 一致
```

---

## 4. Go/No-Go 监控

| 检查点 | 条件 | 动作 |
|--------|------|------|
| +10K | pair_frac ≈ 70-90%（Phase I，应 pair 主导） | 确认 pair_weight 生效 |
| +50K | img_frac < 30%（Phase I 结束） | 确认 image 是辅助 |
| +75K | roll_frac 开始上升（Phase II ramp 中） | 确认 rollout 渐进 |
| +100K | roll_frac ≈ 30-50%（ramp 中） | 正常 |
| +150K | roll_frac ≈ 50-70%，img_frac < 25% | **核心验证** |
| +200K | val_chain_normal_mse vs V3 best（0.000165） | 最终判定 |

---

## 5. 成功标准

| 指标 | V3 200K best | V6 目标 |
|------|-------------|---------|
| val_chain_normal_mse | 0.000165 | < 0.000150（改善 ≥ 9%） |
| full-val NORMAL PSNR | 36.87 dB | > 37.5 dB（+0.6 dB） |
| transport 总梯度占比 @ 150K+ | 7-45% | **≥ 75%** |
| best step 位置 | 86800（43%） | > 150K（75%，plateau 推迟） |
| hop0 梯度集中度 | 无保证 | **≈ 50%**（最难的跳得到最多资源） |

---

## 6. 时间估算

| 阶段 | 耗时 | GPU |
|------|------|-----|
| 代码改动（pair_weight） | ~15 min | 无 |
| V6 从 scratch 训 200K | ~5-6 天 | 1 卡 |
| Full-val eval | ~1h | 1 卡 |
| Path A 诊断 | ~30min | 1 卡 |
| **总计** | ~6 天 | 1 卡 |

---

## 7. 与之前版本的对比

| 维度 | V3 | V5（rollout-heavy resume） | V6 |
|------|-----|--------------------------|-----|
| 训练方式 | 从 scratch 200K | resume 50K | **从 scratch 200K** |
| alpha schedule | 0-5K GT, 5K-20K 混合 | 固定 alpha=1.0 | **0-50K GT, 50K-150K 混合, 150K-200K Pred** |
| pair 权重 | 1.0（默认） | 1.0 | **15.0** |
| rollout λ | 0.02→0.25 | 1.5（固定） | **0→4.0（ramp）** |
| image_aux λ | 0.005→0.12 | 0.08 | **0.04（固定）** |
| step_weights | [1.30,1.20,1.10,1.00]（头重） | [0.8,1.0,1.5,2.5]（尾重） | **[2.5,1.5,1.0,0.5]（头重加强）** |
| transport 梯度占比 | 7-45% | 45-87% | **预期 ≥ 75%** |
| 核心新机制 | 无 | 无 | **pair_weight（3 行代码）** |
| loss 归一化 | 无 | 无 | **不使用**（自然 curriculum） |
