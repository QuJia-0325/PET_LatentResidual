# Transport 突破方向调研与 Idea 设计 — 2026-04-24

## 项目现状

### 已确认的事实

| 事实 | 来源 | 数据 |
|------|------|------|
| Decoder 非主导瓶颈 | Oracle (0422) | ceiling 46-52 dB vs transport 35-36 dB |
| Transport 占 gap 96-99% | E1 (0416) | Gap_Transport >> Gap_Decoder |
| ODE 精度非问题 | E2 (0416) | half-step vs full-step 差异 < 0.17% |
| 所有 config 收敛到同一 plateau | Full-val rerank (0423) | 36.12-36.21 dB |
| pair_loss ≈ 0 但 rollout_loss 仍大 | v3-200K (0424) | 单步精准但多步累积失控 |
| 每 0.0001 latent MSE ≈ 5-9 dB image PSNR | 量化分析 (0424) | decoder 放大效应 |

### 核心矛盾

**pair_loss → 0 但 rollout_loss 不降**：backbone 在 GT 输入下已经学得很好（单步 velocity 精准），但级联 rollout 时误差累积导致 image PSNR 远低于 oracle。

---

## 文献调研

### 方向 1: Exposure Bias in Autoregressive Flow Matching

| 文献 | 会议/引用 | 方法 | 关键发现 |
|------|----------|------|---------|
| **Self-Forcing** (Huang et al.) | arXiv 2506.08009, 2025, **193 引用** | 训练时做自回归 self-rollout 而非 teacher forcing | 弥合 train-test gap，在 video diffusion 中显著提升一致性 |
| **BAgger** (Po et al.) | arXiv 2512.12080, 2025, 7 引用 | Backwards Aggregation：用模型 drifting rollout 反向修正 | 训练用标准 flow matching，不需要大 teacher |
| **Rolling Forcing** (Liu et al.) | arXiv 2509.25161, 2025, 51 引用 | 自回归 video diffusion 中渐进式 rollout forcing | 支持长序列实时生成 |
| **MixFlow** (Li et al.) | arXiv 2512.19311, 2025 | Slowed interpolation mixture 后训练修正 | **直接在 RAE 上实验**，ImageNet 1.43 FID |
| **Context Forcing** (Chen et al.) | arXiv 2602.06028, 2026, 7 引用 | 长上下文一致性训练 | 解决 exposure bias 不需要 self-rollout 的全部开销 |
| **Anti-Exposure Bias** (Zhang et al.) | **ICLR 2025 Spotlight** | Prompt learning 框架，仅增 5% 推理开销 | 通用框架，可适配各种 diffusion/flow 模型 |

**结论**：exposure bias 在 autoregressive flow matching 中是公认问题（2025 年多篇高引），但尚未在 **medical image multi-hop latent transport** 中被系统研究。

### 方向 2: 低剂量 PET 重建最新方法

| 文献 | 会议/年份 | 方法 | 特点 |
|------|----------|------|------|
| **MAP-Diff** (Jing et al.) | arXiv 2026 | Multi-anchor guided progressive denoising | 3D 全身 PET，多锚点引导 |
| **WiD-PET** (Lyu et al.) | **MICCAI 2025** | 小波域 diffusion + 快速推理 | Gradient loss 增强细节 |
| **M2Diff** (Yar et al.) | **TMI 2026** | 多模态多任务 diffusion | MRI 引导 PET 增强 |
| **DECADE** (Zhou et al.) | arXiv 2026 | 时序一致无监督 diffusion | 心脏 PET 动态去噪 |
| **Mamba-PET** (Tang et al.) | **MICCAI 2025** | Mamba + 物理一致渐进网络 | 渐进式从低到高 |

**观察**：主流 PET 方法不使用 multi-hop latent transport 框架。大多数用 diffusion/conditional generation 直接在 image space 或 single-step latent space 操作。我们的 4-hop cascade 是独特的。

### 方向 3: Latent Flow Matching 的训练优化

| 文献 | 关键发现 |
|------|---------|
| **AMC-Solver** (Liu, 2025) | 自适应多步级联 + midpoint prediction 减少 flow sampling 误差 |
| **CFO** (Hou et al., 2025) | Flow-matched neural operators 用于 PDE dynamics，解决 exposure bias |
| **Latent Generative Solvers** (Chen et al., 2026) | 物理模拟中对比 flow matching vs direct regression，发现 exposure bias 是长时演化的主要退化源 |

---

## 3 个 Idea 设计

### Idea 1: Self-Forcing Multi-Hop Transport（最高推荐 ⭐⭐⭐）

#### 动机

Self-Forcing (Huang et al., 2025, 193 引用) 解决了与我们完全相同的问题：
- **他们的场景**：自回归 video diffusion，训练用 GT 帧，推理用预测帧 → exposure bias → 时序不一致
- **我们的场景**：4-hop latent transport，训练 pair_loss 用 GT latent，推理 rollout 用预测 latent → exposure bias → 级联误差累积

他们的方法核心是：**训练时不注入 GT，而是用模型自己的预测作为下一步输入**。

#### 方法

```python
# 当前（Teacher Forcing）：
z_D20_pred = model(z_D50_gt, ...)     # hop0: GT 输入
loss_hop0 = MSE(z_D20_pred, z_D20_gt)

# rollout 用 mix_latent(z_gt, z_pred, alpha) 混合
z_curr = mix_latent(z_D20_gt, z_D20_pred, alpha=0.5)  # GT 混入
z_D10_pred = model(z_curr, ...)        # hop1: 混合输入


# Self-Forcing（提议）：
z_D20_pred = model(z_D50_gt, ...)     # hop0: GT 输入（D50 是真实起点）
loss_hop0 = MSE(z_D20_pred, z_D20_gt)

z_D10_pred = model(z_D20_pred.detach(), ...)  # hop1: 纯预测输入
loss_hop1 = MSE(z_D10_pred, z_D10_gt)

z_D4_pred = model(z_D10_pred.detach(), ...)   # hop2: 纯预测输入
loss_hop2 = MSE(z_D4_pred, z_D4_gt)

z_NORMAL_pred = model(z_D4_pred.detach(), ...)  # hop3: 纯预测输入
loss_hop3 = MSE(z_NORMAL_pred, z_NORMAL_gt)
```

#### 关键设计选择

1. **detach 策略**：每步 detach 防止梯度穿过整条链（否则 GPU 显存爆炸）
2. **alpha 调度**：前 warmup 期仍用 teacher forcing（alpha=0）确保早期稳定，ramp 到 alpha=1（Self-Forcing）
3. **hop0 始终用 GT**：D50 是真实输入（推理时也是），不需要 self-force
4. **与现有 rollout loss 的关系**：Self-Forcing 替换 rollout loss 中的 mix_latent 逻辑，不是新增 loss

#### 与现有代码的差异

```python
# rollout_first_hop.py 中的 mix_latent 修改：
# 旧：
z_curr = mix_latent(z_gt, z_pred, alpha=alpha, straight_through=True)
# 新（Self-Forcing 模式下）：
z_curr = z_pred  # 纯预测，不混合 GT
```

实际上只需要在 rollout config 中加一个 `self_forcing: true` 开关。

#### 代码改动量

**~15 行**（修改 `rollout_first_hop.py` 的 `rollout_multistep_losses_first_hop` 函数）

#### 新颖性分析

| 维度 | 分析 |
|------|------|
| Self-Forcing 本身 | 已发表（Huang et al., 2025），不是新方法 |
| **应用场景新颖** | Self-Forcing 在 medical multi-hop latent transport 中的首次应用 |
| **诊断框架新颖** | 配合 Path A（TF vs RO gap）量化 exposure bias → Self-Forcing 效果的完整诊断链 |
| **level** | Workshop paper 级别（如果效果显著可升级到 main） |

#### 预期效果

- 如果 Path A 显示 exposure_gap ≥ 0.5 dB → Self-Forcing 预期 **+2-5 dB**（消除大部分 exposure bias）
- 如果 exposure_gap < 0.2 dB → Self-Forcing 效果有限（< 0.5 dB），需要 Idea 2/3

#### 风险

- Self-Forcing 训练初期可能不稳定（预测 latent 质量差 → loss 大 → 梯度大）
- 缓解：warmup 期仍用 teacher forcing，渐进过渡

---

### Idea 2: Stochastic Hop Image-Space Loss（推荐 ⭐⭐）

#### 动机

当前 image_aux loss 只在 hop0 计算——backbone 只在 hop0 接收 image-space 梯度信号。hop1-3 完全在 latent MSE 空间训练，不知道 decoder 对哪些 latent 方向更敏感。

quantitative evidence：每 0.0001 latent MSE → 5.4-9.4 dB image PSNR 损失，但 latent MSE 对所有方向一视同仁。

#### 方法

每个训练步：
1. 正常计算 pair_loss + rollout_loss（latent 空间）
2. 从 rollout 的 4 个预测 latent 中**随机选 1 个**
3. decode 该 latent → 计算 image L1 + SSIM loss → 加到 total_loss

```python
# 在 rollout 之后：
hop_choice = random.randint(0, 3)  # 随机选一个 hop
z_chosen = z_preds[hop_choice]
x_chosen_pred = model.decode_crop(z_chosen, crop_size=224)
x_chosen_gt = x_rollout[:, hop_choice + 1]  # 对应的 GT image
loss_img_hop = L1(x_chosen_pred, x_chosen_gt) + 0.25 * SSIM(x_chosen_pred, x_chosen_gt)
total_loss += lambda_img_stochastic * loss_img_hop
```

#### 关键优势

- **零额外 CPU RAM**（图像数据已加载）
- **GPU 显存不变**（还是 1 次 decode，和当前 hop0 image_aux 一样）
- **每个 hop 都有概率接收 decoder 敏感梯度**
- 可与 Idea 1 叠加

#### 代码改动量

**~30 行**（在 training loop 中 rollout loss 计算后添加）

#### 新颖性

在 latent transport 中引入 stochastic image-space supervision 是新颖的——现有工作要么全 image loss（太贵），要么全 latent loss（不够），stochastic hop sampling 是两者的平衡。

#### 预期效果

+1-3 dB（让 velocity 在 decoder 敏感方向更精确）

#### 数据要求

需要 rollout 对应的 GT images。当前 config 中 `train_include_full_x_rollout: false`，需要改为 `true`。

**内存影响**：当前 raw images 已加载 4 个 dose（D50/D20/D10/D4 各 9.2GB = 36.8GB），加载 NORMAL（额外 9.2GB）后总 ~46GB。单实验 RAM 从 ~220GB 增加到 ~230GB。在 503GB 约束下仍可跑 2 个并行实验。

---

### Idea 3: Direct Latent Regression（去掉 velocity 框架）（备选 ⭐）

#### 动机

当前系统预测 velocity（v_pred），然后 z_pred = z_src + v × σ × dt。但 E2 已证明 ODE 精度不是问题（< 0.17%）。velocity 作为中间表示可能是不必要的间接层。

#### 方法

直接预测目标 latent：
```python
z_dst_pred = backbone(z_src, t_src, t_dst, hop_idx)  # 直接输出目标 latent
loss = MSE(z_dst_pred, z_dst_gt)  # 直接回归
```

去掉 velocity normalization（σ、dt）、pair_v_std 等复杂性。

#### 文献支持

Chen et al. (2026) "Latent Generative Solvers" 在物理模拟中发现 direct regression 在某些场景下不输 flow matching，特别是当 dynamics 非光滑时。

#### 代码改动量

**大量**（需要修改 model forward + loss + rollout + eval 全链路）

#### 风险

- 去掉 velocity 中间表示可能丧失归纳偏置（velocity 提供了 "位移方向" 的结构化信息）
- 需要重新校准所有超参数

#### 优先级

低于 Idea 1 和 2。只有在 Path A 显示 VELOCITY_CAPACITY（即使 GT 输入 velocity 也不准）时才考虑。

---

## 执行优先级

```
Step 0: Path A 诊断 (0.5 GPU-day)
        ↓
  EXPOSURE_BIAS          VELOCITY_CAPACITY          MIXED
        ↓                       ↓                    ↓
Step 1: Idea 1              Idea 2 + Idea 3      Idea 1 + Idea 2
  (Self-Forcing)        (Image Loss + Direct)    (两者叠加)
        ↓                       ↓                    ↓
Step 2: + Idea 2           考虑架构变更           评估效果
  (叠加 Image Loss)
        ↓
Step 3: 评估结果
  如果 > 3 dB 改善 → 准备投稿
  如果 < 1 dB → 考虑 Idea 3 或 paradigm shift
```

---

## 实验 Config 设计（Idea 1）

基于 transport_v3（已包含权重修正 + velocity_rebalance）：

```yaml
# 新增字段
rollout:
  self_forcing: true           # 启用 Self-Forcing
  self_forcing_warmup: 5000    # 前 5K 步仍用 teacher forcing
  self_forcing_ramp: 10000     # 5K-15K 步渐进过渡
  # 其余不变
```

---

## 实验 Config 设计（Idea 2）

```yaml
# 新增字段
training:
  stochastic_hop_image_loss:
    enabled: true
    lambda_start: 0.01
    lambda_max: 0.10
    warmup_ratio: 0.10
    ramp_ratio: 0.20

data:
  train_include_full_x_rollout: true  # 需要改为 true
```

---

## 预期时间线

| Day | 任务 | GPU-day |
|-----|------|---------|
| 1 | Path A 诊断 | 0.5 |
| 1 | 200K 继续跑（已在进行） | — |
| 2 | 根据 Path A 结果实施 Idea 1 或 Idea 2 代码 | 0 |
| 3-5 | Idea 1/2 实验（50K 步 → 100K 步） | 2-4 |
| 6-7 | Full-val eval + 结果分析 | 0.5 |
| **总计** | | **3-5 GPU-day** |

如果 Idea 1 带来 > 3 dB 改善 → 直接准备 workshop 投稿。
