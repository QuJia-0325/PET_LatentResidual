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

## Idea 设计

### Idea 1: Two-Phase Self-Forcing Transport（最高推荐 ⭐⭐⭐）

#### 问题定位

pair_loss ≈ 0 证明 backbone 在 GT 输入下单步 velocity 已经很精准。但 rollout_loss 仍大，说明推理时的级联输入分布与训练时不同。

**当前系统的两个 loss 路径**：

| loss | 输入来源 | 与推理一致？ |
|------|---------|------------|
| pair_loss | 永远用 GT latent (z_src_GT, z_dst_GT) | ❌ 训练专属 |
| rollout_loss | alpha ramp: GT→pred 混合 → 最终 alpha=1 纯预测 | ✅ 后期一致 |

rollout_loss 在 alpha=1 后已经是 Self-Forcing 模式（纯预测输入），但 **pair_loss 永远是 teacher-forced**——backbone 的大部分梯度来自 pair_loss（占 70-80%），而这些梯度全部在 GT 输入分布上计算。backbone 从未学会在预测误差上做精确 velocity prediction。

#### 两阶段训练设计

```
Phase 1 (0 - 100K steps): 标准训练
  ├── pair_loss: GT 输入 (z_src_GT → z_dst_GT)
  ├── rollout_loss: alpha 从 0 → 1.0 (scheduled sampling)
  ├── image_aux_loss: hop0 image-space supervision
  └── 目标: backbone 学会基本的 velocity prediction

Phase 2 (100K - 200K steps): Self-Forcing pair_loss
  ├── pair_loss: 预测输入 (z_pred_prev → z_dst_GT)
  │   每步先做一轮 rollout 生成 z_pred 链，
  │   然后用 z_pred[hop] 替代 z_GT[hop] 作为 pair_loss 的 z_src
  ├── rollout_loss: alpha = 1.0 固定 (纯预测)
  ├── image_aux_loss: 保持
  └── 目标: backbone 学会在自己的误差分布上做精确预测
```

#### Phase 2 的具体实现

```python
# Phase 2 中的 pair_loss 计算：
# 1. 先做一轮 detached rollout 获取预测 latent chain
with torch.no_grad():
    z_chain_pred = []
    z_curr = batch["z_rollout"][:, 0]  # z_D50 (GT, 因为是真实起点)
    z_chain_pred.append(z_curr)
    for hop_idx in range(4):
        out = model.predict_latent_step(z_curr, t_src, t_dst, hop_idx)
        z_curr = out["z_pred"]
        z_chain_pred.append(z_curr)

# 2. 用预测链替代 GT 作为 pair_loss 的 z_src
hop = batch["hop_idx"]  # 当前 batch 的 hop index
z_src_self = z_chain_pred[hop].detach()  # 用预测值作为输入
z_dst_gt = batch["z_dst"]               # target 仍然是 GT

# 3. backbone forward 用 self-forced 输入
out = model.predict_latent_step(z_src_self, t_src, t_dst, hop, x_src_img)
pair_loss = MSE(out["v_total_raw"], v_target_from(z_dst_gt, z_src_self))
```

#### 为什么不从一开始就 Self-Forcing

1. **冷启动问题**：Phase 1 的 backbone 预测很差（从 2000 步 smoke 开始），Self-Forcing 输入 = garbage → loss 很大 → 梯度爆炸
2. **需要基础 velocity**：Phase 1 让 backbone 先学到 "从 GT 到 GT" 的正确 velocity 方向
3. **渐进过渡**：100K 步后 backbone 已有基本能力，预测 latent 质量足够作为输入
4. **200K v3 自然对接**：当前 200K v3 训练 = Phase 1，完成后 resume 进入 Phase 2

#### 与 rollout loss 的 alpha=1 有何不同

| 维度 | rollout alpha=1 | Phase 2 Self-Forcing pair_loss |
|------|----------------|-------------------------------|
| 影响的 loss | 只影响 rollout_loss（占 ~20%） | 影响 pair_loss（占 **70-80%**） |
| 梯度来源 | 只有 rollout 链的梯度 | 主要梯度来源切换到预测分布 |
| z_src | rollout 链的 z_pred | per-sample z_pred（来自 detached pre-rollout） |
| 生效时机 | ~15K 步后 | 100K 步后 |

**这是真正的差异**：pair_loss 是 backbone 梯度的主力（70-80%），但它永远在 GT 分布上。Phase 2 把主力梯度也切换到预测分布。

#### 代码改动量

**~25 行**：
- `train_first_hop.py`：在 `compute_pair_losses` 前，根据 step 判断是否做 detached pre-rollout 并替换 batch 的 z_src
- config 新增 `self_forcing_pair_start_step: 100000`

#### Resume 设计

200K v3 完成后（Phase 1 完成），用 `--resume best.pt` 启动 Phase 2：
- `max_steps: 200000`（Phase 2 从 100K resume 到 200K = 额外 100K 步）
- `self_forcing_pair: true`
- `self_forcing_pair_start_step: 0`（因为 resume 后 step 已经 > 100K）

或者做成单次 200K 训练，在 step > 100K 时自动切换。

#### 风险与缓解

| 风险 | 缓解 |
|------|------|
| Phase 2 初期 loss 跳变 | detached pre-rollout 不传梯度到 z_src，只改变 forward 分布 |
| 预测 latent 质量差导致 velocity target 不稳定 | velocity target 仍来自 (z_dst_GT - z_src_self) / dt，GT target 端未变 |
| GPU 显存增加（多一轮 forward） | pre-rollout 在 no_grad 下，显存增加很小 |

#### 预期效果

| Path A 结果 | Self-Forcing 预期 |
|------------|-----------------|
| exposure_gap ≥ 0.5 dB | **+2-5 dB**（消除主要 exposure bias） |
| exposure_gap 0.2-0.5 dB | **+1-2 dB** |
| exposure_gap < 0.2 dB | **< 0.5 dB**（需要 Idea 2/3） |

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
当前: 200K v3 = Phase 1（进行中）
         ↓
Step 0: Path A 诊断 (0.5 GPU-day, 与 200K 并行)
         ↓
  EXPOSURE_BIAS          VELOCITY_CAPACITY          MIXED
         ↓                       ↓                    ↓
Step 1: Phase 2             Idea 2 + Idea 3      Phase 2 + Idea 2
  (Self-Forcing pair)    (Image Loss + Direct)    (两者叠加)
  resume 200K best        Path A < 0.2 dB        Phase 2 优先
  +100K steps             考虑架构变更            + Idea 2 叠加
         ↓                       ↓                    ↓
Step 2: + Idea 2           评估效果              评估效果
  (叠加 Image Loss)
         ↓
Step 3: 如果 > 3 dB → 投稿
        如果 < 1 dB → Idea 3 / paradigm shift
```

---

## 实验 Config 设计（Idea 1 — 两阶段）

### Phase 1: 200K v3（已在运行）

当前 `pet_flow_first_hop_224_200k_transport_v3.yaml` 就是 Phase 1。无需修改。

### Phase 2: Self-Forcing Pair Loss

基于 200K v3 的 best.pt resume：

```yaml
# pet_flow_first_hop_224_100k_selfforcing.yaml
run_name: first_hop_224_100k_selfforcing
max_steps: 100000  # Phase 2: 额外 100K 步

training:
  # Self-Forcing pair loss: 用 detached rollout 预测链替代 GT 作为 pair z_src
  self_forcing_pair:
    enabled: true
    # Phase 2 从第 0 步就启用（因为 resume 自 200K Phase 1 已学基础）
  # rollout alpha 固定 1.0（已完成 ramp）
  rollout:
    alpha_start: 1.0
    alpha_end: 1.0
    warmup_ratio: 0.0
    ramp_ratio: 0.0
```

或做成单次 200K 训练自动切换：

```yaml
# 在 transport_v3 200K config 中添加：
training:
  self_forcing_pair:
    enabled: true
    start_step: 100000  # 前 100K 用 GT pair，后 100K 用 self-forced pair
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

## 执行时间线

| Day | 任务 | GPU-day | 状态 |
|-----|------|---------|------|
| 1 | Path A 诊断 | 0.5 | 待启动 |
| 1-5 | 200K v3 继续跑 = **Phase 1** | ~4 | 进行中（42K/200K） |
| 5 | Phase 1 完成：评估 200K 结果 | 0 | — |
| 5 | 实施 Idea 1 Phase 2 代码（~25 行） | 0 | — |
| 6-9 | **Phase 2: Self-Forcing pair_loss** (resume 200K best → +100K) | 4 | — |
| 6-9 并行 | Idea 2: Stochastic Hop Image Loss（可叠加） | 0 | 可选 |
| 10 | Full-val eval + 结果分析 | 0.5 | — |
| **总计** | | **~9 GPU-day** | |

### 关键决策点

- **Day 1 Path A 结果**：确认 exposure bias 是否是主因 → 决定是否全力投入 Phase 2
- **Day 5 200K v3 结果**：如果 200K 已突破 plateau → Phase 2 可选；如果仍在 36.2 → Phase 2 必做
- **Day 10 Phase 2 结果**：如果 > 3 dB 改善 → 准备投稿；如果 < 1 dB → 叠加 Idea 2 或考虑 Idea 3

如果 Phase 2 带来 > 3 dB 改善 → 直接准备 workshop 投稿。
