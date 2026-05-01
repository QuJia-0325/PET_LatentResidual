# Pixel 结构先验注入 Latent Transport — 方向分析

**日期**: 2026-04-30
**前序**: transport_breakthrough_research_v6.md（V6/V6.1 实验进行中）
**动机**: 从 Latent Forcing (Baade et al., arXiv:2602.11401, ICML 2026) 的逆向思考——在 latent space 做 transport 时，如何有效利用 pixel-space 的结构信息

---

## 0. 问题陈述

### 0.1 当前瓶颈

E1 error budget 分析表明 transport gap 占最终 PSNR 缺口的 ~96%。velocity field 在 latent space 的 MSE 优化（`pair_loss`）是训练的绝对主力，但存在根本性问题：

$$\|z_1 - z_2\|_2 = \|z_3 - z_4\|_2 \quad \not\Rightarrow \quad \|\mathcal{D}(z_1) - \mathcal{D}(z_2)\|_\text{perceptual} = \|\mathcal{D}(z_3) - \mathcal{D}(z_4)\|_\text{perceptual}$$

frozen RAE decoder $\mathcal{D}$ 定义了一个**非等距映射**：latent space 的不同方向对解码图像质量的影响高度不均匀。pair_loss 的 velocity MSE 无法区分"对最终图像重要的 latent 方向"和"对最终图像无关紧要的 latent 方向"。

### 0.2 当前 image_aux 的局限

| 属性 | 值 | 含义 |
|------|---|------|
| 权重 | λ_img = 0.04 | pair_weight=15 的 1/375 |
| 梯度占比 | ~2-3% | 被 pair_loss 淹没 |
| 监督范围 | 仅 hop0 | hop1-3 无 pixel 信号 |
| 计算开销 | 每 step 1 次 decode | 可接受 |
| 实际作用 | "不脱轨保险" | 无法重塑 velocity field |

image_aux 的方向是对的——通过可微 decoder 反传 pixel 梯度。但权重太低、范围太窄，无法真正引导 velocity field 学到"哪些 latent 方向更重要"。

### 0.3 与 Latent Forcing 的关系

Latent Forcing 解决的是**在 pixel space 生成时引入 latent 的语义先验**——先去噪 DINOv2 latent（语义结构），再去噪 pixel（高频细节）。Latent 是临时"草稿纸"，推理后丢弃。

我们的问题是**逆向的**：在 latent space 做 transport 时引入 pixel 的结构先验。Latent 是最终产品（经 decoder 输出图像），不是中间产物。

共同 insight：**生成空间（latent/pixel）的 MSE 不是最终评价标准，需要来自另一个空间的信号来引导。**

---

## 1. 五级方案设计（从弱到强）

### Level 1：当前 image_aux（已实现，baseline）

```
hop0_batch["z_src"] → model.predict_latent_step() → z_pred
                                                       │
                                      model.decode_crop(z_pred, 224)
                                                       │
                                                    x_pred [B,1,224,224]
                                                       │
                          compute_first_hop_image_loss(x_pred, x_gt)
                                                       │
                                 ∂L/∂z_pred 反传穿过 decoder → DiT backbone
```

**损失**: $\mathcal{L}_\text{img} = 1.0 \cdot L_1^{\text{border}} + 0.25 \cdot \mathcal{L}_\text{SSIM} + 0.1 \cdot \mathcal{L}_\text{seam}$

**汇入**: `total_loss += 0.04 × L_img`

**优点**: 已实现，零风险
**缺点**: 权重太低（~2-3%），仅监督 hop0，无法改变 velocity field 主结构

---

### Level 1.5：LPIPS-Augmented image_aux（ICLR 基线）

**定位**: 在尝试任何自定义 reweighting 方案（Level 2/3）之前，先对齐 latent diffusion 领域的标准做法。这是 ICLR/NeurIPS 论文中 latent space 训练的事实基线（Stable Diffusion、LDM、DiT 系列均采用感知损失辅助 latent 训练）。

#### 1.5.1 核心思想

当前 image_aux 的 metric 是纯 low-level pixel 量：

$$\mathcal{L}_\text{img}^{\text{current}} = 1.0 \cdot L_1^{\text{border}} + 0.25 \cdot \mathcal{L}_\text{SSIM} + 0.1 \cdot \mathcal{L}_\text{seam}$$

问题：L1 和 SSIM 都假设 pixel 之间独立或局部窗口独立，无法度量"哪些 pixel 差异在感知上重要"。CT/MRI 重建中典型表现是：模型可以通过整体平滑得到低 L1 但失去诊断细节。

升级为含 LPIPS 的版本：

$$\mathcal{L}_\text{img}^{\text{LPIPS}} = w_{L_1} \cdot L_1^{\text{border}} + w_\text{SSIM} \cdot \mathcal{L}_\text{SSIM} + w_\text{seam} \cdot \mathcal{L}_\text{seam} + w_\text{LPIPS} \cdot \mathcal{L}_\text{LPIPS}(x_\text{pred}, x_\text{gt})$$

LPIPS 通过 pretrained VGG/AlexNet feature 距离度量感知差异。**关键性质**：LPIPS 反传到 latent 的梯度天然包含 decoder Jacobian pullback——

$$\nabla_z \mathcal{L}_\text{LPIPS} = J^\top \nabla_x \mathcal{L}_\text{LPIPS}$$

这在 hop0 上**就是 Level 2 的 in-the-wild 实现**（J^T 的作用通过自动微分隐式完成），无需任何预计算或 Hutchinson 估计。

#### 1.5.2 实现

**代码改动（约 5-10 行）**:

```python
# pet_lr/losses/image_loss.py 或 train_first_hop.py 的 image loss 计算
import lpips

# 模型构建时
self.lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
for p in self.lpips_fn.parameters():
    p.requires_grad = False

# image loss 计算时
def compute_image_loss(x_pred, x_gt):
    # 现有 L1 / SSIM / seam ...
    l1 = compute_l1_border(x_pred, x_gt)
    ssim = compute_ssim(x_pred, x_gt)
    seam = compute_seam(x_pred, x_gt)
    # 新增 LPIPS
    # LPIPS 期望 [-1, 1] 范围 RGB（3 通道），需要适配单通道 CT
    x_pred_rgb = x_pred.repeat(1, 3, 1, 1).clamp(-1, 1)
    x_gt_rgb = x_gt.repeat(1, 3, 1, 1).clamp(-1, 1)
    lpips_val = self.lpips_fn(x_pred_rgb, x_gt_rgb).mean()
    return w_l1 * l1 + w_ssim * ssim + w_seam * seam + w_lpips * lpips_val
```

**配置（建议起点）**:

```yaml
image_aux:
  l1_weight: 0.5            # 当前 1.0 → 0.5
  ssim_weight: 0.15         # 当前 0.25 → 0.15
  seam_weight: 0.05         # 当前 0.10 → 0.05
  lpips_weight: 0.5         # 新增
  lambda_start: 0.20        # 当前 0.04 → 0.20（提升 5×）
  lambda_max: 0.20
```

两个调整**联动**：内部 LPIPS 引入了感知项，外部 λ_img 可以放心从 0.04 提到 0.20，因为感知约束会防止 V3 plateau 时观察到的"image_aux 占比飙到 88-93%"现象——LPIPS 项的梯度在过拟合 pixel 之前就饱和。

#### 1.5.3 风险与缓解

| 风险 | 评估 | 缓解 |
|---|---|---|
| LPIPS 在医学 CT 上的有效性 | 中——LPIPS 训于自然图像，CT 是单通道灰度 | 用 `repeat(3,1,1)` 转伪 RGB；若效果差可改用 medical-LPIPS（一些工作 fine-tune 过 RadImageNet 版本）|
| 增加显存（VGG forward）| 低（~200MB）| 仅 forward，无需 grad 通过 VGG 参数 |
| 增加训练时间 | 低（~5-8% 单步开销）| 可接受 |
| λ_img 提到 0.20 后又陷 V3 plateau | 中 | 监控 img_frac，超过 50% 触发回退 |

#### 1.5.4 与 Level 2/3 的关系

Level 1.5 不是 Level 2/3 的替代品，而是**前置基线**：

- 如果 Level 1.5 已经能将 PSNR 从当前 36.87 dB 推到 37.5+ dB，则 Level 2/3 的额外复杂度难以 justify
- 如果 Level 1.5 进展有限（< 0.2 dB），则证明感知损失辅助不够，需要 Level 2/3 的 latent-space 直接 reweighting
- Level 1.5 给出的 LPIPS 反传在 hop0 已隐式做了 J^T pullback，这等于在不写预计算脚本的情况下验证了 "Jacobian-aware" 思路是否对该任务有效

#### 1.5.5 实施门槛

**极低**：
- 依赖：`pip install lpips`（已成熟库，无 GPL 许可问题）
- 代码：5-10 行
- 配置：1 个新文件（基于 V6.1 yaml）
- 数据：无需预处理
- 计算：单步 +5-8%

**优点**: ICLR/NeurIPS 标准做法，代码改动最小，无预计算，立即可启动
**缺点**: LPIPS 训于自然图像，CT 灰度图上效果有上限；仍仅监督 hop0
**风险**: 低——所有改动可控，failure mode 是"无明显 PSNR 提升"而非"训练不稳定"

---

### Level 2：Latent-Space Perceptual Reweighting（Jacobian 方案）

**核心思想**: 不走 pixel 回路，而是在 latent space 内学一个**重要性权重** $W$，使得 velocity MSE 在"对解码图像影响大"的 latent 方向上权重更高。

**数学推导**:

Decoder 的 Jacobian 矩阵 $J = \frac{\partial \mathcal{D}}{\partial z}$ 描述了 latent 扰动到 pixel 变化的局部线性映射。定义 Fisher Information-like 度量：

$$W = J^\top J$$

则 pixel-space 的 L2 距离近似为：

$$\|\mathcal{D}(z_1) - \mathcal{D}(z_2)\|^2 \approx (z_1 - z_2)^\top W (z_1 - z_2)$$

将 velocity loss 从 $\|v_\text{pred} - v_\text{gt}\|^2$ 改为 $(v_\text{pred} - v_\text{gt})^\top W (v_\text{pred} - v_\text{gt})$。

**实现选项**:

| 变体 | 计算方式 | 开销 | 精度 |
|------|---------|------|------|
| **2a. 离线预计算** | 对一批样本计算 $J^\top J$ 的 per-token 对角近似，固定为权重 | 一次性，训练无开销 | 粗，忽略样本依赖性 |
| **2b. 在线 Hutchinson 估计** | 每 N 步用随机向量探测 $J^\top J$ 的对角元素 | 中等（每 N 步一次 vjp） | 较好，自适应 |
| **2c. 对角 Fischer 代理** | 用 decoder 每层的 feature norm 加权 | 几乎无开销 | 粗糙但实用 |

**推荐**: 2a（离线预计算）作为第一步验证，成本最低。

**伪代码（离线预计算 per-token 权重）**:

```python
# 预计算阶段（一次性，用 val set 的一批样本）
token_importance = torch.zeros(num_tokens, latent_dim)  # [256, 768]
for z_gt in val_loader:
    z_gt.requires_grad_(True)
    x_decoded = model.decode_crop(z_gt, 224)
    # Per-pixel gradient w.r.t. latent
    for c in range(x_decoded.shape[-2]):
        for r in range(x_decoded.shape[-1]):
            grad = torch.autograd.grad(x_decoded[..., c, r].sum(), z_gt, retain_graph=True)[0]
            token_importance += grad.pow(2).mean(dim=0)  # 对角近似
# 归一化为 per-token 权重
token_weights = token_importance.mean(dim=-1)  # [256]
token_weights = token_weights / token_weights.mean()  # 均值归一化
```

**训练时使用**:

```python
# velocity MSE 加权
vel_err = (v_pred - v_gt).pow(2)  # [B, 256, 768]
vel_err_weighted = vel_err * token_weights.view(1, 256, 1)  # per-token 加权
loss_velocity = vel_err_weighted.mean()
```

**优点**: 不增加训练时 decode 开销，直接注入 pair_loss 主通道
**缺点**: 局部线性近似，忽略非线性效应；固定权重不随训练动态变化
**风险**: 低——即使权重不准确，也只是改变了 velocity MSE 的空间加权，不会破坏训练稳定性

---

### Level 3：Pixel-Weighted Velocity Loss（预计算 pixel 差异引导）

**核心思想**: 利用 GT 数据对的 pixel-space 差异图来反推 latent token 的重要性权重，把像素级结构信息**静态烘焙**进 velocity loss。

**原理**:

对于 GT 配对 $(z_\text{src}, z_\text{dst})$，我们可以预计算：

$$\Delta_\text{pixel}(i, j) = |\mathcal{D}(z_\text{dst}) - \mathcal{D}(z_\text{src})|_{i,j}$$

这个 224×224 的差异图显示了"这次 transport 在图像中哪里变化最大"。将其聚合到 16×16 token grid：

$$w_\text{token}(k) = \frac{1}{P^2} \sum_{(i,j) \in \text{patch}_k} \Delta_\text{pixel}(i,j)$$

归一化后作为 velocity loss 的 per-token per-sample 权重。

**关键区别于 Level 2**: 这是 **per-sample 自适应**的——不同的输入图像有不同的权重图（比如病灶区域 token 的权重更高）。

**实现方案**:

```
训练前预计算（一次性）:
  对 train set 的每对 (z_src, z_dst):
    1. x_src = decode(z_src), x_dst = decode(z_dst)
    2. Δ_pixel = |x_dst - x_src|  [1, 224, 224]
    3. 聚合到 16×16 token grid → token_weight [256]
    4. 归一化: token_weight / token_weight.mean()
    5. 保存到数据集 (与 z_src, z_dst 一起存储)

训练时:
  velocity_err = (v_pred - v_gt).pow(2)  # [B, 256, 768]
  velocity_err_weighted = velocity_err * token_weight.view(B, 256, 1)
  loss_velocity = velocity_err_weighted.mean()
```

**数据集改动**:
- `data_first_hop.py` 的 dataset 需要额外返回 `token_weight` 字段
- 预计算脚本需要遍历全部 104,988 对样本，decode 两次
- 存储开销: 每对 256 个 float32 = 1 KB，总计 ~100 MB（可以接受）

**数学分析**:

当前 velocity loss 等价于假设 decoder 是等距的（各方向同等重要）：

$$\mathcal{L}_\text{pair} = \mathbb{E}\left[\|v_\text{pred} - v_\text{gt}\|_2^2\right]$$

Level 3 改为：

$$\mathcal{L}_\text{pair}^{\text{pixel-guided}} = \mathbb{E}\left[\sum_{k=1}^{256} w_k \cdot \|v_\text{pred}^{(k)} - v_\text{gt}^{(k)}\|_2^2\right]$$

其中 $w_k$ 反映 token $k$ 对应的像素区域在 transport 中的变化量。效果：**病灶区域、边缘区域的 velocity 精度被优先保证**。

**优点**:
- Per-sample 自适应，比 Level 2 的固定权重更精确
- 不增加训练时计算（权重预计算）
- 直接注入 pair_loss 主通道（权重占比 ~97%），而非只修改占 ~3% 的 image_aux
- 物理上直观：变化大的区域 → 更重要的 velocity

**缺点**:
- 需要一次性预计算（~105K 对 × 2 次 decode ≈ 数小时）
- 权重基于 GT pair，不随模型训练状态变化
- 只捕获了"哪里变化大"，没捕获"哪里变化难"

**风险**: 低——per-token reweighting 是常见技巧（类似 focal loss 的思路），不改变训练整体框架

---

### Level 4：Multi-Hop Pixel Consistency（全链 pixel 监督）

**核心思想**: 当前 image_aux 只监督 hop0（D50→D20）。对 rollout chain 的**中间和最终输出也做 pixel decode + loss**，让整条链对最终图像质量负责。

**训练时数据流**:

```
z_D50 ─hop0→ ẑ_D20 ─hop1→ ẑ_D10 ─hop2→ ẑ_D4 ─hop3→ ẑ_NORMAL
  │            │             │           │            │
decode       decode        decode      decode       decode
  │            │             │           │            │
x̂_D50       x̂_D20        x̂_D10      x̂_D4        x̂_NORMAL
  │            │             │           │            │
pixel_loss  pixel_loss   pixel_loss  pixel_loss   pixel_loss
```

**损失组合**:

$$\mathcal{L}_\text{chain\_pixel} = \sum_{k=0}^{3} \gamma_k \cdot \mathcal{L}_\text{pixel}(\hat{x}_{k+1}, x_{k+1}^{\text{GT}})$$

其中 $\gamma_k$ 按重要性递增（越靠近 NORMAL，最终质量影响越大）：

```yaml
chain_pixel_weights: [0.5, 1.0, 1.5, 2.0]  # hop0 → hop3
```

**为什么 hop3 权重最高**:
- hop3 输出就是最终的 $z_\text{NORMAL}$，直接决定最终图像质量
- hop0-2 的误差会通过链级联放大，但当前 rollout_loss 在 latent space 已经覆盖
- chain_pixel 的独占价值在于告诉 hop3："你的 latent 输出解码后要长得像 NORMAL 图像"

**实现难度**:

| 挑战 | 影响 | 缓解方案 |
|------|------|---------|
| 计算开销 | 5 次 decode/step（当前 1 次） | 可以 subsample：每 N 步做一次 full chain pixel，或随机选 1 个 hop |
| 内存 | decode 需要额外 GPU 内存 | 在 no_grad 区间 decode hop1-3（只有 hop0 需要梯度穿过 decode） |
| 梯度复杂性 | 全链梯度路径长 | hop1-3 的 pixel loss 用 stop-grad on chain input，只优化当 hop 的 velocity |

**推荐简化版**:

只在 rollout chain 的**最终输出** $\hat{z}_\text{NORMAL}$ 上做 pixel loss：

```python
with torch.no_grad():
    z_chain_normal = rollout_chain(z_D50, model)  # 已有 rollout 计算
x_chain_normal = model.decode_crop(z_chain_normal, 224)
loss_chain_pixel = pixel_loss(x_chain_normal, x_normal_gt)
```

这只增加 1 次额外 decode，且不需要通过整条链反传梯度（用 straight-through 或 reinforce）。

**优点**: 直接优化最终输出的像素质量，不仅仅是中间 velocity 精度
**缺点**: 计算昂贵，梯度路径复杂，训练不稳定风险高
**风险**: 中高——可能引起 rollout 和 pixel loss 之间的梯度冲突

---

## 2. 方案对比

| 维度 | Level 1 | **Level 1.5** | Level 2 | Level 3 | Level 4 |
|------|---------|---------------|---------|---------|---------|
| **名称** | image_aux (当前) | **LPIPS-augmented image_aux** | Jacobian reweight | Pixel-weighted velocity | Multi-hop pixel |
| **注入位置** | 独立辅助 loss | **独立辅助 loss（含感知）** | pair_loss 内 | pair_loss 内 | 独立辅助 loss |
| **权重占比** | ~3% | **~12-15%（λ_img 0.04→0.20）** | ~97% (改造主通道) | ~97% (改造主通道) | 可调 |
| **感知信号** | 无 | **LPIPS（VGG features）** | 间接（J^T J 谱）| 间接（GT Δpixel）| L1/SSIM |
| **Jacobian pullback** | 无 | **隐式（autograd 完成）** | 显式（对角近似）| 无 | 无 |
| **适应性** | 固定 | **per-sample（VGG features 自适应）** | 全局固定 | per-sample 自适应 | per-sample per-hop |
| **hop 覆盖** | 仅 hop0 | **仅 hop0** | 所有 hop | 所有 hop | 所有 hop |
| **训练开销** | +1 decode/step | **+1 decode +1 VGG forward (~5-8%)** | 无 | 无（预计算） | +1~5 decode/step |
| **实现难度** | ✅ 已完成 | **极低（5-10 行）** | 中 | 中 | 高 |
| **代码改动** | 无 | **~10 行** | ~30 行 | ~60 行 + 预计算脚本 | ~120 行 |
| **预计算** | 无 | **无** | 一次性（小批样本）| 一次性（全量 105K 对）| 无 |
| **领域基线对齐** | N/A | **✅ ICLR/NeurIPS 标准**| 🔬 原创 | 🔬 原创 | 🔬 原创 |
| **风险** | 无 | **低** | 低 | 低 | 中高 |

---

## 3. 推荐路径

### 阶段 0（当前，V6/V6.1 训练期间）

**主线训练不改动**。V6/V6.1 跑完之前不引入新变量。

**并行零成本工作**：
- 安装 LPIPS 依赖、阅读 hop0 image_aux 代码路径
- 准备 Level 1.5 的 yaml 草稿（基于 V6.1 + LPIPS 项）
- 准备 H1/H2 预计算脚本（用于后续 Level 3 go/no-go gate）

### 阶段 1（V6/V6.1 完成后）：Level 1.5 优先

**首先实施 Level 1.5（LPIPS-augmented image_aux）**:

**理由**:
1. **领域基线对齐**——这是 latent diffusion 训练的 ICLR/NeurIPS 事实标准，不做这一步直接跳到自定义 Level 2/3 在论文 review 时会被质疑 "why not just use LPIPS"
2. **隐式 Jacobian pullback**——LPIPS 反传到 latent 的梯度自然是 $J^\top \nabla_x \mathcal{L}_\text{LPIPS}$，等于 in-the-wild 验证了 "Jacobian-aware loss" 思路是否对该任务有效，无需任何预计算或 Hutchinson 估计
3. **代码改动 5-10 行，1 周可出结果**——投资回报率最高
4. **Failure mode 安全**——如果无效，只是 PSNR 不动，不会破坏训练稳定性

**实验设计**:

| Run | 配置 | 目的 |
|---|---|---|
| V8a | V6.1 + LPIPS（如上 §1.5.2 配置）| 主对比 |
| V8b | V6.1 + L1/SSIM 但 λ_img=0.20 | 控制变量：是 LPIPS 起作用还是单纯 λ_img 升高起作用 |

**Go/No-Go Gate**:
- V8a 比 V6.1 baseline PSNR 提升 ≥ 0.3 dB → Level 1.5 成功，**评估是否还需要 Level 2/3**
- V8a 比 V6.1 提升 < 0.1 dB → Level 1.5 失效，进入阶段 2
- 0.1-0.3 dB → 边际成功，进入阶段 2 验证 Level 2/3 是否能进一步提升

### 阶段 2（如 Level 1.5 不够，V8 实验后）：H1/H2 预验证

**这一步零训练成本**，是进入 Level 3 的硬门槛：

```python
# tools/precompute_token_weights.py（伪代码）
for batch in val_loader[:8]:  # 仅小批量
    z_src, z_dst = batch["z_src"], batch["z_dst"]
    x_src = decode(z_src); x_dst = decode(z_dst)
    delta = (x_dst - x_src).abs()  # [B, 1, 224, 224]
    token_w = delta.unfold(2, 14, 14).unfold(3, 14, 14).mean((-1,-2,-3))  # [B, 16, 16]
    save(token_w)
```

**Gate 阈值**:
- token_weight 的 std/mean < 0.1 → 接近 uniform → Level 3 退化为 baseline → **放弃 Level 3**
- std/mean > 0.3 且高权重区**不**对应解剖学结构（lung/lesion 边缘）→ 学到 noise → **放弃 Level 3**
- std/mean > 0.3 且高权重对应有意义结构 → 进入阶段 3

### 阶段 3（仅当阶段 2 通过）：Level 2 vs Level 3 三方对比

**实验设计**:

| Run | 配置 | 目的 |
|---|---|---|
| V9a | V8a 最佳 + Level 2 (J^T J Hutchinson 在线估计) | Jacobian 显式重加权 |
| V9b | V8a 最佳 + Level 3 (GT Δpixel 预计算权重) | 经验重加权 |
| V9c | V8a 最佳 + Level 2 离线对角近似 | 廉价版 Jacobian |

**预期排序（基于理论）**: V9a ≥ V9c > V9b（J^T J 与目标对齐，Δpixel 是 proxy）

### 阶段 4（长期，仅当阶段 3 显示 reweighting 有效）

**Level 4 简化版（chain 末端 pixel loss）**作为额外监督，与最佳 reweighting 方案叠加。

### 阶段路径决策树

```
阶段 0 (V6/V6.1 训练) ── 完成 ──→ 阶段 1 (V8: Level 1.5)
                                      │
                                      ├── ≥ 0.3 dB → 评估是否值得继续 Level 2/3
                                      ├── 0.1-0.3 dB → 阶段 2 (H1/H2 gate)
                                      └── < 0.1 dB → 阶段 2 (H1/H2 gate)
                                                       │
                                                       ├── std/mean < 0.1 → 转向其他方向（pair_loss reweight 思路作废）
                                                       └── std/mean > 0.3 → 阶段 3 (V9: Level 2/3 三方对比)
                                                                              │
                                                                              └── 最佳方案 → 阶段 4 (Level 4 叠加)
```

---

## 4. 与 Latent Forcing 的理论对话

| | Latent Forcing (Baade et al.) | 本方案 |
|---|---|---|
| **方向** | latent → pixel（语义先验引导像素生成） | pixel → latent（像素先验引导 latent transport） |
| **先验来源** | DINOv2 self-supervised features | frozen RAE decoder 的 pixel 差异图 |
| **注入方式** | 双时间变量联合去噪 | 预计算权重 / 辅助 pixel loss |
| **先验角色** | scratchpad（草稿纸，生成后丢弃） | importance weight（重要性权重，烘焙进 loss） |
| **任务** | 无条件/类条件生成 | 条件重建（配对的剂量恢复） |

共同思想：**单一空间的 MSE 不够——需要跨空间的信号来引导学习。**

Latent Forcing 证明了这个思想在生成任务中有效（ICML 2026 SOTA）。我们探索的是它在**条件重建**任务中的逆向应用。

---

## 5. 关键实验验证点

### 5.1 Level 1.5 假设（阶段 1）

| # | 假设 | 验证方法 | 成功标准 |
|---|------|---------|---------|
| H0a | LPIPS 梯度在 CT 伪-RGB 输入上非退化 | 在小批上一步反传，检查梯度 norm | LPIPS 梯度与 L1 梯度 norm 同量级 |
| H0b | LPIPS-augmented image_aux 提升解码 PSNR | V8a vs V6.1 baseline 对比 | PSNR 提升 ≥ 0.3 dB |
| H0c | LPIPS 不是只来自 λ_img 从 0.04 升到 0.20 的水趣 | V8a vs V8b（L1/SSIM 但 λ_img=0.20）对比 | V8a > V8b 提升 ≥ 0.15 dB |
| H0d | 不破坏 latent-space chain 一致性 | 对比 val_chain_normal_mse | 不退化或退化 < 5% |

**如果 H0c 失败**：说明主要贡献是 λ_img 全局提高而不是 LPIPS 本身。这本身也是一个发现——意味着当前 V6 系列的 λ_img=0.04 调个五倍就能出成果，复杂方案不必要。

### 5.2 Level 3 假设（阶段 2，预计算 gate）

以下 H1/H2 必须在**实施 Level 3 之前**零训练成本验证：

| # | 假设 | 验证方法 | 成功标准（硬 gate）|
|---|------|---------|---------|
| H1 | Pixel 差异图在 token grid 上分布不均匀 | 可视化预计算的 token_weight 分布 | std/mean ≥ 0.3 |
| H2 | 高权重 token 对应图像的重要结构区域 | 叠加 token_weight heatmap 到原图 | 高权重区对应边缘/病灶边界，不是背景噪声 |
| H3 | Pixel-weighted velocity loss 改善解码后 PSNR | 对比实验：uniform vs pixel-weighted | PSNR 提升 ≥ 0.3 dB 且 ≥ V8a 提升 的 50% |
| H4 | 不会破坏 latent-space chain 一致性 | 对比 val_chain_normal_mse | 不退化或退化 < 5% |

**H1/H2 是进入 Level 3 的硬 gate**——完全成本 < 1 小时，但能防止在 token_weight 退化为 uniform 的情况下浪费 200K 训练资源。

### 5.3 三方对比预期表（阶段 3）

基于理论分析的预期排序：

| 实验 | PSNR 提升预期（相对 V8a）| 依据 |
|---|---|---|
| V9a (Level 2 J^T J Hutchinson 在线) | +0.1 到 +0.4 dB | 与目标严格一阶对齐 |
| V9b (Level 3 Δpixel 预计算) | -0.1 到 +0.3 dB | proxy，可能与目标偏离 |
| V9c (Level 2 对角近似离线) | +0.0 到 +0.3 dB | 粗糙但低成本版 Jacobian |

**如果 V9a/V9c 均不优于 V8a**：说明 Level 1.5 已接近该思路的上限，**放弃 reweighting 路线**，转向 Level 4 或其他方向。

---

## 附录 A：为什么不只是加大 image_aux 权重

最朴素的想法是把 λ_img 从 0.04 调到更大（比如 0.5 或 1.0）。从 V3 经验看这**不可行**：

1. **梯度冲突**：pixel loss 的梯度方向和 velocity MSE 的梯度方向不总是一致——pixel loss 倾向于生成"看起来好"的图像（平滑、高 SSIM），velocity MSE 倾向于精确匹配 latent target。两者在 hop0 区域方向冲突时会互相抵消。
2. **只能监督 hop0**：加大权重只是让 hop0 的 pixel 信号更强，hop1-3 仍然盲区。
3. **V3 的现象（相关，非因果）**：step 86800 后观察到 image_aux 占比升到 88-93%。但这是**果**不是**因**：**果** 是 pair_loss 在 plateau 后什么也学不到了（梯度变小），image_aux 占比被动抓高；**因** 是 E1 budget 结论中的 transport gap 占 96%。误读这个现象为"image_aux 占比高 → 造成 plateau"会导致错误地避免提高 λ_img。

因此设计路径上：

- **Level 1.5** 选择了中途提高 λ_img 到 0.20（变化 5×不是 25×），同时插入 LPIPS 这个能饱和的软项。LPIPS 在完美重建之前梯度趋零，避免了纯 L1 加权会出现的"生成平滑图像捞取奖励"现象。
- **Level 2/3** 的巧妙之处：**不是加大 pixel 信号的权重，而是用 pixel 信息重新分配 velocity MSE 自身的注意力**。这不引入新的梯度方向，只改变 velocity MSE 内部各 token 的权重分配。

## 附录 B：Level 1.5 与 Level 2 的理论关系

Level 1.5 可以看作 **Level 2 的 "in-the-wild" 隐式实现**：

| | Level 2 显式 | Level 1.5 隐式 |
|---|---|---|
| 表达 | $(v_\text{pred}-v_\text{gt})^\top J^\top J (v_\text{pred}-v_\text{gt})$ | LPIPS·VGG features 反传自动产生 $J^\top \nabla_x \mathcal{L}_\text{LPIPS}$ |
| Jacobian 计算 | 需 Hutchinson 估计或预计算 | autograd 隐式完成 |
| Metric 在 pixel space | L2 | LPIPS（感知）|
| 仅 hop0 还是全 hop | 全 hop（pair_loss 内加权）| 仅 hop0（辅助 loss）|
| 需预计算 | 是 | 否 |

重要启示：**如果 Level 1.5 在 hop0 上已获得显著 PSNR 提升**，证明 "Jacobian-aware loss" 思路在该任务上有效，重点应该转向把这种机制扩展到全 hop（这是 Level 2 的全 hop 版本）而不是转向 Level 3（以 Δpixel 作 proxy）。
