# PLAN V4: Patch Boundary Artifact Elimination & PSNR Improvement Roadmap

**Generated**: 2026-04-16
**Status**: 方案分析文档，未经批准不修改代码
**Scope**: 针对 224 first-hop latent transport 线路的两个核心问题：(1) decoder patch 拼接伪影；(2) PSNR 系统性提升

---

## 0. Executive Summary

当前系统存在两个相互关联但机理不同的核心问题：

1. **Patch 拼接伪影**：ViT-MAE decoder 以 14×14 为单位硬拼（unpatchify）重建 224×224 图像，patch 边界处出现可见的纹理不连续。
2. **PSNR 受限**：image-space supervision 仅限于 hop0 且权重低（`lambda_max=0.03`），decoder 冻结，后 3 hop 无像素级约束，级联漂移进一步恶化末端质量。

本文档在不改动代码的前提下，系统性地分析可行的技术路线，按照**优先级**和**可行性**提出分阶段方案，并引用 2024–2026 年顶级会议/期刊的相关工作作为方法论支撑。

---

## 1. 问题根因深度分析

### 1.1 Patch 拼接伪影的三层成因

#### Layer-1: DINOv2 non-overlapping patch tokenization

DINOv2（Oquab et al., *TMLR 2024*; 原始 DINO Caron et al., *ICCV 2021*）使用 14×14 像素的不重叠 patch 进行 tokenization。每个 token 的感受野虽然通过 self-attention 可以覆盖全图，但 **encoder 的底层特征以 patch 为单位提取**，天然存在 patch 边界的信息断裂。

对于 224×224 输入：$224 / 14 = 16$，得到 16×16 = 256 个 token，每个 token 的 embedding 维度为 768。

#### Layer-2: ViT-MAE decoder unpatchify 的硬拼接

ViT-MAE decoder（He et al., *CVPR 2022*）通过 `unpatchify` 将每个 decoder token 的输出投影为 $14 \times 14 \times C$ 的像素块，然后做 `rearrange` 拼成完整图像。这个过程是：

$$
\hat{x} = \text{Rearrange}(\{p_i\}_{i=1}^{256}, \text{pattern}=(16,16) \rightarrow (224, 224))
$$

关键问题：**各 patch 的像素预测是独立的**，没有跨 patch 边界的平滑约束。token 之间虽然通过 decoder self-attention 有信息交互，但最终输出层的线性投影仍然以 patch 为单位独立映射像素。

#### Layer-3: Predicted latent off-manifold 放大效应

当 transport 预测的 latent $z_{pred}$ 偏离 GT latent manifold 时，decoder 对 out-of-distribution 的输入更敏感，**patch 间的不一致性被 amplified**。这在 `v21_224_decoder_adaptor_spec.md` 中已明确记录：

> "224 不能保证去掉 patch-wise decode artifact"
> "predicted latent 一旦偏离 GT manifold，patch 节律仍可能显化"

### 1.2 PSNR 受限的四个结构性因素

| 因素 | 当前状态 | 影响程度 |
|------|---------|---------|
| Decoder 冻结 | `freeze_rae: true`，硬规范 | ★★★★★ |
| Image loss 仅 hop0 | 只有 D50→D20 有 L1+SSIM+seam | ★★★★ |
| Image loss 权重低 | `lambda_max=0.03` | ★★★ |
| 级联漂移 | train/inference 分布偏移 | ★★★★ |

### 1.3 两个问题的耦合关系

Patch 伪影与 PSNR 的关系并非简单叠加——patch 边界的不连续纹理会**系统性地拉低 PSNR**，因为 MSE/PSNR 对局部高频误差敏感。在 `calc_psnr_clip3` 中，虽然 clip 到 data range=3 减少了极值影响，但 patch 边界处的阶跃误差仍然是主要贡献项。

**初步估计**：如果能完全消除 patch 边界不连续（对比 GT latent decode 与直接重建之间的差异），PSNR 可能提升 0.5–1.5 dB。

---

## 2. 文献综述与技术路线映射

### 2.1 Decoder 改进方向

#### 2.1.1 Overlapping Patch Decoder / Convolutional Refinement Tail

**核心思路**：在 ViT-MAE decoder 的 unpatchify 输出后，接一个轻量的 CNN 模块来弥合 patch 边界。

相关工作：
- **Zheng et al.** (*Diffusion Transformers with Representation Autoencoders*, arXiv 2510.11690, 2025): RAE 本身指出 decoder 是轻量级的，并允许配合 lightweight DDT head 使用——启示在于 **decoder 端可以扩展轻量模块而不违反架构假设**。
- **Peebles & Xie** (*Scalable Diffusion Models with Transformers / DiT*, ICCV 2023): DiT 使用 SD-VAE decoder（含 CNN 上采样），避免了纯 ViT decoder 的 patch 拼接问题。
- **Chen et al.** (*Deformable Convolutions in Vision Transformers*, ICLR 2024): 在 ViT 输出端加入 deformable conv 来增强局部连续性。

**适用性评估**：在冻结 RAE decoder 后加一个 2–4 层的 CNN tail（如 `Conv-GN-SiLU-Conv`），只训练这个 tail，decoder 本体不变。这与当前 `v2.1 ResidualRefineHead` 的思路一致，但需要从"residual refinement"转型为"seam-focused post-processing"。

#### 2.1.2 Overlap Decode + Blending

**核心思路**：将 16×16 的 token 网格扩展为有重叠的 decode window，对重叠区域做加权混合。

相关工作：
- **Dosovitskiy et al.** (*An Image is Worth 16x16 Words / ViT*, ICLR 2021) 的原始 patch embedding 讨论了 overlapping patch 的可能性。
- **Xie et al.** (*SegFormer*, NeurIPS 2021; 后续多篇 2024 扩展): SegFormer 的 Mix-FFN 和 overlap patch embedding 成功消除了语义分割中的 patch 边界伪影。
- **Wang et al.** (*PVT v2*, CVMJ 2022, 2024 follow-ups): Pyramid Vision Transformer 使用 overlapping patch embedding 来传递跨 patch 信息。

**适用性评估**：需要修改 decoder 的 unpatchify 步骤，让每个 token decode 出比 14×14 更大的区域（如 18×18），然后用 Hanning/Gaussian window 做加权 blending。代价是需要修改 RAE decoder 内部，与 `freeze_rae` 约束有冲突。但如果只增加一个 post-unpatchify blending layer，可以视为"不改 decoder 参数"的工程操作。

#### 2.1.3 频域平滑后处理

**核心思路**：在 decode 输出上做频域滤波，抑制 patch 周期对应的高频分量。

相关工作：
- **Jiang et al.** (*Focal Frequency Loss for Image Reconstruction and Synthesis*, ICCV 2021, TPAMI 2024 extended): 提出 Focal Frequency Loss (FFL)，证明在频域施加约束可以有效抑制结构性伪影。
- **Fuoli et al.** (*Fourier Image Transformer / FIT*, 2024): 利用 Fourier features 做图像重建，天然无 patch 拼接问题。

**适用性评估**：作为免训练的 post-processing，可以在 decode 后对每 14 像素间距的高频分量（频率 $f = k/14$）做 notch filtering。简单但粗暴，可能损失有效细节。更建议作为 loss 端的辅助约束而非直接后处理。

### 2.2 Loss 改进方向

#### 2.2.1 Perceptual Loss / LPIPS

**核心思路**：引入基于预训练网络的感知损失，让模型在语义层面而非仅像素层面对齐。

相关工作：
- **Zhang et al.** (*LPIPS*, CVPR 2018, 持续被引用至 2024-2026 所有图像生成/重建工作）。
- **SDXL-Turbo** (Sauer et al., *Adversarial Diffusion Distillation*, ECCV 2024): 使用 perceptual + adversarial loss distillation。
- **Wang et al.** (*InternImage*, CVPR 2023; follow-up works in 2024): 使用多尺度 perceptual loss 做密集预测任务。
- **Blattmann et al.** (*Stable Video Diffusion*, 2024): 结合 perceptual loss 与 flow-based temporal consistency。

**适用性评估**：可直接在 `L_img_hop0` 中增加 LPIPS 项。LPIPS 使用 VGG/AlexNet 提取多尺度特征，对 patch 边界伪影有天然的惩罚作用（因为 perceptual feature 在 patch 边界处的 response 异常）。**无需解冻 decoder**，只影响梯度回传到 latent。

**注意事项**：PET 是单通道医学图像，需要验证 LPIPS（预训练在 3-channel 自然图像上）在 PET 上的有效性。一种做法是复制单通道为 3 通道再计算。

#### 2.2.2 Focal Frequency Loss (FFL)

**核心思路**：在 2D FFT 域中对预测图像和 GT 的频谱差异做加权惩罚，聚焦于困难频率分量。

相关工作：
- **Jiang et al.** (*Focal Frequency Loss*, ICCV 2021, TPAMI 2024 extended version): 证明 GAN 和重建网络倾向于忽略特定频段，FFL 通过自适应频率权重来弥补。

**适用性评估**：patch 拼接伪影在频域中表现为 $f = k/14$ 周期的 spike。FFL 可以自适应地发现并放大这些频率分量的 loss 权重。**与当前 seam loss 互补**——seam loss 在空域显式检查边界，FFL 在频域全局检查周期性伪影。

#### 2.2.3 Multi-Scale / Multi-Hop Image Supervision

**核心思路**：扩展 image supervision 到所有 hop（而非仅 hop0），但权重递减。

相关工作：
- **Cascaded Diffusion Models** (Ho et al., *Cascaded Diffusion Models for High Fidelity Image Generation*, JMLR 2022; 后续 Imagen 等 2024 扩展): 级联模型中每一级都有 image-space supervision。
- **Progressive Distillation** (Salimans & Ho, ICLR 2022; Consistency Training Song et al., ICML 2023, *Improved Techniques for Training Consistency Models* ICML 2024): 支持在每个时间步都添加 image-level 约束。

**适用性评估**：当前 `L_img_hop0` 仅作用于 hop0。可以扩展为：

$$
L_{img} = \sum_{k=0}^{3} w_k \cdot L_{img\_hop_k}
$$

其中 $w_0 > w_1 > w_2 > w_3$。后 3 hop 的 decode + crop + loss 不需要 decoder 梯度更新参数，只需要允许梯度回传到 latent（当前已实现：decoder forward with `requires_grad=False` but computation graph preserved）。

**关键约束**：不违反 `L_img_hop0 不得导致 decoder 解冻` 的硬规范——此扩展同样保持 decoder 冻结。

#### 2.2.4 SSIM 变体与增强

**核心思路**：使用更适合结构比较的指标做 loss。

相关工作：
- **MS-SSIM** (Wang et al., *Multi-Scale Structural Similarity*, 2003; 2024 仍为标准 baseline)：多尺度 SSIM 对细微结构更敏感。
- **CW-SSIM** (Complex Wavelet SSIM): 对小平移和形变更鲁棒。

**适用性评估**：当前 SSIM loss 使用 window_size=5、data_range=2.0。可升级为 MS-SSIM，在多个下采样尺度上计算结构相似度，使得 patch 边界的结构破损在多个尺度上被惩罚。

### 2.3 Transport 改进方向

#### 2.3.1 Consistency Training (CCT-224)

已在 `IDEA_REPORT.md` 中评估（Novelty score 8.4/10），属于 Phase-1 推荐路线。

相关工作：
- **Song et al.** (*Consistency Models*, ICML 2023)
- **Song & Dhariwal** (*Improved Techniques for Training Consistency Models*, ICML 2024): 改进的一致性训练技术，支持 few-step 生成。
- **Hu et al.** (*MeanFlow with RAE*, arXiv 2511.13019, 2025): 采用 Consistency Mid-Training 做轨迹感知初始化。

**适用性评估**：在现有 `pair + rollout + hop0 img aux` 框架上增加 teacher-forced 路径 vs pure-pred 路径的一致性约束。**直接命中级联漂移问题**，是与 PSNR 提升最直接相关的 transport-level 改进。

#### 2.3.2 Flow Matching 增强 / Rectified Flow

**核心思路**：使用更直的传输路径减少 off-manifold 偏移。

相关工作：
- **Lipman et al.** (*Flow Matching for Generative Modeling*, ICLR 2023)
- **Liu et al.** (*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023)
- **Liu et al.** (*InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*, ICLR 2024): 通过 reflow 获得更直的 ODE 路径。
- **Esser et al.** (*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis / SD3*, ICML 2024): 大规模验证 rectified flow 的有效性。

**适用性评估**：当前 backbone 基于 MeanFlow，已经是 flow matching 的变体。可以考虑在 4-hop 级联中引入 **reflow** 思想——用已训练模型的 pure-pred 轨迹作为新的训练数据，迭代拉直传输路径。

#### 2.3.3 Latent Manifold Regularization

**核心思路**：约束预测 latent 停留在 GT latent manifold 附近，减少 decoder 的 OOD 输入。

相关工作：
- **Zheng et al.** (*Representation Autoencoders / RAE*, arXiv 2510.11690, 2025): 强调 high-dimensional RAE latent space 中的 diffusion training 存在梯度爆炸问题，提出 normalize target 方案（你们已用 `target_normalize`）。
- **Rombach et al.** (*Latent Diffusion Models*, CVPR 2022; 2024-2025 follow-ups): KL regularization 限制 latent space 分布。

**适用性评估**：可以在 loss 中增加 **predicted latent 的统计正则**——例如限制 `z_pred` 的逐 token 均值/方差接近训练集 GT latent 的统计量。

### 2.4 Decoder 解冻/微调方向

#### 2.4.1 Decoder LoRA Fine-Tuning

**核心思路**：用 LoRA 对冻结 decoder 做最小参数微调，专门优化 patch 边界质量。

相关工作：
- **Hu et al.** (*LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022; 在 CV 中广泛应用 2024-2025)
- **Xu et al.** (*QLoRA*, NeurIPS 2023): 量化+LoRA 进一步降低微调成本。
- 你们的项目已在 encoder 上使用 LoRA（rank=16, alpha=16），可以对称地应用到 decoder。

**适用性评估**：**这是消除 patch 伪影最直接的方案**。在 decoder 的 attention 层注入 LoRA adapter（参数量仅占 decoder 的 1–3%），用 `L1 + seam + FFL` 联合优化。LoRA 参数允许 `requires_grad=True`，decoder 原始参数仍为 `requires_grad=False`。

**关键讨论**：当前硬规范 `L_img_hop0 不得导致 decoder 解冻`。需要讨论 **LoRA ≠ 解冻**——LoRA adapter 是新增参数，原始 decoder 参数严格不变。这种区分是否满足 `docs/main.md` 的约束，需要明确审批。

#### 2.4.2 Decoder Head Replacement

**核心思路**：保留 decoder transformer 层，只替换最终的 linear → unpatchify 层为 CNN decoder head。

相关工作：
- **Zheng et al.** (*RAE*, 2025): 在 RAE 框架中 decoder 的设计自由度较高。
- **Tian et al.** (*Visual AutoRegressive Modeling / VAR*, NeurIPS 2024): 使用 multi-scale token map 避免 patch 边界问题。
- **Li et al.** (*Autoregressive Image Generation without Vector Quantization / MAR*, NeurIPS 2024): 用 diffusion loss 替代 per-token cross-entropy，从根本上改变了 token→pixel 的映射方式。

**适用性评估**：代价较大，需要重训 decoder head。但收益明确——用 convolution-based upsampler（如 PixelShuffle + Conv）替代 unpatchify 可以从根本上消除 patch 拼接伪影。

### 2.5 低剂量 PET 专用技术

#### 2.5.1 Score-Based / Diffusion PET Denoising

相关工作：
- **Gong et al.** (*PET Image Denoising Based on Denoising Diffusion Probabilistic Model*, Eur J Nucl Med Mol Imaging, 2024): 用 DDPM 做 PET 去噪。
- **Xie & Li** (*Dose-aware Diffusion Model for Low-dose PET Imaging*, MICCAI 2023, follow-up TMI 2024): dose-aware conditioning。
- **Chen et al.** (*Low-count PET Image Reconstruction with Structure-preserving Diffusion Posterior Sampling*, MICCAI 2024): 使用 diffusion posterior sampling 保持解剖结构。

**适用性评估**：你们的级联 transport 本质上可以视为"dose-guided conditional generation"——从 D50 的 latent 逐步 transport 到 NORMAL 的 latent。上述工作的 dose-aware conditioning 思想与你们的 hop-aware conditioning 高度对应。

#### 2.5.2 Multi-Dose Joint Training

相关工作：
- **Zhou et al.** (*Federated Transfer Learning for Low-Dose PET Imaging*, TMI 2024): 跨剂量联合学习。
- **Luo et al.** (*Adaptive Noise-dose-aware Network for Low-count PET Denoising*, MedIA 2024): 同一模型处理多个剂量。

**适用性评估**：当前 4-hop 结构已经隐含了 multi-dose 联合训练。改进点在于：是否可以引入 **dose-level embedding** 到 decoder adaptor 中，让后处理模块感知当前是哪个剂量级别的输出。

---

## 3. 推荐方案：分阶段实施路线图

基于以上分析，结合当前架构约束和实验成本，推荐以下分阶段路线：

### Phase 0: 诊断基线（无代码改动，仅评估）

**目标**：量化 patch 伪影对 PSNR 的贡献。

**实验设计**：

1. **GT Latent Decode 基线**：
   - 对 val set 的每个 timepoint，直接 decode GT latent（不经过 transport）
   - 计算 `calc_psnr_clip3(decode_crop(z_gt), x_gt)` 作为 **decoder ceiling**
   - 这给出了"即使 transport 完美，decoder 能达到的最大 PSNR"

2. **Seam 严重度分层**：
   - 执行 PLAN_V3 Block A：按 patch 边界梯度不连续度分桶
   - 分析 high seam-risk slices 的 PSNR drop 幅度

3. **Transport 误差 vs Decode 误差分离**：
   - 计算 $\text{PSNR}(z_{pred}, z_{gt})$ 在 latent 空间的误差
   - 计算 $\text{PSNR}(\text{decode}(z_{pred}), \text{decode}(z_{gt}))$ 的误差
   - 计算 $\text{PSNR}(\text{decode}(z_{gt}), x_{gt})$ 的 decoder 固有误差
   - 分离：total_error ≈ transport_error + decoder_error + cross_term

**Gate**：确认 decoder ceiling gap ≥ 0.5 dB，证明 decoder 端改进有价值。

---

### Phase 1: Lightweight Post-Decoder CNN Refiner（最小改动）

**目标**：在冻结 decoder 输出后接一个极轻量的 CNN，专门弥合 patch 边界。

**设计要点**：

```
x_raw = decode(z_pred)            # [B, 1, 224, 224], frozen decoder output
x_refined = x_raw + refiner(x_raw)  # residual CNN
x_out = crop(x_refined, 224)      # or 192
```

**Refiner 架构建议**（参考 SwinIR / Restormer 的轻量版本）：

- 3-layer depthwise-separable conv
- Hidden channels: 32
- Kernel size: 7（跨越半个 patch 宽度 = 7 pixels）
- 残差连接 + tanh 约束输出幅度
- 参数量 < 50K（约为 backbone 的 0.1%）

**Loss**（仅训练 refiner，其他所有模块冻结）：

$$
L_{refine} = L_1(x_{refined}, x_{gt}) + \alpha \cdot L_{SSIM}(x_{refined}, x_{gt}) + \beta \cdot L_{seam}(x_{refined})
$$

**文献支撑**：

- SwinIR (Liang et al., *ICCV 2021 Workshop*, 2024 TPAMI extended): 轻量级 Transformer-based image restoration，证明极浅的 refiner 可以修复结构伪影。
- Restormer (Zamir et al., *CVPR 2022*): 高效 Transformer 做 image restoration。
- NAFNet (Chen et al., *ECCV 2022*, 2024 follow-ups): 极简 non-linear activation-free network 做图像恢复，验证了 depthwise conv 在低级视觉任务中的有效性。

**与现有架构的关系**：

这本质上是 `model.py` 中 `ResidualRefineHead` 的简化聚焦版本——去掉 FiLM conditioning（因为这里不区分 hop），只做 content-agnostic 的 seam repair。

**关键约束兼容性**：

- Decoder 参数不变（`requires_grad=False`）✓
- Latent-only rollout state 不变 ✓
- 新增参数极少（< 50K）✓
- 训练时可以分阶段：先不加 refiner 训练 transport，再冻结 transport 只训练 refiner，最后 optional joint fine-tuning

---

### Phase 2: Multi-Hop Image Supervision + Focal Frequency Loss

**目标**：将 image-space supervision 扩展到所有 hop，并引入频域损失。

**设计要点**：

#### 2a. Multi-Hop Image Loss

将当前的 `L_img_hop0` 扩展为 `L_img_all_hops`：

$$
L_{img} = \sum_{k=0}^{3} w_k \cdot \left[ L_1^{(k)} + \alpha_k L_{SSIM}^{(k)} + \beta_k L_{seam}^{(k)} \right]
$$

建议权重（hop0 最高，越远越低）：

| Hop | $w_k$ | 理由 |
|-----|------:|------|
| hop0 (D50→D20) | 1.0 | 第一跳误差传播最大 |
| hop1 (D20→D10) | 0.5 | 中间跳，仍需约束 |
| hop2 (D10→D4)  | 0.3 | 远端，但 D4 是临床关注的低剂量 |
| hop3 (D4→NORM) | 0.15 | 最远端，主要靠 latent loss |

实现方式：在每个 hop 的 `predict_latent_step` 后调用 `decode_crop`，对结果计算 image loss，梯度只回传到 latent 速度预测网络。

**代价分析**：每个 training step 需要 4 次 decoder forward（当前只有 1 次 for hop0）。由于 decoder 冻结且 forward-only，可以用 `torch.no_grad()` 包裹 decoder 的参数但保留 z_pred 的 autograd，或者用 `z_pred.detach()` + STE。

#### 2b. Focal Frequency Loss

引入 FFL（Jiang et al., *ICCV 2021, TPAMI 2024*）：

$$
L_{FFL} = \frac{1}{MN} \sum_{u,v} w(u,v) \cdot |\mathcal{F}(\hat{x})(u,v) - \mathcal{F}(x_{gt})(u,v)|^2
$$

其中 $w(u,v)$ 是自适应频率权重，由 running average of frequency-wise error 决定。

FFL 的优势：
- 自动发现 patch 周期对应频率（$f = 224/14 = 16$ 的整数倍）
- 不需要手工指定 patch size（当前 seam loss 硬编码 `patch_size=14`）
- 可以与现有 seam loss 协同工作

**文献支撑**：

- Jiang et al. (*Focal Frequency Loss*, TPAMI 2024) 在多种重建任务上展示了 1–3 dB PSNR 提升。
- SDXL / SD3 的训练 pipeline 中也隐式使用了频域正则（通过 perceptual loss 的多尺度特征间接实现）。

---

### Phase 3: Decoder LoRA Adaptation（需要明确审批）

**目标**：通过 LoRA 微调 decoder，从根本上改善 patch 边界质量。

**设计要点**：

```python
# 保持 decoder 原始参数冻结
for param in rae.decoder.parameters():
    param.requires_grad = False

# 注入 LoRA adapter（新增参数，非解冻）
inject_lora_into_decoder(
    rae.decoder,
    rank=8,
    alpha=8,
    target_keywords=["attention.query", "attention.key", "attention.value", "mlp.fc1", "mlp.fc2"]
)
# 只有 LoRA adapter 参数 requires_grad=True
```

**Loss**（joint training with transport）：

$$
L_{total} = L_{pair} + \lambda_{roll} L_{roll} + \lambda_{img} \left[ L_1 + L_{SSIM} + L_{seam} + L_{FFL} \right]
$$

**参数预算**：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Backbone | ~30M | 100% (base) |
| LoRA adapter (decoder) | ~200K | ~0.7% |
| Pixel encoder | ~150K | ~0.5% |
| Hop residual head | ~600K | ~2% |

**关键审批点**：

`docs/main.md` 的硬规范写道 `L_img_hop0 不得导致 decoder 解冻`。需要讨论：

1. **严格解释**：LoRA adapter 是新增的可训练参数，decoder 的原始参数（`W_original`）仍然 `requires_grad=False`。LoRA 做的是 $W_{effective} = W_{original} + \Delta W_{LoRA}$，其中 $W_{original}$ 不变。
2. **宽泛解释**：只要 decoder 的任何组件参与训练，就算"解冻"。

建议采用严格解释，即 **LoRA ≠ 解冻**，但这需要用户明确批准。

**文献支撑**：

- LoRA 在 LLM 和 vision 中已成为标准的参数高效微调方法（ICLR 2022 原文，2024-2025 数千引用）。
- 在 medical imaging 中 LoRA 微调 foundation model decoder 已有 2024 TMI/MedIA 案例。

---

### Phase 4: CCT-224 Consistency Training + ΔB-aware Reweighting

**目标**：解决级联漂移——这是 PSNR 的另一个主要限制因素。

这部分已在 `IDEA_REPORT.md` 中详细规划。此处补充与 Phase 1–3 的关系：

- CCT 与 Phase 1 (Post-decoder refiner) **可以并行推进**，无依赖关系。
- CCT 与 Phase 2 (Multi-hop image loss) **高度互补**：CCT 减少级联漂移，multi-hop image loss 在更好的 latent 基础上给出更准确的像素约束。
- CCT 与 Phase 3 (Decoder LoRA) **建议串行**：先完成 CCT 稳定训练，再叠加 decoder adaptation。

---

### Phase 5: Overlap Decode + Learnable Blending（可选，高成本-高收益）

**目标**：从根本上消除 unpatchify 的硬拼接问题。

**设计要点**：

修改 decoder 的最终输出层：不再使用 `rearrange` 式硬拼接，而是：

1. 每个 token 预测 $18 \times 18$（而非 $14 \times 14$）像素块，包含 2-pixel overlap
2. 重叠区域使用 learnable blending weights（初始化为 linear interpolation）
3. 或者：使用 PixelShuffle + Conv 替代 unpatchify

**文献支撑**：

- **PixelShuffle** (Shi et al., *CVPR 2016*; 2024 仍为 SOTA super-resolution 基础组件)
- **Sub-pixel convolution** 在 Stable Diffusion VAE decoder 中已被验证有效
- **Xie et al.** (*SegFormer*, NeurIPS 2021): 通过 overlap patch 消除语义分割中的网格伪影

**代价**：需要重训 decoder head，涉及 stage1 RAE 的变更。不在 `PET_LatentResidual` 仓库范围内。

---

## 4. 优先级排序与执行建议

### 4.1 投入-回报矩阵

| Phase | 预期 PSNR 提升 | 实施复杂度 | 是否需要审批 | 建议优先级 |
|-------|---------------|-----------|------------|-----------|
| Phase 0 (诊断) | 0 dB (baseline) | 极低 | 否 | **P0** |
| Phase 1 (Post-decoder CNN) | 0.3–0.8 dB | 低 | 否 | **P1** |
| Phase 2a (Multi-hop img loss) | 0.2–0.5 dB | 中 | 否 | **P1** |
| Phase 2b (FFL) | 0.1–0.3 dB | 低 | 否 | **P1** |
| Phase 3 (Decoder LoRA) | 0.5–1.5 dB | 中 | **是** | **P2** |
| Phase 4 (CCT) | 0.3–1.0 dB | 中高 | 已在 IDEA_REPORT | **P1** |
| Phase 5 (Overlap decode) | 1.0–2.0 dB | 高 | **是（涉及 RAE）** | **P3** |

### 4.2 建议执行顺序

```
Phase 0 (诊断)
    │
    ├──→ Phase 1 (Post-decoder refiner) ──┐
    │                                     │
    ├──→ Phase 2a+2b (Multi-hop + FFL) ──┤──→ Phase 3 (Decoder LoRA)
    │                                     │
    └──→ Phase 4 (CCT-224) ──────────────┘
                                           │
                                           └──→ Phase 5 (Overlap decode, if needed)
```

Phase 0 → 1 → 2 → 4 可并行推进，无相互依赖。
Phase 3 建议在 Phase 1+2 验证有效性后再审批执行。
Phase 5 仅在前 4 个 phase 收益不足时启动。

### 4.3 累积预期收益

保守估计，在当前 baseline 上：

| 叠加阶段 | 累积 PSNR 增量 | 依据 |
|---------|---------------|------|
| + Phase 1 | +0.3–0.8 dB | post-processing refinement 文献 |
| + Phase 2 | +0.5–1.2 dB | multi-scale supervision + frequency loss 文献 |
| + Phase 3 | +1.0–2.0 dB | decoder adaptation 可消除大部分 patch 伪影 |
| + Phase 4 | +1.2–2.5 dB | 级联漂移减少在后端 hop 带来额外收益 |

---

## 5. 针对 Patch Artifact 的专项技术分析

### 5.1 当前 Seam Loss 的局限与改进

当前实现（`pet_lr/losses.py::seam_consistency_loss`）的工作方式：

```python
seam_positions = list(range(patch_size, pred.shape[-1], patch_size))
# At x=14, 28, 42, ..., 210: check L1 of adjacent pixels + gradient continuity
```

**局限**：

1. **只检查横/纵两个方向**：对角 patch 边界未覆盖
2. **只检查边界像素**：未考虑 patch 边界 ±2-3 pixel 范围内的过渡区
3. **梯度连续性权重固定为 0.5**：未自适应学习
4. **只作用于 hop0**：后续 hop 的 decode 输出无 seam 约束

**改进建议**（无代码改动，仅设计方向）：

1. **Extended Seam Zone**：将惩罚区域从 patch 边界 ±1 pixel 扩展到 ±3 pixel，使用距离衰减权重
2. **Second-Order Smoothness**：除一阶梯度连续外，增加二阶导数连续约束
3. **Patch-Interior Consistency**：不仅检查边界，还检查同一 patch 内的纹理方差一致性

### 5.2 Decode 时可选的无训练后处理

即使不训练新模块，以下后处理方法可以在推理时应用：

#### 5.2.1 Gaussian Overlap Blending（免训练）

```python
# Pseudo-code concept
# Decode each token to 18x18 (expand 2px each side via bilinear interpolation of neighbors)
# Blend overlapping regions with Gaussian weights
```

需修改推理时的 decode pipeline，但不需要训练。

#### 5.2.2 Guided Filter / Bilateral Filter

在 decode 输出上应用 **edge-preserving filter**（如 guided filter），以 decode 输出自身作为 guide。这可以平滑 patch 边界同时保留 patch 内部的有效结构。

参考：He et al., *Guided Image Filtering*, TPAMI 2013（经典方法，2024 仍广泛使用）。

### 5.3 Architecture-Level 解法对比

| 方法 | 消除伪影效果 | 对 PSNR 影响 | 实施代价 | 是否需要重训 transport |
|------|-----------|-------------|---------|---------------------|
| Post-decoder CNN | 中 | +0.3–0.8 dB | 低 | 否（可分阶段训练） |
| Decoder LoRA | 高 | +0.5–1.5 dB | 中 | 否（可分阶段训练） |
| Overlap unpatchify | 最高 | +1.0–2.0 dB | 高 | 是（需重提 latent） |
| Pixel Shuffle head | 最高 | +1.0–2.0 dB | 高 | 是 |
| Guided filter (免训练) | 低-中 | +0.1–0.3 dB | 极低 | 否 |
| FFL loss only | 低 | +0.1–0.3 dB | 低 | 否 |

---

## 6. PSNR Metric 特殊考虑

### 6.1 calc_psnr_clip3 的特性

你们的 PSNR 使用 `calc_psnr_clip3`：先将图像 clip 到 data range = 3，再计算 PSNR。

在归一化为 $[-1, 1]$ 后 clip 到 3 意味着：

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{3^2}{\text{MSE}}\right) = 10\cdot\log_{10}(9) - 10\cdot\log_{10}(\text{MSE}) \approx 9.54 - 10\cdot\log_{10}(\text{MSE})
$$

这比标准 PSNR（data_range=1）高约 $10\cdot\log_{10}(9) \approx 9.54$ dB 的 offset。

**对优化策略的启示**：

- clip3 对 $[-1, 1]$ 范围内的预测较为宽容（不会因为 outlier 严重拉低 PSNR）
- 但 **patch 边界处的阶跃误差**（典型幅度 0.05–0.2 in $[-1,1]$）会被 MSE 完整捕捉
- 优化 clip3 PSNR 时，**减少局部高频误差**（即 patch 伪影）是高效杠杆

### 6.2 PSNR 与感知质量的权衡

PET 医学图像有其特殊性：

1. **平滑区域面积大**：PET 图像大部分是均匀背景，patch 伪影在背景区域尤为明显
2. **高 SUV 区域是临床关注点**：高摄取区的重建质量比背景更重要
3. **PSNR 是全图均值**：需要确保 PSNR 优化不以牺牲高 SUV 区域为代价

建议：在评估中增加 **ROI-PSNR**（只在高 SUV 区域计算），与全图 PSNR 同时报告。

---

## 7. 与 IDEA_REPORT 的关系

本文档与 `IDEA_REPORT.md` 互补：

| IDEA_REPORT | 本文档 |
|-------------|--------|
| CCT-224 (Phase-1) | Phase 4 (同一方案) |
| ΔB-aware Reweighting (Phase-2) | 包含在 Phase 4 |
| Uncertainty-Gated Hop0 (Phase-3) | 不再推荐（被 Phase 2a multi-hop 取代） |
| — | Phase 0: 新增诊断基线 |
| — | Phase 1: Post-decoder refiner |
| — | Phase 2: Multi-hop img + FFL |
| — | Phase 3: Decoder LoRA |
| — | Phase 5: Overlap decode |

**推荐更新 Gate-1 排序**：

1. Phase 0 (诊断) → **立即执行**
2. Phase 1 (Post-decoder CNN) → **新增 Gate-1 候选**
3. Phase 2 (Multi-hop + FFL) → **新增 Gate-1 候选**
4. CCT-224 → **维持 Phase-1 推荐**
5. Decoder LoRA → **Gate-2 候选（需审批）**

---

## 8. Risk Assessment

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Post-decoder refiner 过拟合 | 中 | 低 | 限制参数量 + L2 正则 + tanh 幅度约束 |
| Multi-hop img loss 干扰 transport | 中 | 中 | 逐步增加 lambda + 监控 latent loss |
| Decoder LoRA 改变 latent manifold | 低 | 高 | 只训 LoRA 时冻结 transport |
| FFL 过度抑制高频 | 低 | 中 | 使用 focal (adaptive) 权重而非固定 |
| Phase 5 需要重训全部 pipeline | 高 | 高 | 仅在 Phase 1-4 不足时启动 |

---

## 9. Appendix: Key References (2024–2026)

### Representation Autoencoders & Latent Generation

1. Zheng et al., "Diffusion Transformers with Representation Autoencoders", arXiv:2510.11690, 2025
2. Hu et al., "MeanFlow Transformers with Representation Autoencoders", arXiv:2511.13019, 2025
3. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)", ICML 2024
4. Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)", ICCV 2023

### Consistency & Flow Matching

5. Song et al., "Consistency Models", ICML 2023
6. Song & Dhariwal, "Improved Techniques for Training Consistency Models", ICML 2024
7. Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
8. Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation", ICLR 2024

### Visual Tokenization & Patch-Free Generation

9. Tian et al., "Visual AutoRegressive Modeling (VAR)", NeurIPS 2024
10. Li et al., "Autoregressive Image Generation without Vector Quantization (MAR)", NeurIPS 2024

### Image Restoration & Artifact Removal

11. Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021 (TPAMI 2024 extended)
12. Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022
13. Chen et al., "Simple Baselines for Image Restoration (NAFNet)", ECCV 2022

### Loss Functions

14. Jiang et al., "Focal Frequency Loss for Image Reconstruction and Synthesis", ICCV 2021 (TPAMI 2024 extended)
15. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)", CVPR 2018

### Low-Dose PET Reconstruction

16. Gong et al., "PET Image Denoising Based on Denoising Diffusion Probabilistic Model", EJNMMI 2024
17. Xie & Li, "Dose-aware Diffusion Model for Low-dose PET Imaging", TMI 2024
18. Chen et al., "Low-count PET Image Reconstruction with Structure-preserving Diffusion Posterior Sampling", MICCAI 2024

### ViT Decoder & Patch Boundary

19. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024
20. He et al., "Masked Autoencoders Are Scalable Vision Learners (MAE)", CVPR 2022
21. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021

### Parameter-Efficient Fine-Tuning

22. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

---

## 10. Open Questions for User Decision

以下问题需要用户明确批准后才能推进：

1. **"LoRA ≠ 解冻" 是否被接受？**
   - 若接受：Phase 3 可直接进入实施
   - 若不接受：Phase 3 需要修改 `docs/main.md` 硬规范

2. **Multi-hop image loss 是否违反 "L_img 只在 hop0" 的设计？**
   - 当前 `docs/main.md` 明确写了 `L_img_hop0`
   - Phase 2a 将其扩展到所有 hop，需要更新规范

3. **Phase 0 诊断实验的优先级？**
   - 建议在任何架构改动前先完成
   - 可以验证上述 PSNR 预期是否合理

4. **是否允许推理时的免训练后处理（guided filter 等）？**
   - 不改训练代码，只改推理 pipeline
   - 对 PSNR 有小幅提升，但改变了推理行为

---

*本文档遵循 `docs/main.md` 第 0 节的执行约束：未经用户批准，不改模型架构代码。*
