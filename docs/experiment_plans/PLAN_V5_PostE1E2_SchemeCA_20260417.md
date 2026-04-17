# PLAN V5: Post-E1/E2 实验路线 — Scheme C + Scheme A

**Date**: 2026-04-17
**Status**: 实施中（代码已推送，待服务器运行）
**Branch**: `foc_lite_hop0`
**Supersedes**: PLAN_V4 中关于 patch artifact 预期的部分（E1 证伪了 decoder 是主要瓶颈的假设）

---

## 1. E1/E2 关键发现（完整 val 7403 slices）

### E1: 误差预算分解

| 剂量 | Decoder Ceiling | E2E 实际 | 总 Gap | Transport 占比 | Decoder 占比 |
|------|----------------|---------|--------|--------------|-------------|
| D20 | 46.64 dB | 35.49 dB | 11.15 dB | **96.3%** (10.74 dB) | 3.7% (0.41 dB) |
| D10 | 48.74 dB | 35.86 dB | 12.88 dB | 95.7% | 4.3% |
| D4 | 50.83 dB | 36.42 dB | 14.41 dB | 97.0% | 3.0% |
| NORMAL | 52.63 dB | 36.74 dB | 15.89 dB | 98.8% | 1.2% |

**结论**：
- **Transport 误差绝对主导**——decoder gap 仅 0.41 dB (D20)
- Patch 伪影对 PSNR 的影响远低于 PLAN_V4 估计的 0.5–1.5 dB
- 224 管线的 decoder 质量已经很好（ceiling 46.64 dB）
- 瓶颈 100% 在速度场预测精度

### E2: FOC Gap 测量

| 指标 | 值 |
|------|-----|
| 相对 gap `‖z_full − z_half‖/‖z_full‖` | **0.17%** |
| Half-step vs Full-step PSNR | **−0.0445 dB** (half 更差) |
| Gap↔Deficit 相关性 | **r = 0.44** |

**结论**：
- ODE 积分误差极小（0.17% << 5% 门禁）→ **FOC-lite 假设被证伪**
- Half-step 反而更差 → stop-gradient 目标有害
- 速度场的问题不是"数值求解不准"，而是"预测方向/幅度系统偏差"

### 根因诊断

```
D50→D20 的 10.7 dB gap 分解：
├── Transport 速度场系统偏差: ~10.7 dB ← 唯一主矛盾
│   ├── DINOv2 语义化导致噪声下结构信息被掩盖
│   ├── hop0 pixel forcing 方向正确但效力不足
│   │   ├── 注入点太浅（backbone 入口，经 28 层 self-attention 被稀释）
│   │   ├── pixel encoder 容量受限（~150K params, 5% cap）
│   │   └── image_aux 权重过低（lambda_max=0.03, 仅占总 loss 2-3%）
│   └── ODE 积分误差可忽略（0.17%）
└── Decoder 误差: ~0.41 dB ← 次要，224 管线 decoder 质量已好
```

---

## 2. PLAN_V4 预期的修正

| 方案 | PLAN_V4 预期 | E1/E2 后修正 | 原因 |
|------|------------|------------|------|
| FOC-lite | P1, +0.3-0.8 dB | **降级 P3 / 暂缓** | 积分误差仅 0.17%, half-step 更差 |
| Post-decoder CNN | P1, +0.3-0.8 dB | **降级 P3** | decoder gap 仅 0.41 dB, 天花板太低 |
| Decoder LoRA | P2, +0.5-1.5 dB | **降级 P3** | 同上 |
| **提高 image_aux 权重** | P1 | **升级 P0** | pixel forcing 被"饿死" |
| **中间层表征对齐 (iREPA)** | 未列 | **新增 P0** | 直接约束 backbone 中间表征质量 |

---

## 3. Scheme C: 提高 Image Auxiliary 权重

**Config**: `pet_flow_first_hop_224_50k_imgaux_boost.yaml`
**Commit**: `faa574b`

### 设计理据

E1 证明 transport 占 96.3% 的 gap，但当前 `image_aux` 的 `lambda_max=0.03` 仅占总 loss 的 ~2-3%。
pixel forcing 机制（`FirstHopPixelEncoder` 注入 D50 像素信息到 backbone 输入）方向正确，
但训练信号不奖励 backbone 利用这些信息——pixel-level 的结构反馈被 latent-level 的 pair/rollout loss 淹没。

### 改动

```yaml
# 仅改 config，不改代码
image_aux:
  lambda_max: 0.12    # 从 0.03 提到 0.12（4×）
  ramp_ratio: 0.15    # 从 0.20 缩短（更快 ramp-up）
```

### 预期

- 如果 D20 PSNR 显著跳变（+0.5 dB 以上）→ pixel forcing 机制本身有效，只是被"饿死"
- 如果无变化 → 注入方式本身需要升级到 Scheme A

### 风险

- 过高权重可能导致 pair/rollout loss 退化
- 监控：`pair_frac`, `roll_frac`, `img_frac` 的 balance

---

## 4. Scheme A: iREPA-Style 空间表征对齐

**Config**: `pet_flow_first_hop_224_50k_irepa_align.yaml`
**Commits**: `11e1ffa` (实现), `cba7dcd` (bugfix)

### 设计理据

当前 pixel forcing 在 backbone **输入层**注入 pixel 信息：
```
z_in = z_D50 + gate × pixel_encoder(x_D50)
v = backbone(z_in, ...)  # 28 层 self-attention 稀释 pixel 信息
```

E1 证明速度场预测偏差是主矛盾（10.7 dB），而 pixel 信息在 28 层 transformer 中被稀释。
受 REPA (Yu et al., ICLR 2025 Oral) 启发，改为在 backbone **中间层**做表征对齐：
直接约束 backbone 的中间 hidden state 与 GT target latent 对齐。

### 为什么用 iREPA 的 Spatial Projector 而不是原版 REPA 的 MLP

| 特性 | REPA (原版) | iREPA (改进版) |
|------|-----------|--------------|
| Projector | MLP (逐 token 独立投影) | **Conv2d** (保留 2D 空间结构) |
| 空间关系 | 丢失 | **保留** (3×3 conv 覆盖相邻 token) |
| 适合场景 | FID 评估的图像生成 | **密集预测** / 像素级精度任务 |

PET 的解剖结构空间连续性是核心信息，spatial projector 更适合。

### 架构

```
backbone block[6] (DiT 第 7 层，共 12+2=14 层)
    ↓ forward hook 提取
hidden: [B, 256, 384]
    ↓ reshape to 2D
    [B, 384, 16, 16]
    ↓ SpatialAlignmentProjector
    Conv2d(384→256, 3×3) → GN → SiLU
    Conv2d(256→256, 3×3) → GN → SiLU
    Conv2d(256→768, 1×1)
    ↓
projected: [B, 768, 16, 16]
    ↓ MSE loss with
z_dst (GT): [B, 768, 16, 16]  (detached, float32)
```

### 关键设计决策

1. **Hook-based extraction**: 不修改 RAE 仓库的 backbone 代码，用 PyTorch forward hook 非侵入提取
2. **Layer index = 6**: backbone 12 层 encoder 的中间层（第 7 层），信息已经经过充分交互但还未被最终输出层坍缩
3. **AMP 安全**: hook 输出强制 `.float()` 避免 half/float 混合
4. **Alignment target = z_dst**: GT 目标 latent，detached 防止梯度穿透到 data pipeline
5. **Warmup schedule**: `warmup_ratio=0.05, ramp_ratio=0.20, lambda_max=0.10`

### 与 Scheme C 的关系

iREPA config **包含** Scheme C 的 image_aux boost（lambda_max=0.12）。
两者是叠加关系，不是替代。

### 参数预算

| 组件 | 新增参数 |
|------|---------|
| SpatialAlignmentProjector | ~400K |
| 总新增 | ~400K (~1.3% of backbone) |

---

## 5. 运行指南

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull origin foc_lite_hop0

# Scheme C（先跑，验证 pixel forcing 是否被"饿死"）
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml

# Scheme A（可并行跑，验证中间层对齐是否带来额外收益）
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_irepa_align.yaml
```

### 评估

```bash
# 训练完成后，用 full-val eval
python eval_first_hop_224_clip3.py \
    --config <对应 config> \
    --checkpoint <best.pt 或 last.pt> \
    --split val --max-slices 0
```

### 评估对照

| 实验 | Config | 对比基线 |
|------|--------|---------|
| chainstable (baseline) | `chainstable.yaml` | — |
| Scheme C | `imgaux_boost.yaml` | chainstable |
| Scheme A | `irepa_align.yaml` | Scheme C (因为 A 包含 C) |

---

## 6. 决策树

```
Scheme C 训练完成
    ├── D20 gain ≥ +0.5 dB vs chainstable
    │   → pixel forcing 被"饿死"确认
    │   → 进一步调 lambda_max（扫 {0.08, 0.12, 0.15, 0.20}）
    │   → 同时看 Scheme A 是否叠加收益
    │
    ├── D20 gain 0.1–0.5 dB
    │   → pixel forcing 有微弱效果但注入方式仍是瓶颈
    │   → 重点看 Scheme A 是否补足
    │
    └── D20 gain < 0.1 dB
        → pixel forcing 机制本身失效（不仅是"饿死"）
        → 需要根本性改变注入方式（multi-layer injection / cross-attention）

Scheme A 训练完成
    ├── Gain vs Scheme C ≥ +0.3 dB
    │   → 中间层表征对齐有效
    │   → 探索 layer_idx 扫描（4, 6, 8, 10）
    │
    └── Gain vs Scheme C < 0.1 dB
        → 对齐目标或位置不对
        → 考虑其他 alignment targets（如 z_D20_gt 的 encoder 特征而非 latent）
```

---

## 7. 文献依据

| 方案 | 核心参考 | 会议/期刊 |
|------|---------|----------|
| Scheme A (iREPA) | REPA: Representation Alignment for Generation (Yu et al.) | ICLR 2025 Oral |
| Scheme A (spatial proj) | iREPA: improved spatial projector variant | 2025 follow-up |
| Scheme C (loss reweight) | E1/E2 实验内部证据 | 本项目诊断实验 |
| Velocity field 表征学习 | VA-VAE / LightningDiT (Yao et al.) | arXiv 2501.01423, 2025 |
| Equivariance in latent | EQ-VAE (Kouzelis et al.) | ICML 2025 |
