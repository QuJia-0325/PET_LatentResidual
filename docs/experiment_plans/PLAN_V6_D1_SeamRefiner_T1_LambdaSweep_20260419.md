# PLAN V6: D1 (Seam Refiner) + T1 (Lambda Sweep)

**Date**: 2026-04-19
**Status**: 代码已实现，待服务器运行
**Branch**: `foc_lite_hop0`
**Context**: 基于 Scheme C full-val 结果 (+0.08 dB transport_avg vs chainstable)

---

## 0. 动机

### 两个正交的目标

E1/E2 + Scheme C/A 实验后，项目有两个**正交**的核心目标：

| 目标 | 核心指标 | 当前状态 | 主要手段 |
|------|---------|---------|---------|
| **消除 patch 伪影** | 视觉评估 / boundary ratio | 14px 网格可见，医生不通过 | **D1: decoder 端 SeamRefiner** |
| **提升 PSNR** | calc_psnr_clip3 | D20=35.57 (ok but can improve) | **T1: lambda_max 扫描** |

E1 证明 decoder gap 仅 0.41 dB（PSNR 影响小），但 patch 伪影在视觉上是**临床卡点**。
这两个目标需要并行推进。

---

## 1. D1: Post-Decoder SeamRefiner

### 问题定位

ViT-MAE decoder 的 `unpatchify` 将 256 个 14×14 像素块硬拼成 224×224 图像。
每个 patch 的像素预测**独立**，无跨边界平滑约束，产生 14px 网格伪影。

E1 数据：GT latent decode ceiling = 46.64 dB（很好），但 predicted latent decode 时
off-manifold 放大效应使 patch 间不一致性更明显。

RAE 仓库已验证：像素空间 post-processing（BA-DRN）效果有限（+0.44 dB），
但那是在 196 管线（有 pad 污染）上。224 管线上的 post-refiner 有更好的起点。

### 架构设计

```
decode_crop(z) = x_decoded [B, 1, 224, 224]
                    ↓
              SeamRefiner
                    ↓
              x_refined = x_decoded + tanh(residual) × max_residual
```

**SeamRefiner** 结构 (~9K params):
- Stem: Conv2d(1→32, 3×3) + GN + SiLU
- Body: 3 × [DepthwiseConv(32, 7×7) + GN + SiLU + PointwiseConv(32→32) + GN + SiLU]
- Tail: Conv2d(32→1, 3×3) — **zero-init**
- Output: `x + tanh(tail(body(stem(x)))) × 0.15`

**关键设计决策**：
- **Kernel size = 7**: 覆盖半个 patch 宽度（14/2=7），跨越相邻 patch 的边界区域
- **Depthwise-separable**: 空间信息通过 DW conv 跨 patch 传播，channel mixing 通过 PW conv
- **Zero-init tail + tanh clamp**: 初始输出 = decode 原图，训练开始时不退化 PSNR
- **max_residual = 0.15**: 在 [-1,1] 归一化范围内限制修正幅度

### 配合的 Loss 改进：Extended Seam Loss

`pet_lr/losses.py` 新增 `extended_seam_loss(pred, gt, patch_size=14, zone_width=3)`：
- **±3 pixel zone**: 边界区域从 ±1 扩展到 ±3，距离衰减权重
- **GT 参照**: 不仅检查 pred 的自一致性，还与 GT 的梯度/曲率对比
- **二阶平滑**: 除一阶梯度连续外，增加曲率匹配约束

### Config

文件：`pet_flow_first_hop_224_50k_seam_refiner.yaml`
- 基于 Scheme C (lambda_max=0.12)
- 新增 `first_hop.seam_refiner.enabled: true`
- SeamRefiner **与 transport 联合训练**（非分阶段）

### 运行

```bash
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_seam_refiner.yaml
```

### 评估

**定量**：
```bash
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_seam_refiner.yaml \
    --checkpoint <best.pt> --split val --max-slices 0
```

**视觉**（最重要）：
对比 refiner 前后的 decode 输出，检查 14px 网格可见度。
建议随机选 10 个 val slices，生成 {GT, baseline_decode, refined_decode} 三列对比图。

### Gate

- **视觉 PASS**: 14px 网格在背景区域不再肉眼可辨
- **PSNR 不退化**: Δtransport_avg ≥ −0.02 dB（zero-init 保证初始不退化）
- **seam_loss 下降**: 对比 refiner vs no-refiner 的 seam_consistency_loss 值

---

## 2. T1: Image Auxiliary Lambda Sweep

### 动机

Scheme C (lambda_max=0.12) 在 chainstable (0.03) 基础上带来了 +0.08 dB。
问题：这个增益是否可以通过继续提权来持续？还是已经到了当前架构的天花板？

### 实验设计

| Config | lambda_max | vs Scheme C (0.12) |
|--------|-----------|-------------------|
| `imgaux_lam0_18.yaml` | 0.18 | 1.5× |
| `imgaux_lam0_25.yaml` | 0.25 | 2.1× |

其他所有参数与 Scheme C 完全一致（run_name 不同）。

### 运行

```bash
# GPU 0
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_lam0_18.yaml

# GPU 1
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_lam0_25.yaml
```

### 评估

```bash
python eval_first_hop_224_clip3.py \
    --config <config> --checkpoint <best.pt> --split val --max-slices 0
```

### Gate

| 结果模式 | 结论 | 后续 |
|---------|------|------|
| 0.18 > 0.12 > 0.03 | transport 优化空间仍大 | 继续推 0.30/0.35 |
| 0.18 ≈ 0.12 | 天花板在 0.12 附近 | 转向 pixel encoder 扩容 |
| 0.18 < 0.12 | 过高权重干扰 pair/rollout | 回退到 0.12 |
| 0.25 退化且 0.18 退化 | 当前架构已到极限 | 结构升级(multi-layer injection 等) |

### 需要监控的指标

- `pair_frac`, `roll_frac`, `img_frac`: loss balance 是否被 image_aux 破坏
- `val_pair_total`: pair loss 是否退化
- `val_rollout_total`: rollout loss 是否退化
- `val_chain_*_mse`: 各 hop 的 chain quality

---

## 3. 实施优先级

```
可并行：
├── D1: SeamRefiner（GPU 0）— 解决临床视觉卡点
├── T1a: lambda=0.18（GPU 1）— PSNR headroom 探索
└── T1b: lambda=0.25（GPU 2）— PSNR headroom 探索

D1 和 T1 完全独立，无依赖关系。
```

---

## 4. 文件变更清单

### 新增/修改代码

| 文件 | 变更 |
|------|------|
| `pet_lr/model_first_hop.py` | 新增 `SeamRefiner` 类；`decode_crop()` 集成 refiner |
| `pet_lr/losses.py` | 新增 `extended_seam_loss()` 函数 |
| `pet_lr/losses_first_hop.py` | 导入 `extended_seam_loss` |

### 新增 Config

| 文件 | 用途 |
|------|------|
| `pet_flow_first_hop_224_50k_seam_refiner.yaml` | D1: Scheme C + SeamRefiner |
| `pet_flow_first_hop_224_50k_imgaux_lam0_18.yaml` | T1: lambda=0.18 |
| `pet_flow_first_hop_224_50k_imgaux_lam0_25.yaml` | T1: lambda=0.25 |

### 向后兼容

- chainstable 等旧 config 不含 `seam_refiner` 节 → `enabled=False` 默认值 → 无影响
- `extended_seam_loss` 是新增函数，不改现有 `seam_consistency_loss`
- SeamRefiner 的 zero-init 保证初始输出 = 原始 decode（无 PSNR 退化风险）
