# 0423 实验决策与 Transport 分析进展

**日期**: 2026-04-23  
**分支**: `foc_lite_hop0`

---

## 1. Backbone Checkpoint 调查结论

经 Codex 服务器检索确认：

| 项目 | 值 |
|------|-----|
| checkpoint | `pet_flow_224_small_rollout_hopaware_smoke/best.pt` |
| **训练步数** | **2000 步**（smoke 测试级别） |
| target_normalize | false |
| pos_embed 形状 | `[1, 256, 384]`（16×16 tokens, 224px ✓） |
| EMA | 有 (decay=0.9999) |
| alpha_end | 1.0 |
| best_metric | rollout_total = 0.0185 |
| batch × grad_accum | 4 × 8 = 32 |

**结论**：backbone 从 2000 步 smoke checkpoint 开始，PET_LatentResidual 的 50K 步训练实际上是在大幅学习 velocity prediction（非微调成熟模型）。

---

## 2. 已完成实验结果对比

| 实验 | pixel_forcing | backbone_lr | 0422 Fix | EMA | Best Score | Best Step |
|------|--------------|-------------|---------|-----|-----------|-----------|
| 旧 Scheme C | ON | 0.4× (3.2e-5) | ❌ | ❌ | 0.000532 | ~46400 |
| **N1 (ablation)** | **OFF** | 0.4× (3.2e-5) | ✅ (resume旧ckpt) | ❌ | **0.000525** | 46400 |
| v2 (进行中) | ON | 0.4× (3.2e-5) | ✅ | ✅ | 0.000546 | 35200 |
| N2 standalone | ON (frozen) | 0.4× (3.2e-5) | ✅ | ❌ | 0.000514 | 17200 |

### 关键发现

1. **N1 (pixel forcing OFF) = 0.000525 略优于旧 Scheme C (0.000532)**
   - pixel forcing 路径整体消融后性能不降反升
   - 暗示 pixel forcing 在当前设置下无正面贡献（或在噪声范围内）

2. **v2 还在训练中**（38K/50K），best 还在下降，需等完成后公平对比

3. **N2 的 val_select_score=0.000514 看起来最优，但注意**：
   - N2 transport 是冻结的（只训 refiner），score 来自 Scheme C 的冻结 transport
   - refined vs raw MSE 差异极小（1-3%），refiner 贡献微弱

---

## 3. 新实验：Backbone LR = 1.0

### 实验设计

基于 N1（当前 best）做**单变量改动**：

| 参数 | N1 (baseline) | 新实验 |
|------|--------------|--------|
| backbone_lr_mult | **0.4** (3.2e-5) | **1.0** (8.0e-5) |
| pixel_forcing | disabled | disabled |
| first_hop_weight_decay | 0.0 | 0.0 |
| lambda_hop_init | 0.10 | 0.10 |
| EMA | 无 | 无 |
| 其余参数 | — | 完全相同 |

### 动机

- backbone 占模型总参数的 ~97%（~30M vs first-hop ~0.5M）
- 从 2000 步 smoke checkpoint 出发需要大量学习
- 当前 0.4× LR 使 backbone 学习速度只有 first-hop 的 1/9
- 所有先前实验都在此 LR 下运行——backbone 可能一直在学不够快

### Config

`configs/pet_flow/pet_flow_first_hop_224_50k_backbone_lr1.yaml`

### 运行命令

```bash
# GPU-1: Backbone LR=1.0 (50K steps)
CUDA_VISIBLE_DEVICES=1 TQDM_DISABLE=1 \
python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_backbone_lr1.yaml
```

### 判据

- **score < 0.000500**（比 N1 改善 > 5%）→ backbone LR 是瓶颈
- **score ≈ 0.000525**（与 N1 相近）→ backbone LR 不是瓶颈，问题在更深层
- **score > 0.000550 或训练不稳定** → LR 过高，需回调

### 监控重点

- `lambda_hop[0:4]` 增长趋势（LR 提高后 hop residual 应更快激活）
- 训练 loss 波动性（LR 提高可能导致 pair_loss 波动加大）
- `val_select_score` 最早 converge 的 step（如果比 N1 更早说明学习加速）

---

## 4. GPU 状态

| GPU | 任务 | 状态 |
|-----|------|------|
| GPU-1 | **backbone_lr1 (新)** | 待启动 |
| GPU-3 | v2 (schemec_v2) | 进行中 (~38K/50K) |

---

## 5. 后续实验路线（取决于 backbone_lr1 结果）

| 如果结果 | 下一步 |
|---------|--------|
| backbone_lr1 显著改善 (< 0.000500) | 尝试 backbone_lr_mult=2.0 或 200K 步 |
| backbone_lr1 无变化 (≈ 0.000525) | 换方向：增大 backbone 容量（DiT-B）或增加 grad_accum |
| backbone_lr1 退化或不稳定 | 回调 LR，尝试 backbone_lr_mult=0.7 |

---

## 6. 架构理解勘误

之前分析中的错误已纠正：
- ❌ ~~r203 config 的 alpha_end=0.35 与 PET_LR 不匹配~~ → r203 是 192px 旧版本，与当前 224 系统无关
- ❌ ~~input_size 14 vs 16 不匹配~~ → smoke checkpoint pos_embed 确认为 [1,256,384]（16×16），匹配 224px
- ❌ ~~backbone 从 pre-trained 模型微调~~ → backbone 从 2000 步 smoke 开始，50K 步训练是主体学习
- ✅ RAE 只提供 encoder-decoder + 模型架构定义，PET_LatentResidual 有自己完整的 transport 训练实现
