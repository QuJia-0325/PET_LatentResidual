# 0425 实验执行说明 — Self-Forcing 快速验证

## 当前状态

| GPU | 任务 | 状态 |
|-----|------|------|
| GPU-0 | 空闲 | ✅ 可用 |
| GPU-1 | 200K v3 | 进行中 (~58K/200K) |
| GPU-2 | 空闲 | ✅ 可用 |
| GPU-3 | 空闲 | ✅ 可用 |

## 可视化确认的关键发现

从 `review/0425/visual_artifact_assessment_20260425.md` 和 11 张可视化图像：

1. **级联误差累积是核心问题**：D50→NORMAL vs D4→NORMAL 差 5-7 dB
2. **伪影模式是高摄取区纹理斑驳**，不是 patch seam
3. **简单样本已可用**（slice 1500: D50→NORMAL 42.79 dB）
4. **困难样本差距大**（slice 1100: D50→NORMAL 26.60 dB）

## 并行实验矩阵

### 实验 A: Path A 诊断（GPU-0, 4h）

**目的**：量化 exposure bias vs velocity capacity。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_tf_rollout_gap.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C
```

**预期输出**：
- `tf_rollout_gap_val.json`：exposure_gap_dB 和 ceiling_gap_dB
- verdict: EXPOSURE_BIAS / VELOCITY_CAPACITY / MIXED / NEAR_CEILING

### 实验 B: Self-Forcing Pair 50K（GPU-2, ~16h）

**目的**：验证 Self-Forcing pair_loss 能否打破 36.2 dB plateau。

**前提**：需要先实现 Self-Forcing 代码（~25 行 `train_first_hop.py`）和创建 config。

**Config**: `pet_flow_first_hop_224_50k_selfforcing_from_C.yaml`

关键参数：
- `self_forcing_pair.enabled: true`
- `self_forcing_pair.warmup_steps: 5000`（前 5K 步不 SF，让 optimizer state 适应 resume）
- `self_forcing_pair.ramp_steps: 10000`（5K-15K 渐进 GT→pred z_src）
- `rollout.alpha_start/end: 1.0`（rollout 固定纯预测）
- `--resume` 旧 Scheme C best.pt

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_selfforcing_from_C.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

### 实验 C: Rollout-Up 对照 50K（GPU-3, ~16h）

**目的**：消融对照——只提升 rollout 权重（不做 SF），看是否能达到同等效果。

**Config**: `pet_flow_first_hop_224_50k_rollup_from_C.yaml`

关键参数：
- `self_forcing_pair.enabled: false`（pair_loss 仍用 GT）
- `rollout.lambda_start/end: 1.0`（rollout 权重从 0.25 提升到 1.0）
- `rollout.alpha_start/end: 1.0`
- `--resume` 旧 Scheme C best.pt

```bash
CUDA_VISIBLE_DEVICES=3 python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_rollup_from_C.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

## 实施步骤

### Step 1: 实现 Self-Forcing 代码（~25 行）

在 `train_first_hop.py` 的训练循环中 `compute_pair_losses` 调用前插入 self-forcing 逻辑：
- 读取 config `training.self_forcing_pair`
- 根据 step 计算 alpha_sf（渐进 ramp）
- 在 no_grad 下做 4-hop pre-rollout 生成预测 chain
- 按 hop_idx 选对应的 z_pred 替换 `main_batch["z_src"]`
- `compute_pair_losses` 代码不需要修改

### Step 2: 创建 2 个 Config

基于 `imgaux_boost.yaml` 复制，修改：
1. `selfforcing_from_C.yaml`：加 `self_forcing_pair` 段 + rollout alpha 固定 1.0
2. `rollup_from_C.yaml`：不加 SF + rollout lambda 提升到 1.0

### Step 3: 启动

Path A 立即可启动（无代码改动）。
实验 B/C 在 Step 1-2 完成后启动（预计 1-2h 实现）。

## 成功判据

| 对比 | 阈值 | 判断 |
|------|------|------|
| SF-pair > Scheme C (36.206) | Δ > 0.3 dB | SF 有效 |
| SF-pair > Rollout-Up | Δ > 0.3 dB | "pair 主梯度"假设正确 |
| SF-pair ≈ Rollout-Up | Δ < 0.1 dB | 只需加 rollout 权重 |
| 两者都 ≈ Scheme C | Δ < 0.1 dB | exposure bias 不是主因 |

## Day 3 后的决策

| 结果 | 下一步 |
|------|--------|
| SF >> RU >> baseline | SF 方法成立 → resume 200K best 跑 SF 100K |
| SF ≈ RU >> baseline | rollout 权重更重要 → 简化方法 |
| SF ≈ RU ≈ baseline | exposure bias 不是主因 → 转 Idea 2 (stochastic hop image loss) |
| SF 训练不稳定 | 降低 alpha_sf_end 或增加 ramp_steps |
