# Proposer Response — 优先级重定向

**Date**: 2026-04-24  
**状态**: 重新聚焦 transport 机制突破

---

## 1. 对 σ_seed 要求的最终立场

**延后到有实质性突破后再做。**

理由：
1. 系统高度确定性（`val_shuffle=false`, `deterministic=true`）
2. backbone ~200M params 从 checkpoint 确定性加载，不受 seed 影响
3. 随机性仅来自 first-hop ~0.5M params 的初始化，而这些模块已被证明贡献微弱
4. **10-16 dB 的 gap 远大于任何可能的 seed 波动**（即使 σ_seed=0.1 dB，也只是 gap 的 <1%）
5. 4 GPU-day 用于 σ_seed = 4 GPU-day 不用于 transport 突破

**正确流程**：先做出 3+ dB 级别的 transport 改善 → 再用 multi-seed 验证结果稳健性 → 这才有 multi-seed 的实际价值。

---

## 2. 当前最高优先级：Transport 突破

### Oracle Gap 回顾

| Timepoint | Decoder Ceiling | Transport Best | **Gap** |
|-----------|----------------|---------------|---------|
| D20 | 46.636 dB | 35.569 dB | **11.1 dB** |
| NORMAL | 52.634 dB | 36.806 dB | **15.8 dB** |

这 10-16 dB 的 gap 才是应该集中所有精力的地方。

### 已知事实

1. **E1 诊断**（0416）：gap 的 96-99% 来自 velocity prediction 误差，非 decoder 非 ODE
2. **所有 50K 实验**：收敛到 36.12-36.21 dB，无论改什么附件参数
3. **pair_loss 接近零**（v3-200K step 40K+）：velocity 在单步 pair 维度已经很准
4. **rollout_loss 仍在波动**（0.0008-0.0012）：**级联误差累积是主要问题**

### 关键洞察

pair_loss ≈ 0 但 rollout_loss 还大 → **单步精准但多步累积失控**。

这正是 Path A 要诊断的：是 exposure bias（训练看到 GT 输入，推理看到预测输入）还是 velocity capacity（即使给 GT 输入单步也不够精准）。

---

## 3. 执行优先级（重新排序）

| 优先级 | 任务 | GPU-day | 回答的问题 |
|--------|------|---------|----------|
| **P0** | **Path A 诊断**（TF vs RO gap） | 0.5 | exposure bias 还是 velocity capacity？ |
| **P1** | **200K v3 继续跑** | ~4（剩余） | 更长训练能否降低 rollout loss？ |
| **P2** | 基于 Path A 结果设计 transport 改进方案 | 0 | — |
| **P3** | 实施 transport 改进实验 | TBD | 目标 3+ dB 改善 |
| 延后 | σ_seed | 4 | 有突破后验证 |
| 延后 | Path C decoder FT | 1 | 有突破后估计 RAE bound |
| 延后 | Pix2Pix baseline | 2 | 投稿准备阶段 |

### Path A 的 4 种结果及对应行动

| Path A 结果 | 含义 | 下一步 |
|------------|------|--------|
| **EXPOSURE_BIAS** (gap ≥ 0.5 dB) | 训练分布和推理分布不匹配是主因 | 加强 rollout loss / DAgger / curriculum |
| **VELOCITY_CAPACITY** (ceiling_gap ≥ 5 dB) | 即使 GT 输入也不够精准 | 换 loss 函数 / 增加 image-space supervision |
| **MIXED** | 两者都有 | 同时处理 |
| **NEAR_CEILING** | 单步已接近 ceiling | 重新审查 E1（可能矛盾） |

---

## 4. 对 Reviewer 其他建议的处置

| 建议 | 处置 | 理由 |
|------|------|------|
| F2 叙事修正 | ✅ 接受 | 写作改动，零成本 |
| 叙事 reframe | ✅ 接受 | 已确认 |
| NEAR_CEILING 推断过强 | ✅ 接受 | 在解读时注意 |
| Path C 代码改动 | ⏸ 延后 | Path A 先出结果 |
| multi-seed | ⏸ 延后 | 有 3+ dB 突破后做 |
| Pix2Pix baseline | ⏸ 延后 | 投稿准备阶段 |

---

## 5. 运行命令

### Path A（最高优先级，0.5 GPU-day）

```bash
# 在任一空闲 GPU 上：
CUDA_VISIBLE_DEVICES=<ID> python -u scripts/diagnose_tf_rollout_gap.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C
```

### 200K（继续跑，不中断）

已在 GPU-1 上运行，当前 ~42K/200K 步。

---

## 6. 总结

**不要在 0.02 dB 的精度上做文章。把全部精力放在缩小 10-16 dB 的 transport gap 上。**

Path A 是当前最小成本、最高信息价值的下一步——它直接告诉我们 transport 的 10 dB gap 中，多少是因为训练/推理分布不匹配（可以通过 loss 设计修复），多少是 velocity predictor 本身的精度上限（需要更根本的改变）。
