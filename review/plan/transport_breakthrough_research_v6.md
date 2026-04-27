# Transport 突破实验计划 V6 — 2026-04-27

**Supersedes**: V5（rollout-heavy resume 实验，待 full-val 结果确认）
**核心方向**: Transport-First 从 scratch 训练（恢复 transport 为梯度主力）
**关键改进**: 延长 alpha schedule + 降低 image_aux + rollout 渐进加权

---

## 0. V6 的出发点

### 从 V3-V5 学到的关键事实

| 事实 | 数据来源 |
|------|---------|
| D50→D20 是整个 chain 中最大的质量瓶颈（CeilGap=11.07 dB） | Path A |
| image_aux 占梯度 80-90%，transport（pair+rollout）仅 10-20% | V3/V5 训练日志 |
| 从 V3 best resume 后改 loss 权重，~6K 步就出现指标震荡 | V4 SF + V5 rollout-heavy |
| V3 的 alpha schedule 在 step 5K-20K 就完成 GT→Pred 过渡（太早） | V3 config |
| V3 200K best 在 step 86.8K 就达到了（后续未刷新），说明 plateau 很早 | V3 训练日志 |
| `velocity_rebalance` 在当前公式下恒为 1.0（endpoint ≡ velocity） | V5 训练日志 |

### V6 的设计哲学

1. **Transport 是主力，image_aux 是辅助**——梯度比例应匹配设计意图
2. **延长 GT 训练期**——模型先把 velocity 学准再面对 Pred chain
3. **从 scratch 训**——避免 resume 的梯度突变/LR 不匹配问题
4. **渐进式改变**——alpha 和 rollout λ 同步 ramp，不做单点跳大步

---

## 1. 实验设计

### 1.1 V6-main: Transport-First 200K

从零训练 200K 步。**核心改动 vs V3 200K baseline**：

| 参数 | V3 200K | V6-main | 变化 | 理由 |
|------|---------|---------|------|------|
| **alpha schedule** | 0-5K GT, 5K-20K ramp, 20K+ Pred | **0-50K GT, 50K-150K ramp, 150K-200K Pred** | 大幅延长 | 模型先学准 velocity 再面对 Pred |
| **rollout λ** | 0.02→0.25 (ramp 同 alpha) | **0.02→0.50** (ramp 50K-150K) | 最终 2× | 加强 rollout 但不过激 |
| **image_aux λ** | 0.005→0.12 (ramp 0-20K) | **0.005→0.03** (ramp 0-50K) | 最终 ÷4 | 降低到真正的辅助角色 |
| step_weights | [1.30, 1.20, 1.10, 1.00] | **[1.00, 1.00, 1.20, 1.50]** | 温和尾重 | 轻微加重末端，不激进反转 |
| pair_loss_weights | [1.20, 1.10, 1.05, 1.00] | **不变** | — | 保持 V3 头重（D50→D20 最需要 pair 监督） |
| lr | 8e-5 | **不变** | — | |
| warmup_ratio | 0.15 | **不变** | — | |
| ema | enabled, 0.9999 | **不变** | — | |

### 1.2 Alpha + Rollout λ 同步 ramp

```
Step 0-50K:     alpha=0 (纯 GT), λ_roll ramp 0.02→0.10
                → 模型在 GT 输入上专心学 velocity
                → rollout 权重低但在场，开始建立 chain awareness

Step 50K-150K:  alpha 0→1 (GT→Pred), λ_roll ramp 0.10→0.50
                → alpha 上升 = Pred 输入增多 = 链式误差增加
                → λ_roll 同步上升 = 模型对链式误差的关注度增加
                → 两者同步：避免"关注高但输入还是 GT"或"输入已是 Pred 但关注低"

Step 150K-200K: alpha=1.0 (纯 Pred), λ_roll=0.50 (最终值)
                → 训练分布 = 推理分布
                → rollout 在最终权重下精修
```

### 1.3 Image_aux 定位为辅助

```
Step 0-50K:     λ_img ramp 0.005→0.03
                → 前期逐步引入像素监督

Step 50K-200K:  λ_img=0.03（保持不变）
                → 辅助角色，不再主导梯度
```

预期梯度占比：

| 阶段 | pair_frac | roll_frac | img_frac |
|------|-----------|-----------|----------|
| 0-50K (GT, λ_roll 低) | ~60-70% | ~5-10% | ~25-30% |
| 50K-150K (ramp) | ~40-50% | ~20-30% | ~15-20% |
| 150K-200K (Pred, λ_roll 高) | ~30-40% | ~40-50% | ~10-15% |

**transport（pair+rollout）始终是梯度主力，image_aux 始终是辅助**。

### 1.4 Config 参数规格

```yaml
# configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml

training:
  max_steps: 200000

  rollout:
    enabled: true
    warmup_ratio: 0.25      # alpha warmup: 0-50K (25%)
    ramp_ratio: 0.50        # alpha ramp: 50K-150K (50%)
    alpha_start: 0.0
    alpha_end: 1.00
    lambda_start: 0.02      # rollout λ 从 0.02 开始
    lambda_end: 0.50        # 最终 0.50（V3 的 2×）
    step_weights:
      - 1.00
      - 1.00
      - 1.20
      - 1.50

  image_aux:
    enabled: true
    warmup_ratio: 0.0
    ramp_ratio: 0.25        # image ramp: 0-50K
    lambda_start: 0.005
    lambda_max: 0.03        # 最终 0.03（V3 的 ÷4）

  self_forcing_pair:
    enabled: false

  # 其余参数与 V3 200K 完全一致
```

---

## 2. 与 V3 200K 的控制变量分析

| 参数 | V3 200K | V6-main | 是否有意改动 |
|------|---------|---------|------------|
| alpha warmup_ratio | 0.10 (5K) | **0.25** (50K) | ✅ 核心改动 1 |
| alpha ramp_ratio | 0.30 (15K) | **0.50** (100K) | ✅ 核心改动 2 |
| rollout lambda_end | 0.25 | **0.50** | ✅ 核心改动 3 |
| image_aux lambda_max | 0.12 | **0.03** | ✅ 核心改动 4 |
| rollout step_weights | [1.30,1.20,1.10,1.00] | **[1.00,1.00,1.20,1.50]** | ✅ 核心改动 5 |
| pair_loss_weights | [1.20,1.10,1.05,1.00] | 不变 | — |
| best_metric d20 weight | 0.50 | 不变 | — |
| lr | 8e-5 | 不变 | — |
| warmup_ratio (LR) | 0.15 | 不变 | — |
| ema | enabled, 0.9999 | 不变 | — |
| max_steps | 200000 | 不变 | — |
| batch_size | 8 | 不变 | — |

**5 个有意改动，其余严格一致。**

---

## 3. 监控与 Go/No-Go

### 阶段 I (0-50K): GT 训练期

| 检查点 | 指标 | 期望 |
|--------|------|------|
| 10K | pair_loss, roll_frac | pair 应为主导（60%+），rollout 在场但低（~5%） |
| 25K | val_chain_d20_mse | 应持续下降（GT 训练期 velocity 精度提升） |
| 50K | full-val eval | 与 V3 @50K 对比——GT 期更长是否让 D50→D20 更准 |

### 阶段 II (50K-150K): Alpha ramp

| 检查点 | 指标 | 期望 |
|--------|------|------|
| 75K | roll_frac | 应开始上升（15-20%） |
| 100K | val_pair_total | < 0.00005（无相变信号） |
| 100K | val_chain_normal_mse | 应开始改善（rollout 开始对抗 exposure bias） |
| 150K | full-val eval | 关键里程碑：alpha 达到 1.0 前的 checkpoint |

### 阶段 III (150K-200K): 纯 Pred 精修

| 检查点 | 指标 | 期望 |
|--------|------|------|
| 175K | roll_frac | 应在 40-50%（transport 主导） |
| 175K | img_frac | 应在 10-15%（辅助角色） |
| 200K | full-val eval + Path A redo | 最终结果 |

---

## 4. 成功标准

| 优先级 | 指标 | 阈值 | V3 baseline |
|--------|------|------|------------|
| 1 | full-val NORMAL PSNR | **≥ 37.5 dB** | 36.87 dB |
| 2 | full-val D20 PSNR | **≥ 36.0 dB** | 35.54 dB |
| 3 | Path A ExpoGap (D4→NORMAL) | **≤ 5.0 dB** | ~6.32 dB (Scheme C) |
| 4 | roll_frac @ 150K+ | **≥ 35%** | ~10% |
| 5 | img_frac @ 150K+ | **≤ 20%** | ~85% |

---

## 5. 失败回退

| 失败模式 | 时间 | 动作 |
|---------|------|------|
| 50K 时 val_chain_d20_mse 比 V3 @50K 差 | 50K | 检查 image_aux 是否太低（试 λ=0.06） |
| 100K 时 val_pair_total 相变 | 100K | 降 rollout λ_end 到 0.25 |
| 150K 时 NORMAL PSNR < 36.5 dB | 150K | 延长 alpha ramp 到 175K |
| 200K 时 NORMAL PSNR < V3 baseline | 200K | V6 失败；考虑 loss 归一化方案 |

---

## 6. 时间线

| Day | 产出 |
|-----|------|
| 0 | 创建 V6 config + launch script，启动训练 |
| 4 | 50K checkpoint，full-val eval #1 |
| 8 | 100K checkpoint，检查 ramp 状态 |
| 10 | 150K checkpoint，full-val eval #2 |
| 12-13 | 200K 完成，full-val eval #3 + Path A redo |

总计 ~12-13 天（200K 步 @ ~2s/step = ~111h ≈ 4.6 天 × 单卡）。如果用 2 卡可以缩短但不能并行同一实验。

---

## 7. 与之前实验的关系

| 实验 | 学到了什么 | V6 如何应用 |
|------|-----------|-----------|
| V3 200K | img_frac=85% 导致 transport 被压制 | V6 降 image_aux 到 0.03 |
| V4 SF | 改 pair 输入分布会触发 GT 退化 | V6 不改 pair 输入，只改 rollout 权重 |
| V5 rollout-heavy | resume 后突变权重不稳定 | V6 从 scratch 训，渐进 ramp |
| Path A | ExpoGap 在 alpha=1.0 后仍上升 | V6 延长 GT 期到 50K，让 velocity 先学准 |
| Codex counter-review | rolling-val 不可信 | V6 在 50K/150K/200K 做 full-val |

---

## 8. 代码改动

| 文件 | 改动 | 状态 |
|------|------|------|
| `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 config | ❌ 待创建 |
| `scripts/launch_v6_transport_first.sh` | 启动脚本 | ❌ 待创建 |
| `train_first_hop.py` | **0 行改动** | ✅ 无需修改 |

**V6 是纯 config 实验，不需要任何代码修改。**

---

## 9. 开放问题（V6 之后考虑）

| 方向 | 描述 | 条件 |
|------|------|------|
| Loss 归一化 | 让 λ 直接控制梯度比例 | V6 证明 transport-first 有效后，作为工程优化 |
| Consistency Loss | TF vs Rollout gap 作为正则化 | V6 的 ExpoGap 仍 > 4 dB |
| 多步 ODE | D50→D20 用 2-3 步 ODE 代替 1 步 | D50→D20 CeilGap 仍 > 9 dB |
| 更大 backbone | DiT-B 或 DiT-L | V6 的 backbone 容量成为瓶颈时 |
