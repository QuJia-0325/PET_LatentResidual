# Transport 突破实验计划 v5 — 2026-04-26

**Supersedes**: v4（SF-pair 方向已暂停）  
**核心方向转换**: SF-pair → Rollout-Heavy（加大已有 rollout_loss 权重）  
**依据**: v4 SF pilot 分析 + reviewer critique + 代码级 rollout/SF 对比（§14-17）

---

## 0. 决策依据摘要

| 实验 | 结论 | 对 V5 的含义 |
|------|------|-------------|
| Path A 多 ckpt (v3) | ExpoGap=4.80 dB，单调上升 | exposure bias 是核心问题 ✅ |
| SF-pair v3 | schedule bug 导致无效 | 修复后重跑 |
| SF-pair v4 pilot | α≤0.10 微弱改善(5%)，α≥0.15 相变崩溃 | SF-pair 方向暂停 |
| rollout alpha 机制 (§14) | GT→混合→pred 三阶段 + 每跳 GT loss | 设计正确，权重不足 |
| 梯度占比 (§16) | roll_frac ~10%, img_frac ~85% | rollout 信号被淹没 |

**V5 唯一假设**：加大 rollout_loss 权重，让正确的机制拿到足够的梯度影响力。

---

## 1. 实验设计

### 1.1 V5-1A：Rollout-Heavy 保守版

从 200K v3 best.pt（step=86800）resume，50K 新增步。

| 参数 | baseline (200K v3) | V5-1A | 变化 | 理由 |
|------|-------------------|-------|------|------|
| `rollout.lambda` | 0.25 | **1.0** | 4× | 将 roll_frac 从 ~10% 提升到 ~30% |
| `image_aux.lambda` | 0.12 | **0.08** | ÷1.5 | 保守降低，保护 hop0 像素质量 |
| `step_weights` | [1.0, 1.1, 1.2, 1.3] | **[0.8, 1.0, 1.5, 2.5]** | 温和末端加重 | 见 §1.2 |
| `self_forcing_pair` | — | **disabled** | — | SF 方向暂停 |
| 其余参数 | — | **不变** | — | 单轴控制变量 |

### 1.2 step_weights 设计理由

```
                ExpoGap    CeilGap    val_chain_MSE    step_weight
D50→D20         0.00 dB    11.07 dB   0.000243 (最大)  0.8
D20→D10         3.38 dB    9.42 dB    0.000217         1.0
D10→D4          4.71 dB    9.62 dB    0.000189         1.5
D4→NORMAL       6.32 dB    9.51 dB    0.000175 (最小)  2.5
```

**D50→D20 保持 0.8 而非更低的理由**：
- D50→D20 的 val_chain_MSE 是四跳中最大的（0.000243），且 CeilGap=11.07 也最大
- 该跳的质量问题**不是 exposure bias**（ExpoGap=0），而是 velocity prediction 精度本身
- 过度降低 hop0 权重可能导致 D50→D20 质量退化（噪声增加 / 纹理丢失）
- 0.8 是保护性权重：不让 hop0 在 rollout 中被忽略，同时留更多梯度给 hop1-3

**末端温和加重的理由**：
- 选 [0.8, 1.0, 1.5, 2.5] 而非 [0.5, 1.0, 2.0, 4.0]——更保守
- hop3 (D4→NORMAL) 权重是 hop0 的 3.1× 而非 8×，降低末端 hop 过拟合风险
- 如果 1A 趋势正向，1B 可以进一步加重末端

### 1.3 V5-1B：激进版（仅当 1A 通过后）

| 参数 | 1A | 1B |
|------|----|----|
| `rollout.lambda` | 1.0 | **2.0** |
| `image_aux.lambda` | 0.08 | **0.04** |
| `step_weights` | [0.8, 1.0, 1.5, 2.5] | **[0.5, 1.0, 2.0, 4.0]** |

1A → 1B 升级条件（所有"步"均指 resume 后新增步数）：
- +25K 步时 `val_select` 改善 ≥ 3%
- +25K 步时 `val_chain_normal_mse` 改善 ≥ 2%
- hop0 PSNR 不退化 > 0.3 dB
- `pair_frac` 仍 ≥ 2%
- **val_pair_total 始终 < 0.00005**（无相变信号）

---

## 2. Go/No-Go 监控

### 2.1 相变早期警报（最高优先）

| 检查点 | val_pair_total 阈值 | 动作 |
|--------|---------------------|------|
| +5K | > 0.00005 | ⛔ **立即停**，检查参数是否被推出 GT 盆地 |
| +10K | > 0.00003 | ⚠️ 关注，+15K 再看 |
| +25K | > 0.00002 | ⛔ 止损 |

**理由**：V4 SF pilot 中 val_pair_total 从 0→0.000196 是相变的最早信号（比 val_select 提前 1-2 个 val 点）。rollout_loss 也用模型自预测输入，理论上有同样风险（虽然较低）。

### 2.2 性能监控

| 检查点 | val_select 阈值 | val_chain_normal_mse | hop0 PSNR |
|--------|----------------|---------------------|-----------|
| +10K | > baseline × 1.05 → 止损 | > baseline × 1.10 → 止损 | < baseline − 0.5 dB → 止损 |
| +25K | 无改善（≥ baseline）→ 止损 | 无改善 → 止损 | < baseline − 0.3 dB → 止损 |
| +50K | full-val eval + Path A redo | — | — |

### 2.3 梯度占比监控

训练日志中每 50 步检查一次（从 stdout `[train]` 行读取）：
- `roll_frac` 目标：**25-40%**（从当前 ~10% 提升）
- `img_frac` 目标：**40-60%**（从当前 ~85% 下降）
- `pair_frac` 底线：**≥ 2%**（pair 监督不能被压死）

如果 `roll_frac` 未达到 20%，说明 λ_roll=1.0 仍不够→ 早期可以考虑直接升 1B。

---

## 3. 成功标准（挂钩医生需求）

| 优先级 | 指标 | 阈值 | 含义 |
|--------|------|------|------|
| 1 | `val_chain_normal_mse` | 改善 ≥ 5% vs baseline | NORMAL 档是医生看的输出 |
| 2 | Path A exposure_gap (D4→NORMAL) | ≤ 5.69 dB（降 ≥ 10%，当前 6.32） | 末端 hop exposure bias 缓解 |
| 3 | `val_select_score` | < baseline | chain 整体不拆东墙补西墙 |
| 4 | `val_chain_d20_mse` | 不恶化 > 5% | 保护 D50→D20 质量 |

---

## 4. Claims Matrix

| 场景 | 允许的 claim |
|------|-------------|
| 1A > baseline 且 exposure_gap ↓ | 加权 rollout 策略有效对抗 cascade exposure bias |
| 1A > baseline 但 exposure_gap 不变 | rollout 权重提升改善了 chain 精度（更好的 velocity 拟合），但非通过减少 exposure bias |
| 1A ≈ baseline（±3%） | 梯度信号不足不是瓶颈；需要结构性改变 |
| 1A < baseline | rollout 过强压制了 pair/image 信号，需回退或寻找平衡点 |
| 1A 触发 val_pair_total 相变 | 加大 rollout 也导致参数盆地切换——exposure bias 可能需要架构级解决方案 |

---

## 5. 时间线

| Day | GPU-0 | GPU-1 | 产出 |
|-----|-------|-------|------|
| 0 | 准备 config + launch script | 200K v3 继续 | config + script ready |
| 0 PM | **V5-1A 启动**（from step 86800） | 200K v3 继续 | — |
| 1 AM | +5K 相变检查 | 200K v3 继续 | val_pair_total 读数 |
| 1 PM | +10K 性能检查 | 200K v3 继续 | go/no-go 第一关 |
| 2 | +25K 检查 | 200K v3 可能完成 | go/no-go 第二关 |
| 3 | +50K 完成 + full-val eval | — | 最终结果 |
| 3-4 | Path A redo on 1A best | — | exposure_gap 对比 |
| 4 | 决策：1B 或换方向 | — | — |

---

## 6. 代码改动

| 文件 | 改动 | 状态 |
|------|------|------|
| config: `pet_flow_first_hop_224_v5_rollout_heavy_1a.yaml` | 新 config | ❌ 待创建 |
| `scripts/launch_v5_rollout_heavy.sh` | 自动读 ckpt step + 设 max_steps + 启动 | ❌ 待创建 |
| `train_first_hop.py` | **0 行改动** | ✅ 无需修改 |

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| rollout 加权导致相变（val_pair_total 上涨） | +5K 步即检查，早于 SF pilot 的发现速度 |
| hop0 像素退化（image_aux 降低） | 仅降 ÷1.5（保守），hop0 PSNR 做独立监控 |
| pair_loss 被压死（roll_frac 过高） | pair_frac ≥ 2% 底线，低于即止损 |
| step_weights 末端加重导致 backbone 在末端 hop 过拟合 | 温和版 [0.8, 1.0, 1.5, 2.5]，d20 MSE 独立监控 |
| 86800 快照不是最强起点 | 1A 是趋势验证，不追求绝对最优；1B 可用 200K final best |

---

## 8. 从 V4 吸取的教训（硬编码进 V5 流程）

1. **不做"单点跳大步"**：1A 只改 rollout 4× + image ÷1.5，不同时做多个大变化
2. **val_pair_total 是相变早期警报**：独立监控，阈值 0.00005
3. **rolling val 仅用于趋势**：最终结论必须 full-val eval
4. **max_steps 必须是 resume_step + 新增步数**：launch script 自动算
5. **SF 相关功能保持 disabled**：不引入新变量
