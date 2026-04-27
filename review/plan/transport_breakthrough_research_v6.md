# Transport 突破实验计划 v6 — 2026-04-27

**Supersedes**: v5（rollout-heavy resume 实验，rolling-val 结果待 full-val 确认）
**核心发现**: V3 200K 训练数据证明 **rollout 梯度占比高（32%）时模型达到 best（step 86800），之后 image_aux 重新主导（88-93%），模型不再改善**
**V6 核心方向**: Transport-First 从 scratch 训练，通过 loss 归一化让 λ 直接控制梯度比例

---

## 0. V3 200K 训练的关键教训

```
step  1000: pair=54%  roll= 1%  img=46%   (alpha=0, 早期)
step 86800: pair=13%  roll=32%  img=55%   ← BEST (rollout 影响力最大)
step150000: pair= 2%  roll= 5%  img=93%   ← 退化 (image_aux 重新淹没 rollout)
```

| 观察 | 含义 |
|------|------|
| Best 出现在 roll_frac=32% 的窗口 | rollout 梯度影响力与模型质量正相关 |
| 86800 之后 rollout_loss 降低但 image_aux 没降 | 梯度比例自然偏移，不是 config 变了 |
| 86800 后训了 86K 步无改善 | image_aux 主导的训练已经 plateau |

**V6 核心假设**：如果能**持续保持** rollout 梯度在 30-50%（而非只在 alpha ramp 完成的短暂窗口），模型可以持续改善而非 plateau。

---

## 1. V6 设计：Loss 归一化 + Transport-First

### 1.1 Loss 归一化机制（~15 行代码）

当前问题：λ 的语义被 loss 数值量级扭曲。设 `λ_roll=0.25, λ_img=0.12`，但实际 roll_frac=5% img_frac=93%。

**修复**：对每个 loss 项做 running mean 归一化，让 λ 直接代表梯度比例：

```python
# 在 train loop 中维护 EMA running mean
ema_pair = EMAScalar(decay=0.99, init=1e-4)
ema_roll = EMAScalar(decay=0.99, init=1e-3)
ema_img  = EMAScalar(decay=0.99, init=1e-2)

# 每步更新
ema_pair.update(pair_loss.detach())
ema_roll.update(rollout_loss.detach())
ema_img.update(img_loss.detach())

# 归一化 loss
pair_norm = pair_loss / ema_pair.value.clamp_min(1e-8)
roll_norm = rollout_loss / ema_roll.value.clamp_min(1e-8)
img_norm  = img_loss / ema_img.value.clamp_min(1e-8)

# 此时 λ 直接控制比例
total = λ_pair * pair_norm + λ_roll * roll_norm + λ_img * img_norm
# λ_pair=0.3 → pair 占 ~30%, λ_roll=0.5 → rollout 占 ~50%, λ_img=0.2 → image 占 ~20%
```

**好处**：
- λ 的设置直观（想要 50% rollout 就设 λ_roll=0.5）
- 训练过程中梯度比例自动稳定（不会因为某个 loss 降了而偏移）
- 向后兼容（加 `loss_normalization: false` 时行为不变）

### 1.2 V6 Config（从 scratch 训 200K）

```yaml
training:
  max_steps: 200000
  
  # === Loss 归一化（V6 新增）===
  loss_normalization:
    enabled: true
    ema_decay: 0.99
    
  # === 梯度比例意图 ===
  # transport (pair + rollout) = 80%, image_aux = 20%
  # pair: 单跳 velocity 精度
  # rollout: chain 级联一致性（对抗 exposure bias）
  # image_aux: hop0 像素质量补充
  
  rollout:
    enabled: true
    lambda_start: 0.0         # alpha=0 时 rollout 无意义
    lambda_end: 0.50          # 目标: rollout 占 50% 梯度
    warmup_ratio: 0.25        # 0-50K: lambda=0
    ramp_ratio: 0.50          # 50K-150K: lambda 0→0.50
    alpha_start: 0.0
    alpha_end: 1.0
    # alpha schedule 与你的提案对齐:
    # 0-50K: alpha=0 (纯 GT)
    # 50K-150K: alpha 0→1 (GT→Pred)
    # 150K-200K: alpha=1 (纯 Pred)
    step_weights: [0.8, 1.0, 1.5, 2.5]   # 末端温和加重
    
  image_aux:
    enabled: true
    lambda_start: 0.20        # 前期 image 占 20%
    lambda_max: 0.10          # 后期降到 10%
    warmup_ratio: 0.0
    ramp_ratio: 0.25          # 前 50K 步 lambda 从 0.20 降到 0.10

  # pair_loss 权重 = 1.0 - rollout - image_aux 的剩余部分
  # 通过归一化自动平衡
```

### 1.3 训练三阶段（与你的 alpha schedule 提案对齐）

| 阶段 | Step 范围 | alpha | λ_roll | λ_img | 梯度预期 |
|------|----------|-------|--------|-------|---------|
| **I. 精度** | 0-50K | 0（纯 GT） | 0 | 0.20 | pair ~80%, img ~20%, roll ~0% |
| **II. 过渡** | 50K-150K | 0→1 | 0→0.50 | 0.20→0.10 | pair 30%→20%, roll 0%→50%, img 20%→10% |
| **III. 精修** | 150K-200K | 1.0 | 0.50 | 0.10 | pair ~20%, roll ~50%, img ~10% |

**阶段 I**：纯 GT 输入，模型专心学 velocity 精度。image_aux 20% 保护 hop0 像素质量。
**阶段 II**：alpha 和 λ_roll 同步上升——输入逐步从 GT 变为 Pred，同时 rollout 梯度增强。
**阶段 III**：纯 Pred chain + rollout 主导（50%），和推理一致。**这个状态会持续 50K 步**，不像 V3 只有短暂窗口。

### 1.4 为什么这样设计能避免 V3 的 plateau

V3 的问题：
- rollout 在 step 20K 就到 alpha=1.0 + lambda=0.25
- 但此时 rollout_loss 绝对值还大（模型不够准），roll_frac 暂时高（32%）
- 模型变好后 rollout_loss 降低 → roll_frac 自然衰减到 5% → image_aux 重新主导 → plateau

V6 的解决：
- **Loss 归一化**让 roll_frac **恒定在 50%**，不随 loss 降低而衰减
- **alpha ramp 延长到 150K**——模型有更长时间在 GT→Pred 过渡中学习
- **λ_roll 从 0 渐进到 0.50**——不是突然加大，避免梯度突变

---

## 2. 代码改动清单

| 文件 | 改动 | 行数 | 优先级 |
|------|------|------|--------|
| `train_first_hop.py` | EMAScalar 类 + loss 归一化 | ~25 行 | **P0** |
| `train_first_hop.py` | `loss_normalization` config 解析 | ~5 行 | P0 |
| config: `pet_flow_first_hop_224_v6_transport_first.yaml` | V6 config | 新文件 | P0 |
| `scripts/launch_v6_transport_first.sh` | launch script | 新文件 | P1 |

### 2.1 EMAScalar 实现

```python
class EMAScalar:
    """Exponential moving average for scalar loss normalization."""
    def __init__(self, decay: float = 0.99, init: float = 1.0):
        self.decay = decay
        self.value = init
        self._initialized = False
    
    def update(self, x: float) -> None:
        x = float(x)
        if not self._initialized:
            self.value = x
            self._initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
```

---

## 3. Go/No-Go 监控

| 检查点 | 条件 | 动作 |
|--------|------|------|
| +10K | roll_frac 是否 ≈ 0%（应为 0，alpha=0 阶段） | 确认归一化生效 |
| +50K | pair_frac 是否 ≈ 80%（α=0，纯 GT 阶段） | 确认 pair 主导 |
| +75K | roll_frac 开始上升（alpha ramp 中） | 确认 rollout 渐进 |
| +100K | roll_frac ≈ 25-35%（ramp 中） | 正常 |
| +150K | roll_frac ≈ 50%，img_frac ≈ 10% | **归一化核心验证点** |
| +200K | val_chain_normal_mse vs V3 best（0.000165） | 最终判定 |

---

## 4. 成功标准

| 指标 | V3 200K best | V6 目标 |
|------|-------------|---------|
| val_chain_normal_mse | 0.000165 | < 0.000150（改善 ≥ 9%） |
| full-val NORMAL PSNR | 36.87 dB | > 37.5 dB（+0.6 dB） |
| roll_frac @ 150K+ | 5-9%（V3 plateau 区） | **40-50%（归一化保证）** |
| best step 位置 | 86800（43% 处） | > 150K（75% 处，plateau 推迟） |

---

## 5. 与 V5 实验的关系

- V5 的 01/03 实验结果仍然有价值（确认 resume fine-tune 是否有效）
- V6 是**独立于 V5 的 from-scratch 实验**
- 如果 V5 full-val 显示 rollout-heavy resume 有效 → V6 仍然值得做（从 scratch 训可能更好）
- 如果 V5 full-val 显示无效 → V6 成为主方向

---

## 6. 时间估算

| 阶段 | 耗时 | GPU |
|------|------|-----|
| P0 代码改动（loss 归一化） | ~1h | 无 |
| V6 从 scratch 训练 200K | ~5-6 天 | 1 卡 |
| Full-val eval | ~1h | 1 卡 |
| Path A 诊断 | ~30min | 1 卡 |
| **总计** | ~6 天 | 1 卡 |
