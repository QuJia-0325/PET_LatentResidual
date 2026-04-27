# V6 Plan Supervisor Review — 潜在问题分析

**日期**：2026-04-27
**角色**：Supervisor
**评审对象**：[`review/plan/transport_breakthrough_research_v6.md`](../../plan/transport_breakthrough_research_v6.md)
**关联代码**：
- [`pet_lr/rollout_first_hop.py`](../../../pet_lr/rollout_first_hop.py)
- [`train_first_hop.py`](../../../train_first_hop.py) L1912-1918
**关联实测**：[`review/0427/logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl`](../logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl)

---

## 0. 总体结论

V6 的核心方向（pair_weight 放大 + alpha schedule 延长 + 通道-跳分离加权）方向正确，但当前 plan 的具体参数与论证存在 **3 个 Critical 级缺陷** 和 **3 个 Important 级缺陷**，若不修复直接 200K 训练将大概率重演 V3 的 plateau 退化。

| 级别 | 问题 | 是否阻塞 200K run |
|:-:|---|:-:|
| 🔴 Critical | image_aux λ 全程固定 0.04 | 是 |
| 🔴 Critical | pair_weight=15 cold-start 不稳定风险 | 否（但需早期监控） |
| 🔴 Critical | §1.5 Phase III 梯度分布预测算错 step 区间 | 否（影响 Go/No-Go 判定） |
| 🟡 Important | Phase I（0-50K）GT-only 过长 | 否 |
| 🟡 Important | step_weights[0]=0.5 与 §1.4 论证自相矛盾 | 否 |
| 🟡 Important | `lambda_scale_mode: none` 与 alpha 解耦的隐患 | 否 |
| 🟢 Minor | 命名歧义、未列 align/foc 系数 | 否 |

**最关键修复**：问题 #1（image_aux 退火）。这是 V3 → V6 真正的"治本"改动，其他都是边际。

---

## 1. 🔴 Critical：image_aux λ 全程固定 0.04 — V3 plateau 病灶未根除

### 1.1 plan 当前设计

V6 §1.6：
```yaml
image_aux:
  warmup_ratio: 0.0
  ramp_ratio: 0.0
  lambda_start: 0.04
  lambda_max: 0.04   # 全程固定
```

理由："image_aux 是 hop0 的固定辅助，不需要随训练阶段变化"。

### 1.2 V3 实测反驳

V3 200K JSONL 显示 `img_raw` 在 transport 收敛过程中**反向单调上升**：

| step | 含义 | img_raw |
|---:|---|---:|
| 86800 | best | 2.51e-3 |
| 150000 | plateau 中 | 5.08e-3 |
| 172600 | plateau 末 | 9.40e-3 |

img_raw 上升 **3.7×**。机制：当 transport 通道的 latent 收敛时，VAE decoder 把 latent 映射回像素仍有不可消除的 reconstruction floor，backbone 为追末端高频细节会**把 latent 推离 VAE manifold**，此时 image_aux 的梯度反而成为破坏因子。

### 1.3 用 V6 权重重算 Phase III 末期梯度比

直接套 V6 的 λ：

| step | pair_raw | roll_raw | img_raw | pair_w | roll_w | img_w | pair_frac | roll_frac | img_frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 86800  | 7.16e-5 | 6.90e-4 | 2.51e-3 | 1.07e-3 | 2.76e-3 | 1.00e-4 | 27.2% | 70.2% | **2.6%** |
| 150000 | 1.36e-5 | 1.38e-4 | 5.08e-3 | 2.04e-4 | 5.52e-4 | 2.03e-4 | 21.3% | 57.5% | **21.1%** |
| 172600 | 1.48e-5 | 1.07e-4 | 9.40e-3 | 2.22e-4 | 4.28e-4 | 3.76e-4 | 21.6% | 41.7% | **36.7%** |

**结论**：plan §1.5 把 86800 的快照投影到"Phase III 末期"，得到 image=3% 的乐观预测；但 V3 实际在 150K 之后 img_raw 持续上升，**V6 用同样的 λ 预测末期 image_frac ≈ 37%**，远超 §5 列的 ≤25% 红线。

### 1.4 修复建议

**Option A（推荐）**：阶梯式退火

```yaml
image_aux:
  lambda_schedule:
    - {step: 0,      lambda: 0.04}
    - {step: 130000, lambda: 0.04}
    - {step: 170000, lambda: 0.01}   # Phase III 中段开始撤
    - {step: 200000, lambda: 0.005}
```

**Option B（最小代码）**：与 alpha 同步 decay

```yaml
image_aux:
  # λ_eff = λ_max · (1 − α(t))，α=0 时 0.04，α=1 时 0
  decouple_from_alpha: false
```

**Option C（保守）**：保持 plan 当前设计，但将 §5 的 "img_frac < 25%" 改成 "img_frac < 40%"，并提前到 +130K 加 Go/No-Go 检查；如果 img_frac 上升趋势出现，touch 一个 SIGNAL_FILE 让训练 loop 切换 λ_img 到 0.01。

---

## 2. 🔴 Critical：`pair_weight=15` from-scratch cold-start 不稳定

### 2.1 风险分析

V3 全程 `pair_weight=1`，从未测试 15× 放大下的 cold-start 行为。LR=8e-5 是基于 pair=1 标定的。

数量级估算（hop0）：
- v_std² = 0.0096² ≈ 9.2e-5（GT velocity 模长平方）
- pair_loss 期望尺度 ~ v_std² 量级
- 乘 pair_weight=15 后，单 step pair 梯度等价于把 pair 通道 LR 拉到 ~1.2e-3

random init 下 velocity head 输出 ~ N(0, σ²) 的 σ 可能远大于 0.0096 量级，前 1K step 会出现：

1. velocity head 输出爆炸（loss 在前 100-500 step 飙到 1e-2 以上）
2. AdamW β2=0.999 二阶动量未收敛期（前 ~2000 step），15× 放大让 update 偏置严重
3. align/foc loss 被压抑（plan 未列它们 V6 取值）

### 2.2 修复建议

**Option A（推荐）**：pair_weight 短 ramp

```yaml
loss:
  pair_weight_schedule:
    - {step: 0,     pair_weight: 1.0}
    - {step: 10000, pair_weight: 15.0}
```

10K 步 ramp 让 backbone 先在 pair=1 下 warm up（与 V3 一致），再切到 V6 配方。

**Option B（最小改动）**：plan §4 监控加上 +1K / +3K checkpoint，监控 pair_loss 是否 explode（>5e-4 就 abort）。

---

## 3. 🔴 Critical：plan §1.5 Phase III 梯度分布预测错算 step 区间

### 3.1 错算复现

§1.5 写："全局分布（V3 step 86800 raw loss 估算）：pair ≈ 27%，rollout ≈ 70%，image ≈ 3%"

但 86800 是 V3 **best step**，对应 V6 的 Phase II 中段（α≈0.4），不是 Phase III（150K-200K，α=1）。Phase III 的 raw loss 应该用 V3 step 150K-172600 区间。

按 §1.3 推导：

| 用 V3 step | pair_frac | roll_frac | img_frac | 对应 V6 阶段 |
|---:|---:|---:|---:|---|
| 86800（plan 引用）  | 27% | 70% | **3%** | Phase II 中段 |
| 150000（应该用）    | 21% | 58% | **21%** | Phase II 末 / Phase III 早 |
| 172600（最准）      | 22% | 42% | **37%** | Phase III 中段 |

差距高达 **12×**（3% vs 37%）。

### 3.2 影响

§5 成功标准依赖 §1.5：
```
transport 总梯度占比 @ 150K+ : ≥ 75%
```

按错算预测 = 27%+70% = 97%，看起来轻松达成；按正确预测 = 22%+42% = **64%**，远低于 75% 红线。

**这导致 Go/No-Go 在训练后期会被"成功标准看似达成"误判通过，但模型实际仍处于 V3-style image-dominated 状态。**

### 3.3 修复建议

1. plan §1.5 重写表格，注明用的是 V3 step 172600 raw loss
2. §5 的 "≥ 75%" 阈值要么降到 ≥60%，要么必须配合问题 #1 的 image_aux 退火（两者必须二选一）

---

## 4. 🟡 Important：Phase I（0-50K，λ_roll=0）窗口过长

### 4.1 问题

V3 在 step 5K-20K 已开始混合 alpha；V6 把 GT-only 拉长到 0-50K（占总训练量 25%）。代价：

- hop1/2/3 在前 50K 只见 `(1-t)·z_GT^(k) + t·z_GT^(k+1)` 平直插值，对 pred-input 完全无暴露
- 50K 后 alpha 从 0 ramp，模型相当于"突然换数据分布"，前 ~5K step 适应抖动
- 50K 步 GT-only × pair_weight=15 极易让 backbone 在 GT segment 上过拟合（hop0 v_std=0.0096 GT 分布很窄）

### 4.2 修复建议

```yaml
rollout:
  warmup_ratio: 0.10   # 0-20K 纯 GT（从 0.25 降到 0.10）
  ramp_ratio: 0.65     # 20K-150K alpha+lambda ramp（从 0.50 提到 0.65）
```

总 ramp 长度从 100K 提到 130K，alpha 上升更平滑；早期 GT-only 缩短到 20K（与 V3 量级一致）。

---

## 5. 🟡 Important：`step_weights[0]=0.5` 与 §1.4 论证自相矛盾

### 5.1 矛盾点

§1.4 明确写：
> "rollout[hop0] 输入 ≡ GT z_D50，和 pair[hop0] 退化重叠，加权是浪费"

但 yaml 给 `step_weights = [0.5, 2.0, 1.5, 1.0]`，hop0 仍占 rollout 总权重的 10%（0.5/5.0）。

逻辑要么"完全重叠 → step_weights[0]=0"，要么"有边际信息 → 加权合理（但 §1.4 论证作废）"。**当前 0.5 是两边都不彻底的妥协。**

### 5.2 边际信息分析

rollout[hop0] 与 pair[hop0] 的差异：
- pair[hop0]：GT segment 上随机 t 采样的 velocity 监督
- rollout[hop0]：固定 t=0 → t_D20，输出与 z_GT_D20 的 endpoint 监督

确实有 endpoint 信息，不完全退化。但 endpoint 信息已被 pair_loss 的 endpoint 项（`endpoint_weight=1.0`）覆盖。

### 5.3 修复建议

**Option A（贯彻 §1.4 论证）**：

```yaml
step_weights: [0.0, 2.0, 1.5, 1.0]
```

腾出的预算保持原比例。

**Option B（保留 0.5）**：把 §1.4 改写为"hop0 rollout 仅作 endpoint 一致性辅助，不视为主要监督源"。

---

## 6. 🟡 Important：`lambda_scale_mode: none` 与 alpha 解耦的隐患

### 6.1 问题

设 effective λ_roll = 纯线性 ramp(0→4.0)，**不受 alpha 调制**。意味着：

- step 60K：α=0.2，λ_roll=0.8。此时 rollout chain 80% 输入仍是 GT，rollout signal 与 pair 高度同源，但仍以 0.8× 权重挤占梯度预算。
- step 100K：α=0.5，λ_roll=2.0。50% pred-input，rollout 开始有独占价值，但 λ 还没到峰值。
- step 150K：α=1.0，λ_roll=4.0。同步达峰，是合理的。

50K-80K 这 30K 步中，rollout 通道贡献边际有限却消耗预算。

### 6.2 修复建议

```yaml
rollout:
  lambda_scale_mode: alpha   # λ_eff = α · λ_end
  lambda_end: 4.0
```

让 rollout 真正与 exposure bias 同步上升，前期不浪费 rollout 预算。

---

## 7. 🟢 Minor 问题

### 7.1 命名歧义

4 个相关参数容易混淆：

| 参数 | 通道 | 维度 | V6 取值 |
|---|---|---|---|
| `pair_weight` | pair | scalar λ | 15.0 |
| `pair_loss_weights` | pair | per-hop (4,) | [2.5, 1.0, 1.0, 1.0] |
| `pair_sample_probs` | pair | per-hop (4,) | [0.25, 0.25, 0.25, 0.25] |
| `step_weights` | rollout | per-hop (4,) | [0.5, 2.0, 1.5, 1.0] |

建议在 config yaml 顶部加对照注释。

### 7.2 plan 未列 align / foc 的 V6 取值

`total_loss = pair + roll + img + foc + align`（[`train_first_hop.py` L1912-1918](../../../train_first_hop.py)）。§1.5 只算 3 项忽略 align/foc。如果它们沿用 V3 设置且 raw loss 未变，会偏移 §1.5 百分比 ~5-10%。建议附录列全 5 项 V6 系数。

---

## 8. 修复优先级与最小改动集

如果只能改 **3 处**，按以下优先级：

| 顺序 | 改动 | 文件 | 行数 |
|:-:|---|---|:-:|
| 1 | image_aux 130K 后退火至 0.005 | V6 yaml | 4 行 |
| 2 | pair_weight 0-10K ramp 1→15 | V6 yaml + train_first_hop.py | ~10 行 |
| 3 | §1.5 表格用 V3 step 172600 重算 | plan md | 1 表 |

如果可以改 **6 处**，加上：

| 顺序 | 改动 | 行数 |
|:-:|---|:-:|
| 4 | warmup_ratio 0.25→0.10 | 1 行 |
| 5 | step_weights[0] 0.5→0.0 | 1 行 |
| 6 | lambda_scale_mode none→alpha | 1 行 |

---

## 9. 不需要改的部分（保持原样）

- ✅ `pair_loss_weights = [2.5, 1.0, 1.0, 1.0]`（hop0 头重，与 v_std 一致）
- ✅ `step_weights` 中重押 hop1（ExpoGap × v_std 的 sweet spot 论证成立）
- ✅ `λ_pair=15, λ_roll=4, λ_img=0.04` 量级（only image 需要后期 decay）
- ✅ 200K from-scratch（不 resume V3）
- ✅ 三阶段训练框架

---

## 10. 决策树

```
启动 V6 200K 之前
├── 必须修复 #1 (image_aux 退火)
├── 必须修复 #3 (§1.5 数值 + §5 阈值同步)
├── 强烈建议修复 #2 (pair_weight ramp 或加 +1K ckpt 监控)
└── 可选 #4/#5/#6（性能优化，不阻塞）

训练中
├── +1K：检查 pair_loss 是否 explode（>5e-4 abort）
├── +10K：pair_frac ≈ 70-90%（plan §4 已有）
├── +50K：img_frac < 30%（plan §4 已有）
├── +130K：img_raw 是否开始上升（新增）
│   └── 上升趋势出现 → 切换 λ_img 到 0.01
├── +150K：transport 梯度占比 ≥ 60%（阈值从 75% 调低）
└── +200K：val_chain_normal_mse < 0.000150
```
