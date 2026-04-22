# Scheme C v2 实验计划 — 2026-04-22

## 背景：为什么必须中断旧实验并启动 v2

### Oracle 实验结论（本轮关键发现）

GT latent 直接 decode 的 PSNR ceiling:

| Timepoint | Oracle PSNR | Transport Best | Gap |
|-----------|------------|---------------|-----|
| D50 | 42.622 dB | 42.622 dB | 0 dB（输入） |
| D20 | 46.636 dB | 35.924 dB | **10.7 dB** |
| D10 | 48.744 dB | 36.094 dB | **12.6 dB** |
| D4 | 50.832 dB | 36.147 dB | **14.7 dB** |
| NORMAL | 52.634 dB | 36.450 dB | **16.2 dB** |

**结论**：frozen decoder 完全不是瓶颈（ceiling 46-52 dB），transport backbone 的预测能力严重不足（只达到 35-36 dB，gap 10-16 dB）。

### 0422 审计发现的根本原因

通过 4-agent transport 深度审计，发现之前**所有实验**都在以下 bug 下运行：

1. **weight_decay=0.01 压死 gate** → `g_pix_raw` 和 `lambda_hop_raw[0:4]` 被 AdamW 持续衰减到 floor
2. **lambda_hop_init=0.01 + hop_residual_last_init_std=0.002** → hop residual 梯度被压制 ~100×，分支从未激活
3. **无 EMA** → RAE 训练器用 EMA 保存 checkpoint（decay=0.9999），PET_LatentResidual 训练器不用 EMA → checkpoint 质量差

**这意味着之前所有实验中 hop residual 和 pixel forcing 两个核心分支形同虚设。** 所有 ±0.1 dB 的改进都是在一个核心组件被静默禁用的系统中噪声级别的波动。

### 已实施的修复

| Fix | Before | After | Commit |
|-----|--------|-------|--------|
| weight_decay on gates | 0.01 | **0.0** (first_hop group) | `0b109f8` |
| lambda_hop_init | 0.01 | **0.10** (10× increase) | `0b109f8` |
| hop_residual_last_init_std | 0.002 | **0.01** (5× increase) | `0b109f8` |
| EMA support | 无 | **decay=0.9999** (match RAE) | `d2ac918` |
| N2 enforce --resume | 无检查 | RuntimeError if missing | `a8fd9dd` |
| pixel semantic check | 硬编码 True | 读取实际 config | `a8fd9dd` |

---

## v2 实验配置

**Config**: `configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml`

**关键参数对比**:

| 参数 | 旧 Scheme C | v2 |
|------|-----------|-----|
| run_name | first_hop_224_50k_imgaux_boost | **first_hop_224_50k_schemec_v2** |
| first_hop_weight_decay | 0.01 (继承 global) | **0.0** |
| lambda_hop_init | 0.01 | **0.10** |
| hop_residual_last_init_std | 0.002 | **0.01** |
| EMA | 无 | **enabled, decay=0.9999** |
| 其余参数 | — | 完全相同 |

**其他不变的参数**: seed=42, max_steps=50000, batch_size=8, backbone_lr_mult=0.4, first_hop_lr_mult=3.5, lambda_max=0.12

---

## 运行命令

```bash
# GPU-3: Scheme C v2 (50K steps)
CUDA_VISIBLE_DEVICES=3 TQDM_DISABLE=1 \
python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml \
    2>&1 | tee review/0422/exp/schemec_v2_train_gpu3.log
```

---

## 为什么必须中断当前训练

1. **N1 (pixel encoder ablation)** — 在旧参数下运行，pixel forcing 分支从未激活，消融结果无意义
2. **任何旧 Scheme C 的后续实验** — 都在 bug 参数下，结论不可信

v2 是修复 bug 后的**第一个公平 baseline**。只有 v2 结果出来后，才能判断：
- hop residual 和 pixel forcing 是否真正有效
- 是否需要进一步调整 backbone_lr_mult / velocity_rebalance / step_weights

---

## v2 成功判据

### 必须观察的指标

1. **`gate_pix` 是否增长**（旧实验中趋向 floor ~0.001，v2 应明显上升）
2. **`lambda_hop_0` 是否增长**（旧实验中停滞在 ~0.01，v2 应向 0.2-1.0 增长）
3. **`pix_delta_abs_hop0` 是否非零**（旧实验中 ~0，v2 应 > 0.01）
4. **`v_hop_abs_hop0` 是否非零**（旧实验中 ~0，v2 应 > 0.001）

### PSNR 判据

- 如果 val_select_score **< 0.000500**（比旧 best 0.000532 改善 > 6%）→ fix 有效
- 如果 val_select_score **与旧值相近**（±0.000020）→ fix 无影响，需追查更深层问题
- 如果 **退化** → 初始化过大，需回调

---

## 后续实验矩阵（v2 之后）

| 实验 | 条件 | GPU | 目的 |
|------|------|-----|------|
| v2 + backbone_lr=1.0 | v2 结果出来后 | 待分配 | 测试 backbone 学习速率对 transport 的影响 |
| v2 + velocity_rebalance | v2 结果出来后 | 待分配 | 测试 hop 间梯度平衡 |
| v2 + 100K steps | v2 结果出来后 | 待分配 | 测试更长训练是否继续改善 |

---

## Codex 审计合规性

本次变更已通过以下审计：
- ✅ 0422 adversarial audit (16 subagent, 5 dimension) → 4 fixes applied
- ✅ Codex f33ddaa audit (Mendel/gpt-5.4) → P0-1/P0-2 fixed, P1-2 overruled
- ✅ Oracle ceiling experiment → decoder NOT bottleneck, transport IS
- ✅ Transport deep dive (4 agent) → EMA missing identified and fixed
- ✅ py_compile: ALL OK
- ✅ YAML parse: ALL OK

**请求**: 中断当前 N1 训练（旧参数下结果无参考价值），释放 GPU 用于 v2 实验。
