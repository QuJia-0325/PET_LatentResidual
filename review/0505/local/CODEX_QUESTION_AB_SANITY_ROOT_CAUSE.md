# Question for Remote Codex — A/B σ-normalize Sanity FAIL 工程根因诊断

**目的**：远程 codex 帮助定位 A/B sanity（应数学严格等价的两个 SGD 路径）在 fp32 deterministic 训练下产生 trajectory 分叉的工程根因，并给出具体修复方案。

**背景一句话**：A 和 B 的 SGD 梯度在 fp64 下逐步等价（残差 ~1e-20），训练设置 `amp=false / deterministic=true / seed=42 / num_workers=0`，前 5000 步（Phase I, λ_roll=0）loss/pair/grad 三项完全 0.00% bit-equivalent，**但** λ_roll ramp 一启动（step 5050）后立刻分叉，到 step 10000 已偏 30%，最终 50K sanity gate 失败：

```
Tier 1 val_pair_total                rel_err=10.16%   (threshold 0.01%) FAIL
Tier 2 val_rollout_total             rel_err= 1.85%   (threshold  1%)  FAIL
Tier 3 val_rollout_step_3_raw        rel_err= 4.92%   (threshold  1%)  FAIL
Tier 4 val_chain_normal_mse          rel_err=12.73%   (threshold  5%)  FAIL
```

我（本地 Claude）的初步假说是 bf16 AMP 浮点不可交换，但代码读完确认 `amp: false` —— 训练是纯 fp32，假说被推翻。**这是为什么需要远程 codex 帮忙定位真正根因**。

---

## 1. 关键数据 — 逐步分叉时间线

来自 [review/0505/logs_sanity/run_ablation_A_sanity_train.log](../logs_sanity/run_ablation_A_sanity_train.log) 与 B 对应文件，按 train step 对齐每行 `[train] step=...` metric line：

| step | λ_roll | A_loss | B_loss | rel% | A_pair | B_pair | rel% | A_grad_total | B_grad_total |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 0.000 | 0.003553 | 0.003553 | **+0.00%** | 0.000198 | 0.000198 | **+0.00%** | 1.8022e-02 | 1.8022e-02 |
| 500 | 0.000 | 0.002606 | 0.002606 | +0.00% | 0.000162 | 0.000162 | +0.00% | 1.1408e-02 | 1.1408e-02 |
| 1000 | 0.000 | 0.003261 | 0.003261 | +0.00% | 0.000176 | 0.000176 | +0.00% | 1.3559e-02 | 1.3559e-02 |
| 2000 | 0.000 | 0.002603 | 0.002603 | +0.00% | 0.000146 | 0.000146 | +0.00% | 9.3666e-03 | 9.3666e-03 |
| 3000 | 0.000 | 0.003562 | 0.003562 | +0.00% | 0.000212 | 0.000212 | +0.00% | 1.3898e-02 | 1.3898e-02 |
| 5000 | 0.000 | 0.005581 | 0.005581 | +0.00% | 0.000345 | 0.000345 | +0.00% | 1.4840e-02 | 1.4840e-02 |
| **5050** | **0.020** (ramp 启动) | — | — | — | — | — | — | — | — |
| 5400 | ≈0.32 | — | — | — | 1.66e-04 | 1.67e-04 | **+0.60%** (首次 >0.1%) | — | — |
| 10000 | 2.000 | 0.001142 | 0.001018 | **−10.86%** | 0.000033 | 0.000023 | **−30.30%** | 1.6871e-02 | 1.6950e-02 |
| 20000 | 4.000 | 0.006722 | 0.007471 | +11.14% | 0.000166 | 0.000180 | +8.43% | 7.3404e-02 | 8.1046e-02 |

**核心观察**：
1. λ_roll=0 期间 6 个 checkpoint 全部 0.00% bit-equivalent（含 grad_total 这种 reduction 后量），证明 RNG / dataloader / weight init / forward of pair channel / forward of rollout (for logging) 全部 byte-identical。
2. λ_roll 从 0 跳到 0.02 后 350 步内（5050 → 5400）pair_loss 首次出现 0.6% 偏差。pair_loss 自身的代码路径**完全不经过 step_normalizers**（见下文 §3），所以 pair_loss 的偏差只能来自参数已经飘了。
3. 350 步内参数飘到能让 pair_loss 偏 0.6% 是非常剧烈的放大。1e-7 fp32 ULP 单步误差 → 0.6% 偏差需要 4 个数量级的放大；Adam / chaotic SGD 在 lr=2e-4 下能否做到这个量级，需要远程 codex 判断。

---

## 2. 训练设置（已 100% 确认）

[train_first_hop.py L43-52](../../../train_first_hop.py#L43-L52) seed 设置：
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
```

[review/0505/logs_sanity/run_ablation_A_sanity_config.resolved.yaml](../logs_sanity/run_ablation_A_sanity_config.resolved.yaml) 与 B 对应文件 grep 结果：
```
A: seed: 42, num_workers: 0, amp: false, deterministic: true
B: seed: 42, num_workers: 0, amp: false, deterministic: true
```

A 和 B 的 yaml 实质差异**只有两处**（其他都是注释）：
- `sigma_normalize.enabled`：A=false / B=true
- `step_weights`：A=`[0.5, 2.0, 1.5, 1.0]` / B=`[1.7908, 1.8606, 0.8691, 0.4795]` (= w_v6 × n)

`n = [3.5816, 0.9303, 0.5794, 0.4795]`，`Σ(w_v6 × n) = 5.0 = Σw_v6`（preserve_v6_sum 模式），所以 `Σ((w_v6×n) × (raw/n)) ≡ Σ(w_v6 × raw)` 在 fp64 下 bit-by-bit 严格相等（fp64 残差 ~1e-20，[review/0502/scripts/verify_normalizers.py](../../0502/scripts/verify_normalizers.py) 已证）。

---

## 3. A 和 B 在前向计算中的唯一差异路径

[pet_lr/rollout_first_hop.py L97-119](../../../pet_lr/rollout_first_hop.py#L97-L119)：

```python
for hop_idx in range(num_steps):
    ...
    z_pred = out["z_pred"]
    z_gt = z_rollout[:, hop_idx + 1]
    step_loss = _latent_loss(z_pred, z_gt, loss_type=loss_type)   # F.mse_loss, fp32
    step_losses_raw.append(step_loss.detach())                    # 仅诊断用
    if step_normalizers is not None:                              # ← 这里是 A 和 B 的唯一差异
        step_loss = step_loss / float(step_normalizers[hop_idx])
    step_losses.append(step_loss)
    ...

w = torch.tensor(step_weights, dtype=step_losses[0].dtype, device=step_losses[0].device)
stacked = torch.stack(step_losses, dim=0)
total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
```

逐 hop 在 fp32 下计算：
| 路径 | step_loss 操作 | 进入 reduction 的元素 |
|---|---|---|
| A | `_latent_loss` 只算一次 | `[raw_0, raw_1, raw_2, raw_3]` × `[0.5, 2.0, 1.5, 1.0]` |
| B | `_latent_loss` + 一次 fp32 除法 | `[raw_0/3.58, raw_1/0.93, raw_2/0.58, raw_3/0.48]` × `[1.79, 1.86, 0.87, 0.48]` |

**fp64 等价性**：B 的每个 product `(raw_k / n_k) × (w_v6_k × n_k) ≡ raw_k × w_v6_k` 在 fp64 下严格相等。

**fp32 下的 ULP 来源**（我能想到的，请 codex 复核）：
1. `step_loss / n_k` 的 fp32 除法本身有 ~0.5 ULP 截断（≈6e-8 相对）。
2. 后续 `step_loss * w_k` 的 fp32 乘法引入第二次 ~0.5 ULP（≈6e-8 相对）。
3. `(stacked * w).sum()` 的 4 元素 reduction 顺序在 fp32 下不一定 commutative；A 和 B 的 stacked 元素**绝对量级范围不同**（A 的 raw_k ≈ 5e-4 全相近；B 的 raw/n 在 1.4e-4 ~ 1.04e-3 跨 7×），可能触发 PyTorch / cuBLAS 选不同的 reduction kernel 或 SIMD 路径。
4. `_latent_loss` 内部的 `F.mse_loss(pred, target)` 是 `((pred-target)**2).mean()`，pred/target 在 A 和 B 之间在 step 5050 之前是 byte-identical 的（已由 Phase I 数据证），所以 `_latent_loss` 输出在 step 5050 那一步**也是 byte-identical**。差异只能来自 `/n_k` 之后。

**预期单步偏差量级**：~1e-7 相对；**实测**：350 步累积到 0.6% pair_loss 偏差，相当于参数空间漂移 ~1e-3 量级。

---

## 4. pair_loss 不经过 normalizer，但仍然首先暴露偏差

[train_first_hop.py L292-358](../../../train_first_hop.py#L292-L358) `compute_pair_losses`（pair channel）和 [rollout_first_hop.py](../../../pet_lr/rollout_first_hop.py) `rollout_multistep_losses_first_hop`（rollout channel）是两条**不共享 normalizer 的代码路径**：

- pair: `vel_err = (v_pred - v_target).pow(2).mean()` (无 normalizer)，权重来自 `pair_loss_weights = [2.5, 1.0, 1.0, 1.0]`，与 A/B 的差异点完全无关。
- rollout: 上面 §3 描述。

**合理推论**：pair_loss 在 step 5400 偏差 0.6% **只能**来自 model 参数已经飘了（因为 pair forward pass 在 A 和 B 之间应该 byte-identical，输入 z_src/z_dst 也 byte-identical）。参数飘的唯一来源是 step 5050~5400 之间 350 步的 rollout backward 梯度差异。

→ **rollout backward 的 fp32 ULP 在 350 SGD 步里被放大 ~10⁴ 倍才能解释这个观测**。

---

## 5. 我对 codex 的 5 个具体问题

### Q1：fp32 + deterministic + 同 seed 是否真的应该 bit-equivalent？

我的理解：在 PyTorch fp32 + `torch.use_deterministic_algorithms(True)` + cudnn.deterministic=True + amp=false 下，相同 input shape & dtype & values 应该产生 bit-identical output（forward）和 bit-identical gradient（backward）。**但 A 和 B 的 input 不是 byte-identical**（B 的 stacked 已经 /n_k），所以：
- forward kernel（reduction sum）即使 deterministic，对不同 input 仍然合法地产生不同 output，因为 reduction 顺序可能依赖 operand 大小（e.g., Kahan summation 路径选择）。
- 这是否就是 fp32 deterministic 模式下"等价数学公式不保证 byte-equal"的合理解释？

**请 codex 确认或纠正**：在 PyTorch 2.x fp32 deterministic 模式下，下面两种写法
```python
# A
total_a = (torch.stack([r0, r1, r2, r3]) * torch.tensor([0.5, 2.0, 1.5, 1.0])).sum()
# B
total_b = (torch.stack([r0/n0, r1/n1, r2/n2, r3/n3]) * torch.tensor([0.5*n0, 2.0*n1, 1.5*n2, 1.0*n3])).sum()
```
是否预期 `total_a == total_b` 严格 byte-equal？还是预期有 ~1e-7 ULP 偏差？

### Q2：单步 1e-7 ULP → 50K 步累积 10% 是否合理放大？

请 codex 基于你对 SGD chaos 的理解判断：
- AdamW + lr=2e-4 + cosine schedule
- 模型 ~XXX M 参数（预计），rollout backward grad_total ~1.5e-2
- 单步 grad 相对偏差 ~1e-7
- 50K 步后 val_pair_total 偏 10%

**这是否在 fp32 SGD irreproducibility 文献的"正常范围"内**？还是说 4 个数量级的放大暗示有 ULP 之外的 bug（例如某个非 deterministic op 漏标）？

参考文献我能想到：Pham et al. NeurIPS 2020 "Reproducibility in ML" 实测 fp32 deterministic 下同 seed 同代码 GPU 间也能差 ~0.1% 测试精度。我们这里 10% 的 val_pair 偏差是不是太大了？

### Q3：是否应该把 step_loss 除法 cast 到 fp64？

修复路径 A —— 在 [rollout_first_hop.py L100](../../../pet_lr/rollout_first_hop.py#L100) 改为：
```python
if step_normalizers is not None:
    step_loss = (step_loss.double() / float(step_normalizers[hop_idx])).float()
```

预期效果：
- 把 `/n_k` 的 ULP 从 6e-8 降到 1e-16
- 但下游 `(stacked * w).sum()` 还是 fp32，仍可能因 operand 范围差异选不同 kernel
- 实际 trajectory 是否能因此 byte-equal？还是只是减小漂移幅度？

**请 codex 评估**：(a) 这个改动是否能让 sanity Tier 1 通过（<0.01%）？(b) 有没有更彻底的方式（例如把整个 `total` reduction 都做 fp64）？(c) fp64 reduction 的训练速度损失估计？

### Q4：是否应该把 [stacked * w].sum() / w.sum() 整体改成 fp64？

修复路径 B —— [rollout_first_hop.py L117-119](../../../pet_lr/rollout_first_hop.py#L117-L119) 改为：
```python
w = torch.tensor(step_weights, dtype=torch.float64, device=step_losses[0].device)
stacked = torch.stack(step_losses, dim=0).double()
total = ((stacked * w).sum() / w.sum().clamp_min(1e-8)).float()
```

仅 4 元素 reduction，fp64 开销可忽略，但能让 A 和 B 的 `total` 值在 fp64 域下严格相等（数学等价性已证）。然后 `.float()` cast 回 fp32 进 backward — backward 路径的 grad w.r.t. raw_k 也会因为 fp64 中间量更接近精确值而更稳定。

**请 codex 评估**：这个方案能否真正修复 A==B sanity？特别是 backward 路径的 grad 是否仍然 byte-equal？

### Q5：sanity gate 阈值合理性

[review/0502/SIGMA_NORMALIZE_ABLATION_PLAN.md §3.4](../../0502/SIGMA_NORMALIZE_ABLATION_PLAN.md) 锁定的 4-tier 阈值：
```
Tier 1 val_pair_total       <0.01%   (bit-equal)
Tier 2 val_rollout_total    <1%
Tier 3 step*_raw            <1%
Tier 4 chain_*              <5%
```

**前提**：在数学等价的 SGD 路径下，50K 步训练后这些指标**应该**满足这些阈值。

**问题**：如果远程 codex 确认 "fp32 ULP 单步 1e-7 在 50K 步累积到 1-10% 是 SGD chaos 的物理 floor"，那 Tier 1 阈值 0.01% **物理上根本不可达** —— 这个 sanity gate 的设计本身有问题。

**请 codex 给出建议**：
- (a) 把 sanity 改成在 Q3/Q4 提供的 fp64 修复后跑，重新评估阈值是否可达？
- (b) 还是接受 fp32 trajectory chaos floor，把阈值放宽到现实 floor（Tier 1: 2%, Tier 2: 5%, Tier 3: 10%, Tier 4: 15%）？
- (c) 还是改用更严格的 sanity protocol（例如：只跑 1 个 SGD step 比较 grad，或只跑 100 步前比较 loss / param）？

---

## 6. 想让 codex 直接看的 4 个文件

按重要性顺序：

1. [pet_lr/rollout_first_hop.py L48-122](../../../pet_lr/rollout_first_hop.py#L48-L122) — `rollout_multistep_losses_first_hop` 主函数，A/B 唯一差异点在这里
2. [train_first_hop.py L390-465](../../../train_first_hop.py#L390-L465) — `_compute_sigma_dt_normalizers`，证明 A==B 的数学构造
3. [review/0502/scripts/verify_normalizers.py](../../0502/scripts/verify_normalizers.py) — fp64 残差 ~1e-20 的解析验证脚本
4. [review/0505/logs_sanity/run_ablation_sanity_gpu1_20260503.launch.log](../logs_sanity/run_ablation_sanity_gpu1_20260503.launch.log) 末尾 30 行 — 官方 sanity gate 输出

如果 codex 想看其他文件，明确列出我会补传。

---

## 7. 我希望 codex 回复的格式

用以下结构回答（可以加扩展但请保留这 5 个 section）：

### A. fp32 deterministic 下 A==B byte-equal 的预期
（Q1 的回答 — 是 yes 还是 no，理由是什么）

### B. fp32 ULP 累积放大量级评估
（Q2 的回答 — 我们观测到的 350 步内 6e-3 漂移、50K 步 10% val_pair 偏差，是否在 SGD chaos 文献的"合理范围"，还是暗示 bug）

### C. 最优修复路径推荐
（Q3 vs Q4 二选一或都做；附预期效果与训练速度代价）

### D. sanity gate 阈值建议
（Q5 — 修阈值还是修代码还是改 protocol）

### E. 其他你怀疑的根因
（如果 A-D 都不能解释，请列出你怀疑的其他可能 — 例如某个 PyTorch op 在我们的模型里漏标 deterministic、某个 eval pipeline 内部用了 nondeterministic reduction、checkpoint save/load 有截断、num_workers=0 仍有 dataloader randomness 等）

---

## 8. 我已经排除的 candidates（不要重复挖）

- ❌ RNG seed 不同：seed=42 一致，且 Phase I 0.00% bit-equivalent 证实数据流一致
- ❌ pair channel 实现 bug：pair 不经过 step_normalizers，但仍偏 — 说明是参数已飘，不是 pair 自身
- ❌ σ-normalize 数学公式错：fp64 残差 1e-20 已证
- ❌ σ-normalize 实现 bug：Phase I 已生效（roll0_A=0.001194 vs roll0_B=0.000333=0.001194/3.58）但 loss/grad 仍 0.00% — 证实 σ-normalize 在 Phase I 数学严格等价
- ❌ bf16 AMP 浮点不可交换：amp=false，训练是纯 fp32

---

## 9. 谢谢

如果远程 codex 需要本地 Claude 跑一些 sanity 验证脚本（例如：编一个 mini reproducer 验证 fp32 reduction order 假说），告诉我具体脚本设计，我会跑然后回报数据。
