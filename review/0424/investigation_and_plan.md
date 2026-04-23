# 0424 调查报告与实验规划

## 1. 36.2 dB Plateau 根因分析

### 已排除的假设

| 假设 | 实验 | 结果 | 结论 |
|------|------|------|------|
| gate/lambda 被 weight_decay 压死 | v2 (fix init + wd=0) | 36.197 dB | ❌ 不是瓶颈 |
| 缺少 EMA | v2 (EMA enabled) | 36.197 dB | ❌ 不是瓶颈 |
| pixel forcing 有帮助 | N1 (pixel OFF) | 36.123 dB | ❌ 无正面贡献 |
| backbone LR 太低 | lr1 (0.4→1.0) | 36.188 dB | ❌ 不是瓶颈 |
| decoder 是瓶颈 | Oracle (GT decode) | 46-52 dB | ❌ decoder 完美 |

### Plateau 的真正原因：latent MSE 到 image PSNR 的非线性放大

**关键数据**：
- 每 0.0001 latent MSE → 损失 5.4 dB (D20) 到 9.4 dB (NORMAL) image PSNR
- 当前 transport latent MSE ≈ 0.0002 per element
- 要从 36 dB 提升到 46 dB → 需要把 latent MSE 降低 10×

**所有实验的 latent MSE 都稳定在 0.00017-0.00021**，无论改什么参数。

### 两个可能的突破方向

**方向 A：训练不够充分（50K 步 = 3.8 遍数据）**
- latent MSE 可能还没真正收敛
- 200K 步 = 15.2 遍数据，给模型更多学习机会
- 这是最保守、风险最低的改动

**方向 B：loss 机制的结构性缺陷（image loss 只覆盖 hop0）**
- 当前 image_aux loss 只在 hop0 对 backbone 施加 decoder 敏感方向的梯度
- hop1-3 的 backbone 只接收 latent MSE 梯度，不知道 decoder 对哪些 latent 方向更敏感
- 可通过随机采样 hop 做 image loss 来解决（零额外内存）

---

## 2. 全实验结果总表（full-val rerank, clip3 PSNR）

| 实验 | backbone LR | pixel | EMA | 权重修复 | **transport_avg** |
|------|------------|-------|-----|---------|-----------------|
| **C best** | 0.4× | ON | ❌ | ❌ | **36.206** |
| C last | 0.4× | ON | ❌ | ❌ | 36.204 |
| v2 best | 0.4× | ON | ✅ | init fix | 36.197 |
| lr1 last | 1.0× | OFF | ❌ | init fix | 36.188 |
| N1 last | 0.4× | OFF | ❌ | init fix | 36.161 |
| N1 best | 0.4× | OFF | ❌ | init fix | 36.123 |

**所有结果在 36.12-36.21 范围内，差异 < 0.1 dB。**

---

## 3. 新实验：200K 步 Transport v3

### 实验目的

测试 36.2 dB plateau 是否因为训练覆盖不足（3.8 遍数据 → 15.2 遍）。

### Config

`configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml`

### 与 v3-50K 的差异

| 参数 | v3-50K | v3-200K |
|------|--------|---------|
| max_steps | 50000 | **200000** |
| save_interval | 10000 | **20000** |
| 其余 | — | 完全相同 |

### v3 已包含的 4 项修复

1. pair_loss_weights 反转：`[1.20, 1.10, 1.05, 1.00]`
2. rollout step_weights 反转：`[1.30, 1.20, 1.10, 1.00]`
3. val_multi_objective D20 权重：0.15 → 0.50
4. velocity_rebalance：enabled（sqrt_ratio, clip [1.0, 4.0]）

### 运行命令

```bash
# GPU (50K v3 完成后的空闲 GPU):
CUDA_VISIBLE_DEVICES=<GPU_ID> TQDM_DISABLE=1 \
python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml \
    2>&1 | tee review/0424/logs/transport_v3_200k_train.log
```

### 训练时间预估

- 50K 步 ≈ 16h → 200K 步 ≈ **64h（~2.7 天）**
- 保存 10 个 checkpoint（每 20K 步）

### 内存预估

- CPU RAM：~220 GB（与 50K 实验相同）
- GPU 显存：与 50K 实验相同

---

## 4. 监控要点

### 收敛观察

每 50K 步检查 val_select_score 是否还在下降：

```bash
# 从 metrics.jsonl 提取 val_select_score 趋势
python3 -c "
import json
with open('/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3/metrics.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('event') == 'val' and d['step'] % 10000 == 0:
            print(f'step={d[\"step\"]:6d} select={d.get(\"val_select_score\",0):.6f} d20={d.get(\"val_chain_d20_mse\",0):.6f}')
"
```

### 关键判断时间点

| Step | 数据遍历次数 | 对应之前的哪个实验 | 期望 |
|------|------------|------------------|------|
| 50K | 3.8× | 与 v3-50K 对比 | score ≈ v3-50K |
| 100K | 7.6× | 2× 之前训练量 | 如果 score 还在下降 → 继续有价值 |
| 150K | 11.4× | — | 如果 plateau → 训练覆盖不是瓶颈 |
| 200K | 15.2× | — | 最终结果 |

### 提前停止条件

- 如果 step 100K-150K 之间 score 完全不动（< 0.5% 变化）→ 可提前终止，节省 GPU 时间
- 如果 score 持续下降 → 跑满 200K

---

## 5. 后续实验路线

### 如果 200K 突破 plateau（transport_avg > 36.5 dB）

→ **训练覆盖不足是主因**，考虑：
- 400K 步实验
- grad_accum=4 提高梯度稳定性

### 如果 200K 未突破（transport_avg ≈ 36.2 dB）

→ **loss 机制的结构性问题**，实施方案 B：
- 随机 hop image loss（每步随机选 1 个 hop decode + image loss）
- 零额外 CPU/GPU 内存
- 让 backbone 在所有 hop 接收 decoder 敏感方向梯度
- ~20 行代码改动

### 如果 v3-50K 和 200K 都没突破

→ **需要重新审视 transport 范式**：
- 考虑减少 hop 数量（4→2：D50→D10→NORMAL）
- 考虑增大 latent 空间中 image loss 的权重
- 考虑部分解冻 decoder（只训 decoder 最后 1-2 层）

---

## 6. 服务器资源约束

| 资源 | 上限 | 当前使用 |
|------|------|---------|
| CPU RAM | 503 GB | ~220 GB/实验 → 最多 2 个并行 |
| GPU | 4 块 | v3-50K 占 1 块，200K 需要 1 块，backbone_lr1 已完成（空闲 1 块） |

**建议**：
- 如果 v3-50K 跑完后 GPU 空闲 → 立即启动 200K
- 200K 和 v3-50K 不要同时跑（CPU RAM 不够 3 个实验）
