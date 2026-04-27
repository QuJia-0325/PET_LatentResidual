# 0427 实验设计：V5 Rollout-Heavy 归因分析

## 背景：为什么需要这组实验

V5-main（rollout λ=1.5, image_aux=0.08, step_weights=[0.8,1.0,1.5,2.5]）跑了 14400 步后，
我们观察到 rolling-val 指标恶化，并初步判定为"相变失败"。

**但 Codex counter-review 指出了关键缺陷**：

1. **"step 93200 相变"实际是 rolling-val window 回卷**
   - step 92800 评估的是 val set 尾部（window_start=3648, 54 batches）
   - step 93200 评估的是 val set 头部（window_start=0, 64 batches）
   - val_pair_total 从 0→0.000231 的"跳变"可能只是不同数据子集的固有差异
   - V4 和 V5 在同一 step 出现"同步跳变"是因为共享相同的 eval schedule

2. **val_pair_total 后续回落到 0.000014-0.000046**
   - 如果是不可逆"盆地切换"，val_pair_total 不应回落
   - 回落说明模型参数并未被永久推离 GT-optimal manifold

3. **V4 和 V5 不是同构干预**
   - V4 SF: detach() 的 pred-state 单步 pair loss（无链式梯度）
   - V5 rollout: alpha=1.0 的多步 BPTT（有链式梯度）
   - 不能简单说"两者是同一种失败"

**结论**：我们之前基于 rolling-val 做出的"V5 失败"判断可能是过度推断。
需要用严格的评估协议重新判断。

---

## 实验设计

### 实验 01：Full-Val 评估（消除 rolling-val 噪声）

**要回答的问题**：V5 best.pt 在全量 val set 上的 PSNR/MSE 相比 V3 baseline 到底是改善、持平还是恶化？

**为什么需要**：
- 训练中的 rolling-val 每次只看 64 batches（~512 samples），且 window 位置每次不同
- rolling-val 的 CV（变异系数）≈ 29%，单点不可信
- Full-val 使用全部 7403 val slices，消除窗口选择偏差

**具体做法**：
1. 对 V3 200K best.pt 跑 `eval_first_hop_224_clip3.py --max-slices 0`（全量）
2. 对 V5 best.pt 跑同样的 eval
3. 对 V5 最新 checkpoint 跑同样的 eval
4. 比较三者的 `chain_normal_psnr`（医生关心的 NORMAL 档）和 `transport_avg_psnr`

**关键输出**：每个 ckpt 生成一个 JSON 文件，包含 per-hop 的 PSNR 和 MSE。

**脚本**：`01_fullval_eval.sh [GPU_ID]`

---

### 实验 02：Path A 诊断（V3 baseline ExpoGap）

**要回答的问题**：200K v3 best.pt 的 exposure bias gap 是多少？

**为什么需要**：
- 之前的 Path A 数据（ExpoGap=4.80 dB）来自 Scheme C best（step ~46K），不是 200K v3
- 200K 训练了更多步，ExpoGap 可能已经变化
- 需要一个公平的 ExpoGap baseline 来与 V5 对比

**具体做法**：
- 对 V3 200K best.pt 跑 `diagnose_tf_rollout_gap.py --max-slices 0`
- 输出 Teacher-Forcing vs Rollout 的 per-hop PSNR gap

**关键输出**：
```
Path A: Teacher-Forcing vs Rollout Gap Diagnostic
Hop          PSNR_TF    PSNR_RO    ExpoGap    CeilGap
D50->D20     ...        ...        ...        ...
D20->D10     ...        ...        ...        ...
D10->D4      ...        ...        ...        ...
D4->NORMAL   ...        ...        ...        ...
```

**脚本**：`02_pathA_baseline.sh [GPU_ID]`

---

### 实验 03：Null-Control（区分 V5 loss 改动 vs resume/LR 效应）

**要回答的问题**：如果从 V3 best.pt resume 继续训 50K 步，但**不改任何 loss 权重**，val 会怎样？

**为什么需要**：这是最关键的对照实验。V5-main 同时做了两件事：
1. 改了 loss 权重（rollout λ ↑, image_aux ↓, step_weights 反转）
2. 从 step 86800 resume，LR 重新走 cosine schedule（max_steps=136800 vs 原 200000）

如果 null-control（只做第 2 件事）也退化——说明退化是 resume/LR 造成的，不是 V5 的 loss 改动。
如果 null-control 持平——说明 V5 的 loss 改动确实是退化主因。

**具体做法**：
- 使用 200K v3 **原始 config**（所有 loss 权重不变）
- 仅改 run_name、resume 兼容性、max_steps
- 从 V3 best.pt resume 继续训 50K 步
- 对比训练结束后的 val 指标与 V3 baseline

**Config**：`configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml`
- 基于 200K v3 config 复制
- 仅改：`run_name`, `strict_resume_compat: false`, `resume_allow_*: true`, `progress_bar: false`, `log_grad_norms: true`
- **所有 loss 权重完全不变**（rollout λ=0.25, image_aux=0.12, step_weights=[1.30,1.20,1.10,1.00]）

**脚本**：`03_null_control.sh [GPU_ID]`

---

### 实验 04：Path A 诊断（V5 best ExpoGap）

**要回答的问题**：V5 best.pt 的 exposure bias 相比 V3 baseline 是否有变化？

**为什么需要**：
- 即使 full-val PSNR 没变，ExpoGap 可能已经改善——说明 rollout-heavy 在缓解 exposure bias
- 反之，如果 ExpoGap 恶化——说明强 rollout 反而加剧了链式误差

**脚本**：`04_pathA_v5_best.sh [GPU_ID]`

---

## 判定矩阵

### Full-Val（01）结果

| V5 best full-val vs V3 full-val | 结论 | 下一步 |
|--------------------------------|------|--------|
| V5 PSNR 改善 ≥ 0.3 dB | **V5 有效**——rolling-val 掩盖了真实改善 | 继续跑 V5 更多步 |
| V5 PSNR ± 0.1 dB | **持平**——V5 没改善也没恶化 | 看 null-control 区分原因 |
| V5 PSNR 恶化 ≥ 0.3 dB | **V5 有害** | 停止 V5，看 null-control 定位原因 |

### Null-Control（03）+ V5 交叉判定

| null-control 50K | V5 50K | 诊断 |
|-----------------|--------|------|
| 持平/改善 | 恶化 | **V5 loss 改动是退化主因** |
| 持平 | 持平 | 继续训练无效，plateau 是真实瓶颈 |
| 退化 | 退化 | **resume + LR schedule 是主因**，与 V5 loss 无关 |
| 改善 | 改善 | 两者都受益于继续训练，V5 的贡献待 ablation |

### Path A（02 + 04）交叉判定

| V5 ExpoGap vs V3 ExpoGap | 含义 |
|--------------------------|------|
| V5 下降 ≥ 10% | rollout-heavy 在缓解 exposure bias |
| V5 ≈ V3 | rollout 权重变化未影响 exposure bias |
| V5 上升 ≥ 10% | rollout-heavy 反而加剧了链式误差 |

---

## 文件清单

```
review/0427/run_me/
├── README.md                  ← 本文件（设计说明）
├── 01_fullval_eval.sh         ← Full-val 评估 V3/V5
├── 02_pathA_baseline.sh       ← V3 Path A ExpoGap baseline
├── 03_null_control.sh         ← Null-control 训练（50K 步）
└── 04_pathA_v5_best.sh        ← V5 Path A ExpoGap

configs/
└── pet_flow/pet_flow_first_hop_224_v5_null_control.yaml  ← Null-control config
```

## 执行顺序

```bash
cd /home/qujiaxiang/project/PET_LatentResidual && git pull

# P0（立即，~1.5h）——回答"V5 full-val 到底有没有改善"
bash review/0427/run_me/01_fullval_eval.sh 0
bash review/0427/run_me/02_pathA_baseline.sh 0

# P1（并行，03 需要 ~30h）——回答"退化是 V5 loss 还是 resume/LR"
bash review/0427/run_me/03_null_control.sh 2
bash review/0427/run_me/04_pathA_v5_best.sh 0
```
