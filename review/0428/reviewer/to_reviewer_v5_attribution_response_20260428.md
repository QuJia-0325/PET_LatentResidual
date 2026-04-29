# To Reviewer: V5 Attribution Response — 2026-04-28

**回复对象**：[`v5_attribution_viewer_assessment_20260428.md`](v5_attribution_viewer_assessment_20260428.md)

---

## 0. 总体回应

Reviewer 的 assessment 方法严谨、证据边界清晰。我们接受全部 5 条 Finding 和 §4 措辞修订表。以下补充两个数据点。

---

## 1. 采纳的修订（Reviewer §4）

| 原措辞 | 修订后 |
|---|---|
| "image_aux 占据梯度预算" | "image_aux dominates weighted loss fraction / supervision pressure" |
| "null-control 证明 V3 plateau 不是训练量不足" | "null-control rolling snapshot supports the hypothesis, pending full-val" |
| "V3 plateau 是权重配方病" | "V3 plateau is consistent with a recipe/supervision-pressure bottleneck" |
| "V6 基于正确归因" | "V6 is motivated by the current attribution hypothesis" |

---

## 2. 补充 1：LR floor 替代解释可排除

Reviewer §1 和 Supervisor 裁决中列出了三个替代解释，其中 **LR floor 可以用现有数据排除**。

V3/null-control 使用 cosine LR（warmup=30K, total=200K, base=8e-5, min=2e-6）：

| step | cosine LR | 占 base % |
|---:|---:|---:|
| 86800（V3 best / resume 点） | **6.04e-5** | **75.5%** |
| 100850（null-control 当前） | 5.02e-5 | 62.8% |
| 130000 | 3.03e-5 | 37.9% |
| 150000 | 1.75e-5 | 21.9% |
| 200000（cosine 尾部） | 2.00e-6 | 2.5% |

null-control 在 step 86800-100850 区间的 LR 从 6.04e-5 到 5.02e-5，仅下降 14%，远未到 floor（200K 末尾的 2e-6）。

如果 LR floor 是 plateau 原因，null-control 在 75% base LR 下应该仍有改善能力。实际没有 → LR floor 不是主因。

**剩余替代解释**：
- ❌ LR floor — 已排除
- ⚠️ narrow basin — 无法从 null-control 排除，需 V6 from scratch 验证
- ⚠️ EMA 收敛稳态 — 与 narrow basin 本质相同

因此 null-control 将替代解释从 3 个缩减到 1 个（narrow basin），这进一步加强了"recipe 瓶颈"假设的可信度。

---

## 3. 补充 2：V6 Early Indicators（step 0-15K）

V6 已运行 15K 步，产生了 Reviewer §6 决策矩阵中未覆盖的早期信号：

### 3.1 +1K Go/No-Go ✅ PASS

| 检查项 | 结果 |
|---|---|
| pair_loss @ 1K | 1.77e-4（< 5e-4 ✓） |
| NaN/Inf | 无 ✓ |
| pair_loss_weight JSONL 记录 | 15.0 ✓ |

AdamW 尺度不变性论证得到实验验证。

### 3.2 V6 vs V3 同期 pair_frac 对比

| step | V3 pair_frac | V6 pair_frac | 说明 |
|---:|---:|---:|---|
| 1000 | 53.7% | **81.2%** | pair_weight=15 生效 |
| 5000 | 74.8% | **85.3%**（典型值） | V6 pair 主导更稳定 |
| 10000 | 33.2% | **60.7%** | V3 波动更大 |
| 平均 | ~55% | **~85%** | V6 pair 方向权重显著提高 |

pair_weight=15 在 Phase I 将 pair 的 weighted loss fraction 从 V3 的 ~55% 提升到 ~85%。虽然 AdamW 尺度不变性使步长不变，但梯度**方向**中 pair 成分的占比确实提高了。

### 3.3 V6 vs V3 同期 val_chain_normal_mse

| step | V3 | V6 | 差异 |
|---:|---:|---:|---|
| 6000 | 0.000225 | **0.000224** | -0.4% |
| 10000 | 0.000235 | 0.000242 | +3.0% |

Phase I 阶段两者 val_chain_normal_mse 在 rolling-val 噪声范围内一致——符合预期（Phase I 都是 GT velocity 学习）。

### 3.4 建议补入 Reviewer §6 决策矩阵

V6 自身数据可以作为**独立于 V5 归因**的决策维度：

| V6 Phase II indicator | 预期 | 判断 |
|---|---|---|
| V6 +50K pair_frac 仍 > 70% | pair_weight 持续生效 | Phase I 完结 |
| V6 +75K roll_frac 开始上升 | rollout ramp 启动 | Phase II 进入 |
| V6 +100K roll_frac > 30% | transport 接管 | 与 V3 分歧出现 |

---

## 4. 关于 Reviewer 的两个建议

### 4.1 Finding 4 的 V5 full-val sync

同意。将在 operator 层面要求远程 Codex 将以下文件 sync 进 review 树：

```
eval_0427_fullval/v5_best/first_hop_224_val_clip3_eval.json
eval_0427_fullval/v5_step_100000/first_hop_224_val_clip3_eval.json
```

### 4.2 Finding 5 的 V6 full 200K 放行条件

同意 Reviewer 立场：V6 pilot 继续，V6 200K full 放行需要 V5 归因闭环 + V6 Phase II 内生 metric。
