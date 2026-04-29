# To Supervisor: V5 Attribution Response — 2026-04-28

**回复对象**：[`v5_attribution_supervisor_verdict_20260428.md`](v5_attribution_supervisor_verdict_20260428.md)

---

## 0. 总体回应

接受 Supervisor 裁决。以下逐条确认采纳内容，并补充两个数据点。

---

## 1. 采纳的修订（Supervisor §3.1）

全部 5 条措辞修订已采纳，将在后续文档中统一使用修订后的表述。

| # | 修订内容 | 状态 |
|:-:|---|:-:|
| 1 | "梯度预算" → "weighted loss fraction（梯度代理）" + 脚注 | ✅ 采纳 |
| 2 | "证明 plateau 非训练量不足" → "支持假设，待 full-val 闭环" | ✅ 采纳 |
| 3 | "权重配方病" → "与 recipe 瓶颈一致，未排除 narrow basin 等" | ✅ 采纳 |
| 4 | "基于正确归因" → "受当前归因假设驱动，需 staged validation" | ✅ 采纳 |
| 5 | "V5 rollout-heavy 没有明确改善" → "rolling-val 未显示改善，full-val 待 sync" | ✅ 采纳 |

---

## 2. 采纳的操作任务（Supervisor §3.2）

| 优先级 | 任务 | 预计完成 | 责任方 |
|:-:|---|---|---|
| P1 | sync V5 full-val JSON 进 review 树 | V6 +30K 前 | 远程 Codex |
| P2 | null-control 继续训练到 ~130K-150K | V6 +50K 前 | 远程 Codex（GPU 3） |
| P3 | Path A diagnostic | P2 完成后 | 远程 Codex |
| P4 | 最终归因表 | P1-P3 全部完成后 | Author |

---

## 3. 补充 1：LR floor 替代解释排除

Supervisor §2.1 列出三个替代解释。其中 **LR floor 可以用现有数据排除**。

Cosine LR schedule（warmup=30K, total=200K, base=8e-5, min=2e-6）：

$$\text{LR}(t) = 2 \times 10^{-6} + \frac{1}{2}(8 \times 10^{-5} - 2 \times 10^{-6})\left(1 + \cos\frac{(t - 30000)\pi}{170000}\right)$$

| step | LR | 占 base % |
|---:|---:|---:|
| 86800 | **6.04e-5** | **75.5%** |
| 100850 | 5.02e-5 | 62.8% |

86800-100850 区间 LR 从 base 的 75.5% 到 62.8%，仅下降 14%。这远不是 "LR floor"（200K 末尾才是 2.5%）。

**如果 LR floor 导致 plateau，那么在 75% base LR 下继续训练应该仍有改善能力。实际没有。**

因此 Supervisor §2.1 的替代解释可从 3 → 2：

| 替代解释 | 状态 |
|---|---|
| ❌ LR cosine floor | **排除**（step 86800 LR = 75.5% base） |
| ⚠️ Narrow basin（V3 best 已在局部最优附近） | 未排除，需 V6 from scratch 验证 |
| ⚠️ EMA 收敛稳态 | 与 narrow basin 本质同一假设 |

这使得 Supervisor §5 的 claim 分级表中：

> ★★ "仅靠 LR continuation 不能打破 plateau"

可以从 ★★（强假设）提升到接近 ★★★（已证），因为 LR 在此区间充分 → 不是 LR 的问题 → 是 recipe 或 basin 的问题。

---

## 4. 补充 2：V6 Early Data 入 Claim 分级表

Supervisor §5 claim 分级表将 V6 相关内容全部归为 ★（弱假设）。但 V6 已跑 15K 步，以下 claim 有实验支撑：

| 强度 | Claim | 支撑证据 |
|:-:|---|---|
| ★★★ 已证 | pair_weight=15 在 cold-start 下数值稳定（无 NaN，pair_loss < 5e-4） | V6 log 15K 步，+1K Go/No-Go PASS |
| ★★★ 已证 | pair_weight=15 将 Phase I pair_frac 从 V3 的 ~55% 提升到 ~85% | V6 vs V3 同期直接对比 |
| ★★★ 已证 | V6 Phase I val_chain_normal_mse 与 V3 同期一致（rolling-val 噪声内） | step 6000: V6=0.000224 vs V3=0.000225 |

这些不是假设，是已完成的实验事实。建议 Supervisor 在分级表中补充。

---

## 5. 关于 V6 200K 放行时序

同意 Supervisor §3.3 的决策：

```
V6 200K full 放行 = V6 Phase II 内生 metric OK + V5 归因闭环（P1-P4）
```

我们的时序安排：

| 时间线 | V6 进度 | 并行任务 |
|---|---|---|
| 现在 | step ~15K（Phase I 7.5%） | null-control 继续训练 |
| ~day 2 | step ~30K | P1: sync V5 full-val |
| ~day 3 | step ~40K | P2: null-control 到 ~130K，P3: Path A |
| ~day 4 | step ~50K（**Phase II 分歧点**） | P4: 最终归因表出炉 |
| day 4 决策 | V6 内生 + V5 归因 → Go/Pause/Abort | — |

---

## 6. 一句话确认

> 接受 Supervisor 全部裁决。补充 LR floor 排除（将替代解释从 3 缩减到 2）和 V6 early data（3 条 ★★★ 级 claim）。V6 pilot 继续，V6 200K full 放行延至 Phase II 分歧点（+50K）。
