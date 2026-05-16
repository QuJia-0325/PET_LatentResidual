# Step-Weights 理论参考（汇总）— 20260516

> 这是把项目内分散的 step_weights 数学推导整合到一处的引用文档。所有原始推导出处都在 [review/plan/ARCHITECTURE_ANALYSIS_20260501.md](../plan/ARCHITECTURE_ANALYSIS_20260501.md)，本文件不发明新结论，只做汇总与索引。
>
> 触发问题：在 [PLANF_FINAL_ANALYSIS_20260516.md §7](./PLANF_FINAL_ANALYSIS_20260516.md) 的 next-step 讨论中，"加权 NORMAL"看起来像经验调参。但 review 历史显示：**多 hop 加权目标下的最优 step_weights 已有闭式解**；V7 的 `step_weights` 就是这个闭式解在 `β=[0.5, 0.45, 0.9, 1.5]` + `L_i≈1` 下的实例化。因此 V9 的修改不是经验扰动，而是同一闭式解在 `β_NORMAL=2.5` 处的重新实例化。

---

## 1. 框架

Chain 是离散 ODE：

$$
z_{k+1} = z_k + \Delta t_k \cdot v_\theta(z_k, t_k, t_{k+1}, k) \equiv F_k(z_k), \qquad k=0,\dots,K-1
$$

单步误差 $\epsilon_k = F_k(z_k) - F_k^*(z_k)$，累积误差 $\delta_k = z_k - z_k^*$。Lipschitz 假设 $\|F_k(a)-F_k(b)\|\le L_k\|a-b\|$，其中 $L_k = 1 + \Delta t_k \cdot \mathrm{Lip}(v_\theta)$。

应用离散 Grönwall 不等式（[ARCHITECTURE_ANALYSIS §16.4.1](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）：

$$
\|\delta_K\|^2 \;\le\; \sum_{k=0}^{K-1}\left(\prod_{j=k+1}^{K-1} L_j^2\right) \|\epsilon_k\|^2 \tag{Grönwall}
$$

---

## 2. 两套闭式解

### 2.1 纯末端目标 — `min E‖δ_K‖²` （[§16.4.2](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）

由 Cauchy–Schwarz 等号条件：

$$
\boxed{\quad w_k^{*,\text{endpoint}} \;\propto\; \prod_{j=k+1}^{K-1} L_j^2 \quad}
$$

定性：早期 hop 重，末端 hop 轻。

### 2.2 多 hop 加权目标 — `min E Σ_k β_k ‖δ_k‖²` （[§18.3.3](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）

把 (Grönwall) 代入加权目标得：

$$
\boxed{\quad w_j^{*,\text{multi-hop}} \;\propto\; \sum_{k=j+1}^{K} \beta_k \prod_{i=j+1}^{k-1} L_i^2 \quad}
$$

**极限退化**：
- 当 $\beta_K=1,\ \beta_{<K}=0$（纯末端）→ 退化为 §2.1 的 Grönwall。
- 当 $\beta$ 均匀且 $L_i\equiv1$ → $w_j^* \propto K-j$ （线性递减）。

**对我们项目的解读**：[V6/V7 best_metric_terms](../0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml) 选的就是 $\sum_k \beta_k \|\delta_k\|^2$ 形式的多 hop 加权 selector，因此 (§2.2) 才是正确闭式解，**不是** (§2.1)。

---

## 3. 量级补偿项（被很多人忽略的部分）

来自 [§18.3.4 / §21.3](../plan/ARCHITECTURE_ANALYSIS_20260501.md)：rollout step loss 的自然量级是 $(σ_j \cdot \Delta t_j)^2$。我们的数值是

| hop | σ_j | dt_j | (σ_j·dt_j)² | 相对值 |
|---|---|---|---|---|
| 0 | 0.00963 | 3 | 8.35e-4 | **7.45×** |
| 1 | 0.00295 | 5 | 2.17e-4 | 1.94× |
| 2 | 0.00078 | 15 | 1.35e-4 | 1.20× |
| 3 | 0.000141 | 75 | 1.12e-4 | 1.0× |

跨 4 hop 差 7.5×。当 $L_i \approx 1$ 时 Grönwall 项 $\prod L_i^2 \approx 1$，§2.2 闭式解中 $\sum \beta_k \prod L_i^2 \approx \sum_{k>j} \beta_k = \Phi_j$ 退化为 β 的反向累计，此时 step_weights 的真正作用是：

$$
w_j \;\approx\; \underbrace{\Phi_j}_{\text{β-cascade}} \cdot \underbrace{\frac{1}{(\sigma_j\Delta t_j)^2}}_{\text{magnitude compensation}}
$$

这正是 V7 config header 中的实际计算（"w_raw ∝ Φ_j / (σ_j·dt_j)²"）。**V7 的 step_weights 是这两项乘积的归一化结果，不是经验调参。**

---

## 4. 各 V 系列实验在闭式解中的位置

| 版本 | β = `best_metric_terms` weights | step_weights 出处 | 闭式解？ |
|---|---|---|---|
| V6 | [0.5, 0.45, 0.9, 1.5] | [0.5, 2.0, 1.5, 1.0]（经验，无 σ·dt 补偿） | hop2/3 ≈ 闭式解 4-11%，hop0 反向 4× |
| V7 | [0.5, 0.45, 0.9, 1.5] | [0.6106, 2.0, 2.7041, 2.0423]（§2.2 闭式解 + σ·dt 补偿，hop1=2.0 锚定） | ✅ 严格闭式 |
| V8 | [0.5, 0.45, 0.9, 1.5] | 同 V7（V8 只关掉 image_aux） | ✅ 严格闭式 |
| V6_NOISE | 同 V6 | 同 V6 | seed-only 控制 |
| V9（draft） | **[0.5, 0.45, 0.9, 2.5]** | [0.5870, 2.0, 2.8333, 2.5173]（同闭式解，β_NORMAL=2.5） | ✅ 严格闭式（同一公式，不同 β） |

**结论**：V9 不是经验扰动，而是在同一闭式解框架内改变 β_NORMAL。改 β 的合法性来自 selector 本身就是 β 的函数 —— 改 β 是改"要优化什么"，不是改"怎么权重它"。

---

## 5. 真正未做的实验

§2.2 闭式解的两个未验证假设：

### 5.1 假设 A：`L_i ≈ 1`

V6/V7 当前都基于 `L_i≈1` 的隐含估计（来自 §17.1 联合权重比反解，得 $L_j \in [0.96, 1.13]$）。这是个**反解结果**而不是直接测量。如果 V7 收敛后真实 $L_j$ 显著偏离 1，则 §2.2 闭式解的 step_weights 会变形，而我们用的 σ·dt 补偿可能严重次优。

[§16.4.4 / §19.4](../plan/ARCHITECTURE_ANALYSIS_20260501.md) 给出 10 分钟可测的脚本（见本文件 §6）。这是 review 历史中明确列入 `§19.6 行动 1` 的待办，**至今未执行**。

### 5.2 假设 B：rollout loss 应在 σ-normalized 空间里算

[§20.4 #1 / §20.6 行动 7 / §21.3](../plan/ARCHITECTURE_ANALYSIS_20260501.md) 明确说：rollout step_loss 当前对 raw $z$ 算 MSE，所以 σ·dt 补偿被吸进了 step_weights。如果在 σ-normalized 空间里算 rollout loss，step_weights 应该退回到 §2.2 闭式解中的纯 Φ_j 形式（不需要 σ·dt 补偿），此时 V6/V7 的 step_weights 应该重新收敛到不同形状。

这才是真正 "critical ablation"，比改 β 更值得做。

---

## 6. Lipschitz 测量脚本（10 分钟可跑，已写好骨架）

直接从 [§16.4.4 / §19.4](../plan/ARCHITECTURE_ANALYSIS_20260501.md) 复制过来，我把它整理为可执行文件 [tools/estimate_per_hop_lipschitz.py](./STEP_WEIGHTS_THEORY_REFERENCE_lipschitz_skeleton.py)（同目录），后续真正实现时只需把 dataset/model 接入。骨架：

```python
@torch.no_grad()
def estimate_per_hop_lipschitz(model, val_loader, n_samples=64, eps=1e-3):
    L_per_hop = []
    for k in range(4):
        z = sample_z_at_hop(val_loader, hop=k, n=n_samples)
        delta = torch.randn_like(z) * eps
        F0 = model.predict_latent_step(z,         t_src[k], t_dst[k], hop_idx=k)["z_pred"]
        F1 = model.predict_latent_step(z + delta, t_src[k], t_dst[k], hop_idx=k)["z_pred"]
        L_k = ((F1 - F0).norm(dim=(1,2,3)) / delta.norm(dim=(1,2,3))).mean().item()
        L_per_hop.append(L_k)
    return L_per_hop
```

判定（来自 [§19.4 表](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）：

| 偏差 | 行动 | 论文叙述 |
|---|---|---|
| 闭式解与 V7 step_weights 对齐 < 20% | 不改任何配置，直接 claim emergent 命中 | "We show V7's step_weights emergently satisfy a closed-form solution derived from multi-hop Grönwall propagation under clinical-prior β." |
| 偏差 20-40% | 跑 1 组用真实 L_j 重算的 step_weights | "We further refine the weights using the analytic closed-form with measured L_j, achieving X% improvement." |
| 偏差 > 40% | chain dynamics 中 L_j 显著大于 1 → §2.2 真起作用 → 严格按测得 L_j 重新派生 | 必须做 σ-normalize ablation 解耦量级补偿与 Grönwall 加权 |

预测（基于 §17.1 反解）：偏差 < 25%。

---

## 7. 我们到底"经验"了多少

| 配置项 | 推导来源 | "经验"成分 |
|---|---|---|
| `best_metric` = val_multi_objective | 工程选择（chain selector） | low |
| β = [0.5, 0.45, 0.9, 1.5] | 临床先验（"NORMAL > D4 > D20 > D10"） | medium — 临床优先，但 1.5/0.9/0.5/0.45 的相对幅度是 heuristic |
| step_weights 公式 `Φ_j/(σdt)²` | §2.2 闭式解 + §3 量级补偿 | **none**（严格闭式） |
| `L_i≈1` 假设 | §17.1 反解 + 未实测 | **medium — 待 §6 验证** |
| rollout loss raw vs σ-normalized | §20.4 #1 未做 | **medium — 待 §5.2 验证** |
| `pair_loss_weights[0]=2.5` | gate activation 设计（§18.3.4 修正 2） | low（架构相关） |

**真正可以减少 "经验" 的下一步**：执行 §6 的 Lipschitz 测量。这是无需训练、10 分钟可跑、能直接判定 V7 是否已经命中闭式解的实验。

---

## 8. 与 paper 相关文献的关系

- **MAP-Diff** 的 $w(t)=(1-t/T)^p$（[§15 / §16.1 / §16.3](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）：经验启发，无严格证明，reviewer 会质疑形式选择。
- **Min-SNR-γ** ([§16.3](../plan/ARCHITECTURE_ANALYSIS_20260501.md))：Pareto-optimal 但只针对 single-step diffusion；我们的 `target_normalize=true` 在 pair loss 上是它的精神等价物。
- **EDM** ([§16.3](../plan/ARCHITECTURE_ANALYSIS_20260501.md))：噪声调度优化，与 step_weights 是正交的。
- **离散 Grönwall**（数值分析标准结果）：本项目使用的理论起点；推广到 multi-hop 加权目标得 §2.2 是我们项目自己的贡献（见 [§19.5 论文 contribution 升级](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）。

---

## 9. TL;DR

1. **不是经验**：V7 的 step_weights 已经是多 hop Grönwall 闭式解（§2.2）+ 量级补偿（§3）在 `L_i≈1` 下的实例化，公式见 V7 config header 与本文 §2.2 / §3。
2. **V9 也不是经验扰动**：同一公式，只换 β_NORMAL=2.5；step_weights 是 closed-form 重派生，不是手调。
3. **真正未做的"理论性"实验**两个：
   - (A) Lipschitz 测量（§6）：10 分钟脚本，直接验证 `L_i≈1` 假设。**首推**。
   - (B) σ-normalize rollout loss（§5.2 / [§20.6 行动 7](../plan/ARCHITECTURE_ANALYSIS_20260501.md)）：解耦量级补偿与 Grönwall 加权。1 次完整训练。
4. **V9 的实验价值不变**：β-axis 扰动有它自己的价值（验证多 hop selector 的优化目标对 weight shape 的敏感性），但若与 (A) 并行，可以同时回答两个独立问题。
