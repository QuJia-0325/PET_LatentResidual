# σ-normalize Rollout Ablation —— 完整说明

> **状态**：待实施。本文档独立自洽，不依赖读者读过 ARCHITECTURE_ANALYSIS_20260501.md §21（但建议先读 §21.3 + §21.8 + §21.11.4）。

---

## §1. 问题陈述

V6 baseline rollout 的 step_loss 是：

$$
\mathcal{L}_{\text{rollout}} = \sum_{j=0}^{K-1} w_j \cdot \mathbb{E}\left[\|z_{j+1}^{\text{pred}} - z_{j+1}^{\text{GT}}\|^2\right]
$$

V6 配置 $w = [0.5, 2.0, 1.5, 1.0]$（hop 索引 j = 0..3，分别对应 D50→D20、D20→D10、D10→D4、D4→NORMAL）。

**问题**：这组 $w_j$ 起的是哪个作用？

| 假设 | 描述 |
|------|------|
| H1 | 命中 multi-hop Grönwall 闭式解 $w_j^* \propto \sum_{k=j+1}^K \beta_k \prod L_i^2$（与 clinical-prior validation objective 对齐） |
| H2 | 只是补偿 step_loss 的自然量级 $(\sigma_j \cdot dt_j)^2$（让所有 hop 的 effective velocity loss 相当） |
| H3 | 两者叠加 |

H1/H2/H3 在 raw step_loss 域**不可分离**。本 ablation 的设计目标：**通过坐标变换明确分离**。

---

## §2. 数学等价性证明

### §2.1 raw → σ-normalize 的恒等关系（preserve_v6_sum 模式）

把 step_loss 写到 normalized velocity space：

$$
\|z_{j+1}^{\text{pred}} - z_{j+1}^{\text{GT}}\|^2 = (\sigma_j \cdot dt_j)^2 \cdot \|\delta v_j\|^2
$$

其中 $\delta v_j = v_j^{\text{pred,norm}} - v_j^{\text{GT,norm}}$ 是 normalized velocity 残差（在 `target_normalize=true` 下，pair velocity loss 直接以 $\|\delta v_j\|^2$ 形式存在）。

记 $\rho_j = (\sigma_j \cdot dt_j)^2$。**关键是 [`pet_lr/rollout_first_hop.py`](../../pet_lr/rollout_first_hop.py) 计算的是加权"平均"，不是加权"和"**：

```python
total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
```

所以两个 condition 完全等价（loss + 梯度都恒等）需要**分子和分母都相等**：

* **raw rollout** (A，σ-norm OFF, 权重 $w$)：

$$
\mathcal{L}_A = \frac{\sum_j w_j \, \rho_j \, \|\delta v_j\|^2}{\sum_j w_j}
$$

* **σ-normalized rollout** (B，σ-norm ON，每步除以 $n_j$，权重 $w'$)：

$$
\mathcal{L}_B = \frac{\sum_j w'_j \cdot (\rho_j / n_j) \cdot \|\delta v_j\|^2}{\sum_j w'_j}
$$

要求 $\mathcal{L}_A \equiv \mathcal{L}_B$，**充分条件**为：

$$
w'_j = w_j \cdot n_j \quad \text{且} \quad \sum_j w'_j = \sum_j w_j
$$

代入第一个条件 $w' = w \cdot n$ 到第二个条件，得 $\sum_j w_j n_j = \sum_j w_j$，即 $n$ 关于 $w$ 加权的均值必须为 1。把这个回代到 $n_j \propto \rho_j$，唯一选择是：

$$
\boxed{\; n_j \;=\; \rho_j \;\Big/\; \frac{\sum_k w_k \, \rho_k}{\sum_k w_k} \;}
$$

这就是 `relative_to: preserve_v6_sum` 的 normalizer 定义（取 $w$ 为 V6 anchor [0.5, 2.0, 1.5, 1.0]）。代入数值：

$$
n = \rho \,/\, 2.3323\!\times\!10^{-4}\;=\;[3.5816,\;0.9303,\;0.5794,\;0.4795]
$$

* **B step_weights** $= w_{V6} \cdot n = [1.7908, 1.8606, 0.8691, 0.4795]$，$\sum = 5.0$（与 V6 完全一致）
* **逐 hop SGD 梯度** = $\sum_j \rho_j \|\delta v_j\|^2 \cdot w_{V6,j} / 5.0$（A 和 B 完全相同，残差 < 1e-19）

→ A 和 B 不仅梯度方向相同，**loss 标量本身也恒等**。这是修正 v2 错误的关键改动：v2 用了 `relative_to: median`，导致 B 的 $\sum w' = 6.624 \ne 5.0 = \sum w$，从而 B 的 effective lambda_roll 是 A 的 0.755×，训练出的模型也不等价。

### §2.2 三组对照配置在 velocity-space 的真实分布

* **A_control**: σ-norm OFF, $w = [0.5, 2.0, 1.5, 1.0]$
* **B_sanity**: σ-norm ON (preserve_v6_sum), $w = [1.7908, 1.8606, 0.8691, 0.4795]$
* **C_uniform**: σ-norm ON (preserve_v6_sum), $w = [1.25, 1.25, 1.25, 1.25]$（rescale uniform 到 $\sum w = 5.0$）
* **D_closed_form**: σ-norm ON (preserve_v6_sum), $w = [3.5795, 0.7910, 0.4149, 0.2146]$（rescale closed-form 到 $\sum w = 5.0$）

**所有 4 个 condition 共享相同的 effective rollout strength**：$\lambda_{\text{eff}} = \lambda_{\text{roll}} / \sum w = 4.0 / 5.0 = 0.8$，单变量为"hop 间 step_weights 形状"。

velocity-space 上每个 condition 对 loss_total 的贡献分布（基础事实：在 σ-norm + preserve_v6_sum 下，hop $j$ 的贡献 $\propto w_j$ 因为 $\rho_j / n_j$ 恒等）：

| | hop 0 (D50→D20) | hop 1 (D20→D10) | hop 2 (D10→D4) | hop 3 (D4→NORMAL) |
|---|---|---|---|---|
| $\rho_j = (\sigma_j dt_j)^2$ | 8.353e-4 | 2.170e-4 | 1.351e-4 | 1.118e-4 |
| **A == B** velocity-space hop fraction | 0.358 | 0.372 | 0.174 | 0.096 |
| **C uniform** velocity-space hop fraction | 0.250 | 0.250 | 0.250 | 0.250 |
| **D closed-form** velocity-space hop fraction | 0.716 | 0.158 | 0.083 | 0.043 |

数值由 [`scripts/verify_normalizers.py`](scripts/verify_normalizers.py) 直接打印（含恒等性自检）。

→ V6 是 **"hop 0 + hop 1 双峰，逐步衰减"** 的中段重分布；闭式解是 **"hop 0 极度集中"** 的 1.000 vs 0.221 形状；uniform 是真正的均匀。三个形状显著区别于 V6 经验权重——本 ablation 因此能干净地区分形状的影响。

**与 v2 的差异**：v2 用 `relative_to: median` 时，C 的 $\sum w = 4.0$，D 的 $\sum w = 22.19$，effective lambda_roll 分别是 A 的 1.25× 和 0.226×。这种"形状不同 + 强度不同"的双变量混淆使得 ablation 无法干净归因。v3 通过 `preserve_v6_sum + Σw=5.0 rescale` 把 effective lambda_roll 锁定为常数。

### §2.3 ablation 真正测试的等价问题

| Condition | velocity-space 分布 | 测试假设 |
|----------|--------------------|---------|
| **A (control)** = raw + V6 [0.5, 2.0, 1.5, 1.0] | [0.358, 0.372, 0.174, 0.096] 中段重 | 现状 |
| **B (sanity)** = σ-norm + [1.79, 1.86, 0.87, 0.48] | 与 A **数学完全等价** | 验证实现正确（loss 标量同时恒等）|
| **C (uniform)** = σ-norm + [1.25, 1.25, 1.25, 1.25] | [0.25, 0.25, 0.25, 0.25] 均匀 | 形状是否重要？|
| **D (closed-form)** = σ-norm + [3.58, 0.79, 0.41, 0.21] | [0.72, 0.16, 0.08, 0.04] hop 0 集中 | 闭式解是否最优？|

→ 三角对比：A vs C（V6 形状贡献），A vs D（闭式解 vs V6），C vs D（uniform vs hop 0 集中）。所有对比共享 effective lambda_roll = 0.8。

---

## §3. 4 个真实陷阱

### §3.1 陷阱 1 — 数值不稳定

直接除以 $\rho_j$（hop 3 = 1.118e-4）会把 step_loss 放大 ~9000×，引发梯度爆炸。

**对策**：使用相对归一化。`relative_to=preserve_v6_sum` 把 normalizer 缩放到 $[0.48, 3.58]$ 区间（median ≈ 1）：

```text
weighted_mean(ρ, w_v6) = 2.3323e-4
n = ρ / 2.3323e-4 = [3.5816, 0.9303, 0.5794, 0.4795]
```

→ step_loss 量级与 baseline 相近，不需要重调 lambda_roll 等超参。同时这个 normalizer 唯一保证 $\sum_j w_{V6,j} n_j = \sum_j w_{V6,j} = 5.0$（见 §2.1），从而 A==B 严格恒等。

### §3.2 陷阱 2 — pair loss 通道不变

代码事实（[`train_first_hop.py:300-310`](../../train_first_hop.py)）：pair velocity loss 已经在 normalized velocity space 训练。`pair_loss_weights = [2.5, 1.0, 1.0, 1.0]` 是 hop 0 显著加重。

**含义**：本 ablation 只切换 rollout 通道一个变量。如果 V6 在 hop 0 的训练信号大部分来自 pair[0]=2.5（已知动机：驱动 pixel forcing gate），那 rollout step_weights 形状的影响可能被 pair 通道掩盖。

**对策**：

1. **承认**这个 ablation 是 "rollout-only 单变量"——论文章节明写：
   > "We isolate the effect of rollout-channel step_weights by holding pair_loss_weights fixed at the V6 default. A full disentanglement between pair[0]=2.5 and rollout step_weights[0] would require a 2-D sweep (rollout × pair) and is left to future work."
2. 如果 main 结果显示 C ≈ A，再计划 follow-up 跑 `pair_loss_weights[0]=1.0` 的额外条件，确认 hop 0 的训练实际由谁主导。

### §3.3 陷阱 3 — 成本与步数选择

V6 baseline 在 200K steps 收敛（best @ ~156K，Phase III 中段）。每条 condition 完整 Phase III ≈ 1 GPU·week，3 conditions × 完整收敛 = 3 GPU·weeks，**不现实**。

**关键警告**：80K steps 仅覆盖 V6 的 Phase II 早期（Phase II = 50K-150K），此时 chain MSE 远未到 best（V6 step 80K ≈ 0.00029 vs best @ 156K ≈ 0.000167）。**80K 不足以判断 ablation conditions 的相对优劣**——可能两条都还在 underfitting。

**修订后的分阶段策略**：

| 阶段 | Condition | steps | 关注指标 | 用途 |
|------|----------|-------|---------|------|
| 1（必跑）| A + B | 50K each | §3.4 多层判据：Tier 1 val_pair_total<1e-4 · Tier 2 val_rollout_total<1% · Tier 3 step_*_raw<1% · Tier 4 chain_*<5% | sanity check |
| 2（必跑）| A + C | **120K each**（Phase II 后段）| Phase II 收敛速度 + 形状 | early signal |
| 3（推荐）| 阶段 2 winner | 200K（完整 Phase III）| best ckpt 对比 | final verdict |
| 4（可选）| D | 120K-200K | hop0 集中 vs 中段重 | 闭式解对照 |

阶段 1+2 = 1 + 6 = **7 GPU·days minimum viable**。阶段 3 视情况 +5 GPU·days。

**关于步数的进一步说明**：

* 80K **不能**作为 final verdict——只能作为 "early indicator"
* 120K 才能给出可信的 "Phase II 后段" 比较（已经过 alpha+lambda ramp 完成阶段）
* 200K 是 V6 原训练长度，给最 fair 的对比
* 如果阶段 2 @ 120K 显示 C ≈ A（< 5% 差异），可以高置信度推断 "step_weights shape 不重要"；如果 |C - A| > 10%，需要走阶段 3 才能下结论

**关于 schedule scaling（v3 新增警告）**：

[`run_ablation.sh`](scripts/run_ablation.sh) 的 `resolve_yaml()` 通过覆写 `max_steps` 来给 sanity vs main 不同步数。但 V6 的 schedule 是用 ratio 表达的（`warmup_ratio: 0.25`, `ramp_ratio: 0.50`），所以缩短 `max_steps` 也会**等比例压缩 schedule**：

| condition | max_steps | warmup 实际 | ramp 实际 | Phase III 起点 |
|----------|----------|-----------|---------|---------------|
| V6 baseline | 200K | 50K | 100K | 150K |
| A_sanity / B | 50K | 12.5K | 25K | 37.5K |
| A_main / C / D | 120K | 30K | 60K | 90K |

→ A_sanity 和 B 共享相同的 50K 压缩 schedule（**A==B 等价性不受影响**，因为 ratio 一致）。A_main 和 C / D 共享相同的 120K 压缩 schedule（在 step 90K 才进入 Phase III，120K 时仅 Phase III 早期 30K）。

**含义**：

* sanity check（A_sanity vs B）@ 50K 是公平的（schedule 完全一致）
* main 比较（A_main vs C）@ 120K 是 "Phase III 早期" 的比较，已比 Phase II 单独跑要 fair——但**仍非完整 Phase III 收敛**
* 如果 main 发现 |C - A| 在 5%-10% 范围、不能下定论，需要把 max_steps 改回 200K 跑一轮（约 5-7 GPU·days），见阶段 3

### §3.4 陷阱 4 — sanity check 的正确判定标准（v3 修订）

数学结论（§2.1）：使用 `preserve_v6_sum` normalizer + B step_weights = $w_{V6} \cdot n$，A 和 B 的 SGD 梯度**逐 hop 完全恒等**，loss 标量也恒等到机器精度。

**与 v2 的差异（重要修订）**：v2 用 `relative_to: median`，导致 B 的分母 $\sum w' = 6.624 \ne \sum w = 5.0$，进而：

* `loss_total_B / loss_total_A = 5.0 / 6.624 ≈ 0.755`（B 系统性低 24.5%）
* 等价地，B 的 effective lambda_roll = 0.755 × A 的 effective lambda_roll
* 训练出的模型不一致：B 的 chain MSE 也与 A 不同（v2 错误地声称"per-hop val_chain MSE 与 normalizer 选择无关"，**这是错的**——chain MSE 测的是训练好的模型，不同 effective lambda_roll 训练出不同模型）

v3 切换到 `preserve_v6_sum` 后，A==B 在每个 SGD step 都严格成立（FP64 残差 ~2.7e-20，已由 [`scripts/verify_normalizers.py`](scripts/verify_normalizers.py) 验证），所以**多个指标**都可以做 sanity check：

> **FP32 vs FP64**：verify_normalizers.py 用 Python `float`（FP64）计算，残差 `~1e-20` 是机器精度。训练运行在 FP32（可能叠加 bf16/AMP）下，normalizer 除法与加权求和的浮点顺序会引入个位起跳的随机偏差，预期：Tier 1–3 位于 ~`1e-6`..`1e-5` 量级，Tier 4 位于 ~`1e-3`..`1e-2`（依赖 EMA / ckpt 抓取点采样）。超过下表阈值即表明实现/配置有 bug，不是训练噪声。

**判定规则（v3 修订版，按严格度从强到弱）**：

1. **最强：val_pair_total**（完全不经 rollout）
   * A 与 B 必须**逐 bit 相同**（< 1e-4 relative 阈值、实际期望 ~1e-6 量级）
   * 这是 RNG / 数据流 / lr_schedule 的基线测试，不一致 = 实现 bug
2. **强：val_rollout_total**
   * A 与 B 应在 < 1% relative 内（FP32 训练随机性叠加，预期 ~1e-5 量级）
   * 偏差 > 1% = sigma_normalize 的 entry 点错误（如 train 入口与 eval 入口不一致）
3. **可比：val_rollout_step_*_raw**（patch 已在 metrics.jsonl 持久化）
   * 这些是**未除以 normalizer** 的原始 rollout step loss，A 与 B 应在 < 1% relative 内
   * 比较这个能直接定位是哪一 hop 的实现出了问题
4. **辅助：val_chain_*_mse**（4 个 chain 指标）
   * 测的是 chain forward 后的 z_NORMAL 重构误差，依赖训练好的模型
   * 因为 A==B 训练出同一模型，这 4 个值应在 < 5% relative 内（FP32 训练噪声 + EMA 抓取采样抖动）

> [`scripts/run_ablation.sh sanity`](scripts/run_ablation.sh) 末尾的 pure-python 多层 comparator 就是这 4 层阈值的机械化实现，逐项报 PASS/FAIL、最后报总 PASS/FAIL。

**判定流程**：

```text
若 val_pair_total 不一致（>1e-4 rel） → 数据/RNG bug，停下来 debug
若 val_pair_total 一致但 val_rollout_total 偏差 > 1%
    → sigma_normalize 入口配置不一致或 normalizer 计算错误
若 val_rollout_total 一致但 val_rollout_step_*_raw 某 hop 偏差 > 1%
    → 该 hop 的 step_normalizers 错位（检查 anchor_step_weights 顺序）
若 上述都通过但 val_chain_* 偏差 > 5%
    → 不应该发生；可能 lr_schedule/EMA/ckpt 采样点隐性差异
```

**为什么 v2 的 "loss_total 不能用作 sanity check" 在 v3 不再适用？**

v2 的限制是 `relative_to: median` 导致分母不等。preserve_v6_sum 修正后，分母恒等，所以 loss_total 恢复成最直接的 sanity 信号。这是 v3 比 v2 更严格的核心改进。

**可能 bug 来源（按发生概率排序）**：

* `step_loss = step_loss / float(normalizer)` 的位置不对（必须在 step_weights 加权之前）
* `rollout_times` / `pair_v_std` 不一致（normalizer 用 `model.pair_v_std`，rollout_times 用 `cfg["data"]["t_map"]`，两者要从同一份 yaml 中取）
* train_first_hop 与 evaluate_first_hop 的 `sigma_normalize` 配置入口不一致（patch 同时改了两处，需确认 yaml 落到两处，且 `anchor_step_weights` 一致）

---

## §4. 解读矩阵

阶段 2 结果（Config C @ **120K** steps val 指标 vs Config A @ **120K** steps）。

> **警告**：在 80K 看到的任何趋势都**不能**作为 final verdict——V6 在 80K 仅达到 chain_normal_mse ≈ 0.00029（距 best 0.000167 还有 1.7× 距离）。下表判定需 120K +。

| C 相对 A 的 chain_normal_mse （@120K）| 论文 narrative | contribution 强度 | 下一步 |
|------|---------------|------------------|-------|
| C ≈ A（差异 < 5%）| "Step weights primarily compensate magnitude; closed-form Grönwall structure is not essential in this L≈1 regime" | 弱（仍可发表 σ²·dt² 自动补偿这一独立 finding）| 不需进一步验证；可选跑 D 验证均一性 |
| C 略弱于 A（5-15%）| "Hand-tuned middle-heavy velocity weighting outperforms uniform" | 中等（可叙述 V6 中段重分布的设计价值）| 推荐阶段 3（200K 完整收敛）验证 |
| C 显著弱（>20%）| "V6's step_weights structure is essential beyond magnitude compensation" | 强（但需 D 验证是不是闭式解最优）| **必跑** D + 阶段 3 |
| **C 比 A 更优** | uniform 反而最优 | V6 经验调参其实有偏，论文 framing 改成 "uniform velocity training is competitive in L≈1 regime" | **补阶段 3** 收敛后重评 |

阶段 4（D）结果（如跑）：

| D 相对 A 的 chain_normal_mse | 含义 |
|------|------|
| D > A（D 更优）| 闭式解 hop 0 集中是最优——但仍有 confound（pair[0]=2.5 与 rollout step_weights[0] 都偏向 hop 0）|
| D ≈ A | 闭式解与 V6 经验等价 |
| D < A | 闭式解 hop 0 集中过头——V6 中段重更稳健 |

> **D 的 L≈1 假设警告（v3 新增）**：闭式解 Grönwall 系数推导（§18.3.3）假设 hop-wise Lipschitz 系数 $L_j \approx 1$。这一条件只在 Phase II 之后（V6 step ≥ 60K，本 ablation @120K 时已进入 Phase III 早期）才成立。所以 D 的判定**主要看 step 60K 之后**的 chain MSE。如果 D 在 step < 60K 期间表现明显差于 A，但 step 60K 后追平，**这是预期行为**，不是 bug。建议在 metrics.jsonl 中关注 `step ≥ 60000` 区间的 best ckpt。

---

## §5. 最终 narrative 决策树

```text
                  阶段 2 完成 (A vs C)
                        │
              ┌─────────┴─────────┐
        C ≈ A (≤5%)            C 弱于 A
              │                     │
   contribution = "magnitude    contribution = "shape
   compensation suffices"        matters; v6 vector
              │                  outperforms uniform"
              │                     │
   论文 framing:                论文 framing: + 跑 D
   - σ²·dt² as design         (闭式解 vs V6)
     principle                     │
   - L≈1 regime caveat        D > A (闭式 best)
   - 不强推 Grönwall              │
                              "closed-form Gronwall
                               validated, V6 是 hand-
                               tuned approximation"
```

---

## §6. 对 ARCHITECTURE_ANALYSIS_20260501.md 的回填计划

| 章节 | 现状 | ablation 完成后回填 |
|------|------|-------------------|
| §0 状态表 | V6 emergent 命中 Grönwall（已降级）| 加 "validated by sigma-normalize ablation: contribution=<weak/medium/strong>" |
| §18.3.4 | 给出 V6 vs 闭式解 raw 对比表 + emergent 命中降级 | 增加 velocity-space 对比表（§2.2）+ 三角对比结果 |
| §19.5 论文 contribution 段 | DRAFT，需重写 | 用 §4 解读矩阵的对应 narrative 行重写 |
| §21.3 σ²·dt² 替代解释 | 假设 | "verified/refuted by sigma-normalize ablation"|
| §21.9 行动 1 | 必做 | 状态改为 "completed: results = ..." |

---

## §7. patch 文件清单与功能

| Patch | 修改 | 行数 | 风险 |
|-------|------|------|------|
| `patches/rollout_first_hop.py.patch` | `rollout_multistep_losses_first_hop` 加 `step_normalizers` 参数 + `step_losses_raw` 输出 | ~10 行新增 | 低（默认 None 时行为不变）|
| `patches/train_first_hop.py.patch` | 加 `_compute_sigma_dt_normalizers` helper + 在 `compute_rollout_losses` 与 val rollout 入口注入 | ~30 行新增 | 低（默认 sigma_normalize.enabled=false 时行为不变）|

**无破坏性改动**：所有现存 yaml 配置在不开 `sigma_normalize` 时行为完全不变。

详见 `patches/` 目录的 unified diff 文件。
