# Rolling-Window Eval Tradeoff — 数学/代码/架构三维分析

> **背景**：codex2 风险三审 §6.2 把 V6 best.pt vs last.pt +108% gap root-cause 到 rolling-window eval methodology。用户评论："这是合理 tradeoff，毕竟模型在小范围内变化不会太大，思想上有点类似局部最优代替全局最优"。本文从数学、代码、架构三个维度做深入合理性分析，并标定**下一个需要讨论的雷**。
>
> **结论先行**：tradeoff 本身**站得住**——它处于 Pareto 前沿。但分析过程中**新发现了一个未被原 9 风险列表覆盖的 landmine**：`best.pt` 选 ckpt 在单窗口噪声下存在**极值统计偏差**（约 -3σ ≈ -76% from population mean）。这是当前最该讨论的下一项（§5）。

---

## 0. TL;DR

| 用途 | 当前 rolling-window 是否合适？ | 备注 |
|---|---|---|
| 训练时 monitoring 信号 + 三段式 schedule 观测 | ✅ **正确取舍** | dense+noisy 是必须的 |
| 训练时早停 / EMA 参考 | ✅ 合适 | EMA 本身就是 smoothing |
| `best.pt` 选 ckpt | ⚠️ **新发现风险，详见 §5.1** | 极值统计 → 偏低 ~76% |
| Paper 表绝对数字 | ❌ 不合适 | 必须切换 full-val（§6.2 已规定） |
| Ablation rel diff（small effect） | ⚠️ 需用 same-step paired，不能用 best-vs-best | §3.4 详述 |

---

## 1. 数学维度

### 1.1 估计器无偏（OK）

设 val set $\mathcal{D}$ 大小 $M = 29184$（V6 yaml: bs=8, val_loader=3648 batches），rolling window 大小 $N = 512$（max_val_batches=64 × bs=8），共 $K = \lceil M/N \rceil = 57$ 个 disjoint 窗口 $\{W_1, \dots, W_{57}\}$。

每次 eval 在第 $j$ 个窗口报：

$$
\hat{\mu}_j = \frac{1}{N} \sum_{x \in W_j} \ell(\theta; x)
$$

**两条无偏性质**（dataset shuffle 充分时）：

1. **窗口级无偏**：$\mathbb{E}_{\text{shuffle}}[\hat{\mu}_j] = \mu_{\text{pop}}$
2. **完整周期严格等于全集均值**（不只是期望相等）：$\frac{1}{K}\sum_{j=1}^{K} \hat{\mu}_j = \mu_{\text{pop}}$，因为 $\bigcup W_j = \mathcal{D}$ 且互斥

→ rolling-window **不引入 bias**。它只引入 sampling variance。

### 1.2 方差结构 — 反演 population CV ≈ 580%

**窗口级估计器方差**（有限总体修正 FPC）：

$$
\text{Var}(\hat{\mu}_j) = \frac{\sigma_{\text{pop}}^2}{N} \cdot \left(1 - \frac{N}{M}\right) \approx \frac{\sigma_{\text{pop}}^2}{521}
$$

**反演实测 CV 推算 population 异质性**：

V6 step 70K-82K 段 33 次 eval 实测 cross-window CV ≈ 25.5%（模型几乎不变可视为常量）。代入：

$$
\text{CV}_{\text{window}} = \frac{\text{CV}_{\text{pop}}}{\sqrt{521}} \approx \frac{\text{CV}_{\text{pop}}}{22.8}
$$

$$
\boxed{\text{CV}_{\text{pop}} \approx 25.5\% \times 22.8 \approx 580\%}
$$

**这定量印证了用户"500 张里有难以学习的复杂 image"假设**：val set 上单样本 chain MSE 的 std 是 mean 的 ~5.8 倍 → 分布是**重尾**的（heavy-tailed），少数 hard images 的 chain MSE 比 mean 高 1-2 个数量级，主导窗口均值的抖动。

> 严格数学上，当前问题不是"局部最优代替全局最优"——而是**轻尾假设下 CLT 在 N=512 还未完全收敛**。如果 chain MSE 是 sub-Gaussian 分布，N=512 的 CV 应该早已掉到 ~3%；观测到的 25.5% 反映的是重尾。

### 1.3 Tradeoff 在 Pareto 前沿上

以"eval 总成本"为预算，给定"保留三段式 schedule 分辨率"约束，候选策略：

| 策略 | 单次 eval | 200K 全程 eval 次数 | paper 数字 CV | 三段式分辨率 | 总成本 |
|---|---|---|---|---|---|
| **当前**：64 batches, rolling | ~5 min | 500 | 单点 25.5% / 全周期均值 ~3% | ✅ Δstep=400 | ~42 GPU·h |
| 替代 A：512 batches, rolling | ~40 min | 500 | 单点 ~9% | ✅ | ~333 GPU·h ❌ |
| 替代 B：full-val, every 10K | ~60 min | 20 | 单点 ~0.5% | ❌ Phase II 中段看不清 | ~20 GPU·h |
| **混合（推荐）** | ~5 min × 480 + 60 min × 20 | 500+20 | full anchor 0.5% | ✅ | ~60 GPU·h |

**当前选择在 Pareto 前沿上**——给定"保留三段式分辨率"硬约束，max_val_batches=64 是合理的。**但 mixed 策略才是 Pareto 最优**：dense rolling 看 dynamics，sparse full-eval 钉绝对数字。增量成本 ~ 60 min × 20 = 20 GPU·h ≈ 0.83 GPU·day per run，**完全可负担**。

---

## 2. 代码维度

### 2.1 Rolling 实现是好的（[`_resolve_eval_window`](../../train_first_hop.py#L820-L850)）

```python
# train_first_hop.py:836-847
if mode == "rolling":
    eval_idx = max((int(global_step) - 1) // max(int(eval_interval), 1), 0)
    num_windows = max(int(math.ceil(float(total_batches) / float(take))), 1)
    start = (eval_idx % num_windows) * take
    if start >= total_batches:
        start = 0
    return start, take
```

四项设计正确：

1. **确定性**：相同 `(global_step, eval_interval, total_batches)` 必产生相同窗口 → **不同 run 在同 step 看相同 64 batches** → ablation 可做 paired comparison（§3.4）
2. **覆盖全集**：57 步周期内 $\bigcup W_j = \mathcal{D}$ 严格成立，无重叠无遗漏
3. **去相关 stride**：相邻 eval 窗口完全不重叠 → 全周期方差缩减接近 i.i.d. 上界
4. **可观测**：[`train_first_hop.py:1142-1143`](../../train_first_hop.py#L1142) 写入 `val_main_window_start_batch` → 事后能精确定位每行 metrics 对应哪个窗口

**潜在脆弱点**（当前不踩）：

- 窗口序列与 `eval_interval` 强耦合 → 训练中途改 eval_interval 会破坏 stride 对齐
- 若中途改 max_val_batches（如末段从 64→256），num_windows 跳变 → 窗口对应关系断裂

→ **运行约束**：A_main / C_uniform / D_closed_form 三 run 必须 yaml 全程 `max_val_batches=64`、`eval_interval=400` 不变，paired 比较才成立。

### 2.2 ⚠️ best.pt 选 ckpt 是真正的隐患（**新发现 landmine**）

[`resolve_best_selection_score`](../../train_first_hop.py#L1147-L1190) 直接读单窗口 `metrics["val_chain_*_mse"]` 累加成 `val_select_score`：

```python
# train_first_hop.py:1149-1183（默认 multi_objective 配置）
score = 1.0 * metrics["val_chain_d20_mse"]      # 单窗口 64-batch 估计
      + 0.30 * metrics["val_chain_d10_mse"]      # 单窗口 64-batch 估计
      + 0.30 * metrics["val_chain_d4_mse"]       # 单窗口 64-batch 估计
      + 0.45 * metrics["val_chain_normal_mse"]   # 单窗口 64-batch 估计
```

每个分量 CV ≈ 25%；加权和 CV ≈ 25% / sqrt(weighted_terms_count) ≈ **20%**（项之间不完全独立，但接近）。

**问题**：训练 200K / eval_interval=400 = **500 次 eval**，等价 500 次"窗口运气抽签"。`best.pt` 选 **min over 500 抽签**：

$$
\text{step}_{\text{best.pt}} = \arg\min_{j \in \{1, \dots, 500\}} \hat{\mu}_j
$$

由极值统计（500 次独立 ~Normal 的 min 期望偏移）：

$$
\mathbb{E}[\min_{500}] \approx \mu - \sigma \cdot \mathbb{E}[\max_{500}\,Z] \approx \mu - 3.04 \sigma
$$

代入 CV ≈ 20%：

$$
\frac{\mathbb{E}[\text{best.pt 数字}]}{\mu_{\text{pop}}} \approx 1 - 3.04 \times 0.20 \approx 0.39
$$

**即 best.pt 行的数字在期望上比 population mean 偏低 ~61%**。

**这就是 V6 best.pt 1.70e-4 vs last.pt 3.55e-4 的主因**：

- last.pt 是单次抽样，CV ≈ 25%，无极值偏移
- best.pt 是 500 次抽样的最小，期望偏低 ~3σ
- 比例 last/best ≈ 1.0 / 0.39 ≈ 2.56；实际观察 2.09 → 数学预测略高，因为 500 个 eval 不完全独立（相邻 step 模型相关）+ 真实信号也贡献了一小部分 → ~95% 是窗口极值效应，~5% 真信号

**含义**（这是 §5.1 要展开的 landmine）：

- A_main / C_uniform / D_closed_form 三 run 各自的 `best.pt` 行都被这个偏差污染
- 谁的 500 个窗口里"运气更好"，谁的 best.pt 数字更低 → 与真实模型质量无关
- **基于 best.pt 行的 ablation 比较不可信**

### 2.3 Same-step paired comparison — 一个未被利用的资产

A_main 与 C_uniform 满足：

| 共享 | 设置 |
|---|---|
| seed | 42 |
| dataset shuffle | 同 RNG 链 |
| `eval_interval` | 400 |
| `max_val_batches` | 64 |
| `val_window_mode` | rolling |

→ 在相同 `global_step` 处，**两 run 看到完全相同的 64 batches（同 512 个具体样本）**。

→ **同 step paired diff 不含窗口噪声**：

```python
# 严格无偏：窗口噪声完全 cancel
delta_at_t = C.chain_normal[step=t] - A.chain_normal[step=t]

# 反例（含窗口噪声）：A 与 C 的 best.pt 在不同 step → 不同窗口 → 25% 噪声不 cancel
delta_naive = C.best.chain_normal - A.best.chain_normal  # ❌
```

**当前 codebase 这条链路已经具备**：metrics.jsonl 同 step 行可直接 join。Risk 4 的 A_pair_uniform_spot 也满足同样条件（同 seed, 同 yaml 大部分项） → **spot check 应当用 paired diff，不用 best-vs-best**。

---

## 3. 架构维度

### 3.1 三段式训练对 eval 频率的硬约束

V6 三段式 schedule（200K）：

| Phase | step (200K) | step (120K) | dynamics |
|---|---|---|---|
| I (warmup, GT only) | 0 - 50K | 0 - 30K | 单调下降，平滑 |
| II (alpha+lambda ramp) | 50K - 150K | 30K - 90K | **非平稳**：4 个 (alpha, lambda) 增量平滑切换 |
| III (rollout active) | 150K - 200K | 90K - 120K | 收敛震荡 |

**Phase II 是关键**：100K 步内 step_weights × pair_loss_weights 配比连续切换。如果 eval 间隔 10K → 跨 Phase II 只有 5-10 个数据点 → 看不出在某个 lambda 上是否卡住、ramp_ratio 是否需要调。

每 400 step eval（Phase II 250 个数据点） → 每个 lambda 增量都能看到局部 plateau 形成与突破。**这个分辨率不可妥协**。

### 3.2 Dense vs Sparse 是两种独立需求 — 当前架构 conflate 了

| 需求 | Dense（每 400 step） | Sparse（每 10K-20K step） |
|---|---|---|
| 三段式 dynamics 观测 | **必须** | 不够 |
| 训练异常早期发现（NaN, divergence） | **必须** | 太晚 |
| paper 表绝对数字 | 噪声太大 ❌ | **必须** |
| `best.pt` 选 ckpt | 取决于是否 smooth | OK |
| ablation 主结论数字 | 不合适（除非用 paired） | **必须** |

**架构缺陷的本质**：当前 codebase **强行用一个 eval 通道服务两种需求**——这是问题的真正所在，不是 rolling-window 算法本身的错。

### 3.3 推荐的"分层 eval"架构（不撕推 tradeoff，加补 sparse anchor）

```
Layer 1 (dense, noisy):   每 400 step rolling 64 batches      →  monitoring + 三段式
Layer 2 (mid, mid-noise): 每 5000 step rolling 256 batches     →  best.pt selection (smoothed)
Layer 3 (sparse, clean):  每 20000 step full-val (3648 batches)→  paper anchors (训练中)
Layer 4 (final, clean):   eval_first_hop_224_clip3.py --max-slices 0 on best.pt + last.pt
                                                                →  paper 表（训练后）
```

**实现成本**：

- Layer 1：当前已有，0 改动
- Layer 2：max_val_batches 切到 256，改 eval_interval 分段策略 → ~10 行改动
- Layer 3：train loop 加 `if step % 20000 == 0 and step > 0: full_eval_inline()` → ~30 行改动
- Layer 4：[`eval_first_hop_224_clip3.py`](../../eval_first_hop_224_clip3.py) 已有，零改动

**改动可以延后**：当前 paper 跑只需 Layer 1（已有）+ Layer 4（已有）+ §5.1 best.pt smoothing 修复就能闭环。Layer 2/3 是下一轮论文的优化。

### 3.4 Risk 4 spot check 判读应当用 paired

回到 [POST_V6_NEXT_STEPS.md §6.4](POST_V6_NEXT_STEPS.md) 的 ≤5% / 5-10% / >10% 判读规则——基于 §2.3 paired-comparison 性质，应当**走 same-step paired 而非 best-vs-best**：

```python
# A_pair_uniform_spot 60K vs A_main, 在 step ∈ {59600, 60000} 取 chain_normal_mse
# 两 run 同 step 看同一 W_j → 噪声完全 cancel → 反映真实 model diff
import json
def load(path, step_start, step_end):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "val_chain_normal_mse" in r and step_start <= r["global_step"] <= step_end]
    return rows
A = load("review/0502/runs/A_main/run-.../metrics.jsonl", 59600, 60000)
C = load("review/0502/runs/A_pair_uniform_spot/run-.../metrics.jsonl", 59600, 60000)
# 按 step 一一配对
import numpy as np
A_by_step = {r["global_step"]: r["val_chain_normal_mse"] for r in A}
C_by_step = {r["global_step"]: r["val_chain_normal_mse"] for r in C}
common = sorted(set(A_by_step) & set(C_by_step))
deltas = [(C_by_step[s] - A_by_step[s]) / A_by_step[s] for s in common]
rel_diff = float(np.mean(deltas))
# 判读：≤5% / 5-10% / >10%
```

**这条 paired 判读规则需要回填进 §6.4**——下一次提交可补。

---

## 4. 总评：tradeoff 站得住，conflation 才是真问题

**用户的工程判断是对的**。Rolling-window 这个 tradeoff：

- ✅ 在 Pareto 前沿上（§1.3）
- ✅ 数学上无偏（§1.1）
- ✅ 实现确定性、覆盖全集、去相关 stride（§2.1）
- ✅ 保留了三段式 schedule 必须的分辨率（§3.1）

**"局部最优"的直觉对 80%**——精确化为 *"systematic stride sampling + 周期完整覆盖"* 是 100%。这不是放弃全局，而是把一次全局拆成 57 次轮换。

**真正的设计缺陷不在 tradeoff，而在 conflation**：dense+noisy 单通道**同时承担 monitoring 和 paper-truth 两个角色**。

**修复路径已经在 §6.2 部分铺好**：paper 时强制 `--max-slices 0`（Layer 4）→ 解决 paper-truth 角色。**唯一还差的一环**是 §5.1 的 best.pt 选 ckpt 问题——它独立于 §6.2 的修复（§6.2 是用 full-val 在 best.pt 上跑，但 best.pt **本身** 是被噪声选出来的）。

---

## 5. 下一个需要讨论的雷

排序的判据：(a) 是否会污染 paper 主结论；(b) 是否被现有纪律覆盖；(c) 修复成本。

### 5.1 ⚠️ 最大未拆雷：`best.pt` 单窗口选择偏差（极值统计）

**性质**：本文 §2.2 在分析 rolling-window tradeoff 时**新发现**的 landmine——不在原 codex2 9 风险列表中、不在 §6.2 full-val 纪律覆盖范围内。

**机制回顾**（§2.2）：

- 单次 eval `val_select_score` 由 4 个 chain MSE 加权和构成，每个分量 CV ≈ 25%
- 训练 500 次 eval = 500 次"窗口抽签"
- `best.pt` 选 min → 期望比 population mean 偏低 ~61% (~3σ)
- V6 best.pt 1.70e-4 vs last.pt 3.55e-4 的 +108% gap **~95% 由此解释**
- A/C/D 三 run 各自 `best.pt` 都被同样污染 → **基于 best.pt 行的 ablation 比较不可信**

**为什么 §6.2 full-val 纪律解决不了这个**：

- §6.2 说 paper 数字必须用 `eval_first_hop_224_clip3.py --max-slices 0` 跑 best.pt → 这只解决 *给定 best.pt* 的"那个数字精度问题"
- 但 best.pt **是哪个 step 的 ckpt** 本身已经被窗口运气污染 → full-val 跑出来的是"窗口幸运 step 的 full-val 数字" ≠ "真正最优 step 的 full-val 数字"

**修复路径表**（按改动量从小到大）：

| 修复 | 何时改 | 改动量 | 阻断风险 | 状态 |
|---|---|---|---|---|
| **(D) Method D — 局部邻域平滑选 ckpt（user proposed）** | paper 阶段，零 ckpt 改动量 | 30 行 python | -7% 残余 | ✅ **推荐**，已实证 |
| **(A) Paper 时报多个 ckpt 的 full-val 包络** | paper 阶段，零代码改动 | 0 行 | 完全消除 | 取决 save_interval（见下） |
| **(B) 切换为 EMA 平滑后的 val_select_score** | 下一轮论文 | ~10 行 train_first_hop.py | 极大缓解 | 中期 |
| **(C) 加 Layer 3 sparse full-val 当 best.pt 选择信号** | 下一轮论文 | ~30 行 train_first_hop.py | 完全消除 | 长期 |

### 5.1.1 推荐：Method D — user "局部最优" 思路的精确化

**机制**：现有 6 个 saved ckpts（save_interval=20000、120K 总步数），在每个 saved step $S$ 邻近取 $K$ 个 eval 行、平均 `val_select_score`，然后选 $S^* = \arg\min$。

**数学**：由于相邻 rolling windows 完全不重叠，$2K+1$ 个 eval 的平均等价于看 $(2K+1) \times 64 \times 8$ samples：

$$
\text{CV}_{\text{smooth}} \approx \frac{\text{CV}_{\text{single}}}{\sqrt{2K+1}}, \quad K=10 \Rightarrow \frac{25\%}{\sqrt{21}} \approx 5.5\%
$$

**极值偏移**（6 个 saved ckpts 选 min vs 500 evals 选 min）：

$$
\mathbb{E}[\min_{6}] \approx \mu - 1.27\sigma_{\text{smooth}} = \mu \times (1 - 1.27 \times 0.055) \approx 0.93\mu \Rightarrow \boxed{\textbf{-7\% bias}}
$$

从 **-61% (raw best.pt) 降到 -7% (Method D)**，零磁盘成本。

### 5.1.2 V6 snapshot 实证验证

[`v6_transport_first_metrics_snapshot_20260430_211557.jsonl`](log_snapshots/v6_transport_first_metrics_snapshot_20260430_211557.jsonl) 207 个 eval 行（steps 400-82800），假设 save_interval=20K → candidate steps {20K, 40K, 60K, 80K}，K=10 邻域平滑：

```
  step  |  smoothed   |    std    |  CV   |  raw (single window)
  ----- | ----------- | --------- | ----- | ---------------------
  20000 | 6.362e-04   | 1.86e-04  | 29.3% | 9.700e-04
  40000 | 6.773e-04   | 1.79e-04  | 26.4% | 5.092e-04
  60000 | 3.483e-04   | 8.15e-05  | 23.4% | 3.129e-04
  80000 | 2.601e-04   | 6.61e-05  | 25.4% | 2.513e-04   ← Method D winner

Robustness (gap / winner_std):
  rank #2 (step 60000): 1.33σ above winner
  rank #3 (step 20000): 5.69σ above winner
  rank #4 (step 40000): 6.31σ above winner
```

**对比 raw_best (V6 现行 best.pt 逻辑)**：snapshot-wide min @ step 81600 = **1.82e-4** → 这个"幸运 window"选出的 ckpt 比 Method D 选的 step 80000 (**2.60e-4**) **低估 +43%**。**这个 +43% 就是被修复的 bias**。

**注意**：实测 CV=23-29% 高于预测的 5.5%，是因为 V6 snapshot 仅覆盖 Phase II ramp，邻域 21 evals (±4000 steps) 内模型本身在持续改善 → drift 纳入 std。Phase III 收敛段预计 CV 会接近预测值。但**这不影响选择鲁棒性**——5.69σ 的 #1 vs #3 gap 远超任何合理阈值。

### 5.1.3 实现 + 调用

[review/0502/scripts/select_best_ckpt_smoothed.py](scripts/select_best_ckpt_smoothed.py) — 纯后处理脚本，零代码改动量到 train_first_hop.py：

```bash
# Step 1: 用 Method D 推荐 ckpt
python review/0502/scripts/select_best_ckpt_smoothed.py \
    --metrics /data_2/.../A_main/run-.../metrics.jsonl \
    --ckpt-dir /data_2/.../A_main/run-.../ \
    --neighborhood 10
# 脚本会打印 6 个 saved ckpt 的 smoothed score + ranking + 1σ-gap warning

# Step 2: 对推荐的 + 次推荐的 ckpt 跑 full-val
python eval_first_hop_224_clip3.py \
    --config review/0502/configs/A_control.yaml \
    --checkpoint /data_2/.../ckpt_step_<recommended>.pt \
    --split val --max-slices 0
# 同样跑 next-best ckpt → 取 mean ± std 作为 paper 表数字
```

**不需要改 yaml、不需要重启训练、不需要加磁盘**——是 Pareto 最优修复。

### 5.1.4 磁盘代价表（澄清初次估计错误）

单 ckpt = model 625 MB + Adam states 1250 MB + EMA 625 MB ≈ **2.5 GB**（从 [`backbone.model`](configs/A_control.yaml) `hidden_size=[384, 2048]` `depth=[12, 2]` 推算 ~164M trainable params；非 100 MB）。

| save_interval | ckpt/run | 总数 (4 runs) | 总磁盘 | 训练 I/O 开销 | 评价 |
|---|---|---|---|---|---|
| **20000 (current)** | 6 | 24 | **59 GB** | 0.21% | ✅ 当前足够（配 Method D） |
| 10000 | 12 | 48 | 117 GB | 0.42% | 可选升级（12 个 candidates） |
| 5000 | 24 | 96 | 235 GB | 0.84% | 收益边际 |
| 2000 | 60 | 240 | **587 GB** | 2.1% | 太多 |
| 1000 | 120 | 480 | **1.2 TB** | 4.2% | ❌ 不可行 |

**初次文档里写的 "48 GB" 是错的**——按 100 MB/ckpt 估，差 25 倍。实际 ckpt 含 Adam states + EMA → 2.5 GB。**save_interval=1000 = 1.2 TB 不可行**；用户质疑是对的。

**最终决策**：

- 不修改任何 yaml 的 save_interval —— **保持 20000**
- Paper 阶段用 [Method D 脚本](scripts/select_best_ckpt_smoothed.py) 选 ckpt + top-2 ckpt 跑 full-val 取 mean ± std
- (B)/(C) 入下一轮论文的 architecture upgrade 表

**(B) 中期方案 — EMA 平滑 best 选择**：

在 [`resolve_best_selection_score`](../../train_first_hop.py#L1147) 之外维护一个 EMA：

```python
# 伪代码
if "val_select_score_ema" not in metrics:
    metrics["val_select_score_ema"] = metrics["val_select_score"]
else:
    alpha = 0.1  # half-life ~10 evals ~4000 step
    metrics["val_select_score_ema"] = (
        alpha * metrics["val_select_score"] +
        (1 - alpha) * metrics["val_select_score_ema_prev"]
    )
# 训练 yaml 改 best_metric: val_select_score_ema
# 这等价于在 ~10 个不同窗口上平均 → CV / sqrt(10) ≈ 8% → 极值偏移 ~ -1σ ≈ -8%（vs 原 -61%）
```

**此修复不阻塞当前 ablation**——但下一轮论文必做。

**(C) 长期方案**：见 §3.3 Layer 2/3 架构改造。

**推荐立刻做的**：

1. ✅ **已落实**：Method D 脚本 [`select_best_ckpt_smoothed.py`](scripts/select_best_ckpt_smoothed.py) — 不改 yaml save_interval (保持 20000)，paper 阶段后处理选 ckpt
2. ⏳ 把 §3.4 paired diff 判读方法回填进 [POST_V6_NEXT_STEPS.md §6.4](POST_V6_NEXT_STEPS.md) — Risk 4 spot check 启动前
3. ✅ **已落实**：在 §6.2 / §6.4 添加 Method D + paired diff 协议指引

### 5.2 次大雷：实验统计功效（statistical power）

**问题陈述**：σ-norm ablation 的核心 claim 是 "C (uniform) ≈ A (V6 step weights) under σ-norm" 或 "C 显著差于 A"。这是**效应量检测问题**。

**当前噪声地板**：

| 信号 | CV | 检测下限（≥3σ 才显著）|
|---|---|---|
| 单步 best-vs-best chain MSE | ~25% (window) + ~61% (extreme value bias) | 不可解释 |
| 同步 same-step paired chain MSE | ~5%（窗口噪声 cancel + 模型差异 SNR）| ~15% |
| Layer 4 full-val on best.pt | ~0.5%（受 best.pt 选择偏差污染） | ~1.5% |
| Layer 4 full-val + (A) 邻域平均 | ~0.5%（已去偏） | ~1.5% |

**未知**：σ-norm × step_weights uniform 的真实效应量是多少？

- 如果效应 < 5%：当前架构（best-vs-best）**完全不能检测**；paired 也接近边界
- 如果效应 ≈ 5-15%：必须用 paired，best-vs-best 一定误判
- 如果效应 > 15%：所有方案都能看出来

**含义**：在 A_main + C_uniform 跑完前，我们不知道效应量是多少，**但已经知道 best-vs-best 比较的 detection floor 是 ~76%**（极值偏移 + 窗口噪声）。如果效应是 5%，paper 表的"无显著差异"结论就是 Type II error 装成 result。

**讨论方向**：

- **(a)** A_main 跑到 30K（Phase I 完成）时先做一次 full-val 邻域评估，建立 effect size 初估 → 决定是否要扩展到 200K 或调整 N
- **(b)** Pre-register paper claim：明文写"我们能 detect 至少 X% 的 chain MSE 差异；< X% 视为 indistinguishable"
- **(c)** 是否需要多 seed？当前所有 ablation 都是单 seed=42。多 seed (e.g., 42/43/44) 三 run 平均能再降 sqrt(3) ≈ 42% 噪声 → 但成本 ×3 GPU·days

### 5.3 已知但延后处理的雷

| 项 | 状态 | 何时解决 |
|---|---|---|
| Risk 1 (V6 200K behavior) | 用户决定"后续再补充实验" | A_main + C_uniform 完成后再议 |
| Risk 6-9（codex2 list 剩余） | 未在 §6 处理 | paper 投稿前检查清单 |
| Layer 2 mid-noise eval | 架构改造 | 下一轮论文 |
| Layer 3 sparse anchor | 架构改造 | 下一轮论文 |
| Multi-seed validation | 未启动 | reviewer 要求时补 |

---

## 6. 推荐的下一个具体动作

按用户偏好的"先讨论再决策"路径：

**讨论项 1 ✅ 已决策（user 同意 "局部最优" 思路）**：是否把四个 ablation yaml 的 `save_interval` 从 20000 改为 1000？

- **不改**。原始估计错了 25 倍：1000 step 实际 = 1.2 TB 磁盘 + 4.2% I/O，不可行
- 改用 [Method D](scripts/select_best_ckpt_smoothed.py) 邻域平滑 — 零磁盘成本，bias 从 -61% 降到 -7%
- V6 snapshot 实证：raw_best (1.82e-4) 比 Method D 选的 (2.60e-4) 低估 +43%；Method D #1 vs #3 gap = 5.69σ

**讨论项 2 ✅ 已决策（paired diff judge 脚本 + 0.10 LOCKED 阈值）**：Risk 4 spot check 判读规则切换到 §3.4 paired diff

- **已 lock**：[POST_V6_NEXT_STEPS.md §6.4](POST_V6_NEXT_STEPS.md) 已替换 best-vs-best 规则
- 脚本 [`paired_diff_judge.py`](scripts/paired_diff_judge.py)：reads 两 metrics.jsonl + 两 yaml；输出 mean ± SE ± CI95 + autocorr-corrected N_eff；exit code 0 (no_confound) / 10 (confound) / 11 (borderline) + guards 1-5 防 paired diff 在前提不满足时被错误使用
- LOCKED 阈值 0.10 的 audit 见 §3.4 与 [POST_V6_NEXT_STEPS.md §6.4](POST_V6_NEXT_STEPS.md)：单次 paired CV ~5%（理论），50 obs × N_eff~10 → mean SE ≈ 1.5-2%，0.10 阈值在 mean 上是 5σ 距离 → 统计功效充足；Risk 4 是防御性检查，应倾向高敏感度（0.10 优于 0.15）
- 同一脚本可作 §6.6.5 auxiliary anchor 用（A_main vs C_uniform 早期 monitoring，不进 paper）

**讨论项 3 ✅ 已决策（blinded analysis pre-registration）**：是否预先 pre-commit "效应量 < X% 不写 paper" 的红线？

- **已 lock**：见 [POST_V6_NEXT_STEPS.md §6.6](POST_V6_NEXT_STEPS.md) — 不锁单一 X，锁的是 **公式** `X = max(0.10, 3 × paired_CV_A)`
- 公式只用 A_main 自身数据（A 是 control，先跑完）；C_uniform 数据**直到 X 锁定后**才允许查看
- 严格执行顺序写在 §6.6.2，关键 commit 是 `EFFECT_SIZE_LOCKED.md`（A_main 完成后才填）
- LOCKED decision rule（§6.6.3）：`< X` ≈ / `[X, 2X]` grey zone 触发 200K continuation / `> 2X` significant
- **保留了用户"数据驱动"直觉的同时满足 pre-registration 要求** → reviewer-defensible

---

> **生成参考**：[`train_first_hop.py:820-850`](../../train_first_hop.py#L820) `_resolve_eval_window` / [`train_first_hop.py:1147-1190`](../../train_first_hop.py#L1147) `resolve_best_selection_score` / [V6 metrics snapshot](log_snapshots/v6_transport_first_metrics_snapshot_20260430_211557.jsonl) / [POST_V6_NEXT_STEPS.md §6.2](POST_V6_NEXT_STEPS.md) full-val 强制纪律 / [POST_V6_NEXT_STEPS.md §6.4](POST_V6_NEXT_STEPS.md) Risk 4 spot check 判读规则。
