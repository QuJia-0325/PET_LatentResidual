# PET_LatentResidual 当前总体设计与实验矩阵深度审查

> 日期：2026-05-03 01:40 CST  
> 分支：`foc_lite_hop0`  
> HEAD：`2cf3a89` (`review/0502: v3.1 follow-up — sanity sentinel freshness + summarize_run paper-table caveat`)  
> 审查范围：first-hop latent transport 主线、V6/V6.1、review/0502 σ-normalize ablation bundle、当前 sanity gate 运行状态、训练/eval 口径、数学推导与代码落点。  
> 使用技能：`research-review`（研究设计审查）、`formula-derivation`（σ-normalize 推导核查）、`monitor-experiment`（当前实验状态抽取）。

---

## 0. Executive Summary

当前总体设计已经从“继续堆机制”转向一个更正确的研究问题：**V6 的 rollout step_weights 形状是否有独立贡献，还是只是补偿不同 hop 的自然尺度 `(sigma * dt)^2`**。这个方向是合理的，也是目前最应该优先完成的实验。

最重要结论如下：

1. **当前 σ-normalize ablation 的数学设计是正确的。**  
   因为 `rollout_first_hop.py` 中 rollout loss 是加权平均而不是加权和，所以 A/B 等价必须同时保持分子和 `sum(weights)`。当前 `preserve_v6_sum` normalizer 正确解决了这个问题，离线验证 FP64 误差为 `2.71e-20`。

2. **当前实验矩阵合理，但结论边界必须收窄。**  
   `A_main/C_uniform/D_closed_form` 是 120K 压缩 schedule，不是 V6 200K full schedule。因此它能支持“120K 压缩 schedule 下 rollout weight shape 的相对贡献”，不能直接支持“200K full convergence 下最终最优形状”。

3. **C/D 必须等待 full sanity Tier 1-4 PASS。**  
   light sanity 只验证 Tier 1-3，不验证 chain MSE。现在 GPU1 上正在运行 `SANITY_STEPS=20000` 的 full mini sanity，这是正确动作。C/D 在 sentinel 前启动会污染结论。

4. **V6 本身不是强于 V3 的证据。**  
   V6 best.pt 在 step `185600`，`val_select_score=0.0006072739`，`val_chain_normal_mse=0.0001702655`。V6 last step `200000` 明显退化到 `val_chain_normal_mse=0.0003548939`。所以所有后续比较必须使用 best.pt 行，而不是 last，也不能包装为“V6 明显优于 V3”。

5. **当前最大科学风险不是代码 bug，而是 claim 过强。**  
   `loss.pair_weight=15` 且 `pair_loss_weights=[2.5,1,1,1]` 强化了 pair/hop0 通道。若 `C_uniform ≈ A_main`，只能说明在当前强 pair 监督下 rollout weight shape 边际贡献不明显；不能说 hop weighting 普遍无意义。

6. **当前最大工程风险是资源与非确定性。**  
   full sanity/full chain eval 会加载 train latents、val latents 和 val 全部 rollout image，单进程 RSS 约 140-160GB；当前机器可以承受，但不应再叠加多个 full PET 训练。代码设置 `deterministic=true`，但 PyTorch 实际以 `warn_only=True` 运行，日志显示仍有非确定 CUDA 算子，所以 A/B 不应要求 bit-identical，只能按 tier 阈值判定。

---

## 1. 当前实验状态快照

### 1.1 GPU / 进程状态

截至本文件生成时：

| 任务 | GPU | 状态 | 备注 |
|---|---:|---|---|
| `sigma_full_sanity_gpu1_0503` | 1 | 运行中 | `A_sanity` 已进入训练，20K 后自动接 `B`，最后 comparator |
| `sigma_sanity_light_gpu2_0502` | 2 | 运行中 | light A 正在跑；只覆盖 Tier 1-3，不可替代 full sanity |
| `v6_1_gpu3_0430` | 3 | 运行中 | V6.1 rollout-floor 继续训练 |
| V6 transport-first | - | 已完成 | best.pt 已确定，last 明显退化 |
| GPU0 | 0 | 被其他用户任务占用 | 非本项目 PET 训练 |

GPU1 full sanity 的启动命令：

```bash
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python \
SANITY_STEPS=20000 GPU=1 \
bash review/0502/scripts/run_ablation.sh sanity
```

当前 full sanity `A_sanity` 进度：

| 指标 | 值 |
|---|---:|
| last train step | 600 |
| first val step | 400 |
| `val_select_score` @ 400 | `0.0007849058` |
| `val_chain_normal_mse` @ 400 | `0.0002290948` |
| `lambda_roll` | `0.0` |
| `alpha` | `0.0` |

注意：step 400 仍在 Phase I，`lambda_roll=0`，只能说明 pipeline 正常，不代表 sanity 结论。真正的 sanity 结论必须等 A 20K + B 20K 全部完成后看 comparator。

### 1.2 V6 / V6.1 当前指标

| Run | 状态 | best step | best `val_select_score` | best `val_chain_normal_mse` | last val step | last `val_chain_normal_mse` | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| V6 transport-first | 完成 | 185600 | `0.0006072739` | `0.0001702655` | 200000 | `0.0003548939` | best 明显优于 last；报告必须用 best.pt 行 |
| V6.1 rollout-floor | 运行中 | 68000 | `0.0007203944` | `0.0002135197` | 68800 | `0.0002503021` | 仍处 Phase II 早段，不能 final 判断 |
| sanity light A | 运行中 | 3600 | `0.0009073296` | `0.0` | 15200 | `0.0` | `require_chain_metrics=false`，chain 值为占位，不可用于结论 |

---

## 2. 总体代码架构审查

### 2.1 架构主线

当前 first-hop transport 架构可以概括为：

```text
raw PET dose images -> frozen RAE encoder -> latent z_t
                         |
                         v
                 latent-only multi-hop transport
                         |
                         v
                  frozen RAE decoder for eval/image_aux
```

模型核心是 `PETFlowDiTFirstHop`：

| 模块 | 作用 | 代码证据 |
|---|---|---|
| Frozen RAE + pretrained backbone | latent 表示和主干 transport | `pet_lr/model_first_hop.py:269-286`, `569-594` |
| Hop0 pixel forcing | 只在 hop0 注入 D50 image 条件 | `pet_lr/model_first_hop.py:444-476` |
| Hop residual head | 在 shared velocity 上加小幅 hop-specific residual | `pet_lr/model_first_hop.py:191-261` |
| Mean-flow update | `z_pred = z_src + sigma * v_raw * dt` | `pet_lr/model_first_hop.py:505-511` |
| Optional seam refiner | 解码后局部 residual refine | `pet_lr/model_first_hop.py:115-188`, `538-566` |

当前架构优点：

1. **状态空间清晰。** Rollout 状态始终是 latent，而不是反复 decoder/encoder 的 pixel-state，这避免了级联 decoder artifact 进入下一 hop 的状态。
2. **pixel 只在 hop0 使用，设计保守。** 当前视觉 artifact 明显时，避免让 hop1+ 反复使用错误 pixel prior 是合理的。
3. **强 fail-fast。** path guard、alignment audit、fresh output dir、resume signature、chain metric requirement 都在降低“错误实验跑很久”的概率。
4. **监控指标丰富。** 训练日志记录 `pair_frac/roll_frac/img_frac`、`lambda_roll/alpha`、gate、hop residual、raw rollout step loss 等，对定位失败很有用。

当前架构不足：

1. **Hop residual 实际可能很弱。** V6 训练中 `lambda_hop_0`、`v_hop_abs` 一直较小，说明主要能力仍来自 shared backbone。若论文声称“hop residual 是核心建模能力”，证据不足；更合理说法是“hop residual 是局部校正 safety branch”。
2. **Hop1+ 没有 image-level correction。** 这是保守设计，但也解释了视觉质量提升受限。当前模型主要优化 latent MSE，不保证 decoded texture/lesion contrast 最优。
3. **Pair 通道过强，可能遮蔽 rollout 权重形状。** `loss.pair_weight=15`，并且 pair hop0 加权 `[2.5,1,1,1]`，这会让 rollout shape ablation 的效应变小。
4. **训练入口很大，职责集中。** `train_first_hop.py` 同时承担 dataloader、schedule、loss、eval、ckpt、resume、monitor。优点是可控，缺点是未来继续加 ablation 容易引入隐藏交互。

---

## 3. 数学推导审查：σ-normalize 是否正确

### 3.1 要解释的现象

V6 rollout step weights 为：

```text
w_V6 = [0.5, 2.0, 1.5, 1.0]
```

问题不是“这组数是否好看”，而是：

```text
H1: 这组权重体现 multi-hop Gronwall / clinical objective 的结构性最优。
H2: 这组权重只是补偿不同 hop 的自然误差量级 (sigma * dt)^2。
H3: 两者都有。
```

当前 σ-normalize ablation 的目的，就是把 H1/H2 分开。

### 3.2 不变量：weighted-average rollout objective

代码中 rollout loss 是：

```python
total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
```

代码位置：`pet_lr/rollout_first_hop.py:117-119`。

因此真正的不变量不是简单的 `sum_j w_j loss_j`，而是：

```text
L = sum_j w_j * loss_j / sum_j w_j
```

在 target-normalized velocity 空间中，latent rollout step loss 近似为：

```text
loss_j_raw = rho_j * ||delta v_j||^2
rho_j = (sigma_j * dt_j)^2
```

其中：

```text
sigma = [0.009634, 0.002946, 0.000775, 0.000141]
dt    = [3, 5, 15, 75]
rho   = [8.3533e-04, 2.1697e-04, 1.3514e-04, 1.1183e-04]
```

### 3.3 A/B 等价条件

A_control：不开 sigma-normalize：

```text
L_A = sum_j w_j * rho_j * ||delta v_j||^2 / sum_j w_j
```

B_sanity：开启 sigma-normalize，每 hop 除以 `n_j`，权重用 `w'_j`：

```text
L_B = sum_j w'_j * (rho_j / n_j) * ||delta v_j||^2 / sum_j w'_j
```

要使 `L_A == L_B`，充分条件是：

```text
w'_j = w_j * n_j
sum_j w'_j = sum_j w_j
```

这要求 `n_j` 在 V6 权重下的加权均值为 1：

```text
n_j = rho_j / weighted_mean(rho, w_V6)
```

当前验证输出：

```text
weighted_mean(rho, w_v6) = 2.3323e-04
normalizer = [3.5816, 0.9303, 0.5794, 0.4795]
B_sanity weights = [1.7908, 1.8606, 0.8691, 0.4795]
sum(B) = sum(A) = 5.0
max |contribA - contribB| = 2.710505e-20
```

对应代码：

| 功能 | 代码位置 |
|---|---|
| normalizer 计算 | `train_first_hop.py:395-464` |
| training rollout 注入 normalizer | `train_first_hop.py:517-540` |
| validation rollout 注入 normalizer | `train_first_hop.py:975-996` |
| step loss 除 normalizer | `pet_lr/rollout_first_hop.py:99-104` |
| 加权平均 | `pet_lr/rollout_first_hop.py:117-119` |
| 离线验证脚本 | `review/0502/scripts/verify_normalizers.py:39-132` |

结论：**A/B 的数学等价设计是成立的，且代码落点正确。**

### 3.4 C/D 真正测试的变量

在 preserve_v6_sum 下：

| Condition | sigma-norm | weights | sum | velocity-space hop contribution |
|---|---|---|---:|---|
| A_control | off | `[0.5,2.0,1.5,1.0]` | 5.0 | `[0.358,0.372,0.174,0.096]` |
| B_sanity | on | `[1.7908,1.8606,0.8691,0.4795]` | 5.0 | 与 A 等价 |
| C_uniform | on | `[1.25,1.25,1.25,1.25]` | 5.0 | `[0.25,0.25,0.25,0.25]` |
| D_closed_form | on | `[3.5795,0.7910,0.4149,0.2146]` | 5.0 | `[0.716,0.158,0.083,0.043]` |

因此 C/D 的变量确实是权重形状，而不是 rollout 总强度。

### 3.5 数学推导的边界

不能声明的内容：

1. 不能说 D_closed_form 是严格理论最优。它依赖 `L_j ≈ 1` 的简化假设。
2. 不能说 C≈A 就证明 hop weighting 无意义。只能说在当前 pair-weighted V6 setup 下，rollout shape 的边际贡献弱。
3. 不能把 raw chain MSE 跨 hop 横比当作模型难度。跨 hop 自然尺度由 `(sigma * dt)^2` 主导，应当归一化后再讨论。
4. 不能用 `summarize_run.sh` 的 per-metric minima 作为论文表格主数值。paper 表必须用 best.pt 对应 step 的整行指标。

---

## 4. 实验矩阵审查

### 4.1 当前矩阵

| 实验 | 目的 | 步数 | 是否必须 | 当前判断 |
|---|---|---:|---|---|
| sanity light A/B | 快速检查 A/B rollout/pair/raw-step 是否接近 | 50K + 50K | 可选/辅助 | 正在 GPU2 跑，但不覆盖 chain |
| full mini sanity A/B | Tier 1-4 gate，验证 A/B 等价链路 | 20K + 20K | 必须 | 正在 GPU1 跑，是正确动作 |
| A_main | 120K compressed schedule control | 120K | 必须 | full sanity PASS 前可跑，但需资源 |
| C_uniform | 形状是否重要的主 ablation | 120K | 必须 | 必须等 sanity sentinel |
| D_closed_form | 理论闭式解对照 | 120K | 条件必跑 | 只有 A/C 明显分离时优先级变高 |
| V6.1 | floor=0.05 是否有 final 收益 | 200K | 独立补充 | 当前还未到 final 判断窗口 |
| V6 best eval | PSNR/seam/视觉复核 | full val | 应补 | 不阻塞 σ-norm，但影响论文可信度 |

### 4.2 为什么 full sanity 必须先于 C/D

`run_sanity_light.sh` 仅比较：

```text
val_pair_total
val_rollout_total
val_rollout_step_*_raw
```

代码位置：`review/0502/scripts/run_sanity_light.sh:60-99`。

它的配置：

```yaml
require_chain_metrics: false
val_include_full_x_rollout: false
val_image_timepoints: [D50, D20]
```

配置位置：`review/0502/configs/A_sanity_light.yaml:51-55`, `83-97`；`B_sanity_light.yaml:45-49`, `78-91`。

因此 light sanity 不能验证 `val_chain_*_mse`，不能解锁 C/D。正式 `run_ablation.sh sanity` 会强制检查 4 层 tier：

| Tier | 指标 | 阈值 | 代码位置 |
|---|---|---:|---|
| 1 | `val_pair_total` | `<1e-4 rel` | `run_ablation.sh:168` |
| 2 | `val_rollout_total` | `<1% rel` | `run_ablation.sh:169` |
| 3 | `val_rollout_step_*_raw` | `<1% rel` | `run_ablation.sh:170-173` |
| 4 | `val_chain_*_mse` | `<5% rel` | `run_ablation.sh:174-179` |

脚本还会检查 missing keys fail-strict：`run_ablation.sh:183-189`, `195-210`。

### 4.3 Sentinel gate 是必要的

C/D 依赖 sigma-normalize 代码路径。如果 normalizer 有 bug，C/D 的 120K 结果会全部污染。当前脚本做了三层保护：

1. C/D/main 需要 `.sanity_pass`：`run_ablation.sh:242-260`。
2. sentinel 必须记录 `sanity_steps >= 20000`：`run_ablation.sh:263-282`。
3. sentinel 指向的 A/B metrics 文件必须仍存在：`run_ablation.sh:284-299`。

这是合理设计。

一个小问题：当前 `main)` 分支在 `run_ablation.sh:399-405` 会在启动 A_main 前就要求 sentinel。文档中说 A_main 可以与 sanity 并行，脚本中单独 `A)` 分支确实允许无 sentinel：`run_ablation.sh:411-416`。因此操作上应使用：

```bash
# 可与 sanity 并行
GPU=<free> bash review/0502/scripts/run_ablation.sh A

# sanity PASS 后再跑
GPU=<free> bash review/0502/scripts/run_ablation.sh C
```

不要用 `main` 快捷命令来实现并行策略。

### 4.4 120K schedule 的合理性与限制

`train_first_hop.py` 会把 `warmup_ratio/ramp_ratio` 转为实际 step 数：`train_first_hop.py:1473-1506`。

因此：

| Run | max_steps | warmup_ratio | ramp_ratio | warmup steps | ramp steps | Phase III 起点 |
|---|---:|---:|---:|---:|---:|---:|
| V6 | 200K | 0.25 | 0.50 | 50K | 100K | 150K |
| full sanity | 20K | 0.25 | 0.50 | 5K | 10K | 15K |
| A_main/C/D | 120K | 0.25 | 0.50 | 30K | 60K | 90K |

合理性：

- A_main 与 C/D 使用同一个 120K 压缩 schedule，因此**相对比较是公平的**。
- 120K 比 80K 更合理，因为已进入 Phase III 早期。

限制：

- 120K compressed schedule 不是 200K full schedule。
- 如果 A/C 差异处于 5%-10% 或主/次指标方向冲突，必须延展到 200K 才能下定论。

### 4.5 Best selection 口径

当前 best selection 由 `training.best_metric: val_multi_objective` 决定。代码会计算加权和并写入 `val_select_score`：`train_first_hop.py:1147-1183`, `2435-2447`。

best.pt 保存使用 EMA 参数：`train_first_hop.py:2511-2532`。

关键注意：

- `val_multi_objective` 是配置名，不是 metrics.jsonl key。
- metrics.jsonl 里应读 `val_select_score`。
- `summarize_run.sh` 报每个 metric 独立最小值，不等价于 best.pt 行。

这点对论文表非常重要。

---

## 5. 当前设置是否合理

### 5.1 合理部分

1. **研究问题清晰。** 现在不是盲目试 V7，而是在审查 V6 中最可疑的机制：rollout weight shape vs `(sigma*dt)^2` magnitude compensation。
2. **A/B sanity 是强设计。** 它先验证数学等价与实现路径，再允许 C/D，避免把 bug 当 finding。
3. **C_uniform 是主实验。** 它直接回答“权重形状是否重要”。这是最少 GPU 成本下信息量最高的实验。
4. **D_closed_form 是解释性实验。** 它不应该先跑，但当 C 与 A 分离时，它能告诉我们“V6 形状好”还是“closed-form hop0-heavy 更好”。
5. **V6.1 独立保留是合理的。** 它回答 rollout floor 是否只提供 early stabilization，和 σ-normalize ablation 不冲突。
6. **full-val / chain metrics 的 gate 是必要的。** 因为最终问题是级联到 NORMAL，不是单步 pair loss。

### 5.2 不足与风险

#### Risk A: Claim boundary risk

如果 C≈A，只能得到：

```text
在当前 pair_weight=15、pair_loss_weights=[2.5,1,1,1]、120K compressed schedule 下，
rollout step-weight shape 的边际贡献不明显。
```

不能得到：

```text
hop weight shape 普遍不重要。
Gronwall 理论被否定。
PET hop difficulty 完全由 (sigma*dt)^2 决定。
```

#### Risk B: Pair channel confounding

pair loss 已经在 normalized velocity space 中训练：`train_first_hop.py:292-350`。而且 pair 通道全局权重为 15，hop0 pair 权重为 2.5。这可能把 rollout shape 的作用盖住。

如果 C≈A，下一阶段最有价值的补实验是：

```text
pair_loss_weights=[1,1,1,1] 下的 A/C 小规模复核
```

但这不是当前阶段的 blocker。

#### Risk C: Schedule compression confounding

A/C/D 的 120K 压缩 schedule 会让 ramp 更早结束，训练动态和 V6 200K 不同。若结果清晰（<5% 或 >10%），仍可做 early evidence；若灰区，必须 200K。

#### Risk D: Non-determinism

代码 `set_seed(... deterministic=True)` 使用 `torch.use_deterministic_algorithms(True, warn_only=True)`：`train_first_hop.py:40-49`。当前 full sanity 日志已出现 memory efficient attention 和 adaptive_avg_pool2d backward 非确定 warning。

含义：

- Tier 1 目标可以很严格，但不应期待真正 bit-identical。
- 如果 A/B 差异小于阈值，接受；如果失败，需要先看失败在哪一层，而不是直接否定数学。

#### Risk E: Resource pressure

full sanity 会加载：

- train latents: 115.3GB
- val latents: 32.5GB
- train images D50/D20
- val images D50/D20/D10/D4/NORMAL

当前 GPU1 full sanity 进程 RSS 已达约 148GB，V6.1 约 164GB。系统仍可用，但不建议同时启动另一个 full PET training。light sanity 通过 `latent_mmap` 和 D50/D20-only 降内存，但它不能替代 full sanity。

#### Risk F: V6 final/last instability

V6 best @ 185600 明显优于 last @ 200000：

```text
best chain_normal = 0.0001702655
last chain_normal = 0.0003548939
```

这说明 Phase III 后段存在震荡或 overfit/selection mismatch。后续 A/C/D 报告必须：

- 报 best.pt 行。
- 同报 last.pt 行作为稳定性风险。
- 不把单点 best 包装为稳定收敛。

---

## 6. 优势总结

1. **问题被收窄到可证伪的机制。** 这比继续设计 V7 更有科学价值。
2. **数学和代码闭环较好。** normalizer 推导、verify 脚本、train/eval 注入、raw-step logging、sentinel gate 都能互相验证。
3. **矩阵有层次。** sanity -> A/C -> D -> 200K extension 的顺序合理，避免一次性烧 GPU。
4. **文档已修正多处历史误导。** 包括 V6 vs V3 方向、best.pt 口径、`val_select_score` key、`summarize_run.sh` 不能做 paper 表等。
5. **实现保护强。** path_guard、fresh output、chain metric requirement、missing-key fail-strict gate 让实验更可复查。

---

## 7. 不足总结

1. **论文主 claim 仍偏弱。** 如果最终 C≈A，contribution 更像“尺度归一化解释和负结果”，不是强新方法。
2. **视觉质量问题没有被当前矩阵直接解决。** σ-normalize ablation 解释 latent rollout loss，不一定改善 decoded image artifact。
3. **pair 通道 confound 未解。** 当前矩阵固定 pair 权重，这是合理单变量，但会限制结论外推。
4. **120K 不是 final verdict。** 它是成本可控的 early signal。灰区必须 200K。
5. **full sanity 成本高。** 但这是当前架构下为了 Tier 4 chain metrics 必须付出的代价。
6. **非确定算子会影响 A/B 理想等价。** 需要按 tier 阈值和多指标判断，不要过度解读个位差异。

---

## 8. 建议的执行顺序

当前不建议启动 C/D。建议严格执行：

1. 等 GPU1 full mini sanity 完成。
2. 如果 full sanity PASS，写入 `.sanity_pass` 后启动 `A_main` 或 `C_uniform`。
3. 如果 GPU 资源允许，先跑 `A_main`，再跑 `C_uniform`；若只能单卡，按 `A_main -> C_uniform`。
4. C 结果出来后按下面决策：

| A vs C 结果 | 下一步 |
|---|---|
| `val_select_score` 和 `val_chain_normal_mse` 都 <5% | 认为 shape 边际贡献弱；D 可选 |
| 两者都 >10%，且 C 更差 | 跑 D，判断 closed-form vs V6 |
| C 更好 | 暂停 D，优先确认 C @ 200K 或复跑 seed |
| 5%-10% 或两个指标方向冲突 | 扩到 200K，不写强结论 |

5. V6.1 继续跑完 200K。它的判断和 σ-normalize ablation 分开：

```text
V6.1 final ≈ V6: floor 只提供 early stabilization，最终被 ramp 抹平。
V6.1 final 显著优于 V6: floor 有 final 收益，才考虑 V6.2 sweep。
```

6. V6 best.pt 需要补 full-val PSNR/seam/视觉复核，尤其是 tail slices。当前 latent MSE 指标不能独自支撑“视觉质量改善”。

---

## 9. 论文/报告应采用的安全表述

推荐表述：

```text
We isolate rollout-channel step weighting under a fixed pair-supervised transport setup.
A sigma-normalized coordinate system removes the natural (sigma*dt)^2 scale from per-hop latent rollout errors.
The A/B sanity condition is algebraically equivalent to the raw V6 rollout objective and verifies the implementation path before testing uniform and closed-form hop weighting.
```

如果 C≈A：

```text
Under the current pair-heavy V6 setup and a 120K compressed schedule, rollout step-weight shape contributes little beyond scale compensation.
```

如果 C 明显差于 A：

```text
After scale normalization, V6's middle-heavy weighting remains beneficial, suggesting a genuine hop-shape effect beyond (sigma*dt)^2 magnitude compensation.
```

禁止表述：

```text
V6 proves Gronwall optimality.
V6 is clearly better than V3.
Uniform result at 120K proves final 200K behavior.
D_closed_form is a theorem-level optimum.
Chain NORMAL MSE can be directly compared to D20 MSE without scale normalization.
```

---

## 10. Action Items

| 优先级 | 动作 | 原因 |
|---|---|---|
| P0 | 等待 GPU1 full sanity 完成，检查 Tier 1-4 | 解锁 C/D 的唯一可信 gate |
| P0 | 若 FAIL，按 tier 定位：pair/RNG、rollout total、raw step、chain | 不要盲目重跑 |
| P1 | full sanity PASS 后跑 A_main/C_uniform | 当前最高信息量实验 |
| P1 | A/C 报告必须使用 best.pt step 整行指标 | 避免 per-metric minima 混用 |
| P1 | 同时报告 last.pt 行 | 暴露训练稳定性，不 cherry-pick |
| P2 | 根据 A/C 结果决定是否跑 D | D 是解释性实验，不是当前 blocker |
| P2 | V6 best full-val PSNR/seam/visual tail review | 补视觉证据 |
| P3 | 若 C≈A，再考虑 pair weights uniform 的小复核 | 解除 pair confound |

---

## 11. 审查结论

当前总体设计与实验设置**可以继续执行**，但必须保持以下纪律：

1. C/D 不得绕过 full sanity sentinel。
2. 120K 结果只作为 compressed-schedule evidence。
3. A/C/D paper 表必须使用 best.pt 对应 step 的整行 metrics。
4. V6 不能包装为优于 V3；当前可说的是 V6 matches V3-level chain MSE with different training dynamics。
5. 若 σ-normalize 结果显示 shape 不重要，应接受这是一个有价值的负结果，而不是继续堆新机制掩盖事实。

我的总体判断：**当前实验矩阵是合理的，且是目前最值得跑的矩阵；它的主要价值是澄清机制，而不是立即产生一个更强的生成效果。**
