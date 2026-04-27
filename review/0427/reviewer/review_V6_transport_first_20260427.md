# Reviewer Notes — V6 Transport-First Plan

**审查日期**: 2026-04-27  
**审查对象**: `review/plan/transport_breakthrough_research_v6.md`  
**审查范围**: V6 方案正文、V3 200K 最新训练快照、V3 full-val 结果、0427 run_me 修复说明、`train_first_hop.py` 当前实现、V3/V5/null-control configs。  
**总体立场**: V6 的问题意识正确，Transport-First 是值得验证的下一方向；但当前 V6 还不应直接作为 200K 正式长跑启动。建议先补齐 0427 P0/P1 归因闭环，再把 V6 降级为 pilot + staged escalation。

---

## 0. Executive Summary

V6 抓住了一个真实现象：V3 200K 的 best rolling-val 出现在 `step=86800`，当时 rollout weighted loss fraction 达到约 `31.6%`；后续训练中 image_aux 重新主导到 `90%+`，rolling-val 没再刷新 best。这说明“后段训练目标偏向 image_aux，transport 监督影响力不足”是一个合理怀疑。

但 V6 当前有四个关键问题：

1. **把相关性写成了因果**：`roll_frac` 高时出现 best，并不能单独证明 rollout 占比导致 best。
2. **参数推导过于确定**：`lambda_pair=15, lambda_roll=4, lambda_img=0.04` 是合理候选，不是由数据唯一推出的结论。
3. **实现准备不足**：仓库当前没有 V6 config，`loss.pair_weight` 也尚未接入 `total_loss` 和 logging/watch 体系。
4. **实验顺序过急**：V5 full-val、V5 Path A、null-control full-val 还没有形成闭环，不应直接宣布 V6 supersedes V5。

推荐动作：

| 优先级 | 动作 | 目的 |
|---:|---|---|
| P0 | 完成 0427 full-val / Path A / null-control full-val | 先确认 V5 真失败，以及失败归因 |
| P1 | 实现 `loss.pair_weight`，但先 dry-run 验证日志 fraction 正确 | 避免监控指标失真 |
| P1 | 新增 V6 pilot config，而非直接 200K | 低成本验证 `pair_weight` + rollout ramp 是否稳定 |
| P2 | 若 pilot 通过，再 escalates 到 full V6 200K | 控制 GPU 风险 |

---

## 1. 有证据支持的部分

### 1.1 V3 best 确实出现在 rollout fraction 相对较高的窗口

来自 `review/0427/logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl` 的关键 train rows：

| step | pair_frac | roll_frac | img_frac | lambda_roll | lambda_img | alpha |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.537 | 0.0057 | 0.457 | 0.002 | 0.005 | 0.0 |
| 86800 | 0.131 | 0.316 | 0.552 | 0.25 | 0.12 | 1.0 |
| 150000 | 0.0206 | 0.0525 | 0.927 | 0.25 | 0.12 | 1.0 |
| 172600 | 0.0127 | 0.0230 | 0.964 | 0.25 | 0.12 | 1.0 |

V3 当前 best rolling-val 仍是：

| metric | value |
|---|---:|
| best step | 86800 |
| `val_select_score` | 0.0005923647 |
| `val_chain_normal_mse` | 0.0001645001 |
| `val_chain_tail_mse` | 0.0001766701 |

因此 V6 的核心观察成立：后半程训练确实出现 `img_frac` 强主导，且 rolling-val 没再刷新 best。

### 1.2 从 scratch 避开 V5 resume 混淆是合理的

0427 reviewer 已指出，V5/null-control 的关键混淆之一是 resume 后 `max_steps` 改写可能压缩 cosine LR。当前代码已经加入 `lr_schedule.total_steps_override`，V5/null-control config 也已有 `total_steps_override: 200000`。

V6 如果从 scratch 训练 200K，可以直接避免 resume compatibility、best checkpoint basin、LR schedule override 等混淆，是合理方向。

### 1.3 回到头重 step_weights 有经验依据

V3 的 rollout step weights 是 `[1.30, 1.20, 1.10, 1.00]`，本来就是头重。V6 改成 `[2.5, 1.5, 1.0, 0.5]` 是在 V3 成功配置方向上加大 early-hop bias，而不是引入相反机制。

`first_hop.pair_v_std` 也支持 hop0 难度更高：

| hop | v_std |
|---|---:|
| D50->D20 | 0.009634 |
| D20->D10 | 0.002946 |
| D10->D4 | 0.000775 |
| D4->NORMAL | 0.000141 |

因此“优先保护 D50->D20 / early transport anchor”是合理的。

### 1.4 V3 full-val baseline 已经有可靠锚点

0427 新增了 V3 200K best full-val 结果，评估对象是完整 val split `n=7403`，`max_slices=0`，PSNR 使用 `src.utils.metrics.calc_psnr_clip3`。

| rollout node | full-val PSNR mean |
|---|---:|
| D50 | 42.6224 |
| D20 | 35.5444 |
| D10 | 35.9768 |
| D4 | 36.5579 |
| NORMAL | 36.8650 |

V6 用 `NORMAL > 37.5 dB` 作为强目标有明确对照基线。

---

## 2. 证据不足或需要降级的部分

### 2.1 “rollout fraction 高 -> best”目前只是相关，不是因果

V6 写法倾向于把 `step=86800` 的 best 归因给 rollout fraction 最高。但至少还有以下共变量：

| 共变量 | 为什么会混淆 |
|---|---|
| LR schedule | 86800 与 150000 的 effective LR 不同 |
| alpha stage | 86800 已进入 alpha=1.0 后不久，150000 是长时间 pred-chain 后 |
| rolling window | best 是 rolling-val selection，不是 full-val multi-ckpt rerank |
| loss absolute scale | 后段 pair/roll raw loss 本身下降，fraction 被 img loss 放大 |

建议把 V6 的核心 claim 改成：

> V3 后段训练显示 weighted loss pressure 自然漂移到 image_aux；V6 测试主动维持 transport pressure 是否能推迟 plateau。

不要写成：

> rollout 梯度占比高导致模型质量最好。

### 2.2 `pair_frac/roll_frac/img_frac` 是 weighted loss fraction，不是真实梯度占比

当前训练日志中的 fraction 是：

```python
pair_weighted = pair_losses["total"]
roll_weighted = lambda_roll * rollout_losses["loss_total"]
img_weighted = lambda_img * loss_img
weighted_total = pair_weighted + roll_weighted + img_weighted + foc_weighted
```

这不是 per-parameter gradient norm，也不是 per-hop gradient share。V6 文档多处使用“梯度占比”，需要改成“weighted loss fraction”，或补充真实梯度范数统计。

尤其 V6 的 per-hop 梯度分布表不应被当成真实梯度分布，因为：

1. rollout loss 内部按 `sum(step_weights)` 归一化；
2. alpha=1.0 时是多步 BPTT，后续 hop loss 会通过 pred state 影响前序 hop 计算图；
3. 同一个模型参数被所有 hop 共享或部分共享，无法从 step_weights 直接线性分配梯度；
4. image_aux 只在 hop0 aux batch 上计算，但其梯度对共享模块的影响不能简单写成 hop0 的固定百分比。

建议 V6 将“hop0 拿到 50.1% 梯度”降级为“按 loss coefficient 粗略估算的监督压力”。

### 2.3 `L_r/L_p ≈ 10 全程稳定` 说法过强

用当前 metrics snapshot 抽样计算：

| step | raw pair | raw rollout | `L_r/L_p` |
|---:|---:|---:|---:|
| 1000 | 9.06e-5 | 4.82e-4 | 5.3 |
| 86800 | 7.16e-5 | 6.90e-4 | 9.6 |
| 150000 | 1.36e-5 | 1.38e-4 | 10.2 |
| 172600 | 1.48e-5 | 1.07e-4 | 7.3 |

中后期确实接近 10，但“训练全程稳定”不成立。V6 参数可以继续作为候选，但需要说明它是基于中后期近似，而非全程严格统计。

### 2.4 `L_i/L_p` 波动不能直接归因于训练不稳定

V3 的 image_aux lambda 从 `0.005` ramp 到 `0.12`，本身就是 24x 的 schedule 变化。后段 `L_i/L_p` 变大，很大程度来自：

1. `lambda_img` 已到 0.12；
2. pair/roll raw loss 继续下降；
3. image loss 有 SSIM/seam 等项，尺度与 latent MSE 不同。

因此“`L_i/L_p = 35~370` 是梯度不稳定来源”这个说法需要降级。更稳妥的写法是：

> V3 后段 weighted objective 由 image_aux 主导，这可能压低 transport 更新的相对影响；V6 通过降低 `lambda_img` 并提高 transport 权重测试这一假设。

### 2.5 V3 alpha schedule 对比表存在事实错误

V6 第 7 节写 V3 alpha schedule 是 `0-5K GT, 5K-20K 混合`。对 200K V3 config 来说并不准确。

V3 200K config 中：

```yaml
rollout:
  warmup_ratio: 0.10
  ramp_ratio: 0.30
  alpha_start: 0.0
  alpha_end: 1.0
```

在 `max_steps=200000` 下对应：

| stage | step |
|---|---:|
| alpha=0 warmup | 0-20000 |
| alpha 0->1 ramp | 20000-80000 |
| alpha=1 | 80000-200000 |

V6 的 `0-50K / 50K-150K / 150K-200K` 确实更慢，但应与正确的 V3 schedule 对比。

---

## 3. 实现层面的缺口

### 3.1 当前没有 V6 config 文件

仓库当前没有 `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml`。V6 方案中只是给出了 YAML snippet，尚未形成可运行 config。

启动 V6 前至少需要：

1. 从 V3 200K config 复制一份 V6 config；
2. 明确 `run_name`、`max_steps`、`save_interval`、`metrics_jsonl_mode`；
3. 明确 rollout 的 `lambda_scale_mode`；
4. 明确 `loss.pair_weight` 放置位置；
5. dry-run 检查 config parse 与首条 train log。

### 3.2 `loss.pair_weight` 尚未接入训练代码

当前 `train_first_hop.py` 的 total loss 仍是：

```python
total_loss = (
    pair_losses["total"]
    + rollout_losses["lambda_roll"] * rollout_losses["loss_total"]
    + float(lambda_img) * loss_img
    + foc_losses["lambda_foc"] * foc_losses["loss_total"]
    + float(lambda_align) * loss_align
)
```

也就是说，V6 的核心机制 `loss.pair_weight: 15.0` 目前不会生效。

实现时不能只改 `total_loss`。还必须同步：

| 位置 | 需要改什么 | 不改的后果 |
|---|---|---|
| total loss | `pair_weight * pair_losses["total"]` | 核心机制不生效 |
| `pair_weighted` | 用同一个 `pair_weight` | `pair_frac` 日志错误 |
| loss balance watch | 使用加权后的 `pair_weighted` | Go/No-Go 判断错误 |
| train log / metrics JSONL | 记录 `pair_loss_weight` | 结果不可追溯 |
| config validation | 检查正数/finite | 防止 silent bad config |

因此 V6 所谓“3 行代码”低估了工程改动。核心很小，但可靠落地大约是 15-25 行，加一次 dry-run 验证。

### 3.3 `lambda_scale_mode` 必须显式指定

V3/V5 config 使用过：

```yaml
lambda_scale_mode: alpha_floor
lambda_scale_alpha_floor: 0.10
```

如果 V6 从 V3 config 复制而忘记改，那么 effective lambda 并不是文档中的简单 `0->4.0`。代码逻辑是：

```python
lambda_roll_eff = lambda_roll_base * max(alpha, alpha_floor)
```

例如在 V6 ramp 中点 `step=100K`，若 base lambda 为 2.0、alpha 为 0.5，则 effective lambda 是 1.0，而不是 2.0。

这不是一定错误，但文档必须写清楚：

| 选择 | 行为 | 建议 |
|---|---|---|
| `lambda_scale_mode: none` | effective lambda 按文档线性 0->4 | 更符合 V6 当前文字 |
| `alpha_floor` | effective lambda 被 alpha 再缩放 | 更保守，但需重算预期 fraction |

### 3.4 total loss scale 会显著变大，需考虑 grad clip / effective LR

以 V3 raw losses 粗投影 V6 权重：

| step | projected pair_frac | projected roll_frac | projected img_frac |
|---:|---:|---:|---:|
| 1000 | 68.8% | 0.0% | 31.2% |
| 86800 | 49.0% | 46.4% | 4.6% |
| 150000 | 21.2% | 57.6% | 21.2% |
| 172600 | 21.6% | 41.8% | 36.6% |

这个投影支持 V6 “transport-first”方向，但也显示：在某些后段 batch，image 仍可能超过 25%；并且 total weighted loss 比 V3 大很多，可能改变 grad_clip 命中率和 effective update size。

建议 V6 pilot 必须记录：

| metric | 目的 |
|---|---|
| `grad_total` | 检测总梯度是否被放大 |
| `grad_clip_pre` | 检测是否频繁 clip |
| `pair_loss_weight` | 追踪实际权重 |
| `pair_frac/roll_frac/img_frac` | 验证目标分布 |
| `val_pair_total` | 监控 GT-input anchor 是否损伤 |

---

## 4. 实验设计风险

### 4.1 不应在 V5 P0/P1 完成前宣布 supersedes V5

0427 当前已新增 V3 best full-val，但没有看到以下结果文件：

| 结果 | 当前状态 |
|---|---|
| V5 best full-val | 未在仓库发现 |
| V5 latest full-val | 未在仓库发现 |
| V3 Path A @ 86800 | 未在仓库发现 |
| V5 Path A | 未在仓库发现 |
| null-control full-val | 未在仓库发现 |

所以 V5 的最终归因还没闭环。V6 可以作为候选下一方案，但 `Supersedes: v5` 这个表述应改成：

> Candidate successor if 0427 P0/P1 confirms V5 loss design is harmful or unhelpful.

### 4.2 V6 的 `lambda_roll=4.0` 比 V5 更激进

V5 已经把 rollout lambda 从 V3 的 `0.25` 提高到 `1.5`，但没有可靠改善证据。V6 进一步提高到 `4.0`，是 V3 的 16x、V5 的 2.67x。

V6 用 `lambda_pair=15` 保护 pair anchor，这是合理补救，但还没有实证表明它足以抵消强 rollout BPTT 的副作用。

因此完整 200K V6 的启动风险较高。建议先做 pilot：

| run | max_steps | pair_weight | lambda_roll_end | 目的 |
|---|---:|---:|---:|---|
| V6-pilot-A | 20000 | 10 | 2.0 | 检查机制能否稳定生效 |
| V6-pilot-B | 50000 | 15 | 2.0 | 检查 pair_weight=15 是否安全 |
| V6-full | 200000 | 15 | 4.0 | pilot 通过后再跑 |

### 4.3 Go/No-Go 只看 fraction 不够

V6 第 4 节监控主要看 `pair_frac/roll_frac/img_frac`，但这只能证明权重分布符合预期，不能证明模型质量不被损伤。

建议补充 gate：

| checkpoint | 必看指标 | go/no-go 建议 |
|---:|---|---|
| 10K | train fraction + `grad_clip_pre` | pair_weight 生效且无频繁 clip |
| 20K pilot | rolling-val + small full-val/fixed subset | 不明显差于 V3 同步点 |
| 50K | full-val PSNR + Path A subset | D20/NORMAL 不崩，Path A 不恶化 |
| 100K | full-val PSNR | 若低于 V3 best 太多，停止 |
| 150K | full-val + `val_pair_total` | 核心 go/no-go，决定是否冲 200K |

### 4.4 成功标准应拆成 “minimum go” 和 “strong success”

V6 写 `NORMAL PSNR > 37.5 dB`，相对 V3 full-val `36.865 dB` 是 `+0.635 dB`。这个是强成功标准，不适合作为早期 go/no-go。

建议：

| 层级 | 标准 |
|---|---|
| minimum viable | full-val NORMAL 不低于 V3 best - 0.1 dB，且 Path A exposure gap 下降 |
| useful improvement | full-val NORMAL +0.2~0.3 dB，D20 不下降 |
| strong success | full-val NORMAL > 37.5 dB，且 Path A gap 下降 >= 10% |

---

## 5. 建议修订版 V6 执行计划

### Step 0: 不跳过 0427 P0/P1

先完成：

1. `01_fullval_eval.sh`：V3/V5 best/last/latest full-val。
2. `02_pathA_baseline.sh`：V3 Path A @ 86800。
3. `03_null_control.sh` + `05_fullval_null_control.sh`：resume/LR vs loss 改动归因。
4. `04_pathA_v5_best.sh`：V5 是否改善 exposure gap。

只有当 P0/P1 显示 V5 无益或有害，V6 才应正式成为下一主线。

### Step 1: 实现 `loss.pair_weight`

实现要求：

```python
pair_loss_weight = float(cfg.get("loss", {}).get("pair_weight", 1.0))
```

并用于：

```python
pair_weighted = pair_loss_weight * pair_losses["total"]
total_loss = pair_weighted + roll_weighted + img_weighted + foc_weighted + align_weighted
```

必须确认：

1. 旧 config 不设置 `pair_weight` 时完全兼容；
2. train log 中出现 `pair_loss_weight=...`；
3. `pair_frac` 反映加权后的 pair contribution；
4. 首个 50 step dry-run 中 `pair_w` 约等于 raw pair 的 `pair_weight` 倍。

### Step 2: 新建 V6 pilot config

建议先不要直接用 `lambda_roll=4.0` 完整 200K。pilot 可以使用：

```yaml
run_name: first_hop_224_v6_transport_first_pilot

training:
  max_steps: 20000
  save_interval: 5000
  rollout:
    warmup_ratio: 0.25
    ramp_ratio: 0.50
    alpha_start: 0.0
    alpha_end: 1.0
    lambda_start: 0.0
    lambda_end: 2.0
    lambda_scale_mode: none
    step_weights: [2.5, 1.5, 1.0, 0.5]
  image_aux:
    warmup_ratio: 0.0
    ramp_ratio: 0.0
    lambda_start: 0.04
    lambda_max: 0.04

loss:
  pair_weight: 10.0
```

如果 pilot 稳定，再升到 `pair_weight=15` / `lambda_roll_end=4.0`。

### Step 3: 设定 full V6 的断点验证

完整 V6 200K 如果启动，建议强制在以下节点保存并 full-val/fixed-subset：

| step | 动作 |
|---:|---|
| 50K | full-val or fixed subset；确认 Phase I 没把 D20/NORMAL 做坏 |
| 100K | full-val；检查 rollout ramp 中段质量 |
| 150K | full-val + Path A；核心 go/no-go |
| 200K | final full-val + Path A |

---

## 6. 建议改写 V6 文档中的关键表述

| 当前表述 | 建议改写 |
|---|---|
| `V6 Supersedes v5` | `V6 is a candidate successor pending 0427 P0/P1 attribution` |
| `rollout 梯度占比高时模型达到 best` | `rollout weighted loss fraction 较高的窗口对应当前 rolling-val best` |
| `L_r/L_p ≈ 10 训练全程稳定` | `L_r/L_p 在中后期约 7-10，早期/局部 batch 有波动` |
| `L_i/L_p 波动是梯度不稳定来源` | `image_aux weighted pressure 后段主导，可能压低 transport 更新影响` |
| `hop0 拿到 50.1% 梯度` | `按 loss coefficient 粗估，hop0 监督压力最高；真实梯度需 grad norm 验证` |
| `代码改动 ~5 行` | `核心逻辑小，但 logging/watch/config validation 也需同步更新` |

---

## 7. Verdict

| 维度 | 评分 | 判断 |
|---|---:|---|
| 问题定位 | 8/10 | 抓住了 V3 后段 image_aux 主导和 transport pressure 下降的问题 |
| 参数论证 | 5/10 | 有启发，但过于确定，需降级为候选超参 |
| 工程准备 | 4/10 | `pair_weight` 未实现，V6 config 未创建 |
| 实验风险控制 | 5/10 | 缺 pilot 和中途 full-val gate |
| 值得继续程度 | 7/10 | 值得做，但应分阶段验证 |

**最终建议**: 暂缓完整 200K V6。先完成 0427 P0/P1 归因；同时实现 `loss.pair_weight` 并跑 V6 pilot。若 pilot 显示 weighted loss fraction 达标、`val_pair_total` 不异常、small/full-val 不弱于 V3 同步点，再启动完整 V6 200K。
