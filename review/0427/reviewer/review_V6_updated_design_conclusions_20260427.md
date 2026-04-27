# V6 更新版实验设计结论审查

**日期**: 2026-04-27  
**审查对象**: `review/plan/transport_breakthrough_research_v6.md`（修订版）  
**相关文档**:
- `review/0427/reviewer/review_V6_transport_first_20260427.md`
- `review/0427/reviewer/v6_step_weights_math_derivation_20260427.md`
- `review/0427/logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl`
- `review/0427/logs_eval/v3_200k_best_fullval_clip3_summary.log`
- `train_first_hop.py`

**一句话结论**: 更新后的 V6 已经修正了旧版最严重的 rollout hop0 头重问题，核心方向现在更自洽；`loss.pair_weight` 也已经接入训练代码。但 V6 config、dry-run 验证和 staged pilot 仍未完成，因此还不能直接当作 200K 正式实验启动。

---

## 0. 总体判断

我对更新版 V6 的判断从“方向对但权重设计有明显错误”上调为：

> **设计方向合理，通道-跳分离的核心权重逻辑成立；但证据表述、实现准备和实验门控仍需补强。**

新版 V6 的最大改善是把旧版单纯 rollout 头重的方案改成：

```yaml
loss:
  pair_weight: 15.0

training:
  rollout:
    lambda_end: 4.0
    lambda_scale_mode: none
    step_weights: [0.5, 2.0, 1.5, 1.0]
  image_aux:
    lambda_start: 0.04
    lambda_max: 0.04

transport:
  pair_loss_weights: [2.5, 1.0, 1.0, 1.0]
```

这套设计的关键不是“所有 transport 都加大”，而是把不同 loss 通道分配给它们真正擅长的 hop：

| 通道 | 新版 V6 的职责 | 我的判断 |
|---|---|---|
| pair_loss | 通过 `pair_loss_weights[0]=2.5` 强化 hop0 GT-input velocity | 合理，hop0 是级联基础且 pair 是独占监督 |
| rollout_loss | 通过 `[0.5, 2.0, 1.5, 1.0]` 押 hop1/hop2 cascade drift | 合理，避免 rollout[hop0] 重复 pair |
| image_aux | 固定 `0.04`，作为 hop0 pixel-domain 辅助 | 可接受，但后段是否重新主导要监控 |

---

## 1. 更新版已经修正的关键问题

### 1.1 旧版 rollout hop0 头重问题已被修正

旧版 V6 的主要问题是 `step_weights=[2.5,1.5,1.0,0.5]`。它把“hop0 最难”这个事实错误地投射到了 rollout 通道上。

但代码结构决定：rollout[hop0] 的输入始终是 GT `D50` latent。

```python
z_curr = z_rollout[:, 0]  # GT D50
```

因此 rollout[hop0] 不承载 pred-chain exposure bias。它和 pair[hop0] 在输入分布上高度重叠，只是 pair 是 GT segment 全段随机采样，rollout 是起点单点监督。旧版给 rollout[hop0] 大权重，会造成表面 hop0 占比高、实际独占信息少的“虚胖”。

新版改成 `step_weights=[0.5,2.0,1.5,1.0]` 后，这个问题基本解决。

### 1.2 pair 头重 + rollout 中重的组合是自洽的

更新版 V6 把 hop0 难度交给 pair 通道处理：

```yaml
transport:
  pair_loss_weights: [2.5, 1.0, 1.0, 1.0]
```

这比旧版更符合训练目标的物理意义：

| hop | 主要困难 | 更合适的通道 |
|---|---|---|
| hop0 D50->D20 | GT-input velocity 本身难，CeilGap 最大 | pair_loss + image_aux |
| hop1 D20->D10 | exposure bias 开始出现，v_std 仍足够大 | rollout_loss |
| hop2 D10->D4 | exposure drift 更大，但 v_std 已下降 | rollout_loss 次重点 |
| hop3 D4->NORMAL | ExpoGap 最大，但 v_std/Jacobian 很小 | 不宜单独重押 |

也就是说，新版 V6 不再混淆“hop 难度”和“rollout 通道应该押哪里”。这是一次实质性升级。

### 1.3 `lambda_scale_mode: none` 是必要修正

新版 snippet 显式写了：

```yaml
lambda_scale_mode: none
```

这是必要的。如果沿用 V3/V5 的 `alpha_floor` 或 alpha scaling，那么文档里写的 `lambda_roll 0 -> 4.0` 不再成立，实际 rollout effective coefficient 会被 alpha 再缩放。当前 V6 的三阶段叙事要求 effective rollout lambda 按文档线性 ramp，因此 `none` 是更一致的选择。

---

## 2. 当前仍然成立的风险

### 2.1 V6 仍不应直接 supersede V5

V6 文档开头仍写：

```markdown
Supersedes: v5（rollout-heavy resume 实验）
```

这个表述仍然过早。

截至当前证据，V5 相关闭环还没有完全完成：

| 问题 | 为什么影响 V6 立项 |
|---|---|
| V5 full-val 是否真的失败 | rolling-val 不能单独决定方案失败 |
| V5 Path A 是否改善 ExpoGap | 即使 PSNR 不升，也可能有机制改善 |
| null-control full-val 是否持平 | 用来分离 loss 改动和 resume/LR 混淆 |

更稳妥的写法应该是：

> V6 是 V5 归因实验之后的候选 successor；在 V5 full-val / Path A / null-control 完成前，不应声明 supersedes。

### 2.2 “梯度占比”表述仍应降级为 weighted loss fraction

当前 `train_first_hop.py` 里的 fraction 计算是：

```python
pair_weighted = pair_losses["total"]
roll_weighted = rollout_losses["lambda_roll"] * rollout_losses["loss_total"]
img_weighted = pair_weighted.new_tensor(float(lambda_img)) * loss_img
weighted_total = pair_weighted + roll_weighted + img_weighted + foc_weighted
```

这不是 per-parameter gradient norm，也不是严格的真实梯度占比。新版 V6 已经在 per-hop 表里使用“监督压力分布”这个更谨慎的词，但文档前半部分仍多次使用“梯度占比”。

建议统一改成：

| 当前说法 | 建议说法 |
|---|---|
| 梯度占比 | weighted loss fraction |
| per-hop 梯度分布 | 按 coefficient 估算的 per-hop 监督压力 |
| transport 梯度占比 ≥75% | transport weighted loss pressure ≥75% |

如果后续确实要声称“梯度占比”，需要额外记录 pair/roll/img 对共享参数的 gradient norm。

### 2.3 `L_r/L_p ≈ 10 全程稳定` 仍然过强

根据 V3 snapshot 抽样：

| step | 近似 `L_r/L_p` | 解释 |
|---:|---:|---|
| 1000 | 约 5.3 | 早期不接近 10 |
| 86800 | 约 9.6 | 中期接近 10 |
| 150000 | 约 10.2 | 后期一度接近 10 |
| 172600 | 约 7.3 | 后期又下降 |

因此参数推导可以说“基于中后期 raw loss 量级，`L_r/L_p` 约 7-10”，但不应写“训练全程稳定”。

### 2.4 `image_aux` 的后段风险没有完全消失

V6 把 `lambda_img` 从 V3 后段的 `0.12` 降到固定 `0.04`，这是合理的。但 raw image loss 在 V3 后段上升明显，仅降低 coefficient 不一定保证后段 image fraction 永远低于 25%。

用 V3 raw loss 代入 V6 权重粗估：

| step | 使用 V6 实际 schedule 时 pair | rollout | image |
|---:|---:|---:|---:|
| 1000 | 68.8% | 0.0% | 31.2% |
| 86800 | 49.0% | 46.4% | 4.6% |
| 150000 | 21.2% | 57.6% | 21.2% |
| 172600 | 21.6% | 41.8% | 36.6% |

这里有两个含义：

1. V6 在 Phase II 中段不会立刻达到文档写的 rollout 70%，因为 `lambda_roll` 还在 ramp。
2. 如果 V6 后段 raw image loss 像 V3 一样上升，`image_frac < 25%` 不是自动保证。

所以 image_aux 固定 0.04 可以作为初版，但必须加后段监控；如果 `img_frac` 在 Phase III 连续超过 30%，需要考虑 decay 到 0.02 或 0.01。

### 2.5 V3 alpha schedule 对比仍有事实错误

V6 第 7 节把 V3 写成：

```text
0-5K GT, 5K-20K 混合
```

但 200K V3 config 是：

```yaml
warmup_ratio: 0.10
ramp_ratio: 0.30
```

即：

| 阶段 | V3 200K 实际 step |
|---|---:|
| alpha=0 | 0-20K |
| alpha 0->1 ramp | 20K-80K |
| alpha=1 | 80K-200K |

V6 的 `0-50K / 50K-150K / 150K-200K` 的确更慢，但必须和正确的 V3 schedule 对比。

---

## 3. 实现状态：pair_weight 已接入，但 V6 还不能直接跑

### 3.1 还没有 V6 config

当前仓库没有：

```text
configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml
```

V6 设计文档中的 YAML 只是 snippet，还不是完整可运行配置。正式启动前需要从 V3 200K config 派生完整 config，并明确：

- `run_name`
- `max_steps: 200000`
- `lr_schedule.total_steps_override` 是否保留为 200000
- `rollout.lambda_scale_mode: none`
- `transport.pair_weighting: manual`
- `transport.pair_loss_weights: [2.5,1.0,1.0,1.0]`
- metrics JSONL、save interval、eval window 设置

### 3.2 `loss.pair_weight` 已经接入训练代码

当前 `train_first_hop.py` 已经新增：

```python
pair_loss_weight = float(cfg["loss"].get("pair_weight", 1.0))
if pair_loss_weight < 0 or not math.isfinite(pair_loss_weight):
  raise ValueError(f"loss.pair_weight must be finite and non-negative, got {pair_loss_weight}")
if pair_loss_weight != 1.0:
  print(f"[config] loss.pair_weight = {pair_loss_weight:.4f}", flush=True)
```

total loss 也已经改成：

```python
total_loss = (
  pair_loss_weight * pair_losses["total"]
    + rollout_losses["lambda_roll"] * rollout_losses["loss_total"]
    + float(lambda_img) * loss_img
    + foc_losses["lambda_foc"] * foc_losses["loss_total"]
    + float(lambda_align) * loss_align
)
```

logging fraction 也同步使用加权后的 pair：

```python
pair_weighted = pair_loss_weight * pair_losses["total"]
```

metrics JSONL 中也记录了：

```python
"pair_loss_weight": pair_loss_weight,
```

因此旧结论中“`loss.pair_weight` 尚未实现”的判断已经过时。当前剩余风险转为：实现虽已完成，但还需要 dry-run 验证日志、fraction 和数值稳定性。

| 检查项 | 状态 | 仍需确认 |
|---|---|---|
| config parse | 已实现 | `pair_weight=15` dry-run 是否打印正确 |
| total loss | 已实现 | `pair_w` 是否约等于 raw pair 的 15x |
| logging fraction | 已实现 | `pair_frac` 是否使用加权后 pair |
| metrics JSONL | 已实现 | 日志是否实际写出 `pair_loss_weight` |
| loss_balance_watch | 间接已使用加权 `pair_weighted` | dominance gate 是否符合预期 |

这次实现覆盖了我之前要求同步修改的关键路径。下一步不是再讨论代码是否接入，而是做 50-200 step dry-run，把 `pair_loss_weight`、`pair_w`、`pair_frac`、`grad_clip_pre` 和 NaN/Inf 检查落到实际日志上。

---

## 4. 我认可的 V6 claim 与不认可的 claim

### 4.1 可以保留的 claim

| claim | 判断 |
|---|---|
| V3 后段 weighted objective 自然漂移到 image_aux 主导 | 可以保留 |
| 从 scratch 训练能避开 V5 resume/LR 混淆 | 可以保留 |
| V6 应主动维持 transport pressure | 可以保留 |
| pair 通道应头重 hop0 | 可以保留 |
| rollout 通道不应头重 hop0，应押 hop1/hop2 | 可以保留 |
| `lambda_scale_mode: none` 与 V6 三阶段 schedule 更一致 | 可以保留 |

### 4.2 需要降级或改写的 claim

| 当前 claim | 问题 | 建议改写 |
|---|---|---|
| rollout 梯度影响力与模型质量正相关 | 目前只是相关，不是因果 | V6 测试维持 rollout pressure 是否推迟 plateau |
| `L_r/L_p≈10` 训练全程稳定 | 早期和后段都不严格 | 中后期约 7-10，可作为粗略参数依据 |
| `L_i/L_p` 波动是梯度不稳定来源 | 部分由 schedule 和尺度差异导致 | image_aux 后段 weighted pressure 过高是一个合理怀疑 |
| transport 梯度占比 ≥75% | 日志不是真实梯度占比 | transport weighted loss pressure ≥75% |
| V6 supersedes V5 | V5 P0/P1 未闭环 | V6 是候选 successor |

---

## 5. 推荐的执行顺序

我不建议直接跑 V6 full 200K。建议按以下顺序推进。

### Step 1: 先完成 V5 / null-control 归因闭环

必须先回答：

| 实验 | 目标 |
|---|---|
| V5 full-val | 确认 V5 是否真的没有 full-val 改善 |
| V3 Path A | 建立可靠 ExpoGap baseline |
| V5 Path A | 看 rollout-heavy 是否改善 cascade drift |
| null-control full-val | 分离 V5 loss 改动 vs resume/LR 混淆 |

这些实验决定 V6 是“基于 V5 失败的替代方案”，还是“在 V5 可能有机制收益的基础上继续改”。

### Step 2: 验证 `loss.pair_weight` dry-run

代码已接入。下一步先做 50-200 step dry-run，检查：

| 检查项 | 预期 |
|---|---|
| `pair_loss_weight` 日志 | 显示 15.0 |
| `pair_frac` | 使用加权后的 pair |
| total loss scale | 无 NaN/Inf |
| grad_clip_pre | 不长期爆炸 |
| metrics JSONL | 可追溯记录 pair weight |

### Step 3: 先跑 V6 pilot，而不是 full 200K

推荐 pilot 设计：

| pilot | 配置 | 目的 |
|---|---|---|
| V6-P0 | `pair_weight=10`, `lambda_end=2.0`, 50K | 保守验证稳定性 |
| V6-P1 | `pair_weight=15`, `lambda_end=4.0`, 50K | 验证正式权重是否可控 |
| V6-P2 | 若 P1 稳定，延长到 100K/150K | 看 rollout ramp 是否按预期接管 |

Pilot 的 Go/No-Go 不应只看 fraction，还要看：

- fixed-window chain MSE
- full-val subset NORMAL PSNR
- `val_pair_total`
- Path A ExpoGap
- `grad_clip_pre` / NaN / loss spike

### Step 4: pilot 通过后再 full 200K

Full V6 的最低启动条件：

| 条件 | 要求 |
|---|---|
| V5 归因 | P0/P1 已完成或明确不阻塞 |
| 代码 | `pair_weight` 已实现且 dry-run 通过 |
| config | V6 config 完整可运行 |
| pilot | 至少 50K 无数值异常，目标分布大致符合预期 |
| eval | 有固定窗口或 full-val subset 快速比较 |

---

## 6. 最终结论

更新后的 V6 已经解决旧版中最关键的设计错误：它不再把 hop0 难度错误地压到 rollout step weight 上，而是通过 `pair_loss_weights[0]` 保护 hop0，通过 rollout 中重权重处理 hop1/hop2 的 cascade drift。这是合理的 Transport-First 方向。

但是，V6 当前仍不是可直接执行的正式实验：

1. `loss.pair_weight` 已在代码中实现，但还缺 dry-run 日志验证。
2. V6 config 尚未创建。
3. V6 plan 中仍有“梯度占比”“Supersedes V5”“L_r/L_p 全程稳定”等过强表述。
4. V6 plan 的代码片段部分存在一个重复/残留 code block，需要清理。
5. `image_aux` 后段是否重新主导仍需监控。
6. V5 / null-control 的归因闭环仍应先完成。

我的最终建议是：

> **把更新版 V6 保留为当前最强候选方案，但执行上仍按 staged pilot 推进。下一步应创建 V6 config 并做 50-200 step dry-run；dry-run 确认 pair_weight 生效且数值稳定后，再启动 50K pilot。pilot 稳定并显示 transport pressure 可控后，再投入 full 200K。**

---

## 7. 本轮二次核对结论（2026-04-27）

本轮重新查看修改后的 V6 plan 和当前代码后，结论更新如下：

| 项目 | 当前状态 | 判断 |
|---|---|---|
| 通道-跳分离权重 | V6 plan 已采用 `step_weights=[0.5,2.0,1.5,1.0]` + `pair_loss_weights=[2.5,1.0,1.0,1.0]` | 已吸收核心修正 |
| `loss.pair_weight` 代码 | `train_first_hop.py` 已实现 parse、total loss、fraction、metrics JSONL | 代码路径基本正确 |
| V6 config | `configs/pet_flow/*v6*` 未发现 | 仍未落地为可运行实验 |
| staged pilot | V6 plan 已新增 P0/P1/P2/Full 矩阵 | 设计比之前更稳健 |
| image_aux worst-case | V3 后段 raw image loss 若复现，V6 Phase III 仍可能 `img_frac > 25%` | 需要保留 130K/150K image gate |
| 文档一致性 | V6 plan 仍有过强表述和一个重复 code block | 建议清理后再作为执行文档 |

特别是 image_aux 风险需要更精确理解：用 V3 后段 raw loss 作为 proxy 时，V6 在 `step=150000` 量级仍可维持 transport pressure 约 79%；但如果接近 V3 `step=172600` 的 worst image raw loss，image fraction 可能升到约 36.6%。这说明 V6 的权重设计能显著改善 V3 后段 image 主导，但不能数学保证 `image_frac < 25%` 全程成立。V6 plan 新增的 `+130K / +150K` image gate 是必要的。
