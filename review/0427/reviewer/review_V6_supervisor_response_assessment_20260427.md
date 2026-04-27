# Reviewer Assessment — V6 Supervisor Response

**日期**: 2026-04-27  
**审查对象**:
- `review/0427/supervisor/v6_plan_supervisor_review_20260427.md`
- `review/0427/supervisor/to_supervisor_v6_response_20260427.md`
- `review/plan/transport_breakthrough_research_v6.md`
- `train_first_hop.py`

**一句话结论**: 对 supervisor 的 6 点回复总体可以接受，尤其是 `pair_weight` 冷启动和 `step_weights[0]=0.5` 的解释更精确；但回复中对 image_aux 风险和 V3 raw-loss 外推的反驳略偏乐观。当前最合理决策是：**允许进入 V6 config + dry-run + 50K pilot；仍不允许直接启动 full 200K。**

---

## 0. Verdict

| 阶段 | 判定 | 理由 |
|---|---|---|
| 创建 V6 config | **GO** | 核心代码 `loss.pair_weight` 已接入，设计参数已稳定 |
| 50-200 step dry-run | **GO** | 必须验证日志和数值稳定性 |
| V6-P1 50K pilot | **Conditional GO** | dry-run 通过后可直接跑正式权重 pilot |
| V6-P2 150K extension | **Gate required** | 需要 50K fixed-window/full-val subset 质量不退化 |
| V6-Full 200K | **NO-GO for now** | V6 config 未创建、dry-run 未做、V5/null-control 归因未闭环 |

Reviewer 立场：supervisor response 使 V6 的执行条件从“先 P0 保守试探”推进到“可尝试 P1 正式权重 pilot”，但没有强到可以跳过 pilot 直接 200K。

---

## 1. 对 6 个 Supervisor 问题的复审

### 1.1 image_aux 固定 0.04

**Supervisor 原判定**: Critical，建议预设退火。  
**Response 决策**: 不预设退火，只加 +130K 监控。  
**Reviewer 复审**: **部分接受 response，但保留风险。**

Response 的关键反驳是对的：V3 后段 `img_raw` 上升不是可直接移植到 V6 的前提，因为 V6 的目标正是改变 raw-loss 轨迹。用 V3 `172600` 的失败态 raw loss 直接预测 V6 Phase III，确实会把 V3 的失败机制当成 V6 的必然结果。

但 response 中“image 从来没有机会主导”的说法仍偏强。V6 能否维持 transport raw loss 在 V3-best 量级，是实验要验证的主 claim，而不是可以预设的事实。一旦 transport raw loss 下降到 V3 后段量级，固定 `lambda_img=0.04` 仍可能让 `img_frac` 超过 30%。

**执行判定**:

| 项目 | 决策 |
|---|---|
| pilot 是否必须预设 image decay | 不必须 |
| full 200K 是否应无条件固定 0.04 | 不建议 |
| 最低要求 | 保留 +130K / +150K image gate |
| 触发动作 | `img_frac > 30%` 且连续上升时停止或切换到 decay variant |

建议在 V6 config/runner 层面至少准备一个备用 config：`lambda_img=0.02` 或 Phase III decay 版本。即使不默认启用，也要能在 pilot 失败时快速切换。

### 1.2 pair_weight=15 冷启动

**Supervisor 原判定**: Critical，担心 from-scratch 不稳定。  
**Response 决策**: 不 ramp，加 +1K 安全检查。  
**Reviewer 复审**: **基本接受 response。**

AdamW 的尺度不变性推导成立，尤其在 gradient magnitude 远大于 epsilon 时，单纯把 pair loss 乘 15 并不等价于把学习率放大 15 倍。这个反驳有效。

但仍有两个边界：

1. 多 loss 项组合时，`pair_weight=15` 改变的是合成梯度方向，不只是 scale。
2. grad clipping、EMA、weight decay、AMP scaler 等训练系统行为仍可能受 total loss/grad norm 影响。

因此不需要预设 pair_weight ramp，但必须做 dry-run 和 +1K gate。

**执行判定**:

| 条件 | 判定 |
|---|---|
| 50-200 step dry-run 无 NaN/Inf | required |
| `pair_w ≈ 15 * pair_raw` | required |
| `grad_clip_pre` 无持续异常 | required |
| +1K pair_loss < 5e-4 | recommended gate |

### 1.3 §1.5 Phase III 分布预测

**Supervisor 原判定**: Critical，认为用 step 86800 预测 Phase III 错。  
**Response 决策**: 降级为 V5/V6 raw-loss 归因问题。  
**Reviewer 复审**: **接受 response 的核心，但要求 V6 plan 改写措辞。**

Response 说得对：V6 是从 scratch 新目标训练，不能把 V3 150K/172K 的失败态 raw loss 当成 V6 必然轨迹。

但 supervisor 也有一个正确点：V6 plan §1.5 现在写“Phase III（150K-200K）全局分布（V3 step 86800 raw loss 估算）”，这在表述上仍会误导读者，以为 `pair≈27/roll≈70/img≈3` 是 Phase III 的强预测。

建议改成：

> 用 V3 best-step raw loss 估算的是 V6 成功态/transport-active 态的目标分布，不是 Phase III 后段保证分布；Phase III 实际分布由 pilot 的 130K/150K checkpoint 实测决定。

**执行判定**: §1.5 不阻塞 dry-run，但在作为正式执行文档前应改写。

### 1.4 Phase I 0-50K GT-only

**Supervisor 原判定**: Important，担心 GT-only 太长。  
**Response 决策**: 保持原设计。  
**Reviewer 复审**: **暂时接受，但必须由 pilot 验证。**

长 Phase I 的合理性在于先把 GT-input velocity anchor 学稳，再引入 pred-chain。但 50K 是否过长没有直接证据，尤其 pair_weight=15 会加强 GT segment bias。

**执行判定**:

| 检查点 | 需要看什么 |
|---:|---|
| 10K | pair loss 是否下降、D20 fixed-window 是否改善 |
| 50K | rollout 尚未参与时，chain NORMAL 不应明显劣于 V3 同步点 |
| 75K | rollout ramp 开始后是否平滑接管 |

若 50K 表现为 pair loss 很好但 chain 质量差，说明 Phase I 可能过长或 pair-only 过拟合，需要把 warmup 从 0.25 降到 0.10/0.15。

### 1.5 `step_weights[0]=0.5`

**Supervisor 原判定**: Important，认为与“退化重叠”矛盾。  
**Response 决策**: 保持 0.5，修正措辞为 endpoint 一致性辅助。  
**Reviewer 复审**: **接受 response。**

`step_weights[0]=0.5` 只占 rollout step weight 总和的 10%。考虑 rollout[hop0] 仍提供固定全步 endpoint 检查，把它降到 0.5 比完全置 0 更稳健。关键是不要再把 hop0 rollout 当成主要独占信号。

**执行判定**: 保持 `[0.5,2.0,1.5,1.0]`。

### 1.6 `lambda_scale_mode: none`

**Supervisor 原判定**: Important，担心与 alpha 解耦。  
**Response 决策**: pilot 保持 none。  
**Reviewer 复审**: **接受作为 pilot 默认，但保留 alpha variant。**

`none` 的好处是 schedule 可解释：effective lambda 就是线性 `0->4`。`alpha` 模式更保守，但会把 rollout 的有效 ramp 变成二次增长，可能使 100K 前 rollout 压力不足。

**执行判定**:

| 默认 | 备用 |
|---|---|
| `lambda_scale_mode: none` | 若 60K-80K roll_frac 过高且质量不升，再试 `alpha` |

---

## 2. 当前文档/代码一致性检查

### 2.1 已对齐的部分

| 项目 | 状态 |
|---|---|
| `pair_weight` parse | `train_first_hop.py` 已实现 |
| total loss 乘 pair weight | 已实现 |
| `pair_frac` 使用加权 pair | 已实现 |
| metrics JSONL 记录 `pair_loss_weight` | 已实现 |
| V6 plan 中 `step_weights` | 已更新为 `[0.5,2.0,1.5,1.0]` |
| V6 plan 中 `pair_loss_weights` | 已更新为 `[2.5,1.0,1.0,1.0]` |
| V6 plan 中 +1K/+130K gate | 已加入 |

### 2.2 仍未对齐的部分

| 问题 | 位置 | 建议 |
|---|---|---|
| 没有 V6 config | `configs/pet_flow/*v6*` 未发现 | 从 V3 200K config 派生正式 V6 config |
| §2.1 有重复残留 code block | V6 plan 代码改动段落 | 清理多余片段，避免执行者误读 |
| §8.3 仍写 `loss.pair_weight` 代码实现“待完成” | V6 plan Pilot 最低启动条件 | 改为“已实现，待 dry-run 验证” |
| `Supersedes: v5` 仍偏强 | V6 plan header | 改为 candidate successor pending V5 attribution |
| V3 alpha schedule 仍写 0-5K/5K-20K | V6 plan 对比表 | 改成 0-20K/20K-80K/80K-200K |
| “梯度占比”仍偏强 | V6 plan 多处 | 改为 weighted loss fraction / 监督压力 |

---

## 3. 推荐下一步执行

### 3.1 最小可执行路径

| 顺序 | 动作 | 输出 |
|---:|---|---|
| 1 | 清理 V6 plan 的 stale 文本和重复 code block | 可执行版 V6 plan |
| 2 | 创建 `pet_flow_first_hop_224_v6_transport_first.yaml` | V6-P1 config |
| 3 | 50-200 step dry-run | dry-run log / metrics JSONL |
| 4 | 检查 `pair_loss_weight`, `pair_w`, `pair_frac`, `grad_clip_pre` | dry-run pass/fail |
| 5 | 启动 V6-P1 50K | first pilot result |

### 3.2 是否还需要 V6-P0

Response 建议直接启动 V6-P1（正式权重），不跑保守 P0。Reviewer 判断：可以接受，但有条件。

| 条件 | 若满足 | 若不满足 |
|---|---|---|
| dry-run 无数值问题 | 直接 V6-P1 | 跑 V6-P0 或降低 pair_weight |
| +1K pair_loss 正常 | 继续 P1 | stop and inspect |
| +10K pair_frac 70-90% 且 no clip storm | 继续 P1 | 降为 P0 配方 |

也就是说，P0 可以不作为独立预跑，但 P1 的前 10K 必须承担 P0 的安全验证功能。

---

## 4. Reviewer Final Decision

当前 V6 的核心方法论已经足够清晰：

1. 用 `pair_weight=15` 恢复 GT-input velocity 对 backbone 的主导权。
2. 用 `pair_loss_weights[0]=2.5` 保护 hop0。
3. 用 rollout `step_weights=[0.5,2.0,1.5,1.0]` 押 hop1/hop2 的 exposure-bias sweet spot。
4. 用 `lambda_roll=0->4` 延长 ramp，让 pred-chain pressure 后移。
5. 用 `lambda_img=0.04` 把 image_aux 限定为辅助，而非后段主导目标。

Supervisor response 对这些选择给出了可接受的辩护，因此 reviewer 不再要求预设 image decay、pair_weight ramp 或 `lambda_scale_mode=alpha` 作为启动前置条件。

但 full 200K 仍必须等 staged evidence：

| Gate | 通过标准 |
|---|---|
| Dry-run | 无 NaN/Inf，`pair_w` 和 `pair_frac` 正确，grad clip 无异常 |
| +1K | pair_loss < 5e-4，loss 无 spike |
| +10K | pair_frac 70-90%，img_frac 不主导 |
| +50K | fixed-window/full-val subset 不明显差于 V3 同步点 |
| +130K | img_frac 不连续上升到 >30% |
| +150K | rollout 接管，transport pressure 仍为主，质量指标有改善趋势 |

**最终判定**: **GO for V6 config + dry-run + V6-P1 pilot; NO-GO for direct V6-Full 200K.**
