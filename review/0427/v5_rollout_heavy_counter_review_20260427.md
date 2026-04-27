# V5 Rollout-Heavy Failure Analysis 的反驳与修正审计

日期: 2026-04-27  
分支: `foc_lite_hop0`  
审计对象: `review/0427/v5_rollout_heavy_analysis_20260427.md`  
参与审查: Codex 主审 + 3 个 subagent

## 0. 总结结论

Claude 的弱结论可以保留: **V5 当前 rolling-val 日志没有证明稳定增益，简单增强 open-loop / pred-state 训练压力不是一个已验证成功的方向**。

但 Claude 的强结论需要反驳或降级:

| Claude 观点 | 本审计结论 | 处理建议 |
|---|---|---|
| `step=93200` 发生相变 | 证据不足。`92800 -> 93200` 正好是 rolling validation window 从尾部回卷到开头 | 不应把 `93200` 作为机制断点 |
| V4/V5 同步崩溃证明 GT-optimal basin 极窄 | 过度推断。二者共享相同 eval schedule 和 resume step，同步现象高度可能来自 window schedule | 需要 fixed-window/full-val 或 null-control |
| `val_pair_total` 跳变证明模型被不可逆推出 GT-optimal manifold | 不成立。后续 `val_pair_total` 会回落到 `1e-5` 量级 | 不能用 rolling-val 单点证明不可逆 |
| V4 SF 和 V5 rollout-heavy 是同一种失败机制 | 只能说都引入 pred-state/open-loop 压力；梯度路径不同 | 需要分开分析 SF 单步 pred-state 和 rollout BPTT |
| V4/V5 `val_select_score` 可以直接比较 | 不严谨。V4/V5 的 D20 权重不同 | 必须用同一 objective 重算或 full-val 统一评估 |

本审计建议: **先做固定协议 full-val / fixed-window rerank，再决定是否停止或修改 V5 系列。不要基于 `step=93200` 的 rolling-val 单点定义相变。**

## 1. 反驳点一: `step=93200` 不是可信相变点，而是 rolling-val window 回卷点

### 1.1 代码机制

V4/V5 都使用 rolling validation window:

| 文件 | 行号 | 机制 |
|---|---:|---|
| `configs/pet_flow/pet_flow_first_hop_224_v4_sf_pilot.yaml` | 45-50 | `eval_interval=400`, `max_val_batches=64`, `val_window_mode=rolling` |
| `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 65-69 | V5 同样使用 `max_val_batches=64`, `val_window_mode=rolling` |
| `train_first_hop.py` | 737-759 | `_resolve_eval_window()` 使用 `eval_idx % num_windows` 计算窗口起点 |
| `train_first_hop.py` | 753-756 | rolling 模式下窗口起点为 `(eval_idx % num_windows) * take` |
| `train_first_hop.py` | 268-272 | val loader 默认 `val_shuffle=false`，因此窗口顺序固定 |
| `pet_lr/data_first_hop.py` | 131-134 | dataset index 顺序是 `pair_idx` 外层、`slice_idx` 内层，窗口不是随机样本 |
| `train_first_hop.py` | 1032-1037 | 每次 val 会输出 `val_main_window_start_batch` 和 batch/sample 统计 |

关键点: `step=92800` 和 `step=93200` 不是同一验证子集。

### 1.2 日志证据

`step=92800 -> 93200` 的原始 rolling-val 表:

| Exp | step | window_start | batches | samples | pair | normal | d20 | select |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | 92000 | 3520 | 64 | 512 | 0.000000 | 0.000226 | 0.000310 | 0.000723 |
| V5 | 92000 | 3520 | 64 | 512 | 0.000000 | 0.000213 | 0.000296 | 0.000791 |
| V4 | 92400 | 3584 | 64 | 512 | 0.000000 | 0.000198 | 0.000261 | 0.000630 |
| V5 | 92400 | 3584 | 64 | 512 | 0.000000 | 0.000189 | 0.000250 | 0.000693 |
| V4 | 92800 | 3648 | 54 | 428 | 0.000000 | 0.000170 | 0.000217 | 0.000538 |
| V5 | 92800 | 3648 | 54 | 428 | 0.000000 | 0.000163 | 0.000209 | 0.000591 |
| V4 | 93200 | 0 | 64 | 512 | 0.000196 | 0.000166 | 0.000253 | 0.000551 |
| V5 | 93200 | 0 | 64 | 512 | 0.000231 | 0.000158 | 0.000243 | 0.000614 |
| V4 | 93600 | 64 | 64 | 512 | 0.000210 | 0.000286 | 0.000425 | 0.000939 |
| V5 | 93600 | 64 | 64 | 512 | 0.000246 | 0.000272 | 0.000409 | 0.001044 |
| V4 | 94000 | 128 | 64 | 512 | 0.000203 | 0.000315 | 0.000452 | 0.001025 |
| V5 | 94000 | 128 | 64 | 512 | 0.000236 | 0.000303 | 0.000437 | 0.001143 |

解释:

- `92800` 是尾部 partial window: `window_start=3648`, 只有 `54` batches / `428` samples。
- `93200` 是回卷后的第一个窗口: `window_start=0`, `64` batches / `512` samples。
- Claude 把 `92800 -> 93200` 当成连续同分布时间序列，这是不成立的。
- V4/V5 同步出现非零 `val_pair_total`，更可能是因为二者同步切到 window 0，而不是训练动力学在同一步发生相变。

### 1.3 可接受的弱结论

可以说: **window 0/64/128 等验证窗口上，V4/V5 的 `val_pair_total` 都比尾部窗口更高**。

不能说: **模型在 `step=93200` 被训练推出 GT-optimal manifold**。

要证明后者，至少需要在相同 validation window 或 full-val 上比较 `92800`、`93200`、`latest` checkpoint。

## 2. 反驳点二: V4/V5 的 `val_select_score` 不能直接比较

### 2.1 代码机制

| 文件 | 行号 | 机制 |
|---|---:|---|
| `train_first_hop.py` | 1041-1077 | `resolve_best_selection_score()` 按 `best_metric_terms` 加权求和 |
| `configs/pet_flow/pet_flow_first_hop_224_v4_sf_pilot.yaml` | 68-77 | V4 的 `val_chain_d20_mse` 权重是 `0.15` |
| `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 87-96 | V5 的 `val_chain_d20_mse` 权重是 `0.50` |
| `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml` | 68-78 | 200K V3 baseline 的 D20 权重是 `0.50` |

因此，V5 的 logged `val_select_score` 会因为 D20 权重更高而系统性偏大。直接比较 V4 logged select 与 V5 logged select 会误导。

### 2.2 同一 objective 重算后的对齐表

下表用 V5/V3 的统一权重重算 V4/V5 的 common objective:  
`0.50*d20 + 0.45*d10 + 0.90*d4 + 1.50*normal`

| step | window | V4 logged | V5 logged | V4 common | V5 common | delta common |
|---:|---:|---:|---:|---:|---:|---:|
| 92000 | 3520 | 0.000723 | 0.000791 | 0.000832 | 0.000792 | -0.000040 |
| 92400 | 3584 | 0.000630 | 0.000693 | 0.000722 | 0.000693 | -0.000029 |
| 92800 | 3648 | 0.000538 | 0.000591 | 0.000614 | 0.000591 | -0.000023 |
| 93200 | 0 | 0.000551 | 0.000614 | 0.000640 | 0.000615 | -0.000026 |
| 93600 | 64 | 0.000939 | 0.001044 | 0.001088 | 0.001043 | -0.000045 |
| 94000 | 128 | 0.001025 | 0.001143 | 0.001184 | 0.001143 | -0.000041 |

含义:

- 按 logged select 看，V5 比 V4 差。
- 按同一套 V5/V3 权重重算，V5 在这些对齐窗口上反而略低。
- 这个差异很小，仍然不足以证明 V5 成功，但足以说明 Claude 对 V4/V5 logged select 的直接比较不严谨。

## 3. 反驳点三: V4 SF 与 V5 rollout-heavy 不是同构干预

Claude 的统一解释是: V4 和 V5 都改变 backbone 梯度分布，所以在相同步数后触发同类相变。

这个说法可以保留一半: **二者都增强了 pred-state/open-loop 压力**。

但不能说它们是同构机制，因为梯度路径明显不同。

### 3.1 V4 SF: pred-state 输入上的单步 pair loss

| 文件 | 行号 | 机制 |
|---|---:|---|
| `configs/pet_flow/pet_flow_first_hop_224_v4_sf_pilot.yaml` | 106-117 | V4 开启 `self_forcing_pair`，resume-relative schedule |
| `train_first_hop.py` | 470-481 | docstring 明确 SF 只改变 pair `z_src` 的 forward value |
| `train_first_hop.py` | 489-507 | SF 的 `alpha_sf` schedule 从 resume-relative step 计算 |
| `train_first_hop.py` | 541-557 | pre-rollout 逐 hop 预测，但每步 `z_pred.detach()` |
| `train_first_hop.py` | 568-574 | 用 `alpha_sf` 混合 GT `z_src` 与 pred `z_src` |
| `train_first_hop.py` | 1808-1823 | 训练时替换 `main_batch["z_src"]`，之后仍走普通单步 pair forward |

关键修正:

- V4 SF 的 pre-rollout 是 `detach()`，因此不是多步 BPTT。
- Backbone 参数只通过后续单步 `predict_latent_step()` 的 pair loss 接收梯度。
- V4 的风险是输入分布从 GT-state 偏向 pred-state，而不是 rollout chain loss 沿多跳反传。

### 3.2 V5 rollout-heavy: open-loop chain BPTT

| 文件 | 行号 | 机制 |
|---|---:|---|
| `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 127-129 | V5 禁用 `self_forcing_pair` |
| `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 131-161 | V5 开启 rollout-heavy，`lambda=1.5`, `step_weights=[0.8,1.0,1.5,2.5]` |
| `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 142-149 | V5 从 resume 后立即 `alpha=1.0`, `lambda_roll=1.5` |
| `train_first_hop.py` | 420-440 | runtime 解析 rollout `alpha`、`lambda_roll`、ST 开关 |
| `train_first_hop.py` | 442-450 | 调用 `rollout_multistep_losses_first_hop()` 计算 rollout loss |
| `pet_lr/rollout_first_hop.py` | 22-36 | `mix_latent()` 中 `alpha>=1` 直接返回 `z_pred` |
| `pet_lr/rollout_first_hop.py` | 72-104 | 多步 rollout 用上一跳预测作为下一跳输入 |
| `pet_lr/rollout_first_hop.py` | 106-108 | `step_weights` 会按权重和归一化，不是简单未归一化累加 |
| `train_first_hop.py` | 1907-1910 | 总 loss 中直接加 `lambda_roll * rollout_loss` |

关键修正:

- Claude 说 V5 `straight_through=True` 带来链式梯度，这个表述不准确。
- 在 V5 `alpha=1.0` 时，`mix_latent()` 直接返回 `z_pred`，不会进入 STE 分支。
- V5 的链式梯度成立，但原因是 `alpha=1.0` 且中间 `z_pred` 没有 detach。
- `step_weights` 被归一化，因此 V5 不等于每个 rollout step 都统一乘以 6 倍；有效权重是 `1.5 * w / sum(w)`。

V5 相对 V3 的有效 rollout step 系数:

| hop | V3 weights `[1.3,1.2,1.1,1.0]`, lambda=0.25 | V5 weights `[0.8,1.0,1.5,2.5]`, lambda=1.5 | 相对变化 |
|---|---:|---:|---:|
| D50->D20 | 0.0707 | 0.2069 | 2.93x |
| D20->D10 | 0.0652 | 0.2586 | 3.97x |
| D10->D4 | 0.0598 | 0.3879 | 6.48x |
| D4->NORMAL | 0.0543 | 0.6466 | 11.90x |

因此 V5 的真实机制不是“整体 rollout 6x”这么简单，而是明显加强 tail open-loop 梯度，同时相对削弱 hop0 的 rollout 权重。

## 4. 反驳点四: V5 并没有“阶段 I alpha=0 无干预区间”

Claude 文档中有“阶段 I（alpha=0 无干预区间）”的说法，这不适用于 V5。

| 文件 | 行号 | 机制 |
|---|---:|---|
| `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 139-149 | V5 从开始就是 `warmup_ratio=0`, `ramp_ratio=0`, `alpha_start=1.0`, `alpha_end=1.0`, `lambda_start=1.5`, `lambda_end=1.5` |
| `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml` | 115-127 | V3 baseline 才是 `alpha_start=0`, `alpha_end=1`, `lambda_start=0.02`, `lambda_end=0.25` |

所以 V5 的早期 `86800 -> 92800` 不是“无干预适应期”，而是已经在 rollout-heavy 配置下训练。

更准确的说法是:

- V5 在 resume 后立即施加 `lambda_roll=1.5`、`alpha=1.0` 的 open-loop rollout loss。
- 早期指标没有大幅恶化，只能说明短期 rolling-val 上没有明显爆炸。
- 不能把早期段解释为 alpha=0 的 baseline 参考段。

## 5. 反驳点五: `val_pair_total` 后续回落，不支持不可逆盆地假说

Claude 把 `val_pair_total` 从 `0` 到 `~2e-4` 的变化解释为模型被推出 GT-optimal manifold。

这个解释至少缺少两个证据:

1. `93200` 是不同 validation window。
2. 后续 `val_pair_total` 会明显回落。

V5 rolling-val 后续片段:

| step | window_start | val_pair_total | normal | select |
|---:|---:|---:|---:|---:|
| 93200 | 0 | 0.000231 | 0.000158 | 0.000614 |
| 101200 | 1280 | 0.000046 | 0.000414 | 0.001668 |
| 102800 | 1536 | 0.000014 | 0.000174 | 0.000646 |
| 103600 | 1664 | 0.000018 | 0.000219 | 0.000822 |

如果是不可逆地离开 GT-optimal basin，至少需要证明在同一固定验证集或 full-val 上 `val_pair_total` 持续不可恢复。当前 rolling-val 不满足这个条件。

## 6. 仍然支持 Claude 的部分

本审计不是说 V5 成功。相反，当前 V5 仍然有以下风险:

| 风险 | 证据 | 代码定位 |
|---|---|---|
| open-loop/pred-state 压力确实增强 | V5 `alpha=1.0`, `lambda_roll=1.5`，后续 hop 使用前一 hop 预测 | `pet_lr/rollout_first_hop.py:96-104`, `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml:142-149` |
| rollout tail 梯度明显加强 | V5 tail effective coefficient 是 V3 的约 11.9x | `pet_lr/rollout_first_hop.py:106-108`, `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml:157-161` |
| pair anchor 相对较弱 | 总 loss 中 pair 只是不加权单项，V5 rollout 权重大幅增加 | `train_first_hop.py:1907-1910` |
| image_aux 被降低 | `lambda_img 0.12 -> 0.08` | `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml:163-170` |
| LR schedule 与 `max_steps` 耦合 | launcher 把 `max_steps` 写成 `resume_step + 50000`，scheduler 用该值算 cosine | `scripts/launch_v5_rollout_heavy.sh:31-43`, `train_first_hop.py:1795-1802` |

因此可以保留的结论是:

> V5 当前没有可靠增益证据，且其 tail-heavy open-loop BPTT 机制可能对 hop0/GT one-step anchor 不利。这个方向需要更严格 fixed/full-val 复评和更小步长的消融，而不是仅凭 rolling-val 继续声称有效。

## 7. 需要优先介入的代码/实验点

### 7.1 评估协议优先修正

| 介入 | 目标 | 代码定位 |
|---|---|---|
| full-val rerank | 消除 rolling window 混淆 | `train_first_hop.py:746-747`，`max_val_batches <= 0` 可评全量 |
| fixed-window eval | 对比相同子集上的 checkpoint 演化 | `train_first_hop.py:751-756` 当前只有 `head` / `rolling`，可增加固定 start |
| 统一 `best_metric_terms` | 防止 V4/V5 select 不可比 | `train_first_hop.py:1041-1077`，V4/V5 config 的 `best_metric_terms` |
| 记录窗口 id 与样本数 | 已有，但报告中必须显式使用 | `train_first_hop.py:1032-1037` |

最小复评集合:

| 模型 | ckpt | 目的 |
|---|---|---|
| V3 200K | `best.pt` | 公平 baseline |
| V4 SF | `best.pt` / `last.pt` | SF 路线真实 full-val 效果 |
| V5 rollout-heavy | `best.pt` / latest checkpoint | 验证 V5 是否真的无效 |
| null-control | V3 best resume 但不改 loss 继续训练 | 分离继续训练/LR schedule 与 V5 loss 的影响 |

### 7.2 如果 full-val 后 V5 仍无增益

优先级建议:

| 优先级 | 修改 | 理由 | 定位 |
|---:|---|---|---|
| 1 | 加 null-control resume | 先确认失败来自 V5 loss，而不是继续训练或 LR schedule | `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml`, `scripts/launch_v5_rollout_heavy.sh` |
| 2 | 降低 `lambda_roll` 或做 ramp | 避免 resume 后立即 tail-heavy BPTT | `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml:139-149` |
| 3 | 恢复/增强 image_aux | 保护 hop0 图像锚定 | `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml:163-170` |
| 4 | rollout detach/teacher variant | 降低自回归链式梯度对前面 hop 的扰动 | `pet_lr/rollout_first_hop.py:96-104` |
| 5 | scheduler_total_steps_override | 避免 launcher 改 `max_steps` 同时改变 LR 曲线 | `scripts/launch_v5_rollout_heavy.sh:31-43`, `train_first_hop.py:1795-1802` |

## 8. 最终立场

本审计建议对 Claude 文档做如下修正:

1. 删除或降级“`93200` 相变”表述。
2. 删除“任何 loss 调整都会推出窄盆地”的强推断。
3. 保留“V5 当前 rolling-val 没有可靠改善证据”的结论。
4. 保留“V4/V5 都在加强 pred-state/open-loop 压力”的弱机制解释。
5. 增加关键限定: V4 是 pred-state 单步 pair loss，V5 是 open-loop rollout BPTT，二者不是同构梯度路径。
6. 在所有 V4/V5 对比中统一 `best_metric_terms` 或直接使用 full-val objective。
7. 下一步先做 full-val / fixed-window rerank，不要继续用 rolling-val 单点做机制判断。

一句话版本:

> Claude 对“V5 未证明有效”的谨慎态度是对的，但把 `93200` 解释为相变、把 V4/V5 同步现象解释为窄盆地失败，是被 rolling-val window 回卷和 select-score 不同定义混淆后的过度推断。真正需要介入的是评估协议和 open-loop/tail-heavy 梯度设计，而不是把当前日志直接定性为不可逆相变。
