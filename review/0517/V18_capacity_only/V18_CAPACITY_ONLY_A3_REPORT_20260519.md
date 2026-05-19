# V18-r32-capacity-only A3 完成报告

- 日期: 2026-05-19
- 分支: `foc_lite_hop0`
- 实验: `V18-r32-capacity-only / Stage C A3`
- 配置: `review/0517/V18_capacity_only/V18_capacity_only.yaml`
- 训练输出目录: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_capacity_only/run/first_hop_224_v18_capacity_only`
- 原始训练日志附件: `review/0517/V18_capacity_only/V18_capacity_only_train_gpu1_20260518_173445_tmux.log`
- 原始 metrics 快照: `review/0517/V18_capacity_only/V18_capacity_only_metrics_20260518.jsonl`
- KL drift / direct decode probe 附件目录: `review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/`

## 1. 本次完成的是哪个实验

已完成的是 `V18_capacity_only`，即 Round 13/14 A3 控制实验。它从 `V7.best.pt` 的 step 160000 warm-start，训练到 step 170000。

核心变量隔离:

| 项 | 设置 | 目的 |
|---|---:|---|
| LoRA rank | 32 | 与 V18 相同，保留 decoder capacity 变量 |
| LoRA blocks | last 2 decoder layers | 与 V18 相同 |
| `lambda_kl` | 0.0 | 关闭 KL pullback，只保留 capacity |
| `max_steps` | 170000 | V7 160K + 10K LoRA 训练 |
| `save_interval` | 5000 | 产生 step_165000 / step_170000 对照点 |
| image aux | 与 V18/V7 保持一致 | 不引入 image loss 变量 |

训练完成证据:

- 日志包含 `[startup] resume from:` 和 `[resume] loaded step=160000`。
- 日志中所有训练行均显示 `lambda_kl=0.0000`、`decoder_kl=0.000000`。
- 日志末尾包含 `Training done.`。
- 输出目录已有 `step_165000.pt`、`step_170000.pt`、`best.pt`、`last.pt`、`metrics.jsonl`。

## 2. 训练过程指标

`metrics.jsonl` 中记录了 200 条 train、25 条 rolling-val、2 条 full-val。full-val 使用完整 val 集，`val_chain_unique_slices=7403`。

| step | val_select_score | D20 MSE | D10 MSE | D4 MSE | NORMAL MSE | hop0 image total |
|---:|---:|---:|---:|---:|---:|---:|
| 165000 | 0.0009037353 | 0.0003304433 | 0.0002966598 | 0.0002630874 | 0.0002454921 | 0.0075349176 |
| 170000 | 0.0009039954 | 0.0003307622 | 0.0002969189 | 0.0002632126 | 0.0002454063 | 0.0075398990 |
| delta 170K-165K | +0.0000002601 | +0.0000003189 | +0.0000002592 | +0.0000001252 | -0.0000000858 | +0.0000049814 |

解释:

- 按 full-val `val_select_score`，170K 比 165K 略差，差异极小但方向不是提升。
- D20/D10/D4 chain MSE 均轻微变差，NORMAL MSE 轻微变好，幅度可忽略。
- 因此从 transport chain 指标看，A3 这 10K capacity-only 训练没有带来实质链式收益。
- `best.pt` 更可能对应 165K full-val 选择点，`last.pt` 对应 170K。

注意: 上表是 trainer 的 chain MSE，不是 canonical PSNR_clip3。A3 的机制判决使用下面的 direct `decode(z_GT)` PSNR_clip3 probe。

## 3. KL drift / direct decode full-val probe

我补丁扩展了 `tools/probe_v18_kl_drift.py`，新增可选参数:

- `--v18-step170k-ckpt`
- `--v18-cap-ckpt`
- `--v18-cap-config`

probe 协议:

- split: `val`
- slices: 7403 full-val
- metric: `src.utils.metrics.calc_psnr_clip3`
- 计算对象: `PSNR_clip3(decode_crop(z_GT), x_target)`
- 不经过 transport rollout，只测 decoder 在 GT latent manifold 上的直接重建质量。

结果:

| checkpoint | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | 46.6356 | 48.7436 | 50.8319 | 52.6341 |
| V18.best | 165000 | 46.6850 | 48.7993 | 50.9012 | 52.7314 |
| V18.step170k | 170000 | 46.7212 | 48.8408 | 50.9527 | 52.8027 |
| V18.last | 200000 | 46.7482 | 48.8630 | 50.9635 | 52.7980 |
| V18-cap.last | 170000 | 46.7221 | 48.8420 | 50.9542 | 52.8047 |

主判据: `V18-cap.last(170K) - V18.step170k`

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18-cap.last - V18.step170k | +0.0009 | +0.0012 | +0.0015 | +0.0020 |

相对 V7.best 的 direct decode GT-manifold 增益:

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18.best - V7.best | +0.0493 | +0.0557 | +0.0692 | +0.0973 |
| V18.step170k - V7.best | +0.0855 | +0.0972 | +0.1207 | +0.1686 |
| V18.last - V7.best | +0.1126 | +0.1194 | +0.1315 | +0.1639 |
| V18-cap.last - V7.best | +0.0865 | +0.0984 | +0.1223 | +0.1707 |

## 4. 判决

A3 的主判据是 matched-step: `V18-cap.last(170K) - V18.step170k`。

- NORMAL 差值只有 `+0.0020 dB`，远小于预注册阈值 `0.02 dB`。
- D20/D10/D4/NORMAL 四个 timepoint 都只有 `0.001 dB` 量级差异。
- 结论: 在 direct `decode(z_GT)` 指标上，关闭 KL 后的 capacity-only 几乎完全复现 V18@170K 的 GT-manifold 提升。

因此，本实验支持候选 B:

- V18 在 direct GT latent manifold 上相对 V7 的 PSNR 提升，主要可以由 LoRA decoder capacity 解释。
- 该 direct decode 提升不能作为 KL pullback 设计有效的证据。
- 但这不等价于 transport 级联已经改善；从本实验的 chain full-val MSE 看，capacity-only 对 chain 指标没有实质提升。

## 5. 对当前研究线的影响

1. A3 基本排除了“V18 direct decode(z_GT) 提升来自 KL pullback”的解释。
2. 如果继续讨论 KL，需要聚焦 `z_pred` 路径或 transport rollout 行为，而不是把 `decode(z_GT)` PSNR 提升当作 KL 成功证据。
3. LoRA capacity 对 decoder GT-manifold 有稳定正效应，但它不自动转化为 D50->D20->D10->D4->NORMAL chain 改善。
4. 后续更重要的验证应继续看 transport chain full-val、per-hop rollout、以及 direct decode 与 rollout 指标之间的脱钩。

## 6. 附件清单

- `V18_capacity_only_train_gpu1_20260518_173445_tmux.log`: 原始训练日志。
- `V18_capacity_only_metrics_20260518.jsonl`: 原始 metrics 快照。
- `kl_drift_with_cap_20260519_180206/KL_DRIFT_REPORT.md`: probe 自动报告。
- `kl_drift_with_cap_20260519_180206/KL_DRIFT_SUMMARY.json`: probe 聚合 JSON。
- `kl_drift_with_cap_20260519_180206/KL_DRIFT_PER_SLICE.csv`: full-val per-slice PSNR_clip3 明细。
- `kl_drift_with_cap_20260519_180206/probe.log`: probe 原始运行日志。
