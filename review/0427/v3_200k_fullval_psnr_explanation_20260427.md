# V3 200K Full-Val PSNR 结果说明

生成时间：2026-04-27 18:23 CST

## 结论

这组 `D50/D20/D10/D4/NORMAL` PSNR 是 **V3 200K 实验 best.pt 在完整 val 集上的 full-val eval 结果**，不是训练过程中的 rolling-val，也不是抽样子集。

关键证据：

- eval JSON: `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v3_best/first_hop_224_val_clip3_eval.json`
- eval CSV: `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval/v3_best/first_hop_224_val_clip3_eval.csv`
- eval log 摘要: `review/0427/logs_eval/v3_200k_best_fullval_clip3_summary.log`
- eval raw log extract: `review/0427/logs_eval/v3_200k_best_fullval_clip3_extract.log`
- `split = val`
- `max_slices = 0`
- `num_eval_slices = 7403`
- `psnr_metric = src.utils.metrics.calc_psnr_clip3`
- `decode_mode = both`

因此这里的均值是对全部 `7403` 个 val slices 计算的 mean PSNR。我们当前没有 test 集，val 集是正确评估集。

## 评估对象

使用的 checkpoint：

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3/best.pt
```

该 `best.pt` 来自 V3 200K 训练，目前对应训练过程中的 best rolling-val step：

```text
best_rolling_val_step = 86800
best_rolling_val_select_score = 0.0005923646864545162
```

注意：V3 200K 训练当前仍在继续，约在 `171.2K/200K` 附近；但截至本文档生成时，后续 checkpoint 没有刷新 `best.pt`。

## 表格字段含义

这个表不是“每个输入剂量都直接推到 NORMAL”的表，而是 **从 D50 开始的 open-loop 级联 rollout，每个中间节点单独 decode，并与对应 GT 节点比较 PSNR**。

| 字段 | 实际含义 | 是否包含 transport |
|---|---|---:|
| `D50` | 直接 decode 初始 `z_D50`，与 `GT D50 image` 比 PSNR | 否 |
| `D20` | `D50 -> Pred D20`，decode 后与 `GT D20 image` 比 PSNR | 是，1 hop |
| `D10` | `D50 -> Pred D20 -> Pred D10`，decode 后与 `GT D10 image` 比 PSNR | 是，2 hops |
| `D4` | `D50 -> Pred D20 -> Pred D10 -> Pred D4`，decode 后与 `GT D4 image` 比 PSNR | 是，3 hops |
| `NORMAL` | `D50 -> Pred D20 -> Pred D10 -> Pred D4 -> Pred NORMAL`，decode 后与 `GT NORMAL image` 比 PSNR | 是，4 hops |

所以：

- `D50 = 42.62 dB` 主要表示 D50 latent 经过 decoder 的重建质量，不能当作 transport 到 NORMAL 的结果。
- `D20 = 35.54 dB` 是第一跳 `D50 -> D20` 的图像 PSNR。
- `D10 = 35.98 dB` 是两跳级联后的 `Pred D10` 图像 PSNR。
- `NORMAL = 36.87 dB` 才是完整从 D50 级联到 NORMAL 后的最终图像 PSNR。

## Full-Val PSNR 结果

PSNR 使用封装好的 `calc_psnr_clip3`，即按 `clip_max=3` 方案计算。

| rollout node | mean PSNR | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| D50 | 42.6224 | 7.2563 | 31.1373 | 72.0684 | 7403 |
| D20 | 35.5444 | 8.8713 | 22.9026 | 75.7600 | 7403 |
| D10 | 35.9768 | 8.3883 | 23.5885 | 73.7178 | 7403 |
| D4 | 36.5579 | 8.0042 | 24.7518 | 71.4307 | 7403 |
| NORMAL | 36.8650 | 7.5454 | 25.1003 | 67.2847 | 7403 |

## 与训练 rolling-val 的区别

训练日志里的 `val_select_score` / `val_chain_*_mse` 是训练期间的 rolling-window validation，通常只评估一个窗口，不等价于 full-val PSNR。

截至本文档生成时的训练状态：

| 指标 | step | value |
|---|---:|---:|
| best rolling-val `val_select_score` | 86800 | 0.0005923647 |
| latest observed rolling-val `val_select_score` | 171200 | 0.0011259580 |
| latest train step | 171200 | still running |

这说明：

- V3 的 `best.pt` 当前仍来自 86.8K step。
- 后半程训练还在继续，但 rolling-val 没有继续刷新 best。
- 当前这个 full-val PSNR 表只代表 `best.pt`，不代表未来训练结束后的 `last.pt` 或 `step_200000.pt`。

## 读取方式建议

如果用于对外汇报，建议写成：

> V3 best checkpoint was evaluated on the full validation set (`7403` slices, `max_slices=0`) using `calc_psnr_clip3`. Starting from D50, the model performs open-loop latent rollout through D20, D10, D4, and NORMAL; each reported PSNR is computed after decoding the corresponding rollout node and comparing it to the matching GT image at the same timepoint.

不要写成：

> D50 到 NORMAL 的 PSNR 是 42.62 dB。

正确说法是：

> D50 latent direct decoder reconstruction is 42.62 dB, while the final D50-to-NORMAL open-loop rollout PSNR is 36.87 dB.
