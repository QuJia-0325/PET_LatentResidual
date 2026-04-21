# D1 实验结果报告（2026-04-21）

## 1. 实验对象
- 实验名：`first_hop_224_50k_seam_refiner`（D1）
- 输出目录：`/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_seam_refiner`
- 训练结束：`step=50000`（日志末尾显示 `Training done`）

## 2. 归档内容（本目录）
- 训练配置：
  - `logs_train/d1_seam_refiner_config.yaml`
- 训练日志：
  - `logs_train/d1_seam_refiner_metrics.jsonl`
  - `logs_train/d1_seam_refiner_train_gpu3_20260420_150733.log`
- 评估日志（full-val）：
  - `logs_eval/d1_seam_refiner_best_fullval_eval_20260421.log`
  - `logs_eval/d1_seam_refiner_last_fullval_eval_20260421.log`
- full-val 结果：
  - `results/d1_seam_refiner_best_fullval_eval_20260421.json`
  - `results/d1_seam_refiner_best_fullval_eval_20260421.csv`
  - `results/d1_seam_refiner_last_fullval_eval_20260421.json`
  - `results/d1_seam_refiner_last_fullval_eval_20260421.csv`
- 结构化摘要：
  - `artifacts/d1_seam_refiner_training_summary_20260421.json`
  - `artifacts/d1_seam_refiner_val_curve_20260421.csv`
  - `artifacts/d1_fullval_best_vs_last_summary_20260421.json`
- 关键点表：
  - `results/d1_keypoints_20260421.csv`
  - `results/d1_fullval_best_vs_last_metrics_20260421.csv`

## 3. 原始数据表（关键 checkpoint 点）

| Point | step | val_select_score | val_d1_select_score | guard_ok | val_chain_normal_mse | val_hop0_img_seam |
|---|---:|---:|---:|---:|---:|---:|
| best_transport | 46400 | 0.000596130 | 0.001944582 | 1.0 | 0.000199077 | 0.001944582 |
| best_d1_any | 44800 | 0.000658383 | 0.001667064 | 0.0 | 0.000220320 | 0.001667064 |
| best_d1_guard | 46400 | 0.000596130 | 0.001944582 | 1.0 | 0.000199077 | 0.001944582 |
| final | 50000 | 0.000748478 | 0.003527198 | 0.0 | 0.000246821 | 0.003527198 |

补充统计：
- val 评估点数：`130`（从 step `400` 到 `50000`）
- `guard_ok=1` 的点数：`14`

## 4. 关键发现
1. **D1 训练已完成且保存链路正常**
   - 已产出 `best.pt`、`best_d1.pt`、`last.pt`，并完成 `step_050000.pt`。

2. **当前 run 的“可用最佳点”在 step=46400**
   - `best_transport` 与 `best_d1_guard` 重合在 `46400`，说明在 guard 约束下，传输主目标与 D1 lane 并未冲突。

3. **绝对最小 seam 点（step=44800）被 guard 拦截**
   - `best_d1_any` 的 `val_d1_select_score` 更低（`0.001667064`），但 `guard_ok=0`，其 `val_chain_normal_mse=0.000220320` 超过 guard 限制，不能作为 D1 正式 best。

4. **训练末尾（50000）较最佳点有明显回退**
   - `val_select_score` 相对 best_transport 上升约 `25.56%`。
   - `val_d1_select_score` 相对 best_d1_guard 上升约 `81.39%`。
   - 结论：该 run 的可报告 checkpoint 应优先使用 `best.pt`/`best_d1.pt`，不建议直接用 `last.pt` 做主结果。

## 5. Full-val 结果（best.pt vs last.pt）

| Metric | best.pt | last.pt | Delta (last - best) |
|---|---:|---:|---:|
| PSNR D50 | 38.591577 | 38.635481 | +0.043904 |
| PSNR D20 | 35.923970 | 35.903491 | -0.020479 |
| PSNR D10 | 36.094169 | 36.083271 | -0.010898 |
| PSNR D4 | 36.146838 | 36.109041 | -0.037797 |
| PSNR NORMAL | 36.449659 | 36.437741 | -0.011918 |
| PSNR transport_avg (D20,D10,D4,NORMAL) | **36.153659** | **36.133386** | **-0.020273** |
| PSNR all_avg (5tp) | 36.641243 | 36.633805 | -0.007438 |
| seam_consistency transport_avg (lower better) | **0.003611892** | **0.003649164** | **+0.000037272** |
| extended_seam transport_avg (lower better) | **0.002145011** | **0.002146932** | **+0.000001921** |

结论：
- full-val 上 `best.pt` 整体仍优于 `last.pt`（PSNR 主指标略优，seam 指标也略优）。
- 这与训练窗口结论一致：`last.pt` 不是本 run 的最优报告点。

## 6. 建议
1. D1 对外主结果继续采用 `best.pt`，`last.pt` 仅作训练终点参考。
2. 若要完整闭环 D1 claim，下一步补跑 `best_d1.pt` 的 full-val，并与 `best.pt/last.pt` 三者并列。
3. 训练日志体积偏大（包含 tqdm 连续进度条）；后续建议保持 `TQDM_DISABLE=1` 或降低可视化频度，避免日志膨胀。

## 7. 备注
- 本报告包含训练期 val 窗口分析 + full-val（best/last）分析。
