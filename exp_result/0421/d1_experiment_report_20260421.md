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
- 结构化摘要：
  - `artifacts/d1_seam_refiner_training_summary_20260421.json`
  - `artifacts/d1_seam_refiner_val_curve_20260421.csv`
- 关键点表：
  - `results/d1_keypoints_20260421.csv`

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

## 5. 建议
1. 对 D1 立即补齐 `full-val eval`，至少评估三者：`best.pt`、`best_d1.pt`、`last.pt`。
2. 对外汇报建议主用 `best.pt`（transport 主线），同时附 `best_d1.pt` 作为 D1 lane 证据。
3. 训练日志体积偏大（包含 tqdm 连续进度条）；后续建议保持 `TQDM_DISABLE=1` 或降低可视化频度，避免日志膨胀。

## 6. 备注
- 本报告结论基于训练期 val 窗口指标（`metrics.jsonl`），不替代 full-val 评估结论。
