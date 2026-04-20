# T1 实验结果报告（2026-04-20）

## 1. 实验目标
对比 T1 的两组 `image_aux lambda_max`：
- `T1a`: `lambda_max=0.18`
- `T1b`: `lambda_max=0.25`

在相同训练框架下，比较训练收敛表现与 `best.pt` 的 full-val 评估结果，给出后续建议。

## 2. 归档内容（已放入 0420）
- 训练日志（以 `metrics.jsonl` 为主）：
  - `logs_train/t1a_lam018_metrics.jsonl`
  - `logs_train/t1b_lam025_metrics.jsonl`
- 训练配置：
  - `logs_train/t1a_lam018_config.yaml`
  - `logs_train/t1b_lam025_config.yaml`
- 评估日志：
  - `logs_eval/t1a_lam018_best_fullval_eval_20260420.log`
  - `logs_eval/t1b_lam025_best_fullval_eval_20260420.log`
- full-val 结果（JSON/CSV）：
  - `results/t1a_lam018_best_fullval_eval_rerun.json`
  - `results/t1a_lam018_best_fullval_eval_rerun.csv`
  - `results/t1b_lam025_best_fullval_eval_rerun.json`
  - `results/t1b_lam025_best_fullval_eval_rerun.csv`
- 附加审阅材料：
  - `artifacts/t1_fullval_summary_20260420.json`
  - `artifacts/t1a_lam018_val_curve.csv`
  - `artifacts/t1b_lam025_val_curve.csv`

## 3. 训练侧结果
- T1a (`lam=0.18`):
  - best `val_select_score`: **0.00052590** @ step 46400
  - last `val_select_score`: **0.00068899** @ step 50000
- T1b (`lam=0.25`):
  - best `val_select_score`: **0.00053324** @ step 46400
  - last `val_select_score`: **0.00069342** @ step 50000

训练指标上，T1a 略优于 T1b。

## 4. Full-val（best.pt）对比

| Metric (PSNR clip3) | T1a (0.18) | T1b (0.25) | Delta (T1b - T1a) |
|---|---:|---:|---:|
| D50 | 42.622378 | 42.622378 | +0.000000 |
| D20 | 35.567297 | 35.550385 | -0.016912 |
| D10 | 35.918870 | 35.881714 | -0.037156 |
| D4 | 36.475285 | 36.431085 | -0.044200 |
| NORMAL | 36.754551 | 36.662231 | -0.092321 |
| transport_avg (D20,D10,D4,NORMAL) | **36.179001** | **36.131354** | **-0.047647** |
| all_avg (5tp) | 37.467676 | 37.429559 | -0.038118 |

结论：`lambda_max=0.25` 相比 `0.18` 出现系统性小幅退化，尤其在 `NORMAL` 与低剂量链路后段更明显。

## 5. 建议
1. **T1 主线建议采用 `lambda_max=0.18`**（当前最稳健，full-val 更优）。
2. 若还要继续探索，可在 `0.15~0.22` 做更细粒度 sweep（建议步长 `0.02`），不建议再上推到 `0.25+`。
3. 论文/汇报口径建议优先使用 `transport_avg` 作为主指标，并附 `D20/D10/D4/NORMAL` 分项，能清楚体现 `0.25` 的退化模式。
4. 如需进一步公平性对比，可补跑一次 `last.pt` full-val（两组都跑），检查“训练终点模型”的结论是否一致。

## 6. 复现命令（本次评估日志对应）
```bash
# T1a
CUDA_VISIBLE_DEVICES=1 TQDM_DISABLE=1 \
python -u eval_first_hop_224_clip3.py \
  --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_lam0_18.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_lam018/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:0 \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/t1a_lam018_best_fullval_20260420_rerun

# T1b
CUDA_VISIBLE_DEVICES=2 TQDM_DISABLE=1 \
python -u eval_first_hop_224_clip3.py \
  --config configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_lam0_25.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_lam025/best.pt \
  --split val --max-slices 0 --batch-size 8 --device cuda:0 \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/t1b_lam025_best_fullval_20260420_rerun
```
