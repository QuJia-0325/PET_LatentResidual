# GPU1 训练完成说明（N1 / pixenc ablation）

时间：2026-04-23 01:19:22 +0800  
分支：`foc_lite_hop0`

## 1) 对应任务

- 任务名：`first_hop_224_50k_pixenc_ablation`
- 运行日志：
  - `review/0423/logs_completed/first_hop_224_50k_pixenc_ablation_train_gpu1_resume_20260422_1109.log`
- 输出目录：
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation`

## 2) 完成状态（关键证据）

来自日志关键行：

- `... at step=46400`：出现最后一次 best 更新
  - `new best val_multi_objective ... =0.000525 at step=46400`
- `step=50000`：达到 max step
  - `step=50000 ... val_select_score=0.000677`
- 最终完成标记
  - `Training done. Outputs at: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation`

结论：**GPU1 上该训练已正常完成（到 50k 步并正常落盘）。**

## 3) 关于 `resume` 的说明

本次 GPU1 训练是 `resume` 续训（不是 fresh start）：

- 日志中存在：
  - `[startup] resume from: .../step_010000.pt`
  - `[resume] loaded step=10000, ...`

因此该 run 的有效起点是 `step_010000.pt`，而不是从 0 重新开始。

## 4) 附件

已将完整训练日志附在仓库中，便于 Claude/同事直接审阅：

- `review/0423/logs_completed/first_hop_224_50k_pixenc_ablation_train_gpu1_resume_20260422_1109.log`

