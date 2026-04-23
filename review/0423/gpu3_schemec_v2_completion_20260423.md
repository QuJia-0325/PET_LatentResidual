# GPU3实验完成记录（Scheme C V2）

- 日期：2026-04-23
- 实验名：`first_hop_224_50k_schemec_v2`
- 配置文件：`configs/pet_flow/pet_flow_first_hop_224_50k_schemec_v2.yaml`
- 运行设备：`GPU3`
- 状态：`已完成（Training done）`
- 外部输出目录：`/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2`
- 外部原始日志：`/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2_train_gpu3_20260422_1118.log`
- 仓库内归档日志：`review/0423/logs_completed/first_hop_224_50k_schemec_v2_train_gpu3_20260422_1118.log`

## 运行时间（按日志文件时间戳）

- 开始（Birth）：`2026-04-22 13:09:35 +0800`
- 结束（Modify）：`2026-04-23 04:56:08 +0800`
- 粗略总时长：约 `15h 46m`

## 关键训练结果

- 最优验证目标（`val_multi_objective` / `val_select_score`）：`0.0005261241 @ step 46400`
- 最终步：`step 50000`
- 最终 `val_select_score`：`0.0006713206`
- 最终 `val_rollout_total`：`0.0006280126`
- 最终链路 MSE：
  - `val_chain_d20_mse = 0.0002648519`
  - `val_chain_d10_mse = 0.0002463777`
  - `val_chain_d4_mse = 0.0002242082`
  - `val_chain_normal_mse = 0.0002126236`

## Checkpoint信息（可复现实验追踪）

- `best.pt`：`/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/best.pt`
  - SHA256: `d9296e4354838462cbbd697263213a18e8308779f9c0adacc03219a42e381cfa`
- `last.pt`：`/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2/last.pt`
  - SHA256: `60b8cb4f7b83e2b5b5518d75267005b30e5cfcee777467a4e03ef8a28d3a9b7f`

## 备注

- 日志中明确出现 `Training done. Outputs at: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_schemec_v2`。
- 该实验最佳点在 `step 46400`，晚期（`step 50000`）相对有轻微回升，后续对比建议优先使用 `best.pt` 做主结果，同时保留 `last.pt` 作为完整训练终点参考。
