# Idea2 实验记录与失败原因分析（2026-04-14）

## 1. 实验目标
验证 `Idea2: Alpha-CCT`（CCT 课程化 + 冲突阻尼）是否能在 50k 预算下稳定优于 `chainstable` 基线，尤其改善 `D50->D20` 并带动 `D50->...->NORMAL`。

## 2. 运行记录（可复现）

- 代码分支：`idea2_alpha_cct_gpu3`
- 关键提交：`5209d4d`
- 训练配置：
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml)
- 训练输出目录：
  [first_hop_224_50k_idea2_alpha_cct_seed43_gpu3](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3)
- 训练日志指标：
  [metrics.jsonl](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3/metrics.jsonl)

### 全量评估（val 全集，7403 slices）
- best checkpoint 评估目录：
  [first_hop_224_50k_idea2_alpha_cct_seed43_gpu3_eval_clip3_best_full](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3_eval_clip3_best_full)
- last checkpoint 评估目录：
  [first_hop_224_50k_idea2_alpha_cct_seed43_gpu3_eval_clip3_last_full](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3_eval_clip3_last_full)
- 基线对照诊断：
  [chainstable50k_best_full](/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/chainstable50k_best_full/hop_difficulty_clip3_val_chainstable50k_best_full.json)
  [strict_best_full](/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/strict_best_full/hop_difficulty_clip3_val_strict_best_full.json)

## 3. 结果总表（PSNR clip3, val full）

| Model | D50 | D20 | D10 | D4 | NORMAL | tail_avg(D20~N) | all_avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_best_full | 42.6224 | 35.3120 | 35.5490 | 35.9845 | 36.0810 | 35.7316 | 37.1098 |
| chainstable_best_full | 42.6224 | 35.4881 | 35.8620 | 36.4207 | 36.7422 | 36.1283 | 37.4271 |
| idea2_best_full | 42.6224 | 35.4947 | 35.8483 | 36.4111 | 36.7104 | 36.1161 | 37.4174 |
| idea2_last_full | 42.6224 | 35.5036 | 35.8670 | 36.4294 | 36.7733 | 36.1433 | 37.4392 |

### 关键对比

1. `idea2_best_full - chainstable_best_full`
- D20: `+0.0066 dB`
- D10: `-0.0137 dB`
- D4: `-0.0096 dB`
- NORMAL: `-0.0319 dB`
- tail_avg: `-0.0121 dB`
- all_avg: `-0.0097 dB`

2. `idea2_last_full - chainstable_best_full`
- D20: `+0.0156 dB`
- D10: `+0.0050 dB`
- D4: `+0.0087 dB`
- NORMAL: `+0.0310 dB`
- tail_avg: `+0.0151 dB`
- all_avg: `+0.0121 dB`

结论：如果只看 `best.pt`，idea2 略退化；但 `last.pt` 实际超过 chainstable。

## 4. 训练动态与失败表征

按 `metrics.jsonl` 的 val 事件统计：
- `best_by_select_step = 46800`, `val_select_score = 5.397e-4`
- `last_val_step = 50000`, `val_select_score = 6.836e-4`（较 best 恶化约 `+26.7%`）

`best_by_select -> last` 的变化：
- `val_chain_d20_mse`: `2.360e-4 -> 2.673e-4`（恶化）
- `val_chain_normal_mse`: `1.645e-4 -> 2.177e-4`（恶化）
- `val_chain_tail_mse`: `1.851e-4 -> 2.315e-4`（恶化）
- `val_pair_total`: `1.856e-4 -> 1.238e-4`（继续下降）
- `val_rollout_total`: `9.212e-4 -> 5.909e-4`（继续下降）
- `val_hop0_img_total`: `7.034e-3 -> 1.027e-2`（显著上升）
- `val_cct_total`: `4.453e-4 -> 2.921e-4`（下降）

解释：后期出现“局部训练目标继续下降，但链路终端指标反向变差”的目标拉扯。

## 5. 失败原因分析（结合双 agent 审计）

## 原因 A（高优先级）：`matching` 分支有效梯度不足
- 代码位置：
  [train_first_hop.py:920](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:920)
  [train_first_hop.py:927](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:927)
- 配置位置：
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:150](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:150)
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:162](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:162)
- 现象：matching 占 CCT 配额，但梯度贡献弱，导致“CCT 数值不低，实际约束偏弱”。

## 原因 B（高优先级）：课程进度绑定 rollout alpha，前期一致性权重过低
- 代码位置：
  [train_first_hop.py:762](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:762)
  [train_first_hop.py:773](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:773)
  [train_first_hop.py:1990](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1990)
- 配置位置：
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:109](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:109)
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:157](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:157)
- 现象：早期 `cct_progress` 长时间接近 0，一致性约束启动偏慢。

## 原因 C（中优先级）：conflict_damp 偏强，持续压制 consistency
- 代码位置：
  [train_first_hop.py:936](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:936)
  [train_first_hop.py:948](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:948)
- 配置位置：
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:166](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:166)
- 现象：`val_cct_conflict_ratio` 常 > 1.5，`val_cct_conflict_damp` 常 < 0.85。

## 原因 D（中优先级）：best checkpoint 选择受 rolling-window 偏差影响
- 代码位置：
  [train_first_hop.py:1024](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1024)
  [train_first_hop.py:1073](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:1073)
  [train_first_hop.py:2572](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:2572)
- 配置位置：
  [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:45](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml:45)
- 证据：`last_full` 明显优于 `best_full`，说明“选点误差”是退化来源之一。

## 原因 E（结构盲区）：CCT 对 hop0 consistency 天然为 0
- 代码位置：
  [train_first_hop.py:884](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:884)
  [train_first_hop.py:901](/home/qujiaxiang/project/PET_LatentResidual/train_first_hop.py:901)
- 现象：第 0 步 teacher/pure 输入相同，`cct_cons_step_0` 先天不提供约束，首跳仍主要靠 pair/rollout。

## 6. 结论与下一步建议（最小改动）

结论：
- Idea2 不是“完全无效”，而是当前参数与选择策略下，`best.pt` 没有稳定超过 chainstable。
- full-eval 显示 `last.pt` 已可小幅超过 chainstable，说明方法方向可救，主要是训练配方与checkpoint选择问题。

建议按低风险顺序执行：
1. 将 `detach_teacher_matching=false`（或直接把 `matching_scale_*` 设为 0 做 pure consistency 对照）。
2. `progress_mode` 改为 `schedule`，并单独给 `warmup_steps/ramp_steps`（如 `500/1000`）。
3. 放松冲突阻尼：`conflict_target=1.6`, `conflict_gamma=0.3`, `conflict_min_scale=0.7`。
4. `val_window_mode` 改为 `head`（或固定窗口），并在训练结束强制比较 `best_full` vs `last_full` 后再定稿。

## 7. Agent 调用记录
- Quant 复盘 agent：`Pauli`（完成）
- 机制归因 agent：`James`（完成）
- 本报告基于两位 agent 的结果与本地补充 full-eval（`last_full`）整合。
