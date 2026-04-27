# 0427 run_me 实验包修复说明

日期: 2026-04-27  
目标: 修复 Claude `review/0427/run_me` 实验设计中的执行/归因问题，并准备运行 P0。

## 1. 修复内容

### 1.1 Full-Val 脚本补齐 best / last / latest-step

文件: `review/0427/run_me/01_fullval_eval.sh`

修复前:

- 只评估 V3 `best.pt`、V5 `best.pt`、V5 最新 `step_*.pt`。
- 没有显式评估 `last.pt`。
- 注释中说“使用相同 config (V5 config)”，但实际 V3/V5 使用各自 config。
- 输出说明使用了不存在的字段名 `chain_normal_psnr` / `transport_avg_psnr`。

修复后:

- 对 V3: 评估 `best.pt`，若存在则评估 `last.pt`。
- 对 V5: 评估 `best.pt`，若存在则评估 `last.pt`，并评估最新 `step_*.pt`。
- 每个 checkpoint 使用各自训练 config 构建 dataset/model。
- 明确输出 JSON 路径与真实字段名:
  - `summary_psnr_clip3.NORMAL.mean`
  - `summary_psnr_clip3.D20.mean`
  - `summary_psnr_clip3.NORMAL_raw.mean` / `D20_raw.mean` when `decode-mode=both`

## 2. Null-Control 的 full-val 收口

新增文件: `review/0427/run_me/05_fullval_null_control.sh`

原因:

- Null-control 训练过程中仍然使用 rolling-val，只能做健康监控。
- 本轮争论的核心正是 rolling-val window 混淆，因此 null-control 不能再用 rolling-val 结论做最终归因。

新增脚本行为:

- 对 null-control 输出目录 `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v5_null_control` 中的 `best.pt`、`last.pt`、最新 `step_*.pt` 做 full-val。
- 输出到 `/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval_null_control`。

同时更新 `review/0427/run_me/03_null_control.sh`:

- 启动后写出 PID 文件: `review/0427/logs_train/v5_null_control_gpu${GPU_ID}.pid`。
- 明确提示训练完成后必须运行:

```bash
bash review/0427/run_me/05_fullval_null_control.sh ${GPU_ID}
```

## 3. README 修复

文件: `review/0427/run_me/README.md`

修复内容:

- 将 full-val 的关键字段改为真实 JSON 字段。
- 说明 `eval_first_hop_224_clip3.py` 输出 decoded PSNR / seam 指标，不输出 latent MSE。
- 说明 latent MSE 与 TF-vs-rollout gap 由 `scripts/diagnose_tf_rollout_gap.py` 输出。
- 增加 null-control full-val 脚本到文件清单和执行顺序。
- 明确 null-control 的归因结论必须基于 full-val，不基于 rolling-val。

## 4. velocity_rebalance 解释边界

README 中新增说明: 当前 `velocity_rebalance` 不应被解释为有效机制变量。

原因:

- 当前 checkpoint `target_normalize=false`。
- config 使用 `endpoint_dt_normalize=true`。
- 模型定义为 `z_pred = z_src + v * dt`。

因此:

```text
endpoint_err = ((z_pred - z_dst) / dt)^2
             = (v - (z_dst - z_src) / dt)^2
             = velocity_err
```

所以 `loss_endpoint / loss_velocity = 1`，`vel_reb = 1.0`。训练日志也显示 V4/V5 中 `vel_reb` 全程为 `1.0`。

Null-control 保留 `velocity_rebalance.enabled=true` 只是为了匹配 V3 config，不代表该机制在当前公式下有实际作用。

## 5. 修复后的 P0 运行方式

本次 P0 使用 GPU2，因为当前 V5 训练会被中断并释放 GPU2。

顺序:

```bash
bash review/0427/run_me/01_fullval_eval.sh 2
bash review/0427/run_me/02_pathA_baseline.sh 2
```

日志位置:

```text
review/0427/logs_eval/p0_01_fullval_gpu2.log
review/0427/logs_eval/p0_02_pathA_baseline_gpu2.log
```

输出位置:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0427_fullval
/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_v3_200k_best
```

## 6. 剩余注意事项

- 如果 V5 `last.pt` 不存在，`01_fullval_eval.sh` 会跳过并继续评估 `best.pt` 和最新 `step_*.pt`。
- 当前 V5 目录已有 `best.pt` 和 `step_100000.pt`，但未看到 `last.pt`。
- P0 不会启动 null-control；null-control 属于 P1。
