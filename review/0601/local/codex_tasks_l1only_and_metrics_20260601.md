# Codex 任务包 #2 — L1-only 消融(新训练,已批准)+ 扩展医学影像指标

日期: 2026-06-01
分支: `foc_lite_hop0`
背景: 接 `review/0601/local/codex_tasks_seam_provenance_strata.md`。Task A(decoder provenance)已收官:
V7/V13/A4 共用同一冻结 `GeneralDecoder.decoder_pred` + hard `unpatchify`,novelty 边界 = "decoder-aware
latent transport",不是 decoder redesign。本包新增两件事。

## 监督判断(为什么做这两件事)

1. **论文中心 claim 目前是空头支票**: abstract + method §3.2 声称 "intensity-only(L1)insufficient,
   **can underperform a purely latent baseline**",但 canonical 结果表里**没有 L1-only 这一列**——
   现有 variant 全是复合损失 `L1+0.25·SSIM+0.10·seam` 的 λ 扫描。必须补一个 L1-only run 才能验证。
2. **headline 数字之前锚错**: canonical 写 "+0.11 vs V7"(两个复合 run 内部 λ 边际)。真正消融是
   **A4-mid vs V13(image_aux 开/关)= +0.40 dB @ NORMAL,win 93.6%,seam −22.7%,ext-seam −11.4%**
   (已本地复核 7403 slice)。
3. **单一 PSNR 不足以支撑"方法优越"**: 需补常见医学影像 full-reference 指标(SSIM/MS-SSIM/NRMSE/
   RMSE/MAE),但必须区分**训练对齐指标**(循环)和**独立指标**(可作优越性证据)。

## 执行顺序与论文策略

- **论文先按方案 B 写**: 把 "intensity-only 更差" 降级为 pulled-back metric 的**理论论证**
  (method Eq.3 已有),**不**在正文声称经验对照。等 Task C 成功再切换到方案 A 的经验表述。
- Task C(L1-only 训练)与 Task B-ext(扩展指标)可并行;Task C 产出的 checkpoint 跑完后并入 Task B-ext
  的指标面板与分层表。
- **全局硬约束**(承前): 仅 Task C 这一次新训练获批,**不得**再起其它新 run;PSNR 一律
  `src.utils.metrics.calc_psnr_clip3`;输出写 `/data_2/...` 与 `review/`,禁写仓库内 outputs;
  统计用 effect size + 64-slice block-bootstrap + win-rate,不写 patient-level p 值;不臆测、不编数。

---

## Task C — L1-only ablation(新训练,已批准的唯一一次)

```
仓库: PET_LatentResidual (branch: foc_lite_hop0)
目的: 验证论文中心 claim —— 在 λ 对齐下,只用 intensity-only(L1)image_aux 是否
      ≤ latent-only(V13)。这决定论文能否从方案 B(理论)切到方案 A(经验对照)。

基准配置(完整复制,逐键对齐,只改三处):
  review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml
当前 A4-mid 的 image_aux 块为:
  image_aux: { l1_weight: 1.0, ssim_weight: 0.25, seam_weight: 0.1,
               border_width: 14, border_weight: 2.0 }
  lambda_start: 0.08, lambda_max: 0.08

L1-only 改动(且仅此改动):
  - ssim_weight: 0.25 -> 0.0
  - seam_weight: 0.1  -> 0.0
  - l1_weight: 1.0 (保持), border_width/border_weight 保持, lambda 0.08 保持
  - 新 output_dir / run_name(勿覆盖 A4),seed=42 与 A4-mid 一致
  - RAE checkpoint / 冻结 / LoRA / epoch / 所有其它超参与 A4-mid 完全一致

跑完后:
  - 用 eval_first_hop_224_clip3.py 做 full-val(val, n=7403)eval,decode_mode=default
  - 与 V13、A4-mid 三方并列,报告全 timepoint PSNR_clip3 + aggregate seam/ext-seam
  - per-slice 落盘,便于做 win-rate / block-bootstrap

判定(写清楚,不要只给均值):
  - L1-only vs V13: ΔPSNR_NORMAL 的均值/64-block CI/win-rate。
    若 L1-only ≤ V13(CI 含 0 或为负)=> 支持方案 A,论文可切换为经验对照。
    若 L1-only > V13 => 方案 A 不成立,论文保持方案 B(理论论证),并据实说明
       "即使 intensity-only 也优于 latent-only,但 composite 进一步提升"。
  - L1-only vs A4-mid(composite): 量化 SSIM+seam 两项带来的增量。

产出: review/0601/server(或指定目录)下的三方对照表 + JSON/CSV + 训练/eval log + 一段结论。
```

---

## Task B-ext — 扩展医学影像指标面板 + seam 分层(合并原 Task B)

```
仓库: PET_LatentResidual (branch: foc_lite_hop0)
eval 入口: eval_first_hop_224_clip3.py
现状: per-slice CSV 仅含 psnr* 列;seam 只聚合。需扩展为完整指标面板 + per-slice 落盘。

1. 最小改动 evaluate()/write_csv,使 per-slice 行落盘以下全部指标(每 timepoint 一组):
   - PSNR_clip3(已有,口径不变)
   - SSIM、MS-SSIM
   - NRMSE、RMSE、MAE
   - seam(seam_consistency_loss, patch=14)、ext-seam(extended_seam_loss, zone=3)
   所有指标必须在与 PSNR_clip3 相同的 clip3 调窗域上计算(或显式记录各自域),保证可比、可复算。
   不改 PSNR 口径,不改训练。

2. 用现有 best.pt 重跑 full-val(val, n=7403):V13、A4-mid;Task C 完成后并入 L1-only。
   输出写 /data_2/...,仓库内禁写。已存在产物可复用,勿重复跑。

3. 循环性披露(关键,直接影响"优越性"能不能被审稿人接受):
   - 训练对齐指标(与 loss 项重叠 => 偏循环,只报告不作为优越性主证据):
     MAE/L1、SSIM(单尺度)、seam。
   - 独立指标(不在 loss 中 => 作为优越性主证据):
     PSNR_clip3、NRMSE、RMSE(MSE 族,loss 用 L1≠L2)、MS-SSIM(多尺度 vs 训练用单尺度 SSIM)。
   表格中必须明确标注每个指标属"training-aligned"还是"independent"。
   headline 优越性结论只能由 independent 指标承载;training-aligned 指标作为一致性佐证。

4. 每个指标做 per-slice 配对比较(A4-mid 减 V13,L1-only 减 V13):
   报告 mean、64-slice block-bootstrap 95% CI、win-rate。

5. seam 分层(回应历史 D2 SUSTAINED):
   - 用【V13(aux-off)】per-slice seam_NORMAL 作为与处理无关的 seam-risk 分层器,
     分位数切 low/mid/high(报告各桶 n 与阈值)。
   - 每桶报告 ΔPSNR、Δseam、Δext-seam、ΔNRMSE、win-rate;三桶全报告,不挑桶。
   - 预期:high-seam 桶 ΔPSNR 与 Δseam 最大。

6. 边界(不要越界):
   - 当前无 lesion/ROI/SUV mask,**不要**计算 CNR/SUV recovery/任务级指标,也不要声称病灶级优越。
   - 只做 full-reference fidelity 指标;若要 ROI 级证据,另行索取 mask。

产出:
- review/0601/server/seam_strata/seam_strata_summary.csv(timepoint, stratum, n, seam_threshold,
  psnr_V13, psnr_A4mid, dPSNR_mean, dPSNR_blkCI_lo/hi, win_rate, seam_V13, seam_A4mid,
  dseam_abs, dseam_pct, dnrmse 等)。
- review/0601/server/metrics_summary.csv:全指标 × {V13, A4-mid, L1-only} × 全 timepoint,
  每个 delta 附 CI/win-rate,并标注 independent / training-aligned。
- 一段 md 解读:哪些独立指标支持优越性、量级多少、与 +0.40 dB PSNR 的一致性;
  以及为何全图 PSNR 低估局部 seam 改善。
- 所有数字附原始 per-slice CSV 路径,可复算。
```

---

## 给论文写作组的衔接说明(本地,不推给 Codex 执行)

- experiments.tex headline 用 **A4-mid vs V13 = +0.40 dB**,不用 +0.11 vs V7。
- 指标表用 metrics_summary.csv,**优越性结论只引 independent 指标**(PSNR/NRMSE/RMSE/MS-SSIM);
  SSIM/MAE/seam 标为 training-aligned 一致性佐证。
- seam 的非循环防御三件套:PSNR↑ + ext-seam(对 GT)↓ + win-rate 全分层稳定。
- §1 的 "intensity-only underperforms" 暂按方案 B(理论)写;Task C 回来后据结果决定是否切方案 A。
- 仍未解外部风险:缺同 split 对外部已发表 PET 方法对照 —— 在补齐前不写任何 "SOTA / beats prior"。
- 选定性图避开纯噪声主导难例(如 slice 1100);按 average / hard / high-seam / failure 四类固定规则选。
