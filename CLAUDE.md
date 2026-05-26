# CLAUDE.md

This file provides guidance for engineering work in `/home/qujiaxiang/project/PET_LatentResidual`.

## Current Canonical State (2026-05-26)

**Read this section first. Several older sections below are historical and superseded.**

Current project direction:

- Paper headline: **A4-mid**, i.e. V7-style transport with `training.image_aux.lambda_start=lambda_max=0.08`, full image auxiliary loss, no LoRA, no KL.
- Mechanism hypothesis: stronger frozen-decoder image auxiliary supervision supplies decoder-aware pixel-space gradient and anchors `z_pred` to a well-decodable manifold (M1+M4).
- V18 / decoder LoRA / KL are **secondary ablations**, not the main story.
- X1-lite is running to test whether L1-through-decoder alone explains A4-mid.
- A4-mid-seed1337 is the current robustness replicate; it is exactly one seed replicate, not a seed sweep.

Canonical full-val `PSNR_clip3` anchors (val, n=7403):

| run | NORMAL | delta vs V7 | role |
|---|---:|---:|---|
| V13.best (`image_aux=0`) | 36.494330 | -0.286621 | true image_aux-off control |
| V7.best (`image_aux=0.04`) | 36.780951 | 0 | baseline |
| V14.best (`V7 seed=1337`) | 36.780632 | -0.000320 | seed perturbation at lambda=0.04 |
| A4-low.best (`image_aux=0.02`) | 36.700958 | -0.079994 | weak image_aux bracket |
| **A4-mid.best (`image_aux=0.08`)** | **36.893917** | **+0.112966** | **headline result** |
| V18.last | 36.842645 | +0.061693 | secondary decoder LoRA/KL ablation |
| X3.last (`image_aux=0.08` + LoRA, KL off) | 36.828787 | +0.047836 | non-additive LoRA result |

Important current constraints:

- Do **not** use stale A4 numbers such as `36.835 / +0.054`; canonical A4-mid is `36.893917 / +0.112966 vs V7`.
- Do **not** launch V18-clean / V19 / decoder rank sweep / X3-extend unless a new user-signed review explicitly overturns the stop rule.
- Do **not** launch extra A4 lambda points or seed sweeps. A4-mid-seed1337 is the only approved seed replicate.
- Patient IDs are not recoverable from current preprocessed artifacts; do not write patient-level significance claims.
- F0 paired-slice/bootstrap is not closed unless explicit F0 report artifacts exist.

Current source-of-truth docs:

- `review/0525/REVIEW_INTEGRATION_round18_20260525.md`
- `review/0525/REVIEW_INTEGRATION_round18_prep_20260525.md`
- `review/0525/REVIEW_INTEGRATION_round19_next_slot_after_X3_20260526.md`
- `review/0525/Round19-TODO.md`
- `review/0525/CODEX_TASK_ROUND19_A4_MID_SEED1337_20260526.md`

---

## Rules

**重要约束**：在计划和讨论模型架构时，没有用户明确批准，**不允许对代码进行任何修改**。必须先讨论方案，获得用户批准后再实施。

**指标硬规范**：PSNR 评估必须使用 `src.utils.metrics.calc_psnr_clip3`（先调窗到 3，再计算 PSNR）。

**工程边界**：本仓库是独立实验线，优先复用 `RAE` 现有稳定组件，不在未批准情况下改动 `RAE` 主干代码。

## Project Overview

`PET_LatentResidual` 是在 `RAE` 基础上的独立 PET 侧实验仓库，当前主线是 **224 first-hop latent transport**。

核心目标：

1. 保持 `latent-only` 的 4-hop 级联 transport 主状态不变。
2. 在第一跳 `D50 -> D20` 引入最小像素条件（hop0 pixel forcing）缓解第一跳 gap。
3. 在不引入 dual-state transport 的前提下，提升链路稳定性与可传递性。

当前主路径：

- rollout path: `D50 -> D20 -> D10 -> D4 -> NORMAL`
- 训练入口: `train_first_hop.py`
- 评估入口: `eval_first_hop_224_clip3.py`

## Current Architecture (Implemented)

### 1. Data Path

- 数据集类：`pet_lr/data_first_hop.py::PETFirstHopAligned4HopDataset`
- latent 输入：`/data_2/qujiaxiang/lowdose_pet_ct/latents_224/latents_{train|val}.pt`
- raw 图像输入：`/data_2/qujiaxiang/preprocessed_data_*.pt`
- 训练按 pair 样本组织：
  - `D50->D20`
  - `D20->D10`
  - `D10->D4`
  - `D4->NORMAL`
- rollout 监督链条：`[D50, D20, D10, D4, NORMAL]`

### 2. Alignment Gate (Fail-Fast)

- 对齐检查脚本：`check_alignment_224_clip3.py`
- 对齐审计文件：`alignment_audit_json`
- 训练启动前强制校验：
  - 审计文件存在
  - 指标名必须是 `src.utils.metrics.calc_psnr_clip3`
  - train/val 的 D50/D20/NORMAL 对齐信号为正

### 3. Model Path

- 主模型：`pet_lr/model_first_hop.py::PETFlowDiTFirstHop`
- 复用 backbone：`RAE` 的 `PETFlowDiTDHHopAware`
- 新增第一跳机制：
  - `FirstHopPixelEncoder`（弱像素编码器）
  - `HopResidualVelocityHead`（hop-specific 速度残差头）

前向关键实现：

1. hop0 像素注入（仅 `hop_idx == 0`）：

```python
z_in = z_src + gate_pix * pixel_encoder(x_src_img)
```

2. 速度预测与 hop residual：

```python
v_total_raw = v_shared + v_hop
z_pred = z_src + v_total * dt
```

其中 `v_hop = lambda_hop[hop] * head_hop(v_shared_sub)`。

### 4. Rollout Path

- 模块：`pet_lr/rollout_first_hop.py`
- 训练 rollout 损失：`rollout_multistep_losses_first_hop`
- 规则：
  - 第 0 步可用 `x_rollout_first`（D50 图像）
  - 后续 hop 严格 latent-only
- 支持 GT/pred 混合推进：`alpha` 调度 + `straight_through` 控制

### 5. Loss Design (Current)

总损失：

```text
L_total = L_pair + lambda_roll * L_rollout + lambda_img * L_img_hop0 (+ optional regularizer)
```

- `L_pair`：单跳 latent 监督
  - velocity term + endpoint term
  - 支持 pair sampling / pair loss weights / dt normalize
- `L_rollout`：4-hop rollout 级联损失（step weights 可配置）
- `L_img_hop0`：仅 hop0 图像辅助（`L1 + SSIM + seam`）
  - SSIM 采用 map-level clamp 的非负稳定版本（实现于 `pet_lr/losses.py`）

## Canonical Commands

所有命令在仓库根目录执行：`/home/qujiaxiang/project/PET_LatentResidual`

### 1) 生成对齐审计（必须）

```bash
/home/qujiaxiang/.conda/envs/rae/bin/python check_alignment_224_clip3.py \
  --latents-dir /data_2/qujiaxiang/lowdose_pet_ct/latents_224 \
  --raw-data-dir /data_2/qujiaxiang \
  --rae-ckpt /data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/alignment_224_audit
```

### 2) 训练（current production：V6 transport-first / foc_lite / σ-norm ablation）

```bash
# A_main：V6 transport-first（σ-norm ablation 主对照）
/home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py \
  --config /home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml

# C_uniform：σ-normalizer ON, uniform step weight（盲分析对照）— 仅在 X 锁定后才允许跑 full-val
/home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py \
  --config /home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_foc_lite.yaml

# V6.1：rollout floor 加固
/home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py \
  --config /home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_v6_1_rollout_floor.yaml
```

### 3) clip3 评估（JSON/CSV 可复核）

```bash
/home/qujiaxiang/.conda/envs/rae/bin/python eval_first_hop_224_clip3.py \
  --config /home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/<run_dir>/best.pt \
  --split val \
  --max-slices 0 \
  --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/<run_dir>_eval_clip3_best_full
```

> **C_uniform full-val 顺序硬约束**：在 `review/0502/EFFECT_SIZE_LOCKED.md` 已 commit + push 之前，**禁止**对 `C_uniform` 跑 `--max-slices 0` full-val。详见 §8.7。

### 4) §6.6 effect-size lock-in（A_main 完成后，C_uniform full-val 之前必须执行）

```bash
python review/0502/scripts/lock_effect_size_threshold.py \
    --metrics-a /data_2/qujiaxiang/outputs/PET_LatentResidual/A_main/run-.../metrics.jsonl \
    --config-a  review/0502/configs/A_control.yaml \
    --output    review/0502/EFFECT_SIZE_LOCKED.md \
    --c-uniform-output-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/C_uniform/
```

详见 [review/0502/POST_V6_NEXT_STEPS.md](review/0502/POST_V6_NEXT_STEPS.md) §6.6.2 + §8.7。

## Core Files

- `train_first_hop.py`: 主训练流程、调度、日志、checkpoint
- `pet_lr/data_first_hop.py`: 224 数据读取 + 对齐检查 + fail-fast
- `pet_lr/model_first_hop.py`: hop0 pixel forcing + hop residual
- `pet_lr/rollout_first_hop.py`: 4-hop rollout 训练/推理
- `pet_lr/losses.py`: SSIM/L1/seam 等基础 loss
- `pet_lr/losses_first_hop.py`: hop0 图像辅助 loss 聚合
- `check_alignment_224_clip3.py`: 对齐审计
- `eval_first_hop_224_clip3.py`: 统一 clip3 评估 JSON/CSV 输出

## Experiment Ledger (Completed)

### A. 10k 系列（clip3，val，全量）

| run | D50 | D20 | D10 | D4 | NORMAL | tail_avg(D20~N) | all_avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_best | 42.6224 | 35.3122 | 35.5493 | 35.9848 | 36.0813 | 35.7319 | 37.1100 |
| strict_last | 42.6224 | 35.3122 | 35.5493 | 35.9848 | 36.0813 | 35.7319 | 37.1100 |
| archfix_best | 42.6224 | 35.2308 | 35.4839 | 35.8827 | 36.1713 | 35.6922 | 37.0782 |
| item1_best_full | 42.6224 | 35.2602 | 35.5075 | 35.9246 | 36.0002 | 35.6731 | 37.0630 |
| item2_best_full | 42.6224 | 35.2687 | 35.5103 | 35.9301 | 36.0095 | 35.6796 | 37.0682 |

对应评估文件位于：

- `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_10k_*_eval_clip3_*/first_hop_224_val_clip3_eval.{json,csv}`

### B. 50k chainstable 与 strict full 诊断对比

| run | D50 | D20 | D10 | D4 | NORMAL | tail_avg(D20~N) | all_avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_best_full | 42.6224 | 35.3120 | 35.5490 | 35.9845 | 36.0810 | 35.7316 | 37.1098 |
| chainstable50k_best_full | 42.6224 | 35.4881 | 35.8620 | 36.4207 | 36.7422 | 36.1283 | 37.4271 |

增量（chainstable50k - strict）：

- D20: `+0.1761 dB`
- D10: `+0.3130 dB`
- D4: `+0.4362 dB`
- NORMAL: `+0.6612 dB`

诊断文件：

- `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/strict_best_full/hop_difficulty_clip3_val_strict_best_full.json`
- `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/chainstable50k_best_full/hop_difficulty_clip3_val_chainstable50k_best_full.json`

## Important Metric Semantics

### 1) D50 列含义

在 `eval_first_hop_224_clip3.py` 里，`preds[0]` 直接是输入 `z_d50`，因此 D50 列是“输入基线解码 PSNR”，不是 transport 预测结果。

### 2) 第一跳是否有效（当前证据）

从 `hop_difficulty_clip3` 诊断：

- `no_transport D20_from_D50`: `32.9309 dB`
- `strict tf D50->D20`: `35.3120 dB`（+2.3811 dB）
- `chainstable50k tf D50->D20`: `35.4881 dB`（+2.5572 dB）

结论：第一跳机制有正向贡献，但第一跳仍是全链路最难段。

## Current Understanding

1. 当前实现已形成可落地工程链路：
   - 对齐可审计
   - 训练可运行
   - 评估可复核（JSON/CSV）
2. hop0 机制不是失活，但对第一跳提升幅度小于 tail 段提升。
3. chainstable 的目标函数/评估选择偏 tail，导致“前段小幅、后段显著”的现象。

## Next Engineering Plan (Pending Approval) — SUPERSEDED 2026-05-03

> ⚠️ **此节自 2026-05-03 起 superseded**。当前主线已切换为 V6 transport-first + σ-normalize ablation；详见下一节 "Current Status (as of 2026-05-03)"。本节保留以保历史可回溯。

1. 做同预算因果消融（50k）：`hop0_off` vs `hop0_on`。
2. 若目标是优先修第一跳：
   - 提升 `val_chain_d20_mse` 的 best-metric 权重；
   - 调整 rollout step weights 为前重后轻；
   - 维持 clip3 统一评估与 JSON/CSV 固化输出。
3. 保持不改主状态（latent-only），不引入 dual-state transport。

## Current Status (as of 2026-05-03) — SUPERSEDED BY 2026-05-26 STATUS ABOVE

**主线已切换**：`50k_formal_v3_chainstable` → `50k_foc_lite` / `v6_transport_first` / `v6_1_rollout_floor` 系列。Best-metric 已统一为 `val_select_score`（multi-objective），rolling window evaluation `max_val_batches=64, eval_interval=400, val_window_mode=rolling`。

**协议源**（canonical, 不要在 CLAUDE.md 里复述细节）：

- [review/0502/POST_V6_NEXT_STEPS.md](review/0502/POST_V6_NEXT_STEPS.md) — V6 → ablation transition master plan
  - §6.4 LOCKED：Risk 4 paired-diff threshold = 0.10；判读由 `review/0502/scripts/paired_diff_judge.py` 自动化
  - §6.6 LOCKED：blinded effect-size pre-registration `X = max(0.10, 3 × paired_CV_A)`，window `[40000, 60000]`，metric `val_select_score`；锁定由 `review/0502/scripts/lock_effect_size_threshold.py` 自动化（exit codes 1/2/3/4/7）
  - §8.6 / §8.7：执行顺序硬约束（X 必须在 C_uniform full-val 之前锁定）

**当前 ablation 三支（待 operator 确认，见 [review/0503/local/OPERATOR_QUESTIONS.md](review/0503/local/OPERATOR_QUESTIONS.md) Q1）**：

- `A_main`：V6 transport-first，step-weight middle-heavy（paper 主张的 "σ-norm 是核心" 的对照实验 baseline）
- `C_uniform`：B_sanity 上加 σ-normalizer ON，step-weight uniform（盲分析对照）
- `V6.1`：rollout floor 加固版（独立产线，非 σ-norm ablation 的一部分）

**强约束（do not violate）**：

1. 在 `EFFECT_SIZE_LOCKED.md` 写入并 push 到 `gitee/foc_lite_hop0` 之前，**禁止**对 C_uniform 跑 full-val（顺序违例 = pre-registration 失效 = desk-reject 风险）
2. §6.4 / §6.6 的 LOCKED 参数（floor=0.10, slope=3, threshold=0.10, window=[40000,60000]）**禁止改动**；只能通过 `--deviation-note` 在 lock 文件里公开窗口偏离
3. CCT-224 / ΔB-aware reweighting / uncertainty-gated hop0（见 `IDEA_REPORT.md`）当前**全部 deferred**；σ-norm ablation 落地后再回到 idea backlog
4. 当前在跑的实验保留 `save_interval=20000` 以避免 disrupt；下一批新实验再切换为协议规范版本

**最近 milestone commit**（gitee/foc_lite_hop0）：

- `8f65584` / `9811d02`：Method D ckpt selection + ROLLING_WINDOW_TRADEOFF 决策
- `867b5c0`：§6.6 blinded pre-registration 锁定
- `e9ae9e5`：§6.4 paired_diff_judge.py + threshold 0.10 锁定
- `7100839`：lock_effect_size_threshold.py（自动化 §6.6.2 Step 3-4） + README 公开版润色

## Notes

- 所有训练、评估、诊断产物必须直接写入 `/data_2/qujiaxiang/outputs/PET_LatentResidual`
- 不允许在仓库根目录保留 `outputs` 目录或符号链接
- 历史遗留产物迁移脚本：`tools/migrate_repo_outputs_to_data_disk.sh`
- 训练与评估请固定 conda 环境：`rae`
- 如需新增架构改动，必须先经用户批准再实施。
