# 服务器任务: 五项质疑的实验收口

日期: 2026-06-03
发起: 本地 supervisor 复核（paper 侧）
执行: 远程服务器 `/home/qujiaxiang/project/PET_LatentResidual`，分支 `foc_lite_hop0`
环境: `/home/qujiaxiang/.conda/envs/rae/bin/python`
PSNR 硬规范: `src.utils.metrics.calc_psnr_clip3`（先调窗到 3 再算）
canonical anchors（val, n=7403, decode_mode=default）:
- V13.best (image_aux=0): NORMAL PSNR **36.494330**
- A4-mid.best (image_aux=0.08): NORMAL PSNR **36.893917**  ← headline
- ΔNORMAL = **+0.399587 dB**

## 0. 背景与边界（先读）

本任务不改训练主干、不改模型架构、不改 `RAE`。除"已批准的 L1-only / seed1337 收尾"外**不启动任何新训练**，不新增 lambda 点、不做 seed sweep、不碰 V18/V19/decoder rank sweep（遵守 `CLAUDE.md` 停机规则）。S2 为**只读分析脚本**（前向 + 自动微分），不写回任何 checkpoint。

五项质疑分类:

| # | 质疑 | 是否需服务器跑 | 任务 |
|---|---|---|---|
| 1 | headline +0.40 dB 偏小，且是同一管线 self-ablation，无外部 baseline | 否（叙述边界） | 见 §4 |
| 2 | 多数指标在 loss 内（SSIM/seam/MAE），自证；真正独立证据≈PSNR+MS-SSIM | 否（叙述边界） | 见 §4 |
| 3 | 核心理论 M=JᵀJ 各向异性**从未被直接测量**，增益未挂到高-M 方向 | **是** | **S2（最高优先）** |
| 4 | 复合损失（L1+SSIM+seam）未拆解，缺 L1-only 对照 | 是（已在训练，收尾） | S1 |
| 5 | 单 seed、单 split、无 patient 级统计；64-slice block-bootstrap 可能低估方差 | 是（多为分析） | S3 |

---

## S1 — L1-only 对照收尾（质疑 #4）【已在轨，收口】

**目的**: 判断 A4 的增益是来自"decoder-aware latent routing"，还是仅仅来自额外的 SSIM+seam 图像先验。L1-only 把图像辅助退化为"仅 L1（仍过同一冻结 decoder）"，是拆解复合损失归因的载重对照。

配置已存在: `review/0601/C_l1_only/A4_l1_only_lambda_08.yaml`
输出目录: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0601_runs/C_l1_only_lambda_08`

步骤:
1. 确认该 run 是否已训练完成（`metrics.jsonl` 是否到 `max_steps=160000` / best.pt 是否存在）。若未完成，按既有配置续训至完成，**不改任何超参**。
2. full-val 评估（与 V13/A4 完全同口径）:
```bash
/home/qujiaxiang/.conda/envs/rae/bin/python eval_first_hop_224_clip3.py \
  --config review/0601/C_l1_only/A4_l1_only_lambda_08.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0601_runs/C_l1_only_lambda_08/best.pt \
  --split val --max-slices 0 \
  --out-dir review/0603/server/c_l1_only_first_hop_224_val_clip3_eval
```
3. 把 L1-only 追加进 `review/0601/server/metrics_summary.csv` 与 seam 分层表，新建对照三联表 **V13 / L1-only / A4-mid**，所有 timepoint 报告: PSNR、MS-SSIM、NRMSE、seam、ext-seam、ΔvsV13、95% block CI、win-rate。

**判读规则（写清楚，不要含糊）**:
- 若 L1-only 已复现 A4 的**大部分** NORMAL PSNR 增益（例如 ≥70% 的 +0.40 dB）→ 则"复合损失中的 SSIM+seam"不是主因，主因是"通过冻结 decoder 的像素域 L1 约束"，论文必须如实改写归因，**不能把功劳归给 seam/SSIM 项**。
- 若 L1-only 仅复现一小部分 → 复合结构确有必要，但仍须报告分解数字，不能只给复合总账。

交付: `review/0603/server/c_l1_only_first_hop_224_val_clip3_eval.{json,csv}` + 更新后的三联对照表 md。

---

## S2 — M 各向异性的直接验证（质疑 #3）【新分析，最高优先，只读】

**问题**: 论文 §3.2 用 pulled-back metric M=J_Gᵀ J_G 论证"decoder-aware 监督会优先修正高-M（高 Jacobian 增益）方向的 latent 误差"。但**全项目从未测过 M**：没有谱、没有条件数、没有把增益与高-M 方向关联。当前 M 是装饰性论证，不是被验证的机制。评审一眼能看穿。S2 用两个只读分析把它坐实或证伪。

新建独立脚本 `tools/analyze_pullback_metric.py`（仅前向 + autograd jvp/vjp，加载已有 frozen RAE decoder 与 A4/V13 模型，**不训练、不写 checkpoint、不改 train/model/RAE**）。

### S2.0 — 逐跳证据：D50→D20 是否真是"最坏跳"（论文叙事的经验地基）
重排后的 §1/§3.2 以"经验优先"开场，首句就断言 D50→D20 是信息损失最重、解码 seam 最重的跳。这个断言目前只有 `σ_0 ≈ 2跳倍阶` 一个间接依据，必须用逐跳数据坐实，否则是贤话。只读分析：
- 对每个 hop k∈{D50→D20, D20→D10, D10→D4, D4→NORMAL}，在 val GT latent 上统计：（1）跳位移幅度 `‖z_{k+1}−z_k‖`（均值/分位数）与估计的 `σ_k`；（2）将该跳的 latent 残差（用 V13/A4 预测，或直接用 GT 跳差作为上界）过 frozen G 解码后的 **seam/ext-seam 幅度**。
- 报告一张逐跳表：hop × {位移幅度, σ_k, 解码 seam%, ext-seam%}。
- **判读**：若 D50→D20 同时在"位移幅度/σ"与"解码 seam"两轴上都明显领先（例如 ≥1 个量级）→ "最坏跳"叙事成立，§1/§3.2 可以理直气壮地"先经验"；若 seam 最重的并非首跳 → 叙事须改，不得把 image_aux 只挂在首跳的理由写成"seam 最重"。

### S2.a — decoder Jacobian 谱（M 是否真各向异性）
- 在 val GT latents 中按 timepoint（至少 D20 与 NORMAL）随机抽 N≈64 个 latent `z*`（C×h×w=768×16×16）。
- 对每个 `z*`，用 `torch.autograd.functional`（或手写 jvp/vjp）做 matrix-free 的 `M v = J_Gᵀ(J_G v)`，跑随机化/Lanczos 谱估计（不显式构造 J_G；image 维 224×224，latent 维 196608，必须 matrix-free）。
- 报告每个 `z*` 的: top-k 与 bottom-k 奇异值（σ_i(J_G)）、条件数 σ_max/σ_med、有效秩（参与化 spectral entropy）、谱在前 1%/10% 方向上的能量占比。
- **判读**: 若条件数 ≫ 1 且能量高度集中在少数方向 → M 强各向异性成立，§3.2 前提为真；若谱接近各向同性 → §3.2 论证不成立，必须删掉或降级为 motivation。

### S2.b — 把增益挂到高-M 方向（机制是否如所述）
- 用 read-only 前向，dump A4 与 V13 在 **NORMAL 端点** 的预测 latent `z_pred`（同一批 val slices，建议先取 N≈512 slice 子集，跑通后再扩到 full-val）。
- 误差: `δ_V13 = z_pred_V13 − z_GT_NORMAL`，`δ_A4 = z_pred_A4 − z_GT_NORMAL`。
- 在每个对应 `z*=z_GT_NORMAL` 的局部 M-本征基上，把 δ 投影到 **高-M 子空间**（top 方向）与 **低-M 子空间**（bottom 方向）。
- 报告:
  - M-加权误差范数 `‖δ‖_M² = δᵀ M δ`（≈ 解码后像素误差，见 §3.2 Eq.3）对 V13 vs A4 的下降量；
  - 高-M 子空间内 `‖δ‖²` 的下降量 vs 低-M 子空间内的下降量；
  - 像素域实测 `‖G(z_pred)−G(z_GT)‖²` 与 `δᵀ M δ` 的相关性（验证一阶近似是否成立）。
- **判读（这是论文的命门）**: 若 A4 相对 V13 的误差下降**优先集中在高-M 方向**、且 `‖δ‖_M` 的下降幅度大于普通 `‖δ‖₂` → 机制被证实，M 从装饰变成证据，§3.2 可保留并引用该图表。若下降在高/低-M 方向无差别 → "decoder-aware 优先修高-M 方向"的说法证伪，必须改写为更弱的"像素域约束整体降低误差"，**不得保留 M 各向异性叙述**。

### S2.c — M→seam 桥接测试（高-M 方向是否就是 patch 边界方向）
**这是 §1/§3.2 重排后新增的关键断言**：重排后论文明确写"硬 unpatchify decoder 的高-M（高 Jacobian 增益）方向与 14px patch 格对齐，所以最坏跳误差被优先解码成 seam"。这是把 M（S2.a/b）与 seam（S2.0）两条证据缝合起来的唯一环节，必须直接测。
- 在若干 `z*`（NORMAL 端）上，用 matrix-free 幂迭代 / Lanczos 取 M=J_GᵀJ_G 的 **top-r 本征向量** `u_i`（latent 空间，768×16×16）。
- 把每个 `u_i` 通过 `J_G u_i`（jvp）映射到图像空间（224×224），得到该高增益方向的**像素空间响应图** `J_G u_i`。
- 构造二值 **patch 格掩模** `P`：14px 格点阵的边界像素带（±1px）置 1，其余置 0（与 seam 损失用的边界掩模一致）。
- 量化重合度：（1）seam 能量占比 `∑_{P} (J_G u_i)² / ∑ (J_G u_i)²`；（2）与随机单位方向（及底-M 本征向量）的 baseline 重合度对比；（3）`J_G u_i` 的空频谱是否在 14px 周期（及谐波）上出现能量峰。
- **判读（决定 §1/§3.2 叙事是否成立）**：若 top-M 方向的像素响应显著集中在 patch 边界（seam 能量占比 ≫ 随机/底-M baseline，且空频在 14px 周期出峰）→ "高-M = patch 边界方向"被证实，M（几何）与 seam（现象）打通，§3.2 的"高-M 方向与 patch 格对齐"可保留为核心证据。若高-M 方向与 patch 格无相关 → 该句证伪，§1/§3.2 必须删除"高-M 与 patch 格对齐"，seam 只能作为纯经验观察保留，M 降级为未验证的 motivation。

交付:
- `review/0603/server/per_hop_seam_evidence.{json,csv}`（S2.0 逐跳表）;
- `review/0603/server/m_anisotropy_spectrum.{json,csv}` + 谱直方图 + 条件数分布图;
- `review/0603/server/m_directional_gain.{json,csv}` + 高-M vs 低-M 误差下降对比图;
- `review/0603/server/m_seam_bridge.{json,csv}` + top-M 本征向量的像素响应图（叠 patch 格）+ seam 能量占比对比图;
- `review/0603/server/S2_pullback_metric_report_20260603.md`（含 S2.0/a/b/c 四个判读结论，明确写"支持 / 不支持 §1 经验叙事"与"支持 / 不支持 §3.2 M 机制"）。

---

## S3 — 统计稳健性收尾（质疑 #5）【多为分析，无新训练】

**目的**: 回应"单 seed、单 split、block-bootstrap 可能低估方差"。

1. **seed 复现**: 对已批准的 `A4-mid-seed1337` best.pt 跑 full-val（同 S1 口径），与 seed42 headline 并列报告 NORMAL PSNR/MS-SSIM 的 seed 间差。明确标注"n=2 seed，仅作 robustness replicate，非 seed sweep"。**不新增 seed。**
2. **per-slice seam/ext-seam 导出**: 当前 CSV 缺 per-slice seam 字段（见 systematic 文档 §6.4）。在 eval 输出里补导 per-slice `seam_*` 与 `ext_seam_*`。
3. **2D 分层**: 用 V13（aux-off）的 `seam_NORMAL` 三分位 × `D50 PSNR` 三分位做 seam-risk × input-quality 9 格分层，每格报告 ΔPSNR、Δseam%、Δext-seam%、win-rate、block CI。分层器固定用 V13，**不得用 A4 结果做选择器**（避免选择偏差）。
4. **CI 稳健性自检**: 当前 64-slice 连续块 block-bootstrap 在相邻 slice 强相关时可能偏窄。用更粗的块（如 128-slice、整 volume 块若可恢复 volume 边界）重算 NORMAL ΔPSNR 的 95% CI，报告 CI 随块长的变化。**说明**: patient ID 不可恢复（CLAUDE.md），故不写 patient 级显著性，仅报告"slice 级 + 块长敏感性"。

交付: `review/0603/server/seed1337_eval.{json,csv}`、`review/0603/server/seam_2d_strata.csv`、`review/0603/server/S3_robustness_report_20260603.md`。

---

## §4 — 不需要服务器跑的两条（质疑 #1、#2）

仅叙述边界，**无实验**，由 paper 侧处理:
- #1（效应小 + self-ablation）: 论文保留"no state-of-the-art claim"，并明确 headline 是同一冻结管线内 image_aux on/off 的受控对比（V13 vs A4），claim 限定为 "controlled ablation of decoder-aware supervision under a fixed frozen decoder"，不外推为 SOTA。
- #2（自证指标）: 沿用 `bext_v13_a4_metrics_explanation` 已有的 independent（PSNR/MS-SSIM/NRMSE/RMSE）vs training-aligned（SSIM/MAE/seam）划分，headline 只用 independent 轴；training-aligned 仅作一致性佐证。注意 NRMSE/RMSE 与 PSNR 同属 MSE 家族（单调相关），故真正独立轴是 {MSE 家族, MS-SSIM} 两条——论文措辞要诚实，不要把它们说成四条独立证据。

---

## 优先级与停机规则
1. **S2 > S1 > S3**。S2 是唯一能把"理论从装饰变证据"的实验，最高优先。S2 内部顺序 **S2.0 → S2.a → S2.b → S2.c**：先用 S2.0 坐实"最坏跳 = 首跳"的经验叙事，再用 S2.a/b/c 验证 M 机制与 M→seam 桥接。
2. 任一环节证伪即停、回报本地 supervisor：若 S2.0 证伪"首跳 seam 最重" → §1/§3.2 经验开场需重写；若 S2.b 证伪"高-M 优先修正"或 S2.c 证伪"高-M = patch 边界" → §3.2 M 叙事需降级或删除。
3. 所有 eval 用 `eval_first_hop_224_clip3.py` + `calc_psnr_clip3`，full-val n=7403，decode_mode=default。
4. 任何需要改模型/训练主干的需求，先回报、获明确批准再做（CLAUDE.md 硬约束）。
