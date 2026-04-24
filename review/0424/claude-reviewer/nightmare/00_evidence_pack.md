# Nightmare Review 0424 — Evidence Pack

**Scope**: PET_LatentResidual 仓库，224 first-hop latent transport 主线
**Mode**: NIGHTMARE（5 角色 × 5+ 轮对抗）
**Date**: 2026-04-24

本文档是所有 reviewer subagent 的唯一 shared context。每位 reviewer 必须基于此文件给出判断；允许引用文件路径但不假设其他人看过。

---

## A. 仓库核心主张（Claims under review）

### A1. Problem claim
**"PET low-dose → normal-dose 的 4-hop latent transport 中，第一跳 D50→D20 是全链路瓶颈。"**
- 支撑数据（from CLAUDE.md §Current Understanding，诊断文件 `hop_difficulty_clip3`）:
  - `no_transport D20_from_D50` = 32.93 dB（直接把 z_D50 当 z_D20 解码）
  - `strict tf D50→D20` = 35.31 dB（+2.38 dB）
  - `chainstable50k tf D50→D20` = 35.49 dB（+2.56 dB）
  - Decoder GT oracle = 46–52 dB（远高于 transport）

### A2. Method claim（三件套，已实现）
1. **FirstHopPixelEncoder**: 只在 hop0 把 `x_D50` 弱条件注入到 `z_D50`（`z_in = z_src + g_pix · pixel_encoder(x_D50)`），浅 CNN + AdaptiveAvgPool + 1×1 proj；参数量 ≤ backbone 5%
2. **HopResidualVelocityHead**: 每 hop 一个小 1×1 conv head，`v_total = v_shared + λ_hop[h] · head_h(v_shared)`；零初始化 + softplus 门
3. **hop0 image auxiliary loss**: `L_img_hop0 = L1+SSIM+seam` on `Crop192(Dec(z_D20_pred))`，decoder 冻结但梯度穿过
4. **（范围蔓延）** 还实现了 `SeamRefiner`（post-decoder 7×7 depthwise refiner）、`SpatialAlignmentProjector`（iREPA-style）、EMA、velocity_rebalance、N1 ablation 等

### A3. Result claim（from `review/0423/multiagent_transport_review_and_rerank_20260423.md`）
Full-val clip3 PSNR（n=7403），transport_avg = mean(D20,D10,D4,NORMAL)：

| Scheme | D20 | D10 | D4 | NORMAL | transport_avg |
|---|---:|---:|---:|---:|---:|
| C best | 35.569 | 35.942 | 36.505 | 36.806 | **36.206** |
| v2 best | 35.566 | 35.924 | 36.500 | 36.798 | **36.197** |
| N1 best | 35.558 | 35.875 | 36.407 | 36.654 | **36.123** |
| N1 last | 35.557 | 35.897 | 36.457 | 36.735 | **36.161** |

**所有实验差异 < 0.1 dB，全部聚集在 36.12–36.21 dB plateau。**

### A4. 0424 当前行动
- 正在启动 200K-step transport v3（从 50K 延长 4×，15.2 遍数据）
- `v3` 做了 4 项修改：pair/rollout 权重反转（前重后轻）、D20 在 val-select 权重 0.15→0.50、velocity_rebalance 启用
- 对 plateau 的归因（investigation_and_plan.md）：「latent MSE 卡在 0.0002 per-element，需要 10× 下降才能到 46 dB」

---

## B. 自评（Self-reported scores）

来自 `IDEA_REPORT.md`（自评）:
- CCT-224 (Counterfactual Consistency Training): **novelty 8.4/10** — 但已移到 `idea1_cct_test` 分支，master 未采用
- ΔB-aware Adaptive Reweighting: **novelty 7.6/10**
- Uncertainty-Gated Hop0 Forcing: **6.3/10**

来自 `AUTO_REVIEW.md` Round 1:
- TotalScore: **4.84/10** — FAIL
- Vetoes: `causal not established`, `evidence gate failed`
- 缺失产物：`tf_vs_pure_gap.json`、`seam_strata_summary.csv`、variant 全量 clip3 JSON/CSV

---

## C. 关键对比与红旗（Red flags）

### C1. 第一跳机制的真实贡献
- hop0 机制（pixel forcing + residual head + img aux）声称解决"首跳瓶颈"
- 但 50k chainstable 相比 strict baseline：D20 只 +0.18 dB，而 NORMAL +0.66 dB
- **说明增益主要发生在 tail，而非第一跳** — 与叙事相反

### C2. Plateau 归因矛盾
- 0424 调查排除了 wd、EMA、pixel forcing、backbone LR、decoder — 全部不是瓶颈
- Oracle decoder 能达到 46–52 dB → transport 端缺了 10 dB
- 当前方案（200K 步）只是延长训练，**没有回答"为什么 latent MSE 卡住"**

### C3. 范围蔓延（scope creep）
- `model_first_hop.py` 已包含 4 个独立模块：PixelEncoder / ResidualHead / SpatialAlignmentProjector (iREPA) / SeamRefiner
- 25+ configs (`configs/pet_flow/*.yaml`)，每个都是一次尝试
- CLAUDE.md / main.md 合计 960 行"实施规范"

### C4. 对齐检查层层加码，但核心机制未验证
- alignment_audit，fail-fast，decoded_vs_raw_l1，seam_strata，…
- 但没有单独的 **"去掉 hop0 机制 vs 保留"** 的 full-val clip3 控制实验
- N1 (pixel OFF) best = 36.123，C best (pixel ON) = 36.206 → 差 0.08 dB，统计显著性未检验

### C5. Novelty positioning
- "hop0 pixel forcing" 本质 = 把源图作为 side conditioning，类似 ControlNet / image-conditioned diffusion
- "hop residual velocity head" 本质 = per-class/per-task LoRA-style residual
- "hop0 image aux loss through frozen decoder" = perceptual loss 的一个变体
- 单独看每个组件都不新；组合的新意在于"只第一跳注入"——但 C1 显示这个组合并未显著解决第一跳问题

### C6. Story–Evidence gap
- Story: "first-hop 是 the bottleneck"
- Evidence: hop0 PSNR (35.5 dB) 确实低于 NORMAL (36.8 dB)，但差距只有 1.3 dB；且 tail 在持续改善
- 如果真的是第一跳瓶颈，应该做 D50→NORMAL 单跳对比，而不是 4-hop chain vs no-transport

---

## D. 实现细节（供 Implementation Auditor）

### D1. 模型入口
- `pet_lr/model_first_hop.py::PETFlowDiTFirstHop.predict_latent_step`
- 前向：`z_in = z_src + g_pix(softplus) · pixel_encoder(x_D50)`（仅 hop=0）；`v_total = v_shared + λ_hop(softplus)·head(v_shared)`
- `pixel_forcing_disabled` flag → N1 ablation（注意：encoder frozen 但 gate metric 仍记录）

### D2. 数据
- 104,988 training pairs（26,247 slices × 4 hops），29,612 val pairs
- latent: 768×14×14 (docs/main.md 写成 16×16 但 model 里 latent_size 也是 16，需要确认实际尺寸，见 D6)

### D3. Loss
- `L_total = L_pair + λ_roll(step)·L_rollout + λ_img(step)·L_img_hop0`
- `L_pair` 带 `target_normalize`, `pair_weighting`, `straight-through mixing`
- `L_img_hop0` 只在 hop0 subbatch；decoder 冻结但梯度穿过

### D4. 已知 velocity_rebalance 实现
- `loss_velocity = (vel_err · sample_weights).mean()`
- `ratio = loss_endpoint.detach() / loss_velocity.detach()`
- `rebalance_scale = clamp(sqrt(ratio), 1.0, 4.0)` — 单向放大
- 注入方式：`velocity_weight_eff = v_weight · rebalance_scale`

### D5. 已知模型容量分析
- DiT-S: ~200M params, 1892 params/sample (ImageNet 参考是 67)
- 结论：模型已偏大，不该增容

### D6. 可疑点（待 Implementation Auditor 核查）
- `docs/main.md` 中 `FirstHopPixelEncoder` 输出 `N=196, D_enc`（14×14）
- 但 `model_first_hop.py` 的 `latent_size` 默认 16 → 实际 tile 是 16×16 还是 14×14？
- `chainstable` best_metric 权重 D20=0.15 vs NORMAL=0.50（v2）vs D20=0.50（v3）—— 是否真的对 selection 有影响？
- `lambda_hop_init=1e-3` + `softplus` 意味着 λ 起始值 ≈ `ln(1+e^-20) ≈ 2e-9` — 接近 0，但不是 0；早期基本关闭

---

## E. 要求 Reviewer 回答的问题

每位 reviewer 必须就以下 5 个维度打分（0–10）并给出证据：

1. **Novelty**: 方法相比已有 image-conditioned diffusion / per-hop residual / perceptual loss 的增量
2. **Story coherence**: claims ↔ evidence 是否闭合；first-hop 叙事是否立得住
3. **Implementation soundness**: 代码与声明是否一致；有无实现错误；范围蔓延程度
4. **Causal evidence**: hop0 机制对结果的独立贡献是否被隔离验证
5. **Submission-readiness**: 以当前状态投顶会的风险等级

此外：
- **必须**给出 3 条最尖锐的批评
- **必须**给出 1 条让作者最难反驳的具体问题
- **不得**抄袭其他 reviewer 的措辞（每轮公开前不共享）
