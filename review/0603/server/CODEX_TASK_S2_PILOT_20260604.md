# 服务器任务: S2 pilot — 把 M=JᵀJ 从"假设"升级为"证据"（或证伪）

日期: 2026-06-04
发起: 本地 supervisor 复核（paper 侧）
执行: 远程服务器 `/home/qujiaxiang/project/PET_LatentResidual`，分支 `foc_lite_hop0`
环境: `/home/qujiaxiang/.conda/envs/rae/bin/python`
PSNR 硬规范: `src.utils.metrics.calc_psnr_clip3`（先调窗到 3 再算）
canonical anchors（val, n=7403, decode_mode=default）:
- V13.best (image_aux=0): NORMAL PSNR **36.494330**
- A4-mid.best (image_aux=0.08): NORMAL PSNR **36.893917** ← headline
- L1-only.best: NORMAL PSNR **36.715786**（已收口，见 CODEX_L1_S2_STATUS_20260604.md）

## 0. 背景与边界（先读）

不改训练主干、不改模型架构、不改 `RAE`、不启动任何新训练、不新增 lambda 点、不做 seed sweep。
本任务**全部为只读分析脚本**（前向 + 自动微分），不写回任何 checkpoint，遵守 `CLAUDE.md` 停机规则。

### 0.1 对你（Codex）上一轮"暂不跑 S2.a/b/c"决定的回应

你在 `CODEX_L1_S2_STATUS_20260604.md` 里给的**方法学批评全部采纳**（见 §0.2），但你"暂不启动"的**理由不成立**，本任务据此重新下达：

- 你的论据：「S2.0 只弱支持首跳最坏，强机制实验即使跑通也不能支撑‘首跳数量级最坏’的论文开场」。
- 问题：这把**两个独立问题混为一谈**。
  1. **首跳是不是数量级最坏**（§1 经验开场）——已由 S2.0 回答：是最坏跳，但只 ~2×/~4 dB，非数量级。论文措辞已据此改正（删除 "orders of magnitude"，改 "roughly twice / about 4 dB"）。**这条不依赖 M。**
  2. **M=JᵀJ 各向异性、且增益挂在高-M 方向**（§3.2 理论机制）——这是**另一条独立 liability**。无论首跳差距多大，§3.2 只要写了 M 解释，就必须被测，否则评审会判它是装饰性论证。
- 结论：M 验证的必要性**与首跳量级无关**。S2.0 弱不弱，不构成跳过 M 验证的理由。故本轮把 S2 重新定义为**轻量 pilot**（采纳你的 N=16/top-spectrum/相关性先验改法），低成本试探，能升级就升级、不能就如实证伪。

### 0.2 已采纳的方法学修正（你提出的，写进硬约束）

1. **禁止报 bottom-k 奇异值、条件数、有效秩**。196608 维 latent、matrix-free 下 bottom spectrum 不可靠。只报 **top spectrum + Hutchinson trace + top-energy concentration**。
2. **一阶近似 δᵀMδ≈像素误差只在小扰动成立**。因此把**相关性检验当作前置闸门**（S2.b-pilot 先跑），不达标就**停掉整条 M 叙事**，不再跑 S2.a/S2.c。
3. **"top-M = patch boundary" 是强断言，可能被证伪**。pilot 若证伪，论文保留 seam 的经验观察，但删去"seam 源于 top-M 与 patch-grid 对齐"的因果归因（目前论文已把该耦合降级为 hypothesis，正好对应）。
4. 样本量先 **N=16**（D20 取 8 + NORMAL 取 8），通过再扩 N=32。

---

## 执行顺序（pilot 是有闸门的流水线，按序跑，不要并行抢结论）

新建独立脚本 `tools/analyze_pullback_metric.py`（仅前向 + autograd jvp/vjp；加载已有 frozen RAE decoder G 与 A4/V13 模型；**不训练、不写 checkpoint、不改 train/model/RAE**）。所有 latent 维度 768×16×16=196608，image 维 224×224，**必须 matrix-free**（不显式构造 J_G）。

JVP/VJP 约定：
- `J_G v`：对 `G(z)` 在 `z*` 处的前向模式雅可比向量积（`torch.func.jvp` 或 double-vjp）。
- `Mv = J_Gᵀ(J_G v)`：先 jvp 得 image 方向，再 vjp 拉回 latent。一次 Mv = 1 次 jvp + 1 次 vjp。
- 单位：在 clip3 SUV [0,3] 解码域上做（与 seam/PSNR 同域），先固定 decode_mode=default。

### 【闸门】S2.b-pilot — δᵀMδ 与真实像素 MSE 的相关性（先跑，决定 M 叙事生死）

**目的**：验证一阶 pullback 近似 `image_err ≈ δᵀMδ` 在本任务真实扰动幅度下是否成立。不成立则 M 的整段几何解释失去定量根基，必须停。

- 取 N=16 个 (slice, hop) 样本：D20 destination 8 个 + NORMAL destination 8 个。
- 对每个样本，`δ = z_pred − z_gt`（用 A4 与 V13 各算一组，hop0 用 GT source 单步预测，口径同 S2.0 single-step）。
- 计算三个量：
  1. `q = δᵀMδ`（matrix-free：`= ‖J_G δ‖² `，一次 jvp 即可，无需显式 M）。
  2. `e = ‖G(z_gt+δ) − G(z_gt)‖²`（真实解码像素误差，clip3 域）。
  3. `l = ‖δ‖²`（朴素 latent 误差，作为对照基线）。
- 报告：`corr(q, e)` 与 `corr(l, e)`（Pearson + Spearman），以及 log-log 散点（q-vs-e、l-vs-e）。
- **判读闸门**：
  - 若 `corr(q,e)` 显著高于 `corr(l,e)`（例如 Spearman 高 ≥0.15，且 q-vs-e 接近线性）→ **M 近似成立**，pullback 几何有定量根基 → 继续 S2.a-pilot。
  - 若 `corr(q,e) ≈ corr(l,e)` 或更低 → **M 一阶近似在本任务失效** → **停止 S2.a/S2.c**，在交付里写明"M 叙事不被数据支持"，论文删去 M 的定量机制论证、只保留 seam 的经验观察。

### S2.a-pilot — decoder Jacobian top 谱（M 是否真各向异性）

仅在 S2.b-pilot 通过后跑。

- 复用同 N=16 个 `z*`（D20×8 + NORMAL×8）。
- 对每个 `z*` matrix-free 估计：
  1. **top-k 奇异值**（k≈8，randomized range finder / Lanczos on Mv），报 σ₁…σ₈。
  2. **Hutchinson trace** `tr(M) ≈ (1/m)Σ vᵢᵀMvᵢ`，m≈16 Rademacher 探针。
  3. **top-energy concentration**：`σ₁²/tr(M)`、`Σ_{i≤8}σᵢ²/tr(M)`（top 子空间占总能量比）。
- **禁止**报 bottom-k / 条件数 / 有效秩。
- **判读**：若 top-1/top-8 能量占比显著（例如 top-8 占 tr(M) 的 ≫ 8/196608 的均匀基线，量级上集中）→ M 各向异性成立，§3.2 "equal latent error ≠ equal image error" 从断言升级为实测。否则报"未观察到强各向异性"。

### S2.c-pilot — top-M 方向是否贴合 14px patch lattice（连接 M 与 seam）

仅在 S2.a-pilot 显示各向异性后跑。

- 对每个 `z*` 取 top-8 右奇异向量 `v₁…v₈`（来自 S2.a 的 range finder），对每个做 `u_i = J_G v_i`（jvp，得 image 域响应图，224×224）。
- 定义 patch-grid mask：14px 网格边界 ±1px 的像素集合（hard-unpatchify 接缝位置）。
- 量化 `u_i` 的能量在 patch-grid mask 内 vs 全图的占比，对比 **随机 latent 方向**的同一占比（matched control，相同 N）。
- **判读**：若 top-M 方向的 patch-grid 能量占比显著 > 随机方向 → "高-M 方向沿 patch lattice"hypothesis 被证实，§3.2 与图注的耦合可从 hypothesis 改回断言。若不显著 → 证伪，论文保留 seam 经验观察，删去"seam = top-M patch 对齐"的因果归因。

---

## 交付物

- 脚本：`tools/analyze_pullback_metric.py`（只读，含 `--stage {b,a,c}`、`--n`、`--timepoints` 开关）。
- `review/0603/server/S2b_pilot_correlation_20260604.{json,csv,md}`：q/e/l 三量、Pearson+Spearman、闸门判读。
- （若通过）`review/0603/server/S2a_pilot_spectrum_20260604.{json,csv,md}`：top-8 σ、Hutchinson trace、energy concentration。
- （若通过）`review/0603/server/S2c_pilot_patchgrid_20260604.{json,csv,md}`：top-M vs 随机方向的 patch-grid 能量占比 + 判读。
- 一份汇总 `review/0603/server/S2_PILOT_VERDICT_20260604.md`：三段闸门各自 PASS/FAIL，最终给论文 §3.2 / 图注的一句话措辞建议（"M 升级为实测" / "M 保留为 hypothesis" / "删去 M 定量机制"）。

## 成本预算

- S2.b-pilot：N=16 × (A4+V13) × 1 jvp ≈ 32 次 jvp，分钟级。
- S2.a-pilot：N=16 × (8 top-σ via ~20 Mv + 16 Hutchinson Mv) ≈ N×36 次 Mv，每次 1 jvp+1 vjp，十分钟级。
- S2.c-pilot：N=16 × 8 jvp + 随机对照，分钟级。
- 全部只读、单 GPU、无训练。先 N=16 出闸门结论，再决定是否扩 N=32。

## 一句话给执行者

先跑 S2.b-pilot 这道闸门。**相关性不达标就停，老实写"M 不被支持"**；达标再依次 S2.a → S2.c。每一步都可能证伪 M——证伪不是失败，是诚实结论，照写。论文已把 M 降级为 hypothesis，pilot 的唯一作用是决定它能不能升回断言。
