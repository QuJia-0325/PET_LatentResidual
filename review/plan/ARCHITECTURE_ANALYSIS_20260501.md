# PET_LatentResidual 总体架构分析（2026-05-01，2026-05-02 更新）

> **生成依据**：穷尽阅读 11 个 `pet_lr/` 模块、3 个顶层脚本、所有 `configs/`、`docs/`、`review/plan/`、`CLAUDE.md`、`AUTO_REVIEW.md`、`IDEA_REPORT.md` 与最新 V6/V6.1 训练快照（最新 2026-05-02 01:46:27）。
> **范围**：First-Hop Transport 主线（v3 → v6 → v6.1），不涵盖已废弃的 v2.1 残差精修分支。
> **目标读者**：项目作者、合作者、审稿人。
> **本文件不修改任何代码**，仅做事实性梳理 + 设计意图重建 + 当前状态盘点。

---

## 文档版本与导航（2026-05-02 整理）

本文件经过多日增补，**章节按时间分层**，后写的章节会修正/替换前面的中间结论。阅读时请优先看下表标记为 **AUTHORITATIVE** 的章节：

| 章节 | 主题 | 状态 |
|------|------|------|
| §0–§12 | 架构基线（仓库 / 数据 / 模型 / 损失 / 训练 / 评估 / 配置 / 历史 / 纪律 / 待解问题） | **AUTHORITATIVE**（项目静态结构） |
| §13 | 2024+ 顶级文献检索 + Idea 重估 + Related Work | **AUTHORITATIVE**（已压缩为最新版） |
| §14 | Trajectory Anchor Loss 深度对照（MAP-Diff vs 当前实现） | **AUTHORITATIVE** |
| §15 | Timestep 权重 $w(t)$ 是否值得借鉴——前置思考 | **HISTORICAL（思考过程）**；结论已合并到 §16/§18-§20 |
| §16 | Timestep 权重的数学根基 + Grönwall 引理推导 | **AUTHORITATIVE**（理论基础）；§16.5 后的行动条目已移到 §20.6 |
| §17 | V6 经验数据 vs Grönwall 单通道分析 | **SUPERSEDED**（仅保留 §17.1 权重口径；§17.2-§17.8 已被 §18-§19 替换） |
| §18 | V6 真实设计公式 $S = \text{Gap} \times v_\text{std}$ + 多 hop Grönwall 闭式解 | **AUTHORITATIVE**（emergent 命中发现，§18.3.4 已 2026-05-02 修订）|
| §19 | 用户确认：emergent 理论命中成立 + 论文 contribution 升级 | **PARTIALLY SUPERSEDED**（被 §21 修订）——§19 的 contribution 强度需按 §21.8 重写 |
| §20 | 整体 Loss 公式 + 梯度流动深入审查 + 5 个真实漏洞 + 优先级行动清单 | **AUTHORITATIVE**（最新行动入口） |
| §21 | **独立架构审视的关键发现（2026-05-02）**——§18.3.4 off-by-one 修正、pair[0]=2.5 真实动机、σ²·dt² 替代解释、chain MSE 横比警告、safety net 设计意图、STE 独立价值、β 来源验证、contribution 强度修订 + **§21.11 二次审计 5 点补充**（β 不是 SGD 反传权重、lambda_hop_init=0.10、pair endpoint 是 σ²不是 σ²dt²、emerge 措辞修正、chain MSE 归一化该用 (σdt)²） | **AUTHORITATIVE**（最高优先级，覆盖 §18-§19 的 emergent 命中叙事） |

**最新结论入口**：**§21.8（contribution 强度评估，2026-05-02 修订）** + §21.9（critical experiments 列表）+ §20.6（优先级行动清单）。
**理论卖点入口（修订后）**：§18.3.4（已修正 off-by-one + emergent 命中降级）+ §21.3（σ²·dt² 量级补偿替代解释）+ §21.9 行动 1（σ-normalize ablation 是判定的 critical experiment）。
**已实现 vs 待办盘点入口**：§12.4（IDEA_REPORT 候选改进的真实状态）。

---

## 0. 一页速读（TL;DR）

PET_LatentResidual 是一个**224 px 分辨率、单跳 First-Hop 形式的潜在空间 Mean-Flow 传输模型**，用 LoRA 微调的 DINOv2 + 冻结 ViT-MAE 解码器把多剂量 PET（D50→D20→D10→D4→NORMAL）级联到 NORMAL 域。

核心设计可以压成五条命题：

1. **Latent 是唯一 rollout 状态**。像素只在 hop0 作为条件输入；hop1+ 不再回 raw 域，避免链式 decoder 误差累积。
2. **共享主干 + Hop-Residual 速度头**。主干输出 v_shared，4 个 Hop-Residual 头（≤10% 主干参数）输出残差 v_hop_residual，乘以 softplus 门 λ_hop 加回主干。
3. **Hop0-Only Image Aux**。像素重建损失只在 hop0 计算（`decode_crop` + SeamRefiner），承担"对齐 latent 与像素分布"的局部任务；不参与 hop1+ 是经过 V4/V5 系列实验**显式排除**的设计选择。
4. **三段式 λ_roll 调度**。Phase I 锁 λ_roll=0（V6.1 改为 0.05 floor）专注 pair；Phase II 0→4 线性升；Phase III 锁 λ_roll=4 收敛。
5. **失败即回滚**。FOC-lite 先验在 NIGHTMARE 多评审中被 VETO（3.53/10），项目纪律：假设证伪即停。

当前状态（2026-05-02 01:46:27 快照）：

| 跑道 | GPU | step | best_step | best val_select | D20 mse | NORMAL mse | α | λ_roll | pair/roll/img |
|------|-----|------|-----------|-----------------|---------|------------|---|--------|---------------|
| V6   | gpu1 | **161 450** | 156 400 | **0.000610** | 0.000301 | 0.000229（best 0.000167） | 1.0 | 4.0 | 7 / 70 / 22 % |
| V6.1 | gpu3 | 49 100 | 12 000 | 0.000762 | 0.000613 | 0.000596（Phase I best 0.000213） | 0 | 0.05 | 90 / 1 / 9 % |

**V6 已稳定进入 Phase III（α=1, λ_r=4）**，自上一个快照（2026-04-30）+78 600 step：best val_select 改进 −7.7%（0.000661→0.000610），best chain_normal_mse 改进 −8.2%（0.000182→0.000167）。Phase III 内 29 条 val 记录振荡幅度 max/min ≈ 2.57×，需要 EMA 或多 ckpt 组合稳定。

**V6.1 处于 Phase I 末期（即将进入 Phase II）**，当前与 V6 同期对比表现良好——详见 §20.7。早期把 V6 Phase III best 与 V6.1 Phase I 同步比较的判断**已被推翻**：实测 V6.1 在 step ≤ 50K 范围内 head-to-head 胜 V6 11/13 次（85%），Phase I best (0.000213@12K) 比 V6 Phase I best (0.000224@6K) 好 4.9%。floor=0.05 产生的 roll_frac mean=1.24% 与数学预测 1.2% 完美吻合。**真正的判定窗口在 V6.1 step 50K-150K（Phase II 起势速度），当前数据不支持"V6.1 失败"结论**。

> **⚠️ chain MSE 横比警告（2026-05-02 增补 + 二次审计修正）**：上表中 D20 mse 与 NORMAL mse **不可直接横比**——rollout step error 的自然量级是 $(\sigma_k \cdot dt_k)^2$，跨 hop 差 7.5×（[8.35e-4, 2.17e-4, 1.35e-4, 1.12e-4]）。要做跨 hop 对比，应使用 **$(\sigma_k \cdot dt_k)^2$-归一化** 或 **target variance-归一化** chain MSE，不是只除 $\sigma_k^2$。这同时影响 §0 后续叙述里所有"chain_normal 比 D20 低 24%"之类的语言——它们是 $(\sigma \cdot dt)^2$ 量级主导，不是模型主导。

详见 §21.4 + §21.11.5（二次审计修正）。

**架构命题修订（2026-05-02）**：原命题 2 描述"共享主干 + Hop-Residual 速度头"为主力模块。V6 step 161K 实测 lambda_hop_0=0.059（yaml init=0.10）, gate_pix=0.037（yaml init=0.02）, v_hop_abs=0.000238——hop_residual / pixel forcing 在训练中**始终保持在低激活状态附近**（既没有爆涨也没有衰退到 floor=0.001），相对 v_total 贡献 ≪ 1%。**这是设计意图**（用户确认）：hop_residual 是 backbone 的局部修正，不是主力。命题 2 的"加回主干"措辞应理解为"小幅补偿"，不是"等量贡献"。

---

## 1. 仓库布局与文件职责

```
PET_LatentResidual/
├── CLAUDE.md                       # 项目硬规则与 Ledger（必读）
├── README.md                       # 入口说明
├── AUTO_REVIEW.md                  # NIGHTMARE 多评审会话记录（FOC-lite 否决）
├── IDEA_REPORT.md                  # Idea 1/2/3 评分（CCT-224 8.4，ΔB-aware 7.6，Uncertainty HOLD）
├── train_first_hop.py              # 2457 行，唯一训练入口
├── eval_first_hop_224_clip3.py     # 322 行，PSNR/SSIM 评估（calc_psnr_clip3）
├── check_alignment_224_clip3.py    # 344 行，对齐审计（matched/roll1/random）
│
├── pet_lr/                         # 主代码库（11 个模块，~2000 行）
│   ├── __init__.py
│   ├── bootstrap.py                # sys.path 注入（兼容服务器布局）
│   ├── path_guard.py               # 输出路径强约束 → /data_2/qujiaxiang/outputs/...
│   ├── data.py                     # 旧 v2.1 像素域数据集（保留兼容）
│   ├── data_first_hop.py           # 主数据集 PETFirstHopAligned4HopDataset (357 行)
│   ├── model.py                    # 旧 v2.1 残差精修模型（200 行，已停产）
│   ├── model_first_hop.py          # 主模型 PETFlowDiTFirstHop (601 行)
│   ├── rollout.py                  # 旧 v2.1 rollout（已停产）
│   ├── rollout_first_hop.py        # 主 rollout（159 行）
│   ├── losses.py                   # SSIM / weighted_l1 / seam (198 行)
│   ├── losses_first_hop.py         # compute_first_hop_image_loss (60 行)
│   └── ema.py                      # EMA 装饰器（125 行，decay=0.9999）
│
├── configs/                        # YAML 配置矩阵
│   ├── pet_flow_first_hop_224_200k_transport_v3.yaml   # V3 200K 参考线
│   ├── pet_flow_first_hop_224_v6_transport_first.yaml  # V6 主跑道
│   ├── pet_flow_first_hop_224_v6_1_rollout_floor.yaml  # V6.1 单变量分支
│   └── （V4/V5 历史配置）
│
├── docs/
│   ├── main.md                     # 742 行，规范性最强的实现说明
│   ├── background_first_hop_design.md   # 324 行，latent-only state 论证
│   └── background_v21_residual_refinement.md  # 150 行，旧 v2.1 立项材料
│
├── review/                         # 历史审计与计划
│   ├── plan/
│   │   ├── pixel_prior_in_latent_transport.md
│   │   ├── transport_breakthrough_research_v3..v6.md
│   │   └── ARCHITECTURE_ANALYSIS_20260501.md   ← 本文件
│   ├── 0428/, 0429/, 0430/         # 日志快照
│
└── scripts/, tools/                # 启动 & 监控辅助
```

**文件总规模**：核心代码 5 054 行（含 `train_first_hop.py` 2 457 行）、规范文档 ~1 200 行、规划档案 ~1 500 行。

---

## 2. 数据管线

### 2.1 PETFirstHopAligned4HopDataset（`pet_lr/data_first_hop.py`）

**职责**：返回 4 对 (z_src, z_tgt) latent 对 + 5 张 raw 像素（用于 hop0 条件 / 像素监督）。

```
slices on disk:
  D50.pt   D20.pt   D10.pt   D4.pt   NORMAL.pt   （都是 .pt latent + 对应 raw nii/png）

per __getitem__ returns:
  pairs = [
      (z_D50,    z_D20,    t=2.0  → 5.0,   pair_idx=0),
      (z_D20,    z_D10,    t=5.0  → 10.0,  pair_idx=1),
      (z_D10,    z_D4,     t=10.0 → 25.0,  pair_idx=2),
      (z_D4,     z_NORMAL, t=25.0 → 100.0, pair_idx=3),
  ]
  raws = [raw_D50, raw_D20, raw_D10, raw_D4, raw_NORMAL]   # 仅 hop0 用 raw_D50
```

关键步骤：

* **alignment 审计 fail-fast**：`_load_alignment_audit` 读取 `alignment_audit.json`；若 `alignment_signal_positive=false` 直接 raise，禁止"用噪声 latent 训练"。
* **Raw 归一化**：`raw.clamp(0, 10) / 10 * 2 - 1 → [-1, 1]`，bicubic 224×224。
* **Latent 通道**：768，分辨率 16×16（DINOv2 patch=14, image=224）。
* **Hop0OnlyViewDataset**：`pair_idx==0` 过滤器，给 hop0 batch every-step 路径用。

### 2.2 配额与采样

* **batch every step**：每个 step 同时给一份 hop0-only batch（保 hop0 像素监督）+ 一份 full-4-pair batch（保 transport）。
* **shuffle / num_workers / pin_memory** 由 yaml 配置；V6 标准 `batch_size=8, num_workers=4`。

---

## 3. 模型架构（`pet_lr/model_first_hop.py`，601 行）

总入口 **`PETFlowDiTFirstHop`**，五个子模块：

```
PETFlowDiTFirstHop
├── backbone : PETFlowDiTDHHopAware       # 共享主干，~XXM 参数
├── pixel_encoder : FirstHopPixelEncoder  # CNN → 768 通道，仅 hop0 启用
├── velocity_head : HopResidualVelocityHead  # 4 个 hop 残差头
├── alignment_projector : SpatialAlignmentProjector  # 对齐 hook（block 6）
└── seam_refiner : SeamRefiner            # 边界缝合精修（hop0 解码后可选）
```

### 3.1 Backbone：`PETFlowDiTDHHopAware`

* `input_size = 16`（latent 空间分辨率）
* `in_channels = 768`，`patch_size = 1`（每个 latent token = 1 patch）
* **Dual-Hidden** 设计：浅层 hidden=384，深层 hidden=2048
* depth=[12, 2]（12 浅 + 2 深 transformer block）
* num_heads=[6, 16]，mlp_ratio=4.0
* **Hop-Aware**：把 hop_idx (0-3) 嵌入成 timestep token 加到序列首
* LoRA 可训练：默认 r=16, α=16

### 3.2 FirstHopPixelEncoder

* 输入：raw_D50 224×224×1，`[-1, 1]`
* 4 层 Conv + GroupNorm + GELU，downsample 到 16×16
* 自适应 pool + `1×1 Conv → 768` 通道
* **最后一层 zero-init** → 训练初期 pixel 路径默认关闭，保证收敛稳定
* **门控**：`gate_pix = floor + softplus(raw)`，floor 默认 0.0，softplus 起手 ~0.69
* **仅 hop0 注入**：`if hop_idx == 0: x = x + gate_pix * pixel_token else: pass`
* 设计纪律：参数量上限 ≤ backbone 5%

### 3.3 HopResidualVelocityHead

```
shared_head:    Linear(768 → 768) + GELU + Linear(768 → 768)
hop_head[k]:    Linear(768 → 256) + GELU + Linear(256 → 768)   for k in {0,1,2,3}
gate_hop[k]:    floor + softplus(raw_param[k])
```

forward：
```
v_shared = shared_head(h)
v_hop_k  = hop_head[hop_idx](h)
v_total  = v_shared + gate_hop[hop_idx] * v_hop_k
```

* **每个 hop_head 最后一层 zero-init** → λ_hop 初始有效幅度 = 0
* 所有 hop_head 总参数 ≤ shared_head 10%
* 这是模型对**多 hop 之间动态系统差异**的唯一显式刻画

### 3.4 SpatialAlignmentProjector（对齐 Hook）

* 注册到 backbone 第 6 个 transformer block 输出
* 结构：`Conv2d(384 → 256) → Conv2d(256 → 768)`，bilinear upsample
* 与 z_target 计算 cosine 距离 → `align_loss`
* α_align = `1.0 - exp(-step / 30 000)` ramp，最大 1.0

### 3.5 SeamRefiner（解码后精修）

* depthwise-separable conv，kernel=7
* num_blocks=3, hidden=32
* 输出 tanh-clamped residual，`max_residual=0.15`
* **仅作用于 hop0 解码后的像素图**；通过 `decode_crop(skip_refiner=True)` 控制 ablation

---

## 4. 前向流程（端到端）

以 hop0（D50→D20）为例：

```
1. raw_D50 (B,1,224,224)  z_D50 (B,768,16,16)
                                    ↓                ↓
                          FirstHopPixelEncoder       (无修改)
                                    ↓
                          pixel_token (B,768,16,16)
                                    ↓
                          x = z_D50 + gate_pix · pixel_token   ← 仅 hop0
                                    ↓
2. backbone(x, t_src=2.0, hop_idx=0) → h (B,768,16,16)
                                    ↓
3. velocity_head(h, hop_idx=0)
       v_shared = shared_head(h)
       v_hop0   = hop_head[0](h)
       v_total  = v_shared + gate_hop[0] * v_hop0
                                    ↓
4. dt = (t_tgt - t_src) / scale       [Mean-Flow 显式时间步]
       z_pred = z_D50 + dt * v_total  [target_normalize=true 下做归一化]
                                    ↓
5. Loss 计算
       L_pair  = vel_err(v_total, v_target) + end_err(z_pred, z_D20)
       L_align = cos_dist(alignment_proj(h6), z_target)
       (hop0) decode_crop(z_pred) → img_pred → SeamRefiner →  img_refined
       L_img  = 1.0·L1 + 0.25·SSIM + 0.10·seam(border_w=14, border_weight=2.0)
```

hop1/2/3 的差异仅在两点：
- 不注入 pixel_token（`gate_pix · pixel_token` 项不参与）
- 不计算 `L_img`

---

## 5. 损失合成

总损失（每 step）：

$$
\mathcal{L} = \lambda_p \cdot \mathcal{L}_\text{pair} + \lambda_r(t) \cdot \mathcal{L}_\text{roll} + \lambda_i \cdot \mathcal{L}_\text{img}^\text{hop0} + \lambda_a(t) \cdot \mathcal{L}_\text{align} + \lambda_f \cdot \mathcal{L}_\text{foc}
$$

| 项 | 权重（V6） | 计算位置 | 来源文件 |
|----|-----------|----------|----------|
| `L_pair` | λ_p = 15 | 4 对 (z_src, z_tgt) 的 vel_err + end_err，加权 `pair_loss_weights=[2.5, 1.0, 1.0, 1.0]`，乘 `step_weights=[0.5, 2.0, 1.5, 1.0]` | `train_first_hop.compute_pair_losses` |
| `L_roll` | λ_r 0→4 三段 | rollout 链 hop0→hop3 的 step-wise 误差，alpha 线性升 | `train_first_hop.compute_rollout_losses` |
| `L_img` | λ_i = 0.04 (V6 固定) | `compute_first_hop_image_loss(hop0_only=True)` | `pet_lr/losses_first_hop.py` |
| `L_align` | α_align ramp 0→1 | cosine(align_proj(h6), z_target) | `model_first_hop.alignment_hook` |
| `L_foc` | 0（NIGHTMARE 否决） | FOC-lite 半步先验 | `train_first_hop.compute_foc_losses`（保留但禁用） |

补充机制：

* **velocity_rebalance**：`sqrt_ratio_clip(1, 4)`，把 4 对的 vel_err 量级拉齐到主导对，避免 D50 对支配训练。
* **straight_through_mode='fixed'**（V6 默认）：rollout 时用 `z_rollout_first.detach() * st_scale + z_real * (1-st_scale)` 做 straight-through，st_scale=0.0 即纯 detach。
* **lambda_scale_mode='running_mean'**：λ_roll 的 effective magnitude 随 mean(|L_roll|) 自适应。
* **best 选择**：`val_multi_objective`，权重 `d20:1.0, d10:0.30, d4:0.30, normal:0.45`；同时记录 Best-D1 lane。

---

## 6. 训练脚本剖析（`train_first_hop.py`）

**2 457 行的"全功能控制塔"**，主要功能模块：

| 区段 | 行数大致 | 职责 |
|------|---------|------|
| Imports / utils | 1–200 | 路径守卫、随机种子、日志 |
| Dataset & DataLoader | 200–400 | 双 batch（hop0-only + full-4-pair）|
| `compute_pair_losses` | 400–700 | 4 对 vel + end，velocity_rebalance |
| `compute_rollout_losses` | 700–1 000 | linear-α rollout，straight-through |
| `compute_self_forcing_z_src` | 1 000–1 100 | 已大半禁用 |
| `compute_foc_losses` | 1 100–1 250 | FOC-lite（VETO 后保留壳） |
| `compute_hop0_image_losses` | 1 250–1 400 | decode_crop + SeamRefiner + L_img |
| `evaluate` | 1 400–1 700 | val 64-batch、unique-slice chain MSE、PSNR_clip3 |
| `resolve_best_selection_score` | 1 700–1 800 | 多目标 best 选择 |
| `strict_resume_compat` | 1 800–1 900 | arch-change warm-start，仅放行 `seam_refiner.` / `alignment_projector.` |
| Watchdogs | 1 900–2 100 | dead_branch / img_loss_zero / gate_explosion / hop0_coverage / loss_balance_dominance |
| Main loop | 2 100–2 457 | EMA、AMP、ckpt、日志、调度器 |

### 6.1 Watchdog 列表

* **dead_branch**：连续 N step `gate_pix < 1e-3` → ALERT
* **img_loss_zero**：`L_img` 连续 50 step ≈ 0 → ALERT
* **gate_explosion**：任何 gate > 5.0 → ALERT
* **hop0_coverage**：当前窗口 hop0 占比 < 20% → ALERT
* **loss_balance_dominance**：单项 loss 占比 > 70% 持续 100 step → ALERT

### 6.2 EMA / AMP / ckpt

* `pet_lr/ema.py`：decay=0.9999；`average_parameters()` context manager 用于 val。
* AMP：bfloat16，`autocast(device_type='cuda', dtype=torch.bfloat16)`。
* checkpoint：每 `ckpt_interval`（默认 5 000）存 step 与 best；resume 通过 `strict_resume_compat` 处理 arch 变更。

---

## 7. 评估管线

### 7.1 `eval_first_hop_224_clip3.py`（322 行）

* 全链 decode（D50→D20→D10→D4→NORMAL）
* 每张切片用 `src.utils.metrics.calc_psnr_clip3`（CLAUDE.md 强制规范，clip 至 max=3 后计算）
* `decode_mode ∈ {default, raw, both}`：
  - `default`：经 SeamRefiner
  - `raw`：跳过 refiner
  - `both`：两路都给（refiner ablation 必用）

### 7.2 `check_alignment_224_clip3.py`（344 行）

* 加载 RAE + LoRA，把 latent 解到像素
* 三路对照：matched / roll1 / random
* 输出 `alignment_signal_positive` 标志，是 dataset 加载的 fail-fast 依据

### 7.3 监控 KPI（review/0430 快照）

| 指标 | 含义 | V3 200K best | V6 当前 | V6.1 当前 |
|------|------|---------|---------|-----------|
| val_select | 多目标加权 mse | 0.000693 (step 86 800) | **0.000661** | 0.000762 |
| D20 mse | 链式 D20 步 | 0.000247 | 0.000238 | 0.000260 |
| NORMAL mse | 链终点 | 0.000204 | 0.000182 | 0.000213 |
| pair / roll / img 占比 | 三大 loss 比例 | 36/32/32 | 53/21/26 | 25/3/71 |

---

## 8. 配置矩阵

### 8.1 V3（参考线）`pet_flow_first_hop_224_200k_transport_v3.yaml`

* 总步数 200 000
* λ_p = 10, λ_roll 全程线性升 0→3, λ_i = 0.04 固定
* 单 phase（无 phase 切换）
* **历史 best**：step 86 800，val_select 0.000693，roll_frac=32%
* **缺陷**：之后 image_aux 占比慢慢回升到 ~50%，pair 退到 ~30%，转向 image dominance，val_select 长期不再下行

### 8.2 V6（主跑道）`pet_flow_first_hop_224_v6_transport_first.yaml`

* 三段式 200 000 步：Phase I 50 K / Phase II 100 K / Phase III 50 K
* λ_p = **15**（V3 的 1.5×）
* λ_roll：Phase I = **0**（严格关闭）; Phase II = 0→4 线性; Phase III = 4 固定
* λ_i = **0.04** 固定（与 V3 同）
* `pair_loss_weights = [2.5, 1.0, 1.0, 1.0]`（强化 D50→D20 对）
* `step_weights = [0.5, 2.0, 1.5, 1.0]`（rollout 中段更重要）
* `straight_through_mode = fixed`，`st_scale = 0.0`
* **设计动机**（见 `transport_breakthrough_research_v6.md`）：把 V3 best 的损失分布（pair 36 / roll 32 / img 32）反向工程，结论是 **pair 必须先收敛、roll 才能稳定升压**

### 8.3 V6.1（单变量分支）`pet_flow_first_hop_224_v6_1_rollout_floor.yaml`

* V6 的唯一差异：Phase I `λ_roll_start = 0.05`（不再是严格 0）
* 动机：V6 在 step ~40 K 出现 chain composition regression（hop1/2/3 链式误差不再下降），假设 **λ_roll=0 让链组合信号完全断流**；V6.1 用 0.05 floor 提供最小链监督
* 其他超参完全继承 V6
* 当前 step 20 950，仍在 Phase I floor 模式

---

## 9. 历史实验账本（按时间序）

| 阶段 | 关键变量 | 结果 / 教训 |
|------|---------|-------------|
| **10 K 探针** | λ_roll=0, λ_i=0 | 验证 pair 单项可降；alignment_audit 通过 |
| **50 K rampup** | λ_roll 0→3 | rollout 项可学习，但与 pair 抢权 |
| **V3 200 K** | 全程线性 ramp | best @ 86 800（roll_frac=32%）；之后 image dominance 回潮 |
| **V4 系列** | 试图让 image_aux 在 hop1+ 也参与 | 解码 stitching 伪影 → 像素监督在非 hop0 反而拉低 latent 质量 → 排除 |
| **V5 NC** | 引入 noise consistency | 在 73% 预算（步 ~146 K）停止；导师批准；释放 GPU |
| **FOC-lite 实验** | 半步先验补偿 | ODE 集成误差 0.17%（可忽略）；半步预测反而更差；NIGHTMARE 多评审 3.53/10 + double VETO → **完全否决** |
| **V6** (current) | λ_p=15、三段、Phase I λ_roll=0 | 在 step 81 600 取得 val_select=0.000661，**已超过 V3 best** |
| **V6.1** (current) | V6 + λ_roll floor 0.05 | 仍在 Phase I 早期，best @ 12 000 = 0.000762，需要等 Phase II 才能与 V6 对照 |

E1 budget 分解（来自 transport_breakthrough_research_v6）：

* 总误差 100%
* 其中 transport（latent 链式偏移）= **96–99%**
* 其中 decoder（ViT-MAE）≈ 0.4 dB（< 4%）
* 结论：**优化重点是 latent transport，不是解码器**

---

## 10. 硬规则与纪律（`CLAUDE.md` 提炼）

1. **PSNR 必须用 `src.utils.metrics.calc_psnr_clip3`**：clip 至 max=3 后计算，禁止用 raw PSNR。
2. **alignment_audit fail-fast**：dataset 加载时若 `alignment_signal_positive=false` 直接 raise。
3. **freeze_rae=true**：DINOv2 + ViT-MAE decoder 参数全冻结，仅 LoRA 可训。
4. **hop0 batch every step**：每步必须有 hop0-only sample，否则 watchdog 触发。
5. **outputs 强制 `/data_2/qujiaxiang/outputs/PET_LatentResidual/...`**：`pet_lr/path_guard.py` 在 import 阶段断言。
6. **rae conda env 固定**：`/home/qujiaxiang/.conda/envs/rae/bin/python`，不允许换。
7. **NIGHTMARE 协议**：任何新 idea 进入实验前必须经多评审 + 红队，VETO 即停（FOC-lite 是已落地案例）。
8. **arch-change resume 白名单**：仅 `seam_refiner.` 与 `alignment_projector.` 前缀允许 warm-start，其它结构变化必须 from-scratch。

---

## 11. 设计纪律：为什么是"latent-only state + hop0-only image_aux"

这是项目最容易被审稿质疑的地方，重建一遍论证链：

### 11.1 为什么 latent 是唯一 rollout 状态

* 若 hop1+ 也回 raw 域，需要 `decode → encode` 闭环；ViT-MAE decoder 不可逆 + LoRA-DINOv2 不是恒等映射，这条闭环误差是 **乘性累积**。
* 实测（review/0428 alignment）：解码-再编码后 latent 的 cosine 相似度 < 0.85，而 latent 链式预测的 cosine 相似度 > 0.97 → latent 链信噪比高一个数量级。
* 因此设计：**latent 链只在 latent 空间 rollout**，最后只解码一次给出像素结果。

### 11.2 为什么 image_aux 只在 hop0

* image_aux 是**局部任务**：把 hop0 latent 的解码图压向 raw_D50，约束 latent 与像素分布的对齐。
* 若把 image_aux 推到 hop1+：
  - 多 hop image loss 与 latent transport loss 的量级、梯度方向冲突，难以同时收敛
  - 需要在每个 hop 都做 `decode_crop`，开销 ×4，且每次解码都引入 stitching artifacts
  - V4 实验显式验证：hop1+ image loss 反而**降低 latent 质量**（可视化看到 patch boundary 缝合伪影）
* 结论：image_aux 是 latent 训练的"对齐辅助"，不是"重建目标"——它的设计角色就是只在 hop0 起作用。

### 11.3 为什么共享主干 + 残差速度头

* 4 hop 全部独立主干 → 参数 ×4，且 hop 之间无知识迁移
* 4 hop 共享单一速度头 → 无法刻画各 hop 的动态差异（D50→D20 的 t-step=3 vs D4→NORMAL 的 t-step=75）
* 折中：**主干共享，速度头分 hop**，且残差幅度通过 softplus 门控制 → 既共享底层表示，又允许 hop-specific 速度修正

---

## 12. 待解问题与下一步

### 12.1 直接来自审稿/导师反馈

* **B3：多阶段 PET 重建的物理约束**——需要从 PET 物理（光子计数 Poisson、衰减矫正、PVE）出发解释为什么"低剂量→高剂量"的级联是物理可解的，目前文档仅描述工程化级联，缺物理建模一节。
* **B4：MICCAI/TMI 的具体修改建议**——pixel_prior 计划（review/plan/pixel_prior_in_latent_transport.md）的 Level 1.5 LPIPS baseline 仍未跑；缺 BraTS-style ablation table。

### 12.2 视觉证据待补

* `review/0425/visuals/first_hop_224_200k_transport_v3_best_clip3/` 的可视化未做系统性对比（hop0 vs hop1+ 解码图、是否能看到 stitching 伪影），是 image_aux 排除论证的关键证据。

### 12.3 实验进行中

* V6 进入 Phase II 上升期，需要观察：
  - α_align 是否在 0.5 附近平稳，避免 alignment 项喧宾夺主
  - λ_roll 上到 4 后 chain MSE 是否进一步下降
  - 关键 gate：step 100 K（Phase II 中点）、150 K（Phase III 起点）
* V6.1 仍需等 Phase II 才能与 V6 拉直接对比；目前 Phase I 数据不足以判断 floor 假设是否成立。

### 12.4 候选改进（来自 `IDEA_REPORT.md` 2026-04-09 Gate-1 评审）

> **2026-05-02 状态更新**：原 IDEA_REPORT 三条 Idea **目前都未合入 master 主线**，但项目主线在此期间**走出了与 IDEA_REPORT 不同的方向**（emergent Grönwall 理论命中，见 §18-§19）。最新可执行优先级见 §20.6。

| # | Idea | 当前状态 | 与主线关系 |
|---|------|---------|-----------|
| 1 | CCT-224 (Counterfactual Consistency Training) | 在测试分支 `idea1_cct_test`，**未合 master** | 仍可作为后续 phase 增量；优先级低于 §20.6 行动 6+7 |
| 2 | ΔB-aware Adaptive Reweighting | **未实施**；与现有 `velocity_rebalance`（局部 sqrt-ratio）不同——后者在 V6 配置下实测为死代码（§20.4 #2） | 暂缓 |
| 3 | Uncertainty-Gated Hop0 Forcing | **未实施**，HOLD 状态 | 暂缓 |

**注**：IDEA_REPORT 三项的详细 Proposer/Skeptic 论证见 [IDEA_REPORT.md](PET_LatentResidual/IDEA_REPORT.md)。本架构分析文件不再重复其内容。

---

## 附录 A：关键超参速查表（V6 配置）

```yaml
# 模型
backbone:
  input_size: 16
  in_channels: 768
  hidden_size: [384, 2048]
  depth: [12, 2]
  num_heads: [6, 16]
  patch_size: 1
  mlp_ratio: 4.0
lora:
  r: 16
  alpha: 16
pixel_encoder:
  zero_init_last: true
  gate_pix_floor: 0.0
velocity_head:
  hop_head_hidden: 256
  zero_init_last: true
  gate_floor: 0.0
seam_refiner:
  num_blocks: 3
  hidden: 32
  kernel_size: 7
  max_residual: 0.15

# 损失
loss:
  pair_weight: 15
  pair_loss_weights: [2.5, 1.0, 1.0, 1.0]
  step_weights: [0.5, 2.0, 1.5, 1.0]
  rollout:
    schedule: three_phase
    phase1_steps: 50000
    phase2_steps: 100000
    phase3_steps: 50000
    lambda_roll_phase1: 0.0       # V6.1 = 0.05
    lambda_roll_phase2_end: 4.0
    lambda_roll_phase3: 4.0
    lambda_scale_mode: running_mean
    straight_through_mode: fixed
    st_scale: 0.0
  image_aux:
    weight: 0.04
    hop0_only: true
    w_l1: 1.0
    w_ssim: 0.25
    w_seam: 0.10
    border_width: 14
    border_weight: 2.0
  alignment:
    block_idx: 6
    ramp_steps: 30000

# 训练
optimizer:
  type: AdamW
  lr: 1.0e-4
  weight_decay: 0.01
ema:
  decay: 0.9999
amp:
  dtype: bfloat16
val:
  max_val_batches: 64
  multi_objective:
    d20: 1.0
    d10: 0.30
    d4: 0.30
    normal: 0.45
```

## 附录 B：训练曲线读法

* **pair / roll / img 占比**：理想是接近 V3 best 时的 36/32/32；V6 当前 53/21/26 偏 pair-heavy（Phase II 早期正常），随 λ_roll 升压会向 33/40/27 靠拢。
* **α_align**：该值线性升到 1.0；若超过 0.7 就要警惕 alignment 项压住主任务。
* **λ_hop**（4 个 softplus 门）：初始全 ~0.69，理想轨迹是 4 个值各自分化（不同 hop 学到不同幅度）；如果 4 个值长期粘在一起→残差头无效。
* **gate_pix**：初始 0.69；hop0 信号强则上升，但应 < 5.0；watchdog `gate_explosion` 在 5 触发。
* **chain MSE per hop**：D20 应 < 0.0003；D10/D4 略高；NORMAL 应回落到 < 0.00025（V6 已达成）。

## 附录 C：文件清单（精确行数）

| 文件 | 行数 |
|------|------|
| `pet_lr/__init__.py` | 13 |
| `pet_lr/bootstrap.py` | 20 |
| `pet_lr/data.py`（v2.1 旧路径） | 111 |
| `pet_lr/data_first_hop.py` | **357** |
| `pet_lr/ema.py` | 125 |
| `pet_lr/losses.py` | 198 |
| `pet_lr/losses_first_hop.py` | 60 |
| `pet_lr/model.py`（v2.1 旧路径） | 200 |
| `pet_lr/model_first_hop.py` | **601** |
| `pet_lr/path_guard.py` | 29 |
| `pet_lr/rollout.py`（v2.1 旧路径） | 58 |
| `pet_lr/rollout_first_hop.py` | 159 |
| `train_first_hop.py` | **2 457** |
| `eval_first_hop_224_clip3.py` | 322 |
| `check_alignment_224_clip3.py` | 344 |
| **核心代码合计** | **5 054** |

---

**结语**：该项目处于"假设驱动 + 多评审 + 增量实验"的健康状态。当前 V6 已经在 step 81 600 取得 val_select=0.000661，**优于 V3 200 K best（0.000693）**，主线设计可证伪、可复现、可比较。下一步关键节点是 V6 step 100 K（Phase II 中点）与 V6.1 进入 Phase II 后的直接对照。

---

# 13. 2024+ 顶级文献检索：邻居论文与 idea 重估（2026-05-01 增补）

> **检索方式**：通过 arXiv API 在 2026-05-01 直接发起 8 路检索（按 `submittedDate` 降序、年份 ≥ 2024），覆盖 PET 去噪、Mean Flow、Self-Forcing/Exposure Bias、自适应损失加权、级联医学生成、RAE/DINOv2 医学应用、医学一致性模型。每路检索取 12 条，相关度过滤后入此节。**未做 Semantic Scholar / 期刊版本检索**——仅 arXiv 视角，TMI/MICCAI/Med IA 的最新刊物结果需要补刀。

## 13.1 与本项目最近的 PET 多剂量论文（必须正面应对）

这是论文撰写时**审稿人最可能拿来对比**的几篇近邻：

| ID | 时间 | 标题 | 与本项目的关系 |
|----|------|------|---------------|
| [arXiv:2603.02012](https://arxiv.org/abs/2603.02012) | 2026-03 | **MAP-Diff**: Multi-Anchor Guided Diffusion for **Progressive** 3D Whole-Body Low-Dose PET Denoising | **最直接竞品**。也使用"中间剂量作为 trajectory anchor + timestep 加权 anchor loss"。报告内部 PSNR 42.48 → 43.71 dB（+1.23 dB），SSIM 0.986。**3D 全身**而非 2D；anchor 通过 degradation matching 校准而非显式 t_map。 |
| [arXiv:2604.16925](https://arxiv.org/abs/2604.16925) | 2026-04 | **Rethinking Cross-Dose PET Denoising**: Mitigating Averaging Effects via Residual Noise Learning | 直接命中"单剂量训练在跨剂量泛化失败"这一痛点；论证了"`one-size-for-all` 模型隐含地优化跨剂量分布的期望，导致平均化"——和我们 V3 image dominance 现象同源。方案是**残差噪声学习**（直接预测噪声，不预测全剂量图）。两中心多剂量数据集。 |
| [arXiv:2505.05112](https://arxiv.org/abs/2505.05112) | 2025-05 | **MDAA-Diff**: CT-Guided **Multi-Dose Adaptive Attention** Diffusion Model for PET Denoising | "Dose-Adaptive Attention (DAA)"——把剂量级显式喂进 channel-spatial attention 权重计算，与我们 `HopResidualVelocityHead` 的 hop-aware 思路同构。但需要 **CT 引导**，我们没有 CT。18F-FDG + 68Ga-FAPI 双示踪剂。 |
| [arXiv:2505.22489](https://arxiv.org/abs/2505.22489) | 2025-05 | **Cascaded 3D Diffusion** for Whole-body 3D 18-F FDG PET/CT synthesis from Demographics | 级联结构（low-res → super-res 残差），但任务是从人口学合成 PET/CT，不是去噪。 |
| [arXiv:2509.13614](https://arxiv.org/abs/2509.13614) | 2025-09 | **Generative Consistency Models for Kinetic Parametric Image Posteriors in Total-Body PET** | Consistency model 用于 TB-PET 动力学参数图后验估计；和我们 latent flow 思路相邻但任务不同。 |
| [arXiv:2603.09075](https://arxiv.org/abs/2603.09075) | 2026-03 | **M2Diff**: Multi-Modality Multi-Task Enhanced Diffusion for MRI-Guided LD PET | 多模态多任务 + MRI 引导；多模态条件路线。 |
| [arXiv:2603.09931](https://arxiv.org/abs/2603.09931) | 2026-03 | **ACADiff**: Adaptive Clinical-Aware Latent Diffusion for Multimodal Brain Image Generation | 自适应临床条件的潜在扩散，缺失模态填补。 |

**关键风险**：MAP-Diff 已经把"progressive multi-dose anchor"这条故事线**讲了一遍**，并报告 +1.23 dB PSNR。我们必须明确答出：

1. 我们的 4-hop **physical t_map (2/5/10/25/100)** 是**确定性的、由 PET 物理给出的**（剂量比例直接对应 t），而 MAP-Diff 的 anchor timestep 是通过"degradation matching"经验校准的——这是一个**机理 vs 经验**的清晰差异。
2. 我们在 **latent 空间 rollout**（DINOv2 latent + 冻结 ViT-MAE 解码），MAP-Diff 在像素空间扩散——我们的 transport 误差 = 96–99% 的总误差预算（来自 `transport_breakthrough_research_v6.md`），这是 latent 空间专属的优化空间。
3. 我们用 **Mean Flow（v 预测）而非 DDPM**——单步采样，推理代价 ≪ MAP-Diff 的 DDPM 反向链。

## 13.2 Mean Flow 在 2026 已成方法主线（验证我们方法的"合时性"）

| ID | 时间 | 标题 | 启示 |
|----|------|------|------|
| [arXiv:2604.24586](https://arxiv.org/abs/2604.24586) | 2026-04 | **Point-MF**: One-step Point Cloud Generation via Mean Flows | Mean Flow → 3D 点云 |
| [arXiv:2603.20690](https://arxiv.org/abs/2603.20690) | 2026-03 | **MFSR**: MeanFlow Distillation for One-Step Real-ISR | Mean Flow → 超分 |
| [arXiv:2603.01469](https://arxiv.org/abs/2603.01469) | 2026-03 | **Mean-Flow based One-Step VLA** | Mean Flow → 机器人动作 |
| [arXiv:2602.18104](https://arxiv.org/abs/2602.18104) | 2026-02 | **MeanVoiceFlow**: One-step Voice Conversion with Mean Flows | Mean Flow → 语音 |

**结论**：Mean Flow 在 2026 上半年已扩散到 SR / 3D / 机器人 / 语音四个域，但**尚未见到任何 PET / 医学影像应用**。我们若按时投出，**"首例 Mean Flow 应用于多剂量 PET 级联去噪"** 仍是一个干净的方法论 claim。

## 13.3 Self-Forcing 已成 2026 视频扩散范式族（Idea 1 必须重新定位）

| ID | 时间 | 标题 | 与 CCT-224 的关系 |
|----|------|------|------------------|
| [arXiv:2603.21366](https://arxiv.org/abs/2603.21366) | 2026-03 | **Relax Forcing**: Relaxed KV-Memory for Consistent Long Video | 自我修正 AR 视频 |
| [arXiv:2604.25819](https://arxiv.org/abs/2604.25819) | 2026-04 | **Mutual Forcing**: Dual-Mode Self-Evolution for Audio-Video | 双模自我进化 |
| [arXiv:2604.21221](https://arxiv.org/abs/2604.21221) | 2026-04 | **Sparse Forcing**: Native Trainable Sparse Attention for AR Diffusion | 改 attention 层 |
| [arXiv:2603.15240](https://arxiv.org/abs/2603.15240)（搜索结果中 AvatarForcing 类） | 2026-03 | **AvatarForcing**: One-Step Streaming Talking Avatars | 解决 AR forcing exposure bias |
| [arXiv:2603.12655](https://arxiv.org/abs/2603.12655)（同族） | 2026-03 | OmniForcing: Real-time Joint Audio-Visual | bidirectional → AR distillation |
| [arXiv:2604.20902](https://arxiv.org/abs/2604.20902) | 2026-04 | **Frequency-Forcing**: Soft Frequency Guidance | flow-matching 频率序 |
| [arXiv:2512.19311](https://arxiv.org/abs/2512.19311) | 2025-12 | **MixFlow Training**: Alleviating Exposure Bias with Slowed Interpolation Mixture | **最近邻 idea**：在 RAE 模型上专门解决 exposure bias，用 slowed interpolation mixture 后训练；ImageNet 256 FID 1.43，被报告为 RAE 上的 SOTA |
| [arXiv:2604.01545](https://arxiv.org/abs/2604.01545) | 2026-04 | **RAE-AR**: Taming Autoregressive Models with Representation Autoencoders | **几乎是我们的孪生**：把 RAE（DINO/SigLIP/MAE latent）接到 AR 模型，明确指出"high-dim 放大 exposure bias"，方案是**训练时 Gaussian noise injection** |

**Idea 1（CCT-224）的处境**：

* 概念上**不再具有 standalone 新颖性**——2025–2026 已有 MixFlow Training（slowed interpolation）+ RAE-AR（noise injection）+ 整个 Forcing 家族在做"消除训练-推理分布偏移"。
* **CCT 路线必须以"PET 多剂量物理结构"为差异化锚点**，不能再用"我们提出消除 exposure bias 的 consistency 训练"作为主 claim。
* 推荐叙述位移：**不是"提出 CCT"，而是"在物理可解释的多剂量 trajectory 上设计反事实一致性约束"**——claim 是 *constraint design*，不是 *consistency idea*。

## 13.4 自适应损失加权（Idea 2 的同期工作）

| ID | 时间 | 标题 | 与 ΔB-aware Reweighting 的关系 |
|----|------|------|-------------------------------|
| [arXiv:2604.15938](https://arxiv.org/abs/2604.15938) | 2026-04 | **VADF**: Vision-Adaptive Diffusion Policy Framework | "硬负样本不均衡 + 缺乏样本难度感知 → 收敛慢"；diffusion policy 的自适应难度采样 |
| [arXiv:2506.16688](https://arxiv.org/abs/2506.16688) | 2025-06 | **Variational Adaptive Weighting** for Diffusion Planning | 离线 RL diffusion 的变分自适应加权 |
| [arXiv:2508.13452](https://arxiv.org/abs/2508.13452) | 2025-08 | **Adaptive Loss Balancing for Hierarchical Multi-Label Classification** | 层次多任务的自适应平衡 |
| [arXiv:2507.02687](https://arxiv.org/abs/2507.02687) | 2025-07 | **APT**: Adaptive Personalized Training for Diffusion | 限定数据下的自适应训练 |

**Idea 2（ΔB-aware Adaptive Reweighting）的处境**：

* "Adaptive loss reweighting" 是 ML 通用方法；2025–2026 有若干同领域近邻，但**没有任何一篇把 A/B/C/D 误差预算诊断闭环到训练**——这是 Engineering Skeptic 在 IDEA_REPORT 中已指出的"诊断-控制闭环"独特角度。
* 仍然是 PASS Phase-2，但叙述要避免落入"我们提出 X 项加权"的 trick 陷阱，应该写成"**用因果误差预算驱动的训练控制器**"。

## 13.5 RAE 在 2026 上半年的爆发（验证我们的 RAE 选择）

| ID | 时间 | 标题 | 启示 |
|----|------|------|------|
| [arXiv:2602.08620](https://arxiv.org/abs/2602.08620) | 2026-02 | Improving Reconstruction of Representation Autoencoder | RAE 解码质量提升 |
| [arXiv:2601.16208](https://arxiv.org/abs/2601.16208) | 2026-01 | Scaling T2I Diffusion Transformers with RAE | RAE → T2I 大模型 |
| [arXiv:2603.16099](https://arxiv.org/abs/2603.16099) | 2026-03 | OneWorld: 3D Unified Representation Autoencoder | RAE → 3D scene |
| [arXiv:2603.09241](https://arxiv.org/abs/2603.09241) | 2026-03 | RAE-NWM: Navigation World Model in RAE space | RAE → world model |
| [arXiv:2511.23386](https://arxiv.org/abs/2511.23386) | 2025-11 | VQRAE: Quantization Autoencoders | RAE 的离散化变体 |

**结论**：RAE（用预训练表征编码器代替 VAE 做扩散/AR 的 latent space）在 2025-Q4 → 2026-Q1 出现**集中爆发**。我们的 DINOv2 + 冻结 ViT-MAE 解码即是该范式的早期实例，但**目前没有任何 RAE 论文落到 PET / 多剂量医学**。这是清晰的 unique selling point。

## 13.6 DINOv2 在医学影像（基础模型路线的同期）

| ID | 时间 | 启示 |
|----|------|------|
| [arXiv:2604.22557](https://arxiv.org/abs/2604.22557) | 2026-04: 自然域基础模型用于 cardiac MRI 重建 |
| [arXiv:2603.14086](https://arxiv.org/abs/2603.14086) | 2026-03: 3D 医学配准的 domain-specialized DINO 预训练 |
| [arXiv:2512.13608](https://arxiv.org/abs/2512.13608) | 2025-12: DBT-DINO 乳腺 tomosynthesis 基础模型 |
| [arXiv:2512.09307](https://arxiv.org/abs/2512.09307) | 2025-12: SAM → DINOv2 distillation 用于 polyp seg |

**结论**：自然域 DINOv2 直接迁移到医学已被广泛尝试；**但用 DINOv2 做 latent 空间生成/去噪几乎没人做**——主流仍在 segmentation/registration/classification。我们的"DINOv2 latent 上做 transport"是稀缺角度。

---

## 13.7 重估 Idea 1（CCT-224）与 Idea 2（ΔB-aware Reweighting）

**2026-05-02 状态**：原 IDEA_REPORT 2026-04-09 给的 Phase-1（CCT-224）/ Phase-2（ΔB-aware）实施序列**已被主线优先级取代**——见 §18-§19 的 emergent Grönwall 命中发现 + §20.6 行动清单。

简要更新（保留供未来参考）：

* **Idea 1 CCT-224**：解决的问题（exposure bias）真实存在，但同期 MixFlow / RAE-AR / Forcing 家族已大量覆盖，**纯 consistency loss 新颖度已 ≤ 6/10**；如果实施，叙述需重写为 "physics-grounded counterfactual anchoring on multi-dose PET trajectories"，并必配 RAE-AR noise + MixFlow slowed-interp 双 baseline。代码在分支 `idea1_cct_test`，**未合 master**。
* **Idea 2 ΔB-aware**：与 VADF 等"样本难度"加权差异化在于"误差预算 A/B/C/D 闭环"。叙述需重写为 "diagnostic-driven training control loop"。**未实施**。

## 13.9 2024+ 文献启发的新候选 idea（在 IDEA_REPORT 之外）

按可行性排序：

### Idea A：Trajectory Anchor Loss（受 MAP-Diff 启发，但落到 latent）

> **完整内容见 §14**（与 MAP-Diff 的 12 维对比 + Tier-1/2/3 改造路线）。本节仅作为索引：核心结论是 V6 现有 `compute_rollout_losses` 本质 ≈ MAP-Diff `L_anch`，**只缺 timestep-weighted 项 $w(t)$**——这个缺口在 §16-§19 已通过 multi-hop Grönwall 闭式解填补，无需新增 anchor loss。

### Idea B：Residual Noise Learning（受 2604.16925 启发）

* **核心**：把 v 预测改写成"预测 latent 的噪声残差"——形式上等价但训练目标统计性质不同；2604.16925 报告这能避免"averaging effect"。
* **与现有实现的关系**：我们已经是 v 预测，velocity_rebalance 已经在做类似归一化；但显式残差噪声目标可以另设单独 loss head。
* **风险**：与 V6 现有 v 预测目标可能冲突，需要谨慎。
* **评分**：**可行性 6/10、新颖性 5/10**（已有论文）、**与 V6 兼容性 6/10**——优先级低于 Idea 1/2。

### Idea C：Cross-Tracer / Cross-Scanner 验证（受 MAP-Diff & MDAA-Diff 启发）

* **核心**：MAP-Diff 用了内部（Siemens Quadra）+ 外部（United Imaging uEXPLORER）双扫描仪验证；MDAA-Diff 用了 18F-FDG + 68Ga-FAPI 双示踪剂。我们目前只用单数据源——审稿人会质疑泛化性。
* **建议**：**这不是新方法 idea，而是必须补的实验**；MICCAI/TMI 投稿前必做的外部验证。
* **优先级**：**HIGH**——这是真审稿门槛。

### Idea D：Slowed Interpolation Mixture（直接 port MixFlow Training）

* **核心**：训练时把输入 latent 微调向 model 自身生成的方向，模拟推理时分布；代价低（无第二次 forward）。
* **与 CCT-224 关系**：是更便宜的 exposure bias 缓解；**应作为 Idea 1 的 baseline 而不是替代**。
* **评分**：**可行性 9/10、与 V6 兼容性 9/10、独立新颖性 3/10**（同期工作太多）。
* **建议**：**作为 Idea 1 实验的对照组之一，不单独成线**。

### Idea E：Gaussian Noise Injection（直接 port RAE-AR）

* **核心**：训练时给 z_src 加小噪声，最低成本缓解 exposure bias。
* **评分**：**可行性 10/10、独立新颖性 1/10**。
* **建议**：**作为 Idea 1 实验的最便宜对照基线**。

### Idea F：Frequency-Conditioned Velocity Head（受 Frequency-Forcing 启发）

* **核心**：把 hop_idx 在残差速度头里替换成 frequency-band 编码——不同频段（低频结构 vs 高频细节）由不同子头处理。
* **与现有实现关系**：和 HopResidualVelocityHead 同构，但 conditioning 信号从 hop_idx 改为频段。
* **评分**：**可行性 7/10、新颖性 7/10、与 V6 兼容性 5/10**（需要重训残差头）。
* **建议**：放入 backlog，V6.X 收敛后考虑。

---

## 13.10 综合行动建议

> **2026-05-02 更新**：本节早期版本（2026-05-01 给出的 P0/P1/P2 排序）已被 §20.6 的最终行动清单取代。请直接参考 §20.6（结合 §19.6）作为权威清单。

### 文献 claim 必须正面应对的 5 篇

撰写论文 Related Work 时，**这 5 篇必须显式比较 + 差异化**：

1. **MAP-Diff (2603.02012)** — 最直接竞品（progressive + anchor + multi-dose）
2. **MDAA-Diff (2505.05112)** — Multi-Dose Adaptive Attention（架构相似）
3. **Cross-Dose PET / Residual Noise (2604.16925)** — 直接命中跨剂量泛化痛点
4. **MixFlow Training (2512.19311)** — 同时期的 exposure bias 解法
5. **RAE-AR (2604.01545)** — RAE + AR exposure bias 孪生工作

---


# 14. Trajectory Anchor Loss 深度对照（MAP-Diff vs 当前实现）

> **检索依据**：MAP-Diff 论文（[arXiv:2603.02012v1](https://arxiv.org/abs/2603.02012)，2026-03-02）HTML 版完整方法节、训练目标公式、ablation Table 2；我们的实现 `pet_lr/rollout_first_hop.py:48-115`、`train_first_hop.py:289-460`、`pet_lr/model_first_hop.py` 中 `predict_latent_step`。

## 14.1 MAP-Diff 的 anchor loss 数学定义

MAP-Diff 在标准 DDPM 噪声预测之上叠加了一项 **anchor reconstruction loss**：

$$
\mathcal{L}_\text{anch} = \mathbb{E}_t \left[ w(t) \, \big\lVert \hat{x}_0(x_t, t \mid y) - a(t) \big\rVert_2^2 \right]
$$

其中：

* **anchor 集合** $\mathcal{A} = \{a_1, a_2, a_3, a_4\}$：4 张临床中间剂量图（1/10、1/4、1/2、full），全部来自真实采集
* **piecewise anchor 映射** $a(t) = \sum_{j=1}^4 \mathbf{1}[t \in \mathcal{I}_j] \cdot a_j$：
  * $\mathcal{I}_1 = [\tau_1, T]$（高噪声段 → 1/10 dose anchor）
  * $\mathcal{I}_2 = [\tau_2, \tau_1)$（中噪 → 1/4 dose）
  * $\mathcal{I}_3 = [\tau_3, \tau_2)$ → 1/2 dose
  * $\mathcal{I}_4 = [0, \tau_3)$ → full dose
* **boundary 校准** $\tau_j^\star = \arg\min_t \lVert d^\text{sim}(t) - d_j^\text{clin} \rVert_2^2$，其中 degradation signature $d = (\text{NMAE}, 1-\text{SSIM})$ 通过 body mask 计算
* **timestep 加权** $w(t) = (1 - t/T)^p$（多项式，对低噪声晚期阶段加权更大）
* **总损失** $\mathcal{L} = \mathcal{L}_\text{noise} + \lambda \mathcal{L}_\text{anch}$

**Ablation 关键发现（Table 2，PSNR）**：

| 设置 | anchors | timestep weight | PSNR |
|------|---------|----------------|------|
| 3D DDPM baseline | 无 | — | 42.48 |
| S1（推荐） | {1/10, 1/4, 1/2} | poly $(1-t/T)^p$ | **43.71** |
| S5 | {1/10} 单 anchor | poly | 43.50 |
| S7 | {1/2} 单 anchor | poly | 42.79 |
| **S8** | {1/10, 1/4, 1/2} | **constant** | **41.86** ⚠️ |

**核心结论**：

1. anchor 越早（更靠近高噪声段）越关键——单 anchor 时 {1/10} > {1/4} > {1/2}
2. **去掉 timestep 权重** $w(t)$ 是灾难性的（−1.85 dB，比 baseline 还差！）——说明在 MAP-Diff 框架里，"anchor 在哪个 timestep 监督"和"anchor 是什么"同等重要

## 14.2 我们当前实现的损失结构（精确口径）

### 14.2.1 Pair loss（`train_first_hop.py:compute_pair_losses` L289-358）

对每对 $(z_\text{src}^{(k)}, z_\text{dst}^{(k)})$，$k \in \{0,1,2,3\}$（D50→D20, D20→D10, D10→D4, D4→NORMAL），同时监督 velocity 与 endpoint：

$$
\mathcal{L}_\text{pair}^{(k)} = w_v^\text{eff} \cdot \big\lVert v_\theta(z_\text{src}, t_\text{src}, t_\text{dst}, k) - v^*_k \big\rVert_2^2 + w_e \cdot \big\lVert z_\text{src} + \Delta t \cdot v_\theta - z_\text{dst} \big\rVert_2^2
$$

其中：

* $v^*_k = (z_\text{dst} - z_\text{src}) / \Delta t$（target_normalize 时再除以 $\sigma_k$）
* $w_v^\text{eff} = w_v \cdot \text{rebalance\_scale}$，$\text{rebalance\_scale} = \mathrm{clip}(\sqrt{\mathcal{L}_e / \mathcal{L}_v}, 1, 10)$（动态把 v 项拉齐到 endpoint 量级）
* `pair_loss_weights` 在 V6 = $[2.5, 1.0, 1.0, 1.0]$，按 hop 加权（强化 D50→D20）

### 14.2.2 Rollout loss（`rollout_first_hop.py:rollout_multistep_losses_first_hop` L48-115）

链式从 $z_\text{D50}$ 出发，逐 hop 预测下一个 latent，每个 hop **直接对真实中间剂量 latent** 监督：

$$
\mathcal{L}_\text{roll}^{(k)} = \big\lVert \hat{z}_k - z^*_k \big\rVert_2^2, \quad k = 1,2,3,4
$$

其中 $\hat{z}_k = \mathrm{model}(\hat{z}_{k-1}^\text{mix}, t_{k-1}, t_k, \text{hop}=k-1)$ 是链式预测，$z^*_k$ 是该剂量的真实 latent。

* $\hat{z}_{k-1}^\text{mix} = \mathrm{mix\_latent}(z^*_{k-1}, \hat{z}_{k-1}, \alpha_\text{mix})$：teacher-forcing 与 free rollout 的可调比例（$\alpha=0$ 全 teacher，$\alpha=1$ 全 free）
* straight-through gradient：forward 用 mix 值，backward 只通过 $\hat{z}$ 路径（$z^*$ 不出 grad）
* $\alpha_\text{mix}$ 线性 ramp（V6: warmup 后 0→1，与 $\lambda_\text{roll}$ 同步）
* `step_weights` 在 V6 = $[0.5, 2.0, 1.5, 1.0]$，对 hop1/hop2 加权
* 总：$\mathcal{L}_\text{roll} = \sum_k w_k^\text{step} \mathcal{L}_\text{roll}^{(k)} / \sum_k w_k^\text{step}$

### 14.2.3 总损失

$$
\mathcal{L} = \lambda_p \mathcal{L}_\text{pair} + \lambda_r(t) \mathcal{L}_\text{roll} + \lambda_i \mathcal{L}_\text{img}^\text{hop0} + \lambda_a(t) \mathcal{L}_\text{align}
$$

V6 配置：$\lambda_p=15$、$\lambda_r$ 三段（0/0→4/4）、$\lambda_i=0.04$、$\lambda_a$ ramp 0→1。

## 14.3 逐项对照表（MAP-Diff anchor loss vs 我们的 rollout loss）

| 对比维度 | MAP-Diff (`L_anch`) | 我们 (`L_roll` + `L_pair`) | 谁更优 / 等价 |
|---------|---------------------|---------------------------|--------------|
| **Anchor 定义** | 4 张真实采集的剂量图作为 anchor | 4 个真实采集剂量的 **latent** 作为 anchor | **概念等价**——我们在 latent 空间，他们在像素空间 |
| **Anchor 数量** | 3 个中间 + 1 个终点（{1/10, 1/4, 1/2, full}） | 4 个目标（z_D20, z_D10, z_D4, z_NORMAL） | **等价**（我们略多 1 个，因起点 D50 是输入而非 anchor） |
| **轴的语义** | DDPM **连续噪声轴** $t \in [0, T]$；anchor 通过 **degradation matching 校准**到 $\tau_j$ 离散点 | **离散物理 hop 轴** $k \in \{0,1,2,3\}$；hop 与剂量 **一一对应**（PET 物理给定） | **我们更优**——零校准开销，无 degradation matching 误差，物理可解释 |
| **Anchor 在哪些 timestep 起作用** | 整个 timestep 区间 $t \in \mathcal{I}_j$（softmax-style piecewise） | 仅在该 hop 的预测点 $\hat{z}_k$ 上 | **MAP-Diff 更密**——他们在每个 t 都有 anchor 监督；我们只在 4 个离散点 |
| **Timestep 权重 $w(t)$** | 显式 $w(t) = (1-t/T)^p$，poly 衰减 | 显式 step_weights = [0.5, 2.0, 1.5, 1.0]，**固定常数** | **MAP-Diff 更细**——他们的 w(t) 是 timestep 函数；我们是 hop 常数 |
| **训练-推理一致性** | DDPM 训练加噪、推理反向；anchor 只是**正则**，不改变推理路径 | rollout 训练 = sample_chain 推理（结构同构）；teacher forcing 通过 alpha_mix **显式调度** | **我们更优**——结构上无 train-test gap |
| **推理代价** | $T = 1000$ DDPM 步 | 4 hop × 1 step = 4 forward passes | **我们快约 250 倍** |
| **中间剂量产物** | 推理时记录 $\hat{x}_0(x_t,t)$ 在 $t = \tau_1, \tau_2, \tau_3$ → 中间剂量合成 | 直接是 `sample_chain_first_hop` 的逐 hop 输出，每个解码即得 | **我们更原生**——中间剂量是 first-class output |
| **额外组件**（pair loss / vel rebalance） | 无——只有 noise + anchor 两项 | 有：pair loss（vel+end + sqrt rebalance）+ pair_loss_weights | **我们更复杂**，但 V3/V6 实验证明 vel rebalance 防止 endpoint 主导是必要的 |
| **Anchor 监督形式** | $\hat{x}_0$（每个 t 都从 noisy $x_t$ 估计 clean）和 $a(t)$ 比 | $\hat{z}_k$（链式累积预测）和 $z^*_k$ 比 | **形式不同**——MAP-Diff 是单步反演，我们是链式累积 |
| **Exposure bias 处理** | 无显式处理；anchor 一致性间接缓解 | `alpha_mix` 显式调度 + straight-through | **我们更显式** |
| **报告增益** | DDPM baseline 42.48 → 43.71 dB（+1.23 dB） | V3 best 0.000693 → V6 best 0.000661（mse，对应 PSNR ~+0.2 dB） | 任务量级不同（3D 全身 vs 2D 224 切片），无法直接比 |

## 14.4 关键差异的本质（一句话总结）

> **MAP-Diff 解决"DDPM 噪声轴本来与剂量轴无关"的问题——他们把 anchor 通过 degradation matching 校准到噪声轴上，然后用 timestep weighting 控制监督密度。**
>
> **我们一开始就让 hop 轴 = 剂量轴（PET 物理 t_map），rollout loss 在每个 hop 直接对应 anchor，无需校准；代价是只在 4 个离散点监督，不是连续 timestep 区间。**

## 14.5 优势对照表（论文卖点角度）

| 我们相对 MAP-Diff 的优势 | 我们相对 MAP-Diff 的劣势 |
|------------------------|------------------------|
| ✅ 零 boundary calibration 开销，t_map 物理给定 | ❌ 监督密度更稀（4 离散点 vs 连续 timestep） |
| ✅ 训练-推理结构一致，无 distributional shift（除 alpha_mix 显式控制部分） | ❌ 没有 timestep-weighted 加权——固定 step_weights 在 V6 已有但**不是 timestep 函数** |
| ✅ 推理快约 250×（4 step vs 1000 step） | ❌ 不是 score model，不直接享受 DDPM 的统计建模充分性 |
| ✅ 中间剂量输出是 native（每个 hop 都可解码） | ❌ 需要额外 pair loss + velocity rebalance + image_aux 等子损失，超参面更大 |
| ✅ Mean Flow + latent 空间 → 端到端误差预算 96-99% 可压缩 | ❌ 缺 cross-tracer / cross-scanner 验证（MAP-Diff 已做） |
| ✅ alpha_mix 显式调度 exposure bias | ❌ 没有像 MAP-Diff S1 vs S8 那样的 step_weights ablation 数据 |

## 14.6 可借鉴的具体改造点（Idea A 升级路线）

如果要做"Trajectory Anchor Loss for latent multi-dose PET"作为单独贡献，**最小可行改动**是：

### Tier-1（零代码改动，只是 ablation 实验）

1. **重做 step_weights ablation**：参照 MAP-Diff Table 2，对照 V6 的 `[0.5, 2.0, 1.5, 1.0]` vs `[1, 1, 1, 1]`（constant，对应 S8） vs `[0.25, 0.5, 1.0, 2.0]`（对低噪声/晚期 hop 加权，模拟 $w(t)=(1-t/T)^p$） vs 其他 schedule。
   * 假说：MAP-Diff 在像素 DDPM 上去掉 timestep 权重 −1.85 dB；我们 latent Mean Flow 更结构化，影响应较小但仍可观测。
   * 这一组 ablation **不动代码**，只改 yaml；强烈推荐立即执行。

2. **重做 anchor coverage ablation**（Tier-1.5）：对照"全 4 anchor（V6 当前）" vs "只 D20 anchor + D4 + NORMAL"（隐去 D10）vs "只 NORMAL"——证明每个 anchor 的边际贡献。这需要一行配置修改（把对应 step_weight 设 0）。

### Tier-2（小改动）

3. **timestep-weighted step_weights**：把当前的固定 `step_weights[k]` 改成关于 hop 进度的多项式：

   ```python
   # 候选：模拟 MAP-Diff w(t) = (1 - t/T)^p，但用 hop-based progress
   # progress[k] = (t_k - t_min) / (t_max - t_min) ∈ [0,1]
   step_weight[k] = (1 - progress[k] * decay)**p   # decay∈[0,1], p∈[0.5, 2]
   ```

   * 实现成本：`rollout_first_hop.py:60-72` 里 `step_weights` 改成可计算的函数即可。
   * 当前主分支 V6 的 `[0.5, 2.0, 1.5, 1.0]` 是经验值，没有理论支撑；多项式形式可以做 sweep。

### Tier-3（方法创新）

4. **Boundary calibration ablation 反向证伪**：MAP-Diff 必须做 degradation matching；我们用 PET 物理 t_map。可以做一个**反向实验**：
   * 用 `degradation matching`（NMAE+SSIM 在 latent 空间）拟合 4 个最佳 t_map 值
   * 对照物理 t_map = [2, 5, 10, 25, 100]
   * **如果物理 t_map 显著优于 fitted t_map**，这就是 PET 多剂量去噪的"物理优于经验"硬证据，足以独立成 paper section。
   * 风险：fitted 可能略好（因为是端到端优化）；但只要差距小，物理 t_map 仍因可解释性获胜。

## 14.7 与 §13.9 Idea A 的最终衔接

**§13.9 Idea A 的"评分 9/10/6/9"维持不变**——它仍是与 V6 兼容性最高、可作为 Idea 1 附属 ablation 的改造路径。但论文叙述应该明确：

> 我们的 **rollout loss 本质上已经是 trajectory anchor loss**——不需要新增 anchor 项，只需要在论文里**显式命名**为"Latent Trajectory Anchor Loss"，并对照 MAP-Diff 的 timestep-weighted anchor。
> 真正的方法论贡献是**"physical t_map 取代 degradation matching"**——这一点用 §14.6 Tier-3 的反向 ablation 可以做出独立 claim。
> 实施优先级：Tier-1（零代码 ablation）必做，Tier-2（参数化 step_weights）可在 V6 收敛后做单变量替换，Tier-3（boundary calibration 反证）作为论文 ablation 章节的硬证据。

## 14.8 一行结论

> **我们的 `compute_rollout_losses` ≈ MAP-Diff `L_anch`，但更干净（物理 t_map）、更快（4 vs 1000 step）、训推一致（结构同构），代价是监督只在 4 个离散点（vs 连续 timestep）。MAP-Diff 唯一显著优于我们的设计是 timestep-weighted 权重 $w(t)=(1-t/T)^p$——这是我们应当 port 进 step_weights 的具体可借鉴点。**

---

# 15. Timestep 权重 $w(t)$ 是否值得借鉴——深度分析

> **背景**：§14.3 的对照表里有一行结论说"MAP-Diff 唯一明显胜出我们的设计是 $w(t)=(1-t/T)^p$"。本节专门回答："**这条经验能否、应当如何、以多大代价 port 到我们的框架？**"

## 15.1 关键前提：t 轴语义在两个框架里**方向相反**

这是一个**容易被忽略但会决定加权方向**的差异。

### MAP-Diff 的 t

```
t ∈ [0, T=1000]
t = 0     ←  clean image x_0          ← 目标分布
t = T     ←  pure Gaussian noise      ← 起点
w(t) = (1 - t/T)^p   →  小 t（clean 端）权重大
```

MAP-Diff 的语义：**生成结束附近（小 t、低噪声）权重最大**——这是反向去噪的"最后一公里"，视觉质量最敏感。

### 我们的 t（来自 V6 yaml `t_map`）

```
D50:  t = 2.0        ←  noisy 起点（输入 latent）
D20:  t = 5.0
D10:  t = 10.0
D4:   t = 25.0
NORMAL: t = 100.0    ←  clean 目标 latent
```

我们的语义：**t 越大越接近 clean target**（与 MAP-Diff 完全相反）。这是 Mean Flow / Flow Matching 的标准约定（t∈[0,1]，1 是 target），论文里 t_map 用了非归一化的物理时间（exposure time 比例），但**方向是 t↑ 表示越干净**。

### 直接结论

**不能** naively 套 $w(t)=(1-t/T)^p$ 公式——会把权重压在我们的 D50（noisy 起点）那一端，与 MAP-Diff 的设计意图（"最后一公里加权"）完全相反。

正确的方向映射是：

$$
w_\text{ours}(t) = \left( \frac{t - t_\min}{t_\max - t_\min} \right)^p \quad \text{（"距 clean 越近权重越大"，与 MAP-Diff 同义）}
$$

或者用 hop_idx 表达：

$$
w_\text{ours}(k) = \left( \frac{k}{K-1} \right)^p, \quad k \in \{0, 1, \dots, K-1\}, \ K = 4
$$

如果想保留"早期重"作为对照（这是 V6 的实际做法），就用：

$$
w_\text{ours}^\text{early}(k) = \left( 1 - \frac{k}{K-1} \right)^p
$$

## 15.2 $w(\cdot)$ 在我们框架里**应该作用于哪个对象**？四个候选

我们的损失结构有 4 个权重出口，每个都可以挂 $w(\cdot)$，但语义不同：

| 候选 | 作用对象 | 当前 V6 值 | 单调性 | 对应 MAP-Diff 类比 |
|------|---------|-----------|--------|-------------------|
| **A** | `step_weights[k]`（rollout chain loss） | `[0.5, 2.0, 1.5, 1.0]` | 非单调（hop1 峰） | 最直接——MAP-Diff 在 anchor 监督上加 $w(t)$ 等价于这里 |
| **B** | `pair_loss_weights[k]`（pair v+e loss） | `[2.5, 1.0, 1.0, 1.0]` | 单调递减 | 弱类比——MAP-Diff 没有 pair-style loss |
| **C** | pair 内 vel/end 子项权重 | velocity_rebalance sqrt(L_e/L_v) clip(1,10) | 动态 | 不直接类比 |
| **D** | 连续 t 上加权（采样 t∈[t_min,t_max] 而不是固定 4 个 hop） | 不存在 | — | 最 faithful 但**需要重大代码改动**和数据流改造 |

**讨论**：

* **候选 D（连续 t）** 是 MAP-Diff 真正的等价物，但它需要：(1) 数据集生成 4 个真实剂量之外的"interpolated latent"作为 anchor target；(2) 在采样的 t 处对 anchor zone 做 piecewise 选择；(3) 引入 boundary calibration $\tau_j$。这等于**重做 MAP-Diff 的整套 anchor 机制**——背离了我们"hop = 物理剂量"的简洁优势。**不建议**。
* **候选 A（step_weights）** 是 V6 已有变量，只需改值或参数化函数，**零代码 / 一行函数改动即可 ablation**。**首选**。
* **候选 B（pair_loss_weights）** 与 A 性质类似，但语义不同：pair 是单步监督（无 compound error），其权重逻辑应该跟"哪一跳本身最难"挂钩，而不是"哪个 t 最重要"。**作为对照**而非主路线。
* **候选 C** 是动态 rebalance，已经有自洽机制，不应被静态 $w(t)$ 干扰。**不动**。

> **结论**：对 `step_weights[k]`（候选 A）做 timestep-style 加权是唯一**值得 port、且代价最小**的方向。

## 15.3 三种 schedule 候选的分析论据

V6 当前 `step_weights = [0.5, 2.0, 1.5, 1.0]` 不是单调，是经验调出来的"hop1 峰"。下面比较三种参数化候选：

### 候选 S-Inc：递增（"MAP-Diff 同向"）

$$
w_k^\text{inc}(p) = \left( \frac{k+1}{K} \right)^p, \quad p \in \{0.5, 1.0, 2.0\}
$$

| p | step_weights（K=4） | 解读 |
|---|---------------------|------|
| 0.5 | [0.50, 0.71, 0.87, 1.00] | 弱递增 |
| 1.0 | [0.25, 0.50, 0.75, 1.00] | 线性递增 |
| 2.0 | [0.0625, 0.25, 0.5625, 1.00] | 强递增（重押 hop3） |

**论据 PRO**：MAP-Diff Table 2 显示去掉 timestep 权重 −1.85 dB（S1→S8），最后阶段最重要，符合"视觉质量最敏感于最后修正"的直觉。

**论据 CON**：在我们的 chain rollout 里，**hop3 的 z_pred 是 hop0/1/2 误差累积的产物**——hop3 loss 大不一定因为 hop3 难，而是上游误差残留。强加权 hop3 可能让模型只优化 final refinement，**牺牲上游精度**——这与我们 V6 的诊断（"D50→D20 才是 dominant error source"，pair_loss_weights[0]=2.5）冲突。

### 候选 S-Dec：递减（"V6 同向"）

$$
w_k^\text{dec}(p) = \left( 1 - \frac{k}{K} \right)^p, \quad p \in \{0.5, 1.0, 2.0\}
$$

| p | step_weights（K=4） | 解读 |
|---|---------------------|------|
| 0.5 | [1.00, 0.87, 0.71, 0.50] | 弱递减 |
| 1.0 | [1.00, 0.75, 0.50, 0.25] | 线性递减 |
| 2.0 | [1.00, 0.5625, 0.25, 0.0625] | 强递减 |

**论据 PRO**：(1) 误差累积论据——早期 hop 错了下游全错；(2) latent magnitude 跳跃 D50→D20 最大（V3→V6 lessons 已证）；(3) 与 pair_loss_weights[0]=2.5 同向，加强一致性。

**论据 CON**：和 MAP-Diff 经验**正面冲突**——MAP-Diff 在 PET 像素 DDPM 上证明"最后阶段权重大"是正解。但**机制不同**：MAP-Diff 的 t=0 是真正的 clean target，整个 reverse process 集中信息于此；我们 hop3 是 chain 的末端，不是单点输出。

### 候选 S-Const：常数（"MAP-Diff S8"对照）

$$
w_k^\text{const} = 1, \quad \forall k
$$

**用途**：作为 ablation 对照。MAP-Diff 报告 −1.85 dB；我们预期影响**显著但更小**（latent + Mean Flow 比 pixel + DDPM 更结构化）。

### 候选 S-V6：V6 现状（"非单调中峰"）

$$
w_k^\text{v6} = [0.5, 2.0, 1.5, 1.0]
$$

**特点**：hop1 峰 → hop3 衰减。可解读为"补偿 + middle-heavy"：hop0 已被 pair_loss_weights[0]=2.5 强权，所以 step_weights 故意压低 hop0 防过权；hop1/hop2 是 chain 中段误差敏感区；hop3 端点权重一般。

**论据**：这是 V3→V6 经验调出来的**实证最优**，但缺乏理论支撑和参数化形式——所以无法系统 sweep。

## 15.4 与 V6 现有 step_weights / pair_loss_weights 的关系（重要！）

### 15.4.1 双权重耦合现象

我们有**两套** hop 维度权重在叠加：

$$
\text{有效hop权重}(k) \approx \underbrace{\lambda_p \cdot w_k^\text{pair}}_{\text{pair 通道}} + \underbrace{\lambda_r \cdot w_k^\text{step}}_{\text{rollout 通道}}
$$

V6 的设计：**pair 通道头重（hop0 主导），rollout 通道中重（hop1/2 主导），合起来覆盖整条 chain**。如果只动 step_weights 而不动 pair_loss_weights，可能让"中段"被双重弱化或强化，不是单变量 ablation。

**严格的 ablation 需要**：固定 `pair_loss_weights = [1, 1, 1, 1]` 或 `pair_loss_weights = pair_loss_weights_v6`，**只对 step_weights 做 sweep**。

### 15.4.2 target_normalize 的二阶交互

V6 启用 `target_normalize=true` ⇒ `v_target = (z_dst - z_src) / dt / sigma_for_hop`。这一步**已经按 hop 做了量级归一化**——pair 损失里不同 hop 的 v_err 已经在同一量级。

但 **rollout step_loss 没有 target_normalize**——它直接是 `MSE(z_pred, z_gt)`。不同 hop 的 z 分布范围不同（D50 vs NORMAL），所以原始 loss 量级**本身不一**。如果不先把 rollout step losses 归一化到同量级，加 schedule 的"权重含义"就不纯（被 base loss magnitude 主导）。

**纠正措施**（写代码前必做）：

* 先对 rollout step_losses 做"几何归一"——比如除以 EMA 的 step_loss 量级
* 或者对每个 hop 的 z_gt 做 sigma 归一化后再算 MSE

否则，"换 schedule" 的实验信号会被原始量级差异淹没。

### 15.4.3 lambda_roll ramp 的相互作用

V6 用 `lambda_roll` 三段 ramp（0/0→4/4），rollout loss 的总贡献本身在动态变化。任何 step_weights schedule sweep 都应该**在同一 lambda_roll 阶段做对照**，否则结果是 (schedule × lambda_roll 阶段) 的混合效应。

## 15.5 后续展开

> **2026-05-02 更新**：本节早期版本（§15.5 风险与陷阱、§15.6 8-cell ablation 矩阵 E0-E7、§15.7 代码改动方案、§15.8 决策树、§15.9 结论）**已被 §16-§19 重写**——具体地：
> - "盲扫 8 组 ablation"被 §20.6（优先级行动清单：捷径 1+2 → 3–4 组带理论支撑的实验）取代
> - 进一步被 §19.4（基于 Lipschitz 测量的闭式解 + 25% 偏差判定）取代
> - 风险清单的"量级不一致"项（最严重）被升级为 §20.4 #1（真正的论文级 ablation）
> - 决策树被 §20.6 + §19.6 的统一行动清单取代
>
> **本节早期讨论保留供历史追溯**——如需当前可执行决策，请直接看 §20.6。

---

# 16. Timestep 权重的数学根基——能否"用证明代替消融"

> **背景**：用户希望"借鉴已有发表物的公式"以减少消融工作量。本节系统检索 diffusion 加权理论中**有严格数学证明的公式**，逐项判断能否直接套用到我们的 latent Mean Flow + chain rollout 框架，并指出**真正能直接抄的"理论加权"是什么、不能抄的原因是什么、什么数学是真正适配我们的**。
>
> **TL;DR**：MAP-Diff 的 $w(t)=(1-t/T)^p$ **没有严格数学证明**，是经验启发。但 diffusion 文献里**有 3 个有严格证明的加权公式**（P2、Min-SNR-γ、EDM）；其中 **Min-SNR-γ 的精神**已经被我们的 `target_normalize=true` 部分实现。真正能"省消融"的捷径是：(1) 把 Min-SNR 推广到 rollout loss（一行代码），(2) 用 Grönwall 误差传播理论给出 step_weights 的解析形式。**这两条都能给出"理论值"作为 ablation 的中心点，把 sweep 半径从 [0, ∞) 缩到一个邻域**（完整推导见 §16.4 + §18.3.3，行动清单见 §20.6）。

## 16.1 MAP-Diff $w(t)=(1-t/T)^p$ 的论文论据审查

### 16.1.1 论文中实际的 justification

复读 MAP-Diff（[arXiv:2603.02012v1](https://arxiv.org/abs/2603.02012)）方法节，原文给出 $w(t)=(1-t/T)^p$ 的论据是：

> "We use a polynomial weight to emphasize the later (low-noise) timesteps, where the visual quality is most sensitive to anchor consistency."

* 没有定理、没有引理、没有 SNR 推导
* $p$ 是超参，作者通过 **ablation Table 2 网格搜索**给出推荐值（$p=2$ 附近）
* S1 vs S8 的 −1.85 dB **是经验观测，不是定理结论**

### 16.1.2 严格地讲

MAP-Diff 的 $w(t)=(1-t/T)^p$ **缺乏严格数学证明**，属于"启发式 + 实证"。直接 cite 这个公式作为我们 step_weights 的设计依据，论文 reviewer 会反问"为什么这个形式而不是 $\exp(-t)$、$\sin(\pi t/T)$、Beta 分布？"——而 MAP-Diff 自己也回答不了。

**结论**：不能拿 MAP-Diff 的公式当"已发表的数学证明"来用。

## 16.2 Diffusion 文献中**有**严格证明的加权公式（3 个）

下面这 3 个公式都有正经的数学推导，**可作为我们论文的引用基础**。

### 16.2.1 P2 加权（Choi et al., CVPR 2022）

**Paper**：*Perception Prioritized Training of Diffusion Models*

**公式**：

$$
w_\text{P2}(t) = \frac{1}{(k_\text{P2} + \mathrm{SNR}(t))^\gamma}
$$

其中 $\mathrm{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$ 是 DDPM 的信噪比。

**推导基础**：

* 把 DDPM denoising 过程分成"内容阶段"（高 SNR）+"感知阶段"（中 SNR）+"清理阶段"（低 SNR）
* 通过 Fisher information 论证感知阶段对最终质量贡献最大
* 解析推出权重应**反向衰减 SNR**——SNR 越大权重越小

**数学严谨度**：基于 score matching 的 ELBO 分解，是 ICLR/CVPR-grade 的推导。

### 16.2.2 Min-SNR-γ（Hang et al., ICCV 2023）

**Paper**：*Efficient Diffusion Training via Min-SNR Weighting Strategy*

**公式**：

$$
w_\text{minSNR}(t) = \min(\mathrm{SNR}(t), \gamma)
$$

**推导基础**：

* 把每个 t 视为一个独立的多任务学习子任务
* 相邻 t 的子任务**梯度方向部分冲突**（Pareto front 论证）
* 推导出"梯度量级均衡"的最优权重应该 clamp 极端 SNR
* 给出 Pareto-optimal 权重的闭式解

**数学严谨度**：基于多任务学习的 Pareto front 理论 + 闭式解，**比 P2 更严格**。

**关键性质**：

* 高 SNR 端（clean）：被 $\gamma$ clip，防止过权 → "防止干净端损失项主导训练"
* 低 SNR 端（noisy）：取 SNR 本身（小） → "降低纯噪声端权重"

### 16.2.3 EDM 预条件（Karras et al., NeurIPS 2022）

**Paper**：*Elucidating the Design Space of Diffusion-Based Generative Models*

**公式**（损失加权部分）：

$$
w_\text{EDM}(\sigma) = \frac{\sigma^2 + \sigma_d^2}{(\sigma \cdot \sigma_d)^2}
$$

其中 $\sigma$ 是噪声尺度，$\sigma_d$ 是数据标准差。

**推导基础**：

* 从"网络输出应该 norm-1"的预条件目标出发
* 通过 skip connection 形式化解出权重函数
* 同时给出 input/output/skip/noise 4 路 preconditioning

**数学严谨度**：解析推导，**是这 3 个里最严格的**——直接从期望损失最小化解出来。

## 16.3 这 3 个公式能否直接套用到我们框架？逐项判断

### 16.3.1 适配判定矩阵

| 公式 | 假设的 loss 形式 | 假设的 t 含义 | 我们的形式 | 我们的 t 含义 | **能直接套？** |
|------|----------------|--------------|----------|--------------|-------------|
| **P2** | DDPM noise prediction $\\|\epsilon - \epsilon_\theta\\|^2$ | $t \in [0, T]$ DDPM 加噪步 | Mean Flow velocity $\\|v_\theta - v^*\\|^2$ + endpoint MSE | 物理 t_map (2.0 → 100.0) | **❌ 不能直接套** |
| **Min-SNR-γ** | DDPM noise prediction | DDPM 加噪步 | Mean Flow velocity + endpoint | 物理 t_map | **⚠️ 精神可移植** |
| **EDM** | Score matching with $\sigma$ | 噪声尺度 $\sigma$ | Mean Flow velocity | 物理时间 | **⚠️ 类比可借鉴** |

### 16.3.2 为什么不能 naive 套？三个根本差异

1. **目标不同**：P2/Min-SNR/EDM 的推导**前提是预测 noise 或 score**——这两者的统计性质（高斯方差、ELBO 分解）撑起了整个推导。我们预测的是 **velocity field**（Mean Flow $v^*$）和 **endpoint $z$**，性质完全不同。

2. **t 不是噪声步**：上 3 个公式里的 t 都是 forward diffusion 的"加噪进度"，与 SNR 一一对应。我们的 t_map=(2,5,10,25,100) **是物理 exposure time 比例**，不对应任何 noise schedule，**SNR(t) 在我们的框架里没有定义**。

3. **链结构不同**：上 3 个推导都是**单步反演** loss（$\hat{x}_0$ 一步到位），**没有 chain rollout 的误差累积**。我们的 rollout loss 内部有 chain dynamics，纯 single-step 的加权理论不覆盖。

### 16.3.3 但 Min-SNR 的"精神"已经在我们 V6 里实现了！

读 [model_first_hop.py L437, L507](PET_LatentResidual/pet_lr/model_first_hop.py#L437)：

```python
def sigma_for_hop(self, hop_idx):
    return self.pair_v_std[hop_idx.long()].view(-1, 1, 1, 1)

# in predict_latent_step:
if self.target_normalize:
    sigma = self.sigma_for_hop(hop_idx)
    v_total = v_total_raw * sigma   # un-normalize
```

而 [train_first_hop.py L289-358 `compute_pair_losses`](PET_LatentResidual/train_first_hop.py#L289) 计算 $v^*_k = (z_\text{dst}-z_\text{src})/\Delta t / \sigma_k$（normalized target）。

**这一步等价于 Min-SNR-γ 的 "per-hop magnitude balancing"**：把不同 hop 的 velocity target 拉到统一量级（方差≈1），**相当于自适应给每个 hop 加权 $1/\sigma_k^2$**。

**即**：

> $\text{有效 pair loss}(k) = \mathrm{Var}\left[\frac{v^*_k}{\sigma_k}\right] \cdot \mathrm{Var}\left[v_\theta\right] = \text{const across } k$
>
> ⟺ 每个 hop 在 pair loss 中获得**相同的有效梯度量级**——这正是 Min-SNR-γ 的 Pareto-optimal 解的精神（量级均衡）。

**关键观察**：**pair loss 已经是 Min-SNR-style 加权了，但 rollout loss 没有**——`rollout_multistep_losses_first_hop` 直接对 raw $z_\text{pred}$ 和 raw $z_\text{gt}$ 算 MSE，没有 sigma 归一。

## 16.4 真正适配我们框架的数学：链式误差传播（Grönwall 引理）

由于我们的核心结构是 chain rollout 而不是 single-step diffusion，**正确的数学起点不是 SNR 加权理论，而是 ODE 数值求解的误差传播**。

### 16.4.1 模型作为离散动力系统

我们的 chain：

$$
z_{k+1} = z_k + \Delta t_k \cdot v_\theta(z_k, t_k, t_{k+1}, k) \equiv F_k(z_k)
$$

每个 hop 的真实 update 是 $z^*_{k+1} = F_k^*(z^*_k)$。设单步预测误差 $\epsilon_k = F_k(z_k) - F_k^*(z_k)$，累积误差 $\delta_k = z_k - z^*_k$，则有

$$
\delta_{k+1} = F_k(z_k) - F_k^*(z^*_k) = \underbrace{F_k(z_k) - F_k(z^*_k)}_{\text{Lipschitz term}} + \underbrace{F_k(z^*_k) - F_k^*(z^*_k)}_{\epsilon_k}
$$

应用 Lipschitz 假设 $\|F_k(a) - F_k(b)\| \leq L_k \|a - b\|$（其中 $L_k = 1 + \Delta t_k \cdot \mathrm{Lip}(v_\theta)$），得到**离散 Grönwall 不等式**：

$$
\boxed{\quad \|\delta_K\|^2 \leq \sum_{k=0}^{K-1} \left( \prod_{j=k+1}^{K-1} L_j^2 \right) \|\epsilon_k\|^2 \quad}
$$

### 16.4.2 推论：最优 step_weights 的解析形式

如果训练目标是最小化最终误差 $\mathbb{E}\|\delta_K\|^2$，且每个 $\epsilon_k$ 独立可优化，则**Cauchy-Schwarz 等号成立时**等价于损失加权

$$
\boxed{\quad w_k^\text{optimal} \propto \prod_{j=k+1}^{K-1} L_j^2 \quad}
$$

**直观解读**：

* 早期 hop（k 小）的误差**会被后续所有 hop 的 Lipschitz 放大** → 应**重权**
* 末端 hop（k=K-1）的 product 是空积=1 → 权重最低
* 这是**严格的"early-hop heavy"理论结论**

### 16.4.3 与 V6 经验权重对照

V6 step_weights = $[0.5, 2.0, 1.5, 1.0]$，对照公式 $w_k \propto \prod_{j>k} L_j^2$：

| k | $\prod_{j>k} L_j^2$（理论） | V6 $w_k$ | 一致性 |
|---|---------------------------|---------|--------|
| 0 | $L_1^2 L_2^2 L_3^2$（最大） | 0.5（最小！）| **❌ 不一致** |
| 1 | $L_2^2 L_3^2$ | 2.0（最大）| 不一致 |
| 2 | $L_3^2$ | 1.5 | — |
| 3 | 1（最小） | 1.0 | **✅ 一致** |

**V6 step_weights 与误差传播理论矛盾**——理论说"越早越重"，V6 说"中段最重，早段最轻"。但 V6 也有 pair_loss_weights $[2.5, 1.0, 1.0, 1.0]$ 在另一通道**已经把 hop0 强权重**——所以**联合后的有效 hop0 权重**仍然显著高于 hop3。

**这说明 V6 的设计实质是**：

> "把 hop0 加权放到 pair loss（容易学，单步监督）；把 hop1/hop2 加权放到 rollout loss（chain dynamics 主战场）；hop3 在两个 loss 上都是基线权重"

——这是一个**有道理但缺乏理论形式**的工程方案。

### 16.4.4 Lipschitz 常数 $L_j$ 是可测量的（这是省消融的关键）

$L_j$ 的 PyTorch 估计：

```python
@torch.no_grad()
def estimate_lipschitz_per_hop(model, z_sample_per_hop, eps=1e-3):
    L = []
    for k, z0 in enumerate(z_sample_per_hop):
        delta = torch.randn_like(z0) * eps
        F0 = model.predict_latent_step(z0, t_src[k], t_dst[k], hop_idx=k)["z_pred"]
        F1 = model.predict_latent_step(z0+delta, t_src[k], t_dst[k], hop_idx=k)["z_pred"]
        L_k = (F1 - F0).norm() / delta.norm()
        L.append(L_k.item())
    return L  # 4 numbers
```

* 在已收敛的 V6 checkpoint 上跑这个估计，**只需 ~10 分钟**
* 直接得到 $w_k^\text{optimal} = \prod_{j>k} L_j^2 / Z$
* 这个 step_weights **就是论文里"理论推导"的中心点**——sweep 只需在它附近做小邻域确认，不必盲扫

## 16.5 关键结论 → §18-§19 / §20.6

> **2026-05-02 更新**：原 §16.5（三条捷径）/ §16.6（综合建议替换 8 组 sweep）/ §16.7（最终回答）已被合并到后续章节：
> - **捷径 1（rollout sigma 归一化）** → 升级为 **§20.4 #1**（论文级 critical ablation，必做）
> - **捷径 2（Lipschitz 测量给闭式解）** → 升级为 **§19.4**（已有 50 行脚本模板 + 25% 偏差判定）
> - **捷径 3（P2 + 物理 SNR 类比）** → 已判定不推荐，无后续
>
> §16.4 的 Grönwall 闭式解 $w_k^* \propto \prod_{j>k} L_j^2$ 在 §18.3.3 进一步推广到**多 hop 加权目标**，得到 $w_j^* \propto \sum_{k>j} \beta_k \prod_{i=j+1}^{k-1} L_i^2$——这是 V6 真正命中的形式。

**一行结论**：MAP-Diff 的 $w(t)=(1-t/T)^p$ 没有严格证明；Min-SNR-γ 的精神已被 V6 `target_normalize=true` 在 pair loss 上实现；rollout loss 上还差一步（即 §20.4 #1）。Grönwall 链式误差传播给出 step_weights 的闭式解（§16.4 + §18.3.3），是论文里"理论指导"的硬证据。

---

# 17. V6 经验数据 vs Grönwall 理论加权——单通道分析（已被 §18-§19 修正）

## 17.1 V6 实际权重的精确口径

从 [pet_flow_first_hop_224_v6_transport_first.yaml](PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml#L141) 拿到的精确数值：

```yaml
transport:
  velocity_loss_weight: 1.0
  endpoint_loss_weight: 1.0
  pair_sample_probs:    [0.25, 0.25, 0.25, 0.25]   # 每 hop 等概率采样
  pair_loss_weights:    [2.5,  1.0,  1.0,  1.0]
rollout_loss:
  lambda_start: 0.0
  lambda_end:   4.0       # late-phase λ_r
  endpoint_loss_weight: 0.25
  step_weights:         [0.5,  2.0,  1.5,  1.0]
```

外层（来自训练循环，conversation summary §14.2.3 已确认）：

```
λ_p (pair total) = 15
λ_r (rollout total, late phase) = 4
```

## 17.2 → 17.8 已被 §18-§19 替换

> 早期 §17 的内容（联合权重计算 §17.2、反向解 Lipschitz §17.3、三档判定 §17.4、5 个工程问题 §17.5、综合诊断表 §17.6、5 条结论 §17.7、一句话答案 §17.8）**已被 §18-§19 修正与替换**。
>
> 早期 §17 假设 V6 是纯经验调，并用单通道 Grönwall 与 V6 对比，结论是"近似但不严格满足"。
>
> **§18 发现 V6 实际有半理论公式** $S = \text{ExpoGap} \times v_\text{std}$ 作为指导，**§18.3.3 推导多 hop 加权目标的闭式解** $w_j^* \propto \sum_{k>j}\beta_k\prod L_i^2$，与 V6 step_weights 在 hop1/2/3 上 emergent 命中。**§19 的用户确认锁定 emergent 理论命中成立**。
>
> 早期 §17.5 的 5 个工程问题中，**问题 1（rollout 量级不归一）已升级为 §20.4 #1**——是论文级 critical ablation，必做。问题 2-5 整体已被 §20.4-§20.5 的梯度流动分析覆盖。
>
> **如需历史推导细节，请参考 git 历史；本节仅保留 §17.1 的权重口径以便引用。**


# 18. 修正：V6 step_weights / pair_loss_weights 的**真实设计依据**

> **背景**：用户指出"V6 是经验调出来的"。但通读 [transport_breakthrough_research_v6.md §1.4](PET_LatentResidual/review/plan/transport_breakthrough_research_v6.md) 与 [v4_sf_pilot_early_analysis_20260426.md §10-11](PET_LatentResidual/review/0426/v4_sf_pilot_early_analysis_20260426.md) 后发现：**V6 的权重并非纯经验**——它有一个**半理论公式 $S = \text{ExpoGap}_k \times v_\text{std}^{(k)}$**作为指导，配合"通道-跳分离"原则。本节修正 §17 的"V6 缺乏数学依据"判断，并明确指出 V6 现有公式与 Grönwall 理论的**真实关系**——它们其实在**回答两个不同的问题**。

## 18.1 V6 的真实设计公式（来自 v6 design doc §1.4）

V6 step_weights 的设计基于"rollout 通道在 hop k 的有效信号强度":

$$
\boxed{\quad S_\text{rollout}^{(k)} = \text{ExpoGap}_k \times v_\text{std}^{(k)} \quad}
$$

实测数据（V3 best.pt 上 Path A 多 ckpt diagnostic）：

| Hop | ExpoGap (dB) | $v_\text{std}^{(k)}$ | $S = \text{Gap} \times v_\text{std}$ | 归一化 | 实际 step_weight |
|-----|-------------|---------------------|-----------------------------------|--------|------------------|
| 0 (D50→D20) | **0.00** | 0.00963 | 0.0000 | 0.00 | 0.5（与 pair 重叠，压低） |
| 1 (D20→D10) | 3.38 | 0.00290 | 0.00980 | 2.65 | **2.0（sweet spot）** |
| 2 (D10→D4) | 4.71 | 0.00078 | 0.00367 | 0.99 | 1.5 |
| 3 (D4→NRM) | 6.32 | 0.000141 | 0.00089 | 0.24 | 1.0（Jacobian 饱和，压不下去就保持） |

**核心洞察**（V6 doc 原话）：

> "hop3 虽然 ExpoGap 最大（6.32），但 v_std 极小（0.000141），Jacobian 饱和导致加权无效。"

物理意义：

* **ExpoGap_k**：hop k 在 chain rollout 与 teacher-forcing 之间的 PSNR 偏差——对应"该 hop 实际产生的 exposure bias 累积量"
* **$v_\text{std}^{(k)}$**：该 hop velocity 真值的标准差——对应"该 hop 上有多少可学的信号"
* **乘积 S**：在该 hop 加权能产生多少**有效**学习信号——既需要"误差大（值得学）"，又需要"velocity 不饱和（学得动）"

**通道-跳分离原则**（V6 doc §1.4）：

* `pair` 通道：在 GT 输入上学单步 velocity，**hop0 是独占信号**（rollout 在 hop0 的输入退化为 GT z_D50，与 pair 重复）→ pair_loss_weights 头重 [2.5, 1.0, 1.0, 1.0]
* `rollout` 通道：在 pred chain 上学链式一致性，**hop1-3 才是独占信号** → step_weights 中重 [0.5, 2.0, 1.5, 1.0]
* image_aux：固定 0.04，hop0 像素辅助
* **目标**：每个通道在自己擅长的 hop 上发力，不重复别人的工作

## 18.2 修正 §17：V6 不是"纯经验"，而是"半理论 + 实证微调"

§17 的"V6 缺乏数学依据"判断**不准确**，应修正为：

| 评估维度 | §17 旧结论 | 修正后结论 |
|---------|----------|----------|
| 设计依据 | 纯经验调参 | **半理论公式 $S = \text{Gap} \times v_\text{std}$** 指导 |
| ExpoGap_k | 未提及 | 来自 V3 best.pt 多 ckpt diagnostic 实测 |
| v_std_k | 未提及 | 来自 GT velocity 经验分布的 std |
| 数学严谨度 | 0/10 | **5/10**（启发式但有可验证物理量） |
| 与 Grönwall 理论的关系 | "近似但不严格" | **回答了不同问题**（见 §18.3） |

## 18.3 V6 的 $S = \text{Gap} \times v_\text{std}$ vs Grönwall $w \propto \prod L_j^2$——本质差异

这两个公式**都对**，但在**回答不同的优化目标**。

### 18.3.1 它们各自的优化目标

* **Grönwall 加权**：优化目标 $\mathbb{E}\|\delta_K\|^2$（**仅**末端误差）
  - 推导出 $w_k^* \propto \prod_{j>k} L_j^2$
  - 早期 hop 误差被后续 Lipschitz 放大 → "**早期 hop 重**"

* **V6 加权**：优化目标 $\mathbb{E}\sum_k \beta_k \|\delta_k\|^2$（**多 hop 加权和**）
  - 经验启发：$w_k \propto \text{ExpoGap}_k \times v_\text{std}^{(k)}$
  - 平衡"误差信号大小"与"学习容量"→ "**中段 hop 重**"

### 18.3.2 `val_select` 已经透露 V6 真实优化目标是多 hop

V6 yaml 里实际的 selector：

```python
val_select = 0.5*val_chain_d20_mse + 0.45*val_chain_d10_mse + 0.9*val_chain_d4_mse + 1.5*val_chain_normal_mse
```

这是**带权 4 hop 总误差**，**不是**纯末端误差。所以：

* V6 的真实优化目标 $\approx \beta_1 \|\delta_1\|^2 + \beta_2 \|\delta_2\|^2 + \beta_3 \|\delta_3\|^2 + \beta_4 \|\delta_4\|^2$
* Grönwall 闭式解 $w \propto \prod L_j^2$ **不适用**——它假设了纯末端目标

### 18.3.3 多 hop 目标下的正确闭式解

如果优化目标是 $\sum_k \beta_k \|\delta_k\|^2$，应用同一套链式 Grönwall 推导：

$$
\|\delta_k\|^2 \leq \sum_{j=0}^{k-1} \left( \prod_{i=j+1}^{k-1} L_i^2 \right) \|\epsilon_j\|^2
$$

代入目标得：

$$
\mathbb{E}\sum_k \beta_k \|\delta_k\|^2 \leq \sum_{j=0}^{K-1} \left[ \sum_{k>j} \beta_k \prod_{i=j+1}^{k-1} L_i^2 \right] \|\epsilon_j\|^2
$$

由 Cauchy-Schwarz 等号条件，**多 hop 加权目标的最优 step_weights**：

$$
\boxed{\quad w_j^{*,\text{multi-hop}} \propto \sum_{k=j+1}^{K} \beta_k \prod_{i=j+1}^{k-1} L_i^2 \quad}
$$

**当 $\beta_K = 1, \beta_{<K} = 0$（纯末端）时退化为 Grönwall**。
**当 $\beta_k$ 全部均匀且 $L_j \approx 1$ 时退化为 $w_j \propto K - j$（线性递减）**。

### 18.3.4 与 V6 的对照（2026-05-02 修订）

> **2026-05-02 修订**：原 §18.3.4 表格存在 off-by-one 索引 bug + 把 pair_loss_weights[0]=2.5 错误地解读为 "Grönwall 补偿"。已用户确认两处均不成立。下面是修正后的版本，并诚实标注当前 emergent 命中论证的强度。

#### 修正 1：β-sensitivity 公式的正确索引

step_j 直接产生 hop j+1 的输出 z_{j+1}^pred，所以 step_j 的误差 ε_j 出现在 c_{j+1}, c_{j+2}, ..., c_K 中。**Σ 应从 k=j+1 开始**，**不应排除 β_{j+1}**：

$$
w_j^* \propto \sum_{k=j+1}^{K} \beta_k \prod_{i=j+1}^{k-1} L_i^2
$$

代入 V6 的 $\beta = [\beta_1, \beta_2, \beta_3, \beta_4] = [0.5, 0.45, 0.9, 1.5]$（对应 chain_d20 / d10 / d4 / normal），假设 $L_i \approx 1$：

| j | hop k 范围 | $\sum_{k=j+1}^{4} \beta_k$ | 闭式解 $w_j^*$（修正后） | V6 实际 $w_j^\text{step}$ | normalized 对比 |
|---|-----------|---------------------------|------------------------|--------------------------|----------------|
| 0 | 1..4 (全部) | 0.5+0.45+0.9+1.5 = **3.35** | 3.35 | 0.5 | 闭式 1.0 vs V6 0.25 → **差 4×** |
| 1 | 2..4 | 0.45+0.9+1.5 = **2.85** | 2.85 | 2.0 | 闭式 0.85 vs V6 1.0 → 差 18% |
| 2 | 3..4 | 0.9+1.5 = **2.4** | 2.4 | 1.5 | 闭式 0.72 vs V6 0.75 → 差 4% |
| 3 | 4 | 1.5 | **1.5** | 1.0 | 闭式 0.45 vs V6 0.5 → 差 11% |

**修正后的命中评分**：

* hop2/hop3：normalized 误差 4-11%——**确实接近**
* hop1：normalized 误差 18%——**勉强算接近**
* hop0：normalized **闭式解最高 (1.0) 而 V6 最低 (0.25)，方向相反，差 4 倍**——**严重不命中**

整体 L1 距离 = 0.985（normalized 后 4 个 hop 的差异之和），其中 hop0 单独贡献 0.75。

#### 修正 2：早期 §18.3.4 的"pair_loss_weights[0]=2.5 补偿 hop0"论证不成立

早期版本声称 hop0 的 step_weight=0.5 anomaly 是由 pair_loss_weights[0]=2.5 补偿的。**用户确认此论证错误**：

* pair loss 是 normalized velocity loss（已经 σ-normalize），rollout step loss 是 raw latent loss——两者梯度 distribution 不同
* pair_loss_weights[0]=2.5 的**真实动机**：hop0 有 pixel forcing 依赖（gate_pix + pixel encoder），需要额外的训练信号驱动 gate 生长，避免 gate 死在 floor
* 这是**激活特定架构组件**的设计选择，**与 Grönwall β-加权无关**

#### 修正 3：还有一个 V6 step_weights 在做的事情——σ²·dt² 量级补偿

rollout step_loss 的天然量级是 (σ_j·dt_j)²，跨 4 hop 差 7.5×：

| hop | σ_j | dt_j | (σ_j·dt_j)² | 相对值 |
|-----|-----|------|-------------|--------|
| 0 | 0.00963 | 3 | 8.35e-4 | **7.45×** |
| 1 | 0.00295 | 5 | 2.17e-4 | 1.94× |
| 2 | 0.00078 | 15 | 1.35e-4 | 1.20× |
| 3 | 0.000141 | 75 | 1.12e-4 | 1.0× |

V6 step_weights × 自然量级 normalized = **[0.96, 1.0, 0.47, 0.26]**——大致单调递减。**V6 的 step_weights 设计可能主要是在做 σ²·dt² 量级补偿**，让 4 个 hop 的有效 loss 量级接近——而不是在严格命中 Grönwall 闭式解。

#### 修正后的 emergent 命中判定

| 论证 | 早期声称 | 修正后 |
|------|---------|--------|
| step_weights 命中 multi-hop closed-form | hop1/2/3 几乎完全一致 | hop2/3 接近（4-11%），hop1 勉强（18%），**hop0 反向（4×）** |
| pair[0]=2.5 补偿 hop0 | 显式补偿成立 | **不成立**——pair[0]=2.5 是 pixel forcing gate 驱动信号 |
| 论文 contribution 强度 | "emergent 命中多 hop Grönwall 闭式解"（强声明） | "**hop2/3 接近闭式解，hop0/1 偏离**——主要是 σ²·dt² 量级补偿主导，Grönwall 加权效应在 L≈1 下本身就微弱" |

**为什么 σ²·dt² 主导而 Grönwall 弱？**因为 §17.1 反解的 L_j ∈ [0.96, 1.13]，几乎 = 1，此时 Π L_i² ≈ 1，β-加权和近似等于 β 反向累计（线性递减），这种情况下 step_weights 的关键作用退化为"补偿 σ²dt² 自然量级 + 给末端基础权重"，**而不是"放大早期 hop 以对抗 Lipschitz amplification"**——后者只在 L 显著大于 1 时才重要。

**对论文的影响**：§19 的"emergent 命中多 hop Grönwall 闭式解"叙事必须**降级**为：

> "We show V6's empirically tuned step_weights satisfy two engineering constraints simultaneously: (1) they compensate for the natural rollout MSE magnitude (σ²·dt²) which varies 7.5× across hops; (2) they implicitly track the multi-hop weighted objective in hops 2-3, with hop 0 and hop 1 deviating due to additional architectural concerns (pixel forcing gate activation). The closed-form Grönwall correction is small in our regime since the implicit L_j≈1, making the σ²·dt² compensation the dominant effect."

这是一个**更弱但更诚实**的 contribution。配合 §20.6 行动 7（σ-normalize ablation），可以变成 critical experiment：

* **如果**在 σ-normalized rollout 下 V6 重训仍命中 [0.5, 2.0, 1.5, 1.0]，则量级补偿不是主因，Grönwall 是真正的 emergent 命中（强 contribution）
* **如果**重训后 step_weights 收敛到 [1.0, 1.0, 1.0, 1.0] 附近，则 σ²·dt² 补偿是主因，Grönwall 命中是巧合（弱 contribution，但仍有 σ²·dt² 自动补偿这一发现）

→ 从 V6 联合权重比 $W_k/W_{k+1}$ 反解得到的隐含 $L_j \in [0.96, 1.13]$（口径见 §17.1）依然成立，但**不再能解释为"emergent 命中多 hop Grönwall"**——它只能说明 V6 的联合权重在量级上不偏离理论极限。

## 18.4 ExpoGap × v_std heuristic 的"理论位置"

V6 的 $S = \text{Gap} \times v_\text{std}$ 启发式是这两套公式的**简化代理**：

* $\text{ExpoGap}_k$ ≈ Grönwall 中的"累积误差因子" $\prod_{j<k} L_j$（实测代理）
* $v_\text{std}^{(k)}$ ≈ "该 hop 上学习信号的容量"（不在 Grönwall 中显式出现，但与 Lipschitz 反向相关——velocity 越小局部越线性、Jacobian 越饱和、$L_j$ 越接近 1）
* 乘积 ≈ "考虑学习容量加权后的误差累积量"

**理论解读**：

* 在末端 hop（k=3），ExpoGap 最大但 v_std 极小 → 学不动（饱和）→ 不应过权
* 在中段 hop（k=1-2），两个因子都中等 → sweet spot
* 在起始 hop（k=0），ExpoGap=0（无 exposure bias）→ 不应被 rollout 通道加权

这与 §18.3.3 多 hop Grönwall 的结论方向**完全一致**，但 V6 用了**实测代理**而不是理论 Lipschitz 计算——更鲁棒（不依赖 $L_j$ 估计精度），但也更经验。

## 18.5 修订后的"理论 vs 经验"判定

| 子项 | V6 实际做法 | 理论严谨版 | 状态 |
|------|-----------|----------|------|
| 优化目标 | val_select 多 hop 加权 $\beta=[0.5, 0.45, 0.9, 1.5]$ | $\sum_k \beta_k \|\delta_k\|^2$ | **理论一致** ✅ |
| step_weights[1-3] 形态 | [2.0, 1.5, 1.0] | 多 hop Grönwall $\sum_{k>j} \beta_k \prod L_i^2$ | **emergent 接近最优** ✅ |
| step_weights[0] = 0.5 | 故意压低 | 公式给 2.85——但 pair 通道补偿 | **跨通道补偿** ⚠️ |
| pair_loss_weights[0] = 2.5 | 头重 | hop0 独占信号 + 跨通道补偿 | **设计合理** ✅ |
| ExpoGap × v_std heuristic | 实测代理 | 严格说应该是 $\prod L_j$ × Jacobian 修正 | **代理合理** ✅ |

**重新结论**：V6 不是"凭空乱调"，而是用 **(a) 多 hop 加权目标** + **(b) 两通道分工** + **(c) ExpoGap × v_std 实测代理**实现了多 hop Grönwall 推广闭式解的近似。**它已经接近 emergent 理论最优**——只是文档没有用这种语言写出来。

## 18.6 仍需用户确认的关键问题

虽然 V6 的设计依据已经从文档恢复，但仍有 4 个问题决定后续行动方向：

### Q1：ExpoGap_k = [0, 3.38, 4.71, 6.32] 数据来自 V3 best.pt——V6 训练后是否重测过？

* 如果 V6 收敛后 ExpoGap 分布与 V3 显著不同（例如 V6 能压低末端 ExpoGap），则 step_weights 应该重算
* 如果几乎不变，则当前形态可保留

### Q2：v_std_k = [0.0096, 0.0029, 0.00078, 0.000141] 是从 GT velocity 算的，还是从模型预测算的？

* 从 GT 算 → 这是数据本身的物理量，与模型无关，可作为长期常数
* 从模型算 → V6 训练后会变，需要更新

### Q3：val_select 的 β = [0.5, 0.45, 0.9, 1.5]——这套权重的设计来源是什么？

* 如果它本身也是经验调的，则 §18.3 的"理论一致"链条断在这里，因为 β 是被优化的目标但其本身无理论
* 如果有论据（例如来自下游 PET 临床指标的相对重要性），则整个推导链就有了起点

### Q4：是否愿意把 §18.3.3 的多 hop Grönwall 闭式解作为论文卖点？

* 优点：把 V6 的"经验调参"重新叙述为"emergent 理论命中"，提升论文 contribution
* 代价：需要一节理论推导 + 一组对照 ablation（用闭式解 vs 当前 V6 vs constant，3 组训练）
* 风险：emergent 一致性可能在不同任务/超参下断裂——需要敏感性分析

---

# 19. 用户确认的关键事实（2026-05-02）—— Emergent 理论命中成立

> **⚠️ 2026-05-02 后续修订（PARTIALLY SUPERSEDED）**：本节捕捉了用户当时给出的关键事实确认，**这些事实本身仍然成立**（ExpoGap 稳定、β 是临床先验、§17 单通道 Grönwall 已落入双通道分工等）。但本节随后给出的 contribution 强度叙事（"emergent 命中多 hop Grönwall 闭式解"）在同一天的独立审视中被进一步修订——见 **§21**：
>
> 1. §18.3.4 表存在 off-by-one 求和 bug，已修正；emergent 命中评分由"hop1/2/3 几乎完全一致"降级为"hop2/3 接近，hop0/1 偏离"
> 2. "pair_loss_weights[0]=2.5 补偿 hop0 step_w=0.5"论证不成立——pair[0] 的真实动机是激活 pixel forcing gate
> 3. 替代解释：σ²·dt² 量级补偿（跨 hop 7.5×）很可能是 V6 step_weights 的主导效应，Grönwall 加权效应在 L≈1 下本身微弱
> 4. 论文 contribution 强度需按 §21.8 重写
>
> 阅读建议：先读 §21（最新结论），再回头看本节作为 emergent 命中叙事的早期版本。

> **本节锁定 §18.6 四个问题的答案**——用户已亲自确认。这些答案把 §18.3 的"假设性推导"升级为**已验证的论文级 contribution**。

## 19.1 四个问题的确认答复

| # | 问题 | 用户答复 | 含义 |
|---|------|---------|------|
| Q1 | V6 是否重测过 ExpoGap？ | ✅ **已重测，与 V3 几乎一致（差异 < 0.5 dB）** | ExpoGap 是 chain dynamics 的稳定属性，V6 没把它消除 |
| Q2 | v_std 是 GT 还是模型预测算的？ | ✅ **从 GT velocity 算的（pair_v_std 来自数据集预计算）** | v_std 是 PET 物理 transport 的客观属性，与训练无关 |
| Q3 | val_select β 的设计来源？ | ✅ **有外部依据（临床/文献来源）** | β 不是循环论证——多 hop 加权目标有外部理论起点 |
| Q4 | 论文策略？ | ✅ **策略 C：先更新数据看是否真接近闭式解** | 但 Q1+Q2+Q3 已把 C 的前提条件全部满足 |

## 19.2 这四个 yes 一起意味着什么

每个起点都有外部依据，整套推导链不是循环：

```
[临床/文献] → β = [0.5, 0.45, 0.9, 1.5]                      ← Q3 ✅
[GT 物理]   → v_std = [0.0096, 0.0029, 0.00078, 0.000141]     ← Q2 ✅
[V6 实测]   → ExpoGap = [0, 3.38, 4.71, 6.32]（≈V3）          ← Q1 ✅
                            ↓
            多 hop Grönwall 闭式解 w_j* ∝ Σ_{k>j} β_k ∏ L_i²
                            ↓
                  对照 V6 step_weights = [0.5, 2.0, 1.5, 1.0]
                            ↓
              hop1/2/3 闭式解 = [2.4, 1.5, 0]，V6 = [2.0, 1.5, 1.0]
                            ↓
              hop2 完美对齐，hop1 差 17%，hop3 由 base term 解释
                            ↓
              hop0 由 pair 通道补偿（pair_loss_weights[0]=2.5）
                            ↓
                **EMERGENT 理论命中成立** ✅
```

## 19.3 这意味着 V6 是什么？

**V6 不是经验调参的产物——它是临床先验 + PET 物理常数 + 多 hop 加权目标 + 通道分工"emergent"形成的多 hop Grönwall 闭式解的近似实现。**

具体地：

* **β = [0.5, 0.45, 0.9, 1.5]** 是临床先验输入（外部锚点 1）
* **v_std = [0.0096, 0.0029, 0.00078, 0.000141]** 是 GT 数据物理常数（外部锚点 2）
* **ExpoGap = [0, 3.38, 4.71, 6.32]** 是 chain dynamics 的稳定可观测量（V3/V6 一致）
* 多 hop Grönwall 闭式解给出最优形态
* V6 通过 (a) ExpoGap × v_std 启发式 + (b) 通道-跳分离原则**emergent 命中**了这个最优解
* 偏差仅 hop1 的 17%——可解释（hop1 sweet spot 边界判断）

## 19.4 策略 C → 验证步骤（用户选了 C，且前提已满足）

用户选了"策略 C：先更新数据看是否真接近闭式解"。Q1+Q2 已经确认数据稳定，所以**剩余的验证**只有 1 步：

### 唯一剩余验证：测量真实 $L_j$，对比闭式解

**做法**（[§16.5 捷径 2](review/plan/ARCHITECTURE_ANALYSIS_20260501.md) 的 Lipschitz 测量脚本）：

```python
@torch.no_grad()
def estimate_per_hop_lipschitz(model, val_loader, n_samples=64, eps=1e-3):
    """在 V6 当前 best ckpt 上估计 4 个 L_j。"""
    L_per_hop = []
    for k in range(4):
        # 从 val 取 z_src for hop k
        z = sample_z_at_hop(val_loader, hop=k, n=n_samples)
        delta = torch.randn_like(z) * eps
        F0 = model.predict_latent_step(z, t_src[k], t_dst[k], hop_idx=k)["z_pred"]
        F1 = model.predict_latent_step(z + delta, t_src[k], t_dst[k], hop_idx=k)["z_pred"]
        L_k = ((F1 - F0).norm(dim=(1,2,3)) / delta.norm(dim=(1,2,3))).mean().item()
        L_per_hop.append(L_k)
    return L_per_hop
```

代入测得的 $L_j$ 到闭式解 $w_j^* \propto \sum_{k>j} \beta_k \prod_{i=j+1}^{k-1} L_i^2$，对照 V6 step_weights：

| 验证情境 | 行动 | 论文叙述 |
|---------|------|---------|
| 闭式解与 V6 step_weights 对齐 < 20% | **不改 V6 任何配置**，直接写论文 emergent 理论命中 | "We show V6's empirically tuned weights emergently satisfy a closed-form solution derived from multi-hop Grönwall propagation under clinical-prior $\beta$." |
| 偏差 20-40% | 跑 1 组小幅修正（用闭式解替换 step_weights）| "We further refine the weights using the analytic closed-form, achieving X% improvement." |
| 偏差 > 40% | 假设需要重新审视——可能 chain dynamics 中 $L_j$ 已经不接近 1 | 按 §20.6 优先级清单处理（rollout sigma 归一化 + Lipschitz 实测） |

**预测**：基于 §17.1 权重口径反解得到的隐含 $L_j \in [0.96, 1.13]$，预期对齐误差 < 25%。

## 19.5 论文 contribution 升级（基于 §19 全部确认）

原 IDEA_REPORT 给的"V6 设计"叙述可以**直接重写**成：

> **A weighted multi-hop chain training objective with closed-form weights derived from
> (1) clinical-prior dose-importance prior $\beta$,
> (2) PET physical velocity invariants $v_\text{std}$,
> (3) measured per-hop Lipschitz constants $L_j$,
> via multi-hop Grönwall error propagation. We show this closed-form is emergently approximated by a simpler heuristic $S_k = \text{ExpoGap}_k \times v_\text{std}^{(k)}$, providing both theoretical grounding and a practical proxy.**

这相比"step_weights = [0.5, 2.0, 1.5, 1.0]，经验调出来"的叙述，**理论 contribution 是 ICCV/CVPR-grade 的提升**——同时还有 emergent 一致性作为"the design is robust" 的证据。

## 19.6 接下来的精确行动清单（按依赖排序）

| # | 行动 | 依赖 | 代价 | 产出 |
|---|------|------|------|------|
| 1 | 写 50 行 `estimate_per_hop_lipschitz.py` | V6 best.pt | 30 分钟 | 测得 $L_j$ |
| 2 | 把 $\beta, L_j$ 代入闭式解 $w_j^*$，对照 V6 | 步骤 1 | 5 分钟 | 偏差表 |
| 3 | 如果偏差 < 25% → 论文 §3 用 §19.3 推导链 | 步骤 2 | 1 天 | 论文章节 |
| 4 | 如果偏差 25-40% → 加 1 组修正 ablation | 步骤 2 | 1 次训练 | 论文 Table |
| 5 | 仍考虑 §16.5 捷径 1（rollout sigma 归一化）作为单独 ablation | 独立 | 10 行代码 + 1 次训练 | 控制实验 |

**最小可发表组合**：步骤 1+2+3 → 不需新训练，纯从已有 V6 ckpt + 文献起点 → 整套理论叙述就位。

## 19.7 一句话总结

> **V6 不是经验调出来的——它是 (临床先验 β) + (PET 物理 v_std) + (chain dynamics ExpoGap) 经过通道分工"emergent"接近多 hop Grönwall 闭式解的工程实现。**
>
> **唯一剩余的验证是测量 $L_j$（50 行脚本，10 分钟）**——如果偏差 < 25%，这套理论命中可以直接成为论文的 ICCV/CVPR-grade contribution，无需新训练。

---


---

# §20. 整体 Loss 公式与梯度流动深入审查（2026-05-02）

> **触发问题**：用户提问"整体 loss 的公式设计的数学逻辑是否合理？此时梯度的流动是否平滑？"
>
> **方法**：把 train_first_hop.py:1918-1934 的 `total_loss = ...` 反向追到每一项的源头，做四件事：
> 1. **公式重写**：把所有缩放、调度、归一化展开成单行数学
> 2. **量级分析**：用 V6 实际 σ_k、Δt_k、典型 v_err 估算每项数值规模
> 3. **梯度流动图**：标注每项 loss 的反向传播路径与终点参数
> 4. **客观问题清单**：列出 5 个真实存在的设计漏洞（不掩饰，不夸大）

## §20.1 总公式重写（V6 active 配置下）

V6 的 yaml 中 `foc.enabled=False`、无 `alignment` 块，所以实际只有 **pair + rollout + image_aux** 三项。
[train_first_hop.py L1918-1924](PET_LatentResidual/train_first_hop.py#L1918):

```python
total_loss = (
    pair_loss_weight * pair_losses["total"]                        # 15 × L_pair
    + rollout_losses["lambda_roll"] * rollout_losses["loss_total"] # λ_r(t) × L_roll
    + float(lambda_img) * loss_img                                 # 0.04 × L_img
    + foc_losses["lambda_foc"] * foc_losses["loss_total"]          # 0 (disabled)
    + float(lambda_align) * loss_align                             # 0 (disabled)
)
```

**展开成数学（V6 静态/动态权重全部展开）**：

$$
\boxed{\quad
\mathcal{L}_\text{total}(\theta; t) =
\underbrace{15 \cdot \mathcal{L}_\text{pair}(\theta)}_\text{transport 主项 ≈ 75-95\%}
+ \underbrace{4\,\rho(t) \cdot \mathcal{L}_\text{roll}(\theta)}_\text{chain 修正 ≈ 0.1-3\%}
+ \underbrace{0.04 \cdot \mathcal{L}_\text{img}(\theta)}_\text{decoder anchor ≈ 5-15\%}
\quad}
$$

其中 $\rho(t) = \mathrm{clip}((t - 50\text{K})/100\text{K},\, 0,\, 1)$ 为线性 ramp。

### §20.1.1 Pair loss 的完整展开

[train_first_hop.py:289-358](PET_LatentResidual/train_first_hop.py#L289):

$$
\mathcal{L}_\text{pair}(\theta)
= w_v^\text{eff}(t) \cdot \mathcal{L}_v
+ w_e \cdot \mathcal{L}_e
$$

$$
\mathcal{L}_v = \mathbb{E}_{(z,k)\sim B}\left[
w_k^\text{pair} \cdot \frac{1}{D}\left\| v_\theta(z_s, t_s, t_d, k) - \frac{(z_d - z_s)}{\sigma_k \cdot \Delta t_k} \right\|_2^2
\right]
\quad\text{(velocity, σ-normalized)}
$$

$$
\mathcal{L}_e = \mathbb{E}_{(z,k)\sim B}\left[
w_k^\text{pair} \cdot \frac{1}{D}\left\| \frac{z_\text{pred} - z_d}{\Delta t_k} \right\|_2^2
\right]
\quad\text{(endpoint, dt-normalized, NO σ)}
$$

**动态系数**（V6 `velocity_rebalance.enabled=true, mode=sqrt_ratio, clip=[1,4]`）：

$$
w_v^\text{eff}(t) = w_v \cdot \mathrm{clip}\!\left(\sqrt{\mathcal{L}_e^\text{detach}/\mathcal{L}_v^\text{detach}},\ 1,\ 4\right)
$$

**静态系数**（V6 `pair_loss_weights=[2.5, 1.0, 1.0, 1.0]`、`pair_sample_probs=[0.25, 0.25, 0.25, 0.25]`）：

$$
w_k^\text{pair} \in \{2.5, 1.0, 1.0, 1.0\},\quad
P(k) = 0.25 \;\forall k
$$

预测公式（[model_first_hop.py L478-520](PET_LatentResidual/pet_lr/model_first_hop.py#L478)）：

$$
v_\theta = v_\text{shared} + v_\text{hop},\quad
z_\text{pred} = z_s + \underbrace{(v_\theta \cdot \sigma_k)}_\text{v\_total} \cdot \Delta t_k
$$

### §20.1.2 Rollout loss 的完整展开

[rollout_first_hop.py L48-115](PET_LatentResidual/pet_lr/rollout_first_hop.py#L48)：

$$
\mathcal{L}_\text{roll}(\theta) =
\frac{\sum_{k=0}^{3} w_k^\text{step} \cdot \mathbb{E}_B\!\left[\frac{1}{D}\|z_k^\text{pred} - z_k^\text{gt}\|_2^2\right]}{\sum_{k=0}^{3} w_k^\text{step}}
\quad\text{(NO σ normalization)}
$$

**链式状态传递**（mix_latent + STE，[rollout_first_hop.py L21-37](PET_LatentResidual/pet_lr/rollout_first_hop.py#L21)）：

```
forward (numerical):  z_curr_{k+1} = (1-α) z_k^gt + α z_k^pred
backward (gradient):  ∂z_curr_{k+1}/∂z_k^pred = 1   (STE 强制 z_gt 路径无梯度)
                      ∂z_curr_{k+1}/∂z_k^gt   = 0
```

**关键洞察**：α 调度只影响**前向输入分布**（GT 还是 pred），**不影响梯度路径**。链式梯度永远从后续 hop 反传到前面 hop 的 z_pred。

**静态系数**（V6）：$w_k^\text{step} = (0.5, 2.0, 1.5, 1.0)$，$\sum = 5.0$。

### §20.1.3 Image_aux loss 的完整展开

[train_first_hop.py L697-734](PET_LatentResidual/train_first_hop.py#L697)，仅 hop0：

$$
\mathcal{L}_\text{img}(\theta)
= 1.0 \cdot \mathcal{L}_\text{L1} + 0.25 \cdot \mathcal{L}_\text{SSIM} + 0.10 \cdot \mathcal{L}_\text{seam},\quad
\text{border\_weight}=2.0
$$

其中 $z_\text{pred,hop0}$ 经过 frozen ViT-MAE decoder（freeze_rae=true）回到像素空间。梯度路径：

```
L_img → decoder.forward (frozen, 不更新) → z_pred(hop0) → backbone + hop_residual + pixel_encoder
```

## §20.2 量级分析（用 V6 实测 σ、Δt 估算）

### §20.2.1 各 hop 的 σ_k、Δt_k、σ²Δt²（链式损失放大因子）

V6 `pair_v_std`、t_map：

| hop k | σ_k | Δt_k | σ_k²·Δt_k² | σ_k·Δt_k |
|---|---|---|---|---|
| 0 (D50→D20)   | 9.63e-3 | 3   | 8.34e-4 | 2.89e-2 |
| 1 (D20→D10)   | 2.95e-3 | 5   | 2.17e-4 | 1.47e-2 |
| 2 (D10→D4)    | 7.75e-4 | 15  | 1.35e-4 | 1.16e-2 |
| 3 (D4→Normal) | 1.41e-4 | 75  | 1.12e-4 | 1.06e-2 |

**关键观察**：σ²·Δt²（rollout step loss 的天然量级）跨 4 个 hop 只差 **7.4×**——比 σ² 单独的 4700× 差异要温和得多，因为更长的 Δt 部分补偿了更小的 σ。但仍**不是同尺度**，这是 §20.4 #1（rollout step_loss 未做 σ 归一化）的根源。

### §20.2.2 Phase III（>150K，全权重激活）的实测量级

假设 backbone 已部分收敛，典型 `vel_err_normalized` ≈ 0.02–0.05（这是 v_θ_raw 与 GT 归一化速度的差），逐项估算：

| 损失项 | 公式 | 典型值 | 与 ×权重后 | 占总 loss 比 |
|---|---|---|---|---|
| $\mathcal{L}_v$ | E[w·\|v_err_norm\|²] | 1e-3 | $15 \cdot 1\cdot 1e\text{-3} = 1.5e\text{-2}$ | **~85%** |
| $\mathcal{L}_e$ | $\sigma_k^2 \cdot \mathcal{L}_v \approx 9e\text{-5}\cdot 1e\text{-3}$ | 9e-8 | $15 \cdot 9e\text{-8} = 1.4e\text{-6}$ | <0.01% |
| $\mathcal{L}_\text{step,k}$ | $\sigma_k^2 \Delta t_k^2 \cdot \mathcal{L}_{v,k}$ | ~1e-7 | (累计后 ~1e-6) | – |
| $\mathcal{L}_\text{roll}$ | 加权平均 | 1e-6 | $4 \cdot 1e\text{-6} = 4e\text{-6}$ | **~0.02%** |
| $\mathcal{L}_\text{img}$ | l1+0.25·ssim+0.1·seam | 0.05 | $0.04\cdot 0.05 = 2e\text{-3}$ | **~12%** |

**$\Rightarrow$ 数值数量级排序：pair (15e-3) ≫ image_aux (2e-3) ≫ rollout (4e-6)**，比例约 **3750 : 500 : 1**。

### §20.2.3 这个比例真的合理吗？

**辩护方（合理）**：

- **L_v 和 L_step 不是直接可比的损失**——L_v 是归一化速度的均方误差（无量纲），L_step 是非归一化潜空间 MSE（带 σ²Δt² 量纲）。比较它们的数值大小本身没意义，应该比较**梯度向量长度**。
- 实际梯度上，rollout 的链式反传给每个 hop 的 backbone 参数提供了 transport 主任务**之外**的修正信号——这个信号方向不同，所以即使量级小也有效。

**质疑方（不合理）**：

- 如果只看 pair_total 和 rollout_total 的标量值，rollout 占 0.02%——这意味着 `loss_balance_watch` 监控（默认阈值 0.85，参见 yaml L67-72）几乎一定会**永久警告 pair 主导**，但实际不会因此触发任何修正动作（`loss_balance_watch_enforce=false`）。
- 这种"号称三任务但实际是单任务 + 噪声"的情形是 multi-task learning 文献里被反复警告的反模式（Sener & Koltun 2018，GradNorm Chen et al. 2018）。

**我的判断**：**目前比例不合理**，但**理由比表面看到的更微妙**——见 §20.4 的问题 #1 详细论证。

## §20.3 梯度流动图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          L_total = 15·L_pair + 4ρ(t)·L_roll + 0.04·L_img │
└──────┬──────────────────────────┬─────────────────────────┬─────────────┘
       │                          │                         │
       ▼                          ▼                         ▼
┌──────────────┐         ┌────────────────┐        ┌──────────────────┐
│   L_pair     │         │    L_roll      │        │   L_img (hop0)   │
│ (per-batch   │         │ (chain MSE,    │        │  l1+SSIM+seam    │
│  pair sample)│         │  4 hops)       │        │  pixel space     │
└──────┬───────┘         └────────┬───────┘        └────────┬─────────┘
       │                          │                         │
       │ via 1 forward            │ via 4 forwards          │ via 1 forward
       │ (random hop)             │ (sequential chain)      │ + frozen decode
       │                          │                         │
       ▼                          ▼                         ▼
┌──────────────┐         ┌────────────────┐        ┌──────────────────┐
│ v_total_raw  │         │ z_curr → z_pred│        │ z_pred(hop0)     │
│ z_pred       │         │ (chain via STE)│        │ → ViT-MAE decode │
└──────┬───────┘         └────────┬───────┘        └────────┬─────────┘
       │                          │                         │ (frozen,no ∇)
       │                          │                         ▼
       └─────────┬────────────────┴─────────────────────┐
                 │                                       │
                 ▼                                       │
        ┌─────────────────────────────────────────────┐ │
        │  Trainable params (反传终点)                 │◄┘
        │  • backbone DiT (shared velocity head)       │
        │  • hop_residual_head (per-hop adjustment)    │
        │  • pixel_encoder (hop0 only path)            │
        │  • g_pix_raw, lambda_hop_raw (gates)         │
        └─────────────────────────────────────────────┘
```

### §20.3.1 各 loss 项对参数的"梯度强度图谱"

| 参数 | L_pair | L_roll (Phase III) | L_img |
|---|---|---|---|
| backbone DiT | ✓✓✓ (主信号) | ✓✓ (链式修正) | ✓ (hop0 路径) |
| hop_residual_head[0] | ✓✓ | ✓ | ✓ |
| hop_residual_head[1,2,3] | ✓ | ✓ | ✗ |
| pixel_encoder | ✓ (hop0 only) | ✓ (hop0 only) | ✓✓ (强像素信号) |
| g_pix_raw | ✓ (hop0) | ✓ (hop0) | ✓✓ |
| lambda_hop_raw | ✓ | ✓ | ✓ (hop0) |

**关键观察 1**：链式损失通过 STE 把后续 hop 的梯度灌入**前面 hop 的 backbone 调用**——这是隐性的"递归 BPTT"。当 α=1 时，hop3 的损失梯度需要穿过 4 次 backbone 前向才能到达 hop0 的参数（深度膨胀 4 倍的有效计算图）。

**关键观察 2**：image_aux 完全只走 hop0 路径，对 hop1-3 的 hop_residual_head 没有直接梯度。它是"hop0 锚定器"，不是"全局图像质量调节器"。

## §20.4 客观问题清单（5 个真实漏洞）

### 🔴 问题 #1：rollout step_loss 没有 σ 归一化，step_weights 同时在做两件事

**事实**：

- pair velocity 损失：`||v_θ - v*/σ||²`——σ-归一化，跨 hop 同尺度。
- rollout step 损失：`MSE(z_pred, z_gt)`——**没有 σ 归一化**。

**后果**：根据 §20.2.1，rollout step loss 的天然量级（σ²·Δt²）在 hop 0 vs hop 3 上差 7.4×。当前 step_weights=[0.5, 2.0, 1.5, 1.0] 同时承担两个任务：

1. 多 hop Grönwall 闭式解（§18.3 推导出的 chain importance：$w_j^* \propto \sum_{k>j}\beta_k\prod L_i^2$）
2. 量级补偿（按 §20.2.1 应该是 [1.00, 0.26, 0.16, 0.13] 比例，反向于 σ²Δt²）

**这两个目标方向不一致**——闭式解要求 hop1 weight 较高，而量级补偿要求 hop0 weight 较高。当前 V6 的 [0.5, 2.0, 1.5, 1.0] 是两者妥协的结果，**任何一个目标都没有干净实现**。

**论文意义**：§19 推断的"V6 emergent 命中多 hop Grönwall 闭式解" 在 hop2 完美对齐——但部分原因是 σ²Δt² 在 hop 1-3 比较平（差 1.9×）所以量级补偿需求小。**hop0 偏离闭式解推荐值 (2.85) 而是用 0.5 的真实原因，可能是 σ²Δt² 量级最大、step loss 最容易爆炸需要压制**。也就是说——闭式解命中和量级补偿在 hop1-3 是协同的，但在 hop0 是冲突的。

**如何验证**：把 `_latent_loss` 改成 σ-normalized 版本（10 行代码）：

```python
def _latent_loss_normalized(pred, target, sigma_k, dt_k, loss_type='mse'):
    # 把 z 空间残差转换到归一化速度空间，跨 hop 同尺度
    delta_v_norm = (pred - target) / (sigma_k * dt_k).clamp_min(1e-8)
    return F.mse_loss(delta_v_norm, torch.zeros_like(delta_v_norm))
```

如果归一化后 V6 step_weights 仍然最优 → emergent 命中是 chain importance 而非量级补偿；
如果归一化后最优解变成更接近 [0.5, 2.4, 1.5, 0]（hop3 的 base term）→ 当前是量级补偿污染。

**这是一个 1 次训练就能切割的关键消融**。

### 🟠 问题 #2：velocity_rebalance 在 V6 配置下基本是死代码

**事实**：

- $\mathcal{L}_e \approx \sigma_k^2 \cdot \mathcal{L}_v$（§20.2.2 推导：endpoint 用 dt 归一但用了非归一化的 z_pred 与 z_dst，结果 endpoint = σ²·velocity）
- $\sqrt{\mathcal{L}_e/\mathcal{L}_v} = \sigma_k$，对 hop0 而言 $\approx 0.0096 \ll 1$
- 经过 `clip_min=1.0` → 永远是 1.0
- `velocity_weight_eff = 1.0 × 1.0 = 1.0` 恒等于 `velocity_weight`

→ velocity_rebalance 在 V6 的 target_normalize=true 情境下**完全不起作用**。每个 batch 都触发 `.item()` 同步（block CUDA stream，约 1-2ms 浪费）。

**证据**：[train_first_hop.py L335-345](PET_LatentResidual/train_first_hop.py#L335) `clip(scale_t, min=clip_min, max=clip_max)` 加 `.item()`。

**建议**：要么关掉它（`enabled: false`，省 1-2ms），要么把 clip_min 降到 0.1 让它真的能动态平衡（但前提是先确认 endpoint 信号有意义——见问题 #3）。

### 🟠 问题 #3：endpoint loss 在 target_normalize=true 时几乎无信号

**事实**：从 §20.2.2 推导，endpoint 项 = $\sigma_k^2 \cdot$ velocity 项。对 hop3 而言 $\sigma^2 = 2 \cdot 10^{-8}$。

数值上：
- pair_total = $w_v^\text{eff} \mathcal{L}_v + w_e \mathcal{L}_e \approx 1 \cdot \mathcal{L}_v + 1 \cdot 10^{-8}\mathcal{L}_v \approx \mathcal{L}_v$
- endpoint 项的相对贡献 < 10⁻⁶

→ V6 的 pair loss 实际上**就是 σ-normalized velocity loss**，endpoint 完全是装饰。这本身不是错误，但意味着：

1. `endpoint_pair_weighting=true` 没有实际效果
2. `endpoint_dt_normalize=true` 没有实际效果（已经被 σ² 压死）
3. `endpoint_loss_weight=1.0` 没有实际效果

**为什么是问题**：MeanFlow 的原始论文（Geng et al. 2024, "Mean Flows for One-step Generative Modeling"）endpoint loss 是用来稳定大 dt 跳跃的辅助监督。在我们的 target_normalize=true 设置下这个稳定作用被 σ² 抹平了。

**建议**：要么把 endpoint loss 也归一化为 $\|z_\text{pred} - z_d\|^2 / (\sigma_k\Delta t_k)^2$（这样就和 velocity 同尺度），要么承认 endpoint 是死代码并删掉。

### 🟠 问题 #4：α 调度的"心理预期"与实际梯度不符

**事实**（[rollout_first_hop.py L21-37](PET_LatentResidual/pet_lr/rollout_first_hop.py#L21)）：

```python
def mix_latent(z_gt, z_pred, alpha, straight_through=True):
    if alpha <= 0:
        return z_pred + (z_gt - z_pred).detach() if straight_through else z_gt
    if alpha >= 1:
        return z_pred
    mixed = (1.0 - alpha) * z_gt + alpha * z_pred
    if straight_through:
        return z_pred + (mixed - z_pred).detach()  # ← 关键
    return mixed
```

STE 的语义：**前向值 = mixed（含 GT 成分），反向梯度只穿过 z_pred**。

**结果**：当 α=0 时，前向 = z_gt（教师强制），但 ∂z_curr/∂z_pred = 1（链式梯度照样穿过）。

**心理预期 vs 实际**：

| 阶段 | yaml 注释心理预期 | 实际行为 |
|---|---|---|
| Phase I (0-50K, α=0, λ_r=0) | "纯 GT，无链式梯度" | λ_r=0 ⟹ rollout loss 没贡献，**确实无链式梯度**（但走了一遍前向计算+rollout step loss 的 detach 副本，浪费算力） |
| Phase II (50K-150K, α↑, λ_r↑) | "α 控制 student 自我曝光程度" | **α 不改变梯度路径**，只改变前向 z_curr 的数值。链式 BPTT 在 λ_r > 0 后立即激活到全部 4 hops |
| Phase III (>150K, α=1) | "全 student 自驱" | 前向也用 z_pred，与梯度路径终于一致 |

**问题在哪？** 这套机制本身不错（教师强制前向 + 学生反向是 Inverse Scheduled Sampling 的合理变体），但：

- yaml 注释写"Phase I 纯 GT（warmup_ratio=0.25 × 200K = 50K）"——读者会误以为 Phase I 没有链式梯度，但**实际上 λ_r=0 才是真正阻止链式梯度的开关**，与 α 无关
- 如果未来误以为"我可以单独调 α 来控制链式训练强度"——错。能控制的只有 λ_r
- **α 调度在 Phase I 完全无意义**（λ_r=0 时 rollout 整体不参与），这部分计算是浪费

**建议**：要么把 α 调度起点也设为 50K（与 λ_r 同步），要么在 Phase I 直接 `if lambda_roll == 0: skip rollout forward`。前者是修文档，后者是改代码（省 ~30% 训练时间）。

### 🟡 问题 #5：三项损失的标量量级悬殊（3750:500:1），无 GradNorm

**事实**：从 §20.2.2，pair_total : image_total : rollout_total = 1.5e-2 : 2e-3 : 4e-6 ≈ **3750 : 500 : 1**。

**为什么这本身不是 fatal**：

- 多任务损失的"梯度方向"可以差很大但量级悬殊不必然导致问题——只要每项梯度的有效信号都不被噪声淹没
- pair 信号方向 = "学好单步速度场"，rollout 方向 = "学好链式累积"，两者**不正交**——大量梯度方向相同
- AdamW 自适应步长部分抵消了量级差异（每个参数按自己的二阶矩缩放）

**为什么这仍然值得指出**：

- 如果按"标量数值占比"看任何监控盘（例如 `loss_balance_watch`），会**永久看到 pair_frac > 99%**——这对人类调参员形成误导，让人以为 rollout 完全无效
- 实际上 V6 step ~83K（即 Phase II 中段）才达到 best ckpt，说明 rollout 在 Phase II 起步阶段就有显著贡献——但从总 loss 数值上看不出来
- 没有 GradNorm/PCGrad/MGDA 等技术做梯度向量级动态平衡，纯靠静态权重 + AdamW 自适应

**建议**（不是必须做）：

- 加一个 diagnostic：每 1000 步记录三项损失对 backbone 末层的**梯度范数比**（不是损失数值比），写入 metrics.jsonl。这是**真实的多任务平衡指标**。10 行代码即可。

```python
# diagnostic only, no behavior change
for name, l in [("pair", pair_w*pair_total), ("roll", lam_r*roll_total), ("img", lam_i*loss_img)]:
    g = torch.autograd.grad(l, model.backbone.parameters(), retain_graph=True, allow_unused=True)
    g_norm = sum((gi.norm()**2 for gi in g if gi is not None))**0.5
    metrics[f"grad_norm_{name}"] = g_norm.item()
```

## §20.5 梯度平滑性总评估

### §20.5.1 不平滑源（按危害程度排序）

| 来源 | 影响 | 严重度 |
|---|---|---|
| velocity_rebalance 用 `.item()` 把 sqrt_ratio 转成标量，每 batch 重算 | 在 V6 因 clip 卡 1.0 → 实际无扰动 | 0（在 V6 被 clip 屏蔽） |
| `pair_loss_weights` 按 `.mean()` 而非 `.weighted_mean()` 应用 | batch 内 hop 比例随机，导致权重期望与实际有 ~30% 方差 | 中 |
| STE forward != backward | 标准 STE 偏差，理论上不收敛到真梯度 | 低（业界标准做法） |
| step_weights 同时承担量级补偿+chain importance | 两目标方向冲突时会震荡 | 中（问题 #1） |
| α 与 λ_r 的 ramp 同步但语义错位 | Phase II 前期过强的链式梯度可能导致不稳定 | 低 |
| grad_clip=0.5 全局裁剪 | 稳健的安全网 | 0（保护机制） |

### §20.5.2 平滑性正面因素

- **所有调度都是线性 ramp**：alpha、lambda_roll、lambda_img 没有突变
- **AdamW + weight_decay=0.01**：自适应学习率、L2 正则
- **EMA decay=0.9999**：参数空间的极强平滑
- **lr_schedule warmup_ratio=0.15**：30K 步学习率 warmup
- **freeze_rae=true**：decoder 不参与训练，避免 image_aux 梯度污染 latent 空间

### §20.5.3 总体判断

> **梯度流动整体是平滑的**，主要靠 AdamW + EMA + grad_clip + 线性调度的组合。上面问题 #1-#5 中**没有任何一项会导致梯度爆炸/消失/震荡**。但有 **3 个数学逻辑可改进**：
>
> 1. **rollout sigma normalization**（问题 #1）：是论文级 contribution，不是 bug fix
> 2. **velocity_rebalance 在 target_normalize=true 下死代码**（问题 #2）：清理掉省 1-2ms
> 3. **endpoint loss 在 target_normalize=true 下无信号**（问题 #3）：清理或归一化
>
> 问题 #4 是文档与实际语义不符（中等）；问题 #5 是监控盘误导（轻微）。
>
> **结论**：V6 的总 loss 公式**不是数学上的精品**，但工程上是 **work**。其"work"的数学根源是 §18-§19 推导的：pair velocity loss 提供主梯度方向、rollout 提供链式修正、image_aux 提供 hop0 锚定。三者的方向互补程度比量级比例所暗示的要高得多。

## §20.6 优先级行动清单（继 §19 的 1+2+3 之后）

| # | 行动 | 代价 | 产出 | 优先级 |
|---|---|---|---|---|
| 6 | 加梯度范数 diagnostic（§20.4 #5 末尾代码片段） | 30 min | metrics.jsonl 新增 3 列 | 高（论文 ablation 必备） |
| 7 | rollout step_loss σ-normalize 实验（§20.4 #1） | 1 次训练 | 验证 emergent 命中是否独立于量级补偿 | 高（论文 critical ablation） |
| 8 | 删除/修复 velocity_rebalance（§20.4 #2） | 改 yaml | -1ms/step | 低（不影响结果） |
| 9 | endpoint loss 归一化实验（§20.4 #3） | 1 次训练 | 验证 endpoint 是否真的无信号 | 中 |
| 10 | Phase I rollout skip-forward 优化（§20.4 #4） | 5 行代码 | -30% Phase I 时间 | 中 |
| 11 | **V6.1 Phase II 起势观测**（详见 §20.7） | gpu3 持续训练 +20K step | 验证 floor=0.05 设计是否在 Phase II 持续优于 V6 | 高（决定 V6.1 是否成 ablation 卖点） |

**对论文的关键启示**：§19 的"V6 emergent 命中多 hop Grönwall 闭式解" 必须在 §20.4 问题 #1 切割之后才能下定论。**即先做行动 7（σ-normalize 消融），再写 emergent 命中的论文章节**。否则匿名审稿人极可能问："你说命中闭式解，但你的 step_weights 也在补偿 σ²Δt² 量级——能不能控制量级变量后再展示？"——这是必杀问题。

---

## §20.7 V6.1 rollout-floor 模式诊断（2026-05-02 修订）

> **2026-05-02 修订**：本节的早期版本错误地把 V6 Phase III best (step 156K) 与 V6.1 Phase I (step 49K) 直接对比，得出"V6.1 失败"的结论。同步快照重新分析后**修正为：V6.1 在 Phase I 实测打败 V6**，"失败"判断撤回。下面用同步对齐的数据重新诊断。

### §20.7.1 同步对齐数据（V6 vs V6.1, step ≤ 50K）

来自 [v6_1_rollout_floor_gpu3_train_snapshot_20260502_014627.log](PET_LatentResidual/review/0428/operator/log_snapshots/v6_1_rollout_floor_gpu3_train_snapshot_20260502_014627.log) 与 V6 同期快照：

| Step | V6 chain_normal_mse | V6.1 chain_normal_mse | 比值 V6.1/V6 | 赢家 |
|------|---------------------|-----------------------|--------------|------|
| 2 000 | 0.000257 | 0.000257 | 1.000 | tie |
| 6 000 | 0.000224 | 0.000223 | 0.996 | V6.1 |
| 10 000 | 0.000242 | 0.000233 | 0.963 | V6.1 |
| 14 000 | 0.000623 | 0.000521 | 0.836 | V6.1 |
| 18 000 | 0.000502 | 0.000319 | 0.635 | **V6.1** |
| 22 000 | 0.000752 | 0.000419 | **0.557** | **V6.1** |
| 26 000 | 0.001343 | 0.000889 | 0.662 | V6.1 |
| 30 000 | 0.001132 | 0.000935 | 0.826 | V6.1 |
| 34 000 | 0.000751 | 0.000581 | 0.774 | V6.1 |
| 38 000 | 0.000975 | 0.000708 | 0.726 | V6.1 |
| 42 000 | 0.000512 | 0.000434 | 0.848 | V6.1 |
| 46 000 | 0.000424 | 0.000364 | 0.858 | V6.1 |
| 50 000 | 0.000423 | 0.000596 | 1.409 | V6 |

**V6.1 head-to-head 胜率：11/13 = 85%**。V6.1 在 V6 出现 U 型膨胀的 step 22K-30K 区间把恶化压到 V6 的 56-83%——floor=0.05 的 chain anchoring 起作用。

### §20.7.2 Phase I best 对比

| 指标 | V6 Phase I | V6.1 Phase I | V6.1 优势 |
|------|-----------|--------------|----------|
| best chain_normal_mse | 0.000224 @ step 6 000 | **0.000213 @ step 12 000** | −4.9% |
| best val_select | 0.000772 @ step 6 000 | **0.000762 @ step 12 000** | −1.3% |
| best 出现步数 | 6 000 | 12 000 | V6.1 收敛更慢 |
| best 之后停滞步数（截至当前） | 62 000（被 step 68 000 超越） | 37 100（仍在停滞） | V6.1 仅 V6 当年 60% |

**关键事实**：V6 自己的 Phase I best 停滞了 62 000 步才被 Phase II 中段（step 68 000）超越。V6.1 当前停滞 37 100 步**完全在正常范围内**——它甚至还没进 Phase II（V6 的 Phase II 在 step 50K 附近开始）。

### §20.7.3 floor=0.05 的实际效果验证

数学预测：在 λ_roll=0.05、L_r/L_p ≈ 3.8 假设下，
$$
\text{expected roll\_frac} = \frac{0.05 \times L_r}{15 \times L_p + 0.05 \times L_r + \lambda_\text{img} \times L_i} = \frac{0.19 \cdot L_p}{15 \cdot L_p + 0.19 \cdot L_p + \text{img}} \approx 1.2\%
$$

实测（982 个 train events）：

| 指标 | 实测值 | 预测/参考 |
|------|--------|----------|
| roll_frac mean | **1.24%** | 1.2% （±5%）✅ |
| roll_frac median | 0.80% | — |
| roll_frac P95 | 3.40% | — |
| roll_frac max | 18.9% | （batch 级 L_p 偶发极小时）|
| % steps with roll_frac > 1% | 38.2% | — |

**结论**：floor=0.05 没有失效。它就是设计来产生 ~1% chain anchoring signal 的——既不主导 loss（避免训练不稳），也不归零（防止 chain composition 漂移）。

### §20.7.4 pair_frac 摆动是 Phase I 结构性现象

| 占比 | V6 Phase I | V6.1 Phase I | 差异 |
|------|-----------|--------------|------|
| pair_frac min/max/std | 0.029 / 0.996 / **0.204** | 0.023 / 0.993 / **0.206** | ~相同 |
| roll_frac min/max/std | 0.000 / 0.000 / 0.000 | 0.002 / 0.189 / 0.013 | V6.1 微弱 |
| img_frac min/max/std | 0.004 / 0.971 / 0.204 | 0.004 / 0.935 / 0.198 | ~相同 |

V6 与 V6.1 的 pair/img 摆动幅度**几乎完全相同**——这是 pair-only / pair-dominant 训练下 batch 级别的固有现象（latent 距离小的 batch L_p 极小、img_aux 主导；距离大的 batch 反过来），**不是 V6.1 特有**。

### §20.7.5 修正后的判定与下一步

**当前不能下结论"V6.1 失败"**。正确的判定窗口：

| 窗口 | 关键指标 | 判定 |
|------|---------|------|
| Phase II 起势（V6.1 step 50K → 70K） | chain_mse 改善幅度 | V6 同期改善 54%（0.000423 → 0.000196）。V6.1 应至少同等。 |
| Phase II 中后段（V6.1 step 100K → 130K） | best chain_mse 突破 Phase I best？ | 应突破 V6.1 自己的 Phase I best 0.000213 |
| Phase III 收敛（V6.1 step 150K+） | best val_select 与 V6 比 | V6 Phase III best 0.000610。V6.1 应在 ±10% 内或更优 |

**当前行动建议**（替换早期"终止 V6.1"提议）：

| 选项 | 操作 | 理由 |
|------|------|------|
| **A. 继续观察 V6.1**（推荐） | gpu3 保持训练，等到 Phase II 起势（约 step 60K-70K） | floor=0.05 设计验证还需 ~10-20K step 才能定论 |
| B. 同时做 σ-normalize ablation | gpu1 V6 训练完成（接近 ckpt 选优）后，把 gpu1 转向 §20.6 行动 7 | 不挤占 V6.1 |
| ~~C. 终止 V6.1~~ | ~~弃用~~ | **数据不支持，撤回此建议** |

**真正值得监测的指标**：不是当前 chain_mse 振荡（Phase I 必然现象），而是 **V6.1 step 50K-70K 的改善速度**。如果 V6.1 在 Phase II 起势速度 ≥ V6（54% 改善幅度），floor=0.05 设计完整成立，可成为 ablation 卖点（"全程保持 chain anchoring vs Phase I 完全关 rollout"）；如果显著慢于 V6，再考虑改 floor 值或终止。

---

## §21 独立架构审视的关键发现（2026-05-02）

> 本节是 2026-05-02 一次完整独立代码审视 + 7 道关键问题与用户对答的综合记录。每条都有用户确认 / 反驳，状态清晰。这些发现**直接影响论文 contribution 措辞和必做实验**——比 §0-§20 任何单独章节都更重要，请优先阅读。
>
> **2026-05-02 二次审计补充**：§21.11 记录外部 agent 在 §21 写完后发现的 5 处细节修正——全部被代码核实。读者应**同时读 §21.1-§21.10 + §21.11**，两者叠加才是最终状态。

### §21.1 数学错误：§18.3.4 off-by-one bug（已修正）

| 项 | 状态 |
|---|------|
| 错误 | 早期 §18.3.4 表的 β-sensitivity 求和公式排除了 β_{j+1}，得到 [2.85, 2.4, 1.5, 0] |
| 正确 | 求和应包含 β_{j+1}（step_j 直接产生 hop j+1 的输出 z_{j+1}^pred），得到 [3.35, 2.85, 2.4, 1.5] |
| 用户确认 | "doc §18.3.4 是 off-by-one bug，需要重写" |
| 修正位置 | §18.3.4（已就地修正） |
| 修正后 emergent 命中评分 | hop2/3 接近（4-11%）；hop1 勉强（18%）；**hop0 反向（V6 0.25 vs 闭式 1.0，差 4×）**；L1 总差 0.985 |

### §21.2 论证错误：pair_loss_weights[0]=2.5 不是"补偿 hop0 step_w 偏低"（已修正）

| 项 | 状态 |
|---|------|
| 早期叙事 | hop0 step_weight=0.5 anomaly 由 pair_loss_weights[0]=2.5 显式补偿 |
| 用户反驳 | "已知它们在 hop0 不真正分工" |
| 真实动机 | **hop0 有 pixel forcing 依赖（gate_pix + pixel encoder），需要额外训练信号驱动 gate 生长，避免 gate 死在 floor**——即 pair[0]=2.5 是激活特定架构组件的设计，不是 Grönwall 补偿 |
| 修正位置 | §18.3.4 修正 2 段（已就地修正） |

### §21.3 替代解释浮出：σ²·dt² 量级补偿主导（待 ablation 验证）

V6 step_weights = [0.5, 2.0, 1.5, 1.0] 实际同时在做两件事：

1. **补偿 σ²·dt² 自然量级**（rollout MSE 跨 hop 量级差 7.5×）
2. （可能）附加一个弱 Grönwall β-加权

但因为 §17.1 反解的 L_j ≈ [0.96, 1.13]——几乎 = 1，Grönwall 加权效应在我们的 regime 下本身就微弱。**主导效应大概率是 σ²·dt² 补偿**，不是 Grönwall。

| V6 step_weights × (σ_j·dt_j)² normalized | hop0 | hop1 | hop2 | hop3 |
|---|---|---|---|---|
| 有效贡献 | 0.96 | 1.0 | 0.47 | 0.26 |

→ 大致单调递减，与 NORMAL 末端为最重要 ckpt 的临床先验一致。

**用户确认**："σ-normalize ablation 必须做"。这个 ablation 是判定 emergent contribution 强弱的 critical experiment：

| ablation 设置 | rollout step_loss | step_weights 应该重训出什么？ | contribution 强度 |
|--------------|-------------------|-------------------------------|------------------|
| V6 baseline（当前） | raw MSE，自然量级 (σ_j·dt_j)² 主导 | [0.5, 2.0, 1.5, 1.0]（已知 emergent） | 暧昧（量级 + Grönwall 混在一起） |
| σ-normalize ablation（待做） | step_loss / (σ_j·dt_j)²，去除自然量级 | 如果仍 emerge [0.5, 2.0, 1.5, 1.0]：Grönwall 是真原因 → 强 contribution；如果 emerge [1, 1, 1, 1] 附近：σ²dt² 是主因 → contribution 降级为"自动量级补偿" | 决定性 |

### §21.4 chain MSE 横比警告（已修正 §0）

V6 实测 chain_d20 → chain_normal 数值单调递减（0.000301 → 0.000229）。早期把这解读为"模型在末端 hop 表现更好"。**用户确认**：σ_j 沿链衰减 68×（[0.0096, 0.0029, 0.00078, 0.000141]），target 量级本身在缩小，**chain MSE 跨 hop 数值差异主要是 σ_j 衰减驱动**，不是模型主导。

| 推论 |
|------|
| §0 状态表的 D20 mse 与 NORMAL mse **不可直接横比** |
| 真正可对比的指标是 σ-normalized chain MSE = chain_k_mse / σ_k² |
| §17.3 反解 L_j ∈ [0.96, 1.13] 仍成立（这条用的是 chain MSE 比值，不依赖横比） |
| 修正位置 | §0 状态表后已加 caveat（2026-05-02 增补） |

### §21.5 hop_residual / pixel forcing 的低激活是设计意图（已修正 §0 命题）

> **2026-05-02 二次修订（外部 agent 审计）**：早期版本写"lambda_hop_0 从 init=0.001 长大 60×"——init 数值错。当前 V6 yaml 的 `lambda_hop_init: 0.10`、`pixel_gate_init: 0.02`，**0.001 是 floor 不是 init**。所以 lambda_hop_0=0.0589 实际是从 init=0.10 **轻微下降**到 0.06；gate_pix=0.0374 从 init=0.02 长大不到 2×。"低激活 by design" 结论仍成立，但增长叙事需要重写。

V6 step 161,450 train 行实测（核对当前 V6 yaml `pet_flow_first_hop_224_v6_transport_first.yaml`）：

```
lambda_hop_0 = 0.0589  （init=0.10, floor=0.001 → 比 init 略低，未自发降到 floor 也未爆涨）
gate_pix     = 0.0374  （init=0.02, floor=0.001 → 长大不到 2×）
v_hop_abs    = 0.000238 （hop_residual head 输出量级，相对 v_total ≪ 1%）
pix_delta_abs= 0.0     （当前 step 没有 hop0 batch 触发 pixel forcing）
```

→ hop_residual / pixel forcing 在 V6 跑了 161K 步后**始终处于"低激活、不爆涨也不死"状态**——这是 yaml 用 init=0.10 / 0.02（不在 floor 上）+ softplus + l2 衰减组合得到的稳定低激活，符合 "safety net" 设计语义。

| 早期质疑 | 用户回答 |
|---------|---------|
| 这两条分支是否 dead？是不是架构 over-engineered？ | "设计就是低激活——hop_residual 是 '安全网' 不是主力" |

→ 命题 2/3 的"加回主干 + 像素 forcing"应理解为**小幅修正**（safety net），不是与 backbone 等量的主力贡献。已修正 §0 命题描述。

**论文措辞建议**：

> "The hop residual heads and Hop-0 pixel forcing branch act as **safety-net modules** with low effective activation (λ_hop_0 ≈ 0.06 at convergence). They provide bounded local correction without taking over the backbone's learning signal—this is by design, ensuring backbone-shared representation dominates while leaving room for hop-specific adjustment when needed."

### §21.6 Phase I STE-rollout 的独立训练价值（已确认）

mix_latent 函数在 α=0 时返回 `z_pred + (z_gt - z_pred).detach()`——forward = z_gt（teacher forcing），backward 通过 z_pred 链。

| 早期质疑 | 用户回答 |
|---------|---------|
| Phase I STE-rollout 是否冗余 pair loss？两者在 z_GT 输入下做同一件事？ | "有独立价值——chain composition 的 Jacobian 在 STE 下还是经过 z_pred 链" |

→ Phase I rollout 提供的 chain composition Jacobian 信号 ≠ pair loss（pair 只看单步），即使 forward 路径相同，反向梯度图不同。V6.1 的 floor=0.05 在 Phase I 给 chain anchoring 是**有效的设计**，不是冗余。

### §21.7 β=[0.5, 0.45, 0.9, 1.5] 来源验证（已确认非 tautology）

val_select 的 β 来自 yaml `best_metric_terms` 的 weights：[chain_d20: 0.50, chain_d10: 0.45, chain_d4: 0.90, chain_normal: 1.50]。

| 早期质疑 | 用户回答 |
|---------|---------|
| 如果 β 是从 V6 训练曲线反推出来的，那"emergent 命中"就是 tautology | "β 早于 V6 step_weights，基于临床先验（NORMAL 最重要）" |

→ β 是**先验**（基于临床上 NORMAL 剂量是最终诊断价值最高的图），不是从训练曲线反推。emergent 论证**非 tautology**——但仍受 §21.3（σ²·dt² 量级补偿主导）削弱。

### §21.8 综合 contribution 强度评估

| 论点 | 早期声称 | 修正后 |
|------|---------|--------|
| V6 step_weights emergent 命中多 hop Grönwall 闭式解 | 强（hop1/2/3 几乎完全一致） | **降级**：hop2/3 接近，hop0/1 偏离；σ²dt² 量级补偿主导，Grönwall 在 L≈1 下本身微弱 |
| pair[0]=2.5 与 step_w[0]=0.5 是 Grönwall 通道分工 | 显式补偿 | **错误**——pair[0]=2.5 是 pixel forcing gate 驱动信号 |
| Hop-Residual + pixel forcing 是主力分支 | 与 backbone 等量贡献 | **降级**——safety net 设计，低激活 by design |
| Phase I STE 提供独立 chain anchoring 信号 | 假设成立 | **确认成立**——chain composition Jacobian ≠ pair loss |
| β=[0.5, 0.45, 0.9, 1.5] 是 V6 训练后反推的 | 早期未严格区分 | **确认**：β 是临床先验，先于 V6 step_weights，emergent 论证非 tautology |

### §21.9 论文 critical experiments 列表（更新）

按重要性降序：

| # | 实验 | 状态 | 关键判定 |
|---|------|------|---------|
| 1 | **σ-normalize ablation**（§21.3） | **必做，未做** | step_weights 在去除 σ²dt² 自然量级后是否仍命中 [0.5, 2.0, 1.5, 1.0] |
| 2 | β-sensitivity sweep | 未做 | 改 β=[1, 1, 1, 1]（中性临床先验）训练，看 step_weights 是否仍 emerge 同样模式——验证 β 与 step_weights 的耦合 |
| 3 | hop_residual ablation | 未做 | 移除 hop_residual heads（只留 backbone）训练，量化 safety net 贡献 |
| 4 | V6.1 完成训练 | 进行中 | Phase II 起势速度（§20.7.5） |
| 5 | EMA 收敛稳定化 | 未做 | Phase III val 振荡 max/min 2.57×，需要 EMA 平滑 |

### §21.10 §0-§20 章节状态修订

| 章节 | 之前状态 | 修订后 |
|------|---------|--------|
| §0 状态表 | AUTHORITATIVE | AUTHORITATIVE + 增加 chain MSE 横比 caveat（已加） |
| §0 命题 2/3 | AUTHORITATIVE | 措辞修订为"safety net"低激活（已加） |
| §17.1 反解 L | AUTHORITATIVE | 仍成立（不依赖横比） |
| §17.3 反解 L | AUTHORITATIVE | 仍成立 |
| §18.3.4 闭式解 vs V6 | AUTHORITATIVE | **修订**：off-by-one 已修，emergent 命中降级（已加） |
| §18.3.4 pair 通道补偿论证 | AUTHORITATIVE | **撤销**：pair[0]=2.5 与 hop0 Grönwall 无关（已加） |
| §19 论文 narrative | DRAFT | **必须重写**：contribution 强度按 §21.8 降级 |
| §20.6 行动 7 σ-normalize | 未来工作 | **升级为 critical experiment**（§21.3） |
| §20.7 V6.1 判定 | AUTHORITATIVE | 仍成立 |

### §21.11 外部 agent 二次审计补充修正（2026-05-02）

> 外部 agent 在 §21 写完后做了二次审计，并核对了当前 V6 yaml + 训练代码。它捕获了 5 处 §21 没说清楚或写错的细节。**全部 5 处已被代码确认**，下面逐条记录。

#### §21.11.1 pair endpoint vs rollout step_loss 的量级区分（澄清，§21.3 不变但需明示）

**事实核对**（`train_first_hop.py` L311 + `model_first_hop.py` L329-356）：

| 通道 | 公式 | 在 `target_normalize=true` + `endpoint_dt_normalize=true` 下展开 | 量级 |
|------|------|-----------------------------------------------|------|
| pair velocity loss | `(v_total_raw - v_target_norm)²`，其中 v_target_norm = (z_dst-z_src)/(σ·dt) | 直接是 normalized velocity 残差² | $O(1)$ |
| pair endpoint loss | `((z_pred - z_dst) / dt)²`，z_pred = z_src + σ·dt·v_total_raw | = $\sigma^2 \cdot (\text{v\_err})^2$ | **$\sigma^2$** |
| rollout step_loss | `(z_pred - z_dst)²`（不除 dt） | = $\sigma^2 \cdot dt^2 \cdot (\text{v\_err})^2$ | **$\sigma^2 \cdot dt^2$** |

→ pair endpoint 与 rollout step_loss 在量级上**差一个 $dt^2$ 因子**（hop0 dt=3 → 9×；hop3 dt=75 → 5625×）。早期某些版本（外部 agent 称为 agent1）把"endpoint 缺 $dt^2$"过度推广到 active V6——**这个推广不成立**：`endpoint_dt_normalize=true` 显式抵消了 dt²。

§21.3 表给的"自然量级 $(\sigma \cdot dt)^2$"特指 **rollout step_loss**，不是 pair endpoint。这一区分以前没说透，**论文中谈"V6 step_weights 在补偿 $\sigma^2 \cdot dt^2$"必须明确指 rollout 通道**，避免读者套到 pair 通道上。

#### §21.11.2 lambda_hop_init / pixel_gate_init 的当前值（§21.5 已修正）

**事实核对**（`pet_flow_first_hop_224_v6_transport_first.yaml` L269-272）：

```yaml
pixel_gate_init: 0.02
pixel_gate_floor: 0.001
lambda_hop_init: 0.10
lambda_hop_floor: 0.001
```

**早期 §21.5 写法**："lambda_hop_0 从 init=0.001 长大 60×"——**错误**。0.001 是 floor，不是 init。当前 V6 init 是 0.10，所以观察到 lambda_hop_0=0.0589 实际是从 0.10 **轻微下降**（softplus + l2 衰减组合），不是从 floor 长大。

修正后的解读：lambda_hop_0 / gate_pix 在训练全过程中**稳定保持在低激活区域附近**（既不爆涨也不衰减到 floor），这与 "safety net" 设计语义完全一致。§21.5 已用代码核对值替换。

#### §21.11.3 best_metric_terms 的 β 是 ckpt 选择目标，不是 SGD 反传权重（重要口径）

**事实核对**（`train_first_hop.py` L1041-1084 `resolve_best_selection_score`）：

* `best_metric_terms` 在 `resolve_best_selection_score` 里只接受 `metrics: Dict[str, float]`（**Python 浮点数**，不是 torch tensor）
* 函数返回 `(score, name)` 用于 ckpt selection（哪个 ckpt 文件保存为 `best.pt`）
* 它**不进入** SGD 反传图

这意味着：

| 角色 | β 的真实身份 | 文档措辞 |
|------|-------------|---------|
| **SGD 真正最小化的 loss** | `pair_loss × pair_loss_weights[hop] + step_loss × step_weights[hop] + image_aux × λ_img` | 这是反传梯度 |
| **β=[0.5, 0.45, 0.9, 1.5]** | **ckpt selection / clinical-prior validation objective** | 不是反传梯度 |

→ §18.3.3 多 hop Grönwall 闭式解推导**仍然有意义**，但其叙事必须重新定位为：

> "**Given a clinical-prior weighted validation objective $J = \sum_k \beta_k \cdot \|c_k\|^2$**, the optimal training step_weights (in terms of minimizing $J$ at validation) would be $w_j^* \propto \sum_{k=j+1}^K \beta_k \prod L_i^2$. We empirically observe V6's hand-tuned step_weights match this prediction in hops 2-3."

→ "V6 step_weights 是 SGD loss 的最优反传权重"——**不是这个声明**；
→ "V6 step_weights 让训练显式最小化的 loss，与某个 clinical-prior 验证目标的最优配比经验吻合"——**这是正确声明**。

这影响 §19.5 论文 contribution 措辞和 §21.8 的"对论文的影响"段落，需要按上述 framing 重写。

#### §21.11.4 "step_weights emerge" 措辞修正

step_weights = [0.5, 2.0, 1.5, 1.0] 是 yaml **手工配置**（V6 commit 时手调），**不是训练学出来的**。所以"emerge"这个词在英文论文里会误导审稿人理解为 learned。正确措辞：

| ❌ 错误 | ✅ 正确 |
|---------|--------|
| "V6 step_weights emerge to closed-form" | "V6's hand-tuned step_weights match closed-form prediction" |
| "step_weights 自发命中" | "step_weights 经验调优后与闭式解吻合" |
| "emergent 命中" | "post-hoc theoretical alignment" / "经验匹配理论" |

σ-normalize ablation 的描述也应改为：**"在 σ-normalized rollout 下重新搜索最优 step_weights，看是否仍收敛到 [0.5, 2.0, 1.5, 1.0] 附近"**——是搜索/sweep，不是"看是否 emerge"。

#### §21.11.5 chain MSE 跨 hop 归一化的正确量级（§21.4 增补）

§21.4 写"σ-normalized chain MSE = chain_k_mse / σ_k²"——**部分正确，需要细化**。

* 如果对比的是 **chain rollout step error**（z_k^pred - z_k^GT），自然量级是 $(\sigma_k \cdot dt_k)^2$，应除 $(\sigma_k \cdot dt_k)^2$
* 如果对比的是 **target latent variance**（z_k^GT 自身的方差），可用 empirical target variance normalize

更稳妥的做法：**同时报告**

1. raw chain MSE
2. $(\sigma_k \cdot dt_k)^2$-normalized
3. empirical target variance-normalized

→ §0 状态表的 caveat 应更新为"使用 $(\sigma_k \cdot dt_k)^2$ 或 target variance 归一化后再横比"，不是只除 $\sigma^2$。

#### §21.11 修订汇总

| # | 修正内容 | 影响 § | 状态 |
|---|---------|-------|------|
| 1 | pair endpoint 是 $\sigma^2 \cdot v\_err^2$（不是 $\sigma^2 \cdot dt^2 \cdot v\_err^2$） | §21.3 | 已澄清（本节 §21.11.1） |
| 2 | lambda_hop_init=0.10 / pixel_gate_init=0.02（不是 0.001） | §21.5 | 已 inline 修正 |
| 3 | β 是 ckpt selection / val objective，不是 SGD loss | §18.3 / §21.8 / §19.5 | 待重写 narrative |
| 4 | "emerge" 措辞改"hand-tuned + post-hoc match" | §19 / §21.8 / 论文 | 待批量替换 |
| 5 | chain MSE 归一化用 $(\sigma \cdot dt)^2$ 或 target variance | §0 / §21.4 | 待更新 caveat |

**论文级影响**：此 5 条不改变 §21 主结论，但强化 §21.9 行动 1（σ-normalize ablation）的必要性，并把 §19 的论文 contribution narrative 推向更严谨的"empirical match to closed-form derived from clinical-prior validation objective"——而不是"emergent learned solution"。




