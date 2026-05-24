# Round 18 — A4 Bracket Result Analysis + Strategic Re-evaluation

- date: 2026-05-25
- branch: foc_lite_hop0 (commit f2206cc)
- substrate: A4 bracket {λ=0.02, λ=0.08} full-val canonical eval 已完成
- 目的: 在 A4-mid 出现项目最大单实验信号 (+0.113 dB, 280× V14 noise) 后, 重审整个研究线; 给出 **3-4 个真正能改变量级的方向**, 不再调小参

---

## 1. 全项目实验全景 (canonical full-val PSNR_clip3, n=7403)

| 实验 | image_aux λ | 其它关键变量 | NORMAL | Δ vs V7 | Δ vs V14 noise (×) |
|---|---:|---|---:|---:|---:|
| V6_NOISE.best | 0.04 | V6 step_weights, seed=1337 | 36.7437 | −0.0373 | −93× |
| V8.best | 0.00 | Grönwall step_weights + train config 改 | 36.4729 | −0.3081 | −770× |
| V13.best | 0.00 | 真 image_aux off, V7 train config | 36.4943 | −0.2867 | −717× |
| **A4-low.best** | **0.02** | V7 base | **36.7010** | **−0.0800** | **−200×** |
| V7.best | 0.04 | Grönwall raw | 36.7810 | 0 | 0 |
| V14.best | 0.04 | V7 + seed=1337 | 36.7806 | −0.0004 | (noise floor) |
| V18.best | 0.04 | + decoder LoRA r32 + KL pullback | 36.8112 | +0.0302 | +75× |
| V18.last | 0.04 | (上 + 训到 200K) | 36.8426 | +0.0617 | +154× |
| **A4-mid.best** | **0.08** | V7 base | **36.8939** | **+0.1130** | **+283×** |

**A4-mid 是项目历史上最大单实验改进**, 超过 V18 整套 decoder LoRA + KL 工程 (+0.062 dB) 接近 2 倍. 它的代价是 0 行代码改动, 0 个新模块, 仅 1 个数值 (0.04 → 0.08).

---

## 2. A4 结果的 4 个非平凡含义

### 2.1 image_aux 远未饱和; λ=0.04 是历史偶然不是 sweet spot

四点 [0, 0.02, 0.04, 0.08] 在 NORMAL 上**单调递增**, 没看到拐点. λ=0.04 这个项目用了 ~1 年的默认值不是经过 sweep 选出的, 只是 V6 沿用过来的, 现在被证伪.

- λ=0.16? λ=0.32? 谁也不知道
- 但 user 已经禁止"再调小参", 所以不能简单做 λ sweep — 这是死路

### 2.2 image_aux 是**真正的 main lever**, 其它都是噪声级

把 A4-mid 信号当尺子, 重看历史:

| 工程 | 投入 | 收益 (NORMAL dB) | 与 A4-mid (+0.113) 比 |
|---|---|---:|---|
| Grönwall step_weights (V7) | 数月 train recipe 设计 | ~0 (V13-V8 = +0.02 in noise) | ~0% |
| Decoder LoRA + KL (V18.best) | 6 周设计 + 训练 + 7 轮 review | +0.030 | 27% |
| Decoder LoRA 额外 35K 训练 (V18.last) | 7d GPU | +0.032 | 28% |
| Decoder capacity (A3 vs V18) | 7d GPU + KL 否定 | 0 (capacity 可解释 V18) | 0% |
| **image_aux 0.04 → 0.08** | **1 yaml 字段, 0 代码** | **+0.113** | **100%** |

V18 整套 decoder LoRA + KL story 现在的相对地位是 **A4-mid 信号的 27%**. paper 主图必然以 image_aux 为头条, V18 降为"我们也试过 decoder LoRA, 收益小于 image_aux schedule 调整"的 ablation.

### 2.3 image_aux 为什么 work 没人知道

`image_aux` 实现是 `l1(decode(z_pred), x_target) + 0.25 * ssim + 0.1 * seam`. 它把 pixel-space supervision 从 frozen decoder 反传回 transport latent. 候选机制:

- **M1**: transport 在 latent 空间 undertrained, pixel-space gradient 提供更强信号 (类似 GAN 的 perceptual loss vs MSE)
- **M2**: SSIM 子项捕获 structure, 而 latent MSE 只对 magnitude 敏感
- **M3**: seam 子项防 patch boundary artifact, 改善 chain 串行 decode 时的累积 error
- **M4**: image_aux 起到隐式 regularizer 作用, 防 z_pred 漂离 decoder 可识别 manifold (机制上类似 A3 LoRA capacity 但反向: 不是 decoder 改, 是 transport 被 anchor 到 decoder)

**没有任何当前实验能告诉我们哪个机制真**. 这是 paper 主结果背后的黑箱.

### 2.4 V18 ablation 现在彻底变小

V18 用 image_aux λ=0.04 + decoder LoRA + KL. 如果换 image_aux λ=0.08 + 同 LoRA + 同 KL, 三种可能:

- (a) +0.113 + 0.030 = +0.143 dB additive → V18 仍有价值
- (b) ≤ +0.113 (decoder LoRA 收益被 image_aux 吃掉) → V18 整体冗余
- (c) 不可加 (decoder LoRA 与高 image_aux 梯度冲突) → V18 在新 image_aux schedule 下 worse

**没有实验告诉我们哪个真**.

---

## 3. 战略空间重排 (user 约束: 不再小打小闹)

### 3.1 应该被排除的"调参"路径

按 user "不想是调调参数之类的小打小闹" 指令排除:

- ❌ **λ sweep**: 跑 λ=0.06 / 0.10 / 0.12 / 0.16 找 sweet spot — 这正是 user 拒绝的
- ❌ **V14b/c/d 多 seed 加强 noise floor**: 已知 V14 |Δ|=0.0004, 加 seed 信号不变
- ❌ **A4 + decoder LoRA 简单组合**: 仍是 yaml 字段调整, 不是新设计
- ❌ **image_aux 内部权重调** (l1_weight / ssim_weight / seam_weight): 一样是小参数

### 3.2 真正的 substantive 方向 (5 选项, 详细论证后让 reviewer 选)

#### Option X1 — 机制深拆: image_aux 子项 ablation + activation analysis

**问题**: image_aux 工作机制未知; paper 主结果如果只是"加大 λ 涨 +0.113 dB"会被 reviewer 当 trivial 拒.

**实验**:
- X1a: 三个 λ=0.08 ablation runs: l1-only, ssim-only, seam-only (各 1 seed, 7 天 × 3 = 21d GPU)
- X1b: 中间 activation probe — 看 transport 输出 z_pred 在 image_aux on/off 下 latent distribution 差异 (~1 GPU day)
- X1c: 写 paper 主线 = "pixel-space supervision via frozen decoder is the dominant signal for PET latent transport; SSIM/L1/seam contribute X/Y/Z fractions"

**EV**: 高. 转化 +0.113 dB 现象为 publishable mechanism story.
**风险**: X1a 是 3 runs, 仍有"调子项"味道; 但每个 run 是独立 ablation 不是 sweep, paper-defensible.

#### Option X2 — 架构升级: pixel-aware transport

**问题**: 如果 image_aux 这么强, 说明 transport 的 latent loss 太弱; 应该让 transport **架构上**就看到 pixel-space, 而不是靠 auxiliary.

**实验**:
- X2a: 替换 transport DiT → consistency model / shortcut model / flow matching with stochastic interpolant — 这些架构对 pixel-space supervision 自带 native gradient path
- X2b: pixel decoder + latent transport 联合训练 (代替 frozen decoder + image_aux)
- X2c: 论文叙事 "pure latent transport is fundamentally weak for medical imaging; pixel-aware transport architectures yield K× improvement"

**EV**: 中-高. 如果 X2 出 +0.3+ dB, 是 MICCAI/TMI 主刊级结果.
**风险**: 4-8 周设计 + 训练, 是新项目级开销; 失败率 ~50%.

#### Option X3 — 端到端联合优化: 不冻结 decoder

**问题**: V18 已经在做 "decoder LoRA + KL" 但用错路径 (KL 走 z_pred, capacity 无 chain transfer). 真正的 end-to-end 是把 image_aux 当主 loss, decoder LoRA 当 capacity 缓冲.

**实验**:
- X3a: V18-like 但 lambda_kl=0, lambda_img=0.08, LoRA on (rank 32, last 2 blocks). 测 A4-mid + decoder LoRA additive
- X3b: 同上但 unfreeze decoder 全部 last 4 blocks (更激进 capacity)
- X3c: 比 X2 便宜, 是 V18 framework 修正版而非新架构

**EV**: 中. 如果 additive, V18 family 救活; 如果不 additive, paper narrative 简化 (image_aux only).
**风险**: 仍在 V18 框架内, 可能被 reviewer 看作 "微改的 V19" — 视设计而定.

#### Option X4 — Multi-step refinement / chain redesign

**问题**: 当前 first_hop 是 hop_0 only forward. chain 是 D50 → D20 → D10 → D4 → NORMAL 4 步. 是否 chain step 不够 / 不对?

**实验**:
- X4a: 加 second hop, 双 hop forward; image_aux 同步加到两 hop
- X4b: chain inference-time refinement — 不改训练, 推理时迭代多次 chain rollout
- X4c: chain dose schedule 重排 (e.g. logarithmic instead of fixed timepoints)

**EV**: 不确定. 历史上 hop_0 only 是设计选择, second_hop 没尝试.
**风险**: 改 chain 几何会动 V13/V14/V18 baseline 可比性, 现有数据点失去对照价值.

#### Option X5 — Cross-anatomy / cross-tracer generalization (paper 加分项)

**问题**: 所有当前实验在一个 PET 数据集. paper claim "image_aux works for PET" 是单数据集声明.

**实验**:
- X5a: 在另一个 PET 数据集 (如果可获取) 重跑 V13 + A4-mid + V18, 看 image_aux 主导性是否泛化
- X5b: 在 CT-PET 联合 / cross-tracer (FDG / DOTATATE / PSMA) 上验证

**EV**: 高 if 有数据. 把 single-dataset paper 升级为 generalizable claim.
**风险**: 数据获取本身可能 4-8 周, 出 Round 17 范围.

---

## 4. 当前已知 limitation (paper 必须承认的)

1. **Single-seed**: V14 已证伪"seed 影响 < 0.001 dB", 但所有实验仍 single-seed.
2. **Slice-level inference**: patient grouping 在 preprocessing 阶段不可恢复 (user 已确认), paper 不能 claim patient-level p-value.
3. **Single PET dataset**: 无 cross-dataset / cross-tracer 验证.
4. **image_aux mechanism unknown**: 黑箱.
5. **未尝试架构变化**: 整个项目都是 transport DiT + frozen RAE decoder framework.
6. **V18-clean (use_pred_latent=false) 未跑**: B42 之后 Round 16 decision 撤; 在 image_aux 主导后是否值得复活, 待定.

---

## 5. 我的初步建议 (待 reviewer 验证)

按 EV / cost / paper-impact 排序:

1. **X1 (机制深拆)** — 必做. 否则 paper 没有 mechanism story 只有 "tune λ" trick.
2. **X3 (image_aux + LoRA additive 测试)** — 应做. 1 个新 run, 决定 V18 是否冗余, 7d 内出.
3. **X2 (架构升级)** — 探索性, 看 reviewer 是否觉得值得开新项目分支.
4. **X5 (cross-dataset)** — 如有数据则做; 是 paper 终态 vs 投递档次的关键.
5. **X4 (chain redesign)** — 风险高, 收益不确定, 优先级低.

但 user 已说"不想小打小闹", 因此 X1 + X3 这种 1-3 run incremental 路径可能仍被 user 拒. 真正符合 user 期待的是 **X2 (architecture pivot) + X5 (data expansion)**.

---

## 6. 给 reviewer 的核心问题 (peer review prompt 主轴)

1. A4-mid +0.113 dB 真的是项目重心? 还是个 single-seed 假信号?
2. image_aux mechanism (M1-M4) 哪个候选最可能真? 需要哪个最小 experiment 区分?
3. user "不调小参" 约束下, X1-X5 哪 1-2 个真值得 ≥ 7d GPU + 设计投入?
4. paper narrative 现在应不应该立刻撤下 V18 主线, 改写为 image_aux 主线 + V18 ablation?
5. 是否漏关键方向 X6+?
