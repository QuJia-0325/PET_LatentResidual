# 10 dB Oracle Gap — Mechanism Analysis & Breakthrough Roadmap

Date: 2026-04-24  
Author: Nightmare-mode synthesis (post-Round-5)  
Context: Follow-up to `review/0424/nightmare/05_final_consensus_and_roadmap.md`  
Question answered: "10 dB oracle 差距无机制解释，无突破路径" — **修正**

---

## TL;DR

"10 dB gap 无机制解释" 这个结论是**部分错误的**。项目内部在 0416 已经跑过 full-val (n=7403) 的误差预算分解，**机制已经定位到 velocity 端 (96-99% 占比)**，只是结论被埋在 `review/0416/conclusion.md` 里没进主叙事。基于这份已有证据 + E2 FOC-lite gate 失败，新的诊断必须回答的唯一问题是：

> **Transport 端的误差是 exposure bias 还是 velocity 容量?**

本文：
1. 把 0416 E1/E2 结论重新挖出来并系统化。
2. 定义"stop criterion 容差 > 配置方差"这一具体问题与修复方式。
3. 提出一个 **Path A 诊断实验**（Teacher-Forcing vs Rollout Gap），配套脚本 [`scripts/diagnose_tf_rollout_gap.py`](../../scripts/diagnose_tf_rollout_gap.py) 已实现，0.5 GPU-day 可跑完。
4. 画出 4 条突破路径 (A/B/C/D) 按 ROI 排序的决策树。

**最重要的新结论**：10 dB gap 的下一步**不是"尝试新架构"**（Nightmare 已把这条放进 KILL list），而是**运行 Path A**，根据结果三选一：改 loss / 做 decoder FT 诊断 / 直接 reframe。

---

## 1. "Stop criterion 容差 > 配置方差" 拆解

### 1.1 Proposer 的原始承诺 (Round 3)

> 若 150K checkpoint 的 PSNR 与 50K 相比在 **±0.05 dB** 以内，则终止 200K run。

### 1.2 三个数字的对撞

| 量 | 值 | 来源 |
|---|---:|---|
| Proposer 停止准则的容忍窗口 | **0.10 dB** (±0.05) | Proposer R3 |
| 5 个主力 config 在 50K 的实际 PSNR 跨度 | **0.083 dB** | [CLAUDE.md L150-165](../../CLAUDE.md#L150-L165) |
| σ_seed (同 config 不同随机种子标准差) | **未知** (一般 0.02-0.1 dB 量级) | 尚未测量 |

### 1.3 为什么这是设计缺陷

这是一个 **measurement ceiling** 问题：

> 你试图用一把最小刻度 10 厘米的尺子，去量一颗直径约 8 厘米的球。结果永远是"≤10 厘米"——无论真球多大。

对 200K 实验的直接后果：

| 场景 | 准则判定 | 科学含义 |
|---|---|---|
| ΔPSNR(150K−50K) = 0.03 dB | 终止 | 也许是真提升，被判错 |
| ΔPSNR(150K−50K) = 0.09 dB | 终止 | 也许在 CI 内完全无意义 |
| ΔPSNR(150K−50K) = 0.12 dB | 继续 | 也许只是 seed 波动 |
| ΔPSNR(150K−50K) = 0.00 dB | 终止 | 与 "plateau" 一致，但也与 "训练崩" 一致 |

**任何结果都不能区分**：
- "hop0 机制有效" vs "hop0 机制无效"
- "模型还能训" vs "模型已经饱和"

### 1.4 正确的准则设计 (if one still insists on 200K)

先做 Exp 1 测出 σ_seed，再按以下方式重写准则：

```
Let σ_seed = paired std over 3 seeds at 50K (to be measured).
Let D     = PSNR(150K) - PSNR(50K) with paired bootstrap 95% CI = [D_lo, D_hi].

Stop condition (any one triggers abandon):
  (a) D_hi < 2·σ_seed                      # upper bound of improvement < noise
  (b) D_lo < 0 AND D < 0.01 dB              # possibly degrading
  (c) σ_seed itself >= 0.08 dB              # effect cannot be isolated at all
```

若 (c) 首先触发，根本不用跑 200K——先停下来写负结果论文。

---

## 2. 10 dB Oracle Gap — 现有证据的再认识

### 2.1 已做的诊断 (但没进入主叙事)

Script: [`scripts/diagnose_error_budget.py`](../../scripts/diagnose_error_budget.py) (E1)  
Script: [`scripts/diagnose_foc_gap.py`](../../scripts/diagnose_foc_gap.py) (E2)  
Source: [`review/0416/conclusion.md`](../0416/conclusion.md), n=7403 full val

### 2.2 E1 Error Budget (复现)

| TP | Decoder Ceiling | E2E (rollout) | **Gap_Transport** | **Gap_Decoder** | Transport 占比 |
|---|---:|---:|---:|---:|---:|
| D20 | 46.64 | 35.49 | **10.74** | 0.41 | **96.3%** |
| D10 | 48.74 | 35.86 | **12.33** | 0.55 | 95.7% |
| D4 | 50.83 | 36.42 | **13.98** | 0.43 | 97.0% |
| NORMAL | 52.63 | 36.74 | **15.70** | 0.19 | **98.8%** |

- `Gap_Transport` = `PSNR_ceiling − PSNR(decode(z_pred) vs decode(z_gt))` → 归因于 latent 预测本身
- `Gap_Decoder` = `PSNR(decode(z_pred) vs decode(z_gt)) − PSNR(decode(z_pred) vs x_raw)` → 归因于 decode(z_gt) 本身不完美

**关键**：三个主要假设中只有 H3 存活：

| 假设 | 预言 | 观察 | 结论 |
|---|---|---|---|
| H1: Decoder 对 off-manifold 放大误差 | Gap_Decoder ≫ Gap_Transport | 0.2-0.5 dB 占 <4% | **证伪** |
| H2: ODE 积分步数太少 | half-step 比 full-step 差 ≥ 5% | 0.17% 且 half 略好 | **证伪** |
| H3: Velocity 回归本身不准 | Transport 占比 > 90% | 96-99% | **支持** |

### 2.3 下一层问题：Velocity 为什么不准？

这是当前证据链的终点。两个候选机制：

| 候选 | 机制 | 检测方式 |
|---|---|---|
| **M1 Exposure bias** | 训练时 velocity 输入是 teacher-forced 的干净 `z_gt(t_k)`；推理时输入是自己 rollout 的 `z_pred(t_k)`。分布 shift → 误差累积 → 10 dB | **Path A**: 比较 1-step TF vs 1-step RO 的 PSNR gap |
| **M2 Velocity 容量** | 即使给干净 `z_gt(t_k)`，网络也无法回归到 `z_gt(t_{k+1})` 所需精度。网络/loss/latent 空间不匹配 | **Path A**: `PSNR_TF vs PSNR_ceiling` 之间的差 |

两者可以共存，Path A 的 4 种可能结果对应 4 种修复方向（见 §4 决策树）。

---

## 3. Path A 诊断实验设计

### 3.1 实验目标

**一次数据采集，同时解耦 M1 和 M2。**

### 3.2 方法

对每个 hop k ∈ {0, 1, 2, 3}，在同一个 checkpoint 上做两种 1-step 预测：

- **Arm TF** (teacher-forced 输入):  
  `z_pred_TF[k+1] = model.predict_step(z_gt[t_k] → t_{k+1})`

- **Arm RO** (rollout 输入):  
  先完整跑一遍 rollout 得到 `z_ro[t_0..t_4]`，然后  
  `z_pred_RO[k+1] = model.predict_step(z_ro[t_k] → t_{k+1})`

对每个预测都 decode 后计算 PSNR(vs `x_gt[t_{k+1}]`)。

> 注：k=0 时 `z_ro[t_0] = z_gt[t_0] = z_d50`，所以 hop 0 上 TF == RO（这是构造保证）。真实 exposure 信号在 hop 1-3。

### 3.3 三个量的定义

| 量 | 公式 | 含义 |
|---|---|---|
| `exposure_gap_dB` | `PSNR_TF − PSNR_RO` | **exposure bias 强度**（>0 表示 RO 输入比 TF 输入差） |
| `ceiling_gap_dB` | `PSNR_ceiling − PSNR_TF` | **velocity intrinsic capacity 差距**（即使给 GT 输入还差多少） |
| `lmse_ratio_ro_over_tf` | `mean(LMSE_RO) / mean(LMSE_TF)` | latent 空间的误差放大比 |

### 3.4 Verdict Matrix

| `exposure_gap_dB` (hops 1-3 mean) | `ceiling_gap_dB` (hops 1-3 mean) | 判决 | 行动 |
|---|---|---|---|
| **≥ 0.5 dB** | any | `EXPOSURE_BIAS_DOMINATES` | 强化 chainstable loss / scheduled sampling / DAgger |
| < 0.2 dB | ≥ 5 dB | `VELOCITY_CAPACITY_BOUND` | 换 velocity 表示 / 增容量 / pivot paradigm |
| < 0.2 dB | < 2 dB | `NEAR_CEILING` | 与 E1 矛盾，复查数据管线 |
| 0.2-0.5 dB | any | `MIXED_CAUSE` | 两者并行修复 |

阈值在 CLI 中可调 (`--exposure-threshold 0.5 --exposure-near-zero 0.2 --capacity-threshold 5.0`)。

### 3.5 实施与运行

脚本: [`scripts/diagnose_tf_rollout_gap.py`](../../scripts/diagnose_tf_rollout_gap.py) (488 行)

```bash
PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual \
/home/qujiaxiang/.conda/envs/rae/bin/python scripts/diagnose_tf_rollout_gap.py \
    --config     configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost/best.pt \
    --split val --max-slices 0 --batch-size 8 --device cuda:0 \
    --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/tf_rollout_gap_imgaux_boost
```

**产出**：
- `tf_rollout_gap_val.json` — 每 hop 的聚合统计 + paired bootstrap 95% CI + 自动 verdict
- `tf_rollout_gap_val_per_slice.csv` — 7403 行 × 20 列原始数据，供后续做分层分析

**运行时间估计**：
- 224×224, batch 8, decode 5 次/hop × 2 arm = 每 slice ~10 decode calls
- full val (7403 slices) ≈ 0.5 GPU-day 在 H100，~1 GPU-day 在 3090/4090
- 若想快速 sanity check 先用 `--max-slices 512`，约 10-20 分钟

**诚实度 gate**：
- 至少在 **3 个** checkpoint 上跑 (推荐: `imgaux_boost` C, `pixenc_ablation` N1, `chainstable50k`)
- 如果 3 个 checkpoint 的 verdict 不一致 → mixed cause，必须两手抓
- 结果不论好坏都诚实报告，不做 cherry-pick

---

## 4. 4 条突破路径决策树

```
                        ┌─────────────────────────┐
                        │   Exp 1: σ_seed (Day 1) │
                        │   4 GPU-day             │
                        └───────────┬─────────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
           σ ≥ 0.08 dB       σ 0.04-0.08       σ < 0.04 dB
           effect dead      effect marginal   effect alive
                │                │                │
                ▼                ▼                ▼
         ══════════════════════════════════════════════════
         │  Path A: TF-vs-Rollout Gap Diagnostic (Day 2)  │
         │  0.5 GPU-day, 3 checkpoints                    │
         ══════════════════════════════════════════════════
                                    │
          ┌─────────────────┬───────┴────────┬────────────────┐
          ▼                 ▼                ▼                ▼
    EXPOSURE_BIAS    VELOCITY_CAPACITY   MIXED_CAUSE     NEAR_CEILING
          │                 │                │                │
          ▼                 ▼                ▼                ▼
       Path A+             Path C         Path A+ + Path C  data pipeline
       fix loss       decoder FT dx       (both, iterate)   re-audit
```

### 4.1 Path A+ (Exposure Bias 分支) — 中等成本

**前提**: `exposure_gap_dB ≥ 0.5 dB`

**假设**: chainstable 叙事正确但执行不够强。

**实验**:
- A+.1 加重 alpha ramp: 早期更多 teacher-forcing，后期更多 rollout (当前 alpha 固定)
- A+.2 重加权 step_weights: 给 rollout 后段 (hop 2/3) 更大 loss
- A+.3 DAgger-style: 把当前 checkpoint 的 rollout 轨迹回放进下轮训练

**成本**: 2-3 GPU-week (需要重训)  
**风险**: chainstable 叙事在 Nightmare 已被指出 "tail 增益 > head 增益"；如果 Path A 显示 exposure bias 不主导，这条路等于浪费  
**目标**: +0.5 dB 到 +1.5 dB (workshop paper "how to fix exposure bias in cascaded latent transport")

### 4.2 Path B (Velocity 表示) — 中等成本

**前提**: `ceiling_gap_dB ≥ 5 dB` 且 `exposure_gap_dB` 小

**假设**: flow matching 连续场对这个 latent 过参数化；直接 z-to-z 回归或 diffusion 更合适。

**实验** (3 arms, 同一个 DiT backbone):
- F: flow matching + Euler (当前)
- D: 直接 `z_{t+1} = f(z_t, t, hop)` 回归 (1-step, 无 ODE)
- C: conditional diffusion + CFG

**成本**: 3-4 GPU-week  
**价值**: 如果 D 接近或超过 F，**本身就是可发表发现** ("flow matching unnecessary when latent dynamics are non-smooth")

### 4.3 Path C (Decoder FT Diagnostic) — 低成本，高信息量

**注意**: Nightmare KILL list 第 K3 是 "不要把 decoder FT 当方法救命稻草"。**此处是把 decoder FT 当诊断实验，不当主方法。**

**前提**: 任一 verdict，作为 ground-truth bound 估计

**实验**:
- 解冻 decoder 的最后 1-2 个 upsample block (不是全部 FT)
- 在 transport 预测出来的 `z_pred` 上做 5K steps FT
- 测新的 E2E PSNR

**预期** (2 种情形):

| 观察 | 结论 |
|---|---|
| FT 后 PSNR 提升 +3 dB 或以上 | **瓶颈在 RAE frozen decoder 的表达力**，不在 transport。这是一个硬科学结论，足以支撑 negative-result paper 的核心论点。 |
| FT 后 PSNR 提升 < 0.5 dB | **瓶颈在 latent 信息本身 (RAE encoder null space)**，decoder 救不了。更强的 negative 结论。 |

**成本**: 2-4 GPU-day  
**策略选择**: **无论 Path A 的结果是什么，Path C 都该跑**——因为它是唯一能给 10 dB gap 提供"在 paradigm 内能做到多好"上界的实验。论文里只要说清楚"Path C 是诊断不是 proposed method"即可。

### 4.4 Path D (Paradigm Reframe) — 0 GPU-day

**前提**: Path A + Path C 结果一致指向 "paradigm 失败"

**可能的具体结论**:
- "frozen RAE + latent-only transport" 在 4-hop low-dose PET 上结构性无法突破 36.2 dB
- 10 dB gap 的 70%+ 可归因于 RAE encoder null space；进一步需要联合训练 encoder/decoder
- 4-hop 级联是 overkill；2-hop 或 1-hop 在同 compute budget 下可能更好

**产出**: MIDL 2027 / MICCAI workshop 负结果论文  
**成本**: 0 GPU-day + 4 周写作

---

## 5. 10 GPU-day 执行清单

| Day | 实验 | GPU-day | 决策 |
|---|---|---:|---|
| 1 | **Exp 1 σ_seed** (3 seeds × 2 configs, 50K each) | 4 | 决定 hop0 效应死活 |
| 2 | **Path A** 在 3 checkpoints 上 | 1.5 | 定位 exposure / capacity |
| 2 | **Path C** 小规模 decoder FT 诊断 | 1 | 估计 RAE bound |
| 3 | (可选) Exp 2 mediation test | 0.5 | 补 story 因果 |
| 3 | 解读结果, 冻结实验, 开始写作 | 0 | — |
| 4-10 | Workshop 短文 (8 页) 撰写 + mock review + 外部 baseline (Pix2Pix 或 published PET-DDPM) | ≤ 3 | 投稿准备 |

**总计**: ~10 GPU-day + 2 周写作。

---

## 6. 与 Nightmare Round 5 的关系

- 不冲突 KILL list：K1 (200K) K2 (新架构) K6 (训更久) 仍然 KILL。
- **Path A 是新增的必做实验**，不在原 Nightmare 的 3-experiment list 里 (原为 Exp 1 σ_seed + Exp 2 mediation + Exp 3 200K)。本文档主张用 **Path A 替换 Exp 3**。
- **Path C 是原 KILL list K3 的精细化**：原 K3 = "decoder FT 不作为方法"；此处 = "decoder FT 作为 bound 诊断"。语义上无冲突。
- Reframe 方向 (Path D) 与 Round 5 一致：MIDL/MICCAI workshop 负结果论文。

---

## 7. 最终一句话

> **10 dB oracle gap 的机制其实已经被项目自己的 0416 诊断定位到 velocity 端 (96-99%)；剩下的只是在 velocity 端再走一步——用 Path A (0.5 GPU-day) 分开 exposure bias 和 capacity，用 Path C (2-4 GPU-day) 估 RAE 上界——这两步数据一旦拿到，项目的故事 (无论是正向还是负向) 就可以闭环。**

---

## Artifacts

| 文件 | 状态 |
|---|---|
| [`scripts/diagnose_tf_rollout_gap.py`](../../scripts/diagnose_tf_rollout_gap.py) | **新增** (488 行, syntax-checked) |
| [`scripts/diagnose_error_budget.py`](../../scripts/diagnose_error_budget.py) | 已存在, E1 来源 |
| [`scripts/diagnose_foc_gap.py`](../../scripts/diagnose_foc_gap.py) | 已存在, E2 来源 |
| [`review/0416/conclusion.md`](../0416/conclusion.md) | 已存在, 本分析的实证锚点 |
| [`review/0424/nightmare/05_final_consensus_and_roadmap.md`](./nightmare/05_final_consensus_and_roadmap.md) | 已存在, 本文档的母体 |
| [`review/0424/breakthrough_analysis_10dB_gap.md`](./breakthrough_analysis_10dB_gap.md) | **本文件** |

---

*End of 10 dB Gap Breakthrough Analysis. Next: run Exp 1 σ_seed + Path A in parallel; Path C after Path A data arrives.*
