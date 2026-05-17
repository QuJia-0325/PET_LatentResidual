# Round 5 Peer Review Integration (20260517 深夜)

- date: 2026-05-17 深夜
- branch: foc_lite_hop0
- HEAD when written: e9733cc
- reviewers: agent1, agent2, agent3 (Round 5 独立评审)
- inputs: [PEER_REVIEW_PROMPT_round5_slot_allocation_20260517.md](./PEER_REVIEW_PROMPT_round5_slot_allocation_20260517.md) + 三份独立 review
- purpose: 整合三位 reviewer 共识；**关键发现**：本轮 review 揭示了 3 个新 bug，**全部影响 V18 (slot 1)，比 slot 2/3 决策严重 10×**。

---

## §0 — TL;DR

Round 5 本意是审 slot 2/3 分配，但三位 reviewer **同时**指出 3 个我没看见的 V18 设计 bug：

| Bug | 来源 | 严重度 | 影响 |
|---|---|---|---|
| **B8** | agent3 独家 | **CRITICAL** | V21 整个概念错（conv_head.py 不是 post-decode refiner），从未来候选删除 |
| **B9** | agent3 独家 | **HIGH** | V18 `use_pred_latent=true` 让 KL 与 image_aux 在同一 z_pred 输出上**梯度对抗**，V18 越成功 KL 越拉回 |
| **B10** | agent3 独家 | **HIGH** | "11.2 dB" 被 Round 4 定性为不可吃后，又被同一份 Round 4 用来抬 rank=32 + +0.30 dB 阈值（B5 同型复发）；真实 attackable ≈ 0.5 dB |

三位 reviewer 的 slot 决策共识：
- slot 2 = **V13**（2/3 推荐，agent2 推 V14）
- slot 3 = **null**（agent1/agent3 共识，agent2 同意 mix）
- V9a 改走 CPU 通道（不占 slot）
- V21 retire

---

## §1 — Round 5 共识矩阵

| 议题 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| **slot 2** | V13 | V14 | V13 | **V13 (2/3)** |
| **slot 3** | V21 (Day 3-4) | V9a → null | null | **null + V9a CPU** |
| V14 EV | decorative | **必要** (V14 是唯一干净 d_pure) | decorative (0.02-0.04 dB) | **V14 主要价值是 d_pure precision，与 V18 EV 修正后边界场景重叠少** |
| V13 被错降 | yes | yes (排在 V14 之后) | yes (M1 ≠ V13 预期) | **是** |
| V21 launch | yes Day 3-4 | no | **NO (代码不符)** | **不上**（agent3 撤销 V21 概念） |
| V9a 占 slot 3 | replace | OK | replace (CPU 即可) | **不占 slot** |
| Round 5 是否合法 | half + B8-bias | yes + 新偏差 | yes_partially + B8 sunk cost | **合法但有新 bias** |

---

## §2 — 3 个新 bug 的详细分析

### 2.1 B8 — V21 整个概念与代码不符（最严重）

我之前在 Round 4 §6 + Round 5 prompt §1 Q4 + REVIEW_INTEGRATION_round4 §1 都把 V21 描述为：
> "freeze V7 整体 + 在 decode(z_pred^V7) 之上训 RAE 自带的 ConvDecoderHead"

agent3 去读了实际代码 [RAE/RAE/src/stage1/decoders/conv_head.py:92-117](../../RAE/RAE/src/stage1/decoders/conv_head.py#L92)：

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, N, C=512] — transformer 输出，cls token 已 removed
    # 内部 reshape 到 [B, C, 14, 14]，做 token-domain 3×3 Conv → PixelShuffle(14) → [B, 1, 196, 196]
```

**`ConvDecoderHead` 接的是 transformer hidden state，是 `decoder_pred + unpatchify` 的替换品**。它**不是**接在 `decode(z_pred)` [B,1,224,224] 像素张量之后的后处理 ConvNet。

含义：
1. V21 要做实际上是：hook MAE decoder 倒数第二步抽 hidden_states `[B, 196, 512]` → 绕开 `decoder_pred` → 接 ConvDecoderHead → 训新头。这是**改 decoder 架构**，与 V18 LoRA on `decoder_layers.{6,7}` 是**同一参数族**。
2. **V21 和 V18 不正交**，不能作为 hedge。
3. agent2 还指出 RAE 自己的 [patch_artifact_analysis.md](../../RAE/RAE/docs/patch_artifact_analysis.md) 显示 ConvDecoderHead 历史上比 baseline 低 5.39 dB —— 是已知失败方案。
4. 真正的"post-decode pixel ConvNet refiner" 代码里**不存在**，需要新设计（agent3 命名 V22），规模 ≈ V18，不应在 Round 5 决策窗口内。

**动作**：从所有文档撤销 V21；Round 4 §6 决策树的 fallback 应改为 "保持 V18 + 后续重设计 V22"。

### 2.2 B9 — V18 `use_pred_latent=true` 制造 KL 与 image_aux 梯度对抗

Round 4 §2.2 我从 `use_pred_latent=false` 改成 `true`，理由错了。

agent3 的逐项推：

| 配置 | KL 项 | image_aux 项 | 关系 |
|---|---|---|---|
| `use_pred_latent=false` (Round 4 之前) | `MSE(decode_lora(z_GT), decode_frozen(z_GT))` | `MSE(decode_lora(z_pred), x_target)` | **不同样本，梯度正交** |
| `use_pred_latent=true` (Round 4 改后) | `MSE(decode_lora(z_pred), decode_frozen(z_pred))` | `MSE(decode_lora(z_pred), x_target)` | **同一样本，目标矛盾** |

`use_pred_latent=true` 时两个 loss **作用在同一 `decode_lora(z_pred)` 输出上**，分别拉向 V7-frozen 输出和 x_target。**V18 越成功（输出离 V7 越远、离 x_target 越近），KL 项越大，梯度越反向**。

Round 4 我的错误推理：

> "use_pred_latent=true 让 V18 适应 pred path 而不是 GT path"

正确推理应该是：

> KL 作用在哪条 path = 在哪条 path 上**约束 LoRA 不能动**。
> - `false` 路径 = 在 z_GT 上约束（保持 stage 2 的 PET reconstruction），让 V18 在 z_pred 路径自由适配
> - `true` 路径 = 在 z_pred 上约束（保持 V7 的 z_pred → image），完全冻结 V18 想做的事

**正确设计应该是 `use_pred_latent=false`**。

**动作**：
- V18 训练**不停**（kill switch 数据未到，且 KL 在 step 0-2000 warmup 内尚未介入）
- 让 codex 在 step 180K 评估时**同时 fork 一个 V18b**（use_pred_latent=false 旁支），从 V18 step 180K ckpt 继续 20K step，看是否能解锁 ΔPSNR

### 2.3 B10 — 11.2 dB 数字被买回去（B5 同型复发）

Round 4 §1 明确写 "11.2 dB transport gap 是 round-trip ceiling，**不是 V18 可吃的**" → CRITICAL bug。

但 Round 4 决策规则又**用同一 11.2 dB 数字**触发 rank=32 + +0.30 dB 阈值。这是 B5（authoritative-but-incomplete decomposition）在同一份文档里的**自我循环**。

agent3 给出正确的 attackable 代理量（从 [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) 实测数据）：

| 量 | 值 | 含义 |
|---|---:|---|
| `psnr_ceil = PSNR(decode(z_GT), x_target)` | 46.64 dB | RAE round-trip 上限 |
| `psnr_transport = PSNR(decode(z_pred^V7), x_target)` | 35.44 dB | V7 实际 |
| `psnr_transport_vs_ceil = PSNR(decode(z_pred), decode(z_GT))` | **35.94 dB** | **V7 z_pred decode vs GT z decode** |
| `psnr_ceil - psnr_transport` | 11.20 dB | "round-trip gap"（**不可全吃**） |
| **`psnr_transport_vs_ceil - psnr_transport`** | **0.50 dB** | **decoder 在 z_pred 上的真实改进空间** |

V18 改 decoder 只能从这 0.5 dB 里吃。**V18 真实 EV ≈ 0.05-0.30 dB**（与 V11/V17 同档），不是 0.5-2.0 dB。

**动作**：
- V18 success threshold 从 **+0.30 dB → +0.15 dB**
- V18 partial threshold 从 [0.05, 0.30] → [0.03, 0.15]
- V18 kill threshold 不变（+0.05 dB at step 20K）
- 更新 [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) 反映 EV 修正

---

## §3 — Slot 决策（基于 EV 修正后）

### 3.1 Round 5 三位 reviewer 推荐

| reviewer | slot 2 | slot 3 |
|---|---|---|
| agent1 | V13 | V21 (Day 3-4) ← **取消，B8** |
| agent2 | V14 | V9a → null |
| agent3 | V13 | null (V9a CPU) |

### 3.2 整合后决定

**slot 2 = V13** （2/3 推荐 + agent2 也同意 V13 排第二）：
- V13 = V7 + image_aux off (single-axis)，[V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) 已就绪
- 价值：第一次把 V7-V8 的 +0.308 dB NORMAL Δ 分解为 image_aux 真贡献 vs step_weights 真贡献
- 两端结论都强：
  - ΔPSNR(V7-V13) ≈ 0 → image_aux 整族被证伪，省 30+ GPU-day
  - ΔPSNR ≈ +0.30 dB → image_aux 是 NORMAL 改善的因果通道，V18 是必要补充

**slot 3 = null + V9a 走 CPU 通道**：
- V9a 是 pandas + metrics.jsonl 重新算 selector + 1 次可选 full-val eval (~10 分钟可挤进 V18 自己的 eval 队列)。**不占 slot**。
- slot 3 留作 V18 eval 紧急通道（step 180K/200K full-val + B9 旁支 fork）
- agent3 关键论点：历史上前 4 轮所有 critical 时刻都卡在 "eval 等不出来" 而不是 "训练 quota 不够"

**V14 不上**（agent1 + agent3 一致）：
- d_pure 真值估计 0.015-0.040 dB，V14 输出范围与 V7-V6_NOISE 上界相同量级
- V18 修正 EV 后真实带宽 0.05-0.30 dB，落在 0.03-0.05 dB（V14 唯一能 unlock 的窗口）概率 < 20%
- 性价比远不如 V13

**V21 retire**（B8）。

### 3.3 GPU 排程

| Day | slot 1 | slot 2 | slot 3 |
|---|---|---|---|
| **今天** | V18 running (step ~180K) | **V13 yaml smoke + launch** | idle / V9a CPU pandas |
| Day 1-2 | V18 → step 200K | V13 0-30K | idle / V18 eval |
| Day 3 | V18 eval + 若 ΔPSNR<+0.10 dB **fork V18b** (use_pred_latent=false) | V13 30-60K | (可能) V18b 用 slot 3 跑 20K step |
| Day 4-7 | V18/V18b 评估 + 决策 | V13 60-160K | idle / V13 eval / V18 评估 |

---

## §4 — 立即执行的动作

### 4.1 V18 yaml KL 修正

V18 训练**继续**（kill switch 未触发，KL 在 step 0-2K warmup 内尚未介入）。但 yaml 的 `use_pred_latent=true` 是 bug，需要：

- **不在线改 yaml**（会触发 resume_allow_config_mismatch 失败）
- 在 step 180K eval 后**fork V18b** = 从 V18 step 180K ckpt 出发，新 yaml `use_pred_latent=false`，再训 20K
- 用 V18b vs V18 比较 KL design 的影响

### 4.2 V18 success threshold 修正

更新 [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) §2.3：

| 阈值 | 旧 | 新 |
|---|---|---|
| PRIMARY SUCCESS | +0.30 dB | **+0.15 dB** |
| PARTIAL | [+0.05, +0.30] | [+0.03, +0.15] |
| KILL @ 20K | +0.05 dB | 不变 |

依据：[B10](#23-b10---112-db-数字被买回去b5-同型复发) 修正后真实 attackable ≈ 0.5 dB。

### 4.3 V21 retire

撤销以下文档中 V21 引用：
- [REVIEW_INTEGRATION_round4_20260517.md §6.2 / §7](./REVIEW_INTEGRATION_round4_20260517.md)
- [PEER_REVIEW_PROMPT_round5_slot_allocation_20260517.md](./PEER_REVIEW_PROMPT_round5_slot_allocation_20260517.md) §1 Q4
- 本文件 §3.1 已标记

未来真要做 post-decode pixel refiner（V22），需重新设计，不复用 ConvDecoderHead。

### 4.4 V13 launch

服务器执行（codex）：
```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p review/0516/V13_true_image_aux_ablation/logs
CUDA_VISIBLE_DEVICES=0 nohup /home/qujiaxiang/.conda/envs/rae/bin/python -u \
    train_first_hop.py --config review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml \
    > review/0516/V13_true_image_aux_ablation/logs/V13_train_${TS}.log 2>&1 &
echo $! > review/0516/V13_true_image_aux_ablation/logs/V13_pid.txt
```

**warning**: V13 是 image_aux=off 的 full 160K 训练，这条代码路径在 train_first_hop.py 里**从未跑过完整 160K**（V8 是 V6 step_weights + img off，与 V7 不同；V13 是 V7 step_weights + img off，第一次跑这组合）。**Codex 应先跑 smoke 1-step test 确认 image_aux=off branch 没有 NaN 等问题，再 launch 正式训练**。

### 4.5 V9a CPU 通道

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
python3 <<PY
import json
with open('/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/metrics.jsonl') as f:
    records = [json.loads(l) for l in f if json.loads(l).get('event') == 'val_full']

def score_v9(r): return 0.5*r['val_chain_d20_mse']+0.45*r['val_chain_d10_mse']+0.9*r['val_chain_d4_mse']+2.5*r['val_chain_normal_mse']
def score_v7(r): return 0.5*r['val_chain_d20_mse']+0.45*r['val_chain_d10_mse']+0.9*r['val_chain_d4_mse']+1.5*r['val_chain_normal_mse']

v9 = min(records, key=score_v9); v7 = min(records, key=score_v7)
print(f"V7 selector (β=1.5): step={v7['step']}, NORMAL={v7['val_chain_normal_mse']:.6f}")
print(f"V9 selector (β=2.5): step={v9['step']}, NORMAL={v9['val_chain_normal_mse']:.6f}")
print(f"V9 NORMAL improvement: {(v7['val_chain_normal_mse']-v9['val_chain_normal_mse'])*1000:.3f} (×1e-3)")
PY
```

如果 V9 selector 选不同 step 且 NORMAL MSE 改善 > 3·d_pure_estimate，再考虑用 V18 eval 队列里余量跑一次该 step 的 full-val。**不需要 GPU slot**。

---

## §5 — 元教训：第 8/9/10 次 bias

**B8 = "推荐时复用过期描述"**：V21 我推荐了 3 轮（Round 4 + Round 5 prompt + Round 5 整合草稿），每次都用 "RAE 已有 conv_head.py = 几乎零代码" 这个错误描述。从未读 conv_head.py 真实 forward signature。agent3 一次性证伪。

**B9 = "改设计时只看自己想要的 path，不看 loss 的实际作用域"**：Round 4 我把 use_pred_latent false→true，理由是"让 V18 适应 pred path"，没分析 KL 和 image_aux 是否作用在同一样本上。

**B10 = "B5 同型复发，文档里同一个数字一会儿当 ground truth 一会儿当 bug"**：Round 4 同一份文档里 §1 说 "11.2 dB 不是 V18 可吃的"，§2.3 又用 11.2 dB 触发 rank=32 + +0.30 dB 阈值。

**统一形态**：B1-B7 是 "用单一数字驱动决策"，B8-B10 是 **"用过期/未验证的对象描述驱动决策"**。修复方法相同：**在任何决定之前，读它依赖的最新源码 + 数据**。

memory 记录：
> 任何引用代码模块、配置参数、loss 数学公式的决定，必须在 commit 前重读最新源码并 cross-check 实际 forward signature / state_dict keys / 实测数值。

---

## §6 — 用户决策点

1. **接受 §4 全部 4 项动作**（V18 继续 + B9/B10 修正待 step 180K fork + V21 retire + V13 launch + V9a CPU）？（推荐）
2. **不动 V18，只 retire V21 + 不 launch slot 2/3，等 V18 200K 结果**？（最保守）
3. **不接受 reviewer 共识，按原 V14 + V9a 推荐执行**？（不推荐 — 与 agent2/3 多数共识反向）

我倾向 (1)，因为 V13 与 V18 完全独立、V9a 不占 slot、V18 KL bug 不影响当前训练（warmup 内）但需要后续验证。
