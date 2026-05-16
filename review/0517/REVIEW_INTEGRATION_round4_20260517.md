# Round 4 Peer Review Integration (20260517 晚)

- date: 2026-05-17 晚
- branch: foc_lite_hop0
- HEAD when written: 65198e3 (will bump after this commit)
- reviewers: agent1 (codex-style), agent2 (gemini-style), agent3 (deep-arch)
- inputs: [PEER_REVIEW_PROMPT_round4_domain_adaptation_chain_20260517.md](./PEER_REVIEW_PROMPT_round4_domain_adaptation_chain_20260517.md) + 三份独立 review 全文
- purpose: 整合 Round 4 三位 reviewer 的发现 + 立即执行最便宜的 P0 (gap-decomposition probe + 3 sanity checks + KL fix)

---

## §0 — TL;DR

Round 4 三位 reviewer **高度共识**地确认我的 V18 推荐有 5 处问题，并提出一个**1 GPU-hour 不可绕开的前置实验**：

| # | 问题 | 严重度 | fix |
|---|---|---|---|
| 1 | "11.2 dB transport gap" 是 round-trip ceiling，**不是 V18 可吃的 gap** | **CRITICAL** | Day 0 morning gap-decomposition probe（已写）|
| 2 | KL pull-back λ=0.5 比 image_aux 大 12.5× → 会**冻死** V18 decoder LoRA | **CRITICAL** | yaml 改 λ=0.05、use_pred_latent=true、warmup=2000（已修）|
| 3 | Stage 2 decoder 是 **full-finetune** 不是 LoRA；rank=8 在已饱和容量上可能太弱 | HIGH | 预注册 rank=8 / rank=32 二选一基于 probe 结果 |
| 4 | V7 best.pt 可能有 stage 2 留下的 decoder LoRA keys（V18 wrap 会出错）| MEDIUM | probe 自动 audit ckpt keys + decoder Linear type |
| 5 | RAE 自带 conv head `src/stage1/decoders/conv_head.py` → V21 = post-decode conv head 是 V18 fallback，几乎零代码 | MEDIUM | 起草 V21 config 备用 |

**5th bias 形态**：三人都说 yes，新形态是 "**authoritative-but-incomplete decomposition**" — 用单个观察量 (11.2 dB) 替代分解，结构上与 B1-B4 同型。

---

## §1 — 三位 reviewer 共识矩阵

| 议题 | agent1 (codex) | agent2 (gemini) | agent3 (deep-arch) | 共识 |
|---|---|---|---|---|
| stage 2 decoder 形式 | **full-finetune** (cited train_stage1.py:289) | **full-finetune** (cited train_lora_dinov2.py) | **full-finetune** (cited train_lora_dinov2.py:251-253) | full-finetune；agent3 正确归因到 train_lora_dinov2.py |
| 11.2 dB gap 可被 V18 吃 | partial (<30%) | partial（口径不对，需 decomp）| **no**（wrong direction）| **必须做 decomp，不能直接外推** |
| V18 EV 期望值 | 0.2-0.5 dB | 不能定 | **0.05-0.30 dB**（与 V11/V17 同档）| 严重高估 |
| KL λ=0.5 | **calibration bug 12.5×** | 推荐拆 kl_gt / img_pred | **推荐 λ=0.05 + use_pred_latent=true 或删** | **λ=0.5 必死** |
| use_pred_latent | 默认 false 是对的（停 drift）| 不能定 | **应改 true**（让 V18 适应 pred path）| agent3 论证最强 |
| rank=8 | "对的，但理由错" | 不能定 | **rank≥32 更诚实** | 用 probe 结果分流 |
| step-0 等价 | 加 wrap-order assertion | 加 ckpt key audit | **30 秒 type() check 即可** | 三个都要做 |
| V21 fallback | V20 = latent-residual refiner | RAE loading 修正 | **V21 = conv head（RAE 已有 conv_head.py，0 代码）** | **V21 是最便宜 fallback** |
| 5th bias | yes (over-confidence after partial source reading) | yes (source-reading overconfidence) | **yes (authoritative-but-incomplete decomposition, B5 与 B1-B4 同型)** | **yes** |
| hard timebox | Day 2 launch 必须 | 3 天 | **不延期，但 Day 0 必加 1h decomp probe** | **Day 2 launch 不变，Day 0 加 probe** |

---

## §2 — 已立即执行的修正

### 2.1 写 gap-decomposition probe

[tools/probe_v18_gap_decomposition.py](../../tools/probe_v18_gap_decomposition.py) (~280 行)：

- 在 V7 best.pt 上跑全 val (n=7403, ~30-60 min)
- 计算 3 个量：
  - `psnr_ceil = PSNR(decode(z_GT_D20), x_D20)` ← decoder 单独的能力上限
  - `psnr_transport = PSNR(decode(z_pred^V7), x_D20)` ← V7 实际表现
  - `latent_l2_rel = ‖z_pred − z_GT‖₂ / ‖z_GT‖₂` ← latent 域 transport error
- 自动 audit：decoder Linear type + V7 ckpt 里 decoder LoRA keys 数量
- 按 pre-registered 阈值输出决策：
  - `attackable_gap < 2 dB` → V18 wrong direction → 启动 V21
  - `2-5 dB` → V18 rank=8 + KL fix + 阈值降到 +0.20 dB
  - `≥ 5 dB` → V18 rank=32 + KL fix + 原阈值 +0.30 dB
- 输出三份工件：`GAP_DECOMP_REPORT.md` / `GAP_DECOMP_PER_SLICE.csv` / `GAP_DECOMP_SUMMARY.json`

### 2.2 修 V18 yaml KL calibration

[V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) `decoder_kl_pullback`：

```yaml
# 旧
lambda_kl: 0.5
use_pred_latent: false
warmup_steps: 0

# 新（agent1 + agent3）
lambda_kl: 0.05            # 从 0.5 降到 0.05（grad ratio 1.25× 而非 12.5×）
use_pred_latent: true      # KL 作用在 predicted-path 而非 GT-path
warmup_steps: 2000         # KL 在 V18 启动 2K step 后才开始介入
kill_switch:
  check_step: 5000
  min_kl_change_pct: 5.0
  min_image_aux_improvement_pct: 10.0
  v7_image_aux_at_5k: 0.007893
  kl_drift_warn_ratio: 3.0
```

### 2.3 起草 V21 conv head fallback config（待写）

V21 = "freeze V7 整体 + 在 decode(z_pred^V7) 之上训 RAE 自带的 ConvDecoderHead"
- 复用 [RAE/RAE/src/stage1/decoders/conv_head.py](../../../RAE/RAE/src/stage1/decoders/conv_head.py)
- 0 新模型代码；只需 1 个新 yaml + 1 个 wrapper module 调 conv_head
- 训练成本：~2 天（比 V18 的 7 天少 5 天）
- 触发条件：gap-decomposition probe 显示 `attackable_gap < 2 dB`

V21 yaml 尚未写（等 probe 结果出来再决定参数）。

---

## §3 — Day 0 修订计划（替换 CODEX_RUNBOOK_V18 §3）

| Day 0 任务 | 时间 | 用途 |
|---|---|---|
| **NEW** Phase 0: 跑 `tools/probe_v18_gap_decomposition.py` 全 val | **30-60 min** | 决定 V18 / V21 / 停滞 |
| Phase A: RAE inference-time eval/train mode probe | 5 min | 验证 RAE 是 deterministic operator |
| Phase B: V7 中间 ckpt 存活检查 (`ls step_*.pt`) | 1 min | V9a feasibility（已降级，可选）|
| Phase C: V9a 重选（如 ckpt 存活）| 1-2 h | 可选 |

**Day 0 总时间**：~1-2 GPU-hour（包含 V9a）或 < 1 GPU-hour（不包含）。

### Day 0 完成后的 binary decision

读 `review/0517/V18_decoder_lora/GAP_DECOMP_REPORT.md` 里的 `## Verdict` 段：

| Decision | Day 1 动作 |
|---|---|
| `V18_WRONG_DIRECTION_LAUNCH_V21` | 起草 V21 yaml + conv head wrapper module + launch V21 |
| `V18_LAUNCH_RANK8_KLFIX_DOWNGRADE_THRESHOLD` | 按当前 V18 yaml（已含 KL fix）launch；success 阈值改 +0.20 dB |
| `V18_LAUNCH_RANK32_KLFIX` | 把 V18 yaml `decoder_lora.rank: 8 → 32` 后 launch |

---

## §4 — V18 launch 前的 4 个 mandatory sanity（来自 agent1）

写在 train_first_hop.py V18 branch（codex 实施时确保这 4 个 assertion 不被绕过）：

1. **Wrap-order assertion**（在 wrap_decoder_with_lora 调用之后、load_state_dict 之前/之后必须验证）：
   ```python
   v7_ckpt = torch.load(resume_path, map_location='cpu')['model']
   sentinel_key = "rae.decoder.decoder_layers.11.attention.attention.query.linear.weight"
   v7_key = sentinel_key.replace(".linear.weight", ".weight")
   if v7_key in v7_ckpt:
       expected = v7_ckpt[v7_key]
       actual = dict(model.named_parameters())[sentinel_key].detach()
       assert torch.allclose(actual, expected), \
           f"V18 wrap-order bug: {sentinel_key} != V7 weight"
   ```

2. **Step-0 forward equivalence**:
   ```python
   rae_lora.eval()
   x_lora  = rae_lora.decode(z_test).detach()
   x_frozen = rae_frozen.decode(z_test).detach()
   err = (x_lora - x_frozen).abs().max().item()
   assert err < 1e-5, f"V18 step 0 forward differs from V7 by {err}"
   rae_lora.train()
   ```

3. **Decoder Linear type check**（probe 已自动做，但 trainer 也要做一次）：
   ```python
   q = model.rae.decoder.decoder_layers[-1].attention.attention.query
   assert type(q).__name__ == "LinearWithLoRA", \
       f"V18 wrap did not produce LinearWithLoRA at last block, got {type(q).__name__}"
   ```

4. **resume missing key whitelist**（确保 V18 加载 V7 ckpt 时 lora_A/lora_B missing 是被允许的）：
   ```python
   missing, unexpected = model.load_state_dict(sd['model'], strict=False)
   allowed_missing = [k for k in missing if k.endswith(".lora_A") or k.endswith(".lora_B")]
   unexpected_missing = [k for k in missing if k not in allowed_missing]
   if unexpected_missing:
       raise RuntimeError(f"V18 resume: unexpected missing keys: {unexpected_missing[:5]}")
   ```

---

## §5 — 三位 reviewer 提的其他改进（暂未做，但记录）

| 来源 | 建议 | 优先级 | 状态 |
|---|---|---|---|
| agent1 | V20 = latent-residual refiner R(z_pred) | low | 备选，未起草 |
| agent2 | 训练时把 decoder/KL/img 拆三项独立记录 | medium | 实施时考虑 |
| agent2 | KL 用三档 (kl_gt / img_pred / 可选 kl_pred) | medium | 暂用 agent3 单一 use_pred_latent=true |
| agent3 | V21 conv head（RAE 已有）| high | **触发后再写** yaml |
| agent3 | last_n_blocks=2 可能限制太多，sweep {2, 4, 6} | medium | V18 首跑用 last_n=2，partial-success 后再 sweep |
| 三人 | train_first_hop.py 缺 `--dry-run` | medium | codex 在 Day 1 implementation 时加 |

---

## §6 — V18 vs V21 vs V14 launch 决策树（Day 2 GPU 排程）

| Day 0 probe 结果 | Day 2 GPU 0 | Day 2 GPU 1 |
|---|---|---|
| V18_WRONG_DIRECTION (gap<2dB) | **V21 (conv head)** ~2 day | V14 (true d_pure) ~7 day |
| V18_LAUNCH_RANK8 (gap 2-5dB) | **V18 rank=8 + KL fix** ~7 day | V14 ~7 day |
| V18_LAUNCH_RANK32 (gap≥5dB) | **V18 rank=32 + KL fix** ~7 day | V14 ~7 day |

V14 不受 gap-decomposition 影响 —— 它的目的是测真 d_pure，所有后续 SNR 判定都需要。无条件 launch。

---

## §7 — 元教训

agent3 的 5th bias 概括最准：

> **B5 = "I read 3 files = I understand the chain"**。读了 `rae.py` + `decoder.py` + `lora.py` 三个文件，但 **stage 2 实际跑的 train_lora_dinov2.py 没读完**，于是 V18 EV 被夸大。
>
> 整套 5 次 bias 共同形态：**用单个观察量当 ground truth，跳过分解**。
> - B1: ablation Δ → action
> - B2: absolute level → action
> - B3: single-step PSNR → action
> - B4: null measurement → more diagnostics
> - **B5: 11.2 dB gap → V18 EV 0.5-2 dB**

修复方式：**在任何决定之前，强制做最便宜的分解实验**。Round 4 三人独立提出的 1-hour gap decomposition 就是这个分解。

**纪律**：从今往后任何"基于单一数字做的训练决定"必须先 commit 一个 < 1 GPU-hour 的分解 probe + 预注册决策树。Round 4 的 probe 是这条纪律的第一个实例。

---

## §8 — 已 commit + push 的工件

| 文件 | 状态 |
|---|---|
| [tools/probe_v18_gap_decomposition.py](../../tools/probe_v18_gap_decomposition.py) | 新增，~280 行，syntax 检查通过 |
| [review/0517/V18_decoder_lora/V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) | KL block 修正（λ 0.5→0.05、use_pred_latent true、warmup 2000、kill_switch）|
| [review/0517/REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) | 本文件 |
| [review/0517/CODEX_RUNBOOK_V18_20260517.md](./CODEX_RUNBOOK_V18_20260517.md) | 待加 Phase 0 + binary decision rule |
| V21 yaml + V21 runbook | **待写**，触发条件 = gap<2dB |

---

## §9 — 用户决策点

1. 接受这个整合方案，让 codex Day 0 跑 probe → 按决策树决定 V18 / V21？（推荐）
2. 还想发 Round 5 prompt？（**不推荐** — 三位 reviewer 都说停止 review，做实验）
3. 暂停所有 GPU，先做更深 codebase audit？（不推荐 — 已经有 5 次同型 bias，Day 0 probe 才是正解）
