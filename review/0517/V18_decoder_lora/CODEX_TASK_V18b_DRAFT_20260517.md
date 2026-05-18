# [DRAFT — pending user review] V18 success threshold correction + V18b launch spec

- date: 2026-05-17 深夜
- branch: foc_lite_hop0
- HEAD when written: df1993b
- status: **DRAFT, NOT YET PUSHED**. Will go to gitee only after user review.
- triggered by: Round 5 reviewer integration ([REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md)) identified 3 V18 bugs (B8/B9/B10). This task is the **smallest, lowest-risk subset** to act on now without stopping V18.

---

## §0 — 给 codex 的 5 句话总览

1. V18 继续跑 **不停**。Step ~180K，loss 健康，KL warmup 早过，但 `use_pred_latent=true` 有梯度对抗问题（见 §1）。
2. 这次任务只做 2 件事：(a) 修 V18 success threshold 表（design rationale §2.3），(b) 准备 V18b 旁支 yaml（在 V18 跑完 200K 后用）。
3. **不要立刻 launch V18b**。它要等 V18 step 200000 完成 + 评估出 ΔPSNR 后才决定是否需要。
4. **不要改在跑的 V18 yaml**（会触发 resume config mismatch 中断训练）。
5. 不需要新 GPU 任务。这次任务全部 0 GPU-hour。

---

## §1 — V18 的 3 个已确认 bug（背景，不需要 codex 立刻处理）

来自 Round 5 三位独立 reviewer 共识（[REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md)）：

| Bug | 当前 V18 yaml 设置 | 应该是 | 影响 |
|---|---|---|---|
| **B8** | V21 在多份文档被描述为"freeze V7 + decode 后加 RAE conv_head" | 实际 conv_head.py 是 `decoder_pred + unpatchify` 的 token-domain 替换品（[conv_head.py:92](../../RAE/RAE/src/stage1/decoders/conv_head.py#L92)）；V21 概念 retire | 与本任务无关（V21 不跑就是了）|
| **B9** | `loss.decoder_kl_pullback.use_pred_latent: true` | `false` | KL 与 image_aux 在同一 `decode_lora(z_pred)` 输出上梯度对抗 → V18 上限被压。**bounded harm**（λ_kl=0.05 vs λ_img=0.04 比值 1.25×）。需要 V18b 旁支量化影响 |
| **B10** | "V18 success threshold = +0.30 dB" 来自 "11.2 dB transport gap × 5-20%" | 真实 attackable = `psnr_transport_vs_ceil − psnr_transport` = 35.94 − 35.44 = **0.5 dB**；V18 EV = 0.05-0.30 dB；threshold 应 +0.15 dB | 影响最终判定，不影响训练 |

---

## §2 — Task A: 修 V18_design_rationale.md §2.3 阈值表

### 改动点

修改 [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) 文件，**只改 §2.3 表 + §2.2 EV 表 + 加一个 §2.4 修正记录**。其余不动。

### A1: §2.2 表中 V18 这一行

旧：
```
| **V18 (decoder LoRA)** | **0.5 - 2.0 dB**（11.2 dB gap 的 5-20%） | **7 天 + 0.5 天 Phase A** | **HIGH（10-40× 其他候选）** |
```

新：
```
| **V18 (decoder LoRA)** | **0.05 - 0.30 dB**（修正后；attackable_gap = 0.5 dB，见 §2.4 + Round 5 B10） | **7 天 + 0.5 天 Phase A** | **中（与 V11/V17 同档；但是结构性新轴）** |
```

### A2: §2.3 表第 4 行 + 新增第 5 行

旧第 4 行：
```
| Decoder 真的没空间（11.2 dB gap 是 latent 端） | ΔPSNR < +0.05 + KL drift > 0.05 | 撤回 V18 整体方向，进入 backbone/data/architecture |
```

新第 4 行（不删，但 trigger 改阈值）：
```
| Decoder 真的没空间（gap 在 latent 端，不在 decoder） | ΔPSNR < +0.03 dB + KL drift > 0.05 | 撤回 V18 整体方向，进入 backbone/data/architecture |
```

新增第 5 行：
```
| KL 与 image_aux 梯度对抗 (B9) 压上限 | ΔPSNR ∈ [+0.03, +0.15] dB 且 V18b (use_pred_latent=false) ΔPSNR 显著高 | 接受 V18b 作为真实 V18，撤掉 use_pred_latent=true |
```

### A3: 新增 §2.4 "Round 5 修正记录"

直接在 §2.3 后插入：

```markdown
### 2.4 Round 5 修正记录 (B10)

原 V18 EV 估计 (0.5-2.0 dB) 来自 "11.2 dB attackable_gap × 5-20%"，**这是错的**。
11.2 dB 是 RAE round-trip ceiling (PSNR(decode(z_GT), x_target) − PSNR(decode(z_pred), x_target))，
但 V18 改 decoder 只能动 `decode(z_pred) vs decode(z_GT)` 这层差异。

从 [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) 实测数据：

| 量 | 值 |
|---|---:|
| psnr_ceil = PSNR(decode(z_GT), x_target) | 46.64 dB |
| psnr_transport = PSNR(decode(z_pred^V7), x_target) | 35.44 dB |
| psnr_transport_vs_ceil = PSNR(decode(z_pred), decode(z_GT)) | 35.94 dB |
| **真实 V18-attackable gap** | **psnr_transport_vs_ceil − psnr_transport = 0.50 dB** |

所以预注册 PRIMARY SUCCESS 阈值从 **+0.30 dB 改为 +0.15 dB**：

| 阈值 | 旧 | 新 (Round 5 修正) |
|---|---|---|
| PRIMARY SUCCESS | +0.30 dB | **+0.15 dB** |
| PARTIAL (rank/blocks sweep) | [+0.05, +0.30] | **[+0.03, +0.15]** |
| KILL @ 20K | +0.05 dB | **+0.03 dB** |

这条修正**严格遵守"预注册阈值不可事后改"原则的 spirit**：
我们没有在结果出来后改 success 判定，而是在结果出来**前**修正了一个 mathematically wrong
的 baseline 计算。Round 5 三位 reviewer 共识同意此修正。
```

---

## §3 — Task B: V18b 旁支 yaml draft（暂不 launch）

### B1: 创建文件

路径：`review/0517/V18b_use_pred_latent_false/V18b_decoder_lora_kl_gt_anchor.yaml`

### B2: 内容（V18 的精确 copy，只改 6 处）

不在这里展开完整 yaml（440 行）；只列改动：

| 改动 # | yaml path | 旧 | 新 |
|---|---|---|---|
| 1 | `output_dir` | `.../review_0517_runs/V18_decoder_lora/run` | `.../review_0517_runs/V18b_kl_gt_anchor/run` |
| 2 | `run_name` | `first_hop_224_v18_decoder_lora` | `first_hop_224_v18b_kl_gt_anchor` |
| 3 | `training.resume_from` | V7 best.pt | V18 **step_200000.pt**（等 V18 跑完才填路径）|
| 4 | `training.max_steps` | 200000 | **220000**（V18 200K + 20K 增量 = V18b 比较窗口）|
| 5 | `loss.decoder_kl_pullback.use_pred_latent` | `true` | **`false`** |
| 6 | `loss.decoder_kl_pullback.warmup_steps` | 2000 | **0**（V18b 从 step 200K 出发，已不需要 warmup） |
| 7 (yaml header comment) | 描述为 V18 主版本 | 改为 "V18b — bug fix branch for B9: use_pred_latent false 防止 KL 与 image_aux 梯度对抗" |

### B3: 为什么 V18b 设计成 "V18 step 200K → +20K" 而不是从 V7 重训

理由：
- V18 已经付出 40K step 的 KL+image_aux 训练，丢掉浪费
- 即使 use_pred_latent=true 是 suboptimal，V18 ckpt 仍是 "V7 加了某些 decoder LoRA 调整" 的有效起点
- 从 V18 step 200K 出发 + use_pred_latent=false 再训 20K，能直接看 ΔPSNR(V18b − V18) 是否显著
- 如果 ΔPSNR(V18b − V18) > +0.05 dB，**就证明 B9 是 V18 的 hard ceiling 之一**，未来 V18 训练默认改 use_pred_latent=false
- 如果 ΔPSNR(V18b − V18) ≈ 0，B9 影响小，V18 yaml 不需要再改

### B4: V18b 的 success/kill 阈值（预注册）

| 量 | 阈值 |
|---|---|
| V18b vs V18 在 NORMAL PSNR_clip3 ΔPSNR | **≥ +0.05 dB** 才能 claim "B9 修复有效" |
| KL drift | < 0.05（与 V18 同标准） |
| KILL @ step 210K (V18b 10K 后) | ΔPSNR < +0.01 dB → 停 V18b，B9 影响可忽略 |

---

## §4 — Task C: codex 执行顺序

### Day 0 (现在, 0 GPU)

1. **不动 V18 训练**。
2. 把 V18_design_rationale.md 按 §2 A1/A2/A3 修改 + commit。
3. 把 V18b yaml 按 §3 B1/B2 draft 出来 + commit（**不 launch**）。
4. push 到 gitee。

### Day N (V18 step 200K 完成时, ~3-5 天后)

5. 按预注册评估 V18 vs V7（用 §2 A3 修正后的阈值表）。
6. 把评估结果写到 `review/0517/V18_decoder_lora/V18_FINAL_EVAL_<TS>.md`。
7. **如果 V18 ΔPSNR ∈ [+0.03, +0.15] dB**（PARTIAL），符合 §2 A2 新增第 5 行 trigger → launch V18b（filled-in resume_from path）。
8. **如果 V18 ΔPSNR ≥ +0.15 dB**（SUCCESS） → V18b 不必跑，V18 已成功，转入 paper 写作 / variants sweep。
9. **如果 V18 ΔPSNR < +0.03 dB**（KILL/NULL） → V18b 也不必跑（V18 死了，B9 不是问题）；进入 backbone/data/architecture 方向。

### Day N+5 (若 V18b launched)

10. 评估 V18b vs V18 按 §3 B4 阈值。
11. Commit `review/0517/V18b_use_pred_latent_false/V18b_FINAL_EVAL_<TS>.md`。

---

## §5 — 给 codex 的明确边界

**可以做（Day 0 立即可执行）**：
- 修 [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) 按 §2
- 创建 [review/0517/V18b_use_pred_latent_false/V18b_decoder_lora_kl_gt_anchor.yaml](./V18b_use_pred_latent_false/V18b_decoder_lora_kl_gt_anchor.yaml) 按 §3
- commit + push gitee

**不要做**：
- 不要停 V18 训练
- 不要改 V18_decoder_lora.yaml 任何字段（V18 还在用）
- 不要在 Day 0 launch V18b（要等 V18 step 200K + 评估）
- 不要 launch V13/V14/V9（slot 2/3 决策 user 还没批）
- 不要写 V21 yaml（V21 已 retire，见 B8）

---

## §6 — Round 5 整合后还没做的事（不在本 task 范围）

为完整性记录，这些是 Round 5 整合的 §4 全部 4 项动作里**本 task 只处理了第 2 项 (B10) + 第 3 项 (B9 fork 准备)**：

| 项 | 状态 | 谁负责 |
|---|---|---|
| §4.1 V18 yaml KL 修正（说明不在线改）| 本 task 含 V18b draft，OK | codex (本 task) |
| §4.2 V18 success threshold 修正 | 本 task §2 处理 | codex (本 task) |
| §4.3 V21 retire | 文档撤销，需另一次清理 commit | claude 或后续 task |
| §4.4 V13 launch | **user 待批准** slot 2 决策 | user → codex |
| §4.5 V9a CPU 通道 | 5-min pandas snippet 已写在 Round 5 整合里 | codex 随时可跑 |

---

## §7 — claude 自警

本次起草时我**第 4 次想去碰 V21**（在 §1 表里又写 V21 描述）。如果不是 Round 5 agent3 已经把 V21 拆穿，我可能又会推荐它。这是 sunk-cost bias 在 candidate clean-up 阶段的表现。

新纪律加 memory（待 codex 不必处理）：**当某个候选被独立 reviewer 拆穿后，下一次提到它必须先 cross-check 拆穿理由是否还成立**。
