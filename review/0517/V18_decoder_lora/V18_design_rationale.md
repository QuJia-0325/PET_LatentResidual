# V18 Design Rationale — Why RAE Decoder LoRA

- date: 2026-05-17
- companion to: [CODEX_RUNBOOK_V18_20260517.md](../CODEX_RUNBOOK_V18_20260517.md), [V18_decoder_lora.yaml](./V18_decoder_lora.yaml), [pet_lr/decoder_lora.py](../../../pet_lr/decoder_lora.py)
- supersedes / retracts: [REVIEW_INTEGRATION_20260516.md §7 V9/V11'/V13/V14 priority queue](../../0516/REVIEW_INTEGRATION_20260516.md)

> 这份文件只回答一个问题：**为什么 V18 (RAE 解码器 LoRA) 是当前项目状态下唯一值得花 7 GPU-day 的训练**。所有更早的方向（V9 / V11 / V11' / V13 / V15 / V17 / V17'）的撤回理由见 [DIAGNOSTIC_FINDING_20260517.md](../DIAGNOSTIC_FINDING_20260517.md)。

---

## §1 — 三条不可绕开的事实

### 事实 1：V7 image_aux 路径已经饱和

来自 V7 训练日志 [grep](../DIAGNOSTIC_FINDING_20260517.md#1-v7-image_aux-路径已经饱和m1-数据)：

| 窗口 | val_hop0_img_total |
|---|---:|
| step 30K-60K | 0.00772 |
| step 130K-160K | 0.00751 |
| **末/早比** | **1.028（仅 2.7% 改善）** |

含义：任何在 image_aux loss 形状上的改动（V11/V11'/V17/V17'/V15）期望增益 ≤ 2.7% × 已饱和路径残余，远 < d_pure。

### 事实 2：transport gap 是 11.2 dB

| | 值 |
|---|---:|
| V7 chain D20 PSNR_clip3 (full-val n=7403) | 35.435 dB |
| RAE decoder ceiling on D20 (`PSNR(decode(encode(x_D20)), x_D20)`) | ~46.64 dB |
| **transport gap** | **~11.2 dB** |

含义：transport 模型（latent dynamics + image_aux）在过去半年只吃掉了 RAE encoder-decoder 能力的 76%。**剩 24% 是 decoder 的非等距性 + 高 SUV 区域的 Jacobian 病态**，与 latent transport 无关。

### 事实 3：decoder 被项目自己锁死

[train_first_hop.py:1298-1302](../../../train_first_hop.py#L1298)：

```python
if not bool(train_cfg_boot.get("freeze_rae", True)):
    raise RuntimeError(
        "docs/main.md requires training.freeze_rae=true for first-hop training. "
        "Decoder unfreezing is not allowed in this trainer."
    )
```

[model_first_hop.py:432-436](../../../pet_lr/model_first_hop.py#L432) `assert_decoder_frozen()` 在循环中 enforce。

含义：**11.2 dB 的最大单一可控变量被两行硬 assert 锁住**，6 天 review 链没人想到去撤掉它。

---

## §2 — V18 设计：最小可逆改动撬动 decoder

### 2.1 设计哲学

**所有改动可逆 + 步 0 输出与 V7 完全相同 + 默认安全**：

| 设计选择 | 为什么 |
|---|---|
| LoRA rank=8，alpha=16 (scale=2.0) | 业界默认；trainable params << decoder 总 params (~0.1%)；保留 base decoder weight，LoRA 是 add-on |
| init_scale_zero=True (LoRA B 零初始化) | B@A=0 at step 0 → V18 step 0 输出与 V7 best.pt 完全相同。第一个 sanity check 就是 "V18@step0 PSNR == V7 PSNR" |
| 仅 last 2 blocks LoRA | 越靠近输出层，影响越局部；image space 失败模式（高 SUV underestimation）大概率出在最后几层 |
| LR 5e-7 (decoder_lr_mult=0.00625) | 比 backbone 8e-5 低 160×；让 LoRA 在 40K step 内最多移动 ~0.1% 的 decoder output |
| KL pull-back loss (λ_kl=0.5) | MSE(decode_lora(z_gt) - decode_frozen(z_gt))；防 decoder 在 GT manifold 上漂走 |
| warm-start from V7 best.pt + 40K finetune | 不重训 160K；只在已经训好的 backbone 之上微调 decoder 末层 |
| max_steps 160K → 200K | 总训练资源 = V7 已用 + 40K 增量 |

### 2.2 V18 与所有撤回候选的 EV 对比

| 实验 | 期望 ΔNORMAL PSNR_clip3 vs V7 | 训练成本 | 期望 EV/GPU-hour |
|---|---:|---:|---:|
| V11 / V11' / V17 / V17' / V15 | 0.02 - 0.10 dB（image_aux 饱和路径残余） | 7 天 × 1-3 个 | 极低 |
| V9b (β=2.5 + step_weights 重派) | 0 - 0.05 dB (selector 偏置 + hop3 退化) | 7 天 | 极低 |
| V13 (true image_aux ablation) | 信息价值（验证 image_aux 贡献），非 PSNR | 7 天 | 中（backward-looking）|
| V14 (true d_pure) | 0（design 是测 noise） | 7 天 | 中（前置条件） |
| **V18 (decoder LoRA)** | **0.5 - 2.0 dB**（11.2 dB gap 的 5-20%） | **7 天 + 0.5 天 Phase A** | **HIGH（10-40× 其他候选）** |

### 2.3 V18 失败的 4 个具体可能 + 应对

| 失败模式 | 触发信号 | Day 9 应对 |
|---|---|---|
| LoRA 容量太小 (rank=8 不够) | ΔPSNR ∈ [+0.05, +0.30] | 跑 V18-r16 / V18-r32 sweep |
| LoRA 解锁位置不对 (last 2 blocks 不够) | ΔPSNR ∈ [+0.05, +0.30] + KL drift 小 | 跑 V18-last4 / V18-all-blocks 比较 |
| KL pull-back 太强压制了适应 | KL drift = 0 但 ΔPSNR ≈ 0 | 跑 V18-no-KL / V18-kl=0.1 |
| Decoder 真的没空间（11.2 dB gap 是 latent 端） | ΔPSNR < +0.05 + KL drift > 0.05 | 撤回 V18 整体方向，进入 backbone/data/architecture |

---

## §3 — 为什么不直接全解冻 decoder？

**第一原则：reversibility**。LoRA 的最大优势是 "失败了删掉 LoRA params 就回 V7"。全解冻：

- 改动 decoder 所有参数 → forward pass 全程改变 → 与 V7 已对齐的 latent encoder 端不再 compatible
- Catastrophic forgetting 风险：RAE decoder 是大量 OpenPET 数据训出来的，在我们的 PET 切片上仅 40K finetune 不足以替代原 generalization
- 无 fallback：如果 V18 失败，全解冻产生的 ckpt 不能 partial recover；LoRA ckpt 可以丢掉 lora_A/lora_B 回退

**第二原则：trainable params 占比**。LoRA r=8 on 2 blocks of typical 1024-dim 8-layer ViT decoder：

```
per block trainable: 6 Linear (q/k/v/o/fc1/fc2) × 2 (LoRA A + B) × rank × in_or_out_dim
   ≈ 6 × 2 × 8 × 1024 ≈ 98K params
2 blocks total: ~200K params
backbone trainable: ~50M params
ratio: 0.4%
```

意味着即便 LoRA 学坏，对 backbone gradient 的扰动很小。

**第三原则：可解释性**。LoRA 失败的 forensics 容易：`||lora_A||` / `||lora_B||` / `||delta_output||` 都是有意义的量。全解冻则只能 diff 整个 state_dict。

---

## §4 — V18 与 paper narrative 的关系

V7-only 论文叙事可能的卖点：
- "Multi-hop Grönwall closed-form for step_weights"（理论 + 实测对齐 < 1%）
- "Hop0 pixel forcing for first-hop bottleneck"（V7 vs V8 +0.3 dB）

V18 加进来后的卖点（如果 ΔPSNR ≥ +0.30 dB）：
- "Two-stage transport: latent stage + decoder LoRA adaptation"
- "Frozen decoder is a soft constraint, not a hard ceiling"
- "Per-domain LoRA adaptation enables transport-pretrained backbones to specialize without forgetting"

后两条**只有跑了 V18 才能拿到**。这是 V18 在写作端的价值，独立于 PSNR 数字。

---

## §5 — 不可逾越的红线

1. **V18 init 必须可被 sanity-check**。Day 1 完成检查必须包括 "V18 step 0 输出 = V7 best.pt 输出"（zero-init LoRA 保证）。如果不成立，绝对不要 launch。
2. **预注册阈值不可事后改**。§2.3 的 4 个 outcome 写死，不允许在 Day 9 看到结果后调整。
3. **不允许再发 peer review**。Round 1/2/3 共 3 千行 markdown 已经够，再发就是 [DIAGNOSTIC_FINDING_20260517.md §5](../DIAGNOSTIC_FINDING_20260517.md) 描述的 review-meta-review 永动机。
4. **Day 9 评估是 binary**。SUCCESS → 进 V18 variants；PARTIAL/NULL → 改 sweep；REGRESSION → 完全撤回。中间不允许"或许我们再训一个看看"。

---

## §6 — Day 0/1/2 给 codex 的最短行动

按 [CODEX_RUNBOOK_V18_20260517.md](../CODEX_RUNBOOK_V18_20260517.md) §3-§5 执行。要点：

- Day 0：Phase A inference-time probe + V7 中间 ckpt 存活检查。≤ 1 小时。
- Day 1：7 个具体代码改动（runbook §4.2 A-G） + Day 1 完成判定（§4.3）。≤ 1 天。
- Day 2：V18 + V14 同时 launch。≤ 1 小时。
- Day 2-9：训练跑，论文 outline 写起。

**Hard timebox: Day 2 必须有 V18 在 GPU 上跑**。如果 Day 1 任何代码改造遇到 > 1 天障碍，撤回 LoRA-r8 改为最简版："撤 assert + 整个 last block full unfreeze + LR 5e-8 + 同 KL"。**不要再升一轮 review**。
