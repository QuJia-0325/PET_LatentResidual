# Peer Review Round 4 — V18 design under domain-adaptation chain

- date: 2026-05-17 晚
- branch: foc_lite_hop0
- HEAD when written: c96e6dd (will bump after this commit)
- purpose: **Round 4** — 用户提示 RAE 不只是个黑盒，而是 "DINOv2 在自然图像预训练 → 用户用 PET 训练集对 encoder-decoder 做 LoRA → 我现在想再对 decoder 加 V18 LoRA"。**这个跨域 + 双层 LoRA 结构 claude 在前 3 轮 review 完全没考虑**。请评估 V18 设计在这个真实背景下是否还成立。
- target reviewers: 2-3 个 AI 独立评审，**互不见草稿**
- **审稿人执行环境**：数据/checkpoint 在服务器，reviewer 只能凭 git artifacts 评估

> Round 1/2/3 的 claude 偏差记录见 §0。Round 3 共识让 claude 决定推 V18 (RAE decoder LoRA)；今天读 RAE 源码后又发现 RAE 自带 LoRA + encoder 已 LoRA-tune。**用户进一步提示**让 claude 意识到现在面对的不是 "frozen decoder 加 LoRA" 而是 "已经 LoRA 适配过的 encoder-decoder 之上再加 LoRA"。**新的盲区可能更深**。

---

## 0. Round 1/2/3 + 今日修订时间线

| 时点 | 关键决定 | 后续被证伪 |
|---|---|---|
| Round 1 (5/16) | V7-V8 = 纯 image_aux ablation；推 V11 multi-hop image_aux | NO — V7/V8 step_weights 也不同（agent1） |
| Round 1 整合 | 撤 V11，推 V17 (ROI weighting) | NO — D20 top-1% mask 用 noisy x_D20（agent2） |
| Round 2 整合 | 撤 V17，推 5 P0 + Round 3 prompt | NO — diagnostic laundering（agent3） |
| Round 3 整合 | 撤 5 P0，推 V18 (RAE decoder LoRA) | ? |
| **5/17 晚 #1** | 读 RAE 源码：decoder 是 vanilla ViT-MAE，attribute path 改为 rae.decoder.decoder_layers，复用 RAE 自带 LinearWithLoRA | 部分修正 |
| **5/17 晚 #2（本 prompt）** | 用户指出 encoder 已 LoRA-tune，需要 peer review | ? |

claude 4 次 confirmation bias 的形态：B1 = ablation delta → action；B2 = absolute level → action；B3 = diagnostic signal → action；B4 = avoidance → diagnostic laundering。**B5 风险**：用 "我读了源码" 当 over-confidence 理由，但读到的仍是局部事实，不是全局领域适配理解。

---

## 1. 给 reviewer 的 5 个新问题

### Q1 — Domain adaptation chain 的累积影响

完整链条：
1. DINOv2 在自然图像上预训练（ImageNet-22K 等）
2. 用户用 PET 训练集 finetune DINOv2 encoder + ViT-MAE decoder，**通过 LoRA** —— 这是一个完整 stage1，输出是 `pet_lora_dinov2_pt_224/best_model.pt`
3. V6/V7 在这个 LoRA-tuned RAE 之上训练 backbone（transport flow），RAE 全程 frozen
4. **V18 提议**：在 V7 best.pt 之上再 LoRA fine-tune RAE decoder 的 last 2 layers

**请评估**：
- Stage 2 (用户的 RAE LoRA) **已经做过** 一次 domain adapt PET → 它的目的就是让 RAE 能解码 PET latent。V18 (Stage 4) 想再做一次 decoder adapt 的边际价值是什么？
- 如果 stage 2 已经把 RAE decoder 推到 PET 数据的局部最优，V18 在 V7 backbone 的 latent 分布上再 LoRA 是否在双重 adapt 上叠加（forgetting risk + capacity 浪费）？
- 反过来：如果 stage 2 的 LoRA 容量不够（lora_rank=16，2024 RAE convention），V18 加新 LoRA 反而是把 stage 2 没解决的 PET-domain 容量补上 — 这种情况下 V18 该用更大的 rank（32 / 64）而不是 rank=8？

### Q2 — "11.2 dB transport gap" 的口径是否成立

我的 V18 论证依赖：
> V7 chain D20 PSNR_clip3 = 35.4 dB；RAE decoder ceiling on D20 ≈ 46.64 dB；transport gap = 11.2 dB 可挖。

但 "RAE decoder ceiling 46.64 dB" 来自 [docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md](../../docs/experiment_plans/PLAN_V5_PostE1E2_SchemeCA_20260417.md)，那是用 stage 2 LoRA-tuned RAE 做 `PSNR(decode(encode(x_D20)), x_D20)` 测的。

**请评估**：
- 这个 ceiling 是 **encoder + decoder 一起** 的能力上限（即 round-trip reconstruction error），不是单独 decoder 的能力上限。V18 只动 decoder 不动 encoder，能否真的逼近这个 ceiling？
- 如果 transport 模型输出的 z_pred 与 GT z_D20 在 latent 空间已经差距很小（V7 pair MSE 已经 < 1e-6），那 decoder LoRA 改善的不是 transport 误差，而是 decoder 自己的 reconstruction 误差。这两件事可能在 latent 空间没重叠。**V18 是在治错病**？
- 真正合理的 ceiling 应该是 `PSNR(decode(z_pred), x_D20)` 而不是 `PSNR(decode(encode(x_D20)), x_D20)`。前者只有训练完才能测，后者是先验上限。两者差距可能 < 11.2 dB。

### Q3 — 双层 LoRA 的数学语义

RAE encoder 的 LinearWithLoRA wrap：
```
W_encoder_effective = W_dino_pretrained + (B_stage2 @ A_stage2) * (alpha_stage2 / rank_stage2)
```

V18 decoder 的 LinearWithLoRA wrap 加在 vanilla ViT-MAE decoder 上：
```
W_decoder_effective = W_vit_mae_pretrained + (B_v18 @ A_v18) * (alpha_v18 / rank_v18)
```

两层独立，没有真"叠加"。但**功能上**：
- Stage 2 LoRA 已经把 RAE encoder + decoder **一起** 训过 PET 数据
- V18 假设 stage 2 没把 decoder 训到位 → 再 LoRA decoder 一次

**请评估**：
- Stage 2 时 decoder 是怎么训的？如果是 full fine-tune（[RAE/RAE/src/train_stage1.py:289](../../../RAE/RAE/src/train_stage1.py#L289) `rae.decoder.requires_grad_(True)`），decoder 已经在 PET 数据上 full 训过，V18 LoRA 只是在 full-tuned 权重上再加增量 — rank=8 很可能不够拿到新增益。
- 如果 stage 2 用的是 ViT-MAE 默认初始化（没 PET pre-train，直接 random init + full finetune 160K），decoder 容量约束在 ckpt 文件里。V18 LoRA 不解锁 ckpt 里固化的 prior，只在 forward path 上加调整 — 收益边界很难预估。
- 想清楚 stage 2 decoder 实际权重的"来源"：HF ViT-MAE checkpoint (pretrained on ImageNet)？scratch init？这关系到 V18 该不该用 LoRA 还是该重训 decoder。

### Q4 — V18 init 的 "step 0 == V7" 保证是否会被 stage 2 LoRA 破坏

我的 V18 设计依赖 `LinearWithLoRA` zero-init B → 步 0 输出 == base 输出。但：

- V7 best.pt 里的 decoder 权重 = **HF ViT-MAE pretrained weights + 用户 stage 2 full finetune 后的 delta**（如果是 full finetune），或者 = HF ViT-MAE pretrained + stage 2 LoRA delta（如果是 LoRA）。**保存格式取决于 stage 2 训练脚本**。
- V18 加载 V7 best.pt 时 `strict=False`，会装载 stage 2 给的 decoder 权重。然后 V18 LoRA wrap **包裹 stage 2 finetuned 后的 Linear**。
- 这没问题，但如果 stage 2 decoder 是 LoRA 形式存的（即 V7 ckpt 里有 `rae.decoder.*.lora_A/lora_B`），那 V18 wrap 的 base.linear 就是**还没合并 LoRA 的 base** — 步 0 输出会 != V7 best.pt 输出！

**请评估**：
- Codex Day 0 必须 `print(type(rae.decoder.decoder_layers[-1].attention.attention.query))` 确认是 `nn.Linear` 还是 `LinearWithLoRA`。如果是后者，V18 wrap 时要先 `merge_lora()` 把 stage 2 LoRA 合并到 base.linear.weight，再 wrap V18 的 LoRA。
- pet_lr/decoder_lora.py 当前**没有这个 merge 步骤**。这是潜在 bug。

### Q5 — KL pull-back loss 的参考目标是否对

V18 的 KL pull-back：
```
L_kl = MSE(decode_lora_v18(z_gt) - decode_frozen_v7(z_gt))
```

其中 `decode_frozen_v7` 是 "V7 best.pt 时的 RAE decoder 输出"。

**请评估**：
- KL pull-back 的目的是**防止 decoder 漂走**。但漂走的参考是什么？
  - (a) HF ViT-MAE 原始 decoder（natural image 域）
  - (b) Stage 2 RAE-tuned decoder（PET reconstruction 域，用户 LoRA finetune 结果）
  - (c) V7 best.pt 时的 decoder = stage 2 + V7 训练里 RAE 没动过 = (b)
- 我的实现是 (c)，但 (c) = (b)，对应**保留 PET reconstruction quality**。这看起来对。
- 但**反直觉的视角**：如果 V18 想让 decoder 适应 V7 backbone 输出的 z_pred 分布（与 GT z_D20 略不同），KL pull-back 反而**阻止** 这种适应。是否应该用 use_pred_latent=True，把 KL 改成 `MSE(decode_lora(z_pred) - decode_frozen(z_pred))` —— 只惩罚预测路径的漂移而不是 GT 路径的漂移？
- 当前 V18 yaml `use_pred_latent: false`。这个默认是否正确？

---

## 2. 资料目录

按读的顺序：

### 2.1 V18 核心设计与本轮发现

- [REVIEW_INTEGRATION_20260516.md](../0516/REVIEW_INTEGRATION_20260516.md) — Round 1 V7/V8 confounder 修正
- [DIAGNOSTIC_FINDING_20260517.md](./DIAGNOSTIC_FINDING_20260517.md) — Round 3 反思 + V18 提出
- [CODEX_RUNBOOK_V18_20260517.md](./CODEX_RUNBOOK_V18_20260517.md) — V18 实施手册
- [V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) — V18 训练配置
- [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) — V18 设计动机
- **[RAE_ARCHITECTURE_FINDING_20260517.md](./V18_decoder_lora/RAE_ARCHITECTURE_FINDING_20260517.md)** — 今天读 RAE 源码后修订（attribute path / target_keywords / 复用 LinearWithLoRA）

### 2.2 RAE 源码（reviewer 必须读，本地仓库就有）

- [RAE/RAE/src/stage1/rae.py](../../../RAE/RAE/src/stage1/rae.py) — RAE class 定义
- [RAE/RAE/src/stage1/decoders/decoder.py](../../../RAE/RAE/src/stage1/decoders/decoder.py) — GeneralDecoder + ViTMAELayer
- [RAE/RAE/src/utils/lora.py](../../../RAE/RAE/src/utils/lora.py) — LinearWithLoRA + inject_lora_into_dinov2_attention
- [RAE/RAE/src/pet_flow/inference_pet_flow.py](../../../RAE/RAE/src/pet_flow/inference_pet_flow.py) — load_rae_model (装 encoder LoRA)
- **stage 2 训练脚本**（关键，reviewer 必读以确认 decoder 是 full finetune 还是 LoRA）：
  - [RAE/RAE/src/train_stage1.py](../../../RAE/RAE/src/train_stage1.py) — 看 line 289 `rae.decoder.requires_grad_(True)`
  - [RAE/RAE/src/train_lora_dinov2.py](../../../RAE/RAE/src/train_lora_dinov2.py)
  - [RAE/RAE/src/train_lora_dinov2_clipfree.py](../../../RAE/RAE/src/train_lora_dinov2_clipfree.py)
  - [RAE/RAE/src/train_lora_dinov2_overlap.py](../../../RAE/RAE/src/train_lora_dinov2_overlap.py)
  - **请确认用户实际跑的是哪一个**，对应的 decoder 权重保存格式是什么

### 2.3 V18 代码

- [pet_lr/decoder_lora.py](../../../pet_lr/decoder_lora.py) — V18 LoRA wrap + KL pull-back（今天修订后版本）

### 2.4 V7 训练数据

- [V7_config.resolved.yaml](../0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml) — 含 `rae:` 块（encoder LoRA 配置）和 `rae.checkpoint_path: /data_2/.../pet_lora_dinov2_pt_224/best_model.pt`
- [V7 training metrics summary](../0511/log_snapshots_20260516_163900/status/training_metrics_summary_20260516_163900.json)

### 2.5 历史相关 review

- [review/0430/reviewer/literature_architecture_image_aux_review_20260501.md](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) — 2024+ PET 文献

---

## 3. Reviewer 输出格式

```
Q1 domain-adaptation verdict: [V18 worth doing / V18 redundant / V18 wrong-direction]
Q1 reasoning: <2-4 sentences>

Q2 11.2 dB gap reality:
  ceiling_metric_correct: [yes / no / partial]
  V18_can_close_gap: [yes / no / partial]
  reasoning: <2-4 sentences>

Q3 two-layer LoRA semantics:
  stage_2_decoder_form: [full-finetune / LoRA / scratch / unknown — codex must verify]
  V18_rank=8_appropriate: [yes / no / cant-tell-until-codex-confirms]
  reasoning: <2-4 sentences>

Q4 step-0-equivalence guarantee:
  holds_in_practice: [yes / no / depends-on-stage-2-form]
  code_change_needed: <if no, what specifically>

Q5 KL pullback target:
  current_design_correct: [yes / no]
  recommendation: <use_pred_latent? change reference? remove KL?>

Meta-bias check:
  is_claude_committing_5th_bias: [yes / no]
  if_yes: <what bias, specifically>
  if_no: <what makes this round different from previous 4>

Overall recommendation:
  next_concrete_action: <single command-level item codex should do on Day 0>
  V18_launch_blocked_by: <if any>
  alternative_to_V18: <if V18 should not launch, what should>
  confidence: [low / medium / high]
```

---

## 4. Claude 自己未消除的偏见（坦白）

1. **"读了 RAE 源码 → V18 设计稳了" 的 over-confidence**：我只读了 `rae.py` + `decoder.py` + `lora.py` 三个文件 + 部分 stage 2 train scripts 的开头。**没有完整看 stage 2 训练流程**：data pipeline、loss、schedule、validation 协议都不清楚。这意味着我不知道 stage 2 给 V7 留下的 decoder 在 PET 域到底有多少未利用容量。

2. **V18 设计预设 "decoder 是瓶颈"**：基于 chain PSNR 35.4 vs RAE ceiling 46.6 = 11.2 dB gap。但这个 gap 中有多少是 transport 模型贡献的（V7 backbone 输出 z_pred 与 GT z 在 latent 空间的差距），多少是 decoder 本身在 PET 域 reconstruction 误差，**我没分解**。

3. **"复用 RAE LinearWithLoRA → 实现稳了" 的 over-confidence**：忽略了 LinearWithLoRA wrap 一个**已经被 LoRA wrap 过**的 Linear 时的语义。Codex Day 0 必须 print 类型才能确认 V18 wrap 的是真 nn.Linear 还是 stage 2 留下的 LinearWithLoRA。

4. **"V18 vs V11/V17 = 10× EV"** 这种比较有问题：V11/V17 的 expected gain 是测出来的（V7-V8 +0.3 dB 量级），V18 的 expected gain 是**纯外推**（11.2 dB gap 的 5-20% = 0.5-2.0 dB）。外推数字不该当作 ground truth 用于 EV 比较。

---

## 5. 请尽量对抗式批判

我倾向于 "被指出第 5 种解读 / 第 6 个盲点" 胜过 "被告知 V18 设计合理"。**特别欢迎**：
- 指出 V18 应该被推迟（先做某个零训练的实验）
- 指出 V18 该改为 V18'（不同 rank / 不同位置 / 不同 KL 形式）
- 指出整个 image_aux + transport + decoder 路径都该被替换为别的方向
