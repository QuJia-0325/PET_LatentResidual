# Peer Review Round 18-Prep — X1-lite + X3 Codex Task Deep Code Review

- date: 2026-05-25
- branch: foc_lite_hop0 (commit a6f7e32)
- 主审对象: [CODEX_TASK_ROUND18_X1_X3_20260525.md](./CODEX_TASK_ROUND18_X1_X3_20260525.md)
- 上游: Round 18 集成 ([REVIEW_INTEGRATION_round18_20260525.md](./REVIEW_INTEGRATION_round18_20260525.md)) + Round 18 战略分析 ([ROUND_18_A4_BRACKET_ANALYSIS_20260525.md](./ROUND_18_A4_BRACKET_ANALYSIS_20260525.md))
- 触发: claude 已起草 X1-lite + X3 codex task md 并 push (commit a6f7e32). user 要求 push 给 codex 执行**之前**做 AI peer review, 防止 Round 17-Prep B75-B82 同型 yaml/CLI/path bug 重复发生
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 18 (战略) 已签 = Hybrid X1-lite + X3 + 立即 paper draft.
Round 18 集成 / Round 18 分析 已签 = X1-lite (mechanism falsification) + X3 (additive test).
本轮**不再审战略选择**, 只审 codex task md 的**代码 + 设计执行可行性**.

**本轮审**:
- yaml 6 字段 diff 是否真的覆盖所有需要改的字段, 不多不少
- trainer 代码路径 (`train_first_hop.py`) 在 X1-lite / X3 配置下是否真的产生预期行为
- LoRA 训练所需 optimizer / freeze_rae 等隐藏依赖是否被正确继承
- 比较协议 (X1-lite vs A4-mid, X3 vs A3, X3 vs V18) 是否真正"matched-step paired"
- shell 脚本是否有 race condition / GPU 冲突 / OOM 风险
- 早期停止 / 中间监控协议是否足以避免浪费 GPU
- NOT-DO 是否覆盖所有易踩坑场景

**本轮不审**:
- Round 18 战略选 X1/X3 vs X2/X4/X5 (4/4 已签)
- image_aux mechanism M1-M4 哪个对 (X1-lite 出来后再讨论)
- paper venue / 写作时间表 (Round 18 集成已签 MICCAI 2027)

---

## 1. 必读 substrate (按顺序)

### 1.1 本轮主审

- [CODEX_TASK_ROUND18_X1_X3_20260525.md](./CODEX_TASK_ROUND18_X1_X3_20260525.md) (本轮唯一主审, 476 行)

### 1.2 必须 grep / 读源码 (不能凭描述)

- `train_first_hop.py` 第 851-895 行 — `compute_hop0_image_losses` 函数, 验证 X1-lite 关闭 SSIM/seam 子项的真实行为
- `train_first_hop.py` 第 2215-2245 行 — KL pullback compute block, 验证 X3 的 `lambda_kl=0.0` 是否真完全跳过 (双重 guard)
- `train_first_hop.py` 第 1819-1830 行 — `loss_balance_watch_*` config 解析, 验证 X3 高 image_aux λ 是否会触发 watchdog kill
- `pet_lr/losses_first_hop.py` `compute_first_hop_image_loss` — 验证 w_ssim=0 / w_seam=0 是否真"语义等价"于关闭子项 (or 仍计算 then *0)
- `review/0517/V18_decoder_lora/V18_decoder_lora.yaml` 全文 — V18 24 字段全 list, 检查 X3 是否漏改某个 KL-related 字段
- `review/0505/local/configs/V7_gronwall_raw.yaml` 全文 — V7 baseline, 检查 X1-lite 是否漏改某个 image_aux-related 字段
- `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml` — A4-mid yaml, 对比 X1-lite (X1-lite vs A4-mid 应只差 ssim/seam weights)

### 1.3 已知历史 substrate (作为信号校准用)

| 实验 | image_aux λ | LoRA | KL | step | NORMAL | 来源 |
|---|---:|---|---:|---:|---:|---|
| V7.best | 0.04 | 无 | 0 | 160000 | 36.7810 | review/0511/.../planf_v7_best_*.json |
| V14.best | 0.04 | 无 | 0 | 160000 | 36.7806 | review/0516/full_eval_json/v14_*.json |
| V13.best | 0.00 | 无 | 0 | 160000 | 36.4943 | review/0516/full_eval_json/v13_*.json |
| A4-low.best | 0.02 | 无 | 0 | 160000 | 36.7010 | review/0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_SUMMARY_20260525.json |
| A4-mid.best | 0.08 | 无 | 0 | 160000 | 36.8939 | 同上 |
| V18.best | 0.04 | r32 last-2 + KL on | 0.05 | 165000 | 36.8112 | review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_*.json |
| V18.last | 0.04 | r32 last-2 + KL on | 0.05 | 200000 | 36.8426 | 同上 v18_last_*.json |
| A3 (V18-cap.last) | 0.04 | r32 last-2 (KL off) | 0.00 | 170000 | (≈ V18.step170k) | review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_SUMMARY.json |

---

## 2. 5 个 candidate issues (claude self-review 提出, 请 verify/refute/extend)

claude 已经做了一轮 self-review, 提出以下 5 个 candidate issue. 请逐条 verify (grep 源码 + 读 yaml + 重算), 给 verdict.

### CI-1 (HIGH claim) — GPU 选择 race condition

**Claude 声明**: §3 B.2 staggered launch 用 `nvidia-smi sort -k2 -rn | head -1`. 如果 X1-lite 刚 launch 时 dataloader 未启动, GPU 0 仍显示 22GB free, X3 staggered 选择可能再次落在 GPU 0, 触发 OOM.

**请 verify**:
- 实际 X1-lite 启动 5min 后 nvidia-smi 上的 memory free 是多少 (PET 数据集 latents_train.pt 115GB, 加载需要 ~3 min, 内存峰值可能巨大)
- A4-mid smoke test 时 GPU 0 free memory 数字 (从 smoke log 可查)
- staggered +30min 时 X1-lite 是否已经稳定占用 ≥ 9 GB
- 是否 race condition 真存在, 还是 claude 过度担心

### CI-2 (MEDIUM claim) — LR schedule total_steps_override

**Claude 声明**: X3 yaml 继承 V18 的 `lr_schedule.total_steps_override: 200000` 不改, max_steps=170000. 这意味 LR 在 step 170K 时 cosine progress = (170-30)/(200-30) = 0.824, lr ≈ 2.4e-5. claude 认为这是"正确的, 与 V18.step170k 和 A3 一致".

**请 verify**:
- A3 yaml 是否真有 `total_steps_override: 200000` (grep A3 yaml)
- A3 训练 log 在 step 170K 时 logged LR 值 (grep A3 训练 log)
- 这个 LR 配置是否真的复现 V18.step170k 的优化状态 (A3 best.pt at step ~170K NORMAL chain 与 V18.step170k chain 是否真接近)
- 是否有更隐蔽的 LR 配置 (e.g. backbone lr mult, decoder lr mult) 在 X3 vs V18 vs A3 上有差异

### CI-3 (MEDIUM claim) — 早期监控缺失

**Claude 声明**: X1-lite 7d 训练, task md §A.2 监控表只检查 alive + step 进度, 没有 outcome 早停信号. 建议 +24h (step 30K) 与 V7 metrics.jsonl 同时点对比, 差 > 5e-5 NORMAL MSE 则早停.

**请 verify**:
- V7 metrics.jsonl 在 step 30000 的 rolling val NORMAL MSE 是什么 (实际查 V7 metrics 文件)
- 5e-5 阈值是否合理 (vs A4-mid 在 step 30K 时 rolling val NORMAL MSE, 应该已经显示 image_aux 强度差异)
- 早停协议是否会被误触发 (例如 PET dataloader 第一个 epoch 时 rolling val 噪声大)
- 是否应该用更稳健 metric (e.g. step 50K full-val 而不是 step 30K rolling val)

### CI-4 (LOW claim) — SSIM/seam 在 w=0 时仍计算

**Claude 声明**: `compute_first_hop_image_loss` 总是计算 SSIM 和 seam loss, w_ssim=0 / w_seam=0 只是在 total 里乘 0. 语义 ✓ 干净, 计算 ~2-5% overhead.

**请 verify**:
- 是否真的"梯度通过乘 0 完全不传" (例如 `loss = 0 * loss_ssim` 在 PyTorch 下是否真把 loss_ssim 的 grad 断开, 还是仍传播 0 梯度但占用 backward 时间)
- 是否有任何 conditional branch 在 w_ssim=0 时被特殊处理 (查 weighted_l1_loss / ssim_loss / seam_consistency_loss 实现)
- 是否有 issue: SSIM 计算本身可能修改 model state (BN stats / dropout RNG state), 即使 w=0 也会影响下一步训练
- 2-5% wall-clock overhead 估计是否准确

### CI-5 (LOW claim) — `decoder_kl_pullback.enabled: true` 留在 X3

**Claude 声明**: X3 inherits V18's `enabled: true`, 只把 lambda_kl 设 0. trainer 有 `if lambda_kl > 0.0:` 内层 guard, 所以安全. 但建议 cleanest 是同时 `enabled: false`.

**请 verify**:
- trainer 代码在 `enabled: true + lambda_kl=0.0` 时是否真的什么也不做 (verify line 2218-2231)
- 是否有任何 side effect (e.g. KL warmup 计算, KL log 写入 metrics jsonl) 即使 lambda_kl=0 也运行
- 是否有 model.rae freeze 状态被 KL config 改变 (verify freeze_rae 处理)

---

## 3. 给 reviewer 的 8 个深审问题

### Q1 — X1-lite yaml 6 字段 diff 是否真的够?

请独立列 V7 → X1-lite 应该改的字段, 与 task md §A.0 列出的 6 字段对比:

| 类别 | 字段 | 应改 | task md 改 | 一致? |
|---|---|---|---|---|
| 文件隔离 | output_dir | ✓ | ✓ | |
| 文件隔离 | run_name | ✓ | ✓ | |
| 功能变量 | training.image_aux.lambda_start | ✓ → 0.08 | ✓ | |
| 功能变量 | training.image_aux.lambda_max | ✓ → 0.08 | ✓ | |
| 子项关闭 | loss.image_aux.ssim_weight | ✓ → 0.0 | ✓ | |
| 子项关闭 | loss.image_aux.seam_weight | ✓ → 0.0 | ✓ | |
| 其它? | ??? | ??? | | |

**特别关注**:
- 是否需要也改 `loss.image_aux.border_weight` (border 仍参与 L1) — task md 没改
- 是否需要改 `loss.image_aux.seam_patch_size` — task md 没改
- 是否需要改 `loss.image_aux.use_extended_seam` — task md 没改
- 是否需要改 `training.image_aux.warmup_ratio` / `ramp_ratio` (V7 都是 0.0, X1-lite 保留)

### Q2 — X3 yaml 6 字段 diff 是否真的够?

类似 Q1, V18 → X3 6 字段:

| 类别 | 字段 | 应改 | task md 改 | 一致? |
|---|---|---|---|---|
| 文件隔离 | output_dir | ✓ | ✓ | |
| 文件隔离 | run_name | ✓ | ✓ | |
| 训练长度 | training.max_steps | 200000 → 170000 | ✓ | |
| 功能变量 | training.image_aux.lambda_start | 0.04 → 0.08 | ✓ | |
| 功能变量 | training.image_aux.lambda_max | 0.04 → 0.08 | ✓ | |
| 杀 KL | loss.decoder_kl_pullback.lambda_kl | 0.05 → 0.0 | ✓ | |
| 其它? | loss.decoder_kl_pullback.enabled | ??? | ✗ | |
| 其它? | loss.decoder_kl_pullback.use_pred_latent | ??? | ✗ | |
| 其它? | loss.decoder_kl_pullback.warmup_steps | ??? | ✗ | |
| 其它? | loss.decoder_kl_pullback.kill_switch.* | ??? | ✗ | |
| 其它? | optimizer.decoder_lr_mult | 必保留 (LoRA) | ✓ (继承不改) | |
| 其它? | optimizer.decoder_weight_decay | 必保留 (LoRA) | ✓ (继承不改) | |
| 其它? | training.freeze_rae | V18 是 false, 必保留 | ✓ (继承不改) | |

**特别关注**:
- V18 有 `kill_switch` 5 个字段, 这些是 doc-only 还是真被 trainer 消费 (grep 全 codebase)
- `decoder_kl_pullback.warmup_steps: 2000` 在 X3 (lambda_kl=0) 是否产生任何影响 (例如 log 字段 / metric 计算)
- LoRA-related fields 是否真的都被正确继承 (test by flatten + diff V7 vs X3, 看 LoRA fields 出现在 diff 中)

### Q3 — X1-lite vs A4-mid 真"matched"吗?

claude 声称 X1-lite vs A4-mid 是干净 paired 对比 (差 = SSIM + seam 子项贡献).

**请 verify**:
- X1-lite yaml clone V7, A4-mid yaml clone V7. 两者都应该 V7-based.
- A4-mid yaml 实际内容 = V7 + 4 字段 (output_dir, run_name, λ 两个). X1-lite = V7 + 6 字段 (上述 + ssim/seam = 0).
- **A4-mid 与 V7 在 ssim_weight / seam_weight 上是否真一致** (= V7 默认值 0.25 / 0.1)
- **若 A4-mid yaml 实际不是 V7-clone (e.g. base 不一样), X1-lite vs A4-mid 就含其它 confound**

### Q4 — X3 vs A3 真"matched"吗?

claude 声称 X3 vs A3 是最干净对比 (差 = image_aux λ 0.04 vs 0.08).

**请 verify**:
- A3 yaml = V18 + 4 字段 (output_dir, run_name, max_steps=170000, lambda_kl=0.0)
- X3 yaml = V18 + 6 字段 (上述 4 + image_aux λ × 2)
- **A3 与 V18 在 image_aux 上是否真一致** (= V18 默认 λ=0.04)
- **A3 yaml 实际跑出来的 A3.best/last 与 V18.step170k 是否真接近** (grep KL_DRIFT_SUMMARY 看 capacity-only chain MSE vs V18@170K chain MSE, A3 报告应该列了)
- **若 A3 与 V18.step170k 差异已经 > 0.02 dB, X3 vs A3 的 paired-comparison 假设破产**

### Q5 — Loss balance watchdog 是否会 kill X3?

X3 用 lambda_img=0.08 (是 V18 默认 0.04 的 2 倍). image_aux loss 的 weighted fraction 会显著上升.

**请 verify**:
- V18 训练 log 中 img_frac 的典型范围 (grep `img_frac=` 在 V18 log)
- A4-mid 训练 log (lambda_img=0.08, 无 LoRA) 中 img_frac (grep `img_frac=` 在 A4 smoke log 或 final log)
- X3 = LoRA + lambda_img=0.08, img_frac 预期更高还是更低
- `loss_balance_dominance_threshold: 0.85` + `loss_balance_watch_enforce: false` — watchdog 会 warn 但不 kill, 但是否 warn 会污染 user 判断
- 是否值得给 X3 显式 disable `loss_balance_watch_enabled: false` 或 raise threshold

### Q6 — 早停协议设计

claude 提议 +24h step 30K 时与 V7 trajectory 对比, 差 > 5e-5 NORMAL MSE 则早停.

**请 verify**:
- V7 metrics.jsonl @step 30000 的 rolling val NORMAL MSE 实际值 (grep)
- A4-mid metrics.jsonl @step 30000 同字段值 — 已知 A4-mid 最终好于 V7, 但在 30K 时是否已显示差异
- 5e-5 阈值 (≈ 0.1 dB NORMAL PSNR 差异) 是否过严或过松
- 是否应该用 **multi-objective val_select_score** 而非单 NORMAL MSE
- 早停触发后, 是 commit-and-stop 还是 kill 进程 (codex 协议要 explicit)

### Q7 — X3 用 10K 短训是否 underestimate LoRA 收益?

A3 用 10K 短训, V18 用 40K LoRA 训练. claude 说 X3 也用 10K, 与 A3 paired.

**请 verify**:
- V18.best vs V18.last NORMAL = 36.8112 vs 36.8426 = +0.0314 dB 在 165K → 200K (35K 训练) 内涨幅
- 按比例外推 V18 在 165K → 175K (10K) 内涨幅大约 +0.009 dB
- A3 用 10K 训练, 与 V18.step170k 在 chain 上几乎 tie. 说明 LoRA 10K 短训对 chain 影响微弱
- 若 X3 也只 10K LoRA 短训, **X3 outcome 可能 ≈ A4-mid** (无 LoRA 贡献), 但这是因为 10K 不够, 不是因为 V18 redundant
- 是否应该让 X3 训到 200K (40K LoRA 训练, 与 V18.last 同) 来真正测试? cost: 7d 而不是 1-2d
- 若不 extend, paper 应该如何 disclaim 这个 short-training limitation

### Q8 — NOT-DO 列表完整性

请检查 §7 NOT-DO 是否漏关键禁项:

- [ ] 禁止改 X1-lite 的 `loss.image_aux.l1_weight` (保留 V7 默认 1.0)
- [ ] 禁止改 X3 的 LoRA target_keywords / target_root / alpha / dropout / init_scale_zero
- [ ] 禁止改 X3 的 optimizer.decoder_lr_mult / decoder_weight_decay
- [ ] 禁止启 X1-lite v2 (ssim-only) / v3 (seam-only) 不论 X1-lite outcome
- [ ] 禁止 X3 extend to 200K (即使 outcome 中性) 除非 user round 19 explicit approval
- [ ] 禁止在 paper draft 阶段把 X3 outcome 当 V18 救活证据 (不论数字)

### Q9 — 新偏差 (B93+)

请检查本 prompt 是否引入:
- **claude self-review echo chamber**: 5 个 candidate issue 全是 claude 自己提的, 是否 prompt 引导 reviewer 只 verify 不发掘新问题
- **设计完美 anchor**: 本 prompt 默认 X1-lite + X3 是对的, 只检查执行, 是否漏了"实验设计本身有缺陷"层次
- **MICCAI 2027 时间压力 bias**: prompt 强调 paper draft 立即起, 是否让 reviewer 倾向"放过小问题让 codex 快跑"

---

## 4. 输出格式

每位 reviewer 独立产出 markdown:

### 4.1 5 个 candidate issue 逐条 (VERIFY / REFUTE / EXTEND)

对 CI-1 ~ CI-5, 给 verdict + grep 证据 + 严重性重评估.

### 4.2 9 个 deep 问题逐条 (APPROVE / MODIFY / REJECT)

### 4.3 主 verdict
- **READY (push to codex as-is)**: 所有 issue 都是 nice-to-have, codex 拿现稿能正确执行
- **MODIFY-BEFORE-PUSH**: 有 ≥ 1 项必改 (列出, 按 HIGH / MED severity)
- **REJECT (推翻重写)**: 实验设计有根本问题, 不能 push

### 4.4 必改项 (若 MODIFY)
按 (location, original wording, fix, severity) 列表.

### 4.5 设计层关注 (D-class, 不必 HIGH 但战略上重要)

包括但不限于:
- X1-lite outcome 模糊 (≪ A4-mid 但 > V7) 时 paper 怎么写
- X3 short-training (10K) limitation 怎么避免被读者质疑
- X1-lite 和 X3 都 single-seed (Round 17-Stats 已签 slice-level only)

### 4.6 新偏差 (B93+)

---

## 5. 角色提示

你不在审实验是否值得做 (Round 18 4/4 已签 X1-lite + X3). 你在审**已起草的 codex task md 是否能 push**:

- 错过 yaml field bug → codex 训 7 天后发现实验设计破产, 浪费 1 张 GPU × 7d
- 错过 KL kill_switch bug → X3 中途被 kill, 浪费 1 张 GPU × 12h + 重启延迟
- 错过 LR schedule bug → X3 outcome 与 V18.step170k 不可 paired-比较, paper 必填 disclaimer
- 错过 GPU race condition → 一个 GPU OOM 两个实验同时挂

请极其严格. 30 min reviewer 必须能 trade off 1 张 7d GPU + reviewer 阅读时间.

如果你认为现稿 READY, 直接说; 不要为"稳妥"列 5 个 nice-to-have 让 user 决策疲劳.
如果你认为现稿有 HIGH bug, 直接 MODIFY-BEFORE-PUSH 并说哪行哪字段.
