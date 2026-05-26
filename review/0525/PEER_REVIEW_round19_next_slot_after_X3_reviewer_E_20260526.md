# PEER REVIEW (Reviewer E) — Round19 Next Slot After X3

- date: 2026-05-26
- reviewer: E
- scope: after X3 non-additive result, decide whether to use 1 freed training slot while X1-lite is still running
- base prompt: PEER_REVIEW_PROMPT_round19_next_slot_after_X3_20260526.md

---

## 0) Reviewer-E Bottom Line

结论先行：使用空出的 slot，但只允许 1 个 run，且只允许 Option A (A4-mid-seed1337)。

备选是 Option C (不启动新训练，直接等 X1-lite)。

Option B 现在启动属于前置过度反应，Option D 属于明显 sunk-cost 回流。

这不是为了“GPU 不闲着”，而是为了补齐当前 paper 最大脆弱点：headline 结果 A4-mid 仍是单 seed。

---

## 1) Code-and-Design Evidence Chain (硬证据，不靠叙事)

### E1. image_aux 组件权重确实来自 loss.image_aux，且直接进入训练损失

在训练代码中，image_aux 三项权重由配置读取并送入 image loss:
- l1_weight
- ssim_weight
- seam_weight

这意味着 A4-mid 与 X1-lite 的机制对比，在实现层面是可被清晰控制的，前提是 yaml 仅改预注册字段。

### E2. image_aux 的 lambda 调度由 training.image_aux 控制，A4-mid 固定 0.08 时是可复现实验设定

训练循环按 warmup/ramp 解析 training.image_aux，再在 step 级别应用 lambda_img。A4-mid 配置采用 lambda_start=lambda_max=0.08、warmup/ramp=0，本质是恒定强监督，不是随训练期变化的动态策略。

### E3. X3 的 KL 支路不是“默认生效”，而是需显式 enabled 且 lambda_kl>0

训练代码里 decoder_kl_pullback 只有在 loss.decoder_kl_pullback.enabled 为 true 时才进入分支，并按 lambda_kl 参与总损失。X3 设计报告宣称 lambda_kl=0 且关闭 pullback，因此 X3 的 non-additive 结论可解释为 image_aux+LoRA 短窗测试，不应被“隐性 KL 干扰”解释。

### E4. X3 的“160K->170K 短窗 warmstart”与代码 resume 语义一致

对于 decoder_lora_enabled 的 warm-start，代码会保留 checkpoint step 继续计步，而不是重置为 step 0。结合 X3 报告中的 165K(best)/170K(last)与 full-val 输出，X3 结论属于实现一致、语义闭合的负结果。

### E5. LR total_steps_override 与 max_steps 解耦，seed replicate 必须锁定该字段

训练代码允许 lr_schedule.total_steps_override 与 training.max_steps 不同。A4-mid 使用 total_steps_override=200000、max_steps=160000。若 seed replicate 漏锁该字段，结果可因 LR 曲线变化而被污染，失去“仅改 seed”的解释性。

### E6. 关键现实问题：仓库当前看不到 X1-lite yaml 文件本体

在当前工作区中，X1-lite 路径由任务文档多次引用，但对应 yaml 文件未出现在可读路径下。这不否定 run 正在跑，但意味着“配置可审计性”不足。对 Round19 决策的影响是：更应优先做 A4-mid seed replicate 这种可定义、可锁字段、可快速审计的 run，而不是提前启动 X1-v2 这种依赖 X1-lite outcome 的条件实验。

---

## 2) Q1-Q7 逐条回答

### Q1 — 空 slot 是否应该使用？

- verdict: APPROVE (有条件使用)
- 理由:
  - 当前空档约 7 天，且 X1-lite 与 paper 可以并行。
  - 最高风险是 headline 单-seed，不是机制第二层歧义。
  - 只允许一个高信息增益 run；不是“看起来有空就跑”。

### Q2 — Option A 是否最高 EV？

- verdict: APPROVE
- 理由:
  - A4-mid 是当前主结论，单-seed 是审稿最直击的漏洞。
  - V14 证明的是 V7 seed-stability，不可外推到 A4-mid (目标函数权重结构不同，优化地形已变)。
  - A 的信息价值直接决定 paper 叙事可信度上限。

### Q3 — Option B 是否应提前跑？

- verdict: REJECT (现在不跑)
- 理由:
  - B 是条件实验，仅在 X1-lite 落入 interior 区间时才有必要。
  - 提前跑 B 属于 CL1 overreaction，会触发“先开分支后找理由”。
  - 治理上也更接近 no-small-tuning spirit 的边界，争议高于 A。

### Q4 — Option C 是否过于保守？

- verdict: MODIFY
- 理由:
  - 作为主方案偏保守，会放弃一个可直接补齐 headline 证据链的窗口。
  - 作为备选合理：当运维负担或稳定性不足以保证 A 的干净执行时，立即降级为 C。

### Q5 — Option D 是否应直接排除？

- verdict: APPROVE (排除 D)
- 理由:
  - 已与 X3 hard-stop 和 Round18 治理方向冲突。
  - X3 已证短窗 non-additive，继续追 200K 本质上是 V18 sunk-cost 回流。
  - 即便小幅上升，仍未必触及 A4-mid，paper 价值低。

### Q6 — 是否存在更好 Option E？

- verdict: REJECT (无更优单-run E)
- 说明:
  - 在约束条件 (单 run、无外部数据、不可拖延 X1/paper、不搞小调参) 下，没有比 A 更高 EV 的训练选项。
  - 若硬要给 E：E 应是“无训练”工作包，不应是新模型变体。

### Q7 — 偏差审查 (B97+)

- verdict: APPROVE (可控，但必须写入 stop rule)
- 偏差结论:
  - A4-seed fetish: 中风险。控制手段是仅 1 次 replicate，不展开多-seed campaign。
  - CL1 overreaction: 中高风险。控制手段是 X1-v2 必须等待 X1-lite outcome gate。
  - V18 sunk-cost: 高风险。控制手段是 D 明确禁行。
  - slot-utilization bias: 中风险。控制手段是“按信息增益排序，而非按 GPU 空闲排序”。

---

## 3) 主 Verdict (单选 + 备选)

- Primary: A (A4-mid-seed1337)
- Backup: C (No new training, wait for X1-lite)

切换到 C 的触发条件:
- 未来 24 小时内无法保证 run 启动与监控质量
- 或 paper 关键写作节点出现资源冲突

否则执行 A。

---

## 4) If choose a run: Minimal Design Spec (严格最小改动)

Base config:
- review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml

只允许改 4-6 个字段:
1. output_dir -> 新隔离目录 (Round19)
2. run_name -> 新 run 标识
3. seed: 42 -> 1337
4. training.require_fresh_output_dir: 保持 true (锁)
5. training.max_steps: 保持 160000 (锁)
6. lr_schedule.total_steps_override: 保持 200000 (锁)

必须保持不变 (否则实验无效):
- training.image_aux.lambda_start = 0.08
- training.image_aux.lambda_max = 0.08
- loss.image_aux.l1_weight = 1.0
- loss.image_aux.ssim_weight = 0.25
- loss.image_aux.seam_weight = 0.10
- freeze_rae = true
- decoder_lora 相关字段不存在或 disabled
- decoder_kl_pullback 不启用

Hard stop:
- 只跑 1 个 seed replicate
- 禁止追加 seed sweep
- 禁止借机追加 lambda sweep / X1-v2 / X3-extend / V18-family continuation

---

## 5) If choose no run: Idle Slot Action Pack

若切换到 C，本周必须完成:
1. paper outline 与 Results 主表框架
2. 预写 A4 seed 三段解释模板:
   - close match (<=0.02)
   - partial drop (>0.02 且 >V7+0.05)
   - collapse to V7-range
3. 预写 X1-lite interior 情况下的 CL1 limitation 语句
4. X1-lite 监控与 full-val 后处理脚本清单

---

## 6) 新偏差定义 (B97+)

- B97 Headline Fragility Blindness
  - 把单-seed headline 当作已稳健结论
- B98 Conditional-Run Premature Trigger
  - 在 gate 未触发时提前启动 X1-v2
- B99 Governance Drift by Idle-GPU Rationalization
  - 用“GPU 不能闲”替代“证据优先级”
- B100 Resume-Family Resurrection
  - 在 stop-rule 后用 extend/v2 话术复活 V18/X3 线
- B101 Config-Audit Gap
  - 运行中实验缺少可追溯 yaml 快照，导致结论审计困难

---

## 7) Reviewer-E Final Statement

Round19 的问题不是“还想不想再跑一次”，而是“哪一个单次 run 可以最大幅度降低论文被击穿概率”。

答案是 A4-mid-seed1337，不是 X1-v2，不是 X3-extend。

执行 A，严格 one-run hard stop；并行推进 paper。若执行质量无法保障，则切换 C，而不是切换到任何探索性替代 run。
