# Peer Review Round 6 — V18b draft（threshold 修正 + use_pred_latent=false 旁支）审稿

- date: 2026-05-17 深夜
- branch: foc_lite_hop0（draft 仍在本地，未 push）
- 待审文件：[CODEX_TASK_V18b_DRAFT_20260517.md](./V18_decoder_lora/CODEX_TASK_V18b_DRAFT_20260517.md)
- target reviewers: 2-3 个 AI 独立评审，**不互见草稿**
- 评审目标：批准 / 修改 / 否决该 draft，再决定是否 push 到 gitee

> 前 5 轮 review 整合见 [REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md)。本轮**不重审 V18 设计、不重审 slot 分配**，只审 "V18b draft 这份给 codex 的指令文档本身是否过关"。

---

## 0. Round 6 的范围与边界

Round 5 三位 reviewer 共识识别出 V18 的 3 个 bug（B8 / B9 / B10）。claude 据此起草了 [CODEX_TASK_V18b_DRAFT_20260517.md](./V18_decoder_lora/CODEX_TASK_V18b_DRAFT_20260517.md)，**作用范围 = 处理 B9 + B10**（B8 = V21 retire，由后续清理 commit 处理）。

本轮 review 只评这份 draft：
- 它对 B9 / B10 的处理方式是否正确？
- 它新提出的预注册阈值修正是否站得住脚？
- 它新设计的 V18b 旁支 yaml（V18 step 200K → +20K, use_pred_latent=false）是否合理？
- 它给 codex 的执行顺序是否安全 / 清楚 / 完备？

**不在本轮范围**：
- V18 主训练是否该停（Round 5 已结论：不停）
- slot 2/3 GPU 分配（Round 5 已 user 决策中）
- V21 是否值得复活（Round 5 已 retire）

---

## 1. 给 reviewer 的 7 个问题

### Q1 — V18 success threshold 从 +0.30 → +0.15 dB 是否站得住脚？

draft §2.4 用 [GAP_DECOMP_REPORT](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) 的实测：
- psnr_transport_vs_ceil − psnr_transport = 35.94 − 35.44 = **0.50 dB** = V18 真实 attackable gap
- 因此 PRIMARY SUCCESS 从 +0.30 dB（11.2 dB × 5-20% 的错误算法）改为 +0.15 dB（0.5 dB × 30%）

请评估：
- 0.5 dB 作为 attackable gap 是否在数学上正确？（psnr_transport_vs_ceil 真的捕捉了 V18 LoRA 能动的全部空间吗？）
- "30% of attackable gap" 这个 0.30 比例本身是否有依据，还是又一次任意挑数？
- draft §2.4 自称"修正而非事后改阈值"是否成立？V18 还在跑，没出 ΔPSNR 结果 → **技术上是 ex-ante 修正**。但 0.5 dB 这个数字本身是 V7 best 评估后才知道的 → **算 ex-post**？
- 如果你认为这条修正违反预注册原则，请明确指出，并给替代方案（保持 +0.30 / 撤销 V18 / 重设计）

### Q2 — V18b 设计 "V18 step 200K → +20K with use_pred_latent=false" 是否合理？

draft §3 B3 论证：从 V18 ckpt 续训而非从 V7 重训。

请评估：
- 从 V18 step 200K 出发 + use_pred_latent=false 再训 20K，**ΔPSNR(V18b - V18) 真的能干净归因于 B9 修复吗**？
- 反方观点：V18 前 40K step 已经用 use_pred_latent=true 把 decoder LoRA 训成"某种适配 z_pred 的形态"。从这里切换到 anchor on z_GT，可能 LoRA 权重需要重新适配，**20K step 不够**，会低估 B9 修复价值
- 如果反方成立，正确做法是否应是 V18b 从 V7 best.pt 重训 200K（与 V18 配对）？
- 20K vs 200K 的训练成本差异 = 7 天 vs 0.7 天。是否值得为 cleaner attribution 付出？

### Q3 — V18b 阈值 "ΔPSNR(V18b vs V18) ≥ +0.05 dB" 是否合理？

draft §3 B4 设了 +0.05 dB 作为 "B9 修复有效" 阈值。

请评估：
- 0.05 dB 接近 d_pure 上界（0.02-0.03 dB）。**信噪比足够吗**？
- 若 V18 自身 ΔPSNR 只有 +0.08 dB（PARTIAL 区间），V18b 再 +0.05 dB → V18b 总 +0.13 dB（仍 PARTIAL）。**这种情况下 B9 修复有效但 V18 整体仍 NULL**，draft 怎么处理？没说清楚
- KILL @ step 210K (V18b 10K 后) ΔPSNR < +0.01 dB 是否过早？10K step 对 LoRA 适配可能不够

### Q4 — Task 排序（Day 0 改文档 vs Day N launch V18b）安全吗？

draft §4 把任务拆成两段：Day 0（0 GPU，改 design rationale + 准备 V18b yaml，不 launch），Day N（V18 跑完 200K 后再决定 launch V18b）。

请评估：
- Day 0 改 V18_design_rationale.md 时 V18 还在跑，会触发训练中断吗？**理论上不会**（rationale 文档不被训练读），但需要确认
- Day 0 写 V18b yaml 时，`training.resume_from` 填什么？draft §3 B2 说 "等 V18 跑完才填路径"，但 Day 0 commit 时该字段是 placeholder 还是空？如果是空 yaml 会被误 launch 吗？需要明确 sentinel（如 `__FILL_AFTER_V18_DONE__`）
- Day N step 9 "ΔPSNR < +0.03 → KILL/NULL → 进入 backbone/data/architecture 方向" — 这条决策权属于 user 还是 codex？draft 没说

### Q5 — draft 是否对 B8 (V21 retire) 处理充分？

draft §1 表里写 "B8 与本任务无关（V21 不跑就是了）"，但 §6 把 V21 retire 列为 "另一次清理 commit"。

请评估：
- V21 retire 的文档清理（[V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) §2.2 表里 V21 行、其他可能提及 V21 的地方）是否应该**合并进本 task**，避免两次 push？
- 留两次 commit 的风险：第二次清理可能被遗忘 → V21 描述错位的文档继续误导未来 reviewer

### Q6 — draft §7 "claude 自警" 这段是否该保留？

draft 末尾自认 "第 4 次想去碰 V21" 是 sunk-cost bias。

请评估：
- 这段对 codex 执行没有意义（codex 不需要知道 claude 的认知偏差史）
- 但作为 claude 的内部 metacognition 日志，提高未来 review 质量
- 是否应该 split：留 §7 在草稿里给 user review，**push 时删掉**只保留 §0-§6 给 codex？

### Q7 — 是否漏了什么 round 5 应该被处理的 bug？

draft 自称处理 B9 + B10，B8 由后续清理。但 Round 5 整合中是否还有其他 confirmation bias / 设计缺陷未被纳入本 draft？

请评估：
- 重新过一遍 [REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md)，列出**所有**应被本 draft 处理但被遗漏的项
- 特别检查：rank=32 选择、blocks=[6,7] 选择、lambda_kl=0.05 选择 这三个数值是否仍站得住脚？还是 Round 5 有人提过疑问？

---

## 2. 资料目录

按读的顺序：

### 2.1 本轮主审对象

- **[CODEX_TASK_V18b_DRAFT_20260517.md](./V18_decoder_lora/CODEX_TASK_V18b_DRAFT_20260517.md)** — 本轮要 review 的 draft

### 2.2 draft 引用的上游

- [REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md) — Round 5 三 reviewer 共识与 B8/B9/B10 来源
- [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) — draft 要修改的目标文档
- [V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) — V18 当前 yaml（draft 不改它）
- [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) — draft §2.4 的数字来源（0.5 dB attackable gap）

### 2.3 V18 当前状态

- [V18_EXECUTION_REPORT_20260517.md](./V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md) — V18 launch 报告
- [AUDIT_LORA_PARAM_COUNT_20260517.md](./V18_decoder_lora/AUDIT_LORA_PARAM_COUNT_20260517.md) — V18 实施验证（rank=32, 589,824 params 确认）
- 当前 V18 step ≈ 180K，距 max_steps=200K 约 20K step

### 2.4 V21 / conv_head 相关（用于评 Q5）

- [RAE/RAE/src/stage1/decoders/conv_head.py](../../../RAE/RAE/src/stage1/decoders/conv_head.py) — Round 5 B8 拆穿 V21 描述的根据

---

## 3. 输出格式（请 reviewer 严格遵守）

每位 reviewer 独立产出一份 markdown，包含：

### 3.1 7 个问题的逐条回答
对 Q1-Q7 每条**明确**给出 1 个结论：`APPROVE` / `MODIFY (说明改什么)` / `REJECT (说明为什么)`。

### 3.2 整体判断
对整份 [CODEX_TASK_V18b_DRAFT_20260517.md](./V18_decoder_lora/CODEX_TASK_V18b_DRAFT_20260517.md) 给一个 verdict：
- `READY TO PUSH`：可直接 push 到 gitee 给 codex 执行
- `MODIFY THEN PUSH`：列出必改项后可 push
- `BLOCK`：有不可接受的设计/逻辑错误，需重新起草

### 3.3 新 bias / 新 bug 发现
若你发现 draft 引入了新的 confirmation bias（claude 在 Round 5 修复 B9/B10 时是否又制造了新偏差）或新技术 bug，请单独列出。

### 3.4 (可选) 替代方案
若 verdict = MODIFY THEN PUSH 或 BLOCK，给一份你认为更合理的 draft outline（不必完整写）。

---

## 4. 约束与提醒

- **不要重审** V18 主设计、slot 分配、V21 是否复活 — 这些 Round 4/5 已结论
- **不要假设** 你能跑代码 / 看 wandb / 触 GPU — 只凭 git artifacts 评
- **关注** draft 的可执行性、归因 cleanliness、阈值 ex-ante vs ex-post 性质、commit 安全性
- **优先质疑** "0.5 dB × 30% = 0.15 dB" 这条算式 — 它是整个 draft 的逻辑支点
