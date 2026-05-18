# Peer Review Round 14 — A3 V18-capacity-only Task + A4 design_rationale §5.2 (post Round 13 user decision A1/A3/A4)

- date: 2026-05-18
- branch: foc_lite_hop0
- 主审对象 (3 个):
  1. **[V18_capacity_only.yaml](./V18_capacity_only/V18_capacity_only.yaml)** (A3 control 实验配置, 4 字段 diff vs V18)
  2. **[CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md](./CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md)** (codex 执行指令)
  3. **[V18_design_rationale.md §5.2](./V18_decoder_lora/V18_design_rationale.md)** (A4 新增 B43-B47 standing rules)
- 上游: Round 13 三 reviewer 共识 + user 决策 = A1 (V13 push slot 2) + A3 (V18-capacity-only launch slot 1) + A4 (B43-B47 standing rules)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**
- 目标: 验证 yaml diff + task md + standing rule 三者一致且无新偏差; 决定是否 push

---

## 0. 本轮范围与边界

Round 13 完成战略决策 (Stage C v3 = E 扩展版). Round 14 是 **execution prep review** 不是战略 review:
- 不再质疑 "是否该 launch V18-capacity-only"
- 不再质疑 "B42 mechanism reversal 是否真"
- 只审 v3 yaml + task md 的 **executability** + **standing rules 是否真正 propagate**

**本轮只审**:
- yaml diff 4 字段是否精确正确 (不动 V18 yaml, 不动 V7 ckpt)
- task md 是否真覆盖 B42 反驳 + capacity-only 设计逻辑 + 3 种可能结果解读
- A4 §5.2 B43-B47 standing rules 是否真在 design_rationale 落地 (不是 phantom rules)
- 是否引入 Round 14 新偏差 (B48+)

**本轮不审**:
- V13 设计 (Round 1-5)
- V18 原始设计 (Round 7 user 签字 + Round 12 buggy KL 已 flag B43)
- audit DRAFT release (与本 task 独立)

---

## 1. 给 reviewer 的 6 个问题

### Q1 — V18_capacity_only.yaml 4 字段 diff 是否精确?

期望仅 4 字段改 (其他全同 V18):
| 字段 | V18 | V18-capacity-only |
|---|---|---|
| output_dir | .../V18_decoder_lora/run | .../V18_capacity_only/run |
| run_name | first_hop_224_v18_decoder_lora | first_hop_224_v18_capacity_only |
| training.max_steps | 200000 | 170000 |
| loss.decoder_kl_pullback.lambda_kl | 0.05 | **0.0** |

请 reviewer 用 python yaml diff 验证 (claude 已实测 4 字段 ✓):
- 其他字段全同 V18? 特别: `training.decoder_lora.rank=32`, `training.decoder_lora.last_n_blocks=2`, `training.resume_from=V7 best.pt`, `lr_schedule.total_steps_override=200000` (cosine 同 V18 cosine 起点), `seed=42`, `loss.decoder_kl_pullback.use_pred_latent=true` (保留 V18 原值, 但 lambda_kl=0 使其 moot)
- 是否漏了应改但未改的字段 (e.g. `warmup_steps`, `kill_switch`, `eval_interval`)?
- **[NEW B1 高优先]** `loss.decoder_kl_pullback.kill_switch` 块 (check_step=5000, min_kl_change_pct=5.0) 被 capacity-only yaml 按 V18 原样继承. 但 train_first_hop.py:2230 上面 `if lambda_kl > 0.0:` 守门 → lambda_kl=0 时 `loss_kl` **从不计算**. watchdog 实现是否在 `lambda_kl == 0` 分支跳过 kill_switch 检查? 若未跳过 → step 5K 可能 spurious kill capacity-only run. reviewer 应 grep watchdog 代码确认, 或要求在 task md 加 grep verify 步骤.

### Q2 — capacity 是否真正"隔离" (control 设计是否真 control?)

V18-capacity-only 与 V18 唯一**计算路径**差异 = lambda_kl=0 (KL pullback 损失 = 0). 但:
- `use_pred_latent: true` 仍在 yaml (lambda_kl=0 使其 moot, 但代码路径仍走 z_pred lookup; 一旦 lambda_kl > 0 又激活)
- max_steps=170K vs 200K 减 30K step — 是否影响 capacity 表达 (LoRA 在 10K vs 40K step 上限不同)? V7 之上 10K step 是否足够 LoRA 表达 capacity?
- 10K step 短训 + V7 best.pt resume + 同 cosine schedule → V18-capacity-only step 170K 时 LR ≈ V18 step 170K 时 LR (cosine 同步) → ckpt comparability 是否成立?
- **[NEW B2 高优先]** V18 yaml `save_interval=10000` + `max_steps=200000` → server 上应有 V18@step_{10K,20K,...,200K}.pt 全集, 含 **V18@step_170000.pt** — 它是 V18-cap.last(170K) 的 bit-equivalent 同步训练阶段比较点. 但 task md §B1 probe 调用只传 V18.best(165K) 和 V18.last(200K), 未传 V18@170K → disambig 阈值 "≈ +0.10 dB → B 真" 模糊 (V18.best=+0.097, V18.last=+0.164, 中间差 0.07 dB 足够让 outcome 难判). reviewer 应要求§B1 加 `--v18-step170k-ckpt`, §B2 表加第 5 行.

替代选项:
- 10K step → 40K step (170K → 200K), 与 V18 训练步数完全等齐, 但成本 ~4× (~3-5 天 vs ~24-48h)
- 设计 yaml 更激进: 显式 `use_pred_latent: false` 即使 lambda_kl=0 (清理 dead field, 避免未来 reactivation)

### Q3 — task md §1 B42 + 3 候选解读是否准确传达?

task md §1.1 引 train_first_hop.py:2230 代码, §1.2 给 B 候选 (LoRA capacity), §1.3 3 种 outcome map. 

请 verify:
- §1.1 代码引用是否 verbatim (claude 已 grep, 与 reviewer3 R13 一致)?
- §1.3 三种 outcome 是否 cover 所有情况? 缺否第 4 种 (e.g. capacity-only 显著差于 V7, 即 LoRA capacity 反而**有害**)?
- task md 是否过度 anchor B 候选 (§1.2 写 "默认最简释"), 让 codex 自我说服 B 答案而忽视测出 A?

### Q4 — A4 §5.2 B43-B47 standing rules 是否真有 actionable 防御?

**[NEW Q4.0 — ship/revise 规则]** 若 reviewer 仅对 §5.2 措辞 (非机制错误) 有修订建议, **不阻塞 push**; 记录为 Round 15 单独 PR 修订. 仅当 reviewer 发现 §5.2 含 **机制错误** (如 B44 在 outcome 2 自证 false) 才阻塞.

**[NEW Q4.0b — 编号体系]** §5.2 header 称 "B8-B44 整理为 standing rules" 但实列 5 条 rule 编号 **B43-B47**, B43/B44 同时是源偏差编号和 rule 编号双义. 建议 rule 重编号 SR43-SR47 与源偏差脱钩, 或 header 明示.

**[NEW Q4.0c — B44 表述过强]** B44 现文 "V18 KL pullback 与 decode(z_GT) 路径机制隔离" + "机制上不可能" 过强. 精确事实: KL **直接**只在 z_pred 路径算 loss, 但其 gradient 仍更新共享 LoRA 参数 → eval 时 decode(z_GT) 用同一组 LoRA 权重 → KL 间接调控 z_GT manifold 仍可能. 若 capacity-only outcome 2 (≈ 0 dB) 出现, B44 现文会自证 false. 推荐改写: "V18 KL pullback 无直接 z_GT 路径, 仅通过共享 LoRA 参数间接耦合; V18 GT manifold 现象的解读必须区分 'KL 间接耦合' vs 'capacity 副产品'; capacity-only 控制是 disambig 必要手段."

§5.2 新增 5 条 standing rules:
- B43: V18 buggy KL config 标记 (任何 V18 数字必须 prefix "in V18 buggy KL config")
- B44: V18 KL 与 decode(z_GT) 路径机制隔离
- B45: substrate 阶段 reviewer 必须介入
- B46: 4×2=8 数字 comparison-label 规则
- B47: mechanical script must assert-on-fail

请 verify 每条:
- B43: 未来 V18 数字解读是否会主动加 prefix? 这条是文化约定还是 mechanical (e.g. 起 V18_FINAL_RESULTS 模板必须含 prefix)?
- B44: B42 mechanism 的精确描述; B45 / B46 / B47 是否在文档之外有 mechanical enforcement (script / CI / template)?
- 哪几条是文化约定 vs mechanical enforcement?

### Q5 — task md NOT-DO 是否完备?

§5 NOT-DO 9 条. 是否漏了:
- 不修改 V18-capacity-only yaml (本 task 已 ready, 不再改)
- 不在 capacity-only 训练期间 push 其他 unrelated commit
- 不在 probe 时改 V18 / V7 ckpt 路径常量
- 不复用 V18 训练 dir 的 ckpt 文件名空间 (output_dir 隔离已防, 但 codex 可能误 cp)?

### Q6 — 是否引入 Round 14 新偏差 (B48+)?

请独立检查:
- yaml header 注释 (lines 1-30) 是否含未 grep verify 的 historical claim? (e.g. 说 V18 KL pullback design 是某 Round 设计)
- task md §1.2 把 B 候选当 "默认最简释" 是否 anchor (与 §6 Q2 严防 anchor 矛盾)?
- A4 §5.2 standing rule B43 文字 "B43 — V18 use_pred_latent=true 是 buggy KL config" 中 "buggy" 是否过强标签 (Round 5 仅 flag 为 "潜在 gradient 对抗", 实际未实证)?
- design_rationale §5.1 第 3 条 "不允许再发 peer review" 被注释 "Round 5-13 已突破此红线 13 轮", 是否应**改为新 standing rule** 而非保留 + 加注释 (避免自相矛盾红线)?

如发现 B48+, 列编号 + 形态 + 修法.

---

## 2. 资料目录

按读的顺序:

### 2.1 本轮主审对象

- [V18_capacity_only.yaml](./V18_capacity_only/V18_capacity_only.yaml) (本轮主审 #1)
- [CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md](./CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md) (本轮主审 #2)
- [V18_design_rationale.md §5.2](./V18_decoder_lora/V18_design_rationale.md) (本轮主审 #3, A4 新增段)

### 2.2 上游 (本轮不审)

- [REVIEW_INTEGRATION_round13_20260518.md](./REVIEW_INTEGRATION_round13_20260518.md) (Round 13 整合, B42 来源)
- [V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) (V18 原配置, 本轮不动)
- [tools/probe_v18_kl_drift.py](../../tools/probe_v18_kl_drift.py) (probe, 本轮不审; task md §B1 用)

### 2.3 代码事实 verify

- [train_first_hop.py:2230](../../train_first_hop.py) `z_kl = main_out["z_pred"] if use_pred_latent else ...` (B42 reviewer3 grep)
- [pet_lr/decoder_lora.py:95, 123-130](../../pet_lr/decoder_lora.py) (init_scale_zero False silent ignored, B32 历史)

---

## 3. 输出格式

每位 reviewer 独立产出:

### 3.1 6 个问题逐条 (APPROVE / MODIFY / REJECT)

### 3.2 整体 verdict (3 对象)
- yaml: READY TO USE / MODIFY / REJECT
- task md: READY TO PUSH / MODIFY / REJECT
- §5.2 standing rules: READY / MODIFY / REJECT

### 3.3 推荐 push 顺序
- 选项 A: push yaml + task md + design_rationale §5.2 (一次 commit)
- 选项 B: push commit 7486776 先 (V13 launch + V21 retire 文档清理), 后 push A3 (隔离 commit)
- 选项 C: push A3 先, 之后再 push 7486776 (V13 可推迟一周)

### 3.4 新偏差识别 (B48+)

---

## 4. 约束与提醒

- **本轮不重审** Round 13 战略决策 (user 已签 A1/A3/A4)
- **特别关注**:
  - Q1 yaml 4 字段 diff 是否精确, 是否漏字段联动
  - Q2 capacity 是否真 "isolated" (lambda_kl=0 是必要不充分条件)
  - Q3 task md 是否过度 anchor B 候选让 codex 偏读
  - Q4 standing rules 是文化约定 vs mechanical (B43 最弱, B47 最强)
  - Q6 新偏差: claude 在 substrate review 阶段 anchor 倾向 (Round 12/13 已证)
- **不假设** 你能跑代码 — 但可读 yaml / py / md grep verify

---

## 5. Standing constraints

1. ≤ 3 并行训练任务 (capacity-only 占 slot 1 + V13 占 slot 2 = 2/3)
2. V18 不动
3. V18b 永久撤销
4. 阈值不改
5. user 决策 (R7/R8/R9/R10/R12/R13) 不可推翻
6. Round 14 是 A3 execution prep, 不是战略
7. capacity-only 完成后 (~24-48h) 必起 Round 15 战略 review (B 候选 / A 候选 / 混合 outcome 的 Stage C v4 决策, 见 task md §B2 三种 outcome 解读)
