# Peer Review Round 8 — Phase A codex task draft 审稿

- date: 2026-05-17 深夜
- branch: foc_lite_hop0 (草稿, 未 push)
- 主审对象:
  1. **[CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md)** (主, codex 可执行 task md)
  2. [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](./NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) (上游架构设计, 阶段 A 部分; 阶段 B/C 不审)
- target reviewers: 2-3 个 AI 独立评审, 不互见草稿
- 评审目标: 批准 / 修改 / 否决 codex task md, 然后决定是否 push gitee 交给 codex 执行
- 硬约束声明 (standing): **服务器内存最多 3 个并行训练任务**, slot 1 = V18 在跑 (不可动), 本 task 最多新增 slot 2 = V13

---

## 0. 本轮范围与边界

Round 1-5 审"该做什么实验", Round 6 审"V18b draft 是否过关", Round 7 审"NEXT_STAGE 设计是否合理", **Round 8 是首次审"给 codex 的可执行 task 文档本身"** — 关注点从 design 转 execution.

**本轮只审**:
- Phase A task md 是否清楚、安全、可执行
- pass/fail 准则是否覆盖关键失败模式
- NOT-DO 列表是否完整 (codex 越界风险)
- 失败决策树是否避免 codex 自行 escalate

**本轮不审**:
- V18 设计 / V13 设计 (Round 1-5 已穷尽)
- 三阶段路线 / 3×3 决策矩阵 / V18-clean threshold (Round 7 user 已签字)
- dual-track 阈值 / SECONDARY 列 / step 180K early-eval (Round 7 已 reject)

---

## 1. 给 reviewer 的 7 个问题

### Q1 — V13 smoke override 路径风险

[CODEX_TASK_PHASE_A §3.2](./CODEX_TASK_PHASE_A_20260517.md) 使用 `--max-steps-override` + `--output-dir-override` CLI flag.

请评估:
- 这两个 flag 是否真存在于 [train_first_hop.py](PET_LatentResidual/train_first_hop.py)? 请 grep 验证
- 若不存在, task md 给的 fallback ("cp 原 yaml + sed 改") 是否安全 (会不会污染原 yaml)?
- 是否应改用 yaml override 文件 (e.g. `V13_smoke.yaml` 单独保存) 而不是 CLI override?
- smoke run dir 是否应排除在 git tracking 外 (避免 commit smoke ckpt 浪费空间)?

### Q2 — V13 smoke pass 准则的代码事实校准

§3.5 列了 6 个 pass 准则. 请代码层 verify:

- **准则 2** (`image_aux.enabled = false` log 显示): train_first_hop.py 是否真的会 print effective config? 用什么关键词 grep? 若日志格式不同, 准则会 false-fail
- **准则 4** (`loss_img = 0.0`): metrics.jsonl 字段名是 `loss_img` 还是 `img_loss` 还是 `loss_image_aux` 还是其他? 与 grep 模式是否一致?
- **准则 5** (throughput ≤ V7 baseline × 1.1): "V7 baseline 200 step 时间" 来源是什么? 是否有历史 log 可对照?
- **准则 6** (GPU memory): 从 log 抓还是从 nvidia-smi history? V7 baseline memory 数字哪来?

若准则的 grep pattern 与实际 log 格式不匹配, codex 会出现 silent pass-as-fail 或 fail-as-pass.

### Q3 — Task B V21 retire grep 策略是否完备

§2 (Task B) 让 codex 用 `grep V21` 找 4 文档里所有引用. 但:

- "V21" 字面可能出现在不是 "V21 fallback" 的上下文 (例如 "V21 → V22 transition note") — 这种情况应该加 retire 标记吗?
- 若某文档 V21 引用是**作为反例**或**作为历史决策记录** (e.g. "我们曾考虑 V21, 后 retire"), 重复加 retire 标记是冗余
- 是否应该让 codex 先列出**每一处** V21 grep 命中**给 user 看**, user 决定哪些加标记, 而不是 codex 自行决定?

### Q4 — NOT-DO 列表的可执行性

§6 列了 12 条 NOT-DO. 但 NOT-DO 难以执行验证:
- codex 是 AI agent, 它自己不会"主动想做" NOT-DO 项 — NOT-DO 主要防的是 LLM 滑坡推理
- 但 §7 失败决策树要求 codex 在 fail 时**停下报告**, 而 LLM 默认倾向 "尝试 fix" — 这是真实 NOT-DO 违反风险

请评估:
- §7 失败决策树是否覆盖了所有可能让 codex 想"自行 fix" 的场景?
- 是否应该在 task md **每个 task 末尾**重复一次 "fail → 停, 不 escalate", 而不是统一在 §7?
- 是否应该把 §6 NOT-DO 列表的关键项 **嵌入** 到每个 task 的开头, 让 codex 执行每 task 时都看到?

### Q5 — Push gitee 时机

§5.2 Task E 要 codex 在 V13 launch 后 ~5 min 就 push gitee. 但:
- V13 是 7 天训练, 5 min 后 push 时 V13 状态实际是 "刚启动, 未知是否收敛"
- 若 V13 在 push 之后 30 min OOM 死亡, push 的报告里 V13 状态是错的
- 是否应在 push 前**等 V13 跑过 1 个 eval interval** (~5000 step ≈ 4 h), 确认 V13 正常工作再 push?
- 反方: 不 push 拖延 4 h 也不好, A/B 文档清理工作没必要等

### Q6 — git commit 范围

§5.2 列了 7 类要 add 的文件. 但:
- `V13_train_*.log` 是 7 天滚动 log, push 时只有几行, 大部分内容会在未来产生 — 是否应该不 commit log file, 让 codex 在 V13 完成后单独 commit final log?
- `smoke_runs/smoke_*/smoke.pid` 是 PID 文件, commit 它有意义吗? (push 后 PID 在远程无效)
- `V13_train.pid` 同上

是否应该收窄 commit 范围到**只**:
- 4 份 doc 修订 (Task A/B 产出)
- PHASE_A_EXECUTION_REPORT (本 task 报告)
- smoke.log (verification artifact)
- 不 commit pid / 不 commit 未来还会写入的 log file?

### Q7 — codex 自警 §E1 报告模板是否充分?

§5.1 报告模板要求 codex 自查 B14/B18 复发. 但:
- B12 (mid-run 改阈值)、B19/B20 (KL ramp/max_steps 含义) 也是 Round 7 新加偏差, 报告模板没问
- 是否应该列**全部** Round 7 偏差让 codex 自查 (B12/B13/B14/B15/B16/B17/B18/B19/B20)?
- 反方: 太长的自查模板 codex 会敷衍, 不如只问 3 条最关键的

---

## 2. 资料目录

按读的顺序:

### 2.1 本轮主审对象

- **[CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md)** — codex 可执行 task md (本轮要 review)

### 2.2 上游设计 (本轮不审, 仅作 context)

- [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](./NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) — Round 7 user 签字的架构 (Phase A 范围对应本 task md)
- [REVIEW_INTEGRATION_round7_20260517.md](./REVIEW_INTEGRATION_round7_20260517.md) — Round 7 整合, user 决策 `全推荐` 记录

### 2.3 V13 配置 + 代码

- [V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) — V13 yaml
- [train_first_hop.py](../../train_first_hop.py) — 关键代码 (Q1/Q2 验证用)
  - L1770: `img_enabled = bool(image_aux_cfg.get("enabled", True))`
  - L2160-2161: `if img_enabled: img_losses = compute_hop0_image_losses(...)`
  - L2368: `if img_enabled: total_loss += ...`
- [V18_TRAIN_COMMAND_20260517.txt](./V18_decoder_lora/V18_TRAIN_COMMAND_20260517.txt) — V18 launch 命令实证 (V13 从这复制不需 --resume)

### 2.4 V21 retire 上游

- [REVIEW_INTEGRATION_round5_20260517.md](./REVIEW_INTEGRATION_round5_20260517.md) — B8 来源 (V21 描述与 conv_head.py 不符)
- [RAE/RAE/src/stage1/decoders/conv_head.py](../../../RAE/RAE/src/stage1/decoders/conv_head.py) — V21 概念错位的代码事实

### 2.5 V18 不动的硬约束

- [V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) — V18 配置 (本 task 完全不动)
- V18 当前 step ≈ 180K / 200K, ~12-18h 剩余

---

## 3. 输出格式 (请 reviewer 严格遵守)

每位 reviewer 独立产出 markdown:

### 3.1 7 个问题逐条回答
对 Q1-Q7 每条**明确**给出 1 个结论:
- `APPROVE` — 当前写法可接受
- `MODIFY (说明改什么)` — 需要修改但方向 OK
- `REJECT (说明为什么)` — 不可接受, 需重写

### 3.2 整体 verdict (对 [CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md))
- `READY TO PUSH` — 可直接 push 给 codex
- `MODIFY THEN PUSH` — 列出必改项后可 push
- `BLOCK` — 不可执行的设计/逻辑错误, 需重新起草

### 3.3 新偏差 / 新 bug
若你发现本 task draft 引入了**新的** confirmation bias (例如 claude 在写 task md 时是否又造新偏差), 或新代码 bug, 单独列出, 用 B21 / B22 ... 编号.

### 3.4 (可选) 代码层 spot-check
若你 verify 了 §3 准则的 grep pattern 与实际 log 格式不匹配, 给具体证据 (代码片段 + log 实例).

### 3.5 (可选) 替代方案
若 verdict = MODIFY/BLOCK, 给一份你认为更合理的 task md outline.

---

## 4. 约束与提醒

- **本轮不重审** V18/V13/V18-clean 设计、三阶段路线、3×3 决策矩阵 — Round 1-7 已结论
- **不假设** 你能跑代码 / 看 wandb / 触 GPU — 仅凭 git artifacts 评
- **特别关注**: 
  - codex 是 LLM agent, 倾向 "尝试 fix" 而非 "停下报告" — 你的 review 应该 stress-test 失败决策树
  - pass/fail 准则的 grep pattern 必须与实际代码 log 格式匹配, 否则 codex 会执行错误判定
  - NOT-DO 列表的真实风险 = LLM 滑坡推理, 不是 codex 主动违反
- **优先质疑**:
  - §3.2 `--max-steps-override` / `--output-dir-override` 是否存在
  - §3.5 准则 2/4 的 grep pattern 是否与日志匹配
  - §5.2 push 时机 (5 min 后 push vs 4h 等 eval interval)
  - §6 NOT-DO 列表是否覆盖 codex 真实滑坡风险

---

## 5. 给 reviewer 的硬约束摘要 (standing, 不可推荐违反)

任何 reviewer 提议都必须满足:

1. ≤ 3 并行训练任务 (本 task 已用 slot 1 = V18 + slot 2 = V13, slot 3 = null)
2. V18 不动 (slot 1 训练中, 任何 yaml/code 改动会触发 config mismatch)
3. V18b 永久撤销 (Round 6 共识 + 代码 hard blocker)
4. 阈值不改 (Round 7 user 签字 `全推荐` = agent3 不改路线)
5. 不预先 pre-register V18-clean (阶段 C 才 pre-register, Round 7 B18 教训)
6. 不起 Round 9 (本 task 自己就是 Round 8 终点)

违反任意一条 → reviewer 提议自动作废.
