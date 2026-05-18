# Peer Review Round 10 — Phase A v3 + Code Deep Audit DRAFT 审稿

- date: 2026-05-18
- branch: foc_lite_hop0 (v3 + audit DRAFT 仍未 push)
- 主审对象 (2 份):
  1. **[CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md)** (Round 9 选 C 修订, 7 处 ~47 行 vs v2)
  2. **[CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md](./CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md)** (阶段 A 完成后才发的 audit task DRAFT)
- 上游: [REVIEW_INTEGRATION_round9_20260517.md](./REVIEW_INTEGRATION_round9_20260517.md) (user 决策 = 选 C: 修 2 hard + 5 should-fix; 并请求阶段 A 完成后做代码深审)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**
- 评审目标:
  - 验证 v3 真修了 Round 9 列的 2 hard blocker (B26 / B-V21-scope) + 5 should-fix
  - 验证 v3 没引入新偏差 (Round 9 元教训: 6/8 偏差是 prior round 偏差的新形态; 必须 catch v3 是否复发)
  - 评 CODE_DEEP_AUDIT DRAFT 范围 + 方法论 + Q 列表是否合理, 是否值得阶段 A 完成后真做

---

## 0. 本轮范围与边界

Round 1-7 审 design (做什么). Round 8 审 execution v1 (codex 能不能跑). Round 9 审 v2 修复. **Round 10 验证 v3 + 评 audit DRAFT 范围**.

**本轮只审**:
- v3 vs v2 的 7 处局部修订是否真修了 Round 9 hard fix + should-fix (B26 / B-V21-scope / B27 / B28 / B29 / B30 / B-getctime)
- v3 是否引入新 v3-specific 偏差 (B31+)
- CODE_DEEP_AUDIT DRAFT 的 §1 范围 / §2 方法论 / §3 22 个具体 Q 是否合理
- DRAFT 的 §5 NOT-DO 是否覆盖 audit-specific LLM 滑坡风险

**本轮不审**:
- V18/V13 设计 (Round 1-5 已穷尽)
- 三阶段路线 / V18-clean (Round 7 user 签字)
- v2 任何内容 (已 supersede by v3)
- Q5 push 时机 (user Round 8 已签 Option B)
- 修复范围 (user Round 9 已签 C)
- audit DRAFT 是否该做 (user 已请求, 不重审决策)

---

## 1. 给 reviewer 的 10 个问题 (Q1-Q7 针对 v3; Q8-Q10 针对 audit DRAFT)

### Q1 — v3 7 处修订是否全部落实? (B26 / B-V21-scope hard + 5 should-fix)

v3 §0.5 列了 7 处修订:

| # | 位置 | 期望修复 |
|---|---|---|
| 1 | §C0 | SUPPORTED_FLAGS print → if-fail-exit (B30) |
| 2 | §C1 | `cfg["training"]["log_interval"] = 10` (B26) |
| 3 | §C2 | `START_TS=$(date +%s)` (B-getctime) |
| 4 | §C4 PASS_2 | 阈值 50 → 15 (B26) |
| 5 | §C4 PASS_5 | `END_TS - START_TS` 替代 getctime (B-getctime) |
| 6 | §B0 | grep 全目录 → 白名单 only (B-V21-scope) |
| 7 | §A1 + §E1 §6 | §2.3 row 1 修正 + §2.2 改区间 + attestation 12→16 (B27/B28/B29) |

请逐条 verify:
- v3 文档每处修订**真的**落地了吗? grep / cross-check v3 文本
- v3 §10 自查表 13 项是否真覆盖? 还是 claude 自己勾的, 没真验证?
- 第 7 处修订 (B27 §2.3 row 1) 是否真在 v3 §A1 patch 表里? 还是只在 §0.5 总览提到, 正文漏掉?

### Q2 — B26 修复后, PASS_2 阈值 (≥15) + log_interval=10 是否真能匹配?

v3 §C1 python 改 `log_interval=10`, max_steps=200 → 期望 20 行 train event. PASS_2 阈值 `≥15`.

请 verify:
- train_first_hop.py 的训练循环是否真在 `step % log_interval == 0` 时**且** `step >= 1` 时写 train event? 还是 step=0 也写? 若 step=0 也写, smoke 实际是 21 行 (含 step 0/10/20/.../200); 若不写, 是 20 行 (step 10/20/.../200). 这影响阈值 15 是否有足够 margin
- 训练循环里是否有别的条件 (warmup / pair_only / no_op step) 跳过 metrics_payload write? 若有, 20 行可能变 15-18 行, ≥15 margin 紧张
- log_interval=10 是否影响 V13 full launch 后的训练? **不会** (因 full launch 用原 yaml `log_interval: 50`, smoke yaml 是副本), 但请二次确认

### Q3 — B-V21-scope 修复后, 4 白名单 grep 是否真覆盖且 sanity range [10, 35] 合理?

v3 §B0 grep 4 白名单文件, sanity range [10, 35] (Round 8 实测 21 + ±50% margin).

请 verify:
- 重新 grep 实际 hit 数, 看 [10, 35] 是否真包含
- agent2 Round 9 报告全局 grep 169 hits — 这数字是 prompt md / task md / integration md 等审稿文件本身的 V21 hits 加总. v3 范围 4 白名单后是否真的避开这些非白名单文档?
- B1 是否仍说 "对每处 V21 设计描述加标记 + 不对元数据" — 这个判断规则给 codex 是否够清楚? Round 9 agent3 提议加 "段落开头 V21 = / V21 fallback → 设计描述" 正例, v3 是否采纳? (Round 9 整合标 should-fix 未必修)

### Q4 — B-getctime 修复后, START_TS / END_TS 是否在所有 fail 路径都被正确捕获?

v3 §C2 `START_TS=$(date +%s)` 在 nohup launch 前; §C3 `END_TS=$(date +%s)` 在 `wait $SMOKE_PID` 后.

请 verify:
- 若 smoke 在 TIMEOUT (30 min) 触发 kill, 是否仍 reach END_TS 行? v3 §C3 现有结构里, timeout 路径 `exit 1` 直接退, **没** END_TS 设置 → 后面 ELAPSED_MIN 计算会用未定义变量 → bash 默认替换为空 → 算术 `0 - START_TS = 负数` → PASS_5 可能 trivially pass 或 报错
- 若 smoke 自然完成但 `wait` 失败 (e.g. zombie process), END_TS 是否仍可信?
- v3 是否需在 §C3 timeout 分支也加 `END_TS=$(date +%s)` 后再 exit?

### Q5 — B27 §2.3 row 1 修正措辞 + B28 §2.2 区间措辞是否真合理?

v3 §A1 patch 表行 3 (B28): `0.5 - 2.0 dB → 0.05 - 3 dB (区间, 见 §2.4; 中位 0.1-0.3 dB)`
v3 §A1 patch 表行 4 (B27): `LoRA 容量太小 (rank=8 不够) ... → LoRA 容量仍偏小 (rank=32 也可能不够) ... 跑 V18-r64 sweep (推迟到阶段 B 决策)`

请评估:
- 行 3 措辞 "区间, 见 §2.4; 中位 0.1-0.3 dB" 是否真规避了 B28 stealth 下调? 中位 "0.1-0.3 dB" 同样是 PARTIAL 区间, 仅比 v2 "0.05-0.30" 略宽 — 是否实质改变 EV 预期? 还是更稳健?
- 行 4 提到 "V18-r64 sweep (推迟到阶段 B 决策)" — V18-r64 这个候选**是否在 Round 1-9 任何整合文档里被讨论过**? 还是 v3 凭记忆引入新候选名? 若新, 这是 B25 / B28 复发 (虚构未来候选)
- 行 4 "rank=32 也可能不够" 措辞是否过早 (V18 还没出结果, 不应预设 rank=32 不够)?

### Q6 — B29 attestation 扩到 16 项是否真无遗漏?

v3 §5 §E1 §6 列了 16 项 NOT-DO 与 master §6 一一对应.

请逐条对照 v3 §6 master 16 条 vs v3 §E1 §6 attestation 16 项, 找:
- 是否每条 master 对应 attestation 一项, 一一映射?
- attestation 第 6 条改为 "未起 Round 10 review" — 但**本 prompt 自身就是 Round 10**, 这是否构成 codex 违反 attestation 的悖论? 还是 codex 不需关心 user 起的 review (只关心 codex 自己别起)?
- attestation 16 项是否有重复 / 冗余 (例如 item 14 与 item 7 都涉及不改 train_first_hop)?

### Q7 — v3 是否引入 v3-specific 新偏差 (B31+)? (Round 9 元教训核心)

Round 9 元教训: 6/8 新偏差是 prior round 偏差的新形态. 必须 catch v3 修 B26-B30 时是否又造新偏差.

请独立检查:
- v3 §C1 加 `log_interval=10` 是否引入新 yaml 字段联动盲区? log_interval=10 是否影响 train_first_hop.py 其他依赖 log_interval 的代码路径 (e.g. checkpoint save / eval trigger / progressbar)?
- v3 §C0 SUPPORTED_FLAGS assert 化的 grep pattern `"['\"]--[a-z_-]+['\"]"` 是否真覆盖所有 add_argument 形态? 若 train_first_hop.py 用了 `parser.add_argument("-x", "--xxx")` 短选项, grep 会漏 short flag, assert 可能 false-pass
- v3 §B0 白名单数组 hardcode 4 个路径, 若任一文档被 rename / move, codex 不会知道 → silent skip
- v3 §A1 patch 行 4 (V18-r64) 引入未在 prior round 讨论的候选 — 这是新 B25 (杜撰 future state)
- v3 §10 自查表全是 claude 自勾, 没机制 force 真验证 — 这是新 B30 (non-asserting self-check)

### Q8 — CODE_DEEP_AUDIT DRAFT §1 范围 (M1-M8) 是否合理?

DRAFT §1 列了 P0 (M1-M4) / P1 (M5-M6) / P2 (M7-M8) 共 8 模块.

请评估:
- M1 (train_first_hop.py 全文) 优先级是否过高? 还是合理?
- M2/M3 (decoder_lora.py / model_first_hop.py) 是否是 V18-clean launch 前的真正瓶颈? 还是过度?
- M4 (path_guard.py 30 行) 列 P1 是否过严? 是否可降 P2 或省?
- 缺失的高优先模块? (e.g. dataloader 性能问题影响 V13 7 天完成时长? 但 DRAFT 明确排除 dataloader)
- 范围 (8 模块, ~6-8 h) vs ROI 是否合理? 还是太大 (拖延阶段 B 启动)? 还是太小 (漏关键模块)?

### Q9 — DRAFT §3 22 个具体 Q 是否真有 actionable answer?

DRAFT §3 列 Q3.1.1-Q3.5.2 共 22 个具体问题, 来自 Round 1-9 reviewer 列出但未 verify.

请 verify (抽样 3-5 个):
- 这些 Q 是否真在 Round 1-9 整合 / prompt 中被提过? 还是 claude 凭记忆/合理推测列出来?
- Q3.1.5 "lr_schedule.total_steps_override cosine 起点是 0 还是 ckpt step" — 这问题 codex 真能从静态代码读出答案吗? 还是需要 dynamic trace?
- Q3.1.7 best.pt criterion — 是否在 Round 8 agent2 已经部分回答? v3 audit Q 是否重复 prior verify?
- 22 个 Q 总共预计产出 ~600 行报告, 是否真有 600 行价值 / 其实大部分能在 30 min spot-check 内回答?

### Q10 — DRAFT 是否引入 audit-specific LLM 滑坡风险, NOT-DO 是否够?

DRAFT §5 列 10 条 NOT-DO. 但 audit-specific 滑坡风险与 Phase A 不同:

请评估:
- audit 时 codex 倾向 "顺手修一个看着 obvious 的 bug"? DRAFT §5 #5 "不修复发现的 bug" 是否够强?
- audit 时 codex 倾向 "如果某段代码看不懂就跳过"? DRAFT 没明确说 "看不懂段必须标 FLAG 不许跳过" — 是否补?
- audit 时 codex 倾向 "对 deprecated/ 目录顺手 audit"? DRAFT §1.3 明确排除, 但 §5 NOT-DO 没重复 — 是否补?
- DRAFT §6 失败决策树是否覆盖: M1 太大半途想拆? M1 audit 时发现 V13 因 image_aux=false 在 step 5K 死掉?
- DRAFT §10 "DRAFT 状态说明" 是否真起作用? codex 看到 [DRAFT] 字样会不会以为是 informational 不执行? user 必须先改名去 DRAFT 字样才能发送 — 这流程是否在 v3 阶段 A push 任务里被记录?

---

## 2. 资料目录

按读的顺序:

### 2.1 本轮主审对象

- **[CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md)** — v3 task md (本轮要 review)
- **[CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md](./CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md)** — audit DRAFT (本轮 Q8-Q10 评)

### 2.2 上游 (本轮不审, 仅作 context)

- [CODEX_TASK_PHASE_A_v2_20260517.md](./CODEX_TASK_PHASE_A_v2_20260517.md) — v2 (已 supersede; verify v3 真的修了 Round 9 列的 v2 问题)
- [REVIEW_INTEGRATION_round9_20260517.md](./REVIEW_INTEGRATION_round9_20260517.md) — Round 9 整合 (2 hard + 5 should-fix; user 决策 C)
- [REVIEW_INTEGRATION_round8_20260517.md](./REVIEW_INTEGRATION_round8_20260517.md) — Round 8 整合 (5 hard fix)
- [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](./NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) — Round 7 用户签字架构

### 2.3 代码事实验证 (二次确认)

- [train_first_hop.py:2400-2510](../../train_first_hop.py) — `if step % log_interval == 0:` metrics_payload write gate
- [train_first_hop.py:1355-1357](../../train_first_hop.py) — argparse 仅 `--config` / `--resume`
- [path_guard.py](../../pet_lr/path_guard.py) — output_dir 必须 /data_2/
- [V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) line 153 — `log_interval: 50`
- [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) — Task A 修改目标; verify §2.1 / §2.2 / §2.3 现状

---

## 3. 输出格式 (请 reviewer 严格遵守)

每位 reviewer 独立产出 markdown:

### 3.1 10 个问题逐条回答
对 Q1-Q10 每条**明确**给出 1 个结论:
- `APPROVE` — v3 修了 / DRAFT 合理, 可接受
- `MODIFY (说明)` — 一部分对, 还有改进空间但不阻塞
- `REJECT (说明)` — 没真修 / 引入新 bug / DRAFT 范围不合理, 阻塞

### 3.2 整体 verdict
对 **v3 task md**:
- `READY TO PUSH` — 可直接 push 给 codex 执行阶段 A
- `MODIFY THEN PUSH` — 列出必改项后可 push
- `BLOCK` — 有未修的 hard blocker 或新 hard blocker, 需 v4

对 **audit DRAFT** (独立 verdict):
- `READY TO USE AS-IS` — 阶段 A 完成时 user 直接 rename + push
- `MODIFY THEN USE` — 列出必改项
- `RESCOPE` — 范围严重不合理 (太大/太小/缺关键模块), 需大改
- `DELETE` — DRAFT 不值得做, user 应放弃

### 3.3 v3 新偏差 (重点, Round 9 元教训)
若你发现 v3 在 Round 9 修复中引入了新 confirmation bias / silent bug, 用 B31+ 编号单独列出, 含:
- 偏差形态
- 在 v3 哪一段
- 代码/文档证据
- 修复建议

### 3.4 audit DRAFT 范围调整建议 (若 Q8-Q10 提出 MODIFY/RESCOPE)

### 3.5 (可选) 代码层 spot-check
若你 verify 了 v3 / DRAFT 某段与实际代码/文档不符, 给具体证据.

---

## 4. 约束与提醒

- **本轮不重审** 修复范围 (user Round 9 已签 C), V18/V13 设计, 三阶段路线, Q5 push 时机
- **不假设** 你能跑代码 / 看 wandb / 触 GPU
- **特别关注**:
  - v3 是否真修了 7 处 (不只是 §0.5 总览说修了, 而是正文真改了)
  - v3 §10 自查表是 claude 自勾, **reviewer 必须独立 verify, 不信任 claude 自评**
  - audit DRAFT Q3 列表是否真有 actionable answer 还是产出会很 vague
  - DRAFT §10 DRAFT 状态机制是否会被 codex 误执行
- **优先质疑**:
  - Q4 (timeout 路径 END_TS 是否设置) — 这是 v3 引入的最隐蔽边界条件
  - Q5 (V18-r64 sweep 是否在 prior round 讨论过) — 这是 v3 引入的潜在 B25 复发
  - Q7 (v3-specific 新偏差) — Round 9 元教训核心
  - Q10 (audit DRAFT NOT-DO 是否覆盖 audit 特有滑坡)

---

## 5. 给 reviewer 的硬约束摘要 (standing, 不可推荐违反)

任何 reviewer 提议都必须满足:

1. ≤ 3 并行训练任务 (本 task 已用 slot 1 = V18 + slot 2 = V13, slot 3 = null)
2. V18 不动 (slot 1 训练中)
3. V18b 永久撤销 (Round 6 共识 + 代码 hard blocker)
4. 阈值不改 (Round 7 user 签字 `全推荐` = agent3 不改路线)
5. 不预先 pre-register V18-clean (阶段 C 才 pre-register, Round 7 B18 教训)
6. user 决策 Option B (5 min push + 措辞降级) + 决策 C (修 2 hard + 5 should-fix) 已签, 不可推翻
7. **Round 10 是 execution + audit-prep 审稿终点**. 若 v3 仍需 v4, 走第二轮 v3→v4 review (不算 Round 11, 是 Round 10b). audit DRAFT 修改不算新 round, 是同步整合
8. **不审 audit 是否该做** (user 已请求, 不重审决策); 只审 audit DRAFT 范围/方法论/Q 列表合理性

违反任意一条 → reviewer 提议自动作废.

---

## 6. 本轮元说明

| 轮次 | 范围 | 是否 user 仲裁? |
|---|---|---|
| Round 1-7 | design | YES (R7 全推荐) |
| Round 8 | execution v1 | YES (Q5 Option B) |
| Round 9 | execution v2 | YES (C: 2 hard + 5 should) |
| **Round 10** | **execution v3 + audit DRAFT** | **决于本轮 verdict** |

预期 Round 10 outcome:
- 最佳: 3/3 READY TO PUSH (v3) + 2-3/3 READY TO USE (audit DRAFT) → 直接 push v3 + 保留 audit DRAFT 备用
- 中等: 3/3 MODIFY THEN PUSH 给 ≤3 should-fix → claude 速改 v3 (~20 min) → 直接 push
- 最差: v3 有新 hard blocker → v4 + Round 10b (但 user Round 9 已立 "v3 是终点" 期望, 是否走 v4 是 user 决策点)
