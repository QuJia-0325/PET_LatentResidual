# Round 17-Slots Peer Review — Reviewer A (independent)

- date: 2026-05-22
- reviewer: **Reviewer A** (GitHub Copilot, Claude Opus 4.7 xhigh, independent draft, did not see Reviewer B/C/D drafts)
- 主审对象: prompt §0 唯一问题 — 是否打破 Round 17 stop rule "no relaunch A4-v2/v3", 把 A4-light 升级为 2-3 slot 并行 image_aux sweep
- 立场基线: R17 整合 4/4 共识 + Round 17-Prep R17-Stats 待审, **当前 stop rule 还是 1-probe 版本**

---

## 0. 主 verdict (一句话)

**接受 revisit, 选方案 E (2 slot, λ=0.02 + λ=0.08), 拒绝方案 A/B/C**.

原因 3 行:
- R17 整合 L173 / L194 / L196 精神是 "**one** probe, **1** slot, idle-only" — prompt §1 用 "relaunch ≠ parallel launch" 字面解读**绕过精神共识**, 这本身是新偏差 B83
- user 主动 slot 信号是合法 new-info trigger (R17 整合 L173 自己列了 "if GPU slot is idle" 条件), 可 revisit; 但 revisit 应**最小幅度**, 不是直接跳到 sweep
- 方案 E (2 slot) 是最小幅度 revisit + 给 paper 添 1 个数据点 + user 已明示 "2 安全 3 RAM 争抢"; 方案 A/B 直接违反 R17 多条共识, 方案 C 引入无关 V13b scope

备选: **方案 D** (守原 1 slot). 若 reviewer 共识严守 stop rule 精神, D 是合法 fallback.

---

## 1. R17 整合精神到底是不是 "1 probe"

prompt §1 措辞策略:

> "stop rule 字面是禁 **relaunch**, 没禁**预先并行**"

verify [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md):

| 行 | 原文 | 关键词 |
|---|---|---|
| L21 | "Review D: `B + A4-light` (A4 **not gating**, paper viable without it)" | A4 不挡 paper |
| L173 | "**A4 — one image_aux schedule probe**, pre-registered with stop rule (`<+0.05 dB → do not re-run`). **Launch only if GPU slot is idle**. Cost: **~7 days × 1 slot**" | "one" / "1 slot" / "idle" 三连 |
| L179 | "**V14b/V14c multi-seed: low priority; F0 covers significance question more cheaply**" | V14b 已被 R17 降级 |
| L196 | "Optionally launch **one pre-registered A4** image_aux schedule probe if GPU is idle. Do not relaunch on failure." | "one" 再确认 |

**结论**: R17 共识精神 = 1 probe, 1 slot, idle-only, 7 day cost. prompt §1 用 "字面 vs 精神" 划分**把字面解读放大** — 字面 "no relaunch" 解读为只禁串行, 但精神 "one probe / 1 slot" 4 处明说 1 个. 这不是字面/精神 distinction, 是**忽略另外 4 条措辞**.

**这构成新偏差 B83** (见 §5).

---

## 2. user 信号是不是合法 revisit trigger

user 原话: "此时只运行了一个训练? 我认为可以同时运行两个, 最多运行三个"

合法性分析:
- R17 L173 自己列了 "Launch only if GPU slot is idle" 作为 A4 launch 条件 → **slot 状态是 R17 共识里的合法变量**
- R17 制定时, reviewer 隐含假设 slot 紧张 (因为 V18 + V13 + V14 三个刚跑完, R17 launch 期可能争抢)
- 当前 V13/V14 都已经完成, V18 不动, 实际 slot = 3 idle
- user 作为 PI 主动指出 slot 信号 ≠ scope creep B74 ("**悄悄**变 paper + 无限实验")
  - B74 关键词 = **悄悄** = 缺乏 user 决策
  - user 主动信号 ≠ 悄悄

**结论**: user 信号是合法 new-info trigger, 可触发 revisit. 但 revisit 必须:
- 给新 stop rule (不能从 "1 slot stop rule" 直接变 "无 stop rule")
- revisit 幅度最小化 (从 1 → 2, 不是 1 → 5 点 sweep)
- 不挑战 R17 其它降级条款 (V14b "low priority" 不应被一并复活)

---

## 3. 方案逐一审

### 方案 A — 3 slot bracket (0.02/0.08/0.12) — **REJECT**

理由:
- λ=0.12 EV 最低. V7 已经经验调出 0.04, 物理直觉是 sweet spot 在 0.04 附近. λ=0.12 大概率回归到 V7 以下, **+1 slot RAM 风险换 1 个 negative ablation 点**, EV 失衡.
- 3 slot 直接挑战 user "3 个时 RAM 争抢" 警告. user 给的是软上限不是强制, 但**首次 revisit 不应贴上限**.
- 5-点 response curve (V13 + 0.02 + V7 + 0.08 + 0.12) 是 paper figure padding — image_aux 主结论已由 V13 vs V7 = +0.287 dB 锁定, A4-high 不增加 single-variable 结论, 只增加曲线形状. **incremental ≠ necessary**.
- 杀 stop rule 幅度过大: 从 "1 probe" 一次跳到 "3-point sweep", 这种 step-size 让未来每一轮 stop rule 都失效力.

### 方案 B — 2 A4 + V14b 第 3 seed — **REJECT** (最强反对)

理由:
- 直接违反 R17 L179 "**V14b/V14c multi-seed: low priority; F0 covers significance question more cheaply**". 这是 R17 整合明文降级条款, 不是 implicit assumption.
- 把 V14b 与 A4 sweep 绑定 = 一次 revisit 杀**两条** R17 共识 (image_aux scope + V14b priority). 滑坡风险翻倍.
- "V14b 喂 R17-Stats noise floor" 是合理理由, 但应**单独**走 R17-Stats revisit 流程, 不应搭车 A4 sweep.
- F0 paired-t 一旦 ready (R17-Stats 审完), 直接给 significance 答案, V14b 第 3 seed 信息**冗余**.
- 3-slot RAM 风险 + 跨实验 footprint 不一致 (V14b 是 V7 clone ~20GB, A4 也是 V7 footprint 同级, 但跨 run 调度 IO 争抢不可预测).

### 方案 C — A4-mid + V13b — **REJECT**

理由:
- V13b 与 image_aux sweep 主题完全无关 (V13b 是 "image_aux off + V7 train config" 用来清算 V13-vs-V8 train config 残差).
- 把无关实验塞进 image_aux sweep revisit = **scope mixing**. 如果 V13b 真值得跑, 应该走独立 Round 17-X revisit 路径.
- V13 vs V8 +0.02 dB 残差是低优先级 puzzle, 远低于 image_aux sweep 的 paper relevance.
- 2-slot 利用率 ≠ 必须填满, **空 slot 是合法 buffer**.

### 方案 D — 守原 R17 1 slot — **APPROVE as backup**

理由:
- 严守 R17 精神. 无新风险.
- 缺点: 浪费 user 给的 slot 信号; A4 单点 outcome 只能写一句话 "lambda=0.08 给 ±X dB"; paper figure 维度感弱.
- 作为方案 E 的 fallback 合理.

### 方案 E — 2 slot bracket (0.02/0.08) — **APPROVE 为主推**

理由:
- **最小幅度 revisit**: 从 1 probe → 2 probe, 不是 → 5 点 sweep. 杀 stop rule 影响最小.
- **honors user 信号**: 用 2 slot (user 明示 "可以同时运行两个"), 留 1 slot buffer (user 明示 "3 个有 RAM 争抢").
- **paper 增量真实**: V13(0) + A4-low(0.02) + V7(0.04) + A4-mid(0.08) = 4 点 image_aux response curve, 能看上升段 + 是否 0.04 已 saturated. 是 paper figure-quotable.
- **不挑战 R17 其它条款**: 不复活 V14b "low priority", 不引入无关 V13b.
- **物理上 0.02 + 0.08 是对称 bracket**: 围绕 V7 0.04 各 1 个倍数 (0.5× / 2×), 双方向 saturation check.
- 缺 λ=0.12 上方饱和点的代价**可接受** — V7 (0.04) → A4-mid (0.08) 已有 2× 倍率, 若 0.08 仍 increasing, 0.12 进一步增长几乎不可能 (image_aux 是 reconstruction 辅助损失, 必然 dominate at high lambda 后破坏 latent transport 学习).

新 stop rule 措辞 (paper-ready):
```
Round 17-Slots stop rule: image_aux schedule sweep 至 A4-bracket {λ=0.02, λ=0.08} 2 slot
闭合. 此后**不**因任何 outcome 再 launch λ ∈ {0.06, 0.10, 0.12, 0.16, ...} 或 V14b/c/d 或
其它 ablation-extender. paper 主图 image_aux response curve 4 点 (V13 λ=0 + A4-low λ=0.02
+ V7 λ=0.04 + A4-mid λ=0.08), 终态.
```

---

## 4. 6 个问题逐条 (per prompt §3)

### Q1 — Stop rule 该不该 revisit? **APPROVE**

- 字面: "no relaunch" 不直接禁 parallel launch, **技术上**留口子
- 精神: R17 L173/L196 明说 "one" / "1 slot", **完整共识**是 1 probe
- 因此 "字面允许但精神反对". user 主动 slot 信号是合法 new-info trigger (R17 L173 自己列 slot 状态为条件), 可 revisit
- 但 revisit **必须最小幅度** + 必须给新 stop rule

### Q2 — image_aux sweep 真有 EV? **MODIFY — 部分有, 但被 prompt 夸大**

- 真实 EV: 2 点比 1 点强 (能 fit 二次曲线 + 判断 0.04 是否 sweet spot), 3-5 点比 2 点强 (saturation evidence)
- 但 paper 主结论 (image_aux +0.287 dB) 已由 V13 vs V7 单变量 lock, sweep 是 **incremental** 不是 **necessary**
- R17 L21 自己说 "A4 not gating, paper viable without it" → A4 整体只是 nice-to-have
- 因此 sweep EV 真实但不大. 应该接受**最小**增量 (2 点), 不应做 full sweep (3+ 点)
- "如果 0.08 显著高于 V7 触发追 0.06 / 0.10" 是 prompt §Q2 自己点出的滑坡风险 — Reviewer A 同意, **新 stop rule 必须先发**

### Q3 — 4 方案选 1? **方案 E 主, 方案 D 备**

见 §3 详细分析.

### Q4 — 3-slot RAM 风险? **建议 staggered, 限 workers**

- 方案 E 是 2-slot, RAM 风险显著低于 3-slot, 但仍建议 staggered launch
- T=0 起 A4-mid (0.08), T+30min IO + RAM health check 后再起 A4-low (0.02)
- dataloader workers: V7 默认值需 verify, 建议 2 个 A4 都设 `num_workers=4` (V7 通常 8 → 降一半防 IO 争抢)
- 不需要 user SSH 手 verify, codex 可 `free -g` + `nvidia-smi --query-gpu=memory.used` 自动 check
- 若选方案 A (3-slot), 必须 staggered + 至少 1 hour 间隔 + user 手 SSH verify

### Q5 — A4-low (0.02) 还是 A4-high (0.12) 先 launch? **方案 E 不含 0.12**

- 方案 E 只用 0.02 + 0.08, λ=0.12 不入
- 若用方案 A (3-slot): 优先级 0.08 > 0.02 > 0.12
  - 0.08 = 上调 2× 探 V7 是否 under-fit, EV 最高 (信息量最大)
  - 0.02 = 下调 0.5× 探 V7 是否 over-fit, EV 中
  - 0.12 = 3× 探过量, EV 最低 (大概率回归 V7 以下, 只能做 negative ablation)
- 因此 prompt §Q5 "省 0.12 = 方案 E" 是对的, Reviewer A 直接选 E

### Q6 — 需正式 Round 17-Slots 整合 + 更新 task md? **APPROVE**

必须做的:
1. 写新 Round 17-Slots 整合文档 [REVIEW_INTEGRATION_round17_slots_20260522.md](./REVIEW_INTEGRATION_round17_slots_20260522.md) 标注:
   - stop rule revisit 来源 (user 主动 slot 信号)
   - 新 stop rule 措辞 (§3 方案 E 段已给)
   - 新 anti-check pattern
2. CODEX_TASK §3 (A4 section): 改为 A4-bracket 2 yaml (`A4_image_aux_lambda_02.yaml` + 现有 `A4_image_aux_lambda_08.yaml`)
3. CODEX_TASK §A4.5: stop rule 措辞改为 "A4-bracket 2 slot 闭合, 不再 launch any image_aux variant 或 V14b/c"
4. CODEX_TASK §5 anti-check: 改为
   ```bash
   anti_check "exactly 2 A4 image_aux variant yaml" bash -c "[[ \$(ls review/0521/A4_image_aux_lambda_*.yaml 2>/dev/null | wc -l) -ne 2 ]]"
   anti_check "no A4 variant outside {02,08}" bash -c "ls review/0521/A4_image_aux_lambda_*.yaml 2>/dev/null | grep -v -E 'lambda_(02|08)\.yaml$' | grep -q ."
   anti_check "no V14b/c yaml" bash -c "find review/0521 -name 'V14[bc]*.yaml' 2>/dev/null | grep -q ."
   ```
5. CODEX_TASK §6 NOT-DO #6: 改为 "**启 A4 第 3 个 variant (λ ∉ {0.02, 0.08})** — Round 17-Slots stop rule, 防 scope creep"
6. 新增 yaml: `review/0521/A4_image_aux_lambda_02/A4_image_aux_lambda_02.yaml` (clone V7, `training.image_aux.lambda_start = lambda_max = 0.02`)
7. Round 17-Stats 审完前**不**启动 A4 (避免与 F0 statistical 评审锁定冲突)

### Q7 — 新偏差? **YES, B83-B85**

见 §5.

---

## 5. 新偏差 B83-B85

### B83 [HIGH] Stop-rule literal-vs-spirit elasticity (字面绕精神)

**触发**: prompt §1 把 stop rule 解读为只禁 "relaunch" 不禁 "parallel launch", 但 R17 整合 4 处明说 "one probe / 1 slot / one pre-registered A4".

**模式**: 当 user/claude 想 revisit stop rule, 选取**最有利于扩展**的字面措辞, 忽略整合文档里更强的精神共识. 这是 selective-citation 偏差.

**counter rule**: 任何 stop rule revisit prompt 必须**完整引用** R17 整合所有相关行 (不只引 stop rule 一行), 让 reviewer 看到 "one" / "1 slot" / "idle-only" 的完整精神信号.

### B84 [MED] User-signal-as-revisit-trigger 弹性 (合法但需 discipline)

**触发**: user 一句 "可以同时运行两个" 触发 stop rule revisit. user 信号合法 (R17 L173 自己列 slot 为条件), 但**未来每一轮 stop rule 都可以被类似 trigger 打开**.

**风险**: 滑坡. user 任何 sentiment ("slot 空着浪费", "数据点少", "再多 1 个 seed 更好") 都可能被解读为 new-info.

**counter rule**: user-signal 触发 stop rule revisit 必须满足 3 条:
1. user 信号是**具体可操作**的 (e.g. "可以并行 2-3 个" 具体, "感觉应该再多跑点" 不具体)
2. revisit **必须配新 stop rule** (不能从 "1 slot stop rule" 退化为 "无 stop rule")
3. revisit 幅度**最小化** (1 → 2 OK, 1 → 5 NO)

### B85 [MED] Paper-figure anchor (sweep curve > single point)

**触发**: prompt §2.3 方案 A 暗示 "5 点 response curve 比 2 点 paper 强"; §3.2 方案 D 被框成 "paper figure 弱".

**问题**: image_aux 主结论 (V13 vs V7 = +0.287 dB) 已由 single-variable 锁定. sweep curve 是**incremental evidence** 不是**necessary evidence**. R17 L21 明说 "A4 not gating, paper viable without it" — paper 可以无 A4 整体.

**counter rule**: 任何 "更多数据点 = 更强 paper" 论证必须先指出 baseline (V13+V7 2 点) 是否已足够支撑主结论. 若是, 增量数据点应明示为 "strengthening" 不是 "saving".

---

## 6. 其它必改 (prompt 本身的小问题)

| § | 问题 | 修法 | severity |
|---|---|---|---|
| §0 | "Round 17-Prep (codex execution) **已签**" | 实际 Round 17-Prep 已审出 5 HIGH bug, 在 modify-before-push 状态, 未真正 "已签" | MED (事实陈述不准) |
| §1 | "stop rule 没有禁止的: 预先起 2-3 个 image_aux variant 并行" | 该行精神禁了 (R17 L173 "one"), 应补充: "字面留口子, 但精神禁; revisit 合法基础是 user 新信号, 不是字面解读" | HIGH (B83 同型) |
| §2.3 | 列了 5 个方案 (A/B/C/D/E) 还说 "请直接给观点不要 4 个方案推给 user 决策疲劳" | 方案数 5 vs 文字 4, 计数不一致 | LOW |
| §6 | 末段同样说 "不要因稳妥列 4 个方案" | 但本 prompt 自己列了 5 个方案 → prompt 内部自相矛盾 | LOW |

---

## 7. 输出元

- reviewer: **Reviewer A**
- methodology: R17 整合文档行级 inspection + CODEX_TASK §A4.5/§5/§6 原文 verify + EXEC_FIXES 修改范围 check + prompt §1 引用与原文 cross-check
- 未与 Reviewer B/C/D 交流
- 关键独立结论:
  1. prompt §1 字面/精神 distinction 是 selective-citation (B83): R17 整合 4 处明说 "one / 1 slot", 不只是 "no relaunch" 一句
  2. user slot 信号是合法 revisit trigger (R17 L173 列 slot 为条件), **但必须最小幅度 + 配新 stop rule**
  3. **主推方案 E** (2 slot, λ=0.02 + λ=0.08), 备选 D
  4. 拒绝方案 A (3-slot RAM 风险 + λ=0.12 EV 最低), 拒绝方案 B (V14b 直接违反 R17 L179 降级), 拒绝方案 C (V13b 无关 scope mixing)
  5. 新 stop rule 措辞已给 (§3 方案 E 段), task md 改 6 处已列 (§4 Q6)
  6. 新偏差 B83 / B84 / B85 (selective-citation + user-signal-trigger discipline + paper-figure anchor)
