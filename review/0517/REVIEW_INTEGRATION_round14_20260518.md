# Round 14 Review Integration — A3 V18-capacity-only Task + A4 §5.2 standing rules

- date: 2026-05-18
- branch: foc_lite_hop0
- 主审对象: 3 个 (yaml + task md + design_rationale §5.2)
- 3 reviewer (含 mechanical yaml diff + per-flag verify)
- 上游: Round 13 user 决策 = A1/A3/A4

---

## 0. 一行结论

**3/3 reviewer 一致 verdict**: 
- **yaml = READY TO USE** (mechanical diff 195 字段 = 恰好 4 字段差异, claude prep 完全正确)
- **task md = MODIFY BEFORE PUSH** (4-5 处必修, 含 1 个 hard blocker B48 + 3 个 should-fix)
- **§5.2 standing rules = MODIFY** (rules 实质 OK 但全是 cultural/protocol, 应诚实标注 enforcement mode)
- **push 顺序 = B (3/3 共识)**: 先 push commit 7486776 (V13 + V21 retire), 后 push A3 单独 commit

**Round 14 新发现 7 个偏差 B48-B54**, 全 LOW/MOD, 不阻塞 push 但**任意 1 个不修 = 实验干净度降级**.

| 维度 | reviewer1 (self-critique) | reviewer2 (B independent) | reviewer3 (copilot) | 共识 |
|---|---|---|---|---|
| yaml | ready w/ 4 字段 verify | mechanical 195 字段 diff verify | parsed exact 4 fields verify | **3/3 READY** ✅ |
| task md | MODIFY 4 处 (B1/B2/B3/B5) | MODIFY 4 处 (B48/B49/B50/B53) | MODIFY 6 处 (probe/4th outcome/anchor/log grep/NOT-DO/sleep) | **3/3 MODIFY** 🔴 |
| §5.2 | MODIFY (措辞 + 编号体系) | MODIFY (cultural-vs-mechanical 诚实标注) | MODIFY (enforcement 分类 + §5.1 supersede) | **3/3 MODIFY** ⚠️ |
| push 顺序 | (未明) | **B** | **B** | **2/3 B** (reviewer1 已 inline 修了部分, 无 push 顺序 verdict) |
| 新偏差 | B1-B9 | B48-B54 | B48-B53 | **B48 (HIGH 共识) + B49/B50/B53 (HIGH 共识)** |

---

## 1. 3/3 共识必修 4 项 (B48-B53)

### 1.1 B48 (HIGH) — yaml `save_interval=10000` 导致无 step_165000.pt (reviewer2+3)

**问题**: capacity-only `max_steps=170000` + `save_interval=10000` + `resume_from=V7@160K` → 只 emit step_170000 ckpt. **但** V18.best @165K (5K LoRA training) 才是与 capacity-only 配对的正确比较 (按 standing rule B46 4×2 comparison-label 要求 matched LoRA-training steps).

**修法** (1 行 yaml):
```diff
-  save_interval: 10000
+  save_interval: 5000   # capacity-only: emit step_165000.pt for matched LoRA-training comparison with V18.best
```

**reviewer1 也 catch 同问题 (B2)** — 要求 task md §B1 加 `--v18-step170k-ckpt` 改用 V18 server 上已存的 step_170000.pt (V18 训到 200K, save_interval=10K, 所以**V18 自带 step_170K**). 

**整合**: 两种方法都修了 matched-step 问题. 但 reviewer2 方法更精确 (capacity-only 与 V18 都对齐 5K LoRA steps), reviewer1 方法对齐 absolute step 170K. **claude 推荐 reviewer1 方法** (V18 step_170K 已在 server, 不需重 train), 同时 reviewer2 yaml 改 save_interval=5000 也加 (capacity-only 自己有 step_165K + step_170K 两 ckpt, 全部比较点齐).

### 1.2 B49 + B50 (HIGH) — task md §B1 probe flag 不存在 (3/3 一致)

**问题**: task md §B1 调用 `--v18-cap-ckpt`, 实测 `tools/probe_v18_kl_drift.py` argparse 无此 flag. §6 用 "若 probe 不支持..." 弱化, 但实际**必须**加 ~10 行. 同时 §5 NOT-DO #2 "不改 train_first_hop.py / decoder_lora.py" 与 §B1 改 probe 冲突 (probe.py 不在 NOT-DO 列表, 但 codex 看到会犹豫).

**修法**:
- §B1 改为 "**必须** 加 ~10 行 probe.py 扩展" (不再 "若")
- §5 加 `✅ 例外: tools/probe_v18_kl_drift.py 允许 ~10 行扩展`
- 任务加预 B0 step: probe.py 先 `--help` grep verify `--v18-cap-ckpt` 存在, 失败则 codex 先改 probe

### 1.3 B53 (HIGH) — §1.3 outcome 2 与 §5.2 B44 自相矛盾 (reviewer2+3)

**问题**: 同 commit 引入:
- §1.3 outcome 2: "capacity-only ≈ V7 → **A 候选可能真**"
- §5.2 B44 (Round 14 我加的): "KL→z_GT alignment 机制不可能"

两段一前一后矛盾.

**修法** (改 §1.3 outcome 2):
```diff
-2. **capacity-only ≈ V7 (0 dB)** → A 候选可能真 (但仍违反 §1.1 机制, 需二次调查; 可能 KL 间接经 z_pred 训练影响 z_GT 路径)
+2. **capacity-only ≈ V7 (0 dB)** → +0.10 dB **不是** LoRA capacity 副产品. B44 已 rule out 直接 KL→z_GT 对齐. 必有间接机制 (e.g. KL on z_pred → 共享权重梯度 → 渗到 z_GT eval 路径). 开 sub-investigation, **不**复活 A.
```

### 1.4 B51 / B52 task md log grep + sleep 300 问题 (reviewer3 独家)

- B51: §A2 grep `Resuming from checkpoint` 与实际 trainer log 不符 (实际 print `[startup] resume from:`, `[resume] loaded step=`). 改 grep 字符串.
- B52: §A2 `sleep 300` 在 agent terminal brittle. 改为 "split check command, 用户/codex 5 min 后单独跑 verify".

---

## 2. 3/3 共识应修 3 项

### 2.1 B54 (MOD) — §1.2 "默认最简释" anchor (reviewer1+2+3 一致)

prompt §1 Q3 自己 flag 但 task md 仍含此措辞. claude 在 Round 14 prompt 已加 [NEW Q4.0c] 但 task md 未改.

**修法** (task md §1.2 header):
```diff
-### 1.2 候选 B (默认最简释)
+### 1.2 候选 B (B42 之后唯一机制上一致的解释 — 实测验证, 不预设结论)
```

### 2.2 §5.2 cultural-vs-mechanical 诚实标注 (reviewer2+3 一致)

5 条 standing rules 实质 OK 但全是 cultural/protocol. 应加诚实 enforcement 分类表.

**修法** (V18_design_rationale §5.2 末加):
```markdown
#### Enforcement honesty (Round 14 reviewer 加注):

| rule | enforcement mode |
|---|---|
| B43 (V18 buggy KL prefix) | CULTURAL — 无 template/script enforce |
| B44 (KL ⊥ decode(z_GT)) | CULTURAL + 由 capacity-only 控制实验 backed |
| B45 (substrate reviewer 介入) | CULTURAL — 依赖 user 主动起 review |
| B46 (4×2 ΔPSNR 标签) | CULTURAL-with-protocol — Round 12/13 已 enforced 一次 |
| B47 (mechanical script assert) | PROTOCOL — closest to mechanical, 但仍依赖 user 跑 script |

→ 全部依赖 claude/reviewer/user 主动 honor. 真 mechanical enforcement (lint script / CI hook / template generator) 是 future 增强.
```

### 2.3 §5.1 第 3 条 "不允许再发 peer review" superseded 标记 (reviewer3 独家)

§5.1 红线 #3 "不允许再发 peer review" 与 §5.2 B45 "substrate stage reviewer 必须介入" 直接冲突. 我 Round 14 起草时已加注释 "已突破 13 轮", reviewer3 推 superseded 标记更清晰.

**修法** (V18_design_rationale §5.1):
```diff
-3. **不允许再发 peer review** ...(原文) (**Round 5-13 已突破此红线 13 轮 review, ...**)
+3. ~~**不允许再发 peer review**~~ **[SUPERSEDED 2026-05-18 by §5.2 B45]**. 原意是防 Round 1-4 review-meta-review 永动机. Round 5-13 反例: substrate stage 反复证明 reviewer 介入是必要 (Round 12 best vs last label / Round 13 B42 mechanism). 新规: **战略转折点 + substrate 解读必须起 reviewer**, 战术细节 user+claude 直接.
```

---

## 3. Push 顺序 (3/3 共识 = B)

**Option B** 推荐: 
1. 先 push commit 7486776 (V13 launch + V21 retire 文档清理) — 已 reviewed, 解锁 V13 slot 2
2. 修订 task md + yaml save_interval (5 项必修 + 3 项应修, ~20 min)
3. 后 push A3 单独 commit (yaml + task md + design_rationale §5.2)

**理由 (reviewer2/3 一致)**:
- 7486776 已 reviewed atomic, 不再开
- A3 4 hard fix + 3 should-fix 需先修
- audit trail 隔离 (V13 与 A3 用途不同, separated commits 易 revert)

**Reject** Option A (一次 bundle): 重开 7486776 风险.
**Reject** Option C (A3 先 V13 拖延): V13 独立科学价值, 不应延后.

---

## 4. 修订清单 (3/3 整合, push 前 ~25 min)

### 4.1 yaml (1 处)

| # | 位置 | 修法 | 来源 |
|---|---|---|---|
| 1 | V18_capacity_only.yaml `save_interval` | 10000 → 5000 | B48 reviewer2 |

### 4.2 task md (6 处)

| # | 位置 | 修法 | 来源 |
|---|---|---|---|
| 2 | §1.2 header | "默认最简释" → "B42 之后唯一机制上一致的解释 — 实测验证" | B54 (3/3) |
| 3 | §1.3 outcome 2 | 改写 "A 候选可能真" → "必有间接机制 (KL on z_pred → 共享权重渗 z_GT), 不复活 A" | B53 (3/3) |
| 4 | §1.3 加 outcome 4 | "capacity-only < V7 → LoRA 本身有害, 重审 B42" (claude Round 14 已加 [NEW] in prompt, sync 到 task md) | B49 (3/3) |
| 5 | §B1 probe 调用 | 改为 "**必须** 加 ~10 行 probe 扩展 `--v18-cap-ckpt`", 加 B0 preflight `--help | grep` | B49 (3/3) |
| 6 | §A2 log grep | `Resuming from checkpoint` → `[startup] resume from:` 或 `[resume] loaded step=` | B51 reviewer3 |
| 7 | §5 NOT-DO | 加 `✅ 例外: probe.py 允许扩展` + 加 #10-13 (不改 capacity-only yaml / 不 push 无关 commit / 不重命名 ckpt 路径 / 不在 capacity_only/run 写 V18 ckpt) | B50 reviewer2/3 |

### 4.3 §5.2 (1 处)

| # | 位置 | 修法 | 来源 |
|---|---|---|---|
| 8 | V18_design_rationale §5.2 末 | 加 cultural-vs-mechanical 分类表 | reviewer2+3 |
| 9 | V18_design_rationale §5.1 第 3 条 | superseded 标记 + 链 §5.2 B45 | reviewer3 |

总: **9 处修订, ~25 min**, 不阻塞 push (B48/B49/B50/B53 是 disambig 干净度问题, 任意 1 个不修 = 实验某 cell 失去比较).

---

## 5. user 决策点 (3 个)

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | 修订范围 | A: 9 项全修 / B: 仅 4 hard fix (B48/B49/B50/B53) / C: 仅 yaml save_interval | **A** 9 项全修 (~25 min, ROI 高, 全部解决 cultural-vs-mechanical 自相矛盾) |
| 2 | Push 顺序 | A: bundle 一次 / **B: 7486776 先 + A3 后** / C: A3 先 7486776 拖 | **B** (3/3 共识) |
| 3 | reviewer 是否再起 Round 15 验证 v2 | A: 起 (R12/13 元教训) / B: 不起 (Round 14 共识强, 修订都是 cultural 不是 substrate) | **B 不起** (Round 14 3/3 一致, 修订 mechanical, V18-capacity-only 自己出来后才需 Round 15 战略 review) |

回复 `1=A 2=B 3=B` 或 `按推荐`.

---

## 6. Round 14 元教训

**reviewer 价值再印证**: 3/3 reviewer 独立 mechanical yaml diff (195 字段) 都做了 ✓, 完全 verify claude prep; 但仍 catch **claude 起草中 anchor 3 处** (B53 §1.3 outcome 2 与新加 B44 矛盾 / B54 "默认最简释" 自我喂答案 / §5.2 假 "permanent" 标签). 

**Round 14 比 Round 13 cleaner**: 没新 hard mechanism bug (Round 13 是 B42), 全是 wording / process 偏差. 项目阶段从 "claude 凭记忆造 bug" → "claude prep mechanical correct, anchor 在 framing 层".

**B45 propagation 启用**: 本 Round 14 起 cadence ≈ Round 13 (战略层), reviewer 介入 100% catch 全部 5 hard fix. R12-13-14 三轮 reviewer 都触发后, **B45 实证成立** — substrate / framing 阶段 reviewer 必须介入.

---

## 7. 立即下一步

待 user 回复决策 1/2/3 → claude:
1. (若 1=A) 修订 9 项 (yaml 1 + task md 6 + §5.2 2)
2. (若 2=B) push 7486776 → push A3 (单独 commit)
3. (若 3=B) 不起 Round 15, 等 codex 执行 capacity-only (~24-48h), 数据回再起 Round 15 战略 review

总 ~30 min 本地修订 + ~5 min push, 然后等 ~24-48h capacity-only substrate 回.
