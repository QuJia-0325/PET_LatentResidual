# Peer Review Round 11 — Phase A v3 (Round 10 hard-fixed) + Audit DRAFT (D1-D6) 审稿

- date: 2026-05-18
- branch: foc_lite_hop0 (v3 + audit DRAFT 仍未 push)
- 主审对象 (2 份):
  1. **[CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md)** (Round 10 选 C 后修订, 7 处 v2 修订 + Round 10 增 4 处, 含 §10.1 mechanical verify 脚本)
  2. **[CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md](./CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md)** (Round 10 D1-D6 修订, 含 PENDING USER RELEASE 顶部警告 + M5 升 P0 + M4 降 P2 + Q3.2.1 方向反 + NOT-DO 扩 13 条)
- 上游: [REVIEW_INTEGRATION_round10_20260518.md](./REVIEW_INTEGRATION_round10_20260518.md) (3 reviewer 共识必修 B31 + agent1 漏修 git add + 5 should-fix; user 决策 C 全部 1-11)
- claude self-review: 已跑 §10.1 mechanical 脚本, **13/13 通过** (修正脚本 false positive 后)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 1-7 design / Round 8 v1 / Round 9 v2 / Round 10 v3+audit / **Round 11 验证 v3 Round 10 修订**.

**本轮只审**:
- v3 是否真修了 Round 10 列的 4 hard fix (B31 §0.5 vs §B0 / git add 通配 / diff 估计 / item 6 措辞) + 5 应修 (timeout END_TS / §10.1 mechanical script)
- audit DRAFT 是否真改了 D1-D6 (Q3.2.1 方向反 / NOT-DO #11/#12/#13 / §10 trigger 机制 / M5↑M4↓ / 单题产出限 3-8 行)
- claude **self-review §10.1 脚本** 是否真有效 (跑过 13/13 pass; reviewer 必须独立跑一次确认结果)
- v3 / audit DRAFT 是否引入 **Round 11-specific 新偏差** (Round 10 元教训 B31 "修偏差 PR 自身造同型偏差" 是否第 N 次复发)

**本轮不审**:
- V18/V13 设计 (Round 1-5)
- 三阶段路线 / V18-clean (Round 7 user 签字)
- Round 9 决策 C (user 签)
- audit 是否该做 (user 已请求)
- v2 任何内容 (已 supersede by v3)

---

## 1. 给 reviewer 的 8 个问题

### Q1 — v3 §10.1 mechanical verify 脚本是否真有效?

v3 §10.1 新增 user 验收脚本, 含 13 个 grep check (row1-row11). claude 已 self-review 跑过, 13/13 pass.

请 reviewer 独立 run 此脚本:
```bash
cd /Users/jiaxiang/Desktop/文件/先进院文件/latent_flow/PET_LatentResidual
# 复制 §10.1 脚本到 terminal 执行
```

- 13 项是否真全 pass?
- 13 项 grep pattern 是否真覆盖 7 处 v2 修订 + Round 10 4 处增?
- row4 / row5 用 `^test "\$N_TRAIN" -ge ...` 锚行首区分 active code vs 说明文本 — 是否够精确? 是否有 LLM 滑坡风险 (codex 看到 grep 提示后绕过)?
- row10 (B31 §0.5 vs §B0 sanity range 一致性): 是否真 cross-check 两处? bash variable parse 在不同 shell 是否会失败?
- 缺什么 grep? (例如 §C3 timeout 分支的 END_TS defensive 是否有专门 check?)

### Q2 — Round 10 4 处 hard fix 真落地了吗?

| Round 10 hard fix | v3 落地位置 | 期望验证 |
|---|---|---|
| B31 §0.5 sanity range "[15, 30]" → "[10, 35]" | §0.5 行 6 | reviewer grep `"\[15, 30\]"` 应 0 命中 (除注释), `"\[10, 35\]"` ≥ 2 命中 |
| Round 9 §5.3 git add 通配 → 显式 ${SMOKE_YAML} | §E2 commit 段 | reviewer grep `git add.*V13_smoke_\*` 应 0 命中, `git add "\${SMOKE_YAML}"` ≥ 1 命中 |
| Round 9 §5.3 diff 估计 "~+60 行" → "~+80-130 行" | §E1 §8 | reviewer grep `"~\+60 行"` 应 0, `"~\+80 - 130 行"` ≥ 1 |
| item 6 "未起 Round 10 review" → "codex 未自发地起 review/audit" | §6 master + §E1 §6 attestation | reviewer grep `"未起 Round 10"` 应 0, `"codex 在 Phase A 执行期间未自发地"` ≥ 1 |

请 reviewer 逐条 grep verify, 报告任何不一致.

### Q3 — Round 10 2 处应修真落地了吗?

| Round 10 应修 | v3 落地位置 |
|---|---|
| Q4 §C3 timeout 分支补 END_TS defensive | §C3 timeout 分支 (`Round 10 Q4 defensive` 标记) |
| B31a §10 加 user 必须独立 grep verify | §10 标题改 + §10.1 mechanical script |

请 verify:
- §C3 timeout 分支 END_TS 加在 `exit 1` 之前 (LLM 多 shell 执行时仍可读)? 还是 `exit 1` 之后 (dead code)?
- §10.1 脚本是否真**强制** user 跑 (而非 optional)? 措辞 "user 验收 protocol: 跑上述脚本, 全部 ✓ 才能 push gitee" 是否够强?

### Q4 — audit DRAFT D1-D6 真落地了吗?

| Round 10 D | DRAFT 落地位置 | 期望验证 |
|---|---|---|
| D1 (B32) Q3.2.1 方向反 | §3 Q3.2.1 | grep `"init_scale_zero=False"` ≥ 1, 应描述 silent ignored / WARN 路径 |
| D2 §5 NOT-DO #11 "看不懂段标 UNCERTAIN" | §5 NOT-DO 11 | grep `"看不懂"` ≥ 1, `"silent skip"` ≥ 1 |
| D3 §10 trigger 改文件名后缀 | §10 + 顶部警告 blockquote | grep `"PENDING USER RELEASE"` ≥ 1, `"DO NOT EXECUTE"` ≥ 1, 顶部 blockquote 4 步 release 流程 |
| D4 M5 升 P0 | §1.1 必审模块表 | grep `M5.*P0` ≥ 1, 不在 §1.2 |
| D5 M4 降 P2 | §1.2 选审模块表 | grep `M4.*P2` ≥ 1, 不在 §1.1 |
| D6 单题产出限 3-8 行 | §8 总时长 + D6 注释 | grep `"3-8 行"` ≥ 1 |

请 reviewer 逐条 grep verify.

### Q5 — v3 / audit DRAFT 是否引入 v3-specific 新偏差? (Round 10 元教训核心)

Round 10 B31 = "修偏差 PR 自身造同型偏差". claude 在 Round 11 起草 v3+audit 修订时是否又复发?

请独立检查:
- v3 §10.1 mechanical script 自身是否有 bug? (例如 row10 sanity range cross-check 的 bash variable parsing 在某些 shell 失败)
- v3 §E2 commit message 模板自身是否措辞乐观 (e.g. "smoke 200-step pass" 是否仍含 7-day 推断? B23 守是否落实)
- audit DRAFT 顶部 PENDING USER RELEASE 4 步 release 流程 — 是否清楚到 codex (不是 user) 也不会误执行? `[PENDING USER RELEASE — DO NOT EXECUTE]` 标题是否够强?
- audit DRAFT §10 ACTIVATION 状态机制是否清楚区分 PENDING vs ACTIVE?
- audit DRAFT §3.2.1 改方向后, 是否引入"audit 时 codex 应试图修复 init_scale_zero 行为"的暗示 (违反 NOT-DO #5)?

如发现 v3-round11 偏差, 用 B33+ 编号列出.

### Q6 — claude self-review §10.1 13/13 pass 可信吗?

claude 报告 self-review 13/13 pass (修正 row4/row5 grep pattern 后).

请 reviewer:
- 独立 clone repo 或在 local 跑同 13 个 grep
- 比对结果是否真 13/13 pass
- 若有任何 ✗, 是 claude self-review **谎报** (Round 11 严重新偏差 = B33) 还是 grep pattern 在不同环境差异?
- (双 verify) 修正前 row4/row5 false positive 现已修正用 `^test "\$N_TRAIN" -ge 15` 锚行首 — 此 fix 是否真锁定 active code (而非误报)?

### Q7 — Round 10 5 should-fix 是否真全部落实? (cross-check user 决策 C)

user Round 10 决策 C = "全部 1-11" 含 v3 (1-5) + audit DRAFT (6-11). claude 实施 11 项.

请 verify cross-check:
- v3 修订 5 项 (B31 / git add / diff / item 6 / timeout defensive / §10.1 script) — 实际 5 还是 4? (timeout defensive + §10.1 script 是否算同 1 项 "Round 10 应修"?)
- audit DRAFT 修订 6 项 (D1-D6) — 全部落地?
- 是否有 user 决策 C 隐含但 Round 10 整合未明列的项, claude 漏修?

### Q8 — v3 + audit DRAFT 整体可用性

最后整体评估:
- v3 push gitee 后, codex 是否真能一次执行 Phase A 5 件事不出 silent bug?
- audit DRAFT 阶段 A 完成后 release, codex 是否真能一次完成 8-12 h audit 不滑坡?
- Round 11 是否是 execution + audit-prep 审稿最终终点?

---

## 2. 资料目录

按读的顺序:

### 2.1 本轮主审对象

- **[CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md)** — v3 (Round 10 修订后)
- **[CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md](./CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md)** — audit DRAFT (D1-D6 修订后)

### 2.2 上游 (本轮不审, 仅作 context)

- [REVIEW_INTEGRATION_round10_20260518.md](./REVIEW_INTEGRATION_round10_20260518.md) — Round 10 整合 (3 reviewer + user 决策 C)
- [REVIEW_INTEGRATION_round9_20260517.md](./REVIEW_INTEGRATION_round9_20260517.md) — Round 9 (B26/B-V21-scope hard fix 来源)
- [REVIEW_INTEGRATION_round8_20260517.md](./REVIEW_INTEGRATION_round8_20260517.md) — Round 8 (B21-B25 来源)

### 2.3 代码事实验证 (Round 10 已 spot-check, Round 11 二次确认)

- [train_first_hop.py:1355-1357](../../train_first_hop.py) argparse 仅 `--config` / `--resume`
- [train_first_hop.py:2404 + 2433](../../train_first_hop.py) `if step % log_interval == 0:` + `"event": "train"`
- [path_guard.py](../../pet_lr/path_guard.py) output_dir 必须 /data_2/
- [decoder_lora.py:95, 123-130](../../pet_lr/decoder_lora.py) `init_scale_zero=False` silent ignored (B32 来源)
- [V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) `log_interval: 50`, `image_aux.enabled: false`, `require_fresh_output_dir: true`
- [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) Task A 修改目标

---

## 3. 输出格式 (请 reviewer 严格遵守)

每位 reviewer 独立产出 markdown:

### 3.1 8 个问题逐条回答
对 Q1-Q8 每条**明确**给出 1 个结论:
- `APPROVE` — 真修了 / 真做了 / 真可用
- `MODIFY (说明)` — 一部分对, 还有改进空间但不阻塞 push
- `REJECT (说明)` — 没真修 / 引入新 bug, 阻塞

### 3.2 整体 verdict (双对象)
对 **v3 task md**:
- `READY TO PUSH` — 可直接 push 给 codex 执行
- `MODIFY THEN PUSH` — 列出必改项后可 push (期望 0 项, 因 self-review 13/13 pass)
- `BLOCK` — 有未修 hard blocker

对 **audit DRAFT**:
- `READY TO HOLD AS-IS` — 阶段 A 完成时 user 按 §10 4 步 release 流程即可
- `MODIFY THEN HOLD` — 列出必改项
- `RESCOPE / DELETE` — 范围 / 价值有根本问题

### 3.3 self-review §10.1 独立验证结果 (Round 11 必答)
reviewer 必须**独立跑**一次 §10.1 13 个 grep, 报告:
- 实际通过项数 / 13
- 任何 ✗ 列出 + 是 claude 谎报 / pattern bug / 真 bug
- 若 13/13 pass, 强化 v3 可信度

### 3.4 v3-round11 新偏差 (B33+)
若发现 Round 10 修订引入新偏差 (B31 同型第 N 次复发), 列编号 + 形态 + 修法.

### 3.5 (可选) 替代方案

---

## 4. 约束与提醒

- **本轮不重审** Round 1-10 已签内容
- **不假设** 你能跑代码 / 看 wandb / 触 GPU (但可跑 grep / verify 文档)
- **特别关注**:
  - **Q1 + Q6**: §10.1 mechanical script 是否真有效 + claude self-review 是否谎报
  - **Q5**: Round 10 元教训 "修偏差 PR 自身造同型偏差" 第 N 次复发?
  - **Q2 + Q3**: 4 hard fix + 2 应修是否每条 grep 都过
- **优先质疑**:
  - claude self-review 13/13 pass 是 false reassurance 还是真验证 (Q1 + Q6 联动)
  - audit DRAFT 顶部 [DRAFT] → [PENDING USER RELEASE] 改名是否真防 codex 误执行
  - v3 §E2 commit message 模板含 "smoke 200-step pass (img=0, finite, ≤ 15min)" — 这是否仍是 B23 同型 (隐含 7-day 推断)?

---

## 5. 给 reviewer 的硬约束摘要 (standing, 不可推荐违反)

1. ≤ 3 并行训练任务
2. V18 不动 (slot 1)
3. V18b 永久撤销
4. 阈值不改 (Round 7 user 全推荐)
5. V18-clean 阶段 C 才 pre-register (Round 7 B18)
6. user Round 8 Option B + Round 9 C + Round 10 C 已签, 不可推翻
7. **Round 11 是 execution + audit-prep 审稿绝对终点**. 若 v3 还需 v4, 必须有 hard blocker, 否则 user 单方面拍板 push, 不再起 Round 12
8. 不审 audit 是否该做 (user 已请求)

违反任意一条 → reviewer 提议自动作废.

---

## 6. 本轮元说明

| 轮次 | 范围 | 是否 user 仲裁? |
|---|---|---|
| 1-7 | design | YES (R7 全推荐) |
| 8 | execution v1 | YES (R8 Option B) |
| 9 | execution v2 | YES (R9 C) |
| 10 | execution v3 + audit DRAFT | YES (R10 C 全部 1-11) |
| **11** | **execution v3+R10 修订 + audit DRAFT D1-D6 修订** | **决于本轮 verdict** |

预期 Round 11 outcome:
- 最佳: 3/3 READY TO PUSH (v3) + 3/3 READY TO HOLD (audit DRAFT) → 直接 push v3, 保留 audit DRAFT
- 中等: 3/3 MODIFY THEN PUSH 给 ≤2 should-fix → claude 速改 (~10 min) → push
- 最差: v3 / audit DRAFT 有新 hard blocker → v4 起草 (但 standing rule #7 立 "Round 11 终点", 是 user 决策点)

**Round 11 与 Round 10 不同点**:
- Round 10 是 reviewer 首次审 v3 + audit DRAFT
- Round 11 验证 Round 10 修订真落地 + claude self-review 可信度
- 引入 mechanical script (§10.1) 作为反 LLM 滑坡的客观 verify, 而非依赖 reviewer 主观读
