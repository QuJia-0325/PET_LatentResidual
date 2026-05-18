# Round 11 Review Integration — v3 (R10 hard-fixed) + audit DRAFT (D1-D6) 评审交叉核对

- date: 2026-05-18
- branch: foc_lite_hop0 (v3 + audit DRAFT 仍未 push)
- 主审对象: [CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md) + [CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md](./CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md)
- 三 reviewer 独立评审 (含 claude self-review §10.1 跑)
- 上游: [REVIEW_INTEGRATION_round10_20260518.md](./REVIEW_INTEGRATION_round10_20260518.md) (user 决策 C 全部 1-11)

---

## 0. 一行结论

**3/3 reviewer 独立 verify §10.1 13/13 真 pass** (claude self-review **未谎报**) + 3/3 verdict = v3 MODIFY THEN PUSH + audit DRAFT MODIFY THEN HOLD. 

**新发现 5 个偏差 B33a/B33b/B33c/B34/B35**, 全部是 **B27/B30/B31 同型在不同层级复发** (印证 Round 10 元教训 N=3-4 层深度). 但全部 **cosmetic / non-blocker**, ~15-20 min 修完即可 push.

**关键**: agent3 发现 **B33b (§10.1 script 自身非 assert, 总 exit 0)** — 这是 B30 同型第 3 层复发, 必修, 否则 user 决策 C 字面失败.

**Round 11 是终点**: 修 5 个 B33+ 共 ~20 min 后, **直接 push, 不起 Round 12**.

| 维度 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| §10.1 13/13 实测 | 13/13 ✓ | 13/13 ✓ | 13/13 ✓ | **3/3 真 pass** ✅ |
| Q1 §10.1 有效性 | APPROVE w/ count caveat | APPROVE | APPROVE w/ should-fix | **3/3 APPROVE** ✅ |
| Q2 R10 4 hard fix | APPROVE | APPROVE | APPROVE 4/4 | **3/3 ✅** |
| Q3 R10 2 应修 | APPROVE | APPROVE | APPROVE | **3/3 ✅** |
| Q4 audit DRAFT D1-D6 | APPROVE | APPROVE | APPROVE 6/6 | **3/3 ✅** |
| **Q5 新偏差** | **0 发现** (轻量) | **B33+B34+B35** | **B33a+B33b+B33c** | **agent2/3 各发现独立** ⚠️ |
| Q6 claude self-review 可信 | APPROVE | APPROVE | APPROVE | **3/3 非谎报** ✅ |
| Q7 user 决策 C cross-check | APPROVE | MODIFY (计数链漂移) | APPROVE | 2/3 APPROVE |
| Q8 整体可用性 | READY/HOLD | MODIFY THEN PUSH/HOLD | MODIFY THEN PUSH/HOLD | **2/3 应修** |
| v3 verdict | **READY TO PUSH** | MODIFY THEN PUSH | MODIFY THEN PUSH | **2/3 MODIFY** ⚠️ |
| audit DRAFT verdict | **READY TO HOLD AS-IS** | MODIFY THEN HOLD | MODIFY THEN HOLD | **2/3 MODIFY** ⚠️ |

---

## 1. claude self-review 13/13 独立 reproduction (Q1 + Q6)

3 reviewer 都在 macOS 本地独立跑了 §10.1 13 项 grep, **全部 13/13 ✓ 通过**. 详细 row 状态:

```
✓ row1  (B30 SUPPORTED_FLAGS assert)
✓ row2  (B26 log_interval=10)
✓ row3  (B-getctime START_TS)
✓ row4  (B26 阈值 ≥15)  +  active 残留 ≥50 = 0
✓ row5  (B-getctime PASS_5)  +  active 残留 getctime = 0
✓ row6  (B-V21-scope 白名单)
✓ row7a (B27 §2.3 row 1 "rank=32 也可能不够")
✓ row7b (B28 区间 "0.05 - 3 dB")
✓ row7c (B29 attestation 实际 16 项)
✓ row8a (Round 9 §5.3 git add 显式)
✓ row8b (Round 9 §5.3 diff +80-130 行)
✓ row9a (Round 10 Q4 timeout END_TS defensive)
✓ row9b (Round 10 item 6 措辞)
✓ row10 (B31 §0.5=§B0 sanity range = [10, 35])
✓ row11 (B12 守 trigger +0.05/+0.30)
```

**claude self-review 未谎报**. 但 agent3 发现 13 项 ✓ ≠ 13 row pattern (实际 row 1-11 展开 15 个 ✓ echo). 这是 agent2 B34 计数 stale 的局部例 (见 §2.3).

---

## 2. Round 11 5 个新偏差 (B33a/B33b/B33c/B34/B35) 全部 verify 真

### 2.1 B33a — §10.1 row 10 vacuous-pass false positive (agent3 独家, HIGH 严重度)

**代码事实**:
- §10.1 row 10 用 `range_summary=$(grep ...); range_script=$(grep ...); test "$range_summary" = "$range_script"`
- claude **terminal 实测** `a=""; b=""; test "$a" = "$b"` → exit 0 → **silently ✓ pass**
- 若 user 重排 §0.5 表行号导致 `| 6 | §B0 |` 不命中, 或 §B0 注释里改了 "expected" 措辞 → 两 grep 都返回空 → vacuous pass → B31 sanity range 不一致**不会被 catch**

**讽刺度**: B31 (Round 10) 是 "摘要表 vs 正文不一致", §10.1 row 10 修法是 cross-check 一致性. 但 cross-check 自身在 corner case (两 grep 都空) silently pass → **B27/B31 同型第 4 层复发**.

**修法** (agent3 提议):
```bash
range_summary=$(grep '| 6 | §B0 |' "$F" | grep -oE '\[[0-9]+, [0-9]+\]')
range_script=$(grep -oE 'expected \[[0-9]+, [0-9]+\]' "$F" | head -1 | grep -oE '\[[0-9]+, [0-9]+\]')
if [ -z "$range_summary" ] || [ -z "$range_script" ]; then
    echo "✗ row10 grep 返回空 (§0.5 表结构或 §B0 注释可能已变): summary='$range_summary' script='$range_script'"
elif [ "$range_summary" = "$range_script" ]; then
    echo "✓ row10 (B31 §0.5=§B0 range = $range_summary)"
else
    echo "✗ row10 不一致: §0.5=$range_summary vs §B0=$range_script"
fi
```

### 2.2 B33b — §10.1 script 整体非 assert (agent3 独家, HIGH 严重度)

**代码事实**: 每条 check 用 `grep -qE ... && echo "✓" || echo "✗"`. grep fail 时 `||` 接管 → `echo "✗"` exit 0 → 整个表达式 exit 0 → 外层无错传播 → **脚本总是 exit 0**, user 必须**视觉**数 ✗ 行.

**讽刺度**: Round 9 B30 = "self-check 非 assert", Round 10 修法 = 引入 §10.1 script. 但 script 自身仍是 echo-only, 不能在 fail 时 exit 1. **B30 同型第 3 层复发**: B30 (Round 9 markdown checkbox) → B31a (Round 10 user 必须独立 grep) → §10.1 script (Round 10 写了 script) → B33b (script 自身非 assert).

**修法** (agent3 提议):
```bash
ERR=0
# 在每个 ✗ echo 后加 ERR=$((ERR+1))
# 或包装函数: check() { if ! eval "$1"; then echo "✗ $2"; ERR=$((ERR+1)); else echo "✓ $2"; fi; }
...
echo "=== 验证结束 ==="
if [ "$ERR" -gt 0 ]; then
    echo "FAIL: $ERR 项未通过. 不可 push." >&2
    exit 1
fi
echo "PASS: 13/13. user 可 push gitee."
```

修后 user 可 `bash verify.sh && git push` 链式调用, CI 直接 gate `$?`.

### 2.3 B34 — v3 摘要计数 stale "7 处" 应 "9 处" (agent2 独家, MEDIUM)

**代码事实**: claude terminal 实测:
- `grep -nE "7 处|9 处" v3.md` 命中 3 处 `7 处`: line 8 / line 19 / line 23
- §0.5 表实际**9 行** (R9 7 项 + R10 增 2 项: 行 8 git add/diff, 行 9 timeout/item 6)
- 计数链漂移: prompt → integration → v3 §0.5 三处口径不一

**讽刺度**: B31 = "摘要表 vs 正文不一致". v3 修 B31 时把 sanity range 改一致, 但 §0.5 表行数从 7 变 9 时, 上游叙述 line 8/19/23 仍写 "7 处". **B27/B31 同型第 4 层复发** (与 B33a 不同侧面).

**修法** (~2 min):
- line 8: "仅 7 处局部修订" → "仅 9 处局部修订"
- line 19: "7 处 v3 修订" → "9 处 v3 修订 (Round 9 修 7 + Round 10 增 2)"
- line 23: "(7 处)" → "(9 处)"

### 2.4 B33c — audit DRAFT Q3.2.1 违反 NOT-DO #9 (agent3 独家, MEDIUM)

**代码事实**: claude terminal 实测:
- `grep -n "应改为 raise" audit_DRAFT.md` 命中 line 150 Q3.2.1: "yaml 字段是否**应改为 raise 而非 print**?"
- audit §5 NOT-DO #9: "不在 audit 报告里写'建议立即修 X'"
- Q3.2.1 后半句要求 codex 输出 design recommendation = **违反 NOT-DO #9**

**讽刺度**: D1 修 B32 (Round 10) 时方向反过来, 但**修法本身**引入 audit DRAFT 内部不自洽 (§3 question vs §5 NOT-DO). **B27/B31 同型第 5 层** (audit DRAFT 内自相矛盾).

**修法** (~3 min):
Q3.2.1 改为纯事实+影响, 移除 design recommendation:
> "yaml `init_scale_zero=False` 路径在 decoder_lora.py:95, 123-130 的实际行为: 字段是否 silently 被忽略 (仅 print WARN 后仍走 zero-init)? 实际行为与字段名/用户期望 (False = 非 zero init) 的 gap 是什么? 影响阶段 C V18-clean 是否能用此 yaml 字段调试 LoRA init? **(audit 仅报现状与 gap, 不建议具体 patch; 修复决策属 user, 见 §5 NOT-DO #9)**"

### 2.5 B35 — 双环境路径 by design 但未显式标注 (agent2 独家, LOW)

**事实**: §C0/§C1/§E2 用 `/home/qujiaxiang/...` (服务器), §10.1 user 验收脚本用 `/Users/jiaxiang/Desktop/...` (macOS). **by design** (不同执行者), 但未显式标注 → 第三方 reviewer 易误读为 inconsistency.

**修法** (~1 min): §10.1 节首加 "(在 user 本地 macOS 仓库执行, 不是服务器)" 注释.

---

## 3. agent2 独家 B33 (Round 10 reviewer false negative)

agent2 在 Round 11 自查时发现, 自己在 **Round 10** 把 `_assert_v18_step0_equivalence` / `assert_decoder_frozen` 标为"全 repo 零命中" — **agent2 当时 grep scope 漏读** (实际两函数都存在). claude (本 round 整合) 在 Round 10 整合 §4 已纠正此 false negative.

**Round 11 状态**:
- agent2 B33 实质 = **Round 10 reviewer agent3 的 false negative** (我在 Round 10 整合 §4 已标注 "agent3 grep scope 漏了")
- audit DRAFT 起草时 claude (drafter) 写了**真存在**的符号, **没被 Round 10 false negative 污染**
- 所以 audit DRAFT Q3.1.3 / Q3.2.3 **不需要 D 修订改** — 它们引用的函数都真实存在
- agent2 提议 "Q3.2.3 加注 startup-only" 是**改善**性建议 (codex audit 时更明确), 非必修

**结论**: agent2 B33 是 Round 10 偏差的 forensic 复盘, **不在 Round 11 新增修订列表**, 但 Round 11 standing rule 新增: "任何 reviewer 关于 'X 不存在' 的断言必须贴 grep 证据".

---

## 4. user 决策 C cross-check (Q7, agent2 MODIFY)

agent2 指出: prompt 写 "Q7 v3 修订 5 项... 实际 5 还是 4?", Round 10 整合 §9 列 4 hard + 2 应修 = 6 项, v3 §0.5 合并为 9 行 (R9 7 + R10 增 2).

**实质**: 计数口径在 3 处不一致, 但**没有 user 决策 C 隐含但漏修的项**. 是 B34 同型 (摘要 stale), 修 B34 时顺手把所有计数链统一.

---

## 5. 修订清单 (3 必修 + 2 应修, ~20 min)

### 5.1 必修 (v3 push 前)

| # | 位置 | 修法 | 来源 | 时间 |
|---|---|---|---|---|
| 1 | v3 §10.1 row 10 | 加 `-z` 空值守, 分 3 case | agent3 B33a | 3 min |
| 2 | v3 §10.1 末尾 | 加 `ERR` 计数 + `exit 1` if `$ERR > 0` | agent3 B33b | 5 min |
| 3 | audit DRAFT §3 Q3.2.1 | 删 "应改为 raise", 改纯事实+gap, 标 "修复决策属 user" | agent3 B33c | 3 min |

### 5.2 应修 (v3 push 前, 不修也不阻塞但建议修)

| # | 位置 | 修法 | 来源 | 时间 |
|---|---|---|---|---|
| 4 | v3 line 8/19/23 + §0.5 标题 | "7 处" → "9 处" 全 3 处统一 | agent2 B34 | 2 min |
| 5 | v3 §10.1 节首 | 加 "(在 user 本地 macOS 仓库执行)" | agent2 B35 | 1 min |

### 5.3 不修 (低优先级 / agent2 提议但 Round 11 整合后认定非必)

- agent2 Q1 "13/13 pass 计数不严谨" → cosmetic 措辞, 实质 13 row pattern × 15 echo 输出, 计数差异是 row7c (1 row 但有 attestation 子计数)
- agent2 B33 (Round 10 false negative 关于符号存在) → 历史复盘, 不在 Round 11 修订范围, standing rule 新增即可

---

## 6. user 决策点

只有 1 个真正需要 user 拍板:

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | 修订范围 | A: 仅必修 1-3 (~11 min) <br>B: 必修 + 应修 1-5 (~14 min) <br>C: 全部 1-5 + standing rule 新增 | **B** (B33b 是 B30 同型第 3 层复发不修等于 user 决策 C 字面失败; B34 修是 ROI 高的 cosmetic) |

**A 是绝对底线** (修 3 个 should-fix, 跳过 cosmetic).
**B 推荐** (~14 min, ROI 最高).
**C 加 standing rule**: B30 mechanical script 必须 assert (有 ERR 计数 + exit 1) + B-grep-evidence (reviewer 断言 "不存在" 必贴 grep 命中) — 这两条加进 [V18_design_rationale.md §5.2](PET_LatentResidual/review/0517/V18_decoder_lora/V18_design_rationale.md) 作为永久教训.

回复: `1=B` 或 `按推荐` → claude 修订 → user 过一眼 → **直接 push gitee, 不起 Round 12** (standing rule #7).

---

## 7. Round 11 元教训

### 7.1 偏差复发深度统计 (Round 8 → Round 11)

| 形态家族 | Round 8 首次 | 复发深度 | 最新形态 |
|---|---|---|---|
| **B27/B31 摘要-正文不一致** | B25 文档结构假设 | 第 5 层 | B33a row10 vacuous pass / B33c audit DRAFT 内自矛盾 / B34 计数 stale |
| **B30 self-check 非 assert** | B30 markdown checkbox | 第 3 层 | B33b script 整体 echo-only |
| **B21 杜撰 / 凭记忆** | B21 CLI flag 杜撰 | 第 2 层 | (Round 10 agent3 关于符号存在的 false negative, 已 forensic 在 R10 整合) |

**核心发现**: mechanical script 是必要但不充分. **每修一层偏差, 自身有概率引入下一层同型偏差**. 真正闭环需要:
1. **assert-on-fail** (exit code propagation, agent3 B33b)
2. **defensive null check** (empty-vs-empty 视为 fail, agent3 B33a)
3. **evidence-required claims** (reviewer "X 不存在" 必贴 grep, agent2 B33 forensic)

### 7.2 Standing rule 建议 (永久加 design_rationale §5.2)

| # | rule | 触发场景 |
|---|---|---|
| B36 | mechanical verify script 必须 assert: 末尾 `if ERR > 0; exit 1` | 任何 user 验收 / CI gate |
| B37 | bash 变量比较前必须 `-z` 空值守 (避免 vacuous pass) | 任何 `test "$a" = "$b"` |
| B38 | reviewer "X 不存在" 断言必须贴 `grep -rn` 命中行号; 无 evidence 降级为 "audit 中 verify" | 任何 reviewer 报告 |
| B39 | 修订 PR 必须设 mechanical check: 摘要数 = 正文表格行数 = 5 句话总览数 | 任何含 "N 处修订" 的文档 |

这 4 条加进 design_rationale, 未来 Round 12+ 起草时 claude 自检.

### 7.3 Round 1-11 总览

| 轮次 | 范围 | 新偏差 | 元教训 |
|---|---|---|---|
| 1-5 | design | B1-B10 | 用代理量当 attackable |
| 6 | V18b draft | B-LoRA-resume blocker | reviewer 不读源码 |
| 7 | 架构 | B11-B14 | 4 个代码事实揭穿假设 |
| 8 | v1 execution | B15-B25 | 凭记忆杜撰 CLI/path |
| 9 | v2 execution | B26-B30 | quantitative threshold 不 anchor 代码 |
| 10 | v3 + audit DRAFT | B31, B32 | 修偏差 PR 自身造同型偏差 |
| **11** | **v3 R10fix + D1-D6** | **B33a/B33b/B33c/B34/B35** | **mechanical script 必要不充分, 必须 assert-on-fail + 空值守 + evidence-required** |

---

## 8. 立即下一步

待 user 回复决策 1 → claude:
1. 修 v3 §10.1 row 10 (B33a, 3 min)
2. 修 v3 §10.1 末尾 ERR + exit 1 (B33b, 5 min)
3. 修 audit DRAFT Q3.2.1 (B33c, 3 min)
4. [若选 B/C] 修 v3 计数 7→9 (B34, 2 min)
5. [若选 B/C] 修 v3 §10.1 节首加 macOS 注释 (B35, 1 min)
6. [若选 C] 加 B36-B39 standing rule 进 design_rationale §5.2 (5 min)

总 ~14-19 min, 修完 → user 过一眼 → **直接 push gitee** (standing rule #7 立 Round 11 是终点).
