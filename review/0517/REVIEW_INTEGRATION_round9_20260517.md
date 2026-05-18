# Round 9 Review Integration — CODEX_TASK_PHASE_A_v2 评审交叉核对

- date: 2026-05-18 凌晨
- branch: foc_lite_hop0 (v2 仍未 push)
- 主审对象: [CODEX_TASK_PHASE_A_v2_20260517.md](./CODEX_TASK_PHASE_A_v2_20260517.md)
- 三位独立 reviewer: agent1 / agent2 / agent3
- 上游: [REVIEW_INTEGRATION_round8_20260517.md](./REVIEW_INTEGRATION_round8_20260517.md) (列 Round 8 5 个 hard fix + user 决策 Option B)

---

## 0. 一行结论

**3/3 reviewer 一致定位 B26 (`log_interval=50` 让 PASS_2 ≥50 物理上不可达) 为新 hard blocker**, 加 2-4 个 should-fix. agent1 verdict=MODIFY, agent2 verdict=**BLOCK**, agent3 verdict=MODIFY. 

**好消息**: v2 在结构性 hard fix (B21 CLI / B22 path_guard / B25 文档结构) 全部到位; 剩余问题是**局部**执行细节, 5 个 P0 patch 共 ~25 行 markdown 即可修完, 直接 v3 + push, 不需 Round 10.

| 维度 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| Q1 CLI flag (B21) | APPROVE | APPROVE | APPROVE | **3/3 已修** ✅ |
| Q2 path_guard + yaml 嵌套 (B22) | APPROVE | APPROVE | APPROVE | **3/3 已修** ✅ |
| Q3 metrics pass 准则 | **REJECT** (B26 + B27 getctime) | **REJECT** (B27+B28) | **REJECT** (B26 hard) | **3/3 reject, B26 hard blocker** 🔴 |
| Q4 V18_design_rationale 结构 | MODIFY (A1 §2.2 EV stealth + A3 §5 重组缺全文) | MODIFY (A2 §2.4 三档 archaeology) | APPROVE w/ should-fix (B27 §2.3 row 1 残留 rank=8) | **2/3 MODIFY, 1/3 边界 APPROVE** ⚠️ |
| Q5 V21 retire | APPROVE w/ note | **REJECT** (B26: grep scope 全目录 169 hits ≠ 期望 [5,50]) | APPROVE | **2/3 vs 1/3, 需澄清** ⚠️ |
| Q6 NOT-DO + 决策树 | MODIFY (attestation 12 vs master 16 错位 = B29) | MODIFY (补 4 fail 场景) | APPROVE | **2/3 MODIFY** |
| Q7 Option B 措辞 | APPROVE w/ note | APPROVE w/ minor | APPROVE w/ note | **3/3 已修** ✅ |
| Q8 git commit 范围 | MODIFY (通配符 → 显式 ${TS}) | MODIFY (同 agent1) | MODIFY (B29 diff estimate ~60 实际 ~110) | **3/3 MODIFY** |
| Q9 新偏差 | 4 (B26-B29) | 4 (B26-B29) | 5 (B26-B30) | **B26 hard, B27-B30 各异** |
| Overall | MODIFY THEN PUSH | **BLOCK** | MODIFY THEN PUSH | **2/3 MODIFY, 1 BLOCK** |

---

## 1. 3/3 共识: B26 是新 hard blocker (smoke 100% spurious fail)

### 1.1 代码事实 (agent1+2+3 独立 verify, 全部锁定)

```python
# train_first_hop.py 训练循环
for step in range(start_step + 1, max_steps + 1):     # V13 from-scratch: step 1..200
    ...
    if step % log_interval == 0:                      # gate
        metrics_payload = {"event": "train", ...}     # 仅在此条件下写
        metrics_fp.write(json.dumps(metrics_payload, ...) + "\n")
```

V13 yaml `log_interval: 50`. V13 smoke v2 patch 改了 `max_steps=200` 但**没改** `log_interval`. → 200 step 内 `step % 50 == 0` 的 step = {50, 100, 150, 200} = **恰好 4 行 train event**.

v2 §C4 PASS_2 要求 `N_TRAIN ≥ 50` → **物理上不可能满足** → smoke 100% spurious FAIL → 触发 §C5 STOP → 永远不进 Task D → 阶段 A 死锁.

### 1.2 修法 (3/3 共识 Option A)

§C1 python yaml patch 加 1 行 + §C4 PASS_2 阈值改:

```python
# §C1 python 脚本现有 4 字段之后追加:
cfg["training"]["log_interval"] = 10    # smoke 200 step / 10 = 20 行 jsonl
```

```bash
# §C4 PASS_2 改:
test "$N_TRAIN" -ge 15 || { echo "FAIL_2: only $N_TRAIN train rows (expected ≥15). STOP."; exit 1; }
```

理由: 守 B12 spirit (不通过抬高 max_steps 间接放宽; 改 log_interval 是 smoke-only 配置, 不污染 V13 full launch).

### 1.3 B26 元教训

B26 与 Round 8 B21 (CLI flag 杜撰) **完全同型**: 设了一个 quantitative threshold (CLI flag 存在/N_TRAIN ≥ 50), 没去 verify code-level 决定该 threshold 是否可达的字段 (argparse/log_interval). 

**Standing rule (待加进 design_rationale §5.2)**: 任何 pass criterion 形如 "≥ N rows / files / calls" MUST 推导自 `floor(max_steps / log_interval) - margin`, 且 max_steps + log_interval 必须**同时**在 patched yaml 中 verify.

---

## 2. agent2 独家 hard blocker: V21 grep scope 全目录扫描会炸 (B-V21-scope)

### 2.1 agent2 发现

v2 §B0 写:
```bash
N=$(grep -rn "V21" review/0517/ 2>/dev/null | grep -v ".pyc:" | wc -l)
if [ "$N" -lt 5 ] || [ "$N" -gt 50 ]; then
    echo "FAIL: V21 hit count $N out of expected [5, 50]. STOP."
fi
```

agent2 实测 `grep -rn "V21" review/0517/**/*.md` = **169 hits**. 原因: `review/0517/` 全目录扫描会扫到所有 round prompt + v1/v2 task + Round 5/6/7/8 integration 等历史 markdown, 全部含 V21 引用 (讨论 V21 retire 行为本身).

`169 ∉ [5, 50]` → §B0 STOP → 阶段 A 死锁 #2.

### 2.2 agent1 / agent3 为何没 catch

- agent1 信任 Round 8 reviewer 给的 "21 hits" 数 (Round 8 当时大概也只扫了白名单 4 个文件)
- agent3 同样信任 Round 8 数据未 re-verify

agent2 是唯一独立 re-verify 实际 grep 结果的 reviewer. **这是 Round 9 最有价值的发现**.

### 2.3 修法 (采纳 agent2)

§B0 grep 改为**只对 4 白名单文件**, 不对全目录:

```bash
# 仅 grep 白名单
WHITELIST=(
  "review/0517/REVIEW_INTEGRATION_round4_20260517.md"
  "review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md"
  "review/0517/CODEX_RUNBOOK_V18_20260517.md"
  "review/0517/V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md"
)

for f in "${WHITELIST[@]}"; do
    hits=$(grep -c "V21" "$f" 2>/dev/null || echo 0)
    echo "$f: $hits hits"
done

# Sanity: 白名单总 hits 应在 [15, 30] 区间 (Round 8 实测 21)
TOTAL=$(printf '%s\n' "${WHITELIST[@]}" | xargs grep -c "V21" 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Total whitelist V21 hits: $TOTAL"
if [ "$TOTAL" -lt 15 ] || [ "$TOTAL" -gt 30 ]; then
    echo "FAIL: whitelist V21 hit count $TOTAL out of expected [15, 30]. STOP."
    exit 1
fi
```

非白名单文档可单独输出报告但**不**作为 STOP 条件.

---

## 3. agent1 独家 (B28 §2.2 EV stealth 下调)

v2 §A1 把 V18_design_rationale §2.2 EV 表里 V18 行 "0.5 - 2.0 dB" 改为 "0.05 - 0.30 dB (区间下界 0.5 dB × 10%)". 

agent1 论证: 
- "× 10%" multiplier 在 §2.4 / prior round / R7 user 签字中都**不存在**
- 改动效果: V18 success ceiling 从 +2.0 dB 砍到 +0.30 dB → V18 最好仅 PARTIAL, 不再可能 SUCCESS
- 形式上是 "修 stale fact", 实际是 EV 重估 → **B12 擦边 (不改 §2.3 trigger 但改 §2.2 EV)**

agent2/agent3 未发现这条 (聚焦在 metrics/V21 grep). 但 agent1 论证扎实, **必修**.

**修法** (agent1 Option A): §2.2 EV 行改 "0.05 - 3.0 dB (见 §2.4 区间, 中位 0.1-0.3 dB)", 删 "× 10%". 与 §2.4 文字 EV 区间 [0.05, 3] dB 保持一致, 不偷偷下调.

---

## 4. agent3 独家 (B27 §2.3 row 1 残留 rank=8)

agent3 verify V18_design_rationale §2.3 行 1:
```
| LoRA 容量太小 (rank=8 不够) | ΔPSNR ∈ [+0.05, +0.30] | 跑 V18-r16 / V18-r32 sweep |
```

v2 §A1 patch 表只改 §2.1 行和 §2.2 行, **不动** §2.3 row 1. 修完后文档自相矛盾:
- §2.1: "V18 uses rank=32"
- §2.3 row 1: "failure mode = rank=8 不够" + "remediation = 跑 V18-r32 sweep" (V18 IS r32)

这是 **B25 在 v2 修 B25 过程中再次复发** — agent3 称为 B27 (我们整合后保留这编号).

**修法** (agent3): §A1 patch 表加第 3 行:
```
| §2.3 行 1: "LoRA 容量太小 (rank=8 不够)... 跑 V18-r16/V18-r32 sweep" 
| → "LoRA 容量仍偏小 (rank=32 也不够). ΔPSNR ∈ [+0.05, +0.30] | 跑 V18-r64 sweep (推迟到阶段 B)"
```

或保留原行 + 加 footnote `[Round 5 update: V18 actual rank=32; 本行记录 Round 1 hypothesis under rank=8]`.

---

## 5. 三 reviewer 共识 should-fix (3 项)

### 5.1 B27/B28 getctime Linux 语义 (agent1+2+3 都提)

v2 §C4 PASS_5 用 `os.path.getctime - getmtime` 估 elapsed. agent1/agent2/agent3 一致指出 Linux ext4/xfs 上 ctime = inode metadata change time, 不是 creation. actively-written file 的 ctime ≈ mtime, elapsed ≈ 0 → trivially pass, 失去检测意义.

**修法**: 用 shell `START_TS=$(date +%s)` (§C2 launch 前) + `END_TS=$(date +%s)` (§C3 wait 后) + `ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))`.

### 5.2 B29 attestation 错位 (agent1 独家但论证强)

master §6 = 16 项 NOT-DO; §E1 §6 attestation = 12 项. multi 4 (item 13-16) 没强制 codex 勾.

**修法**: §E1 §6 扩到 16 项 (与 master 一一对应).

### 5.3 git add 通配符 + diff 估计 (agent1+agent3 提)

- v2 `git add review/0516/.../smoke_runs/V13_smoke_*.yaml` 通配可能匹配旧 smoke 残留
- **修法**: 改 `git add "${SMOKE_YAML}"` 单文件
- v2 §E1 §8 "(~+60 行)" 实际是 ~+110 行 (agent3 算: §2.4 ~50 + §5.2 ~55 + §5 重组 ~4 + §2.1 ~2)
- **修法**: 改 "(~+80 - 130 行)" 范围, 避免 codex 看到 ~110 误判 "意外改动"

---

## 6. 各 reviewer 独家 (低优先级, 可选)

| reviewer | 提议 | 优先级 |
|---|---|---|
| agent1 | §6 item 6 "不起 Round 9" → "不起 Round 10" stale wording | should-fix |
| agent2 | §7 失败决策树补 4 场景: V18 崩溃 / V18 到 200K / git push 网络失败 / metrics.jsonl 0 行 | should-fix |
| agent3 | §C0 SUPPORTED_FLAGS 改 assert 而非 print (B30 non-asserting spot-check) | should-fix |
| agent3 | §A1 "Round 4 audit" 历史归因未 verify, 建议删 | should-fix |
| agent1 | §B1 加正例规则 "段落开头 V21 = / V21 fallback → 设计描述" | minor |
| agent1 | commit message "healthy 度" → "稳定性评估" | minor |

---

## 7. 新偏差汇总 (Round 9 识别)

| # | 名称 | 来源 | 形态 |
|---|---|---|---|
| **B26 (HARD)** | log_interval / max_steps 联动盲区 | 3/3 一致 | smoke 200 step + log_interval=50 = 4 行 train event, PASS_2 ≥50 物理不可达 (= B21 同型, threshold 没 anchor 代码字段) |
| **B-V21-scope (HARD)** | V21 grep scope = 全目录 | agent2 独家 | review/0517/ 全目录 V21 hits=169 远超 [5,50] sanity range, §B0 STOP |
| **B27** | §2.3 row 1 残留 rank=8 (B25 在 B25 修复中复发) | agent3 独家 | v2 §A1 只修 §2.1/§2.2, §2.3 row 1 仍写 rank=8 → 文档自相矛盾 |
| **B28** | §2.2 EV stealth 下调 (B12 擦边) | agent1 独家 | "0.5-2.0 dB → 0.05-0.30 dB (× 10%)" multiplier 杜撰, V18 success ceiling 被偷砍 10× |
| **B29** | master NOT-DO 与 attestation 错位 | agent1 独家 | master §6 16 项, attestation §E1 §6 12 项, 错 4 |
| **B30** | non-asserting spot-check | agent3 独家 | §C0 SUPPORTED_FLAGS 只 print 不 assert, codex 可忽略不符 |
| **B-getctime** | Linux ctime 语义假设 | agent1+2+3 一致 | macOS 心智模型迁移 Linux 失败, PASS_5 trivially pass |
| **B-diff-estimate** | doc diff 估计与实际不符 | agent3 独家 | "(~+60 行)" 实际 ~110 行, 可能触发 codex "意外改动" 警报 |

**最严重**: B26 (3/3 共识, 物理死锁) + B-V21-scope (agent2 独家但 agent1/3 未 re-verify → 极易遗漏的高致命 bug).

**元教训**: B26 与 B21 同型 + B27 是 B25 复发 + B28 是 B12 擦边 → **Round 8 修复偏差时**, claude **再次**犯了同类偏差. 这强烈暗示 Round 8 standing rules (B21/B25) 写得不够具体, 必须升级为可机械执行的 checklist.

---

## 8. user 决策点

只有 1 个真正需要 user 拍板:

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | v3 修复范围 | A: 仅修 1 个 hard blocker (B26) <br>B: 修 2 个 hard blocker (B26 + B-V21-scope) <br>C: 修 2 hard + 5 should-fix (B27/B28/B29/B30/B-getctime) <br>D: 全修 (含低优先级独家) | **C** |

**B 是绝对底线**: 不修 B-V21-scope, codex 在 §B0 就 STOP, 跟不修 B26 同样致命. agent2 独家发现, 不修等于忽视.

**C 是 claude 推荐**: 5 个 should-fix 总共 ~15-25 行 markdown, ROI 极高; B29 (attestation 错位) 不修就是给 LLM 滑坡留口子; B-getctime 不修则 PASS_5 失去检测意义.

**D (全修)** 风险: 涉及 commit message 措辞 + §7 决策树扩充 + §6 item 6 stale + §A1 历史归因等, 改动面较大, 可能引入新 B25/B26.

回复: `1=C` 或 `按推荐` 或 `1=D` 等.

---

## 9. v3 修订工作量估计

按 user 选 C (推荐):

| 修复项 | 修改位置 | 估计行数 |
|---|---|---|
| B26 hard fix | §C1 (+1 行 python) + §C4 PASS_2 (改阈值) | 3 |
| B-V21-scope hard fix | §B0 (重写 grep 逻辑) | 15 |
| B27 §2.3 row 1 残留 | §A1 patch 表 (+1 row) | 5 |
| B28 §2.2 EV stealth | §A1 patch 表第 2 row (改措辞) | 3 |
| B29 attestation 错位 | §E1 §6 (扩 12→16 项) | 6 |
| B30 spot-check assert | §C0 (改 print 为 if-then-exit) | 5 |
| B-getctime | §C2 (+START_TS) + §C4 PASS_5 (重写) | 10 |
| **总** | | **~47 行** |

修订时间估计 ~30 min. 修订后 → user 过一眼 → **直接 push gitee** (不再 Round 10, B26+B-V21-scope hard fix 后无 hard blocker, 三 reviewer 共识).

---

## 10. Round 9 元教训

**Round 8 → Round 9 增量**: 

| 维度 | Round 8 发现 | Round 9 发现 |
|---|---|---|
| 主类型 | 结构性 hard blocker (CLI 杜撰 / path_guard / 文档结构假设) | 局部 quantitative threshold 未 anchor 代码 (B26 PASS_2 / B-V21-scope) |
| 偏差性质 | 写文档凭记忆, 没 verify code | **修文档时, 再次**凭记忆, 没 verify code (= 同类偏差复发) |
| 防御 | 加 B21-B25 standing rules | 升级 B21/B25 为更具体的 mechanical checklist |

**关键观察**: 6 个新偏差里, **B26 是 B21 同型, B27 是 B25 同型, B28 是 B12 擦边** — 全部是 prior round 已识别偏差的**新形态**. 说明 standing rules 写得抽象 ("verify code"), 但具体应用时仍漏 (没 verify log_interval / 没 verify §2.3 row 1 / 没 verify EV 表 multiplier 来源).

**真正修法**: 不是再加 standing rules, 而是给现有 rules 配 **mechanical checklist**, 例如:

```markdown
B21 mechanical checklist (CLI flag):
- [ ] grep "add_argument" <script> 列出所有 flag
- [ ] task md 中每个 CLI flag 在上述列表中
- [ ] task md 不假设任何不在列表中的 flag

B25 mechanical checklist (文档/yaml 字段):
- [ ] grep "^### " <doc> 列出所有 section heading  
- [ ] task md 引用的每个 heading 在上述列表中
- [ ] task md 引用的每个 yaml 字段在 yaml 实际存在 (用 python yaml.safe_load 验证)
- [ ] task md 假设的字段值 (rank=N, log_interval=N, ...) 在 yaml 实际值匹配

B26 mechanical checklist (NEW, quantitative threshold):
- [ ] 每个 "≥ N rows / files / calls" 阈值, list 它依赖的所有 yaml 字段
- [ ] N 必须 ≤ floor(max_steps_after_patch / log_interval_after_patch) - margin
- [ ] 阈值与 yaml 患在**同次** spot-check 中 verify
```

这是 **v3 之后** (push 给 codex 后) claude 写未来 task md 时应内化的 standing rule, **不**在 v3 task md 本身扩散 (避免 v3 膨胀).

---

## 11. 立即下一步

待 user 回复决策 1 → claude 起 v3 (~30 min 修订, ~47 行改动) → user 过一眼 → push gitee 给 codex 执行 Phase A.

**不起 Round 10**. 三 reviewer 共识修完 B26 + B-V21-scope 即可 push, 不出新 hard blocker.
