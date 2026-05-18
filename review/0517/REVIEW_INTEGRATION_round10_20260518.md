# Round 10 Review Integration — v3 + audit DRAFT 评审交叉核对

- date: 2026-05-18
- branch: foc_lite_hop0 (v3 + audit DRAFT 仍未 push)
- 主审对象:
  1. [CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md)
  2. [CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md](./CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md)
- 三位独立 reviewer: agent1 / agent2 / agent3
- 上游: [REVIEW_INTEGRATION_round9_20260517.md](./REVIEW_INTEGRATION_round9_20260517.md) (user 决策 C)

---

## 0. 一行结论

**3/3 reviewer 共识 v3 = MODIFY THEN PUSH**, audit DRAFT = MODIFY THEN USE. 

**关键发现 (3/3 一致)**: **B31 = §0.5 摘要表 vs §B0 实际 sanity range 不一致** ([15, 30] vs [10, 35]). 这是 **B27 在 v3 修 B27 时复发**, 印证 Round 9 元教训 "修偏差的 PR 自身造同型偏差". 必修.

**agent3 独家 B32**: `init_scale_zero=False` 路径 silent ignored ([decoder_lora.py:123-130](PET_LatentResidual/pet_lr/decoder_lora.py)). 影响 audit DRAFT Q3.2.1 问错方向, **不影响 v3 push**.

**好消息**: 不需要 Round 10b. v3 ≤15 min 修完, audit DRAFT ≤30 min 修完, 即可 push v3.

| 维度 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| Q1 7 处修订落实 | MODIFY (漏 Round 9 §5.3 git add/diff) | APPROVE (6/7 落地, B31) | MODIFY (B31) | **3/3 v3 真修了主体, 但有遗漏** |
| Q2 PASS_2 + log_interval | APPROVE | APPROVE | APPROVE | **3/3 ✅ 真修** |
| Q3 V21 白名单 grep | MODIFY (§0.5 vs §B0 不一致) | MODIFY (B31) | MODIFY (B31 range 不一致) | **3/3 共识 B31** 🔴 |
| Q4 Timeout END_TS | MODIFY (建议补 defensive, 不阻塞) | MODIFY (defensive) | APPROVE (dead path 无害) | **2/3 应加 defensive** |
| Q5 V18-r64 凭空? | 非凭空 (Round 9 line 158) | 非凭空 | 非凭空, 但 Round 9 user 签 C 时未拦截 | **3/3 ✅ 非 v3 bug, prompt 担心错了** |
| Q6 Attestation 16 项 | MODIFY (item 6 措辞) | APPROVE (item 6 非悖论) | MODIFY (item 6 措辞 + item 7/14 重叠) | **2/3 item 6 改措辞** |
| Q7 v3 新偏差 | B31a (§10 自查) + B31b (§0.5 vs §B0) | B31a + B31b | B31 + B32 (audit DRAFT 范围) | **3/3 B31 真; B31a (self-check 非 assert) 2/3** |
| Q8 audit M1-M8 范围 | (未独立评分) | APPROVE w/ M5 升 P0 | MODIFY w/ M4 降 P2 | **微调, 非阻塞** |
| Q9 audit 22 个 Q actionable | (未独立评分) | APPROVE | REJECT (Q3.1.3/Q3.2.3 引用不存在函数) — **agent3 verify 错** | **核实后纠正** |
| Q10 audit NOT-DO 滑坡 | (未独立评分) | MODIFY (短证据表) | MODIFY (加 #11 看不懂段标 FLAG + §10 改文件名后缀) | **2/3 应加 NOT-DO #11** |
| Overall v3 | MODIFY THEN PUSH | MODIFY THEN PUSH | MODIFY THEN PUSH | **3/3 MODIFY THEN PUSH** |
| Overall audit | MODIFY THEN USE | MODIFY THEN USE (M5 升 P0) | MODIFY THEN USE (符号清理) | **3/3 MODIFY THEN USE** |

---

## 1. 3/3 共识必修: B31 (§0.5 vs §B0 sanity range 不一致)

### 1.1 文档证据

| 位置 | 值 |
|---|---:|
| [v3 §0.5 行 6](PET_LatentResidual/review/0517/CODEX_TASK_PHASE_A_v3_20260518.md) "sanity 范围改" | **[15, 30]** |
| [v3 §B0 注释 + 实际脚本](PET_LatentResidual/review/0517/CODEX_TASK_PHASE_A_v3_20260518.md) `if [ "$TOTAL" -lt 10 ] \|\| [ "$TOTAL" -gt 35 ]` | **[10, 35]** |

codex 看到 §0.5 [15, 30] 但执行 [10, 35], 实际 21 hits 落入 [10, 35] (真实 ~21 落入 [15, 30] 边界) — 不会立刻 STOP, 但若 V21 retire 期间文档数变化 (e.g. 阶段 B 加新 V21 references), [15, 30] 早 trip 但 [10, 35] 仍 pass, 行为不可预期.

### 1.2 元意义 (Round 9 元教训印证)

Round 9 B27 = "v2 §A1 stale fact 与 §2.3 row 1 不一致, 文档自相矛盾". 
B31 = "v3 修订摘要表 §0.5 与正文代码 §B0 数字不一致, 文档自相矛盾". 

**B31 严格是 B27 在 v3 修 B27 自身时的镜像复发**. 100% 印证 Round 9 元教训.

### 1.3 修法

二选一:
- 选项 A: §0.5 行 6 改 "[10, 35]" 与 §B0 一致
- 选项 B: §B0 实际脚本改 [15, 30] 与摘要一致

claude 推荐 **A** (与 Round 8 实测 21 + ±50% margin 论据相符, 改动小).

---

## 2. agent1 独家漏修: Round 9 §5.3 git add 通配符 + diff 估计

### 2.1 漏修证据

Round 9 整合 §5.3 明文列为 should-fix:
> "v2 `git add review/0516/.../smoke_runs/V13_smoke_*.yaml` 通配可能匹配旧 smoke 残留. 改 `git add "${SMOKE_YAML}"` 单文件"
> "v2 §E1 §8 '(~+60 行)' 实际是 ~+110 行. 改 '(~+80 - 130 行)' 范围"

v3 §0.5 表声称 "7 处修订", 但 git add 通配 + diff 估计**未列**. v3 §E2 沿用 v2, 仍是通配 + ~+60 行估计.

### 2.2 user 决策 C 的覆盖范围

user Round 9 决策 "C 修 2 hard + 5 should-fix" = 选项 C 含全部 5 should-fix. **git add 通配是其中第 4 个 should-fix**.

→ v3 漏修了 user 已签的 should-fix. **必修**.

### 2.3 修法

```bash
# v3 §E2 改:
git add "${SMOKE_YAML}"   # 不是 review/0516/.../V13_smoke_*.yaml
git add "${SMOKE_LOG}"    # 不是 review/0516/.../V13_smoke_*.log

# v3 §E1 §8 改:
# 预期改动文件:
# - V18_design_rationale.md (~+80 - 130 行)   ← 不是 ~+60
```

---

## 3. Q6 item 6 措辞 (2/3 建议)

### 3.1 当前措辞悖论

v3 §E1 §6 attestation 第 6 项:
> "未起 Round 10 review"

但**本份 prompt 就是 Round 10**. 字面上 codex 勾 [x] 不矛盾 (review 由 user 起, 非 codex 起), 但语义上奇怪, 容易让 reviewer 误判 v3 的 anchor 时机.

### 3.2 修法

agent3 推荐: "未自发地另起 Round 11+ review / 未对其他 task md 做 audit"

更精简: "codex 在 Phase A 执行期间未自发地起任何 review / audit / V18-clean 准备".

---

## 4. agent3 关于 audit DRAFT 符号不存在的纠正

agent3 称 `_assert_v18_step0_equivalence` 和 `assert_decoder_frozen` "全 repo 零命中". claude **二次 verify** 显示:

```
PET_LatentResidual/train_first_hop.py:130  def _assert_v18_step0_equivalence(...)
PET_LatentResidual/train_first_hop.py:1895     _assert_v18_step0_equivalence(...)
PET_LatentResidual/train_first_hop.py:1453     model.assert_decoder_frozen()
```

→ **两个函数都存在**. agent3 grep scope 漏了 (可能 .gitignore 排除 train_first_hop.py). 

**含义**: audit DRAFT Q3.1.3 / Q3.2.3 引用是**正确**的, 不需要删除. agent3 的"符号不存在"判断错了, 但 **B32 (init_scale_zero False silent ignored, decoder_lora.py:123-130)** 经 claude verify **真实**.

→ audit DRAFT 实际只需修 1 处 (Q3.2.1 方向反过来), 不是 agent3 列的 3 处.

---

## 5. agent3 独家 B32 (audit DRAFT 影响, 非 v3)

### 5.1 代码证据 ([decoder_lora.py:95, 123-130](PET_LatentResidual/pet_lr/decoder_lora.py))

```python
init_zero = bool(lora_cfg.get("init_scale_zero", True))
...
if not init_zero:
    # We don't support that here; warn loudly.
    print(f"[decoder_lora] WARN: init_scale_zero=false ignored. ...")
```

yaml 字段 `init_scale_zero` 在 False 路径**被 silently 忽略**, 实际行为始终 = True (LinearWithLoRA B 矩阵硬编码 0 init).

### 5.2 影响

- **不影响 v3** (v3 不涉及 yaml `init_scale_zero` 字段, V13/V18 都用 default True)
- **影响 audit DRAFT Q3.2.1**: 当前问 "True 路径是否触发?" — 但 silent bug 在 False 路径 (用户期望 False 不 zero-init, 实际仍 zero-init). 问错方向

### 5.3 修法 (audit DRAFT)

Q3.2.1 改为:
> "`init_scale_zero=False` 路径是否真的 silently 忽略? yaml 字段是否应改为 raise 而非 print WARN? 这影响阶段 C V18-clean 是否能用 yaml 调试 LoRA init"

---

## 6. Q4 Timeout END_TS (2/3 建议加 defensive)

### 6.1 现状

agent1+agent3 一致: v3 timeout 分支 `exit 1` 在 `END_TS` 设置**之前**, dead path 不会执行 PASS_5, 无 silent bug. agent2 担心 "codex 按 markdown block 分块执行" 导致 START_TS 跨 shell 丢失 — 这是 codex 执行模型差异, 不是 v3 内在 bug.

### 6.2 是否要加 defensive

- **不阻塞 push**
- 加 defensive 1 行 (timeout 分支 `END_TS=$(date +%s); echo "timeout at $END_TS"` 后再 exit 1) 增加可审计性, ROI 高
- claude 推荐 **加** (不增加复杂度, 防御 codex 多 shell 执行)

---

## 7. v3-specific 新偏差 (B31a / B31b → 整合为 B31)

agent2 提两个 sub-偏差:
- **B31a**: §10 自查表 13 项 ✓ 全 claude 自勾, 无 grep-as-test 强制 (B30 同型复发)
- **B31b**: §0.5 vs §B0 sanity range 不一致 (本整合主 B31)

agent3 + agent1 + agent2 三人一致认 B31b 是真实必修. B31a (自查表非 assert) 是 should-fix:
- agent2 建议改成 13 条 grep/test 命令
- 但这把 task md 从 "instructions" 变成 "shell script", 显著膨胀
- claude 推荐 **不全改 grep**, 只在 §10 末尾加一句: "user 验收时**不**信 claude 自勾, 必须独立 grep verify §0.5 与 §B0 数字一致"

---

## 8. audit DRAFT 修订汇总

| # | reviewer | 修法 | 必修? |
|---|---|---|---|
| D1 | agent3 (经 claude 纠正) | Q3.2.1 方向反过来 (False 路径才是 bug) | YES (B32 真实) |
| D2 | agent3 | §5 加 NOT-DO #11 "看不懂段必须标 UNCERTAIN, 不许跳过" | YES |
| D3 | agent3 | §10 改 trigger 机制: 文件名加 `_PENDING_USER_RELEASE` 后缀 + 首行 `DO NOT EXECUTE — USER MUST RENAME FIRST` | should |
| D4 | agent2 | M5 (eval_first_hop_224_clip3.py) 升 P0 (阶段 B V18 200K eval 直接依赖) | should |
| D5 | agent3 | M4 (path_guard.py 30 行) 降 P2 | minor |
| D6 | agent2 | §3 22 个 Q 多数有"已知 Round X agent Y 答过"标注, 限制产出 ~3-8 行/题 | should |

agent3 提的"Q3.1.3 / Q3.2.3 删假符号"**不采纳** — claude verify 函数都存在.

---

## 9. 修订后 v3 push 前最终清单

**必修 (push 前)**:
1. **B31**: §0.5 行 6 "[15, 30]" → "[10, 35]" (与 §B0 一致)
2. **agent1 漏修**: §E2 `git add` 通配 → `${SMOKE_YAML}` / `${SMOKE_LOG}` 显式; §E1 §8 diff 估计 "(~+60 行)" → "(~+80 - 130 行)"
3. **Q6 item 6**: 措辞改 "codex 在 Phase A 执行期间未自发起任何 review / audit / V18-clean 准备"

**应修 (push 前)**:
4. **Q4 defensive**: §C3 timeout 分支 `exit 1` 前加 `END_TS=$(date +%s); echo "timeout END_TS=$END_TS, elapsed=$((END_TS - START_TS))s"`
5. **B31a**: §10 末尾加 "user 验收时不信 claude 自勾, 必须独立 grep verify §0.5 与 §B0 数字一致"

**audit DRAFT 修订 (本地, 不 push)**:
6. **D1 (B32)**: Q3.2.1 方向反过来
7. **D2**: §5 NOT-DO 加 #11
8. **D3**: 改 trigger 机制 + 文件名后缀
9. **D4**: M5 升 P0
10. **D5**: M4 降 P2
11. **D6**: §3 限制单题产出 3-8 行

总工作量 ~30-40 min (v3 ~15 min, audit DRAFT ~20 min).

---

## 10. user 决策点

只有 1 个真正需要 user 拍板:

| # | 决策 | 选项 | claude 推荐 |
|---|---|---|---|
| 1 | v3 修订范围 | A: 仅必修 1-3 <br>B: 必修 1-3 + 应修 4-5 <br>C: 全部 1-11 (v3 + audit DRAFT) | **C** (全部 ~35 min, ROI 高) |

- **B 是绝对底线**: 不修 B31 会被未来 reviewer catch 同型 bug, 不修 agent1 漏修 should-fix 是违反 user 决策 C 的字面.
- **C 是 claude 推荐**: audit DRAFT 同步修, 阶段 A 完成时直接 rename + push 用, 不需要再修
- **A 不推荐**: Q4 defensive 0 风险 1 行, B31a 0 风险 1 行, 不修留隐患

回复: `1=C` 或 `按推荐` 或其他.

---

## 11. Round 10 元教训

**B31 印证 Round 9 元教训**: "修偏差的 PR 自身造同型偏差" (B27 在 v3 修 B27 时复发为 B31). 100% 命中.

**Standing rule 升级建议** (加进 V18_design_rationale §5.2 作为 B26 mechanical checklist 扩展):

> **B31 mechanical checklist** (新):  
> 任何修偏差的 PR 必须在 attestation 中 cross-check **修订摘要表与正文代码每项数字一致**.
> 具体做法: 写完 task md 后, 对每个数字 (阈值/范围/行数估计/字段值), 强制做一次 grep 搜全文确认所有出现处一致.

这条加进 §5.2 后, 未来 v4/v5 起草时, claude 必须在自查 (§10) 用 grep 命令而非自勾.

**Round 1-10 总览**:

| 轮次 | 范围 | 修订对象 | 主要发现 |
|---|---|---|---|
| 1-5 | design | V18 设计 | 11.2 dB 当 attackable (B10) |
| 6 | execution prep | V18b draft | LoRA→LoRA resume blocker (agent2) |
| 7 | architecture | NEXT_STAGE | 4 个代码事实 (KL warmup global step / V18-clean 40K / PSNR_clip3 路径 / resume_from 死字段) |
| 8 | execution v1 | Phase A v1 | 5 hard blocker (CLI 杜撰 / path_guard / pass 准则 / 文档结构假设 / commit 范围) |
| 9 | execution v2 | Phase A v2 | 2 hard (B26 log_interval / B-V21-scope) + 5 should-fix |
| **10** | **execution v3 + audit DRAFT** | **Phase A v3 + audit** | **B31 (修订摘要 vs 正文不一致, B27 复发) + B32 (init_scale_zero silent ignored)** |

每轮都识别 prior round 偏差的新形态 — 共识 Round 10 也不是终点, 但 v3 + audit DRAFT 修完后, 阶段 A push gitee, 后续 Round 11+ 应**只**在 codex 执行结果 (PHASE_A_EXECUTION_REPORT) 出来后由 user 决策是否起.

---

## 12. 立即下一步

待 user 回复决策 1 → claude 起 v4 (~10-15 min 修 v3 必修+应修) + 直接编辑 audit DRAFT (~20 min 修 D1-D6) → user 过一眼 → push v3 gitee, audit DRAFT 留本地等阶段 A 完成.

**不起 Round 11**. user 决策 C 后, 三 reviewer 共识修完即可 push.
