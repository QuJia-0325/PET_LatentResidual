# Round 8 Review Integration — CODEX_TASK_PHASE_A 评审交叉核对

- date: 2026-05-17 深夜
- branch: foc_lite_hop0（task md 仍未 push）
- 主审对象: [CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md)
- 三位独立 reviewer: agent1 / agent2 / agent3
- 硬约束: 内存 max 3 任务; V18 (slot 1) 不可动

---

## 0. 一行结论

**3/3 reviewer REJECT 当前 task md (verdict: MODIFY/MODIFY/MODIFY-then-PUSH, agent2 BLOCK)**. 一致定位 **3 个 hard blocker** (CLI flag 不存在 / path_guard 拒 repo 内 output_dir / pass 准则 grep 模式与代码 log 不匹配) + **3 处文档假设错** + **4 个新偏差 B21-B24**.

**好消息**: 不需要 Round 9. 修完 3 个 hard blocker + 4 个 MODIFY 项后可直接 push 给 codex.

| 维度 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| 整体 verdict | MODIFY THEN PUSH | **BLOCK** | MODIFY THEN PUSH | 不可原样 push ⚠️ |
| Q1 CLI flag (`--max-steps-override`) | REJECT (CRITICAL) | REJECT | REJECT | **3/3 hard blocker** 🔴 |
| Q2 grep pattern | MODIFY (HIGH) | REJECT (准则 1-6 多处错) | MODIFY (6/6 准则 5 个有问题) | **3/3 不可执行** 🔴 |
| Q3 V21 grep 策略 | MODIFY (加 sanity 范围 6-15) | MODIFY (先 grep preflight) | MODIFY (加白名单) | **3/3 MODIFY** |
| Q4 NOT-DO 列表 | APPROVE 现状 | MODIFY (嵌入 per-task) | MODIFY (加 4 条新 NOT-DO) | **分歧** |
| Q5 push 时机 | APPROVE (5 min OK) | MODIFY (措辞降级) | **REJECT** (拆 2 个 push, V13 等 1h soak) | **分歧** |
| Q6 commit 范围 | MODIFY (HIGH, 不 commit pid+rolling log) | MODIFY (同 agent1) | MODIFY (加 git diff --stat 强制) | **3/3 必修** |
| Q7 自警范围 | MODIFY (补 B12/B19/B20) | MODIFY (短表) | MODIFY (12-项勾选式 + 补 B12/B13/B19/B20) | **3/3 必修** |

---

## 1. 3 个 hard blocker (必修, 不修 codex 一启动即 fail)

### 1.1 Q1 — CLI flag 杜撰 (B21)

**代码事实** (agent1+2+3 独立 verify, `train_first_hop.py:1355-1357`):
```python
parser.add_argument("--config", required=True)
parser.add_argument("--resume", default="")
# 仅此 2 个 flag
```

`--max-steps-override` 和 `--output-dir-override` 都不存在 → argparse `unrecognized arguments` → 立即 SystemExit(2).

**当前 §3.2 的实际行为**:
1. codex 读 PRIMARY 命令 (含不存在 flag) → 秒退
2. fallback 注释 ("若不支持改 cp+sed") 让 LLM 倾向 retry + escalate, 违反 §6 NOT-DO

**修复**: 删除 CLI override 整个路径, cp+sed/python yaml 改成 **唯一** PRIMARY 路径.

### 1.2 path_guard 拒 repo 内 output_dir (B22-prime, agent2 独家发现)

**代码事实** ([path_guard.py](PET_LatentResidual/pet_lr/path_guard.py)):
```python
DATA_DISK_ROOT = Path("/data_2").resolve()
def resolve_data_disk_dir(path_like, *, arg_name):
    ...
    if DATA_DISK_ROOT != resolved and DATA_DISK_ROOT not in resolved.parents:
        raise RuntimeError(f"{arg_name} must be under {DATA_DISK_ROOT}...")
```

V18/V13/任何训练 yaml 的 `output_dir` **必须在 `/data_2/...` 下**.

**当前 §3.2 写**: `SMOKE_DIR=review/0516/V13_true_image_aux_ablation/smoke_runs/...` — 是 **repo 内** 路径 → path_guard 立即 raise RuntimeError.

即便 agent1 给的 fallback (cp yaml + 改 output_dir → SMOKE_DIR) 也会被 path_guard 拦. **这是 §1.1 修复方案里 agent1 也没意识到的**.

**修复**: smoke 的 output_dir 必须改到 `/data_2/qujiaxiang/outputs/PET_LatentResidual/smoke_runs/v13_smoke_<TS>`. repo 内**只**放 smoke yaml 副本 + smoke.log (从 nohup 重定向).

### 1.3 Q2 — pass 准则 grep 模式与代码不匹配 (B23)

**多处问题** (3/3 reviewer 不同角度都命中):

| 准则 | 问题 | 来源 |
|---|---|---|
| **准则 2** (`image_aux.enabled = false` grep stdout) | 代码不显式 print 这条 effective config; agent1+2+3 都说 grep pattern 与代码 log 实际格式不匹配, 高概率 false-fail | agent1/2/3 一致 |
| **准则 3** (`grep -i "nan\|inf"`) | **agent3 重大发现**: 代码在 `image_aux.enabled=false` 时把 `ssim_raw_min/max/mean` 设为 nan tensor ([train_first_hop.py:2166-2178](PET_LatentResidual/train_first_hop.py)), log 大概率含 `nan`, 当前 grep 过滤 (`grep -v manifold -v annotated`) 抓不掉 → 准则 false-fail | agent2/3 |
| **准则 4** (grep stdout `loss_img`) | 代码 `image_aux.enabled=false` 时 `img_losses` dict 在 [train_first_hop.py:2172](PET_LatentResidual/train_first_hop.py) 路径; metrics.jsonl 字段是 `img` 不是 `loss_img`; stdout 是否 print 未验证 | agent2 |
| **准则 5/6** (throughput/memory ≤ V7 × 1.1) | V7 baseline 数字**完全没在 task md 给出** → codex 无数据点比较 → 准则不可评估 | agent1/3 一致 (B24) |

**修复**: 
- 删准则 2 (与准则 4 合并, 用 jsonl 直接验证)
- 准则 3 改为只在 `[nonfinite]` 异常分支 print 上找 NaN, 或检查特定字段 (total_loss/grad_norm), 排除 ssim_raw_*
- 准则 4 改为读 metrics.jsonl 而非 grep stdout (用 python 解析)
- 准则 5/6 删除, 改为绝对值 ("≤ 15 min" 替代 "≤ V7 × 1.1")

---

## 2. 3 处文档假设错 (高严重度但非 hard blocker)

### 2.1 B21 — V18_design_rationale.md 现有结构与 task md 假设不符 (agent2 独家)

**任务 §A1 假设**: §2.3 是 "阈值表" → 加 §2.4 在其后.

**实际结构** ([V18_design_rationale.md](PET_LatentResidual/review/0517/V18_decoder_lora/V18_design_rationale.md), claude 已 verify):
- §2.3 标题 = "V18 失败的 4 个具体可能 + 应对" (是 failure mode 表, **不是 PRIMARY/PARTIAL/KILL 阈值表**)
- §5 红线 = 阈值相关讨论 ("预注册阈值不可事后改" 在 §5.2)
- **没有任何一节叫 "阈值表"**, 阈值实际散在 §2.3 第 4 行 trigger 列里

**含义**:
- 任务 §A1 "在 §2.3 阈值表之后, §3 之前" → codex 找不到锚点
- 任务 §A2 假设 "§5.2 已有 standing rules" → 实际 §5 是 4 条红线 (1.可被 sanity-check / 2.阈值不可事后改 / 3.不再发 peer review / 4.Day 9 binary), 不是 standing rules 段

**额外风险**: V18_design_rationale.md 仍含 **stale facts**:
- §2.1 写 "LoRA rank=8, alpha=16" → 实际 V18 用 rank=32 ([V18_decoder_lora.yaml](PET_LatentResidual/review/0517/V18_decoder_lora/V18_decoder_lora.yaml))
- §2.1 写 "λ_kl=0.5" → 实际 yaml 是 0.05
- §2.2 EV 表写 "0.5-2.0 dB" → Round 5 已识别 B10 错算
- §2.3 trigger 列写 "ΔPSNR < +0.05 + KL drift > 0.05" → Round 7 改成区间 [0.05, 3] dB

**修复** (重大改写 §A1/§A2):
- 任务 §A1 不能简单 "新增 §2.4", 必须先**修正 §2.1 stale facts** (rank=8 → 32, λ_kl=0.5 → 0.05) 再加 §2.4
- 任务 §A2 不能假设 "§5.2 已有 standing rules", 必须在 §5 红线 #2 之后**新增 §5.2 "Round 5-7 偏差自警 (B12-B20)"** 段

### 2.2 require_fresh_output_dir 联动 (B22)

agent1+3 独立指出: V13 yaml [require_fresh_output_dir: true](PET_LatentResidual/review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml). 若 smoke 复用同一 SMOKE_DIR retry, 第二次 raise; 若 SMOKE_DIR 在 mkdir 后已有 V13_smoke.yaml, 也可能被 fresh check raise.

**修复**: smoke yaml 副本里强制 `training.require_fresh_output_dir: false`, 或 SMOKE_DIR 用每次新建 timestamp 子目录 + yaml 副本放在父目录.

### 2.3 V21 grep 范围 (B-V21-grep)

agent2 verify: V18_EXECUTION_REPORT_20260517.md 当前 grep `V21` **0 命中**. 任务 §B1 列 4 文档全部 commit 标记 → V18_EXECUTION_REPORT 会被加一个**假**的 retire warning.

实际命中:
- REVIEW_INTEGRATION_round4: 19 处
- GAP_DECOMP_REPORT: 1 处
- CODEX_RUNBOOK_V18: 1 处
- V18_EXECUTION_REPORT: **0 处**

**修复**: §B 改为 "先 preflight grep, 只对有命中文档加标记, 命中 = 0 的文档跳过, 在执行报告里说明".

---

## 3. 3 reviewer 分歧 (需 user 决策)

### 3.1 Q5 push 时机

| reviewer | 推荐 | 论据 |
|---|---|---|
| agent1 | APPROVE 5 min push, 措辞 "alive at +5min" | 拖 4h 没收益, Phase A 主要产物是文档不是 V13 健康度 |
| agent2 | MODIFY (5 min push OK 但措辞降级) | 同 agent1, 加要求报告里不写 "健康" 这种 7-day 推断 |
| agent3 | **REJECT, 拆 2 个 push** | 5 min 是 PyTorch warmup 期, push 后若 V13 死掉 → gitee 永久 misinformation; 第 2 个 push 应在 V13 1h soak 后 |

**核心冲突**: gitee push 是 immutable 操作 vs Phase A 应该 1 次完成.

**claude 推荐**: **折衷方案** — 5 min 后 push 但 task md 强制 **执行报告措辞降级**:
- 报告里 V13 状态写 "Process alive at +5 min after launch; GPU utilized. **未经长期 soak 验证**, 7 天训练健康度由阶段 B/C 跟踪"
- 不允许出现 "V13 健康" / "V13 稳定" / "V13 收敛正常" 字样
- push 后若 V13 在 30 min 内死掉, codex 单独 commit 一份 `V13_LAUNCH_FAILURE_<TS>.md`, 不修改 Phase A 报告

这等价于 agent1+agent2 折衷, 不采纳 agent3 的拆 2 push (避免 task md 自身复杂度上升).

### 3.2 Q4 NOT-DO 嵌入方式

- agent1: APPROVE 现状 (§6 集中 + §7 决策树足够)
- agent2: MODIFY (per-task 末尾嵌入)
- agent3: MODIFY (加 4 条新 NOT-DO)

**claude 推荐**: **采纳 agent3 + 部分 agent2**:
- 加 4 条新 NOT-DO (不改 V13 原 yaml / 不 commit V13 原 yaml / CLI 不存在时不改 train_first_hop / smoke fail 不 rm -rf 重跑)
- 每个 task 末尾**重复一次** "fail → 停下报告, 不 escalate" (agent2 提议; 1 行不膨胀, 抗 LLM 滑坡有效)
- 不全 12 条嵌入 (太 verbose)

### 3.3 Q7 自警范围

| reviewer | 推荐 |
|---|---|
| agent1 | 补 B12/B19/B20 (与 task A 文档操作直接相关) |
| agent2 | 短表 (5 项 yes/no) |
| agent3 | **12-项 NOT-DO 勾选式 attestation** + 6 项 bias 自查 |

**claude 推荐**: **agent3 12-项勾选式**. 理由: codex 倾向模糊回答 ("基本没"); 强制勾选可被 user 一眼审查; 12 项不算多 (NOT-DO 本来就 12 条).

---

## 4. 新偏差 (Round 8 识别)

| # | 名称 | 来源 | 形态 |
|---|---|---|---|
| **B21** | CLI flag 杜撰 | agent1+2+3 一致 | LLM 默认假设 "显而易见的工具应该存在" (--max-steps-override / --output-dir-override); 设计文档语义污染执行文档 |
| **B22** | yaml override + require_fresh_output_dir + path_guard 三重联动未查 (B14 复发) | agent1+2+3 不同侧重 | smoke yaml 副本 + repo 内 output_dir + fresh check 三处可独立 raise, 但 §3.2 全部假设 OK |
| **B23** | pre-mature push (premature claim) | agent3 独家 | 把 "launch 成功" 等同 "训练健康"; push 后 gitee 留永久 misinformation |
| **B24** | 引用 baseline 数字但未提供数字 | agent1+3 | "≤ V7 baseline × 1.1" 但 V7 baseline 数字头脑里有, task md 没写 → codex 不可评估准则 |
| **B25 (合并自 agent2 B21)** | 假设 stale 文档结构与现状一致 | agent2 | task md §A1 假设 V18_design_rationale 有 "§2.3 阈值表" / "§5.2 standing rules", 实际两个 section 都不存在 / 含义不同 |

B22 最严重 (3 重独立 raise 点叠加), B25 影响最广 (Task A 整个改写后才能执行).

---

## 5. 修订后的执行清单 (claude 推荐)

### 5.1 Hard fix (必修, 不修不可 push)

1. **§3.2 重写 V13 smoke** (B21 + B22 修复):
   - 删除所有 CLI override 路径
   - smoke yaml 在 repo 内副本: `review/.../smoke_runs/v13_smoke_<TS>.yaml`
   - smoke output_dir 在 `/data_2/qujiaxiang/outputs/PET_LatentResidual/smoke_runs/v13_smoke_<TS>/run`
   - 强制改 smoke yaml 4 字段: `max_steps=200`, `output_dir=...`, `run_name=...`, `require_fresh_output_dir=false`
   - 加 sanity check: sed 后 `grep "max_steps: 200" $SMOKE_YAML` 必须成功
   - 加 md5 守卫: 原 V13 yaml 操作前后 md5 必须相同

2. **§3.5 重写 pass 准则** (Q2 修复):
   - 准则 2 删除 (与准则 4 合并)
   - 准则 3 NaN 检查只针对 `total_loss/grad_norm/loss_pair/loss_roll` 字段, 排除 `ssim_raw_*`
   - 准则 4 改 python 读 metrics.jsonl 直接验证 `loss_img` 或 `img` 字段全为 0
   - 准则 5/6 删除 (或给具体 V7 baseline 数字, 推荐删除)
   - 新增准则: metrics.jsonl 文件存在且非空 (验证训练真的写了数据)

3. **§1 Task A 重写** (B25 修复):
   - §A1 改为: (a) 先修正 V18_design_rationale.md §2.1 stale facts (rank=8 → 32, λ_kl=0.5 → 0.05), (b) 在 §2.3 之后插入 §2.4 EV 修正记录
   - §A2 改为: 在 §5 红线 #2 之后新增 §5.2 "Round 5-7 standing rules (B12-B20)" 段
   - 给 codex sed/python 模板, 不只 "把这段加进去"

4. **§5.2 commit 范围收紧** (Q6 修复, B23 防御):
   - 不 commit `*.pid`
   - 不 commit `V13_train_*.log` (rolling)
   - 加 `*.pid` 到 .gitignore (新建 patch)
   - 强制要求执行报告含 `git diff --stat` 输出
   - 报告里 V13 状态措辞降级 (Process alive at +5min, 未经 soak)

### 5.2 Should fix (强烈推荐, 但 user 可选不修)

5. **§2 V21 retire 改 preflight + 白名单** (Q3): 先 grep 全部命中, 报告里 list, 只对 4 文档白名单 + 有命中的加标记; V18_EXECUTION_REPORT 命中 = 0 → 跳过.

6. **§6 加 4 条新 NOT-DO** (Q4 agent3):
   - 不改 V13 原始 yaml
   - 不 commit V13 原始 yaml
   - CLI flag 不存在时不改 train_first_hop.py
   - smoke fail 不 rm -rf 重跑

7. **§5.1 执行报告改 12-项勾选式 attestation** (Q7 agent3):
   - 12 项 NOT-DO 全勾选 (`[x]`)
   - 加 6 项 bias 自查 (B12/B13/B14/B18/B19/B20 每条 yes/no + 1 行说明)

### 5.3 不修 (claude 评估)

8. Q5 push 时机: 采纳折衷 (5 min push + 措辞降级), 不拆 2 个 push (agent3 提议).
9. Q4 NOT-DO 完整嵌入 per-task: agent1 现状已够, 仅每个 task 末尾加 1 行 "fail → 停" 即可.

---

## 6. user 决策点

只有 1 个真正分歧点需要 user 拍板:

| # | 决策 | claude 推荐 |
|---|---|---|
| 1 | Q5 push 时机: 5 min 措辞降级 / 拆 2 push (5 min + 1h) | **5 min + 措辞降级** (折衷, 不增复杂度) |

其他全部按 claude 推荐 (§5.1 必修 4 条 + §5.2 推荐 3 条) 执行, 不需 user 单独确认.

回复格式: `1=折衷` 或 `1=拆2push` 或 `按推荐`.

---

## 7. 修订后流程

收到 user 回复 → claude 重写 [CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md) (~30 min, 改动 §1/§3/§5/§6 4 大段). 重写后 **不再起 Round 9** (3/3 reviewer 共识不需要), user 过一眼 → push gitee 给 codex 执行.

---

## 8. Round 8 元教训

**Round 8 与 Round 1-7 根本不同**: 前 7 轮审 design (做什么 / 怎么决策), Round 8 首次审 execution (codex 能不能真跑). 揭示了一类全新偏差:

- **设计层正确** (decision tree, threshold, ablation logic) ≠ **执行层正确** (CLI 实际存在, log 格式实际匹配, path 实际允许)
- LLM agent (claude) 写 task md 时倾向用 **"理想 CLI / 理想日志 / 理想路径"** 描述, 而非 **"实际 CLI / 实际日志 / 实际路径"**
- 防御: task md 在 push 前必须做 "脱稿执行核查" — 每个 shell 命令的每个 flag/path/字段在源码 grep 一次

**B21+B22+B23+B24+B25** 共同特征 = "claude 凭记忆写文档, 没去 verify 实际代码/文档当前状态". 全部 5 个偏差**都被 reviewer 通过读代码 catch**, 没有一个是概念错.

**含义**: 未来所有 codex task md 起草时, claude 必须执行 spot-check protocol:
- 每个 CLI flag → `grep add_argument`
- 每个 yaml override → 验证字段名拼写 + 类型 + 联动字段
- 每个 grep pattern → 找一段真实历史 log 验证模式匹配
- 每个引用文档 section → 读目标文档现状, 不依赖记忆中的结构
- 每个 path → cross-check 是否有 path_guard / fresh check / 类似限制
