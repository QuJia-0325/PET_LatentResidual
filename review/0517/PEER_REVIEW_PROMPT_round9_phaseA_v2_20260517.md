# Peer Review Round 9 — Phase A v2 codex task draft 审稿

- date: 2026-05-18 凌晨
- branch: foc_lite_hop0 (草稿, 未 push)
- 主审对象: **[CODEX_TASK_PHASE_A_v2_20260517.md](./CODEX_TASK_PHASE_A_v2_20260517.md)** (v2, Round 8 hard-fixed)
- 上游: Round 8 整合 [REVIEW_INTEGRATION_round8_20260517.md](./REVIEW_INTEGRATION_round8_20260517.md) + user 决策 Option B (5 min push + 措辞强制降级)
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**
- 评审目标: 验证 v2 是否真的修复 Round 8 全部 5 个 hard blocker, 并搜索 v2 引入的**新** bug

---

## 0. 本轮范围与边界

Round 8 三 reviewer (agent1/2/3) 共识 BLOCK v1, 列出 5 个 hard fix + 3 个 should fix. v2 (本轮主审对象) 是 claude 按 Round 8 整合**全部修订**后的产物. **本轮唯一任务 = 验证 v2 fix 真的修了, 且没引入新问题**.

**本轮只审**:
- v2 的 5 个 Round 8 hard fix 是否真修复 (Q1/Q2/B22/B25/Q6)
- v2 的 3 个 Round 8 should fix 是否落实 (Q3/Q4/Q7)
- v2 是否引入**新** code/文档/执行 bug (起草过程中 claude 是否复发 B21-B25 类偏差)
- user 决策 Option B (5 min push + 措辞降级) 是否在 v2 §5 E1 模板里被忠实落实

**本轮不审**:
- V18/V13 设计 (Round 1-5 已穷尽)
- 三阶段路线 / 3×3 决策矩阵 / V18-clean (Round 7 user 已签字)
- v1 任何内容 (已 supersede)
- Q5 push 时机 (user 已决策 Option B)

---

## 1. 给 reviewer 的 9 个问题

### Q1 — B21 (CLI flag 杜撰) 真修了吗?

v2 §3 完全删除 `--max-steps-override` / `--output-dir-override`, 改用 `cp + python yaml` 改 4 字段后只用 `--config` launch.

请 verify:
- v2 §3 是否还残留任何**不存在**的 CLI flag 引用?
- §C0 spot-check 段 `SUPPORTED_FLAGS` 的 grep 命令是否正确? grep `add_argument` 在 train_first_hop.py 实际输出是什么? 与 v2 期望的 "--config 和 --resume" 是否一致?
- §3 是否存在新 CLI flag 杜撰 (例如 codex 在 V13 full launch §4 D1 是否又凭记忆加了 flag)?
- v2 §4 D1 完整命令 `nohup python train_first_hop.py --config <yaml>` 是否真符合 V13 yaml 的 launch 路径 (e.g. V13 yaml 是否需要其他 env vars, V13 yaml 是否含 `decoder_lora` 之类 V13 不需要的 section 但训练器又强制要求)?

### Q2 — B22 (path_guard + require_fresh_output_dir) 真修了吗?

v2 §3 C1:
- smoke output_dir = `/data_2/qujiaxiang/outputs/PET_LatentResidual/smoke_runs/v13_smoke_${TS}`
- smoke yaml 强制改 `require_fresh_output_dir: false`

请 verify:
- v2 写的 `/data_2/qujiaxiang/outputs/PET_LatentResidual/smoke_runs/...` 路径是否真在 path_guard 允许范围? `path_guard.py` 是否对 `smoke_runs/` 子目录有额外限制?
- v2 strong-assume `cfg["output_dir"]` 是 yaml 顶层字段. **请直接读 V13 yaml**, 验证 `output_dir` 是否真在顶层 (而非 `cfg["training"]["output_dir"]` 或其他嵌套位置)
- 同理验证 `cfg["training"]["max_steps"]` / `cfg["training"]["require_fresh_output_dir"]` / `cfg["run_name"]` 的实际嵌套位置
- 若 v2 python 脚本路径与实际 yaml 结构不符, codex 会写出 yaml 字段错位的 smoke yaml → smoke 启动 raise 或 silently 用默认值

### Q3 — Q2 (pass 准则) 真修了吗?

v2 §3 C4 5 条新准则全部读 metrics.jsonl 而非 grep stdout. 字段名用 `img` (而非 `loss_img`).

请 verify:
- v2 准则 3 (`all train rows have img=0.0`) 是否真覆盖 image_aux disabled 检测? **请验证** train_first_hop.py 在 image_aux disabled 时, `metrics_payload` 是否真的总是写 `img=0.0`, 或者可能 skip 写整个 train event row
- v2 准则 2 (`>=50 train rows`) 在 200 step 内是否一定成立? metrics.jsonl 的 train row 写入频率是 every step / every N steps?
- v2 准则 4 critical fields = `["loss", "pair", "roll"]`. 是否漏了应该 check 的字段 (e.g. `vel`, `end`, `lambda_roll`)?
- v2 准则 5 用 `os.path.getmtime - getctime` 估 elapsed. macOS 和 Linux 上 `getctime` 含义不同 (Linux = inode 修改时间不是 creation time). 在服务器 (Linux) 上这个估计是否准确? 替代方案?
- 如果 V13 yaml `metrics_jsonl: false`, metrics.jsonl 根本不会写, 所有准则会 false-fail. 是否需要 v2 在 C0 spot-check 加 yaml `metrics_jsonl` 字段确认?

### Q4 — B25 (V18_design_rationale 文档结构) 真修了吗?

v2 §1 Task A 加了 A0 spot-check + A1 修 stale facts + A3 §5 重组. 

请 verify:
- v2 A0 spot-check 的 grep pattern `"^### 2\.1\|^### 2\.2\|..."` 是否真匹配文档当前 section heading? **请直接读** V18_design_rationale.md, 确认 section heading 用 `###` 还是 `##` 或其他级别
- v2 A1 修 §2.1 表格行的措辞 (e.g. "Round 4 audit 后从 rank=8 调整") — Round 4 是否真有 audit 过 rank? 还是 Round 4 audit 是别的 (e.g. LoRA param count)? v2 是否在修 stale facts 时引入新的 historical claim 杜撰
- v2 A3 §5 重组要求 codex 把现有 4 条红线**重新嵌入** §5.1 子段. 这是手工编辑器操作 (不是 sed), codex 执行时容易出错. 是否应该提供 A3 的 **完整 §5 重写后内容** 让 codex 直接整段替换, 而不是描述 "把 X 改成 Y"?
- v2 A4 pass 准则 "rank=8 在 §2 范围内只出现在历史引用" 措辞模糊, codex 难以自动判定. 是否应该改成 "rank=8 在 §2.1 设计哲学段消失"?

### Q5 — V21 retire 实施细节 (Q3 改进版)

v2 §2 B1 给了 retire 标记模板:
- 顶部 warning
- 每段 `[RETIRED ...]` blockquote

请 verify:
- v2 B1 说 "对每处 V21 设计描述加段末 retire 标记" + "不对元数据/历史引用加". 但 19 处命中 (REVIEW_INTEGRATION_round4) 中, 如何让 codex 判定哪是 "设计描述" vs "元数据"? 是否应该给 codex 一个**具体决策规则** (e.g. "段落开头出现 'V21 = ' / 'V21 is ' → 设计描述; 段落出现 'retire V21' / 'V21 was' → 元数据")?
- v2 B0 sanity 范围 [5, 50]. 实际 (Round 8 verify) = 21 hits (19+1+1+0). 范围合理, 但 codex 如果只看 v2 文档不知道实际数, sanity check 是否需要给出预期数 ~21?
- 模板 retire 文本里有 strikethrough? Round 8 agent1 推荐 strikethrough + blockquote 双信号. v2 现在只有 blockquote (Q3 没采纳 agent1 strikethrough). 是否影响 audit trail?

### Q6 — NOT-DO 嵌入与 fail 决策树

v2 §6 NOT-DO 扩到 16 条, §7 失败决策树详细列每种 fail 的处理.

请 verify:
- v2 §6 16 条是否真覆盖**新**的 LLM 滑坡风险? agent3 Round 8 提的 4 条 (不改 V13 原 yaml / 不 commit V13 原 yaml / 不改 train_first_hop 加 flag / 不 rm -rf 重跑) 是否全部在 v2 §6?
- v2 每个 task 末尾的 "fail → 停下报告, 不要尝试 fix, 不要 retry, 不要 escalate" 重复 5 次. 是否足够? 或者过度?
- §7 失败决策树**没有**覆盖以下场景, 是否应补?
  - V18 在 codex 执行 Task A 期间崩溃 (slot 1 死) — task md 没说该怎么办
  - V18 跑到 step 200K (Phase A 期间) — Phase A 报告需要更新吗? 还是阶段 B task 接手?
  - git push 时认证失败 / 网络断
  - smoke 完成但 metrics.jsonl 0 行 (训练根本没开始)

### Q7 — Option B push 时机落实度

v2 §5 E1 模板 §4 Task D 强制原文措辞: "V13 process alive at +5min after launch, GPU utilized. **未经长期 soak 验证**, 7 天训练健康度由阶段 B/C 跟踪."

请 verify:
- v2 模板 §4 是否真的让 codex **无法** 写 "V13 健康" / "稳定" / "收敛正常"? LLM agent 可能在自由填空里仍写出乐观措辞
- 是否应该把这条措辞作为 git commit message 模板的一部分 (而不仅是报告内措辞)? commit message 里写 "V13 launch healthy" 同样会留在 gitee 永久
- v2 §5.2 commit message 模板 ("Phase A v2: ...") 是否符合 Option B 措辞要求? 当前写 "V13 smoke pass + launch" + "healthy 度由阶段 B/C 跟踪" — 但前半句仍隐含 "smoke pass = 整体 ok" 的乐观推断, 是否要再降级?

### Q8 — git commit 范围 (Q6 改进版)

v2 §5 E2 显式 add 6 类文件, 排除 pid + rolling V13 train log.

请 verify:
- v2 add 的 `smoke_runs/V13_smoke_*.yaml` 和 `smoke_runs/V13_smoke_*.log` 用了通配符. 若 SMOKE_DIR 之前残留旧 smoke 文件 (例如 Phase A v1 trial run 留的), 会被 git add 进来. 是否应改为显式 `${SMOKE_YAML}` `${SMOKE_LOG}` 单文件 add?
- v2 没要求 codex 在 commit 之前 verify `.gitignore` 是否含 `*.pid` (B22 防御). 如果 .gitignore 没排除 pid, 即使 codex 没显式 `git add *.pid`, 后续 `git add review/0516/V13_true_image_aux_ablation/` 通配可能误 add. 是否应在 v2 加一条 ".gitignore preflight check"?
- v2 E1 §8 git diff --stat 预期改动文件 7 个. 但 §A3 §5 重组涉及修改 `V18_design_rationale.md` 多段, "+60 行" 估计是否合理? 实际可能 +30 或 +90, codex 无法判定 "预期符合" vs "意外改动"
- v2 commit message 是否符合 Conventional Commits 或项目约定? (本轮 reviewer 不评 commit 风格, 只评是否符合现有 repo 习惯)

### Q9 — 起草 v2 时 claude 是否复发 B21-B25 (新偏差识别)

v2 是 claude 在 Round 8 教训后起草. **请独立检查**:

- v2 是否引入新 B21 (CLI flag 杜撰)? 例如 §C0 `SUPPORTED_FLAGS` 命令是否真有效, 是否还有别处 CLI 假设?
- v2 是否引入新 B22 (path/yaml 字段路径假设)? 例如 v2 §3 python 脚本访问 `cfg["training"]["best_select_full_eval_interval"]` — 这字段在 V13 yaml 真存在吗? 还是只在某些 yaml? `setdefault("training", {})` 是否有副作用 (e.g. 覆盖 yaml 已有的 training section 某些字段)?
- v2 是否引入新 B23 (premature claim)? 例如 §5 E3 完成通知 "PHASE A v2 COMPLETE" — 这本身就是声明, 是否过早?
- v2 是否引入新 B24 (baseline 数字未给)? 例如 v2 准则 5 "≤ 15 min" 是绝对值, 但这数字哪来的 (V7 200 step 实际多久? agent1 Round 8 也问了这个)?
- v2 是否引入新 B25 (假设文档/代码结构)? 例如 v2 §A1 假设 V18_design_rationale §2.1 是表格 (Table form). 实际是否是 markdown table? 还是只是有序列表?
- v2 是否引入**全新** Bxx 偏差 (Round 8 没识别过)?

---

## 2. 资料目录

按读的顺序:

### 2.1 本轮主审对象

- **[CODEX_TASK_PHASE_A_v2_20260517.md](./CODEX_TASK_PHASE_A_v2_20260517.md)** — codex task v2 (本轮要 review)

### 2.2 上游 (本轮不审, 仅作 context)

- [CODEX_TASK_PHASE_A_20260517.md](./CODEX_TASK_PHASE_A_20260517.md) — v1 (已 supersede; verify v2 真的修了 Round 8 列的 v1 问题)
- [REVIEW_INTEGRATION_round8_20260517.md](./REVIEW_INTEGRATION_round8_20260517.md) — Round 8 整合 (列出 5 hard fix + 3 should fix, user 决策 Option B)
- [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](./NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) — Round 7 用户签字架构

### 2.3 代码事实验证 (Round 8 已 spot-check, Round 9 二次确认)

- [train_first_hop.py:1355-1357](../../train_first_hop.py) — argparse 仅有 `--config` / `--resume`
- [train_first_hop.py:2166-2178](../../train_first_hop.py) — image_aux disabled 时 ssim_raw=nan
- [train_first_hop.py:2446](../../train_first_hop.py) — metrics_payload["img"] 字段
- [train_first_hop.py:1815-1817](../../train_first_hop.py) — metrics.jsonl 写入条件 + 路径
- [pet_lr/path_guard.py](../../pet_lr/path_guard.py) — output_dir 必须在 /data_2/ 下
- [V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) — V13 配置, 验证字段嵌套结构
- [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) — Task A 要修改的目标文档

### 2.4 V18 不动的硬约束

- V18 当前 step ≈ 180-200K (~12-18h 剩余), slot 1, **本 task 完全不动**

---

## 3. 输出格式 (请 reviewer 严格遵守)

每位 reviewer 独立产出 markdown:

### 3.1 9 个问题逐条回答
对 Q1-Q9 每条**明确**给出 1 个结论:
- `APPROVE` — v2 真修了/真做了, 可接受
- `MODIFY (说明改什么)` — v2 修了一部分, 还有改进空间但不阻塞
- `REJECT (说明为什么)` — v2 没真修, 或引入新 bug, 阻塞 push

### 3.2 整体 verdict (对 [CODEX_TASK_PHASE_A_v2_20260517.md](./CODEX_TASK_PHASE_A_v2_20260517.md))
- `READY TO PUSH` — 可直接 push 给 codex 执行
- `MODIFY THEN PUSH` — 列出必改项后可 push (≤ 3 项)
- `BLOCK` — 有未修的 hard blocker 或引入新 hard blocker, 需重新起草 (v3)

### 3.3 v2 引入的新偏差 (重点)
若你发现 v2 在 Round 8 修复中引入了新 confirmation bias (例如修 B21 时又造新 B26), 用 B26+ 编号单独列出, 含:
- 偏差形态
- 在 v2 哪一段
- 代码/文档证据
- 修复建议

### 3.4 (可选) 代码层 spot-check
若你 verify 了 v2 某段与实际代码/文档不符, 给具体证据 (代码片段 + 行号 + 实际值).

### 3.5 (可选) 替代方案
若 verdict = MODIFY/BLOCK, 给一份你认为更合理的修订要点 (不必完整重写).

---

## 4. 约束与提醒

- **本轮不重审** V18/V13/V18-clean 设计、三阶段路线、Q5 push 时机 (user 已签 Option B)
- **不假设** 你能跑代码 / 看 wandb / 触 GPU — 仅凭 git artifacts 评
- **特别关注**:
  - v2 是 claude 在 "知道 Round 8 教训" 后的产物. 它**应该**没复发 B21-B25, 但**可能**复发. Reviewer 的核心价值 = catch claude 自己看不到的盲区
  - codex 是 LLM agent, 倾向 "尝试 fix" 而非 "停下报告" — v2 §7 决策树是否真够强?
  - v2 文档假设的代码/文档结构必须**全部**与实际匹配 (B25 教训, 不依赖 claude 的记忆)
- **优先质疑**:
  - Q4 (V18_design_rationale 结构): 这是 Round 8 最难修的一处, v2 修法是否真的 actionable for codex
  - Q9 新偏差: claude 是否在修 B25 时又造新 B25-prime
  - Q3 metrics.jsonl 准则: 字段名 / 写入频率 / 路径 三处 verify 是否准

---

## 5. 给 reviewer 的硬约束摘要 (standing, 不可推荐违反)

任何 reviewer 提议都必须满足:

1. ≤ 3 并行训练任务 (本 task 已用 slot 1 = V18 + slot 2 = V13, slot 3 = null)
2. V18 不动 (slot 1 训练中)
3. V18b 永久撤销 (Round 6 共识 + 代码 hard blocker)
4. 阈值不改 (Round 7 user 签字 `全推荐` = agent3 不改路线)
5. 不预先 pre-register V18-clean (阶段 C 才 pre-register, Round 7 B18 教训)
6. 不推 Round 10 (本 task 自己就是 Round 9 终点; 修完直接 push)
7. user Option B (5 min push + 措辞降级) 已签, 不可推翻

违反任意一条 → reviewer 提议自动作废.

---

## 6. 本轮元说明

Round 1-7 审 design (做什么 / 怎么决策).
Round 8 首次审 execution (codex 能不能真跑).
Round 9 审 execution fix 的 correctness (Round 8 修复是否真有效).

如果 Round 9 reviewer 一致 READY TO PUSH 或仅给 < 3 个 should-fix → 直接 push gitee 给 codex.

如果 Round 9 出现新 hard blocker → 起 v3 + Round 10 (但 Round 8 已立 "Round 9 是 execution 审稿终点" 期望, Round 10 是 user 决策点).
