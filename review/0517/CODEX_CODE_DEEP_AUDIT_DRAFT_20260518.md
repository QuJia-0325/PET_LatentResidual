# Codex Code Deep Review Request [PENDING USER RELEASE — DO NOT EXECUTE]

> ⚠️ **DO NOT EXECUTE — USER MUST RENAME FILE FIRST**
> 本文件是 DRAFT. user release 流程:
> 1. 重命名文件: `CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md` → `CODEX_TASK_CODE_DEEP_AUDIT_<TS>.md` (去 DRAFT 字样, 加 ACTIVE 时间戳)
> 2. 删除本顶部警告段落 (连同 blockquote 与本说明)
> 3. 修改 §10 ACTIVATION 状态为 "已 release 于 <TS>"
> 4. push gitee 后 codex 才可执行
> 若 codex 看到本顶部警告未被删除, 必须立即停下报告 user, 不执行任何 task.

- date: 2026-05-18 (draft)
- branch: foc_lite_hop0
- status: **DRAFT**. 此文档只在 Phase A v3 push gitee + codex 执行完 + V13 slot 2 训练正常之后才发送给 codex.
- 触发条件: Phase A v3 [CODEX_TASK_PHASE_A_v3_20260518.md](./CODEX_TASK_PHASE_A_v3_20260518.md) 执行完毕, PHASE_A_EXECUTION_REPORT 显示全部 PASS, V13 (slot 2) alive at +5min.
- 上游: [REVIEW_INTEGRATION_round9_20260517.md](./REVIEW_INTEGRATION_round9_20260517.md) §10 元教训 + user 请求 "阶段 A 完成后对代码进行一次深入 peer review"
- 硬约束: V18 (slot 1) 仍在跑 / 阶段 B 未启动 / 不动任何 yaml / 此 task 是 **0 GPU read-only audit**

---

## §0 — 给 codex 的 5 句话总览

1. 这是 **0 GPU read-only code audit**, 不跑任何训练 / eval, 不改任何代码.
2. 目的: 找出 Round 1-9 用 markdown spot-check **没覆盖**的代码 silent bug, 为阶段 B (V18 200K eval + KL drift) / 阶段 C (V18-clean launch, 可能触发) 提供代码层信心.
3. 产出: 一份 `CODE_AUDIT_REPORT_<TS>.md`, 按 §2-§7 模块逐节写发现.
4. **不要修复任何 bug**. 只 report. 修复决策属 user.
5. 不动 V18 (slot 1) / 不动 V13 (slot 2). 不起任何 review (此 task 自身是 audit, 终点 = 报告). 

---

## §1 — Audit 范围 & 优先级

### 1.1 必审模块 (P0, codex 必须完成)

| # | 文件 | 范围 | 优先级 | 为何 |
|---|---|---|---|---|
| M1 | train_first_hop.py | 全文 ~2800 行 | P0 | Round 8/9 已 spot-check 关键段 (resume / image_aux / metrics / KL pullback), 但**未全文 audit**. 任何被前 9 轮 review 遗漏的 silent bug 都直接影响 V18 / V13 / V18-clean. 阶段 A 完成 + V13 已在 slot 2 跑后, 是 audit 此文件的最佳窗口 (V18 100% 训完前不抢 GPU 资源) |
| M2 | pet_lr/decoder_lora.py | 全文 | P0 | V18 LoRA 实现核心. 阶段 C V18-clean from V7 launch 前的最后一道防线. `compute_kl_pullback_loss` 是否真的与 frozen decoder paired 比较? `_assert_v18_step0_equivalence` 是否漏 edge case? (B32 需验, 见 Q3.2.1) |
| M3 | pet_lr/model_first_hop.py | 全文 | P0 | `assert_decoder_frozen()` 在循环中 enforce 的实现细节; rae.decode / decode_crop 路径; V18 step 0 equivalence 的可证性 |
| **M5** | **eval_first_hop_224_clip3.py** | **PSNR_clip3 计算路径 + V7 baseline 比较逻辑** | **P0** | **(Round 10 agent2 升 P0)** 阶段 B V18 200K eval 必须用此脚本. 阈值判定 (PRIMARY +0.30) 直接依赖 PSNR_clip3 数字正确性 |

### 1.2 选审模块 (P1/P2, codex 时间允许时做)

| # | 文件 | 范围 | 优先级 | 为何 |
|---|---|---|---|---|
| M4 | pet_lr/path_guard.py | 全文 (30 行) | **P2** | (Round 10 agent3 降 P2) Round 8/9 spot-check 已证 全文简单, 5 min audit 可完 |
| M6 | tools/probe_v18_gap_decomposition.py | KL drift probe (阶段 B 重用此) | P1 | 阶段 B B2 KL drift 测量直接复用此 probe |
| M7 | RAE/RAE/src/stage1/decoders/conv_head.py | V21 retire 依据 | P2 | Round 5 B8 拆穿 V21 的代码事实来源. 阶段 A 已 retire V21, 但 audit 一次确认 B8 描述与代码一致 (preempt future round 复活 V21) |
| M8 | pet_lr/losses_first_hop.py / rollout_first_hop.py | 损失项 + rollout 实现 | P2 | Plan F / FOC-lite 相关, 与 V13 (image_aux off) outcome 解读相关 |

### 1.3 不审 (NOT-DO)

- 不审 Stage 1 RAE 训练相关 (与 V18/V13 无关, 范围爆炸)
- 不审任何 dataset 加载 / preprocessing 代码 (V7 已 lock, 阶段 B/C 不会动)
- 不审 wandb / logging 工具代码
- 不审 deprecated/ 目录任何文件
- 不审 tests/ 任何文件 (即使发现 test 缺失, 不在本 audit 范围)

---

## §2 — 给 codex 的 audit 方法论

### 2.1 read-only 工作流

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull gitee foc_lite_hop0
git rev-parse HEAD   # 记录 HEAD, 写进报告

# 创建 audit 工作目录 (不动训练数据)
AUDIT_DIR=review/0518/code_audit_$(date +%Y%m%d_%H%M%S)
mkdir -p "$AUDIT_DIR"
```

### 2.2 每个模块的 audit 模板

对 M1-M8 每个文件, 按以下结构 audit:

```markdown
## Mn: <文件路径>

### 文件元数据
- 行数: <wc -l>
- 最后修改: <git log -1 --format='%h %ai %s' -- <file>>
- 与 V18/V13/V18-clean 关系: <2-3 句>

### 已知 reviewer 已 spot-check 段 (Round 1-9 verify 过的)
- 列出 Round 5-9 整合文件里引用过的 file:line 范围
- 这些段 codex **简单 cross-verify** 一次, 不重做深 audit

### 未 spot-check 段 (本次 audit 主战场)
按代码内逻辑切分, 每段:

#### 段 N: <函数名 / 类名 / 行范围>
- 功能: <1 句>
- 输入 / 输出 / 副作用
- silent bug 怀疑点 (按以下 7 类 checklist):
  1. **未检查的索引 / 数组越界** (off-by-one, 空 list 取 [0], dict 缺 key)
  2. **NaN / Inf 在中间步骤** (除以可能为 0 的量, log/sqrt 负数, exp(很大数))
  3. **浮点比较** (`== 0.0` 而非 `< eps`)
  4. **资源泄漏** (file 不关 / DataLoader worker 不 kill / GPU memory 累积)
  5. **race condition** (多 process / 多 thread / 多 GPU 共享变量)
  6. **silent type coercion** (numpy / torch tensor 在边界自动 broadcast)
  7. **与 V18/V13/V18-clean spec 不一致** (e.g. yaml 字段语义与代码实际行为差别 = B19/B20 同型)

### 模块判定
- HEALTHY: 无 silent bug 怀疑 → 阶段 B/C 可放心依赖
- WATCH: 有 ≤ 3 个 LOW severity 怀疑点, 不阻塞但建议 user 看一眼
- FLAG: 有 ≥ 1 个 MEDIUM/HIGH severity 怀疑点, 阶段 B/C launch 前必须 user 决策
```

### 2.3 sanity 范围

| 模块 | 期望 audit 时长 | 期望产出行数 |
|---|---:|---:|
| M1 train_first_hop.py (~2800 行) | 2-3 h | 200-400 行 audit |
| M2 decoder_lora.py | 30 min | 50-80 行 |
| M3 model_first_hop.py | 1 h | 100-150 行 |
| M4 path_guard.py | 5 min | 10 行 |
| M5 eval_first_hop_224_clip3.py | 1.5 h | 100-180 行 (提为 P0) |
| M6 probe_v18_gap_decomposition.py | 20 min | 40-60 行 |
| M7 conv_head.py | 20 min | 30-50 行 |
| M8 losses_first_hop.py + rollout_first_hop.py | 1 h | 100-150 行 |
| 总 | **~8-12 h** | **~13 个 actionable Q × §3.0 限制 → 报告 §3 总计 ~400-600 行** (Round 10 修正: 原估 22 Q × 30 行 = 660 行, 实际查后 ×Y 重复/无法静态答, 只剩 ~13 真 actionable, 每题限 3-8 行) |

任何模块 audit 超 2× 预期 → 停下报告, 不要 escalate scope.

---

## §3 — codex 必须回答的具体问题 (Round 1-9 reviewer 列的待 verify)

下列问题在前 9 轮 review 中被列出但**没**在本地 spot-check 中 verify, 必须在本 audit 全部回答:

### 3.1 train_first_hop.py 待 verify 列表

| # | 问题 | 来源 | 期望产出 |
|---|---|---|---|
| Q3.1.1 | KL pullback `compute_kl_pullback_loss` 的实现里, `decode_frozen(z_kl)` 是否真的 `torch.no_grad()` 包装? 若没有, frozen path 会触发梯度 retention | Round 6 agent2 | 代码片段 + 结论 |
| Q3.1.2 | image_aux 与 KL pullback 是否共享同一次 forward 或两次独立 forward? 若两次, batch 数据是否同一个? 这决定 B9 真实梯度对抗程度 | Round 6 agent2 + Round 7 | 代码路径图 + 结论 |
| Q3.1.3 | `_assert_v18_step0_equivalence` 检查精度 `1e-5` 是否在 fp16 / bf16 上太严? V18-clean 若用混合精度训, 这步可能 raise | Round 6 + Round 8 | 测试结果 + 结论 |
| Q3.1.4 | resume 路径里 `start_step` 是从 ckpt 取还是从 yaml 取? V18-clean from V7 best.pt (step 160000) 的实际 start_step 是 160000 还是 0? B20 假设是 160000, 但**没**代码层 verify | Round 7 + Round 8 | 函数调用链 + 结论 |
| Q3.1.5 | `lr_schedule.total_steps_override` 在 V7→V18 resume 时, cosine decay 起点是 0 还是 ckpt step? V18 当前训练用了 `total_steps_override: 200000`, 实际 cosine 曲线形状是什么? | Round 7 agent1 | LR vs step 的实际计算公式 |
| Q3.1.6 | `metrics_jsonl_mode = "append"` 在 resume 时是否真的不覆盖前 ckpt 的 metrics? Round 8 verify 了字段名, 但没 verify resume 行为 | Round 8 agent2 | 代码片段 + 结论 |
| Q3.1.7 | `best.pt` 选择 criterion 是什么? `val_chain_normal_mse` 最低? `val_multi_objective` 最高? V18-clean 与 V18 best comparison 必须用同一 criterion | Round 7 + Round 8 | 代码片段 + 结论 |
| Q3.1.8 | watchdog / hang detection 机制是否存在? V13 7 天训练若卡 dataloader, codex 是否有自动 kill? | 历史经验 | 代码搜索结果 |

### 3.2 decoder_lora.py / model_first_hop.py 待 verify

| # | 问题 |
|---|---|
| Q3.2.1 | (Round 10 B32 修正方向 + Round 11 B33c 修措辞) yaml `init_scale_zero=False` 路径在 `pet_lr/decoder_lora.py:95, 123-130` 的实际行为: 字段是否 silently 被忽略 (仅 print WARN 后仍走 zero-init)? 实际行为与字段名/用户期望 (False = 非 zero init) 的 gap 是什么? 影响阶段 C V18-clean 是否能用此 yaml 字段调试 LoRA init? **(audit 仅报现状与 gap, 不建议具体 patch; 修复决策属 user, 见 §5 NOT-DO #9)** |
| Q3.2.2 | `rae.decode` 调用链是否真的 frozen / lora-wrapped 路径区分清晰? 是否存在某条 forward path 漏走 LoRA |
| Q3.2.3 | `assert_decoder_frozen()` 在 V18 训练中是否真的每 step 都 enforce? 还是只在 startup 一次? |
| Q3.2.4 | LoRA A/B 的 gradient 是否在某条 path 上被意外 zeroed (e.g. detach 误用) |

### 3.3 path_guard.py 待 verify

| # | 问题 |
|---|---|
| Q3.3.1 | symlink 跨 `/data_2` 边界时 `Path.resolve()` 是否正确 reject? TOCTOU 风险? |
| Q3.3.2 | `ensure_repo_local_outputs_absent` 是否在所有 launch 路径都调用? V13 launch 是否真触发? |

### 3.4 eval_first_hop_224_clip3.py 待 verify

| # | 问题 |
|---|---|
| Q3.4.1 | PSNR_clip3 计算公式是什么? (`-10 * log10(MSE_clipped)` 还是 `20 * log10(MAX/RMSE_clipped)`) clip 是输入 clip 还是输出 clip? |
| Q3.4.2 | baseline-checkpoint 比较时, V7 vs V18 用同 val split? split 是否 deterministic (seed 固定)? |
| Q3.4.3 | NORMAL chain vs D20/D10/D4 chain 的 PSNR 计算是否一致? |

### 3.5 conv_head.py 待 verify (B8 复审)

| # | 问题 |
|---|---|
| Q3.5.1 | ConvDecoderHead 输入是否真的是 `[B, N, 512]` (token domain)? PixelShuffle(14) → 196×196 真符合 V21 retire 时 B8 描述? |
| Q3.5.2 | 是否有别处 (非 conv_head.py) 实现了"真正的 post-decode pixel refiner"? 若有, 阶段 C 可能有新候选 (但不在本 audit 范围决策, 仅 flag) |

---

## §4 — 报告输出格式

文件: `review/0518/code_audit_<TS>/CODE_AUDIT_REPORT_<TS>.md`

```markdown
# Code Audit Report (post Phase A)

- date: <TS>
- HEAD: <git rev-parse HEAD>
- audit 范围: M1-M3 + M5 (P0 必), M4 (P2 仅 5 min), M6-M8 (P1/P2 选)  ← Round 10 调整
- 触发: user 请求 + Round 9 元教训
- 与 V18/V13/V18-clean 关系: 此 audit 不阻塞 V18 (在跑) / V13 (在跑); 阻塞阶段 B/C launch 前的 user 决策

## §1 — 顶层结论 (TL;DR)

| 模块 | 判定 | 关键发现 |
|---|---|---|
| M1 train_first_hop.py | HEALTHY / WATCH / FLAG | <1 句> |
| M2 decoder_lora.py | ... | ... |
| ... | | |

阶段 B (V18 200K eval) launch 前需 user 决策的 FLAG 项: <N>
阶段 C (V18-clean from V7) launch 前需 user 决策的 FLAG 项: <N>

## §2 — Mn 模块 audit (按 §2.2 模板, 每模块一节)

## §3 — Q3.1-Q3.5 问题逐条回答

(每题: 代码片段 + 结论 + 对 V18/V13/V18-clean 的影响)

## §4 — 新偏差识别 (B31+)

(若 audit 中发现 Round 1-9 没识别的偏差类型, 用 B31+ 编号, 写形态 + 建议 standing rule)

## §5 — 不属本 audit 但顺手发现 (TODO list)

(不修, 仅 list; 例如 deprecated/ 里有死代码 / docs/ 里有 stale 描述 / 测试覆盖缺失等)

## §6 — codex 自警 (本 audit 自身是否复发 Round 1-9 偏差?)

按 16 项 NOT-DO + B12-B30 列表 yes/no 自查
```

---

## §5 — NOT-DO (此 audit task)

1. ❌ 不改任何代码 (only read)
2. ❌ 不跑训练 / eval (V18 在 slot 1, V13 在 slot 2, 不抢)
3. ❌ 不动任何 yaml
4. ❌ 不停 V13 / 不重启 V13
5. ❌ 不修复发现的 bug (只 report, 修复决策属 user)
6. ❌ 不起 Round 11 / 不二次 audit 自己 (此 task 自身是终点)
7. ❌ 不顺手做未列范围的事 (M9 / M10 不审, 阶段 B yaml 不写, V18-clean yaml 不写)
8. ❌ 不在 commit message 写"也顺便审了 X"
9. ❌ 不在 audit 报告里写"建议立即修 X" (修复决策属 user)
10. ❌ 不 push audit 报告之前不停问 user "要不要继续" — codex 一次性完成全部模块 audit + 报告 + commit, 然后停下等 user 反馈
11. ❌ **不跳过看不懂的代码段** (Round 10 新加) — 看不懂的段必须在模块报告里标 `UNCERTAIN: <原因>`, 该段报 FLAG 等 user 决策. **不允许 silent skip** (是 LLM audit 特有滑坡风险)
12. ❌ **不审 deprecated/ 目录任何文件** (Round 10 重述 §1.3, 防止 audit 时顺手拓展)
13. ❌ **不审 dataset / preprocessing / wandb / logging 工具代码** (§1.3 已排除, 本项重述)

---

## §6 — 失败决策树

```
Audit 某模块时发现该模块 import 的 sub-module 也需要 audit?
  → 在报告 §5 TODO list 标记, 不扩大本 audit 范围

Audit 中发现 V18 (slot 1) 训练状态异常 (e.g. log 显示 NaN)?
  → 不要尝试干预 V18 (slot 1 不在本 task 范围). 在报告 §4 标记 "FLAG: V18 状态异常需 user 关注", 继续 audit 其他模块

Audit 中发现 V13 (slot 2) 训练状态异常?
  → 同上, 标记 FLAG, 不动 V13

Audit 时间超 2× 预期 (>16 h)?
  → 停, 不完成的模块在报告 §1 标 "未完成", 等 user 决策是否分批

发现 Round 1-9 已 verify 段实际有错 (即前 reviewer 错了)?
  → 在报告 §4 高优 flag (说明哪一轮 review 错了 + 代码事实), 不修复
```

---

## §7 — push gitee

audit 完成后:

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

git add review/0518/code_audit_<TS>/CODE_AUDIT_REPORT_<TS>.md

# 验证 git status 无意外文件
git status

# commit
git commit -m "Code deep audit (post Phase A): M1-M8 review, <N> FLAG items, <M> WATCH items

- M1 train_first_hop.py: <judgment>
- M2 decoder_lora.py: <judgment>
- ...
- 阶段 B launch 前需 user 决策: <N> FLAG
- 阶段 C launch 前需 user 决策: <N> FLAG
- 不修任何 bug, 只 report (修复决策属 user)
- 触发: REVIEW_INTEGRATION_round9 §10 元教训 + user 请求"

git push gitee foc_lite_hop0
```

---

## §8 — 总时长估计 (Round 10 修正: M5 升 P0 + M4 降 P2)

| Task | 时长 |
|---|---|
| M1-M3, M5 (P0 必) | 5-7 h (M1 ~3h + M2/M3 ~1.5h + M5 ~1.5h) |
| M4 (P2 仅 5 min) + M6-M8 (P1/P2 选) | 2-3 h |
| 报告整合 + commit + push (限制单题产出 3-8 行) | 1-1.5 h |
| **总** | **~8-12 h** (1-1.5 个工作日) |

D6 (Round 10 agent2): §3 的 22 个 Q 有大约 9 个是 Round 1-9 已 spot-check 需重复验证 (如 KL no_grad / metrics_jsonl_mode), 可用“已知 Round X agent Y 答过, 仅纳入 audit report”过. 真需新答的约 13 个, 单题 3-8 行足够.

GPU 占用增量: 0 (read-only audit).

---

## §9 — 与 Round 1-9 工作流的关系

| 轮次 | 范围 | 本 audit 与之关系 |
|---|---|---|
| Round 1-5 | V18 设计审查 | 本 audit 不重审设计 |
| Round 6 | V18b draft 审查 | V18b 已永久撤销, 不涉及 |
| Round 7 | NEXT_STAGE 设计审查 (三阶段路线) | 本 audit 为阶段 B/C launch 提供代码层信心 |
| Round 8 | v1 task md 执行审查 (5 hard blocker) | 本 audit 验证 Round 8 spot-check 没遗漏 |
| Round 9 | v2 task md 执行审查 (2 hard blocker) | 本 audit 完善 Round 9 元教训 (从抽象 standing rule 升级为 mechanical checklist) |
| **本 audit** | **整代码全文 review (P0-P2 8 模块)** | **填补前 9 轮 spot-check 漏的 silent bug, 不再开 Round 11** |

阶段 B / 阶段 C 启动前的最后准备工作.

---

## §10 — ACTIVATION 状态

**当前状态**: PENDING USER RELEASE (未激活, codex 不执行).

发送时机: Phase A v3 push gitee → codex 执行完 → PHASE_A_EXECUTION_REPORT 显示 PASS → V13 (slot 2) 5 min alive 验证通过.

发送方式 (user 决策 "OK 发" 后):
1. **重命名**: `CODEX_CODE_DEEP_AUDIT_DRAFT_20260518.md` → `CODEX_TASK_CODE_DEEP_AUDIT_<TS>.md` (去 DRAFT 字样, 加 ACTIVE TS)
2. **删顶部警告**: 删除文件最顶的 `> ⚠️ DO NOT EXECUTE — USER MUST RENAME FILE FIRST` blockquote (5 行)
3. **改本节**: 将本节顶部 "PENDING USER RELEASE (未激活..." 改为 "已 release 于 <TS>, ACTIVE"
4. **push gitee**: 单独 commit, 不与其他 task 合并

**双保险机制** (B23 同型防御): codex 拉下未重命名文件 → 看到顶部 ⚠️ 警告 → 立即停下报告 user, 不执行. 隔离了 LLM 可能误判 [DRAFT] 为 informational 的风险.

claude 起草本 DRAFT 的目的:
1. 把 user "阶段 A 完成后做代码深审" 的需求 **固化** 为可执行 task md, 避免日后忘记
2. 让 user 提前过一眼范围 + 方法论 + NOT-DO, 阶段 A 完成时 user 不需要现想 audit 怎么做
3. 给 Round 9 元教训一个 actionable outcome (从抽象 standing rule 到具体代码 audit task)

如 user 现在不想要本 DRAFT (例如认为 Round 1-9 spot-check 已够, 不需深审), 可删除本文档. 但 claude 强烈推荐保留 — V18-clean from V7 launch 是高 GPU 成本操作 (~1.4 天 slot 3), audit 0 GPU 成本 8 h, ROI 极高.
