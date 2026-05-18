# Round 6 Review Integration — V18b draft 评审交叉核对

- date: 2026-05-17 深夜
- branch: foc_lite_hop0（V18b draft 仍未 push）
- 主审对象: [CODEX_TASK_V18b_DRAFT_20260517.md](./V18_decoder_lora/CODEX_TASK_V18b_DRAFT_20260517.md)
- 三位独立 reviewer: agent1 / agent2 / agent3（互不知情）
- 本文目的: 交叉核对三份 review，识别共识、分歧、新偏差，给出 push/重写决策

---

## 0. 一行结论

**三位 reviewer 一致 REJECT 当前 draft（不可原样 push）**。共识区域比 Round 5 更窄、更尖锐：阈值修改与 V18b yaml 都不过关，但**修改方向不一致**——需要 user 仲裁。

| 维度 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| Overall verdict | REJECT (§3 yaml + §2.4 重写) | MODIFY THEN PUSH | REJECT_AS_DESIGNED | **不可原样 push** ✅ |
| §2 阈值改 +0.30→+0.15 | REJECT，goalpost-moving | 降级措辞可接受 | REJECT，B11 bias 同型 | **2/3 强 REJECT** ⚠️ |
| §3 V18b yaml | REJECT，4 处 bug | REJECT，5 处必改 | REJECT，4 处 bug | **3/3 REJECT** 🔴 |
| §6 V21 retire 推迟 | REJECT，本次必清 | 顺手清 | (未强调) | **2/3 REJECT** |
| Day 0 不停 V18 | APPROVE | APPROVE | APPROVE | **3/3 APPROVE** ✅ |
| B9 机制存在 | YES (代码确认) | YES，但措辞过强 | YES | **3/3 确认** ✅ |
| 0.5 dB 是 attackable | NO，是下界 | NO，是 proxy | NO，是悲观下界 | **3/3 确认是错** 🔴 |

---

## 1. 三 reviewer 完全一致的事实层结论（无争议）

### 1.1 B9 在代码层确实存在

agent1（读 train_first_hop.py:2220-2237）+ agent2（读同段 + losses_first_hop.py）+ agent3（信任 Round 5 + 自查）一致确认：

```python
z_kl = main_out["z_pred"] if use_pred_latent else main_batch["z_dst"]
```

`use_pred_latent=true` → KL 与 image_aux 作用在同一套 RAE LoRA decoder 参数上的同一 z_pred 路径。**B9 不是文档偏差，是真实代码行为**。

但 agent2 给了重要 nuance：**KL 用 main batch，image_aux 用 hop0 auxiliary batch**——不是同一次 forward，也不一定同一批样本。所以 draft §1 "同一 decode_lora(z_pred) 输出上梯度对抗" 措辞不精确。

### 1.2 0.5 dB ≠ V18 attackable gap

三人独立指出同一件事：

| reviewer | 用词 | 数值表述 |
|---|---|---|
| agent1 | "B11 = B10 的镜像错" | 区间 [0.5, ~3] dB |
| agent2 | "conservative pixel penalty proxy" | 不是严格上界 |
| agent3 | "悲观下界" | 区间 [0.05, 3] dB |

**MSE 分解（agent3 给出，已核对 GAP_DECOMP）**：
- decode(z_pred) vs x_target MSE = 2.764e-4 (35.44 dB) = 100%
- decode(z_pred) vs decode(z_GT) MSE ≈ 92%（transport error 主导）
- decode(z_GT) vs x_target MSE ≈ 8%（decoder ceiling）

0.5 dB 假设 V18 LoRA 只能吃 8%（decoder ceiling 那段），**忽略了 V18 训练目标含 image_aux on (z_pred, x_target)，字面意义就是让 decoder 学 z-domain 补偿**。能否吃到 92% 是 empirical 问题。

**新偏差 B11**: 用"较小 scalar"替换"较大 scalar"做 attackable，**结构上与 B10 同型，仅方向相反**。

### 1.3 V18b "从 V18 step 200K → +20K" 设计本身被 B9 自身污染

三人独立给出同一论证：
- 若 B9 真有害，V18 step 200K ckpt 已经是 corrupted 起点（40K step 在 antagonistic gradient 下训练）
- +20K with use_pred_latent=false 测的是 "switch 后的短期 recovery"，**不是 B9 的 clean ablation**
- 若 ΔPSNR(V18b - V18) ≈ 0，无法 disentangle "B9 不重要" vs "B9 重要但 V18b 没机会摆脱起点"

干净 ablation: **V18b 应从 V7 best.pt 出发，use_pred_latent=false，max_steps=200K，与 V18 schedule 对称**。代价 = 7 GPU-day（与 V18 等价）。

### 1.4 LR schedule 致命问题

agent1 + agent2 + agent3 都独立发现，但 agent1 / agent3 写得最清楚：

V18 yaml 里 `lr_schedule.total_steps_override: 200000`。V18 跑到 step 200K 时：
- base lr → min_lr = 2.0e-6
- decoder lr = 2.0e-6 × 0.00625 = **1.25e-8**

V18b 若从 step 200K 出发 +20K，**全程 LR 钉在 1.25e-8**（V18 launch 时 LR 5e-7 的 0.025×）。**LoRA 参数几乎不能动**——即使 B9 修复有效，也测不到任何 ΔPSNR 信号。

如果改 `total_steps_override: 220000`，又引入 schedule confound（V18b 与 V18 schedule 不同 = B5 同型）。

### 1.5 V18b yaml 在代码层的 hard blockers（agent2 独家发现，必读）

agent2 通过读 train_first_hop.py 代码确认两个**会让 V18b 一启动就失败**的问题：

**Blocker 1**: `training.resume_from` 字段 **不被 train_first_hop.py 读取**。它只读 CLI `--resume`。draft §3 B2 改这个字段没用。codex 启动命令必须显式 `--resume <V18 step_200000.pt>`。

**Blocker 2**: 当前 resume 逻辑是为"V7 非-LoRA → V18 LoRA warm-start"写的，会要求 decoder LoRA resume 时发生 base Linear → `.linear` remap，并跑 step0 equivalence check。V18 step 200K 已是 LoRA checkpoint，**不会触发 remap**，且 LoRA 已训过后**也不应等价于 frozen decoder**。按现有代码，**V18b 从 V18 ckpt 续训很可能直接 raise**。

**含义**：V18b launch 不是 "改 yaml 6 字段" 这种轻量改动，而是需要先**给训练器加 LoRA→LoRA resume 支持**的代码补丁。draft 完全没意识到这点。

---

## 2. 三 reviewer 分歧点（需 user 仲裁）

### 2.1 阈值修改如何处理（agent1 vs agent2 vs agent3）

| reviewer | 推荐做法 |
|---|---|
| agent1 | **双轨披露**：原 +0.30 与新 +0.15 并列，评估时同时报告两套判定结果，写明 "post-hoc-PASS, pre-reg-FAIL" |
| agent2 | **降级措辞**：阈值改可保留，但说成"ex-ante 发现 baseline 错误后的保守重标定"，不要写成严格数学推导 |
| agent3 | **完全不改阈值**：只在 §2.4 文档化 EV 区间 [0.05, 3] dB，保持 +0.30/+0.05 原阈值，加 SECONDARY +0.05 dB 作为 "结构性信号" |

**核心分歧**：阈值是否可在 V18 还在跑（已 180K/200K = 90%）时改？

- agent1 视角：launch 之后改 = post-hoc，违反 §5.2 "预注册不可事后改" 红线
- agent2 视角：可改但要诚实标注
- agent3 视角：不改才是干净的，加 SECONDARY 满足 hedge 需求

**user 决策点 1**: 选 agent1（双轨）/ agent2（降级措辞）/ agent3（不改）？

claude 倾向 **agent3**，理由：
- 最少争议、最稳健
- "不改阈值 + 加 SECONDARY" 等价于"事实上的双轨"但表面上没改原阈值，规避 goalpost-moving 指控
- 评估时如果 V18 = +0.20 dB，可以同时报告 "原阈值 PARTIAL / SECONDARY ABOVE / EV 区间内"，三种信号给读者自己判断

### 2.2 V18b 是否应该现在写（agent1 vs agent2 vs agent3）

| reviewer | 推荐 |
|---|---|
| agent1 | 重新设计后可写，但优先级低于 step 180K early-eval |
| agent2 | 必须先加 LoRA→LoRA resume 代码补丁才能讨论 |
| agent3 | **不写**。等 V18 step 200K eval 完，按结果决定 design |

**user 决策点 2**: 现在写 V18b yaml / 不写？

claude 倾向 **agent3 + agent2 的代码 blocker 警示**：现在不画 V18b yaml，原因：
- 代码 blocker 没解决，写了也不能 launch
- design 选择（从 V7 vs 从 V18 ckpt）等 V18 eval 出来才能决
- 现在画 = sunk-cost framing，承担 4 处 bug 风险，0 收益

### 2.3 是否补 step 180K early-eval（agent1 独家提议）

agent1 §6: V18 step 180K 已有 metrics.jsonl 记录。在 step 180K 提前评估，三档触发：
- ΔPSNR ≥ +0.10 dB → V18 continue 到 200K
- ΔPSNR ∈ [+0.03, +0.10] → continue + 准备 V18-clean
- ΔPSNR < +0.03 dB → V18 EARLY-KILL，slot 1 改 V18-clean from V7

agent2/agent3 没提这个。agent3 隐含同意（"V18 不是黑盒，180K 已有数据"），但没明说要 early-eval。

**user 决策点 3**: 加 step 180K early-eval protocol 到本次 commit / 留作下一次?

claude 倾向 **加**，理由：
- 0 GPU-hour（pandas op on existing metrics.jsonl）
- 防止 sunk-cost continue 浪费 slot 1
- 与 V13 launch（Round 5 共识 slot 2）的决策联动

---

## 3. 共识可立即执行项（无争议）

如果用户选最保守方案，本次 commit **只做**：

| 项 | 来源 | 操作 |
|---|---|---|
| **新增 §2.4 文档化 B10 修正** | 3/3 共识 | 但要写 EV 区间 **[0.05, 3] dB**，不写 0.5 dB 单点 |
| **§2.2 EV 列改区间** | agent1 + agent3 | "0.5-2.0 dB" → "[0.05, 3] dB (区间, 见 §2.4)" |
| **§2.4 加 B11 自警** | agent1 提议 | 明确写"我们差点用 0.5 dB 重蹈 B10 覆辙" |
| **§2.4 加 B12 自警** | agent1 提议 | "baseline 修正话术保护带" 作为方法论偏差 |
| **V21 retire 残留清理** | agent1 强 REJECT 推迟 | 4 个文档加 `[RETIRED 2026-05-17 Round 5: 见 B8]` 标记，**本次 commit 必做** |
| **不改 §2.3 阈值表** | agent3 + agent1（双轨）vs agent2（降级措辞） | 默认采 agent3 = 不改 |
| **不画 V18b yaml** | 3/3 共识 reject 当前 design | 推迟到 V18 200K eval 后 |
| **不停 V18** | 3/3 APPROVE | — |
| **V18 step 180K early-eval** | agent1 独家 | 推荐加，0 GPU-hour |

---

## 4. 新偏差（Round 6 识别）

继 Round 5 B8/B9/B10 之后：

| # | 名称 | 来源 | 形态 |
|---|---|---|---|
| **B11** | 0.5 dB 当 attackable | agent1 + agent3 独立指出 | B10 镜像错：单一 scalar 替代分布，方向反过来 |
| **B12** | 预注册阈值的 "baseline 修正" 话术保护带 | agent1 独家 | Lakatosian rescue: 用"修正错误 baseline"包装 post-hoc 改阈值，未来任何阈值修改都有现成辩护 |
| **B13** | sunk-cost-resume | agent1 + agent3 独立指出 | "V18 已付出 40K，从 V18 ckpt 续比从 V7 重训省"——但若 B9 真有害，40K 已 corrupted |
| **B14** | 配置改动只看显式字段 | agent2 独家 | 改 yaml 6 字段时，忽略了 `lr_schedule.total_steps_override` 联动 + `resume_from` 字段不被代码读取 + LoRA→LoRA resume 缺代码支持 |

B12 最严重——它是**方法论层 meta-bias**，会让后续所有"数据出来后调 metric"的操作都有现成话术。**必须作为 design rationale §5.2 自警第二条加进去**。

---

## 5. 推荐修订后的 Day 0 任务清单（替换 draft §4）

如果 user 同意保守方案，本次给 codex 的最终任务：

```
Day 0 (今天, 0 GPU-hour)

Task A: V18_design_rationale.md 修订
  A1. §2.2 EV 表 V18 行: "0.5-2.0 dB" → "[0.05, 3] dB (不确定区间, 见 §2.4)"
  A2. §2.3 阈值表: 不动 PRIMARY/PARTIAL/KILL 数字；
      加 SECONDARY +0.05 dB "结构性信号"列, 用于触发是否做 follow-up（不改 SUCCESS 定义）
  A3. 新增 §2.4 "Round 5/6 修正记录":
      - 文档化 B10: 11.2 dB 不是 attackable
      - 文档化 B11 自警: 0.5 dB 也不是 attackable, 真实是区间
      - 文档化 B12 自警: 我们差点用 "baseline 修正" 话术改阈值, 决定不改
      - 明确: PRIMARY/PARTIAL/KILL 数字不变, 仅 EV 解释从单点变区间
  A4. §5.2 自警条目加 B12: "禁止用 'baseline 错了' 当借口修改预注册阈值"

Task B: V21 retire 残留清理 (本次必做)
  在以下文档加 "[RETIRED 2026-05-17 Round 5: 见 B8 — V21 描述与 conv_head.py 不符]" 标记 (不删原文, 保留 audit trail):
  - review/0517/REVIEW_INTEGRATION_round4_20260517.md §6.2、§7
  - review/0517/V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md §Phase 0
  - review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md 决策表
  - review/0517/CODEX_RUNBOOK_V18_20260517.md

Task C: V18 step 180K early-eval (0 GPU-hour, ~5 min pandas)
  - 读 review/0517/V18_decoder_lora/run_snapshots/V18_rank32_metrics_snapshot_*.jsonl
  - 读到 step 180000 的 val_chain_normal_psnr_clip3
  - 与 V7 best.pt 同 metric 对比
  - 写 V18_STEP180K_EARLY_EVAL.md, 按 agent1 §6 三档判定
  - **不据此做任何 kill 决策**, 仅作为 slot 2 launch (V13/V18-clean/null) 的 input

Task D: NOT DO
  - 不画 V18b yaml (3/3 reviewer reject design + 代码 blocker 未解)
  - 不改 V18 yaml 任何字段 (V18 还在跑)
  - 不 launch V18b
  - 不 launch V13 (slot 2 等 user 决策 + Task C 输入)
  - 不写 LoRA→LoRA resume 代码补丁 (推迟到决定要 V18b 时)

push 到 gitee, 等 user 收到 codex 执行结果后再决策 slot 2/3。
```

---

## 6. user 决策点汇总（请回复）

| # | 决策 | claude 推荐 |
|---|---|---|
| 1 | 阈值处理: agent1 双轨 / agent2 降级措辞 / agent3 不改+SECONDARY | **agent3** |
| 2 | V18b yaml: 现在写 / 不写 | **不写** |
| 3 | step 180K early-eval: 本次加 / 推迟 | **本次加** |
| 4 | V21 retire 清理: 本次做 / 推迟 | **本次做** |
| 5 | §7 "claude 自警" push 时: 删 / 保留 | **保留** (作为公开 metacognition 记录, 提高未来 review 质量) |

user 4 个决策回完, claude 立即按结果重写 [CODEX_TASK_V18b_DRAFT_20260517.md](./V18_decoder_lora/CODEX_TASK_V18b_DRAFT_20260517.md) → 起 Round 7 final review prompt (如需要) → 通过后 push gitee。

---

## 7. 三 reviewer 偏差/质量评估

| reviewer | 强项 | 弱项 |
|---|---|---|
| agent1 | 概念论证最深 (B11/B12 命名)、双轨制建议有学界先例 | 代码层验证较少, V18b LR schedule 论证依赖 yaml 字段而非读训练器 |
| agent2 | **代码层验证最深** (read 训练器 ~1500 行, 找到 LoRA→LoRA resume hard blocker + resume_from 不被读取)，rescue diagnostic framing 务实 | 阈值问题处理太温和 (允许改+降级措辞), 没识别 B12 |
| agent3 | MSE 分解最严格 (92% transport vs 8% decoder)、Issue 4 launch 条件树逻辑反转最锐利 | 没读代码, 没发现 LoRA→LoRA resume blocker |

**三人组合后的覆盖度比 Round 5 高一档**: 概念 (agent1) × 代码 (agent2) × 逻辑 (agent3) 互补。

特别提示: **agent2 的代码 blocker (LoRA→LoRA resume)** 是本轮最关键发现。如果没有 agent2，draft 即使阈值/EV 都改对，V18b 也会一启动就 raise。这条单独够格成为 reject 理由。
