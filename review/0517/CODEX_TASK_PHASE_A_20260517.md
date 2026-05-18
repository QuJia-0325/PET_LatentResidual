# Codex Task — Phase A (today, 0 GPU-hour training; 1 short smoke + 1 launch)

- date: 2026-05-17
- branch: foc_lite_hop0
- status: **DRAFT, awaiting Round 8 AI peer review before push to gitee**
- 设计来源: [NEXT_STAGE_ARCH_CODE_FINAL_20260517.md](./NEXT_STAGE_ARCH_CODE_FINAL_20260517.md) §1
- 决策签字: user 选择 = `全推荐` (REVIEW_INTEGRATION_round7 §6)
- 硬约束: 服务器内存最多 3 个并行训练任务. Slot 1 = V18 (在跑, 不动). 本 task 最多新增 1 个 task (slot 2 = V13). Slot 3 永远 0.

---

## §0 — 给 codex 的 5 句话总览

1. **不停 V18**. 不改 V18 yaml 任何字段. V18 在 slot 1 跑到 step 200K (~12-18h) 后由阶段 B (另一份 task md) 接管.
2. 本 task 4 件事: (A) 改 1 份 doc, (B) 给 4 份 doc 加 retire 标记, (C) V13 200-step smoke, (D) smoke pass 后 V13 full launch (slot 2).
3. **每一步都有 pass/fail 准则**. 任何一步 fail, 停下来报告, 不要自行 escalate.
4. V13 是 from-scratch (yaml 无 resume_from), **不需要 `--resume`**.
5. 完成所有 step 后, 给 user 写一份执行报告 + push gitee. 不要做 task 范围外的事 (例如不起草阶段 B/C task).

---

## §1 — Task A: 修订 V18_design_rationale.md

### A1: 新增 §2.4 "Round 5/6/7 EV 修正记录"

**位置**: [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md), 在 §2.3 (阈值表) 之后, §3 之前.

**完整内容** (照抄):

```markdown
### 2.4 Round 5/6/7 EV 修正记录 (不改阈值, 仅更新 interpretation)

#### attackable_gap 不是单一数字

| 量 | 值 | 物理含义 |
|---|---:|---|
| psnr_ceil = PSNR(decode_V7(z_GT), x_target) | 46.64 dB | RAE round-trip 上限 |
| psnr_transport = PSNR(decode_V7(z_pred^V7), x_target) | 35.44 dB | V7 baseline 实测 |
| psnr_transport_vs_ceil = PSNR(decode_V7(z_pred), decode_V7(z_GT)) | 35.94 dB | "若 z_pred=z_GT 仍用 V7 decoder" |

**V18 attackable gap = 区间 [0.05, 3] dB**, 不是单点估计:
- 下界 ~0.5 dB: 假设 V18 LoRA 只能学到 "decode(z_pred) 对齐到 decode(z_GT)" (吃 8% decoder ceiling 部分)
- 上界 ~3 dB: LoRA rank=32 on last 2 blocks (589K trainable / 24M total ≈ 2.5%) 物理容量约束
- 中位估计 0.1-0.3 dB (落在 PARTIAL [+0.05, +0.30] 区间)

#### 历史偏差 (作为前车之鉴, 不修改预注册阈值)

| Bug | 形态 | 识别于 |
|---|---|---|
| B10 | 11.2 dB 当 attackable (B5 同型, 高估) | Round 4 |
| B11 | 0.5 dB 当 attackable (B10 镜像错, 低估) | Round 5 |
| B18 | 在数据出来前 pre-design 每个 outcome 的响应分支 | Round 7 |
| B19 | yaml `decoder_kl_pullback.warmup_steps` 字段名误导, 实际是 `ramp_steps` (train_first_hop.py:2222-2228 硬编码 `warmup_steps=0`) | Round 7 |
| B20 | resume 后 `max_steps` 是绝对 step 上限, 不是新训步数. V18-clean from V7 best.pt (step 160000) + max_steps=200000 = 实际仅训 ~40K step, 非 200K | Round 7 |

#### 评估方式 (供 §3 决策使用)

V18 200K eval 时同时报告:
1. SUCCESS / PARTIAL / KILL by **原阈值** (单一判定, 无 dual-track, 无 SECONDARY 列)
2. ΔPSNR 在 [0.05, 3] dB 区间的位置 (纯描述性 interpretation, 不参与 SUCCESS 定义)
3. KL drift 数据 (psnr_v18_at_zgt vs psnr_v7_at_zgt = 46.64 dB), 用于 PARTIAL outcome 触发是否做 V18-clean
```

### A2: §5.2 自警条目扩展

**位置**: [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) §5.2 已有 standing rules 段落; 在其末尾追加 B12-B20.

**追加内容** (在 §5.2 现有条目之后):

```markdown
B12 — 禁止以 "baseline 计算错了" mid-run 改预注册阈值. 修正只能通过 §2.4 文字描述,
      不通过替换 PRIMARY 数字, 不通过加 SECONDARY 列, 不通过 dual-track 并列.
      (Round 5 试图把 +0.30 改 +0.15, Round 6/7 reject)

B13 — 禁止 sunk-cost-resume 设计. paired ablation 必须从同一干净起点出发.
      (Round 6 永久撤销 V18b "从 V18 ckpt 续训" 思路, 改 V18-clean from V7)

B14 — yaml 改字段必须 cross-check 联动:
      - LR schedule total_steps_override
      - --resume CLI flag (training.resume_from yaml 字段是死字段, 不被 train_first_hop.py 读取)
      - LoRA→LoRA resume 不被支持 (_assert_v18_step0_equivalence 会 raise non-zero LoRA A/B)

B15 — contingent slot 的真实概率必须披露 (e.g. V18-clean trigger 概率, 不是 "0% 默认")

B16 — "outcome 独立" 框架的 narrow scope: 测量层独立 ≠ 项目优先级层独立.
      但 launch 决策只用测量层独立性.

B17 — paired comparison threshold 必须 cross-check 自身 noise floor (V18 vs V18-clean
      的 paired noise ≠ V7 vs V7-seed1337 的 same-architecture paired noise)

B18 — 禁止 pre-design every branch. 数据出来前不为每个 outcome 设计响应分支.
      (Round 7 元教训; 阶段 A 只做不依赖 V18 数字的事)

B19 — KL config `warmup_steps` 实际是 ramp_steps; 任何引用此字段处必须注 [实际=ramp_steps]

B20 — resume 后 max_steps 是绝对 step 上限; 任何 resume 实验文档必须显式注 "实际训练 N step"
```

### A3: pass 准则

- [ ] [V18_design_rationale.md](./V18_decoder_lora/V18_design_rationale.md) §2.3 阈值表数字**未改动** (PRIMARY +0.30 / PARTIAL [+0.05, +0.30] / KILL +0.05 维持)
- [ ] §2.4 新增段落完整出现在 §2.3 之后
- [ ] §5.2 自警 B12-B20 追加在现有条目之后, 未覆盖 B1-B11
- [ ] markdown 渲染无破坏 (本地 `mdcat` 或 vscode preview 验证)

任何一条 fail → 停下报告, 不要 push.

---

## §2 — Task B: V21 retire 残留清理

### B1: 加 retire 标记 (不删原文)

**目标文档** (4 份):

1. [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) — 任何 V21 引用处 (grep `V21` 找全)
2. [V18_EXECUTION_REPORT_20260517.md](./V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md) — §Phase 0 等 V21 引用处
3. [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) — "Pre-Registered Decision Rule" 表中 V21 行
4. [CODEX_RUNBOOK_V18_20260517.md](./CODEX_RUNBOOK_V18_20260517.md) — 任何 V21 引用处

**操作模板**:

每个 V21 引用段落 (例如 `V21 = freeze V7 + conv head on decode output`) 改为:

```markdown
~~原文 V21 描述~~

> **[RETIRED 2026-05-17 Round 5: B8]** V21 描述与 RAE conv_head.py 实际代码不符.
> conv_head.py 是 `decoder_pred + unpatchify` 的 token-domain 替换 (输入 [B, N, 512] →
> Conv + PixelShuffle(14) → 196×196 输出), **不是** post-decode pixel refiner.
> V21 概念已 retire, 不再列为候选方案.
> 见 [REVIEW_INTEGRATION_round5_20260517.md §B8](./REVIEW_INTEGRATION_round5_20260517.md).
```

(strikethrough `~~...~~` 保留原文, 加 blockquote 标记 retire. markdown 渲染时双信号 — strikethrough 视觉 + blockquote 解释.)

### B2: 文档顶部 deprecation warning

每份目标文档的最顶部 (在原 title 之后, 第一段之前) 加:

```markdown
> ⚠️ **NOTE (2026-05-17)**: 本文档含已 retired 设计 V21.
> 见正文中 `[RETIRED 2026-05-17 Round 5: B8]` 标记 + Round 5 B8 偏差识别.
```

### B3: pass 准则

- [ ] 4 份目标文档**每一处** V21 引用都被 strikethrough + blockquote 标记
- [ ] 4 份文档顶部 deprecation warning 已加
- [ ] 原文未被删除 (audit trail 保留)
- [ ] `git diff --stat` 显示 4 个文件改动, 无其他文件

⚠️ 不要扩大改动范围. 不要顺手"修复"V21 段落里其他错误 — 那是另一个 task.

---

## §3 — Task C: V13 smoke (200 step, ~20 min)

### C1: 前置确认

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull gitee foc_lite_hop0   # 确保最新

# 确认 V13 yaml 存在且未被修改
test -f review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml && echo "V13 yaml OK" || echo "MISSING"

# 确认 GPU 状态: slot 1 是 V18, slot 2 空闲
nvidia-smi  # 期望: 1 张卡跑 V18, 至少 1 张空闲
```

如果 V13 yaml 不存在或 slot 2 没有空 GPU, **停下报告**.

### C2: smoke 启动命令

```bash
SMOKE_DIR=review/0516/V13_true_image_aux_ablation/smoke_runs/smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SMOKE_DIR"

# 用 yaml override 跑 200 step (不修改原 yaml)
# 若 train_first_hop.py 不支持 --max-steps-override / --output-dir-override CLI, 
# 改为 cp 原 yaml + sed 改 max_steps + output_dir, 再 launch.
# 优先 CLI override, 因为更不易污染原 yaml.

# 选哪张空闲 GPU (假设 slot 2 = GPU 0, 实际看 nvidia-smi)
export CUDA_VISIBLE_DEVICES=<空闲 GPU id>

nohup python train_first_hop.py \
  --config review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml \
  --max-steps-override 200 \
  --output-dir-override "$SMOKE_DIR" \
  > "$SMOKE_DIR/smoke.log" 2>&1 &
SMOKE_PID=$!
echo $SMOKE_PID > "$SMOKE_DIR/smoke.pid"

echo "Smoke launched: PID=$SMOKE_PID, dir=$SMOKE_DIR"
```

### C3: smoke 监控 (~20 min, 阻塞等待)

```bash
# 等 smoke 完成 (200 step 应在 5-15 分钟内完成)
wait $SMOKE_PID
echo "Smoke exit code: $?"
```

### C4: smoke pass/fail 检查 (codex 必须全部满足才能进 §4)

```bash
LOG=$SMOKE_DIR/smoke.log

# 检查 1: 程序正常退出
test -f "$SMOKE_DIR/smoke.log" || { echo "FAIL: no log file"; exit 1; }

# 检查 2: image_aux.enabled = false (effective config print)
grep -E "image_aux.*enabled.*[Ff]alse" "$LOG" && echo "PASS: image_aux disabled" || echo "FAIL: image_aux not disabled"

# 检查 3: 200 step 内无任何 NaN
grep -i "nan\|inf" "$LOG" | grep -v "manifold" | grep -v "annotated" && echo "FAIL: NaN/Inf found" || echo "PASS: no NaN"

# 检查 4: loss_img 在所有 train step 上 = 0
# (从 log 抓 loss_img 列, 验证全为 0 或 'NA'/'-')
grep -oE "loss_img[=: ]*-?[0-9.eE+-]+" "$LOG" | head -50
# 期望: 所有数值 = 0.0 或字段不存在

# 检查 5: step throughput
# 计算 200 step 用时, 与 V7 baseline 比较 (V7 smoke ~5 min for 200 step)
grep "step=200" "$LOG" || grep "Steps:.*200" "$LOG"

# 检查 6: GPU memory 与 V7 baseline 相当
# 从 log 或 nvidia-smi history 抓峰值
```

### C5: smoke pass 准则 (全部满足)

- [ ] smoke 程序正常退出 (exit code 0)
- [ ] log 显示 `image_aux.enabled = false`
- [ ] 200 step 全程**无任何 NaN / Inf**
- [ ] log 中 `loss_img = 0.0` (所有出现处) 或字段被 skip
- [ ] 200 step 完成时间 ≤ V7 baseline 同 200 step 时间 × 1.1
- [ ] GPU memory peak ≤ V7 baseline × 1.1
- [ ] 无任何 RuntimeError / CUDA error in stderr

### C6: smoke fail 处理

任何一条 fail → **停下报告**, 不要 launch full V13. 写一份 `SMOKE_FAILURE_REPORT_<TS>.md` 给 user:
- 哪一条 fail
- log 关键片段 (~50 行)
- 怀疑的 root cause
- 不要尝试 fix, 等 user 决策

---

## §4 — Task D: V13 full launch (slot 2, ~7 天)

**前置**: §3 smoke 全部 pass.

### D1: launch 命令

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

LAUNCH_TS=$(date +%Y%m%d_%H%M%S)
LOG=review/0516/V13_true_image_aux_ablation/V13_train_${LAUNCH_TS}.log

# 使用 smoke 同张 GPU
export CUDA_VISIBLE_DEVICES=<同 §3 GPU id>

nohup python train_first_hop.py \
  --config review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml \
  > "$LOG" 2>&1 &
V13_PID=$!
echo $V13_PID > review/0516/V13_true_image_aux_ablation/V13_train.pid

echo "V13 launched: PID=$V13_PID, log=$LOG"
```

**注意**:
- V13 是 from-scratch (yaml 无 `resume_from`), **不需要 `--resume`**
- 不要传 `--resume`. 传了会污染 from-scratch 实验
- yaml 内 `training.max_steps: 160000` 是绝对值 (从 step 0 起), 与 V7 schedule lock

### D2: launch 后 ~5 min 验证

```bash
# 等 5 min, 检查训练真的在跑
sleep 300
ps -p $V13_PID > /dev/null && echo "PASS: V13 still running" || echo "FAIL: V13 died"

# 检查 log 显示 effective config 与 V18 不冲突
grep -i "image_aux.*enabled" "$LOG" | head -5
grep -i "max_steps" "$LOG" | head -3
grep -i "resume" "$LOG" | head -5    # 期望: "no resume" 或不出现

# 检查 GPU 占用
nvidia-smi
```

### D3: D pass 准则

- [ ] V13 process 启动 5 min 后仍存活
- [ ] log 显示 `image_aux.enabled = false`, `max_steps = 160000`
- [ ] log 显示 **未走 resume 路径** (e.g. 不出现 "Resuming from checkpoint" 之类)
- [ ] GPU 占用正常 (与 smoke 期间相当)
- [ ] 无 OOM / CUDA error

### D4: D fail 处理

任何一条 fail → kill V13 process, 写 `LAUNCH_FAILURE_REPORT_<TS>.md`, 等 user.

---

## §5 — Task E: 执行报告 + push gitee

### E1: 写执行报告

文件: `review/0517/PHASE_A_EXECUTION_REPORT_$(date +%Y%m%d_%H%M%S).md`

模板:

```markdown
# Phase A Execution Report

- date: <TS>
- branch: foc_lite_hop0
- HEAD before: <git rev-parse HEAD>
- HEAD after: <new HEAD>

## Task A — V18_design_rationale.md 修订
- §2.4 新增: DONE / FAIL (reason)
- §5.2 B12-B20 追加: DONE / FAIL
- §2.3 数字未改: VERIFIED / VIOLATED

## Task B — V21 retire 清理
- REVIEW_INTEGRATION_round4: <N 处标记>
- V18_EXECUTION_REPORT: <N 处标记>
- GAP_DECOMP_REPORT: <N 处标记>
- CODEX_RUNBOOK_V18: <N 处标记>
- 顶部 warning 已加: 4/4

## Task C — V13 smoke
- launch PID: <pid>
- smoke dir: <path>
- 6 个 pass 准则: <details, 全 PASS or 列 fail>
- smoke 用时: <min>
- 结论: PASS / FAIL (-> stopped, see SMOKE_FAILURE_REPORT)

## Task D — V13 full launch
- launch PID: <pid>
- launch log: <path>
- 5 个 pass 准则: <details>
- GPU 占用: slot 2 = GPU <id>
- 结论: PASS / FAIL

## Slot 占用状态 (push 时)
- Slot 1 (V18): step <current>, 健康
- Slot 2 (V13): step <current>, 健康
- Slot 3: 空

## Round 7 自警检查 (codex 本人是否复发偏差?)
- B14: 改 yaml 时是否 cross-check 联动? <YES/NO + 说明>
- B18: 是否做了 task 范围外的事 (例如准备阶段 B/C)? <NO 或说明>
- 其他: <any noticed>

## 下一步
- V18 step 200K 预计完成时间: <TS>
- 触发条件达成后, user 发起阶段 B task md
```

### E2: push gitee

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

git add review/0517/V18_decoder_lora/V18_design_rationale.md
git add review/0517/REVIEW_INTEGRATION_round4_20260517.md
git add review/0517/V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md
git add review/0517/V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md
git add review/0517/CODEX_RUNBOOK_V18_20260517.md
git add review/0517/PHASE_A_EXECUTION_REPORT_*.md
git add review/0516/V13_true_image_aux_ablation/V13_train_*.log
git add review/0516/V13_true_image_aux_ablation/V13_train.pid

# smoke run dir 不 commit 训练数据本身, 只 commit log:
git add review/0516/V13_true_image_aux_ablation/smoke_runs/smoke_*/smoke.log
git add review/0516/V13_true_image_aux_ablation/smoke_runs/smoke_*/smoke.pid

# 检查 git status, 确认没有意外文件
git status

# commit
git commit -m "Phase A: V21 retire + design_rationale §2.4/§5.2 update + V13 launch (slot 2)

- V18_design_rationale: 新增 §2.4 EV 修正 (区间 [0.05, 3] dB, 不改阈值) + §5.2 B12-B20 standing rules
- V21 retire: 4 文档加 [RETIRED 2026-05-17 Round 5: B8] 标记 + 顶部 deprecation warning, 不删原文
- V13 launch: smoke 200 step pass (image_aux=false, no NaN, no OOM), full 160K launched on slot 2
- 不动 V18 (slot 1 继续), slot 3 空
- 来源: NEXT_STAGE_ARCH_CODE_FINAL_20260517.md Phase A"

# push
git push gitee foc_lite_hop0
```

### E3: 完成通知

执行报告 push 后, 在 terminal 打印一行明显的完成标记:

```
========================================
PHASE A COMPLETE
- V18 (slot 1): step <X>, 继续
- V13 (slot 2): step <Y>, 健康
- slot 3: 空
- next: 等 V18 step 200K (~12-18h), user 发阶段 B task
========================================
```

---

## §6 — 显式 NOT-DO (codex 必须遵守)

不在 Phase A 范围内, **不允许**做:

- ❌ 停 V18 / 改 V18 yaml 任何字段
- ❌ launch V14 / V9 / V21 任何变体
- ❌ launch V18b 任何形态 (永久撤销)
- ❌ 准备 V18-clean yaml (推迟到阶段 C)
- ❌ V18 step 180K early-eval (Round 7 共识删除, 输出不影响任何决策)
- ❌ 起 Round 8 review (此 task 自己**等** Round 8 review, 不发起)
- ❌ 改 train_first_hop.py 任何代码 (本 task 不需要代码补丁)
- ❌ 加 LoRA→LoRA resume 支持 (V18-clean 走 V7→LoRA 路径, 不需要)
- ❌ 优化 / 重构 / 重命名 / 移动 任何代码或文件 (与 task 无关)
- ❌ 跑任何 eval (slot 1 V18 在用 GPU, 阶段 B 才 eval)
- ❌ 在 commit message 里写"也顺便做了 X"

任何 NOT-DO 项目被违反 → user 会 hard revert. 严格守界限.

---

## §7 — 失败模式快速决策树

```
Task A fail?
  → 停, 写 TASK_A_FAILURE.md, 不 push 任何东西, 等 user

Task B fail (例如 grep 不到 V21 引用)?
  → 报告实际情况, 询问是否文档已被前 task 部分清理, 等 user

Task C smoke NaN?
  → 不 launch full V13, 写 SMOKE_FAILURE_REPORT, slot 2 维持空, 等 user
  → 不要尝试 "再 smoke 一次试试"

Task C smoke OOM?
  → 报告 GPU memory peak, slot 2 维持空, 等 user 决定是否减 batch_size
  → 不要自行减 batch_size launch (污染配置)

Task D launch 5 min 后 dead?
  → kill 残留 process, 写 LAUNCH_FAILURE_REPORT, 等 user
  → 不要 "再 launch 一次"

Task D launch 后 OOM (V13 与 V18 抢同卡)?
  → kill V13, 报告. user 可能需要重新分配 GPU
  → 不要动 V18

Task E push 时 git conflict?
  → git fetch + git log 显示远程改动, 报告 conflict 文件, 等 user
  → 不要 git push --force
```

---

## §8 — 代码 audit 状态 (供 codex 参考, 不需要执行)

[CODEX_CODE_AUDIT_REQUEST_20260517.md](./CODEX_CODE_AUDIT_REQUEST_20260517.md) 提了 7 个 area 17 个代码问题. **Phase A 不依赖这些 audit 完成**:

- Q1 (resume CLI / yaml): 已通过 V18_TRAIN_COMMAND_20260517.txt 实证, V13 from-scratch 不需 resume
- Q2 (LoRA→LoRA resume): 阶段 A 不涉及, V18-clean 用 V7→LoRA 路径 (V18 已验证)
- Q3 (config mismatch): V13 不 resume, 不触发
- Q4 (KL pullback): V13 yaml `decoder_kl_pullback.enabled: false`, 不触发
- Q5 (image_aux path): V13 smoke 验证 `image_aux.enabled=false` 分支
- Q6 (LR/ckpt): 阶段 A 不涉及, 阶段 C 才需要 V18 best.pt criterion 确认
- Q7 (历史): 加速验证, 不阻塞

**但**: codex 若在 Phase A 执行中**顺手**发现 audit 答案 (e.g. smoke log 显示有用代码信息), 在执行报告 §6 (Round 7 自警) 里附一段, 不要另起 audit 回复文档.

---

## §9 — 总时长估计

| Task | 时长 |
|---|---|
| A 文档修订 | 10 min |
| B V21 retire 4 文档 | 15 min |
| C V13 smoke | ~20 min (含等待) |
| D V13 full launch + 5 min 验证 | 10 min |
| E 执行报告 + push | 10 min |
| **总** | **~65 min** |

GPU 占用增量: slot 2 从空 → V13 from-scratch (~7 天). Slot 1 (V18) 不动. Slot 3 不动 (空).
