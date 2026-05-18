# Round 7 Review Integration — NEXT_STAGE_EXPERIMENT_DESIGN 评审交叉核对

- date: 2026-05-17 深夜
- branch: foc_lite_hop0（NEXT_STAGE 仍未 push）
- 主审对象: [NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md](./NEXT_STAGE_EXPERIMENT_DESIGN_20260517.md)
- 三位独立 reviewer: agent1 / agent2 / agent3
- 硬约束: 服务器内存最多 3 个并行训练任务（Slot 1 = V18 ~step 180K/200K）

---

## 0. 一行结论

**三 reviewer 一致 REJECT dual-track 阈值表 + early-eval 三档判定**。但**新发现 4 个 Round 6 没识别的代码事实**，**完全改写 V18-clean 的成本/可行性**——这才是 Round 7 真正的产出。

| 维度 | agent1 | agent2 | agent3 | 共识 |
|---|---|---|---|---|
| Overall verdict | READY AFTER §3 修复 | MODIFY THEN PUSH | Option 1 (大幅简化) | **不可原样 push** ✅ |
| §1.1 dual-track 阈值表 | REJECT 措辞 (mashup) | MODIFY (PRIMARY 不动, SECONDARY 仅作 follow-up trigger) | **删** (三种诠释都失败) | **3/3 reject 当前形态** 🔴 |
| §2 step 180K early-eval | APPROVE 但弱化 (boolean) | MODIFY (metrics.jsonl 没 PSNR_clip3) | **删** (B4 同型 + 不影响决策) | **3/3 现形态不可用** 🔴 |
| §3 V13 launch | APPROVE | APPROVE (smoke 先) | APPROVE (Option 1 立即 launch) | **3/3 APPROVE** ✅ |
| §4.3 V18-clean 设计 | PARTIAL (缺 CLI + d_pure sanity) | APPROVE WITH FIX (40K 不是 7 天!) | 推迟 pre-register 到 V18 数据已知 | **设计 OK, 成本/时机 必修** ⚠️ |
| §6 timeline 算术 | BUG (Day 8/11 not Day 10) | (隐含同意) | (未提) | **必修** ⚠️ |
| §7 mid-draft self-catch | APPROVE | (未提) | APPROVE | **2/3 + 0 反对** ✅ |
| §8 6 个新问题 | (未直接评) | (回答了) | **删** (开 Round 7 违反"不再 review"教训) | **分歧** |

---

## 1. Round 7 最大产出: 4 个新代码事实（Round 6 没发现）

这 4 条**完全改写下一步可行性图**。三位 reviewer 中 agent2 独立验证全部 4 条，agent3 独立验证 #1，agent1 隐含确认 #4。

### 1.1 KL warmup 是 absolute global_step，V18 KL 已经满载 ~20K 步

代码 ([train_first_hop.py:2222-2228](PET_LatentResidual/train_first_hop.py)):

```python
lambda_kl = get_linear_schedule_value(
    global_step=step,              # ← absolute (V18 resume 后 step 从 160000 起)
    warmup_steps=0,                # ← 硬编码 0
    ramp_steps=int(kl_cfg.get("warmup_steps", 0)),   # ← yaml warmup_steps 其实是 ramp_steps
    start=0.0, end=lambda_kl_max,  # = 0.05
)
```

V18 从 step 160000 resume，`step=160001` 时 `(160001-0)/2000` clipped → **λ_kl 从 V18 第一步起就是满 0.05**。

**含义** (agent3 重大反转):
- Round 6 假设 "V18 KL warmup 期 + 后期" 区分 → **错误**，V18 全程满 KL
- 如果 B9 真有那么有害, V18 早就该崩 (实测 loss 健康) → **B9 实际影响很可能小**, 不是 hard ceiling
- 这弱化 (但不否定) V18-clean 的 EV

### 1.2 V18-clean from V7 best.pt 不是 7 天 / 200K，是 ~1.4 天 / 40K

agent2 通过读 [train_first_hop.py:1820-1900](PET_LatentResidual/train_first_hop.py) 确认: resume 逻辑保留 checkpoint 的 `start_step`。V7 best.pt 的 step ≈ 160000。V18-clean yaml `max_steps: 200000` → **实际只训 40000 step**, 与 V18 continuation 等成本, ~1.4 天而不是 7 天。

**含义** (huge):
- §6 timeline 算术彻底错: V18-clean Day 4 launch → Day 5.5 finish, 不是 Day 11
- slot 3 commitment 大幅降低 → V18-clean 几乎是"低成本必做"而不是"7 天昂贵 contingent"
- agent1 的 B15 (contingent slot inflation) 也因此弱化, 因为真实成本远低于他估的

### 1.3 metrics.jsonl 不记录 `val_chain_normal_psnr_clip3`

agent2 grep 确认: 训练器 metrics.jsonl 只有 `val_chain_normal_mse` / `val_multi_objective` / `val_select_score`, **PSNR_clip3 只在离线 `eval_first_hop_224_clip3.py` 算**。

**含义**:
- §2 早 eval 脚本 (read `val_chain_normal_psnr_clip3`) **不可执行**
- 0 GPU 路径只能读 MSE trend
- 若要 PSNR_clip3, 必须 GPU eval (~30 min 占 slot)

### 1.4 `training.resume_from` yaml 字段死字段, 必须 CLI `--resume`

agent1 + agent2 + agent3 都独立确认 (V18_TRAIN_COMMAND_20260517.txt: V18 用 CLI `--resume`):

```bash
python train_first_hop.py --config <yaml> --resume <V7 best.pt 绝对路径>
```

V18-clean launch 指令**必须**包含 `--resume`, NEXT_STAGE §4.3 缺这条 (B14 复发)。

---

## 2. 三 reviewer 完全一致的事实层结论

### 2.1 dual-track 阈值表三种诠释都失败 (agent3 最锐利)

| 诠释 | 后果 |
|---|---|
| 用 dual-track 做 SUCCESS 决策 | = B12 (mid-run goalpost shift) |
| 仅披露 (PRIMARY 唯一定义) | = 添加噪音, 决策无变化 |
| SECONDARY +0.05 单独 trigger | = 实际上变成 V18-clean launch 条件 → 需独立 pre-register |

**§7 自警印证**: claude 自己起草时一度把 SECONDARY 抬高过 PRIMARY → dual-track 在心理学上不稳定。

→ **3/3 共识: PRIMARY +0.30 / PARTIAL [+0.05, +0.30] / KILL +0.05 不动, §2.4 文字只描述 EV 区间 [0.05, 3] dB, 不引入第二列阈值, 不引入 SECONDARY**。

### 2.2 step 180K early-eval 三档判定都 launch V13 = 等价于 boolean = B4 同型

agent1 + agent3 独立指出: 三档判定 slot 2 列**全部 = V13 launch**, slot 3 列只在 "draft yaml" (0 GPU markdown 操作) 上有差别 → 早 eval 输出**不影响任何 launch 决策**。

agent2 加码: metrics.jsonl 根本没 PSNR_clip3, 脚本不可执行。

→ **3/3 共识: 删 §2 early-eval 三档判定**。

### 2.3 V13 立即 launch (slot 2, Day 1)

3/3 APPROVE。V13 outcome 与 V18 outcome 独立 (代码层验证: image_aux=false 分支跳 hop0 image aux forward, 不共享 image loss 路径)。

agent2 加: launch 前 200-step smoke 必做 (确认 `img=0`, 无 NaN, coverage/watchdog 正常)。

### 2.4 V18-clean 设计架构正确 (从 V7 出发, 1-bit diff)

3/3 APPROVE 架构。但**时机/成本/阈值**分歧 (见 §3)。

---

## 3. 三 reviewer 分歧 (需 user 仲裁)

### 3.1 V18-clean 何时 pre-register?

| reviewer | 推荐 |
|---|---|
| agent1 | 阶段 A pre-register + 加 step 20K early-kill gate + 4-tp 一致性 sanity |
| agent2 | 阶段 A pre-register, 但 threshold 4 档 (< +0.02 / +0.02-+0.08 / +0.08-+0.15 / ≥ +0.15) 替代单点 +0.05 |
| agent3 | **推迟到阶段 C** (V18 200K eval 结果出来后再 pre-register) |

**核心分歧**: 提前 pre-register 是 commit (agent1/2 视角 = 增加 discipline) 还是 anchor (agent3 视角 = 限制未来 flexibility)？

考虑到 §1.2 新事实 (V18-clean 实际 ~1.4 天不是 7 天), commitment 风险大幅降低 → **agent1/agent2 路径成本可接受**。但 agent3 "等数据来了再 pre-register" 在方法论上也站得住 (避免对每个 outcome branch 预先设计响应)。

**claude 倾向 agent3 + 部分采纳 agent2 的 4 档 threshold**: 阶段 A 仅写"V18-clean 是 contingent 候选, threshold 在 launch 时 pre-register"; 阶段 C 真正 launch 时, 用 agent2 的 4 档 threshold (< +0.02 negligible / +0.02-+0.08 inconclusive / +0.08-+0.15 helpful / ≥ +0.15 strong)。

### 3.2 阈值表新增 SECONDARY 列?

- agent1: 加 SECONDARY +0.05 作为 follow-up trigger (信号列, 不参与 SUCCESS)
- agent2: 同 agent1
- agent3: **不加** (SECONDARY 在 PARTIAL 区间已经 implicit triggered, 加这一列是冗余且诱发 B12)

agent3 的论证: PARTIAL [+0.05, +0.30] **本身就触发**"写 interpretation note + 评估是否 follow-up", 不需要额外 SECONDARY +0.05。SECONDARY 在 PARTIAL 区间一定 TRIGGERED, 在 KILL 区间一定 NOT, 在 SUCCESS 区间一定 TRIGGERED → 0 额外信息。

**claude 倾向 agent3**: 不加 SECONDARY 列, §2.4 改用"PARTIAL outcome 自动触发 follow-up 评估"措辞。

### 3.3 §8 6 个新问题 (= 启动 Round 8) 该写吗?

- agent1: 未直接评 (默认 OK)
- agent2: 回答了 (默认 OK)
- agent3: **删** (违反"不再开新 review"Round 4 教训 + 6 轮已穷尽设计空间)

**claude 倾向 agent3**: 删 §8。Round 7 后下一步是**执行**, 不是再起 Round 8。如果阶段 C 真要 launch V18-clean, 那时可起一份很短的 V18-clean launch checklist review (10 分钟), 不需要全 6 问。

### 3.4 是否加 KL drift 测量 (agent3 独家提议)

agent3 §阶段 B2: 在 V18 200K eval 时同步测 `PSNR(decode_V18(z_GT), x_target)` (重用 GAP_DECOMP probe 改 50 行)。这给 "B9 实际影响多大" 提供 *数据* 依据, 而不是猜测。

agent1 + agent2 未提。但这条**含金量极高** (~30 min 实施, 直接决定 V18-clean 是否值得)。

**claude 强烈推荐采纳**。

---

## 4. 新偏差 (Round 7 识别)

继 Round 6 B11-B14 之后:

| # | 名称 | 来源 | 形态 |
|---|---|---|---|
| **B15** | contingent slot inflation | agent1 | 把 "contingent" 当 "0% 默认", 隐藏真实承诺概率 |
| **B16** | independence 框架的 narrow scope | agent1 | "V13/V18 outcome 独立" 在测量层成立, 在项目优先级层不独立 |
| **B17** | paired comparison threshold 不查自身 noise floor | agent1 | 用 V7-V6_NOISE 的 same-architecture noise 估 V18 vs V18-clean (不同 use_pred_latent) 的 paired noise |
| **B18 (NEW)** | "pre-design every branch" anti-pattern | agent3 | 在数据出来前为每个 outcome 设计响应分支, 与 "wait, evaluate, decide" 对立 |
| **B19 (NEW)** | yaml 字段语义混淆 | agent2 + agent3 | `warmup_steps` 在 KL config 里实际是 `ramp_steps`, 命名误导 |
| **B20 (NEW)** | resume 后 max_steps 含义混淆 | agent2 | yaml `max_steps: 200000` 在 V7-resume 上下文是"训到绝对 step 200000"不是"训 200K 个新 step", 文档 §4.3 默认理解错 |

B18 + B20 是最严重的:
- B18 是**方法论 meta-bias**: 整份 NEXT_STAGE 都在 pre-design V18 outcome 的所有响应分支, 而非等 12 小时让数据自己回答
- B20 是**事实理解错**: 直接导致 V18-clean 成本估算错 5×

---

## 5. 推荐最终修订路径

综合三 reviewer + Round 7 新事实, 推荐如下（基本采纳 agent3 Option 1 + 补 agent2 代码事实 + 部分采纳 agent1 d_pure sanity）:

### 阶段 A — 今天 (0 GPU, ≤30 min codex)

```
A1. V18_design_rationale.md 修订:
    §2.3 阈值表: 完全不动 (原 +0.30 / [+0.05, +0.30] / +0.05 三档)
    §2.4 新增 EV 修正记录:
      - B10/B11 文档化 (11.2 dB 高估, 0.5 dB 低估, 真实 [0.05, 3] dB 区间)
      - 明确不引入 dual-track 不引入 SECONDARY 列
      - 评估时同时报告: (a) 三档分类 by ORIGINAL thresholds, (b) 在 [0.05, 3] dB 区间的位置 (纯描述)
    §5.2 自警条目加 B12/B13/B14/B18/B19/B20:
      B12 — 禁止以 "baseline 错了" mid-run 改阈值
      B13 — 禁止 sunk-cost-resume 设计
      B14 — yaml 改字段必须 cross-check 联动 (LR / resume CLI / config mismatch)
      B18 — 禁止 pre-design every branch; 等数据来再设计响应
      B19 — KL config "warmup_steps" 实际是 ramp_steps, 在文档每处出现都加 [实际=ramp_steps] 注释
      B20 — resume 后 max_steps 是绝对值, 不是新训步数; 任何 resume 实验必须显式注 "实际训练 N step"

A2. V21 retire 4 文档加 [RETIRED 2026-05-17 Round 5: B8] 标记 (不删原文)

A3. V13 launch (slot 2):
    A3.1 服务器 smoke 200 step 确认 image_aux=false 分支 (img=0, 无 NaN, coverage OK)
    A3.2 nohup + tmux launch full 160K
    A3.3 V13 是 from-scratch (yaml 无 resume_from), 不需 --resume

push gitee。
```

### 阶段 A 显式 NOT-DO

- ❌ dual-track 阈值表 (3/3 reject)
- ❌ SECONDARY +0.05 列 (agent3 论证冗余)
- ❌ step 180K early-eval 三档判定 (B4 同型 + 脚本不可执行)
- ❌ V18-clean pre-register (B18; 推迟到阶段 C)
- ❌ V18-clean yaml draft (推迟)
- ❌ §8 6 个 reviewer 问题 (= Round 8, 违反 "不再 review" 教训)
- ❌ 改 V18 yaml 任何字段

### 阶段 B — Day +1~+2 (V18 200K 完成, GPU 释放)

```
B1. V18 200K full-val eval (eval_first_hop_224_clip3.py, ~30 min, slot 1 已空):
    - 与 V7 best.pt 同 split, NORMAL chain PSNR_clip3
    - 写 V18_FINAL_EVAL_<TS>.md
    - 报告 SUCCESS / PARTIAL / KILL by ORIGINAL thresholds (无 dual-track)
    - 同时报告 ΔPSNR 在 [0.05, 3] dB 区间位置 (纯描述)

B2. KL drift 测量 (~30 min, 重用 GAP_DECOMP probe):
    psnr_v18_at_zgt = PSNR(decode_V18(z_GT), x_target)
    与 V7 baseline 同度量比较:
      - drift < 0.5 dB → KL pullback 没显著拉走 decoder (B9 影响小)
      - drift > 1.0 dB → KL 显著拉走 decoder → B9 是 V18 上限 ceiling

资源: B1+B2 在 V18 释放后的 slot 1 上, V13 在 slot 2, slot 3 空 = 3 slot 内
```

### 阶段 C — Day +2~+3 (4 行决策, 写在 V18_FINAL_EVAL.md 末尾)

```
case V18 SUCCESS (ΔPSNR ≥ +0.30, in EV interval "high"):
   → slot 3 = null. Start paper draft. V18 family 主路径成立.

case V18 PARTIAL ([+0.05, +0.30]):
   Look at B2 KL drift:
     - drift 显著 (>1.0 dB) → B9 is the cause → launch V18-clean on slot 3
       pre-register at C 时刻 (不是阶段 A), 用 agent2 4 档:
         < +0.02 dB = B9 影响可忽略
         [+0.02, +0.08) = inconclusive (信噪比不足)
         [+0.08, +0.15) = B9 fix materially helpful
         ≥ +0.15 dB = B9 fix strongly effective
       同时报告 4-tp 一致性 (NORMAL/D20/D10/D4 同方向 ✓ = high confidence)
     - drift 不显著 → B9 不是 bottleneck → slot 3 = backbone/data 方向 phase plan
       V18 sweep rank/blocks 不再尝试 (已知 EV 区间内 PARTIAL 是上限)

case V18 KILL (ΔPSNR < +0.05):
   → slot 3 = backbone/data/architecture 方向. V18 family retire.
   V18-clean 不做 (B9 fix 救不了 KILL).
```

### V18-clean (only if triggered in C)

```
起点: V7 best.pt (同 V18, 干净 paired ablation)
yaml diff vs V18:
  - use_pred_latent: true → false
  - run_name / output_dir 改
  - warmup_steps 保持 2000 (但实际不会重新 ramp, 因 KL schedule global_step)
  - 其他全部不动 (rank=32, blocks=[6,7], λ_kl=0.05, λ_img=0.04, schedule, seed)
launch CLI (必须显式):
  python train_first_hop.py \
    --config review/0517/V18_clean_kl_gt_anchor/V18_clean_kl_gt_anchor.yaml \
    --resume /data_2/qujiaxiang/.../V7/run/.../best.pt
真实成本: ~1.4 天 / 40K step (不是 7 天 / 200K!), 因 V7 best.pt start_step=160000
评估: V18-clean step_200000/last.pt vs V18 step_200000/last.pt (matched endpoint)
     secondary: best.pt vs best.pt
```

---

## 6. user 决策点汇总 (请回复)

| # | 决策 | claude 推荐 |
|---|---|---|
| 1 | 接受阶段 A NOT-DO 全集 (dual-track / SECONDARY / early-eval / V18-clean pre-reg / §8 / 改 V18 yaml 都不做) | **接受全集** |
| 2 | V13 阶段 A 立即 launch / 等 V18 200K 后再 launch | **立即 launch (agent3 Option 1)** |
| 3 | KL drift 测量 (B2) 阶段 B 加入 / 不加 | **加入 (agent3 独家提议, 高 EV)** |
| 4 | V18-clean pre-register 时机: 阶段 A / 阶段 C | **阶段 C** (B18 教训) |
| 5 | V18-clean threshold: 单点 +0.05 / agent2 4 档 | **agent2 4 档** |
| 6 | V13 launch 前 smoke 200 step | **必做** (agent2 提议) |

回复格式: 如 `1接 2A 3加 4C 5四档 6做` 或全接受推荐 = `全推荐`。

---

## 7. 总体评估: Round 7 的元教训

**Round 6 → Round 7 增量**: 4 个新代码事实 (KL global_step / V18-clean 40K / metrics.jsonl 无 PSNR_clip3 / resume CLI), 3 个新偏差 (B18 pre-design every branch / B19 yaml 命名误导 / B20 resume max_steps 混淆)。

**Round 7 比 Round 1-6 都更技术性**: 主要发现来自 reading actual code (agent2 ~1500 行 + agent3 选段验证), 而不是概念审查。这是好信号——意味着设计层面争议趋近收敛, 剩下的是 "执行细节是否对" 而非 "做什么对"。

**最大 actionable insight**: V18-clean **不是 7 天 commitment, 是 1.4 天**。这单条事实让 slot 3 routing 从 "高风险 contingent" 变成 "低成本必做 if PARTIAL"。整个 NEXT_STAGE §4.2 表的风险/收益结构因此完全改写。

**B18 是 Round 1-7 最深的方法论 bias**: 我们花 7 轮 review 在 pre-design V18 outcome 的响应分支, 而 V18 数据 12 小时后就出来了。**正确做法是阶段 A 只做不依赖 V18 数字的事 (文档清理 + V21 retire + V13 launch), 等数据来了再设计阶段 C 响应**。这条教训应固化到 design rationale §5.2 作为 standing rule。
