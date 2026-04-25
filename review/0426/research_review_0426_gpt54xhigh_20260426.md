# 0426 方案外部审查（GPT-5.4 xhigh）

## 元信息
- 审查时间: 2026-04-26
- 审查方式: `$research-review`（二级代理多轮）
- 审稿代理: `Hubble`（agent id: `019dc5c8-0e16-7393-a8cd-60bcdcb41707`）
- 审查对象:
  - `review/0426/v3_results_analysis_20260426.md`
  - `review/0426/v3p1_redo_experiment_plan_20260426.md`
  - `train_first_hop.py`（commit `591d34c`）

---

## Round 1: 关键批评（按严重性）

### S1. 旧 B/C（0425）结论不可用于方法 claim
- 观察: 两条 run 都从 `step=46400` resume 到 `50000`，有效新增仅 3600 step。
- 后果: 旧版 SF schedule 在 absolute 语义下直接进入 `alpha_sf=1.0` 区间，warmup/ramp 实际失效。
- 结论: `B vs C` 原比较污染，不能支持 SF 机制性结论。

### S2. 当前评估协议不满足“可主张因果差异”
- 问题: 训练内使用 rolling val window (`val_window_mode: rolling`, `max_val_batches:64`)。
- 后果: 不同 step 的 val 分数跨窗口比较噪声大，可能误判优劣。
- 要求: claim run 必须切到 fixed/full-val 协议。

### S3. 文档叙事与实际 LR 语义存在偏差
- 问题: 文档将 resume 后 warmup 近似理解为“相对起点”；代码是按 `step/max_steps` 绝对语义执行。
- 风险: 计划预期与实际优化过程不一致，导致实验解释偏差。

### S4. 0426 分析文档里有可被质疑的推理点
- 典型点: “B/C 梯度量级几乎相同”在日志上不稳固，存在显著量级差异的区段。
- 风险: rebuttal 时容易被抓住“结论先行”。

### S5. 控制变量仍有松动
- 例如旧 B/C 某些 watchdog 阈值不一致（虽为 warn-only）。
- 风险: “单轴对照”叙事被削弱。

---

## Round 1: 认可点

- `591d34c` 代码修复方向正确，关键修复有效:
  - SF 新增 `schedule_origin`，支持 `resume_relative`
  - 注入 `resume_start_step`
  - stdout 打印 `sf_alpha/sf_gap`
- 这三点确实直接对准了 0425 的污染根因。

---

## Round 2: 落地化执行建议（48h 版本）

### P0（必须先做）
1. 加硬性 preflight guard（CPU）:
   - `min_resume_steps` 不足直接 fail
   - `SF + absolute + start_step >= warmup+ramp` 直接 fail（除非显式 override）
   - 启动时打印 schedule audit（resume_step/effective_step/alpha 边界/LR 边界）
2. 配置差异白名单检查（B' vs C'）:
   - 仅允许预定义 key 变化，防止“隐式多变量”。

### P1（低成本 GPU 验证）
1. Smoke-A/B（各 +200 step）:
   - A: `resume_relative`，应看到 `sf_alpha` 按设计 0->1
   - B: `absolute`，应复现首步即高 alpha
2. 产出单页“调度通过证据”。

### P2（10k pilot，不是直接 50k）
1. 三臂短跑: `Base-continue`、`B'`、`C'`
2. 全程使用 claim-safe eval（fixed/full-val）
3. 10k 内达到门槛才升级到 +50k。

### P3（长跑条件）
- 仅对“pilot winner + 最强对照”长跑，避免再烧 2~3 GPU-day 无结论。

---

## Results-to-Claims Matrix（审稿版）

1. `PathA强` 且 `B' > C'` 且 `B' > Base` 且 `B' exposure_gap下降`
- 允许 claim: SF 机制在该设定下有效降低 exposure bias 并改善指标。
- 不允许 claim: 泛化到更广任务/数据。

2. `PathA强` 且 `C' >= B'` 且 `C' > Base`
- 允许 claim: rollout 权重策略有效。
- 不允许 claim: SF 是必要核心贡献。

3. `PathA强` 且 `B' ≈ C' > Base`
- 允许 claim: 两类干预都可能有效，但无法证明 SF 特异性优势。

4. `PathA强` 且 `B' ≈ C' ≈ Base`
- 允许 claim: 当前无足够证据支持机制增益。
- 不允许 claim: 方法有效。

5. `PathA强` 且 `B' < Base`（同时 SF 确认生效）
- 允许 claim: 当前 SF 配方在此 regime 下有害。
- 不允许 claim: exposure bias 不是瓶颈（该推断不成立）。

6. 200k ckpt 上 `PathA弱`（gap 明显塌缩）
- 允许 claim: 当前训练阶段 exposure hypothesis 不再成立或显著减弱。
- 不允许 claim: 继续用旧 50k 的 PathA 证据锚定新机制结论。

---

## 建议的最小代码改动清单（主干）

1. `train_first_hop.py`
- 增加 `training.min_resume_steps` 硬失败
- 增加 SF schedule 合法性硬失败
- 增加 startup schedule/LR 审计打印
- 增加 `training.claim_eval_mode`（`off/fixed/full`）并做协议检查

2. `scripts/guard_config_diff.py`（新文件）
- B' vs C' 配置白名单差异检查工具

---

## 48h Go/No-Go 判定树（建议阈值）

1. 预检不通过 -> 立即停止，不开跑。
2. Smoke-A/B 不符合预期 -> 立即停止，先修调度实现。
3. +1k pilot 退化 >35%（相对 Base） -> 停该臂。
4. +5k 时两臂都比 Base 差 >15% -> 全停，回调参/协议。
5. +10k 无一臂稳定优于 Base（连续两次） -> 不进入 +50k。
6. 进入 +50k 后，24h 内未达到“优于 Base 且 exposure_gap 实降” -> 在 +30k 提前止损。

---

## 最终审查结论

### 结论
- **0426 的代码修复（`591d34c`）是正确且必要的。**
- **0426 的实验设计仍需补齐“评估协议与预检护栏”，才具备可发表级 claim 安全性。**

### 现阶段可执行判定
- 可以进入 v3.1，但不建议直接开 50k 双长跑。
- 推荐严格执行: `preflight -> smoke -> 10k pilot -> winner long-run`。

---

## 需跟踪的残余风险

1. rolling val 与 fixed/full-val 口径混用导致的结论漂移。
2. resume 相关 schedule 在其他分支（非 SF）是否也有同类锚点歧义。
3. 文档叙事与代码语义不同步，导致组内复现实验时再次踩坑。
