# Peer Review Round 17-Slots — Stop Rule Revisit: A4-Light vs A4-Bracket

- date: 2026-05-22
- branch: foc_lite_hop0 (commit 1feaa8b)
- 主审对象: **是否打破 Round 17 stop rule "no relaunch A4-v2/v3"**, 把 1-slot A4-light 扩展为 2-3 slot 并行 image_aux bracket sweep
- 触发: user 观察 "此时只运行了一个训练? 我认为可以同时运行两个, 最多运行三个"
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 17 (战略) 已签 = Hybrid B + F0 + **A4-light** (1 个 image_aux schedule probe).
Round 17-Prep (codex execution) 已签.
Round 17-Stats (F0 statistical methodology) 已 push, 待评审中.

**本轮唯一审**:

> Round 17 stop rule "**无论 A4 outcome 不 relaunch A4-v2/v3**" (写在 §A4.5 + §6 NOT-DO #6) 是不是应该被 revisit; 应不应该**主动**起 2-3 个 image_aux schedule 变体并行跑形成完整 sweep curve.

**本轮不审**:
- Round 17 战略 (Hybrid B + F0 + A4 整体方向)
- F0 / A4 yaml / CLI 执行细节 (Round 17-Prep / codex EXEC_FIXES 已签)
- F0 paired-t 统计降级 (Round 17-Stats 独立审中)

---

## 1. 当前 stop rule 措辞 (本地 verify)

来自 [CODEX_TASK_ROUND17_F0_A4_20260522.md](./CODEX_TASK_ROUND17_F0_A4_20260522.md) §A4.5:

> "**无论 outcome, 不 relaunch A4-v2 / v3** (Round 17 stop rule, 防 scope creep B74)"

来自 §6 NOT-DO #6:

> "启 A4-v2 / A4-v3 / 其它 image_aux variant — Round 17 stop rule, 防 scope creep (B74)"

来自 §5 self-check anti-check:

> `anti_check "no extra A4 variant" bash -c "[[ \$(find review/0521 -name 'A4_image_aux_lambda_*.yaml' 2>/dev/null | wc -l) -gt 1 ]]"`

**stop rule 原本要防的是什么**: B74 = "Hybrid 可以悄悄变成 paper + 无限实验". 具体场景是 A4 跑完结果不好, claude 心理上还想再试 v2 v3 v4, 项目永远不进入写作期.

**stop rule 没有禁止的**: **预先**起 2-3 个 image_aux variant **并行**跑, 形成 sweep curve, 跑完就停.

---

## 2. 战略上下文 (本地 verify)

### 2.1 已知 image_aux 数据点

| run | image_aux λ | NORMAL PSNR_clip3 | 备注 |
|---|---:|---:|---|
| V13 | 0.0 | 36.4943 | image_aux 完全关 |
| V7 | 0.04 | 36.7810 | 项目主线, 单变量贡献 +0.287 dB vs V13 |
| (待) A4-low | 0.02 | ? | |
| (待) A4-mid (原 A4) | 0.08 | ? | |
| (待) A4-high | 0.12 | ? | |

### 2.2 V13/V14/V7 已经把 image_aux 锁定为单一最大贡献 (Round 17 X2)

V7 − V13 = +0.287 dB, 是项目 +0.099 dB 总变化的 ~3×. paper 主结果几乎必然要绕 image_aux 写.

### 2.3 候选 sweep 方案

#### 方案 A — A4-bracket 3 点 (3 slot 并行 7 天)

| slot | run | λ |
|---|---|---:|
| 1 | A4-low | 0.02 |
| 2 | A4-mid (原 A4) | 0.08 |
| 3 | A4-high | 0.12 |

5 个数据点 (V13 λ=0, A4-low λ=0.02, V7 λ=0.04, A4-mid λ=0.08, A4-high λ=0.12) 构成 paper 主图 response curve. 杀 stop rule.

#### 方案 B — A4-bracket 2 点 + V14b 多 seed (3 slot 并行 7 天)

| slot | run | 配置 |
|---|---|---|
| 1 | A4-low | image_aux λ=0.02 |
| 2 | A4-mid (原 A4) | image_aux λ=0.08 |
| 3 | V14b | V7 + seed=2024 |

A4 双向探针 + V14b 给 Round 17-Stats 的 noise floor 加第 3 个 seed 数据点.

#### 方案 C — A4-mid + V13b (2 slot, 留 1 slot 空 buffer)

| slot | run | 配置 |
|---|---|---|
| 1 | A4-mid (原 A4) | image_aux λ=0.08 |
| 2 | V13b | image_aux off + V7 train config + seed=1337 |

清算 V13-vs-V8 +0.02 dB 残差是 train config 还是 Grönwall step_weights.

#### 方案 D — 保持 Round 17 原计划 (1 slot A4-mid, 2 slot 空)

不动 stop rule, 守原 A4-light.

#### 方案 E — A4-bracket 2 点 (2 slot, 留 1 slot 空)

| slot | run | λ |
|---|---|---:|
| 1 | A4-low | 0.02 |
| 2 | A4-mid (原 A4) | 0.08 |

折中: 部分破 stop rule 但只到 2-slot, 不挑战 RAM.

### 2.4 已知硬约束

- user 明确: ≤ 3 并行训练, **3 个时会有内存争抢**
- 服务器历史: V18 单跑稳定, A3 + V13 + V14 三 slot 并行成功过 (Round 15 launch 验证)
- F0 不占 GPU 长时, 只 ~12 min 一次性占用 (V13/V14 per-slice eval), 不算 slot
- A4 / V14b / V13b 都是 from-scratch ~7d, 同 V7-class 资源占用

---

## 3. 给 reviewer 的 6 个问题

### Q1 — Round 17 stop rule 该不该 revisit?

stop rule 原文 = "无论 A4 outcome 不 relaunch A4-v2 / v3 (Round 17 stop rule, 防 scope creep B74)". 它防的是**串行**重试 (A4 出来不好就再起 v2 找 sweet spot). 它没明说**并行**多个 variant 一起跑也违规.

请评估:
- stop rule 字面是否禁止 §2.3 方案 A (3 点并行)?
- 是否字面允许但精神违背? 还是字面允许且精神也允许 (scope creep ≠ planned sweep)?
- user 现在主动 revisit, 是否应该接受 (slot 利用率) 还是拒绝 (stop rule 神圣性)?

### Q2 — image_aux sweep 真的有 EV 吗?

V7 (λ=0.04) 与 V13 (λ=0) 已经给了 image_aux 单变量 +0.287 dB. paper 主结果几乎已写好.

请评估:
- 加 A4-low/mid/high 3 个点跑出 response curve, 对 paper 收益**实际**多大?
- 还是 3 点 sweep 只是把 +0.287 dB 单点结果包装得更厚, 而 reviewer 已能从 V13 → V7 1 点斜率推 sweet spot 位置?
- 如果 sweep 出来 0.04 是真 sweet spot (A4-mid 0.08 < V7, A4-low 0.02 < V7), 这个 confirm 是否值得 21d GPU?
- 如果 sweep 出来 0.08 显著高于 V7 (A4-mid > V7 + 0.05 dB), 是否会触发 user 想再追 λ=0.06 / 0.10 / 0.16 — 形成 stop-rule 的真违规?

### Q3 — 4 方案 (A/B/C/D/E) 哪个主推?

请明确选 1:
- 方案 A: A4-bracket 3 点 (3 slot, 0.02/0.08/0.12)
- 方案 B: A4-bracket 2 点 + V14b (3 slot, 0.02/0.08 + seed=2024)
- 方案 C: A4-mid + V13b (2 slot, 0.08 + train config 清算)
- 方案 D: 守 Round 17 原计划 (1 slot, 0.08 only)
- 方案 E: A4-bracket 2 点 (2 slot, 0.02/0.08)
- 自定义 F (说明)

并给 1 个备选 + 拒绝其它的理由.

### Q4 — 3-slot RAM 争抢风险

user 原话: "3 个时会有内存争抢". Round 15 launch 时 A3 + V13 + V14 三 slot 并行成功. 但 image_aux λ 变体本身不改 model size, 所以理论上跟 V13/V14 同 footprint.

请评估:
- 是否应该 staggered launch (T=0 起 2 个, T+30min health check 后再起第 3 个)? 还是直接 3-slot 同时 launch?
- 是否应该限制每个 task 的 dataloader worker 数 (e.g. 从默认 8 降到 4) 防 RAM 争抢?
- 是否值得 user 自己先 SSH 上去看 RAM 实时再决定 launch 第 3 个? 还是 codex 自动决策?

### Q5 — sweep 顺序 / 优先级

如果选方案 A 或 E (A4-bracket):

- A4-low (λ=0.02) 和 A4-high (λ=0.12) 哪个 EV 更高? 哪个先 launch?
- 物理直觉: λ=0.04 已经是项目用了 ~1 年的设置, V7 选 0.04 不是偶然, 大概率是经验调出的 sweet spot. λ=0.02 测下方饱和, λ=0.08 测上方饱和, λ=0.12 测过量.
- 是否 λ=0.12 EV 最低 (大概率回归 V7 以下), 可以省掉只跑 0.02 + 0.08 (= 方案 E)?

### Q6 — 是否需要重起 Round 17-Slots 整合 + 更新 task md?

如果 reviewer 共识 ≥ 2/3 同意打破 stop rule:

- 是否需要正式 Round 17-Slots 整合文档, 标注 stop rule revisit + 标注新的 stop rule (e.g. "3 slot 跑完不再 v4/v5/v6, sweep 闭合")?
- task md 修改范围: §3 A4 section 全改为 A4-bracket, §A4.5 stop rule 改措辞, §5 anti-check 改为 "wc -l <= 3", §6 NOT-DO #6 改措辞
- 是否需要新 yaml 文件 (A4_image_aux_lambda_02.yaml / A4_image_aux_lambda_12.yaml)?

### Q7 — 是否引入新偏差 (B83+)?

请检查本 prompt 是否引入:
- **slot-utilization anchor**: "3 slot 不用就浪费" 是不是 sunk cost 同型偏差 (slot 空着是合理 buffer)?
- **sweep narrative anchor**: "3 点比 1 点强 10×" 是不是 paper-figure bias?
- **stop rule slippage**: 一次 revisit 是否会让未来 stop rule 都失去约束力?
- **A4-light 误读**: Round 17 reviewers 当时是不是已经隐含拒绝 sweep (claude 现在重新解释)?

---

## 4. 资料目录

### 4.1 本轮主审

- [CODEX_TASK_ROUND17_F0_A4_20260522.md](./CODEX_TASK_ROUND17_F0_A4_20260522.md) §A4.5 (stop rule) + §6 NOT-DO #6 + §5 anti-check
- [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md) (Round 17 4/4 共识 A4-light)

### 4.2 substrate (本轮不重审, 仅 verify image_aux 单变量贡献)

- [V13_V14_full_eval_analysis_20260521.md](../0516/V13_V14_full_eval_analysis_20260521.md)
- [PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md](./PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md) §1.3 (image_aux X2 = +0.287 dB)

### 4.3 历史 stop rule 设立先例

- [REVIEW_INTEGRATION_round15_20260518.md](../0517/REVIEW_INTEGRATION_round15_20260518.md) (Round 15 也面临 3-slot 决策, 用 staggered launch + IO health check 协议解决)
- [V18_design_rationale.md](../0517/V18_decoder_lora/V18_design_rationale.md) §5.2 (standing rules B43-B47 + B58 staggered launch)

---

## 5. 输出格式

每位 reviewer 独立产出 markdown:

### 5.1 6 个问题逐条 (APPROVE / MODIFY / REJECT)

### 5.2 主 verdict
请明确选 1 主 + 1 备:

- 方案 A: A4-bracket 3 点 (3 slot, 0.02/0.08/0.12)
- 方案 B: A4-bracket 2 点 + V14b (3 slot, 0.02/0.08 + seed=2024)
- 方案 C: A4-mid + V13b (2 slot, 0.08 + train config 清算)
- 方案 D: 守 Round 17 原计划 (1 slot, 0.08 only)
- 方案 E: A4-bracket 2 点 (2 slot, 0.02/0.08)

### 5.3 stop rule 措辞更新
若 ≥ 方案 C, 给出新 stop rule 一句话措辞 (写成 paper-ready 形式, 防未来 scope creep).

### 5.4 launch 协议
- 同时 launch / staggered / 看 RAM 实时决定?
- 是否限 dataloader workers?

### 5.5 新偏差 (B83+)

---

## 6. 角色提示

你不在审是否跑 image_aux sweep 这个研究方向 (它显然合理). 你在审一个**项目治理决策**:

> 当 user 提出 "slot 利用率不够" 这个直觉理由, 我们应该接受 (信号 = user 看着空 GPU 觉得浪费, 实质 = 把 A4-light 升级为 A4-sweep), 还是拒绝 (信号 = stop rule 一旦松动, 每轮都会有新理由再松), 还是部分接受 (2 slot 而非 3 slot, 留 buffer)?

这个决策直接影响:
- 项目 wall-clock 还是 7d, 但 paper 资产从 1 点变 2-3 点
- 未来类似 "slot 空着就用" 决策会不会成立例
- A4 stop rule 在 paper 写作期还守不守得住

请直接给观点, 不要因为 "稳妥" 列 4 个方案推给 user 决策疲劳.
