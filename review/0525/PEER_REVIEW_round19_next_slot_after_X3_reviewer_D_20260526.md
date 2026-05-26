# PEER REVIEW (Reviewer D) — Round19 Next Slot After X3

- date: 2026-05-26
- reviewer: D
- scope: `PEER_REVIEW_PROMPT_round19_next_slot_after_X3_20260526.md`
- substrate checked:
  - `Round19-TODO.md`
  - `X3_FULLVAL_EVAL_REPORT_20260526.md`
  - `server/codex_x1_x3_design_analysis_20260525.md`

---

## 4.1 Q1-Q7 逐条回答

### Q1 — 空 slot 是否应该使用?
- decision: **APPROVE (use slot now)**
- rationale:
  - X3 已经完成并释放资源，且 X1-lite 仍需约 7 天；这段窗口可并行补齐“paper headline 的稳健性证据”。
  - 当前最薄弱点不是机制分解，而是主结论 A4-mid 仍是单 seed。这个风险会在审稿时直接被问到。

### Q2 — Option A (A4-mid-seed1337) 是否最高 EV?
- decision: **APPROVE**
- rationale:
  - 是最高 EV 选项。它直接服务主结论可信度，而不是扩展次线。
  - 已有 `V14` 对 `V7` 显示 seed 稳定，不能自动外推到 `A4-mid`，仍需直接测 `A4-mid` 的 seed 稳健性。
  - 该实验单 run、无外部依赖、解释清晰，最符合当前阶段目标。

### Q3 — Option B (X1-v2-balanced) 是否应提前跑?
- decision: **REJECT (not now)**
- rationale:
  - X1-v2 是条件性实验：只有当 X1-lite 落在 interior 区间时才有价值。
  - 现在提前跑，命中“无效运行”的概率高，且更易触发 scope creep / tuning 争议。
  - 按现有治理，先等待 X1-lite outcome 再判定更合理。

### Q4 — Option C (no new training) 是否过保守?
- decision: **MODIFY**
- rationale:
  - 作为“备选”可接受，但作为“主选”过保守：会错失一个完整 7 天窗口。
  - 建议把 C 作为 fallback（当算力不稳定或 paper 时间线突发时）。

### Q5 — Option D (X3-extend) 是否应直接排除?
- decision: **APPROVE (exclude)**
- rationale:
  - 与已签 stop rule 冲突，且高概率落入 V18 sunk-cost 复活。
  - 即便有增益，也很可能仍难超过 A4-mid，对当前主线贡献低。

### Q6 — 是否漏掉更好的 Option E?
- decision: **REJECT (no better E now)**
- rationale:
  - 在“单 run、无外部数据、不违反 no-small-tuning、不拖延 X1/paper”约束下，A 已接近 Pareto 最优。
  - 暂不建议新增 Option E，避免把决策从“执行”再拉回“发散”。

### Q7 — 偏差审查 (B97+)
- decision: **APPROVE (biases identified, manageable)**
- checks:
  - A4-mid seed-robustness fetish: 有风险，但当前场景属于“必要稳健性验证”，不是过拟合检查项。
  - CL1 overreaction: 当前拒绝提前跑 X1-v2，可控。
  - V18 sunk-cost: 通过排除 D 可控。
  - slot-utilization bias: 存在，但在本轮被“主线证据增益”正当化。

---

## 4.2 主 verdict (单选 + 备选)

- **Primary: A — A4-mid-seed1337**
- **Backup: C — No new training, wait for X1-lite**

Decision rule for backup switch:
- 若未来 24h 内出现资源/稳定性异常（GPU 频繁抢占、IO 不稳、训练失败重启频发），切换到 C。
- 否则执行 A。

---

## 4.3 If choose a run, minimal design spec

### Selected run: A4-mid-seed1337

Base: clone `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`

Minimal yaml diff (6 fields):
1. `output_dir` -> `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0526_runs/A4_image_aux_lambda_08_seed1337`
2. `run_name` -> `first_hop_224_a4_image_aux_lambda_08_seed1337`
3. `seed` -> `1337`
4. `training.resume_from` -> `""` (保持 from-scratch)
5. `training.max_steps` -> `160000` (与 A4-mid 对齐)
6. `training.save_interval` -> 保持与 A4-mid 一致（不改数值，仅明确“不得偏离”）

Hard stop rule:
- single-run only, no seed sweep (`seed!=1337` 禁止)
- no lambda sweep (`lambda_img` 必须保持 0.08)
- no LoRA/KL modifications
- 不得并发启动任何 X3/V18-family 续跑

Success criterion:
- 完成 canonical full-val (`max_slices=0`) on best+last
- 与 `A4-mid.best`、`V7.best` 给出 NORMAL PSNR delta

---

## 4.4 If choose no run, paper/task action

If fallback C is selected:
1. 立即启动 paper outline（主线写法锁定：image_aux 为主，V18/X3 为 secondary ablation）。
2. 写好 X1-lite outcome 的三段式模板（close/interior/near-V7）避免结果出来后临时叙事漂移。
3. 在 Methods/Limitations 预放 CL1 confound 文字框架（仅在 X1-lite interior 时启用）。

---

## 4.5 新偏差 (B97+)

- **B97: Headline fragility under single-seed optimism**
  - manifestation: 用 A4-mid 单 seed 直接当主结论强证据。
  - mitigation: 执行 Option A。

- **B98: Conditional-experiment pretrigger bias**
  - manifestation: 在条件未满足时提前启动 X1-v2。
  - mitigation: 严格等 X1-lite outcome 再判定。

- **B99: Governance drift via idle-slot rationalization**
  - manifestation: “有空位就跑点什么”。
  - mitigation: 只允许直接提升主结论稳健性的 run（A），禁止探索型加跑。

---

## Final Recommendation

执行 **Option A (A4-mid-seed1337)**，并把 **Option C** 作为稳定性 fallback。

这是当前约束下风险-收益比最高、且最能提升 paper 可发表性的 next-slot 决策。