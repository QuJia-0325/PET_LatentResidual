# Peer Review Round 15 — 3-Slot Launch Decision (A3 + V13 + V14)

- date: 2026-05-18
- branch: foc_lite_hop0 (commit 24de2ab pushed)
- 主审对象: **3-slot launch 决策 (选项 B)**
- 上游: Round 13 (A1/A3/A4 user 决策) + Round 14 (A3 task md 修订完成, push 完成) + user 当前决策 = "slot 3 = V14 multi-seed (机会主义)"
- target reviewers: 2-3 个 AI 独立评审, **不互见草稿**

---

## 0. 本轮范围与边界

Round 14 完成 A3 execution prep. **Round 15 是 launch order + slot allocation review**:
- 不再质疑 A3 / V13 是否该跑 (Round 13 user 签 A1/A3, Round 14 verify)
- 不再质疑 yaml / task md (Round 14 reviewer 共识 ready)
- 只审 **slot 3 加跑 V14 是否真是机会主义最优**

**本轮审**:
- 3-slot 并行 launch 是否真无 GPU / 内存 / dataloader 冲突
- V14 (multi-seed d_pure) 加跑的 ROI 评估 (paper value vs 7 天 GPU)
- launch order (谁先谁后) + dependency
- 是否漏关键 control 实验 (e.g. V13-r32-capacity-only / V13-no-image-aux-no-step-weights)
- 是否引入 Round 15 新偏差 (B55+)

**本轮不审**:
- A3 / V13 / V14 yaml 设计 (各 round 已 signed)
- Round 13/14 standing rules (已 push 进 design_rationale §5.2)
- audit DRAFT release

---

## 1. 关键背景 (本地 verify)

### 1.1 当前 GPU slot 状态

| slot | 当前 | 容量 |
|---|---|---|
| 1 | 空 (V18 已完成 200K) | 24h × 全 GPU |
| 2 | 空 (V13 还未 launch, task md 在 commit 7486776 ready 但 codex 未执行) | 24h × 全 GPU |
| 3 | 空 | 24h × 全 GPU |

硬约束: 服务器内存最多 3 个并行训练任务. 当前 0/3, 全空闲.

### 1.2 3 个候选实验

| 实验 | yaml | task md | 训练时长 | 预期产出 |
|---|---|---|---|---|
| **A3 V18-capacity-only** | [V18_capacity_only.yaml](PET_LatentResidual/review/0517/V18_capacity_only/V18_capacity_only.yaml) | [CODEX_TASK_STAGE_C_A3](PET_LatentResidual/review/0517/CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md) | ~24-48 h (V7 best + 10K) | KL drift 候选 B vs A 间接 vs 4 disambig |
| **V13 image_aux ablation** | [V13_true_image_aux_off.yaml](PET_LatentResidual/review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) | [CODEX_TASK_PHASE_A_v3](PET_LatentResidual/review/0517/CODEX_TASK_PHASE_A_v3_20260518.md) Task C+D | ~7 天 (160K from-scratch) | V7-V13 = 真 image_aux 单变量贡献 |
| **V14 multi-seed d_pure** | [V14_v7_seed1337.yaml](PET_LatentResidual/review/0516/V14_true_d_pure/V14_v7_seed1337.yaml) | **无 task md** (只有 yaml) | ~7 天 (160K from-scratch) | d_pure noise upper bound (paper 阈值判定必备) |

### 1.3 历史数字 (用作判定 ROI)

| 累积 ΔPSNR | 值 |
|---|---:|
| V6 → V18.last (项目 1 年累积) | +0.099 dB |
| V18 vs V7 best-vs-best NORMAL | +0.030 dB |
| V18 vs V7 last-vs-best NORMAL | +0.062 dB |
| V18 GT manifold (decode(z_GT)) drift | +0.10 dB (signed) |

### 1.4 V14 ROI 评估 (claude 推荐 B 的论据)

**支持 V14 加跑 (B 选项)**:
- slot 3 不跑 = 7 天 GPU 时间机会成本浪费 (V18/V13 在 slot 1/2 跑, 不互斥)
- V14 yaml 已 ready (review/0516/V14_true_d_pure/V14_v7_seed1337.yaml), 0 prep 成本
- d_pure noise floor (V7-V14 NORMAL Δ) 是 paper 报告 ΔPSNR 时的必备 noise upper bound — 没有它, V18 +0.030 dB 算不算 statistically significant 都没参考
- 与 A3 / V13 完全独立 (V14 是 V7 + seed=1337, 不动 image_aux, 不动 V18 LoRA), 不互相干扰
- 7 天后 V14 与 V13 同时完成, 一个 paper write-up window

**反对 V14 加跑 (A 选项)**:
- user Round 13 明确说 "不考虑 multi-seed, 先跑出结果, 再考虑 multi-seed" — 严格守这个决策
- A3 48h 回来后, slot 3 可能有更优 follow-up (e.g. capacity-only sweep 或 transport intervention), 灵活
- 当前 capacity-only outcome 未知, V14 数据现在拿到也可能因 V18 framework 整体被 retire 而无 paper 价值
- 7 天 V14 跑完, paper narrative 可能因 capacity-only 改向, V14 数据可能 stale

---

## 2. 给 reviewer 的 6 个问题

### Q1 — V14 加跑 (slot 3) 真的 ROI 高吗?

请评估 §1.4 两组论据, 给主推 + 反对理由:
- d_pure noise floor 真的 "paper 必备" 还是 "nice to have"?
- 7 天 GPU 机会成本 vs 等 A3 outcome 后 slot 3 灵活 — 哪个更优?
- user "不跑 multi-seed" 决策是绝对 (不可推翻) 还是 contextual (基于 Round 13 时点信息, 现在可重审)?

### Q2 — 3 slot 真无并发冲突?

3 个训练任务同时 launch:
- A3: V7 best.pt resume + LoRA (small model + small dataloader overhead)
- V13: from-scratch, Plan F config (full latent transport pipeline)
- V14: from-scratch, V7 config (与 V13 同 pipeline 但不同 seed)

请评估:
- 服务器 GPU 内存是否真够 3 个独立训练 (每个 V18-scale ~10-20 GB)?
- dataloader 路径是否真无冲突 (V13/V14 同读 lowdose_pet_ct latents_224, 同 dataloader 多进程)?
- watchdog / metrics jsonl 文件是否互不污染 (各 output_dir 独立, 应 OK)?
- 历史: V18 在 slot 1 跑时, 服务器是否 ever 同时跑 ≥ 3 任务? 若没有, 是否有未知冲突风险?

### Q3 — Launch order 重要吗?

3 个实验都可立即 launch, 但 A3 48h 最先回, V13/V14 都 7 天.

- 应该全部同时 launch (节省 wall clock, 浪费 slot 1 后 48h)?
- 还是 A3 先 launch, 等 48h 出结果再决定 V13/V14 是否仍想跑?
- 或 A3 先 launch + V13 同时 (按 user 原 Round 13 决策 = A1+A3 同时), V14 等 A3 outcome 后决定?

### Q4 — V14 缺 task md, 是否阻塞?

V14 yaml ready, **但**无 task md (V13 在 commit 7486776 有 Phase A v3 task md, A3 有专属 task md, V14 无).

请评估:
- codex 能否 just `python train_first_hop.py --config V14_v7_seed1337.yaml` 直接跑, 不需 task md?
- 还是 user 应 prep V14 launch task md (~10 min), 含 pass 准则 + NOT-DO + commit 流程?
- 不 prep task md 会引入 B14 (yaml cross-check) / B23 (premature claim) 等历史偏差风险?

### Q5 — 是否漏关键 control 实验?

3-slot 跑 A3 + V13 + V14 后, 我们能 disambig:
- V18 +0.10 GT drift = capacity vs KL 间接 (A3 答)
- V7-V8 +0.308 dB 中 image_aux 单变量贡献 (V13 答)
- d_pure noise floor (V14 答)

请评估是否漏关键控制:
- V13-capacity-only? (V13 + LoRA rank=32 + no KL, 验证 V13 在 V18-like decoder 下是否 +0.308 仍成立)
- V8 重训 with V7 train config (即 step_weights=V6 + image_aux off + V7 train config, 分离 train config 与 step_weights 联合 effect)
- V18-clean (use_pred_latent=false) 仍未跑, Round 6/7 永久撤销但 Round 13 KL drift inverse 后是否值得复活?
- V19 (新 decoder LoRA 变体, e.g. blocks=[0,1] vs [6,7], 测 decoder 哪层最有用)?

### Q6 — Round 15 新偏差识别 (B55+)

claude 当前 plan (3 slot launch = A3 + V13 + V14) 是否引入新 confirmation bias:
- 推 V14 是 "slot 3 不浪费" 的 sunk cost framing? (V14 7 天 GPU 也是 sunk, 应独立判 ROI 而非 "反正空着")
- "d_pure noise floor paper 必备" 是 anchor 到 paper 路径 (V18 framework 还没 confirm 值得 paper)?
- 3 slot 全 launch 是 "filling capacity" bias? (slot 3 空也是合理选择, 留 buffer 等 capacity-only outcome 决定下一步)
- user 原 Round 13 决策 "不跑 multi-seed" 推翻太快 (claude 改决策是基于 "slot 3 空" 这一新信息, 但这信息在 Round 13 决策时也已存在, 真新信息只有 KL drift inverse — 它不直接影响 V14 ROI)

---

## 3. 资料目录

### 3.1 本轮主审对象 (3 yaml + 1 task md gap)

- [V18_capacity_only.yaml](PET_LatentResidual/review/0517/V18_capacity_only/V18_capacity_only.yaml) (A3 实验)
- [V13_true_image_aux_off.yaml](PET_LatentResidual/review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) (V13)
- [V14_v7_seed1337.yaml](PET_LatentResidual/review/0516/V14_true_d_pure/V14_v7_seed1337.yaml) (V14, **无 task md**)
- [CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md](PET_LatentResidual/review/0517/CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md) (A3 task)
- [CODEX_TASK_PHASE_A_v3_20260518.md](PET_LatentResidual/review/0517/CODEX_TASK_PHASE_A_v3_20260518.md) Task C+D (V13 launch 部分)

### 3.2 上游 context (本轮不审)

- [REVIEW_INTEGRATION_round13_20260518.md](PET_LatentResidual/review/0517/REVIEW_INTEGRATION_round13_20260518.md) §3.3 (3 slot 并行计划)
- [REVIEW_INTEGRATION_round14_20260518.md](PET_LatentResidual/review/0517/REVIEW_INTEGRATION_round14_20260518.md) (A3 task 修订 完成)
- [V18_design_rationale.md §5.2](PET_LatentResidual/review/0517/V18_decoder_lora/V18_design_rationale.md) B43-B47 standing rules

### 3.3 历史数据 (Q1/Q5 ROI 评估用)

- V6→V18 累积 +0.099 dB (项目 1 年优化总量)
- V18 vs V7 NORMAL ΔPSNR +0.030/+0.062 dB
- KL drift inverse +0.10 dB GT manifold
- V13 expected: V7-V13 NORMAL Δ ≈ ? (V7-V8 +0.308 dB 中 image_aux 贡献待估)
- V14 expected: V7-V14 NORMAL Δ ≈ ±0.01-0.03 dB (d_pure noise)

---

## 4. 输出格式

每位 reviewer 独立产出 markdown:

### 4.1 6 个问题逐条 (APPROVE / MODIFY / REJECT)

### 4.2 slot 3 决策 verdict
- **A**: slot 3 空, 守 user 原决策
- **B**: slot 3 = V14, claude 推荐
- **C**: slot 3 = 其他实验 (具体说明)

### 4.3 Launch order verdict
- 全部同时 launch
- A3 先, V13/V14 等 48h 后再 launch
- A3 + V13 先, V14 等 capacity-only outcome 后再决定

### 4.4 V14 task md 是否需要 prep
- 需要 (~10 min, 含 pass 准则)
- 不需要 (codex 直接 `python --config` 跑 V14)
- 重用 Phase A v3 task md 模板套 V14

### 4.5 新偏差 (B55+)
若发现 claude 推 B 是 sunk cost / paper anchor / filling capacity bias.

---

## 5. 约束与提醒

- **本轮不重审** A3 / V13 / V14 yaml 设计
- **不假设** 你能跑代码 / 触 GPU — 凭 git artifacts + yaml 评
- **特别关注**:
  - Q1 V14 ROI 是真 paper-necessary 还是 anchor
  - Q2 3 任务并发风险 (服务器历史是否跑过 3 任务?)
  - Q5 是否漏关键 control (V13-cap-only / V18-clean 等)
  - Q6 claude 推 B 是否 sunk cost / anchor

---

## 6. Standing constraints

1. ≤ 3 并行训练任务 (本 task 完全用满 3 slot 是 B 选项, A 选项用 2)
2. V18 不动 (已完成)
3. V18b 永久撤销
4. 阈值不改
5. user 决策 (R7/R8/R9/R10/R12/R13/R14) 不可推翻 — **但 R13 "不跑 multi-seed" 是 contextual 还是 absolute? 待 reviewer 评 (Q1)**
6. Round 15 是 launch order review, 非战略
7. 等 capacity-only outcome (~48h) 后必起 Round 16 战略 review (含 V18 framework retire / paper draft 决策)

---

## 7. 本轮元说明

| 轮次 | 范围 | substrate |
|---|---|---|
| 1-7 | design | spec |
| 8-11 | execution doc | markdown |
| 12 | V18 PSNR substrate | 数字 |
| 13 | strategy (KL drift inverse 触发) | 历史 PSNR |
| 14 | execution prep (A3 + A4) | yaml + task md |
| **15** | **launch order + slot allocation** | **GPU slot + 实验 ROI 评估** |

预期 Round 15 outcome:
- 最佳: 3/3 共识 B (3 slot 全 launch + V14 task md prep) → claude 起 V14 launch task md + push + codex 3 slot 跑
- 中等: 3/3 共识 A (slot 3 留空) → 只 launch A3 + V13, slot 3 等
- 最差: reviewer 找到关键 control (V13-cap-only / V18-clean) ROI > V14, slot 3 改跑其他
