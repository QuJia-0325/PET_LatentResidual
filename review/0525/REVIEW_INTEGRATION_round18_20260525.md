# REVIEW INTEGRATION — Round 18 (Post-A4-Bracket Strategic Decision)

- date: 2026-05-25
- scope: integrate 4 independent Round 18 strategic reviews
- sources:
  - PEER_REVIEW_ROUND18_STRATEGIC_NEXT_STEP_REVIEWER_A_20260525.md (Reviewer A)
  - PEER_REVIEW_ROUND18_REVIEWER_B_20260525.md (Reviewer B)
  - PEER_REVIEW_round18_strategic_next_step_reviewer_C_20260525.md (Reviewer C)
  - PEER_REVIEW_round18_reviewer_D_20260525.md (Reviewer D)

---

## 1) Strong consensus (4/4)

### 1.1 V18 降为次要 ablation (4/4 选 option b)

V18 整套 decoder LoRA + KL 工程在 paper 里的角色全部下降。A4-mid (+0.113 dB) ≈ 2× V18.last (+0.062 dB). image_aux schedule 必然是 paper 头条, V18 写成 "we also tried decoder LoRA but it gave smaller gains than image_aux schedule adjustment". 不再以 V18 为主线推任何新长跑.

### 1.2 image_aux 机制 = M1 + M4 联合主推 (4/4)

四份评审独立得出相同机制候选:
- **M1**: latent transport DiT 在纯 latent MSE 下 undertrained, pixel-space gradient 提供更强信号
- **M4**: image_aux 把 z_pred 锚定到 decoder 可解码 manifold

M2 (SSIM) 次要, M3 (seam) 优先级最低. 没人主推 M2/M3.

### 1.3 立即写 paper, 不等任何新长跑 (4/4)

paper outline + methods + results table 立即启动. 不要等 X1/X3/X5 任何一个跑完再开始. MICCAI 2027 (deadline ~2026-12) 可行, 前提是接下来 6-8 周机制最小闭环 + 叙事收敛.

### 1.4 X2 (架构升级) / X4 (chain redesign) 立即 REJECT (4/4)

- **X2 REJECT 理由共识**: 4-8 周设计 + 50% 失败率, 在 A4-mid 已经给出可发表核心后开高风险新分支是净负. Reviewer A/D 都说 X2 可作 follow-on project, 不应阻塞当前 paper.
- **X4 REJECT 理由共识**: 改 chain 几何动 V13/V14/V18/A4 baseline 可比性, reopen substrate.

### 1.5 X1-lite 必做, NOT 全 3-run X1 (4/4)

四份评审都明确**反对** prompt 原版 X1 (3 个 ablation runs: l1-only / ssim-only / seam-only, 21d GPU). 替代版本一致:

> 一次 λ=0.08 的最具区分力的 ablation run (e.g. l1-only OR no-SSIM/no-seam) + 一次 no-training gradient/latent probe

这是 **causal attribution / mechanism falsification**, 不是 component sweep. 论证强度: 如果 l1-only 保留大部分 A4-mid 增益 → 主因偏 M1/M4 (强支持); 如果显著掉点 → 再考虑 M2/M3 (下一轮再做, 不本轮全铺开).

X1-lite 不违反 user "不调小参" 约束的理由是: 它是 mechanism control experiment, 不是 sweet-spot search. 论文必需 mechanism story.

---

## 2) 战略主路径 verdict tally

| reviewer | 主推 | 备选 | X3 立场 | X5 立场 |
|---|---|---|---|---|
| A | X1-lite + X5-gated | X3 hard-stop single run | backup | primary if data reachable in 2w |
| B | X3 + X5 | X3 + light X1 | primary | primary |
| C | Paper-now + X1-lite + X5 feasibility | X5 feasibility + paper | conditional after X1-lite | primary |
| D | X1(light) + X3 | X2 (only after X1/X3 close) | primary parallel | depends on data access |

**强收敛**: 4/4 都包含 X1-lite. 3/4 都包含 X3 (作为 primary 或 backup). 3/4 都看好 X5 但都说"data 可达性 gated".

**两个 cluster 形成**:
- **Cluster 机制优先** (A, C): X1-lite 先回答机制, 再用机制结果决定是否做 X3 / X5
- **Cluster 并行** (B, D): X1-lite + X3 同时做, X5 并行评估 data 可达性

集成解决方案: **采纳 B+D 并行方案**, 因为 X1-lite (1 run, 7d) 与 X3 (1 run, 7d) 不互斥, 都用 GPU slot 但 ≤ 2 slot 满足 user 3-slot 内约束. X5 走 data feasibility 评估轨道 (不占 GPU).

---

## 3) 新增方向 X6+ (reviewers 补)

| ID | 提出者 | 内容 | 共识 |
|---|---|---|---|
| **X6 临床/任务级评估** | A, C | 在 PET clinical / task metric (病灶可见度, 区域 SUV 误差, 视觉评分) 上验证 A4-mid; 可能 0 新训练 | 强建议 (2/4 独立提出) |
| **X7 dose-aware conditional / CFG-style dose guidance** | A, D | 把 dose level 作为 conditional input, 推理时 CFG-style 控制 dose schedule | 中建议, 风险中 |
| **X8 decoder-pullback transport objective** | C | 把 image_aux 的有效性提升为新 loss formulation (decoder-aware gradient preconditioning) | 单人提出, 长期方向 |

X6 进入立即 action set (低成本 + paper credibility). X7/X8 列入 follow-on project.

---

## 4) 偏差审计 B87-B92 (consolidated)

| ID | 偏差 | severity | source |
|---|---|---|---|
| **B87** | A4-mid 单 seed 信号过度浪漫化 (把 +0.113 dB 当"项目救星") | HIGH | A, C, D |
| **B88** | image_aux fetish (winning, 所以其它方向都被 anchor 为不重要) | HIGH | C |
| **B89** | "不调小参" 过度教条化 (mechanism ablation 被误归类为 tuning) | MED | A, B, C, D |
| **B90** | Architecture pivot optimism (X2 失败率 50% 但被列为高 EV) | HIGH | A, C |
| **B91** | V18 sunk-cost (X3 可能被悄悄变成 V18.v2 campaign) | MED | A, B, D |
| **B92** | PSNR monoculture (无 clinical / task-level metric) | MED | A, C |

**Root cause pattern**: A4-mid 是项目历史最大单实验信号, 4/4 评审独立警告 claude 容易陷入 "image_aux 神化" + "X2 架构幻想" 两个极端 — 而正确路径是中间的 mechanism falsification + 立即 paper draft.

---

## 5) 最终集成主战略

### 5.1 主路径 = **X1-lite + X3 + 立即 paper draft (X5 feasibility 并行评估, X6 立即做)**

时间表:

| 时间窗 | 动作 | 负责 / 资源 |
|---|---|---|
| **Week 0 (现在)** | (a) paper outline + methods + results 主体 skeleton; (b) X1-lite + X3 task md 起草并 push 给 codex; (c) X5 data feasibility audit (检查 cross-tracer/cross-dataset 是否实际可达, 不申请新数据资源); (d) X6 clinical/task metric 评估在已有 val artifact 上跑 | claude + user (paper), codex (task md), data feasibility (user) |
| **Week 1-2** | (a) X1-lite 1 run (~7d, slot 1) — λ=0.08 + l1-only (或 no-SSIM/no-seam); (b) X3 1 run (~7d, slot 2) — λ_img=0.08 + decoder LoRA r32 last-2 + λ_kl=0; (c) paper draft 第 1 版完成 | codex (训练), claude (paper) |
| **Week 2-3** | (d) X1-lite + X3 full-val canonical eval + paired-slice bootstrap; (e) X6 clinical metric 整合; (f) X5 决策点 (data 可达 → 启动 Week 4-6 cross-dataset run; 不可达 → 增强 mechanism section) | codex (eval), claude (整合) |
| **Week 4-6** | 任 1: X5 cross-dataset 重跑 V7 baseline + A4-mid (~3w wall clock); 任 2: 加深 mechanism section (X1 第二个 component ablation run, 仅在 X1-lite 不够 conclusive 时启动) | codex + claude |
| **Week 7-8** | paper 全文 + ablation table 收敛; figure 制作; rebuttal 材料预备 | claude + user |
| **Month 3-4** | MICCAI 2027 投递准备 | user |

### 5.2 V18 角色固化 = (b) 次要 ablation

paper 主图: V13 + A4-low + V7 + A4-mid 的 image_aux response curve.
ablation 表: V18.best, V18.last, A3 capacity-only, X3 (image_aux + LoRA) 作为 "we also tried decoder-side interventions" 一段, 占 paper 篇幅 < 15%.

任何 V18 follow-up 长跑 (除 X3 单次 additive 测试外) **不允许**. X3 stop rule: 1 run, 不论 outcome 不 relaunch.

### 5.3 image_aux mechanism story 锁定 = M1 + M4

paper Methods 章节写:
> "We hypothesize that pixel-space supervision via frozen decoder backprop addresses two coupled deficiencies of pure latent transport: (M1) latent MSE under-constrains the transport DiT on perceptually relevant modes, and (M4) image_aux gradient implicitly anchors predicted latents to the decoder's well-decodable manifold. The X1-lite ablation (l1-only at λ=0.08) supports this attribution by retaining X% of the +0.113 dB gain."

X% 由 X1-lite 结果决定.

### 5.4 paper venue 决策

- **主投**: MICCAI 2027 (deadline ~2026-12, 7 个月窗口)
- **备投 (升级路径)**: TMI / MedIA, 仅在 **X5 cross-dataset 成功** OR **X8 decoder-pullback formulation 成形** 后启动. 不应 block MICCAI 主线.

---

## 6) 立即必做 (Week 0 action items)

1. **claude 立即写**: Round 18 codex task md (X1-lite + X3 双 task, staggered 2-slot launch, F0 paired-slice bootstrap 协议复用)
2. **claude 立即写**: paper outline (按 5.2/5.3 narrative structure)
3. **user 决策**: X5 data feasibility — 是否有 second PET dataset / cross-tracer 在 2 周内可申请? 若不可, X5 转为 paper limitation section "future work".
4. **user 决策**: X6 clinical metric — 现有 val 7403 slice 是否有任何 clinical label / 病灶 ROI / SUV 分层 metadata? 若有, X6 在 Week 1 完成; 若无, X6 转为 future work.
5. **codex (Round 18 task md push 后)**: X1-lite + X3 2 slot 并行 launch + 全 substrate eval

---

## 7) 立即不做 (Week 0 negative list)

1. ❌ X2 (架构升级) 任何设计或 spike — 留作 follow-on project
2. ❌ X4 (chain redesign) 任何讨论 — break baseline 风险高
3. ❌ X1 全 3-run (l1/ssim/seam 各 1 run) — 这是 component sweep 不是 mechanism falsification
4. ❌ V18-family 任何新长跑 (V19 / V18-clean 复活 / decoder rank sweep) — 已被 4/4 拒
5. ❌ image_aux λ 继续 sweep (0.06, 0.10, 0.12, 0.16) — 这正是 user 拒绝的"调小参"
6. ❌ V14b/c 多 seed — 已被 Round 17-Stats 决议 (slice-level only, patient_id 不可恢复)
7. ❌ paper 头条仍写 V18 — 4/4 一致 V18 → 次要

---

## 8) Round 19 触发条件

Round 19 集成轮触发当 ≥ 2/3 完成:
1. X1-lite run + eval 完成 (mechanism attribution 数字出)
2. X3 run + eval 完成 (additive 判定数字出)
3. paper outline + Methods + Results skeleton 第 1 稿完成 (claude 可以独立完成, 不需等实验)

X5 / X6 / X7 不作为 Round 19 触发条件, 它们走独立轨道.
