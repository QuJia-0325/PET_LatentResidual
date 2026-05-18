# Next-Stage Experiment Design — V18 evaluation + slot 2/3 routing (post Round 6)

- date: 2026-05-17 深夜
- branch: foc_lite_hop0（草稿, 未 push）
- 上游: [REVIEW_INTEGRATION_round6_20260517.md](./REVIEW_INTEGRATION_round6_20260517.md)
- user 决策 (Round 6 §6): `ABAAA` = 阈值双轨披露 + 不写 V18b yaml + 加 step 180K early-eval + V21 retire 本次清理 + 保留 §7 自警
- status: **DRAFT, 待 AI peer review (Round 7)**, push gitee 仅在通过 review 后
- 硬约束 (standing): **服务器内存最多 3 个并行训练任务**。Slot 1 = V18 已占用 (~step 180K / 200K, 剩 12-18h)。slot 2/3 候选必须在此约束下评估。

---

## §0 — 给 reviewer 的 5 句话总览

1. 本文档不重审 V18 / V13 / V14 / V21 设计 (Round 1-5 已穷尽), 只设计 **V18 跑完后的 4 阶段实验路径**。
2. 用 **dual-track threshold** (PRIMARY +0.30 / Round-5 修正 +0.15 + SECONDARY +0.05) 评估 V18, 不 mid-run 改预注册。
3. Slot 2 = V13 (Round 5 共识), launch 时机 = 收到 step 180K early-eval **之后** + V18 200K 评估**之前**。
4. Slot 3 = **contingent**, 由 V18 200K 结果决定: SUCCESS → null / PARTIAL → V18-clean from V7 / NULL → backbone 方向。
5. V18b (从 V18 ckpt 续) 已**永久撤销** (Round 6 reviewer 共识 + agent2 代码 blocker)。

---

## §1 — 阶段 0 (今天, 0 GPU): 文档与清理

### 1.1 V18_design_rationale.md 修订 (双轨披露)

修 §2.3 阈值表为双列:

```markdown
| 情形 | 原注册 (launch time 2026-05-17) | Round 5 修正版 (lower-bound 校正) | Action |
|---|---|---|---|
| PRIMARY SUCCESS | ΔPSNR ≥ +0.30 dB | ΔPSNR ≥ +0.15 dB | V18 主路径成立 |
| PARTIAL | [+0.05, +0.30] dB | [+0.03, +0.15] dB | sweep rank/blocks |
| KILL @ 20K | ΔPSNR < +0.05 | < +0.03 | early stop |
| **SECONDARY (新, 用于 follow-up 触发)** | — | ΔPSNR ≥ +0.05 dB | 触发是否做 V18-clean |

**评估时同时报告原 + 修正两套判定** (例如 "原阈值 PARTIAL, 修正阈值 SUCCESS")。
不删原数字, 保留 audit trail。Round 5 修正出现在 V18 launch 之后, 严格说是 post-hoc;
双轨披露是 medical trial / cosmology 处理 pre-reg 修正的标准做法。
```

新增 §2.4 "Round 5/6 修正记录" 文档化:
- B10 (11.2 dB ≠ attackable, 它是 round-trip ceiling)
- B11 (0.5 dB 也不是 attackable, 它是 lower-bound, 真实 EV 区间 **[0.05, 3] dB**)
- B12 (禁止用 "baseline 错了" 当借口 mid-run 改阈值; 双轨披露是唯一合规做法)
- B13 (sunk-cost-resume: 拒绝 "V18 已付出 40K 不能浪费" 框架; 若 B9 真有害, 40K 已 corrupted)
- B14 (改 yaml 必须 cross-check 联动字段; LR schedule / resume_from / config mismatch / LoRA→LoRA support 都要 verified)

新增 §5.2 自警第二条: **"V12 禁令"**: 禁止 mid-run 修改 pre-registered SUCCESS/KILL threshold, 即使 baseline 计算后来发现错误; 修正只能通过 **dual-track disclosure** + **SECONDARY threshold**, 不能替换 PRIMARY。

### 1.2 V21 retire 残留清理

4 个文档加 `[RETIRED 2026-05-17 Round 5: B8 — V21 描述与 conv_head.py 不符, 它是 token-domain 替换不是 post-decode refiner]` 标记 (不删原文):
- [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md) §6.2, §7
- [V18_EXECUTION_REPORT_20260517.md](./V18_decoder_lora/V18_EXECUTION_REPORT_20260517.md) §Phase 0
- [GAP_DECOMP_REPORT.md](./V18_decoder_lora/gap_decomp_fullval_20260517/GAP_DECOMP_REPORT.md) "Pre-Registered Decision Rule" 表
- [CODEX_RUNBOOK_V18_20260517.md](./CODEX_RUNBOOK_V18_20260517.md)

---

## §2 — 阶段 1 (今天-明天, 0 GPU): V18 step 180K early-eval

### 2.1 目标 & 非目标

**目标**: 用现有 metrics.jsonl @ step 180000 informally 估计 V18 趋势, 给 slot 2/3 决策提供 input。

**非目标 (硬约束)**:
- **不据此 kill V18**。即使 step 180K ΔPSNR < +0.03 dB, V18 仍跑完 200K (剩 12-18h, sunk cost, 同时也避免 "单点 metric kill" 偏差)。
- **不修改 V18 yaml 任何字段**。

### 2.2 codex 执行 (在服务器)

```bash
cd <V18 run dir>
python -c "
import json, pathlib
mj = pathlib.Path('metrics.jsonl')
records = [json.loads(l) for l in mj.open() if l.strip()]
val = [r for r in records if 'val_chain_normal_psnr_clip3' in r]
print('=== V18 val_full record at step >= 175000 ===')
for r in val:
    if r.get('step', 0) >= 175000:
        print(f\"step={r['step']:>7} val_chain_normal_psnr_clip3={r['val_chain_normal_psnr_clip3']:.4f}  val_chain_normal_kl={r.get('val_chain_normal_kl', 'NA')}\")
"
```

对比 V7 best.pt 同 metric (已在 [disambig report](./disambig/)), 写 `V18_STEP180K_EARLY_EVAL.md`, 三档判定:

| ΔPSNR(V18@180K − V7 best) | 判定 | Slot 2/3 影响 |
|---|---|---|
| ≥ +0.10 dB | V18 趋势健康 | slot 2 = V13 立即 launch; slot 3 = null (等 V18 200K) |
| [+0.03, +0.10] dB | V18 不确定, 偏低 | slot 2 = V13 立即 launch; slot 3 = **draft V18-clean yaml** (但不 launch, 等 V18 200K 确认) |
| < +0.03 dB | V18 趋势死 | slot 2 = V13 立即 launch; slot 3 = **draft V18-clean yaml** + draft backbone 方向 phase plan |

**注意**: 三档判定都 launch V13。V13 与 V18 outcome 独立, 它的诊断价值 (V7-V8 +0.308 dB 中 image_aux 真实贡献) 不受 V18 结果影响。

---

## §3 — 阶段 2 (Day 1, slot 2 launch): V13

### 3.1 V13 = V7 + image_aux off

- yaml: [review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml](../0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml) (已就绪)
- 训练时长: 7 天
- 启动命令: 由 codex 在服务器 smoke 200 step 验证后给出
- 预注册 success: 不需要 (V13 是诊断, 不是优化; 任何 ΔPSNR(V13 vs V7) 都是有效信息)
- 资源占用: slot 2 (slot 1=V18, slot 3=空)

### 3.2 V13 与 V18 的独立性

V18 改 RAE decoder LoRA; V13 改 latent transformer 训练目标 (去掉 image_aux loss term)。两者不共享参数, 不共享 dataloader 关键路径 (V13 跳 image_aux forward), 不互相 confound。

---

## §4 — 阶段 3 (Day 3-4, V18 step 200K eval): 主路径判定

### 4.1 V18 200K 评估协议

**Pre-registered, immutable** (本次草稿即 pre-register):

```
1. V18 max_steps=200K 完成或触发自然 kill_switch。
2. 用 eval_first_hop_224_clip3.py 跑 full val (与 V7 best.pt 同 split)。
3. 报告:
   a. PRIMARY 判定 (原阈值 +0.30 / +0.05): SUCCESS / PARTIAL / KILL
   b. 修正阈值判定 (+0.15 / +0.03): SUCCESS / PARTIAL / KILL  
   c. SECONDARY 判定 (+0.05): TRIGGERED / NOT
   d. EV 区间 [0.05, 3] dB 内的位置 (低/中/高)
   e. KL drift (PSNR(decode_V18(z_GT), x_target) vs V7 baseline 同度量)
4. 写 V18_FINAL_EVAL_<TS>.md, 四档 outcome 决定 slot 3:
```

### 4.2 Slot 3 路由 (V18 200K eval 之后)

| V18 outcome | Slot 3 行动 | 后续 milestone |
|---|---|---|
| 原阈值 SUCCESS (ΔPSNR ≥ +0.30) | null / V9a CPU | 转 paper draft + variants sweep |
| 修正 SUCCESS but 原 PARTIAL (+0.15 ≤ ΔPSNR < +0.30) | **null** (避免新争议; SECONDARY 已 trigger, 但需要先解决 dual-track 报告争议) | 写 V18_DUAL_TRACK_INTERPRETATION.md, 由 user + reviewer 决定是否算成功 |
| 双轨都 PARTIAL ([+0.05, +0.15)) | **V18-clean from V7** (use_pred_latent=false from scratch, mirror V18 schedule) | clean B9 ablation |
| 双轨都 KILL (< +0.05) | **不跑 V18-clean** (B9 fix 救不了) → backbone/data/architecture 方向 phase plan | retire V18 family |

### 4.3 V18-clean 设计 (若触发)

**关键: 不是 V18b**。V18b (从 V18 ckpt 续) 已永久撤销。V18-clean:

- 起点: V7 best.pt (与 V18 同起点)
- 改动 (相对 V18 yaml): `use_pred_latent: true` → `false`, `warmup_steps: 2000` 保持 (从 V7 起点重新 ramp), `max_steps: 200000` 保持, 其他全部不动
- LR schedule: 全部继承 V18 (cosine total_steps_override=200000)
- 输出目录: `.../V18_clean_kl_gt_anchor/run`
- 训练时长: 7 天 (与 V18 等价)
- 资源占用: slot 3
- Pre-registered success: ΔPSNR(V18-clean vs V18) ≥ **+0.05 dB** → B9 fix 有效; < +0.01 dB → B9 不重要; 其间 → inconclusive
- **代码前提**: 不需要 LoRA→LoRA resume (从 V7 出发, 走 V18 已验证路径)。这绕开 agent2 发现的 hard blocker。

### 4.4 V18-clean 与 V18 的 paired comparison cleanness

- 同起点 (V7 best) ✓
- 同 schedule ✓
- 同 hyperparams (rank=32, blocks=[6,7], λ_kl=0.05, λ_img=0.04, batch, seed) ✓
- 仅差 1 bit: `use_pred_latent` ✓ → **干净 B9 ablation**

---

## §5 — 阶段 4 (Day 10+): 收尾

### 5.1 V13 完成

- 评估 V13 vs V7: ΔPSNR(V13 vs V7) = image_aux 贡献量
- 若 ΔPSNR < 0.05 dB → image_aux 在 V7-V8 +0.308 dB 中贡献小 → 重新 attribute 给 Grönwall step_weights
- 若 ΔPSNR > 0.15 dB → image_aux 真有贡献, 但 V7-V8 多数差异由 image_aux 解释
- 不论 outcome, 写 V13_FINAL_ANALYSIS.md 进 paper supplementary

### 5.2 V18-clean 完成 (若 launched)

- ΔPSNR(V18-clean vs V18) → quantify B9 影响
- 若 ≥ +0.05 dB → 未来 V18 family 默认 `use_pred_latent=false`; paper 报告两个数字
- 若 < +0.01 dB → B9 是 nuisance, paper 中提一句即可

### 5.3 V9 (β_NORMAL=2.5) 不动

Round 4-5 共识 V9 性价比低且与 V18/V13 独立, 留作未来 milestone 候选 (BACKLOG)。

---

## §6 — Resource accounting (内存硬约束 max=3)

时间线下 slot 占用:

```
        Slot 1            Slot 2          Slot 3
今天    V18 (180→200K)    空              空            ← 阶段 1 early-eval
Day 1   V18 (190→200K)    V13 launch      空            ← 阶段 2 slot 2 launch
Day 3   完成              V13 (~30%)      [contingent]  ← 阶段 3 V18 200K eval
Day 4   [free]            V13 (~40%)      V18-clean?    ← slot 3 routing
...
Day 10  [V18 paper draft] V13 finish      V18-clean finish
```

**任何时刻 ≤ 3 任务**。slot 3 永远是 **contingent**, 不预占。

---

## §7 — claude 自警 (Round 6 carry forward)

Round 6 agent1 + agent3 联合识别 4 个新偏差 (B11/B12/B13/B14), 本设计文档已内化:
- B11 (0.5 dB 当 attackable) → 用区间 [0.05, 3] dB 替换单点
- B12 (mid-run 改阈值话术) → dual-track disclosure 而非替换
- B13 (sunk-cost-resume) → 永久撤销 V18b, 用 V18-clean from V7
- B14 (yaml 改字段不查联动) → §4.3 明确 V18-clean 字段改动 + cross-check 要求

新自警 (Round 7 写作时): **claude 在 §1.1 起草时一度想把"双轨披露"措辞写得让 SECONDARY 优先级高于 PRIMARY** (例如 "我们建议用 SECONDARY 作为 trigger, PRIMARY 仅作 reference") — 这等价于 soft 替换阈值, 是 B12 同型复发。已在最终稿改回 "SECONDARY 仅触发 follow-up, 不改 SUCCESS 定义"。这是 mid-draft self-catch。

---

## §8 — 给 Round 7 reviewer 的 6 个问题

### Q1 — Dual-track threshold 是否真能避免 goalpost-moving 指控?

§1.1 + §4.1 用 dual-track 同时报告原 +0.30 与修正 +0.15。但:
- 若 V18 = +0.20 dB, "修正 SUCCESS / 原 PARTIAL" 是否在 reader 心理上等于 SUCCESS?
- §4.2 第二行 ("修正 SUCCESS but 原 PARTIAL → null + 由 user+reviewer 决定") 是否仍把决策权 implicit 交给修正阈值?
- 替代: 是否应**完全不写修正阈值**, 仅在 §2.4 文档 EV 区间, 评估时只用 PRIMARY?

### Q2 — V18-clean 是否仍承担 B13 (sunk-cost) 风险?

V18-clean 从 V7 出发, 是干净起点 ✓。但:
- 7 天 GPU 成本 = 必须 slot 3 整段时间, 排挤其他候选
- 若 V18 PARTIAL outcome 是 "decoder 真无空间", V18-clean 注定也 PARTIAL → 7 天 GPU 浪费
- 是否应在 launch V18-clean 之前加 "early-kill" gate? 例如 step 20K eval if ΔPSNR(V18-clean vs V18 @ 20K) < +0.01 → 立刻杀

### Q3 — Step 180K early-eval 的 "三档都 launch V13" 是否冗余?

§2.2 三档都 launch V13。若 V18 < +0.03 dB (趋势死), 是否应改 slot 2 = V18-clean 提前 launch, 而不是 V13? 反方: V13 与 V18 outcome 独立, 不该让 V18 影响 V13 决策。请评估。

### Q4 — V18-clean 的 success threshold +0.05 dB 是否信噪比足够?

§4.3: ΔPSNR(V18-clean vs V18) ≥ +0.05 dB. V7-V6_NOISE = 0.026 dB (上界 d_pure 估计)。+0.05 dB 是 2×d_pure, 信噪比是否足够? 替代: +0.08 dB? +0.10 dB?

### Q5 — V21 retire 标记策略 (加标记 vs 删段) 是否正确?

§1.2 加 `[RETIRED ...]` 标记不删原文。优点 = audit trail。缺点 = 文档仍含 V21 描述, 未来快速浏览者可能仍被误导。是否应在标记之外加 strikethrough? 或在文档顶部 ToC 加 V21 retired 警告?

### Q6 — 内存硬约束 (max 3) 是否还有其他未考虑的约束?

slot 1 = V18 训练任务; slot 2 = V13 训练任务; slot 3 = V18-clean 训练任务 (若触发)。但:
- V18 200K eval 本身需要 GPU (eval_first_hop_224_clip3.py)。eval 期间是否占用 1 slot?
- 若 eval 占 slot 1 (V18 train 已完成 → slot 1 free), 那 eval + V13 train + V18-clean train = 3 个 GPU 任务并行, ok
- 但 eval 可能与 V13 抢同一张卡的内存 (具体看部署), 这是否需要预留 buffer?
- **请 reviewer 评估是否需要在 §6 加 eval 期间的 slot 占用细节**

---

## §9 — 输出格式 (请 reviewer 严格遵守)

每位 reviewer 独立产出:

1. **6 个问题逐条**: APPROVE / MODIFY (说明) / REJECT (说明)
2. **整体 verdict**: READY TO PUSH / MODIFY THEN PUSH / BLOCK
3. **新偏差识别**: 本设计文档是否引入新 confirmation bias (claude 在修复 B11-B14 时是否又制造新偏差)
4. **代码层风险**: 是否有未被识别的代码 blocker (例如 V18-clean 是否真的不需要 LoRA→LoRA resume? 是否有其他 yaml 字段联动?)

## §10 — 约束与提醒

- **内存硬约束 max 3 任务**贯穿全设计, reviewer 不可推荐违反此约束的实验
- **不要重审** V18 / V13 / V14 / V21 设计本身 — Round 1-5 已穷尽
- **不要 假设** 你能跑代码 / 看 wandb / 触 GPU
- **特别关注** §4.1 dual-track 是否是 stable 解, 是否会在 V18 PARTIAL outcome 时复发 B12 (mid-run 改阈值话术)
