# 0505 Local → External AI Peer Review Prompt — Plan F (V7+V8 from-scratch 150K)

**Purpose**: 把这段 prompt 喂给 GPT-5.5（high reasoning）+ Claude Opus 4.7（high reasoning）。两家**独立**审计；之后我会做交叉核对。

**与上一轮 review 的区别**:
- Round 1（`PEER_REVIEW_PROMPT_V7_V8.md`）: 审 V7/V8 **实验设计本身** — 4 reviewers 通过，已结束。
- Round 2（`PEER_REVIEW_PROMPT_PLAN_D.md`）: 审 Plan D（V7 resume V6@40K + V8 from-scratch 100K）— 2 reviewers，1 NO-GO（catch EMA-swap 阻断）+ 1 GO-WITH-FIXES，**结论 Plan D 死亡**。
- Round 3（本轮）: 审 **Plan F** — V7+V8 都从零训练到 150K（300K 总 GPU），删掉所有 resume 路径。

**审查紧迫程度**: 极高。Plan F 一旦立项要消耗 300K steps GPU。Plan D 已经被否决，作者**不允许再走第二次弯路**。Plan F 看似把 Plan D 的所有 resume 风险都删掉了，但**自审已发现一个 critical blocker（V6@step_150000.pt 不存在）**——所以本轮 reviewer 必须重点验证：(1) 这个 blocker 是否真的 blocker，(2) Plan F 还有没有其他被作者遗漏的隐藏陷阱。

**用法**: 复制 `--- PROMPT BEGIN ---` 与 `--- PROMPT END ---` 之间的全部内容，连同附件文件一并粘贴。**不要改 prompt 措辞**；语气有意 adversarial。

---

## 必须随 prompt 一并提供给外部 AI 的附件

| 路径 | 内容 | 为什么需要 |
|---|---|---|
| `review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md` | **Plan F 决策文档**（本轮的审查对象） | 核心 |
| `review/0505/local/ANALYSIS.md` | V7/V8 实验设计、决策矩阵、§7 caveats（含 EMA-swap deferred ticket） | 上下文；reviewer 已被告知不重审实验设计 |
| `review/0505/local/configs/V7_gronwall_raw.yaml` | V7 当前配置（max_steps=150000，schedule 锁定） | 实际状态 |
| `review/0505/local/configs/V8_no_image_aux.yaml` | V8 当前配置（max_steps=150000，dead_branch_watch_steps=150000） | 实际状态 |
| `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 200K 配置 | baseline 对照 |
| `review/0505/operator/v6_v61_training_and_fullval_status_20260505.md` | V6 实测 best.pt(step 185600) + last.pt(step 200000) 全集得分 | 验证 V6 ckpt 实际存在 step 列表 |
| `review/0505/local/scripts/run_v7_v8_sanity.sh` | launcher（默认 SANITY_STEPS=150000） | 启动逻辑 |
| `train_first_hop.py` 行段（**全部贴到 prompt 里**）：<br>• L62-78（`get_warmup_cosine_lr` LR 公式）<br>• L230-275（DataLoader：`WeightedRandomSampler` 主路径 + `RandomSampler(num_samples=max_steps*bs)` hop0 路径）<br>• L1481-1525（image_aux warmup_steps / ramp_steps 解析）<br>• L1560-1600（`_resolve_schedule_steps`、`lr_total_steps = override or max_steps`）<br>• L1660-1670（`img_enabled = bool(image_aux_cfg.get("enabled", True))`）<br>• L2005-2050（hop0 forward：`if img_enabled: compute_hop0_image_losses(...)` 否则 `loss_img=zero`）<br>• L2065-2095（loss 公式 `total = pair·loss_pair + lambda_roll·loss_roll + lambda_img·loss_img + ...`）<br>• L2456-2480（`save_interval` 周期保存 + `with ema.average_parameters():` 上下文管理器） | 让 reviewer 不信我的话——只信代码 |
| `pet_lr/rollout_first_hop.py` L40-130 | `rollout_multistep_losses_first_hop`（`step_weights`、`step_normalizers`、`total = (stacked * w).sum() / w.sum()`） | 验证 step_weights 在 lambda_roll>0 时如何起作用 |

---

## --- PROMPT BEGIN ---

You are a senior reviewer (NeurIPS / ICML / MICCAI level) specializing in **paired-ablation experiment design**, **bit-deterministic training reproducibility**, **EMA-swap checkpoint semantics**, and **statistical power for paired-difference experiments under noise floors**. Your reputation depends on catching subtle protocol mistakes that turn a 300K-GPU-step experiment into apples-vs-oranges and waste a week of compute.

You are reviewing **Plan F** — the third execution proposal for an existing PET-image latent-transport ablation study. Two prior plans were rejected:
- **Plan A/B/C/D** (resume-based variants): rejected because V6 checkpoints are EMA-swapped (model state at save time = EMA weights, but optimizer state = raw-weight Adam moments). Resuming from such a checkpoint runs raw-weight Adam moments on top of EMA-weight model parameters, producing a hybrid state that is neither V6-trajectory-equivalent nor a clean from-scratch run. See `train_first_hop.py:2456-2480` (`with ema.average_parameters(): save_checkpoint(...)`) for the structural cause.
- **Plan E** (from-scratch 200K each): rejected as too expensive (400K total GPU).

**Plan F** = V7 from-scratch to 150K + V8 from-scratch to 150K (300K total GPU). Anchor = V6@step_150000 full-val evaluation (NOT a re-trained V6 — V6 already exists on disk).

The experiment design (V7 = Grönwall closed-form `step_weights`; V8 = remove `image_aux`) was peer-reviewed in Round 1 and is fixed. **Do not re-litigate the design.** Your job is exclusively to assess whether **Plan F's execution protocol** produces a valid paired comparison.

The authors have already done a self-audit and surfaced **one critical blocker**. Your task is to:
1. Verify whether that blocker is real or a false alarm.
2. Find any other blocker the authors missed.
3. Recommend GO / GO-WITH-FIXES / NO-GO.

---

### Background you can take as given (do not audit)

- 4-hop latent-domain cascade `D50 → D20 → D10 → D4 → NORMAL`. Backbone frozen DINOv2 + ViT-MAE decoder + LoRA + per-hop residual head + pixel-forcing branch (only at hop0).
- V6 baseline: 200K steps, `step_weights = [0.5, 2.0, 1.5, 1.0]`, `image_aux.enabled=true`, `lambda_img` ramp `0 → 0.04` with `image_aux.warmup_ratio=0.0` (so image_aux is non-zero from step 0). `seed=42`, `deterministic=true` (warn-only). **V6 yaml `save_interval: 20000`** → V6 step_*.pt ckpts at {20K, 40K, 60K, 80K, 100K, 120K, 140K, 160K, 180K, 200K}. V6 also has best.pt (rolling-val best at step 185600) and last.pt (step 200000).
- V7: V6 baseline with **only** `step_weights` swapped to closed-form `[0.6106, 2.0, 2.7041, 2.0423]`. σ-normalize OFF. Everything else identical. yaml `save_interval: 10000`, `max_steps: 150000`.
- V8: V6 baseline with **only** `image_aux.enabled=false`. Everything else identical. yaml `save_interval: 10000`, `max_steps: 150000`, `dead_branch_watch_steps: 150000`.
- Schedule lock: V7/V8 yamls explicitly set `rollout.warmup_steps=50000`, `rollout.ramp_steps=100000` (lambda_start=0, lambda_end=4), `lr_schedule.total_steps_override=200000` (LR scheduler uses 200000 as denominator, NOT the 150000 max_steps). This means V7/V8 at step `k` (0 ≤ k ≤ 150000) sees the same `lambda_roll(k)` and `lr(k)` as V6 at step `k`, by construction.
- Decision endpoint: step 150000 (= rollout ramp end, lambda_roll=4.0, lr ≈ 0.22 × base_lr).

---

### Plan F — exactly what is proposed

| Arm | Init | max_steps | Decision endpoint | `lambda_roll` at endpoint | `lr` at endpoint |
|---|---|---:|---|---:|---:|
| **V7** | from scratch, `seed=42` | 150000 | step 150K | 4.0 | 1.75e-5 (0.22 × base_lr) |
| **V8** | from scratch, `seed=42` | 150000 | step 150K | 4.0 | 1.75e-5 (0.22 × base_lr) |
| **V6 (anchor)** | (already trained) | 200000 | (full-val at "step 150K") | 4.0 (locked schedule) | 1.75e-5 (locked schedule) |

**Decision rule** (from ANALYSIS.md §6):
- V7: paired comparison V7@best.pt vs V7@last.pt vs V6@step_150000.pt on full-val NORMAL chain MSE. Decision threshold Cohen's d ≥ 0.10.
- V8: paired comparison V8@best.pt vs V8@last.pt vs V6@step_150000.pt. Same threshold. Plus `dead_branch_watch` is kept armed throughout step 0 → 150K (V8 only).

---

### The author's self-audit found ONE critical blocker

> **BLOCKER #1 — V6@step_150000.pt does not exist on disk.**
> V6 was trained with `save_interval: 20000` → V6's `step_*.pt` ckpts are at {20K, 40K, 60K, 80K, 100K, 120K, 140K, 160K, 180K, 200K}. **150K is NOT in this set.** The Plan F decision matrix specifies "V7/V8 vs V6@step_150000.pt", but no such file exists.
>
> Currently evaluated V6 ckpts (per `review/0505/operator/v6_v61_training_and_fullval_status_20260505.md`):
> - V6 best.pt @ step 185600 (rolling-val best): full-val NORMAL chain MSE = 2.430e-4
> - V6 last.pt @ step 200000: full-val NORMAL chain MSE = 2.427e-4

**Audit Q1**: Is BLOCKER #1 real?
- Verify by inspecting V6 yaml save_interval and `train_first_hop.py:2456-2480` periodic save logic.
- If real, what are the candidate fixes? Score each on (1) GPU cost added, (2) statistical validity preserved, (3) protocol simplicity:
  - **Fix A**: Re-train V6 with `save_interval: 10000` (200K extra GPU steps).
  - **Fix B**: Anchor at V6@last.pt (step 200000) instead, and extend V7/V8 to 200K (gain +100K GPU; total 400K = Plan E in disguise).
  - **Fix C**: Anchor at V6@step_140000.pt + V6@step_160000.pt (interpolate or compare to both).
  - **Fix D**: Anchor at V6@best.pt (step 185600) and V6@last.pt (step 200000); compare V7/V8@best.pt and V7/V8@last.pt against the matching V6 stages. Acknowledge that V6@last.pt is at a different schedule stage (lambda_roll=4 fixed for last 50K + lr decayed to 0.025×base) than V7/V8@last.pt (lambda_roll=4 reached only at the END of training, lr at 0.22×base). This breaks endpoint-equivalence.
  - **Fix E**: Run a **single-shot V6 evaluation pass** that loads V6@step_140000.pt, fast-forwards 10K steps with the locked schedule, and saves a "synthetic V6@step_150000.pt". Audit feasibility (the EMA-swap blocker still applies — V6@140K is EMA-swapped, so resume-then-train-10K is the same hybrid-state issue).
  - **Fix F**: Train V7 and V8 to **160000** (matching V6's existing save point), use V6@step_160000.pt as anchor. Cost: +20K GPU (3% increase). Verify schedule fields still align (yes — `total_steps_override=200000` covers k=160K).
  - Other?
- Recommend the cleanest fix.

---

### Other Plan F audit points (not on the author's self-audit list)

#### Audit Q2 — DataLoader `RandomSampler` num_samples discontinuity

`train_first_hop.py:258-262`:
```python
hop0_sampler = RandomSampler(
    hop0_train,
    replacement=True,
    num_samples=max_steps * hop0_batch_size,
)
```

V6 was trained with `max_steps=200000`. V7/V8 use `max_steps=150000`. The `RandomSampler.__iter__` calls `torch.randint(high=N, size=(num_samples,), generator=...)`, which materializes ALL `num_samples` indices at once.

**Audit**:
- (a) Does `torch.randint(high=N, size=(K,))` produce a sequence whose first 150K×bs values are **identical** to those of `torch.randint(high=N, size=(M,))` (M > K, same generator state)? Or does PyTorch's vectorized RNG produce different values for different output shapes? Cite PyTorch source / docs.
- (b) If V6 ran with `num_workers > 0`, each DataLoader worker forks with a worker-specific RNG seed. Does V7's hop0 worker fork RNG match V6's hop0 worker fork RNG, given that `worker_init_fn` (if any) seeds via `torch.initial_seed() + worker_id`?
- (c) Even if (a) and (b) both pass, the `WeightedRandomSampler` for `main_train_loader` consumes RNG independently. Does its consumption pattern depend on `max_steps`? Inspect L230-247.
- (d) Verdict: would V7's training trajectory (sequence of (hop0_batch, main_batch) pairs presented to the optimizer) at step `k` (1 ≤ k ≤ 150000) be **identical**, **statistically-equivalent-but-bit-different**, or **systematically biased** relative to V6's trajectory at the same step `k`?

If the verdict is "systematically biased", Plan F's claim of paired comparison breaks even at step 1.

#### Audit Q3 — V8 RNG asymmetry due to image_aux skip

`train_first_hop.py:2014-2032`:
```python
if img_enabled:
    img_losses = compute_hop0_image_losses(model, hop0_batch, cfg)
    loss_img = img_losses["total"]
else:
    zero = pair_losses["total"].new_zeros(())
    ...
    loss_img = zero
```

In V6, `compute_hop0_image_losses` performs a forward pass through `model.decode(...)` and computes SSIM, L1, seam losses on decoded pixels. This forward pass consumes RNG (dropout, attention, etc.).

In V8 (`img_enabled=False`), this forward pass is skipped → V8 does NOT consume the RNG that V6 consumes.

**Audit**:
- (a) Verify by inspecting `pet_lr.losses_first_hop.compute_first_hop_image_loss` (referenced from `train_first_hop.py:25`) and `train_first_hop.py:786-810` (`compute_hop0_image_losses` body): does this function consume any RNG (random crops, stochastic augmentations, dropout-enabled forward, MC sampling)?
- (b) If yes, V8's RNG state at step k diverges from V6's RNG state at step k starting from step 1. The DataLoader fetched batches will be drawn from the divergent RNG state. Therefore the very thing V8 is supposed to test ("V6 minus image_aux") is **confounded** by an RNG-driven trajectory difference.
- (c) Mitigations: (i) explicitly dummy-call `compute_hop0_image_losses` in V8 with `lambda_img=0` (wasted compute but RNG-aligned); (ii) use a separate `torch.Generator` for image_aux RNG and seed it identically; (iii) accept the confound and increase the decision threshold (Cohen's d ≥ 0.20 instead of 0.10) to absorb the noise; (iv) declare the RNG drift negligible (provide quantitative argument).
- (d) Recommend the cleanest mitigation. If V8's RNG asymmetry is severe, V8 may need to fall back to a **lambda_img=0** scheme (image_aux head exists and is forwarded but contributes 0 to loss) — investigate whether this is actually possible by setting `image_aux.enabled=true` + `lambda_img.lambda_max=0` in V8 yaml and reading the trainer to confirm no other `image_aux.enabled` gate exists.

#### Audit Q4 — Same-GPU determinism requirement

`deterministic=true` (warn-only) in V6 yaml. The σ-normalize A/B sanity in 2026-05-03 showed ~10% drift over 50K steps under fp32 + warn-only deterministic + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. PyTorch warn-only mode does not enforce bit-equivalence across:
- Different physical GPU silicon (compute capability, memory layout, cuBLAS heuristic table)
- Different CUDA / cuDNN / cuBLAS versions
- Different driver versions

V6 was trained on a specific GPU. V7 / V8 must run on **that same physical GPU** for the trajectory comparison to be meaningful.

**Audit**:
- (a) Does the operator pre-flight checklist (in PLAN_F doc §9) include "verify V7/V8 run on the same physical GPU as V6"? Inspect.
- (b) If the operator runs V6 on GPU2 and V7 on GPU0 (both A100 80GB but different physical chips), what is the expected paired drift over 150K steps? Is it larger or smaller than the V7-vs-V6 effect Plan F is trying to detect?
- (c) What is the **fail-loud** mechanism? E.g., a sentinel that records V7's `metrics.jsonl[step==1000]` first-1K-step rolling val and compares to V6's `metrics.jsonl[step==1000]`; if the relative difference exceeds a predefined threshold (e.g. 1%), abort the run before wasting 149K more steps. Recommend specific scalars and tolerances.

#### Audit Q5 — Statistical power at 150K vs paired noise floor

V6 vs V6.1 historical paired comparison (full-val): NORMAL chain MSE diff ≈ 1.275e-6 → effect size Cohen's d ≈ 0.056 (per ANALYSIS.md §3 historical noise reference). V6.1 had ONE protocol change (rollout floor: lambda_roll min ≥ 0.05) but is otherwise identical to V6.

**Audit**:
- (a) Plan F decision threshold is Cohen's d ≥ 0.10. If V7's true effect is `d=0.10`, what is the **per-slice paired-test** power assuming N=7403 val slices and the V6 vs V6.1 noise structure (mean diff 1.275e-6, std diff ≈ 1.275e-6 / 0.056 ≈ 2.28e-5)? Compute power explicitly.
- (b) If power is < 80%, what is the minimum N or minimum effect size that gives 80% power? Does Plan F's design satisfy this?
- (c) The decision matrix uses a **two-tier** threshold: "all chain_*_mse not worse" + "endpoint d ≥ 0.10". Is this multi-test family adequately controlled (e.g. Bonferroni 4-hop × 1-direction = α/4)?
- (d) Alternative: rank-based test (Wilcoxon signed-rank on per-slice paired diffs) is more robust to outlier slices. Recommend whether Plan F should switch from t-test/Cohen's d to rank-based.

#### Audit Q6 — Best.pt selection from 30 candidates

`best_select_full_eval_interval=5000` with `max_steps=150000` → 30 candidate evaluations contributing to best.pt selection. Multiple-comparison inflation of type-I error.

**Audit**:
- (a) Is the comparison "V7@best.pt vs V6@step_150000.pt" valid given best.pt was selected via 30-way max-search on V7's val score? If V7's true effect is null, what is the probability that V7@best.pt beats V6@step_150000.pt by chance alone (no actual improvement)? Compute or bound.
- (b) Pre-registration recommendation: should the comparison be V7@last.pt vs V6@last.pt (no max-search bias) or V7@best.pt vs V6@best.pt (consistent selection rule)? Note V6 best.pt @ step 185600 ≠ Plan F endpoint at step 150K — adopting V6@best.pt as anchor breaks endpoint-equivalence (cf. Q1 Fix D).
- (c) Recommended primary endpoint and supporting secondary endpoints. Specify pre-registration text the authors should commit to before launch.

#### Audit Q7 — Schedule lock formula correctness

V7/V8 yamls set:
```
training.rollout.warmup_steps        = 50000
training.rollout.ramp_steps          = 100000
training.rollout.lambda_start        = 0.0
training.rollout.lambda_end          = 4.0
training.lr_schedule.total_steps_override = 200000
training.lr_schedule.warmup_ratio    = 0.15
training.lr_schedule.min_lr          = 2.0e-6
training.lr                          = 8.0e-5
training.max_steps                   = 150000
```

V6 yaml has `lr_schedule.warmup_ratio=0.15` and the trainer resolves warmup as `int(0.15 * total_steps)`. With `total_steps_override=200000`, warmup = 30000 in V7/V8 (matching V6's resolved warmup since V6 also uses `total_steps=200000`).

**Audit**:
- (a) Verify by reading `train_first_hop.py:62-78` (`get_warmup_cosine_lr`) and L1560-1600 (`_resolve_schedule_steps`, `lr_total_steps = override or max_steps`). Confirm the warmup_steps resolution path: is `warmup_steps = int(warmup_ratio * lr_total_steps)` where `lr_total_steps = total_steps_override = 200000`? Or is it `warmup_ratio * max_steps = 22500`?
- (b) Compute `lr(V7@k)` for k ∈ {30000, 50000, 100000, 150000} symbolically. Confirm equality with `lr(V6@k)`.
- (c) Compute `lambda_roll(V7@k)` for the same set. Confirm equality.
- (d) Are there **other** schedules (image_aux warmup, FOC, alignment, EMA decay) that depend on `max_steps` and would NOT be locked to V6 by this scheme? Inspect L1481-1525 (image_aux ramp resolution) and search for any `* max_steps` or `/ max_steps` patterns. If found, are they actively configured in V7/V8 yamls in a way that locks them to V6 behavior?

#### Audit Q8 — V7 step_weights cancellation when lambda_roll=0

For step k ∈ [0, 50000), `lambda_roll(k) = 0`. In `train_first_hop.py:2065-2095` total_loss formula, the rollout term is `lambda_roll * loss_total` → 0. Therefore `step_weights` (which only enter `loss_total`) cannot affect `total_loss` or any gradient.

**Audit** (this was Round 2 Claim 1, still relevant for Plan F):
- (a) Is the multiplication `lambda_roll * loss_total` the **only** path by which `step_weights` flows into the optimizer? Or are step_weights logged in metrics that drive a watchdog that triggers parameter updates?
- (b) Are step_weights ever used in the **forward** pass of `model(...)`?
- (c) Does the rollout forward pass consume RNG **before** the `lambda_roll * loss_total` multiplication? If yes, V7 vs V6 RNG drifts from step 1 even when both have `lambda_roll=0`.
- (d) Verdict: in V7's [0, 50K) region, is V7's training trajectory `bit-equivalent`, `statistically-equivalent`, or `systematically-different` from V6's [0, 50K) region — assuming Q2 (DataLoader RNG) and Q4 (same GPU) both pass?

If V7@k ≠ V6@k for k ∈ [0, 50K), then V7's [50K, 150K) region builds on a different starting state than V6's [50K, 150K) region, and the V7 step_weights ablation is not a clean comparison even in principle.

---

### Your task — answer each question in order

For each question, give a verdict: **PASS** (Plan F is safe), **CONCERN** (specify issue + severity), or **BLOCKER** (specify required fix). Cite line numbers.

1. **Q1 — V6@step_150000.pt anchor existence + recommended fix**
2. **Q2 — DataLoader RandomSampler RNG behavior across max_steps difference**
3. **Q3 — V8 image_aux skip RNG asymmetry + mitigation**
4. **Q4 — Same-GPU determinism + fail-loud sentinel design**
5. **Q5 — Statistical power at 150K vs noise floor**
6. **Q6 — Best.pt selection multiple-comparison inflation**
7. **Q7 — Schedule lock formula correctness + image_aux/FOC/alignment alignment**
8. **Q8 — V7 step_weights inertness in [0, 50K)**

After Q1–Q8, provide:

9. **Holistic verdict**: GO / GO-WITH-FIXES / NO-GO. List required fixes ordered by criticality.
10. **What I would have done differently**: 1 paragraph if you see a fundamental flaw not caught by Q1–Q8.

---

### Output format

Return your answer as a single markdown document with this exact structure:

```markdown
# Plan F — Reviewer audit (REVIEWER_NAME)

## Summary
{2-3 sentences. Top-line verdict: GO / GO-WITH-FIXES / NO-GO. List the fixes.}

## Q1 — V6@step_150000.pt anchor
**Verdict**: PASS / CONCERN / BLOCKER
{evidence with line numbers; recommended fix from {A,B,C,D,E,F,Other}}

## Q2 — DataLoader RNG num_samples
**Verdict**: ...
### Q2(a) torch.randint truncation
### Q2(b) num_workers worker RNG
### Q2(c) WeightedRandomSampler
### Q2(d) overall trajectory equivalence verdict

## Q3 — V8 image_aux RNG asymmetry
**Verdict**: ...
{recommended mitigation from {(i),(ii),(iii),(iv)}}

## Q4 — Same-GPU determinism + sentinel
**Verdict**: ...
{recommended sentinel scalars and tolerances}

## Q5 — Statistical power
**Verdict**: ...
{computed power; recommended N or effect threshold; rank-test recommendation}

## Q6 — Best.pt multiple-comparison
**Verdict**: ...
{recommended primary endpoint pre-registration text}

## Q7 — Schedule lock + non-rollout/non-LR schedules
**Verdict**: ...
### Q7(a) warmup_steps resolution path
### Q7(b) lr(k) symbolic check
### Q7(c) lambda_roll(k) symbolic check
### Q7(d) image_aux / FOC / alignment lock check

## Q8 — V7 step_weights inertness in [0, 50K)
**Verdict**: ...

## Q9 — Holistic verdict
{GO / GO-WITH-FIXES / NO-GO; ordered fix list}

## Q10 — What I would have done differently
{1 paragraph}
```

Be ruthless. The authors have already burned one peer review round on Plan D. They cannot afford to discover a Plan F flaw mid-training. If you see a flaw not on the question list, surface it under Q10.

## --- PROMPT END ---
