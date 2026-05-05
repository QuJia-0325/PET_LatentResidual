# 0505 Local → External AI Peer Review Prompt — Plan D (V7 resume + V8 from-scratch)

**Purpose**: 把这段 prompt 喂给 GPT-5.5（high reasoning）+ Claude Opus 4.7（high reasoning）。两家**独立**审计；之后我会做交叉核对。

**与上一轮 review 的区别**: 上一轮（`PEER_REVIEW_PROMPT_V7_V8.md`）审的是 V7/V8 **实验设计本身**（hypothesis、决策矩阵、阈值）。本轮审的是基于上一轮 4 位 reviewer 共识反馈做出的 **Plan D 修订方案**——一个具体的"V7 从 V6@40K resume 训练到 100K + V8 从 0 训练到 100K"协议。**不要重审 V7/V8 设计**；那已经过了。**只审 D 方案的可行性、bit/trajectory-equivalence 论证、以及操作风险**。

**审查紧迫程度**: 高。一旦 D 方案立项，将在 GPU 上消耗 160K steps（V7 60K + V8 100K）。如果 D 的 bit-equivalence 论证错误，整组 sanity 都失去对照价值。

**用法**: 复制 `--- PROMPT BEGIN ---` 与 `--- PROMPT END ---` 之间的全部内容，连同附件文件一并粘贴。**不要改 prompt 措辞**；语气有意 adversarial。

---

## 必须随 prompt 一并提供给外部 AI 的附件

| 路径 | 内容 | 为什么需要 |
|---|---|---|
| `review/0505/local/ANALYSIS.md` | V7/V8 实验设计 & 决策矩阵 | 提供上下文；reviewer 已被告知不重审此 |
| `review/0505/local/configs/V7_gronwall_raw.yaml` | V7 当前配置（max_steps=50000，含锁定的 schedule 字段） | 当前状态；D 提议改 max_steps=100000 |
| `review/0505/local/configs/V8_no_image_aux.yaml` | V8 当前配置（max_steps=50000） | 当前状态；D 提议改 max_steps=100000 |
| `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 200K 配置 | 提供 baseline 对照；reviewer 验证 V7@step k(0≤k≤50K) 与 V6 schedule 一致 |
| `train_first_hop.py` 行段（**全部贴到 prompt 里**）：<br>• L40-55（`set_seed`, deterministic 设置）<br>• L240-275（DataLoader / RandomSampler / WeightedRandomSampler 构造）<br>• L381-470（`resolve_rollout_lambda`, `_compute_sigma_dt_normalizers`）<br>• L1255-1290（`save_checkpoint`：含 rng_state, ema, optimizer, scaler）<br>• L1294-1350（`main`：set_seed → build_dataloaders 顺序）<br>• L1480-1580（`_resolve_schedule_steps`, rollout/lr 调度解析）<br>• L1726-1900（resume 路径：load model/optimizer/scaler/rng_state/ema, strict_resume_compat 检查）<br>• L2065-2080（loss 公式：`total = pair·loss_pair + lambda_roll·loss_roll + lambda_img·loss_img + ...`）<br>• L2456-2480（`save_interval`, ckpt 写盘点）| 让 reviewer 不信我的话——只信代码 |
| `pet_lr/rollout_first_hop.py` L40-130 | `rollout_multistep_losses_first_hop` 的全部源码（含 `step_weights`、`step_normalizers`、最终的 `total = (stacked * w).sum() / w.sum()`） | 让 reviewer 验证 step_weights 在 lambda_roll=0 时是否真的不影响梯度 |
| `review/0505/operator/v6_v61_training_and_fullval_status_20260505.md` | V6 实测 best.pt vs last.pt 全集得分 + checkpoint 路径列表 | 验证 V6@40K / V6@100K ckpt 是否真的存在 |

---

## --- PROMPT BEGIN ---

You are a senior reviewer (top conferences: NeurIPS / ICML / MICCAI) specializing in **bit-deterministic training reproducibility**, **DataLoader RNG state semantics**, **EMA / optimizer momentum continuity across resume**, and **schedule alignment in multi-phase ablations**. Your reputation depends on catching subtle non-determinism bugs that turn paired ablations into apples-vs-oranges comparisons. Assume the authors are honest but possibly self-deceiving and possibly under time pressure.

You are reviewing a **revised execution plan** ("Plan D") for an existing PET-image latent-transport ablation study. The underlying experiment design (V7 = Grönwall closed-form `step_weights`; V8 = remove `image_aux`) was peer-reviewed in a previous round. **Do not re-litigate that design.** Your job is exclusively to assess Plan D's claim that it produces a fair paired comparison **at lower GPU cost** than naïve from-scratch baselines.

---

### Background you can take as given (do not audit)

- 4-hop latent-domain cascade `D50 → D20 → D10 → D4 → NORMAL`. Backbone frozen DINOv2 + ViT-MAE decoder + LoRA + per-hop residual head + pixel-forcing branch (only at hop0).
- V6 baseline: 200K steps, `step_weights = [0.5, 2.0, 1.5, 1.0]`, `image_aux.enabled=true` with `lambda_img` ramp `0 → 0.04` and `image_aux.warmup_ratio=0.0` (so image_aux is non-zero from step 0). `seed=42`, `deterministic=true` (warn-only). save_interval=20000 → V6 has on-disk checkpoints at steps {20K, 40K, 60K, 80K, 100K, 120K, 140K, 160K, 180K, 200K}.
- V7: V6 baseline with **only** `step_weights` swapped to closed-form `[0.6106, 2.0, 2.7041, 2.0423]`. σ-normalize OFF. Everything else identical.
- V8: V6 baseline with **only** `image_aux.enabled=false`. Everything else identical.
- Rollout schedule (V6 200K main): `warmup_steps=50000, ramp_steps=100000, lambda_start=0, lambda_end=4` (linear). LR cosine total 200K, warmup 30K (15%).
- The previous reviewer round forced the authors to **lock V7/V8 schedules to V6-200K absolute values** via explicit `rollout.warmup_steps=50000`, `rollout.ramp_steps=100000`, `lr_schedule.total_steps_override=200000`. This means V7/V8 at any step `k` (during their independent run) sees the same `lambda_roll(k)` and `lr(k)` as V6 at step `k`.

---

### Plan D — exactly what is proposed

| Arm | Init | max_steps | Key training window | `lambda_roll` at endpoint |
|---|---|---|---|---|
| **V7** | `--resume /data_2/.../V6/.../step_040000.pt` (V6's checkpoint at step 40000) | 100000 | step 40K → 100K (60K extra GPU-steps) | 2.0 (ramp midpoint, since ramp is 50K→150K) |
| **V8** | from scratch (cannot resume — see below) | 100000 | step 0 → 100K (100K GPU-steps) | 2.0 |
| Anchor | V6@step 100000 ckpt (already on disk) — full-val evaluation only | — | (no training) | 2.0 |

**Total GPU cost**: 160K steps vs the original "naïve from-scratch 50K each = 100K total" plan (already rejected because at step 50K `lambda_roll=0`, leaving the `step_weights` change inert), and vs "from-scratch 100K each = 200K total" (Plan E, the safe fallback).

V8 cannot resume from V6@40K because V6 includes `image_aux` from step 0 (`image_aux.warmup_ratio=0.0`, `lambda_start=0.04`); V6's optimizer / EMA / model state at step 40K is already "image-aux-trained." V8 must be from-scratch to be a clean ablation.

V7 **can** resume from V6@40K because the only V6→V7 difference is `step_weights`, and (claim 1 below) `step_weights` does not affect gradients in the `lambda_roll=0` warmup region — so V7's step 0–50K is equivalent to V6's step 0–50K, and we can borrow V6's already-trained 40K-step state.

---

### The five code-grounded factual claims Plan D rests on

The authors require Plan D to be valid **only if all five claims hold**. Audit each one against the attached source code. Do **not** accept any claim merely because the authors stated it; verify in the code.

#### Claim 1 — `step_weights` are inert when `lambda_roll == 0`

The authors point to:
- `pet_lr/rollout_first_hop.py:117`: `w = torch.tensor(step_weights, ...); total = (stacked * w).sum() / w.sum().clamp_min(1e-8)`
- `train_first_hop.py:2071`: `total_loss = pair_loss_weight * pair_losses["total"] + rollout_losses["lambda_roll"] * rollout_losses["loss_total"] + ...`

They claim: when `lambda_roll == 0` (literal `0.0`, returned by `get_linear_schedule_value` at any `global_step < warmup_steps`), the entire rollout term contributes `0.0` to `total_loss`, and therefore `step_weights` (which only enter `loss_total`) cannot affect `total_loss` or any gradient.

**Audit**: Is the multiplication `lambda_roll * loss_total` the *only* path by which `step_weights` flows into the optimizer's gradient buffer? Specifically:
- (a) Is `loss_total` ever used outside this multiplication for anything that touches `optimizer.step()` (e.g. metric-driven early stopping that triggers parameter updates, watchdogs that modify gradients, gate clipping that conditions on `step_weights`)?
- (b) Are `step_weights` ever used in the **forward** pass of `model(...)` (e.g. as conditioning, as part of a learned attention mask, in `predict_latent_step`)?
- (c) Does the rollout forward pass consume RNG (e.g. dropout, stochastic depth, attention randomness) **before** the `lambda_roll * loss_total` multiplication? If yes, then even when the loss is `0.0`, the RNG state has been advanced, causing downstream RNG drift between V6 and a hypothetical V7 trained from-scratch with same seed.
- (d) Are `step_weights` ever logged in a way that affects the metric used for `best.pt` selection, which in turn could affect a watchdog that triggers a different training path?

If any of (a)–(d) is true, Claim 1 is **false** and Plan D's V7-resume scheme produces a model state that is *not* equivalent to V6@step50K.

#### Claim 2 — V6's checkpoints at step 40000 and 100000 actually exist on disk

V6 yaml has `save_interval: 20000`. `train_first_hop.py:2456` shows: `if step % save_interval == 0: save_checkpoint(..., f"step_{step:06d}.pt")`. So V6's run should have produced `step_020000.pt`, `step_040000.pt`, …, `step_200000.pt`.

**Audit**: Inspect `train_first_hop.py:1605-1620` (or wherever `save_interval` is validated and the periodic-save branch is gated). Are there conditions under which `step_040000.pt` could have been **skipped** (e.g. `save_interval==0` disables it; first save happens at `step_save_first` rather than at `step % save_interval`; an exception during save fails silently)? Is the periodic save guarded by EMA-context-manager that could fail and silently not write?

If V6@40K is missing, Plan D collapses to Plan E (from-scratch 100K each). The authors must confirm existence with `ls` before commit, but you should flag any code-level reason to doubt the file is on disk.

#### Claim 3 — Resume restores enough state for trajectory-equivalent continuation

The authors point to `train_first_hop.py:1255-1290` (save) and `train_first_hop.py:1726-1900` (load). The saved checkpoint contains: `step`, `model.state_dict()`, `optimizer.state_dict()`, `scaler.state_dict()`, `ema.state_dict()`, `rng_state` (python + numpy + torch + cuda), and various best-metric / signature fields. Resume restores all of these.

**Audit**: For V7 to produce the same model state at step 50K that V6 produced at step 50K (during the lambda_roll=0 region, granting Claim 1), what state needs to be identical? Cross-check what is **not** saved/restored:

- (a) **DataLoader iterator state**: `WeightedRandomSampler.__iter__` and `RandomSampler(replacement=True).__iter__` consume RNG when **the iterator is created**, not per-batch. `DataLoader` iterators are not picklable. Resume creates fresh iterators. Specifically, `train_first_hop.py:240-275` shows `WeightedRandomSampler(num_samples=len(train_set.index))` for main_train and `RandomSampler(replacement=True, num_samples=max_steps * hop0_batch_size)` for hop0_train. **Critical question**: does `num_samples=max_steps * hop0_batch_size` mean V7 (max_steps=100000) creates a different-length sample sequence than V6 (max_steps=200000) starting from the same RNG state? If yes, even with identical RNG state, V7's batch indices at step 40K-50K differ from V6's batch indices at step 40K-50K. This breaks "trajectory-equivalent continuation" even under Claim 1.
- (b) **CUDA cuBLAS heuristic state**: PyTorch warn-only deterministic mode does not guarantee bit-identity across runs. The σ-normalize A/B sanity FAILED (~10% drift over 50K steps under fp32 + warn-only) — see attached operator reply. Does the same drift apply to V7-resume vs V6 in the lambda_roll=0 region?
- (c) **EMA decay step counter**: `ema.step` is loaded from ckpt. But EMA decay is `decay = min(0.9999, (1+step)/(10+step))` or similar warmup formula in `torch_ema`. Inspect the EMA library's resume semantics. If V6 EMA at step 40K has `ema.step=40000` (consistent with global step), and V7 resume restores this, then V7's first EMA update at step 40001 uses the correct decay. Verify.
- (d) **Optimizer momentum / Adam moments**: AdamW's `exp_avg` and `exp_avg_sq` are saved in `optimizer.state_dict()`. After restore, the next `optimizer.step()` should use these moments correctly. Verify there's no "warmup" of momentum that depends on step counter inside the optimizer.
- (e) **LR scheduler state**: the authors use a custom cosine LR schedule via `get_linear_schedule_value`-like helpers (not a PyTorch `LRScheduler` object that needs `state_dict()`). LR is recomputed each step from `global_step` and `lr_total_steps`. Verify this is parameter-free and only depends on `global_step` (which is `start_step` after resume).

If any of (a)–(e) is broken, V7-resume produces a model that diverges from V6 even in the lambda_roll=0 region, breaking the "free first 50K steps" benefit of Plan D.

#### Claim 4 — Resume's `strict_resume_compat` check accepts the V7 config

`train_first_hop.py:1797-1830` shows the strict-compat metadata fields checked: `["target_normalize", "rollout_path", "first_hop_pixel_enabled"]`. Plus `best_metric_signature` is checked separately.

**Audit**: Does V7's yaml change anything in these checks?
- V7 vs V6: `step_weights` differs. Is `step_weights` part of `best_metric_signature`? (Inspect `build_best_metric_signature` around L1212.)
- V7 vs V6: schedule fields (`warmup_steps`, `ramp_steps`, `total_steps_override`) added explicitly to V7 yaml. Are these compared against ckpt metadata? (They probably aren't — `_resolve_schedule_steps` is run from yaml, not ckpt — but verify.)
- V7 vs V6: `best_select_full_eval_interval=5000` added to V7 yaml. Does this change `best_metric_signature`? (The authors' previous trainer fix `fc7446d` mentions this is gated to "embed `best_select_full_eval_interval` only when >0 (V6/V6.1 backward compat)." So V6 ckpt's signature does NOT include it; V7 ckpt's signature DOES include it. This causes signature mismatch on resume.)

If the signature mismatches, resume requires `resume_allow_metric_mismatch=true`. The current V7 yaml has `resume_allow_metric_mismatch: false`. **This is an actionable bug.** Is the fix safe (set to `true` for V7) or does it indicate a deeper protocol violation?

#### Claim 5 — At step 100000 with locked schedule, V7/V8/V6 all see `lambda_roll = 2.0` and `lr ≈ 0.97 × base × cos_progress(50K/170K)`

V7/V8/V6 all use the same locked schedule. At step k=100000:
- rollout warmup ends at 50000, ramp 50K→150K → `lambda_roll(100000) = 4 × (100000-50000)/100000 = 2.0` ✓
- lr cosine: warmup 0→30K, then cosine from 30K→200K. At step 100000: cosine progress = (100000-30000)/(200000-30000) = 70/170 ≈ 0.412 → lr = min_lr + (base − min_lr) × 0.5 × (1 + cos(0.412 π)) = …

**Audit**: Verify the LR schedule formula in `train_first_hop.py` matches the cosine-with-warmup formula above. Confirm the LR is identical for V7 (max_steps=100000, total_steps_override=200000) and V6 (max_steps=200000) at step k=100000.

---

### Your task — answer each of the eight questions in order

For each question, give a clear verdict: **PASS** (Plan D is safe on this point), **CONCERN** (specify the issue and severity), or **BLOCKER** (specify the fix required before Plan D can run). Cite line numbers in the attached code where you base your verdict.

1. **Claim 1 audit (step_weights inertness)** — pass / concern / blocker?
2. **Claim 2 audit (V6@40K and V6@100K on-disk)** — what code-level conditions could prevent these files from existing? What should the operator verify before running?
3. **Claim 3 audit (resume trajectory continuity)** — go through (a) DataLoader RNG, (b) CUDA non-determinism, (c) EMA decay step, (d) optimizer momentum, (e) LR scheduler. Which ones break Plan D? For each, propose the **simplest fix** that doesn't degrade Plan D's GPU savings.
4. **Claim 4 audit (`best_metric_signature` mismatch)** — does adding `best_select_full_eval_interval=5000` to V7 yaml cause signature mismatch on resume from V6@40K (where V6's ckpt signature predates this field)? If yes, what's the right fix: (i) set `resume_allow_metric_mismatch=true` for V7, (ii) compute V6's would-be signature offline and check, or (iii) something else?
5. **Claim 5 audit (LR / lambda_roll match at endpoint)** — verify the math. Confirm `lr(V7@100K) == lr(V6@100K)` symbolically.
6. **Holistic risk assessment** — given your answers to Q1–Q5, is Plan D **viable** (after fixes), **partially viable** (V7-resume works but V8 has its own issue), or **not viable** (must fall back to Plan E)? Assume the authors will apply all fixes you specify in Q1–Q5.
7. **What metric should the operator collect at step 50000 of V7** to verify that V7-resume actually matches V6 in the lambda_roll=0 region? Specifically: name 3 lightweight scalars from `metrics.jsonl` whose values at step 50000 should match `V6.metrics.jsonl[step==50000]` to within X% (you specify X), failing which V7 has diverged from V6 in the warmup region and Plan D's premise has collapsed.
8. **Pre-registration completeness** — given Plan D's 100K training horizon and decision matrix relying on Cohen's d ≥ 0.10, is the experiment statistically powered to falsify the H0 / H1 / H2 of either V7 or V8? Or does the 0.026 dB / Cohen's d ≈ 0.056 V6-vs-V6.1 historical noise floor still dominate? If the latter, what is the minimum training horizon you would commit to before running anything?

---

### Output format

Return your answer as a single markdown document with this exact structure:

```markdown
# Plan D — Reviewer audit (REVIEWER_NAME)

## Summary
{2-3 sentences. Top-line verdict: GO / GO-WITH-FIXES / NO-GO. List the fixes.}

## Q1 — step_weights inertness when lambda_roll=0
**Verdict**: PASS / CONCERN / BLOCKER
{evidence with line numbers}

## Q2 — V6@40K and V6@100K existence
{...}

## Q3 — Resume continuity
### Q3(a) DataLoader RNG
### Q3(b) CUDA non-determinism
### Q3(c) EMA decay step
### Q3(d) Optimizer momentum
### Q3(e) LR scheduler

## Q4 — best_metric_signature
{...}

## Q5 — LR / lambda_roll endpoint match
{...}

## Q6 — Holistic verdict
{GO / GO-WITH-FIXES / NO-GO; condition list}

## Q7 — V7@step50K reconciliation metric
{3 scalars; tolerance%; failure action}

## Q8 — Statistical power
{minimum horizon recommendation}

## What I would have done differently
{1 paragraph}
```

Be ruthless. If Plan D has a fundamental flaw not caught by the eight questions, surface it under "What I would have done differently."

## --- PROMPT END ---
