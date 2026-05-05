# Plan F — V7/V8 from-scratch to 160K (supersedes Plan D, supersedes earlier 150K endpoint via Fix F)

**Status**: APPROVED 2026-05-05; cross-AI peer review (Agent1 + Agent2) GO-WITH-FIXES; **Fix F applied 2026-05-05** (endpoint 150K → 160K because V6@step_150000.pt missing on disk — V6 `save_interval=20000`); pre-flight checklist pending; **GPU not yet launched**.
**Author**: jiaxiang + Copilot
**Supersedes**: Plan D (V7 resume from V6@40K + V8 from-scratch to 100K) — abandoned.
**Decision basis**: Two external peer reviews of Plan D (GitHub Copilot Sonnet 4.5 NO-GO; Claude Opus 4.7 GO-WITH-FIXES) + local code verification + cross-AI peer review of Plan F itself (Agent1 + Agent2, 2026-05-05) producing 5 approved fixes (Fix F = 150K → 160K, Fix 2 = GPU UUID lock + step-1000 sentinel, Fix 3 = primary endpoint pre-registration, Fix 4 = V8 RNG-invariance diag, Fix 5 = Bonferroni metric family).

---

## 1. Why Plan D was abandoned

Plan D proposed `--resume /data_2/.../V6/.../step_040000.pt` to save 40K GPU-steps on V7. Local code audit (see [`PEER_REVIEW_PROMPT_PLAN_D.md`](../0505/local/PEER_REVIEW_PROMPT_PLAN_D.md) and the two reviewer reports) established three independent BLOCKERS that no fix can simultaneously remove without forfeiting Plan D's GPU savings:

| # | Blocker | Code evidence | Why no fix saves Plan D |
|---|---|---|---|
| 1 | **EMA-swap checkpoint** — every periodic checkpoint is saved inside `with ema.average_parameters(): save_checkpoint(...)`. `ckpt["model"]` therefore contains EMA-swapped weights, while `ckpt["optimizer"]` holds Adam moments accumulated against the **raw** weight trajectory. Resuming feeds EMA weights into an optimizer state that expects raw weights — an inconsistent state V6 never traversed. | [`train_first_hop.py:2456-2477`](../../train_first_hop.py#L2456-L2477) (periodic save in EMA context); [`train_first_hop.py:1259`](../../train_first_hop.py#L1259) (`"model": model.state_dict()` inside `save_checkpoint`); same pattern at lines 2478, 2523, 2562, 2592, 2606, 2644 (best.pt, best_d1.pt, full-val best.pt, last.pt). **No raw-weight checkpoint exists anywhere on disk**. | Producing a raw-weight V6@40K requires re-training V6 from scratch with a patched trainer — that's 40K GPU-steps spent just to recover the savings, plus a different sampler RNG trajectory. |
| 2 | **DataLoader sampler RNG discontinuity** — `RandomSampler(replacement=True, num_samples=max_steps × hop0_batch_size)` and `WeightedRandomSampler` consume torch RNG at `iter()`-creation time, not per-batch. Resume restores `rng_state` to its V6@40K value, then constructs a fresh iterator that draws `100000 × bs` ints from that state. V7's batches at synthetic steps 40001–50000 are therefore **completely different** from V6's actual batches at 40001–50000. | [`train_first_hop.py:241-262`](../../train_first_hop.py#L241-L262) (sampler construction); [`train_first_hop.py:1861`](../../train_first_hop.py#L1861) (RNG restore); iter creation post-resume in main loop. | Plan D's premise was "V7 step 0–50K equivalent to V6 step 0–50K." That is bit-false. Reframing as "warm-start + 10K continuation" (Agent2's suggestion) keeps Plan D structurally honest but requires a step-50K reconciliation gate that, on top of blockers 1 and 3, almost certainly trips and forces fallback to from-scratch — wasting V7's 60K GPU-steps. |
| 3 | **`best_metric_signature` mismatch** — V7 yaml sets `best_select_full_eval_interval: 5000` (trainer commit `fc7446d` embeds this in the signature only when `>0`). V6@40K ckpt's signature predates the field. Resume currently has `resume_allow_metric_mismatch: false` and will hard-fail at signature check. | [`train_first_hop.py:1228-1234`](../../train_first_hop.py#L1228-L1234) (signature build, gated `>0`); [`V7_gronwall_raw.yaml:152`](../0505/local/configs/V7_gronwall_raw.yaml#L152), [`:175`](../0505/local/configs/V7_gronwall_raw.yaml#L175) | Setting `resume_allow_metric_mismatch: true` is mechanically safe but triggers the warning path that resets `best_val=inf`, breaking best.pt continuity — another departure from "V7 inherits V6's first 40K of training." |

Plan F bypasses all three blockers by removing the resume path entirely.

---

## 2. Plan F — design specification

| Arm | Init | max_steps | Schedule (locked to V6-200K absolute values) | Endpoint state |
|---|---|---|---|---|
| **V7** (Grönwall closed-form raw `step_weights`) | from scratch, `seed=42` | **160000** | `rollout.warmup_steps=50000`, `rollout.ramp_steps=100000` (ramp ends at step 150K), `lr_schedule.total_steps_override=200000` | step 160K = `lambda_roll=4.0` (ramp end clamped, plateau), `lr ≈ 1.22e-5 ≈ 0.15 × base_lr` |
| **V8** (V6 minus `image_aux`) | from scratch, `seed=42` | **160000** | identical schedule lock | identical endpoint state |
| **V6-seed1337** (V6 noise-baseline replica) | from scratch, **`seed=1337`** | **160000** | identical schedule lock | identical endpoint state — bit-equivalent to V6-seed42 except for seed |
| **Anchor** (V6-seed42 baseline) | none — already trained | 200000 (existing) | original V6 schedule | step 160K = exact same `(lambda_roll, lr)` as V7/V8/V6-seed1337 endpoint by schedule-lock construction; **V6@step_160000.pt EXISTS on disk** (V6 save_interval=20000 → grid {20K,40K,...,200K}) |

**GPU cost**: V7 160K + V8 160K + V6-seed1337 160K = **480K total** (post-Q10-A; was 320K post-Fix F two-arm). The +160K (V6-seed1337) is the cost of grounding the PRIMARY threshold in an empirical pure-noise estimate `d_pure` instead of the contaminated V6 vs V6.1 historical `d = 0.067` (recomputed 2026-05-06 from per-slice CSVs in `review/0505/operator/artifacts/`, full-val n=7403, mse_NORMAL, best.pt; |t|=5.83, p≈5.6e-9). See §4 *Pre-registration → Adaptive PRIMARY threshold* for how `d_pure` re-defines the decision boundary.

**Why V6-seed1337 was added (Q10-A, 2026-05-05)**: the historical V6 vs V6.1 paired Cohen's d (recomputed 2026-05-06 from local per-slice CSVs: **|d|=0.067 on best.pt, |d|=0.067 on last.pt**, n=7403, t=−5.83) used to justify Plan F's `d ≥ 0.10` threshold (1.8× margin) is contaminated by **(a)** V6.1's rollout-floor algorithm change vs V6 and **(b)** checkpoint-selection bias (each arm picked its own best.pt at different steps). Without separating these from **(c)** pure run-to-run RNG noise, the 1.8× margin is hand-wavy. V6-seed1337 closes this gap: same algorithm, same step (160K = last.pt), only seed differs → `d_pure = paired_cohen_d(V6-seed42@step_160000, V6-seed1337@step_160000)` is a clean empirical noise floor. The PRIMARY threshold is then `d ≥ max(0.10, 1.8 × d_pure)`. User-approved 2026-05-05 after power analysis showed the test is over-powered (MDE @ 80% power = 0.033, 3× below threshold) and the binding constraint is the noise floor, not statistical power.

**Why 160K and not 150K (Fix F, cross-AI peer review BLOCKER)**:
- V6 was trained with `save_interval=20000` (cf. `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` + train_first_hop.py:2456-2480). V6 step ckpts therefore exist only at {20K, 40K, ..., 200K}. **`step_150000.pt` does not exist on disk**.
- Closest existing V6 ckpts on the rollout-plateau region: `step_140000.pt` (lambda_roll=3.6, ramp not yet complete) and `step_160000.pt` (lambda_roll=4.0 clamped, post-plateau).
- Choosing 160K preserves the original Plan F rationale (compare V7/V8 vs V6 at full lambda_roll=4.0 rollout pressure) and gives both arms an identical `(lambda_roll=4.0, lr=1.22e-5)` endpoint state. The two reviewers' 150K floor (full rollout pressure) is upheld.

**Why 160K and not 100K**:
- At step 100K, `lambda_roll=2.0` (only halfway up the ramp), and the (contaminated) V6/V6.1 noise floor `d = 0.067` likely dominates a Plan D `d ≥ 0.10` decision threshold at half-ramp.
- At step 160K, `lambda_roll` has reached its terminal value `4.0` (clamped) — the rollout pressure has plateaued. Both V7 step_weights modification and V8 image_aux removal have been tested under full rollout pressure.
- 200K (= Plan E) gains a small additional convergence margin but doubles the marginal cost relative to 160K. Reviewers' consensus floor was 150K (now 160K post-Fix F); 200K was "preferred but not required." Choose 160K to release decision faster and keep Plan E in reserve if 160K results are ambiguous.

---

## 3. Anchors

Plan F now has TWO anchor measurements: the existing V6-seed42 baseline (for the V7 / V8 effect-size comparison), and the new V6-seed1337 noise-baseline (for the adaptive PRIMARY threshold).

### 3.1 V6-seed42 @ step 160000 (existing, post Fix F)

V7@160K, V8@160K, and V6-seed1337@160K are all compared against **V6-seed42@step_160000.pt** full-val scores.

> **Fix F note**: original Plan F used `step 150000` as endpoint, but V6 was trained with `save_interval=20000` so `step_150000.pt` does NOT exist on disk — cross-AI peer review BLOCKER. `step_160000.pt` exists and shares the same `(lambda_roll=4.0, lr)` clamp-plateau state, so the experimental logic is unchanged.

**Pre-flight check (operator action required before launch)**:

```bash
# Confirm V6@160K checkpoint exists
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt

# Confirm V6@160K full-val has been run; if not, run it first.
# Reference: review/0505/operator/v6_v61_training_and_fullval_status_20260505.md
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/.../v6_step_160000_fullval.json
```

If V6@160K full-val has NOT been computed yet, **the anchor must be produced first** — otherwise V7/V8 land with no comparison target. This is a pure eval cost (no training), single-GPU run of `eval_first_hop_224_clip3.py` against `step_160000.pt`. Estimated cost: ~7 min (full val once, per V6 prior measurements).

### 3.2 V6-seed1337 @ step 160000 (new, Q10-A noise-baseline)

A second V6 replica is trained from-scratch to 160K with **only the seed changed** (42 → 1337). Everything else — backbone init ckpt, data, schedule (locked), `step_weights=[0.5,2.0,1.5,1.0]`, `image_aux.enabled=true`, `pair_weight=15.0`, `dead_branch_watch_steps=10000` — is bit-identical to V6-seed42 yaml. The yaml deliberately omits `best_select_full_eval_interval` because V6-seed42 didn't have it; adding it would consume extra RNG via 32 full-eval passes and break the seed-only-difference invariant.

See [`review/0505/local/configs/V6_seed1337.yaml`](../0505/local/configs/V6_seed1337.yaml) for the full config + header rationale.

Post-train, `d_pure` is computed by:

```bash
# evaluate both V6 replicas at step_160000.pt on the SAME val set
bash review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
     --ckpt /data_2/.../first_hop_224_v6_transport_first/step_160000.pt \
     --tag V6_seed42_step160k --emit-per-slice-json
bash review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
     --ckpt /data_2/.../V6_NOISE/run/first_hop_224_v6_seed1337/step_160000.pt \
     --tag V6_seed1337_step160k --emit-per-slice-json

# d_pure = paired Cohen's d on val_chain_normal_mse, n=7403 unique slices
python3 review/0505/local/scripts/compute_d_pure.py \
     --a V6_seed42_step160k.per_slice.json \
     --b V6_seed1337_step160k.per_slice.json \
     --metric val_chain_normal_mse \
     --out review/0505/local/runs/V6_NOISE/d_pure.json
```

Expected outcome:
- `d_pure ≤ 0.056` (PRIMARY threshold stays at `d ≥ 0.10`, no change to existing decision logic).
- `0.056 < d_pure ≤ 0.067` (PRIMARY threshold = `max(0.10, 1.8 × d_pure)` ∈ [0.101, 0.121]; minor tightening).
- `d_pure > 0.067` (would indicate the V6 vs V6.1 contaminated-estimate floor was an underestimate) → PRIMARY threshold raises to `d ≥ 1.8 × d_pure`. V7/V8 may flip from "H1 confirmed" to "inconclusive" if their effect was marginal.
- `d_pure ≈ 0.10 or higher` (worst case) → V7/V8 effect would need `d ≥ 0.18` to claim improvement. Would invalidate any marginal V7/V8 win and potentially require Plan E (200K) follow-up.

Cost: pure eval after V6-seed1337 finishes; no separate GPU step. The 160K training run is the cost.

> **Note**: `compute_d_pure.py` is a thin wrapper around `scipy.stats` paired-t and Cohen's d on per-slice deltas. Implementation deferred until V6-seed1337 finishes — straightforward enough that it does not block launch.

---

## 4. Decision matrix (endpoint = step 160K, full-val basis, post Fix F)

Identical structure to the V6 vs V6.1 comparison; endpoint moves from 150K to 160K (Fix F). Comparison metric: `val_multi_objective` (chain D20+D10+D4+NORMAL MSE weighted as in V6 yaml).

### Pre-registration (Fix 3 + Fix 5)

To prevent winner's-curse bias from full-val max-search across 32 candidate ckpts (160K / 5K full-eval interval), and to control family-wise error rate (FWER) across 5 chain metrics, Plan F pre-registers:

**Primary endpoint (Fix 3 + Q10-A adaptive threshold)**:
- Metric: `val_chain_normal_mse` on V7@step_160000.pt vs V6-seed42@step_160000.pt full-val (same for V8).
- Why `step_160000.pt` rather than `best.pt`: under V7/V8 yaml `save_interval: 10000`, `step_160000.pt` = `last.pt` = the unique deterministic endpoint, **NO selection bias**. By contrast, `best.pt` is the argmax over 32 full-val checkpoints — V7's `best.pt` may beat V6's `best.pt` purely from 32-way max-search inflation, even if step_weights had no real effect.
- **Adaptive threshold (Q10-A, 2026-05-05)**: paired Cohen's d ≥ **`max(0.10, 1.8 × d_pure)`** AND chain_normal_mse Δ ≥ 5%, α = 0.05.
  - `d_pure` is the paired Cohen's d between **V6-seed42@step_160000.pt** and **V6-seed1337@step_160000.pt** on `val_chain_normal_mse` over the 7403-slice val set (see §3.2). It is a pure run-to-run RNG noise floor with NO algorithmic / selection contamination.
  - The lower bound `0.10` is preserved as a safety floor in case `d_pure` happens to be very small (e.g., < 0.03), to avoid claiming significance from numerically-tiny effects.
  - The upper-bound multiplier `1.8×` is the same heuristic margin used in the original Plan F ("effect must be at least ~2 noise standard deviations"), now applied to a clean noise estimate instead of the contaminated V6 vs V6.1 = 0.067.
  - This threshold is **finalized** the moment V6-seed1337 completes its 160K run and `d_pure` is computed. V7 / V8 verdict tables (below) are then read with the final threshold value substituted for `0.10`.
  - Power note: at n=7403, 80%-power MDE = 0.033 (plain) / 0.039 (Bonferroni); the test is severely over-powered relative to even a `d ≥ 0.18` worst-case threshold. This threshold is set by the **noise floor**, not by statistical power.

**Secondary endpoint (Fix 3)**:
- Metric: `val_chain_normal_mse` on V7@best.pt vs V6@step_160000.pt.
- Threshold: empirical d ≥ 0.15 (tightened from 0.10 to compensate for 32-way max-search inflation).
- Use: supplementary evidence after primary; **cannot reverse primary decision**.

**Multi-metric family (Fix 5)**:

In addition to primary, four secondary chain metrics are reported. FWER controlled via Bonferroni:

| Metric | Role | Per-test α | Threshold |
|---|---|---|---|
| `val_chain_normal_mse` | **PRIMARY** | 0.05 | d ≥ **`max(0.10, 1.8 × d_pure)`** AND Δ ≥ 5% |
| `val_chain_d4_mse`     | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |
| `val_chain_d10_mse`    | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |
| `val_chain_d20_mse`    | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |
| `val_pair_total`       | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |

Note: Bonferroni applied within secondary family. Primary is independent (α=0.05). A secondary-only positive does not reverse primary; promote to next round (200K main run) instead. The secondary `2.7×` multiplier preserves the original `0.15 / 0.10 = 1.5×` ratio between secondary and primary thresholds.

### Decision verdicts (primary basis)

Let `d_thr = max(0.10, 1.8 × d_pure)` (computed from V6-seed1337 anchor, see §3.2).

| Outcome on `val_chain_normal_mse` (paired Cohen's d, V7 vs V6-seed42@step_160000.pt) | V7 verdict | V8 verdict | Decision |
|---|---|---|---|
| **V7 / V8 favored**: `d ≤ −d_thr` AND chain_normal_mse improved ≥ 5% | H1 confirmed (Grönwall closed-form / image_aux removal helps) | H1 confirmed | Promote V7 (resp. V8) to 200K main run as new baseline |
| **Inconclusive**: `−d_thr < d < +d_thr` | H0: framing-equivalent | H0: image_aux is a wash on chain | Keep V6; document V7/V8 as non-improvements; consider 200K extension only if pixel-level diagnostics motivate |
| **V6 favored**: `d ≥ +d_thr` | H2: V7 step_weights worse | H2: image_aux is doing real work | Drop V7 (resp. V8) candidate; defend V6 |

**Cohen's d note**: paired `d` is computed across the V6@160K full-val slice set (n=7403 unique slices). With per-slice CV ≈ 8–15%, paired SE on relative `chain_normal_mse` difference ≈ 0.15–0.25%. The `d_pure`-based threshold replaces the earlier hand-wavy `1.8 × 0.067 ≈ 0.12 → floored to 0.10` heuristic (which conflated algorithmic, selection, and noise sources) with an empirically grounded **same-algo, same-step, seed-only** noise estimate. If `d_pure ≤ 0.056`, the threshold remains `d ≥ 0.10` and existing Plan F decision logic is unchanged. **Empirical anchor**: V6 vs V6.1 paired Cohen's d on `mse_NORMAL` (full-val, n=7403) computed 2026-05-06 from `review/0505/operator/artifacts/v6_*_fullval_psnr_chain_mse_per_slice.csv`: |d|=0.0677 on best.pt (t=−5.83, p≈5.6e-9), |d|=0.0673 on last.pt (t=−5.79). This represents the upper bound on `d_pure` since V6/V6.1 also includes (a) algorithmic and (b) selection contamination.

**V8 has a known caveat that does NOT bind this 160K sanity** but blocks paper adoption: image_aux's stated purpose is decode-artifact suppression on the hop0 pixel branch, which **chain MSE cannot measure**. Even if V8 wins on chain MSE, a separate qualitative decode-artifact study on a fixed val subset is required before V8 replaces V6.

---

## 5. Required yaml changes (V7 + V8) — APPLIED 2026-05-05; **Fix F adjustment 2026-05-05**

Both yamls were updated from `max_steps: 50000` (Plan F initial) to **`max_steps: 160000`** (post Fix F). Diff:

```diff
 training:
-  max_steps: 50000
+  max_steps: 160000     # Plan F + Fix F (was 150000 — V6@step_150000.pt missing)
```

Additional V8-specific change applied (Fix F update):

```diff
 training:
   dead_branch_watch:
-    dead_branch_watch_steps: 50000   # extended to 50000 to cover full sanity
+    dead_branch_watch_steps: 160000  # extended to 160000 to cover full Plan F + Fix F training window
```

**No other changes required**:
- Schedule lock fields (`warmup_steps=50000`, `ramp_steps=100000`, `total_steps_override=200000`) were already present in both yamls and remain valid for `max_steps=160000` (lambda_roll = 4.0 from step 150K onward, clamped beyond ramp).
- `best_select_full_eval_interval: 5000` correctly handled by trainer commit `fc7446d` (full-val every 5K steps writes best.pt; with max_steps=160000 → 32 candidates; cost ~5h overhead per run, acceptable).
- `save_interval: 10000` produces step ckpts at {10K, 20K, ..., 160K} — includes `step_160000.pt = last.pt`, the primary endpoint.
- `dead_branch_watch_steps: 10000` (V7, V6 default — image_aux still on, no risk) and `= 160000` (V8, full training window).
- `seed: 42` matches V6 — consumer of the same data pipeline.
- Launcher script `run_v7_v8_sanity.sh` default `SANITY_STEPS` updated 50000 → **160000** (Fix F) to prevent silent override of yaml `max_steps`.

**Verify before launch** (validated 2026-05-05 by `/tmp/plan_f_check.py`, all checks PASSED with original 150K; **must rerun with 160K updates**):
- yaml.safe_load on both V7/V8 yamls
- assert `max_steps == 160000`, `warmup_steps == 50000`, `ramp_steps == 100000`, `total_steps_override == 200000`, `best_select_full_eval_interval == 5000`, `seed == 42`, `resume_allow_metric_mismatch == False`
- assert V7 `step_weights == [0.6106, 2.0, 2.7041, 2.0423]`, V7 `image_aux.enabled == True`, V7 `dead_branch_watch_steps == 10000`
- assert V8 `step_weights == [0.5, 2.0, 1.5, 1.0]`, V8 `image_aux.enabled == False`, V8 `dead_branch_watch_steps == 160000`
- LR table cross-check: `lr(30000)=1.000×base, lr(50000)=0.97×base, lr(100000)=0.65×base, lr(150000)=0.22×base, lr(160000)=0.15×base`
- lambda_roll table cross-check: `lambda(50000)=0, lambda(100000)=2, lambda(150000)=4, lambda(160000)=4 (clamped)`
- `bash -n review/0505/local/scripts/run_v7_v8_sanity.sh` syntax-check (Fix F).

---

## 6. Required ANALYSIS.md changes — APPLIED 2026-05-05; **Fix F update 2026-05-05**

[`review/0505/local/ANALYSIS.md`](../0505/local/ANALYSIS.md) was updated from 50K sanity → 150K Plan F → **160K Plan F + Fix F** framing. Edits applied:

1. **Replaced 150K-Plan-F framing with 160K-Plan-F-plus-Fix-F framing** in the title ("V7 & V8 160K Plan-F Ablation") and headline summary.
2. **Updated decision matrix** (§6) endpoint references from "step 150K (V6 step 150000 anchor)" to "step 160K (V6 step 160000 anchor)". Numerical thresholds (`d ≥ 0.10`, 5% margin) unchanged.
3. **Plan D residue removed** from §4 + §7.
4. **Fixed LR magnitude annotation** in §7 caveat #2 ("schedule 对齐"). Per the locked schedule with `warmup_steps=30000`, `total_steps_override=200000`, `min_lr=2e-6`, `base_lr=8e-5`, the LR table now extends to step 160K.
5. **Added §7 caveat #7 "EMA-swap deferred ticket"** noting that V6 step ckpts are EMA-swapped artifacts unsuitable for training resume; this is the structural reason Plan F must train from scratch. See §7 below.
6. **Added §7 caveat #8 "DiT RNG-free verification"** documenting the local code grep that proved Agent2's V8 image_aux RNG concern (Q3) was a false alarm. Fix 4 (RNG-invariance diag) is belt-and-suspenders against future DiT modifications.
7. **Added §6 "Pre-registration" block** declaring `val_chain_normal_mse` on `step_160000.pt` (= last.pt, no selection bias) as PRIMARY endpoint (α=0.05); `best.pt` comparison as SECONDARY with adjusted threshold; secondary chain metrics under Bonferroni α=0.05/4=0.0125.
8. **Added §5 Stage 1.5 "step-1000 sentinel"** and Stage 3.0 "V8 RNG-invariance diag" sections per Fix 2 + Fix 4.

   | step | progress | cosine | lr | lr/base |
   |---:|---:|---:|---:|---:|
   |  30000 | 0     | 1.000 | 8.00e-5 | 1.000 (warmup ends) |
   |  50000 | 0.118 | 0.966 | 7.74e-5 | **0.97** |
   | 100000 | 0.412 | 0.637 | 5.17e-5 | 0.65 |
   | 150000 | 0.706 | 0.199 | 1.75e-5 | **0.22** |
   | **160000** | **0.765** | **0.131** | **1.22e-5** | **0.15** ← Plan F + Fix F endpoint |
   | 200000 | 1.000 | 0.000 | 2.00e-6 | 0.025 (V6 final) |

   The **identity claim** (`lr(V7@k) == lr(V6@k)` for all k under locked schedule) is unchanged and remains the central guarantee.
5. **Added §7 caveat #7 "EMA-swap deferred ticket"** noting that V6 step ckpts are EMA-swapped artifacts unsuitable for training resume; this is the structural reason Plan F must train from scratch. See §7 below.

---

## 7. Open issues (deferred — do NOT block Plan F)

### 7.1 EMA-swap checkpoint bug (trainer-level, deferred)

`train_first_hop.py` saves all checkpoints (step / best / best_d1 / last) inside `with ema.average_parameters(): save_checkpoint(...)`. The serialized `ckpt["model"]` is therefore the EMA-shadow weights, while `ckpt["optimizer"]` holds Adam moments for the **raw** training weights. Consequences:
- Inference / eval load is correct (uses EMA weights, which is what we want at inference).
- `--resume` is structurally broken: loading creates inconsistent (model=EMA, optimizer=raw-moments) state never traversed by the original training trajectory.

**Disposition**: track as standalone trainer ticket. Plan F does **not** depend on this being fixed. Future fix should save TWO model state dicts (`model_raw` + `model_ema`) and have `--resume` load `model_raw`. Not in scope for this milestone.

### 7.2 Pixel-domain V8 decode-artifact study (out of scope for Plan F)

Required before V8 can replace V6 if 160K shows V8 favorable on chain MSE. Separate qualitative protocol; not part of Plan F.

### 7.3 V7 200K main run

If 160K supports H1 (V7 favored), promote to 200K main run as new baseline candidate. Separate ROADMAP entry, not Plan F.

---

## 8. Audit trail

| Date | Document | Verdict |
|---|---|---|
| Earlier 2026-05-05 | [`PEER_REVIEW_PROMPT_V7_V8.md`](../0505/local/PEER_REVIEW_PROMPT_V7_V8.md) → 4 reviewers | Forced schedule lock, deleted Σw confound, decision basis = full-val + Cohen's d |
| 2026-05-05 | [`PEER_REVIEW_PROMPT_PLAN_D.md`](../0505/local/PEER_REVIEW_PROMPT_PLAN_D.md) → Copilot Sonnet 4.5 | NO-GO (EMA-swap, sampler RNG, signature) |
| 2026-05-05 | Same prompt → Claude Opus 4.7 | GO-WITH-FIXES (sampler RNG, signature, LR mag, Q7 gate, horizon ≥150K) |
| 2026-05-05 | Local code verification | Confirmed EMA-swap blocker via [`train_first_hop.py:2456-2477`](../../train_first_hop.py#L2456-L2477); confirmed V7 yaml has `best_select_full_eval_interval: 5000` (Agent2 misread); LR magnitude reconciled |
| 2026-05-05 | Cross-AI peer review of Plan F (Agent1 Sonnet 4.5 + Agent2) | GO-WITH-FIXES; consensus BLOCKER (V6@150K missing) + concerns (best.pt selection bias, multi-test FWER, GPU determinism, V8 RNG); 5 fixes approved by user (Fix F 150K→160K, Fix 2 GPU+sentinel, Fix 3 last.pt pre-registration, Fix 4 V8 RNG diag, Fix 5 Bonferroni) |
| 2026-05-05 | Power analysis on PRIMARY threshold | At n=7403 paired, MDE @ 80% power = 0.033 (α=0.05) / 0.039 (Bonferroni); threshold `d=0.10` is 3.07× above the power-needed minimum. Test is over-powered; threshold is governed by noise floor, not power. |
| 2026-05-05 | User Q10-A decision (after surfacing V6/V6.1 "best.pt selection bias" concern) | Approved adding **V6-seed1337** noise-baseline arm: same algo / same step / only seed differs → `d_pure` is a clean RNG-only noise floor; PRIMARY threshold becomes adaptive `d ≥ max(0.10, 1.8 × d_pure)`. Cost: +160K GPU (V7+V8+V6-seed1337 = 480K total, +50% over Plan F two-arm). |
| 2026-05-05 | This plan (Fix F + Q10-A applied) | 160K endpoint locked; 3-arm design; adaptive PRIMARY threshold pre-registered; pre-flight checklist extended for V6-seed1337 |
| 2026-05-06 | Multi-agent internal pre-review of `PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md` (6 agents: 4 reviewer-role + 2 prompt-quality-role) — see [`ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`](ROUND4_PRE_REVIEW_CONSENSUS_20260506.md) | Convergent findings: (1) `val_pair_total = 10.16%` falsifies any rollout-amplification-only diagnosis since pair_loss is structurally outside `mix_latent`; (2) same-process algebraic golden test (Option F) is the cheap decisive discriminator and should be P0 prerequisite; (3) `pair_v_std` is non-persistent buffer (verified at `pet_lr/model_first_hop.py:336`) so σ-normalize algebraic equivalence holds in exact arithmetic; (4) Plan F's `1.8×d_pure` threshold could inflate to 0.18-0.72 if `d_pure > 0.10` — contingency added as §11 below. |

---

## 9. Pre-launch checklist

Operator must confirm before any GPU launch:

### 9.0  V6-seed1337 noise-baseline (Q10-A, NEW 2026-05-05)

- [ ] V6-seed1337 yaml exists at `review/0505/local/configs/V6_seed1337.yaml`. Static-check passes (seed=1337, max_steps=160000, save_interval=10000, schedule lock at 50000/100000/200000, `image_aux.enabled=true`, `step_weights=[0.5,2.0,1.5,1.0]`, NO `best_select_full_eval_interval`).
- [ ] Output directory `${ABLATION_OUTPUT_ROOT}/V6_NOISE/` is empty (trainer enforces `require_fresh_output_dir: true`).
- [ ] V6-seed1337 launched on the same GPU UUID as V6-seed42, V7, V8 (Fix 2 invariant). May run in parallel with V7 and V8 if multiple GPUs are available, or sequentially after V8 — no dependency.
- [ ] After V6-seed1337 reaches step 160000, `step_160000.pt` (= last.pt) saved.
- [ ] `d_pure` computed (`compute_d_pure.py` wrapper or inline scipy paired-d) BEFORE V7/V8 final verdict tables are generated. Output committed to `review/0505/local/runs/V6_NOISE/d_pure.json`.
- [ ] Final PRIMARY threshold `d_thr = max(0.10, 1.8 * d_pure)` recorded in `review/0505/local/V7_V8_compare.md`. Decision matrix in §4 read with this `d_thr`.

### 9.1  V6 anchor (Fix F: 150K → 160K)

- [ ] V6@step_160000.pt exists on disk (`ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt`).
- [ ] V6@160K full-val has been computed; if not, run `eval_first_hop_224_clip3.py` against V6@160K first. Output JSON committed to `review/0505/operator/`.

### 9.2  YAML / launcher

- [ ] V7 yaml `max_steps` = `160000`; schedule lock fields unchanged (`warmup_steps=50000`, `ramp_steps=100000`, `total_steps_override=200000`).
- [ ] V8 yaml `max_steps` = `160000`; same schedule lock; `dead_branch_watch_steps = 160000`.
- [ ] V6-seed1337 yaml `max_steps` = `160000`; same schedule lock; `dead_branch_watch_steps = 10000` (unchanged from V6 baseline).
- [ ] launcher `run_v7_v8_sanity.sh` supports `V6_NOISE` tag pointing to `V6_seed1337.yaml` (extension applied 2026-05-05).
- [ ] launcher default `SANITY_STEPS = 160000`.
- [ ] ANALYSIS.md updated per §6 above (160K endpoint, pre-registration block with adaptive threshold, Bonferroni metric family, V6-seed1337 Stage 0).
- [ ] Output directory paths under `/data_2/...` confirmed empty for V7, V8, V6_NOISE (trainer enforces `require_fresh_output_dir: true`).
- [ ] Plan F document committed to `review/plan/`.

### 9.3  Determinism / reproducibility (Fix 2)

- [ ] **GPU UUID locked**: `nvidia-smi -L` recorded; V7/V8 reserve the same physical GPU as V6 (cuBLAS heuristic + nondeterministic CUDA op selection are GPU-chip-specific). Output committed to `review/0505/operator/v7v8_env_record_<date>.txt`.
- [ ] **CUDA / cuDNN / driver versions match V6's record**: `nvidia-smi --query-gpu=driver_version --format=csv`, `python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'`. If V6 record is missing, freeze the current versions as the new baseline and document.
- [ ] **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** exported in env (launcher already sets it; pre-flight only verifies via `echo $CUBLAS_WORKSPACE_CONFIG`).
- [ ] **step-1000 sentinel reserved**: after each V7/V8 run reaches step 1000, compare `metrics.jsonl` step-1000 entry against V6 step-1000 on `val_pair_total`, `val_chain_d20_mse`, `val_select_score`. Tolerance 0.5%; abort and investigate if exceeded. (At step 1000, lambda_roll=0 and LR is in warmup; V7 step_weights and V8 image_aux are gradient-irrelevant, so V7@1000 / V8@1000 / V6@1000 should be near-bit-equivalent on these metrics.)

### 9.4  V8 RNG-invariance diag (Fix 4, belt-and-suspenders)

- [ ] Before V8 launch: run `python3 review/0505/local/scripts/diag_v8_rng_invariance.py --config review/0505/local/configs/V8_no_image_aux.yaml --num-batches 1`. Expected: `torch.cuda.get_rng_state()` after one fake hop0 forward step is bit-identical between `image_aux.enabled=true` and `enabled=false` paths. (DiT verified RNG-free via local code grep — no nn.Dropout / DropPath / stochastic_depth in pet_lr/ + RAE pet_flow models; `NormAttention.attn_drop=proj_drop=0` default; `Mlp drop=0`; this diagnostic is defensive against future DiT modifications.)
### 9.5  Round 4 external review pre-launch gates (added 2026-05-06)

Per [`ROUND4_EXTERNAL_CONSENSUS_20260506.md`](ROUND4_EXTERNAL_CONSENSUS_20260506.md), 5/5 substantive external reviewers (Agents 2–6) unanimously required these P0 actions before V7 launches. The tag `round4-tmp-pre-external-review-20260506` listed 4 gates (a)–(d); gates (a)–(c) are SATISFIED in the consensus document itself, gate (d) is the operator-actionable item below.

- [ ] **Gate (d) — Option F same-process algebraic golden test** on V6@step_160000.pt:
  ```bash
  python3 review/0505/local/scripts/sigma_norm_golden_test.py \
      --config /path/to/v6_baseline.yaml \
      --ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt \
      --output-json review/0505/operator/sigma_norm_golden_test_result.json
  ```
  Cost: ~30 GPU-seconds. Outcome routing per §11 below. PASS unblocks V7 launch; AMBIGUOUS routes to Option G; FAIL defers V7 launch.
- [ ] **Action #4 — V6@seed42 same-seed double-pass at 1K-5K steps** (Agent 6 strong recommendation; cheap upper bound on (α) compounded). Two sequential 5K-step V6@seed42 runs on the same GPU; after each reaches step 5000, save `step_5000.pt` and run full-val. Compare `val_pair_total` and `val_chain_normal_mse` between pass-1 and pass-2. **Pre-launch gate**: if `val_pair_total` drift > 1.0%, escalate to §11 Tier 2 (multi-seed Welch's t-test) regardless of V6_NOISE final result. Cost: 10K extra GPU-steps total (≈6% overhead on V6_NOISE).
- [ ] **Action #5 — V7@seed42 same-seed double-pass at 5K-step spot-check** (Agent 6 cost-optimization vs full 160K replica). Two sequential V7@seed42 runs to step 5000; compare full-val `val_pair_total`, `val_chain_normal_mse` at step 5000. **Pre-launch gate**: if V7 pass-1-vs-pass-2 `val_pair_total` drift exceeds the V6 same-seed double-pass measurement from Action #4, V7's path activates additional (α) sources and Option H bundle is mandatory before continuing to step 160000. Cost: 5K extra GPU-steps × 2 = 10K (sequential).
- [ ] **Optional Action #3 — Option H determinism intervention bundle** (math-SDPA + `nn.AdaptiveAvgPool2d` → `F.avg_pool2d` at `pet_lr/model_first_hop.py:50` + `set_float32_matmul_precision("highest")`). Apply to V7/V8/V6_NOISE training from step 0. Expected throughput overhead < 2%. Optional unless Action #5 detects new (α) sources on V7 path; mandatory if so.
After all boxes checked, GPU launch is approved.

---

## 10. Post-run analysis protocol (reminder)

Both V7 and V8 produce `metrics.jsonl` + `step_NNNNNN.pt` series + `best.pt` (full-val-selected via `fc7446d`). V6-seed1337 produces the same artifacts (minus `best.pt` selection mechanism, since that yaml omits `best_select_full_eval_interval` for invariance with V6-seed42).

Decision protocol:
1. Compute `d_pure = paired_cohen_d(V6-seed42@step_160000.pt, V6-seed1337@step_160000.pt)` on `val_chain_normal_mse` over 7403 unique slices.
2. Set `d_thr = max(0.10, 1.8 × d_pure)`. Record value.
3. Compare PRIMARY = `step_160000.pt` (= last.pt, no selection bias) and SECONDARY = `best.pt` (V7/V8 only) against V6-seed42@step_160000.pt via [`review/0505/scripts/`](../0505/local/scripts/) full-val evaluator.
4. Decision per §4 matrix with `d_thr` substituted for the literal `0.10`.

---

## 11. Threshold-inflation contingency (added 2026-05-06)

**Trigger context**: Round 4 multi-agent internal pre-review (see [`ROUND4_PRE_REVIEW_CONSENSUS_20260506.md`](ROUND4_PRE_REVIEW_CONSENSUS_20260506.md)) flagged that the σ-normalize sanity failure of 12.73% on `val_chain_normal_mse` between A and B (same seed, same yaml topology) implies non-trivial run-to-run drift even on identical settings. By extension, V6-seed42 vs V6-seed1337 (different seeds, same yaml) may produce a `d_pure` substantially larger than the 0.056 originally assumed under "V6 vs V6.1 = 0.067 includes algorithmic+selection contamination, so RNG-only is smaller". Agent 6's quantitative estimate puts `d_pure` plausibly in the [0.10, 0.40] range — which would resolve `d_thr = 1.8 × d_pure` to **[0.18, 0.72]**.

At `d_thr = 0.18`, Plan F retains adequate power (80%-power MDE = 0.033 ≪ 0.18). At `d_thr = 0.50+`, Plan F can no longer detect a realistic Grönwall effect on V7 because the predicted V6→V7 effect size is order-of-magnitude smaller than the threshold.

**Pre-registered contingency tiers** (read `d_pure` value when V6_NOISE completes §9.0):

| `d_pure` range | `d_thr` (= max(0.10, 1.8×d_pure)) | Action |
|---|---|---|
| `d_pure ≤ 0.056` | `d_thr = 0.10` (floor) | **No change**. Run §4 decision matrix as written. |
| `0.056 < d_pure ≤ 0.10` | `d_thr = 0.10` (still on the floor) | **No change**. Run §4 decision matrix as written. Document `d_pure` in paper Methods. |
| `0.10 < d_pure ≤ 0.20` | `d_thr` resolves to `0.18 – 0.36` | **Tier 1 contingency**: keep PRIMARY paired analysis as-is. Add a "realized noise floor" footnote to the decision verdict; flag as **higher-risk decision** in narrative. Power remains adequate. |
| `0.20 < d_pure ≤ 0.40` | `d_thr` resolves to `0.36 – 0.72` | **Tier 2 contingency**: switch PRIMARY to **multi-seed Welch's t-test** (Q3 Option C). Concretely: spawn V7-seed1337 and V6-seed1337 (already exists) and V7-seed42 (already exists) and V6-seed42 (already exists). Cost: +1 V7 retrain at 160K = +1 GPU·day. Welch's t-test on per-arm per-slice means with `n_arm = 2 seeds × 7403 slices = 14806`. Threshold reverts to `d ≥ 0.10` paired-effect-size on the pooled per-arm distribution. |
| `d_pure > 0.40` | `d_thr > 0.72` (under-powered) | **Tier 3 contingency**: abandon paired-difference primary; switch to **Q3 Option E** (reframe conclusion language). Report V7 vs V6 Δ with `d_pure` as a mandatory disclosed metric. Defer V7-vs-V6 superiority claim until Option H (math-SDPA + AdaptiveAvgPool replacement) is implemented and a re-run reduces `d_pure` below 0.20. Plan F V7 verdict downgraded from "H1 confirmed if favored" to "V7 differs from V6 by Δ∈[…]; difference is consistent with kernel-noise floor". |

**Pre-launch action** (P0, BEFORE V7/V8/V6_NOISE launch):

Per multi-agent consensus, run **same-process algebraic golden test (Option F)** on V6@step_160000.pt as a pre-flight check. Cost: ~30 GPU-seconds. Concrete protocol per Round 4 prompt Q3 Option F. Outcomes:
- **Bit-identical (≤ 1e-7 abs drift)**: σ-normalize algebraic equivalence verified at inference. Sanity failure is definitively training-time + kernel-noise + 20K-step weight divergence. Proceed with Plan F as written.
- **≥ 1e-3 drift**: algebraic equivalence broken at inference. Defer V7 launch; investigate `step_normalizers` injection bug in `pet_lr/rollout_first_hop.py:99-104` and `train_first_hop.py:395-528`. Plan F is unsafe until algebra is fixed.
- **Drift in [1e-7, 1e-3]**: ambiguous. Run Option G (paired raw-vs-σnorm full-val evaluator) for distributional bound. If max drift on full val < 1e-4, accept algebra and proceed; else investigate.

**Audit reference**: This contingency table is the binding response to Round 4 reviewers' Q2.d (point estimate for `d_pure` and resulting `d_thr`). If reviewers' point estimate diverges from the empirical `d_pure` measured by V6_NOISE, the empirical value governs.

---

## 12. External review reconciliation (added 2026-05-06)

The Round 4 prompt was sent to 6 external reviewer agents after the internal pre-review reached 82/82 prompt integrity (commit `dcdcd2c`, tag `round4-tmp-pre-external-review-20260506`). Full convergence matrix and per-agent verdicts are in [`ROUND4_EXTERNAL_CONSENSUS_20260506.md`](ROUND4_EXTERNAL_CONSENSUS_20260506.md). This section reconciles the external reviewer outputs with §11 and updates the §11 prior tier accordingly.

### 12.1 Cross-agent convergence (5/5 unanimous)

All 5 substantive reviewers (Agents 2–6) agreed on:
- Q1 sub-claim (i) "PyTorch reports nondeterministic Mem-Eff-attention + adaptive_avg_pool2d_backward_cuda" → **VERIFIED FACT**
- Q1 sub-claim (ii) "kernel noise amplified through `mix_latent` is **dominant**" → **PARTIAL or WRONG** (rejected as exclusive cause)
- Dominant mechanism is **(γ) optimizer-state / weight-trajectory divergence after 20K SGD steps under warn-only nondeterministic kernels**
- Q2: V7 raw-rollout structurally eliminates (β) but does **NOT** eliminate (α) kernel non-determinism
- Q2: σ-normalize 12.73% A/B drift is NEITHER (b1) V6-replica drift NOR (b2) V7-vs-V6 signal — it's a third quantity
- Q2: V6_NOISE is necessary but not sufficient; V7-replica gate is required
- Q3: F+G+H bundle BEFORE V7 launch is the strongest pre-launch action (5/5 unanimous #1 ranking)
- Q3: D (inflate K) is the worst option (5/5 unanimous reject)

### 12.2 The one critical divergence — `d_pure` point estimate

| Agent | Point estimate | Mapped §11 tier | Plan F survival |
|---|---|---|---|
| Agent 2 | 0.08 | Tier 0 (floor) | ✅ survives |
| Agent 3 | 0.07 | Tier 0 (floor) | ✅ survives |
| Agent 4 | 0.20 | Tier 2 boundary | ⚠️ Welch's t-test required |
| Agent 5 | 0.08 | Tier 0 (floor) | ✅ survives |
| Agent 6 | 1.0 | **Tier 3+** | ❌ Tier 3 (Option E reframe) mandatory |

Spread: 14× between Agent 3 (0.07) and Agent 6 (1.0). Agents 2/3/5 cluster around 0.07–0.08; Agent 4 is moderate outlier; Agent 6 is the catastrophic outlier (anchored on √8 step-compounding heuristic).

**Resolution**: The empirical V6_NOISE measurement governs. Reviewer estimates are heuristic priors; the §11 tier table absorbs all 5 outcomes (Agents 2/3/5 → Tier 0, Agent 4 → Tier 2, Agent 6 → Tier 3). Plan F's existing tiered contingency therefore covers the entire reviewer estimate range without modification.

### 12.3 Updates to Plan F prior tier (§11) based on external review

The prior on `d_pure` is widened to absorb Agent 6's catastrophic estimate. Specifically:
- **Pre-launch d_pure prior bands** (from external reviewers, before V6_NOISE measurement):
  - 60% probability: d_pure ∈ [0.05, 0.10] (Agents 2/3/5 cluster) → Tier 0
  - 25% probability: d_pure ∈ [0.10, 0.40] (Agent 4 + Agent 6's lower-end range overlap) → Tier 1 or Tier 2
  - 15% probability: d_pure > 0.40 (Agent 6's median) → Tier 3
- **Decision rule**: V6_NOISE empirical d_pure measurement supersedes all priors. The pre-registered §11 tier table executes mechanically once d_pure is computed at §9.0.
- **No structural change to §11**. The 4-tier table already absorbs the full reviewer range; this subsection documents that absorption is intentional, not coincidental.

### 12.4 Mandatory pre-launch additions from external review

Per §9.5 (added 2026-05-06):
- Gate (d): Option F same-process golden test on V6@step_160000.pt (script at [`review/0505/local/scripts/sigma_norm_golden_test.py`](../0505/local/scripts/sigma_norm_golden_test.py))
- Action #4: V6@seed42 same-seed double-pass at 1K-5K steps (Agent 6 strong recommendation)
- Action #5: V7@seed42 same-seed double-pass at 5K steps (cost-optimized V7-replica per Agent 6)
- Action #3 (optional): Option H bundle, mandatory if Action #5 detects new (α) sources on V7 path

### 12.5 Audit trail entry

| Date | Document | Verdict |
|---|---|---|
| 2026-05-06 | Round 4 prompt sent to 6 external reviewer agents (after 82/82 internal pre-review at commit `dcdcd2c`) | Q1: 5/5 unanimous (i)=VERIFIED, (ii)=PARTIAL/WRONG; Q2: 5/5 raw-rollout eliminates β not γ; Q3: 5/5 strongest rec = F+G+H bundle pre-launch + V7-replica gate; `d_pure` point estimates spread 14× across reviewers (0.07 to 1.0) — fully absorbed by §11 4-tier contingency table |
| 2026-05-06 | [`ROUND4_EXTERNAL_CONSENSUS_20260506.md`](ROUND4_EXTERNAL_CONSENSUS_20260506.md) written | Cross-agent convergence matrix; GPU-launch gate (a)=SATISFIED, (b)=FAILED-but-SUBSTITUTED, (c)=SATISFIED, (d)=PENDING (operator runs sigma_norm_golden_test.py) |
| 2026-05-06 | §9.5 (Round 4 external review pre-launch gates) added; §12 reconciliation written | Plan F unblocked for GPU launch contingent on operator running gate (d) and Actions #4/#5 |
| 2026-05-06 | §12.6 added: empirical d_pure prior from existing V6/V6.1 per-slice CSVs (script: [`analyze_v6_self_paired_stats.py`](../0505/local/scripts/analyze_v6_self_paired_stats.py)) | V6 same-yaml self-paired d ∈ [0.156, 0.208]; V6 vs V6.1 cross-arm paired d = 0.067; V6@160K vs V6.1@best d = 0.012. Refined d_pure prior: most-likely Tier 0 (d ≤ 0.10), 90% upper bound Tier 1. Opus 4.7's "d_pure ≥ 0.20 almost certain" claim is REFUTED — it conflates SGD-progress with RNG noise. |

### 12.6 Empirical d_pure prior from V6 step160K full-val anchor (added 2026-05-06)

**Trigger**: After commit [`7605f6c`](https://github.com/...) added V6 step160K full-val anchor artifacts, Opus 4.7 external analysis flagged the V6 step160K vs V6 step200K paired Cohen d = 0.208 as "evidence d_pure ≥ 0.20 almost certain". Independent computation against the existing per-slice CSVs from `review/0505/operator/artifacts/` shows this framing **conflates SGD progress with RNG noise**. Reproducible diagnostic: [`review/0505/local/scripts/analyze_v6_self_paired_stats.py`](../0505/local/scripts/analyze_v6_self_paired_stats.py).

**Measured pairwise paired Cohen d on the existing 7403-slice full-val artifacts** (mse_NORMAL endpoint):

| comparison | semantics | rel diff (%) | paired Cohen d | paired t |
|---|---|---:|---:|---:|
| V6 self: 160K → 200K (40K extra steps) | same yaml, same seed, +40K SGD progress | +0.756 | +0.208 | +17.86 |
| V6 self: 160K → 185.6K best (24K) | same yaml, same seed, +24K SGD progress | +0.619 | +0.208 | +17.90 |
| V6 self: 185.6K best → 200K last (14.4K) | same yaml, same seed, late-stage SGD | +0.136 | +0.156 | +13.42 |
| V6.1 self: best → last | same yaml, same seed, late-stage SGD | +0.125 | +0.153 | +13.14 |
| V6 vs V6.1: best | **different yaml**, paired across algos | −0.522 | −0.068 | −5.83 |
| V6 vs V6.1: last | **different yaml**, paired across algos | −0.534 | −0.067 | −5.79 |
| V6@160K vs V6.1@best | different yaml, different step | +0.093 | +0.012 | +1.05 |

**Decomposition**:
- The d ≈ 0.156 between V6@best and V6@last (only 14.4K extra steps, model essentially saturated) is the **structural noise floor for this metric on full val**: per-slice paired SD ≈ 2.1e−6, which is ~170× smaller than the cross-slice MSE SD (≈ 3.65e−4). Most of the d-magnitude comes from a tiny but **systematically directional** mean diff (model gets monotonically slightly better with more steps), not from pure noise scatter.
- The d ≈ 0.208 between V6@160K and V6@200K is **mostly the same structural floor** (it grows from 0.156 → 0.208 as the step gap doubles from 14.4K → 40K, so SGD progress contributes only ~0.05 of additional d for 25K extra steps).
- The cross-arm V6 vs V6.1 paired d ≈ 0.067 is the **closer analog** for "two trained-from-scratch runs with config differences": even with **different yaml**, paired d stays well below 0.10 because both models converge to similar solutions on most slices.
- V6@160K vs V6.1@best paired d = 0.012 (essentially zero) confirms that two different algorithms at saturation are statistically indistinguishable per-slice.

**Refined d_pure prior** (V6_seed42@step_160000 vs V6_seed1337@step_160000):
- Same yaml + same step ⇒ no SGD progress confound (mean diff component ≈ 0)
- Per-slice paired SD: bounded above by V6 vs V6.1 paired SD (≈ 1.9e−5), bounded below by V6 self best→last paired SD (≈ 2.1e−6). Plausible range: [3e−6, 1.5e−5].
- With mean diff ≈ 0 and finite-sample noise: most-likely d_pure ∈ [0.01, 0.05]; 90% upper bound ≈ 0.10–0.12.

**Updated probability mass on §11 tiers** (replaces §12.3 reviewer-prior table):

| Tier | d_pure range | Action | Empirical posterior probability |
|---|---|---|---:|
| Tier 0 | ≤ 0.10 | d_thr = 0.10, no inflation | **~85%** (was 60% under reviewer prior) |
| Tier 1 | 0.10–0.20 | d_thr = 1.8 × d_pure | ~12% (was 25%) |
| Tier 2 | 0.20–0.40 | switch to Welch's t + multi-seed | ~3% (was 15%) |
| Tier 3 | > 0.40 | reframe to closed-form contribution | <1% (was 15%) |

**Plan F structural verdict**: with > 85% probability the experiment lands in Tier 0 and the existing primary decision rule (5% relative MSE margin) executes unmodified. The §11 4-tier table remains intact as defensive scaffolding for the long-tail outcomes; no pre-emptive narrative reframe is required.

**Constraints preserved**:
- V6_NOISE empirical measurement still governs final tier selection (priors do not bypass measurement)
- Plan F yaml content unchanged
- Round 4 prompt at `dcdcd2c` unchanged (82/82 integrity)
- Option F golden test (gate (d)) still pending operator execution

**Independent corroboration of operator's rolling-val concern**: V6@160K rolling-val NORMAL chain MSE = 2.370e-4 vs full-val NORMAL chain MSE = 2.4454e-4 (relative gap +3.18%). This validates §10 (no rolling-val gating during Plan F) and the operator's recommendation that V6 best.pt selection bias should be treated as a known quantity, not an open question.

**What this changes in §12.4 pre-launch gates**:
- No new gates added.
- Action #4 (V6 same-seed double-pass at 1-5K) remains valuable as a **floor check** — confirms d_pure ≈ 0 when the only difference is RNG init in the dataloader, providing a lower-bound anchor for V6_NOISE interpretation.
- Action #5 (V7 5K-replica spot-check) remains the highest-value cheap diagnostic.
