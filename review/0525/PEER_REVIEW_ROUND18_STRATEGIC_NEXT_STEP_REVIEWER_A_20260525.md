# Peer Review Round 18 - Strategic Next Step - Reviewer A

- date: 2026-05-25
- reviewer: **Reviewer A** (GitHub Copilot, independent)
- scope: Review `ROUND_18_A4_BRACKET_ANALYSIS_20260525.md` and Round 18 strategic options X1-X5
- boundary: report-only review; no code/config/task modifications
- upstream substrate: A4 bracket full-val canonical eval completed, n=7403, `PSNR_clip3`

---

## 0. Main Verdict

**Main strategic verdict: Hybrid = X1-lite mechanism falsification + X5-gated generalization.**

Do **not** continue lambda sweeps. Do **not** pivot the whole project into X2 architecture rewrite yet. The next strategic move should convert A4-mid from "a scalar lambda trick" into a defensible paper claim:

1. **X1-lite (mandatory):** one discriminative mechanism package, not the full 3-run subloss sweep. Run one `lambda_img=0.08` control that removes the most likely confound, plus a no-training gradient/latent probe. This is mechanism falsification, not parameter tuning.
2. **X5-gated (primary if data exists):** if a second PET tracer/anatomy dataset can be made runnable within ~2 weeks, run a minimal cross-dataset check: V7/V13 baseline equivalent + A4-mid only. Generalization is more paper-changing than another local lambda point.
3. **Backup if X5 is blocked:** run X3 as a **single interaction test**, not as V18 resurrection: `lambda_kl=0`, `lambda_img=0.08`, decoder LoRA on. Its purpose is to decide whether V18 is redundant under strong image_aux.

**V18 paper role right now: (b) secondary ablation.** It should no longer be the headline. Keep it as "decoder LoRA produced smaller gains than image_aux schedule" unless the single X3 interaction test shows clear additivity.

**Paper should start now:** outline + methods + result tables immediately. Do not wait for X2. MICCAI 2027 is feasible with A4-mid + mechanism + one closure/generalization result; TMI/MedIA becomes realistic only if X5 succeeds.

---

## 1. Q1 - Is A4-mid +0.113 dB Real?

**Verdict: APPROVE with wording constraints.**

A4-mid is very likely a real model effect, not seed luck. The strongest evidence is not just the +0.1130 dB NORMAL gain:

- full validation uses `n=7403` slices;
- D20/D10/D4/NORMAL all improve in the same direction;
- `best.pt` and `last.pt` are identical at step 160000, so this is not a noisy early checkpoint selection artifact;
- A4-low (`lambda=0.02`) moves in the opposite direction relative to V7 across the chain, so the bracket is informative rather than merely lucky;
- V14 seed noise is reported as about -0.0004 dB relative to V7, making +0.1130 dB too large to dismiss as ordinary d_pure noise.

But the paper must not overclaim. The safe claim is:

> In a single-seed full-validation canonical chain evaluation, increasing hop0 image auxiliary strength from 0.04 to 0.08 improves NORMAL PSNR by +0.113 dB over V7, with same-direction gains at all rollout endpoints.

Unsafe claims:

- patient-level significance;
- multi-seed robustness;
- universal PET generalization;
- mechanism identity.

No extra ablation is needed to treat A4-mid as the current main empirical result. Extra work is needed to explain and generalize it.

---

## 2. Q2 - Which image_aux Mechanism Is Most Likely?

**Verdict: MODIFY the prompt's M1-M4 framing. The likely mechanism is M1+M4, with M2 secondary and M3 currently weak.**

Implementation facts matter:

- `compute_hop0_image_losses` predicts hop0 latent, decodes `z_pred`, and compares decoded `x_pred` to `x_dst` through `compute_first_hop_image_loss`.
- `image_aux` is added to total loss as `lambda_img * loss_img` alongside pair/rollout/FOC losses.
- The loss is weighted L1 + 0.25 SSIM + 0.10 seam by default.
- The default seam term is `seam_consistency_loss(x_pred)` unless extended seam is enabled, so it is mostly a self-consistency/artifact penalty, not a target-matching term.

My mechanism ranking:

| mechanism | probability | reason |
|---|---:|---|
| **M1: latent transport undertrained by latent MSE alone** | high | Stronger decoded-image gradient improves all chain endpoints; this fits a missing supervision signal. |
| **M4: decoder-manifold anchoring** | high | Frozen decoder backprop forces `z_pred` into regions that decode well; this also explains why decoder LoRA/capacity was less central. |
| **M2: SSIM structure signal** | medium | Plausible for PET structure, but no current evidence isolates SSIM from L1. |
| **M3: seam artifact control** | low-medium | Seam may help patch artifacts, but default seam is not target-specific and weight is small. It is unlikely to explain the whole +0.113 dB. |

Potential M5:

**M5 - loss-balance reallocation.** Raising `lambda_img` may change gradient competition against rollout/pair/FOC losses, effectively rebalancing optimization toward clinically visible endpoints. This is not identical to M1; it is about relative gradient budget rather than missing signal type.

Minimal disambiguation under the prompt's <=1 run + 1 probe constraint:

1. **One run:** `lambda_img=0.08`, L1-only, with SSIM and seam weights zeroed. Keep the global lambda and all training settings fixed. If L1-only recovers most A4-mid, M1/M4 dominate; if it collapses toward V7/A4-low, M2/M3 matter.
2. **One probe:** no-training gradient attribution on matched batches from V7 and A4-mid checkpoints: per-subloss gradient norm wrt `z_pred`, cosine with pair/rollout gradients, decoded-image error maps, and latent distribution distance to target latents.

This is not a parameter sweep because the goal is not to select L1-only for deployment; it is to falsify whether SSIM/seam are necessary for the A4-mid gain.

---

## 3. Q3 - Which X1-X5 Are Worth It Under "No Small Tuning"?

**Verdict: choose X1-lite + X5-gated; backup X3. Reject X2/X4 as immediate next steps.**

Ranked decision:

| rank | option | verdict | rationale |
|---:|---|---|---|
| 1 | **X1-lite** | APPROVE | Paper needs mechanism. Use one discriminative run + one probe, not 3-run component sweep. |
| 2 | **X5-gated** | APPROVE if data is reachable quickly | Cross-dataset/tracer evidence changes venue ceiling more than more local knobs. |
| 3 | **X3** | BACKUP / one-run cap | Useful only to decide V18 redundancy/additivity; must not become V19 campaign. |
| 4 | X2 | REJECT for current paper | Too much 4-8 week architecture risk after A4 already creates a viable paper. |
| 5 | X4 | REJECT | Chain redesign breaks comparability and risks resetting the baseline story. |

Specific answers to the prompt's concerns:

- **Is X1 disguised tuning?** Full X1 with l1-only/ssim-only/seam-only as 3 long runs is too close to component tuning. X1-lite is acceptable because it is a falsification experiment with a predeclared interpretive role.
- **Is X2 over-ambitious?** Yes for the next 1-3 months. It is a future project, not the next move for this paper.
- **Is X3 disguised V18.v2?** It becomes V18.v2 if open-ended. It is acceptable only as one interaction test with a hard stop.
- **Is X4 baseline-breaking?** Yes. Changing chain geometry would make V13/V14/V18/A4 evidence less directly comparable.
- **Is X5 realistic?** Only if data and preprocessing are already within reach. If data access is uncertain, do not block paper progress on X5.

Main recommendation in operational terms:

1. Start paper outline now.
2. Run X1-lite mechanism package.
3. In parallel, decide within 1 week whether X5 data is real. If yes, do X5 minimal. If no, run the single X3 interaction test.

---

## 4. Q4 - What Is V18's Paper Role Now?

**Verdict: (b) secondary ablation.**

The headline should move to image_aux. A4-mid has +0.113 dB over V7, while V18.best is +0.030 dB and V18.last is +0.062 dB. Round16 already retired the KL-direct-decoder narrative, so V18 cannot carry the main mechanism story as originally framed.

Recommended paper positioning:

- Main result: hop0 pixel-space supervision through frozen decoder is the dominant improvement lever.
- Secondary ablation: decoder LoRA/capacity was tested and gave smaller or less stable gains.
- V18 narrative: useful negative/limited-positive evidence, not the center.

Do **not** fully delete V18. It is useful because it shows the project tried a more complex capacity route and found a simpler transport-side supervision lever was stronger.

Should X3 be canceled? No, but constrain it tightly. One X3 run is allowed because it answers an interaction question:

- additive: V18 becomes a joint ablation worth one table row;
- non-additive: V18 is demoted permanently;
- negative: V18 becomes evidence that decoder capacity conflicts with strong image_aux.

No rank sweep, no KL revival, no V18-clean resurrection in Round18.

---

## 5. Q5 - Missing Direction X6+

**Verdict: yes, the prompt misses one critical non-training direction.**

### X6 - Clinical endpoint / task-level validation

A4-mid is evaluated by PSNR_clip3. A medical imaging reviewer will ask whether +0.113 dB matters clinically. Before rewriting architectures, add task-facing evaluation:

- lesion or high-uptake ROI PSNR/MAE if ROI masks or thresholded uptake regions are available;
- SUV bias / uptake recovery error;
- contrast-to-noise ratio on relevant regions;
- visual artifact audit at D20/D10/D4/NORMAL;
- radiology-style blind preference on a small fixed panel if feasible.

This is not small tuning and may require zero new training. It can convert A4-mid from "better PSNR" to "better PET restoration behavior". If clinical endpoints contradict PSNR, the paper narrative must change before any venue submission.

### X7 - Dose-conditioned control as future work

If the team later wants a true architecture direction, prefer dose-conditioned control / CFG-style dose guidance over a generic X2 rewrite. It is closer to the project's existing dose-chain structure and less likely to invalidate all baselines than replacing the transport backbone outright.

X6 should be added to the Round18 option set as a near-term paper requirement. X7 is future work, not immediate execution.

---

## 6. Q6 - Paper Timeline

**Verdict: MICCAI 2027 is feasible; TMI/MedIA needs X5 or X6 strength.**

Suggested timeline:

| time | action |
|---|---|
| now | Start outline, methods skeleton, experiment ledger, and figure plan. |
| next 1-2 weeks | X1-lite probe/control + X6 clinical/task metric audit. |
| next 2-4 weeks | If data ready: X5 minimal generalization. If not: one X3 interaction test. |
| after closure package | Freeze experiments and draft full paper. |
| by 2026-08/09 | Internal full draft with figures and limitations. |
| by 2026-12 | MICCAI 2027 submission-ready. |

Venue judgment:

- **MICCAI 2027:** realistic if X1-lite + X6 are clean, even without X5.
- **TMI/MedIA:** stronger if X5 generalization succeeds or X6 clinical endpoint evidence is compelling.
- **Do not wait for X2** for the first submission. X2 is a next-paper branch.

Current paper stage should begin at **outline + methods + results table draft**, not just notes. The core result table already exists.

---

## 7. Q7 - Bias Audit

Round17 already used B83-B89 in nearby reviews, so I continue numbering at B90 to avoid collision.

| ID | bias | severity | mitigation |
|---|---|---|---|
| **B90** | A4-savior over-romanticization | HIGH | Keep single-seed/slice-level/single-dataset labels on every claim. |
| **B91** | Scalar-knob shame | MED | Do not dismiss A4 because it is "just lambda"; the result may reveal a real mechanism. |
| **B92** | Mechanism theater | MED-HIGH | X1 must falsify a concrete mechanism, not produce three decorative ablation bars. |
| **B93** | Architecture-pivot glamour | HIGH | X2 feels substantive but can delay a viable paper by months. Treat as next project. |
| **B94** | V18 sunk-cost preservation | MED-HIGH | Do not keep V18 central because it consumed many review rounds. One interaction test max. |
| **B95** | Single-dataset complacency | HIGH | If X5 is available, it has higher strategic value than another local model tweak. |
| **B96** | PSNR monoculture | HIGH | Add X6 clinical/task metrics before venue decisions. |
| **B97** | Seed-noise false security | MED | V14 noise is reassuring, but it is not a full multi-seed distribution. |

Prompt-specific concern: the current analysis says "A4-mid is the project center" too quickly. I agree with the direction, but the stronger reviewer-safe phrasing is:

> A4-mid is the current strongest canonical-chain empirical result; it becomes the project center only after mechanism and clinical/task relevance are checked.

---

## 8. Final Reviewer A Recommendation

**Do next:** X1-lite + X6 immediately, with X5-gated as the main strategic expansion if data is available. If X5 is blocked, run exactly one X3 interaction test.

**Do not do next:** lambda sweep, X2 architecture pivot, X4 chain redesign, V18 rank/block/KL revival.

**Paper posture:** rewrite around image_aux as the headline, V18 as secondary ablation, limitations explicit: single seed, slice-level only, single dataset unless X5 lands.

This path respects the user's "no small parameter fiddling" constraint without throwing away the paper-ready A4 signal or overreacting into a risky architecture reboot.
