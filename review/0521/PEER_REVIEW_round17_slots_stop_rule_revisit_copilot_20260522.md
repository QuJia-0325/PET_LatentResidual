# Peer Review Round 17-Slots — Stop Rule Revisit

- date: 2026-05-22
- reviewer: GitHub Copilot
- subject: whether to revise A4-light into an image_aux bracket sweep
- scope: governance / slot allocation, not F0/A4 execution details

## 0. Main Verdict

**Main: 方案 E — A4-bracket 2 点 (A4-low λ=0.02 + A4-mid λ=0.08), 2 slots, leave 1 slot as buffer.**

**Backup: 方案 D — keep Round 17 original A4-light (λ=0.08 only).**

I do not recommend 方案 A as default. A planned bracket is not the same failure mode as serial A4-v2/v3 relaunching, so revisiting the stop rule is legitimate. But the slot-utilization argument should not promote a 1-probe plan all the way to a 3-probe sweep. The high-value paper asset is a bounded response curve around the known λ=0.04 point; λ=0.02 + λ=0.08 captures that. λ=0.12 is lower EV and creates the stronger precedent that idle slots should automatically become more experiments.

## 1. Q1 — Should The Stop Rule Be Revisited?

**MODIFY / partial accept.**

The current wording literally bans `A4-v2 / A4-v3 / other image_aux variant`, so 方案 A/E are not allowed by the current task md. The prompt's distinction between serial relaunch and pre-registered parallel sweep is real, but Round 17 integration repeatedly says **one pre-registered A4 probe**. So this is not merely a loophole; it is a scope change requiring an explicit Round 17-Slots decision.

I would accept one controlled revision: replace `A4-light` with a **closed two-point bracket**. That preserves the spirit of B74 because the new scope is fixed before any outcomes are known. I would not accept `3 slots because empty slots feel wasteful` as sufficient reason to run λ=0.12 or additional follow-ups.

## 2. Q2 — Does Image_Aux Sweep Have EV?

**APPROVE for 2-point bracket; REJECT for 3-point default.**

The sweep has real paper EV because image_aux is now the paper headline: V13 λ=0 gives 36.4943, V7 λ=0.04 gives 36.7810, and Round 17 integrated X2 treats that +0.287 dB as the dominant clean design factor. A response curve helps turn one ablation contrast into a controlled design story.

But the marginal value is asymmetric:

- λ=0.08 is the original A4 and tests whether 0.04 is underpowered.
- λ=0.02 tests whether the effect already saturates below 0.04.
- λ=0.12 mostly tests over-regularization and is less likely to change the paper unless λ=0.08 wins.

If λ=0.08 beats V7 by ≥0.05 dB, the correct governance response is still **do not chase λ=0.06/0.10/0.16 in this paper cycle**. The two-point bracket is a bounded pre-registered sweep, not an optimizer.

## 3. Q3 — Scheme Choice

**Main: E. Backup: D.**

Reject A: three image_aux variants overweights figure aesthetics and consumes the buffer on the lowest-EV λ=0.12 point.

Reject B as main: V14b is useful but not more central than preserving operational headroom. Round 17 already says V14b is recommended-but-not-required and should not displace A4. If a third slot is truly healthy after the two A4 launches and F0/paper work shows seed variance is still the blocker, V14b can be a separate user-signed spare-slot decision.

Reject C: V13b answers a secondary train-config residual question. It is not as paper-central as the image_aux response curve.

Reject D as main: it is safe, but it underuses a rare opportunity to pre-register a small bracket around the paper's headline variable.

## 4. Q4 — RAM / IO Contention

**Staggered launch, not simultaneous.**

Use the Round 15 / B58 protocol. V7-class runs share the same `latent_dir`, and V18 design rationale explicitly says `num_workers=0` is necessary but not sufficient for multi-task IO safety.

Recommended protocol:

1. T=0: launch A4-mid (λ=0.08).
2. T=+30 min: run `nvidia-smi` memory check plus `iostat -x 1 5`; verify process alive and training log healthy.
3. If healthy: launch A4-low (λ=0.02).
4. T=+60 min: repeat health check; keep slot 3 empty unless a new signed decision assigns it.

Do **not** change dataloader workers as part of this bracket. V7 has `data.num_workers: 0`; changing it would break the single-variable clone discipline and is unnecessary for lowering workers.

## 5. Q5 — Sweep Priority

**Priority order: A4-mid first, A4-low second, skip A4-high.**

A4-mid is the original Round 17 probe and has the clearest upside: it tests whether λ=0.04 was underpowered. A4-low has the next-highest value because it distinguishes sharp threshold vs early saturation between λ=0 and λ=0.04.

A4-high λ=0.12 is the lowest EV default point. If 0.08 regresses, 0.12 was predictable. If 0.08 wins, 0.12 tempts a new optimizer loop. Either way it is not the right default use of the third slot.

## 6. Q6 — Integration / Task Updates

**APPROVE if ≥2/3 reviewers support E or stronger.**

Yes, write a formal `REVIEW_INTEGRATION_round17_slots_20260522.md` before changing the Codex task. This round is explicitly revising a signed Round 17 action plan, so the revision needs its own audit trail.

Required task-md changes if E is accepted:

- Change A4 section from single `A4_image_aux_lambda_08` to a two-run bracket: `A4_image_aux_lambda_02` and `A4_image_aux_lambda_08`.
- Add one new YAML for λ=0.02. Keep λ=0.08 as the original A4-mid.
- Adjust self-check from `wc -l <= 1` to an exact whitelist of two YAMLs.
- Update NOT-DO #6 to ban λ=0.06/0.10/0.12/ramp/cosine and all other image_aux variants.
- Update launch protocol to staggered two-slot launch and preserve slot 3 as buffer.

## 7. Updated Stop Rule

Suggested replacement:

> A4-bracket is closed at two pre-registered from-scratch V7 clones: λ=0.02 and λ=0.08. After these complete, no additional image_aux λ, schedule, ramp/cosine, or follow-up variant may be launched before paper submission without a new user-signed review round, regardless of whether either result is success, marginal, or regression.

This keeps the conceptual benefit of a planned sweep while preserving the anti-scope-creep force of B74.

## 8. Q7 — New Biases (B83+)

| ID | bias | severity | mitigation |
|---|---|---|---|
| B83 | Slot-utilization anchor: empty GPU feels like waste, even when buffer reduces operational risk | MED | Reserve slot 3 by default; require separate signed decision for third long run |
| B84 | Sweep-figure bias: response curve looks more publishable than its incremental evidence value | MED | Limit to λ=0.02/0.08; skip λ=0.12 unless new round after paper needs it |
| B85 | Stop-rule slippage: one exception weakens future stop rules | HIGH | Make the exception pre-outcome, finite, and written into an integration doc |
| B86 | A4-light reinterpretation: claiming Round 17 never banned parallel variants rewrites the integration | MED | Treat this as an explicit scope revision, not a loophole |
| B87 | Optimizer-loop reentry: λ=0.08 success invites λ=0.06/0.10/0.16 | HIGH | Stop rule says no further image_aux variant regardless of outcome |
| B88 | Third-slot overconfidence from Round 15 success | LOW | Use B58 staggered launch and health checks; do not assume all triples are equal |

## 9. Final Recommendation

Adopt **方案 E** only after a Round 17-Slots integration file records reviewer consensus. Then update the Codex task from A4-light to a closed two-point A4-bracket. Keep the third slot empty as an intentional buffer, not as unused value.
