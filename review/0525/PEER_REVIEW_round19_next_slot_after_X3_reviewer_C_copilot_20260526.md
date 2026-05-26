# Peer Review Round 19-Pre — Next Slot After X3

- date: 2026-05-26
- reviewer: GitHub Copilot / reviewer C
- reviewed prompt: `PEER_REVIEW_PROMPT_round19_next_slot_after_X3_20260526.md`
- scope: decide whether to use one freed training slot after X3 non-additive result

## 0. Main Verdict

**Main verdict: A — launch A4-mid-seed1337.**

**Backup: C — no new training; keep writing paper and wait for X1-lite.**

Use the freed X3 slot for exactly one A4-mid seed replicate. This is not because an idle GPU is wasteful; it is because A4-mid is now the paper headline, and that headline is still one seed. X1-lite answers mechanism. A4-mid-seed1337 answers robustness of the main result. Those are complementary, and Option A does not interfere with X1-lite or the paper draft.

Reject Option B until X1-lite reports. Reject Option D outright.

## 1. Q1 — Use The Freed Slot?

**APPROVE, but only for Option A.**

The slot should be used if the new run directly reduces the paper's largest remaining risk. A4-mid-seed1337 does that. A4-mid is the main result; if it collapses under seed=1337, the paper must be rewritten. If it holds, the paper gains a much stronger robustness story in a context where patient-level inference and external validation are unavailable.

The slot should not be used for experiments that wait on X1-lite's outcome or revive V18. Idle slot has option value, but here the paper-headline robustness value is higher.

## 2. Q2 — Is Option A Highest EV?

**APPROVE.**

V14 proves V7 at λ=0.04 is seed-stable between seed 42 and 1337. It does **not** prove A4-mid at λ=0.08 is seed-stable. The stronger image_aux setting changes optimization pressure, loss balance, and potentially the basin reached by training. It needs its own seed replicate if the paper is going to center on it.

Option A gives three clean paper outcomes:

- **A4-mid-seed1337 ≈ A4-mid within 0.02 dB**: main result becomes much more credible.
- **A4-mid-seed1337 > V7 + 0.05 but below A4-mid**: effect is real, magnitude is seed-sensitive; paper can report a conservative range.
- **A4-mid-seed1337 near V7**: A4-mid was likely lucky; better to discover this before writing claims.

This is more important than another mechanism disambiguation run right now because X1-lite is already running and should answer the first mechanism question.

## 3. Q3 — Should Option B Run Now?

**REJECT for now.**

X1-v2-balanced is only useful if X1-lite lands in an ambiguous middle zone. Running it before X1-lite finishes is premature and weakens the Round 18 hard stop.

| X1-lite outcome | need X1-v2-balanced? |
|---|---|
| within 0.02 dB of A4-mid | no |
| near V7 or below | probably no; L1-only is weak |
| interior, 0.02-0.05 below A4-mid | maybe |

CL1 is a real design caveat, but it is not urgent enough to pre-launch a gradient-budget control before seeing whether ambiguity actually occurs.

## 4. Q4 — Is Option C Too Conservative?

**MODIFY.**

C is a reasonable backup, not the main plan. Keeping the slot idle would preserve governance and focus on paper, but it misses a clean chance to de-risk the headline. Since A4-mid-seed1337 is a single, bounded, content-justified run, it is not the kind of scope creep the stop rules were meant to prevent.

If Option A cannot be launched cleanly without distracting from X1-lite monitoring and paper drafting, then C becomes correct. Under the stated constraint that it does not interfere, A is better.

## 5. Q5 — Should Option D Be Excluded?

**REJECT D.**

X3-extend is the wrong use of the freed slot. The X3 report shows the A3-matched window is non-additive: X3.last reaches 36.8288, remains below A4-mid by 0.0651 dB, and stays in the V18 range. Extending to 200K would directly violate the X3 hard stop and mostly serve V18 sunk-cost psychology.

There is a theoretical scenario where long-window LoRA helps, but that is a separate research question and not the highest-value next slot for the current paper. It should not be smuggled in as a continuation.

## 6. Q6 — Better Option E?

**No better training option.**

The only non-training Option E I would endorse is paper-side work: write the A4/X3 results section and prepare the seed-robustness table shell. Among training options that satisfy the constraints, A is the cleanest.

Do not invent another model variant. Do not start X1-v2 before X1-lite. Do not run another λ point. Do not reopen V18.

## 7. Q7 — Bias Audit B97+

| ID | bias | severity | assessment | mitigation |
|---|---|---|---|---|
| B97 | A4-mid seed robustness fetish | MED | Seed replicate is valuable, but one replicate does not prove seed distribution | Phrase as seed-1337 robustness, not full variance estimate |
| B98 | CL1 overreaction | MED | X1-v2-balanced is tempting but premature | Wait for X1-lite outcome before any balanced control |
| B99 | V18 sunk-cost | HIGH | X3-extend keeps V18 alive after non-additivity | Explicitly ban X3 extension in this round |
| B100 | Slot-utilization bias | MED | A free GPU alone is not a reason | Justify A by paper-headline robustness, not utilization |
| B101 | Single-seed false security | MED | A4-mid-seed1337 is still one more seed, not a distribution | Report conservative language and keep limitations |

## 8. Minimal Design Spec For Option A

Base: clone `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`.

Change exactly these fields:

| field | value |
|---|---|
| `output_dir` | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/A4_mid_seed1337` |
| `run_name` | `first_hop_224_a4_mid_seed1337` |
| `seed` | `1337` |

All other fields must match A4-mid, including:

- `training.image_aux.lambda_start = 0.08`
- `training.image_aux.lambda_max = 0.08`
- full image_aux components: `l1_weight=1.0`, `ssim_weight=0.25`, `seam_weight=0.1`
- `training.max_steps = 160000`
- rollout step weights and LR schedule
- no decoder LoRA, no KL

Suggested stop rule:

> A4-mid-seed1337 is exactly one seed replicate of the locked A4-mid setting. Do not launch seed2024/seed7, λ variants, component-balanced controls, or any V18/X3 continuation from this slot without a new user-signed review.

## 9. If Option A Is Not Launched

If the user chooses C instead, use the idle-slot period for:

1. paper outline and result-table skeleton;
2. X3 non-additive ablation write-up;
3. X1-lite monitoring only;
4. prewriting the A4 response-curve figure and limitations section.

## 10. Final Recommendation

Use the freed slot for **A4-mid-seed1337**. It is the most direct way to de-risk the paper's main claim while X1-lite continues the mechanism track. Keep it bounded to one seed replicate, and do not allow it to become a seed sweep or another A4 tuning branch.
