# Peer Review Round 16 - Review D Audit

- date: 2026-05-19
- reviewer: **Review D (GitHub Copilot)**
- scope: A3 capacity-only result interpretation and V18/KL strategic impact
- prompt: `PEER_REVIEW_PROMPT_round16_A3_results_20260519.md`
- verdict summary: **A, with strict scope limits**

## Executive Verdict

**Main verdict: A** - A3 is strong enough to retire the claim that current V18's `decode(z_GT)` direct gain is evidence for KL pullback success. The matched-step delta `V18-cap.last(170K) - V18.step170k = +0.0020 dB` on NORMAL is a practical tie, and the 165K chain metrics also nearly duplicate V18.best. In plain terms: the observed early V18 gains are explainable by decoder LoRA capacity without KL.

That does **not** mean the whole project is over, nor that every V18-family question is fully answered. It means the KL/direct-decoder story should be withdrawn. The remaining live question is whether any intervention can improve the `z_pred` / rollout chain path in a way that exceeds d_pure and survives V13/V14 interpretation.

## Evidence Checked

- A3 completion report: `review/0517/V18_capacity_only/V18_CAPACITY_ONLY_A3_REPORT_20260519.md`
- A3 direct GT-latent probe: `review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_REPORT.md`
- A3 probe summary JSON: `review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_SUMMARY.json`
- A3 metrics JSONL full-val rows at 165K and 170K: `review/0517/V18_capacity_only/V18_capacity_only_metrics_20260518.jsonl`
- V18 canonical full-val result: `review/0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md`
- B42 code anchor: `train_first_hop.py`, where `use_pred_latent=true` makes KL use `main_out["z_pred"]` rather than `main_batch["z_dst"]`
- Standing-rule context: `review/0517/V18_decoder_lora/V18_design_rationale.md`, especially B43/B44/B45

## Q1 - Does A3 Basically Kill The KL Direct-GT Explanation?

**Verdict: APPROVE A.**

Yes, for the direct `decode(z_GT)` interpretation. A3 was designed precisely to keep decoder LoRA capacity and remove KL. It nearly exactly reproduces V18@170K on the direct GT-manifold probe:

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18-cap.last - V18.step170k | +0.0009 | +0.0012 | +0.0015 | +0.0020 |

This is far below the pre-registered `0.02 dB` practical threshold and is consistent across all four timepoints. In the presence of B42/B44, there is no longer a reasonable basis to say current V18's direct GT-manifold improvement is mainly caused by KL pullback.

The most accurate wording is:

> KL is falsified as the explanation for current V18's direct `decode(z_GT)` improvement. KL effects on `z_pred` / rollout coupling remain unproven, not supported.

I would avoid the shorter phrase "KL 被基本证伪" unless it is immediately qualified. A3 does not test every possible corrected KL design. It tests the actual V18 configuration that was run.

## Q2 - Did A3 Also Reproduce V18@165K Chain Metrics?

**Verdict: APPROVE with one caveat.**

The 165K full-val trainer metrics are essentially identical:

| metric | A3@165K | V18.best@165K | delta |
|---|---:|---:|---:|
| val_select_score | 0.0009037353 | 0.0009037494 | -0.0000000141 |
| NORMAL MSE | 0.0002454921 | 0.0002454947 | -0.0000000026 |

That is strong evidence that at least the 160K -> 165K observed V18 chain behavior is also reproducible without KL. So the statement "V18 worked" should be rewritten as:

> Decoder LoRA capacity gave a tiny early chain/decoder effect, but the KL design has not shown additive contribution.

The caveat: these are trainer full-val MSE metrics, not the canonical full-val PSNR evaluator. They are still highly relevant because the comparison is matched at the same training step and same metric family, but downstream paper claims should not mix this table with canonical PSNR deltas without saying which metric is being used.

## Q3 - Meaning For V18 / Decoder / Transport Lines

**Verdict: MODIFY the research allocation.**

### V18 / KL Line

The actual V18 KL line should be downgraded hard. It has two problems now: B42/B44 already removed direct GT-path causality, and A3 shows the measured direct gain is reproducible with `lambda_kl=0`.

`V18-clean(use_pred_latent=false)` is no longer high priority. It could answer one narrow question: whether a corrected KL anchor on `z_GT` can produce a different effect than the buggy V18 config. But that is a new design rescue experiment, not a necessary next control for interpreting current V18.

### Decoder Capacity Line

A3 supports this statement:

> Decoder capacity can improve direct GT-manifold reconstruction, but that gain is weakly coupled to transport-chain improvement.

This makes rank/block sweeps low expected value unless the goal is explicitly direct decoder reconstruction. If the research goal is NORMAL rollout PSNR, direct decode improvements are not enough.

### Transport Line

A3 strengthens the hypothesis that the unsolved part is the `z_pred` / rollout / chain path. But it also warns that the whole V18 family may be marginal for the main metric. The next transport-side intervention should wait for V13/V14 unless it is just planning or code reading; we still need the image_aux and d_pure baselines to know what improvement scale is meaningful.

## Q4 - What Can Be Concluded Now vs Must Wait

**Verdict: APPROVE the local conclusions; WAIT on project-level claims.**

### Conclusions Safe Now

1. Do not cite `decode(z_GT)` improvement as KL evidence.
2. Current V18's direct GT-manifold gain should be attributed primarily to decoder LoRA capacity.
3. A3@165K nearly duplicates V18.best@165K chain metrics, so early V18 chain gain has no demonstrated KL additive component.
4. Any V18 result must retain the "buggy KL config / use_pred_latent=true" qualifier.
5. Direct decode and transport rollout should be treated as separate metrics with separate claims.
6. New KL controls are not the highest-priority response to A3.

### Conclusions That Must Wait For V13 / V14

1. Whether V18's `+0.030` / `+0.062 dB` canonical NORMAL PSNR deltas exceed d_pure noise.
2. Whether the project has hit a ceiling or merely exhausted this decoder-LoRA branch.
3. How much of the V7/V8 history was true image_aux vs step-weight/config confounding.
4. Whether there is a coherent paper narrative around V7/V13/V14 independent of V18.
5. Whether transport-side work needs a new architecture, new loss, or just better attribution.

## Q5 - Keep V18-clean / New KL Control High Priority?

**Verdict: REJECT high priority.**

V18-clean can remain as a backlog item, but A3 removes it from the urgent path. If kept, it should be framed as a corrected-design experiment answering one narrow question:

> Does a properly anchored KL pullback (`use_pred_latent=false`) change rollout behavior enough to matter beyond seed noise?

It should not be framed as needed to interpret A3 or current V18. Current V18 is already interpretable: capacity explains the direct gain, and KL additive value is not shown.

More decoder LoRA rank/block sweeps are also low priority. They risk optimizing direct `decode(z_GT)` while leaving chain PSNR marginal. The more reasonable next research direction is transport-side, but only after V13/V14 calibrate the significance and attribution landscape.

## Q6 - Paper Narrative Rewrite

**Verdict: MODIFY toward B + C.**

Reject narrative A:

> "KL pullback improves decoder manifold, but transport swallows it."

That is no longer defensible for current V18.

The honest near-term narrative is a combination:

> Decoder capacity can improve direct GT-manifold reconstruction, but this does not automatically convert into meaningful transport-chain gain. The V18 family is currently a marginal / partial result until V13 and V14 resolve image_aux attribution and d_pure significance.

If a paper includes V18 at all, A3 should be included as a negative control or ablation in the main text, not hidden. It is central to the claim boundary. If the paper ultimately pivots away from V18, A3 can move to appendix as an integrity/control story.

Immediate paper action: remove KL-success language now. Keep only capacity/direct-decode and direct-vs-chain decoupling language.

## Q7 - New Biases B61+

### B61 - Over-terminal A3 Bias

A3 is strong, but it only resolves the current V18/KL direct-decode interpretation. It does not resolve V13, V14, project ceiling, or paper viability.

Fix: every strategic conclusion should be labeled `resolved by A3` or `awaiting V13/V14`.

### B62 - Capacity Anchor Bias

A3 makes capacity the best explanation for V18 direct gain, but it does not imply every historical gain came from capacity.

Fix: keep capacity attribution scoped to V18/A3 matched controls.

### B63 - Direct-Decode / Chain Metric Conflation

The prompt is careful overall, but the temptation remains to use direct `decode(z_GT)` PSNR to argue about rollout success.

Fix: separate all claims into direct-decoder, trainer chain MSE, and canonical rollout PSNR buckets.

### B64 - Metric-Family Mixing

Q2 uses trainer full-val MSE while Q6 paper narrative will likely use canonical PSNR. They are compatible for local control interpretation, but not interchangeable.

Fix: when writing integration, keep the A3@165K/V18@165K comparison as "trainer full-val chain MSE tie", not canonical PSNR.

### B65 - KL Rescue Drag

There is a risk of using V18-clean as a way to keep the KL story alive after A3 has answered the current design question.

Fix: V18-clean must be demoted unless it has a fresh, pre-registered rollout claim and a clear stop rule.

### B66 - Waiting-Dependency Drift

The prompt correctly says V13/V14 are pending, but the strength of A3 may pull the integration toward premature ceiling claims.

Fix: no ceiling or significance language until V14; no image_aux attribution until V13.

## Main Verdict

**A: A3 is enough to basically kill the KL direct-decoder narrative; future priority should shift toward transport / `z_pred` path, while project-level conclusions wait for V13/V14.**

I do not choose B because a new KL-specific control is not the next priority. I do not choose C because A3 already supports strong local conclusions. I do not choose D as the main verdict because V18-family downgrade is real, but project/paper viability still depends on V13/V14.

## Immediate Recommendations

### Do Now While V13/V14 Run

1. Write Round 16 integration with a clear claim table: direct decoder, trainer chain MSE, canonical rollout PSNR, pending V13/V14.
2. Update V18 narrative notes to remove KL-success language.
3. Mark V18-clean, rank sweeps, and KL variants as low-priority backlog pending V13/V14.
4. Prepare a transport-side hypothesis list, but do not launch a new long run until V13/V14 return.
5. Preserve A3 as a required negative control for any future V18 discussion.

### Pause Until V13/V14

1. Paper claims that V18 improves beyond noise.
2. Project-ceiling conclusions.
3. New long GPU runs for V18-clean or rank/block sweeps.
4. Any claim attributing V7/V8 gains to image_aux.
5. Any final decision that the transport line is the sole remaining path.