# Round 16 Peer Review — A3 Results Interpretation (GitHub Copilot)

- date: 2026-05-19
- reviewer: **GitHub Copilot**
- prompt: `PEER_REVIEW_PROMPT_round16_A3_results_20260519.md`
- scope: A3 / `V18_capacity_only` completed result, direct `decode(z_GT)` probe, trainer full-val chain metrics, V18 comparison
- stance: mechanism audit first, strategy second; do not collapse direct decoder evidence into transport rollout evidence

---

## 0. Executive Verdict

| item | verdict | reason |
|---|---|---|
| main path | **A — A3 kills the KL direct-decoder narrative** | `V18-cap.last(170K) - V18.step170k` is only `+0.0009/+0.0012/+0.0015/+0.0020 dB` on direct `decode(z_GT)`, far below the pre-registered `0.02 dB` relevance threshold. |
| scope of falsification | **MODIFY / narrow** | What is dead is “V18 direct `decode(z_GT)` gain is evidence for KL pullback.” This does not prove KL has no possible `z_pred` / rollout-path effect. |
| A3 chain implication | **strong but local** | A3@165K and V18.best@165K are essentially identical under trainer full-val chain MSE / `val_select_score`, so early V18 chain behavior is also capacity-explainable on that substrate. |
| V18 family narrative | **downgrade mechanism, not project end** | V18 is no longer a KL-success story. It becomes a decoder-capacity / marginal-transport result until V13/V14 decide image_aux and seed/noise scale. |
| new KL-specific run | **not high priority now** | A fresh V18-clean / KL-specific long run should wait. If revisited, it must target `z_pred` / rollout coupling, not direct GT-manifold reconstruction. |

One-line recommendation: **Withdraw KL-as-direct-decoder claims immediately; keep V13/V14 running; do not start another long KL control until the current substrate package returns.**

---

## 0.5 Evidence Checked

### Direct `decode(z_GT)` probe

Protocol: full `val`, `n=7403`, `PSNR_clip3(decode_crop(z_GT), x_target)`, no transport rollout.

| checkpoint | step | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|---:|
| V7.best | 160000 | 46.6356 | 48.7436 | 50.8319 | 52.6341 |
| V18.best | 165000 | 46.6850 | 48.7993 | 50.9012 | 52.7314 |
| V18.step170k | 170000 | 46.7212 | 48.8408 | 50.9527 | 52.8027 |
| V18.last | 200000 | 46.7482 | 48.8630 | 50.9635 | 52.7980 |
| V18-cap.last | 170000 | 46.7221 | 48.8420 | 50.9542 | 52.8047 |

Matched-step primary delta:

| comparison | D20 | D10 | D4 | NORMAL |
|---|---:|---:|---:|---:|
| V18-cap.last - V18.step170k | +0.0009 | +0.0012 | +0.0015 | +0.0020 |

Probe integrity check: all compared checkpoints loaded with `missing=0 unexpected=0`; V18 and V18-cap both wrapped the same 12 Linear LoRA modules in the last 2 decoder layers.

### Trainer full-val chain metrics

| metric | V18.best@165K | A3@165K | A3@170K |
|---|---:|---:|---:|
| `val_select_score` | 0.0009037494 | 0.0009037353 | 0.0009039954 |
| D20 MSE | 0.0003304456 | 0.0003304433 | 0.0003307622 |
| D10 MSE | 0.0002966660 | 0.0002966598 | 0.0002969189 |
| D4 MSE | 0.0002630942 | 0.0002630874 | 0.0002632126 |
| NORMAL MSE | 0.0002454947 | 0.0002454921 | 0.0002454063 |
| hop0 image total | 0.0075349057 | 0.0075349176 | 0.0075398990 |

Interpretation: A3@165K is a near-exact clone of V18.best@165K on trainer full-val chain metrics. A3@170K does not show a substantive `val_select_score` improvement; it slightly worsens D20/D10/D4 while NORMAL improves by a tiny amount.

### Mechanism code anchor

The KL source remains:

```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```

Because V18 has `use_pred_latent=true`, KL does not directly supervise `decode(z_GT)`. A3 then shows that even the possible indirect shared-LoRA explanation is unnecessary for the observed direct GT-manifold gain.

---

## 1. Q1 — Is the KL Direct-Decoder Narrative Dead?

**Verdict: APPROVE, with precise wording.**

Yes: the narrative “V18’s direct `decode(z_GT)` PSNR gain demonstrates KL pullback helping the decoder on the GT latent manifold” should be treated as basically dead.

The reason is not only B42. B42 already made the original direct path mechanistically suspicious because KL uses `z_pred`, not `z_dst`. A3 adds the empirical closure: with the same LoRA capacity and `lambda_kl=0.0`, `V18-cap.last@170K` matches `V18.step170k` on direct `decode(z_GT)` to within `+0.0020 dB` on NORMAL. That is below any meaningful threshold and is also consistent across D20/D10/D4.

What should not be said:

- “KL is proven useless everywhere.”
- “V18 has no possible `z_pred` effect.”
- “A3 settles transport rollout.”

Correct replacement wording:

> A3 falsifies the use of direct `decode(z_GT)` gains as evidence for KL pullback. If KL has value, it must be demonstrated on the predicted-latent / rollout path, not inferred from GT-latent decoder PSNR.

---

## 2. Q2 — Did A3 Chain Metrics Reproduce V18@165K?

**Verdict: APPROVE for trainer full-val chain metrics; MODIFY for broader chain claims.**

A3@165K essentially reproduces V18.best@165K under the trainer full-val substrate:

- `val_select_score`: A3 `0.0009037353` vs V18 `0.0009037494`
- NORMAL MSE: A3 `0.0002454921` vs V18 `0.0002454947`
- D20/D10/D4 are also separated only at the `1e-9` to `1e-8` displayed scale.

This is a surprisingly strong result. It means V18’s early selected checkpoint is not only direct-decode capacity-explainable; its trainer full-val chain score is also capacity-explainable at the matched 165K point.

Boundary:

- This is trainer full-val chain MSE / selection score, not the canonical full-val `PSNR_clip3` evaluator.
- It does not prove A3 reproduces V18.last@200K or any late-training transport effect.
- A3@170K does not show substantive further chain gain; it is mostly a plateau/noise result.

So the right conclusion is local but strong: **V18.best@165K is no longer evidence that KL contributed to the chain score.**

---

## 3. Q3 — Implications for V18/KL, Decoder Capacity, and Transport

**Verdict: MODIFY the mechanism map.**

### V18 / KL

V18 should be demoted from “KL pullback works” to “decoder LoRA capacity plus existing losses changes the decoder/chain behavior.” Existing V18 numbers remain real measurements, but their causal label changes.

The only remaining plausible KL question is narrower:

- Does KL on `z_pred` improve predicted-latent image realism or rollout stability?
- Does it alter optimization through shared LoRA weights in a way not visible in direct GT decoding?
- Does it matter after 170K under canonical rollout PSNR?

Those are possible, but they are no longer high-priority explanations for the known direct GT-manifold gain.

### Decoder capacity

Decoder LoRA capacity is now a demonstrated cause sufficient to reproduce the direct GT-manifold PSNR gain. This is a clean positive mechanistic result, but it is not the same as a transport-method win.

The paper-safe phrasing is:

> Lightweight decoder adaptation improves direct reconstruction from ground-truth RAE latents, but this improvement is largely decoupled from transport-chain quality.

### Transport

Transport remains the weak link. Direct decoder improvement can coexist with flat or marginal chain metrics. A3@165K matching V18.best@165K suggests the early chain score did not need KL either, but it does not tell us how to improve the transport model.

Immediate implication: the next method work should focus on rollout / `z_pred` path diagnostics, not another attempt to explain direct `decode(z_GT)` drift.

---

## 4. Q4 — What Can Be Concluded Now vs. What Must Wait?

**Verdict: APPROVE a two-tier conclusion.**

### Conclude now

- A3 falsifies KL as the explanation for direct `decode(z_GT)` improvement.
- Decoder LoRA capacity alone is sufficient for the V18@170K direct GT-manifold gain.
- A3@165K essentially reproduces V18.best@165K trainer full-val chain metrics.
- V18’s existing direct-decode evidence must be relabeled from “KL evidence” to “capacity evidence.”
- A new long V18-clean / KL control is not the current best use of compute.

### Must wait for V13 / V14

- Whether image_aux is actually helping or masking transport weaknesses.
- Whether the small V18/V13 deltas exceed seed/noise floor.
- Whether any paper narrative can honestly claim a positive method result rather than a marginal/negative diagnostic result.
- Whether the project should pivot to a transport intervention, decoder-capacity framing, or negative-result story.
- Whether V18’s canonical chain PSNR differences are meaningful relative to `d_pure`.

This is why main verdict A should not become project-ending verdict D yet. A3 closes one mechanism loophole; it does not close the experimental package.

---

## 5. Q5 — Is V18-Clean / New KL Control Still High Priority?

**Verdict: REJECT high priority now.**

I would not launch a new long KL-specific control before V13/V14 return.

Reasons:

- The original reason for V18-clean was to explain direct GT-manifold drift. A3 already explains that drift with capacity alone.
- V18 `use_pred_latent=true` means a clean KL question would need a redesigned target: predicted-latent / rollout behavior, not direct `z_GT` reconstruction.
- V13/V14 are already running and provide higher-level decision substrate: image_aux attribution and noise floor.
- Another KL control now risks chasing a shrinking mechanistic niche while the project’s strategic uncertainty is elsewhere.

What I would allow instead:

- Low-cost offline analysis from existing artifacts: compare A3 vs V18 per-slice direct deltas, per-timepoint tails, and if already feasible, canonical A3 chain PSNR with the same evaluator used for V18.
- After V13/V14, design a **specific** KL-on-`z_pred` control only if the remaining evidence says transport-side KL is still a plausible decision-maker.

---

## 6. Q6 — Paper Narrative Rewrite

**Verdict: MODIFY immediately.**

### Remove / downgrade

- Remove: “KL pullback improves decoder reconstruction on GT latents.”
- Remove: “V18 direct `decode(z_GT)` improvement supports the KL design.”
- Downgrade: “V18 is a successful transport improvement.” Current transport evidence is marginal and confounded by capacity.
- Downgrade: “Direct decoder PSNR is a proxy for rollout improvement.” A3 shows the two can decouple.

### Keep / reframe

- Keep: “Decoder LoRA capacity can improve direct reconstruction from ground-truth latents.”
- Keep: “V18/A3 reveal a gap between decoder-manifold quality and transport-chain quality.”
- Keep as pending: “image_aux contribution,” “seed/noise floor,” and “canonical chain significance.”
- Reframe the project contribution, if still pursued, as a diagnostic result unless V13/V14 produce a stronger positive substrate.

Suggested claim ledger:

| claim | current status | evidence |
|---|---|---|
| LoRA capacity improves direct `decode(z_GT)` | **SUPPORTED** | A3 direct probe matches V18@170K and improves over V7. |
| KL explains direct `decode(z_GT)` gain | **REJECTED** | B42 + A3 `lambda_kl=0` matched-step tie. |
| V18 improves transport chain meaningfully | **PARTIAL / weak** | V18 canonical NORMAL is slightly higher, but trainer chain metrics plateau and A3@165K matches V18.best. |
| image_aux is beneficial | **PENDING** | V13 not returned. |
| small deltas exceed seed noise | **PENDING** | V14 not returned. |

---

## 7. Q7 — New Biases B61+

### B61 — Direct-decode-to-chain overreach

A3 is strongest on direct `decode(z_GT)`. The prompt correctly warns about this, but the chain discussion can still accidentally inherit the direct-decode confidence.

Fix: every conclusion must label the substrate: direct GT decode, trainer full-val chain MSE, or canonical full-val chain PSNR.

### B62 — Capacity-sufficiency-as-sole-causality bias

A3 shows capacity is sufficient to reproduce direct V18@170K. It does not prove no other V18 factor ever matters, especially on `z_pred` or later rollout behavior.

Fix: say “sufficient for the observed direct GT-manifold gain,” not “the only possible mechanism everywhere.”

### B63 — Matched-step tunnel vision

The cleanest direct comparison is A3@170K vs V18.step170k. V18.last@200K has extra training time, and A3 was not trained to 200K.

Fix: do not use A3 to over-answer 170K-to-200K questions. Note that V18.last NORMAL direct decode is not better than A3, but keep late-stage transport claims separate.

### B64 — Trainer-MSE / canonical-PSNR mixing

A3 chain evidence is trainer full-val MSE. V18 final report also has canonical full-val `PSNR_clip3` chain values. These are not interchangeable.

Fix: if A3 canonical chain PSNR is needed, run the same standard evaluator; otherwise keep the comparison explicitly on trainer full-val MSE.

### B65 — V18-clean sunk-cost bias

Because V18-clean was previously planned as a natural follow-up, there is a risk of preserving it after its original rationale has collapsed.

Fix: require a new decision memo before any V18-clean / KL-specific run, with a `z_pred` / rollout-specific hypothesis.

### B66 — V13/V14 dependency bypass

A3 is satisfying and fast, so it can tempt a full strategy decision before the slower controls return.

Fix: A3 can rewrite the V18 mechanism story now, but it cannot decide image_aux, `d_pure`, seed significance, or final paper posture.

### B67 — False precision around tiny dB ties

The `+0.0020 dB` NORMAL delta is numerically precise but scientifically negligible under the pre-registered threshold.

Fix: report it as a tie / near-exact match; do not narrate the sign as meaningful.

### B68 — Negative-result shame bias

There is a temptation to rescue V18 by inventing a subtler KL story because “capacity-only explains it” feels less novel.

Fix: prefer the honest result. A clean falsification plus decoupling diagnosis is more valuable than a fragile positive claim.

---

## 8. Main Verdict

**Choose A:** A3 is enough to basically kill the KL direct-decoder narrative; the next meaningful focus is transport / `z_pred` path.

Secondary notes:

- **Not B:** a new KL-specific control is not the priority before V13/V14.
- **Not C as main path:** A3 permits a strong local mechanistic conclusion now.
- **Not D as main path yet:** V18 family should be downgraded mechanistically, but the whole project/paper posture still depends on V13/V14 and the seed/noise floor.

---

## 9. Immediate Executable Suggestions

### Do now while V13/V14 run

1. Write a short Round16 integration table separating three substrates: direct `decode(z_GT)`, trainer full-val chain MSE, canonical chain PSNR.
2. Update the active claim ledger: mark “KL explains GT direct decode” as rejected; mark “capacity improves GT direct decode” as supported; mark transport/image_aux/noise as pending.
3. Prepare V13/V14 result templates now so their outputs slot into the same claim ledger without another narrative reset.
4. Optionally summarize the existing A3 per-slice CSV for distribution/tail behavior; this is analysis-only and should not launch a new training run.
5. If GPU time is genuinely idle and non-conflicting, run A3 through the same canonical full-val chain PSNR evaluator used in V18 final. Treat it as metric alignment, not a new experiment.

### Pause until V13/V14 return

1. Do not launch V18-clean / new KL-specific long control.
2. Do not write paper claims that V18 is a KL success.
3. Do not decide project termination or pivot solely from A3.
4. Do not compare A3 direct `decode(z_GT)` PSNR against V18 canonical rollout PSNR as if they share substrate.

### After V13/V14 return

Use this decision tree:

| condition | next action |
|---|---|
| V13 shows image_aux matters and V14 noise floor is small | focus on transport/image_aux mechanism; V18-KL remains secondary. |
| V13 weak and V14 noise floor comparable to deltas | downgrade paper to marginal/negative diagnostic unless a new transport intervention is designed. |
| V14 noise floor tiny but V18/A3 still plateau | target rollout / `z_pred` path directly; do not revisit direct decoder KL. |
| strong unexplained V18 canonical rollout advantage remains | design a narrow KL-specific `z_pred` / rollout control with explicit hypotheses. |

---

## 10. Bottom Line

A3 is a clean mechanistic correction. It says: **the impressive-looking V18 direct GT-latent decoder gains were decoder-capacity gains, not KL evidence.** It also strongly suggests V18.best@165K trainer chain behavior does not need KL.

That is not a failure of the review process; it is the point of A3. The project should now stop spending narrative budget on direct-decoder KL and spend its remaining experimental budget on the unresolved substrate: V13 image_aux, V14 seed/noise, and the actual transport / `z_pred` rollout path.