# V5 Attribution Viewer Update - 2026-04-29

## 0. Updated Verdict

This update checks the latest 2026-04-29 training snapshots against the 2026-04-28 reviewer position.

The core reviewer conclusion does **not** need reversal. It needs a status update and one added caution:

> The latest null-control log strengthens the original direction: continuing from V3 best with the V3 recipe remains mostly image-dominated and still does not show stable rolling-val improvement. However, attribution is still not closed because null-control full-val, synced V5 rollout-heavy full-val summaries, and Path A diagnostics are still absent from the local review tree.

Decision remains:

| Decision | Updated status |
|---|---|
| Treat V5 attribution as closed causal proof | NO-GO |
| Approve V6 200K solely from current V5 attribution | NO-GO |
| Use latest NC data as stronger support for the recipe/supervision-pressure hypothesis | YES, with full-val pending |

## 1. Evidence Checked

Local files checked in this update:

| Evidence | Status |
|---|---|
| `review/0428/operator/log_snapshots/v5_null_control_gpu3_train_snapshot_20260429_121604.log` | readable; latest NC step 123550 |
| `review/0428/local/v5_attribution_analysis_20260428.md` | updated with 0429 NC status |
| `review/0428/supervisor/v5_attribution_supervisor_verdict_20260428.md` | methodologically valid, but numeric status is older |
| `review/0428/operator/review/` and `review/0427/` | no synced V5/null-control full-val JSON/CSV or Path A result found locally |

## 2. Null-Control Update

The previous reviewer doc only had null-control through step 100850. The latest snapshot extends it to step 123550, about 36750 new steps after V3 best and roughly 73% of the intended +50K matched budget.

### 2.1 Weighted Loss Fractions

These are weighted scalar loss fractions / supervision-pressure proxies, not direct gradient norm shares.

| Step | pair_frac | roll_frac | img_frac | Interpretation |
|---:|---:|---:|---:|---|
| 88000 | 0.012 | 0.068 | 0.920 | image dominated |
| 92000 | 0.024 | 0.061 | 0.915 | image dominated |
| 96000 | 0.018 | 0.055 | 0.927 | image dominated |
| 100000 | 0.032 | 0.097 | 0.871 | image dominated |
| 105000 | 0.024 | 0.160 | 0.816 | image still dominant |
| 110000 | 0.028 | 0.063 | 0.909 | image dominated |
| 115000 | 0.008 | 0.053 | 0.939 | image dominated |
| 120000 | 0.012 | 0.100 | 0.888 | image dominated |
| 123550 | 0.072 | 0.180 | 0.748 | transport pressure rises, image still largest |

Update to interpretation: the added 100850 -> 123550 span strengthens the observation that V3-recipe continuation remains mostly image-led. The last visible train row has higher transport pressure (25.2%), so the cleanest wording is not "always 90% image," but "mostly image-dominated, with late batch-level transport-pressure spikes."

### 2.2 Rolling Validation

| Step | window_start | val_chain_normal_mse | Note |
|---:|---:|---:|---|
| 87200 | 2752 | 0.000169 | near V3 rolling best |
| 92800 | 3648 | 0.000164 | near V3 rolling best |
| 93200 | 0 | 0.000159 | window wrap; not a phase transition |
| 95600 | 384 | 0.000362 | worse window |
| 100000 | 1088 | 0.000223 | mid window |
| 100800 | 1216 | 0.000294 | worse window |
| 110000 | 2688 | 0.000166 | good window returns |
| 120000 | 576 | 0.000191 | good-to-mid window |
| 123200 | 1088 | 0.000226 | mid window |

Update to interpretation: rolling-val still alternates by window and does not establish stable improvement. The latest log therefore supports the earlier reviewer position rather than replacing it.

## 3. V6 Analysis Location

Detailed V6 analysis has been split out into [v6_phase1_viewer_analysis_20260429.md](v6_phase1_viewer_analysis_20260429.md). This file now stays focused on V5 attribution and null-control evidence.

## 4. Local vs Supervisor Analysis

The latest local document has incorporated the 0429 state and is broadly aligned with the reviewer position after downgrading proof language to hypothesis language.

Two wording caveats remain:

| Local wording | Reviewer-safe update |
|---|---|
| "LR floor 已排除" | "LR-floor / continued-training explanations are weakened by the longer NC run, but not fully excluded until full-val and matched-budget comparison are complete" |
| "image_aux 占据梯度预算" | "image_aux dominates weighted loss fraction / supervision pressure" |

The supervisor verdict remains correct in principle, but it is numerically stale: it discusses null-control around step 100850. Its V5 attribution decision logic still holds: V5 causal closure requires V5 full-val, null-control full-val, and Path A evidence.

## 5. What Needs Updating From The 0428 Reviewer Doc

The 0428 reviewer doc should be read with these updates:

| 0428 item | 0429 update |
|---|---|
| NC evidence through step 100850 | replace with NC through step 123550 |
| NC training budget ~14K new steps | replace with ~36.75K new steps, about 73% of +50K matched budget |
| image fraction repeatedly 87-94% | still mostly true, but latest visible row is 74.8% image / 25.2% transport |
| rolling-val has no stable improvement | unchanged and now better supported by longer NC trace |
| full-val / Path A missing locally | unchanged |

## 6. Recommended Next Operation

Do not make a final V5 attribution claim yet. The next useful operation is still:

1. Let null-control finish the matched +50K budget, or at least reach the agreed checkpoint gate.
2. Sync V5 rollout-heavy full-val summaries into the review tree.
3. Run/sync null-control full-val under the same clip3 protocol.
4. Run/sync Path A diagnostic.
5. For project-level go/no-go decisions, combine the closed V5 attribution table with the standalone V6 analysis.

## 7. Final Updated Position

The 0429 logs make the V5 reviewer position stronger, not weaker. The update is quantitative rather than directional: null-control has now run much longer and still looks consistent with V3-recipe image-dominated plateau behavior, but the evidence boundary remains the same. Full-val and Path A are still required before treating V5 attribution as causal closure.