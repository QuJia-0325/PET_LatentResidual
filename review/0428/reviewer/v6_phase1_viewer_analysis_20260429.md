# V6 Phase I Viewer Analysis - 2026-04-29

## 0. Viewer Verdict

V6 should be evaluated as its own from-scratch experiment, not as part of the V5 attribution file.

Current verdict:

| Decision | Status |
|---|---|
| V6 pilot | GO |
| V6 evidence strength | mixed |
| V6 full 200K approval | conditional, not yet approved |
| Main risk | pair supervision fits, but open-loop chain quality is poor before rollout ramp |

The short version:

> V6 is behaving as designed on supervision pressure: it has shifted training away from V3-style image domination and toward pair/transport pressure. However, V6 has not yet shown end-to-end chain benefit. The late Phase I rolling chain metrics are worse than V3 same-stage context. Because rollout is intentionally disabled before step 50K, this is a caution signal rather than a final failure verdict. The critical evidence will come from step 50K-75K, when rollout lambda and alpha begin to ramp.

## 1. Evidence Checked

| Evidence | Status |
|---|---|
| `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 design/config checked |
| `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml` | V3 baseline config checked |
| `review/0428/operator/log_snapshots/v6_transport_first_gpu1_train_snapshot_20260429_121604.log` | V6 latest snapshot checked through step 38050 |
| `review/0427/logs_train/v3_200k_transport_metrics_snapshot_20260427_1916.jsonl` | same-stage V3 comparison checked through step 40K |
| `review/0427/logs_train/v3_200k_transport_progress_summary_20260427_1916.log` | V3 best/full-run context checked |

## 2. Design Context

V6 is not a direct continuation of V5. It is a from-scratch transport-first run with a deliberately staged schedule:

| Dimension | V3 | V6 | Viewer implication |
|---|---:|---:|---|
| `loss.pair_weight` | 1.0 | 15.0 | V6 should strongly increase pair/endpoint pressure |
| rollout warmup | 20K | 50K | V6 Phase I intentionally disables rollout learning |
| rollout lambda | 0.02 -> 0.25 | 0 -> 4.0 | V6 rollout evidence only becomes meaningful after warmup |
| image aux | 0.005 -> 0.12 | fixed 0.04 | V6 should prevent V3-style late image domination |
| rollout alpha | 0 -> 1 by 80K | 0 -> 1 from 50K to 150K | V6 step 38050 is still pre-rollout |

Therefore, a V6 step-38050 chain metric is a Phase I stress signal, not a final rollout verdict.

## 3. Raw V6 Snapshot

Latest visible V6 snapshot reaches step 38050. It remains in Phase I:

| Field | Latest visible value |
|---|---:|
| latest train step | 38050 |
| latest validation step | 38000 |
| `lambda_roll` | 0.0000 |
| rollout `alpha` | 0.0000 |
| latest train `pair_frac` | 0.854 |
| latest train `img_frac` | 0.146 |

Selected rolling checkpoints:

| Step | pair_frac | img_frac | val_pair_total | val_chain_normal_mse | Viewer note |
|---:|---:|---:|---:|---:|---|
| 6000 | 0.973 | 0.027 | n/a | 0.000224 | early chain near V3 same-stage values |
| 10000 | 0.607 | 0.393 | n/a | 0.000242 | still acceptable rolling window |
| 20000 | 0.921 | 0.079 | n/a | 0.000970 | chain worsens despite pair pressure |
| 30000 | 0.911 | 0.089 | n/a | 0.001132 | poor chain window |
| 34000 | 0.698 | 0.302 | n/a | 0.000751 | chain still elevated |
| 36000 | 0.823 | 0.177 | 0.000002 | 0.000867 | pair objective fit, chain poor |
| 38000 | 0.588 | 0.412 | 0.000002 | 0.000975 | pair fit remains low; chain still poor |
| 38050 | 0.854 | 0.146 | n/a | n/a | latest train row, no val row yet |

Whole-log summary from the 0429 snapshot:

| Window | pair_frac mean | img_frac mean | val_chain_normal_mse mean | val_pair_total mean |
|---|---:|---:|---:|---:|
| V6 all visible steps | 0.775 | 0.225 | 0.000613 | 0.000179 |
| V6 step >= 30000 | 0.776 | 0.224 | 0.000864 | 0.000015 |
| V6 step >= 34000 | 0.755 | 0.245 | 0.000784 | 0.000007 |

## 4. Same-Stage V3 Comparison

Using the V3 200K metrics snapshot for steps up to 40K:

| Window | pair_frac mean | img_frac mean | val_chain_normal_mse mean | val_pair_total mean |
|---|---:|---:|---:|---:|
| V3 step <= 40000 | 0.280 | 0.711 | 0.000406 | 0.000087 |
| V3 step 30000-40000 | 0.089 | 0.896 | 0.000470 | 0.000014 |
| V6 all visible steps | 0.775 | 0.225 | 0.000613 | 0.000179 |
| V6 step >= 30000 | 0.776 | 0.224 | 0.000864 | 0.000015 |

Interpretation:

- V6 successfully changes the loss-pressure regime: V3 is image-led by 30K-40K, while V6 remains pair-led.
- V6 does not yet show chain-quality improvement. In late Phase I, its rolling `val_chain_normal_mse` is worse than V3 same-stage rolling metrics.
- V6 late `val_pair_total` is similar to V3 same-stage late `val_pair_total`, so the pair endpoint/velocity task is being fit. The issue is chain consistency under open-loop evaluation, not obvious failure to learn pair supervision.

## 5. Viewer Judgment

Current V6 evidence is mixed:

| Question | Status | Evidence |
|---|---|---|
| Did V6 implement the intended transport-first pressure? | yes | pair_frac mean 0.775 vs V3 0.280 up to 40K |
| Did V6 suppress image domination? | yes so far | img_frac mean 0.225 vs V3 0.711 up to 40K |
| Is pair supervision numerically fit? | mostly yes | late `val_pair_total` around 0.000002-0.000004 |
| Is chain quality good in Phase I? | no | late `val_chain_normal_mse` around 0.00075-0.00113 |
| Does this prove V6 failure? | no | rollout is intentionally disabled before 50K |
| Does this justify upgrading V6 confidence? | no | chain quality must recover after rollout/alpha ramp begins |

The correct viewer position is:

> V6 is behaving as designed on supervision pressure, but it has not yet shown end-to-end chain benefit. Phase I pair fitting alone is insufficient evidence. The 50K-75K transition is critical: if rollout/alpha ramp does not reduce chain `NORMAL` and tail errors, V6 should be paused or replanned rather than automatically continued to 200K.

## 6. Gate Recommendation

At the next checkpoint, do not judge only by `pair_frac`. Require a chain recovery signal:

| Gate | Required evidence | Action if missing |
|---|---|---|
| Step 50K | pair_frac remains transport-led, no NaN, `val_pair_total` stays low | continue into Phase II but monitor tightly |
| Step 60K-75K | rollout/alpha starts reducing `val_chain_normal_mse` or `val_chain_tail_mse` relative to late Phase I | pause and inspect rollout schedule / step weights |
| Step 100K | transport remains dominant and chain metrics approach or beat V3 same-window context | continue toward 150K/200K |

## 7. Final Position

V6 pilot remains GO, but V6 full-run approval should remain conditional. The run has passed the "changed training pressure" check, but has not passed the "chain quality recovers under rollout" check. The next real decision point is the Phase II ramp after step 50K, especially the 60K-75K window.