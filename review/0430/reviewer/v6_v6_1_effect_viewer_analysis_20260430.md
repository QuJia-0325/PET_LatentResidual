# V6 / V6.1 Effect Viewer Analysis

Date: 2026-04-30
Role: viewer / reviewer

## Evidence Sources

- Pulled Gitee `foc_lite_hop0` to `53c80f2` (`docs(0430): add v6 log-only snapshots`).
- `review/0430/operator/v6_v6_1_training_status_20260430_211557.md`
- `review/0430/operator/log_snapshots/v6_transport_first_metrics_snapshot_20260430_211557.jsonl`
- `review/0430/operator/log_snapshots/v6_1_rollout_floor_metrics_snapshot_20260430_211557.jsonl`
- `review/0430/operator/log_snapshots/v6_transport_first_gpu1_train_log_snapshot_20260430_212930.log`
- `review/0430/operator/log_snapshots/v6_1_rollout_floor_gpu3_train_log_snapshot_20260430_212930.log`
- `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml`
- `configs/pet_flow/pet_flow_first_hop_224_v6_1_rollout_floor.yaml`

## Executive Verdict

V6 is no longer in the earlier Phase-I failure-looking state. Once rollout ramp starts after 50K, chain metrics recover materially: the best rolling point is now step 81.6K with `val_select_score=6.61e-4` and `val_chain_normal_mse=1.82e-4`. This is still not a proven win over V3: V3 best rolling NORMAL MSE was about `1.645e-4`, so V6 best is still about 10.7% worse on that rolling metric, and the latest V6 window at 83.2K is worse than the 81.6K best. But the direction is clearly positive enough to continue and to trigger full-val evaluation of the best V6 checkpoint.

V6.1's rollout floor looks useful in the early regime. It does not destroy the pair-led training regime, because average rollout fraction is only about 1.2% through 20K, but it improves early chain quality versus V6 at aligned steps. The strongest aligned comparison is step 20.8K: V6.1 has `val_chain_normal_mse=4.11e-4` versus V6 `7.10e-4`, about 42% lower. However, V6.1's latest window is worse than its own 12K best, so it should be treated as promising, not yet selected over V6.

The current evidence supports the core transport-first hypothesis more than it refutes it: increasing transport pressure plus activating chain supervision improves open-loop chain behavior. It still does not close the claim that V6/V6.1 beats V3 under the claim metric; that requires full-val clip3 PSNR on the best checkpoints.

## Latest / Best Rolling Metrics

The 21:29 log is slightly newer than the 21:15 JSONL snapshot for train progress and includes V6 validation at 83.2K.

| Experiment | Latest train step | alpha | lambda_roll | pair_frac | roll_frac | img_frac | Latest val step | Latest select | Latest NORMAL | Best rolling step | Best select | Best NORMAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V6 | 83,600 | 0.336 | 1.344 | 0.813 | 0.091 | 0.096 | 83,200 | 0.000763 | 0.000225 | 81,600 | 0.000661 | 0.000182 |
| V6.1 | 21,000 | 0.000 | 0.050 | 0.579 | 0.006 | 0.415 | 20,800 | 0.001301 | 0.000411 | 12,000 | 0.000762 | 0.000213 |

Notes:

- Fractions are weighted loss fractions / supervision-pressure proxies, not direct gradient norm shares.
- All validation metrics here are rolling-window metrics, not full-val claim metrics.
- Lower MSE/select is better.

## V6: Phase-II Recovery Is Real But Not Stable Yet

V6's key question after the 0429 analysis was whether the 50K rollout ramp could repair chain quality. The answer from 50K-83K is yes, directionally.

| V6 segment | Val events | Mean select | Best step | Best select | Mean NORMAL | Best NORMAL |
|---|---:|---:|---:|---:|---:|---:|
| 0-20K | 50 | 0.001241 | 6,000 | 0.000772 | 0.000392 | 0.000224 |
| 20-40K | 50 | 0.001954 | 35,200 | 0.001273 | 0.000833 | 0.000525 |
| 40-50K | 25 | 0.001445 | 46,800 | 0.000953 | 0.000533 | 0.000345 |
| 50-60K | 25 | 0.001154 | 58,400 | 0.000825 | 0.000394 | 0.000276 |
| 60-70K | 25 | 0.001034 | 69,600 | 0.000686 | 0.000314 | 0.000209 |
| 70-80K | 25 | 0.000992 | 75,600 | 0.000678 | 0.000281 | 0.000195 |
| 80K+ | 7 | 0.000867 | 81,600 | 0.000661 | 0.000245 | 0.000182 |

The supervision mix changes in the intended direction:

| V6 train window | Mean pair_frac | Mean roll_frac | Mean img_frac | Mean lambda_roll | Mean alpha |
|---|---:|---:|---:|---:|---:|
| 0-20K | 0.780 | 0.000 | 0.220 | 0.000 | 0.000 |
| 20-40K | 0.771 | 0.000 | 0.229 | 0.000 | 0.000 |
| 40-50K | 0.753 | 0.000 | 0.247 | 0.000 | 0.000 |
| 50-60K | 0.720 | 0.041 | 0.239 | 0.201 | 0.050 |
| 60-70K | 0.685 | 0.115 | 0.200 | 0.601 | 0.150 |
| 70-80K | 0.626 | 0.171 | 0.203 | 1.001 | 0.250 |
| 80K+ | 0.545 | 0.245 | 0.210 | 1.265 | 0.316 |

Interpretation:

- V6 has passed the narrow Phase-II recovery gate: chain quality improves as rollout pressure turns on.
- The improvement is not monotonic. Step 72K is a bad rolling window (`select=0.001559`, `NORMAL=0.000417`), while 81.6K is the best so far. This means rolling-window volatility remains high.
- V6's best rolling NORMAL MSE (`1.82e-4`) is close to but still worse than the known V3 best rolling NORMAL MSE (`1.645e-4`). Therefore the correct claim is "V6 is approaching V3 and may become competitive," not "V6 has beaten V3."

## V6.1: Rollout Floor Helps Early Chain Quality

V6.1 differs from V6 by setting `rollout.lambda_start: 0.05` instead of `0.0`; alpha is still 0 during Phase I, so this is a low-level chain supervision floor rather than open-loop self-feeding.

Aligned early comparisons show V6.1 is consistently no worse at tiny steps and materially better after 12K:

| Target step | V6 select | V6 NORMAL | V6.1 select | V6.1 NORMAL | V6.1/V6 select | V6.1/V6 NORMAL |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.000786 | 0.000230 | 0.000786 | 0.000230 | 0.999 | 0.999 |
| 2,000 | 0.000860 | 0.000257 | 0.000859 | 0.000257 | 0.999 | 0.999 |
| 4,000 | 0.000852 | 0.000252 | 0.000851 | 0.000252 | 0.999 | 0.998 |
| 6,000 | 0.000772 | 0.000224 | 0.000770 | 0.000223 | 0.998 | 0.996 |
| 8,000 | 0.001313 | 0.000371 | 0.001309 | 0.000368 | 0.997 | 0.993 |
| 12,000 | 0.000816 | 0.000249 | 0.000762 | 0.000213 | 0.933 | 0.855 |
| 16,000 | 0.001141 | 0.000401 | 0.000959 | 0.000283 | 0.841 | 0.705 |
| 20,000 | 0.002751 | 0.000970 | 0.002205 | 0.000621 | 0.801 | 0.640 |
| 20,800 | 0.001748 | 0.000710 | 0.001301 | 0.000411 | 0.744 | 0.579 |

V6.1 segment summary so far:

| V6.1 segment | Val events | Mean select | Best step | Best select | Mean NORMAL | Best NORMAL |
|---|---:|---:|---:|---:|---:|---:|
| 0-20K | 50 | 0.001147 | 12,000 | 0.000762 | 0.000331 | 0.000213 |
| 20-40K partial | 2 | 0.001447 | 20,800 | 0.001301 | 0.000457 | 0.000411 |

Interpretation:

- The `lambda_start=0.05` floor appears to reduce early open-loop chain drift without changing the experiment into an image-led or rollout-led regime.
- V6.1's 0-20K mean NORMAL MSE is about 15.6% lower than V6's 0-20K mean NORMAL MSE (`3.31e-4` vs `3.92e-4`).
- V6.1's latest window has degraded from its 12K best. This is not yet a stable superiority claim; it is a promising early signal that needs 40K-50K confirmation.

## What This Says About The Core Hypothesis

The current results are consistent with the E1-derived hypothesis that transport/chain behavior is the dominant bottleneck:

- V6 changed the loss-pressure allocation sharply toward pair/transport, and pair supervision became very small while chain quality initially remained poor. This showed pair fitting alone is insufficient.
- When V6 rollout pressure turns on after 50K, chain metrics improve strongly. That supports the need for explicit chain/open-loop supervision, not only GT-input pair fitting.
- V6.1 adds a small rollout floor during Phase I and improves early chain metrics while preserving pair dominance. That supports the idea that the original V6 strict no-rollout Phase I was under-constraining chain composition.

The hypothesis should still be worded carefully: current logs support "transport/chain supervision is necessary and increasingly effective," not "we have already solved transport" or "V6 beats V3."

## Recommended Next Actions

1. Keep V6 running. It is now in the informative part of Phase II, and stopping before 100K would waste the key ramp evidence.
2. Run full-val clip3 PSNR on V6 best rolling checkpoint around 81.6K, plus the latest/best available checkpoint if the checkpointing layout permits it. This is the highest-priority arbiter.
3. Keep V6.1 running at least to 50K. The main comparison is V6.1 40K-50K versus V6 40K-50K and V6 at 50K, not the 12K best alone.
4. If V6 continues to show large spikes after 90K-100K, inspect whether alpha/lambda ramp is too aggressive or whether rolling-window slice composition explains the spikes before changing the schedule.
5. Do not claim V6/V6.1 success from rolling metrics alone. Use these logs to choose full-val checkpoints and decide whether V6.1 should replace V6 as the main transport-first continuation.
