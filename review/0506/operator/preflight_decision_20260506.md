# Plan F Phase 1 Pre-flight Decision (2026-05-06)

Decision: **PASS**

## Gate Results

- Option F sigma golden test: PASS
- V8 RNG byte-identical: true

## Rolling-val 5K Double-pass Metrics

The runbook decision uses the last rolling `[val]` row at step 4800. The shell wrapper completion marker was not tee'd into the per-pass train log, so the earlier supervisor BLOCKED decision was a log-marker false negative. All four train logs contain `Training done. Outputs at:` and final metrics.

| arm/pass | metric row | val_pair_total | val_chain_normal_mse | val_select_score |
|---|---|---:|---:|---:|
| V6 pass1 | `[val] step=04800` | 0.000499 | 0.000332 | 0.001126 |
| V6 pass2 | `[val] step=04800` | 0.000499 | 0.000332 | 0.001126 |
| V7 pass1 | `[val] step=04800` | 0.000498 | 0.000314 | 0.001069 |
| V7 pass2 | `[val] step=04800` | 0.000498 | 0.000314 | 0.001069 |

## Drift

- drift_v6_pair = 0.0
- drift_v7_pair = 0.0
- drift_v6_chain_normal = 0.0
- drift_v7_chain_normal = 0.0

## Decision Rule

Proceed iff Option F PASS, V8 RNG true, drift_v6_pair <= 1.0%, and drift_v7_pair <= drift_v6_pair.

Conclusion: **proceed to Phase 2 main training**.

## Resource Note

At launch time GPU2/GPU3 were free. GPU1 was occupied by another user's process (`PID 2614451`, `python train.py`), so V7/V8 are launched immediately on GPU2/GPU3 and V6_NOISE is queued to auto-start on GPU1 once it becomes free.
