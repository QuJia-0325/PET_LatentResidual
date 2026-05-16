# Lipschitz Gate Report (§0)

- generated_at: 2026-05-17 00:37 CST
- script: `tools/estimate_per_hop_lipschitz.py`
- log: `review/0517/disambig/logs/lipschitz_v7_v8_v6noise_gpu1_20260517_003433.log`
- protocol: finite-difference estimate of `model.predict_latent_step(z_src)` per hop, `n_samples=64`, `eps=1e-3`, split=`val`.
- checkpoints: V7/V8/V6_NOISE `last.pt` from `review_0505_runs`.

## Results

| Run | L0 | L1 | L2 | L3 | max L | closed-form w(measured L) | current step_weights | alignment distance |
|---|---:|---:|---:|---:|---:|---|---|---:|
| V7 | 1.004051 | 1.008056 | 1.008248 | 1.009087 | 1.009087 | [0.6191, 2.0000, 2.6446, 2.0004] | [0.6106, 2.0000, 2.7041, 2.0423] | 0.570982% |
| V8 | 1.002725 | 1.006494 | 1.005453 | 1.006740 | 1.006740 | [0.6181, 2.0000, 2.6558, 2.0148] | [0.5000, 2.0000, 1.5000, 1.0000] | 14.079950% |
| V6_NOISE | 1.004658 | 1.005004 | 1.005195 | 1.005676 | 1.005676 | [0.6167, 2.0000, 2.6564, 2.0179] | [0.5000, 2.0000, 1.5000, 1.0000] | 14.110852% |

## Gate Decision

For the V9 pre-launch gate, the binding reference is V7:

- `alignment_distance_pct(V7)=0.570982% < 10%`.
- `max(L_j,V7)=1.009087 < 1.10`.

Decision: **L≈1 confirmed**. Under the runbook threshold, V7 step_weights remain a valid closed-form instantiation, and V9 is not blocked by the Lipschitz gate.

## Notes

- V8 and V6_NOISE use the older V6 empirical step_weights, so their alignment distance to the measured-L closed-form is about 14%. This is expected and supports the review claim that V7 vs V8/V6_NOISE is not a single-axis comparison.
- The script is no longer a stub; the JSON files contain per-hop finite-difference statistics and no `stub_note`.
