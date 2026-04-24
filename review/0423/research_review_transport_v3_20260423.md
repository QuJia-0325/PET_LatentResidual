# Transport v3 Research Review (2026-04-23)

## 1. Context

- Branch: `foc_lite_hop0`
- Review method: `$research-review` via external sub-agent (`gpt-5.4`, `xhigh`)
- Agent ID: `019db986-0bf5-7063-97c0-8855d819ae37`
- Goal: audit the **rationality** and **irrationality/risk** of v3 transport changes.

v3 relative to v2:
1. `transport.pair_loss_weights`: `[1.00,1.05,1.10,1.20] -> [1.20,1.10,1.05,1.00]`
2. `training.rollout.step_weights`: `[1.00,1.10,1.20,1.30] -> [1.30,1.20,1.10,1.00]`
3. `training.best_metric_terms(val_chain_d20_mse).weight`: `0.15 -> 0.50`
4. `loss.pair.velocity_rebalance.enabled`: `false -> true`

Prior unified full-val baseline (`clip_max=3`):
- `C_best transport_avg=36.2055`
- `v2_best transport_avg=36.1971`

## 2. Round-1 Review Summary (Brutal)

### 2.1 Reasonable points (accepted)

1. Hop0 reweighting is semantically aligned with dataset pair order (`hop_idx=0` is `D50->D20`).
2. Rollout step0 emphasis is a moderate shift (not an extreme rewrite) and is technically valid.
3. Diagnosing previous selector as tail-biased is directionally fair.
4. `velocity_rebalance` implementation is technically sound (detached ratio, clipped scale).
5. Intervention scale is not over-aggressive for a probe experiment.

### 2.2 Unreasonable / risky points (priority)

1. **Critical**: v3 is a bundled change (3 training + 1 selection), so causal attribution is weak.
2. **Critical**: selector changed under rolling-window val (`max_val_batches=64`), which can produce a fake `best.pt`.
3. **High**: bottleneck story may be overstated; largest displacement alone is insufficient evidence.
4. **High**: external claim metric is full-val clip3 PSNR, but selection metric is weighted chain MSE mix.
5. **High**: hop0 may be overcounted (pair + rollout + selector all pushing hop0).
6. **Medium-High**: rebalance scale is batch-level, not hop-specific; may amplify wrong signal.
7. **Medium**: repeated val-set tuning risks leaderboard fitting for tiny deltas.
8. **Medium**: expected gain may be overclaimed relative to intervention magnitude.

### 2.3 Hidden confounders

- Rolling validation window variance
- Selector-only gain without true training gain
- MSE-vs-PSNR metric mismatch
- Hop0 local gain but downstream chain unchanged/worse
- Single-seed variance around tiny margins

## 3. Round-2 Practical Decision Tree (next 24h)

### 3.1 Continue/stop gates for current running v3

Pre-train health:
- If no `metrics.jsonl` `event=train` within 2h, stop/relaunch.
- If no first `event=val` at `step=400` within 4h, stop/relaunch.

At 2k steps (use recent medians):
- Continue if:
  - `val_chain_d20_mse <= 4.5e-4`
  - `val_chain_normal_mse <= 4.5e-4`
  - `val_rollout_step_0 <= 1.20e-3`
  - train median: `0.20 <= pair_frac <= 0.85`, `vel_reb <= 3.5`
- Stop if any two fail, or `vel_reb >= 3.9` for >=80% of last 20 train records.

At 5k steps:
- Continue if:
  - `val_chain_d20_mse <= 3.8e-4`
  - `val_chain_normal_mse <= 6.0e-4`
  - `val_rollout_step_0 <= 1.05e-3`
  - `0.05 <= pair_frac <= 0.60`
- Stop/relaunch if:
  - `val_chain_d20_mse > 4.5e-4` OR
  - `val_chain_normal_mse > 7.5e-4` OR
  - `val_rollout_step_0 > 1.25e-3`

At 10k steps:
- Continue to 50k only if:
  - `val_chain_d20_mse <= 3.1e-4`
  - `val_chain_normal_mse <= 6.0e-4`
  - `val_rollout_step_0 <= 9.0e-4`
  - `0.02 <= pair_frac <= 0.35`
- If fast full-val available: require `transport_avg > 36.20`; strong go is `>36.24`.

### 3.2 If keeping this v3 run (publishable protocol)

Treat this run as a **compound intervention only**. Do not claim single-factor causality.

Required post-hoc protocol:
1. Full-val evaluate `best.pt`, `last.pt`, and all milestone ckpts (`10k/20k/30k/40k/50k`) under unified clip3 protocol.
2. Report `D20/D10/D4/NORMAL`, `transport_avg`, and chain MSE together.
3. Run offline dual-selector audit on same checkpoints:
   - old selector weights (`D20=0.15`)
   - v3 selector weights (`D20=0.50`)
4. Only claim robust gain if ranking is consistent across selectors and full-val metrics.

### 3.3 If relaunching (highest info per GPU-day)

Ablation order:
1. **W-only**:
   - keep hop0-weight shifts (`pair_loss_weights`, `step_weights`)
   - keep old selector (`D20 weight=0.15`)
   - `velocity_rebalance.enabled=false`
2. If positive at 10k, add only `velocity_rebalance.enabled=true`.
3. Test selector change last (change only `D20 weight 0.15 -> 0.50`).

## 4. Reviewer-facing wording (recommended)

"We treat v3 as a compound training intervention that jointly reweights early-hop supervision and modifies checkpoint selection; therefore we do not attribute gains to any single sub-change. To avoid narrative bias, conclusions are based on a unified full-validation protocol over `best`, `last`, and intermediate checkpoints, with checkpoint ranking audited under both original and v3 selection weights. We report hop-wise outcomes (`D20/D10/D4/NORMAL`) together with `transport_avg`, and interpret improvements as evidence for the bundled intervention only."

## 5. Current Run Status (executed)

Current launched run (tmux persistent):
- Session: `transport_v3_gpu2_0423`
- Config: `configs/pet_flow/pet_flow_first_hop_224_50k_transport_v3_gpu2_tmux1.yaml`
- Run dir: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_transport_v3_gpu2_tmux1`
- Log: `review/0423/logs_train/transport_v3_train_gpu2_tmux1_20260423_165534.log`

Status at doc time:
- `latents_train.pt` loaded (~356s)
- train raw D50/D20/D10/D4/NORMAL image tensors loaded
- now loading `latents_val.pt`

