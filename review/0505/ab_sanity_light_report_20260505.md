# Superseded Note: A/B Sanity-Light Report

This note is superseded by the official `run_ablation.sh sanity` output copied under:

- `review/0505/logs_sanity/run_ablation_sanity_gpu1_20260503.launch.log`
- `review/0505/logs_sanity/run_ablation_A_sanity_train.log`
- `review/0505/logs_sanity/run_ablation_B_sanity_train.log`
- `review/0505/logs_sanity/run_ablation_A_sanity_config.resolved.yaml`
- `review/0505/logs_sanity/run_ablation_B_sanity_config.resolved.yaml`

Correction: the formal `run_ablation.sh sanity` gate result is **FAIL**, not PASS.

The previous version of this file summarized the memory-light `run_sanity_light.sh` A/B logs and was not the formal `run_ablation.sh` gate output. It should not be used as evidence that the official sanity gate passed.

Official comparator excerpt from `run_ablation_sanity_gpu1_20260503.launch.log`:

```text
===== Sanity check (multi-tier, pure-python — no jq required) =====
metric                                        A              B    rel_err  tier_threshold
------------------------------------------------------------------------------------------------
val_pair_total                     1.537353e-07   1.693620e-07    10.16%  Tier 1 (bit-equal) -> FAIL
val_rollout_total                  2.051546e-03   2.089454e-03     1.85%  Tier 2 (<1%) -> FAIL
val_rollout_step_0_raw             1.763927e-03   1.768560e-03     0.26%  Tier 3 (<1%) -> PASS
val_rollout_step_1_raw             1.898111e-03   1.902824e-03     0.25%  Tier 3 (<1%) -> PASS
val_rollout_step_2_raw             2.122597e-03   2.162539e-03     1.88%  Tier 3 (<1%) -> FAIL
val_rollout_step_3_raw             2.395652e-03   2.513605e-03     4.92%  Tier 3 (<1%) -> FAIL
val_chain_d20_mse                  7.356653e-04   7.351346e-04     0.07%  Tier 4 (<5%) -> PASS
val_chain_d10_mse                  6.598809e-04   6.551905e-04     0.71%  Tier 4 (<5%) -> PASS
val_chain_d4_mse                   5.903761e-04   5.791343e-04     1.90%  Tier 4 (<5%) -> PASS
val_chain_normal_mse               5.872322e-04   6.619837e-04    12.73%  Tier 4 (<5%) -> FAIL

Result: FAIL (one or more tiers exceed threshold or missing)

[run_ablation.sh] Sanity FAILED — sentinel cleared. main/C/D will refuse to launch.
```

Seed status from the formal resolved configs:

- `A_sanity`: `seed: 42`, `training.deterministic: true`, `data.num_workers: 0`.
- `B`: `seed: 42`, `training.deterministic: true`, `data.num_workers: 0`.

Therefore, the official formal answer is: same configured RNG seed was used, but the formal sanity gate failed.
