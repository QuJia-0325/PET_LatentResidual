# V5 Rollout-Heavy Design Review

**Date**: 2026-04-27  
**Reviewer**: Codex  
**Remote HEAD reviewed**: `bc41222ac7031564ef0eeac379d5df14f377980a`  
**Branch**: `foc_lite_hop0`  

## Executive Conclusion

V5's high-level direction is reasonable: after V4 showed that self-forcing pair training can push the model away from the GT-input optimum, shifting to a rollout-heavy objective is a defensible next experiment. The design keeps the model size and batch size unchanged, increases open-loop chain supervision, and makes training/evaluation distribution more consistent by using `alpha=1.0`.

However, the current implementation should not be described as "only changing three axes relative to 200K v3." Two implementation details change the experimental interpretation:

1. `velocity_rebalance` is disabled in V5, while it is enabled in the 200K v3 baseline.
2. The resume-time LR schedule is changed by setting `max_steps = resume_step + 50000`, which turns the run into a lower-LR fine-tune relative to continuing the original 200K schedule.

These issues do not make V5 invalid, but they weaken attribution. Before running or before presenting results, the team should either fix them or explicitly document V5 as a "rollout-heavy + low-LR fine-tune + pair-rebalance-off" experiment.

## Files Reviewed

- `review/plan/transport_breakthrough_research_v5.md`
- `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml`
- `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml`
- `scripts/launch_v5_rollout_heavy.sh`
- `train_first_hop.py`
- `pet_lr/rollout_first_hop.py`

## What Is Reasonable

### 1. Moving away from SF-pair is justified

The V4 analysis showed a concrete failure mode: as `sf_alpha` increased, `val_pair_total` rose even under GT validation input. That means the model parameters themselves were being pushed away from the GT-input optimum, not merely failing under self-forced inputs. Given that evidence, prioritizing rollout-heavy training over more aggressive SF-pair training is reasonable.

### 2. V5 makes train/eval rollout distribution more consistent

The V5 config sets:

```yaml
rollout:
  alpha_start: 1.00
  alpha_end: 1.00
  eval_alpha: 1.00
```

This removes the earlier open-loop mismatch where training spent a long time with GT/mixed inputs while evaluation used fully predicted chain inputs. Since the target deployment path is open-loop chain prediction, this is a coherent design change.

### 3. The step-weight math is implemented as documented

The V5 plan computes effective rollout step weights after normalization by the sum of step weights. The code does the same:

```python
w = torch.tensor(step_weights, dtype=step_losses[0].dtype, device=step_losses[0].device)
stacked = torch.stack(step_losses, dim=0)
total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
```

Therefore, the documented relative hop scaling is broadly aligned with runtime behavior.

### 4. The resource assumptions are consistent with the environment

V5 does not increase batch size and does not introduce a larger backbone. That is the right constraint given the recent GPU memory pressure.

## Major Issues

### Issue 1: V5 is not actually a "3-axis only" experiment

The V5 plan states that only three axes are changed relative to 200K v3:

- `rollout.lambda`: `0.25 -> 1.5`
- `image_aux.lambda`: `0.12 -> 0.08`
- `rollout.step_weights`: `[1.30, 1.20, 1.10, 1.00] -> [0.8, 1.0, 1.5, 2.5]`

It also states that the remaining settings match the 200K v3 baseline.

The config does not satisfy that claim. In 200K v3:

```yaml
loss:
  pair:
    velocity_rebalance:
      enabled: true
```

In V5:

```yaml
loss:
  pair:
    velocity_rebalance:
      enabled: false
```

This is not cosmetic. `compute_pair_losses()` uses `velocity_rebalance` to scale the velocity term based on the endpoint/velocity loss ratio:

```python
ratio = (loss_endpoint.detach() + eps) / (loss_velocity.detach() + eps)
scale_t = torch.sqrt(ratio)
scale_t = torch.clamp(scale_t, min=clip_min, max=clip_max)
velocity_weight_eff = v_weight * rebalance_scale
total = velocity_weight_eff * loss_velocity + endpoint_weight * loss_endpoint
```

Changing this flag changes the effective pair loss composition, which affects the exact gradient balance that V5 is trying to study.

**Recommendation**: Set `loss.pair.velocity_rebalance.enabled: true` in V5 if the intended claim is "only the rollout/image/step-weight axes changed."

### Issue 2: The LR schedule is changed by the launch strategy

The launch script computes:

```bash
MAX_STEPS=$((RESUME_STEP + NEW_STEPS))
sed -i "s/^  max_steps: .*/  max_steps: ${MAX_STEPS}/" "${CONFIG}"
```

The training code then uses this `max_steps` to compute cosine LR:

```python
progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine
```

For the current resume point (`step=86800`), this changes the effective LR schedule:

| Setting | `max_steps` | LR at step 86800 | LR at step 136800 |
|---|---:|---:|---:|
| Original 200K v3 schedule | 200000 | ~`6.04e-5` | ~`2.57e-5` |
| Current V5 schedule | 136800 | ~`3.25e-5` | `2.0e-6` |

This may be beneficial for stability, but it is an additional intervention. It means V5 is not simply "200K v3 continued for 50K with stronger rollout." It is closer to "V3 best fine-tuned for 50K with stronger rollout under a compressed cosine schedule."

**Recommendation**: Either document this explicitly as a low-LR fine-tune, or add a scheduler total-step override so V5 can preserve the original 200K LR curve while stopping after 50K new steps.

## Secondary Risks

### 1. Rolling-val thresholds may be noisy

The V5 go/no-go rules use `val_chain_normal_mse` improvements at +10K and +25K. Because `val_window_mode=rolling`, the validation window changes over time. This is fine for early health monitoring, but it is not strong enough for final claims.

**Recommendation**: Use rolling-val only for early stop/continue decisions. For claims, require fixed-window or full-val evaluation against the same checkpoint baseline.

### 2. The launch script mutates a tracked config

`launch_v5_rollout_heavy.sh` edits `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` in place. This can leave the repository dirty and can make later audits ambiguous.

**Recommendation**: Copy the config to an output/log directory, edit the copy, and launch from that runtime copy.

### 3. Python environment is not pinned

The launch script uses plain `python`:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u train_first_hop.py ...
```

In this project, previous stable runs used:

```bash
/home/qujiaxiang/.conda/envs/rae/bin/python
```

**Recommendation**: Use the explicit RAE conda environment path in the script to avoid environment drift.

### 4. "Protect hop0" should be phrased carefully

V5 lowers hop0's relative rollout step weight, but because `lambda_roll` increases 6x, the absolute effective hop0 rollout pressure still increases:

- v3 hop0 effective rollout weight: `0.25 * 1.30 / 4.60 = 0.071`
- V5 hop0 effective rollout weight: `1.5 * 0.80 / 5.80 = 0.207`

So V5 does not reduce hop0 rollout supervision in absolute terms. It reduces hop0's relative priority while still increasing its absolute rollout pressure by about `2.9x`.

## Suggested Minimal Fix Before Running

If the team wants V5-main to be interpretable as the planned rollout-heavy experiment, make these changes:

1. Set `loss.pair.velocity_rebalance.enabled: true`.
2. Decide LR policy explicitly:
   - Option A: keep current compressed schedule and rename the experiment as a low-LR fine-tune.
   - Option B: preserve the original 200K schedule by adding a scheduler-total-steps override.
3. Launch from a copied runtime config rather than editing the tracked config in place.
4. Use `/home/qujiaxiang/.conda/envs/rae/bin/python` in the launcher.

## Go/No-Go Assessment

V5 is a reasonable experiment direction, but the current config has attribution issues. If run exactly as-is, it can still answer whether this combined recipe improves results, but it cannot cleanly answer whether the improvement comes from rollout-heavy weighting alone.

My recommendation is:

- **Do not block the direction.**
- **Fix `velocity_rebalance` before launch.**
- **Explicitly document the LR schedule choice before interpreting results.**

