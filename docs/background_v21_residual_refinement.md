# Background: V2.1 Residual Refinement

Status:

- background positioning document
- describes the `v2.1 residual refinement` line
- not the current first-hop implementation baseline

## Positioning

This workspace does **not** implement joint pixel/latent transport.

It implements:

`hop-aware latent rollout + decoder-side structure residual refinement`

The latent state remains the only rollout state:

`z_D50 -> z_D20 -> z_D10 -> z_D4 -> z_NORMAL`

The image-side residual head only refines the decoded image of the current hop:

`x_{k+1}^pred = Dec(z_{k+1}^pred) + r_{k+1}`

and does not alter the next latent hop.

## Why This Narrower Formulation

The original wider proposal had three avoidable risks:

1. It implicitly created a second rollout state without defining consistent dynamics.
2. It made image tokens too likely to become the real transport shortcut.
3. It blurred the distinction between transport improvement and decoder-side repair.

v2.1 removes those ambiguities.

## State Definition

### Transport State

- Only `z_k`
- Produced and propagated by the existing hop-aware latent backbone

### Refinement Inputs

For each hop:

- `x_k^draft = Crop192(Dec(z_k))`
- `x_{k+1}^draft = Crop192(Dec(z_{k+1}^pred))`
- hop-aware conditioning: `t_src`, `t_dst`, `log_dt`, `hop_id`

### Refinement Output

- `r_{k+1}`: single-channel local residual
- constrained by `tanh` and `max_residual`

Final refined image:

- `x_{k+1}^pred = clamp(x_{k+1}^draft + r_{k+1}, -1, 1)`

## Training Rules

### Backbone

The first experiment freezes the latent backbone.

This preserves latent rollout dynamics and makes the experiment easier to interpret:

- if quality improves, the gain comes from local structure repair
- not from changing the latent transport model

### Decoder

The RAE decoder is frozen.

### Residual Head

The residual head is the only trainable module in the default config.

It must stay small and local:

- no global attention
- no large UNet
- no access to full GT image state as recurrent input

## Losses

### Draft vs Refined

Track both:

- draft image metrics from `Dec(z_pred)`
- refined image metrics from `Dec(z_pred) + r`

### Main Loss

Refined image supervision:

- weighted L1
- SSIM
- seam consistency
- residual amplitude penalty

### Border Treatment

Supervision happens on cropped `192x192` images, so there are no invalid pixels in the literal sense.

The real issue is boundary contamination from pad-affected outer latent tokens.

Therefore the image loss should use:

- border-aware weighting
- seam regularization

instead of a binary valid-mask-only formulation.

### Optional Rollout Endpoint

The residual head may also be supervised on the final rollout endpoint:

- decode the latent-only rollout chain
- refine only the final decoded endpoint
- compute final endpoint image loss

This still does not change latent rollout dynamics.

## Straight-Through Rollout

The latent rollout helper preserves the original straight-through mixing behavior from the RAE experiment line:

- forward path can mix toward GT during training
- gradients remain attached to the predicted latent branch

This is only relevant when the latent backbone is trainable.

When the backbone is frozen, rollout still uses the same helper interface but the training effect is restricted to the residual head.

## First Success Criterion

The first version is successful if it improves:

- seam continuity
- local structure coherence
- endpoint PSNR/SSIM

without changing:

- latent state definition
- hop-aware rollout topology
- next-hop latent dynamics
