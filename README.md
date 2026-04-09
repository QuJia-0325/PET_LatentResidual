# PET Latent Residual

Independent workspace for PET-side follow-up experiments around the original RAE project.

This repo now contains two lines:

- historical `v2.1 residual refinement`
- current `first-hop enhancement` line
- `224 first-hop latent transport` implementation

This folder is intentionally separate from `/home/qujiaxiang/project/RAE` so the new idea does not get mixed into the original RAE codebase.

Current canonical implementation spec:

- `docs/main.md`

## Scope

- Keep the latent rollout state as the only transport state.
- Reuse the existing hop-aware 4-hop latent model and frozen RAE decoder from the original RAE project.
- Add a small image-space residual head after decoding.
- Do not feed the residual output back into the next latent hop.

## Layout

```text
PET_LatentResidual/
  README.md
  docs/
    main.md
    background_first_hop_design.md
    background_v21_residual_refinement.md
    archive/
      first_hop_implementation_spec_old.md
      v21_224_decoder_adaptor_spec.md
  configs/
    pet_flow/
      pet_flow_latent_residual_v21.yaml
      pet_flow_first_hop_224.yaml
  pet_lr/
    __init__.py
    bootstrap.py
    data.py
    data_first_hop.py
    losses.py
    losses_first_hop.py
    model.py
    model_first_hop.py
    rollout.py
    rollout_first_hop.py
  train_v21.py
  train_first_hop.py
```

## What Lives Here

- `pet_lr/data.py`
  Loads aligned 4-hop latent pairs and raw PET slices.
- `pet_lr/data_first_hop.py`
  First-hop aligned dataset with dual-loader support (`main` + `hop0 auxiliary`) for hop0 image supervision.
- `pet_lr/model.py`
  Wraps the frozen hop-aware latent backbone, frozen RAE decoder, and local residual head.
- `pet_lr/model_first_hop.py`
  Adds hop0 pixel forcing and hop-specific latent velocity residual head on top of the hop-aware backbone.
- `pet_lr/rollout.py`
  Runs latent-only rollout and preserves the original straight-through mixing idea.
- `pet_lr/rollout_first_hop.py`
  Runs first-hop rollout where only step 0 receives `x_D50` pixel condition.
- `pet_lr/losses.py`
  Implements `L1 + SSIM + seam + border-aware weighting + residual penalty`.
- `pet_lr/losses_first_hop.py`
  Implements hop0 image auxiliary loss on `Crop(Dec(z_D20_pred))`.
- `train_v21.py`
  Minimal training entry point for the new experiment line.
- `train_first_hop.py`
  Training entry for the 224 first-hop latent transport line.
- `docs/main.md`
  Current canonical implementation spec for the first-hop enhancement line.
- `docs/background_v21_residual_refinement.md`
  Background note for the earlier `v2.1 residual refinement` line.
- `docs/background_first_hop_design.md`
  Draft reasoning document that led to the first-hop implementation spec.
- `docs/archive/v21_224_decoder_adaptor_spec.md`
  Archived design spec for the earlier `224 + decoder adaptor` direction.
- `docs/archive/first_hop_implementation_spec_old.md`
  Archived duplicate kept only for traceability.

## What Still Comes From The Original RAE Project

This workspace reuses stable components from:

- `/home/qujiaxiang/project/RAE/code/RAE/src/pet_flow/models/pet_flow_dit_hop.py`
- `/home/qujiaxiang/project/RAE/code/RAE/src/pet_flow/inference_pet_flow.py`

That reuse is intentional:

- the original RAE project remains unchanged
- all new experiment-specific code stays here

## First Recommended Experiment

The default config freezes both:

- the hop-aware latent backbone
- the RAE decoder

So the first run answers a narrow question:

Can a small, local, decoder-side residual head reduce structure breaks and seam artifacts without changing latent rollout dynamics?

## Example

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
python train_v21.py --config configs/pet_flow/pet_flow_latent_residual_v21.yaml
```

## 224 First-Hop Example

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
python train_first_hop.py --config configs/pet_flow/pet_flow_first_hop_224.yaml
```
