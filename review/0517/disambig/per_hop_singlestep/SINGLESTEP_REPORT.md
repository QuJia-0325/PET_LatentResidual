# Per-hop single-step disambiguation report (2026-05-17)

## Scope

- This is §1 of `review/0516/CODEX_DISAMBIGUATION_RUNBOOK_20260516.md`.
- Protocol: full-val `n=7403`, `PSNR_clip3=src.utils.metrics.calc_psnr_clip3`, each hop evaluated as `GT z_k -> predicted z_{k+1} -> decoded x_{k+1}`.
- This intentionally removes open-loop cascade error. Hop0 still receives GT D50 image conditioning, matching current first-hop pixel-forcing semantics; hop1-hop3 do not receive image input.
- The NORMAL target image comes from `preprocessed_data_D50.pt[split]["x_0"]` through `PETFirstHopAligned4HopDataset`, so log text `NORMAL_from_D50_x0` means canonical normal image, not D50 noisy input.

## Primary result

| hop | target stage | V7_last single-step | V8_last single-step | Δ single-step V7-V8 | paired t | win % | Δ cascade V7-V8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D50->D20 | D20 | 35.4354 | 35.2094 | +0.2260 | 43.10 | 85.8% | +0.2260 |
| D20->D10 | D10 | 39.2081 | 39.1875 | +0.0206 | 5.29 | 71.0% | +0.2297 |
| D10->D4 | D4 | 41.1100 | 41.1104 | -0.0004 | -0.17 | 52.3% | +0.2773 |
| D4->NORMAL | NORMAL | 43.0138 | 43.0325 | -0.0187 | -3.41 | 52.2% | +0.3080 |

## Noise-control reference

| hop | target stage | V7_last single-step | V6_NOISE_last single-step | Δ V7-V6_NOISE | paired t | win % |
|---|---:|---:|---:|---:|---:|---:|
| D50->D20 | D20 | 35.4354 | 35.4217 | +0.0137 | 6.61 | 54.6% |
| D20->D10 | D10 | 39.2081 | 39.1912 | +0.0169 | 9.10 | 58.2% |
| D10->D4 | D4 | 41.1100 | 41.1852 | -0.0753 | -19.83 | 22.6% |
| D4->NORMAL | NORMAL | 43.0138 | 43.0135 | +0.0003 | 0.06 | 47.9% |

## Best/last equivalence

| exp | best step | last step | best_val(best file) | best_val(last file) | single-step means identical |
|---|---:|---:|---:|---:|---:|
| V7 | 160000 | 160000 | 0.000903636957482411 | 0.000903636957482411 | True |
| V8 | 160000 | 160000 | 0.000953388031244193 | 0.000953388031244193 | True |

## Interpretation

- The only large single-step V7-V8 gain is hop0: `+0.2260 dB`, exactly matching the cascade D20 gain `+0.2260 dB`.
- Later GT-input single-step deltas are near zero: hop1 `+0.0206 dB`, hop2 `-0.0004 dB`, hop3 `-0.0187 dB`. This does not support the claim that V7 improves every hop through a shared-backbone direct channel.
- In contrast, cascade deltas grow from D20 to NORMAL: `+0.2260 -> +0.2297 -> +0.2773 -> +0.3080 dB`. Because single-step hop1-hop3 do not show comparable direct gains, the NORMAL advantage is better explained as hop0 improvement propagating through open-loop chain coherence.
- The V7-V6_NOISE single-step deltas are not a pure seed-noise estimate because V6_NOISE also differs in config axes from V7; it is reported only as an additional reference, not a causal decomposition.

## Decision against runbook matrix

- Channel A (chain coherence / hop0-driven propagation) is the dominant interpretation for current checkpoints.
- Channel B (uniform shared-backbone per-hop improvement) is not supported by the single-step panel.
- Practical consequence: before starting expensive new training, prioritize hop0/D50->D20 interventions or true single-variable controls. Do not justify multi-hop image_aux solely from the V7-V8 NORMAL cascade delta.

## Artifacts

- Script: `review/0517/disambig/scripts/run_singlestep_v7_v8_v6noise.sh`
- Tool: `tools/eval_per_hop_singlestep_clip3.py`
- Log: `review/0517/disambig/logs/singlestep_v7_v8_v6noise_gpu1_20260517_003952.log`
- JSON/CSV outputs: `review/0517/disambig/per_hop_singlestep/{V7_last,V8_last,V7_best,V8_best,V6_NOISE_last}/`
