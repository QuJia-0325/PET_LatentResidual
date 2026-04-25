# V3 Three Experiments Summary (2026-04-26)

## Experiment 1: Path A Multi-Checkpoint Diagnostic
- Log: `review/0425/logs_diag/pathA_C_multi_ckpt_gpu0.log`
- Status: completed
- Verdict: `[VERDICT] EXPOSURE_BIAS_DOMINATES`
- Done line: `[DONE ] 2026-04-25 18:34:26 best.pt`
- Output artifacts:
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C_step_010000/tf_rollout_gap_val.json`
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C_step_030000/tf_rollout_gap_val.json`
  - `/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/pathA_C_best/tf_rollout_gap_val.json`

## Experiment 2: Self-Forcing 50K (from C)
- Config: `configs/pet_flow/pet_flow_first_hop_224_50k_selfforcing_from_C_gpu0_r1.yaml`
- Log: `review/0425/logs_train/selfforcing_v3_gpu0_r1.log`
- Resume: `[resume] loaded step=46400, best_val=0.000521`
- Last val line: `[val] step=50000 ... val_select_score=0.000672`
- Last val_select_score: `0.000672`
- Done line: `Training done. Outputs at: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_selfforcing_from_C_gpu0_r1`
- Output dir: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_selfforcing_from_C_gpu0_r1`

## Experiment 3: Rollout-Up 50K (from C)
- Config: `configs/pet_flow/pet_flow_first_hop_224_50k_rollout_up_from_C_gpu0_r1.yaml`
- Log: `review/0425/logs_train/rolloutup_v3_gpu0_r1.log`
- Resume: `[resume] loaded step=46400, best_val=0.000521`
- Last val line: `[val] step=50000 ... val_select_score=0.000670`
- Last val_select_score: `0.000670`
- Done line: `Training done. Outputs at: /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_rollout_up_from_C_gpu0_r1`
- Output dir: `/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_rollout_up_from_C_gpu0_r1`

## Notes
- Experiment 2/3 are resume-style runs from step 46400 and stop at max_steps=50000, so effective additional training is 3600 steps.
- This commit intentionally excludes active 200k training log (`review/0424/logs/transport_v3_200k_train_gpu1.log`).
