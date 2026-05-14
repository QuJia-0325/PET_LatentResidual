# Training Log Snapshot 20260514_202425

- copied_at: 2026-05-14 20:28:13 Asia/Shanghai
- branch: foc_lite_hop0
- base_commit_when_copied: 94c4076
- source_main_logs: review/0505/local/runs/{V7,V8,V6_NOISE}/train.log
- source_preflight_logs: review/0506/local/logs/V{6,7}_5K_pass{1,2}_gpu{2,3}.log
- note: files are copied snapshots; original live logs remain in place and continue to be written by training jobs.

## Runtime Snapshot
```text
0, NVIDIA RTX A6000, 16831, 49140, 100, 82
1, NVIDIA RTX A6000, 35516, 49140, 100, 78
2, NVIDIA RTX A6000, 29964, 49140, 100, 74
3, NVIDIA RTX A6000, 17983, 49140, 22, 61
```

## Compute Processes
```text
3012594, python, 16808
4003657, /home/qujiaxiang/.conda/envs/rae/bin/python, 18426
3013218, python, 17062
3970603, /home/qujiaxiang/.conda/envs/rae/bin/python, 18608
4041922, python, 11328
3970601, /home/qujiaxiang/.conda/envs/rae/bin/python, 17960
```

## Plan-F Training Processes
```text
UID          PID    PPID  C STIME TTY          TIME CMD
qujiaxi+ 3970601 3970600 99 May06 pts/11   151-01:06:57 /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0505/local/runs/V8/config.resolved.yaml
qujiaxi+ 3970603 3970599 99 May06 pts/11   148-12:20:49 /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0505/local/runs/V7/config.resolved.yaml
qujiaxi+ 4003657 4003644 99 May06 pts/14   127-08:07:44 /home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0505/local/runs/V6_NOISE/config.resolved.yaml
```

## Tmux Sessions
```text
planf_v6noise_0506: 1 windows (created Wed May  6 14:00:28 2026)
planf_v7v8_0506: 1 windows (created Wed May  6 13:46:18 2026)
tb_r203_mathfix_gpu3: 1 windows (created Thu Apr 16 01:13:38 2026)
wb_r203_mathfix_gpu3: 1 windows (created Thu Apr 16 01:13:38 2026)
work: 1 windows (created Tue Apr  7 02:05:51 2026)
```

## Latest Metrics

| Experiment | log MB | latest train step | latest val step | val_select_score | chain_normal_mse | chain_d20_mse | chain_tail_mse |
|---|---:|---:|---:|---:|---:|---:|---:|
| V7 | 2.69 | 128700 | 128400 | 0.001107 | 0.000297 | 0.000416 | 0.000328 |
| V8 | 3.15 | 154750 | 154400 | 0.000854 | 0.000239 | 0.000291 | 0.000255 |
| V6_NOISE | 2.94 | 141550 | 141200 | 0.000716 | 0.000203 | 0.000241 | 0.000214 |

## Latest Raw Lines

### V7
```text
[val] step=128400 val_pair_total=0.000002 val_pair_velocity=0.000001 val_pair_endpoint=0.000001 val_rollout_total=0.000874 val_hop0_img_total=0.008512 val_hop0_img_l1=0.003583 val_hop0_img_ssim=0.017094 val_hop0_img_seam=0.006558 val_foc_total=0.000000 val_foc_gap=0.000000 val_chain_d20_mse=0.000416 val_chain_d10_mse=0.000364 val_chain_d4_mse=0.000321 val_chain_normal_mse=0.000297 val_chain_d20_raw_mse=nan val_chain_d10_raw_mse=nan val_chain_d4_raw_mse=nan val_chain_normal_raw_mse=nan val_rollout_step_0=0.000897 val_rollout_step_0_raw=0.000897 val_rollout_step_1=0.000856 val_rollout_step_1_raw=0.000856 val_rollout_step_2=0.000850 val_rollout_step_2_raw=0.000850 val_rollout_step_3=0.000917 val_rollout_step_3_raw=0.000917 val_chain_tail_mse=0.000328 val_chain_samples=512.000000 val_chain_unique_slices=512.000000 val_main_batches_evaluated=64.000000 val_hop0_batches_evaluated=64.000000 val_main_window_start_batch=1920.000000 val_hop0_window_start_batch=1920.000000 val_select_score=0.001107
[train] step=128700 loss=0.002619 pair=0.000131 vel=0.000066 vel_raw=0.000027 end=0.000066 end_raw=0.000027 vel_w=1.000 end_w=1.000 vel_reb=1.000 roll=0.000171 roll0=0.000418 img=0.002794 img_l1=0.001190 img_ssim=0.005518 img_seam=0.002243 ssim_raw_mean=0.994486 ssim_raw_min=0.467336 ssim_raw_max=1.000165 ssim_over1=0.114059 ssim_below0=0.000000 ssim_clamped_mean=0.994482 lambda_roll=3.1480 lambda_roll_base=3.1480 lambda_roll_scale=1.0000 st=1 lambda_img=0.0400 alpha=0.7870 pair_w=0.001969 roll_w=0.000538 img_w=0.000112 pair_frac=0.752 roll_frac=0.205 img_frac=0.043 hop0_coverage=0.250 hop0_main_ratio=0.500 gate_pix=0.0362 gate_pix_raw=-3.3283 lambda_hop_0=0.0579 lambda_hop_0_raw=-2.8375 gate_eff=0.035225 lambda_eff_h0=0.056920 pix_delta_abs=0.052025 pix_delta_abs_hop0=0.104050 v_hop_abs=0.000499 v_hop_abs_hop0=0.000719 sf_alpha=0.000 sf_gap=0.000000 grad_total=6.1610e-02 grad_backbone=6.1231e-02 grad_firsthop=6.8239e-03 grad_clip_pre=6.1610e-02
```

### V8
```text
[val] step=154400 val_pair_total=0.000001 val_pair_velocity=0.000001 val_pair_endpoint=0.000001 val_rollout_total=0.000590 val_hop0_img_total=0.006202 val_hop0_img_l1=0.002499 val_hop0_img_ssim=0.012745 val_hop0_img_seam=0.005168 val_foc_total=0.000000 val_foc_gap=0.000000 val_chain_d20_mse=0.000291 val_chain_d10_mse=0.000273 val_chain_d4_mse=0.000252 val_chain_normal_mse=0.000239 val_chain_d20_raw_mse=nan val_chain_d10_raw_mse=nan val_chain_d4_raw_mse=nan val_chain_normal_raw_mse=nan val_rollout_step_0=0.000559 val_rollout_step_0_raw=0.000559 val_rollout_step_1=0.000555 val_rollout_step_1_raw=0.000555 val_rollout_step_2=0.000559 val_rollout_step_2_raw=0.000559 val_rollout_step_3=0.000723 val_rollout_step_3_raw=0.000723 val_chain_tail_mse=0.000255 val_chain_samples=512.000000 val_chain_unique_slices=512.000000 val_main_batches_evaluated=64.000000 val_hop0_batches_evaluated=64.000000 val_main_window_start_batch=2368.000000 val_hop0_window_start_batch=2368.000000 val_select_score=0.000854
[train] step=154750 loss=0.001744 pair=0.000032 vel=0.000016 vel_raw=0.000008 end=0.000016 end_raw=0.000008 vel_w=1.000 end_w=1.000 vel_reb=1.000 roll=0.000317 roll0=0.000381 img=0.000000 img_l1=0.000000 img_ssim=0.000000 img_seam=0.000000 ssim_raw_mean=nan ssim_raw_min=nan ssim_raw_max=nan ssim_over1=nan ssim_below0=nan ssim_clamped_mean=nan lambda_roll=4.0000 lambda_roll_base=4.0000 lambda_roll_scale=1.0000 st=1 lambda_img=0.0000 alpha=1.0000 pair_w=0.000477 roll_w=0.001267 img_w=0.000000 pair_frac=0.273 roll_frac=0.727 img_frac=0.000 hop0_coverage=0.250 hop0_main_ratio=0.125 gate_pix=0.0352 gate_pix_raw=-3.3596 lambda_hop_0=0.0698 lambda_hop_0_raw=-2.6415 gate_eff=0.034160 lambda_eff_h0=0.068831 pix_delta_abs=0.015675 pix_delta_abs_hop0=0.125400 v_hop_abs=0.000316 v_hop_abs_hop0=0.000911 sf_alpha=0.000 sf_gap=0.000000 grad_total=6.1310e-02 grad_backbone=6.1081e-02 grad_firsthop=5.2866e-03 grad_clip_pre=6.1310e-02
```

### V6_NOISE
```text
[val] step=141200 val_pair_total=0.000381 val_pair_velocity=0.000191 val_pair_endpoint=0.000191 val_rollout_total=0.000801 val_hop0_img_total=0.007337 val_hop0_img_l1=0.002918 val_hop0_img_ssim=0.015737 val_hop0_img_seam=0.004848 val_foc_total=0.000000 val_foc_gap=0.000000 val_chain_d20_mse=0.000241 val_chain_d10_mse=0.000229 val_chain_d4_mse=0.000209 val_chain_normal_mse=0.000203 val_chain_d20_raw_mse=nan val_chain_d10_raw_mse=nan val_chain_d4_raw_mse=nan val_chain_normal_raw_mse=nan val_rollout_step_0=0.000686 val_rollout_step_0_raw=0.000686 val_rollout_step_1=0.000703 val_rollout_step_1_raw=0.000703 val_rollout_step_2=0.000776 val_rollout_step_2_raw=0.000776 val_rollout_step_3=0.001091 val_rollout_step_3_raw=0.001091 val_chain_tail_mse=0.000214 val_chain_samples=512.000000 val_chain_unique_slices=512.000000 val_main_batches_evaluated=64.000000 val_hop0_batches_evaluated=64.000000 val_main_window_start_batch=256.000000 val_hop0_window_start_batch=256.000000 val_select_score=0.000716
[train] step=141550 loss=0.002416 pair=0.000072 vel=0.000036 vel_raw=0.000017 end=0.000036 end_raw=0.000017 vel_w=1.000 end_w=1.000 vel_reb=1.000 roll=0.000243 roll0=0.000403 img=0.011089 img_l1=0.004636 img_ssim=0.022879 img_seam=0.007337 ssim_raw_mean=0.977033 ssim_raw_min=-0.835449 ssim_raw_max=1.000149 ssim_over1=0.107243 ssim_below0=0.000239 ssim_clamped_mean=0.977121 lambda_roll=3.6620 lambda_roll_base=3.6620 lambda_roll_scale=1.0000 st=1 lambda_img=0.0400 alpha=0.9155 pair_w=0.001082 roll_w=0.000891 img_w=0.000444 pair_frac=0.448 roll_frac=0.369 img_frac=0.184 hop0_coverage=0.250 hop0_main_ratio=0.250 gate_pix=0.0350 gate_pix_raw=-3.3632 lambda_hop_0=0.0571 lambda_hop_0_raw=-2.8519 gate_eff=0.034037 lambda_eff_h0=0.056132 pix_delta_abs=0.026493 pix_delta_abs_hop0=0.105972 v_hop_abs=0.000392 v_hop_abs_hop0=0.000751 sf_alpha=0.000 sf_gap=0.000000 grad_total=5.5095e-02 grad_backbone=5.5048e-02 grad_firsthop=2.2690e-03 grad_clip_pre=5.5095e-02
```

## Files
- `review/0511/log_snapshots_20260514_202425/main_training/V6_NOISE_train_snapshot_20260514_202425.log` (3080387 bytes)
- `review/0511/log_snapshots_20260514_202425/main_training/V7_train_snapshot_20260514_202425.log` (2824987 bytes)
- `review/0511/log_snapshots_20260514_202425/main_training/V8_train_snapshot_20260514_202425.log` (3302417 bytes)
- `review/0511/log_snapshots_20260514_202425/phase1_preflight/V6_5K_pass1_gpu2.log` (113569 bytes)
- `review/0511/log_snapshots_20260514_202425/phase1_preflight/V6_5K_pass2_gpu2.log` (115400 bytes)
- `review/0511/log_snapshots_20260514_202425/phase1_preflight/V7_5K_pass1_gpu3.log` (114966 bytes)
- `review/0511/log_snapshots_20260514_202425/phase1_preflight/V7_5K_pass2_gpu3.log` (117374 bytes)
- `review/0511/log_snapshots_20260514_202425/status/training_log_snapshot_summary_20260514_202425.md` (7409 bytes)
