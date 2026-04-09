# PLAN_V1_FirstHop_Rebalance_Conservative.md

## Goal
针对 224 级联异常（D50→NORMAL 下降、D20→NORMAL 上升），最小改动验证是否是**训练重心偏后段**导致 hop0 欠优化。

## Scope
- 不改代码，仅改配置。
- 指标统一 `calc_psnr_clip3`（见 `eval_first_hop_224_clip3.py`）。

## Hypothesis
当前 chainstable 的 `best_metric_terms`、`pair_loss_weights`、`rollout.step_weights` 都偏 tail，导致 D50→D20 梯度不足；适度前移权重可提升 D50→D20，进而拉升 D50→NORMAL。

## Baseline
`configs/pet_flow/pet_flow_first_hop_224_10k_formal_v3_chainstable.yaml`

## Variant (V1)
建议新建配置：`configs/pet_flow/pet_flow_first_hop_224_10k_v1_rebalance_conservative.yaml`

### Required Config Deltas
```yaml
training:
  best_metric: val_chain_d20_mse

transport:
  pair_sample_probs: [0.40, 0.25, 0.20, 0.15]
  pair_loss_weights: [1.80, 1.20, 1.00, 1.00]

training:
  rollout:
    step_weights: [1.50, 1.20, 1.00, 1.00]
```

## Runbook
```bash
python train_first_hop.py --config configs/pet_flow/pet_flow_first_hop_224_10k_formal_v3_chainstable.yaml
python train_first_hop.py --config configs/pet_flow/pet_flow_first_hop_224_10k_v1_rebalance_conservative.yaml

python eval_first_hop_224_clip3.py --config configs/pet_flow/pet_flow_first_hop_224_10k_formal_v3_chainstable.yaml --checkpoint <baseline_best.pt> --split val --max-slices 0 --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/v1_baseline
python eval_first_hop_224_clip3.py --config configs/pet_flow/pet_flow_first_hop_224_10k_v1_rebalance_conservative.yaml --checkpoint <v1_best.pt> --split val --max-slices 0 --out-dir /data_2/qujiaxiang/outputs/PET_LatentResidual/clip3_eval/v1_variant
```

## Primary Metrics
- `PSNR_clip3(D20)`
- `PSNR_clip3(NORMAL)`
- `Δ(D50→NORMAL)` vs baseline

## Gate
- PASS: `ΔD20 >= +0.20 dB` and `NORMAL drop <= 0.10 dB`
- FAIL: `ΔD20 < +0.10 dB` or `NORMAL drop > 0.20 dB`
- GRAY: between PASS and FAIL

## Diagnostic Add-on (mandatory)
对 baseline 与 V1 同时跑：
- no-transport
- teacher-forced chain
- pure-pred chain

用于确认收益是否来自 hop0，而非偶然 tail 波动。

## Risk
如果 D20 升但 NORMAL 降明显，说明只是“局部优化”，需要升级到 V2（rollout 对齐强化）。
