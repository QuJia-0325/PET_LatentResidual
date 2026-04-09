# PLAN_V2_Hop0_Stability_Rollout_Alignment.md

## Goal
在 V1 基础上，强化训练-推理一致性，验证 D50→NORMAL 的劣化是否源于**pure-pred 级联分布偏移**。

## Scope
- 不改代码，仅改配置。
- 保持 hop0 pixel forcing 机制不变，仅调整 schedule 与损失配比。

## Hypothesis
D20→NORMAL 变好而 D50→NORMAL 变差，说明“后段在 teacher-forced 或较净输入下受益”，但首跳噪声输入进入 pure-pred 后误差放大。适度增加 rollout 压力并前置 pred-mixing，可减少级联漂移。

## Baseline
- V1 通过则以 V1 为 baseline。
- V1 未通过则回到 chainstable 作为 baseline。

## Variant (V2)
建议新建配置：`configs/pet_flow/pet_flow_first_hop_224_10k_v2_rollout_alignment.yaml`

### Required Config Deltas
```yaml
training:
  rollout:
    warmup_ratio: 0.05
    ramp_ratio: 0.20
    alpha_start: 0.0
    alpha_end: 1.00
    lambda_start: 0.03
    lambda_end: 0.35
    step_weights: [1.60, 1.25, 1.05, 1.00]

transport:
  pair_sample_probs: [0.45, 0.25, 0.15, 0.15]
  pair_loss_weights: [2.00, 1.30, 1.00, 1.00]

training:
  best_metric: val_chain_d20_mse
```

## Runbook
```bash
python train_first_hop.py --config configs/pet_flow/pet_flow_first_hop_224_10k_v2_rollout_alignment.yaml
python eval_first_hop_224_clip3.py --config configs/pet_flow/pet_flow_first_hop_224_10k_v2_rollout_alignment.yaml --checkpoint <v2_best.pt> --split val --max-slices 0 --out-dir outputs/clip3_eval/v2_variant
```

## Primary Metrics
- `PSNR_clip3(D20)`
- `PSNR_clip3(D50→NORMAL cascade)`
- tail_avg(`D20,D10,D4,NORMAL`)

## Gate
- PASS: `D50→NORMAL >= baseline +0.20 dB` and `D20 not below baseline by >0.10 dB`
- FAIL: `D20 drop >0.15 dB` or `NORMAL no gain`

## Mandatory Causal Decomposition
对 V2 执行 A/B/C/D 分层评估：
- A full-pred
- B gt@D20
- C gt@D10
- D gt@D4

观察 `ΔB`（hop1边际）是否缩小，确认首跳误差传播是否被抑制。

## Risk
过强 rollout 可能牺牲首跳；因此必须与 V1 并行对照，不可单独解读。
