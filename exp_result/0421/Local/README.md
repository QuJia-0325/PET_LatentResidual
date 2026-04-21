# 0421 Local — N1/N2 实验设计与代码

## 背景

NIGHTMARE 审查(4.35/10)指出两个核心缺陷：
1. **hop0 pixel forcing 无独立消融** — Scheme C 的 +0.08 dB 可能全来自 lambda 提权，pixel encoder 可能是死分支
2. **D1 SeamRefiner joint training 污染 transport** — D50 退化 −4 dB，根因是 refiner 梯度反传破坏 velocity 预测

## 实验矩阵（只有 2 个 GPU 槽）

| 实验 | GPU | 目标 | 基于 | 关键改动 |
|------|-----|------|------|---------|
| **N1** | GPU-0 | pixel encoder 消融 | Scheme C config | `pixel_forcing_disabled: true` |
| **N2** | GPU-1 | 解耦 SeamRefiner | Scheme C best.pt | freeze transport, 只训 refiner, 跳过 D50 |

## N1: Pixel Encoder 消融

**假设**: 如果关闭 pixel encoder 后 PSNR 不变，pixel forcing 无效；如果下降 ≥ 0.05 dB，pixel forcing 有效但需增强。

**Config**: `pet_flow_first_hop_224_50k_pixenc_ablation.yaml`
- 基于 Scheme C (lambda_max=0.12)
- `first_hop.pixel_forcing_disabled: true` — 强制 gate_pix=0，pixel_encoder 不贡献梯度
- 其他参数完全一致

**运行**:
```bash
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml
```

**评估**:
```bash
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml \
    --checkpoint <best.pt> --split val --max-slices 0
```

**Gate**: Δtransport_avg vs Scheme C best (35.569/35.942/36.505/36.806 → avg 36.206)

## N2: 解耦 SeamRefiner（Stage-2）

**假设**: 冻结 transport 只训 refiner，且跳过 D50，可以消除 patch 伪影而不退化 PSNR。

**Config**: `pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml`
- `--resume <Scheme_C_best.pt>`
- `training.freeze_backbone: true`（冻结 transport backbone）
- `training.freeze_first_hop_modules: true`（冻结 pixel_encoder, hop_residual_head, g_pix_raw）
- `first_hop.seam_refiner.enabled: true`
- `first_hop.seam_refiner.skip_d50: true`（新增：跳过 D50 passthrough 的 refiner）
- `training.max_steps: 20000`（refiner 只 9K params，不需要 50K 步）
- image_aux seam 为主损失

**运行**:
```bash
python train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

**评估**:
```bash
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml \
    --checkpoint <best.pt> --split val --max-slices 0 --decode-mode both
```

**Gate**:
- D50 PSNR 保持 ≥ 42.5 dB（不退化）
- D20-NORMAL seam_consistency 下降
- transport_avg 不退化 ≥ −0.02 dB

## 代码改动清单

| 文件 | 改动 |
|------|------|
| `pet_lr/model_first_hop.py` | `pixel_forcing_disabled` 支持 + `skip_d50` in decode_crop |
| `train_first_hop.py` | `freeze_first_hop_modules` 支持 |
| 2 个新 config | N1 + N2 |
