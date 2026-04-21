# 0421 Local — N1/N2 实验设计与代码

## 背景

NIGHTMARE 审查(4.35/10, CONDITIONAL HOLD)指出当前系统的核心缺陷。

### 审计评分

| 维度 | 权重 | 分数 |
|------|------|------|
| Problem-Mechanism Fit | 20% | 5/10 |
| Causal Evidence | 20% | 4/10 |
| Robustness | 15% | 4/10 |
| Adversarial Survivability | 15% | 3/10 |
| Artifact Mitigation | 15% | 4/10 |
| Engineering Feasibility | 10% | 7/10 |
| Novelty | 5% | 4/10 |
| **Total** | | **4.35/10** |

### 审计发现的全部问题（按严重度排序）

#### Reviewer-B 10 条攻击

**O1. 所有改进在争夺 ±0.10 dB**
Scheme A+C (最佳结果) transport_avg = 36.229 vs chainstable 36.128，仅 +0.101 dB。
各 hop 的噪声范围 ±0.08-0.12 dB，无置信区间证明这不是评估方差。

**O2. D1 SeamRefiner 是基本设计失败**
D50 PSNR 从 42.622 暴跌到 38.592（−4.03 dB）。refiner 在 GT decode 上施加了
不必要的残差修正。tanh(tail(...)) 在 46K 步后已偏离零初始化。
guard 自身也 flag 了 best_d1_any 为 guard_ok=0。

**O3. Pixel encoder 容量严重不足且无消融**
150K params 将 224×224 压缩到 768×16×16，AdaptiveAvgPool 做 14× 空间压缩。
无任何消融实验单独验证 pixel encoder 的贡献——可能是完全的死分支。

**O4. E1 诊断导致矛盾的资源分配**
E1 证明 decoder gap 仅 0.41 dB (3.7%)，transport gap 10.74 dB (96.3%)。
然后 D1 花 GPU 去修 decoder——这是"数据库在着火时优化 CSS"。

**O5. "级联累积"解释不成立**
SeamRefiner 仅在 decode 时独立应用，不参与 latent rollout。
D4/NORMAL 的 −0.35 dB 退化真因是 **joint training 时 refiner 梯度反传
污染了 transport backbone 的 velocity 预测**。

**O6. E2 证伪 FOC-lite 是空结论**
half-step 比 full-step 差 −0.045 dB，说明积分精度足够。
但没有产出可操作的改进方向。

**O7. iREPA alignment 是死分支**
+0.023 dB 仅在 last.pt 上观察到，未在 best.pt 上验证。
在噪声范围内，无统计检验支持。SpatialAlignmentProjector 增加了
参数和 hook 复杂度但无可证实的收益。

**O8. 无感知/临床指标**
仅报告 PSNR（clip3 变体）。无 SSIM、LPIPS、FID，无放射科医生评估，
无病灶检测灵敏度/特异性。医学影像论文仅有 PSNR 在任何临床期刊不可发表。

**O9. DINOv2 encoder 从未被质疑**
DINOv2 在自然图像上预训练，PET 图像统计特性完全不同（低对比度、泊松噪声）。
LoRA 适配增加极少容量。无对比：从头训练 / PET 预训练 encoder / 简单 CNN encoder。

**O10. 4-hop 级联 schedule 是任意的**
D50→D20→D10→D4→NORMAL 为什么是这些剂量？为什么 4 hop？
无 2-hop 或 8-hop 的消融。hop schedule 是关键超参但从未优化。

#### Reviewer-A 的维度问题

- **机制优先级倒置**：150K pixel encoder 向 196K 维 latent 空间注入，维度不匹配
- **因果证据仅为相关性**：E1 做了分解但不能因果解释为什么 transport 丢 10.74 dB
- **鲁棒性不足**：单数据集、无交叉验证、无 test set、D50 列膨胀 all_avg
- **D1 对抗性低**：refiner 在最简单 case 上退化 4 dB，reviewer 会立刻质疑

#### Reviewer-C 可信度评估

| 结论 | 信度 |
|------|------|
| Transport gap 主导 (96.3%) | 高 |
| FOC-lite 被证伪 | 中高 |
| λ=0.12 是最优 | 中（仅 3 个点） |
| SeamRefiner 有前景 | 低（D50 灾难性退化） |
| iREPA 有帮助 | 极低（噪声范围内） |
| Pixel encoder 有效 | 低-中（零消融） |

### 不支持的声明

1. "Hop0 pixel forcing 有效" — 无单独消融
2. "HopResidualVelocityHead 有正贡献" — 无消融
3. "SeamRefiner 减少 patch artifacts" — D50 退化使结论无效
4. "iREPA 改善 transport" — 噪声范围内
5. "当前架构对此任务近最优" — 无与更简单 baseline 对比
6. "DINOv2 latent 空间适合 PET" — 无 encoder 消融

---

## 实验矩阵（2 个 GPU 槽）

| 实验 | GPU | 目标 | 基于 | 关键改动 |
|------|-----|------|------|---------|
| **N1** | GPU-0 | pixel encoder 消融 | Scheme C config | `pixel_forcing_disabled: true` |
| **N2** | GPU-1 | 解耦 SeamRefiner | Scheme C best.pt | freeze transport, 只训 refiner, 跳过 D50 |

## N1: Pixel Encoder 消融

**假设**: 如果关闭 pixel encoder 后 PSNR 不变，pixel forcing 无效；如果下降 ≥ 0.05 dB，pixel forcing 有效但需增强。

**Config**: `pet_flow_first_hop_224_50k_pixenc_ablation.yaml`
- 基于 Scheme C (lambda_max=0.12)
- `first_hop.pixel_forcing_disabled: true` — 跳过 pixel forcing 路径（`_apply_hop0_pixel_forcing` 直接返回原始 z_src），pixel_encoder 冻结不贡献梯度；gate_pix 仍正常计算用于监控
- 其他参数完全一致

**代码改动**:
- `model_first_hop.py`: 新增 `pixel_forcing_disabled` 读取 + `_apply_hop0_pixel_forcing` 跳过

**Gate**: Δtransport_avg vs Scheme C best (36.206)
- 退化 ≥ 0.05 dB → pixel encoder 有效，考虑扩容
- 退化 < 0.02 dB → pixel encoder 是死分支，需根本重设计

## N2: 解耦 SeamRefiner（Stage-2）

**假设**: 冻结 transport 只训 refiner，且跳过 D50，可消除 patch 伪影而不退化 PSNR。

**Config**: `pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml`
- `--resume <Scheme_C_best.pt>`
- `training.freeze_backbone: true`
- `training.freeze_all_except_seam_refiner: true`（冻结除 seam_refiner 外所有可训练参数）
- `first_hop.seam_refiner.enabled: true`
- `first_hop.seam_refiner.skip_first_tp: true`（跳过 D50 的 refiner）
- `training.max_steps: 20000`（refiner 仅 9K params）

**代码改动**:
- `train_first_hop.py`: 新增 `freeze_all_except_seam_refiner` 支持
- `model_first_hop.py`: 新增 `seam_refiner.skip_first_tp` 配置读取
- `eval_first_hop_224_clip3.py`: eval 时 tp_i==0 且 skip_first_tp 时跳过 refiner

**Gate**:
- D50 PSNR 保持 ≥ 42.5 dB（不退化）
- D20-NORMAL seam_consistency 下降
- transport_avg 不退化 ≥ −0.02 dB

---

## 运行命令

```bash
# N1: Pixel Encoder 消融（GPU-0, 50K steps, ~24h）
CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 \
python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml

# N2: 解耦 SeamRefiner Stage-2（GPU-1, 20K steps, ~10h）
CUDA_VISIBLE_DEVICES=1 TQDM_DISABLE=1 \
python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml \
    --resume /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_imgaux_boost_gpu2_nopbar_20260418_010035/best.pt
```

评估命令（训练完成后）：

```bash
# N1 eval
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_pixenc_ablation/best.pt \
    --split val --max-slices 0

# N2 eval（使用 --decode-mode both 获得 raw/refined 双路对比）
python eval_first_hop_224_clip3.py \
    --config configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml \
    --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_20k_seam_refiner_stage2/best.pt \
    --split val --max-slices 0 --decode-mode both
```

## 代码改动清单

| 文件 | 改动 |
|------|------|
| `pet_lr/model_first_hop.py` | `pixel_forcing_disabled` 支持 + `skip_first_tp` 配置 |
| `train_first_hop.py` | `freeze_all_except_seam_refiner` 支持 |
| `eval_first_hop_224_clip3.py` | `skip_first_tp` 在 eval 中跳过 D50 refiner |
| `configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml` | N1 config |
| `configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml` | N2 config |
