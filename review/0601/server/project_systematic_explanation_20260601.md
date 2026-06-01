# PET LatentResidual 项目系统性说明

日期: 2026-06-01  
工作目录: `/home/qujiaxiang/project/PET_LatentResidual`  
目标目录: `review/0601/server`  
版本背景: `foc_lite_hop0` @ `b88c3e9`  
范围: 基于当前代码、V13/A4 full-val eval、已有训练/eval 日志和可视化切片进行分析；未修改训练/eval 主干代码。

## 0. 我的核心判断

我对当前项目的判断可以压缩为四句话:

1. 当前 PET 主线的技术贡献不是 decoder-head 架构替换，而是在固定冻结 RAE decoder 下，用 PET 侧 `image_aux` 把 latent transport 的输出约束到更可解码、更少 seam artifact 的区域。
2. V13→A4 的主要可支持结论是: 在同一 Stage-1 RAE checkpoint、同一 hard-unpatchify decode path 下，开启 image-domain auxiliary supervision 后，PSNR 上升，同时 seam/extended-seam 指标系统下降。
3. 当前结果已经支持“有效压制残余 patch-boundary/seam artifact”，但不支持“彻底解决低剂量 PET 的原始噪声/低 SNR 问题”；后者是另一个困难源，不能混同为 seam artifact。
4. 论文中最稳妥的表述应是“artifact-aware latent transport through a frozen decoder”，而不是“we redesign the decoder head”。

## 1. 项目的问题拆解

这个项目实际上同时面对三类不同问题:

| 问题 | 现象 | 主要来源 | 当前方法是否直接解决 |
|---|---|---|---|
| Stage-1 decoder 几何伪影 | 14px patch 边界、硬拼接 seam | RAE decoder 的 token-to-pixel 重建路径，尤其是 hard `unpatchify` | 不是从架构上解决；当前主线用 PET 侧约束缓解其可见后果 |
| Transport off-manifold 伪影 | 预测 latent 解码后出现局部不连续、边界增强、纹理破碎 | PET latent dynamics 预测落到 RAE decoder 不友好的 latent 区域 | 是，A4 的 `image_aux` 主要解决这一层 |
| 原始低剂量噪声/信号淹没 | 难图像中噪声强、病灶/组织信号弱，PSNR 或视觉质量受限 | 输入数据自身 SNR 不足，可能叠加 tracer uptake、计数统计、slice anatomy 差异 | 只能间接改善；当前证据不能声称已经根治 |

因此我建议论文叙述不要把所有问题都叫“伪影”。更准确的分层是:

- `seam artifact`: decoder 网格边界附近的结构性不连续；
- `transport artifact`: latent 预测偏离可解码流形后被 decoder 放大的伪影；
- `low-SNR noise`: 低剂量 PET 原始噪声过强导致的信号淹没。

当前 A4 的价值在第二层最明确，并通过第三方可观测的 seam/PSNR 指标体现为第一层可见伪影下降；但它不是一个完整的低剂量去噪理论。

## 2. 当前架构的实际链路

### 2.1 Stage-1 RAE 是固定基座

V7 / V13 / A4-mid / A4-mid seed1337 使用同一个 Stage-1 RAE checkpoint:

`/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt`

代码链路是:

1. `pet_lr/model_first_hop.py` 构造 PET first-hop model，并通过 `build_rae` 加载 RAE。
2. RAE loader 使用 `/home/qujiaxiang/project/RAE/code/RAE/src/pet_flow/inference_pet_flow.py`。
3. RAE 本体在 `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/rae.py` 中构造 `GeneralDecoder`。
4. 默认 decode 进入 `GeneralDecoder.decoder_pred`，再调用 hard `unpatchify`。

checkpoint key 审计显示:

| Checkpoint | `rae.decoder.decoder_pred` | `rae.decoder.conv_head` | `rae.decoder.conv_overlap` | 判定 |
|---|---:|---:|---:|---|
| V7 best | 2 | 0 | 0 | `GeneralDecoder.decoder_pred` |
| V13 best | 2 | 0 | 0 | `GeneralDecoder.decoder_pred` |
| A4-mid seed42 best | 2 | 0 | 0 | `GeneralDecoder.decoder_pred` |
| A4-mid seed1337 best | 2 | 0 | 0 | `GeneralDecoder.decoder_pred` |
| Stage-1 RAE `best_model.pt` | `decoder.decoder_pred.weight (588, 512)` + bias | 0 | 0 | `GeneralDecoder.decoder_pred` |

这说明当前 headline PET 结果没有使用 `ConvDecoderHead` 或 `ConvOverlapHead`。

### 2.2 PET 侧 image_aux 才是 V13→A4 的单轴变化

当前关键 ablation 是:

- V13: `image_aux.enabled=false`，`lambda=0`。
- A4: `image_aux.enabled=true`，`lambda=0.08`。

训练逻辑是:

1. transport model 预测 `z_pred`；
2. `model.decode_crop(out["z_pred"])` 通过冻结 RAE decoder 得到 `x_pred`；
3. `compute_first_hop_image_loss` 对 `x_pred` 和 GT 图像计算 L1 / SSIM / seam；
4. 总训练 loss 加入 `lambda_img * loss_img`。

这条路径的意义是: PET transport 不只在 latent 空间逼近目标，还被要求“通过同一个冻结 decoder 解码后也合理”。这相当于用 decoder 的像素域响应作为约束，避免模型产生 decoder 会放大的 latent error。

## 3. 为什么不是 ConvDecoderHead / ConvOverlapHead

这个问题必须严谨回答，因为它关系到论文贡献归属。

### 3.1 代码与 checkpoint 均不支持该说法

`ConvDecoderHead` 和 `ConvOverlapHead` 在 RAE 仓库中确实存在，也有历史训练脚本和产物。但当前 PET V13/A4 主线:

- 配置指向的是同一个 224 Stage-1 RAE checkpoint；
- PET decode 调用默认 `self.rae.decode(z)`；
- 没有向 RAE decode 传入 overlap-add flag；
- PET checkpoint 内没有 `conv_head` / `conv_overlap` 权重 key；
- full-val eval 的 `decode_mode` 是 `default`。

所以，若论文主结果写成“替换 decoder head 后减少伪影”，会与当前证据冲突。

### 3.2 更合理的叙述

正确的贡献边界应写成:

> We keep the Stage-1 RAE decoder fixed and frozen. The improvement comes from PET-side image-domain auxiliary supervision, which constrains predicted latents through the frozen decoder and reduces visible residual patch-boundary artifacts.

中文表述:

> 本文 PET 主结果不引入新的 decoder head。我们固定并冻结 Stage-1 RAE decoder，通过 PET 侧 image-domain auxiliary supervision 约束 latent transport 的预测结果，使其通过 hard-unpatchify decoder 解码后具有更低的残余 patch-boundary artifact。

如果需要讨论 `ConvDecoderHead` / `ConvOverlapHead`，建议放在“历史探索/未纳入主结果/未来工作”中，而不是作为当前 A4 的原因。

## 4. Full-val eval 结果解释

本次复核使用的是:

- 脚本: `/home/qujiaxiang/project/PET_LatentResidual/eval_first_hop_224_clip3.py`
- metric: `src.utils.metrics.calc_psnr_clip3`
- split: `val`
- slices: `7403`
- decode: `default`
- seam patch size: `14`

结果文件:

- V13 JSON: `review/0601/server/v13_first_hop_224_val_clip3_eval.json`
- V13 CSV: `review/0601/server/v13_first_hop_224_val_clip3_eval.csv`
- V13 log: `review/0601/server/v13_eval.log`
- A4 JSON: `review/0601/server/a4_first_hop_224_val_clip3_eval.json`
- A4 CSV: `review/0601/server/a4_first_hop_224_val_clip3_eval.csv`
- A4 log: `review/0601/server/a4_eval.log`

### 4.1 Aggregate metrics

| Timepoint | V13 PSNR | A4 PSNR | ΔPSNR | V13 seam | A4 seam | seam 变化 | V13 ext-seam | A4 ext-seam | ext-seam 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D50 | 42.6224 | 42.6224 | +0.0000 | 0.010048 | 0.010048 | 0.00% | 0.002265 | 0.002265 | 0.00% |
| D20 | 35.2172 | 35.4918 | +0.2746 | 0.006322 | 0.004733 | -25.13% | 0.003207 | 0.002927 | -8.72% |
| D10 | 35.5971 | 35.8864 | +0.2893 | 0.005592 | 0.004279 | -23.48% | 0.002907 | 0.002660 | -8.52% |
| D4 | 36.1122 | 36.4593 | +0.3470 | 0.005417 | 0.004154 | -23.31% | 0.002627 | 0.002367 | -9.90% |
| NORMAL | 36.4943 | 36.8939 | +0.3996 | 0.005341 | 0.004129 | -22.69% | 0.002425 | 0.002148 | -11.43% |

关键解释:

- D50 完全相同是预期结果，因为 first-hop rollout 以 `z_d50` 作为链条起点，D50 是直接 decode 起点 latent，不经过 V13/A4 transport 更新。
- 从 D20 到 NORMAL，A4 同时提升 PSNR 并降低 seam；这说明 image_aux 不只是提升某个单一标量，而是在预测链条的多个后续时间点持续有效。
- NORMAL 的 PSNR 增益最大，说明误差约束在靠近最终目标时仍有积累收益。

### 4.2 是否已经有效压制伪影

如果“伪影”特指 patch-boundary/seam artifact，那么我认为答案是: 已经有效压制，但不是彻底消除。

支持理由:

1. A4 在 D20/D10/D4/NORMAL 的 seam consistency 全部下降约 23%–25%；
2. extended seam 也下降约 8%–11%，说明不是只优化了预测自身网格连续性，也改善了相对 GT 的边界区误差；
3. PSNR 同时上升，排除了“牺牲整体 fidelity 来抹平 seam”的简单解释；
4. `review/0601/server/slices_v13_vs_a4/` 下的固定 slice 可视化中，7 个抽查 slice 全部显示 A4 seam 下降。

但必须保留边界:

- hard `unpatchify` 本身仍然存在；
- D50 的 seam 均值仍为 `0.010048`，说明 Stage-1 decoder 的硬拼接路径并非天然无伪影；
- A4 是通过训练 transport 让预测 latent 更适配这个 decoder，而不是从 decoder 结构上移除 hard boundary。

所以论文里应写“suppresses residual seam artifacts”，不要写“eliminates artifacts”。

## 5. 对 hard unpatchify 的判断

我对 hard unpatchify 的判断是: 它足够作为当前 frozen decoder baseline，但不是一个理论上理想的最终解。

### 5.1 为什么它仍可用

当前 Stage-1 RAE checkpoint 在 D50 direct decode 上有较高 PSNR:

- D50 PSNR: `42.6224`
- D50 ext-seam: `0.002265`

这说明 hard-unpatchify decoder 并不是完全失败的 decoder；否则 PET transport 结果不可能稳定建立在它上面。

### 5.2 为什么它仍是风险源

同一个 D50 direct decode 的 seam consistency 是:

- D50 seam: `0.010048`

这个数值高于 V13/A4 后续 timepoint 的 seam 均值。直觉上，这意味着 hard-unpatchify 仍会在某些输入分布或局部结构上暴露 patch-boundary 不连续。A4 能缓解预测 latent 解码后的 seam，但它不能证明 hard-unpatchify 本身已经足够优雅。

### 5.3 对论文叙述的影响

当前论文最好把 hard unpatchify 作为一个固定约束条件:

> Given a fixed hard-unpatchify Stage-1 decoder, our image_aux training reduces decoder-visible artifacts in PET latent transport.

不要把 hard unpatchify 描述为方法亮点。真正亮点是: 即使 decoder 不变，我们仍能通过 transport training 改善其输出可见质量。

## 6. 难图像与原始噪声问题

用户提出的担忧是合理的: 在较难图像上，原始噪声可能过大，信号被淹没。我的判断是:

- 这个问题真实存在；
- 但当前 per-slice PSNR 分层结果不支持“A4 只在简单图像有效、难图像无效”；
- 现有证据还不足以证明“A4 已经解决低 SNR 信号淹没”。

### 6.1 用 V13 NORMAL PSNR 作为难度 proxy

按 V13 NORMAL PSNR 三分位分层:

| 分层 | n | ΔPSNR mean | ΔPSNR median | A4 win-rate | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| hard / low baseline | 2468 | +0.3906 | +0.3710 | 96.03% | +0.0978 | +0.7141 |
| mid | 2467 | +0.3635 | +0.3674 | 92.74% | +0.0592 | +0.6756 |
| easy / high baseline | 2468 | +0.4447 | +0.4539 | 91.94% | +0.0429 | +0.8283 |

解释:

- 难图像组的平均收益仍然接近整体均值；
- 难图像组 win-rate 最高；
- 因此 A4 的效果不是只发生在 easy cases。

### 6.2 用 D50 PSNR 作为输入质量 proxy

按 V13 D50 PSNR 三分位分层:

| 分层 | n | ΔPSNR mean | ΔPSNR median | A4 win-rate | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| low D50 quality | 2468 | +0.3955 | +0.3769 | 95.66% | +0.0936 | +0.7324 |
| mid D50 quality | 2467 | +0.3695 | +0.3674 | 94.20% | +0.0827 | +0.6703 |
| high D50 quality | 2468 | +0.4338 | +0.4497 | 90.84% | +0.0228 | +0.8241 |

解释:

- 输入质量较低的组仍有稳定收益；
- A4 在低 D50 quality 组并没有崩溃；
- 但这只是 global PSNR proxy，不等价于真实病灶级 SNR 或低 uptake 区域分析。

### 6.3 相关性

NORMAL ΔPSNR 与 baseline 难度 proxy 的相关性较弱:

- corr(V13 NORMAL PSNR, ΔPSNR) = `0.1107`
- corr(V13 D50 PSNR, ΔPSNR) = `0.1000`
- corr(V13 D20 PSNR, ΔPSNR) = `0.1280`

这说明 A4 的增益与“图像本来容易/困难”没有强耦合。换句话说，当前结果更像是一个相对普遍的 transport regularization 收益，而不是只挑简单样本有效。

### 6.4 仍然缺少什么

当前 CSV 只有 per-slice PSNR，没有 per-slice seam/ext-seam 字段。因此我们还不能做最关键的二维分析:

> 高噪声/低质量输入 × 高 seam-risk slice 上，A4 是否仍然降低 seam？

这也是下一步最该补的证据。已有 aggregate seam 结果很好，但论文如果要专门回应“难图像中噪声淹没信号”，应该补充:

1. per-slice seam/ext-seam 导出；
2. 按 V13 seam risk 三分位分层；
3. 按 D50 quality 三分位分层；
4. 做 `seam-risk × input-quality` 2D strata；
5. 对每格报告 ΔPSNR、Δseam、Δext-seam、win-rate；
6. 固定规则选图，避免 cherry-picking。

## 7. 可视化证据的位置与作用

当前可视化放在:

`review/0601/server/slices_v13_vs_a4/`

包括:

- `slice_0100_normal_v13_vs_a4.png`
- `slice_0300_normal_v13_vs_a4.png`
- `slice_0500_normal_v13_vs_a4.png`
- `slice_0700_normal_v13_vs_a4.png`
- `slice_0900_normal_v13_vs_a4.png`
- `slice_1100_normal_v13_vs_a4.png`
- `slice_1300_normal_v13_vs_a4.png`
- `slice_metrics.json`

这些图的用途是 qualitative sanity check:

- 同一 slice 比较 GT / V13 / A4；
- 展示 absolute error 与 V13-A4 difference；
- 标出 14px seam grid；
- 标题中包含 PSNR/seam 指标。

它们可以支撑“视觉上 seam artifact 被压制”的叙述，但论文最终图建议按明确规则重选:

1. 一个典型平均收益样本；
2. 一个 hard / low baseline 样本；
3. 一个 high seam-risk 样本；
4. 一个 A4 失败或收益很小的样本。

这样比只展示漂亮样本更有说服力。

## 8. 论文 claim 边界

### 8.1 当前数据支持的 claim

我认为当前数据支持以下 claim:

> Under a fixed frozen Stage-1 RAE decoder, PET-side image-domain auxiliary supervision improves first-hop latent transport and suppresses residual patch-boundary artifacts in decoded PET images.

更具体地:

- V13→A4 在 full-val 7403 slices 上 NORMAL PSNR +0.3996 dB；
- NORMAL seam consistency 下降 22.69%；
- NORMAL extended seam 下降 11.43%；
- per-slice NORMAL PSNR win-rate 93.57%；
- 难图像 proxy 分层下仍有稳定收益。

### 8.2 当前数据不支持的 claim

不建议写:

1. “我们提出了新的 decoder head”；
2. “A4 使用 ConvDecoderHead / ConvOverlapHead 消除了伪影”；
3. “hard unpatchify 已经不是问题”；
4. “我们彻底解决了低剂量 PET 噪声淹没信号问题”；
5. “该方法已证明对所有病灶级诊断任务有效”。

这些 claim 要么与代码证据冲突，要么缺少 lesion-level / reader-study / task-level 证据。

### 8.3 推荐论文表述

推荐中文:

> 我们固定并冻结 Stage-1 RAE decoder，使所有 ablation 共享同一 hard-unpatchify 解码路径。A4 相对 V13 的提升来自 PET transport 训练中的 image-domain auxiliary supervision，而非 decoder head 替换。Full-val 结果显示，A4 在提高 PSNR 的同时显著降低 seam consistency 与 extended-seam 指标，说明该约束能将预测 latent 推向更可解码、边界更连续的区域。

推荐英文:

> All PET transport ablations use the same frozen Stage-1 RAE decoder with the standard linear prediction head followed by hard unpatchify. The improvement from V13 to A4 is therefore attributed to PET-side image-domain auxiliary supervision rather than decoder-head redesign. Full-validation results show that this supervision improves PSNR while consistently reducing seam-related metrics, indicating that it steers predicted latents toward decoder-friendly regions with fewer residual patch-boundary artifacts.

## 9. 我对整个项目的更高层理解

### 9.1 项目的强点

这个项目的强点不是单一模块，而是一个清晰的两阶段策略:

1. Stage-1 RAE 学习 PET 图像的 token-latent 表示；
2. PET latent transport 在这个表示空间中建模 dose/timepoint 迁移；
3. image_aux 把 latent-space learning 与 pixel-space clinical image quality 重新连接起来。

这比纯 pixel-to-pixel denoising 更有潜力，因为它把“低剂量到正常剂量”的问题转成了 latent dynamics 问题；也比纯 latent loss 更稳，因为最终质量仍由 decoded image 约束。

### 9.2 当前最重要的科学叙事

我建议把项目主线概括为:

> Low-dose PET recovery is not only a denoising problem; it is also a decoder-aware latent transport problem. A latent prediction that is close in feature space can still decode poorly if it falls into decoder-unfriendly regions. Image-domain auxiliary supervision closes this gap by regularizing the transport trajectory through the frozen decoder.

这个叙事能解释为什么 A4 在 decoder 不变的情况下仍能减少伪影，也能自然引出 seam loss。

### 9.3 当前最大风险

最大风险是贡献边界被写乱:

- 如果把当前 A4 的结果归因到 ConvDecoderHead / ConvOverlapHead，审稿人只要看 checkpoint 或代码 key 就会发现不一致；
- 如果把 seam artifact 与低 SNR 噪声混为一谈，审稿人可能要求 lesion-level 或 reader-study 证据；
- 如果只给 aggregate PSNR，不给 seam/strata/visual failure cases，artifact claim 会显得不够扎实。

### 9.4 当前最值得补的实验

我建议优先补三件事:

1. `per-slice seam/ext-seam` eval 导出: 这是支撑 artifact claim 的关键缺口。
2. `seam-risk × D50-quality` 分层表: 用来回答“难图像/高噪声是否仍有效”。
3. 固定规则的论文图: 包含 average / hard / high-seam / failure case，增强可信度。

如果还有资源，再考虑:

- lesion/background ROI 上的 CNR 或 uptake recovery；
- seed1337 与 seed42 的重复性整理；
- 与 RAE-side conv/overlap 历史探索明确拆分成 supplement 或 future work。

## 10. 最终建议

我建议当前论文采用保守但强的 claim:

> A4 does not solve all PET noise. It solves a narrower but important failure mode: decoder-visible seam artifacts caused or amplified by latent transport under a fixed hard-unpatchify RAE decoder.

这条 claim 的好处是:

- 与代码一致；
- 与 eval 数字一致；
- 能解释 V13→A4 的 improvement；
- 不夸大到低剂量 PET 全局去噪或 decoder 架构创新；
- 留出后续工作空间。

我的最终判断是: 当前项目已经有一个可防守的主结果，但论文必须把“decoder-side historical exploration”和“PET-side image_aux headline contribution”彻底分开。只要这个边界清楚，A4 的结果是有说服力的；如果这个边界混乱，反而会削弱项目可信度。
