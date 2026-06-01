# Seam Decoder Provenance Reply

日期: 2026-06-01  
仓库版本: `foc_lite_hop0` @ `b88c3e9` (`review(0601): Codex task pack — A seam decoder provenance, B seam_strata_summary (A before B)`)  
范围: 只读代码考古；未修改训练/eval 主干，未启动任务 B。

## 结论先行

当前 V7 / V13 / A4-mid / A4-mid seed1337 的训练与评估使用的是同一个冻结 Stage-1 RAE checkpoint:

`/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt`

这个 checkpoint 的 decoder 权重是 `GeneralDecoder.decoder_pred` 线性 head + `unpatchify` 硬拼接路径，不是 `ConvDecoderHead`，也不是 `ConvOverlapHead`。因此 A4-mid 相对 V13 的差异不能归因于 decoder head 架构变化；它来自 PET 侧 transport 训练里的 `image_aux` 像素域监督路径。论文可以写“在固定冻结 RAE decoder 下，PET 侧 image_aux 约束 decoded hop0 prediction，抑制硬 unpatchify 后可见的残余 patch-boundary/seam artifact”；不能写成“本工作引入/替换了 conv/overlap decoder head”。

## 1. V7 / V13 / A4-mid 实际 decoder

### 加载链路

| 证据 | 文件 | 行号 | 结论 |
|---|---|---:|---|
| PET model 初始化先 build RAE，再按 `training.freeze_rae` 冻结 | `pet_lr/model_first_hop.py` | 269-291 | 当前 run 配置 `freeze_rae=true` 时，完整 RAE eval + `requires_grad_(False)` |
| 非 LoRA RAE 参数若可训练会报错 | `pet_lr/model_first_hop.py` | 456-460 | 防止误训练 decoder 本体 |
| PET decode 调 `self.rae.decode(z)`，未传 overlap flags | `pet_lr/model_first_hop.py` | 576-604 | 默认 raw RAE decode；这些配置未启用 `seam_refiner` |
| `build_rae` 调 RAE repo 的 `load_rae_model` | `pet_lr/model_first_hop.py` | 635-639 | decoder 来源在 RAE repo |
| RAE loader 构建 `RAE(..., decoder_config_path, decoder_patch_size=14)` | `/home/qujiaxiang/project/RAE/code/RAE/src/pet_flow/inference_pet_flow.py` | 41-60 | 无 conv head 参数 |
| RAE loader 从 `rae.checkpoint_path` 加载 `ckpt["model"]` | `/home/qujiaxiang/project/RAE/code/RAE/src/pet_flow/inference_pet_flow.py` | 72-80 | checkpoint 决定实际权重 |
| RAE 初始化固定创建 `GeneralDecoder` | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/rae.py` | 55-60 | 类不是 `ConvDecoderHead` / `ConvOverlapHead` |
| RAE 默认 decode 走 `decoder(...).logits` + `unpatchify` | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/rae.py` | 103-167 | `use_overlap_add=False` 默认硬 unpatchify |
| `GeneralDecoder` head 是 `nn.Linear` | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/decoder.py` | 559-562 | head key 应为 `decoder_pred.*` |
| `GeneralDecoder.unpatchify` 是 reshape/einsum 硬拼接 | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/decoder.py` | 636-680 | 无重叠融合 |
| `forward_hidden_overlap` 需要显式 flag | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/decoder.py` | 835-872 | PET 侧未调用 |

### 配置与日志

| Run | 训练配置 | `freeze_rae` / `image_aux` | RAE checkpoint | 训练 log | Eval log |
|---|---|---|---|---|---|
| V7 | `review/0505/local/configs/V7_gronwall_raw.yaml` | `freeze_rae=true`; `image_aux.enabled=true`; `lambda=0.04` | `review/0505/local/configs/V7_gronwall_raw.yaml`:346-373 | `review/0511/log_snapshots_20260516_163900/main_training/V7_train_20260516_163900.log`:47, 61, 3712 | `review/0511/fullval_psnr_clip3_20260516_173941/logs/planf_v7_v8_v6noise_best_last_fullval_psnr_clip3_gpu1_20260516_165948.log`:6, 23, 26, 35-36 |
| V13 | `review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml` | `freeze_rae=true`; `image_aux.enabled=false`; `lambda=0` | `review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml`:365-392 | `review/0516/V13_true_image_aux_ablation/V13_train_20260518_183547.log`:91, 113, 3770 | `review/0516/logs_eval/v13_best_fullval_psnr_clip3_20260521.log`:17, 20, 29-30 |
| A4-mid seed42 | `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml` | `freeze_rae=true`; `image_aux.enabled=true`; `lambda=0.08` | `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`:192-219 | `review/0521/A4_image_aux_lambda_08/A4_train_20260522_013850_gpu0_tmux.log`:36, 50, 3703 | `review/0521/A4_image_aux_lambda_08/fullval_eval/logs/a4_image_aux_lambda_08_best_eval_20260525_012246_gpu0.log`:17, 20, 29-30 |
| A4-mid seed1337 | `review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml` | `freeze_rae=true`; `image_aux.enabled=true`; `lambda=0.08` | `review/0525/A4_mid_seed1337/A4_mid_seed1337.yaml`:192-219 | `review/0525/A4_mid_seed1337/A4_mid_seed1337_train_20260526_152836_gpu3.log`:64, 86, 3738 | `review/0525/A4_mid_seed1337/fullval_eval/logs/a4_mid_seed1337_best_eval_20260530_gpu0.log`:15, 18, 27-28 |

### Checkpoint key 审计

`train_first_hop.py` 保存 `model.state_dict()` 到 `best.pt` / `last.pt`，并保存 `decoder_lora_enabled` 元数据，见 `train_first_hop.py`:1330-1351、2742-2758、2797-2814、2822-2838。因此 PET checkpoint 内的 `rae.decoder.*` key 可以直接验证实际 head。

只读检查结果:

| Checkpoint | `rae.decoder.decoder_pred` | `rae.decoder.conv_head` | `rae.decoder.conv_overlap` | `rae.decoder.pixel_conv` / `token_conv` | `decoder_lora_enabled` | 判定 |
|---|---:|---:|---:|---:|---|---|
| V7 best | 2 | 0 | 0 | 0 / 0 | `None` | `GeneralDecoder.decoder_pred` |
| V13 best | 2 | 0 | 0 | 0 / 0 | `False` | `GeneralDecoder.decoder_pred` |
| A4-mid seed42 best | 2 | 0 | 0 | 0 / 0 | `False` | `GeneralDecoder.decoder_pred` |
| A4-mid seed1337 best | 2 | 0 | 0 | 0 / 0 | `False` | `GeneralDecoder.decoder_pred` |
| Stage-1 RAE `best_model.pt` | `decoder.decoder_pred.weight (588, 512)` + bias | 0 | 0 | 0 / 0 | n/a | `GeneralDecoder.decoder_pred` |

三者完全一致: 同一 `rae.checkpoint_path`、同一 `decoder_config_path=/home/qujiaxiang/project/RAE/vit-mae`、同一 `decoder_patch_size=14`、同一默认 `GeneralDecoder` hard `unpatchify` decode path。

## 2. 历史上是否存在 conv_head / conv_overlap 阶段

存在 RAE 侧历史实验与脚本，但没有证据表明这些 head 被当前 PET V7/V13/A4 系列使用。

| 项 | 证据 | 结论 |
|---|---|---|
| `ConvDecoderHead` 代码存在 | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/conv_head.py`:1-23、31-115 | RAE 侧独立 head；文档写明替代 Linear + unpatchify，用 PixelShuffle |
| `ConvOverlapHead` 代码存在 | `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/conv_overlap_head.py`:1-26、76-139、198-237 | RAE 侧独立 head；dense grid + `F.fold` overlap-add |
| 对应训练脚本存在 | `/home/qujiaxiang/project/RAE/code/RAE/src/train_lora_dinov2_conv_head.py`:6、238、406；`/home/qujiaxiang/project/RAE/code/RAE/src/train_lora_dinov2_conv_overlap_head.py`:2-8、237-254、449 | 这些是 RAE decoder/head 实验，不是 PET transport 主线 |
| ConvOverlap 配置存在 | `/home/qujiaxiang/project/RAE/code/RAE/configs/stage1/pet_dinov2_lora_conv_overlap_head.yaml`:1-9、36-48 | 基于 192 RAE checkpoint，不是当前 224 Stage-1 checkpoint |
| RAE 旧审计称多种 patch artifact 修复尝试失败 | `/home/qujiaxiang/project/RAE/code/RAE/TECHNICAL_AUDIT_REPORT.md`:179-205；`/home/qujiaxiang/project/RAE/code/RAE/docs/patch_artifact_analysis.md`:135-147、182-190、610-628 | 可作为历史上下文；不能直接作为当前 V7/A4 贡献证据 |
| RAE git log 可见历史不足 | `git -C /home/qujiaxiang/project/RAE/code/RAE log -- ...` 只返回 `5951aa8` 与 `da87b66` | 本机 repo 不能给出 conv_head 被 conv_overlap 替换的精确 commit |

时间点可由文件 mtime / 产物推断但不能当 commit 级证据:

- `pet_patch_refine` 产物集中在 2026-02-10 至 2026-02-19。
- `pet_conv_overlap_head_s7_g27` checkpoint 集中在 2026-02-11 至 2026-02-14。
- `compare_three_decoders` 图集中在 2026-03-12。

因此“历史上存在 RAE 侧 conv/overlap/post-refine 探索”成立；“当前论文 before 图就是 conv_head 输出，随后被 conv_overlap 替换”在现有代码/日志中不成立。

## 3. 两类“伪影抑制”的归属边界

### RAE 侧 decoder/head 架构贡献

RAE 侧的 decoder/head 改动包括:

- `ConvDecoderHead`: token-domain conv + PixelShuffle + pixel residual refinement，见 `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/conv_head.py`:1-23、31-115。
- `ConvOverlapHead`: dense grid interpolation + conv fusion + `F.fold` overlap-add + optional pixel refinement，见 `/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/conv_overlap_head.py`:1-26、198-237。

这些如果写入论文，必须归为 RAE decoder architecture/post-processing exploration。当前 PET V7/V13/A4 headline 结果没有加载这些 head，不能把 A4-mid 的提升写成 decoder-head 架构贡献。

### PET 侧 image_aux 贡献

PET 侧 `image_aux` 的实际路径是:

1. `compute_hop0_image_losses` 用 transport model 预测 `z_pred`，再 `model.decode_crop(out["z_pred"])` 解码到像素域，见 `train_first_hop.py`:851-864。
2. `compute_first_hop_image_loss` 对 decoded `x_pred` 和 GT `x_dst` 计算 L1 / SSIM / seam，并按权重求和，见 `train_first_hop.py`:865-876、`pet_lr/losses_first_hop.py`:10-48。
3. seam loss 默认是预测自身 14px 网格处的一阶/梯度连续性惩罚，见 `pet_lr/losses.py`:107-124；extended seam 可用 GT 梯度/曲率，见 `pet_lr/losses.py`:127-194。
4. 训练总 loss 加入 `lambda_img * loss_img`，见 `train_first_hop.py`:2238-2245。
5. V13 训练日志 `lambda_img=0.0000` 且 `img=0`，见 `review/0516/V13_true_image_aux_ablation/V13_train_20260518_183547.log`:113；A4 训练日志 `lambda_img=0.0800` 且 `img/img_l1/img_ssim/img_seam` 非零，见 `review/0521/A4_image_aux_lambda_08/A4_train_20260522_013850_gpu0_tmux.log`:50。

因此 A4-mid 相对 V13 的改善路径是: 在同一个冻结 hard-unpatchify RAE decoder 下，PET transport 的 `z_pred` 被 image-domain L1/SSIM/seam loss 拉向更可解码的区域；不是 decoder head 替换。

## 4. 现有 seam / artifact before-after 图和脚本盘点

### PET_LatentResidual 当前仓库

未找到 `paper/main.pdf` 或 `paper/*.tex`。在当前仓库内没有可直接核验的论文 PDF figure provenance。因此如果外部论文稿里已有 before/after 图，必须回溯图片源文件；不能默认其 before 面板来自 conv head 或 conv overlap。

### RAE 侧现有图与脚本

| 目录 / 脚本 | 现有文件 | Provenance | 是否可作为当前 A4/V13 图 |
|---|---|---|---|
| `/home/qujiaxiang/project/RAE/outputs/compare_three_decoders/` | `slice_0100.png` ... `slice_1300.png`，mtime 2026-03-12 | `/home/qujiaxiang/project/RAE/code/RAE/scripts/compare_three_decoders.py`:2-8、43-51、230-240、269-294；192 pipeline，D50→MeanFlow→NORMAL，同一 predicted latent 上比较 Baseline RAE / BA-DRN / UNETR | 否。不是 V13/A4；不是 224 PET headline run |
| `/home/qujiaxiang/project/RAE/outputs/pet_patch_refine/vis_encoder_decoder/` | 7 张 slice 图，mtime 2026-02-10 | patch-refine config 指向 192 RAE checkpoint `/home/qujiaxiang/project/RAE/outputs/pet_lora_dinov2_pt_20260120_040402/epoch_0040.pt`，见 `/home/qujiaxiang/project/RAE/code/RAE/configs/stage1/pet_patch_refine.yaml`:1-20 | 否。RAE/BA-DRN 旧实验 |
| `/home/qujiaxiang/project/RAE/outputs/pet_patch_refine/vis_full_pipeline/` | 7 张 slice 图，mtime 2026-02-10 | `/home/qujiaxiang/project/RAE/code/RAE/scripts/vis_full_pipeline.py`:1-8、35-41、117-145、180-191、224-270；MeanFlow + RAE decode + BA-DRN refine | 否。不是 V13/A4；用 192 checkpoint |
| `/home/qujiaxiang/project/RAE/outputs/pet_conv_overlap_head/pet_conv_overlap_head_s7_g27/` | `best.pt`、`conv_overlap_head_epoch_0050/0100/0150/0160.pt` | ConvOverlap 训练配置见 `/home/qujiaxiang/project/RAE/code/RAE/configs/stage1/pet_dinov2_lora_conv_overlap_head.yaml`:1-9、36-48；训练脚本保存 `conv_overlap_head_epoch_*`，见 `/home/qujiaxiang/project/RAE/code/RAE/src/train_lora_dinov2_conv_overlap_head.py`:449 | 只能作为 RAE conv-overlap 历史证据；不能作为 A4/V13 before-after |

## Paper before/after 图的明确回答

当前 worktree 没有 `paper/main.pdf` 可检查。如果论文 PDF 中已有 before/after seam 图:

1. 若图片源于当前 V13 vs A4-mid headline 实验，则 before 面板应标注为:
   - variant: V13 true image_aux-off
   - checkpoint: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off/best.pt`
   - RAE decoder: frozen `GeneralDecoder.decoder_pred` + default hard `unpatchify`
   - RAE checkpoint: `/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt`
2. after 面板应标注为:
   - variant: A4-mid image_aux λ=0.08
   - checkpoint: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08/best.pt`（或 seed1337 对应 checkpoint，如果图来自 seed1337）
   - RAE decoder: 同一个 frozen `GeneralDecoder.decoder_pred` + default hard `unpatchify`
3. 若图片源于 RAE `compare_three_decoders` 或 `pet_patch_refine`，它不是 V13/A4 headline figure；图注必须写成 RAE-side decoder/refiner historical visualization，不能用来支撑 PET `image_aux` 贡献。
4. 现有证据不支持“before = conv_head，after = conv_overlap_head”的当前论文主结果叙述。

## 可写入论文的 novelty 边界陈述

> In all reported PET transport ablations, the Stage-1 RAE decoder is fixed and frozen: the runs load the same RAE checkpoint and decode with the standard `GeneralDecoder.decoder_pred` followed by hard `unpatchify`. Therefore, the V13→A4-mid improvement should be attributed to PET-side image-domain auxiliary supervision through the frozen decoder, which steers predicted latents toward images with lower residual patch-boundary artifacts; decoder-head redesigns such as ConvDecoderHead or ConvOverlapHead belong to separate RAE-side explorations and are not part of the PET headline result.

中文等价表述:

> 本文 PET 主结果不 claim 新 decoder head。所有 V7/V13/A4-mid 结果均使用同一冻结 RAE `GeneralDecoder`（linear `decoder_pred` + hard `unpatchify`）。A4-mid 相对 V13 的伪影改善来自 PET 侧 `image_aux` 在 transport 训练中通过冻结 decoder 对 `z_pred` 施加 L1/SSIM/seam 约束，使预测 latent 更可解码并降低残余 patch-boundary artifact；`ConvDecoderHead` / `ConvOverlapHead` 属 RAE 侧历史探索，不能归为本文 PET 主贡献。
