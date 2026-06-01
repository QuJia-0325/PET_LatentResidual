# 远程同学需补充的信息

日期: 2026-06-01  
用途: 帮助确认论文图源、实验归因与难图像分析边界，方便远程同学拉取后直接回复。  
相关说明文档: `review/0601/server/project_systematic_explanation_20260601.md`

## 1. 论文 before/after figure 的原始来源

请确认当前论文中用于展示 patch-boundary / seam artifact before-after 的 figure 原始文件路径。

需要明确:

- 这些图是否来自当前 PET 主线的 V13 vs A4 实验；
- 如果来自 V13/A4，请提供:
  - V13 图像文件路径；
  - A4 图像文件路径；
  - 对应 slice index；
  - 使用的 checkpoint；
  - 使用的 eval / visualization 脚本；
- 如果来自 RAE 旧实验，请说明是否属于:
  - `ConvDecoderHead`;
  - `ConvOverlapHead`;
  - patch refine / BA-DRN / UNETR；
  - `compare_three_decoders`；
- 如果论文图混用了 RAE 旧图和 PET V13/A4 结果，请明确每个 panel 的来源。

我当前的代码与 checkpoint 审计结论是: 当前 V13/A4 headline PET 结果使用同一个冻结 Stage-1 RAE `GeneralDecoder.decoder_pred` + hard `unpatchify`，不是 `ConvDecoderHead` / `ConvOverlapHead`。因此论文 figure provenance 必须与这个归因保持一致。

## 2. 难图像中原始噪声/信号淹没的额外证据

请确认项目是否已有可用于分析“难图像中原始噪声太大、信号被淹没”的额外标注或指标。

优先需要:

- lesion ROI 或病灶 mask；
- background ROI；
- SUV / uptake 相关统计；
- CNR / SNR 指标；
- reader-study 或诊断任务指标；
- 按病灶大小、摄取强度、解剖区域划分的 metadata；
- slice-level 或 patient-level 的难度标注。

如果没有这些信息，当前只能用间接 proxy 分析:

- V13 NORMAL PSNR 作为 global difficulty proxy；
- D50 PSNR 作为 input quality / low-dose quality proxy；
- per-slice seam/ext-seam 作为 artifact risk proxy，但当前 CSV 还没有导出 per-slice seam 字段。

当前已有结果显示 A4 在 low-baseline / low-D50-quality 分层中仍有稳定 PSNR 收益，但这不能等价证明已经解决低 SNR 下的病灶级信号淹没问题。若论文要专门讨论这点，建议补充 lesion/ROI 或至少补充 per-slice seam/ext-seam 分层分析。
