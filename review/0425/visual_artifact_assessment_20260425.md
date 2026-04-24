# 200K Best 可视化伪影检查说明（2026-04-25）

分支：`foc_lite_hop0`  
目的：给远程同学一份针对当前 `200k transport_v3` 可视化结果的简明审查说明，重点看现在的伪影问题到底是什么样。

## 1. 本次可视化使用的权重与图像语义

本次图像**不是 final checkpoint**，而是当前训练过程中的 **best.pt 快照**：

- 配置：[config.yaml](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3/config.yaml)
- 权重：[best.pt](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_200k_transport_v3/best.pt)
- 训练日志：[transport_v3_200k_train_gpu1.log](/home/qujiaxiang/project/PET_LatentResidual/review/0424/logs/transport_v3_200k_train_gpu1.log)

按日志最新快照，当前 best 刷新到了 `step=58400`，对应 `val_select_score=0.000645`。训练还未结束，因此**没有 `last.pt`**，也还没有这组 200k 的正式 full-val 评估。

图像语义已经按参考 notebook 修正为：

- 每一行：一个 source timepoint，分别为 `D50 / D20 / D10 / D4`
- 第 1 列：`<source> Input`
- 第 2 列：`<source> → NORMAL Pred`
- 第 3 列：`GT NORMAL`
- 第 4 列：`|Pred - GT|`

显示方式：

- 图像显示窗口：`SUV clip[0,3]`
- 标题 PSNR：`calc_psnr_clip3`

可视化脚本：
[visualize_first_hop_chain_clip3.py](/home/qujiaxiang/project/PET_LatentResidual/review/0425/visualize_first_hop_chain_clip3.py)

输出目录：
[first_hop_224_200k_transport_v3_best_clip3](/home/qujiaxiang/project/PET_LatentResidual/review/0425/visuals/first_hop_224_200k_transport_v3_best_clip3)

## 2. 代表性样本

建议远程同学优先看这三张：

- 相对干净样本：[slice_1500.png](/home/qujiaxiang/project/PET_LatentResidual/review/0425/visuals/first_hop_224_200k_transport_v3_best_clip3/first_hop_224_200k_transport_v3_best_slice_1500.png)
- 中等难度样本：[slice_0100.png](/home/qujiaxiang/project/PET_LatentResidual/review/0425/visuals/first_hop_224_200k_transport_v3_best_clip3/first_hop_224_200k_transport_v3_best_slice_0100.png)
- 明显困难样本：[slice_1100.png](/home/qujiaxiang/project/PET_LatentResidual/review/0425/visuals/first_hop_224_200k_transport_v3_best_clip3/first_hop_224_200k_transport_v3_best_slice_1100.png)

另外一张典型“整体结构对、内部纹理差”的样本：

- [slice_2100.png](/home/qujiaxiang/project/PET_LatentResidual/review/0425/visuals/first_hop_224_200k_transport_v3_best_clip3/first_hop_224_200k_transport_v3_best_slice_2100.png)

## 3. 我看到的主要伪影模式

### 3.1 不是典型的 checkerboard / patch seam 主导

从当前几张图看，最突出的误差不是规则网格或明显 patch 边界线，而是：

- 高摄取区域内部的**颗粒化 / 斑驳化**
- 亮热点周围的**纹理破碎**
- 局部区域的**对比度压缩与模糊**

误差图主要集中在器官或病灶内部的高亮结构，背景整体比较干净。这更像是：

- transport 预测后的 latent 仍带有结构性偏差
- decoder 在 off-manifold latent 上把这种偏差表现成亮区纹理噪声

而不像单纯的“解码器拼块缝”问题。

### 3.2 误差会随着 rollout 距离增加而变大

从本次 11 个可视化切片的 PSNR 汇总看：

- `D50 -> NORMAL` 平均：`36.3388 dB`
- `D20 -> NORMAL` 平均：`38.1554 dB`
- `D10 -> NORMAL` 平均：`39.8280 dB`
- `D4 -> NORMAL` 平均：`42.8722 dB`

这说明越早开始 rollout，最终到 `NORMAL` 的误差越大。  
这和肉眼观感一致：`D50->NORMAL` 通常最差，`D4->NORMAL` 最接近 GT。

因此，当前伪影并不是“所有 hop 同等糟糕”，而是明显具有**链式累积**特征。

### 3.3 困难样本上的问题很集中

最差样本之一是 `slice 1100`：

- `D50 -> NORMAL = 26.60 dB`
- `D20 -> NORMAL = 28.69 dB`
- `D10 -> NORMAL = 29.92 dB`
- `D4 -> NORMAL = 33.61 dB`

这类样本里，问题不是简单的边缘偏移，而是：

- 内部高亮区被重绘成更“发花”的纹理
- 局部亮度结构被抹平又夹杂假亮点
- 误差图在整个高摄取主体内都很高

这说明当前难点是**高摄取区域的细粒度强度分布恢复**，不只是几何轮廓。

## 4. 初步判断

基于当前图像，我的判断是：

1. 当前问题**更像 transport 误差累积 + off-manifold decode 纹理不稳**，而不是单一 seam artifact。  
2. 全局结构已经基本能对上，说明模型并不是完全不会做；问题主要卡在亮区内部的精细强度模式。  
3. 由于 `D4->NORMAL` 明显好于 `D50->NORMAL`，链式误差传播仍然是核心因素。  
4. 但仅凭这些图，**还不能**把责任直接归到 decoder 或直接归到 exposure bias，仍需要结合后续 Path A / full-val 指标一起判断。

## 5. 远程同学审查时建议关注的问题

建议远程同学重点回答下面 4 个问题：

1. 这些误差更像是 **transport rollout 累积**，还是更像 **decoder 对异常 latent 的纹理放大**？
2. 当前伪影是否主要发生在 **高亮热点内部**，而不是背景或边界？
3. 这些图里是否还能看出明显的 **patch seam / chessboard** 结构？如果没有，就不应把“seam artifact”当主叙事。
4. 从 `D50->NORMAL` 到 `D4->NORMAL` 的持续改善，是否足以支持“问题随 hop 累积”这一判断？

## 6. 当前结论边界

这份文档只回答“现在图像看起来哪里不对、像哪类伪影”，不回答以下问题：

- `200k` 是否已经超过 `50k`
- `200k best` 的正式 full-val 指标是多少
- `200k last` 会不会更好

这些都还需要等训练结束，再做标准 full-val 评估。

