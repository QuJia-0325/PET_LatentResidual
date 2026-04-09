# Archive: V2.1 224 Decoder Adaptor Spec

Status:

- archive
- earlier `224 + decoder adaptor` design line
- not the current first-hop implementation baseline

## 项目名称

`PET Latent Residual v2.1`

## 一句话定义

在**不改变原始 `RAE encoder-decoder + latent-only transport` 主线**的前提下，
先用 `224 clean geometry` 去掉 `192->196` padding 引入的外圈 token 污染，
再针对 `predicted latent decode` 训练一个 **seam-aware / border-aware / hop-aware** 的小型 decoder adaptor，
用 pixel 域的 draft 信息做局部条件修补，
但**不引入第二条 transport 状态，也不改写下一跳 latent 动力学**。

## 文档目的

这份文档用于工程评审与任务拆分，回答以下问题：

1. 哪些部分属于必须保留的研究主线。
2. 为什么当前 `196/pad` 实现会出现明显视觉分块。
3. `224` 方案到底解决了什么，代价是什么。
4. decoder 该怎么更新。
5. pixel 域信息该如何融合，才不会背离 latent-only transport。
6. 实验如何分阶段推进，何时停线。

## 1. 背景与问题定义

当前系统的大方向是正确的：

- 使用 `RAE` 编码器-解码器建立图像到表征的映射。
- 在 latent 空间进行多时相 transport。
- 推理时按 `D50 -> D20 -> D10 -> D4 -> NORMAL` 级联 rollout。

该设计与以下论文主线一致：

- [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)
- [MeanFlow Transformers with Representation Autoencoders](https://arxiv.org/abs/2511.13019)

当前核心问题不是：

- latent transport 这一总体思路错误。

当前核心问题更像是：

- 你们当前这版 `decoder/unpatchify`
- `192 -> 196` padding 几何
- `predicted latent` 相对 `GT latent` 的偏移

三者叠加后，把视觉上的 block / seam / border artifact 放大了。

### 当前实现中的已知敏感点

当前默认 decode 路径是：

1. `DINOv2` 产生 patch token latent
2. ViT-MAE 风格 decoder 输出每个 token 对应的 patch 像素向量
3. `unpatchify()` 直接按 patch 网格重排成图像

相关实现：

- [rae.py](/home/qujiaxiang/project/RAE/code/RAE/src/stage1/rae.py)
- [decoder.py](/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/decoder.py)
- [stage1/README.md](/home/qujiaxiang/project/RAE/code/RAE/src/stage1/README.md)

同时，仓库里已经有一条很重要的旁证：

- 边缘 patch 和角落 patch 的有效像素比例在 `192->196` pad 设置下本来就不一致

见：

- [decoder_validity.py](/home/qujiaxiang/project/RAE/code/RAE/src/stage1/decoders/decoder_validity.py)

因此，当前视觉分块问题不是主观感觉，而是**当前几何与 decoder 实现的真实副作用**。

## 2. 设计边界

### 2.1 必须保留的研究主线

以下部分视为本项目的核心，不在本方案中推翻：

- `RAE encoder-decoder`
- `latent-space transport`
- latent 是唯一 rollout 状态
- hop-aware cascade：
  `D50 -> D20 -> D10 -> D4 -> NORMAL`
- 不引入 image-token 主轨迹
- 不引入 dual-state transport
- 不重做 encoder
- 不先重做 transport backbone

### 2.2 可以调整的实现层

以下部分属于工程实现，不算背离原始主线：

- 输入几何：`192->196 pad` 或 `192->224 resize`
- decoder 如何从 latent 还原图像
- decode 后的局部结构修补
- pad / border / seam 的特殊处理
- decoder-side 模块训练时使用什么分布

## 3. 几何路线设计

本方案分两条线，但只推荐一条主线推进。

### 主线

`224 clean-geometry latent pipeline + predicted-latent-aware decoder adaptor`

### 保底线

保留当前 `196/pad` 主线，只在 `predicted latent decode` 上加 seam adaptor

## 4. 对 `224` 方案的工程判断

### 4.1 当前 `224` 方案到底改了什么

你提供的实现：

- [train_lora_dinov2_clipfree_224.py](/home/qujiaxiang/project/RAE/code/RAE/src/train_lora_dinov2_clipfree_224.py)
- [pet_dinov2_lora_pt_224.yaml](/home/qujiaxiang/project/RAE/code/RAE/configs/stage1/pet_dinov2_lora_pt_224.yaml)
- [pet_rae_pt_224.py](/home/qujiaxiang/project/RAE/code/RAE/src/datasets/pet_rae_pt_224.py)

它不是“保持原图尺寸直接喂 224 模型”，而是：

- 先把 `192x192` 图像归一化
- 再用 `bicubic` resize 到 `224x224`
- 不再执行 `192->196` pad

### 4.2 它带来的收益

因为 `224 / 14 = 16`：

- 输入天然 patch-aligned
- 不再有 `192->196` 的 pad 外圈 token
- 不再存在边缘/角落 patch 有效像素比例不一致的问题

这意味着：

- 如果当前 artifact 里有明显成分来自 pad 外圈污染，
  `224` 路线会直接消掉这部分问题。

### 4.3 它不能自动解决的部分

`224` 并不会自动消除以下问题：

- DINOv2 仍然是 non-overlap patch encoder
- decoder 默认仍然是 patch-wise logits + `unpatchify`
- predicted latent 一旦偏离 GT manifold，patch 节律仍可能显化

因此：

- `224` 能去掉 **pad-induced artifact**
- `224` 不能保证去掉 **patch-wise decode artifact**
- `224` 不能保证修复 **predicted latent off-manifold** 问题

### 4.4 它的代价

这点必须写死：

- 当前 Stage 2 hop-aware 模型默认吃 `14x14` latent
- `224` 会把 latent 网格变成 `16x16`

因此：

- 不能只换 Stage 1 就沿用现有 Stage 2 checkpoint
- 需要重新提取 `224` latent
- 需要重训 Stage 2 transport
- decoder adaptor 也必须按新几何设计

所以 `224` 不是“小修 decoder”，而是：

**新的 clean-geometry latent 基线**

### 4.5 当前 `224` 配置的实验纯度问题

当前配置写着“与 192 配置完全一致”，但实际不是：

- `lora_rank = 16`
- `lora_alpha = 16`

相关配置：

- [pet_dinov2_lora_pt_224.yaml](/home/qujiaxiang/project/RAE/code/RAE/configs/stage1/pet_dinov2_lora_pt_224.yaml)

而当前 `192` 主线成功配置是 rank/alpha = 8。

因此如果要把 `224` 当作几何 A/B：

- 必须先把 LoRA rank/alpha 与 `192` 基线对齐
- 否则比较会同时混入容量变化

## 5. 总工程方案

### 5.1 推荐主方案

先建立新的 `224 clean-geometry latent pipeline`，再在其上加一个
**predicted-latent-aware decoder adaptor**

### 5.2 方案的中心思想

不是替换 transport，
不是让 pixel 成为第二条状态，
而是：

**保留 latent-only rollout，只修 `Dec(z_pred)` 的视觉可用性。**

## 6. Decoder 更新方案

### 6.1 不建议先重写主 decoder

第一阶段不建议：

- 全量换成 UNet decoder
- image-token 主轨迹
- dual-state transport
- 大型 CNN decoder 重构

原因：

- 会打乱原始 latent transport 主线
- 一旦有效，很难解释到底哪个因素起作用
- 你当前的问题不是指标太差，而是视觉不可用

### 6.2 建议引入 `decoder adaptor`

不是替换 decoder，而是在 decoder 后加一个小型局部 adaptor。

定义：

- `x_k^draft = Dec(z_k)`
- `x_{k+1}^draft = Dec(z_{k+1}^{pred})`
- `r_{k+1} = A(x_k^draft, x_{k+1}^draft, cond)`
- `x_{k+1}^{refined} = x_{k+1}^{draft} + r_{k+1}`

其中：

- `A` 是小型局部 residual adaptor
- `cond` 为 hop-aware 条件

### 6.3 adaptor 的职责

它不是第二个生成器，而是做局部修补：

- seam continuity
- patch boundary smoothing
- 局部强度校正
- border artifact 修补

### 6.4 adaptor 的硬约束

为了避免它偷换成 image translator，必须强制：

- 单通道 residual
- 输出层零初始化
- `tanh + max_residual` 幅度限制
- 只用局部卷积 block
- 不允许全局 attention
- 不让 residual 反馈到下一跳 latent

## 7. Pixel 域信息融合设计

### 7.1 不引入 pixel transport state

明确不做：

- `p_k -> p_{k+1}`
- image-token rollout
- dual-domain transport

### 7.2 pixel 域信息来源

pixel 域信息只来自 decoder draft：

- `Dec(z_k)`
- `Dec(z_{k+1}^{pred})`

### 7.3 推荐的 pixel 融合输入

建议 adaptor 输入为 3 个 image-domain 通道：

- `x_k^draft`
- `x_{k+1}^draft`
- `|x_{k+1}^draft - x_k^draft|`

再额外加入一个固定几何先验：

- seam mask
- 或 border band mask

### 7.4 条件信息

建议条件向量包含：

- `hop_id`
- `t_src`
- `t_dst`
- `log_dt`

### 7.5 融合方式

建议最小实现：

- image channels 先 concat
- 时间条件过两层 MLP
- 用 FiLM / AdaLN 去调制几层局部卷积 block
- 输出 residual

### 7.6 为什么第一版不喂真实 `x_k`

第一版不建议让 adaptor 直接看真实 `x_k`：

- 太容易滑向 hop-specific image translator
- 降低实验可解释性
- 你现在要测的是 `predicted latent decode` 是否还能被修

所以第一版只看 **decoded drafts**

## 8. 损失设计

### 8.1 主损失

针对 refined image：

- `L1`
- `SSIM`
- `seam consistency`
- residual amplitude penalty

### 8.2 几何相关加权

#### 如果走 `196/pad` 基线

强调：

- border-aware weighting
- seam 位置单独度量

#### 如果走 `224` 基线

pad 外圈污染消失后：

- `border-aware weighting` 可显著降权
- `seam loss` 仍建议保留

因为 hard unpatchify 的 patch 节律仍在。

### 8.3 rollout 监督位置

建议两级监督：

- 单跳 refined supervision
- final endpoint refined supervision

注意：

**residual 不参与下一跳 latent 状态。**

因此 rollout supervision 只约束视觉输出，不改变 transport 动力学。

## 9. 训练分布要求

这是本方案最关键的一条：

**decoder adaptor 的训练分布必须是 `predicted latent decode`，而不是 `GT latent decode`。**

也就是说，不能再沿用旧的：

- `Dec(E(x)) -> x`

而应该训练在：

- `Dec(z_pred) -> x_target`

建议准备两类 predicted corpus：

- single-hop predicted latents
- full-rollout predicted latents

## 10. 分阶段工程推进顺序

### Phase A：几何验证

目标：
判断 `224` 是否值得成为新主基线。

对照：

1. `196(pad)` 的 `Dec(E(x))`
2. `224(resize)` 的 `Dec(E(x))`

比较指标：

- 全图 PSNR / SSIM
- 外圈 band L1 / MSE
- seam 梯度断裂
- 视觉 block 程度

判断：

- 如果 `224` 明显更干净，pad 是重要因素
- 如果改善不大，主要问题不是 pad，而是 predicted latent decode

### Phase B：224 latent transport baseline

若 Phase A 证明 `224` 值得推进：

1. 训练新的 224 RAE
2. 提取新的 `16x16` latent
3. 训练新的 224 hop-aware transport baseline
4. 暂时不上 adaptor

目标：

判断“去掉 pad 之后，transport 输出还分不分块”

### Phase C：decoder adaptor

只在 Phase B 的 predicted latent decode 上加 adaptor。

推荐顺序：

1. 先跑单跳 `D50 -> D20`
2. 再跑 `D50 -> ... -> NORMAL` endpoint

## 11. 对照组设计

建议至少保留四组：

1. `196/pad + old transport + old decode`
2. `196/pad + old transport + decoder adaptor`
3. `224/resize + new transport + old decode`
4. `224/resize + new transport + decoder adaptor`

回答四个问题：

1. pad 是不是主因
2. decoder adaptor 是否有效
3. `224` 是否本身足够
4. `224 + adaptor` 是否叠加有效

## 12. 关键指标

不要只看 PSNR。

必须新增：

- 全图 PSNR / SSIM
- 外圈 band 的 L1 / MSE
- 14px seam 位置的梯度断裂
- `draft -> refined` 改善量
- 单跳视觉图
- full rollout endpoint 视觉图

## 13. 停线标准

### 对 `224` 基线

如果 `224` 相比 `196/pad`：

- seam 没明显改善
- 视觉块感没明显降低

则不值得重建整个 `224` Stage 2 基线。

### 对 decoder adaptor

如果 adaptor：

- 只能改善很小的视觉细节
- 对 full rollout endpoint 几乎没有稳定改善

则它应被定位成：

`可选后处理`

而不是主线方案。

## 14. 风险

### 风险 1：`224` 会改变 latent geometry

影响：

- 必须重建 Stage 2
- 不能复用现有 transport checkpoint

### 风险 2：adaptor 可能只修“看起来”

影响：

- 视觉更平滑
- 但 transport 动力学本身不改善

### 风险 3：adaptor 可能滑向 image translator

缓解：

- 单通道 residual
- 零初始化
- 幅度限制
- 局部卷积
- 不输入真实 `x_k`

## 15. 最终推荐

### 推荐执行顺序

1. 先把 `224` 做成纯几何 A/B
2. 如果 `224` 确实去掉了大部分 pad 主导 artifact，再建立新的 `224` latent baseline
3. 然后只在 `predicted latent decode` 上加一个 seam-aware、hop-aware、小 residual adaptor

### 不推荐的先手方向

- 双域联合 transport
- image-token 主轨迹
- 大改 encoder
- 大改 transport backbone

## 16. 最终一句话方案

> 保留原始的 `RAE encoder-decoder + latent-only transport` 主线；先用 `224 clean geometry` 去掉 pad 外圈污染，再针对 `predicted latent decode` 训练一个 seam-aware、border-aware、hop-aware 的小型 decoder adaptor，用 pixel 域 draft 信息做局部条件修补，而不引入第二条 transport 状态。
