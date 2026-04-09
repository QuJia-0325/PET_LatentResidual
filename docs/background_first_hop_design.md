# Background: First-Hop Design

Status:

- design draft only
- kept for reasoning history
- superseded for implementation by `main.md`

## 目标

在**不推翻原始 `RAE encoder-decoder + latent-only transport` 主线**的前提下，
针对当前最核心的问题：

`D50 -> D20` 第一跳 latent transport 质量不足

给出现有主线上的**定点增强方案**。

## 总体结论

这不是“第四方案”。

这是：

**现有方案 1 升级成方案 3**

即：

- 保留现有 `hop-aware latent rollout`
- 不做 `segmented latent forcing + stitching`
- 在共享 latent 主干上增加：
  - `shared trunk + lightweight hop-specific heads`
  - `first-hop-only source-pixel conditioning`
  - `first-hop-only image-space supervision`

## 1. 主状态定义

真正 rollout 的状态仍然只有 latent：

`z_D50 -> z_D20 -> z_D10 -> z_D4 -> z_NORMAL`

不引入：

- image-token 主轨迹
- dual-state transport
- pixel rollout state

这点必须保持不变。

## 2. 为什么这样设计

当前第一跳问题的核心不是：

- decoder 单独太弱
- 或 segmented stitching 能绕开级联

更像是：

- `z_D50` 信息贫弱、过度语义化
- `D50 -> D20` 是 retained 4-hop 里最难的一跳
- 当前主干只看 latent，不看 source image
- 当前主干主要是全局 token mixer，对第一跳局部结构恢复不够友好
- 训练目标没有直接把第一跳 decode 后图像质量绑住

因此：

> 解决第一跳的重点应放在“latent update 怎么预测得更对”，而不是最后 decode 完再补。

## 3. 现有主干保留部分

以下部分保留不动：

- `PETFlowDiTDHHopAware` 的条件编码逻辑
  - `t_src`
  - `t_dst`
  - `log_dt`
  - `hop_id`
- `MeanFlow4HopRollout` 的 latent rollout
- straight-through rollout mixing
- 现有 4-hop latent-only path

## 4. 新增结构一：shared trunk + hop-specific heads

### 4.1 目的

当前 trunk 虽然已经 hop-aware，但最终输出映射几乎仍是共享的。

第一跳和后续跳任务性质差异很大：

- 第一跳更像结构恢复 / denoise
- 后续跳更像 refinement

所以需要让输出层有轻量级 hop specialization。

### 4.2 设计

保留 shared latent trunk。

在 shared output 之外，叠加一个轻量 hop-specific residual head：

`v_shared = Head_shared(h, c)`

`v_hop = Head_hop[k](h)`

`v = v_shared + v_hop`

其中：

- `Head_hop[k]` 很小
- 零初始化
- 只做 residual correction

### 4.3 约束

必须满足：

- 不能把 hop head 做大
- 不能让每个 hop 变成独立小模型
- shared head 仍然是主输出
- hop head 只做小补偿

## 5. 新增结构二：first-hop-only source-pixel conditioning

### 5.1 目的

当前第一跳只看 `z_D50`。

如果 `z_D50` 已经丢失了部分局部几何，第一跳 transport 就缺少恢复结构所需的信息。

最合理的补法不是让 pixel 成为第二条状态，
而是让 source pixel 特征作为**第一跳 latent update 的条件分支**。

### 5.2 形式

只在第一跳：

`z_D20^pred = F(z_D50, phi(x_D50), t_src, t_dst, hop_id)`

后续三跳仍然是：

`z_next^pred = F(z_curr, t_src, t_dst, hop_id)`

也就是说：

- pixel conditioner 只服务 hop 0
- rollout 主状态仍然只有 latent

### 5.3 `phi(x_D50)` 的设计

建议新增一个轻量 `FirstHopPixelEncoder`：

- 输入：`x_D50`
- 单通道 `192x192`
- 浅层 CNN stem
- 下采样到 latent 对应 token lattice
- 投影到 trunk hidden size

输出：

`f_pix in R^{B x N x d}`

### 5.4 注入方式

不建议复杂 cross-attention 作为第一版。

建议最小实现：

- 仅在 trunk 前几层注入
- 用可学习门控
- 门控初始值设为 0

形式：

`h_0 = h_0 + g * f_pix`

其中：

- `g` 是可学习标量或通道门
- 初始化为 0

这保证训练开始时模型严格退化为当前 baseline。

### 5.5 约束

必须满足：

- pixel conditioner 只在第一跳启用
- 只注入 trunk 前几层
- 不扩散到全路径
- 不成为第二条 rollout 状态

## 6. 新增结构三：first-hop-only image-space supervision

### 6.1 目的

当前训练主要约束 latent loss：

- velocity
- latent endpoint
- rollout consistency

但第一跳 decode 后图像质量没有被直接绑住。

这会导致：

- latent 数值“还行”
- decode 后结构已经明显偏了

### 6.2 形式

仅对第一跳样本加入：

`x_D20^pred = Dec(z_D20^pred)`

并约束：

`x_D20^pred ~= x_D20^gt`

### 6.3 损失

建议第一版只加辅助损失：

- `L1`
- `SSIM`
- `Seam / border-aware loss`

### 6.4 约束

必须满足：

- 只对 hop 0 计算
- 只作用于有效 `192x192` 区域
- 只是辅助目标
- 不能盖过 latent rollout 主目标

## 7. 数据流草图

### 训练时

对于一般 hop：

1. 取 `z_src, z_dst, t_src, t_dst, hop_idx`
2. 走 shared trunk
3. 叠加 hop-specific residual head
4. 计算 latent velocity / endpoint loss

对于第一跳：

1. 额外取 `x_D50`
2. `FirstHopPixelEncoder(x_D50) -> f_pix`
3. 将 `f_pix` 注入 trunk 前几层
4. 预测 `z_D20^pred`
5. 额外 decode 得到 `x_D20^pred`
6. 加 first-hop image-space auxiliary loss

### 推理时

第一跳：

`z_D20^pred = F(z_D50, phi(x_D50), t_src, t_dst, hop_id=0)`

后续三跳：

`z_next^pred = F(z_curr, t_src, t_dst, hop_id)`

最终仍然是 latent-only cascade。

## 8. 为什么这不是第四方案

因为以下核心都没变：

- latent 仍然是唯一 rollout state
- 现有 hop-aware latent rollout 框架不变
- 现有 straight-through rollout 逻辑不变
- 现有 `D50 -> D20 -> D10 -> D4 -> NORMAL` 物理路径不变

变化只是：

- 第一跳 latent update 信息更充分
- 输出层允许轻量 hop specialization
- 第一跳 decode 质量被辅助约束

所以这是：

**当前主线的定点增强**

不是新体系。

## 9. 预期收益

理论上最可能改善的是：

- 第一跳局部结构恢复
- 第一跳 decoder-friendliness
- 第一跳之后的误差积累速度
- 最终 endpoint 的视觉可用性

## 10. 风险

### 风险 1
hop-specific heads 过强，滑向“隐式四个模型”

### 风险 2
pixel conditioner 扩散到全路径，变成双域系统

### 风险 3
image-space loss 过重，主干退化成 image translator

### 风险 4
第一跳根因如果主要是 latent 表征信息已不可恢复，则 pixel conditioning 收益有限

## 11. 当前建议

这份 draft 的建议顺序是：

1. 保留现有 hop-aware latent rollout 作为骨架
2. 先加 lightweight hop-specific heads
3. 再只给第一跳加 source-pixel conditioning
4. 最后再加 first-hop image-space supervision

不要一次同时大改。

## 最终一句话

> 在不改变 latent-only rollout 的前提下，把第一跳从“只看语义化 latent”升级成“共享 latent 主干 + 轻量 hop 专门输出 + source pixel 条件支持 + decode 图像辅助监督”的定点增强版本。
