# Archive: First-Hop Implementation Spec Old

Status:

- archive
- older duplicate of the first-hop implementation spec
- superseded by `main.md`

## 0. Scope

本规范用于 `PET_LatentResidual` 仓库内的新实验线。

目标：

- 保留 `latent-only rollout` 主状态
- 保留共享 `hop-aware` latent trunk
- 只在 `hop0: D50 -> D20` 上增加受控增强

不做：

- segmented latent forcing + stitching
- image-token rollout state
- dual-state transport
- decoder-side residual adaptor 作为主解

---

## 1. 模块改动

### 1.1 `pet_lr/data.py`

保留现有 4-hop 对齐数据集思路，新增 first-hop 所需字段与校验。

目标类：

- `PETFirstHop4HopDataset`

职责：

- 加载 `latents_{split}.pt`
- 加载 `preprocessed_data_*.pt`
- 返回 latent pair、rollout latent path、raw image pair、rollout raw image path
- 启动时执行一次 latent/image slice 对齐校验

必须返回的字段：

- `z_src`: `[768, 14, 14]`
- `z_dst`: `[768, 14, 14]`
- `x_src`: `[1, 192, 192]`
- `x_dst`: `[1, 192, 192]`
- `z_rollout`: `[5, 768, 14, 14]`
- `x_rollout`: `[5, 1, 192, 192]`
- `t_src`: `float32`
- `t_dst`: `float32`
- `hop_idx`: `long`
- `pair_idx`: `long`
- `slice_idx`: `long`
- `hop0_mask`: `bool`

约束：

- 图像统一为单通道、`192x192`、`[-1, 1]`
- `hop0_mask = (hop_idx == 0)`
- 不因为 later hops 需要纯 latent，就删掉 image 字段；统一接口，trainer 负责只在 hop0 使用
- 必须显式检查 latent 与 image 的 slice 顺序一致，不能只靠“默认假设”

### 1.2 `pet_lr/model_firsthop.py`

新增 first-hop 实验模型文件，不覆盖当前 `pet_lr/model.py` 的 `v2.1 residual refinement`。

目标模块：

- `HopResidualVelocityHead`
- `FirstHopPixelEncoder`
- `FirstHopConditionInjector`
- `FirstHopAssistModel`

#### `HopResidualVelocityHead`

职责：

- 在 shared output 之外增加轻量 hop-specific residual velocity correction

形式：

`v = v_shared + lambda_hop[hop_idx] * v_hop`

硬约束：

- `v_hop` 只看 trunk 最后层特征
- `lambda_hop` 可学习，初值为 `0`
- `v_hop` 最后一层零初始化
- `Head_hop` 参数量必须显著小于 shared output head

#### `FirstHopPixelEncoder`

职责：

- 把 `x_D50` 变成弱表达的局部结构条件

输入输出：

- 输入：`x_src`，形状 `[B, 1, 192, 192]`
- 输出：`f_pix`，形状 `[B, 14*14, d_enc]`

实现约束：

- 浅层 CNN
- 使用 `AdaptiveAvgPool2d(14, 14)` 对齐到 latent lattice
- 再线性投影到 encoder hidden size
- 不允许全局 Transformer
- 不允许深 U-Net

说明：

- `192 -> 14x14` 是弱对齐辅条件，不宣称严格 patch 对齐

#### `FirstHopConditionInjector`

职责：

- 只在 hop0 的 encoder 前几层，把 `f_pix` 注入 shared trunk

形式：

`h = h + g_pix * f_pix`

约束：

- `g_pix` 可学习，初值 `0`
- 只允许注入 encoder 前 `2~4` 层
- decoder side 不注入
- hops `1~3` 完全关闭

#### `FirstHopAssistModel`

职责：

- 复用现有 hop-aware backbone
- 在不改变 rollout state 定义的前提下支持 hop0 pixel conditioning
- 提供统一的 pair-step 与 rollout-step 接口

必须暴露的方法：

- `predict_latent_step(z_src, t_src, t_dst, hop_idx, x_src=None) -> dict`
- `forward_pair(z_src, z_dst, t_src, t_dst, hop_idx, x_src=None) -> dict`

`predict_latent_step` 返回字段：

- `v_shared`
- `v_hop`
- `v_pred`
- `z_pred`
- `pixel_used`

约束：

- `x_src` 只有 `hop_idx == 0` 时才会被读取
- later hops 必须退化为普通 latent-only path
- decoder 参数冻结，但 hop0 图像损失允许梯度穿过 decoder 回到 `z_pred`

### 1.3 `pet_lr/losses.py`

保留当前基础损失实现，新增 hop0 辅助图像损失组合。

新增函数：

- `hop0_image_aux_loss(pred, target, border_map, patch_size, cfg) -> dict`

输出字段：

- `loss_img_total`
- `loss_img_l1`
- `loss_img_ssim`
- `loss_img_seam`

### 1.4 `pet_lr/rollout.py`

保留 straight-through mixing 逻辑，新增 first-hop 外部条件入口。

必须新增的方法：

- `rollout_latent_chain_firsthop(model, z_rollout, x_rollout, rollout_times, alpha, straight_through) -> List[dict]`

行为约束：

- 第 0 跳调用 `predict_latent_step(..., x_src=x_rollout[:, 0])`
- 第 1 到 3 跳调用 `predict_latent_step(..., x_src=None)`
- pixel 信息不形成新的内部 rollout state

### 1.5 `train_firsthop.py`

新增训练入口，不修改 `train_v21.py`。

职责：

- 训练 `FirstHopAssistModel`
- 复用现有优化器、scheduler、EMA、mixed precision 框架
- 记录 hop0 专属指标

---

## 2. 接口规范

### 2.1 Dataset -> Trainer

每个 batch 必须统一包含：

- latent pair
- image pair
- latent rollout path
- image rollout path
- `hop0_mask`

不允许 trainer 通过 `hop_idx == 0` 时再额外读磁盘。

### 2.2 Trainer -> Model

pair-step 接口：

```python
out = model.forward_pair(
    z_src=batch["z_src"],
    z_dst=batch["z_dst"],
    t_src=batch["t_src"],
    t_dst=batch["t_dst"],
    hop_idx=batch["hop_idx"],
    x_src=batch["x_src"],
)
```

要求：

- trainer 永远传 `x_src`
- model 内部只在 `hop0_mask.any()` 时启用 pixel branch

rollout-step 接口：

```python
preds = rollout_latent_chain_firsthop(
    model=model,
    z_rollout=batch["z_rollout"],
    x_rollout=batch["x_rollout"],
    rollout_times=rollout_times,
    alpha=alpha,
    straight_through=straight_through,
)
```

### 2.3 Inference / Sampler

推理入口必须支持：

```python
z_d20 = model.predict_latent_step(z_d50, t_src, t_dst, hop_idx=0, x_src=x_d50)["z_pred"]
z_d10 = model.predict_latent_step(z_d20, t_src, t_dst, hop_idx=1, x_src=None)["z_pred"]
```

推理约束：

- `x_D50` 只作为 hop0 外部条件
- 从 hop1 开始只传播 `z_curr`
- 不缓存、不传播 pixel hidden state

---

## 3. Loss 规范

### 3.1 基础 latent loss

保留现有两部分：

- `L_pair_latent`
- `L_rollout_latent`

来源：

- velocity loss
- latent endpoint loss
- multi-endpoint rollout loss

### 3.2 hop0 image auxiliary loss

只在 `hop0_mask == True` 的样本上计算：

`x_d20_pred = Dec(z_d20_pred)`

`L_img_hop0 = w_l1 * L1 + w_ssim * SSIM + w_seam * Seam`

约束：

- 只在 `192x192` crop 上计算
- decoder 参数冻结
- decoder forward 不得包在 `no_grad()`
- hops `1~3` 不算任何图像监督

### 3.3 总损失

第一版总损失固定为：

`L_total = L_pair_latent + lambda_rollout * L_rollout_latent + lambda_img(step) * L_img_hop0`

其中：

- `lambda_img(step)` 需要 warmup
- 起始值必须远小于 latent 主损失
- 第一版不加 rollout endpoint image loss

### 3.4 推荐调度

- `lambda_img = 0` for warmup period
- 之后线性升到小值上限
- 推荐先把上限控制在 latent 主损失量级以下

---

## 4. Trainer 改动

### 4.1 数据流

每个训练 step 按以下顺序：

1. 计算所有 hop 的 `L_pair_latent`
2. 计算全路径 `L_rollout_latent`
3. 仅对 batch 内 `hop0_mask` 子集计算 `L_img_hop0`
4. 按调度组合成 `L_total`
5. 反向传播到 shared trunk、hop head、pixel encoder

### 4.2 冻结策略

第一版固定：

- shared hop-aware backbone：可训练
- hop-specific residual head：可训练
- first-hop pixel encoder / injector：可训练
- RAE decoder：冻结参数

### 4.3 采样策略

必须保留或增强 hop0 的采样权重。

要求：

- 不能让 hop0-only 模块只吃自然均匀采样
- 继续使用 pair weighting 或 sample prob 提高 hop0 更新频次

### 4.4 日志指标

至少记录：

- `loss_pair_latent`
- `loss_rollout_latent`
- `loss_img_hop0`
- `hop0_decode_l1`
- `hop0_decode_ssim`
- `hop0_decode_seam`
- `lambda_img`
- `mean_abs_v_hop0`
- `g_pix`

### 4.5 启动校验

训练启动时必须先打印：

- latent/image slice 对齐检查结果
- hop0 batch 占比
- decoder 是否冻结
- `lambda_hop` 与 `g_pix` 初始化值

---

## 5. Sampler 改动

### 5.1 训练 sampler

rollout 时：

- hop0 使用 `x_rollout[:, 0]`
- hop1~hop3 不再读取 `x_rollout`

### 5.2 推理 sampler

新增一条明确规则：

- sampler 允许接收 `x_start`
- 仅在第一跳将其传入 model
- 后续 hops 一律传 `None`

### 5.3 禁止事项

sampler 不允许：

- 保存 pixel hidden state 到下一跳
- 在 hop1~hop3 继续使用 `x_D50`
- 调用多个分段模型再做轨迹拼接

---

## 6. 实验顺序

### E0

控制组：

- 当前 shared hop-aware latent rollout baseline

### E1

只加 `HopResidualVelocityHead`

目的：

- 验证 hop specialization 本身是否能改善 first hop

### E2

在 E1 基础上加 `FirstHopPixelEncoder + gated injection`

约束：

- 不加任何 image loss

目的：

- 验证 source pixel 条件是否能改善 first-hop latent update

### E3

在 E2 基础上加 `hop0 image auxiliary loss`

约束：

- decoder 冻结
- `lambda_img` 小值 warmup

目的：

- 验证 decode-aware 约束是否进一步提升 hop0 可解码性

### E4

只在 E3 生效后再调：

- hop0 sample weight
- `lambda_img` 上限
- pixel injection 层数

### 停线规则

出现以下任一情况即停止继续加复杂度：

- hop0 latent 指标无改善
- hop0 decode 指标改善但全链 endpoint 退化
- later hops 明显受损
- pixel 分支在 later hops 形成隐式依赖

---

## 7. 版本结论

第一版实现目标不是重写 transport，而是：

- 在 shared latent rollout 上
- 用轻量 hop-specific output correction
- 加上 hop0 source-pixel 弱条件
- 再用 hop0 decode-aware 辅助监督

把第一跳做成受控增强，而不是把系统改成双域轨迹。
