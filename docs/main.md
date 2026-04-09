# Main

Status:

- current baseline
- first-hop 方案的唯一实现基准
- `Idea1 / CCT-224` 已在源代码与正式 `chainstable` 配置中落地

## 0. Idea 管理（持续维护）

本文件是实现规范主文档；idea 决策以本节 + `IDEA_REPORT.md` 同步维护。

当前时间点（2026-04-09）用于 `224 first-hop` 线路的 Gate-1 排序：

1. **CCT-224（Counterfactual Consistency Training）**
   - 在现有 `pair + rollout + hop0 image aux` 框架上增加“teacher-forced 路径 vs pure-pred 路径”的一致性约束。
   - 目标：降低训练/推理分布偏移，减少级联漂移。
   - Novelty score: **8.4/10**
2. **ΔB-aware Adaptive Reweighting**
   - 用 A/B/C/D 诊断中的 `ΔB=B-A` 作为在线重加权信号，动态调节 hop0 与中后跳权重。
   - 目标：把“诊断结论”闭环到训练控制。
   - Novelty score: **7.6/10**
3. **Uncertainty-Gated Hop0 Forcing（备选）**
   - 将 hop0 门控从全局标量扩展为样本自适应。
   - Novelty score: **6.3/10**

Gate-1 结论：**先做 1，再叠加 2；3 暂缓。**

执行约束：
- 未经用户批准，不改模型架构代码。
- 先完成实验设计与评估口径固化，再进入实现。
- 所有新想法以不破坏本规范第 1.1 节硬约束为前提。
- 每个 idea 在 Gate 评审时必须提供：
  1) **Novelty score（数值）**；
  2) **Three-party adversarial review**（Proposer / Novelty Skeptic / Engineering Skeptic）；
  3) **Gate decision**（PASS/HOLD/DROP）与理由。

## 1. 适用范围

本规范只定义一条新实验线：

- 保留 `RAE encoder-decoder + latent-only 4-hop rollout`
- 保留 `D50 -> D20 -> D10 -> D4 -> NORMAL`
- 不做 `segmented latent forcing + stitching`
- 不引入 image-token 主轨迹
- 不引入 dual-state transport
- 不改变 rollout 主状态，主状态始终只有 latent

本规范承接以下既有结论：

- [background_v21_residual_refinement.md](/home/qujiaxiang/project/PET_LatentResidual/docs/background_v21_residual_refinement.md)
  - latent 是唯一 rollout state
  - pixel 模块不能反馈成下一跳 latent state
- [background_first_hop_design.md](/home/qujiaxiang/project/PET_LatentResidual/docs/background_first_hop_design.md)
  - 第一跳是当前主瓶颈
  - 允许 `lightweight hop-specific heads`
  - 允许 `first-hop-only source-pixel conditioning`
  - 允许 `first-hop-only image-space supervision`

本规范不替换现有 `v2.1 residual refinement` 线路，只定义 first-hop transport 增强线路。

## 1.1 硬规范

以下约束属于 `v1` 的硬门槛，不允许在实现时弱化：

- latent / image 对齐校验失败必须直接中止训练
- 每个 training step 必须包含 hop0 样本
- pair-hop0、rollout-step-0、inference-step-0 必须共用同一条内部实现路径
- `FirstHopPixelEncoder` 必须保持弱表达，参数量受上限约束
- `L_img_hop0` 不得导致 decoder 解冻

以下约束属于 `v1` 的实现优化项，可在不改变主设计的前提下调整：

- 训练集是否保留完整 `x_rollout`

## 2. 模块拆分

### 2.1 数据模块

新增文件：

- `pet_lr/data_first_hop.py`

新增类：

- `PETFirstHopAligned4HopDataset`

职责：

- 读取 `latents_{split}.pt`
- 读取 `preprocessed_data_*.pt`
- 只使用 `D50/D20/D10/D4/NORMAL`
- 返回 latent pair、rollout latent path、source/target 图像
- 在数据层显式保留 `slice_idx`
- 提供一次性对齐检查入口
- 提供抽样内容一致性检查所需的 slice 访问入口

关键规则：

- 图像使用单通道 `192x192`
- 图像归一化到 `[-1, 1]`
- 不把 `196x196` pad 图像作为 pixel conditioner 输入
- `NORMAL` 仍从任意 PT 文件的 `x_0` 读取
- `latents_*.pt` 和 `preprocessed_data_*.pt` 的 slice 顺序必须先校验再训练
- 对齐校验失败必须 `raise` 并在第一个 optimizer step 前中止训练
- 训练启动日志必须打印对齐校验摘要和抽样 slice 检查结果
- 对齐检查不能只验证 shape、slice 数和 index；必须额外做抽样内容一致性检查
- 内容一致性检查必须基于 `Crop192(Dec(z_gt))` 与对应 raw image 的粗匹配结果
- 抽样内容一致性检查至少覆盖 `D50`、`D20`、`NORMAL`
- 每个时间态至少检查 `alignment_check_num_samples` 个 slice
- 必须记录并打印 `decoded_vs_raw_l1` 与 `decoded_vs_raw_corr` 摘要；若出现系统性异常则中止训练

`v1` 默认阈值：

- `alignment_check_num_samples = 16`
- matched 对照的平均 `decoded_vs_raw_l1` 必须优于 mismatched 对照至少 `10%`
- matched 对照的平均 `decoded_vs_raw_corr` 必须高于 mismatched 对照至少 `0.10`
- 若任一时间态不满足上述条件，则视为存在系统性错位并 fail-fast

规范接口：

```python
class PETFirstHopAligned4HopDataset(Dataset):
    def __init__(
        self,
        latent_path: str,
        raw_data_dir: str,
        split: str = "train",
        clamp_max: float = 10.0,
        t_map: dict | None = None,
        rollout_timepoints: list[str] | None = None,
        verify_alignment: bool = True,
        alignment_check_num_samples: int = 16,
        include_x_rollout_first: bool = True,
        include_full_x_rollout: bool = False,
    ) -> None: ...
```

实现约束：

- `verify_alignment` 在 `v1` 中必须保持为 `True`
- 不允许提供“跳过对齐检查后继续训练”的运行模式
- `alignment_check_num_samples` 在 `v1` 中必须为正整数
- 训练集默认 `include_x_rollout_first=True`、`include_full_x_rollout=False`
- 验证集可开启 `include_full_x_rollout=True`

`__getitem__` 返回字段：

```python
{
    "z_src": FloatTensor[768, 14, 14],
    "z_dst": FloatTensor[768, 14, 14],
    "x_src": FloatTensor[1, 192, 192],
    "x_dst": FloatTensor[1, 192, 192],
    "t_src": FloatTensor[],
    "t_dst": FloatTensor[],
    "hop_idx": LongTensor[],
    "pair_idx": LongTensor[],
    "slice_idx": LongTensor[],
    "z_rollout": FloatTensor[5, 768, 14, 14],
    "x_rollout_first": FloatTensor[1, 192, 192],   # optional
    "x_rollout": FloatTensor[5, 1, 192, 192],      # optional, val/vis only
}
```

补充字段约束：

- `x_src/x_dst` 对所有 hop 都返回，trainer 只在 `hop_idx == 0` 使用图像辅助分支
- `x_rollout_first` 若返回，则必须对应 `D50`
- `x_rollout` 若返回，则 `x_rollout[0]` 对应 `D50`、`x_rollout[1]` 对应 `D20`
- 实现时除顺序校验外，还必须做少量抽样内容一致性检查
- 抽样内容检查至少比较 `Crop192(Dec(z_gt))` 与对应 raw image 是否大体匹配
- 若抽样内容一致性检查显示系统性错位，也必须 fail-fast

### 2.2 主模型模块

新增文件：

- `pet_lr/model_first_hop.py`

新增类：

- `FirstHopPixelEncoder`
- `HopResidualVelocityHead`
- `PETFlowDiTFirstHop`

#### `FirstHopPixelEncoder`

职责：

- 只处理 `x_D50`
- 只提供第一跳的弱局部结构提示
- 不承担 image translation 职责

结构约束：

- 输入：`[B, 1, 192, 192]`
- 浅层 CNN stem
- `AdaptiveAvgPool2d((14, 14))`
- `1x1 conv` 或 `linear proj` 投到 encoder hidden size
- 输出：`f_pix in [B, N=196, D_enc]`
- 不使用全局 attention
- 不使用 self-attention
- 不使用 cross-attention
- 不使用深 U-Net
- 参数量不得超过 shared trunk 的 `5%`

原因：

- 现有 latent token lattice 为 `14x14`
- `192` 与 `14` 不是严格 patch 对齐关系
- 第一版只要求弱对齐条件，不要求 pixel token 与 latent token 严格一一对应

#### `HopResidualVelocityHead`

职责：

- 在共享输出之外提供极小的 hop-specific residual correction

结构约束：

- 只看 trunk 最后一层特征
- 不重新编码时间和图像条件
- 每个 hop 一个很小的 residual head
- 输出层零初始化
- 增加可学习缩放系数 `lambda_hop`
- `lambda_hop` 初始值为 `0`

输出形式：

```python
v_pred = v_shared + lambda_hop[hop_idx] * v_hop
```

工程约束：

- 单个 hop head 参数量不超过 shared output head 的 `10%`
- 不允许每个 hop 复制一整套 trunk

#### `PETFlowDiTFirstHop`

职责：

- 复用现有 `PETFlowDiTDHHopAware` 的 shared trunk 与时间条件
- 在 hop 0 时额外接收 pixel 条件
- 在所有 hop 上支持 hop residual head

建议初始化方式：

- 从已有 hop-aware checkpoint 加载 shared trunk / shared head
- 新增卷积/投影层保持零初始化
- `g_pix` / `lambda_hop` 的门控标量允许为 `0` 或极小正值，但必须由配置显式给出并可审计

规范接口：

```python
class PETFlowDiTFirstHop(nn.Module):
    def predict_latent_step(
        self,
        z_src: torch.Tensor,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
        hop_idx: torch.Tensor,
        x_src_img: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]: ...
```

前向规则：

- `hop_idx != 0` 时忽略 `x_src_img`
- `hop_idx == 0` 时允许 `x_src_img`
- pixel 特征只注入 encoder 前 `2~4` 层
- 注入门控 `g_pix` 初值必须为 `0` 或极小正值，且由配置显式声明

注入形式：

```python
h = h + g_pix * f_pix
```

其中：

- `g_pix` 可以是标量门或通道门
- 不在 decoder side 注入
- 不在每一层重复注入

实现约束：

- `predict_latent_step(...)` 是 pair、rollout、inference 共用的唯一 step-level 入口
- 若保留 `forward(...)`，它必须是 `predict_latent_step(...)` 的薄包装，不允许复制 hop0 逻辑
- hop0 的 pixel gating、hop residual、latent update 不允许在 trainer、rollout helper、sampler 各写一套

### 2.3 损失模块

复用文件：

- `pet_lr/losses.py`

复用函数：

- `weighted_l1_loss`
- `ssim_loss`
- `seam_consistency_loss`
- `make_border_weight_map`

新增文件：

- `pet_lr/losses_first_hop.py`

新增函数：

- `compute_first_hop_image_loss(...)`

职责：

- 只对 hop 0 样本计算图像辅助损失
- 聚合 `L1 + SSIM + seam`
- 支持 border-aware weighting

### 2.4 Rollout / Sampler 模块

新增文件：

- `pet_lr/rollout_first_hop.py`

新增函数：

- `predict_next_train_first_hop(...)`
- `rollout_multistep_losses_first_hop(...)`
- `sample_one_step_first_hop(...)`
- `sample_chain_first_hop(...)`

职责：

- 保留 latent-only rollout
- 在 rollout 第一步允许传入 `x_D50`
- 后续三步严格只传 `z_curr`
- rollout-step-0 必须调用模型统一的 `predict_latent_step(...)`

建议接口：

```python
def sample_one_step_first_hop(
    model,
    z_start: torch.Tensor,
    t_start: float,
    t_end: float,
    hop_idx: int,
    x_src_img: torch.Tensor | None = None,
) -> torch.Tensor: ...
```

```python
def sample_chain_first_hop(
    model,
    z_d50: torch.Tensor,
    x_d50: torch.Tensor,
    rollout_times: list[float],
) -> list[torch.Tensor]: ...
```

推理规则：

- hop 0：`z_D20^pred = F(z_D50, x_D50, t_src, t_dst, hop=0)`
- hops 1~3：`z_next^pred = F(z_curr, t_src, t_dst, hop)`
- `x_D50` 只作为第一跳外部条件，不形成新的内部 state
- inference-step-0 不允许实现独立的 hop0 门控分支；必须复用训练时的统一 step helper

### 2.5 训练入口

新增文件：

- `train_first_hop.py`
- `configs/pet_flow/pet_flow_first_hop_v1.yaml`

职责：

- 加载 first-hop 专用数据集与模型
- 从现有 hop-aware 4-hop checkpoint warm start
- 同时训练 latent 主损失与 hop0 图像辅助损失
- 保留现有 straight-through mixing

## 3. 接口规范

### 3.1 Batch 接口

`v1` 训练固定使用两个 batch：

- `main_batch`
- `hop0_aux_batch`

`main_batch` 字段：

- `z_src`
- `z_dst`
- `t_src`
- `t_dst`
- `hop_idx`
- `z_rollout`
- `x_rollout_first` 可选

`hop0_aux_batch` 字段：

- `z_src`
- `z_dst`
- `x_src`
- `x_dst`
- `t_src`
- `t_dst`
- `hop_idx`

使用规则：

- pair loss：只使用 `main_batch.z_src/z_dst`
- rollout latent loss：只使用 `main_batch.z_rollout`
- rollout 第一步 pixel 条件：优先使用 `main_batch.x_rollout_first`
- hop0 image loss：只使用 `hop0_aux_batch.x_src/x_dst`
- `hop0_aux_batch.hop_idx` 必须全为 `0`

`v1` 优化说明：

- `v1` 训练集默认不返回 full `x_rollout`
- `v1` 训练集最小必要 image 字段为：`hop0_aux_batch.x_src`、`hop0_aux_batch.x_dst`、`main_batch.x_rollout_first`
- full `x_rollout` 优先保留给验证和可视化
- `v1` 默认建议：train split 只保留 hop0 auxiliary 图像字段与可选 `x_rollout_first`
- validation / visualization 再保留 full `x_rollout`

### 3.2 模型接口

训练主接口：

```python
out = model.predict_latent_step(
    z_src=z_src,
    t_src=t_src,
    t_dst=t_dst,
    hop_idx=hop_idx,
    x_src_img=x_src_if_hop0,
)
```

约束：

- 不允许把 `x_dst` 喂回模型前向
- 不允许把 `x_refined` 或其他 image 输出回写到下一跳 latent
- pair-hop0、rollout-step-0、inference-step-0 必须共享这一接口

### 3.3 解码接口

图像辅助监督通过冻结 RAE decoder 计算：

```python
x_pred = Crop192(Dec(z_pred))
```

实现约束：

- decoder 参数冻结
- decode 前向不放进 `torch.no_grad()`
- 允许梯度从 image loss 回到 `z_pred` 和 transport backbone
- decoder 参数必须 `requires_grad=False`
- 允许计算图穿过 decoder，但 decoder 参数不允许接收可训练梯度

## 4. Loss 规范

总损失：

```python
L_total =
    L_pair_latent
    + lambda_roll(step) * L_rollout_latent
    + lambda_img(step) * L_img_hop0
    + lambda_cct(step) * L_cct
```

### 4.1 `L_pair_latent`

对所有 hop 计算：

- velocity loss
- endpoint latent loss

继承当前 4-hop hop-aware MeanFlow 设定：

- `target_normalize`
- `pair_weighting`
- `straight-through mixing`

### 4.2 `L_rollout_latent`

对完整 `D50 -> D20 -> D10 -> D4 -> NORMAL` 路径计算：

- 逐跳 latent endpoint 监督
- 第一步 rollout 可使用 `x_rollout_first` 作为附加条件
- 后三步严格不使用 pixel 条件

### 4.3 `L_cct`

`Idea1` 的核心一致性项：

```python
L_cct =
    mean_hop(
        w_cct(hop) * D(
            z_pred_pure_pred(hop),
            stopgrad(z_pred_teacher_forced(hop))
        )
    )
```

约束：

- teacher-forced 路径与 pure-pred 路径必须共用统一的 `predict_latent_step(...)`
- hop 0 的两条路径都允许接收同一个 `x_rollout_first`
- 后三跳严格 latent-only
- 默认 `stopgrad(teacher)`，避免双向牵引导致目标漂移
- `D` 第一版只允许 `mse` 或 `l1`

### 4.4 `L_img_hop0`

只对 `hop_idx == 0` 的 pair batch 元素计算：

```python
L_img_hop0 =
    w_l1 * L1_border(x_D20_pred, x_D20_gt)
    + w_ssim * SSIM(x_D20_pred, x_D20_gt)
    + w_seam * Seam(x_D20_pred)
```

约束：

- 只在 `192x192` crop 上计算
- `x_D20_pred = Crop192(Dec(z_D20_pred))`
- `x_D20_gt = x_dst`
- 不使用 `x_D20_gt` 作为模型输入
- decoder 冻结
- `L_img_hop0` 只允许更新 shared trunk、`HopResidualVelocityHead`、`FirstHopPixelEncoder`、`g_pix`、`lambda_hop`

### 4.5 权重与调度

必须提供三个 warmup：

- `lambda_roll(step)`
- `lambda_img(step)`
- `lambda_cct(step)`

默认策略：

- `lambda_roll` 复用现有 rollout warmup/ramp
- `lambda_img` 从 `0` warm up 到小值
- `lambda_cct` 从 `0` warm up 到小值

默认建议：

- `lambda_img_max` 小于 latent 主损失同量级
- 第一版只允许 `0.01 ~ 0.10` 范围内搜索

### 4.6 可选正则

仅作为可选项：

- `L_hop_residual = ||lambda_hop||^2`
- `L_pix_gate = ||g_pix||^2`

默认：

- 先不开启
- 只有出现 hop head 抢主导时再加入

## 5. Trainer 改动

### 5.1 初始化

训练前执行：

1. 加载现有 hop-aware 4-hop checkpoint
2. 载入 shared trunk / shared head
3. 新增 `FirstHopPixelEncoder` 零初始化
4. 新增 `HopResidualVelocityHead` 零初始化
5. `lambda_hop` 和 `g_pix` 初值设为 `0` 或极小正值，并在配置中显式声明
6. 显式断言所有 decoder 参数 `requires_grad=False`
7. 运行并记录 latent / image 对齐校验摘要
8. 运行并记录抽样内容一致性检查摘要
9. 若 `decoded_vs_raw_l1` 或 `decoded_vs_raw_corr` 出现系统性异常，则直接中止训练

### 5.2 DataLoader

保留现有按 hop 展平的组织方式，但 `v1` 固定采用双 loader：

- `main_loader`：自然 hop 分布 batch，用于 latent 主损失
- `hop0_loader`：只采样 hop0 的 auxiliary mini-batch，用于 hop0 image supervision 与 hop0 监控

原因：

- hop0 专属模块只在约 `25%` 样本上收到直接监督

硬要求：

- 第一版必须保证每个 training step 都有 `hop0_loader` 提供的 hop0 子批次
- 不再把“平均覆盖率足够”视为可接受实现
- 若 `hop0_loader` 为空、耗尽或未提供 hop0 子批次，trainer 必须 fail-fast

固定实现：

- `main_loader` 负责 `L_pair_latent + L_rollout_latent`
- `hop0_loader` 负责 `L_img_hop0`，必要时也可记录 hop0 latent 指标
- 每个 step 同时消费一个 main batch 和一个 hop0 auxiliary batch

默认建议：

- `hop0_loader` batch size 建议不超过 main batch 的 `25% ~ 50%`
- `hop0_coverage` 目标区间写为 `0.25 ~ 0.50`
- `hop0_coverage = hop0_aux_batch_size / main_batch_size`
- `hop0_main_ratio = mean(main_batch.hop_idx == 0)` 仅作为自然分布观测量，不参与 fail-fast 判定
- `v1` 起始默认值：`hop0_aux_batch_size = ceil(0.25 * main_batch_size)`
- 验证集保持自然分布

### 5.3 Train Step

训练步顺序：

1. 从 `main_loader` 取 natural-distribution batch
2. 从 `hop0_loader` 取 hop0 auxiliary batch
3. 断言 auxiliary batch 全部为 hop0
4. 用 `main_loader` batch 通过统一 `predict_latent_step(...)` 计算 `L_pair_latent`
5. 用 `main_loader` batch 通过统一 step helper 计算 `L_rollout_latent`
6. 用同一 `main_loader` batch 计算 teacher-forced vs pure-pred 的 `L_cct`
7. 只 decode `hop0_loader` 子批次的 `z_D20_pred`
8. 用 hop0 auxiliary batch 计算 `L_img_hop0`
9. 聚合总损失
10. 反向传播

需要记录的指标：

- `loss_pair`
- `loss_velocity`
- `loss_endpoint`
- `loss_rollout_total`
- `loss_rollout_step_0/1/2/3`
- `loss_cct`
- `loss_cct_step_0/1/2/3`
- `lambda_cct`
- `loss_img_hop0`
- `loss_img_l1`
- `loss_img_ssim`
- `loss_img_seam`
- `hop0_coverage`
- `hop0_main_ratio`
- `gate_pix`
- `lambda_hop_0/1/2/3`

### 5.4 Validate

验证必须同时输出四组结果：

- latent pair / rollout 指标
- CCT 一致性指标
- hop0 image auxiliary 指标
- 全链路采样后的 decode 指标

全链路验证规则：

- 输入 `z_D50` 与 `x_D50`
- 第一步用 first-hop 条件
- 后续三步 latent-only
- decode `D20/D10/D4/NORMAL`

fail-fast 指标：

- 若 `hop0_coverage` 连续 `100` 个 step 落在目标区间外，则中止训练并检查 loader
- 若 `loss_img_hop0` 长时间为 `0` 或 `NaN`，则中止训练并检查 hop0 auxiliary 路径
- 若 `gate_pix` 或 `lambda_hop_0` 在 warmup 早期出现异常爆炸，也应中止训练

### 5.5 Checkpoint

checkpoint 额外保存：

- `target_normalize`
- `rollout_path`
- `first_hop_pixel_enabled`
- `state_dict` 中新增模块参数

## 6. Sampler 改动

现有 sampler 不足：

- 只支持 latent-only 输入
- 无法在 hop0 传入 `x_D50`

因此需要新增 first-hop sampler 副本。

采样规则：

```python
z_d20 = sample_one_step_first_hop(model, z_d50, 2.0, 5.0, hop_idx=0, x_src_img=x_d50)
z_d10 = sample_one_step_first_hop(model, z_d20, 5.0, 10.0, hop_idx=1)
z_d4  = sample_one_step_first_hop(model, z_d10, 10.0, 25.0, hop_idx=2)
z_n   = sample_one_step_first_hop(model, z_d4, 25.0, 100.0, hop_idx=3)
```

硬约束：

- sampler 内部不缓存 pixel state
- sampler 返回值始终只有 latent
- sampler-step-0 必须复用 `predict_latent_step(...)`

## 7. 实验顺序

### E0. 数据与退化检查

- 完成 latent/image slice 对齐检查
- 验证新增模型在 `g_pix=0`、`lambda_hop=0` 时严格退化为现有 hop-aware baseline

停止条件：

- 退化结果与 baseline 不一致则停止

`v1` 退化一致性容差：

- 必须在 `eval + fp32 + fixed batch` 条件下检查
- `max_abs_diff(v_pred)` 必须 `<= 1e-6`
- `max_abs_diff(z_pred)` 必须 `<= 1e-6`
- 若超过该容差，优先视为实现错误而不是训练问题

### E1. 只加 hop residual head

- 开启 `HopResidualVelocityHead`
- 关闭 pixel conditioning
- 关闭 hop0 image loss

目标：

- 判断 hop specialization 是否单独带来第一跳收益

### E2. 再加 hop0 pixel conditioning

- 开启 `FirstHopPixelEncoder`
- `g_pix` 从 `0` 或极小正值开始学习，并在配置中显式声明
- 仍关闭 hop0 image loss

目标：

- 只评估 source-pixel 条件是否改善第一跳 latent update

### E3. 最后加 hop0 image auxiliary loss

- 开启 `L_img_hop0`
- decoder 冻结
- `lambda_img` warmup

目标：

- 把第一跳 decode 结构绑住
- 检查是否改善第一跳后续误差积累

### E4. 加入 CCT-224

- 开启 `L_cct`
- 显式记录 train/val 的 `cct_step_0/1/2/3`
- 只允许在不改变 rollout state 的前提下增加 consistency 路径

目标：

- 对齐 teacher-forced 与 pure-pred 的分布
- 降低部署时 pure-pred 级联漂移

### E5. 采样与权重消融

只在 E4 成立后做：

- hop0 上采样强度
- `lambda_img_max`
- `lambda_cct_max`
- 注入层数 `2/3/4`
- `g_pix` 标量门 vs 通道门

## 8. 不在本规范内的路线

以下路线不属于本实施规范：

- `224 clean-geometry + decoder adaptor` 的 v2.1 路线
- `segmented latent forcing + stitching`
- image-token 主轨迹
- dual-state transport
