# image_aux 仅第一跳的架构分析（20260516）

> **⚠️ 20260516 PEER REVIEW CORRECTION**：本文件 §3 / §4 中“image_aux 沿 cascade 单调放大”与 §10 "D20 是真瓶颈" 的论证**部分依赖错误标注的 V7 − V8 对照**。V7 与 V8 同时改变了 image_aux 和 step_weights（见 [PLANF_FINAL_ANALYSIS_20260516.md 顶部警示](./PLANF_FINAL_ANALYSIS_20260516.md)），**不能被当作纯 image_aux ablation 使用**。
>
> 以下仍然成立：
> - §1、§2、§8 的代码事实（hop0-only 实现位置、数据可扩展性）与代码、数据集结构紧紧相关。
> - §10.3 V11a/b/c/d 子候选本身是可能的低成本动作，但优先级需要在 V13 (true image_aux ablation) 结果出来后重新评估。
> - V11 撚回仍然有效，但撚回的逻辑依据也被错标注污染。
>
> **补救实验**：见 [REVIEW_INTEGRATION_20260516.md §3](./REVIEW_INTEGRATION_20260516.md)。

---

- generated_at: 2026-05-16 Asia/Shanghai
- branch: foc_lite_hop0
- commit_when_generated: acc7a9d
- 触发问题：为提高图像效果，第一跳引入了 `image_aux`；为什么其他跳没有？这设计合理吗？
- 关键前置文献：[review/0430/reviewer/literature_architecture_image_aux_review_20260501.md](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)（已对 image_aux 做过完整文献 + 实现审查）
- 关键实证：[review/0516/PLANF_FINAL_ANALYSIS_20260516.md §4](./PLANF_FINAL_ANALYSIS_20260516.md)（V7 vs V8 在 cascade 上单调放大的 PSNR delta）

> **2026-05-16 修订说明**：本文件初始版本在 §5-7 推荐了 V11 (full-hop image_aux)，理由是“NORMAL PSNR 受益最大”。**该推荐根据同一批数据重新检查后被撚回**，原因是该推理混淆了“间接传播收益”与“未挖掘监督空间”。正确结论见 §10。本文件的 §1-§4 （代码事实、数据可扩展性、cascade 单调上升的实证）仍然成立；结论章节被 §10 覆盖。

---

## 1. 代码事实（image_aux 真的只在 hop0）

通读 `train_first_hop.py` + `pet_lr/` 后的确认事项：

### 1.1 三个 hop0-only 机制是分离的

| 机制 | 文件:行 | 实际作用 |
|---|---|---|
| **Pixel forcing**（推理时把 D50 image 加到 z_D50 上） | [model_first_hop.py:444-476](../../pet_lr/model_first_hop.py#L444) | `_apply_hop0_pixel_forcing` 用 `hop_idx==0` mask；其他 hop 直接 `return z_src` |
| **FirstHopPixelEncoder**（轻量像素 → latent encoder） | [model_first_hop.py:30-66](../../pet_lr/model_first_hop.py#L30) | 与 backbone 容量比 ≤ 5%（`pixel_encoder_max_ratio` 硬卡），输出经 `gate_pix` 软门控 |
| **image_aux loss**（解码 z_pred 再算 L1+SSIM+seam） | [train_first_hop.py:786-825](../../train_first_hop.py#L786) | `compute_hop0_image_losses` 调 `model.predict_latent_step` 一次（hop_idx 来自 batch，但 batch 是 hop0-only） |

### 1.2 image_aux 的数据来源是严格的 hop0 batch

- 主 batch 来自 `PETFirstHopAligned4HopDataset`，混合 4 个 hop。
- image_aux 走的是另一条独立 dataloader：[pet_lr/data_first_hop.py:396-411 `Hop0OnlyViewDataset`](../../pet_lr/data_first_hop.py#L396)。
  > "Dataset view that only keeps hop0 (D50->D20) samples."
  
  按 `pair_idx==0` 索引筛选，物理上不可能采到非 hop0。
- 因此 `hop0_batch["hop_idx"]` 一定全是 0，`compute_hop0_image_losses` 内部的 `predict_latent_step` 必然走 pixel_forcing 路径，输出再 decode → 与 GT `x_D20` 比较。

### 1.3 rollout chain 也只有 hop0 输入像素

[pet_lr/rollout_first_hop.py:89](../../pet_lr/rollout_first_hop.py#L89)：

```python
x_src_img = x_rollout_first if hop_idx == 0 else None
```

[pet_lr/rollout_first_hop.py:159](../../pet_lr/rollout_first_hop.py#L159)（采样函数 `sample_chain_first_hop`）：

```python
x_src = x_d50 if hop_idx == 0 else None
```

后续跳 `x_src_img=None`，`_apply_hop0_pixel_forcing` 在 `x_src_img is None` 时直接 `return z_src`。

**结论**：架构在三个地方一致地把"图像信号注入"限定在 hop0，无任何 hop1/2/3 的等价机制。

---

## 2. 数据层能否支持扩展到后续跳？

可以。

- `PETFirstHopAligned4HopDataset` 已有 `include_full_x_rollout: bool` 开关（[data_first_hop.py:384-395](../../pet_lr/data_first_hop.py#L384)）。
- V7/V8/V6_NOISE 的 yaml 都已开启 `val_include_full_x_rollout: true`（验证阶段加载全部 5 个 timepoint 的图像）。
- 训练阶段是 `train_include_full_x_rollout: false`——**这是当前唯一阻碍 hop1/2/3 用 image_aux 的开关**，不是数据集问题。

也就是说：**hop1/2/3 image_aux 的数据已经准备好了**。是不是已经实现的训练逻辑没用它。

---

## 3. 设计依据（基于代码 + 历史 review 的真实理由）

### 3.1 三层"为什么"理由

**(a) Pixel forcing 的 hop0-only 是因果约束，不是工程选择**：
- hop0 输入 z_D50 来自 RAE encoder 对 *真实 PET 图像 x_D50* 的编码——这张图是已知的，可以作为 side-info 重新喂给模型。
- hop1 输入 z_D20 是模型上一跳的*预测*，对应的"图像"`x_D20_pred = decode(z_D20_pred)` 还要解码一次。如果我们仍用 GT `x_D20` 注入，就是把答案直接告诉模型 → 推理时无此信号 → train/test mismatch。
- 所以 pixel forcing 只能在 hop0：那是唯一一个"输入图像"和"GT 输入图像"同源的位置。

**(b) image_aux loss 的 hop0-only 是历史与成本选择，不是因果约束**：
- image_aux loss 用的是 **解码 z_pred → 与 GT x_dst 比较**。这不需要"注入像素"，只需要在 forward 后多跑一次 decoder。
- hop1 image_aux：decode(z_D10_pred) ↔ GT `x_D10`，无信息泄露，物理可行。
- hop2/3 同理。NORMAL 那一跳尤其关键——它是我们最终临床关心的目标。
- **历史原因**：[review/0430/reviewer §1.3](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) 已明确指出："image_aux is hop0-only, low-weight, low-level, and by default its seam term is prediction self-continuity rather than GT seam alignment"。
- **成本原因**：每跳 image_aux 要多调一次 RAE decoder（4× cost），且 `train_include_full_x_rollout: false` 时数据 batch 也得改。

**(c) Pair loss 已经多跳监督**：
- `compute_pair_losses` 在所有 hop 上算 velocity + endpoint loss。
- 所以"多跳监督"并非缺失，缺失的是 *像素域* 的多跳监督。pair loss 是 *latent 域* 的。
- 这一区分很关键：latent 域 MSE 不等价于像素域 PSNR，原因是 RAE decoder 是非等距的。

### 3.2 cascade 实证（来自 [review/0516/PLANF_FINAL_ANALYSIS_20260516.md §4](./PLANF_FINAL_ANALYSIS_20260516.md)）

image_aux 的 paired-PSNR delta（V7 − V8，7403 切片）沿链单调放大：

| stage | ΔPSNR | 解读 |
|---|---:|---|
| D20 | +0.226 dB | hop0 的直接监督效果 |
| D10 | +0.230 dB | 间接（chain coherence 传递） |
| D4 | +0.277 dB | 间接（累积放大） |
| NORMAL | +0.308 dB | 间接（最大累积） |

**反直觉但关键**：hop0 image_aux 的最大 PSNR 受益不在 hop0（D20），而在 NORMAL。这是因为 hop0 学得好 → 后续每跳起点更准 → NORMAL 累积误差更小。

**所以"为什么 NORMAL 没有 image_aux 反而 NORMAL 收益最大"**？答案是：当前 NORMAL 的 PSNR 改善是 **间接传播**而非 **直接监督**。直接监督理论上应该更强。这就是改造空间。

---

## 4. 架构是否合理 — 判定

### 4.1 当前设计的合理之处

1. **Pixel forcing 的 hop0-only 是正确的**——不能扩展，扩展就泄露答案。
2. **FirstHopPixelEncoder 容量受控**（5% backbone hard cap）——避免该路径吞掉模型容量。
3. **gate_pix 软门控**——允许模型学到合适注入强度，初始接近 0 避免破坏 backbone 训练。
4. **image_aux loss 与 pair loss / rollout loss 分离**——多目标解耦，符合 [STEP_WEIGHTS_THEORY_REFERENCE.md §1-§2](./STEP_WEIGHTS_THEORY_REFERENCE.md) 的多 hop 加权目标框架。

### 4.2 当前设计的不合理 / 待改进之处

| 问题 | 严重度 | 处置 |
|---|---|---|
| image_aux loss 仅 hop0 监督 NORMAL 通过 chain 间接传播 | **中** | 见 §5.A |
| seam 项默认 `use_extended_seam: false`，只是 prediction self-continuity，不与 GT seam 对齐 | 中 | 见 §5.B（[0501 review §6](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)）|
| `lambda_max=0.04` 让 image_aux 占总 loss 5-15%（[ARCHITECTURE_ANALYSIS §20.1](../plan/ARCHITECTURE_ANALYSIS_20260501.md)），rollout 仅 0.1-3%，可能并未充分利用 | 中 | 在 V9 之后做权重 sweep；[V9_PREREGISTRATION.md](./V9_normal_emphasis/V9_PREREGISTRATION.md) 已隐式覆盖 |
| 复杂高 SUV 区域无 ROI-PSNR 优化（只全图 L1+SSIM）| 高（按 [0501 review §3](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)）| 见 §5.C |
| 无 LPIPS / perceptual loss——结构细节可能被 over-smoothed | 中 | 见 §5.D |

---

## 5. 可执行的扩展路径

按"先量化 → 再改训练"排序（与 [0501 review §7](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) 推荐顺序一致）：

### 5.A 把 image_aux 扩展到所有 hop（最直接，物理上无障碍）

**代码改造点**：
1. yaml: `train_include_full_x_rollout: true` 加载所有 hop 的 GT image。
2. `rollout_first_hop.py:rollout_multistep_losses_first_hop` 收集每个 step 的 `z_pred`，新增 `decode → image_loss(x_pred, x_rollout[:, k+1])`。
3. 把每跳 image_aux 用与 step_weights 同源的权重（§5.A.1）注入 total_loss。

**理论权重**（与 [STEP_WEIGHTS_THEORY_REFERENCE.md §2.2](./STEP_WEIGHTS_THEORY_REFERENCE.md) 一致）：

$$
\lambda_\text{img}^{(j)} \;=\; \lambda_\text{img,base} \cdot \frac{w_j^{*,\text{multi-hop}}}{\sum_i w_i^{*,\text{multi-hop}}}
$$

每跳 image_aux 与 step_weights 同形，避免引入新的 heuristic。

**成本**：每 step 多 K=4 次 RAE decode。RAE decoder 是 ~50M 参数级 transformer，本身 forward 时间约 30% 整 step → 总训练时间增 ~30-50%。对 160K step 是约 +2-3 天。

**预期收益**：当前 hop0 image_aux 通过 chain 间接给 NORMAL 带来 +0.308 dB。直接 NORMAL image_aux 的预期上限至少应等量，且与现有 NORMAL chain MSE supervision 互补。**这是最高优先级的架构扩展**。

### 5.B 启用 `use_extended_seam=true`（最便宜）

V6/V7/V8 默认 `use_extended_seam: false`，seam 项只是预测自洽性，不与 GT 对齐。开启 extended seam 是一行 yaml 改动，把 seam 信号从"平滑"升级为"对齐 GT"。

不需要架构改造，可与 §5.A 同时做。但优先级低于 §5.A，因为 V3 视觉上没有明显 block artifact（[0501 review §1.4](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)）。

### 5.C 高 SUV / 复杂区域 ROI-PSNR 与 ROI-weighted loss

[0501 review §3](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) 已列出指标：top-10%/5%/1% SUV mask、high-gradient mask、SUVmax/SUVmean、SBR/CBR/SNR/CNR。

**评估优先**：先在现有 V7/V8/V6_NOISE checkpoint 上跑 ROI-PSNR 面板。如果 ROI-PSNR 差距比全图大，说明 image_aux 没有 attend 到复杂区域 → 验证 §5.D 必要性。

**训练改造**：pair-stage 用 `border_weight`/ROI mask 重新加权 image_aux 的 L1。比 §5.A 便宜，但定向窄。

### 5.D LPIPS / 感知损失

加 LPIPS pullback 到 `image_aux`（[0501 review §7.2 step 4](../0430/reviewer/literature_architecture_image_aux_review_20260501.md)）。这是 over-smoothing 问题的标准处方，但需要额外 VGG/AlexNet 网络在 224×224 上 forward，成本与 §5.A 接近。

**触发条件**：仅在 §5.C ROI-PSNR 显示复杂区域系统性 over-smoothing 时启动。

### 5.E 多跳 stop-grad pixel consistency

[0501 review §5](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) 提到的"roll-stage stop-grad multi-hop pixel consistency"：在 rollout 阶段对中间 hop 解码，stop-grad chain input，只优化当前 hop 的解码 vs GT。这是 §5.A 的"安全保守版"——避免直接监督导致 chain gradient 不稳。

**优先级**：在 §5.A 出现 chain 训练不稳定时回退到此方案。

---

## 6. 与正在做的 V9 的关系

V9（β_NORMAL: 1.5 → 2.5，[V9_PREREGISTRATION.md](./V9_normal_emphasis/V9_PREREGISTRATION.md)）改的是 *latent 域多 hop 加权* 的目标权重。它和 §5.A（*像素域多 hop 监督*）是 **正交的两个轴**：

| 实验 | 操作的层 | 操作的轴 |
|---|---|---|
| V9 | latent 域 | 多 hop selector β |
| §5.A | 像素域 | hop coverage（hop0-only → 全 hop） |

**建议串行**：
1. 先做 [V9_PREREGISTRATION.md §0](./V9_normal_emphasis/V9_PREREGISTRATION.md) 的 Lipschitz pre-launch gate（10 分钟）→ 确认 V7 闭式解前提。
2. 然后并行启动 V9（β_NORMAL=2.5）与 V11 ≡ §5.A 全 hop image_aux（同 V7 baseline）。
3. 两者独立可比较：
   - V9 vs V7 → 验证 β_NORMAL 灵敏度
   - V11 vs V7 → 验证像素域多 hop 监督
   - V11 + β_NORMAL=2.5（V12）→ 双轴叠加

每组都用 [PLANF_FINAL_ANALYSIS §2.2](./PLANF_FINAL_ANALYSIS_20260516.md) 的 paired t-test + per-slice win rate 框架评估。

---

## 7. 一句话答复

> Pixel forcing 限定 hop0 是因果上不能扩展（hop1+ 没有同源真实图像可注入而不泄露答案）。但 image_aux loss 限定 hop0 **不是**因果约束——它只是历史与成本上的选择。当前 NORMAL 的 PSNR 收益完全靠 chain coherence 间接传播；直接在 hop3 加 image_aux loss 是合理的下一步架构扩展，命名为 **V11 (full-hop image_aux)**。它与 V9（β-axis）正交，可在 V9 launch 前后并行准备。

---

## 8. 参考文件索引

| 文件 | 提供的关键内容 |
|---|---|
| [pet_lr/model_first_hop.py](../../pet_lr/model_first_hop.py) | `FirstHopPixelEncoder`, `_apply_hop0_pixel_forcing`, gate/lambda_hop 软门控 |
| [pet_lr/rollout_first_hop.py](../../pet_lr/rollout_first_hop.py) | `rollout_multistep_losses_first_hop`（多跳 chain forward），`sample_chain_first_hop`（推理） |
| [pet_lr/losses_first_hop.py](../../pet_lr/losses_first_hop.py) | `compute_first_hop_image_loss`（L1+SSIM+seam 组合 + border_weight）|
| [pet_lr/data_first_hop.py](../../pet_lr/data_first_hop.py) | `PETFirstHopAligned4HopDataset` (`include_full_x_rollout`), `Hop0OnlyViewDataset` |
| [train_first_hop.py](../../train_first_hop.py) | `compute_hop0_image_losses` (line 786), `compute_pair_losses`, total_loss 组合 |
| [review/0430/reviewer/literature_architecture_image_aux_review_20260501.md](../0430/reviewer/literature_architecture_image_aux_review_20260501.md) | 2024+ PET 文献调研 + image_aux 现状审查 + ROI 指标建议 |
| [review/plan/ARCHITECTURE_ANALYSIS_20260501.md](../plan/ARCHITECTURE_ANALYSIS_20260501.md) §20.1 | total_loss 公式展开（image_aux 占总 loss 5-15%）|
| [review/0516/PLANF_FINAL_ANALYSIS_20260516.md §4](./PLANF_FINAL_ANALYSIS_20260516.md) | image_aux PSNR delta 沿 cascade 单调放大的实证 |
| [review/0516/STEP_WEIGHTS_THEORY_REFERENCE.md](./STEP_WEIGHTS_THEORY_REFERENCE.md) | 多 hop Grönwall 闭式解（§5.A 的权重公式来源）|

---

## 10. 修订 — V11 (full-hop image_aux) 撚回，改为 V11' (强化 hop0)

### 10.1 触发

用户反驳调起同一批全 val 数据重检后发现：

| 阶段 | chain PSNR_clip3 (V7 last, n=7403) | chain MSE |
|---|---:|---:|
| **D20** | **35.44 dB**（最低） | **3.30e-4**（最高）|
| D10 | 35.82 | 2.96e-4 |
| D4 | 36.37 | 2.63e-4 |
| NORMAL | 36.78（最高） | 2.46e-4（最低）|

事实纠正：

1. **“误差累计导致 PSNR 下降”在数据上不成立**。链 PSNR 单调上升、链 MSE 单调下降。原因是 v_std 沿链衰减 68×（hop0=0.00963 vs hop3=0.000141）比误差累计衰减快得多。
2. **“D4 和 NORMAL 已经非常接近”部分成立**：ΔPSNR(NORMAL-D4) = +0.41 dB（NORMAL 反而比 D4 更好）。NORMAL 是“最容易”的阶段。
3. **真正的瓶颈是 D20**（链起点），不是 NORMAL。

### 10.2 撚回原 V11 建议的逻辑漏洞

原论点：“NORMAL 从 image_aux 换来 +0.308 dB 收益，直接在 NORMAL 加 image_aux 预期更好。”

逻辑 bug：

- NORMAL +0.308 dB **是 hop0 被监督后通过 chain coherence 传播过来的，不是 NORMAL 本身留了未挖掘的监督空间**。
- NORMAL 已经是 PSNR 最高、MSE 最低、v_std 最小的阶段。加 image_aux 没多少 learning signal 可拿。
- v_std=0.00014 上加 image_aux，梯度量级会被 latent loss 淹没；装不住需要把 λ 改到很高，又会扭曲其他 loss 平衡。

### 10.3 V11'：正确方向 — 强化 hop0 image_aux

既然 D20 是真正瓶颈且 chain coherence 会把 hop0 改善放大到 NORMAL，正确动作是把资源集中在 hop0：

| 候选 | 理由 | 估计成本 |
|---|---|---|
| **V11a：提高 hop0 image_aux 权重** （`lambda_max` 0.04 → 0.08 或 0.12）| 现 image_aux 只占总 loss 5-15%，上调提供更多像素梯度 | 一行 yaml，0 代码 |
| **V11b：启用 extended seam + GT 对齐** （`use_extended_seam: true`）| 当前 seam 只是预测自洽性，extended 版与 GT 梯度/曲率对齐 | 一行 yaml，0 代码 |
| **V11c：hop0 image_aux 高 SUV ROI 加权** （在 `compute_first_hop_image_loss` 加 ROI mask 权重）| 现 border_weight=2 只区分边界，没区分临床重要区域 | ~50 行；losses_first_hop.py |
| **V11d：FirstHopPixelEncoder 容量 5% → 10%** | 现 cap 在 5% backbone；hop0 是瓶颈 → 值得多容量 | 一行 yaml `pixel_encoder_max_ratio` |
| ~~V11：multi-hop image_aux~~ | NORMAL 已饱和；多 4× decoder cost | **撚回** |

### 10.4 修订后与 V9 的关系

V9（β_NORMAL）和 V11'（强化 hop0）不再是“正交两轴”的平衡设计：

- **V9** 把 selector 重量换到 NORMAL，同时通过闭式解调整 step_weights。但如果 NORMAL 已经是 PSNR 最高的阶段，V9 的预期收益也要重新评估——它可能在加重一个已饱和的目标。
- **V11'** 直接加强 hop0 的像素监督，面向真瓶颈。

**严格说两者都需要被 peer review 重新判定**。现阶段 V11' 在数据上依据更充分。

### 10.5 这个错误是怎么产生的（记在这里避免重犯）

原始推理错在：看到 “NORMAL ablation delta 最大” 就跳到 “NORMAL 是重点”。正确读法是：

- *delta* 是两个设置（有/无 image_aux）之间的差。
- delta 在 NORMAL 最大代表的是 “hop0 质量的下游传播限制 NORMAL 的变动范围最大”，不代表 “NORMAL 本身未被充分监督”。
- 判“哪个阶段需要额外监督”要看绝对 PSNR / MSE 水平，不是 delta。

记录：以后看到 “delta 最大” 类说法时，必须同时查看 absolute level 才能下推荐。
