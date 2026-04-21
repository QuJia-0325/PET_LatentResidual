# 五维对抗审计报告 — 2026-04-22

## 审计方法

使用 16 个独立 subagent 对五个维度进行对抗审计：
- 每维度 1 个 **Author**（防守方，通读代码写结构化 defense）
- 每维度 1 个 **Harsh Reviewer**（攻击方，独立读码找 Author 遗漏的问题，3 轮 rebuttal）
- 每维度 1 个 **Area Chair (AC)**（仲裁，验证代码后 SUSTAINED / OVERRULED / PARTIAL）
- 1 个 **Program Chair (PC)** 汇总 5 个 AC 结论

五个维度：**总体架构 / 实验设计 / 代码实现 / 接口调用 / 模型参数**

---

## 问题汇总

| # | AC 来源 | 严重度 | 问题 | 裁决 | 本轮是否修复 |
|---|--------|--------|------|------|-------------|
| 1 | AC5-R1 | **CRITICAL** | `weight_decay=0.01` 作用于标量 gate（g_pix_raw, lambda_hop_raw），AdamW 持续衰减导致 gate 趋向 floor | SUSTAINED | ✅ 已修 |
| 2 | AC5-R2 | **CRITICAL** | `lambda_hop_init=0.01` + `hop_residual_last_init_std=0.002` 双重抑制，hop residual 梯度 ~1e-6 | SUSTAINED | ✅ 已修 |
| 3 | AC3-R1 | **BLOCKER** | alignment loss 通道截断：当 `proj_out > latent_channels` 时 `z_ref[:, :C_out]` 无法扩展 | SUSTAINED | ✅ 已修 |
| 4 | AC1-R2 | **MEDIUM** | `decode_crop(apply_refiner=True)` 被 `self.seam_refiner_enabled` 双重检查拦截，接口参数无效 | SUSTAINED | ✅ 已修 |
| 5 | AC1-R1 | **CRITICAL** | 训练/推理分布不匹配（mix_latent 训练时混入 GT，推理时纯预测） | SUSTAINED | ⏸ 保留讨论 |
| 6 | AC2-R1 | **CRITICAL** | N1 消融设计混淆：同时冻结 encoder + 跳过推理，无法区分两个效应 | SUSTAINED | ⏸ 保留讨论 |
| 7 | AC2-R2 | **HIGH** | val_multi_objective 权重偏向 tail（D20=0.15 vs NORMAL=1.50），压制 hop0 信号 | SUSTAINED | ⏸ 保留讨论 |
| 8 | AC2-R3 | **HIGH** | N2 "stage-2" 实质是从头训练 seam_refiner（Scheme C 无 refiner 权重） | SUSTAINED | ✅ 已修（重命名 standalone） |
| 9 | AC4-R2 | **MEDIUM** | x_src_img 多通道静默截断（无 warning） | PARTIAL | ✅ 已修（加 warning） |
| 10 | AC5-R3 | **MEDIUM** | dead branch 监控 AND 条件过宽 | PARTIAL | ⏸ 不改（init 修复已降低风险） |
| 11 | AC4-R3 | **LOW** | t_src/t_dst 形状未文档化 | PARTIAL | ✅ 已修（docstring + assert） |
| 12 | AC3-R4 | **LOW** | mix_latent STE 实现缺少文档注释 | PARTIAL | ✅ 已修（加 STE 注释） |
| — | AC3-R2 | — | chain metrics 返回 0.0 | **OVERRULED** | — |
| — | AC3-R3 | — | eval 硬编码索引 | **OVERRULED** | — |
| — | AC4-R1 | — | SeamRefiner AttributeError | **OVERRULED** | — |
| — | AC1-R3 | — | pixel gate 死分支（softplus 防止） | **OVERRULED** | — |

---

## 本轮已修复的问题

### Fix 1: weight_decay 应用于标量 gate [AC5-R1, CRITICAL]

**问题分析**：

`train_first_hop.py` 将所有 first-hop 参数（包括 `g_pix_raw`、`lambda_hop_raw` 等标量 gate）
放入同一个 optimizer param group，共享 `first_hop_weight_decay`。当该值为 0.01 时：

```
AdamW 每步: param -= lr * 0.01 * param
```

经过 50K 步训练，gate 参数被持续衰减，最终趋向 `pixel_gate_floor` / `lambda_hop_floor`，
导致 pixel forcing 和 hop residual 分支实质失效。

**修复方案**：在 3 个活跃配置文件的 `optimizer:` 段添加：

```yaml
first_hop_weight_decay: 0.0
```

代码中 `train_first_hop.py:1161` 已有读取逻辑：
```python
first_hop_weight_decay = float(opt_cfg.get("first_hop_weight_decay", global_weight_decay))
```

**修改文件**：
- `configs/pet_flow/pet_flow_first_hop_224_50k_pixenc_ablation.yaml`
- `configs/pet_flow/pet_flow_first_hop_224_50k_imgaux_boost.yaml`
- `configs/pet_flow/pet_flow_first_hop_224_20k_seam_refiner_stage2.yaml`

---

### Fix 2: lambda/head 初始化双重抑制 [AC5-R2, CRITICAL]

**问题分析**：

`HopResidualVelocityHead` 的 last layer 零初始化（std=0.002 ≈ 0），加上
`lambda_hop_init=0.01` 经过 inverse-softplus 后得到极小的 raw 参数。

两级抑制的乘积效应：
```
v_hop = lambda_hop[hop] × head_output
     ≈ 0.01 × ~0 = ~0
∂L/∂head_w ∝ ∂L/∂v_hop × lambda_hop ≈ 0.01 × gradient
```

相比 backbone velocity 的 1.0 级别梯度，hop residual 的有效梯度被压制约 100×，
需要数千步才能激活。

**修复方案**：

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `lambda_hop_init` | 0.01 | **0.10** | 将 hop residual 有效梯度从 ~1% 提升到 ~10% |
| `hop_residual_last_init_std` | 0.002 | **0.01** | 使 last layer 输出非零，加速分支激活 |

**修改文件**：同上 3 个活跃配置文件

---

### Fix 3: alignment loss 通道截断 bug [AC3-R1, BLOCKER]

**问题分析**：

`train_first_hop.py:1717-1718` 原代码：
```python
if z_ref.shape[1] != align_proj.shape[1]:
    z_ref = z_ref[:, :align_proj.shape[1]]
```

当 `align_proj` 通道数 > `z_ref` 通道数时（如 proj_out=1024, latent=768），
`z_ref[:, :1024]` 无法扩展张量——返回原始 768 通道，接下来
`F.mse_loss(align_proj, z_ref)` 因形状不匹配而报错或静默广播。

**修复方案**：在截断前添加明确的断言：

```python
if align_proj.shape[1] > z_ref.shape[1]:
    raise RuntimeError(
        f"alignment projector output channels ({align_proj.shape[1]}) > "
        f"latent channels ({z_ref.shape[1]}); fix alignment.proj_out_channels"
    )
```

**影响范围**：当前 N1/N2 配置均 `alignment.enabled: false`，此 bug 不会触发。
但修复后保证未来启用 alignment 时不会静默失败。

**修改文件**：`train_first_hop.py`

---

### Fix 4: decode_crop 接口契约修复 [AC1-R2, MEDIUM]

**问题分析**：

原 `decode_crop()` 中：
```python
use_refiner = self.seam_refiner_enabled if apply_refiner is None else apply_refiner
if use_refiner and self.seam_refiner_enabled:  # ← 双重检查
```

当调用者显式传入 `apply_refiner=True` 但 `seam_refiner_enabled=False` 时，
第二个条件 `self.seam_refiner_enabled` 仍为 False → refiner 被跳过。
这使得 `apply_refiner` 参数在 override 场景下无效，违反接口契约。

**修复方案**：

```python
use_refiner = self.seam_refiner_enabled if apply_refiner is None else apply_refiner
if use_refiner and hasattr(self, "seam_refiner"):  # ← 检查模块是否存在
```

- `hasattr` 检查确保模块确实被实例化（安全）
- `apply_refiner=True` 现在能正确覆盖 module 默认值
- `apply_refiner=False` 仍然跳过 refiner

**修改文件**：`pet_lr/model_first_hop.py`

---

## 保留讨论的问题（含文献调研结论）

### [AC1-R1] 训练/推理分布不匹配（CRITICAL — 设计决策）

**Reviewer 攻击**：`mix_latent()` 在训练时将 GT 混入 rollout 中间状态，但推理时纯预测。

**AC 裁决**：SUSTAINED，但当前配置已有 `alpha_end: 1.0` + `eval_alpha: 1.0` 缓解：
- 训练后期 alpha→1.0，等同于纯预测
- 验证时 eval_alpha=1.0，与推理一致

**文献支撑**：

| 文献 | 会议/年份 | 关键发现 |
|------|----------|---------|
| **MixFlow Training** (Li et al., arXiv 2512.19311) | 2025 | 专门研究 flow matching 的 exposure bias。**直接在 SiT、REPA、RAE 上实验**（与我们同源架构）。发现训练时用 GT 插值、推理时用预测值的差异是真实问题。提出 "slowed interpolation mixture" 作为后训练修正。RAE 上达到 ImageNet 1.43 FID |
| **Anti-Exposure Bias in Diffusion Models** (Zhang et al.) | **ICLR 2025 Spotlight** | 提出 anti-bias prompt 学习框架缓解 exposure bias。仅增 5% 推理开销。被 AC 评为 "interesting to the general diffusion model community" |

**决议**：✅ **保持不动**。当前 α ramp（scheduled sampling）是标准做法。MixFlow 论文在 RAE 上验证了此类问题存在但可缓解。论文写作时引用 MixFlow 和 Anti-Exposure Bias 作为 related work。如后续 PSNR 提升不足，可考虑 MixFlow 的后训练修正方案。

---

### [AC2-R1] N1 消融设计混淆（CRITICAL — 实验设计）

**Reviewer 攻击**：N1 同时 (a) 冻结 pixel_encoder 和 (b) 跳过 `_apply_hop0_pixel_forcing` 推理，
无法区分 "encoder 没用" 还是 "encoder 没学到"。

**AC 裁决**：SUSTAINED

**背景说明**：
- 我们的 RAE 仓库已有 DINOv2 encoder + LoRA → lightweight ViT-MAE decoder 的完整训练流程
- PET_LatentResidual 中的 pixel_encoder 是 **额外** 添加的轻量 CNN，用于在 hop0 注入像素信息
- N1 的目标是测试"有没有 pixel forcing 路径对 transport 有帮助"
- 由于 encoder 被禁用后不再产生梯度（输出被跳过），冻结是自然的附带效果

**文献支撑**：组件消融通常做的是**整条路径移除**（如 MCAD, MICCAI 2024 中移除整个 CT 条件分支），而非拆分 inference vs training。路径消融是标准做法。

**决议**：✅ **保持当前设计**。N1 测试 "pixel forcing 路径整体贡献"，结论表述为整体消融即可。拆分 3 组消融需额外 2 GPU-day，性价比低。

---

### [AC2-R2] val_multi_objective 权重偏向 tail

**Reviewer 攻击**：D20 权重 0.15 vs NORMAL 1.50，10× 差距压制 hop0 改善信号。

**文献支撑**：

| 文献 | 做法 |
|------|------|
| **MCAD** (Cui et al., MICCAI 2024) | PSNR + SSIM + LPIPS 多指标评估，**不加权合成单一指标** |
| **Mamba-Powered Progressive Network** (Tang et al., MICCAI 2025) | 分剂量级别分别报告 PSNR/SSIM，不做 weighted sum |
| **Generalizable Unpaired PET Enhancement** (Luo et al., TNSRE 2026) | 同上：分任务分别评估 |

主流 PET 重建论文**不做加权多目标 checkpoint selection**，而是分 dose 分别报告指标。

**决议**：✅ **不改权重**（保持已有实验可比性）。评估报告时按 D20/D10/D4/NORMAL 分 hop 列出。N1/N2 额外关注 D20 单独指标作为辅助参考。

---

### [AC2-R3] N2 "stage-2" 实质从头训练

**事实**：Scheme C checkpoint 无 seam_refiner 权重 → N2 resume 后 refiner 随机初始化。

**决议**：✅ **重命名为 "seam_refiner standalone training"**（准确描述）。不从 D1 warm-start（D1 效果不佳，D50 -4.03 dB）。
已修改 config `run_name: first_hop_224_20k_seam_refiner_standalone`。

---

### [AC4-R2] x_src_img 多通道静默截断（MEDIUM）

**问题**：传入 `[B,3,H,W]` 时静默取第一通道，无 warning。

**决议**：✅ **已修** — 加 `print` warning，异常时及时发现。

---

### [AC5-R3] dead branch 监控 AND 条件过宽（MEDIUM）

**问题**：gate、lambda、pix_delta 三者同时为零才报警，单个为零不报。

**决议**：✅ **暂不修改**。观察 N1/N2 训练日志后再决定是否需要 per-hop lambda 监控。

---

### [AC4-R3] t_src/t_dst 形状未文档化（LOW）

**决议**：✅ **已修** — 补充 docstring 和 `assert t_src.dim() == 1` / `assert t_dst.dim() == 1`。

---

### [AC3-R4] mix_latent STE 实现缺少文档注释（LOW）

**决议**：✅ **已修** — 加 2 行 STE 语义注释（forward=mixed, backward=z_pred only）。

---

## 文献参考（补充）

### 低剂量 PET 重建最新方向

| 文献 | 会议/年份 | 方法 | 与本项目关系 |
|------|----------|------|-------------|
| MCAD (Cui et al.) | MICCAI 2024 | 多模态条件 adversarial diffusion + OT | 用 OT 做 PET，像素空间 |
| PET Tracer Separation (Huang et al.) | arXiv 2025 | Multi-latent space DiT | 多潜空间 + DiT，最接近我们的 latent transport |
| DECADE (Zhou et al.) | arXiv 2026 | 时序一致无监督 diffusion (心脏 PET) | 强调时序一致性，类似 cascade chain |
| Mamba-Powered Progressive (Tang et al.) | MICCAI 2025 | Mamba + 物理一致 | 渐进式从低到高，类似 multi-hop |
| Generalizable Unpaired PET Enhancement (Luo et al.) | TNSRE 2026 | Two-stage diffusion | 无配对数据增强 |

### Patch Boundary / SeamRefiner 相关

| 文献 | 关键点 |
|------|--------|
| Seamless High-Res Terrain Reconstruction (Rafaeli et al., arXiv 2507.09681, 2025) | ViT patch boundary seam 消除：使用 **boundary blending mask** 在 unpatchify 阶段平滑过渡。更根本但需改 decoder（与我们冻结约束冲突）|

### Weight Decay 标量参数排除

**SAC: Adaptive Learning Rate Scaling** (Li et al., 2025) 明确写道：
> "excluding scalar parameters like biases"

这是 LLM/ViT 训练的标准最佳实践，印证我们 `first_hop_weight_decay: 0.0` 的修复。

---

## 验证清单

- [x] `python -m py_compile train_first_hop.py` ✓
- [x] `python -m py_compile pet_lr/model_first_hop.py` ✓
- [x] `python -m py_compile eval_first_hop_224_clip3.py` ✓
- [x] 3 个 YAML 配置正确解析 ✓
- [x] `first_hop_weight_decay = 0.0` 在所有活跃配置中 ✓
- [x] `lambda_hop_init = 0.10` 在所有活跃配置中 ✓
- [x] `hop_residual_last_init_std = 0.01` 在所有活跃配置中 ✓
- [x] Code review: 4/4 PASS ✓

## 未修复的旧 config

Review 发现 14 个历史配置文件（10k/50k formal/chainstable 等）仍有旧参数值。
这些是已完成实验的配置快照，**不需更新**（不会重新运行）。
