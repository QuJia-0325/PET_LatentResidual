# RAE 架构发现（影响 V18 设计）— 20260517

- date: 2026-05-17 晚
- branch: foc_lite_hop0
- triggered by: 用户提醒检查 RAE/RAE 源码（之前 V18 设计假设 decoder 是黑盒）
- supersedes: V18_decoder_lora.yaml 初版的 `target_root: rae.decoder.blocks` 与 decoder_lora.py 自实现 LoRA

> 本文件记录在写 V18 yaml 之后才发现的 RAE 架构事实。**所有事实都已读 [RAE/RAE/src](../../../RAE/RAE/src) 源码验证**，不再是 "假设需 codex 在服务器查"。同时已更新 V18 yaml + decoder_lora.py 反映真实结构。

---

## §1 — RAE 整体结构（已验证）

来源：[RAE/RAE/src/stage1/rae.py](../../../RAE/RAE/src/stage1/rae.py)

```
RAE (nn.Module)
├── encoder        = Dinov2withNorm (DINOv2)
│   └── (PET_LatentResidual V7 cfg.rae.use_lora=true → 已 LoRA-tuned)
│   └── 在 V18 中：保持 frozen，不变
├── decoder        = GeneralDecoder
│   ├── decoder_embed         nn.Linear(hidden=384 → decoder_hidden=768)
│   ├── decoder_pos_embed     nn.Parameter (fixed sin-cos, requires_grad=False)
│   ├── decoder_layers        nn.ModuleList[ViTMAELayer, 12 层 for ViTB]   ← V18 LoRA 注入点
│   ├── decoder_norm          nn.LayerNorm
│   └── decoder_pred          nn.Linear(decoder_hidden → patch²·channels)
└── encoder_mean / encoder_std (buffers)
```

**关键事实**：
- RAE encoder 已经在 PET 数据上 LoRA-tuned，state_dict 在 `/data_2/qujiaxiang/outputs/pet_lora_dinov2_pt_224/best_model.pt`（V7 cfg `rae.checkpoint_path`）。V7 训练时 RAE 整体 frozen，但 encoder 的 LoRA params 是从这个 ckpt 加载的，不是 fresh init。
- RAE decoder 是 **vanilla ViT-MAE**，没有 LoRA。在 RAE 自己的 pretraining 里 ([RAE/RAE/src/train_stage1.py:289](../../../RAE/RAE/src/train_stage1.py#L289))：`rae.decoder.requires_grad_(True)` —— decoder 是 full fine-tune。
- 这意味着 **V18 是项目第一次给 RAE decoder 加 LoRA**。

---

## §2 — 每个 ViTMAELayer 的 Linear 命名（已验证）

来源：[RAE/RAE/src/stage1/decoders/decoder.py:334-540](../../../RAE/RAE/src/stage1/decoders/decoder.py#L334)

```
ViTMAELayer
├── attention (ViTMAEAttention)
│   ├── attention (ViTMAESelfAttention)
│   │   ├── query        nn.Linear(d, d)
│   │   ├── key          nn.Linear(d, d)
│   │   └── value        nn.Linear(d, d)
│   └── output (ViTMAESelfOutput)
│       └── dense        nn.Linear(d, d)
├── intermediate (ViTMAEIntermediate)
│   └── dense            nn.Linear(d, 4d)   ← FFN 第 1 层 (fc1)
├── output (ViTMAEOutput)
│   └── dense            nn.Linear(4d, d)   ← FFN 第 2 层 (fc2)
├── layernorm_before     nn.LayerNorm
└── layernorm_after      nn.LayerNorm
```

每个 layer 有 **6 个 Linear**（3 attn projections + 1 attn output + 2 FFN）。

按 `named_modules()` 输出的相对路径：
- `attention.attention.query`
- `attention.attention.key`
- `attention.attention.value`
- `attention.output.dense`
- `intermediate.dense`
- `output.dense`

---

## §3 — RAE 已有的 LoRA 工具（直接复用）

[RAE/RAE/src/utils/lora.py](../../../RAE/RAE/src/utils/lora.py) 提供：

| 函数/类 | 用途 |
|---|---|
| `LinearWithLoRA(linear, rank, alpha, dropout)` | 直接 wrap 一个 `nn.Linear`，自动 freeze base.weight/bias + zero-init lora_B（保证 step 0 输出 == base 输出） |
| `inject_lora_into_dinov2_attention(model, rank, alpha, dropout, target_keywords)` | 整模型 walk + match keyword + in-place wrap。RAE encoder 自己用的就是这个 |
| `collect_lora_parameters(model)` | yield 所有 `lora_A` / `lora_B` params |

**RAE pretrained encoder ckpt 用的 target_keywords**（来自 V7 yaml `rae.lora_target_keywords`）：

```yaml
- "attention.qkv"
- "attention.projection"
- "attention.attention.query"
- "attention.attention.key"
- "attention.attention.value"
- "attention.output.dense"
- ".mlp.fc1"
- ".mlp.fc2"
```

注意这里前两个 `attention.qkv` / `attention.projection` 是 DINOv2 encoder 的命名（fused QKV）；后面才是 ViT-MAE decoder 命名。`.mlp.fc1/fc2` 是 DINOv2 的；ViT-MAE decoder 等价物是 `intermediate.dense` / `output.dense`。所以**直接复制 V7 encoder 的 target_keywords 到 V18 decoder 配置会无 match**。

---

## §4 — V18 设计的具体修正

### 4.1 V18 yaml `training.decoder_lora` 块（已更新）

旧（错的）：
```yaml
decoder_lora:
  target_root: "rae.decoder.blocks"   # 不存在
  target_module_types: ["Linear"]
  name_regex: ".*(qkv|proj|fc1|fc2|to_q|to_k|to_v|to_out).*"  # ViT-MAE 不用这些命名
```

新（正确）：
```yaml
decoder_lora:
  enabled: true
  target_root: "rae.decoder.decoder_layers"   # 验证过的 nn.ModuleList[ViTMAELayer]
  last_n_blocks: 2                            # 12 层中的 layer 10, 11
  rank: 8
  alpha: 16
  dropout: 0.0
  init_scale_zero: true
  target_keywords:
    - "attention.attention.query"
    - "attention.attention.key"
    - "attention.attention.value"
    - "attention.output.dense"
    - "intermediate.dense"     # ViTMAEIntermediate.dense ≡ fc1
    - "output.dense"           # ViTMAEOutput.dense       ≡ fc2
```

每 layer 6 个 Linear，last 2 layers = 12 个 Linear；rank=8 → trainable params 约 `12 × 2 × 8 × 768 ≈ 150K`（< RAE decoder 总 params 的 0.2%）。

### 4.2 V18 decoder_lora.py 改造（已更新）

旧（错的）：自实现 `LoRAAdapter` class。

新（正确）：**复用 RAE 自带的 `LinearWithLoRA`**。从 `src.utils.lora import LinearWithLoRA`。

好处：
- 同 RAE encoder LoRA 使用同一份 LoRA 实现 → ckpt 格式兼容
- RAE LinearWithLoRA 已经 zero-init B（[lora.py:34](../../../RAE/RAE/src/utils/lora.py#L34) `nn.init.zeros_(self.lora_B)`）→ 自动保证 V18 step 0 输出 == V7 best.pt
- 减少维护负担

`pet_lr/decoder_lora.py` 现只剩 ~80 行核心逻辑：
- `wrap_decoder_with_lora(rae, cfg)` — 仅在 last N decoder layers 调 `LinearWithLoRA` 替换
- `build_frozen_reference_decoder(cfg, device)` — 直接调 `load_rae_model(cfg, device)` 得到第二份 frozen RAE
- `compute_kl_pullback_loss(rae_lora, rae_frozen, z_gt, crop_size)` — 不变

### 4.3 V18 step 0 sanity check 现在是 free 的

因为 `LinearWithLoRA.reset_parameters` 总是 `nn.init.zeros_(lora_B)`，B@A = 0 at init，前向输出 = base.linear(x)。所以**不需要 codex 额外验证 zero-init**；只要 RAE 的 LoRA 没改实现，就保证 V18 step 0 输出 = V7 best.pt 输出。

### 4.4 取消 codex Day 0 的 "RAE 结构 inspection"

[CODEX_RUNBOOK_V18_20260517.md §3 Phase A](./CODEX_RUNBOOK_V18_20260517.md) 之前要 codex 在服务器 `print(rae)` 找 block list 路径。**现在不需要了** —— attribute path 已经确定是 `rae.decoder.decoder_layers`。

但 Phase A 的另一件事（RAE eval/train mode 切换是否产生数值差异）仍然有用 —— 它验证 LoRA wrap 不会因 train mode 触发 dropout/BN drift 而坏掉。保留这个 probe。

---

## §5 — Encoder LoRA 与 Decoder LoRA 的命名冲突隐患（重要）

RAE encoder LoRA 与 V18 decoder LoRA 都会用相同的 `lora_A` / `lora_B` 参数名。在 state_dict 里它们被 attribute path 区分：

- Encoder LoRA: `rae.encoder.encoder.blocks.X.attn.qkv.lora_A`（DINOv2 路径）
- Decoder LoRA (V18): `rae.decoder.decoder_layers.X.attention.attention.query.lora_A`（ViT-MAE 路径）

**Resume from V7 best.pt 的 state_dict 处理**：
- V7 ckpt 里有 encoder LoRA params（已 trained）
- V7 ckpt 里**没有** decoder LoRA params（V7 decoder 是 vanilla frozen）
- V18 model 创建后会有两套 LoRA params；其中 encoder 那套需要从 V7 ckpt 加载
- `load_state_dict(strict=False)` + 检查 missing_keys 全在 `rae.decoder.decoder_layers.*.lora_*` 即可

具体 resume 代码 codex 写在 train_first_hop.py 的修改里（见 [CODEX_RUNBOOK_V18_20260517.md §4.2 F](./CODEX_RUNBOOK_V18_20260517.md)）—— 已经覆盖了 `allowed_missing = [k for k in missing_keys if k.endswith(".lora_A") or k.endswith(".lora_B")]` 这条逻辑，正确。

---

## §6 — V18 设计完整性 checklist（不再有 TODO）

| 项 | 状态 |
|---|---|
| attribute path 已确定 | ✅ `rae.decoder.decoder_layers` |
| Linear 命名已确定 | ✅ 6 个 keywords 已列举 |
| LoRA wrapper 来源 | ✅ 复用 RAE 自带 `LinearWithLoRA` |
| Step 0 == V7 best.pt 保证 | ✅ `LinearWithLoRA` 已 zero-init B |
| KL pull-back loss 数学 | ✅ MSE on 同一 GT latent |
| 第二份 frozen RAE 构建 | ✅ `load_rae_model(cfg, device)` 再调一次 |
| Resume from V7 best.pt 处理 | ✅ strict=False + 检查 missing 全是 lora_A/lora_B |
| ViTB 12 层假设 | ⚠️ V7 yaml `rae.decoder_config_path: /home/qujiaxiang/project/RAE/vit-mae` 没明说 ViTB/L/XL；Codex Day 0 `print(len(rae.decoder.decoder_layers))` 验证（30 秒），把数字写到 logs 里。如果是 24 层 (ViTL) 或 28 层 (ViTXL) 也没关系，last_n_blocks=2 仍 valid |

---

## §7 — Codex Day 0 简化版

原本 [CODEX_RUNBOOK_V18_20260517.md §3](./CODEX_RUNBOOK_V18_20260517.md) Day 0 Phase A 让 codex 花 1 小时跑 inference-mode probe + 结构 inspection。**简化后**：

| 动作 | 时间 |
|---|---|
| Phase A1: load V7 best.pt + `print(len(rae.decoder.decoder_layers))` | 1 min |
| Phase A2: 在 eval vs train mode 下 decode 同一 GT latent，diff 检查 | 5 min |
| Phase B: `ls /data_2/.../V7/run/step_*.pt` 中间 ckpt 存活检查 | 1 min |
| Phase C: V9a 重选（若 ckpt 存活，可选） | 30 min |

**总 Day 0 时间 < 1 小时**，比原 runbook 估的还快。

---

## §8 — 给团队的诚实总结

发现 RAE 自带 LoRA 工具 + V7 已经在用 encoder LoRA + decoder 是 vanilla ViT-MAE 这三件事，**完全改变了 V18 实现负担**：

- 从"自己写 LoRA wrapper + 想清楚 LoRA-on-LoRA 语义" → "直接复用 RAE 工具 + 单层 LoRA"
- 从"Codex Day 0 要花 1 小时探 decoder 结构" → "已经全部读源码验证过"
- 从"V18 是项目第一次给 RAE 加 LoRA" → "V18 是项目第一次给 RAE **decoder** 加 LoRA，但加 LoRA 这件事 RAE encoder 已经做过"

**Day 0/1 实施风险显著下降**。但元教训也成立：之前 V18 yaml 写出来时**假设了 decoder 是黑盒**，而实际上源码就在 `RAE/RAE/src/` 里随时可读。读 1 小时源码可以省 1 天 codex 摸索。

下次设计训练实验前，强制 checklist："已经 `read_file` 过涉及的 every 模型源文件了吗？"
