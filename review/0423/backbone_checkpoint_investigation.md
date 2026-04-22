# Transport 深度分析 — Backbone Checkpoint 调查请求

**日期**: 2026-04-23  
**分支**: `foc_lite_hop0`  
**请求**: 需要 Codex 在服务器上检索 backbone checkpoint 的训练信息

---

## 背景

PET_LatentResidual 的所有实验都使用同一个 backbone 预训练权重作为 transport velocity predictor 的起点：

```
/data_2/qujiaxiang/outputs/pet_flow_224_small_rollout_hopaware_smoke/mean_flow_224_small_rollout4hop_hopaware_smoke/best.pt
```

这个 checkpoint 在 `build_backbone()` 中加载（`pet_lr/model_first_hop.py:571-600`），然后 `train_first_hop.py` 在此基础上继续微调（`freeze_backbone: false`）。

---

## 需要检索的信息

### 1. Checkpoint 元数据

请加载 checkpoint 并提取：

```python
import torch
ckpt = torch.load(
    "/data_2/qujiaxiang/outputs/pet_flow_224_small_rollout_hopaware_smoke/"
    "mean_flow_224_small_rollout4hop_hopaware_smoke/best.pt",
    map_location="cpu"
)
print("Keys:", list(ckpt.keys()))
print("Step:", ckpt.get("step", "NOT FOUND"))
print("target_normalize:", ckpt.get("target_normalize", "NOT FOUND"))
print("best_metric:", ckpt.get("best_metric", "NOT FOUND"))
print("best_val:", ckpt.get("best_val", "NOT FOUND"))
print("Has model_ema:", "model_ema" in ckpt)
print("Has model:", "model" in ckpt)

# 检查 positional embedding 形状（确认 input_size）
state = ckpt.get("model_ema", ckpt.get("model", {}))
for k, v in state.items():
    if "pos_embed" in k:
        print(f"  {k}: shape={v.shape}")
```

### 2. 训练配置

查找该 checkpoint 对应的训练 config：

```bash
# 查找输出目录下的 config
ls -la /data_2/qujiaxiang/outputs/pet_flow_224_small_rollout_hopaware_smoke/mean_flow_224_small_rollout4hop_hopaware_smoke/
cat /data_2/qujiaxiang/outputs/pet_flow_224_small_rollout_hopaware_smoke/mean_flow_224_small_rollout4hop_hopaware_smoke/config.yaml 2>/dev/null || echo "No config.yaml found"

# 查找训练日志
find /data_2/qujiaxiang/outputs/pet_flow_224_small_rollout_hopaware_smoke/ -name "*.log" -o -name "*.jsonl" | head -10
```

### 3. 训练代码来源

checkpoint 名字含 "smoke"（烟雾测试），需确认：

```bash
# 查找产生该 checkpoint 的代码/分支
# 可能在 RAE 的某个分支或独立脚本中
find /home/qujiaxiang/project/ -name "*.py" -path "*hopaware*" 2>/dev/null | head -10
find /home/qujiaxiang/project/ -name "*.yaml" -path "*224*hopaware*" 2>/dev/null | head -10
```

---

## 关键问题

| 问题 | 为什么重要 |
|------|----------|
| **训练了多少步？** | "smoke" 暗示可能只训练了很短时间。如果只有 5K-10K 步，backbone 严重欠训练 |
| **target_normalize 是 true 还是 false？** | PET_LatentResidual 从 checkpoint 读取此值，决定 velocity 是否归一化 |
| **alpha_end 是多少？** | 决定 backbone 在多大程度上学会了自回归预测（vs teacher forcing） |
| **最终 loss 是多少？** | 评估 backbone 收敛程度 |
| **pos_embed 形状？** | 确认是 [1, 256, 384]（224/14=16 tokens）还是其他 |

---

## Oracle 实验对比（已确认）

| Timepoint | Decoder Ceiling | Transport Best | Gap |
|-----------|----------------|---------------|-----|
| D20 | 46.636 dB | 35.924 dB | **10.7 dB** |
| D10 | 48.744 dB | 36.094 dB | **12.6 dB** |
| D4 | 50.832 dB | 36.147 dB | **14.7 dB** |
| NORMAL | 52.634 dB | 36.450 dB | **16.2 dB** |

transport 与 decoder ceiling 有 10-16 dB 的巨大 gap。backbone 预训练质量是关键因素之一。

---

## 已确认的修复（v2 config）

| 修复项 | 旧值 | 新值 |
|--------|------|------|
| first_hop_weight_decay | 0.01 | 0.0 |
| lambda_hop_init | 0.01 | 0.10 |
| hop_residual_last_init_std | 0.002 | 0.01 |
| EMA | 无 | decay=0.9999 |

这些修复确保了 hop residual 和 pixel forcing 分支能够正常激活，但**不影响 backbone 本身的预训练质量**。

---

## 后续方向（取决于检索结果）

- 如果 backbone 训练步数 < 20K：**backbone 严重欠预训练**，需要在 RAE 侧重新训练更长时间
- 如果 backbone 训练步数 > 50K：问题在 PET_LatentResidual 的微调策略（LR、batch size 等）
- 如果 target_normalize 与预期不符：需要对齐
