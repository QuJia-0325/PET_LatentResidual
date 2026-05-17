# Codex Task — V18 LoRA Param Audit (5-minute, non-blocking)

- date: 2026-05-17
- branch: foc_lite_hop0
- HEAD when written: f4c797e
- urgency: HIGH (potential implementation bug; want answer in < 5 min)
- blocks: nothing (V18 training continues; this is a read-only audit)
- triggered by: 用户问 "为什么需要再对 decoder 加 LoRA"，claude 在数学上无法把 codex 报告的 `numel=589,824` 与 yaml `rank=32` + log `wrapped 12 Linear` 调和。

---

## §1 — 不一致的事实

Smoke log [review/0517/V18_decoder_lora/smoke_v18_train/V18_train_smoke_1step_gpu1_retry_20260517_044921.log](./smoke_v18_train/V18_train_smoke_1step_gpu1_retry_20260517_044921.log) 明确：

```
[decoder_lora] wrapped 12 Linear modules across last 2 decoder layers of rae.decoder.decoder_layers; trainable LoRA params (A+B per Linear) = 589,824
[optimizer] decoder_lora param group: tensors=24, numel=589,824, lr=5.000e-07, lr_mult=0.006250, wd=0.000e+00
```

但 yaml [V18_decoder_lora.yaml:226](./V18_decoder_lora.yaml) `rank: 32`，ViTB decoder hidden_size=768、intermediate_size=3072，12 个 Linear 应该是：

| Linear | in_features | out_features | LoRA A=(rank,in) | LoRA B=(out,rank) | per Linear |
|---|---:|---:|---:|---:|---:|
| attn.query | 768 | 768 | 24,576 | 24,576 | **49,152** |
| attn.key | 768 | 768 | 24,576 | 24,576 | **49,152** |
| attn.value | 768 | 768 | 24,576 | 24,576 | **49,152** |
| attn.output.dense | 768 | 768 | 24,576 | 24,576 | **49,152** |
| intermediate.dense | 768 | 3072 | 24,576 | 98,304 | **122,880** |
| output.dense | 3072 | 768 | 98,304 | 24,576 | **122,880** |
| **per layer (6 Linear)** | | | | | **442,368** |
| **2 layers** | | | | | **884,736** |

- Codex 报告: **589,824**
- 期望（rank=32 全 wrap）: **884,736**
- 差额: **294,912**

差额恰好 = `2 layers × 2 MLP Linear × (rank × (3072 − 768)) = 2×2×32×2304 = 294,912`，即 **fc1/fc2 的 LoRA matrices 实际比预期小**。

## §2 — 4 个可能的解释（按可能性排序）

1. **fc1/fc2 没被 wrap**（target_keywords 没匹配到 `intermediate.dense` / `output.dense`）。但 log 说 wrapped 12 Linear，这与"每层只 wrap 4 attention Linear（8 个）"矛盾。
2. **fc1/fc2 被 wrap 但 rank 是 16 not 32**（intermediate.dense 与 attention.output.dense 共享 name "dense" → 某种命名冲突 →fallback）。
3. **fc1/fc2 被 wrap 时 LinearWithLoRA 的 in/out_features 没用真实的 (768, 3072)/(3072, 768) 而是 (768, 768)**（不可能，因为 LinearWithLoRA 从 base.linear 读 in/out_features）。
4. **Codex 在打印 numel 时用了错误公式**（log 误报，实际 wrap 正确）。

只有（4）不影响训练；（1）（2）（3）都意味着 V18 实际容量比设计小 33%，**应该 stop + restart**。

## §3 — Codex 应执行的 5 分钟读 audit（不停训练）

V18 训练正在 GPU 1 跑着，**不要停**。你可以：

### 选项 A — 从最新 V18 checkpoint 读 state_dict（推荐，零干扰）

V18 训练每 10K step 存 ckpt，但目前才 ~400 step。所以**用 V18 的 best.pt 或 last.pt**（它们在 V18 trainer 启动后立即写）：

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

CKPT_DIR=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora
# Pick whatever V18 ckpt exists right now (best.pt 或 last.pt 或 step_*.pt)
ls -la $CKPT_DIR/*.pt | tail -5
```

然后运行（在登录 shell，不需要 GPU）：

```bash
python3 <<'PY'
import torch
from collections import defaultdict
from pathlib import Path

ckpt_path = Path("/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/last.pt")
print(f"loading {ckpt_path}")
sd = torch.load(ckpt_path, map_location="cpu")["model"]

# Find all LoRA params on rae.decoder.decoder_layers.*
decoder_lora_keys = sorted(k for k in sd.keys() if "rae.decoder.decoder_layers" in k and ("lora_A" in k or "lora_B" in k))
print(f"\n=== Total decoder LoRA tensors: {len(decoder_lora_keys)} ===\n")

total_numel = 0
by_module = defaultdict(list)
for k in decoder_lora_keys:
    t = sd[k]
    n = t.numel()
    total_numel += n
    # strip the lora_A/B suffix to group by module
    mod = k.rsplit(".", 1)[0]
    by_module[mod].append((k.rsplit(".", 1)[1], tuple(t.shape), n))

for mod, entries in sorted(by_module.items()):
    short_mod = mod.replace("rae.decoder.decoder_layers.", "")
    for name, shape, n in entries:
        print(f"  {short_mod}.{name}  shape={shape}  numel={n:,}")
    print()

print(f"=== TOTAL LoRA numel: {total_numel:,} ===")
print()
print(f"Expected (rank=32 × 12 Linear × full ViTB ViTMAELayer): 884,736")
print(f"Codex log reported:                                     589,824")
print(f"Match expected?  {'YES' if total_numel == 884736 else 'NO'}")
print(f"Match log?       {'YES' if total_numel == 589824 else 'NO'}")
PY
```

### 选项 B — 从正在跑的 model 读 named_parameters（也不停训练）

如果没有 V18 ckpt（训练刚启动还没存盘），可以从 RAE LinearWithLoRA 静态算：

```bash
python3 <<'PY'
# Simulate V18 wrap without loading actual model — pure shape inspection
import sys, torch
sys.path.insert(0, '/home/qujiaxiang/project/PET_LatentResidual')
sys.path.insert(0, '/home/qujiaxiang/project/RAE/code/RAE')

import yaml
from pet_lr.model_first_hop import PETFlowDiTFirstHop
from pet_lr.decoder_lora import wrap_decoder_with_lora

cfg = yaml.safe_load(open('review/0517/V18_decoder_lora/V18_decoder_lora.yaml'))
device = torch.device('cpu')   # CPU only for inspection; no GPU needed

# Build model (without loading ckpt — just to inspect structure)
model = PETFlowDiTFirstHop(cfg, device)
# Wrap decoder
lora_params, wrapped_names = wrap_decoder_with_lora(model.rae, cfg)

# Report
from collections import defaultdict
by_linear = defaultdict(list)
total = 0
for n, p in model.rae.named_parameters():
    if 'rae.decoder.decoder_layers' in 'rae.'+n or 'decoder.decoder_layers' in n:
        if p.requires_grad and ('lora_A' in n or 'lora_B' in n):
            total += p.numel()
            # Strip lora_A/B
            parts = n.rsplit('.', 1)
            by_linear[parts[0]].append((parts[1], tuple(p.shape), p.numel()))

print(f"Total LoRA tensors trainable: {sum(len(v) for v in by_linear.values())}")
print(f"Total LoRA numel: {total:,}")
print()
for k in sorted(by_linear.keys()):
    short = k.replace('decoder.decoder_layers.', '')
    for name, shape, n in by_linear[k]:
        print(f"  {short}.{name}  shape={shape}  numel={n:,}")
    print()
PY
```

## §4 — 期望输出与决策

### 若 audit 显示 numel = 884,736（matches my expectation）

- Codex 早期 log 的 "589,824" 是误打。
- 实际 wrap 正确：每层 6 Linear 都被 wrap，含 fc1/fc2。
- V18 训练继续，按计划跑到 step 200000。
- 在 audit 报告里 commit 一条 "Wrap count corrected: actual 884,736 (log misprinted)" 即可。

### 若 audit 显示 numel = 589,824（matches log）

- 实施有 bug：fc1/fc2 漏掉或被错 rank wrap。
- **不必立即 stop V18**：当前 step ~400，还在 KL warmup 内（< 2000 step），LoRA 几乎没动。
- Codex 在 audit md 文档里列出 missing wraps（哪些 module 没在 audit 输出里）。
- Claude 整合后决定是否 stop V18 + 修 wrap + 重启（成本：丢 400 step 训练 ≈ 30 min GPU）。

### 若 audit 显示其他数

- 例如 393,216（只 wrap 8 attention Linear）或 442,368（rank 16 等）。
- 同上：列详情，让 claude 决定。

## §5 — Audit 输出文件

Codex 把上面 python snippet 输出写到：

```
review/0517/V18_decoder_lora/AUDIT_LORA_PARAM_COUNT_20260517.md
```

格式建议：

````markdown
# V18 LoRA Param Audit (codex, $(date))

## Command run

```
（粘 codex 实际跑的 python snippet 与 ckpt 路径）
```

## Result

```
（粘上面 python 输出）
```

## Decision

- Total numel: <数字>
- Match expected 884,736? <YES/NO>
- Match log 589,824? <YES/NO>

## What to do next (codex's read of the situation)

- 一句话：建议 stop / 不 stop V18 + 原因
````

## §6 — 不需要做的事

- 不要停 V18 训练
- 不要修改任何 yaml / py 代码
- 不要发新 review prompt
- 不要 launch V14（V14 与本 audit 无关，独立决定）

## §7 — 这件事的元教训

Claude 在 commit 时只检查了 codex log 里的 trainable params count，未独立 cross-check 与 yaml 数学一致性。**第 7 次 confirmation bias**: 把 codex 的 log 输出当作 ground truth。修复：claude 在任何 commit 前必须从 first principles 算一遍关键数字，不能复制 codex 的数字而不质疑。这条加入 `/memories/repo/pet_latent_residual_eval.md`。
