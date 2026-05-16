# Codex Runbook — V18 RAE Decoder LoRA (Day 1 Implementation + Day 2 Launch)

- date: 2026-05-17
- author: claude (Opus 4.7) for codex on the GPU server
- branch: foc_lite_hop0
- HEAD when written: 80576c9 (will bump after this commit)
- prior context: read [DIAGNOSTIC_FINDING_20260517.md](./DIAGNOSTIC_FINDING_20260517.md) first — explains why V18 exists and what was retracted

> **TL;DR for codex**: 3 independent reviewers in Round 3 confirmed claude's 4th confirmation bias (diagnostic laundering — 6 days of review, 0 GPU launches). Real bottleneck is the **frozen RAE decoder**, locked by `train_first_hop.py:1298` hard assert. V18 unlocks last 1-2 decoder blocks via LoRA with KL pull-back, expected +0.5-2.0 dB NORMAL PSNR_clip3 (3-10× the entire V11/V17 family combined). **Hard timebox: Day 2 must have V18 + V14 on GPUs.**

> **20260517 下午更新**：claude 在写本 runbook 后读了 [RAE/RAE/src](../../RAE/RAE/src)，验证了 decoder 结构、复用了 RAE 自带 `LinearWithLoRA`。完整发现见 [RAE_ARCHITECTURE_FINDING_20260517.md](./V18_decoder_lora/RAE_ARCHITECTURE_FINDING_20260517.md)。**Day 0 依赖 inspection 变 < 5 分钟**，其余部分（§3 Phase B/C 与 §4 Day 1）仍有效。

> **20260517 晚 更新（Round 4 integration）**：3 位独立 reviewer 共识指出 V18 设计有 5 处问题，最关键的是 "11.2 dB gap" 是 round-trip ceiling 不是 V18 可吃的 gap，且 KL λ=0.5 会**冻死** V18 decoder LoRA。详见 [REVIEW_INTEGRATION_round4_20260517.md](./REVIEW_INTEGRATION_round4_20260517.md)。已立即修：(1) yaml KL block λ=0.05 / use_pred_latent=true / warmup=2000；(2) 写 `tools/probe_v18_gap_decomposition.py`。**Day 0 新增 Phase 0**（必跑）：probe 决定 launch V18 (rank=8/32) 还是 V21 (conv head)。Day 2 launch 截止时间不变。

---

## §1 — 项目当前 decision state

### 撤回（不要再花 GPU）

- V11 / V11' / V17 / V17' / V15：image_aux 围墙花园全族。M1 数据显示 val_hop0_img_total 末期/早期=1.028（仅改 2.7%），路径已饱和。
- V13 (true image_aux ablation)：image_aux 已饱和，V13 ΔPSNR 期望 < 0.05 dB，没诊断价值。
- V9b：β=2.5 selector 改动配合 hop3 step_weight 翻倍预期会让 hop3 single-step 进一步退化。

### 保留

- **V14**：V7 + seed=1337，唯一真 d_pure 测量。yaml 已就绪：[review/0516/V14_true_d_pure/V14_v7_seed1337.yaml](../0516/V14_true_d_pure/V14_v7_seed1337.yaml)。**Day 2 launch GPU 1**。
- **V18**：本 runbook 主轴。**Day 2 launch GPU 0**。

### 可选（廉价）

- **V9a**：用 V7 保留的 16 个中间 ckpt（step_10K..step_160K）+ β=2.5 selector 重选 best.pt，跑 full-val。**只要服务器上中间 ckpt 存活**就 1-2 GPU-hour 结案。如果重选结果 = 原 best.pt（step_160K），V9 训练直接被证伪为 no-op。**Day 0 检查 ckpt 存活，Day 1 跑**。

---

## §2 — V18 设计核心（DIAGNOSTIC_FINDING §6.2 的展开）

### 2.1 攻击的具体瓶颈

| 量 | 值 | 含义 |
|---|---|---|
| V7 chain D20 PSNR_clip3 | 35.4 dB | 我们到的位置 |
| RAE decoder ceiling on D20 | ~46.6 dB | 物理上限（同一 RAE encode→decode 不通过 backbone 时的 PSNR） |
| **transport gap** | **~11.2 dB** | **整个 V11/V17 全族在 0.05-0.3 dB 区间打转，没碰过的 region** |
| `freeze_rae=True` 硬 assert | train_first_hop.py:1298 | **项目自己锁住这 11.2 dB** |

### 2.2 V18 的 6 个独立改动（全部基于 V7 best.pt warm-start）

| # | 改动 | 文件 / 字段 | 默认值 |
|---|---|---|---|
| 1 | 撤掉 freeze_rae 硬 assert | [train_first_hop.py:1298-1302](../../train_first_hop.py#L1298) | 改为 "if not freeze_rae AND not decoder_lora.enabled, raise" |
| 2 | `freeze_rae: false` | yaml `training.freeze_rae` | false |
| 3 | LoRA wrap RAE decoder 最后 2 层 | yaml `training.decoder_lora` 块 | rank=8, alpha=16 |
| 4 | KL pull-back loss | yaml `loss.decoder_kl_pullback` 块 | λ_kl=0.5 |
| 5 | LoRA 专属 LR = 5e-7 | yaml `optimizer.decoder_lr_mult: 0.00625` | 100× 低于 backbone LR |
| 6 | warm-start from V7 best.pt + 额外 40K step | yaml `training.resume_from` + `max_steps=200000` | V7 stop at 160K, V18 = V7 + 40K finetune |

**关键设计原则**：`init_scale_zero=True` 让 LoRA B 矩阵零初始化，B@A=0 at step 0 → **V18 step 0 输出与 V7 best.pt 完全相同**。这是 Day 2 launch 后的第一个 sanity check。

### 2.3 预注册 success / kill 阈值（写死在 yaml 头注释，不可事后改）

| 阈值 | 数值 | 触发 |
|---|---|---|
| PRIMARY | ΔNORMAL PSNR_clip3 (V18@200K - V7@160K) ≥ **+0.30 dB** | 11.5× 我们最严格的 d_pure 上限 0.026 dB |
| SAFETY image_aux | image_aux full-val loss 比 V7 best.pt 增加 < 30% | 防 decoder 漂走导致 image_aux 反向 |
| SAFETY KL | KL pull-back loss < 0.1 (绝对) | 防 decoder 偏离 RAE 原始 manifold |
| SAFETY gate | gate_pix / lambda_hop[0] 末期值偏离 V7 末期 < ±50% | 防 hop0 路径被打乱 |
| KILL (mid-train) | 20K step 时 NORMAL ΔPSNR < +0.05 dB **AND** KL loss > 0.05 | 立即停训，方向错 |

---

## §3 — Day 0 (今天, ≤ 1 hour total)

所有操作 codex 在服务器跑。

### Phase A — RAE inference-time sanity（30 min, 0 训练）

目标：确认 RAE decoder 在 train mode vs eval mode 下输出**几乎相同**（否则有 dropout/BN 漂移，需要在 V18 设计里加 `rae.eval()`-like 保护）。

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only gitee foc_lite_hop0

# Create a 30-line probe script
cat > /tmp/v18_phase_a_probe.py <<'PYEOF'
import sys, torch
sys.path.insert(0, '.')
from pet_lr.model_first_hop import PETFlowDiTFirstHop
import yaml

cfg = yaml.safe_load(open('review/0511/log_snapshots_20260516_163900/configs/V7_config.resolved.yaml'))
device = torch.device('cuda:0')
model = PETFlowDiTFirstHop(cfg, device).to(device)
sd = torch.load('/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt', map_location='cpu')
model.load_state_dict(sd['model'], strict=False)
model.eval()

# Probe: take a real GT latent, decode in eval mode vs train mode, compare.
from pet_lr.data_first_hop import PETFirstHopAligned4HopDataset
ds = PETFirstHopAligned4HopDataset(cfg['data'], split='val')
batch = ds[0]
z_gt_D20 = batch['z_rollout'][1].unsqueeze(0).to(device)  # GT D20 latent

with torch.no_grad():
    model.rae.eval()
    x_eval = model.decode_crop(z_gt_D20).cpu()
    model.rae.train()
    x_train = model.decode_crop(z_gt_D20).cpu()
    model.rae.eval()  # restore

diff = (x_eval - x_train).abs().mean().item()
print(f"|decode_eval - decode_train| mean abs diff = {diff:.6e}")
if diff < 1e-5:
    print("PASS: RAE decoder is deterministic (no dropout/BN drift). LoRA wrap is safe.")
else:
    print(f"WARN: RAE has internal stochastic layers; diff={diff:.4e}. Need to force .eval() on non-LoRA paths.")
PYEOF
CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python /tmp/v18_phase_a_probe.py 2>&1 | tee review/0517/V18_decoder_lora/PHASE_A_PROBE.log
```

**Outcomes**：
- diff < 1e-5 (typical for ViT decoder with no dropout): **PASS** → 直接走 §4 实现
- diff > 1e-5: 需要在 LoRA wrap 时只 enable LoRA 部分的 train mode，其他部分保持 eval。我已在 [decoder_lora.py](../../pet_lr/decoder_lora.py) 注释里 flag 了这点，需要 codex 在 model_first_hop.py 修改时加 `self.rae.eval(); for m in lora_wrapped_modules: m.train()` 的精细控制

### Phase B — V7 中间 ckpt 存活检查（5 min）

```bash
ls -la /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/step_*.pt | wc -l
ls -la /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/step_*.pt | head -20
```

- 期望 16 个 step_*.pt (10K..160K @ save_interval=10000)
- **若 ≥ 16**：V9a 可做（§3.3 Phase C）
- **若 < 16**：V9a 跳过，直接进 §4 V18 实现

### Phase C — V9a 重选（若 ckpt 存活，1-2 GPU-hour，可与 Phase A 同步进行）

```bash
# Re-evaluate V7 metrics.jsonl under β_NORMAL=2.5 selector
python3 <<PYEOF
import json
records = []
with open('/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/metrics.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('event') == 'val_full':
            records.append(r)
print(f"Found {len(records)} full-val records")
# β=2.5 selector
def score25(r):
    return 0.5*r['val_chain_d20_mse'] + 0.45*r['val_chain_d10_mse'] + 0.9*r['val_chain_d4_mse'] + 2.5*r['val_chain_normal_mse']
best = min(records, key=score25)
print(f"β=2.5 selected step: {best['step']}, score: {score25(best):.6f}, NORMAL MSE: {best['val_chain_normal_mse']:.6f}")
# original β=1.5
def score15(r):
    return 0.5*r['val_chain_d20_mse'] + 0.45*r['val_chain_d10_mse'] + 0.9*r['val_chain_d4_mse'] + 1.5*r['val_chain_normal_mse']
best15 = min(records, key=score15)
print(f"β=1.5 (original) selected step: {best15['step']}, score: {score15(best15):.6f}, NORMAL MSE: {best15['val_chain_normal_mse']:.6f}")
PYEOF
```

如果两个 β 选出**相同 step**：V9a 已 falsified；不需要训 V9b。把这条记到 V9 yaml 的 deprecation 注释里。

如果 β=2.5 选出**不同 step**：跑 full-val eval on that ckpt（30 min），看是否实际改善。

---

## §4 — Day 1 (V18 implementation)

### 4.1 已就绪的工件（claude 在本地写好）

- [V18_decoder_lora.yaml](./V18_decoder_lora/V18_decoder_lora.yaml) — 训练配置，16 顶层块齐全，已 yaml.safe_load 通过
- [pet_lr/decoder_lora.py](../../pet_lr/decoder_lora.py) — LoRA wrapper + KL pull-back loss helper（数学完整，标了 `# Codex TODO` 的部分是 model/data wiring）

### 4.2 Codex 需要做的 7 个具体改动

#### A. train_first_hop.py:1298-1302 — 替换 freeze_rae 硬 assert

旧：
```python
if not bool(train_cfg_boot.get("freeze_rae", True)):
    raise RuntimeError(
        "docs/main.md requires training.freeze_rae=true for first-hop training. "
        "Decoder unfreezing is not allowed in this trainer."
    )
```

新：
```python
decoder_lora_enabled = bool(train_cfg_boot.get("decoder_lora", {}).get("enabled", False))
if not bool(train_cfg_boot.get("freeze_rae", True)) and not decoder_lora_enabled:
    raise RuntimeError(
        "training.freeze_rae=false requires training.decoder_lora.enabled=true "
        "(no full decoder unfreeze allowed; only LoRA partial unfreeze is supported)."
    )
```

#### B. model_first_hop.py:432-436 — assert_decoder_frozen 改为只检查非-LoRA 参数

旧：
```python
def assert_decoder_frozen(self) -> None:
    for name, p in self.rae.named_parameters():
        if p.requires_grad:
            raise RuntimeError(f"Decoder param must be frozen but is trainable: {name}")
```

新：
```python
def assert_decoder_frozen(self) -> None:
    """Allow lora_A / lora_B params (added by decoder_lora.wrap_decoder_with_lora) to be trainable.
    All other RAE params must remain frozen."""
    for name, p in self.rae.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        if p.requires_grad and not is_lora:
            raise RuntimeError(f"Non-LoRA decoder param must be frozen but is trainable: {name}")
```

#### C. model_first_hop.py PETFlowDiTFirstHop.__init__ — 调用 LoRA wrapper

在 `if self.freeze_rae:` 块之后（line ~286）加：

```python
# V18: optionally wrap decoder with LoRA (when freeze_rae=false + decoder_lora.enabled)
self.rae_frozen = None  # set below if KL pull-back enabled
self._decoder_lora_params: list = []
if not self.freeze_rae:
    from .decoder_lora import wrap_decoder_with_lora
    self._decoder_lora_params, wrapped_names = wrap_decoder_with_lora(self.rae, cfg)
    # Set RAE to eval globally; LoRA-wrapped Linear modules ignore this
    self.rae.eval()
    # Build reference frozen decoder if KL pull-back enabled
    kl_cfg = cfg.get("loss", {}).get("decoder_kl_pullback", {})
    if bool(kl_cfg.get("enabled", False)):
        # Re-build a SECOND RAE copy as frozen reference
        self.rae_frozen = build_rae(cfg, device)
        for p in self.rae_frozen.parameters():
            p.requires_grad_(False)
        self.rae_frozen.eval()
```

#### D. train_first_hop.py — 把 LoRA params 加入 optimizer 第三个 param group

在 line ~1444 `param_groups = []` 之后加：

```python
# V18: separate LoRA param group with much smaller LR
decoder_lora_params = list(getattr(model, "_decoder_lora_params", []))
decoder_lora_ids = {id(p) for p in decoder_lora_params}
# remove LoRA params from first_hop_params to avoid double-counting
first_hop_params = [p for p in first_hop_params if id(p) not in decoder_lora_ids]
```

然后在两个现有 param group 后面追加：

```python
if decoder_lora_params:
    decoder_lr_mult = float(opt_cfg.get("decoder_lr_mult", 0.00625))
    decoder_wd = float(opt_cfg.get("decoder_weight_decay", 0.0))
    param_groups.append({
        "name": "decoder_lora",
        "params": decoder_lora_params,
        "lr": base_lr * decoder_lr_mult,
        "lr_scale": decoder_lr_mult,
        "weight_decay": decoder_wd,
    })
    print(f"[optimizer] decoder_lora param group: n_params={len(decoder_lora_params)}, "
          f"lr={base_lr*decoder_lr_mult:.3e} (mult={decoder_lr_mult:.4f})", flush=True)
```

#### E. train_first_hop.py main loop — 加 KL pull-back loss

在 total_loss 组合附近（line ~2066）加：

```python
# V18: KL pull-back loss (only when LoRA + KL enabled)
loss_kl = total_loss.new_zeros(())
lambda_kl = 0.0
if getattr(model, "rae_frozen", None) is not None:
    kl_cfg = cfg.get("loss", {}).get("decoder_kl_pullback", {})
    kl_warmup = int(kl_cfg.get("warmup_steps", 0))
    if step >= kl_warmup:
        lambda_kl = float(kl_cfg.get("lambda_kl", 0.0))
        if lambda_kl > 0:
            from pet_lr.decoder_lora import compute_kl_pullback_loss
            # use GT latent from main batch's hop0 (z_dst = D20 GT)
            z_gt = main_batch["z_dst"]
            loss_kl = compute_kl_pullback_loss(
                rae_lora=model.rae,
                rae_frozen=model.rae_frozen,
                z_gt=z_gt,
                crop_size=int(cfg["data"].get("image_size", 224)),
            )
        # optionally also use_pred_latent → use main_out["z_pred"]; left as future work
```

然后在 `total_loss = (...)` 求和中加 `+ lambda_kl * loss_kl`。

并在 metrics.jsonl 写入 `{"loss_kl": float(loss_kl), "lambda_kl": lambda_kl}`。

#### F. Resume logic — 处理 LoRA params 不在 V7 ckpt 里

train_first_hop.py 的 resume 路径要允许：

- LoRA params (lora_A / lora_B) **不在 ckpt 里**（V7 ckpt 没有），用 LoRA wrapper 初始值
- Decoder base params **在 ckpt 里**且与 LoRA 共存（base 装载，LoRA 用 fresh init）

具体做法：load state_dict with `strict=False`，然后 verify：

```python
missing_keys, unexpected_keys = model.load_state_dict(sd['model'], strict=False)
# Allowed missing: lora_A / lora_B (fresh init)
allowed_missing = [k for k in missing_keys if k.endswith(".lora_A") or k.endswith(".lora_B")]
unexpected_missing = [k for k in missing_keys if k not in allowed_missing]
if unexpected_missing:
    raise RuntimeError(f"Resume found unexpected missing keys: {unexpected_missing[:5]}")
if unexpected_keys:
    raise RuntimeError(f"Resume found unexpected ckpt keys: {unexpected_keys[:5]}")
print(f"[resume] OK from {resume_from}: allowed_missing(LoRA)={len(allowed_missing)}, "
      f"expected_keys_loaded={len(sd['model']) - len(unexpected_keys)}")
```

#### G. save_checkpoint — 把 LoRA params 包含进保存

确认 `model.state_dict()` 自动包含 `lora_A / lora_B`（因为 LoRAAdapter 是 nn.Module 子类，注册了 nn.Parameter）。但要在 ckpt meta 加：

```python
"decoder_lora_enabled": getattr(model, "rae_frozen", None) is not None,
"decoder_lora_wrapped_count": len(getattr(model, "_decoder_lora_params", [])),
```

### 4.3 Day 1 完成判定（在跑 V18 之前必须验证）

```bash
# 1. yaml parses (already verified)
python3 -c "import yaml; c = yaml.safe_load(open('review/0517/V18_decoder_lora/V18_decoder_lora.yaml')); print('OK')"

# 2. decoder_lora.py imports cleanly
python3 -c "from pet_lr.decoder_lora import LoRAAdapter, wrap_decoder_with_lora; print('OK')"

# 3. train_first_hop.py instantiates with V18 yaml (no assert errors)
CUDA_VISIBLE_DEVICES=0 timeout 60 /home/qujiaxiang/.conda/envs/rae/bin/python \
    train_first_hop.py --config review/0517/V18_decoder_lora/V18_decoder_lora.yaml --dry-run 2>&1 | tee /tmp/v18_dryrun.log

# 4. Verify step 0 output == V7 best.pt output (zero-init LoRA sanity)
#    This is critical: if not exactly equal, init_scale_zero is broken.
#    Run a small eval before V18 step 1 actually moves any weights.
```

如果步骤 4 失败（PSNR != V7 baseline），**绝对不要** launch training。LoRA 初始化有 bug。

---

## §5 — Day 2 (Launch V18 + V14)

### 5.1 V18 launch

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# Pre-flight checks
# - Verify output dir doesn't exist (require_fresh_output_dir is set in yaml)
ls /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run 2>&1
# If exists, you forgot to remove a failed run — investigate before deleting

# Launch
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p review/0517/V18_decoder_lora/logs
CUDA_VISIBLE_DEVICES=0 nohup /home/qujiaxiang/.conda/envs/rae/bin/python \
    train_first_hop.py --config review/0517/V18_decoder_lora/V18_decoder_lora.yaml \
    > review/0517/V18_decoder_lora/logs/V18_train_${TS}.log 2>&1 &

# After ~30 min, sanity check the log:
tail -50 review/0517/V18_decoder_lora/logs/V18_train_${TS}.log | grep -E "step=|decoder_lora|loss_kl"
# Should see:
# - [decoder_lora] wrapped N Linear modules across last 2 blocks of ...
# - [optimizer] decoder_lora param group: n_params=..., lr=5.000e-07
# - First few [train] lines with loss_kl small (< 0.01 expected if zero-init worked)
```

### 5.2 V14 launch（GPU 1, 同时跑）

```bash
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p review/0516/V14_true_d_pure/logs
CUDA_VISIBLE_DEVICES=1 nohup /home/qujiaxiang/.conda/envs/rae/bin/python \
    train_first_hop.py --config review/0516/V14_true_d_pure/V14_v7_seed1337.yaml \
    > review/0516/V14_true_d_pure/logs/V14_train_${TS}.log 2>&1 &
```

### 5.3 不允许的事

- **不允许** Day 2 之后再发 Round 4 peer review prompt
- **不允许** Day 2 之后再设计 P0-6 / P0-7 / 任何新 diagnostic
- **不允许** 修改 V18 yaml 的预注册 success/kill 阈值（§2.3）

### 5.4 允许的事（在 V18 跑的 7 天内）

- 写论文 outline，锁 V7 为 baseline
- 整理 V11/V13/V15/V17 系列的 retraction summary 进 [REVIEW_INTEGRATION_20260516.md](../0516/REVIEW_INTEGRATION_20260516.md)
- monitor V18 训练曲线，第 20K step 触发 KILL 阈值就立即停训

---

## §6 — Day 9 — V18 + V14 完成后的评估

```bash
# 标准 full-val PSNR_clip3 评估（复用现有工具）
for EXP in V18 V14; do
    case $EXP in
        V18) BASE=/data_2/.../review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora
             CFG=review/0517/V18_decoder_lora/V18_decoder_lora.yaml ;;
        V14) BASE=/data_2/.../review_0516_runs/V14_true_d_pure/run/first_hop_224_v14_v7_seed1337
             CFG=review/0516/V14_true_d_pure/V14_v7_seed1337.yaml ;;
    esac
    for KIND in last best; do
        CKPT=$BASE/${KIND}.pt
        CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python \
            eval_first_hop_224_clip3.py --config $CFG --checkpoint $CKPT \
            --max-slices 0 --decode-mode both \
            --out-dir review/0524/eval_clip3/${EXP}_${KIND}
        # Also ROI panel
        CUDA_VISIBLE_DEVICES=0 /home/qujiaxiang/.conda/envs/rae/bin/python \
            tools/eval_roi_psnr.py --config $CFG --checkpoint $CKPT \
            --max-slices 0 --out-dir review/0524/roi_psnr/${EXP}_${KIND}
    done
done
```

### 评估 V18 vs V7（pre-registered judgment）

| 测量 | 判定 |
|---|---|
| ΔNORMAL PSNR_clip3 (V18@200K - V7@160K) ≥ +0.30 dB | **SUCCESS** → V18 进 paper，启动 V18 variants（rank sweep, blocks sweep） |
| ΔNORMAL ∈ [+0.05, +0.30] dB | **PARTIAL** → decoder LoRA 方向对但 hyperparams 次优；下一步是 rank/alpha/blocks 微调 |
| ΔNORMAL ∈ [-0.05, +0.05] dB | **NULL** → LoRA 部分解冻不够；需要尝试全解冻 last block 或 unfreeze 更多层 |
| ΔNORMAL < -0.05 dB OR KL drift > 0.1 | **REGRESSION** → V18 撤回；image_aux 与 decoder 路径都不动，反思更深结构问题 |

### 评估 V14（d_pure）

| 测量 | 用途 |
|---|---|
| ΔPSNR_clip3 (V14 - V7) on each stage × ROI metric | 第一次的真 d_pure。所有后续 SNR 判定要用这个数 |
| V14 NORMAL d_pure vs V7-V6_NOISE estimate 0.026 dB | 比较有多大差异；如果真 d_pure >> 0.026，之前 V11/V17 撤回的 "stat significant" 论据要重新检视 |

---

## §7 — 给团队的简短诚实总结

**真正的事实**：6 天前 V7 训练完成那天，跑这份 runbook 的 §3 Phase A/B/C（≤ 1 GPU-hour）就能得出"image_aux 已饱和、freeze_rae 是结构 ceiling"。我们用 3 轮 peer review、3000 行 markdown 才到达同一结论。

**这次必须落地的纪律**：
1. Day 2 launch V18 + V14（GPU 0 + GPU 1）
2. Day 9 评估，按 §6 表格做 binary judgment，不许加 nuance
3. 任何 "中间结果不清晰" 的处理：按预注册阈值走，不能事后改阈值
4. 不允许再设计 Round 4 prompt
