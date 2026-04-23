# Transport v3 审查与实验指导 — 2026-04-23

## 1. velocity_rebalance 实现审查结论

### 审查结果：✅ 实现正确，可安全使用

**完整链路追踪**：

```
compute_pair_losses() 内部:
  loss_velocity = (vel_err * sample_weights).mean()
  loss_endpoint = (end_err * sample_weights).mean()
  
  ratio = loss_endpoint.detach() / loss_velocity.detach()   ← detach 阻止二阶梯度
  rebalance_scale = clamp(sqrt(ratio), 1.0, 4.0)            ← 只放大不缩小
  
  velocity_weight_eff = v_weight * rebalance_scale
  total = velocity_weight_eff * loss_velocity + endpoint_weight * loss_endpoint
  
  → pair_losses["total"] 包含 rebalance 后的值
  → 进入 total_loss → backward() → 梯度正确
```

**关键设计点**：
- `clip_min=1.0`：velocity 权重只增不减（当 endpoint >> velocity 时放大 velocity）
- `clip_max=4.0`：防止极端放大
- `sqrt_ratio`：平滑缩放（ratio=16 → scale=4.0；ratio=4 → scale=2.0）
- `detach()`：rebalance_scale 是常数标量，不参与 backward

**监控**：`metrics_jsonl` 中 `vel_reb` 字段记录每步的实际 scale 值

---

## 2. v3 Config 4 项修改详情

### 修改 1: pair_loss_weights 反转

```yaml
# v2 (旧 — 倒置):
pair_loss_weights: [1.00, 1.05, 1.10, 1.20]

# v3 (新 — 正确):
pair_loss_weights: [1.20, 1.10, 1.05, 1.00]
```

**原因**：D50→D20 承担 43.9% 的 latent 位移（最大），应获得最高权重。

### 修改 2: rollout step_weights 反转

```yaml
# v2 (旧):
step_weights: [1.00, 1.10, 1.20, 1.30]

# v3 (新):
step_weights: [1.30, 1.20, 1.10, 1.00]
```

**原因**：与 pair_loss_weights 对齐，优先级给第一跳。

### 修改 3: val_multi_objective D20 权重提升

```yaml
# v2 (旧 — D20 只占 5%):
best_metric_terms:
  - name: val_chain_d20_mse
    weight: 0.15

# v3 (新 — D20 占 15%):
best_metric_terms:
  - name: val_chain_d20_mse
    weight: 0.50
```

**原因**：D20 在选优中只占 5%（vs NORMAL 50%），导致 best.pt 完全忽视 hop0 质量。

### 修改 4: velocity_rebalance 启用

```yaml
# v2 (旧):
velocity_rebalance:
  enabled: false

# v3 (新):
velocity_rebalance:
  enabled: true
  mode: sqrt_ratio
  clip_min: 1.0
  clip_max: 4.0
  eps: 1.0e-12
```

**原因**：endpoint loss 可能数值上远大于 velocity loss，导致 velocity 监督被淹没。

---

## 3. 实验运行指导

### Config 文件

```
configs/pet_flow/pet_flow_first_hop_224_50k_transport_v3.yaml
```

### 运行命令

```bash
# GPU-3: Transport v3 (50K steps, ~16h)
CUDA_VISIBLE_DEVICES=3 TQDM_DISABLE=1 \
python -u train_first_hop.py \
    --config configs/pet_flow/pet_flow_first_hop_224_50k_transport_v3.yaml \
    2>&1 | tee review/0423/logs_train/transport_v3_train_gpu3.log
```

注意：
- `require_fresh_output_dir: true` — 输出目录不能已存在
- 不需要 `--resume`（fresh start from smoke backbone）
- `print_train_line_with_pbar: false` + `progress_bar: true` — 训练行不打印，需查 metrics.jsonl

### 日志归档

训练完成后将以下文件归档到 `review/0423/`：

```bash
# 训练日志
cp /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_transport_v3/*.log review/0423/logs_train/

# 关键 metrics
cp /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_transport_v3/metrics.jsonl review/0423/artifacts/transport_v3_metrics.jsonl
```

---

## 4. 监控要点（训练期间检查）

### 必须监控的 metrics_jsonl 字段

| 字段 | 预期 | 异常信号 |
|------|------|---------|
| `vel_reb` | 初期 ~2-4，后期 ~1-2 | 持续 =4.0 说明 endpoint 仍远大于 velocity |
| `vel_w` | 初期 ~2-4（= v_weight × vel_reb），后期 ~1-2 | 持续 =1.0 说明 rebalance 未生效 |
| `pair_frac` | 0.4-0.7 | > 0.9 说明 rollout/img loss 太小 |
| `roll0` | 应随训练下降 | 不下降说明 hop0 rollout 未学习 |
| `gate_pix` | 应增长（pixel forcing 仍 enabled） | 趋向 floor 说明分支无效 |

### 快速健康检查命令

```bash
# 检查 vel_reb 值（rebalance 是否生效）
python3 -c "
import json
with open('/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_transport_v3/metrics.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('event') == 'train' and d['step'] % 1000 == 0:
            print(f'step={d[\"step\"]:6d} vel_reb={d.get(\"vel_reb\",0):.3f} vel={d.get(\"vel\",0):.6f} end={d.get(\"end\",0):.6f}')
"
```

---

## 5. 成功判据

### 与旧 Scheme C 对比（full-val rerank 后）

| 指标 | Scheme C best | v3 期望 |
|------|-------------|--------|
| transport_avg | 36.206 | **> 36.30**（改善 > 0.1 dB → 4 项修复有效） |
| D20 PSNR | 35.569 | **> 35.65**（D20 权重修正后应优先改善） |
| val_select_score | 0.000532 | **< 0.000520** |

### 判断逻辑

- transport_avg **> 36.30 dB** → v3 修复有效，继续 200K 步实验
- transport_avg **≈ 36.20 dB** → 权重修复不是瓶颈，转向 backbone LR / training steps
- transport_avg **< 36.10 dB** → 某项修复有副作用，需拆分排查

---

## 6. 当前 GPU 状态

| GPU | 任务 | 状态 |
|-----|------|------|
| GPU-1 | backbone_lr1 | 进行中 |
| GPU-3 | **transport_v3 (新)** | 待启动 |

---

## 7. 权重倒置问题的量化证据

```
D50→D20 latent 位移 (σ×dt):  0.028902 = 43.9% of total （最大）

训练权重分配:
  pair_loss:    D50→D20 = 23.0% （最小） ← 倒置!
  rollout:      D50→D20 = 21.7% （最小） ← 倒置!
  val 选优:     D50→D20 =  5.0% （远低于 NORMAL 50%） ← 严重倒置!

v3 修正后:
  pair_loss:    D50→D20 = 27.6% （最大） ✓
  rollout:      D50→D20 = 28.3% （最大） ✓
  val 选优:     D50→D20 = 14.9% （从 5% 提升 3×） ✓
```
