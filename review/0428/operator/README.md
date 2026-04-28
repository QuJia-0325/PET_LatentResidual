# 0428 实验部署：V6 Transport-First + V5 归因闭环

## 总览

本次部署包含 **2 条独立实验线**，可并行执行（如有 2 GPU）或串行执行（1 GPU 优先 V6）。

```
实验线 A（V6 主线）：  01_v6_train.sh → 02_v6_monitor.sh → 03_v6_fullval.sh
实验线 B（V5 归因）：  04_v5_attribution.sh（补齐 0427 遗留）
```

---

## 实验架构

### 当前项目状态

| 版本 | 状态 | 关键结果 |
|---|---|---|
| V3 200K | ✅ 已完成 | best step 86800, NORMAL PSNR=36.87 dB, val_chain_normal_mse=0.000165 |
| V4 SF-pair | ❌ 方案否决 | SF 在 alpha≥0.15 时单调退化 |
| V5 rollout-heavy resume | ❌ 方案否决 | step 93200 phase transition（rolling-val 窗口回卷混淆 + resume LR 混淆） |
| V5 null-control | ⏸ 0427 启动，待确认完成 | 分离 V5 loss 改动 vs resume/LR 影响 |
| **V6 transport-first** | 🆕 **本次启动** | from scratch 200K, pair_weight=15, lambda_roll 0→4 |

### V6 的核心假设

V3 在 step 86800 达到 best（rollout 梯度占比 32%），之后 image_aux 重新主导（88-93%），模型 plateau。V6 通过 `pair_weight=15` + `lambda_roll=4.0` 强制维持 transport 梯度主导，预期推迟或消除 plateau。

### V6 与 V3 的 9 个参数差异

| # | 参数 | V3 值 | V6 值 | 说明 |
|:-:|---|---|---|---|
| 1 | `loss.pair_weight` | 1.0（隐式） | **15.0** | 核心改动 |
| 2 | `rollout.lambda_end` | 0.25 | **4.0** | 16× 增强 |
| 3 | `rollout.lambda_start` | 0.02 | **0.0** | Phase I 无 rollout |
| 4 | `rollout.warmup_ratio` | 0.10 | **0.25** | Phase I 延长到 50K |
| 5 | `rollout.ramp_ratio` | 0.30 | **0.50** | Phase II 100K ramp |
| 6 | `rollout.lambda_scale_mode` | alpha_floor | **none** | 纯线性 ramp |
| 7 | `rollout.step_weights` | [1.30,1.20,1.10,1.00] | **[0.5,2.0,1.5,1.0]** | 中重押 hop1 |
| 8 | `transport.pair_loss_weights` | [1.20,1.10,1.05,1.00] | **[2.5,1.0,1.0,1.0]** | 头重 hop0 |
| 9 | `image_aux.lambda_*` | 0.005→0.12 | **固定 0.04** | 压制 image |

**其余所有参数（模型架构、数据、优化器、EMA、velocity_rebalance 等）与 V3 完全一致。**

---

## 实验线 A：V6 Training

### 脚本

| 脚本 | 用途 | 预计耗时 |
|---|---|---|
| `01_v6_train.sh [GPU_ID]` | 启动 V6 200K from scratch | ~5-6 天 |
| `02_v6_monitor.sh` | 读取 metrics JSONL，自动检查 Go/No-Go | <1 秒 |
| `03_v6_fullval.sh [GPU_ID]` | V6 best.pt 全量 eval + V3 baseline 对比 | ~40 分钟 |

### 启动命令

```bash
# 1. 启动训练（nohup 后台运行）
cd /home/qujiaxiang/project/PET_LatentResidual
nohup bash review/0428/operator/01_v6_train.sh 0 > /dev/null 2>&1 &

# 2. 定期检查 Go/No-Go（训练运行中随时可执行）
bash review/0428/operator/02_v6_monitor.sh

# 3. 训练完成后全量评估
bash review/0428/operator/03_v6_fullval.sh 0
```

### Go/No-Go 检查点

训练启动后，按以下时间节点用 `02_v6_monitor.sh` 检查：

| 检查点 | 预计到达时间 | PASS 条件 | FAIL 动作 |
|---|---|---|---|
| **+1K** | ~30 分钟 | pair_loss < 5e-4, 无 NaN | abort，检查代码 |
| **+10K** | ~5 小时 | pair_frac > 70% | abort，pair_weight 未生效 |
| **+50K** | ~30 小时 | img_frac < 30% | abort，image 仍主导 |
| **+130K** | ~78 小时 | img_frac < 30%, img_raw 无上升趋势 | 考虑停止实验 |
| **+150K** | ~90 小时 | transport ≥ 60% | abort，分析日志 |
| **+200K** | ~120 小时 | val_chain_normal_mse < 0.000150 | 看 fullval 绝对值 |

### 三阶段训练 schedule

```
Phase I  (step 0-50K):     alpha=0, lambda_roll=0    → 纯 GT pair 学习
Phase II (step 50K-150K):  alpha 0→1, lambda 0→4.0   → rollout 渐进接管
Phase III(step 150K-200K): alpha=1, lambda=4.0        → 纯 Pred chain 精修
```

### 预期的 metrics JSONL 关键字段

```json
{
  "pair_loss_weight": 15.0,        // V6 新增字段，应恒等于 15.0
  "pair_frac": 0.80,               // Phase I 应 70-90%
  "roll_frac": 0.00,               // Phase I 应 ~0%
  "img_frac": 0.20,                // Phase I 应 10-30%
  "lambda_roll": 0.0,              // Phase I 应 0
  "lambda_img": 0.04,              // 全程固定
  "alpha": 0.0                     // Phase I 应 0
}
```

---

## 实验线 B：V5 归因闭环

### 背景

0427 启动了以下实验，状态可能未完成：

| 实验 | 脚本 | 状态 |
|---|---|---|
| V3 fullval | 0427/operator/01_fullval_eval.sh | **需确认** |
| V5 fullval | 同上 | **需确认** |
| V5 null-control 训练 | 0427/operator/03_null_control.sh | **需确认** |
| Null-control fullval | 0427/operator/05_fullval_null_control.sh | **需确认** |

### 执行

```bash
# 检查/补齐 V3/V5 fullval；如果 null-control 已完成，也会补齐 null-control fullval
bash review/0428/operator/04_v5_attribution.sh 0
```

脚本会自动检查 0427 的结果是否存在，仅执行未完成的 fullval 部分。
如果 null-control 训练还没有完成，需要先运行 `review/0427/operator/03_null_control.sh`。

### V5 归因结果如何影响 V6

| V5 归因结果 | 对 V6 的影响 |
|---|---|
| V5 fullval 与 V3 持平 → V5 确实失败 | V6 继续，V5 的 resume 策略确认无效 |
| V5 fullval 优于 V3 → V5 有改善 | V6 继续，但需分析 V5 改善的机制来源 |
| Null-control 也退化 → resume 本身有问题 | V6 继续（from scratch 不受影响） |
| Null-control 持平 → V5 loss 改动是退化原因 | V6 参考，但 V6 的权重配方完全不同 |

**无论 V5 归因结果如何，V6 pilot 都不被阻塞**——V6 是 from scratch 训练，不继承 V5 的任何状态。

---

## GPU 分配（2 GPU 并行）

```
GPU 0: V6 训练（01_v6_train.sh 0）         ← 主线，~5-6 天
GPU 1: V5 归因（04_v5_attribution.sh 1）    ← 并行，含 null-control 训练 ~30h + fullval
```

### 启动命令

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# GPU 0: V6 训练（后台）
nohup bash review/0428/operator/01_v6_train.sh 0 > /dev/null 2>&1 &

# GPU 1: V5 null-control 训练（后台）
# 注意：如果 null-control 训练未完成，需要先启动 null-control
#   bash review/0427/operator/03_null_control.sh 1
# null-control 训练完成后再跑归因：
#   bash review/0428/operator/04_v5_attribution.sh 1
nohup bash review/0427/operator/03_null_control.sh 1 > /dev/null 2>&1 &
```

### V5 归因完整流程（GPU 1）

```
Step 1: null-control 训练（~30h）
  bash review/0427/operator/03_null_control.sh 1

Step 2: V5 归因 fullval（null-control 训练完成后）
  bash review/0428/operator/04_v5_attribution.sh 1

Step 3: GPU 1 空闲 → 待命，等 V6 完成后跑 fullval
  bash review/0428/operator/03_v6_fullval.sh 1
```

---

## 代码前提

以下改动已合并到 `train_first_hop.py`（`loss.pair_weight` 实现）：

```python
# L1361-1365: config 解析
pair_loss_weight = float(cfg["loss"].get("pair_weight", 1.0))

# L1919: total_loss
total_loss = pair_loss_weight * pair_losses["total"] + ...

# L1936: logging fraction
pair_weighted = pair_loss_weight * pair_losses["total"]

# L2138: JSONL
"pair_loss_weight": pair_loss_weight,
```

**验证方法**：在服务器上执行 `grep -n "pair_loss_weight" train_first_hop.py`，应看到 L1361、L1362、L1363、L1364、L1365、L1919、L1936、L2138 共 8 处。

---

## Config 文件

| Config | 路径 | 用途 |
|---|---|---|
| V6 | `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` | V6 200K from scratch |
| V3（baseline） | `configs/pet_flow/pet_flow_first_hop_224_200k_transport_v3.yaml` | 对比基准 |
| V5 rollout-heavy | `configs/pet_flow/pet_flow_first_hop_224_v5_rollout_heavy.yaml` | 归因对比 |
| V5 null-control | `configs/pet_flow/pet_flow_first_hop_224_v5_null_control.yaml` | 归因对比 |

---

## 成功标准（V6 完成后）

| 指标 | V3 200K best | V6 目标 | 说明 |
|---|---|---|---|
| val_chain_normal_mse | 0.000165 | < 0.000150 | 主判据（绝对 metric） |
| full-val NORMAL PSNR | 36.87 dB | > 37.5 dB | +0.6 dB |
| transport weighted fraction @ 150K+ | 7-45% | ≥ 60% | 辅助判据 |
| best step 位置 | 86800（43%） | > 150K（75%） | plateau 推迟 |
