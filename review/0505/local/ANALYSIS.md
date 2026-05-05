# V7 & V8 160K Plan-F Ablation — 实验设计与说明

**Date**: 2026-05-05 (初稿 50K sanity → Plan D resume → Plan F 150K from-scratch → **Plan F + Fix F: 160K from-scratch**)
**Author**: 本地 Claude
**Status**: 文档 + yaml 就绪，**未运行**（需远程 GPU 执行；pre-flight checklist 见 [review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md](../../plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md) §9）。
**Scope**: 两组 160K from-scratch ablation 用于回答 §1 列出的两个核心问题。

**安全说明**: 原设计是 50K sanity。2026-05-05 经 4 位 reviewer 审核 + 2 位 Plan D reviewer 审核
后需求升级为 150K；随后第三轮 cross-AI peer review (Agent1 + Agent2) 进一步提出 **Fix F**：V6
主训的 `save_interval=20000` 表示 V6@step_150000.pt **不存在于磁盘**（V6 step ckpt 仅在
{20K, 40K, ..., 200K} 出现），因此 endpoint 从 150K 调整为 160K —— V6@step_160000.pt 存在，且与
150K 同处于 rollout 平台（lambda_roll=4.0 clamped）。原选择 150K 的两条理由 (rollout ramp 末点
+ V7 step_weights 首次在完整 lambda_roll=4.0 下被检验) 在 160K 仍然成立。Plan D（V7 从
V6@40K resume）被代码审核判为不可行（EMA-swap checkpoint + DataLoader sampler RNG 不连续 +
best_metric_signature 不匹配，三重 BLOCKER）。Plan F + Fix F 是 Plan D 的严格风险子集。

---

## 1. 问题背景

来自 2026-05-05 conversation:

> "你的分析是否解决了你的疑问？" → 用户对 V6 step_weights 的 Grönwall 闭式解符合性质疑后，我（本地 Claude）补充诊断了 image_aux 与 Grönwall 框架的关系。

**两个待回答的问题**:

| Q | 假设 | 验证方法 |
|---|---|---|
| Q1 | V6 hop2/hop3 step_weight (1.5/1.0) 偏离 Grönwall 闭式解 (2.7/2.04) 是经验选择，**不能**用 image_aux 解释 | V7 = 直接用闭式 raw 权重 [0.6106, 2.0, 2.7041, 2.0423] 训 **160K Plan F (Fix F)**，在 step=160K 全集对比 V6@160K |
| Q2 | image_aux 在压制 hop0 decode 伪影；论文里“为了压制伪影”的叙事还成立吗？V6 拿掉 image_aux 后效果如何？ | V8 = V6 关掉 image_aux，训 **160K Plan F (Fix F)**，在 step=160K 全集对比 V6@160K |

两组 ablation **互不干扰**：V7 只改 step_weights，V8 只改 image_aux；其他超参数（pair_weight=15, λ_roll ramp 0→4, schedule, optimizer, EMA, seed=42）100% 一致。**两者都 from-scratch 训练到 160000 step**（Plan F + Fix F）。

---

## 2. V7 配置：Grönwall 闭式 raw step_weights（σ-norm OFF）

[configs/V7_gronwall_raw.yaml](configs/V7_gronwall_raw.yaml)

### 数学推导（来自 [review/plan/ARCHITECTURE_ANALYSIS_20260501.md §21.1](../../plan/ARCHITECTURE_ANALYSIS_20260501.md) + §18.3.3）

clinical-prior 加权 validation 目标：

$$ J = \sum_{k=1}^{4} \beta_k \cdot \|c_k\|^2,\quad \beta = [0.5,\, 0.45,\, 0.9,\, 1.5] $$

多 hop Grönwall 误差传播（假设 hop-wise Lipschitz $L_j \approx 1$，已 §17.3 实测验证为 [0.96, 1.13]）下，最优 raw 域 step_weights：

$$ w_j^* \propto \frac{\Phi_j}{(\sigma_j \cdot dt_j)^2},\quad \Phi_j = \sum_{k=j+1}^{4} \beta_k $$

代入 $\sigma_j \cdot dt_j = (0.0289, 0.01473, 0.01163, 0.01058)$：

| hop $j$ | $\Phi_j$ | $(\sigma_j dt_j)^2$ | $w_j^*$ unnorm | $w_j^*$ (hop1=2.0) | V6 实际 |
|---|---|---|---|---|---|
| 0 | 3.35 | 8.35e-4 | 4 010 | **0.6106** | 0.5 |
| 1 | 2.85 | 2.17e-4 | 13 135 | **2.0000** | 2.0 |
| 2 | 2.40 | 1.35e-4 | 17 759 | **2.7041** | 1.5 |
| 3 | 1.50 | 1.12e-4 | 13 413 | **2.0423** | 1.0 |

**与 V6 偏差**：hop0 +22%，hop1 0%，hop2 **+80%**，hop3 **+104%**。V6 在 hop2/hop3 显著低于闭式预测。

### 为什么不用 D_closed_form (σ-normalize ON 路径)

[review/0502/configs/D_closed_form.yaml](../../0502/configs/D_closed_form.yaml) 已有"Grönwall 闭式 step_weights"实验，但走 σ-normalize ON 路径（step_weights = [3.58, 0.79, 0.41, 0.21] × σ-norm 除法）。在 fp64 下 D 与 V7 数学等价，**但**：

1. A/B sanity 2026-05-03 **FAIL**（[review/0505/operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md](../operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md)）：σ-normalize ON 在 fp32 + warn-only deterministic 训练下 50K 漂移 ~10%，存在 trajectory drift 噪声。
2. V7 走 σ-normalize OFF 路径，**仅一行 step_weights 修改 vs V6**，是更纯粹的 ablation —— 不受 σ-normalize 实现 bug 污染。

### 关于 Σw 的误会（已澄清，2026-05-05）

早期草稿曾认为 V7 Σw = 7.357 vs V6 Σw = 5.0 会引入 effective lambda_roll 混淆（"V7 比 V6 少 32% rollout 压力"）。在阅读 [pet_lr/rollout_first_hop.py L120](../../../pet_lr/rollout_first_hop.py#L120) 后该说法已被推翻：

```python
total = (stacked * w).sum() / w.sum().clamp_min(1e-8)
```

代码是加权**均值**而非加权**和**：分母 Σw 与分子同阶抑制，所以 Σw 完全抑制。V6 (Σw=5.0) 与 V7 (Σw=7.357) 面临的 effective lambda_roll 严格一致。**V7 只隔离 step_weights SHAPE 一项**；原计划中的 V7b（同 shape 但 Σ=5）与 V7 数学上零差别，已取消。

---

## 3. V8 配置：拿掉 image_aux

[configs/V8_no_image_aux.yaml](configs/V8_no_image_aux.yaml)

### 修改点

1 行配置变更：`training.image_aux.enabled: true → false`。代码已支持此 flag（[train_first_hop.py L1618 + L1968-1985](../../../train_first_hop.py#L1618)），无需任何代码改动。

### 关闭 image_aux 后的代码行为

- [L1968-1985](../../../train_first_hop.py#L1968) ：`img_enabled=False` 走 zero/nan 分支，`loss_img = 0`，不进 backward。
- [L2027 / L2044](../../../train_first_hop.py#L2027)：`lambda_img · loss_img` 自动归零，`img_frac=0`。
- [L2152](../../../train_first_hop.py#L2152)：`img_loss_zero_eps` watchdog 守住 `if img_enabled` 分支，关闭后不会触发 RuntimeError。
- hop0_aux dataloader 仍运行（保 dataset 顺序与 V6 对齐），仅 image-side forward 跳过。

### V8 的 SANITY CONFOUND

V6 image_aux 通过 backbone backward 间接驱动 `gate_pix` / `lambda_hop_0`（pixel forcing 分支）。关闭 image_aux 后，pixel forcing 分支只能从 `pair_loss` 间接拿到信号，可能导致 gate_pix / lambda_hop_0 漂到 floor（"dead branch"）。这本身是**有意义的 ablation 信号**：如果 dead，说明 pixel forcing 是 image_aux 驱动的，不是真正"独立的安全网"。

**记录指标**：V8 训练 log 里 `gate_pix`、`lambda_hop_0`、`v_hop_abs_hop0` 在 step=50K / 100K / 150K / 160K 四个点的值，与 V6 对应 step 同点比对。`dead_branch_watch_steps=160000` 涵盖整个 Plan F + Fix F 训练窗口。

### V8 不能回答的问题

V8 **不能**直接验证“image_aux 是否压制 decode 伪影”。chain MSE 在 latent 域，伪影在 decode 域。需要补一组**定性视觉评估**（V6_last vs V8_last 在固定 validation 子集上 decode → 人工对比）。这是 V8 后续行动，**不在本 Plan F + Fix F 160K 范围**。

---

## 4. 对比基线：V6@step_160000 全集评分（不重训 V6）

V6 200K 主训练已完成（[review/0505/operator/logs_train/v6_transport_first_train_200k.log](../operator/logs_train/v6_transport_first_train_200k.log) + metrics.jsonl）。与初稿 50K sanity 不同，Plan F + Fix F 的决策点是 **step 160000**，anchor 为 V6@step_160000.pt 的全集 full-val 评分。

> **为何不是 150K？** V6 主训 `save_interval=20000` → V6 step ckpt 仅在 {20K,40K,...,200K} 出现，V6@step_150000.pt **不存在**。cross-AI peer review BLOCKER。Fix F 为调整为 160K —— V6 在 step 150K 和 160K 都处于 rollout 平台（lambda_roll=4.0 clamped），未改变实验逻辑。Cost：+20K GPU/arm × 2 = +40K，约 +6.7%。

**启动前必须确认**（operator pre-flight，见 [PLAN_F](../../plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md) §9）：

```bash
# 1. V6@160K ckpt 存在
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt

# 2. V6@160K full-val 已跑（未跑则需先补 anchor）
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/.../v6_step_160000_fullval.json
```

V6 metrics.jsonl 路径（操作员需根据实际部署确认）：
```
/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl
```

如该路径不准确，操作员设置 `V6_METRICS=<actual_path>` 后跑 `compare`。

---

## 5. 启动顺序

> **启动顺序总览（Q10-A, 2026-05-05）**：本 Plan F 现有 3 个训练 arm（V7、V8、V6-seed1337）+ 1 个 anchor（V6-seed42 已存在）。V6-seed1337 是新增的"纯噪声地板"测量 arm，与 V7/V8 之间**无依赖**，可与 V7/V8 并行（不同 GPU）或串行跑。但 `d_pure` 必须在生成最终 V7/V8 决策表之前算出，否则 PRIMARY 阈值 `d_thr = max(0.10, 1.8 × d_pure)` 没法填值。Stage 0 是 V6-seed1337 启动；Stage 1-4 与原有结构相同（pre-flight → V7 → V8 → 比对）。

### Stage 0 — V6-seed1337 noise-baseline (Q10-A, NEW)

```bash
GPU=<same_uuid_as_V6_seed42> SANITY_STEPS=160000 \
    bash review/0505/local/scripts/run_v7_v8_sanity.sh V6_NOISE
```

预计耗时：与 V7/V8 持平（≈0.80×V6 200K wall-clock）。可在另一块 GPU 上与 V7/V8 并行；GPU UUID 必须与 V6-seed42 训练时一致（Fix 2）。
跑完后立即跑：

```bash
# 评估 V6-seed42 与 V6-seed1337 在 step 160000.pt 上的全集分数（per-slice JSON）
bash review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
     --ckpt /data_2/.../first_hop_224_v6_transport_first/step_160000.pt \
     --tag V6_seed42_step160k --emit-per-slice-json
bash review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
     --ckpt /data_2/.../V6_NOISE/run/first_hop_224_v6_seed1337/step_160000.pt \
     --tag V6_seed1337_step160k --emit-per-slice-json

# d_pure = paired Cohen's d on val_chain_normal_mse, n=7403 unique slices
python3 review/0505/local/scripts/compute_d_pure.py \
     --a V6_seed42_step160k.per_slice.json \
     --b V6_seed1337_step160k.per_slice.json \
     --metric val_chain_normal_mse \
     --out review/0505/local/runs/V6_NOISE/d_pure.json
```

`compute_d_pure.py` 是一个薄包装（scipy paired-t + Cohen's d on per-slice deltas），实现可以延后到 V6-seed1337 完成时再写——简单到不会阻塞 launch。

### Stage 1 — pre-flight check

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# 验证 V7 yaml 数学
python3 - <<'PY'
sigma = [0.009634, 0.002946, 0.000775, 0.000141]
dt = [3, 5, 15, 75]
beta = [0.5, 0.45, 0.9, 1.5]
rho = [(s*d)**2 for s, d in zip(sigma, dt)]
Phi = [sum(beta[j:]) for j in range(4)]
w_un = [Phi[j]/rho[j] for j in range(4)]
scale = 2.0 / w_un[1]
w = [round(x*scale, 4) for x in w_un]
print("V7 expected step_weights:", w)
print("yaml has:", [0.6106, 2.0000, 2.7041, 2.0423])
assert all(abs(a-b) < 1e-3 for a, b in zip(w, [0.6106, 2.0, 2.7041, 2.0423])), "V7 weights drift!"
print("OK")
PY

# 验证 yaml 语法 + 关键 fields（含 max_steps=160000 检查）
python3 - <<'PY'
import yaml
for tag, p in [("V7", "review/0505/local/configs/V7_gronwall_raw.yaml"),
               ("V8", "review/0505/local/configs/V8_no_image_aux.yaml")]:
    c = yaml.safe_load(open(p))
    sw = c["training"]["rollout"]["step_weights"]
    img = c["training"]["image_aux"]["enabled"]
    ms = c["training"]["max_steps"]
    ws = c["training"]["rollout"]["warmup_steps"]
    rs = c["training"]["rollout"]["ramp_steps"]
    ts = c["lr_schedule"]["total_steps_override"]
    bsfei = c["training"].get("best_select_full_eval_interval", 0)
    print(f"{tag}: step_weights={sw} image_aux.enabled={img} max_steps={ms} "
          f"warmup={ws} ramp={rs} lr_total_override={ts} full_eval_interval={bsfei}")
    assert ms == 160000, f"{tag} max_steps must be 160000 (Plan F + Fix F), got {ms}"
    assert ws == 50000 and rs == 100000 and ts == 200000, f"{tag} schedule lock broken"
    assert bsfei == 5000, f"{tag} best_select_full_eval_interval must be 5000"
print("OK")
PY

# 3. V6@160K anchor (Fix F: V6 save_interval=20000 → step_150000.pt missing, use 160K)
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt

# 4. (Fix 2) GPU UUID lock + CUDA / cuDNN / driver 记录 — V7/V8 必须与 V6 在同一块 GPU 上跑
nvidia-smi -L                          # 记录 GPU UUID、与 V6 训练时 GPU UUID 比对
nvidia-smi --query-gpu=driver_version --format=csv
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())'
# 输出全部记入 review/0505/operator/v7v8_env_record_<date>.txt

# 5. (Fix 2) CUBLAS_WORKSPACE_CONFIG 必须设为 :4096:8（launcher 已处理，pre-flight 只记录证据）
echo "$CUBLAS_WORKSPACE_CONFIG"          # 期望：:4096:8
```

#### Stage 1.5 (Fix 2) — V7/V8 step-1000 sentinel

启动后首个 1000 步是低代价检查点：这个阶段 lambda_roll=0、LR 在 warmup，V7/V8 与 V6 在梯度层面应该不可区分（V7 step_weights 只在 rollout active 时生效；V8 image_aux off 只减一项加项 loss，不影响 pair-only forward 的 RNG 轨迹）。运行后对比 metrics.jsonl 第 1000 行：

```bash
# 预期：v7/v8 与 v6 在三个 sentinel 指标上偏差 ≤ 0.5%
for m in val_pair_total val_chain_d20_mse val_select_score; do
    v6=$(jq -r 'select(.step==1000) | .'"$m"  /data_2/.../v6_transport_first/metrics.jsonl | head -n1)
    v7=$(jq -r 'select(.step==1000) | .'"$m"  /data_2/.../V7/run/.../metrics.jsonl | head -n1)
    rel=$(python3 -c "print(abs($v7 - $v6) / max(abs($v6), 1e-12) * 100)")
    echo "step1000 $m  v6=$v6  v7=$v7  rel=$rel%"
done
# 如果任一偏差 > 0.5%，立即 abort、检查环境差异。
```

### Stage 2 — V7 (160K, Plan F + Fix F)

```bash
GPU=0 SANITY_STEPS=160000 \
    bash review/0505/local/scripts/run_v7_v8_sanity.sh V7
```

预计耗时：V6 200K 实测约 X 小时 → 160K ≈ 0.80×X（操作员替换实际数）。含 32 个 full-val中间点（每 5K 一个×7.5分/个）额外约 4h 开销。

### Stage 3 — V8 (160K, Plan F + Fix F)

#### Stage 3.0 — (Fix 4) V8 image_aux RNG diagnostic (belt-and-suspenders)

Agent2 提出 V8 关闭 image_aux 可能跳过消耗 RNG 的代码路径。反证本仓 DiT 无 dropout/DropPath（[pet_lr/pet_flow_dit.py:120](../../../../RAE/RAE/src/pet_flow/models/pet_flow_dit.py#L120) Mlp drop=0；model_utils.py:170-171 NormAttention attn_drop=proj_drop=0；pet_lr/ 无 nn.Dropout），理论上 forward 不动 RNG。但为避免漏洞，launch V8 前先跑一个一欧 hop0 训练步的 RNG 闭环诊断：

```bash
python3 review/0505/local/scripts/diag_v8_rng_invariance.py \
    --config review/0505/local/configs/V8_no_image_aux.yaml \
    --num-batches 1
# 期望输出：enabled=true 与 enabled=false 两路径下， torch.cuda.get_rng_state()
# 在 hop0 forward 后位一致。不一致则说明 DiT 在某处潜藏随机调用 → 必须重新设计
print "OK: V8 image_aux on/off 不影响 RNG轨迹”
```

该 diag 脚本仅跑 1–2 步，耗时 < 30s。只在 V8 首次启动前跑一次。

#### Stage 3.1 — V8 launch

```bash
GPU=0 SANITY_STEPS=160000 \
    bash review/0505/local/scripts/run_v7_v8_sanity.sh V8
```

可与 V7 同 GPU 串行，或不同 GPU 并行（GPU=1，但看 §7.3）。

### Stage 4 — 比对

```bash
V6_METRICS=/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/metrics.jsonl \
    bash review/0505/local/scripts/run_v7_v8_sanity.sh compare > review/0505/local/V7_V8_compare.md
```

输出 markdown 表格 → 写进本目录 `V7_V8_compare.md`。

---

## 6. 决策矩阵

### 决策口径：full-val，不是 rolling val

rolling-val 窗口 (64 batch × 8 = 512 slice) 在 V6 main log 里测出 CV 25–28%，任何 ≤5% 的判定阈值都在噪声下。为此：

1. yaml 中设 `best_select_full_eval_interval: 5000`——训练器 fc7446d 每 5K 跨全集 (n=7403) 打分，仅在全集上改善时才写 best.pt。
2. V7/V8 vs V6 的最终判定不看 metrics.jsonl 的单行 rolling 值，而是全集评估赋分：

```bash
# 跑 V6/V7/V8 的 best.pt 和 last.pt 的全集评估
bash review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
     --ckpt /data_2/.../V7/best.pt --tag V7_best
bash review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
     --ckpt /data_2/.../V7/last.pt --tag V7_last
# (对 V8、V6 同样跑)
```

### Pre-registration（Fix 3 + Fix 5 + Q10-A adaptive threshold，cross-AI peer review CONCERN 回应）

为防“全集跨多 checkpoint 选最优”产生的 winner's curse 偏差、在多个 chain 指标上重复测试产生的假阳性 (FWER)，以及历史 V6 vs V6.1 paired Cohen's d=0.067（n=7403, full-val mse_NORMAL, best.pt; 2026-05-06 由本地 per-slice CSV 重算）被算法差异与选择偏差污染的问题，本 Plan F 预注册以下决策结构：

#### Primary endpoint (Fix 3 + Q10-A)

- **指标**：`val_chain_normal_mse` 在 V7@step_160000.pt vs V6-seed42@step_160000.pt 全集评分（V8 同一结构）。
- **起点**：step_160000.pt = last.pt under save_interval=10000；**不受选择偏差影响**。
- **阈值（adaptive，2026-05-05 Q10-A）**：paired Cohen's d ≥ **`max(0.10, 1.8 × d_pure)`** AND chain_normal_mse Δ ≥ 5%，α=0.05。
  - `d_pure` 在 V6-seed1337 跑完后由“V6-seed42@step_160000.pt vs V6-seed1337@step_160000.pt 在 val_chain_normal_mse 上的 paired Cohen's d”实测得出（详 §5 Stage 0）。
  - 为什么 1.8×：与原 Plan F 中的启发式保守裕度一致（“效应至少是噪声的 ~2 倍”），只是现在应用在**干净的**噪声估计上，而不是被污染的 V6/V6.1 d=0.067。
  - **额外实证支撑（2026-05-06，已有数据）**：V6 best.pt mse_NORMAL = 2.4303e-4 > V6 last.pt mse_NORMAL = 2.4270e-4（rolling-val 选的 best 反而比 last 略差），证实 rolling-val best 选择带局部噪声偏差。这从同一 V6 run 内部独立证实了 PRIMARY 必须锁在 `step_160000.pt = last.pt` 的设计选择。来源：[v6_best/last_fullval_psnr_chain_mse_per_slice.csv](../operator/artifacts/)。
  - 为什么 floor 0.10：Power 分析显示 80%-power MDE 在 n=7403 下仅为 0.033（Bonferroni 0.039），阈值控制因素是噪声地板而非 statistical power。设 0.10 为 hard floor 避免 d_pure 面临奇异偏小时 (e.g., ≤ 0.03) 让阈值脱锉。
- **原因**：`best.pt` 源于全集 32 次 (160K/5K) max-search，max(X_1..X_32) 存在随机偏差。微调 V7 best.pt 优于 V6 同期 best.pt 可能仅反映 “32 次选中面”，而非 step_weights 本质改进。

#### Secondary endpoint (Fix 3)

- **指标**：`val_chain_normal_mse` 在 V7@best.pt vs V6-seed42@step_160000.pt。
- **阈值**：Bonferroni-调整后阈值 —— 在 α_local=0.05/2=0.025 上要求 d ≥ **`max(0.15, 2.7 × d_pure)`** 以补偿 32-路 max-search inflation（与 primary 阈值同比例放大：0.15/0.10 = 1.5×）。
- **使用**：primary 阈值报告后作为补充证据；**不能反转 primary 决策**。

#### Multi-metric family (Fix 5)

除 primary 外，有四个 secondary chain 指标同时被报告。使用 Bonferroni 修正 FWER：

| Metric | 角色 | 单点 α | 阈值 |
|---|---|---|---|
| `val_chain_normal_mse` | **PRIMARY** | 0.05 | d ≥ **`max(0.10, 1.8 × d_pure)`** 且 Δ ≥ 5% |
| `val_chain_d4_mse`     | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |
| `val_chain_d10_mse`    | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |
| `val_chain_d20_mse`    | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |
| `val_pair_total`       | secondary  | 0.05/4 = **0.0125** | d ≥ **`max(0.15, 2.7 × d_pure)`** (empirical, scaled) |

注：Family-Wise Error Rate 补偿针对 secondary 家族内部。Primary (`val_chain_normal_mse`) 作为独立决策，保持原始 α=0.05。任一 secondary 结果在 Bonferroni 阈值下反转 primary 决策需起于下轮生产运行（200K）。secondary 与 primary 阈值比例保持 0.15/0.10 = 1.5×，在 d_pure 驱动阶段同样采用 2.7×/1.8× = 1.5×。

考虑到 V6 vs V6.1 的 paired Cohen's d=0.067 (best.pt) / 0.067 (last.pt)（2026-05-06 重算，n=7403, full-val, mse_NORMAL；t=−5.83, p≈5.6e-9）**是被算法差异 (V6.1 rollout-floor) 与选择偏差 (各自 best.pt) 污染的**，Q10-A 增加了 V6-seed1337 跨种子复制走“同算法同 step 仅 seed 差异”路径，得出 `d_pure` 作为干净噪声估计。指定阈值 “decision-margin” 代表三个必须同时满足的条件：

- mean(chain_normal_mse) 改善 ≥ 0.5%（全集）
- paired Cohen's d ≥ `max(0.10, 1.8 × d_pure)`（不是可忽略）
- val_multi_objective 不倒退

### V7 决策（endpoint = step 160000, Plan F + Fix F + Q10-A）

记 `d_thr = max(0.10, 1.8 × d_pure)`（由 V6-seed1337 跨种子跡后计算，文梣完后填入）。

| 160K 全集结果（PRIMARY: V7@step_160000.pt vs V6-seed42@step_160000.pt）| 决策 |
|---|---|
| V7 满足上述三条件且 chain_d20/d10 不劣于 V6 超过 1% | ✅ **跑 V7 200K main**，paper 改用 V7 baseline + 闭式解理论命中（强 contribution） |
| V7 与 V6 的任一指标都在 effect-size 阈值内（Cohen's d < d_thr） | ➖ 结论：V6 step_weights 与 Grönwall 闭式解在该设置下**不可判别**；paper 写成“闭式解与经验选择在本设置下一致”即可 |
| V7 的 chain_normal_mse 劣于 V6 且 Cohen's d ≥ d_thr | ❌ V6 hop2/hop3 deweight 是经验最优；paper 写“Grönwall 给上下界，hop2/hop3 经验微调更优” |

### V8 决策（endpoint = step 160000, Plan F + Fix F + Q10-A）

V8 仅限于测 chain MSE；decode 伪影问题需另外定性实验（OUT OF SCOPE）。下表中 `d_thr = max(0.10, 1.8 × d_pure)`。

| 160K 全集结果（PRIMARY: V8@step_160000.pt vs V6-seed42@step_160000.pt）| 决策 |
|---|---|
| 所有 chain_*_mse(V8) 在 effect-size 阈值内不差于 V6 | ➖ chain MSE 层面 V8 不倒退；是否采纳 V8 **必须** 补定性 decode artifact 对比，本 sanity 不能结论 |
| 所有 chain_*_mse(V8) 不劣于 V6 且 dead_branch 未在 160K 内触发 | ✅ chain 层面绿灯；走 200K main 前**必须**先补 decode artifact 定性评估 |
| 任一 chain_*_mse(V8) 劣于 V6 且 Cohen's d ≥ d_thr | ❌ image_aux 在该阶段对 chain MSE 是有实质贡献的；paper 维持 V6，加这条 ablation 作证据 |
| dead_branch_watch 在 160K 内触发（gate_pix 或 lambda_hop_0 到 floor）| 🔴 V8 设置下 pixel forcing 分支灭灯；消息本身即是结论，paper 里报告“image_aux 是 pixel forcing 分支的唯一梯度源” |

---

## 7. 已知 caveats / 风险

1. **rolling-val 噪声【已代】**：64-batch (×8) = 512-slice rolling val window CV 在 V6 main 上实测 25–28%。为此 yaml 设 `best_select_full_eval_interval: 5000`，训练器（fc7446d）每 5K 一次全集 7403-slice 评估、仅在全集改善时写 best.pt。最终决策调用 [review/0505/scripts/eval_first_hop_fullval_psnr_chain_mse.py](../scripts/eval_first_hop_fullval_psnr_chain_mse.py) 跨 best.pt / last.pt 跑全集，**不看** metrics.jsonl 的单行 rolling 值。

2. **schedule 对齐【已代】**：V6 主训 max_steps=200000，在 step 50000 面临的是 rollout warmup 末点（lambda_roll = 0）、LR cosine 刚开始下降（≈0.97×base）。如果 V7/V8 用 max_steps=160000 + warmup_ratio=0.25，V7/V8@k 会被压缩、与 V6@k 不在同一阶段。为此在两个 yaml 里显式锁定：

   ```yaml
   training:
     rollout:
       warmup_steps: 50000           # 覆盖 warmup_ratio
       ramp_steps: 100000            # 覆盖 ramp_ratio
     lr_schedule:
       total_steps_override: 200000  # 让 cosine 按 200K 解释
   ```

   trainer 中 `_resolve_schedule_steps`（steps_key 优先于 ratio_key）与 `lr_total_steps = override or max_steps`（L1566）均已支持。`get_warmup_cosine_lr`（L62-78）只依赖 `step` / `lr_total_steps` / `warmup_steps`（= 0.15 × 200000 = 30000）。在锁定调度下V7/V8@k 与 V6@k 同阶段：

   | step | lambda_roll | lr | lr/base |
   |---:|---:|---:|---:|
   | 30000 | 0 | 8.00e-5 | 1.000 (warmup ends) |
   | 50000 | 0 | 7.74e-5 | **0.97** (rollout warmup ends) |
   | 100000 | 2.0 | 5.17e-5 | 0.65 (ramp midpoint) |
   | 150000 | 4.0 | 1.75e-5 | 0.22 (ramp ends) |
   | **160000** | **4.0** (clamped) | **1.22e-5** | **0.15** (← Plan F + Fix F endpoint) |
   | 200000 | 4.0 | 2.0e-6 | 0.025 (V6 final) |

3. **同 GPU 串行 vs 不同 GPU 并行【Fix 2 加强】**：seed=42 + deterministic=true 不保证不同 GPU 间 bit-equivalent（[OPERATOR_REPLY](../operator/OPERATOR_REPLY_run_ablation_sanity_failure_20260505.md) §2 强调 PyTorch 的 nondeterministic CUDA op 警告 + cuBLAS heuristic 跨 GPU 不同）。**严格要求**：V6/V7/V8 **必须**跑在同一块 GPU 上。Fix 2 补充预启动 checklist：
   - `nvidia-smi -L` 记录 V6 训练 GPU UUID，V7/V8 必须预留同一 UUID；
   - V6 训练时 driver / CUDA / cuDNN / torch 版本全部记入 `review/0505/operator/v6_env_record.txt`，V7/V8 启动前重复记录并逐项比对；
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在 launcher 已处理（run_v7_v8_sanity.sh L60），pre-flight 只需 echo 证据；
   - step-1000 sentinel（§5 Stage 1.5）：val_pair_total / val_chain_d20_mse / val_select_score 与 V6 偏差 ≤0.5%，否则 abort。

4. **Cohen's d 与阈值污染问题（已代—Q10-A）**：历史 V6 vs V6.1 全集比对 paired Cohen's d=0.0677 (best.pt) / 0.0673 (last.pt)（n=7403, full-val mse_NORMAL；t=−5.83, p≈5.6e-9；2026-05-06 由本地 per-slice CSV 重算，更新历史 0.056 估计）**是被三种源混合的**：(a) V6.1 的 rollout-floor 算法改动 vs V6；(b) 检查点选择偏差（各自 best.pt 在不同 step；现已确认 V6 自身 best.pt 也比 last.pt 略差，证明该选择存在偏差）；(c) 纯随机 noise。原 Plan F 阈值 0.10 ≈ 1.8 × 0.056 是启发式（实际 1.8 × 0.067 = 0.121，PRIMARY 在 max() 下仍取 0.10 floor，不变）；Q10-A 引入 V6-seed1337 跨种子复制（同算法/同 step/仅 seed 差异）计算 `d_pure`；PRIMARY 阈值调为 `max(0.10, 1.8 × d_pure)` —— 1.8× 保守裕度不变，但底型从被污染的 0.067 变为干净的 d_pure。Power 分析确认该阈值受噪声地板控制（不是受 power 控制）：在 n=7403 paired 下 80%-power MDE = 0.033 (α=0.05) / 0.039 (Bonferroni)。详 §6 Pre-registration 与 §5 Stage 0。

5. **V8 测不了 decode artifact**：image_aux 在论文里的 framing 是“压制 hop0 像素分支的 decode 伪影”。V8 只能测 chain MSE（latent 域），本 Plan F + Fix F 160K 不能证明或反驳“image_aux 压制伪影”该主张。V8 “200K main” 的兑现依赖于一个单独的 decode-domain 定性实验（V6_last vs V8_last 跨固定验证子集 decode 后人工对比），不在本 Plan F 范围。

6. **dead_branch_watch 覆盖面**：V8 已将 `dead_branch_watch_steps` 从 10000 提到 **160000**（Plan F + Fix F endpoint），覆盖整个训练窗口。gate_pix 或 lambda_hop_0 如果在该区间跳下 floor，警报会被 trainer 抓到（见 train_first_hop.py dead_branch_patience 逻辑）。

7. **EMA-swap checkpoint bug【延迟处理，不阻塞 Plan F + Fix F】**：V6 磁盘上所有 `step_NNN.pt` / `best.pt` / `best_d1.pt` / `last.pt` 都在 `with ema.average_parameters(): save_checkpoint(...)` 上下文内保存（[train_first_hop.py:2456-2477](../../../train_first_hop.py#L2456-L2477) + L2478, L2523, L2562, L2592, L2606, L2644 6 处调用点）。这意味着 `ckpt["model"]` = EMA 权重，但 `ckpt["optimizer"]` 里的 Adam moments 是从 raw 权重轨迹累积出来的。**inference / eval 不受影响**（加载 EMA 权重是我们要的），但 `--resume` 会进入 model=EMA / moments=raw 的内在不一致状态——这是 Plan D 被判定为不可行的根本原因。Plan F + Fix F 全 from-scratch，不踩这个坑。单独记为 trainer ticket，以后修复时应保存两份 `model_raw` + `model_ema`，`--resume` 走 raw 路径。

8. **DiT RNG-free 验证【Fix 4 依据】**：Agent2 cross-AI peer review 提出 V8 关闭 image_aux 可能产生 RNG 轨迹偏移。本仓 DiT 【PETFlowDiTDHHopAware】验证以下点：
   - [model_utils.py NormAttention](../../../../RAE/RAE/src/pet_flow/models/model_utils.py)：`attn_drop=0.0`、`proj_drop=0.0` 默认值；F.scaled_dot_product_attention 的 `dropout_p=self.attn_drop.p if self.training else 0.` 在 train 模式仍为 0。
   - [pet_flow_dit.py Block:120](../../../../RAE/RAE/src/pet_flow/models/pet_flow_dit.py)：Mlp 采用 `drop=0` 明示设置。
   - 本仓 RAE pet_flow models 全部 grep 未发现 `DropPath` / `drop_path` / `stochastic_depth`。
   - `pet_lr/` 未发现 `nn.Dropout`。
   结论：DiT forward 在 train 模式 bit-deterministic，**不消耗 RNG**。Agent2 BLOCKER 为假警报。Fix 4 补充一步 RNG 闭环诊断脚本作为 belt-and-suspenders，防中途 DiT 被修改后出现潜藏随机调用。

---

## 8. 文件清单

```
review/plan/
└── PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md  (決策文档：为什么放弃 D / 采纳 F)

review/0505/local/
├── ANALYSIS.md                       (本文件：实验设计 + 决策矩阵 + caveats)
├── PEER_REVIEW_PROMPT_V7_V8.md      (第一轮设计评审 prompt，4 reviewers)
├── PEER_REVIEW_PROMPT_PLAN_D.md     (第二轮 Plan D 评审 prompt，2 reviewers)
├── PEER_REVIEW_PROMPT_PLAN_F.md     (第三轮 Plan F 评审 prompt，Agent1+Agent2)
├── CODEX_QUESTION_AB_SANITY_ROOT_CAUSE.md  (上一轮文档，未推送)
├── configs/
│   ├── V7_gronwall_raw.yaml          (Grönwall raw step_weights, σ-norm OFF, max_steps=160000)
│   ├── V8_no_image_aux.yaml          (V6 minus image_aux, max_steps=160000)
│   └── V6_seed1337.yaml              (V6 noise-baseline replica, seed=1337, max_steps=160000；Q10-A 2026-05-05)
└── scripts/
    ├── run_v7_v8_sanity.sh           (启动器，V7|V8|V6_NOISE|both|all|compare；名中留 "sanity" 仅为历史原因)
    ├── diag_v8_rng_invariance.py     (Fix 4 防御性 RNG 闭环诊断脚本)
    └── compare_v6_v7_v8.py           (post-hoc markdown 表格生成器)
```

---

## 9. 提交建议

V7/V8 跑完后，把 `metrics.jsonl` 和 `train.log` 拷到 `review/0505/local/runs/V7/` 和 `review/0505/local/runs/V8/`，然后 commit：

```bash
git add review/0505/local/{ANALYSIS.md,configs/,scripts/,runs/V7,runs/V8,V7_V8_compare.md}
git commit -m "review(0505): V7 (Grönwall raw) + V8 (no image_aux) 160K Plan F + Fix F"
git push gitee foc_lite_hop0
```

**不要** push `/data_2/...` 下的 ckpts —— 太大且 path_guard 已经把它们隔离在 review/ 之外。

---

## 10. 维护责任

如果在本目录的 yaml / script 上做改动，请：
1. 在文件头注释里加 changelog（"v2 (2026-MM-DD): ..."）
2. 更新本文件的 §决策矩阵 / §caveats
3. 不要直接修改 `review/0502/configs/*.yaml` —— 那是 σ-normalize ablation 的封锁配置，改它会失效 sanity gate sentinel
