# Plan F Launch Runbook — 3-GPU 并行 (2026-05-06)

**Date**: 2026-05-06
**Branch**: `foc_lite_hop0`
**Pre-launch HEAD**: `fd05a58` (V7/V8/V6_NOISE 全部就绪)
**Locked baseline tag**: `round4-tmp-pre-external-review-20260506` (在 `dcdcd2c`)
**Audience**: 远程 GPU operator
**Hardware constraint (operator)**: 一次最多并发 **3 个实验**

---

## 0. 概要

本文档是 Plan F 三组 ablation (V7 / V8 / V6_NOISE) **从启动到最终判决的完整执行手册**。所有
设计、yaml、launcher、scripts 均已在 `fd05a58` 上落定，**operator 不需要修改任何 yaml 或源码**。

唯一在跑实验前必须完成的强阻塞 gate（来自 ROUND4_EXTERNAL_CONSENSUS 5/5 共识）：

1. **Gate (d)**: V6@step_160000.pt 上的 σ-normalize 同进程代数 golden test (`sigma_norm_golden_test.py`，~30 GPU-sec)
2. **Action #4**（推荐）: V6@seed42 same-seed 5K double-pass，量化 RNG-only 噪声地板
3. **Action #5**（推荐）: V7@seed42 5K spot-check double-pass，决定是否需要 Option H bundle

本 runbook 把这些 gate 全部塞进 **Phase 1 三卡并行 ~1 小时墙钟时间**。Phase 2 起开始 160K 主实验。

参考文档：

- 实验设计 (Plan F): [`review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md`](../../plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md)
- Round 4 共识: [`review/plan/ROUND4_EXTERNAL_CONSENSUS_20260506.md`](../../plan/ROUND4_EXTERNAL_CONSENSUS_20260506.md)
- 原始 Round 4 prompt（已锁定，md5=`abeb80289bc5bdc068d8e226b54b9544`）: [`review/0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md`](../../0505/local/PEER_REVIEW_PROMPT_SIGMA_VS_RAW_20260506.md)
- V6 全集 anchor (operator 已上传): [`review/0505/operator/v6_step160k_fullval_anchor_20260506.md`](../../0505/operator/v6_step160k_fullval_anchor_20260506.md)
- V6 自比经验先验 (refute Opus 4.7 d≥0.20): [`PLAN_F §12.6`](../../plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md#126-empirical-d_pure-prior-from-v6-step160k-full-val-anchor-added-2026-05-06)

---

## 1. GPU 分配

V6 历史训练用的是物理 **GPU2** (UUID=`GPU-4915051c-84ae-540c-8e84-5071cff15640`)。
为了让 V7（与 V6 最直接的对比 arm）继承同一张物理 GPU 的 cuBLAS heuristic，**V7 优先排在 GPU2**。

**前提**：3 张 GPU 必须是 **同一型号**（如全部 RTX A6000）；UUID 不必相同（不同 UUID 仅影响
per-card 微弱 FP nondeterminism，远低于 §12.6 测得的 V6 vs V6.1 paired d=0.067 噪声地板）。

| 物理 GPU | 历史用途 | Phase 1 (~1.5 hr) | Phase 2 (~30 hr) |
|---|---|---|---|
| **GPU2** | V6@seed42 主训练（lineage anchor） | Gate (d) Option F → V6 5K pass-1 → V6 5K pass-2 | **V7 (160K)** |
| **GPU3** | （新分配） | V8 RNG-invariance diag → V7 5K pass-1 → V7 5K pass-2 | **V8 (160K)** |
| **GPU1** | （新分配） | （空闲，可做磁盘/环境检查） | **V6_NOISE (160K)** |

如果 operator 的 3 张卡编号不是 1/2/3，请把下文所有 `GPU=N` 替换为实际编号；启动 V7
时使用与 V6 相同 UUID 的卡，其余两张分配给 V8 和 V6_NOISE。

---

## 2. Phase 0 — 环境记录（~5 min，单卡顺序）

**目的**：留下 audit trail，满足 Plan F §13 §9.3 的环境锁定要求。

```bash
cd /home/qujiaxiang/project/PET_LatentResidual  # 或 operator 当前 repo 路径

# 0.1 GPU UUID 锁定（输出贴到 v7v8_env_record_20260506.txt）
nvidia-smi -L

# 0.2 driver / cuda / cudnn 三件套
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python -c "import torch; print('torch =', torch.__version__); print('cuda  =', torch.version.cuda); print('cudnn =', torch.backends.cudnn.version())"

# 0.3 CUBLAS workspace（launcher 已自动 export，但记录一下）
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-<unset, will be set by launcher>}"

# 0.4 磁盘空间（main 阶段每 arm ckpt ≈ 32 GB × 16 saves = 512 GB；3 arm ≈ 1.5 TB）
df -h /data_2

# 0.5 V6@step_160000.pt 必须存在（Gate (d) 输入）
ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt
```

把以上输出保存到：

```bash
mkdir -p review/0506/operator
{
  echo "=== nvidia-smi -L ==="; nvidia-smi -L
  echo ""; echo "=== driver / torch / cuda / cudnn ==="
  nvidia-smi --query-gpu=driver_version --format=csv,noheader
  python -c "import torch; print('torch =', torch.__version__); print('cuda  =', torch.version.cuda); print('cudnn =', torch.backends.cudnn.version())"
  echo ""; echo "=== CUBLAS_WORKSPACE_CONFIG ==="
  echo "${CUBLAS_WORKSPACE_CONFIG:-<unset, will be set by launcher to :4096:8>}"
  echo ""; echo "=== /data_2 free ==="
  df -h /data_2
  echo ""; echo "=== V6@step_160000.pt ==="
  ls -lh /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt
} | tee review/0506/operator/v7v8_env_record_20260506.txt
```

**Pass criterion**: `/data_2` 自由空间 ≥ 2 TB；V6 ckpt 存在；3 张 GPU 同型号。

---

## 3. Phase 1 — Pre-flight（3 卡并行，~1 hr 墙钟）

### 3.1 时间线表

| 时刻 (rel) | GPU2 | GPU3 | GPU1 |
|---|---|---|---|
| t=0 min | **Gate (d) Option F** (~30 sec) | **V8 RNG diag** (~10 sec) | （监控/磁盘检查） |
| t=1 min | **V6 5K pass-1** 启动 | **V7 5K pass-1** 启动 | （空闲） |
| t=~30 min | V6 5K pass-1 完成 | V7 5K pass-1 完成 | |
| t=~31 min | **V6 5K pass-2** 启动 | **V7 5K pass-2** 启动 | |
| t=~60 min | V6 5K pass-2 完成 | V7 5K pass-2 完成 | |
| t=~61 min | **从 train.log 抓 [val] 行计算 drift（见 §3.2-E），写决策记录** | | |

### 3.2 Phase 1 命令清单（按执行顺序）

#### A. Gate (d) Option F golden test（GPU2，强阻塞 V7 启动）

```bash
# 在 GPU2 上跑 σ-normalize 同进程代数对比，~30 秒
CUDA_VISIBLE_DEVICES=2 python3 review/0505/local/scripts/sigma_norm_golden_test.py \
  --config configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml \
  --ckpt /data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_v6_transport_first/step_160000.pt \
  --output-json review/0506/operator/sigma_norm_golden_test_result.json
```

**退出码语义** (Plan F §11.0)：

| exit | 含义 | 动作 |
|---|---|---|
| 0 | PASS | V7 路径不引入 σ-normalize 类残留差异，**解锁 V7 启动** |
| 2 | setup error | 检查 ckpt / yaml 路径，重跑 |
| 3 | AMBIGUOUS | **见 §3.3 Step 3 结构隔离论证**：V7/V8/V6_NOISE 三个 yaml 都没有 `sigma_normalize` 块（已 grep 验证），训练时 `step_normalizers=None`，[`rollout_first_hop.py:102`](../../../pet_lr/rollout_first_hop.py) 的除法分支永不执行 → σ-normalize 算法路径与训练 disjoint。三个 arm 可与 Option G 并行启动；Option G 结果仅作 σ-normalize 历史 forensic 归因 |
| 4 | FAIL | **暂不启动全部三个 arm**；与本人确认 §11 Tier 处理路径 |

#### B. V8 RNG-invariance diagnostic（GPU3，强阻塞 V8 启动）

```bash
# 在 GPU3 上跑 V8 image_aux 开/关 RNG 字节级一致性检查，~10 秒
CUDA_VISIBLE_DEVICES=3 python3 review/0505/local/scripts/diag_v8_rng_invariance.py \
  --config review/0505/local/configs/V8_no_image_aux.yaml \
  --num-batches 1 \
  --output-json review/0506/operator/v8_rng_invariance_result.json
```

**Pass criterion**: 脚本退出码 0 且输出 `rng_byte_identical=true`。如果 false，**暂不启动 V8**；
与本人确认。（DiT 已通过本地 grep 确认无随机 op，本检查纯属防御性。）

#### C. V6@seed42 5K double-pass（Action #4，GPU2 顺序两次）

```bash
# pass-1，~30 min
GPU=2 bash review/0506/local/scripts/preflight_double_pass.sh V6 1

# pass-2，~30 min（必须等 pass-1 完成）
GPU=2 bash review/0506/local/scripts/preflight_double_pass.sh V6 2
```

#### D. V7@seed42 5K double-pass（Action #5，GPU3 顺序两次）

```bash
# pass-1，~30 min（与 GPU2 V6 pass-1 并发）
GPU=3 bash review/0506/local/scripts/preflight_double_pass.sh V7 1

# pass-2，~30 min（必须等 pass-1 完成）
GPU=3 bash review/0506/local/scripts/preflight_double_pass.sh V7 2
```

#### E. Phase 1 drift 测量（不需要跑 full-val 脚本）

由于 V6/V7 yaml 都设了 `eval_interval=400`，5K 训练过程中会在 step 4800（最近的
400 倍数 ≤ 5000）触发 rolling-val。**直接从 train.log 抓 [val] 行即可**，无需另跑
full-val 脚本（rolling-val 的 absolute number 与 full-val 有 +3.18% 偏差，但
**pass-1 vs pass-2 drift 是同一确定性 512-slice 窗口的相对量，rolling-val 完全够用**）。

```bash
# 从 4 个 5K 的 train.log 抓 step ≥ 4800 的最后一行 [val]
for f in review/0506/local/logs/{V6,V7}_5K_pass{1,2}_gpu*.log; do
  echo "=== ${f##*/} ==="
  grep "^\[val\] step=" "${f}" | tail -1 | grep -oE "step=[0-9]+|val_pair_total=[0-9.e+-]+|val_chain_normal_mse=[0-9.e+-]+"
done
```

把抓出的 4 组 (val_pair_total, val_chain_normal_mse) 写到
`review/0506/operator/preflight_decision_20260506.md`（见 §3.3 Decision Tree）。

### 3.3 Phase 1 Decision Tree（决定能否启动 Phase 2）

跑完 4 次 5K full-val 后，从输出 JSON 读出 `val_pair_total` 和 `val_chain_normal_mse`，按如下判决：

```
Step 1: V6 pass-1 vs V6 pass-2
  drift_v6 = abs(pass2.val_pair_total - pass1.val_pair_total) / pass1.val_pair_total

  if drift_v6 ≤ 1.0%   → V6 RNG floor 正常，进入 Step 2
  if drift_v6 > 1.0%   → 强制升级 §11 Tier 2 (Welch's t + 多 seed)，
                         即使 V6_NOISE 最终 d_pure ≤ 0.10 也按 Tier 2 处理。
                         继续 Phase 2，但提交回 d_pure 时附带本警告。

Step 2: V7 pass-1 vs V7 pass-2
  drift_v7 = abs(pass2.val_pair_total - pass1.val_pair_total) / pass1.val_pair_total

  # Margin derivation: drift_v6 / drift_v7 are each single-shot |Δ| measurements
  # from one V6/V7 double-pass. Under H0 (V7 path determinism = V6 path), they are
  # i.i.d. samples; bare "drift_v7 > drift_v6" has ~50% H0 false-positive rate.
  # Using V6's single drift as the local scale estimate, the conservative margin is:
  #   epsilon = max(drift_v6, 1e-7)
  # which gives gate threshold = drift_v6 + epsilon = max(2*drift_v6, drift_v6 + 1e-7).
  # The 1e-7 absolute floor matches sigma_norm_golden_test rel_tolerance and prevents
  # "two near-zero noise floors compete" pseudo-triggers when both drifts are tiny.
  # The factor-of-2 multiplicative term is the single-trial MDE rule-of-thumb; under
  # half-normal approximation this corresponds to ~1-sigma margin, ~16% H0 FPR.
  epsilon = max(drift_v6, 1e-7)
  gate    = drift_v6 + epsilon       # = max(2*drift_v6, drift_v6 + 1e-7)

  if drift_v7 ≤ gate → V7 与 V6 在确定性特性上一致（一个 σ 之内或低于绝对地板），
                       **正常启动 Phase 2**
  if drift_v7 > gate → V7 引入超出 V6 RNG floor 一个 σ 的新 (α) 源；启动 Phase 2
                       前必须先应用 Option H bundle（math-SDPA + AdaptiveAvgPool2d
                       替换 + matmul_precision('highest')）。
                       联系本人确认 Option H 应用方式。

Step 3: Gate (d) 验证
  if Option F exit 0 (PASS)         → 完全解锁 Phase 2
  if Option F exit 2 (setup error)  → 修复 ckpt/yaml 路径后重跑；暂缓 Phase 2
  if Option F exit 3 (AMBIGUOUS)    → **结构隔离论证**：V7/V8/V6_NOISE 三个 yaml 都没有
                                      `sigma_normalize` 块（启动前 grep 验证）→ 训练时
                                      step_normalizers=None → rollout_first_hop.py:102
                                      的除法分支永不执行 → σ-normalize 算法路径与训练
                                      disjoint。**三个 arm 可与 Option G 并行启动**
                                      （Option G ~30 min 完成，仅作 σ-normalize 历史
                                      forensic 归因，不阻塞训练）。Phase 2 启动后
                                      operator 须并行触发 Option G 并把结果写入
                                      `review/0506/operator/option_g_result_20260506.md`
  if Option F exit 4 (FAIL)         → algebra 真的破了；rollout 代码路径异常，**暂缓
                                      全部三个 arm**（V7/V8/V6_NOISE 都使用同一
                                      rollout codepath）；联系本人确认 §11 Tier 处理
```

**Phase 1 结束后必须提交的 artifact**：

- `review/0506/operator/v7v8_env_record_20260506.txt`
- `review/0506/operator/sigma_norm_golden_test_result.json`
- `review/0506/operator/v8_rng_invariance_result.json`
- `review/0506/local/logs/V6_5K_pass{1,2}_gpu*.log`（两个 pass 的 train.log，体积 < 1 MB 各）
- `review/0506/local/logs/V7_5K_pass{1,2}_gpu*.log`
- `review/0506/local/preflight/{V6,V7}_BASE_5K_pass{1,2}.resolved.yaml`（4 个 resolved yaml，每个 ~8 KB）
- `review/0506/operator/preflight_decision_20260506.md`（一段自然语言决策记录：
  drift_v6=…, drift_v7=…, Gate(d) result=…, V8 RNG diag=…, 结论=继续/暂停/Option H）

---

## 4. Phase 2 — 主实验启动（3 卡并行，~30 hr 墙钟）

### 4.1 启动命令（必须按以下顺序在 3 个 tmux/screen session 里启动）

打开 3 个 session，分别做：

#### Session 1: V7 + V8 并行启动（GPU2 + GPU3）

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# 默认 SANITY_STEPS=160000；GPU_V7=2、GPU_V8=3 即按 §1 分配
SANITY_STEPS=160000 GPU_V7=2 GPU_V8=3 \
  bash review/0505/local/scripts/run_v7_v8_sanity.sh parallel
```

启动后 launcher 会先做两个 arm 的 prep（freshness 检查），然后 fork 两个 python 进程，
一个在 GPU2 跑 V7，一个在 GPU3 跑 V8。两个 train.log 分别写到：

- `review/0505/local/runs/V7/train.log`
- `review/0505/local/runs/V8/train.log`

#### Session 2: V6_NOISE 独立启动（GPU1）

```bash
cd /home/qujiaxiang/project/PET_LatentResidual

# V6_NOISE = V6_seed1337 (Q10-A noise-baseline replica)
SANITY_STEPS=160000 GPU=1 \
  bash review/0505/local/scripts/run_v7_v8_sanity.sh V6_NOISE
```

train.log 在 `review/0505/local/runs/V6_NOISE/train.log`。

#### Session 3: 监控（保留一个 session 用来 tail）

```bash
# 用 tmux/iTerm 分屏 tail 三个 log
tail -f review/0505/local/runs/V7/train.log         # 注意 step 进度 + GPU 利用率
tail -f review/0505/local/runs/V8/train.log
tail -f review/0505/local/runs/V6_NOISE/train.log
```

### 4.2 step-1000 sentinel（**强阻塞**）

每个 arm 走到 step 1000 时，从其 metrics.jsonl 抓 `val_pair_total`、`val_chain_d20_mse`、
`val_select_score`，与 V6 step-1000 对比。容差 0.5%。如果超过则**立即停止该 arm**（kill
python 进程），与本人确认。

V6 step-1000 参考值在 `review/0505/operator/logs_train/v6_transport_first_train_200k.log`，
找 `[val] step=1000` 那行。

```bash
# 抽取 V6 step-1000 参考
grep "^\[val\] step=1000 " review/0505/operator/logs_train/v6_transport_first_train_200k.log

# 抽取 V7/V8/V6_NOISE step-1000
for arm in V7 V8 V6_NOISE; do
  echo "=== ${arm} ==="
  grep "^\[val\] step=1000 " review/0505/local/runs/${arm}/train.log | head -1
done
```

理论上 step 1000 时 lambda_roll=0 + LR 仍在 warmup，V7 的新 step_weights 还没进 rollout
loss、V8 的 image_aux 关闭对 backbone 梯度无影响，所以三个 arm 在 `val_pair_total` /
`val_chain_d20_mse` 上应该与 V6 几乎 bit-equivalent。如果偏离 > 0.5%，就是有 RNG 路径
新增了 (α) 源，需要 Option H bundle。

### 4.3 中途监控（每 10K step 一次）

```bash
# 每 ~3 hr 检查一次，确认没炸
for arm in V7 V8 V6_NOISE; do
  echo "=== ${arm} latest train row ==="
  grep "^\[train\] step=" review/0505/local/runs/${arm}/train.log | tail -1 | cut -c1-200
  echo "=== ${arm} latest val row ==="
  grep "^\[val\] step="   review/0505/local/runs/${arm}/train.log | tail -1 | cut -c1-200
done

# GPU 利用率（应当三张都接近 100%）
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
```

### 4.4 训练完成判定

每个 arm 训完后 train.log 末尾会有 `Training done. Outputs at: ...`。step_160000.pt
应当存在于 `${output_dir}/step_160000.pt`：

```bash
for arm in V7 V8 V6_NOISE; do
  out_dir="/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/${arm}/run"
  echo "=== ${arm} ==="
  ls -lh "${out_dir}/step_160000.pt" 2>/dev/null || echo "  step_160000.pt MISSING"
  ls -lh "${out_dir}/best.pt"        2>/dev/null || echo "  best.pt MISSING (V6_NOISE 没有 best.pt 是预期的)"
done
```

---

## 5. Phase 3 — 全集评分（每 arm ~6 min × 2 modes）

每个 arm 的 step_160000.pt 都要跑两次 full-val，分别用 `latent` 和 `both` 模式。`both` 模式
还会输出 decode chain MSE 用于副刊比较。

```bash
EVAL_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/eval_0506_planf_endpoint
mkdir -p "${EVAL_OUT}"

# V7（GPU2）
CUDA_VISIBLE_DEVICES=2 python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config review/0505/local/configs/V7_gronwall_raw.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/step_160000.pt \
  --tag v7_step160k --split val --batch-size 8 --device cuda:0 \
  --decode-mode both --out-dir "${EVAL_OUT}"

# V8（GPU3）
CUDA_VISIBLE_DEVICES=3 python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config review/0505/local/configs/V8_no_image_aux.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V8/run/step_160000.pt \
  --tag v8_step160k --split val --batch-size 8 --device cuda:0 \
  --decode-mode both --out-dir "${EVAL_OUT}"

# V6_NOISE（GPU1）
CUDA_VISIBLE_DEVICES=1 python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config review/0505/local/configs/V6_seed1337.yaml \
  --checkpoint /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V6_NOISE/run/step_160000.pt \
  --tag v6_seed1337_step160k --split val --batch-size 8 --device cuda:0 \
  --decode-mode both --out-dir "${EVAL_OUT}"
```

每次产出：

- `${EVAL_OUT}/<tag>_psnr_chain_mse.json` (~5 KB headline)
- `${EVAL_OUT}/<tag>_psnr_chain_mse_per_slice.csv` (~3 MB, 7403 行)

提交时复制到 `review/0506/operator/artifacts/`：

```bash
mkdir -p review/0506/operator/artifacts
cp ${EVAL_OUT}/*.json ${EVAL_OUT}/*.csv review/0506/operator/artifacts/
```

---

## 6. Phase 4 — d_pure 计算 + 最终判决

### 6.1 计算 d_pure

V6_NOISE = V6@seed1337@step_160000；anchor = V6@seed42@step_160000（已存在于
`review/0505/operator/artifacts/v6_step160k_fullval_psnr_chain_mse_per_slice.csv`）。

可直接用 §12.6 的脚本（`analyze_v6_self_paired_stats.py`）扩展，或写一个 10 行 scipy
wrapper：

```python
# review/0506/local/scripts/compute_d_pure.py
import csv, math, json, sys, pathlib

def load(p):
    return [float(r['mse_NORMAL']) for r in csv.DictReader(open(p))]

a = load('review/0505/operator/artifacts/v6_step160k_fullval_psnr_chain_mse_per_slice.csv')
b = load('review/0506/operator/artifacts/v6_seed1337_step160k_psnr_chain_mse_per_slice.csv')
assert len(a) == len(b) == 7403, f"slice count mismatch: {len(a)} vs {len(b)}"

diffs = [bi - ai for ai, bi in zip(a, b)]
n = len(diffs)
mean = sum(diffs) / n
var  = sum((d - mean) ** 2 for d in diffs) / (n - 1)
sd   = math.sqrt(var)
d_pure = abs(mean) / sd if sd > 0 else 0.0  # absolute Cohen d (sign agnostic)
t_stat = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
d_thr = max(0.10, 1.8 * d_pure)

result = dict(n=n, mean_diff=mean, sd_paired=sd, d_pure=d_pure, t_stat=t_stat, d_thr=d_thr)
out = pathlib.Path('review/0506/operator/d_pure_result.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
```

```bash
python3 review/0506/local/scripts/compute_d_pure.py
```

### 6.2 §11 tier 决策（机械执行，不需联系本人）

读 `d_pure_result.json` 的 `d_pure`：

| d_pure 区间 | tier | 决策规则 | 操作 |
|---|---|---|---|
| ≤ 0.10 | Tier 0 | 用 `d_thr = 0.10`（不放大） | 直接看 V7 vs V6@160K paired d 是否 ≥ 0.10 + rel MSE 改善 ≥ 5% → V7 wins |
| 0.10–0.20 | Tier 1 | 用 `d_thr = 1.8 × d_pure` | 同上但门槛抬高 |
| 0.20–0.40 | Tier 2 | 切 Welch's t-test + 至少 1 个额外 V7 seed | **联系本人**，新增 V7-seed1337 重训（+1 GPU·day） |
| > 0.40 | Tier 3 | 反过来重述贡献：V7 = closed-form 推导本身 + reproducibility 分析，不主张优于 V6 | **联系本人**讨论叙事路线 |

§12.6 经验先验告诉我们 ~85% 概率落在 Tier 0；不要被 Opus 4.7 之前的 d≥0.20 框定吓到。

### 6.3 最终判决表（V7_V8_compare.md）

V7 vs V6@160K paired Cohen d 和相对 MSE 改善要做两组：

| 比较 | d_thr | 实测 paired d | rel MSE 改善 | tier 内 verdict |
|---|---|---|---|---|
| V7 vs V6@160K (NORMAL chain MSE) | 计算结果 | 计算 | 计算 | win/tie/loss |
| V8 vs V6@160K (NORMAL chain MSE) | 同上 | | | |
| V7 vs V6@160K (rollout total)    | | | | secondary |
| V8 vs V6@160K (rollout total)    | | | | secondary |

把这张表 + d_pure 输出 + tier 决策结果 一并写到：
`review/0506/operator/V7_V8_compare_20260506.md`，commit + push。

---

## 7. 提交清单（Phase 1 + Phase 4 各提交一次）

### Phase 1 commit（pre-flight 完成时）

```
git add review/0506/operator/v7v8_env_record_20260506.txt \
        review/0506/operator/sigma_norm_golden_test_result.json \
        review/0506/operator/v8_rng_invariance_result.json \
        review/0506/local/logs/ \
        review/0506/local/preflight/ \
        review/0506/operator/preflight_decision_20260506.md
git commit -m "review/0506: Phase 1 pre-flight artifacts (Gate (d) + Action #4 + Action #5)"
git push origin foc_lite_hop0
git push gitee  foc_lite_hop0
```

### Phase 4 commit（最终判决时）

```
git add review/0506/operator/artifacts/ \
        review/0506/operator/d_pure_result.json \
        review/0506/operator/V7_V8_compare_20260506.md
git commit -m "review/0506: Plan F final verdict — V7 + V8 + V6_NOISE @ step_160000 full-val"
git push origin foc_lite_hop0
git push gitee  foc_lite_hop0
```

---

## 8. 故障应对

| 问题 | 处理 |
|---|---|
| Phase 1 Gate (d) FAIL/AMBIGUOUS | 暂不启动 V7；按 §3.2-A 退出码表格处理；联系本人 |
| Phase 1 V8 RNG diag false | 暂不启动 V8；联系本人；DiT 可能被改了 |
| Phase 1 drift_v6 > 1.0% | 强制升级 Tier 2，但仍可继续 Phase 2，需在 commit 注释里说明 |
| Phase 1 drift_v7 > drift_v6 | 暂不启动 V7；联系本人讨论 Option H bundle 应用方式 |
| Phase 2 step-1000 sentinel 偏差 > 0.5% | 立即 kill 该 arm，联系本人 |
| Phase 2 训练崩溃（OOM / NaN / 进程 die） | 截 train.log 最后 100 行 + nvidia-smi 输出，联系本人；**不要**直接 resume |
| Phase 2 GPU 突然 hang | nvidia-smi 检查；如必要 kill 该 python，从 step_140000 等近邻 ckpt resume（trainer 支持 `resume_path`） |
| Phase 3 eval 报 OOM | `--batch-size 4`（默认 8 已经很轻；理论上不会 OOM） |
| Phase 4 d_pure NaN | 检查两个 CSV 行数都是 7403 且 slice_idx 一致（应自动满足） |

---

## 9. 文件位置速查

| 文件 | 路径 |
|---|---|
| 主 launcher（V7/V8/V6_NOISE） | `review/0505/local/scripts/run_v7_v8_sanity.sh` |
| Pre-flight launcher (V6/V7 5K) | `review/0506/local/scripts/preflight_double_pass.sh`（本 commit 新增） |
| Option F golden test | `review/0505/local/scripts/sigma_norm_golden_test.py` |
| V8 RNG-invariance diag | `review/0505/local/scripts/diag_v8_rng_invariance.py` |
| V6@seed42 yaml (Phase 1 Action #4 源) | `configs/pet_flow/pet_flow_first_hop_224_v6_transport_first.yaml` |
| V7 yaml | `review/0505/local/configs/V7_gronwall_raw.yaml` |
| V8 yaml | `review/0505/local/configs/V8_no_image_aux.yaml` |
| V6_NOISE yaml (V6_seed1337) | `review/0505/local/configs/V6_seed1337.yaml` |
| Full-val eval 脚本 | `review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py` |
| V6@step_160000 anchor (已 in-repo) | `review/0505/operator/artifacts/v6_step160k_fullval_psnr_chain_mse{,.json,_per_slice.csv}` |
| Plan F 主文档 | `review/plan/PLAN_F_V7_V8_FROM_SCRATCH_150K_20260505.md` |
| Round 4 共识 | `review/plan/ROUND4_EXTERNAL_CONSENSUS_20260506.md` |

---

## 10. 时间预算汇总

| 阶段 | 三卡墙钟 | 说明 |
|---|---|---|
| Phase 0 环境记录 | 5 min | 单卡顺序 |
| Phase 1 pre-flight | ~1 hr | GPU2 = Option F + V6 5K × 2；GPU3 = V8 diag + V7 5K × 2；GPU1 空闲。drift 从 train.log [val] 行直接读，**不跑 full-val** |
| Phase 2 主训练 | ~30 hr | V7（GPU2）+ V8（GPU3）+ V6_NOISE（GPU1）三卡并行 160K |
| Phase 3 全集评分 | ~20 min | 三卡并行，每 arm 一次 full-val（both 模式） |
| Phase 4 d_pure + 判决 | ~10 min | 单卡或 CPU |
| **总计** | **~31.5 hr** | 假设 step 时间 ~0.7 sec @ batch=8 RTX A6000 |

实际 step 时间在 step-1000 sentinel 那一步就能验证。如果实测 step 时间 > 1.0 sec/step，
ETA 顺延约 30%（Phase 2 变 ~40 hr）。

---

**END OF RUNBOOK**

如有 Phase 1 之外的任何 deviation，先停下来联系本人，**不要**自行修改 Plan F yaml 或源码。
