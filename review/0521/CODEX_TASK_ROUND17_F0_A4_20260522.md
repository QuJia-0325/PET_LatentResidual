# Codex Task — Round 17 Stage F0 + A4 (Post-V13/V14 Strategic Execution)

- date: 2026-05-22
- branch: foc_lite_hop0
- status: **READY FOR CODEX EXECUTION**
- 上游: Round 17 集成 ([REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md)) 4/4 共识 = Hybrid B + F0 + A4-light
- 触发: V13/V14 已完成, image_aux 已确认为 +0.287 dB 主线收益, V18 信号显著性需 paired-t 验证, image_aux schedule 调优是唯一仍有 EV 的新 GPU 实验
- 硬约束: ≤ 3 并行训练任务; F0 不跑训练, 但若 V13/V14 per-slice CSV 缺失需先用 1 个空闲 GPU 生成 canonical eval CSV; A4 是 1 个新训练任务 (slot 1); paper draft 同时进行 (与 codex 无关)

---

## §0 — 给 codex 的 5 句话总览

1. **本 task 含 2 个独立子任务**: F0 (slice-level 统计分析; 必要时先用 1 个空闲 GPU 生成 V13/V14 per-slice eval, 不训练, 必做) + A4 (image_aux schedule probe, 1 训练任务, **GPU 空闲时做**).
2. F0 使用/生成 canonical chain per-slice CSV, 输出 V18.best/last vs V7 / V13 vs V7 / V14 vs V7 的 mean Δ、win-rate、bootstrap CI 和 paired-t. V18-cap 只做 F0b direct-decode substrate, 不和 chain rollout 混算. **不跑训练**.
3. A4 = 在 V7 yaml 基础上**仅改 1 个变量** (image_aux `lambda_max: 0.04 → 0.08`), 其它一切保持 V7 完全一致, from-scratch 训到 step 160000, ~7 天.
4. **绝对不**改 train_first_hop.py / V7 yaml / V18 已有 ckpt / V13 V14 已有 ckpt. **绝对不**复活 V18-clean / V19 / decoder rank sweep.
5. 完成后用 canonical full-val eval 脚本评 A4 best.pt + last.pt, 与 V7 / V13 / V14 一起放进同一张 PSNR 表.

---

## §1 — 物理含义 (user 必读)

### 1.1 Round 17 共识背景

| 已知事实 (canonical full-val NORMAL PSNR_clip3, n=7403) | 值 (dB) |
|---|---:|
| V13.best (image_aux off, V7 config) | 36.4943 |
| V14.best (V7 + seed=1337) | 36.7806 |
| V7.best (seed=42) | 36.7810 |
| V18.best @165K (LoRA r32 + KL on) | 36.8112 |
| V18.last @200K (LoRA r32 + KL on) | 36.8426 |

派生事实:
- **X1 (d_pure 噪声地板)**: V14 − V7 = −0.0004 dB (单次 seed perturbation, 不是 std)
- **X2 (image_aux 单变量贡献)**: V14 − V13 = +0.2863 dB ≈ V7 − V13 = +0.2867 dB
- **X3 (Grönwall step_weights)**: V13 − V8 = +0.0214 dB (V8=36.4729, 但 train config 仍 confound)

### 1.2 F0 为什么必须做

Round 17 prompt §2 给的 "V18 +0.06 dB SNR=75-150× noise" 是**方法论错误** — 它把单次 V14 |Δ|=0.0004 当 noise σ. 真正的显著性需要 per-slice paired-t:

> 对 7403 slice 逐个算 `psnr_clip3_V18(slice_i) - psnr_clip3_V7(slice_i)`, 然后算 mean / std / paired SEM / t = mean / SEM / two-sided p.

这种方法已有先例: `review/0516/PLANF_FINAL_ANALYSIS_20260516.md` 在 V7 vs V8 上算过 paired t = 74.6 over 7403 slices. F0 = 复用同型方法套 V18 vs V7.

F0 分两步: 若 V13/V14 per-slice CSV 缺失, 先用 canonical eval 脚本在 1 个空闲 GPU 上生成 CSV; 之后统计分析在 CPU 上完成. 全程不训练. F0 是 V18 进入 paper 作 ablation 的**前置 gate**.

### 1.3 A4 为什么是唯一仍有 EV 的新训练实验

V13/V14 已经把 image_aux 锁定为项目最大单变量收益 (+0.287 dB). 当前 V7 yaml 用 `lambda_max=0.04` flat schedule. 这个值从未做过 schedule 扫描. 自然假设:

- **H1**: image_aux 在 0.04 已饱和, 更高 lambda 不带来额外收益 → 出 0 dB 增量
- **H2**: image_aux 在 0.04 未饱和, 0.08 (double) 带来 +0.05 ~ +0.15 dB → paper 多一条 headline-additive 证据
- **H3**: image_aux 0.08 过强, transport loss 被淹, 出现 -0.05 dB 倒退 → 说明 0.04 已经是 sweet spot, paper 多一条 negative ablation

H2 / H3 都 paper-useful. H1 也 paper-useful (饱和证据). 所以 A4 是 win-win-win.

**A4 stop rule (Round 17 共识)**: 如果 A4 best.pt NORMAL < V7 + 0.05 dB, **不再 relaunch v2 / v3**. 出 1 个数据点入 paper, 项目进入写作期.

### 1.4 不做什么 (Round 17 拒绝清单)

- ❌ V19 / V18-clean (use_pred_latent=false) — Round 16 已签 KL direct-decoder 死
- ❌ V18-rank-sweep / V18-blocks-sweep — decoder 已不是 chain bottleneck, EV 低
- ❌ V14b / V14c (多 seed) — F0 更便宜地回答显著性
- ❌ Architecture pivot (DiT → flow matching) — 出 Round 17 范围
- ❌ Data pivot — 出 Round 17 范围
- ❌ A4-v2 (在 A4 出来前提前设计第二个 schedule) — 防止 scope creep (B74)

---

## §2 — F0: Slice-Level Paired Statistics (必做, 不训练)

### F0.0 输入资源 (Round 17-Prep 修订: B75-B79 fix)

**重要事实** (Round 17-Prep review B77 verify):
- V18 / V7 per-slice CSV **已存在** (canonical eval 脚本默认就输出 `*_per_slice.csv` 在 `artifacts/` 子目录)
- V13 / V14 per-slice CSV **不存在**, JSON 是 summary-only — 需 §F0.0a 重跑
- V18-cap CSV 是 direct `decode(z_GT)` substrate, **不能**和 chain rollout PSNR 做 paired-t

| ckpt | 文件 | 状态 |
|---|---|---|
| V7.best | `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv` | ✓ 已存在 (7404 rows incl header, 21 cols) |
| V18.best | `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv` | ✓ 已存在 |
| V18.last | `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv` | ✓ 已存在 |
| V13.best | (需重跑 §F0.0a 生成) | ✗ summary-only JSON |
| V14.best | (需重跑 §F0.0a 生成) | ✗ summary-only JSON |
| V18-cap.last | `review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_PER_SLICE.csv` | ✓ 已存在 (direct decode substrate, **F0b only**) |

CSV 列名 (canonical schema, F0 用 `psnr_NORMAL` 即 clip3, 不是 `psnr_raw_NORMAL`):
```
slice_idx, mse_D10, mse_D20, mse_D4, mse_D50, mse_NORMAL, mse_raw_D10, ..., psnr_D10, psnr_D20, psnr_D4, psnr_D50, psnr_NORMAL, psnr_raw_D10, ..., psnr_raw_NORMAL
```

### F0.0a — 重跑 V13 / V14 canonical eval 生成 per-slice CSV (必做)

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only origin foc_lite_hop0
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}

# verify CLI flag (B30 standing rule)
SUPPORTED=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ' | sed 's/ $//')
EXPECTED="--config --resume"
if [ "$SUPPORTED" != "$EXPECTED" ]; then echo "FAIL: trainer CLI changed: $SUPPORTED"; exit 1; fi

# verify eval script flag is --out-dir (NOT --output-dir; Round 17-Prep B76 fix)
grep -q -- '--out-dir' review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py || { echo "FAIL: --out-dir not in eval script"; exit 1; }

V13_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off
V14_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V14_true_d_pure/run/first_hop_224_v14_v7_seed1337

# eval script enforces --out-dir under /data_2 via pet_lr.path_guard.
# Write raw eval outputs to /data_2, then copy JSON/CSV/log snapshots into repo.
F0_EVAL_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_f0_v13_v14_per_slice
REPO_F0_DIR=review/0521/v13_v14_per_slice
mkdir -p "$F0_EVAL_OUT" "$REPO_F0_DIR/artifacts" "$REPO_F0_DIR/logs"

# pick a free GPU (Round 17-Prep B80 fix)
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | nl -v0 | sort -k2 -rn | head -1 | awk '{print $1}')
if [ -z "$FREE_GPU" ]; then echo "FAIL: no free GPU"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$FREE_GPU

for name in v13 v14; do
  if [ "$name" = "v13" ]; then OUT=$V13_OUT; else OUT=$V14_OUT; fi
  "$PYTHON" review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
    --config $OUT/config.yaml \
    --checkpoint $OUT/best.pt \
    --tag ${name}_best \
    --split val \
    --max-slices 0 \
    --batch-size 8 \
    --decode-mode both \
    --out-dir "$F0_EVAL_OUT" \
    2>&1 | tee "$REPO_F0_DIR/logs/${name}_eval_$(date +%Y%m%d_%H%M%S).log"
  # canonical sanity: V13 NORMAL ≈ 36.4943, V14 NORMAL ≈ 36.7806 (must match Round 17 §1.2)
done

# Copy JSON/CSV artifacts back into repo for reproducibility and review.
cp "$F0_EVAL_OUT"/v13_best_fullval_psnr_chain_mse.json "$REPO_F0_DIR/artifacts/"
cp "$F0_EVAL_OUT"/v13_best_fullval_psnr_chain_mse_per_slice.csv "$REPO_F0_DIR/artifacts/"
cp "$F0_EVAL_OUT"/v14_best_fullval_psnr_chain_mse.json "$REPO_F0_DIR/artifacts/"
cp "$F0_EVAL_OUT"/v14_best_fullval_psnr_chain_mse_per_slice.csv "$REPO_F0_DIR/artifacts/"

# Verify per-slice CSVs copied under repo artifacts/
ls -la review/0521/v13_v14_per_slice/artifacts/v13_best_fullval_psnr_chain_mse_per_slice.csv
ls -la review/0521/v13_v14_per_slice/artifacts/v14_best_fullval_psnr_chain_mse_per_slice.csv

# Sanity-check NORMAL means against canonical
"$PYTHON" - <<'PY'
import csv, statistics
for name, expect in [('v13', 36.4943), ('v14', 36.7806)]:
    path = f'review/0521/v13_v14_per_slice/artifacts/{name}_best_fullval_psnr_chain_mse_per_slice.csv'
    rows = list(csv.DictReader(open(path)))
    mean = statistics.fmean(float(r['psnr_NORMAL']) for r in rows)
    print(f'{name} per-slice n={len(rows)} mean psnr_NORMAL={mean:.4f} (expect {expect})')
    assert abs(mean - expect) < 0.01, f'{name} mean drift: got {mean}, expect {expect}'
print('V13/V14 per-slice sanity OK')
PY
```

**cost**: ~12-15 min total on 1 GPU. **必须**在 F0.2 之前完成.

### F0.2 paired-t 实施

写一个**新脚本** `tools/paired_t_v18_vs_v7.py` (~120 行), 完成以下:

```python
#!/usr/bin/env python3
"""Round 17 F0 — slice-level paired statistics on canonical PSNR_clip3.

Compares each ckpt vs V7.best on per-slice basis, computes:
  - mean Δ (dB)
  - paired SEM (std(Δ) / sqrt(n))
  - paired t = mean Δ / paired SEM
  - two-sided p (scipy.stats.ttest_rel)
  - effect size (Cohen's d = mean Δ / std(Δ))
  - slice win-rate
  - slice-bootstrap 95% CI for mean Δ

Output: review/0521/F0_paired_t_report.md + review/0521/F0_paired_t_summary.json
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel

REPO = Path(__file__).resolve().parents[1]

# 数据源 (per-slice CSV, n=7403, canonical chain PSNR_clip3)
# Round 17-Prep B75-B78 fix: 全部用 per-slice CSV; V18-cap 是不同 substrate 必须排除
DATA = {
    'V7.best':       'review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv',
    'V13.best':      'review/0521/v13_v14_per_slice/artifacts/v13_best_fullval_psnr_chain_mse_per_slice.csv',
    'V14.best':      'review/0521/v13_v14_per_slice/artifacts/v14_best_fullval_psnr_chain_mse_per_slice.csv',
    'V18.best':      'review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv',
    'V18.last':      'review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv',
    # 注意: V18-cap 在 KL_DRIFT_PER_SLICE.csv 是 direct decode(z_GT) substrate,
    # 跟上面所有 chain rollout PSNR 完全不可比 (~16 dB 绝对量级差距).
    # F0 chain paired-t **完全不**包括 V18-cap. 它走 Stage F0b 单独跑.
}

TIMEPOINTS = ['D20', 'D10', 'D4', 'NORMAL']
PSNR_COL = lambda tp: f'psnr_{tp}'   # canonical clip3 column, NOT psnr_raw_*
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260522

def bootstrap_ci(delta: np.ndarray, *, seed: int, n_boot: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Slice-level nonparametric bootstrap CI for mean Δ.
    This is not patient-level resampling because patient/volume IDs are not
    available in the current artifacts.
    """
    rng = np.random.default_rng(seed)
    n = int(delta.shape[0])
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(delta, size=n, replace=True)
        means[i] = float(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

def load_per_slice(path: str, timepoint: str) -> np.ndarray:
    """Load per-slice canonical PSNR_clip3 column for given timepoint.
    Schema: CSV with header line, columns include 'slice_idx', 'psnr_D20',
    'psnr_D10', 'psnr_D4', 'psnr_NORMAL' (clip3) and 'psnr_raw_*' (raw).
    We use 'psnr_<tp>' which is the clip3 metric (canonical per Round 16).
    """
    import csv
    col = PSNR_COL(timepoint)
    vals = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        if col not in rdr.fieldnames:
            raise ValueError(f'{path}: missing column {col}; have {rdr.fieldnames}')
        for row in rdr:
            vals.append(float(row[col]))
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) != 7403:
        raise ValueError(f'{path}: expected 7403 slices, got {len(arr)}')
    return arr

def main():
    out = {}
    ref = load_per_slice(DATA['V7.best'], 'NORMAL')
    n_ref = len(ref)
    assert n_ref == 7403, f"expected n=7403, got {n_ref}"
    for tp in TIMEPOINTS:
        ref_tp = load_per_slice(DATA['V7.best'], tp)
        out[tp] = {}
        for name, path in DATA.items():
            if name == 'V7.best':
                continue
            arr = load_per_slice(path, tp)
            assert len(arr) == n_ref, f"{name}@{tp}: n={len(arr)} != {n_ref}"
            delta = arr - ref_tp
            mean = float(delta.mean())
            std  = float(delta.std(ddof=1))
            sem  = std / np.sqrt(n_ref)
            t, p = ttest_rel(arr, ref_tp)
            d    = mean / std if std > 0 else float('inf')
            win_rate = float((delta > 0).mean())
            seed = BOOTSTRAP_SEED + sum(ord(c) for c in f'{tp}:{name}')
            ci_lo, ci_hi = bootstrap_ci(delta, seed=seed)
            out[tp][name] = {
                'n':        n_ref,
                'mean_dB':  mean,
                'std_dB':   std,
                'sem_dB':   sem,
                'bootstrap_ci95_low_dB': ci_lo,
                'bootstrap_ci95_high_dB': ci_hi,
                't':        float(t),
                'p_value':  float(p),
                'cohen_d':  d,
                'win_rate': win_rate,
                'sig_5pct': bool(p < 0.05),
                'sig_1pct': bool(p < 0.01),
            }
    # 写 JSON
    out_json = REPO / 'review/0521/F0_paired_t_summary.json'
    out_json.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_json}")
    # 写 Markdown
    out_md = REPO / 'review/0521/F0_paired_t_report.md'
    write_markdown(out, out_md)
    print(f"Saved {out_md}")

def write_markdown(out: dict, path: Path):
    lines = [
        '# F0 Slice-Level Paired Statistics Report',
        '',
        '- date: 2026-05-22',
        '- ref: V7.best',
        '- n=7403 validation slices',
        '- metric: canonical PSNR_clip3 (`psnr_<timepoint>` columns)',
        '- caveat: patient/volume IDs are unavailable, so t/p/bootstrap are slice-level statistics, not patient-level independent inference.',
        '',
    ]
    for tp, comps in out.items():
        lines.append(f'## {tp}')
        lines.append('')
        lines.append('| ckpt | mean Δ (dB) | bootstrap 95% CI (dB) | SEM (dB) | t | p | Cohen d | win-rate | sig 5% | sig 1% |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|')
        for name, r in comps.items():
            ci = f"[{r['bootstrap_ci95_low_dB']:+.4f}, {r['bootstrap_ci95_high_dB']:+.4f}]"
            lines.append(f"| {name} | {r['mean_dB']:+.4f} | {ci} | {r['sem_dB']:.5f} | {r['t']:+.2f} | {r['p_value']:.3e} | {r['cohen_d']:+.4f} | {100.0*r['win_rate']:.1f}% | {'✓' if r['sig_5pct'] else '✗'} | {'✓' if r['sig_1pct'] else '✗'} |")
        lines.append('')
    path.write_text('\n'.join(lines))

if __name__ == '__main__':
    main()
```

### F0.3 F0 pass 准则

- `review/0521/F0_paired_t_report.md` 存在并含 4 timepoint × ≥4 ckpt 对比表
- `review/0521/F0_paired_t_summary.json` 存在
- 每个 slice-level paired 行含 mean Δ, bootstrap CI, paired SEM, t, p, Cohen d, win-rate, 显著性 flag
- 报告必须明确声明: patient/volume IDs 不可用, 因此这是 slice-level evidence, 不是 patient-level independent inference
- 任何 `assert n == 7403` 失败 → 立即停止, 不写 report

### F0.4 F0 输出解读 (留给 user 整合)

F0 完成后, user 会看 mean Δ / win-rate / bootstrap CI / slice-level p-value 决定 V18 paper 命运:

- 若 V18.best vs V7 mean Δ > 0, bootstrap CI 不跨 0, win-rate > 50%, 且 slice-level p < 0.01 → V18 可入 paper ablation 表, 但 wording 限定为 slice-level evidence
- 若 V18.best vs V7 bootstrap CI 跨 0 或 win-rate ≈ 50% → V18 不写 "+0.03 dB" 正向 claim, 只写 capacity/transport decoupling
- V14 vs V7 (期望 mean ≈ 0, win-rate ≈ 50%) 是 sanity check, 验证 paired protocol 本身
- V13 vs V7 (期望 mean ≈ -0.29, win-rate < 50%) 是 sanity check, 验证 image_aux 主效应

---

## §3 — A4: Image_aux Schedule Probe (GPU 空闲时做)

### A4.0 前置确认

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only origin foc_lite_hop0
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}

# GPU 状态 — 必须 ≥ 1 个全 free GPU
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# CLI flag 仍是 --config --resume (B30 standing rule)
SUPPORTED=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ' | sed 's/ $//')
EXPECTED="--config --resume"
if [ "$SUPPORTED" != "$EXPECTED" ]; then
    echo "FAIL: CLI flag changed. Detected: '$SUPPORTED'. STOP."
    exit 1
fi
echo "CLI flags OK ✓"
```

### A4.1 yaml 创建 (机械步骤)

**A4 = V7 yaml 完全克隆, 仅改 4 个字段**:

| 字段 | V7 原值 | A4 新值 | 含义 |
|---|---|---|---|
| `output_dir` | `<V7 output_dir>` | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08` | 隔离输出 |
| `run_name` | `first_hop_224_v7_gronwall_raw` | `first_hop_224_a4_image_aux_lambda_08` | 区分 |
| `training.image_aux.lambda_start` | `0.04` | `0.08` | **唯一功能变量** |
| `training.image_aux.lambda_max` | `0.04` | `0.08` | **唯一功能变量 (与 lambda_start 同步)** |

**Round 17-Prep B75 fix**: image_aux schedule 真实位置是 `training.image_aux.*`, **不是** `transport.image_aux.*`. 后者在 V7 yaml 不存在, 用错命名空间会 KeyError. `loss.image_aux.*` 是另一组字段 (l1/ssim/seam/border weights), 不要改.

**绝对不**改: seed (=42), max_steps (=160000), step_weights (Grönwall raw), best_select_full_eval_interval, hop0_coverage_target, lambda_kl (=0), backbone path, transport method, image_aux 内部 l1/ssim/seam/border weights, lr_schedule.

```bash
# 创建 A4 目录
mkdir -p review/0521/A4_image_aux_lambda_08

# 克隆 V7 yaml
cp review/0505/local/configs/V7_gronwall_raw.yaml review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml

# 改 4 字段 (用 python 改 yaml, 不要手编辑, 防破坏注释外结构)
"$PYTHON" << 'PY'
from pathlib import Path
import yaml

p = Path('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml')
d = yaml.safe_load(p.read_text())

d['output_dir'] = '/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08'
d['run_name']   = 'first_hop_224_a4_image_aux_lambda_08'
d['training']['image_aux']['lambda_start'] = 0.08
d['training']['image_aux']['lambda_max']   = 0.08

p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print('A4 yaml updated.')
PY

# 验证只改了 4 个字段
"$PYTHON" << 'PY'
import yaml
v7 = yaml.safe_load(open('review/0505/local/configs/V7_gronwall_raw.yaml'))
a4 = yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml'))

def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

f7 = flatten(v7); f4 = flatten(a4)
diff = {k: (f7.get(k), f4.get(k)) for k in set(f7) | set(f4) if f7.get(k) != f4.get(k)}
ALLOWED = {'output_dir', 'run_name', 'training.image_aux.lambda_start', 'training.image_aux.lambda_max'}
unexpected = set(diff) - ALLOWED
if unexpected:
    print(f'FAIL: unexpected diffs: {unexpected}'); raise SystemExit(1)
print(f'A4 yaml diff verified ✓ (exactly {len(diff)} fields changed: {set(diff)})')
PY

# commit yaml
git add review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml
git commit -m "Round 17 A4: image_aux lambda 0.04 → 0.08 schedule probe yaml"
```

### A4.2 launch

```bash
TS=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG=review/0521/A4_image_aux_lambda_08/A4_train_${TS}.log

# 选择 free GPU (Round 17-Prep B80 fix: fail-loud, 不用 <...> 占位符)
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | nl -v0 | sort -k2 -rn | head -1 | awk '{print $1}')
if [ -z "$FREE_GPU" ]; then echo "FAIL: no free GPU detected"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$FREE_GPU

START_TS=$(date +%s)
echo "A4 launch START_TS=$START_TS GPU=$FREE_GPU"

nohup "$PYTHON" train_first_hop.py \
  --config review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml \
  > "$LAUNCH_LOG" 2>&1 &
A4_PID=$!

echo "A4 launched: PID=$A4_PID, log=$LAUNCH_LOG"

# 5 分钟后存活检查
sleep 300
if ! ps -p $A4_PID > /dev/null; then
    echo "FAIL: A4 process died within 5 min."
    tail -100 "$LAUNCH_LOG"
    exit 1
fi
echo "A4 alive at +5min ✓"

# 验证 log 里 lambda_img 是 0.08
grep -E "lambda_img=0\.0800" "$LAUNCH_LOG" | head -1 || echo "WARN: lambda_img=0.08 not yet in log (may be too early)"
```

### A4.3 训练监控

| 时点 | 检查项 |
|---|---|
| +5 min | 进程存活; 若仍在加载 115GB train latents, 这是预期冷启动 IO, 不要误判为 hang |
| +1 h | 至少已有训练 step 日志, 且首条 train line 含 `lambda_img=0.0800` |
| +12 h | 期望接近或超过 step=5000, 并开始出现 full-val/best-selection 相关日志 (按 V7 schedule) |
| +24 h | step ≥ 30000, 平均训练速度 ≈ V7 历史值 ±20% |
| +5 days | step ≈ 120000 |
| +7 days | step = 160000, `Training done.` |

### A4.4 训练完成后的 canonical full-val eval

```bash
A4_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08
A4_EVAL_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08/fullval_eval
REPO_A4_EVAL=review/0521/A4_image_aux_lambda_08/fullval_eval

mkdir -p "$A4_EVAL_OUT" "$REPO_A4_EVAL/artifacts" "$REPO_A4_EVAL/logs"

for tag in best last; do
    "$PYTHON" review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
        --config $A4_OUT/config.yaml \
        --checkpoint $A4_OUT/${tag}.pt \
        --out-dir "$A4_EVAL_OUT" \
        --tag a4_image_aux_lambda_08_${tag} \
        --split val \
        --max-slices 0 \
        --batch-size 8 \
        --decode-mode both \
        2>&1 | tee "$REPO_A4_EVAL/logs/a4_${tag}_eval_$(date +%Y%m%d_%H%M%S).log"

    cp "$A4_EVAL_OUT"/a4_image_aux_lambda_08_${tag}_fullval_psnr_chain_mse.json "$REPO_A4_EVAL/artifacts/"
    cp "$A4_EVAL_OUT"/a4_image_aux_lambda_08_${tag}_fullval_psnr_chain_mse_per_slice.csv "$REPO_A4_EVAL/artifacts/"
done
```

### A4.5 A4 pass / stop 准则

完成后写 `review/0521/A4_image_aux_lambda_08/A4_REPORT.md` 含:

```markdown
# A4 Image_aux Lambda 0.08 Probe — Report

| ckpt | step | D20 | D10 | D4 | NORMAL | vs V7 (+0.04) baseline |
|---|---:|---:|---:|---:|---:|---:|
| A4.best | ? | ? | ? | ? | ? | +/- ? dB |
| A4.last | 160000 | ? | ? | ? | ? | +/- ? dB |
| V7.best | 160000 | 35.4354 | 35.8194 | 36.3736 | **36.7810** | (ref) |

## Verdict

- 若 A4.best NORMAL ≥ 36.831 (V7 + 0.05) → SUCCESS, image_aux 未饱和, 加入 paper headline
- 若 36.781 ≤ A4.best NORMAL < 36.831 → MARGINAL, image_aux 接近饱和, paper 提一笔
- 若 A4.best NORMAL < 36.781 (低于 V7) → REGRESSION, 0.04 已是 sweet spot, paper 作 negative ablation
- **无论 outcome, 不 relaunch A4-v2 / v3** (Round 17 stop rule, 防 scope creep B74)
```

---

## §4 — Commit / Push 协议

### 4.1 文件命名 (与 Round 16/17 风格一致)

- yaml: `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`
- launch log: `review/0521/A4_image_aux_lambda_08/A4_train_<TS>.log`
- metrics 快照: `review/0521/A4_image_aux_lambda_08/A4_metrics_<TS>.jsonl`
- full-val JSON: `review/0521/A4_image_aux_lambda_08/fullval_eval/artifacts/a4_*_fullval_psnr_chain_mse.json`
- full-val logs: `review/0521/A4_image_aux_lambda_08/fullval_eval/logs/a4_*_eval_*.log`
- F0 报告: `review/0521/F0_paired_t_report.md` + `review/0521/F0_paired_t_summary.json`
- F0 脚本: `tools/paired_t_v18_vs_v7.py`

### 4.2 Commit 顺序 (atomic)

```bash
# Commit 1: F0 脚本 + V13/V14 per-slice 数据 (若需要)
git add tools/paired_t_v18_vs_v7.py
git add review/0521/v13_v14_per_slice/ 2>/dev/null  # 若 codex 跑了 V13/V14 per-slice eval
git commit -m "Round 17 F0: paired-t script + V13/V14 per-slice canonical eval"

# Commit 2: F0 报告
git add review/0521/F0_paired_t_report.md review/0521/F0_paired_t_summary.json
git commit -m "Round 17 F0: paired-t significance report (V13/V14/V18 vs V7)"

# Commit 3: A4 yaml
git add review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml
git commit -m "Round 17 A4: image_aux lambda 0.04 → 0.08 schedule probe yaml"

# Commit 4: A4 launch log + metrics 快照 (训练完成后)
cp $A4_OUT/metrics.jsonl review/0521/A4_image_aux_lambda_08/A4_metrics_<TS>.jsonl
git add review/0521/A4_image_aux_lambda_08/A4_train_*.log
git add review/0521/A4_image_aux_lambda_08/A4_metrics_*.jsonl
git commit -m "Round 17 A4: training log + metrics snapshot (step=160000 done)"

# Commit 5: A4 full-val eval + report
git add review/0521/A4_image_aux_lambda_08/fullval_eval/
git add review/0521/A4_image_aux_lambda_08/A4_REPORT.md
git commit -m "Round 17 A4: canonical full-val eval + verdict report"

# 全部 commit 后 push
git push origin foc_lite_hop0
```

### 4.3 中间状态汇报 (建议 codex 在 push 后给 user 一条简短摘要)

格式:
```
[Round 17 codex status]
- F0: done / pending / failed (附 V18 vs V7 p-value if done)
- A4: not launched / training step X/160000 / done (附 NORMAL PSNR if done)
- 下一动作: <e.g. wait V13/V14 already done, just A4 monitor>
```

---

## §5 — 机械自检脚本

在所有 commit 完成前, codex 必须跑一遍下面的自检:

```bash
#!/usr/bin/env bash
set -u
PYTHON=${PYTHON:-/home/qujiaxiang/.conda/envs/rae/bin/python}
ERR=0
check() {
    local label="$1"; shift
    if "$@"; then echo "PASS: $label"; else echo "FAIL: $label"; ERR=$((ERR+1)); fi
}
anti_check() {
    local label="$1"; shift
    if "$@"; then echo "FAIL (should NOT exist): $label"; ERR=$((ERR+1)); else echo "PASS (correctly absent): $label"; fi
}

# Row 1: F0 脚本存在
check "F0 script present" test -f tools/paired_t_v18_vs_v7.py

# Row 2: F0 报告存在 (若 F0 已跑)
if [ -f review/0521/F0_paired_t_summary.json ]; then
    check "F0 report present" test -f review/0521/F0_paired_t_report.md
    check "F0 summary 含 V18.best 行" grep -q '"V18.best"' review/0521/F0_paired_t_summary.json
    check "F0 summary 含 V13.best 行" grep -q '"V13.best"' review/0521/F0_paired_t_summary.json
fi

# Row 3: A4 yaml 存在 (若 A4 已起)
if [ -d review/0521/A4_image_aux_lambda_08 ]; then
    check "A4 yaml present" test -f review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml
    check "A4 yaml lambda 0.08" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml')); assert y['training']['image_aux']['lambda_max']==0.08, f'got {y[\"training\"][\"image_aux\"][\"lambda_max\"]}'"
    check "A4 yaml seed=42" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml')); assert y['seed']==42"
    check "A4 yaml max_steps=160000" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml')); assert y['training']['max_steps']==160000"
fi

# Row 4: 没有意外文件
anti_check "no V19 yaml" test -f review/0521/V19_decoder_lora.yaml
anti_check "no V18-clean yaml" find review/0521 -name '*v18*clean*.yaml' 2>/dev/null | grep -q .
anti_check "no extra A4 variant" bash -c "[[ \$(find review/0521 -name 'A4_image_aux_lambda_*.yaml' 2>/dev/null | wc -l) -gt 1 ]]"

# Row 5: CLI flag 没被改
SUP=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ' | sed 's/ $//')
check "CLI flags = '--config --resume'" test "$SUP" = "--config --resume"

# Row 6: train_first_hop.py:2230 B42 未变
check "B42 line still verbatim" grep -q 'z_kl = main_out\["z_pred"\] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch\["z_dst"\]' train_first_hop.py

# Row 7: V7 / V13 / V14 / V18 既有 ckpt 未被改
for f in \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off/best.pt \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V14_true_d_pure/run/first_hop_224_v14_v7_seed1337/best.pt \
    /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/best.pt; do
    check "existing ckpt untouched: $(basename $(dirname $f))/$(basename $f)" test -f "$f"
done

# 汇总
echo "---"
if [ $ERR -gt 0 ]; then
    echo "SELF-CHECK FAILED ($ERR errors)"
    exit 1
fi
echo "SELF-CHECK PASSED"
```

任何 PASS/FAIL 不达预期 → 立即停止, 不 push, 报告给 user.

---

## §6 — NOT-DO 清单 (Round 17 拒绝列表)

| ❌ 不允许 | 理由 |
|---|---|
| 改 train_first_hop.py | Round 16/17 已固化代码, 无变量重设计需求 |
| 改 V7 yaml 本身 | 任何 V7 yaml 修改污染 V7 baseline |
| 改 V18 / V13 / V14 已有 ckpt 或 yaml | 它们是已 published 的实验基线 |
| 启 V19 / V18-clean / V18-rank-sweep | Round 16 已签 KL direct-decoder 死, decoder 不是 chain bottleneck |
| 启 V14b / V14c (多 seed) | F0 paired-t 更便宜地回答显著性 |
| 启 A4-v2 / A4-v3 / 其它 image_aux variant | Round 17 stop rule, 防 scope creep (B74) |
| 启 architecture pivot / data pivot 类实验 | 出 Round 17 范围 |
| F0 跳过 V14 vs V7 / V13 vs V7 sanity check | sanity check 是 F0 协议本身的验证 |
| F0 把 V18-cap (direct decode) 与 V7 chain PSNR 混算 paired-t | substrate 不一致 (direct decode vs chain rollout) |
| 在 paper 草稿里宣称 V18 +0.06 dB 显著, 在 F0 paired-t 出来前 | Round 17 B61: 单次 seed |Δ| 不是 σ |
| 把 V18-cap.last 命名为 "A3" 在新 doc 里 | 历史已用 A3 = V18-capacity-only run, 新概念用新名 (B64) |
| 在没有 user 复审前 push 任何 narrative / claim 类 markdown | 仅数据 / 报告 / yaml 类 push, 解读类 push 需 user 签 |

---

## §7 — 资料目录 (codex 阅读顺序建议)

1. 本 task md (本文件)
2. [REVIEW_INTEGRATION_round17_20260522.md](./REVIEW_INTEGRATION_round17_20260522.md) (4/4 共识背景)
3. [PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md](./PEER_REVIEW_PROMPT_round17_strategy_post_V13V14_20260521.md) (战略选项原文)
4. [V13_V14_full_eval_analysis_20260521.md](../0516/V13_V14_full_eval_analysis_20260521.md) (V13/V14 数字)
5. [V7_gronwall_raw.yaml](../0505/local/configs/V7_gronwall_raw.yaml) (A4 yaml 克隆基准)
6. [eval_first_hop_fullval_psnr_chain_mse.py](../0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py) (canonical eval 脚本)
7. [PLANF_FINAL_ANALYSIS_20260516.md](../0516/PLANF_FINAL_ANALYSIS_20260516.md) (paired-t 同型先例)

---

## §8 — 完成后的预期 push 状态

```
origin/foc_lite_hop0 (gitee) latest commits (in order):
  <hash> Round 17 F0: paired-t script + V13/V14 per-slice canonical eval
  <hash> Round 17 F0: paired-t significance report (V13/V14/V18 vs V7)
  <hash> Round 17 A4: image_aux lambda 0.04 → 0.08 schedule probe yaml
  <hash> Round 17 A4: training log + metrics snapshot (step=160000 done)  [若 A4 完成]
  <hash> Round 17 A4: canonical full-val eval + verdict report           [若 A4 完成]
```

如果 GPU 紧张, 只完成 F0 也 OK; A4 可以后做.

---

## §9 — 给 codex 的最后提醒

- 任何 yaml 字段不确定 → 不要猜, 先 grep 现有 V7 yaml 看实际 key 名
- 任何 eval 脚本 flag 不确定 → 先读脚本 head + `--help`
- 任何 ckpt 路径不在 disk → 先 `find /data_2 -name 'best.pt' -path '*V7*'`
- 训练 5 min 没 alive → 立即 tail log, 不 silent
- 任何与 NOT-DO 清单冲突的动作 → 立即停止, 不要 "顺手做完"

完成所有任务后, 给 user 一句话:

```
Round 17 codex done: F0 paired-t out (V18.best vs V7 p=X), A4 [pending/training/done with NORMAL=Y dB]. Awaiting user Round 18 decision.
```
