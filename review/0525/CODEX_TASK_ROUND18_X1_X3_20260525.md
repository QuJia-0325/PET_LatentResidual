# Codex Task — Round 18 (X1-lite + X3 Parallel Execution)

- date: 2026-05-25
- branch: foc_lite_hop0
- status: **READY FOR CODEX EXECUTION (commit a764742, after Round 18-Prep 4 HIGH + 6 MED fixes)**
- 上游: Round 18 集成 ([REVIEW_INTEGRATION_round18_20260525.md](./REVIEW_INTEGRATION_round18_20260525.md)) 4/4 共识 = X1-lite + X3 并行
- 上游 code review: Round 18-Prep 集成 ([REVIEW_INTEGRATION_round18_prep_20260525.md](./REVIEW_INTEGRATION_round18_prep_20260525.md)) 3/3 MODIFY-BEFORE-PUSH, 全部修复已 land
- user 决策 2026-05-25: X5/X6 暂搁置 (无外部数据 / 无 clinical metadata), 仅做 X1-lite + X3
- 硬约束: ≤ 3 并行训练; X1-lite + X3 = 2 slot, slot 3 留 buffer

---

## §0 — 给 codex 的 5 句话总览

1. **本 task = 2 个独立训练实验, 并行 launch**: X1-lite (mechanism falsification) + X3 (image_aux + LoRA additive test).
2. X1-lite = V7 yaml clone + image_aux λ=0.08 (= A4-mid) + ssim/seam weights 全 0 (只保留 l1). from-scratch 训到 160K, ~7d, slot 1.
3. X3 = V18 yaml clone + lambda_kl=0 + image_aux λ=0.08. warmstart V7.best @160K, 训到 170K (A3 协议), ~1-2d, slot 2.
4. **绝对不**改 train_first_hop.py / V7 yaml / V18 已有 ckpt / A4 已有 ckpt. **绝对不**复活 V18-clean / V19 / 任何 λ ∉ {0.02, 0.08} 的新 A4 变体.
5. 两实验都完成后用 canonical full-val eval 评 best.pt + last.pt, 与 V7 / V13 / V14 / V18 / A3 / A4-low / A4-mid 一起放入统一表.

---

## §1 — 物理含义 (user 必读)

### 1.1 上游已建立事实 (Round 18 共识)

| run | image_aux λ | LoRA | NORMAL PSNR_clip3 | Δ vs V7 |
|---|---:|---|---:|---:|
| V13 | 0.00 | 无 | 36.4943 | −0.287 |
| A4-low | 0.02 | 无 | 36.7010 | −0.080 |
| V7 | 0.04 | 无 | 36.7810 | 0 |
| V18.best @165K | 0.04 | r32 last-2 + KL | 36.8112 | +0.030 |
| V18.last @200K | 0.04 | r32 last-2 + KL | 36.8426 | +0.062 |
| A3 (V18-cap) @170K | 0.04 | r32 last-2 (no KL) | (≈ V18.step170k chain) | ≈ +0.030 |
| **A4-mid** | **0.08** | **无** | **36.8939** | **+0.113** |

### 1.2 X1-lite (mechanism falsification)

**问题**: A4-mid +0.113 dB 是因为 (a) pixel L1 supervision 提供 transport DiT 缺失的梯度方向 (M1+M4), 还是因为 (b) SSIM + seam 子项提供 structure-aware 信号 (M2+M3)?

**实验**: V7 yaml + 把 image_aux **λ 提到 0.08** (= A4-mid 强度), 但把 ssim_weight 和 seam_weight **置零** (只保留 l1).

**判读**:
- 若 X1-lite ≈ A4-mid (差 < 0.02 dB) → SSIM + seam 子项贡献 ≈ 0, 主因是 pixel L1 强 supervision (**M1 + M4 主推**, paper Methods 可以写 "pixel-space L1 gradient through frozen decoder is sufficient to explain the gain")
- 若 X1-lite 明显 < A4-mid (差 > 0.05 dB) → SSIM / seam 携带不可忽略信号 (M2 / M3 上升, 需要进一步拆)
- 若 X1-lite 介于 V7 (36.7810) 和 A4-mid (36.8939) 之间 → l1 不饱和, 但 ssim+seam 贡献也非 0; 写 "pixel L1 contributes X dB, structure-aware terms contribute Y dB"

**这不是 sweet-spot search**. λ=0.08 已经 lock (来自 A4-mid). 关闭 ssim/seam 是**机制 falsification**, 不是 component tuning. (Round 18 集成 B89: "mechanism ablation 不是 tuning".)

### 1.3 X3 (image_aux + decoder LoRA additive test)

**问题**: A4-mid 给 +0.113 dB (无 LoRA). V18 给 +0.062 dB (有 LoRA, image_aux=0.04). 如果把 image_aux 提到 0.08 + 同样 LoRA + 杀 KL, 总收益是 A4-mid + LoRA-marginal (additive), 还是 ≈ A4-mid (LoRA 被 image_aux 吸收), 还是 < A4-mid (gradient 冲突)?

**实验**: V18 yaml + lambda_kl=0 (从 A3 借) + image_aux λ=0.08 (从 A4-mid 借). 同 A3 协议: 从 V7.best @160K warmstart + LoRA + 训 10K 步到 170K.

**判读**:
- 若 X3.last@170K ≈ A4-mid + (A3 - V7) → 完全 additive, V18 LoRA 还有独立价值; **paper 保留 V18 ablation 加深 "decoder capacity + image_aux additive" 子结**
- 若 X3.last@170K ≈ A4-mid (LoRA marginal ≈ 0) → V18 LoRA 在强 image_aux 下冗余; **paper 完全把 V18 降级**
- 若 X3.last@170K < A4-mid → LoRA + 高 image_aux 梯度冲突; **V18 family 死, 不必再讨论**

**关键对比**: X3 vs A3 (V18-cap.last@170K @ λ=0.04). 同协议, 仅 image_aux λ 差. X3 − A3 ≈ A4-mid − V7 = +0.113 dB? 这测**在 LoRA 在场情况下 image_aux 提升的"复用率"**.

### 1.4 不做什么 (Round 18 共识拒绝清单)

- ❌ X1 全 3-run (l1-only / ssim-only / seam-only 各 1 run, 21d) — 4/4 reviewer 拒绝, 改为 X1-lite (1 run 答 M1+M4 vs M2+M3 主问题)
- ❌ X2 / X4 任何设计或 spike — Round 18 4/4 一致 REJECT 作 immediate next step
- ❌ X3 v2 / v3 — 1 run hard-stop, 不论 outcome 不 relaunch
- ❌ V19 / V18-clean / V18-rank-sweep — Round 16-18 一致拒
- ❌ image_aux λ ∉ {0.02, 0.08} 任何新 A4 变体 — Round 17-Slots stop rule
- ❌ V14b/c 多 seed — Round 17-Stats 已签 slice-level only

---

## §2 — Task A: X1-lite Launch (slot 1, from-scratch 7d)

### A.0 yaml 创建 (机械步骤)

X1-lite = V7 yaml clone, 仅改 **6 个字段**:

| 字段 | V7 原值 | X1-lite 新值 | 含义 |
|---|---|---|---|
| `output_dir` | (V7) | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X1_lite_l1_only` | 隔离 |
| `run_name` | `first_hop_224_v7_gronwall_raw` | `first_hop_224_x1_lite_l1_only` | 区分 |
| `training.image_aux.lambda_start` | 0.04 | **0.08** | 与 A4-mid 同强度 |
| `training.image_aux.lambda_max` | 0.04 | **0.08** | 同上 |
| `loss.image_aux.ssim_weight` | 0.25 | **0.0** | 关闭 SSIM 子项 |
| `loss.image_aux.seam_weight` | 0.1 | **0.0** | 关闭 seam 子项 |

**绝对不**改: seed (=42), max_steps (=160000), step_weights (Grönwall raw), lambda_kl, image_aux 其它子项 (l1_weight, border_*, seam_patch_size), lr_schedule, backbone, transport, hop0_coverage_target.

### A.1 launch sequence

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
git pull --ff-only origin foc_lite_hop0

PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python
export PYTHONPATH=/home/qujiaxiang/project/PET_LatentResidual:${PYTHONPATH:-}

# CLI 检查
SUPPORTED=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ' | sed 's/ $//')
EXPECTED="--config --resume"
[ "$SUPPORTED" = "$EXPECTED" ] || { echo "FAIL CLI"; exit 1; }

# 创建 yaml
mkdir -p review/0525/X1_lite_l1_only
cp review/0505/local/configs/V7_gronwall_raw.yaml review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml

"$PYTHON" << 'PY'
from pathlib import Path
import yaml
p = Path('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')
d = yaml.safe_load(p.read_text())
d['output_dir'] = '/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X1_lite_l1_only'
d['run_name']   = 'first_hop_224_x1_lite_l1_only'
d['training']['image_aux']['lambda_start'] = 0.08
d['training']['image_aux']['lambda_max']   = 0.08
d['loss']['image_aux']['ssim_weight'] = 0.0
d['loss']['image_aux']['seam_weight'] = 0.0
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print('X1-lite yaml updated.')
PY

# 验证 6 字段 diff
"$PYTHON" << 'PY'
import yaml
v7 = yaml.safe_load(open('review/0505/local/configs/V7_gronwall_raw.yaml'))
x1 = yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml'))
def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict): out.update(flatten(v, key))
        else: out[key] = v
    return out
f7 = flatten(v7); f1 = flatten(x1)
diff = {k: (f7.get(k), f1.get(k)) for k in set(f7) | set(f1) if f7.get(k) != f1.get(k)}
ALLOWED = {'output_dir', 'run_name', 'training.image_aux.lambda_start', 'training.image_aux.lambda_max', 'loss.image_aux.ssim_weight', 'loss.image_aux.seam_weight'}
unexpected = set(diff) - ALLOWED
if unexpected: print(f'FAIL unexpected diff: {unexpected}'); raise SystemExit(1)
print(f'X1-lite diff OK ({len(diff)} fields): {sorted(diff.keys())}')
PY

# free GPU 选择
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | nl -v0 | sort -k2 -rn | head -1 | awk '{print $1}')
[ -z "$FREE_GPU" ] && { echo "FAIL no free GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES=$FREE_GPU

TS=$(date +%Y%m%d_%H%M%S)
X1_LOG=review/0525/X1_lite_l1_only/X1_lite_train_${TS}.log

nohup "$PYTHON" train_first_hop.py \
  --config review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml \
  > "$X1_LOG" 2>&1 &
X1_PID=$!
echo "X1-lite launched: PID=$X1_PID GPU=$FREE_GPU log=$X1_LOG"

# Round 18-Prep H2 fix: persist PID/GPU to sidecar files for X3 staggered launch.
# (PID is shell-local; trainer log does NOT contain PID= line.)
echo "$X1_PID" > review/0525/X1_lite_l1_only/X1_lite.pid
echo "$FREE_GPU" > review/0525/X1_lite_l1_only/X1_lite.gpu

# 5min 存活 + lambda_img / ssim_weight verify
sleep 300
ps -p $X1_PID > /dev/null || { echo "FAIL X1 dead"; tail -100 "$X1_LOG"; exit 1; }
grep -q 'lambda_img=0\.0800' "$X1_LOG" || echo "WARN: lambda_img=0.08 not yet logged"
# Round 18-Prep M3 fix: ssim/seam loss is ALWAYS computed (just multiplied by 0 in total).
# img_ssim log field shows RAW (unweighted) SSIM loss, NOT 0. Health check should verify
# img ≈ img_l1 (total ≈ L1 component) instead of expecting img_ssim ≈ 0.
grep -E 'img=[0-9.eE+-]+ img_l1=[0-9.eE+-]+' "$X1_LOG" | head -3
echo "X1-lite alive at +5min"
```

### A.2 X1-lite 监控

| 时点 | 检查项 | 处理动作 |
|---|---|---|
| +5 min | 进程存活 + log 含 `lambda_img=0.0800` + `img ≈ img_l1` (SSIM/seam loss raw values 仍计算但 total 中权重为 0; img 总值 ≈ L1 分量) | 任何失败 → 立即 kill, tail log, push status |
| +1 h | ≥ 1 条 `[val]` 行 | 记录 baseline rolling val |
| +12 h | step ≥ 5000, `step_5000.pt` 存在 | 仅 log, 不 kill |
| +24 h (step ≈ 30K) | 记录当前 rolling val_chain_normal_mse + img_frac | **仅诊断, 不 auto-kill**; 若 val_chain_normal_mse > 0.0015 → codex 在 task md 同目录写 `MANUAL_REVIEW_NEEDED_step30K.md` 含 5 行取样 + 当前 val_chain_normal_mse 值 + V7 baseline, **继续训练** |
| +48 h (step ≈ 50K) | 首个 full-val ckpt; 与 V7 step-50K 全验证 baseline 对比 | **仅诊断, 不 auto-kill**; 若 full-val NORMAL MSE 高于 V7 baseline > 10% → 同上写 `MANUAL_REVIEW_NEEDED_step50K.md`, 继续训练 |
| +7 d | step = 160000, `Training done.` | 进入 eval 阶段 |
| +7 d | step = 160000, `Training done.` | 进入 eval 阶段 |

**Round 18-Prep M4 fix**: 原提议的 "+24h step 30K 与 V7 rolling val 差 > 5e-5 → auto-kill" 被 3/3 reviewer 以阅证据否定 (V7 与 A4-mid 在 step 30K rolling val 本身几乎重合 ~0.00093). Rolling val 噪声太大, 单点阈值不能成为 auto-kill 依据. 仅保留 OOM / NaN / 进程死亡这三个 hard kill 条件.

---

## §3 — Task B: X3 Launch (slot 2, warmstart 1-2d)

### B.0 yaml 创建

X3 = V18 yaml clone, 仅改 **8 个字段** (Round 18-Prep M1 + M2 fix 后 从 6 字段扩到 8):

| 字段 | V18 原值 | X3 新值 | 含义 |
|---|---|---|---|
| `output_dir` | (V18) | `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X3_image_aux_lora` | 隔离 |
| `run_name` | `first_hop_224_v18_decoder_lora` | `first_hop_224_x3_image_aux_lora` | 区分 |
| `training.max_steps` | 200000 | **170000** | 同 A3 协议, 10K LoRA 训练 |
| `training.save_interval` | 10000 | **5000** | M2 fix: 与 A3 一致, 产出 `step_165000.pt` + `step_170000.pt` matched-step ckpt |
| `training.image_aux.lambda_start` | 0.04 | **0.08** | 与 A4-mid 同强度 |
| `training.image_aux.lambda_max` | 0.04 | **0.08** | 同上 |
| `loss.decoder_kl_pullback.lambda_kl` | 0.05 | **0.0** | 杀 KL (Round 16 已签 KL 死) |
| `loss.decoder_kl_pullback.enabled` | true | **false** | M1 fix: 避免在 model 构造期建立 frozen RAE reference (VRAM 低 + 语义更净) |

**绝对不**改: seed (=42), resume_from (=V7.best), LoRA rank=32 / blocks=last-2, KL pullback `use_pred_latent` (moot 当 lambda_kl=0), image_aux 子项, lr_schedule, backbone, transport.

### B.1 launch sequence

```bash
# 创建 yaml
mkdir -p review/0525/X3_image_aux_lora
cp review/0517/V18_decoder_lora/V18_decoder_lora.yaml review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml

"$PYTHON" << 'PY'
from pathlib import Path
import yaml
p = Path('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')
d = yaml.safe_load(p.read_text())
d['output_dir'] = '/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X3_image_aux_lora'
d['run_name']   = 'first_hop_224_x3_image_aux_lora'
d['training']['max_steps'] = 170000
d['training']['save_interval'] = 5000
d['training']['image_aux']['lambda_start'] = 0.08
d['training']['image_aux']['lambda_max']   = 0.08
d['loss']['decoder_kl_pullback']['lambda_kl'] = 0.0
d['loss']['decoder_kl_pullback']['enabled'] = False
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print('X3 yaml updated.')
PY

# 验证 6 字段 diff
"$PYTHON" << 'PY'
import yaml
v18 = yaml.safe_load(open('review/0517/V18_decoder_lora/V18_decoder_lora.yaml'))
x3 = yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml'))
def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict): out.update(flatten(v, key))
        else: out[key] = v
    return out
fv = flatten(v18); fx = flatten(x3)
diff = {k: (fv.get(k), fx.get(k)) for k in set(fv) | set(fx) if fv.get(k) != fx.get(k)}
ALLOWED = {'output_dir', 'run_name', 'training.max_steps', 'training.save_interval', 'training.image_aux.lambda_start', 'training.image_aux.lambda_max', 'loss.decoder_kl_pullback.lambda_kl', 'loss.decoder_kl_pullback.enabled'}
unexpected = set(diff) - ALLOWED
if unexpected: print(f'FAIL unexpected diff: {unexpected}'); raise SystemExit(1)
print(f'X3 diff OK ({len(diff)} fields): {sorted(diff.keys())}')
PY
```

### B.2 staggered launch (等 X1-lite +30min health check 后)

```bash
# X1-lite health check (Round 18-Prep H2 fix: read PID/GPU from sidecar, NOT trainer log)
X1_PID=$(cat review/0525/X1_lite_l1_only/X1_lite.pid 2>/dev/null)
X1_GPU=$(cat review/0525/X1_lite_l1_only/X1_lite.gpu 2>/dev/null)
[ -z "$X1_PID" ] && { echo "FAIL X1_PID sidecar missing"; exit 1; }
[ -z "$X1_GPU" ] && { echo "FAIL X1_GPU sidecar missing"; exit 1; }
X1_LOG=$(ls -t review/0525/X1_lite_l1_only/X1_lite_train_*.log | head -1)
ps -p "$X1_PID" > /dev/null || { echo "FAIL X1-lite dead (PID=$X1_PID), abort X3"; exit 1; }
grep -q 'lambda_img=0\.0800' "$X1_LOG" || { echo "FAIL X1 lambda wrong"; exit 1; }
grep -qE 'OOM|out of memory|NaN' "$X1_LOG" && { echo "FAIL X1 has OOM/NaN"; exit 1; }
echo "X1-lite health OK at +30min (PID=$X1_PID GPU=$X1_GPU)"

# 选另一个 free GPU (Round 18-Prep H3 fix: 显式排除 X1 的 GPU)
FREE_GPU2=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -v exclude=$X1_GPU -F',' '$1+0 != exclude {print $1, $2}' \
  | sort -k2 -rn | head -1 | awk '{print $1}')
[ -z "$FREE_GPU2" ] && { echo "FAIL no second free GPU (excluded X1 GPU $X1_GPU)"; exit 1; }
[ "$FREE_GPU2" = "$X1_GPU" ] && { echo "FAIL GPU race: chose same GPU as X1 ($X1_GPU)"; exit 1; }
export CUDA_VISIBLE_DEVICES=$FREE_GPU2

TS=$(date +%Y%m%d_%H%M%S)
X3_LOG=review/0525/X3_image_aux_lora/X3_train_${TS}.log

# Round 18-Prep H1 fix: 显式传 --resume V7.best.pt (yaml training.resume_from 不被 trainer 读)
V7_CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt
[ -f "$V7_CKPT" ] || { echo "FAIL V7.best.pt missing at $V7_CKPT"; exit 1; }

nohup "$PYTHON" train_first_hop.py \
  --config review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml \
  --resume "$V7_CKPT" \
  > "$X3_LOG" 2>&1 &
X3_PID=$!
echo "X3 launched: PID=$X3_PID GPU=$FREE_GPU2 log=$X3_LOG"
echo "$X3_PID" > review/0525/X3_image_aux_lora/X3.pid
echo "$FREE_GPU2" > review/0525/X3_image_aux_lora/X3.gpu

# 5min 存活 + verify
sleep 300
ps -p $X3_PID > /dev/null || { echo "FAIL X3 dead"; tail -100 "$X3_LOG"; exit 1; }
grep -q '\[startup\] resume from:' "$X3_LOG" || { echo "FAIL X3 not resumed (no [startup] resume from: line)"; exit 1; }
grep -q '\[resume\] loaded step=160000' "$X3_LOG" || { echo "FAIL X3 resume wrong step (expected step=160000)"; exit 1; }
grep -q 'lambda_img=0\.0800' "$X3_LOG" || echo "WARN: X3 lambda_img not yet logged"
echo "X3 alive + resumed @ 160K at +5min"
```

### B.3 X3 监控

| 时点 | 检查项 |
|---|---|
| +5 min | 进程存活 + resume @ 160K + `lambda_img=0.0800` + `lambda_kl=0.0000` |
| +6 h | step ≥ 162000 |
| +24 h | step ≥ 167000 |
| +36-48 h | step = 170000, `Training done.` |

---

## §4 — Eval Protocol (X1-lite + X3 训练完后)

### 4.1 canonical full-val eval

```bash
# eval 输出必须在 /data_2/ (path_guard 要求)
F0_EVAL_OUT_BASE=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_eval

# X1-lite
X1_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X1_lite_l1_only/first_hop_224_x1_lite_l1_only
mkdir -p "$F0_EVAL_OUT_BASE/x1_lite_eval" review/0525/X1_lite_l1_only/fullval_eval/{artifacts,logs}

for tag in best last; do
  "$PYTHON" review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
    --config $X1_OUT/config.yaml \
    --checkpoint $X1_OUT/${tag}.pt \
    --tag x1_lite_l1_only_${tag} \
    --split val --max-slices 0 --batch-size 8 --decode-mode both \
    --out-dir "$F0_EVAL_OUT_BASE/x1_lite_eval" \
    2>&1 | tee "review/0525/X1_lite_l1_only/fullval_eval/logs/x1_lite_${tag}_eval_$(date +%Y%m%d_%H%M%S).log"
done
cp "$F0_EVAL_OUT_BASE/x1_lite_eval"/x1_lite_l1_only_*.json review/0525/X1_lite_l1_only/fullval_eval/artifacts/
cp "$F0_EVAL_OUT_BASE/x1_lite_eval"/x1_lite_l1_only_*.csv  review/0525/X1_lite_l1_only/fullval_eval/artifacts/

# X3
X3_OUT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/X3_image_aux_lora/first_hop_224_x3_image_aux_lora
mkdir -p "$F0_EVAL_OUT_BASE/x3_eval" review/0525/X3_image_aux_lora/fullval_eval/{artifacts,logs}

for tag in best last; do
  "$PYTHON" review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
    --config $X3_OUT/config.yaml \
    --checkpoint $X3_OUT/${tag}.pt \
    --tag x3_image_aux_lora_${tag} \
    --split val --max-slices 0 --batch-size 8 --decode-mode both \
    --out-dir "$F0_EVAL_OUT_BASE/x3_eval" \
    2>&1 | tee "review/0525/X3_image_aux_lora/fullval_eval/logs/x3_${tag}_eval_$(date +%Y%m%d_%H%M%S).log"
done
cp "$F0_EVAL_OUT_BASE/x3_eval"/x3_image_aux_lora_*.json review/0525/X3_image_aux_lora/fullval_eval/artifacts/
cp "$F0_EVAL_OUT_BASE/x3_eval"/x3_image_aux_lora_*.csv  review/0525/X3_image_aux_lora/fullval_eval/artifacts/
```

### 4.2 unified report

写 `review/0525/ROUND_18_X1X3_UNIFIED_REPORT_20260525.md` 含:

```markdown
# Round 18 X1-lite + X3 Unified Eval Report

| run | image_aux λ | ssim/seam | LoRA | KL | step | NORMAL | Δ vs V7 | Δ vs A4-mid |
|---|---:|---|---|---:|---:|---:|---:|---:|
| V7.best | 0.04 | full | - | - | 160000 | 36.7810 | 0 | -0.113 |
| V18.best | 0.04 | full | r32 last-2 | on | 165000 | 36.8112 | +0.030 | -0.083 |
| V18.last | 0.04 | full | r32 last-2 | on | 200000 | 36.8426 | +0.062 | -0.051 |
| A3 (V18-cap) | 0.04 | full | r32 last-2 | off | 170000 | ? (查既有) | ? | ? |
| A4-mid | 0.08 | full | - | - | 160000 | 36.8939 | +0.113 | 0 |
| **X1-lite.best** | **0.08** | **off (l1-only)** | - | - | 160000 | ? | ? | ? |
| **X3.last** | **0.08** | full | r32 last-2 | off | 170000 | ? | ? | ? |

## Mechanism Verdict (X1-lite)

- 若 X1-lite ≈ A4-mid (差 < 0.02 dB) → M1+M4 dominant (pixel L1 sufficient)
- 若 X1-lite < A4-mid by 0.02-0.05 dB → M2+M3 partial (SSIM/seam carry some signal)
- 若 X1-lite ≤ V7 → M2+M3 dominant (l1 alone insufficient)

## Additivity Verdict (X3)

- 若 X3 ≈ A4-mid + (V18 - V7) → additive, V18 LoRA 有独立价值
- 若 X3 ≈ A4-mid → LoRA in strong image_aux regime is redundant
- 若 X3 < A4-mid → LoRA + 高 image_aux conflict, V18 family dead

## Recommendation
[等 outcome 后填]
```

---

## §5 — Commit / Push 协议

```bash
# Commit 1: yaml
git add review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml
git commit -m "Round 18 X1-lite + X3 yaml (mechanism falsification + additive test)"

# Commit 2: training logs + metrics (训练完成后)
cp $X1_OUT/metrics.jsonl review/0525/X1_lite_l1_only/X1_lite_metrics_$(date +%Y%m%d_%H%M%S).jsonl
cp $X3_OUT/metrics.jsonl review/0525/X3_image_aux_lora/X3_metrics_$(date +%Y%m%d_%H%M%S).jsonl
git add review/0525/X1_lite_l1_only/X1_lite_train_*.log review/0525/X1_lite_l1_only/X1_lite_metrics_*.jsonl
git add review/0525/X3_image_aux_lora/X3_train_*.log review/0525/X3_image_aux_lora/X3_metrics_*.jsonl
git commit -m "Round 18 X1-lite + X3 training logs + metrics snapshots"

# Commit 3: full-val eval + unified report
git add review/0525/X1_lite_l1_only/fullval_eval/ review/0525/X3_image_aux_lora/fullval_eval/
git add review/0525/ROUND_18_X1X3_UNIFIED_REPORT_20260525.md
git commit -m "Round 18 X1-lite + X3 canonical full-val eval + unified report"

git push origin foc_lite_hop0
```

---

## §6 — 机械自检 (push 前必跑)

```bash
#!/usr/bin/env bash
set -u
ERR=0
check() { local label="$1"; shift; if "$@"; then echo "PASS: $label"; else echo "FAIL: $label"; ERR=$((ERR+1)); fi; }
anti_check() { local label="$1"; shift; if "$@"; then echo "FAIL: $label"; ERR=$((ERR+1)); else echo "PASS: $label"; fi; }

# yaml 存在
check "X1-lite yaml" test -f review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml
check "X3 yaml" test -f review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml

# X1-lite 关键字段
check "X1 ssim_weight=0" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')); assert y['loss']['image_aux']['ssim_weight']==0.0"
check "X1 seam_weight=0" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')); assert y['loss']['image_aux']['seam_weight']==0.0"
check "X1 lambda_max=0.08" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')); assert y['training']['image_aux']['lambda_max']==0.08"
check "X1 seed=42" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')); assert y['seed']==42"
check "X1 max_steps=160000" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')); assert y['training']['max_steps']==160000"
check "X1 lambda_kl=0 (V7 默认)" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X1_lite_l1_only/X1_lite_l1_only.yaml')); assert y['loss'].get('decoder_kl_pullback', {}).get('lambda_kl', 0.0) == 0.0"

# X3 关键字段 (Round 18-Prep M1 + M2 fix)
check "X3 lambda_kl=0" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert y['loss']['decoder_kl_pullback']['lambda_kl']==0.0"
check "X3 KL enabled=false (M1)" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert y['loss']['decoder_kl_pullback']['enabled'] is False"
check "X3 save_interval=5000 (M2)" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert y['training']['save_interval']==5000"
check "X3 lambda_max=0.08" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert y['training']['image_aux']['lambda_max']==0.08"
check "X3 max_steps=170000" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert y['training']['max_steps']==170000"
check "X3 resume_from=V7.best (yaml 记录, 但 trainer 从 CLI --resume 读)" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert 'V7' in y['training']['resume_from']"
check "X3 LoRA rank=32" "$PYTHON" -c "import yaml; y=yaml.safe_load(open('review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml')); assert y['training']['decoder_lora']['rank']==32"

# CLI flag
SUP=$(grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u | tr -d "'\"" | tr '\n' ' ' | sed 's/ $//')
check "CLI flags" test "$SUP" = "--config --resume"

# B42 verbatim
check "B42 line verbatim" grep -q 'z_kl = main_out\["z_pred"\] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch\["z_dst"\]' train_first_hop.py

# 既有 ckpt 未被改 (sanity)
for f in \
  /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0505_runs/V7/run/first_hop_224_v7_gronwall_raw/best.pt \
  /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/best.pt \
  /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0521_runs/A4_image_aux_lambda_08/first_hop_224_a4_image_aux_lambda_08/best.pt; do
  check "ckpt untouched: $(basename $(dirname $f))/$(basename $f)" test -f "$f"
done

# 没有意外文件
# Round 18-Prep H4 fix (v2): 改为 inline check, 避免函数抽象 + 路径 scope bug.
# A4 yaml 实际在 review/0521 (不是 0525), 必须 search `review` 并排除 smoke variants.
A4_COUNT=$(find review -name 'A4_image_aux_lambda_*.yaml' -not -name '*_smoke.yaml' 2>/dev/null | wc -l | tr -d ' ')
if [ "$A4_COUNT" -gt 2 ]; then
    echo "FAIL: A4 variant count > 2 (found $A4_COUNT, allowed: 2 = lambda_02 + lambda_08)"; ERR=$((ERR+1))
else
    echo "PASS: A4 variant count ≤ 2 (found $A4_COUNT, allowed: 2)"
fi

anti_check "no X2 yaml" find review/0525 -name 'X2_*.yaml' 2>/dev/null | grep -q .
anti_check "no V19 yaml" find review/0525 -name 'V19*.yaml' 2>/dev/null | grep -q .
anti_check "no X3 v2" find review/0525 -name 'X3*v2*' 2>/dev/null | grep -q .

# Round 18-Prep H2 fix: X1 PID/GPU sidecar files 存在 (if X1 已 launch)
if [ -f review/0525/X1_lite_l1_only/X1_lite.pid ]; then
    check "X1 PID sidecar present" test -s review/0525/X1_lite_l1_only/X1_lite.pid
    check "X1 GPU sidecar present" test -s review/0525/X1_lite_l1_only/X1_lite.gpu
fi

echo "---"
[ $ERR -gt 0 ] && { echo "SELF-CHECK FAILED ($ERR)"; exit 1; }
echo "SELF-CHECK PASSED"
```

---

## §7 — NOT-DO 清单

| ❌ 不允许 | 理由 |
|---|---|
| 改 train_first_hop.py | 无修复需求 |
| 改 V7 / V18 / A4 已有 yaml 或 ckpt | baseline 保护 |
| 启 X1 v2 (e.g. ssim-only) 在 X1-lite 完成前 | Round 18 共识 X1-lite ≠ 全 X1 |
| 启 X3 v2 / v3 | Round 18 stop rule, 1 run hard-stop |
| 启 X2 / X4 / V19 / V18-clean | Round 16-18 已签拒 |
| 起 A4 第三 / 第四 lambda variant (0.06 / 0.10 / 0.12 / 0.16) | Round 17-Slots stop rule |
| 改 X3 LoRA rank / blocks (e.g. r=64, last-4) | 这是 V18 family 复活, Round 18 拒 |
| 在 paper 草稿宣称 X3 是 V18 救活 | V18 已降级, X3 outcome 是冗余测试不是救活 |
| 把 X1-lite eval 与 V18 chain PSNR 混算 paired-t (Round 17-Stats) | substrate substrate substrate |
| 在 user 复审 unified report 前 push paper-narrative markdown | 数据 push 不需 review, 解读 push 需要 |

---

## §8 — 资料目录

- 本 task md
- [REVIEW_INTEGRATION_round18_20260525.md](./REVIEW_INTEGRATION_round18_20260525.md) (4/4 共识 X1-lite + X3)
- [ROUND_18_A4_BRACKET_ANALYSIS_20260525.md](./ROUND_18_A4_BRACKET_ANALYSIS_20260525.md) (战略分析)
- [A4_BRACKET_FULLVAL_REPORT_20260525.md](../0521/A4_image_aux_lambda_bracket/A4_BRACKET_FULLVAL_REPORT_20260525.md) (A4 结果)
- [V18_decoder_lora.yaml](../0517/V18_decoder_lora/V18_decoder_lora.yaml) (X3 clone 基准)
- [V7_gronwall_raw.yaml](../0505/local/configs/V7_gronwall_raw.yaml) (X1-lite clone 基准)
- [V18_FINAL_RESULTS_20260518.md](../0517/V18_decoder_lora/V18_FINAL_RESULTS_20260518.md) (V18 canonical 对照)

---

## §9 — 完成后 status 摘要格式

```
[Round 18 codex status]
- X1-lite: training step X/160000 / done (best NORMAL=Y dB if done)
- X3: training step X/170000 / done (best NORMAL=Y dB if done)
- 下一动作: 等 user Round 19 决策 (paper 加深 / X5 启动 / 项目封口)
```
