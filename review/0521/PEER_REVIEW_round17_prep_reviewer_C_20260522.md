# Round 17-Prep Peer Review — Reviewer C (independent)

- **reviewer name**: **C** (consistent with R15/R16/R17)
- date: 2026-05-22
- branch: foc_lite_hop0
- substrate read independently (no other reviewer drafts seen):
  - CODEX_TASK_ROUND17_F0_A4_20260522.md (main object)
  - REVIEW_INTEGRATION_round17_20260522.md (4/4 consensus claim)
  - PEER_REVIEW_PROMPT_round17_prep_codex_task_20260522.md (this prompt)
  - review/0516/full_eval_json/v1{3,4}_*_fullval_psnr_chain_mse.json (substrate)
  - review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py (canonical eval script)
  - review/0505/local/configs/V7_gronwall_raw.yaml (A4 clone base)
  - train_first_hop.py lines 856, 1356-1357, 1588, 2095-2100, 2228-2232 (B42 + image_aux schedule loading)
  - review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/*.{json,csv} (V18 actual paths)
  - review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_PER_SLICE.csv (148,061 lines = 1 header + 148,060 data)
- mandate per prompt §7: review is to catch fabrications / wrong paths / wrong flags so codex doesn't silently waste a GPU week.

---

## TL;DR (Reviewer C verdict)

| Item | Verdict |
|---|---|
| Q1 F0 data sources real? | **REJECT** — 4 HIGH-severity path/schema problems. F0 cannot run as written. |
| Q2 F0 substrate consistency | **REJECT** — substrate-cross paired-t leak via `tp != 'NORMAL'` skip rule. |
| Q3 A4 yaml 4-field diff complete? | **REJECT** — wrong namespace. Actual knob is `training.image_aux.*`, not `transport.image_aux.*`. yaml mutation will crash. |
| Q4 A4 from-scratch vs resume | **APPROVE** — from-scratch is correct and matches V13/V14 design. |
| Q5 A4 stop-rule effective? | **MODIFY** — file-name anti-check is too narrow; doesn't catch `A4_image_aux_lambda_12.yaml`-style sibling. |
| Q6 partial-completion recovery | **MODIFY** — silent partial-completion protocol absent; should be added. |
| Q7 §5 self-check coverage gaps | **MODIFY** — missing image_aux subfield-stability check, missing `lambda_img` log-grep check, missing paired-t `ttest_rel` use-check. |
| Q8 §6 NOT-DO over/under-strict | **APPROVE with minor** — list is mostly tight; one item ("V18-cap.last named A3") is overscope for codex. |

**Push decision (one line)**: **MODIFY-BEFORE-PUSH**. Four HIGH-severity blockers (B75-B78) plus three MED items. Codex would silently execute the wrong A4 (no functional change) and crash on F0 data loading. ROI of fixing-before-push >> ROI of pushing as-is.

---

## 0. Mechanical verifications I ran first

Per prompt §7 ("the highest-value review is catching fabrication"), I grep-verified every CLI flag, file path, yaml field, and Python attribute the task md cites against the actual repo state.

### 0.1 Eval script CLI flags (✗ task md fabricates `--output-dir`)

`review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py` argparse:

```
--config (required)
--checkpoint (required)
--tag (required)
--split val
--batch-size 8
--device cuda:0
--max-slices 0
--decode-mode {default,raw,both}
--out-dir <default: /data_2/.../eval_0505_v6_v61_fullval>
```

Task md §F0.1 V7 eval block uses **`--output-dir`**. **The actual flag is `--out-dir`** (single hyphen, "out" not "output"). If codex copies the block verbatim:

```bash
python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
  --config ... \
  --checkpoint ... \
  --output-dir review/0521/v7_per_slice \    # ← FABRICATED. Will crash.
  --tag v7_best \
  ...
```

argparse will reject with `error: unrecognized arguments: --output-dir review/0521/v7_per_slice`. The same wrong flag appears in §A4.4 (twice, for A4.best and A4.last canonical eval). **3 occurrences total.**

This is **B76 (HIGH)**.

### 0.2 V7 yaml — image_aux namespace (✗ task md fabricates `transport.image_aux.*`)

Task md §A4.1 modifies:
- `transport.image_aux.lambda_start: 0.04 → 0.08`
- `transport.image_aux.lambda_max: 0.04 → 0.08`

Actual V7 yaml state:

```python
>>> y['transport'].keys()
['method', 'loss_type', 'velocity_loss_weight', 'endpoint_loss_weight',
 'endpoint_pair_weighting', 'endpoint_dt_normalize', 'pair_sample_probs',
 'pair_weighting', 'pair_loss_weights']
>>> y['transport'].get('image_aux')
None     # ← does not exist
```

Where the actual knob lives:

```python
>>> y['training']['image_aux']
{'enabled': True,
 'warmup_ratio': 0.0,
 'ramp_ratio': 0.0,
 'lambda_start': 0.04,
 'lambda_max': 0.04}
```

And training code:
```python
# train_first_hop.py:1588
image_aux_cfg = train_cfg.get("image_aux", {})     # train_cfg = cfg["training"]
# train_first_hop.py:2095-2100
lambda_img = get_linear_schedule_value(
    ...,
    start=float(image_aux_cfg.get("lambda_start", 0.0)),
    end=float(image_aux_cfg.get("lambda_max", 0.05)),
)
```

So the functional knob is **`training.image_aux.{lambda_start, lambda_max}`**, not `transport.image_aux.*`.

**What happens if codex executes §A4.1 verbatim**:

```python
d['transport']['image_aux']['lambda_start'] = 0.08
```

This raises `KeyError: 'image_aux'` because `d['transport']` has no `image_aux` subkey. Codex either:
- (a) **Best case** — sees the error, stops, asks user. (Best outcome.)
- (b) **Bad case** — "fixes" by creating `d['transport']['image_aux'] = {'lambda_start': 0.08, 'lambda_max': 0.08}` (new subkey). yaml diff verify script's ALLOWED set passes (4 fields, matches). Training code reads `training.image_aux.*` (unchanged from V7), so **A4 silently trains identical to V7-from-scratch (seed=42)**. 7 days of GPU produces an exact V7 reproduction; A4 vs V7 ≈ 0 by construction. **Wasted week.**
- (c) **Worse** — codex guesses `training.image_aux.lambda_max` and fixes mid-flight without telling user; yaml diff verify ALLOWED set then fails because the changed field is not in ALLOWED. Codex either disables the verify or hand-edits ALLOWED — either way the audit trail rots.

This is **B75 (HIGH)** — same severity class as `--output-dir`.

### 0.3 V18 fullval JSON paths (✗ task md fabricates path)

Task md §F0.0:
```
review/0517/V18_decoder_lora/fullval_eval_20260518/v18_decoder_lora_best_fullval_psnr_chain_mse.json
review/0517/V18_decoder_lora/fullval_eval_20260518/v18_decoder_lora_last_fullval_psnr_chain_mse.json
```

Actual layout (verified by `find review -name '*v18*fullval*'`):
```
review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse.json
review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv
review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse.json
review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv
```

**Two errors** in the path:
1. Missing `artifacts/` subdirectory.
2. Base name is `v18_best` (matches `--tag v18_best`), not `v18_decoder_lora_best`.

This is **B77 (HIGH)** (one of two parts).

### 0.4 V13/V14 per-slice DOES NOT EXIST anywhere on disk

Task md §F0.0 lists V13/V14 JSON paths:
```
review/0516/full_eval_json/v13_true_image_aux_off_best_fullval_psnr_chain_mse.json
review/0516/full_eval_json/v14_true_d_pure_best_fullval_psnr_chain_mse.json
```

These exist. But I loaded each and verified their schema:

```python
>>> json.load(open(...))['summary_psnr_clip3']['NORMAL'].keys()
dict_keys(['n', 'mean', 'std', 'min', 'max'])
```

**Only summary statistics**, no per-slice array. Independent search (`find review -name '*per_slice*'`) found:
- V6 / V7 / V8 / V6_NOISE per-slice at `review/0511/.../artifacts/planf_*_per_slice.csv`
- V18.best / V18.last per-slice at `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_{best,last}_*_per_slice.csv`
- ROI / per_hop CSVs in `review/0517/disambig/` (different metrics)
- V18-cap multi-ckpt direct-decode in `KL_DRIFT_PER_SLICE.csv`

**No V13 or V14 per-slice CSV exists anywhere.** V13/V14 were evaluated using the canonical script (which DOES emit per-slice CSV by default — see §0.5), but **the per-slice CSV outputs were not committed** to git. Without them, F0 cannot compute V13 vs V7 or V14 vs V7 paired-t.

This is **B77 (HIGH)** (second part).

**Cost to recover**: V13/V14 eval needs to be re-run with the canonical script. From the V13/V14 JSON `meta.runtime_sec ≈ 374-377 seconds` (~6 minutes per ckpt on GPU). Total: ~12 minutes of GPU + write CSV. Not free — costs 1 GPU slot for ~15 minutes — but recoverable.

### 0.5 Canonical eval script always emits per-slice CSV (no extra flag needed)

```python
# eval_first_hop_fullval_psnr_chain_mse.py:274-280
csv_path = out_dir / f"{args.tag}_fullval_psnr_chain_mse_per_slice.csv"
...
write_csv(csv_path, rows)
print(f"Saved CSV:  {csv_path}", flush=True)
```

Per-slice CSV is the **default output**. Task md §F0.1 suggests "如脚本不支持 --save-per-slice, 检查脚本是否默认输出 per-slice CSV" — but the comment about "≤ 20 行改动" patching the script is unnecessary. **Codex must NOT patch the eval script** (would violate "no modify canonical script" hygiene).

Per-slice CSV column names: `slice_idx, mse_D10, mse_D20, ..., psnr_D10, psnr_D20, psnr_D4, psnr_D50, psnr_NORMAL, psnr_raw_D10, ..., psnr_raw_NORMAL`. The `psnr_*` columns are produced via `calc_psnr_clip3` on **default-decode** chain output; `psnr_raw_*` is the **raw-decode** variant. **Task md never specifies which column F0 should read.** Strong default = `psnr_NORMAL` (matches canonical NORMAL PSNR_clip3 = 36.4943 etc.).

### 0.6 V7 per-slice already exists — no re-eval needed

Task md §F0.1 spends 25+ lines on a fallback "if V7 per-slice doesn't exist, re-run V7 canonical eval". The file is already on disk:

```
review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv
(7404 lines: 1 header + 7403 data; columns identical to V18.best per-slice CSV)
```

Same canonical script, same `calc_psnr_clip3` metric. Substrate-consistent with V18 per-slice. No re-eval needed for V7.

Wasted-cost if codex follows §F0.1 fallback: ~6 minutes of GPU + an unnecessary commit. Not catastrophic but indicates the task md substrate hunt was incomplete.

### 0.7 KL_DRIFT_PER_SLICE.csv verification (✓ matches task md claim)

```
$ wc -l review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_PER_SLICE.csv
148061 …KL_DRIFT_PER_SLICE.csv      # 1 header + 148060 data
$ head -2 .../KL_DRIFT_PER_SLICE.csv
ckpt,step,slice_idx,timepoint,psnr_clip3
V7.best,160000,0,D20,50.58320163176708
```

Task md claim "148060 行, 5 ckpt × 4 timepoint × 7403 slice": 5×4×7403 = 148,060 ✓ exact.

Substrate of KL_DRIFT_PER_SLICE.csv: each row is **direct decode of `z_GT`** (not chain rollout). PSNR values ~50 dB at D20 confirm this (chain rollout V7 D20 is ~35 dB; direct decode is ~50 dB). This is **the GT-manifold decode substrate**, not chain.

### 0.8 train_first_hop.py:2230 B42 (✓ verbatim)

```python
# train_first_hop.py lines 2228-2232
if lambda_kl > 0.0:
    z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
    loss_kl = compute_kl_pullback_loss(
        rae_lora=model.rae,
```

Line matches task md §5 PASS check 6. ✓ Verbatim preserved.

### 0.9 CLI flag set (✓ unchanged)

```
$ grep "add_argument" train_first_hop.py | grep -oE "['\"]--[a-z_-]+['\"]" | sort -u
'--config'
'--resume'
```

Matches task md §5 PASS check 5: `--config --resume`. ✓

---

## 1. Q1 — F0 data sources reality check

**Verdict: REJECT.**

Four substrate problems prevent F0 from running as written:

| # | problem | severity |
|---|---|---|
| F1 | V18 JSON path missing `artifacts/`, wrong base name (§0.3) | HIGH |
| F2 | V13/V14 per-slice CSV does NOT exist anywhere (§0.4) | HIGH |
| F3 | `--output-dir` is fabricated; actual flag is `--out-dir` (§0.1) | HIGH |
| F4 | V7 per-slice claimed "needs locating" but is already at `review/0511/.../planf_v7_best_*_per_slice.csv` (§0.6) | MED — wastes ~6 min GPU + extra commit |

### 1.1 Concrete required fixes for §F0.0 / §F0.1

Replace the `DATA` dict template with:

```python
DATA = {
    # V7 per-slice: ALREADY EXISTS, do NOT re-run.
    'V7.best':       'review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv',
    # V13/V14 per-slice: DOES NOT EXIST. Must re-run canonical eval (~6 min/ckpt).
    # Output target: review/0521/v13_v14_per_slice/v13_best_fullval_*_per_slice.csv (etc.)
    'V13.best':      'review/0521/v13_v14_per_slice/v13_best_fullval_psnr_chain_mse_per_slice.csv',
    'V14.best':      'review/0521/v13_v14_per_slice/v14_best_fullval_psnr_chain_mse_per_slice.csv',
    # V18 per-slice: EXISTS at artifacts/ subdir.
    'V18.best':      'review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse_per_slice.csv',
    'V18.last':      'review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_last_fullval_psnr_chain_mse_per_slice.csv',
    # V18-cap: separate substrate (direct decode, not chain). Hold for F0b only.
    'V18-cap.last':  'review/0517/V18_capacity_only/kl_drift_with_cap_20260519_180206/KL_DRIFT_PER_SLICE.csv',
}
```

### 1.2 Required §F0.1 rewrite

Replace the V7 fallback re-run with a V13/V14 re-run because V13/V14 per-slice doesn't exist. V7 per-slice is in place.

```bash
# V7 per-slice: confirm exists.
test -f review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv

# V13 / V14: per-slice CSV missing, re-run canonical eval to emit per-slice.
V13_CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V13_true_image_aux_ablation/run/first_hop_224_v13_true_image_aux_off/best.pt
V14_CKPT=/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0516_runs/V14_true_d_pure/run/first_hop_224_v14_v7_seed1337/best.pt

mkdir -p review/0521/v13_v14_per_slice

for pair in "V13:$V13_CKPT" "V14:$V14_CKPT"; do
    name="${pair%%:*}"; ckpt="${pair#*:}"
    cfg="$(dirname $ckpt)/config.yaml"
    tag=$(echo $name | tr '[:upper:]' '[:lower:]')_best
    python review/0505/operator/scripts/eval_first_hop_fullval_psnr_chain_mse.py \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --tag "$tag" \
        --split val \
        --max-slices 0 \
        --batch-size 8 \
        --decode-mode both \
        --out-dir review/0521/v13_v14_per_slice \
        2>&1 | tee review/0521/v13_v14_per_slice/${tag}_eval.log
done
```

Total GPU cost ≈ 12 minutes (6 min/ckpt × 2 ckpt) vs the task md's implicit 0 GPU. The "F0 完全不跑训练" claim in §0 holds (no *training*), but **F0 does require ~12 min of GPU eval** before paired-t can be computed.

### 1.3 Implementation of `load_per_slice` in F0 script

Task md leaves this as `NotImplementedError`. Codex must read CSV (not JSON) and select the right column:

```python
import pandas as pd
def load_per_slice(path: str, timepoint: str) -> np.ndarray:
    df = pd.read_csv(path)
    # CSV columns are 'psnr_D10', 'psnr_D20', 'psnr_D4', 'psnr_D50', 'psnr_NORMAL'
    # (default decode, calc_psnr_clip3) and 'psnr_raw_*' (raw decode variant).
    # Use default decode for canonical NORMAL = 36.4943-level numbers.
    col = f'psnr_{timepoint}'
    if col not in df.columns:
        raise KeyError(f"column {col} not in {path}")
    assert len(df) == 7403, f"{path}: n={len(df)} != 7403"
    return df[col].to_numpy()
```

KL_DRIFT_PER_SLICE.csv has different schema (`ckpt, step, slice_idx, timepoint, psnr_clip3`) and is **direct decode only** — needs a separate loader if F0 is extended to V18-cap (which it shouldn't be in F0 main; defer to F0b).

---

## 2. Q2 — F0 paired-t substrate consistency

**Verdict: REJECT — substrate-cross paired-t leak via skip condition.**

### 2.1 The skip bug

Task md §F0.2 template:

```python
for tp in TIMEPOINTS:
    ref_tp = load_per_slice(DATA['V7.best'], tp)
    out[tp] = {}
    for name, path in DATA.items():
        if name == 'V7.best':
            continue
        if 'V18-cap' in name and tp != 'NORMAL':
            continue  # cap CSV 仅 direct decode; 不与 chain PSNR 同 substrate
        arr = load_per_slice(path, tp)
        ...
```

The annotation says "**仅 direct decode; 不与 chain PSNR 同 substrate**". The intended semantics: **skip V18-cap entirely** (because it's direct-decode, not chain). The coded semantics: **skip V18-cap only at D20/D10/D4**, but **let V18-cap through at NORMAL**.

If codex copies this and the loader returns a value for V18-cap NORMAL (it can — KL_DRIFT_PER_SLICE.csv contains V18-cap.last rows for NORMAL timepoint), the paired-t computes:

```
arr  = V18-cap.last per-slice NORMAL PSNR_clip3 via direct decode(z_GT)        # ≈ 52.8 dB
ref  = V7.best     per-slice NORMAL PSNR_clip3 via chain rollout              # ≈ 36.8 dB
delta = arr - ref ≈ +16 dB                                                    # cross-substrate!
```

The resulting t-statistic and p-value are **mathematically computed but semantically meaningless** — they compare two different metrics. Anyone reading the F0 report will see "V18-cap.last vs V7.best paired-t = +∞" and either (a) catch the bug, or (b) cite the +16 dB as a V18-cap discovery. (b) is the silent-corruption failure mode.

**Severity**: HIGH. This is exactly the failure mode §6 NOT-DO #9 forbids ("F0 把 V18-cap (direct decode) 与 V7 chain PSNR 混算 paired-t"). The skip rule contradicts the NOT-DO.

**B78 (HIGH)**.

### 2.2 Fix

Change skip to all-timepoint:

```python
if 'V18-cap' in name:
    continue  # cross-substrate; deferred to separate F0b within direct-decode substrate
```

Or — better — add an explicit `substrate` field to the data dict and refuse cross-substrate pairs at runtime:

```python
DATA = {
    'V7.best':      {'path': '...', 'substrate': 'chain_rollout'},
    'V13.best':     {'path': '...', 'substrate': 'chain_rollout'},
    'V14.best':     {'path': '...', 'substrate': 'chain_rollout'},
    'V18.best':     {'path': '...', 'substrate': 'chain_rollout'},
    'V18.last':     {'path': '...', 'substrate': 'chain_rollout'},
    'V18-cap.last': {'path': '...', 'substrate': 'direct_decode'},
}
REF = 'V7.best'
REF_SUBSTRATE = DATA[REF]['substrate']
for name, info in DATA.items():
    if name == REF: continue
    if info['substrate'] != REF_SUBSTRATE:
        print(f"SKIP cross-substrate: {name} ({info['substrate']}) vs {REF} ({REF_SUBSTRATE})")
        continue
    ...
```

This makes the substrate guard structural, not comment-based.

---

## 3. Q3 — A4 yaml 4-field diff completeness

**Verdict: REJECT — wrong namespace; yaml mutation will crash; verify script's ALLOWED is wrong; if codex "fixes" silently, A4 trains as V7 reproduction (~7 day waste).**

### 3.1 The namespace bug

Task md §3.1 says 4 fields change:
- `output_dir`
- `run_name`
- `transport.image_aux.lambda_start: 0.04 → 0.08`
- `transport.image_aux.lambda_max: 0.04 → 0.08`

Actual V7 yaml state (verified in §0.2):
- `transport` block contains `[method, loss_type, velocity_loss_weight, ..., pair_loss_weights]`. **No `image_aux` subkey.**
- `training` block contains `[..., image_aux]`. `training.image_aux = {enabled: True, warmup_ratio: 0.0, ramp_ratio: 0.0, lambda_start: 0.04, lambda_max: 0.04}`.
- Training code reads `cfg['training']['image_aux']` (train_first_hop.py:1588, 2095-2100).

### 3.2 What happens if §A4.1 Python mutation runs as-is

```python
d = yaml.safe_load(p.read_text())
d['transport']['image_aux']['lambda_start'] = 0.08      # KeyError: 'image_aux'
d['transport']['image_aux']['lambda_max']   = 0.08
```

The first line raises `KeyError: 'image_aux'` because `d['transport']` has no `image_aux` key.

If codex catches the error and "fixes" by initializing the subkey:
```python
d.setdefault('transport', {}).setdefault('image_aux', {})['lambda_start'] = 0.08
```
then the yaml gains a NEW `transport.image_aux` block which the trainer **never reads**. yaml diff verify script's `ALLOWED = {'output_dir', 'run_name', 'transport.image_aux.lambda_start', 'transport.image_aux.lambda_max'}` would then pass (yes, exactly 4 fields differ). But `training.image_aux.{lambda_start, lambda_max}` remain at 0.04. **A4 silently trains as V7-from-scratch.** Result: A4 vs V7 ≈ 0 dB by construction; ~7 days of GPU produces a noisy V7 reproduction.

**Severity**: HIGH. This is the worst failure mode in the task md — codex would push commits, the audit trail looks clean, and the only signal something went wrong would be a paper-week-later realization that A4 result matches V7 within noise.

**B75 (HIGH)**.

### 3.3 Fix

Required §A4.1 changes:

```python
# Replace these two lines:
d['transport']['image_aux']['lambda_start'] = 0.08
d['transport']['image_aux']['lambda_max']   = 0.08
# With:
d['training']['image_aux']['lambda_start'] = 0.08
d['training']['image_aux']['lambda_max']   = 0.08
```

And replace `ALLOWED` in the verify script:
```python
ALLOWED = {
    'output_dir',
    'run_name',
    'training.image_aux.lambda_start',
    'training.image_aux.lambda_max',
}
```

### 3.4 Other A4 yaml hygiene

Independent confirmation that the **other** image_aux subfields are properly held constant:

V7 `loss.image_aux` = `{l1_weight: 1.0, ssim_weight: 0.25, seam_weight: 0.1, border_width: 14, border_weight: 2.0, seam_patch_size: 14}`. These control the *form* of the image_aux loss; A4 must inherit unchanged. The yaml diff verify script's `flatten(...)` + `ALLOWED = 4` set correctly catches any drift here. ✓ adequate after the namespace fix.

V7 `training.image_aux.{enabled, warmup_ratio, ramp_ratio}` = `{True, 0.0, 0.0}`. A4 must inherit unchanged; `lambda_max=0.08` with `warmup_ratio=0.0` and `ramp_ratio=0.0` means a constant 0.08 schedule (no warmup, no ramp). This is the intended "flat-doubled lambda" probe. Confirms A4 design intent.

A4 from-scratch (no `--resume`) at seed=42, max_steps=160000 — fair vs V7 (also from-scratch, seed=42, 160K).

---

## 4. Q4 — A4 from-scratch vs resume

**Verdict: APPROVE.**

§3.2 launch command has no `--resume`:
```bash
nohup python train_first_hop.py \
  --config review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml \
  > "$LAUNCH_LOG" 2>&1 &
```

V13 and V14 also from-scratch (160K each, verified from V13_V14_full_eval_analysis_20260521.md). A4 from-scratch is the right design for a clean V7-vs-A4 single-variable comparison.

Task md §0 line 3 explicitly says "from-scratch 训到 step 160000, ~7 天" so no ambiguity. ✓

If codex were tempted to add `--resume` for speed (e.g., resume from V7.best @ 160K and train 10K more), the result would be A4-as-fine-tune (analogous to V18) and the V7-vs-A4 comparison would be cross-substrate. Adding an explicit anti-check in §5 self-check would harden this:

```bash
check "A4 launch script has NO --resume" bash -c '! grep -q -- "--resume" review/0521/A4_image_aux_lambda_08/A4_train_*.log 2>/dev/null'
```

(Optional minor; the §0 explicit statement is already adequate guard.)

---

## 5. Q5 — A4 stop-rule effectiveness

**Verdict: MODIFY — anti-check too narrow.**

§5 anti-check #3 only blocks `*A4*v2*.yaml` filenames:
```bash
anti_check "no A4-v2 yaml" find review/0521 -name '*A4*v2*.yaml' 2>/dev/null | grep -q .
```

A codex eager to "explore one more lambda" can sidestep with `A4_image_aux_lambda_12.yaml` or `A4_warmup_probe.yaml`. The Round 17 stop rule intent is **no second image_aux training run**, not "no v2 in filename".

### 5.1 Stronger anti-check

```bash
anti_check "only ONE A4-* directory exists" bash -c '[ $(find review/0521 -type d -name "A4_*" | wc -l) -gt 1 ]'
```

This catches any sibling A4 directory regardless of naming. ✓

### 5.2 Verdict-table consistency

§A4.5 table explicitly says "无论 outcome, 不 relaunch A4-v2 / v3 (Round 17 stop rule, 防 scope creep B74)" ✓ adequate.

§F0.4 should also be reinforced: "F0 output is read-only artifact for paper; F0 does NOT trigger any further experiment design". Currently §F0.4 says "留给 user 整合" which is permissive enough but could be made explicit. Optional.

---

## 6. Q6 — Commit-order partial-completion recovery

**Verdict: MODIFY — partial-completion protocol absent.**

If A4 dies at step 100K, task md gives no instruction. Codex options:
- (a) Wait silently — user discovers days later from monitoring.
- (b) Commit metrics.jsonl with step=100000 + a status note — explicit but not currently described.
- (c) Skip commit 4-5 entirely — clean but loses partial result audit trail.

### 6.1 Add §4.4 partial-completion protocol

```markdown
### 4.4 Partial-completion recovery

If at any monitoring checkpoint (§A4.3) the process is no longer alive, OR if F0 is blocked (missing data, eval-script error), codex must:

1. Snapshot current metrics: `cp $A4_OUT/metrics.jsonl review/0521/A4_image_aux_lambda_08/A4_metrics_partial_<TS>.jsonl` (if A4) or write a `review/0521/F0_STATUS_BLOCKED.md` describing what's missing (if F0).
2. tail launch log (last 100 lines) + commit.
3. Write a 5-line status note: cause / last step reached / next-step recommendation.
4. Commit all of the above as one atomic commit: `git commit -m "Round 17 [A4|F0]: partial completion, see status note"`.
5. STOP. Do not silently re-launch. Wait for user.
```

### 6.2 Also add §0 fifth bullet

```markdown
6. Any failure (eval error, training crash, missing data) → commit current state with a status note and STOP. Do not silently re-launch or "work around".
```

---

## 7. Q7 — §5 self-check coverage gaps

**Verdict: MODIFY — missing 3 important checks.**

### 7.1 Add: A4 yaml image_aux subfield stability

After the namespace fix (§3.3), add:

```bash
check "A4 loss.image_aux unchanged from V7" python3 -c "
import yaml
v7 = yaml.safe_load(open('review/0505/local/configs/V7_gronwall_raw.yaml'))
a4 = yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml'))
assert v7['loss']['image_aux'] == a4['loss']['image_aux'], 'loss.image_aux drifted'
"
check "A4 training.image_aux.{enabled,warmup_ratio,ramp_ratio} unchanged" python3 -c "
import yaml
v7 = yaml.safe_load(open('review/0505/local/configs/V7_gronwall_raw.yaml'))
a4 = yaml.safe_load(open('review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml'))
for k in ['enabled', 'warmup_ratio', 'ramp_ratio']:
    assert v7['training']['image_aux'][k] == a4['training']['image_aux'][k], f'training.image_aux.{k} drifted'
"
```

### 7.2 Add: A4 training-log evidence that lambda_img is in effect

```bash
check "A4 log shows lambda_img≈0.08 after warmup" bash -c '
  # warmup_ratio=ramp_ratio=0 means lambda_img = 0.08 from step 1.
  grep -E "lambda_img=0\.0[78]" review/0521/A4_image_aux_lambda_08/A4_train_*.log 2>/dev/null | head -1 | grep -q "lambda_img"
'
```

(Allows 0.078-0.080 in case the linear schedule prints slightly below at step 1.)

### 7.3 Add: F0 script uses scipy.stats.ttest_rel (not ttest_ind)

```bash
check "F0 uses paired t (ttest_rel)" grep -q "ttest_rel" tools/paired_t_v18_vs_v7.py
check "F0 does NOT use unpaired ttest_ind" bash -c '! grep -q "ttest_ind" tools/paired_t_v18_vs_v7.py'
```

### 7.4 Add: F0 substrate guard

```bash
check "F0 V18-cap is excluded from chain paired-t" python3 -c "
import json
out = json.load(open('review/0521/F0_paired_t_summary.json'))
for tp in out:
    assert 'V18-cap.last' not in out[tp], f'V18-cap leaked into chain paired-t at {tp}'
"
```

This guards against the §2 substrate-cross bug.

### 7.5 Add: V7 per-slice n=7403 baseline check

```bash
check "V7 per-slice CSV has 7403 rows" bash -c '[ $(wc -l < review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv) -eq 7404 ]'  # 7403 + header
```

---

## 8. Q8 — NOT-DO §6 over/under-strict

**Verdict: APPROVE with one minor.**

| §6 item | reviewer C verdict |
|---|---|
| 改 train_first_hop.py | KEEP |
| 改 V7 yaml | KEEP |
| 改 V13/V14/V18 已有 ckpt 或 yaml | KEEP |
| 启 V19 / V18-clean / V18-rank-sweep | KEEP (R17 consensus) |
| 启 V14b / V14c | KEEP (F0 is cheaper substitute) |
| 启 A4-v2 / A4-v3 / 其它 image_aux variant | KEEP — but anti-check needs strengthening per §5 above |
| 启 architecture / data pivot | KEEP |
| F0 跳过 V14/V13 sanity check | KEEP — sanity check is critical |
| F0 把 V18-cap 与 chain PSNR 混算 | KEEP — but code substrate guard needs strengthening per §2 above |
| paper 草稿宣称 V18 +0.06 显著 在 F0 出来前 | KEEP — Round 17 B61-class guard |
| 把 V18-cap.last 命名为 "A3" 在新 doc 里 | **REDUCE** — this is a "doc hygiene" rule for user, not codex. Codex doesn't write new docs in this task. Move to a separate doc-style guideline. |
| push narrative / claim 类 markdown 前需 user 复审 | KEEP — but boundary ("数据" vs "解读") could be clearer: codex CAN commit JSON / CSV / log / verdict tables; codex CANNOT commit paper draft fragments or narrative paragraphs. Adding this distinction would close ambiguity. |

Overall: §6 is mostly tight. The "A3 naming" rule is for human authors, not codex; safe to drop from codex-facing task md.

---

## 9. Push decision and required fixes

### 9.1 Decision: **MODIFY-BEFORE-PUSH**

| § | original wording | required fix | severity |
|---|---|---|---|
| §F0.1 (V7 eval cmd line) | `--output-dir review/0521/v7_per_slice` | `--out-dir review/0521/v7_per_slice` (3 occurrences total across §F0.1 and §A4.4) | **HIGH** |
| §F0.0 V18 paths | `review/0517/V18_decoder_lora/fullval_eval_20260518/v18_decoder_lora_{best,last}_*.json` | `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_{best,last}_*_per_slice.csv` (use CSV not JSON; add `artifacts/` subdir; base name `v18_best` not `v18_decoder_lora_best`) | **HIGH** |
| §F0.0 V13/V14 paths + §F0.1 V7 fallback | "(待定位) V7 ... 若仅有 JSON summary 而无 per-slice, 需要先跑 V7 canonical eval" | V7 per-slice EXISTS at `review/0511/...`. V13/V14 per-slice does NOT exist — re-run canonical eval to emit per-slice (~12 min total GPU). Provide explicit V13/V14 re-run script (per §1.2 above). | **HIGH** |
| §F0.2 substrate skip | `if 'V18-cap' in name and tp != 'NORMAL': continue` | `if 'V18-cap' in name: continue` (skip all timepoints) — or add explicit `substrate` field per §2.2 | **HIGH** |
| §A4.1 yaml mutation | `d['transport']['image_aux']['lambda_start'] = 0.08` and `lambda_max` | `d['training']['image_aux']['lambda_start'] = 0.08` and `lambda_max` (training, not transport) | **HIGH** |
| §A4.1 verify ALLOWED | `{'output_dir', 'run_name', 'transport.image_aux.lambda_start', 'transport.image_aux.lambda_max'}` | `{'output_dir', 'run_name', 'training.image_aux.lambda_start', 'training.image_aux.lambda_max'}` | **HIGH** |
| §A4.2 GPU placeholder | `FREE_GPU=<根据 nvidia-smi 填>` literal | Add a fail-loud guard: `FREE_GPU=${1:?usage: pass GPU index}` or `if [[ -z "$FREE_GPU" || "$FREE_GPU" == *"根据"* ]]; then echo FAIL; exit 1; fi` | MED |
| §F0.2 `load_per_slice` | `raise NotImplementedError("Codex implement based on actual file schema")` | Provide concrete CSV loader template using `psnr_<timepoint>` column (per §1.3 above) | MED |
| §4.x | (missing partial-completion protocol) | Add §4.4 per §6.1 above | MED |
| §5 | 7 PASS + 4 anti-check | Add the 5 additional checks in §7 above (subfield stability, lambda_img log evidence, ttest_rel, F0 substrate guard, V7 n=7403) | MED |
| §5 anti-check #3 | `find review/0521 -name '*A4*v2*.yaml'` | `[ $(find review/0521 -type d -name "A4_*" \| wc -l) -gt 1 ]` (catches any sibling A4 dir) | LOW |
| §6 NOT-DO #11 | "把 V18-cap.last 命名为 'A3' 在新 doc 里" | Remove or move to a separate doc-hygiene rule; codex doesn't write new docs in this task | LOW |
| §0 / §9 | (no failure-stop policy) | Add bullet: "Any failure → commit current state with status note and STOP. No silent re-launch." | LOW |

### 9.2 If only the 5 HIGH items are fixed, task md becomes pushable

Acceptable minimal fix set: B75 + B76 + B77 + B78 (= rows 1-6 of the table above). MED/LOW items can land in a v2 task md or be applied inline by user.

### 9.3 If pushed as-is

Most likely codex execution path:
1. F0 step 1 — eval script crash on `--output-dir`. Codex either fixes or stops.
2. F0 step 2 — even if flag fixed, V18 JSON paths 404. Codex re-searches or stops.
3. F0 step 3 — V13/V14 per-slice not found. Codex re-runs eval (which would work) or stops.
4. F0 step 4 — paired-t runs, but V18-cap NORMAL leaks into chain comparison, producing a +16 dB anomaly. Either codex catches and fixes, or it commits a corrupted report.
5. A4 step 1 — yaml mutation crashes on `KeyError: 'image_aux'`. Codex either fixes (and might guess `training.image_aux` correctly) or initializes `transport.image_aux` silently (the worst path; 7 days wasted).

Expected wasted GPU if codex auto-recovers conservatively (e.g., creates the wrong namespace): **7 days** for A4 alone. Expected wasted GPU if codex stops at every error and asks user: **0 days but 5-10 round-trips of user-codex dialog**.

ROI of 30 minutes of `MODIFY-BEFORE-PUSH` >> either outcome.

---

## 10. New biases (B75-B82)

Per integration md, B61-B74 are accounted for. Next available is B75.

| ID | bias | severity | description |
|---|---|---|---|
| **B75** | A4 yaml namespace fabrication | **HIGH** | Task md §A4.1 modifies `transport.image_aux.*`; actual functional knob is `training.image_aux.*` (verified at train_first_hop.py:1588, 2095-2100, and in V7 yaml). yaml mutation will crash, or worse, silently train V7-as-A4. |
| **B76** | Eval-script CLI flag fabrication | **HIGH** | Task md uses `--output-dir` (3 occurrences in §F0.1 + §A4.4); actual flag is `--out-dir` (verified at eval_first_hop_fullval_psnr_chain_mse.py:55). argparse will reject immediately. |
| **B77** | F0 data path fabrication / missing per-slice | **HIGH** | (a) V18 JSON path missing `artifacts/` subdir and uses wrong base name (`v18_decoder_lora_best` instead of `v18_best`). (b) V13/V14 per-slice CSV does NOT exist anywhere on disk — only summary JSON exists. F0 requires ~12 min GPU eval to recover; task md claims F0 is 0-GPU. |
| **B78** | F0 substrate-cross leak via half-skip | **HIGH** | §F0.2 skip rule `if 'V18-cap' in name and tp != 'NORMAL': continue` lets V18-cap NORMAL through, but V18-cap is direct-decode while ref V7 is chain rollout. The resulting paired-t mixes substrates — exactly what §6 NOT-DO #9 forbids. |
| **B79** | V7 per-slice "needs locating" overclaim | MED | §F0.1 spends 25+ lines on a fallback to re-run V7 canonical eval. V7 per-slice already exists at `review/0511/.../planf_v7_best_*_per_slice.csv` (7404 lines). Recommended fallback is unnecessary work. |
| **B80** | A4 launch literal placeholder | MED | `FREE_GPU=<根据 nvidia-smi 填>` is a literal string; if codex copies without filling, `CUDA_VISIBLE_DEVICES=<根据 nvidia-smi 填>` is malformed and (depending on shell) may default to GPU 0 silently. Add fail-loud guard. |
| **B81** | Per-slice CSV column ambiguity | LOW | Eval script emits `psnr_NORMAL` (default decode, calc_psnr_clip3) and `psnr_raw_NORMAL` (raw decode); task md never specifies which column F0 should read. Strong default = `psnr_NORMAL` (matches canonical 36.4943-level numbers), but absence of explicit statement is a silent-pick risk. |
| **B82** | Partial-completion protocol absent | MED | Prompt §6 explicitly asks for handling of mid-flight failure; task md does not address. Codex could (a) silently retry, (b) commit half-done state without note, (c) wait indefinitely. Add §4.4 explicit protocol. |

### 10.1 What I did NOT find

- No fabrication in §1.1 R17 substrate numbers — all match my Round 17 grep results (V13=36.4943, V14=36.7806, V7=36.7810, V18.best=36.8112, V18.last=36.8426). ✓
- No fabrication in §1.2 PLANF paired-t = 74.6 reference — accepted on integration md's authority (4/4 reviewers signed). Not independently verified in this round (out of scope per §0 boundary).
- No fabrication in V13/V14/V18 ckpt paths (cannot verify remote `/data_2/...` paths from local; consistent with V13/V14 JSON `meta.checkpoint` field which uses identical structure).
- No fabrication in train_first_hop.py:2230 B42 (verbatim ✓).
- No fabrication in CLI flag set (`--config --resume` ✓).
- KL_DRIFT_PER_SLICE.csv row count and substrate description ✓.
- §1.4 stop-rule values (≥+0.05 SUCCESS, ≥0 MARGINAL, <0 REGRESSION) — internally consistent.
- §A4.5 V7.best row D20=35.4253 / D10=35.8105 / D4=36.3680 / NORMAL=36.7810: NORMAL matches (36.7810 ✓); D20/D10/D4 I cannot independently verify in this round (would require V7 JSON read) but no contradicting evidence.

---

## 11. Output per prompt §4

### 11.1 Per-question

| Q | verdict | reason |
|---|---|---|
| Q1 F0 data sources | **REJECT** | B76, B77, B79 (HIGH×2 + MED). Path/flag/data-existence fabrications. |
| Q2 F0 substrate consistency | **REJECT** | B78 (HIGH). Substrate-cross paired-t leak via half-skip. |
| Q3 A4 yaml diff completeness | **REJECT** | B75 (HIGH). Wrong namespace; mutation crashes or silently produces V7-clone. |
| Q4 A4 from-scratch vs resume | **APPROVE** | Explicit in §0 line 3. Matches V13/V14 design. |
| Q5 A4 stop-rule effectiveness | **MODIFY** | Anti-check too narrow; strengthen to "≥1 sibling A4_* dir" check. |
| Q6 commit-order recovery | **MODIFY** | Add §4.4 partial-completion protocol. |
| Q7 §5 self-check gaps | **MODIFY** | Add 5 checks: subfield stability, lambda_img log evidence, ttest_rel, F0 substrate guard, V7 n=7403. |
| Q8 §6 NOT-DO over/under | **APPROVE with minor** | List is tight; drop "A3 naming" rule (doc hygiene, not codex). |

### 11.2 Push decision

**MODIFY-BEFORE-PUSH.** 5 HIGH-severity blockers (4 unique fixes: B75/B76/B77/B78) must be corrected before push. Approximate effort: 30 minutes to edit task md + 1 verification pass.

### 11.3 Required fixes (HIGH only — minimum push-ready set)

| § | original | fix | severity |
|---|---|---|---|
| §F0.0 V18 paths | `…/fullval_eval_20260518/v18_decoder_lora_{best,last}_*.json` | `…/fullval_eval_20260518/artifacts/v18_{best,last}_*_per_slice.csv` | HIGH (B77) |
| §F0.0 V13/V14 paths | `…/full_eval_json/v1{3,4}_*_best_…json` (summary-only) | `review/0521/v13_v14_per_slice/v1{3,4}_best_…_per_slice.csv` (requires re-run, see §1.2 above) | HIGH (B77) |
| §F0.0 V7 path | "(待定位)" + 25-line fallback | `review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_…_per_slice.csv` | HIGH/MED (B79 — saves time) |
| §F0.1 + §A4.4 eval cmd | `--output-dir` (3×) | `--out-dir` | HIGH (B76) |
| §F0.2 substrate skip | `if 'V18-cap' in name and tp != 'NORMAL': continue` | `if 'V18-cap' in name: continue` (all timepoints) | HIGH (B78) |
| §A4.1 yaml mutation | `d['transport']['image_aux']` (×2) | `d['training']['image_aux']` (×2) | HIGH (B75) |
| §A4.1 verify ALLOWED | `transport.image_aux.{lambda_start,lambda_max}` | `training.image_aux.{lambda_start,lambda_max}` | HIGH (B75) |

### 11.4 New biases (B75-B82)

Detailed §10. Headline: **B75-B78 are HIGH and gate push**; B79-B82 are MED/LOW polish.

---

## 12. One-paragraph executive summary

The Codex task md has 4 HIGH-severity substrate fabrications that prevent it from pushing as-is: (1) **B75** — A4 yaml modifies `transport.image_aux.*` but the functional knob is `training.image_aux.*`, so codex either crashes on `KeyError` or silently trains a 7-day V7 reproduction; (2) **B76** — eval script uses fabricated `--output-dir` flag (actual: `--out-dir`), so V7 and A4 canonical eval will not launch; (3) **B77** — V18 JSON path is missing the `artifacts/` subdir and uses wrong base name, AND V13/V14 per-slice CSVs do not exist anywhere on disk (only summary JSONs do), so F0 paired-t cannot run; (4) **B78** — F0 skip rule `'V18-cap' in name and tp != 'NORMAL'` lets V18-cap NORMAL leak into chain paired-t, producing a +16 dB cross-substrate phantom result that §6 NOT-DO #9 explicitly forbids. **Recommendation: MODIFY-BEFORE-PUSH** with the 4 HIGH fixes plus 3 MED additions (partial-completion protocol, expanded self-check coverage, fail-loud GPU placeholder guard). All HIGH fixes are mechanical text edits; total estimated correction time ≈ 30 minutes. The strategic direction (Hybrid B + F0 + A4-light) and the integration md's 4/4 consensus are sound — this is a substrate execution problem, not a strategic one.

---

## 13. What I did NOT review

- Round 17 strategic decision (4/4 consensus already signed in REVIEW_INTEGRATION_round17_20260522.md).
- F0/A4 *whether-to-do* (R17 already signed).
- V13/V14 eval protocol / yaml (R17 already signed).
- KL pullback design (R16 already signed).
- The specific claim that PLANF V7 vs V8 paired-t = 74.6 (cited in §1.2; accepted on integration md authority).
- V7 best.pt actual ckpt content (remote-only path).
