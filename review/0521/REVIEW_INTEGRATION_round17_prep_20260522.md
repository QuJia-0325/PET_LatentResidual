# REVIEW INTEGRATION — Round 17-Prep (Codex Task md)

- date: 2026-05-22
- scope: integrate 4 independent execution-prep reviews on `CODEX_TASK_ROUND17_F0_A4_20260522.md`
- sources:
  - PEER_REVIEW_ROUND17_PREP_REVIEWER_A_20260522.md (Reviewer A)
  - PEER_REVIEW_round17_prep_reviewer_C_20260522.md (Reviewer C)
  - PEER_REVIEW_round17_prep_review_D_20260522.md (Review D)
  - PEER_REVIEW_round17_prep_codex_task_copilot_20260522.md (GitHub Copilot)

---

## 1) Unanimous verdict

**All 4 reviewers select MODIFY-BEFORE-PUSH.** Strategic direction (Round 17 = Hybrid B + F0 + A4-light) is unaffected. The blockers are purely execution-level: codex 100% will fail on first command if task md is pushed as-is.

---

## 2) Consensus HIGH blockers (4/4 found same set)

Verified on local repo at commit `a61d7e9`. Each blocker is independently grep-verifiable.

### H1 — A4 yaml namespace fabricated (HIGH)

| where task md says | actual V7 yaml position | symptom |
|---|---|---|
| `transport.image_aux.lambda_start` | `training.image_aux.lambda_start` | python yaml mutation → `KeyError: 'image_aux'` |
| `transport.image_aux.lambda_max` | `training.image_aux.lambda_max` | same |

`transport.image_aux` **does not exist** in V7 yaml. `loss.image_aux` exists but only contains `l1_weight / ssim_weight / seam_weight / border_*` (these must NOT be changed). The functional schedule knob is at `training.image_aux.{enabled, warmup_ratio, ramp_ratio, lambda_start, lambda_max}`.

Affected sections in task md: §3.1 (yaml mutation script), §3.1 (verify ALLOWED set), §5 self-check (yaml lambda_max grep).

### H2 — Eval CLI flag fabricated (HIGH)

| where task md says | actual script flag | symptom |
|---|---|---|
| `--output-dir` (3 occurrences) | `--out-dir` | argparse `unrecognized argument: --output-dir` |

`eval_first_hop_fullval_psnr_chain_mse.py` only declares: `--config --checkpoint --tag --split --batch-size --device --max-slices --decode-mode --out-dir`.

Affected sections: §F0.1 (V7 per-slice re-run command), §A4.4 (A4 best/last eval loop).

### H3 — V18 JSON path wrong (HIGH)

| where task md says | actual path | symptom |
|---|---|---|
| `review/0517/V18_decoder_lora/fullval_eval_20260518/v18_decoder_lora_best_fullval_psnr_chain_mse.json` | `review/0517/V18_decoder_lora/fullval_eval_20260518/artifacts/v18_best_fullval_psnr_chain_mse.json` | `FileNotFoundError` |

Two errors: missing `artifacts/` subdir AND wrong file basename (`v18_decoder_lora_best_*` should be `v18_best_*`). Same problem for `v18_last_*`.

Affected sections: §F0.0 table, §F0.2 DATA dict.

### H4 — V13/V14 per-slice CSV does NOT exist anywhere (HIGH)

V13/V14 JSON files at `review/0516/full_eval_json/v13_true_image_aux_off_*.json` and `v14_true_d_pure_*.json` contain **summary stats only** (`{n, mean, std, min, max}`), no per-slice arrays.

`find review -name '*v13*per_slice*' -o -name '*v14*per_slice*'` returns nothing.

Affected sections: §F0.0 (lists V13/V14 JSON as per-slice data sources), §F0.2 (DATA dict assumes per-slice extractable from these JSON).

Resolution required: add explicit §F0.0a step to **re-run canonical eval for V13.best and V14.best** with the standard script (which natively writes `*_per_slice.csv` under `artifacts/`). Cost: ~6 min/ckpt on free GPU.

### H5 — F0 substrate-cross guard is inverted (HIGH)

Task md §2.2 guard:
```python
if 'V18-cap' in name and tp != 'NORMAL':
    continue  # cap CSV 仅 direct decode; 不与 chain PSNR 同 substrate
```

This **only skips** D20/D10/D4. V18-cap NORMAL **still enters the paired-t loop**, even though V18-cap NORMAL is direct `decode(z_GT)` (52.8 dB) and V7 NORMAL is chain rollout (36.8 dB). The paired delta will be ~+16 dB cross-substrate phantom signal.

§6 NOT-DO #9 forbids this exact mistake. The guard logic is wrong.

Resolution required: change to `if 'V18-cap' in name: continue` (drop V18-cap from chain paired-t entirely), OR add explicit `substrate` field per ckpt and refuse cross-substrate paired-t.

---

## 3) Consensus MED issues (≥ 2/4 found)

### M1 — V7 per-slice already exists; §F0.1 re-run instruction is wasteful

`review/0511/fullval_psnr_clip3_20260516_173941/artifacts/planf_v7_best_fullval_psnr_chain_mse_per_slice.csv` (7404 rows, 21 columns, full schema match) exists in the repo.

Task md §F0.1 tells codex to re-run V7 canonical eval (~6-12 min GPU). Should instead point at the existing CSV. Found by Reviewer C + Copilot.

### M2 — Per-slice CSV column ambiguity (MED)

CSV has two columns per timepoint: `psnr_NORMAL` (clip3) and `psnr_raw_NORMAL` (raw). Task md does not specify which to use. F0 should use `psnr_NORMAL` (clip3 is canonical metric per Round 16 standing rule).

### M3 — `FREE_GPU=<...>` placeholder is a literal string (MED)

§3.2 has `FREE_GPU=<根据 nvidia-smi 填>`. If codex shell-evaluates as-is, `CUDA_VISIBLE_DEVICES=<根据 nvidia-smi 填>` is set literally. Need a fail-loud guard.

### M4 — Anti-check pattern `*A4*v2*` too narrow (MED)

§5 anti-check only catches `A4_image_aux_lambda_08_v2.yaml`-style names. Codex could create `A4_image_aux_lambda_12.yaml` (a second variant with different lambda value) and bypass the anti-check. Should broaden to `find review/0521 -name 'A4_image_aux_lambda_*.yaml' | wc -l == 1`.

### M5 — Partial completion protocol absent (MED)

§4.2 commit order assumes all 5 commits complete in sequence. If A4 dies at step 100K, no instruction on what to commit. Add: "any failure → commit current state + `<task>_STATUS_BLOCKED.md` explaining where, then stop".

### M6 — A4 baseline V7 D-stage numbers off by ~0.01 dB

Reviewer A flagged §A4.5 verdict table uses V7 baseline `D20=35.4253` but PLANF canonical is `35.4354`. Not strategically meaningful but a verify-the-baseline standing rule violation.

---

## 4) Non-issues / cross-checked false alarms

- Reviewer A's H3 (eval CLI flag) and Reviewer C's H3 are the same finding — single HIGH, not double.
- Reviewer D and Copilot independently flagged the substrate guard inversion (H5) — confirms.
- `tools/paired_t_v18_vs_v7.py` script template using `scipy.stats.ttest_rel` is correct (no reviewer disputed).
- B42 line verbatim verify in §5 is correct (no reviewer disputed).
- `--config --resume` CLI flag list for trainer is correct.

---

## 5) New biases B75-B82 (consolidated)

| ID | bias | severity | source |
|---|---|---|---|
| B75 | Yaml namespace fabrication (`transport.image_aux` doesn't exist) | HIGH | C, D, Copilot |
| B76 | Eval CLI flag fabrication (`--output-dir`) | HIGH | A, C, D, Copilot |
| B77 | Per-slice substrate fabrication (assumed V13/V14 JSON had per-slice arrays) | HIGH | A, C, D, Copilot |
| B78 | Substrate-guard inverted logic (V18-cap NORMAL still leaks to chain paired-t) | HIGH | A, C, D, Copilot |
| B79 | Existing V7 per-slice CSV ignored (re-run instruction wasteful) | MED | C, Copilot |
| B80 | `<...>` placeholder treated as variable | MED | C, Copilot |
| B81 | CSV column ambiguity (psnr vs psnr_raw) | MED | C |
| B82 | Partial-completion protocol missing | MED | D, Copilot |

Root cause pattern (4/4 agree): **verify-before-write violation at execution-doc level**. Task md author (claude) declared specific file paths, CLI flags, and yaml namespaces without grep-verify against the actual repo. Round 17 strategic content was correct; execution-spec content was not grep-verified.

This is the same `B45-class` pattern (substrate-stage claude self-correction unreliable, requires reviewer gate) — now manifest at the codex-execution layer rather than the substrate-interpretation layer.

---

## 6) Integrated fix plan

### 6.1 HIGH fixes (mandatory before push)

1. **H1**: `transport.image_aux.*` → `training.image_aux.*` everywhere (§3.1 mutation, §3.1 verify, §5 self-check grep).
2. **H2**: `--output-dir` → `--out-dir` (§F0.1, §A4.4, all 3 occurrences).
3. **H3**: V18 JSON paths → add `artifacts/` + `v18_decoder_lora_*` → `v18_*` (§F0.0 table, §F0.2 DATA dict).
4. **H4**: Insert §F0.0a — re-run V13.best and V14.best canonical eval to generate per-slice CSV (with correct `--out-dir` flag and `artifacts/` output). Cost ~12-15 min on free GPU.
5. **H5**: F0 substrate guard → drop V18-cap entirely from chain paired-t; comment why; explicitly route V18-cap NORMAL to a separate Stage F0b (capacity vs V18@170K direct-decode paired-t) so it's still useful.

### 6.2 MED fixes (recommended same edit pass)

6. **M1**: §F0.1 — first try existing CSV at `review/0511/.../planf_v7_best_*_per_slice.csv`; only re-run if missing.
7. **M2**: F0 script — use `psnr_NORMAL` column (not `psnr_raw_NORMAL`); document choice.
8. **M3**: §3.2 launch — replace `FREE_GPU=<...>` literal with explicit shell guard: `FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | nl -v0 | sort -k2 -rn | head -1 | awk '{print $1}'); [[ -z "$FREE_GPU" ]] && { echo FAIL no free gpu; exit 1; }`
9. **M4**: §5 anti-check — `[[ $(ls review/0521/A4_image_aux_lambda_*.yaml 2>/dev/null | wc -l) -le 1 ]]`
10. **M5**: §4 — add partial completion protocol section.

### 6.3 LOW (skip unless trivial)

11. **M6**: §A4.5 — refresh V7 baseline D-stage numbers from PLANF.

---

## 7) Final integrated verdict

**Round 17-Prep verdict = MODIFY-BEFORE-PUSH** with 5 HIGH + 5 MED fixes listed above. Total fix effort ≈ 30-45 minutes of mechanical edits on the task md (no new design). After fixes, the task md becomes safely pushable to gitee for codex execution.

**Strategic conclusions are unchanged.** Round 17 Hybrid B + F0 + A4-light stands. Only the codex execution spec needs repair.

---

## 8) Action

Claude will now:
1. Apply all 5 HIGH + 5 MED fixes to `CODEX_TASK_ROUND17_F0_A4_20260522.md` in a single edit pass.
2. Re-run task md sanity grep on each fix.
3. Commit as "Round 17 codex task: apply Round 17-Prep review fixes (5 HIGH + 5 MED, B75-B82)".
4. Push to gitee.

After this push, codex can pull and execute. No further peer review round expected before codex pulls.
