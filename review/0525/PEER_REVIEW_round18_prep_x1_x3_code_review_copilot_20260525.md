# Peer Review Round 18-Prep — X1-lite + X3 Deep Code Review

- date: 2026-05-25
- reviewer: GitHub Copilot
- reviewed task: `CODEX_TASK_ROUND18_X1_X3_20260525.md`
- scope: code-level execution feasibility and design-quality review before Codex execution

## 0. Main Verdict

**MODIFY-BEFORE-PUSH.**

The experimental design is directionally sound: X1-lite is a valid first mechanism falsification of A4-mid, and X3 is a useful A3-matched additive test for decoder LoRA under stronger image_aux. But the current Codex task is not push-safe. The YAML diffs are mostly correct; the blockers are in shell execution, resume semantics, self-checks, and one hidden KL side effect.

## 1. High-Impact Findings

### H1 — X3 launch omits `--resume`

**Severity: HIGH. Location: §3 B.2.**

The task says X3 warm-starts from V7.best at step 160000, but the launch command is:

```bash
nohup "$PYTHON" train_first_hop.py \
  --config review/0525/X3_image_aux_lora/X3_image_aux_lora.yaml \
  > "$X3_LOG" 2>&1 &
```

`train_first_hop.py` only uses the CLI `--resume`; `training.resume_from` in YAML is not consumed by the launch path. Without `--resume`, X3 trains from scratch to 170K and the A3/V18 matched-step comparison collapses.

**Fix:** pass V7.best explicitly via `--resume`, and make the health check require `[startup] resume from:` plus `[resume] loaded step=160000` in the log.

### H2 — X3 health check cannot recover X1 PID

**Severity: HIGH. Location: §3 B.2.**

X1 launch prints `PID=$X1_PID` to the controlling shell, not into `$X1_LOG`. Later B.2 tries to grep `PID=...` from `$X1_LOG`, so `X1_PID` will be empty in a fresh Codex step.

**Fix:** write launch sidecars:

```bash
echo "$X1_PID" > review/0525/X1_lite_l1_only/X1_lite.pid
echo "$FREE_GPU" > review/0525/X1_lite_l1_only/X1_lite.gpu
```

Read those files before launching X3. Do the same for X3.

### H3 — X3 GPU selection does not exclude X1 GPU

**Severity: HIGH. Location: §3 B.2.**

Selecting `FREE_GPU2` by sorting global free memory can pick X1's GPU during transient loading or memory-reporting gaps. This is avoidable.

**Fix:** record `X1_lite.gpu`, filter it out when choosing X3's GPU, and fail if no distinct GPU has enough free memory.

### H4 — `decoder_kl_pullback.enabled=true` still builds frozen RAE

**Severity: MED-HIGH. Location: §3 B.0 / `pet_lr/model_first_hop.py`.**

`lambda_kl=0` skips `compute_kl_pullback_loss` in the training loop, but model construction still checks `decoder_kl_pullback.enabled`. If true, `PETFlowDiTFirstHop` builds a second frozen RAE reference and stores it as `rae_frozen`. A3 survived this, but X3 is a 2-slot parallel run and should not carry unused memory/startup cost.

**Fix:** add a seventh X3 diff:

```yaml
loss.decoder_kl_pullback.enabled: false
```

Keep `lambda_kl=0.0` as a redundant guard. Update allowed-diff and self-check lists.

### H5 — X1-lite `img_ssim` check is semantically wrong

**Severity: MED. Location: §2 A.1/A.2.**

`compute_first_hop_image_loss` always computes raw L1, SSIM, and seam losses, then combines them as `w_l1*l1 + w_ssim*ssim + w_seam*seam`. With `ssim_weight=0` and `seam_weight=0`, raw `img_ssim` and `img_seam` still log nonzero. The weighted `img` should approximately equal `img_l1`, not zero SSIM.

**Fix:** remove “`img_ssim` 极小” from the health check. Verify YAML weights and optionally assert `img ~= img_l1` from a train line.

### H6 — self-check anti-checks are broken

**Severity: HIGH. Location: §6.**

Patterns such as:

```bash
anti_check "no X2 yaml" find review/0525 -name 'X2_*.yaml' 2>/dev/null | grep -q .
```

are unsafe. `find` returns success even when no file matches, and `ERR` updates inside a pipeline happen in a subshell and can be lost.

**Fix:** replace with explicit forbidden-file helper using `find ... -print -quit`, not pipelines.

## 2. Candidate Issues CI-1 to CI-5

| ID | verdict | severity | assessment |
|---|---|---|---|
| CI-1 GPU race | **EXTEND** | HIGH | Race is plausible, but PID recovery and GPU exclusion are the concrete blockers. |
| CI-2 LR schedule | **VERIFY** | LOW | `total_steps_override=200000`, `decoder_lr_mult=0.00625`, and `decoder_weight_decay=0.0` are correct for A3/V18 matching. The prompt's LR arithmetic is off, but A3 uses the same schedule. |
| CI-3 early monitor | **EXTEND** | MED | Alive/step-only monitoring is weak, but one 30K rolling-val threshold is too brittle. Use full-val/`val_select_score` diagnostics and no auto-kill on one point. |
| CI-4 w=0 SSIM/seam | **VERIFY + EXTEND** | MED | Zero weights remove gradient contribution, but raw losses still compute and log nonzero. Fix the log expectation. |
| CI-5 KL enabled true | **EXTEND** | MED-HIGH | Loss is skipped at `lambda_kl=0`, but frozen RAE construction still happens if `enabled=true`. Set `enabled=false`. |

## 3. Deep Questions Q1-Q9

### Q1 — X1-lite 6-field diff

**APPROVE with monitoring fix.** The six intended fields are enough: output isolation, run name, `training.image_aux.lambda_start/max=0.08`, and `loss.image_aux.ssim_weight/seam_weight=0.0`. Do not change `l1_weight`, `border_width`, `border_weight`, `seam_patch_size`, warmup/ramp, rollout, LR, backbone, or transport settings.

### Q2 — X3 6-field diff

**MODIFY.** Add `loss.decoder_kl_pullback.enabled=false`. Also add `training.save_interval=5000` if the task expects A3-style `step_165000.pt` and `step_170000.pt`; A3 used `save_interval=5000`, while V18's base YAML has `save_interval=10000`.

### Q3 — X1-lite vs A4-mid matching

**APPROVE.** A4-mid is V7 + λ=0.08; X1-lite is V7 + λ=0.08 + SSIM/seam weights zero. This cleanly isolates full image_aux vs L1-only image_aux at the same strength, assuming the strict YAML diff check remains.

### Q4 — X3 vs A3 matching

**REJECT until H1 is fixed.** Conceptually, X3 should be A3 + λ_img=0.08. As written, missing `--resume` means it is not matched to A3 at all. After adding `--resume`, the comparison is valid for the A3-matched 10K LoRA adaptation window.

### Q5 — Loss balance watchdog

**APPROVE.** `loss_balance_watch_enforce=false`, so it warns but does not kill. Keep it enabled; warnings are useful diagnostics for high image_aux and LoRA interaction.

### Q6 — Early stop protocol

**MODIFY.** Do not auto-stop at step 30K on `val_chain_normal_mse`. A4-mid's early rolling/full-val trajectory is not reliable enough for a hard stop. Use status-only checkpoints at 50K/100K full-val, require multiple bad signals, and ask user before killing unless there is OOM/NaN/dead process/wrong config.

### Q7 — X3 10K short-training limitation

**MODIFY interpretation.** X3 at 170K is correct for A3-matched additivity. It cannot prove long-run LoRA redundancy. If X3 ≈ A4-mid, write: “under the A3-matched 10K LoRA adaptation window, LoRA adds little beyond strong image_aux.” Do not claim decoder LoRA can never add under longer training.

### Q8 — NOT-DO completeness

**MODIFY.** Add explicit bans on changing X1 L1/border settings, X3 LoRA target/root/rank/alpha/dropout/init fields, `optimizer.decoder_lr_mult`, `optimizer.decoder_weight_decay`, X1-lite v2/v3, X3 extension to 200K, and any paper phrasing that treats X3 as “V18 rescued.”

### Q9 — New bias check

**EXTEND.** The prompt's five candidate issues were useful but missed the biggest launch failure: missing `--resume`. This is B93-class self-review tunnel vision: task authors verified YAML diff but not CLI semantics.

## 4. Mandatory Fixes Before Push

| severity | location | issue | required fix |
|---|---|---|---|
| HIGH | §3 B.2 | X3 launch omits `--resume`; YAML `training.resume_from` is ignored | pass V7.best via `--resume`; require resume log checks |
| HIGH | §3 B.2 | X1 PID is grepped from a log that never records it | write/read `X1_lite.pid` sidecar |
| HIGH | §3 B.2 | X3 GPU selection can reuse X1 GPU | record/read `X1_lite.gpu`; exclude it |
| HIGH | §6 | anti-check pipelines do not fail reliably | replace with direct forbidden-file helper |
| MED-HIGH | §3 B.0 | X3 KL disabled only by lambda; frozen RAE still built | set `decoder_kl_pullback.enabled=false` |
| MED | §3 B.0 | A3 protocol implies 5K save interval | add `training.save_interval=5000` or remove step_165K expectation |
| MED | §2 A.1/A.2 | raw `img_ssim` expected tiny | check YAML and `img≈img_l1` instead |
| MED | §4 report | X3 result can be overclaimed | label as 10K A3-matched window only |

## 5. Design-Level Concerns

1. **X1-lite has a useful middle zone.** If it lands between V7 and A4-mid, write a fractional attribution story rather than launching more component runs immediately.
2. **X3 is short-window by design.** A neutral result should stop V18 follow-up for this paper, but not be written as universal LoRA impossibility.
3. **Both experiments remain single-seed.** Keep claims at fixed-run full-val/slice-level evidence; no patient-level or seed-robust inference.
4. **`enabled=false` for KL is a cleanup, not a scientific variable.** It removes unused frozen RAE construction while keeping KL conceptually off.

## 6. New Biases B93+

| ID | bias | severity | mitigation |
|---|---|---|---|
| B93 | YAML-field sufficiency anchor | HIGH | verify CLI semantics, not only config diffs |
| B94 | Config-field/CLI confusion | HIGH | `training.resume_from` does not replace `--resume` |
| B95 | Health-check telemetry illusion | HIGH | write PID/GPU sidecars, do not infer from train logs |
| B96 | Zero-weight metric confusion | MED | distinguish raw diagnostic losses from weighted total |
| B97 | Self-check theater | HIGH | no pipeline anti-checks; `ERR` must update in current shell |
| B98 | X3 short-training overclaim | MED | report as A3-matched 10K test only |
| B99 | KL-off half measure | MED-HIGH | set both `enabled=false` and `lambda_kl=0.0` |

## 7. Final Recommendation

Do **not** push the current task md to Codex as-is. Apply the mandatory fixes above, then the task should be safe to push. The core design is reasonable after fixes: X1-lite is a valid mechanism falsification, and X3 is a valid A3-matched additivity test if and only if it truly resumes from V7.best and keeps its interpretation bounded to the 10K adaptation window.
