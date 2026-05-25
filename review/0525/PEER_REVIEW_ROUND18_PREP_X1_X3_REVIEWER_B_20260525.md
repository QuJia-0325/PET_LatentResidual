# Peer Review — Round 18-Prep X1-lite + X3 (Reviewer B)

- Date: 2026-05-25
- Role: Reviewer B (independent)
- Target: review/0525/CODEX_TASK_ROUND18_X1_X3_20260525.md
- Scope: Execution/design feasibility only (not strategy re-selection)

## 1) Candidate Issues (CI-1..CI-5)

### CI-1 — GPU selection race (claimed HIGH)
- Verdict: EXTEND (real risk exists, but the immediate blocker is PID check logic, not only GPU race)
- Severity: HIGH
- Evidence:
  - B.2 checks X1 liveness by parsing PID from X1 log:
    - `X1_PID=$(grep -oE 'PID=[0-9]+' "$X1_LOG" ...)`
  - But `PID=...` is printed to terminal (`echo "X1-lite launched: PID=..."`), not appended into `X1_LOG` in A.1.
  - Result: `X1_PID` can be empty at B.2, `ps -p "$X1_PID"` fails, X3 launch is aborted even if X1 is healthy.
  - Separately, B.2 chooses GPU by max free memory without explicitly excluding X1 GPU; under transient memory patterns, same-device reuse is possible.
- Conclusion: There is a hard launch blocker plus a secondary race/OOM risk.

### CI-2 — `total_steps_override=200000` with `max_steps=170000` (claimed MED)
- Verdict: VERIFY
- Severity: LOW (design choice, not correctness bug)
- Evidence:
  - A3 config keeps `lr_schedule.total_steps_override: 200000` (review/0517/V18_capacity_only/V18_capacity_only.yaml).
  - X3 inherits same schedule convention from V18 and only changes allowed fields.
  - A3 metrics/log lines around 170k show consistent continuation state (resume from 160k, then short-horizon progression with KL=0 path).
  - KL drift summary reports near-tie between A3-cap and V18@170k on direct-decode substrate.
- Conclusion: This is consistent with A3 paired design intent; keep as-is.

### CI-3 — early monitoring/early stop weakness (claimed MED)
- Verdict: VERIFY (with caveat on threshold calibration)
- Severity: MEDIUM
- Evidence:
  - Task monitoring table mostly checks alive/step milestones; no explicit stop rule tied to expected trajectory.
  - A4-mid file has step-30000 val_full available (`review/0521/A4_image_aux_lambda_08/A4_metrics_20260525_final.jsonl`, includes `event=val_full, step=30000, val_chain_normal_mse`).
  - Therefore, adding a robust checkpointed early gate (prefer val_full over noisy rolling val) is feasible.
  - I did not find a canonical V7 step-30000 jsonl snapshot under current review paths, so a fixed `5e-5` threshold should be treated as provisional until baseline file is pinned.
- Conclusion: Add explicit early-stop protocol, but define threshold relative to a pinned baseline artifact (or use a rank-based rule).

### CI-4 — SSIM/seam still computed at weight 0 (claimed LOW)
- Verdict: VERIFY
- Severity: LOW
- Evidence:
  - `pet_lr/losses_first_hop.py` computes SSIM and seam unconditionally, then composes:
    - `total = w_l1*loss_l1 + w_ssim*loss_ssim + w_seam*loss_seam`
  - With `w_ssim=w_seam=0`, gradients from those branches are zero-contributed to final scalar objective.
  - No evidence of module-state mutation side effects in these loss functions (functional operations, no BN/dropout state updates inside the losses).
- Conclusion: Semantically correct for ablation; only compute overhead remains.

### CI-5 — keep `decoder_kl_pullback.enabled=true` while `lambda_kl=0` (claimed LOW)
- Verdict: VERIFY
- Severity: LOW
- Evidence:
  - In trainer KL block, actual KL computation is gated by scheduled lambda:
    - `if lambda_kl > 0.0: ...`
  - With X3 `lambda_kl=0.0`, KL term is not computed and contributes zero.
  - Keeping `enabled=true` is not harmful functionally, though `enabled=false` is cleaner and reduces ambiguity.
- Conclusion: Optional cleanup, not mandatory for correctness.

## 2) Deep Questions (Q1..Q9)

### Q1 — X1-lite 6-field diff sufficiency
- Verdict: APPROVE
- Notes:
  - Core mechanism contrast is clean: only lambda increase + SSIM/seam weights to zero.
  - Not changing `border_weight`, seam patch settings, and extended seam flags is acceptable because seam term is zero-weighted and L1 path remains intentionally unchanged.

### Q2 — X3 6-field diff sufficiency
- Verdict: APPROVE (with one clarity edit)
- Notes:
  - For paired comparison against A3, 6-field diff is correct and minimal.
  - KL `kill_switch.*` fields appear config-resident/documentary under current code path (no consumption found in trainer scan).
  - Keep LoRA optimizer/freeze inheritance untouched (correct).

### Q3 — X1-lite vs A4-mid matchedness
- Verdict: APPROVE
- Evidence:
  - A4-mid keeps V7-style `ssim_weight=0.25`, `seam_weight=0.1`, while X1-lite sets both to 0.
  - This supports intended mechanism isolation.

### Q4 — X3 vs A3 matchedness
- Verdict: APPROVE
- Evidence:
  - A3 is V18-capacity-only short-horizon control; X3 differs by image_aux lambda only.
  - Existing KL drift summary indicates A3/V18@170k direct-decode near-tie, so paired assumption is acceptable.

### Q5 — loss balance watchdog and X3 kill risk
- Verdict: APPROVE (no hard kill expected)
- Evidence:
  - Trainer reads `loss_balance_watch_enforce` and defaults false in these configs.
  - Dominance watch may warn but should not terminate run under current settings.
- Recommendation:
  - Log warning policy explicitly in task doc to avoid operator confusion.

### Q6 — early-stop protocol
- Verdict: MODIFY
- Reason:
  - Current protocol lacks explicit stop criterion; this is a cost-control gap.
  - Use a pinned baseline artifact and val_full-based checkpoint gate (not only rolling val).

### Q7 — X3 10K short-train may underestimate LoRA
- Verdict: APPROVE with limitation note
- Notes:
  - For paired X3 vs A3, 10K is consistent.
  - For broad claim "LoRA redundant overall", 10K is insufficient; must be framed as "under this short-horizon protocol".

### Q8 — NOT-DO completeness
- Verdict: MODIFY
- Missing/under-specified items:
  - Explicitly ban changing `loss.image_aux.l1_weight` in X1-lite.
  - Explicitly ban touching LoRA optimizer knobs in X3.
  - Explicitly document no automatic X3 extension to 200k without next-round approval.

### Q9 — bias checks (B93+)
- Verdict: EXTEND
- Notes:
  - Prompt has mild execution-urgency bias; reviewer should still allow design-level caveats.
  - No severe echo-chamber lock detected, but candidate-issue framing may underweight new failure modes (the PID parsing bug is such a mode and should be elevated).

## 3) Main Verdict

## MODIFY-BEFORE-PUSH

Rationale: At least one HIGH-severity execution blocker exists in launch script logic (PID parsing from wrong source). If uncorrected, codex can fail before starting X3 despite healthy X1.

## 4) Must-Fix Items (before push)

1. Location: `§3 B.2 staggered launch` in target task doc  
   Original: `X1_PID=$(grep -oE 'PID=[0-9]+' "$X1_LOG" | ...)`  
   Fix: Persist PID to file in A.1 and read it in B.2, or directly parse shell job PID from launch step artifact. Example:
   - A.1 add: `echo "$X1_PID" > review/0525/X1_lite_l1_only/X1.pid`
   - B.2 use: `X1_PID=$(cat review/0525/X1_lite_l1_only/X1.pid)`
   Severity: HIGH

2. Location: `§3 B.2 GPU selection`  
   Original: choose `FREE_GPU2` by max free memory only  
   Fix: Explicitly exclude X1 GPU (tracked from A.1 launch metadata), then pick best remaining GPU.
   Severity: MEDIUM

3. Location: `§A.2 / §B.3 monitoring protocol`  
   Original: progress/liveness-only checkpoints  
   Fix: Add explicit early gate using pinned baseline artifact and val_full checkpoint comparison (e.g., step 30k/50k decision rule).
   Severity: MEDIUM

4. Location: `§7 NOT-DO`  
   Original: missing constraints on `l1_weight` and LoRA optimizer knobs  
   Fix: Add explicit non-modification clauses for `loss.image_aux.l1_weight`, `optimizer.decoder_lr_mult`, `optimizer.decoder_weight_decay`, and no auto-extend to 200k without next-round approval.
   Severity: MEDIUM

## 5) Design-Level Notes (D-class)

- D1: X3 result interpretation must be scoped to short-horizon (10K LoRA) protocol.
- D2: If X3 ~= A4-mid, conclude "under current horizon" first, avoid overgeneralizing to full-horizon LoRA irrelevance.
- D3: Keep mechanism and additivity claims separated in reporting (X1-lite for mechanism; X3 vs A3 for additivity).
