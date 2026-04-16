# Research Review: Idea2 (Alpha-CCT)

- Date: 2026-04-14
- Project: `/home/qujiaxiang/project/PET_LatentResidual`
- Skill mode: `research-review`
- External reviewer agent:
  - Agent ID: `019d8a26-603c-7320-aba2-bfad814a9691`
  - Model: `gpt-5.4`
  - Reasoning effort: `xhigh`
- Internal supporting agents:
  - Quant review: `019d89fe-1d3e-71e1-9045-183a00f35ef6` (Pauli)
  - Mechanism audit: `019d89fe-1d78-7530-a51a-659cee99b052` (James)

## 1) Context Snapshot

Idea2 intends to fix train/inference mismatch for the fixed latent cascade:
`D50 -> D20 -> D10 -> D4 -> NORMAL`

Current experiment line:
- Config: [pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml](/home/qujiaxiang/project/PET_LatentResidual/configs/pet_flow/pet_flow_first_hop_224_50k_idea2_alpha_cct_seed43_gpu3.yaml)
- Training run: [first_hop_224_50k_idea2_alpha_cct_seed43_gpu3](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3)
- Training metrics: [metrics.jsonl](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3/metrics.jsonl)
- Full eval (best): [idea2 best full eval](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3_eval_clip3_best_full/first_hop_224_val_clip3_eval.json)
- Full eval (last): [idea2 last full eval](/data_2/qujiaxiang/outputs/PET_LatentResidual/first_hop_224_50k_idea2_alpha_cct_seed43_gpu3_eval_clip3_last_full/first_hop_224_val_clip3_eval.json)
- Baseline diagnostics:
  - [chainstable best full](/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/chainstable50k_best_full/hop_difficulty_clip3_val_chainstable50k_best_full.json)
  - [strict best full](/data_2/qujiaxiang/outputs/PET_LatentResidual/diagnostics/strict_best_full/hop_difficulty_clip3_val_strict_best_full.json)

## 2) Round-by-Round Review

## Round 1 (External xhigh review)
Reviewer verdict:
- Claim `"Alpha-CCT + conflict damping reduces mismatch"` is **not supported**.
- Narrow claim `"some late checkpoints can beat baseline best"` is **partially supported**.

Primary criticisms:
1. Matching branch can be effectively gradient-dead yet still consumes CCT curriculum budget.
2. Rolling-window validation selection makes `best.pt` vs `last.pt` evidence unreliable for method conclusions.
3. `conflict_damp` uses loss ratio proxy, not true gradient conflict.
4. Full-eval gain is tiny/asymmetric for top-tier claim.
5. CCT has hop0 consistency blind spot by construction.

Author-side evidence response:
- Agreed that `idea2_best_full` is slightly below chainstable best on D10/D4/NORMAL.
- Confirmed with full eval that `idea2_last_full` is above chainstable best, implying selection artifact is real.

## Round 2 (Targeted follow-up)
Reviewer provided implementation-ready recovery plan:
- E0: checkpointing audit first (full-eval symmetric comparisons)
- E1: pure-consistency reference
- E2: pure-consistency + lower image_aux
- E3: live matching (no damp)
- E4: live matching + relaxed damp

Hard stop/go gates were specified (tail/NORMAL thresholds + seed-expansion policy).

## 3) Consolidated Quantitative Findings

Full-val PSNR (7403 slices):

| Model | D20 | D10 | D4 | NORMAL | tail_avg |
|---|---:|---:|---:|---:|---:|
| chainstable_best_full | 35.4881 | 35.8620 | 36.4207 | 36.7422 | 36.1283 |
| idea2_best_full | 35.4947 | 35.8483 | 36.4111 | 36.7104 | 36.1161 |
| idea2_last_full | 35.5036 | 35.8670 | 36.4294 | 36.7733 | 36.1433 |

Interpretation:
- `idea2_best_full - chainstable_best_full` = small net negative.
- `idea2_last_full - chainstable_best_full` = small net positive.
- Therefore, method quality and checkpoint-selection quality are entangled.

Training dynamics (`metrics.jsonl`):
- best_by_select step: `46800`
- last step: `50000`
- `val_select_score` worsens from `5.397e-4` to `6.836e-4`.
- Yet local losses (`val_pair_total`, `val_rollout_total`, `val_cct_total`) decrease.
- Chain terminal metrics worsen late; indicates objective tension / mismatch in selection signal.

## 4) Final Consensus (Actionable)

1. Current Alpha-CCT recipe is **not paper-ready**.
2. Direction is **not dead** (late checkpoint can exceed strong baseline).
3. Highest-priority blocker is **evaluation/selection validity**, not only method knobs.
4. Next cycle should simplify first, then re-add complexity only if each increment proves value.

## 5) Results-to-Claims Matrix

| Outcome | Claim status | Allowed claim | Disallowed claim |
|---|---|---|---|
| Current evidence only | `partial/no` | "Late CCT checkpoints can slightly outperform one baseline-best checkpoint." | "Alpha-CCT robustly solves train/inference mismatch." |
| After E0, win disappears | `no` | "Prior win was selection artifact." | Any superiority claim |
| E1/E2 win, E3/E4 no extra gain | `partial` | "Simple consistency regularization helps this cascade." | Matching/damping as core contribution |
| E3 wins E1/E2, E4 no gain | `partial` | "Live matching may help under this setup." | Conflict-damp necessity |
| E4 wins reproducibly (3 seeds, fixed selector) | `yes (narrow)` | "Curriculum consistency training modestly improves pure-pred PET latent cascade robustness in this codebase." | Broad generic mismatch-solving claims |

## 6) Prioritized TODO + Compute Budget

Assume single-GPU 50k run ~24 GPU-hours, full-val eval ~0.5 GPU-hours.

1. E0 checkpoint audit (eval-only) — ~2-3 GPU-hours
- Full-eval: `chainstable best/last`, `idea2 best/last`, optional late steps.
- Must define true reference using symmetric protocol.

2. E2 pure-consistency + lower image aux — ~24 GPU-hours
- Keep method simple; verify whether CCT signal is drowned.

3. E3 matching live (no damp) — ~24 GPU-hours (only if E2 passes)
- Isolate matching value without damp confound.

4. E4 relaxed damp — ~24 GPU-hours (only if E3 passes)
- Keep only if it beats E3 on full-val chain metrics.

5. 3-seed confirmation for winner — +48 GPU-hours
- Require consistent tail/NORMAL gains.

## 7) Stop/Go Policy

Define `Ref` after E0:
- `Ref = stronger of chainstable_best_full and chainstable_last_full`.

Single-seed promotion gate:
- `tail_avg >= Ref + 0.01 dB`
- `NORMAL >= Ref + 0.02 dB`
- `D10 >= Ref - 0.01 dB`
- `D4 >= Ref - 0.01 dB`

If fail: stop that branch.
If pass: run +2 seeds and re-check mean gains.

## 8) Narrative Guidance for Paper Positioning

If continuing this line, narrative should be:
- "Deployment-aligned consistency training for fixed PET latent cascade"
- Not: "new generic conflict-damping theory"
- Lead with full-chain open-loop evidence and strict symmetric checkpoint protocol.

## 9) Related Artifacts

- Failure report: [REPORT_IDEA2_ALPHA_CCT_FAILURE_20260414.md](/home/qujiaxiang/project/PET_LatentResidual/docs/experiment_plans/REPORT_IDEA2_ALPHA_CCT_FAILURE_20260414.md)
- Skill review source prompt/response kept in agent history (`Hume`).
