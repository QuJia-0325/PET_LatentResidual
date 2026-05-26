# REVIEW INTEGRATION — Round 19-Pre (Next Slot After X3)

- date: 2026-05-26
- scope: integrate independent reviews on how to use the freed X3 training slot while X1-lite is still running
- sources:
  - PEER_REVIEW_round19_next_slot_after_X3_reviewer_D_20260526.md
  - PEER_REVIEW_round19_next_slot_after_X3_reviewer_C_copilot_20260526.md
  - PEER_REVIEW_round19_next_slot_after_X3_reviewer_E_20260526.md

---

## 1) Unanimous verdict

**3/3 reviewers select Option A: launch exactly one A4-mid seed replicate (`seed=1337`).**

Backup in all reviews: **Option C — no new training, wait for X1-lite**, if operational stability or writing bandwidth cannot support one extra run.

Rejected unanimously:
- **Option B (X1-v2-balanced)**: premature before X1-lite outcome; only useful if X1-lite lands in the interior ambiguity zone.
- **Option D (X3-extend to 200K)**: V18-family sunk-cost; violates X3 hard stop and has low paper value after X3 non-additive result.
- Any invented Option E: no better single-run training option under current constraints.

---

## 2) Rationale for Option A

A4-mid is now the paper headline, but it is still single-seed. V14 proves V7 at `lambda_img=0.04` is seed-stable (`V14 - V7 ~= -0.0004 dB`), but this cannot be automatically extrapolated to A4-mid at `lambda_img=0.08`, because the stronger pixel supervision changes the optimization pressure and potentially the basin.

A4-mid-seed1337 is therefore the highest-EV use of the freed slot because it directly de-risks the paper's main result. It is not justified by slot utilization; it is justified by headline robustness.

---

## 3) Locked experimental design

Base config:
- `review/0521/A4_image_aux_lambda_08/A4_image_aux_lambda_08.yaml`

Allowed functional diff:
- `seed: 42 -> 1337`

Required isolation diff:
- `output_dir -> /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0525_runs/A4_mid_seed1337`
- `run_name -> first_hop_224_a4_mid_seed1337`

Everything else must remain identical to A4-mid:
- `training.image_aux.lambda_start = 0.08`
- `training.image_aux.lambda_max = 0.08`
- full image_aux components (`l1_weight=1.0`, `ssim_weight=0.25`, `seam_weight=0.1`)
- `training.max_steps = 160000`
- `lr_schedule.total_steps_override = 200000`
- no decoder LoRA / no KL / no resume

---

## 4) Stop rule

A4-mid-seed1337 is exactly one seed replicate of the locked A4-mid setting.

Do not launch:
- seed2024 / seed7 / any additional seed sweep
- any new lambda point
- X1-v2-balanced before X1-lite outcome
- X3-extend / V19 / V18-clean / decoder-rank variants

If A4-mid-seed1337 collapses, paper must be rewritten. If it holds, paper gains stronger robustness. If it partially drops but remains above V7 + 0.05 dB, paper reports a conservative seed-sensitive range.

---

## 5) Expected interpretation

| outcome | interpretation |
|---|---|
| within 0.02 dB of A4-mid | headline robust under seed=1337; strong paper support |
| > V7 + 0.05 dB but >0.02 below A4-mid | effect real but seed-sensitive; report range |
| near V7 | A4-mid likely seed-lucky; downgrade headline |

---

## 6) Bias audit

| ID | bias | mitigation |
|---|---|---|
| B97 | A4-mid seed robustness fetish | one replicate only; do not claim full variance estimate |
| B98 | CL1 overreaction | wait for X1-lite before any X1-v2-balanced |
| B99 | idle-slot rationalization | justify by paper-headline robustness, not utilization |
| B100 | V18 sunk-cost | explicitly ban X3-extend / V18-family continuation |
| B101 | single-seed false security | phrase as seed-1337 robustness, not multi-seed significance |

---

## 7) Action

Create and push `CODEX_TASK_ROUND19_A4_MID_SEED1337_20260526.md` for Codex execution. No further peer review required before launch because all three reviewers agree and the task is a simple seed-only replicate with mechanical YAML diff verification.
