# Round 15 Peer Review — Reviewer B (GitHub Copilot)

- date: 2026-05-18
- reviewer slot: **Reviewer B**
- reviewer name: **GitHub Copilot**
- prompt: `PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md`
- scope: 3-slot launch allocation for A3 V18-capacity-only + V13 image_aux ablation + V14 d_pure multi-seed
- stance: independent launch-order / ROI audit; not a re-review of A3/V13/V14 yaml design

---

## 0. Executive Verdict

| item | verdict | reason |
|---|---|---|
| slot 3 decision | **B — run V14** | V14 is not merely paper polish; it is the clean d_pure / seed-noise substrate needed to interpret V13 thresholds and small V18 deltas. |
| launch order | **launch all three, but gated/staggered** | Do not literally fire three long jobs at once. Launch A3 + V13 first, then V14 after a fresh GPU/RAM check and V14 task-md smoke. No need to wait 48h for A3 outcome. |
| V14 task md | **required** | V14 has yaml only. Direct `python --config` launch would bypass pass criteria, NOT-DOs, smoke handling, and commit/report discipline. |
| missing better control | **none higher-ROI for slot 3 right now** | V18-clean / V13-capacity-only / V19 are all downstream of A3/V13 results; V14 is independent and already defined. |
| concurrency risk | **MODIFY, not reject** | Output dirs and dataloaders are isolated, but historical 3-job runs produced tight system RAM. Add hard resource gates and staggered launch. |

One-line recommendation: **Proceed with 3-slot plan only after preparing a small V14 launch task md; use staggered launch with per-job resource gates.**

---

## 0.5 Evidence Checked

### Yaml / run isolation

| experiment | output_dir | run_name | seed | workers | batch | max_steps | fresh dir |
|---|---|---|---:|---:|---:|---:|---|
| A3 | `review_0517_runs/V18_capacity_only/run` | `first_hop_224_v18_capacity_only` | 42 | 0 | 8 | 170000 | true |
| V13 | `review_0516_runs/V13_true_image_aux_ablation/run` | `first_hop_224_v13_true_image_aux_off` | 42 | 0 | 8 | 160000 | true |
| V14 | `review_0516_runs/V14_true_d_pure/run` | `first_hop_224_v14_v7_seed1337` | 1337 | 0 | 8 | 160000 | true |

No output-dir or run-name collision found. `num_workers=0` on all three lowers dataloader concurrency risk.

### Task-md state

- A3 has dedicated task md: `CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md`.
- V13 has Phase A v3 task md, but that task explicitly says slot 3 remains empty and NOT-DO #2 says **未 launch V14 / V9 / V21 任何变体**.
- V14 has yaml but **no task md**.

Therefore V14 must not be shoehorned into the V13 Phase A v3 task. It needs a separate launch task.

### Concurrency history

Historical snapshot from 2026-05-06 shows V7/V8/V6_NOISE all running, but also warns:

> System RAM is tight because three project loaders plus existing GPU jobs keep large latent tensors resident; monitor OOM risk.

The same snapshot shows 503 GiB RAM total, 453 GiB used, 49 GiB available, and multiple GPU jobs. This proves multi-job operation has precedent, but **does not prove 3-slot launch is risk-free**.

---

## 1. Q1 — V14 Slot-3 ROI

**Verdict: APPROVE B, with reframing.**

The strongest argument for V14 is not “slot 3 would otherwise be wasted.” Empty capacity has option value, so that framing is biased. The real argument is:

- V14 is the first clean V7-vs-seed control under the same Gronwall + image_aux configuration.
- V13’s own yaml pre-registers an image_aux threshold using d_pure: `3 * d_pure`, with fallback `0.026 dB` only until V14 exists.
- The project’s active deltas are now tiny (`+0.030`, `+0.062`, `+0.099 dB`), so a true seed-noise estimate is decision substrate, not cosmetic paper garnish.
- V14 is independent of A3 and V13. A3 deciding that V18 KL is useless does not make the V7 seed-noise floor stale.

The earlier “不跑 multi-seed” stance should be treated as contextual, not absolute, because the current user decision explicitly reopens slot 3. Still, the review should not pretend slot idleness is new evidence; the new defensible basis is that V14 also stabilizes V13 interpretation.

### Anti-B arguments I still respect

- If capacity-only returns in 48h and demands two new urgent long runs, V14 occupies one possible long-run slot.
- If the project pivots away from this V7/V18 lineage, V14 becomes less valuable for method development.

But A3 itself frees slot 1 after 24-48h. That preserves one follow-up lane, so V14’s opportunity cost is acceptable.

---

## 2. Q2 — 3-Slot Concurrency Risk

**Verdict: MODIFY.**

The plan is plausible but the prompt overstates “无冲突.” The safe statement is:

> Three independent tasks are structurally isolated in yaml, but launch must be gated by live GPU/RAM preflight and staggered verification.

### What looks safe

- Output dirs are distinct.
- Run names are distinct.
- `require_fresh_output_dir=true` on all three full runs reduces accidental overwrite risk.
- All three set `num_workers=0`, so dataloader process fan-out should not multiply.
- A3 is a resume + small LoRA run, likely lighter than full from-scratch V13/V14.

### What is not proven safe

- Host RAM. Prior 3-project-loader operation left only ~49 GiB available on a 503 GiB machine and was explicitly marked tight.
- “Server memory max 3 tasks” is a rule of thumb, not a proof that any three current jobs plus external jobs are safe.
- If another user process exists, the safe task count may drop below 3.

### Required launch gates

Before each long launch, Codex should record:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
free -h
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -20
```

Suggested hard stop:

- fewer than 3 truly free GPUs -> do not force 3-slot launch
- `MemAvailable < 80GiB` before V14 -> do not launch V14 yet
- any selected GPU has >2 GiB used by unknown process -> choose another GPU or stop

---

## 3. Q3 — Launch Order

**Verdict: launch all three, but staggered/gated.**

I reject two extremes:

- Reject “literal simultaneous launch” because it hides resource-failure attribution. If RAM/OOM happens, you cannot tell which job caused it.
- Reject “wait 48h before V13/V14” because V13 and V14 are independent and their 7-day wall clock dominates.

Recommended operational order:

1. **A3 first**: short, high-information, resume-based. Verify alive at +5 min and that resume/log strings are correct.
2. **V13 second**: run its existing smoke path, then full launch if smoke passes.
3. **V14 third**: only after a separate V14 task md exists, a 200-step smoke passes, and RAM/GPU gates still pass.

This is “3-slot launch” in scheduling terms, not “press enter three times in one terminal” launch.

---

## 4. Q4 — V14 Task Md Requirement

**Verdict: required.**

V14 yaml is ready, but a naked command is not enough. The project’s own failure history argues for task-level guardrails: CLI flag checks, smoke, pass criteria, NOT-DOs, and explicit commit/report paths.

Also, the current Phase A v3 task cannot be reused directly because it explicitly says no V14 launch. Reusing it without a V14-specific wrapper would create a self-contradictory execution record.

### Recommended V14 task contents

Create a small file such as:

`CODEX_TASK_STAGE_C_V14_DPURE_MULTISEED_20260518.md`

Required sections:

- yaml sanity: `seed=1337`, `run_name=first_hop_224_v14_v7_seed1337`, `output_dir` under `V14_true_d_pure`, `image_aux.enabled=true`, no `resume_from`, `max_steps=160000`
- CLI flag assert: only `--config --resume`
- 200-step smoke copied from V13 template with `log_interval=10`, unique smoke output dir, no mutation of original V14 yaml
- full launch after smoke pass
- +5 min live check and GPU memory check
- NOT-DO: do not modify V14 yaml, do not launch V9/V21/V18-clean, do not infer 7-day convergence from smoke, do not write paper claims from V14 alone
- commit/report instructions that do not add `/data_2` checkpoints

This is a 10-20 minute prep cost and removes the biggest execution-process gap in option B.

---

## 5. Q5 — Missing Higher-ROI Control

**Verdict: no replacement for V14 now. Defer the other controls.**

| candidate | decision | reason |
|---|---|---|
| V13-capacity-only | **defer** | Undefined design, mixes V13 image_aux-off with V18-like decoder LoRA. It is not a clean substitute for V14. |
| V8 retrain with V7 config | **defer** | Potentially useful if V13 result is surprising, but V13 already addresses the main image_aux single-axis question first. |
| V18-clean | **defer until A3** | A3 is the gate for whether KL/design mechanism is worth reviving. |
| V19 / decoder LoRA variants | **reject for slot 3** | Decoder variants are lower EV until A3 tells whether LoRA capacity or KL matters. |
| leave slot 3 empty | **reject as default** | Buffer value is real, but A3 frees a buffer slot quickly; V14’s decision value is durable. |

The current trio maps well to the immediate unknowns:

- A3: V18 GT drift mechanism
- V13: true image_aux contribution
- V14: true seed/d_pure scale

That is a coherent substrate package for Round 16.

---

## 6. Q6 — New Biases B55+

### B55 — Filling-capacity bias

The prompt’s “slot 3 不跑 = 浪费” language is biased. Empty slot 3 is not waste; it is option value.

Fix: justify V14 by d_pure/V13-threshold value, not by GPU guilt.

### B56 — Paper-anchor bias

“d_pure noise floor is paper 必备” over-anchors on a future paper path before A3 returns.

Fix: say V14 is needed for **decision calibration** and possible paper reporting. Paper is secondary.

### B57 — User-decision drift

The plan risks treating the earlier “不跑 multi-seed” decision as already obsolete because a slot is open. Slot openness was not new information.

Fix: record the current user decision as a new contextual override, not as something Claude can infer unilaterally.

### B58 — Concurrency confidence bias

The prompt says all slots are empty and server max is 3 tasks, but historical evidence shows system RAM can become tight during 3-task runs.

Fix: add live RAM/GPU gates and staggered launch.

### B59 — Task-md gap bias

Because V14 yaml exists, the prompt underweights execution drift from missing task md.

Fix: V14 task md is mandatory before launch.

### B60 — Phase A v3 scope conflict

Phase A v3 NOT-DO explicitly forbids launching V14. If Codex launches V14 under that task umbrella, the audit trail contradicts itself.

Fix: launch V14 from a separate Stage C task md and separate report/commit.

### B61 — Expected d_pure range anchor

The prompt gives V14 expected `±0.01-0.03 dB`. That may be plausible but should not become a pass/fail prior.

Fix: V14 reports the measured seed-noise distribution; do not label results surprising solely because they exit this guessed range.

### B62 — Simultaneous-vs-parallel conflation

“3-slot launch” is scheduling parallelism, not simultaneous process creation.

Fix: use staged launch with per-job verification.

---

## 7. Prompt-Format Output

### 7.1 Six Questions

| Q | verdict | short answer |
|---|---|---|
| Q1 V14 ROI | **APPROVE B / MODIFY framing** | Run V14, but justify by d_pure/V13 threshold, not “slot waste.” |
| Q2 concurrency | **MODIFY** | Structurally isolated, but require live GPU/RAM gates and staggered launch. |
| Q3 launch order | **MODIFY** | Launch all three without waiting 48h, but in order A3 -> V13 -> V14 with verification between jobs. |
| Q4 V14 task md | **APPROVE need task md** | Required; direct run is too loose and conflicts with Phase A v3 NOT-DO scope. |
| Q5 missing controls | **APPROVE current trio** | No higher-ROI replacement for slot 3 before A3/V13 data lands. |
| Q6 new bias | **B55-B62 found** | Main risks: filling-capacity, paper-anchor, concurrency overconfidence, missing task md. |

### 7.2 Slot 3 Decision

**B: slot 3 = V14**, after a dedicated V14 task md and live resource gates.

### 7.3 Launch Order Verdict

**All three should be launched in the current wave, but staggered:**

1. A3 first
2. V13 second after A3 +5 min health check
3. V14 third after V14 task-md smoke and fresh RAM/GPU check

Do not wait 48h for A3 unless the first two launches already reveal RAM/GPU pressure.

### 7.4 V14 Task Md Verdict

**Needs prep.** Use the V13 Phase A v3 smoke/pass-criteria pattern as a template, but create a separate V14-specific Stage C task. Do not reuse Phase A v3 as-is.

### 7.5 Final Recommendation

Proceed with option **B** only in this safer form:

> A3 + V13 + V14 in one launch wave, resource-gated and staggered, with a new V14 task md before V14 launch.

This preserves wall-clock efficiency while avoiding the two real Round 15 risks: filling-capacity bias and under-specified V14 execution.
