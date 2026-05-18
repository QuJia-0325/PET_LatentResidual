# Peer Review Round 14 — Copilot A3 Capacity-Only Execution Prep Review

- date: 2026-05-18
- reviewer: GitHub Copilot
- prompt: `PEER_REVIEW_PROMPT_round14_A3_capacity_only_20260518.md`
- scope: `V18_capacity_only.yaml`, `CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md`, `V18_design_rationale.md` §5.2

---

## 0. Executive Verdict

| object | verdict | reason |
|---|---|---|
| `V18_capacity_only.yaml` | **READY TO USE** | Parsed YAML diff vs V18 is exactly 4 fields. The intended invariants hold. |
| `CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md` | **MODIFY BEFORE PUSH** | Launch is mostly executable, but eval step assumes `--v18-cap-ckpt`, which the current probe does not support. Outcome space and pass greps also need tightening. |
| `V18_design_rationale.md` §5.2 | **MODIFY** | B43-B47 are useful, but mostly cultural rather than mechanically enforced; old §5.1 “no more peer review” redline now conflicts with B45. |

Recommended push order: **B — push commit `7486776` first, then push A3 in a separate commit after the task-md fixes.** Keep V13/V21-cleanup and A3 capacity-only isolated for auditability.

---

## 0.5 Mechanical Checks Performed

### YAML Diff

I parsed `V18_decoder_lora.yaml` and `V18_capacity_only.yaml` with `yaml.safe_load` and recursively compared dictionaries.

Result: **exactly 4 parsed diffs**.

| path | V18 | V18-capacity-only |
|---|---|---|
| `loss.decoder_kl_pullback.lambda_kl` | `0.05` | `0.0` |
| `output_dir` | `.../V18_decoder_lora/run` | `.../V18_capacity_only/run` |
| `run_name` | `first_hop_224_v18_decoder_lora` | `first_hop_224_v18_capacity_only` |
| `training.max_steps` | `200000` | `170000` |

Invariant checks all passed:

| check | result |
|---|---|
| output_dir ends with `V18_capacity_only/run` | true |
| run_name is `first_hop_224_v18_capacity_only` | true |
| max_steps is `170000` | true |
| lambda_kl is `0.0` | true |
| decoder_lora rank is `32` | true |
| decoder_lora last_n_blocks is `2` | true |
| resume_from equals V18 resume_from | true |
| lr_schedule.total_steps_override is `200000` | true |
| seed is `42` | true |
| use_pred_latent remains `true` | true |

### Probe CLI Check

`tools/probe_v18_kl_drift.py` currently supports:

```text
--config --v18-config --v7-ckpt --v18-best-ckpt --v18-last-ckpt --out-dir --batch-size --max-slices --split --device --num-workers
```

It does **not** currently support `--v18-cap-ckpt` / `--v18-capacity-ckpt`. The task’s B1 command will fail unless Codex first patches the probe.

### Trainer CLI / Resume Check

`train_first_hop.py` only exposes `--config` and `--resume`, matching the task’s A0 CLI check. Resume logging uses actual strings like:

- `[startup] resume from: ...`
- `[resume][warm-start] architecture changed → start_step=160000, ...`
- `[resume] loaded step=160000, ...`

The task’s pass criterion should grep those strings, not the literal phrase `Resuming from checkpoint`.

---

## 1. Six Questions

### Q1 — YAML 4-Field Diff: APPROVE

The parsed diff is exactly the four expected fields. No extra parsed YAML drift exists.

Important notes:

- `training.decoder_lora.rank=32`, `last_n_blocks=2`, `alpha=16`, `dropout=0.0`, and target keywords are unchanged.
- `training.resume_from` remains V7 best.pt, matching V18.
- `lr_schedule.total_steps_override=200000` remains unchanged, so the 170K endpoint is on the same absolute cosine schedule as V18@170K.
- `use_pred_latent=true` remains, but `lambda_kl=0.0` makes KL inactive because `train_first_hop.py` only computes `z_kl` inside `if lambda_kl > 0.0`.

No YAML modification required.

### Q2 — Capacity Isolation: MODIFY

The control is valid for the **early V18.best-scale question**, not for the full V18.last trajectory.

Why the 10K design is defensible:

- V18.best was selected at 165K, only +5K from V7, and already showed the main `+0.097 dB` GT-latent improvement.
- A 170K capacity-only run gives LoRA twice that early window while preserving same LR/schedule stage.
- The result should be compared primarily against **V18.best**, not V18.last.

Boundary condition to document more explicitly:

- If capacity-only is near V18.best, B is strongly supported.
- If capacity-only is near V7, do **not** resurrect the original A phrasing (“KL pullback aligns z_GT”). The correct interpretation is narrower: a KL-dependent optimization interaction or z_pred-mediated indirect effect may be present, despite direct z_GT KL being absent.
- If capacity-only is worse than V7, LoRA capacity under no-KL is harmful/unstable at 10K and the control falsifies “generic capacity is sufficient.”

Do not change to 40K now. That would answer a later/full-trajectory question and cost too much before the quick disambiguation returns.

### Q3 — Task §1 B42 + Outcome Map: MODIFY

The code quote is accurate:

```python
z_kl = main_out["z_pred"] if bool(kl_cfg.get("use_pred_latent", True)) else main_batch["z_dst"]
```

But the task needs three fixes:

1. **Probe precondition:** B1 currently says the probe supports adding `--v18-cap-ckpt`; it does not. Add a required pre-eval Task B0: patch `tools/probe_v18_kl_drift.py` to accept a fourth checkpoint, or provide a separate `probe_v18_capacity_kl_drift.py`.
2. **Fourth outcome:** add “capacity-only < V7” as a distinct result bucket. This means capacity-only LoRA is harmful or under-regularized at 10K, not merely “A/B/mixed.”
3. **Anchor language:** change “B is default / A impossible” to “B is the strongest prior after B42; the control decides whether generic capacity is sufficient.” Otherwise the task primes Codex to narrate toward B regardless of results.

### Q4 — §5.2 Standing Rules: MODIFY

The rules are useful but not equally enforceable.

| rule | status | review |
|---|---|---|
| B43 | cultural | Good qualifier, but no template/check enforces “in V18 buggy KL config.” Also “buggy” is a strong label; safer wording: “known-flagged KL config.” |
| B44 | cultural + control-backed | Mechanism description is accurate; capacity-only control makes it actionable. |
| B45 | cultural | Good policy, but conflicts with §5.1 redline “不允许再发 peer review.” §5.1 should be marked superseded, not left as a live redline with a parenthetical. |
| B46 | cultural unless scripts/templates change | Good comparison rule, but no report template or checker enforces 4×2 numbers. |
| B47 | potentially mechanical | Strongest rule, but only applies when verification scripts exist; task markdown should include assert-on-fail checks, not just prose. |

Recommended edit: add an enforcement column or tags: `[policy]`, `[template-required]`, `[script-enforced]`.

### Q5 — NOT-DO Completeness: MODIFY

Add these explicit NOT-DOs:

- Do not modify `V18_capacity_only.yaml` during execution unless A0 fails and user approves a corrected commit.
- Do not push unrelated commits while the capacity-only run is active.
- Do not reuse or write into the V18 decoder LoRA run directory.
- Do not edit V7/V18 checkpoint paths inside the probe command except to correct an absolute path failure with explicit reporting.
- Do not commit `.pid`, checkpoint, or `/data_2` outputs.
- Do not interpret capacity-only output as final paper/strategy verdict; it only gates Round 15.

Also revise the task’s “sleep 300” launch check for agent execution. Prefer “check after the terminal reports/after a manual 5-minute wait” or split into a separate follow-up command. Long sleeps in an agent terminal are brittle.

### Q6 — New Biases B48+

#### B48 — Probe-readiness phantom

The task implies the KL drift probe already supports a capacity checkpoint flag, but the current script does not. This is a readiness mismatch, not a fatal design flaw.

Fix: add Task B0 with an assert-on-fail argparse check before training finishes:

```bash
python tools/probe_v18_kl_drift.py --help | grep -q -- '--v18-cap-ckpt' || exit 1
```

or explicitly assign Codex to patch the probe before eval.

#### B49 — Outcome-space truncation

The task lists only V18-like / V7-like / intermediate outcomes. It omits capacity-only underperforming V7.

Fix: add fourth bucket: `capacity-only < V7 by >0.02 dB` → LoRA capacity without KL is harmful/unstable or 10K undertrained; do not infer A.

#### B50 — Prior-as-verdict anchor

Header and task text say “候选 B 是机制上唯一可能” and “B 是默认最简释.” This is reasonable as a prior but too strong for an execution prompt.

Fix: replace with “B is the current strongest prior; this task tests whether B is sufficient.”

#### B51 — Log-string mismatch in pass criteria

The task asks for `Resuming from checkpoint`, but the trainer prints `[startup] resume from:` / `[resume] loaded step=` / `[resume][warm-start] ... start_step=`.

Fix: grep actual strings.

#### B52 — Cultural standing rules labeled as permanent safeguards

B43/B45/B46 are policy statements, not mechanical guardrails. Calling them “permanent” can create false confidence.

Fix: tag enforcement mode and create templates/checkers for B43/B46 if they are meant to be hard gates.

#### B53 — Superseded redline left live

`V18_design_rationale.md` §5.1 still says “不允许再发 peer review,” while §5.2 B45 says substrate review must force reviewer intervention.

Fix: mark §5.1 item 3 as **SUPERSEDED by B45** or rewrite it as “avoid review-meta-review loops for tactical docs; substrate pivots require review.”

---

## 2. Overall Verdict

### YAML

**READY TO USE.** Parsed diff is exact. No hidden YAML drift found.

### Task Markdown

**MODIFY BEFORE PUSH.** Required edits:

1. Add explicit probe patch/preflight for `--v18-cap-ckpt`.
2. Add fourth harmful outcome bucket.
3. Neutralize B-as-verdict language.
4. Replace `Resuming from checkpoint` grep with actual trainer log strings.
5. Add missing NOT-DOs.
6. Avoid `sleep 300` as an agent instruction; split the check.

### Standing Rules

**MODIFY.** Keep B43-B47, but classify enforcement mode and resolve the §5.1 no-review conflict.

---

## 3. Recommended Push Order

**Option B:** push commit `7486776` first, then push A3 separately after task-md/rationale fixes.

Rationale:

- V13 launch/V21 retire cleanup and A3 capacity-only are separate audit trails.
- A3 task markdown needs minor but real fixes before execution.
- Keeping A3 isolated makes it easier to revert or review if the capacity-only setup changes.

---

## 4. Final Prompt-Format Answer

| Q | verdict |
|---|---|
| Q1 | APPROVE — YAML parsed diff exactly 4 fields |
| Q2 | MODIFY — valid early control; compare mainly to V18.best; document limits |
| Q3 | MODIFY — code quote accurate; add probe-preflight + fourth outcome + less anchoring |
| Q4 | MODIFY — rules useful but mostly cultural; classify enforcement and supersede old no-review redline |
| Q5 | MODIFY — add NOT-DOs and fix launch-check wording |
| Q6 | B48-B53 found |

Push after fixes: **yaml ready, task md modify, §5.2 modify, push order B**.
