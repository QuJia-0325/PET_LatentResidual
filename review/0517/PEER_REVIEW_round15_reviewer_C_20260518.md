# Round 15 Peer Review — Reviewer C (independent)

- **reviewer name**: **C** (per user instruction at Round 15 dispatch)
- date: 2026-05-18
- branch: foc_lite_hop0 (commit 24de2ab)
- substrate read independently (no other reviewer drafts seen):
  - PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md (the prompt)
  - review/0517/V18_capacity_only/V18_capacity_only.yaml (read in full)
  - review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml (key fields)
  - review/0516/V14_true_d_pure/V14_v7_seed1337.yaml (key fields, full directory listing)
  - review/0517/CODEX_TASK_STAGE_C_A3_V18_CAPACITY_ONLY_20260518.md (A3 task md)
  - review/0517/CODEX_TASK_PHASE_A_v3_20260518.md (V13 launch task md, NOT-DO §6)
  - review/0517/REVIEW_INTEGRATION_round13_20260518.md (R13 INTEGRATION, §3.3 + §6 user decision)
  - review/0517/REVIEW_INTEGRATION_round14_20260518.md (R14 INTEGRATION header)
- mandate per prompt §0/§4: launch-order + slot-allocation review, focus on V14 ROI + concurrency + Round-15 new biases.
- mechanical verifications I ran first (see §0 below).

---

## TL;DR (Reviewer C verdict)

| Item | Verdict |
|---|---|
| Slot 3 = V14 | **APPROVE WITH CONDITIONS** (V14 noise-floor is genuinely paper-necessary; but it requires a launch task md and an explicit override of Phase A v3 NOT-DO #2) |
| Launch order | **A3 + V13 同时 launch first; V14 launch after V13 smoke passes** (NOT all-3-simultaneous) |
| V14 task md needed | **YES, required** (~15 min prep). Not blocking but not skippable. |
| New biases | **B55 (HIGH), B56-B59** — 5 new. B55 is the headline finding: prompt §1.4 fabricates a R13 user quote that doesn't exist in any artifact. |

---

## 0. Mechanical verifications I ran first

### 0.1 R13 user quote on multi-seed — **NOT FOUND in any artifact**

Prompt §1.4 quotes user as having said in Round 13: *"不考虑 multi-seed, 先跑出结果, 再考虑 multi-seed"*. I grep-searched the entire `review/0516/` and `review/0517/` tree for this phrase (and variants `先跑出结果` / `不考虑.*multi` / `不跑.*V14` / `multi.?seed`):

```
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:6 (R15 prompt itself)
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:20 (R15 prompt itself)
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:50 (R15 prompt itself)
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:71 (R15 prompt itself)
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:85 (R15 prompt itself)
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:136 (R15 prompt itself)
review/0517/PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md:210 (R15 prompt itself)
```

**Zero hits outside the R15 prompt.** No R13 integration entry, no chat transcript, no Phase A v3 NOT-DO line carries this quote. R13 INTEGRATION §6 ("user 决策点 (3 个)") shows claude's *recommendation* `1=A` was "3 slot 同时 (V18-cap + V13 + V14)" — V14 was always part of the recommended plan. R13 §3.3 explicitly assigns `slot 3 = V14`.

This is **B55 (HIGH)**: a fabricated constraint that didn't exist becomes the centerpiece of the prompt's framing ("R13 user 决策推翻太快"). The whole Q1/Q6 dilemma — "is V14 a sunk-cost-driven reversal of user's Round 13 decision?" — dissolves once you check the source: there was no such Round 13 decision.

### 0.2 Phase A v3 NOT-DO #2 conflict — confirmed but scoped

Phase A v3 line 9: *"Slot 3 永远 0"*, and line 395 NOT-DO #2: *"未 launch V14 / V9 / V21 任何变体"*. These look like binding constraints against V14. But contextually:
- Phase A v3 was written when slot 1 = V18 was running, slot 2 = V13 was about to launch, and codex was given a narrow scope ("don't get clever, just do V13").
- The "Slot 3 永远 0" applies to *the Phase A v3 task itself*, not to all future Stage C work.
- Phase A v3 line 5: *"supersedes ... CODEX_TASK_PHASE_A_v2"* — i.e. it can itself be superseded by a later task md.

So this is **not an absolute constraint against V14**. It is a scoping constraint for Phase A v3 codex execution. But a clean V14 launch **must explicitly state** it supersedes Phase A v3 §6 NOT-DO #2. Otherwise codex (or future-you) will read both and freeze.

### 0.3 yaml differential snapshot (V13, V14, A3)

| field | V13 (image_aux off) | V14 (seed twin) | A3 (capacity-only) |
|---|---|---|---|
| `seed` | 42 | **1337** | 42 (V18 default) |
| `resume_from` | (none → from-scratch) | (none → from-scratch) | V7 best.pt |
| `training.max_steps` | 160000 | 160000 | 170000 |
| `training.image_aux.enabled` | **false** | true (V7 default) | true (V7 default) |
| `training.image_aux.lambda_max` | 0.0 | 0.04 | 0.04 |
| `training.decoder_lora` | n/a (V7-style) | n/a (V7-style) | rank=32, last_n_blocks=2 |
| `loss.decoder_kl_pullback.lambda_kl` | n/a | n/a | **0.0** (capacity-only) |
| `data.latent_dir` | shared | shared | shared |
| `data.num_workers` | 0 | 0 | 0 |
| `data.batch_size` | 8 | 8 | 8 |
| ETA | ~7 days | ~7 days | ~24-48h |

`num_workers=0` is the critical concurrency point — see §2 below.

### 0.4 V14 task md gap — confirmed

`find review -iname "*v14*"` returns **only** the yaml file at `review/0516/V14_true_d_pure/V14_v7_seed1337.yaml`. No task md, no smoke procedure, no NOT-DO, no commit-message template. Phase A v3 covers V13; A3 task md covers V18-capacity-only. V14 launch has zero codex guidance.

---

## 1. Q1 — V14 ROI: paper-necessary or anchor?

**Verdict: APPROVE V14 — but on its scientific merits, not on the prompt's framing.**

### 1.1 The prompt's framing is corrupted by B55

The prompt presents Q1 as "is V14 worth the 7-day GPU cost / is claude reversing a user decision". Both halves of that framing are bad:
- The "user decision" doesn't exist (B55).
- The "7-day GPU cost" is presented as opportunity cost. But slot 3 is empty regardless of V14 — there is no alternative experiment yaml prepped for that slot. The actual choice is "V14 or idle slot 3", not "V14 or X". Sunk-cost framing applies to the *latter* (skipping V14 to keep slot 3 'open' for unknown future X), not the former.

### 1.2 Scientific merit: d_pure noise floor is genuinely paper-necessary

This is the core question. The argument is:
- V18 vs V7 best-vs-best NORMAL Δ = **+0.030 dB**
- V18 vs V7 last-vs-best NORMAL Δ = **+0.062 dB**
- Without a noise-floor estimate of "V7 vs V7'" (same model, different seed), you have **no Type-I-error denominator**. A reviewer (any AI or human) reading "V18 beats V7 by 0.030 dB" can legitimately ask: "is that bigger than two independently-seeded V7 runs differ from each other?"

If V7@seed=42 vs V7@seed=1337 NORMAL Δ ≈ ±0.02 dB → V18 +0.030 dB is at noise edge, *barely* defensible
If V7@seed=42 vs V7@seed=1337 NORMAL Δ ≈ ±0.05 dB → V18 +0.030 dB is **inside noise**, paper claim collapses
If V7@seed=42 vs V7@seed=1337 NORMAL Δ ≈ ±0.005 dB → V18 +0.030 dB is meaningful

V14 is **the** prerequisite for the significance claim. It is not "nice to have"; it is the denominator. Calling it "multi-seed" in the prompt mislabels it as a robustness check. It is actually **the noise upper-bound estimate for every other paper number**.

### 1.3 Counter-argument: "wait for A3 outcome before committing 7 days"

The prompt's reverse argument is: if A3 (~48h) reveals V18 is mechanism-broken (capacity-only ≈ V18), then the V18 framework retires, and V14 became unnecessary. This is reasonable on its face but has a flaw:
- Even if V18 retires, **V7 is still the paper baseline**. Any future direction (V19, transport intervention, pivot) still needs to claim "we beat V7 by X". That claim still needs V14 as a noise denominator.
- V14 is upstream of *any* V7-baselined comparison, not just V18.
- 7 days is therefore not "V14 might be wasted on V18 framework retirement" — V14 outputs are framework-invariant.

→ V14 is genuinely paper-necessary regardless of A3 outcome. APPROVE.

### 1.4 But: do not launch V14 *simultaneously* with V13

This is where I differ from claude's "all 3 simultaneous" framing. Reason in §2 below.

---

## 2. Q2 — 3-slot concurrency risk

**Verdict: MODIFY — A3 + V13 simultaneous is fine; V14 should wait for V13 smoke pass.**

### 2.1 GPU memory: requires verification I cannot do remotely

The prompt asserts slot 1/2/3 are all empty and each can hold ~24h × full GPU. I cannot grep this — it requires `nvidia-smi`. Pre-launch verification must include: `nvidia-smi --query-gpu=index,memory.free --format=csv` returning ≥3 GPUs each with ≥12 GB free.

### 2.2 Dataloader concurrency: low risk by yaml design

All three yamls share `data.latent_dir: /data_2/qujiaxiang/lowdose_pet_ct/latents_224` and `data.num_workers: 0`. `num_workers=0` means each training process does its own foreground IO (no shared `DataLoader` worker subprocesses), so there is **no shared-memory contention** between V13/V14/A3. Three independent processes reading the same on-disk latents directory will contend at the kernel page cache level, but that's fine.

**However**: V13 and V14 are both from-scratch on the *exact same dataset directory*, with `batch_size=8` and 160K steps. The disk IO patterns will overlap heavily. If `/data_2` is a single physical disk (likely), three concurrent training jobs reading it could saturate IO bandwidth. The prior precedent is V18 + (idle) — never 3 concurrent.

**Recommendation**: launch A3 + V13 first, observe disk-IO + GPU utilization for 30 minutes, then launch V14 if no IO bottleneck. This is cheaper than launching all 3 simultaneously and discovering an IO-induced 30%+ training slowdown after a day.

### 2.3 Watchdog / metrics jsonl: clean

Each yaml has distinct `output_dir`. No file collision risk. ✓

### 2.4 Historical precedent: ≥3 concurrent training never tested

The prompt §2 Q2 asks: "服务器是否 ever 同时跑 ≥ 3 任务?". My grep didn't find an explicit log of this. Standing rule #1 says "≤ 3 并行训练任务", so the *policy* permits it, but I have no evidence the *system* has been stress-tested at 3. Recommendation in §2.2 (staged launch with mid-launch IO check) is the safer execution path.

---

## 3. Q3 — Launch order

**Verdict: MODIFY — staged, not simultaneous.**

| timing | action | rationale |
|---|---|---|
| T=0 | Launch **A3** on slot 1 (~48h) | Lowest scientific risk; smallest training delta (LoRA on top of V7); A3 task md is reviewed and ready |
| T=0 | Launch **V13** on slot 2 (Phase A v3 Task C+D, ~7 days) | Phase A v3 task md is committed (7486776). V13 has highest single scientific value (isolates image_aux confound) |
| T=0+30 min | Verify A3+V13 GPU + disk-IO healthy via `nvidia-smi` + `iostat 1 5` | If healthy, proceed. If saturated, **postpone V14** to after V13 smoke pass (~12h) |
| T=0+30 min (if IO healthy) OR T=12h | Launch **V14** on slot 3 | Requires task md (see Q4) and explicit Phase A v3 §6 NOT-DO #2 override note |
| T=48h | A3 returns → start interpretation (separate task) | Independent of V13/V14 7-day window |
| T=7 days | V13 + V14 return ≈ same time → paper-window starts | Both inputs available simultaneously |

Rejected alternative: "A3 first, V13/V14 wait 48h" — wastes 48h × 2 slots for no information gain. A3 outcome doesn't change V13 design (image_aux ablation is needed regardless) or V14 design (noise floor is needed regardless).

Rejected alternative: "all 3 simultaneous at T=0" — IO contention is real risk (§2.2). Costless to stage by 30 min.

---

## 4. Q4 — V14 task md prep

**Verdict: REQUIRED. ~15 min cost. Cannot skip.**

### 4.1 What V14 task md must contain

Reusing Phase A v3 template, V14 task md needs:

1. **Explicit Phase A v3 supersede**: "This task supersedes Phase A v3 §6 NOT-DO #2 (`未 launch V14`) — V14 is now launched as Stage C v3 slot 3 per R13 INTEGRATION §3.3 + R15 reviewer consensus."
2. **smoke procedure**: same 200-step smoke + md5 yaml integrity check + grep `lambda_kl|seed` in effective config to confirm seed=1337 was actually picked up
3. **launch command**: `python train_first_hop.py --config review/0516/V14_true_d_pure/V14_v7_seed1337.yaml` (no `--resume` — from-scratch)
4. **pass criteria**: process alive at +5 min, GPU memory >12 GB, log shows `seed=1337` in effective config, log shows no resume (V14 is from-scratch)
5. **NOT-DO**: don't launch V9/V21/V19/V18 variants; don't modify V14 yaml; don't `cp` ckpts; don't claim "V14 healthy/stable" before 1K steps (B23)
6. **commit & push template**

### 4.2 Why "codex just runs python --config" is NOT enough

The prompt §2 Q4 raises this as an option. Rejected. Reasons:
- **B14 (yaml cross-check)** — V14 yaml was written 2026-05-16, before the V13/A3 yaml hygiene improvements. Without a task md grep step ("verify image_aux still ON, seed actually 1337"), codex might launch with a corrupted resume of someone's overrides.
- **B23 (premature claim)** — without explicit "don't claim healthy" NOT-DO, codex commit message might say "V14 launched, training healthy" after 5 min, which is a 7-day extrapolation from 5 min of data.
- **Phase A v3 §6 NOT-DO #2 conflict** — codex reading both Phase A v3 (forbids V14 launch) and a bare `python --config` instruction will pause to ask. The supersede must be explicit and in writing.

→ V14 task md = **mandatory**, ~15 min cost. Trivial relative to 7-day training.

---

## 5. Q5 — Missing control experiments?

**Verdict: APPROVE current 3 controls (A3 + V13 + V14). Defer the additional controls below to Round 16+.**

The prompt suggests 4 candidates for additional controls. My evaluation:

| candidate | priority now | rationale |
|---|---|---|
| **V13-capacity-only** (V13 + LoRA rank=32, no KL) | **DEFER to R16** | Only relevant if V13 shows image_aux is dominant AND A3 shows LoRA capacity is real. Wait for both. |
| **V8 re-train with V7 train config** | **DEFER to R16** | V13 is the cleaner isolation of image_aux. V8 retrain is post-hoc cleanup. |
| **V18-clean** (`use_pred_latent=false`) | **DEFER to R16** | R13 INTEGRATION §3.2 already shows A3 disambiguates this. V18-clean is only worth it if A3 outcome 1 (capacity dominant) lands AND user wants to confirm KL on z_GT path would have helped. |
| **V19** (LoRA blocks=[0,1] vs [6,7]) | **DEFER to R17+** | Cannot design without knowing V18 outcome first. Premature. |

None of these is more valuable *right now* than A3 + V13 + V14. The current 3-slot plan is the highest-EV configuration available without first seeing A3 results.

---

## 6. Q6 — New biases (Round 15)

**5 new biases identified. B55 is HIGH; B56-B59 are LOW/MOD.**

### B55 (HIGH) — Fabricated R13 user quote anchors the entire Q1/Q6 framing

Detailed in §0.1 above. Grep verified: the quote *"user Round 13 明确说 '不考虑 multi-seed, 先跑出结果, 再考虑 multi-seed'"* appears nowhere in the repo except inside the R15 prompt itself. R13 INTEGRATION §3.3 and §6 both have V14 as part of the recommended 3-slot plan. The "user decision reversal" tension in the prompt is therefore fabricated.

**Impact**: Q1 framing collapses (V14 is not a reversal); Q6's first bullet ("user 原 Round 13 决策 推翻太快") is non-sequitur; standing constraint #5 modification ("R13 '不跑 multi-seed' 是 contextual 还是 absolute?") is moot.

**Fix**: prompt should either (a) cite the actual user quote with date/source, or (b) drop this framing entirely and present V14 ROI on its own merits as in §1.2 above.

### B56 (MODERATE) — "Slot 3 永远 0" Phase A v3 line 9 conflicts with R15 plan

Phase A v3 line 9 says "Slot 3 永远 0", line 395 NOT-DO #2 says "未 launch V14". These are *scoped to Phase A v3 codex execution* but read like absolute prohibitions. V14 launch in R15 must explicitly supersede them.

**Fix**: V14 task md §0 must state: "This task supersedes Phase A v3 line 9 and §6 NOT-DO #2 per R13 INTEGRATION §3.3 + R15 reviewer consensus."

### B57 (MODERATE) — "机会主义" framing for V14 is wrong category

The prompt §1.4 calls V14 "机会主义 slot 3 加跑". This frames V14 as opportunistic capacity-filling. It is not. V14 is the noise floor for **every** Δ-PSNR claim in the project; it is upstream of A3 / V18 / V13 / V19 / any future pivot. Mislabeling it as opportunistic invites future reviewers to dismiss it.

**Fix**: rename in all downstream docs: "V14 noise-floor experiment (paper prerequisite)", not "V14 multi-seed (opportunistic)".

### B58 (LOW) — "3 slot 同时 launch" framing ignores IO contention

The prompt §2.2 lists "服务器 GPU 内存是否真够" but skips disk IO. With three concurrent from-scratch trainings reading the same `latent_dir` and `num_workers=0`, IO bandwidth is the more likely bottleneck. Staged launch (§3) addresses this.

**Fix**: staged launch order + 30-min IO check before launching slot 3.

### B59 (LOW) — Q6 self-criticism is incomplete

The prompt §2 Q6 asks reviewer to check for sunk-cost / paper-anchor / filling-capacity bias. Good. But Q6 *does not* ask to verify the R13 user quote it depends on. The self-criticism is one level shallow.

**Fix**: prompt-template-level: when citing a prior-round user quote, the prompt should grep verify it itself and link to source line/commit.

---

## 7. Per §4 output format

### 7.1 Per-question

| Q | verdict | reason |
|---|---|---|
| Q1 V14 ROI | **APPROVE on merits, REJECT prompt framing** | V14 is paper-necessary noise floor; B55 invalidates the "reversal" framing |
| Q2 3-slot concurrency | **MODIFY** | GPU memory likely OK but IO contention untested; staged launch in §3 |
| Q3 Launch order | **MODIFY** | Staged: A3+V13 at T=0, V14 at T=30min (after IO check) |
| Q4 V14 task md | **REQUIRED** | ~15 min prep; cannot skip per B14/B23/B56 |
| Q5 Missing controls | **APPROVE current 3** | V13-cap-only / V8-retrain / V18-clean / V19 all defer to R16+ |
| Q6 New biases | **5 new, B55 HIGH** | See §6 |

### 7.2 Slot 3 decision verdict

**B** (slot 3 = V14), but on different rationale than claude's:
- Not "opportunistic"
- Not "filling capacity"
- Not "reversal of user decision" (B55: that decision never existed)
- Because: V14 is the noise floor for every Δ-PSNR claim and is upstream of all downstream paper math.

### 7.3 Launch order verdict

**Staged, not simultaneous**:
- T=0: A3 (slot 1) + V13 (slot 2)
- T=0+30 min: IO health check
- T=0+30 min (if IO healthy) or T=12h (after V13 smoke): V14 (slot 3)

### 7.4 V14 task md prep

**REQUIRED** (~15 min). Reuse Phase A v3 template. Must explicitly supersede Phase A v3 line 9 + §6 NOT-DO #2. Include: smoke procedure, pass criteria, NOT-DO list, commit template.

### 7.5 New biases (B55-B59)

5 new, summarized §6 above. **B55 (HIGH)** is the headline: prompt §1.4 fabricates a R13 user quote with zero artifact support.

---

## 8. One-paragraph executive summary

The R15 prompt frames V14 as an opportunistic slot-3 fill that may reverse a user's Round 13 decision against multi-seed runs. Grep verification shows that "user decision" doesn't exist anywhere in the artifact tree — R13 INTEGRATION §3.3 and §6 both have V14 in the recommended 3-slot plan, and the multi-seed quote appears only in the R15 prompt itself (B55, HIGH severity). Once that fabricated framing is removed, V14 stands on its own merits: it is the noise floor for every Δ-PSNR significance claim in the paper, upstream of A3/V13/V18/any pivot. Slot 3 = V14: APPROVE. But launch should be staged (A3+V13 first, IO health check at +30 min, V14 after) rather than all-3-simultaneous, because three concurrent trainings reading the same `latent_dir` with `num_workers=0` has never been stress-tested and IO contention is the unmeasured risk. V14 also requires a ~15-minute launch task md to (a) explicitly supersede Phase A v3 line 9 + §6 NOT-DO #2, (b) carry B14/B23 NOT-DO clauses, and (c) give codex unambiguous instructions. The Round 15 prompt is otherwise sound; biggest take-away is that prompts citing prior-round user quotes should grep-verify those quotes before dispatch (proposed standing rule extension of B36).

---

## 9. What I did NOT review (per prompt §0 boundary)

- A3 / V13 / V14 yaml internal design (signed in R7/R8/R13/R14)
- Round 13/14 standing rules already in design_rationale §5.2
- audit DRAFT release timing
- R12/R13 substrate verification (already verified by previous reviewers)
