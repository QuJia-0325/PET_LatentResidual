# Peer Review Round 15 — Review D Audit

- date: 2026-05-18
- reviewer: **Review D (GitHub Copilot)**
- scope: 3-slot launch decision for A3 + V13 + V14
- prompt: `PEER_REVIEW_PROMPT_round15_3slot_launch_20260518.md`
- verdict summary: **MODIFY, then APPROVE B**

## Executive Verdict

**Slot 3 verdict: B = run V14**, but only after creating a small V14 launch task / 3-slot wrapper that explicitly supersedes the stale Phase A v3 constraints saying "slot 3 empty" and "do not launch V14".

My recommendation is: **launch A3 + V13 + V14 in the same launch window, but operationally stagger them with per-slot preflight and +5min health checks**. Do not wait 48h for A3 before starting V14, because V14 is not only a V18-paper accessory; it is also the d_pure noise floor needed to interpret V7-vs-V13 and any small PSNR deltas under the V7 configuration.

However, Claude's current argument for B is slightly contaminated by "empty slot = waste" framing. The correct argument is not GPU utilization. The correct argument is: V14 answers an already-identified measurement-integrity gap from the V7/V6_NOISE mislabeling, costs no new model design risk, and its result remains useful even if V18 is retired after A3.

## Evidence Read

- A3 has isolated `output_dir`, `run_name`, `resume_from=V7 best.pt`, `max_steps=170000`, `num_workers=0`, `require_fresh_output_dir=true`, and `metrics_jsonl_mode=write`: `review/0517/V18_capacity_only/V18_capacity_only.yaml`.
- V13 has isolated `output_dir`, `run_name`, `seed=42`, `max_steps=160000`, `num_workers=0`, `freeze_rae=true`, `require_fresh_output_dir=true`, and `metrics_jsonl_mode=write`: `review/0516/V13_true_image_aux_ablation/V13_true_image_aux_off.yaml`.
- V14 has isolated `output_dir`, `run_name`, `seed=1337`, `max_steps=160000`, `num_workers=0`, `freeze_rae=true`, `require_fresh_output_dir=true`, and `metrics_jsonl_mode=write`: `review/0516/V14_true_d_pure/V14_v7_seed1337.yaml`.
- V13 Phase A v3 task still says `Slot 3 永远 0`, `未 launch V14`, and commit message says `slot 3 空`: `review/0517/CODEX_TASK_PHASE_A_v3_20260518.md`.
- Round 13 integration already identified V13 + V14 + capacity-only as the high-information control set, but Round 14 split execution docs and left V14 without a launch task: `review/0517/REVIEW_INTEGRATION_round13_20260518.md`, `review/0517/REVIEW_INTEGRATION_round14_20260518.md`.

## Q1 — V14 Slot 3 ROI: MODIFY

V14 is **more than nice-to-have**, but the prompt overstates it as "paper必备" in a way that anchors too early to a V18 paper path.

The stronger ROI case is:

1. V14 repairs a known measurement gap: historical V7-vs-V6_NOISE was not pure d_pure because it also changed step weights.
2. V14 is needed to decide whether V7-vs-V13 is a real image_aux effect or within seed noise.
3. V14 remains useful even if A3 causes V18 to be retired, because the V7/V13 ablation story still needs a noise floor.
4. V14 has low design risk because it is V7 with only `seed: 1337` changed.

The weaker / biased ROI case is:

1. "Slot 3 would otherwise be wasted" is not valid scientific reasoning. A 7-day run is still a real opportunity cost.
2. "Paper必备" is premature until A3 decides whether V18 has a credible mechanism.
3. Round 13's "不考虑 multi-seed" should not be silently overwritten; it needs an explicit statement that the new launch decision is a contextual revision driven by the control-set gap, not by capacity filling.

**Q1 verdict: MODIFY.** Run V14, but document it as a control/noise-floor run for V7/V13 interpretability, not as a paper-claim dependency for V18.

## Q2 — 3-Slot Concurrency: MODIFY

The artifacts suggest no obvious file-level or dataloader collision:

1. All three runs have distinct output directories and run names.
2. V13 and V14 read the same latent dataset, but `num_workers: 0` means no dataloader multiprocessing storm from those YAMLs.
3. `metrics_jsonl_mode: write` plus distinct output dirs should prevent metrics pollution.
4. `require_fresh_output_dir: true` protects against accidental reuse.

The remaining risks are operational, not YAML-level:

1. All YAMLs say `device: cuda:0`; this is acceptable only if each process is launched under a distinct `CUDA_VISIBLE_DEVICES=<physical_gpu>`, so each process sees its assigned GPU as `cuda:0`.
2. A3 has `freeze_rae: false` and decoder LoRA enabled, so its memory profile may be higher than V13/V14 even if still likely manageable.
3. There is no evidence in the provided artifacts that this server has successfully run three full training jobs at once.
4. Shared disk reads from the same latent directory can still create I/O contention even with `num_workers: 0`; this is a slowdown risk, not a correctness blocker.

**Q2 verdict: MODIFY.** Allow 3 slots only with a preflight table recording chosen physical GPU IDs, free memory before launch, distinct logs, and +5min checks after each launch. If any run fails the +5min check or memory is unexpectedly tight, kill only the newest run and preserve the failure log.

## Q3 — Launch Order: APPROVE With Staggered Execution

I recommend **same launch window, staggered validation**:

1. Launch A3 first; validate alive/GPU/log at +5min.
2. Launch V13; validate alive/GPU/log at +5min.
3. Launch V14; validate alive/GPU/log at +5min.

This preserves the wall-clock benefit without pretending that three `nohup` commands fired at the same second is safer. Waiting 48h for A3 before V14 is too conservative because V14's value is not conditional solely on V18. V13 should not wait for A3 either; Round 14 already concluded V13 has independent value.

**Q3 verdict: APPROVE same-window launch, but operationally staggered.**

## Q4 — V14 Task MD Gap: REJECT Direct Launch

Direct `python train_first_hop.py --config V14_v7_seed1337.yaml` would probably run technically, but it is not acceptable for this repo's current process discipline.

The blocker is not CLI support. The blocker is auditability:

1. V13 Phase A v3 explicitly says slot 3 is empty and V14 must not be launched.
2. V14 has no smoke/pass criteria, no NOT-DO list, no log naming convention, and no commit/report policy.
3. This project has repeated B-series failures from stale docs, overclaiming, and unverified launch assumptions; V14's missing task md is exactly the kind of gap those rules are meant to catch.

**Q4 verdict: V14 task md is required.** A 10-minute task is enough. It should reuse Phase A v3's launch template but be much smaller:

- preflight: YAML exists, CLI flags are `--config --resume`, output_dir/run_name/seed/max_steps verified, output_dir fresh, selected GPU recorded.
- launch: no `--resume`, distinct `CUDA_VISIBLE_DEVICES`, log path under `review/0516/V14_true_d_pure/`.
- +5min pass: process alive, GPU memory >1GB, no OOM/CUDA error, log/metrics started, no resume string.
- NOT-DO: do not modify V14 YAML, do not touch V13/A3/V18, do not claim convergence after +5min, do not add rolling 7-day logs to git.
- explicit supersession: this Round 15 task supersedes the older Phase A v3 lines that said slot 3 remains empty / V14 not launched.

## Q5 — Missing Controls: MODIFY, Not Blockers

The proposed A3 + V13 + V14 set covers the current highest-value disambiguation axes:

1. A3 tests whether V18 GT-manifold drift is KL-related or LoRA-capacity-related.
2. V13 tests true image_aux contribution under V7 settings.
3. V14 measures pure seed noise under V7 settings.

I do not recommend replacing V14 with V13-capacity-only, V18-clean, or V19 for slot 3 now.

- **V13-capacity-only** is a second-wave experiment. It mixes the image_aux question with decoder adaptation and should be designed after V13/A3 return.
- **V8 retrain with V7 train config / no image_aux / V6 step weights** may become important if V13 shows the V7-V8 gap was mostly step-weight/config-driven, but it does not replace d_pure.
- **V18-clean (`use_pred_latent=false`)** should stay retired until A3 shows KL has useful indirect effect. If A3 says capacity-only explains the gain, V18-clean is low ROI.
- **V19 layer/location sweeps** are architecture search and should wait for A3 to show decoder LoRA has durable value.

**Q5 verdict: MODIFY.** Add these as Round 16 candidate controls, but do not use them to block V14.

## Q6 — New Biases B55+

I would log the following Round 15 biases:

- **B55 Slot-fill bias**: treating an empty GPU slot as a loss. Fix: justify V14 by information value, not utilization.
- **B56 Paper-anchor bias**: calling d_pure "paper必备" before A3 validates a paper-worthy V18 mechanism. Fix: describe V14 as a control/noise-floor run for V7/V13 and future claims.
- **B57 Decision-reversal blur**: reversing "不考虑 multi-seed" without making the contextual revision explicit. Fix: state that Round 15 supersedes that tactical preference because the V7/V13 interpretation now needs a pure seed-noise control.
- **B58 Launch-doc contradiction**: V13 Phase A v3 still says slot 3 empty / no V14 while Round 15 wants V14. Fix: create a V14 task or 3-slot wrapper that explicitly supersedes those lines.
- **B59 Concurrency optimism**: assuming 3-slot training is safe without server history. Fix: require per-slot preflight and +5min health evidence.

## Final Decisions

| Item | Review D decision |
|---|---|
| 6-question overall | **MODIFY** |
| Slot 3 | **B: V14** |
| Launch order | **All three in same launch window, staggered by +5min validation** |
| V14 task md | **Required; direct launch rejected** |
| Missing controls | **Track for Round 16, not blockers for V14** |
| New bias | **B55-B59 logged above** |

## Minimal Action Before Launch

Before asking Codex to occupy all three slots, create one small file, e.g. `CODEX_TASK_V14_D_PURE_LAUNCH_20260518.md`, or a single `CODEX_TASK_ROUND15_3SLOT_LAUNCH_20260518.md` wrapper. It must explicitly state:

1. Round 15 supersedes Phase A v3's `slot 3 空` / `未 launch V14` constraints for this launch only.
2. A3, V13, and V14 each get a distinct physical GPU through `CUDA_VISIBLE_DEVICES`.
3. V14 is launched without `--resume`.
4. V14's +5min status is not a convergence or health claim beyond process/GPU/log sanity.
5. If three-way concurrency fails, preserve logs and downgrade to A3 + V13, not silent retry.

After that file exists, Review D approves **B**.