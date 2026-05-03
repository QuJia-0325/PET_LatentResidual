# Operator Reply to Pre-C1 Local Questions - 2026-05-03

**Repo**: `/home/qujiaxiang/project/PET_LatentResidual`  
**Branch**: `foc_lite_hop0`  
**HEAD after pull**: `23a11d4 review(0503): draft REV1 §10 (round-2 multi-agent absorption) + pre-C1 operator questions`  
**Canonical remote on operator host**: `origin = git@gitee.com:jqu9/PET_LatentResidual.git`  
**Status sample time**: `2026-05-03 22:44 CST`

This file answers `review/0503/local/OPERATOR_QUESTIONS_pre_C1_20260503.md` Q9-Q15 from the GPU host side.

---

## Q9 - Sanity B current state + path

Answer: **(a)**, but B has not finished yet.

Current B sanity state at the last sample:

- Process: PID `803742`, GPU1, alive, command `/home/qujiaxiang/.conda/envs/rae/bin/python train_first_hop.py --config /home/qujiaxiang/project/PET_LatentResidual/review/0502/runs/B/config.resolved.yaml`
- Metrics path: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/B/run/first_hop_224_sigma_norm_B/metrics.jsonl`
- Rows: `422` total, `46` val rows
- Last train row: step `18800`
- Last val row: step `18400`
- Last val key numbers: `val_select_score=0.0018183143`, `val_chain_normal_mse=0.0005981120`, `val_rollout_total=0.0015367585`, `val_pair_total=1.5688423e-07`
- Sentinel: `review/0502/runs/.sanity_pass` does **not** exist yet

I copied the currently available B val trace to:

```text
review/0503/operator/B_sanity_metrics_val50_schema_reference.jsonl
```

Important: the file currently contains **46 val rows**, not 50, because B has not emitted step 18800/19200/19600/20000 val rows yet. It is still useful as a stronger schema/golden-trace fixture than the 5-row A tail. If the test must require exactly 50 rows, wait for B completion and refresh this file.

---

## Q10 - Canonical remote and `.review_canonical_remote`

Choose **(i) single-line URL or URL-fragment**, not a literal remote name.

Recommended tracked content:

```text
gitee.com:jqu9/PET_LatentResidual
```

The resolver should scan `git remote -v` and return the local remote name whose URL contains this fragment. On this operator host that resolves to `origin`. On the Mac/local side it may resolve to `gitee`. This avoids hardcoding either host's remote nickname.

Operational correction: any plan text that says `git push gitee foc_lite_hop0` or checks `gitee/foc_lite_hop0` is **not valid on this host** unless a new `gitee` remote is added. The portable wording should be `git push <canonical-remote> foc_lite_hop0` and `<canonical-remote>/foc_lite_hop0`.

---

## Q11 - `A_pair_uniform_spot` rerun scope

Choose **(a-with-cleanup)**.

Semantic change: `training.max_steps: 60000 -> 120000` so the warmup/ramp/Phase-III boundaries match A_main: `[0,30K] / [30K,90K] / [90K,120K]`.

No other training semantics should change. Keep the spot-specific purpose intact:

- `transport.pair_loss_weights: [1.0, 1.0, 1.0, 1.0]`
- `save_interval: 10000` is acceptable because it changes checkpoint density, not optimization dynamics
- cleanup stale comments that still claim 60K or ambiguous 200K schedule

Recommended new file: `review/0502/configs/A_pair_uniform_spot_aligned.yaml`.

---

## Q12 - A_main launch command

Use the existing ablation launcher, not a direct `train_first_hop.py --output-dir` command. `train_first_hop.py` currently only supports `--config` and `--resume`.

Canonical launch recipe:

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
PYTHON=/home/qujiaxiang/.conda/envs/rae/bin/python GPU=<free_gpu> \
  bash review/0502/scripts/run_ablation.sh A
```

Expected resolved outputs:

```text
review/0502/runs/A_main/config.resolved.yaml
review/0502/runs/A_main/train.log
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_main/run/first_hop_224_sigma_norm_A_main/metrics.jsonl
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_main/run/first_hop_224_sigma_norm_A_main/best.pt
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_main/run/first_hop_224_sigma_norm_A_main/last.pt
```

Do not launch A_main until the C1-C5d tooling fixes and R1a protocol skeleton are merged, unless the user explicitly accepts the audit risk.

---

## Q13 - Output-dir convention for `A_seed=43`

Use a **physical yaml in R1a**, but follow the current run-index/data-disk convention rather than an ad hoc dated output path.

Recommended convention:

- Config file: `review/0502/configs/A_seed43.yaml`
- Allowed semantic delta from `A_control.yaml`: `seed: 43`
- Allowed identity/path deltas: `run_name`, `output_dir`
- Run name: `first_hop_224_sigma_norm_A_seed43`
- Data output root if using a launcher tag `A_seed43`: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_seed43/run`
- Final run dir: `/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0502_runs/A_seed43/run/first_hop_224_sigma_norm_A_seed43/`

If local wants to force Q13's two-option framing, this is closest to **(i)**: concrete physical yaml, but with the project-standard `review_0502_runs/<tag>/run` layout instead of a date-stamped free-form path.

---

## Q14 - GPU availability and A_main launch timing

Current GPU status at the last sample:

| GPU | PET-related job | Memory | Utilization | Notes |
|---:|---|---:|---:|---|
| 0 | unrelated python job | ~21.6 GB | 100% | not available |
| 1 | B sanity | ~18.4 GB | sampled 0%, process CPU-active | not available until B finishes |
| 2 | unrelated job + A_sanity_light | ~40.1 GB | 100% | not available |
| 3 | V6.1 | ~18.4 GB | sampled 47% | not available |

A_main should start only after:

1. B sanity completes and writes/validates the sentinel, or local explicitly decides A_main may start before the sentinel.
2. C1-C5d and R1a are merged.
3. One A6000 is free enough to avoid host-memory/GPU-memory contention.

Given B is near 20K but not complete, the earliest safe launch is after B finishes plus the tooling/protocol patches land. I would not commit to a clock time; operationally it is "after B sentinel + R1a, on the first free GPU".

---

## Q15 - Conference / paper deadline anchor

No hard external conference or paper deadline is known from the operator host/context. Treat this as **no hard deadline** unless the PI/user provides a concrete date.

This supports keeping the aligned `A_pair_uniform_spot` rerun in scope if Risk 4 robustness remains useful.

---

## Additional Operator Flags on REV1 §10 Drafts

1. **Remote name wording must be fixed.** This host has only `origin`, and `origin` is the canonical Gitee remote. Any hard-coded `gitee/foc_lite_hop0` or `git push gitee ...` will fail here. Use the resolver from Q10.
2. **C1/T1 is still unpatched in code.** `lock_effect_size_threshold.py`, `paired_diff_judge.py`, and `select_best_ckpt_smoothed.py` still assume `global_step` in several places. Real trainer metrics use `step`.
3. **C3/T3 is still unpatched in code.** Current training writes `step_020000.pt`, `best.pt`, `last.pt`; `select_best_ckpt_smoothed.py` still searches primarily for `ckpt_step_*.pt` and `ckpt_last.pt`.
4. **`run_ablation.sh` help text is stale.** It still says data products include `ckpt_*.pt`; real products are `step_*.pt`, `best.pt`, and `last.pt`.
5. **B trace has 46 val rows, not 50.** If local's unit test wants exactly 50 val rows, wait for B completion before freezing the fixture.
6. **`best_metric` guard wording must distinguish config vs metrics.** Config-side `best_metric` is `val_multi_objective`; metrics-side computed key is `val_select_score`. A correct guard should verify both, not require config `best_metric == val_select_score`.
