# Codex 0525 X1-lite + X3 Design Analysis

Date: 2026-05-25
Branch: foc_lite_hop0
Scope: Round 18 X1-lite + X3 experiment design review
Author: Codex operator

## Executive Verdict

The 0525 design is reasonable and executable. It should be understood as a mechanism-attribution and project-closure experiment, not as a new architecture breakthrough.

| Run | Question | Verdict |
|---|---|---|
| X1-lite | Is A4-mid mainly explained by pixel L1 through the frozen decoder, rather than SSIM/seam? | Necessary. This is the minimum mechanism falsification run. |
| X3 | Does decoder LoRA still add value when image_aux is strengthened to lambda=0.08 and KL is disabled? | Reasonable. It tests short-window additivity and whether V18-family work should remain in the paper. |

## Why This Design Is Reasonable

X1-lite is the right next mechanism experiment. A4-mid gave the largest single improvement so far:

| Run | image_aux lambda | NORMAL PSNR_clip3 | Delta vs V7 |
|---|---:|---:|---:|
| V7 | 0.04 | 36.7810 | 0 |
| A4-mid | 0.08 | 36.8939 | +0.113 |

Without X1-lite, the paper story risks looking like loss-weight tuning. X1-lite holds lambda=0.08 fixed and removes SSIM/seam weights, so it tests whether the dominant gain can be explained by pixel L1 alone. This is a mechanism control, not a lambda sweep.

X3 is a bounded V18-family decision test. It is not intended to revive V18 as the main path. It tests whether LoRA has marginal value under the stronger image_aux regime.

## Execution Protocol Assessment

The latest 0525 task correctly handles previous high-risk launcher bugs:

- X3 passes CLI --resume explicitly; YAML training.resume_from alone is not enough.
- X1 writes PID/GPU sidecars instead of reading PID from trainer logs.
- X3 excludes X1 GPU in the documented auto-selection path.
- X3 sets decoder_kl_pullback.enabled=false when lambda_kl=0.
- X3 uses save_interval=5000 to match the A3-style checkpoint cadence.

Current execution note: the operator launched the two runs in parallel per user instruction, rather than following the task document staggered +30min launch suggestion.

| Run | GPU | Session | Expected behavior |
|---|---:|---|---|
| X1-lite | GPU1 | round18_x1_gpu1_0525 | from scratch to 160k |
| X3 | GPU3 | round18_x3_gpu3_0525 | resume from V7 best at step 160k, train to 170k |

Early health checks showed:

- X1: lambda_img=0.08, lambda_kl=0, and img == img_l1, so the L1-only ablation is active.
- X3: log confirms resume loaded step=160000, so it is not training from scratch.
- Both jobs entered GPU training without OOM.

## Main Limitations

1. X1-lite is not a perfectly clean mechanism decomposition.

Setting ssim_weight=0 and seam_weight=0 removes SSIM/seam terms, but it also changes the total image_aux gradient scale and gradient spectrum. If X1-lite is below A4-mid, we cannot immediately conclude that SSIM/seam are mechanistically essential. The drop may partly reflect reduced total supervision strength.

2. X1-lite cannot separate M1 from M4.

X1-lite can test whether L1 through the decoder is enough, but it cannot distinguish pure latent-transport undertraining from decoder-manifold anchoring. If X1-lite succeeds, the honest claim is that pixel-space L1 supervision through the frozen decoder is sufficient to explain most of the A4-mid gain.

3. X3 only tests short-window LoRA additivity.

X3 follows an A3-style 10k adaptation window from V7 best. If it does not exceed A4-mid, it rules out short-window additivity, not all possible long-training LoRA benefits. A negative X3 result should demote V18-family work for this paper, but should not be written as universal proof that decoder LoRA is useless.

4. Evidence remains single-dataset and slice-level.

Without patient IDs or external datasets, the paper should avoid patient-level statistical claims. Recommended wording: We report slice-level paired statistics on the held-out validation set.

## Architecture-Level Concern

The biggest architecture signal from A4-mid is not that LoRA is the missing piece. The signal is that pure latent transport loss is poorly aligned with the final decoded image metric.

In other words, the transport model may have enough capacity, but its training objective is not decoder-aware enough.

The current image_aux implementation solves this by backpropagating pixel-space loss through a frozen decoder. This is effective, but it is still a post-hoc auxiliary objective.

Future architecture directions worth considering after paper closure:

| Direction | Rationale | Current-paper priority |
|---|---|---|
| Decoder-aware latent metric or pullback loss | Make latent loss reflect decoder sensitivity | High future direction |
| Multi-hop decoder-aware chain supervision | Current chain quality depends on repeated first-hop behavior | Medium |
| Dose-conditioned transport | Current transport may under-model dose-specific transitions | Medium |
| Full architecture pivot | Could produce larger gains but reopens the whole substrate | Not recommended now |

For the current paper, the safer narrative is not: we invented a new architecture. The safer narrative is: PET latent transport needs decoder-aware pixel supervision, and a simple frozen-decoder image loss provides the dominant improvement over pure latent objectives.

## Interpretation Rules

### X1-lite

| Outcome | Interpretation |
|---|---|
| Within 0.02 dB of A4-mid | L1 through frozen decoder explains most of the gain. |
| 0.02-0.05 dB below A4-mid | L1 contributes, but SSIM/seam or total gradient budget may also matter. |
| Near V7 or below | L1-only is insufficient; further decomposition may be needed. |

### X3

| Outcome | Interpretation |
|---|---|
| Clearly above A4-mid | LoRA has additive value under strong image_aux. |
| Approximately equal to A4-mid | LoRA is redundant in the strong image_aux regime. |
| Below A4-mid | LoRA and strong image_aux may conflict; V18-family should be demoted. |

## Remote Confirmation

No blocking issue currently requires remote confirmation.

Non-blocking items for remote reviewers:

1. X1 and X3 were launched in parallel on GPU1/GPU3 per user instruction, rather than using staggered launch.
2. X1-lite is a mechanism falsification experiment, but not a perfectly normalized gradient-budget ablation.
3. X3 should be interpreted only as a 10k warmstart LoRA additivity test.
4. Logs may contain val_chain_*_raw_mse=nan; this is a raw-diagnostic field without a valid value, not a training loss NaN.

## Recommendation

Continue both current runs.

If X1-lite matches A4-mid, stop further image_aux component training and write the mechanism as: pixel L1 through frozen decoder is sufficient.

If X1-lite is inconclusive, ask the user before launching any X1-v2 component ablation. Do not automatically start ssim-only or seam-only runs.

If X3 does not improve over A4-mid, demote V18-family work to a secondary ablation and focus the paper on decoder-aware pixel supervision.
