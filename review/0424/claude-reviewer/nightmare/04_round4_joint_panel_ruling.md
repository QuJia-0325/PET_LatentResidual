# Round 4 — Joint Panel (E Gatekeeper + A Novelty + B Story) Ruling

Date: 2026-04-24  
Mode: NIGHTMARE, Round 4 / 5+  
Panel: E (Submission Gatekeeper) + A (Novelty Skeptic) + B (Story Critic)  
Task: Resolve Disputes D1–D4; Issue authoritative recommendation

---

## D1. Diagnostic-Framework-as-Contribution

### Panel Ruling: **SPLIT** (A: FOR-with-conditions; B: AGAINST; E: ABSTAIN)

| Aspect | Detail |
|--------|--------|
| **Core question** | Is A/B/C/D per-hop decomposition a genuine methodological contribution or last-ditch reframe? |
| **A's position (FOR)** | The decomposition CAN be a contribution IF (1) formalized mathematically, (2) shown to generalize beyond PET (any multi-hop flow), (3) published diagnostic findings are independently verifiable. Novelty 5/10 under these conditions. |
| **B's position (AGAINST)** | The "framework" is post-hoc rationalization of negative results. No evidence it generalizes. The A/B/C/D labels are arbitrary axis names, not a methodology. Without positive method results, this is writing exercise, not contribution. |
| **E's position (ABSTAIN)** | Venue-dependent. MIDL/workshop accepts "here's what we learned" papers. TMI/MICCAI main requires positive result. |

### Evidence Required to Finalize

1. **Cross-domain demonstration**: Apply A/B/C/D decomposition to at least ONE other multi-hop flow task (e.g., video prediction, multi-scale SR) and show it produces actionable diagnostic insights
2. **Formalization**: Write explicit equations for each diagnostic axis, not just prose labels
3. **Independent validation**: Someone other than Proposer can use the framework to diagnose a different model and reach non-trivial conclusions

### Severity: **MEDIUM**

Does not block submission to workshop venues. Blocks submission as main contribution to any venue.

### Effect on Final Score

- If D1 resolves FOR: Novelty +1.5 (3→4.5), Story +0.5 (3.5→4.0)
- If D1 resolves AGAINST: No change (current scores assume no diagnostic contribution)

---

## D2. Venue Path: Reframe vs. Pivot vs. Kill

### Panel Ruling: **FOR REFRAME** (unanimous)

| Option | EV Estimate | Timeline | Expected Tier |
|--------|-------------|----------|---------------|
| **(i) Reframe** as negative/analysis | **Highest** | 3–4 weeks | MIDL workshop (55%), MICCAI workshop (40%), NeurIPS ML4H workshop (35%) |
| (ii) Pivot | Medium | 8–12 weeks | MICCAI main (15%), TMI (10%), lower workshop (60%) |
| (iii) Kill | Zero | — | — |

### Rationale

1. **Reframe has lowest cost, highest probability of publication**: Current data already demonstrates a rigorous investigation. Honest "what doesn't work" papers are publishable at workshops.

2. **Pivot is high-risk**: 8–12 weeks of new development with no guarantee the pivot hypothesis (reduced hops / hybrid latent-pixel / partial decoder FT) will work. The 36.2 dB plateau may be intrinsic to the RAE encoder, not addressable by downstream transport changes.

3. **Kill is premature**: There IS publishable content here — a systematic negative result on first-hop injection mechanisms with honest analysis of the 10 dB oracle gap.

### Forced Recommendation

> **REFRAME immediately. Target MIDL 2027 workshop track (deadline ~January 2027) as primary, with MICCAI 2027 workshop as fallback.**

The paper should be titled something like: "Pixel Injection Does Not Break the First-Hop Bottleneck: A Systematic Analysis of Latent Flow Matching for PET Enhancement"

### Evidence Required

- Seed variance experiment (D3 resolution)
- Patient-level split verification (D4 resolution)
- Honest manuscript with "negative result" framing

### Severity: **HIGH**

Determines resource allocation for next 4–12 weeks.

### Effect on Final Score

- Reframe path: Submission +1.5 (2.5→4.0), Story +1.0 (3.5→4.5) if executed honestly
- Pivot path: No immediate score change; future potential varies
- Kill path: N/A

---

## D3. 200K Run: Falsification or Delay?

### Panel Ruling: **AGAINST** (kill 200K run; redirect to Exp 1)

| Aspect | Detail |
|--------|--------|
| **Panel's finding** | The 200K run is NOT a genuine falsification attempt. It is delay masquerading as rigor. |
| **Mathematical reasoning** | 5 configs span 36.123–36.206 dB (range = 0.083 dB). Proposer's stop criterion is "±0.05 dB of 50K result". This 0.10 dB tolerance window is LARGER than the observed config variance (0.083 dB). The criterion is almost certain to trigger "abandon" regardless of training dynamics — it tests nothing. |
| **Opportunity cost** | 200K run = 2.7 GPU-days. Exp 1 (seed variance) = 4 GPU-days but provides the SINGLE number (σ_seed) that determines whether ANY claim is publishable. |

### Ruling

> **Kill the 200K run NOW. Redirect all compute to Exp 1 (seed variance × clean ablation).**

### Justification

1. **σ_seed is the load-bearing unknown**: Without seed variance, C−N1=0.083 dB is uninterpretable. With σ_seed, we know whether the plateau is "real" or noise.
2. **200K tests the wrong hypothesis**: Even if plateau breaks by 0.3 dB at 200K, this proves "more compute helps" — not "hop0 mechanism helps". The hop0 contribution question requires ablation, not continuation.
3. **Face-saving is not science**: A principled negative result paper requires honest stopping, not "we tried everything including 4× training budget."

### Evidence Required to Reverse

If Proposer can provide:
1. Pre-registered hypothesis that 200K tests DIFFERENT from "plateau will break"
2. Explicit prediction (e.g., "hop0 gain will widen from 0.08 dB at 50K to >0.20 dB at 200K")
3. Analysis showing this prediction is NOT already falsified by current data

…then 200K could be reconsidered. But current justification is "maybe plateau breaks" — this is not a falsifiable hypothesis.

### Severity: **BLOCKER** (for resource allocation)

### Effect on Final Score

- If 200K killed → Causal +0.5 (2.0→2.5) for principled stopping
- If 200K runs and plateau holds → Causal unchanged, 2.7 GPU-days wasted
- If 200K runs and plateau breaks → Causal +0.3, but still need seed variance

---

## D4. Patient-Level Split (NEW RED FLAG)

### Panel Ruling: **BLOCKER UNTIL VERIFIED**

### Severity Assessment: **AUTOMATIC DESK-REJECT RISK**

| Aspect | Detail |
|--------|--------|
| **The bug** | Medical imaging venues (TMI, MedIA, MICCAI, MIDL, IPMI) require patient-level train/val/test splits. If slices from the same patient appear in both train and val, the model sees near-identical anatomy during training and validation → artificially inflated metrics → **DATA LEAKAGE** |
| **Known examples** | This is the #1 methodological failure mode in medical imaging ML. Notable retractions/criticisms include early COVID-19 X-ray papers (Roberts et al., 2021 "Common pitfalls"), many chest X-ray studies, and multiple skin lesion papers. Reviewers at medical venues are trained to check for this FIRST. |
| **Current evidence of leak** | NONE confirmed, but NONE ruled out either |

### Code Audit Findings

1. **PET_LatentResidual has ZERO patient/subject handling**:
   - Searched `pet_lr/data_first_hop.py`: no references to "patient", "case_id", "subject", "split_by"
   - Split is via `slice_idx` loaded from upstream `latents_train.pt` / `latents_val.pt`
   - [data_first_hop.py](pet_lr/data_first_hop.py) L52-130: split handled by filename (`latents_{split}.pt`), not by patient

2. **Upstream RAE has MIXED evidence**:
   - NPY format uses `train_subjects.txt` / `val_subjects.txt` → **patient-level split** ✓
   - PT format uses `preprocessed_data_*.pt[split]` → **split origin UNDOCUMENTED** ⚠️
   - [RAE/TECHNICAL_AUDIT_REPORT.md](../../../RAE/RAE/TECHNICAL_AUDIT_REPORT.md) L63-68: "380 subjects" + "Train: 26,247 slices, Val: 7,403 slices"
   - [RAE/src/pet_flow/train_pet_flow.py](../../../RAE/RAE/src/pet_flow/train_pet_flow.py) L310: "Training function for PT format dataset (simple slice-based)" ← **ALARMING**

3. **Possible scenarios**:
   - **Scenario A (OK)**: PT files were created from NPY files using `train_subjects.txt` → train slices and `val_subjects.txt` → val slices. Split is patient-level, just not documented downstream.
   - **Scenario B (CATASTROPHIC)**: PT files were created by random slice shuffle, ignoring subject boundaries. Split is slice-level → ALL RESULTS INVALID.

### Verification Commands

Run these on the data server to verify:

```bash
# Step 1: Count slices per subject in train vs val
cd /data_2/qujiaxiang/lowdose_pet_ct/suv

# List train subjects
cat train_subjects.txt | head -5

# List val subjects  
cat val_subjects.txt | head -5

# Count total train slices (should match 26,247)
for s in $(cat train_subjects.txt); do
  find subjects/$s -name "*.npy" -exec python -c "import numpy as np; print(np.load('{}').shape[0])" \;
done | awk '{s+=$1}END{print "Train slices:", s}'

# Count total val slices (should match 7,403)
for s in $(cat val_subjects.txt); do
  find subjects/$s -name "*.npy" -exec python -c "import numpy as np; print(np.load('{}').shape[0])" \;
done | awk '{s+=$1}END{print "Val slices:", s}'
```

```bash
# Step 2: Verify PT files match subject-split slice counts
cd /data_2/qujiaxiang/lowdose_pet_ct/preprocessed_pt

python -c "
import torch
d = torch.load('preprocessed_data_D50.pt', map_location='cpu')
print('Train slices in PT:', d['train']['x_T'].shape[0])
print('Val slices in PT:', d['val']['x_T'].shape[0])
"
# Expected: Train=26247, Val=7403 (matching NPY counts)
```

```bash
# Step 3: Trace PT file creation (find the preprocessing script)
grep -r "torch.save.*preprocessed" /path/to/RAE --include="*.py"
# Find the script that created preprocessed_data_*.pt and verify it uses train/val_subjects.txt
```

### Panel Ruling

| Question | Answer |
|----------|--------|
| How severe is this in practice? | **CRITICAL**. This is the single most common methodological failure in medical imaging ML. Any paper at TMI/MICCAI/MIDL will be scrutinized for this FIRST. |
| What specific verification would close this gap? | Run the 3-step verification above. If PT slice counts match subject-split slice counts AND the PT creation script uses `*_subjects.txt`, leak is ruled out. Otherwise, ALL RESULTS ARE INVALID. |
| If Proposer cannot document patient-level split, does this invalidate all current results? | **YES.** No amount of architectural contribution can save a paper with data leakage. This supersedes all other disputes. |

### Evidence Required to Finalize

1. **Written documentation** in PET_LatentResidual/CLAUDE.md or README explicitly stating: "Train/val split is patient-level. Train set contains slices from subjects in train_subjects.txt only. Val set contains slices from subjects in val_subjects.txt only. No subject has slices in both sets."

2. **Verification log** showing the 3-step commands above were run with expected outputs.

3. **Provenance chain**: Document the exact command used to create `preprocessed_data_*.pt` and `latents_{train,val}.pt`, showing they derive from `train_subjects.txt` / `val_subjects.txt`.

### Severity: **BLOCKER** (highest priority)

### Effect on Final Score

- If D4 verified OK: No score change (baseline assumes correct split)
- If D4 verification FAILS: All scores → 0/10, paper is dead, results invalidated, pivot to new dataset required

---

## Joint Recommendation (Authoritative)

### Must-Do-This-Week (BLOCKERS)

| # | Action | Effort | Blocking |
|---|--------|--------|----------|
| 1 | **Verify patient-level split (D4)** | 1 hour | ALL submission paths |
| 2 | **Kill 200K run, redirect to Exp 1** | 0 hours (decision) | Resource allocation |
| 3 | **Start Exp 1: 3-seed × 2-config** | 4 GPU-days | Any claim about hop0 effect |

### Must-Do-Before-Submission

| # | Action | Effort | Blocking |
|---|--------|--------|----------|
| 4 | Report σ_seed and CI on (C−N1) from Exp 1 | 1 day analysis | Statistical validity |
| 5 | Reframe narrative: "negative result" paper | 2-3 days writing | Story coherence |
| 6 | Add ≥1 external baseline (Pix2Pix or published PET-DDPM) | 5-7 days | Interpretability of 36.2 dB |
| 7 | Document 10 dB oracle gap as open problem, not failure | 1 day writing | Honest framing |
| 8 | Add dataset/ethics statement with explicit patient-split claim | 1 day | Any medical venue |

### Can-Defer-Post-Submission

| # | Action | Rationale |
|---|--------|-----------|
| 9 | Causal mediation analysis (Exp 2) | Nice-to-have for rebuttal; not required for workshop |
| 10 | Cross-domain diagnostic framework validation | Only if pursuing D1-as-contribution path |
| 11 | Ablation on individual modules | Workshop-level doesn't require factorial ablation |

### Explicit KILL List

| # | What to Stop | Reason |
|---|--------------|--------|
| K1 | 200K continuation run | Tests wrong hypothesis; delay tactic |
| K2 | New architectural experiments (CCT-224, velocity_rebalance variants) | Plateau is structural; architecture iteration is wasted |
| K3 | Scope expansion (SeamRefiner, SpatialAlignmentProjector) | Dead code should stay dead |
| K4 | Claims about hop0 "fixing" bottleneck | Evidence shows opposite; honest reframe required |
| K5 | Full conference paper targeting (MICCAI main, CVPR, NeurIPS main) | Current results don't support; workshop is ceiling |

---

## Updated Fused Score Forecast (Post-Round 4)

### Score Scenarios

| Dimension | Weight | R3 Score | R4 (D4 FAIL) | R4 (D4 OK, Exp1 σ>0.04) | R4 (D4 OK, Exp1 σ<0.04) |
|-----------|--------|----------|--------------|-------------------------|-------------------------|
| Novelty | 20% | 3.0 | 0 | 3.0 | 3.5 |
| Story | 25% | 3.5 | 0 | 3.0 | 4.0 |
| Implementation | 15% | 5.3 | 0 | 5.3 | 5.3 |
| Causal | 25% | 2.0 | 0 | 1.5 | 3.0 |
| Submission | 15% | 2.5 | 0 | 2.0 | 3.5 |
| **FUSED** | — | **3.15** | **0.00** | **2.74** | **3.73** |

### Interpretation

- **D4 FAIL (data leakage)**: Project is dead. Score = 0/10.
- **D4 OK but σ_seed > 0.04 dB**: C−N1 effect is null. Must reframe as pure negative result. Workshop viable at ~40%.
- **D4 OK and σ_seed < 0.04 dB**: C−N1 effect is marginal but real. Honest "small effect" paper at workshop viable at ~55%.

### Most Likely Outcome

Based on the pattern (5 configs → same plateau, effect size = 0.083 dB, no prior σ measurement), the most likely scenario is:

> **D4 OK, σ_seed ≈ 0.05–0.08 dB → C−N1 effect is null → pure negative result paper**

Expected fused score: **2.7–3.0 / 10**

This is publishable at MIDL/MICCAI workshop with honest "what we learned" framing. Not publishable at any main venue.

---

## The Single Most Important Thing Proposer Should Do in the Next 48 Hours

### **VERIFY PATIENT-LEVEL SPLIT (D4)**

Run the 3-step verification commands above. Document the results in `PET_LatentResidual/docs/DATA_SPLIT_VERIFICATION.md`.

**If verification PASSES**: Proceed to Exp 1.

**If verification FAILS or is AMBIGUOUS**: STOP ALL WORK. The project is dead until a properly-split dataset is created from scratch.

This is the single highest-leverage action because:
1. It takes 1 hour
2. It determines whether the project continues or dies
3. All other work (experiments, writing, analysis) is wasted if the split is wrong
4. Medical venue reviewers WILL check this — better to know now than at desk-reject

---

## Appendix: Verification Template

Create `/Users/jiaxiang/Desktop/文件/先进院文件/latent_flow/PET_LatentResidual/docs/DATA_SPLIT_VERIFICATION.md`:

```markdown
# Data Split Verification

**Date**: 2026-04-24
**Verified by**: [NAME]

## 1. Subject Lists

- `train_subjects.txt` contains: [X] subjects
- `val_subjects.txt` contains: [Y] subjects
- Overlap: [NONE / list overlapping IDs]

## 2. NPY Slice Counts

- Total train slices (from train subjects): [NUMBER]
- Total val slices (from val subjects): [NUMBER]

## 3. PT File Slice Counts

- `preprocessed_data_D50.pt['train']['x_T'].shape[0]`: [NUMBER]
- `preprocessed_data_D50.pt['val']['x_T'].shape[0]`: [NUMBER]

## 4. Match Verification

- NPY train slices == PT train slices: [YES/NO]
- NPY val slices == PT val slices: [YES/NO]

## 5. Provenance

PT files were created by: [SCRIPT PATH]
Command used: [EXACT COMMAND]
Script uses train_subjects.txt/val_subjects.txt: [YES/NO - cite line number]

## Conclusion

Patient-level split verified: [YES/NO]

If NO, describe the issue and remediation plan.
```

---

*End of Round 4 Joint Panel Ruling*
