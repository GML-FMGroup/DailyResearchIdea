# Residual-Gated Memory Correction for High-Resolution Video Matting

## Motivation

MatAnyone-E inherits Cutie's one-way compression assumption: low-resolution propagation selects memory candidates and uncertain regions, and high-resolution matching is applied only to those selections, with no path for high-resolution evidence to correct the low-resolution selection. This structural one-way flow means that when low-resolution uncertainty gating mislocalizes an uncertain region or selects a memory frame that lacks the needed high-frequency detail, the error is uncorrectable — the high-resolution stage can only refine within the wrong region using the wrong memory, as in MatAnyone-E's two-stage scheme. The same limitation recurs across memory-based video matting and segmentation methods (Cutie's object queries, RAVOS's ROI tracker), where low-resolution or predicted selection is treated as a fixed prior rather than a hypothesis subject to high-resolution falsification.

## Key Insight

High-resolution matching residuals, computed on the selected memory candidates, are a direct falsification signal for the low-resolution selection itself: a large residual on a candidate that low-resolution gating deemed relevant indicates that the selection was wrong, not that the region is merely uncertain, so feeding the residual back as a reweighting/reselection signal converts the one-way pipeline into a closed-loop error-corrective system without retraining the low-resolution selector.

## Method

**What it is.** We introduce **RGMC** (Residual-Gated Memory Correction), a closed-loop wrapper around MatAnyone-E's two-stage propagation. Inputs: a frame, the low-resolution matte and uncertainty map, and the low-resolution-selected memory candidates. Output: a corrected high-resolution matte and an updated memory bank. The core mechanism is a high-resolution residual pass that scores each selected memory candidate and each uncertain region, then feeds those scores back to reweight or reselect memory and to expand/shrink the uncertain region set before the final high-resolution matting.

**How it works.**

```pseudocode
# Stage 1: low-resolution propagation (unchanged from MatAnyone-E)
F_lr, M_lr, U_lr = lowres_propagate(frame, memory_bank)
# U_lr: per-pixel uncertainty in [0,1]
R_lr = threshold(U_lr, tau_u)                 # uncertain region mask, tau_u=0.5
C_lr = select_memory(M_lr, memory_bank, k=3)  # low-res memory candidates

# Stage 2: high-resolution residual scoring on the selected candidates
for each candidate c in C_lr:
    R_c = highres_match_residual(frame, c, region=R_lr)   # per-pixel residual
    # Calibrated selection-validity score: deviation from null residual distribution
    # Null distribution estimated on held-out VideoMatte240K split (512 frames) as
    # per-region (8x8 grid) mean and std of high-res residual under correct selection
    # (ground-truth memory correspondence from ground-truth mattes).
    z_c = (mean(R_c) - mu_null) / (sigma_null + eps)      # z-score of residual vs null
    s_c = z_c                                              # candidate reliability score
    # low z_c => candidate consistent with high-res evidence under correct selection

# Stage 3: closed-loop correction
# Require agreement across at least two candidates before reselecting:
# compute pairwise residual agreement a_c = fraction of pixels in R_lr where
# |R_c - R_c'| < tau_agree (tau_agree=0.05) for the other candidate c' with lowest z.
# Only candidates with a_c >= 0.5 are eligible for reselection.
w_c = softmax(-s_c / T)                     # T=0.1, reweight candidates
C_corr = top_k(C_lr, w_c, k=2)              # reselect if a candidate is downweighted
R_corr = R_lr ∪ {pixels where max_c R_c > tau_r}   # expand region where residual high
R_corr = R_corr \ {pixels where all R_c < tau_low} # shrink region where residual low

# Stage 4: high-resolution matting on corrected region and memory
matte_hr = highres_matting(frame, C_corr, region=R_corr)
matte_hr = composite(matte_hr, M_lr, region=~R_corr)  # reuse low-res elsewhere
update_memory(memory_bank, matte_hr, frame)
return matte_hr
```

Hyperparameters: `tau_u=0.5` (uncertainty threshold), `k=3` initial candidates, `T=0.1` softmax temperature, `tau_r=0.3` residual expansion threshold, `tau_low=0.1` residual shrink threshold, `tau_agree=0.05` residual agreement threshold. All thresholds are set on a held-out validation split of VideoMatte240K and fixed for all benchmarks. The null residual distribution parameters `mu_null`, `sigma_null` are estimated per 8x8 region on a held-out calibration set of 512 frames from VideoMatte240K, using ground-truth memory correspondence (i.e., residual computed between frame and the ground-truth memory frame that overlaps most in ground-truth matte).

**Why this design.** We chose to compute residuals on the *selected* candidates rather than on all memory frames because MatAnyone-E's efficiency comes from sparse selection, and a full high-resolution search would erase that gain; the trade-off is that a candidate never selected cannot be recovered, so we accept a bounded recall loss in exchange for keeping the high-resolution pass sparse. We chose to reweight candidates with a softmax over negative calibrated residual scores rather than hard reselection because hard reselection is discontinuous and destabilizes the memory bank across frames; the trade-off is that a persistently wrong candidate receives a small but nonzero weight, which we mitigate by the top-k truncation at k=2. We chose to expand *and* shrink the uncertain region using two residual thresholds rather than only expanding, because MatAnyone-E's low-resolution uncertainty overestimates uncertainty in high-frequency but well-matched regions; the trade-off is two extra hyperparameters, which we fix on validation. We chose to leave the low-resolution selector and the lightweight decoder frozen and only add a residual-scoring head trained with a matte-reconstruction loss on the corrected region, because retraining the selector would require end-to-end high-resolution backpropagation through memory, which is the cost MatAnyone-E was designed to avoid; the trade-off is that the correction is a wrapper rather than a jointly optimized system. A domain expert would not describe this as a variant of Cutie's object-query memory reading or RAVOS's ROI tracking: those methods treat the selection as a fixed prior, whereas RGMC's residual score is explicitly a falsification test of the selection, and the correction path changes which memory is used, not just how it is read.

**Why this measures what we claim.** The per-candidate calibrated residual score `s_c` measures *selection validity* because we assume that a memory candidate selected by low-resolution gating should, if truly relevant, have a high-resolution matching residual that is statistically consistent with the null distribution of residuals under correct selection (estimated on held-out data); this assumption fails when the target undergoes fast motion or occlusion, in which case a high residual reflects motion blur or appearance change rather than wrong selection, and `s_c` reflects appearance change instead. The region-expansion residual `max_c R_c > tau_r` measures *mislocalized uncertainty* because we assume that high-resolution residual is high precisely where low-resolution uncertainty failed to flag a region; this assumption fails when the high-resolution matcher itself is noisy on textureless regions, in which case the expansion reflects matcher noise rather than mislocalization. The reweighting `w_c = softmax(-s_c/T)` measures *corrected memory relevance* because we assume that residual magnitude is monotone in candidate mismatch and that near-duplicate candidates do not arbitrarily split weight; this assumption fails when two candidates are near-duplicates, in which case the softmax spreads weight arbitrarily and reflects redundancy rather than relevance. The agreement check `a_c >= 0.5` measures *robust reselection* because we assume that a truly wrong selection will produce a high residual that is not shared by other candidates; this assumption fails when matcher noise is correlated across candidates, in which case the agreement check does not prevent spurious reselection.

## Contribution

["A closed-loop residual-gated correction mechanism (RGMC) that feeds high-resolution matching residuals back to reweight and reselect low-resolution memory candidates and to expand/shrink the uncertain region set, converting MatAnyone-E's one-way two-stage pipeline into an error-corrective loop without retraining the low-resolution selector.", 'A design principle: high-resolution residuals on selected memory are a falsification signal for low-resolution selection, and two asymmetric residual thresholds (expansion and shrinkage) are needed because low-resolution uncertainty both under- and over-estimates uncertainty in high-frequency regions.', 'A validation protocol and threshold-selection procedure on VideoMatte240K that makes the residual scores comparable across frames and scenes, enabling the correction wrapper to be dropped onto any memory-based video matting backbone.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VideoMatte240K held-out split | High-res mattes with fast motion and occlusion. |
| Primary metric | Matte boundary error on uncertain-region subset | Directly probes residual-corrected region quality. |
| Baseline | MatAnyone 2 | Strong dense-memory matting reference. |
| Baseline | MatAnyone-E (frozen) | Tests whether the wrapper adds value over the base. |
| Baseline | Cutie-style memory reading | Tests against fixed-prior query selection. |
| Baseline | Confidence-based memory reweighting | Reweights candidates by low-resolution uncertainty; isolates high-res residual gain. |
| Ablation-of-ours | RGMC without residual reweighting | Isolates the memory-reselection contribution. |
| Ablation-of-ours | RGMC with raw residual score (no calibration) | Tests whether calibration is necessary for selection validity. |
| Ablation-of-ours | RGMC with low-resolution-uncertainty score instead of residual | Isolates whether high-resolution evidence is doing the work. |

### Why this setup validates the claim

The central claim is that high-resolution residual scoring on selected memory candidates can falsify and correct low-resolution selection errors, yielding better mattes specifically where low-resolution uncertainty misjudges the target. VideoMatte240K's held-out split supplies the fast-motion and occlusion instances where this failure is predicted, so it is the right stress test rather than a convenience dataset. MatAnyone 2 tests whether the wrapper beats a strong dense-memory reference, establishing that sparsity plus correction is not merely cheaper but competitive. Frozen MatAnyone-E is the critical control: any gain over it must come from the residual correction, not from the underlying propagation, so it directly tests the wrapper's marginal value. The Cutie-style baseline tests the sub-claim that treating selection as a fixed prior is insufficient, since its query reading never revisits which memory is used. The confidence-based memory reweighting baseline tests whether the high-resolution residual signal provides gains beyond a simple low-resolution uncertainty-based reweighting, directly addressing the novelty concern. The no-reweighting ablation isolates the memory-reselection mechanism from the region-expansion mechanism, so a gain that survives only with reweighting confirms the causal role of corrected memory relevance. The raw-residual ablation tests whether the calibration step (z-scoring against null distribution) is necessary for the residual to measure selection validity, as opposed to raw magnitude. The low-resolution-uncertainty-score ablation replaces the residual score with a low-resolution-uncertainty-based score to isolate whether high-resolution evidence is doing the work. Boundary error on the uncertain-region subset is the right metric because the predicted effect is localized there; a whole-frame average would dilute it and could mask a real correction with parity elsewhere.

### Expected outcome and causal chain

**vs. MatAnyone 2** — On a case where the target turns quickly and a limb re-enters after brief occlusion, MatAnyone 2's dense memory carries stale appearance from before the turn and blurs the re-entering boundary because it has no falsification step on selected candidates. Our method scores each selected candidate against high-resolution evidence, downweights the stale one via the calibrated residual softmax, and reselects, so we expect a visible boundary-error reduction concentrated on fast-turn and occlusion frames, with near parity on slow, well-matched sequences.

**vs. MatAnyone-E (frozen)** — On a case where low-resolution uncertainty underestimates error over a high-frequency textured region, frozen MatAnyone-E commits its low-resolution selection and never expands the uncertain set, producing a confidently wrong matte edge. Our residual pass detects high matching residual there, expands the corrected region, and remattes at high resolution, so we expect the gain over the frozen base to be largest on the mislocalized-uncertainty subset and near zero where low-resolution uncertainty was already calibrated.

**vs. Cutie-style memory reading** — On a case where a near-duplicate memory frame is selected and the true relevant frame is ranked second, the fixed-prior reader spreads attention over both and yields a soft, doubled edge because selection is never questioned. Our residual reweighting plus top-k truncation collapses weight onto the better-matching candidate, so we expect sharper boundaries on redundancy-heavy sequences and parity where candidates are already distinct.

**vs. Confidence-based memory reweighting** — On a case where low-resolution uncertainty is high due to appearance change but the selected candidate is actually correct, the confidence-based baseline will downweight the correct candidate and blur the boundary. Our calibrated residual score, by comparing to the null distribution under correct selection, will not flag this candidate as invalid, so we expect our method to outperform the confidence-based baseline on sequences with high appearance change but correct selection, and to be comparable elsewhere.

**vs. RGMC without residual reweighting** — On a case where a wrong candidate is selected, the no-reweighting ablation will still use it and produce a blurred or incorrect boundary. Our full RGMC will downweight and reselect, so we expect a clear gain on sequences with wrong selection, and parity on sequences where selection is already correct.

**vs. RGMC with raw residual score (no calibration)** — On a case where motion blur inflates the residual uniformly across candidates, the raw-residual variant will trigger spurious reselection and destabilize the matte. Our calibrated version, by normalizing against the null distribution, will not fire on blur alone, so we expect our method to be more stable and accurate on fast-motion sequences.

**vs. RGMC with low-resolution-uncertainty score instead of residual** — On a case where low-resolution uncertainty is miscalibrated (e.g., high uncertainty on well-matched high-frequency regions), the low-resolution-score variant will fail to correct selection. Our high-resolution residual provides independent evidence, so we expect our method to outperform this ablation on mislocalized-uncertainty subsets, confirming that high-resolution evidence is doing the work.

### What would falsify this idea

If the boundary-error gain over frozen MatAnyone-E is uniform across all subsets rather than concentrated on the fast-motion, occlusion, and mislocalized-uncertainty subsets where the residual is predicted to fire, the residual score is not measuring selection validity and the central claim is wrong. Likewise, if the no-reweighting ablation matches full RGMC, the memory-correction mechanism is not doing the work we attribute to it. Additionally, if the confidence-based memory reweighting baseline matches full RGMC, the high-resolution residual signal provides no gain beyond low-resolution uncertainty, falsifying the novelty claim.

## References

1. MatAnyone-E: Sparse Memory Propagation for Efficient High-Resolution Video Matting
2. Putting the Object Back into Video Object Segmentation
3. Global Spectral Filter Memory Network for Video Object Segmentation
4. Region Aware Video Object Segmentation With Deep Motion Modeling
