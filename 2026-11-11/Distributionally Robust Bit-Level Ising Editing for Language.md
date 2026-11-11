# Distributionally Robust Bit-Level Ising Editing for Language Model Repair

## Motivation

Tree-Ising (the sole prior work in this direction) demonstrates that bit-level Ising edits can repair two layers of Qwen3.5-0.8B, but its entire objective — suppress the loop direction, reinforce the concluding direction, and preserve forward statistics on productive tokens — is constructed from forward-pass statistics gathered on a single evaluation distribution (MATH-500). The method therefore has no mechanism to distinguish an edit that genuinely repairs the computation from one that merely overfits the statistics of the calibration split: forward-statistic preservation is enforced pointwise, so any shift in the token distribution (new prompt style, domain, or time slice) can silently invalidate the preservation term while the QUBO solution remains unchanged. This is the structural gap: the edit's validity is defined relative to a fixed empirical distribution, and nothing in the Ising formulation couples the edit to the family of distributions the model will actually be evaluated on.

## Key Insight

Because each per-row QUBO's preservation term is a linear functional of the row's forward-statistic vector, worst-case degradation over a shift family is exactly a reweighting of that vector — so robust preservation can be folded into the same bit-level Ising objective as a single adversarial reweighting step, without leaving the discrete optimization that makes the edit admissible.

## Method

## DROBE: Distributionally Robust Bit-level Ising Editing

**(A) What it is.** DROBE takes as input the base model's BF16 weights for the two localized layers, a calibration set of forward-statistic vectors from the in-distribution split, and a *shift family* of evaluation harness splits (held-out domains / prompt styles / time slices). It outputs a set of bit-level edits (sign flips and exponent shifts) for those layers whose forward-statistic preservation holds not just on the calibration split but under the worst-case reweighting over the shift family. **Load-bearing assumption (stated explicitly):** the loop-suppression and concluding-reinforcement directions identified from forward statistics remain the causal directions driving looping under every shifted distribution in the shift family, so that a linear objective built from those statistics is a valid proxy for behavioral repair on the worst-case split. DROBE makes this assumption operational by re-estimating the objective coefficients per split and selecting the worst-vertex split's coefficients jointly with its preservation vector, so the robust term protects the actual repair objective rather than a fixed calibration proxy.

**(B) How it works.**

```pseudocode
INPUT: layers L (2), base weights W (BF16), calibration stats S_cal
       = {(x_i, h_i) : productive-token activations}, shift splits D_1..D_K
       (each yielding stats S_k), QUBO solver (simulated bifurcation, GPU)
OUTPUT: edited weights W'

# Phase 1: build per-split statistic vectors AND per-split objective coefficients
for k in 1..K:
    S_k <- forward_stats(model(W), D_k)      # loop dir, concluding dir, productive activations
S_0 <- S_cal
for k in 0..K:                               # k=0 is calibration split
    a_r^k <- loop-suppression coeffs from S_k   # per output row r
    b_r^k <- concluding-reinforcement coeffs from S_k

# Phase 2: per-row robust QUBO with distribution-aware objective
for each output row r in L:
    P_r  <- [p_r^0, p_r^1, ..., p_r^K]  # preservation vectors per split
    A_r  <- [a_r^0, a_r^1, ..., a_r^K]  # loop-suppression coeffs per split
    B_r  <- [b_r^0, b_r^1, ..., b_r^K]  # concluding-reinforcement coeffs per split
    # distributionally robust objective AND preservation: worst-case over simplex weights
    #   max_{w in simplex} sum_k w_k * [ -a_r^k.e + b_r^k.e + lambda * (e . p_r^k) ]
    #   equivalently minimize over edit the max over k of the full per-split QUBO objective
    # solved by alternating:
    #   (i)  fix edit e, set w = argmax_k [ -a_r^k.e + b_r^k.e + lambda * (e . p_r^k) ]   (vertex of simplex)
    #   (ii) fix w, solve QUBO:  min_e  sum_k w_k [ -a_r^k.e + b_r^k.e + lambda * (e.p_r^k) ]
    #        via simulated bifurcation on GPU, lambda = 0.5 (preservation weight)
    repeat (i)-(ii) until w stable (<= 3 iterations, vertex solutions only)
    e_r^* <- argmin_e of the final QUBO

# Phase 3: apply edits
W' <- apply_bit_edits(W, {e_r^*})   # sign flips + exponent shifts, group-exponent safe
return W'
```

The robust objective and preservation term are a max over the K+1 candidate per-split QUBO objectives, which (because the simplex optimum is attained at a vertex) reduces to selecting the single worst split per iteration. This keeps the problem a QUBO — the same solver used by Tree-Ising applies unchanged. The objective coefficients a_r^k, b_r^k are re-estimated per split in Phase 1 and reweighted jointly with the preservation vector in the alternating loop, so no split's loop/concluding direction is left unrepresented.

**(C) Why this design.** We chose a *vertex-based adversarial reweighting* over a smooth KL-penalty / DRO-with-temperature because the simplex maximum is attained at a vertex, so the inner problem is exact and requires no temperature hyperparameter, accepting the cost that the robust term can be conservative when many splits are near-tied. We chose to *re-solve the QUBO per outer iteration* rather than optimize a soft surrogate jointly over edits and weights because edits are discrete and the QUBO solver is the only component that respects the group-exponent/mantissa constraint; a joint continuous relaxation would produce inadmissible edits, accepting the cost of up to 3 solver calls per row. We chose *simulated bifurcation on GPU* (the solver Tree-Ising already validated) over a generic ILP solver because the per-row QUBO is large and dense and the GPU path is the only one that scales to full rows, accepting the cost of heuristic (non-certified) optimality. We chose to *re-estimate the objective coefficients per split and reweight them jointly with the preservation vector* rather than keep a_r, b_r fixed to the calibration split because the load-bearing assumption is that the loop/concluding directions transfer across shifts; reweighting the full QUBO makes the robust term protect the actual repair objective, accepting the cost of computing a_r^k, b_r^k for each of the K+1 splits in Phase 1. A domain expert would not describe DROBE as "Tree-Ising + DRO," because Tree-Ising's preservation term is a single fixed vector and its QUBO is solved once; DROBE's load-bearing object is the *coupling* between the discrete edit and the worst-case split, which changes the QUBO's coefficients per iteration and is what makes the edit's validity distribution-free rather than calibration-set-specific.

**(D) Why it measures what we claim.** The quantity `max_k [ -a_r^k.e + b_r^k.e + lambda * (e . p_r^k) ]` measures **distributional robustness of behavioral repair and forward-statistic preservation** because the assumption that the shift family's statistic vectors span the relevant evaluation distributions makes the worst-case vertex an upper bound on degradation over any mixture of the splits; this assumption fails when a deployment distribution lies outside the convex hull of D_1..D_K, in which case the robust term reflects only the worst *seen* split and can still degrade on unseen shifts. The per-split loop-suppression coefficient `a_r^k` and concluding-reinforcement coefficient `b_r^k` measure **edit validity on split k** because the assumption that the loop/concluding directions identified on split k remain the causal directions under that split makes the linear objective a proxy for behavioral repair on that split; this assumption fails when a shift changes which direction drives looping, in which case the edit repairs the calibration behavior but not the shifted one — which is why the objective coefficients are re-estimated per split and the worst-vertex split's coefficients are used in the QUBO. The per-row decomposition measures **local, admissible repair** because the assumption that output rows are statistically independent given the layer input makes per-row QUBOs exact; this assumption fails when row outputs are coupled through downstream nonlinearities, in which case the row-wise optimum is only a first-order approximation of the layer-level effect. The min-over-splits accuracy metric measures **worst-case preservation robustness** because the assumption that the shift family's statistic vectors span the deployment distributions makes the minimum over K a proxy for worst-case deployment performance; this assumption fails when an out-of-hull deployment distribution degrades despite a high min-over-splits score, in which case the metric reflects only the worst *seen* split.

## Contribution

(1) DROBE, a distributionally robust bit-level Ising editing framework that folds worst-case-over-shift-family preservation into the per-row QUBO via exact vertex reweighting, keeping edits admissible (sign flips / exponent shifts) and solvable by the same GPU simulated-bifurcation path. (2) A distribution-shift evaluation protocol for bit-level edits that separates edit validity from calibration-set overfitting by reporting preservation and behavioral repair on held-out domain, prompt-style, and time-shifted harness splits. (3) The empirical design principle that robust preservation is achievable without leaving the discrete optimization — i.e., that distributional robustness and bit-level admissibility are compatible objectives rather than competing ones.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MATH-500 under held-out prompt-style and domain shift splits (K=4: original MATH-500 style, GSM8K-style, code-style, time-slice) | Enables worst-case-over-splits robustness test |
| Primary metric | Accuracy on worst shift split (min over K) | Directly measures robust preservation, not average |
| Baseline | Astra agent's edit (51.7% MATH-500) | Strong fixed-edit reference on same layers |
| Baseline | Tree-Ising (single-vector preservation) | Tests whether robustness term is load-bearing |
| Baseline | Tree-Ising trained on union of all shift splits' statistics (pooled vector) | Tests whether DROBE is just augmented calibration |
| Baseline | Base Qwen3.5-0.8B (7.9%) | Floor showing repair actually occurred |
| Ablation-of-ours | DROBE with mean-over-splits preservation (no max) | Isolates vertex worst-case from averaging |
| Ablation-of-ours | DROBE with fixed calibration objective (a_r, b_r from S_0 only) | Isolates distribution-aware objective from fixed proxy |
| Diagnostic | Cosine similarity of top activation directions across splits | Tests shift-invariance assumption empirically |
| Diagnostic | Out-of-hull held-out split (e.g., physics word problems) | Tests convex-hull-spanning assumption |

### Why this setup validates the claim

The central claim is that DROBE's validity is distribution-free across the shift family rather than calibration-set-specific, so the falsifiable test is whether the gain concentrates on the worst shifted split. Tree-Ising is the critical baseline: it shares DROBE's QUBO solver, layer choice, and edit representation but uses a single fixed preservation vector, so any gap between them isolates the contribution of the vertex worst-case coupling rather than solver or representation confounds. The pooled-vector Tree-Ising baseline tests whether DROBE is reducible to 'Tree-Ising with augmented calibration data': if DROBE does not beat it on the worst split, the novelty claim collapses. The Astra agent's edit tests whether a strong hand-crafted fixed edit already captures the robust region, which would undercut the need for adversarial reweighting. The base model anchors that repair occurred at all. The mean-over-splits ablation is the sharpest control: it uses the same shift family and the same QUBO but collapses the max to an average, so if DROBE's advantage is truly from worst-case vertex selection, the ablation should recover average-case performance while losing specifically on the worst split. The fixed-calibration-objective ablation isolates the contribution of re-estimating a_r, b_r per split: if DROBE's advantage is truly from distribution-aware objective coefficients, this ablation should lose on the worst split. The direction-stability diagnostic measures cosine similarity of top activation directions across splits; if these are low, the shift-invariance assumption is violated and the robust term protects a stale objective. The out-of-hull split tests the convex-hull-spanning assumption: if DROBE degrades on it despite a high min-over-splits score, the metric reflects only seen splits. Worst-split accuracy (min over K) is the right metric because average accuracy would mask exactly the failure mode DROBE claims to fix: a method can look strong on average while collapsing on one shifted domain, and only the minimum detects that.

### Expected outcome and causal chain

**vs. Astra agent's edit** — On a case where a shifted prompt style elicits a concluding-direction activation pattern that differs from the calibration distribution, the Astra edit produces correct reasoning on the original style but reverts to looping under the new style because its bit edits were optimized against a single fixed statistic vector and never saw the shifted direction. Our method instead selects edits whose preservation and objective hold at the worst vertex over the shift family because the alternating QUBO reweights toward the split most violated by the current edit, so we expect near-parity with Astra on the calibration-style split but a visible gap on the worst shifted split — roughly the difference between a mid-double-digit and a low-double-digit worst-split score.

**vs. Tree-Ising** — On a case where two shift splits pull the preservation constraint in near-opposite directions, Tree-Ising produces an edit that satisfies its single calibration vector but degrades on whichever split that vector least represents because its QUBO is solved once with fixed coefficients. Our method instead re-solves the QUBO per outer iteration against the currently worst split because the vertex-selected weights change the QUBO coefficients, so we expect Tree-Ising and DROBE to be close on the split Tree-Ising implicitly favored and to diverge on the split it neglected, with DROBE's minimum-over-splits accuracy exceeding Tree-Ising's by a clear margin while their mean accuracies stay comparable.

**vs. Tree-Ising (pooled vector)** — On a case where the union of shift splits' statistics produces a single pooled vector that averages out the outlier domain's activation geometry, the pooled-vector Tree-Ising produces an edit tuned to the average split because pooling lets the outlier's preservation vector be outvoted, so it degrades on that outlier while matching DROBE elsewhere. Our method instead lets the outlier become the worst vertex and forces the edit to satisfy it because the max term cannot be outvoted, so we expect the pooled-vector baseline to match DROBE on typical splits but to lose a concentrated chunk of accuracy on the outlier split — a gap that appears only in the min-over-splits metric and vanishes in the mean. If DROBE does not beat this baseline on the worst split, the novelty claim collapses.

**vs. mean-over-splits ablation** — On a case where one split is an outlier domain with a distinct activation geometry, the ablation produces an edit tuned to the average split because averaging lets the outlier's preservation vector be outvoted, so it degrades on that outlier while matching DROBE elsewhere. Our method instead lets the outlier become the worst vertex and forces the edit to satisfy it because the max term cannot be outvoted, so we expect the ablation to match DROBE on typical splits but to lose a concentrated chunk of accuracy on the outlier split — a gap that appears only in the min-over-splits metric and vanishes in the mean.

**vs. fixed-calibration-objective ablation** — On a case where a shift changes which direction drives looping, the fixed-calibration-objective ablation produces an edit that satisfies the worst-split preservation vector but optimizes a stale behavioral objective because its a_r, b_r are fixed to S_0, so it degrades on the shifted split while matching DROBE on the calibration split. Our method instead re-estimates a_r^k, b_r^k per split and uses the worst-vertex split's coefficients jointly with its preservation vector because the alternating loop reweights the full QUBO, so we expect the ablation to match DROBE on the calibration split but to lose a concentrated chunk of accuracy on the shifted split — a gap that appears only in the min-over-splits metric and vanishes in the mean.

### What would falsify this idea

If DROBE's gain over Tree-Ising, the pooled-vector baseline, and the mean-over-splits ablation were uniform across all shift splits rather than concentrated on the worst split and the outlier domain, the worst-case vertex coupling would not be doing the work claimed and the central robustness argument would be wrong. A second falsifier: if DROBE matched the Astra agent's edit on the worst split without any min-over-splits advantage, the shift family would be too narrow to make distributional robustness the load-bearing mechanism. A third falsifier: if the direction-stability diagnostic shows low cosine similarity across splits and DROBE still matches the fixed-calibration-objective ablation, the distribution-aware objective is not load-bearing.

## References

1. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits
