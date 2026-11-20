# Calibrated Commit Gates for MLLM-Verified World States

## Motivation

MLLM-based world models such as WM^MLLM (arxiv:2611.24193) treat the MLLM's acceptance of a candidate state as a reliable oracle, committing every candidate the verifier approves into the persistent panorama state. This inherits the general in-context visual reasoning generalization of MLLMs, which the broader literature shows is not a soundness guarantee: SymDPO (arxiv:2411.11909) documents that LMMs fail to align multimodal demonstrations with outputs and follow textual shortcuts, and Cross-modal Information Flow (f88c232e) shows visual evidence is only partially routed to the prediction position. Consequently, a verifier that accepts a candidate because it superficially matches the request can silently corrupt state, and a revisit at the same revision no longer reproduces the earlier view — breaking WM^MLLM's core persistence invariant. The root cause is structural: acceptance is a binary oracle judgment with no estimate of its own error under distribution shift or adversarial edits, and no mechanism to abstain when that error is unbounded.

## Key Insight

Verifier soundness can be made gateable by estimating acceptance error from paired counterfactual edits whose ground-truth acceptability is known by construction, so the gate threshold is set on a held-out reliability envelope rather than on the verifier's own confidence.

## Method

**What it is.** We introduce CAG (Calibrated Acceptance Gate), a wrapper around WM^MLLM's MLLM candidate verifier. Its inputs are the current panorama state `S_t` with revision history, a user request `r`, a candidate next state `S_{t+1}` proposed by the image/video generators, and a verifier acceptance score `a = V(S_t, r, S_{t+1}) ∈ [0,1]`. Its output is a decision in {COMMIT, ABSTAIN→RE-VERIFY, REJECT} together with a calibrated acceptance probability `p̂`. CAG does not replace the verifier; it decides whether the verifier's acceptance lies inside a validated reliability envelope.

**Load-bearing assumption (stated explicitly).** CAG's validity is conditional on the calibration edit-operator set covering the verifier failure modes encountered at deployment. Concretely: the isotonic map `g` is fit on counterfactual pairs whose labels are sound by construction only for the operator families in the calibration taxonomy (region-shift, lighting-shift, object-swap, request-ignoring). If a test candidate's features fall outside the support of `D_cal`, `g(f)` reflects calibration-set coverage rather than true acceptance error, and the gate must abstain rather than trust `g(f)`. We therefore make this condition checkable via a support test (Phase 1, step 3) rather than leaving it as an unstated transferability assumption.

**Edit-operator taxonomy and label soundness.** We partition the calibration edit operators into four families and record, per family, whether the negative label is sound by construction or requires audit:
- `region-shift`: translate the edit mask by 8–32 px; label 0 is sound by construction (the request's target region is not modified).
- `lighting-shift`: apply a global color-temperature change of ±1500 K outside the edit mask; label 0 is sound by construction (request-faithful edit absent).
- `object-swap`: replace the requested object with a same-category instance from a held-out panorama; label 0 is sound by construction (identity mismatch).
- `request-ignoring`: candidate is `S_t` with a plausible but unrelated edit (e.g., add a tree when the request was 'change lighting'); label 0 requires human audit on a 200-pair subsample (we report audit agreement; pairs failing audit are dropped from `D_cal`).

**How it works.**

```pseudocode
# Phase 0: offline reliability-envelope construction (once per verifier checkpoint)
D_cal = []
for each (S, r) in calibration_panoramas:            # 400 held-out scenes, disjoint from eval
    S_ok  = apply_request(S, r)                      # ground-truth acceptable state (label 1)
    for op_family in {region-shift, lighting-shift, object-swap, request-ignoring}:
        S_bad = corrupt(S_ok, op_family)             # counterfactual edits with known label 0
        for S' in sample(S_bad, k=2 per family):     # 8 negatives per (S,r) total
            a = V(S, r, S')                          # verifier acceptance score
            D_cal.append((features(S, r, S', a), label(S')))
# D_cal size: 400 scenes × (1 positive + 8 negatives) = 3,600 pairs
fit isotonic regressor g: R^d -> [0,1] on D_cal      # monotone calibration map, d=4 features
choose threshold τ* = argmax_τ  TPR(τ) s.t. FPR(τ) ≤ ε   # ε = 0.05 target false-commit rate
reliability_envelope = { (a, g(a)) : g(a) ≥ τ* }    # accepted-score region
# Support model for coverage test: fit a Gaussian KDE on D_cal features (bandwidth via Scott's rule)
# and record the 5th-percentile log-density ρ_5 over D_cal as the support floor.

# Phase 1: online gated commitment (per request)
for each user request r at state S_t:
    S_cand = generator.propose(S_t, r)
    a      = V(S_t, r, S_cand)
    f      = features(S_t, r, S_cand, a)             # 4-d: score a, edit-region overlap, revision distance, modality entropy
    # 3. Support / coverage test (checkable condition for the load-bearing assumption)
    if KDE_log_density(f) < ρ_5:
        ABSTAIN (out-of-support) and return the failing edit region to the user
    p̂      = g(f)                                    # calibrated acceptance probability
    if p̂ ≥ τ*:
        COMMIT S_cand into panorama with new revision id
    elif τ*_low ≤ p̂ < τ*:                            # τ*_low = τ* − δ, δ = 0.1
        RE-VERIFY: run V on m = 3 counterfactual perturbations of S_cand
                   (region-shifted, lighting-shifted, object-swapped);
                   commit only if all m agree with acceptance AND mean p̂ ≥ τ*
    else:
        REJECT and return the failing edit region to the user
```

**Why this design.** We chose a *counterfactual-pair calibration set with known labels* over using the verifier's raw score `a` as a confidence signal because the raw score is exactly the quantity whose calibration is in question; a held-out probe with construction-guaranteed labels gives an external anchor, accepting the cost that calibration must be redone per verifier checkpoint and per scene distribution. We chose *isotonic regression* over a parametric logistic head because isotonic regression is monotone and non-parametric, so it cannot manufacture confidence where the empirical data shows none, accepting the cost of requiring a sufficiently large calibration set (we use 3,600 pairs, with a learning-curve pilot over 250/500/1k/2k/3.6k pairs to confirm fit stability). We chose a *three-way gate with a re-verification band* over a binary commit/reject because a single threshold discards candidates near the boundary where verifier error is highest; the band routes them to additional counterfactual checks, accepting extra verifier calls (m=3) on a minority of requests. We added the *support test* because the calibration labels are sound only on the operator families we constructed; the KDE support floor makes the coverage condition checkable at test time rather than assumed. This differs from standard selective-prediction wrappers (e.g., Geifman & El-Yaniv) and from retrieval-style nearest-neighbor gating: the gate is not selecting a similar past case, it is estimating the *verifier's own error rate* on a constructed counterfactual distribution and requiring the acceptance to lie inside an envelope whose false-commit rate was empirically bounded on held-out scenes, with an explicit abstention when the test feature falls outside the calibration support. A domain expert would not describe this as a variant of selective prediction because the calibration labels come from *synthetic edits with known ground truth* (plus audited request-ignoring pairs), not from held-out natural examples, which is what makes the envelope valid under adversarial edits within the covered operator families.

**Why it measures what we claim.** The calibrated probability `p̂ = g(f)` measures *verifier acceptance soundness* because the isotonic map is fit on counterfactual pairs whose acceptability is known by construction (or audited for the request-ignoring family); this assumption fails when the synthetic edit distribution does not cover the adversarial edits encountered at test time, in which case `p̂` reflects calibration-set coverage rather than true acceptance error — which is why the support test forces ABSTAIN when `KDE_log_density(f) < ρ_5`. The threshold `τ*` measures the *false-commit rate bound* because it is chosen as the empirical TPR-maximizing threshold subject to FPR ≤ ε on held-out scenes; this assumption fails when the held-out scenes are not exchangeable with deployment scenes, in which case `τ*` reflects the calibration split's difficulty rather than the deployment bound. The re-verification band width `δ` measures *boundary uncertainty* because candidates near `τ*` are those where the isotonic map's local slope is steepest; this assumption fails when the slope is steep due to sparse calibration data rather than genuine uncertainty, in which case the band reflects data sparsity rather than verifier ambiguity (we test this with a δ-from-density ablation). The counterfactual agreement count `m` measures *edit-local soundness* because perturbations target the specific edit region the verifier claimed to check; this assumption fails when the verifier's failure mode is global (e.g., ignoring the request entirely), in which case agreement reflects perturbation robustness rather than request-faithful verification — the request-ignoring operator family is included in `D_cal` specifically to cover this global failure mode.

## Contribution

(1) A calibrated acceptance gate (CAG) that wraps an MLLM candidate verifier with a counterfactual-pair isotonic calibration map and a three-way commit/abstain/reject decision, providing an empirically bounded false-commit rate on held-out scenes. (2) A construction procedure for counterfactual edit pairs with known acceptability labels that yields a reliability envelope for verifier acceptance under region-shifted, lighting-shifted, and object-swapped edits. (3) An analysis of how verifier acceptance error concentrates near edit-region boundaries, motivating the re-verification band and its width selection.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MBench-A with synthetic edit-request suite (400 calibration scenes, disjoint eval scenes) | Matches WM^MLLM's environment-recovery protocol; supplies edit-local ground truth |
| Primary metric | False-commit rate at fixed edit-recovery (FPR ≤ 0.05 target) | Directly tests the calibration claim |
| Secondary metric | Coverage diagnostic: false-commit rate vs. KDE log-density distance between test features and `D_cal` support | Bounds the envelope's validity region empirically |
| Baseline | WM^MLLM raw verifier | Tests whether gating beats raw acceptance |
| Baseline | Selective-prediction threshold on raw score (Geifman-style, held-out naturals) | Tests counterfactual labels vs held-out naturals |
| Baseline | Binary commit/reject gate (no re-verify band) | Tests whether re-verify band matters |
| Baseline | CAG without support test (no KDE abstention) | Tests whether the coverage condition is load-bearing |
| Ablation-of-ours | CAG without re-verification band | Isolates boundary-band contribution |
| Ablation-of-ours | δ set from KDE data-density estimate instead of isotonic slope | Tests whether band reflects ambiguity vs sparsity |
| Ablation-of-ours | Calibration set with request-ignoring family removed | Tests whether global-failure coverage matters |
| Pilot | Isotonic learning curve over 250/500/1k/2k/3.6k pairs; wall-clock for one envelope build | Backs feasibility with measured numbers |

### Why this setup validates the claim

The central claim is that a counterfactually calibrated envelope bounds the verifier's false-commit rate under adversarial edits better than trusting the raw acceptance score, *conditional on the test candidate lying inside the calibration support*. MBench-A with a synthetic edit-request suite is the right substrate because it supplies both the environment-recovery metric WM^MLLM already uses and the edit-local ground truth needed to label counterfactual pairs across the four operator families (region-shift, lighting-shift, object-swap, request-ignoring). The raw-verifier baseline tests sub-claim one: that uncalibrated acceptance is overconfident on region-shifted and object-swapped edits. The selective-prediction baseline tests sub-claim two: that synthetic-edit labels give a tighter envelope than held-out natural examples, which is the specific divergence from Geifman-style wrappers. The binary-gate baseline and the no-band ablation jointly test sub-claim three: that the re-verification band recovers boundary cases the single threshold discards. The no-support-test baseline and the coverage diagnostic jointly test the load-bearing assumption: if removing the KDE abstention does not raise false-commit rate on out-of-support candidates, the coverage condition is not doing the work attributed to it. False-commit rate at fixed edit-recovery is the right primary metric because it is the quantity the envelope explicitly bounds; a method that improves average quality without lowering false-commits would not support the claim.

### Expected outcome and causal chain

**vs. WM^MLLM raw verifier** — On a case where the request is "change lighting to dusk" and the generator returns a candidate that alters object geometry, the raw verifier produces a high acceptance score because its score tracks surface plausibility rather than edit-region fidelity. Our method instead maps that score through an isotonic function fit on lighting-shifted counterfactuals with known labels, so the same raw score lands below τ* and the candidate is rejected. We expect a noticeable false-commit reduction concentrated on lighting- and region-shifted edits, with near-parity on unedited panorama revisits.

**vs. Selective-prediction threshold on raw score** — On a case where an object-swap edit is subtly wrong but resembles a natural held-out example, the held-out-natural threshold accepts it because its calibration distribution never contained construction-guaranteed negatives near that score. Our method instead rejects it because the counterfactual calibration set contains object-swapped negatives at that exact score band, so g(a) falls under τ*. We expect the gap to appear specifically on adversarial-edit subsets and to shrink on natural-scene subsets.

**vs. Binary commit/reject gate** — On a case where a candidate sits just below τ* with a correct edit region but shifted lighting, the binary gate rejects a recoverable candidate because a single threshold cannot distinguish boundary uncertainty from genuine error. Our method instead routes it to three counterfactual perturbations, which agree on acceptance, and commits it. We expect the binary gate to show lower edit-recovery at equal false-commit rate, with the difference concentrated in the τ*_low-to-τ* band.

**vs. CAG without support test** — On a test candidate whose features fall below the 5th-percentile KDE log-density (e.g., a request-ignoring edit whose feature vector is far from any calibration family), the no-support-test variant trusts `g(f)` and commits if `p̂ ≥ τ*`, whereas full CAG forces ABSTAIN. We expect the no-support-test variant to show elevated false-commit rate specifically on the out-of-support subset, and near-parity on in-support candidates, confirming that the coverage condition is load-bearing.

**Coverage diagnostic (false-commit vs. KDE distance)** — We expect false-commit rate to rise monotonically as test-feature KDE log-density falls below ρ_5, bounding the envelope's validity region empirically rather than asserting it.

### What would falsify this idea

If CAG's false-commit reduction is uniform across edit types rather than concentrated on the region-shifted, lighting-shifted, and object-swapped subsets where the calibration distribution has counterfactual coverage, then the envelope is acting as a generic confidence filter rather than a verifier-error estimator, and the central claim is wrong. A second falsifier: if the no-band ablation matches full CAG on band-localized candidates, the re-verification mechanism is not doing the work attributed to it; a third: if the no-support-test variant matches full CAG on out-of-support candidates, the coverage condition is not load-bearing.

## References

1. WM^MLLM: Turning MLLMs into World Models
