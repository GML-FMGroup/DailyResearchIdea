# Regime-Conditional Calibration Maps for Preference Judges of Autoregressive Video Generators

## Motivation

Preference judges used to evaluate autoregressive video generators — the HF-Embodied-trained evaluator in WorldSimBench and the MLLM-as-a-Judge in StreamEval — are treated as domain-invariant black-box oracles, yet the input distribution they score is a deterministic function of the generation regime: retrieval-augmented rollouts (as in VRAG from Learning World Models for Interactive Video Generation) present the judge with retrieved-context-conditioned frames, streaming rollouts present temporally ordered request-conditioned frames, and static rollouts present isolated clips. Because no prior work re-validates or recalibrates the judge under this shift, scores are not comparable across regimes, so benchmark differences between retrieval-augmented and streaming generators may reflect judge miscalibration rather than generation quality. This is a meta-level identifiability failure: the regime coordinate is observable from the generation trace but the field never exploits it.

## Key Insight

Because the regime coordinate is observable from the generation trace, paired generations that differ only in regime but share the same underlying content give label-free rank constraints from which a per-regime monotone calibration map is identifiable, making scores comparable across distribution shift without retraining the judge.

## Method

**What it is.** RECAP (Regime-Conditional Calibration of Preference judges) takes as input a frozen preference judge J, a set of generation traces each tagged with an observable regime coordinate r ∈ {retrieval-augmented, streaming, static}, a regime-balanced contrastive probe set P of paired generations (x⁺, x⁻) that differ only in regime while holding prompt and content fixed, and a small externally-anchored calibration set A_ext (n_ext = 200 prompts) for which content quality is pinned by a regime-invariant reference judge on controlled synthetic clips; it outputs a calibrated score function J̃(x, r) = g_r(J(x)) where g_r is a monotone map fitted per regime, so that J̃ is directly comparable across regimes.

**Load-bearing assumption (stated explicitly).** RECAP assumes that same-prompt, same-seed rollouts under different regimes have equal expected content quality, so any systematic judge score gap between regimes is attributable to judge miscalibration rather than to a genuine quality difference induced by the regime. This assumption is known to be violated when retrieval augmentation genuinely improves fidelity (as in VRAG) or when streaming vs. static rollouts differ in temporal consistency by construction. To make the assumption testable rather than untestable, RECAP replaces it with a measurable anchor: the externally-anchored calibration set A_ext, where content quality is pinned by a regime-invariant reference judge on controlled synthetic clips, and the calibration is fit only up to the constraint that these anchors map to equal calibrated scores. The residual gap is reported as 'regime-attributable score gap' rather than 'bias', and the confound is explicitly bounded by the regime's measured quality effect on A_ext.

**How it works.**

```pseudocode
INPUT: frozen judge J, video model G, prompt set Q (n=2000), regimes R={ret, str, sta},
       externally-anchored calibration set A_ext (n_ext=200 prompts, quality pinned by
       regime-invariant reference judge on controlled synthetic clips)
OUTPUT: monotone maps {g_r}_{r in R}, audit_report

Phase 1 — Regime-balanced contrastive probe construction
  for each prompt q in Q:
    for each regime r in R:
      # generate a rollout under regime r; trace records r explicitly
      v_{q,r} = G.rollout(q, regime=r)          # same seed, same prompt
      s_{q,r} = J(v_{q,r})                       # raw judge score
  # form contrastive pairs that share content but differ in regime
  P = { (v_{q,r}, v_{q,r'}) : q in Q, r != r' }   # |P| = 3 * n pairs per regime pair

Phase 2 — Label-free rank constraint extraction
  # content is held fixed by construction (same q, same seed); only regime varies.
  # For a regime-invariant judge, the *content quality* of v_{q,r} and v_{q,r'}
  # is equal in expectation, so any systematic score gap is regime bias.
  for each regime pair (r, r'):
    b_{r,r'} = median_{q} [ s_{q,r} - s_{q,r'} ]   # estimated regime offset
    # rank-preservation constraint: for two prompts q1, q2 with equal
    # content-quality ordering under r, the ordering must survive under r'.
    # Use a held-out anchor set A (n/2 prompts) to estimate the monotone map.

Phase 3 — Monotone calibration map fitting
  # Fit g_r as a monotone piecewise-linear spline with K=32 knots
  # by minimizing, over the anchor set A and the externally-anchored set A_ext:
  #   L(g) = sum_{(r,r')} sum_{q in A} [ g_r(s_{q,r}) - g_{r'}(s_{q,r'}) ]^2
  #        + lambda * sum_r TV(g_r)          # TV = total variation, lambda=0.01
  #        + mu * sum_{q in A_ext} sum_{(r,r')} [ g_r(s_{q,r}) - g_{r'}(s_{q,r'}) ]^2
  #          # mu=1.0; anchors pin content quality to equal calibrated scores
  # subject to g_r strictly increasing (monotonicity constraint via
  # isotonic regression on the knots, projected gradient, 500 iters).
  # The quadratic term enforces cross-regime comparability; the TV term
  # prevents overfitting to the finite probe set; the A_ext term replaces
  # the equal-expected-quality assumption with a measurable anchor.

Phase 4 — Bias audit
  # Report, per regime pair, the residual median gap after calibration;
  # flag a regime as biased if |residual| > tau (tau = 0.05 * score range).
  # Also report the regime's measured quality effect on A_ext to bound the confound.
  return {g_r}, audit_report
```

**Why this design.** We chose a *label-free* monotone calibration over supervised recalibration because no ground-truth preference labels exist for the new regimes, accepting that the map is identified only up to the content-invariance assumption enforced by same-seed paired rollouts and the externally-anchored calibration set A_ext. We chose *piecewise-linear isotonic splines* (K=32 knots) over a learned neural calibration head because isotonic regression gives a monotone map by construction with no training data beyond the probe set, accepting reduced flexibility for a small number of probes. We chose to *co-generate* the probe set with the video model's own rollouts rather than reusing an external benchmark because the regime shift is induced by the generation pipeline itself, so probes must be drawn from the same pipeline to be on-distribution, accepting that probes inherit the generator's content biases. We chose a *median* regime offset over a mean because judge scores under streaming rollouts have heavy tails from occasional frame corruption, accepting a small efficiency loss for robustness. We chose to add the externally-anchored calibration set A_ext (n_ext=200) because the equal-expected-quality assumption is known to fail when retrieval augmentation genuinely improves fidelity, so the anchor makes the assumption testable rather than untestable. A domain expert would not describe this as a variant of WorldSimBench or StreamEval: those works fix the judge and vary the generator, whereas RECAP fixes the generator and varies the judge's calibration, exploiting the regime coordinate that those benchmarks record but never use.

**Why it measures what we claim.** The cross-regime residual gap `g_r(s_{q,r}) − g_{r'}(s_{q,r'})` measures *regime-attributable score gap* because same-prompt same-seed rollouts under different regimes are assumed to have equal expected content quality, so any systematic gap is attributable to the judge's regime-dependent response; this assumption fails when the regimes themselves change content quality (e.g., retrieval augmentation genuinely improves fidelity), in which case the residual reflects a real quality difference rather than bias, and the externally-anchored calibration set A_ext bounds this confound by measuring the regime's quality effect directly. The monotone spline `g_r` measures the *calibration map* because monotonicity is assumed to be the only structure the judge's score needs to preserve for ranking, so any monotone reparameterization is rank-equivalent; this assumption fails when the judge's score is non-monotone in true quality (e.g., saturation at the top of the scale), in which case the spline cannot recover the lost ordering. The probe-set balance across regimes measures *distribution-shift coverage* because the probe set is assumed to span the regime-conditional input distribution; this assumption fails when the generator's rollouts under one regime collapse to a narrow mode, in which case the calibration is fit to a non-representative slice.

## Contribution

(1) RECAP, a regime-conditional monotone calibration framework for frozen preference judges of autoregressive video generators that exploits the observable regime coordinate in the generation trace to make scores comparable across retrieval-augmented, streaming, and static regimes without retraining the judge. (2) A label-free contrastive probe construction that co-generates same-prompt same-seed rollouts across regimes and extracts rank-preservation constraints from them, yielding a bias-audit procedure that flags regimes where the judge's scores are not comparable. (3) An empirical finding that regime-induced judge miscalibration is a first-order contributor to cross-benchmark score discrepancies between retrieval-augmented and streaming generators, and that monotone recalibration substantially reduces these discrepancies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Same-seed regime-paired rollouts (ret/str/sta) on Q=2000 prompts, plus externally-anchored calibration set A_ext (n_ext=200 prompts with quality pinned by regime-invariant reference judge on controlled synthetic clips) | Holds content fixed, isolates regime bias, and replaces the equal-quality assumption with a measurable anchor |
| Primary metric | Residual cross-regime median gap after calibration on held-out anchors | Directly measures claimed regime-attributable score gap |
| Secondary metric | Within-regime rank correlation (Spearman ρ) on held-out prompts | Checks that calibration does not shrink the gap by flattening scores |
| Baseline | Frozen judge, no calibration | Tests whether calibration is needed at all |
| Baseline | Global (regime-agnostic) monotone recalibration | Tests whether per-regime maps are necessary |
| Baseline | Mean-offset shift correction | Tests whether monotone spline beats a scalar shift |
| Baseline | Domain-adaptation calibration: CORAL, quantile mapping, group-wise Platt scaling | Tests whether RECAP's regime-specific structure captures distortions these cannot |
| Ablation-of-ours | RECAP with shuffled regime labels | Tests whether regime coordinate carries real signal |
| Ablation-of-ours | RECAP without A_ext anchor (equal-expected-quality only) | Tests whether the anchor is necessary to bound the confound |

### Why this setup validates the claim

The central claim is that a frozen preference judge carries a systematic, regime-dependent score gap that a per-regime monotone map can remove while preserving content ranking, and that this gap is identifiable as regime-attributable rather than a genuine quality difference because the externally-anchored calibration set A_ext pins content quality. Each baseline targets one link in that chain. The uncalibrated judge establishes that a gap exists at all; if it does not, RECAP has nothing to fix. The global recalibration baseline tests the sub-claim that the gap is regime-specific rather than a single shared distortion, so a per-regime map is warranted. The mean-offset baseline tests whether the correction needs monotone flexibility or merely a location shift. The domain-adaptation calibration baselines (CORAL, quantile mapping, group-wise Platt scaling) test whether RECAP's regime-specific structure captures opposite-direction distortions across regime pairs that these methods cannot. The shuffled-label ablation is the sharpest test: if the regime coordinate carries no real signal, shuffling it should cost nothing, so any gain from true labels over shuffled labels is evidence the coordinate is doing causal work. The no-A_ext ablation tests whether the anchor is necessary to bound the confound from genuine regime quality effects. The residual cross-regime median gap is the right metric because it is exactly the quantity the method is designed to minimize, computed on held-out anchors, and it is label-free, matching the regime where no ground-truth preference labels exist. A method that shrinks this gap on held-out prompts while leaving within-regime rank correlations intact and while the A_ext anchor confirms equal calibrated scores on pinned-quality clips has validated the claim; one that shrinks it by flattening scores has not, which is why rank preservation is checked alongside.

### Expected outcome and causal chain

**vs. Frozen judge, no calibration** — On a case where the same prompt and seed are rolled out under streaming versus static regimes, the frozen judge produces a systematically lower score for the streaming rollout because its score scale was fit to static frame statistics and treats streaming artifacts as quality loss. Our method instead fits a per-regime monotone map that absorbs that offset while preserving ordering, so we expect the residual median gap to shrink substantially on streaming-static pairs while the raw gap stays large; the signal is a large pre-calibration gap concentrated in streaming pairs and near-parity in static-static pairs.

**vs. Global (regime-agnostic) monotone recalibration** — On a case where retrieval-augmented rollouts and streaming rollouts are biased in opposite directions, a single global monotone map produces a compromise correction that over-corrects one regime and under-corrects the other because it assumes one shared distortion. Our method instead fits separate maps per regime, so we expect the global baseline to leave a visible residual gap on at least one regime pair while RECAP drives all pairs near the audit threshold; the signal is a non-uniform residual pattern across regime pairs for the global baseline versus a flat one for ours.

**vs. Mean-offset shift correction** — On a case where the judge's streaming bias is compressive at the top of the scale, a scalar mean offset produces a shifted but still mis-scaled score because it cannot stretch the upper range. Our method instead fits a monotone spline that can expand or compress different score regions, so we expect parity between the two on the median gap but a clear advantage for RECAP on top-quantile prompts where saturation bites; the signal is a gap that is small in the bulk and widens in the upper tail for the shift baseline.

**vs. Domain-adaptation calibration (CORAL, quantile mapping, group-wise Platt scaling)** — On a case where retrieval-augmented and streaming rollouts are distorted in opposite directions, these methods assume a single shared transformation or a unimodal shift and cannot capture opposite-direction distortions across regime pairs. Our method instead fits separate monotone maps per regime with the A_ext anchor, so we expect these baselines to leave a visible residual gap on at least one regime pair while RECAP drives all pairs near the audit threshold; the signal is a non-uniform residual pattern across regime pairs for the baselines versus a flat one for ours.

**vs. RECAP with shuffled regime labels** — On a case where the true regime coordinate is replaced by a random permutation, the calibration is fit to noise and produces maps that shrink the residual gap only by chance because the regime signal has been destroyed. Our method with true labels exploits the real coordinate, so we expect a clear separation: shuffled-label residual gaps stay near the uncalibrated level while true-label gaps drop; the signal is a gap reduction that vanishes under shuffling, confirming the coordinate carries causal information.

**vs. RECAP without A_ext anchor** — On a case where retrieval augmentation genuinely improves fidelity, the no-anchor variant absorbs the real quality difference into g_r and reports it as regime-attributable score gap, over-correcting the retrieval regime. Our method with the A_ext anchor constrains the calibration so that pinned-quality clips map to equal calibrated scores, so we expect the no-anchor variant to show a larger residual gap on retrieval pairs and a larger discrepancy between the residual gap and the measured quality effect on A_ext; the signal is a residual gap that shrinks with the anchor while the measured quality effect remains stable.

### What would falsify this idea

If the residual cross-regime gap after calibration is statistically indistinguishable from the shuffled-label ablation, or if the gain is uniform across all regime pairs rather than concentrated where the predicted regime bias is largest, the claim that the regime coordinate carries exploitable judge bias is wrong. A second falsifier: if calibration shrinks the residual gap only by collapsing within-regime rank correlation, or if the A_ext anchor shows that the residual gap tracks the regime's measured quality effect rather than being removed, the monotone map is destroying content signal or absorbing genuine quality differences rather than removing bias.

## References

1. Learning World Models for Interactive Video Generation
2. Diffusion Models Are Real-Time Game Engines
3. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
4. GAIA-1: A Generative World Model for Autonomous Driving
5. MAGVIT: Masked Generative Video Transformer
6. Phenaki: Variable Length Video Generation From Open Domain Textual Description
7. StreamEval: Evaluating Interactive Streaming Video Generation Models
