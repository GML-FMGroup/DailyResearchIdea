# Covariate-Conditioned Mismatch-Budget Estimation for On-Policy Distillation

## Motivation

The mismatch-tolerance budget in on-policy distillation is currently a descriptive point estimate derived from three student-teacher pairs, so it cannot be queried for an unseen pair without rerunning the entire rollout-precision grid. This is a structural limitation of the measurement paradigm inherited from GKD (Agarwal et al., 'On-Policy Distillation of Language Models'), which treats teacher feedback as reliable supervision and mismatch as a nuisance to be measured per-configuration rather than a covariate-indexed function that can be transferred. The result is that system designers have no mechanism to certify, before training, whether a given model family, scale, and task will tolerate a planned rollout precision.

## Key Insight

The mismatch-tolerance threshold is identifiable as a smooth function of model-family, scale, and task descriptors because the degradation onset is governed by the ratio of quantization-induced logit perturbation to the teacher-student divergence gap, a ratio that is monotone in these covariates and therefore transferable across held-out pairs.

## Method

**What it is.** We introduce CABER (Covariate-Conditioned Adaptive Budget Estimator for Rollout mismatch), which takes as input a descriptor vector for a student-teacher pair and the observed rollout-precision grid, and outputs a predicted mismatch-tolerance threshold τ̂ (the largest normalized mismatch level at which training quality remains in the flat region) together with a calibration interval. The descriptor vector is d = [family one-hot (K=6 families), log10(parameter count), task category one-hot (T=3), teacher-student capacity ratio, quantization bit-width b ∈ {4,8}, quantization scheme one-hot (S=3: uniform, GPTQ, SmoothQuant), teacher-student logit temperature ratio T_t/T_s, tokenizer-overlap statistic (fraction of student tokens present in teacher vocabulary)]. The load-bearing assumption, stated plainly: τ is a smooth, transferable function of these descriptors alone, so that a ridge regression on a handful of held-out pairs can predict τ for an unseen pair better than reading that pair's own curve. This assumption is made explicit because the descriptor now includes the physically causal covariates (bit-width, scheme, temperature ratio, tokenizer overlap) that enter the perturbation/divergence ratio; if family/scale/task are weak predictors, the ratio-structured form below still carries the transfer.

**How it works.** CABER is a two-stage estimator: a monotone Gaussian-process surrogate over the mismatch-quality curve, followed by a threshold-extraction rule and a conformal calibration step. The covariate prior is a ratio-structured parametric form τ ≈ g(perturbation magnitude / divergence gap) with free parameters fit across pairs, replacing the fixed degree-2 ridge prior.

```pseudocode
Input: descriptor d = [family, log10(scale), task, capacity_ratio, bit-width b, scheme, temp_ratio, tokenizer_overlap]
       observed grid G = {(m_i, q_i)}  # mismatch level, training quality
       conformal calibration set C = {(d_j, τ_j)} from held-out pairs

Phase 1 - Curve surrogate (per-pair GP fit):
  fit GP over G with Matern-5/2 kernel, lengthscale 0.3 (in normalized mismatch)
  posterior mean μ(m) and variance σ²(m) for any m in [0,1]

Phase 2 - Threshold extraction:
  reference quality q0 = max_i q_i  (low-mismatch plateau)
  τ_raw = inf{ m : μ(m) < q0 - δ },  δ = 0.02·q0  (flatness tolerance)

Phase 3 - Covariate regression on τ_raw across training pairs:
  compute perturbation magnitude P(d) = c1 * (16 - b) / 16 * s_scheme * (T_t/T_s)   # s_scheme ∈ {1.0, 0.8, 0.6} for uniform/GPTQ/SmoothQuant
  compute divergence gap D(d) = c2 * capacity_ratio * tokenizer_overlap
  fit ratio-structured form  τ_raw ≈ g(P(d)/D(d))  where g(z) = a * z^p + b0,  free params {a, p, b0, c1, c2} fit by least squares across training pairs (λ=1e-2 ridge on log-params)
  gives prior mean τ_prior(d) for unseen pair

Phase 4 - Conformal calibration:
  residual r_j = |τ_raw(d_j) - τ_prior(d_j)| on C
  output τ̂(d) = τ_prior(d), interval [τ̂ - q_{1-α}, τ̂ + q_{1-α}], α = 0.1

Optional Phase 5 - Active refinement (if budget allows):
  run 2 additional rollout-precision points at m near τ̂, refit GP, update τ̂

Output: τ̂(d) and conformal interval
```

**Why this design.** We chose a monotone GP surrogate over a parametric sigmoid fit because the flat-then-degrade shape is not guaranteed to be sigmoidal for all pairs (some pairs show a shoulder), and the GP's posterior variance gives a principled way to flag extrapolation, accepting the cost of O(n³) fitting on grids of ≤10 points per pair. We chose a ratio-structured parametric prior over a neural covariate model because the calibration set has only a handful of held-out pairs, and a high-capacity model would overfit descriptors while the ratio form encodes the physically causal perturbation/divergence ratio that prior work (Agarwal et al.) shows varies smoothly with scale; the cost is that the ratio form assumes the threshold is monotone in P/D, which may fail under outlier emergence. We chose split-conformal calibration over a Bayesian credible interval because conformal gives a distribution-free coverage guarantee under exchangeability of held-out pairs, accepting the cost that the interval width is only as tight as the calibration set size allows. A domain expert would not describe CABER as a variant of GKD or of standard scaling-law fitting because neither produces a covariate-indexed threshold with a coverage certificate; GKD fixes a teacher and measures one curve, while CABER explicitly models the cross-pair transfer function and certifies it.

**Why this measures what we claim.** The GP posterior mean μ(m) measures the expected training quality at mismatch level m because the Matern kernel assumes the quality curve is twice mean-square differentiable in mismatch; this assumption fails when quantization induces a sharp phase transition rather than a smooth shoulder, in which case μ(m) reflects a smoothed interpolation across the transition rather than the true quality. The extracted τ_raw measures the mismatch-tolerance threshold because the flatness criterion δ·q0 assumes the plateau is well-defined and the degradation is monotone past onset; this assumption fails when quality is non-monotone (e.g., transient recovery), in which case τ_raw reflects the first crossing rather than the sustainable budget. The ratio-structured prior τ_prior(d) measures transferable identifiability because the form g(P/D) assumes the perturbation/divergence ratio is a sufficient statistic for the threshold; this assumption fails when a pair's threshold is dominated by a factor outside the ratio (e.g., activation outlier structure not captured by bit-width), in which case τ_prior reflects the population trend and the conformal interval absorbs the residual as a distributional shift rather than a modeling error. The conformal interval measures calibrated uncertainty because exchangeability of held-out pairs assumes the calibration pairs are drawn from the same pair distribution as deployment; this assumption fails under distribution shift to a new model family, in which case the interval reflects in-distribution coverage rather than deployment coverage.

## Contribution

(1) CABER, a covariate-conditioned mismatch-budget estimator that maps model-family, scale, and task descriptors to a predicted tolerance threshold with a split-conformal coverage certificate, replacing the descriptive point estimate with a transferable identifiability mechanism. (2) A monotone-Gaussian-process plus ridge-regression decomposition that separates within-pair curve fitting from cross-pair threshold transfer, enabling threshold prediction for held-out student-teacher pairs without rerunning the full rollout-precision grid. (3) An active-refinement protocol that spends a small number of additional rollout-precision evaluations near the predicted threshold to tighten the conformal interval, with a documented trade-off between evaluation budget and interval width.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OPD rollout-mismatch sweep (summarization, translation, arithmetic) | Directly instantiates mismatch-quality curves per pair |
| Primary metric | Held-out τ̂ prediction error vs. measured threshold | Directly tests the transfer claim, not downstream accuracy |
| Baseline | Low-mismatch OPD quality reference | Establishes the flat-region quality ceiling |
| Baseline | Fixed-δ threshold readout (no covariate regression) | Tests whether cross-pair transfer adds anything |
| Baseline | Sigmoid-fit threshold estimator | Tests whether GP surrogate is necessary |
| Baseline | Power-law scaling fit (joint across pairs) | Tests whether CABER is merely scaling-law extrapolation |
| Baseline | Hierarchical Bayesian model (partial pooling) | Tests whether GP+ratio+conformal stack is re-labeling |
| Ablation-of-ours | CABER without conformal calibration | Tests whether the interval carries real coverage |
| Ablation-of-ours | CABER with degree-2 ridge prior (original) | Tests whether ratio-structured prior is necessary |

### Why this setup validates the claim

The central claim is that CABER predicts a covariate-conditioned mismatch-tolerance threshold for an unseen student-teacher pair, with a certified interval, better than reading a threshold off that pair's own curve. The design makes this falsifiable by holding out pairs and comparing predicted τ̂ against the threshold measured from a dense mismatch sweep on the held-out pair. The fixed-δ readout baseline tests the sub-claim that the cross-pair covariate regression transfers information the per-pair curve alone cannot supply; if transfer is real, CABER should beat it specifically on pairs whose own grid is sparse. The sigmoid-fit baseline tests the sub-claim that a flexible monotone surrogate is needed rather than a parametric shape, so CABER should win on pairs exhibiting a shoulder rather than a clean sigmoid. The power-law and hierarchical Bayesian baselines test whether the GP+ratio+conformal stack is merely a re-labeling of standard scaling-law extrapolation or partial pooling; CABER should win when the ratio-structured prior captures the perturbation/divergence mechanism that these baselines ignore. The no-conformal ablation tests whether the interval is calibrated or merely decorative, measured by empirical coverage of the held-out threshold. The degree-2 ridge ablation tests whether the ratio-structured prior is necessary. The primary metric — prediction error of τ̂ against the densely measured threshold — is the right one because it isolates the quantity CABER claims to estimate, rather than confounding it with downstream quality that depends on many other training choices. A uniform win across all subsets would be suspicious; the argument predicts a structured win.

### Expected outcome and causal chain

**vs. Fixed-δ threshold readout** — On a held-out pair whose observed grid has only a few mismatch points clustered near the plateau, the fixed-δ readout produces a threshold at the first sampled crossing, which lands well past the true onset because the sparse grid never samples the shoulder. Our method instead uses the ratio-structured covariate prior over bit-width, scheme, temperature ratio, and tokenizer overlap to place τ̂ near where comparable pairs degrade, then sharpens it with the GP on the sparse grid, so we expect a noticeable error reduction on sparse-grid pairs and rough parity on dense-grid pairs.

**vs. Sigmoid-fit threshold estimator** — On a pair whose curve shows a shoulder (quality flat, then a slow decline, then a steep drop), the sigmoid fit produces a threshold displaced toward the steep drop because a sigmoid cannot represent the shoulder and absorbs it into the transition width. Our method instead fits a Matern GP whose posterior mean tracks the shoulder, so τ_raw is extracted at the true flatness boundary, and we expect a systematic gap on shoulder-bearing pairs with parity on clean-sigmoid pairs.

**vs. Power-law scaling fit (joint across pairs)** — On a pair whose threshold is driven by quantization bit-width and scheme rather than by parameter count, the power-law fit in log-scale and capacity ratio extrapolates from the scale trend and misses the perturbation-driven onset. Our method instead uses the ratio-structured prior g(P/D) that includes bit-width and scheme, so we expect lower error on pairs with non-default quantization settings and parity on pairs where scale dominates.

**vs. Hierarchical Bayesian model (partial pooling)** — On a pair from a model family with few calibration examples, the hierarchical model shrinks toward the global mean and underfits the family-specific threshold. Our method instead uses the ratio-structured prior that transfers via the perturbation/divergence mechanism, so we expect lower error on rare-family pairs and parity on well-represented families.

**vs. CABER without conformal calibration** — On a pair from a model family under-represented in the calibration set, the uncalibrated point estimate produces an interval whose nominal width understates the actual deviation from the measured threshold, because ratio-structured residuals on rare families are larger than the in-distribution residual quantile. Our method instead inflates the interval via the conformal quantile over held-out residuals, so we expect empirical coverage near the nominal level for the calibrated variant and under-coverage concentrated on rare families for the ablation.

**vs. CABER with degree-2 ridge prior (original)** — On a pair whose threshold is dominated by bit-width or tokenizer overlap, the degree-2 ridge prior over family/scale/task/capacity ratio cannot represent the perturbation/divergence ratio and fits a population trend. Our method instead uses the ratio-structured prior that includes these causal covariates, so we expect lower error on pairs with varied quantization settings and parity on pairs where family/scale/task are sufficient.

### What would falsify this idea

If CABER's prediction error is uniform across sparse-grid, shoulder-bearing, and rare-family subsets rather than concentrated where the covariate prior is supposed to supply missing information, the cross-pair transfer claim is wrong and the method is no better than reading each pair's own curve. Likewise, if the conformal interval's empirical coverage on held-out pairs is indistinguishable from the uncalibrated ablation, the coverage certificate is not doing real work.

## References

1. How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Large Language Models Are Reasoning Teachers
