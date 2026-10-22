# Context-Marginalized Shapley Attribution for Spectral-Spatial Channels

## Motivation

Spectral-Spatial Interpretation (SSI) defines per-position conditional Shapley games by fixing the observed surrounding spatial context as a constant baseline, so its channel attributions are not context-invariant: any confounder embedded in that single fixed context propagates into every channel score. This fixed-context assumption was inherited unchanged from the attention-based ancestor, and no existing branch introduces a mechanism to vary or marginalize the spatial context, leaving a shared residual gap in context-invariance and debiasing across all branches. The root cause is structural: a point-conditioned value function cannot separate a channel's intrinsic contribution from its interaction with one arbitrarily chosen context.

## Key Insight

Replacing SSI's point-conditioned value function with an expectation over a spatial-context distribution turns each channel's Shapley value into a marginal (context-averaged) functional, which is invariant to any single baseline's confounders because the confounder enters as a mean-zero fluctuation across the context ensemble rather than as a fixed bias.

## Method

**What it is.** Context-Marginalized Shapley (CMS) computes a spectral-spatial attribution matrix whose per-channel value is the Shapley value of a *marginalized* game: the value of a channel coalition is its expected contextual contrast over a spatial-context distribution, rather than over one fixed observed context. Inputs: a trained HSI classifier f, a spatial position set, and a context sampler q; outputs: an attribution matrix A with one prior-averaged score per (position, channel).

**Load-bearing assumption (stated explicitly).** CMS assumes that the context ensemble C_p is exchangeable with respect to the spatial-context confounder, so that the confounder's contribution is mean-zero across C_p and v_p(S) recovers a prior-averaged channel effect rather than a context-free effect. This assumption is load-bearing and can fail when the context distribution q_p is derived from f's own softmax, because the softmax is miscalibrated exactly at boundary/rare-mixture positions (Guo et al., 2017), so C_p is drawn from the wrong class mixture precisely where the claimed advantage lives. To make the assumption testable rather than circular, CMS uses an externally specified, calibration-independent context distribution q_p^ext (empirical land-cover class frequencies from training labels within radius r=5 of p, not f's softmax) as the primary sampler, and reports a sensitivity bound over a family of priors q (including the softmax-derived q_p^soft) rather than claiming unconditional context-invariance. The estimand is explicitly the prior-averaged marginal contribution E_q[phi_i], not a context-free effect.

**How it works.**

```pseudocode
# Phase 1: build context distribution q
for each position p:
    # Primary (calibration-independent) prior:
    q_p^ext = empirical distribution over training patches within radius r=5 of p,
              weighted by land-cover class frequencies from training LABELS
              (not f's softmax); K=32 contexts sampled i.i.d. from q_p^ext
    # Sensitivity prior (for ablation only):
    q_p^soft = same but weighted by f's softmax class prior on p
    C_p = { c_1..c_K }  # K=32 contexts drawn from q_p^ext (primary)

# Phase 2: marginalized value function
v_p(S) = (1/|C_p|) * sum_{c in C_p} [ f(x_p with channels S kept, others
                                        replaced by baseline b, context c)
                                        - f(x_p with all channels replaced
                                        by b, context c) ]
# b = per-channel training mean

# Phase 3: Shapley over channels, per position
for each position p, channel i:
    A[p,i] = Shapley_i(v_p)   # exact for d_channels<=12,
                              # permutation MC with m=2000 for larger d

# Phase 4: debias / invariance check (falsifiable)
# (a) same-prior resample: recompute A on C'_p ~ q_p^ext (measures sampling drift)
# (b) adversarial reweight: recompute A on C''_p drawn from a reweighted prior
#     q_p^adv that up-weights rare/mixed land-cover classes by factor 2
#     (measures residual confounding drift)
# report drift_same = ||A(C_p) - A(C'_p)|| and drift_adv = ||A(C_p) - A(C''_p)||
# variance-decomposition control: drift_same estimates MC/sampling variance;
# drift_adv - drift_same estimates confounder-induced drift
```

Each position's context set C_p is drawn from a calibration-independent, label-derived class-matched prior so that the marginalization averages over plausible contexts rather than arbitrary ones; the baseline b is the per-channel training mean. The softmax-derived prior is retained only as a sensitivity ablation.

**Why this design.** We chose an *expectation over a class-matched context ensemble* over conditioning on the single observed context (as SSI does) because the fixed-context game is a point estimate whose confounders cannot be averaged out, accepting the cost that CMS requires K forward passes per coalition and thus is K times more expensive than SSI. We chose *exact Shapley for low channel counts and permutation Monte Carlo (m=2000) for high counts* over a single estimator for all regimes because exactness removes sampling noise exactly where it is affordable while keeping high-dimensional HSI tractable, accepting a small residual MC bias for large d. We chose a *class-matched sampling prior* over uniform random contexts because uniform contexts inject land-cover mismatch as a new confounder, accepting the cost that the prior must be estimated; to avoid circularity we estimate it from training labels rather than f's softmax, and we report a sensitivity bound over a family of priors q (including the softmax-derived q_p^soft) rather than claiming unconditional context-invariance. A domain expert would not describe CMS as a variant of MC-Owen hierarchical accounting (the SSI baseline) because Owen-style accounting averages over *external spatial coalitions inside a single game*, whereas CMS marginalizes the *value function itself* across an ensemble of contexts, changing the game rather than the coalition algebra — the two differ in what is being averaged, not in how the average is computed.

**Why it measures what we claim.** The computational quantity `v_p(S)` — the mean contextual contrast over C_p — measures the *prior-averaged channel contribution* because the assumption that the label-derived context ensemble is exchangeable with respect to the confounder makes the confounder's contribution mean-zero across the ensemble, leaving only the channel's stable effect; this assumption fails when the classifier's context dependence is itself class-conditional in a way the label-derived prior does not capture (e.g., rare land-cover mixtures), in which case v_p(S) reflects the average over the sampled prior rather than a truly context-free effect. The quantity `A[p,i] = Shapley_i(v_p)` measures *debiased credit* because the efficiency axiom of the Shapley value on the marginalized game guarantees that summed channel attributions recover the ensemble-averaged position contrast, an identity that holds only if the marginalization is exact; this fails under permutation MC with finite m, in which case the sum recovers the contrast up to MC error. The quantity `drift_adv - drift_same` measures *residual context sensitivity* because the assumption that C''_p is drawn from an adversarially reweighted prior (not the same prior) makes the difference a direct estimate of confounder-induced drift rather than sampling variance; this fails when the held-out sets are too small, in which case both drift terms reflect sampling variance rather than true context dependence.

## Contribution

(1) A context-marginalized Shapley attribution framework (CMS) that replaces SSI's point-conditioned value function with an expectation over a class-matched spatial-context distribution, yielding context-invariant spectral-spatial channel attributions. (2) A design principle — class-matched context sampling as the debiasing mechanism — that shows how the choice of context prior, not the coalition algebra, controls confounder leakage in local Shapley games. (3) An empirical drift metric that quantifies context-invariance by comparing attributions across held-out context ensembles, applicable to any position-local attribution method.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HLS Burn Scar (hyperspectral) | Matches SSI baseline's native benchmark for direct comparison |
| Primary metric | Context-drift of attribution matrix, decomposed into drift_same (sampling) and drift_adv - drift_same (confounder) | Directly quantifies claimed context-invariance and separates sampling noise |
| Baseline | MC-Owen (SSI) | Tests whether marginalizing the value function beats coalition averaging |
| Baseline | Fixed-context Shapley | Isolates the effect of context ensembling alone |
| Baseline | Uniform-context CMS | Tests whether class-matched prior matters versus arbitrary contexts |
| Ablation-of-ours | CMS with softmax-derived prior (q_p^soft) | Tests sensitivity of the class-matched prior to classifier calibration |
| Ablation-of-ours | CMS with label-derived prior (q_p^ext) | Primary configuration; calibration-independent |

### Why this setup validates the claim

The central claim is that CMS yields prior-averaged, debiased channel attributions because it marginalizes the value function over a label-derived context ensemble rather than conditioning on one observed context. To falsify this, we need a design where the predicted failure mode of each alternative is observable as a distinct signal. The HLS Burn Scar dataset carries exactly the kind of spatial-context confounders (burn-scar boundaries, land-cover transitions) that make fixed-context attribution unstable, so context-drift is a sensitive readout. MC-Owen tests the sub-claim that averaging over coalitions inside one game is insufficient — if it were sufficient, its drift would match CMS. Fixed-context Shapley isolates whether the context ensemble itself is the active ingredient, versus any Shapley computation. Uniform-context CMS isolates whether the class-matched prior is doing work beyond mere ensembling. The softmax-prior ablation tests the stated coupling risk: if attribution is fragile to calibration, the prior is not doing what we claim. Context-drift is the right metric because it is the direct operationalization of "prior-averaged invariance" — the property the method is built to deliver — and it is measurable without ground-truth attributions. The variance-decomposition control (drift_same vs. drift_adv - drift_same) separates MC/sampling drift from confounder-induced drift, so a small drift_same cannot masquerade as invariance.

### Expected outcome and causal chain

**vs. MC-Owen (SSI)** — On a position straddling a burn-scar/vegetation boundary, MC-Owen conditions on the single observed context, so its channel attributions absorb whichever land-cover mixture happens to sit in that patch; re-scoring against a held-out context set shifts the attribution because the confounder is baked into the game. CMS instead averages the value function over a label-derived class-matched context ensemble, so the confounder's contribution is mean-zero across the ensemble and the channel's stable effect survives. We expect a noticeably larger confounder-drift (drift_adv - drift_same) for MC-Owen on boundary positions, with near-parity on interior, single-class positions where no confounder exists.

**vs. Fixed-context Shapley** — On a position whose observed context is a rare land-cover mixture, fixed-context Shapley produces an attribution shaped by that one mixture: the channel ranking is dominated by whichever channels interact with the specific context, and the efficiency identity holds only for that context, not for the class. Our method replaces the point-estimate game with an expectation over plausible contexts, so the efficiency identity recovers the ensemble-averaged contrast. We expect fixed-context Shapley to show high confounder-drift and unstable channel rankings across held-out contexts, while CMS shows low confounder-drift and stable rankings, with the gap concentrated on positions with rare or mixed contexts.

**vs. Uniform-context CMS** — On a position in a homogeneous region, uniform random contexts inject land-cover mismatch, so the marginalized value function averages over implausible contexts and the attribution reflects that mismatch rather than the channel's effect; drift rises because the held-out class-matched set disagrees with the uniform prior. Our label-derived prior keeps the ensemble exchangeable with the confounder, so the average is taken over contexts the classifier actually faces. We expect uniform-context CMS to show elevated confounder-drift and attenuated channel scores specifically where class priors are sharp, with CMS matching it only in genuinely mixed regions.

**vs. CMS with softmax-derived prior (q_p^soft)** — The softmax prior is miscalibrated at boundary/rare-mixture positions, so C_p is drawn from the wrong class mixture exactly where the confounder lives; the mean-zero argument fails and confounder-drift rises. The label-derived prior is calibration-independent, so its ensemble remains exchangeable with the confounder. We expect the softmax-prior ablation to show higher confounder-drift than the label-derived CMS on boundary/rare-mixture positions, with near-parity on interior positions where calibration is good.

### What would falsify this idea

If CMS's confounder-drift advantage over MC-Owen were uniform across boundary and interior positions rather than concentrated on the boundary/rare-mixture subset where the confounder mechanism predicts a gap, the claim that marginalizing the value function removes context confounders would be wrong. Likewise, if the softmax-prior ablation matched the label-derived CMS exactly, the calibration-independent prior would not be doing the work we attribute to it.

## References

1. Spectral-Spatial Interpretation
2. Spectral-Spatial Attention Transformer with Dense Connection for Hyperspectral Image Classification
3. Faith-Shap: The Faithful Shapley Interaction Index
