# Calibrated Monotone Fixer-Likelihood Scoring for Localization-Set Refinement

## Motivation

The frozen-fixer localizer of Let the Fixer Decide trains weighted SFT on raw patch likelihoods but never establishes that these likelihoods are calibrated or that they behave monotonically as the localizer's candidate localization sets are refined, so the ranking it optimizes is entangled with fixer-specific scoring bias. Structurally, this is a field-level problem: any pipeline that substitutes a learned oracle score for a heuristic one without a calibration-and-consistency layer inherits the oracle's uncalibrated scale, and the recent replace of heuristic scoring by fixer likelihood is exactly that swap. Because no calibration or refinement-monotonicity constraint is imposed, the localizer cannot distinguish a genuinely better localization from one the fixer happens to score higher for incidental reasons, which caps the modest gains reported (8 to 11 of 78 tasks).

## Key Insight

A frozen fixer's patch likelihood can be made a self-consistent scoring distribution over nested localization sets by fitting a candidate-set-cardinality-conditioned monotone calibration on held-out verified repairs, so that ranking correctness is separated from fixer-specific scale bias rather than assumed away.

## Method

**What it is.** MONICA (MONotone Calibrated scoring for localization) wraps a frozen fixer's patch-likelihood oracle with a held-out-fitted calibration-and-monotonicity layer. Inputs: a localizer's ranked candidate localization sets $S_1 \subset S_2 \subset \dots \subset S_K$ for each bug, the frozen fixer's raw patch likelihood $\ell(S_k)$ for each set, and a held-out split of bugs with verified patches. Outputs: calibrated scores $\hat{c}(S_k)$ that are non-increasing in set refinement and a re-ranked candidate list used to weight localizer SFT.

**Load-bearing assumption (stated explicitly).** MONICA assumes that, *within each candidate-set-cardinality bucket*, the frozen fixer's raw patch log-likelihood $\ell(S_k)$ is a monotone transform of the probability that the fixer emits a test-passing patch from $S_k$. This is what licenses a per-bucket Platt fit (a monotone logistic transform) to recover a calibrated pass-probability. The assumption is known to be fragile: LLM sequence log-likelihoods are dominated by surface artifacts (patch length, token frequency/entropy) and calibration degrades off-distribution (Kadavath et al. 2022; Zhao et al. ICLR 2023), and in APR patch likelihood is confounded by patch length and token priors (Jiang et al. ICSE 2023). If within a bucket the raw likelihood ranks passing patches below failing ones, no monotone transform can recover pass-probability. MONICA therefore does **not** assert monotonicity by construction; it (i) fits a discriminative calibration head on held-out features that include the artifact covariates, (ii) keeps the isotonic cardinality factor only as a residual term fitted *after* the discriminative model, and (iii) validates the monotone-transform premise empirically per bucket (rank correlation between $\ell$ and $y$ within bucket) and reports it. The Platt-only path is retained as an ablation, not as the default.

**How it works.**

```pseudocode
# Phase 0: cardinality-drift diagnostic (run before any calibration)
for each held-out bug b with verified patch p*:
    for k in 1..K:
        S_k <- localizer's top-k localization set
        l_k  <- frozen_fixer.loglik(p* | S_k)          # raw oracle score
        y_k  <- 1[fixer emits a test-passing patch from S_k]
        f_k  <- feature_vector(l_k, len(p*), entropy(p*), token_freq(p*), k)
        store (l_k, k, y_k, f_k)
# Diagnostic: per cardinality bucket B_j, report Spearman rho(l_k, y_k)
# and mean l_k vs k. A near-zero/negative rho within a bucket falsifies the
# monotone-transform premise for that bucket and is reported, not hidden.

# Phase 1: collect held-out calibration data (same loop as Phase 0)
# Held-out split: 20% of verified bugs, stratified by cardinality bucket.
# Pre-registered bucket sizes: B_1 (k=1), B_2 (k=2), B_3 (k=3-4), B_4 (k=5).
# Minimum 40 verified repairs per bucket; buckets below 40 are merged with
# the adjacent bucket and the merge is reported.

# Phase 2: fit calibration + monotone correction
# 2a. Discriminative calibration head (default path)
#     Logistic regression (L2, C=1.0) on features f_k = [l_k, len(p*),
#     entropy(p*), token_freq(p*), k] predicting y_k. Fit on held-out.
#     z_k <- sigmoid(w . f_k + b)          # calibrated pass-probability
#     (GBM alternative: 200 trees, depth 3, lr 0.05, same features.)
# 2b. Residual isotonic factor over cardinality, fitted AFTER 2a
#     Fit monotone non-increasing g: cardinality -> [0,1] via PAVA on
#     (k, z_k) using the residuals y_k - z_k as the response, so g(k)
#     captures only residual cardinality-driven inflation not already
#     explained by the discriminative head. No smoothing beyond bucket pooling.
# 2c. Combine: candidate-set-size-dependent monotone correction
#     c_hat(S_k) <- z_k * g(k)          # non-increasing in k by construction
#
# 2a-alt (ablation only): per-bucket Platt scaling
#     for each bucket B_j: fit (a_j, b_j) by logistic regression of y on l
#     z_k <- sigmoid(a_{j(k)} * l_k + b_{j(k)})

# Phase 3: enforce nesting + monotonicity at inference
for each new bug:
    build nested S_1..S_K from localizer
    compute c_hat(S_k) for all k
    assert c_hat(S_1) >= c_hat(S_2) >= ... >= c_hat(S_K)   # projection if violated
    rank candidates by c_hat; use as SFT weights for localizer update
```

Hyperparameters: $K=5$ candidate-set sizes; cardinality buckets $B_1{=}1, B_2{=}2, B_3{=}3\text{--}4, B_4{=}5$ with minimum 40 verified repairs per bucket (merge adjacent if below); discriminative head = logistic regression (L2, $C{=}1.0$) with GBM (200 trees, depth 3, lr 0.05) as a robustness check; PAVA isotonic residual fit with no smoothing beyond bucket pooling; one EM round as in the base pipeline. Held-out split = 20% of verified bugs, stratified by bucket.

**Why this design.** We chose a discriminative calibration head over a per-bucket Platt scaling as the default because the load-bearing monotone-transform premise is known to fail when patch likelihood is confounded by length and token priors, and a head that sees those covariates can correct for them while a monotone transform of $\ell$ alone cannot; we keep Platt as an ablation to isolate exactly how much the artifact covariates buy. We chose to fit the isotonic factor $g(k)$ on residuals after the discriminative head rather than on raw scores because otherwise $g(k)$ would absorb difficulty confounded with cardinality rather than set-size drift, accepting the cost that the residual fit is noisier and needs the pre-registered per-bucket counts. We chose to enforce nesting as a hard projection at inference over a soft penalty during training because the structural property must hold for every candidate set the localizer emits, accepting the cost that occasional projection can override a raw score the fixer preferred. Unlike the raw fixer-likelihood weighting of Let the Fixer Decide, which treats the oracle score as directly usable, MONICA inserts a fitted calibration map between oracle and localizer, so a domain expert would not describe it as a variant of that pipeline but as a post-hoc scoring layer with its own held-out estimation.

**Why it measures what we claim.** The calibrated score $\hat{c}(S_k)$ measures the probability that the fixer produces a test-passing patch from localization set $S_k$ because the discriminative head assumes that the held-out features (raw log-likelihood, patch length, token entropy/frequency, cardinality) jointly carry the pass-probability signal and that the head is fit on a split representative of test bugs; this assumption fails when the held-out split is distribution-shifted from test bugs or when an unobserved artifact drives correctness, in which case $\hat{c}$ reflects the head's fit to the calibration split rather than a transferable pass-probability. The residual isotonic factor $g(k)$ measures the residual cardinality-driven inflation of the calibrated score because it assumes that any monotone dependence of pass probability on set size, after the discriminative head, is captured by cardinality alone and that cardinality is unconfounded with bug difficulty in the held-out split; this assumption fails when cardinality and bug difficulty are confounded, in which case $g(k)$ absorbs difficulty rather than set-size bias. The nesting projection measures self-consistency of the scoring distribution because it assumes that a superset of a localization set can only help the fixer; this assumption fails when extra files inject misleading context, in which case the projection masks a genuine non-monotonicity rather than removing a scoring artifact.

## Contribution

(1) A calibration-and-monotonicity wrapper (MONICA) that turns a frozen fixer's point-estimate patch likelihood into a cardinality-conditioned, refinement-monotone scoring distribution over nested localization sets. (2) A design principle that separates ranking correctness from fixer-specific scale bias by fitting the calibration on held-out verified repairs and enforcing nesting as a hard inference-time projection. (3) A reusable held-out calibration protocol that any frozen-fixer localizer pipeline can adopt without retraining the fixer.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | 1,629 test-verified Kimi K3 trajectories on Gemma 4 E4B | Matches the base pipeline's held-out bug distribution. |
| Primary metric | Tasks solved (test-passing patch) after one EM round | Directly reflects fixer-useful localization, not proxy scores. |
| Baseline | Raw fixer-likelihood weighting (Let the Fixer Decide) | Tests whether calibration adds anything over raw oracle score. |
| Baseline | Global temperature scaling (single scalar) | Tests whether per-cardinality fit beats one global correction. |
| Baseline | Matched control (unweighted localizer SFT) | Tests whether any reweighting helps at all. |
| Baseline | Same calibration layer on a heuristic spectrum-based localizer score | Tests whether calibration generalizes beyond fixer likelihood. |
| Ablation-of-ours | MONICA without isotonic g(k) factor | Isolates the cardinality-monotonicity contribution. |
| Ablation-of-ours | MONICA with per-bucket Platt only (no discriminative head) | Isolates the artifact-covariate correction. |
| Diagnostic | Per-bucket Spearman rho(l_k, y_k) and mean l_k vs k on held-out | Empirically confirms or falsifies the cardinality-drift premise. |
| Stratified analysis | Solved-rate by bug-difficulty tercile within each cardinality bucket | Tests the unconfoundedness assumption behind g(k). |

### Why this setup validates the claim

The central claim is that inserting a held-out-fitted calibration-and-monotonicity layer between the frozen fixer's likelihood and the localizer's SFT weights improves localization usefulness beyond what the raw oracle score provides. The matched control establishes the floor: if unweighted SFT already solves as many tasks, reweighting is pointless. The raw fixer-likelihood baseline is the direct competitor — it is exactly the mechanism MONICA claims to improve upon, so any gain over it is the gain attributable to calibration. The global-temperature baseline isolates whether per-cardinality conditioning matters, since it removes drift with one scalar; if MONICA ties it, the cardinality buckets are doing no work. The no-isotonic ablation isolates the monotonicity factor from the discriminative head, and the Platt-only ablation isolates the artifact-covariate correction from the monotone-transform premise. The heuristic-oracle baseline applies the same calibration layer to a non-fixer score, testing whether the calibration idea is specific to the fixer-likelihood swap or generalizes. The per-bucket Spearman diagnostic directly tests the load-bearing monotone-transform premise before any calibration is fitted, and the difficulty-stratified analysis tests the unconfoundedness assumption behind g(k). Tasks-solved after one EM round is the right metric because it is the downstream quantity the SFT weights are supposed to move; a calibration that improves scores without improving solved tasks has failed its own stated purpose. Together these form a falsifiable ladder: each rung removes one piece of the mechanism, and the predicted pattern is that performance degrades monotonically as pieces are removed.

### Expected outcome and causal chain

**vs. Raw fixer-likelihood weighting (Let the Fixer Decide)** — On a bug where the fixer reads a small candidate set and emits a high raw likelihood for a patch that passes, raw weighting already works; but on a bug where the localizer offers a large candidate set, the fixer's likelihood inflates from reading more context and the baseline over-weights that set, steering SFT toward context-heavy bugs. Our method instead corrects within cardinality buckets via the discriminative head and applies the residual isotonic factor g(k), so the inflated large-set score is corrected before weighting. We expect a noticeable gap on the large-cardinality subset but near-parity on small-cardinality bugs, where the drift is minimal. The per-bucket Spearman diagnostic should show rho(l_k, y_k) degrading with k on the baseline, confirming the drift the method corrects.

**vs. Global temperature scaling** — On a bug whose correct file sits in a mid-sized candidate set, a single global temperature fitted across all cardinalities under-corrects small sets and over-corrects large ones, so the reweighted SFT still misallocates effort. Our method fits the discriminative head with cardinality as a feature and a residual isotonic factor per bucket, so each set size gets its own correction. We expect the global baseline to show a residual deficit concentrated in the extremes of the cardinality distribution, while MONICA shows a flatter per-bucket solved-rate profile.

**vs. Matched control (unweighted localizer SFT)** — On a bug where the fixer's oracle score is informative but the localizer's own ranking is noisy, unweighted SFT cannot exploit the fixer signal at all and solves fewer tasks. Our method injects the calibrated score as a weight, so bugs where the fixer's preference is reliable get up-weighted. We expect MONICA to beat the control broadly but with the largest margin on bugs where the fixer's raw score and the localizer's rank disagree, since that is where the extra signal has room to act.

**vs. Same calibration layer on a heuristic spectrum-based localizer score** — If the calibration layer is a general post-hoc scoring layer rather than a fixer-specific patch, applying it to a heuristic oracle should also improve over that oracle's raw score, though by a smaller margin because the heuristic score is less informative. We expect a positive but smaller gain, and we report it to show the calibration idea is not tied to the fixer-likelihood swap.

**vs. MONICA without isotonic g(k) factor** — Removing g(k) leaves the discriminative head's calibrated score, which corrects artifact covariates but not residual cardinality-driven inflation. We expect a small deficit concentrated on large-cardinality bugs, where residual inflation is largest.

**vs. MONICA with per-bucket Platt only (no discriminative head)** — This ablation tests the load-bearing monotone-transform premise directly. If the premise holds within buckets, Platt-only should match full MONICA; if it fails (as the diagnostic predicts for large buckets), Platt-only should underperform, with the gap concentrated where per-bucket Spearman rho is lowest.

### What would falsify this idea

If MONICA's gain over the raw fixer-likelihood baseline is uniform across cardinality buckets rather than concentrated on large-cardinality bugs where drift is predicted, the per-cardinality calibration is doing no causal work. Likewise, if the no-isotonic ablation matches full MONICA everywhere, the monotonicity factor is inert; if the Platt-only ablation matches full MONICA despite low per-bucket Spearman rho, the discriminative head is unnecessary; and if the difficulty-stratified analysis shows g(k) absorbing difficulty rather than set-size bias, the residual-factor claim is unsupported.

## References

1. Let the Fixer Decide: Training a Code Localizer from the Fixer's Likelihood
