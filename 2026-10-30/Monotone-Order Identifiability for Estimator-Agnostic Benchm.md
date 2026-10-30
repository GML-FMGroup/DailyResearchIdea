# Monotone-Order Identifiability for Estimator-Agnostic Benchmark Difficulty

## Motivation

The benchmark-as-empirical-measure framework of 'Your Benchmark Is an Empirical Measure Over Difficulty' aggregates per-item difficulty estimates into a benchmark-level difficulty distribution, but its own diagnostics consume difficulty scalars from mutually incompatible estimator families — IRT calibration (Fluid Benchmarking), Bradley-Terry LLM-compare, and RIDE's LLM-simulated ranker — each carrying different calibration assumptions, so the aggregated measure is estimator-contingent rather than a property of the benchmark. Across these paper-trees every ancestor treats its scalar difficulty estimate as ground truth for downstream diagnostics without formalizing when different estimators must agree, which means measure-mismatch and coverage-gap conclusions can be artifacts of the estimator chosen rather than facts about the benchmark.

## Key Insight

Under a weak monotone-likelihood condition (each estimator's response probability is monotone decreasing in a latent difficulty), all three estimator families must preserve the same pairwise ordering of items on any pair whose latent difficulty gap exceeds their respective noise floors, so the intersection of their orderings is an estimator-agnostic partial order from which the shared difficulty measure is recoverable and the non-shared residual is a measurable diagnostic.

## Method

**What it is.** MORI (Monotone-Order Resolvent for Identifiability) takes as input the per-item response matrices from the three estimator families (IRT-calibrated model correctness, Bradley-Terry LLM-compare outcomes, RIDE simulated-ranker scores) and outputs (i) a partial order on item pairs declared estimator-agnostic, (ii) a single difficulty measure defined on that partial order, and (iii) a per-item residual quantifying each estimator's contingent component.

**Load-bearing assumption (stated explicitly).** MORI assumes that each family's scalar difficulty estimate d_f is a monotone (order-preserving) function of a single shared latent difficulty theta up to estimator-specific noise, and that each family's bootstrap standard error se_f(i,j) is a calibrated proxy for its latent-gap noise floor on that shared scale. Under this assumption, sign agreement across families plus a kappa*se margin on every pair is equivalent to agreement on the shared latent ordering. This assumption is construct-validity, not noise-validity: it can fail when a family's noise is non-monotone in difficulty (e.g., LLM-compare saturates on very hard pairs and its sign becomes uninformative while its bootstrap SE stays small because the judge is consistently biased). In that case the agnostic set A reflects agreement of the estimators' noise/bias models rather than shared latent order, and the residual share misattributes a real difficulty dimension to estimator noise. MORI therefore reports A as a diagnostic of cross-family construct agreement, not as a proof of shared latent order, and the experiment includes a simulation with known latent ordering where the SE-calibration assumption is deliberately violated to show the diagnostic degrades predictably.

**How it works.**

```pseudocode
INPUT: items 1..N; response matrices
  M_IRT  (models x items, 0/1 correctness)
  M_BT   (LLM-compare pair outcomes, item x item)
  M_RIDE (35 simulated-ranker scores per item)

PHASE 1 -- per-family scalar difficulty (each family's own estimator)
  d_IRT[i]  = MLE of IRT 2PL difficulty (logistic ICC, discrimination fixed at a=1.0)
  d_BT[i]   = MLE of Bradley-Terry strength from pairwise wins
  d_RIDE[i] = mean simulated-ranker difficulty score (RIDE ranker output)
  (all standardized to zero mean, unit variance within family)

PHASE 2 -- pairwise ordering + within-family order uncertainty
  for each family f in {IRT, BT, RIDE}:
    for each item pair (i,j):
      sign_f(i,j) = sign(d_f[i] - d_f[j])
      se_f(i,j)   = bootstrap std of (d_f[i]-d_f[j])   # B=200 resamples
  declare pair (i,j) AGNOSTIC iff sign_f(i,j) is identical for all f
    AND |d_f[i]-d_f[j]| > kappa * se_f(i,j) for all f   (kappa = 2.0)
  -> agnostic set A forms a partial order (transitive closure; cycles dropped)

PHASE 3 -- shared measure via isotonic projection onto A
  fit theta[i] minimizing sum_{(i,j) in A} (theta[i]-theta[j] - mean_f(d_f[i]-d_f[j]))^2
    subject to theta[i] > theta[j] for all (i,j) in A   # isotonic regression on the DAG
  -> theta is the estimator-agnostic benchmark difficulty measure

PHASE 4 -- estimator-contingent residual (diagnostic)
  for each family f and item i:
    r_f[i] = d_f[i] - theta[i]
  residual_share = Var(r) / Var(d)   # fraction of difficulty variance that is estimator-specific
OUTPUT: partial order A, shared measure theta, residual_share, per-item r_f
```

**Why this design.** We chose to derive the shared measure by isotonic projection onto the agnostic partial order rather than by averaging the three scalar estimates, because averaging silently assumes the three estimators are unbiased measurements of one quantity and produces a number even when they disagree in sign; isotonic projection instead returns a measure only on pairs where the estimators provably agree, accepting the cost that items in few agnostic pairs get weakly identified difficulty. We chose a per-pair bootstrap standard-error gate (kappa=2.0) over a global correlation threshold because the identifiability condition is pairwise and a global threshold would let a few well-ordered items mask many ambiguous ones, accepting the cost of B=200 resamples per family. We chose to keep each family's own estimator in Phase 1 rather than refitting all three under one parametric model, because the point is to test invariance across the families as they are actually used in the frontier framework (Fluid Benchmarking's IRT, LLM-compare's Bradley-Terry, RIDE's ranker); refitting under one model would beg the question. A domain expert would not describe this as a variant of averaging or of ensemble calibration because the object produced is a partial order with a residual decomposition, not a point estimate: the closest named prior, LLM-compare's Bradley-Terry aggregation, produces a single ordering with no notion of which pairs are estimator-invariant.

**Why it measures what we claim.** The agnostic partial order A measures estimator-agnostic difficulty ordering because the monotone-likelihood assumption (each family's response probability is monotone decreasing in latent difficulty) makes sign agreement plus a noise-floor margin equivalent to a shared latent ordering; this assumption fails when an estimator's noise is non-monotone in difficulty (e.g., LLM-compare saturates on very hard pairs and its sign becomes uninformative), in which case A reflects agreement of the estimators' noise models rather than shared latent order. The isotonic measure theta measures the shared difficulty scalar because projecting family-mean pairwise gaps onto A is the unique least-squares scalar consistent with all invariant orderings under the assumption that agnostic pairs span the item set; this fails when A is sparse (few invariant pairs), in which case theta is only identified up to the connected components of A and reflects within-component arbitrary placement. The residual share measures the estimator-contingent component of difficulty because r_f = d_f - theta is the part of each family's estimate not constrained by the invariant order under the assumption that theta captures all shared structure; this fails when a real difficulty dimension is invisible to two of three families, in which case that dimension is misattributed to estimator noise.

## Contribution

(1) An estimator-agnostic identifiability condition for benchmark difficulty — a monotone-order partial order preserved across IRT, Bradley-Terry, and simulated-ranker families — with an isotonic resolvent that yields a single shared difficulty measure and a per-item residual decomposition. (2) A design principle: benchmark-level difficulty claims should be reported on the agnostic partial order with an explicit residual share, so that measure-mismatch and coverage-gap diagnostics are certified as benchmark properties rather than estimator artifacts. (3) An open diagnostic, MORI, that ingests the three estimator families' outputs and emits the partial order, shared measure, and residual share for any benchmark with the requisite response data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 25 benchmarks spanning 7 domains (difficulty-annotated), pre-registered by name | Pre-existing multi-domain item difficulty ground truth |
| Primary metric | Sign-agreement rate on held-out item pairs | Directly tests estimator-agnostic ordering claim |
| Baseline | Sign-average of three family difficulties | Tests whether naive aggregation suffices |
| Baseline | Single-family Bradley-Terry ordering | Tests whether one estimator already generalizes |
| Baseline | Global-correlation-gated isotonic (no per-pair gate) | Tests whether pairwise gating matters |
| Ablation-of-ours | MORI with kappa=0 (no noise-floor margin) | Isolates the bootstrap margin's contribution |
| Simulation check | Known-latent-ordering simulation with deliberately violated SE-calibration (LLM-compare saturation) | Tests whether the diagnostic degrades predictably when the load-bearing assumption fails |

### Why this setup validates the claim

The central claim is that MORI's agnostic partial order identifies pairs whose difficulty ordering is invariant across estimator families, and that the isotonic measure on that order recovers a shared difficulty scalar that naive aggregation cannot. The sign-average baseline tests the sub-claim that averaging produces confident orderings even where families disagree in sign, so if averaging matches MORI's sign-agreement rate on the agnostic subset, the invariance machinery is unnecessary. The single-family Bradley-Terry baseline tests whether one estimator already captures shared structure, so if it matches MORI, the multi-family decomposition adds nothing. The global-correlation-gated isotonic baseline isolates the pairwise noise-floor gate, testing whether per-pair gating (rather than a global threshold) is what produces correct agnostic pairs. The kappa=0 ablation tests whether the bootstrap margin, rather than the sign-agreement condition alone, is doing the identifiability work. The simulation check with known latent ordering and deliberately violated SE-calibration tests the load-bearing assumption directly: if the diagnostic degrades predictably (residual share concentrates on the saturated hard items, agnostic set shrinks on those pairs) when the assumption is violated, the method's reported A and residual_share are interpretable as construct-agreement diagnostics rather than as proofs of shared latent order. Sign-agreement rate on held-out pairs is the right metric because the claim is explicitly about pairwise ordering correctness under estimator disagreement, not about a global correlation or a point-estimate error; it directly detects the predicted subset-concentrated effect.

### Expected outcome and causal chain

**vs. Sign-average of three family difficulties** — On a case where two families agree on an item pair's sign but the third disagrees (e.g., LLM-compare saturating on very hard pairs and flipping sign), the sign-average produces a confident ordering because it treats all three estimates as unbiased measurements of one quantity and lets the two agreeing families outvote the third. Our method instead withholds the pair from the agnostic set because the sign condition fails, and the isotonic measure is constrained only by the pairs where all three agree. We expect a noticeable gap in sign-agreement rate on the subset of pairs with one dissenting family, and near-parity on the subset where all three agree — the failure mode is concentrated exactly where the mechanism predicts.

**vs. Single-family Bradley-Terry ordering** — On a case where the Bradley-Terry family is well-ordered on easy items but its pairwise outcomes are sparse or noisy on hard items (few comparisons, saturation), the single-family ordering produces confident ranks because it has no notion of which pairs are estimator-invariant and treats its own noise as signal. Our method instead flags those hard pairs as non-agnostic when the IRT or RIDE families disagree in sign or fail the noise-floor margin, so theta is weakly identified there and the residual share concentrates. We expect the single-family baseline to show a sign-agreement advantage on easy-item pairs but a clear deficit on hard-item pairs, with MORI's residual share elevated on exactly those hard items.

**vs. Global-correlation-gated isotonic (no per-pair gate)** — On a case where a handful of well-ordered easy items drive a high global correlation between families while many hard-item pairs are sign-ambiguous, the global gate admits the ambiguous pairs into the agnostic set because the aggregate correlation looks strong, so the isotonic projection is contaminated by pairs whose ordering is not actually invariant. Our method instead rejects those pairs per-pair via the kappa=2.0 bootstrap margin, so theta is fit only on genuinely invariant pairs. We expect the global-gate baseline to show inflated sign-agreement on the easy subset but degraded agreement on the hard subset, with MORI's advantage concentrated on the hard subset — the pattern the per-pair gate is designed to produce.

**vs. Simulation check with violated SE-calibration** — On a simulation where the latent ordering is known and LLM-compare saturation is injected on the hardest 20% of pairs (sign flips with small bootstrap SE because the judge is consistently biased), the load-bearing assumption fails. We expect the agnostic set A to shrink on those saturated pairs (they fail the sign condition), the residual share to concentrate on the saturated hard items, and the sign-agreement rate of theta against the known latent ordering to degrade predictably on the saturated subset while remaining high on the non-saturated subset. This confirms that A and residual_share are construct-agreement diagnostics whose failure mode is detectable, not silent.

### What would falsify this idea

If MORI's sign-agreement advantage over the sign-average and single-family baselines is uniform across the agnostic and non-agnostic subsets rather than concentrated on the pairs where families disagree in sign, then the invariance machinery is not doing the work the claim attributes to it. Likewise, if the kappa=0 ablation matches full MORI on the hard-item subset, the bootstrap noise-floor margin is not the mechanism producing the predicted effect; and if the violated-SE-calibration simulation shows no predictable degradation of A or residual_share, the load-bearing assumption is not merely unverified but false for these estimators.

## References

1. Your Benchmark Is an Empirical Measure Over Difficulty
2. Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation
3. Fluid Language Model Benchmarking
4. Estimating problem difficulty without ground truth using Large Language Model comparisons
5. RIDE: Difficulty Evolving Perturbation with Item Response Theory for Mathematical Reasoning
6. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations
7. metabench -- A Sparse Benchmark of Reasoning and Knowledge in Large Language Models
8. Improving Model Evaluation using SMART Filtering of Benchmark Datasets
