# Certified Closure of the Composed Self-Improvement Operator for Robot Policy Learning

## Motivation

The field's trajectory toward recursive self-improvement (RSI) in physical AI requires that improvement mechanisms themselves improve with experience, yet no existing method demonstrates that the composition of a curation operator with a preference-aggregation operator yields a map that is closed and idempotent across RSI iterations. The synthesis in 'Toward Recursive Self-Improvement in Physical AI' explicitly identifies this structural gap: it notes that systems change the improver's knowledge or tools but cannot guarantee that gains in the improvement mechanism carry forward without degradation, because the composed operator is neither shown to map improved policies back into improved mechanisms nor to reach a fixed point. CUPID's influence-based curation, for instance, curates data for a fixed policy but provides no mechanism by which the curated data improves the curator itself in a closed loop.

## Key Insight

The composition of influence-based curation and preference aggregation is a monotone operator on a lattice of improvement mechanisms, and its closure and idempotence follow from the fact that the fixed point of this composition is exactly the set of policies whose influence scores are invariant under re-curation and whose preference rankings are invariant under re-aggregation.

## Method

# RSI-CLOSE: Recursive Self-Improvement with Certified Closure
# Input: initial policy π0, initial improvement mechanism M0 = (C0, A0)
#   C: curation operator (influence-function based)
#   A: preference-aggregation operator (pairwise ranking)
# Output: fixed-point mechanism M* and policy π*

1. Initialize M0 = (C0, A0) where C0 uses influence scores from π0 and A0 uses a Bradley-Terry model on pairwise preferences.
2. For iteration t = 0, 1, 2, ...:
   a. Collect trajectory dataset D_t by running π_t in the environment.
   b. Compute influence scores I_t(s,a) = -∇_θ log π_t(a|s) · H^{-1} ∇_θ log π_t(a|s) for each transition, where H is the Fisher information matrix estimated from D_t with damping λ=1e-3. (In practice, H^{-1} is approximated by LiSSA with 1000 iterations and a batch size of 256 to avoid full Fisher inversion.)
   c. Curate D_t^cur = { (s,a) : I_t(s,a) ≥ τ_t }, where τ_t is the (1-α) quantile of I_t with α=0.1.
   d. Aggregate preferences: for each pair (τ_i, τ_j) in D_t^cur, obtain human/robot preference label y_ij ∈ {0,1}; fit Bradley-Terry model p(i ≻ j) = σ( (r_φ(s_i,a_i) - r_φ(s_j,a_j)) / β ) with β=1, yielding reward r_φ. (Preference labels are drawn from a fixed pre-collected pool of 10,000 pairwise comparisons per task, so no new human queries are made after iteration 0.)
   e. Update policy π_{t+1} = argmax_π E_{s,a~D_t^cur}[ r_φ(s,a) ] - λ_KL KL(π || π_t) with λ_KL=0.1.
   f. Update mechanism: C_{t+1} uses influence scores from π_{t+1}; A_{t+1} uses the updated reward model r_φ.
   g. Check closure: if the curated set D_t^cur is identical (up to 1% symmetric difference) to D_{t-1}^cur AND the preference ranking induced by A_{t+1} is identical to that of A_t on D_t^cur, then declare fixed point and stop. (Closure is certified only if the Jaccard similarity between D_t^cur and D_{t-1}^cur is ≥ 0.99 AND the Kendall-tau rank correlation between A_{t+1} and A_t is ≥ 0.99, both computed over 100 bootstrap resamples of D_t^cur; the fixed point is declared only if the lower bound of the 95% bootstrap confidence interval for Jaccard is ≥ 0.99 and for Kendall-tau is ≥ 0.99.)
3. Return M* = (C_T, A_T), π* = π_T.

**Why this design** We chose a quantile-based threshold τ_t over a fixed influence cutoff because the influence score distribution shifts as the policy improves; a fixed cutoff would either discard too much data early or retain noise late, whereas the quantile adapts to the policy's evolving uncertainty, accepting the cost that the curation set size varies across iterations. We chose a Bradley-Terry model with a fixed temperature β=1 over a learned temperature because a fixed β makes the preference aggregation operator idempotent: re-fitting on the same pairwise comparisons yields identical rankings, whereas a learned β would introduce a drift that breaks closure. We chose to check closure via set equality and ranking equality rather than via a policy-performance plateau because performance plateaus can occur without mechanism convergence (e.g., due to exploration noise), whereas set and ranking equality directly certify that the composed operator has reached a fixed point. A domain expert would not describe this as a variant of CUPID because CUPID applies influence-based curation once for a fixed policy, whereas our mechanism iterates the composition and certifies that the curator itself improves—the fixed-point condition is a property of the operator, not of a single curation step.

**Why it measures what we claim** The influence score I_t(s,a) measures the causal necessity of a transition for the current policy's parameters because the influence function is the first-order approximation of the change in log-likelihood when that transition is upweighted; this assumption fails when the policy is highly non-convex (e.g., deep networks with batch normalization), in which case the score reflects local gradient alignment rather than global necessity. The curated set D_t^cur measures the minimal sufficient dataset for policy improvement because the quantile threshold assumes that transitions with influence below τ_t contribute negligibly to the policy gradient; this assumption fails when rare but critical transitions (e.g., near-collision states) have low influence but high value, in which case the curated set misses safety-critical data. The Bradley-Terry ranking measures the consistency of preference aggregation because the model assumes transitive preferences and a fixed temperature; this assumption fails when human preferences are noisy or context-dependent, in which case the ranking reflects the noise model rather than true preference order. The closure check (set and ranking equality) measures idempotence of the composed operator because equality across iterations implies the operator maps its own output to itself; this assumption fails when the dataset is finite and sampling variance causes spurious differences, in which case the check may declare a false fixed point. To control this, we replace the 1% tolerance with a bootstrap-based statistical test: we compute the 95% confidence interval for Jaccard similarity and Kendall-tau rank correlation over 100 bootstrap resamples of D_t^cur, and declare closure only if the lower bounds exceed 0.99. This makes the closure certificate robust to sampling noise.

## Contribution

['A composed self-improvement operator RSI-CLOSE that integrates influence-based curation with Bradley-Terry preference aggregation into a single iterative map, with an explicit closure certification step that detects fixed points via set and ranking equality.', 'An empirical demonstration that the composed operator reaches a fixed point within a finite number of iterations on robot manipulation tasks, and that the fixed-point mechanism yields policies whose influence scores and preference rankings are invariant under re-application, establishing idempotence across RSI iterations.', 'A diagnostic analysis identifying when the closure check fails (e.g., due to sampling variance or non-convexity) and quantifying the gap between the certified fixed point and the true fixed point of the operator.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RoboMimic manipulation suites (Lift, Can, Square) | Diverse state coverage exposes curation drift |
| Primary metric | Curated-set Jaccard stability across iterations (with 95% bootstrap CI) | Directly certifies operator idempotence, the central claim |
| Secondary metric | Downstream policy return (success rate) | Tests whether stability translates to behavioral improvement |
| Baseline | CUPID single-pass influence curation | Tests whether iteration, not just curation, drives gains |
| Baseline | Fixed-cutoff influence curation | Tests whether quantile adaptivity matters |
| Baseline | Uniform preference aggregation (no BT) | Tests whether ranking idempotence is load-bearing |
| Ablation-of-ours | RSI-CLOSE without closure check | Tests whether the certificate itself yields stable mechanisms |

### Why this setup validates the claim

The central claim is that iterating the composed curation-plus-aggregation operator, and certifying its closure, yields a self-improving mechanism rather than a one-shot curation. The primary metric — Jaccard stability of the curated set across consecutive iterations, computed with a 95% bootstrap confidence interval — is chosen because it is exactly the quantity the closure condition asserts; a method that improves behavior without converging its mechanism cannot fake this metric, and a method that stabilizes its set without improving behavior would show up as high stability with flat downstream return. CUPID tests the iteration sub-claim: if a single curation pass already achieves the same stability and return, iteration is doing no work. Fixed-cutoff curation tests the quantile sub-claim: if adaptivity is irrelevant, the threshold choice is decorative. Uniform aggregation tests the Bradley-Terry sub-claim: if ranking idempotence is not load-bearing, removing BT should not destabilize closure. The ablation isolates whether the closure certificate itself (as opposed to simply running more iterations) is what produces a stable fixed point. Together these form a falsifiable design: each baseline removes one causal ingredient, and the metric localizes where the predicted effect lives.

### Expected outcome and causal chain

**vs. CUPID single-pass influence curation** — On a case where the policy's influence distribution shifts sharply after the first update (e.g., early Lift rollouts dominated by grasp-approach transitions, later ones by lift-transition noise), the baseline produces a curated set calibrated to the initial policy and never re-curates, because its mechanism applies influence scoring once for a fixed policy. Our method instead re-estimates influence scores from π_{t+1} each iteration, so the quantile threshold tracks the moving distribution. We expect a noticeable gap in Jaccard stability on the later-stage subsets (post-grasp transitions) but near-parity on the early-approach subset where both methods agree. Additionally, we expect our method to achieve higher downstream success rate on Lift (e.g., 10% absolute improvement) while the baseline plateaus.

**vs. Fixed-cutoff influence curation** — On a case where influence scores are small in magnitude early (weakly trained policy, flat Fisher spectrum) and large late, the baseline produces either an over-pruned set early or a noise-retaining set late because a fixed cutoff cannot adapt to the shifting scale. Our method instead re-derives τ_t as the (1-α) quantile of the current influence distribution, keeping set size roughly constant in fraction. We expect the fixed-cutoff baseline to show a large early-to-late swing in curated-set size and a corresponding drop in Jaccard stability, while ours stays within a narrow stability band across iterations.

**vs. Uniform preference aggregation (no BT)** — On a case where pairwise preference labels are noisy and partially intransitive (e.g., near-tie Square insertion preferences), the baseline produces a ranking that flips between iterations because a uniform aggregator has no idempotent fixed point under re-fitting. Our method instead fits a Bradley-Terry model with fixed β=1, which is idempotent on repeated comparisons, so re-fitting yields identical rankings. We expect the ranking-equality closure condition to fire for ours within a few iterations but to oscillate for the baseline, showing as repeated closure-check failures concentrated on the noisiest preference subset.

### What would falsify this idea

If the Jaccard stability advantage of our method over CUPID is uniform across all transition subsets rather than concentrated on the late-stage, distribution-shifted subsets where the influence-shift failure is predicted to occur, then iteration is not doing the causal work we claim and the central mechanism story is wrong. Likewise, if the no-closure ablation matches ours on stability, the closure certificate is decorative rather than load-bearing.

## References

1. Toward Recursive Self-Improvement in Physical AI
