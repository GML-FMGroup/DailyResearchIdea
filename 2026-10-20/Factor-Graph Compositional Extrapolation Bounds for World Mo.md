# Factor-Graph Compositional Extrapolation Bounds for World Models

## Motivation

World-model generalization theory currently handles only marginal covariate shifts: the extrapolation framework underlying the First Steps line bounds error when a single state factor moves outside training support, but says nothing about the compositional shifts that AnyGoal constructs, where each factor and action stays in-support while their joint combinations are underrepresented. AnyGoal empirically shows every world-model family fails on these recombinations, yet without a bound that decomposes joint-composition error into per-factor terms plus interaction terms, practitioners cannot tell whether a failure is an irreducible interaction effect or a fixable marginal-support problem, so benchmark scores remain diagnostic-free.

## Key Insight

Joint-composition extrapolation error decomposes exactly over a factor graph into a sum of marginal extrapolation terms plus interaction terms that are individually estimable from in-support data, so the interaction-strength coefficients — not the raw marginal errors — are the irreducible quantity that governs compositional generalization.

## Method

**What it is.** FACET (FACtor-graph Extrapolation bound for Compositional generalization) takes as input a trained world model $f$, a factor-graph decomposition of the state/action space into factors $\{X_i\}$ with edges $E$, and a held-out set of in-support counterfactual rollouts; it outputs a scalar predicted extrapolation error $\hat{B}(g)$ for each goal-shifted variant $g$ in an AnyGoal-style benchmark, plus a per-edge attribution of that error to interaction strength.

**How it works.**

```pseudocode
Input: world model f, factor graph G=(V,E), training support S, goal variants {g}
Hyperparams: ridge lambda=1e-3, interaction-order cap K=2, held-out split rho=0.2,
             support-distance bins B=5, envelope confidence=0.9

1. MARGINAL TERMS
   For each factor i in V:
     fit local predictor f_i on S restricted to factor i
     eps_i(g) = sup_{x_i in support(g_i)} ||f(x_i) - f_i(x_i)||   # marginal extrapolation

2. INTERACTION TERMS (the load-bearing step)
   For each edge (i,j) in E:
     collect in-support pairs P_ij = {(x_i,x_j) in S}
     fit pairwise residual r_ij(x_i,x_j) = f(x_i,x_j) - f_i(x_i) - f_j(x_j)
     interaction strength w_ij = ||r_ij||_{L2(P_ij)} / (||f_i|| + ||f_j||)
     edge extrapolation e_ij(g) = w_ij * eps_i(g) * eps_j(g)

   SUPPORT-TRANSFER CALIBRATION (explicit assumption A and envelope):
     Assumption A: the residual field r_ij(x_i,x_j) has bounded variation across the
     support boundary, i.e., there exists a constant L_ij such that for any in-support
     pair p and out-of-support pair q at support distance d(p,q), |r_ij(q) - r_ij(p)| <= L_ij * d(p,q).
     Estimate L_ij by:
       - partition P_ij into B=5 nested subsets of decreasing density (by k-NN distance)
       - for each subset, compute w_ij^{(b)} = ||r_ij||_{L2(subset)} / (||f_i|| + ||f_j||)
       - fit a linear model w_ij^{(b)} = a + b * density^{(b)} + noise, take L_ij = |b| * max_density
       - compute an upper envelope w_ij^{upper} = w_ij + z_{0.9} * std(w_ij^{(b)}) / sqrt(B)
         where z_{0.9} is the 90th percentile of a standard normal (envelope confidence=0.9)
     Use w_ij^{upper} in e_ij(g) to obtain a conservative bound.

3. ASSEMBLE BOUND
   B_hat(g) = sum_i eps_i(g) + sum_{(i,j) in E} w_ij^{upper} * eps_i(g) * eps_j(g)
   (higher-order terms above order K are bounded by the residual of step 2
    on held-out rho-split, added as a scalar correction)

4. VALIDATE / ATTRIBUTE
   compare B_hat(g) to measured error on AnyGoal variants
   report per-edge e_ij as attribution
   flag edges where the support-transfer calibration indicates high variation (L_ij > median L)
Output: B_hat(g), {(i,j): e_ij}, {(i,j): L_ij}
```

**Why this design.** We chose a factor-graph decomposition with only pairwise interaction terms (order cap $K=2$) over a full joint model because the joint model is exactly what fails under compositional shift — estimating it requires the out-of-support combinations we lack, while pairwise residuals are estimable from in-support pairs, accepting the cost that genuine 3-way interactions are only captured through the held-out residual correction. We chose the multiplicative form $w_{ij}\cdot\epsilon_i\cdot\epsilon_j$ over an additive interaction term because marginal extrapolation only amplifies joint error when both marginals are stressed and the factors are coupled, accepting that the bound is looser (but still valid) when $w_{ij}$ is estimated with noise. We chose a ridge-regularized residual fit ($\lambda=10^{-3}$) over an unregularized fit because interaction strengths must be stable across the $\rho=0.2$ held-out split, accepting slight bias toward weaker interactions. A domain expert would not describe this as a variant of standard PAC-Bayes or marginal extrapolation bounds (e.g., the First Steps framework) because those have no interaction-strength coefficient at all — the mechanism's contribution is the estimable $w_{ij}$ that turns an uncomputable joint bound into a computable one. The support-transfer calibration makes the load-bearing assumption explicit: we assume the residual field has bounded variation across the support boundary (Assumption A), and we estimate an envelope $w_{ij}^{upper}$ that maps in-support residual magnitude to an out-of-support bound, turning an unverified transfer claim into a stated, testable condition.

**Why it measures what we claim.** The marginal term $\epsilon_i(g)$ measures per-factor extrapolation stress because it is the sup-norm deviation of the model from a factor-local predictor on the goal-shifted support; this assumption fails when the factor-local predictor is itself misspecified, in which case $\epsilon_i$ reflects local model bias rather than extrapolation. The interaction strength $w_{ij}$ measures the coupling between factors $i,j$ because it is the normalized L2 norm of the pairwise residual left after removing both marginals on in-support pairs; this assumption (that in-support residual magnitude transfers to out-of-support composition) fails when the interaction is itself support-dependent, in which case $w_{ij}$ reflects in-support coupling only and underestimates joint error. The product $e_{ij}= w_{ij}\epsilon_i\epsilon_j$ measures the compositional extrapolation contribution of edge $(i,j)$ because it assumes joint error is first-order in each marginal stress times coupling; this fails for strong nonlinear interactions, in which case the held-out residual correction absorbs the discrepancy and the per-edge attribution is conservative. The support-transfer calibration estimates the variation of $r_{ij}$ near the support boundary via $L_{ij}$, providing a diagnostic for when the support-invariance assumption is likely violated.

## Contribution

(1) FACET, a factor-graph decomposition that converts joint compositional extrapolation error into estimable marginal plus interaction terms, with interaction strength as an explicit, computable coefficient. (2) The design principle that interaction strength — not marginal support overlap — is the quantity that predicts whether a world model generalizes to recombined factors, giving a per-edge attribution of AnyGoal failures. (3) A reusable protocol for fitting and validating the bound on AnyGoal-style goal-shifted variants, released alongside the attribution tooling.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | AnyGoal goal-shifted variants | Compositional shift is the target claim |
| Dataset | Synthetic factorized environment with known ground-truth interaction function (e.g., 2D object with position and velocity, interaction term $\sin(x)\cdot v$) | Provides computable true joint error to verify bound validity and measure looseness |
| Primary metric | Rank correlation of B_hat vs measured error | Tests predicted ordering, not absolute scale |
| Secondary metric | Bound validity: fraction of variants where B_hat >= measured error | Tests conservative bound claim |
| Secondary metric | Looseness: mean(B_hat - measured error) on synthetic ground-truth | Measures tightness of bound |
| Baseline | First Steps marginal extrapolation bound | No interaction term; isolates w_ij contribution |
| Baseline | Joint-model residual error predictor | Fails out-of-support; shows decomposition necessity |
| Baseline | Uniform-error heuristic (constant B_hat) | Tests whether structure beats no-structure |
| Ablation-of-ours | FACET without pairwise terms (K=0 marginals only) | Isolates whether edges carry the signal |
| Ablation-of-ours | FACET with point estimate w_ij (no support-transfer envelope) | Tests whether calibration improves validity |

### Why this setup validates the claim

The central claim is that extrapolation error under compositional goal shift is predictable from in-support quantities, and specifically that the pairwise interaction coefficient $w_{ij}$ is what makes the bound track the true error ordering. Rank correlation on AnyGoal variants is the right metric because the method outputs a scalar $\hat{B}(g)$ per variant; if the mechanism is real, the predicted ordering across variants must match measured error ordering even if absolute scale is off. The First Steps marginal baseline tests the sub-claim that marginals alone are insufficient: if FACET's rank correlation exceeds it, the interaction term is doing work. The joint-model residual predictor tests the sub-claim that out-of-support joint estimation is the wrong tool. The uniform-error heuristic tests whether any structure exists to exploit. The K=0 ablation is the decisive control: it strips the load-bearing step, so any gain over it must come from $w_{ij}\epsilon_i\epsilon_j$, not from the marginal terms that both share. The synthetic ground-truth environment provides a direct test of bound validity and looseness, because the true joint error is computable, allowing us to verify that the support-transfer envelope yields a conservative bound and to measure how looseness scales with interaction nonlinearity. The point-estimate ablation tests whether the calibration step is necessary for validity.

### Expected outcome and causal chain

**vs. First Steps marginal extrapolation bound** — On a variant that stresses two coupled factors (e.g. object position and velocity jointly shifted), the marginal bound produces a flat, undifferentiated error estimate across variants because it sums per-factor stress with no coupling coefficient, so it cannot distinguish a variant where stressed factors are independent from one where they interact. Our method instead multiplies each marginal stress by the estimable $w_{ij}$ from in-support residuals, so it up-weights variants whose stressed factors are strongly coupled. We expect a noticeable rank-correlation gap on the coupled-factor subset but near-parity on variants that stress a single factor, where $w_{ij}$ terms are small.

**vs. Joint-model residual error predictor** — On a variant combining factor values never co-occurring in support, the joint predictor produces large, erratic errors because it was fit on the joint distribution that compositional shift removes, so its residual is dominated by out-of-support extrapolation noise rather than true error. Our method instead never estimates the joint directly; it composes estimable marginals and pairwise residuals, so its prediction stays ordered. We expect our method to show stable rank correlation across all variants while the joint predictor degrades sharply on the most out-of-support variants, producing a widening gap as support distance grows.

**vs. Uniform-error heuristic** — On the full variant set, the uniform heuristic produces identical predictions for every variant, so its rank correlation sits at chance by construction. Our method instead assigns distinct $\hat{B}(g)$ driven by per-variant marginal stress and edge attribution, so it should rank variants in the same order as measured error. We expect a clear positive rank correlation for FACET versus near-zero for the heuristic, and the per-edge attribution should concentrate predicted high-error variants on the edges with largest $w_{ij}$.

**vs. FACET with point estimate w_ij (no support-transfer envelope)** — On variants where the interaction is support-dependent, the point estimate $w_{ij}$ underestimates out-of-support coupling, leading to bound violations (B_hat < measured error). Our method uses the upper envelope $w_{ij}^{upper}$ from support-transfer calibration, which should increase the fraction of variants where B_hat >= measured error, at the cost of increased looseness. We expect the calibrated version to achieve higher bound validity on the synthetic ground-truth environment and on AnyGoal variants with high support-distance, while the point estimate may have higher rank correlation but lower validity.

### What would falsify this idea

If FACET's rank correlation over AnyGoal variants is statistically indistinguishable from the K=0 marginal-only ablation, the pairwise interaction term carries no predictive signal and the central claim about estimable coupling fails. Equally fatal: if the gain over First Steps is uniform across all variants rather than concentrated on the coupled-factor subset where $w_{ij}$ is large, the mechanism is not doing what we claim. Additionally, if the support-transfer calibration does not improve bound validity on the synthetic ground-truth environment (i.e., B_hat frequently underestimates true error), the load-bearing assumption of support-invariance is violated and the bound is not conservative.

## References

1. AnyGoal: On World Model Generalization Beyond Expert Rollouts
