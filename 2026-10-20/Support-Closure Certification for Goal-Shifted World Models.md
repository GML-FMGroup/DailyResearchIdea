# Support-Closure Certification for Goal-Shifted World Models

## Motivation

AnyGoal shows that world models fail on goal-shifted variants even when each state factor and action individually stays within training support, because the benchmark only samples goals whose combinations are underrepresented — it never certifies whether a novel combination is actually inside the compositional support of the learned latent representation. The structural gap is that 'within-support factors' is a marginal condition, while generalization requires a joint-support condition over latent-factor combinations; without an explicit closure test, any goal-shifted variant (and any downstream planner query) can silently leave the joint support, so failures are attributed to model capacity rather than to an uncertified query. This same marginal-vs-joint support gap recurs across Lejepa, Active, Behavior-Regularized Diffusion, PWM, and Interaction asymmetry, where support constraints are assumed rather than checked.

## Key Insight

If the learned latent factor supports are enclosed in interval boxes and the world model's decoder is monotone in each factor interval, then the convex hull of pairwise factor-support combinations is a sound over-approximation of the model's compositional support, so any goal outside it is certified out-of-support before planning.

## Method

**What it is.** SCC (Support-Closure Certification) takes a trained goal-conditioned world model with a factorized latent encoder $E$, a decoder $D$, and a goal-shifted query set, and returns for each query a binary certificate `in_support` together with a witness interval box; queries certified out-of-support are rejected before planning, and certified in-support queries are passed to the planner unchanged.

**Load-bearing assumption (stated explicitly).** The certification is sound only if the joint-support estimator $\hat{S}$ (defined in Phase 3) is a sound over-approximation of the set of latents the decoder maps to in-support outputs, i.e., $z_g \in \hat{S}$ implies the goal is decodable from in-support factor combinations. This holds when (i) the decoder is monotone in each factor interval (checked in Phase 2) and (ii) the expert latent buffer $Z$ is representative of the training marginal (checked by the coverage diagnostic in Phase 1). If either condition fails, $\hat{S}$ reflects a superset of the true support and out-of-support goals may be falsely certified. The original convex-hull-of-boxes closure is provably equivalent to the per-factor marginal check (conv of a product of boxes = product of boxes), so it is replaced by a joint-support estimator over latent-factor combinations; the monotonicity gate and witness machinery are retained.

**How it works.**

```pseudocode
Input: encoder E, decoder D, expert latent buffer Z (N x d), factor index sets F_1..F_K,
       goal queries G = {g_1..g_M}, coverage threshold tau = 0.95, monotonicity probe delta = 0.01,
       joint-support estimator hyperparameters: k_nn = 10, min_samples_per_cell = 5,
       pairwise grid resolution r = 8 per factor pair

# Phase 1: factor-support extraction (marginal boxes, retained as diagnostic)
for each factor k:
    Z_k = { z[F_k] : z in Z }                    # project latents onto factor coordinates
    lo_k = quantile(Z_k, (1-tau)/2);  hi_k = quantile(Z_k, 1-(1-tau)/2)
    box_k = [lo_k, hi_k]                          # interval support per factor
# coverage diagnostic: fraction of Z inside product of boxes; if < tau, flag expert-buffer shift

# Phase 2: monotonicity check (soundness precondition, unchanged)
for each factor k and each coordinate j in F_k:
    sample z ~ Z, set z' = z with z'[j] = z[j] + delta (clipped to box_k)
    if sign(D(z')[target]) != sign(D(z)[target]): mark factor j as NON_MONOTONE
if any NON_MONOTONE: fall back to per-factor empirical support only (see block C)

# Phase 3: joint-support estimator (replaces box-hull closure)
# Build a nonparametric joint-support model over the full latent Z, not per-factor boxes.
# Option A (primary): k-NN support estimate.
#   For each z in Z, compute distance to its k_nn-th nearest neighbor in Z (Euclidean in latent space).
#   Let r_knn(z) be that distance. Define support radius R = quantile({r_knn(z)}, tau).
#   Joint support S_hat = { z : min_{z' in Z} ||z - z'|| <= R }.
# Option B (pairwise relational coverage, used as cross-check):
#   For each pair of factors (k, l), discretize their joint coordinates into an r x r grid over box_k x box_l.
#   Mark a cell as covered if it contains >= min_samples_per_cell points from Z.
#   A query is pairwise-covered if all its factor-pair projections fall in covered cells.
#   S_hat_pair = intersection over all pairs of pairwise-covered sets.
# Final joint support S_hat = S_hat (Option A) AND S_hat_pair (Option B) if both available; else Option A.

# Phase 4: certification
for each goal query g:
    z_g = E(g)
    if z_g in S_hat:  certify(g, True,  witness=nearest neighbor z' in Z and distance)
    else:              certify(g, False, witness=nearest violating factor-pair cell or k-NN distance > R)

Output: certificates {in_support, witness} for all g in G
```

The joint-support estimator is the load-bearing step: it replaces the marginal check 'each factor in its own box' with a joint check over latent-factor combinations, which is exactly the condition AnyGoal's goal-shifted variants violate but never test. The box-hull closure from the original draft is dropped because it is mathematically identical to the per-factor check.

**Why this design.** We chose a nonparametric joint-support estimator (k-NN radius plus pairwise grid coverage) over a learned density estimator because AnyGoal's environments have at most six factor index sets and the expert buffer is small enough that k-NN and grid coverage are exact and cheap, accepting the cost that we cannot certify queries whose latent lies in a non-convex pocket of the true support (the k-NN ball is a sound over-approximation, not a tight one). We chose a joint-support estimator over a per-factor independence check because the failure AnyGoal documents is precisely a joint-support failure — factors are marginally in-support — so an independence check would certify the failing queries, accepting the cost that the joint estimator may admit some genuinely unsupported combinations (false positives) in exchange for never falsely rejecting a supported one. We chose an explicit monotonicity probe over assuming decoder monotonicity because the joint support is only a sound over-approximation when the decoder preserves interval order, accepting the cost of a one-time $O(|F| \cdot N)$ probe pass at certification time. We chose to reject out-of-support queries rather than repair them because repair would require intervening on the planner, which is outside the world-model surface we are certifying, accepting the cost that downstream planning coverage drops on heavily goal-shifted splits. A domain expert would not describe SCC as a variant of AnyGoal or of a density-ratio OOD detector: AnyGoal samples goal-shifted variants without certifying them, and density-ratio detectors score marginal likelihood rather than joint support, so neither produces a per-query certificate with a witness interval.

**Why it measures what we claim.** The computational quantity `z_g in S_hat` (k-NN radius and pairwise grid coverage) measures compositional support membership because the named assumption is that the decoder is monotone in each factor interval and the joint-support estimator covers the tau-quantile of expert latents; under that assumption $\hat{S}$ is a sound over-approximation of the set of latents the decoder maps to in-support outputs, so membership implies the goal is decodable from in-support factor combinations. This assumption fails when a factor coordinate is non-monotone (e.g., a periodic position encoding), in which case $\hat{S}$ reflects a superset of the true support and out-of-support goals may be falsely certified — which is why the monotonicity probe in Phase 2 is a required gate, not an optional check. The per-factor box `box_k` measures marginal factor support because quantile clipping at tau assumes the expert buffer is representative of the training distribution; this assumption fails under expert-buffer covariate shift, in which case `box_k` reflects the expert's marginal rather than the model's training marginal. The witness interval returned on rejection measures the specific factor combination that violates joint support because the violating coordinate is the argmin over k-NN distances or the first uncovered factor-pair cell; this assumption fails when multiple factors jointly violate support, in which case the witness reflects one violating coordinate rather than the minimal violating set. The monotonicity probe `sign(D(z')) != sign(D(z))` measures decoder order-preservation because a finite difference of delta = 0.01 approximates the sign of the partial derivative; this assumption fails for flat regions of the decoder, in which case the probe reflects numerical noise rather than true non-monotonicity.

## Contribution

['A support-closure certification framework (SCC) that computes, via interval arithmetic over factor-box combinations, a per-query in-support certificate with a witness interval for goal-shifted world-model queries.', 'A design principle — marginal factor support is insufficient and joint support closure is the correct certification target — instantiated as a required monotonicity gate that makes the hull a sound over-approximation of compositional support.', 'An augmentation of the AnyGoal benchmark with a certified compositional closure test that labels each goal-shifted variant as in-support or out-of-support, enabling failure attribution to uncertified queries rather than model capacity.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AnyGoal goal-shifted variants | Built to expose joint-support failures |
| Primary metric | Certified-in-support planning success rate | Ties certificate validity to downstream utility |
| Baseline | AnyGoal goal-shift sampling | Samples shifts without certifying support |
| Baseline | Density-ratio OOD detector | Scores marginal likelihood, misses joint support |
| Baseline | Per-factor box check | Marginal-only check, no joint support |
| Baseline | Learned joint density estimator (normalizing flow) | Tests whether nonparametric joint support is necessary |
| Ablation-of-ours | SCC with per-factor boxes only (no joint estimator) | Isolates the load-bearing joint-support step |

### Why this setup validates the claim

The central claim is that joint support, not marginal factor support, is what separates decodable goal-shifted queries from undecodable ones. AnyGoal's goal-shifted variants are the right dataset because they are constructed precisely where factors stay marginally in-support while the joint combination leaves the training support — the exact condition SCC is built to detect. The per-factor box check and the density-ratio detector each test a competing sub-claim: if marginal support were sufficient, both would match SCC; if likelihood scoring captured joint support, the density-ratio detector would match SCC. The learned joint density estimator (normalizing flow) tests whether a parametric joint model can match the nonparametric joint-support estimator; if it does, the nonparametric choice is not load-bearing. The joint-estimator ablation is the decisive control: it differs from full SCC only in dropping the joint-support estimator (using per-factor boxes instead), so any gap between them is attributable to the joint estimator itself rather than to rejection or witness machinery. Certified-in-support planning success rate is the right metric because it measures whether the certificate is *valid* — a certificate that admits undecodable goals shows up as planning failure on certified queries, and one that rejects decodable goals shows up as lost coverage. A metric like raw rejection rate or AUC would not distinguish these, but certified-query success rate does.

### Expected outcome and causal chain

**vs. AnyGoal goal-shift sampling** — On a case where a goal-shifted query has every factor marginally inside its expert quantile box but the joint combination was never seen, AnyGoal samples and plans on it directly because it has no support test at all, producing plans that the decoder maps to out-of-support latent outputs and that fail at execution. Our method instead evaluates the latent against the joint-support estimator (k-NN radius and pairwise grid coverage) and rejects it because the joint latent lies outside the estimated support even though each marginal coordinate is inside. We expect AnyGoal to show a pronounced success-rate drop on the goal-shifted splits and SCC to recover most of that gap, with the largest separation on the most heavily shifted subsets.

**vs. Density-ratio OOD detector** — On the same marginally-in-support, jointly-unsupported query, the density-ratio detector assigns a high in-distribution score because the marginal likelihood under the factorized density is a product of per-factor terms, each of which is unremarkable, so it certifies the query and the planner fails. Our method instead catches the query at the joint-support step because the k-NN radius and pairwise grid coverage encode the joint constraint. We expect the density-ratio detector to sit close to AnyGoal on joint-failure subsets while matching SCC on genuinely marginal failures, so the diagnostic signal is a gap concentrated on the joint-support subset rather than uniform across all shifts.

**vs. Per-factor box check** — On a query where each factor coordinate lies inside its own box but the combination is off the training manifold, the per-factor check certifies it because it only asks marginal membership and never composes factors, so the planner receives an undecodable goal. Our method instead passes the query through the joint-support estimator, which finds the latent outside the k-NN radius or in an uncovered factor-pair cell, rejecting it with a witness. We expect the per-factor check to match SCC on queries that fail marginally and to diverge sharply on the joint-failure subset, which is the direct evidence that joint support rather than box membership carries the certification.

**vs. Learned joint density estimator (normalizing flow)** — On the joint-failure queries, the normalizing flow may assign high likelihood if it overfits to the training manifold or underfits the joint constraint; if it matches SCC, the nonparametric estimator is not uniquely necessary. We expect the flow to perform between the per-factor check and SCC, with SCC showing a modest advantage on the most heavily shifted subsets due to its nonparametric coverage guarantee.

**vs. SCC with per-factor boxes only (ablation)** — On the joint-failure queries above, the ablation behaves like the per-factor check because it drops exactly the joint-support estimator, certifying jointly-unsupported goals and letting them reach the planner. Full SCC instead rejects them via the joint-support test, so the certified-query success rate rises while coverage on genuinely supported queries stays flat. We expect the ablation to reproduce the per-factor baseline's failure pattern and full SCC to close most of the gap, confirming that the joint-support estimator — not the rejection policy or the witness machinery — is what produces the gain.

### What would falsify this idea

If SCC's advantage over the per-factor and density-ratio baselines were uniform across all goal-shifted subsets rather than concentrated on the joint-support-failure subset, the claim that joint support captures the failure would be wrong. Equivalently, if the joint-estimator ablation matched full SCC on the joint-failure subset, the load-bearing step would be doing nothing and the central mechanism claim would fail.

## References

1. AnyGoal: On World Model Generalization Beyond Expert Rollouts
