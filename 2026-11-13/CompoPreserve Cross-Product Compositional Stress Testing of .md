# CompoPreserve: Cross-Product Compositional Stress Testing of World Preservation in Video Generators

## Motivation

WorldDraft (If Code Builds the World, Can Video Models Just Render It?) organizes its 432 paired cases by six independent dynamic challenge categories and reports world preservation per category, and Cosmos treats physical dynamics as a monolithic prior; both structurally assume that preservation of one dynamic is informative about preservation under composition, yet neither constructs the cross-product of challenge categories, so it is unknown whether preservation composes or collapses when collision, fluid dynamics, and deformable-object coupling occur simultaneously. This is a shared residual gap across BlenderAlchemy, Cosmos, VPO, and WorldDraft: the field's evaluation trajectory has moved from single-category fidelity to multi-category realism but has never tested the compositional algebra of preservation itself, so a model that scores well per category may silently fail on the families where those categories interact.

## Key Insight

World preservation is a compositional quantity whose behavior under category cross-product is governed by an interaction-order structure — if preservation were a separable per-category property, per-category scores would predict composed scores with zero residual, so the residual of a fitted additive model on the cross-product is itself the measurable signal that preservation does not compose.

## Method

**What it is.** CompoPreserve is a benchmark-construction and analysis protocol that takes WorldDraft's six dynamic challenge categories, builds executable 4D world families for every cross-product family up to order 3, generates strictly paired draft/prompt test cases per family, and reports a *composition residual* — the gap between a model's measured preservation on a composed family and the prediction of a regime-matched additive per-category model fitted on the same model's single-category scores after partialling out scene-difficulty and tracker-observability covariates.

**Load-bearing assumption (stated explicitly).** The composition residual r(M,S) is claimed to isolate coupling-induced non-separability only under the assumption that the single-category families used to fit the additive model are matched to each composed family on (i) scene difficulty, operationalized as the per-clip contact-count distribution, and (ii) tracker observability, operationalized as the per-clip tracker-confidence distribution. If these are not matched, r(M,S) absorbs a regime shift (composed scenes are harder and less observable) rather than an interaction. The protocol therefore enforces this matching by construction (Phase 1b) and by covariate adjustment (Phase 4), and validates it with a tunable-coupling synthetic control (Phase 6).

**How it works.**

```pseudocode
INPUT: categories C = {collision, fluid, deformable, gravity, friction, articulation}
       orders k in {1, 2, 3}
       draft granularity g in {abstract, mid, previs}  # WorldDraft 3x3 factorial
       appearance alignment a in {neutral, conflicting}

PHASE 1 - Family synthesis (extends WorldDraft's 48 families)
  for each subset S of C with |S| = k:
      build executable 4D world family F_S with code-defined kinematics for ALL
      categories in S simultaneously (shared solver timestep dt = 1/240 s;
      categories coupled through the same rigid/fluid/deformable state)
      reject F_S if any category in S is inert (contact count < 5 per clip)
  -> yields |C| + C(6,2) + C(6,3) = 6 + 15 + 20 = 41 composed families

PHASE 1b - Difficulty/observability matching (NEW, enforces load-bearing assumption)
  for each composed family F_S and each single-category family F_{c}:
      compute per-clip contact-count vector n(F) and tracker-confidence vector q(F)
      (tracker confidence = mean per-frame detection score of the automatic tracker)
  for each composed family F_S, construct a matched single-category reference set
      Ref(S) = { (F_{c}, clip subset) : c in S, clip subset chosen by 1:1 nearest-neighbor
                 matching on (n, q) to F_S clips, without replacement }
      reject F_S if any c in S has < 5 matched clips in Ref(S)
  -> each composed family now has a difficulty- and observability-matched
     single-category reference set of the same size

PHASE 2 - Paired case generation
  for each F_S, for each (g, a) in 3x3:
      emit case = (code-built world spec, text prompt) with ground-truth
      trajectories + camera path recorded from the executable world

PHASE 3 - Model evaluation
  for each model M in 12 representative video models:
      for each case: generate video V, compute preservation score
      p(M, S, g, a) = 1 - normalized trajectory+camera error(V, GT)
      also record tracker confidence q(V) per case
  pre-registered exclusion rule: drop any case with q(V) < 0.5 (tracker failure);
      report excluded fraction per family; if > 20% excluded, flag family as
      tracker-limited and report its residual separately

PHASE 4 - Regime-matched composition residual
  fit additive model on matched single-category reference scores by least squares,
  with difficulty and observability covariates partialled out:
      p(M, {c}) ~ w_{M,c} * p(M, {c}) + beta_1 * n + beta_2 * q + eps
  then predict composed score from the fitted per-category weights and the
  composed family's own (n, q):
      p_hat(M, S) = sum_{c in S} w_{M,c} * p(M, {c}) + beta_1 * n(S) + beta_2 * q(S)
  residual r(M, S) = p(M, S) - p_hat(M, S)
  report residual matrix R[M, S] and its rank / sign structure
  flag a model as NON-COMPOSING if mean |r| over k=3 exceeds 2x the
  bootstrap std of r on held-out matched single-category cases

PHASE 5 - Interaction attribution
  for each composed family S with large negative r, ablate one category at a
  time (re-render F_{S\{c}}) and attribute the residual drop to category c
  via Shapley value over the |S| ablations (exact for |S| <= 3)

PHASE 6 - Tunable-coupling synthetic control (NEW, validates residual identifies coupling)
  build a synthetic family F_S(theta) where theta in {0, 0.25, 0.5, 0.75, 1.0}
  scales the coupling strength between two categories (e.g., fluid-to-deformable
  force multiplier), holding contact-count and tracker-confidence distributions
  fixed across theta by construction
  for each theta, compute r(M, S(theta)) with the Phase 4 procedure
  verify r(M, S(theta)) is monotone decreasing in theta (Spearman rho < -0.8)
  and that r(M, S(0)) is within bootstrap noise of zero
```

**Why this design.** We chose to build composed families by coupling categories through a *shared physical state* (one solver timestep, one contact manifold) rather than by concatenating WorldDraft's independent single-category clips, because concatenation would make preservation trivially additive and the residual would be zero by construction, accepting the cost that family synthesis is far more expensive and some cross-products are physically unrealizable and must be rejected. We chose an additive least-squares model over a learned interaction predictor because a learned predictor would absorb the very residual we want to measure, accepting the cost that the additive model is misspecified if preservation is genuinely multiplicative. We chose order-3 truncation over full 2^6 cross-product because order-3 already contains every pairwise interaction and the 41 families keep evaluation cost within a single-GPU-week budget, accepting the cost that higher-order interactions (e.g., four-way coupling) are invisible. We chose exact Shapley attribution over leave-one-out because leave-one-out conflates a category's marginal effect with its interaction effect, accepting the cost of |S|! re-renders at order 3. We added Phase 1b and the (n, q) covariates in Phase 4 because without them the residual is not identified: a composed scene is both harder and less observable than a single-category scene, so an unmatched additive fit would absorb the regime shift as a spurious negative residual. We added Phase 6 because a monotone response of r to a tunable coupling parameter, with contact count and tracker confidence held fixed, is the direct test that r tracks coupling rather than difficulty or tracker failure. A domain expert would not describe this as a variant of WorldDraft: WorldDraft's evaluation is a per-category scorecard, whereas CompoPreserve's object of measurement is the *residual of a fitted additive model after difficulty and observability are partialled out*, a quantity WorldDraft never computes.

**Why it measures what we claim.** The composition residual r(M,S) measures *non-composability of world preservation* because the additive model p_hat is the maximum-likelihood prediction under the assumption that preservation is a separable per-category property, and the (n, q) covariates partial out the regime shift so that r reflects coupling-induced degradation rather than scene difficulty or tracker failure; this assumption fails when the matched reference set is too small to estimate the (n, q) coefficients (fewer than 5 matched clips per category), in which case r reflects covariate-estimation noise and we flag the family as unmatched. The preservation score p(M,S,g,a) measures *world preservation* because it is the normalized trajectory-and-camera error against executable ground truth; this assumption fails when the automatic tracker loses the object (occlusion by fluid), in which case p reflects tracker failure rather than model failure — we mitigate by the pre-registered q(V) < 0.5 exclusion rule and by reporting the excluded fraction per family. The Shapley attribution measures *which category drives the interaction* because exact Shapley values are the unique attribution satisfying efficiency and symmetry over the ablation game; this assumption fails when two categories are perfectly correlated in the family (e.g., friction and collision always co-occur), in which case Shapley splits credit arbitrarily and we report the pair jointly. The tunable-coupling control (Phase 6) measures *that r tracks coupling* because theta scales only the coupling force while (n, q) are held fixed by construction; this assumption fails when the synthetic family's solver becomes unstable at high theta, in which case r reflects solver divergence and we reject that theta value.

## Contribution

(1) A composed-family benchmark of 41 executable 4D world families spanning all order-1, order-2, and order-3 cross-products of six dynamic challenge categories, with strictly paired draft/prompt cases under WorldDraft's 3x3 factorial. (2) The composition residual — a per-model, per-family quantity defined as the gap between measured and additive-predicted world preservation — together with an exact Shapley attribution over category ablations that localizes which category drives non-composability. (3) An empirical design principle: a model's single-category preservation score is not a sufficient statistic for its composed-family preservation, and the residual matrix exposes a rank structure that per-category scorecards cannot reveal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CompoPreserve 41 composed families (orders 1–3) with Phase 1b matched single-category reference sets | Coupled shared-state families make residual non-trivial; matching enforces the load-bearing assumption |
| Primary metric | Regime-matched composition residual r(M,S) with (n, q) covariates partialled out | Directly tests non-separability after difficulty and tracker-observability are controlled |
| Baseline | WorldDraft per-category scorecard | The additivity assumption we claim fails |
| Baseline | Additive-fit predictor without (n, q) covariates (no ablation) | Tests whether residual is just regime-shift misspecification |
| Baseline | Concatenated-clip families, matched on (n, q) | Control where residual should be near zero |
| Ablation-of-ours | Leave-one-out attribution instead of Shapley | Isolates whether Shapley is doing real work |
| Control | Tunable-coupling synthetic family F_S(theta), theta in {0, 0.25, 0.5, 0.75, 1.0} | Validates r scales monotonically with coupling strength at fixed (n, q) |

### Why this setup validates the claim

The central claim is that world preservation is non-composable: a model's score on a composed family deviates systematically from the regime-matched additive prediction built from its single-category scores. The WorldDraft scorecard baseline embodies exactly the separability assumption we are attacking, so if its per-category scores predict composed scores well, our residual is noise and the claim dies. The additive-fit baseline without (n, q) covariates is the confound check: if the residual shrinks to zero once difficulty and observability are partialled out, then the original residual was a regime shift, not coupling. The concatenated-clip control, now matched on (n, q), is the sharpest falsifier: because concatenation severs shared physical state, the residual there must be near zero by construction — if we see large residuals on matched concatenated families too, our residual is measuring something other than coupling. The tunable-coupling synthetic control is the direct validation that r tracks coupling: theta scales only the coupling force while contact count and tracker confidence are held fixed, so a monotone decrease of r with theta (Spearman rho < -0.8) and r near zero at theta=0 is the observable signature that r is not a difficulty or tracker artifact. Leave-one-out versus Shapley isolates whether attribution depends on the interaction structure we claim exists. The composition residual is the right metric because it is defined relative to each model's own matched single-category performance, so it cancels model-level capability differences and surfaces only the coupling-induced gap.

### Expected outcome and causal chain

**vs. WorldDraft per-category scorecard** — On a family where a fluid jet drives a deformable cloth into a collision, the scorecard predicts composed preservation as a weighted sum of the three single-category scores because it assumes each category is evaluated in isolation. Our method instead renders the same coupled world with one shared solver state and partials out the (n, q) regime shift, so the fluid-deformable-collision feedback degrades trajectories in a way no single-category score captures. We expect a pronounced negative residual concentrated on fluid-plus-deformable-plus-collision families, with near-zero residual on families whose categories are only weakly coupled.

**vs. Additive-fit predictor without (n, q) covariates** — On order-3 families, the least-squares additive fit without covariates can absorb some coupling into its weights and some regime shift into its intercept, so a naive residual might shrink toward zero or be uniformly negative. Our method re-fits weights on held-out matched single-category cases with (n, q) partialled out and reports the residual against that held-out prediction, so absorbed coupling and regime shift cannot masquerade as good fit. We expect the covariate-adjusted held-out residual to remain clearly negative on strongly coupled families while the unadjusted in-sample fit looks deceptively good — a gap between unadjusted and adjusted residual is the signature that the (n, q) covariates are doing real work.

**vs. Concatenated-clip families (matched on (n, q))** — On a family built by concatenating independent single-category clips and matched to the composed family on contact count and tracker confidence, the additive prediction is correct by construction because the clips share no physical state. Our method's residual on these matched concatenated families should therefore sit within bootstrap noise of zero, while the shared-state families show a clear negative shift. We expect a bimodal residual distribution: tight around zero for matched concatenated families, shifted negative for shared-state families — the separation is the observable signal.

**vs. Leave-one-out attribution** — On a composed family where friction and collision interact, leave-one-out attributes the drop to whichever category has the larger marginal effect and conflates interaction with main effect. Our method's exact Shapley values over the ablation game split credit according to the interaction structure, so the attributed category ordering should differ specifically on families with strong pairwise coupling and agree on weakly coupled families. We expect Shapley and leave-one-out to diverge only where the residual is large, confirming attribution tracks interaction rather than marginal effect.

**vs. Tunable-coupling synthetic control** — On the synthetic family F_S(theta), theta scales only the coupling force while contact count and tracker confidence are held fixed by construction. If r is a coupling signal, it should decrease monotonically with theta (Spearman rho < -0.8) and sit within bootstrap noise of zero at theta=0. If r is flat in theta or already negative at theta=0, the residual is a difficulty or tracker artifact and the central claim is not supported.

### What would falsify this idea

If the regime-matched composition residual on shared-state families is statistically indistinguishable from that on matched concatenated-clip families, or if the residual does not decrease monotonically with the tunable coupling parameter theta at fixed (n, q), then preservation is effectively separable or the residual is a regime artifact and the central non-composability claim is wrong. A second falsifier: if the covariate-adjusted held-out residual and the unadjusted in-sample residual coincide everywhere, the additive model is not misspecified and there is no interaction to attribute.

## References

1. WorldDraft: If Code Builds the World, Can Video Models Just Render It?
