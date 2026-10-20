# Compositional Identifiability of Shared Goal Latents via Permutation-Invariant, Composition-Closed World Models

## Motivation

AnyGoal shows that current world models systematically fail when familiar state factors and actions are recombined, because their shared latent goal representations are learned from expert rollouts that never exhibit those recombinations and therefore carry no constraint tying the latent to the compositional structure of goals. This is a meta-level failure recurring across PWM-style shared-latent world models, Lejepa-style joint-embedding predictors, and interaction-asymmetry formulations: each learns a latent that is only identifiable up to an arbitrary, task-specific reparameterization, so compositional goal shifts land outside the latent's support. The structural root cause is that no existing objective imposes identifiability under factor permutation and composition, so the latent is free to entangle factors in whatever way fits the training support.

## Key Insight

A shared goal latent is compositionally identifiable exactly when its encoder is equivariant to factor permutations and its decoder is closed under factor composition, a pair of algebraic constraints that pin the latent to the goal's factor structure up to a single global permutation, independent of which factor combinations expert rollouts happen to cover.

## Method

**What it is.** CIGL (Compositionally Identifiable Goal Latent) is a training objective and inference procedure that augments a world model's shared goal encoder `E` and decoder `D` with two algebraic constraints — permutation equivariance and composition closure — so the latent `z = E(goal)` is identifiable up to one global factor permutation. Inputs: (goal, transition) pairs from AnyGoal's goal-shifted benchmark. Outputs: a latent encoder/decoder pair whose latent support is closed under composition, plus a per-factor slot assignment.

**Load-bearing assumption (stated explicitly).** CIGL's composition-closure term `L_span` assumes that the goal composition operator is linear in the latent slot basis: for disjoint-support goals `g_a`, `g_b`, the composed latent `z_ab[k]` lies in the linear span of `z_a[k]` and `z_b[k]` for each slot `k`. This assumption is load-bearing because `L_span` is a linear-subspace projection; if the true composition law is nonlinear (e.g., multiplicative or conjunctive goal factors), `L_span` either becomes vacuous (when `E` is already linear per slot) or distorts the latent geometry to satisfy a false algebraic identity. We therefore (i) state the assumption here, (ii) verify it empirically on held-out compositions before interpreting `L_span` as a closure measure (see Phase 0 below), and (iii) restrict the identifiability claim to the factor-permutation part (`L_slot` + `L_perm`) when the linearity check fails.

**How it works.**

```pseudocode
# Phase 0: linearity-premise calibration (run once before Phase 1)
# Fit a per-slot linear probe from (z_a[k], z_b[k]) to z_ab[k] on a calibration set of 512 held-out compositions.
# Report R^2 per slot. If median R^2 >= 0.8, treat L_span as a valid closure proxy.
# If median R^2 < 0.8, set lambda_span = 0 and restrict claims to L_slot + L_perm (permutation identifiability only).

# Phase 1: factor-slot discovery via contrastive factorization
for each minibatch of (goal g, next-state s') from AnyGoal:
    z = E(g)                      # shape [K, d], K factor slots; K estimated by elbow on held-out split (see Phase 3)
    # slot-wise contrastive loss: each slot must predict its own state factor
    L_slot = sum_k InfoNCE(z[k], s'[k], negatives=shuffled_batch)  # tau=0.1, batch=256
    # permutation-equivariance: permuting factor order of g must permute z identically
    L_perm = || E(pi(g)) - pi(E(g)) ||^2   for random pi in S_K  # 4 random permutations per minibatch

# Phase 2: composition-closure regularizer (gated by Phase 0)
sample g_a, g_b with disjoint factor supports (AnyGoal guarantees these exist)
    z_ab = E(compose(g_a, g_b))
    L_comp = || D(z_ab) - compose(D(z_a), D(z_b)) ||^2
    # closure: z_ab must lie in the span of {z_a, z_b} slot-wise
    L_span = sum_k || proj_{span(z_a[k], z_b[k])}(z_ab[k]) - z_ab[k] ||^2

# Phase 3: joint training
L = L_dynamics + lambda_slot*L_slot + lambda_perm*L_perm + lambda_comp*L_comp + lambda_span*L_span
# lambda_slot=1.0, lambda_perm=0.5, lambda_comp=1.0, lambda_span=0.5 (tuned on held-out AnyGoal splits; lambda_span set to 0 if Phase 0 fails)
# K estimation: elbow criterion on held-out reconstruction of s' from z; sweep K in {2,3,4,5,6}; report sensitivity in experiment.

# Inference: compositional goal shift
z_new = E(compose(g_a, g_b))   # guaranteed in-support by L_span when Phase 0 passes
plan with D(z_new) as goal-conditioned reward
```

**Why this design.** We chose a slot-wise contrastive objective over a monolithic reconstruction loss because reconstruction from expert rollouts (the AnyGoal training distribution) does not constrain how factors align across samples, accepting the cost that slot assignment requires K to be specified or estimated by a held-out elbow criterion. We chose an explicit permutation-equivariance penalty over architectural weight-sharing across slots because weight-sharing forces equivariance only for the architecture's symmetry group and cannot express factor-specific dynamics, accepting the cost of an extra loss term and a random-permutation sampler. We chose a span-projection closure term over simply training on composed goals because composed goals are exactly what AnyGoal shows are out-of-support, so training on them would leak the test distribution; the span term instead constrains the latent geometry without requiring composed supervision, accepting the cost that it only enforces closure up to the span of observed factor latents — and, as stated in the load-bearing assumption above, only when the goal composition operator is linear in the latent slot basis. Unlike a standard information-bottleneck or β-VAE disentanglement objective — the closest named prior — CIGL does not rely on statistical independence of latent coordinates, which expert rollouts cannot supply; it relies on the algebraic closure of the latent under a composition operator that is defined by the goal space, not by the data.

**Why it measures what we claim.** The slot-wise InfoNCE term `L_slot` measures factor alignment because the assumption that each goal factor independently determines one state factor makes the slot-to-factor map a bijection; this assumption fails when two goal factors jointly determine a single state factor (e.g., position and velocity jointly determining reachability), in which case the term reflects predictive sufficiency rather than factor identity. The permutation penalty `L_perm` measures equivariance because the assumption that the goal space admits a well-defined factor permutation action makes `E(pi(g)) = pi(E(g))` a necessary condition; this assumption fails when the factor set is not closed under permutation (e.g., a goal with a variable number of objects), in which case it measures only approximate symmetry. The span-projection term `L_span` measures composition closure because the assumption that composed goals lie in the linear span of their factor latents makes out-of-span mass a direct violation of closure; this assumption fails when composition is nonlinear in latent space (e.g., multiplicative or conjunctive goal factors), in which case it measures linear closure only — which is why Phase 0 calibrates the linearity premise (median per-slot R^2 >= 0.8 on 512 held-out compositions) before `L_span` is interpreted as a closure measure, and sets `lambda_span = 0` otherwise. Together these three quantities operationalize the identifiability condition — invariant to permutation and closed under composition — that the motivation claims is missing from AnyGoal's shared latent.

## Contribution

(1) CIGL, a training objective combining a slot-wise contrastive factorization, a permutation-equivariance penalty, and a span-projection composition-closure regularizer that together impose compositional identifiability on a world model's shared goal latent. (2) The design principle that composition closure can be enforced without composed-goal supervision by constraining latent geometry to the span of observed factor latents, which avoids leaking AnyGoal's out-of-support test distribution. (3) An inference procedure that maps compositional goal shifts into the identifiable latent support, evaluated on AnyGoal's multi-task benchmark across world-model families.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AnyGoal goal-shifted benchmark | Exactly the out-of-support compositional split under test |
| Dataset (secondary) | Nonlinear-composition split of AnyGoal (multiplicative/conjunctive factor pairs) | Tests where linear closure premise breaks |
| Primary metric | Compositional generalization success rate | Directly measures planning on held-out factor compositions |
| Secondary metric | Per-slot linear-probe R^2 of z_ab on span(z_a, z_b) | Calibrates the linearity premise (Phase 0) |
| Baseline | AnyGoal shared-latent world model | Tests whether identifiability is the missing ingredient |
| Baseline | β-VAE disentanglement world model | Tests algebraic closure against statistical independence prior |
| Baseline | Reconstruction-only goal encoder | Tests slot-contrastive alignment against monolithic reconstruction |
| Ablation-of-ours | CIGL without L_span closure term | Isolates the composition-closure sub-claim |
| Ablation-of-ours | CIGL with L_span on nonlinear-composition split | Tests the load-bearing linearity assumption |
| Sensitivity | K in {2,3,4,5,6} via elbow criterion | Quantifies K-estimation risk |

### Why this setup validates the claim

The central claim is that a latent which is identifiable up to permutation and closed under composition generalizes to out-of-support composed goals where prior world models fail. AnyGoal's goal-shifted benchmark is the right dataset because it is the exact distribution where composed goals are held out, so success there cannot be explained by memorization. The primary metric, compositional generalization success rate on planning, is the right metric because it directly detects whether the latent geometry supports goals never seen in training; a metric like reconstruction loss would not distinguish in-support fit from out-of-support structure. The AnyGoal shared-latent baseline tests the core sub-claim that identifiability is what is missing: if it fails where CIGL succeeds, the algebraic constraints are doing the work. The β-VAE baseline tests the secondary sub-claim that algebraic closure beats statistical independence, since it shares the goal-conditioned setup but relies on a different disentanglement principle. The reconstruction-only baseline tests whether slot-wise contrastive alignment, rather than mere predictive sufficiency, drives the effect. The L_span ablation is the decisive control: it removes only composition closure while preserving slot alignment and permutation equivariance, so any drop isolates closure as the causal ingredient rather than the whole package. The secondary nonlinear-composition split and the per-slot linear-probe R^2 directly test the load-bearing assumption that the goal composition operator is linear in the latent slot basis: if median R^2 falls below 0.8, the closure claim is restricted to the permutation part, and the nonlinear split reports where linear closure breaks.

### Expected outcome and causal chain

**vs. AnyGoal shared-latent world model** — On a case where a composed goal combines a factor value from one training goal with a disjoint factor value from another, the baseline produces a latent that interpolates unpredictably between seen encodings because its shared encoder has no constraint tying factor slots to state factors, so the composed goal lands off the learned manifold. Our method instead places the composed goal inside the span of its factor latents via L_span and aligns each slot to its own state factor via L_slot, so the composed latent is a valid, decodable goal. We expect a large gap in compositional success rate concentrated on the disjoint-support composition subset, with near-parity on in-support goals.

**vs. β-VAE disentanglement world model** — On a case where two goal factors are statistically correlated in the expert rollouts, the baseline produces entangled latent coordinates because its independence prior is violated by the training distribution, so a composed goal shifts unintended factors. Our method instead relies on algebraic closure and permutation equivariance, which do not require statistical independence, so the same composed goal shifts only its intended slot. We expect a moderate but consistent gap on correlated-factor compositions, with the gap shrinking on factors that happen to be independent in the data.

**vs. Reconstruction-only goal encoder** — On a case where a goal factor is weakly predictive of the next state but still compositionally relevant, the baseline produces a latent that underrepresents that factor because reconstruction loss does not force per-factor alignment across samples. Our method instead forces each slot to predict its own state factor via slot-wise InfoNCE, so weakly predictive but compositionally relevant factors remain represented. We expect a gap on compositions involving low-reconstruction-salience factors, with parity on high-salience factors.

**vs. CIGL without L_span** — On a case where a composed goal lies outside the span of observed factor latents, the ablation produces a latent that is slot-aligned but not closed under composition, so decoding yields an inconsistent goal. Our method instead projects out-of-span mass to zero, keeping composed latents in-support. We expect the ablation to retain in-support performance but lose most of the compositional gain, confirming closure as the causal ingredient.

**vs. CIGL with L_span on nonlinear-composition split** — On a case where the true composition operator is multiplicative or conjunctive (e.g., position and velocity jointly determining reachability), the linear-span premise fails (median per-slot R^2 < 0.8 in Phase 0). We expect CIGL with L_span active to show distorted latent geometry and reduced success on this split relative to CIGL with lambda_span = 0, confirming the boundary of the linearity assumption and motivating restriction of the closure claim to the permutation part when the premise fails.

### What would falsify this idea

If CIGL's gain over the AnyGoal shared-latent baseline is uniform across in-support and out-of-support composition subsets rather than concentrated on the disjoint-support compositions where the predicted failure mode occurs, the identifiability mechanism is not what drives the result. If the L_span ablation matches full CIGL on compositional success, composition closure is not the causal ingredient and the central claim is wrong. If Phase 0 reports median per-slot R^2 < 0.8 on the primary AnyGoal split, the linearity premise is false and the closure claim must be restricted to the permutation part.

## References

1. AnyGoal: On World Model Generalization Beyond Expert Rollouts
