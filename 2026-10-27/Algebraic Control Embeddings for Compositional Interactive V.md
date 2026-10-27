# Algebraic Control Embeddings for Compositional Interactive Video World Models

## Motivation

In-context conditioning frameworks such as World-in-Context treat view, action, and camera tokens as a flat self-attention context, so the latent transition induced by a control sequence is learned implicitly and carries no guarantee that composing controls composes their effects; the embedding space is never constrained to be a homomorphism of the control algebra. This is a Forest-level structural gap: the same flat-context treatment recurs across in-context learner and FullDiT2-style branches, meaning any control combination unseen during training (e.g., simultaneous view switch plus camera pan plus action) has no principled reason to produce the correct latent transition, and the model must memorize combinations rather than compose them. Because no structural constraint ties token embeddings to the group structure of control operations, generalization to unseen compositions is left entirely to data coverage.

## Key Insight

If control-token embeddings are constrained to be a group homomorphism from the control algebra (view/action/camera operations with their composition and identity) into a structured latent space, then the latent transition for any unseen composition is forced to equal the algebraic composition of the individual transitions, making compositional generalization a structural consequence rather than a data-coverage accident.

## Method

**What it is.** We introduce **AlgebraFormer**, an interactive video world model whose control tokens (view, action, camera) are embedded as elements of a structured latent group. Inputs are a video latent history plus a control sequence; outputs are predicted future latents. Each control operation `g` maps to a latent group element `φ(g)` such that `φ(g₁ ∘ g₂) = φ(g₁) · φ(g₂)` and `φ(identity) = e`, where `·` is a learned but fixed group operation on the latent space.

**Load-bearing assumption (stated explicitly).** The method assumes that (i) the discretized control set is closed under composition up to binning, and (ii) the latent group element `φ(g)` carries the compositional effect of `g` for the non-commuting view/action/camera compositions that constitute the headline use case, not merely an abelian projection of them. The original commuting-Cartan parameterization violates (ii) by construction: a homomorphism from a non-abelian group into a commutative one has the commutator subgroup in its kernel, so all non-commutative structure is invisible to `φ`. To make the assumption testable rather than hidden, the revised method (a) replaces the single commuting Cartan parameterization with a non-commutative Lie-group parameterization (full `so(d)` generators per factor, composed by actual matrix multiplication) and (b) treats the control set as a groupoid/monoid with explicit non-invertible and saturating elements, reporting the closure-violation rate. The residual MLP is retained but its contribution to held-out compositional gains is measured by ablation (see Phase 4 and the experiment).

**How it works.**

```pseudocode
# Phase 1: Control algebra definition
#   Control operations form a groupoid G = G_view × G_action × G_camera
#   G_view   : {identity, switch_to_view_k}  (finite set, k=1..K_view, K_view=8)
#   G_action : SE(2)-like translation/rotation generators, discretized to K_action bins (K_action=16)
#   G_camera : SE(3)-like pan/tilt/zoom generators, discretized to K_camera bins (K_camera=16)
#   Composition g1 ∘ g2 = apply g2 then g1 (matrix multiplication per factor)
#   Non-invertible/saturating elements: camera pan/tilt/zoom saturate at ±S (S=2.0 in normalized units);
#     such elements are marked as non-invertible and composition is defined only when the result stays in range.
#   Closure check: for all sampled pairs (g1,g2), compute bin(g1∘g2) and record whether it equals the composed bin;
#     report closure-violation rate = fraction of pairs where bin(g1∘g2) ≠ bin(g1)∘bin(g2).

# Phase 2: Non-commutative homomorphic embedding
#   Latent dimension d=32 (so expm is cheap; precompute φ_g for all discrete bins once per forward pass).
#   For each factor f in {view, action, camera}:
#     e_f = E_f[g.f]                     # lookup, dim d_alg=8 per factor (total 24, padded to d=32)
#     A_f = sum_{i=1}^{d_alg} e_f[i] * T_f[i]   # T_f[i] are fixed so(d) generators (skew-symmetric d×d matrices)
#     # T_f[i] are chosen so that different factors use distinct, non-commuting subalgebras:
#     #   T_view[i] act on a 2D subspace, T_action[i] on a disjoint 2D subspace, T_camera[i] on a third 2D subspace,
#     #   with cross-factor commutators [T_f[i], T_h[j]] ≠ 0 for f≠h (verified numerically at init).
#   A_total = A_view + A_action + A_camera
#   phi_g = expm(A_total)                # d×d orthogonal matrix => group element under matrix multiply
#   # Homomorphism is NOT enforced by construction (since [A_f, A_h] ≠ 0); it is enforced by the
#   # compositionality loss in Phase 4, which uses the Baker-Campbell-Hausdorff (BCH) correction:
#   #   log(expm(A)expm(B)) = A + B + 1/2[A,B] + 1/12[A,[A,B]] - 1/12[B,[A,B]] + ...
#   #   We truncate at second order and add a learned correction MLP_corr on the commutator terms.

# Phase 3: Latent transition
for each timestep t:
    h_t = phi_{g_t} @ h_{t-1}          # group action on latent state
    h_t = h_t + MLP_residual(h_t)      # small residual, weight 0.1, for non-group content

# Phase 4: Compositionality loss (training)
#   Sample triples (g1, g2, g1∘g2) from G, including non-commuting pairs (e.g., view switch then camera pan vs. reverse).
#   L_comp = || phi_{g1∘g2} - phi_{g1} @ phi_{g2} ||_F^2
#   L_id   = || phi_identity - I ||_F^2
#   L_bch  = || log(phi_{g1} @ phi_{g2}) - (A_1 + A_2 + 1/2[A_1,A_2] + 1/12[A_1,[A_1,A_2]] - 1/12[A_2,[A_1,A_2]] + MLP_corr([A_1,A_2])) ||_F^2
#   L_pred = video diffusion / flow-matching loss on predicted latents
#   L = L_pred + lambda_comp * (L_comp + L_id + lambda_bch * L_bch)   # lambda_comp = 0.5, lambda_bch = 0.1

# Phase 5: Held-out composition evaluation
#   Train on pairs (g1, g2); test on g1∘g2 never seen jointly.
#   Report: held-out composition prediction loss, closure-violation rate, residual-ablation gap (full vs. residual-removed).
```

**Why this design.** We chose a *non-commutative Lie-group parameterization with per-factor generators in distinct so(d) subalgebras and a BCH-based composition loss* over a plain MLP embedding because a plain MLP cannot guarantee `φ(g₁∘g₂)=φ(g₁)φ(g₂)`, and over the original commuting-Cartan parameterization because a homomorphism into an abelian subgroup has the commutator subgroup in its kernel, making non-commuting view/action interactions invisible to `φ`; the cost is that exact homomorphism is no longer by construction and must be enforced by `L_comp` and `L_bch`. We chose the *compositionality loss on sampled triples* over only architectural enforcement because architectural enforcement alone cannot correct discretization/binning errors at the group level, accepting the cost of extra sampling compute per step. We chose *separate per-factor embedding tables* over a single joint table because separate tables let the homomorphism constraint factorize per algebra generator, accepting the cost that cross-factor interactions must be learned by the BCH correction and the residual MLP. Unlike **World-in-Context**, which feeds control tokens as flat context and lets self-attention infer transitions, AlgebraFormer makes the transition operator itself a group element, so a domain expert would not describe it as a variant of in-context conditioning: the constraint lives in the transition operator, not in the attention context.

**Why this measures what we claim.** The Frobenius norm `||φ_{g1∘g2} − φ_{g1}φ_{g2}||_F` measures *compositional consistency* because the assumption that the control set forms a groupoid under the defined composition makes homomorphism violation exactly the algebraic defect; this assumption fails when real control sequences are non-invertible or noisy (e.g., a camera pan that saturates at a boundary), in which case the metric reflects binning mismatch rather than true algebraic inconsistency. The identity loss `||φ_identity − I||_F` measures *identity preservation* because the assumption that the identity control produces no latent change makes the identity element the group unit; this fails when the video has autonomous dynamics (moving objects), in which case the metric conflates control-identity with scene-motion. The held-out composition prediction loss measures *compositional generalization* because the assumption that held-out pairs are truly unseen compositions (not merely unseen token pairs) makes the test a genuine algebraic extrapolation; this fails if the training set already contains the individual factors in correlated contexts, in which case the metric reflects interpolation rather than composition. The closure-violation rate measures *discretization fidelity* because the assumption that the discretized control set is closed under composition makes any violation a binning artifact; this fails when saturation or non-invertibility dominates, in which case the rate reflects the groupoid structure rather than the algebra. The residual-ablation gap measures *the group constraint's causal contribution* because the assumption that the residual MLP contributes negligibly to held-out transitions makes the gap attributable to the group element; this fails when the residual absorbs error, in which case the gap shrinks and the group constraint is not the driver.

## Contribution

['(1) A homomorphic control-embedding framework (AlgebraFormer) that maps view/action/camera tokens into a structured latent group via a commuting Lie-algebra parameterization, with a compositionality loss enforcing algebraic consistency.', '(2) A design principle: constraining the latent transition operator to be a group element, rather than the token embedding to be flat context, is what enables generalization to unseen control compositions.', '(3) A held-out composition protocol for interactive video world models that separates interpolation from algebraic extrapolation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WBench navigation sequences | Matches baseline benchmark for direct comparison |
| Primary metric | Held-out composition prediction loss | Directly probes algebraic extrapolation, not interpolation |
| Secondary metric | Closure-violation rate | Quantifies discretization fidelity of the control groupoid |
| Secondary metric | Residual-ablation gap | Isolates group constraint from free-form correction |
| Baseline | World-in-Context-5B | Tests in-context conditioning vs operator-level constraint |
| Baseline | 22B Ec model | Tests whether scale alone closes compositional gap |
| Baseline | MLP-embedding control (no group) | Tests whether homomorphism structure matters at all |
| Ablation-of-ours | AlgebraFormer minus residual MLP | Isolates group enforcement from free-form correction |
| Ablation-of-ours | AlgebraFormer with commuting-Cartan parameterization | Tests whether non-commutative parameterization is necessary |

### Why this setup validates the claim

The central claim is that placing the control transition operator inside a structured latent group yields compositional generalization that flat in-context conditioning cannot reach. WBench navigation is the right dataset because it contains the exact failure trigger we care about — sequences of view switches, actions, and camera moves whose pairwise compositions are semantically meaningful but whose joint tokens are rare. World-in-Context-5B tests the sub-claim that attention-based context inference alone cannot enforce homomorphism; the 22B Ec baseline tests whether the effect is merely a capacity artifact; the MLP-embedding baseline tests whether the gain comes from the group structure specifically rather than from having any learned control embedding. The primary metric — prediction loss on held-out compositions — is the right detector because it is the only measurement that separates algebraic extrapolation from interpolation: a method that merely memorizes correlated factor contexts will look fine on seen pairs but degrade sharply here. The closure-violation rate and residual-ablation gap are added because the load-bearing assumption (that the discretized control set is closed under composition and that the residual MLP contributes negligibly to held-out transitions) must be verified, not assumed; without them, a gain could be attributed to binning artifacts or to the residual escape hatch. The ablation removes the residual MLP so we can attribute any gain to the group constraint rather than to a free-form corrector, and the commuting-Cartan ablation tests whether the non-commutative parameterization is necessary for the non-commuting compositions that constitute the headline use case.

### Expected outcome and causal chain

**vs. World-in-Context-5B** — On a case where a view switch is followed by a camera pan whose composition was never jointly observed, the baseline produces a latent transition that blends the two operations as context tokens and lets self-attention average them, because its mechanism has no operator that must satisfy φ(g₁∘g₂)=φ(g₁)φ(g₂). Our method instead applies the composed group element directly to the latent state because the transition operator is the group element by construction, so we expect a clear gap on held-out-composition sequences but rough parity on single-factor sequences.

**vs. 22B Ec model** — On the same held-out composition, the larger baseline produces a smoother but still non-compositional transition because scale improves in-context inference without adding an algebraic constraint on the operator. Our method instead carries the constraint in the transition itself, so the residual error does not shrink with parameter count, and we expect our smaller model to match or beat the 22B baseline specifically on the composition subset while trailing it on raw single-step fidelity.

**vs. MLP-embedding control** — On a case where two controls compose to a third discretized bin, the MLP baseline produces a φ for the composite that is unrelated to the product of the factor embeddings because nothing ties them, so the composition loss stays high. Our method instead ties them through the non-commutative Lie-group parameterization and BCH-based composition loss, so we expect the composition-consistency norm to drop by roughly an order of magnitude relative to the MLP baseline while identity loss stays near the same floor.

**vs. AlgebraFormer minus residual MLP** — On held-out compositions, the residual-removed ablation tests whether the group element alone carries the compositional effect; if the full model's gain over the ablation is small (residual-ablation gap < 5% of the held-out loss), the group constraint is the causal driver; if the gap is large, the residual is absorbing error and the algebraic guarantee is not doing the work.

**vs. AlgebraFormer with commuting-Cartan parameterization** — On non-commuting compositions (e.g., view switch then camera pan vs. reverse), the commuting-Cartan ablation cannot distinguish order because its φ is abelian by construction; our non-commutative parameterization should show a clear gap on order-sensitive held-out pairs, confirming that the non-abelian structure is necessary for the headline use case.

### What would falsify this idea

If our gain over the MLP-embedding baseline is uniform across seen and held-out composition subsets rather than concentrated on held-out compositions, then the group structure is not doing the algebraic work we claim and the improvement is just a better-conditioned embedding. Likewise, if the residual-removed ablation matches the full model on held-out compositions, the group constraint is not the causal driver; and if the closure-violation rate is high (>10%), the groupoid assumption fails and the held-out metric reflects binning artifacts rather than algebraic extrapolation.

## References

1. World-in-Context: Interactive Video World Modeling with In-Context Conditioning
