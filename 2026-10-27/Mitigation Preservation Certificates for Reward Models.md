# Mitigation Preservation Certificates for Reward Models

## Motivation

Prior mitigations for reward hacking justify their intervention by a property of the reward model—flatness of the optimum in gradient-regularized RLHF (Gradient Regularization Mitigates Reward Hacking) or interpretable-feature support in SAE-based removal (IR3)—but neither establishes that this property survives the mitigation operator itself. This is a structural gap across both trees: the property that justifies the cure is assumed to be closed under the cure, with no checkable condition guaranteeing it. The field therefore lacks a closure theory for mitigation operators, leaving open whether post-mitigation reward models still satisfy the very property that motivated the intervention.

## Key Insight

A mitigation operator preserves the property that justified it exactly when the operator is a projection onto the property's defining algebraic variety, and this preservation is decidable by checking whether the operator's Jacobian annihilates the variety's defining ideal at the pre-mitigation point.

## Method

**What it is.** We introduce **MPC (Mitigation Preservation Certificate)**, a procedure that takes a pre-mitigation reward model $r$ (parameters $\theta$), a mitigation operator $T$ (e.g., KL-reset + gradient-norm penalty, or SAE-feature ablation), and a property $P$ (flatness of the optimum, or interpretable-feature support), and outputs a boolean certificate $\mathrm{Cert}(T,P,r)$ together with a witness (a scalar residual or a counterexample input).

**Load-bearing assumption (stated explicitly).** The certificate is meaningful only when the property set $V_P$ is a smooth algebraic variety whose tangent space at $\theta$ is well-defined and non-trivial, and when $T$ is differentiable at $\theta$. This assumption is false for the naive first-order flatness functional (identically zero at interior optima) and for the $\ell_0$ support functional (a union of coordinate subspaces, not a manifold). We therefore replace the smooth-variety formulation with a **stratified, one-sided definition**: preservation is defined as the directional derivative of $g_P$ along $T$'s actual finite displacement, evaluated on each stratum of $V_P$ separately, and we certify only when $T$ maps each stratum into itself. For flatness we use a second-order measure with the implicit-function correction for $x^*(\theta)$; for support we use the stratum-wise support-change indicator.

**How it works.**

```pseudocode
Input: reward model r (parameters θ), mitigation operator T (maps θ → θ'), property P
Output: certificate bit c ∈ {0,1}, witness w

1. Define property strata V_P^s for s = 1..S:
     For P = flat:  g_flat(θ) = λ_min(∇²_x r_θ(x*(θ)))  (smallest Hessian eigenvalue at the θ-dependent optimum)
                    with x*(θ) obtained by 10-step Newton from the pre-mitigation optimum; strata s index sign(λ_min) ∈ {+, 0, -}.
     For P = supp:  g_supp(θ) = ||φ_supp(θ)||_0 - k  (SAE feature support size)
                    strata s index the exact support set S(θ) = {i : φ_i(θ) ≠ 0}, with |S(θ)| = k.

2. Compute a one-sided Clarke/subgradient differential J_T^s(θ) for each stratum s:
     - For smooth T (KL penalty, gradient regularization): central finite differences with step ε=1e-3.
     - For discontinuous T (KL-reset): one-sided finite differences on each side of the reset boundary, ε=1e-3; report both.
     - For ℓ0 support: subgradient of the support indicator, defined as the set of coordinate directions where support changes.

3. Compute the tangent cone T_θ V_P^s for each stratum:
     - For flat: ker(∇g_flat(θ)) with ∇g_flat including the implicit-function correction
       ∂g_flat/∂θ = ∂g_flat/∂θ|_x* + (∂g_flat/∂x*)(∂x*/∂θ), where ∂x*/∂θ = - (∇²_x r)^{-1} (∂∇_x r/∂θ).
     - For supp: the linear span of coordinate directions that preserve the support set S(θ).

4. Certify preservation on stratum s iff  J_T^s(θ) · v = 0  for all v ∈ T_θ V_P^s,
     i.e., the operator's differential annihilates the stratum's tangent directions.
     Implement as: σ_max(J_T^s(θ) · Π_{T_θ V_P^s}) ≤ τ  (τ=1e-4).
     Additionally require finite-step invariance: T(θ) ∈ V_P^s (checked by evaluating g_P(T(θ)) and comparing stratum label).

5. If certified on all strata, output (1, max_s σ_max^s); else output (0, argmax_{v,s} ||J_T^s(θ) v||).
```

**Why this design.** We chose to certify via the Jacobian of the mitigation operator restricted to the property's tangent space, rather than via direct post-hoc measurement of $P(T(\theta))$, because a post-hoc check cannot distinguish a mitigation that preserves $P$ by construction from one that happens to land near $V_P$ by luck—the former is what a closure theorem requires, and the Jacobian condition is the infinitesimal form of that guarantee, accepting the cost that we only certify local preservation (a neighborhood of $\theta$). We chose a finite-difference estimate of $J_T$ over an analytic derivation because the KL-reset operator involves a discontinuous reference-model swap that has no closed-form derivative, accepting the cost of step-size sensitivity $\varepsilon$. We chose the SAE support-size functional $g_{\mathrm{supp}}$ as a hard $\ell_0$ constraint rather than a soft sparsity penalty because the property that justifies feature ablation in IR3 is exact support membership, not sparsity level, accepting the cost that $g_{\mathrm{supp}}$ is non-differentiable and requires a subgradient. A domain expert would not describe this as a variant of standard gradient regularization (the closest prior method) because gradient regularization *adds* a penalty to the loss, whereas MPC *checks* whether an existing mitigation operator lies in the class of $P$-preserving maps—it is an audit of the operator, not another operator.

**Why it measures what we claim.** The quantity $\sigma_{\max}(J_T^s(\theta) \cdot \Pi_{T_\theta V_P^s})$ measures preservation of property $P$ on stratum $s$ under $T$ because the assumption that $V_P^s$ is a smooth stratum and $T$ is differentiable on that stratum makes the first-order term $J_T^s \cdot v$ the exact infinitesimal change of $g_P$ along $T$'s action; this assumption fails when $T$ has a discontinuous component (the KL-reset swap), in which case the quantity reflects the one-sided Jacobian on each side of the reset boundary rather than a single derivative. The functional $g_{\mathrm{flat}}(\theta) = \lambda_{\min}(\nabla^2_x r_\theta(x^*(\theta)))$ measures flatness of the optimum because the assumption that $x^*(\theta)$ is a strict local maximum makes the smallest Hessian eigenvalue a faithful second-order flatness proxy; this assumption fails when the reward landscape has multiple near-tied optima, in which case the quantity reflects the curvature at one arbitrarily selected optimum. The functional $g_{\mathrm{supp}}(\theta) = \|\phi_{\mathrm{supp}}(\theta)\|_0 - k$ measures interpretable-feature support because the assumption that SAE features are monosemantic makes support size equivalent to the number of active interpretable concepts; this assumption fails when features are polysemantic, in which case the quantity reflects the number of active directions rather than concepts.

## Contribution

['MPC, a checkable certificate for mitigation operators that decides whether a mitigation preserves the property that justified it, via a Jacobian-tangent-space condition with a finite-difference estimator for discontinuous operators.', 'A counterexample family of standard mitigations (KL-reset schedules, SAE feature ablations) that violate the certificate, demonstrating that the property-preservation assumption is not automatic.', 'A design principle for mitigation operators: they should be projections onto the property variety, which yields a constructive recipe for building mitigations that are certified by construction.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Reward-model pairs with known hack signatures (IR3-style) | Contains ground-truth hacking labels for falsifiable subsetting |
| Primary metric | Certificate agreement with finite-step property drift (Δg_P over step size δ=0.1) | Directly tests whether Cert predicts actual preservation |
| Baseline | KL penalty toward reference model | Standard mitigation; tests sub-claim that luck ≠ construction |
| Baseline | Reference Resets of KL penalty | Discontinuous operator; tests our one-sided finite-difference handling |
| Baseline | Explicit gradient regularization | Closest prior method; tests audit-vs-add distinction |
| Ablation-of-ours | MPC with post-hoc check replacing Jacobian condition | Isolates whether infinitesimal form is doing the work |
| Ablation-of-ours | MPC with first-order flatness g_flat=||∇_x r|| (original naive form) | Tests whether second-order correction is necessary |

### Why this setup validates the claim

The central claim is that MPC's Jacobian-restricted-to-tangent-space condition predicts genuine property preservation better than a post-hoc measurement of $P(T(\theta))$, and that this distinction matters exactly where mitigations land near $V_P$ by luck rather than by construction. The KL-penalty baseline tests the "luck vs. construction" sub-claim: it is a mitigation that reduces hacking empirically but was never designed to preserve flatness or support, so MPC should flag it as uncertified while post-hoc checks pass it. Reference Resets test the discontinuous-operator sub-claim, since the one-sided finite-difference Jacobian is the only quantity that can even be computed there. Explicit gradient regularization tests the audit-vs-add distinction: it changes the objective, so it may certify trivially, and MPC should distinguish that from operators that preserve $P$ without altering the loss. The ablation—replacing the Jacobian condition with a direct post-hoc $g_P(T(\theta))$ threshold—is the decisive control: if MPC's advantage survives only because of the infinitesimal form, the ablation should collapse to baseline behavior on the luck-prone subset. The second ablation—using the naive first-order flatness functional—tests whether the second-order correction and implicit-function term are necessary to avoid a vacuous certificate. Certificate agreement with finite-step property drift is the right primary metric because it is the only quantity that directly measures whether the certificate's bit tracks the thing it claims to track; a method that certifies everything or nothing would score poorly regardless of downstream reward accuracy.

### Expected outcome and causal chain

**vs. KL penalty toward reference model** — On a case where the KL penalty happens to pull the policy into a flat region by coincidence of the reference model's own optimum, the baseline produces a post-hoc flatness pass with no guarantee it survives perturbation because its mechanism is a soft pull, not a structural constraint. Our method instead checks whether the KL operator's differential annihilates the flatness tangent directions and returns uncertified when it does not, because the Jacobian condition detects the absence of a construction guarantee. We expect a noticeable certificate-vs-finite-step-drift disagreement concentrated on the lucky-landing subset, with parity on cases where KL genuinely preserves flatness.

**vs. Reference Resets of KL penalty** — On a case where the reset boundary falls near the optimum, the baseline produces a discontinuous jump whose one-sided derivative is undefined, so any post-hoc flatness reading reflects the average over the boundary rather than the true local behavior. Our method instead estimates the one-sided Jacobian via finite differences on each side of the boundary and reports the residual as a witness, because the one-sided construction is defined even across the discontinuity. We expect MPC to flag reset cases as weakly certified or uncertified with a witness residual noticeably larger than on smooth KL cases, exposing the boundary sensitivity the baseline hides.

**vs. Explicit gradient regularization** — On a case where adding a gradient penalty flattens the optimum but also shifts the reward's argmax, the baseline produces a flat optimum at a different location, which post-hoc flatness rewards but which is not preservation of the original property. Our method instead certifies only if the operator's differential preserves the tangent space at the original $\theta$ and the finite-step image remains in the same stratum, because the audit checks the map rather than the endpoint. We expect MPC to certify gradient regularization only on the subset where the argmax is unchanged, and to withhold certification where the penalty relocated the optimum—a pattern the baseline cannot express.

**vs. MPC with post-hoc check (ablation)** — On the luck-prone subset, the ablation replaces the Jacobian condition with a direct threshold on $g_P(T(\theta))$, so it certifies any operator that lands near $V_P$ regardless of construction. Our method instead requires the differential to annihilate the tangent space and the finite step to stay in the stratum, so it withholds certification on lucky landings. We expect the ablation to agree with finite-step drift at a lower rate than full MPC specifically on the luck-prone and discontinuous-operator subsets, confirming that the infinitesimal form adds discriminating power.

**vs. MPC with first-order flatness (ablation)** — On any interior optimum, the first-order functional $\|\nabla_x r_\theta(x^*)\|$ is identically zero, making $V_{\mathrm{flat}}$ the entire parameter space and the tangent space everything, so the certificate becomes vacuous. Our method instead uses the second-order measure $\lambda_{\min}(\nabla^2_x r_\theta(x^*(\theta)))$ with the implicit-function correction, which is non-degenerate at strict local maxima. We expect the first-order ablation to certify almost all operators (near-100% certificate rate) and thus to show poor agreement with finite-step drift, while full MPC shows high agreement.

### What would falsify this idea

If MPC's certificate agrees with finite-step property drift at the same rate as the post-hoc-threshold ablation across all subsets—i.e., the Jacobian condition adds no discriminating power precisely on the luck-prone and discontinuous-operator subsets where it was supposed to matter—then the central claim that infinitesimal construction guarantees are the right object to certify is wrong. A uniform gain across subsets rather than a gain concentrated where the predicted failure modes occur would likewise falsify the causal story.

## References

1. Gradient Regularization Mitigates Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards
2. IR3: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking
