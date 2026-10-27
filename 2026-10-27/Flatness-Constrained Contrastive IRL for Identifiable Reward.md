# Flatness-Constrained Contrastive IRL for Identifiable Reward Decomposition

## Motivation

Contrastive inverse reinforcement learning (IR3) reconstructs an implicit reward from behavioral shifts and decomposes it into sparse-autoencoder features to expose reward-hacking signatures, but the decomposition is fundamentally underdetermined: many latent feature assignments explain the same observed behavioral shifts, so contribution analysis cannot tell a genuine hacking feature from an arbitrary rotation of the latent basis. Gradient regularization mitigates reward hacking in RLHF by biasing toward flat optima where reward accuracy and policy robustness coincide, yet that flatness is defined over policy/reward parameters and never touches the latent decomposition that IR3 depends on. The field therefore lacks an identifiability condition for latent reward components under partial observability — the very object IR3's mitigation strategies act on.

## Key Insight

Flatness of the contrastive reward-reconstruction loss with respect to the latent feature decomposition is exactly the missing identifiability condition: among all decompositions consistent with the behavioral shifts, the flat one is the unique one whose recovered hacking features remain invariant to small reward perturbations, so the same curvature penalty that suppresses hacking also pins down the decomposition.

## Method

**What it is.** FLAIR (FLAtness-Identifiable IRL) takes paired baseline/post-alignment response sets and a sparse-autoencoder (SAE) feature basis, and returns a latent reward decomposition whose components are both behaviorally consistent and identifiable, plus a per-component curvature score used to flag hacking features.

**Load-bearing assumption (stated explicitly).** FLAIR assumes that the empirical Fisher information of the Bradley-Terry contrastive loss with respect to the latent decomposition — i.e., the (weighted) covariance of SAE feature differences `f(x, y_post) - f(x, y_base)` — has full column rank over the k SAE components on the behavioral-shift data. Under this rank condition, a component is identifiable exactly when its corresponding Fisher eigenvalue is large (high curvature = pinned by data), and a component with near-zero Fisher eigenvalue is unconstrained (data-sparse, unidentified). The method therefore penalizes small Fisher eigenvalues (ill-conditioning) rather than small gradients; a small gradient norm at a stationary point is a null direction, not a pinned one. When the rank condition fails (sparse shifts), the Fisher eigenvalue reflects data coverage rather than genuine identifiability, and we report the per-component data-coverage count alongside it to expose this failure.

**How it works.**

```pseudocode
Input: behavioral shifts D = {(x, y_post, y_base)}, SAE encoder f, decoder g,
       latent dim k, conditioning weight lambda=0.1, ridge eps_f=1e-4, epochs E
Initialize latent reward components w in R^k
for epoch in 1..E:
    # 1. Contrastive reward reconstruction
    r_post = sum_j w_j * f_j(x, y_post);  r_base = sum_j w_j * f_j(x, y_base)
    L_irl = -mean[ log sigmoid(r_post - r_base) ]          # Bradley-Terry contrastive loss
    # 2. Empirical Fisher (feature-difference covariance) in latent space
    #    d_i = f(x_i, y_post) - f(x_i, y_base) in R^k
    #    F = (1/N) sum_i sigmoid'(r_post_i - r_base_i) * d_i d_i^T  + eps_f * I
    F = empirical_fisher(D, w, f, eps_f)                   # k x k, computed analytically
    # 3. Conditioning penalty on small Fisher eigenvalues (identifiability)
    eigvals = eigvalsh(F)                                  # ascending, k values
    L_cond = -sum_j log(eigvals_j + eps_f)                 # log-det conditioning term
    # 4. Coupled single-objective update
    L = L_irl + lambda * L_cond
    w <- w - eta * dL/dw
    # 5. Identifiability readout
    store curvature_j = eigvals_j per component (via diagonal of F^{-1} or eigenvector projection)
Return: w, {curvature_j}, hacking flags = {j : curvature_j < tau and contribution_j high}
```

**Why this design.** We chose a single coupled objective over a two-stage pipeline (first fit C-IRL, then post-hoc regularize the decomposition) because the identifiability condition must constrain the same optimization that selects the decomposition — post-hoc regularization would select among already-fixed components and could not resolve the rotation ambiguity, accepting the cost that the conditioning penalty requires one k×k Fisher eigendecomposition per step (O(k^3), negligible for k≤512). We chose the analytic empirical Fisher over a finite-difference gradient-norm estimate because the SAE decoder is non-smooth (ReLU/top-k activations) and a Hessian is undefined at the kinks, whereas the Fisher of the linear-in-w Bradley-Terry loss is exactly the weighted feature-difference covariance and is well-defined; this also removes the 2k forward passes per step. We chose to couple conditioning to the contrastive loss rather than to the policy objective because IR3's latent variables live in SAE feature space, not policy-parameter space, so a policy-space KL penalty (the standard RLHF tool) has no gradient path to the decomposition; accepting that the penalty must be re-derived as curvature of the reconstruction loss. This differs from the gradient-regularized RLHF method of the T_0 card, which appends a gradient-norm penalty to the policy objective and never addresses latent non-identifiability; a domain expert would not describe FLAIR as a variant of that method because the penalized quantity is the Fisher conditioning of the contrastive reward reconstruction over SAE latents, not the policy loss over reward-model parameters.

**Why it measures what we claim.** The per-component Fisher eigenvalue `curvature_j` measures latent-component identifiability because, under the full-column-rank assumption on the feature-difference covariance, a large eigenvalue means the behavioral-shift data pins that component (changing it changes the loss), while a near-zero eigenvalue means the component is unconstrained; this assumption fails when the behavioral shifts are too sparse to constrain a component (rank deficiency), in which case `curvature_j` reflects data sparsity rather than genuine identifiability, and we report the per-component data-coverage count alongside it to expose this failure. The contrastive loss `L_irl` measures behavioral consistency of the reconstructed reward because it is the Bradley-Terry likelihood of the observed post-over-baseline preference under the reconstructed reward; this assumption fails when the baseline policy is itself near-optimal (shifts are tiny), in which case `L_irl` reflects noise in the pairing rather than reward fidelity. The contribution score used in the hacking flag measures a component's causal influence on the behavioral shift because it is the counterfactual change in `L_irl` when that component is ablated; this assumption fails when components are strongly correlated in the SAE basis, in which case the score reflects shared variance rather than independent causal contribution.

## Contribution

(1) FLAIR, a flatness-constrained contrastive IRL procedure that re-derives the reward-accuracy/flatness link as a curvature condition on the latent SAE decomposition, yielding identifiable reward components under partial observability. (2) A per-component flatness-and-contribution readout that distinguishes data-pinned hacking features from arbitrary latent rotations, giving a principled flag for IR3-style mitigation. (3) An empirical demonstration that flatness-constrained decompositions recover stable hacking signatures across reward-model configurations where unregularized C-IRL decompositions vary.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Paired baseline/post-alignment response sets with SAE feature basis (≥10k pairs; SAE trained on the same corpus with k=256, top-k=32, ReLU) | Provides behavioral shifts and interpretable latent basis |
| Primary metric | Hacking-flag precision-recall against known hacking features (synthetic ground truth + human-annotated sycophancy/length-bias subsets) | Directly tests identifiability-based detection claim |
| Baseline | Standard KL penalty toward reference model (β=0.1) | Standard RLHF tool for reward hacking comparison |
| Baseline | Reference Resets of KL penalty (reset every 500 steps) | Tests whether flatness-finding alone suffices |
| Baseline | IR3 two-stage contrastive IRL (fit then post-hoc Fisher selection) | Tests our coupled single-objective vs post-hoc pipeline |
| Ablation-of-ours | FLAIR without conditioning penalty (lambda=0) | Isolates contribution of identifiability term |
| Ablation-of-ours | FLAIR with gradient-norm penalty (original first-order proxy) | Tests whether curvature vs gradient matters |

### Why this setup validates the claim

The central claim is that coupling a Fisher-conditioning penalty to the contrastive reward reconstruction over SAE latents yields identifiable components that flag hacking features, and that this coupling is necessary rather than incidental. The KL and Reference-Resets baselines test the sub-claim that policy-space regularization cannot reach the latent decomposition: if flatness in policy space were sufficient, these baselines would match our flag precision-recall, so a gap on them is evidence for the mechanism gap we assert. The IR3 two-stage baseline tests the sub-claim that post-hoc regularization cannot resolve rotation ambiguity: if it matched us, our single coupled objective would be unnecessary complexity. The lambda=0 ablation isolates the conditioning term itself, ensuring any gain is attributable to identifiability rather than to the contrastive reconstruction alone. The gradient-norm ablation directly tests the corrected identifiability logic: if the first-order proxy matched the Fisher-conditioning penalty, the curvature argument would be unnecessary. Hacking-flag precision-recall is the right metric because our claim is specifically about which components get flagged, not about aggregate reward fidelity; a metric like reward accuracy could improve for reasons unrelated to identifiability and would not discriminate the predicted effect. We additionally report per-component data-coverage counts to diagnose rank deficiency, and evaluate on a synthetic case with known ground-truth decomposition to verify that high Fisher eigenvalues correspond to true components.

### Expected outcome and causal chain

**vs. Standard KL penalty** — On a case where the post-alignment policy exploits a spurious SAE feature that correlates with high reward on the training distribution, the KL baseline produces a policy that stays close to reference yet still drifts toward that feature because its penalty operates in policy-parameter space with no gradient path to the latent decomposition. Our method instead assigns low Fisher curvature to that feature's component because the behavioral-shift data does not pin it (near-zero eigenvalue), so we expect the KL baseline to show low flag precision on the spurious-feature subset while FLAIR flags it, with parity on genuinely reward-relevant features (high curvature).

**vs. Reference Resets of KL penalty** — On a case where resetting the reference model finds a flatter policy region that nonetheless retains a hacking feature, the reset baseline produces lower gradient norm in policy space but unchanged latent identifiability because flatness in policy parameters does not imply conditioning of the reward reconstruction over SAE latents. Our method instead measures Fisher curvature of the contrastive loss directly in latent space, so we expect Reference Resets to reduce policy gradient norm without improving flag precision-recall, while FLAIR's gain is concentrated on the identifiability diagnostic rather than on policy-space flatness.

**vs. IR3 two-stage contrastive IRL** — On a case where the fitted reward admits a rotation among correlated SAE components, the IR3 baseline produces a decomposition fixed before regularization, so post-hoc Fisher selection among already-rotated components cannot recover the true hacking feature and yields unstable flags across reward-model configurations. Our method instead constrains the same optimization that selects the decomposition, so we expect IR3 to show high flag variance across configurations and FLAIR to show a noticeable precision gap on the correlated-component subset with parity on well-separated components.

**vs. FLAIR without conditioning penalty (lambda=0)** — On a case where the SAE basis has correlated components, the lambda=0 ablation produces a decomposition that fits the behavioral shifts but leaves the rotation among correlated components unconstrained, so flags are arbitrary. Our method instead penalizes small Fisher eigenvalues, driving the decomposition toward the well-conditioned solution, so we expect lambda=0 to show low flag precision on the correlated-component subset while FLAIR improves it, with parity on well-separated components.

**vs. FLAIR with gradient-norm penalty (original first-order proxy)** — On a case where a component is unconstrained (data-sparse), the gradient-norm penalty assigns low penalty to that component (small gradient), actively selecting the non-identifiable direction, so flags are inverted. Our method instead penalizes small Fisher eigenvalues, so we expect the gradient-norm variant to show below-chance flag precision on the data-sparse subset while FLAIR flags those components as unidentified (low curvature) and achieves above-chance precision.

### What would falsify this idea

If FLAIR's flag precision-recall gain over the lambda=0 ablation, the gradient-norm ablation, and over IR3 were uniform across all subsets rather than concentrated on the correlated-component and data-sparse subsets where the predicted failure modes occur, the identifiability mechanism would not be doing the work we claim. Equivalently, if Reference Resets matched FLAIR on flag precision-recall, the latent-space coupling claim would be wrong.

## References

1. Gradient Regularization Mitigates Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards
2. IR3: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking
