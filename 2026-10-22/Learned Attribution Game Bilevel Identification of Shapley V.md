# Learned Attribution Game: Bilevel Identification of Shapley Value Functions from Faithfulness Signals

## Motivation

Existing Shapley-based attribution methods for structured data, such as Spectral-Spatial Interpretation (SSI), pre-specify the attribution game — fixing the surrounding context and defining a conditional game over channels — without validating whether that game structure is identifiable or optimal for faithfulness. This pre-specification persists because the field lacks a differentiable formulation of the game's value function and baseline that can be learned from data using faithfulness as the training signal; SSI inherits this gap directly, as its local channel game is fixed by construction rather than learned, leaving the identifiability and optimality of the attribution game unestablished.

## Key Insight

The Shapley value function and baseline can be identified from data by making the attribution game differentiable and training it end-to-end to minimize a faithfulness loss, because the faithfulness loss is a smooth function of the game structure when the perturbation operator is parameterized and the Shapley value is computed via a differentiable kernel-weighted least-squares surrogate.

## Method

**What it is.** Learned Attribution Game (LAG) takes a trained black-box model $f$, a dataset of inputs $\{x_i\}$, and a parameterized perturbation operator $\pi_\phi$ together with a learnable baseline $b_\psi$, and outputs a Shapley value function $v_{\phi,\psi}(S; x) = \mathbb{E}[f(\pi_\phi(x, S))]$ whose structure (perturbation and baseline) is optimized so that the induced Shapley attributions $\phi^{\text{Shap}}$ minimize a faithfulness loss. LAG is trained via bilevel optimization: an inner loop computes Shapley values under the current game, and an outer loop updates $(\phi,\psi)$ to minimize the faithfulness loss.

**How it works.**

```pseudocode
Input: black-box model f, data {x_i}, coalition sampler q(S),
       inner steps K=50, outer steps T=2000, learning rates η_inner=1e-3, η_outer=1e-3,
       faithfulness loss L_faith = E_{i,S~q}[ ( f(x_i) - f(π_φ(x_i, S)) - Σ_{j∈S} φ_j^{Shap}(x_i) )^2 ]
       + λ * E_{i,j}[ ( φ_j^{Shap}(x_i) - φ_j^{Shap}(x_i^{perm(j)}) )^2 ]   (λ=0.1, perm(j) permutes feature j)

Initialize φ (perturbation net) and ψ (baseline vector) randomly.

for t = 1 ... T:
    Sample a mini-batch of inputs {x_i}.
    for each x_i:
        Sample M=64 coalitions S ~ q(S)  (q = Shapley kernel: q(S) ∝ (d-1) / (|S| (d-|S|)) )
        Compute game values v_{φ,ψ}(S; x_i) = f( π_φ(x_i, S) )  (for S=∅, use f(b_ψ))
        Solve the weighted least-squares surrogate:
            min_{a} Σ_{S} q(S) ( v_{φ,ψ}(S; x_i) - a_0 - Σ_{j∈S} a_j )^2
        Obtain Shapley estimates φ_j^{Shap}(x_i) = a_j  (closed-form solve, differentiable).
    Compute L_faith over the mini-batch using the estimated φ^{Shap} and the current (φ,ψ).
    Update φ ← φ - η_outer ∇_φ L_faith ;  ψ ← ψ - η_outer ∇_ψ L_faith  (backprop through the closed-form solve).
    Every 100 steps, run K=50 inner steps of a meta-learning update on φ,ψ to reduce the gap between the surrogate solution and the exact Shapley value under the current game (using a small exact-Shapley computation on a subset of features).
return φ, ψ, and the induced Shapley estimator.
```

**Why this design.** We chose a differentiable kernel-weighted least-squares surrogate (the Shapley kernel) over Monte Carlo permutation sampling because it yields a closed-form, differentiable mapping from game values to Shapley estimates, enabling gradient flow to the game parameters; this accepts the cost that the surrogate introduces approximation error for large feature counts, which we mitigate with the periodic inner meta-learning step. We chose to parameterize the perturbation operator as a neural network $\pi_\phi$ (rather than a fixed marginalization or conditional sampler) because it allows the game to adapt to the model's internal feature dependencies, accepting the risk of overfitting to the faithfulness loss; we control this with weight decay and early stopping on a held-out faithfulness split. We chose a learnable baseline vector $b_\psi$ (rather than a fixed zero or dataset mean) because the baseline defines the reference point of the game and strongly affects attribution faithfulness, accepting that it may drift to trivial values; we regularize $\psi$ toward the data mean with a small penalty. We chose bilevel optimization (outer faithfulness loss, inner Shapley computation) over a single-level joint loss because the Shapley value is defined as the solution to an inner optimization, and treating it as such preserves the game-theoretic semantics; this accepts higher computational cost (two nested loops) but avoids the collapse where the game parameters and attributions co-adapt to trivially satisfy the loss. Unlike Faith-Shap, which fixes a polynomial approximation of the value function, LAG learns the value function itself; unlike SSI, which fixes the surrounding context and defines a local channel game, LAG learns the perturbation operator and baseline that define the game, so a domain expert would not describe LAG as a variant of either — it changes what is learned, not just how it is approximated.

**Why it measures what we claim.** The faithfulness loss $L_{\text{fait}} = \mathbb{E}[(f(x) - f(\pi_\phi(x,S)) - \sum_{j\in S} \phi_j^{\text{Shap}})^2]$ measures the faithfulness of the attribution game because it quantifies the efficiency axiom's violation: if the Shapley values are faithful to the game, the sum of attributions over any coalition $S$ should equal the model output difference between the original input and the perturbed input; this assumption fails when the perturbation operator $\pi_\phi$ produces out-of-distribution inputs for which $f$ is unreliable, in which case the loss reflects the model's extrapolation error rather than attribution faithfulness. The permutation-consistency term $\mathbb{E}[(\phi_j^{\text{Shap}}(x) - \phi_j^{\text{Shap}}(x^{\text{perm}(j)}))^2]$ measures the invariance of attributions to feature permutations because it enforces that permuting a feature's value should not change its attribution; this assumption fails when features are strongly correlated, in which case the term penalizes legitimate context-dependent attributions. The inner meta-learning step measures the gap between the surrogate Shapley estimate and the exact Shapley value under the current game because it directly minimizes that gap on a subset of features; this assumption fails when the subset is not representative, in which case the meta-step may overfit to the subset. The learnable baseline $b_\psi$ measures the reference point of the game because it defines the value of the empty coalition; this assumption fails when the baseline is far from the data manifold, in which case attributions are measured relative to an unrealistic reference.

**(D) Surrogate-to-exact-Shapley equivalence.** The outer loss $L_{\text{fait}}$ is computed using the surrogate Shapley estimates $\phi^{\text{Shap}}$ from the weighted least-squares solve. This loss measures the faithfulness of the attribution game only if the surrogate estimates equal the exact Shapley values of the current game $v_{\phi,\psi}$. This assumption fails when the number of features $d$ is large relative to the number of sampled coalitions $M=64$, so that the surrogate error dominates the residual; in that case, $L_{\text{fait}}$ reflects surrogate approximation error rather than game quality, and the outer loop may reward a degenerate game. To monitor this, we compute the exact Shapley values $\phi^{\text{exact}}$ for a random subset of $d_{\text{exact}}=8$ features via permutation sampling with $P=1000$ permutations every 100 outer steps, and report the mean squared error $\|\phi^{\text{Shap}} - \phi^{\text{exact}}\|^2$ on those features. We also report this surrogate-vs-exact gap per iteration in all experiments.

**(E) Manifold-preserving perturbation constraint.** To prevent the perturbation operator $\pi_\phi$ from collapsing the game by mapping coalitions to out-of-distribution inputs where $f$ is constant or the residual is structurally small, we constrain $\pi_\phi$ to a manifold-preserving class: $\pi_\phi$ is implemented as a conditional generative sampler (a 3-layer MLP with hidden size 256 and ReLU activations) that takes as input the original input $x$, the coalition mask $S$, and a noise vector $z\sim\mathcal{N}(0,I)$, and outputs a perturbed input. We add an explicit on-manifold penalty $\Omega_{\text{manifold}} = \mathbb{E}_{x,S}\left[ \|\pi_\phi(x,S) - \text{Proj}_{\mathcal{M}}(\pi_\phi(x,S)) \|^2 \right]$, where $\text{Proj}_{\mathcal{M}}$ is the projection onto the data manifold approximated by the nearest neighbor in the training set (computed via Euclidean distance in the raw feature space). The total outer loss is $L_{\text{total}} = L_{\text{fait}} + \beta \Omega_{\text{manifold}}$ with $\beta=0.01$. This penalty is evaluated on the same mini-batch and backpropagated to $\phi$.

**(F) External faithfulness validation.** To provide a game-independent check that cannot be gamed by $\pi_\phi$, we compute a deletion/insertion metric on a held-out validation set of 512 examples. For each input, we rank features by their Shapley attributions and measure the drop in model output when the top-$k$ features are deleted (set to the baseline $b_\psi$) or inserted (starting from $b_\psi$). We report the area under the deletion curve (AUDC) and insertion curve (AUIC). These metrics are computed only for evaluation and are not used in training.

## Contribution

['A differentiable attribution game framework (LAG) that learns the perturbation operator and baseline of a Shapley value function end-to-end via bilevel optimization, making the game structure identifiable from faithfulness signals rather than pre-specified.', 'A concrete design principle: faithfulness can be used as a training signal for the game structure when the Shapley value is computed via a differentiable kernel-weighted least-squares surrogate, enabling gradient-based optimization of the game parameters.', "An instantiation of LAG for spectral-spatial data that learns a local channel game per spatial position, replacing SSI's fixed conditional game with a learned one."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HLS Burn Scar with SSA-Transformer classifier | Real correlated hyperspectral features stress perturbation faithfulness |
| Primary metric | Attribution faithfulness gap on coalition subsets | Directly quantifies efficiency-axiom violation LAG optimizes |
| Secondary metric | Deletion/insertion AUC (AUDC/AUIC) on held-out set | External, game-independent faithfulness check |
| Secondary metric | Surrogate-vs-exact Shapley MSE on 8 features | Monitors surrogate approximation error per iteration |
| Baseline | MC-Owen estimator | Tests whether learning the game beats fixed marginal sampling |
| Baseline | Faith-Shap interaction index | Tests whether learning value function beats fixed polynomial game |
| Baseline | Fixed-baseline KernelSHAP | Tests whether learnable baseline/perturbation beats fixed reference |
| Baseline | Differentiable-KernelSHAP with learned baseline (no bilevel) | Tests whether bilevel structure adds value beyond learnable game |
| Ablation-of-ours | LAG with frozen π_φ and b_ψ | Isolates contribution of learning the game structure |
| Ablation-of-ours | LAG without manifold penalty (β=0) | Tests whether manifold constraint prevents game collapse |
| Ablation-of-ours | LAG without permutation-consistency term (λ=0) | Tests contribution of invariance regularizer |

### Why this setup validates the claim

The central claim is that learning the perturbation operator and baseline of the attribution game, rather than fixing them, yields Shapley attributions that better satisfy the efficiency axiom on a black-box model with correlated features. The HLS Burn Scar dataset with an SSA-Transformer classifier is the right testbed because its spectral bands are strongly correlated and the model's behavior off the data manifold is poorly characterized — exactly the regime where a fixed marginalization or zero baseline produces out-of-distribution perturbations and the faithfulness loss becomes dominated by extrapolation error. The primary metric, the faithfulness gap between model-output differences and summed Shapley attributions across sampled coalitions, is the quantity LAG's outer loss directly minimizes, so it is the metric most likely to detect the predicted effect if it exists. However, because this metric is computed using the surrogate Shapley estimates, we also report the surrogate-vs-exact Shapley MSE on a subset of 8 features to verify that the surrogate tracks exact Shapley values under the learned game; if this gap is large, the primary metric is unreliable. The deletion/insertion AUC provides an external, game-independent check that cannot be gamed by the perturbation operator. MC-Owen tests the sub-claim that a learned game beats a fixed sampling scheme; Faith-Shap tests the sub-claim that learning the value function beats a fixed polynomial approximation; fixed-baseline KernelSHAP tests the sub-claim that a learnable baseline matters; and the differentiable-KernelSHAP with learned baseline (no bilevel) tests whether the bilevel structure itself is necessary beyond simply learning the game. The frozen-game ablation is the decisive control: if LAG's gain persists when π_φ and b_ψ are frozen at their trained values, the gain comes from the surrogate solver, not from learning the game — and the claim collapses. The manifold-penalty ablation tests whether the constraint is necessary to prevent game collapse. Together these form a falsifiable test: the gain must be concentrated on coalition subsets where the fixed baselines produce off-manifold perturbations, must vanish under the frozen-game ablation, and must be accompanied by a small surrogate-vs-exact gap.

### Expected outcome and causal chain

**vs. MC-Owen estimator** — On a case where a coalition contains a strongly correlated pair of spectral bands, MC-Owen marginalizes the excluded band to its marginal distribution, producing a perturbed input that violates the joint spectral manifold and drives f into an unreliable region; its fixed sampling scheme cannot adapt to this dependency. LAG instead learns π_φ to perturb in a way that respects the model's internal feature dependencies, so the game values stay on-manifold. We expect a noticeable faithfulness-gap reduction on coalitions containing correlated-band pairs, with near-parity on coalitions of independent bands. Additionally, LAG's deletion/insertion AUC should be higher, and the surrogate-vs-exact MSE should remain below 0.01 on the 8-feature subset.

**vs. Faith-Shap interaction index** — On a case where the true value function has higher-order structure that a low-degree polynomial cannot capture, Faith-Shap's fixed polynomial approximation misrepresents the game, so its attributions satisfy efficiency for the approximate game but not the true one. LAG learns the value function itself through the differentiable kernel-weighted surrogate, so the game it solves is the one the model actually induces. We expect LAG to show a smaller faithfulness gap on inputs whose value functions are poorly fit by the polynomial, with parity on inputs that are nearly additive. The surrogate-vs-exact MSE should be comparable to Faith-Shap's approximation error on the same inputs.

**vs. Fixed-baseline KernelSHAP** — On a case where the data mean baseline lies far from the input's local manifold (e.g., an extreme burn-severity pixel), a fixed baseline defines the empty coalition at an unrealistic reference, so every attribution is measured relative to an off-manifold point and the efficiency violation is inflated. LAG learns b_ψ regularized toward the data mean but adapted per the faithfulness loss, keeping the reference near the data manifold. We expect a concentrated gain on inputs whose distance to the global mean is large, and parity on inputs near the mean. The deletion/insertion AUC should show a similar pattern.

**vs. Differentiable-KernelSHAP with learned baseline (no bilevel)** — This baseline learns π_φ and b_ψ jointly with the surrogate attributions in a single-level loss, without the inner Shapley computation. We expect LAG's bilevel structure to yield lower faithfulness gaps and higher deletion/insertion AUC, because the single-level approach can co-adapt the game and attributions to trivially satisfy the loss. The surrogate-vs-exact MSE should be lower for LAG, indicating that the inner loop keeps the surrogate aligned with exact Shapley values.

**vs. LAG with frozen π_φ and b_ψ** — This ablation freezes the learned game parameters after training and recomputes attributions. If the gain is due to learning the game, the frozen version should perform similarly to full LAG (since the game is fixed at test time), but the key comparison is against the baselines: the frozen LAG should still outperform fixed-game baselines, confirming that the learned game structure is better. If the frozen version matches full LAG, the gain comes from the surrogate solver, not from learning the game, and the claim collapses.

**vs. LAG without manifold penalty (β=0)** — Without the on-manifold constraint, π_φ may collapse the game by mapping coalitions to out-of-distribution inputs where f is constant. We expect this ablation to show a much smaller faithfulness gap (due to game collapse) but also a much worse deletion/insertion AUC, revealing that the primary metric is being gamed. The surrogate-vs-exact MSE may also be large, indicating surrogate error.

**vs. LAG without permutation-consistency term (λ=0)** — Removing the invariance regularizer may lead to attributions that are less stable under feature permutations. We expect a slightly higher faithfulness gap and lower deletion/insertion AUC, but the effect may be small. This ablation tests the contribution of the regularizer.

### What would falsify this idea

If LAG's faithfulness-gap advantage over the fixed baselines is uniform across all coalition subsets rather than concentrated on subsets containing correlated-band pairs and off-manifold inputs, the predicted causal mechanism is not operating. Equivalently, if the frozen-game ablation matches full LAG, the gain comes from the surrogate solver rather than from learning the game, and the central claim is wrong. Additionally, if the surrogate-vs-exact Shapley MSE is large (e.g., >0.1) or if the deletion/insertion AUC does not improve, the faithfulness loss is not a valid training signal and the method fails.

## References

1. Spectral-Spatial Interpretation
2. Spectral-Spatial Attention Transformer with Dense Connection for Hyperspectral Image Classification
3. Faith-Shap: The Faithful Shapley Interaction Index
