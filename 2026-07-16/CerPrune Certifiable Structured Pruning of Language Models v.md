# CerPrune: Certifiable Structured Pruning of Language Models via Interval Bound Propagation

## Motivation

Existing pruning methods for LLMs, like the layer similarity approach in 'The Unreasonable Ineffectiveness of the Deeper Layers', rely on heuristic importance scores without formal guarantees on output distribution preservation. This leads to unpredictable degradation, especially when pruning layers or neurons involved in reasoning or safety-critical knowledge, as there is no certificate that the pruned model's outputs remain close to the original. A formal bound on the divergence is needed for trustworthy compression.

## Key Insight

Interval bound propagation (IBP) computes a convex over-approximation of the set of possible logits after weight removal, enabling a sound and computationally tractable upper bound on the KL divergence between the original and pruned output distributions.

## Method

### CerPrune: Certifiable Structured Pruning of Language Models via Interval Bound Propagation

**Motivation:** Existing pruning methods for LLMs, like the layer similarity approach in 'The Unreasonable Ineffectiveness of the Deeper Layers', rely on heuristic importance scores without formal guarantees on output distribution preservation. This leads to unpredictable degradation, especially when pruning layers or neurons involved in reasoning or safety-critical knowledge, as there is no certificate that the pruned model's outputs remain close to the original. A formal bound on the divergence is needed for trustworthy compression.

**Key insight:** Interval bound propagation (IBP) computes a convex over-approximation of the set of possible logits after weight removal, enabling a sound and computationally tractable upper bound on the KL divergence between the original and pruned output distributions.

#### (A) What it is

CerPrune is a pruning criterion that, given a pre-trained LLM and a set of candidate neurons to prune, computes a certifiable upper bound on the KL divergence between the original model's output distribution and the pruned model's output distribution using interval bound propagation. It prunes neurons individually in an iterative greedy fashion, recomputing interval bounds after each pruning decision to account for shifted activations. Neurons are pruned only when the bound is below a user-specified threshold τ.

#### (B) How it works (pseudocode)

```python
# Input: original model f with parameters θ, input batch X, pruning threshold τ
# Output: set of pruned neurons P (ordered by pruning order)

P = []
θ_cur = copy(θ)  # current parameters
for epoch in range(max_epochs):  # iterative greedy until no new neurons
    candidates = all remaining non-pruned neurons (structured groups: attention heads or MLP neurons)
    best_kl = None
    best_n = None
    for each neuron n in candidates:
        # Step 1: Compute original logits with current θ_cur
        z_orig = f(X; θ_cur)
        # Step 2: Create temporary pruned parameter set θ_tmp where only neuron n is zeroed
        θ_tmp = copy(θ_cur)
        zero_out(θ_tmp, n)
        # Step 3: Compute interval bounds on logits for θ_tmp using IBP
        # IBP forward pass: treat each weight w as interval [0, w] if w > 0 else [w, 0]
        # Propagate intervals through layers to get logit intervals [L_i, U_i]
        [L, U] = interval_bound_propagation(X, θ_tmp, original_weights=True)
        # Step 4: Compute worst-case infinity-norm logit difference
        delta_inf = max_i max(|L_i - z_orig_i|, |U_i - z_orig_i|)
        # Step 5: Compute KL upper bound using known inequality: KL ≤ 2 * delta_inf
        kl_bound = 2 * delta_inf
        if kl_bound < τ and (best_kl is None or kl_bound < best_kl):
            best_kl = kl_bound
            best_n = n
    if best_n is None:
        break  # no more neurons can be pruned
    else:
        zero_out(θ_cur, best_n)
        P.append(best_n)
return P
```
Hyperparameters: τ (default 0.01) controls conservatism; IBP uses standard affine transformations with ReLU (see IBP literature) with no extra hyperparameters; max_epochs = number of candidate neurons (guaranteed convergence).

#### (C) Why this design

We chose IBP-based sound bounds over Monte Carlo approaches (e.g., sampling pruned models) because the latter offer only probabilistic guarantees and may miss worst-case behaviors; the cost is that IBP can be computationally more expensive (about 2× forward pass cost) and may produce looser bounds on deep networks due to interval over-approximation. We used the KL ≤ 2||Δz||_∞ bound (derived from Lipschitz constant of softmax KL) instead of computing the exact maximum KL over the interval set, because the latter requires solving a convex optimization per input; the trade-off is a looser but closed-form bound that avoids iterative optimization. We prune neurons iteratively rather than jointly because joint IBP over all candidate neurons would require propagating multi-dimensional intervals and grow exponentially; iterative pruning gives a sound bound for the cumulative set because after each pruning, the interval bounds are recomputed on the shifted activations, ensuring that the bound for the next neuron accounts for previous changes. The greedy selection of the neuron with the smallest kl_bound mimics a safe ordering. We chose the infinity norm bound over L1 or L2 because it aligns with the interval structure (each logit independent), while L1 would require summing intervals and yield a looser bound.

#### (D) Why it measures what we claim

The computational quantity `kl_bound = 2 * delta_inf` measures the worst-case KL divergence between the original and pruned output distributions because:
- `delta_inf` measures the maximum possible change in any logit after pruning, assuming the interval bounds [L_i, U_i] are sound over-approximations of the true pruned logits; this assumption fails when IBP introduces looseness due to non-linearities like ReLU, in which case `delta_inf` overestimates the true change and `kl_bound` becomes a conservative (safe) overestimate.
- The factor 2 measures the worst-case multiplicative effect of logit differences on KL divergence, assuming the Lipschitz constant of the mapping from logits to KL(softmax(z_orig)||softmax(z')) is exactly 2; this assumption fails when the true logit difference is aligned with the gradient direction that maximizes KL, in which case the bound may be loose but remains an upper bound.
- The interval propagation process itself measures the effect of parameter removal on activations by treating each removed weight as an interval containing zero and the original weight; this captures the exact set of possible values if the neuron is pruned, assuming that the input distribution to the neuron is the same as in the current model (i.e., after previous prunings). This assumption holds because we recompute bounds iteratively after each pruning, thus the interval propagation always uses the correct input distribution from the current pruned model. Thus, CerPrune provides a sound bound for each individual pruning decision within the iterative process, certifying that the KL remains below τ throughout the progressive pruning.

**Limitation and mitigation:** The iterative greedy approach assumes that the bound for the next neuron does not depend on which other neurons have been pruned beyond the shifted activations. However, non-linear interactions (e.g., through residual connections or attention) may cause the bound tightness to degrade after many prunings. To mitigate this, we enforce a maximum number of pruned neurons or recompute the bound on the full set at the end to verify the cumulative bound is still below τ; if violated, we revert the last pruning. This is included as a safety check in the implementation but not shown in pseudocode for brevity.

## Contribution

(1) A novel pruning criterion for LLMs that provides a sound upper bound on the KL divergence between original and pruned output distributions using interval bound propagation, enabling certifiable compression. (2) The first adaptation of interval bound propagation (IBP) to the structured pruning setting, treating weight removal as a parameter interval perturbation. (3) A design principle that links convex relaxation of pruning to a formal guarantee on output distribution preservation, applicable beyond LLMs to any deep network.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | WikiText-2 + Jigsaw Toxic Comment Classification | Standard LM + safety-critical task. |
| Primary metric | True KL divergence (WikiText-2), Toxicity detection F1 (Jigsaw) | Measures distribution shift + practical impact. |
| Baseline 1 | Random pruning (same pruning ratio) | Naive baseline for pruning criterion. |
| Baseline 2 | Magnitude pruning (by weight norm) | Widely used heuristic baseline. |
| Baseline 3 | Layer similarity pruning (remove layers with cosine similarity >0.9) | Related heuristic from motivation. |
| Ablation-of-ours | CerPrune w/o IBP (Monte Carlo sampling, N=1000) | Isolates effect of sound bound vs sampling. |
| Ablation-of-ours | Bound tightness validation (Monte Carlo estimate of max KL over ±epsilon weight perturbations) | Checks if interval bounds are conservative. |

### Why this setup validates the claim

This setup tests the central claim that CerPrune provides a sound, certifiable upper bound on KL divergence after pruning. Using WikiText-2 for language modeling allows direct computation of true KL between original and pruned model output distributions on a held-out set. The additional Jigsaw dataset evaluates whether the certified bound translates to safety-critical performance preservation. Comparing against random, magnitude, and layer similarity pruning—all lack certification—tests whether CerPrune avoids excessive distribution shift. The ablations isolate the value of IBP's soundness: the Monte Carlo sampling ablation tests whether sampling-based bound may underestimate worst-case KL; the bound tightness validation quantifies how much IBP overestimates the true worst-case KL. The primary metric, true KL, directly measures the quantity CerPrune claims to bound, making the test falsifiable: if CerPrune's pruned models consistently exceed the threshold τ, the method fails. This design ensures that any observed advantage stems from the certification mechanism rather than task-specific artifacts.

### Expected outcome and causal chain

**vs. Random pruning** — On a case where random pruning happens to remove a critical neuron, the baseline produces a large KL spike because it ignores functional importance. Our method instead prunes iteratively only when the certified KL bound is below τ, avoiding such spikes due to its interval propagation that captures worst-case logit changes after each pruning. We expect random pruning to show high variance and frequent threshold violations (e.g., median true KL 3× above τ), while CerPrune stays below τ on most neurons, with only occasional exceedances due to bound looseness.

**vs. Magnitude pruning** — On a case where a neuron with small weights is crucial for specific inputs (e.g., rare tokens), magnitude pruning prunes it, causing a large KL on those inputs because it only considers weight norms, not output sensitivity. Our method instead retains such neurons because the interval bounds would show that zeroing its weights can produce large logit shifts for those inputs. We expect magnitude pruning to have low average KL but high worst-case KL on rare tokens, whereas CerPrune maintains uniformly low KL across the distribution, showing a smaller gap between average and max KL.

**vs. Layer similarity pruning** — On a case where layers with high similarity still contribute to output diversity (e.g., in residual stream), layer similarity pruning removes them, causing a large KL due to accumulation of residual errors. Our method avoids this because IBP captures the cumulative effect after previous prunings, ensuring that only neurons with safe impact are removed. We expect layer similarity pruning to exhibit sudden KL jumps after pruning certain layers, while CerPrune's KL remains bounded by τ throughout.

**vs. CerPrune w/o IBP (Monte Carlo)** — On a case where sampling underestimates the true KL due to limited samples (e.g., a neuron affects only a small subset of inputs), the ablation prunes it, later causing a KL violation because the Monte Carlo estimate missed the worst-case effect. Our method uses IBP to soundly bound all inputs, avoiding such violations. We expect the ablation to show occasional KL exceedances (e.g., >τ on 10% of neurons) while CerPrune stays below τ on all neurons, but at the cost of pruning fewer neurons overall.

**Bound tightness validation** — We expect that the IBP-based KL bound is at most 2× the Monte Carlo estimate of maximum KL over ±10% weight perturbations, confirming that the bound is not overly loose.

### What would falsify this idea

If CerPrune's pruned models frequently exhibit true KL exceeding τ (e.g., >5% of neurons violate) while the ablation's exceedance rate is lower, then IBP's soundness claim is undermined. Conversely, if CerPrune prunes too few neurons to show any performance gain over random, then the bound is too loose to be useful.

## References

1. ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation
2. The Unreasonable Ineffectiveness of the Deeper Layers
3. LLM Pruning and Distillation in Practice: The Minitron Approach
4. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
5. The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
6. LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models
7. Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt
8. Training Trajectories of Language Models Across Scales
