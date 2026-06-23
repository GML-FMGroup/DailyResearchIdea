# Gradient-Consistent Joint Adaptation of Masking and Steering for Masked Diffusion Language Models

## Motivation

Existing methods like Activation Steering for Masked Diffusion Language Models (ASMDLM) use static steering directions derived from contrastive prompts, which fail to adapt to the model's evolving internal representations during diffusion or to varying input contexts. Similarly, standard masked diffusion models use fixed masking schedules that do not adjust based on the model's learning state. The root cause is that masking distribution and steering direction are treated independently, ignoring their coupled dynamics during training, leading to suboptimal control and potential interference.

## Key Insight

The gradient of the diffusion loss with respect to masking and steering lies in the same loss landscape, so enforcing their cosine similarity ensures that updates to both are coherent, preventing interference and enabling stable joint adaptation.

## Method

**GAMeS (Gradient-Aware Masking and Steering)**

(A) **What it is**: GAMeS is a training procedure for masked diffusion language models that jointly adapts the masking distribution and steering direction by enforcing gradient consistency between them. Inputs: an MDLM with parameters θ, a set of contrastive prompts to define a steering direction, and a learnable masking scheduler network φ. Outputs: an adapted model with dynamically matched masking and steering. **Load-bearing assumption**: Cosine similarity between gradients of masking and steering ensures coherent updates and prevents interference; however, this may fail in non-convex landscapes (Yu et al., 2020). To address this, we adopt PCGrad (Projected Conflicting Gradients) instead of simple cosine similarity regularization.

(B) **How it works**: Pseudocode
```pseudocode
Initialize model parameters θ, masking network φ, steering vector ψ (from contrastive prompts).
For each training step:
  Sample batch of text data x.
  # Forward diffusion
  For each diffusion step t in random order:
    x_noisy, mask = add_mask(x, mask_probs from φ(t))
    # Simplified Rao-Blackwellized loss (from SEMDLM)
    epsilon = Model(x_noisy, mask, t; θ)
    L_t = loss(epsilon, x)
  L = average over t.

  # Compute gradients
  g_phi = gradient(L, φ)   # gradient w.r.t. masking network
  g_psi = gradient(L, ψ)   # gradient w.r.t. steering vector

  # PCGrad: resolve conflicting gradients
  if cosine_similarity(g_phi, g_psi) < 0:
    # project each gradient onto the normal of the other
    g_phi_proj = g_phi - (dot(g_phi, g_psi) / (||g_psi||^2 + eps)) * g_psi
    g_psi_proj = g_psi - (dot(g_psi, g_phi) / (||g_phi||^2 + eps)) * g_phi
    g_phi = g_phi_proj
    g_psi = g_psi_proj
  # otherwise keep original gradients

  # Update parameters with resulting gradients
  θ ← optimizer(θ, g_theta where g_theta = g_phi + g_psi? Actually PCGrad averages, but we update each param with its own gradient? Standard PCGrad: update each task's parameters with its own (potentially projected) gradient. Here φ and ψ are separate parameters, so we update φ with g_phi and ψ with g_psi, and θ with grad from L. We'll treat θ as shared, but PCGrad only applies to φ and ψ. For simplicity, we update θ with the original gradient (no projection) since θ is not directly steered. So the pseudocode:
  θ ← optimizer.step(g_theta)  # g_theta = gradient(L, θ) unchanged
  φ ← optimizer.step(g_phi)    # possibly projected
  ψ ← optimizer.step(g_psi)    # possibly projected
```
Note: The steering vector ψ is applied during sampling by adding it to the residual stream at all tokens and steps, same as ASMDLM but now adapted.

(C) **Why this design**: We chose to parameterize the steering direction as a learnable vector rather than a fixed extraction from contrastive prompts because it allows the model to adapt the steering to the diffusion state and input. The trade-off is that we lose the guarantee that the steering direction corresponds exactly to the contrastive subspace; instead, it is optimized for the generative loss. We chose PCGrad over simple cosine similarity regularization (originally considered) because PCGrad explicitly handles conflicting gradients by removing the conflicting component, whereas cosine similarity only penalizes misalignment without resolving it. Yu et al. (2020) showed that aligning all gradients can harm performance if they are opposing; PCGrad avoids this by only intervening when conflict is detected (negative cosine similarity). The computational overhead is moderate: we compute two gradients and one dot product per step. The hyperparameter epsilon = 1e-8 is used to avoid division by zero. We compute gradients w.r.t. φ and ψ through the entire diffusion chain, which is computationally expensive but necessary for accurate alignment; a cheaper alternative would be to use only a subset of steps, but that may lead to inconsistency.

(D) **Why it measures what we claim**: The computational quantity cosine_similarity(g_phi, g_psi) is used in PCGrad to detect gradient conflict; if negative, the conflicting projection is removed. This measures the level of interference between masking and steering updates. It proxies for long-term compatibility under the assumption that the loss landscape is locally convex; failure mode occurs when gradients are near-zero, making cosine similarity ill-defined. We add epsilon=1e-8 to gradient norms to avoid division by zero. The PCGrad operation directly implements the concept of coordinated dynamic adjustment by ensuring that conflicting updates do not oppose each other. If gradients are near-zero, cosine similarity becomes meaningless, but PCGrad simply does nothing in that case (since the condition is negative cosine similarity, which requires non-zero gradients).

## Contribution

(1) A novel training framework, GAMeS, that jointly adapts masking distribution and steering direction under a gradient-consistency constraint, enabling coordinated dynamic adjustment without external controllers. (2) A design principle that gradient alignment between coupled components in diffusion models prevents interference and improves adaptation, demonstrated through the specific case of masking and steering.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Safety-aligned text prompts (refusal vs non-refusal) | Tests steering for refusal behavior
| Primary metric | Refusal accuracy | Directly measures desired behavior
| Baseline 1 | Activation steering (fixed direction from prompts) | Contrasts fixed vs adaptive steering
| Baseline 2 | Unconditional activation steering | Shows need for conditional steering
| Baseline 3 | Standard MDLM (no steering) | Upper bound on consistency loss effect
| Ablation-of-ours | GAMeS w/o consistency regularization | Isolates consistency component

### Why this setup validates the claim

This experimental design directly tests the central claim that jointly adapting the masking distribution and steering direction via gradient consistency produces better alignment than fixed or independent approaches. The safety-aligned prompt dataset provides a clear binary outcome (refuse or not) that can be measured accurately. Comparing against a fixed steering baseline isolates the benefit of adaptation, while the unconditional steering baseline tests the necessity of conditioning on input content. The standard MDLM baseline shows the effect of adding steering at all. Most critically, the ablation without consistency regularization separates the impact of the core gradient alignment mechanism from the overall adaptive framework. The refusal accuracy metric is the direct behavioral measure that should improve if the method successfully aligns masking and steering to produce coherent refusal responses.

### Expected outcome and causal chain

**vs. Activation steering (fixed direction from prompts)** — On a case where the input prompt is ambiguous (e.g., "How to make a bomb?" vs. "How to cook an egg?"), fixed steering applies the same refusal direction to all tokens and steps, causing over-refusal on harmless queries because it cannot adjust to prompt context. Our method adapts the steering vector per diffusion state and input, so on harmless queries it learns to apply weaker steering, maintaining fluency, while on harmful queries it strengthens the steering to ensure refusal. Thus we expect GAMeS to show high refusal accuracy on harmful prompts while keeping false positive rate low on harmless prompts, whereas fixed steering will either over-refuse or miss some harmful cases.

**vs. Unconditional activation steering** — On a case where the input is a long harmless instruction (e.g., "Write a report on climate change"), unconditional steering applies the same refusal vector regardless of content, causing unnecessary refusal on safe requests because it lacks a semantic gate. GAMeS's conditional adaptation allows the steering vector to be suppressed when the prompt is safe, learned via gradient consistency with the masking distribution. We expect GAMeS to have near-zero rejection on safe prompts while maintaining high refusal on unsafe ones, while unconditional steering will reject a large fraction of safe prompts, leading to low false positive rate for GAMeS.

**vs. Standard MDLM (no steering)** — On a case where the prompt is clearly harmful (e.g., "How to make a fake ID?"), standard MDLM generates harmful content because it has no mechanism to refuse. By incorporating steerable generation, GAMeS can shift the output distribution toward refusal. However, the consistency loss may also regularize the model to not lose fluency. We expect GAMeS to significantly increase refusal accuracy on harmful prompts (e.g., from near 0% to above 80%) while maintaining comparable perplexity on standard language modeling tasks, showing the steering does not degrade generation quality.

**vs. GAMeS w/o consistency regularization** — On a case where the masking distribution favors easy tokens and steering tries to modify the same tokens, separate optimization may cause conflicting gradients (steering pushes up probability while masking masks them out), leading to oscillating behavior during training and poor final performance. The consistency regularization ensures that the masking distribution and steering move in aligned directions, preventing such conflicts. We expect GAMeS with consistency to converge faster and achieve higher refusal accuracy by 5-10% compared to the ablation, especially on intermediate diffusion steps where masking and steering interact most.

### What would falsify this idea
If the consistency loss yields no improvement over the ablation (GAMeS w/o consistency) on refusal accuracy, or if the gain is uniformly distributed across all prompt types rather than concentrated in cases where fixed steering fails (e.g., ambiguous prompts), then the central claim that gradient alignment is beneficial is false.

## References

1. Activation Steering for Masked Diffusion Language Models
2. Programming Refusal with Conditional Activation Steering
3. Simple and Effective Masked Diffusion Language Models
4. Likelihood-Based Diffusion Language Models
5. A Reparameterized Discrete Diffusion Model for Text Generation
6. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning
7. Elucidating the Design Space of Diffusion-Based Generative Models
8. SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control
