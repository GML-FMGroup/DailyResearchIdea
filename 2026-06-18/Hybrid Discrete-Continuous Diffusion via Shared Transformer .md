# Hybrid Discrete-Continuous Diffusion via Shared Transformer for Efficient Language Generation

## Motivation

Existing discrete diffusion models (e.g., Scaling Beyond Masked Diffusion Language Models) operate purely in token space, lacking a continuous latent representation that prevents efficient hybrid generation combining autoregressive and diffusion sampling. This structural limitation stems from the assumption that discrete token-space is sufficient, ignoring the flexibility of continuous methods for tasks requiring fine-grained control or interpolation. We address this by coupling a continuous latent variable that evolves via Gaussian diffusion with discrete tokens via categorical diffusion, denoised by a single transformer with shared parameters.

## Key Insight

Sharing the denoising transformer across continuous and discrete modalities forces the model to learn a joint representation where the continuous latent provides a low-dimensional bridge for efficient hybrid sampling, enabling the discrete token process to be guided by continuous trajectories.

## Method

We propose **Hybrid Discrete-Continuous Diffusion (HyDiCD)**.

**(A) What it is:** HyDiCD is a unified denoising framework that takes as input a sequence of discrete tokens and a continuous latent vector, both corrupted at diffusion step t, and outputs denoised predictions for both modalities using a single transformer with shared parameters. Its output is a joint distribution over clean tokens and a clean latent vector.

**Load-bearing assumption:** The shared transformer can jointly denoise the continuous Gaussian-corrupted latent and the discrete uniformly-masked tokens, forcing the model to learn a joint representation where the latent provides complementary global semantics that improve token generation. We verify this assumption by ablating the latent (replacing with noise at test time) and measuring the impact on perplexity; if perplexity increases, it confirms the latent's role.

**(B) How it works:**

```python
# Training
for each batch (x, z):  # x: discrete tokens (length L), z: continuous latent (dim D=64)
    t ~ Uniform(1, T=1000)
    # Discrete corruption: uniform masking (mask token id M=2, vocab V=267k for WikiText-103)
    # Masking schedule: linear increase from 0 to 1 over T steps
    x_noisy = uniform_mask(x, t, mask_token_id=M, vocab_size=V)
    # Continuous corruption: Gaussian with cosine schedule
    # α_bar(t) = cos^2((t/T + s)/(1+s)*π/2) with s=0.008
    epsilon ~ N(0, I)
    z_noisy = sqrt(α_bar(t)) * z + sqrt(1 - α_bar(t)) * epsilon
    # Shared transformer with 12 layers, hidden size 768, 12 attention heads, feed-forward dim 3072, GeLU
    h = transformer(embed(x_noisy) + pos_enc, project(z_noisy))   # shape: L x 768
    # mlp_discrete: 2-layer MLP, hidden 2048, output V=267k
    logits = mlp_discrete(h)   # L x V
    # mlp_continuous: 2-layer MLP, hidden 2048, output D=64
    z_pred = mlp_continuous(h) # L x D (per token latent predictions)
    # Loss: cross-entropy on tokens + MSE on latent, weighted by λ=0.1
    loss = cross_entropy(logits, x) + λ * MSE(mean_pool(z_pred), z)
    update parameters with AdamW, lr=1e-4, batch_size=64, 500k steps on 4×A100 80GB (~100 hours)
```

```python
# Sampling (generate from scratch, e.g., 4 steps, using DDIM for continuous)
z_T ~ N(0, I)
x_T = [MASK] * L
for t in reversed(range(1, T+1, T//4)):  # 4 uniform steps
    h = transformer(embed(x_t) + pos_enc, project(z_t))
    logits = mlp_discrete(h)
    z_pred = mlp_continuous(h)
    # Discrete update: predict clean tokens, then re-mask using uniform diffusion reverse
    # p(x_{t-1} | x_t, x_0) = Cat( (1 - β_t) * one_hot(x_0) + β_t * Uniform ) for masked positions (Austin et al.)
    x_0_pred = argmax(logits)  # or sample from logits
    x_{t-1} = discrete_reverse_step(x_t, x_0_pred, t, beta_t=1/(T-t+1))
    # Continuous update: DDIM reverse (η=0)
    z_{t-1} = sqrt(α_bar(t-1)) * mean_pool(z_pred) + sqrt(1 - α_bar(t-1)) * ((z_t - sqrt(α_bar(t))*mean_pool(z_pred)) / sqrt(1-α_bar(t)))
```

**(C) Why this design:**
We chose a single shared transformer over separate networks for discrete and continuous because joint representation encourages the model to leverage complementary information (e.g., the latent can capture global semantics while tokens handle local structure), accepting the cost that modality-specific dynamics may compete for limited model capacity. We applied Gaussian diffusion to the latent for its closed-form posterior and efficient sampling, while discrete tokens use uniform categorical diffusion for compatibility with continuous-time extensions and to match prior work (e.g., SaDDLe). The weighting hyperparameter λ=0.1 was selected to prioritize token reconstruction quality (verified via perplexity on a hold-out set) while still providing a meaningful gradient for the latent; a higher λ degraded token accuracy. This design enables hybrid generation where the continuous latent can be used to guide discrete token refinement, and the shared parameters allow for a single training phase without separate modality-specific modules. The discrete reverse step follows the derivation of uniform diffusion (Austin et al.), adapted for masking. Unlike prior hybrid approaches that use separate models (e.g., a VAE for latent and a diffusion for tokens), our unified denoising creates a single latent space that is directly accessible for conditioning, eliminating the need for a separate encoder-decoder.

**(D) Why it measures what we claim:**
The cross-entropy loss on discrete tokens measures the model's ability to recover original tokens from corrupted representations; this operationalizes "discrete generation quality" because the loss is minimized when the predicted distribution matches the true token under the assumption that the corruption process is invertible; this assumption fails when the corruption is too aggressive (high t) so that the signal is lost, in which case cross-entropy reflects pure randomness. The MSE on the continuous latent measures the reconstruction error of the latent variable; this operationalizes "continuous representation fidelity" because L2 distance is the natural metric for Gaussian noise under the assumption that the latent is isotropic Gaussian; this assumption fails when the latent distribution is non-Gaussian (e.g., multimodal), in which case MSE may penalize valid modes. The shared transformer's joint denoising ensures that the continuous latent contributes to discrete prediction via the shared hidden states; this assumption holds when the latent captures global context that is useful for token prediction, but fails when the latent is dominated by noise or is uncorrelated with the discrete tokens, in which case the discrete prediction relies solely on token-only features. By conditioning the discrete denoising on the latent through the transformer, the model learns to exploit the latent for efficient hybrid generation, and we measure this via ablation studies (removing latent degrades perplexity). Additionally, we measure topic consistency (cosine similarity between mean latent and topic embedding of generated text) to directly assess whether the latent encodes global semantics.

## Contribution

(1) HyDiCD, a hybrid diffusion framework that jointly denoises continuous latent and discrete tokens using a single transformer with shared parameters, enabling a unified training and sampling procedure. (2) Demonstration that the shared transformer learns a joint representation where the continuous latent serves as a global guide for discrete token generation, facilitating flexible hybrid sampling with speed-quality trade-offs. (3) A combined training objective and sampling schedule that can be tuned to balance discrete and continuous fidelity, validated by perplexity and latent reconstruction metrics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WikiText-103 | Standard benchmark for language modeling.
| Primary metric | Perplexity (PPL) | Measures discrete token generation quality.
| Coherence metric | Topic consistency (TC): cosine similarity between mean latent over generation and topic vector (from LDA) of reference text | Measures whether latent captures global semantics.
| Training budget | 4×A100 80GB, ~100 hours, batch size 64 | Confirms feasibility and reproducibility.
| Baseline 1 | Autoregressive (AR) | Dominant generative approach.
| Baseline 2 | Masked Diffusion (MDM) | Strong discrete diffusion baseline.
| Ablation-of-ours | HyDiCD w/o latent | Isolates impact of continuous component.

### Why this setup validates the claim
This experimental design directly tests the central claim that joint discrete-continuous diffusion via a shared transformer improves generation by leveraging complementary information. WikiText-103 provides a standard language modeling task with long-range dependencies where global semantics (captured by the latent) and local structure (tokens) both matter. The primary metric, perplexity, measures how well the model predicts discrete tokens after denoising, directly reflecting generation quality. However, perplexity alone may be low even with incoherent text due to local fluency; thus we also compute topic consistency (TC) to explicitly measure whether the latent encodes a global semantic context. Comparing against AR tests whether our diffusion approach can match or exceed the strong autoregressive baseline. Against MDM, we isolate the benefit of adding a continuous latent stream. The ablation (removing the latent) is critical: if HyDiCD outperforms its own ablation on both PPL and TC, then the latent is indeed providing useful global context. Together, these comparisons form a falsifiable test: if our method outperforms both baselines and the ablation on perplexity and topic consistency, it validates the claim that joint modeling is beneficial; a null result (e.g., no improvement over MDM) would indicate that the continuous latent adds no value—a pattern we can diagnose by checking performance on examples requiring long-range coherence.

### Expected outcome and causal chain

**vs. Autoregressive (AR)** — On a case involving long-range dependencies (e.g., a paragraph where pronoun resolution requires tracking a subject across many tokens), the AR model generates locally coherent but globally inconsistent text because it sees only left-context and cannot plan ahead. Our method, by contrast, uses the continuous latent to encode global semantics from the start, so during iterative denoising it can refine tokens to maintain global consistency. We expect our method to achieve lower perplexity on long documents (e.g., >20% gap on WikiText-103 test set, measured on sequences >512 tokens) and higher topic consistency (e.g., TC >0.5 vs. AR's <0.3), but comparable perplexity on short sequences where context is limited.

**vs. Masked Diffusion (MDM)** — On a case where the discrete token sequence is highly structured (e.g., a mathematical expression), the MDM model must recover tokens independently from masked positions without a global latent representation, leading to frequent grammatical errors (e.g., mismatched parentheses). Our method leverages the continuous latent to capture the overall structure (e.g., bracket depth), so denoising is guided by a global plan. Therefore, we expect a noticeable gap on structured data (e.g., 10-15% lower perplexity on WikiText-103's mathematical sections, and higher bracket accuracy) but parity on random unstructured text.

**vs. Ablation (HyDiCD w/o latent)** — On a case requiring both local fluency and global coherence (e.g., a story), the ablation model relies solely on token-level interactions, so it may generate locally plausible but thematically drifting sequences. Our full method uses the latent to maintain a consistent topic vector across steps, so its generations stay on-topic. We expect a consistent mid-single-digit perplexity improvement (e.g., 3-5% lower) across the full test set, with larger gains on longer contexts, and topic consistency improvement of >0.1 on long sequences.

### What would falsify this idea
If our method's perplexity is not significantly lower than the ablation (HyDiCD w/o latent) on long-range contexts, or if the improvement is uniform across all subsets rather than concentrated where global semantics matter (e.g., long sequences), then the claim that the continuous latent provides complementary information is falsified. Additionally, if topic consistency does not improve significantly with the latent (e.g., TC difference <0.05), then the latent is not encoding global semantics as intended.

## References

1. Scaling Beyond Masked Diffusion Language Models
2. The Diffusion Duality
3. Esoteric Language Models
4. Scaling Behavior of Discrete Diffusion Language Models
5. One Step Diffusion via Shortcut Models
6. Beyond Autoregression: Fast LLMs via Self-Distillation Through Time
7. Scaling up Masked Diffusion Models on Text
8. One-Step Diffusion with Distribution Matching Distillation
