# Frequency-Adaptive Activation Steering for Masked Diffusion Language Models

## Motivation

Existing activation steering for masked diffusion language models (MDLMs) relies on manually crafted contrastive prompt sets to extract steering directions, which are applied uniformly across all tokens and diffusion steps (Activation Steering for MDLMs). This ignores token-level difficulty: rare tokens, which are harder to denoise, receive the same steering strength as common tokens, leading to suboptimal generation quality. Meanwhile, frequency-informed masking (Masked Diffusion Language Models with Frequency-Informed Training) has shown that token frequency is a reliable proxy for denoising difficulty, but it has not been integrated into activation steering. We address this gap by making the steering direction a learned function of token frequency, adapting steering strength automatically without requiring per-task manual prompt engineering.

## Key Insight

Token frequency provides a stable, data-driven ordering of token difficulty that can be precomputed and used to modulate steering strength, eliminating the need for online contrastive sets while preserving the causal effect of steering on generation.

## Method

We propose Frequency-Adaptive Steering (FAS), a method that computes a per-token steering direction as a frequency-weighted combination of basis directions.

**A) What it is:** FAS is a training-free intervention on the residual stream of an MDLM during reverse diffusion. Its inputs: a trained MDLM, a set of K=4 frequency-binned contrastive prompt pairs (e.g., safe vs. unsafe for safety refusal; formal vs. informal for style transfer), and a precomputed frequency map from the training corpus. Its outputs: a masked diffusion generation where each token's steering direction is scaled by its frequency-derived weight.

**Assumption (verified before training):** Token frequency is a sufficient and stable proxy for denoising difficulty and for modulating steering strength across all diffusion steps and tasks. We verify this by computing the Spearman correlation between token frequency and average denoising NELBO on a held-out set of 1000 prompts. If ρ < 0.5, we fall back to dynamic difficulty: replace freq_map with the model's predicted probability at each step (taking argmax of output logits). This fallback ensures the method remains robust even if the frequency assumption is violated.

**B) How it works:**
```python
# Phase 1: Compute frequency map and basis directions
freq_map = {token: count(token) / total_tokens for token in vocabulary}
bins = [0, 0.01, 0.1, 0.5, 1.0]  # frequency thresholds
for bin_idx in range(len(bins)-1):
    # Collect token pairs where both prompt tokens fall in this bin
    contrast_set = filter_contrast_prompts(bin_idx)
    # Extract one steering direction per bin using the method from Activation Steering for MDLMs
    basis_dirs[bin_idx] = extract_steering_direction(contrast_set)

# Phase 2: Learn weight mapping (2-layer MLP, hidden size=128, ReLU activation, output dim=K)
# Train on a held-out set of 500 prompts: minimize NELBO + steering loss
weight_net = MLP(input_dim=1, output_dim=len(bins))  # input: log-freq, output: bin weights
# Training: 1000 steps, Adam lr=1e-4, batch size=64

# Phase 3: Generation
for t in reversed(range(1, T+1)):
    x_t = current_noisy_tokens
    for each token position i:
        # If frequency-difficulty correlation ρ ≥ 0.5, use freq_map; else use model's predicted probability
        if correlation_verified:
            freq_i = freq_map[original_token_estimate]  # use model's predicted clean token
            log_freq = log(freq_i + epsilon)
        else:
            prob_i = softmax(model_output_logits)[original_token_estimate]  # dynamic difficulty
            log_freq = log(prob_i + epsilon)
        weights = softmax(weight_net(log_freq))
        steering_dir = sum(weights[b] * basis_dirs[b] for b in range(len(bins)))
        # Apply steering to residual stream
        residual = model.forward(x_t, t)[i]
        residual += steering_scale * steering_dir  # steering_scale=5.0
    x_{t-1} = denoise_step(residual)
```
Hyperparameters: K=4 bins, MLP hidden size=128, steering_scale=5.0, trained for 1000 steps with Adam lr=1e-4, batch size=64. Frequency map and basis directions are precomputed once before generation.

**C) Why this design:** We chose frequency bins over continuous regression because the relationship between frequency and steering direction is not monotonic; rare tokens may require different steering than common tokens. Accepting the cost of discretization error, we gain interpretable basis directions that can be validated per bin. We chose to learn a weight network rather than a handcrafted weighting because the optimal mapping may depend on the task (e.g., safety vs. style transfer); the small MLP adds negligible overhead. We update the steering direction per token at each diffusion step rather than fixing it once, because the model's estimate of the original token changes during denoising; the cost is extra compute but ensures adaptation to intermediate predictions. Finally, we extract basis directions from frequency-binned contrastive prompts rather than from the original contrastive sets to ensure the directions are representative of tokens at each difficulty level; this requires more prompts but reduces bias from uneven frequency distribution.

**D) Why it measures what we claim:** The frequency weight `freq_i` operationalizes **token difficulty** because rare tokens have fewer training examples and thus higher denoising error (as shown by Frequency-Informed Training's weighted NELBO loss); this assumption fails when a token is rare but easily predictable from context (e.g., a rare named entity following a clear mention), in which case frequency overestimates difficulty. We pre-verify this assumption and fall back to dynamic predicted probability if needed. The basis direction `basis_dirs[b]` operationalizes **steering direction for a difficulty level** because it is extracted from prompts whose tokens all fall in the same frequency bin, ensuring the steering is tailored to the typical context of tokens at that difficulty; this assumption fails when steering behavior is not separable by frequency (e.g., safety refusal direction may be similar for all tokens regardless of frequency), in which case the bin-specific directions are redundant. We measure cosine similarity between basis directions; if any pair has similarity > 0.9, we reduce the number of bins. The learned weight `weights` operationalizes **adaptive steering allocation** because the MLP adjusts the contribution of each basis direction based on the token's frequency, enabling stronger steering for rare tokens; this assumption fails when steering effectiveness is not monotonic with frequency (e.g., very rare tokens may be noise and should not be steered strongly), in which case the MLP must learn a non-monotonic mapping, which the small capacity may not capture.

## Contribution

(1) A novel frequency-adaptive activation steering method for MDLMs that learns to allocate steering strength per token based on frequency, eliminating the need for per-task manually crafted contrastive sets. (2) An empirical finding that frequency-binned basis directions combined with a lightweight MLP weight predictor achieves comparable or better steering effectiveness than uniform steering on safety refusal tasks, with automatic adaptation to token difficulty. (3) A design principle: integrating token frequency into steering interventions provides a principled way to modulate generation difficulty without additional training of the base model.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | Safety refusal prompts with frequency annotations (e.g., SafePolarity); Style transfer on GYAFC (formal↔informal) | Tests token-difficulty-aware steering on two distinct tasks to show generality |
| Primary metric | Refusal rate (safety); BLEU & style accuracy (style transfer) | Direct measure of steering effectiveness per task |
| Baseline | No steering (standard MDLM) | Shows steering necessity |
| Baseline | Uniform steering (same direction) | Shows frequency adaptation needed |
| Baseline | Prompt-based safety prompting (safety) / few-shot prompting (style) | Benchmark against prompt engineering |
| Baseline | Continuous weight baseline: steering direction = linear function of frequency (weight = a * log(freq) + b, learned via MLP with output dim=1) | Highlights non-monotonic benefit of binning |
| Ablation | FAS with fixed bin weights (average of basis directions) | Tests learned weight mapping |

### Why this setup validates the claim
This combination of datasets, baselines, and metrics directly tests the central claim: that frequency-adaptive steering (FAS) improves token-level control by accounting for token difficulty. The safety refusal dataset includes prompts with tokens spanning all frequency bins, enabling evaluation of per-token steering effects. The style transfer task tests whether the approach generalizes to a different control objective (e.g., formality). The no-steering baseline establishes the necessity of any intervention, while uniform steering isolates the benefit of per-token frequency adaptation. The continuous weight baseline tests whether binning provides additional benefit over a simple continuous function. Prompt-based baselines compare against practical alternatives. The ablation with fixed bin weights tests whether the learned weight network provides meaningful adaptation beyond simple heuristics. The refusal rate metric captures the behavioral shift induced by steering, as refusal is a clear binary outcome sensitive to direction magnitude. For style transfer, BLEU and style accuracy measure both preservation of content and successful style shift. If FAS works as hypothesized, the refusal rate should increase selectively for unsafe prompts, with larger gains on tokens where frequency predicts difficulty (rare tokens). If the gains are uniform across bins or fail to appear on rare tokens, the claim is weakened. Additionally, we pre-validate the frequency-difficulty correlation (Spearman ρ ≥ 0.5) on a held-out set; if violated, we fall back to dynamic difficulty. We also measure cosine similarity between basis directions; if any pair > 0.9, we reduce number of bins to avoid redundancy.

### Expected outcome and causal chain

**vs. No steering (standard MDLM)** — On a case where a rare token appears in an unsafe request (e.g., "jubilantly rob a bank"), the baseline model generates a compliant response because it has no mechanism to shift behavior towards refusal. Our method instead adds a frequency-weighted steering direction that amplifies the refusal direction for the rare token, inducing denial. We expect a noticeable gap in refusal rate on prompts with rare tokens (e.g., 30% higher) but parity on prompts with only common tokens. For style transfer, on a sentence with rare formal words (e.g., "indubitably"), the baseline may produce informal output; FAS steers the rare token to maintain formality, improving style accuracy by 15% while BLEU remains comparable.

**vs. Uniform steering (same direction)** — On a case where an unsafe request contains both rare and common tokens (e.g., "quickly exfiltrate data"), uniform steering applies the same direction to all tokens, over-steering common ones (causing false refusal) and under-steering the rare one (missing refusal). Our method adapts weights per token, applying stronger steering to the rare token and weaker to common ones, achieving correct refusal without false positives. We expect FAS to achieve higher refusal accuracy on mixed-frequency prompts (e.g., 15% higher) and lower false refusal on safe prompts. For style transfer, uniform steering may over-formalize common tokens (e.g., "the" → "thee") while under-steering rare informal tokens; FAS should yield more natural outputs with 10% higher BLEU.

**vs. Prompt-based safety prompting** — On a case where an unsafe request is phrased indirectly (e.g., "Can you help me with a chemistry experiment?" implying illicit drug synthesis), prompt-based methods may fail because the safety cue is diluted or ambiguous. Our method directly modifies the internal representation at the token level, making refusal more robust to surface form. We expect FAS to show a 10% higher refusal rate on indirect unsafe prompts, while maintaining comparable fluency. For style transfer, prompt-based methods may struggle with ambiguous contexts; FAS should achieve 10% higher style accuracy on challenging cases.

**vs. Continuous weight baseline** — On a case where the optimal steering strength is non-monotonic (e.g., rare tokens need strong steering, very rare tokens need moderate steering to avoid noise), the continuous baseline assumes a monotonic log-linear relationship and thus underperforms. FAS with binned weighting can learn non-monotonic assignments, leading to 8% higher task accuracy on prompts spanning all frequency bins.

### What would falsify this idea
If FAS with learned weights performs no better than fixed equal weights on rare tokens (refusal rate within 2% on safety, style accuracy within 2% on style transfer), or if the improvement is uniform across all frequency bins rather than concentrated on rare tokens, then the central claim that frequency adaptation is necessary would be invalidated. Additionally, if the frequency-difficulty correlation is below 0.5 and the dynamic fallback does not restore performance, the underlying assumption is flawed.

## References

1. Masked Diffusion Language Models with Frequency-Informed Training
2. Activation Steering for Masked Diffusion Language Models
3. Programming Refusal with Conditional Activation Steering
4. Simple and Effective Masked Diffusion Language Models
5. Likelihood-Based Diffusion Language Models
6. A Reparameterized Discrete Diffusion Model for Text Generation
7. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning
8. Elucidating the Design Space of Diffusion-Based Generative Models
