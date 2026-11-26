# Encoder-Consistent Generative Streams: Cycle-Closing the Shared Visual Token Space in Unified Multimodal Models

## Motivation

Unified multimodal models such as Integral (Let Vision-Language Models Generate Images through Their Representations) route a frozen vision encoder's hierarchical tokens into a lightweight generative stream, but the encoder is treated as a passive semantic oracle: nothing in the training objective forces generated images to re-encode into the same shared tokens that conditioned their synthesis, so understanding-generation consistency is only an empirical artifact that can silently drift. This structural gap recurs across the unified-generation tree (LMFusion's parallel diffusion modules, TIGeR's autoregressive generation, discrete-token MLLMs), because in every case the generative pathway is optimized against pixel- or token-reconstruction targets while the encoder's representation manifold is never imposed as a constraint on the output distribution. The root cause is that frozen-encoder drift is undetectable by construction: the loss never queries the encoder on generated outputs, so no gradient can oppose representational divergence.

## Key Insight

Because the frozen encoder defines the shared token space that both understanding and generation already use, re-encoding every generated image and penalizing divergence from its conditioning tokens converts that space from an unenforced convention into a structural invariant of the generative stream's output manifold.

## Method

**What it is.** We introduce ECC-Gen (Encoder-Cycle-Consistent Generation), a training procedure for the lightweight generative stream of a unified multimodal model (Integral-style: frozen vision encoder + Abstraction Fusion shared tokens + Block-wise MoT generative stream). Inputs: a text prompt and its conditioning shared visual tokens $z$ produced by Abstraction Fusion from the frozen encoder. Outputs: an updated generative stream whose sampled images re-encode into the same token neighborhood as $z$.

**How it works.**

```pseudocode
# Frozen: vision encoder E, Abstraction Fusion F, VLM understanding blocks
# Trainable: generative stream G (Block-wise MoT)
# Calibration: per-prompt encoder-sensitivity gate s(z) in [0,1]

for (text, image) in loader:
    z = F(E(image))                       # conditioning shared tokens (frozen)
    x_hat = G.sample(text, z)             # one-step-free diffusion sample, K=4 DDIM steps

    # --- cycle branch (differentiable path) ---
    z_hat = F(E(x_hat))                   # re-encode generated image
    L_cycle = 1 - cos(z_hat, z)           # token-space cycle consistency

    # --- sensitivity gate (calibration) ---
    # Estimate local injectivity of F∘E at z via finite-difference Jacobian probe:
    #   for k in 1..K_probe (K_probe=4):
    #       delta_k ~ N(0, sigma^2 I), sigma=0.01
    #       J_k = (F(E(x_hat + delta_k)) - z_hat) / sigma
    #   s(z) = min(1, ||J||_F^2 / tau) with tau=1e-3
    # Gate the cycle loss: L_cycle_gated = s(z) * L_cycle

    # --- manifold anchor (prevents degenerate collapse) ---
    z_bank = nearest_neighbors(z, bank, m=8)   # frozen token bank from real images
    L_anchor = mean_j || z_hat - z_bank_j ||^2  # pull toward real-token manifold

    # --- standard generation objective ---
    L_gen = diffusion_loss(G, image, text, z)

    L = L_gen + lambda_c * L_cycle_gated + lambda_a * L_anchor
    # lambda_c = 0.5, lambda_a = 0.1, warmup lambda_c 0 -> 0.5 over 5k steps
    update(G)
```

Data flow: text + real image → frozen encoder → shared tokens $z$ → generative stream → $\hat{x}$ → frozen encoder → $\hat{z}$ → cosine penalty against $z$ (gated by local sensitivity $s(z)$), plus anchor against a frozen bank of real-image tokens.

**Why this design.** We chose a *differentiable* re-encoding cycle through the frozen encoder over a non-differentiable reward or discriminator because the gradient must reach the generative stream's parameters directly, accepting the cost that backpropagating through the encoder is memory-heavy (mitigated by gradient checkpointing). We chose a cosine penalty in the shared token space over an L2 pixel loss because the shared tokens are what understanding actually consumes, accepting that token-space distance is a coarser signal than perceptual pixel distance. We chose to freeze the encoder and only train $G$ (as in Integral's Block-wise MoT) over jointly fine-tuning the encoder because unfreezing would let the encoder drift toward the generator and dissolve the invariant, accepting that the encoder's inductive biases cap achievable generation fidelity. We added a nearest-neighbor manifold anchor over a pure cycle loss because a cycle loss alone admits the degenerate solution where $G$ learns to output images that map to a single token regardless of $z$, accepting a small pull toward real-image statistics. We added a per-prompt sensitivity gate $s(z)$ over an ungated cycle loss because the cosine penalty is only meaningful where the frozen encoder is locally injective; the gate downweights the cycle term in flat manifold regions where many images collapse to similar tokens, accepting that the constraint is weaker exactly where the encoder is least informative. A domain expert would not call this a variant of Integral: Integral *uses* frozen tokens as inputs; ECC-Gen makes the frozen encoder an *active constraint on the output distribution* via a closed loop, which Integral's architecture never computes.

**Why this measures what we claim.** The cosine cycle loss $1-\cos(\hat{z}, z)$ measures *understanding-generation representational consistency* because it assumes the frozen encoder is a stable semantic map, so equal tokens imply equal semantics; this assumption fails when the encoder is locally insensitive (many images map to similar tokens), in which case the loss reflects coarse semantic agreement rather than exact visual faithfulness. The sensitivity gate $s(z)$ measures *local injectivity of the encoder at the conditioning token* because it assumes the finite-difference Jacobian norm $\|J\|_F^2$ tracks the encoder's local sensitivity, so high $s(z)$ implies distinct images map to distinguishable tokens; this fails when the probe perturbations $\delta_k$ are too small to escape the flat region, in which case $s(z)$ underestimates sensitivity and the gate over-penalizes legitimate novelty. The nearest-neighbor anchor $\|\hat{z}-z_{\text{bank}}\|^2$ measures *manifold membership* because it assumes real-image tokens densely cover the encoder's valid output region, so proximity implies the generated image lies in-distribution; this fails for rare prompts whose valid tokens are sparse in the bank, in which case the anchor penalizes legitimate novelty. The warmup schedule on $\lambda_c$ measures *non-degenerate constraint strength* because it assumes early training needs generation fidelity before cycle pressure, so ramping avoids collapse to the degenerate fixed-point; this assumption fails if fidelity and consistency require simultaneous shaping, in which case warmup delays convergence.

## Contribution

(1) A cycle-consistent training objective for the generative stream of unified multimodal models that closes the loop through the frozen encoder, turning shared-token consistency into an enforced constraint rather than an empirical byproduct. (2) A manifold-anchored variant of the cycle loss that prevents degenerate token collapse, establishing the design principle that representation-space closure requires both a pull toward conditioning tokens and a pull toward the real-token manifold. (3) An empirical characterization of the consistency-fidelity trade-off as a function of cycle weight and warmup, identifying where closure pressure helps versus where it suppresses legitimate visual novelty.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TIGeR-Bench (creative + knowledge-intensive splits) | Spans both free-form and knowledge-heavy prompts |
| Primary metric | Token-space cycle cosine error on held-out prompts | Directly measures the claimed invariant |
| Secondary metric | Encoder local sensitivity $s(z)$ (Jacobian norm) on generated samples | Quantifies when cosine proxy is valid |
| Baseline | Integral (frozen-token MoT, no cycle) | Tests whether closed loop adds anything |
| Baseline | LMFusion-style parallel diffusion stream | Tests against alternate stream-only adaptation |
| Baseline | DDT-LLaMA-style discrete-token generation | Tests against alternate token-space generation |
| Ablation-of-ours | ECC-Gen minus manifold anchor (λ_a=0) | Isolates anchor's role against degenerate collapse |
| Ablation-of-ours | ECC-Gen minus sensitivity gate (s(z)=1) | Isolates gate's role in flat regions |
| Ablation-of-ours | ECC-Gen with L2 pixel reconstruction instead of L_cycle | Tests whether token-space cycle is distinct from pixel loss |
| Ablation-of-ours | ECC-Gen with LPIPS perceptual loss instead of L_cycle | Tests whether token-space cycle is distinct from perceptual loss |

### Why this setup validates the claim

The central claim is that a differentiable re-encoding cycle through the frozen encoder, plus a nearest-neighbor anchor and a sensitivity gate, makes the generative stream produce images whose shared tokens land in the same neighborhood as the conditioning tokens — i.e. understanding-generation consistency without sacrificing fidelity. The dataset choice matters because TIGeR-Bench deliberately mixes creative and knowledge-intensive prompts, and the predicted failure mode (token-space insensitivity, sparse valid-token regions) is concentrated in the knowledge-intensive split; a uniform gain across both splits would not support the mechanism story. Integral is the right primary baseline because it uses the identical frozen tokens as *inputs* but never closes the loop, so any gap between ECC-Gen and Integral is attributable to the cycle constraint rather than to architecture or tokenizer. The LMFusion-style and DDT-LLaMA-style baselines test whether the cycle is doing work that alternative adaptation or tokenization schemes already capture. The ablation with λ_a=0 is the diagnostic that separates "cycle helps" from "cycle collapses" — if the anchor is load-bearing, removing it should degrade the metric disproportionately on rare-prompt cases. The ablation with s(z)=1 tests whether the sensitivity gate is necessary to avoid applying the cycle loss in flat regions where it is vacuous. The L2-pixel and LPIPS ablations test whether the token-space cycle is distinct from existing pixel-level regularizers; if the token-space cycle does not yield a larger gain on the knowledge-intensive split than these alternatives, the contribution is an empirical diagnostic rather than a novel method. The primary metric, token-space cosine cycle error on held-out prompts, is chosen because it is exactly the quantity the loss optimizes, so a null result there would be self-refuting; pixel-level FID is deliberately not primary because the claim is about representational consistency, not raw photorealism. The secondary metric, encoder local sensitivity $s(z)$, is reported to quantify when the cosine proxy is valid, as the adversarial alert requires.

### Expected outcome and causal chain

**vs. Integral (frozen-token MoT, no cycle)** — On a knowledge-intensive prompt where the frozen encoder's token manifold is locally flat (many distinct renderings map to similar tokens), Integral's stream is trained only against the diffusion objective and has no pressure to land in the conditioning token's neighborhood, so it produces images that are plausible but semantically off-target — the re-encoded tokens drift. ECC-Gen instead backpropagates the cosine penalty from $\hat{z}$ into $G$, pulling the generated image toward the conditioning token region, so we expect a noticeable cycle-error gap on the knowledge-intensive subset but near-parity on simple creative prompts where the manifold is sharp. The sensitivity gate $s(z)$ will be lower on the knowledge-intensive subset, confirming that the cycle loss is applied where it is meaningful.

**vs. LMFusion-style parallel diffusion stream** — On a prompt requiring compositional spatial relations, the LMFusion-style stream processes images through parallel modality-specific modules with shared attention, but its generation objective never references the understanding-side token space, so it optimizes pixel fidelity without an explicit consistency signal and its re-encoded tokens can diverge from the conditioning tokens. ECC-Gen adds the closed loop through the same frozen encoder, so we expect lower cycle error concentrated on compositional prompts, with the gap shrinking on single-object prompts where pixel fidelity and token consistency coincide.

**vs. DDT-LLaMA-style discrete-token generation** — On a prompt whose target involves fine attribute detail, the discrete-timestep tokenizer's recursive tokens compensate for progressive attribute loss during denoising, but the token vocabulary is optimized for reconstruction, not for matching the conditioning tokens that understanding consumes, so the generated image can be visually close yet token-distant. ECC-Gen's cosine penalty operates directly on the shared token space that understanding reads, so we expect a modest but consistent cycle-error advantage, largest on attribute-heavy prompts and smallest on coarse layout prompts.

**vs. ECC-Gen minus manifold anchor (λ_a=0)** — On rare prompts where the valid token region is sparse, a pure cycle loss admits the degenerate fixed point where $G$ outputs images mapping to a single token regardless of $z$; without the anchor this collapse is unopposed, so the ablation should show cycle error that initially drops then plateaus while output diversity collapses. ECC-Gen's anchor pulls $\hat{z}$ toward real-image token neighbors, so we expect the ablation to match ECC-Gen on common prompts but diverge sharply on rare ones, with visibly reduced output diversity.

**vs. ECC-Gen minus sensitivity gate (s(z)=1)** — On prompts where the encoder is locally flat, the ungated cycle loss applies a strong gradient that can push $\hat{z}$ toward $z$ even when the image is far from the conditioning semantics, leading to over-regularization and reduced fidelity. ECC-Gen's gate downweights the cycle term in these regions, so we expect the ungated ablation to show lower cycle error but worse pixel fidelity (higher FID) on flat-manifold prompts, while matching ECC-Gen on sharp-manifold prompts.

**vs. ECC-Gen with L2 pixel reconstruction instead of L_cycle** — The L2 pixel loss directly penalizes pixel differences between $\hat{x}$ and the real image, which can improve pixel fidelity but does not explicitly constrain the shared token space that understanding consumes. ECC-Gen's token-space cycle is expected to yield a larger gain on the knowledge-intensive split, where semantic consistency matters more than pixel-level accuracy. If the L2 ablation matches or exceeds ECC-Gen on cycle error, the token-space cycle is not doing distinct work.

**vs. ECC-Gen with LPIPS perceptual loss instead of L_cycle** — The LPIPS loss penalizes perceptual differences, which may correlate with token-space consistency but is not identical. ECC-Gen's token-space cycle is expected to yield a larger gain on the knowledge-intensive split, where the encoder's token space is the relevant metric. If the LPIPS ablation matches or exceeds ECC-Gen on cycle error, the token-space cycle is not distinct from perceptual regularization.

### What would falsify this idea

If ECC-Gen's cycle-error advantage over Integral is uniform across creative and knowledge-intensive subsets rather than concentrated where the predicted manifold-flatness failure occurs, the mechanism story is wrong. Likewise, if removing the manifold anchor (λ_a=0) does not produce a diversity collapse or a rare-prompt degradation relative to full ECC-Gen, the anchor is not load-bearing and the central claim reduces to a plain cycle loss. If the sensitivity gate s(z) does not correlate with the cycle-error gap, the local-injectivity assumption is not the operative factor.

## References

1. Let Vision-Language Models Generate Images through Their Representations
